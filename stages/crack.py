"""Recover a USM decryption key from the file's own video stream.

A video payload is decrypted with a chained mask: each 0x20 block is XORed with a
running mask that then resets to `plaintext ^ video_mask2`. Against `s[i]`, the
running XOR of ciphertext blocks 0..i, the chain expands to:

    even i -> plaintext[i] = s[i] ^ video_mask2
    odd i  -> plaintext[i] = s[i]

So odd blocks need no key, and even blocks are a repeating-key XOR against
video_mask2. Compressed VP9 is near-uniform, but `00 00` byte pairs still run above
chance - enough to rank mask candidates once hundreds of thousands of blocks pool.

video_mask1 has 32 entries but only 7 free bytes (see `USM.build_mask`) that unlock
disjoint groups, so a beam search fixes one key byte per stage rather than 2^56.

The result is confirmed by solving two independent pools separately and requiring
all 56 bits to agree: a thin sample makes each half chase its own noise, and that is
the case to reject. Structural checks can't substitute - every payload's IVF frame
keeps its length prefix in the unmasked first 0x40 bytes, so the chain parses under
any key.
"""

from contextlib import closing
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from stages.usm import BLOCK, CIPHER_START, is_masked, read_chunks
from utils.logger import log


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from utils.reporter import Reporter


SAMPLE_STEPS = (10_000_000, 30_000_000)  # combined budgets; each pool sees half of one
MIN_SAMPLE_BYTES = 100_000  # floor for attempting a solve at all
BEAM = 50  # candidates carried from one stage to the next
WIDE_BEAM = 300  # the 16-bit stage scores too few entries to trust a narrow beam
BIGRAM_WEIGHT = 25  # total weight shared between the 00,00 and FF,FF terms
BIGRAM_MIN_HITS = 100  # plaintext pairs needed before measuring how to share it
BIGRAM_RATIO = (1.0, 5.0)  # clamp, so one lopsided file cannot zero out either term
BIGRAM_FALLBACK = (10, 4)  # zero/ff weights when there are too few hits to measure


# These mirror `USM.build_mask` line for line, but vectorized: `m` is a (32, n) array
# of mask candidates and `value` the (n,) column of key bytes tried. A mask entry is
# the row index, so `m[0x08] = (m[0x02] + m[0x01]) & 0xFF` reads like the scalar
# original while updating every candidate. Any change to `build_mask` lands here too.
def expand_0(m: np.ndarray, value: np.ndarray) -> None:
    m[0x00] = value
    m[0x07] = m[0x00] ^ 0xFF
    m[0x09] = (m[0x01] - m[0x07]) & 0xFF
    m[0x0C] = (m[0x0B] + m[0x09]) & 0xFF
    m[0x11] = m[0x10] ^ m[0x07]


def expand_1_2(m: np.ndarray, value: np.ndarray) -> None:
    m[0x01] = value >> 8
    m[0x02] = value & 0xFF
    m[0x08] = (m[0x02] + m[0x01]) & 0xFF
    m[0x0A] = m[0x02] ^ 0xFF
    m[0x0B] = m[0x01] ^ 0xFF
    m[0x0F] = (m[0x0A] - m[0x0B]) & 0xFF
    m[0x10] = (m[0x08] - m[0x0F]) & 0xFF
    m[0x12] = m[0x0F] ^ 0xFF


def expand_3(m: np.ndarray, value: np.ndarray) -> None:
    m[0x03] = value
    m[0x0D] = (m[0x08] - m[0x03]) & 0xFF
    m[0x0E] = m[0x0D] ^ 0xFF
    m[0x13] = m[0x03] ^ 0x10
    m[0x17] = (m[0x13] - m[0x0F]) & 0xFF
    m[0x19] = (0x21 - m[0x13]) & 0xFF
    m[0x1C] = (m[0x17] + 0x44) & 0xFF


def expand_4(m: np.ndarray, value: np.ndarray) -> None:
    m[0x04] = value
    m[0x14] = (m[0x04] - 0x32) & 0xFF
    m[0x1A] = m[0x14] ^ m[0x17]
    m[0x1D] = (m[0x03] + m[0x04]) & 0xFF
    m[0x1F] = m[0x1D] ^ m[0x13]


def expand_5(m: np.ndarray, value: np.ndarray) -> None:
    m[0x05] = value
    m[0x15] = (m[0x05] + 0xED) & 0xFF
    m[0x18] = (m[0x15] + m[0x07]) & 0xFF
    m[0x1E] = (m[0x05] - m[0x16]) & 0xFF


def expand_6(m: np.ndarray, value: np.ndarray) -> None:
    m[0x06] = value
    m[0x16] = m[0x06] ^ 0xF3
    m[0x1B] = (m[0x16] + m[0x16]) & 0xFF


class Stage(NamedTuple):
    """One beam-search step: try `span` values for a key byte and score what it unlocks."""

    expand: Callable[[np.ndarray, np.ndarray], None]
    span: int
    beam: int  # candidates this step keeps for the next one
    entries: tuple[int, ...]  # mask entries this step determines
    pairs: tuple[int, ...]  # adjacent entry pairs that first become scorable here


def plan_stages() -> list[Stage]:
    """Sequence the expansions and work out when each adjacent pair becomes scorable.

    Dependency order, not numeric order: entry 6 has to land before entry 5 because
    `expand_5` reads m[0x16].
    """
    order = [
        (expand_1_2, 0x10000, WIDE_BEAM, (0x01, 0x02, 0x08, 0x0A, 0x0B, 0x0F, 0x10, 0x12)),
        (expand_0, 0x100, BEAM, (0x00, 0x07, 0x09, 0x0C, 0x11)),
        (expand_3, 0x100, BEAM, (0x03, 0x0D, 0x0E, 0x13, 0x17, 0x19, 0x1C)),
        (expand_4, 0x100, BEAM, (0x04, 0x14, 0x1A, 0x1D, 0x1F)),
        (expand_6, 0x100, BEAM, (0x06, 0x16, 0x1B)),
        (expand_5, 0x100, BEAM, (0x05, 0x15, 0x18, 0x1E)),
    ]

    known: set[int] = set()
    scored: set[int] = set()
    stages = []
    for expand, span, beam, entries in order:
        known.update(entries)
        pairs = tuple(j for j in range(BLOCK - 1) if j not in scored and {j, j + 1} <= known)
        scored.update(pairs)
        stages.append(Stage(expand, span, beam, entries, pairs))
    return stages


STAGES = plan_stages()


class Stats:
    """Per-entry byte and adjacent-pair counts pooled over even blocks."""

    def __init__(self):
        self.unigram = np.zeros((BLOCK, 256), dtype=np.int64)
        self.bigram = np.zeros((BLOCK - 1, 65536), dtype=np.int64)
        self.zero_pairs = 0
        self.ff_pairs = 0
        self.blocks = 0
        self.contents: set[int] = set()  # digests of the payloads folded in

    def add(self, payload: bytes) -> int:
        """Fold one video payload in. Returns the cipher bytes consumed."""
        rows = (len(payload) - CIPHER_START) // BLOCK
        if rows < 2:
            return 0

        self.contents.add(hash(payload))

        body = np.frombuffer(payload, dtype=np.uint8, count=rows * BLOCK, offset=CIPHER_START)
        running = np.bitwise_xor.accumulate(body.reshape(rows, BLOCK), axis=0)
        even = running[0::2]  # plaintext ^ video_mask2
        odd = running[1::2]  # plaintext, free of the key

        for j in range(BLOCK):
            self.unigram[j] += np.bincount(even[:, j], minlength=256)

        pairs = (even[:, :-1].astype(np.int32) << 8) | even[:, 1:]
        for j in range(BLOCK - 1):
            self.bigram[j] += np.bincount(pairs[:, j], minlength=65536)

        # Odd blocks are plaintext, so they measure this file's 00,00 vs FF,FF split.
        left, right = odd[:, :-1], odd[:, 1:]
        self.zero_pairs += int(np.count_nonzero((left == 0) & (right == 0)))
        self.ff_pairs += int(np.count_nonzero((left == 0xFF) & (right == 0xFF)))

        self.blocks += rows
        return rows * BLOCK

    def tables(self) -> tuple[np.ndarray, np.ndarray]:
        """Score tables folding each candidate together with its complement.

        A mask entry v is evidenced both by plaintext FF (running XOR lands on v)
        and by plaintext 00 (it lands on v ^ FF), so both are summed up front.
        """
        byte_flip = np.arange(256) ^ 0xFF
        pair_flip = np.arange(65536) ^ 0xFFFF
        unigram = self.unigram + self.unigram[:, byte_flip]

        if self.zero_pairs + self.ff_pairs < BIGRAM_MIN_HITS:
            zero_weight, ff_weight = BIGRAM_FALLBACK
        else:
            low, high = BIGRAM_RATIO
            ratio = min(max(self.zero_pairs / max(self.ff_pairs, 1), low), high)
            zero_weight = round(BIGRAM_WEIGHT * ratio / (1.0 + ratio))
            ff_weight = BIGRAM_WEIGHT - zero_weight

        bigram = ff_weight * self.bigram + zero_weight * self.bigram[:, pair_flip]
        return unigram, bigram


def solve(unigram: np.ndarray, bigram: np.ndarray) -> list[int]:
    """Beam search over the 7 free key bytes. Returns the best video_mask1."""
    masks = np.zeros((BLOCK, 1), dtype=np.int32)
    scores = np.zeros(1, dtype=np.int64)

    for stage in STAGES:
        candidates = np.repeat(masks, stage.span, axis=1)
        values = np.tile(np.arange(stage.span, dtype=np.int32), masks.shape[1])
        stage.expand(candidates, values)

        total = np.repeat(scores, stage.span)
        for j in stage.entries:
            total += unigram[j][candidates[j]]
        for j in stage.pairs:
            total += bigram[j][(candidates[j] << 8) | candidates[j + 1]]

        width = min(stage.beam, total.size)
        keep = np.argpartition(-total, width - 1)[:width]
        masks, scores = candidates[:, keep], total[keep]

    return masks[:, int(np.argmax(scores))].tolist()


def split_key(mask: list[int]) -> tuple[bytes, bytes]:
    """Invert `USM.build_mask`. Entry 7 of the key is never read by the mask."""
    key1 = bytes([mask[0x00], mask[0x01], mask[0x02], (mask[0x03] + 0x34) & 0xFF])
    key2 = bytes([(mask[0x04] - 0xF9) & 0xFF, mask[0x05] ^ 0x13, (mask[0x06] - 0x61) & 0xFF, 0])
    return key1, key2


class Sample(NamedTuple):
    """One pass over the file, pooled into two independent halves."""

    left: Stats
    right: Stats
    first_payload: bytes | None  # None when the file carries no video at all
    used: int


class Recovery(NamedTuple):
    """Outcome of a recovery attempt. `reason` is empty when `key` is set."""

    key: tuple[bytes, bytes] | None
    reason: str


def decline(usm_file: Path, reason: str) -> Recovery:
    log.warning(f"Could not recover a key for {usm_file.name}: {reason}.")
    return Recovery(None, reason)


def collect(usm_file: Path, reporter: Reporter, budget: int) -> Sample:
    """Pool payload statistics into two independent halves, alternating between them."""
    pools = (Stats(), Stats())
    first: bytes | None = None
    pool = 0  # toggled 0/1 to deal masked payloads alternately into the two halves
    used = 0
    file_size = usm_file.stat().st_size

    # closing() drops the .usm handle when the budget breaks the walk early; a
    # suspended generator would otherwise hold it open until garbage collection.
    with (
        reporter.task("crack", total=file_size, unit="B") as task,
        closing(read_chunks(usm_file)) as chunks,
    ):
        for count, (header, payload) in enumerate(chunks, start=1):
            task.advance(header.data_size + 8)
            if count % 100 == 0:
                reporter.checkpoint()

            if header.signature != b"@SFV" or header.data_type & 0x3 != 0:
                continue
            if first is None:
                first = payload
            if is_masked(len(payload)):
                used += pools[pool].add(payload)
                pool ^= 1
            if used >= budget:
                break

        task.set_completed(file_size)

    return Sample(pools[0], pools[1], first, used)


def evaluate(sample: Sample) -> tuple[list[int] | None, str]:
    """Solve both halves, or say why they cannot vouch for a key between them."""
    if sample.left.contents == sample.right.contents:
        # Byte-identical pools are one piece of evidence counted twice: they agree on
        # whatever mask the content favours, right or not. Placeholder assets land here.
        return None, "the video repeats the same payload, so the halves prove nothing"

    left = solve(*sample.left.tables())
    right = solve(*sample.right.tables())
    if left != right:
        return None, f"independent halves of the video disagree ({sample.used} bytes sampled)"
    return left, ""


def crack_key(usm_file: Path, reporter: Reporter) -> Recovery:
    """Recover (key1, key2) from the video stream, or explain why it cannot be trusted.

    Escalates through SAMPLE_STEPS: most files converge on the first pass. One that
    doesn't is either short on video (reading further won't help) or statistically
    weak (more video can fix it). A retry re-walks from the start, restarting its bar.
    """
    log.info(f"Recovering decryption key from {usm_file.name}...")
    reason = ""

    for budget in SAMPLE_STEPS:
        sample = collect(usm_file, reporter, budget)
        first = sample.first_payload
        if first is None or first[:4] != b"DKIF":
            return decline(usm_file, "no IVF video stream in this file")
        if sample.used < MIN_SAMPLE_BYTES:
            return decline(usm_file, f"only {sample.used} bytes of encrypted video")

        mask, reason = evaluate(sample)
        if mask is not None:
            blocks = sample.left.blocks + sample.right.blocks
            log.info(f"Recovered decryption key from {blocks} blocks.")
            return Recovery(split_key(mask), "")

        if sample.used < budget:
            break  # the whole file is sampled already, so a larger budget adds nothing

        log.info(f"{sample.used} bytes were inconclusive, retrying with more video...")

    return decline(usm_file, f"inconclusive - {reason}")
