import random

import numpy as np
import pytest

from conftest import chunk
from stages.crack import (
    STAGES,
    Recovery,
    Sample,
    Stats,
    crack_key,
    evaluate,
    expand_0,
    expand_1_2,
    expand_3,
    expand_4,
    expand_5,
    expand_6,
    solve,
    split_key,
)
from stages.usm import BLOCK, CIPHER_START, MASK_START, MIN_MASKED, USM, is_masked


KEY1, KEY2 = bytes([0x11, 0x22, 0x33, 0x44]), bytes([0x55, 0x66, 0x77, 0x00])

# The value each stage feeds its expansion, read back out of a known-good mask.
SEEDS = {
    expand_1_2: lambda m: (m[0x01] << 8) | m[0x02],
    expand_0: lambda m: m[0x00],
    expand_3: lambda m: m[0x03],
    expand_4: lambda m: m[0x04],
    expand_5: lambda m: m[0x05],
    expand_6: lambda m: m[0x06],
}


def key_pairs(count: int) -> list[tuple[bytes, bytes]]:
    rng = random.Random(0)
    # Byte 3 of key2 is never read by build_mask, so it is always recovered as 0.
    return [
        (
            bytes(rng.randrange(256) for _ in range(4)),
            bytes(rng.randrange(256) for _ in range(3)) + b"\x00",
        )
        for _ in range(count)
    ]


def video_plaintext(rng, blocks: int) -> bytes:
    """Compressed VP9 is close to uniform, but 00,00 and FF,FF byte pairs run above
    chance - the only signal the solver has to work with."""
    data = bytearray(rng.randrange(256) for _ in range(blocks * BLOCK))
    for _ in range(len(data) // 40):
        i = rng.randrange(len(data) - 1)
        data[i] = data[i + 1] = 0x00 if rng.random() < 0.7 else 0xFF
    return bytes(data)


def encrypt(plain: bytes, key1: bytes, key2: bytes) -> bytes:
    """Inverse of the chained region of USM.decrypt_video: the running mask starts at
    video_mask2 and is reset to `plaintext ^ video_mask2` after every block."""
    mask2 = bytes(b ^ 0xFF for b in USM.build_mask(key1, key2))
    payload = bytearray(CIPHER_START)  # the head is masked separately, so leave it blank
    m = mask2
    for i in range(0, len(plain), BLOCK):
        block = plain[i : i + BLOCK]
        payload += bytes(b ^ k for b, k in zip(block, m, strict=True))
        m = bytes(b ^ k for b, k in zip(block, mask2, strict=True))
    return bytes(payload)


def video_chunk(rng, blocks: int = 900) -> bytes:
    """One @SFV chunk carrying an encrypted IVF payload."""
    return chunk(b"@SFV", b"DKIF" + encrypt(video_plaintext(rng, blocks), KEY1, KEY2)[4:])


def test_stages_cover_every_mask_entry_exactly_once():
    """A stage that forgets an entry silently drops it from scoring."""
    covered = [entry for stage in STAGES for entry in stage.entries]
    assert sorted(covered) == list(range(BLOCK))


def test_pair_schedule_covers_every_adjacent_pair_exactly_once():
    scored = [pair for stage in STAGES for pair in stage.pairs]
    assert sorted(scored) == list(range(BLOCK - 1))


@pytest.mark.parametrize(("key1", "key2"), key_pairs(12))
def test_expansions_reproduce_build_mask(key1, key2):
    """The expansions mirror USM.build_mask by hand; this pins them to it."""
    truth = USM.build_mask(key1, key2)

    mask = np.zeros((BLOCK, 1), dtype=np.int32)
    for stage in STAGES:
        stage.expand(mask, np.array([SEEDS[stage.expand](truth)], dtype=np.int32))

    assert bytes(mask[:, 0].tolist()) == truth


@pytest.mark.parametrize(("key1", "key2"), key_pairs(12))
def test_split_key_inverts_build_mask(key1, key2):
    assert split_key(list(USM.build_mask(key1, key2))) == (key1, key2)


def test_solve_recovers_a_known_key():
    """The module's core claim: ciphertext in, the mask that produced it back out."""
    rng = random.Random(7)
    stats = Stats()
    for _ in range(8):
        stats.add(encrypt(video_plaintext(rng, 500), KEY1, KEY2))

    assert split_key(solve(*stats.tables())) == (KEY1, KEY2)


def test_crack_key_recovers_from_a_usm(tmp_path, reporter):
    """End to end over a real chunk layout: walk the file, pool both halves, agree."""
    rng = random.Random(3)
    usm_file = tmp_path / "Cs_Test.usm"
    usm_file.write_bytes(b"".join(video_chunk(rng) for _ in range(8)))

    assert crack_key(usm_file, reporter) == Recovery((KEY1, KEY2), "")


def test_crack_key_declines_a_file_with_no_video(tmp_path, reporter):
    usm_file = tmp_path / "Cs_Test.usm"
    usm_file.write_bytes(chunk(b"@SFA", b"audio"))

    recovery = crack_key(usm_file, reporter)

    assert recovery.key is None
    assert "no IVF video stream" in recovery.reason


def test_crack_key_declines_too_little_video(tmp_path, reporter):
    """A short cutscene cannot outvote the noise, so the sample floor rejects it
    before the two halves are ever compared."""
    payload = b"DKIF" + bytes(MASK_START + MIN_MASKED)
    usm_file = tmp_path / "Cs_Test.usm"
    usm_file.write_bytes(b"".join(chunk(b"@SFV", payload) for _ in range(4)))

    recovery = crack_key(usm_file, reporter)

    assert recovery.key is None
    assert "bytes of encrypted video" in recovery.reason


def test_crack_key_declines_a_repeated_payload(tmp_path, reporter):
    """Enough video to clear the sample floor, but it is one frame over and over, so
    the halves are a single piece of evidence counted twice."""
    payload = b"DKIF" + bytes(CIPHER_START + 1600 * BLOCK - 4)
    usm_file = tmp_path / "Cs_Test.usm"
    usm_file.write_bytes(b"".join(chunk(b"@SFV", payload) for _ in range(2)))

    recovery = crack_key(usm_file, reporter)

    assert recovery.key is None
    assert "repeats the same payload" in recovery.reason


def test_identical_payloads_are_not_independent_evidence():
    """Split-half only proves anything if the two pools saw different bytes.

    A placeholder asset - zero-filled, or one frame repeated - makes both pools
    identical, so they agree on an arbitrary mask. Stats.contents is what catches it.
    """
    payload = bytes(CIPHER_START + BLOCK * 400)
    left, right = Stats(), Stats()
    for i in range(20):
        (left if i % 2 == 0 else right).add(payload)

    assert solve(*left.tables()) == solve(*right.tables())  # they do agree
    assert left.contents == right.contents  # but on the same evidence twice


def test_differing_payloads_clear_the_content_guard():
    """The mirror of the case above, and the only branch where evaluate() accepts.

    Pools that saw different bytes are genuinely independent, so the guard stands
    aside and the two solves get to vouch for each other.
    """
    rng = random.Random(5)
    left, right = Stats(), Stats()
    for i in range(8):
        (left if i % 2 == 0 else right).add(encrypt(video_plaintext(rng, 900), KEY1, KEY2))

    mask, reason = evaluate(Sample(left, right, b"DKIF", (left.blocks + right.blocks) * BLOCK))

    assert reason == ""
    assert split_key(mask) == (KEY1, KEY2)


def test_split_half_rejects_disagreeing_noise():
    """The core safety property: two halves that don't agree yield no key at all.

    Pure noise carries no 00,00/FF,FF signal, so each half chases its own tail and
    solves a different mask. The pools hold different bytes, so this clears the
    content guard and lands squarely on the split-half disagreement it is meant to
    catch - the branch that keeps a thin sample from ever accepting a wrong key."""
    rng = random.Random(11)
    left, right = Stats(), Stats()
    for i in range(30):
        body = bytes(rng.randrange(256) for _ in range(BLOCK * 60))
        (left if i % 2 == 0 else right).add(bytes(CIPHER_START) + body)

    assert left.contents != right.contents  # not the identical-pools shortcut
    used = (left.blocks + right.blocks) * BLOCK
    mask, reason = evaluate(Sample(left, right, b"DKIF", used))

    assert mask is None
    assert "disagree" in reason


def test_is_masked_matches_decrypt_video(tmp_path):
    """The sampler skips payloads decrypt_video leaves alone; the two must agree."""
    usm = USM(tmp_path / "Cs_Test.usm", bytes([1, 2, 3, 4]), bytes([5, 6, 7, 0]))
    threshold = MASK_START + MIN_MASKED

    for size in (threshold - 1, threshold, threshold + BLOCK):
        data = bytearray(size)
        usm.decrypt_video(data)
        changed = any(data)
        assert changed == is_masked(size), f"size {size:#x}"
