import struct

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from utils.errors import CharlotteError
from utils.logger import log


if TYPE_CHECKING:
    from collections.abc import Generator
    from io import BufferedWriter

    from utils.reporter import Reporter


BLOCK = 0x20  # the mask is applied one block at a time
MASK_START = 0x40  # first byte of a video payload a mask ever touches
CIPHER_START = 0x140  # start of the chained-mask region within a video payload
MIN_MASKED = 0x200  # payloads with less than this past MASK_START are left in the clear
HEADER_SIZE = 32
MIN_DATA_OFFSET = 0x18  # data_offset counts from byte 8 of the header, so this is its floor


def is_masked(payload_size: int) -> bool:
    """Whether `decrypt_video` actually masks a video payload of this size."""
    return payload_size - MASK_START >= MIN_MASKED


class ChunkHeader(NamedTuple):
    signature: bytes
    data_size: int
    data_offset: int
    padding_size: int
    channel_no: int
    data_type: int

    @classmethod
    def from_bytes(cls, raw: bytes) -> ChunkHeader:
        return cls._make(struct.unpack(">4s I x B H B 2x B 16x", raw))


def read_chunks(file_path: Path) -> Generator[tuple[ChunkHeader, bytes]]:
    """Walk a USM file, yielding each chunk header with its payload.

    Sole owner of the on-disk chunk layout, so demuxing and key recovery cannot
    drift apart in how they read it. Typed as a Generator rather than an Iterator
    because a caller that stops early closes it to drop the file handle.
    """
    with open(file_path, "rb") as fp:
        while True:
            raw = fp.read(HEADER_SIZE)
            if len(raw) < HEADER_SIZE:
                return

            header = ChunkHeader.from_bytes(raw)
            payload_size = header.data_size - header.data_offset - header.padding_size
            # An undersized data_offset seeks back into the header just read, leaving the
            # walk to creep forward a few bytes at a time re-parsing its own garbage.
            if payload_size < 0 or header.data_offset < MIN_DATA_OFFSET:
                raise CharlotteError(f"Corrupt USM chunk in {file_path.name}")

            fp.seek(header.data_offset - MIN_DATA_OFFSET, 1)
            payload = fp.read(payload_size)
            fp.seek(header.padding_size, 1)
            yield header, payload


class USM:
    def __init__(self, file_path: Path, key1: bytes, key2: bytes):
        self.file_path = Path(file_path)
        self.video_mask1 = self.build_mask(key1, key2)
        self.video_mask2 = bytes(b ^ 0xFF for b in self.video_mask1)

    @staticmethod
    def build_mask(key1: bytes, key2: bytes) -> bytes:
        m = bytearray(0x20)

        m[0x00] = key1[0]
        m[0x01] = key1[1]
        m[0x02] = key1[2]
        m[0x03] = (key1[3] - 0x34) & 0xFF
        m[0x04] = (key2[0] + 0xF9) & 0xFF
        m[0x05] = (key2[1] ^ 0x13) & 0xFF
        m[0x06] = (key2[2] + 0x61) & 0xFF
        m[0x07] = (m[0x00] ^ 0xFF) & 0xFF
        m[0x08] = (m[0x02] + m[0x01]) & 0xFF
        m[0x09] = (m[0x01] - m[0x07]) & 0xFF
        m[0x0A] = (m[0x02] ^ 0xFF) & 0xFF
        m[0x0B] = (m[0x01] ^ 0xFF) & 0xFF
        m[0x0C] = (m[0x0B] + m[0x09]) & 0xFF
        m[0x0D] = (m[0x08] - m[0x03]) & 0xFF
        m[0x0E] = (m[0x0D] ^ 0xFF) & 0xFF
        m[0x0F] = (m[0x0A] - m[0x0B]) & 0xFF
        m[0x10] = (m[0x08] - m[0x0F]) & 0xFF
        m[0x11] = (m[0x10] ^ m[0x07]) & 0xFF
        m[0x12] = (m[0x0F] ^ 0xFF) & 0xFF
        m[0x13] = (m[0x03] ^ 0x10) & 0xFF
        m[0x14] = (m[0x04] - 0x32) & 0xFF
        m[0x15] = (m[0x05] + 0xED) & 0xFF
        m[0x16] = (m[0x06] ^ 0xF3) & 0xFF
        m[0x17] = (m[0x13] - m[0x0F]) & 0xFF
        m[0x18] = (m[0x15] + m[0x07]) & 0xFF
        m[0x19] = (0x21 - m[0x13]) & 0xFF
        m[0x1A] = (m[0x14] ^ m[0x17]) & 0xFF
        m[0x1B] = (m[0x16] + m[0x16]) & 0xFF
        m[0x1C] = (m[0x17] + 0x44) & 0xFF
        m[0x1D] = (m[0x03] + m[0x04]) & 0xFF
        m[0x1E] = (m[0x05] - m[0x16]) & 0xFF
        m[0x1F] = (m[0x1D] ^ m[0x13]) & 0xFF

        return bytes(m)

    def decrypt_video(self, data: bytearray) -> None:
        if not is_masked(len(data)):
            return

        mask2 = int.from_bytes(self.video_mask2)
        end = len(data)

        m = mask2
        pos = CIPHER_START
        while pos + BLOCK <= end:
            dec = int.from_bytes(data[pos : pos + BLOCK]) ^ m
            data[pos : pos + BLOCK] = dec.to_bytes(BLOCK)
            m = dec ^ mask2
            pos += BLOCK
        if pos < end:
            tail = end - pos
            dec = int.from_bytes(data[pos:end]) ^ (m >> (8 * (BLOCK - tail)))
            data[pos:end] = dec.to_bytes(tail)

        # The head is masked last, folding in the block 0x100 bytes further along.
        m = int.from_bytes(self.video_mask1)
        for pos in range(MASK_START, CIPHER_START, BLOCK):
            m ^= int.from_bytes(data[pos + 0x100 : pos + 0x100 + BLOCK])
            dec = int.from_bytes(data[pos : pos + BLOCK]) ^ m
            data[pos : pos + BLOCK] = dec.to_bytes(BLOCK)

    def demux(
        self,
        output_path: Path,
        reporter: Reporter,
        file_paths: dict[str, list[Path]] | None = None,
    ) -> dict[str, list[Path]]:
        base_name = self.file_path.stem
        streams: dict[Path, BufferedWriter] = {}
        if file_paths is None:
            file_paths = {}
        known = {b"CRID", b"@SFV", b"@SFA", b"@CUE", b"@APP", b"@ALP", b"@SBT"}
        file_size = self.file_path.stat().st_size

        with (
            reporter.task("demux", total=file_size, unit="B") as task,
            ExitStack() as open_streams,
        ):

            def write_to(filename: str, kind: str, payload: bytes) -> None:
                path = output_path / filename
                if path not in streams:
                    streams[path] = open_streams.enter_context(open(path, "wb"))
                    file_paths.setdefault(kind, []).append(path)
                streams[path].write(payload)

            for chunks, (header, data) in enumerate(read_chunks(self.file_path), start=1):
                payload_type = header.data_type & 0x3
                if header.signature == b"@SFV" and payload_type == 0:
                    buffer = bytearray(data)
                    self.decrypt_video(buffer)
                    write_to(f"{base_name}.ivf", "ivf", buffer)
                elif header.signature == b"@SFA" and payload_type == 0:
                    write_to(f"{base_name}_{header.channel_no}.hca", "hca", data)
                elif header.signature not in known:
                    known.add(header.signature)  # warn once per signature
                    log.warning(f"Unknown signature {header.signature!r}")

                task.advance(header.data_size + 8)
                if chunks % 100 == 0:
                    reporter.checkpoint()

            task.set_completed(file_size)

        return file_paths
