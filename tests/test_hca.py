"""Tests for the parts of stages/hca.py that are Charlotte's own: how a malformed file
is rejected, what -nc leaves on disk, and how the stream is handed to ffmpeg.

The decryption itself is deliberately not tested. It is a line-for-line port of the C#
implementation, and any test written here would have to encrypt with the same tables the
code decrypts with - which passes whether or not the port matches the original. Real
test vectors would have to come from the C# side; the corpus is the check that matters.
"""

import struct

import pytest

from conftest import flag_value
from stages.hca import HCA
from utils.errors import CharlotteError
from utils.ffmpeg import FFMPEG_MISSING


KEY1, KEY2 = bytes([0x11, 0x22, 0x33, 0x44]), bytes([0x55, 0x66, 0x77, 0x00])

# How far read_header advances past each chunk signature. A chunk carries no length of
# its own, so the walk is these constants - which is why a file has to be built out of
# them to be parseable at all.
CHUNK_SIZES = {b"fmt\x00": 16, b"comp": 16, b"ciph": 6}


def hca_bytes(
    *,
    ciph_type: int = 0,
    block_size: int = 0x20,
    block_count: int = 2,
    data: bytes | None = None,
) -> bytes:
    """A minimal but structurally real .hca: the chunk sequence read_header walks, then
    block_count blocks of block_size bytes, zeroed unless `data` says otherwise."""
    chunks: list[tuple[bytes, dict]] = [
        (b"fmt\x00", {8: (">I", block_count)}),
        (b"comp", {4: (">H", block_size)}),
        (b"ciph", {4: (">H", ciph_type)}),
    ]

    header = bytearray(8)
    header[0:4] = b"HCA\x00"
    struct.pack_into(">H", header, 4, 0x0200)  # version
    for tag, fields in chunks:
        base = len(header)
        header += bytearray(CHUNK_SIZES[tag])
        header[base : base + 4] = tag
        for offset, (fmt, value) in fields.items():
            struct.pack_into(fmt, header, base + offset, value)

    header += bytearray(2)  # the trailing checksum read_header recomputes
    struct.pack_into(">H", header, 6, len(header))  # data_offset

    body = bytes(block_size * block_count) if data is None else data
    return bytes(header) + bytes(body)


def write_hca(tmp_path, blob: bytes, name: str = "Cs_Test_0.hca"):
    path = tmp_path / name
    path.write_bytes(blob)
    return path


def make_hca(tmp_path, name: str = "Cs_Test_0.hca") -> HCA:
    return HCA(write_hca(tmp_path, hca_bytes(), name), KEY1, KEY2)


# --- header walk ---


def test_header_fields_parsed(tmp_path):
    """The positive control for the rejection tests below: without it a builder that
    produced nonsense would still make all of them pass, just for the wrong reason."""
    path = write_hca(tmp_path, hca_bytes(ciph_type=0x38, block_size=0x40, block_count=3))
    hca = HCA(path, KEY1, KEY2)

    assert hca.block_count == 3
    assert hca.block_size == 0x40
    assert hca.ciph_type == 0x38
    assert len(hca.data) == 0x40 * 3  # bounded by the declared blocks, not end of file


# --- header rejections ---


def test_file_too_short_raises(tmp_path):
    path = write_hca(tmp_path, b"HCA\x00")
    with pytest.raises(CharlotteError, match="Invalid HCA file"):
        HCA(path, KEY1, KEY2)


def test_bad_magic_raises(tmp_path):
    path = write_hca(tmp_path, hca_bytes().replace(b"HCA\x00", b"XXXX", 1))
    with pytest.raises(CharlotteError, match="Invalid HCA header"):
        HCA(path, KEY1, KEY2)


def test_missing_fmt_chunk_raises(tmp_path):
    path = write_hca(tmp_path, hca_bytes().replace(b"fmt\x00", b"junk", 1))
    with pytest.raises(CharlotteError, match="fmt chunk not found"):
        HCA(path, KEY1, KEY2)


def test_missing_compression_chunk_raises(tmp_path):
    path = write_hca(tmp_path, hca_bytes().replace(b"comp", b"junk", 1))
    with pytest.raises(CharlotteError, match="comp/dec chunk not found"):
        HCA(path, KEY1, KEY2)


def test_zero_block_size_raises(tmp_path):
    """The C# reference allows it, but block_size is the stride the checksum walk steps
    by, so leaving it unchecked turns a header-only file into a bare ValueError - which
    escapes the per-file handler and takes the whole batch with it."""
    path = write_hca(tmp_path, hca_bytes(block_size=0, block_count=0))
    with pytest.raises(CharlotteError, match="no audio blocks"):
        HCA(path, KEY1, KEY2)


def test_unknown_cipher_type_raises(tmp_path):
    """Only 0, 1 and 0x38 have a table. Anything else would silently build an all-zero
    one and translate the whole stream to zeros."""
    path = write_hca(tmp_path, hca_bytes(ciph_type=2))
    with pytest.raises(CharlotteError, match="Invalid cipher type"):
        HCA(path, KEY1, KEY2)


def test_truncated_header_raises(tmp_path):
    """data_offset claims a longer header than the file holds, so a field read runs off
    the end. struct.error is translated rather than escaping as a bare traceback."""
    path = write_hca(tmp_path, hca_bytes()[:12])
    with pytest.raises(CharlotteError, match="Corrupt HCA header"):
        HCA(path, KEY1, KEY2)


# --- save ---


def test_save_overwrites_the_source_with_the_in_memory_stream(tmp_path):
    """-nc keeps the intermediate .hca, and what lands there is the decrypted stream:
    header and data as they now stand, not the bytes that were read in."""
    # A non-zero payload on purpose: zeros pass through the cipher table untouched, so
    # a zeroed body would let this pass on the header edit alone.
    body = bytes(range(0x20)) * 4
    original = hca_bytes(ciph_type=0x38, block_count=4, data=body)
    path = write_hca(tmp_path, original)
    hca = HCA(path, KEY1, KEY2)
    hca.decrypt()

    hca.save()

    written = path.read_bytes()
    assert written == bytes(hca.header) + bytes(hca.data)
    assert written[len(hca.header) :] != body  # the audio really was rewritten
    assert HCA(path, KEY1, KEY2).ciph_type == 0  # and it reads back as a plain HCA


# --- convert ---


def test_convert_pipes_the_stream_to_ffmpeg(ffmpeg, tmp_path):
    """The stream is fed on stdin so the decrypted audio never round trips through disk."""
    hca = make_hca(tmp_path)
    output = hca.convert(output_path=tmp_path, codec="flac")

    assert output == tmp_path / "Cs_Test_0.flac"
    assert flag_value(ffmpeg.cmd, "-f") == "hca"
    assert flag_value(ffmpeg.cmd, "-i") == "pipe:0"
    assert ffmpeg.cmd[-1] == str(output)
    assert ffmpeg.input == bytes(hca.header) + bytes(hca.data)


def test_convert_extension_and_args_follow_the_codec(ffmpeg, tmp_path):
    output = make_hca(tmp_path).convert(output_path=tmp_path, codec="opus")

    assert output == tmp_path / "Cs_Test_0.mka"
    assert flag_value(ffmpeg.cmd, "-c:a") == "libopus"


def test_convert_unknown_codec_falls_back_to_flac(ffmpeg, tmp_path):
    output = make_hca(tmp_path).convert(output_path=tmp_path, codec="nonsense")

    assert output == tmp_path / "Cs_Test_0.flac"
    assert flag_value(ffmpeg.cmd, "-compression_level") == "8"


def test_convert_reports_ffmpeg_failure(ffmpeg, tmp_path, caplog):
    ffmpeg.returncode = 1
    ffmpeg.stderr = b"Invalid data found when processing input"

    with pytest.raises(CharlotteError, match="Audio conversion failed"):
        make_hca(tmp_path).convert(output_path=tmp_path)

    # ffmpeg's own diagnosis is surfaced, not swallowed behind the exit code.
    assert "Invalid data found" in caplog.text


def test_convert_without_ffmpeg_raises(ffmpeg, tmp_path):
    """A missing bundled binary is the one failure that tells the user what to do about
    it, so the message is pinned rather than pattern-matched."""
    ffmpeg.missing = True
    with pytest.raises(CharlotteError) as excinfo:
        make_hca(tmp_path).convert(output_path=tmp_path)
    assert str(excinfo.value) == FFMPEG_MISSING
