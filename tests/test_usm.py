import struct

import pytest

from conftest import FakeReporter, chunk
from stages.usm import USM
from utils.errors import Cancelled, CharlotteError


class CancellingReporter(FakeReporter):
    """Reports a pending cancel so the next checkpoint raises."""

    def cancel_requested(self):
        return True


def make_usm(tmp_path, chunks: bytes) -> USM:
    usm_file = tmp_path / "Cs_Test.usm"
    usm_file.write_bytes(chunks)
    return USM(usm_file, bytes(4), bytes(4))


@pytest.fixture
def out_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    return out


def test_demux_extracts_streams(tmp_path, out_dir, reporter):
    # A video chunk this small is below the encryption threshold, so it must
    # pass through decrypt_video unchanged.
    data = (
        chunk(b"@SFV", b"video")
        + chunk(b"@SFA", b"audio0", channel=0)
        + chunk(b"@SFA", b"audio1", channel=1)
    )
    file_paths = make_usm(tmp_path, data).demux(out_dir, reporter)

    assert (out_dir / "Cs_Test.ivf").read_bytes() == b"video"
    assert (out_dir / "Cs_Test_0.hca").read_bytes() == b"audio0"
    assert (out_dir / "Cs_Test_1.hca").read_bytes() == b"audio1"
    assert [path.name for path in file_paths["ivf"]] == ["Cs_Test.ivf"]
    assert len(file_paths["hca"]) == 2


def test_metadata_chunks_skipped(tmp_path, out_dir, reporter):
    data = chunk(b"@SFV", b"meta", data_type=1) + chunk(b"@SFV", b"video")
    make_usm(tmp_path, data).demux(out_dir, reporter)
    assert (out_dir / "Cs_Test.ivf").read_bytes() == b"video"


def test_unknown_signature_warned_once(tmp_path, out_dir, reporter, caplog):
    data = chunk(b"@XXX", b"a") + chunk(b"@XXX", b"b") + chunk(b"@XXX", b"c")
    make_usm(tmp_path, data).demux(out_dir, reporter)
    warnings = [record for record in caplog.records if "Unknown signature" in record.message]
    assert len(warnings) == 1


def test_known_metadata_signatures_skipped_silently(tmp_path, out_dir, reporter, caplog):
    """CRID/@CUE/@APP and friends are recognized containers, not payloads: they are
    dropped with neither an output stream nor the unknown-signature warning."""
    data = (
        chunk(b"CRID", b"header")
        + chunk(b"@CUE", b"cue")
        + chunk(b"@APP", b"app")
        + chunk(b"@SFV", b"video")
    )
    file_paths = make_usm(tmp_path, data).demux(out_dir, reporter)

    assert [record for record in caplog.records if "Unknown signature" in record.message] == []
    assert set(file_paths) == {"ivf"}  # only the video became an output stream


def test_corrupt_chunk_raises(tmp_path, out_dir, reporter):
    bad = struct.pack(">4sIxBHB2xB16x", b"@SFA", 4, 0x18, 0, 0, 0)  # data_size < data_offset
    with pytest.raises(CharlotteError, match="Corrupt USM chunk"):
        make_usm(tmp_path, bad).demux(out_dir, reporter)


def test_undersized_data_offset_raises(tmp_path, out_dir, reporter):
    """A data_offset below 0x18 would seek back into the header just read, so the walk
    would creep through the file yielding overlapping garbage instead of stopping."""
    bad = struct.pack(">4sIxBHB2xB16x", b"@SFA", 0, 0, 0, 0, 0)  # data_offset 0 < 0x18
    with pytest.raises(CharlotteError, match="Corrupt USM chunk"):
        make_usm(tmp_path, bad).demux(out_dir, reporter)


def test_cancel_mid_demux_records_partial_output(tmp_path, out_dir):
    """On cancel, the caller-supplied dict still records what was written, so the
    caller can clean up (demux itself deletes nothing)."""
    # Enough chunks to pass the every-100-chunks checkpoint.
    data = b"".join(chunk(b"@SFA", b"x") for _ in range(150))
    file_paths = {}
    with pytest.raises(Cancelled):
        make_usm(tmp_path, data).demux(out_dir, CancellingReporter(), file_paths=file_paths)
    assert [path.name for path in file_paths["hca"]] == ["Cs_Test_0.hca"]
    assert (out_dir / "Cs_Test_0.hca").exists()
