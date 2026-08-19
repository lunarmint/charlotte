from types import SimpleNamespace

import pytest

import pipeline
import resources.keys

from conftest import FakeReporter, chunk
from pipeline import Options, crack_all, crack_usm, probe_usm, process_usm
from resources.keys import Keys, calculate_key_from_filename
from resources.subtitles import local_subtitle_path
from stages.crack import Recovery
from utils.errors import Cancelled, CharlotteError


KEYS_DATA = {"list": [{"videoKey": 111, "videos": ["Cs_A"]}]}


def write_subtitle(stem, lang):
    path = local_subtitle_path(stem, lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")


def last_probe_event(reporter):
    kind, data = reporter.events[-1]
    assert kind == "probe"
    return data


def test_probe_reports_available(tmp_app_root, reporter, monkeypatch):
    monkeypatch.setattr(pipeline, "find_vs_script", lambda stem: "Cs_A")
    write_subtitle("Cs_A", "EN")
    write_subtitle("Cs_A", "JP")

    probe_usm(tmp_app_root / "Cs_A.usm", KEYS_DATA, reporter)

    assert last_probe_event(reporter) == {
        "file": "Cs_A.usm",
        "stem": "Cs_A",
        "key": True,
        "subtitles": ["EN", "JP"],
        "vs_script": "Cs_A",
    }


def test_probe_reports_missing_and_never_prompts(tmp_app_root, reporter, monkeypatch):
    monkeypatch.setattr(pipeline, "find_vs_script", lambda stem: None)

    probe_usm(tmp_app_root / "Cs_A.usm", {}, reporter)

    data = last_probe_event(reporter)
    assert data["key"] is False
    assert data["subtitles"] == []
    assert data["vs_script"] is None
    assert reporter.prompts == []


class CancelDuringDemux(FakeReporter):
    """Stays quiet through process_usm's two pre-demux checkpoints, then reports a
    cancel at the first checkpoint inside the demux loop."""

    def __init__(self):
        super().__init__()
        self.checks = 0

    def cancel_requested(self):
        self.checks += 1
        return self.checks > 2


def make_options(tmp_path, no_cleanup: bool = False) -> Options:
    (tmp_path / "out").mkdir(exist_ok=True)
    return Options(
        output=str(tmp_path / "out"),
        no_cleanup=no_cleanup,
        vapoursynth=False,
        crf=0.0,
        preset="fast",
        x265_params="",
    )


def make_cancel_run(tmp_path, no_cleanup: bool):
    usm_file = tmp_path / "Cs_Test.usm"
    usm_file.write_bytes(b"".join(chunk(b"@SFA", b"x") for _ in range(150)))
    keys = SimpleNamespace(decryption_key=lambda stem: (bytes(4), bytes(4)))
    return usm_file, make_options(tmp_path, no_cleanup), keys


def test_cancel_mid_demux_cleans_partial_files(tmp_path):
    usm_file, opts, keys = make_cancel_run(tmp_path, no_cleanup=False)
    with pytest.raises(Cancelled):
        process_usm(usm_file, opts, CancelDuringDemux(), keys)
    assert not list((tmp_path / "out" / "Cs_Test").glob("*.hca"))


def test_cancel_mid_demux_keeps_files_with_no_cleanup(tmp_path):
    usm_file, opts, keys = make_cancel_run(tmp_path, no_cleanup=True)
    with pytest.raises(Cancelled):
        process_usm(usm_file, opts, CancelDuringDemux(), keys)
    assert (tmp_path / "out" / "Cs_Test" / "Cs_Test_0.hca").exists()


def test_missing_key_falls_back_to_cracking(tmp_path, reporter, monkeypatch):
    """A key that is neither on disk nor upstream is recovered from the video itself,
    and the file is only skipped once that has also come up empty."""
    cracked = []

    def crack(usm_file, reporter):
        cracked.append(usm_file)
        return Recovery(None, "no IVF video stream in this file")

    monkeypatch.setattr(pipeline, "crack_key", crack)
    # A real Keys with no keys.json and no upstream, so every lookup genuinely misses.
    monkeypatch.setattr(resources.keys, "fetch_upstream_keys", lambda: None)
    usm_file = tmp_path / "Cs_Test.usm"
    usm_file.write_bytes(chunk(b"@SFA", b"x"))

    process_usm(usm_file, make_options(tmp_path), reporter, Keys(reporter))

    assert cracked == [usm_file]
    assert ("job_skipped", {"file": "Cs_Test.usm", "reason": "no_key"}) in reporter.events


# --- key recovery ---


def last_crack_event(reporter):
    kind, data = reporter.events[-1]
    assert kind == "crack"
    return data


def test_crack_reports_key_and_video_key(tmp_app_root, reporter, monkeypatch):
    """videoKey is the keys.json half: what is left of the combined key once the
    filename hash is subtracted back out."""
    combined = (calculate_key_from_filename("Cs_A") + 777) & 0xFFFFFFFFFFFFFF
    key_bytes = combined.to_bytes(8, "little")
    monkeypatch.setattr(
        pipeline, "crack_key", lambda f, r: Recovery((key_bytes[:4], key_bytes[4:]), "")
    )

    crack_usm(tmp_app_root / "Cs_A.usm", reporter)

    assert last_crack_event(reporter) == {
        "file": "Cs_A.usm",
        "stem": "Cs_A",
        "key": combined,
        "video_key": 777,
        "reason": "",
    }


def test_crack_failure_keeps_the_same_event_shape(tmp_app_root, reporter, monkeypatch):
    """The GUI deserializes every crack event into one record, so the fields are fixed."""
    monkeypatch.setattr(pipeline, "crack_key", lambda f, r: Recovery(None, "no IVF video stream"))

    crack_usm(tmp_app_root / "Cs_A.usm", reporter)

    assert last_crack_event(reporter) == {
        "file": "Cs_A.usm",
        "stem": "Cs_A",
        "key": None,
        "video_key": None,
        "reason": "no IVF video stream",
    }


def test_crack_batch_continues_past_an_unreadable_file(tmp_app_root, reporter, monkeypatch):
    def crack(usm_file, reporter):
        if usm_file.name == "Cs_Bad.usm":
            raise CharlotteError("Corrupt USM chunk in Cs_Bad.usm")
        return Recovery((bytes(4), bytes(4)), "")

    monkeypatch.setattr(pipeline, "crack_key", crack)

    crack_all([tmp_app_root / "Cs_Bad.usm", tmp_app_root / "Cs_A.usm"], reporter)

    assert [kind for kind, _ in reporter.events] == ["error", "crack", "crack_summary"]
    assert reporter.events[-1][1] == {"recovered": 1, "unrecovered": 1}


def test_crack_batch_stops_cleanly_on_cancel(tmp_app_root, reporter, monkeypatch):
    """A cancel mid-batch is a cancelled event naming the file, not a traceback."""

    def crack(usm_file, reporter):
        raise Cancelled

    monkeypatch.setattr(pipeline, "crack_key", crack)

    crack_all([tmp_app_root / "Cs_A.usm", tmp_app_root / "Cs_B.usm"], reporter)

    assert reporter.events == [("cancelled", {"file": "Cs_A.usm"})]


def test_probe_remaps_subtitle_stem_only(tmp_app_root, reporter, monkeypatch):
    """BASENAME_FIXES applies to the subtitle lookup, while the key and the VapourSynth
    script keep using the original stem."""
    seen_vs_stems = []
    # list.append takes the stem and returns None, doubling as a "no script found" stub.
    monkeypatch.setattr(pipeline, "find_vs_script", seen_vs_stems.append)
    write_subtitle("Cs_DQAQ200211_WanYeXianVideo", "EN")

    probe_usm(tmp_app_root / "Cs_200211_WanYeXianVideo.usm", {}, reporter)

    data = last_probe_event(reporter)
    assert data["stem"] == "Cs_200211_WanYeXianVideo"
    assert data["subtitles"] == ["EN"]  # found under the remapped stem
    assert seen_vs_stems == ["Cs_200211_WanYeXianVideo"]
