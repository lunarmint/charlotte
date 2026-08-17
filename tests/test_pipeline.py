from types import SimpleNamespace

import pytest

import pipeline

from conftest import FakeReporter
from pipeline import Options, probe_usm, process_usm
from resources.subtitles import local_subtitle_path
from test_usm import chunk
from utils.errors import Cancelled


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


def make_cancel_run(tmp_path, no_cleanup: bool):
    usm_file = tmp_path / "Cs_Test.usm"
    usm_file.write_bytes(b"".join(chunk(b"@SFA", b"x") for _ in range(150)))
    opts = Options(
        output=str(tmp_path / "out"),
        no_cleanup=no_cleanup,
        vapoursynth=False,
        crf=0.0,
        preset="fast",
        x265_params="",
    )
    (tmp_path / "out").mkdir()
    keys = SimpleNamespace(decryption_key=lambda name: (bytes(4), bytes(4)))
    return usm_file, opts, keys


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
