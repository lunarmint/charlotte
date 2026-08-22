import types

import pytest
import typer

from typer.testing import CliRunner

import main

from conftest import forbid_call
from utils.errors import Cancelled, CharlotteError
from utils.version import __version__


runner = CliRunner()


def make_usm(directory, name="Cs_Test.usm"):
    path = directory / name
    path.write_bytes(b"")
    return path


@pytest.fixture
def pipeline_stub(monkeypatch):
    """Stub everything demux runs after flag validation; records process_usm calls.
    Tests that need a failing pipeline re-patch main.process_usm on top."""
    stub = types.SimpleNamespace(files=[], opts=None)

    def fake_process(usm_file, opts, reporter, keys):
        stub.files.append(usm_file)
        stub.opts = opts

    monkeypatch.setattr(main, "process_usm", fake_process)
    monkeypatch.setattr(main, "Keys", lambda reporter, manual_key=None: None)
    monkeypatch.setattr(main, "sync_subtitles", lambda reporter: None)
    monkeypatch.setattr(main, "fetch_font", lambda: None)
    return stub


# --- choice helpers ---


def test_choice_option_metavar_lowercases():
    option = main.choice_option("--lang", help="", choices=["JA", "en"])
    assert option.metavar == "[ja|en]"


def test_choice_normalizer_case_insensitive():
    normalize = main.choice_normalizer(["ja", "EN"])
    assert normalize("JA") == "ja"
    assert normalize("en") == "EN"


def test_choice_normalizer_rejects_unknown():
    normalize = main.choice_normalizer(["ja", "en"])
    with pytest.raises(typer.BadParameter, match="Must be one of: ja, en"):
        normalize("xx")


# --- modes that need no input files ---


def test_version_prints_and_exits(monkeypatch):
    monkeypatch.setattr(main, "collect_files", forbid_call)
    result = runner.invoke(main.app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_update_runs_without_input_files(monkeypatch):
    called = []
    monkeypatch.setattr(main, "run_update", lambda reporter, json_mode: called.append(json_mode))
    monkeypatch.setattr(main, "collect_files", forbid_call)

    assert runner.invoke(main.app, ["--update"]).exit_code == 0
    assert called == [False]


@pytest.mark.parametrize("extra", [["Cs_A.usm"], ["--probe"], ["--crack"], ["--key", "1"]])
def test_update_rejects_every_other_mode(monkeypatch, extra):
    """--update replaces the binary and exits; combining it with a run would leave the
    engine mid-batch on a version that just got swapped out from under it."""
    monkeypatch.setattr(main, "run_update", forbid_call)
    assert runner.invoke(main.app, ["--update", *extra]).exit_code == 1


def test_no_input_is_an_error():
    assert runner.invoke(main.app, []).exit_code == 1


# --- input collection ---


def test_rejects_non_usm_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"")
    assert runner.invoke(main.app, [str(path)]).exit_code == 1


def test_rejects_empty_directory(tmp_path):
    assert runner.invoke(main.app, [str(tmp_path)]).exit_code == 1


def test_rejects_missing_path(tmp_path):
    assert runner.invoke(main.app, [str(tmp_path / "nope.usm")]).exit_code == 1


def test_duplicate_inputs_processed_once(pipeline_stub, tmp_path):
    usm = make_usm(tmp_path)
    args = [str(usm), str(usm), "-o", str(tmp_path / "out")]
    assert runner.invoke(main.app, args).exit_code == 0
    assert pipeline_stub.files == [usm]


def test_directory_glob_sorted(pipeline_stub, tmp_path):
    second = make_usm(tmp_path, "Cs_B.usm")
    first = make_usm(tmp_path, "Cs_A.usm")
    args = [str(tmp_path), "-o", str(tmp_path / "out")]
    assert runner.invoke(main.app, args).exit_code == 0
    assert pipeline_stub.files == [first, second]


# --- flag validation ---


def test_key_requires_single_input(tmp_path):
    files = [make_usm(tmp_path, "Cs_A.usm"), make_usm(tmp_path, "Cs_B.usm")]
    result = runner.invoke(main.app, [*map(str, files), "--key", "1"])
    assert result.exit_code == 1


def test_crack_rejects_probe_and_key(tmp_path):
    """--crack reports keys instead of using them, so neither flag has a meaning here."""
    usm = make_usm(tmp_path)
    assert runner.invoke(main.app, [str(usm), "--crack", "--probe"]).exit_code == 1
    assert runner.invoke(main.app, [str(usm), "--crack", "--key", "1"]).exit_code == 1


def test_invalid_choice_flag_is_usage_error(tmp_path):
    usm = make_usm(tmp_path)
    assert runner.invoke(main.app, [str(usm), "-da", "xx"]).exit_code == 2
    assert runner.invoke(main.app, [str(usm), "-ds", "xx"]).exit_code == 2
    assert runner.invoke(main.app, [str(usm), "-ac", "xx"]).exit_code == 2


def test_flags_normalized_into_options(pipeline_stub, tmp_path):
    usm = make_usm(tmp_path)
    args = [
        str(usm),
        "-o", str(tmp_path / "out"),
        "--default-audio", "EN",
        "--default-sub", "chs",
        "-ac", "OPUS",
        "-nc",
        "-f",
    ]  # fmt: skip
    assert runner.invoke(main.app, args).exit_code == 0

    opts = pipeline_stub.opts
    assert opts.default_audio == "en"
    assert opts.default_subtitle == "CHS"
    assert opts.audio_codec == "opus"
    assert opts.no_cleanup is True
    assert opts.flat is True


def test_default_options(pipeline_stub, tmp_path):
    usm = make_usm(tmp_path)
    assert runner.invoke(main.app, [str(usm), "-o", str(tmp_path / "out")]).exit_code == 0

    opts = pipeline_stub.opts
    assert opts.default_audio == "ja"
    # Defaults run through the normalizer too: "en" becomes the canonical "EN" code.
    assert opts.default_subtitle == "EN"
    assert opts.audio_codec == "flac"


# --- run outcomes ---


def test_processing_failure_exits_nonzero(pipeline_stub, monkeypatch, tmp_path):
    def fail_process(usm_file, opts, reporter, keys):
        raise CharlotteError("boom")

    monkeypatch.setattr(main, "process_usm", fail_process)
    usm = make_usm(tmp_path)
    result = runner.invoke(main.app, [str(usm), "-o", str(tmp_path / "out")])
    assert result.exit_code == 1


def test_cancelled_is_clean_exit(pipeline_stub, monkeypatch, tmp_path):
    def cancel_process(usm_file, opts, reporter, keys):
        raise Cancelled

    monkeypatch.setattr(main, "process_usm", cancel_process)
    usm = make_usm(tmp_path)
    result = runner.invoke(main.app, [str(usm), "-o", str(tmp_path / "out")])
    assert result.exit_code == 0


def test_probe_skips_sync_and_pipeline(monkeypatch, tmp_path):
    probed = []
    monkeypatch.setattr(main, "probe_usm", lambda usm_file, keys, reporter: probed.append(usm_file))
    monkeypatch.setattr(main, "load_local_keys", dict)
    monkeypatch.setattr(main, "Keys", forbid_call)
    monkeypatch.setattr(main, "sync_subtitles", forbid_call)
    monkeypatch.setattr(main, "process_usm", forbid_call)

    usm = make_usm(tmp_path)
    assert runner.invoke(main.app, [str(usm), "--probe"]).exit_code == 0
    assert probed == [usm]


def test_crack_skips_keys_and_pipeline(monkeypatch, tmp_path):
    """--crack needs neither keys.json nor subtitles: the file is the only input."""
    cracked = []
    monkeypatch.setattr(main, "crack_all", lambda files, reporter: cracked.extend(files))
    monkeypatch.setattr(main, "Keys", forbid_call)
    monkeypatch.setattr(main, "sync_subtitles", forbid_call)
    monkeypatch.setattr(main, "process_usm", forbid_call)

    usm = make_usm(tmp_path)
    assert runner.invoke(main.app, [str(usm), "--crack"]).exit_code == 0
    assert cracked == [usm]
