import io

from types import SimpleNamespace

import orjson
import pytest
import typer

from rich.progress import DownloadColumn, MofNCompleteColumn, TransferSpeedColumn

from utils.errors import Cancelled
from utils.reporter import PROTOCOL_VERSION, ConsoleReporter, JsonReporter
from utils.reporter.console import SpeedColumn, make_progress


def make_reporter(stdin_text=""):
    """JsonReporter writing to a captured StringIO; StringIO stdin exercises the
    readline (non-pipe) path."""
    return JsonReporter(out=io.StringIO(), stdin=io.StringIO(stdin_text))


def events_of(reporter):
    """Every NDJSON event the reporter has emitted so far, parsed."""
    return [orjson.loads(line) for line in reporter.out.getvalue().splitlines()]


# --- event shapes (the contract with the GUI frontend) ---


def test_session_start_announces_protocol():
    reporter = make_reporter()
    assert events_of(reporter) == [{"type": "session_start", "protocol": PROTOCOL_VERSION}]


def test_log_event_shape():
    reporter = make_reporter()
    reporter.log("warning", "watch out")
    assert events_of(reporter)[-1] == {"type": "log", "level": "warning", "message": "watch out"}


def test_custom_event_shape():
    reporter = make_reporter()
    reporter.event("job_start", file="a.usm", stem="a")
    assert events_of(reporter)[-1] == {"type": "job_start", "file": "a.usm", "stem": "a"}


def test_non_ascii_payload_not_dropped():
    """A Windows-style cp1252 stdout would raise UnicodeEncodeError (a ValueError) on
    non-ASCII text, which emit swallows. The reporter forces its stream to UTF-8 so
    free-form payloads like release notes survive instead of vanishing."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="")
    reporter = JsonReporter(out=stream, stdin=io.StringIO())
    reporter.event("update", notes="もふもふとりさん → v2")
    stream.flush()
    lines = raw.getvalue().decode("utf-8").splitlines()
    assert orjson.loads(lines[-1]) == {"type": "update", "notes": "もふもふとりさん → v2"}


def test_stage_events_wrap_progress():
    reporter = make_reporter()
    with reporter.task("demux", 4, unit="chunk") as task:
        task.advance()
        task.advance(3)

    stage = events_of(reporter)[1:]
    assert stage == [
        {"type": "stage", "stage": "demux", "status": "start", "total": 4, "unit": "chunk"},
        {"type": "progress", "stage": "demux", "current": 1, "total": 4},
        {"type": "progress", "stage": "demux", "current": 4, "total": 4},
        {"type": "stage", "stage": "demux", "status": "end"},
    ]


def test_progress_throttled_to_whole_percents():
    reporter = make_reporter()
    with reporter.task("encode", 1000) as task:
        for _ in range(1000):
            task.advance()

    progress = [event for event in events_of(reporter) if event["type"] == "progress"]
    # One event per whole percent instead of 1000 (and not fewer), the final tick always lands.
    assert 100 <= len(progress) <= 102
    assert progress[-1]["current"] == 1000


def test_progress_at_total_not_re_emitted():
    """Once a stage reports 100%, further zero-advances emit no duplicate events."""
    reporter = make_reporter()
    with reporter.task("demux", 4) as task:
        task.advance(4)
        task.advance(0)
        task.advance(0)

    progress = [event for event in events_of(reporter) if event["type"] == "progress"]
    assert progress == [{"type": "progress", "stage": "demux", "current": 4, "total": 4}]


def test_set_completed_lands_final_tick_without_duplicating():
    """demux ends by snapping the bar to the file size (set_completed), covering any
    trailing bytes too short to be a chunk. From short of the total that emits the
    closing 100% tick; already at the total it must not emit a duplicate."""
    short = make_reporter()
    with short.task("demux", 100) as task:
        task.advance(97)
        task.set_completed(100)
    progress = [event for event in events_of(short) if event["type"] == "progress"]
    assert progress == [
        {"type": "progress", "stage": "demux", "current": 97, "total": 100},
        {"type": "progress", "stage": "demux", "current": 100, "total": 100},
    ]

    already = make_reporter()
    with already.task("demux", 100) as task:
        task.advance(100)
        task.set_completed(100)
    progress = [event for event in events_of(already) if event["type"] == "progress"]
    assert progress == [{"type": "progress", "stage": "demux", "current": 100, "total": 100}]


# --- ask / cancel over stdin ---


def test_ask_emits_question_and_reads_answer():
    reporter = make_reporter('{"type": "answer", "id": "q0", "value": true}\n')
    assert reporter.ask("Overwrite?", default=False) is True
    assert events_of(reporter)[-1] == {
        "type": "question",
        "id": "q0",
        "prompt": "Overwrite?",
        "default": False,
    }


def test_ask_skips_garbage_and_wrong_ids():
    lines = (
        "not json\n"
        '{"type": "answer", "id": "q9", "value": true}\n'
        '{"type": "answer", "id": "q0", "value": false}\n'
    )
    reporter = make_reporter(lines)
    assert reporter.ask("Overwrite?", default=True) is False


def test_ask_returns_default_on_eof():
    reporter = make_reporter("")
    assert reporter.ask("Overwrite?", default=True) is True


def test_cancel_during_ask_sticks():
    reporter = make_reporter('{"type": "cancel"}\n')
    assert reporter.ask("Overwrite?", default=False) is False
    assert reporter.cancel_requested() is True
    with pytest.raises(Cancelled):
        reporter.checkpoint()


# --- ConsoleReporter (the same contract, drawn instead of serialized) ---


def columns_of(progress):
    return {type(column) for column in progress.columns}


def test_byte_stages_get_byte_columns():
    """demux and download count bytes, so they read as sizes and transfer rates rather
    than as a raw count of somethings."""
    assert {DownloadColumn, TransferSpeedColumn} <= columns_of(make_progress("B"))


def test_other_stages_get_counted_columns():
    assert {MofNCompleteColumn, SpeedColumn} <= columns_of(make_progress("frame"))


def test_speed_column_labels_its_unit():
    column = SpeedColumn("frame")
    assert "2.5 frame/s" in column.render(SimpleNamespace(finished_speed=None, speed=2.5)).plain
    # finished_speed wins once a task ends, so the last rate shown isn't a decaying one.
    assert "9.0 frame/s" in column.render(SimpleNamespace(finished_speed=9.0, speed=2.5)).plain


def test_speed_column_before_the_first_sample():
    column = SpeedColumn("frame")
    assert "-- frame/s" in column.render(SimpleNamespace(finished_speed=None, speed=None)).plain


def test_console_log_dispatches_by_level(caplog):
    ConsoleReporter().log("warning", "watch out")
    record = caplog.records[-1]
    assert (record.message, record.levelname) == ("watch out", "WARNING")


def test_console_task_tracks_progress():
    """The progress bar is transient, so what is asserted is the task state behind it."""
    reporter = ConsoleReporter()
    with reporter.task("demux", 10, unit="B") as task:
        task.advance(4)
        task.set_completed(10)
        progress, task_id = task.handle
        assert progress.tasks[task_id].completed == 10
    assert task.current == 10


def test_console_ask_defers_to_typer(monkeypatch):
    asked = []
    monkeypatch.setattr(typer, "confirm", lambda prompt, default: asked.append((prompt, default)))
    ConsoleReporter().ask("Overwrite?", default=True)
    assert asked == [("Overwrite?", True)]
