import contextlib
import struct

import pytest

import resources.fonts
import resources.keys
import resources.subtitles

from utils.reporter import Reporter, Task


def chunk(sig: bytes, payload: bytes, channel: int = 0, data_type: int = 0) -> bytes:
    """One USM chunk: 32-byte header (payload at the standard 0x18 offset, no
    padding) followed by the payload."""
    data_size = 0x18 + len(payload)
    header = struct.pack(">4sIxBHB2xB16x", sig, data_size, 0x18, 0, channel, data_type)
    return header + payload


def flag_value(cmd, flag):
    """The argument following a flag in an ffmpeg argument list,
    e.g. flag_value(cmd, "-preset") == "slower"."""
    return cmd[cmd.index(flag) + 1]


def forbid_call(*args, **kwargs):
    """Stub for anything a code path must not reach (network fetches, the pipeline)."""
    pytest.fail("Must not be called on this code path")


class FakeReporter(Reporter):
    """Test double: records logs/events/prompts, answers ask() with a scripted response."""

    def __init__(self, answer: bool = False):
        self.answer = answer
        self.logs = []
        self.events = []
        self.prompts = []

    def log(self, level, msg):
        self.logs.append((level, msg))

    @contextlib.contextmanager
    def task(self, stage, total, unit="it"):
        yield Task(self, stage, total)

    def update_task(self, handle, current, total):
        pass

    def ask(self, prompt, *, default=False):
        self.prompts.append(prompt)
        return self.answer

    def event(self, kind, **data):
        self.events.append((kind, data))


@pytest.fixture
def reporter():
    return FakeReporter()


@pytest.fixture(autouse=True)
def tmp_app_root(tmp_path, monkeypatch):
    """Redirect every module that persists files next to the executable (keys.json,
    Subtitle/, font/) into a scratch dir so tests don't affect the real ones."""
    for module in (resources.keys, resources.subtitles, resources.fonts):
        monkeypatch.setattr(module, "app_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def clear_upstream_cache():
    """fetch_upstream_keys is from functools.cache and keep results from leaking across tests."""
    resources.keys.fetch_upstream_keys.cache_clear()
