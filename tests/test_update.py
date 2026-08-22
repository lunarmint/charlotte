import sys

import pytest

import utils.update

from utils.errors import CharlotteError
from utils.update import (
    UpdateInfo,
    asset_download_url,
    check_for_update,
    is_standalone_exe,
    looks_like_exe,
    parse_version,
    report_update,
)
from utils.version import __version__


# The fixed field set the `update` event must always carry, regardless of outcome.
UPDATE_FIELDS = {"current", "latest", "available", "url", "notes", "download", "reason"}


def release(tag: str, url: str = "https://example/rel", body: str = "notes") -> dict:
    return {"tag_name": tag, "html_url": url, "body": body}


@pytest.mark.parametrize("tag", ["v0.4.0", "0.4.0", "v2.0", "v1.2.3b1", "1.2.3rc2"])
def test_parse_version_accepts_release_tags(tag):
    parse_version(tag)  # must not raise on any supported tag form


def test_parse_version_ignores_leading_v():
    assert parse_version("v1.2.3") == parse_version("1.2.3")


def test_parse_version_orders_numerically():
    # Lexicographic ordering would wrongly rank "0.9.0" above "0.10.0".
    assert parse_version("v0.10.0") > parse_version("v0.9.0")


def test_parse_version_mixed_length_orders_by_release():
    # A flat comparison key would rank the phase rank of "2.0" against the patch
    # digit of "2.0.1"; grouping the release numbers keeps 2.0 below 2.0.1.
    assert parse_version("v2.0.1") > parse_version("v2.0")


def test_version_is_a_usable_tag():
    # __version__ comes from pyproject.toml, so a typo there would otherwise only surface
    # as a bad comparison during an update check.
    assert parse_version(__version__)


def test_prereleases_sort_before_final():
    ascending = ["1.2.3a1", "1.2.3b1", "1.2.3b2", "1.2.3rc1", "1.2.3"]
    keys = [parse_version(tag) for tag in ascending]
    assert keys == sorted(keys)


def test_newer_prerelease_beats_older_final():
    assert parse_version("1.3.0b1") > parse_version("1.2.9")


def test_update_available(monkeypatch):
    monkeypatch.setattr(utils.update, "fetch_latest_release", lambda: release("v99.0.0"))
    info = check_for_update()
    expected = UpdateInfo(
        current=__version__,
        latest="v99.0.0",
        available=True,
        url="https://example/rel",
        notes="notes",
        download=None,
        reason=None,
    )
    assert info == expected


def test_update_carries_download_url(monkeypatch):
    with_asset = release("v99.0.0") | {
        "assets": [{"name": "charlotte.exe", "browser_download_url": "https://example/dl"}]
    }
    monkeypatch.setattr(utils.update, "fetch_latest_release", lambda: with_asset)
    assert check_for_update().download == "https://example/dl"


def test_up_to_date(monkeypatch):
    monkeypatch.setattr(utils.update, "fetch_latest_release", lambda: release(f"v{__version__}"))
    info = check_for_update()
    assert info.available is False
    assert info.latest == f"v{__version__}"
    assert info.reason is None


def test_current_ahead_of_release(monkeypatch):
    # A dev build ahead of the last tag must not report an update.
    monkeypatch.setattr(utils.update, "fetch_latest_release", lambda: release("v0.0.1"))
    assert check_for_update().available is False


def test_network_failure(monkeypatch):
    monkeypatch.setattr(utils.update, "fetch_latest_release", lambda: None)
    assert check_for_update() == UpdateInfo(current=__version__, reason="network error")


def test_missing_tag(monkeypatch):
    monkeypatch.setattr(utils.update, "fetch_latest_release", lambda: {"html_url": "x"})
    info = check_for_update()
    assert info.available is False
    assert info.latest is None
    assert info.reason == "no release tag found"


def test_unrecognized_tag(monkeypatch):
    # A tag parse_version cannot digest must come back as a declined check, not a crash.
    monkeypatch.setattr(utils.update, "fetch_latest_release", lambda: release("nightly"))
    info = check_for_update()
    assert info.available is False
    assert info.reason == "unrecognized release tag"


def test_event_shape_fixed_on_success(monkeypatch, reporter):
    monkeypatch.setattr(utils.update, "fetch_latest_release", lambda: release("v99.0.0"))
    report_update(reporter)
    assert len(reporter.events) == 1
    kind, data = reporter.events[0]
    assert kind == "update"
    assert set(data) == UPDATE_FIELDS


def test_event_shape_fixed_on_failure(monkeypatch, reporter):
    monkeypatch.setattr(utils.update, "fetch_latest_release", lambda: None)
    report_update(reporter)
    kind, data = reporter.events[0]
    assert kind == "update"
    assert set(data) == UPDATE_FIELDS  # identical field set even when the check failed


# --- self-apply (step 2) ---


def test_asset_download_url_picks_exe():
    release = {
        "assets": [
            {"name": "keys.json", "browser_download_url": "u1"},
            {"name": "charlotte.exe", "browser_download_url": "u2"},
        ]
    }
    assert asset_download_url(release) == "u2"


def test_asset_download_url_none_when_no_exe():
    only_md = {"assets": [{"name": "readme.md", "browser_download_url": "u"}]}
    assert asset_download_url(only_md) is None
    assert asset_download_url({}) is None


def test_apply_update_declines_without_asset(reporter):
    # No .exe attached to the release: fail up front, before any download starts.
    info = UpdateInfo(current=__version__, latest="v99.0.0", available=True)
    assert utils.update.apply_update(info, reporter) is False


def test_looks_like_exe(tmp_path):
    good = tmp_path / "a.exe"
    good.write_bytes(b"MZ\x90\x00rest")
    bad = tmp_path / "b.exe"
    bad.write_bytes(b"<!doctype html>")
    assert looks_like_exe(good) is True
    assert looks_like_exe(bad) is False
    assert looks_like_exe(tmp_path / "missing.exe") is False


@pytest.mark.parametrize(
    "frozen, json_mode, expected",
    [
        (False, False, False),  # source run: nothing to swap
        (True, True, False),  # --json: the GUI drives updates
        (True, False, True),  # standalone frozen CLI: may self-apply
    ],
)
def test_is_standalone_exe(monkeypatch, frozen, json_mode, expected):
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    assert is_standalone_exe(json_mode) is expected


def test_swap_binary_replaces_and_keeps_old(monkeypatch, tmp_path):
    exe = tmp_path / "charlotte.exe"
    exe.write_bytes(b"OLD")
    new = tmp_path / "charlotte.exe.new"
    new.write_bytes(b"NEW")
    monkeypatch.setattr(utils.update, "running_exe", lambda: exe)

    utils.update.swap_binary(new)
    assert exe.read_bytes() == b"NEW"
    assert (tmp_path / "charlotte.exe.old").read_bytes() == b"OLD"
    assert not new.exists()


def test_swap_binary_rolls_back_when_new_missing(monkeypatch, tmp_path):
    exe = tmp_path / "charlotte.exe"
    exe.write_bytes(b"OLD")
    missing_new = tmp_path / "charlotte.exe.new"  # never created, so the rename raises
    monkeypatch.setattr(utils.update, "running_exe", lambda: exe)

    with pytest.raises(CharlotteError):
        utils.update.swap_binary(missing_new)
    assert exe.read_bytes() == b"OLD"  # rolled back: the working exe is left intact


def test_clear_stale_binary_removes_old(monkeypatch, tmp_path):
    exe = tmp_path / "charlotte.exe"
    exe.write_bytes(b"NEW")
    old = tmp_path / "charlotte.exe.old"
    old.write_bytes(b"OLD")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(utils.update, "running_exe", lambda: exe)

    utils.update.clear_stale_binary()
    assert not old.exists()
    assert exe.exists()


def test_clear_stale_binary_noop_from_source(monkeypatch):
    # Not frozen: no on-disk exe to clean, so it must do nothing and not raise.
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    utils.update.clear_stale_binary()
