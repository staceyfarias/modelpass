"""Where the connection store lands, across the rename.

Four states a real machine can be in, and each one has to be right the first
time: a fresh install, a machine that configured connections under the old name,
a machine that sets both environment variables, and one that still sets only the
old one. Getting the second wrong loses a user's connections; getting the fourth
wrong loses them silently, which is worse.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from modelpass import store as store_module
from modelpass.store import ConnectionStore, default_home


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """A home nobody else has touched, and no inherited environment.

    These tests are the only ones that exercise the *default* root, so they are
    also the only ones that could reach a developer's real ``~/.subpass`` and
    copy it somewhere. They cannot: ``Path.home`` is redirected first.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("MODELPASS_HOME", raising=False)
    monkeypatch.delenv("SUBPASS_HOME", raising=False)
    monkeypatch.setattr(store_module, "_said_legacy_env", False)
    monkeypatch.setattr(store_module, "_said_copy_forward", False)
    return home


def _legacy_home_with_connections(home: Path) -> Path:
    legacy = home / ".subpass"
    legacy.mkdir()
    (legacy / "connections.toml").write_text(
        'version = 1\n\n[connections.claude-sub]\nruntime = "anthropic-sdk"\n'
        'authMode = "subscription"\ncredentialRef = "native-login"\n',
        encoding="utf-8",
    )
    (legacy / "runs.jsonl").write_text('{"connection": "claude-sub"}\n', encoding="utf-8")
    (legacy / "secrets.toml").write_text('[secrets]\n', encoding="utf-8")
    return legacy


def test_fresh_install_uses_the_new_home_and_copies_nothing(_isolated_home):
    assert default_home() == _isolated_home / ".modelpass"
    assert ConnectionStore().path == _isolated_home / ".modelpass" / "connections.toml"
    # A fresh install has no connections and no directory conjured for it.
    assert not (_isolated_home / ".modelpass").exists()
    assert ConnectionStore().load() == {}


def test_an_existing_old_home_is_copied_forward_once_with_a_note(_isolated_home, capsys):
    legacy = _legacy_home_with_connections(_isolated_home)
    new = _isolated_home / ".modelpass"

    root = default_home()

    assert root == new
    assert (new / "connections.toml").read_text(encoding="utf-8") == (
        legacy / "connections.toml"
    ).read_text(encoding="utf-8")
    assert (new / "runs.jsonl").is_file()
    assert (new / "secrets.toml").is_file()
    # Copied, never moved.
    assert (legacy / "connections.toml").is_file()

    note = capsys.readouterr().err
    assert str(legacy) in note
    assert str(new) in note
    assert "left untouched" in note

    # The connection survives the copy as a connection, not just as bytes.
    assert list(ConnectionStore().load()) == ["claude-sub"]

    # Second call: no second copy, no second note.
    default_home()
    assert capsys.readouterr().err == ""


def test_copy_forward_does_not_overwrite_an_existing_new_home(_isolated_home, capsys):
    _legacy_home_with_connections(_isolated_home)
    new = _isolated_home / ".modelpass"
    new.mkdir()
    (new / "connections.toml").write_text("version = 1\n", encoding="utf-8")

    assert default_home() == new
    assert (new / "connections.toml").read_text(encoding="utf-8") == "version = 1\n"
    assert not (new / "runs.jsonl").exists()
    assert capsys.readouterr().err == ""


def test_both_env_vars_set_means_the_new_one_wins_silently(_isolated_home, monkeypatch):
    monkeypatch.setenv("MODELPASS_HOME", str(_isolated_home / "new"))
    monkeypatch.setenv("SUBPASS_HOME", str(_isolated_home / "old"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        root = default_home()

    assert root == _isolated_home / "new"
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_the_old_env_var_alone_is_honoured_with_a_deprecation_warning(
    _isolated_home, monkeypatch
):
    monkeypatch.setenv("SUBPASS_HOME", str(_isolated_home / "old"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        root = default_home()
        default_home()

    assert root == _isolated_home / "old"
    assert ConnectionStore().path == _isolated_home / "old" / "connections.toml"

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, "the old env var is deprecated once, not once per store"
    message = str(deprecations[0].message)
    assert "SUBPASS_HOME" in message
    assert "MODELPASS_HOME" in message


def test_an_env_var_never_triggers_the_copy_forward(_isolated_home, monkeypatch):
    """The fallback is about *the default home*, not about any named directory.

    Someone who points ``MODELPASS_HOME`` at an empty directory asked for an
    empty directory. Seeding it from ``~/.subpass`` behind their back would
    silently re-introduce connections they moved away from.
    """
    _legacy_home_with_connections(_isolated_home)
    monkeypatch.setenv("MODELPASS_HOME", str(_isolated_home / "elsewhere"))

    assert default_home() == _isolated_home / "elsewhere"
    assert not (_isolated_home / "elsewhere").exists()
    assert not (_isolated_home / ".modelpass").exists()
