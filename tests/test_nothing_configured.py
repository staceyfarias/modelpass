"""R10: zero connections is a legitimate state, on every read path.

A fresh install has no AI connectivity by design (D2), so "nothing configured"
is not an error and never has been. What this file pins is the harder half of
that rule: the *other* four ways a machine can have nothing to read, including
the two where the files exist and this process cannot get at them.

The matrix, parametrised over every read path in the library:

===========================  ===========================================
``missing-home``             no modelpass home directory at all
``empty-home``               the directory exists and holds nothing
``empty-file``               a connection file with a version and no tables
``unreadable-connections``   the connection file cannot be opened
``unreadable-secrets``       the secrets file cannot be opened
===========================  ===========================================

The contract, stated once and asserted for each:

* Reads return **empty** and do not raise. Not a default connection, not a
  guess, not an exception a caller has to know the name of.
* For the two unreadable cases a **note** is attached saying the file could not
  be read, because an empty list that silently means "your file is there and I
  cannot see it" is worse than a crash.
* **Writes refuse** on an unreadable file, with a named error. Emitting this
  build's shape over bytes nobody parsed is how a user loses what was in there.

Permission denied is simulated rather than staged with ``chmod``: the suite runs
as a user who can read their own files on POSIX and as one whose ACLs are not
the CI runner's business on Windows, so the only portable way to get an
``OSError`` out of ``open()`` on a file that exists is to make ``open()`` raise.
"""

from __future__ import annotations

import pytest

from modelpass.bridge import Bridge
from modelpass.capabilities import Capability, CapabilityRegistry
from modelpass.cli import main
from modelpass.connections import Connection, CredentialRef
from modelpass.errors import ConfigError, NoSuchConnection
from modelpass.runtimes import Runtime
from modelpass.secrets import SecretStore
from modelpass.store import ConnectionStore
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode

STATES = (
    "missing-home",
    "empty-home",
    "empty-file",
    "unreadable-connections",
    "unreadable-secrets",
)

#: The two states where a file is there and this process cannot read it.
UNREADABLE = ("unreadable-connections", "unreadable-secrets")


def _deny(monkeypatch, target):
    """Make ``open()`` on exactly ``target`` raise PermissionError, as the OS would."""
    import pathlib

    real = pathlib.Path.open

    def guarded(self, *args, **kwargs):
        if self == target:
            raise PermissionError(13, "Permission denied")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", guarded)


@pytest.fixture(params=STATES)
def nothing(request, tmp_path, monkeypatch):
    """A bridge over a home in one of the five states. Returns (bridge, state)."""
    state = request.param
    root = tmp_path / "home"
    if state != "missing-home":
        root.mkdir(parents=True)
    store = ConnectionStore(root)
    if state == "empty-file":
        store.path.write_text("version = 1\n", encoding="utf-8")
    if state == "unreadable-connections":
        store.path.write_text("version = 1\n", encoding="utf-8")
        _deny(monkeypatch, store.path)
    if state == "unreadable-secrets":
        SecretStore(root).path.write_text("version = 1\n", encoding="utf-8")
        _deny(monkeypatch, SecretStore(root).path)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={
            Runtime.ANTHROPIC_SDK: FakeAdapter([]),
            Runtime.OPENAI_SDK: FakeAdapter([]),
        },
        env={},
    )
    return bridge, state


# --- 1. every read path answers "nothing", and none of them raises ----------------


def test_the_bridge_lists_no_connections(nothing):
    bridge, _ = nothing
    assert bridge.connections() == ()
    assert bridge.accounts() == ()


def test_find_matches_nothing_on_every_axis(nothing):
    bridge, _ = nothing
    assert bridge.find() == ()
    assert bridge.find(vendor="anthropic") == ()
    assert bridge.find(runtime=Runtime.ANTHROPIC_SDK) == ()
    assert bridge.find(auth_mode=AuthMode.SUBSCRIPTION) == ()
    assert bridge.find(capability=Capability.CHAT) == ()


def test_the_store_lists_and_names_nothing(nothing):
    bridge, _ = nothing
    assert bridge.store.list() == ()
    assert bridge.store.names() == ()
    assert bridge.store.load() == {}
    assert bridge.store.has("claude-sub") is False


def test_compatibility_answers_rather_than_raising(nothing):
    bridge, state = nothing
    compatibility = bridge.store.compatibility()
    assert compatibility.carried_keys == {}
    assert compatibility.refused == {}
    if state == "unreadable-connections":
        assert compatibility.unreadable is not None
        assert compatibility.writable is False
    else:
        assert compatibility.unreadable is None


def test_the_secret_store_lists_no_entries_and_no_orphans(nothing):
    bridge, _ = nothing
    assert bridge.secrets.entries() == ()
    assert bridge.secrets.orphans(bridge.connections()) == ()
    assert bridge.secrets.has("anything") is False
    # Permissions are an observation about a file, and it must be safe to ask
    # for one that is absent or unreadable.
    assert isinstance(bridge.secrets.permissions().note, str)


def test_settings_and_the_run_log_fall_back_to_their_defaults(nothing):
    bridge, _ = nothing
    assert bridge.store.settings().run_log is True
    assert bridge.run_log is not None


def test_validate_reports_a_missing_connection_rather_than_crashing(nothing):
    """A *lookup* of a name that is not there is still NoSuchConnection.

    R10 is about listings degrading, not about inventing a connection nobody
    configured. ``validate`` turns the lookup into a report, which is the shape
    a Settings page wants, and a caller that asked the store directly gets the
    named error rather than a KeyError.
    """
    bridge, _ = nothing
    report = bridge.validate("claude-sub")
    assert report.ok is False
    assert report.problems
    with pytest.raises(NoSuchConnection):
        bridge.connection("claude-sub")


def test_validate_still_works_on_a_connection_handed_to_it(nothing):
    """The offline validator does not need the file it is validating against."""
    bridge, _ = nothing
    report = bridge.validate(
        Connection(
            name="claude-sub",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            credential_ref=CredentialRef.native_login(),
        )
    )
    assert report.ok is True


# --- 2. the surfaces a user actually looks at -------------------------------------


@pytest.mark.parametrize("argv", (["list"], ["accounts"], ["check"]))
def test_the_cli_exits_zero_and_says_so(nothing, argv):
    import io

    bridge, _ = nothing
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, bridge=bridge, out=out, err=err, confirm=lambda _q: False)
    assert code == 0, err.getvalue()
    assert "nothing to check" in out.getvalue() or "No " in out.getvalue()
    assert err.getvalue() == ""


def test_the_bench_index_renders(nothing):
    pytest.importorskip("flask")
    from modelpass.bench.app import create_app

    bridge, _ = nothing
    client = create_app(bridge=bridge).test_client()
    assert client.get("/").status_code == 200


# --- 3. an unreadable file is said out loud, and refuses writes -------------------


def test_an_unreadable_file_attaches_a_note(nothing):
    bridge, state = nothing
    if state not in UNREADABLE:
        assert bridge.store.compatibility().notes() == ()
        assert bridge.secrets.notes() == ()
        return
    if state == "unreadable-connections":
        notes = bridge.store.compatibility().notes()
    else:
        notes = bridge.secrets.notes()
    assert notes, "an unreadable file must say so, not just read as empty"
    assert "could not be read" in notes[0]
    assert "Permission denied" in notes[0]


def test_check_prints_the_note_for_an_unreadable_file(nothing):
    import io

    bridge, state = nothing
    if state not in UNREADABLE:
        return
    out = io.StringIO()
    assert main(["check"], bridge=bridge, out=out, err=io.StringIO()) == 0
    assert "could not be read" in out.getvalue()


def test_writes_refuse_on_an_unreadable_file(nothing):
    bridge, state = nothing
    connection = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    if state == "unreadable-connections":
        with pytest.raises(ConfigError, match="Permission denied"):
            bridge.store.add(connection)
        return
    if state == "unreadable-secrets":
        with pytest.raises(ConfigError, match="Permission denied"):
            bridge.secrets.set("claude-api", "sk-not-a-real-key")
        return
    # Everywhere else a write is the ordinary thing, and it works.
    bridge.store.add(connection)
    assert bridge.store.names() == ("claude-sub",)
