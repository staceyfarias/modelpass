"""The secrets file, and the rules that keep it in step with the connection file.

Ticket 1.3, requirement R14. Four things are being pinned here and each one is a
way this could go quietly wrong:

* a **pointer** is still all that lands in the shareable file;
* a **permission** is reported, never claimed -- 0600 asserted on POSIX, the
  Windows ACL attempt recorded with its real outcome;
* the **two files never diverge silently** -- delete and rename have written
  rules, an orphan is reported and never collected, and a referenced entry is
  never removed;
* a **value** appears nowhere but the one method that exists to return it.
"""

from __future__ import annotations

import os
import stat
import subprocess
from unittest import mock

import pytest

from modelpass.bridge import Bridge
from modelpass.cli import main
from modelpass.connections import Connection, CredentialKind, CredentialRef
from modelpass.errors import (
    ConfigError,
    CredentialRefIsSecret,
    InvalidConnection,
    NoSuchSecret,
    PreflightFailed,
    SecretStillReferenced,
)
from modelpass.preflight import plan_launch, resolve_credential
from modelpass.runtimes import Runtime
from modelpass.secrets import SECRETS_VERSION, SecretStore
from modelpass.store import ConnectionStore
from modelpass.types import AuthMode

KEY = "sk-not-a-real-key-0123456789-abcdefghij"
OTHER_KEY = "sk-also-not-a-real-key-9876543210"

POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt", reason="POSIX file modes; Windows has its own ACL path"
)
WINDOWS_SHAPED = pytest.mark.skipif(
    os.name != "nt", reason="the icacls path only runs on Windows"
)


def api_connection(
    name: str = "claude-api",
    *,
    credential_ref: str = "secret:claude-api",
    runtime: Runtime = Runtime.ANTHROPIC_API,
) -> Connection:
    return Connection(
        name=name,
        runtime=runtime,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(credential_ref),
    )


@pytest.fixture
def secrets(tmp_path) -> SecretStore:
    return SecretStore(tmp_path / "modelpass-home")


@pytest.fixture
def home(tmp_path):
    """A store and a secrets file that are siblings, as they are on disk."""
    root = tmp_path / "modelpass-home"
    return ConnectionStore(root), SecretStore(root)


# --- 1. the reference ------------------------------------------------------------


def test_the_secret_kind_and_its_wire_form():
    assert CredentialKind.SECRET.value == "secret"
    ref = CredentialRef.parse("secret:vendor-key")
    assert ref.kind is CredentialKind.SECRET
    assert ref.locator == "vendor-key"
    assert ref.to_str() == "secret:vendor-key"


def test_the_reference_describes_the_file_and_the_entry():
    ref = CredentialRef.parse("secret:vendor-key")
    assert ref.describe() == "the modelpass secrets file (entry vendor-key)"
    assert ref.to_dict()["describe"] == ref.describe()


@pytest.mark.parametrize(
    "entry", ["ok", "a-b_c.d", "A1", "x" * 64]
)
def test_entry_names_follow_the_connection_name_rules(entry):
    assert CredentialRef.parse(f"secret:{entry}").locator == entry


@pytest.mark.parametrize("entry", ["-leading", "with space", "x" * 65, "sl/ash"])
def test_an_entry_name_outside_those_rules_is_refused(entry):
    with pytest.raises(InvalidConnection):
        CredentialRef.parse(f"secret:{entry}")


def test_a_literal_key_in_a_credential_ref_is_still_refused():
    """``_reject_secret`` runs before the entry-name rule and keeps its job."""
    with pytest.raises(CredentialRefIsSecret):
        CredentialRef.parse(f"secret:{KEY}")
    with pytest.raises(CredentialRefIsSecret):
        CredentialRef.parse("secret:sk-ant-api03-aaaa")


def test_a_secret_reference_is_refused_on_a_runtime_that_launches_a_child():
    """A child process can only be handed a credential through the environment.

    Accepting ``secret:`` on an agent runtime would write a connection whose
    child launches with no key and silently falls back to the vendor login --
    the exact D2 failure -- so it is refused where it is written, not where it
    is billed.
    """
    with pytest.raises(InvalidConnection, match="only supported by the API runtimes"):
        api_connection(runtime=Runtime.ANTHROPIC_SDK)


def test_a_secret_reference_round_trips_through_the_connection_store(home):
    store, _ = home
    store.add(api_connection())
    assert store.get("claude-api").credential_ref.to_str() == "secret:claude-api"
    # And the shareable file still holds only the pointer.
    assert KEY not in store.path.read_text(encoding="utf-8")


# --- 2. the store ----------------------------------------------------------------


def test_a_missing_file_is_no_secrets_rather_than_an_error(secrets):
    assert secrets.exists() is False
    assert secrets.entries() == ()
    assert secrets.orphans(()) == ()
    with pytest.raises(NoSuchSecret):
        secrets.get("nothing")


def test_set_get_entries_and_remove(secrets):
    secrets.set("one", KEY)
    secrets.set("two", OTHER_KEY)
    assert secrets.entries() == ("one", "two")
    assert secrets.get("one") == KEY
    assert secrets.get("two") == OTHER_KEY
    secrets.remove("one")
    assert secrets.entries() == ("two",)


def test_the_file_shape_is_a_version_and_a_table_per_entry(secrets):
    secrets.set("one", KEY)
    text = secrets.path.read_text(encoding="utf-8")
    assert f"version = {SECRETS_VERSION}" in text
    assert "[secrets.one]" in text
    assert "apiKey" in text
    assert secrets.path.name == "secrets.toml"
    assert secrets.path.parent == secrets.root


def test_the_file_sits_beside_the_connection_file(home):
    store, secrets = home
    assert secrets.path.parent == store.path.parent


def test_setting_an_empty_value_is_refused(secrets):
    with pytest.raises(ConfigError, match="nothing to store"):
        secrets.set("one", "   ")


def test_an_entry_that_holds_no_value_reads_as_absent(secrets):
    secrets.set("one", KEY)
    secrets.path.write_text(
        'version = 1\n[secrets.one]\napiKey = ""\n', encoding="utf-8"
    )
    with pytest.raises(NoSuchSecret):
        secrets.get("one")


def test_no_exception_message_ever_carries_a_value(secrets):
    secrets.set("one", KEY)
    for exc_type, call in (
        (NoSuchSecret, lambda: secrets.get("absent")),
        (NoSuchSecret, lambda: secrets.remove("absent")),
        (ConfigError, lambda: secrets.set("two", "")),
    ):
        with pytest.raises(exc_type) as caught:
            call()
        assert KEY not in str(caught.value)


def test_a_pending_secret_redacts_itself(secrets):
    """The one object in modelpass that holds a pasted key, and it gets printed.

    ``ConnectionPlan`` carries a :class:`PendingSecret` between "show the
    receipt" and "write it", and a caller debugging a Settings page will print
    that plan. The dataclass-generated repr would put the key in their log
    (ticket 1.4).
    """
    from modelpass.secrets import PendingSecret

    pending = PendingSecret("claude-api", KEY)
    assert KEY not in repr(pending)
    assert "redacted" in repr(pending)
    assert pending.get("claude-api") == KEY


# --- 2b. compatibility, the ticket 0.1 policy applied to the second file ----------


def test_an_unknown_key_in_an_entry_is_carried_through(secrets):
    secrets.path.parent.mkdir(parents=True, exist_ok=True)
    secrets.path.write_text(
        'version = 1\n\n[secrets.one]\napiKey = "old"\norgId = "org_123"\n',
        encoding="utf-8",
    )
    secrets.set("one", KEY)
    text = secrets.path.read_text(encoding="utf-8")
    assert 'orgId = "org_123"' in text
    assert secrets.get("one") == KEY


def test_a_higher_file_version_reads_but_refuses_writes(secrets):
    secrets.path.parent.mkdir(parents=True, exist_ok=True)
    secrets.path.write_text(
        'version = 2\n\n[secrets.one]\napiKey = "future"\n', encoding="utf-8"
    )
    assert secrets.entries() == ("one",)
    assert secrets.get("one") == "future"
    with pytest.raises(ConfigError, match="newer build of modelpass is needed"):
        secrets.set("two", KEY)


def test_an_unreadable_file_reads_as_no_secrets_and_refuses_writes(secrets):
    secrets.path.parent.mkdir(parents=True, exist_ok=True)
    secrets.path.write_text("this is not = = toml\n", encoding="utf-8")
    assert secrets.entries() == ()
    with pytest.raises(NoSuchSecret):
        secrets.get("one")
    with pytest.raises(ConfigError, match="not valid TOML"):
        secrets.set("one", KEY)


# --- 3. permissions --------------------------------------------------------------


@POSIX_ONLY
def test_the_file_is_created_0600(secrets):
    secrets.set("one", KEY)
    mode = stat.S_IMODE(secrets.path.stat().st_mode)
    assert mode == 0o600
    assert secrets.permissions().group_or_other_readable is False
    assert "restricted to your account" in secrets.permissions().note


@POSIX_ONLY
def test_a_widened_mode_is_reported_and_not_refused(secrets):
    secrets.set("one", KEY)
    os.chmod(secrets.path, 0o644)
    permissions = secrets.permissions()
    assert permissions.group_or_other_readable is True
    assert "readable beyond your account" in permissions.note
    assert "chmod 600" in permissions.note
    # Reported, never refused: the value is still readable.
    assert secrets.get("one") == KEY


@WINDOWS_SHAPED
def test_the_windows_acl_attempt_is_recorded_when_it_succeeds(secrets):
    completed = subprocess.CompletedProcess(["icacls"], 0, stdout="ok", stderr="")
    with mock.patch("modelpass.secrets.subprocess.run", return_value=completed) as run:
        secrets.set("one", KEY)
    assert run.call_count == 1
    argv = run.call_args.args[0]
    assert argv[0] == "icacls"
    assert "/inheritance:r" in argv and "/grant:r" in argv
    permissions = secrets.permissions()
    assert permissions.windows_acl == "restricted"
    assert permissions.note == "the secrets file is restricted to your account"


@WINDOWS_SHAPED
def test_a_failed_windows_acl_is_reported_honestly(secrets):
    completed = subprocess.CompletedProcess(
        ["icacls"], 5, stdout="", stderr="Access is denied."
    )
    with mock.patch("modelpass.secrets.subprocess.run", return_value=completed):
        secrets.set("one", KEY)
    permissions = secrets.permissions()
    assert permissions.windows_acl == "failed"
    note = permissions.note
    assert "could not restrict" in note
    assert "Access is denied." in note
    assert "the profile directory's own ACL is the floor" in note
    # Never claims what was not applied.
    assert "restricted to your account" not in note


@WINDOWS_SHAPED
def test_an_icacls_that_cannot_be_run_is_a_recorded_failure_not_a_crash(secrets):
    with mock.patch(
        "modelpass.secrets.subprocess.run", side_effect=OSError("not found")
    ):
        secrets.set("one", KEY)
    assert secrets.get("one") == KEY
    assert secrets.permissions().windows_acl == "failed"


@WINDOWS_SHAPED
def test_the_acl_is_attempted_once_at_creation_and_not_on_every_write(secrets):
    completed = subprocess.CompletedProcess(["icacls"], 0, stdout="", stderr="")
    with mock.patch("modelpass.secrets.subprocess.run", return_value=completed) as run:
        secrets.set("one", KEY)
        secrets.set("two", OTHER_KEY)
        secrets.remove("two")
    assert run.call_count == 1


def test_permissions_of_a_missing_file_claim_nothing(secrets):
    permissions = secrets.permissions()
    assert permissions.exists is False
    assert permissions.note.startswith("no secrets file yet")
    assert permissions.to_dict()["exists"] is False


# --- 4. resolution ---------------------------------------------------------------


def test_resolve_credential_reads_the_store_at_the_moment_of_use(home):
    store, secrets = home
    connection = api_connection()
    secrets.set("claude-api", KEY)
    assert resolve_credential(connection, {}, secrets=secrets) == KEY
    # Rotated behind the library's back, and the next call sees the new one.
    secrets.set("claude-api", OTHER_KEY)
    assert resolve_credential(connection, {}, secrets=secrets) == OTHER_KEY
    assert store.path.exists() is False


def test_a_missing_entry_is_a_preflight_failure_naming_the_entry(home):
    _, secrets = home
    with pytest.raises(PreflightFailed, match="claude-api") as caught:
        resolve_credential(api_connection(), {}, secrets=secrets)
    assert KEY not in str(caught.value)


def test_the_launch_plan_for_a_secret_connection_carries_no_value(home):
    _, secrets = home
    secrets.set("claude-api", KEY)
    plan = plan_launch(api_connection(), {})
    assert plan.env == {}
    assert KEY not in repr(plan)


def test_the_receipt_names_the_secrets_file_and_never_the_key(home):
    from modelpass.preflight import api_preflight, credential_fingerprint

    _, secrets = home
    secrets.set("claude-api", KEY)
    connection = api_connection()
    receipt = api_preflight(
        connection, plan_launch(connection, {}), {}, secrets=secrets
    )
    assert receipt.ok is True
    assert receipt.credential_source == "the modelpass secrets file (entry claude-api)"
    assert receipt.account == credential_fingerprint(KEY)
    assert KEY not in receipt.summary()
    assert KEY not in str(receipt.to_dict())


def test_the_bridge_resolves_secrets_beside_its_own_store(tmp_path):
    root = tmp_path / "modelpass-home"
    bridge = Bridge(store=ConnectionStore(root))
    assert bridge.secrets.root == root
    bridge.secrets.set("claude-api", KEY)
    assert resolve_credential(api_connection(), {}, secrets=bridge.secrets) == KEY


# --- 5. referencing, orphans, delete and rename rules ----------------------------


def test_referenced_by_names_the_connections_pointing_at_an_entry(secrets):
    connections = [
        api_connection("one", credential_ref="secret:shared"),
        api_connection("two", credential_ref="secret:shared"),
        api_connection("three", credential_ref="env:VENDOR_KEY"),
    ]
    assert secrets.referenced_by("shared", connections) == ("one", "two")
    assert secrets.referenced_by("other", connections) == ()


def test_orphans_are_reported_and_never_collected(secrets):
    secrets.set("claude-api", KEY)
    secrets.set("left-over", OTHER_KEY)
    connections = [api_connection()]
    assert secrets.orphans(connections) == ("left-over",)
    # Asking twice changes nothing: reporting is all it does.
    assert secrets.entries() == ("claude-api", "left-over")


def test_removing_a_referenced_secret_is_refused_naming_the_connections(secrets):
    secrets.set("shared", KEY)
    connections = [
        api_connection("one", credential_ref="secret:shared"),
        api_connection("two", credential_ref="secret:shared"),
    ]
    with pytest.raises(SecretStillReferenced) as caught:
        secrets.remove("shared", connections=connections)
    assert "'one'" in str(caught.value) and "'two'" in str(caught.value)
    assert secrets.entries() == ("shared",)


def test_deleting_a_connection_removes_its_own_unshared_entry(secrets):
    secrets.set("claude-api", KEY)
    note = secrets.forget_for_connection(api_connection(), [])
    assert note == "removed secret entry 'claude-api'"
    assert secrets.entries() == ()


def test_deleting_a_connection_leaves_an_entry_another_connection_shares(secrets):
    secrets.set("claude-api", KEY)
    other = api_connection("second", credential_ref="secret:claude-api")
    note = secrets.forget_for_connection(api_connection(), [other])
    assert "left secret entry 'claude-api' in place" in note
    assert "'second'" in note
    assert secrets.entries() == ("claude-api",)


def test_deleting_a_connection_leaves_an_entry_that_is_not_its_own_name(secrets):
    secrets.set("shared", KEY)
    connection = api_connection("claude-api", credential_ref="secret:shared")
    note = secrets.forget_for_connection(connection, [])
    assert "left secret entry 'shared' in place" in note
    assert "shared locator" in note
    assert secrets.entries() == ("shared",)


def test_deleting_a_connection_with_no_secret_says_nothing(secrets):
    connection = api_connection(credential_ref="env:VENDOR_KEY")
    assert secrets.forget_for_connection(connection, []) == ""


def test_rename_moves_an_entry_and_leaves_the_value_untouched(secrets):
    secrets.set("old", KEY)
    secrets.rename("old", "new")
    assert secrets.entries() == ("new",)
    assert secrets.get("new") == KEY


def test_rename_onto_an_existing_entry_is_refused(secrets):
    secrets.set("old", KEY)
    secrets.set("new", OTHER_KEY)
    with pytest.raises(ConfigError, match="already exists"):
        secrets.rename("old", "new")


# --- 6. the command line ---------------------------------------------------------


class _Stdin:
    """A stdin stand-in that can claim to be, or not to be, a terminal."""

    def __init__(self, text: str, *, tty: bool = False) -> None:
        self._text = text
        self._tty = tty

    def read(self) -> str:
        return self._text

    def isatty(self) -> bool:
        return self._tty


def cli_bridge(tmp_path) -> Bridge:
    return Bridge(store=ConnectionStore(tmp_path / "modelpass-home"))


def test_no_cli_option_anywhere_accepts_a_key_value_inline():
    """Argv never carries a key -- asserted over the whole parser, not one flag.

    A command line is visible in process listings and lands in shell history, so
    the guarantee has to be about the *surface*, not about the one option this
    ticket happened to add.
    """
    from modelpass.cli import build_parser

    parser = build_parser("modelpass")
    subparsers = [
        action
        for action in parser._actions
        if hasattr(action, "choices") and isinstance(action.choices, dict)
    ]
    seen = 0
    for action in subparsers:
        for sub in action.choices.values():
            for option in sub._actions:
                if not option.option_strings:
                    continue
                seen += 1
                takes_a_value = option.nargs is None or option.nargs != 0
                names = " ".join(option.option_strings)
                if "key" in names and "env" not in names:
                    assert not takes_a_value, (
                        f"{names} would take a key on the command line"
                    )
    assert seen > 0


def test_connect_with_api_key_stdin_writes_the_secret_then_the_connection(
    tmp_path, capsys
):
    bridge = cli_bridge(tmp_path)
    code = main(
        ["connect", "anthropic", "--api-key-stdin", "-y"],
        bridge=bridge,
        stdin=_Stdin(KEY + "\n"),
    )
    assert code == 0
    assert bridge.secrets.get("claude-api") == KEY
    connection = bridge.store.get("claude-api")
    assert connection.runtime is Runtime.ANTHROPIC_API
    assert connection.auth_mode is AuthMode.API_KEY
    assert connection.credential_ref.to_str() == "secret:claude-api"
    out = capsys.readouterr().out
    assert KEY not in out
    assert "the modelpass secrets file (entry claude-api)" in out


def test_connect_writes_nothing_at_all_when_the_receipt_is_declined(tmp_path):
    bridge = cli_bridge(tmp_path)
    code = main(
        ["connect", "anthropic", "--api-key-stdin"],
        bridge=bridge,
        stdin=_Stdin(KEY + "\n"),
        confirm=lambda _q: False,
    )
    assert code == 1
    assert bridge.secrets.exists() is False
    assert bridge.store.names() == ()


def test_only_one_trailing_newline_is_stripped(tmp_path):
    bridge = cli_bridge(tmp_path)
    main(
        ["connect", "anthropic", "--api-key-stdin", "-y"],
        bridge=bridge,
        stdin=_Stdin(KEY + "\n\n"),
    )
    assert bridge.secrets.get("claude-api") == KEY + "\n"


def test_an_empty_stdin_is_refused(tmp_path, capsys):
    bridge = cli_bridge(tmp_path)
    code = main(
        ["connect", "anthropic", "--api-key-stdin", "-y"],
        bridge=bridge,
        stdin=_Stdin("\n"),
    )
    assert code == 1
    assert "empty value" in capsys.readouterr().err
    assert bridge.secrets.exists() is False


def test_a_terminal_stdin_is_refused_with_instructions(tmp_path, capsys):
    bridge = cli_bridge(tmp_path)
    code = main(
        ["connect", "anthropic", "--api-key-stdin", "-y"],
        bridge=bridge,
        stdin=_Stdin(KEY, tty=True),
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "will not read from a terminal" in err
    assert "--api-key-stdin" in err


def test_the_two_key_options_are_mutually_exclusive(tmp_path, capsys):
    bridge = cli_bridge(tmp_path)
    code = main(
        ["connect", "anthropic", "--api-key-stdin", "--api-key-env", "K", "-y"],
        bridge=bridge,
        stdin=_Stdin(KEY),
    )
    assert code == 1
    assert "pick one" in capsys.readouterr().err


def test_check_reports_the_permission_note_and_orphan_entries(tmp_path, capsys):
    bridge = cli_bridge(tmp_path)
    main(
        ["connect", "anthropic", "--api-key-stdin", "-y"],
        bridge=bridge,
        stdin=_Stdin(KEY),
    )
    bridge.secrets.set("left-over", OTHER_KEY)
    capsys.readouterr()
    main(["check"], bridge=bridge)
    out = capsys.readouterr().out
    assert "secrets" in out
    assert "left-over" in out
    assert "permissions" in out
    assert KEY not in out and OTHER_KEY not in out


def test_check_says_orphans_none_when_every_entry_is_referenced(tmp_path, capsys):
    bridge = cli_bridge(tmp_path)
    main(
        ["connect", "anthropic", "--api-key-stdin", "-y"],
        bridge=bridge,
        stdin=_Stdin(KEY),
    )
    capsys.readouterr()
    main(["check"], bridge=bridge)
    assert "orphans      none" in capsys.readouterr().out


def test_rename_moves_the_connection_and_its_own_secret(tmp_path, capsys):
    bridge = cli_bridge(tmp_path)
    main(
        ["connect", "anthropic", "--api-key-stdin", "-y"],
        bridge=bridge,
        stdin=_Stdin(KEY),
    )
    capsys.readouterr()
    assert main(["rename", "claude-api", "claude-work"], bridge=bridge) == 0
    out = capsys.readouterr().out
    assert "moved secret entry 'claude-api' to 'claude-work'" in out
    assert bridge.store.names() == ("claude-work",)
    assert bridge.store.get("claude-work").credential_ref.to_str() == (
        "secret:claude-work"
    )
    assert bridge.secrets.entries() == ("claude-work",)
    assert bridge.secrets.get("claude-work") == KEY
    assert KEY not in out


def test_rename_leaves_a_shared_entry_where_it_is_and_says_so(tmp_path, capsys):
    bridge = cli_bridge(tmp_path)
    bridge.secrets.set("shared", KEY)
    bridge.store.add(api_connection("one", credential_ref="secret:shared"))
    bridge.store.add(api_connection("two", credential_ref="secret:shared"))
    assert main(["rename", "one", "uno"], bridge=bridge) == 0
    out = capsys.readouterr().out
    assert "stays where it is" in out
    assert "'two'" in out
    assert bridge.secrets.entries() == ("shared",)
    assert bridge.store.get("uno").credential_ref.to_str() == "secret:shared"


def test_rename_onto_an_existing_connection_is_refused(tmp_path, capsys):
    bridge = cli_bridge(tmp_path)
    bridge.store.add(api_connection("one", credential_ref="env:VENDOR_KEY"))
    bridge.store.add(api_connection("two", credential_ref="env:VENDOR_KEY"))
    assert main(["rename", "one", "two"], bridge=bridge) == 1
    assert "already exists" in capsys.readouterr().err
    assert bridge.store.names() == ("one", "two")


def test_connect_says_plainly_that_nothing_can_drive_the_connection_yet(
    tmp_path, capsys, monkeypatch
):
    """An offline-OK receipt is not the same as a runnable connection.

    Every check ``api_preflight`` makes really did pass, so the status is OK --
    but a runtime whose adapter has not landed cannot run anything, and a setup
    tool that let a reader infer otherwise would be leaving them with a false
    picture of what they had just configured.

    **Every vendor alias now has an adapter** -- ``anthropic`` since ticket 1.6,
    ``openai`` since 1.9, ``compatible`` since 1.10 and ``google`` since 1.11 --
    so the temporary alias monkeypatch this test used to carry is gone with the
    runtime it stood in for. The line is still reachable, and the way a user
    reaches it now is the only way left: the adapter exists and its vendor
    package is not installed. ``is_available()`` says so, ``load_adapter``
    refuses, and setup says it plainly at the moment of setup rather than at the
    moment of the first failed run.
    """
    from modelpass.adapters.google_api import GoogleAPIAdapter

    monkeypatch.setattr(GoogleAPIAdapter, "is_available", classmethod(lambda cls: False))
    bridge = cli_bridge(tmp_path)
    main(
        ["connect", "google", "--api-key-stdin", "-y"],
        bridge=bridge,
        stdin=_Stdin(KEY),
    )
    out = capsys.readouterr().out
    assert "There is no adapter for runtime 'google-api'" in out
    assert "inspected and checked, and not yet run" in out


def test_connect_no_longer_says_it_on_the_second_runtime_with_an_adapter(
    tmp_path, capsys, monkeypatch
):
    """``openai-api`` left the driverless state in ticket 1.9.

    Availability is faked on, exactly as the case above fakes it off: the
    subject is what setup prints for a runtime that has an adapter, and a
    machine without the ``openai-api`` extra installed -- which is every
    machine CI runs on (D2) -- must read the same line.
    """
    from modelpass.adapters.openai_api import OpenAIAPIAdapter

    monkeypatch.setattr(OpenAIAPIAdapter, "is_available", classmethod(lambda cls: True))
    bridge = cli_bridge(tmp_path)
    main(
        ["connect", "openai", "--api-key-stdin", "-y"],
        bridge=bridge,
        stdin=_Stdin(KEY),
    )
    out = capsys.readouterr().out
    assert "There is no adapter" not in out
    assert "openai-api" in out


def test_connect_no_longer_says_it_on_the_runtime_that_has_an_adapter(
    tmp_path, capsys, monkeypatch
):
    from modelpass.adapters.anthropic_api import AnthropicAPIAdapter

    monkeypatch.setattr(AnthropicAPIAdapter, "is_available", classmethod(lambda cls: True))
    bridge = cli_bridge(tmp_path)
    main(
        ["connect", "anthropic", "--api-key-stdin", "-y"],
        bridge=bridge,
        stdin=_Stdin(KEY),
    )
    out = capsys.readouterr().out
    assert "There is no adapter" not in out
    assert "claude-api" in out
