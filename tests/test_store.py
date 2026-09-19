from __future__ import annotations

import tomllib

import pytest

from modelpass.connections import (
    Account,
    AccountBinding,
    Connection,
    CredentialRef,
    Guards,
    QuotaAction,
    QuotaPolicy,
)
from modelpass.errors import (
    ConfigError,
    CredentialRefIsSecret,
    DuplicateConnection,
    InvalidConnection,
    NoSuchConnection,
    RuntimeGated,
)
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore
from modelpass.types import AuthMode


def test_a_fresh_install_has_no_connections(store):
    assert store.exists() is False
    assert store.list() == ()
    assert store.names() == ()
    with pytest.raises(NoSuchConnection) as excinfo:
        store.get("claude-sub")
    assert "no connections are configured" in str(excinfo.value)


def test_round_trip_preserves_every_field(store, subscription_connection):
    store.add(subscription_connection)
    assert store.get("claude-sub") == subscription_connection


def test_round_trip_of_a_fully_populated_connection(store):
    connection = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("keychain:modelpass/anthropic"),
        guards=Guards(
            warn_at_tokens=1,
            stop_at_tokens=2,
            on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "other"),
        ),
        model="some-model",
        description="metered fallback",
        config_dir=str(store.root / "claude-work"),
    )
    store.add(connection)
    assert store.get("claude-api") == connection


def test_config_dir_is_a_first_class_connection_field(store, tmp_path):
    root = tmp_path / "claude-work"
    connection = Connection(
        name="work",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        config_dir=str(root),
    )
    store.add(connection)
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert raw["connections"]["work"]["configDir"] == str(root)
    assert store.get("work").config_dir == str(root)


def test_account_nickname_round_trips_without_changing_the_stable_id(store):
    account = Account(
        name="anthropic-home",
        nickname="Anthropic Home",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
    )
    store.add(account)
    loaded = store.get("anthropic-home")
    assert loaded.nickname == "Anthropic Home"
    assert loaded.display_name == "Anthropic Home"
    assert loaded.vendor == "anthropic"
    assert loaded.name == "anthropic-home"
    assert isinstance(loaded, Connection)


def test_verified_account_identity_round_trips_as_non_secret_metadata(store):
    account = Account(
        name="anthropic-work",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        account_binding=AccountBinding(
            email="Work@Example.com", organization_id="org-work"
        ),
    )
    store.add(account)
    loaded = store.get(account.name)
    assert loaded.account_binding == AccountBinding(
        email="work@example.com", organization_id="org-work"
    )
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert raw["connections"][account.name]["verifiedIdentity"] == {
        "email": "work@example.com",
        "organizationId": "org-work",
    }


def test_config_dir_refuses_relative_or_unsupported_runtime_paths():
    with pytest.raises(InvalidConnection, match="absolute path"):
        Connection(
            name="relative",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            config_dir=".claude-work",
        )
    with pytest.raises(InvalidConnection, match="only supported"):
        Connection(
            name="google",
            runtime=Runtime.GOOGLE_CLI,
            auth_mode=AuthMode.SUBSCRIPTION,
            config_dir="/tmp/google",
            experimental=True,
        )


def test_a_connection_with_no_guards_writes_no_guards_table(store, subscription_connection):
    """There are no default thresholds, so 'nothing configured' is written as nothing.

    Emitting ``warnAtTokens = 0`` would read as a guard the user deliberately
    switched off, which is a different (and untrue) statement (D4 amendment).
    """
    connection = subscription_connection.with_guards(Guards.disabled())
    store.add(connection)
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert "guards" not in raw["connections"]["claude-sub"]
    assert store.get("claude-sub").guards == Guards()
    assert store.get("claude-sub").guards.configured is False


def test_zero_in_a_hand_written_file_still_means_disabled(store):
    """Existing files that spell out '0' keep working and keep meaning 'off'."""
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        "version = 1\n\n[connections.x]\nruntime = 'anthropic-sdk'\n"
        "authMode = 'subscription'\n\n[connections.x.guards]\n"
        "warnAtTokens = 0\nstopAtTokens = 0\n",
        encoding="utf-8",
    )
    assert store.get("x").guards.configured is False


def test_configured_guards_are_written_and_round_trip(store, subscription_connection):
    store.add(subscription_connection)
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert raw["connections"]["claude-sub"]["guards"] == {
        "warnAtTokens": 100,
        "stopAtTokens": 250,
        "onQuotaExhausted": {"action": "stop"},
    }


def test_written_file_is_valid_toml_and_carries_a_version(store, subscription_connection):
    store.add(subscription_connection)
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["connections"]["claude-sub"]["runtime"] == "anthropic-sdk"
    assert raw["connections"]["claude-sub"]["authMode"] == "subscription"
    assert raw["connections"]["claude-sub"]["credentialRef"] == "native-login"


def test_the_file_never_contains_a_secret(store, api_connection):
    store.add(api_connection)
    text = store.path.read_text(encoding="utf-8")
    assert "env:ANTHROPIC_API_KEY" in text
    assert "sk-" not in text


def test_add_refuses_to_clobber_without_overwrite(store, subscription_connection):
    store.add(subscription_connection)
    with pytest.raises(DuplicateConnection):
        store.add(subscription_connection)
    store.add(subscription_connection.with_guards(Guards.disabled()), overwrite=True)
    assert store.get("claude-sub").guards == Guards.disabled()


def test_remove(store, subscription_connection):
    store.add(subscription_connection)
    store.remove("claude-sub")
    assert store.list() == ()
    with pytest.raises(NoSuchConnection):
        store.remove("claude-sub")


def test_a_newer_config_version_reads_but_refuses_writes(store, subscription_connection):
    """The compatibility policy, in one line. Its full teeth are in
    tests/test_store_compat.py."""
    store.add(subscription_connection)
    store.path.write_text(
        store.path.read_text(encoding="utf-8").replace("version = 1", "version = 99"),
        encoding="utf-8",
    )
    assert store.names() == ("claude-sub",)
    with pytest.raises(ConfigError, match="config version"):
        store.add(subscription_connection, overwrite=True)


def test_unknown_keys_are_carried_rather_than_refused(store):
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        "version = 1\n\n[connections.x]\nruntime = 'anthropic-sdk'\n"
        "authMode = 'subscription'\nsomethingNewer = 'a value'\n",
        encoding="utf-8",
    )
    assert store.names() == ("x",)
    assert store.compatibility().carried_keys["x"] == ("somethingNewer",)


def test_malformed_toml_is_reported_with_the_path(store):
    """Reported, and since ticket 1.4 reported through the read rather than at it.

    R10 makes zero connections a legitimate state, so a file this build cannot
    parse reads as zero connections and carries the reason; it is the *write*
    that refuses, because rewriting bytes nobody parsed would destroy them.
    """
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text("version = = 1\n", encoding="utf-8")
    assert store.load() == {}
    assert "not valid TOML" in (store.unreadable_reason() or "")
    assert str(store.path) in store.unreadable_reason()
    with pytest.raises(ConfigError, match="not valid TOML"):
        store.save(())


def test_modelpass_home_moves_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELPASS_HOME", str(tmp_path / "elsewhere"))
    assert ConnectionStore().path == tmp_path / "elsewhere" / "connections.toml"


def test_credential_ref_rejects_anything_that_looks_like_a_secret():
    with pytest.raises(CredentialRefIsSecret):
        CredentialRef.parse("env:sk-ant-api03-XXXX")
    with pytest.raises(CredentialRefIsSecret):
        CredentialRef.parse("keychain:not a pointer but a very long pasted secret value here")


def test_credential_ref_wire_forms():
    assert CredentialRef.parse("native-login") == CredentialRef.native_login()
    assert CredentialRef.parse("env:OPENAI_API_KEY").to_str() == "env:OPENAI_API_KEY"
    assert CredentialRef.parse("keychain:modelpass/anthropic").to_str() == (
        "keychain:modelpass/anthropic"
    )
    with pytest.raises(InvalidConnection, match="unknown credentialRef kind"):
        CredentialRef.parse("file:/etc/secret")


def test_api_key_mode_cannot_use_the_native_login():
    with pytest.raises(InvalidConnection, match="must point at a credential"):
        Connection(
            name="bad",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.API_KEY,
            credential_ref=CredentialRef.native_login(),
        )


def test_runtime_that_cannot_do_a_mode_is_refused():
    with pytest.raises(InvalidConnection, match="cannot run in auth mode"):
        Connection(
            name="gemini-sub",
            runtime=Runtime.GOOGLE_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            credential_ref=CredentialRef.parse("env:GEMINI_API_KEY"),
            experimental=True,
        )


def test_google_runtimes_are_gated_behind_an_explicit_flag():
    with pytest.raises(RuntimeGated):
        Connection(
            name="gemini",
            runtime=Runtime.GOOGLE_CLI,
            auth_mode=AuthMode.SUBSCRIPTION,
        )


def test_failover_must_name_a_connection():
    with pytest.raises(InvalidConnection, match="requires a 'failover'"):
        QuotaPolicy(action=QuotaAction.FAILOVER)


def test_a_config_file_naming_a_warn_above_its_stop_is_refused(store, tmp_path):
    """Replaces the old ``Guards(warn=10, stop=5)`` raise (2026-08-17).

    The strictness moved to where it belongs rather than being dropped: a *file*
    that says two contradictory things is a typo the user will otherwise trust,
    so reading it fails loudly and names the connection. The plain constructor
    clamps instead -- see test_guards.py, "per-call ergonomics".
    """
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        "version = 1\n"
        "[connections.claude-sub]\n"
        'runtime = "anthropic-sdk"\n'
        'authMode = "subscription"\n'
        "[connections.claude-sub.guards]\n"
        "warnAtTokens = 10\n"
        "stopAtTokens = 5\n",
        encoding="utf-8",
    )
    with pytest.raises(InvalidConnection, match="must not exceed"):
        store.load()


def test_writing_a_warn_above_a_stop_is_refused_too():
    with pytest.raises(InvalidConnection, match="must not exceed"):
        Guards.for_config(warn_at_tokens=10, stop_at_tokens=5)
