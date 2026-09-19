"""The store's forward-compatibility policy (2026-09-13).

Every test here describes what a build of modelpass does with a connections file
some *other* build wrote. The policy itself is stated in ``store.py``'s header
comment and in the README's configuration reference; these are its teeth.
"""

from __future__ import annotations

import tomllib

import pytest

from modelpass.bridge import Bridge
from modelpass.errors import ConfigError, InvalidConnection
from modelpass.store import CONFIG_VERSION, ConnectionStore


def _write(store: ConnectionStore, text: str) -> None:
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(text, encoding="utf-8")


def _raw(store: ConnectionStore) -> dict:
    with store.path.open("rb") as handle:
        return tomllib.load(handle)


_ONE = """\
version = 1

[connections.claude-sub]
runtime = "anthropic-sdk"
authMode = "subscription"
credentialRef = "native-login"
"""


# --- 1. the version check is "newer than this build", not "different" -------------


def test_an_older_file_version_is_read_without_complaint(store):
    _write(store, _ONE.replace("version = 1", "version = 0"))
    assert store.names() == ("claude-sub",)
    assert store.compatibility().writable is True
    assert store.compatibility().notes() == ()


def test_a_file_with_no_version_is_read_as_this_build_s_version(store):
    _write(store, _ONE.replace("version = 1\n", ""))
    assert store.names() == ("claude-sub",)
    assert store.compatibility().file_version == CONFIG_VERSION


def test_a_newer_file_version_still_reads_everything_parseable(store):
    _write(store, _ONE.replace("version = 1", "version = 99"))
    assert store.names() == ("claude-sub",)
    assert store.get("claude-sub").model is None
    compatibility = store.compatibility()
    assert compatibility.file_version == 99
    assert compatibility.writable is False


def test_a_newer_file_version_refuses_writes_and_names_both_versions(store, api_connection):
    _write(store, _ONE.replace("version = 1", "version = 99"))
    with pytest.raises(ConfigError) as excinfo:
        store.add(api_connection)
    message = str(excinfo.value)
    assert "99" in message
    assert str(CONFIG_VERSION) in message
    assert "newer" in message
    # The refusal did not touch the file.
    assert _raw(store)["version"] == 99
    assert set(_raw(store)["connections"]) == {"claude-sub"}


def test_a_newer_file_version_never_stops_the_listing_from_speaking(store):
    _write(store, _ONE.replace("version = 1", "version = 99"))
    bridge = Bridge(store=store)
    assert [c.name for c in bridge.connections()] == ["claude-sub"]
    assert bridge.validate("claude-sub").connection == "claude-sub"


def test_a_non_integer_version_is_still_a_malformed_file(store):
    """Nothing can be concluded from it, including that it is safe to rewrite.

    Since ticket 1.4 the read side answers "no connections" instead of raising
    (R10: zero connections is a legitimate state, and a user whose file this
    build cannot parse still needs 'modelpass list' able to speak). The write
    side is unchanged and still refuses, naming the version.
    """
    _write(store, _ONE.replace("version = 1", 'version = "one"'))
    assert store.load() == {}
    assert "version" in (store.unreadable_reason() or "")
    assert any("version" in note for note in store.compatibility().notes())
    with pytest.raises(ConfigError, match="version"):
        store.save(())


# --- 2. unknown connection keys are preserved and reported ------------------------


_WITH_UNKNOWN_KEYS = """\
version = 1

[connections.claude-sub]
runtime = "anthropic-sdk"
authMode = "subscription"
credentialRef = "native-login"
retryPolicy = "never"
maxOutputTokens = 4096

[connections.claude-sub.verifiedIdentity]
email = "someone@example.com"
tenantId = "t-1"

[connections.claude-sub.guards]
warnAtTokens = 100
warnAtDollars = 5.0

[connections.claude-sub.guards.onQuotaExhausted]
action = "stop"
cooldownSeconds = 60
"""


def test_an_unknown_connection_key_is_not_an_error(store):
    _write(store, _WITH_UNKNOWN_KEYS)
    connection = store.get("claude-sub")
    assert connection.runtime.value == "anthropic-sdk"


def test_unknown_keys_survive_a_load_save_round_trip_verbatim(store, api_connection):
    _write(store, _WITH_UNKNOWN_KEYS)
    store.add(api_connection)  # rewrites the whole file
    table = _raw(store)["connections"]["claude-sub"]
    assert table["retryPolicy"] == "never"
    assert table["maxOutputTokens"] == 4096
    assert table["verifiedIdentity"]["tenantId"] == "t-1"
    assert table["guards"]["warnAtDollars"] == 5.0
    assert table["guards"]["onQuotaExhausted"]["cooldownSeconds"] == 60
    # and the keys this build does understand are still there
    assert table["guards"]["warnAtTokens"] == 100
    assert table["verifiedIdentity"]["email"] == "someone@example.com"
    assert set(_raw(store)["connections"]) == {"claude-sub", "claude-api"}


def test_an_unknown_key_in_a_table_this_build_would_omit_is_still_carried(store):
    _write(
        store,
        _ONE + '\n[connections.claude-sub.guards]\nwarnAtDollars = 5.0\n',
    )
    connection = store.get("claude-sub")
    assert connection.guards.configured is False
    store.save([connection])
    assert _raw(store)["connections"]["claude-sub"]["guards"]["warnAtDollars"] == 5.0


def test_unknown_keys_are_reported_by_path(store):
    _write(store, _WITH_UNKNOWN_KEYS)
    carried = store.compatibility().carried_keys["claude-sub"]
    assert carried == (
        "guards.onQuotaExhausted.cooldownSeconds",
        "guards.warnAtDollars",
        "maxOutputTokens",
        "retryPolicy",
        "verifiedIdentity.tenantId",
    )


def test_unknown_keys_are_reported_on_the_validation_report(store):
    _write(store, _WITH_UNKNOWN_KEYS)
    report = Bridge(store=store).validate("claude-sub")
    note = "; ".join(report.notes)
    assert "retryPolicy" in note
    assert "preserved" in note
    # A key a newer build wrote is not this connection's fault.
    assert report.ok is True


# --- 3. an unknown runtime or auth mode refuses that connection only ---------------


_WITH_UNKNOWN_RUNTIME = """\
version = 1

[connections.claude-sub]
runtime = "anthropic-sdk"
authMode = "subscription"
credentialRef = "native-login"

[connections.future]
runtime = "anthropic-batch"
authMode = "api_key"
credentialRef = "env:ANTHROPIC_API_KEY"

[connections.odd-mode]
runtime = "anthropic-sdk"
authMode = "prepaid"
credentialRef = "native-login"
"""


def test_an_unknown_runtime_refuses_that_connection_not_the_file(store):
    _write(store, _WITH_UNKNOWN_RUNTIME)
    assert store.names() == ("claude-sub",)
    refused = store.compatibility().refused
    assert set(refused) == {"future", "odd-mode"}
    assert "anthropic-batch" in refused["future"]
    assert "anthropic-sdk" in refused["future"]  # the known set is named
    assert "prepaid" in refused["odd-mode"]


def test_a_refused_connection_is_named_as_refused_not_as_missing(store):
    _write(store, _WITH_UNKNOWN_RUNTIME)
    with pytest.raises(InvalidConnection, match="anthropic-batch"):
        store.get("future")


def test_a_refused_connection_is_not_dropped_on_rewrite(store, api_connection):
    _write(store, _WITH_UNKNOWN_RUNTIME)
    store.add(api_connection)
    connections = _raw(store)["connections"]
    assert set(connections) == {"claude-sub", "future", "odd-mode", "claude-api"}
    assert connections["future"]["runtime"] == "anthropic-batch"
    assert connections["odd-mode"]["authMode"] == "prepaid"


def test_a_refused_connection_can_still_be_removed_by_name(store):
    _write(store, _WITH_UNKNOWN_RUNTIME)
    store.remove("future")
    assert set(_raw(store)["connections"]) == {"claude-sub", "odd-mode"}


def test_a_refused_connection_reports_through_the_bridge(store):
    _write(store, _WITH_UNKNOWN_RUNTIME)
    bridge = Bridge(store=store)
    assert [c.name for c in bridge.connections()] == ["claude-sub"]
    report = bridge.validate("future")
    assert report.ok is False
    assert "anthropic-batch" in "; ".join(report.problems)


def test_a_malformed_connection_that_is_not_an_unknown_value_still_refuses_the_file(store):
    """Only *unknown values* are quarantined. A typo is still worth failing on."""
    _write(
        store,
        _ONE + '\n[connections.broken]\nruntime = "anthropic-sdk"\n'
        'authMode = "subscription"\nenabled = "no"\n',
    )
    with pytest.raises(InvalidConnection, match="enabled"):
        store.load()
