"""``enabled = false``: a connection taken out of service (2026-08-17).

The property under test is the split between *running* and *looking*. A disabled
connection is still listed, still preflighted, still reported on -- what it
refuses is spend. Anything that made it invisible would be the same mistake as
deleting it: the connection stops being something modelpass can say anything
about.
"""

from __future__ import annotations

import tomllib

import pytest

from modelpass.connections import Connection, Guards, QuotaAction, QuotaPolicy
from modelpass.errors import ConnectionDisabled, InvalidConnection
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode

MESSAGE = "hello"


def test_connections_are_enabled_by_default(subscription_connection):
    assert subscription_connection.enabled is True


def test_disabled_round_trips_through_the_store(store, subscription_connection):
    store.add(subscription_connection.with_enabled(False))
    assert store.get("claude-sub").enabled is False
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert raw["connections"]["claude-sub"]["enabled"] is False


def test_enabled_is_not_written_when_it_is_true(store, subscription_connection):
    """`enabled = true` on every connection is noise; only the exception is written."""
    store.add(subscription_connection)
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert "enabled" not in raw["connections"]["claude-sub"]


def test_a_non_boolean_enabled_is_refused_rather_than_coerced(store):
    """`enabled = "false"` is a truthy string; silently running it would be the bug."""
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        "version = 1\n\n[connections.x]\nruntime = 'anthropic-sdk'\n"
        "authMode = 'subscription'\nenabled = 'false'\n",
        encoding="utf-8",
    )
    with pytest.raises(InvalidConnection, match="enabled must be true or false"):
        store.load()


def test_the_constructor_refuses_a_non_boolean():
    with pytest.raises(InvalidConnection, match="enabled must be true or false"):
        Connection(
            name="x",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            enabled="yes",  # type: ignore[arg-type]
        )


def test_chat_refuses_a_disabled_connection(bridge_factory, subscription_connection):
    bridge, fake = bridge_factory(subscription_connection.with_enabled(False))
    with pytest.raises(ConnectionDisabled, match="claude-sub"):
        bridge.chat(connection="claude-sub", message=MESSAGE)
    # Refused before anything ran: no preflight, no adapter call, no spend.
    assert fake.requests == []
    assert fake.preflights == []


def test_a_disabled_connection_is_still_listed_and_still_preflights(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(subscription_connection.with_enabled(False))
    assert [c.name for c in bridge.connections()] == ["claude-sub"]
    receipt = bridge.preflight("claude-sub")
    assert receipt.ok is True


def test_a_disabled_failover_target_is_refused_before_the_first_run(store):
    from modelpass.bridge import Bridge
    from modelpass.capabilities import CapabilityRegistry

    primary = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        guards=Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api")),
    )
    target = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref="env:ANTHROPIC_API_KEY",
        enabled=False,
    )
    store.add(primary)
    store.add(target)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter()},
        env={"ANTHROPIC_API_KEY": "x"},
    )
    with pytest.raises(ConnectionDisabled, match="claude-api"):
        bridge.chat(connection="claude-sub", message=MESSAGE)


def test_disabling_the_failover_target_mid_run_stops_the_failover(store):
    """The primary run can take minutes; a stale read must not run a connection
    somebody switched off in the meantime."""
    from modelpass.bridge import Bridge
    from modelpass.capabilities import CapabilityRegistry
    from modelpass.testing import quota_exhausted
    from modelpass.types import TerminalEvent, TerminalStatus

    primary = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        guards=Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api")),
    )
    target = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref="env:ANTHROPIC_API_KEY",
    )
    store.add(primary)
    store.add(target)

    def script(request):
        # Somebody disables the metered fallback while the primary is running.
        store.add(target.with_enabled(False), overwrite=True)
        yield quota_exhausted()

    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter(script=script)},
        env={"ANTHROPIC_API_KEY": "x"},
    )
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    # Reported on the quota terminal, not raised: the first run genuinely did
    # stop cleanly, and the failover simply never happened.
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED
    assert terminal.connection == "claude-sub"
    assert "could not run" in (terminal.reason or "")
    assert "disabled" in (terminal.reason or "")


def test_summary_says_so(subscription_connection):
    assert "[disabled]" in subscription_connection.with_enabled(False).summary()
    assert "[disabled]" not in subscription_connection.summary()


def test_cli_list_marks_a_disabled_connection(store, subscription_connection):
    import io

    from modelpass.bridge import Bridge
    from modelpass.capabilities import CapabilityRegistry
    from modelpass.cli import main

    store.add(subscription_connection.with_enabled(False))
    bridge = Bridge(store=store, registry=CapabilityRegistry(), env={})
    out = io.StringIO()
    assert main(["list"], bridge=bridge, out=out) == 0
    assert "[disabled -- runs are refused]" in out.getvalue()
