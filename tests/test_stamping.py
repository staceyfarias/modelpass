"""End-to-end audit of D3's stamp: every run says which connection and which
auth mode it actually used.

D3 is in the *guaranteed* layer of D4 -- not configurable, not best-effort, part
of the product's identity. The individual behaviors are tested where they live;
this file exists to walk every way a run can end and assert the stamp survives
all of them, including the paths added in Phase 5. A guarantee with a hole in
one exit path is not a guarantee, and the exit paths are exactly where it would
be easiest to forget.

The ways a run can end, all covered below:

* ok, error
* guard_stop -- including a stop taken mid-run off an interim usage report
* quota_exhausted -- with failover configured, and without
* cancelled -- both a runtime-reported cancellation and a caller closing the
  stream
* an adapter attempting to forge the stamp
* a failover, where the answer is *two* connections and the terminal has to name
  both
"""

from __future__ import annotations

import pytest

from modelpass.bridge import Bridge
from modelpass.capabilities import CapabilityRegistry
from modelpass.connections import Connection, CredentialRef, Guards, QuotaAction, QuotaPolicy
from modelpass.errors import AuthModeMismatch
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter, usage
from modelpass.types import (
    AuthMode,
    FailoverEvent,
    ReceiptEvent,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    TokenUsage,
    UsageEvent,
    UsageScope,
)

MESSAGE = "hi"


def terminal_of(events) -> TerminalEvent:
    """The one terminal event, asserting there is exactly one."""
    terminals = [e for e in events if isinstance(e, TerminalEvent)]
    assert len(terminals) == 1, f"expected one terminal, got {len(terminals)}"
    return terminals[0]


def assert_stamped(terminal: TerminalEvent, connection: str, auth_mode: AuthMode) -> None:
    """The stamp, in full. Never empty, never defaulted, never the adapter's."""
    assert terminal.connection == connection
    assert terminal.auth_mode is auth_mode
    assert terminal.runtime in tuple(Runtime)
    assert terminal.connection != ""


def vendor_terminal(status: TerminalStatus, reason: str = "") -> TerminalEvent:
    """What an adapter emits: a status, with placeholder identity fields."""
    return TerminalEvent(
        status=status,
        connection="",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        reason=reason or None,
    )


# --- every terminal status, on a subscription connection -------------------------


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ([TextDeltaEvent(text="done")], TerminalStatus.OK),
        ([vendor_terminal(TerminalStatus.ERROR, "HTTP 500")], TerminalStatus.ERROR),
        ([usage(input_tokens=300)], TerminalStatus.GUARD_STOP),
        (
            [vendor_terminal(TerminalStatus.QUOTA_EXHAUSTED, "allowance gone")],
            TerminalStatus.QUOTA_EXHAUSTED,
        ),
        (
            [vendor_terminal(TerminalStatus.CANCELLED, "cancelled by caller")],
            TerminalStatus.CANCELLED,
        ),
    ],
    ids=["ok", "error", "guard_stop", "quota_exhausted", "cancelled"],
)
def test_the_stamp_holds_for_every_way_a_run_can_end(
    bridge_factory, subscription_connection, script, expected
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter(script))
    terminal = terminal_of(list(bridge.chat(connection="claude-sub", message=MESSAGE)))
    assert terminal.status is expected
    assert_stamped(terminal, "claude-sub", AuthMode.SUBSCRIPTION)
    assert terminal.failed_over_from is None


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ([TextDeltaEvent(text="done")], TerminalStatus.OK),
        ([usage(input_tokens=300)], TerminalStatus.GUARD_STOP),
        (
            [vendor_terminal(TerminalStatus.QUOTA_EXHAUSTED)],
            TerminalStatus.QUOTA_EXHAUSTED,
        ),
    ],
    ids=["ok", "guard_stop", "quota_exhausted"],
)
def test_the_stamp_holds_on_a_metered_connection_too(
    bridge_factory, api_connection, script, expected
):
    """The mode that costs money by the token is the one worth being sure about."""
    metered = api_connection.with_guards(Guards(stop_at_tokens=250))
    bridge, _ = bridge_factory(
        metered, FakeAdapter(script), env={"ANTHROPIC_API_KEY": "sk-not-a-real-key"}
    )
    terminal = terminal_of(list(bridge.chat(connection="claude-api", message=MESSAGE)))
    assert terminal.status is expected
    assert_stamped(terminal, "claude-api", AuthMode.API_KEY)


# --- the stamp reflects what happened, not what was declared ---------------------


def test_a_run_that_would_bill_differently_than_declared_never_reaches_a_stamp(
    bridge_factory, subscription_connection
):
    """The failure D3's stamp exists to make impossible: silent metered billing.

    Detection disagreeing with the declaration is refused before the run starts,
    so there is no terminal event at all -- there is nothing to stamp, because
    nothing ran.
    """
    fake = FakeAdapter([usage(input_tokens=300)], detected_auth_mode=AuthMode.API_KEY)
    bridge, _ = bridge_factory(subscription_connection, fake)
    with pytest.raises(AuthModeMismatch):
        list(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert fake.requests == []


def test_an_adapter_cannot_forge_the_stamp_on_any_status(
    bridge_factory, subscription_connection
):
    forged = TerminalEvent(
        status=TerminalStatus.QUOTA_EXHAUSTED,
        connection="somebody-elses-connection",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.API_KEY,
        reason="allowance gone",
    )
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([forged]))
    terminal = terminal_of(list(bridge.chat(connection="claude-sub", message=MESSAGE)))
    assert_stamped(terminal, "claude-sub", AuthMode.SUBSCRIPTION)
    assert terminal.runtime is Runtime.ANTHROPIC_SDK
    # The one thing the adapter *is* allowed to say survives.
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED
    assert terminal.reason == "allowance gone"


# --- guard stops, including the mid-run ones Phase 5 added -----------------------


def test_a_mid_run_guard_stop_is_stamped(bridge_factory, subscription_connection):
    script = [
        TextDeltaEvent(text="round one"),
        UsageEvent(usage=TokenUsage(input_tokens=300)),
        TextDeltaEvent(text="round two"),
    ]
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter(script))
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    terminal = terminal_of(events)
    assert terminal.status is TerminalStatus.GUARD_STOP
    assert_stamped(terminal, "claude-sub", AuthMode.SUBSCRIPTION)
    assert terminal.usage.total_tokens == 300
    assert fake.cancelled == 1


def test_a_stop_taken_off_a_run_total_report_is_stamped(
    bridge_factory, subscription_connection
):
    script = [UsageEvent(usage=TokenUsage(input_tokens=400), scope=UsageScope.RUN_TOTAL)]
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter(script))
    terminal = terminal_of(list(bridge.chat(connection="claude-sub", message=MESSAGE)))
    assert terminal.status is TerminalStatus.GUARD_STOP
    assert_stamped(terminal, "claude-sub", AuthMode.SUBSCRIPTION)


# --- cancellation ----------------------------------------------------------------


def test_a_runtime_reported_cancellation_is_stamped(bridge_factory, subscription_connection):
    script = [TextDeltaEvent(text="partial"), vendor_terminal(TerminalStatus.CANCELLED)]
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter(script))
    terminal = terminal_of(list(bridge.chat(connection="claude-sub", message=MESSAGE)))
    assert terminal.status is TerminalStatus.CANCELLED
    assert_stamped(terminal, "claude-sub", AuthMode.SUBSCRIPTION)


def test_closing_the_stream_emits_no_unstamped_terminal(
    bridge_factory, subscription_connection
):
    """A caller that walks away gets no terminal -- and crucially, not a bad one.

    The stream contract promises one stamped terminal on a stream that
    *completes*. Abandoning it is not completing it, so the honest outcome is
    nothing rather than a synthesized terminal claiming an ending nobody
    observed.
    """
    script = [TextDeltaEvent(text="a"), TextDeltaEvent(text="b"), TextDeltaEvent(text="c")]
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter(script))
    stream = bridge.chat(connection="claude-sub", message=MESSAGE)
    seen = [next(stream), next(stream)]
    stream.close()

    assert not any(isinstance(e, TerminalEvent) for e in seen)
    assert fake.cancelled == 1


# --- failover: the case where the honest answer is two connections ---------------


@pytest.fixture
def failover_bridge(store):
    """claude-sub (subscription) configured to fail over to claude-api (metered)."""
    primary = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        guards=Guards(
            stop_at_tokens=250,
            on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api"),
        ),
    )
    secondary = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
        guards=Guards(stop_at_tokens=250),
    )
    store.add(primary)
    store.add(secondary)

    def make(first, second):
        runs = iter([first, second])
        fake = FakeAdapter(lambda request: next(runs))
        return (
            Bridge(
                store=store,
                registry=CapabilityRegistry(),
                adapters={Runtime.ANTHROPIC_SDK: fake},
                env={"ANTHROPIC_API_KEY": "sk-not-a-real-key"},
            ),
            fake,
        )

    return make


def test_a_failed_over_run_names_both_connections(failover_bridge):
    bridge, _ = failover_bridge(
        [vendor_terminal(TerminalStatus.QUOTA_EXHAUSTED, "allowance gone")],
        [TextDeltaEvent(text="answered on the key")],
    )
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    terminal = terminal_of(events)

    # The connection that finished...
    assert_stamped(terminal, "claude-api", AuthMode.API_KEY)
    # ...and the one that started, without which a mixed-billing run would be
    # indistinguishable from one that was metered all along (D3).
    assert terminal.failed_over_from == "claude-sub"


def test_the_failover_event_itself_names_both_modes(failover_bridge):
    bridge, _ = failover_bridge(
        [vendor_terminal(TerminalStatus.QUOTA_EXHAUSTED)], [TextDeltaEvent(text="x")]
    )
    switch = next(
        e
        for e in bridge.chat(connection="claude-sub", message=MESSAGE)
        if isinstance(e, FailoverEvent)
    )
    assert switch.from_connection == "claude-sub"
    assert switch.from_auth_mode is AuthMode.SUBSCRIPTION
    assert switch.to_connection == "claude-api"
    assert switch.to_auth_mode is AuthMode.API_KEY
    assert switch.crosses_to_metered is True


def test_a_guard_stop_on_the_second_connection_is_stamped_to_the_second(failover_bridge):
    bridge, _ = failover_bridge(
        [vendor_terminal(TerminalStatus.QUOTA_EXHAUSTED)],
        [usage(input_tokens=300), TextDeltaEvent(text="never")],
    )
    terminal = terminal_of(list(bridge.chat(connection="claude-sub", message=MESSAGE)))
    assert terminal.status is TerminalStatus.GUARD_STOP
    assert_stamped(terminal, "claude-api", AuthMode.API_KEY)
    assert terminal.failed_over_from == "claude-sub"


def test_a_refused_failover_stays_stamped_to_the_first_connection(store):
    """Nothing was billed to the second connection, so nothing may name it."""
    primary = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        guards=Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api")),
    )
    secondary = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
    )
    store.add(primary)
    store.add(secondary)
    fake = FakeAdapter([vendor_terminal(TerminalStatus.QUOTA_EXHAUSTED)])
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: fake},
        env={},  # the key the failover connection names is not set
    )
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    terminal = terminal_of(events)

    assert not any(isinstance(e, FailoverEvent) for e in events)
    assert_stamped(terminal, "claude-sub", AuthMode.SUBSCRIPTION)
    assert terminal.failed_over_from is None
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED


# --- the stamp is serializable, because an audit trail has to be written down ----


def test_the_stamp_survives_serialization(bridge_factory, subscription_connection):
    """D11: the vocabulary is protocol. An audit that cannot be logged is no audit."""
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([usage(input_tokens=1)]))
    terminal = terminal_of(list(bridge.chat(connection="claude-sub", message=MESSAGE)))
    data = terminal.to_dict()
    assert data["type"] == "terminal"
    assert data["connection"] == "claude-sub"
    assert data["auth_mode"] == "subscription"
    assert data["runtime"] == "anthropic-sdk"


def test_a_failover_event_survives_serialization(failover_bridge):
    bridge, _ = failover_bridge(
        [vendor_terminal(TerminalStatus.QUOTA_EXHAUSTED)], [TextDeltaEvent(text="x")]
    )
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    switch = next(e for e in events if isinstance(e, FailoverEvent))
    data = switch.to_dict()
    assert data["type"] == "failover"
    assert data["from_connection"] == "claude-sub"
    assert data["to_auth_mode"] == "api_key"
    # The receipt is not on this event; it is the ReceiptEvent that follows it
    # (first-consumer feedback, 2026-08-17 -- one shape for one fact).
    assert "receipt" not in data
    after = events[events.index(switch) + 1]
    assert isinstance(after, ReceiptEvent)
    assert after.to_dict()["receipt"]["connection"] == "claude-api"
    assert terminal_of(events).to_dict()["failed_over_from"] == "claude-sub"
