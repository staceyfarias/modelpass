"""``onQuotaExhausted`` failover (D4, Phase 5).

Failover is the one place where a single ``chat()`` call touches two
connections, and it is the only path in modelpass that can move a run from a
subscription onto metered billing. Everything here exists to make that path
loud, bounded, and impossible to reach by accident:

* it happens only because somebody named a connection in config;
* the second connection's own preflight runs and its receipt reaches the caller
  as an event *before* it does any work;
* the terminal event names both connections, so the switch survives in whatever
  the caller logs;
* it never chains, never loops, and never contradicts an ``expect_auth_mode``
  assertion.
"""

from __future__ import annotations

import pytest

from modelpass.bridge import Bridge
from modelpass.capabilities import CapabilityRegistry, Support
from modelpass.connections import Connection, CredentialRef, Guards, QuotaAction, QuotaPolicy
from modelpass.errors import (
    AuthModeMismatch,
    CapabilityNotSupported,
    InvalidConnection,
    NoSuchConnection,
    QuotaExhausted,
)
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter, usage
from modelpass.tools import ToolDef
from modelpass.types import (
    AuthMode,
    FailoverEvent,
    ReceiptEvent,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    TokenUsage,
)

EXHAUSTED = TerminalEvent(
    status=TerminalStatus.QUOTA_EXHAUSTED,
    connection="",
    runtime=Runtime.ANTHROPIC_SDK,
    auth_mode=AuthMode.SUBSCRIPTION,
    reason="subscription allowance exhausted (seven_day)",
)

ENV = {"ANTHROPIC_API_KEY": "sk-not-a-real-key", "PATH": "/usr/bin"}


def failing_over_to(name: str) -> Guards:
    return Guards(
        warn_at_tokens=100,
        stop_at_tokens=250,
        on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, name),
    )


@pytest.fixture
def primary() -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        guards=failing_over_to("claude-api"),
    )


@pytest.fixture
def metered() -> Connection:
    return Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
        guards=Guards(stop_at_tokens=500),
    )


@pytest.fixture
def wired(store, primary, metered):
    """A bridge over two connections sharing one scripted adapter.

    One adapter instance serves both, which is realistic (both connections drive
    the same runtime) and lets a test script the exhausted run and the failover
    run as one sequence.
    """

    def make(script, *, second: Connection | None = None, env=None):
        target = second or metered
        store.add(primary if second is None else primary.with_guards(
            failing_over_to(target.name)
        ))
        store.add(target)
        fake = FakeAdapter(script)
        bridge = Bridge(
            store=store,
            registry=CapabilityRegistry(),
            adapters={Runtime.ANTHROPIC_SDK: fake, Runtime.OPENAI_SDK: fake},
            env=dict(ENV if env is None else env),
        )
        return bridge, fake

    return make


def run(bridge, **kwargs):
    return list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            **kwargs,
        )
    )


def script_two_runs(first, second):
    """Serve the first run's events, then the second's, from one fake adapter."""
    calls = iter([first, second])

    def script(request):
        return next(calls)

    return script


# --- the happy path -------------------------------------------------------------


def test_a_spent_allowance_moves_the_run_to_the_named_connection(wired):
    bridge, fake = wired(
        script_two_runs(
            [TextDeltaEvent(text="partial"), EXHAUSTED],
            [TextDeltaEvent(text="the answer"), usage(input_tokens=7)],
        )
    )
    events = run(bridge)

    switch = next(e for e in events if isinstance(e, FailoverEvent))
    assert switch.from_connection == "claude-sub"
    assert switch.to_connection == "claude-api"
    assert "".join(e.text for e in events if isinstance(e, TextDeltaEvent)) == (
        "partialthe answer"
    )
    assert [r.connection.name for r in fake.requests] == ["claude-sub", "claude-api"]


def test_the_stream_still_ends_with_exactly_one_terminal(wired):
    bridge, _ = wired(
        script_two_runs([EXHAUSTED], [TextDeltaEvent(text="ok")])
    )
    events = run(bridge)
    assert sum(isinstance(e, TerminalEvent) for e in events) == 1
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].status is TerminalStatus.OK


def test_the_terminal_makes_the_switch_auditable(wired):
    """D3: a run that started on a subscription and ended metered must say so."""
    bridge, _ = wired(script_two_runs([EXHAUSTED], [TextDeltaEvent(text="ok")]))
    terminal = run(bridge)[-1]
    assert terminal.connection == "claude-api"
    assert terminal.auth_mode is AuthMode.API_KEY
    assert terminal.failed_over_from == "claude-sub"


def test_the_second_connections_receipt_reaches_the_caller_before_it_runs(wired):
    """D4(d)'s guarantee, now carried by the same event type the primary uses.

    Until 2026-08-17 the receipt was a dict on the ``failover`` event. It is a
    ``ReceiptEvent`` now -- one shape for one fact -- emitted immediately after
    the failover announcement and still before the second connection does any
    work, which is the property that actually mattered.
    """
    bridge, fake = wired(script_two_runs([EXHAUSTED], [TextDeltaEvent(text="ok")]))
    events = run(bridge)

    index = next(i for i, e in enumerate(events) if isinstance(e, FailoverEvent))
    # Emitted before anything the second connection produced.
    assert not any(isinstance(e, TextDeltaEvent) for e in events[index:index + 1])
    switch_receipt = events[index + 1]
    assert isinstance(switch_receipt, ReceiptEvent)
    assert switch_receipt.connection == "claude-api"
    assert switch_receipt.auth_mode is AuthMode.API_KEY
    # And it carries the same evidence a user would see at setup time.
    assert switch_receipt.receipt.detected_auth_mode is AuthMode.API_KEY
    assert "ANTHROPIC_API_KEY" in switch_receipt.receipt.preserved
    assert switch_receipt.summary
    # Nothing the second connection produced came before it.
    assert not any(isinstance(e, TextDeltaEvent) for e in events[: index + 2])
    # The preflight really ran; it was not assumed to pass.
    assert [r.connection.name for r in fake.preflights] == ["claude-sub", "claude-api"]


def test_exactly_two_receipt_events_arrive_and_they_name_both_connections(wired):
    """One per connection that ran, in the order they ran. Never more, never fewer."""
    bridge, _ = wired(script_two_runs([EXHAUSTED], [TextDeltaEvent(text="ok")]))
    receipts = [e for e in run(bridge) if isinstance(e, ReceiptEvent)]
    assert [r.connection for r in receipts] == ["claude-sub", "claude-api"]
    assert [r.auth_mode for r in receipts] == [AuthMode.SUBSCRIPTION, AuthMode.API_KEY]


def test_a_refused_failover_produces_no_second_receipt(store, primary, metered):
    """The second connection never ran, so nothing may claim it did."""
    store.add(primary)
    store.add(metered)
    fake = FakeAdapter(script_two_runs([EXHAUSTED], []))
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: fake},
        env={},  # ANTHROPIC_API_KEY is not set, so the failover cannot proceed
    )
    events = run(bridge)
    assert [e.connection for e in events if isinstance(e, ReceiptEvent)] == ["claude-sub"]
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED


def test_the_failover_event_names_the_billing_change_explicitly(wired):
    bridge, _ = wired(script_two_runs([EXHAUSTED], []))
    switch = next(e for e in run(bridge) if isinstance(e, FailoverEvent))
    assert switch.from_auth_mode is AuthMode.SUBSCRIPTION
    assert switch.to_auth_mode is AuthMode.API_KEY
    assert switch.crosses_to_metered is True
    assert switch.reason == EXHAUSTED.reason


def test_a_subscription_to_subscription_failover_does_not_claim_a_billing_change(
    store, primary
):
    second = Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
    )
    store.add(primary.with_guards(failing_over_to("codex-sub")))
    store.add(second)
    fake = FakeAdapter(script_two_runs([EXHAUSTED], [TextDeltaEvent(text="ok")]))
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: fake, Runtime.OPENAI_SDK: fake},
        env={},
    )
    switch = next(e for e in run(bridge) if isinstance(e, FailoverEvent))
    assert switch.crosses_to_metered is False
    assert switch.to_runtime is Runtime.OPENAI_SDK


def test_what_the_first_connection_spent_is_not_lost(wired):
    """The terminal reports the second run; the failover event keeps the first."""
    bridge, _ = wired(
        script_two_runs(
            [usage(input_tokens=40), EXHAUSTED],
            [usage(input_tokens=9)],
        )
    )
    events = run(bridge)
    switch = next(e for e in events if isinstance(e, FailoverEvent))
    assert switch.usage == TokenUsage(input_tokens=40)
    assert events[-1].usage == TokenUsage(input_tokens=9)


def test_the_second_connection_gets_its_own_guards_not_the_firsts(wired):
    """A run that already spent one allowance must not arrive half-budgeted."""
    bridge, _ = wired(
        script_two_runs(
            [usage(input_tokens=200), EXHAUSTED],
            # 300 would trip claude-sub's 250 stop, but claude-api allows 500.
            [usage(input_tokens=300), TextDeltaEvent(text="finished")],
        )
    )
    events = run(bridge)
    assert events[-1].status is TerminalStatus.OK
    assert "finished" in "".join(e.text for e in events if isinstance(e, TextDeltaEvent))


def test_the_second_connections_own_guards_still_apply(wired):
    bridge, _ = wired(
        script_two_runs([EXHAUSTED], [usage(input_tokens=600), TextDeltaEvent(text="never")])
    )
    events = run(bridge)
    assert events[-1].status is TerminalStatus.GUARD_STOP
    assert events[-1].connection == "claude-api"
    assert "never" not in "".join(e.text for e in events if isinstance(e, TextDeltaEvent))


def test_the_messages_carry_across_unchanged(wired):
    bridge, fake = wired(script_two_runs([EXHAUSTED], []))
    run(bridge)
    assert fake.requests[0].messages == fake.requests[1].messages


# --- the refusals ---------------------------------------------------------------


def test_failover_never_chains(store, primary, metered):
    """Two hops is a spend path nobody drew on purpose."""
    chained = metered.with_guards(
        Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-sub"))
    )
    store.add(primary)
    store.add(chained)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter([EXHAUSTED])},
        env=dict(ENV),
    )
    with pytest.raises(InvalidConnection, match="never chains"):
        run(bridge)


def test_a_connection_cannot_fail_over_to_itself(store):
    connection = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        guards=failing_over_to("claude-sub"),
    )
    store.add(connection)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter([EXHAUSTED])},
        env={},
    )
    with pytest.raises(InvalidConnection, match="itself"):
        run(bridge)


def test_a_missing_failover_target_is_caught_before_the_first_run(store, primary):
    store.add(primary)
    fake = FakeAdapter([EXHAUSTED])
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: fake},
        env={},
    )
    with pytest.raises(NoSuchConnection):
        run(bridge)
    assert fake.requests == []


def test_a_failover_target_that_contradicts_an_assertion_is_refused(wired):
    """expect_auth_mode is about how *this call* may be billed, failover included."""
    bridge, fake = wired(script_two_runs([EXHAUSTED], []))
    with pytest.raises(AuthModeMismatch):
        run(bridge, expect_auth_mode=AuthMode.SUBSCRIPTION)
    assert fake.requests == []


def test_a_failover_target_whose_preflight_fails_reports_instead_of_running(store, primary):
    """The first run genuinely stopped on quota; that outcome is not thrown away."""
    second = Connection(
        name="codex-sub", runtime=Runtime.OPENAI_SDK, auth_mode=AuthMode.SUBSCRIPTION
    )
    store.add(primary.with_guards(failing_over_to("codex-sub")))
    store.add(second)
    first_adapter = FakeAdapter([EXHAUSTED])
    second_adapter = FakeAdapter(
        [TextDeltaEvent(text="never")],
        runtime=Runtime.OPENAI_SDK,
        ok=False,
        problem="codex is not logged in",
    )
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={
            Runtime.ANTHROPIC_SDK: first_adapter,
            Runtime.OPENAI_SDK: second_adapter,
        },
        env={},
    )
    events = run(bridge)

    assert not any(isinstance(e, FailoverEvent) for e in events)
    terminal = events[-1]
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED
    assert terminal.connection == "claude-sub"
    assert terminal.auth_mode is AuthMode.SUBSCRIPTION
    assert "could not run" in terminal.reason
    assert "codex is not logged in" in terminal.reason
    # The failover connection was checked and never launched.
    assert second_adapter.preflights != []
    assert second_adapter.requests == []


def test_a_failover_target_missing_its_credential_reports_instead_of_running(
    store, primary, metered
):
    store.add(primary)
    store.add(metered)
    fake = FakeAdapter(script_two_runs([EXHAUSTED], []))
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: fake},
        env={},  # ANTHROPIC_API_KEY is not set
    )
    terminal = run(bridge)[-1]
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED
    assert "ANTHROPIC_API_KEY" in terminal.reason


def test_a_failover_target_that_cannot_do_the_requested_tools_reports(
    store, primary, metered
):
    registry = CapabilityRegistry()
    registry.refine(Runtime.ANTHROPIC_SDK, {"tools_in_process": Support.SUPPORTED})
    store.add(primary)
    store.add(metered)
    fake = FakeAdapter(script_two_runs([EXHAUSTED], []))
    bridge = Bridge(
        store=store, registry=registry, adapters={Runtime.ANTHROPIC_SDK: fake}, env=dict(ENV)
    )
    # Support is withdrawn after the first gate passes, standing in for a
    # failover target on a runtime that cannot do what the caller asked for.
    stream = bridge.chat(
        connection="claude-sub",
        message="hi",
        tools=[ToolDef(name="t", description="d", handler=lambda a: "x")],
    )
    registry.refine(Runtime.ANTHROPIC_SDK, {"tools_in_process": Support.UNSUPPORTED})
    terminal = list(stream)[-1]
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED
    assert terminal.connection == "claude-sub"
    assert "tools_in_process" in terminal.reason


def test_an_ordinary_quota_stop_is_unaffected_without_a_failover(
    bridge_factory, subscription_connection
):
    """The default is still a clean stop, and always has been (D4)."""
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([EXHAUSTED]))
    events = run(bridge)
    assert not any(isinstance(e, FailoverEvent) for e in events)
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED
    assert events[-1].failed_over_from is None


def test_a_guard_stop_does_not_trigger_failover(wired):
    """Failover answers "the plan ran out", not "you told me to stop"."""
    bridge, fake = wired(script_two_runs([usage(input_tokens=300)], []))
    events = run(bridge)
    assert events[-1].status is TerminalStatus.GUARD_STOP
    assert not any(isinstance(e, FailoverEvent) for e in events)
    assert [r.connection.name for r in fake.requests] == ["claude-sub"]


def test_an_error_does_not_trigger_failover(wired):
    failed = TerminalEvent(
        status=TerminalStatus.ERROR,
        connection="",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        reason="HTTP 500",
    )
    bridge, _ = wired(script_two_runs([failed], []))
    events = run(bridge)
    assert events[-1].status is TerminalStatus.ERROR
    assert not any(isinstance(e, FailoverEvent) for e in events)


# --- raise_on_stop --------------------------------------------------------------


def test_raise_on_stop_does_not_fire_when_the_failover_succeeded(wired):
    bridge, _ = wired(script_two_runs([EXHAUSTED], [TextDeltaEvent(text="ok")]))
    events = run(bridge, raise_on_stop=True)
    assert events[-1].status is TerminalStatus.OK


def test_raise_on_stop_names_the_connection_that_actually_ran_out(wired):
    bridge, _ = wired(script_two_runs([EXHAUSTED], [EXHAUSTED]))
    with pytest.raises(QuotaExhausted) as excinfo:
        run(bridge, raise_on_stop=True)
    assert excinfo.value.connection == "claude-api"


def test_the_shipped_test_helper_triggers_the_same_path(wired):
    """Downstream tools test their own failover handling with this, not a hand roll."""
    from modelpass.testing import quota_exhausted

    bridge, _ = wired(script_two_runs([quota_exhausted()], [TextDeltaEvent(text="ok")]))
    events = run(bridge)
    assert any(isinstance(e, FailoverEvent) for e in events)
    assert events[-1].status is TerminalStatus.OK


def test_capability_gating_still_applies_to_the_failover_target(store, primary):
    second = Connection(
        name="codex-sub", runtime=Runtime.OPENAI_SDK, auth_mode=AuthMode.SUBSCRIPTION
    )
    registry = CapabilityRegistry()
    registry.refine(Runtime.OPENAI_SDK, {"chat": Support.UNSUPPORTED})
    store.add(primary.with_guards(failing_over_to("codex-sub")))
    store.add(second)
    bridge = Bridge(
        store=store,
        registry=registry,
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter([EXHAUSTED])},
        env={},
    )
    with pytest.raises(CapabilityNotSupported):
        run(bridge)
