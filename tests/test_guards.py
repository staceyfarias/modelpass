"""Guard behavior (D4) and the 2026-08-17 amendment that removed its defaults.

Two things are being pinned here. First, that modelpass ships **no** invented
threshold and makes that absence visible instead. Second, that the tracker can
fold in usage as it arrives -- which is what lets a guard stop a tool loop
mid-run rather than reporting on it afterwards -- without double-counting when a
runtime reports both per-turn usage and a final total.
"""

from __future__ import annotations

import pytest

from modelpass.connections import Connection, Guards, QuotaAction, QuotaPolicy
from modelpass.errors import SubpassError
from modelpass.guards import TOKENS_GUARD, GuardTracker
from modelpass.preflight import Receipt, plan_launch
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter, usage
from modelpass.types import (
    AuthMode,
    GuardStopEvent,
    GuardWarningEvent,
    TerminalStatus,
    TextDeltaEvent,
    TokenUsage,
    UsageEvent,
    UsageScope,
)

# --- no invented numbers (D4 amendment, 2026-08-17) ----------------------------


def test_guards_have_no_default_thresholds():
    guards = Guards()
    assert guards.warn_at_tokens is None
    assert guards.stop_at_tokens is None
    assert guards.configured is False


def test_an_unconfigured_guard_never_fires_however_much_is_spent():
    tracker = GuardTracker(Guards(), "claude-sub")
    assert tracker.observe(TokenUsage(input_tokens=50_000_000)) == []
    assert tracker.stopped is False


def test_a_quota_policy_alone_is_not_a_spend_guard():
    """Stopping when the allowance runs out is not a bound on a single run."""
    guards = Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api"))
    assert guards.configured is False


def test_the_receipt_says_when_a_connection_has_no_spend_guards(subscription_connection):
    bare = subscription_connection.with_guards(Guards())
    receipt = Receipt.from_plan(plan_launch(bare, {}))
    assert receipt.guards_configured is False
    assert "no spend guards configured" in receipt.summary()
    assert "no spend guards configured" in (receipt.guard_note or "")
    # It names where to decide, and pointedly does not suggest a number.
    assert "docs/guards.md" in (receipt.guard_note or "")


def test_the_receipt_is_quiet_when_guards_do_exist(subscription_connection):
    receipt = Receipt.from_plan(plan_launch(subscription_connection, {}))
    assert receipt.guards_configured is True
    assert receipt.guard_note is None
    assert "no spend guards" not in receipt.summary()


def test_an_adapter_filling_in_notes_cannot_lose_the_guard_line(subscription_connection):
    """The guard line is a field, not a note, precisely so this cannot happen."""
    bare = subscription_connection.with_guards(Guards())
    receipt = Receipt.from_plan(plan_launch(bare, {}), notes=("claude-agent-sdk 0.2.139",))
    assert receipt.notes == ("claude-agent-sdk 0.2.139",)
    assert receipt.guards_configured is False


# --- usage scope: folding in interim reports without double counting -----------


def test_deltas_accumulate():
    tracker = GuardTracker(Guards(warn_at_tokens=100), "c")
    tracker.observe(TokenUsage(input_tokens=60))
    events = tracker.observe(TokenUsage(input_tokens=60))
    assert tracker.total.total_tokens == 120
    assert [type(e) for e in events] == [GuardWarningEvent]


def test_a_run_total_replaces_rather_than_adding():
    """The vendor's final total must not be added on top of the turns it covers."""
    tracker = GuardTracker(Guards(stop_at_tokens=1_000), "c")
    tracker.observe(TokenUsage(input_tokens=100), UsageScope.DELTA)
    tracker.observe(TokenUsage(input_tokens=100), UsageScope.DELTA)
    tracker.observe(TokenUsage(input_tokens=200), UsageScope.RUN_TOTAL)
    assert tracker.total.total_tokens == 200
    assert tracker.stopped is False


def test_a_run_total_never_lowers_what_was_already_observed():
    """If a 'total' turns out to be per-turn, err toward over- not under-reporting."""
    tracker = GuardTracker(Guards(), "c")
    tracker.observe(TokenUsage(input_tokens=500), UsageScope.DELTA)
    tracker.observe(TokenUsage(input_tokens=50), UsageScope.RUN_TOTAL)
    assert tracker.total.input_tokens == 500


def test_each_guard_fires_at_most_once():
    tracker = GuardTracker(Guards(warn_at_tokens=10, stop_at_tokens=20), "c")
    first = tracker.observe(TokenUsage(input_tokens=25))
    second = tracker.observe(TokenUsage(input_tokens=25))
    assert [type(e) for e in first] == [GuardWarningEvent, GuardStopEvent]
    assert second == []
    assert first[0].guard == TOKENS_GUARD


# --- mid-run interruption, end to end through the bridge -----------------------


def test_an_interim_usage_report_stops_a_run_before_it_finishes(
    bridge_factory, subscription_connection
):
    """The Phase 6 gap: an uncapped tool loop must be stoppable *during* the loop."""
    script = [
        TextDeltaEvent(text="turn one"),
        UsageEvent(usage=TokenUsage(input_tokens=200)),  # interim, mid-loop
        TextDeltaEvent(text="turn two"),
        UsageEvent(usage=TokenUsage(input_tokens=200)),
        TextDeltaEvent(text="turn three"),
    ]
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter(script))
    events = list(bridge.chat(connection="claude-sub", message="hi"))

    # The threshold (250) is crossed by the second interim report, so the loop is
    # cut off there -- the third turn never runs. Before Phase 5 the only usage
    # report was the run's final one and this loop would have completed in full.
    text = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert text == "turn oneturn two"
    assert "turn three" not in text
    assert events[-1].status is TerminalStatus.GUARD_STOP
    assert fake.cancelled == 1


def test_a_final_run_total_does_not_double_count_the_turns_it_covers(
    bridge_factory, subscription_connection
):
    script = [
        UsageEvent(usage=TokenUsage(input_tokens=100)),
        UsageEvent(usage=TokenUsage(input_tokens=100)),
        UsageEvent(usage=TokenUsage(input_tokens=200), scope=UsageScope.RUN_TOTAL),
    ]
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter(script))
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    # 250 is the stop threshold; naive accumulation would reach 400 and stop.
    assert events[-1].status is TerminalStatus.OK
    assert events[-1].usage.total_tokens == 200


def test_the_running_total_reaches_the_caller_on_every_usage_event(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(
        subscription_connection,
        FakeAdapter([usage(input_tokens=10), usage(output_tokens=5)]),
    )
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    assert [e.cumulative.total_tokens for e in events if isinstance(e, UsageEvent)] == [10, 15]


def test_a_connection_with_no_guards_runs_to_completion(
    bridge_factory, subscription_connection
):
    bare = subscription_connection.with_guards(Guards())
    bridge, _ = bridge_factory(bare, FakeAdapter([usage(input_tokens=10_000_000)]))
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    assert not any(isinstance(e, (GuardWarningEvent, GuardStopEvent)) for e in events)
    assert events[-1].status is TerminalStatus.OK


def test_dollar_thresholds_are_not_in_the_v1_schema():
    """Decided, not deferred: tokens only (D4 amendment, 2026-08-17)."""
    fields = {f for f in Guards.__dataclass_fields__}
    assert not any(
        word in name.lower()
        for name in fields
        for word in ("usd", "dollar", "cost", "price", "spend", "credit")
    )
    with pytest.raises(TypeError):
        Guards(stop_at_usd=5)  # type: ignore[call-arg]


def test_a_connection_object_reports_its_own_guard_state():
    connection = Connection(
        name="c", runtime=Runtime.ANTHROPIC_SDK, auth_mode=AuthMode.SUBSCRIPTION
    )
    assert connection.guards.configured is False


# --- per-call ergonomics (first-consumer feedback, 2026-08-17) -------------------
#
# Two footguns reported from the first real integration, both hit by the same
# gesture: "run this one reading under a tighter ceiling".
#
#   guards = modelpass.Guards(stop_at_tokens=5_000)     # -> InvalidConnection
#
# It failed because the connection's warn was above the new stop, and it failed
# with an error naming the *store*, about a file nobody had edited. The consumer
# worked around it with min(DEFAULT_WARN, stop) at every call site, which is the
# library making its caller do arithmetic to express something obvious.


def test_lowering_a_stop_for_one_call_clamps_the_warn_instead_of_failing():
    """A warn beyond a stop is not a contradiction; it is unreachable."""
    guards = Guards(warn_at_tokens=40_000, stop_at_tokens=5_000)
    assert guards.stop_at_tokens == 5_000
    assert guards.warn_at_tokens == 5_000


def test_for_call_keeps_what_it_was_not_asked_about():
    configured = Guards(
        warn_at_tokens=40_000,
        stop_at_tokens=60_000,
        on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api"),
    )
    tightened = Guards.for_call(configured, stop_at_tokens=5_000)

    assert tightened.stop_at_tokens == 5_000
    assert tightened.warn_at_tokens == 5_000
    # Lowering a ceiling must never quietly disarm a configured failover.
    assert tightened.on_quota_exhausted == configured.on_quota_exhausted
    # And it changed nothing about the connection's own guards.
    assert configured.stop_at_tokens == 60_000


def test_for_call_leaves_an_unmentioned_threshold_alone():
    configured = Guards(warn_at_tokens=100, stop_at_tokens=250)
    assert Guards.for_call(configured).warn_at_tokens == 100
    assert Guards.for_call(configured, warn_at_tokens=50).stop_at_tokens == 250


def test_zero_means_no_guard_for_this_call():
    configured = Guards(warn_at_tokens=100, stop_at_tokens=250)
    off = Guards.for_call(configured, stop_at_tokens=0, warn_at_tokens=0)
    assert off.configured is False


def test_a_bad_per_call_guard_does_not_implicate_the_connection_store():
    """The second footgun: the error named a file that was fine."""
    from modelpass.errors import ConfigError, InvalidGuards

    with pytest.raises(InvalidGuards) as excinfo:
        Guards(stop_at_tokens=-1)
    assert not isinstance(excinfo.value, ConfigError)
    assert "cannot be negative" in str(excinfo.value)
    # Catchable both ways: as a bad argument, and as a modelpass error.
    with pytest.raises(ValueError):
        Guards(stop_at_tokens=-1)
    with pytest.raises(SubpassError):
        Guards(stop_at_tokens="lots")


def test_the_config_path_stays_strict():
    """A file saying two contradictory things is a typo, not something to clamp."""
    from modelpass.errors import InvalidConnection

    with pytest.raises(InvalidConnection, match="must not exceed"):
        Guards.for_config(warn_at_tokens=40_000, stop_at_tokens=5_000)
    # And a bad value read from a file is still a connection problem.
    with pytest.raises(InvalidConnection, match="cannot be negative"):
        Guards.for_config(stop_at_tokens=-1, where="connection 'claude-sub'")


# --- the chat() shorthands ------------------------------------------------------


def test_chat_takes_the_threshold_shorthand(bridge_factory, subscription_connection):
    """The gesture the consumer wanted: one keyword, no Guards object, no min()."""
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([usage(input_tokens=60)]))
    events = list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            stop_at_tokens=50,
        )
    )
    assert events[-1].status is TerminalStatus.GUARD_STOP
    assert [e.threshold for e in events if isinstance(e, GuardStopEvent)] == [50]
    # The connection's warn was 100 -- above the new stop, so clamped to it.
    assert [e.threshold for e in events if isinstance(e, GuardWarningEvent)] == [50]


def test_the_shorthand_never_touches_the_stored_connection(
    bridge_factory, subscription_connection, store
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([]))
    list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            stop_at_tokens=7,
        )
    )
    assert store.get("claude-sub").guards.stop_at_tokens == 250
    assert store.get("claude-sub").guards.warn_at_tokens == 100


def test_zero_through_the_shorthand_runs_with_no_guard(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([usage(input_tokens=900)]))
    events = list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            stop_at_tokens=0,
            warn_at_tokens=0,
        )
    )
    assert events[-1].status is TerminalStatus.OK


def test_passing_both_forms_is_refused_rather_than_resolved(
    bridge_factory, subscription_connection
):
    """A caller that passed both meant something; guessing which half to drop is
    how a ceiling silently goes missing."""
    from modelpass.errors import InvalidGuards

    bridge, fake = bridge_factory(subscription_connection, FakeAdapter([]))
    with pytest.raises(InvalidGuards, match="not both"):
        bridge.chat(
            connection="claude-sub",
            message="hi",
            guards=Guards(stop_at_tokens=10),
            stop_at_tokens=20,
        )
    assert fake.preflights == []
