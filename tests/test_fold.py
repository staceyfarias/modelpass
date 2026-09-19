"""The event fold, asked directly (ticket 1.13).

Every rule here was already covered end to end through ``bridge.chat`` and
``Session.send``, and those tests stay exactly as they were -- this file is the
unit-level statement of the same rules, now that they live in one object instead
of in two loops. It exists because :mod:`modelpass._fold` is what the sync and
the async faces *share*: a rule that moved into it silently is a rule two faces
would then get wrong together.
"""

from __future__ import annotations

import pytest

from modelpass._fold import RunFold, TurnFold, raise_for
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.errors import GuardStop, QuotaExhausted, VendorRunFailed
from modelpass.guards import GuardTracker
from modelpass.preflight import Receipt, plan_launch
from modelpass.runtimes import Runtime
from modelpass.testing import usage
from modelpass.types import (
    AuthMode,
    ReceiptEvent,
    Retryable,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    TokenUsage,
    UsageEvent,
    UsageScope,
    VendorEvent,
)

ENV = {"ANTHROPIC_API_KEY": "sk-test"}


def connection(**kwargs) -> Connection:
    fields = {
        "name": "claude-api",
        "runtime": Runtime.ANTHROPIC_SDK,
        "auth_mode": AuthMode.API_KEY,
        "credential_ref": CredentialRef.parse("env:ANTHROPIC_API_KEY"),
    }
    fields.update(kwargs)
    return Connection(**fields)


def receipt_for(conn: Connection) -> Receipt:
    return Receipt.from_plan(
        plan_launch(conn, ENV),
        detected_auth_mode=conn.auth_mode,
        credential_source=conn.credential_ref.describe(),
    )


def fold(*, guards: Guards | None = None, conn: Connection | None = None) -> RunFold:
    conn = conn or connection()
    effective = guards or Guards()
    return RunFold(
        connection=conn,
        receipt=receipt_for(conn),
        guards=effective,
        tracker=GuardTracker(effective, conn.name),
        deadline=None,
    )


def turn_fold(*, guards: Guards | None = None) -> TurnFold:
    conn = connection()
    effective = guards or Guards()
    return TurnFold(
        connection=conn,
        receipt=receipt_for(conn),
        guards=effective,
        tracker=GuardTracker(effective, conn.name),
        deadline=None,
        reason_for=lambda event: event.reason,
    )


# --- the shape of the stream ------------------------------------------------------


def test_a_fold_begins_with_exactly_one_receipt_event():
    events = fold().begin()
    assert len(events) == 1
    assert isinstance(events[0], ReceiptEvent)
    assert events[0].connection == "claude-api"


def test_an_ordinary_event_passes_through_untouched():
    delta = TextDeltaEvent(text="hello")
    assert fold().feed(delta) == [delta]


def test_usage_is_folded_and_the_cumulative_is_injected():
    run = fold()
    run.feed(usage(input_tokens=10))
    (event,) = run.feed(usage(output_tokens=4))
    assert isinstance(event, UsageEvent)
    assert event.cumulative == TokenUsage(input_tokens=10, output_tokens=4)
    assert run.tracker.total.total_tokens == 14


def test_a_run_total_scope_does_not_double_count_a_turn():
    run = turn_fold()
    run.feed(UsageEvent(usage=TokenUsage(input_tokens=10), scope=UsageScope.RUN_TOTAL))
    run.feed(UsageEvent(usage=TokenUsage(input_tokens=10), scope=UsageScope.RUN_TOTAL))
    assert run.turn_usage == TokenUsage(input_tokens=10)


def test_the_allowance_is_observed_on_the_way_past_and_never_intercepted():
    run = fold()
    event = VendorEvent(
        name="rate_limit",
        runtime=Runtime.ANTHROPIC_SDK,
        data={"utilization": 0.5},
    )
    assert run.feed(event) == [event]
    assert run.allowance == {"utilization": 0.5}


# --- the stops --------------------------------------------------------------------


def test_crossing_the_ceiling_stops_the_run_and_names_the_threshold():
    run = fold(guards=Guards(stop_at_tokens=10))
    run.feed(usage(input_tokens=4))
    assert not run.done
    events = run.feed(usage(input_tokens=20))
    # The usage event and its guard event still reach the caller; the stop is
    # the fold's own decision about what comes after them.
    assert any(isinstance(e, UsageEvent) for e in events)
    assert run.done
    terminal = run.finish()
    assert terminal.status is TerminalStatus.GUARD_STOP
    assert "stopAtTokens threshold 10" in (terminal.reason or "")


def test_a_session_ceiling_ends_the_session_and_says_so():
    run = turn_fold(guards=Guards(stop_at_tokens=5))
    run.feed(usage(input_tokens=50))
    assert run.done
    assert "for this session" in (run.session_stopped or "")


def test_a_vendor_failure_is_a_terminal_not_an_exception():
    run = fold()
    run.vendor_failed(VendorRunFailed("the model was rejected by the runtime"))
    terminal = run.finish()
    assert terminal.status is TerminalStatus.ERROR
    assert terminal.reason == "the model was rejected by the runtime"
    # Read off typed fields, never off the reason string: an error carrying no
    # code is reported as carrying none (R6).
    assert terminal.status_code is None


def test_anything_else_escaping_the_adapter_is_a_bug_report():
    error = fold().failed(RuntimeError("boom"))
    assert error is not None
    assert "anthropic-sdk" in str(error)
    assert "boom" in str(error)


def test_an_undecided_fold_finishes_ok():
    assert fold().finish().status is TerminalStatus.OK


# --- the stamp --------------------------------------------------------------------


def test_the_fold_stamps_the_connection_and_the_auth_mode_an_adapter_cannot_forge():
    run = fold()
    run.feed(
        TerminalEvent(
            status=TerminalStatus.OK,
            connection="lies",
            runtime=Runtime.OPENAI_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            reason=None,
        )
    )
    terminal = run.finish()
    assert terminal.connection == "claude-api"
    assert terminal.runtime is Runtime.ANTHROPIC_SDK
    assert terminal.auth_mode is AuthMode.API_KEY


def test_a_stateless_leg_and_a_session_turn_classify_retries_differently():
    conn = connection()
    stateless = RunFold(
        connection=conn,
        receipt=receipt_for(conn),
        guards=Guards(),
        tracker=GuardTracker(Guards(), conn.name),
        deadline=None,
    )
    assert stateless.stateless is True
    assert turn_fold().stateless is False


def test_a_retryable_verdict_is_computed_from_typed_fields():
    run = fold()
    run.feed(
        TerminalEvent(
            status=TerminalStatus.ERROR,
            connection="",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.API_KEY,
            reason="rate limited",
            status_code=429,
            retry_after=3.0,
        )
    )
    terminal = run.finish()
    assert isinstance(terminal.retryable, Retryable)
    assert terminal.status_code == 429
    assert terminal.retry_after == 3.0


# --- raise_on_stop ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (TerminalStatus.GUARD_STOP, GuardStop),
        (TerminalStatus.QUOTA_EXHAUSTED, QuotaExhausted),
    ],
)
def test_raise_for_turns_a_stop_into_the_promised_exception(status, error):
    run = fold(guards=Guards(stop_at_tokens=10))
    terminal = run.terminal(status, "stopped")
    with pytest.raises(error):
        raise_for(
            terminal,
            guards=run.guards,
            tracker=run.tracker,
            connection="claude-api",
            deadline=None,
            timeout_reason="the run timed out",
        )


def test_raise_for_returns_quietly_on_an_ok_run():
    run = fold()
    raise_for(
        run.finish(),
        guards=run.guards,
        tracker=run.tracker,
        connection="claude-api",
        deadline=None,
        timeout_reason="the run timed out",
    )
