"""The receipt joins the stream (first-consumer feedback, 2026-08-17).

``Bridge.chat`` has always run a preflight before handing back an iterator --
that is what makes "holding an iterator means the auth check passed" true. What
it did not do was hand the *evidence* over, so the first real consumer, which
logs which subscription paid for each sentence, ran the preflight a second time
itself: two round trips for one fact, and two chances for the answers to differ.

The rule these tests pin: **a ``receipt`` event always precedes any event
produced by the connection it describes.** One per connection that runs, first
in the stream, and the failover connection gets one on exactly the same terms
(see test_failover.py for that half).
"""

from __future__ import annotations

import json

import pytest

from modelpass.connections import Guards
from modelpass.errors import PreflightFailed
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter, quota_exhausted, usage
from modelpass.types import (
    AuthMode,
    ReceiptEvent,
    TerminalEvent,
    TextDeltaEvent,
    ThinkingEvent,
)

MESSAGE = "hi"


def test_the_receipt_is_the_first_event_of_every_stream(
    bridge_factory, subscription_connection
):
    script = [ThinkingEvent(text="hmm"), TextDeltaEvent(text="hello")]
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter(script))
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))

    assert isinstance(events[0], ReceiptEvent)
    assert sum(isinstance(e, ReceiptEvent) for e in events) == 1


def test_it_arrives_even_when_the_adapter_produces_nothing_at_all(
    bridge_factory, subscription_connection
):
    """An empty run still spent a preflight, and still names what would have paid."""
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([]))
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert isinstance(events[0], ReceiptEvent)
    assert isinstance(events[-1], TerminalEvent)


def test_it_carries_the_live_receipt_not_a_copy_of_some_of_it(
    bridge_factory, subscription_connection
):
    """The consumer's whole ask: `_receipt_fields(event.receipt)` with no second call."""
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter([]))
    event = next(iter(bridge.chat(connection="claude-sub", message=MESSAGE)))

    assert event.receipt.connection == "claude-sub"
    assert event.receipt.account == "fake-account"
    assert event.receipt.plan_name == "Fake Plan"
    assert event.receipt.effective_auth_mode is AuthMode.SUBSCRIPTION
    assert event.summary == event.receipt.summary()
    # Exactly one preflight ran for the whole call: the point of the change.
    assert len(fake.preflights) == 1


def test_the_stamp_fields_match_the_terminals(bridge_factory, subscription_connection):
    """Same three fields, same values, at both ends of the run."""
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([usage(input_tokens=5)]))
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    start, end = events[0], events[-1]
    assert (start.connection, start.runtime, start.auth_mode) == (
        end.connection,
        end.runtime,
        end.auth_mode,
    )


def test_the_stamp_follows_the_detected_mode(bridge_factory, api_connection):
    bridge, _ = bridge_factory(
        api_connection, FakeAdapter([]), env={"ANTHROPIC_API_KEY": "sk-not-a-real-key"}
    )
    event = next(iter(bridge.chat(connection="claude-api", message=MESSAGE)))
    assert event.auth_mode is AuthMode.API_KEY
    assert event.runtime is Runtime.ANTHROPIC_SDK


def test_a_refused_run_produces_no_receipt_event_because_it_produces_no_stream(
    bridge_factory, subscription_connection
):
    """Preflight failures still raise, before any event exists to carry evidence."""
    bridge, _ = bridge_factory(
        subscription_connection, FakeAdapter([], ok=False, problem="no login found")
    )
    with pytest.raises(PreflightFailed):
        bridge.chat(connection="claude-sub", message=MESSAGE)


def test_a_guard_stop_run_still_leads_with_its_receipt(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([usage(input_tokens=300)]))
    events = list(
        bridge.chat(connection="claude-sub", message=MESSAGE, guards=Guards(stop_at_tokens=10))
    )
    assert isinstance(events[0], ReceiptEvent)


def test_a_quota_stop_run_still_leads_with_its_receipt(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([quota_exhausted()]))
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert isinstance(events[0], ReceiptEvent)


def test_the_whole_stream_is_loggable_as_json(bridge_factory, subscription_connection):
    """D11 end to end: a consumer can write the run to a log without a codec."""
    bridge, _ = bridge_factory(
        subscription_connection, FakeAdapter([TextDeltaEvent(text="hi"), usage(output_tokens=2)])
    )
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    line = json.dumps([e.to_dict() for e in events])
    assert json.loads(line)[0]["type"] == "receipt"
    assert json.loads(line)[0]["receipt"]["connection"] == "claude-sub"
