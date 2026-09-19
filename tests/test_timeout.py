"""Per-call timeouts (R6, ticket 1.12), and the cancellation contract beside them.

The four consumers all built a wall-clock bound by hand before the library had
one -- a retrieval service and a desktop agent app with threading watchdogs, a
batch scraping tool with ``multiprocessing`` and ``join(timeout)``, a RAG
evaluation harness with a named 30s
constant on every API-key path -- and every one of them is a workaround for the
same absence. These tests are the bound arriving, plus the guarantee that
adding it changed nothing about cancelling by closing the iterator, which a
consumer's own cancellation contract depends on.
"""

from __future__ import annotations

import threading

import pytest

from modelpass.bridge import Bridge
from modelpass.connections import Connection, CredentialRef
from modelpass.errors import RunTimedOut
from modelpass.runtimes import Runtime
from modelpass.testing import StallingAdapter, fake_bridge, usage
from modelpass.types import (
    AuthMode,
    Retryable,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    Timeout,
)


@pytest.fixture
def connection() -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


def _bridge(tmp_path, connection, adapter):
    return fake_bridge(
        connections=[connection], adapter=adapter, home=tmp_path / "modelpass"
    )


# --- the Timeout value itself ---------------------------------------------------


def test_a_timeout_takes_seconds_or_nothing_and_refuses_an_expired_bound():
    assert Timeout(total=30).total == 30
    assert Timeout().first_token is None
    # Both axes unbounded is a Timeout that does nothing, and says so.
    assert not Timeout()
    assert Timeout(first_token=1)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="greater than zero"):
            Timeout(total=bad)
    with pytest.raises(ValueError, match="number of seconds"):
        Timeout(total="30")  # type: ignore[arg-type]


def test_a_bare_number_means_total():
    assert Timeout.coerce(12) == Timeout(total=12.0)
    assert Timeout.coerce(None) is None
    assert Timeout.coerce(Timeout(first_token=2)) == Timeout(first_token=2)


# --- the bound actually bounds ---------------------------------------------------


def test_an_adapter_that_stalls_forever_is_bounded(tmp_path, connection):
    """The whole point: a run that will never answer ends anyway.

    ``StallingAdapter`` blocks inside ``next()``, which is the shape of the real
    hang -- a subprocess-driven runtime that came up and then said nothing. A
    clock checked between events never fires there.
    """
    adapter = StallingAdapter()
    bridge, _, _ = _bridge(tmp_path, connection, adapter)

    events = list(
        bridge.chat(
            connection="claude-sub",
            message="hello",
            timeout=Timeout(first_token=0.05),
        )
    )

    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert terminal.status is TerminalStatus.TIMED_OUT
    assert "first_token" in (terminal.reason or "")
    assert adapter.cancelled == 1


def test_the_total_bound_is_the_whole_run(tmp_path, connection):
    adapter = StallingAdapter(before=[TextDeltaEvent(text="thinking...")])
    bridge, _, _ = _bridge(tmp_path, connection, adapter)

    events = list(
        bridge.chat(connection="claude-sub", message="hi", timeout=0.1)
    )

    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert terminal.status is TerminalStatus.TIMED_OUT
    assert "total" in (terminal.reason or "")
    # The first-token bound never applied, so a delta that arrived did not
    # excuse the run from the bound that did.
    assert any(isinstance(e, TextDeltaEvent) for e in events)


def test_usage_spent_before_the_bound_is_on_the_terminal(tmp_path, connection):
    """A timed-out run is still a billed run.

    An exception or a terminal that dropped the spend would be the one failure
    mode the receipt exists to prevent: tokens paid for and not accounted.
    """
    adapter = StallingAdapter(
        before=[TextDeltaEvent(text="part"), usage(input_tokens=120, output_tokens=30)]
    )
    bridge, _, _ = _bridge(tmp_path, connection, adapter)

    events = list(
        bridge.chat(connection="claude-sub", message="hi", timeout=Timeout(total=0.15))
    )

    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert terminal.status is TerminalStatus.TIMED_OUT
    assert terminal.usage.input_tokens == 120
    assert terminal.usage.output_tokens == 30


def test_the_run_log_records_the_timed_out_run(tmp_path, connection):
    adapter = StallingAdapter(before=[usage(input_tokens=10)])
    bridge, _, _ = _bridge(tmp_path, connection, adapter)

    list(bridge.chat(connection="claude-sub", message="hi", timeout=0.1))

    records = bridge.run_log.read()
    assert records
    assert records[-1]["status"] == "timed_out"


def test_ask_raises_run_timed_out_carrying_the_spend(tmp_path, connection):
    adapter = StallingAdapter(before=[usage(input_tokens=40)])
    bridge, _, _ = _bridge(tmp_path, connection, adapter)

    with pytest.raises(RunTimedOut) as caught:
        bridge.ask("claude-sub", "hello", timeout=Timeout(total=0.1))

    assert caught.value.connection == "claude-sub"
    assert caught.value.which == "total"
    assert caught.value.seconds == 0.1
    assert caught.value.usage.input_tokens == 40


def test_a_first_token_timeout_on_a_stateless_run_is_the_transient_one(
    tmp_path, connection
):
    """The single timeout the table calls retryable outright.

    Nothing answered at all, and the run is stateless, so a second attempt
    repeats a request rather than duplicating a half-finished one. A ``total``
    expiry is UNKNOWN, because the run may have been most of the way there.
    """
    bridge, _, _ = _bridge(tmp_path, connection, StallingAdapter())
    events = list(
        bridge.chat(
            connection="claude-sub", message="hi", timeout=Timeout(first_token=0.05)
        )
    )
    assert events[-1].retryable is Retryable.YES

    bridge2, _, _ = _bridge(
        tmp_path / "two", connection, StallingAdapter(before=[TextDeltaEvent(text="x")])
    )
    events2 = list(
        bridge2.chat(connection="claude-sub", message="hi", timeout=Timeout(total=0.1))
    )
    assert events2[-1].retryable is Retryable.UNKNOWN


def test_a_run_that_finishes_inside_the_bound_is_untouched(tmp_path, connection):
    bridge, _, adapter = fake_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="done"), usage(input_tokens=5)],
        home=tmp_path / "modelpass",
    )
    events = list(
        bridge.chat(connection="claude-sub", message="hi", timeout=Timeout(total=30))
    )
    assert events[-1].status is TerminalStatus.OK
    # Exhausted normally: no cancel at all, timer or otherwise.
    assert adapter.cancelled == 0


# --- the connection-level default -------------------------------------------------


def test_the_connection_supplies_the_bound_when_the_call_passes_none(tmp_path):
    bounded = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        timeout_seconds=0.1,
    )
    bridge, _, _ = _bridge(tmp_path, bounded, StallingAdapter())
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    assert events[-1].status is TerminalStatus.TIMED_OUT


def test_a_call_that_names_a_bound_replaces_the_connection_default(tmp_path):
    """Replaced, never clamped. A caller who named a number has said what they want."""
    bounded = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        timeout_seconds=0.05,
    )
    assert Bridge._call_timeout(bounded, 300) == Timeout(total=300.0)
    assert Bridge._call_timeout(bounded, None) == Timeout(total=0.05)
    # An empty Timeout is a deliberate "unbounded", and overrides the
    # connection rather than falling back to it.
    assert Bridge._call_timeout(bounded, Timeout()) is None


# --- cancellation stays exactly as documented (D10) ------------------------------


def test_closing_the_iterator_mid_stream_cancels_once(tmp_path, connection):
    """A consumer's contract: ``close()`` is the cancel, and it happens once."""
    bridge, _, adapter = fake_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="a"), TextDeltaEvent(text="b"), usage(input_tokens=3)],
        home=tmp_path / "modelpass",
    )
    stream = bridge.chat(connection="claude-sub", message="hi")
    next(stream)  # the receipt
    next(stream)  # the first delta
    stream.close()

    assert adapter.cancelled == 1


def test_closing_the_iterator_cancels_once_with_a_timeout_configured(
    tmp_path, connection
):
    """The two cancel paths must not both fire.

    A double cancel on a real runtime terminates a second process, and a test
    asserting "cancelled once" would pass for the wrong reason if the watchdog
    were still armed behind it.
    """
    bridge, _, adapter = fake_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="a"), TextDeltaEvent(text="b")],
        home=tmp_path / "modelpass",
    )
    stream = bridge.chat(
        connection="claude-sub", message="hi", timeout=Timeout(total=30)
    )
    next(stream)
    next(stream)
    stream.close()

    assert adapter.cancelled == 1
    # And the timer is not still out there waiting to cancel a second time.
    threading.Event().wait(0.05)
    assert adapter.cancelled == 1


def test_the_timeout_path_cancels_once_and_does_not_double_up(tmp_path, connection):
    adapter = StallingAdapter()
    bridge, _, _ = _bridge(tmp_path, connection, adapter)
    list(
        bridge.chat(
            connection="claude-sub", message="hi", timeout=Timeout(first_token=0.05)
        )
    )
    threading.Event().wait(0.05)
    assert adapter.cancelled == 1


def test_a_cancelled_run_is_not_a_timed_out_one(tmp_path, connection):
    """The reason TIMED_OUT is its own member rather than a CANCELLED reason.

    A retry loop wants the timeout and must never touch the cancel: a run the
    caller deliberately stopped is not a run to try again.
    """
    bridge, _, _ = fake_bridge(
        connections=[connection],
        script=[
            TextDeltaEvent(text="a"),
            TerminalEvent(
                status=TerminalStatus.CANCELLED,
                connection="",
                runtime=Runtime.ANTHROPIC_SDK,
                auth_mode=AuthMode.SUBSCRIPTION,
                reason="cancelled by caller",
            ),
        ],
        home=tmp_path / "modelpass",
    )
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    terminal = events[-1]
    assert terminal.status is TerminalStatus.CANCELLED
    assert terminal.retryable is Retryable.NO


# --- sessions get the same bound, per turn ----------------------------------------


def test_a_session_turn_is_bounded_and_the_session_survives(tmp_path, connection):
    from modelpass.testing import FakeSessionAdapter, fake_session_bridge

    adapter = FakeSessionAdapter(
        script=[TextDeltaEvent(text="hello"), usage(input_tokens=4)]
    )
    bridge, _, _ = fake_session_bridge(
        connections=[connection], adapter=adapter, home=tmp_path / "modelpass"
    )
    session = bridge.new_chat(connection="claude-sub", persist=False)
    # A generous bound on a fake that answers immediately changes nothing.
    events = list(session.send("hi", timeout=Timeout(total=30)))
    assert events[-1].status is TerminalStatus.OK
    assert not session.closed
