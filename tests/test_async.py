"""The async face: ``achat``, ``aask``, ``asend`` (R1, ticket 1.13).

The claim under test is not "async works" but "async is the *same* call". Every
test here asks one of the async doors a question its sync twin was already asked
somewhere in this suite, and asserts the two answers are the same object graph --
because the two faces share one pre-run pipeline and one event fold, and the
only way that stays true is if somebody checks.

No plugin: an ``async def`` here is driven by :func:`run`, which is
``asyncio.run`` with a name. The suite has no asyncio plugin declared and this
ticket is not the place to add one.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.errors import (
    AdapterFailed,
    ConnectionDisabled,
    GuardStop,
    QuotaExhausted,
    SessionBusy,
)
from modelpass.runtimes import Runtime
from modelpass.testing import (
    FakeAdapter,
    fake_bridge,
    fake_session_bridge,
    quota_exhausted,
    structured,
    usage,
    vendor_failure,
)
from modelpass.types import (
    AuthMode,
    ReceiptEvent,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    UsageEvent,
)

SCHEMA = {
    "title": "Score",
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
}


def run(coro: Any) -> Any:
    """One loop, one call, no plugin."""
    return asyncio.run(coro)


@pytest.fixture
def connection() -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


def bridged(tmp_path, connection, script=(), **kwargs):
    return fake_bridge(
        connections=[connection], script=script, home=tmp_path / "modelpass", **kwargs
    )


async def drain(stream) -> list:
    return [event async for event in stream]


def shapes(events) -> list[str]:
    return [event.type for event in events]


# --- the two faces answer the same ------------------------------------------------


def test_achat_streams_what_chat_streams(tmp_path, connection):
    script = [TextDeltaEvent(text="the answer"), usage(input_tokens=11, output_tokens=3)]
    sync_bridge, _, _ = bridged(tmp_path / "sync", connection, script)
    async_bridge, _, _ = bridged(tmp_path / "async", connection, script)

    expected = list(sync_bridge.chat(connection="claude-sub", message="hi"))
    actual = run(drain(async_bridge.achat(connection="claude-sub", message="hi")))

    assert shapes(actual) == shapes(expected)
    assert [type(e) for e in actual] == [type(e) for e in expected]
    assert actual[-1].usage == expected[-1].usage
    assert actual[-1].status is TerminalStatus.OK


def test_aask_answers_what_ask_answers(tmp_path, connection):
    script = [TextDeltaEvent(text="7"), usage(input_tokens=4)]
    sync_bridge, _, _ = bridged(tmp_path / "sync", connection, script)
    async_bridge, _, _ = bridged(tmp_path / "async", connection, script)

    expected = sync_bridge.ask("claude-sub", "how many?")
    actual = run(async_bridge.aask("claude-sub", "how many?"))

    assert actual.text == expected.text == "7"
    assert actual.status is expected.status
    assert actual.usage == expected.usage
    assert actual.receipt.summary() == expected.receipt.summary()


def test_the_async_face_carries_the_same_receipt_and_stamp(tmp_path, connection):
    bridge, _, _ = bridged(tmp_path, connection, [usage(input_tokens=2)])
    events = run(drain(bridge.achat(connection="claude-sub", message="hi")))
    receipt = events[0]
    assert isinstance(receipt, ReceiptEvent)
    assert receipt.connection == "claude-sub"
    terminal = events[-1]
    assert terminal.connection == "claude-sub"
    assert terminal.auth_mode is AuthMode.SUBSCRIPTION


def test_a_schema_bound_async_call_answers_structured(tmp_path, connection):
    bridge, _, _ = bridged(
        tmp_path,
        connection,
        [structured({"score": 4}, schema=SCHEMA)],
    )
    answer = run(bridge.aask("claude-sub", "score it", schema=SCHEMA))
    assert answer.structured == {"score": 4}


def test_tools_and_sampling_reach_the_adapter_through_the_async_face(
    tmp_path, connection
):
    bridge, _, adapter = bridged(tmp_path, connection, [usage(input_tokens=1)])
    run(
        drain(
            bridge.achat(
                connection="claude-sub",
                message="hi",
                sampling={"temperature": 0.0},
                system_prompt="be brief",
            )
        )
    )
    (request,) = adapter.requests
    assert request.sampling is not None
    assert request.system_prompt_chars > 0


# --- the refusals -----------------------------------------------------------------


def test_a_disabled_connection_refuses_at_the_call_not_at_the_first_step(tmp_path):
    off = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        enabled=False,
    )
    bridge, _, _ = bridged(tmp_path, off)
    # Not "raises when awaited": raises when called, exactly as chat() does, so
    # a caller mistake is reported at the line that made it.
    with pytest.raises(ConnectionDisabled):
        bridge.achat(connection="claude-sub", message="hi")


def test_a_guard_stop_raises_the_same_error_on_both_faces(tmp_path):
    bounded = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        guards=Guards(stop_at_tokens=10),
    )
    bridge, _, _ = bridged(tmp_path, bounded, [usage(input_tokens=50)])
    with pytest.raises(GuardStop):
        run(
            drain(
                bridge.achat(
                    connection="claude-sub", message="hi", raise_on_stop=True
                )
            )
        )


def test_a_vendor_failure_is_a_terminal_on_the_async_face_too(tmp_path, connection):
    adapter = FakeAdapter([TextDeltaEvent(text="half")], error=vendor_failure())
    bridge, _, _ = bridged(tmp_path, connection, adapter=adapter)
    events = run(drain(bridge.achat(connection="claude-sub", message="hi")))
    terminal = events[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert "rejected" in (terminal.reason or "")


def test_an_adapter_bug_is_still_an_adapter_failure(tmp_path, connection):
    adapter = FakeAdapter([], error=RuntimeError("boom"))
    bridge, _, _ = bridged(tmp_path, connection, adapter=adapter)
    with pytest.raises(AdapterFailed):
        run(drain(bridge.achat(connection="claude-sub", message="hi")))


def test_aask_raises_on_an_exhausted_allowance(tmp_path, connection):
    bridge, _, _ = bridged(tmp_path, connection, [quota_exhausted()])
    with pytest.raises(QuotaExhausted):
        run(bridge.aask("claude-sub", "hi"))


# --- the loop is not blocked ------------------------------------------------------


def test_the_preflight_of_an_agent_runtime_runs_off_the_loop(tmp_path, connection):
    """The bug a batch scraping tool reported: a blocking call inside an ``async def``.

    ``preflight`` is a subprocess on the agent runtimes, so the async face hands
    it to a thread. The adapter records which thread asked.
    """
    seen: list[int] = []
    adapter = FakeAdapter([usage(input_tokens=1)])
    original = adapter.preflight

    def watched(request):
        seen.append(threading.get_ident())
        return original(request)

    adapter.preflight = watched  # type: ignore[method-assign]
    bridge, _, _ = bridged(tmp_path, connection, adapter=adapter)

    async def go():
        loop_thread = threading.get_ident()
        await drain(bridge.achat(connection="claude-sub", message="hi"))
        return loop_thread

    loop_thread = run(go())
    assert seen and seen[0] != loop_thread


def test_many_calls_share_one_loop(tmp_path, connection):
    """The thread-safety contract, extended: one loop, many ``achat`` calls."""
    bridge, _, _ = bridged(tmp_path, connection, [usage(input_tokens=1)])

    async def go():
        streams = [
            drain(bridge.achat(connection="claude-sub", message=f"{n}"))
            for n in range(4)
        ]
        return await asyncio.gather(*streams)

    runs = run(go())
    assert len(runs) == 4
    for events in runs:
        assert events[-1].status is TerminalStatus.OK


# --- failover crosses connections on this face too ---------------------------------


def test_a_failover_runs_both_legs_on_the_async_face(tmp_path):
    from modelpass.connections import QuotaAction, QuotaPolicy

    primary = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        guards=Guards(
            on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api")
        ),
    )
    metered = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
    )
    legs = iter([[quota_exhausted()], [TextDeltaEvent(text="paid for"), usage(input_tokens=5)]])
    adapter = FakeAdapter(lambda request: next(legs))
    bridge, _, _ = fake_bridge(
        connections=[primary, metered],
        adapter=adapter,
        home=tmp_path / "modelpass",
        env={"ANTHROPIC_API_KEY": "sk-test"},
    )
    events = run(drain(bridge.achat(connection="claude-sub", message="hi")))
    kinds = shapes(events)
    assert kinds.count("receipt") == 2
    assert "failover" in kinds
    assert kinds.count("terminal") == 1
    assert events[-1].failed_over_from == "claude-sub"


# --- sessions ---------------------------------------------------------------------


def session_bridge(tmp_path, connection=None, **kwargs):
    return fake_session_bridge(
        connections=[connection or session_connection()],
        home=tmp_path / "modelpass",
        **kwargs,
    )


def session_connection() -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


def test_asend_runs_a_turn_like_send(tmp_path):
    bridge, _, _ = session_bridge(
        tmp_path, script=[TextDeltaEvent(text="hello"), usage(input_tokens=3)]
    )
    session = bridge.new_chat(connection="claude-sub")
    events = run(drain(session.asend("hi")))
    assert shapes(events)[0] == "receipt"
    assert events[-1].status is TerminalStatus.OK
    assert session.usage.total_tokens == 3
    session.close()


def test_a_session_takes_one_caller_whichever_face_it_is(tmp_path):
    """R7's one-caller rule holds *across* the two faces, not within each."""
    bridge, _, _ = session_bridge(
        tmp_path, script=[TextDeltaEvent(text="hello"), usage(input_tokens=3)]
    )
    session = bridge.new_chat(connection="claude-sub")
    sync_turn = session.send("hi")
    next(sync_turn)  # takes the lock on the first step, as documented

    async def second():
        return await drain(session.asend("and again"))

    with pytest.raises(SessionBusy):
        run(second())

    list(sync_turn)
    # And once the sync turn is done, the async one is welcome.
    events = run(second())
    assert events[-1].status is TerminalStatus.OK
    session.close()


def test_a_sync_send_is_refused_while_an_async_turn_holds_the_session(tmp_path):
    bridge, _, _ = session_bridge(
        tmp_path, script=[TextDeltaEvent(text="hello"), usage(input_tokens=3)]
    )
    session = bridge.new_chat(connection="claude-sub")

    async def go():
        turn = session.asend("hi")
        await turn.__anext__()
        with pytest.raises(SessionBusy):
            list(session.send("meanwhile"))
        await drain(turn)

    run(go())
    session.close()


def test_an_async_turn_records_its_line_in_the_ledger(tmp_path):
    bridge, _, _ = session_bridge(
        tmp_path, script=[TextDeltaEvent(text="hello"), usage(input_tokens=3)]
    )
    session = bridge.new_chat(connection="claude-sub")
    run(drain(session.asend("hi")))
    records = bridge.run_log.read()
    assert len(records) == 1
    assert records[0]["connection"] == "claude-sub"
    session.close()


def test_an_async_turn_that_fails_reports_the_vendor_failure(tmp_path):
    bridge, _, adapter = session_bridge(
        tmp_path, script=[TextDeltaEvent(text="half")]
    )
    adapter.raise_at_end = vendor_failure()
    session = bridge.new_chat(connection="claude-sub")
    events = run(drain(session.asend("hi")))
    assert events[-1].status is TerminalStatus.ERROR
    session.close()


def test_events_are_the_same_objects_not_copies(tmp_path, connection):
    """The fold passes an ordinary event through; the async face does not re-wrap."""
    delta = TextDeltaEvent(text="one")
    bridge, _, _ = bridged(tmp_path, connection, [delta, usage(input_tokens=1)])
    events = run(drain(bridge.achat(connection="claude-sub", message="hi")))
    assert any(event is delta for event in events)
    assert any(isinstance(event, UsageEvent) for event in events)
    assert isinstance(events[-1], TerminalEvent)
