"""Cancelling an async run (R2, ticket 1.13).

R2 asks for this file by name, and the reason is that cancellation is the one
place in the async face where a bug hides: everything else is visible in the
events, and a thread that was never joined is visible in nothing at all until a
consumer's process will not exit.

Four promises, and the file is honest about which of them are guarantees:

* ``aclose()`` mid-stream cancels the run **exactly once**;
* ``aclose()`` after a bound already fired does **not** cancel a second time,
  because the watchdog already did -- the same rule the sync face states, and
  the one a consumer's own cancellation contract rests on;
* a caller who abandons the iterator **without** ``aclose()`` does not leave a
  worker thread alive forever -- best effort, at a moment the garbage collector
  chooses, and documented as best effort rather than promised;
* a timed-out ``achat`` finishes its own teardown **without deadlocking the
  loop**, which is the failure a bounded wait inside a coroutine invites.

The sync close-as-cancel regression from ticket 1.12 lives in
``tests/test_timeout.py`` and is untouched by any of this.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from typing import Any

import pytest

from modelpass.adapters.base import RunRequest
from modelpass.connections import Connection, CredentialRef
from modelpass.errors import RunTimedOut
from modelpass.preflight import plan_launch
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter, StallingAdapter, fake_bridge, usage
from modelpass.types import AuthMode, TerminalStatus, TextDeltaEvent, Timeout

#: Long enough that a machine under load does not fail a passing test, short
#: enough that a broken one fails rather than hanging the suite.
PATIENCE = 10.0


def run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture
def connection() -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


def bridged(tmp_path, connection, adapter):
    return fake_bridge(
        connections=[connection], adapter=adapter, home=tmp_path / "modelpass"
    )


def request_for(connection: Connection) -> RunRequest:
    return RunRequest(
        connection=connection,
        messages=(),
        plan=plan_launch(connection, {}),
        model=None,
        options={},
    )


def worker_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("modelpass-arun-")]


def wait_for(predicate, patience: float = PATIENCE) -> bool:
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --- aclose() is the async cancel --------------------------------------------------


def test_aclose_mid_stream_cancels_once(tmp_path, connection):
    """The async twin of "closing the iterator cancels" (D10).

    Once, not twice: the graceful cancel and the transport teardown are two
    halves of one stop, and a runtime that took two of them could be asked to
    terminate a process it had already replaced.
    """
    adapter = StallingAdapter(before=[TextDeltaEvent(text="a"), usage(input_tokens=4)])
    bridge, _, _ = bridged(tmp_path, connection, adapter)

    async def go():
        events = bridge.achat(connection="claude-sub", message="hi")
        seen = [await events.__anext__(), await events.__anext__()]
        await events.aclose()
        return seen

    seen = run(go())
    assert seen[0].type == "receipt"
    assert adapter.cancelled == 1
    # And nothing is still out there waiting to cancel a second time.
    threading.Event().wait(0.05)
    assert adapter.cancelled == 1


def test_aclose_joins_the_worker_thread(tmp_path, connection):
    """R2's other half: the thread the default ``arun`` used is gone afterwards."""
    adapter = StallingAdapter(before=[TextDeltaEvent(text="a")])
    bridge, _, _ = bridged(tmp_path, connection, adapter)

    async def go():
        events = bridge.achat(connection="claude-sub", message="hi")
        await events.__anext__()
        await events.__anext__()
        assert worker_threads(), "the default arun drives run() on a worker"
        await events.aclose()

    run(go())
    assert wait_for(lambda: not worker_threads())


def test_an_abandoned_run_leaves_no_line_in_the_ledger(tmp_path, connection):
    """Exactly what the sync face does, and the reason is the same.

    A run nobody finished has no terminal, and the ledger is a record of
    terminals: writing a line here would put a run in the log that never ended
    and whose spend nobody ever totalled. The cancel still happens; the
    bookkeeping does not, on either face.
    """
    adapter = StallingAdapter(before=[TextDeltaEvent(text="a")])
    bridge, _, _ = bridged(tmp_path, connection, adapter)

    async def go():
        events = bridge.achat(connection="claude-sub", message="hi")
        # The receipt, then the first event the runtime produced: a run that has
        # actually started is what there is to cancel.
        await events.__anext__()
        await events.__anext__()
        await events.aclose()

    run(go())
    assert bridge.run_log.read() == ()
    assert adapter.cancelled == 1


def test_aclose_on_a_finished_run_cancels_nothing(tmp_path, connection):
    adapter = FakeAdapter([TextDeltaEvent(text="all of it"), usage(input_tokens=2)])
    bridge, _, _ = bridged(tmp_path, connection, adapter)

    async def go():
        events = bridge.achat(connection="claude-sub", message="hi")
        drained = [event async for event in events]
        await events.aclose()
        return drained

    drained = run(go())
    assert drained[-1].status is TerminalStatus.OK
    # The run ended on its own. Cancelling a finished run would be a call the
    # runtime has no answer for.
    assert adapter.cancelled == 0


# --- the bound and the close do not both cancel ------------------------------------


def test_aclose_after_a_timeout_does_not_double_cancel(tmp_path, connection):
    """The watchdog cancelled from the timer thread; the teardown must not repeat it."""
    adapter = StallingAdapter()
    bridge, _, _ = bridged(tmp_path, connection, adapter)

    async def go():
        events = bridge.achat(
            connection="claude-sub", message="hi", timeout=Timeout(first_token=0.05)
        )
        drained = [event async for event in events]
        await events.aclose()
        return drained

    drained = run(go())
    assert drained[-1].status is TerminalStatus.TIMED_OUT
    threading.Event().wait(0.1)
    assert adapter.cancelled == 1


def test_a_timed_out_achat_finishes_without_deadlocking_the_loop(tmp_path, connection):
    """The bug a bounded join inside a coroutine invites.

    The watchdog cancels from a timer thread while the loop is parked inside
    ``await``; the teardown then has to join a worker without blocking the loop
    that is doing the joining. If that were wrong this test would not fail, it
    would hang -- so it is run with its own wall-clock bound and the bound is
    the assertion.
    """
    adapter = StallingAdapter()
    bridge, _, _ = bridged(tmp_path, connection, adapter)

    async def go():
        return await asyncio.wait_for(
            _drain(
                bridge.achat(
                    connection="claude-sub",
                    message="hi",
                    timeout=Timeout(first_token=0.05),
                )
            ),
            timeout=PATIENCE,
        )

    events = run(go())
    assert events[-1].status is TerminalStatus.TIMED_OUT
    assert wait_for(lambda: not worker_threads())


def test_raise_on_stop_still_raises_on_the_async_face(tmp_path, connection):
    adapter = StallingAdapter()
    bridge, _, _ = bridged(tmp_path, connection, adapter)

    async def go():
        events = bridge.achat(
            connection="claude-sub",
            message="hi",
            timeout=Timeout(first_token=0.05),
            raise_on_stop=True,
        )
        return [event async for event in events]

    with pytest.raises(RunTimedOut):
        run(go())


async def _drain(stream) -> list:
    return [event async for event in stream]


# --- the abandoned iterator, best effort -------------------------------------------


def test_an_abandoned_arun_does_not_leave_a_worker_thread_alive(connection):
    """**Best effort, and documented as best effort** (R2).

    A consumer who neither drains nor closes gets no promise about *when* the
    worker stops -- that is whenever the garbage collector reaches the object --
    but it must stop. The alternative is a thread, and on the agent runtimes a
    subprocess, living as long as the process that forgot about it.

    ``aclose()`` is the guarantee. This is the safety net under callers who
    forget it, and it is why the worker is a daemon thread as well.
    """
    adapter = StallingAdapter(before=[TextDeltaEvent(text="a")])

    async def go():
        stream = adapter.arun(request_for(connection))
        await stream.__anext__()
        assert worker_threads()
        del stream
        gc.collect()

    run(go())
    assert wait_for(lambda: not worker_threads()), (
        "a forgotten run left its worker thread alive"
    )
    assert adapter.cancelled == 1


def test_the_worker_stops_when_the_consumer_stops_reading(connection):
    """A bounded queue must never trap the worker behind a consumer who left.

    The worker blocks on ``put`` when the queue fills. If that block ignored the
    stop flag, every abandoned fast run would leak a thread -- which is the one
    failure mode the bound itself introduces.
    """
    script = [TextDeltaEvent(text=f"{n}") for n in range(500)]
    adapter = FakeAdapter(script)

    async def go():
        stream = adapter.arun(request_for(connection))
        await stream.__anext__()
        await stream.aclose()

    run(go())
    assert wait_for(lambda: not worker_threads())


def test_an_exception_from_the_worker_reaches_the_caller(connection):
    adapter = FakeAdapter([TextDeltaEvent(text="a")], error=RuntimeError("boom"))

    async def go():
        stream = adapter.arun(request_for(connection))
        return [event async for event in stream]

    with pytest.raises(RuntimeError, match="boom"):
        run(go())
    assert wait_for(lambda: not worker_threads())


# --- the sync contract is untouched -------------------------------------------------


def test_closing_the_sync_iterator_still_cancels(tmp_path, connection):
    """The ticket 1.12 regression, restated here because this file is where a

    reader looks for the cancellation contract. ``tests/test_timeout.py`` owns
    the original and it was not edited by this ticket.
    """
    adapter = FakeAdapter([TextDeltaEvent(text="a"), TextDeltaEvent(text="b")])
    bridge, _, _ = bridged(tmp_path, connection, adapter)
    stream = bridge.chat(connection="claude-sub", message="hi")
    next(stream)
    next(stream)
    stream.close()
    assert adapter.cancelled == 1
