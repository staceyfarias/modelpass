"""A cancel reaches an API run that is blocked, and no tool runs after one.

The four API runtimes used to read their cancel flag in exactly one place: the
top of each tool round. Everything in between was out of reach --

* a request whose answer never started, or a stream that stalled part-way, held
  the run until the vendor SDK's own read timeout (600 s on anthropic and
  openai). ``cancel()`` set a flag nobody was awake to read, so the timeout
  bound of ticket 1.12 -- a timer that *cancels the run* -- bounded nothing on
  these runtimes, on either face;
* a cancel that arrived while a round was streaming did not stop the tool
  handlers that round asked for. They ran, with their side effects, and only
  then did the run end ``cancelled``.

What is held here, on the sync and the async face of all four:

1. **Real SDK clients against a server that goes silent** -- before the response
   headers, and after them. The run ends ``cancelled`` promptly; through the
   bridge with a bound, it ends ``timed_out``. No fake can hold this: the defect
   is a thread blocked inside the vendor's HTTP stack.
2. **No handler runs after a cancel**, wherever in the round it lands, and the
   stream still ends on exactly one terminal.
3. **A cancel that races a normal ending** raises nothing and adds no second
   terminal.

The server is a loopback socket that never answers. Nothing leaves the machine
and nothing is spent.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import socket
import threading
import time
from typing import Any

import pytest

import test_async_adapters as async_fakes
from modelpass.adapters import base
from modelpass.adapters.base import AsyncInterrupt, RunCancel, RunInterrupted, interrupt_client
from modelpass.tools import ToolDef
from modelpass.types import (
    TerminalEvent,
    TerminalStatus,
    Timeout,
    ToolCallEvent,
    ToolResultEvent,
)

#: How long any wait here may take before the test fails instead of hanging the
#: suite. A cancel that works lands in milliseconds; the SDK timeouts it stands
#: in for are 600 seconds.
PATIENCE = 10.0

RUNTIMES = {
    "anthropic-api": "test_adapter_anthropic_api",
    "openai-api": "test_adapter_openai_api",
    "google-api": "test_adapter_google_api",
    "openai-compatible": "test_adapter_openai_compatible",
}


@pytest.fixture(params=sorted(RUNTIMES))
def fakes(request):
    """One adapter's own test module: its connection, request and adapter builders."""
    module = importlib.import_module(RUNTIMES[request.param])
    if module.NEEDS_SDK.args[0]:
        pytest.skip(module.NEEDS_SDK.kwargs["reason"])
    return module


def terminals(events: list[Any]) -> list[TerminalEvent]:
    return [event for event in events if isinstance(event, TerminalEvent)]


def only_terminal(events: list[Any]) -> TerminalEvent:
    found = terminals(events)
    assert len(found) == 1, events
    assert events[-1] is found[0], events
    return found[0]


# --- a server that takes the request and then says nothing ------------------------


class SilentServer:
    """Accepts, reads the request, and goes quiet -- before or after the headers."""

    def __init__(self, answer_headers: bool) -> None:
        self._answer_headers = answer_headers
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._held: list[socket.socket] = []
        #: Set once a request has arrived (and, in the headers mode, been
        #: answered as far as it is ever going to be).
        self.engaged = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._listener.getsockname()[1]}"

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            self._held.append(conn)
            threading.Thread(target=self._take, args=(conn,), daemon=True).start()

    def _take(self, conn: socket.socket) -> None:
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                more = conn.recv(65536)
                if not more:
                    return
                data += more
            if self._answer_headers:
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"content-type: text/event-stream\r\n"
                    b"transfer-encoding: chunked\r\n\r\n"
                )
        except OSError:
            return
        self.engaged.set()
        # ... and nothing more, ever.

    def close(self) -> None:
        self._listener.close()
        for conn in self._held:
            try:
                conn.close()
            except OSError:
                pass


@pytest.fixture(params=["before the headers", "after the headers"])
def silent(request):
    server = SilentServer(answer_headers=request.param == "after the headers")
    yield server
    server.close()


def _real(fakes: Any, server: SilentServer) -> tuple[Any, Any]:
    """The adapter with its **default** client factories, aimed at the silent server."""
    conn = fakes.connection(base_url=server.url)
    return fakes.adapter(None), fakes.run_request(conn)


def _blocked(server: SilentServer) -> None:
    """Wait until the run is, as near as can be told from outside, blocked in a read."""
    assert server.engaged.wait(PATIENCE), "the request never reached the server"
    # The client is now waiting on bytes that will not come. A moment for it to
    # get from "sent" into the read; the outcome asserted does not depend on it.
    time.sleep(0.2)


class Drain(threading.Thread):
    def __init__(self, stream: Any) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self.events: list[Any] = []
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            for event in self._stream:
                self.events.append(event)
        except BaseException as exc:
            self.error = exc

    def finished(self) -> list[Any]:
        self.join(PATIENCE)
        assert not self.is_alive(), "the cancel did not unblock the run"
        if self.error is not None:
            raise self.error
        return self.events


# --- 1. a blocked run is interrupted ----------------------------------------------


def test_cancel_unblocks_a_sync_run_waiting_on_a_silent_server(fakes, silent, monkeypatch):
    # The default factory, observed: the adapter still builds (and so owns) the
    # client, and the test can see which sockets the cancel is about to cut.
    adapter_module = importlib.import_module(fakes.adapter(None).__class__.__module__)
    built: list[Any] = []
    default = adapter_module._default_client

    def observed(**kwargs: Any) -> Any:
        built.append(default(**kwargs))
        return built[-1]

    monkeypatch.setattr(adapter_module, "_default_client", observed)
    adapter, request = _real(fakes, silent)

    stream = adapter.run(request)
    drain = Drain(stream)
    drain.start()
    _blocked(silent)
    # The path ``interrupt_client`` walks to the socket, held against the real
    # client: an SDK upgrade that moves it must fail here, not hang a user.
    assert len(base._client_sockets(built[0])) == 1

    began = time.monotonic()
    stream.cancel()
    terminal = only_terminal(drain.finished())

    assert terminal.status is TerminalStatus.CANCELLED
    assert terminal.reason == "cancelled by caller"
    assert time.monotonic() - began < PATIENCE


def test_cancel_unblocks_an_async_run_waiting_on_a_silent_server(fakes, silent):
    adapter, request = _real(fakes, silent)

    async def main() -> list[Any]:
        stream = adapter.arun(request)

        async def drain() -> list[Any]:
            return [event async for event in stream]

        task = asyncio.ensure_future(drain())
        await asyncio.to_thread(_blocked, silent)
        # From another thread, the way the timeout watchdog cancels.
        canceller = threading.Thread(target=stream.cancel)
        canceller.start()
        try:
            return await asyncio.wait_for(task, PATIENCE)
        finally:
            canceller.join()
            await stream.aclose(cancel=False)

    terminal = only_terminal(asyncio.run(main()))
    assert terminal.status is TerminalStatus.CANCELLED
    assert terminal.reason == "cancelled by caller"


def _bridge(fakes: Any, server: SilentServer, store: Any, **changes: Any) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = {"base_url": server.url}
    if fakes.__name__ == "test_adapter_openai_compatible":
        kwargs["verified"] = fakes.all_verified()
    conn = dataclasses.replace(fakes.connection(**kwargs), **changes)
    bridge, _ = fakes.bridge_for(conn, None, store)
    return bridge, conn


def test_a_timeout_ends_a_sync_run_the_server_never_answers(fakes, silent, store):
    """Ticket 1.12's bound, on the runtimes where it used to bound nothing."""
    bridge, conn = _bridge(fakes, silent, store)
    drain = Drain(bridge.chat(connection=conn.name, message="hi", timeout=Timeout(total=0.5)))
    drain.start()
    terminal = only_terminal(drain.finished())
    assert terminal.status is TerminalStatus.TIMED_OUT, terminal.reason


def test_a_timeout_ends_an_async_run_the_server_never_answers(fakes, silent, store):
    bridge, conn = _bridge(fakes, silent, store)

    async def main() -> list[Any]:
        stream = bridge.achat(connection=conn.name, message="hi", timeout=Timeout(total=0.5))

        async def drain() -> list[Any]:
            return [event async for event in stream]

        return await asyncio.wait_for(drain(), PATIENCE)

    terminal = only_terminal(asyncio.run(main()))
    assert terminal.status is TerminalStatus.TIMED_OUT, terminal.reason


def test_a_connections_own_timeout_seconds_is_enough(fakes, silent, store):
    """``timeoutSeconds`` on the connection is the same bound, with nothing passed."""
    bridge, conn = _bridge(fakes, silent, store, timeout_seconds=0.5)
    drain = Drain(bridge.chat(connection=conn.name, message="hi"))
    drain.start()
    terminal = only_terminal(drain.finished())
    assert terminal.status is TerminalStatus.TIMED_OUT, terminal.reason


# --- 2. no handler runs after a cancel --------------------------------------------


class Handler:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, arguments: Any) -> str:
        self.calls += 1
        return "found"

    def tool(self) -> ToolDef:
        return ToolDef(
            name="look_up",
            description="Look something up.",
            parameters={"type": "object", "properties": {}},
            handler=self,
        )


def _tool_then_text(fakes: Any) -> list[Any]:
    return [async_fakes.tool_round_for(fakes, "call_1", "look_up", {}), fakes.text_round("done")]


def _drive_sync(fakes: Any, rounds: list[Any], handler: Handler, cancel_on: Any) -> list[Any]:
    """Run over scripted rounds, cancelling the first time ``cancel_on(event)`` is true."""
    stream = fakes.adapter(fakes.Factory(rounds)).run(fakes.run_request(tools=[handler.tool()]))
    events: list[Any] = []
    cancelled = False
    for event in stream:
        events.append(event)
        if not cancelled and cancel_on(event):
            cancelled = True
            stream.cancel()
    return events


def _drive_async(fakes: Any, rounds: list[Any], handler: Handler, cancel_on: Any) -> list[Any]:
    build = {module: case for case, module in async_fakes.CASES.values()}[fakes]
    adapter, _, _ = build(rounds)

    async def main() -> list[Any]:
        stream = adapter.arun(fakes.run_request(tools=[handler.tool()]))
        events: list[Any] = []
        cancelled = False
        async for event in stream:
            events.append(event)
            if not cancelled and cancel_on(event):
                cancelled = True
                stream.cancel()
        return events

    return asyncio.run(main())


DRIVERS = {"sync": _drive_sync, "async": _drive_async}


@pytest.fixture(params=sorted(DRIVERS))
def drive(request):
    return DRIVERS[request.param]


def test_a_cancel_while_the_round_streams_runs_no_handler(fakes, drive):
    """The reported scenario: cancelled mid-stream, the tool the stream asks for must not run."""
    handler = Handler()
    events = drive(fakes, _tool_then_text(fakes), handler, lambda event: True)

    assert handler.calls == 0
    assert not [e for e in events if isinstance(e, (ToolCallEvent, ToolResultEvent))]
    assert only_terminal(events).status is TerminalStatus.CANCELLED


def test_a_cancel_on_seeing_the_call_announced_runs_no_handler(fakes, drive):
    handler = Handler()
    events = drive(
        fakes, _tool_then_text(fakes), handler, lambda event: isinstance(event, ToolCallEvent)
    )

    assert handler.calls == 0
    assert not [e for e in events if isinstance(e, ToolResultEvent)]
    assert only_terminal(events).status is TerminalStatus.CANCELLED


def test_a_cancel_during_one_handler_stops_the_next(fakes, drive):
    """The cancel lands on the first result; the next round's call never runs."""
    handler = Handler()
    rounds = [
        async_fakes.tool_round_for(fakes, "call_1", "look_up", {}),
        async_fakes.tool_round_for(fakes, "call_2", "look_up", {}),
        fakes.text_round("done"),
    ]
    events = drive(fakes, rounds, handler, lambda event: isinstance(event, ToolResultEvent))

    assert handler.calls == 1
    assert only_terminal(events).status is TerminalStatus.CANCELLED


# --- 3. a cancel that races a normal ending ---------------------------------------


def test_a_cancel_after_the_terminal_changes_nothing(fakes, drive):
    handler = Handler()
    events = drive(
        fakes, [fakes.text_round("done")], handler, lambda e: isinstance(e, TerminalEvent)
    )
    assert only_terminal(events).status is TerminalStatus.OK


def test_a_cancel_on_the_last_event_before_the_terminal_still_ends_once(fakes, drive):
    """Whichever way the race falls, the stream ends on exactly one terminal."""
    handler = Handler()
    seen: list[Any] = []

    def on_the_last_one(event: Any) -> bool:
        seen.append(event)
        return False

    script = drive(fakes, [fakes.text_round("done")], handler, on_the_last_one)
    last_before_terminal = len(script) - 2

    counter = iter(range(len(script)))
    events = drive(
        fakes,
        [fakes.text_round("done")],
        handler,
        lambda event: next(counter) == last_before_terminal,
    )
    assert only_terminal(events).status in (TerminalStatus.OK, TerminalStatus.CANCELLED)


@pytest.mark.parametrize(
    ("module", "package", "base_url"),
    [
        ("anthropic_api", "anthropic", None),
        ("openai_api", "openai", None),
        ("google_api", "google.genai", None),
        ("openai_compatible", "openai", "https://gateway.example/v1"),
    ],
)
def test_interrupting_an_idle_real_client_is_harmless_and_repeatable(module, package, base_url):
    """The hook also fires for a run that already ended: nothing in flight, no error."""
    pytest.importorskip(package)
    adapter = importlib.import_module(f"modelpass.adapters.{module}")
    client = adapter._default_client(api_key="a-key", base_url=base_url)
    assert base._client_sockets(client) == []
    interrupt_client(client)
    interrupt_client(client)


def test_interrupting_something_that_is_not_a_client_is_harmless():
    interrupt_client(object())


# --- the async interrupt, on its own ----------------------------------------------


def test_async_interrupt_from_another_thread_ends_the_wait():
    async def main() -> None:
        cancel = RunCancel()
        scope = AsyncInterrupt(cancel)
        threading.Timer(0.05, cancel.cancel).start()
        with pytest.raises(RunInterrupted):
            await asyncio.wait_for(scope.run(asyncio.sleep(PATIENCE * 4)), PATIENCE)
        # ... and nothing is started once the run is cancelled.
        started = []

        async def never() -> None:
            started.append(True)

        with pytest.raises(RunInterrupted):
            await scope.run(never())
        assert started == []

    asyncio.run(main())


def test_async_interrupt_leaves_the_callers_own_cancellation_alone():
    """A caller cancelling its task gets ``CancelledError``, not a ``cancelled`` terminal."""

    async def main() -> None:
        scope = AsyncInterrupt(RunCancel())
        waiting = asyncio.Event()

        async def wait() -> None:
            waiting.set()
            await asyncio.sleep(PATIENCE * 4)

        task = asyncio.ensure_future(scope.run(wait()))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
