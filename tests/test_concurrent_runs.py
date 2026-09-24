"""Concurrent runs on one bridge do not cancel each other.

Reported by a RAG evaluation harness on 2026-09-23: four worker threads sharing
one :class:`~modelpass.Bridge` on an ``anthropic-sdk`` subscription connection
lost 95 of 200 judge calls to ``cancelled by caller``. Nobody had cancelled
anything, and sequential calls never failed.

The mechanism, re-derived from the code rather than taken from the report:

* a bridge keeps **one adapter per runtime** for its lifetime
  (:meth:`~modelpass.Bridge.adapter_for`);
* the adapters kept **one run's** cancel state on that shared instance -- a
  cancelled flag, and on the agent runtimes the live client or process;
* every run that ended on its own terminal was then cancelled by the bridge's
  teardown (and, on ``anthropic-sdk``, by the adapter's own generator), and a
  timed-out run was cancelled by the watchdog, all through ``adapter.cancel()``
  -- which reached **whichever run had started last**, not the one ending.

The expected outcome in every test here is derived from the contract, not from
what the code does: a run that nobody cancelled ends ``ok``, and only the run a
caller (or its own deadline) cancelled ends cancelled or timed out.

Every test forces the one interleaving that exposes the defect -- run B is in
flight when run A ends -- with events rather than sleeps, so the outcome is
deterministic. No network, no vendor package call, no credential.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from modelpass.adapters.anthropic import AnthropicAdapter
from modelpass.adapters.openai import OpenAIAdapter
from modelpass.bridge import Bridge
from modelpass.capabilities import CapabilityRegistry
from modelpass.connections import Connection, CredentialRef
from modelpass.preflight import Receipt
from modelpass.runtimes import Runtime
from modelpass.tools import ToolDef
from modelpass.types import AuthMode, TerminalEvent, TerminalStatus, Timeout

#: How long any wait in this file may take before the test fails instead of
#: hanging the suite. Nothing here should come near it.
PATIENCE = 10.0


def terminal_of(events: list[Any]) -> TerminalEvent:
    terminals = [e for e in events if isinstance(e, TerminalEvent)]
    assert len(terminals) == 1, events
    return terminals[0]


class Worker(threading.Thread):
    """Run one call on its own thread and keep what it produced, or what it raised."""

    def __init__(self, call) -> None:
        super().__init__(daemon=True)
        self._call = call
        self.events: list[Any] | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.events = self._call()
        except BaseException as exc:  # re-raised by result()
            self.error = exc

    def result(self) -> TerminalEvent:
        self.join(PATIENCE)
        assert not self.is_alive(), "the run never finished"
        if self.error is not None:
            raise self.error
        assert self.events is not None
        return terminal_of(self.events)


def _offline_receipt(request) -> Receipt:
    return Receipt.from_plan(
        request.plan,
        detected_auth_mode=request.connection.auth_mode,
        runtime_available=True,
        ok=True,
    )


# --- anthropic-sdk: a fake Agent SDK whose turns end when the test says -----------
# The event mapper dispatches on class name, so these carry the SDK's names.


@dataclass
class TextBlock:
    text: str


@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantMessage:
    content: list[Any] = field(default_factory=list)
    model: str = "claude-sonnet-5"
    error: str | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class ResultMessage:
    subtype: str = "success"
    is_error: bool = False
    session_id: str = "s"
    usage: dict[str, Any] | None = None
    total_cost_usd: float | None = None
    result: str | None = None
    terminal_reason: str | None = None
    api_error_status: int | None = None
    num_turns: int = 1
    duration_ms: int = 10


class GatedClient:
    """A ``ClaudeSDKClient`` whose one turn answers only when released.

    ``interrupt()`` behaves the way the real one does from the consumer's side:
    the turn stops and no ``ResultMessage`` arrives. Every interrupt is counted,
    because an interrupt that reached the wrong client is the defect.
    """

    def __init__(self, options: Any) -> None:
        self.options = options
        self.connected = threading.Event()
        self.release = threading.Event()
        self.interrupts = 0

    async def connect(self, prompt: Any) -> None:
        self.connected.set()

    async def receive_response(self):
        # Something arrives at once, as the real init message does, so a caller
        # can hold a run that is demonstrably live before it is released.
        yield SystemMessage(subtype="init")
        while not self.release.is_set():
            await asyncio.sleep(0.005)
        if self.interrupts:
            return
        yield AssistantMessage(content=[TextBlock("done")])
        yield ResultMessage(result="done")

    async def interrupt(self) -> None:
        self.interrupts += 1
        self.release.set()

    async def disconnect(self) -> None:
        # The real one waits for the CLI child to exit, which takes a while. The
        # wait matters here: while it lasts, the old adapter still held the
        # finished run's teardown and a newer run's client in the same slot.
        await asyncio.sleep(0.2)


class GatedSdk:
    def __init__(self) -> None:
        self.clients: list[GatedClient] = []
        self.created = threading.Condition()

    def ClaudeAgentOptions(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def ClaudeSDKClient(self, options: Any) -> GatedClient:
        client = GatedClient(options)
        with self.created:
            self.clients.append(client)
            self.created.notify_all()
        return client

    def client(self, index: int) -> GatedClient:
        """The ``index``-th client, once it exists and has connected."""
        with self.created:
            assert self.created.wait_for(lambda: len(self.clients) > index, PATIENCE)
            client = self.clients[index]
        assert client.connected.wait(PATIENCE)
        return client


class OfflineAnthropicAdapter(AnthropicAdapter):
    """The real adapter with the two machine-touching seams stubbed.

    ``run()``, ``cancel()`` and everything between them are the production code;
    only the preflight (which launches the CLI) and the SDK import are replaced.
    """

    def __init__(self, sdk: GatedSdk) -> None:
        super().__init__()
        self._fake_sdk = sdk

    @classmethod
    def is_available(cls) -> bool:
        return True

    def _sdk(self) -> Any:  # type: ignore[override]
        return self._fake_sdk

    def preflight(self, request):  # type: ignore[override]
        return _offline_receipt(request)


@pytest.fixture
def claude_sub() -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


@pytest.fixture
def agent_bridge(store, claude_sub):
    sdk = GatedSdk()
    store.add(claude_sub)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: OfflineAnthropicAdapter(sdk)},
        env={},
    )
    return bridge, sdk


def test_a_finished_run_does_not_cancel_one_still_in_flight(agent_bridge):
    """The reported defect, on the sync face: A ends ``ok``, B must too."""
    bridge, sdk = agent_bridge

    run_a = Worker(lambda: list(bridge.chat(connection="claude-sub", message="A")))
    run_a.start()
    client_a = sdk.client(0)
    # B starts second, so on the old code it owns the adapter's shared client
    # slot -- the slot A's teardown then cancelled.
    run_b = Worker(lambda: list(bridge.chat(connection="claude-sub", message="B")))
    run_b.start()
    client_b = sdk.client(1)

    client_a.release.set()
    assert run_a.result().status is TerminalStatus.OK
    client_b.release.set()
    terminal_b = run_b.result()

    assert terminal_b.status is TerminalStatus.OK, terminal_b.reason
    assert client_b.interrupts == 0
    # And a run that ended on its own terminal is not interrupted after it,
    # either: there is nothing left to cancel.
    assert client_a.interrupts == 0


def test_a_finished_run_does_not_cancel_one_still_in_flight_async(agent_bridge):
    """The same interleaving through ``achat``, whose teardown is ``aclose()``."""
    bridge, sdk = agent_bridge

    async def collect(message: str) -> list[Any]:
        return [event async for event in bridge.achat(connection="claude-sub", message=message)]

    run_a = Worker(lambda: asyncio.run(collect("A")))
    run_a.start()
    client_a = sdk.client(0)
    run_b = Worker(lambda: asyncio.run(collect("B")))
    run_b.start()
    client_b = sdk.client(1)

    client_a.release.set()
    assert run_a.result().status is TerminalStatus.OK
    client_b.release.set()
    terminal_b = run_b.result()

    assert terminal_b.status is TerminalStatus.OK, terminal_b.reason
    assert client_b.interrupts == 0


def test_a_timed_out_run_cancels_itself_and_nothing_else(agent_bridge):
    """The watchdog's cancel is aimed at its own run, not at the shared adapter."""
    bridge, sdk = agent_bridge

    run_a = Worker(
        lambda: list(
            bridge.chat(connection="claude-sub", message="A", timeout=Timeout(total=0.3))
        )
    )
    run_a.start()
    client_a = sdk.client(0)  # never released: only its own deadline ends it
    run_b = Worker(lambda: list(bridge.chat(connection="claude-sub", message="B")))
    run_b.start()
    client_b = sdk.client(1)

    assert run_a.result().status is TerminalStatus.TIMED_OUT
    assert client_a.interrupts == 1
    client_b.release.set()
    terminal_b = run_b.result()

    assert terminal_b.status is TerminalStatus.OK, terminal_b.reason
    assert client_b.interrupts == 0


def test_cancelling_one_run_by_closing_its_stream_leaves_the_other_alone(agent_bridge):
    """A caller abandoning A cancels A -- the D10 contract -- and only A."""
    bridge, sdk = agent_bridge

    stream_a = bridge.chat(connection="claude-sub", message="A")
    next(stream_a)  # the receipt
    next(stream_a)  # the init message: run A is live
    client_a = sdk.client(0)
    run_b = Worker(lambda: list(bridge.chat(connection="claude-sub", message="B")))
    run_b.start()
    client_b = sdk.client(1)

    stream_a.close()
    assert client_a.interrupts >= 1
    client_b.release.set()
    terminal_b = run_b.result()

    assert terminal_b.status is TerminalStatus.OK, terminal_b.reason
    assert client_b.interrupts == 0


# --- openai-sdk on codex exec: one process per run --------------------------------

THREAD_STARTED = {"type": "thread.started", "thread_id": "00000000-feed-7000-0000-000000000001"}
TURN_STARTED = {"type": "turn.started"}
AGENT_MESSAGE = {
    "type": "item.completed",
    "item": {"id": "i", "type": "agent_message", "text": "OK"},
}
TURN_COMPLETED = {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1}}


class _Stdin:
    def write(self, text: str) -> int:
        return len(text)

    def close(self) -> None:
        return None


class GatedProc:
    """A ``codex exec`` child whose JSONL finishes only when released."""

    def __init__(self) -> None:
        self.stdin = _Stdin()
        self.release = threading.Event()
        self.killed = False
        self.exited = False
        self.stdout = self._lines()

    def _lines(self):
        for payload in (THREAD_STARTED, TURN_STARTED):
            yield json.dumps(payload) + "\n"
        self.release.wait(PATIENCE)
        if self.killed:
            return
        for payload in (AGENT_MESSAGE, TURN_COMPLETED):
            yield json.dumps(payload) + "\n"

    def wait(self, timeout: float | None = None) -> int:
        self.exited = True
        return -9 if self.killed else 0

    def poll(self) -> int | None:
        if self.killed:
            return -9
        return 0 if self.exited else None

    def kill(self) -> None:
        self.killed = True
        self.release.set()


class GatedSpawn:
    def __init__(self) -> None:
        self.procs: list[GatedProc] = []
        self.spawned = threading.Condition()

    def __call__(self, argv, env, cwd) -> GatedProc:
        proc = GatedProc()
        with self.spawned:
            self.procs.append(proc)
            self.spawned.notify_all()
        return proc

    def proc(self, index: int) -> GatedProc:
        with self.spawned:
            assert self.spawned.wait_for(lambda: len(self.procs) > index, PATIENCE)
            return self.procs[index]


class OfflineCodexAdapter(OpenAIAdapter):
    @classmethod
    def is_available(cls) -> bool:
        return True

    def preflight(self, request):  # type: ignore[override]
        return _offline_receipt(request)


def test_a_finished_codex_run_does_not_cancel_one_still_in_flight(store):
    connection = Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    store.add(connection)
    spawn = GatedSpawn()
    adapter = OfflineCodexAdapter(codex_bin="codex", spawn=spawn, mcp_list=lambda b, e: "[]")
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.OPENAI_SDK: adapter},
        env={"PATH": "/usr/bin"},
    )

    def chat(message: str) -> list[Any]:
        return list(
            bridge.chat(connection="codex-sub", message=message, options={"transport": "exec"})
        )

    # B is launched first and held mid-turn; A then runs to its terminal.
    run_b = Worker(lambda: chat("B"))
    run_b.start()
    proc_b = spawn.proc(0)
    run_a = Worker(lambda: chat("A"))
    run_a.start()
    proc_a = spawn.proc(1)
    proc_a.release.set()
    assert run_a.result().status is TerminalStatus.OK

    proc_b.release.set()
    terminal_b = run_b.result()
    assert terminal_b.status is TerminalStatus.OK, terminal_b.reason
    assert proc_b.killed is False


# --- the four API runtimes: a flag checked at every tool-round boundary -----------


@dataclass(frozen=True)
class ApiCase:
    module: str
    tool_round: Any
    connection_kwargs: Any = None


def _verified() -> dict[str, Any]:
    module = importlib.import_module("test_adapter_openai_compatible")
    return {"verified": module.all_verified()}


API_CASES = [
    ApiCase("test_adapter_anthropic_api", lambda m: m.tool_round("call_1", "look_up", {})),
    ApiCase("test_adapter_openai_api", lambda m: m.tool_round("call_1", "look_up", {})),
    ApiCase("test_adapter_google_api", lambda m: m.tool_round("look_up", {}, call_id="call_1")),
    ApiCase(
        "test_adapter_openai_compatible",
        lambda m: m.tool_round("call_1", "look_up", {}),
        _verified,
    ),
]

API_IDS = [case.module.removeprefix("test_adapter_") for case in API_CASES]


class SharedClient:
    """Hand every run the same fake client, so the rounds are consumed in call order.

    Whether an adapter builds a client per run or caches one is its own business;
    sharing one here fixes the order the scripted rounds are handed out in, which
    is what lets the test say which run receives which answer.
    """

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._client: Any = None
        self._lock = threading.Lock()

    def __call__(self, **kwargs: Any) -> Any:
        with self._lock:
            if self._client is None:
                self._client = self._factory(**kwargs)
            return self._client


@pytest.mark.parametrize("case", API_CASES, ids=API_IDS)
def test_a_finished_api_run_does_not_cancel_a_tool_loop_in_flight(store, case):
    """B is inside its tool handler when A ends; B's next round must still run."""
    module = importlib.import_module(case.module)
    if module.NEEDS_SDK.args[0]:  # a bridge refuses a runtime whose SDK is absent
        pytest.skip(module.NEEDS_SDK.kwargs["reason"])

    in_tool = threading.Event()
    a_done = threading.Event()

    def look_up(arguments: Any) -> str:
        in_tool.set()
        assert a_done.wait(PATIENCE)
        return "found"

    tool = ToolDef(
        name="look_up",
        description="Look something up.",
        parameters={"type": "object", "properties": {}},
        handler=look_up,
    )
    # In call order: B's first round asks for the tool, A's only round answers,
    # then B's second round answers.
    factory = module.Factory(
        [case.tool_round(module), module.text_round("A"), module.text_round("B")]
    )
    kwargs = case.connection_kwargs() if case.connection_kwargs else {}
    conn = module.connection(**kwargs)
    bridge, _ = module.bridge_for(conn, SharedClient(factory), store)

    run_b = Worker(lambda: list(bridge.chat(connection=conn.name, message="B", tools=[tool])))
    run_b.start()
    assert in_tool.wait(PATIENCE)
    terminal_a = terminal_of(list(bridge.chat(connection=conn.name, message="A")))
    assert terminal_a.status is TerminalStatus.OK, terminal_a.reason
    a_done.set()
    terminal_b = run_b.result()

    assert terminal_b.status is TerminalStatus.OK, terminal_b.reason


@pytest.mark.parametrize("case", API_CASES, ids=API_IDS)
def test_abandoning_one_api_run_does_not_cancel_a_tool_loop_in_flight(store, case):
    """A caller walking away from A cancels A (D10) -- and B's loop carries on."""
    module = importlib.import_module(case.module)
    if module.NEEDS_SDK.args[0]:
        pytest.skip(module.NEEDS_SDK.kwargs["reason"])

    in_tool = threading.Event()
    a_gone = threading.Event()

    def look_up(arguments: Any) -> str:
        in_tool.set()
        assert a_gone.wait(PATIENCE)
        return "found"

    tool = ToolDef(
        name="look_up",
        description="Look something up.",
        parameters={"type": "object", "properties": {}},
        handler=look_up,
    )
    factory = module.Factory(
        [case.tool_round(module), module.text_round("A"), module.text_round("B")]
    )
    kwargs = case.connection_kwargs() if case.connection_kwargs else {}
    conn = module.connection(**kwargs)
    bridge, _ = module.bridge_for(conn, SharedClient(factory), store)

    run_b = Worker(lambda: list(bridge.chat(connection=conn.name, message="B", tools=[tool])))
    run_b.start()
    assert in_tool.wait(PATIENCE)
    stream_a = bridge.chat(connection=conn.name, message="A")
    next(stream_a)  # the receipt
    next(stream_a)  # A's first streamed event: run A is live
    stream_a.close()
    a_gone.set()
    terminal_b = run_b.result()

    assert terminal_b.status is TerminalStatus.OK, terminal_b.reason


def test_abandoning_one_codex_run_does_not_cancel_another(store):
    connection = Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    store.add(connection)
    spawn = GatedSpawn()
    adapter = OfflineCodexAdapter(codex_bin="codex", spawn=spawn, mcp_list=lambda b, e: "[]")
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.OPENAI_SDK: adapter},
        env={"PATH": "/usr/bin"},
    )
    options = {"transport": "exec"}

    run_b = Worker(
        lambda: list(bridge.chat(connection="codex-sub", message="B", options=options))
    )
    run_b.start()
    proc_b = spawn.proc(0)
    stream_a = bridge.chat(connection="codex-sub", message="A", options=options)
    next(stream_a)  # the receipt
    next(stream_a)  # thread.started: run A's process is live
    proc_a = spawn.proc(1)
    stream_a.close()
    assert proc_a.killed is True

    proc_b.release.set()
    terminal_b = run_b.result()
    assert terminal_b.status is TerminalStatus.OK, terminal_b.reason
    assert proc_b.killed is False


def test_adapter_cancel_still_reaches_a_run_started_through_it(claude_sub):
    """``adapter.cancel()`` with the adapter in hand keeps its D10 meaning."""
    sdk = GatedSdk()
    adapter = OfflineAnthropicAdapter(sdk)
    from modelpass.adapters.base import RunRequest
    from modelpass.preflight import plan_launch
    from modelpass.types import Message, Role

    request = RunRequest(
        connection=claude_sub,
        messages=(Message(Role.USER, "hi"),),
        plan=plan_launch(claude_sub, {}),
    )
    run = Worker(lambda: list(adapter.run(request)))
    run.start()
    client = sdk.client(0)
    adapter.cancel()
    terminal = run.result()

    assert terminal.status is TerminalStatus.CANCELLED
    assert client.interrupts == 1


# --- sessions: a bounded turn is cancelled through its own handle -----------------


class _StallingTurnHandle:
    """A session handle whose turn hangs until its own cancel releases it."""

    def __init__(self) -> None:
        self.cancels = 0
        self.released = threading.Event()
        self.id = None

    def send(self, message: str, *, effort: str | None = None):
        from modelpass.adapters.base import RunStream

        def turn():
            self.released.wait(PATIENCE)
            yield TerminalEvent(
                status=TerminalStatus.CANCELLED,
                connection="",
                runtime=Runtime.ANTHROPIC_SDK,
                auth_mode=AuthMode.SUBSCRIPTION,
                reason="interrupted",
            )

        def cancel() -> None:
            self.cancels += 1
            self.released.set()

        return RunStream(turn(), cancel)

    def history(self):
        return ()

    def close(self) -> None:
        return None


def test_a_timed_out_session_turn_cancels_its_turn_not_the_shared_adapter(tmp_path, claude_sub):
    """The adapter behind a session also carries every stateless call on the runtime.

    So the turn's bound must reach the turn -- through the handle's own cancel --
    and ``adapter.cancel()``, which on a real adapter cancels those stateless
    calls, must not be called at all.
    """
    from modelpass.testing import FakeSessionAdapter, fake_session_bridge

    handle = _StallingTurnHandle()

    class Adapter(FakeSessionAdapter):
        def open_session(self, request):
            self.session_requests.append(request)
            return handle

    adapter = Adapter()
    bridge, _, _ = fake_session_bridge(
        connections=[claude_sub], adapter=adapter, home=tmp_path / "modelpass"
    )
    with bridge.new_chat(connection="claude-sub") as session:
        events = list(session.send("hi", timeout=Timeout(total=0.2)))

    assert terminal_of(events).status is TerminalStatus.TIMED_OUT
    assert handle.cancels == 1
    assert adapter.cancelled == 0
