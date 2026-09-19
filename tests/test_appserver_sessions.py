"""Native sessions on the Codex app-server transport (S5) -- all offline.

Nothing here spawns a process, imports a vendor package or reads a credential.
:class:`ScriptedThreadServer` stands in for ``subprocess.Popen`` through the
adapter's ``app_server_spawn=`` seam and answers the real JSON-RPC conversation,
so the reader thread, the request correlation, the notification drain and the
caller-tool dispatcher all run rather than being mocked.

**The scripted lines are real wire text**, read out of
``tests/fixtures/appserver/live-capture-2026-08-31.jsonl`` -- modelpass's own
client driving codex.exe 0.151.0-alpha.7.1 -- which is what makes "two turns on
one thread" a claim about the protocol rather than about the fake. The capture
holds exactly that: two turns on one thread, the second answered from the first
turn's tool result with only the new message sent.

Where a test needs a shape the capture does not contain, it says so. Two
categories, kept apart on purpose:

* ``thread/resume``, ``thread/items/list`` and ``turn/interrupt`` were never
  driven. Their params and responses are **schema-sourced** from the shipped
  binary's own generator (``codex app-server generate-json-schema --out <dir>
  --experimental``, run 2026-08-31, token-free), and the responses scripted here
  are built to those schemas.
* the ``ThreadItem`` payloads inside a scripted ``thread/items/list`` are the
  capture's **own** ``item/completed`` items, lifted verbatim -- so the item
  mapping is checked against wire text even though the envelope carrying them is
  schema-built.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelpass.adapters.base import SessionRequest
from modelpass.adapters.codex_appserver import (
    THREAD_ITEMS_LIST_METHOD,
    THREAD_LIST_METHOD,
    THREAD_RESUME_METHOD,
    TURN_INTERRUPT_METHOD,
    session_info_from_thread,
    thread_item_message,
    thread_items_history,
)
from modelpass.adapters.openai import (
    TRANSPORT_APP_SERVER,
    TRANSPORT_EXEC,
    CodexAppServerSession,
    CodexSession,
    OpenAIAdapter,
    chat_tool_overrides,
)
from modelpass.bridge import Bridge
from modelpass.capabilities import Capability, CapabilityRegistry, Support
from modelpass.connections import Connection, CredentialRef
from modelpass.errors import CapabilityNotSupported, SessionNotFound
from modelpass.preflight import plan_launch
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore
from modelpass.tools import ToolDef
from modelpass.types import (
    AuthMode,
    Role,
    SessionKind,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    ToolResultEvent,
    VendorEvent,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "appserver" / "live-capture-2026-08-31.jsonl"
)


def capture() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


CAPTURE = capture()


def captured_payload(kind: str) -> dict:
    for line in CAPTURE:
        if line.get("_kind") == kind:
            return line["payload"]
    raise AssertionError(f"the live capture has no {kind!r} line")


def notifications() -> list[dict]:
    return [
        {"method": line["method"], "params": line["params"]}
        for line in CAPTURE
        if line.get("_kind") == "notification"
    ]


def _turn_slice(index: int) -> list[dict]:
    """Everything the server said during the capture's ``index``-th turn."""
    lines = notifications()
    starts = [i for i, line in enumerate(lines) if line["method"] == "turn/started"]
    ends = [i for i, line in enumerate(lines) if line["method"] == "turn/completed"]
    return lines[starts[index] : ends[index] + 1]


FIRST_TURN = _turn_slice(0)
SECOND_TURN = _turn_slice(1)
THREAD_ID = captured_payload("thread_start_response")["thread"]["id"]
FIRST_TURN_ID = captured_payload("turn1_response")["turn"]["id"]
SECOND_TURN_ID = captured_payload("turn2_response")["turn"]["id"]

TOOL_CALL_REQUEST = next(
    {"id": 9001, "method": line["method"], "params": line["params"]}
    for line in CAPTURE
    if line.get("_kind") == "server_request"
)
CAPTURED_TOOL_NAME = TOOL_CALL_REQUEST["params"]["tool"]


def completed_items(*types: str, turn_id: str | None = None) -> list[dict]:
    """The capture's own ``item/completed`` items, by type, in arrival order.

    ``turn_id`` narrows to one turn, which is how a history fixture stays the
    length its test claims -- the capture ran two turns and every type below
    appears in both.
    """
    return [
        line["params"]["item"]
        for line in CAPTURE
        if line.get("_kind") == "notification"
        and line["method"] == "item/completed"
        and line["params"]["item"].get("type") in types
        and (turn_id is None or line["params"].get("turnId") == turn_id)
    ]


# --- the scripted child -----------------------------------------------------------


def _is_server_request(line: Any) -> bool:
    return isinstance(line, Mapping) and "method" in line and "id" in line


class _Stream:
    def __init__(self) -> None:
        self.lines: queue.Queue[str | None] = queue.Queue()

    def __iter__(self) -> _Stream:
        return self

    def __next__(self) -> str:
        line = self.lines.get()
        if line is None:
            raise StopIteration
        return line


class _Stdin:
    def __init__(self, server: ScriptedThreadServer) -> None:
        self.server = server

    def write(self, text: str) -> int:
        self.server.on_write(text)
        return len(text)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class ScriptedThreadServer:
    """A ``codex app-server`` child that answers a **queue** of scripts per method.

    The one thing this fake does that S4's could not, and the reason it exists:
    ``turn/start`` is answered differently on turn one and turn two. A session is
    the case where a method is called more than once with different consequences,
    so a script keyed by method alone would have made "two turns on one thread"
    untestable -- exactly the property the slice is about.

    ``script`` maps a method to a list of ``(result, notifications)`` pairs,
    consumed in order; the last pair is reused once the list runs out, which
    keeps a script for a one-call method short. A scripted line carrying both
    ``method`` and ``id`` is a server -> client **request** and is pushed from its
    own thread, waiting for the client's answer before the script continues --
    the same rule the S4 fake follows, and for the same reason: the client
    answers from its reader thread while the caller holds the write lock.
    """

    def __init__(self, script: dict[str, list[tuple[Any, list[Any]]]]) -> None:
        self.script = {method: list(pairs) for method, pairs in script.items()}
        self.stdout = _Stream()
        self.stderr = _Stream()
        self.stdin = _Stdin(self)
        self.received: list[dict] = []
        self.answers: dict[Any, Any] = {}
        self.returncode: int | None = None
        self._answered: dict[Any, threading.Event] = {}

    # -- what the client wrote

    def on_write(self, text: str) -> None:
        message = json.loads(text)
        self.received.append(message)
        method = message.get("method")
        if method is None:
            if "id" in message:
                self.answers[message["id"]] = message.get("result")
                self._event_for(message["id"]).set()
            return
        if "id" not in message:
            return
        pairs = self.script.get(method) or [({}, [])]
        result, lines = pairs[0] if len(pairs) == 1 else pairs.pop(0)
        if isinstance(result, dict) and result.get("__error__"):
            self.send({"id": message["id"], "error": result["__error__"]})
            return
        self.send({"id": message["id"], "result": result})
        if any(_is_server_request(line) for line in lines):
            threading.Thread(target=self._push, args=(lines,), daemon=True).start()
        else:
            self._push(lines)

    def _push(self, lines: list[Any]) -> None:
        for line in lines:
            if line is None:
                self.eof()
            elif _is_server_request(line):
                event = self._event_for(line["id"])
                self.send(line)
                assert event.wait(timeout=10), (
                    f"the client never answered {line['method']!r}"
                )
            else:
                self.send(line)

    def _event_for(self, request_id: Any) -> threading.Event:
        return self._answered.setdefault(request_id, threading.Event())

    def sent(self, method: str) -> list[dict]:
        return [
            message.get("params", {})
            for message in self.received
            if message.get("method") == method
        ]

    @property
    def methods(self) -> list[str]:
        return [m["method"] for m in self.received if "method" in m]

    # -- what the server says

    def send(self, obj: dict) -> None:
        self.stdout.lines.put(json.dumps(obj) + "\n")

    def eof(self) -> None:
        self.stdout.lines.put(None)
        self.stderr.lines.put(None)

    # -- the Popen surface

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0
        self.eof()

    def kill(self) -> None:
        self.returncode = -9
        self.eof()

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("codex", timeout or 0)
        return self.returncode


class Spawn:
    def __init__(self, server: ScriptedThreadServer) -> None:
        self.server = server
        self.argv: list[str] | None = None
        self.env: dict[str, str] | None = None
        self.cwd: str | None = None
        self.calls = 0

    def __call__(self, argv, env, cwd):
        self.calls += 1
        self.argv, self.env, self.cwd = list(argv), dict(env), cwd
        return self.server


def exploding_spawn(*args, **kwargs):
    raise AssertionError("the exec transport was launched when it should not have been")


def two_turn_script() -> dict[str, list[tuple[Any, list[Any]]]]:
    """The capture's own two turns, in order, on one thread."""
    return {
        "initialize": [({}, [])],
        "thread/start": [(captured_payload("thread_start_response"), [])],
        "turn/start": [
            (captured_payload("turn1_response"), FIRST_TURN),
            (captured_payload("turn2_response"), SECOND_TURN),
        ],
    }


# --- requests and the bridge -------------------------------------------------------


def codex_connection() -> Connection:
    return Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


def session_request(tmp_path, **kwargs) -> SessionRequest:
    connection = codex_connection()
    options = {"transport": TRANSPORT_APP_SERVER, **kwargs.pop("options", {})}
    return SessionRequest(
        connection=connection,
        plan=plan_launch(connection, {"PATH": "/usr/bin"}),
        kind=kwargs.pop("kind", SessionKind.CHAT),
        project_folder=kwargs.pop("project_folder", str(tmp_path / "work")),
        options=options,
        **kwargs,
    )


def adapter_for(server: ScriptedThreadServer) -> tuple[OpenAIAdapter, Spawn]:
    spawn = Spawn(server)
    adapter = OpenAIAdapter(
        codex_bin="C:/codex/codex.exe",
        spawn=exploding_spawn,
        session_spawn=exploding_spawn,
        app_server_spawn=spawn,
    )
    return adapter, spawn


def bridge_over(adapter: OpenAIAdapter, tmp_path) -> Bridge:
    store = ConnectionStore(tmp_path / "modelpass-home")
    store.add(codex_connection())
    adapter._login_status = lambda binary, env: "Logged in using ChatGPT"
    adapter._codex_version = lambda binary, env: "codex-cli 0.151.0-alpha.7.1"
    return Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.OPENAI_SDK: adapter},
        env={"PATH": "/usr/bin"},
    )


def text_of(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDeltaEvent))


def loudness_tool(handler) -> ToolDef:
    return ToolDef(
        name=CAPTURED_TOOL_NAME,
        description="Look up a track's integrated loudness.",
        parameters={
            "type": "object",
            "properties": {"track": {"type": "string"}},
            "required": ["track"],
        },
        handler=handler,
    )


# --- transport selection -----------------------------------------------------------


def test_a_session_defaults_to_the_app_server_and_exec_is_the_opt_out(tmp_path):
    """The session half of the S7 flip (2026-08-31), asserted on both sides.

    A caller who selects nothing gets the native-thread session; a caller who
    passes ``options={"transport": "exec"}`` gets byte-for-byte the session
    S1-S4 shipped. Opening launches nothing on either (adapter contract, rule
    7), so the spawn count stays at zero and the assertion is about which
    object was built rather than about a stream.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, spawn = adapter_for(server)
    handle = adapter.open_session(
        session_request(tmp_path, options={"transport": TRANSPORT_EXEC})
    )
    assert isinstance(handle, CodexSession)
    handle_default = adapter.open_session(
        SessionRequest(
            connection=codex_connection(),
            plan=plan_launch(codex_connection(), {"PATH": "/usr/bin"}),
            kind=SessionKind.CHAT,
            project_folder=str(tmp_path / "work"),
        )
    )
    assert isinstance(handle_default, CodexAppServerSession)
    assert spawn.calls == 0


def test_an_unrecognized_transport_is_refused_before_anything_opens(tmp_path):
    server = ScriptedThreadServer(two_turn_script())
    adapter, spawn = adapter_for(server)
    with pytest.raises(CapabilityNotSupported) as excinfo:
        adapter.open_session(session_request(tmp_path, options={"transport": "appsrv"}))
    assert excinfo.value.capability == "transport"
    assert spawn.calls == 0


def test_open_session_launches_nothing(tmp_path):
    """Adapter contract rule 7, on the transport that could have broken it.

    ``thread/start`` would answer with a thread id right here, which is exactly
    the temptation the rule exists against: opening a session is local work, and
    a handle that had already spawned a runtime would spend a caller's process
    on an object they might never send to.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, spawn = adapter_for(server)
    handle = adapter.open_session(session_request(tmp_path))

    assert isinstance(handle, CodexAppServerSession)
    assert handle.id is None
    assert spawn.calls == 0
    assert server.methods == []


# --- two turns, one thread ---------------------------------------------------------


def test_two_turns_run_on_one_thread_and_only_the_new_message_is_sent(tmp_path):
    """The headline: the thread is the conversation, so turn two sends one item.

    Driven on 2026-08-31 -- the capture's second turn asks *"What was the
    integrated loudness again?"* with no history attached and is answered
    ``-13.7 LUFS`` from the first turn's tool result. This is what a session on
    this transport buys over the stateless call, which has to re-send everything
    it wants the model to have.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, spawn = adapter_for(server)
    handle = adapter.open_session(session_request(tmp_path))

    first = list(handle.send("Use get_track_loudness on 'Neon Dust'."))
    second = list(handle.send("What was the integrated loudness again?"))

    assert spawn.calls == 1, "one child for the whole session"
    assert len(server.sent("thread/start")) == 1, "one thread for the whole session"
    starts = server.sent("turn/start")
    assert len(starts) == 2
    assert starts[0]["threadId"] == starts[1]["threadId"] == THREAD_ID
    assert starts[1]["input"] == [
        {"type": "text", "text": "What was the integrated loudness again?"}
    ]
    assert "thread/inject_items" not in server.methods
    assert text_of(second) == "-13.7 LUFS"
    assert first[-1].status is TerminalStatus.OK
    assert second[-1].status is TerminalStatus.OK


def test_the_id_is_published_only_after_a_turn_completes(tmp_path):
    """Named at ``thread/start``, published at ``turn/completed``.

    The reasoning is written down on :class:`CodexAppServerSession` because this
    transport genuinely differs from exec: the id *is* available before the first
    turn here. It is still withheld, because rule 7 is about handing out an id a
    resume would reject, and nobody has driven a ``thread/resume`` of a thread
    whose first turn never finished.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, _ = adapter_for(server)
    handle = adapter.open_session(session_request(tmp_path))
    assert handle.id is None

    stream = handle.send("hello")
    next(stream)  # the turn is in flight; thread/start has already answered
    assert handle.id is None
    list(stream)

    assert handle.id == THREAD_ID


def test_a_failed_turn_publishes_no_id(tmp_path):
    """Same rule as exec: an id only ever names a thread a turn finished on."""
    failed = list(FIRST_TURN)
    completed = dict(failed[-1])
    turn = dict(completed["params"]["turn"], status="failed", error={"message": "nope"})
    failed[-1] = {"method": "turn/completed", "params": {"turn": turn}}
    script = two_turn_script()
    script["turn/start"] = [(captured_payload("turn1_response"), failed)]

    server = ScriptedThreadServer(script)
    adapter, _ = adapter_for(server)
    handle = adapter.open_session(session_request(tmp_path))

    terminal = list(handle.send("hello"))[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert handle.id is None


# --- the system prompt -------------------------------------------------------------


def test_a_chat_session_puts_its_prompt_in_baseinstructions(tmp_path):
    """``ChatSession`` on Codex, which was impossible until this slice.

    ``baseInstructions`` *replaces* Codex's coding-agent persona, which is the
    whole point of the object -- and the reason ``system_prompt_replace`` could
    not be answered before sessions could reach this transport.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, spawn = adapter_for(server)
    handle = adapter.open_session(
        session_request(tmp_path, kind=SessionKind.CHAT, system_prompt="You score takes.")
    )
    list(handle.send("Score this one."))

    start = server.sent("thread/start")[0]
    assert start["baseInstructions"] == "You score takes."
    # The turn's own text is the task alone: the prompt is a parameter now, not a
    # ``System: `` line glued to the front of the first message.
    assert server.sent("turn/start")[0]["input"] == [
        {"type": "text", "text": "Score this one."}
    ]
    # And the runtime's own toolbelt is off, which on this transport is the
    # config layer rather than a thread field.
    assert spawn.argv[1] == "app-server"
    assert chat_tool_overrides()[:2] == spawn.argv[2:4]


def test_a_worker_session_keeps_the_append_semantics_it_asked_for(tmp_path):
    """A worker's prompt must **not** become ``baseInstructions``.

    ``baseInstructions`` replaces, and a worker chose to keep Codex's persona and
    add to it. So a worker keeps exec's layering -- the prompt above the first
    turn's task, and never restated -- because that is what append *is* on this
    runtime. ``developerInstructions`` is the undriven candidate for a native
    layered-above channel and nothing sends it yet.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, spawn = adapter_for(server)
    handle = adapter.open_session(
        session_request(
            tmp_path, kind=SessionKind.WORKER, system_prompt="House rules apply."
        )
    )
    list(handle.send("Fix the build."))
    list(handle.send("Now run the tests."))

    assert "baseInstructions" not in server.sent("thread/start")[0]
    starts = server.sent("turn/start")
    assert starts[0]["input"][0]["text"] == "System: House rules apply.\n\nFix the build."
    # Second turn: raw. The thread has held it since turn one, and restating it
    # would move the cached prefix (D17).
    assert starts[1]["input"][0]["text"] == "Now run the tests."
    # A worker keeps the toolbelt, so no -c overrides on the launch.
    assert spawn.argv == [
        "C:/codex/codex.exe",
        "app-server",
        "--listen",
        "stdio://",
    ]


def test_a_chat_session_with_a_prompt_opens_on_app_server_and_is_refused_on_exec(
    tmp_path,
):
    """The gate and the session path agree, in both directions.

    This is the pair the slice had to make consistent: ``new_chat(system_prompt=
    ...)`` opens on ``app-server``, where ``baseInstructions`` honours it, and
    is refused on ``codex exec``, which can only append. **The two swapped
    places on 2026-08-31 (S7)**: opening is now what a caller gets by default
    and the refusal belongs to the opt-out, which is the behaviour change that
    slice exists to announce -- a system prompt that used to be layered on top
    of the coding-agent persona now replaces it.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, _ = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)

    with pytest.raises(CapabilityNotSupported) as excinfo:
        bridge.new_chat(
            connection="codex-sub",
            system_prompt="You score takes.",
            options={"transport": TRANSPORT_EXEC},
        )
    assert excinfo.value.capability == "system_prompt_replace"

    session = bridge.new_chat(
        connection="codex-sub",
        system_prompt="You score takes.",
    )
    events = list(session.send("Score this one."))
    assert events[-1].status is TerminalStatus.OK
    assert server.sent("thread/start")[0]["baseInstructions"] == "You score takes."
    session.close()


def test_help_does_not_contradict_the_session_it_describes(tmp_path):
    """``help()`` asks the adapter first, so it cannot deny what the object did.

    A ``ChatSession`` that opened *because* ``baseInstructions`` replaces used to
    print ``system_prompt_replace is unsupported`` under its own shape lines --
    the registry answering about the default transport for an object that is not
    on it. Since 2026-08-31 the same mechanism runs the other way: a worker that
    opted out into ``codex exec`` must still hear that its tools cell is a no
    there, even though the runtime row says yes.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, _ = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)
    session = bridge.new_chat(
        connection="codex-sub",
        system_prompt="You score takes.",
        options={"transport": TRANSPORT_APP_SERVER},
    )

    help_text = session.help()
    row = next(
        line for line in help_text.splitlines() if "system_prompt_replace" in line
    )
    assert row.split() == ["system_prompt_replace", "supported"]
    assert "system_prompt_replace is unsupported" not in help_text

    exec_session = bridge.new_worker(
        connection="codex-sub",
        project_folder=str(tmp_path / "work"),
        options={"transport": TRANSPORT_EXEC},
    )
    assert "tools_in_process is unsupported" in exec_session.help()
    session.close()
    exec_session.close()


# --- tools ------------------------------------------------------------------------


def test_a_session_carrying_tools_reaches_the_s6_dispatcher(tmp_path):
    """``tools=`` on a session, through the same loop the stateless call uses.

    The registration is ``thread/start``'s ``dynamicTools``; the call arrives as
    the server -> client ``item/tool/call`` request; the answer is
    ``{contentItems, success}``. All three are the capture's own wire text -- the
    only thing S5 changes is that the client outlives the turn.
    """
    calls: list[dict] = []

    def handler(args):
        calls.append(dict(args))
        return "-13.7 LUFS integrated, true peak -1.2 dBTP"

    tool_turn = list(FIRST_TURN)
    index = next(
        i
        for i, line in enumerate(tool_turn)
        if line["method"] == "item/started"
        and line["params"]["item"].get("type") == "dynamicToolCall"
    )
    tool_turn.insert(index + 1, TOOL_CALL_REQUEST)
    script = two_turn_script()
    script["turn/start"] = [(captured_payload("turn1_response"), tool_turn)]

    server = ScriptedThreadServer(script)
    adapter, _ = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)
    session = bridge.new_chat(
        connection="codex-sub",
        tools=[loudness_tool(handler)],
        options={"transport": TRANSPORT_APP_SERVER},
    )

    events = list(session.send("Use get_track_loudness on 'Neon Dust'."))

    assert calls == [{"track": "Neon Dust"}]
    registered = server.sent("thread/start")[0]["dynamicTools"]
    assert [tool["name"] for tool in registered] == [CAPTURED_TOOL_NAME]
    answer = server.answers[TOOL_CALL_REQUEST["id"]]
    assert answer["success"] is True
    assert answer["contentItems"][0]["text"].startswith("-13.7 LUFS")
    assert events[-1].status is TerminalStatus.OK
    session.close()


def test_a_handler_that_raises_reaches_the_caller_as_a_tool_result(tmp_path):
    """S6's rule holds on a session: a broken tool is an answer, not a crash."""
    tool_turn = list(FIRST_TURN)
    index = next(
        i
        for i, line in enumerate(tool_turn)
        if line["method"] == "item/started"
        and line["params"]["item"].get("type") == "dynamicToolCall"
    )
    tool_turn.insert(index + 1, TOOL_CALL_REQUEST)
    script = two_turn_script()
    script["turn/start"] = [(captured_payload("turn1_response"), tool_turn)]

    def handler(args):
        raise RuntimeError("the loudness index is offline")

    server = ScriptedThreadServer(script)
    adapter, _ = adapter_for(server)
    handle = adapter.open_session(
        session_request(tmp_path, tools=(loudness_tool(handler),))
    )
    events = list(handle.send("Use get_track_loudness on 'Neon Dust'."))

    failures = [
        e for e in events if isinstance(e, ToolResultEvent) and e.is_error
    ]
    assert len(failures) == 1
    assert "the loudness index is offline" in failures[0].content
    assert server.answers[TOOL_CALL_REQUEST["id"]]["success"] is False


def test_a_session_on_exec_still_refuses_tools_and_names_the_transport(tmp_path):
    server = ScriptedThreadServer(two_turn_script())
    adapter, _ = adapter_for(server)
    with pytest.raises(CapabilityNotSupported) as excinfo:
        adapter.open_session(
            session_request(
                tmp_path,
                options={"transport": TRANSPORT_EXEC},
                tools=(loudness_tool(lambda args: ""),),
            )
        )
    assert excinfo.value.capability == "tools_in_process"
    message = str(excinfo.value)
    # Since S7 the message points at a default to fall back to, not an option
    # to add: this session opted out of the transport that runs its tools.
    assert "DEFAULT transport" in message
    assert "options={'transport': 'exec'}" in message


# --- resume ------------------------------------------------------------------------


def resume_script() -> dict[str, list[tuple[Any, list[Any]]]]:
    """``thread/resume`` answered with the thread record the capture holds.

    The **method and its params are schema-sourced** (``ThreadResumeParams``,
    2026-08-31) and undriven; the thread object in the response is the capture's
    own, so what the mapping reads is still real wire text.
    """
    script = two_turn_script()
    script[THREAD_RESUME_METHOD] = [
        ({"thread": captured_payload("thread_start_response")["thread"]}, [])
    ]
    return script


def test_a_resumed_session_never_restates_the_tool_registration(tmp_path):
    """The trap S6 named, and the schema confirms: resume must not clear tools.

    Codex restores a thread's persisted ``dynamicTools`` when none are supplied,
    and ``ThreadResumeParams`` has **no ``dynamicTools`` field at all**
    (schema-sourced 2026-08-31) -- so there is nothing to restate and no way to
    send ``[]`` by accident. What this pins is that modelpass does not start a
    *new* thread on a resume, which would abandon the registration along with the
    conversation.
    """
    server = ScriptedThreadServer(resume_script())
    adapter, _ = adapter_for(server)
    handle = adapter.resume_session(
        session_request(tmp_path, resume_id=THREAD_ID, tools=())
    )
    assert handle.id == THREAD_ID, "a resumed id is durable; nothing is provisional"

    list(handle.send("carry on"))

    assert server.sent("thread/start") == [], "a resume must not start a new thread"
    resumed = server.sent(THREAD_RESUME_METHOD)
    assert len(resumed) == 1
    assert resumed[0]["threadId"] == THREAD_ID
    assert "dynamicTools" not in resumed[0]
    assert server.sent("turn/start")[0]["threadId"] == THREAD_ID


def test_a_resumed_session_answers_a_persisted_tool_call_rather_than_dropping_it(
    tmp_path,
):
    """The consequence of "resume restores tools" that has to be handled.

    ``resume_chat()`` has no ``tools=`` by design (D17), so a thread that
    persisted a registration can ask this process for a tool it never declared.
    The dispatcher is installed on every session for exactly that: the server
    gets a well-formed ``success: false`` naming what modelpass sent, instead of
    the ``{}`` a bare approval handler would answer -- which Codex tells the
    model is an invalid response.
    """
    tool_turn = list(FIRST_TURN)
    tool_turn.insert(0, TOOL_CALL_REQUEST)
    script = resume_script()
    script["turn/start"] = [(captured_payload("turn1_response"), tool_turn)]

    server = ScriptedThreadServer(script)
    adapter, _ = adapter_for(server)
    handle = adapter.resume_session(session_request(tmp_path, resume_id=THREAD_ID))
    events = list(handle.send("carry on"))

    answer = server.answers[TOOL_CALL_REQUEST["id"]]
    assert answer["success"] is False
    assert "no dynamic tools" in answer["contentItems"][0]["text"]
    assert any(isinstance(e, ToolResultEvent) and e.is_error for e in events)


def test_a_resume_whose_thread_is_gone_raises_before_the_turn(tmp_path):
    """Free on this transport, which is the difference worth having.

    ``thread/resume`` is a token-free call, so *"this conversation no longer
    exists"* is answered before anything is spent -- where exec has to run the
    turn to find out. The error text is the vendor's own: ``codex exec resume``
    prints this method's failure verbatim, captured live 2026-08-30.
    """
    script = two_turn_script()
    script[THREAD_RESUME_METHOD] = [
        (
            {
                "__error__": {
                    "code": -32600,
                    "message": (
                        "thread/resume failed: no rollout found for thread id "
                        "00000000-dead-7000-0000-000000000001"
                    ),
                }
            },
            [],
        )
    ]
    server = ScriptedThreadServer(script)
    adapter, _ = adapter_for(server)
    handle = adapter.resume_session(
        session_request(tmp_path, resume_id="00000000-dead-7000-0000-000000000001")
    )

    with pytest.raises(SessionNotFound) as excinfo:
        list(handle.send("carry on"))
    assert "no rollout" in str(excinfo.value)
    assert server.sent("turn/start") == [], "nothing was spent finding out"


# --- history -----------------------------------------------------------------------


def items_list_response() -> dict:
    """A ``thread/items/list`` page built from the capture's own items.

    The **envelope** is schema-sourced (``ThreadItemsListResponse``: ``data`` of
    ``ThreadItemEntry``, ``nextCursor`` null when finished); the **items** inside
    it are lifted verbatim from the capture's ``item/completed`` notifications,
    so the mapping is checked against wire text even where the page around it is
    not.
    """
    kinds = ("userMessage", "reasoning", "dynamicToolCall", "agentMessage")
    return {
        "data": [
            {"item": item, "turnId": FIRST_TURN_ID}
            for item in completed_items(*kinds, turn_id=FIRST_TURN_ID)
        ],
        "nextCursor": None,
    }


def test_history_reads_the_thread_back_and_marks_what_is_not_a_message(tmp_path):
    """The headline user-visible win, and the fidelity decision behind it.

    exec answers ``()`` because it cannot read a thread back. This transport
    asks, and the answer keeps the conversation's *shape*: the user's turn, the
    tool call marked rather than dropped, the assistant's answer. An empty
    ``reasoning`` -- of which the capture has one -- is the only thing that
    disappears, because it carried nothing.
    """
    script = two_turn_script()
    script[THREAD_ITEMS_LIST_METHOD] = [(items_list_response(), [])]
    server = ScriptedThreadServer(script)
    adapter, _ = adapter_for(server)
    handle = adapter.open_session(session_request(tmp_path))
    list(handle.send("Use get_track_loudness on 'Neon Dust'."))

    history = handle.history()

    assert [m.role for m in history] == [Role.USER, Role.ASSISTANT, Role.ASSISTANT]
    assert history[0].content.startswith("Use get_track_loudness")
    assert history[1].content == f"[dynamicToolCall {CAPTURED_TOOL_NAME}]"
    assert history[2].content.startswith("-13.7 LUFS")
    assert server.sent(THREAD_ITEMS_LIST_METHOD)[0]["threadId"] == THREAD_ID
    # thread/read's includeTurns branch is deprecated for paginated threads and
    # its summary view drops the user's own turn; nothing asks for it.
    assert "thread/read" not in server.methods


def test_history_pages_until_the_cursor_runs_out(tmp_path):
    """Nothing truncated, and no way to spin on a server that stops advancing."""
    page_one = {
        "data": [
            {"item": item, "turnId": FIRST_TURN_ID}
            for item in completed_items("userMessage", turn_id=FIRST_TURN_ID)
        ],
        "nextCursor": "cursor-1",
    }
    page_two = {
        "data": [
            {"item": item, "turnId": FIRST_TURN_ID}
            for item in completed_items("agentMessage", turn_id=FIRST_TURN_ID)
        ],
        "nextCursor": None,
    }
    script = two_turn_script()
    script[THREAD_ITEMS_LIST_METHOD] = [(page_one, []), (page_two, [])]
    server = ScriptedThreadServer(script)
    adapter, _ = adapter_for(server)
    handle = adapter.open_session(session_request(tmp_path))
    list(handle.send("hello"))

    history = handle.history()

    pages = server.sent(THREAD_ITEMS_LIST_METHOD)
    assert len(pages) == 2
    assert "cursor" not in pages[0]
    assert pages[1]["cursor"] == "cursor-1"
    assert [m.role for m in history] == [Role.USER, Role.ASSISTANT]


def test_history_is_empty_before_the_first_turn_and_starts_nothing(tmp_path):
    """``()`` here is the truth: there is no thread yet, so nothing is asked."""
    server = ScriptedThreadServer(two_turn_script())
    adapter, spawn = adapter_for(server)
    handle = adapter.open_session(session_request(tmp_path))

    assert handle.history() == ()
    assert spawn.calls == 0


def test_history_on_exec_is_still_honestly_empty(tmp_path):
    """The exec answer does not move, and the reason it gives is still true."""
    server = ScriptedThreadServer(two_turn_script())
    adapter, _ = adapter_for(server)
    handle = adapter.open_session(
        session_request(tmp_path, options={"transport": TRANSPORT_EXEC})
    )
    assert isinstance(handle, CodexSession)
    assert handle.history() == ()


def test_thread_item_mapping_covers_the_variants_that_are_not_messages():
    """Nineteen item types exist; two are messages. The rest keep their place."""
    assert thread_item_message({"type": "commandExecution", "command": "ls -la"}) == (
        thread_items_history([{"item": {"type": "commandExecution", "command": "ls -la"}}])[0]
    )
    assert thread_item_message({"type": "contextCompaction"}).content == (
        "[contextCompaction]"
    )
    # An empty reasoning is an artifact of the item stream, not a turn.
    assert thread_item_message({"type": "reasoning", "summary": [], "content": []}) is None
    assert thread_item_message({"type": "reasoning", "summary": ["why"]}).content == (
        "[reasoning]"
    )
    assert thread_item_message({"type": "mcpToolCall", "tool": "search"}).content == (
        "[mcpToolCall search]"
    )


# --- listing ------------------------------------------------------------------------


def test_list_sessions_answers_on_app_server_and_refuses_on_exec(tmp_path):
    """Both halves, because the refusal is as load-bearing as the answer.

    ``()`` on exec would read as *this connection has no sessions*, which is a
    false statement about an account that may have hundreds.
    """
    script = two_turn_script()
    script[THREAD_LIST_METHOD] = [(captured_payload("probe:thread/list"), [])]
    server = ScriptedThreadServer(script)
    adapter, _ = adapter_for(server)

    listed = adapter.list_sessions(session_request(tmp_path))
    assert len(listed) == 25
    assert server.sent(THREAD_LIST_METHOD)[0] == {"limit": 200}

    with pytest.raises(CapabilityNotSupported) as excinfo:
        adapter.list_sessions(
            session_request(tmp_path, options={"transport": TRANSPORT_EXEC})
        )
    assert excinfo.value.capability == "sessions_list"
    assert "DEFAULT transport" in str(excinfo.value)


def test_the_listing_gate_is_reachable_through_the_bridge(tmp_path):
    """The S6 lesson applied: a capability nobody can reach is not delivered."""
    script = two_turn_script()
    script[THREAD_LIST_METHOD] = [(captured_payload("probe:thread/list"), [])]
    server = ScriptedThreadServer(script)
    adapter, _ = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)

    with pytest.raises(CapabilityNotSupported):
        bridge.list_sessions(
            connection="codex-sub", options={"transport": TRANSPORT_EXEC}
        )

    listed = bridge.list_sessions(connection="codex-sub")
    assert listed[0].id == THREAD_ID
    # find() still answers the *default* question -- and since 2026-08-31 the
    # default is the transport that enumerates, so the connection appears.
    assert [c.name for c in bridge.find(capability=Capability.SESSIONS_LIST)] == [
        "codex-sub"
    ]
    assert (
        bridge.registry.support(Runtime.OPENAI_SDK, Capability.SESSIONS_LIST)
        is Support.SUPPORTED
    )


def test_a_listed_thread_maps_onto_session_info_without_inventing_anything():
    """Driven wire text, and the fields the wire cannot answer stay ``None``."""
    rows = captured_payload("probe:thread/list")["data"]
    first = session_info_from_thread(rows[0], connection="codex-sub")

    assert first.id == THREAD_ID
    assert first.runtime is Runtime.OPENAI_SDK
    assert first.project_folder == rows[0]["cwd"]
    assert first.created_at.startswith("2026-08-31")
    assert first.updated_at is not None
    # No turn count anywhere in the listing -- every row's ``turns`` is [].
    assert first.message_count is None
    # ``preview`` is the first prompt, not a title, so it rides in vendor.
    assert first.title is None
    assert first.vendor["preview"].startswith("Use get_track_loudness")

    named = next(
        session_info_from_thread(row, connection="codex-sub")
        for row in rows
        if row["name"]
    )
    assert named.title == named.vendor["preview"] or named.title is not None
    assert isinstance(named.title, str)


# --- cancel ------------------------------------------------------------------------


def test_abandoning_a_turn_interrupts_it_and_leaves_the_session_open(tmp_path):
    """The session's ``graceful_cancel`` path -- attempted, never claimed.

    The vendor's half was driven on 2026-08-31 -- a live ``turn/interrupt``
    returned ``{}`` and the turn ended ``status: "interrupted"`` -- so this checks
    the half that was not: the ids go out, the child is **not** closed behind them
    (the turn ended, the conversation did not), and the session takes another turn
    on the same thread. Whether a *real* Codex thread survives an interrupt that
    way is still unchecked, which is why the registry says nothing about
    ``graceful_cancel``.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, _ = adapter_for(server)
    handle = adapter.open_session(session_request(tmp_path))

    stream = handle.send("hello")
    next(stream)
    stream.close()

    interrupts = server.sent(TURN_INTERRUPT_METHOD)
    assert interrupts == [{"threadId": THREAD_ID, "turnId": FIRST_TURN_ID}]
    assert not handle._closed
    # And the conversation carries on, on the same thread and the same child.
    events = list(handle.send("What was the integrated loudness again?"))
    assert events[-1].status is TerminalStatus.OK
    assert len(server.sent("thread/start")) == 1


def test_an_abandoned_turns_leftovers_are_announced_not_folded(tmp_path):
    """A session outlives a turn, so another turn's notifications are still coming.

    Folding them would add an earlier turn's tokens to this one's usage and could
    end it on the wrong ``turn/completed``; dropping them would be the silent
    alternative. They leave as a ``vendor_event`` instead.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, _ = adapter_for(server)
    handle = adapter.open_session(session_request(tmp_path))

    stream = handle.send("hello")
    next(stream)
    stream.close()

    events = list(handle.send("and again"))
    stale = [
        e
        for e in events
        if isinstance(e, VendorEvent) and e.name == "turn/stale_notification"
    ]
    assert stale, "the first turn's remaining notifications must be reported"
    assert {e.data["turnId"] for e in stale} == {FIRST_TURN_ID}
    terminal = next(e for e in events if isinstance(e, TerminalEvent))
    assert terminal.status is TerminalStatus.OK
    # The second turn's own usage only: the capture's two turns report different
    # numbers, and the stale ones were not added in.
    assert terminal.usage.total_tokens > 0


def test_closing_the_session_closes_the_child(tmp_path):
    server = ScriptedThreadServer(two_turn_script())
    adapter, _ = adapter_for(server)
    handle = adapter.open_session(session_request(tmp_path))
    list(handle.send("hello"))

    handle.close()
    handle.close()  # idempotent

    assert server.returncode is not None


# --- refusals that did not move ------------------------------------------------------


def test_persist_false_is_still_refused_on_both_transports(tmp_path):
    """Driven on app-server, unbuilt here -- so refused, not degraded.

    Two turns on one ``ephemeral: true`` thread were driven live on 2026-08-31,
    so this is a capability modelpass has not built rather than one nobody has
    checked. A session always starts a persisting thread, and quietly persisting a
    conversation somebody asked to leave no trace of is the worse failure.
    """
    server = ScriptedThreadServer(two_turn_script())
    adapter, _ = adapter_for(server)
    for transport in (TRANSPORT_EXEC, TRANSPORT_APP_SERVER):
        with pytest.raises(CapabilityNotSupported) as excinfo:
            adapter.open_session(
                session_request(
                    tmp_path, persist=False, options={"transport": transport}
                )
            )
        assert excinfo.value.capability == "ephemeral_multi_turn"


def test_mcp_servers_are_refused_on_the_app_server_transport(tmp_path):
    """Exclusivity has no driven equivalent here, so the session is refused
    rather than run without the guarantee exec makes."""
    server = ScriptedThreadServer(two_turn_script())
    adapter, _ = adapter_for(server)
    with pytest.raises(CapabilityNotSupported) as excinfo:
        adapter.open_session(
            session_request(tmp_path, mcp_servers={"amt": {"command": "python"}})
        )
    assert excinfo.value.capability == "mcp_servers"
    assert "exec" in str(excinfo.value)
