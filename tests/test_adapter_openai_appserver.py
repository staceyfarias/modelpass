"""The stateless ``run()`` on the Codex app-server transport (S4) -- all offline.

Nothing here spawns a process, imports a vendor package or reads a credential:
:class:`ScriptedAppServer` stands in for ``subprocess.Popen`` through the
adapter's ``app_server_spawn=`` seam and answers the real JSON-RPC conversation,
so the reader thread, the request correlation and the notification drain are all
exercised rather than mocked.

**The scripted lines are real wire text.** They are read out of
``tests/fixtures/appserver/live-capture-2026-08-31.jsonl``, the raw recording of
modelpass's own client driving codex.exe 0.151.0-alpha.7.1 on 2026-08-31, so a
test cannot pass against a shape the server never sent. Where a test needs a
condition the capture does not contain -- a failed turn, an interrupted one, a
child that dies mid-turn -- it says so and builds the notification from the
protocol's own ``TurnStatus`` enum, which is schema-sourced.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest

from modelpass.adapters.anthropic import AnthropicAdapter
from modelpass.adapters.base import RunRequest
from modelpass.adapters.codex_appserver import (
    DYNAMIC_TOOL_CALL_METHOD,
    check_dynamic_tool_names,
    dynamic_tools_param,
)
from modelpass.adapters.openai import (
    _TRANSCRIPT_PREAMBLE,
    DEFAULT_TRANSPORT,
    INJECT_ITEMS_METHOD,
    TRANSPORT_APP_SERVER,
    TRANSPORT_EXEC,
    CodexAppServerSession,
    OpenAIAdapter,
    app_server_base_instructions,
    app_server_history_items,
    app_server_input_items,
    app_server_turn_input,
    chat_tool_overrides,
    inject_items_params,
    interrupt_app_server_turn,
    render_prompt,
    resolve_transport,
)
from modelpass.bridge import Bridge
from modelpass.capabilities import Capability, CapabilityRegistry, Support
from modelpass.connections import Connection, CredentialRef
from modelpass.errors import CapabilityNotSupported, InvalidTool, VendorRunFailed
from modelpass.preflight import plan_launch
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore
from modelpass.tools import ToolDef
from modelpass.types import AuthMode, Message, Role, TerminalStatus, ToolResultEvent

FIXTURE = (
    Path(__file__).parent / "fixtures" / "appserver" / "live-capture-2026-08-31.jsonl"
)


def capture() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def captured_notifications(*methods: str) -> list[dict]:
    """The recorded notifications, in arrival order, filtered by method.

    Returned as bare ``{"method", "params"}`` wire objects, with the capture's
    ``_kind`` envelope stripped -- that field is the recorder's, not the
    server's.
    """
    kept = []
    for line in capture():
        if line.get("_kind") != "notification":
            continue
        if methods and line["method"] not in methods:
            continue
        kept.append({"method": line["method"], "params": line["params"]})
    return kept


def captured_payload(kind: str) -> dict:
    """One recorded *response* payload, by the recorder's ``_kind`` label."""
    for line in capture():
        if line.get("_kind") == kind:
            return line["payload"]
    raise AssertionError(f"the live capture has no {kind!r} line")


def _first_turn() -> list[dict]:
    """Everything the server said during the capture's first turn, verbatim.

    From ``turn/started`` through ``turn/completed`` inclusive: the tool call and
    its result, seventeen agent-message deltas, the completed message, two usage
    updates, the rate-limit notifications that arrive unprompted, and the
    vendor's own terminal.
    """
    notifications = captured_notifications()
    methods = [n["method"] for n in notifications]
    return notifications[methods.index("turn/started") : methods.index("turn/completed") + 1]


FIRST_TURN_NOTIFICATIONS = _first_turn()


# --- the scripted child -----------------------------------------------------------


def _is_server_request(line: Any) -> bool:
    """A scripted line that is a request *to* the client: method **and** id."""
    return isinstance(line, Mapping) and "method" in line and "id" in line


class _Stream:
    """A blocking line iterator fed from a queue; ``None`` is EOF."""

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
    def __init__(self, server: ScriptedAppServer) -> None:
        self.server = server
        self.closed = False

    def write(self, text: str) -> int:
        if self.closed:
            raise ValueError("write to closed stdin")
        self.server.on_write(text)
        return len(text)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class ScriptedAppServer:
    """A ``codex app-server`` child that answers requests from a script.

    ``script`` maps a method name to ``(result, notifications)``: the result is
    the JSON-RPC response, and the notifications are pushed onto stdout straight
    after it, which is how the real server behaves -- ``turn/start`` answers
    ``inProgress`` immediately and the turn's events follow on the stream.

    ``None`` in a notification list means *the child dies here*, which is the
    only way to script a mid-turn EOF: the run's drain blocks on the stream, so
    the death has to be scheduled by the server rather than by the test.

    A scripted line carrying **both** ``method`` and ``id`` is a server ->
    client **request** -- ``item/tool/call`` is the one that matters -- and it is
    not a notification at all. Two things follow, and both are the real server's
    behaviour rather than test scaffolding:

    * the rest of the script **waits for the client's answer** before it is
      sent, because a real Codex does not complete a dynamic tool item until it
      has the response; without the wait, an ``item/completed`` could overtake
      the answer that produced it and the test would exercise an ordering the
      wire cannot produce; and
    * such a script is pushed from **its own thread**. The client answers a
      server request from its reader thread, under the same stdin write lock the
      caller's ``request()`` is holding while this runs -- pushing the script
      inline would deadlock the fake, not the code under test. Scripts with no
      server request keep the old synchronous path exactly.

    ``answers`` collects what the client replied, keyed by request id, which is
    where a test reads the ``contentItems`` / ``success`` payload back out.
    """

    def __init__(self, script: dict[str, tuple[Any, list[dict]]]) -> None:
        self.script = script
        self.stdout = _Stream()
        self.stderr = _Stream()
        self.stdin = _Stdin(self)
        self.received: list[dict] = []
        self.answers: dict[Any, Any] = {}
        self.returncode: int | None = None
        self.terminated = 0
        self._answered: dict[Any, threading.Event] = {}

    # -- what the client wrote

    def on_write(self, text: str) -> None:
        message = json.loads(text)
        self.received.append(message)
        method = message.get("method")
        if method is None:
            if "id" in message:
                # The client's answer to a server -> client request.
                self.answers[message["id"]] = message.get("result")
                self._event_for(message["id"]).set()
            return
        if "id" not in message:
            return  # a notification from the client
        result, notifications = self.script.get(method, ({}, []))
        self.send({"id": message["id"], "result": result})
        if any(_is_server_request(n) for n in notifications):
            threading.Thread(
                target=self._push, args=(notifications,), daemon=True
            ).start()
        else:
            self._push(notifications)

    def _push(self, notifications: list[dict | None]) -> None:
        for notification in notifications:
            if notification is None:
                self.eof()
            elif _is_server_request(notification):
                event = self._event_for(notification["id"])
                self.send(notification)
                assert event.wait(timeout=10), (
                    f"the client never answered {notification['method']!r}"
                )
            else:
                self.send(notification)

    def _event_for(self, request_id: Any) -> threading.Event:
        return self._answered.setdefault(request_id, threading.Event())

    def params_of(self, method: str) -> dict:
        for message in self.received:
            if message.get("method") == method:
                return message.get("params", {})
        raise AssertionError(f"the client never sent {method!r}: {self.methods}")

    def sent(self, method: str) -> bool:
        return any(message.get("method") == method for message in self.received)

    @property
    def methods(self) -> list[str]:
        return [m["method"] for m in self.received if "method" in m]

    # -- what the server says

    def send(self, obj: dict) -> None:
        self.stdout.lines.put(json.dumps(obj) + "\n")

    def eof(self) -> None:
        self.stdout.lines.put(None)
        self.stderr.lines.put(None)

    def say_on_stderr(self, text: str) -> None:
        self.stderr.lines.put(text)

    # -- the Popen surface

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
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
    def __init__(self, server: ScriptedAppServer) -> None:
        self.server = server
        self.argv: list[str] | None = None
        self.env: dict[str, str] | None = None
        self.cwd: str | None = None
        self.calls = 0

    def __call__(self, argv, env, cwd):
        self.calls += 1
        self.argv, self.env, self.cwd = argv, env, cwd
        return self.server


def exploding_spawn(*args, **kwargs):
    raise AssertionError("the exec transport was launched when it should not have been")


def turn_completed(status: str, error: dict | None = None) -> dict:
    """A ``turn/completed`` with a chosen status.

    ``completed`` is what the capture holds; ``failed`` and ``interrupted`` are
    the other members of the protocol's ``TurnStatus`` enum and are
    schema-sourced -- neither was produced by the live run, which succeeded
    twice.
    """
    turn = dict(captured_payload("turn1_response")["turn"])
    turn.update(status=status, error=error)
    return {
        "method": "turn/completed",
        "params": {"threadId": "01a058a0-050b-7a60-9587-4e9075f9cea0", "turn": turn},
    }


def script_for(notifications: list[dict] | None = None) -> dict:
    """The default happy-path script: the capture's own responses."""
    return {
        "initialize": ({}, []),
        "thread/start": (
            captured_payload("thread_start_response"),
            captured_notifications("thread/started"),
        ),
        "turn/start": (
            captured_payload("turn1_response"),
            FIRST_TURN_NOTIFICATIONS if notifications is None else notifications,
        ),
    }


# --- requests ---------------------------------------------------------------------


def codex_connection() -> Connection:
    return Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


def request_for(messages: tuple[Message, ...] | None = None, **kwargs) -> RunRequest:
    connection = codex_connection()
    options = {"transport": TRANSPORT_APP_SERVER, **kwargs.pop("options", {})}
    return RunRequest(
        connection=connection,
        messages=messages or (Message(role=Role.USER, content="Reply with exactly OK"),),
        plan=plan_launch(connection, {"PATH": "/usr/bin"}),
        options=options,
        **kwargs,
    )


def adapter_for(server: ScriptedAppServer) -> tuple[OpenAIAdapter, Spawn]:
    spawn = Spawn(server)
    adapter = OpenAIAdapter(
        codex_bin="C:/codex/codex.exe",
        spawn=exploding_spawn,
        app_server_spawn=spawn,
    )
    return adapter, spawn


# --- transport selection -----------------------------------------------------------


def test_app_server_is_the_default_and_exec_is_the_opt_out():
    """The flip itself, in one function (S7, 2026-08-31).

    Everything else in this slice is a consequence of these four lines: a
    caller who selects nothing gets ``codex app-server``, and ``exec`` is
    still reachable, still spelled the same way, and still a transport modelpass
    drives -- this is a change of default, not a removal.
    """
    assert resolve_transport({}) == TRANSPORT_APP_SERVER
    assert resolve_transport({"codex_bin": "x"}) == TRANSPORT_APP_SERVER
    assert resolve_transport({"transport": TRANSPORT_EXEC}) == TRANSPORT_EXEC
    assert resolve_transport({"transport": TRANSPORT_APP_SERVER}) == TRANSPORT_APP_SERVER
    assert DEFAULT_TRANSPORT == TRANSPORT_APP_SERVER


def test_an_unrecognized_transport_is_refused_rather_than_defaulted():
    """A typo must not quietly run the other transport.

    The two render a system prompt and a conversation differently, so falling
    back to exec would change what the run *is* and say nothing about it.
    """
    with pytest.raises(CapabilityNotSupported) as excinfo:
        resolve_transport({"transport": "appserver"})
    assert excinfo.value.capability == "transport"
    assert "appserver" in str(excinfo.value)


def test_the_exec_opt_out_never_touches_the_app_server_seam():
    """The exec path is untouched by S7 -- only which one you get for free moved.

    The inverse now holds too and is worth pinning in the same test: a run
    with no options at all must not reach the exec spawn, because that is what
    "the default flipped" means operationally rather than in prose.
    """
    server = ScriptedAppServer(script_for())
    spawn = Spawn(server)
    adapter = OpenAIAdapter(codex_bin="codex", spawn=exploding_spawn, app_server_spawn=spawn)
    connection = codex_connection()

    def request_with(options):
        return RunRequest(
            connection=connection,
            messages=(Message(role=Role.USER, content="hi"),),
            plan=plan_launch(connection, {"PATH": "/usr/bin"}),
            options=options,
        )

    with pytest.raises(AssertionError, match="exec transport was launched"):
        list(adapter.run(request_with({"transport": TRANSPORT_EXEC})))
    assert spawn.calls == 0

    # No options: the app-server seam is what runs, and exec is never touched.
    list(adapter.run(request_with({})))
    assert spawn.calls == 1


def test_the_app_server_launch_carries_the_chat_tool_overrides():
    """D23: a stateless call is chat-shaped, so Codex's own toolbelt is off.

    This assertion used to read as the bare four-element argv, which is what
    the launch was between S4 and D23 -- and that gap is the defect D23
    repairs, not a property worth preserving. The overrides go where
    ``app_server_argv`` puts them: after the subcommand, before ``--listen``.
    """
    server = ScriptedAppServer(script_for())
    adapter, spawn = adapter_for(server)
    list(adapter.run(request_for()))
    assert spawn.argv == [
        "C:/codex/codex.exe",
        "app-server",
        *chat_tool_overrides(),
        "--listen",
        "stdio://",
    ]
    assert spawn.env == {"PATH": "/usr/bin"}


def test_native_tools_opts_back_into_the_runtimes_own_toolbelt():
    """The escape hatch for a caller who wants Codex driving its own shell.

    The whole option surface, asserted together: the key is honoured *and* it
    is in ``option_keys``, because an option the adapter reads but does not
    declare gets a "will be ignored" note stamped on the receipt of the one
    call that honours it.
    """
    server = ScriptedAppServer(script_for())
    adapter, spawn = adapter_for(server)
    list(adapter.run(request_for(options={"native_tools": True})))
    assert spawn.argv == ["C:/codex/codex.exe", "app-server", "--listen", "stdio://"]
    assert "native_tools" in OpenAIAdapter.option_keys
    assert OpenAIAdapter.unknown_option_keys({"native_tools": True}) == ()


def test_the_receipt_names_which_transport_ran():
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    adapter._login_status = lambda binary, env: "Logged in using ChatGPT"
    adapter._codex_version = lambda binary, env: "codex-cli 0.151.0"

    app_server = adapter.preflight(request_for())
    exec_default = adapter.preflight(
        request_for(options={"transport": TRANSPORT_EXEC})
    )

    assert any("codex transport: app-server" in note for note in app_server.notes)
    assert any("baseInstructions" in note for note in app_server.notes)
    assert any("codex transport: exec" in note for note in exec_default.notes)
    # The guard-timing sentence is the one that is not true of both.
    assert any("can fire mid-run" in note for note in app_server.notes)
    assert any("end of the run only" in note for note in exec_default.notes)


# --- the flow ---------------------------------------------------------------------


def test_a_turn_streams_the_captured_answer_and_ends_ok():
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    events = list(adapter.run(request_for()))

    assert server.methods[:3] == ["initialize", "initialized", "thread/start"]
    assert server.methods[3] == "turn/start"
    text = "".join(e.text for e in events if e.type == "text_delta")
    assert text == "-13.7 LUFS integrated, true peak -1.2 dBTP"
    terminal = events[-1]
    assert terminal.type == "terminal"
    assert terminal.status is TerminalStatus.OK
    # The capture's own numbers, with the cache read taken out of the input.
    assert terminal.usage.cached_input_tokens == 12032
    assert terminal.usage.total_tokens == 13083 + 13144


def test_the_turn_is_addressed_to_the_thread_that_thread_start_returned():
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    list(adapter.run(request_for()))
    thread_id = captured_payload("thread_start_response")["thread"]["id"]
    assert server.params_of("turn/start")["threadId"] == thread_id


def test_the_in_progress_turn_start_response_is_not_read_as_completion():
    """``turn/start`` answers before the turn runs; the drain decides the outcome.

    The recorded response carries ``status: "inProgress"``. A path that treated
    it as the verdict would report an unfinished turn as finished -- the
    correction the 2026-08-31 capture forced onto the S2/S3 design note. Here the
    server answers it and then sends *nothing*, and the run is reported as a
    vendor failure rather than a silent success with no answer.
    """
    assert captured_payload("turn1_response")["turn"]["status"] == "inProgress"
    # Answered, then the child dies without ever sending turn/completed.
    server = ScriptedAppServer(
        {**script_for(), "turn/start": (captured_payload("turn1_response"), [None])}
    )
    adapter, _ = adapter_for(server)
    server.say_on_stderr("codex: the model provider hung up\n")

    with pytest.raises(VendorRunFailed, match="stopped before the turn completed"):
        list(adapter.run(request_for()))


def test_the_client_is_closed_on_the_success_path():
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    list(adapter.run(request_for()))
    assert server.terminated == 1
    assert server.stdin.closed
    assert adapter._client is None


def test_the_client_is_closed_when_the_run_raises():
    """A refusal mid-flight still tears the child down.

    ``thread/start`` answering without a thread id is a protocol change the run
    cannot continue through, and the child must not outlive it: a stray
    ``codex app-server`` holds a rollout open and nobody is left to close it.
    """
    server = ScriptedAppServer({**script_for(), "thread/start": ({"thread": {}}, [])})
    adapter, _ = adapter_for(server)
    with pytest.raises(VendorRunFailed, match="without a thread id"):
        list(adapter.run(request_for()))
    assert server.terminated == 1
    assert adapter._client is None


def test_an_abandoned_generator_still_closes_the_child():
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    events = adapter.run(request_for())
    next(events)  # start the turn, then walk away
    events.close()
    assert server.terminated == 1


# --- native shapes replacing the exec workarounds -----------------------------------


def test_the_system_prompt_becomes_base_instructions_not_a_prompt_prefix():
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    list(
        adapter.run(
            request_for(
                (
                    Message(role=Role.SYSTEM, content="Answer in exactly one word."),
                    Message(role=Role.USER, content="Loudness?"),
                )
            )
        )
    )
    assert server.params_of("thread/start")["baseInstructions"] == (
        "Answer in exactly one word."
    )
    assert server.params_of("turn/start")["input"] == [
        {"type": "text", "text": "Loudness?"}
    ]


def test_no_system_message_leaves_the_vendors_own_base_instructions_alone():
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    list(adapter.run(request_for()))
    assert "baseInstructions" not in server.params_of("thread/start")


def test_the_labelled_transcript_is_now_only_the_fallback_shape():
    """S4's rendering, kept and no longer sent on the ordinary path.

    It was right for what was known then: an input item has no role, so a
    previous assistant turn had nowhere role-shaped to live and the label was
    the only way the model could tell its own past answers from the user's.
    ``thread/inject_items`` is the seam that gives those turns a real role, so
    the labels moved to the fallback -- the shape used when a build refuses the
    method. The rendering itself is unchanged.
    """
    items = app_server_input_items(
        (
            Message(role=Role.USER, content="Loudness?"),
            Message(role=Role.ASSISTANT, content="-13.7 LUFS"),
            Message(role=Role.USER, content="True peak?"),
        )
    )
    assert items == [
        {"type": "text", "text": "User: Loudness?"},
        {"type": "text", "text": "Assistant: -13.7 LUFS"},
        {"type": "text", "text": "User: True peak?"},
    ]
    assert all(_TRANSCRIPT_PREAMBLE not in item["text"] for item in items)


def test_a_lone_user_message_is_sent_unlabelled():
    assert app_server_input_items((Message(role=Role.USER, content="hi"),)) == [
        {"type": "text", "text": "hi"}
    ]


def test_several_system_messages_are_joined_rather_than_last_wins():
    messages = (
        Message(role=Role.SYSTEM, content="Be brief."),
        Message(role=Role.SYSTEM, content="Be exact."),
        Message(role=Role.USER, content="hi"),
    )
    assert app_server_base_instructions(messages) == "Be brief.\n\nBe exact."
    assert app_server_input_items(messages) == [{"type": "text", "text": "hi"}]
    assert app_server_turn_input(messages) == [{"type": "text", "text": "hi"}]


# --- history via thread/inject_items (2026-08-31) -----------------------------------
#
# Scripted from ``tests/fixtures/appserver/live-inject-items-2026-08-31.jsonl``,
# the raw recording of the run that drove this: an injected user/assistant
# exchange on an **ephemeral** thread -- the kind the stateless call creates --
# then a question that could only be answered from what was injected. The model
# answered ``-6.4 LUFS``, with no labels anywhere.

INJECT_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "appserver"
    / "live-inject-items-2026-08-31.jsonl"
)


def inject_capture() -> list[dict]:
    return [
        json.loads(line)
        for line in INJECT_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _inject_lines() -> tuple[dict, dict, list[dict]]:
    """The capture's ``thread/started`` payload, its turn, and the turn's stream."""
    lines = inject_capture()
    methods = [line["method"] for line in lines]
    started = lines[methods.index("thread/started")]["params"]
    completed = lines[methods.index("turn/completed")]["params"]
    turn = lines[methods.index("turn/started") : methods.index("turn/completed") + 1]
    return started, completed, turn


INJECT_THREAD, INJECT_COMPLETED, INJECT_TURN_NOTIFICATIONS = _inject_lines()
INJECT_THREAD_ID = INJECT_THREAD["thread"]["id"]

#: The question the live run asked, and the answer only the injected history
#: could produce.
INJECT_QUESTION = "What number did I tell you my song reads? Just the number and unit."
INJECT_ANSWER = "-6.4 LUFS"


def inject_script() -> dict:
    """The capture's happy path: start, inject, ask, answer.

    ``thread/inject_items`` answers ``{}`` because that is what the server
    answers -- success is the absence of an error, and nothing reads a payload
    back off it.
    """
    return {
        "initialize": ({}, []),
        "thread/start": (INJECT_THREAD, []),
        INJECT_ITEMS_METHOD: ({}, []),
        "turn/start": (
            {"turn": {"id": INJECT_COMPLETED["turn"]["id"], "status": "inProgress"}},
            INJECT_TURN_NOTIFICATIONS,
        ),
    }


def conversation() -> tuple[Message, ...]:
    """The exchange the live run injected, plus the question it then asked."""
    return (
        Message(role=Role.USER, content="My song reads -6.4 LUFS."),
        Message(role=Role.ASSISTANT, content="Got it -- that is very loud."),
        Message(role=Role.USER, content=INJECT_QUESTION),
    )


def test_the_method_is_snake_case_on_the_wire():
    """A trap that cost a live run: the SDK's types imply the other spelling.

    ``thread/injectItems`` is what the generated types suggest and it is rejected
    as an unknown method variant. Pinned as a constant so a future refactor
    cannot quietly camelCase it back.
    """
    assert INJECT_ITEMS_METHOD == "thread/inject_items"
    assert INJECT_ITEMS_METHOD != "thread/injectItems"


def test_prior_turns_carry_real_roles_and_the_responses_api_content_types():
    """``input_text`` for the user, ``output_text`` for the assistant.

    The asymmetry is the Responses API's own -- one is what went in, the other is
    what came out -- and both shapes were driven live. This is the fact the
    ``Assistant: `` labels were standing in for.
    """
    assert app_server_history_items(conversation()) == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "My song reads -6.4 LUFS."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Got it -- that is very loud."}
            ],
        },
    ]


def test_the_params_are_thread_id_and_items():
    assert inject_items_params("t-1", conversation()) == {
        "threadId": "t-1",
        "items": app_server_history_items(conversation()),
    }


def test_history_becomes_one_inject_items_call_in_order(tmp_path):
    """One call, after thread/start and before turn/start, roles in order."""
    server = ScriptedAppServer(inject_script())
    adapter, _ = adapter_for(server)
    events = list(adapter.run(request_for(conversation())))

    assert server.methods.count(INJECT_ITEMS_METHOD) == 1
    order = [m for m in server.methods if m != "initialized"]
    assert order == ["initialize", "thread/start", INJECT_ITEMS_METHOD, "turn/start"]

    params = server.params_of(INJECT_ITEMS_METHOD)
    assert params["threadId"] == INJECT_THREAD_ID
    assert [(i["role"], i["content"][0]["type"]) for i in params["items"]] == [
        ("user", "input_text"),
        ("assistant", "output_text"),
    ]
    assert events[-1].status is TerminalStatus.OK
    assert "".join(e.text for e in events if e.type == "text_delta") == INJECT_ANSWER


def test_the_current_message_is_not_injected(tmp_path):
    """It is the question, not the transcript. Injected items are answered *from*.

    ``thread/inject_items`` appends to the thread's history before the turn runs,
    so a current message put there would become part of what the model is
    answering from rather than the thing it is answering.
    """
    server = ScriptedAppServer(inject_script())
    adapter, _ = adapter_for(server)
    list(adapter.run(request_for(conversation())))

    texts = [
        i["content"][0]["text"] for i in server.params_of(INJECT_ITEMS_METHOD)["items"]
    ]
    assert INJECT_QUESTION not in texts
    assert server.params_of("turn/start")["input"] == [
        {"type": "text", "text": INJECT_QUESTION}
    ]


def test_no_history_means_no_inject_call_at_all():
    """A fresh single-turn call is the run S4 shipped, minus labels it never had."""
    server = ScriptedAppServer(inject_script())
    adapter, _ = adapter_for(server)
    list(adapter.run(request_for()))
    assert not server.sent(INJECT_ITEMS_METHOD)
    assert server.params_of("turn/start")["input"] == [
        {"type": "text", "text": "Reply with exactly OK"}
    ]


def test_a_system_prompt_still_lands_on_base_instructions_alongside_history():
    """The two channels are separate: instructions to the thread, turns to history."""
    server = ScriptedAppServer(inject_script())
    adapter, _ = adapter_for(server)
    messages = (Message(role=Role.SYSTEM, content="Be exact."), *conversation())
    list(adapter.run(request_for(messages)))

    assert server.params_of("thread/start")["baseInstructions"] == "Be exact."
    roles = [i["role"] for i in server.params_of(INJECT_ITEMS_METHOD)["items"]]
    assert roles == ["user", "assistant"]


class _RefusingInjectItems(ScriptedAppServer):
    """A build that does not carry the method -- the case the fallback exists for."""

    error: ClassVar[dict] = {
        "code": -32601,
        "message": "unknown variant `thread/inject_items`",
    }

    def on_write(self, text: str) -> None:
        message = json.loads(text)
        self.received.append(message)
        if message.get("method") == INJECT_ITEMS_METHOD:
            self.send({"id": message["id"], "error": dict(self.error)})
            return
        super().on_write(text)


def test_a_refused_injection_falls_back_to_the_labels_and_says_so():
    """A downgrade, announced. Losing the conversation would be far worse.

    This surface moves -- the snake/camel spelling proves it -- so a build that
    refuses the method gets S4's labelled transcript rather than a run with no
    history in it. The ``vendor_event`` is what keeps that from being a silent
    downgrade, and it names what the fallback costs.
    """
    server = _RefusingInjectItems(inject_script())
    adapter, _ = adapter_for(server)
    events = list(adapter.run(request_for(conversation())))

    announced = [
        e
        for e in events
        if e.type == "vendor_event" and e.name == "history/inject_items_refused"
    ]
    assert len(announced) == 1
    data = announced[0].data
    assert data["method"] == INJECT_ITEMS_METHOD
    assert data["code"] == -32601
    assert "unknown variant" in data["detail"]
    assert data["items"] == 2
    assert "Assistant: " in data["fallback"]

    assert server.params_of("turn/start")["input"] == [
        {"type": "text", "text": "User: My song reads -6.4 LUFS."},
        {"type": "text", "text": "Assistant: Got it -- that is very loud."},
        {"type": "text", "text": f"User: {INJECT_QUESTION}"},
    ]
    # And the run still completes: a refused seam is a different shape, not a
    # failure.
    assert events[-1].status is TerminalStatus.OK


def test_the_announcement_comes_before_the_turn_it_describes():
    """A caller reading the stream learns about the downgrade before the answer."""
    server = _RefusingInjectItems(inject_script())
    adapter, _ = adapter_for(server)
    types = [e.type for e in adapter.run(request_for(conversation()))]
    assert types.index("vendor_event") < types.index("text_delta")


def test_the_exec_transport_is_untouched_by_any_of_this():
    """``render_prompt`` has no seam to move to, and did not move."""
    assert render_prompt(conversation()) == "\n".join(
        [
            _TRANSCRIPT_PREAMBLE,
            "",
            "User: My song reads -6.4 LUFS.",
            "Assistant: Got it -- that is very loud.",
            f"User: {INJECT_QUESTION}",
        ]
    )


def test_the_stateless_call_asks_for_an_ephemeral_thread():
    """The ``--ephemeral`` directive, as a parameter rather than a flag."""
    request = request_for()
    assert any(d.name == "--ephemeral" for d in request.plan.directives)
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    list(adapter.run(request))
    assert server.params_of("thread/start")["ephemeral"] is True


def test_the_schema_rides_the_turn_and_writes_no_file(tmp_path):
    """``outputSchema`` is per turn -- no ``--output-schema <FILE>`` to clean up."""
    schema = {
        "type": "object",
        "properties": {"lufs": {"type": "number"}},
        "required": ["lufs"],
        "additionalProperties": False,
    }
    answer = {
        "method": "item/completed",
        "params": {
            "item": {"type": "agentMessage", "id": "msg_s", "text": '{"lufs": -13.7}'},
            "threadId": "t",
            "turnId": "u",
        },
    }
    server = ScriptedAppServer(
        {**script_for(), "turn/start": (
            captured_payload("turn1_response"), [answer, turn_completed("completed")]
        )}
    )
    adapter, _ = adapter_for(server)
    events = list(adapter.run(request_for(options={"cwd": str(tmp_path)}, schema=schema)))

    assert server.params_of("turn/start")["outputSchema"] == schema
    structured = [e for e in events if e.type == "structured_output"]
    assert structured and structured[0].data == {"lufs": -13.7}
    assert events[-1].status is TerminalStatus.OK
    assert list(tmp_path.iterdir()) == []


def test_the_model_is_named_on_the_thread():
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    list(adapter.run(request_for(model="gpt-5.6-luna")))
    assert server.params_of("thread/start")["model"] == "gpt-5.6-luna"


# --- terminal status mapping --------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", TerminalStatus.OK),
        ("failed", TerminalStatus.ERROR),
        ("interrupted", TerminalStatus.CANCELLED),
    ],
)
def test_the_turns_own_status_decides_the_terminal(status, expected):
    error = None if status == "completed" else {"message": "the model provider said no"}
    server = ScriptedAppServer(
        {**script_for(), "turn/start": (
            captured_payload("turn1_response"), [turn_completed(status, error)]
        )}
    )
    adapter, _ = adapter_for(server)
    terminal = list(adapter.run(request_for()))[-1]
    assert terminal.status is expected
    if status != "completed":
        assert "the model provider said no" in (terminal.reason or "")


def test_a_spent_allowance_is_a_clean_stop_not_a_failure():
    error = {"message": "out of credits", "codexErrorInfo": "usageLimitExceeded"}
    server = ScriptedAppServer(
        {**script_for(), "turn/start": (
            captured_payload("turn1_response"), [turn_completed("failed", error)]
        )}
    )
    adapter, _ = adapter_for(server)
    terminal = list(adapter.run(request_for()))[-1]
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED


# --- refusals -----------------------------------------------------------------------


def test_mcp_servers_are_refused_rather_than_dropped_on_this_transport():
    server = ScriptedAppServer(script_for())
    adapter, spawn = adapter_for(server)
    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(adapter.run(request_for(mcp_servers={"docs": {"command": "npx"}})))
    assert excinfo.value.capability == "mcp_servers"
    assert "exclusivity" in str(excinfo.value)
    assert spawn.calls == 0


# --- cancellation (the method is driven; modelpass's use of it is not) ------------------


def test_cancel_attempts_the_interrupt_and_still_synthesizes_the_terminal():
    """The politer step is tried; the kill behind it is what the run relies on.

    modelpass does not wait for the ``turn/completed`` that ``turn/interrupt``
    produces -- driven live 2026-08-31, ``status: "interrupted"`` -- so the
    cancelled terminal is synthesized exactly as it is on the exec transport, and
    the child is closed either way. That not-waiting is why ``graceful_cancel``
    stays unclaimed even though the vendor's mechanism is now proven.
    """
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    events = adapter.run(request_for())
    # Far enough in that the turn knows its own id; item/started carries it.
    for _ in range(3):
        next(events)

    adapter.cancel()
    terminal = list(events)[-1]

    assert server.sent("turn/interrupt")
    assert server.params_of("turn/interrupt") == {
        "threadId": captured_payload("thread_start_response")["thread"]["id"],
        "turnId": captured_payload("turn1_response")["turn"]["id"],
    }
    assert terminal.status is TerminalStatus.CANCELLED
    assert "turn/interrupt was attempted" in (terminal.reason or "")
    assert server.terminated == 1


def test_interrupt_needs_both_ids_and_never_raises():
    """Defensive by construction: the method has not been driven live.

    A missing id means no attempt, and a server that refuses or a transport that
    is already gone is reported as ``False`` rather than as an exception -- the
    kill behind it is what actually ends the run.
    """

    class Refusing:
        def request(self, *args, **kwargs):
            raise VendorRunFailed("no such turn")

    assert interrupt_app_server_turn(Refusing(), None, "turn1") is False
    assert interrupt_app_server_turn(Refusing(), "thread1", None) is False
    assert interrupt_app_server_turn(Refusing(), "thread1", "turn1") is False

    class Accepting:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def request(self, method, params=None, *, timeout=None):
            self.calls.append((method, dict(params or {})))
            return {}

    accepting = Accepting()
    assert interrupt_app_server_turn(accepting, "thread1", "turn1") is True
    assert accepting.calls == [
        ("turn/interrupt", {"threadId": "thread1", "turnId": "turn1"})
    ]


# --- in-process caller tools (S6) ----------------------------------------------------
#
# The whole loop, scripted from the live capture: the ``item/started`` and
# ``item/completed`` notifications are the recorded ones, and the
# ``item/tool/call`` request between them is the recorded ``_kind:
# server_request`` line with an id added -- the recorder stripped ids, and the
# JSON-RPC id is the one field a test has to supply to correlate an answer.


def captured_server_request(request_id: int = 900) -> dict:
    """The recorded ``item/tool/call`` request, with a JSON-RPC id attached."""
    for line in capture():
        if line.get("_kind") == "server_request":
            return {
                "id": request_id,
                "method": line["method"],
                "params": dict(line["params"]),
            }
    raise AssertionError("the live capture has no server_request line")


TOOL_CALL_REQUEST = captured_server_request()
assert TOOL_CALL_REQUEST["method"] == DYNAMIC_TOOL_CALL_METHOD
#: The tool the live run registered and the model actually called.
CAPTURED_TOOL_NAME = TOOL_CALL_REQUEST["params"]["tool"]
CAPTURED_CALL_ID = TOOL_CALL_REQUEST["params"]["callId"]


def tool_turn_notifications() -> list[dict]:
    """The capture's first turn, with the tool-call request spliced back in.

    The request is a *request*, not a notification, so ``captured_notifications``
    cannot carry it. It goes exactly where the recorder saw it: after the
    ``item/started`` for the ``dynamicToolCall`` and before the
    ``item/completed`` that reports its result.
    """
    lines = list(FIRST_TURN_NOTIFICATIONS)
    for index, line in enumerate(lines):
        item = line["params"].get("item") if line["method"] == "item/started" else None
        if isinstance(item, dict) and item.get("type") == "dynamicToolCall":
            return [*lines[: index + 1], TOOL_CALL_REQUEST, *lines[index + 1 :]]
    raise AssertionError("the capture's first turn has no dynamicToolCall item/started")


def loudness_tool(handler) -> ToolDef:
    """A ToolDef named for the tool the live run registered."""
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


def tool_script() -> dict:
    return script_for(tool_turn_notifications())


def _raise_offline(args):
    raise RuntimeError("the loudness index is offline")


# -- registration


def test_a_tooldef_becomes_a_dynamic_tools_entry_field_for_field():
    """``ThreadStartParams.dynamicTools``: type/name/description/inputSchema.

    The vendor's own ``DynamicToolSpec::Function`` field names (codex.exe
    0.151.0, 2026-08-31), and the shape driven live the same day. The schema is
    ``ToolDef.json_schema()``, so a caller who handed in a bare properties
    mapping gets the same wrapping here as ``schema=`` gives them elsewhere.
    """
    tool = loudness_tool(lambda args: "")
    assert dynamic_tools_param((tool,)) == [
        {
            "type": "function",
            "name": CAPTURED_TOOL_NAME,
            "description": "Look up a track's integrated loudness.",
            "inputSchema": {
                "type": "object",
                "properties": {"track": {"type": "string"}},
                "required": ["track"],
            },
        }
    ]


def test_thread_start_omits_dynamic_tools_entirely_when_there_are_none():
    """Absence, not an empty list -- they are different statements.

    ``[]`` says "this thread has zero dynamic tools"; omitting the key says
    nothing at all. Codex persists a registration in the rollout and restores it
    on ``thread/resume`` when none is supplied, so the empty list is the reading
    that could one day clear something it only meant to leave alone. It is also
    what the tool-free live capture sent.
    """
    assert dynamic_tools_param(()) is None

    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    list(adapter.run(request_for()))
    assert "dynamicTools" not in server.params_of("thread/start")


def test_the_registration_reaches_thread_start_on_a_tool_bearing_run():
    server = ScriptedAppServer(tool_script())
    adapter, _ = adapter_for(server)
    list(adapter.run(request_for(tools=(loudness_tool(lambda args: "-13.7 LUFS"),))))
    assert server.params_of("thread/start")["dynamicTools"] == dynamic_tools_param(
        (loudness_tool(lambda args: ""),)
    )


# -- the round trip


def test_the_handlers_answer_goes_back_as_content_items():
    """``item/tool/call`` -> the caller's function -> ``{contentItems, success}``.

    The arguments the model sent are handed to the handler as a mapping, and its
    return value comes back as one ``inputText`` block with ``success: true`` --
    the response shape driven live, and the two required fields of
    ``DynamicToolCallResponse``.
    """
    seen: list[dict] = []

    def handler(args):
        seen.append(dict(args))
        return "-13.7 LUFS integrated, true peak -1.2 dBTP"

    server = ScriptedAppServer(tool_script())
    adapter, _ = adapter_for(server)
    events = list(adapter.run(request_for(tools=(loudness_tool(handler),))))

    assert seen == [{"track": "Neon Dust"}]
    assert server.answers[TOOL_CALL_REQUEST["id"]] == {
        "contentItems": [
            {
                "type": "inputText",
                "text": "-13.7 LUFS integrated, true peak -1.2 dBTP",
            }
        ],
        "success": True,
    }
    assert events[-1].status is TerminalStatus.OK
    # The server's own completed item is still what produces the tool_result on
    # the happy path -- modelpass adds nothing to a call that worked.
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [(r.name, r.is_error) for r in results] == [(CAPTURED_TOOL_NAME, False)]
    assert results[0].content.startswith("-13.7 LUFS")


def test_a_handler_may_return_the_mcp_content_shape():
    """The ToolDef contract is one promise, whichever runtime honours it.

    ``{"content": [...]}`` is what the Anthropic adapter accepts, so it is what
    this one accepts, flattened to text by the shared helper rather than by a
    second reading of the same documented shape.
    """
    server = ScriptedAppServer(tool_script())
    adapter, _ = adapter_for(server)
    tool = loudness_tool(
        lambda args: {"content": [{"type": "text", "text": "-13.7 LUFS"}]}
    )
    list(adapter.run(request_for(tools=(tool,))))
    assert server.answers[TOOL_CALL_REQUEST["id"]]["contentItems"] == [
        {"type": "inputText", "text": "-13.7 LUFS"}
    ]
    assert server.answers[TOOL_CALL_REQUEST["id"]]["success"] is True


def test_an_async_handler_runs_on_the_reader_thread_without_a_loop():
    """ToolDef promises sync *or* async, and the reader thread has no loop."""

    async def handler(args):
        return f"async saw {args['track']}"

    server = ScriptedAppServer(tool_script())
    adapter, _ = adapter_for(server)
    list(adapter.run(request_for(tools=(loudness_tool(handler),))))
    assert server.answers[TOOL_CALL_REQUEST["id"]]["contentItems"] == [
        {"type": "inputText", "text": "async saw Neon Dust"}
    ]


# -- failure is a result


def test_a_raising_handler_answers_success_false_and_the_turn_continues():
    """A caller's bug must not wedge the reader thread or hang the turn.

    Three things are asserted together because they are one property: the server
    gets a **well-formed** ``success: false`` (a malformed answer would only tell
    the model that modelpass is broken), the caller gets a ``tool_result`` with
    ``is_error=True`` carrying the exception, and the run still reaches its
    ordinary terminal. The reader thread answering is what makes the last one
    possible -- an exception there would strand every pending request.
    """
    server = ScriptedAppServer(tool_script())
    adapter, _ = adapter_for(server)
    events = list(adapter.run(request_for(tools=(loudness_tool(_raise_offline),))))

    answer = server.answers[TOOL_CALL_REQUEST["id"]]
    assert answer["success"] is False
    assert answer["contentItems"][0]["type"] == "inputText"
    text = answer["contentItems"][0]["text"]
    assert "RuntimeError" in text and "the loudness index is offline" in text

    failures = [e for e in events if isinstance(e, ToolResultEvent) and e.is_error]
    assert [(f.id, f.name) for f in failures] == [(CAPTURED_CALL_ID, CAPTURED_TOOL_NAME)]
    assert "RuntimeError" in failures[0].content

    terminal = events[-1]
    assert terminal.status is TerminalStatus.OK
    assert terminal.reason is None


def test_a_reported_failure_is_not_doubled_by_the_servers_completed_item():
    """One call, one ``tool_result``. The item still arrives as a vendor_event.

    modelpass emits the failure from its own knowledge because whether the server
    mirrors ``success: false`` onto the completed item is undriven. The dedup is
    what keeps that honesty from becoming a duplicate: the completed item is
    recognized by its call id and passed through whole instead.
    """
    server = ScriptedAppServer(tool_script())
    adapter, _ = adapter_for(server)
    events = list(adapter.run(request_for(tools=(loudness_tool(_raise_offline),))))
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    vendor_names = [getattr(e, "name", "") for e in events if e.type == "vendor_event"]
    assert "item/completed:dynamicToolCall" in vendor_names


def test_a_handler_reporting_is_error_itself_is_a_failure_too():
    """Saying no is not crashing, and both take the same route back."""
    server = ScriptedAppServer(tool_script())
    adapter, _ = adapter_for(server)
    tool = loudness_tool(
        lambda args: {
            "content": [{"type": "text", "text": "no such track"}],
            "is_error": True,
        }
    )
    events = list(adapter.run(request_for(tools=(tool,))))
    assert server.answers[TOOL_CALL_REQUEST["id"]] == {
        "contentItems": [{"type": "inputText", "text": "no such track"}],
        "success": False,
    }
    failures = [e for e in events if isinstance(e, ToolResultEvent) and e.is_error]
    assert [f.content for f in failures] == ["no such track"]


def test_a_call_for_a_tool_this_run_never_registered_is_answered_not_ignored():
    """An unanswered server request hangs the turn, so there is no silent path."""
    server = ScriptedAppServer(tool_script())
    adapter, _ = adapter_for(server)
    other = ToolDef(
        name="something_else",
        description="Not the tool the model asked for.",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "",
    )
    events = list(adapter.run(request_for(tools=(other,))))
    answer = server.answers[TOOL_CALL_REQUEST["id"]]
    assert answer["success"] is False
    assert "something_else" in answer["contentItems"][0]["text"]
    assert events[-1].status is TerminalStatus.OK


def test_approvals_are_still_declined_on_a_tool_bearing_run():
    """Adding tools must not quietly stop refusing to authorize commands.

    The dispatcher replaces the client's default handler, so the test that
    matters is that it *delegates* rather than shadows.
    """
    approval = {
        "id": 901,
        "method": "item/commandExecution/requestApproval",
        "params": {"threadId": "t", "turnId": "u", "command": "rm -rf /"},
    }
    notifications = tool_turn_notifications()
    notifications.insert(0, approval)
    server = ScriptedAppServer(script_for(notifications))
    adapter, _ = adapter_for(server)
    list(adapter.run(request_for(tools=(loudness_tool(lambda args: "ok"),))))
    assert server.answers[901] == {"decision": "decline"}


# -- names, refused before anything is launched


class _NamedTool:
    """A ToolDef-shaped stand-in carrying a name ``ToolDef`` would reject.

    The point of writing the vendor's rule down separately is that a name which
    reached the transport some other way -- a future loosening of
    ``TOOL_NAME_PATTERN``, a caller building the request by hand -- fails at the
    check rather than at ``thread/start`` on somebody's machine.
    """

    description = "d"

    def __init__(self, name: str) -> None:
        self.name = name

    @staticmethod
    def handler(args):
        return ""

    def json_schema(self):
        return {"type": "object", "properties": {}}


def test_a_name_the_vendor_would_reject_is_refused_before_launch():
    """The vendor's own regex (codex.exe 0.151.0, 2026-08-31), checked early."""
    for name in ("has space", "dot.name", "curly{}"):
        with pytest.raises(InvalidTool, match=r"\^\[a-zA-Z0-9_-\]\+\$"):
            check_dynamic_tool_names((_NamedTool(name),))

    with pytest.raises(InvalidTool, match="must not be empty"):
        check_dynamic_tool_names((_NamedTool(""),))
    with pytest.raises(InvalidTool, match="leading or trailing whitespace"):
        check_dynamic_tool_names((_NamedTool(" padded "),))
    with pytest.raises(InvalidTool, match="64"):
        check_dynamic_tool_names((_NamedTool("a" * 65),))
    with pytest.raises(InvalidTool, match="duplicate"):
        check_dynamic_tool_names((_NamedTool("ok"), _NamedTool("ok")))


def test_every_name_tooldef_accepts_is_a_name_the_vendor_accepts():
    """The two constraints must not drift apart; this is what pins them."""
    check_dynamic_tool_names(
        tuple(
            ToolDef(name=name, description="d", parameters={}, handler=lambda args: "")
            for name in ("a", "get_loudness", "Get-Loudness9", "z" * 64)
        )
    )


def test_a_bad_name_refuses_the_run_before_the_child_is_spawned():
    server = ScriptedAppServer(tool_script())
    adapter, spawn = adapter_for(server)
    with pytest.raises(InvalidTool):
        list(adapter.run(request_for(tools=(_NamedTool("bad name"),))))
    assert spawn.calls == 0


class _RefusingThreadStart(ScriptedAppServer):
    """A server that rejects ``thread/start`` with a JSON-RPC error."""

    error: ClassVar[dict] = {
        "code": -32602,
        "message": "dynamic tool name is reserved: shell",
    }

    def on_write(self, text: str) -> None:
        message = json.loads(text)
        self.received.append(message)
        if message.get("method") == "thread/start":
            self.send({"id": message["id"], "error": dict(self.error)})
            return
        super().on_write(text)


def test_a_reserved_name_is_the_servers_refusal_made_legible():
    """The one rule that cannot be pre-checked gets a message instead.

    ``dynamic tool name is reserved`` compares against the thread's own active
    toolbelt, which does not exist until the thread does. So the server has the
    last word, and modelpass turns its JSON-RPC code into a sentence naming the
    tools it sent and the reason a caller would otherwise have to guess.
    """
    adapter, _ = adapter_for(_RefusingThreadStart(tool_script()))
    with pytest.raises(VendorRunFailed) as excinfo:
        list(adapter.run(request_for(tools=(_NamedTool("shell"),))))
    message = str(excinfo.value)
    assert "shell" in message and "RESERVED" in message
    assert "dynamic tool name is reserved" in message


def test_a_thread_start_refusal_without_tools_is_left_alone():
    """The wrapping is about tool names; a plain run's error keeps its own shape."""
    server = _RefusingThreadStart(script_for())
    server.error = {"code": -32603, "message": "no model available"}
    adapter, _ = adapter_for(server)
    with pytest.raises(VendorRunFailed) as excinfo:
        list(adapter.run(request_for()))
    assert "no model available" in str(excinfo.value)
    assert "RESERVED" not in str(excinfo.value)


# -- and the exec transport keeps refusing, with a message that says where to go


def test_tools_are_still_refused_on_exec_and_the_message_names_the_transport():
    """The retired absence claim must not come back (plan of record, S6/S7).

    "Codex is an MCP client only" was a statement about the vendor and is false:
    the same binary registers caller functions on its app-server surface, and
    modelpass drives that loop. What is true is narrower and is what the refusal
    says. Since S7 the second half of the message changed direction with the
    default: reaching this refusal means the caller OPTED OUT of the transport
    that runs their tools, so the fix is to drop an option, not add one.
    """
    adapter = OpenAIAdapter(codex_bin="C:/codex/codex.exe", spawn=exploding_spawn)
    connection = codex_connection()
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection, {"PATH": "/usr/bin"}),
        tools=(loudness_tool(lambda args: ""),),
        options={"transport": TRANSPORT_EXEC},
    )
    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(adapter.run(request))
    message = str(excinfo.value)
    assert excinfo.value.capability == "tools_in_process"
    assert "DEFAULT transport" in message
    assert "options={'transport': 'exec'}" in message
    assert "MCP client only" not in message


# --- reaching it through bridge.chat() (S7, 2026-08-31) ------------------------------
#
# S6 built a working in-process tool loop and could not write this test. The gate
# in ``bridge.chat()`` called ``registry.require(runtime, TOOLS_IN_PROCESS)``
# before it looked at ``options``, and that cell reads ``unsupported`` because it
# describes ``codex exec`` -- correctly. So the loop was reachable through the
# adapter and refused through the bridge, which is every caller.
#
# S7 splits the two questions the registry was answering at once. The table still
# says what a caller can rely on **by default**, which is what ``find()`` selects
# on and is deliberately unchanged; ``Adapter.support_for(capability, options)``
# says whether **this request** will work, and is asked first.


def bridge_over(adapter: OpenAIAdapter, tmp_path) -> Bridge:
    """A real :class:`Bridge` over one Codex connection and this adapter.

    The two probes the preflight would otherwise shell out for are stubbed the
    same way the receipt tests stub them -- no process is launched here either,
    on any transport.
    """
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


def test_the_tool_loop_runs_end_to_end_through_bridge_chat(tmp_path):
    """The test S6 could not write: caller functions, through the public API.

    Registration, the server's ``item/tool/call``, the caller's own function on
    the reader thread, the ``contentItems`` answer and the run's terminal -- all
    of it via ``bridge.chat()``, which is the only surface a consumer has.
    """
    seen: list[dict] = []

    def handler(args):
        seen.append(dict(args))
        return "-13.7 LUFS integrated"

    server = ScriptedAppServer(tool_script())
    adapter, spawn = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)

    events = list(
        bridge.chat(
            connection="codex-sub",
            message="How loud is Neon Dust?",
            tools=[loudness_tool(handler)],
            options={"transport": TRANSPORT_APP_SERVER},
        )
    )

    assert seen == [{"track": "Neon Dust"}]
    assert server.answers[TOOL_CALL_REQUEST["id"]] == {
        "contentItems": [{"type": "inputText", "text": "-13.7 LUFS integrated"}],
        "success": True,
    }
    # D23 and S6 together, which is the pairing that had to be proved: the
    # runtime's own toolbelt is off on the command line and the caller's own
    # tools still reach the wire and still run. Driven live 2026-08-31 as well
    # -- toolbelt off, `dynamicTools` registered, handler invoked, answer
    # correct -- so this is the offline half of a fact, not a guess.
    assert spawn.argv == [
        "C:/codex/codex.exe",
        "app-server",
        *chat_tool_overrides(),
        "--listen",
        "stdio://",
    ]
    # The tools really were registered on the wire, not merely accepted by the gate.
    names = [t["name"] for t in server.params_of("thread/start")["dynamicTools"]]
    assert names == [CAPTURED_TOOL_NAME]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [(r.name, r.is_error) for r in results] == [(CAPTURED_TOOL_NAME, False)]
    assert events[-1].status is TerminalStatus.OK


def test_the_gate_refuses_tools_on_the_exec_opt_out_and_says_why(tmp_path):
    """And with no option at all the same call now runs (S7, 2026-08-31).

    The refusal must keep its S6 quality: a bare *capability
    'tools_in_process' is unsupported* names a table cell and leaves the caller
    to discover on their own that the thing they asked for is one option away.
    The registry's note is where that sentence already lives, and the gate
    carries it. What changed with the flip is which side needs the sentence:
    the default now runs the loop, so the refusal belongs to the opt-out.
    """
    server = ScriptedAppServer(tool_script())
    adapter, spawn = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)

    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(
            bridge.chat(
                connection="codex-sub",
                message="How loud is Neon Dust?",
                tools=[loudness_tool(lambda args: "")],
                options={"transport": TRANSPORT_EXEC},
            )
        )
    message = str(excinfo.value)
    assert excinfo.value.capability == "tools_in_process"
    assert excinfo.value.support == "unsupported"
    assert "transport" in message and "app-server" in message
    # Refused before anything launched, on either transport.
    assert spawn.calls == 0


def test_a_typo_in_the_transport_is_refused_by_the_gate_rather_than_ignored(tmp_path):
    """``support_for`` resolves the transport, so a bad one is named as itself.

    Falling back to the registry here would refuse the run with a
    ``tools_in_process`` message, sending the caller to look at a capability
    table when what they actually did was misspell an option.
    """
    server = ScriptedAppServer(tool_script())
    adapter, _ = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)

    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(
            bridge.chat(
                connection="codex-sub",
                message="hi",
                tools=[loudness_tool(lambda args: "")],
                options={"transport": "appserver"},
            )
        )
    assert excinfo.value.capability == "transport"


def test_a_plain_chat_call_on_app_server_still_asks_the_gate_nothing(tmp_path):
    """No tools, no schema, no gate -- S7 must not add a check to the D7 call."""
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)
    events = list(
        bridge.chat(
            connection="codex-sub",
            message="Reply with exactly OK",
            options={"transport": TRANSPORT_APP_SERVER},
        )
    )
    assert events[-1].status is TerminalStatus.OK
    assert "dynamicTools" not in server.params_of("thread/start")


def test_mcp_servers_are_still_refused_on_app_server_by_the_adapter(tmp_path):
    """``support_for`` answers for one capability, and only where it was driven.

    ``mcp_servers`` is ``supported`` on this runtime -- the cell describes exec,
    where it is true -- and the app-server path has no equivalent that has been
    driven. The adapter has no opinion, so the gate passes and the *adapter*
    refuses, which is exactly where the reason for the refusal is written.
    """
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)
    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(
            bridge.chat(
                connection="codex-sub",
                message="hi",
                mcp_servers={"amt": {"command": "python"}},
                options={"transport": TRANSPORT_APP_SERVER},
            )
        )
    assert excinfo.value.capability == "mcp_servers"
    assert "exec" in str(excinfo.value)


def test_find_still_answers_the_default_question_and_not_the_request_one(tmp_path):
    """The two questions must not silently merge back into one.

    ``find(capability=...)`` is asked *before* a request exists -- somebody
    picking a connection has not written the call yet, let alone its options --
    so the answer it gives is the answer **about the default**. On 2026-08-31
    that answer changed for ``tools_in_process`` because the default did, and
    it changed in the registry rather than through the request hook: the proof
    is that ``support_for`` has no opinion for the very options ``find`` would
    have had to consult. ``mcp_servers`` is the same test read the other way --
    the Codex connection stopped appearing there for the same reason.
    """
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)

    assert [c.name for c in bridge.find(runtime=Runtime.OPENAI_SDK)] == ["codex-sub"]
    assert [
        c.name for c in bridge.find(capability=Capability.TOOLS_IN_PROCESS)
    ] == ["codex-sub"]
    assert bridge.find(capability=Capability.MCP_SERVERS) == ()
    assert (
        bridge.registry.support(Runtime.OPENAI_SDK, Capability.TOOLS_IN_PROCESS)
        is Support.SUPPORTED
    )
    # find() reads the table, not the hook -- which has no opinion by default.
    assert adapter.support_for(Capability.TOOLS_IN_PROCESS, {}) is None


def test_the_session_gate_now_answers_for_the_transport_too(tmp_path):
    """The successor to *sessions are untouched by the request-level gate* (S5).

    That test pinned a refusal whose reason was **where the code stood, not what
    the vendor could do**: while sessions ran only on ``codex exec``, a session
    gate consulting ``support_for`` would have said yes and then opened a session
    on the transport that cannot do it. S5 makes sessions transport-selectable,
    so the gate and the session path agree again -- and this checks both
    directions rather than only the new one.
    """
    server = ScriptedAppServer(script_for())
    adapter, _ = adapter_for(server)
    bridge = bridge_over(adapter, tmp_path)

    # exec: still refused, now by the adapter narrowing rather than the table.
    with pytest.raises(CapabilityNotSupported) as excinfo:
        bridge.new_chat(
            connection="codex-sub",
            project_folder=str(tmp_path / "work"),
            tools=[loudness_tool(lambda args: "")],
            options={"transport": TRANSPORT_EXEC},
        )
    assert excinfo.value.capability == "tools_in_process"

    # No options at all: opens, and the handle is the native-thread one.
    session = bridge.new_chat(
        connection="codex-sub",
        project_folder=str(tmp_path / "work"),
        tools=[loudness_tool(lambda args: "")],
    )
    assert isinstance(session._handle, CodexAppServerSession)
    session.close()


def test_the_request_hook_inverted_when_the_default_did():
    """Same mechanism, same two questions, mirrored direction (S7, 2026-08-31).

    Until S7 this hook widened: the registry described ``codex exec`` and the
    hook named the three things app-server could do that exec could not. The
    flip made those the registry's own answer, so what is left for the hook is
    the mirror image -- an explicit ``transport="exec"`` NARROWS what a request
    can do. The mechanism was kept rather than deleted because the two
    questions it separates are both still real, and ``find()`` still needs the
    first one answered without the second in the way.
    """
    adapter = OpenAIAdapter(codex_bin="C:/codex/codex.exe", spawn=exploding_spawn)
    exec_options = {"transport": TRANSPORT_EXEC}

    # Six cells the opt-out narrows, each a checked fact about codex exec.
    for capability in (
        Capability.TOOLS_IN_PROCESS,
        Capability.SYSTEM_PROMPT_REPLACE,
        Capability.SESSIONS_LIST,
        Capability.INTERIM_USAGE,
        Capability.INCREMENTAL_TEXT,
    ):
        assert adapter.support_for(capability, exec_options) is Support.UNSUPPORTED
    # thinking narrows to UNVERIFIED, not to a no: nobody drove it on exec.
    assert adapter.support_for(Capability.THINKING, exec_options) is Support.UNVERIFIED
    # And one it WIDENS: per-run MCP servers are exec-only.
    assert adapter.support_for(Capability.MCP_SERVERS, exec_options) is Support.SUPPORTED

    # On the default -- selected or not -- the hook has no opinion at all, so
    # the registry answers and find() keeps working on the default question.
    for options in ({}, {"transport": TRANSPORT_APP_SERVER}):
        for capability in Capability:
            assert adapter.support_for(capability, options) is None

    # Still unclaimed on either side, and after 2026-08-31 for three different
    # reasons: thread/fork is undriven and unbuilt; turn/interrupt is driven but
    # modelpass does not wait for the terminal it produces and no drive has shown
    # a thread surviving one; ephemeral multi-turn is driven and not built here.
    for capability in (
        Capability.GRACEFUL_CANCEL,
        Capability.EPHEMERAL_MULTI_TURN,
        Capability.SESSIONS_FORK,
    ):
        assert adapter.support_for(capability, exec_options) is None


def test_an_adapter_with_no_opinion_is_the_default():
    """The base hook answers ``None`` for everything, so nothing else moves."""
    adapter = AnthropicAdapter()
    for capability in Capability:
        assert adapter.support_for(capability, {"transport": "app-server"}) is None
