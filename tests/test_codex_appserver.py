"""Codex app-server transport tests -- all offline (S2 of the migration plan).

No test spawns a process, imports a vendor package or reads a credential. The
child is :class:`FakeAppServer`, a scripted stand-in for ``subprocess.Popen``
that parses what the client writes and answers it, so handshake ordering,
request correlation, server->client requests and EOF handling are all exercised
against the real reader thread rather than a mock of it.

Wire shapes are the ones recorded in the module docstring: verified live on
2026-08-31 for the handshake and the three line shapes, schema-sourced from
``codex-rs/app-server-protocol/schema/json`` in openai/codex for the approval
responses.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import pytest

from modelpass.adapters.codex_appserver import (
    APPROVAL_REQUEST_METHODS,
    AppServerClient,
    AppServerRequestFailed,
    AppServerTimeout,
    AppServerTransportClosed,
    AppServerTurn,
    Notification,
    app_server_argv,
    decline_approvals,
    map_appserver_event,
    token_usage_from_breakdown,
    turn_error_detail,
)
from modelpass.runtimes import Runtime
from modelpass.types import (
    CALLER_TOOL_SERVER,
    TerminalStatus,
    TextDeltaEvent,
    ThinkingEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    UsageScope,
    VendorEvent,
)

#: The raw app-server conversation recorded on 2026-08-31 by driving modelpass's
#: own :class:`AppServerClient` against codex.exe 0.151.0-alpha.7.1. Every line
#: is one wire message, wrapped in a ``_kind`` envelope naming what it was.
FIXTURE = Path(__file__).parent / "fixtures" / "appserver" / "live-capture-2026-08-31.jsonl"

# --- the fake child --------------------------------------------------------------


class FakeStream:
    """A blocking line iterator fed from a queue; ``None`` is EOF."""

    def __init__(self) -> None:
        self.lines: queue.Queue[str | None] = queue.Queue()

    def __iter__(self) -> FakeStream:
        return self

    def __next__(self) -> str:
        line = self.lines.get()
        if line is None:
            raise StopIteration
        return line


class FakeStdin:
    """The child's stdin: records every line and hands it to the responder."""

    def __init__(self, server: FakeAppServer) -> None:
        self.server = server
        self.closed = False
        self.flushes = 0

    def write(self, text: str) -> int:
        if self.closed:
            raise ValueError("write to closed stdin")
        self.server.on_write(text)
        return len(text)

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


class FakeAppServer:
    """A scripted ``codex app-server`` child.

    ``responder(server, message) -> iterable of dicts`` decides what the server
    says back to each client message. The default answers every request with an
    empty result, which is enough for a handshake.
    """

    def __init__(self, responder=None) -> None:
        self.stdout = FakeStream()
        self.stderr = FakeStream()
        self.stdin = FakeStdin(self)
        self.received: list[dict] = []
        self.responder = responder
        self.returncode: int | None = None
        self.terminated = 0
        self.killed = 0
        self.dies_on_terminate = True

    # -- what the client wrote

    def on_write(self, text: str) -> None:
        message = json.loads(text)
        self.received.append(message)
        replies = (
            self.responder(self, message)
            if self.responder is not None
            else ([{"id": message["id"], "result": {}}] if "id" in message else [])
        )
        for reply in replies or ():
            self.send(reply)

    @property
    def requests(self) -> list[dict]:
        return [m for m in self.received if "id" in m and "method" in m]

    @property
    def methods(self) -> list[str]:
        return [m["method"] for m in self.received if "method" in m]

    # -- what the server says

    def send(self, obj: dict) -> None:
        self.stdout.lines.put(json.dumps(obj) + "\n")

    def send_raw(self, line: str) -> None:
        self.stdout.lines.put(line)

    def eof(self) -> None:
        self.stdout.lines.put(None)

    def say_on_stderr(self, text: str) -> None:
        self.stderr.lines.put(text)

    def stderr_eof(self) -> None:
        self.stderr.lines.put(None)

    # -- the Popen surface the client uses

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        if self.dies_on_terminate:
            self.returncode = 0
            self.eof()
            self.stderr_eof()

    def kill(self) -> None:
        self.killed += 1
        self.returncode = -9
        self.eof()
        self.stderr_eof()

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("codex", timeout or 0)
        return self.returncode


class FakeSpawn:
    def __init__(self, server: FakeAppServer) -> None:
        self.server = server
        self.argv: list[str] | None = None
        self.env: dict[str, str] | None = None
        self.cwd: str | None = None
        self.calls = 0

    def __call__(self, argv, env, cwd):
        self.calls += 1
        self.argv, self.env, self.cwd = argv, env, cwd
        return self.server


def wait_until(predicate, timeout: float = 3.0) -> bool:
    """Poll a predicate. Threaded tests need a deadline, not a sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def make_client(server: FakeAppServer, **kwargs) -> tuple[AppServerClient, FakeSpawn]:
    spawn = FakeSpawn(server)
    client = AppServerClient(
        binary="C:/codex/codex.exe",
        env={"PATH": "/usr/bin"},
        cwd="/work",
        spawn=spawn,
        **kwargs,
    )
    return client, spawn


# --- launch ----------------------------------------------------------------------


def test_argv_is_the_fixed_app_server_launch():
    assert app_server_argv("codex.exe") == [
        "codex.exe",
        "app-server",
        "--listen",
        "stdio://",
    ]


def test_no_argv_element_carries_a_newline():
    """The S1 property, pinned here too.

    This argv is fixed and could not carry one, which is exactly why it is worth
    asserting: the pin is what keeps a later addition from reintroducing the
    ``.cmd`` shim hazard on a transport that had been declared immune to it.
    """
    assert all("\n" not in arg for arg in app_server_argv("codex.exe"))


def test_the_scrubbed_environment_is_passed_verbatim():
    """No merge with ``os.environ``: the reason modelpass owns this transport."""
    server = FakeAppServer()
    client, spawn = make_client(server)
    with client:
        assert spawn.env == {"PATH": "/usr/bin"}
        assert spawn.cwd == "/work"
        assert spawn.argv == ["C:/codex/codex.exe", "app-server", "--listen", "stdio://"]


def test_a_launch_failure_is_a_transport_error_not_an_oserror():
    def boom(argv, env, cwd):
        raise OSError(193, "%1 is not a valid Win32 application")

    client = AppServerClient(
        binary="codex", env={}, cwd="/work", spawn=boom
    )
    with pytest.raises(AppServerTransportClosed) as excinfo:
        client.start()
    assert "app-server" in str(excinfo.value)


def test_start_is_not_repeated_on_a_second_call():
    server = FakeAppServer()
    client, spawn = make_client(server)
    with client:
        client.start()
        assert spawn.calls == 1
        assert server.methods == ["initialize", "initialized"]


# --- handshake -------------------------------------------------------------------


def test_handshake_sends_initialize_then_initialized_and_nothing_before():
    server = FakeAppServer()
    client, _ = make_client(server)
    with client:
        pass
    assert server.methods[:2] == ["initialize", "initialized"]

    initialize = server.received[0]
    assert "id" in initialize
    params = initialize["params"]
    assert params["clientInfo"]["name"] == "modelpass"
    assert params["clientInfo"]["title"] == "modelpass"
    assert isinstance(params["clientInfo"]["version"], str) and params["clientInfo"]["version"]
    assert params["capabilities"] == {"experimentalApi": True}


def test_initialized_is_a_notification_with_no_id_and_no_params():
    """``ClientNotification.json`` specifies ``initialized`` with ``method`` alone."""
    server = FakeAppServer()
    client, _ = make_client(server)
    with client:
        pass
    initialized = server.received[1]
    assert initialized == {"method": "initialized"}


def test_initialize_is_answered_before_initialized_is_sent():
    """Ordering, asserted on the server's own view of the sequence.

    The fake only sends ``initialized``'s predecessor a response when it sees
    the ``initialize`` request, so an implementation that fired both without
    waiting would still produce the same two lines. What pins the ordering is
    that the client blocks: the response arrives, *then* the notification.
    """
    order: list[str] = []

    def responder(srv, message):
        order.append(f"got:{message['method']}")
        if message["method"] == "initialize":
            order.append("answered:initialize")
            return [{"id": message["id"], "result": {"userAgent": "codex"}}]
        return []

    server = FakeAppServer(responder)
    client, _ = make_client(server)
    with client:
        pass
    assert order == ["got:initialize", "answered:initialize", "got:initialized"]


def test_a_handshake_that_is_never_answered_times_out():
    server = FakeAppServer(responder=lambda srv, message: [])
    client, _ = make_client(server, handshake_timeout=0.15)
    with pytest.raises(AppServerTimeout) as excinfo:
        client.start()
    assert "initialize" in str(excinfo.value)
    client.close()


# --- request / response correlation ----------------------------------------------


def test_responses_are_matched_by_id_even_when_they_arrive_out_of_order():
    pending: list[dict] = []

    def responder(srv, message):
        if message.get("method") == "initialize":
            return [{"id": message["id"], "result": {}}]
        if message.get("method") in ("thread/start", "thread/list"):
            pending.append(message)
            if len(pending) == 2:
                # Answer the second one first: correlation must be by id, not
                # by arrival order.
                return [
                    {"id": pending[1]["id"], "result": {"which": "second"}},
                    {"id": pending[0]["id"], "result": {"which": "first"}},
                ]
        return []

    server = FakeAppServer(responder)
    client, _ = make_client(server)
    results: dict[str, object] = {}
    with client:
        first = threading.Thread(
            target=lambda: results.update(
                first=client.request("thread/start", {"cwd": "/work"})
            )
        )
        first.start()
        assert wait_until(lambda: len(pending) == 1)
        results["second"] = client.request("thread/list", {})
        first.join(timeout=3)

    assert results["first"] == {"which": "first"}
    assert results["second"] == {"which": "second"}


def test_a_json_rpc_error_response_keeps_the_vendor_code_and_message():
    def responder(srv, message):
        if message["method"] == "initialize":
            return [{"id": message["id"], "result": {}}]
        if "id" not in message:
            return []
        return [
            {
                "id": message["id"],
                "error": {
                    "code": -32600,
                    "message": "thread/resume failed: no rollout found for thread id x",
                    "data": {"threadId": "x"},
                },
            }
        ]

    server = FakeAppServer(responder)
    client, _ = make_client(server)
    with client:
        with pytest.raises(AppServerRequestFailed) as excinfo:
            client.request("thread/resume", {"threadId": "x"})
    error = excinfo.value
    assert error.code == -32600
    assert "no rollout found" in error.detail
    assert error.data == {"threadId": "x"}
    assert error.method == "thread/resume"


def test_request_before_start_is_refused():
    server = FakeAppServer()
    client, _ = make_client(server)
    with pytest.raises(AppServerTransportClosed):
        client.request("thread/list", {})


def test_request_after_close_is_refused_without_writing():
    server = FakeAppServer()
    client, _ = make_client(server)
    client.start()
    client.close()
    sent = len(server.received)
    with pytest.raises(AppServerTransportClosed):
        client.request("thread/list", {})
    assert len(server.received) == sent


# --- server -> client requests ----------------------------------------------------


def test_a_server_request_is_answered_with_the_same_id_and_a_decline():
    def responder(srv, message):
        if message.get("method") == "initialize":
            return [{"id": message["id"], "result": {}}]
        return []

    server = FakeAppServer(responder)
    client, _ = make_client(server)
    with client:
        server.send(
            {
                "id": 77,
                "method": "item/commandExecution/requestApproval",
                "params": {"command": "rm -rf /"},
            }
        )
        assert wait_until(lambda: any(m.get("id") == 77 for m in server.received))
    answer = next(m for m in server.received if m.get("id") == 77)
    assert answer == {"id": 77, "result": {"decision": "decline"}}


def test_an_unknown_server_request_is_answered_with_an_empty_result():
    server = FakeAppServer(
        responder=lambda srv, m: [{"id": m["id"], "result": {}}]
        if m.get("method") == "initialize"
        else []
    )
    client, _ = make_client(server)
    with client:
        server.send({"id": 5, "method": "some/future/request", "params": {}})
        assert wait_until(lambda: any(m.get("id") == 5 for m in server.received))
    assert next(m for m in server.received if m.get("id") == 5) == {"id": 5, "result": {}}


def test_an_injected_handler_replaces_the_default():
    seen: list[tuple[str, dict]] = []

    def handler(method, params):
        seen.append((method, dict(params)))
        return {"decision": "accept"}

    server = FakeAppServer(
        responder=lambda srv, m: [{"id": m["id"], "result": {}}]
        if m.get("method") == "initialize"
        else []
    )
    client, _ = make_client(server, server_request_handler=handler)
    with client:
        server.send({"id": 9, "method": "item/tool/call", "params": {"tool": "lookup"}})
        assert wait_until(lambda: any(m.get("id") == 9 for m in server.received))
    assert seen == [("item/tool/call", {"tool": "lookup"})]
    assert next(m for m in server.received if m.get("id") == 9)["result"] == {
        "decision": "accept"
    }


def test_a_server_request_arriving_mid_request_is_still_answered():
    """The property that decides the threading model.

    ``turn/start`` does not answer until the turn is over, and an approval
    arrives in the middle of it. A client that could only read while idle would
    deadlock here: the server waits for the approval, the client waits for the
    turn.
    """

    def responder(srv, message):
        method = message.get("method")
        if method == "initialize":
            return [{"id": message["id"], "result": {}}]
        if method == "turn/start":
            srv.turn_id = message["id"]
            return [
                {
                    "id": 400,
                    "method": "item/fileChange/requestApproval",
                    "params": {"path": "/work/x"},
                }
            ]
        if method is None and message.get("id") == 400:
            # The client answered the approval while blocked in turn/start.
            srv.approval = message["result"]
            return [{"id": srv.turn_id, "result": {"status": "completed"}}]
        return []

    server = FakeAppServer(responder)
    client, _ = make_client(server)
    with client:
        result = client.request("turn/start", {"input": "hello"})
    assert result == {"status": "completed"}
    assert server.approval == {"decision": "decline"}


def test_a_handler_that_raises_still_answers_the_server():
    def handler(method, params):
        raise RuntimeError("caller's tool blew up")

    server = FakeAppServer(
        responder=lambda srv, m: [{"id": m["id"], "result": {}}]
        if m.get("method") == "initialize"
        else []
    )
    client, _ = make_client(server, server_request_handler=handler)
    with client:
        server.send({"id": 11, "method": "item/tool/call", "params": {}})
        assert wait_until(lambda: any(m.get("id") == 11 for m in server.received))
        assert next(m for m in server.received if m.get("id") == 11) == {
            "id": 11,
            "result": {},
        }
        assert "server_request_handler failed" in client.stderr_tail()


# --- notifications ----------------------------------------------------------------


def test_notifications_reach_the_queue_in_order():
    server = FakeAppServer(
        responder=lambda srv, m: [{"id": m["id"], "result": {}}]
        if m.get("method") == "initialize"
        else []
    )
    client, _ = make_client(server)
    with client:
        server.send({"method": "thread/started", "params": {"threadId": "t1"}})
        server.send({"method": "item/agentMessage/delta", "params": {"delta": "Hi"}})
        first = client.next_notification(timeout=3)
        second = client.next_notification(timeout=3)
    assert first == Notification("thread/started", {"threadId": "t1"})
    assert second == Notification("item/agentMessage/delta", {"delta": "Hi"})


def test_a_notification_without_params_arrives_with_an_empty_mapping():
    server = FakeAppServer(
        responder=lambda srv, m: [{"id": m["id"], "result": {}}]
        if m.get("method") == "initialize"
        else []
    )
    client, _ = make_client(server)
    with client:
        server.send({"method": "thread/closed"})
        assert client.next_notification(timeout=3) == Notification("thread/closed", {})


def test_the_notification_stream_ends_with_exactly_one_sentinel():
    server = FakeAppServer(
        responder=lambda srv, m: [{"id": m["id"], "result": {}}]
        if m.get("method") == "initialize"
        else []
    )
    client, _ = make_client(server)
    client.start()
    server.send({"method": "turn/started", "params": {}})
    assert client.next_notification(timeout=3).method == "turn/started"
    server.eof()
    assert client.next_notification(timeout=3) is None
    # Both the reader's EOF and the explicit close end the stream; only one
    # sentinel may be published, or a consumer sees two endings.
    client.close()
    client.close()
    with pytest.raises(queue.Empty):
        client.next_notification(timeout=0.2)


# --- failure and lifecycle ---------------------------------------------------------


def test_eof_fails_a_blocked_request_and_reports_the_stderr_tail():
    started = threading.Event()

    def responder(srv, message):
        if message.get("method") == "initialize":
            return [{"id": message["id"], "result": {}}]
        started.set()
        return []

    server = FakeAppServer(responder)
    client, _ = make_client(server)
    client.start()

    failure: list[BaseException] = []

    def call():
        try:
            client.request("turn/start", {"input": "hi"})
        except BaseException as exc:  # recording it is the assertion
            failure.append(exc)

    worker = threading.Thread(target=call)
    worker.start()
    assert started.wait(timeout=3)

    server.say_on_stderr("ERROR: codex app-server: not logged in\n")
    server.stderr_eof()
    assert wait_until(lambda: "not logged in" in client.stderr_tail())
    server.eof()

    worker.join(timeout=3)
    assert failure and isinstance(failure[0], AppServerTransportClosed)
    assert "exited" in str(failure[0])
    assert "not logged in" in str(failure[0])
    client.close()


def test_close_terminates_politely_and_is_idempotent():
    server = FakeAppServer(
        responder=lambda srv, m: [{"id": m["id"], "result": {}}]
        if m.get("method") == "initialize"
        else []
    )
    client, _ = make_client(server)
    client.start()
    client.close()
    client.close()
    client.close()
    assert server.terminated == 1
    assert server.killed == 0
    assert server.stdin.closed is True
    assert client.running is False


def test_a_child_that_ignores_terminate_is_killed():
    server = FakeAppServer(
        responder=lambda srv, m: [{"id": m["id"], "result": {}}]
        if m.get("method") == "initialize"
        else []
    )
    server.dies_on_terminate = False
    client, _ = make_client(server)
    client.start()
    client.close()
    assert server.terminated == 1
    assert server.killed == 1


def test_a_malformed_line_is_recorded_rather_than_dropped():
    server = FakeAppServer(
        responder=lambda srv, m: [{"id": m["id"], "result": {}}]
        if m.get("method") == "initialize"
        else []
    )
    client, _ = make_client(server)
    with client:
        server.send_raw("this is not json\n")
        assert wait_until(lambda: "this is not json" in client.stderr_tail())
        assert "[unroutable]" in client.stderr_tail()


def test_a_response_to_an_unknown_id_is_recorded_rather_than_dropped():
    server = FakeAppServer(
        responder=lambda srv, m: [{"id": m["id"], "result": {}}]
        if m.get("method") == "initialize"
        else []
    )
    client, _ = make_client(server)
    with client:
        server.send({"id": 9999, "result": {"stray": True}})
        assert wait_until(lambda: "9999" in client.stderr_tail())


# --- the default handler ------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(APPROVAL_REQUEST_METHODS))
def test_approvals_are_declined_by_default(method):
    assert decline_approvals(method, {}) == {"decision": "decline"}


def test_the_approval_methods_are_the_two_the_protocol_defines():
    assert APPROVAL_REQUEST_METHODS == {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }


def test_anything_else_gets_an_empty_result():
    assert decline_approvals("item/tool/call", {"tool": "x"}) == {}


# --- event mapping (S3) ------------------------------------------------------------
#
# Fixtures are built from the v2 schemas in openai/codex
# (codex-rs/app-server-protocol/schema/json/v2), read on 2026-08-31. Where a
# shape has not also been driven live, the module docstring says so; these
# fixtures carry the schema's required fields so a test cannot pass on a shape
# the server would never send.


def note(method: str, **params) -> Notification:
    return Notification(method=method, params=params)


def command_item(**overrides) -> dict:
    """A ``CommandExecutionThreadItem``, required fields present."""
    item = {
        "id": "item_cmd_1",
        "type": "commandExecution",
        "command": "ls -la",
        "commandActions": [],
        "cwd": "/work",
        "status": "completed",
    }
    item.update(overrides)
    return item


def mcp_item(**overrides) -> dict:
    """An ``McpToolCallThreadItem``, required fields present."""
    item = {
        "id": "item_mcp_1",
        "type": "mcpToolCall",
        "server": "docs",
        "tool": "search",
        "arguments": {"q": "cache"},
        "status": "completed",
    }
    item.update(overrides)
    return item


def dynamic_item(**overrides) -> dict:
    """A ``DynamicToolCallThreadItem``, required fields present."""
    item = {
        "id": "item_dyn_1",
        "type": "dynamicToolCall",
        "tool": "lookup_price",
        "arguments": {"sku": "A1"},
        "status": "completed",
    }
    item.update(overrides)
    return item


def usage_note(last: dict, total: dict, window: int | None = 272000) -> Notification:
    return note(
        "thread/tokenUsage/updated",
        threadId="t1",
        turnId="turn1",
        tokenUsage={"last": last, "total": total, "modelContextWindow": window},
    )


def breakdown(inp: int, out: int, cached: int = 0, write: int = 0) -> dict:
    """One ``TokenUsageBreakdown``, shaped **the way the wire shapes it**.

    ``inp`` is the vendor's ``inputTokens``, which is *inclusive* of ``cached``
    (verified live 2026-08-31), so ``totalTokens`` is the vendor's own
    arithmetic: ``input + output``. This helper used to add all four fields up,
    which is the same mistake the mapper made -- a fixture built to modelpass's
    convention rather than Codex's is one no test can catch the double-count
    with. ``write`` was ``0`` in every captured breakdown, so whether it too
    sits inside ``inputTokens`` is undriven and it is left out of the total.
    """
    return {
        "inputTokens": inp,
        "outputTokens": out,
        "cachedInputTokens": cached,
        "cacheWriteInputTokens": write,
        "reasoningOutputTokens": 0,
        "totalTokens": inp + out,
    }


# --- text and thinking ---------------------------------------------------------------


def test_agent_message_delta_becomes_a_text_delta():
    (event,) = map_appserver_event(
        "item/agentMessage/delta",
        {"threadId": "t", "turnId": "u", "itemId": "m1", "delta": "Hel"},
    )
    assert isinstance(event, TextDeltaEvent) and event.text == "Hel"


@pytest.mark.parametrize(
    "method", ["item/reasoning/textDelta", "item/reasoning/summaryTextDelta"]
)
def test_both_reasoning_streams_become_thinking(method):
    (event,) = map_appserver_event(method, {"itemId": "r1", "delta": "hmm"})
    assert isinstance(event, ThinkingEvent) and event.text == "hmm"


def test_a_delta_without_text_falls_back_to_a_vendor_event():
    (event,) = map_appserver_event("item/agentMessage/delta", {"itemId": "m1"})
    assert isinstance(event, VendorEvent)
    assert event.name == "item/agentMessage/delta"
    assert event.runtime is Runtime.OPENAI_SDK


# --- the delta / completed dedup rule --------------------------------------------------


def test_a_completed_message_that_never_streamed_becomes_a_text_delta():
    """A server that does not stream still produced an answer."""
    turn = AppServerTurn()
    (event,) = turn.observe(
        note(
            "item/completed",
            threadId="t",
            turnId="u",
            completedAtMs=1,
            item={"id": "m1", "type": "agentMessage", "text": "The whole answer"},
        )
    )
    assert isinstance(event, TextDeltaEvent) and event.text == "The whole answer"
    assert turn.agent_messages == ["The whole answer"]
    assert turn.final_answer == "The whole answer"


def test_a_completed_message_that_streamed_does_not_repeat_the_text():
    turn = AppServerTurn()
    streamed = [
        turn.observe(note("item/agentMessage/delta", itemId="m1", delta="The ")),
        turn.observe(note("item/agentMessage/delta", itemId="m1", delta="answer")),
    ]
    assert [e[0].text for e in streamed] == ["The ", "answer"]

    (event,) = turn.observe(
        note(
            "item/completed",
            item={"id": "m1", "type": "agentMessage", "text": "The answer"},
        )
    )
    # The item still reaches the caller whole -- nothing dropped -- but not as a
    # second text_delta, which would double the answer for anyone concatenating.
    assert isinstance(event, VendorEvent)
    assert event.name == "item/completed:agentMessage"
    assert event.data["item"]["text"] == "The answer"
    assert turn.agent_messages == ["The answer"]


def test_the_dedup_is_per_item_not_per_turn():
    """A second message in the same turn that did not stream still emits."""
    turn = AppServerTurn()
    turn.observe(note("item/agentMessage/delta", itemId="m1", delta="first"))
    turn.observe(note("item/completed", item={"id": "m1", "type": "agentMessage", "text": "first"}))
    (event,) = turn.observe(
        note("item/completed", item={"id": "m2", "type": "agentMessage", "text": "second"})
    )
    assert isinstance(event, TextDeltaEvent) and event.text == "second"
    assert turn.agent_messages == ["first", "second"]
    assert turn.final_answer == "second"


# --- tool calls and results ------------------------------------------------------------


def test_a_command_execution_start_is_attributed_to_codex_itself():
    (event,) = map_appserver_event("item/started", {"item": command_item()})
    assert isinstance(event, ToolCallEvent)
    # The exec transport's name, deliberately: one runtime identity across the
    # transport flip means a consumer filtering on it does not have to change.
    assert event.name == "command_execution"
    assert event.server == "codex"
    assert event.arguments == {"command": "ls -la"}
    assert event.id == "item_cmd_1"


def test_an_mcp_tool_call_keeps_the_server_name():
    (event,) = map_appserver_event("item/started", {"item": mcp_item()})
    assert isinstance(event, ToolCallEvent)
    assert (event.name, event.server) == ("search", "docs")
    assert event.arguments == {"q": "cache"}


def test_a_dynamic_tool_call_is_attributed_to_the_caller():
    (event,) = map_appserver_event(
        "item/started", {"item": dynamic_item(namespace="pricing")}
    )
    assert isinstance(event, ToolCallEvent)
    assert event.name == "lookup_price"
    assert event.server == CALLER_TOOL_SERVER == "caller"
    # namespace is the vendor's grouping, not a place the tool runs.
    assert event.arguments == {"sku": "A1", "__namespace": "pricing"}


def test_a_tool_item_with_no_usable_name_falls_back_to_a_vendor_event():
    (event,) = map_appserver_event(
        "item/started", {"item": {"id": "x", "type": "mcpToolCall", "server": "docs"}}
    )
    assert isinstance(event, VendorEvent)
    assert event.name == "item/started:mcpToolCall"


def test_a_completed_command_reports_its_output_and_exit_code():
    (event,) = map_appserver_event(
        "item/completed",
        {"item": command_item(aggregatedOutput="total 0\n", exitCode=0)},
    )
    assert isinstance(event, ToolResultEvent)
    assert event.content == "total 0\n"
    assert event.is_error is False


def test_a_nonzero_exit_code_is_an_error():
    (event,) = map_appserver_event(
        "item/completed",
        {"item": command_item(aggregatedOutput="nope", exitCode=1, status="failed")},
    )
    assert event.is_error is True


def test_a_declined_command_is_an_error_even_with_a_null_exit_code():
    """The 2026-08-30 lesson, carried across transports.

    ``declined`` is the sandbox refusing a command. Read from the status and not
    from the exit code, because a declined command with ``exitCode: null`` would
    otherwise be reported as a success.
    """
    (event,) = map_appserver_event(
        "item/completed",
        {"item": command_item(status="declined", aggregatedOutput="", exitCode=None)},
    )
    assert event.is_error is True


def test_an_mcp_result_is_flattened_to_text():
    (event,) = map_appserver_event(
        "item/completed",
        {"item": mcp_item(result={"content": [{"type": "text", "text": "found it"}]})},
    )
    assert isinstance(event, ToolResultEvent)
    assert event.content == "found it"
    assert event.is_error is False


def test_an_mcp_error_wins_over_the_result():
    (event,) = map_appserver_event(
        "item/completed",
        {
            "item": mcp_item(
                status="failed",
                error={"message": "server unreachable"},
                result={"content": [{"type": "text", "text": "stale"}]},
            )
        },
    )
    assert event.is_error is True
    assert event.content == "server unreachable"


def test_a_dynamic_tool_result_flattens_its_content_items():
    (event,) = map_appserver_event(
        "item/completed",
        {
            "item": dynamic_item(
                success=True,
                contentItems=[
                    {"type": "inputText", "text": "$14.00"},
                    {"type": "inputImage", "imageUrl": "data:image/png;base64,AAAA"},
                ],
            )
        },
    )
    assert isinstance(event, ToolResultEvent)
    # The image is a marker, not a data URI pasted into a text field.
    assert event.content == "$14.00\n[inputImage]"
    assert event.is_error is False


def test_an_explicit_success_false_is_an_error_whatever_the_status_says():
    (event,) = map_appserver_event(
        "item/completed",
        {"item": dynamic_item(status="completed", success=False, contentItems=[])},
    )
    assert event.is_error is True


def test_an_unrecognized_item_type_names_itself_on_the_vendor_event():
    (event,) = map_appserver_event(
        "item/completed", {"item": {"id": "w1", "type": "webSearch", "query": "x"}}
    )
    assert isinstance(event, VendorEvent)
    assert event.name == "item/completed:webSearch"
    assert event.data["item"]["query"] == "x"


# --- usage: last vs total ---------------------------------------------------------------


def test_a_usage_update_emits_the_last_breakdown_and_keeps_the_payload():
    events = map_appserver_event(
        "thread/tokenUsage/updated",
        usage_note(breakdown(100, 20, cached=8, write=4), breakdown(9000, 900)).params,
    )
    usage, vendor = events
    assert isinstance(usage, UsageEvent)
    # 100 on the wire is inclusive of the 8 cache reads, so 92 is the fresh
    # input; modelpass counts the two side by side (2026-08-31 measurement).
    assert usage.usage == TokenUsage(
        input_tokens=92, output_tokens=20, cached_input_tokens=8, cache_write_tokens=4
    )
    assert usage.scope is UsageScope.DELTA
    # total, modelContextWindow, reasoningOutputTokens and totalTokens have no
    # home on TokenUsage and are not dropped to make the normalized half tidy.
    assert isinstance(vendor, VendorEvent)
    assert vendor.data["tokenUsage"]["total"]["inputTokens"] == 9000
    assert vendor.data["tokenUsage"]["modelContextWindow"] == 272000


def test_turn_usage_sums_the_last_breakdowns_and_never_the_thread_total():
    """The trap: ``total`` is thread-cumulative and wrong for a resumed turn.

    The thread here starts at 9,000 input tokens from earlier turns. Summing
    ``total`` would report this turn as having spent all of them.
    """
    turn = AppServerTurn()
    turn.observe(usage_note(breakdown(100, 20), breakdown(9100, 920)))
    turn.observe(usage_note(breakdown(30, 5), breakdown(9130, 925)))

    assert turn.usage == TokenUsage(input_tokens=130, output_tokens=25)
    assert turn.thread_total_usage == TokenUsage(input_tokens=9130, output_tokens=925)
    assert turn.model_context_window == 272000


def test_a_usage_update_without_a_last_breakdown_is_only_a_vendor_event():
    (event,) = map_appserver_event(
        "thread/tokenUsage/updated", {"tokenUsage": {"total": breakdown(1, 2)}}
    )
    assert isinstance(event, VendorEvent)


def test_missing_token_fields_read_as_zero_rather_than_failing():
    assert token_usage_from_breakdown({"inputTokens": 7}) == TokenUsage(input_tokens=7)
    assert token_usage_from_breakdown(None) == TokenUsage()


# --- usage: Codex nests its cache read inside input (2026-08-31) ----------------------------

#: The ``last`` breakdown from the second ``thread/tokenUsage/updated`` of the
#: live capture, verbatim (fixture line 43 of
#: ``tests/fixtures/appserver/live-capture-2026-08-31.jsonl``). This is the run
#: that found the double-count: the vendor's **own** ``totalTokens`` is 13,144,
#: which is ``inputTokens + outputTokens`` and nothing else -- so the 12,032
#: cache reads are inside the 13,123, not beside them.
LIVE_LAST_BREAKDOWN = {
    "totalTokens": 13144,
    "inputTokens": 13123,
    "cachedInputTokens": 12032,
    "cacheWriteInputTokens": 0,
    "outputTokens": 21,
    "reasoningOutputTokens": 0,
}


def test_input_tokens_are_inclusive_of_the_cache_read():
    """The live numbers, and the arithmetic that has to come out of them.

    modelpass's :class:`~modelpass.types.TokenUsage` counts the cache read *beside*
    the input, so the wire number has to have it taken out. Reading the field
    straight through counted it twice and inflated every Codex run by roughly
    65% -- into the run log and into ``stop_at_tokens``, so guards fired early.

    ``total_tokens`` landing back on the vendor's own 13,144 is the assertion
    that matters: modelpass's convention is preserved, and the transports agree
    with the vendor rather than with each other's mistakes.
    ``tests/test_adapter_openai.py`` pins the same fact for ``codex exec``, and
    ``tests/test_anthropic_adapter.py`` pins that Anthropic does **not** nest.
    """
    usage = token_usage_from_breakdown(LIVE_LAST_BREAKDOWN)
    assert usage.input_tokens == 1091
    assert usage.cached_input_tokens == 12032
    assert usage.cache_write_tokens == 0
    assert usage.output_tokens == 21
    assert usage.total_tokens == LIVE_LAST_BREAKDOWN["totalTokens"] == 13144


def test_the_live_breakdown_is_the_one_the_captured_wire_carries():
    """Guard the constant above against drift from the committed capture.

    A pinned number that no longer matches the recording it claims to come from
    is worse than no fixture at all, so the fixture reads itself.
    """
    lines = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    breakdowns = [
        line["params"]["tokenUsage"]["last"]
        for line in lines
        if line.get("method") == "thread/tokenUsage/updated"
    ]
    assert LIVE_LAST_BREAKDOWN in breakdowns
    # Every captured breakdown agrees: the vendor's total is input + output.
    for last in breakdowns:
        assert last["totalTokens"] == last["inputTokens"] + last["outputTokens"]


def test_a_cache_read_larger_than_the_input_clamps_at_zero():
    """A vendor inconsistency must not produce a negative token count.

    Never observed, and deliberately handled anyway: a negative ``input_tokens``
    would reach a run record and *subtract* from a guard's running total, which
    is a worse failure than the over-count the subtraction exists to fix.
    """
    usage = token_usage_from_breakdown(
        {"inputTokens": 10, "cachedInputTokens": 99, "outputTokens": 1}
    )
    assert usage.input_tokens == 0
    assert usage.cached_input_tokens == 99


# --- turn outcome -------------------------------------------------------------------------


def completed(status: str, error: dict | None = None) -> Notification:
    turn: dict = {"id": "turn1", "items": [], "status": status}
    if error is not None:
        turn["error"] = error
    return note("turn/completed", threadId="t1", turn=turn)


def test_a_completed_turn_is_ok_and_publishes_its_ids():
    turn = AppServerTurn()
    (event,) = turn.observe(completed("completed"))
    assert isinstance(event, VendorEvent) and event.name == "turn/completed"
    assert turn.status is TerminalStatus.OK
    assert turn.reason is None
    assert (turn.thread_id, turn.turn_id) == ("t1", "turn1")


def test_a_failed_turn_carries_the_vendor_code_and_message():
    turn = AppServerTurn()
    turn.observe(
        completed(
            "failed",
            {
                "message": "stream disconnected",
                "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 502}},
                "additionalDetails": "after 3 attempts",
            },
        )
    )
    assert turn.status is TerminalStatus.ERROR
    assert turn.reason == (
        "responseStreamDisconnected: stream disconnected: after 3 attempts"
    )


def test_an_interrupted_turn_is_cancelled_not_an_error():
    turn = AppServerTurn()
    turn.observe(completed("interrupted"))
    assert turn.status is TerminalStatus.CANCELLED
    assert turn.reason == "the turn was interrupted"


def test_a_spent_allowance_is_a_clean_stop():
    turn = AppServerTurn()
    turn.observe(
        completed(
            "failed",
            {"message": "You've hit your usage limit", "codexErrorInfo": "usageLimitExceeded"},
        )
    )
    assert turn.status is TerminalStatus.QUOTA_EXHAUSTED
    assert "allowance exhausted" in (turn.reason or "")


def test_a_neighbouring_error_code_is_not_read_as_a_spent_allowance():
    """Narrow on purpose: a misread failure could trigger a metered failover."""
    turn = AppServerTurn()
    turn.observe(
        completed("failed", {"message": "slow down", "codexErrorInfo": "rateLimitExceeded"})
    )
    assert turn.status is TerminalStatus.ERROR


def test_an_unknown_turn_status_is_named_rather_than_read_as_success():
    turn = AppServerTurn()
    turn.observe(completed("someFutureStatus"))
    assert turn.status is TerminalStatus.ERROR
    assert "someFutureStatus" in (turn.reason or "")


def test_in_progress_on_a_completed_turn_is_not_a_success():
    turn = AppServerTurn()
    turn.observe(completed("inProgress"))
    assert turn.status is TerminalStatus.ERROR


# --- the error notification ------------------------------------------------------------------


def test_a_retryable_error_does_not_fail_the_turn():
    """``willRetry: true`` means the runtime intends to recover from it."""
    turn = AppServerTurn()
    (event,) = turn.observe(
        note(
            "error",
            threadId="t1",
            turnId="turn1",
            willRetry=True,
            error={"message": "502 from upstream"},
        )
    )
    assert isinstance(event, VendorEvent) and event.name == "error"
    assert turn.status is TerminalStatus.OK


def test_a_final_error_fails_the_turn():
    turn = AppServerTurn()
    turn.observe(
        note(
            "error",
            willRetry=False,
            error={"message": "unauthorized", "codexErrorInfo": "unauthorized"},
        )
    )
    assert turn.status is TerminalStatus.ERROR
    assert turn.reason == "unauthorized: unauthorized"


def test_the_turns_own_verdict_outranks_an_earlier_error_notification():
    turn = AppServerTurn()
    turn.observe(completed("failed", {"message": "the real reason"}))
    turn.observe(note("error", willRetry=False, error={"message": "a later noise line"}))
    assert turn.reason == "the real reason"


def test_an_unreadable_error_keeps_its_raw_text():
    assert turn_error_detail("plain string") == "plain string"
    assert turn_error_detail(None) == ""
    assert "unexpected" in turn_error_detail({"unexpected": "shape"})


# --- totality ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        "thread/started",
        "turn/started",
        "turn/plan/updated",
        "account/rateLimits/updated",
        "item/commandExecution/outputDelta",
        "some/method/from/a/future/release",
    ],
)
def test_everything_unrecognized_passes_through_intact(method):
    params = {"threadId": "t1", "payload": {"nested": [1, 2]}}
    (event,) = map_appserver_event(method, params)
    assert isinstance(event, VendorEvent)
    assert event.name == method
    assert event.data == params
    assert event.runtime is Runtime.OPENAI_SDK


def test_no_notification_maps_to_nothing():
    """Rule 3, as a property: every method produces at least one event."""
    turn = AppServerTurn()
    methods = [
        "item/agentMessage/delta",
        "item/reasoning/textDelta",
        "item/started",
        "item/completed",
        "thread/tokenUsage/updated",
        "turn/completed",
        "error",
        "thread/status/changed",
    ]
    for method in methods:
        assert turn.observe(note(method)), f"{method} produced no event"
