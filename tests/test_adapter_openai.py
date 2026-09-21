"""OpenAI adapter tests -- all offline (implementation plan, "Testing strategy").

Fixture payloads are real ``codex exec --json`` output captured live on
2026-08-16 (see docs/api-and-runtimes.md §2.2a) and 2026-08-30
(``tests/fixtures/tools/``, see docs/api-and-runtimes.md §2.2a), with
thread ids sanitized. No test spawns a process, imports a vendor package, or
reads a credential; the transport is a scripted fake.
"""

from __future__ import annotations

import base64
import json
import time
import tomllib
from pathlib import Path

import pytest

from modelpass.adapters.base import RunRequest, SessionRequest
from modelpass.adapters.openai import (
    _TRANSCRIPT_PREAMBLE,
    CodexSession,
    OpenAIAdapter,
    _account_profile_from_account_read,
    _display_path,
    agents_md_sources,
    chat_tool_overrides,
    extract_error_detail,
    is_missing_thread,
    is_quota_exhausted,
    map_codex_event,
    mcp_config_args,
    read_configured_cli_path,
    read_credential_status,
    render_prompt,
    render_session_prompt,
)
from modelpass.connections import Connection, CredentialRef
from modelpass.errors import (
    AuthModeMismatch,
    CapabilityNotSupported,
    InvalidSession,
    PreflightFailed,
    SessionClosed,
    SessionNotFound,
)
from modelpass.preflight import plan_launch
from modelpass.runtimes import Runtime
from modelpass.tools import ToolDef
from modelpass.types import (
    AuthMode,
    Message,
    Role,
    SessionKind,
    TerminalStatus,
    TextDeltaEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    VendorEvent,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- captured fixtures (sanitized) ----------------------------------------------

THREAD_STARTED = {"type": "thread.started", "thread_id": "00000000-feed-7000-0000-000000000001"}
TURN_STARTED = {"type": "turn.started"}
AGENT_MESSAGE = {
    "type": "item.completed",
    "item": {"id": "item_0", "type": "agent_message", "text": "OK"},
}
TURN_COMPLETED = {
    "type": "turn.completed",
    "usage": {"input_tokens": 51618, "cached_input_tokens": 2432, "output_tokens": 16},
}
# Captured verbatim shape: the server's JSON error envelope arrives as a string.
MODEL_ERROR = {
    "type": "error",
    "message": (
        '{"type":"error","status":400,"error":{"type":"invalid_request_error",'
        '"message":"The \'gpt-5.6-sol\' model requires a newer version of Codex. '
        'Please upgrade to the latest app or CLI and try again."}}'
    ),
}
TURN_FAILED = {
    "type": "turn.failed",
    "error": {"message": MODEL_ERROR["message"]},
}

SUCCESS_STREAM = [THREAD_STARTED, TURN_STARTED, AGENT_MESSAGE, TURN_COMPLETED]
FAILURE_STREAM = [THREAD_STARTED, TURN_STARTED, MODEL_ERROR, TURN_FAILED]


# --- fakes ----------------------------------------------------------------------


class FakeStdin:
    """The child's stdin pipe: what was written to it, and whether it was closed.

    ``closed`` is asserted on rather than assumed. The real CLI reads its prompt
    to EOF, so a stdin left open is a turn that never starts -- a hang, not a
    failure, and the kind of thing a fake that only recorded ``write`` would let
    through.
    """

    def __init__(self) -> None:
        self.value = ""
        self.closed = False

    def write(self, text: str) -> int:
        self.value += text
        return len(text)

    def close(self) -> None:
        self.closed = True


class FakeProc:
    """A subprocess.Popen stand-in fed with scripted JSONL payloads."""

    def __init__(self, payloads, returncode: int = 0):
        self.stdout = iter(json.dumps(p) + "\n" for p in payloads)
        self.stdin = FakeStdin()
        self.returncode = returncode
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode if self.killed else None

    def kill(self):
        self.killed = True


class FakeSpawn:
    def __init__(self, proc: FakeProc):
        self.proc = proc
        self.argv: list[str] | None = None
        self.env: dict[str, str] | None = None
        self.cwd: str | None = None

    def __call__(self, argv, env, cwd):
        self.argv, self.env, self.cwd = argv, env, cwd
        return self.proc


def codex_connection(auth_mode=AuthMode.SUBSCRIPTION) -> Connection:
    ref = (
        CredentialRef.native_login()
        if auth_mode is AuthMode.SUBSCRIPTION
        else CredentialRef.parse("env:OPENAI_API_KEY")
    )
    return Connection(
        name="codex-sub" if auth_mode is AuthMode.SUBSCRIPTION else "codex-api",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=auth_mode,
        credential_ref=ref,
    )


#: Every request built in this module selects ``codex exec`` explicitly.
#:
#: This file **is** the exec suite: its fixtures are captured ``codex exec
#: --json`` output and its fake transport speaks that stream. S7 flipped the
#: runtime's default to ``codex app-server`` on 2026-08-31, which is a change of
#: default and not a removal -- so these tests keep asserting exactly what they
#: asserted before, about exactly the transport they were written for, and the
#: option is how they say which one that is. ``tests/test_adapter_openai_
#: appserver.py`` and ``tests/test_appserver_sessions.py`` cover the default.
EXEC = {"transport": "exec"}


def request_for(
    connection: Connection, env: dict[str, str] | None = None, **kwargs
) -> RunRequest:
    plan = plan_launch(connection, env if env is not None else {"PATH": "/usr/bin"})
    kwargs.setdefault("options", EXEC)
    return RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="Reply with exactly OK"),),
        plan=plan,
        **kwargs,
    )


# --- event mapping (pure) -------------------------------------------------------


def test_success_stream_maps_to_normalized_events():
    events = [e for payload in SUCCESS_STREAM for e in map_codex_event(payload)]
    assert isinstance(events[0], VendorEvent) and events[0].name == "thread.started"
    assert isinstance(events[1], VendorEvent) and events[1].name == "turn.started"
    assert isinstance(events[2], TextDeltaEvent) and events[2].text == "OK"
    assert isinstance(events[3], UsageEvent)
    # 51,618 on the wire is inclusive of the 2,432 cache reads, so 49,186 is the
    # fresh input. See test_codex_input_tokens_are_inclusive_of_the_cache_read.
    assert events[3].usage.input_tokens == 49186
    assert events[3].usage.cached_input_tokens == 2432
    assert events[3].usage.output_tokens == 16


# --- usage: Codex nests its cache read inside input (2026-08-31) -----------------


#: The ``turn.completed`` usage block from the live exec run of 2026-08-31, the
#: capture that found the double-count. Pinned verbatim: these four numbers are
#: what the vendor sent, and the vendor's own total for them is 14,233.
LIVE_EXEC_USAGE = {
    "input_tokens": 14228,
    "cached_input_tokens": 9984,
    "cache_write_input_tokens": 0,
    "output_tokens": 5,
}


def test_codex_input_tokens_are_inclusive_of_the_cache_read():
    """The live numbers, and the arithmetic that has to come out of them.

    Measured 2026-08-31 on ``codex exec`` (and on ``codex app-server``, which
    ``tests/test_codex_appserver.py`` pins the other half of). Codex reports
    ``input_tokens`` **inclusive** of ``cached_input_tokens``, while modelpass's
    :class:`~modelpass.types.TokenUsage` counts them side by side -- so reading the
    wire field straight into ``input_tokens`` counted the cache read twice and
    inflated every Codex run by roughly 65%. That number reached the run log and
    ``stop_at_tokens``, firing guards early.

    ``total_tokens`` landing on 14,233 is the assertion that matters: it is the
    vendor's own ``input + output``, reached through modelpass's parallel
    convention rather than by abandoning it.
    """
    (event,) = map_codex_event({"type": "turn.completed", "usage": LIVE_EXEC_USAGE})
    assert isinstance(event, UsageEvent)
    assert event.usage.input_tokens == 4244
    assert event.usage.cached_input_tokens == 9984
    assert event.usage.cache_write_tokens == 0
    assert event.usage.output_tokens == 5
    assert event.usage.total_tokens == 14233


def test_a_cold_prefix_is_no_longer_reported_as_free_on_exec():
    """``cache_write_input_tokens`` rides the exec wire and was never read.

    The field is in the 2026-08-31 capture and this mapper ignored it until
    then, so a run that paid to *write* a prefix reported the write as nothing.
    Cache writes are billed as input, which is why they belong in
    ``billable_input_tokens`` and not only in the record.
    """
    (event,) = map_codex_event(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 6222,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 6220,
                "output_tokens": 12,
            },
        }
    )
    assert event.usage.cache_write_tokens == 6220
    assert event.usage.billable_input_tokens == 6222 + 6220


def test_a_cache_read_larger_than_the_input_clamps_at_zero():
    """A vendor inconsistency must not produce a negative token count.

    Never observed, and deliberately handled anyway: a negative ``input_tokens``
    would reach the run record and *subtract* from a guard's running total,
    which is a worse failure than the over-count the subtraction exists to fix.
    """
    (event,) = map_codex_event(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "cached_input_tokens": 99, "output_tokens": 1},
        }
    )
    assert event.usage.input_tokens == 0
    assert event.usage.cached_input_tokens == 99


def test_reasoning_items_become_thinking_events():
    payload = {"type": "item.completed", "item": {"type": "reasoning", "text": "hmm"}}
    (event,) = map_codex_event(payload)
    assert isinstance(event, ThinkingEvent) and event.text == "hmm"


def test_unknown_payloads_pass_through_as_vendor_events():
    payload = {"type": "some.future.event", "data": 1}
    (event,) = map_codex_event(payload)
    assert isinstance(event, VendorEvent)
    assert event.name == "some.future.event"
    assert event.data == payload


def test_error_detail_double_parses_the_embedded_envelope():
    detail = extract_error_detail(MODEL_ERROR)
    assert detail.startswith("400: ")
    assert "requires a newer version of Codex" in detail
    # turn.failed nests the same string one level deeper
    assert extract_error_detail(TURN_FAILED) == detail
    # a plain message survives unchanged
    assert extract_error_detail({"type": "error", "message": "boom"}) == "boom"


# --- prompt rendering -----------------------------------------------------------


def test_single_user_message_passes_through_verbatim():
    prompt = render_prompt((Message(role=Role.USER, content="hello there"),))
    assert prompt == "hello there"


def test_multi_message_prompt_becomes_a_labelled_transcript():
    prompt = render_prompt(
        (
            Message(role=Role.SYSTEM, content="Be terse."),
            Message(role=Role.USER, content="hi"),
            Message(role=Role.ASSISTANT, content="hello"),
            Message(role=Role.USER, content="bye"),
        )
    )
    assert "System: Be terse." in prompt
    assert prompt.index("User: hi") < prompt.index("Assistant: hello") < prompt.index("User: bye")
    assert prompt.splitlines()[-1] == "User: bye"


# --- run() ----------------------------------------------------------------------


def test_run_success_streams_events_and_terminal_ok():
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    events = list(adapter.run(request_for(codex_connection())))

    types = [e.type for e in events]
    assert types == ["vendor_event", "vendor_event", "text_delta", "usage", "terminal"]
    terminal = events[-1]
    assert terminal.status is TerminalStatus.OK
    # The cache read taken out of the wire's inclusive input, as it is on the
    # mapper (2026-08-31); the terminal carries what the run actually spent.
    assert terminal.usage.input_tokens == 49186
    assert terminal.usage.cached_input_tokens == 2432
    assert terminal.usage.output_tokens == 16


def test_run_launches_with_scrubbed_env_and_safe_flags():
    ambient = {"PATH": "/usr/bin", "OPENAI_API_KEY": "sk-live-secret", "HOME": "/home/u"}
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    list(adapter.run(request_for(codex_connection(), env=ambient)))

    assert spawn.env is not None
    assert "OPENAI_API_KEY" not in spawn.env  # the whole point
    assert spawn.env["HOME"] == "/home/u"
    assert spawn.argv is not None


def test_run_launches_an_isolated_account_with_its_codex_home(tmp_path):
    home = tmp_path / "codex-work"
    connection = Connection(
        name="openai-work",
        nickname="OpenAI Work",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        config_dir=str(home),
    )
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    list(adapter.run(request_for(connection, env={"CODEX_HOME": str(tmp_path / "wrong")})))
    assert spawn.env["CODEX_HOME"] == str(home)
    assert spawn.argv[0] == "codex"
    for flag in ("exec", "--json", "--skip-git-repo-check", "--ephemeral"):
        assert flag in spawn.argv
    # The PROMPT slot says "read it from stdin", and the prompt is there.
    assert spawn.argv[-1] == "-"
    assert spawn.proc.stdin.value == "Reply with exactly OK"
    assert spawn.proc.stdin.closed is True


def test_prompt_rides_stdin_because_cmd_truncates_argv_at_newline():
    """The defect this transport exists to prevent, pinned.

    On Windows the resolved ``codex`` is ``codex.cmd``, a batch shim, and
    ``cmd.exe`` ends its command line at the first newline: every line of a
    multi-line prompt after the first was silently dropped and the run reported
    ``ok`` anyway (reproduced 2026-08-31 with a throwaway ``.cmd`` that echoed
    its arguments; no AI involved, nothing spent).

    So the assertion is not merely "the prompt arrived" -- it is that **no argv
    element contains a newline at all**. An argv that carries one is an argv the
    shim can cut, whatever else is true of it.
    """
    connection = codex_connection()
    messages = (
        Message(role=Role.SYSTEM, content="Be terse."),
        Message(role=Role.USER, content="line one\nline two"),
        Message(role=Role.ASSISTANT, content="ack"),
        Message(role=Role.USER, content="now\nfinish\nit"),
    )
    request = RunRequest(
        connection=connection,
        messages=messages,
        plan=plan_launch(connection, {"PATH": "/usr/bin"}),
        options=EXEC,
    )
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    list(OpenAIAdapter(codex_bin="codex", spawn=spawn).run(request))

    expected = render_prompt(messages)
    assert "\n" in expected, "the fixture must actually be multi-line"
    assert spawn.argv[-1] == "-"
    assert spawn.proc.stdin.value == expected  # byte-intact, transcript and all
    assert spawn.proc.stdin.closed is True
    assert not any("\n" in arg for arg in spawn.argv)


# --- MCP passthrough and tool events (D12) --------------------------------------
#
# Item type names were extracted from the shipped codex.exe 0.117.0; the field
# names are third-party-sourced, so the mapper must degrade to vendor_event on
# anything that does not look the way it was documented. See
# docs/api-and-runtimes.md §2.2a.

MCP_CALL_STARTED = {
    "type": "item.started",
    "item": {
        "id": "item_5",
        "type": "mcp_tool_call",
        "server": "amt",
        "tool": "measure",
        "arguments": {"take": 3},
        "status": "in_progress",
    },
}
MCP_CALL_COMPLETED = {
    "type": "item.completed",
    "item": {
        "id": "item_5",
        "type": "mcp_tool_call",
        "server": "amt",
        "tool": "measure",
        "arguments": {"take": 3},
        "result": {"content": [{"type": "text", "text": "-14.2 LUFS"}]},
        "status": "completed",
    },
}


def test_an_mcp_tool_call_maps_started_to_call_and_completed_to_result():
    """One event each, not a synthesized pair -- Codex emits both halves itself."""
    (call,) = map_codex_event(MCP_CALL_STARTED)
    (result,) = map_codex_event(MCP_CALL_COMPLETED)

    assert call.type == "tool_call"
    assert (call.name, call.server, call.id) == ("measure", "amt", "item_5")
    assert call.arguments == {"take": 3}

    assert result.type == "tool_result"
    assert (result.id, result.name) == ("item_5", "measure")
    assert result.content == "-14.2 LUFS"
    assert result.is_error is False


def test_a_failed_mcp_tool_call_reports_the_error_text():
    payload = {
        "type": "item.completed",
        "item": {
            "id": "item_6",
            "type": "mcp_tool_call",
            "server": "amt",
            "tool": "measure",
            "error": {"message": "no such take"},
            "status": "failed",
        },
    }
    (result,) = map_codex_event(payload)
    assert result.is_error is True
    assert result.content == "no such take"


def test_a_command_execution_is_a_tool_call_from_the_runtime_itself():
    """Codex cannot switch its built-ins off, so name the server honestly."""
    (call,) = map_codex_event(
        {
            "type": "item.started",
            "item": {"id": "i1", "type": "command_execution", "command": "bash -lc ls"},
        }
    )
    assert (call.name, call.server) == ("command_execution", "codex")
    assert call.arguments == {"command": "bash -lc ls"}

    (result,) = map_codex_event(
        {
            "type": "item.completed",
            "item": {
                "id": "i1",
                "type": "command_execution",
                "command": "bash -lc ls",
                "aggregated_output": "a\nb",
                "exit_code": 0,
                "status": "completed",
            },
        }
    )
    assert (result.id, result.content, result.is_error) == ("i1", "a\nb", False)


def test_a_nonzero_exit_code_is_a_failed_tool_result():
    (result,) = map_codex_event(
        {
            "type": "item.completed",
            "item": {
                "id": "i1",
                "type": "command_execution",
                "aggregated_output": "nope",
                "exit_code": 127,
                "status": "failed",
            },
        }
    )
    assert result.is_error is True


# --- the live command_execution capture (2026-08-30) ----------------------------
#
# A real run whose command the Windows sandbox refused. Driven from the recorded
# payloads rather than from dicts written to match the code, so a runtime that
# changes shape breaks a test instead of quietly changing the answer.

COMMAND_EXECUTION_RUN = load_jsonl("tools/openai-command-execution-2026-08-30.jsonl")


def declined_item() -> dict:
    (item,) = [
        payload["item"]
        for payload in COMMAND_EXECUTION_RUN
        if payload.get("type") == "item.completed"
        and payload["item"].get("type") == "command_execution"
    ]
    return item


def test_the_live_capture_confirms_the_documented_command_execution_fields():
    """The field names were third-party-sourced until this run recorded them."""
    assert set(declined_item()) == {
        "id",
        "type",
        "command",
        "aggregated_output",
        "exit_code",
        "status",
    }


def test_the_live_declined_run_maps_to_a_call_and_a_failed_result():
    events = [event for payload in COMMAND_EXECUTION_RUN for event in map_codex_event(payload)]
    call = next(e for e in events if isinstance(e, ToolCallEvent))
    result = next(e for e in events if isinstance(e, ToolResultEvent))

    assert (call.id, call.name, call.server) == ("item_1", "command_execution", "codex")
    assert "powershell.exe" in call.arguments["command"]
    assert result.id == "item_1"
    assert result.is_error is True
    assert "blocked by policy" in result.content
    # started -> tool_call and completed -> tool_result, never both from one event.
    assert len([e for e in events if isinstance(e, ToolCallEvent)]) == 1
    assert len([e for e in events if isinstance(e, ToolResultEvent)]) == 1


def test_a_declined_command_is_an_error_even_without_an_exit_code():
    """The bug this fixture was captured for.

    ``declined`` is a fourth ``status`` value the docs do not list, and until
    2026-08-30 nothing matched on it: the live payload only reached
    ``is_error=True`` incidentally, through its ``exit_code`` of -1. Strip the
    exit code -- the shape the CLI emits while a command is still in flight --
    and a refused command was reported as a **success**.
    """
    item = declined_item() | {"exit_code": None}
    (result,) = map_codex_event({"type": "item.completed", "item": item})
    assert result.is_error is True

    # ... and the non-declined shapes still read the way they always did.
    completed = declined_item() | {"exit_code": 0, "status": "completed"}
    (ok,) = map_codex_event({"type": "item.completed", "item": completed})
    assert ok.is_error is False


def test_a_declined_mcp_tool_call_is_an_error_too():
    """``status`` is one field shared by both tool item types."""
    (result,) = map_codex_event(
        {
            "type": "item.completed",
            "item": {
                "id": "i9",
                "type": "mcp_tool_call",
                "server": "amt",
                "tool": "measure",
                "result": None,
                "status": "declined",
            },
        }
    )
    assert result.is_error is True


def test_a_tool_item_missing_its_documented_fields_falls_back_to_vendor_event():
    """The field names are third-party-sourced; a half-populated tool call would lie."""
    (event,) = map_codex_event(
        {"type": "item.started", "item": {"id": "x", "type": "mcp_tool_call"}}
    )
    assert isinstance(event, VendorEvent)
    assert event.name == "item.started:mcp_tool_call"


def test_untouched_item_types_still_pass_through_as_vendor_events():
    (event,) = map_codex_event(
        {"type": "item.completed", "item": {"id": "w", "type": "web_search"}}
    )
    assert isinstance(event, VendorEvent)
    assert event.name == "item.completed:web_search"


def test_mcp_servers_become_per_invocation_c_overrides():
    servers = {"amt": {"command": "python", "args": ["-m", "amt"], "env": {"A": "1"}}}
    assert mcp_config_args(servers) == [
        "-c",
        'mcp_servers.amt={command = "python", args = ["-m", "amt"], env = {A = "1"}}',
    ]


def test_an_http_mcp_server_serializes_its_url():
    assert mcp_config_args({"docs": {"url": "https://example.com/mcp"}}) == [
        "-c",
        'mcp_servers.docs={url = "https://example.com/mcp"}',
    ]


def test_no_mcp_servers_means_no_overrides_at_all():
    assert mcp_config_args({}) == []


@pytest.mark.parametrize(
    "config",
    [
        {"command": "python", "args": ["-m", "amt"], "env": {"A": "1"}},
        {"url": "https://example.com/mcp", "bearer_token_env_var": "TOKEN"},
        {"command": 'weird "quoted" \\ path', "args": []},
        {"command": "x", "enabled": True, "startup_timeout_sec": 30},
        {"command": "x", "cwd": None},  # a None is dropped, never a literal
    ],
)
def test_the_c_override_value_is_toml_codex_can_actually_parse(config):
    """The CLI parses the value portion as TOML, so verify it against tomllib.

    Same discipline as the connection-store writer: the encoder earns its
    correctness by round-tripping through the stdlib parser, not by looking
    right.
    """
    (_flag, override) = mcp_config_args({"amt": config})
    key, _, value = override.partition("=")
    assert key == "mcp_servers.amt"
    parsed = tomllib.loads(f"x = {value}")["x"]
    assert parsed == {k: v for k, v in config.items() if v is not None}


def test_a_server_name_that_would_break_the_dotted_path_is_refused():
    with pytest.raises(ValueError, match="not usable"):
        mcp_config_args({"a.b": {"command": "x"}})


def mcp_request(connection: Connection, servers, options=None) -> RunRequest:
    return RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection, {}),
        mcp_servers=servers,
        options={**EXEC, **(options or {})},
    )


def test_mcp_overrides_reach_the_command_line():
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn, mcp_list=lambda b, e: "[]")
    request = mcp_request(codex_connection(), {"amt": {"command": "python"}})
    list(adapter.run(request))

    assert spawn.argv is not None
    assert 'mcp_servers.amt={command = "python"}' in spawn.argv
    # The prompt slot stays last, so the override cannot be swallowed as its
    # value; the prompt itself rides stdin.
    assert spawn.argv[-1] == "-"
    assert spawn.proc.stdin.value == "hi"


# Shape captured live from `codex mcp list --json` on 2026-08-16 (fields beyond
# name/enabled elided; the code must not depend on them).
CONFIGURED_SERVERS = json.dumps(
    [
        {"name": "amt", "enabled": True, "disabled_reason": None},
        {"name": "node_repl", "enabled": True, "disabled_reason": None},
        {"name": "dormant", "enabled": False, "disabled_reason": "user"},
    ]
)


def test_exclusivity_disables_every_configured_server_not_requested():
    from modelpass.adapters.openai import exclusivity_disable_args

    args = exclusivity_disable_args(CONFIGURED_SERVERS, {"docs": {"url": "https://x"}})
    assert args == [
        "-c",
        "mcp_servers.amt.enabled=false",
        "-c",
        "mcp_servers.node_repl.enabled=false",
    ]  # 'dormant' is already off; requested 'docs' is not configured


def test_exclusivity_spares_a_requested_server_that_is_also_configured():
    from modelpass.adapters.openai import exclusivity_disable_args

    args = exclusivity_disable_args(CONFIGURED_SERVERS, {"amt": {"command": "python"}})
    assert "mcp_servers.amt.enabled=false" not in args
    assert "mcp_servers.node_repl.enabled=false" in args


def test_an_mcp_run_carries_the_disable_overrides():
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(
        codex_bin="codex", spawn=spawn, mcp_list=lambda b, e: CONFIGURED_SERVERS
    )
    list(adapter.run(mcp_request(codex_connection(), {"docs": {"url": "https://x"}})))
    assert spawn.argv is not None
    assert "mcp_servers.amt.enabled=false" in spawn.argv
    assert "mcp_servers.node_repl.enabled=false" in spawn.argv


def test_an_mcp_run_fails_closed_when_enumeration_fails():
    """Running without a promised exclusivity guarantee is the worse failure."""
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn, mcp_list=lambda b, e: None)
    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(adapter.run(mcp_request(codex_connection(), {"docs": {"url": "https://x"}})))
    assert "allow_configured_mcp_servers" in str(excinfo.value)
    assert spawn.argv is None  # nothing launched, nothing spent


def test_an_mcp_run_fails_closed_on_malformed_enumeration():
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(
        codex_bin="codex", spawn=spawn, mcp_list=lambda b, e: '{"not": "a list"}'
    )
    with pytest.raises(CapabilityNotSupported):
        list(adapter.run(mcp_request(codex_connection(), {"docs": {"url": "https://x"}})))
    assert spawn.argv is None


def test_the_exclusivity_opt_out_skips_enumeration_entirely():
    calls: list[str] = []

    def recording_list(binary, env):
        calls.append(binary)
        return CONFIGURED_SERVERS

    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn, mcp_list=recording_list)
    request = mcp_request(
        codex_connection(),
        {"docs": {"url": "https://x"}},
        options={"allow_configured_mcp_servers": True},
    )
    list(adapter.run(request))
    assert calls == []  # explicit opt-out: the user's servers ride along, knowingly
    assert spawn.argv is not None
    assert "mcp_servers.amt.enabled=false" not in spawn.argv


def test_a_plain_chat_run_never_enumerates_mcp_config():
    """Phase 4's live-validated launch path stays byte-identical without MCP."""
    calls: list[str] = []
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(
        codex_bin="codex",
        spawn=spawn,
        mcp_list=lambda b, e: calls.append(b) or CONFIGURED_SERVERS,
    )
    list(adapter.run(request_for(codex_connection())))
    assert calls == []
    assert spawn.argv is not None
    assert not any("mcp_servers" in a for a in spawn.argv)


def test_in_process_tools_are_refused_rather_than_silently_dropped():
    """Answering without the tools the caller asked for would be the worse failure."""
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    connection = codex_connection()
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection, {}),
        tools=(ToolDef(name="look_up", description="d", handler=lambda a: "x"),),
        options=EXEC,
    )
    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(adapter.run(request))
    assert excinfo.value.capability == "tools_in_process"
    assert spawn.argv is None


def test_run_model_pin_is_caller_supplied_not_hardcoded():
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    connection = codex_connection()
    plan = plan_launch(connection, {})
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan,
        model="gpt-5.4",
        options=EXEC,
    )
    list(adapter.run(request))
    assert spawn.argv is not None
    assert "--model" in spawn.argv and "gpt-5.4" in spawn.argv

    # and without a caller pin, no --model appears (2026-08-31 model swap survival)
    spawn2 = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter2 = OpenAIAdapter(codex_bin="codex", spawn=spawn2)
    list(adapter2.run(request_for(connection)))
    assert spawn2.argv is not None and "--model" not in spawn2.argv


def test_run_failure_stream_yields_terminal_error_with_parsed_reason():
    spawn = FakeSpawn(FakeProc(FAILURE_STREAM, returncode=1))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    events = list(adapter.run(request_for(codex_connection())))

    terminal = events[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert terminal.reason is not None and "400" in terminal.reason
    assert "requires a newer version of Codex" in terminal.reason
    # the raw failure payloads still passed through as vendor events
    names = [e.name for e in events if isinstance(e, VendorEvent)]
    assert "error" in names and "turn.failed" in names


# --- quota exhaustion (Phase 5) -------------------------------------------------
#
# The markers below come from the shipped codex.exe 0.117.0, extracted
# 2026-08-17: OpenAI documents no error taxonomy for `codex exec --json`, so the
# binary is the only first-party evidence available offline. A real exhausted
# allowance has never been observed end to end -- doing that on purpose means
# burning a real plan -- so these tests pin the *detection rule*, and the
# fallback-to-error tests below matter as much as the positive ones.


def quota_stream(message: str):
    return [
        THREAD_STARTED,
        TURN_STARTED,
        {"type": "turn.failed", "error": {"message": message}},
    ]


@pytest.mark.parametrize(
    "message",
    [
        "You've hit your usage limit. Upgrade to Pro to continue using Codex",
        '{"type":"error","status":429,"error":{"message":"Rate limit reached"}}',
        '{"error":{"code":"usage_limit_exceeded","message":"limit"}}',
        '{"error":{"code":"usageLimitExceeded"}}',
    ],
)
def test_a_spent_allowance_is_a_clean_stop_not_an_error(message):
    spawn = FakeSpawn(FakeProc(quota_stream(message), returncode=1))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    terminal = list(adapter.run(request_for(codex_connection())))[-1]
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED
    assert "allowance exhausted" in (terminal.reason or "")


@pytest.mark.parametrize(
    "message",
    [
        '{"type":"error","status":400,"error":{"message":"bad request"}}',
        "sandbox_error: command not permitted",
        "context_window_exceeded",
        '{"usage":{"input_tokens":429}}',
    ],
)
def test_anything_not_recognized_as_quota_stays_an_error(message):
    """The important direction. A misread error would trigger a configured
    failover onto metered billing for no reason, so the detection errs narrow."""
    spawn = FakeSpawn(FakeProc(quota_stream(message), returncode=1))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    terminal = list(adapter.run(request_for(codex_connection())))[-1]
    assert terminal.status is TerminalStatus.ERROR


def test_quota_detection_is_a_pure_function_of_the_payload():
    payload = {"type": "error", "message": "You've hit your usage limit."}
    assert is_quota_exhausted(payload, extract_error_detail(payload)) is True
    assert is_quota_exhausted(MODEL_ERROR, extract_error_detail(MODEL_ERROR)) is False


def test_the_receipt_says_guards_only_fire_at_the_end_on_this_runtime():
    """codex exec --json reports usage once, at turn.completed."""
    adapter = OpenAIAdapter(
        codex_bin="codex", login_status=lambda b, e: "Logged in using ChatGPT"
    )
    receipt = adapter.preflight(request_for(codex_connection()))
    assert any("end of the run only" in note for note in receipt.notes)
    assert any("interim_usage" in note for note in receipt.notes)


def test_run_nonzero_exit_without_failure_event_is_an_error():
    spawn = FakeSpawn(FakeProc([THREAD_STARTED, TURN_STARTED], returncode=3))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    events = list(adapter.run(request_for(codex_connection())))
    terminal = events[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert terminal.reason is not None and "code 3" in terminal.reason


def test_cancel_kills_the_process_and_synthesizes_cancelled():
    proc = FakeProc([THREAD_STARTED, TURN_STARTED], returncode=1)
    spawn = FakeSpawn(proc)
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    stream = adapter.run(request_for(codex_connection()))
    assert next(stream).type == "vendor_event"
    adapter.cancel()
    events = list(stream)
    assert proc.killed
    terminal = events[-1]
    assert terminal.status is TerminalStatus.CANCELLED
    assert terminal.reason is not None and "no vendor-side record" in terminal.reason


# --- preflight ------------------------------------------------------------------


def preflight_with(status_output, connection=None, env=None, account_read=None):
    connection = connection or codex_connection()
    adapter = OpenAIAdapter(
        codex_bin="codex",
        login_status=lambda b, e: status_output,
        account_read=account_read,
        # Scripted, like every other subprocess in this file: the real probe
        # runs `codex --version`, which this suite never does.
        codex_version=lambda b, e: "codex-cli 0.117.0",
    )
    return adapter.preflight(request_for(connection, env=env))


def test_preflight_detects_chatgpt_subscription_login():
    ambient = {"PATH": "/usr/bin", "OPENAI_API_KEY": "sk-live-secret"}
    receipt = preflight_with("Logged in using ChatGPT\n", env=ambient)
    assert receipt.ok
    assert receipt.detected_auth_mode is AuthMode.SUBSCRIPTION
    assert "OPENAI_API_KEY" in receipt.scrubbed
    assert "ChatGPT" in receipt.credential_source
    receipt.require_ok()
    receipt.require_auth_mode()  # requested subscription, detected subscription


def test_preflight_exposes_openai_account_identity_without_vendor_secrets():
    receipt = preflight_with(
        "Logged in using ChatGPT\n",
        account_read=lambda b, e: {
            "account": {
                "type": "chatgpt",
                "email": "person@example.com",
                "planType": "example-plan",
                "accessToken": "must-not-escape",
            }
        },
    )
    assert receipt.account == "person@example.com"
    assert receipt.plan_name == "example-plan"
    assert receipt.account_profile is not None
    assert receipt.account_profile.auth_method == "chatgpt"
    assert "must-not-escape" not in json.dumps(receipt.to_dict())


def test_openai_account_profile_ignores_unknown_top_level_payloads():
    assert _account_profile_from_account_read({"accessToken": "secret"}) is None


def _isolated_openai(home) -> Connection:
    return Connection(
        name="openai-work",
        nickname="OpenAI Work",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        config_dir=str(home),
    )


# --- credential detection (auth.json expiry) -------------------------------------


def _codex_jwt(exp: float) -> str:
    """A JWT with only an 'exp' claim -- codex-rs's token_data.rs parses the
    same claim client-side, so this is the minimum shape read_credential_status
    needs, never a real signature."""
    raw = base64.urlsafe_b64encode(json.dumps({"exp": int(exp)}).encode())
    payload = raw.rstrip(b"=").decode("ascii")
    return f"header.{payload}.sig"


def _write_auth_json(home: Path, *, access_exp: float, refresh: str) -> None:
    home.mkdir(exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps(
            {"tokens": {"access_token": _codex_jwt(access_exp), "refresh_token": refresh}}
        ),
        encoding="utf-8",
    )


def test_a_live_codex_login_is_reported_usable(tmp_path):
    home = tmp_path / "codex-work"
    _write_auth_json(home, access_exp=time.time() + 3600, refresh="r")
    status = read_credential_status(home)
    assert status.present and status.usable and not status.expired


def test_an_expired_codex_login_without_refresh_is_not_usable(tmp_path):
    home = tmp_path / "codex-work"
    _write_auth_json(home, access_exp=time.time() - 3600, refresh="")
    status = read_credential_status(home)
    assert status.present and status.expired and not status.refreshable
    assert not status.usable


def test_an_expired_but_refreshable_codex_login_stays_usable(tmp_path):
    home = tmp_path / "codex-work"
    _write_auth_json(home, access_exp=time.time() - 3600, refresh="r")
    assert read_credential_status(home).usable


def test_a_missing_codex_login_is_reported_absent(tmp_path):
    status = read_credential_status(tmp_path / "codex-work")
    assert not status.present and not status.usable


def test_an_api_key_only_auth_json_is_reported_absent_not_unusable(tmp_path):
    """{'openai_api_key': ...} with no 'tokens' entry is not a subscription
    login at all -- reported absent so preflight trusts 'codex login status'
    instead of failing a run that was never a ChatGPT login to begin with."""
    home = tmp_path / "codex-work"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps({"openai_api_key": "sk-x"}), encoding="utf-8")
    status = read_credential_status(home)
    assert not status.present


def test_isolated_openai_account_accepts_keyring_storage(tmp_path):
    """CODEX_HOME isolates keyring credentials too, so nothing is refused.

    Codex keys the keyring entry by the codex home itself -- service "Codex
    Auth", account key the SHA-256 of the canonicalized path
    (codex-rs/login/src/auth/storage.rs) -- so a second directory reads a second
    entry. modelpass used to refuse this configuration on the opposite belief.
    """
    home = tmp_path / "codex-work"
    home.mkdir()
    (home / "config.toml").write_text(
        'cli_auth_credentials_store = "keyring"\n', encoding="utf-8"
    )
    receipt = preflight_with(
        "Logged in using ChatGPT\n", connection=_isolated_openai(home)
    )
    assert receipt.ok is True
    assert any(
        "credential store: keyring" in note and "keyed to this CODEX_HOME" in note
        for note in receipt.notes
    )


def test_isolated_openai_account_accepts_auto_storage_with_no_auth_file(tmp_path):
    home = tmp_path / "codex-work"
    home.mkdir()
    (home / "config.toml").write_text(
        'cli_auth_credentials_store = "auto"\n', encoding="utf-8"
    )
    receipt = preflight_with(
        "Logged in using ChatGPT\n", connection=_isolated_openai(home)
    )
    assert receipt.ok is True
    assert any("credential store: auto" in note for note in receipt.notes)


def test_the_isolated_home_reaches_both_vendor_probes_as_codex_home(tmp_path):
    """Identity is only meaningful if the probes ran under the profile's own root."""
    home = tmp_path / "codex-work"
    home.mkdir()
    seen: dict[str, str | None] = {}

    def login_status(binary, env):
        seen["login"] = env.get("CODEX_HOME")
        return "Logged in using ChatGPT\n"

    def account_read(binary, env):
        seen["account"] = env.get("CODEX_HOME")
        return {"account": {"type": "chatgpt", "email": "work@example.com"}}

    adapter = OpenAIAdapter(
        codex_bin="codex",
        login_status=login_status,
        account_read=account_read,
        codex_version=lambda b, e: "codex-cli 0.117.0",
    )
    receipt = adapter.preflight(request_for(_isolated_openai(home)))
    assert receipt.ok is True
    assert seen == {"login": str(home), "account": str(home)}
    assert receipt.account == "work@example.com"


def test_the_openai_identity_probe_is_cached_per_binary_and_codex_home(tmp_path):
    """account/read starts a whole app-server; a chat call must not pay for one."""
    calls = []

    def account_read(binary, env):
        calls.append((binary, env.get("CODEX_HOME")))
        return {"account": {"type": "chatgpt"}}

    adapter = OpenAIAdapter(codex_bin="codex", account_read=account_read)
    for _ in range(3):
        adapter._cached_account_read("codex", {"CODEX_HOME": "/home"}, Path("/home"))
    assert calls == [("codex", "/home")]

    adapter._cached_account_read("codex", {"CODEX_HOME": "/work"}, Path("/work"))
    assert len(calls) == 2

    adapter.invalidate_identity_cache()
    adapter._cached_account_read("codex", {"CODEX_HOME": "/home"}, Path("/home"))
    assert len(calls) == 3


def test_isolated_openai_account_accepts_file_backed_auth_and_names_its_path(tmp_path):
    home = tmp_path / "codex-work"
    home.mkdir()
    (home / "config.toml").write_text(
        'cli_auth_credentials_store = "file"\n', encoding="utf-8"
    )
    (home / "auth.json").write_text("{}", encoding="utf-8")
    receipt = preflight_with(
        "Logged in using ChatGPT\n", connection=_isolated_openai(home)
    )
    assert receipt.ok is True
    assert str(home / "auth.json") in receipt.credential_source
    assert any(
        f"credential store: file -- {home / 'auth.json'}" in note
        for note in receipt.notes
    )


def test_preflight_flags_api_key_login_as_a_mismatch():
    receipt = preflight_with("Logged in using an API key\n")
    assert receipt.detected_auth_mode is AuthMode.API_KEY
    with pytest.raises(AuthModeMismatch):
        receipt.require_auth_mode()


def test_an_expired_but_refreshable_codex_login_is_verified_and_refreshed(tmp_path):
    """Parity with the Anthropic adapter's fix for the real-world failure: a
    receipt saying ok=True on a login whose refresh silently failed lets a run
    start and then die on the first token, because 'codex login status' can
    keep saying 'logged in' after a permanently-failed refresh without ever
    clearing the stored tokens. Preflight now re-checks auth.json after one
    forced CLI probe instead of trusting the text answer alone.
    """
    home = tmp_path / "codex-work"
    _write_auth_json(home, access_exp=time.time() - 10, refresh="r")
    calls = {"n": 0}

    def login_status(binary, env):
        calls["n"] += 1
        if calls["n"] == 2:
            _write_auth_json(home, access_exp=time.time() + 3600, refresh="r")
        return "Logged in using ChatGPT\n"

    adapter = OpenAIAdapter(
        codex_bin="codex",
        login_status=login_status,
        codex_version=lambda b, e: "codex-cli 0.117.0",
    )
    receipt = adapter.preflight(request_for(_isolated_openai(home)))

    assert receipt.ok
    assert calls["n"] == 2
    assert "was refreshed during preflight" in " ".join(receipt.notes)


def test_a_permanently_failed_codex_refresh_fails_closed_and_says_to_log_in_again(tmp_path):
    """auth.json still expired after the forced re-probe is exactly the
    permanent-refresh-failure shape (dead refresh_token, revoked grant) that
    'codex login status' alone does not surface -- the run must not proceed."""
    home = tmp_path / "codex-work"
    _write_auth_json(home, access_exp=time.time() - 10, refresh="r")

    adapter = OpenAIAdapter(
        codex_bin="codex",
        login_status=lambda b, e: "Logged in using ChatGPT\n",
        codex_version=lambda b, e: "codex-cli 0.117.0",
    )
    receipt = adapter.preflight(request_for(_isolated_openai(home)))

    assert not receipt.ok
    assert "could not be refreshed" in receipt.problem
    assert "codex login" in receipt.problem


def test_an_expired_codex_login_without_refresh_token_fails_closed(tmp_path):
    home = tmp_path / "codex-work"
    _write_auth_json(home, access_exp=time.time() - 10, refresh="")

    adapter = OpenAIAdapter(
        codex_bin="codex",
        login_status=lambda b, e: "Logged in using ChatGPT\n",
        codex_version=lambda b, e: "codex-cli 0.117.0",
    )
    receipt = adapter.preflight(request_for(_isolated_openai(home)))

    assert not receipt.ok
    assert "carries no refresh token" in receipt.problem


def test_preflight_fails_closed_when_not_logged_in():
    receipt = preflight_with("Not logged in\n")
    assert not receipt.ok
    with pytest.raises(PreflightFailed):
        receipt.require_ok()


def test_preflight_fails_closed_on_unrecognized_output():
    receipt = preflight_with("codex burst into song\n")
    assert not receipt.ok
    assert receipt.problem is not None and "unrecognized" in receipt.problem


def test_the_receipt_records_the_binary_as_a_field_not_only_as_a_note():
    """So the run log can name it, and "was this run affected" becomes a query.

    A note is prose for a human reading a receipt; the field is what the ledger
    stores. Two Codex builds coexist routinely and a transport-level defect in
    one of them is only attributable if the record says which one ran
    (2026-08-31).
    """
    receipt = preflight_with("Logged in using ChatGPT\n")
    assert receipt.binary == "codex"
    assert receipt.to_dict()["binary"] == "codex"

    # Every path *after* the binary is resolved names it, including the two that
    # fail: a preflight that could not establish the auth mode is exactly when
    # knowing which executable answered matters.
    unreadable = preflight_with(None)  # 'codex login status' would not run
    assert not unreadable.ok and unreadable.binary == "codex"
    unrecognized = preflight_with("codex burst into song\n")
    assert not unrecognized.ok and unrecognized.binary == "codex"


def test_a_receipt_with_no_binary_to_name_says_so_with_none():
    """anthropic-sdk never resolves a path -- its own SDK owns the subprocess."""
    from modelpass.preflight import Receipt

    connection = codex_connection()
    receipt = Receipt.from_plan(plan_launch(connection, {"PATH": "/usr/bin"}))
    assert receipt.binary is None


def test_preflight_reports_missing_binary():
    adapter = OpenAIAdapter(login_status=lambda b, e: "unreachable")
    connection = codex_connection()
    plan = plan_launch(connection, {"PATH": ""})
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan,
    )
    import shutil as _shutil

    original = _shutil.which
    _shutil.which = lambda name: None
    try:
        receipt = adapter.preflight(request)
    finally:
        _shutil.which = original
    assert not receipt.runtime_available
    assert not receipt.ok


# --- receipt disclosure: which binary, and whose instructions (2026-08-30) -------
#
# Two things shape a Codex run that modelpass cannot take away: which of possibly
# several installed binaries PATH resolves to, and the AGENTS.md files Codex
# folds into every prompt as user_instructions. Both were silent until the
# operational finding in the 2026-08-30 chat/session decision record, and
# the honest handling for an influence you cannot remove is to name it.


def codex_home_dir(tmp_path, *, config: str | None = None, agents_md: str | None = None):
    home = tmp_path / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (home / "config.toml").write_text(config, encoding="utf-8")
    if agents_md is not None:
        (home / "AGENTS.md").write_text(agents_md, encoding="utf-8")
    return home


def disclosure_receipt(home, *, cwd=None, version="codex-cli 0.117.0", binary="codex"):
    connection = codex_connection()
    adapter = OpenAIAdapter(
        codex_bin=binary,
        login_status=lambda b, e: "Logged in using ChatGPT\n",
        codex_version=lambda b, e: version,
        codex_home=home,
    )
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection, {"PATH": "/usr/bin"}),
        options=EXEC if cwd is None else {**EXEC, "cwd": str(cwd)},
    )
    return adapter.preflight(request)


def test_the_receipt_names_the_resolved_codex_binary_and_its_version(tmp_path):
    receipt = disclosure_receipt(codex_home_dir(tmp_path), binary="/opt/codex/codex")
    assert any(
        "codex binary: /opt/codex/codex (codex-cli 0.117.0)" in note for note in receipt.notes
    )


def test_a_version_the_probe_could_not_read_is_an_absence_not_a_guess(tmp_path):
    receipt = disclosure_receipt(codex_home_dir(tmp_path), version=None)
    (line,) = [note for note in receipt.notes if note.startswith("codex binary:")]
    assert line == "codex binary: codex"


def test_the_receipt_names_a_configured_codex_build_that_is_not_the_resolved_one(tmp_path):
    """The 0.117.0-shim-vs-newer-build trap, pre-announced instead of hit mid-run."""
    home = codex_home_dir(
        tmp_path, config='model = "gpt-5.6-luna"\nCODEX_CLI_PATH = "/opt/newer/codex.exe"\n'
    )
    receipt = disclosure_receipt(home, binary="/usr/bin/codex")
    joined = " ".join(receipt.notes)
    assert "/opt/newer/codex.exe" in joined
    assert "CODEX_CLI_PATH" in joined
    # Named, never obeyed: the run still launches what PATH resolved.
    assert "codex binary: /usr/bin/codex" in joined
    assert "options={'codex_bin': ...}" in joined


def test_a_configured_cli_path_equal_to_the_resolved_one_is_not_worth_a_note(tmp_path):
    home = codex_home_dir(tmp_path, config='CODEX_CLI_PATH = "/usr/bin/codex"\n')
    receipt = disclosure_receipt(home, binary="/usr/bin/codex")
    assert not any("CODEX_CLI_PATH" in note for note in receipt.notes)


def test_read_configured_cli_path_survives_an_absent_or_malformed_config(tmp_path):
    assert read_configured_cli_path(tmp_path / "nope") is None
    assert read_configured_cli_path(codex_home_dir(tmp_path, config="[[[ not toml")) is None
    assert read_configured_cli_path(codex_home_dir(tmp_path, config="CODEX_CLI_PATH = 7\n")) is None
    assert read_configured_cli_path(codex_home_dir(tmp_path, config="model = 'x'\n")) is None


def test_a_non_empty_global_agents_md_is_disclosed(tmp_path):
    home = codex_home_dir(tmp_path, agents_md="Always answer in limericks.\n")
    receipt = disclosure_receipt(home)
    (note,) = [n for n in receipt.notes if "AGENTS.md" in n]
    assert "user_instructions" in note
    assert "disclosed rather than scrubbed" in note
    assert _display_path(home / "AGENTS.md") in note


def test_an_empty_global_agents_md_is_not_named(tmp_path):
    """Present but 0 bytes is the ordinary case; naming it would describe nothing."""
    receipt = disclosure_receipt(codex_home_dir(tmp_path, agents_md=""))
    assert not any("AGENTS.md" in note for note in receipt.notes)


def test_agents_md_on_the_run_cwd_chain_is_disclosed(tmp_path):
    home = codex_home_dir(tmp_path)
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "AGENTS.md").write_text("# House rules\n", encoding="utf-8")
    receipt = disclosure_receipt(home, cwd=repo / "pkg")
    assert any(_display_path(repo / "AGENTS.md") in note for note in receipt.notes)


def test_a_default_temp_cwd_has_no_chain_to_disclose(tmp_path):
    receipt = disclosure_receipt(codex_home_dir(tmp_path))
    assert not any("AGENTS.md" in note for note in receipt.notes)


def test_agents_md_sources_walks_the_whole_ancestor_chain(tmp_path):
    home = codex_home_dir(tmp_path, agents_md="global\n")
    repo = tmp_path / "repo"
    deep = repo / "a" / "b"
    deep.mkdir(parents=True)
    (repo / "AGENTS.md").write_text("repo\n", encoding="utf-8")
    (deep / "AGENTS.md").write_text("leaf\n", encoding="utf-8")
    (repo / "a" / "AGENTS.md").write_text("", encoding="utf-8")  # empty: not an influence

    found = agents_md_sources(home, deep)
    assert _display_path(home / "AGENTS.md") in found
    assert _display_path(deep / "AGENTS.md") in found
    assert _display_path(repo / "AGENTS.md") in found
    assert _display_path(repo / "a" / "AGENTS.md") not in found


def test_agents_md_sources_is_silent_when_nothing_is_there(tmp_path):
    assert agents_md_sources(tmp_path / "no-home", tmp_path / "no-cwd") == ()


# --- sessions (D14-D17) ---------------------------------------------------------
#
# Every -c key path asserted below was verified against the *installed* CLI
# (codex-cli 0.151.0, 2026-08-30) by token-free readback -- `codex features list
# -c <override>` printing the effective value, and an invalid value on a known
# key returning a typed error naming the path. The assertions are here so a
# future upgrade that moves one of them fails a test rather than shipping an
# override the CLI silently ignores.


class FakeSessionSpawn:
    """The session spawn seam: one scripted process per turn, stderr included."""

    def __init__(self, *turns):
        # each turn is a payload list, or (payloads, returncode, stderr)
        self.turns = [t if isinstance(t, tuple) else (t, 0, "") for t in turns]
        self.calls: list[list[str]] = []
        self.cwds: list[str] = []
        self.envs: list[dict] = []
        self.procs: list[FakeProc] = []

    def __call__(self, argv, env, cwd, stderr):
        payloads, returncode, err = self.turns.pop(0) if self.turns else ([], 0, "")
        self.calls.append(list(argv))
        self.cwds.append(cwd)
        self.envs.append(env)
        if err:
            stderr.write(err)
        proc = FakeProc(payloads, returncode)
        self.procs.append(proc)
        return proc

    @property
    def argv(self) -> list[str]:
        return self.calls[-1]


NO_ROLLOUT_STDERR = (
    "Error: thread/resume: thread/resume failed: no rollout found for thread id "
    "00000000-feed-7000-0000-000000000001 (code -32600)\n"
)


def session_request(
    connection: Connection | None = None,
    *,
    kind: SessionKind = SessionKind.CHAT,
    project_folder: str = "/work/repo",
    env: dict | None = None,
    **kwargs,
) -> SessionRequest:
    resolved = connection or codex_connection()
    kwargs.setdefault("options", EXEC)
    return SessionRequest(
        connection=resolved,
        plan=plan_launch(resolved, env if env is not None else {"PATH": "/usr/bin"}),
        kind=kind,
        project_folder=project_folder,
        **kwargs,
    )


def session_adapter(spawn: FakeSessionSpawn, **kwargs) -> OpenAIAdapter:
    return OpenAIAdapter(codex_bin="codex", session_spawn=spawn, **kwargs)


# --- opening ---------------------------------------------------------------------


def test_open_session_launches_nothing_and_publishes_no_id():
    """Rule 7: a session is opened locally and created by running."""
    spawn = FakeSessionSpawn()
    session = session_adapter(spawn).open_session(session_request())

    assert session.id is None
    assert spawn.calls == []


def test_the_id_appears_only_after_the_first_turn_completes():
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(session_request())

    events = list(session.send("hello"))

    assert events[-1].status is TerminalStatus.OK
    assert session.id == THREAD_STARTED["thread_id"]


def test_a_failed_first_turn_publishes_no_id_even_though_the_thread_started():
    """thread.started fires at the start of a turn; the rollout is written at the end."""
    spawn = FakeSessionSpawn(FAILURE_STREAM)
    session = session_adapter(spawn).open_session(session_request())

    events = list(session.send("hello"))

    assert events[-1].status is TerminalStatus.ERROR
    assert session.id is None


def test_the_first_turn_is_a_plain_exec_and_the_next_one_resumes():
    spawn = FakeSessionSpawn(SUCCESS_STREAM, SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(session_request())

    list(session.send("first"))
    first = spawn.calls[0]
    assert first[1] == "exec" and "resume" not in first
    assert first[-1] == "-"
    assert spawn.procs[0].stdin.value == "first"

    list(session.send("second"))
    second = spawn.calls[1]
    assert second[1:3] == ["exec", "resume"]
    # The id keeps its position and the PROMPT slot after it holds "-".
    assert second[-2:] == [THREAD_STARTED["thread_id"], "-"]
    assert spawn.procs[1].stdin.value == "second"


def test_a_session_never_passes_ephemeral():
    """directives_for() adds --ephemeral for every openai-sdk connection because it
    was written for the stateless call; a session drops it, because here the
    rollout file is the conversation."""
    request = session_request()
    assert any(d.name == "--ephemeral" for d in request.plan.directives)

    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(request)
    list(session.send("hello"))

    assert "--ephemeral" not in spawn.argv


def test_a_turn_runs_in_the_project_folder():
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(session_request(project_folder="/work/x"))
    list(session.send("hello"))

    assert spawn.cwds == ["/work/x"]


def test_a_turn_launches_with_the_scrubbed_environment():
    ambient = {"PATH": "/usr/bin", "OPENAI_API_KEY": "sk-live-secret", "HOME": "/home/u"}
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(session_request(env=ambient))
    list(session.send("hello"))

    assert "OPENAI_API_KEY" not in spawn.envs[0]
    assert spawn.envs[0]["HOME"] == "/home/u"


# --- the toolbelt ----------------------------------------------------------------


def test_chat_tool_overrides_are_the_key_paths_verified_against_the_installed_cli():
    """Pinned deliberately. Each was read back on codex-cli 0.151.0, and two paths
    the design record named are absent because that CLI accepts and ignores them."""
    assert chat_tool_overrides() == [
        "-c", "features.shell_tool=false",
        "-c", "features.view_image=false",
        "-c", "features.browser_use=false",
        "-c", "features.image_generation=false",
        "-c", "features.apps=false",
        "-c", "web_search=disabled",
    ]
    flat = " ".join(chat_tool_overrides())
    # tools.view_image stopped being a validated key path on 0.151.0 (a bad value
    # is now silently ignored), and features.unified_exec is accepted and ignored
    # by readback -- an override that does nothing must not look like a setting.
    assert "tools.view_image" not in flat
    assert "unified_exec" not in flat


def test_a_chat_session_switches_the_runtime_toolbelt_off():
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(session_request(kind=SessionKind.CHAT))
    list(session.send("hello"))

    for override in chat_tool_overrides():
        assert override in spawn.argv
    assert "features.shell_tool=false" in spawn.argv


def test_a_worker_session_keeps_the_runtime_toolbelt_on():
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(session_request(kind=SessionKind.WORKER))
    list(session.send("hello"))

    assert not any("features." in arg for arg in spawn.argv)
    assert not any("web_search" in arg for arg in spawn.argv)


def test_a_stateless_exec_call_switches_the_runtime_toolbelt_off():
    """D23. A stateless call is chat-shaped too, and until 2026-08-31 only
    ``anthropic-sdk`` treated it that way -- this path ran with Codex's full
    coding-agent toolbelt while ``Bridge.chat``'s own docstring promised
    "built-in tools are off by construction here"."""
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    list(adapter.run(request_for(codex_connection())))

    for override in chat_tool_overrides():
        assert override in spawn.argv
    # The prompt still arrives on stdin, and its marker is still last: the
    # overrides go in front of it, never between it and the end (S1).
    assert spawn.argv[-1] == "-"


def test_a_stateless_exec_call_can_opt_back_into_the_toolbelt():
    """``options={"native_tools": True}`` is for a caller who wants Codex
    driving its own shell -- the runtime as a coding agent, not as a model."""
    spawn = FakeSpawn(FakeProc(SUCCESS_STREAM))
    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    list(adapter.run(request_for(
        codex_connection(), options={**EXEC, "native_tools": True})))

    assert not any("features." in arg for arg in spawn.argv)
    assert not any("web_search" in arg for arg in spawn.argv)
    assert spawn.argv[-1] == "-"


def test_native_tools_is_declared_so_the_receipt_never_calls_it_ignored():
    """An option the adapter reads but does not declare gets a "will be
    ignored" note stamped on the receipt of the very call that honours it."""
    assert "native_tools" in OpenAIAdapter.option_keys
    assert OpenAIAdapter.unknown_option_keys({"native_tools": True}) == ()


def test_the_toolbelt_stays_off_on_every_later_turn():
    """A prefix that changed between turns would be cache-cold on every one (D17)."""
    spawn = FakeSessionSpawn(SUCCESS_STREAM, SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(session_request())
    list(session.send("first"))
    list(session.send("second"))

    assert "features.shell_tool=false" in spawn.calls[1]


# --- the system prompt -----------------------------------------------------------


def test_render_session_prompt_layers_the_prompt_above_the_task():
    assert render_session_prompt("Be terse.", "summarize this") == (
        "System: Be terse.\n\nsummarize this"
    )
    assert render_session_prompt(None, "hi") == "hi"
    assert render_session_prompt("", "hi") == "hi"


def test_a_worker_layers_its_system_prompt_on_the_first_turn_only():
    spawn = FakeSessionSpawn(SUCCESS_STREAM, SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(
        session_request(kind=SessionKind.WORKER, system_prompt="House rules.")
    )

    list(session.send("first"))
    assert spawn.calls[0][-1] == "-"
    assert spawn.procs[0].stdin.value == "System: House rules.\n\nfirst"
    assert spawn.procs[0].stdin.closed is True

    # The thread holds it now; restating it would move the cached prefix.
    list(session.send("second"))
    assert spawn.calls[1][-1] == "-"
    assert spawn.procs[1].stdin.value == "second"


# --- resume ----------------------------------------------------------------------


def test_resume_session_reports_its_id_immediately():
    """The caller supplied a durable id and the runtime accepted it; there is
    nothing provisional left to be quiet about."""
    spawn = FakeSessionSpawn()
    session = session_adapter(spawn).resume_session(
        session_request(resume_id="00000000-feed-7000-0000-000000000001")
    )
    assert session.id == "00000000-feed-7000-0000-000000000001"
    assert spawn.calls == []


def test_a_resumed_session_resumes_on_its_very_first_turn():
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).resume_session(session_request(resume_id="abc-123"))
    list(session.send("carry on"))

    assert spawn.argv[1:3] == ["exec", "resume"]
    assert spawn.argv[-2:] == ["abc-123", "-"]
    assert spawn.procs[0].stdin.value == "carry on"  # raw: no system prompt on a resume


def test_a_resume_whose_thread_is_gone_raises_session_not_found():
    """Captured live 2026-08-30: exit 1, nothing on stdout, the reason on stderr."""
    spawn = FakeSessionSpawn(([], 1, NO_ROLLOUT_STDERR))
    session = session_adapter(spawn).resume_session(session_request(resume_id="gone-1"))

    with pytest.raises(SessionNotFound) as excinfo:
        list(session.send("carry on"))
    assert "gone-1" in str(excinfo.value)
    assert "no rollout" in str(excinfo.value)


def test_a_first_turn_that_fails_is_a_terminal_not_a_missing_session():
    """Nothing was being resumed, so there is no thread to have gone missing."""
    spawn = FakeSessionSpawn(([], 1, NO_ROLLOUT_STDERR))
    session = session_adapter(spawn).open_session(session_request())

    events = list(session.send("hello"))
    assert events[-1].status is TerminalStatus.ERROR


def test_resume_session_refuses_a_system_prompt():
    adapter = session_adapter(FakeSessionSpawn())
    with pytest.raises(InvalidSession) as excinfo:
        adapter.resume_session(session_request(resume_id="abc", system_prompt="new rules"))
    assert "middle of a conversation" in str(excinfo.value)


def test_is_missing_thread_reads_the_captured_stderr_and_nothing_looser():
    assert is_missing_thread(NO_ROLLOUT_STDERR)
    assert not is_missing_thread("Error: stream disconnected before the first token")
    assert not is_missing_thread("")


# --- failure reporting -----------------------------------------------------------


def test_a_non_zero_exit_names_what_the_cli_said_on_stderr():
    """run() can only report the code; a session keeps stderr, which is the whole
    reason it has its own spawn seam."""
    spawn = FakeSessionSpawn(([], 3, "Error: the sandbox refused to start\n"))
    session = session_adapter(spawn).open_session(session_request())

    terminal = list(session.send("hello"))[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert "code 3" in terminal.reason
    assert "the sandbox refused to start" in terminal.reason


def test_a_vendor_failure_payload_still_ends_the_turn_with_a_terminal():
    spawn = FakeSessionSpawn(FAILURE_STREAM)
    session = session_adapter(spawn).open_session(session_request())

    events = list(session.send("hello"))
    assert events[-1].status is TerminalStatus.ERROR
    assert "requires a newer version of Codex" in events[-1].reason


def test_a_spent_allowance_is_a_clean_stop_on_a_session_too():
    quota = {"type": "turn.failed", "error": {"message": "You have hit your usage limit"}}
    spawn = FakeSessionSpawn([THREAD_STARTED, TURN_STARTED, quota])
    session = session_adapter(spawn).open_session(session_request())

    terminal = list(session.send("hello"))[-1]
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED


def test_a_session_reports_a_missing_binary_rather_than_raising():
    adapter = session_adapter(FakeSessionSpawn())
    session = CodexSession(adapter=adapter, request=session_request(), binary=None)

    terminal = list(session.send("hello"))[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert terminal.reason == "codex CLI not found"


# --- two agent messages in one turn ----------------------------------------------


def test_both_agent_messages_in_one_turn_reach_the_caller():
    """Live 2026-08-30: one turn emitted a full answer and then a shorter
    restatement. Each completed agent_message becomes its own text_delta, so a
    caller concatenating deltas gets both -- nothing overwrites anything."""
    second = {
        "type": "item.completed",
        "item": {"id": "item_1", "type": "agent_message", "text": "Short version."},
    }
    spawn = FakeSessionSpawn(
        [THREAD_STARTED, TURN_STARTED, AGENT_MESSAGE, second, TURN_COMPLETED]
    )
    session = session_adapter(spawn).open_session(session_request())

    texts = [e.text for e in session.send("hello") if isinstance(e, TextDeltaEvent)]
    assert texts == ["OK", "Short version."]


# --- MCP -------------------------------------------------------------------------


def test_mcp_servers_are_declared_on_every_turn_and_exclusivity_resolved_once():
    calls: list[str] = []

    def mcp_list(binary, env):
        calls.append(binary)
        return json.dumps([{"name": "theirs", "enabled": True}])

    spawn = FakeSessionSpawn(SUCCESS_STREAM, SUCCESS_STREAM)
    adapter = session_adapter(spawn, mcp_list=mcp_list)
    session = adapter.open_session(
        session_request(mcp_servers={"docs": {"command": "npx", "args": ["-y", "docs-mcp"]}})
    )

    list(session.send("first"))
    list(session.send("second"))

    for argv in spawn.calls:
        assert any(arg.startswith("mcp_servers.docs=") for arg in argv)
        assert "mcp_servers.theirs.enabled=false" in argv
    # Resolved once: a config change mid-conversation must not move the prefix.
    assert calls == ["codex"]


def test_a_session_fails_closed_when_mcp_exclusivity_cannot_be_established():
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    adapter = session_adapter(spawn, mcp_list=lambda b, e: None)
    session = adapter.open_session(session_request(mcp_servers={"docs": {"command": "npx"}}))

    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(session.send("hello"))
    assert "allow_configured_mcp_servers" in str(excinfo.value)
    assert spawn.calls == []


# --- refusals --------------------------------------------------------------------


def test_a_session_refuses_in_process_tools():
    adapter = session_adapter(FakeSessionSpawn())
    tool = ToolDef(name="look_up", description="d", handler=lambda a: "x")
    with pytest.raises(CapabilityNotSupported) as excinfo:
        adapter.open_session(session_request(tools=(tool,)))
    assert excinfo.value.capability == "tools_in_process"


def test_a_session_refuses_persist_false():
    adapter = session_adapter(FakeSessionSpawn())
    with pytest.raises(CapabilityNotSupported) as excinfo:
        adapter.open_session(session_request(persist=False))
    assert excinfo.value.capability == "ephemeral_multi_turn"
    assert "series of one-shots" in str(excinfo.value)


def test_list_sessions_refuses_rather_than_reporting_an_empty_account():
    """And since S7 the refusal points at a default, not at an option to add."""
    adapter = session_adapter(FakeSessionSpawn())
    with pytest.raises(CapabilityNotSupported) as excinfo:
        adapter.list_sessions(session_request())
    assert excinfo.value.capability == "sessions_list"
    message = str(excinfo.value)
    # The caller opted out of the transport that answers, so the fix is to drop
    # an option rather than to add one -- and the message has to say which.
    assert "DEFAULT transport" in message
    assert "options={'transport': 'exec'}" in message


# --- history and lifecycle -------------------------------------------------------


def test_history_is_empty_because_exec_cannot_read_a_thread_back():
    session = session_adapter(FakeSessionSpawn()).open_session(session_request())
    assert session.history() == ()


def test_close_is_idempotent_and_a_closed_session_refuses_to_send():
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(session_request())
    session.close()
    session.close()

    with pytest.raises(SessionClosed):
        list(session.send("hello"))


def test_closing_does_not_delete_the_thread():
    """A persisted session stays resumable; close ends modelpass's hold on it."""
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(session_request())
    list(session.send("hello"))
    thread_id = session.id
    session.close()

    assert session.id == thread_id


def test_an_abandoned_turn_terminates_the_process():
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).open_session(session_request())
    stream = session.send("hello")
    next(stream)
    stream.close()

    assert spawn.procs[0].killed


def test_a_resume_advisory_does_not_fail_the_turn():
    """Captured live on 0.151.0: resuming with a different model emits
    ``item.completed`` items of type ``error`` and carries on. They are advisories
    wrapped in an item, not the top-level ``error`` payload that ends a run, so
    they reach the caller as vendor events and the turn still completes."""
    advisory = {
        "type": "item.completed",
        "item": {
            "id": "item_0",
            "type": "error",
            "message": (
                "This session was recorded with model `gpt-5.4-mini` but is resuming "
                "with `gpt-5.4`."
            ),
        },
    }
    spawn = FakeSessionSpawn(
        [THREAD_STARTED, advisory, TURN_STARTED, AGENT_MESSAGE, TURN_COMPLETED]
    )
    session = session_adapter(spawn).resume_session(session_request(resume_id="abc-123"))

    events = list(session.send("carry on"))
    warned = [e for e in events if isinstance(e, VendorEvent) and "error" in e.name]
    assert warned and warned[0].name == "item.completed:error"
    assert events[-1].status is TerminalStatus.OK


def test_a_resumed_turn_keeps_the_id_the_caller_supplied():
    """thread.started is re-emitted on resume; it is the same thread, not a new one."""
    spawn = FakeSessionSpawn(SUCCESS_STREAM)
    session = session_adapter(spawn).resume_session(
        session_request(resume_id=THREAD_STARTED["thread_id"])
    )
    list(session.send("carry on"))
    assert session.id == THREAD_STARTED["thread_id"]


def test_a_single_shot_with_instructions_is_layered_not_enveloped():
    """system + user is one shot, not a conversation to be continued.

    Wrapping it in the transcript envelope would ask the model to continue a
    dialogue that never happened, and would move the prefix a scoring loop
    depends on keeping still.
    """
    rendered = render_prompt(
        (
            Message(Role.SYSTEM, "You are a rubric."),
            Message(Role.USER, "Score this."),
        )
    )
    assert rendered == "System: You are a rubric.\n\nScore this."
    assert _TRANSCRIPT_PREAMBLE not in rendered


def test_a_real_conversation_still_gets_the_envelope():
    """Three turns have nowhere else to live in `codex exec`."""
    rendered = render_prompt(
        (
            Message(Role.USER, "one"),
            Message(Role.ASSISTANT, "two"),
            Message(Role.USER, "three"),
        )
    )
    assert _TRANSCRIPT_PREAMBLE in rendered
    assert "Assistant: two" in rendered
