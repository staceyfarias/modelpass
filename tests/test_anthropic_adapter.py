"""Offline tests for the Anthropic adapter.

No network, no credential, no vendor package import at module scope. The event
mapping is a pure function of a vendor message, so it is tested against stand-in
objects shaped exactly like the SDK's dataclasses (matched by class name, which
is what the mapper dispatches on).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from modelpass.adapters.anthropic import (
    STRUCTURED_OUTPUT_MAX_TURNS,
    AnthropicAdapter,
    _account_profile_from_auth_status,
    _run_coroutine_threadsafe,
    _tool_options,
    assert_expected_auth_mode,
    build_caller_tool_server,
    map_message,
    parse_cli_version,
    session_info_from_sdk,
    session_messages_to_history,
    split_messages,
    split_tool_name,
    supports_prompt_cache_ttl,
    token_usage,
)
from modelpass.adapters.anthropic import _scrubbed_process_env as scrubbed_process_env
from modelpass.adapters.base import RunRequest, SessionRequest
from modelpass.connections import Guards
from modelpass.errors import (
    AuthModeMismatch,
    CapabilityNotSupported,
    InvalidSession,
    SessionNotFound,
    SubpassError,
    VendorRunFailed,
)
from modelpass.guards import GuardTracker
from modelpass.preflight import plan_launch
from modelpass.runtimes import Runtime
from modelpass.tools import ToolDef
from modelpass.types import (
    AuthMode,
    Message,
    ReceiptEvent,
    Role,
    SessionKind,
    StructuredOutputEvent,
    TerminalEvent,
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


@pytest.fixture
def request_factory(subscription_connection):
    """Build a RunRequest with whatever tool surface a test needs."""

    def make(**kwargs) -> RunRequest:
        return RunRequest(
            connection=subscription_connection,
            messages=(Message(Role.USER, "hi"),),
            plan=plan_launch(subscription_connection, {}),
            **kwargs,
        )

    return make


@pytest.fixture
def subscription_request(request_factory) -> RunRequest:
    return request_factory()

# --- stand-ins for the SDK's message dataclasses -------------------------------
# Matched by class name, exactly as claude_agent_sdk names them.


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str = ""


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: bool = False


@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserMessage:
    content: Any = field(default_factory=list)
    parent_tool_use_id: str | None = None


@dataclass
class AssistantMessage:
    content: list[Any] = field(default_factory=list)
    model: str = "claude-sonnet-5"
    error: str | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class StreamEvent:
    event: dict[str, Any]
    uuid: str = "u"
    session_id: str = "s"


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
    structured_output: Any = None


@dataclass
class RateLimitInfo:
    status: str
    rate_limit_type: str | None = None
    resets_at: float | None = None
    utilization: float | None = None
    overage_status: str | None = None
    overage_resets_at: float | None = None
    overage_disabled_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitEvent:
    rate_limit_info: RateLimitInfo


# --- message serialization -----------------------------------------------------


def test_single_user_message_is_passed_through_verbatim():
    """A one-turn chat must not be wrapped: the wrapper would change the ask."""
    system, prompt = split_messages((Message(Role.USER, "Reply with exactly OK"),))
    assert system is None
    assert prompt == "Reply with exactly OK"


def test_system_messages_become_the_system_prompt():
    system, prompt = split_messages(
        (Message(Role.SYSTEM, "be terse"), Message(Role.USER, "hi"))
    )
    assert system == "be terse"
    assert prompt == "hi"


def test_history_is_serialized_into_the_prompt():
    """v1 is stateless (D7): history rides in the prompt, not a session."""
    _, prompt = split_messages(
        (
            Message(Role.USER, "one"),
            Message(Role.ASSISTANT, "two"),
            Message(Role.USER, "three"),
        )
    )
    assert "Human: one" in prompt
    assert "Assistant: two" in prompt
    assert prompt.rstrip().endswith("Assistant:")


# --- usage ---------------------------------------------------------------------


def test_cache_creation_is_its_own_field_and_a_guard_still_sees_it():
    """A guard must count cache writes; a reader must be able to tell them apart.

    Folding writes into ``input_tokens`` satisfied the first and broke the
    second -- a cold prefix and genuinely new content became one number, which
    is exactly the distinction "is my caching working" turns on.
    """
    usage = token_usage(
        {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_creation_input_tokens": 6,
            "cache_read_input_tokens": 3,
        }
    )
    assert usage == TokenUsage(
        input_tokens=10,
        output_tokens=4,
        cached_input_tokens=3,
        cache_write_tokens=6,
    )
    # The guard's number is unchanged by the split.
    assert usage.billable_input_tokens == 16
    assert usage.total_tokens == 23


def test_missing_usage_is_zero_not_an_error():
    assert token_usage(None) == TokenUsage()
    assert token_usage({}) == TokenUsage()


def test_anthropic_input_tokens_are_not_reduced_by_the_cache_read():
    """**Do not "fix" this mapper the way the Codex mappers were fixed.**

    On 2026-08-31 both Codex mappers gained
    ``input_tokens = max(0, wire_input - cached)``, because Codex reports
    ``input_tokens`` *inclusive* of ``cached_input_tokens`` and modelpass was
    double-counting the cache read on every run. Anthropic does the opposite:
    it reports the two **in parallel**, and a live row read
    ``input_tokens 2`` beside ``cache_read_input_tokens 1901`` -- a shape no
    nested field can produce, since a nested one could never be smaller than the
    part it contains.

    So applying the same subtraction here would clamp almost every cached
    Anthropic run's fresh input to zero and under-report what it spent. This
    test exists to fail loudly if someone propagates the Codex correction across
    the runtimes on the assumption that the vendors agree. They do not; that is
    the whole finding.
    """
    usage = token_usage(
        {"input_tokens": 2, "output_tokens": 40, "cache_read_input_tokens": 1901}
    )
    assert usage.input_tokens == 2
    assert usage.cached_input_tokens == 1901
    assert usage.total_tokens == 1943


# --- event mapping -------------------------------------------------------------


def test_text_blocks_become_text_deltas():
    events = map_message(AssistantMessage(content=[TextBlock("OK")]))
    assert events == [TextDeltaEvent("OK")]


def test_thinking_blocks_become_thinking_events():
    events = map_message(AssistantMessage(content=[ThinkingBlock("hmm", "sig")]))
    assert events == [ThinkingEvent("hmm")]


def test_partial_text_suppresses_assembled_blocks():
    """With token deltas streaming, the assembled block would be a duplicate."""
    events = map_message(AssistantMessage(content=[TextBlock("OK")]), partial_text=True)
    assert events == []


def test_stream_event_text_delta_becomes_a_text_delta():
    msg = StreamEvent(
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "O"}}
    )
    assert map_message(msg, partial_text=True) == [TextDeltaEvent("O")]


def test_stream_event_thinking_delta_becomes_thinking():
    msg = StreamEvent(
        event={
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "why"},
        }
    )
    assert map_message(msg, partial_text=True) == [ThinkingEvent("why")]


def test_unknown_blocks_become_vendor_events_never_dropped():
    """Blocks outside the normalized vocabulary survive as vendor_event.

    ``ToolUseBlock`` used to land here; Phase 6 normalized it (see the tool
    mapping tests below), so this now guards the rule with a block the mapper
    genuinely does not know -- which is the case the rule exists for.
    """

    @dataclass
    class ServerToolUseBlock:
        id: str
        name: str
        input: dict[str, Any] = field(default_factory=dict)

    events = map_message(
        AssistantMessage(content=[ServerToolUseBlock("id1", "web_search", {"q": "x"})])
    )
    assert len(events) == 1
    assert isinstance(events[0], VendorEvent)
    assert events[0].name == "assistant.ServerToolUseBlock"


# --- tool events (D12) ---------------------------------------------------------
#
# Shapes match the 2026-08-16 tools pass: ToolUseBlock(id, name,
# input) inside an AssistantMessage; ToolResultBlock(tool_use_id, content,
# is_error) inside a UserMessage.


def test_a_tool_use_block_becomes_a_tool_call_split_into_server_and_name():
    events = map_message(
        AssistantMessage(
            content=[ToolUseBlock("tu_1", "mcp__caller__look_up", {"topic": "codex"})]
        )
    )
    assert events == [
        ToolCallEvent(name="look_up", arguments={"topic": "codex"}, id="tu_1", server="caller")
    ]


def test_an_mcp_server_tool_reports_the_server_it_came_from():
    events = map_message(
        AssistantMessage(content=[ToolUseBlock("tu_2", "mcp__amt__measure", {})])
    )
    assert events[0].server == "amt"
    assert events[0].name == "measure"


def test_an_unqualified_tool_name_is_reported_as_a_builtin():
    """Should not happen while tools=[] holds -- so make it visibly odd if it does."""
    events = map_message(AssistantMessage(content=[ToolUseBlock("tu_3", "Bash", {"c": "ls"})]))
    assert events[0].server == "builtin"
    assert events[0].name == "Bash"


def test_a_tool_result_arrives_on_a_user_message_and_correlates_by_id():
    names: dict[str, str] = {}
    call = map_message(
        AssistantMessage(content=[ToolUseBlock("tu_1", "mcp__caller__look_up", {})]),
        tool_names=names,
    )
    result = map_message(
        UserMessage(content=[ToolResultBlock("tu_1", "the answer")]), tool_names=names
    )
    assert result == [ToolResultEvent(id="tu_1", name="look_up", content="the answer")]
    assert result[0].id == call[0].id


def test_a_tool_result_without_a_remembered_call_is_unnamed_not_guessed():
    events = map_message(UserMessage(content=[ToolResultBlock("tu_9", "x")]), tool_names={})
    assert events[0].name == ""
    assert events[0].content == "x"


def test_a_failed_tool_result_is_flagged():
    events = map_message(UserMessage(content=[ToolResultBlock("tu_1", "boom", is_error=True)]))
    assert events[0].is_error is True


def test_content_blocks_are_flattened_to_text_and_non_text_blocks_are_named():
    """Images cannot be normalized honestly, but their presence must survive."""
    events = map_message(
        UserMessage(
            content=[
                ToolResultBlock(
                    "tu_1",
                    [
                        {"type": "text", "text": "first"},
                        {"type": "image", "data": "...", "mimeType": "image/png"},
                        {"type": "text", "text": "second"},
                    ],
                )
            ]
        )
    )
    assert events[0].content == "first\n[image]\nsecond"


def test_a_user_message_of_plain_text_produces_nothing():
    """Under the stateless contract the caller's own turns never come back."""
    assert map_message(UserMessage(content="hello")) == []


def test_an_unknown_user_block_still_becomes_a_vendor_event():
    @dataclass
    class WeirdBlock:
        id: str = "w"

    events = map_message(UserMessage(content=[WeirdBlock()]))
    assert isinstance(events[0], VendorEvent)
    assert events[0].name == "user.WeirdBlock"


@pytest.mark.parametrize(
    ("qualified", "expected"),
    [
        ("mcp__caller__look_up", ("caller", "look_up")),
        ("mcp__a__b__c", ("a", "b__c")),
        ("mcp__onlyserver", ("builtin", "mcp__onlyserver")),
        ("mcp__server__", ("builtin", "mcp__server__")),
        ("Read", ("builtin", "Read")),
    ],
)
def test_tool_name_splitting(qualified, expected):
    assert split_tool_name(qualified) == expected


# --- caller tools as an in-process MCP server ----------------------------------


class FakeSdk:
    """Stands in for claude_agent_sdk's tool/create_sdk_mcp_server pair.

    Shaped from the installed 0.2.139 signatures (see the verification doc), so
    the wiring is testable without the vendor package present.
    """

    def __init__(self):
        self.servers = []

    def tool(self, name, description, schema, annotations=None):
        def decorate(handler):
            return {
                "name": name,
                "description": description,
                "schema": schema,
                "handler": handler,
            }

        return decorate

    def create_sdk_mcp_server(self, name, version="1.0.0", tools=None):
        server = {"type": "sdk", "name": name, "version": version, "tools": tools or []}
        self.servers.append(server)
        return server


def test_caller_tools_are_wrapped_with_json_schema_not_python_types():
    sdk = FakeSdk()
    tool = ToolDef(
        name="look_up",
        description="Look it up.",
        parameters={"type": "object", "properties": {"topic": {"type": "string"}}},
        handler=lambda args: "x",
    )
    server = build_caller_tool_server(sdk, [tool], "caller")
    assert server["name"] == "caller"
    assert server["tools"][0]["schema"] == tool.json_schema()


def test_a_sync_handler_is_invoked_off_the_loop_and_its_text_becomes_content():
    sdk = FakeSdk()
    seen = {}

    def handler(args):
        seen.update(args)
        return f"found {args['topic']}"

    server = build_caller_tool_server(
        sdk, [ToolDef(name="look_up", description="d", handler=handler)], "caller"
    )
    result = asyncio.run(server["tools"][0]["handler"]({"topic": "codex"}))

    assert seen == {"topic": "codex"}
    assert result == {"content": [{"type": "text", "text": "found codex"}]}


def test_an_async_handler_is_awaited_directly():
    sdk = FakeSdk()

    async def handler(args):
        return "async answer"

    server = build_caller_tool_server(
        sdk, [ToolDef(name="look_up", description="d", handler=handler)], "caller"
    )
    result = asyncio.run(server["tools"][0]["handler"]({}))
    assert result["content"][0]["text"] == "async answer"


def test_a_raising_handler_becomes_a_failed_tool_result_not_a_dead_run():
    """The model can retry or explain a failed tool; a dead run wastes paid turns."""
    sdk = FakeSdk()

    def handler(args):
        raise ValueError("no such topic")

    server = build_caller_tool_server(
        sdk, [ToolDef(name="look_up", description="d", handler=handler)], "caller"
    )
    result = asyncio.run(server["tools"][0]["handler"]({}))
    assert result["is_error"] is True
    assert "ValueError: no such topic" in result["content"][0]["text"]


def test_a_handler_returning_mcp_content_is_passed_through():
    sdk = FakeSdk()
    payload = {"content": [{"type": "image", "data": "x", "mimeType": "image/png"}]}
    server = build_caller_tool_server(
        sdk, [ToolDef(name="render", description="d", handler=lambda a: payload)], "caller"
    )
    assert asyncio.run(server["tools"][0]["handler"]({})) == payload


def test_a_handler_returning_structured_data_is_json_encoded():
    sdk = FakeSdk()
    server = build_caller_tool_server(
        sdk,
        [ToolDef(name="stats", description="d", handler=lambda a: {"count": 3})],
        "caller",
    )
    result = asyncio.run(server["tools"][0]["handler"]({}))
    assert json.loads(result["content"][0]["text"]) == {"count": 3}


# --- run options ---------------------------------------------------------------


class FakeClient:
    """A ClaudeSDKClient that connects, yields nothing, and disconnects."""

    def __init__(self, options):
        self.options = options

    async def connect(self, prompt):
        return None

    async def receive_response(self):
        return
        yield  # pragma: no cover - makes this an async generator

    async def disconnect(self):
        return None

    async def interrupt(self):
        return None


class RecordingSdk(FakeSdk):
    """Records the ClaudeAgentOptions kwargs a run would launch with."""

    def __init__(self):
        super().__init__()
        self.options_kwargs: dict = {}
        self.prompt_file_text: str | None = None

    def ClaudeAgentOptions(self, **kwargs):
        self.options_kwargs = kwargs
        prompt = kwargs.get("system_prompt")
        if isinstance(prompt, dict) and prompt.get("type") == "file":
            self.prompt_file_text = Path(prompt["path"]).read_text(encoding="utf-8")
        return kwargs

    def ClaudeSDKClient(self, options):
        return FakeClient(options)


def launch_options(monkeypatch, request) -> dict:
    """Drive run() far enough to capture what it would hand the SDK."""
    sdk = RecordingSdk()
    adapter = AnthropicAdapter()
    monkeypatch.setattr(AnthropicAdapter, "_sdk", staticmethod(lambda: sdk))
    list(adapter.run(request))
    return sdk.options_kwargs


def _reasoning_request(connection, level) -> RunRequest:
    connection = replace(connection, reasoning=level)
    return RunRequest(connection=connection, messages=(Message(Role.USER, "hi"),),
                      plan=plan_launch(connection, {}))


def test_reasoning_none_launches_with_thinking_disabled_and_no_effort(
    monkeypatch, subscription_connection
):
    """2026-09-23: 'none' used to reach the wire as effort='low'."""
    options = launch_options(
        monkeypatch, _reasoning_request(subscription_connection, "none"))
    assert options["thinking"] == {"type": "disabled"}
    assert "effort" not in options


def test_a_reasoning_rung_launches_as_effort_and_leaves_thinking_alone(
    monkeypatch, subscription_connection
):
    options = launch_options(
        monkeypatch, _reasoning_request(subscription_connection, "low"))
    assert options["effort"] == "low"
    assert "thinking" not in options


def test_no_reasoning_stated_sets_neither_option(monkeypatch, subscription_request):
    options = launch_options(monkeypatch, subscription_request)
    assert "effort" not in options and "thinking" not in options


def test_a_run_removes_every_builtin_tool_not_just_the_ones_we_listed(
    monkeypatch, subscription_request
):
    """Workspace access is the risk here, so pin the mechanism, not the list.

    ``tools=[]`` is the SDK's "remove every built-in" switch and survives Claude
    Code adding a tool nobody here has heard of; the explicit deny list cannot.
    """
    options = launch_options(monkeypatch, subscription_request)
    assert options["tools"] == []
    assert "Bash" in options["disallowed_tools"]
    assert options["setting_sources"] == []


def test_a_plain_chat_run_asks_for_no_tools_and_one_turn(monkeypatch, subscription_request):
    options = launch_options(monkeypatch, subscription_request)
    assert options["max_turns"] == 1
    assert options["allowed_tools"] == []
    # No MCP server at all -- which is "an empty server set under strict
    # config", not "no MCP option". Leaving the option out let Claude Code load
    # the account's claude.ai connectors into every plain call (traced
    # 2026-09-29); this assertion used to pin that absence.
    assert options.get("mcp_servers", {}) == {}


def test_a_run_passes_the_connections_claude_config_dir_to_the_sdk(
    monkeypatch, subscription_request, tmp_path
):
    root = tmp_path / "claude-work"
    connection = replace(subscription_request.connection, config_dir=str(root))
    request = replace(
        subscription_request,
        connection=connection,
        plan=plan_launch(connection, {"CLAUDE_CONFIG_DIR": str(tmp_path / "wrong")}),
    )
    options = launch_options(monkeypatch, request)
    assert options["env"]["CLAUDE_CONFIG_DIR"] == str(root)


def test_run_options_cannot_replace_the_connections_claude_config_dir(
    monkeypatch, subscription_request, tmp_path
):
    root = tmp_path / "claude-work"
    connection = replace(subscription_request.connection, config_dir=str(root))
    request = replace(
        subscription_request,
        connection=connection,
        plan=plan_launch(connection, {}),
        options={"env": {"CLAUDE_CONFIG_DIR": str(tmp_path / "wrong"), "EXTRA": "1"}},
    )
    options = launch_options(monkeypatch, request)
    assert options["env"]["CLAUDE_CONFIG_DIR"] == str(root)
    assert options["env"]["EXTRA"] == "1"


def test_a_tool_run_reaches_the_sdk_with_the_server_and_no_invented_ceiling(
    monkeypatch, request_factory
):
    request = request_factory(
        tools=(ToolDef(name="look_up", description="d", handler=lambda a: "x"),),
    )
    options = launch_options(monkeypatch, request)
    assert options["tools"] == []  # built-ins still off in a tool run
    assert options["allowed_tools"] == ["mcp__caller__look_up"]
    assert options["mcp_servers"]["caller"]["name"] == "caller"
    # Vendor default (run to completion) -- no invented turn budget. The spend
    # bound is the guard system's, in tokens; a cap exists only if the caller
    # sets one.
    assert options["max_turns"] is None
    assert options["strict_mcp_config"] is True


def test_a_caller_can_override_the_turn_ceiling(monkeypatch, request_factory):
    request = request_factory(
        tools=(ToolDef(name="look_up", description="d", handler=lambda a: "x"),),
        options={"max_turns": 3},
    )
    assert launch_options(monkeypatch, request)["max_turns"] == 3


# --- D22: an absent system_prompt means the minimal prompt, tools or not ---------
#
# The open question D21 left, now closed. These pin the *decision*, not merely
# today's code: the tempting alternative is to map a tool-bearing chat call onto
# the ``claude_code`` preset the way a WorkerSession does, and the reason that is
# wrong is that this call never switches the runtime's own toolbelt on. A refactor
# that reintroduced the preset here would land a coding-agent persona, guidance
# for tools that are not present, and a machine-scoped cache prefix on every
# scoring loop in the four consuming projects.


def test_a_chat_call_with_no_system_prompt_sends_none_not_the_preset(
    monkeypatch, subscription_request
):
    options = launch_options(monkeypatch, subscription_request)
    assert options["system_prompt"] is None


def test_a_chat_call_with_tools_still_sends_no_system_prompt(monkeypatch, request_factory):
    """D22's whole argument, in one assertion pair.

    ``tools=`` here is the caller's own functions on an in-process MCP server.
    The runtime's built-ins stay off, so there is no toolbelt for the preset's
    guidance to be about -- which is what makes this call different from the
    ``WorkerSession`` that D15 maps to the preset.
    """
    request = request_factory(
        tools=(ToolDef(name="look_up", description="d", handler=lambda a: "x"),),
    )
    options = launch_options(monkeypatch, request)
    assert options["system_prompt"] is None
    assert options["tools"] == []


def test_a_chat_calls_system_prompt_is_the_callers_string_verbatim(
    monkeypatch, subscription_connection
):
    """Passing one sidesteps the question, and nothing is wrapped around it."""
    request = RunRequest(
        connection=subscription_connection,
        messages=(Message(Role.SYSTEM, "Score the answer 1-5."), Message(Role.USER, "hi")),
        plan=plan_launch(subscription_connection, {}),
    )
    assert launch_options(monkeypatch, request)["system_prompt"] == "Score the answer 1-5."


def test_a_large_stateless_system_prompt_uses_a_file_and_cleans_it_up(
    monkeypatch, subscription_connection
):
    """Large scoring profiles must not become an oversized CLI argument."""
    prompt = "candidate profile and rubric\n" * 1000
    request = RunRequest(
        connection=subscription_connection,
        messages=(Message(Role.SYSTEM, prompt), Message(Role.USER, "score these jobs")),
        plan=plan_launch(subscription_connection, {}),
    )
    sdk = RecordingSdk()
    adapter = AnthropicAdapter()
    monkeypatch.setattr(AnthropicAdapter, "_sdk", staticmethod(lambda: sdk))

    list(adapter.run(request))

    option = sdk.options_kwargs["system_prompt"]
    assert option["type"] == "file"
    assert sdk.prompt_file_text == prompt
    assert not Path(option["path"]).exists()


def test_a_loop_close_race_does_not_leak_an_unawaited_coroutine(monkeypatch):
    class PendingCoroutine:
        closed = False

        def close(self):
            self.closed = True

    coroutine = PendingCoroutine()
    monkeypatch.setattr(
        asyncio,
        "run_coroutine_threadsafe",
        lambda value, loop: (_ for _ in ()).throw(RuntimeError("loop closed")),
    )

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="loop closed"):
            _run_coroutine_threadsafe(coroutine, loop)
    finally:
        loop.close()

    assert coroutine.closed is True


# --- the cache TTL this runtime actually pins (D20) ------------------------------
#
# The receipt must report what modelpass *does*, not what the runtime supports. The
# version gate is the case worth pinning: setting a variable an older CLI ignores
# costs nothing at the runtime and everything at the receipt, because modelpass
# would be reporting a pin that never happened.


def cache_disclosure(monkeypatch, request, *, version: str | None):
    adapter = AnthropicAdapter()
    # The fake SDK is the subject here, so availability is faked with it: the
    # disclosure has to be exercised where the extra is not installed too.
    monkeypatch.setattr(AnthropicAdapter, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(AnthropicAdapter, "_sdk", staticmethod(FakeSdk))
    monkeypatch.setattr(
        AnthropicAdapter, "_cli_status", staticmethod(lambda sdk: ("/usr/bin/claude", None))
    )
    monkeypatch.setattr(AnthropicAdapter, "_cli_version", lambda self, b, e: version)
    return adapter.cache_eligibility(request)


def test_a_session_reports_the_pinned_hour(monkeypatch, subscription_connection):
    request = SessionRequest(
        connection=subscription_connection,
        plan=plan_launch(subscription_connection, {}),
        kind=SessionKind.CHAT,
        project_folder=".",
        system_prompt="x" * 5000,
    )
    eligibility = cache_disclosure(monkeypatch, request, version="2.1.242")
    assert eligibility.ttl == "1h"
    assert "pinned to 1h" in eligibility.ttl_detail


def test_a_cli_too_old_reports_no_pin_rather_than_one_it_would_not_get(
    monkeypatch, subscription_connection
):
    request = SessionRequest(
        connection=subscription_connection,
        plan=plan_launch(subscription_connection, {}),
        kind=SessionKind.CHAT,
        project_folder=".",
        system_prompt="x" * 5000,
    )
    eligibility = cache_disclosure(monkeypatch, request, version="2.1.236")
    assert eligibility.ttl is None
    assert "2.1.242 or later" in eligibility.ttl_detail


def test_a_stateless_call_reports_the_runtime_default_because_it_pins_nothing(
    monkeypatch, subscription_request
):
    """D17 pins per session and not here, so the receipt says the default applies."""
    eligibility = cache_disclosure(monkeypatch, subscription_request, version="2.1.242")
    assert eligibility.ttl is None
    assert "runtime's own default for a stateless call" in eligibility.ttl_detail


def test_a_worker_is_reported_as_carrying_the_preset_in_front(
    monkeypatch, subscription_connection
):
    """Its prefix is the preset plus an append, so a short append is not sub-floor."""
    request = SessionRequest(
        connection=subscription_connection,
        plan=plan_launch(subscription_connection, {}),
        kind=SessionKind.WORKER,
        project_folder=".",
        system_prompt="Prefer small diffs.",
    )
    eligibility = cache_disclosure(monkeypatch, request, version="2.1.242")
    assert eligibility.preset_prefix
    assert eligibility.likely_eligible


def test_a_chat_session_is_judged_on_the_callers_string_alone(
    monkeypatch, subscription_connection
):
    """Replace means the caller's string *is* the prefix -- nothing sits in front."""
    request = SessionRequest(
        connection=subscription_connection,
        plan=plan_launch(subscription_connection, {}),
        kind=SessionKind.CHAT,
        project_folder=".",
        system_prompt="Score 1-5.",
    )
    eligibility = cache_disclosure(monkeypatch, request, version="2.1.242")
    assert not eligibility.preset_prefix
    assert not eligibility.likely_eligible


def test_a_plain_chat_run_keeps_the_phase_3_single_turn_setup(subscription_request):
    """Phase 6 must be invisible to a run that asked for no tools."""
    assert _tool_options(FakeSdk(), subscription_request) == {}


def test_a_tool_run_removes_the_turn_ceiling_and_restricts_the_tool_set(
    request_factory,
):
    sdk = FakeSdk()
    tool = ToolDef(name="look_up", description="d", handler=lambda a: "x")
    request = request_factory(tools=(tool,), mcp_servers={"amt": {"command": "python"}})

    options = _tool_options(sdk, request)

    # max_turns=1 would forbid a tool result ever arriving; None is the vendor
    # default (run to completion), chosen over any invented number.
    assert options["max_turns"] is None
    assert set(options["mcp_servers"]) == {"caller", "amt"}
    # Caller tools enumerated exactly; a named MCP server gets a wildcard because
    # its tool list is the server's business, not ours.
    assert options["allowed_tools"] == ["mcp__caller__look_up", "mcp__amt__*"]
    assert options["strict_mcp_config"] is True


def test_mcp_servers_alone_need_no_caller_server(request_factory):
    options = _tool_options(FakeSdk(), request_factory(mcp_servers={"amt": {"command": "p"}}))
    assert set(options["mcp_servers"]) == {"amt"}
    assert options["allowed_tools"] == ["mcp__amt__*"]


def test_a_server_named_caller_is_refused_rather_than_silently_shadowed(request_factory):
    request = request_factory(
        tools=(ToolDef(name="t", description="d", handler=lambda a: "x"),),
        mcp_servers={"caller": {"command": "python"}},
    )
    with pytest.raises(CapabilityNotSupported, match="reserved"):
        _tool_options(FakeSdk(), request)


def test_unknown_message_types_become_vendor_events():
    class SomethingNew:
        pass

    events = map_message(SomethingNew())
    assert len(events) == 1
    assert isinstance(events[0], VendorEvent)
    assert events[0].name == "message.SomethingNew"


def test_system_message_becomes_a_namespaced_vendor_event():
    events = map_message(SystemMessage("api_retry", {"attempt": 1}))
    assert events == [VendorEvent(Runtime.ANTHROPIC_SDK, "system.api_retry", {"attempt": 1})]


def test_result_yields_usage_then_vendor_then_terminal():
    events = map_message(
        ResultMessage(usage={"input_tokens": 3, "output_tokens": 2}, total_cost_usd=0.01)
    )
    assert isinstance(events[0], UsageEvent)
    assert events[0].usage == TokenUsage(input_tokens=3, output_tokens=2)
    assert isinstance(events[1], VendorEvent)
    assert isinstance(events[2], TerminalEvent)
    assert events[2].status is TerminalStatus.OK


# --- interim usage: what lets a guard interrupt a tool loop (Phase 5) -----------


def test_an_assistant_turn_reports_its_own_usage_as_it_finishes():
    """AssistantMessage.usage is per-message, verified in the SDK's own parser."""
    events = map_message(
        AssistantMessage(
            content=[TextBlock("OK")],
            usage={"input_tokens": 10, "output_tokens": 4},
        )
    )
    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usage_events) == 1
    assert usage_events[0].usage == TokenUsage(input_tokens=10, output_tokens=4)
    assert usage_events[0].scope is UsageScope.DELTA
    # Reported after the turn's content, so a caller sees the work before its cost.
    assert isinstance(events[0], TextDeltaEvent)


def test_an_assistant_turn_without_usage_reports_none():
    events = map_message(AssistantMessage(content=[TextBlock("OK")]))
    assert not any(isinstance(e, UsageEvent) for e in events)


def test_the_final_result_usage_is_a_run_total_not_another_delta():
    """Otherwise the run's own total would be added on top of the turns it covers."""
    events = map_message(ResultMessage(usage={"input_tokens": 3}))
    assert events[0].scope is UsageScope.RUN_TOTAL


def test_a_tool_loop_is_counted_once_across_its_turns():
    """The whole point: per-turn guards, then a final total that does not double up."""
    tracker = GuardTracker(Guards(stop_at_tokens=1_000), "claude-sub")
    stream = [
        AssistantMessage(content=[ToolUseBlock("tu_1", "mcp__caller__x", {})],
                         usage={"input_tokens": 100, "output_tokens": 10}),
        AssistantMessage(content=[TextBlock("done")],
                         usage={"input_tokens": 120, "output_tokens": 20}),
        ResultMessage(usage={"input_tokens": 220, "output_tokens": 30}),
    ]
    for message in stream:
        for event in map_message(message):
            if isinstance(event, UsageEvent):
                tracker.observe(event.usage, event.scope)
    assert tracker.total == TokenUsage(input_tokens=220, output_tokens=30)
    assert tracker.stopped is False


def test_a_runaway_tool_loop_trips_the_guard_before_the_result_arrives():
    tracker = GuardTracker(Guards(stop_at_tokens=200), "claude-sub")
    fired = []
    for _ in range(3):
        message = AssistantMessage(
            content=[ToolUseBlock("tu", "mcp__caller__x", {})],
            usage={"input_tokens": 100},
        )
        for event in map_message(message):
            if isinstance(event, UsageEvent):
                fired.extend(tracker.observe(event.usage, event.scope))
        if tracker.stopped:
            break
    assert tracker.stopped is True
    assert tracker.total.total_tokens == 200  # stopped on the second turn, not the third
    assert fired


def test_cost_never_reaches_a_normalized_event():
    """D7: tokens, never dollars. total_cost_usd may only ride in vendor_event."""
    events = map_message(ResultMessage(total_cost_usd=1.23, usage={"input_tokens": 1}))
    vendor = [e for e in events if isinstance(e, VendorEvent)]
    assert vendor[0].data["total_cost_usd"] == 1.23
    for event in events:
        if isinstance(event, VendorEvent):
            continue
        assert "1.23" not in json.dumps(event.to_dict())


def test_error_result_with_success_subtype_is_still_an_error():
    """The 401 case observed live: subtype='success' but is_error=True."""
    msg = ResultMessage(
        subtype="success",
        is_error=True,
        api_error_status=401,
        result="Failed to authenticate. API Error: 401 OAuth access token is invalid.",
    )
    terminal = map_message(msg)[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert "401" in terminal.reason


def test_budget_subtype_is_a_clean_guard_stop_not_an_error():
    terminal = map_message(ResultMessage(subtype="error_max_budget_usd", is_error=True))[-1]
    assert terminal.status is TerminalStatus.GUARD_STOP


def test_aborted_result_maps_to_cancelled():
    terminal = map_message(ResultMessage(terminal_reason="aborted_streaming"))[-1]
    assert terminal.status is TerminalStatus.CANCELLED


def test_http_429_maps_to_quota_exhausted():
    terminal = map_message(ResultMessage(is_error=True, api_error_status=429))[-1]
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED


def test_rejected_rate_limit_event_is_quota_exhausted():
    info = RateLimitInfo(status="rejected", rate_limit_type="seven_day")
    events = map_message(RateLimitEvent(info))
    assert isinstance(events[0], VendorEvent)
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED


def test_quota_exhaustion_is_a_clean_stop_and_not_an_error():
    """The plan ran out; nothing failed (D4)."""
    terminal = map_message(RateLimitEvent(RateLimitInfo(status="rejected")))[-1]
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED
    assert terminal.status is not TerminalStatus.ERROR


def test_the_allowance_the_event_carries_reaches_the_caller():
    """RateLimitInfo is the only allowance signal either v1 runtime offers."""
    info = RateLimitInfo(
        status="rejected",
        rate_limit_type="seven_day_opus",
        utilization=1.0,
        resets_at=1_800_000_000,
        overage_status="rejected",
        overage_disabled_reason="not enabled for this account",
        raw={"anything": "else"},
    )
    events = map_message(RateLimitEvent(info))
    data = events[0].data
    assert data["rate_limit_type"] == "seven_day_opus"
    assert data["utilization"] == 1.0
    assert data["overage_disabled_reason"] == "not enabled for this account"
    assert data["raw"] == {"anything": "else"}
    reason = events[-1].reason
    assert "seven_day_opus" in reason
    assert "100% used" in reason
    assert "resets at" in reason
    assert "overage unavailable" in reason


def test_an_allowance_warning_is_information_not_a_guard_event():
    """guard_warning means a threshold the *user* configured was crossed."""
    events = map_message(
        RateLimitEvent(RateLimitInfo(status="allowed_warning", utilization=0.9))
    )
    assert len(events) == 1
    assert isinstance(events[0], VendorEvent)
    assert events[0].data["utilization"] == 0.9


def test_a_rate_limit_event_with_no_info_does_not_explode():
    events = map_message(RateLimitEvent(None))
    assert isinstance(events[0], VendorEvent)
    assert len(events) == 1


def test_adapter_never_stamps_terminal_identity():
    """Adapter contract rule 2: the bridge owns the connection/auth-mode stamp."""
    terminal = map_message(ResultMessage())[-1]
    assert terminal.connection == ""


# --- auth-mode cross-check -----------------------------------------------------


def test_init_reporting_an_api_key_fails_a_subscription_run_closed():
    init = SystemMessage("init", {"apiKeySource": "ANTHROPIC_API_KEY"})
    with pytest.raises(AuthModeMismatch):
        assert_expected_auth_mode(init, AuthMode.SUBSCRIPTION)


def test_init_reporting_no_key_satisfies_a_subscription_run():
    init = SystemMessage("init", {"apiKeySource": "none"})
    assert_expected_auth_mode(init, AuthMode.SUBSCRIPTION)


def test_init_reporting_no_key_fails_an_api_key_run_closed():
    with pytest.raises(AuthModeMismatch):
        assert_expected_auth_mode(SystemMessage("init", {"apiKeySource": "none"}), AuthMode.API_KEY)


def test_non_init_messages_are_ignored_by_the_cross_check():
    assert_expected_auth_mode(SystemMessage("status", {}), AuthMode.SUBSCRIPTION)
    assert_expected_auth_mode(AssistantMessage(), AuthMode.SUBSCRIPTION)


# --- login evidence: `claude auth status`, and never the credential store -------
#
# Owner decision, 2026-11-13: modelpass does not open Claude Code's credential
# store on any platform. Until then a fake ``.credentials.json`` stood behind
# most of these tests; each one now states the same expectation against the
# probe's answer, and says so where the expectation itself had to change.

#: A token-shaped string for a credential file nobody may read.
_POISON = "sk-ant-oat01-POISON-must-never-be-read"

_LOGGED_IN = {
    "loggedIn": True,
    "authMethod": "claude.ai",
    "apiProvider": "firstParty",
    "email": "person@example.com",
    "orgId": "org-123",
    "orgName": "Example Co",
    "subscriptionType": "max",
    "planTier": "max_5x",
    "loginExpiresAt": "2099-01-02T03:04:05.678Z",
}


def _preflight_adapter(monkeypatch, *, auth_status):
    """A subscription adapter wired to a fake CLI. No subprocess ever runs:
    both probes are injected."""
    from modelpass.adapters import anthropic as module

    adapter = module.AnthropicAdapter(
        claude_version=lambda binary, env: "2.1.233 (Claude Code)",
        auth_status=auth_status,
    )
    monkeypatch.setattr(module.AnthropicAdapter, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(module.AnthropicAdapter, "_sdk", staticmethod(lambda: object()))
    monkeypatch.setattr(
        module.AnthropicAdapter,
        "_cli_status",
        staticmethod(lambda _sdk: ("/usr/bin/claude", None)),
    )
    return adapter, module


def _profile_request(subscription_connection, root):
    connection = replace(subscription_connection, config_dir=str(root))
    return RunRequest(connection=connection, messages=(), plan=plan_launch(connection, {}))


def _default_request(subscription_connection):
    return RunRequest(
        connection=subscription_connection,
        messages=(),
        plan=plan_launch(subscription_connection, {}),
    )


def _poison_credential_file(root) -> Path:
    """What Claude Code keeps under a config directory, unreadable on purpose:
    not JSON, and holding a token-shaped string that must turn up nowhere."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".credentials.json"
    path.write_text('{"claudeAiOauth": {"accessToken": "' + _POISON, encoding="utf-8")
    return path


def _forbid_the_credential_file(monkeypatch) -> list[str]:
    """Fail the test the moment anything opens, reads or even looks for it."""
    import builtins
    import io

    touched: list[str] = []

    def guard(original):
        def wrapper(target, *args, **kwargs):
            if ".credentials.json" in str(target):
                touched.append(str(target))
                raise AssertionError(f"the credential file was touched: {target}")
            return original(target, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(io, "open", guard(io.open))
    monkeypatch.setattr(os, "open", guard(os.open))
    monkeypatch.setattr(os, "stat", guard(os.stat))
    for name in ("open", "read_text", "read_bytes", "is_file", "exists", "stat"):
        monkeypatch.setattr(Path, name, guard(getattr(Path, name)))
    return touched


def test_a_reported_login_is_a_subscription_receipt(
    monkeypatch, subscription_connection, tmp_path
):
    """Was: a live credential file is reported usable, with its plan."""
    calls: list[str | None] = []

    def auth_status(binary, env):
        calls.append(env.get("CLAUDE_CONFIG_DIR"))
        return dict(_LOGGED_IN)

    adapter, _ = _preflight_adapter(monkeypatch, auth_status=auth_status)
    receipt = adapter.preflight(_profile_request(subscription_connection, tmp_path))

    assert receipt.ok, receipt.problem
    assert receipt.detected_auth_mode is AuthMode.SUBSCRIPTION
    assert receipt.plan_name == "max"
    # The probe ran inside the connection's own config directory.
    assert calls == [str(tmp_path)]
    assert "claude auth status" in receipt.credential_source
    assert str(tmp_path) in receipt.credential_source
    assert ".credentials.json" not in receipt.credential_source
    joined = " ".join(receipt.notes)
    assert "plan tier: max_5x" in joined
    assert "login expires on 2099-01-02 (claude auth status)" in joined


def test_the_default_directory_is_named_as_the_default(monkeypatch, subscription_connection):
    adapter, _ = _preflight_adapter(monkeypatch, auth_status=lambda b, e: dict(_LOGGED_IN))
    receipt = adapter.preflight(_default_request(subscription_connection))
    assert receipt.ok, receipt.problem
    assert "default config directory" in receipt.credential_source


def test_the_receipt_says_what_it_can_no_longer_know_before_a_run(
    monkeypatch, subscription_connection
):
    """Was: an expired access token with a refresh token proceeds, with a note
    dating the expiry. The token's expiry lived in the credential file, so the
    receipt now says it was not checked rather than staying silent about it."""
    adapter, _ = _preflight_adapter(
        monkeypatch, auth_status=lambda b, e: {"loggedIn": True}
    )
    receipt = adapter.preflight(_default_request(subscription_connection))

    assert receipt.ok, receipt.problem
    joined = " ".join(receipt.notes)
    assert "does not open Claude Code's credential store" in joined
    assert "access-token expiry and refresh are not checked before a run" in joined
    assert "claude /login" in joined
    # Nothing reported, nothing claimed.
    assert "login expires" not in joined
    assert "plan tier" not in joined
    assert "rate limit tier" not in joined
    assert "past its expiry" not in joined
    assert receipt.plan_name is None


def test_a_reported_expiry_that_has_passed_is_stated_and_does_not_refuse(
    monkeypatch, subscription_connection
):
    """What ``loginExpiresAt`` measures is the vendor's to define, so a date in
    the past is reported and the run is left to find out."""
    adapter, _ = _preflight_adapter(
        monkeypatch,
        auth_status=lambda b, e: {"loggedIn": True, "loginExpiresAt": "2020-03-04T00:00:00Z"},
    )
    receipt = adapter.preflight(_default_request(subscription_connection))
    assert receipt.ok, receipt.problem
    joined = " ".join(receipt.notes)
    assert "2020-03-04" in joined and "has passed" in joined
    assert "authentication error" in joined


@pytest.mark.parametrize("raw", [0, 1767225600000, "soon", "", None, True, ["2099-01-01"]])
def test_an_expiry_that_is_not_a_date_is_no_expiry(monkeypatch, subscription_connection, raw):
    """Was: a blanked ``expiresAt`` is no expiry rather than the epoch."""
    adapter, _ = _preflight_adapter(
        monkeypatch, auth_status=lambda b, e: {"loggedIn": True, "loginExpiresAt": raw}
    )
    receipt = adapter.preflight(_default_request(subscription_connection))
    assert receipt.ok, receipt.problem
    joined = " ".join(receipt.notes)
    assert "login expires" not in joined and "has passed" not in joined
    assert "1970" not in joined


def test_a_logged_out_cli_is_refused_with_the_directory_and_the_fix(
    monkeypatch, subscription_connection, tmp_path
):
    """Was: a missing login, and a blanked file, are refused with `claude /login`."""
    adapter, _ = _preflight_adapter(
        monkeypatch,
        # Logged out, with the plan the account last had still reported.
        auth_status=lambda b, e: {"loggedIn": False, "subscriptionType": "max"},
    )
    receipt = adapter.preflight(_profile_request(subscription_connection, tmp_path))

    assert receipt.ok is False
    assert receipt.detected_auth_mode is None
    assert receipt.plan_name is None
    assert "no Claude Code login found" in receipt.problem
    assert "claude auth status reports no login" in receipt.problem
    assert "claude /login" in receipt.problem
    assert "CLAUDE_CONFIG_DIR set to that directory" in receipt.problem
    assert str(tmp_path) in receipt.problem
    with pytest.raises(SubpassError) as raised:
        receipt.require_ok()
    assert subscription_connection.name in str(raised.value)


def test_a_logged_out_default_directory_keeps_the_bare_instruction(
    monkeypatch, subscription_connection
):
    adapter, _ = _preflight_adapter(monkeypatch, auth_status=lambda b, e: {"loggedIn": False})
    receipt = adapter.preflight(_default_request(subscription_connection))
    assert not receipt.ok
    assert "no Claude Code login found" in receipt.problem
    assert "run 'claude /login' (or 'claude setup-token'" in receipt.problem
    assert "CLAUDE_CONFIG_DIR" not in receipt.problem


def test_a_probe_that_returns_nothing_fails_closed(
    monkeypatch, subscription_connection, tmp_path
):
    """No evidence is not a login. It has its own diagnosis, because "log in"
    is the wrong first instruction for a CLI that printed nothing."""
    adapter, _ = _preflight_adapter(monkeypatch, auth_status=lambda b, e: None)
    for check in (adapter.preflight, adapter.identity_preflight):
        receipt = check(_profile_request(subscription_connection, tmp_path))
        assert receipt.ok is False
        assert receipt.detected_auth_mode is None
        assert "could not confirm a Claude Code login" in receipt.problem
        assert "claude auth status" in receipt.problem
        assert "claude /login" in receipt.problem
        assert str(tmp_path) in receipt.problem


@pytest.mark.parametrize(
    "answer",
    [
        {"email": "person@example.com", "subscriptionType": "max"},
        {"loggedIn": None},
        {"loggedIn": "true"},
        {"loggedIn": 1},
    ],
)
def test_only_an_explicit_logged_in_true_passes(monkeypatch, subscription_connection, answer):
    adapter, _ = _preflight_adapter(monkeypatch, auth_status=lambda b, e: dict(answer))
    receipt = adapter.preflight(_default_request(subscription_connection))
    assert receipt.ok is False
    assert receipt.detected_auth_mode is None
    assert "could not confirm" in receipt.problem


def test_the_adapter_never_opens_the_credential_file(
    monkeypatch, subscription_connection, tmp_path
):
    """The guarantee itself. A credential file sits in both places Claude Code
    would keep one, unparseable and holding a token-shaped string; every way of
    opening or looking for it raises; and the verdict follows the probe alone,
    in both directions, on the run check and the setup check."""
    profile = tmp_path / "claude-work"
    home = tmp_path / "home"
    _poison_credential_file(profile)
    _poison_credential_file(home / ".claude")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))
    touched = _forbid_the_credential_file(monkeypatch)

    requests = (
        _profile_request(subscription_connection, profile),
        _default_request(subscription_connection),
    )
    for answer, expected in ((dict(_LOGGED_IN), True), ({"loggedIn": False}, False)):
        adapter, _ = _preflight_adapter(monkeypatch, auth_status=lambda b, e, a=answer: a)
        for request in requests:
            for check in (adapter.preflight, adapter.identity_preflight):
                receipt = check(request)
                assert receipt.ok is expected, receipt.problem
                assert _POISON not in repr(receipt)

    assert touched == []


def test_no_module_names_the_credential_file_in_code():
    """The same guarantee, read off the source: the file's name appears in no
    string in the package, so nothing can build a path to it. Comments may say
    what is not done."""
    import tokenize

    import modelpass

    offenders = []
    for path in Path(modelpass.__file__).parent.rglob("*.py"):
        with tokenize.open(path) as handle:
            for token in tokenize.generate_tokens(handle.readline):
                if token.type == tokenize.STRING and "credentials.json" in token.string:
                    offenders.append(f"{path.name}:{token.start[0]}")
    assert offenders == []


def test_nothing_the_probe_says_beyond_the_allow_list_reaches_a_receipt(
    monkeypatch, subscription_connection
):
    """Was: the credential status type has no field that can hold a token."""
    answer = {**_LOGGED_IN, "oauthToken": _POISON, "accessToken": _POISON}
    adapter, _ = _preflight_adapter(monkeypatch, auth_status=lambda b, e: answer)
    for check in (adapter.preflight, adapter.identity_preflight):
        receipt = check(_default_request(subscription_connection))
        assert receipt.ok
        assert _POISON not in repr(receipt)


# --- a lapsed login is found by the run, and the run says what to do ------------
#
# Evidence (2026-09-30, Windows, Claude Code 2.1.x): with the access token past
# its expiry, repeated ``claude auth status`` runs reported ``loggedIn: true``; a
# one-shot ``claude -p`` model call refreshed it at once. So a lapsed access
# token never did refuse a run, and it cannot now: preflight does not see it.


def _unauthorized(text="Failed to authenticate. API Error: 401 OAuth token has expired."):
    return ResultMessage(
        subtype="success", is_error=True, api_error_status=401, result=text
    )


def test_a_failed_refresh_during_the_run_ends_it_as_an_error_not_a_success():
    """The 401 shape observed live maps to ERROR even though its subtype says
    'success'."""
    terminal = map_message(_unauthorized())[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert "401" in terminal.reason


def test_a_401_on_a_subscription_run_names_the_relogin_command():
    """Preflight used to refuse a login whose refresh token had expired, with
    `claude /login` in the message. That case now reaches the run, so the
    instruction has to be on the run's terminal."""
    terminal = map_message(_unauthorized(), auth_mode=AuthMode.SUBSCRIPTION)[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert "HTTP 401" in terminal.reason
    assert "OAuth token has expired" in terminal.reason
    assert "re-run 'claude /login'" in terminal.reason
    assert "CLAUDE_CONFIG_DIR" in terminal.reason


def test_a_long_401_still_ends_with_the_relogin_command():
    terminal = map_message(_unauthorized("x" * 5000), auth_mode=AuthMode.SUBSCRIPTION)[-1]
    assert "re-run 'claude /login'" in terminal.reason
    assert len(terminal.reason) < 600


def test_a_401_on_an_api_key_run_is_not_told_to_log_in():
    terminal = map_message(_unauthorized(), auth_mode=AuthMode.API_KEY)[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert "claude /login" not in terminal.reason


def test_another_error_on_a_subscription_run_is_not_told_to_log_in():
    msg = ResultMessage(
        subtype="success", is_error=True, api_error_status=500, result="Overloaded"
    )
    terminal = map_message(msg, auth_mode=AuthMode.SUBSCRIPTION)[-1]
    assert "claude /login" not in terminal.reason


def test_a_stateless_run_that_meets_a_401_tells_the_caller_to_log_in(
    monkeypatch, subscription_request
):
    """The hint is on the run path itself, not only on the pure mapper."""

    class UnauthorizedClient(FakeClient):
        async def receive_response(self):
            yield _unauthorized()

    class UnauthorizedSdk(RecordingSdk):
        def ClaudeSDKClient(self, options):
            return UnauthorizedClient(options)

    adapter = AnthropicAdapter()
    monkeypatch.setattr(AnthropicAdapter, "_sdk", staticmethod(lambda: UnauthorizedSdk()))
    terminal = list(adapter.run(subscription_request))[-1]
    assert isinstance(terminal, TerminalEvent)
    assert terminal.status is TerminalStatus.ERROR
    assert "re-run 'claude /login'" in terminal.reason


def test_a_session_turn_that_meets_a_401_tells_the_caller_to_log_in(
    monkeypatch, session_request
):
    sdk = SessionSdk(turns=[[_unauthorized()]])
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())
    terminal = list(handle.send("hello"))[-1]
    assert isinstance(terminal, TerminalEvent)
    assert terminal.status is TerminalStatus.ERROR
    assert "re-run 'claude /login'" in terminal.reason
    handle.close()


# --- every platform, one path ---------------------------------------------------
#
# macOS used to be the special case: no file to read, so the probe was the only
# evidence, and a probe that returned nothing was let through with a note. It is
# now the general case, and nothing returned refuses. Not exercised on a Mac:
# sys.platform is patched, which shows only that no branch reads it.


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_the_verdict_is_the_same_on_every_platform(
    monkeypatch, subscription_connection, tmp_path, platform
):
    import sys

    monkeypatch.setattr(sys, "platform", platform)
    root = tmp_path / "claude-work"
    request = _profile_request(subscription_connection, root)

    adapter, _ = _preflight_adapter(
        monkeypatch, auth_status=lambda b, e: {"loggedIn": True, "email": "work@example.com"}
    )
    receipt = adapter.identity_preflight(request)
    assert receipt.ok is True
    assert receipt.account_profile is not None
    assert receipt.account_profile.email == "work@example.com"
    assert "Keychain" not in receipt.credential_source
    assert adapter.preflight(request).ok is True

    adapter, _ = _preflight_adapter(monkeypatch, auth_status=lambda b, e: {"loggedIn": False})
    receipt = adapter.identity_preflight(request)
    assert receipt.ok is False
    assert "claude auth status reports no login" in receipt.problem
    assert str(root) in receipt.problem
    assert adapter.preflight(request).ok is False

    adapter, _ = _preflight_adapter(monkeypatch, auth_status=lambda b, e: None)
    assert adapter.identity_preflight(request).ok is False
    assert adapter.preflight(request).ok is False


# --- the probe itself -----------------------------------------------------------


class _Completed:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def _fake_cli(monkeypatch, tmp_path, outputs):
    """Stand in for the subprocess. ``outputs`` are stdout strings, or an
    exception to raise, one per launch."""
    from modelpass.adapters import anthropic as module

    binary = tmp_path / "claude"
    binary.write_text("", encoding="utf-8")
    launches: list[tuple[list[str], dict[str, str]]] = []
    remaining = list(outputs)

    def run(argv, **kwargs):
        launches.append((list(argv), dict(kwargs["env"])))
        output = remaining.pop(0)
        if isinstance(output, BaseException):
            raise output
        return _Completed(output)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "_AUTH_STATUS_RETRY_DELAYS", (0.001, 0.001))
    return module, str(binary), launches


def test_the_probe_runs_auth_status_in_exactly_the_environment_it_was_given(
    monkeypatch, tmp_path
):
    module, binary, launches = _fake_cli(monkeypatch, tmp_path, ['{"loggedIn": true}'])
    env = {"CLAUDE_CONFIG_DIR": "/work", "PATH": "/bin"}
    assert module._default_auth_status(binary, env) == {"loggedIn": True}
    assert launches == [([binary, "auth", "status"], env)]


def test_a_logged_out_answer_is_an_answer_whatever_the_exit_code(monkeypatch, tmp_path):
    module, binary, launches = _fake_cli(monkeypatch, tmp_path, ['{"loggedIn": false}'])
    assert module._default_auth_status(binary, {}) == {"loggedIn": False}
    assert len(launches) == 1


def test_the_probe_asks_again_when_it_gets_nothing_readable(monkeypatch, tmp_path):
    """One empty answer among concurrent probes refused 97 of 100 calls on
    2026-10-01. A run depends on this answer now, so it is asked for again."""
    module, binary, launches = _fake_cli(
        monkeypatch, tmp_path, ["", "not json", '{"loggedIn": true}']
    )
    assert module._default_auth_status(binary, {}) == {"loggedIn": True}
    assert len(launches) == 3


def test_the_probe_gives_up_after_its_retries(monkeypatch, tmp_path):
    module, binary, launches = _fake_cli(monkeypatch, tmp_path, ["", "{}", "[1]", "never"])
    assert module._default_auth_status(binary, {}) is None
    assert len(launches) == 3


def test_a_cli_that_will_not_start_is_not_retried(monkeypatch, tmp_path):
    module, binary, launches = _fake_cli(
        monkeypatch, tmp_path, [OSError("nope"), '{"loggedIn": true}']
    )
    assert module._default_auth_status(binary, {}) is None
    assert len(launches) == 1


def test_the_probe_launches_nothing_that_is_not_a_file(monkeypatch, tmp_path):
    module, _binary, launches = _fake_cli(monkeypatch, tmp_path, ['{"loggedIn": true}'])
    assert module._default_auth_status("claude", {}) is None
    assert launches == []


# --- the environment scrub (the whole point, D1/D2) ----------------------------


def test_scrub_removes_an_ambient_api_key_from_the_process_env(
    monkeypatch, subscription_connection
):
    """The SDK merges options.env over os.environ, so the key must leave os.environ.

    This is the proof that a stray ANTHROPIC_API_KEY cannot reach the runtime
    and silently move a subscription run onto metered billing.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-scrubbed")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    plan = plan_launch(subscription_connection, os.environ)

    with scrubbed_process_env(plan) as removed:
        assert "ANTHROPIC_API_KEY" not in os.environ
        assert "CLAUDE_CODE_USE_BEDROCK" not in os.environ
        assert "PATH" in os.environ  # the child still needs a usable environment
        assert "ANTHROPIC_API_KEY" in removed

    # ...and the caller's own environment is put back exactly as it was.
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-should-be-scrubbed"


def test_scrub_restores_the_environment_even_when_the_body_raises(
    monkeypatch, subscription_connection
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    plan = plan_launch(subscription_connection, os.environ)
    with pytest.raises(RuntimeError):
        with scrubbed_process_env(plan):
            assert "ANTHROPIC_API_KEY" not in os.environ
            raise RuntimeError("boom")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-x"


def test_scrub_keeps_a_credential_the_connection_named(monkeypatch, api_connection):
    """D2: a referenced credential survives; everything else in the rule does not."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-referenced")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unreferenced")

    plan = plan_launch(api_connection, os.environ)
    with scrubbed_process_env(plan):
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-referenced"
        assert "ANTHROPIC_AUTH_TOKEN" not in os.environ


def test_scrub_catches_a_variable_the_plan_never_saw(monkeypatch, subscription_connection):
    """The plan may be built from an injected env view; the guarantee is about
    the *live* process environment, so the scrub recomputes against it."""
    plan = plan_launch(subscription_connection, {})  # plan sees nothing
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-appeared-later")
    assert plan.scrubbed == ()

    with scrubbed_process_env(plan):
        assert "ANTHROPIC_API_KEY" not in os.environ


# --- the allowance question the receipt cannot answer ---------------------------


def test_the_receipt_says_that_remaining_allowance_is_unknowable_before_a_run(
    monkeypatch, subscription_connection
):
    """The honest form of "credit/allowance presence in the receipt".

    There is no pre-run credit query on the Agent SDK, so the receipt states the
    absence and names where the answer does arrive instead. A receipt that was
    simply silent about it would leave a user assuming it had been checked.
    """
    plan = plan_launch(subscription_connection, {})
    request = RunRequest(connection=subscription_connection, messages=(), plan=plan)
    receipt = AnthropicAdapter()._subscription_receipt(
        request,
        [],
        login={"loggedIn": True, "subscriptionType": "Max 5x", "planTier": "max_5x"},
    )

    joined = " ".join(receipt.notes)
    assert "not knowable before a run" in joined
    assert "rate_limit" in joined
    assert "plan tier: max_5x" in joined
    assert receipt.plan_name == "Max 5x"


# --- forbidden launch arguments ------------------------------------------------


def test_preflight_does_not_trip_over_its_own_forbidden_list(subscription_connection):
    """Regression: the forbidden *list* is not the args being launched.

    ``plan.forbidden_args`` carries ('--bare',) for transparency. Passing it to
    ``check_launch_args`` made every single run raise UnsafeLaunch, which the
    live smoke test caught. Only arguments we actually contribute get checked.
    """
    from modelpass.adapters.base import RunRequest

    plan = plan_launch(subscription_connection, {})
    request = RunRequest(connection=subscription_connection, messages=(), plan=plan)
    receipt = AnthropicAdapter().preflight(request)
    assert receipt is not None  # did not raise


def test_bare_cannot_be_smuggled_through_extra_args(subscription_connection):
    """--bare never reads OAuth credentials, so it must never reach the runtime."""
    from modelpass.adapters.base import RunRequest
    from modelpass.errors import UnsafeLaunch

    plan = plan_launch(subscription_connection, {})
    request = RunRequest(
        connection=subscription_connection,
        messages=(),
        plan=plan,
        options={"extra_args": {"bare": None}},
    )
    with pytest.raises(UnsafeLaunch, match="--bare"):
        AnthropicAdapter().preflight(request)


# --- availability --------------------------------------------------------------


def test_is_available_does_not_import_the_vendor_package(monkeypatch):
    monkeypatch.delitem(os.sys.modules, "claude_agent_sdk", raising=False)
    AnthropicAdapter.is_available()
    # find_spec locates without executing the module.
    assert "claude_agent_sdk" not in os.sys.modules


# --- sessions (D14-D17) --------------------------------------------------------
#
# Same rules as the rest of this file: no network, no credential, no vendor
# package. The session surface of claude_agent_sdk is stood in for by classes
# shaped like the real ones -- a client that connects once and answers queries,
# and the three module-level functions the listing and history paths call.


class SessionClient:
    """A ClaudeSDKClient that stays connected and answers one turn per query."""

    def __init__(self, options, sdk):
        self.options = options
        self.sdk = sdk
        self.connects = 0
        self.connect_prompt = "unset"
        self.queries = []
        self.disconnects = 0
        self.interrupts = 0

    async def connect(self, prompt=None):
        self.connects += 1
        self.connect_prompt = prompt

    async def query(self, prompt, session_id="default"):
        self.queries.append(prompt)

    async def receive_response(self):
        for message in self.sdk.next_turn():
            yield message

    async def disconnect(self):
        self.disconnects += 1

    async def interrupt(self):
        self.interrupts += 1


class SessionSdk(FakeSdk):
    """claude_agent_sdk's session surface, shaped from the installed 0.2.139."""

    client_cls = SessionClient

    def __init__(self, *, turns=None, listing=(), info=None, messages=()):
        super().__init__()
        self.options_kwargs = None
        self.clients = []
        self.turns = list(turns or [])
        self.listing = list(listing)
        self.info = info
        self.messages = list(messages)
        self.list_calls = []
        self.info_calls = []
        self.message_calls = []

    # options and client
    def ClaudeAgentOptions(self, **kwargs):
        self.options_kwargs = kwargs
        return kwargs

    def ClaudeSDKClient(self, options):
        client = self.client_cls(options, self)
        self.clients.append(client)
        return client

    def next_turn(self):
        if self.turns:
            return self.turns.pop(0)
        return [ResultMessage(session_id="sess-1")]

    # transcript reads
    def list_sessions(self, directory=None, include_worktrees=True):
        self.list_calls.append((directory, include_worktrees))
        return self.listing

    def get_session_info(self, session_id, directory=None):
        self.info_calls.append((session_id, directory))
        return self.info

    def get_session_messages(self, session_id, directory=None):
        self.message_calls.append((session_id, directory))
        return self.messages


@dataclass
class SDKSessionInfo:
    """claude_agent_sdk.SDKSessionInfo, field for field."""

    session_id: str
    summary: str = ""
    last_modified: int = 0
    file_size: int | None = None
    custom_title: str | None = None
    first_prompt: str | None = None
    git_branch: str | None = None
    cwd: str | None = None
    tag: str | None = None
    created_at: int | None = None


@dataclass
class SessionMessage:
    """claude_agent_sdk.SessionMessage, field for field."""

    type: str
    message: Any
    uuid: str = "u"
    session_id: str = "sess-1"
    parent_tool_use_id: None = None


@pytest.fixture
def session_request(subscription_connection, tmp_path):
    """Build a SessionRequest the way the bridge would, minus the gating."""

    def make(**kwargs) -> SessionRequest:
        kwargs.setdefault("kind", SessionKind.CHAT)
        kwargs.setdefault("project_folder", str(tmp_path / "work"))
        return SessionRequest(
            connection=subscription_connection,
            plan=plan_launch(subscription_connection, {}),
            **kwargs,
        )

    return make


def session_adapter(monkeypatch, sdk, *, version="2.1.233 (Claude Code)"):
    """An adapter wired to a fake SDK and a fake CLI. Spawns nothing.

    ``is_available`` is faked along with the SDK it answers about: the vendor
    package is standing right here, so a machine without the ``anthropic``
    extra -- which is every machine CI runs on (D2) -- must exercise the same
    code as one with it, rather than skip.
    """
    adapter = AnthropicAdapter(
        claude_version=lambda binary, env: version,
        auth_status=lambda binary, env: {"loggedIn": True, "subscriptionType": "pro"},
    )
    monkeypatch.setattr(AnthropicAdapter, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(AnthropicAdapter, "_sdk", staticmethod(lambda: sdk))
    monkeypatch.setattr(
        AnthropicAdapter,
        "_cli_status",
        staticmethod(lambda _sdk: ("/usr/bin/claude", None)),
    )
    return adapter


def session_launch_options(monkeypatch, request, **kwargs) -> dict:
    """What a session would hand ClaudeAgentOptions, without running a turn."""
    adapter = session_adapter(monkeypatch, SessionSdk(), **kwargs)
    return adapter.open_session(request)._options_kwargs


# --- the option mapping ---------------------------------------------------------


def test_a_session_runs_in_its_project_folder(monkeypatch, session_request, tmp_path):
    """The isolation the core's scratch-directory default depends on.

    Without cwd the runtime inherits the caller's process directory and writes
    modelpass transcripts into their own ~/.claude/projects/<their-cwd>/, where
    their own claude --continue would find them.
    """
    request = session_request(project_folder=str(tmp_path / "elsewhere"))
    options = session_launch_options(monkeypatch, request)
    assert options["cwd"] == str(tmp_path / "elsewhere")


def test_a_chat_session_replaces_the_persona_and_keeps_the_toolbelt_off(
    monkeypatch, session_request
):
    options = session_launch_options(
        monkeypatch, session_request(system_prompt="You are a scorer.")
    )
    assert options["system_prompt"] == "You are a scorer."
    assert options["tools"] == []
    assert "Bash" in options["disallowed_tools"]
    assert options["max_turns"] == 1
    assert options["setting_sources"] == []


def test_a_chat_session_loads_no_account_connectors(monkeypatch, session_request):
    """A subscription login adds the account's claude.ai connectors as MCP servers,
    which the built-in disallow list does not reach (live 2026-10-21: a chat called
    the Claude Docs connector and ended in error_max_turns). A chat loads only
    the servers modelpass passes, and passes none.
    """
    options = session_launch_options(monkeypatch, session_request())
    assert options["strict_mcp_config"] is True
    assert options["mcp_servers"] == {}


#: A session reply schema, already normalized the way the bridge hands it over.
SESSION_SCHEMA = {
    "type": "object", "properties": {"choice": {"type": "integer"}}, "required": ["choice"]
}


def test_a_session_schema_becomes_the_clients_output_format(monkeypatch, session_request):
    """The stateless mechanism (D13), held for the client's life, with the chat's one-turn ceiling
    raised for the runtime's re-ask after a failed validation."""
    options = session_launch_options(
        monkeypatch, session_request(schema=SESSION_SCHEMA, schema_name="decision")
    )
    assert options["output_format"] == {"type": "json_schema", "schema": SESSION_SCHEMA}
    assert options["max_turns"] == STRUCTURED_OUTPUT_MAX_TURNS


def test_a_session_without_a_schema_has_no_output_format(monkeypatch, session_request):
    options = session_launch_options(monkeypatch, session_request())
    assert "output_format" not in options and options["max_turns"] == 1


def test_a_callers_turn_ceiling_still_wins_over_the_schemas(monkeypatch, session_request):
    options = session_launch_options(
        monkeypatch, session_request(schema=SESSION_SCHEMA, options={"max_turns": 5})
    )
    assert options["max_turns"] == 5


def test_every_turn_of_a_schema_session_ends_in_its_structured_answer(
    monkeypatch, session_request
):
    sdk = SessionSdk(turns=[
        [ResultMessage(session_id="sess-1", result='{"choice": 2}',
                       structured_output={"choice": 2})],
        [ResultMessage(session_id="sess-1", result='{"choice": 5}',
                       structured_output={"choice": 5})],
    ])
    handle = session_adapter(monkeypatch, sdk).open_session(
        session_request(schema=SESSION_SCHEMA, schema_name="decision")
    )

    answers = []
    for message in ("first", "second"):
        events = list(handle.send(message))
        structured = [e for e in events if isinstance(e, StructuredOutputEvent)]
        assert len(structured) == 1 and events.index(structured[0]) == len(events) - 2
        assert structured[0].valid and structured[0].schema_name == "decision"
        assert events[-1].status is TerminalStatus.OK
        answers.append(structured[0].data)

    assert answers == [{"choice": 2}, {"choice": 5}]
    handle.close()


def test_a_schema_turn_with_no_usable_answer_ends_in_error(monkeypatch, session_request):
    """As on a stateless run: prose where JSON was promised is not a success."""
    sdk = SessionSdk(turns=[[ResultMessage(session_id="sess-1", result="I would rather not.")]])
    handle = session_adapter(monkeypatch, sdk).open_session(session_request(schema=SESSION_SCHEMA))

    events = list(handle.send("first"))

    assert events[-1].status is TerminalStatus.ERROR
    handle.close()


def test_a_worker_keeps_the_runtimes_mcp_configuration(monkeypatch, session_request):
    """A worker is the runtime with its toolbelt on; its connectors are its owner's to configure."""
    options = session_launch_options(monkeypatch, session_request(kind=SessionKind.WORKER))
    assert "strict_mcp_config" not in options and "mcp_servers" not in options


def test_a_chat_without_a_prompt_gets_the_empty_replacement_not_the_preset(
    monkeypatch, session_request
):
    """A chat asked for the runtime's persona to be gone; the preset says the opposite."""
    assert session_launch_options(monkeypatch, session_request())["system_prompt"] == ""


def test_a_worker_appends_to_the_preset_and_keeps_the_toolbelt_on(
    monkeypatch, session_request
):
    request = session_request(kind=SessionKind.WORKER, system_prompt="Ship the fix.")
    options = session_launch_options(monkeypatch, request)
    assert options["system_prompt"] == {
        "type": "preset",
        "preset": "claude_code",
        "exclude_dynamic_sections": True,
        "append": "Ship the fix.",
    }
    assert options["tools"] == {"type": "preset", "preset": "claude_code"}
    assert options["disallowed_tools"] == []
    # A toolbelt needs somewhere to go; the ceiling is the guard system's, in tokens.
    assert options["max_turns"] is None


def test_a_worker_with_no_prompt_still_maps_to_the_preset_never_to_omission(
    monkeypatch, session_request
):
    """Omitting system_prompt gives the *minimal* prompt: tools with no guidance."""
    options = session_launch_options(
        monkeypatch, session_request(kind=SessionKind.WORKER)
    )
    assert options["system_prompt"]["preset"] == "claude_code"
    assert "append" not in options["system_prompt"]


def test_only_the_preset_form_carries_exclude_dynamic_sections(
    monkeypatch, session_request
):
    """It has no effect on a string, and a caller's string has no dynamic sections."""
    chat = session_launch_options(monkeypatch, session_request(system_prompt="x"))
    assert chat["system_prompt"] == "x"
    worker = session_launch_options(
        monkeypatch, session_request(kind=SessionKind.WORKER, system_prompt="x")
    )
    assert worker["system_prompt"]["exclude_dynamic_sections"] is True


def test_a_large_system_prompt_travels_as_a_file_not_an_argument(
    monkeypatch, session_request
):
    """Over the OS argument limit a string prompt fails at *spawn*, before any request."""
    prompt = "R" * 20_000
    adapter = session_adapter(monkeypatch, SessionSdk())
    handle = adapter.open_session(session_request(system_prompt=prompt))

    option = handle._options_kwargs["system_prompt"]
    assert option["type"] == "file"
    path = Path(option["path"])
    assert path.read_text(encoding="utf-8") == prompt

    handle.close()
    assert not path.exists()  # the file lives exactly as long as the session


def test_a_small_system_prompt_stays_on_the_long_standing_flag(
    monkeypatch, session_request
):
    options = session_launch_options(monkeypatch, session_request(system_prompt="short"))
    assert options["system_prompt"] == "short"


def test_caller_tools_reach_a_session_the_way_they_reach_a_run(
    monkeypatch, session_request
):
    request = session_request(
        tools=(ToolDef(name="look_up", description="d", handler=lambda a: "x"),)
    )
    options = session_launch_options(monkeypatch, request)
    assert options["allowed_tools"] == ["mcp__caller__look_up"]
    assert options["mcp_servers"]["caller"]["name"] == "caller"
    assert options["strict_mcp_config"] is True
    assert options["max_turns"] is None


# --- the environment a session launches with -----------------------------------


def test_an_ephemeral_session_writes_no_transcript(monkeypatch, session_request):
    """persist=False is implemented by this variable, not by flattening turns."""
    options = session_launch_options(monkeypatch, session_request(persist=False))
    assert options["env"]["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] == "1"


def test_a_session_passes_the_connections_claude_config_dir_to_the_sdk(
    monkeypatch, session_request, tmp_path
):
    request = session_request()
    root = tmp_path / "claude-work"
    connection = replace(request.connection, config_dir=str(root))
    request = replace(
        request,
        connection=connection,
        plan=plan_launch(connection, {"CLAUDE_CONFIG_DIR": str(tmp_path / "wrong")}),
    )
    options = session_launch_options(monkeypatch, request)
    assert options["env"]["CLAUDE_CONFIG_DIR"] == str(root)


def test_a_persisted_session_does_not_inherit_the_run_paths_skip_history(
    monkeypatch, session_request
):
    """run() sets it unconditionally; a session that did would not be resumable."""
    options = session_launch_options(monkeypatch, session_request(persist=True))
    assert "CLAUDE_CODE_SKIP_PROMPT_HISTORY" not in options["env"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2.1.233 (Claude Code)", (2, 1, 233)),
        ("2.1.242", (2, 1, 242)),
        ("2.2.0-beta.1 (Claude Code)", (2, 2, 0)),
        ("", None),
        (None, None),
        ("unknown", None),
    ],
)
def test_cli_version_parsing(text, expected):
    assert parse_cli_version(text) == expected


@pytest.mark.parametrize(
    "version,supported",
    [
        ("2.1.242 (Claude Code)", True),
        ("2.2.0 (Claude Code)", True),
        ("2.1.236 (Claude Code)", False),
        ("2.1.233 (Claude Code)", False),
        (None, False),
        ("not a version", False),
    ],
)
def test_the_ttl_lever_is_believed_only_where_the_cli_reads_it(version, supported):
    """Unknown means no: a variable an older build ignores must not be reported as set."""
    assert supports_prompt_cache_ttl(version) is supported


def test_a_new_enough_cli_gets_the_hour_pinned(monkeypatch, session_request):
    options = session_launch_options(
        monkeypatch, session_request(), version="2.1.242 (Claude Code)"
    )
    assert options["env"]["CLAUDE_CODE_PROMPT_CACHE_TTL"] == "1h"


def test_an_older_cli_is_left_on_the_runtime_default(monkeypatch, session_request):
    """Setting an ignored variable and reporting the TTL as pinned is the silent
    failure this library exists to refuse."""
    options = session_launch_options(
        monkeypatch, session_request(), version="2.1.236 (Claude Code)"
    )
    assert "CLAUDE_CODE_PROMPT_CACHE_TTL" not in options["env"]


def test_the_version_is_probed_once_per_process_not_once_per_preflight():
    calls = []

    def version(binary, env):
        calls.append(binary)
        return "2.1.242 (Claude Code)"

    adapter = AnthropicAdapter(claude_version=version)
    assert adapter._cli_version("/usr/bin/claude", {}) == "2.1.242 (Claude Code)"
    assert adapter._cli_version("/usr/bin/claude", {}) == "2.1.242 (Claude Code)"
    assert calls == ["/usr/bin/claude"]


def test_the_identity_probe_is_cached_per_binary_and_config_dir():
    """~2.4s of subprocess that every run's preflight wants; ask once per window.

    Keyed by the config directory as well as the binary, because two profiles on
    one binary are two different accounts and sharing an entry would report the
    wrong one.
    """
    calls = []

    def status(binary, env):
        calls.append((binary, env.get("CLAUDE_CONFIG_DIR")))
        return {"loggedIn": True}

    adapter = AnthropicAdapter(auth_status=status)
    for _ in range(3):
        adapter._cached_auth_status("claude", {"CLAUDE_CONFIG_DIR": "/home"})
    assert calls == [("claude", "/home")]

    adapter._cached_auth_status("claude", {"CLAUDE_CONFIG_DIR": "/work"})
    adapter._cached_auth_status("claude", {})
    assert calls == [("claude", "/home"), ("claude", "/work"), ("claude", None)]


def test_invalidate_identity_cache_forces_the_next_probe():
    calls = []

    def status(binary, env):
        calls.append(binary)
        return {"loggedIn": True}

    adapter = AnthropicAdapter(auth_status=status)
    adapter._cached_auth_status("claude", {})
    adapter._cached_auth_status("claude", {})
    assert len(calls) == 1
    adapter.invalidate_identity_cache()
    adapter._cached_auth_status("claude", {})
    assert len(calls) == 2


def test_the_identity_probe_expires_rather_than_being_held_for_the_process():
    """A user can 'claude /login' in another terminal; the cache must not outlive that."""
    from modelpass.adapters import anthropic as module

    assert module._IDENTITY_CACHE_TTL_SECONDS <= 300


def test_an_api_key_connection_never_runs_the_identity_probe(
    monkeypatch, api_connection
):
    """It is a subscription question. Paying 2.4s to ask it on a metered run is waste."""
    calls = []

    def status(binary, env):
        calls.append(binary)
        return {"loggedIn": True}

    adapter = AnthropicAdapter(auth_status=status)
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "_sdk", lambda: FakeSdk())
    monkeypatch.setattr(adapter, "_cli_status", lambda sdk: ("claude", None))
    monkeypatch.setattr(adapter, "_cli_version", lambda binary, env: "2.1.242")

    env = {"ANTHROPIC_API_KEY": "sk-not-a-real-key"}
    adapter.preflight(
        RunRequest(
            connection=api_connection,
            messages=(Message(Role.USER, "hi"),),
            plan=plan_launch(api_connection, env),
        )
    )
    assert calls == []


# --- what a caller may and may not reach around --------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "cwd",
        "system_prompt",
        "resume",
        "continue_conversation",
        "session_id",
        "fork_session",
    ],
)
def test_options_a_session_owns_are_refused_rather_than_silently_applied(
    monkeypatch, session_request, key
):
    """Each of these decides *which conversation this object is*."""
    adapter = session_adapter(monkeypatch, SessionSdk())
    with pytest.raises(InvalidSession, match=key):
        adapter.open_session(session_request(options={key: "anything"}))


def test_a_callers_env_is_merged_underneath_the_promises_modelpass_makes(
    monkeypatch, session_request
):
    request = session_request(
        persist=False,
        options={
            "env": {
                "DISABLE_AUTO_COMPACT": "1",
                "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "0",
            }
        },
    )
    options = session_launch_options(monkeypatch, request)
    assert options["env"]["DISABLE_AUTO_COMPACT"] == "1"
    # The ephemeral promise is not something a caller cancels by accident.
    assert options["env"]["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] == "1"


def test_a_caller_can_still_set_the_permission_mode_a_worker_needs(
    monkeypatch, session_request
):
    """modelpass does not widen a worker's permissions on the caller's behalf."""
    default = session_launch_options(monkeypatch, session_request(kind=SessionKind.WORKER))
    assert default["permission_mode"] == "default"
    widened = session_launch_options(
        monkeypatch,
        session_request(
            kind=SessionKind.WORKER, options={"permission_mode": "acceptEdits"}
        ),
    )
    assert widened["permission_mode"] == "acceptEdits"


# --- opening, turns, and the id ------------------------------------------------


def test_opening_a_session_spends_nothing(monkeypatch, session_request):
    """Adapter rule 7: local state and nothing else. No client, no id."""
    sdk = SessionSdk()
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())
    assert sdk.clients == []
    assert handle.id is None


def test_the_first_send_creates_the_session_and_the_id_comes_from_the_runtime(
    monkeypatch, session_request
):
    sdk = SessionSdk(turns=[[ResultMessage(session_id="sess-abc")]])
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())

    events = list(handle.send("hello"))

    assert handle.id == "sess-abc"
    assert sdk.clients[0].queries == ["hello"]
    assert isinstance(events[-1], TerminalEvent)
    handle.close()


def test_one_client_carries_every_turn(monkeypatch, session_request):
    """The whole of D14 on this runtime: the conversation lives in the subprocess."""
    sdk = SessionSdk(
        turns=[[ResultMessage(session_id="sess-1")], [ResultMessage(session_id="sess-1")]]
    )
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())

    list(handle.send("first"))
    list(handle.send("second"))

    assert len(sdk.clients) == 1
    assert sdk.clients[0].connects == 1
    assert sdk.clients[0].connect_prompt is None  # streaming mode, no initial prompt
    assert sdk.clients[0].queries == ["first", "second"]
    handle.close()


def test_an_ephemeral_session_never_hands_out_an_id(monkeypatch, session_request):
    """There is no transcript for that id to name, on any listing or resume."""
    sdk = SessionSdk(turns=[[ResultMessage(session_id="sess-ephemeral")]])
    handle = session_adapter(monkeypatch, sdk).open_session(session_request(persist=False))
    list(handle.send("hello"))
    assert handle.id is None
    handle.close()


def test_a_turn_streams_the_same_normalized_events_a_run_does(
    monkeypatch, session_request
):
    sdk = SessionSdk(
        turns=[
            [
                # Text arrives as stream deltas, exactly as it does in a run with
                # include_partial_messages on -- which a session also defaults to.
                StreamEvent(
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "hi"},
                    }
                ),
                AssistantMessage([TextBlock("hi")]),
                ResultMessage(
                    session_id="s", usage={"input_tokens": 5, "output_tokens": 2}
                ),
            ]
        ]
    )
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())

    events = list(handle.send("hello"))

    assert any(isinstance(e, TextDeltaEvent) and e.text == "hi" for e in events)
    assert any(
        isinstance(e, UsageEvent) and e.scope is UsageScope.RUN_TOTAL for e in events
    )
    assert isinstance(events[-1], TerminalEvent)
    handle.close()


def test_a_session_fails_closed_on_an_api_key(monkeypatch, session_request):
    """The guarantee does not weaken because the call happens to be a session."""
    sdk = SessionSdk(turns=[[SystemMessage("init", {"apiKeySource": "ANTHROPIC_API_KEY"})]])
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())
    with pytest.raises(AuthModeMismatch):
        list(handle.send("hello"))
    handle.close()


class VendorException(Exception):
    """An exception whose defining module is the vendor package."""


VendorException.__module__ = "claude_agent_sdk"


class FailingSessionClient(SessionClient):
    async def receive_response(self):
        raise VendorException("the CLI exited with code 1")
        yield  # pragma: no cover - makes this an async generator


def test_a_vendor_failure_in_a_turn_is_reported_not_raised_raw(
    monkeypatch, session_request
):
    sdk = SessionSdk()
    sdk.client_cls = FailingSessionClient
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())
    with pytest.raises(VendorRunFailed, match="the CLI exited"):
        list(handle.send("hello"))
    handle.close()


def test_two_turns_at_once_are_refused_rather_than_interleaved(
    monkeypatch, session_request
):
    """Two query() calls into one client is two callers disagreeing about the state."""
    sdk = SessionSdk(turns=[[ResultMessage()], [ResultMessage()]])
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())

    first = handle.send("one")
    next(first)  # start it, do not finish it
    with pytest.raises(InvalidSession, match="already streaming"):
        handle.send("two")
    first.close()
    handle.close()


class BlockingSessionClient(SessionClient):
    """Streams one message, then waits for an interrupt before finishing."""

    def __init__(self, options, sdk):
        super().__init__(options, sdk)
        self.gate = asyncio.Event()
        self.honour_interrupt = True

    async def receive_response(self):
        yield StreamEvent(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "part"}}
        )
        await self.gate.wait()
        yield ResultMessage(session_id="sess-1", terminal_reason="aborted_streaming")

    async def interrupt(self):
        self.interrupts += 1
        if self.honour_interrupt:
            self.gate.set()


def test_abandoning_a_turn_interrupts_it_and_leaves_the_session_usable(
    monkeypatch, session_request
):
    """One abandoned turn is not the end of a conversation."""
    sdk = SessionSdk()
    sdk.client_cls = BlockingSessionClient
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())

    stream = handle.send("one")
    next(stream)
    stream.close()

    client = sdk.clients[0]
    assert client.interrupts == 1
    assert handle._broken is None
    list(handle.send("two"))
    assert client.queries == ["one", "two"]
    handle.close()


def test_a_turn_that_will_not_stop_makes_the_session_say_so(
    monkeypatch, session_request
):
    """A queue nobody drains and a producer still writing is how the next turn hangs."""
    from modelpass.adapters import anthropic as module

    monkeypatch.setattr(module, "_ABANDON_DRAIN_SECONDS", 0.2)
    sdk = SessionSdk()
    sdk.client_cls = BlockingSessionClient
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())

    stream = handle.send("one")
    next(stream)
    sdk.clients[0].honour_interrupt = False
    stream.close()

    with pytest.raises(VendorRunFailed, match="did not stop after an interrupt"):
        handle.send("two")
    handle.close()


def test_closing_disconnects_the_client_and_is_idempotent(monkeypatch, session_request):
    sdk = SessionSdk()
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())
    list(handle.send("hello"))

    handle.close()
    handle.close()

    assert sdk.clients[0].disconnects == 1


def test_a_closed_session_refuses_another_turn(monkeypatch, session_request):
    handle = session_adapter(monkeypatch, SessionSdk()).open_session(session_request())
    handle.close()
    with pytest.raises(VendorRunFailed, match="closed"):
        handle.send("hello")


# --- a turn cancelled while connecting, and a turn that fails midway (2026-11-13) --


class HangingConnectClient(SessionClient):
    """A client whose first connect never returns: a hung initialize handshake."""

    hang = True

    def __init__(self, options, sdk):
        super().__init__(options, sdk)
        self.connect_cancelled = False

    async def connect(self, prompt=None):
        self.connects += 1
        if type(self).hang:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.connect_cancelled = True
                raise


def _drain_in_a_thread(stream, seconds=5.0):
    """Consume ``stream`` off-thread, so a turn that never ends fails the test."""
    import threading

    out: dict[str, Any] = {"events": []}

    def consume():
        try:
            for event in stream:
                out["events"].append(event)
        except BaseException as exc:
            out["error"] = exc

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    return thread, out, seconds


def test_a_turn_can_be_cancelled_while_the_client_is_still_connecting(
    monkeypatch, session_request
):
    """The turn's cancel was a no-op until ``connect()`` had returned.

    A bound fires once. On a hung spawn or handshake it reached a handle with
    no client yet, did nothing, and the consumer stayed blocked for as long as
    the runtime did -- then the turn ran anyway, with nobody reading it.
    """

    class Client(HangingConnectClient):
        hang = True

    sdk = SessionSdk()
    sdk.client_cls = Client
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())

    stream = handle.send("one")
    thread, out, seconds = _drain_in_a_thread(stream)
    deadline = time.monotonic() + 2
    while not sdk.clients and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    stream.cancel()  # what the timeout watchdog calls, from its own thread
    thread.join(seconds)
    assert not thread.is_alive(), "the cancelled turn never ended"

    assert "error" not in out
    assert [e.status for e in out["events"]] == [TerminalStatus.CANCELLED]
    client = sdk.clients[0]
    assert client.connect_cancelled is True
    assert client.queries == [], "nothing was sent to a runtime that never came up"

    # Nothing ran, so nothing is out of step: the next turn connects afresh.
    assert handle._broken is None
    Client.hang = False
    events = list(handle.send("two"))
    assert events[-1].status is TerminalStatus.OK
    assert len(sdk.clients) == 2
    assert sdk.clients[1].queries == ["two"]
    handle.close()


class FailsMidTurnClient(SessionClient):
    """Streams part of an answer, then the stream raises: the turn is still running."""

    async def receive_response(self):
        yield StreamEvent(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "part"}}
        )
        raise VendorException("stream decode failed")


def test_a_turn_that_fails_before_its_result_stops_the_runtime_and_ends_the_session(
    monkeypatch, session_request
):
    """The vendor turn was left running and the next send read its leftovers."""
    sdk = SessionSdk()
    sdk.client_cls = FailsMidTurnClient
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())

    with pytest.raises(VendorRunFailed, match="stream decode failed"):
        list(handle.send("one"))

    client = sdk.clients[0]
    assert client.interrupts == 1
    assert client.disconnects == 1
    with pytest.raises(VendorRunFailed, match="failed before the runtime reported"):
        handle.send("two")
    assert client.queries == ["one"]
    handle.close()
    assert client.disconnects == 1, "close() does not disconnect a second time"


def test_an_auth_mode_mismatch_stops_the_turn_it_was_found_in(
    monkeypatch, session_request
):
    """The refusal raised, and the metered turn it refused kept running."""
    sdk = SessionSdk(
        turns=[[SystemMessage("init", {"apiKeySource": "ANTHROPIC_API_KEY"}), ResultMessage()]]
    )
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())

    with pytest.raises(AuthModeMismatch):
        list(handle.send("one"))

    client = sdk.clients[0]
    assert client.interrupts == 1
    assert client.disconnects == 1
    with pytest.raises(VendorRunFailed, match="AuthModeMismatch"):
        handle.send("two")
    handle.close()


class RaisesAfterResultClient(SessionClient):
    """The SDK's habit: an error ``ResultMessage``, then an exception about it."""

    async def receive_response(self):
        if len(self.queries) == 1:
            yield ResultMessage(
                session_id="sess-1", is_error=True, subtype="error_during_execution"
            )
            raise VendorException("the CLI reported an error result")
        yield ResultMessage(session_id="sess-1")


def test_a_failure_after_the_result_leaves_the_session_usable(monkeypatch, session_request):
    """The turn is over on the vendor's side, so there is nothing to stop."""
    sdk = SessionSdk()
    sdk.client_cls = RaisesAfterResultClient
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())

    with contextlib.suppress(VendorRunFailed):
        list(handle.send("one"))

    client = sdk.clients[0]
    assert handle._broken is None
    assert client.interrupts == 0 and client.disconnects == 0
    assert list(handle.send("two"))[-1].status is TerminalStatus.OK
    handle.close()


class FailsToConnectClient(SessionClient):
    async def connect(self, prompt=None):
        self.connects += 1
        raise VendorException("Failed to start Claude Code")


def test_a_failed_connect_is_reported_and_the_next_turn_tries_again(
    monkeypatch, session_request
):
    sdk = SessionSdk()
    sdk.client_cls = FailsToConnectClient
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())
    with pytest.raises(VendorRunFailed, match="Failed to start"):
        list(handle.send("one"))
    assert handle._broken is None
    sdk.client_cls = SessionClient
    assert list(handle.send("two"))[-1].status is TerminalStatus.OK
    handle.close()


# --- history -------------------------------------------------------------------


def test_history_is_read_back_from_the_runtimes_own_transcript(
    monkeypatch, session_request, tmp_path
):
    sdk = SessionSdk(
        turns=[[ResultMessage(session_id="sess-1")]],
        messages=[
            SessionMessage("user", {"role": "user", "content": "hello"}),
            SessionMessage(
                "assistant",
                {"role": "assistant", "content": [{"type": "text", "text": "hi back"}]},
            ),
        ],
    )
    folder = str(tmp_path / "work")
    handle = session_adapter(monkeypatch, sdk).open_session(
        session_request(project_folder=folder)
    )
    list(handle.send("hello"))

    history = handle.history()

    assert [(m.role, m.content) for m in history] == [
        (Role.USER, "hello"),
        (Role.ASSISTANT, "hi back"),
    ]
    assert sdk.message_calls == [("sess-1", folder)]
    handle.close()


def test_history_is_empty_before_a_turn_has_created_the_session(
    monkeypatch, session_request
):
    sdk = SessionSdk(messages=[SessionMessage("user", {"role": "user", "content": "x"})])
    handle = session_adapter(monkeypatch, sdk).open_session(session_request())
    assert handle.history() == ()
    assert sdk.message_calls == []


def test_an_ephemeral_session_has_no_readable_history(monkeypatch, session_request):
    """Nothing was written down, and modelpass keeps no shadow copy of the events."""
    sdk = SessionSdk(
        turns=[[ResultMessage(session_id="sess-1")]],
        messages=[SessionMessage("user", {"role": "user", "content": "x"})],
    )
    handle = session_adapter(monkeypatch, sdk).open_session(session_request(persist=False))
    list(handle.send("hello"))
    assert handle.history() == ()
    assert sdk.message_calls == []
    handle.close()


def test_transcript_tool_traffic_is_named_not_dropped_and_empty_turns_are():
    history = session_messages_to_history(
        [
            SessionMessage("user", {"role": "user", "content": [{"type": "tool_result"}]}),
            SessionMessage("assistant", {"role": "assistant", "content": []}),
            SessionMessage("user", {"role": "user", "content": "real"}),
        ]
    )
    assert [(m.role, m.content) for m in history] == [
        (Role.USER, "[tool_result]"),
        (Role.USER, "real"),
    ]


# --- resume --------------------------------------------------------------------


def test_resume_names_the_session_and_reports_its_id_immediately(
    monkeypatch, session_request
):
    sdk = SessionSdk(info=SDKSessionInfo(session_id="sess-old"))
    handle = session_adapter(monkeypatch, sdk).resume_session(
        session_request(resume_id="sess-old")
    )
    assert handle._options_kwargs["resume"] == "sess-old"
    assert handle.id == "sess-old"


def test_resume_refuses_an_id_the_runtime_does_not_have(monkeypatch, session_request):
    """Learned before a turn is paid for, because the question is free to ask."""
    sdk = SessionSdk(info=None, messages=[])
    with pytest.raises(SessionNotFound, match="sess-gone"):
        session_adapter(monkeypatch, sdk).resume_session(
            session_request(resume_id="sess-gone")
        )


def test_a_session_with_no_summary_is_still_a_session(monkeypatch, session_request):
    """get_session_info() answers None to two different questions; only one is a miss."""
    sdk = SessionSdk(
        info=None, messages=[SessionMessage("user", {"role": "user", "content": "x"})]
    )
    handle = session_adapter(monkeypatch, sdk).resume_session(
        session_request(resume_id="sess-quiet")
    )
    assert handle.id == "sess-quiet"


def test_a_resumed_session_is_looked_for_in_its_own_folder(
    monkeypatch, session_request, tmp_path
):
    sdk = SessionSdk(info=SDKSessionInfo(session_id="sess-old"))
    folder = str(tmp_path / "repo")
    session_adapter(monkeypatch, sdk).resume_session(
        session_request(resume_id="sess-old", project_folder=folder)
    )
    assert sdk.info_calls == [("sess-old", folder)]


# --- listing -------------------------------------------------------------------


def test_listing_is_scoped_to_the_folder_that_is_the_storage_key(
    monkeypatch, session_request, tmp_path
):
    sdk = SessionSdk(listing=[SDKSessionInfo(session_id="a")])
    folder = str(tmp_path / "repo")

    session_adapter(monkeypatch, sdk).list_sessions(
        session_request(project_folder=folder)
    )

    # Not the SDK default: a sibling worktree's sessions ran against other files.
    assert sdk.list_calls == [(folder, False)]


def test_a_listed_session_is_normalized_with_the_rest_in_vendor(
    monkeypatch, session_request, tmp_path
):
    sdk = SessionSdk(
        listing=[
            SDKSessionInfo(
                session_id="sess-1",
                summary="Scoring run",
                custom_title="My scorer",
                last_modified=1_800_000_000_000,
                created_at=1_700_000_000_000,
                cwd=str(tmp_path / "actual"),
                git_branch="main",
                first_prompt="score this",
                file_size=4096,
                tag="nightly",
            )
        ]
    )

    (info,) = session_adapter(monkeypatch, sdk).list_sessions(
        session_request(project_folder=str(tmp_path))
    )

    assert info.id == "sess-1"
    assert info.connection == "claude-sub"
    assert info.runtime is Runtime.ANTHROPIC_SDK
    # The session's own record of where it ran outranks the folder we asked about.
    assert info.project_folder == str(tmp_path / "actual")
    assert info.title == "My scorer"
    assert info.updated_at.startswith("2027-")  # ISO 8601, not epoch milliseconds
    assert info.created_at.startswith("2023-")
    assert info.message_count is None  # not reported rather than guessed
    assert info.vendor["git_branch"] == "main"
    assert info.vendor["first_prompt"] == "score this"
    assert info.vendor["tag"] == "nightly"


def test_a_session_with_no_title_falls_back_to_the_runtimes_summary():
    info = session_info_from_sdk(
        SDKSessionInfo(session_id="s", summary="First prompt here"),
        connection="claude-sub",
        project_folder="/work",
    )
    assert info.title == "First prompt here"
    assert info.created_at is None  # None means not reported, never a guess
    assert info.project_folder == "/work"


def test_an_empty_folder_lists_nothing_rather_than_failing(monkeypatch, session_request):
    adapter = session_adapter(monkeypatch, SessionSdk(listing=[]))
    assert adapter.list_sessions(session_request()) == ()


# --- receipt parity -------------------------------------------------------------


def test_the_receipt_names_the_binary_this_run_would_launch(
    monkeypatch, subscription_connection
):
    """Parity with the Codex receipt. The version decides real behavior: the
    prompt-cache TTL lever exists only from 2.1.242."""
    from modelpass.adapters.base import RunRequest

    sdk = SessionSdk()
    sdk.__version__ = "0.2.139"
    adapter = session_adapter(monkeypatch, sdk, version="2.1.233 (Claude Code)")
    plan = plan_launch(subscription_connection, {})
    receipt = adapter.preflight(
        RunRequest(connection=subscription_connection, messages=(), plan=plan)
    )

    joined = " ".join(receipt.notes)
    assert "claude binary: /usr/bin/claude (2.1.233 (Claude Code))" in joined
    assert "claude-agent-sdk 0.2.139" in joined


def test_claude_account_profile_is_allow_listed_and_never_carries_tokens():
    profile = _account_profile_from_auth_status(
        {
            "loggedIn": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "email": "person@example.com",
            "orgId": "org-123",
            "orgName": "Example Co",
            "subscriptionType": "team",
            "oauthToken": "must-not-escape",
        }
    )
    assert profile is not None
    assert profile.subscription_type == "team"
    assert profile.organization_name == "Example Co"
    assert "must-not-escape" not in json.dumps(profile.to_dict())


# --- the core driving this handle ------------------------------------------------


def bridge_over(monkeypatch, sdk, store, connection, **kwargs):
    """A real Bridge whose Anthropic adapter is wired to a fake SDK.

    Worth the setup: everything above proves the adapter maps what it is handed,
    and this proves the object it hands back is the one modelpass.sessions knows
    how to drive -- the id timing, the receipt, the guard envelope and the
    stamped terminal are all the core's, and none of them are exercised by
    calling the handle directly.
    """
    from modelpass.bridge import Bridge
    from modelpass.capabilities import CapabilityRegistry

    adapter = session_adapter(monkeypatch, sdk, **kwargs)
    store.add(connection)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: adapter},
        env={},
    )
    return bridge


def test_the_handle_satisfies_the_protocol_the_core_drives(monkeypatch, session_request):
    from modelpass.adapters.base import SessionHandle

    handle = session_adapter(monkeypatch, SessionSdk()).open_session(session_request())
    assert isinstance(handle, SessionHandle)


def test_a_chat_session_runs_end_to_end_through_the_bridge(
    monkeypatch, store, subscription_connection, tmp_path
):
    sdk = SessionSdk(
        turns=[[ResultMessage(session_id="sess-live", usage={"input_tokens": 7})]],
        messages=[SessionMessage("user", {"role": "user", "content": "hello"})],
    )
    bridge = bridge_over(monkeypatch, sdk, store, subscription_connection)

    session = bridge.new_chat(
        connection="claude-sub",
        system_prompt="Be terse.",
        project_folder=str(tmp_path / "work"),
    )
    # Construction runs the preflight and creates nothing.
    assert session.id is None
    assert session.receipt.detected_auth_mode is AuthMode.SUBSCRIPTION
    assert sdk.clients == []

    events = list(session.send("hello"))

    assert session.id == "sess-live"
    assert isinstance(events[0], ReceiptEvent)
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    # The bridge owns the stamp; the adapter never claims how a run was billed.
    assert terminal.connection == "claude-sub"
    assert terminal.auth_mode is AuthMode.SUBSCRIPTION
    assert terminal.status is TerminalStatus.OK
    assert [(m.role, m.content) for m in session.get_history()] == [(Role.USER, "hello")]

    session.close()
    assert sdk.clients[0].disconnects == 1


def test_the_bridge_lists_the_sessions_in_a_folder(
    monkeypatch, store, subscription_connection, tmp_path
):
    sdk = SessionSdk(listing=[SDKSessionInfo(session_id="sess-1", summary="Scoring")])
    bridge = bridge_over(monkeypatch, sdk, store, subscription_connection)

    listed = bridge.list_sessions(
        connection="claude-sub", project_folder=str(tmp_path / "work")
    )

    assert [info.id for info in listed] == ["sess-1"]
    assert sdk.list_calls == [(str(tmp_path / "work"), False)]


def test_a_scrub_waiting_on_another_run_still_removes_the_key(
    monkeypatch, subscription_connection
):
    """Concurrent subscription runs (found 2026-09-28, RAGauge TEST 1).

    Run A holds the scrub while it spawns, so the key is out of os.environ.
    Run B arrives meanwhile. If B decides *what to remove* before it holds the
    lock, it looks at A's scrubbed environment, finds no key, removes nothing,
    and spawns after A has put the key back: the runtime reports
    apiKeySource='ANTHROPIC_API_KEY' and the run is refused (9 of 16 judge
    calls in that run). The decision must be made under the lock, against the
    environment as it is when B's own spawn happens.
    """
    import threading

    from modelpass.adapters import anthropic as module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-concurrent")
    plan = plan_launch(subscription_connection, {})  # plan saw no key
    assert plan.scrubbed == ()

    calls = []
    b_decided = threading.Event()
    real = module.env_names_to_scrub

    def spy(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:  # run B deciding what to remove
            b_decided.set()
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "env_names_to_scrub", spy)
    seen_by_b = {}

    def run_b():
        with scrubbed_process_env(plan):
            seen_by_b["key_present"] = "ANTHROPIC_API_KEY" in os.environ

    with scrubbed_process_env(plan):  # run A, mid-spawn
        b = threading.Thread(target=run_b)
        b.start()
        # A buggy scrub decides before taking the lock; give it the chance.
        b_decided.wait(timeout=0.5)
    b.join(timeout=5)

    assert seen_by_b == {"key_present": False}
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-concurrent"


def test_an_extensionless_windows_entry_point_resolves_to_its_runnable_twin(tmp_path, monkeypatch):
    """2026-09-29: claude-agent-sdk 0.2.161 ships no bundled binary, so the
    runtime is found on PATH, where npm installs an extensionless shell shim
    beside claude.cmd. Windows cannot run the shim: 'claude auth status'
    silently returned nothing and every subscription preflight failed. On
    Windows the runnable twin (.exe first, then .cmd) is used instead."""
    from modelpass.adapters import anthropic as module

    shim = tmp_path / "claude"
    shim.write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "claude.cmd").write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr(module.os, "name", "nt")
    assert module._runnable_on_windows(str(shim)) == str(tmp_path / "claude.cmd")
    (tmp_path / "claude.exe").write_bytes(b"MZ")
    assert module._runnable_on_windows(str(shim)) == str(tmp_path / "claude.exe")
    assert module._runnable_on_windows(str(tmp_path / "claude.cmd")) == str(tmp_path / "claude.cmd")
    monkeypatch.setattr(module.os, "name", "posix")
    assert module._runnable_on_windows(str(shim)) == str(shim)


# --- a stateless run carries only what the caller sent (2026-09-29) ------------
# Traced live against Claude Code 2.1.281 under a subscription login: a plain
# one-shot call launched in the CALLER's working directory, with the account's
# claude.ai connectors (and their tools), 18 skills, the project's auto-memory
# folder and the agents-md plugin reading that directory -- 7,282 tokens of
# context around a ten-word message. RAGauge's certified judges ran that way.
# With the four switches below the same call carried 460 tokens and no tools,
# connectors, skills or memory. Each switch was shown necessary by ablation.


def test_a_plain_run_loads_no_mcp_server_the_caller_did_not_pass(
    monkeypatch, subscription_request
):
    """Without this the account's claude.ai connectors join every call."""
    options = launch_options(monkeypatch, subscription_request)
    assert options["strict_mcp_config"] is True
    assert options["mcp_servers"] == {}


def test_a_plain_run_starts_in_an_empty_modelpass_directory(
    monkeypatch, subscription_request
):
    """Never the caller's cwd: its AGENTS.md and project memory would join."""
    options = launch_options(monkeypatch, subscription_request)
    cwd = Path(options["cwd"])
    assert cwd.is_dir()
    assert cwd.resolve() != Path(os.getcwd()).resolve()
    assert list(cwd.iterdir()) == []


def test_a_plain_run_switches_off_auto_memory_and_skills(
    monkeypatch, subscription_request
):
    options = launch_options(monkeypatch, subscription_request)
    assert options["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert "disable-slash-commands" in options["extra_args"]
    assert options["extra_args"]["disable-slash-commands"] is None


def test_caller_options_add_to_the_isolation_rather_than_erase_it(
    monkeypatch, request_factory
):
    """A caller's env and extra_args merge in; the isolation switches stay."""
    options = launch_options(monkeypatch, request_factory(options={
        "env": {"MY_VAR": "x"},
        "extra_args": {"some-flag": "v"},
    }))
    assert options["env"]["MY_VAR"] == "x"
    assert options["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert options["extra_args"]["some-flag"] == "v"
    assert "disable-slash-commands" in options["extra_args"]


def test_a_caller_may_name_its_own_working_directory(
    monkeypatch, request_factory, tmp_path
):
    options = launch_options(monkeypatch, request_factory(options={"cwd": str(tmp_path)}))
    assert options["cwd"] == str(tmp_path)


@pytest.mark.parametrize("key,value", [
    ("strict_mcp_config", False),
    ("mcp_servers", {"x": {"type": "stdio", "command": "x"}}),
    ("setting_sources", ["user"]),
    ("tools", ["Bash"]),
    ("allowed_tools", ["Bash"]),
])
def test_a_caller_option_cannot_silently_undo_isolation(
    monkeypatch, request_factory, key, value
):
    """Fable QA 2026-09-29: these used to be applied as given."""
    with pytest.raises(CapabilityNotSupported):
        launch_options(monkeypatch, request_factory(options={key: value}))


# --- run time checks the login and asks no identity question ---------------------
#
# Owner decision, 2026-10-01: a run compares nothing with the pinned account.
# That stands. What changed on 2026-11-13 is how a run knows a login is there:
# it used to read the credential file and run no subprocess; it now shares the
# cached ``claude auth status`` answer, because modelpass no longer opens that
# file.


def _clock(monkeypatch, module):
    """Drive the adapter's cache clock by hand."""
    now = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    return now


def test_a_run_preflight_reports_no_identity(
    monkeypatch, subscription_connection, tmp_path
):
    """Was: a run preflight never invokes the probe. It does now, and still
    hands back no account for anyone to compare."""
    adapter, _ = _preflight_adapter(monkeypatch, auth_status=lambda b, e: dict(_LOGGED_IN))
    receipt = adapter.preflight(_profile_request(subscription_connection, tmp_path))
    assert receipt.ok, receipt.problem
    assert receipt.account_profile is None
    assert receipt.account is None
    assert "person@example.com" not in repr(receipt)


def test_runs_share_one_probe_per_cache_window(
    monkeypatch, subscription_connection, tmp_path
):
    """The probe is seconds of subprocess; a batch of runs pays for it once."""
    calls: list[int] = []

    def auth_status(binary, env):
        calls.append(1)
        return {"loggedIn": True}

    adapter, module = _preflight_adapter(monkeypatch, auth_status=auth_status)
    now = _clock(monkeypatch, module)
    request = _profile_request(subscription_connection, tmp_path)
    for _ in range(5):
        assert adapter.preflight(request).ok
    assert adapter.identity_preflight(request).ok
    assert len(calls) == 1

    now[0] += module._IDENTITY_CACHE_TTL_SECONDS + 1
    assert adapter.preflight(request).ok
    assert len(calls) == 2


def test_four_concurrent_run_preflights_all_pass_when_the_probe_then_fails(
    monkeypatch, subscription_connection, tmp_path
):
    """The 2026-10-01 failure: an empty probe refused every call for a minute.

    A login confirmed minutes ago is not refused because the probe now returns
    nothing: the run goes on the earlier answer and the receipt says how old
    it is."""
    from concurrent.futures import ThreadPoolExecutor

    answers = [{"loggedIn": True}]

    def auth_status(binary, env):
        return answers.pop(0) if answers else None

    adapter, module = _preflight_adapter(monkeypatch, auth_status=auth_status)
    now = _clock(monkeypatch, module)
    request = _profile_request(subscription_connection, tmp_path)
    assert adapter.preflight(request).ok

    now[0] += module._IDENTITY_CACHE_TTL_SECONDS + 30
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _: adapter.preflight(request), range(8)))
    assert all(r.ok for r in receipts), [r.problem for r in receipts]
    for receipt in receipts:
        assert any(
            "returned nothing just now" in note and "90s ago" in note
            for note in receipt.notes
        )


def test_an_old_answer_is_not_evidence_for_ever(
    monkeypatch, subscription_connection, tmp_path
):
    answers = [{"loggedIn": True}]
    adapter, module = _preflight_adapter(
        monkeypatch, auth_status=lambda b, e: answers.pop(0) if answers else None
    )
    now = _clock(monkeypatch, module)
    request = _profile_request(subscription_connection, tmp_path)
    assert adapter.preflight(request).ok

    now[0] += module._LOGIN_EVIDENCE_GRACE_SECONDS + 1
    receipt = adapter.preflight(request)
    assert receipt.ok is False
    assert "could not confirm a Claude Code login" in receipt.problem


def test_a_setup_check_never_rests_on_an_old_answer(
    monkeypatch, subscription_connection, tmp_path
):
    """Pinning an account from an answer nobody just read would record a guess."""
    answers = [dict(_LOGGED_IN)]
    adapter, module = _preflight_adapter(
        monkeypatch, auth_status=lambda b, e: answers.pop(0) if answers else None
    )
    now = _clock(monkeypatch, module)
    request = _profile_request(subscription_connection, tmp_path)
    assert adapter.identity_preflight(request).ok

    now[0] += module._IDENTITY_CACHE_TTL_SECONDS + 1
    receipt = adapter.identity_preflight(request)
    assert receipt.ok is False
    assert receipt.account_profile is None


def test_a_probe_that_never_answers_refuses_every_run(
    monkeypatch, subscription_connection, tmp_path
):
    """Fail closed has no fallback to fall back on: nothing was ever confirmed."""
    adapter, _ = _preflight_adapter(monkeypatch, auth_status=lambda b, e: None)
    request = _profile_request(subscription_connection, tmp_path)
    receipts = [adapter.preflight(request) for _ in range(3)]
    assert not any(r.ok for r in receipts)
    assert all(r.detected_auth_mode is None for r in receipts)


def test_logging_out_is_seen_once_the_cache_window_passes(
    monkeypatch, subscription_connection, tmp_path
):
    answers = [{"loggedIn": True}, {"loggedIn": False}]
    adapter, module = _preflight_adapter(
        monkeypatch, auth_status=lambda b, e: answers.pop(0)
    )
    now = _clock(monkeypatch, module)
    request = _profile_request(subscription_connection, tmp_path)
    assert adapter.preflight(request).ok

    now[0] += module._IDENTITY_CACHE_TTL_SECONDS + 1
    receipt = adapter.preflight(request)
    assert receipt.ok is False
    assert "claude /login" in receipt.problem


def test_the_setup_preflight_reports_identity(
    monkeypatch, subscription_connection, tmp_path
):
    calls: list[str] = []

    def auth_status(binary, env):
        calls.append(env.get("CLAUDE_CONFIG_DIR"))
        return {"loggedIn": True, "email": "work@example.com"}

    adapter, _ = _preflight_adapter(monkeypatch, auth_status=auth_status)
    receipt = adapter.identity_preflight(_profile_request(subscription_connection, tmp_path))
    assert calls == [str(tmp_path)]
    assert receipt.account_profile is not None
    assert receipt.account_profile.email == "work@example.com"
    assert receipt.account == "work@example.com"


def test_an_empty_identity_answer_is_never_cached(monkeypatch):
    from modelpass.adapters import anthropic as module

    answers = iter([None, {"loggedIn": True}])
    adapter = module.AnthropicAdapter(auth_status=lambda b, e: next(answers))
    assert adapter._cached_auth_status("claude", {"CLAUDE_CONFIG_DIR": "x"}) is None
    assert adapter._cached_auth_status("claude", {"CLAUDE_CONFIG_DIR": "x"}) == {
        "loggedIn": True
    }


def test_concurrent_identity_probes_for_one_config_dir_share_one_answer():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from modelpass.adapters import anthropic as module

    calls = []
    gate = threading.Event()

    def auth_status(binary, env):
        calls.append(1)
        gate.wait(0.2)
        return {"loggedIn": True}

    adapter = module.AnthropicAdapter(auth_status=auth_status)
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(
            pool.map(
                lambda _: adapter._cached_auth_status("claude", {"CLAUDE_CONFIG_DIR": "x"}),
                range(4),
            )
        )
    assert answers == [{"loggedIn": True}] * 4
    assert len(calls) == 1


def test_a_schema_run_leaves_room_for_a_validation_retry(monkeypatch, request_factory):
    """A StructuredOutput submission that fails validation is re-asked; under
    max_turns=1 that retry ended the run error_max_turns (2026-10-13)."""
    from modelpass.adapters.anthropic import STRUCTURED_OUTPUT_MAX_TURNS

    request = request_factory(schema={"type": "object", "properties": {}})
    options = launch_options(monkeypatch, request)
    assert options["max_turns"] == STRUCTURED_OUTPUT_MAX_TURNS
    assert options["output_format"]["type"] == "json_schema"


def test_a_plain_chat_run_keeps_one_turn(monkeypatch, request_factory):
    assert launch_options(monkeypatch, request_factory())["max_turns"] == 1


def test_a_callers_turn_ceiling_still_wins_on_a_schema_run(monkeypatch, request_factory):
    request = request_factory(
        schema={"type": "object", "properties": {}}, options={"max_turns": 5}
    )
    assert launch_options(monkeypatch, request)["max_turns"] == 5
