"""Offline tests for the Anthropic adapter.

No network, no credential, no vendor package import at module scope. The event
mapping is a pure function of a vendor message, so it is tested against stand-in
objects shaped exactly like the SDK's dataclasses (matched by class name, which
is what the mapper dispatches on).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from modelpass.adapters.anthropic import (
    AnthropicAdapter,
    _account_profile_from_auth_status,
    _run_coroutine_threadsafe,
    _tool_options,
    assert_expected_auth_mode,
    build_caller_tool_server,
    credentials_path,
    map_message,
    parse_cli_version,
    read_credential_status,
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
    assert "mcp_servers" not in options


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


# --- credential detection ------------------------------------------------------


def _write_credentials(tmp_path, *, expires_at: float, refresh: str) -> dict[str, str]:
    (tmp_path / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "not-a-real-token",
                    "refreshToken": refresh,
                    "expiresAt": int(expires_at * 1000),
                    "subscriptionType": "pro",
                    "rateLimitTier": "default_claude_ai",
                }
            }
        ),
        encoding="utf-8",
    )
    return {"CLAUDE_CONFIG_DIR": str(tmp_path)}


def test_credentials_path_honours_claude_config_dir(tmp_path):
    env = {"CLAUDE_CONFIG_DIR": str(tmp_path)}
    assert credentials_path(env) == tmp_path / ".credentials.json"


@pytest.mark.skipif(os.sys.platform == "darwin", reason="macOS uses the keychain")
def test_a_live_login_is_reported_usable(tmp_path):
    env = _write_credentials(tmp_path, expires_at=time.time() + 3600, refresh="r")
    status = read_credential_status(env)
    assert status.present and status.usable and not status.expired
    assert status.subscription_type == "pro"


@pytest.mark.skipif(os.sys.platform == "darwin", reason="macOS uses the keychain")
def test_an_expired_login_without_refresh_is_not_usable(tmp_path):
    """The exact shape found on the dev machine: expired, no refresh token."""
    env = _write_credentials(tmp_path, expires_at=time.time() - 3600, refresh="")
    status = read_credential_status(env)
    assert status.present and status.expired and not status.refreshable
    assert not status.usable


@pytest.mark.skipif(os.sys.platform == "darwin", reason="macOS uses the keychain")
def test_an_expired_but_refreshable_login_stays_usable(tmp_path):
    env = _write_credentials(tmp_path, expires_at=time.time() - 3600, refresh="r")
    assert read_credential_status(env).usable


@pytest.mark.skipif(os.sys.platform == "darwin", reason="macOS uses the keychain")
def test_a_missing_login_is_reported_absent(tmp_path):
    status = read_credential_status({"CLAUDE_CONFIG_DIR": str(tmp_path)})
    assert not status.present and not status.usable


def _write_logged_out_credentials(tmp_path) -> dict[str, str]:
    """The shape Claude Code leaves on Windows when the CLI is logged out.

    Observed on a real machine (2026-09-21): the secrets are blanked in place
    and everything else stays, so the file still describes a plan in detail
    while holding nothing that can authenticate.
    """
    (tmp_path / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "",
                    "refreshToken": "",
                    "expiresAt": 0,
                    "subscriptionType": "max",
                    "rateLimitTier": "default_claude_max_20x",
                    "scopes": ["user:inference", "user:profile"],
                },
                "organizationUuid": "not-a-real-uuid",
            }
        ),
        encoding="utf-8",
    )
    return {"CLAUDE_CONFIG_DIR": str(tmp_path)}


@pytest.mark.skipif(os.sys.platform == "darwin", reason="macOS uses the keychain")
def test_a_blanked_expiry_is_no_expiry_rather_than_the_epoch(tmp_path):
    """``expiresAt: 0`` is a cleared field, not a token that lapsed in 1970.

    Read as a timestamp it is always in the past, so `expired` came back True
    for a login that never existed -- a fact stated about nothing.
    """
    status = read_credential_status(_write_logged_out_credentials(tmp_path))
    assert status.expires_at is None
    assert not status.expired


@pytest.mark.skipif(os.sys.platform == "darwin", reason="macOS uses the keychain")
def test_a_logged_out_file_is_told_apart_from_a_missing_one(tmp_path, tmp_path_factory):
    """Both are `present=False` and both want `claude /login`; only one of them
    also explains why a populated plan is sitting in the file."""
    logged_out = read_credential_status(_write_logged_out_credentials(tmp_path))
    missing = read_credential_status(
        {"CLAUDE_CONFIG_DIR": str(tmp_path_factory.mktemp("empty"))}
    )

    assert not logged_out.present and not logged_out.usable
    assert not missing.present and not missing.usable
    assert logged_out.logged_out and not missing.logged_out
    # The plan metadata is still readable, which is exactly what makes the
    # "no login found" wording misleading without the flag.
    assert logged_out.subscription_type == "max"


def test_a_logged_out_cli_is_not_reported_as_no_login_at_all(
    monkeypatch, subscription_connection
):
    """The receipt has to say the file was found and is empty.

    Told only "no Claude Code login found" against a file that plainly holds a
    plan, a reader concludes modelpass looked in the wrong place and goes
    hunting for an external credential store. There is none on this platform:
    the CLI is simply logged out.
    """
    adapter, module = _preflight_adapter(
        monkeypatch, auth_status=lambda binary, env: {"loggedIn": False}
    )
    monkeypatch.setattr(
        module,
        "read_credential_status",
        lambda env=None: module.CredentialStatus(
            source="test", present=False, logged_out=True, subscription_type="max"
        ),
    )

    plan = plan_launch(subscription_connection, {})
    receipt = adapter.preflight(
        RunRequest(connection=subscription_connection, messages=(), plan=plan)
    )

    assert not receipt.ok
    assert "logged out" in receipt.problem
    assert "claude /login" in receipt.problem


def test_a_genuinely_missing_file_keeps_the_wording_it_always_had(
    monkeypatch, subscription_connection
):
    adapter, module = _preflight_adapter(
        monkeypatch, auth_status=lambda binary, env: {"loggedIn": False}
    )
    monkeypatch.setattr(
        module,
        "read_credential_status",
        lambda env=None: module.CredentialStatus(source="test", present=False),
    )

    plan = plan_launch(subscription_connection, {})
    receipt = adapter.preflight(
        RunRequest(connection=subscription_connection, messages=(), plan=plan)
    )

    assert not receipt.ok
    assert "no Claude Code login found" in receipt.problem


# --- expired-but-refreshable: verified, not assumed -----------------------------


def _preflight_adapter(monkeypatch, *, auth_status):
    """A subscription adapter wired to a fake CLI, for the refresh-verification
    tests below. No subprocess ever runs: both probes are injected."""
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


def test_an_expired_but_refreshable_login_is_verified_and_passes_once_refreshed(
    monkeypatch, subscription_connection
):
    """The real-world failure this guards: 'ok=True' on a login whose refresh
    silently failed lets a run start and then die on the first token, because
    Claude Code can report 'logged in' without ever clearing a dead
    refresh_token. Preflight now re-checks after one CLI probe rather than
    trusting the passive note that used to be the whole answer.
    """
    adapter, module = _preflight_adapter(
        monkeypatch, auth_status=lambda binary, env: {"loggedIn": True, "subscriptionType": "pro"}
    )
    answers = iter(
        [
            module.CredentialStatus(
                source="test", present=True, expired=True, refreshable=True,
                expires_at=time.time() - 10,
            ),
            module.CredentialStatus(
                source="test", present=True, expired=False, refreshable=True,
                expires_at=time.time() + 3600, subscription_type="pro",
            ),
        ]
    )
    monkeypatch.setattr(module, "read_credential_status", lambda env=None: next(answers))

    plan = plan_launch(subscription_connection, {})
    receipt = adapter.preflight(
        RunRequest(connection=subscription_connection, messages=(), plan=plan)
    )

    assert receipt.ok
    assert "was refreshed during preflight" in " ".join(receipt.notes)
    assert receipt.plan_name == "pro"


def test_a_permanently_failed_refresh_fails_closed_and_says_to_log_in_again(
    monkeypatch, subscription_connection
):
    """Codex/Claude both record a permanent refresh failure without logging the
    account out (dead refresh_token, revoked grant) -- so a stored token still
    showing 'expired' after the forced re-probe is exactly that failure, and
    the receipt must not let the run proceed to a metered fallback or a
    mid-run auth error."""
    adapter, module = _preflight_adapter(
        monkeypatch, auth_status=lambda binary, env: {"loggedIn": True}
    )
    still_expired = module.CredentialStatus(
        source="test", present=True, expired=True, refreshable=True,
        expires_at=time.time() - 10,
    )
    monkeypatch.setattr(module, "read_credential_status", lambda env=None: still_expired)

    plan = plan_launch(subscription_connection, {})
    receipt = adapter.preflight(
        RunRequest(connection=subscription_connection, messages=(), plan=plan)
    )

    assert not receipt.ok
    assert "could not be refreshed" in receipt.problem
    assert "claude /login" in receipt.problem


# --- macOS ---------------------------------------------------------------------
#
# Implemented from Claude Code's own documentation (the Keychain entry is keyed
# to CLAUDE_CONFIG_DIR; a plaintext .credentials.json under the same directory
# is the fallback when the Keychain refuses the write) and not yet exercised on
# a Mac. Every branch below therefore monkeypatches sys.platform, which is the
# only way this machine can reach them at all.


def _darwin_preflight(monkeypatch, request, *, auth_status):
    from modelpass.adapters import anthropic as module

    adapter = AnthropicAdapter(auth_status=lambda binary, env: auth_status)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "_sdk", lambda: FakeSdk())
    monkeypatch.setattr(adapter, "_cli_status", lambda sdk: ("claude", None))
    monkeypatch.setattr(adapter, "_cli_version", lambda binary, env: "2.1.242")
    return adapter.preflight(request)


def _isolated_request(request_factory, root):
    connection = replace(request_factory().connection, config_dir=str(root))
    return replace(
        request_factory(),
        connection=connection,
        plan=plan_launch(connection, {}),
    )


def test_mac_reads_the_keychain_fallback_file_under_the_config_dir(monkeypatch, tmp_path):
    """When Claude Code fell back to a file, macOS parses it exactly like Linux."""
    from modelpass.adapters import anthropic as module

    root = tmp_path / "claude-work"
    root.mkdir()
    env = _write_credentials(root, expires_at=time.time() + 3600, refresh="r")
    monkeypatch.setattr(module.sys, "platform", "darwin")

    status = read_credential_status(env)
    assert status.present is True
    assert status.keychain_only is False
    assert str(root / ".credentials.json") in status.source


def test_mac_without_a_fallback_file_names_the_keyed_keychain_entry(monkeypatch, tmp_path):
    from modelpass.adapters import anthropic as module

    root = tmp_path / "claude-work"
    monkeypatch.setattr(module.sys, "platform", "darwin")

    status = read_credential_status({"CLAUDE_CONFIG_DIR": str(root)})
    assert status.present is True
    assert status.keychain_only is True
    assert "macOS Keychain" in status.source
    assert str(root) in status.source


def test_mac_config_dir_profile_is_ok_when_auth_status_reports_a_login(
    monkeypatch, request_factory, tmp_path
):
    receipt = _darwin_preflight(
        monkeypatch,
        _isolated_request(request_factory, tmp_path / "claude-work"),
        auth_status={"loggedIn": True, "email": "work@example.com"},
    )
    assert receipt.ok is True
    assert receipt.account_profile is not None
    assert receipt.account_profile.email == "work@example.com"


def test_mac_config_dir_profile_fails_closed_when_auth_status_says_logged_out(
    monkeypatch, request_factory, tmp_path
):
    """The Keychain cannot be read, so `claude auth status` is the whole evidence."""
    root = tmp_path / "claude-work"
    receipt = _darwin_preflight(
        monkeypatch,
        _isolated_request(request_factory, root),
        auth_status={"loggedIn": False},
    )
    assert receipt.ok is False
    assert "claude auth status reports no login" in (receipt.problem or "")
    assert str(root) in (receipt.problem or "")


def test_mac_config_dir_profile_says_so_when_the_probe_returns_nothing(
    monkeypatch, request_factory, tmp_path
):
    receipt = _darwin_preflight(
        monkeypatch,
        _isolated_request(request_factory, tmp_path / "claude-work"),
        auth_status=None,
    )
    assert receipt.ok is True
    assert any("identity could not be confirmed" in note for note in receipt.notes)


@pytest.mark.skipif(os.sys.platform == "darwin", reason="macOS uses the keychain")
def test_credential_status_never_carries_a_secret(tmp_path):
    """The status type must have no field capable of holding a token."""
    env = _write_credentials(tmp_path, expires_at=time.time() + 60, refresh="r")
    status = read_credential_status(env)
    assert "not-a-real-token" not in repr(status)


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
    from modelpass.adapters import anthropic as module
    from modelpass.adapters.base import RunRequest

    monkeypatch.setattr(
        module,
        "read_credential_status",
        lambda env=None: module.CredentialStatus(
            source="Claude Code login (test)",
            present=True,
            subscription_type="Max 5x",
            rate_limit_tier="default",
        ),
    )
    plan = plan_launch(subscription_connection, {})
    request = RunRequest(connection=subscription_connection, messages=(), plan=plan)
    receipt = AnthropicAdapter()._subscription_receipt(request, [])

    joined = " ".join(receipt.notes)
    assert "not knowable before a run" in joined
    assert "rate_limit" in joined
    assert "rate limit tier: default" in joined
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
    adapter = AnthropicAdapter(claude_version=lambda binary, env: version)
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
    from modelpass.adapters import anthropic as module
    from modelpass.adapters.base import RunRequest

    sdk = SessionSdk()
    sdk.__version__ = "0.2.139"
    adapter = session_adapter(monkeypatch, sdk, version="2.1.233 (Claude Code)")
    monkeypatch.setattr(
        module,
        "read_credential_status",
        lambda env=None: module.CredentialStatus(source="test", present=True),
    )
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
    from modelpass.adapters import anthropic as module
    from modelpass.bridge import Bridge
    from modelpass.capabilities import CapabilityRegistry

    adapter = session_adapter(monkeypatch, sdk, **kwargs)
    monkeypatch.setattr(
        module,
        "read_credential_status",
        lambda env=None: module.CredentialStatus(
            source="Claude Code login (test)", present=True, subscription_type="pro"
        ),
    )
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
