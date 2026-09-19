"""The ``anthropic-api`` adapter, end to end, with no network anywhere (ticket 1.6).

Every test here drives a **fake transport**: an object shaped like the pieces of
``anthropic.Anthropic`` this adapter touches (``messages.stream(...)`` and
``models.list(...)``), injected through the adapter's ``client_factory``
constructor seam. Nothing in this file imports the vendor SDK to make a call,
nothing opens a socket, and nothing spends. The live drive that *does* spend is
``tests/live/test_anthropic_api_live.py``, deselected by default.

The fake is deliberately literal about the shapes the adapter reads -- blocks
with ``.type``, a usage object with the four vendor field names, a stream that
yields the SDK's helper events and then answers ``get_final_message()`` -- because
a fake that is looser than the vendor is a test that passes against an adapter
that would not run. Each shape is pinned against the installed SDK's own types in
:func:`test_the_fake_matches_the_installed_sdk_shapes`, which is what keeps this
file honest as the SDK moves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from modelpass.adapters import ADAPTER_REGISTRY, load_adapter
from modelpass.adapters.anthropic import token_usage as agent_token_usage
from modelpass.adapters.anthropic_api import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    SDK_VERSION_READ,
    AnthropicAPIAdapter,
    token_usage,
)
from modelpass.adapters.base import RunRequest, SessionRequest
from modelpass.bridge import Bridge
from modelpass.capabilities import Capability, CapabilityRegistry, Support
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.errors import (
    AdapterNotImplemented,
    CapabilityNotSupported,
    PreflightFailed,
)
from modelpass.preflight import credential_fingerprint, plan_launch
from modelpass.runtimes import Runtime
from modelpass.tools import ToolDef
from modelpass.types import (
    AuthMode,
    CacheControl,
    GuardStopEvent,
    Message,
    Role,
    SessionKind,
    StructuredOutputEvent,
    TerminalEvent,
    TerminalStatus,
    TextBlock,
    TextDeltaEvent,
    ThinkingEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    VendorEvent,
)

KEY = "sk-ant-not-a-real-key-0123456789"
KEY_VAR = "MODELPASS_TEST_ANTHROPIC_KEY"
ENV = {KEY_VAR: KEY}
MODEL = "claude-opus-5"


# --- the fake transport ----------------------------------------------------------


@dataclass
class FakeUsage:
    """The four fields the adapter maps, under the vendor's own names."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class TextBlockOut:
    text: str
    type: str = "text"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass
class ThinkingBlockOut:
    thinking: str
    signature: str = "sig"
    type: str = "thinking"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "thinking", "thinking": self.thinking, "signature": self.signature}


@dataclass
class ToolUseBlockOut:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


@dataclass
class StopDetails:
    category: str | None = None
    type: str = "refusal"


@dataclass
class FakeMessage:
    content: list[Any] = field(default_factory=list)
    stop_reason: str | None = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_details: StopDetails | None = None


@dataclass
class HelperEvent:
    """One of the SDK stream helper's own events (``text``, ``thinking``, ...)."""

    type: str
    text: str = ""
    thinking: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text, "thinking": self.thinking}


@dataclass
class Round:
    """One scripted response: what streams, and what the final message says."""

    message: FakeMessage
    events: list[HelperEvent] = field(default_factory=list)
    raises: Exception | None = None


class FakeStream:
    def __init__(self, round_: Round) -> None:
        self._round = round_
        self.closed = False

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def __iter__(self):
        yield from self._round.events

    def get_final_message(self) -> FakeMessage:
        return self._round.message


class FakeMessages:
    def __init__(self, client: FakeClient) -> None:
        self._client = client

    def stream(self, **params: Any) -> FakeStream:
        self._client.calls.append(params)
        if not self._client.rounds:
            raise AssertionError("the adapter asked for more rounds than were scripted")
        round_ = self._client.rounds.pop(0)
        if round_.raises is not None:
            raise round_.raises
        stream = FakeStream(round_)
        self._client.streams.append(stream)
        return stream


class FakeModels:
    def __init__(self, client: FakeClient) -> None:
        self._client = client

    def list(self, **kwargs: Any) -> Any:
        self._client.model_list_calls += 1
        if isinstance(self._client.models_answer, Exception):
            raise self._client.models_answer
        return self._client.models_answer


class FakeClient:
    def __init__(self, rounds: list[Round], models_answer: Any = None) -> None:
        self.rounds = list(rounds)
        self.calls: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.models_answer = models_answer
        self.model_list_calls = 0
        self.messages = FakeMessages(self)
        self.models = FakeModels(self)


class Factory:
    """Records how the client was constructed -- the point of items 1 and 2."""

    def __init__(self, rounds: list[Round] | None = None, models_answer: Any = None) -> None:
        self.rounds = rounds or []
        self.models_answer = models_answer
        self.kwargs: list[dict[str, Any]] = []
        self.clients: list[FakeClient] = []

    def __call__(self, *, api_key: str, base_url: str | None) -> FakeClient:
        self.kwargs.append({"api_key": api_key, "base_url": base_url})
        client = FakeClient(self.rounds, self.models_answer)
        self.clients.append(client)
        return client


def text_round(text: str, **kwargs: Any) -> Round:
    usage = kwargs.pop("usage", FakeUsage(input_tokens=10, output_tokens=5))
    return Round(
        message=FakeMessage(content=[TextBlockOut(text)], usage=usage, **kwargs),
        events=[HelperEvent(type="text", text=text)],
    )


def tool_round(call_id: str, name: str, arguments: dict[str, Any]) -> Round:
    return Round(
        message=FakeMessage(
            content=[ToolUseBlockOut(id=call_id, name=name, input=arguments)],
            stop_reason="tool_use",
            usage=FakeUsage(input_tokens=8, output_tokens=4),
        ),
        events=[],
    )


# --- connections, requests, bridges ----------------------------------------------


def connection(
    *,
    name: str = "claude-key",
    model: str | None = MODEL,
    base_url: str | None = None,
    credential_ref: str = f"env:{KEY_VAR}",
    guards: Guards | None = None,
) -> Connection:
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(credential_ref),
        model=model,
        base_url=base_url,
        guards=guards or Guards(),
    )


def run_request(
    conn: Connection | None = None,
    *,
    messages: tuple[Message, ...] | None = None,
    **kwargs: Any,
) -> RunRequest:
    conn = conn or connection()
    return RunRequest(
        connection=conn,
        messages=messages or (Message(role=Role.USER, content="hello"),),
        plan=plan_launch(conn, ENV),
        **kwargs,
    )


def adapter(factory: Factory) -> AnthropicAPIAdapter:
    return AnthropicAPIAdapter(client_factory=factory, env=ENV)


def session_request(conn: Connection | None = None) -> SessionRequest:
    """A session request this adapter will never honour, for the refusal tests."""
    conn = conn or connection()
    return SessionRequest(
        connection=conn,
        plan=plan_launch(conn, ENV),
        kind=SessionKind.CHAT,
        project_folder=".",
    )


def drive(request: RunRequest, factory: Factory) -> list[Any]:
    return list(adapter(factory).run(request))


def bridge_for(
    conn: Connection, factory: Factory, store: Any
) -> tuple[Bridge, AnthropicAPIAdapter]:
    store.add(conn)
    api = adapter(factory)
    return (
        Bridge(
            store=store,
            registry=CapabilityRegistry(),
            adapters={Runtime.ANTHROPIC_API: api},
            env=dict(ENV),
        ),
        api,
    )


# The handful of cases below that need the vendor package actually installed:
# they load the adapter through the registry, read ``runtime_available``, or
# drive a :class:`Bridge`, which refuses a runtime whose SDK does not import.
# Everything else in this file runs against the fake transport and needs
# nothing, which is D2: the suite is green on a machine with no extras.
NEEDS_SDK = pytest.mark.skipif(
    not AnthropicAPIAdapter.is_available(),
    reason="needs the modelpass[anthropic-api] extra: anthropic is not installed",
)


# --- 1. registration, and a client built with an explicit key ---------------------


@NEEDS_SDK
def test_the_runtime_resolves_to_this_adapter_through_the_usual_discovery():
    entry = ADAPTER_REGISTRY[Runtime.ANTHROPIC_API]
    assert entry.module == "modelpass.adapters.anthropic_api"
    assert entry.attribute == "AnthropicAPIAdapter"
    assert (entry.extra, entry.package) == ("anthropic-api", "anthropic")
    loaded = load_adapter(Runtime.ANTHROPIC_API)
    assert isinstance(loaded, AnthropicAPIAdapter)
    assert loaded.runtime is Runtime.ANTHROPIC_API


def test_the_client_is_constructed_with_the_connections_own_key():
    factory = Factory([text_round("hi")])
    drive(run_request(), factory)
    assert factory.kwargs == [{"api_key": KEY, "base_url": None}]


def test_a_connections_base_url_reaches_the_client_and_absence_stays_absent():
    factory = Factory([text_round("hi")])
    drive(run_request(connection(base_url="https://proxy.example.com/v1")), factory)
    assert factory.kwargs[0]["base_url"] == "https://proxy.example.com/v1"


def test_a_secret_backed_credential_resolves_at_the_moment_of_use(tmp_path):
    from modelpass.secrets import SecretStore

    secrets = SecretStore(tmp_path / "home")
    secrets.set("anthropic-key", KEY)
    factory = Factory([text_round("hi")])
    api = AnthropicAPIAdapter(client_factory=factory, env={}, secrets=secrets)
    conn = connection(credential_ref="secret:anthropic-key")
    list(api.run(run_request(conn)))
    assert factory.kwargs == [{"api_key": KEY, "base_url": None}]


# --- 2. preflight -----------------------------------------------------------------


@NEEDS_SDK
def test_the_receipt_carries_a_fingerprint_and_none_of_the_subscription_fields():
    factory = Factory()
    receipt = adapter(factory).preflight(run_request())
    assert receipt.ok
    assert receipt.detected_auth_mode is AuthMode.API_KEY
    assert receipt.account == credential_fingerprint(KEY)
    assert KEY not in json.dumps(receipt.to_dict())
    assert receipt.plan_name is None
    assert receipt.binary is None
    assert receipt.runtime_available is True
    assert receipt.credential_source == f"environment variable {KEY_VAR}"


def test_an_unset_credential_fails_the_preflight_rather_than_the_run():
    api = AnthropicAPIAdapter(client_factory=Factory(), env={})
    receipt = api.preflight(run_request())
    assert not receipt.ok
    assert KEY_VAR in (receipt.problem or "")


def test_a_connection_with_no_model_still_preflights_but_the_receipt_says_so():
    """Configuring a connection without a model is legitimate; running one
    without a model anywhere is not, and the two answers live in two places."""
    factory = Factory()
    receipt = adapter(factory).preflight(run_request(connection(model=None)))
    assert receipt.ok
    assert any("no default model" in note for note in receipt.notes)


def test_a_run_with_no_model_anywhere_is_refused_before_anything_is_sent():
    factory = Factory()
    with pytest.raises(PreflightFailed, match="no default model"):
        list(adapter(factory).run(run_request(connection(model=None))))
    assert factory.kwargs == []


def test_a_model_passed_to_the_call_satisfies_a_connection_that_names_none():
    factory = Factory()
    request = run_request(connection(model=None), model=MODEL)
    assert adapter(factory).preflight(request).ok


def test_the_model_list_probe_is_opt_in_and_then_cached():
    listing = type("Listing", (), {"data": [type("M", (), {"id": MODEL})()]})()
    factory = Factory(models_answer=listing)
    api = adapter(factory)
    plain = api.preflight(run_request())
    assert factory.clients == []
    assert not any("probe" in note for note in plain.notes)

    request = run_request(options={"probe_credential": True})
    first = api.preflight(request)
    second = api.preflight(request)
    assert factory.clients[0].model_list_calls == 1
    assert any("model(s) listed" in note for note in first.notes)
    assert first.notes == second.notes

    api.invalidate_identity_cache()
    api.preflight(request)
    assert sum(c.model_list_calls for c in factory.clients) == 2


def test_a_probe_that_cannot_reach_the_endpoint_is_a_reported_finding():
    factory = Factory(models_answer=RuntimeError("nope"))
    api = adapter(factory)
    receipt = api.preflight(run_request(options={"probe_credential": True}))
    assert not receipt.ok
    assert "RuntimeError: nope" in (receipt.problem or "")


@NEEDS_SDK
def test_runtime_available_reports_whether_the_sdk_imports():
    assert AnthropicAPIAdapter.is_available() is True


# --- 3. the stream, the events, and the usage mapping -----------------------------


def test_a_plain_call_streams_text_then_usage_then_a_terminal():
    factory = Factory([text_round("hello there")])
    events = drive(run_request(), factory)
    assert [type(e) for e in events] == [TextDeltaEvent, UsageEvent, TerminalEvent]
    assert events[0].text == "hello there"
    assert events[-1].status is TerminalStatus.OK


def test_thinking_blocks_become_thinking_events():
    round_ = Round(
        message=FakeMessage(content=[ThinkingBlockOut("weighing it"), TextBlockOut("yes")]),
        events=[
            HelperEvent(type="thinking", thinking="weighing it"),
            HelperEvent(type="text", text="yes"),
        ],
    )
    events = drive(run_request(), Factory([round_]))
    assert isinstance(events[0], ThinkingEvent)
    assert events[0].text == "weighing it"


def test_usage_maps_the_cache_fields_exactly_as_the_agent_runtime_does():
    raw = {
        "input_tokens": 11,
        "output_tokens": 22,
        "cache_read_input_tokens": 33,
        "cache_creation_input_tokens": 44,
    }
    expected = TokenUsage(
        input_tokens=11, output_tokens=22, cached_input_tokens=33, cache_write_tokens=44
    )
    # The same dict through the agent adapter, and the same object through this
    # one: a consumer's accounting must not learn that a run changed runtimes.
    assert agent_token_usage(raw) == expected
    assert token_usage(FakeUsage(**raw)) == expected
    assert token_usage(raw) == expected
    assert token_usage(None) == TokenUsage()


def test_the_usage_event_carries_the_cache_split_through_the_stream():
    usage = FakeUsage(
        input_tokens=5,
        output_tokens=6,
        cache_read_input_tokens=700,
        cache_creation_input_tokens=800,
    )
    events = drive(run_request(), Factory([text_round("hi", usage=usage)]))
    reported = next(e for e in events if isinstance(e, UsageEvent))
    assert reported.usage.cached_input_tokens == 700
    assert reported.usage.cache_write_tokens == 800
    assert reported.usage.billable_input_tokens == 805


def test_a_helper_event_with_no_word_in_the_vocabulary_becomes_a_vendor_event():
    round_ = Round(
        message=FakeMessage(content=[TextBlockOut("x")]),
        events=[HelperEvent(type="input_json"), HelperEvent(type="text", text="x")],
    )
    events = drive(run_request(), Factory([round_]))
    vendor = next(e for e in events if isinstance(e, VendorEvent))
    assert vendor.name == "stream.input_json"
    assert vendor.runtime is Runtime.ANTHROPIC_API


def test_the_raw_frames_beneath_the_helper_events_are_not_reported_twice():
    round_ = Round(
        message=FakeMessage(content=[TextBlockOut("x")]),
        events=[
            HelperEvent(type="content_block_delta", text="x"),
            HelperEvent(type="text", text="x"),
            HelperEvent(type="message_stop"),
        ],
    )
    events = drive(run_request(), Factory([round_]))
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["x"]
    assert not [e for e in events if isinstance(e, VendorEvent)]


# --- 4. the prompt: blocks and breakpoints, straight through ----------------------


def test_system_blocks_reach_the_vendor_with_their_cache_control_intact():
    system = Message(
        role=Role.SYSTEM,
        content=[
            TextBlock(text="the rubric", cache_control=CacheControl(ttl="1h")),
            TextBlock(text="today's question"),
        ],
    )
    factory = Factory([text_round("ok")])
    request = run_request(messages=(system, Message(role=Role.USER, content="go")))
    drive(request, factory)
    sent = factory.clients[0].calls[0]
    assert sent["system"] == [
        {"type": "text", "text": "the rubric", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": "today's question"},
    ]
    assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    assert sent["model"] == MODEL
    assert sent["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


def test_conversation_blocks_keep_their_order_their_roles_and_their_markers():
    messages = (
        Message(role=Role.USER, content=[TextBlock("one", cache_control=CacheControl())]),
        Message(role=Role.ASSISTANT, content="two"),
        Message(role=Role.USER, content="three"),
    )
    factory = Factory([text_round("ok")])
    drive(run_request(messages=messages), factory)
    assert factory.clients[0].calls[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "one", "cache_control": {"type": "ephemeral"}}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "two"}]},
        {"role": "user", "content": [{"type": "text", "text": "three"}]},
    ]


def test_a_run_with_no_system_message_omits_the_field_entirely():
    factory = Factory([text_round("ok")])
    drive(run_request(), factory)
    assert "system" not in factory.clients[0].calls[0]


def test_the_receipt_reports_the_breakpoints_as_honoured_on_this_runtime(store):
    conn = connection()
    system = Message(
        role=Role.SYSTEM, content=[TextBlock("rubric", cache_control=CacheControl())]
    )
    bridge, _ = bridge_for(conn, Factory([text_round("ok")]), store)
    receipt = bridge.preflight(
        conn.name, messages=[system, Message(role=Role.USER, content="go")]
    )
    assert receipt.cache_breakpoints_requested == 1
    assert receipt.cache is not None
    assert receipt.cache.breakpoints_honoured is True
    assert not any("dropped" in note for note in receipt.notes)


def test_cache_eligibility_reports_the_callers_own_ttl_rather_than_claiming_a_pin():
    system = Message(
        role=Role.SYSTEM,
        content=[TextBlock("x" * 9000, cache_control=CacheControl(ttl="1h"))],
    )
    request = run_request(messages=(system, Message(role=Role.USER, content="go")))
    eligibility = adapter(Factory()).cache_eligibility(request)
    assert eligibility.explicit_breakpoints == 1
    assert eligibility.ttl is None
    assert "the caller pinned '1h'" in eligibility.ttl_detail


# --- 5. the in-adapter tool loop --------------------------------------------------


def echo_tool(calls: list[dict[str, Any]]) -> ToolDef:
    def handler(args: dict[str, Any]) -> str:
        calls.append(args)
        return f"looked up {args.get('topic')}"

    return ToolDef(
        name="look_up",
        description="Look a topic up.",
        parameters={"type": "object", "properties": {"topic": {"type": "string"}}},
        handler=handler,
    )


def test_the_adapter_runs_the_loop_itself_and_reports_both_halves_as_observations():
    seen: list[dict[str, Any]] = []
    tool = echo_tool(seen)
    factory = Factory(
        [tool_round("call_1", "look_up", {"topic": "rebase"}), text_round("it rewrites history")]
    )
    events = drive(run_request(tools=(tool,)), factory)

    assert seen == [{"topic": "rebase"}]
    call = next(e for e in events if isinstance(e, ToolCallEvent))
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert (call.name, call.id, call.server) == ("look_up", "call_1", "caller")
    assert call.arguments == {"topic": "rebase"}
    assert (result.id, result.name, result.is_error) == ("call_1", "look_up", False)
    assert result.content == "looked up rebase"
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].status is TerminalStatus.OK


def test_the_tool_result_goes_back_as_one_user_message_after_the_assistant_turn():
    tool = echo_tool([])
    factory = Factory(
        [tool_round("call_1", "look_up", {"topic": "x"}), text_round("done")]
    )
    drive(run_request(tools=(tool,)), factory)
    second = factory.clients[0].calls[1]["messages"]
    assert second[-2] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "call_1", "name": "look_up", "input": {"topic": "x"}}
        ],
    }
    assert second[-1]["role"] == "user"
    assert second[-1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": [{"type": "text", "text": "looked up x"}],
            "is_error": False,
        }
    ]
    assert factory.clients[0].calls[1]["tools"][0]["name"] == "look_up"


def test_there_is_no_turn_cap():
    tool = echo_tool([])
    rounds = [tool_round(f"call_{i}", "look_up", {"topic": str(i)}) for i in range(6)]
    factory = Factory([*rounds, text_round("finally")])
    events = drive(run_request(tools=(tool,)), factory)
    assert len([e for e in events if isinstance(e, ToolCallEvent)]) == 6
    assert events[-1].status is TerminalStatus.OK


def test_a_raising_handler_becomes_a_failed_tool_result_and_the_run_continues():
    def explode(args: dict[str, Any]) -> str:
        raise ValueError("the index is offline")

    tool = ToolDef(
        name="look_up", description="Look a topic up.", parameters={}, handler=explode
    )
    factory = Factory(
        [tool_round("call_1", "look_up", {}), text_round("I could not look that up")]
    )
    events = drive(run_request(tools=(tool,)), factory)
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.is_error is True
    assert "tool 'look_up' failed: ValueError: the index is offline" in result.content
    assert events[-1].status is TerminalStatus.OK
    assert factory.clients[0].calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_a_tool_the_run_never_declared_is_a_failed_result_rather_than_a_crash():
    tool = echo_tool([])
    factory = Factory([tool_round("call_1", "ghost", {}), text_round("sorry")])
    events = drive(run_request(tools=(tool,)), factory)
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.is_error is True
    assert "was not declared for this run" in result.content


def test_an_async_handler_is_driven_to_completion():
    async def handler(args: dict[str, Any]) -> str:
        return "async answer"

    tool = ToolDef(name="look_up", description="Look.", parameters={}, handler=handler)
    factory = Factory([tool_round("call_1", "look_up", {}), text_round("done")])
    events = drive(run_request(tools=(tool,)), factory)
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.content == "async answer"


def test_a_handler_with_no_callable_at_all_still_answers_the_model():
    tool = ToolDef(name="look_up", description="Look.", parameters={})
    factory = Factory([tool_round("call_1", "look_up", {}), text_round("done")])
    events = drive(run_request(tools=(tool,)), factory)
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.is_error is True
    assert "has no handler" in result.content


def test_a_tool_use_stop_on_a_run_that_declared_no_tools_ends_rather_than_loops():
    """The model cannot ask for a tool nobody sent it -- but if it ever does,
    answering "not declared" forever is a loop with a bill. A run carrying no
    tools at all stops and says why."""
    factory = Factory([tool_round("call_1", "look_up", {})])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.ERROR
    assert "did not declare" in (events[-1].reason or "")


# --- 6. guards fire between tool rounds -------------------------------------------


@NEEDS_SDK
def test_a_token_guard_stops_the_loop_between_rounds(store):
    tool = echo_tool([])
    conn = connection(guards=Guards(stop_at_tokens=10))
    factory = Factory(
        [tool_round("call_1", "look_up", {"topic": "x"}), text_round("never reached")]
    )
    bridge, _ = bridge_for(conn, factory, store)
    events = list(bridge.chat(connection=conn.name, message="go", tools=[tool]))
    assert any(isinstance(e, GuardStopEvent) for e in events)
    terminal = events[-1]
    assert terminal.status is TerminalStatus.GUARD_STOP
    # The second round never happened, and neither did the tool it would have run.
    assert len(factory.clients[0].calls) == 1
    assert not [e for e in events if isinstance(e, ToolResultEvent)]


# --- 7. structured output ----------------------------------------------------------


SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string"}},
    "required": ["sentiment"],
}


def test_a_schema_is_sent_as_the_sdks_native_structured_output_config():
    factory = Factory([text_round(json.dumps({"sentiment": "positive"}))])
    events = drive(run_request(schema=SCHEMA, schema_name="Verdict"), factory)
    sent = factory.clients[0].calls[0]
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    structured = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert structured.data == {"sentiment": "positive"}
    assert structured.valid is True
    assert structured.schema_name == "Verdict"
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].status is TerminalStatus.OK


def test_the_structured_event_is_the_last_thing_before_the_terminal():
    factory = Factory([text_round(json.dumps({"sentiment": "ok"}))])
    events = drive(run_request(schema=SCHEMA), factory)
    assert isinstance(events[-2], StructuredOutputEvent)


def test_an_answer_that_is_not_json_ends_the_run_as_an_error_carrying_what_arrived():
    factory = Factory([text_round("I would rather not")])
    events = drive(run_request(schema=SCHEMA), factory)
    structured = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert structured.data is None
    assert structured.raw == "I would rather not"
    assert events[-1].status is TerminalStatus.ERROR


def test_an_answer_that_misses_the_schema_is_still_an_answer():
    factory = Factory([text_round(json.dumps({"mood": "unclear"}))])
    events = drive(run_request(schema=SCHEMA), factory)
    structured = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert structured.valid is False
    assert structured.data == {"mood": "unclear"}
    assert events[-1].status is TerminalStatus.OK


def test_a_schema_and_tools_are_never_combined_on_one_call(store):
    conn = connection()
    bridge, _ = bridge_for(conn, Factory([text_round("{}")]), store)
    with pytest.raises(Exception) as excinfo:
        list(
            bridge.chat(
                connection=conn.name,
                message="go",
                schema=SCHEMA,
                tools=[echo_tool([])],
            )
        )
    assert "schema" in str(excinfo.value).lower()


def test_no_schema_means_no_output_config_on_the_wire():
    factory = Factory([text_round("plain")])
    drive(run_request(), factory)
    assert "output_config" not in factory.clients[0].calls[0]


# --- 8. terminal status from stop_reason -------------------------------------------


@pytest.mark.parametrize(
    ("stop_reason", "status"),
    [
        ("end_turn", TerminalStatus.OK),
        ("stop_sequence", TerminalStatus.OK),
        ("max_tokens", TerminalStatus.ERROR),
        ("refusal", TerminalStatus.ERROR),
        ("pause_turn", TerminalStatus.OK),
    ],
)
def test_each_stop_reason_maps_to_its_terminal_status(stop_reason, status):
    factory = Factory([text_round("...", stop_reason=stop_reason)])
    events = drive(run_request(), factory)
    assert events[-1].status is status


def test_a_truncated_answer_says_which_ceiling_it_hit():
    factory = Factory([text_round("half a th", stop_reason="max_tokens")])
    events = drive(run_request(), factory)
    assert "max_output_tokens" in (events[-1].reason or "")


def test_a_refusal_carries_the_vendors_own_category():
    round_ = text_round("no", stop_reason="refusal")
    round_.message.stop_details = StopDetails(category="cyber")
    events = drive(run_request(), Factory([round_]))
    assert "'cyber'" in (events[-1].reason or "")


def test_a_429_is_quota_exhausted_and_is_typed_off_the_status_code():
    class RateLimited(Exception):
        status_code = 429

    factory = Factory([Round(message=FakeMessage(), raises=RateLimited("slow down"))])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED


def test_any_other_vendor_failure_is_a_reported_terminal_not_a_raise():
    class Boom(Exception):
        status_code = 500

    factory = Factory([Round(message=FakeMessage(), raises=Boom("upstream is unwell"))])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.ERROR
    assert "HTTP 500" in (events[-1].reason or "")


def test_a_max_output_tokens_option_reaches_the_request():
    factory = Factory([text_round("ok")])
    drive(run_request(options={"max_output_tokens": 128}), factory)
    assert factory.clients[0].calls[0]["max_tokens"] == 128


def test_an_unknown_option_key_is_reported_rather_than_silently_ignored():
    assert AnthropicAPIAdapter.unknown_option_keys({"max_output_tokens": 1, "temprature": 2}) == (
        "temprature",
    )


# --- 9. cancellation ---------------------------------------------------------------


def test_cancel_stops_the_loop_at_the_next_round_boundary():
    tool = echo_tool([])
    factory = Factory(
        [tool_round("call_1", "look_up", {"topic": "x"}), text_round("never reached")]
    )
    api = adapter(factory)
    events = []
    for event in api.run(run_request(tools=(tool,))):
        events.append(event)
        if isinstance(event, ToolResultEvent):
            api.cancel()
    assert events[-1].status is TerminalStatus.CANCELLED
    assert len(factory.clients[0].calls) == 1


def test_closing_the_iterator_closes_the_vendor_stream():
    factory = Factory([text_round("hello"), text_round("unused")])
    stream = adapter(factory).run(run_request())
    next(stream)
    stream.close()
    assert factory.clients[0].streams[0].closed is True


# --- 10. sessions are refused, and the refusal says what to use instead -------------


def test_the_adapter_keeps_its_own_refusal_wording(store):
    """The sentence this adapter gives, read off the adapter itself.

    Since ticket 1.8 the bridge refuses every session door on this runtime
    before an adapter is loaded (``tests/test_session_policy.py``), so nothing
    routed through ``bridge.new_chat()`` reaches these three methods any more.
    They stay, and so does this test: an adapter that answered a session call
    would be a contract violation, and the message is the one a caller sees if
    anything ever holds the adapter directly.
    """
    api = adapter(Factory())
    for call, method in (
        ("bridge.new_chat()", api.open_session),
        ("bridge.resume_chat()", api.resume_session),
        ("bridge.list_sessions()", api.list_sessions),
    ):
        with pytest.raises(AdapterNotImplemented) as excinfo:
            method(session_request())
        message = str(excinfo.value)
        assert "anthropic-api" in message
        assert call in message
        assert "bridge.chat(history=[...])" in message


def test_new_chat_is_refused_by_the_bridge_before_the_adapter(store):
    conn = connection()
    bridge, _ = bridge_for(conn, Factory(), store)
    with pytest.raises(CapabilityNotSupported) as excinfo:
        bridge.new_chat(connection=conn.name)
    message = str(excinfo.value)
    assert "anthropic-api" in message
    assert "bridge.new_chat()" in message
    assert "bridge.chat(history=[...])" in message


def test_new_worker_is_refused_because_there_is_no_native_toolbelt(store, tmp_path):
    conn = connection()
    bridge, _ = bridge_for(conn, Factory(), store)
    with pytest.raises(CapabilityNotSupported):
        bridge.new_worker(connection=conn.name, project_folder=str(tmp_path))


# --- 11. the capability row, and what each cell was read off -----------------------

MOVED = (
    Capability.CHAT,
    Capability.STREAMING,
    Capability.INCREMENTAL_TEXT,
    Capability.USAGE_TOKENS,
    Capability.INTERIM_USAGE,
    Capability.TOOLS_IN_PROCESS,
    Capability.STRUCTURED_OUTPUT,
    Capability.THINKING,
    Capability.CACHE_BREAKPOINTS,
    Capability.SYSTEM_PROMPT_REPLACE,
    Capability.GRACEFUL_CANCEL,
    Capability.API_KEY_AUTH,
)


@pytest.mark.parametrize("capability", MOVED)
def test_each_moved_cell_is_supported_and_carries_dated_evidence(capability):
    registry = CapabilityRegistry()
    assert registry.support(Runtime.ANTHROPIC_API, capability) is Support.SUPPORTED
    note = registry.note(Runtime.ANTHROPIC_API, capability) or ""
    assert "2026-09-13" in note
    assert SDK_VERSION_READ in note or "tests/" in note


def test_the_cells_that_still_need_a_live_drive_stay_unverified_and_name_the_test():
    registry = CapabilityRegistry()
    # SAMPLING_CONTROLS and MAX_OUTPUT_TOKENS left this list in ticket 1.7,
    # which is the ticket their 1.6 notes named. They did not need a *live*
    # drive to move -- the fake transport shows which fields modelpass sends and
    # the installed SDK's signature shows the vendor takes them -- which is the
    # same two-part evidence every other cell on this row moved on.
    for capability in (
        Capability.TTL_CONTROL,
        Capability.EPHEMERAL_MULTI_TURN,
        Capability.MIDCONVERSATION_SYSTEM,
    ):
        support = registry.support(Runtime.ANTHROPIC_API, capability)
        assert support is not Support.SUPPORTED
    note = registry.note(Runtime.ANTHROPIC_API, Capability.TTL_CONTROL) or ""
    assert "tests/live/test_anthropic_api_live.py" in note


def test_the_sessions_cells_stay_unsupported_because_nothing_holds_a_history():
    registry = CapabilityRegistry()
    for capability in (
        Capability.SESSIONS_RESUME,
        Capability.SESSIONS_LIST,
        Capability.SESSIONS_FORK,
        Capability.MCP_SERVERS,
        Capability.SUBAGENTS,
        Capability.SUBSCRIPTION_AUTH,
    ):
        assert registry.support(Runtime.ANTHROPIC_API, capability) is Support.UNSUPPORTED


# --- 12. the fake is checked against the SDK it stands in for ----------------------


def test_the_fake_matches_the_installed_sdk_shapes():
    """The fake is only worth anything if the real client has the same surface.

    Four things, each one a way this file could pass against an adapter that
    would fail live: the usage field names, the stop reasons the mapping
    branches on, the structured-output parameter, and the fact that
    ``messages.stream`` accepts every key this adapter sends.
    """
    anthropic = pytest.importorskip("anthropic")
    import inspect as _inspect
    import typing as _typing

    from anthropic.resources.messages import Messages
    from anthropic.types import Message as SDKMessage
    from anthropic.types import Usage as SDKUsage

    assert anthropic.__version__.startswith("0."), anthropic.__version__
    for name in FakeUsage.__dataclass_fields__:
        assert name in SDKUsage.model_fields

    annotation = SDKMessage.model_fields["stop_reason"].annotation
    reasons = set(_typing.get_args(_typing.get_args(annotation)[0]))
    assert {"end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal"} <= reasons

    accepted = set(_inspect.signature(Messages.stream).parameters)
    assert {
        "model",
        "messages",
        "system",
        "tools",
        "max_tokens",
        "output_config",
    } <= accepted
