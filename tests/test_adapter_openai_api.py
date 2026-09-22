"""The ``openai-api`` adapter, end to end, with no network anywhere (ticket 1.9).

Deliberately the same file, section by section, as
``tests/test_adapter_anthropic_api.py``. Two adapters that are meant to behave
identically should be *readable* side by side, and a case that appears in one
file and not the other is then a visible difference rather than an invisible one.

Every test here drives a **fake transport**: an object shaped like the pieces of
``openai.OpenAI`` this adapter touches (``responses.stream(...)`` and
``models.list()``), injected through the adapter's ``client_factory`` constructor
seam. Nothing here imports the vendor SDK to make a call, nothing opens a socket,
and nothing spends. The live drive that *does* spend is
``tests/live/test_openai_api_live.py``, deselected by default.

The fake is literal about the shapes the adapter reads -- a ``Response`` with
``status`` / ``incomplete_details`` / ``output`` / ``usage``, stream events under
their real ``type`` strings, function calls whose ``arguments`` are a JSON
*string* -- because a fake looser than the vendor is a test that passes against
an adapter that would not run. Each shape is pinned against the installed SDK's
own types in :func:`test_the_fake_matches_the_installed_sdk_shapes`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from modelpass.adapters import ADAPTER_REGISTRY, load_adapter
from modelpass.adapters.anthropic_api import token_usage as anthropic_token_usage
from modelpass.adapters.base import RunRequest, SessionRequest
from modelpass.adapters.openai_api import (
    SDK_VERSION_READ,
    OpenAIAPIAdapter,
    token_usage,
)
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
    Sampling,
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

KEY = "sk-not-a-real-key-0123456789"
KEY_VAR = "MODELPASS_TEST_OPENAI_KEY"
ENV = {KEY_VAR: KEY}
MODEL = "gpt-4o"


# --- the fake transport ----------------------------------------------------------


@dataclass
class FakeInputDetails:
    cached_tokens: int = 0


@dataclass
class FakeOutputDetails:
    reasoning_tokens: int = 0


@dataclass
class FakeUsage:
    """The Responses usage shape: totals, with the details objects beside them."""

    input_tokens: int = 0
    output_tokens: int = 0
    input_tokens_details: FakeInputDetails = field(default_factory=FakeInputDetails)
    output_tokens_details: FakeOutputDetails = field(default_factory=FakeOutputDetails)


@dataclass
class OutputTextPart:
    text: str
    type: str = "output_text"

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {"type": "output_text", "text": self.text}


@dataclass
class MessageItem:
    content: list[Any]
    type: str = "message"
    role: str = "assistant"

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "type": "message",
            "role": self.role,
            "content": [part.model_dump(**kwargs) for part in self.content],
        }


@dataclass
class ReasoningItem:
    summary: list[Any] = field(default_factory=list)
    type: str = "reasoning"
    id: str = "rs_1"
    encrypted_content: str | None = None

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        dumped = {
            "type": "reasoning",
            "id": self.id,
            "summary": list(self.summary),
            "encrypted_content": self.encrypted_content,
        }
        if exclude_none:
            return {k: v for k, v in dumped.items() if v is not None}
        return dumped


@dataclass
class FunctionCallItem:
    call_id: str
    name: str
    arguments: str
    type: str = "function_call"
    id: str | None = None

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        dumped = {
            "type": "function_call",
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "id": self.id,
        }
        if exclude_none:
            return {k: v for k, v in dumped.items() if v is not None}
        return dumped


@dataclass
class IncompleteDetails:
    reason: str | None = None


@dataclass
class ResponseError:
    message: str = ""


@dataclass
class FakeResponse:
    output: list[Any] = field(default_factory=list)
    status: str | None = "completed"
    usage: FakeUsage = field(default_factory=FakeUsage)
    incomplete_details: IncompleteDetails | None = None
    error: ResponseError | None = None


@dataclass
class StreamEvent:
    """One event off ``responses.stream``, under its real ``type`` string."""

    type: str
    delta: str = ""
    response: FakeResponse | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {"type": self.type, "delta": self.delta}


def terminal_event(response: FakeResponse) -> StreamEvent:
    """The stream event that carries a finished response, chosen by its status."""
    kind = {
        "incomplete": "response.incomplete",
        "failed": "response.failed",
    }.get(response.status or "", "response.completed")
    return StreamEvent(type=kind, response=response)


@dataclass
class Round:
    """One scripted response: what streams, and what the final response says."""

    response: FakeResponse
    events: list[StreamEvent] = field(default_factory=list)
    raises: Exception | None = None

    def frames(self) -> list[StreamEvent]:
        return [*self.events, terminal_event(self.response)]


class FakeStream:
    def __init__(self, round_: Round) -> None:
        self._round = round_
        self.closed = False

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def __iter__(self):
        yield from self._round.frames()


class FakeResponses:
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
        self.responses = FakeResponses(self)
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
        response=FakeResponse(
            output=[MessageItem(content=[OutputTextPart(text)])], usage=usage, **kwargs
        ),
        events=[StreamEvent(type="response.output_text.delta", delta=text)],
    )


def tool_round(call_id: str, name: str, arguments: dict[str, Any]) -> Round:
    return Round(
        response=FakeResponse(
            output=[
                FunctionCallItem(call_id=call_id, name=name, arguments=json.dumps(arguments))
            ],
            usage=FakeUsage(input_tokens=8, output_tokens=4),
        ),
        events=[],
    )


# --- connections, requests, bridges ----------------------------------------------


def connection(
    *,
    name: str = "openai-key",
    model: str | None = MODEL,
    base_url: str | None = None,
    credential_ref: str = f"env:{KEY_VAR}",
    guards: Guards | None = None,
) -> Connection:
    return Connection(
        name=name,
        runtime=Runtime.OPENAI_API,
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


def adapter(factory: Factory) -> OpenAIAPIAdapter:
    return OpenAIAPIAdapter(client_factory=factory, env=ENV)


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
) -> tuple[Bridge, OpenAIAPIAdapter]:
    store.add(conn)
    api = adapter(factory)
    return (
        Bridge(
            store=store,
            registry=CapabilityRegistry(),
            adapters={Runtime.OPENAI_API: api},
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
    not OpenAIAPIAdapter.is_available(),
    reason="needs the modelpass[openai-api] extra: openai is not installed",
)


# --- 1. registration, and a client built with an explicit key ---------------------


@NEEDS_SDK
def test_the_runtime_resolves_to_this_adapter_through_the_usual_discovery():
    entry = ADAPTER_REGISTRY[Runtime.OPENAI_API]
    assert entry.module == "modelpass.adapters.openai_api"
    assert entry.attribute == "OpenAIAPIAdapter"
    # The extra is its own, because the ``openai`` extra installs openai-codex
    # for the *agent* runtime. Two OpenAI runtimes, two packages.
    assert (entry.extra, entry.package) == ("openai-api", "openai")
    assert ADAPTER_REGISTRY[Runtime.OPENAI_SDK].package == "openai-codex"
    loaded = load_adapter(Runtime.OPENAI_API)
    assert isinstance(loaded, OpenAIAPIAdapter)
    assert loaded.runtime is Runtime.OPENAI_API


def test_the_client_is_constructed_with_the_connections_own_key():
    factory = Factory([text_round("hi")])
    drive(run_request(), factory)
    assert factory.kwargs == [{"api_key": KEY, "base_url": None}]


def test_an_ambient_key_is_never_read_even_when_one_is_set(monkeypatch):
    """The decoy that would have billed the wrong account.

    ``tests/test_no_ambient_credentials.py`` owns this rule across the library;
    it is restated here because it is the one promise of this adapter that costs
    money when it breaks, and because the adapter is handed an explicit ``env``
    that does not contain the decoy at all.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-the-ambient-key-that-must-never-be-used")
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
    secrets.set("openai-key", KEY)
    factory = Factory([text_round("hi")])
    api = OpenAIAPIAdapter(client_factory=factory, env={}, secrets=secrets)
    conn = connection(credential_ref="secret:openai-key")
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
    api = OpenAIAPIAdapter(client_factory=Factory(), env={})
    receipt = api.preflight(run_request())
    assert not receipt.ok
    assert KEY_VAR in (receipt.problem or "")


def test_a_connection_with_no_model_still_preflights_but_the_receipt_says_so():
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
    assert OpenAIAPIAdapter.is_available() is True


# --- 3. the stream, the events, and the usage mapping -----------------------------


def test_a_plain_call_streams_text_then_usage_then_a_terminal():
    factory = Factory([text_round("hello there")])
    events = drive(run_request(), factory)
    assert [type(e) for e in events] == [TextDeltaEvent, UsageEvent, TerminalEvent]
    assert events[0].text == "hello there"
    assert events[-1].status is TerminalStatus.OK


def test_text_arrives_one_delta_at_a_time_in_the_order_it_was_produced():
    round_ = Round(
        response=FakeResponse(output=[MessageItem(content=[OutputTextPart("one two")])]),
        events=[
            StreamEvent(type="response.output_text.delta", delta="one "),
            StreamEvent(type="response.output_text.delta", delta="two"),
        ],
    )
    events = drive(run_request(), Factory([round_]))
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["one ", "two"]


def test_reasoning_deltas_become_thinking_events():
    """Both spellings the Responses stream has for reasoning text."""
    round_ = Round(
        response=FakeResponse(output=[MessageItem(content=[OutputTextPart("yes")])]),
        events=[
            StreamEvent(type="response.reasoning_summary_text.delta", delta="weighing it"),
            StreamEvent(type="response.reasoning_text.delta", delta=" further"),
            StreamEvent(type="response.output_text.delta", delta="yes"),
        ],
    )
    events = drive(run_request(), Factory([round_]))
    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    assert [e.text for e in thinking] == ["weighing it", " further"]


def test_a_reasoning_summary_is_only_asked_for_when_the_caller_asks():
    """modelpass does not order billed output nobody requested."""
    factory = Factory([text_round("ok")])
    drive(run_request(sampling=Sampling(reasoning_effort="high"), model="o3-mini"), factory)
    assert factory.clients[0].calls[0]["reasoning"] == {"effort": "high"}

    factory = Factory([text_round("ok")])
    drive(
        run_request(
            sampling=Sampling(reasoning_effort="high"),
            model="o3-mini",
            options={"reasoning_summary": "auto"},
        ),
        factory,
    )
    assert factory.clients[0].calls[0]["reasoning"] == {
        "effort": "high",
        "summary": "auto",
    }


def test_usage_puts_openais_cached_tokens_where_modelpass_keeps_them():
    """The subtraction, against the anthropic mapping of the same real totals.

    OpenAI reports 100 prompt tokens *including* 30 cached; Anthropic reports the
    same call as 70 fresh and 30 read. A consumer's accounting must not be able
    to tell which runtime produced the number, so both must land on the same
    ``TokenUsage``.
    """
    usage = FakeUsage(
        input_tokens=100,
        output_tokens=40,
        input_tokens_details=FakeInputDetails(cached_tokens=30),
        output_tokens_details=FakeOutputDetails(reasoning_tokens=12),
    )
    mapped = token_usage(usage)
    assert mapped == TokenUsage(
        input_tokens=70,
        output_tokens=40,
        cached_input_tokens=30,
        cache_write_tokens=0,
        reasoning_output_tokens=12,
    )
    # ``anthropic-api`` reports no reasoning count at all -- ``anthropic``
    # 0.97.0's ``Usage`` has no details object -- so the parity these two
    # mappings owe each other is on the four counts both vendors report. The
    # fifth is asserted against the ``None`` that says this vendor cannot tell
    # us, rather than made to match by inventing a number.
    from dataclasses import replace

    messages_api = anthropic_token_usage(
        {
            "input_tokens": 70,
            "output_tokens": 40,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 0,
        }
    )
    assert messages_api.reasoning_output_tokens is None
    assert replace(mapped, reasoning_output_tokens=None) == messages_api
    # Reasoning tokens stay *inside* output_tokens -- recorded as a subset, not
    # moved out of it. Subtracting them would make one runtime's output tokens
    # mean something different from another's, and adding them to the total
    # would bill the same tokens twice.
    assert mapped.output_tokens == 40
    assert mapped.total_tokens == 140
    assert mapped.reasoning_output_tokens == 12


def test_usage_reads_a_mapping_as_well_as_a_model_and_never_raises():
    assert token_usage(None) == TokenUsage()
    assert token_usage({"input_tokens": 5, "output_tokens": 6}) == TokenUsage(
        input_tokens=5, output_tokens=6
    )
    assert token_usage(
        {"input_tokens": 9, "input_tokens_details": {"cached_tokens": 4}}
    ) == TokenUsage(input_tokens=5, cached_input_tokens=4)
    # A vendor that ever reported more cached than total must not produce a
    # negative count on a receipt.
    assert token_usage(
        {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 4}}
    ).input_tokens == 0


def test_the_usage_event_carries_the_cache_split_through_the_stream():
    usage = FakeUsage(
        input_tokens=705,
        output_tokens=6,
        input_tokens_details=FakeInputDetails(cached_tokens=700),
    )
    events = drive(run_request(), Factory([text_round("hi", usage=usage)]))
    reported = next(e for e in events if isinstance(e, UsageEvent))
    assert reported.usage.cached_input_tokens == 700
    assert reported.usage.input_tokens == 5
    assert reported.usage.cache_write_tokens == 0


def test_a_stream_event_with_no_word_in_the_vocabulary_becomes_a_vendor_event():
    round_ = Round(
        response=FakeResponse(output=[MessageItem(content=[OutputTextPart("x")])]),
        events=[
            StreamEvent(type="response.function_call_arguments.delta", delta="{"),
            StreamEvent(type="response.output_text.delta", delta="x"),
        ],
    )
    events = drive(run_request(), Factory([round_]))
    vendor = next(e for e in events if isinstance(e, VendorEvent))
    assert vendor.name == "stream.response.function_call_arguments.delta"
    assert vendor.runtime is Runtime.OPENAI_API


def test_the_frames_beneath_the_deltas_are_not_reported_twice():
    round_ = Round(
        response=FakeResponse(output=[MessageItem(content=[OutputTextPart("x")])]),
        events=[
            StreamEvent(type="response.created"),
            StreamEvent(type="response.output_text.delta", delta="x"),
            StreamEvent(type="response.output_text.done", delta="x"),
        ],
    )
    events = drive(run_request(), Factory([round_]))
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["x"]
    assert not [e for e in events if isinstance(e, VendorEvent)]


# --- 4. the prompt: flattened, with the drop named by the bridge -------------------


def test_the_system_prompt_becomes_instructions_and_blocks_are_flattened():
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
    # flat_text semantics: one blank line between blocks, and no marker anywhere.
    assert sent["instructions"] == "the rubric\n\ntoday's question"
    assert "cache_control" not in json.dumps(sent)
    assert sent["input"] == [{"role": "user", "content": "go"}]
    assert sent["model"] == MODEL
    # No ceiling is invented: max_output_tokens is optional on this API.
    assert "max_output_tokens" not in sent


def test_conversation_turns_keep_their_order_and_their_roles():
    messages = (
        Message(role=Role.USER, content=[TextBlock("one"), TextBlock("still one")]),
        Message(role=Role.ASSISTANT, content="two"),
        Message(role=Role.USER, content="three"),
    )
    factory = Factory([text_round("ok")])
    drive(run_request(messages=messages), factory)
    assert factory.clients[0].calls[0]["input"] == [
        {"role": "user", "content": "one\n\nstill one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]


def test_a_run_with_no_system_message_omits_the_field_entirely():
    factory = Factory([text_round("ok")])
    drive(run_request(), factory)
    assert "instructions" not in factory.clients[0].calls[0]


def test_the_receipt_names_the_dropped_breakpoints_on_this_runtime(store):
    """R3, ticket 1.5: a dropped breakpoint is a named line, not a silence."""
    conn = connection()
    system = Message(
        role=Role.SYSTEM, content=[TextBlock("rubric", cache_control=CacheControl())]
    )
    bridge, _ = bridge_for(conn, Factory([text_round("ok")]), store)
    receipt = bridge.preflight(
        conn.name, messages=[system, Message(role=Role.USER, content="go")]
    )
    assert receipt.cache_breakpoints_requested == 1
    assert receipt.cache_breakpoints_honoured == 0
    assert any("dropped" in note for note in receipt.notes)


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
    # The arguments arrive as a JSON string on this runtime and are parsed here.
    assert call.arguments == {"topic": "rebase"}
    assert (result.id, result.name, result.is_error) == ("call_1", "look_up", False)
    assert result.content == "looked up rebase"
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].status is TerminalStatus.OK


def test_the_whole_output_turn_is_replayed_before_the_function_call_output():
    """Reasoning items included -- a reasoning model that cannot see its own
    previous thinking re-derives it every round."""
    tool = echo_tool([])
    first = tool_round("call_1", "look_up", {"topic": "x"})
    first.response.output.insert(0, ReasoningItem(id="rs_7"))
    factory = Factory([first, text_round("done")])
    drive(run_request(tools=(tool,)), factory)
    second = factory.clients[0].calls[1]["input"]
    assert second[-3] == {"type": "reasoning", "id": "rs_7", "summary": []}
    assert second[-2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "look_up",
        "arguments": json.dumps({"topic": "x"}),
    }
    assert second[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "looked up x",
    }
    tool_param = factory.clients[0].calls[1]["tools"][0]
    assert tool_param["type"] == "function"
    assert tool_param["name"] == "look_up"
    assert tool_param["strict"] is False


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
    answered = factory.clients[0].calls[1]["input"][-1]
    assert "the index is offline" in answered["output"]


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


def test_unparseable_call_arguments_reach_the_handler_as_an_empty_mapping():
    """A dead run is worse than a tool that can say what it was given."""
    seen: list[dict[str, Any]] = []
    tool = echo_tool(seen)
    broken = Round(
        response=FakeResponse(
            output=[FunctionCallItem(call_id="call_1", name="look_up", arguments="{not json")]
        ),
    )
    factory = Factory([broken, text_round("done")])
    events = drive(run_request(tools=(tool,)), factory)
    assert seen == [{}]
    assert events[-1].status is TerminalStatus.OK


def test_a_function_call_on_a_run_that_declared_no_tools_ends_rather_than_loops():
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
    assert events[-1].status is TerminalStatus.GUARD_STOP
    # The second round never happened, and neither did the tool it would have run.
    assert len(factory.clients[0].calls) == 1
    assert not [e for e in events if isinstance(e, ToolResultEvent)]


# --- 7. structured output ----------------------------------------------------------


SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string"}},
    "required": ["sentiment"],
}

LOOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["sentiment"],
}


def test_a_schema_is_sent_as_a_strict_json_schema_text_format():
    factory = Factory([text_round(json.dumps({"sentiment": "positive"}))])
    events = drive(run_request(schema=SCHEMA, schema_name="Verdict"), factory)
    sent = factory.clients[0].calls[0]
    assert sent["text"] == {
        "format": {
            "type": "json_schema",
            "name": "Verdict",
            "schema": {
                "type": "object",
                "properties": {"sentiment": {"type": "string"}},
                "required": ["sentiment"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    }
    structured = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert structured.data == {"sentiment": "positive"}
    assert structured.valid is True
    assert structured.schema_name == "Verdict"
    assert events[-1].status is TerminalStatus.OK


def test_an_optional_property_is_tightened_and_the_receipt_says_what_changed():
    """The strict subset has no optional properties, so the rewrite is real and
    is disclosed *before* the call rather than discovered in the answer."""
    factory = Factory([text_round(json.dumps({"sentiment": "ok", "note": None}))])
    request = run_request(schema=LOOSE_SCHEMA, schema_name="Verdict")
    receipt = adapter(factory).preflight(request)
    assert receipt.ok
    note = next(n for n in receipt.notes if "to_openai_strict" in n)
    assert "note" in note

    drive(request, factory)
    sent_schema = factory.clients[0].calls[0]["text"]["format"]["schema"]
    assert sent_schema["required"] == ["sentiment", "note"]
    assert sent_schema["properties"]["note"]["type"] == ["string", "null"]


def test_a_schema_that_is_already_strict_adds_no_note():
    factory = Factory()
    receipt = adapter(factory).preflight(
        run_request(
            schema={
                "type": "object",
                "properties": {"sentiment": {"type": "string"}},
                "required": ["sentiment"],
                "additionalProperties": False,
            }
        )
    )
    assert not any("to_openai_strict" in note for note in receipt.notes)


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


def test_the_structured_event_is_the_last_thing_before_the_terminal():
    factory = Factory([text_round(json.dumps({"sentiment": "ok"}))])
    events = drive(run_request(schema=SCHEMA), factory)
    assert isinstance(events[-2], StructuredOutputEvent)


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


def test_no_schema_means_no_text_format_on_the_wire():
    factory = Factory([text_round("plain")])
    drive(run_request(), factory)
    assert "text" not in factory.clients[0].calls[0]


# --- 8. terminal status from status and incomplete_details -------------------------


@pytest.mark.parametrize(
    ("status", "reason", "expected"),
    [
        ("completed", None, TerminalStatus.OK),
        ("incomplete", "max_output_tokens", TerminalStatus.ERROR),
        ("incomplete", "content_filter", TerminalStatus.ERROR),
        ("failed", None, TerminalStatus.ERROR),
        ("cancelled", None, TerminalStatus.CANCELLED),
        ("in_progress", None, TerminalStatus.OK),
    ],
)
def test_each_status_maps_to_its_terminal_status(status, reason, expected):
    round_ = text_round("...", status=status)
    if reason is not None:
        round_.response.incomplete_details = IncompleteDetails(reason=reason)
    events = drive(run_request(), Factory([round_]))
    assert events[-1].status is expected


def test_a_truncated_answer_says_which_ceiling_it_hit():
    """Word for word anthropic-api's sentence, because the fix is the same one."""
    round_ = text_round("half a th", status="incomplete")
    round_.response.incomplete_details = IncompleteDetails(reason="max_output_tokens")
    events = drive(run_request(), Factory([round_]))
    assert "max_output_tokens" in (events[-1].reason or "")


def test_a_content_filter_stop_is_an_error_and_names_the_filter():
    round_ = text_round("", status="incomplete")
    round_.response.incomplete_details = IncompleteDetails(reason="content_filter")
    events = drive(run_request(), Factory([round_]))
    assert "content filter" in (events[-1].reason or "")


def test_a_failed_response_carries_the_vendors_own_message():
    round_ = text_round("", status="failed")
    round_.response.error = ResponseError(message="the model is overloaded")
    events = drive(run_request(), Factory([round_]))
    assert "the model is overloaded" in (events[-1].reason or "")


def test_a_stream_that_never_delivered_a_response_is_an_error_not_an_empty_ok():
    """``get_final_response()`` raises here; reading the terminal event does not,
    and an answer that never arrived must not be reported as ``ok``."""
    silent = Round(response=FakeResponse(), events=[])
    silent.frames = lambda: []  # type: ignore[method-assign]
    events = drive(run_request(), Factory([silent]))
    assert events[-1].status is TerminalStatus.ERROR
    assert "no answer arrived" in (events[-1].reason or "")


def test_a_429_is_quota_exhausted_and_is_typed_off_the_status_code():
    class RateLimited(Exception):
        status_code = 429

    factory = Factory([Round(response=FakeResponse(), raises=RateLimited("slow down"))])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED


def test_any_other_vendor_failure_is_a_reported_terminal_not_a_raise():
    class Boom(Exception):
        status_code = 500

    factory = Factory([Round(response=FakeResponse(), raises=Boom("upstream is unwell"))])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.ERROR
    assert "HTTP 500" in (events[-1].reason or "")


# --- 9. sampling -------------------------------------------------------------------


def test_the_three_accepted_controls_reach_the_request_under_their_own_names():
    factory = Factory([text_round("ok")])
    drive(
        run_request(
            sampling=Sampling(temperature=0.2, top_p=0.9, max_output_tokens=128),
            model="gpt-4o",
        ),
        factory,
    )
    sent = factory.clients[0].calls[0]
    assert sent["temperature"] == 0.2
    assert sent["top_p"] == 0.9
    assert sent["max_output_tokens"] == 128


def test_reasoning_effort_becomes_the_nested_reasoning_object():
    factory = Factory([text_round("ok")])
    drive(run_request(sampling=Sampling(reasoning_effort="low"), model="o3-mini"), factory)
    assert factory.clients[0].calls[0]["reasoning"] == {"effort": "low"}


def test_top_k_never_reaches_this_runtime_and_the_drop_is_named(store):
    conn = connection()
    bridge, _ = bridge_for(conn, Factory([text_round("ok")]), store)
    receipt = bridge.preflight(conn.name, sampling=Sampling(top_k=40))
    assert "top_k" not in receipt.sampling_applied
    assert any("top_k" in note for note in receipt.sampling_notes)


def test_gpt_5_gets_the_only_temperature_it_accepts_and_the_receipt_says_so(store):
    conn = connection(model="gpt-5")
    bridge, _ = bridge_for(conn, Factory([text_round("ok")]), store)
    receipt = bridge.preflight(conn.name, sampling=Sampling(temperature=0.2))
    assert receipt.sampling_applied["temperature"] == 1.0
    assert any("accepts only 1.0" in note for note in receipt.sampling_notes)


def test_no_ceiling_is_invented_when_nobody_asked_for_one():
    """The difference from anthropic-api, and it is deliberate."""
    factory = Factory([text_round("ok")])
    drive(run_request(), factory)
    assert "max_output_tokens" not in factory.clients[0].calls[0]


def test_the_max_output_tokens_option_is_still_read_as_an_alias():
    factory = Factory([text_round("ok")])
    drive(run_request(options={"max_output_tokens": 128}), factory)
    assert factory.clients[0].calls[0]["max_output_tokens"] == 128


def test_an_unknown_option_key_is_reported_rather_than_silently_ignored():
    assert OpenAIAPIAdapter.unknown_option_keys(
        {"reasoning_summary": "auto", "temprature": 2}
    ) == ("temprature",)


# --- 10. cancellation ---------------------------------------------------------------


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


# --- 11. sessions are refused, and the refusal says what to use instead -------------


def test_the_adapter_keeps_its_own_refusal_wording(store):
    """Since ticket 1.8 the bridge refuses every session door on this runtime
    before an adapter is loaded, so nothing routed through ``bridge.new_chat()``
    reaches these three methods. They stay, and so does this test: an adapter
    that answered a session call would be a contract violation."""
    api = adapter(Factory())
    for call, method in (
        ("bridge.new_chat()", api.open_session),
        ("bridge.resume_chat()", api.resume_session),
        ("bridge.list_sessions()", api.list_sessions),
    ):
        with pytest.raises(AdapterNotImplemented) as excinfo:
            method(session_request())
        message = str(excinfo.value)
        assert "openai-api" in message
        assert call in message
        assert "bridge.chat(history=[...])" in message


def test_new_chat_is_refused_by_the_bridge_before_the_adapter(store):
    conn = connection()
    bridge, _ = bridge_for(conn, Factory(), store)
    with pytest.raises(CapabilityNotSupported) as excinfo:
        bridge.new_chat(connection=conn.name)
    message = str(excinfo.value)
    assert "openai-api" in message
    assert "bridge.new_chat()" in message
    assert "bridge.chat(history=[...])" in message


def test_new_worker_is_refused_because_there_is_no_native_toolbelt(store, tmp_path):
    conn = connection()
    bridge, _ = bridge_for(conn, Factory(), store)
    with pytest.raises(CapabilityNotSupported):
        bridge.new_worker(connection=conn.name, project_folder=str(tmp_path))


# --- 12. the capability row, and what each cell was read off -----------------------

MOVED = (
    Capability.CHAT,
    Capability.STREAMING,
    Capability.INCREMENTAL_TEXT,
    Capability.USAGE_TOKENS,
    Capability.INTERIM_USAGE,
    Capability.TOOLS_IN_PROCESS,
    Capability.STRUCTURED_OUTPUT,
    Capability.SYSTEM_PROMPT_REPLACE,
    Capability.GRACEFUL_CANCEL,
    Capability.SAMPLING_CONTROLS,
    Capability.MAX_OUTPUT_TOKENS,
    Capability.API_KEY_AUTH,
)


@pytest.mark.parametrize("capability", MOVED)
def test_each_moved_cell_is_supported_and_carries_dated_evidence(capability):
    registry = CapabilityRegistry()
    assert registry.support(Runtime.OPENAI_API, capability) is Support.SUPPORTED
    note = registry.note(Runtime.OPENAI_API, capability) or ""
    assert "2026-09-13" in note
    assert SDK_VERSION_READ in note or "tests/" in note


def test_thinking_waits_on_a_live_drive_and_its_note_names_the_test():
    """The one cell that moved on anthropic-api and does not move here.

    The adapter maps both reasoning deltas (proved above); what a fake cannot say
    is whether anything ever arrives, because a summary is only sent when asked
    for and raw reasoning text is gated on the organisation.
    """
    registry = CapabilityRegistry()
    assert registry.support(Runtime.OPENAI_API, Capability.THINKING) is Support.UNVERIFIED
    note = registry.note(Runtime.OPENAI_API, Capability.THINKING) or ""
    assert "tests/live/test_openai_api_live.py" in note


def test_the_cells_that_are_checked_absences_stay_that_way():
    registry = CapabilityRegistry()
    for capability in (
        Capability.CACHE_BREAKPOINTS,
        Capability.TTL_CONTROL,
        Capability.SESSIONS_RESUME,
        Capability.SESSIONS_LIST,
        Capability.SESSIONS_FORK,
        Capability.MCP_SERVERS,
        Capability.SUBAGENTS,
        Capability.SUBSCRIPTION_AUTH,
    ):
        assert registry.support(Runtime.OPENAI_API, capability) is Support.UNSUPPORTED


def test_the_sampling_rules_row_is_now_a_driven_one():
    from modelpass.sampling_rules import rules_for

    rules = rules_for(Runtime.OPENAI_API, MODEL)
    assert rules.verified is True
    assert "top_k" not in rules.accepted
    # No required field and no default: the ceiling is optional on this API.
    assert rules.required == frozenset()
    assert rules.defaults == {}
    assert rules_for(Runtime.OPENAI_API, "o3-mini").reasoning_parameter == "reasoning.effort"


# --- 13. the fake is checked against the SDK it stands in for ----------------------


def test_the_fake_matches_the_installed_sdk_shapes():
    """The fake is only worth anything if the real client has the same surface.

    Six things, each one a way this file could pass against an adapter that would
    fail live: the usage field names and their nesting, the response statuses and
    the incomplete reasons the mapping branches on, the stream event type strings
    it matches, the structured-output parameter, and the fact that
    ``responses.stream`` accepts every key this adapter sends.
    """
    openai = pytest.importorskip("openai")
    import inspect as _inspect
    import typing as _typing
    import warnings

    from openai.resources.responses import Responses
    from openai.types.responses import Response
    from openai.types.responses.response import IncompleteDetails as SDKIncomplete
    from openai.types.responses.response_reasoning_summary_text_delta_event import (
        ResponseReasoningSummaryTextDeltaEvent,
    )
    from openai.types.responses.response_reasoning_text_delta_event import (
        ResponseReasoningTextDeltaEvent,
    )
    from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
    from openai.types.responses.response_usage import (
        InputTokensDetails,
        OutputTokensDetails,
        ResponseUsage,
    )

    # The installed version is *reported*, never required. The shape assertions
    # below are the protection: each one names a field, a literal or a
    # parameter this adapter depends on, so a surface that really moved fails on
    # its own line and says which. Pinning equality here instead failed on every
    # vendor patch release inside the range this project declares it supports --
    # a red suite for a reason that has nothing to do with the code, and a check
    # that teaches its reader to ignore it. openai_api.SDK_VERSION_READ stays the
    # dated evidence the capability notes cite; this warning asks for a re-read.
    installed = openai.__version__
    if installed != SDK_VERSION_READ:
        warnings.warn(
            f"openai {installed} is installed; the shapes asserted here were last "
            f"verified against {SDK_VERSION_READ}. Re-read the surface and move "
            f"openai_api.SDK_VERSION_READ when you have.",
            stacklevel=1,
        )

    # 1. usage, including the nesting the subtraction depends on.
    assert {"input_tokens", "output_tokens"} <= set(ResponseUsage.model_fields)
    assert "cached_tokens" in InputTokensDetails.model_fields
    assert "reasoning_tokens" in OutputTokensDetails.model_fields

    # 2. the statuses terminal_status() branches on.
    status_annotation = Response.model_fields["status"].annotation
    statuses = set(_typing.get_args(_typing.get_args(status_annotation)[0]))
    assert {"completed", "incomplete", "failed", "cancelled", "in_progress"} <= statuses

    # 3. the two incomplete reasons.
    reasons = set(
        _typing.get_args(_typing.get_args(SDKIncomplete.model_fields["reason"].annotation)[0])
    )
    assert {"max_output_tokens", "content_filter"} <= reasons

    # 4. the stream event type strings the mapping matches on.
    def literal(cls) -> str:
        return _typing.get_args(cls.model_fields["type"].annotation)[0]

    assert literal(ResponseTextDeltaEvent) == "response.output_text.delta"
    assert literal(ResponseReasoningSummaryTextDeltaEvent) == (
        "response.reasoning_summary_text.delta"
    )
    assert literal(ResponseReasoningTextDeltaEvent) == "response.reasoning_text.delta"

    # 5 and 6. every key this adapter sends is a parameter of responses.stream.
    accepted = set(_inspect.signature(Responses.stream).parameters)
    assert {
        "model",
        "input",
        "instructions",
        "tools",
        "text",
        "reasoning",
        "temperature",
        "top_p",
        "max_output_tokens",
    } <= accepted


def test_chat_completions_could_not_have_carried_the_thinking_word():
    """The recorded reason for choosing Responses, kept as an assertion.

    ``chat.completions`` reports reasoning only as a token *count*; there is no
    reasoning text anywhere in its response shape, so a ThinkingEvent on that
    transport could only have been invented. If this ever stops being true, the
    transport choice recorded in docs/api-and-runtimes.md §2.0b is worth
    re-reading.
    """
    pytest.importorskip("openai")
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionTokensDetails

    assert "reasoning_tokens" in CompletionTokensDetails.model_fields
    assert not [
        name for name in ChatCompletionMessage.model_fields if "reasoning" in name
    ]
