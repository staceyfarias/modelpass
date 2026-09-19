"""The ``openai-compatible`` adapter, end to end, with no network anywhere
(ticket 1.10).

Deliberately the same file, section by section, as
``tests/test_adapter_openai_api.py`` -- which is itself the same file as
``tests/test_adapter_anthropic_api.py`` -- with two sections the other two do not
have, because this runtime has two things they do not: a connection that may
carry no credential at all, and a capability row that moves per *install*.

Every test drives a **fake transport**: an object shaped like the pieces of
``openai.OpenAI`` this adapter touches -- ``chat.completions.create(...)``
returning an iterable of chunks, ``models.list()``, and ``responses.stream(...)``
for the one test that selects the other wire -- injected through the adapter's
``client_factory`` seam. Nothing opens a socket and nothing spends. The live
drive that does is ``tests/live/test_openai_compatible_live.py``, deselected by
default.

The fake is literal about the Chat Completions streaming shape, because a fake
looser than the wire is a test that passes against an adapter that would not run:
chunks carry ``choices[0].delta``, tool calls arrive as *fragments* keyed by
``index`` with the arguments JSON split across several of them, ``finish_reason``
lands on the last chunk that has a choice, and usage arrives on a final chunk
with **no** choices at all. That last one is the shape
``stream_options={"include_usage": True}`` produces and it is where a naive
``chunk.choices[0]`` crashes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

import test_adapter_openai_api as _oa
from modelpass.adapters import ADAPTER_REGISTRY, load_adapter
from modelpass.adapters.base import RunRequest, SessionRequest
from modelpass.adapters.openai_api import OpenAIAPIAdapter
from modelpass.adapters.openai_compatible import (
    PLACEHOLDER_CREDENTIAL,
    OpenAICompatibleAdapter,
    call_arguments,
    terminal_status,
    token_usage,
)
from modelpass.bridge import Bridge
from modelpass.capabilities import (
    VERIFY_CELLS,
    Capability,
    CapabilityRegistry,
    Support,
    VerifiedCapabilities,
)
from modelpass.cli import main as cli_main
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.errors import (
    AdapterNotImplemented,
    CapabilityNotSupported,
    InvalidConnection,
    PreflightFailed,
    StructuredOutputRejected,
)
from modelpass.preflight import credential_fingerprint, plan_launch
from modelpass.runtimes import Runtime
from modelpass.tools import ToolDef
from modelpass.types import (
    AuthMode,
    GuardStopEvent,
    Message,
    Role,
    Sampling,
    SessionKind,
    StructuredOutputEvent,
    TerminalStatus,
    TextBlock,
    TextDeltaEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    VendorEvent,
)

KEY = "sk-not-a-real-key-0123456789"
KEY_VAR = "MODELPASS_TEST_COMPATIBLE_KEY"
ENV = {KEY_VAR: KEY}
MODEL = "llama3.1"
BASE_URL = "http://localhost:11434/v1"


# --- the fake transport ----------------------------------------------------------


@dataclass
class FakePromptDetails:
    cached_tokens: int = 0


@dataclass
class FakeUsage:
    """The Chat Completions usage shape: two totals and a details object."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: FakePromptDetails = field(default_factory=FakePromptDetails)


@dataclass
class FakeFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class FakeToolCallDelta:
    index: int
    id: str | None = None
    type: str | None = "function"
    function: FakeFunction = field(default_factory=FakeFunction)


@dataclass
class FakeDelta:
    content: str | None = None
    role: str | None = None
    tool_calls: list[FakeToolCallDelta] | None = None


@dataclass
class FakeChoice:
    delta: FakeDelta = field(default_factory=FakeDelta)
    finish_reason: str | None = None
    index: int = 0


@dataclass
class FakeChunk:
    choices: list[FakeChoice] = field(default_factory=list)
    usage: FakeUsage | None = None
    id: str = "chatcmpl-1"

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {"id": self.id, "choices": len(self.choices)}


@dataclass
class Round:
    """One scripted response: the chunks that stream, or the raise instead."""

    chunks: list[FakeChunk] = field(default_factory=list)
    raises: Exception | None = None


class FakeStream:
    """An iterable with a ``close()``, which is what the SDK hands back."""

    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class FakeCompletions:
    def __init__(self, client: FakeClient) -> None:
        self._client = client

    def create(self, **params: Any) -> FakeStream:
        self._client.calls.append(params)
        if not self._client.rounds:
            raise AssertionError("the adapter asked for more rounds than were scripted")
        round_ = self._client.rounds.pop(0)
        if round_.raises is not None:
            raise round_.raises
        stream = FakeStream(round_.chunks)
        self._client.streams.append(stream)
        return stream


class FakeChat:
    def __init__(self, client: FakeClient) -> None:
        self.completions = FakeCompletions(client)


class FakeModels:
    def __init__(self, client: FakeClient) -> None:
        self._client = client

    def list(self, **kwargs: Any) -> Any:
        self._client.model_list_calls += 1
        if isinstance(self._client.models_answer, Exception):
            raise self._client.models_answer
        return self._client.models_answer


class FakeResponses:
    """The Responses surface, borrowed whole from ticket 1.9's own test file.

    Imported rather than rewritten, and that is the point being made: the
    ``wire="responses"`` path is not a second implementation of Responses here,
    it is 1.9's adapter running against this connection's base URL, so the fake
    that pins 1.9's shapes is the right fake to drive it with.
    """

    def __init__(self, client: FakeClient) -> None:
        self._client = client
        self._responses = _oa.FakeResponses(_oa.FakeClient(client.responses_rounds))

    def stream(self, **params: Any) -> Any:
        self._client.responses_calls.append(params)
        return self._responses.stream(**params)


class FakeClient:
    def __init__(self, rounds: list[Round], models_answer: Any = None) -> None:
        # The list is **shared** with the factory, not copied: ``verify``
        # constructs a client per drive, and the rounds must be consumed in one
        # order across all of them rather than restarting for each.
        self.rounds = rounds
        self.calls: list[dict[str, Any]] = []
        self.responses_calls: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.models_answer = models_answer
        self.model_list_calls = 0
        self.responses_rounds: list[Any] = [_oa.text_round("from the responses wire")]
        self.chat = FakeChat(self)
        self.models = FakeModels(self)
        self.responses = FakeResponses(self)


class Factory:
    """Records how the client was constructed -- the point of sections 1 and 2."""

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


@dataclass
class ModelRow:
    id: str


@dataclass
class ModelListing:
    data: list[ModelRow]


def listing(*ids: str) -> ModelListing:
    return ModelListing(data=[ModelRow(model_id) for model_id in ids])


def text_round(
    text: str,
    *,
    finish_reason: str | None = "stop",
    usage: FakeUsage | None = None,
    split: int = 1,
) -> Round:
    """Text streamed in ``split`` deltas, then a finish, then usage."""
    size = max(len(text) // split, 1)
    pieces = [text[index : index + size] for index in range(0, len(text), size)] or [""]
    chunks = [FakeChunk(choices=[FakeChoice(delta=FakeDelta(content=piece))]) for piece in pieces]
    chunks.append(FakeChunk(choices=[FakeChoice(delta=FakeDelta(), finish_reason=finish_reason)]))
    if usage is None:
        usage = FakeUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunks.append(FakeChunk(choices=[], usage=usage))
    return Round(chunks=chunks)


def silent_round(finish_reason: str | None = "stop") -> Round:
    """A round with a terminal and nothing else -- no text, no usage."""
    return Round(
        chunks=[FakeChunk(choices=[FakeChoice(delta=FakeDelta(), finish_reason=finish_reason)])]
    )


def tool_round(call_id: str, name: str, arguments: dict[str, Any]) -> Round:
    """A tool call delivered the way the wire delivers one: in fragments.

    The first fragment carries the id and the name with empty arguments; the
    JSON is then split across two more. Assembling that correctly is the whole
    of :class:`~modelpass.adapters.openai_compatible._CallBuffer`.
    """
    raw = json.dumps(arguments)
    half = len(raw) // 2
    return Round(
        chunks=[
            FakeChunk(
                choices=[
                    FakeChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=0,
                                    id=call_id,
                                    function=FakeFunction(name=name, arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(
                choices=[
                    FakeChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=0, function=FakeFunction(arguments=raw[:half])
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(
                choices=[
                    FakeChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=0, function=FakeFunction(arguments=raw[half:])
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(
                choices=[FakeChoice(delta=FakeDelta(), finish_reason="tool_calls")]
            ),
            FakeChunk(choices=[], usage=FakeUsage(prompt_tokens=8, completion_tokens=4)),
        ]
    )


class Status400(Exception):
    """A vendor 4xx, typed the way the SDK types one."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


# --- connections, requests, bridges ----------------------------------------------


def connection(
    *,
    name: str = "ollama",
    model: str | None = MODEL,
    base_url: str | None = BASE_URL,
    credential_ref: str = f"env:{KEY_VAR}",
    guards: Guards | None = None,
    verified: VerifiedCapabilities | None = None,
) -> Connection:
    return Connection(
        name=name,
        runtime=Runtime.OPENAI_COMPATIBLE,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(credential_ref),
        model=model,
        base_url=base_url,
        guards=guards or Guards(),
        verified_capabilities=verified or VerifiedCapabilities(),
    )


#: Everything a verify drive can prove, as a connection already carrying it.
#: Most tests need it because ``bridge.chat()`` refuses an unverified endpoint,
#: which is section 9's subject and every other section's precondition.
def all_verified() -> VerifiedCapabilities:
    return VerifiedCapabilities(
        supported=tuple(str(cell) for cell in VERIFY_CELLS),
        checked_at="2026-09-13T00:00:00+00:00",
        models=(MODEL,),
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


def adapter(factory: Factory) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(client_factory=factory, env=ENV)


def session_request(conn: Connection | None = None) -> SessionRequest:
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
) -> tuple[Bridge, OpenAICompatibleAdapter]:
    store.add(conn)
    api = adapter(factory)
    return (
        Bridge(
            store=store,
            registry=CapabilityRegistry(),
            adapters={Runtime.OPENAI_COMPATIBLE: api},
            env=dict(ENV),
        ),
        api,
    )


def params_of(factory: Factory, index: int = 0) -> dict[str, Any]:
    return factory.clients[0].calls[index]


# The handful of cases below that need the vendor package actually installed:
# they load the adapter through the registry, read ``runtime_available``, or
# drive a :class:`Bridge`, which refuses a runtime whose SDK does not import.
# Everything else in this file runs against the fake transport and needs
# nothing, which is D2: the suite is green on a machine with no extras.
NEEDS_SDK = pytest.mark.skipif(
    not OpenAICompatibleAdapter.is_available(),
    reason="needs the modelpass[openai-api] extra: openai is not installed",
)


# --- 1. registration, and a client built with an explicit key ---------------------


@NEEDS_SDK
def test_the_runtime_resolves_to_this_adapter_through_the_usual_discovery():
    entry = ADAPTER_REGISTRY[Runtime.OPENAI_COMPATIBLE]
    assert entry.module == "modelpass.adapters.openai_compatible"
    assert entry.attribute == "OpenAICompatibleAdapter"
    # The same extra as openai-api: it is the same wheel, pointed elsewhere.
    assert (entry.extra, entry.package) == ("openai-api", "openai")
    loaded = load_adapter(Runtime.OPENAI_COMPATIBLE)
    assert isinstance(loaded, OpenAICompatibleAdapter)
    assert loaded.runtime is Runtime.OPENAI_COMPATIBLE
    # It is 1.9's adapter with one transport swapped, and that is on purpose.
    assert isinstance(loaded, OpenAIAPIAdapter)


def test_the_client_is_constructed_with_the_connections_own_key_and_base_url():
    factory = Factory([text_round("hi")])
    drive(run_request(), factory)
    assert factory.kwargs == [{"api_key": KEY, "base_url": BASE_URL}]


def test_an_ambient_key_is_never_read_even_when_one_is_set(monkeypatch):
    """D2's in-process clause, and it bites harder here than anywhere else:
    the endpoint is an arbitrary host, so an ambient OpenAI key leaking into
    this client would not merely bill the wrong account -- it would send a real
    credential to somebody else's box."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient-decoy-must-never-be-used")
    factory = Factory([text_round("hi")])
    drive(run_request(), factory)
    assert factory.kwargs[0]["api_key"] == KEY


def test_a_secret_backed_credential_resolves_at_the_moment_of_use(tmp_path):
    from modelpass.secrets import SecretStore

    secrets = SecretStore(tmp_path / "home")
    secrets.set("ollama", "sk-from-the-secrets-file")
    conn = connection(credential_ref="secret:ollama")
    factory = Factory([text_round("hi")])
    api = OpenAICompatibleAdapter(client_factory=factory, env={}, secrets=secrets)
    list(api.run(run_request(conn)))
    assert factory.kwargs[0]["api_key"] == "sk-from-the-secrets-file"


def test_an_unset_credential_fails_the_preflight_rather_than_the_run():
    conn = connection(credential_ref="env:NOT_SET_ANYWHERE")
    receipt = adapter(Factory()).preflight(run_request(conn))
    assert not receipt.ok
    assert "NOT_SET_ANYWHERE" in (receipt.problem or "")


# --- 2. a connection with no credential at all ------------------------------------


def no_credential_connection(**kwargs: Any) -> Connection:
    return connection(credential_ref="none", **kwargs)


def test_only_this_runtime_accepts_a_connection_with_no_credential():
    """The scope is the whole of why 'none' is safe."""
    assert no_credential_connection().credential_ref.kind.value == "none"
    with pytest.raises(InvalidConnection) as excinfo:
        Connection(
            name="openai-key",
            runtime=Runtime.OPENAI_API,
            auth_mode=AuthMode.API_KEY,
            credential_ref=CredentialRef.none(),
        )
    assert "openai-compatible" in str(excinfo.value)


def test_a_connection_with_no_credential_sends_a_visible_placeholder():
    factory = Factory([text_round("hi")])
    drive(run_request(no_credential_connection()), factory)
    assert factory.kwargs == [
        {"api_key": PLACEHOLDER_CREDENTIAL, "base_url": BASE_URL}
    ]
    # And it is nothing like a key, so a log that shows it cannot be misread.
    assert not PLACEHOLDER_CREDENTIAL.startswith("sk-")


def test_the_receipt_says_no_credential_was_configured_and_still_says_it_is_ok():
    """'I configured none' and 'mine is missing' are opposite findings."""
    receipt = adapter(Factory()).preflight(run_request(no_credential_connection()))
    assert receipt.ok
    assert receipt.account is None
    assert any("declares no credential" in note for note in receipt.notes)
    assert any("placeholder" in note for note in receipt.notes)


def test_a_named_credential_still_fingerprints_on_the_receipt():
    receipt = adapter(Factory()).preflight(run_request())
    assert receipt.account == credential_fingerprint(KEY)
    assert receipt.plan_name is None
    assert receipt.binary is None


def test_no_ambient_key_reaches_a_connection_that_declares_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient-decoy-must-never-be-used")
    factory = Factory([text_round("hi")])
    drive(run_request(no_credential_connection()), factory)
    assert factory.kwargs[0]["api_key"] == PLACEHOLDER_CREDENTIAL


# --- 3. the model, the base URL and the probe -------------------------------------


def test_base_url_is_required_by_the_connection_itself():
    with pytest.raises(InvalidConnection):
        connection(base_url=None)


def test_a_run_with_no_model_anywhere_is_refused_before_anything_is_sent():
    factory = Factory([text_round("never sent")])
    with pytest.raises(PreflightFailed) as excinfo:
        list(adapter(factory).run(run_request(connection(model=None))))
    assert "modelpass verify" in str(excinfo.value)
    assert not factory.clients


def test_the_model_list_probe_is_opt_in_and_then_cached():
    factory = Factory(models_answer=listing("llama3.1", "qwen2.5"))
    api = adapter(factory)
    plain = api.preflight(run_request())
    assert not any("credential probe" in note for note in plain.notes)

    probed = api.preflight(run_request(options={"probe_credential": True}))
    assert any("2 model(s) listed" in note for note in probed.notes)
    api.preflight(run_request(options={"probe_credential": True}))
    assert factory.clients[0].model_list_calls == 1


def test_a_probe_that_cannot_reach_the_endpoint_is_a_reported_finding():
    factory = Factory(models_answer=ConnectionError("connection refused"))
    receipt = adapter(factory).preflight(run_request(options={"probe_credential": True}))
    assert not receipt.ok
    assert "connection refused" in (receipt.problem or "")


def test_the_receipt_names_the_wire_and_whether_anybody_has_driven_the_endpoint():
    receipt = adapter(Factory()).preflight(run_request())
    assert any("chat completions" in note for note in receipt.notes)
    assert any("modelpass verify ollama" in note for note in receipt.notes)


def test_a_verified_connection_says_so_on_the_receipt_instead():
    conn = connection(verified=all_verified())
    receipt = adapter(Factory()).preflight(run_request(conn))
    assert any("was driven by 'modelpass verify'" in note for note in receipt.notes)


# --- 4. streaming text, usage and the terminal ------------------------------------


def test_a_plain_call_streams_text_then_usage_then_a_terminal():
    events = drive(run_request(), Factory([text_round("hello")]))
    assert [event.type for event in events] == ["text_delta", "usage", "terminal"]
    assert events[-1].status is TerminalStatus.OK


def test_text_arrives_one_delta_at_a_time_in_the_order_it_was_produced():
    events = drive(run_request(), Factory([text_round("abcdef", split=3)]))
    deltas = [event.text for event in events if isinstance(event, TextDeltaEvent)]
    assert len(deltas) == 3
    assert "".join(deltas) == "abcdef"


def test_include_usage_is_asked_for_on_every_call():
    factory = Factory([text_round("hi")])
    drive(run_request(), factory)
    assert params_of(factory)["stream_options"] == {"include_usage": True}
    assert params_of(factory)["stream"] is True


def test_usage_puts_the_cached_prefix_where_modelpass_keeps_it():
    """Chat Completions counts the cached prefix INSIDE prompt_tokens; modelpass
    counts it beside input_tokens. So the adapter subtracts, exactly as the
    Responses one does, and a consumer's accounting does not learn that a run
    changed transports."""
    usage = FakeUsage(
        prompt_tokens=100,
        completion_tokens=20,
        prompt_tokens_details=FakePromptDetails(cached_tokens=30),
    )
    assert token_usage(usage) == TokenUsage(
        input_tokens=70, output_tokens=20, cached_input_tokens=30, cache_write_tokens=0
    )


def test_usage_reads_a_mapping_as_well_as_a_model_and_never_raises():
    assert token_usage({"prompt_tokens": 5, "completion_tokens": 2}) == TokenUsage(
        input_tokens=5, output_tokens=2
    )
    assert token_usage(None) == TokenUsage()
    assert token_usage(object()) == TokenUsage()


def test_a_server_that_reports_usage_produces_a_usage_event():
    events = drive(
        run_request(),
        Factory([text_round("hi", usage=FakeUsage(prompt_tokens=4, completion_tokens=1))]),
    )
    usage = [event for event in events if isinstance(event, UsageEvent)]
    assert usage and usage[0].usage.input_tokens == 4


def no_usage_round(text: str = "hi") -> Round:
    """A round from a server that ignored ``stream_options.include_usage``."""
    round_ = text_round(text)
    round_.chunks = [chunk for chunk in round_.chunks if chunk.usage is None]
    return round_


def test_a_server_that_reports_no_usage_emits_no_usage_event_at_all():
    """The claim a zeroed TokenUsage would make is 'this run cost nothing'. The
    truth is 'nobody told us', so the adapter says nothing rather than saying
    zero -- and the usage arm of the stream is simply absent."""
    events = drive(run_request(), Factory([no_usage_round()]))
    assert [event.type for event in events] == ["text_delta", "terminal"]
    assert not [event for event in events if isinstance(event, UsageEvent)]
    assert events[-1].status is TerminalStatus.OK


def test_the_same_run_against_a_server_that_does_report_usage_emits_one():
    events = drive(run_request(), Factory([text_round("hi")]))
    usage = [event for event in events if isinstance(event, UsageEvent)]
    assert len(usage) == 1
    assert usage[0].usage.output_tokens == 5


@NEEDS_SDK
def test_a_collected_answer_still_arrives_when_no_usage_was_reported(store):
    conn = connection(verified=all_verified())
    bridge, _ = bridge_for(conn, Factory([no_usage_round()]), store)
    result = bridge.ask(connection=conn.name, message="go")
    assert result.text == "hi"


def test_the_stream_is_closed_even_though_it_is_not_a_context_manager():
    factory = Factory([text_round("hi")])
    drive(run_request(), factory)
    assert factory.clients[0].streams[0].closed


def test_a_chunk_with_no_word_in_the_vocabulary_becomes_a_vendor_event():
    round_ = Round(
        chunks=[
            FakeChunk(choices=[FakeChoice(delta=FakeDelta())]),
            FakeChunk(choices=[FakeChoice(delta=FakeDelta(content="hi"))]),
            FakeChunk(choices=[FakeChoice(delta=FakeDelta(), finish_reason="stop")]),
        ]
    )
    events = drive(run_request(), Factory([round_]))
    vendor = [event for event in events if isinstance(event, VendorEvent)]
    assert len(vendor) == 1
    assert vendor[0].runtime is Runtime.OPENAI_COMPATIBLE
    assert vendor[0].name == "chunk.unrecognized"


def test_the_usage_chunk_has_no_choices_and_does_not_crash_the_mapping():
    """The shape include_usage produces, and where a naive choices[0] dies."""
    events = drive(run_request(), Factory([text_round("hi")]))
    assert [event.type for event in events].count("vendor_event") == 0


def test_there_is_no_thinking_arm_on_this_transport():
    """Chat Completions carries reasoning as a token count and no text, so a
    ThinkingEvent here could only ever be invented -- ticket 1.9's sentence,
    read from the other side."""
    events = drive(run_request(), Factory([text_round("hi")]))
    assert not [event for event in events if event.type == "thinking"]


# --- 5. the system prompt and the conversation -------------------------------------


def test_the_system_prompt_is_the_first_message_and_blocks_are_flattened():
    system = Message(
        role=Role.SYSTEM,
        content=[TextBlock("first"), TextBlock(""), TextBlock("second")],
    )
    factory = Factory([text_round("ok")])
    drive(
        run_request(
            messages=(system, Message(role=Role.USER, content="go")),
        ),
        factory,
    )
    messages = params_of(factory)["messages"]
    assert messages[0] == {"role": "system", "content": "first\n\nsecond"}
    assert messages[1] == {"role": "user", "content": "go"}


def test_a_run_with_no_system_message_sends_no_system_turn():
    factory = Factory([text_round("ok")])
    drive(run_request(), factory)
    assert [m["role"] for m in params_of(factory)["messages"]] == ["user"]


def test_conversation_turns_keep_their_order_and_their_roles():
    factory = Factory([text_round("ok")])
    drive(
        run_request(
            messages=(
                Message(role=Role.USER, content="one"),
                Message(role=Role.ASSISTANT, content="two"),
                Message(role=Role.USER, content="three"),
            )
        ),
        factory,
    )
    assert [m["role"] for m in params_of(factory)["messages"]] == [
        "user",
        "assistant",
        "user",
    ]


# --- 6. the in-adapter tool loop ---------------------------------------------------


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


def test_tools_are_sent_in_the_nested_chat_completions_shape():
    factory = Factory([text_round("ok")])
    tool = echo_tool([])
    drive(run_request(tools=(tool,)), factory)
    sent = params_of(factory)["tools"]
    assert sent[0]["type"] == "function"
    assert sent[0]["function"]["name"] == "look_up"
    # 'strict' is an OpenAI extension most compatible servers have never heard
    # of, and a key a server does not know is a key it may reject.
    assert "strict" not in sent[0]["function"]


def test_the_adapter_runs_the_loop_itself_and_reports_both_halves():
    calls: list[dict[str, Any]] = []
    tool = echo_tool(calls)
    factory = Factory(
        [tool_round("call_1", "look_up", {"topic": "otters"}), text_round("done")]
    )
    events = drive(run_request(tools=(tool,)), factory)
    call = next(event for event in events if isinstance(event, ToolCallEvent))
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert call.arguments == {"topic": "otters"}
    assert result.content == "looked up otters"
    assert calls == [{"topic": "otters"}]
    assert events[-1].status is TerminalStatus.OK


def test_tool_call_fragments_are_assembled_by_index_not_by_arrival():
    """Two tools in one turn, their argument JSON interleaved across chunks --
    the shape that turns into 'one tool called twice' when keyed wrongly."""
    first = json.dumps({"topic": "a"})
    second = json.dumps({"topic": "b"})
    round_ = Round(
        chunks=[
            FakeChunk(
                choices=[
                    FakeChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=0,
                                    id="call_a",
                                    function=FakeFunction(name="look_up", arguments=""),
                                ),
                                FakeToolCallDelta(
                                    index=1,
                                    id="call_b",
                                    function=FakeFunction(name="look_up", arguments=""),
                                ),
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(
                choices=[
                    FakeChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=1, function=FakeFunction(arguments=second)
                                ),
                                FakeToolCallDelta(
                                    index=0, function=FakeFunction(arguments=first)
                                ),
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(choices=[FakeChoice(delta=FakeDelta(), finish_reason="tool_calls")]),
        ]
    )
    calls: list[dict[str, Any]] = []
    tool = echo_tool(calls)
    events = drive(run_request(tools=(tool,)), Factory([round_, text_round("done")]))
    assert calls == [{"topic": "a"}, {"topic": "b"}]
    ids = [event.id for event in events if isinstance(event, ToolCallEvent)]
    assert ids == ["call_a", "call_b"]


def test_the_assistant_turn_is_replayed_before_the_tool_results():
    tool = echo_tool([])
    factory = Factory(
        [tool_round("call_1", "look_up", {"topic": "x"}), text_round("done")]
    )
    drive(run_request(tools=(tool,)), factory)
    second = params_of(factory, 1)["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["id"] == "call_1"
    # The arguments go back as the JSON string that was streamed, not a
    # re-serialization of the parsed form.
    assert second[-2]["tool_calls"][0]["function"]["arguments"] == json.dumps(
        {"topic": "x"}
    )
    assert second[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "looked up x",
    }


def test_there_is_no_turn_cap():
    tool = echo_tool([])
    rounds = [tool_round(f"call_{n}", "look_up", {"topic": str(n)}) for n in range(6)]
    factory = Factory([*rounds, text_round("finally")])
    events = drive(run_request(tools=(tool,)), factory)
    assert len([e for e in events if isinstance(e, ToolResultEvent)]) == 6
    assert events[-1].status is TerminalStatus.OK


def test_a_raising_handler_becomes_a_failed_tool_result_and_the_run_continues():
    def boom(args: dict[str, Any]) -> str:
        raise RuntimeError("the database is on fire")

    tool = ToolDef(
        name="look_up",
        description="Look a topic up.",
        parameters={"type": "object", "properties": {}},
        handler=boom,
    )
    factory = Factory([tool_round("call_1", "look_up", {}), text_round("recovered")])
    events = drive(run_request(tools=(tool,)), factory)
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert result.is_error
    assert "the database is on fire" in result.content
    assert events[-1].status is TerminalStatus.OK


def test_a_tool_the_run_never_declared_is_a_failed_result_rather_than_a_crash():
    tool = echo_tool([])
    factory = Factory([tool_round("call_1", "not_declared", {}), text_round("ok")])
    events = drive(run_request(tools=(tool,)), factory)
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert result.is_error
    assert "not declared" in result.content


def test_unparseable_call_arguments_reach_the_handler_as_an_empty_mapping():
    """More likely here than on Responses: the string was concatenated from
    stream fragments and a truncated turn can end mid-JSON."""
    assert call_arguments('{"topic": "x"') == {}
    assert call_arguments("") == {}
    assert call_arguments("[1, 2]") == {}
    assert call_arguments('{"topic": "x"}') == {"topic": "x"}


def test_a_tool_call_on_a_run_that_declared_no_tools_ends_rather_than_loops():
    factory = Factory([tool_round("call_1", "look_up", {})])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.ERROR
    assert "did not declare" in events[-1].reason
    assert len(factory.clients[0].calls) == 1


@NEEDS_SDK
def test_a_token_guard_stops_the_loop_between_rounds(store):
    tool = echo_tool([])
    conn = connection(guards=Guards(stop_at_tokens=10), verified=all_verified())
    factory = Factory(
        [tool_round("call_1", "look_up", {"topic": "x"}), text_round("never reached")]
    )
    bridge, _ = bridge_for(conn, factory, store)
    events = list(bridge.chat(connection=conn.name, message="go", tools=[tool]))
    assert any(isinstance(e, GuardStopEvent) for e in events)
    assert events[-1].status is TerminalStatus.GUARD_STOP
    assert len(factory.clients[0].calls) == 1


# --- 7. structured output ----------------------------------------------------------


SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string"}},
    "required": ["sentiment"],
}


def test_a_schema_is_sent_as_a_strict_json_schema_response_format():
    factory = Factory([text_round(json.dumps({"sentiment": "positive"}))])
    events = drive(run_request(schema=SCHEMA, schema_name="Verdict"), factory)
    sent = params_of(factory)["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["name"] == "Verdict"
    assert sent["json_schema"]["strict"] is True
    assert sent["json_schema"]["schema"]["additionalProperties"] is False
    structured = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert structured.data == {"sentiment": "positive"}
    assert structured.valid


def test_no_schema_means_no_response_format_on_the_wire():
    factory = Factory([text_round("plain")])
    drive(run_request(), factory)
    assert "response_format" not in params_of(factory)


def test_an_endpoint_that_rejects_json_schema_raises_a_typed_refusal():
    """Many compatible servers do not implement it, so this is a first-class
    outcome rather than a bug -- and it is a capability refusal, not a run
    failure, which is why it raises rather than ending in a terminal."""
    factory = Factory(
        [Round(raises=Status400("Invalid value for 'response_format.type': json_schema"))]
    )
    with pytest.raises(StructuredOutputRejected) as excinfo:
        drive(run_request(schema=SCHEMA), factory)
    assert excinfo.value.capability == "structured_output"
    assert isinstance(excinfo.value, CapabilityNotSupported)


def test_the_rejection_says_what_to_try():
    factory = Factory([Round(raises=Status400("unknown field response_format"))])
    with pytest.raises(StructuredOutputRejected) as excinfo:
        drive(run_request(schema=SCHEMA), factory)
    detail = excinfo.value.detail
    assert "modelpass verify ollama" in detail
    # And it says plainly what modelpass refuses to do on the caller's behalf.
    assert "json_object" in detail


def test_a_400_that_is_not_about_the_schema_stays_an_ordinary_terminal():
    factory = Factory([Round(raises=Status400("model 'nope' not found"))])
    events = drive(run_request(schema=SCHEMA), factory)
    assert events[-1].status is TerminalStatus.ERROR
    assert "HTTP 400" in events[-1].reason


def test_a_rejection_on_a_call_with_no_schema_is_not_reinterpreted():
    factory = Factory([Round(raises=Status400("response_format is unsupported"))])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.ERROR


# --- 8. terminals, failures and sampling -------------------------------------------


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", TerminalStatus.OK),
        ("length", TerminalStatus.ERROR),
        ("content_filter", TerminalStatus.ERROR),
        ("something_new", TerminalStatus.OK),
    ],
)
def test_each_finish_reason_maps_to_its_terminal_status(finish_reason, expected):
    status, _ = terminal_status(finish_reason, unanswered_call=False, answered=True)
    assert status is expected


def test_a_truncated_answer_says_which_ceiling_it_hit_and_how_to_raise_it():
    _, reason = terminal_status("length", unanswered_call=False, answered=True)
    assert "max_output_tokens" in reason


def test_a_stream_with_no_finish_reason_is_ok_if_text_arrived_and_an_error_if_not():
    """Where this diverges from openai-api, and the divergence is the
    transport's: several compatible servers simply never set finish_reason."""
    assert terminal_status(None, unanswered_call=False, answered=True)[0] is TerminalStatus.OK
    status, reason = terminal_status(None, unanswered_call=False, answered=False)
    assert status is TerminalStatus.ERROR
    assert "no answer arrived" in reason


def test_a_silent_stream_ends_as_an_error_rather_than_an_empty_ok():
    events = drive(run_request(), Factory([silent_round(finish_reason=None)]))
    assert events[-1].status is TerminalStatus.ERROR


def test_a_429_is_quota_exhausted_and_is_typed_off_the_status_code():
    factory = Factory([Round(raises=Status400("slow down", status_code=429))])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED


def test_any_other_vendor_failure_is_a_reported_terminal_not_a_raise():
    factory = Factory([Round(raises=ConnectionError("connection refused"))])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.ERROR
    assert "ConnectionError" in events[-1].reason


def test_the_three_accepted_controls_reach_the_request_under_the_wire_names():
    factory = Factory([text_round("ok")])
    drive(
        run_request(sampling=Sampling(temperature=0.3, top_p=0.9, max_output_tokens=256)),
        factory,
    )
    params = params_of(factory)
    assert params["temperature"] == 0.3
    assert params["top_p"] == 0.9
    # max_output_tokens -> max_tokens, which is what every compatible server
    # has implemented since the beginning. See sampling_params for the reason.
    assert params["max_tokens"] == 256
    assert "max_completion_tokens" not in params


def test_top_k_never_reaches_the_wire():
    factory = Factory([text_round("ok")])
    drive(run_request(sampling=Sampling(top_k=40)), factory)
    assert "top_k" not in params_of(factory)
    assert "extra_body" not in params_of(factory)


def test_no_ceiling_is_invented_when_the_caller_names_none():
    factory = Factory([text_round("ok")])
    drive(run_request(), factory)
    assert "max_tokens" not in params_of(factory)


# --- 9. the wire option ------------------------------------------------------------


def test_the_default_wire_is_chat_completions():
    assert OpenAICompatibleAdapter.wire(run_request()) == "chat"
    assert OpenAICompatibleAdapter.wire(run_request(options={"wire": "chat"})) == "chat"


def test_the_responses_wire_runs_ticket_1_9s_path_unchanged():
    """Not a reimplementation of Responses: it IS the openai-api adapter,
    running against this connection's base URL."""
    factory = Factory([text_round("never reached")])
    request = run_request(options={"wire": "responses"})
    assert OpenAICompatibleAdapter.wire(request) == "responses"
    events = list(adapter(factory).run(request))
    client = factory.clients[0]
    # The Responses surface was used and chat completions was never touched.
    assert client.responses_calls and not client.calls
    # And the request was assembled by 1.9's code: ``input`` rather than
    # ``messages``, with no ``stream`` flag anywhere.
    assert "input" in client.responses_calls[0]
    assert "messages" not in client.responses_calls[0]
    assert [event.type for event in events] == ["text_delta", "usage", "terminal"]
    # The terminal is stamped with THIS runtime, not with openai-api -- the one
    # thing the shared path had to be taught.
    assert events[-1].runtime is Runtime.OPENAI_COMPATIBLE


def test_an_unknown_wire_value_falls_back_to_the_default_rather_than_raising():
    assert OpenAICompatibleAdapter.wire(run_request(options={"wire": "grpc"})) == "chat"


def test_wire_is_a_known_option_key_so_a_typo_in_it_is_reported():
    assert "wire" in OpenAICompatibleAdapter.option_keys
    assert OpenAICompatibleAdapter.unknown_option_keys({"wier": "responses"}) == ("wier",)


def test_the_receipt_says_which_wire_a_responses_call_will_use():
    receipt = adapter(Factory()).preflight(run_request(options={"wire": "responses"}))
    assert any("wire: responses" in note for note in receipt.notes)


# --- 10. the per-install drive, and the cells it moves -----------------------------


def verify_rounds() -> list[Round]:
    """The three drives verify runs, all answering yes."""
    return [
        text_round("ok"),
        tool_round("call_1", "modelpass_verify_probe", {}),
        text_round("ok"),
        text_round(json.dumps({"ok": True})),
    ]


def test_the_static_row_is_unverified_by_construction_and_names_the_command():
    registry = CapabilityRegistry()
    assert registry.support(Runtime.OPENAI_COMPATIBLE, Capability.CHAT) is Support.UNVERIFIED
    for capability in Capability:
        note = registry.note(Runtime.OPENAI_COMPATIBLE, capability) or ""
        assert "modelpass verify" in note
    # The cells that are facts about the shape rather than about the endpoint.
    assert registry.support(Runtime.OPENAI_COMPATIBLE, Capability.API_KEY_AUTH) is Support.SUPPORTED
    for capability in (
        Capability.SESSIONS_RESUME,
        Capability.SESSIONS_FORK,
        Capability.SESSIONS_LIST,
        Capability.MCP_SERVERS,
        Capability.SUBAGENTS,
        Capability.CACHE_BREAKPOINTS,
        Capability.TTL_CONTROL,
    ):
        assert registry.support(Runtime.OPENAI_COMPATIBLE, capability) is Support.UNSUPPORTED


def test_an_unverified_endpoint_is_refused_by_chat_before_anything_is_sent(store):
    conn = connection()
    factory = Factory([text_round("never sent")])
    bridge, _ = bridge_for(conn, factory, store)
    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(bridge.chat(connection=conn.name, message="go"))
    assert "modelpass verify" in str(excinfo.value)
    assert not factory.clients


@NEEDS_SDK
def test_a_verified_endpoint_chats(store):
    conn = connection(verified=all_verified())
    bridge, _ = bridge_for(conn, Factory([text_round("hi")]), store)
    events = list(bridge.chat(connection=conn.name, message="go"))
    assert events[-1].status is TerminalStatus.OK


def test_the_drive_moves_six_cells_and_records_the_models_it_saw():
    factory = Factory(verify_rounds(), models_answer=listing("llama3.1", "qwen2.5"))
    report = adapter(factory).verify_capabilities(run_request())
    assert report.ok
    assert set(report.verified.supported) == {str(cell) for cell in VERIFY_CELLS}
    assert report.verified.unsupported == ()
    assert report.verified.models == ("llama3.1", "qwen2.5")
    assert report.verified.checked_at


def test_the_drive_refines_a_registry_copy_and_leaves_the_static_table_alone():
    factory = Factory(verify_rounds(), models_answer=listing(MODEL))
    report = adapter(factory).verify_capabilities(run_request())
    registry = CapabilityRegistry()
    refined = registry.refined(Runtime.OPENAI_COMPATIBLE, report.verified.observed)
    assert refined.support(Runtime.OPENAI_COMPATIBLE, Capability.CHAT) is Support.SUPPORTED
    assert registry.support(Runtime.OPENAI_COMPATIBLE, Capability.CHAT) is Support.UNVERIFIED


def test_an_endpoint_that_is_not_running_writes_nothing_at_all():
    """The worst lie this record could tell is 'driven, and it can do nothing'
    about a box that was switched off."""
    factory = Factory([], models_answer=ConnectionError("connection refused"))
    report = adapter(factory).verify_capabilities(run_request())
    assert not report.ok
    assert not report.verified
    assert "connection refused" in (report.problem or "")


def test_a_server_that_chats_but_rejects_a_schema_records_exactly_that():
    factory = Factory(
        [
            text_round("ok"),
            tool_round("call_1", "modelpass_verify_probe", {}),
            text_round("ok"),
            Round(raises=Status400("response_format json_schema is not supported")),
        ],
        models_answer=listing(MODEL),
    )
    report = adapter(factory).verify_capabilities(run_request())
    assert report.ok
    assert "structured_output" in report.verified.unsupported
    assert "chat" in report.verified.supported
    assert "tools_in_process" in report.verified.supported


def test_a_server_that_ignores_include_usage_records_usage_tokens_as_unsupported():
    rounds = verify_rounds()
    rounds[0].chunks = [chunk for chunk in rounds[0].chunks if chunk.usage is None]
    factory = Factory(rounds, models_answer=listing(MODEL))
    report = adapter(factory).verify_capabilities(run_request())
    assert "usage_tokens" in report.verified.unsupported
    assert "chat" in report.verified.supported


def test_a_model_that_declines_the_tool_records_tools_as_unsupported_and_says_why():
    factory = Factory(
        [text_round("ok"), text_round("I would rather not"), text_round(json.dumps({"ok": True}))],
        models_answer=listing(MODEL),
    )
    report = adapter(factory).verify_capabilities(run_request())
    assert "tools_in_process" in report.verified.unsupported
    assert any("may decline a tool" in note for note in report.notes)


def test_an_endpoint_that_answers_nothing_skips_the_later_drives():
    factory = Factory([silent_round(finish_reason=None)], models_answer=listing(MODEL))
    report = adapter(factory).verify_capabilities(run_request())
    assert report.ok
    assert "chat" in report.verified.unsupported
    assert "tools_in_process" not in report.verified.supported
    assert any("were not attempted" in note for note in report.notes)


def test_the_drive_never_reuses_a_cached_probe():
    factory = Factory(verify_rounds(), models_answer=listing(MODEL))
    api = adapter(factory)
    api.cached_probe(run_request())
    api.verify_capabilities(run_request())
    # Two clients, two listings: verify dropped the cache before asking.
    assert sum(client.model_list_calls for client in factory.clients) == 2


def test_every_other_runtime_has_nothing_to_verify_per_install():
    """The hook's default, and the reason it is a default: driving one install
    of anthropic-api teaches nothing that is not already in the table."""
    from modelpass.adapters.anthropic_api import AnthropicAPIAdapter

    conn = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(f"env:{KEY_VAR}"),
    )
    request = RunRequest(
        connection=conn,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(conn, ENV),
    )
    assert AnthropicAPIAdapter().verify_capabilities(request) is None


# --- 11. modelpass verify, end to end ---------------------------------------------


def run_cli(argv: list[str], bridge: Bridge) -> tuple[int, str]:
    import io

    out = io.StringIO()
    code = cli_main(argv, bridge=bridge, out=out, err=out, confirm=lambda _: True)
    return code, out.getvalue()


def test_verify_drives_the_endpoint_and_writes_the_result_to_the_store(store):
    conn = connection()
    factory = Factory(verify_rounds(), models_answer=listing("llama3.1", "qwen2.5"))
    bridge, _ = bridge_for(conn, factory, store)
    code, output = run_cli(["verify", conn.name], bridge)
    assert code == 0
    assert "llama3.1" in output
    written = store.get(conn.name)
    assert "chat" in written.verified_capabilities.supported
    assert written.verified_capabilities.models == ("llama3.1", "qwen2.5")
    # And the connection can now chat, which it could not before.
    assert bridge.registry_for(written).supports(
        Runtime.OPENAI_COMPATIBLE, Capability.CHAT
    )


def test_verify_against_a_dead_endpoint_writes_nothing(store):
    conn = connection()
    factory = Factory([], models_answer=ConnectionError("connection refused"))
    bridge, _ = bridge_for(conn, factory, store)
    code, output = run_cli(["verify", conn.name], bridge)
    assert code == 1
    assert "Nothing was written." in output
    assert not store.get(conn.name).verified_capabilities


def test_verify_warns_that_it_spends_before_it_does(store):
    conn = connection()
    bridge, _ = bridge_for(
        conn, Factory(verify_rounds(), models_answer=listing(MODEL)), store
    )
    _, output = run_cli(["verify", conn.name], bridge)
    assert "spends" in output


def test_verify_takes_the_subscription_path_for_a_subscription_account(
    store, subscription_connection
):
    """Two verbs behind one word, and this is the other one: untouched."""
    from modelpass.testing import FakeAdapter

    store.add(subscription_connection)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter()},
        env={},
    )
    _, output = run_cli(["verify", subscription_connection.name], bridge)
    # The identity path prints a receipt; the capability drive announces itself
    # and prints a model list. Whichever way the pin falls, this is not that.
    assert "Three short calls" not in output
    assert subscription_connection.name in output


def test_verify_on_another_api_runtime_says_there_is_nothing_to_do(store):
    """No identity to pin and no cell an install could move."""
    conn = Connection(
        name="openai-key",
        runtime=Runtime.OPENAI_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(f"env:{KEY_VAR}"),
        model="gpt-4o",
    )
    store.add(conn)
    bridge = Bridge(store=store, registry=CapabilityRegistry(), adapters={}, env=dict(ENV))
    code, output = run_cli(["verify", conn.name], bridge)
    assert code == 1
    assert "no subscription identity to pin" in output


def test_connect_compatible_with_no_key_flags_writes_a_none_credential(store):
    bridge = Bridge(store=store, registry=CapabilityRegistry(), adapters={}, env={})
    code, output = run_cli(
        ["connect", "compatible", "--base-url", BASE_URL, "--name", "ollama", "--yes"],
        bridge,
    )
    assert code == 0, output
    written = store.get("ollama")
    assert written.credential_ref.kind.value == "none"
    assert written.runtime is Runtime.OPENAI_COMPATIBLE


def test_the_store_round_trips_a_verified_connection(store):
    conn = connection(verified=all_verified())
    store.add(conn)
    reloaded = store.get(conn.name)
    assert reloaded.verified_capabilities == conn.verified_capabilities
    text = store.path.read_text(encoding="utf-8")
    assert "verifiedCapabilities" in text


def test_a_record_that_says_both_yes_and_no_about_one_cell_is_refused():
    with pytest.raises(InvalidConnection):
        VerifiedCapabilities(supported=("chat",), unsupported=("chat",))


# --- 12. sessions: refused, with the alternative named -----------------------------


@pytest.mark.parametrize("method", ["open_session", "resume_session", "list_sessions"])
def test_every_session_door_is_refused_and_names_the_stateless_shape(method):
    api = adapter(Factory())
    with pytest.raises(AdapterNotImplemented) as excinfo:
        getattr(api, method)(session_request())
    message = str(excinfo.value)
    assert "openai-compatible" in message
    assert "bridge.chat(history=[...])" in message


def test_the_bridge_refuses_a_session_before_the_adapter_is_even_loaded(store):
    conn = connection(verified=all_verified())
    bridge, _ = bridge_for(conn, Factory(), store)
    with pytest.raises(CapabilityNotSupported):
        bridge.new_chat(connection=conn.name)
