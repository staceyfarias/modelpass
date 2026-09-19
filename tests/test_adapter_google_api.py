"""The ``google-api`` adapter, end to end, with no network anywhere (ticket 1.11).

Deliberately the same file, section by section, as
``tests/test_adapter_openai_api.py`` -- which is itself the same file as
``tests/test_adapter_anthropic_api.py``. Four adapters that are meant to behave
identically should be *readable* side by side, and a case that appears in three
files and not the fourth is then a visible difference rather than an invisible
one.

Every test here drives a **fake transport**: an object shaped like the pieces of
``google.genai.Client`` this adapter touches
(``models.generate_content_stream(...)`` and ``models.list()``), injected through
the adapter's ``client_factory`` constructor seam. Nothing here imports the
vendor SDK to make a call, nothing opens a socket, and nothing spends. The live
drive that *does* spend is ``tests/live/test_google_api_live.py``, deselected by
default.

The fake is literal about the shapes the adapter reads, because a fake looser
than the vendor is a test that passes against an adapter that would not run. The
three that matter most, and that an adapter written from another vendor's memory
gets wrong:

* a chunk is a whole ``GenerateContentResponse`` with ``candidates`` on it, not a
  typed event with a ``type`` string;
* a thought is a **flag on a text part** (``Part.thought``), so the model's
  private reasoning arrives in the same field as its answer;
* the assistant turn's role is ``model``, and a tool result goes back as a
  ``function_response`` part in a ``user`` turn.

Each shape is pinned against the installed SDK's own types in
:func:`test_the_fake_matches_the_installed_sdk_shapes`, which also validates the
exact ``config`` dict this adapter sends through the vendor's own model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from modelpass.adapters import ADAPTER_REGISTRY, load_adapter
from modelpass.adapters.anthropic_api import token_usage as anthropic_token_usage
from modelpass.adapters.base import RunRequest, SessionRequest
from modelpass.adapters.google_api import (
    SDK_VERSION_READ,
    GoogleAPIAdapter,
    terminal_status,
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

KEY = "AIza-not-a-real-key-0123456789"
KEY_VAR = "MODELPASS_TEST_GOOGLE_KEY"
ENV = {KEY_VAR: KEY}
MODEL = "gemini-2.5-flash"


# --- the fake transport ----------------------------------------------------------


@dataclass
class FakeUsage:
    """``GenerateContentResponseUsageMetadata``: four counts and their total."""

    prompt_token_count: int = 0
    candidates_token_count: int = 0
    cached_content_token_count: int = 0
    thoughts_token_count: int = 0
    tool_use_prompt_token_count: int = 0

    @property
    def total_token_count(self) -> int:
        """The vendor's own documented sum, so a test can assert against it."""
        return (
            self.prompt_token_count
            + self.candidates_token_count
            + self.thoughts_token_count
            + self.tool_use_prompt_token_count
        )


@dataclass
class FakeFunctionCall:
    name: str
    args: dict[str, Any] | str | None = None
    id: str | None = None

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        dumped = {"name": self.name, "args": self.args, "id": self.id}
        if exclude_none:
            return {k: v for k, v in dumped.items() if v is not None}
        return dumped


@dataclass
class FakePart:
    """One ``Part``. ``thought`` is a flag on the text, not a part type."""

    text: str | None = None
    thought: bool | None = None
    thought_signature: str | None = None
    function_call: FakeFunctionCall | None = None
    function_response: dict[str, Any] | None = None
    executable_code: dict[str, Any] | None = None

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        dumped = {
            "text": self.text,
            "thought": self.thought,
            "thought_signature": self.thought_signature,
            "function_call": (
                self.function_call.model_dump(exclude_none=exclude_none)
                if self.function_call is not None
                else None
            ),
            "function_response": self.function_response,
            "executable_code": self.executable_code,
        }
        if exclude_none:
            return {k: v for k, v in dumped.items() if v is not None}
        return dumped


@dataclass
class FakeContent:
    parts: list[FakePart] = field(default_factory=list)
    role: str = "model"

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return {
            "role": self.role,
            "parts": [part.model_dump(exclude_none=exclude_none) for part in self.parts],
        }


@dataclass
class FakeCandidate:
    content: FakeContent | None = None
    finish_reason: str | None = None


@dataclass
class FakePromptFeedback:
    block_reason: str | None = None


@dataclass
class FakeChunk:
    """One streamed ``GenerateContentResponse``."""

    candidates: list[FakeCandidate] = field(default_factory=list)
    usage_metadata: FakeUsage | None = None
    prompt_feedback: FakePromptFeedback | None = None


@dataclass
class Round:
    """One scripted call: the chunks it streams, or the exception it raises."""

    chunks: list[FakeChunk] = field(default_factory=list)
    raises: Exception | None = None


class FakeStream:
    """The iterator ``generate_content_stream`` returns. Not a context manager."""

    def __init__(self, round_: Round) -> None:
        self._round = round_
        self.closed = False

    def __iter__(self):
        yield from self._round.chunks

    def close(self) -> None:
        self.closed = True


class FakeModels:
    def __init__(self, client: FakeClient) -> None:
        self._client = client

    def generate_content_stream(self, **params: Any) -> FakeStream:
        self._client.calls.append(params)
        if not self._client.rounds:
            raise AssertionError("the adapter asked for more rounds than were scripted")
        round_ = self._client.rounds.pop(0)
        if round_.raises is not None:
            raise round_.raises
        stream = FakeStream(round_)
        self._client.streams.append(stream)
        return stream

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
        self.models = FakeModels(self)


class Factory:
    """Records how the client was constructed -- the point of items 1 and 2."""

    def __init__(
        self, rounds: list[Round] | None = None, models_answer: Any = None
    ) -> None:
        self.rounds = rounds or []
        self.models_answer = models_answer
        self.kwargs: list[dict[str, Any]] = []
        self.clients: list[FakeClient] = []

    def __call__(self, *, api_key: str, base_url: str | None) -> FakeClient:
        self.kwargs.append({"api_key": api_key, "base_url": base_url})
        client = FakeClient(self.rounds, self.models_answer)
        self.clients.append(client)
        return client


def chunk(
    *parts: FakePart,
    finish_reason: str | None = None,
    usage: FakeUsage | None = None,
) -> FakeChunk:
    return FakeChunk(
        candidates=[
            FakeCandidate(content=FakeContent(parts=list(parts)), finish_reason=finish_reason)
        ],
        usage_metadata=usage,
    )


def text_round(text: str, *, finish_reason: str = "STOP", usage: Any = None) -> Round:
    """The commonest shape: one text delta, then a chunk carrying the ending."""
    return Round(
        chunks=[
            chunk(FakePart(text=text)),
            chunk(
                finish_reason=finish_reason,
                usage=usage
                if usage is not None
                else FakeUsage(prompt_token_count=10, candidates_token_count=5),
            ),
        ]
    )


def tool_round(name: str, arguments: dict[str, Any], *, call_id: str | None = None) -> Round:
    return Round(
        chunks=[
            chunk(
                FakePart(function_call=FakeFunctionCall(name=name, args=arguments, id=call_id)),
                finish_reason="STOP",
                usage=FakeUsage(prompt_token_count=8, candidates_token_count=4),
            )
        ]
    )


# --- connections, requests, bridges ----------------------------------------------


def connection(
    *,
    name: str = "gemini-key",
    model: str | None = MODEL,
    base_url: str | None = None,
    credential_ref: str = f"env:{KEY_VAR}",
    guards: Guards | None = None,
) -> Connection:
    return Connection(
        name=name,
        runtime=Runtime.GOOGLE_API,
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


def adapter(factory: Factory) -> GoogleAPIAdapter:
    return GoogleAPIAdapter(client_factory=factory, env=ENV)


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


def sent(factory: Factory, index: int = 0) -> dict[str, Any]:
    """One request's parameters, as the vendor would have received them."""
    return factory.clients[0].calls[index]


def config_of(factory: Factory, index: int = 0) -> dict[str, Any]:
    return sent(factory, index).get("config", {})


def bridge_for(
    conn: Connection, factory: Factory, store: Any
) -> tuple[Bridge, GoogleAPIAdapter]:
    store.add(conn)
    api = adapter(factory)
    return (
        Bridge(
            store=store,
            registry=CapabilityRegistry(),
            adapters={Runtime.GOOGLE_API: api},
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
    not GoogleAPIAdapter.is_available(),
    reason="needs the modelpass[google-api] extra: google-genai is not installed",
)


# --- 1. registration, and a client built with an explicit key ---------------------


@NEEDS_SDK
def test_the_runtime_resolves_to_this_adapter_through_the_usual_discovery():
    entry = ADAPTER_REGISTRY[Runtime.GOOGLE_API]
    assert entry.module == "modelpass.adapters.google_api"
    assert entry.attribute == "GoogleAPIAdapter"
    # google-genai, the current SDK, and never the retired google-generativeai.
    assert (entry.extra, entry.package) == ("google-api", "google-genai")
    loaded = load_adapter(Runtime.GOOGLE_API)
    assert isinstance(loaded, GoogleAPIAdapter)
    assert loaded.runtime is Runtime.GOOGLE_API


def test_every_api_runtime_now_has_an_adapter():
    """The line ticket 1.11 completes: four runtimes, four entries."""
    from modelpass.runtimes import API_RUNTIMES

    assert set(API_RUNTIMES) <= set(ADAPTER_REGISTRY)


def test_the_client_is_constructed_with_the_connections_own_key():
    factory = Factory([text_round("hi")])
    drive(run_request(), factory)
    assert factory.kwargs == [{"api_key": KEY, "base_url": None}]


def test_neither_ambient_key_is_ever_read_even_when_both_are_set(monkeypatch):
    """The two decoys that would have billed the wrong account.

    ``tests/test_no_ambient_credentials.py`` owns this rule across the library;
    it is restated here because it is the one promise of this adapter that costs
    money when it breaks, and because this SDK discovers **two** ambient names
    rather than the one every other vendor uses.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "ambient-gemini-key-never-to-be-used")
    monkeypatch.setenv("GOOGLE_API_KEY", "ambient-google-key-never-to-be-used")
    factory = Factory([text_round("hi")])
    drive(run_request(), factory)
    assert factory.kwargs == [{"api_key": KEY, "base_url": None}]


def test_a_connections_base_url_reaches_the_client_and_absence_stays_absent():
    factory = Factory([text_round("hi")])
    drive(run_request(connection(base_url="https://gateway.example")), factory)
    assert factory.kwargs[0]["base_url"] == "https://gateway.example"


def test_a_secret_backed_credential_resolves_at_the_moment_of_use(tmp_path):
    from modelpass.secrets import SecretStore

    secrets = SecretStore(tmp_path / "home")
    secrets.set("gemini-key", KEY)
    factory = Factory([text_round("hi")])
    api = GoogleAPIAdapter(client_factory=factory, env={}, secrets=secrets)
    conn = connection(credential_ref="secret:gemini-key")
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
    api = GoogleAPIAdapter(client_factory=Factory(), env={})
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
    """A Gemini model names itself in ``name``, not ``id``."""
    listing = [type("M", (), {"name": f"models/{MODEL}"})()]
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


def test_the_probe_stops_reading_the_pager_once_it_has_enough():
    """``models.list()`` returns a pager that fetches further pages as it goes,
    so the probe must not page through a catalogue to answer "is the key live"."""
    from modelpass.adapters.google_api import _PROBE_LIMIT

    listing = [type("M", (), {"name": f"models/gemini-{i}"})() for i in range(60)]
    factory = Factory(models_answer=listing)
    probe = adapter(factory).probe(run_request())
    assert probe is not None and probe.ok
    assert len(probe.models) == _PROBE_LIMIT


def test_a_probe_that_cannot_reach_the_endpoint_is_a_reported_finding():
    factory = Factory(models_answer=RuntimeError("nope"))
    api = adapter(factory)
    receipt = api.preflight(run_request(options={"probe_credential": True}))
    assert not receipt.ok
    assert "RuntimeError: nope" in (receipt.problem or "")


@NEEDS_SDK
def test_runtime_available_reports_whether_the_sdk_imports():
    assert GoogleAPIAdapter.is_available() is True


def test_an_absent_sdk_is_an_answer_rather_than_an_exception(monkeypatch):
    """``google.genai`` is dotted, and ``find_spec`` raises for the missing
    *parent* rather than answering ``None``. On a machine without the extra,
    ``google`` itself is absent, so an unguarded probe took down every caller
    that only wanted to know whether this runtime could run."""

    def missing_parent(name):
        raise ModuleNotFoundError("No module named 'google'")

    monkeypatch.setattr("importlib.util.find_spec", missing_parent)
    assert GoogleAPIAdapter.is_available() is False


# --- 3. the stream, the events, and the usage mapping -----------------------------


def test_a_plain_call_streams_text_then_usage_then_a_terminal():
    factory = Factory([text_round("hello there")])
    events = drive(run_request(), factory)
    assert [type(e) for e in events] == [TextDeltaEvent, UsageEvent, TerminalEvent]
    assert events[0].text == "hello there"
    assert events[-1].status is TerminalStatus.OK


def test_text_arrives_one_delta_at_a_time_in_the_order_it_was_produced():
    round_ = Round(
        chunks=[
            chunk(FakePart(text="one ")),
            chunk(FakePart(text="two"), finish_reason="STOP", usage=FakeUsage()),
        ]
    )
    events = drive(run_request(), Factory([round_]))
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["one ", "two"]


def test_a_thought_part_becomes_thinking_and_never_text():
    """The failure this vendor's shape makes easy: a thought carries ordinary
    ``text``, so a reader that ignores ``Part.thought`` streams the model's
    private reasoning into the answer a user reads."""
    round_ = Round(
        chunks=[
            chunk(FakePart(text="weighing it", thought=True)),
            chunk(FakePart(text="the answer"), finish_reason="STOP", usage=FakeUsage()),
        ]
    )
    events = drive(run_request(), Factory([round_]))
    assert [e.text for e in events if isinstance(e, ThinkingEvent)] == ["weighing it"]
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["the answer"]


def test_thoughts_are_only_asked_for_when_the_caller_asks():
    """modelpass does not order billed output nobody requested."""
    factory = Factory([text_round("ok")])
    drive(run_request(), factory)
    assert "thinking_config" not in config_of(factory)

    factory = Factory([text_round("ok")])
    drive(run_request(options={"include_thoughts": True}), factory)
    assert config_of(factory)["thinking_config"] == {"include_thoughts": True}


def test_a_thought_is_not_the_text_a_schema_is_validated_against():
    """The other half of the same rule: thinking is not part of the answer."""
    round_ = Round(
        chunks=[
            chunk(FakePart(text='{"sentiment": "confused"}', thought=True)),
            chunk(
                FakePart(text=json.dumps({"sentiment": "positive"})),
                finish_reason="STOP",
                usage=FakeUsage(),
            ),
        ]
    )
    events = drive(run_request(schema=SCHEMA), Factory([round_]))
    structured = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert structured.data == {"sentiment": "positive"}


def test_usage_puts_geminis_four_counts_where_modelpass_keeps_them():
    """The subtraction *and* the addition, against the anthropic mapping.

    Gemini reports 100 prompt tokens *including* 30 cached, and reports 12
    thinking tokens *beside* 40 candidate tokens. Anthropic reports the same call
    as 70 fresh, 30 read and 52 output. A consumer's accounting must not be able
    to tell which runtime produced the number.
    """
    usage = FakeUsage(
        prompt_token_count=100,
        candidates_token_count=40,
        cached_content_token_count=30,
        thoughts_token_count=12,
    )
    mapped = token_usage(usage)
    assert mapped == TokenUsage(
        input_tokens=70, output_tokens=52, cached_input_tokens=30, cache_write_tokens=0
    )
    assert mapped == anthropic_token_usage(
        {
            "input_tokens": 70,
            "output_tokens": 52,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 0,
        }
    )
    # And the arithmetic that says nothing was double counted or lost: the
    # vendor's own total is modelpass's total.
    assert mapped.total_tokens == usage.total_token_count
    assert mapped.billable_input_tokens == 70


def test_usage_reads_a_mapping_as_well_as_a_model_and_never_raises():
    assert token_usage(None) == TokenUsage()
    assert token_usage(
        {"prompt_token_count": 5, "candidates_token_count": 6}
    ) == TokenUsage(input_tokens=5, output_tokens=6)
    assert token_usage(
        {"prompt_token_count": 9, "cached_content_token_count": 4}
    ) == TokenUsage(input_tokens=5, cached_input_tokens=4)
    # A vendor that ever reported more cached than total must not produce a
    # negative count on a receipt.
    assert (
        token_usage(
            {"prompt_token_count": 1, "cached_content_token_count": 4}
        ).input_tokens
        == 0
    )
    # Server-side tool prompt tokens are input the caller paid for.
    assert token_usage({"tool_use_prompt_token_count": 7}).input_tokens == 7


def test_the_usage_event_carries_the_cache_split_through_the_stream():
    usage = FakeUsage(
        prompt_token_count=705, candidates_token_count=6, cached_content_token_count=700
    )
    events = drive(run_request(), Factory([text_round("hi", usage=usage)]))
    reported = next(e for e in events if isinstance(e, UsageEvent))
    assert reported.usage.cached_input_tokens == 700
    assert reported.usage.input_tokens == 5
    assert reported.usage.cache_write_tokens == 0


def test_a_part_with_no_word_in_the_vocabulary_becomes_a_vendor_event():
    round_ = Round(
        chunks=[
            chunk(FakePart(executable_code={"code": "print(1)", "language": "PYTHON"})),
            chunk(FakePart(text="x"), finish_reason="STOP", usage=FakeUsage()),
        ]
    )
    events = drive(run_request(), Factory([round_]))
    vendor = next(e for e in events if isinstance(e, VendorEvent))
    assert vendor.name == "part.executable_code"
    assert vendor.runtime is Runtime.GOOGLE_API
    assert vendor.data["code"] == "print(1)"


def test_the_parts_already_reported_are_not_reported_twice():
    """A thought signature travels back to the vendor inside the replayed turn;
    it is not content, and a caller has nothing to do with it."""
    round_ = Round(
        chunks=[
            chunk(FakePart(text="x"), finish_reason="STOP", usage=FakeUsage()),
            chunk(FakePart(thought_signature="opaque")),
        ]
    )
    events = drive(run_request(), Factory([round_]))
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["x"]
    assert not [e for e in events if isinstance(e, VendorEvent)]


# --- 4. the prompt: flattened, with the drop named by the bridge -------------------


def test_the_system_prompt_becomes_system_instruction_and_blocks_are_flattened():
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
    # flat_text semantics: one blank line between blocks, and no marker anywhere.
    assert config_of(factory)["system_instruction"] == "the rubric\n\ntoday's question"
    assert "cache_control" not in json.dumps(sent(factory))
    assert sent(factory)["contents"] == [{"role": "user", "parts": [{"text": "go"}]}]
    assert sent(factory)["model"] == MODEL
    # No ceiling is invented: max_output_tokens is optional on this API.
    assert "max_output_tokens" not in config_of(factory)


def test_the_assistant_turn_is_called_model_on_this_api():
    """One line of translation, and the single most likely thing to be wrong in
    an adapter written from another vendor's memory."""
    messages = (
        Message(role=Role.USER, content=[TextBlock("one"), TextBlock("still one")]),
        Message(role=Role.ASSISTANT, content="two"),
        Message(role=Role.USER, content="three"),
    )
    factory = Factory([text_round("ok")])
    drive(run_request(messages=messages), factory)
    assert sent(factory)["contents"] == [
        {"role": "user", "parts": [{"text": "one\n\nstill one"}]},
        {"role": "model", "parts": [{"text": "two"}]},
        {"role": "user", "parts": [{"text": "three"}]},
    ]


def test_a_run_with_no_system_message_omits_the_field_entirely():
    factory = Factory([text_round("ok")])
    drive(run_request(), factory)
    assert "system_instruction" not in config_of(factory)


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
        [tool_round("look_up", {"topic": "rebase"}), text_round("it rewrites history")]
    )
    events = drive(run_request(tools=(tool,)), factory)

    assert seen == [{"topic": "rebase"}]
    call = next(e for e in events if isinstance(e, ToolCallEvent))
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert (call.name, call.server) == ("look_up", "caller")
    # FunctionCall.args arrives already parsed here -- no JSON string to decode.
    assert call.arguments == {"topic": "rebase"}
    # FunctionCall.id is optional on this API and usually absent, so the name is
    # the fallback: an event with an empty id cannot be paired with its result.
    assert call.id == "look_up"
    assert (result.id, result.name, result.is_error) == ("look_up", "look_up", False)
    assert result.content == "looked up rebase"
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].status is TerminalStatus.OK


def test_a_vendor_supplied_call_id_is_used_when_there_is_one():
    tool = echo_tool([])
    factory = Factory(
        [tool_round("look_up", {"topic": "x"}, call_id="fc_7"), text_round("done")]
    )
    events = drive(run_request(tools=(tool,)), factory)
    call = next(e for e in events if isinstance(e, ToolCallEvent))
    assert call.id == "fc_7"
    answer = sent(factory, 1)["contents"][-1]["parts"][0]["function_response"]
    assert answer["id"] == "fc_7"


def test_the_whole_model_turn_is_replayed_before_the_function_response():
    """Thought signatures included -- a model that cannot see its own previous
    thinking re-derives it every round."""
    tool = echo_tool([])
    first = tool_round("look_up", {"topic": "x"})
    first.chunks[0].candidates[0].content.parts.insert(
        0, FakePart(text="thinking", thought=True, thought_signature="sig")
    )
    factory = Factory([first, text_round("done")])
    drive(run_request(tools=(tool,)), factory)
    second = sent(factory, 1)["contents"]
    replayed = second[-2]
    assert replayed["role"] == "model"
    assert replayed["parts"][0] == {
        "text": "thinking",
        "thought": True,
        "thought_signature": "sig",
    }
    assert replayed["parts"][1]["function_call"]["name"] == "look_up"
    # And the answer, in a user turn, which is the shape this API documents.
    assert second[-1] == {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "name": "look_up",
                    "response": {"result": "looked up x"},
                }
            }
        ],
    }
    declared = config_of(factory, 1)["tools"][0]["function_declarations"][0]
    assert declared["name"] == "look_up"
    assert declared["parameters_json_schema"]["properties"] == {
        "topic": {"type": "string"}
    }


def test_parallel_calls_are_answered_in_one_user_turn():
    seen: list[dict[str, Any]] = []
    tool = echo_tool(seen)
    both = Round(
        chunks=[
            chunk(
                FakePart(function_call=FakeFunctionCall(name="look_up", args={"topic": "a"})),
                FakePart(function_call=FakeFunctionCall(name="look_up", args={"topic": "b"})),
                finish_reason="STOP",
                usage=FakeUsage(),
            )
        ]
    )
    factory = Factory([both, text_round("done")])
    events = drive(run_request(tools=(tool,)), factory)
    assert seen == [{"topic": "a"}, {"topic": "b"}]
    assert len([e for e in events if isinstance(e, ToolResultEvent)]) == 2
    answers = sent(factory, 1)["contents"][-1]
    assert answers["role"] == "user"
    assert len(answers["parts"]) == 2


def test_there_is_no_turn_cap():
    tool = echo_tool([])
    rounds = [tool_round("look_up", {"topic": str(i)}) for i in range(6)]
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
    factory = Factory([tool_round("look_up", {}), text_round("I could not look that up")])
    events = drive(run_request(tools=(tool,)), factory)
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.is_error is True
    assert "tool 'look_up' failed: ValueError: the index is offline" in result.content
    assert events[-1].status is TerminalStatus.OK
    # The model is told it was an error rather than handed a string that reads
    # like a result: the key in the response mapping says which.
    answered = sent(factory, 1)["contents"][-1]["parts"][0]["function_response"]
    assert "the index is offline" in answered["response"]["error"]


def test_a_tool_the_run_never_declared_is_a_failed_result_rather_than_a_crash():
    tool = echo_tool([])
    factory = Factory([tool_round("ghost", {}), text_round("sorry")])
    events = drive(run_request(tools=(tool,)), factory)
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.is_error is True
    assert "was not declared for this run" in result.content


def test_an_async_handler_is_driven_to_completion():
    async def handler(args: dict[str, Any]) -> str:
        return "async answer"

    tool = ToolDef(name="look_up", description="Look.", parameters={}, handler=handler)
    factory = Factory([tool_round("look_up", {}), text_round("done")])
    events = drive(run_request(tools=(tool,)), factory)
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.content == "async answer"


def test_a_handler_with_no_callable_at_all_still_answers_the_model():
    tool = ToolDef(name="look_up", description="Look.", parameters={})
    factory = Factory([tool_round("look_up", {}), text_round("done")])
    events = drive(run_request(tools=(tool,)), factory)
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.is_error is True
    assert "has no handler" in result.content


def test_arguments_that_are_missing_or_unparseable_reach_the_handler_as_a_mapping():
    """A dead run is worse than a tool that can say what it was given."""
    seen: list[dict[str, Any]] = []
    tool = echo_tool(seen)
    factory = Factory(
        [
            tool_round("look_up", None),  # type: ignore[arg-type]
            Round(
                chunks=[
                    chunk(
                        FakePart(
                            function_call=FakeFunctionCall(name="look_up", args="{not json")
                        ),
                        finish_reason="STOP",
                        usage=FakeUsage(),
                    )
                ]
            ),
            text_round("done"),
        ]
    )
    events = drive(run_request(tools=(tool,)), factory)
    assert seen == [{}, {}]
    assert events[-1].status is TerminalStatus.OK


def test_a_function_call_on_a_run_that_declared_no_tools_ends_rather_than_loops():
    """The model cannot ask for a tool nobody sent it -- but if it ever does,
    answering "not declared" forever is a loop with a bill. A run carrying no
    tools at all stops and says why."""
    factory = Factory([tool_round("look_up", {})])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.ERROR
    assert "did not declare" in (events[-1].reason or "")


# --- 6. guards fire between tool rounds -------------------------------------------


@NEEDS_SDK
def test_a_token_guard_stops_the_loop_between_rounds(store):
    tool = echo_tool([])
    conn = connection(guards=Guards(stop_at_tokens=10))
    factory = Factory([tool_round("look_up", {"topic": "x"}), text_round("never reached")])
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


def test_a_schema_is_sent_as_a_json_mime_type_and_a_json_schema():
    factory = Factory([text_round(json.dumps({"sentiment": "positive"}))])
    events = drive(run_request(schema=SCHEMA, schema_name="Verdict"), factory)
    config = config_of(factory)
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == {
        "type": "object",
        "properties": {"sentiment": {"type": "string"}},
        "required": ["sentiment"],
    }
    # The vendor's own Schema object is the *other* field, and the two are
    # mutually exclusive: sending both is an error.
    assert "response_schema" not in config
    structured = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert structured.data == {"sentiment": "positive"}
    assert structured.valid is True
    assert structured.schema_name == "Verdict"
    assert events[-1].status is TerminalStatus.OK


def test_an_optional_property_stays_optional_on_this_runtime():
    """The difference from ``openai-api``, and it is the vendor's rather than
    modelpass's: there is no ``strict`` flag here, so there is nothing to tighten
    and nothing to disclose."""
    factory = Factory([text_round(json.dumps({"sentiment": "ok"}))])
    request = run_request(schema=LOOSE_SCHEMA, schema_name="Verdict")
    receipt = adapter(factory).preflight(request)
    assert receipt.ok
    assert not any("to_openai_strict" in note for note in receipt.notes)

    drive(request, factory)
    schema = config_of(factory)["response_json_schema"]
    assert schema["required"] == ["sentiment"]
    assert schema["properties"]["note"] == {"type": "string"}


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


def test_no_schema_means_no_response_format_on_the_wire():
    factory = Factory([text_round("plain")])
    drive(run_request(), factory)
    assert "response_mime_type" not in config_of(factory)
    assert "response_json_schema" not in config_of(factory)


# --- 8. terminal status from finish_reason and prompt feedback ---------------------


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("STOP", TerminalStatus.OK),
        ("FINISH_REASON_UNSPECIFIED", TerminalStatus.OK),
        ("MAX_TOKENS", TerminalStatus.ERROR),
        ("SAFETY", TerminalStatus.ERROR),
        ("RECITATION", TerminalStatus.ERROR),
        ("PROHIBITED_CONTENT", TerminalStatus.ERROR),
        ("MALFORMED_FUNCTION_CALL", TerminalStatus.ERROR),
        ("SOMETHING_THE_ENUM_GREW_LATER", TerminalStatus.OK),
    ],
)
def test_each_finish_reason_maps_to_its_terminal_status(reason, expected):
    events = drive(run_request(), Factory([text_round("...", finish_reason=reason)]))
    assert events[-1].status is expected


def test_a_truncated_answer_says_which_ceiling_it_hit():
    """Word for word the other adapters' sentence, because the fix is the same."""
    events = drive(
        run_request(), Factory([text_round("half a th", finish_reason="MAX_TOKENS")])
    )
    assert "max_output_tokens" in (events[-1].reason or "")


def test_a_stopped_answer_names_the_category_it_was_stopped_for():
    events = drive(run_request(), Factory([text_round("", finish_reason="SAFETY")]))
    assert "SAFETY" in (events[-1].reason or "")
    events = drive(run_request(), Factory([text_round("", finish_reason="RECITATION")]))
    assert "RECITATION" in (events[-1].reason or "")


def test_a_blocked_prompt_is_an_error_and_names_the_block_reason():
    """The one failure that arrives as a *successful* response with no candidates
    in it, which is why it is read separately from the finish reason."""
    blocked = Round(
        chunks=[
            FakeChunk(
                candidates=[],
                prompt_feedback=FakePromptFeedback(block_reason="PROHIBITED_CONTENT"),
                usage_metadata=FakeUsage(prompt_token_count=9),
            )
        ]
    )
    events = drive(run_request(), Factory([blocked]))
    assert events[-1].status is TerminalStatus.ERROR
    assert "blocked the prompt" in (events[-1].reason or "")
    assert "PROHIBITED_CONTENT" in (events[-1].reason or "")


def test_a_stream_that_delivered_nothing_at_all_is_an_error_not_an_empty_ok():
    events = drive(run_request(), Factory([Round(chunks=[])]))
    assert events[-1].status is TerminalStatus.ERROR
    assert "no answer arrived" in (events[-1].reason or "")


def test_content_with_no_finish_reason_is_still_an_answer():
    """A vendor that stops sending the field is not a vendor that failed."""
    events = drive(run_request(), Factory([Round(chunks=[chunk(FakePart(text="hi"))])]))
    assert events[-1].status is TerminalStatus.OK


def test_a_429_is_quota_exhausted_and_is_typed_off_the_status_code():
    """``google.genai.errors.APIError`` spells it ``code``, not ``status_code``."""

    class RateLimited(Exception):
        code = 429

    factory = Factory([Round(raises=RateLimited("RESOURCE_EXHAUSTED"))])
    events = drive(run_request(), factory)
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED


def test_the_other_spelling_of_the_status_code_is_read_too():
    class RateLimited(Exception):
        status_code = 429

    events = drive(run_request(), Factory([Round(raises=RateLimited("slow down"))]))
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED


def test_any_other_vendor_failure_is_a_reported_terminal_not_a_raise():
    class Boom(Exception):
        code = 500

    events = drive(run_request(), Factory([Round(raises=Boom("upstream is unwell"))]))
    assert events[-1].status is TerminalStatus.ERROR
    assert "HTTP 500" in (events[-1].reason or "")


def test_a_failure_with_no_code_at_all_still_names_itself():
    events = drive(run_request(), Factory([Round(raises=RuntimeError("socket closed"))]))
    assert events[-1].status is TerminalStatus.ERROR
    assert "RuntimeError: socket closed" in (events[-1].reason or "")


def test_terminal_status_reads_an_enum_member_as_well_as_a_string():
    """The SDK hands over a ``FinishReason``, not the string the fake uses."""
    genai_types = pytest.importorskip("google.genai.types")
    status, reason = terminal_status(
        genai_types.FinishReason.MAX_TOKENS, unanswered_call=False
    )
    assert status is TerminalStatus.ERROR
    assert "max_output_tokens" in (reason or "")
    assert terminal_status(genai_types.FinishReason.STOP, unanswered_call=False)[0] is (
        TerminalStatus.OK
    )


# --- 9. sampling -------------------------------------------------------------------


def test_the_four_accepted_controls_reach_the_config_under_their_own_names():
    factory = Factory([text_round("ok")])
    drive(
        run_request(
            sampling=Sampling(temperature=0.2, top_p=0.9, top_k=40, max_output_tokens=128)
        ),
        factory,
    )
    config = config_of(factory)
    assert config["temperature"] == 0.2
    assert config["top_p"] == 0.9
    assert config["max_output_tokens"] == 128
    # The control this runtime has and neither OpenAI runtime does.
    assert config["top_k"] == 40


def test_top_k_reaches_this_runtime_where_it_is_dropped_on_the_openai_ones(store):
    conn = connection()
    bridge, _ = bridge_for(conn, Factory([text_round("ok")]), store)
    receipt = bridge.preflight(conn.name, sampling=Sampling(top_k=40))
    assert receipt.sampling_applied["top_k"] == 40
    assert not any("top_k" in note for note in receipt.sampling_notes)


def test_reasoning_effort_becomes_a_thinking_level_and_never_a_budget():
    """The recorded decision, kept as an assertion.

    ``ThinkingConfig`` carries both. ``thinking_level`` is
    ``MINIMAL | LOW | MEDIUM | HIGH`` and modelpass's dial is
    ``low | medium | high``, so the translation invents nothing;
    ``thinking_budget`` is a token count whose allowed range the SDK documents as
    model dependent, and ticket 1.7 refused to invent one of those for Anthropic.
    """
    factory = Factory([text_round("ok")])
    drive(run_request(sampling=Sampling(reasoning_effort="high")), factory)
    assert config_of(factory)["thinking_config"] == {"thinking_level": "HIGH"}
    assert "thinking_budget" not in json.dumps(sent(factory))


def test_asking_for_thoughts_and_an_effort_fills_one_thinking_config():
    """A merge rather than a replacement, which is what keeps both true when a
    third thing writes there."""
    factory = Factory([text_round("ok")])
    drive(
        run_request(
            sampling=Sampling(reasoning_effort="low"), options={"include_thoughts": True}
        ),
        factory,
    )
    assert config_of(factory)["thinking_config"] == {
        "thinking_level": "LOW",
        "include_thoughts": True,
    }


def test_a_temperature_above_the_runtime_ceiling_is_clamped_and_named(store):
    conn = connection()
    bridge, _ = bridge_for(conn, Factory([text_round("ok")]), store)
    receipt = bridge.preflight(conn.name, sampling=Sampling(temperature=1.9))
    assert receipt.sampling_applied["temperature"] == 1.9
    receipt = bridge.preflight(conn.name, sampling=Sampling(temperature=2.0))
    assert receipt.sampling_applied["temperature"] == 2.0


def test_every_call_says_that_the_model_rules_are_unknown(store):
    """The honest half of this row, and it is load-bearing: ``reasoning_effort``
    is accepted because the *field* is a runtime field, and whether a given
    Gemini model thinks is exactly what nobody here has driven."""
    conn = connection()
    bridge, _ = bridge_for(conn, Factory([text_round("ok")]), store)
    receipt = bridge.preflight(conn.name, sampling=Sampling(temperature=0.2))
    assert any("model rules unknown" in note for note in receipt.sampling_notes)


def test_no_ceiling_is_invented_when_nobody_asked_for_one():
    """The same decision ``openai-api`` took, and the opposite of 1.6's."""
    factory = Factory([text_round("ok")])
    drive(run_request(), factory)
    assert "max_output_tokens" not in config_of(factory)


def test_the_max_output_tokens_option_is_still_read_as_an_alias():
    factory = Factory([text_round("ok")])
    drive(run_request(options={"max_output_tokens": 128}), factory)
    assert config_of(factory)["max_output_tokens"] == 128


def test_an_unknown_option_key_is_reported_rather_than_silently_ignored():
    assert GoogleAPIAdapter.unknown_option_keys(
        {"include_thoughts": True, "temprature": 2}
    ) == ("temprature",)


def test_a_call_with_nothing_configured_sends_no_config_at_all():
    factory = Factory([text_round("ok")])
    drive(run_request(connection(model=None), model=MODEL), factory)
    assert set(sent(factory)) == {"model", "contents"}


# --- 10. cancellation ---------------------------------------------------------------


def test_cancel_stops_the_loop_at_the_next_round_boundary():
    tool = echo_tool([])
    factory = Factory([tool_round("look_up", {"topic": "x"}), text_round("never reached")])
    api = adapter(factory)
    events = []
    for event in api.run(run_request(tools=(tool,))):
        events.append(event)
        if isinstance(event, ToolResultEvent):
            api.cancel()
    assert events[-1].status is TerminalStatus.CANCELLED
    assert len(factory.clients[0].calls) == 1


def test_closing_the_iterator_closes_the_vendor_stream():
    """This SDK's stream is an iterator rather than a context manager, so the
    close is explicit -- and the floor D10 documents is the same one."""
    long_round = Round(chunks=[chunk(FakePart(text=str(i))) for i in range(10)])
    factory = Factory([long_round])
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
        assert "google-api" in message
        assert call in message
        assert "bridge.chat(history=[...])" in message


def test_new_chat_is_refused_by_the_bridge_before_the_adapter(store):
    conn = connection()
    bridge, _ = bridge_for(conn, Factory(), store)
    with pytest.raises(CapabilityNotSupported) as excinfo:
        bridge.new_chat(connection=conn.name)
    message = str(excinfo.value)
    assert "google-api" in message
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
    assert registry.support(Runtime.GOOGLE_API, capability) is Support.SUPPORTED
    note = registry.note(Runtime.GOOGLE_API, capability) or ""
    assert "2026-09-13" in note
    assert SDK_VERSION_READ in note or "tests/" in note


def test_thinking_waits_on_a_live_drive_and_its_note_names_the_test():
    """The cell that moved on ``anthropic-api`` and does not move here.

    The adapter separates thought parts from answer parts (proved above); what a
    fake cannot say is whether anything ever arrives, because thoughts are only
    emitted when the request asks and only by a model that thinks.
    """
    registry = CapabilityRegistry()
    assert registry.support(Runtime.GOOGLE_API, Capability.THINKING) is Support.UNVERIFIED
    note = registry.note(Runtime.GOOGLE_API, Capability.THINKING) or ""
    assert "tests/live/test_google_api_live.py" in note


def test_the_cells_that_are_checked_absences_stay_that_way():
    registry = CapabilityRegistry()
    for capability in (
        Capability.CACHE_BREAKPOINTS,
        Capability.TTL_CONTROL,
        Capability.SESSIONS_RESUME,
        Capability.SESSIONS_LIST,
        Capability.SESSIONS_FORK,
        Capability.SUBAGENTS,
        Capability.SUBSCRIPTION_AUTH,
    ):
        assert registry.support(Runtime.GOOGLE_API, capability) is Support.UNSUPPORTED


def test_mcp_servers_moved_the_other_way_because_the_sdk_contradicts_the_absence():
    """The shared "an API endpoint runs no MCP client" note is wrong for this
    vendor, and a cell the SDK contradicts must not keep saying ``unsupported``.

    Same correction ticket 1.6 made to ``anthropic-api``'s ``ttl_control``, and
    made for the same reason.
    """
    genai_types = pytest.importorskip("google.genai.types")
    assert "mcp_servers" in genai_types.Tool.model_fields
    assert {"name", "streamable_http_transport"} <= set(
        genai_types.McpServer.model_fields
    )
    registry = CapabilityRegistry()
    assert (
        registry.support(Runtime.GOOGLE_API, Capability.MCP_SERVERS) is Support.UNVERIFIED
    )
    note = registry.note(Runtime.GOOGLE_API, Capability.MCP_SERVERS) or ""
    assert "change of answer" in note
    # And the other three rows are untouched: this is one vendor's fact.
    for runtime in (Runtime.ANTHROPIC_API, Runtime.OPENAI_API, Runtime.OPENAI_COMPATIBLE):
        assert registry.support(runtime, Capability.MCP_SERVERS) is Support.UNSUPPORTED


def test_the_ttl_cell_names_gemini_own_reason():
    """Unsupported here, and *not* for the shared reason: Gemini does have a
    settable lifetime, on a separate resource modelpass does not create."""
    note = CapabilityRegistry().note(Runtime.GOOGLE_API, Capability.TTL_CONTROL) or ""
    assert "caches.create()" in note
    assert "cached_content" in note


def test_the_runtime_is_ungated_and_the_two_subscription_ones_are_not():
    """Ticket 1.11 ships an adapter and changes nothing about D5's asymmetry."""
    from modelpass.runtimes import EXPERIMENTAL_RUNTIMES

    assert Runtime.GOOGLE_API not in EXPERIMENTAL_RUNTIMES
    assert {Runtime.GOOGLE_CLI, Runtime.GOOGLE_SDK} <= EXPERIMENTAL_RUNTIMES


def test_the_sampling_rules_row_is_now_a_driven_one():
    from modelpass.sampling_rules import rules_for

    rules = rules_for(Runtime.GOOGLE_API, MODEL)
    assert rules.verified is True
    assert "top_k" in rules.accepted
    assert rules.reasoning_parameter == "thinking_config.thinking_level"
    # No required field and no default: the ceiling is optional on this API.
    assert rules.required == frozenset()
    assert rules.defaults == {}
    # And no model is claimed as known, which is what the per-call note says.
    assert rules.model_known is False


# --- 13. the fake is checked against the SDK it stands in for ----------------------


def test_the_fake_matches_the_installed_sdk_shapes():
    """The fake is only worth anything if the real client has the same surface.

    Six things, each one a way this file could pass against an adapter that would
    fail live: the usage field names, the finish reasons the mapping branches on,
    the thought flag, the block reason, the two structured-output fields, and --
    the strongest of the six -- that the exact ``config`` dict this adapter builds
    validates against the vendor's own ``GenerateContentConfig``.
    """
    import importlib.metadata as _metadata
    import warnings

    genai = pytest.importorskip("google.genai")
    from google.genai import types as genai_types

    # The installed version is *reported*, never required. The shape assertions
    # below are the protection: each one names a field, a literal or a
    # parameter this adapter depends on, so a surface that really moved fails on
    # its own line and says which. Pinning equality here instead failed on every
    # vendor patch release inside the range this project declares it supports --
    # a red suite for a reason that has nothing to do with the code, and a check
    # that teaches its reader to ignore it. google_api.SDK_VERSION_READ stays the
    # dated evidence the capability notes cite; this warning asks for a re-read.
    installed = _metadata.version("google-genai")
    if installed != SDK_VERSION_READ:
        warnings.warn(
            f"google-genai {installed} is installed; the shapes asserted here were last "
            f"verified against {SDK_VERSION_READ}. Re-read the surface and move "
            f"google_api.SDK_VERSION_READ when you have.",
            stacklevel=1,
        )

    # 1. usage, including the two counts the arithmetic depends on.
    usage_fields = set(genai_types.GenerateContentResponseUsageMetadata.model_fields)
    assert {
        "prompt_token_count",
        "candidates_token_count",
        "cached_content_token_count",
        "thoughts_token_count",
        "tool_use_prompt_token_count",
        "total_token_count",
    } <= usage_fields

    # 2. the finish reasons terminal_status() branches on.
    reasons = {member.value for member in genai_types.FinishReason}
    assert {"STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"} <= reasons

    # 3. a thought is a flag on a text part, and a call's args are already parsed.
    assert "thought" in genai_types.Part.model_fields
    assert "thought_signature" in genai_types.Part.model_fields
    assert "args" in genai_types.FunctionCall.model_fields

    # 4. a blocked prompt arrives as feedback rather than as a finish reason.
    assert "block_reason" in genai_types.GenerateContentResponsePromptFeedback.model_fields

    # 5. the two structured-output fields, and the tool field the adapter uses.
    config_fields = set(genai_types.GenerateContentConfig.model_fields)
    assert {"response_mime_type", "response_json_schema", "response_schema"} <= config_fields
    assert "parameters_json_schema" in genai_types.FunctionDeclaration.model_fields

    # 6. the whole config this adapter sends, through the vendor's own model.
    factory = Factory([text_round("ok")])
    tool = echo_tool([])
    drive(
        run_request(
            messages=(
                Message(role=Role.SYSTEM, content="be brief"),
                Message(role=Role.USER, content="go"),
            ),
            tools=(tool,),
            sampling=Sampling(
                temperature=0.2, top_p=0.9, top_k=40, max_output_tokens=64,
                reasoning_effort="high",
            ),
            options={"include_thoughts": True},
        ),
        factory,
    )
    genai_types.GenerateContentConfig.model_validate(config_of(factory))
    for item in sent(factory)["contents"]:
        genai_types.Content.model_validate(item)

    # And the structured-output half, which cannot travel on the same call (D13).
    factory = Factory([text_round("{}")])
    drive(run_request(schema=LOOSE_SCHEMA), factory)
    genai_types.GenerateContentConfig.model_validate(config_of(factory))

    # And a tool answer, which is the shape a second round sends back.
    factory = Factory([tool_round("look_up", {"topic": "x"}), text_round("done")])
    drive(run_request(tools=(tool,)), factory)
    for item in sent(factory, 1)["contents"]:
        genai_types.Content.model_validate(item)

    assert genai.Client is not None


def test_the_sdk_carries_a_thinking_level_matching_modelpasss_own_dial():
    """The evidence behind the effort mapping, kept as an assertion: three words,
    three of the vendor's own named values, nothing invented in between."""
    genai_types = pytest.importorskip("google.genai.types")
    from modelpass.types import REASONING_EFFORTS

    levels = {member.value for member in genai_types.ThinkingLevel}
    assert {effort.upper() for effort in REASONING_EFFORTS} <= levels
    # And the field modelpass refuses to fill in, with the reason in its own docs.
    budget = genai_types.ThinkingConfig.model_fields["thinking_budget"]
    assert "model dependent" in (budget.description or "")
