"""The ``openai-compatible`` adapter: Chat Completions, driven in this process
(ticket 1.10).

The third of the four API-key adapters, and the only one whose capability row is
a fact about an **install** rather than about a vendor. ``anthropic-api`` (1.6)
and ``openai-api`` (1.9) each speak to one company's endpoint; this runtime names
an API *shape*, and what answers is whatever the connection's ``baseUrl`` points
at -- Ollama, LM Studio, vLLM, a LiteLLM proxy, OpenRouter, a gateway. Two
installs can legitimately disagree about every cell, which is why the static row
is ``unverified`` by construction and why :func:`~modelpass.cli.main`'s ``verify``
subcommand drives this endpoint and records what it found on the connection.

**Chat Completions by default, and the reason is the opposite of 1.9's**
(recorded in ``docs/api-and-runtimes.md`` §2.0c). ``openai-api`` chose Responses
because its
*event vocabulary* is richer: reasoning text is a stream event there and a token
count on the other transport. None of that reasoning survives the move to this
runtime, because the question here is not "which surface says more" but "which
surface is actually implemented by the thing on the other end":

* **Chat Completions is the lowest common denominator, and it is not close.**
  Ollama, LM Studio, vLLM, llama.cpp's server, LiteLLM's proxy and OpenRouter all
  implement ``POST /v1/chat/completions``. Responses is implemented by OpenAI and
  by a shrinking minority of the rest. An adapter that defaulted to Responses
  would 404 against the commonest configuration this runtime exists for.
* **The richer vocabulary has nothing to be rich about here.** The one thing
  Responses can report that Chat Completions cannot is reasoning *text*, and the
  servers behind this runtime mostly do not produce any; where they do, they
  produce it in a vendor extension nobody agrees on.
* **So the trade runs the other way**, and the two costs are paid openly rather
  than hidden: ``finish_reason`` is a coarser terminal vocabulary than
  ``status`` + ``incomplete_details`` (see :func:`terminal_status`), and the
  max-tokens field is spelled ``max_tokens`` on the wire while modelpass calls it
  ``max_output_tokens`` (see :meth:`OpenAICompatibleAdapter.chat_sampling_params`).

**Responses is still reachable, per connection, through one option key.**
``options={"wire": "responses"}`` runs ticket 1.9's code path **unchanged** --
literally: this adapter subclasses :class:`~modelpass.adapters.openai_api.OpenAIAPIAdapter`
and that branch is one ``super().run(request)``. It exists because a caller who
points this runtime at a gateway that *does* implement Responses (OpenAI's own
endpoint through a proxy, a LiteLLM passthrough) should not have to give up
reasoning text to use it. It is a per-connection choice rather than a default
because modelpass cannot know which endpoint is behind a URL and guessing the
richer one fails loudly against the common case.

**Why a subclass rather than a shared transport module.** The alternative
factoring -- one module with a transport switch, imported by both runtimes -- was
considered and rejected for the reason the plan gave for the whole ticket: it
would mean editing ``openai_api.py``'s run path, and 1.9's behaviour and its
tests are the thing this ticket must not disturb. What is genuinely shared is
already shared, by import: the tool loop's rules (``invoke_handler``), the
credential handling, the preflight, the probe, the session refusals, the 429
typing. What differs is one transport, and it lives here. Three additive changes
were made to ``openai_api.py`` and no behavioural one: two helpers take a
``runtime`` argument that defaults to their old constant, and the no-model
refusal became a class attribute so this subclass can say its own sentence.

**A connection here may carry no credential at all**, and this is the one runtime
where that is true. ``credentialRef = "none"`` says the endpoint authenticates
nobody, which is the plain truth about a local Ollama or LM Studio box.
:func:`~modelpass.preflight.api_preflight` reports it on the receipt in those
words; the client is still constructed **explicitly**, with
:data:`PLACEHOLDER_CREDENTIAL` in the ``api_key=`` slot the SDK requires, so D2's
in-process clause holds exactly as it does on the other three -- nothing ambient
in this process's environment is ever read. ``env:`` and ``secret:`` work here
the way they work everywhere else, and a proxy or a gateway will want one.

**What it deliberately does not do.** Sessions, for ticket 1.8's reason: an API
endpoint holds no history. ``cache_breakpoints``: the OpenAI-shaped request
schema has no ``cache_control``, whatever the server does with prefixes of its
own accord. ``top_k``: it is in neither of ``openai`` 2.32.0's create signatures,
the servers that accept one disagree about where it goes, and the sampling rules
row drops it with a note rather than guessing an ``extra_body`` key that would
ride along on every endpoint that does not take one.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, ClassVar

from ..capabilities import Capability, VerifiedCapabilities, VerifyReport
from ..connections import Connection, CredentialKind
from ..errors import (
    AdapterNotImplemented,
    PreflightFailed,
    RuntimeNotAvailable,
    StructuredOutputRejected,
    SubpassError,
    VendorRunFailed,
)
from ..preflight import ModelListProbe, Receipt, resolve_credential
from ..retry import vendor_error_facts
from ..runtimes import Runtime
from ..sampling_rules import plan_sampling
from ..schema import (
    build_structured_event,
    normalize_schema,
    resolve_schema_name,
    to_openai_strict,
)
from ..tools import ToolDef
from ..types import (
    CALLER_TOOL_SERVER,
    AgentEvent,
    Message,
    Role,
    SessionInfo,
    StructuredOutputEvent,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    UsageScope,
    VendorEvent,
)
from ._mcp_result import ainvoke_handler, flatten_mcp_result
from .base import RunCancel, RunRequest, RunStream, SessionHandle, SessionRequest, native_run
from .openai_api import (
    OpenAIAPIAdapter,
    _bare_terminal,
    _missing_tool,
    _vendor_failure,
    _vendor_payload,
    flatten,
    invoke_handler,
)

__all__ = [
    "PLACEHOLDER_CREDENTIAL",
    "SDK_VERSION_READ",
    "OpenAICompatibleAdapter",
    "call_arguments",
    "terminal_status",
    "token_usage",
]

_MODULE = "openai"
_EXTRA = "openai-api"
_PACKAGE = "openai"

#: The installed SDK this adapter was written against, and whose Chat Completions
#: surface every note below cites. Read 2026-09-13.
SDK_VERSION_READ = "2.32.0"

#: What goes in the ``api_key=`` slot when a connection declares no credential.
#:
#: ``openai.OpenAI`` raises without one -- the argument is required and an empty
#: string is not accepted -- so a connection that says "this endpoint checks
#: nothing" still needs *a* string. It is a visible placeholder rather than
#: something key-shaped on purpose: if it ever reaches a server that does check,
#: the rejection message will contain a value whose meaning is obvious, and no
#: reader of a log will mistake it for a real credential that leaked.
PLACEHOLDER_CREDENTIAL = "modelpass-no-credential"

#: How many model ids :meth:`OpenAICompatibleAdapter.probe` keeps. Higher than
#: ``openai-api``'s, because on this runtime the listing is not just proof that a
#: credential is live -- it is the only reliable answer to "what does this box
#: actually serve", and it is written to the connection by ``modelpass verify``.
#: Still bounded: a LiteLLM proxy in front of everything can list hundreds.
_PROBE_LIMIT = 50


# --- usage -----------------------------------------------------------------------


def _count(usage: Any, key: str) -> int:
    """One integer off a usage object or mapping; anything else is zero."""
    if isinstance(usage, Mapping):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _nested(usage: Any, group: str, key: str) -> int:
    """One integer out of ``prompt_tokens_details`` / ``completion_tokens_details``."""
    if isinstance(usage, Mapping):
        inner = usage.get(group)
    else:
        inner = getattr(usage, group, None)
    return _count(inner, key) if inner is not None else 0


def token_usage(usage: Any) -> TokenUsage:
    """Normalize ``CompletionUsage`` onto modelpass tokens.

    Same arithmetic ``openai_api.token_usage`` does, against the other
    transport's field names: Chat Completions spells the prompt total
    ``prompt_tokens`` and the answer ``completion_tokens``, with
    ``prompt_tokens_details.cached_tokens`` beside them. The cached prefix is
    *inside* ``prompt_tokens`` on the wire and is counted **separately** in
    :class:`~modelpass.types.TokenUsage`, so this subtracts -- and a consumer's
    accounting does not learn that a run changed transports, let alone runtimes.

    ``cache_write_tokens`` is ``0``: no OpenAI-shaped server reports one, because
    none of them bill a write premium for an automatic prefix cache. A server
    that reports nothing at all is a different case and is not this function's
    to answer -- see :meth:`OpenAICompatibleAdapter._chat_loop`, which emits no
    usage event at all rather than a zeroed one.
    """
    cached = _nested(usage, "prompt_tokens_details", "cached_tokens")
    total_input = _count(usage, "prompt_tokens")
    return TokenUsage(
        input_tokens=max(total_input - cached, 0),
        output_tokens=_count(usage, "completion_tokens"),
        cached_input_tokens=cached,
        cache_write_tokens=0,
        # ``completion_tokens_details.reasoning_tokens`` is Optional on
        # ``CompletionUsage`` (openai 2.32.0) and most compatible servers omit
        # the group entirely, which is exactly why absent reads as ``None``
        # here: a local Ollama that reports no reasoning has not told us the
        # model did none.
        reasoning_output_tokens=_reported_nested(
            usage, "completion_tokens_details", "reasoning_tokens"
        ),
    )


def _reported(usage: Any, key: str) -> int | None:
    """One integer off a usage object or mapping, or ``None`` when absent.

    The sibling of :func:`_count` and deliberately not a wrapper of it: a
    reasoning count must keep *no report* distinct from *zero tokens*, which is
    the one distinction ``or 0`` destroys. See
    :attr:`~modelpass.types.TokenUsage.reasoning_output_tokens`.
    """
    if isinstance(usage, Mapping):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _reported_nested(usage: Any, group: str, key: str) -> int | None:
    """:func:`_reported`, one level in. ``None`` when the group itself is absent."""
    if isinstance(usage, Mapping):
        inner = usage.get(group)
    else:
        inner = getattr(usage, group, None)
    return _reported(inner, key) if inner is not None else None


def has_usage(usage: Any) -> bool:
    """Whether the server reported usage at all.

    The question exists because ``stream_options={"include_usage": True}`` is a
    *request*, and a compatible server is free to ignore it -- several do. A
    zeroed :class:`~modelpass.types.TokenUsage` would say "this run cost
    nothing", which is a claim about the bill; ``usage=None`` on the result says
    "nobody told us", which is the truth. So the adapter emits no
    :class:`~modelpass.types.UsageEvent` where this is ``False``.
    """
    if usage is None:
        return False
    names = ("prompt_tokens", "completion_tokens", "total_tokens")
    if isinstance(usage, Mapping):
        return any(name in usage for name in names)
    return any(getattr(usage, name, None) is not None for name in names)


# --- request assembly ------------------------------------------------------------


def messages_param(request: RunRequest) -> list[dict[str, Any]]:
    """The ``messages=`` array: the system prompt first, then the conversation.

    ``{"role": "system", "content": "<text>"}`` as the **first** message, which is
    where this transport puts a system prompt -- there is no ``instructions=``
    field of its own. The blocks are flattened with
    :attr:`~modelpass.types.Message.flat_text` semantics by
    :func:`~modelpass.adapters.openai_api.flatten`, the same join ticket 1.5 chose
    for every runtime that cannot carry blocks; ``cache_control`` markers have
    nowhere to go and the bridge's own disclosure names the drop.

    ``role: "system"`` rather than ``"developer"``: the newer spelling is
    OpenAI's own and a compatible server is under no obligation to know it, while
    ``system`` has been in this shape since the beginning and every server
    implements it. That is this whole module's tie-breaker, applied once more.
    """
    out: list[dict[str, Any]] = []
    system = flatten(request.system_blocks)
    if system:
        out.append({"role": "system", "content": system})
    out.extend(
        {"role": role, "content": flatten(blocks)}
        for role, blocks in request.conversation_blocks
    )
    return out


def tool_params(tools: Sequence[ToolDef]) -> list[dict[str, Any]]:
    """Caller tools as ``tools=`` entries, in the Chat Completions nested shape.

    ``{"type": "function", "function": {"name": ..., "parameters": ...}}`` --
    nested, unlike the Responses flat shape, which is one more reason an adapter
    speaks exactly one transport. ``strict`` is deliberately absent rather than
    ``False``: it is an OpenAI extension to this object, most compatible servers
    have never heard of it, and a key a server does not know is a key it may
    reject. The refusal it would encode is kept anyway -- a tool's parameter
    schema is the caller's own and modelpass does not tighten it.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.json_schema(),
            },
        }
        for tool in tools
    ]


def response_format_param(request: RunRequest) -> dict[str, Any] | None:
    """The native structured-output configuration, or ``None`` (D13).

    ``response_format={"type": "json_schema", "json_schema": {"name": ...,
    "schema": ..., "strict": True}}``, read off ``openai`` 2.32.0 on 2026-09-13.
    The schema goes through :func:`~modelpass.schema.to_openai_strict` for the
    reason ticket 1.9 gave: ``strict: True`` is what makes a server constrain the
    answer at all, and the strict subset has no notion of an optional property.
    What that rewrite did is reported as receipt notes before the call.

    **Many servers reject this object, and that is a first-class outcome here**
    rather than a bug -- see :class:`~modelpass.errors.StructuredOutputRejected`.
    A ``json_schema`` response format is the least widely implemented thing this
    adapter sends; the older ``{"type": "json_object"}`` is more widely taken and
    is *not* substituted for it, because it constrains the answer to be JSON
    without constraining it to be the caller's schema, and quietly delivering
    unvalidated JSON where a schema was asked for is the kind of silent downgrade
    D13 exists to refuse.
    """
    if request.schema is None:
        return None
    normalized = normalize_schema(request.schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": resolve_schema_name(normalized, request.schema_name),
            "schema": to_openai_strict(normalized),
            "strict": True,
        },
    }


# --- event mapping ---------------------------------------------------------------


def _choice(chunk: Any) -> Any | None:
    """The first choice of a streaming chunk, or ``None``.

    ``n`` is never sent, so there is exactly one choice on every chunk that has
    any -- and the final usage chunk has none at all, which is why this is a
    lookup rather than an index.
    """
    choices = getattr(chunk, "choices", None)
    if isinstance(chunk, Mapping):
        choices = chunk.get("choices")
    if not choices:
        return None
    return choices[0]


def _delta(choice: Any) -> Any | None:
    if isinstance(choice, Mapping):
        return choice.get("delta")
    return getattr(choice, "delta", None)


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def chunk_events(chunk: Any, runtime: Runtime) -> list[AgentEvent]:
    """Map one streaming chunk onto the normalized vocabulary.

    Text is the one thing the vocabulary has a word for on this transport.
    **There is deliberately no ``thinking`` arm**: Chat Completions reports
    reasoning as a token count in ``completion_tokens_details.reasoning_tokens``
    and carries no reasoning text anywhere, so a ``ThinkingEvent`` here could
    only ever be invented -- the same sentence ticket 1.9 wrote when it chose the
    other transport, read from the other side.

    Tool-call deltas and the usage chunk are *not* vendor events: both are
    reported another way (as ``tool_call`` once assembled, and as ``usage``), and
    passing them through as well would report the same thing twice. Anything else
    with a shape this adapter does not read becomes ``vendor_event`` named
    ``chunk.<n>`` (rule 3) -- a server's own extension fields reach a caller
    rather than being dropped.
    """
    choice = _choice(chunk)
    if choice is None:
        return []
    delta = _delta(choice)
    content = _field(delta, "content")
    if isinstance(content, str) and content:
        return [TextDeltaEvent(content)]
    if _field(delta, "tool_calls") or _field(delta, "role") or content is not None:
        return []
    if _field(choice, "finish_reason") is not None:
        return []
    return [VendorEvent(runtime, "chunk.unrecognized", _vendor_payload(chunk))]


class _CallBuffer:
    """Tool calls assembled from streaming deltas, keyed by their ``index``.

    The shape this exists for: on this transport a function call does not arrive
    as an object, it arrives as a *sequence of fragments*. The first delta for an
    index usually carries ``id`` and ``function.name`` with an empty
    ``arguments``; every later one carries a slice of the arguments JSON and
    nothing else. Concatenating in index order is the whole algorithm, and
    getting it wrong -- keying on ``id``, which later fragments omit, or on
    position in the list, which repeats -- is the classic way a two-tool turn
    ends up with one tool called twice.
    """

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, Any]] = {}

    def absorb(self, chunk: Any) -> None:
        delta = _delta(_choice(chunk))
        for fragment in _field(delta, "tool_calls") or ():
            index = _field(fragment, "index")
            index = index if isinstance(index, int) else len(self._by_index)
            entry = self._by_index.setdefault(
                index, {"id": "", "name": "", "arguments": ""}
            )
            call_id = _field(fragment, "id")
            if isinstance(call_id, str) and call_id:
                entry["id"] = call_id
            function = _field(fragment, "function")
            name = _field(function, "name")
            if isinstance(name, str) and name:
                entry["name"] = name
            arguments = _field(function, "arguments")
            if isinstance(arguments, str):
                entry["arguments"] += arguments

    def assembled(self) -> list[dict[str, Any]]:
        """The finished calls, in the order the model asked for them."""
        return [self._by_index[index] for index in sorted(self._by_index)]


def call_arguments(raw: str) -> dict[str, Any]:
    """A tool call's arguments, which arrive here as a JSON *string* in pieces.

    The same two rules ``openai_api._call_arguments`` states, against a value
    this transport assembles rather than delivers: unparseable JSON becomes
    ``{}`` rather than an exception, so the handler sees an empty mapping and can
    say so -- a failed tool result the model can act on instead of a dead run.
    That failure mode is *more* likely here than on Responses, because the string
    was concatenated from stream fragments and a truncated turn can end mid-JSON.
    """
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def terminal_status(
    finish_reason: str | None, *, unanswered_call: bool, answered: bool
) -> tuple[TerminalStatus, str | None]:
    """Map ``finish_reason`` onto a terminal status.

    The same five judgements tickets 1.6 and 1.9 made against their vendors' own
    vocabularies, so a consumer matching on a terminal does not learn a third set
    of rules:

    * ``stop`` is ``ok``.
    * **A tool call on a run that declared no tools is ``error``**, checked first
      because such a response is ``tool_calls`` as far as the server is
      concerned and the answer is unfinished either way. Same sentence as 1.6's
      and 1.9's.
    * ``length`` is ``error``, carrying 1.6's wording: the answer is truncated
      mid-sentence, nothing downstream can tell that from a complete one, and the
      fix -- raise the ceiling -- belongs in the message rather than in a doc.
    * ``content_filter`` is ``error``. The server stopped the answer; that is the
      same kind of event as Anthropic's ``refusal``.
    * ``tool_calls`` and ``function_call`` never reach here with a run that
      declared tools: the loop handles them and only calls this once the model
      has stopped asking.

    **A stream that ended with no ``finish_reason`` at all is ``ok`` if text
    arrived and ``error`` if none did**, which is where this diverges from 1.9 and
    the divergence is the transport's. Responses has a terminal *event* whose
    absence is unambiguous; here ``finish_reason`` is a field on a chunk that a
    sloppy server may simply never set, and several do not. Failing every such
    run would break real endpoints that answered correctly, so the honest
    fallback is the observation modelpass can actually make: an answer arrived, or
    it did not.
    """
    if unanswered_call:
        return (
            TerminalStatus.ERROR,
            "the model asked for a tool this run did not declare, so the answer "
            "is unfinished",
        )
    if finish_reason in (None, ""):
        if answered:
            return TerminalStatus.OK, None
        return (
            TerminalStatus.ERROR,
            "the stream ended without a finish_reason and without any text, so no "
            "answer arrived",
        )
    if finish_reason == "stop":
        return TerminalStatus.OK, finish_reason
    if finish_reason == "length":
        return (
            TerminalStatus.ERROR,
            "the answer was truncated at the max_tokens ceiling; raise "
            "max_output_tokens for this call",
        )
    if finish_reason == "content_filter":
        return TerminalStatus.ERROR, "the server's content filter stopped this answer"
    return TerminalStatus.OK, str(finish_reason)


#: Fragments of a rejection message that name the structured-output field.
#: Substring matching, and it is doing the one job substring matching is
#: defensible for: these are the *names of the request fields modelpass sent*,
#: echoed back in an error, not a guess at a vendor's prose. The typing that
#: matters -- is this a 4xx at all -- is read off ``status_code`` (R6).
_SCHEMA_REJECTION_HINTS = (
    "response_format",
    "json_schema",
    "responseformat",
    "structured output",
)


def rejects_response_format(exc: Exception) -> bool:
    """Whether a vendor failure is the endpoint refusing ``response_format``."""
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int) or not 400 <= status_code < 500:
        return False
    text = str(exc).lower()
    return any(hint in text for hint in _SCHEMA_REJECTION_HINTS)


# --- the adapter -----------------------------------------------------------------

#: Said once, because the receipt note and the raise must not drift apart. The
#: advice differs from ``openai-api``'s on purpose: there the fix is to name one
#: of a published catalogue, here it is to ask the box in front of you.
_NO_MODEL = (
    "connection {name!r} names no model and this call passed none. An "
    "OpenAI-compatible endpoint has no default model -- every request names one "
    "-- so there is nothing to send: set a model on the connection, or pass "
    "model= to the call. 'modelpass verify {name}' lists what this endpoint "
    "actually serves"
)

#: The one sentence all three session refusals share.
_NO_SESSIONS = (
    f"the adapter for runtime {Runtime.OPENAI_COMPATIBLE.value!r} has not "
    "implemented {call}, and will not: a session rests on the runtime holding the "
    "conversation (D14) and an OpenAI-shaped endpoint holds none -- every request "
    "carries its whole history. Use bridge.chat(history=[...]), which is the "
    "documented multi-turn shape against a stateless runtime. Stateless "
    "bridge.chat() calls are unaffected"
)


class OpenAICompatibleAdapter(OpenAIAPIAdapter):
    """Drives an OpenAI-shaped endpoint over Chat Completions."""

    runtime = Runtime.OPENAI_COMPATIBLE

    #: A closed set, so an unknown key is *reported* rather than silently
    #: ignored. ``wire`` is this runtime's own: ``"chat"`` (the default) or
    #: ``"responses"``, which runs ticket 1.9's path unchanged.
    #: ``reasoning_summary`` is inherited and reaches only the Responses path --
    #: there is no reasoning text on the default transport for it to ask for.
    option_keys = frozenset(
        {"max_output_tokens", "probe_credential", "reasoning_summary", "wire"}
    )

    no_model_message: ClassVar[str] = _NO_MODEL

    def __init__(
        self,
        *,
        client_factory: Any = None,
        async_client_factory: Any = None,
        env: Mapping[str, str] | None = None,
        secrets: Any = None,
    ) -> None:
        """``client_factory`` is the seam every test in this suite runs through.

        Called as ``client_factory(api_key=..., base_url=...)`` and must return
        something with ``chat.completions.create(...)``, ``models.list()`` and --
        for the ``wire="responses"`` path -- ``responses.stream(...)``. The
        default builds ``openai.OpenAI(api_key=..., base_url=...)``, never the
        bare constructor: an ambient ``OPENAI_API_KEY`` must not be able to reach
        an endpoint the connection never named, which on this runtime would also
        mean sending somebody's real OpenAI key to an arbitrary host.

        ``async_client_factory`` is the same seam for :meth:`arun` (ticket 1.13)
        and carries the same warning: the default is
        ``openai.AsyncOpenAI(api_key=..., base_url=...)``, never the bare
        constructor.
        """
        super().__init__(
            client_factory=client_factory,
            async_client_factory=async_client_factory,
            env=env,
            secrets=secrets,
        )
        if client_factory is None:
            self._client_factory = _default_client
        if async_client_factory is None:
            self._async_client_factory = _default_async_client

    # --- the credential, which may be absent -----------------------------------

    def _client(self, connection: Connection) -> Any:
        """The vendor client for this connection, built with an explicit key.

        The one divergence from the other three API adapters, and it is narrow:
        a connection whose ``credentialRef`` is ``none`` gets
        :data:`PLACEHOLDER_CREDENTIAL` rather than a resolved key. Everything
        else about the construction is identical -- explicit arguments, no
        environment discovery -- so the difference is what is *sent*, never how.
        """
        if connection.credential_ref.kind is CredentialKind.NONE:
            return self._client_factory(
                api_key=PLACEHOLDER_CREDENTIAL, base_url=connection.base_url
            )
        key = resolve_credential(connection, self._env, secrets=self._secrets)
        return self._client_factory(api_key=key, base_url=connection.base_url)

    def _async_client(self, connection: Connection) -> Any:
        """:meth:`_client`'s rules, for the async client (ticket 1.13).

        Including the narrow divergence: a connection whose ``credentialRef`` is
        ``none`` gets :data:`PLACEHOLDER_CREDENTIAL` here too, because a local
        server that wants no key must not be handed one the caller never named
        on either face.
        """
        if connection.credential_ref.kind is CredentialKind.NONE:
            return self._async_client_factory(
                api_key=PLACEHOLDER_CREDENTIAL, base_url=connection.base_url
            )
        key = resolve_credential(connection, self._env, secrets=self._secrets)
        return self._async_client_factory(api_key=key, base_url=connection.base_url)

    # --- preflight -------------------------------------------------------------

    def preflight(self, request: RunRequest) -> Receipt:
        """``openai-api``'s three checks, plus the two facts peculiar to a shape.

        The inherited preflight already reports the credential handling, the
        endpoint, the optional model-list probe, the missing model and any schema
        tightening. What this adds is the pair of sentences a caller of *this*
        runtime needs and no caller of the other three does: which wire the call
        will use, and whether anybody has ever driven this endpoint.
        """
        receipt = super().preflight(request)
        notes = list(receipt.notes)
        notes.append(_WIRE_NOTE[self.wire(request)])
        verified = request.connection.verified_capabilities
        notes.append(
            verified.note()
            if verified
            else (
                "nothing has been driven against this endpoint, so every capability "
                "cell for openai-compatible reads unverified -- including chat, which "
                f"means bridge.chat() refuses it. Run 'modelpass verify "
                f"{request.connection.name}': one short chat, one tool round trip and "
                "one structured-output call, and the cells that actually worked are "
                "recorded on the connection"
            )
        )
        return replace(receipt, notes=tuple(notes))

    def probe(self, request: RunRequest) -> ModelListProbe | None:
        """``models.list()`` -- token-free, and here it is two answers at once.

        On the metered runtimes a listing proves a credential is live. Here the
        endpoint may check no credential at all, so what it proves is that
        *something is running at that URL and speaking this shape* -- which is
        the precondition for every cell ``modelpass verify`` goes on to fill, and
        the reason this probe keeps more ids than ``openai-api``'s: the listing
        is the only reliable way to learn what the box serves.
        """
        try:
            listing = self._client(request.connection).models.list()
        except SubpassError:
            raise
        except Exception as exc:
            return ModelListProbe(ok=False, detail=f"{type(exc).__name__}: {exc}")
        ids = []
        for item in getattr(listing, "data", listing) or ():
            model_id = getattr(item, "id", None)
            if isinstance(model_id, str):
                ids.append(model_id)
        return ModelListProbe(ok=True, models=tuple(ids[:_PROBE_LIMIT]))

    # --- which wire ------------------------------------------------------------

    @staticmethod
    def wire(request: RunRequest) -> str:
        """``"chat"`` or ``"responses"``, from ``options={"wire": ...}``.

        Unknown values fall back to the default rather than raising, and the
        receipt's wire note then says ``chat`` -- which is true, and is what the
        call will do. The typo is not silent: ``wire`` is in
        :attr:`option_keys`, so the *key* is honoured, and a value outside the
        two is reported the way every other honesty note is reported.
        """
        value = request.options.get("wire")
        return "responses" if value == "responses" else "chat"

    def support_for(self, capability: Capability, options: Mapping[str, Any]):
        """Which cells the *Responses* wire can answer for, when it is selected.

        ``None`` -- no opinion -- for ``wire="chat"``, which is the transport the
        capability row describes and the connection's own
        ``verifiedCapabilities`` refine. For ``wire="responses"`` the answer is
        still ``None``, and that is deliberate rather than an omission: selecting
        that wire says something about *modelpass's* code path, not about the
        endpoint, and an endpoint that does not implement Responses would then
        have been promised a capability by an option. The narrowing this hook
        exists for (S7a on ``openai-sdk``) has no counterpart here.
        """
        del capability, options
        return None

    # --- the run ---------------------------------------------------------------

    def chat_sampling_params(self, request: RunRequest) -> dict[str, Any]:
        """The sampling half of the request, in this transport's spelling (R5).

        The per-model decisions are taken once, in
        :func:`~modelpass.sampling_rules.plan_sampling`, whose answer this method
        translates and does not revisit. What is left is two pieces of genuine
        transport knowledge:

        * ``temperature`` / ``top_p`` -> themselves.
        * ``max_output_tokens`` -> **``max_tokens``**, and the choice of spelling
          is the one place this transport costs something. ``openai`` 2.32.0
          takes both ``max_tokens`` (deprecated, still accepted) and
          ``max_completion_tokens`` (its replacement), and the compatible servers
          split: Ollama, LM Studio and llama.cpp have implemented ``max_tokens``
          since the beginning, while ``max_completion_tokens`` is newer and not
          universal. A ceiling a server silently ignores is the worst of the
          available failures -- it is how a run that was supposed to be bounded
          runs long -- so modelpass sends the spelling that is most likely to be
          read. The receipt reports it under modelpass's own name either way.
        * ``top_k`` never arrives: the rules row does not accept it and
          ``plan_sampling`` drops it with a note before this method runs. See
          that row for why an ``extra_body`` guess was refused.
        * ``reasoning_effort`` never arrives either, for the same reason: the row
          has no ``reasoning_parameter``, because there is no spelling of it that
          a compatible server can be assumed to take.

        **There is no default ceiling**, as on ``openai-api``: a caller who names
        none gets the model's own limit rather than a 4096 modelpass invented.
        """
        plan = plan_sampling(
            request.effective_sampling,
            self.runtime,
            request.model or request.connection.model,
        )
        params: dict[str, Any] = {}
        for name, value in plan.applied.items():
            params["max_tokens" if name == "max_output_tokens" else name] = value
        return params

    def chat_request_params(
        self, request: RunRequest, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """One round's request, assembled. Pure: no network, no client.

        ``stream_options={"include_usage": True}`` is sent on every call, because
        usage is otherwise simply absent from a streamed Chat Completions
        response -- there is no end-of-stream envelope carrying it the way
        Responses has. A server that has never heard of ``stream_options``
        ignores it, which is the same outcome as not asking; a server that
        honours it sends one final chunk with no choices and a ``usage`` object.
        """
        params: dict[str, Any] = {
            "model": request.model or request.connection.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        params.update(self.chat_sampling_params(request))
        if request.tools:
            params["tools"] = tool_params(request.tools)
        # D13: a schema and tools are never combined; the bridge refuses the
        # combination before an adapter sees it.
        response_format = response_format_param(request)
        if response_format is not None:
            params["response_format"] = response_format
        return params

    def run(self, request: RunRequest) -> Iterator[AgentEvent]:
        """One stateless call, plus however many tool rounds the model asks for.

        ``options={"wire": "responses"}`` hands the whole run to ticket 1.9's
        implementation, unchanged, which is the point of the subclass: the
        alternative wire is not a reimplementation of Responses, it *is* the
        Responses adapter, running against this connection's base URL.
        """
        if self.wire(request) == "responses":
            return super().run(request)
        if not (request.model or request.connection.model):
            raise PreflightFailed(
                self.no_model_message.format(name=request.connection.name)
            )
        try:
            client = self._client(request.connection)
        except SubpassError:
            raise
        except Exception as exc:  # pragma: no cover - defensive: a constructor
            raise VendorRunFailed(
                f"the {_PACKAGE} client could not be constructed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # This run's own cancel state (2026-09-24). It used to be one flag on
        # the adapter, cleared by every run() and set by every teardown, so
        # one run ending stopped whichever other run reached a round boundary.
        cancel = RunCancel()
        self._runs.add(cancel)
        return RunStream(self._chat_loop(client, request, cancel), cancel.cancel)

    def _chat_loop(
        self, client: Any, request: RunRequest, cancel: RunCancel
    ) -> Iterator[AgentEvent]:
        messages = messages_param(request)
        by_name = {tool.name: tool for tool in request.tools}
        final_text = ""

        while True:
            if cancel.is_set():
                yield _bare_terminal(
                    TerminalStatus.CANCELLED, "cancelled by caller", self.runtime
                )
                return

            params = self.chat_request_params(request, messages)
            buffer = _CallBuffer()
            text = ""
            finish_reason: str | None = None
            usage: Any = None
            try:
                stream = client.chat.completions.create(**params)
                try:
                    for chunk in stream:
                        buffer.absorb(chunk)
                        choice = _choice(chunk)
                        reason = _field(choice, "finish_reason")
                        if isinstance(reason, str) and reason:
                            finish_reason = reason
                        chunk_usage = _field(chunk, "usage")
                        if has_usage(chunk_usage):
                            usage = chunk_usage
                        for event in chunk_events(chunk, self.runtime):
                            if isinstance(event, TextDeltaEvent):
                                text += event.text
                            yield event
                finally:
                    # The SDK's streaming response is a context manager in all
                    # but name: closing it tears the HTTP response down, which is
                    # the transport-level floor the cancellation contract
                    # documents (D10). It matters more here than on Responses,
                    # where a ``with`` block already did it.
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
            except SubpassError:
                raise
            except Exception as exc:
                if request.schema is not None and rejects_response_format(exc):
                    raise StructuredOutputRejected(
                        self.runtime.value,
                        str(Capability.STRUCTURED_OUTPUT),
                        "unsupported",
                        _SCHEMA_REJECTED.format(
                            name=request.connection.name, detail=exc
                        ),
                    ) from exc
                status, reason = _vendor_failure(exc)
                code, after = vendor_error_facts(exc)
                yield _bare_terminal(
                    status, reason, self.runtime, status_code=code, retry_after=after
                )
                return

            # No usage event at all where the server reported none: a zeroed one
            # would say this run cost nothing, and ``Result.usage is None`` says
            # nobody told us. Those are different claims and only one is true.
            if usage is not None:
                yield UsageEvent(usage=token_usage(usage), scope=UsageScope.DELTA)

            if text:
                final_text = text

            calls = buffer.assembled()
            # A run that declared no tools has nothing to answer with, and
            # replying "not declared" round after round would be a loop with a
            # bill on it.
            if not (request.tools and calls):
                yield from self._finish_chat(
                    request, finish_reason, final_text, bool(calls)
                )
                return

            # The assistant turn goes back before the results that answer it,
            # carrying the calls in the shape the server sent them -- arguments
            # as the JSON *string* that was streamed, not a re-serialization of
            # the parsed form, so a handler that saw ``{}`` for unparseable JSON
            # does not also silently repair the transcript.
            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in calls
                    ],
                }
            )
            for call in calls:
                call_id = call["id"]
                name = call["name"]
                arguments = call_arguments(call["arguments"])
                yield ToolCallEvent(
                    name=name,
                    arguments=arguments,
                    id=call_id,
                    server=CALLER_TOOL_SERVER,
                )
                tool = by_name.get(name)
                payload = (
                    invoke_handler(tool, arguments)
                    if tool is not None
                    else _missing_tool(name)
                )
                content = flatten_mcp_result(payload)
                yield ToolResultEvent(
                    id=call_id,
                    name=name,
                    content=content,
                    is_error=bool(payload.get("is_error")),
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": content}
                )

    def arun(self, request: RunRequest) -> AsyncIterator[AgentEvent]:
        """:meth:`run`, on the caller's loop (R1, ticket 1.13).

        ``options={"wire": "responses"}`` hands the whole run to ticket 1.9's
        async implementation, unchanged, exactly as the sync pair does: the
        alternative wire is not a reimplementation of Responses, it *is* the
        Responses adapter, running against this connection's base URL.
        """
        if self.wire(request) == "responses":
            return super().arun(request)
        if not (request.model or request.connection.model):
            raise PreflightFailed(
                self.no_model_message.format(name=request.connection.name)
            )
        try:
            client = self._async_client(request.connection)
        except SubpassError:
            raise
        except Exception as exc:  # pragma: no cover - defensive: a constructor
            raise VendorRunFailed(
                f"the {_PACKAGE} async client could not be constructed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        cancel = RunCancel()
        self._runs.add(cancel)
        return native_run(self, self._achat_loop(client, request, cancel), cancel.cancel)

    async def _achat_loop(
        self, client: Any, request: RunRequest, cancel: RunCancel
    ) -> AsyncIterator[AgentEvent]:
        """:meth:`_chat_loop` with ``await`` where it blocks. Same rounds, same events."""
        messages = messages_param(request)
        by_name = {tool.name: tool for tool in request.tools}
        final_text = ""

        while True:
            if cancel.is_set():
                yield _bare_terminal(
                    TerminalStatus.CANCELLED, "cancelled by caller", self.runtime
                )
                return

            params = self.chat_request_params(request, messages)
            buffer = _CallBuffer()
            text = ""
            finish_reason: str | None = None
            usage: Any = None
            try:
                stream = await client.chat.completions.create(**params)
                try:
                    async for chunk in stream:
                        buffer.absorb(chunk)
                        choice = _choice(chunk)
                        reason = _field(choice, "finish_reason")
                        if isinstance(reason, str) and reason:
                            finish_reason = reason
                        chunk_usage = _field(chunk, "usage")
                        if has_usage(chunk_usage):
                            usage = chunk_usage
                        for event in chunk_events(chunk, self.runtime):
                            if isinstance(event, TextDeltaEvent):
                                text += event.text
                            yield event
                finally:
                    # The streaming response is a context manager in all but
                    # name on this transport too; closing it tears the HTTP
                    # response down, which is the D10 floor.
                    closer = getattr(stream, "aclose", None) or getattr(
                        stream, "close", None
                    )
                    if callable(closer):
                        closed = closer()
                        if inspect.isawaitable(closed):
                            await closed
            except SubpassError:
                raise
            except Exception as exc:
                if request.schema is not None and rejects_response_format(exc):
                    raise StructuredOutputRejected(
                        self.runtime.value,
                        str(Capability.STRUCTURED_OUTPUT),
                        "unsupported",
                        _SCHEMA_REJECTED.format(
                            name=request.connection.name, detail=exc
                        ),
                    ) from exc
                status, reason = _vendor_failure(exc)
                code, after = vendor_error_facts(exc)
                yield _bare_terminal(
                    status, reason, self.runtime, status_code=code, retry_after=after
                )
                return

            if usage is not None:
                yield UsageEvent(usage=token_usage(usage), scope=UsageScope.DELTA)

            if text:
                final_text = text

            calls = buffer.assembled()
            if not (request.tools and calls):
                for out in self._finish_chat(
                    request, finish_reason, final_text, bool(calls)
                ):
                    yield out
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in calls
                    ],
                }
            )
            for call in calls:
                call_id = call["id"]
                name = call["name"]
                arguments = call_arguments(call["arguments"])
                yield ToolCallEvent(
                    name=name,
                    arguments=arguments,
                    id=call_id,
                    server=CALLER_TOOL_SERVER,
                )
                tool = by_name.get(name)
                # The caller's own coroutine, awaited on the caller's own loop.
                payload = (
                    await ainvoke_handler(tool, arguments)
                    if tool is not None
                    else _missing_tool(name)
                )
                content = flatten_mcp_result(payload)
                yield ToolResultEvent(
                    id=call_id,
                    name=name,
                    content=content,
                    is_error=bool(payload.get("is_error")),
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": content}
                )

    def _finish_chat(
        self,
        request: RunRequest,
        finish_reason: str | None,
        final_text: str,
        unanswered_call: bool,
    ) -> Iterator[AgentEvent]:
        """The structured answer, if one was asked for, and then the terminal."""
        status, reason = terminal_status(
            finish_reason,
            unanswered_call=unanswered_call,
            answered=bool(final_text),
        )
        if request.schema is not None:
            event, failure = build_structured_event(
                request.schema,
                raw=final_text,
                schema_name=resolve_schema_name(request.schema, request.schema_name),
            )
            if event is not None:
                yield event
            if failure is not None and status is TerminalStatus.OK:
                status, reason = TerminalStatus.ERROR, failure
        yield _bare_terminal(status, reason, self.runtime)

    # --- the per-install drive --------------------------------------------------

    def verify_capabilities(self, request: RunRequest) -> VerifyReport:
        """Drive this endpoint and report which cells it actually answers.

        **The one place in modelpass where a capability verdict is reached per
        install rather than per runtime**, and the reason is in this module's
        docstring: the row describes an API shape, so only the endpoint can say
        what it does. Four steps, in increasing order of what they cost:

        1. a **token-free probe** (:meth:`probe`) -- is anything there at all, and
           what does it serve. A failure here ends the drive: every later step
           would fail for the same reason and reporting six ``unsupported`` cells
           because a box is switched off would be the worst possible lie for this
           record to tell.
        2. **one short chat**, which decides ``chat``, ``streaming``,
           ``incremental_text`` and ``usage_tokens`` from what the stream carried.
        3. **one tool round trip**, with a trivial tool whose handler answers a
           constant, which decides ``tools_in_process``. Two rounds of tokens.
        4. **one structured-output call**, which decides ``structured_output`` --
           and decides it ``False`` on a
           :class:`~modelpass.errors.StructuredOutputRejected`, which is the
           commonest outcome against a small local server and is a *finding*
           rather than a failure of the drive.

        Each step is independent: a server that chats but cannot do tools records
        the chat cells as supported and ``tools_in_process`` as unsupported,
        because that is what happened. The report is returned rather than written
        -- the caller decides whether to persist it -- which is what keeps this
        method drivable from a test with a fake client and no store.
        """
        notes: list[str] = []
        # verify must never be served a cached answer: it exists to re-read the
        # live one, and the cache is keyed by connection name, which is exactly
        # the key a user has just changed the endpoint behind.
        self.invalidate_identity_cache()
        probe = self.probe(request)
        if probe is None or not probe.ok:
            detail = probe.detail if probe is not None else "the probe was not taken"
            return VerifyReport(
                notes=(f"model list: {detail}",),
                ok=False,
                problem=(
                    f"the endpoint at {request.connection.base_url} did not answer a "
                    f"model listing, so nothing could be driven against it: {detail}"
                ),
            )
        notes.append(
            f"model list: {len(probe.models)} model(s) -- "
            + (", ".join(probe.models[:10]) or "none reported")
        )

        supported: list[str] = []
        unsupported: list[str] = []

        def record(capability: Capability, observed: bool, note: str) -> None:
            (supported if observed else unsupported).append(str(capability))
            notes.append(note)

        chat, chat_note = self._drive_chat(request)
        for capability in (
            Capability.CHAT,
            Capability.STREAMING,
            Capability.INCREMENTAL_TEXT,
        ):
            (supported if chat["answered"] else unsupported).append(str(capability))
        notes.append(chat_note)
        record(
            Capability.USAGE_TOKENS,
            chat["usage"],
            "usage: "
            + (
                "the server honoured stream_options.include_usage"
                if chat["usage"]
                else "the server reported none, so a run's usage reads as None "
                "rather than as zero"
            ),
        )
        if chat["answered"]:
            tools, tools_note = self._drive_tools(request)
            record(Capability.TOOLS_IN_PROCESS, tools, tools_note)
            schema, schema_note = self._drive_schema(request)
            record(Capability.STRUCTURED_OUTPUT, schema, schema_note)
        else:
            notes.append(
                "tools and structured output were not attempted: the endpoint did "
                "not answer a plain chat, so a failure of either would say nothing "
                "about the feature"
            )

        return VerifyReport(
            verified=VerifiedCapabilities(
                supported=tuple(supported),
                unsupported=tuple(unsupported),
                checked_at=datetime.now(UTC).isoformat(timespec="seconds"),
                models=probe.models,
            ),
            notes=tuple(notes),
            ok=True,
        )

    def _verify_request(self, request: RunRequest, **overrides: Any) -> RunRequest:
        """One drive's request: the caller's connection, modelpass's own prompt.

        Deliberately not the caller's messages -- ``verify`` takes a connection,
        not a conversation -- and deliberately small: every drive asks for a
        handful of tokens, because the question is whether a feature works and
        not how well the model writes.
        """
        return replace(
            request,
            messages=(Message(role=Role.USER, content=overrides.pop("prompt")),),
            tools=overrides.pop("tools", ()),
            schema=overrides.pop("schema", None),
            schema_name=overrides.pop("schema_name", ""),
            options={"max_output_tokens": _VERIFY_MAX_TOKENS},
        )

    def _drive_chat(self, request: RunRequest) -> tuple[dict[str, bool], str]:
        """Step 2: one short chat. Decides four cells from one stream."""
        deltas = 0
        usage = False
        text = ""
        status = TerminalStatus.ERROR
        reason: str | None = "nothing arrived"
        try:
            for event in self.run(self._verify_request(request, prompt=_VERIFY_PROMPT)):
                if isinstance(event, TextDeltaEvent):
                    deltas += 1
                    text += event.text
                elif isinstance(event, UsageEvent):
                    usage = True
                elif isinstance(event, TerminalEvent):
                    status, reason = event.status, event.reason
        except SubpassError as exc:
            reason = str(exc)
        answered = bool(text.strip()) and status is TerminalStatus.OK
        note = (
            f"chat: answered in {deltas} text delta(s)"
            if answered
            else f"chat: no answer ({reason})"
        )
        return {"answered": answered, "usage": usage and answered}, note

    def _drive_tools(self, request: RunRequest) -> tuple[bool, str]:
        """Step 3: one tool round trip. The handler answers a constant."""
        called: list[str] = []

        def handler(arguments: Mapping[str, Any]) -> str:
            del arguments
            called.append(_VERIFY_TOOL_NAME)
            return _VERIFY_TOOL_ANSWER

        tool = ToolDef(
            name=_VERIFY_TOOL_NAME,
            description=(
                "Returns a fixed word. modelpass calls this during 'modelpass "
                "verify' to find out whether this endpoint can use a tool."
            ),
            parameters={"type": "object", "properties": {}},
            handler=handler,
        )
        drive = self._verify_request(
            request, prompt=_VERIFY_TOOL_PROMPT, tools=(tool,)
        )
        try:
            events = list(self.run(drive))
        except SubpassError as exc:
            return False, f"tools: the endpoint refused the call ({exc})"
        asked = any(isinstance(event, ToolResultEvent) for event in events)
        if asked and called:
            return True, "tools: the model asked for the tool and used its result"
        return False, (
            "tools: the model was given one tool and did not call it. That is a "
            "finding about this endpoint and not a certainty -- a model may decline "
            "a tool it was offered -- so re-run verify if you believe otherwise"
        )

    def _drive_schema(self, request: RunRequest) -> tuple[bool, str]:
        """Step 4: one structured-output call. A rejection is a finding."""
        drive = self._verify_request(
            request,
            prompt=_VERIFY_SCHEMA_PROMPT,
            schema=_VERIFY_SCHEMA,
            schema_name="modelpass_verify",
        )
        try:
            events = list(self.run(drive))
        except StructuredOutputRejected as exc:
            return False, f"structured output: the endpoint rejected it ({exc.detail})"
        except SubpassError as exc:
            return False, f"structured output: the call failed ({exc})"
        if any(isinstance(event, StructuredOutputEvent) for event in events):
            return True, "structured output: the answer came back as the schema"
        return False, (
            "structured output: the call succeeded but no answer matching the schema "
            "arrived, so the field was accepted and not honoured -- which is worse "
            "than a rejection and is recorded as unsupported"
        )

    # --- sessions: refused, and the refusal says what to use instead ------------

    def open_session(self, request: SessionRequest) -> SessionHandle:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.new_chat()"))

    def resume_session(self, request: SessionRequest) -> SessionHandle:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.resume_chat()"))

    def list_sessions(self, request: SessionRequest) -> tuple[SessionInfo, ...]:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.list_sessions()"))


#: What the receipt says about the transport this call will use.
_WIRE_NOTE = {
    "chat": (
        "wire: chat completions (POST /v1/chat/completions), which is what "
        "Ollama, LM Studio, vLLM, llama.cpp and a LiteLLM proxy all implement. "
        "Pass options={'wire': 'responses'} to use the Responses API instead, "
        "where the endpoint implements it"
    ),
    "responses": (
        "wire: responses (options={'wire': 'responses'}), which runs the "
        "openai-api adapter's own path against this connection's base URL. Most "
        "OpenAI-compatible servers do not implement it; if this call 404s, drop "
        "the option"
    ),
}

#: The detail on a :class:`~modelpass.errors.StructuredOutputRejected`. Says what
#: to try, because the two honest ways forward are not obvious from an HTTP 400.
_SCHEMA_REJECTED = (
    "the endpoint refused the json_schema response_format this call sent, which "
    "many OpenAI-compatible servers do not implement ({detail}). What to try: run "
    "'modelpass verify {name}' to record the absence on the connection, then ask "
    "for JSON in the prompt and parse it yourself -- modelpass will not silently "
    "downgrade to response_format json_object, which constrains the answer to be "
    "JSON without constraining it to be your schema. Or point the connection at a "
    "server that implements the field"
)

#: The tokens each verify drive asks for. Small, and the same for all three:
#: the question is whether a feature works, not how well the model writes.
_VERIFY_MAX_TOKENS = 64

_VERIFY_PROMPT = "Reply with the single word: ok"
_VERIFY_TOOL_NAME = "modelpass_verify_probe"
_VERIFY_TOOL_ANSWER = "ok"
_VERIFY_TOOL_PROMPT = (
    f"Call the {_VERIFY_TOOL_NAME} tool, then reply with exactly what it returned."
)
_VERIFY_SCHEMA_PROMPT = "Answer with ok set to true."
_VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


def _default_client(*, api_key: str, base_url: str | None) -> Any:
    """``openai.OpenAI``, constructed with an explicit key and nothing ambient.

    ``base_url`` is always present on this runtime -- ``Connection`` requires it
    -- but the signature keeps the other adapter's shape so the two factories are
    interchangeable behind the same seam.
    """
    try:
        sdk = importlib.import_module(_MODULE)
    except ImportError as exc:  # pragma: no cover - the extra being absent
        raise RuntimeNotAvailable(
            Runtime.OPENAI_COMPATIBLE.value, _EXTRA, _PACKAGE
        ) from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    return sdk.OpenAI(**kwargs)


def _default_async_client(*, api_key: str, base_url: str | None) -> Any:
    """``openai.AsyncOpenAI``, under exactly :func:`_default_client`'s rules."""
    try:
        sdk = importlib.import_module(_MODULE)
    except ImportError as exc:  # pragma: no cover - the extra being absent
        raise RuntimeNotAvailable(
            Runtime.OPENAI_COMPATIBLE.value, _EXTRA, _PACKAGE
        ) from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    return sdk.AsyncOpenAI(**kwargs)


