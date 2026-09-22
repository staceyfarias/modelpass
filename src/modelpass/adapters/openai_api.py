"""The ``openai-api`` adapter: the Responses API, driven in this process (ticket 1.9).

The second of the four API-key adapters, and a deliberate copy of the first.
:mod:`modelpass.adapters.anthropic_api` set the pattern in ticket 1.6 -- a
``client_factory`` seam, :func:`~modelpass.preflight.api_preflight`, an
in-adapter tool loop with no turn cap, :func:`~modelpass.sampling_rules.rules_for`
deciding the sampling, a 429 typed off ``status_code`` -- and this module keeps
the same shape, the same order and, where the vendors agree, the same sentences,
so the four API adapters read as one family rather than as four dialects.

**Responses, not Chat Completions, and the reason is this module's event
vocabulary** (settled 2026-09-13; see ``docs/api-and-runtimes.md`` §2.0b).
``openai`` 2.32.0 carries both; the
Responses API is the only one of the two that can answer three of the questions
modelpass's stream contract asks:

* **``thinking``.** ``responses.stream`` emits
  ``response.reasoning_summary_text.delta`` and ``response.reasoning_text.delta``
  as first-class stream events. ``chat.completions`` exposes reasoning only as a
  *token count* in ``completion_tokens_details.reasoning_tokens`` -- there is no
  text anywhere in its stream -- so a ``ThinkingEvent`` on that transport could
  only ever be invented.
* **``terminal``.** ``Response.status`` plus ``incomplete_details.reason``
  (``max_output_tokens`` / ``content_filter``) is the same shape as Anthropic's
  ``stop_reason``, which is what lets :func:`terminal_status` be a near-copy of
  ticket 1.6's. Chat Completions' ``finish_reason`` is per *choice* and its
  ``content_filter`` value is a different thing at a different level.
* **``max_output_tokens``.** The Responses request field has modelpass's own
  name and meaning. Chat Completions spells it ``max_completion_tokens`` and
  keeps a deprecated ``max_tokens`` beside it, which is two names for a field
  the receipt has to report under one.

Nothing in the event vocabulary pushed the other way, so there is one transport
here and no option to pick the other. If a caller ever needs Chat Completions,
``openai-compatible`` (ticket 1.10) is the runtime that speaks that shape.

**What is the same as ``anthropic-api``, on purpose.** The event vocabulary, the
tool loop's rules (no turn cap, a raising handler becomes a failed tool result,
cancellation checked between rounds), the receipt fields, the session refusals
and the credential handling: the client is built with an explicit ``api_key=``
from :func:`~modelpass.preflight.resolve_credential`, never the bare constructor,
so an ambient ``OPENAI_API_KEY`` cannot bill an account nobody named.

**Where the two vendors genuinely differ, and where each difference lands.**

* **Usage.** ``ResponseUsage`` reports ``input_tokens`` with the cached prefix
  *inside* it and ``input_tokens_details.cached_tokens`` beside it; modelpass's
  convention is the opposite -- :class:`~modelpass.types.TokenUsage` counts
  ``cached_input_tokens`` **separately from** ``input_tokens``. So
  :func:`token_usage` subtracts. ``cache_write_tokens`` is ``0``: OpenAI's prefix
  cache is automatic and it bills no write premium, so there is no number to
  report and reporting the fresh input as a write would be worse than a zero.
  ``output_tokens_details.reasoning_tokens`` stays folded **inside**
  ``output_tokens``, which is where it already is on the wire and where
  Anthropic's thinking tokens sit too; ``TokenUsage`` has no fifth field and
  inventing a subtraction here would make one runtime's output tokens mean
  something different from another's.
* **No ``cache_control``.** The Responses request schema has no breakpoint
  field: OpenAI's prefix caching is automatic, vendor-side and unaddressable.
  ``cache_breakpoints`` is a checked ``unsupported`` on this row (it has been
  since ticket 1.5), the system prompt is flattened with
  :attr:`~modelpass.types.Message.flat_text` semantics, and the bridge's own
  disclosure names the drop on the receipt. The adapter does not restate it:
  ticket 1.5 put that sentence in one place on purpose.
* **Structured output must be *strict*.** ``text={"format": {"type":
  "json_schema", "strict": True, ...}}`` requires the subset
  :func:`~modelpass.schema.to_openai_strict` produces -- every property required,
  optional ones nullable, ``additionalProperties: false``. The caller's schema is
  run through it rather than refused, and
  :func:`~modelpass.schema.openai_strict_issues` is reported as receipt notes so
  a caller learns *what was tightened* before the call rather than from an answer
  with unexpected nulls in it. Those two functions predate this adapter (they
  were written for ``openai-sdk``) and are reused rather than copied.
* **The final response is read off the stream's own terminal event.** Not
  ``get_final_response()``: that helper raises ``RuntimeError`` when the stream
  ended with ``response.incomplete`` rather than ``response.completed``, which is
  exactly the truncation case :func:`terminal_status` exists to report. Reading
  ``.response`` off whichever of ``response.completed`` / ``response.incomplete``
  / ``response.failed`` arrives turns a raise into the terminal it should always
  have been (``openai`` 2.32.0, read 2026-09-13).
* **Reasoning summaries are opt-in.** The stream carries them, but only when the
  request asks: ``options={"reasoning_summary": "auto"}`` fills
  ``reasoning.summary``. modelpass does not ask on its own -- a summary is billed
  output nobody requested -- which is why ``thinking`` on this row is
  ``unverified`` with a live test named rather than ``supported``.

**What it deliberately does not do.** Sessions, for the reason ticket 1.8 wrote
into the bridge: an API endpoint holds no history. The three methods keep their
own refusal anyway, worded exactly as ``anthropic-api``'s are. Note that the
Responses API *does* have ``conversation`` and ``previous_response_id``, which is
a vendor-held history -- and it is still not a modelpass session: D14's session is
a resumable, listable, forkable thing, and a response id is none of those. That
is a decision for its own record, not a cell this adapter may move.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import threading
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar

from ..capabilities import Capability, Support
from ..connections import Connection
from ..errors import (
    AdapterNotImplemented,
    PreflightFailed,
    RuntimeNotAvailable,
    SubpassError,
    VendorRunFailed,
)
from ..preflight import (
    ModelListProbe,
    Receipt,
    api_preflight,
    resolve_credential,
)
from ..prompt_cache import plan_prompt_cache
from ..retry import vendor_error_facts
from ..runtimes import Runtime
from ..sampling_rules import plan_sampling
from ..schema import (
    build_structured_event,
    normalize_schema,
    openai_strict_issues,
    resolve_schema_name,
    to_openai_strict,
)
from ..tools import ToolDef
from ..types import (
    CALLER_TOOL_SERVER,
    AgentEvent,
    AuthMode,
    SessionInfo,
    TerminalEvent,
    TerminalStatus,
    TextBlock,
    TextDeltaEvent,
    ThinkingEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    UsageScope,
    VendorEvent,
)
from ._mcp_result import ainvoke_handler, flatten_mcp_result, mcp_result_payload
from .base import Adapter, RunRequest, SessionHandle, SessionRequest, native_run

__all__ = ["SDK_VERSION_READ", "OpenAIAPIAdapter", "token_usage"]

_MODULE = "openai"
_EXTRA = "openai-api"
_PACKAGE = "openai"

#: The installed SDK this adapter was written against and whose surface every
#: capability note cites. Data rather than prose for the reason ticket 1.6 gave:
#: the notes are evidence, and evidence names its source. Read on 2026-09-13.
SDK_VERSION_READ = "2.32.0"

#: How many model ids :meth:`OpenAIAPIAdapter.probe` keeps. ``models.list()``
#: here takes no ``limit`` argument (unlike Anthropic's), so the truncation is
#: this side of the wire; the probe answers "is this credential live" and an
#: unbounded list would put a hundred strings on a receipt nobody asked for.
_PROBE_LIMIT = 20


# --- usage -----------------------------------------------------------------------


def _count(usage: Any, key: str) -> int:
    """One integer off a usage object or mapping; anything else is zero."""
    if isinstance(usage, Mapping):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _nested(usage: Any, group: str, key: str) -> int:
    """One integer out of ``input_tokens_details`` / ``output_tokens_details``."""
    if isinstance(usage, Mapping):
        inner = usage.get(group)
    else:
        inner = getattr(usage, group, None)
    return _count(inner, key) if inner is not None else 0


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


def token_usage(usage: Any) -> TokenUsage:
    """Normalize ``ResponseUsage`` onto modelpass tokens. **The subtraction is the
    point.**

    :class:`~modelpass.types.TokenUsage` counts ``cached_input_tokens``
    *separately from* ``input_tokens`` -- stated in its own docstring, and the
    convention ``anthropic-api`` and ``anthropic-sdk`` both follow, where
    ``cache_read_input_tokens`` arrives as its own number beside a fresh
    ``input_tokens``. OpenAI reports the opposite shape: ``usage.input_tokens``
    is the whole prompt and ``input_tokens_details.cached_tokens`` says how much
    of it was a cache hit. Assigning both straight across would count the cached
    prefix twice, so ``input_tokens`` here is ``input_tokens - cached_tokens``
    and a consumer's accounting does not learn that a run changed runtimes.

    Where the other two numbers land:

    * ``cache_write_tokens`` is ``0``, always. OpenAI's prefix cache is automatic
      and carries no write premium, so there is no wire field and nothing to map;
      a zero says "this runtime does not report cache writes", which is true,
      while folding fresh input in there would claim a cost nobody paid.
    * ``output_tokens_details.reasoning_tokens`` stays **inside**
      ``output_tokens``, untouched -- and is *also* carried on
      ``reasoning_output_tokens`` (2026-09-22), which is a subset field rather
      than a fifth addend precisely so that this sentence stays true. Before
      that field existed the number reached a caller only on the
      ``vendor_event`` path, which meant the question "what did my effort
      setting actually cost" could not be asked of the run log.
    """
    cached = _nested(usage, "input_tokens_details", "cached_tokens")
    total_input = _count(usage, "input_tokens")
    return TokenUsage(
        input_tokens=max(total_input - cached, 0),
        output_tokens=_count(usage, "output_tokens"),
        cached_input_tokens=cached,
        cache_write_tokens=0,
        reasoning_output_tokens=_reported_nested(
            usage, "output_tokens_details", "reasoning_tokens"
        ),
    )


# --- request assembly ------------------------------------------------------------


def flatten(blocks: Sequence[TextBlock]) -> str:
    """Blocks as one string, with :attr:`~modelpass.types.Message.flat_text`
    semantics.

    One blank line between non-empty blocks -- deliberately the same join
    ticket 1.5 chose for the runtimes that cannot carry blocks, and for the same
    reason: it is what the flattening produced before blocks existed, so a prompt
    written for this runtime keeps reading the way it always did. The
    ``cache_control`` markers on the blocks have nowhere to go here and are
    dropped; the bridge's own disclosure names the drop on the receipt (R3), and
    this function does not restate it.
    """
    return "\n\n".join(block.text for block in blocks if block.text)


def instructions_param(request: RunRequest) -> str | None:
    """The ``instructions=`` string, or ``None`` when the run carries no system
    message.

    ``instructions`` is the Responses API's system prompt and it is a *string*,
    not a block array, which is the whole of why this runtime flattens where
    ``anthropic-api`` passes through. ``None`` rather than ``""`` so a run with
    no system message omits the field instead of sending an empty one.
    """
    blocks = request.system_blocks
    if not blocks:
        return None
    return flatten(blocks) or None


def input_items(request: RunRequest) -> list[dict[str, Any]]:
    """The ``input=`` array: one message item per non-system turn.

    ``{"role": ..., "content": "<text>"}`` is ``EasyInputMessageParam``, which
    ``openai`` 2.32.0 types as accepting a plain string for ``content`` -- so the
    simplest shape that carries a turn is the one used, rather than a hand-built
    ``input_text`` / ``output_text`` part list that would have to know which part
    type each role takes.
    """
    return [
        {"role": role, "content": flatten(blocks)}
        for role, blocks in request.conversation_blocks
    ]


def tool_params(tools: Sequence[ToolDef]) -> list[dict[str, Any]]:
    """Caller tools as ``tools=`` entries, in the Responses function shape.

    Flat -- ``{"type": "function", "name": ..., "parameters": ...}`` -- rather
    than Chat Completions' nested ``{"type": "function", "function": {...}}``,
    which is one more difference between the two transports and one more reason
    an adapter should speak exactly one of them.

    ``strict`` is ``False``: a tool's parameter schema is the caller's own and
    tightening it silently would change which calls the model can make. That is
    the same refusal :func:`~modelpass.schema.to_openai_strict` documents for
    schemas, applied to tools, and it is deliberately *not* the same answer
    structured output gets -- there, strict is what the feature is.
    """
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.json_schema(),
            "strict": False,
        }
        for tool in tools
    ]


def text_param(request: RunRequest) -> dict[str, Any] | None:
    """The native structured-output configuration, or ``None`` (D13).

    ``text={"format": {"type": "json_schema", "name": ..., "schema": ...,
    "strict": True}}``, read off ``openai`` 2.32.0 on 2026-09-13.
    ``ResponseFormatTextJSONSchemaConfigParam`` types ``name`` as **required**,
    which is why :func:`~modelpass.schema.resolve_schema_name` is called here and
    not only for the event.

    The schema is normalized and then put through
    :func:`~modelpass.schema.to_openai_strict`, because ``strict: True`` is what
    makes the vendor constrain the answer at all and the strict subset has no
    notion of an optional property. What that rewrite *did* is reported as
    receipt notes by :meth:`OpenAIAPIAdapter.preflight`, so the tightening is
    disclosed before the call rather than discovered in the answer.
    """
    if request.schema is None:
        return None
    normalized = normalize_schema(request.schema)
    return {
        "format": {
            "type": "json_schema",
            "name": resolve_schema_name(normalized, request.schema_name),
            "schema": to_openai_strict(normalized),
            "strict": True,
        }
    }


def _item_param(item: Any) -> Any:
    """One response output item, in the shape it goes back up as.

    Output items are replayed verbatim in a tool loop -- reasoning items
    included, ``encrypted_content`` included -- so this dumps whatever the SDK
    handed over rather than rebuilding an item from the fields this adapter
    happens to read. Dropping reasoning items would be the expensive kind of
    wrong on this runtime: a reasoning model that cannot see its own previous
    thinking re-derives it every round.

    ``exclude_none=True`` where the model supports it, because an input item is a
    ``TypedDict`` whose optional keys are *absent* rather than null; a
    ``model_dump()`` full of ``None``s is not the same document.
    """
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except TypeError:  # pragma: no cover - defensive: a dump without the kwarg
            try:
                return dump()
            except Exception:  # pragma: no cover - defensive: a vendor model quirk
                pass
        except Exception:  # pragma: no cover - defensive: a vendor model quirk
            pass
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(item, Mapping):
        return dict(item)
    return item


# --- event mapping ---------------------------------------------------------------

#: The stream event type that carries assistant text, one delta at a time.
_TEXT_DELTA = "response.output_text.delta"

#: The two that carry reasoning text. ``response.reasoning_summary_text.delta``
#: is the summary a caller asked for with ``reasoning.summary``;
#: ``response.reasoning_text.delta`` is raw reasoning text, which only some
#: models and some org verification levels emit. Both mean the same thing to
#: modelpass's vocabulary, so both become ``thinking`` (``openai`` 2.32.0).
_THINKING_DELTAS = frozenset(
    {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}
)

#: Events whose content this adapter already reports another way, so passing
#: them through as vendor events as well would report the same characters twice.
#: The ``.done`` events re-send the text their deltas already streamed; the
#: response-envelope events are the transport the final :class:`Response` is read
#: off, and everything in it reaches the caller as usage and as a terminal.
_SILENT_STREAM_TYPES = frozenset(
    {
        _TEXT_DELTA,
        *_THINKING_DELTAS,
        "response.output_text.done",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.done",
        "response.created",
        "response.in_progress",
        "response.queued",
        "response.completed",
        "response.incomplete",
        "response.failed",
    }
)

#: The three stream events that carry the finished :class:`Response`. All three
#: are read, not just ``completed``: an answer truncated at the ceiling arrives
#: as ``response.incomplete`` and is exactly the case :func:`terminal_status`
#: has to report on.
_FINAL_STREAM_TYPES = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)


def final_response(event: Any) -> Any | None:
    """The finished ``Response`` off a terminal stream event, or ``None``."""
    if getattr(event, "type", None) in _FINAL_STREAM_TYPES:
        return getattr(event, "response", None)
    return None


def _echo_note(sent: str | None, echoed: str | None) -> str | None:
    """The disagreement sentence, borrowed from the Codex adapter.

    Imported rather than reimplemented: the wording is what a caller reads on a
    receipt, and two runtimes describing the same disagreement differently would
    be the same fact in two voices. Imported lazily because that module is a
    sibling adapter and importing one adapter from another at module scope is
    how an optional extra becomes a hard dependency.
    """
    from .openai import reasoning_echo_note

    return reasoning_echo_note(sent, echoed)


def response_reasoning_echo(response: Any) -> str | None:
    """The effort the server says it ran at, off the finished ``Response``.

    **Driven 2026-09-22, and the drive is the reason this exists.** ``openai``
    2.32.0 types ``Response.reasoning`` as ``Optional[Reasoning]``, which
    established only that a field was declared -- a typed-but-never-populated
    field would have made this function a lie. A live call against
    ``gpt-5.4-mini`` settled it: ``reasoning={'effort': 'low'}`` came back as
    ``Reasoning(effort='low', ..., context='current_turn', mode='standard')`` in
    the raw payload, and ``'high'`` came back ``'high'`` -- so the field tracks
    the request rather than repeating a constant, which is the property that
    makes an echo worth reading at all.

    The same drive answered a question nobody had asked: with **no** effort
    sent, ``gpt-5.4-mini`` echoes ``'none'`` and spends zero reasoning tokens.
    This runtime's default is not ``high``, unlike Anthropic's.

    So ``openai-api`` joins ``openai-sdk`` as a runtime whose claim about itself
    can be checked. Anthropic remains the one where the receipt is the only
    record: its ``Message`` carries no effort field, driven the same day.
    """
    reasoning = getattr(response, "reasoning", None)
    effort = getattr(reasoning, "effort", None) if reasoning is not None else None
    return effort if isinstance(effort, str) and effort else None


def reasoning_echo_events(
    request: Any, response: Any, runtime: Runtime
) -> list[AgentEvent]:
    """The echo as a vendor event, in the shape ``openai-sdk`` already emits.

    One shape for both runtimes so the fold and the terminal need no second
    path: ``sent`` beside ``echoed``, and a ``note`` only when they disagree.
    Silent when the server echoed nothing, which is not an error -- a model with
    no reasoning surface answers without the field.
    """
    echoed = response_reasoning_echo(response)
    if echoed is None:
        return []
    sampling = getattr(request, "sampling", None)
    sent = getattr(sampling, "reasoning_effort", None) if sampling else None
    data: dict[str, Any] = {"sent": sent, "echoed": echoed}
    note = _echo_note(sent, echoed)
    if note is not None:
        data["note"] = note
    return [VendorEvent(runtime=runtime, name="reasoning_echo", data=data)]


def stream_events(
    event: Any, runtime: Runtime = Runtime.OPENAI_API
) -> list[AgentEvent]:
    """Map one SDK stream event onto the normalized vocabulary.

    Text and reasoning text are the two the vocabulary has words for. Every other
    event becomes a ``vendor_event`` named ``stream.<type>`` (rule 3) -- a
    function call's arguments streaming in, a web-search call, an ``error`` frame
    -- except those in :data:`_SILENT_STREAM_TYPES`, which would be a second
    report of something already reported.

    ``runtime`` names which runtime stamps the vendor events. It defaults to this
    module's own and exists because ``openai-compatible`` (ticket 1.10) reuses
    this Responses path unchanged when a connection selects ``wire="responses"``,
    and a vendor event stamped with the wrong runtime would be this adapter
    reporting a run that did not happen on it.
    """
    kind = getattr(event, "type", None)
    if kind == _TEXT_DELTA:
        delta = getattr(event, "delta", "")
        return [TextDeltaEvent(delta)] if isinstance(delta, str) and delta else []
    if kind in _THINKING_DELTAS:
        delta = getattr(event, "delta", "")
        return [ThinkingEvent(delta)] if isinstance(delta, str) and delta else []
    if not isinstance(kind, str) or kind in _SILENT_STREAM_TYPES:
        return []
    return [VendorEvent(runtime, f"stream.{kind}", _vendor_payload(event))]


def _vendor_payload(event: Any) -> dict[str, Any]:
    """A JSON-safe payload for a vendor event. Never raises (D11)."""
    dumped = _item_param(event)
    if isinstance(dumped, Mapping):
        try:
            json.dumps(dumped, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return {"repr": repr(event)[:500]}
        return dict(dumped)
    return {"repr": repr(event)[:500]}


def response_text(response: Any) -> str:
    """The assistant text of a finished response, output items concatenated.

    Read off the final response rather than accumulated from the deltas, which
    is what ``anthropic-api`` does and for the same reason: the deltas are a
    report to the caller, and the thing a schema is validated against should be
    the vendor's own final document.
    """
    parts: list[str] = []
    for item in getattr(response, "output", None) or ():
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", None) or ():
            if getattr(part, "type", None) == "output_text":
                text = getattr(part, "text", "")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def function_calls(response: Any) -> list[Any]:
    """The ``function_call`` items a finished response asked for."""
    return [
        item
        for item in (getattr(response, "output", None) or ())
        if getattr(item, "type", None) == "function_call"
    ]


def terminal_status(response: Any, *, unanswered_call: bool) -> tuple[TerminalStatus, str | None]:
    """Map a finished response's ``status`` onto a terminal status.

    Deliberately the same five judgements ticket 1.6 made against Anthropic's
    ``stop_reason``, against this vendor's own vocabulary, so a consumer matching
    on a terminal does not have to learn two sets of rules:

    * ``completed`` is ``ok``.
    * **A tool call on a run that declared no tools is ``error``**, and this is
      checked before ``status`` because such a response is ``completed`` as far
      as the vendor is concerned. The answer is unfinished, and calling it ``ok``
      is how a consumer ships a half-finished response. Same sentence as 1.6's.
    * ``incomplete`` with ``incomplete_details.reason == "max_output_tokens"`` is
      ``error``, carrying 1.6's wording: the answer is truncated mid-sentence,
      nothing downstream can tell that from a complete one, and the fix -- raise
      the ceiling -- belongs in the message rather than in a doc.
    * ``incomplete`` with ``content_filter`` is ``error``. The vendor stopped the
      answer; that is the same kind of event as Anthropic's ``refusal``.
    * ``failed`` is ``error``, carrying ``response.error.message`` where there is
      one.
    * ``cancelled`` is ``cancelled`` -- the vendor-side twin of what
      :meth:`OpenAIAPIAdapter.cancel` reports from this side.

    Anything else -- ``in_progress``, ``queued``, whatever the vocabulary grows
    next -- is ``ok`` with the raw status recorded, because an adapter inventing
    a verdict for a value it does not know is rule 3 broken in the other
    direction. A response that never arrived at all is its own ``error``: a
    stream that ended without one of the three terminal events produced no
    answer, and reporting ``ok`` for it would be reporting an empty success.
    """
    if response is None:
        return (
            TerminalStatus.ERROR,
            "the response stream ended without a response.completed, "
            "response.incomplete or response.failed event, so no answer arrived",
        )
    if unanswered_call:
        return (
            TerminalStatus.ERROR,
            "the model asked for a tool this run did not declare, so the answer "
            "is unfinished",
        )
    status = getattr(response, "status", None)
    if status in (None, "completed"):
        return TerminalStatus.OK, status
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        if reason == "max_output_tokens":
            return (
                TerminalStatus.ERROR,
                "the answer was truncated at the max_output_tokens ceiling; raise "
                "max_output_tokens for this call",
            )
        if reason == "content_filter":
            return (
                TerminalStatus.ERROR,
                "the vendor's content filter stopped this answer",
            )
        return TerminalStatus.ERROR, f"the response is incomplete ({reason!r})"
    if status == "failed":
        error = getattr(response, "error", None)
        message = getattr(error, "message", None)
        return TerminalStatus.ERROR, f"the vendor reported a failed response: {message}"
    if status == "cancelled":
        return TerminalStatus.CANCELLED, "the vendor reported the response cancelled"
    return TerminalStatus.OK, str(status)


def prompt_cache_retention(request: RunRequest) -> str | None:
    """The connection's stated cache lifetime, in this runtime's spelling (2026-09-21).

    The one place in modelpass where a connection's ``promptCache`` reaches a
    wire, and it reaches this one because ``openai`` 2.32.0 has a field for it:
    ``ResponseCreateParamsBase.prompt_cache_retention``, typed
    ``Optional[Literal["in-memory", "24h"]]`` and read on 2026-09-21. The
    connection's value is validated against exactly those two strings when the
    connection is built, so anything arriving here is a value the SDK types.

    ``None`` -- nothing sent, the endpoint's own retention stands -- for a
    connection that stated nothing *and* for one that asked for
    ``promptCache = "default"``. Those are different statements and this
    function deliberately collapses them, because the wire has no way to say
    "cache at whatever you normally do" other than by not saying anything. The
    distinction survives where it is readable: on the receipt.

    Note what this does **not** do. It sets no ``prompt_cache_key``: grouping
    requests so they land on the same cache is a caching *strategy*, it depends
    on what a caller considers one workload, and inventing a key from a
    connection name would be modelpass deciding that for everybody.
    """
    stated = request.connection.prompt_cache
    if stated is None:
        return None
    try:
        return plan_prompt_cache(
            stated, request.connection.runtime, name=request.connection.name
        ).ttl
    except SubpassError:  # pragma: no cover - defensive: refused at build
        return None


def _bare_terminal(
    status: TerminalStatus,
    reason: str | None,
    runtime: Runtime = Runtime.OPENAI_API,
    *,
    status_code: int | None = None,
    retry_after: float | None = None,
) -> TerminalEvent:
    """A terminal carrying a *status* only; the bridge owns the stamp (rule 2).

    ``status_code`` and ``retry_after`` are the two typed facts the bridge needs
    to compute a retryability verdict without reading ``reason`` (R6, ticket
    1.12). They ride here rather than being parsed back out of the message,
    which is the mistake this whole path exists to avoid.
    """
    return TerminalEvent(
        status=status,
        connection="",
        runtime=runtime,
        auth_mode=AuthMode.API_KEY,
        reason=reason,
        status_code=status_code,
        retry_after=retry_after,
    )


# --- the in-process tool loop -----------------------------------------------------


def invoke_handler(tool: ToolDef, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Run one caller tool and return an MCP result envelope. Never raises.

    Word for word ``anthropic_api.invoke_handler``'s two promises, because
    :class:`~modelpass.tools.ToolDef` makes them to a caller who does not know
    which runtime will run their function:

    * **A handler that raises becomes a failed tool result.** The model can
      retry it, try another tool, or explain -- killing the run instead throws
      away the turns already paid for.
    * **An async handler is driven to completion here**, on a private loop,
      because ``run()`` is a sync generator with no loop of its own.
    """
    handler = tool.handler
    if handler is None:
        return {
            "content": [{"type": "text", "text": f"tool {tool.name!r} has no handler"}],
            "is_error": True,
        }
    try:
        result = handler(dict(arguments))
        if inspect.isawaitable(result):
            result = asyncio.run(_await(result))
    except Exception as exc:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"tool {tool.name!r} failed: {type(exc).__name__}: {exc}",
                }
            ],
            "is_error": True,
        }
    return mcp_result_payload(result)


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _missing_tool(name: str) -> dict[str, Any]:
    """The result for a tool the model named and this run never declared."""
    return {
        "content": [{"type": "text", "text": f"tool {name!r} was not declared for this run"}],
        "is_error": True,
    }


def _call_arguments(call: Any) -> dict[str, Any]:
    """A function call's arguments, which arrive as a JSON *string* here.

    The other difference from Anthropic worth a function: ``tool_use.input`` is
    already a parsed object there, while ``ResponseFunctionToolCall.arguments``
    is typed ``str``. Unparseable JSON becomes ``{}`` rather than an exception --
    the handler then sees an empty mapping and can say so, which is a failed tool
    result the model can act on instead of a dead run.
    """
    raw = getattr(call, "arguments", None)
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# --- the adapter -----------------------------------------------------------------


#: Said once, because the receipt and the refusal must not drift apart.
_NO_MODEL = (
    "connection {name!r} names no model and this call passed none. The Responses "
    "API has no default model, so there is nothing to send: set a model on the "
    "connection, or pass model= to the call"
)


class OpenAIAPIAdapter(Adapter):
    """Drives OpenAI's Responses API with an explicit key."""

    runtime = Runtime.OPENAI_API

    #: A closed set, so an unknown key is *reported* rather than silently
    #: ignored. ``max_output_tokens`` is the ticket 1.6 alias
    #: :attr:`~modelpass.adapters.base.RunRequest.effective_sampling` folds into
    #: ``sampling=``; ``probe_credential`` opts into the model-list probe;
    #: ``reasoning_summary`` fills ``reasoning.summary`` for a caller who wants
    #: the ``thinking`` stream this runtime does not send by default.
    option_keys = frozenset({"max_output_tokens", "probe_credential", "reasoning_summary"})

    #: The refusal for a run with no model anywhere, said once so the receipt
    #: note and the raise cannot drift apart. A class attribute rather than a
    #: module constant read directly because ``openai-compatible`` (ticket 1.10)
    #: inherits this preflight and has its own sentence to say: its endpoint has
    #: no default model either, but the advice about where to find one is
    #: different when the endpoint is a box you run yourself.
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

        It is called as ``client_factory(api_key=..., base_url=...)`` and must
        return something with ``responses.stream(...)`` and, for the probe,
        ``models.list()``. The default builds ``openai.OpenAI(api_key=...,
        base_url=...)`` -- **never the bare constructor**, which would let the
        SDK's own discovery read ``OPENAI_API_KEY`` out of the process
        environment and bill an account nobody named (adapter contract rule 1,
        in-process clause; ``tests/test_no_ambient_credentials.py`` is where that
        stays checked).

        ``env`` and ``secrets`` are the resolution sources handed to
        :func:`~modelpass.preflight.resolve_credential`; ``None`` means the
        process environment and the store beside the connection file.
        """
        self._client_factory = client_factory or _default_client
        self._async_client_factory = async_client_factory or _default_async_client
        self._env = env
        self._secrets = secrets
        self._cancelled = threading.Event()

    def bind_resolution(
        self, *, env: Mapping[str, str] | None, secrets: Any = None
    ) -> None:
        """Point credential resolution at a bridge's own environment and secrets.

        Same contract as ``anthropic-api``'s, called by
        :meth:`~modelpass.Bridge.adapter_for` on an adapter it loaded itself:
        without it a bridge pointed at a test root or at an application's own
        directory would read the user's real ``~/.modelpass/secrets.toml``.
        """
        self._env = env
        self._secrets = secrets

    # --- availability ----------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Whether ``openai`` can be imported -- without importing it."""
        return importlib.util.find_spec(_MODULE) is not None

    def _async_client(self, connection: Connection) -> Any:
        """The vendor's *async* client (ticket 1.13), built the same never-ambient way.

        ``openai.AsyncOpenAI`` ships in the same package as ``openai.OpenAI`` and
        carries the same ``responses.stream(...)`` surface, so the async face
        costs no second dependency and no second pin.
        """
        key = resolve_credential(connection, self._env, secrets=self._secrets)
        return self._async_client_factory(api_key=key, base_url=connection.base_url)

    def _client(self, connection: Connection) -> Any:
        """The vendor client for this connection, built with an explicit key."""
        key = resolve_credential(connection, self._env, secrets=self._secrets)
        return self._client_factory(api_key=key, base_url=connection.base_url)

    # --- preflight -------------------------------------------------------------

    def preflight(self, request: RunRequest) -> Receipt:
        """The three API checks of :func:`~modelpass.preflight.api_preflight`.

        Nothing here launches, logs in, or spends. ``runtime_available`` means
        the vendor SDK imports; ``account`` carries a key fingerprint and never a
        key; ``plan_name`` and ``binary`` are ``None`` because an API key has no
        subscription plan and no executable. The model-list probe is **opt-in**
        (``options={"probe_credential": True}``) and cached per connection once
        taken, for the reason :meth:`Adapter.cached_probe` exists.

        Two notes this adapter adds to the three:

        * **No model resolves.** The Responses API has no "whatever the runtime
          defaults to" -- every request names a model -- so a connection with no
          ``model`` can be written and checked, but a call against it must pass
          ``model=``. A note rather than a failure, for the reason 1.6 gave: a
          keyless-model connection is a state setup is allowed to be in, and the
          refusal happens at :meth:`run`, before a single byte is sent.
        * **The schema had to be tightened.** ``strict: True`` requires a subset
          with no optional properties, so
          :func:`~modelpass.schema.openai_strict_issues` is reported here, before
          the call, naming each property that will come back mandatory-and-
          nullable. Otherwise the first report of it is an answer with keys the
          caller did not expect.
        """
        probe = None
        if request.options.get("probe_credential"):
            probe = self.cached_probe(request)
        receipt = api_preflight(
            request.connection,
            request.plan,
            self._env,
            runtime_available=self.is_available(),
            probe=probe,
            secrets=self._secrets,
        )
        if not receipt.ok:
            return receipt
        notes = list(receipt.notes)
        if not (request.model or request.connection.model):
            notes.append(self.no_model_message.format(name=request.connection.name))
        notes.extend(self.strict_schema_notes(request))
        return replace(receipt, notes=tuple(notes))

    @staticmethod
    def strict_schema_notes(request: RunRequest) -> tuple[str, ...]:
        """One note per way the caller's schema is not already OpenAI-strict."""
        if request.schema is None:
            return ()
        issues = openai_strict_issues(normalize_schema(request.schema))
        if not issues:
            return ()
        return (
            "the schema is sent with strict: true, which has no optional "
            "properties: modelpass tightened it with schema.to_openai_strict(), so "
            "every property comes back present and a formerly optional one may be "
            f"null. What was tightened: {'; '.join(issues)}",
        )

    def probe(self, request: RunRequest) -> ModelListProbe | None:
        """``models.list()`` -- token-free, and the only pre-run proof a key is live.

        Call it through :meth:`Adapter.cached_probe`, which is what
        :meth:`preflight` does.
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

    def support_for(
        self, capability: Capability, options: Mapping[str, Any]
    ) -> Support | None:
        """No opinion: nothing in ``options`` changes which vendor surface runs.

        Stated rather than inherited for the reason ``anthropic-api`` states it:
        on ``openai-sdk`` an option selects a transport and the transports differ
        in what they can do. Here the transport decision was taken once, in this
        module's docstring, and there is no option that reopens it.
        """
        del capability, options
        return None

    # --- the run ---------------------------------------------------------------

    def sampling_params(self, request: RunRequest) -> dict[str, Any]:
        """The sampling half of the request, in the vendor's own spelling (R5).

        **The per-model decisions are not taken here**: they are taken once, in
        :func:`~modelpass.sampling_rules.plan_sampling`, whose answer this method
        translates and does not revisit -- so a field this method sends and a
        field the receipt names cannot be two different sets. What is left is the
        translation, which is genuinely vendor knowledge:

        * ``temperature`` / ``top_p`` / ``max_output_tokens`` -> themselves. All
          three are in ``openai`` 2.32.0's ``Responses.create`` signature, and
          ``max_output_tokens`` is the field's actual name on this transport --
          one of the three reasons the Responses API was chosen.
        * ``top_k`` never arrives: it is in neither of the SDK's create
          signatures, the ``openai-api`` rules row does not accept it, and
          ``plan_sampling`` drops it with a note before this method runs.
        * ``reasoning_effort`` -> ``reasoning={"effort": ...}``.
          ``ReasoningEffort`` is
          ``Literal['none','minimal','low','medium','high','xhigh']`` in that
          SDK, so all three positions of modelpass's dial are accepted values.
        * ``options={"reasoning_summary": "auto"}`` joins it in the same
          ``reasoning`` object rather than replacing it. Nothing else writes
          there, but a merge is what keeps that true when something does --
          ``anthropic-api`` learned that lesson the hard way with
          ``output_config``.

        **Unlike ``anthropic-api`` there is no default ceiling.**
        ``max_output_tokens`` is optional on the Responses API, so a caller who
        names none gets the model's own limit; inventing a 4096 here would be
        modelpass truncating answers nobody asked it to truncate. The rules row
        says so too: ``required`` and ``defaults`` are both empty for this
        runtime.
        """
        plan = plan_sampling(
            request.effective_sampling,
            self.runtime,
            request.model or request.connection.model,
        )
        params: dict[str, Any] = {}
        reasoning: dict[str, Any] = {}
        for name, value in plan.applied.items():
            if name == "reasoning_effort":
                reasoning["effort"] = value
            else:
                params[name] = value
        summary = request.options.get("reasoning_summary")
        if isinstance(summary, str) and summary:
            reasoning["summary"] = summary
        if reasoning:
            params["reasoning"] = reasoning
        return params

    def request_params(
        self, request: RunRequest, items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """One round's request, assembled. Pure: no network, no client."""
        params: dict[str, Any] = {
            "model": request.model or request.connection.model,
            "input": items,
        }
        params.update(self.sampling_params(request))
        instructions = instructions_param(request)
        if instructions is not None:
            params["instructions"] = instructions
        if request.tools:
            params["tools"] = tool_params(request.tools)
        # D13: a schema and tools are never combined. The bridge refuses the
        # combination before an adapter sees it; the two branches here cannot
        # both fire on one request, and unlike anthropic-api's output_config
        # these two live in different parameters anyway.
        text = text_param(request)
        if text is not None:
            params["text"] = text
        retention = prompt_cache_retention(request)
        if retention is not None:
            params["prompt_cache_retention"] = retention
        return params

    def run(self, request: RunRequest) -> Iterator[AgentEvent]:
        """One stateless call, plus however many tool rounds the model asks for.

        The two refusals here are both :class:`PreflightFailed`, raised before
        the client is built and therefore before anything is sent: a credential
        that no longer resolves, and a run with no model anywhere. Neither is a
        vendor failure, so neither is a terminal event.
        """
        self._cancelled.clear()
        if not (request.model or request.connection.model):
            raise PreflightFailed(
                self.no_model_message.format(name=request.connection.name)
            )
        try:
            client = self._client(request.connection)
        except SubpassError:
            raise
        except Exception as exc:  # pragma: no cover - defensive: a client constructor
            raise VendorRunFailed(
                f"the {_PACKAGE} client could not be constructed: {type(exc).__name__}: {exc}"
            ) from exc
        return self._loop(client, request)

    def _loop(self, client: Any, request: RunRequest) -> Iterator[AgentEvent]:
        items = input_items(request)
        by_name = {tool.name: tool for tool in request.tools}
        final_text = ""

        while True:
            if self._cancelled.is_set():
                yield _bare_terminal(
                    TerminalStatus.CANCELLED, "cancelled by caller", self.runtime
                )
                return

            params = self.request_params(request, items)
            response = None
            try:
                with client.responses.stream(**params) as stream:
                    for event in stream:
                        response = final_response(event) or response
                        yield from stream_events(event, self.runtime)
            except SubpassError:
                raise
            except Exception as exc:
                status, reason = _vendor_failure(exc)
                code, after = vendor_error_facts(exc)
                yield _bare_terminal(
                    status, reason, self.runtime, status_code=code, retry_after=after
                )
                return

            yield from reasoning_echo_events(request, response, self.runtime)
            yield UsageEvent(
                usage=token_usage(getattr(response, "usage", None)),
                scope=UsageScope.DELTA,
            )

            text = response_text(response)
            if text:
                final_text = text

            calls = function_calls(response)
            # A run that declared no tools has nothing to answer with, and
            # replying "not declared" round after round would be a loop with a
            # bill on it. The model should never ask -- it was sent no tools --
            # so if it does, the run ends and terminal_status() says why.
            if not (request.tools and calls):
                yield from self._finish(request, response, final_text, bool(calls))
                return

            # The whole assistant turn goes back verbatim -- reasoning items
            # included -- before the outputs that answer it.
            items.extend(
                _item_param(item) for item in (getattr(response, "output", None) or ())
            )
            for call in calls:
                call_id = getattr(call, "call_id", "") or ""
                name = getattr(call, "name", "") or ""
                arguments = _call_arguments(call)
                yield ToolCallEvent(
                    name=name,
                    arguments=arguments,
                    id=call_id,
                    server=CALLER_TOOL_SERVER,
                )
                tool = by_name.get(name)
                payload = (
                    invoke_handler(tool, arguments) if tool is not None else _missing_tool(name)
                )
                is_error = bool(payload.get("is_error"))
                content = flatten_mcp_result(payload)
                yield ToolResultEvent(
                    id=call_id,
                    name=name,
                    content=content,
                    is_error=is_error,
                )
                # One item per call, keyed by call_id -- the Responses input
                # array has no "one user message carrying every result" shape to
                # choose, which makes the ordering question anthropic-api had to
                # answer not arise here at all.
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": content,
                    }
                )

    def arun(self, request: RunRequest) -> AsyncIterator[AgentEvent]:
        """:meth:`run`, on the caller's loop, with the vendor's async client (R1).

        Native rather than the base class's worker thread, and the visible
        difference is the tool loop: a coroutine handler is awaited here, on the
        loop that called ``achat``. :meth:`run` is unchanged and is not
        re-implemented over this.
        """
        self._cancelled.clear()
        if not (request.model or request.connection.model):
            raise PreflightFailed(
                self.no_model_message.format(name=request.connection.name)
            )
        try:
            client = self._async_client(request.connection)
        except SubpassError:
            raise
        except Exception as exc:  # pragma: no cover - defensive: a client constructor
            raise VendorRunFailed(
                f"the {_PACKAGE} async client could not be constructed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return native_run(self, self.aloop(client, request))

    async def aloop(self, client: Any, request: RunRequest) -> AsyncIterator[AgentEvent]:
        """:meth:`_loop` with ``await`` where it blocks. Same rounds, same events.

        Public-ish rather than underscored because ``openai-compatible``
        subclasses this adapter and dispatches to it by wire, exactly as it does
        for the synchronous pair.
        """
        items = input_items(request)
        by_name = {tool.name: tool for tool in request.tools}
        final_text = ""

        while True:
            if self._cancelled.is_set():
                yield _bare_terminal(
                    TerminalStatus.CANCELLED, "cancelled by caller", self.runtime
                )
                return

            params = self.request_params(request, items)
            response = None
            try:
                async with client.responses.stream(**params) as stream:
                    async for event in stream:
                        response = final_response(event) or response
                        for out in stream_events(event, self.runtime):
                            yield out
            except SubpassError:
                raise
            except Exception as exc:
                status, reason = _vendor_failure(exc)
                code, after = vendor_error_facts(exc)
                yield _bare_terminal(
                    status, reason, self.runtime, status_code=code, retry_after=after
                )
                return

            for _echo_event in reasoning_echo_events(
                request, response, self.runtime
            ):
                yield _echo_event
            yield UsageEvent(
                usage=token_usage(getattr(response, "usage", None)),
                scope=UsageScope.DELTA,
            )

            text = response_text(response)
            if text:
                final_text = text

            calls = function_calls(response)
            if not (request.tools and calls):
                for out in self._finish(request, response, final_text, bool(calls)):
                    yield out
                return

            items.extend(
                _item_param(item) for item in (getattr(response, "output", None) or ())
            )
            for call in calls:
                call_id = getattr(call, "call_id", "") or ""
                name = getattr(call, "name", "") or ""
                arguments = _call_arguments(call)
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
                is_error = bool(payload.get("is_error"))
                content = flatten_mcp_result(payload)
                yield ToolResultEvent(
                    id=call_id,
                    name=name,
                    content=content,
                    is_error=is_error,
                )
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": content,
                    }
                )

    def _finish(
        self, request: RunRequest, response: Any, final_text: str, unanswered_call: bool
    ) -> Iterator[AgentEvent]:
        """The structured answer, if one was asked for, and then the terminal."""
        status, reason = terminal_status(response, unanswered_call=unanswered_call)
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

    def cancel(self) -> None:
        """Stop the loop at the next round boundary (D10).

        Two halves, and only one of them is this method. The bridge closes the
        iterator, which exits the ``with`` around the stream and tears the
        in-flight HTTP response down -- the transport-level floor the contract
        documents. This flag is the graceful half: a loop that is between rounds
        stops there and reports ``cancelled`` rather than spending another round
        first.
        """
        self._cancelled.set()

    # --- sessions: refused, and the refusal says what to use instead ------------

    def open_session(self, request: SessionRequest) -> SessionHandle:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.new_chat()"))

    def resume_session(self, request: SessionRequest) -> SessionHandle:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.resume_chat()"))

    def list_sessions(self, request: SessionRequest) -> tuple[SessionInfo, ...]:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.list_sessions()"))


#: The one sentence all three session refusals share -- the same shape
#: ``anthropic-api`` uses, with this runtime's name in it (ticket 1.8's bridge
#: policy refuses every session door before an adapter is loaded; these stay
#: because an adapter that answered a session call would be a contract
#: violation).
_NO_SESSIONS = (
    f"the adapter for runtime {Runtime.OPENAI_API.value!r} has not implemented "
    "{call}, and will not: a session rests on the runtime holding the "
    "conversation (D14) and an API endpoint holds none -- every request carries "
    "its whole history. Use bridge.chat(history=[...]), which is the documented "
    "multi-turn shape against a stateless runtime. Stateless bridge.chat() calls "
    "are unaffected"
)

def _vendor_failure(exc: Exception) -> tuple[TerminalStatus, str]:
    """Classify a vendor exception into a terminal status and a reason.

    A 429 is :attr:`TerminalStatus.QUOTA_EXHAUSTED` and not an error, because it
    is the outcome a quota failover exists to act on -- typed off the status code
    the SDK carries rather than matched on the message, which is the substring
    matching that burned a real allowance once already (R6).
    ``openai.RateLimitError`` is an ``APIStatusError`` and carries
    ``status_code``, the same attribute name ``anthropic`` uses, so this is
    ``anthropic_api._vendor_failure`` unchanged.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return (
            TerminalStatus.QUOTA_EXHAUSTED,
            f"the endpoint reported HTTP 429 (rate limit / quota): {exc}",
        )
    if isinstance(status_code, int):
        return TerminalStatus.ERROR, f"HTTP {status_code}: {exc}"
    return TerminalStatus.ERROR, f"{type(exc).__name__}: {exc}"


def _default_client(*, api_key: str, base_url: str | None) -> Any:
    """``openai.OpenAI``, constructed with an explicit key and nothing ambient.

    ``base_url`` is passed only when the connection carries one, so a connection
    without one gets the SDK's own default endpoint rather than a ``None`` the
    constructor has to interpret.
    """
    try:
        sdk = importlib.import_module(_MODULE)
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise RuntimeNotAvailable(Runtime.OPENAI_API.value, _EXTRA, _PACKAGE) from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    return sdk.OpenAI(**kwargs)


def _default_async_client(*, api_key: str, base_url: str | None) -> Any:
    """``openai.AsyncOpenAI``, under exactly the rules of :func:`_default_client`."""
    try:
        sdk = importlib.import_module(_MODULE)
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise RuntimeNotAvailable(Runtime.OPENAI_API.value, _EXTRA, _PACKAGE) from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    return sdk.AsyncOpenAI(**kwargs)
