"""The ``google-api`` adapter: the Gemini API, driven in this process (ticket 1.11).

The fourth and last of the API-key adapters, and -- like the second and the
third -- a deliberate copy of the first.
:mod:`modelpass.adapters.anthropic_api` set the pattern in ticket 1.6 (a
``client_factory`` seam, :func:`~modelpass.preflight.api_preflight`, an
in-adapter tool loop with no turn cap, :func:`~modelpass.sampling_rules.rules_for`
deciding the sampling, a 429 typed off the status code the SDK carries) and this
module keeps the same shape, the same order and, where the vendors agree, the
same sentences. Four adapters, one family.

**Gating: this runtime ships ungated and the two Google *subscription* runtimes
do not.** That asymmetry is the decision rather than an oversight (the D5
amendment recorded in ``docs/legal/google.md`` and in
:data:`~modelpass.runtimes.EXPERIMENTAL_RUNTIMES`): the Antigravity Additional
Terms prohibit third-party software reaching the subscription Service, and the
same terms put an API-key holder under the Google Cloud terms *instead of* them.
Nothing in this ticket touches that gating in either direction.

**Where Gemini genuinely differs from the three runtimes before it**, and where
each difference lands:

* **The stream is a sequence of whole responses, not typed events.**
  ``client.models.generate_content_stream(...)`` yields
  ``GenerateContentResponse`` chunks, each carrying whatever parts arrived --
  there is no ``type`` string to match on. So the event mapping here reads
  *parts* (:func:`chunk_events`) rather than event names, and the vendor-event
  rule 3 fallback is named ``part.<field>`` where the other three say
  ``stream.<type>``.
* **Thinking is a flag on a text part.** ``Part.thought`` is a boolean and a
  thought part carries ordinary ``text``, which is why a naive reader would
  stream the model's private reasoning into the answer. The parts are separated
  here, and the model only emits them at all when the request asks --
  ``options={"include_thoughts": True}`` fills
  ``thinking_config.include_thoughts``, which modelpass does not send on its own
  because thought output is billed output nobody requested. That is ticket 1.9's
  ``reasoning_summary`` decision, applied to this vendor's spelling of it.
* **The assistant role is called ``model``.** A conversation sent with
  ``role="assistant"`` is not a conversation this API understands, so
  :func:`content_items` translates. It is one line, and it is the single most
  likely thing to be wrong in an adapter written from another vendor's memory.
* **Tool results go back as ``function_response`` parts in a ``user`` turn**,
  after the model's own turn is replayed verbatim -- ``thought_signature``
  included, for the same reason ticket 1.9 replays reasoning items: a model that
  cannot see its own previous thinking re-derives it every round.
* **Structured output is a mime type plus a JSON Schema.**
  ``response_mime_type="application/json"`` with ``response_json_schema=<the
  caller's schema>``. The SDK carries two schema fields and they are mutually
  exclusive: ``response_schema`` is the vendor's own ``Schema`` object and
  ``response_json_schema`` takes JSON Schema, which is what a modelpass caller
  already has. No strict rewrite happens here -- that was an OpenAI requirement,
  not a universal one -- so a caller's optional property stays optional.
* **Sampling includes ``top_k``**, which neither OpenAI runtime accepts. The
  reasoning dial lands on ``thinking_config.thinking_level``; see
  :meth:`GoogleAPIAdapter.sampling_params` for why it is a *level* and never a
  budget.
* **Usage arrives in four counts rather than two.** See :func:`token_usage`,
  which is the one function in this module worth reading before trusting a
  number off it.

**No ``cache_control``, and this one deserves a sentence.** Gemini has two
caching mechanisms and neither is an in-message breakpoint: implicit caching is
automatic and unaddressable, and explicit caching is a *separate resource* --
``client.caches.create(...)`` returns a named cached-content object with a TTL
that is then referenced by ``config.cached_content``. An Anthropic-shaped
breakpoint has nowhere to go in either, so the system prompt is flattened with
:attr:`~modelpass.types.Message.flat_text` semantics and the markers are
dropped; the bridge's own R3 disclosure names the drop on the receipt, and
ticket 1.5 put that sentence in one place on purpose. **Explicit context caching
stays out of this ticket**: creating a cache object would make modelpass the
owner of a billed, TTL'd, server-side resource nobody asked it to create, which
is a feature with its own decision record and not a line in an adapter.

**What it deliberately does not do.** Sessions, for the reason ticket 1.8 wrote
into the bridge: an API endpoint holds no history. The three methods keep their
own refusal anyway, worded exactly as the other three adapters' are. Note that
the SDK *does* have ``client.chats.create(...)``, which is a client-side history
helper rather than a vendor-held conversation -- it is a Python list of turns in
this process, which is what ``bridge.chat(history=[...])`` already is, and it is
not D14's resumable, listable, forkable session either.
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
from ..retry import vendor_error_facts
from ..runtimes import Runtime
from ..sampling_rules import plan_sampling
from ..schema import (
    build_structured_event,
    normalize_schema,
    resolve_schema_name,
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

__all__ = ["SDK_VERSION_READ", "GoogleAPIAdapter", "terminal_status", "token_usage"]

_MODULE = "google.genai"
_EXTRA = "google-api"
_PACKAGE = "google-genai"

#: The installed SDK this adapter was written against and whose surface every
#: capability note cites. Data rather than prose for the reason ticket 1.6 gave:
#: the notes are evidence, and evidence names its source. Read on 2026-09-13.
SDK_VERSION_READ = "1.73.1"

#: How many model ids :meth:`GoogleAPIAdapter.probe` keeps. ``models.list()``
#: here returns a ``Pager`` that fetches further pages as it is iterated, so the
#: truncation is also what stops the probe paging through a catalogue to answer
#: "is this credential live".
_PROBE_LIMIT = 20

#: What each modelpass role is called on this API. Gemini's assistant turn is
#: ``model``; everything else this adapter can see is a user turn.
_ROLES: dict[str, str] = {"user": "user", "assistant": "model"}


# --- usage -----------------------------------------------------------------------


def _count(usage: Any, key: str) -> int:
    """One integer off a usage object or mapping; anything else is zero."""
    if isinstance(usage, Mapping):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def token_usage(usage: Any) -> TokenUsage:
    """Normalize ``GenerateContentResponseUsageMetadata`` onto modelpass tokens.

    Four vendor counts, four modelpass fields, and **two of the four assignments
    are arithmetic rather than a rename** -- both of them made so that a
    consumer's accounting cannot tell that a run changed runtimes.

    * ``input_tokens`` is ``prompt_token_count - cached_content_token_count``,
      plus ``tool_use_prompt_token_count``. The SDK's own field description says
      ``prompt_token_count`` includes the cached content when there is any, and
      :class:`~modelpass.types.TokenUsage` counts ``cached_input_tokens``
      *separately from* ``input_tokens`` -- so assigning both straight across
      would count the cached prefix twice. This is ticket 1.9's subtraction
      against a different vendor's spelling of the same shape.
      ``tool_use_prompt_token_count`` is prompt-side tokens for the vendor's
      *own* server-side tools, which this adapter never sends, so in practice it
      is zero; it is added rather than dropped because the total below must come
      out right if one ever arrives.
    * ``cached_input_tokens`` is ``cached_content_token_count``: the read side of
      Gemini's caching, implicit or explicit.
    * ``output_tokens`` is ``candidates_token_count + thoughts_token_count``, and
      this is the assignment worth arguing about. On ``anthropic-api`` and
      ``openai-api`` the thinking tokens are already *inside* the vendor's own
      output count, and both adapters leave them there. Gemini reports them
      beside it -- ``total_token_count`` is documented in the SDK as the sum of
      ``prompt_token_count``, ``candidates_token_count``,
      ``tool_use_prompt_token_count`` and ``thoughts_token_count``, which is the
      evidence that they are not already counted in ``candidates_token_count``
      and so cannot be double-counted by adding them. Leaving them out would
      make a thinking run's ``output_tokens`` mean something narrower here than
      it means on the other three runtimes, and the caller pays for them either
      way.
    * ``cache_write_tokens`` is ``0``, always. A ``generateContent`` response
      reports no write count: implicit caching bills no write premium, and an
      explicit cached-content object is created through a different call this
      adapter does not make.

    The result is that ``TokenUsage.total_tokens`` equals the vendor's own
    ``total_token_count``, which is the arithmetic
    ``tests/test_adapter_google_api.py`` pins.
    """
    cached = _count(usage, "cached_content_token_count")
    prompt = _count(usage, "prompt_token_count")
    return TokenUsage(
        input_tokens=max(prompt - cached, 0) + _count(usage, "tool_use_prompt_token_count"),
        output_tokens=(
            _count(usage, "candidates_token_count") + _count(usage, "thoughts_token_count")
        ),
        cached_input_tokens=cached,
        cache_write_tokens=0,
    )


# --- request assembly ------------------------------------------------------------


def flatten(blocks: Sequence[TextBlock]) -> str:
    """Blocks as one string, with :attr:`~modelpass.types.Message.flat_text`
    semantics.

    One blank line between non-empty blocks -- deliberately the same join
    ticket 1.5 chose for the runtimes that cannot carry blocks, and for the same
    reason: it is what the flattening produced before blocks existed, so a prompt
    written for this runtime keeps reading the way it always did. The
    ``cache_control`` markers on the blocks have nowhere to go here (see this
    module's docstring on Gemini's two caching mechanisms) and are dropped; the
    bridge's own disclosure names the drop on the receipt (R3), and this function
    does not restate it.
    """
    return "\n\n".join(block.text for block in blocks if block.text)


def system_instruction(request: RunRequest) -> str | None:
    """The ``config.system_instruction`` string, or ``None``.

    ``system_instruction`` is typed as a union that includes ``str`` and a
    ``Content``; a string is the shape that carries a text prompt with nothing
    invented around it. ``None`` rather than ``""`` so a run with no system
    message omits the field instead of sending an empty one.
    """
    blocks = request.system_blocks
    if not blocks:
        return None
    return flatten(blocks) or None


def content_items(request: RunRequest) -> list[dict[str, Any]]:
    """The ``contents=`` array: one ``Content`` per non-system turn.

    ``{"role": ..., "parts": [{"text": ...}]}`` is ``ContentDict``, and the role
    translation is the one in :data:`_ROLES`: Gemini's assistant is called
    ``model``.
    """
    return [
        {"role": _ROLES.get(role, "user"), "parts": [{"text": flatten(blocks)}]}
        for role, blocks in request.conversation_blocks
    ]


def tool_params(tools: Sequence[ToolDef]) -> list[dict[str, Any]]:
    """Caller tools as one ``Tool`` carrying every function declaration.

    One entry rather than one per tool, because ``Tool.function_declarations`` is
    a list and that is the shape the API documents.

    ``parameters_json_schema`` rather than ``parameters``: the latter is the
    vendor's own ``Schema`` object, while the former takes JSON Schema, which is
    what :meth:`~modelpass.tools.ToolDef.json_schema` already produces. Handing
    a caller's schema to the field that accepts it is the difference between
    passing it through and translating it, and a translation is where a tool's
    contract quietly changes (``google-genai`` 1.73.1, read 2026-09-13).
    """
    return [
        {
            "function_declarations": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters_json_schema": tool.json_schema(),
                }
                for tool in tools
            ]
        }
    ]


def response_format(request: RunRequest) -> dict[str, Any]:
    """The native structured-output configuration, or ``{}`` (D13).

    ``response_mime_type="application/json"`` plus ``response_json_schema``, read
    off ``google-genai`` 1.73.1 on 2026-09-13. The mime type is **required** when
    a schema is sent -- the SDK's own field documentation says so -- which is why
    both keys move together and neither is sent alone.

    The caller's schema is normalized and passed through. No strict rewrite: that
    was an OpenAI ``strict: true`` requirement, not a universal one, and Gemini
    documents a JSON Schema subset that keeps optional properties optional. What
    it does *not* support (``$comment``, ``patternProperties`` and the rest) is
    the vendor's to report, and modelpass validates the answer it gets back
    either way -- which is the check that catches a keyword the vendor ignored.
    """
    if request.schema is None:
        return {}
    return {
        "response_mime_type": "application/json",
        "response_json_schema": normalize_schema(request.schema),
    }


def _dump(value: Any) -> Any:
    """One vendor model as the plain document it goes back up as.

    ``exclude_none=True`` where the model supports it, because a ``ContentDict``
    has *absent* optional keys rather than null ones; a ``model_dump()`` full of
    ``None``s is not the same document.
    """
    dump = getattr(value, "model_dump", None)
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
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, Mapping):
        return dict(value)
    return value


# --- event mapping ---------------------------------------------------------------

#: The part fields this adapter has a word for. Everything else becomes a vendor
#: event named ``part.<field>`` (rule 3): executable code, a code-execution
#: result, inline data, a file reference, whatever the SDK grows next.
_KNOWN_PART_FIELDS = ("text", "function_call", "function_response")

#: Part fields that are reported another way, or are not content at all, and so
#: must not become a vendor event. ``function_call`` reaches the caller as a
#: :class:`ToolCallEvent` from the tool loop and ``function_response`` is
#: modelpass's own answer coming back in a replayed turn; ``thought`` is the flag
#: that already decided whether the text was a ``thinking`` event; and
#: ``thought_signature`` / ``part_metadata`` are opaque tokens the model turn
#: carries back to the vendor, not something a caller can act on.
_SILENT_PART_FIELDS = frozenset(
    {
        "text",
        "function_call",
        "function_response",
        "thought",
        "thought_signature",
        "part_metadata",
    }
)


def candidates(chunk: Any) -> list[Any]:
    """The candidates on one chunk, or an empty list."""
    return list(getattr(chunk, "candidates", None) or ())


def parts_of(candidate: Any) -> list[Any]:
    """The parts of one candidate's content, or an empty list."""
    content = getattr(candidate, "content", None)
    return list(getattr(content, "parts", None) or ())


def part_events(part: Any, runtime: Runtime = Runtime.GOOGLE_API) -> list[AgentEvent]:
    """Map one content part onto the normalized vocabulary.

    Text is the word the vocabulary has, and ``Part.thought`` is the flag that
    decides which of two words it is: a thought part carries ordinary ``text``
    and is the model's private reasoning, so it becomes a ``thinking`` event and
    never a ``text_delta``. Getting that wrong would stream the model's scratch
    work into the answer a user reads, which is the failure mode this vendor's
    shape makes easy.

    Function calls are handled by the loop rather than here -- they are answered,
    not just reported -- and everything else becomes a vendor event.
    """
    text = getattr(part, "text", None)
    if isinstance(text, str) and text:
        if getattr(part, "thought", False):
            return [ThinkingEvent(text)]
        return [TextDeltaEvent(text)]
    for name in _KNOWN_PART_FIELDS:
        if getattr(part, name, None) is not None:
            return []
    others = _other_fields(part)
    if not others:
        return []
    name, value = others[0]
    return [VendorEvent(runtime, f"part.{name}", _vendor_payload(value))]


def _other_fields(part: Any) -> list[tuple[str, Any]]:
    """The set fields of a part that modelpass has no word for, in a stable order."""
    dumped = _dump(part)
    if not isinstance(dumped, Mapping):
        return []
    return [
        (name, value)
        for name, value in dumped.items()
        if value is not None and name not in _SILENT_PART_FIELDS
    ]


def _vendor_payload(value: Any) -> dict[str, Any]:
    """A JSON-safe payload for a vendor event. Never raises (D11)."""
    dumped = _dump(value)
    if isinstance(dumped, Mapping):
        try:
            json.dumps(dumped, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return {"repr": repr(value)[:500]}
        return dict(dumped)
    return {"value": dumped if isinstance(dumped, (str, int, float, bool)) else repr(value)[:500]}


def _reason_value(reason: Any) -> str | None:
    """A ``FinishReason`` (or a plain string) as the string this module matches on."""
    if reason is None:
        return None
    value = getattr(reason, "value", reason)
    return str(value)


#: Finish reasons that mean "the vendor stopped this answer", with the category
#: named on the terminal. Read off ``google-genai`` 1.73.1's ``FinishReason``
#: enum on 2026-09-13; every one of them is a *content* judgement, which is the
#: same kind of event as Anthropic's ``refusal`` and OpenAI's ``content_filter``.
_BLOCKED_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }
)

#: Finish reasons that mean the answer is unfinished for a mechanical reason.
#: Separate from the set above because the sentence a caller needs is different:
#: nothing was refused, something went wrong.
_BROKEN_REASONS = frozenset({"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "OTHER"})


def terminal_status(
    finish_reason: Any,
    *,
    unanswered_call: bool,
    blocked: str | None = None,
    answered: bool = True,
) -> tuple[TerminalStatus, str | None]:
    """Map a finished generation onto a terminal status.

    Deliberately the same judgements tickets 1.6 and 1.9 made against their own
    vendors' vocabularies, so a consumer matching on a terminal does not have to
    learn a third set of rules:

    * **A blocked prompt is an error and names the category**, and it is checked
      first because a blocked prompt produces no candidates at all -- there is no
      finish reason to read, and a run that reported ``ok`` here would be
      reporting an empty success.
    * **A function call on a run that declared no tools is ``error``**, checked
      before the finish reason because such a response is ``STOP`` as far as the
      vendor is concerned. The answer is unfinished, and calling it ``ok`` is how
      a consumer ships a half-finished response. Same sentence as 1.6's.
    * ``STOP`` (and ``FINISH_REASON_UNSPECIFIED``) is ``ok``.
    * ``MAX_TOKENS`` is ``error``, carrying 1.6's wording: the answer is
      truncated mid-sentence, nothing downstream can tell that from a complete
      one, and the fix -- raise the ceiling -- belongs in the message rather than
      in a doc.
    * ``SAFETY`` / ``RECITATION`` / the rest of :data:`_BLOCKED_REASONS` are
      ``error`` with the category in the reason.
    * ``MALFORMED_FUNCTION_CALL`` and friends are ``error`` too, with a different
      sentence: the answer is broken rather than refused.
    * A stream that produced no finish reason **and no content** is ``error``:
      nothing arrived. A stream that produced content but no finish reason is
      ``ok``, because a vendor that stops sending the field is not a vendor that
      failed.

    Anything else -- a value the enum grows next -- is ``ok`` with the raw
    reason recorded, because an adapter inventing a verdict for a value it does
    not know is rule 3 broken in the other direction.
    """
    if blocked:
        return (
            TerminalStatus.ERROR,
            f"the vendor blocked the prompt before answering ({blocked})",
        )
    if unanswered_call:
        return (
            TerminalStatus.ERROR,
            "the model asked for a tool this run did not declare, so the answer "
            "is unfinished",
        )
    reason = _reason_value(finish_reason)
    if reason in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
        if reason is None and not answered:
            return (
                TerminalStatus.ERROR,
                "the response stream ended without a candidate, a finish reason or "
                "any content, so no answer arrived",
            )
        return TerminalStatus.OK, reason
    if reason == "MAX_TOKENS":
        return (
            TerminalStatus.ERROR,
            "the answer was truncated at the max_output_tokens ceiling; raise "
            "max_output_tokens for this call",
        )
    if reason in _BLOCKED_REASONS:
        return (
            TerminalStatus.ERROR,
            f"the vendor stopped this answer ({reason})",
        )
    if reason in _BROKEN_REASONS:
        return (
            TerminalStatus.ERROR,
            f"the generation ended unfinished ({reason})",
        )
    return TerminalStatus.OK, reason


def _bare_terminal(
    status: TerminalStatus,
    reason: str | None,
    runtime: Runtime = Runtime.GOOGLE_API,
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


def call_arguments(call: Any) -> dict[str, Any]:
    """A function call's arguments, which arrive already parsed here.

    ``FunctionCall.args`` is typed ``Optional[dict[str, Any]]`` -- unlike
    OpenAI's, which is a JSON string -- so there is nothing to parse and nothing
    that can fail to parse. A string is still accepted, because a fake or a
    future wire format that sent one should reach the handler rather than the
    traceback; an unparseable one becomes ``{}``, which the handler can say
    something about.
    """
    raw = getattr(call, "args", None)
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def function_response_part(
    call: Any, name: str, content: str, *, is_error: bool
) -> dict[str, Any]:
    """One tool result, in the part shape the model reads it back as.

    ``response`` is a *mapping*, not a string, so the result is wrapped. The key
    says which kind of outcome it was: ``error`` for a failed tool, ``result``
    otherwise. A model that can see the difference can retry or explain, which is
    the whole reason a raising handler becomes a result instead of a dead run.

    ``id`` is carried when the vendor supplied one. ``FunctionCall.id`` is
    ``Optional`` on the Gemini API and is usually absent, which is why the
    modelpass ``ToolCallEvent`` id falls back to the function's name: an event
    with an empty id cannot be paired with its result by a consumer reading the
    stream.
    """
    part: dict[str, Any] = {
        "function_response": {
            "name": name,
            "response": {"error" if is_error else "result": content},
        }
    }
    call_id = getattr(call, "id", None)
    if isinstance(call_id, str) and call_id:
        part["function_response"]["id"] = call_id
    return part


# --- the adapter -----------------------------------------------------------------


#: Said once, because the receipt and the refusal must not drift apart.
_NO_MODEL = (
    "connection {name!r} names no model and this call passed none. The Gemini "
    "API has no default model, so there is nothing to send: set a model on the "
    "connection, or pass model= to the call"
)


class GoogleAPIAdapter(Adapter):
    """Drives Google's Gemini API with an explicit key."""

    runtime = Runtime.GOOGLE_API

    #: A closed set, so an unknown key is *reported* rather than silently
    #: ignored. ``max_output_tokens`` is the ticket 1.6 alias
    #: :attr:`~modelpass.adapters.base.RunRequest.effective_sampling` folds into
    #: ``sampling=``; ``probe_credential`` opts into the model-list probe;
    #: ``include_thoughts`` asks the model to emit the thought parts this
    #: adapter maps to ``thinking``, which modelpass does not request on its own.
    option_keys = frozenset({"max_output_tokens", "probe_credential", "include_thoughts"})

    #: The refusal for a run with no model anywhere, said once so the receipt
    #: note and the raise cannot drift apart.
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
        return something with ``models.generate_content_stream(...)`` and, for
        the probe, ``models.list()``. The default builds
        ``google.genai.Client(api_key=...)`` -- **never the bare constructor**,
        which would let the SDK's own discovery read ``GEMINI_API_KEY`` *or*
        ``GOOGLE_API_KEY`` out of the process environment and bill an account
        nobody named (adapter contract rule 1, in-process clause; two ambient
        names rather than one, and ``tests/test_no_ambient_credentials.py`` sets
        a decoy for both).

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

        Same contract as the other three API adapters', called by
        :meth:`~modelpass.Bridge.adapter_for` on an adapter it loaded itself:
        without it a bridge pointed at a test root or at an application's own
        directory would read the user's real ``~/.modelpass/secrets.toml``.
        """
        self._env = env
        self._secrets = secrets

    # --- availability ----------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Whether ``google.genai`` can be imported -- without importing it.

        ``_MODULE`` is dotted, and :func:`importlib.util.find_spec` *raises*
        ``ModuleNotFoundError`` for the missing parent rather than answering
        ``None``: on a machine without the extra, ``google`` itself is absent.
        The absence of an optional package is an answer here, never an
        exception, so the raise is caught and reported as "no".
        """
        try:
            return importlib.util.find_spec(_MODULE) is not None
        except (ImportError, ValueError):
            return False

    def _async_client(self, connection: Connection) -> Any:
        """The vendor's *async* client (ticket 1.13), built the same never-ambient way.

        ``google-genai`` puts its async surface on ``client.aio`` rather than in
        a second class, so the default factory hands back ``Client(...).aio``:
        the object this adapter then talks to has the same
        ``models.generate_content_stream`` attribute path as the sync one, and
        :meth:`_aloop` reads as :meth:`_loop` with ``await`` in it rather than as
        a second translation of the same API.
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

        One note this adapter adds to the three: **no model resolves.** The
        Gemini API has no "whatever the runtime defaults to" -- every request
        names a model -- so a connection with no ``model`` can be written and
        checked, but a call against it must pass ``model=``. A note rather than a
        failure, for the reason 1.6 gave: a modelless connection is a state setup
        is allowed to be in, and the refusal happens at :meth:`run`, before a
        single byte is sent.
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
        if request.model or request.connection.model:
            return receipt
        return replace(
            receipt,
            notes=(*receipt.notes, self.no_model_message.format(name=request.connection.name)),
        )

    def probe(self, request: RunRequest) -> ModelListProbe | None:
        """``models.list()`` -- token-free, and the only pre-run proof a key is live.

        Call it through :meth:`Adapter.cached_probe`, which is what
        :meth:`preflight` does. A model here is named ``models/gemini-2.5-flash``
        in its ``name`` field rather than carrying an ``id``, so that is what is
        read; ``id`` is accepted too, because a pager of something else should
        not make the probe silently report an empty catalogue.
        """
        try:
            listing = self._client(request.connection).models.list()
        except SubpassError:
            raise
        except Exception as exc:
            return ModelListProbe(ok=False, detail=f"{type(exc).__name__}: {exc}")
        ids: list[str] = []
        for item in listing or ():
            name = getattr(item, "name", None) or getattr(item, "id", None)
            if isinstance(name, str):
                ids.append(name)
            if len(ids) >= _PROBE_LIMIT:
                break
        return ModelListProbe(ok=True, models=tuple(ids))

    def support_for(
        self, capability: Capability, options: Mapping[str, Any]
    ) -> Support | None:
        """No opinion: nothing in ``options`` changes which vendor surface runs.

        Stated rather than inherited for the reason ``anthropic-api`` states it:
        on ``openai-sdk`` an option selects a transport and the transports differ
        in what they can do. Here there is one surface --
        ``models.generate_content_stream`` -- and no option that reopens the
        choice. ``include_thoughts`` asks for *more of* what that surface sends,
        which is not the same as selecting another one.
        """
        del capability, options
        return None

    # --- the run ---------------------------------------------------------------

    def sampling_params(self, request: RunRequest) -> dict[str, Any]:
        """The sampling half of the config, in the vendor's own spelling (R5).

        **The per-model decisions are not taken here**: they are taken once, in
        :func:`~modelpass.sampling_rules.plan_sampling`, whose answer this method
        translates and does not revisit -- so a field this method sends and a
        field the receipt names cannot be two different sets. What is left is the
        translation:

        * ``temperature`` / ``top_p`` / ``top_k`` / ``max_output_tokens`` ->
          themselves. All four are fields of ``GenerateContentConfig``, and
          **``top_k`` is the one this runtime has that neither OpenAI runtime
          does** -- it is dropped with a note there and sent here, which is the
          whole reason the rules table is keyed by runtime.
        * ``reasoning_effort`` -> ``thinking_config.thinking_level``.

        **The thinking decision, recorded because the ticket asked for it either
        way: a level, never a budget.** ``ThinkingConfig`` carries both
        ``thinking_budget`` and ``thinking_level``. The budget is a *token
        count*, and the SDK's own field documentation says "the default values
        and allowed ranges are model dependent" -- so turning three words into
        three numbers would mean inventing a number per model out of nothing,
        which is exactly what ticket 1.7 refused to do for Anthropic's thinking
        budget. ``ThinkingLevel``, on the other hand, is
        ``MINIMAL | LOW | MEDIUM | HIGH``, and modelpass's own
        :data:`~modelpass.types.REASONING_EFFORTS` is ``low | medium | high``:
        the three positions of the dial map onto three of the vendor's own named
        values with nothing added and nothing chosen. That is a translation
        rather than an invention, so it is the one that ships.

        ``options={"include_thoughts": True}`` joins the same ``thinking_config``
        object rather than replacing it -- a merge is what keeps that true when
        something else writes there, which is the lesson ``anthropic-api``
        learned the hard way with ``output_config``.

        **No default ceiling**, as on ``openai-api``: ``max_output_tokens`` is
        optional here, so a caller who names none gets the model's own limit
        rather than a number modelpass invented.
        """
        plan = plan_sampling(
            request.effective_sampling,
            self.runtime,
            request.model or request.connection.model,
        )
        params: dict[str, Any] = {}
        thinking: dict[str, Any] = {}
        for name, value in plan.applied.items():
            if name == "reasoning_effort":
                thinking["thinking_level"] = str(value).upper()
            else:
                params[name] = value
        if request.options.get("include_thoughts"):
            thinking["include_thoughts"] = True
        if thinking:
            params["thinking_config"] = thinking
        return params

    def request_params(
        self, request: RunRequest, contents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """One round's request, assembled. Pure: no network, no client.

        Everything but ``model`` and ``contents`` goes inside ``config``, which
        is where this API keeps it: the SDK types that argument as a
        ``GenerateContentConfig`` *or the dict of one*, and a dict is what this
        module builds so that request assembly never imports the vendor.
        """
        config: dict[str, Any] = dict(self.sampling_params(request))
        instruction = system_instruction(request)
        if instruction is not None:
            config["system_instruction"] = instruction
        if request.tools:
            config["tools"] = tool_params(request.tools)
        # D13: a schema and tools are never combined. The bridge refuses the
        # combination before an adapter sees it; the two branches here cannot
        # both fire on one request.
        config.update(response_format(request))
        params: dict[str, Any] = {
            "model": request.model or request.connection.model,
            "contents": contents,
        }
        if config:
            params["config"] = config
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
        contents = content_items(request)
        by_name = {tool.name: tool for tool in request.tools}
        final_text = ""

        while True:
            if self._cancelled.is_set():
                yield _bare_terminal(
                    TerminalStatus.CANCELLED, "cancelled by caller", self.runtime
                )
                return

            params = self.request_params(request, contents)
            usage: Any = None
            finish_reason: Any = None
            blocked: str | None = None
            turn: list[dict[str, Any]] = []
            calls: list[Any] = []
            text = ""
            answered = False
            try:
                stream = client.models.generate_content_stream(**params)
                try:
                    for chunk in stream:
                        usage = getattr(chunk, "usage_metadata", None) or usage
                        blocked = blocked or _blocked_reason(chunk)
                        for candidate in candidates(chunk):
                            answered = True
                            finish_reason = (
                                getattr(candidate, "finish_reason", None) or finish_reason
                            )
                            content = getattr(candidate, "content", None)
                            if content is not None and getattr(content, "parts", None):
                                turn.append(_dump(content))
                            for part in parts_of(candidate):
                                part_text = getattr(part, "text", None)
                                if (
                                    isinstance(part_text, str)
                                    and part_text
                                    and not getattr(part, "thought", False)
                                ):
                                    text += part_text
                                call = getattr(part, "function_call", None)
                                if call is not None:
                                    calls.append(call)
                                yield from part_events(part, self.runtime)
                finally:
                    closer = getattr(stream, "close", None)
                    if callable(closer):
                        closer()
            except SubpassError:
                raise
            except Exception as exc:
                status, reason = _vendor_failure(exc)
                code, after = vendor_error_facts(exc)
                yield _bare_terminal(
                    status, reason, self.runtime, status_code=code, retry_after=after
                )
                return

            yield UsageEvent(usage=token_usage(usage), scope=UsageScope.DELTA)

            if text:
                final_text = text

            # A run that declared no tools has nothing to answer with, and
            # replying "not declared" round after round would be a loop with a
            # bill on it. The model should never ask -- it was sent no tools --
            # so if it does, the run ends and terminal_status() says why.
            if not (request.tools and calls):
                yield from self._finish(
                    request,
                    final_text,
                    finish_reason=finish_reason,
                    unanswered_call=bool(calls),
                    blocked=blocked,
                    answered=answered,
                )
                return

            # The whole model turn goes back verbatim -- thought signatures
            # included -- before the parts that answer it.
            contents.extend(turn)
            answers: list[dict[str, Any]] = []
            for call in calls:
                name = getattr(call, "name", "") or ""
                arguments = call_arguments(call)
                # FunctionCall.id is optional here and usually absent, so the
                # name is the fallback: an event with an empty id cannot be
                # paired with its result by a consumer reading the stream.
                call_id = getattr(call, "id", None) or name
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
                answers.append(
                    function_response_part(call, name, content, is_error=is_error)
                )
            # One user turn carrying every answer, which is the shape this API
            # documents for parallel calls -- unlike the Responses API, where
            # each result is its own item.
            contents.append({"role": "user", "parts": answers})

    def arun(self, request: RunRequest) -> AsyncIterator[AgentEvent]:
        """:meth:`run`, on the caller's loop, with the vendor's async surface (R1).

        Native rather than the base class's worker thread, and the visible
        difference is the tool loop: a coroutine handler is awaited here, on the
        loop that called ``achat``. :meth:`run` is unchanged.
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
        return native_run(self, self._aloop(client, request))

    async def _aloop(
        self, client: Any, request: RunRequest
    ) -> AsyncIterator[AgentEvent]:
        """:meth:`_loop` with ``await`` where it blocks. Same rounds, same events.

        The one shape difference from the sync twin is the vendor's:
        ``generate_content_stream`` is a coroutine *returning* an async iterator
        here, where the sync one returns an iterator directly.
        """
        contents = content_items(request)
        by_name = {tool.name: tool for tool in request.tools}
        final_text = ""

        while True:
            if self._cancelled.is_set():
                yield _bare_terminal(
                    TerminalStatus.CANCELLED, "cancelled by caller", self.runtime
                )
                return

            params = self.request_params(request, contents)
            usage: Any = None
            finish_reason: Any = None
            blocked: str | None = None
            turn: list[dict[str, Any]] = []
            calls: list[Any] = []
            text = ""
            answered = False
            try:
                stream = await client.models.generate_content_stream(**params)
                try:
                    async for chunk in stream:
                        usage = getattr(chunk, "usage_metadata", None) or usage
                        blocked = blocked or _blocked_reason(chunk)
                        for candidate in candidates(chunk):
                            answered = True
                            finish_reason = (
                                getattr(candidate, "finish_reason", None) or finish_reason
                            )
                            content = getattr(candidate, "content", None)
                            if content is not None and getattr(content, "parts", None):
                                turn.append(_dump(content))
                            for part in parts_of(candidate):
                                part_text = getattr(part, "text", None)
                                if (
                                    isinstance(part_text, str)
                                    and part_text
                                    and not getattr(part, "thought", False)
                                ):
                                    text += part_text
                                call = getattr(part, "function_call", None)
                                if call is not None:
                                    calls.append(call)
                                for out in part_events(part, self.runtime):
                                    yield out
                finally:
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
                status, reason = _vendor_failure(exc)
                code, after = vendor_error_facts(exc)
                yield _bare_terminal(
                    status, reason, self.runtime, status_code=code, retry_after=after
                )
                return

            yield UsageEvent(usage=token_usage(usage), scope=UsageScope.DELTA)

            if text:
                final_text = text

            if not (request.tools and calls):
                for out in self._finish(
                    request,
                    final_text,
                    finish_reason=finish_reason,
                    unanswered_call=bool(calls),
                    blocked=blocked,
                    answered=answered,
                ):
                    yield out
                return

            contents.extend(turn)
            answers: list[dict[str, Any]] = []
            for call in calls:
                name = getattr(call, "name", "") or ""
                arguments = call_arguments(call)
                call_id = getattr(call, "id", None) or name
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
                answers.append(
                    function_response_part(call, name, content, is_error=is_error)
                )
            contents.append({"role": "user", "parts": answers})

    def _finish(
        self,
        request: RunRequest,
        final_text: str,
        *,
        finish_reason: Any,
        unanswered_call: bool,
        blocked: str | None,
        answered: bool,
    ) -> Iterator[AgentEvent]:
        """The structured answer, if one was asked for, and then the terminal."""
        status, reason = terminal_status(
            finish_reason,
            unanswered_call=unanswered_call,
            blocked=blocked,
            answered=answered or bool(final_text),
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

    def cancel(self) -> None:
        """Stop the loop at the next round boundary (D10).

        Two halves, and only one of them is this method. The bridge closes the
        iterator, which runs this generator's ``finally`` and closes the vendor
        stream -- the transport-level floor the contract documents. This flag is
        the graceful half: a loop that is between rounds stops there and reports
        ``cancelled`` rather than spending another round first.
        """
        self._cancelled.set()

    # --- sessions: refused, and the refusal says what to use instead ------------

    def open_session(self, request: SessionRequest) -> SessionHandle:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.new_chat()"))

    def resume_session(self, request: SessionRequest) -> SessionHandle:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.resume_chat()"))

    def list_sessions(self, request: SessionRequest) -> tuple[SessionInfo, ...]:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.list_sessions()"))


#: The one sentence all three session refusals share -- the same shape the other
#: three API adapters use, with this runtime's name in it (ticket 1.8's bridge
#: policy refuses every session door before an adapter is loaded; these stay
#: because an adapter that answered a session call would be a contract
#: violation).
_NO_SESSIONS = (
    f"the adapter for runtime {Runtime.GOOGLE_API.value!r} has not implemented "
    "{call}, and will not: a session rests on the runtime holding the "
    "conversation (D14) and an API endpoint holds none -- every request carries "
    "its whole history. Use bridge.chat(history=[...]), which is the documented "
    "multi-turn shape against a stateless runtime. Stateless bridge.chat() calls "
    "are unaffected"
)


def _blocked_reason(chunk: Any) -> str | None:
    """The prompt-feedback block reason on a chunk, as a string, or ``None``.

    A blocked prompt is the one failure on this API that arrives as a *successful*
    response with no candidates in it, which is why it is read separately from
    the finish reason and reported ahead of it.
    """
    feedback = getattr(chunk, "prompt_feedback", None)
    if feedback is None:
        return None
    return _reason_value(getattr(feedback, "block_reason", None))


def _vendor_failure(exc: Exception) -> tuple[TerminalStatus, str]:
    """Classify a vendor exception into a terminal status and a reason.

    A 429 is :attr:`TerminalStatus.QUOTA_EXHAUSTED` and not an error, because it
    is the outcome a quota failover exists to act on -- typed off the status code
    the SDK carries rather than matched on the message, which is the substring
    matching that burned a real allowance once already (R6).

    **The attribute is the one difference from the other three adapters.**
    ``anthropic`` and ``openai`` both put the HTTP status on ``status_code``;
    ``google.genai.errors.APIError`` spells it ``code`` (``google-genai``
    1.73.1, read 2026-09-13), with ``status`` holding the RPC name
    (``RESOURCE_EXHAUSTED``) rather than a number. Both attribute names are read,
    in that order, so this function is correct for this vendor without becoming
    wrong for a client that follows the other convention.
    """
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = getattr(exc, "code", None)
    if status_code == 429:
        return (
            TerminalStatus.QUOTA_EXHAUSTED,
            f"the endpoint reported HTTP 429 (rate limit / quota): {exc}",
        )
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return TerminalStatus.ERROR, f"HTTP {status_code}: {exc}"
    return TerminalStatus.ERROR, f"{type(exc).__name__}: {exc}"


def _default_client(*, api_key: str, base_url: str | None) -> Any:
    """``google.genai.Client``, constructed with an explicit key and nothing ambient.

    ``base_url`` travels in ``http_options`` -- this SDK has no ``base_url``
    argument of its own -- and is passed only when the connection carries one, so
    a connection without one gets the SDK's own default endpoint rather than a
    ``None`` the constructor has to interpret. A dict rather than a
    ``types.HttpOptions``: the SDK accepts ``HttpOptionsDict`` and this way the
    only vendor symbol this module touches is the client itself.
    """
    try:
        sdk = importlib.import_module(_MODULE)
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise RuntimeNotAvailable(Runtime.GOOGLE_API.value, _EXTRA, _PACKAGE) from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["http_options"] = {"base_url": base_url}
    return sdk.Client(**kwargs)


def _default_async_client(*, api_key: str, base_url: str | None) -> Any:
    """``google.genai.Client(...).aio``, under exactly :func:`_default_client`'s rules.

    ``.aio`` rather than a second class, because that is where this SDK keeps its
    async surface. Returning it directly is what lets :meth:`GoogleAPIAdapter._aloop`
    reach ``models.generate_content_stream`` at the same attribute path the sync
    loop uses.
    """
    try:
        sdk = importlib.import_module(_MODULE)
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise RuntimeNotAvailable(Runtime.GOOGLE_API.value, _EXTRA, _PACKAGE) from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["http_options"] = {"base_url": base_url}
    return sdk.Client(**kwargs).aio
