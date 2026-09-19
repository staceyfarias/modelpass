"""A LangChain ``BaseChatModel`` backed by a modelpass connection.

**What this is, and what it is not.** This module is a *convenience adapter*,
for applications whose internal LLM seam is already LangChain's
``BaseChatModel`` and which want a subscription-backed model to drop into it
without rewriting that seam. It covers only the chat-shaped subset of modelpass:
stateless chat, and schema-bound (structured) output. The modelpass API itself --
``Bridge.chat`` and the normalized event stream -- is the real surface, and it
is the one to write against for anything new. modelpass capabilities that do not
fit inside ``BaseChatModel`` (runtime-executed tools with their call/result
pairs, MCP servers, subagents, sessions, cancellation, the run log) are not
routed through here and never will be: bending them into a chat-model interface
would mean inventing shapes LangChain does not have. **modelpass is not tied to
LangChain** -- core keeps its zero runtime dependencies (D8), and this module is
one optional leaf over the same public bridge any other framework adapter would
use.

``langchain-core`` arrives through the optional ``modelpass[langchain]`` extra.
Importing this module without it raises an :class:`ImportError` naming the
extra; nothing in ``modelpass`` core imports this module.

What modelpass v1 does not do, and why the caller has to care
----------------------------------------------------------

* **No sampling controls.** ``temperature`` / ``top_p`` do not exist -- the
  agent runtimes do not expose them. A caller that passes a temperature gets a
  loud warning rather than a silent drop, because an application that sets
  ``temperature=0`` is usually doing it to make an extraction deterministic, and
  that expectation is now wrong.
* **No output-length control.** ``max_tokens`` is the same story. The nearest
  real thing is ``stop_at_tokens``, which is a *guard* -- it stops the run, it
  does not shape the answer.
* **Stop sequences are ignored.** The runtimes do not accept them, and they are
  not emulated here. A caller that depended on one gets a longer answer either
  way; pretending otherwise would only hide that a layer further down.
* **Structured output is native, and the two runtimes disagree about the
  schema.** ``anthropic-sdk`` accepts the loose form a Python dataclass or
  pydantic model converts to; ``openai-sdk`` requires the Responses API's
  *strict* subset (``additionalProperties: false``, every property in
  ``required``) and rejects anything else before generating a token. So
  :meth:`ChatSubpass.bind_tools` converts the schema once and, on an
  ``openai-sdk`` connection only, runs it through
  :func:`modelpass.to_openai_strict` -- which makes optional fields
  nullable-but-required, and is why the answer comes back needing its ``None``
  values stripped (:func:`_strip_nulls`) before a constructor is handed them.
* **The schema name is wire protocol, not a label (D13).** On ``anthropic-sdk``
  the runtime's structured-output mechanism *is* a tool call, so the name the
  caller gives travels to the model and prompts can refer to it. modelpass never
  invents one, and neither does this adapter.
* **Exactly one schema per call.** modelpass refuses ``schema=`` together with
  ``tools=`` on both runtimes, and there is no caller-executed tool loop to
  choose between several. A list of more than one is a caller mistake.
* **Three roles only.** modelpass v1 messages are system / user / assistant.
  ``ToolMessage`` and the function-call message shapes have no subscription
  equivalent and are refused rather than flattened into a user turn, where they
  would read as the user having said them.
* **No prompt-cache markers.** ``cache_control`` blocks mean nothing here. The
  runtime does its own caching; it is not the same knob and it is not reported
  the same way.
* **A vendor failure is a terminal event, not an exception.** modelpass ends such
  a run normally, carrying ``status="error"``, the receipt, and what was spent
  before the failure. LangChain callers -- retry loops especially -- expect a
  raise, so this module performs that conversion and carries the run's context
  into the exception rather than throwing it away.
* **The runtime wraps the prompt in its own harness system prompt.** Output is
  not identical to the same model over its API. Evaluate before switching
  anything that matters.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, ClassVar

from .connections import QuotaAction
from .errors import SubpassError
from .runtimes import Runtime
from .schema import to_openai_strict
from .types import (
    FailoverEvent,
    GuardWarningEvent,
    ReceiptEvent,
    Retryable,
    Sampling,
    StructuredOutputEvent,
    TerminalEvent,
    TerminalStatus,
    TextBlock,
    TextDeltaEvent,
    TokenUsage,
    UsageEvent,
)

#: langchain-core is the one dependency this module has, and it is optional.
#: The guard is at import time rather than lazy because the class statement
#: below cannot even be *defined* without ``BaseChatModel``. modelpass's own
#: imports are above it so that a missing extra fails on the langchain line,
#: naming the extra, rather than somewhere less obvious.
_LANGCHAIN_HINT = (
    "the modelpass LangChain adapter needs langchain-core, which is not part of "
    "modelpass core (core has zero runtime dependencies on purpose -- D8). "
    "Install it with 'pip install modelpass[langchain]'"
)

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        SystemMessage,
    )
    from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ImportError as exc:  # pragma: no cover - exercised by the extra's absence
    raise ImportError(f"{_LANGCHAIN_HINT} ({exc})") from exc

__all__ = [
    "ChatSubpass",
    "SubpassGuardStopError",
    "SubpassQuotaExhaustedError",
    "SubpassRunError",
    "SubpassTimeoutError",
    "to_json_schema",
]

logger = logging.getLogger(__name__)


class SubpassRunError(RuntimeError):
    """A modelpass run ended badly.

    A vendor failure in modelpass is a *terminal event with an error status*, not
    an exception -- the stream still completes, still carries the receipt, and
    still reports what was spent before the failure. LangChain callers expect a
    raise, so this is where that conversion happens, and the run's context comes
    with it rather than being thrown away.

    Attributes:
        reason: What the runtime said went wrong.
        connection: The modelpass connection name the run used.
        receipt_summary: The one-line preflight receipt for that connection --
            which account, which auth mode, which model. Included because a
            rejected model is only diagnosable from what the receipt named.
        partial_text: Assistant text produced before the failure, if any.
        usage: The token usage modelpass reported up to the failure, in the
            ``input_tokens`` / ``output_tokens`` / ``cached_input_tokens`` shape.
        retryable: modelpass's verdict for this run, read off the terminal event
            rather than re-derived from ``reason`` (R6, ticket 1.12b).
        retry_after: The seconds the vendor asked for, where one said so.
        status_code: The HTTP status the vendor reported, where there was one.
        terminal: The :class:`~modelpass.types.TerminalEvent` this was raised
            for, so nothing it knew is lost at the raise.

    The four verdict attributes are *additions*: nothing here was renamed, and
    the two consumers that match these classes by name across the MRO -- a
    retrieval service and a desktop agent app -- keep every spelling they
    already match on.
    """

    def __init__(
        self,
        reason: str,
        *,
        connection: str = "",
        receipt_summary: str = "",
        partial_text: str = "",
        usage: dict | None = None,
        retryable: Retryable = Retryable.UNKNOWN,
        retry_after: float | None = None,
        status_code: int | None = None,
        terminal: TerminalEvent | None = None,
    ):
        self.reason = reason
        self.connection = connection
        self.receipt_summary = receipt_summary
        self.partial_text = partial_text
        self.usage = dict(usage or {})
        self.retryable = retryable
        self.retry_after = retry_after
        self.status_code = status_code
        self.terminal = terminal
        where = f" on modelpass connection '{connection}'" if connection else ""
        detail = f" [{receipt_summary}]" if receipt_summary else ""
        super().__init__(f"modelpass run failed{where}: {reason}{detail}")


class SubpassGuardStopError(SubpassRunError):
    """A configured token guard stopped the run before it finished."""


class SubpassTimeoutError(SubpassRunError):
    """The wall-clock bound on this call ran out (R6, ticket 1.12).

    The LangChain-side face of :class:`~modelpass.errors.RunTimedOut`, and a
    distinct class for the same reason :class:`SubpassQuotaExhaustedError` is
    one: a retry loop wrapped around a chat model has to tell a bound that
    expired from an answer that was refused, and the two consumers who match
    these classes do it **by name across the MRO** so modelpass stays an optional
    import (a retrieval service and a desktop agent app, both read 2026-09-13).
    A timeout arriving as a bare :class:`SubpassRunError` would be indexed with
    every other failure and retried or not on somebody else's rule.

    ``retryable`` is the library's verdict for this run, carried across so a
    loop reads a typed value rather than re-deriving one from ``reason``. It is
    :attr:`~modelpass.types.Retryable.YES` only where nothing answered at all
    before the ``first_token`` bound on a stateless run; a ``total`` expiry is
    ``UNKNOWN``, because the run may have been most of the way through an
    answer. **modelpass still never retries** -- this is the verdict, and the
    loop is the caller's.
    """

    def __init__(
        self,
        reason: str,
        *,
        retryable: Retryable = Retryable.UNKNOWN,
        **kwargs: object,
    ) -> None:
        # Kept as an explicit keyword because it is the signature this class
        # shipped with in 1.12; the base now holds it, along with the rest of
        # the verdict (1.12b), so it is forwarded rather than set twice.
        super().__init__(reason, retryable=retryable, **kwargs)  # type: ignore[arg-type]


class SubpassQuotaExhaustedError(SubpassRunError):
    """The subscription allowance behind this connection ran out.

    Kept distinct from :class:`SubpassRunError` deliberately. It is the one
    outcome a user can act on directly -- wait for the allowance to reset, or
    configure a failover connection -- and neither vendor exposes a *pre-run*
    "remaining quota" API, so hitting this is the only way an app finds out.
    """


def _flatten_content(content: Any) -> str:
    """Reduce LangChain message content to plain text.

    The fallback, and no longer the normal path: :func:`_convert_content` keeps
    a block list as blocks so a ``cache_control`` marker survives (R3). This is
    what is left for content shapes modelpass has no block for -- an image part,
    a tool-use part -- where joining the text is the most that can honestly be
    carried.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n\n".join(part for part in parts if part)
    return str(content)


def _convert_content(content: Any) -> str | tuple[TextBlock, ...]:
    """LangChain message content as modelpass content (R3).

    **This is the path a downstream agent host and a retrieval service are on.**
    Both build their prompts as
    LangChain block lists -- ``[{"type": "text", "text": ..., "cache_control":
    {"type": "ephemeral"}}, ...]`` -- because that is how prompt caching is
    expressed on ``langchain-anthropic``, and until now this adapter joined
    those blocks into one string and dropped the markers at the door. The
    consumer got no error, no warning, and no caching.

    A plain string stays a string: unchanged bytes, unchanged rendering. A block
    list becomes :class:`~modelpass.types.TextBlock` objects **only when every
    part is text**; anything else -- an image part, a tool-use part, a shape
    modelpass has no vocabulary for -- falls back to :func:`_flatten_content`,
    because a partial conversion would silently lose the part it could not carry
    while looking like it had carried everything.

    Whether the markers then reach the vendor is a runtime question, and the
    receipt answers it. What ends here is the *silent* strip.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or not content:
        return _flatten_content(content)
    blocks: list[TextBlock] = []
    for part in content:
        if isinstance(part, str):
            blocks.append(TextBlock(part))
            continue
        if not isinstance(part, dict) or part.get("type") != "text":
            return _flatten_content(content)
        try:
            blocks.append(TextBlock.coerce(part))
        except (TypeError, ValueError):
            # A marker modelpass cannot read -- an unknown ttl, a type it does
            # not know. Reported rather than raised, and the text is still
            # carried: this is a message conversion, and the honest degradation
            # is the one this function already has for a block shape it cannot
            # represent.
            logger.warning(
                "modelpass could not read a cache_control marker (%r) on a "
                "LangChain text block; the text is carried without it",
                part.get("cache_control"),
            )
            blocks.append(TextBlock(str(part.get("text", ""))))
    return tuple(blocks)


#: The bound-kwarg names a schema travels under between :meth:`bind_tools` and
#: :meth:`ChatSubpass._generate`. Namespaced because they ride LangChain's
#: generic ``**kwargs`` all the way through ``generate_prompt``.
_SCHEMA_KWARG = "modelpass_schema"
_SCHEMA_NAME_KWARG = "modelpass_schema_name"

#: The sampling kwargs ``bind(...)`` may carry, LangChain's name -> modelpass's.
#: **Not** namespaced like the schema kwargs above, on purpose: these are the
#: names every other LangChain chat model already binds, and a caller swapping
#: ``ChatOpenAI(temperature=0)`` for this class should not have to rename the
#: thing it binds. ``max_tokens`` is the only one that differs, and it differs
#: in the direction of LangChain rather than modelpass for the same reason.
_BOUND_SAMPLING_KWARGS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "max_output_tokens",
    "reasoning_effort": "reasoning_effort",
}


def to_json_schema(tool: Any) -> dict:
    """The JSON Schema for one structured-output schema class.

    Accepts whatever LangChain's own converter accepts -- a ``@dataclass``, a
    pydantic model, a ``TypedDict``, a function, or a schema dict. Using
    LangChain's converter is deliberate rather than incidental: it is the same
    function ``bind_tools`` uses on every other provider, so what a subscription
    connection is asked for matches what an API-key connection is asked for,
    rather than being a second, subtly different rendering of the same class.
    """
    function = convert_to_openai_tool(tool)["function"]
    schema = dict(function.get("parameters") or {"type": "object", "properties": {}})
    # ``title`` is what modelpass falls back to when no ``schema_name`` is given.
    # A caller should still pass one (the name is wire protocol -- D13), but a
    # schema that names itself is easier to read in a captured run log.
    schema.setdefault("title", function.get("name", ""))
    description = function.get("description")
    if description and "description" not in schema:
        schema["description"] = description
    return schema


def _strip_nulls(value: Any) -> Any:
    """Drop ``None`` values, recursively, from a strict-schema answer.

    :func:`modelpass.to_openai_strict` expresses "optional" as *required and
    nullable*, because OpenAI's strict subset has no notion of an absent key. So
    an ``openai-sdk`` run reports every field it had nothing to say about as
    ``null``, and handing that to a dataclass or model constructor overwrites
    the class's own default with ``None`` -- a ``list`` field becomes ``None``
    and the next thing to iterate it raises, three layers from here.

    Removing the keys restores exactly the shape the loose schema produces, so
    both runtimes hand the same dict to the same constructor.
    """
    if isinstance(value, dict):
        return {k: _strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_nulls(item) for item in value]
    return value


class ChatSubpass(BaseChatModel):
    """A ``BaseChatModel`` that runs on a subscription instead of an API key.

    Stateless per call, like modelpass itself: the whole message list is
    serialized into one ephemeral runtime session every time. There is no
    session id to hold and no server-side history, so a caller that re-sends its
    own conversation maps onto it exactly -- and a caller that sends only the
    newest turn loses the conversation entirely.

    Args:
        connection: The modelpass connection name, from the user's
            ``~/.modelpass/connections.toml``. Only the name is needed; the
            connection binds runtime, auth mode and credential together (D3).
        model: Model id to request, or ``None`` to let the connection (then the
            runtime) decide. The receipt reports which.
        stop_at_tokens: Per-call token ceiling. ``None`` uses whatever the
            connection itself configures, which may well be nothing.
        allow_failover: Whether a connection configured with
            ``onQuotaExhausted.failover`` may actually move this call onto the
            failover connection -- typically a metered API key. Default
            ``False``: the opt-in belongs to the application as well as to the
            modelpass config, and a billing-mode change should never be something
            an app inherited without saying so.
        temperature: Sampling temperature, passed through to
            ``Bridge.chat(sampling=...)``. **Whether it arrives depends on the
            connection's runtime and model** -- see the sampling note below.
        top_p / top_k: The other two sampling controls, same treatment.
        max_tokens: A ceiling on the answer. LangChain's own spelling for what
            modelpass calls ``max_output_tokens``; the name is kept because every
            other chat model in a LangChain app uses it.
        reasoning_effort: ``"low"``, ``"medium"`` or ``"high"``, or ``None``.
        bridge: An injected ``modelpass.Bridge``, for tests. ``None`` builds one
            over the user's real store.

    **A system message plus one human message already takes the single-shot
    path.** modelpass renders that pair as instructions above a task on both
    runtimes, with no transcript envelope, so a scoring or classification loop
    built on this adapter is already on the clean path. What it cannot give you
    is the signature: ``ChatSubpass.invoke(messages)`` is LangChain's own
    interface and takes a flat list, while ``Bridge.chat`` asks for the new turn
    as ``message=``, the instructions as ``system_prompt=`` and prior turns as
    ``history=`` -- which is what makes plain that the history is yours to hold.
    Reach for ``bridge.chat`` directly when you want that, and the caching
    guidance in its docstring; stay here when you want a LangChain chat model.

    **Sampling: passed through, and reported rather than promised** (R5, ticket
    1.7). Until 1.7 this class took ``temperature`` and ``max_tokens`` only to
    warn about them and then drop them, because the two agent runtimes honour
    neither -- and that warning did real damage: a desktop agent app stopped
    handing this class a temperature at all rather than route a value into a
    hole, so a user who set one on a subscription configuration silently got
    nothing.

    Both halves of that are repaired now. The values are handed to
    ``Bridge.chat(sampling=...)``, where an API runtime applies them and an agent
    runtime does not; and either way the answer's ``response_metadata`` carries
    ``subpass_sampling_applied`` -- what was actually sent, in modelpass's field
    names -- and ``subpass_sampling_notes``, one sentence per field that was
    dropped or coerced. The constructor no longer warns, because a warning is
    what you emit when you have nowhere to put the truth, and there is now
    somewhere.

    Per-call overrides go through LangChain's own ``bind``:
    ``model.bind(temperature=0)`` for one call, leaving the instance alone.
"""

    connection: str
    model: str | None = None
    stop_at_tokens: int | None = None
    allow_failover: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    bridge: Any = None

    # ``bridge`` is an injected modelpass object, not a pydantic-modelable type,
    # and ``model`` would otherwise collide with pydantic's ``model_`` namespace.
    model_config: ClassVar[dict] = {
        "arbitrary_types_allowed": True,
        "protected_namespaces": (),
    }

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._log_sampling_stance()

    @property
    def _llm_type(self) -> str:
        return "modelpass"

    @property
    def _identifying_params(self) -> dict:
        return {
            "connection": self.connection,
            "model": self.model,
            "stop_at_tokens": self.stop_at_tokens,
            "allow_failover": self.allow_failover,
        }

    def _log_sampling_stance(self) -> None:
        """Note the configured sampling at debug level. **Never a warning** (R5).

        This used to warn, loudly, on every construction that named a
        temperature or a max_tokens, because modelpass honoured neither and a
        silent drop is worse than a noisy one. Ticket 1.7 removed the premise
        underneath it in both directions: the values are forwarded now, and
        where they are still dropped the receipt names each one on the call that
        dropped it (``sampling_notes``, reaching a LangChain caller as
        ``response_metadata["subpass_sampling_notes"]``).

        So the warning had become the wrong shape twice over. It fired at
        *construction*, before a connection was resolved and therefore before
        anything could know whether this runtime honours the value -- it would
        now shout at an ``anthropic-api`` connection that honours all of it. And
        it cost more than noise: a desktop agent app routes around this class
        rather than be shouted at, which is a consumer opting out of the honesty
        instead of receiving it.

        Debug rather than nothing: "which knobs did this model think it had" is
        a real question while reading a log, and it is the one thing here that
        is knowable at construction time.
        """
        configured = {
            name: value
            for name, value in self._sampling_fields().items()
            if value is not None
        }
        if not configured:
            return
        logger.debug(
            "modelpass connection '%s' configured with %s; what is actually sent "
            "is per runtime and per model, and rides every answer's "
            "response_metadata as subpass_sampling_applied",
            self.connection,
            ", ".join(f"{name}={value!r}" for name, value in configured.items()),
        )

    def _sampling_fields(self) -> dict:
        """This instance's sampling settings under **modelpass's** field names.

        ``max_tokens`` becomes ``max_output_tokens`` here and nowhere else: the
        LangChain name stays on the constructor because every other chat model
        in an app uses it, and the modelpass name is what travels, because that
        is what the receipt reports back. One rename, in one place.
        """
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }

    def _sampling_for(self, overrides: Mapping[str, Any]) -> Sampling | None:
        """The instance's sampling with one call's ``bind(...)`` kwargs over it.

        ``None`` when neither says anything, so a caller who has never heard of
        sampling produces a request with no sampling on it and a receipt that
        says only what the runtime itself required.
        """
        merged = {**self._sampling_fields(), **dict(overrides)}
        resolved = {name: value for name, value in merged.items() if value is not None}
        return Sampling(**resolved) if resolved else None

    @staticmethod
    def _bound_sampling(kwargs: dict) -> dict:
        """Take this call's sampling overrides out of LangChain's ``**kwargs``.

        ``model.bind(temperature=0)`` is how a LangChain caller overrides one
        setting for one call, and the kwargs arrive here on ``_generate`` /
        ``_stream``. They are *popped* rather than read, the way the bound
        schema is: forwarding a sampling kwarg onward as well would be a second
        channel for a field that already has one.
        """
        overrides = {}
        for langchain_name, modelpass_name in _BOUND_SAMPLING_KWARGS.items():
            if langchain_name in kwargs:
                overrides[modelpass_name] = kwargs.pop(langchain_name)
        return overrides

    # --- modelpass plumbing ------------------------------------------------------

    def _get_bridge(self):
        """The bridge to run on, built over the user's real store if not injected."""
        if self.bridge is not None:
            return self.bridge
        from .bridge import Bridge

        return Bridge()

    def _connection_record(self, bridge):
        """The stored ``modelpass.Connection``, or ``None`` if it will not resolve.

        Resolving it is the bridge's job and it raises a better error a moment
        from now, so a failure here is swallowed rather than pre-empted. What
        this is for is the two questions worth answering *before* the call:
        which runtime is about to be talked to, and whether a configured
        failover is about to be declined.
        """
        try:
            return bridge.connection(self.connection)
        except SubpassError:
            return None

    def _announce_failover_stance(self, connection) -> None:
        """Say out loud what ``allow_failover`` is about to do to this call.

        The mechanism itself is one keyword to ``Bridge.chat`` -- modelpass's
        ``allow_failover=`` exists for exactly this decision, and there is
        deliberately no spelling of it that could *create* a failover. What the
        keyword cannot do is tell the user that the failover they configured on
        the connection is being declined, which is worth a line in the log:
        otherwise "the run stopped when the allowance ran out" looks like the
        failover failing rather than the caller declining it.
        """
        if connection is None:
            return
        policy = connection.guards.on_quota_exhausted
        if policy.action is not QuotaAction.FAILOVER:
            return
        if self.allow_failover:
            logger.info(
                "modelpass connection '%s' may fail over to '%s' when its "
                "allowance is exhausted; this model has opted in.",
                self.connection,
                policy.failover,
            )
        else:
            logger.warning(
                "modelpass connection '%s' is configured to fail over to '%s' "
                "when its allowance runs out, but this model has not opted in "
                "-- the run will stop cleanly instead. Set allow_failover=True "
                "on the ChatSubpass to permit it.",
                self.connection,
                policy.failover,
            )

    def _schema_for(self, connection, schema: dict, state: dict) -> dict:
        """The schema as *this connection's runtime* will accept it.

        The one place the two runtimes genuinely disagree. ``anthropic-sdk``
        takes the loose form a dataclass converts to -- optional properties, no
        ``additionalProperties`` -- and ``openai-sdk`` requires the Responses
        API's strict subset and 400s on anything else, before generating
        anything.

        The decision is made from the **connection's runtime**, never from
        anything the calling application believes about the provider: which
        schema dialect goes on the wire is a fact about the store.

        ``state`` records that the conversion happened, because the caller has
        to undo half of it: strict mode has no absent keys, so every optional
        field comes back explicitly null. See :func:`_strip_nulls`.
        """
        runtime = getattr(connection, "runtime", None)
        if runtime is not Runtime.OPENAI_SDK:
            return schema
        state["strict_schema"] = True
        return to_openai_strict(schema)

    def _to_modelpass_messages(self, messages: Sequence[BaseMessage]) -> list[dict]:
        """Map LangChain messages onto modelpass role dicts.

        System / Human / AI are the only three roles modelpass v1 has. Anything
        else (tool messages, and the function-call shapes) has no subscription
        equivalent and is refused rather than flattened into a user turn where
        it would read as the user having said it.
        """
        out: list[dict] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif message.type == "human":
                role = "user"
            else:
                raise ValueError(
                    f"ChatSubpass cannot map a {message.type!r} message: modelpass v1 "
                    "carries system, user and assistant turns only."
                )
            out.append({"role": role, "content": _convert_content(message.content)})
        return out

    def _split_for_chat(
        self, messages: Sequence[BaseMessage]
    ) -> tuple[str | tuple[TextBlock, ...] | None, list[dict], str | tuple[TextBlock, ...]]:
        """Split a LangChain list into ``(system_prompt, history, message)``.

        ``Bridge.chat`` takes the new user turn on its own and everything before
        it as ``history`` (D21), while a ``BaseChatModel`` is handed one flat
        list per call. So the split is purely positional: the last turn is the
        one being taken now, and every earlier turn rides ``history`` in the
        order it arrived.

        **A leading system message is hoisted into ``system_prompt=`` only when
        it carries content blocks** (R3), and the narrowness is the decision. A
        string system message stays in ``history`` exactly as it always has:
        that argument is assembled ahead of ``history``, so hoisting from a list
        where the system message might not have been first would reorder the
        prompt, and a reordered prompt is a moved cache prefix. Nothing about an
        existing caller's bytes moves.

        A *block* system message is a different case, and it is the one the
        agent host and the retrieval service above are in: blocks exist to mark
        where the cacheable prefix ends,
        and `Bridge.chat` puts ``system_prompt`` at the front of the prefix,
        which is the only position where that marking means anything. Hoisting
        the leading one puts it exactly where the caller's own breakpoints say
        it belongs. A block system message that is *not* first is left in
        ``history`` untouched, for the reordering reason above.

        A trailing non-user turn is refused rather than relabelled: ``message``
        *is* the user's turn, and quietly filing an assistant message under it
        would put words in the user's mouth -- the same reason
        :meth:`_to_modelpass_messages` refuses a tool message.
        """
        converted = self._to_modelpass_messages(messages)
        system_prompt: str | tuple[TextBlock, ...] | None = None
        if (
            len(converted) > 1
            and converted[0]["role"] == "system"
            and isinstance(converted[0]["content"], tuple)
        ):
            system_prompt = converted[0]["content"]
            converted = converted[1:]
        if not converted:
            raise ValueError(
                f"ChatSubpass (modelpass connection '{self.connection}') was called "
                "with no messages; there is no turn to send."
            )
        last = converted[-1]
        if last["role"] != "user":
            raise ValueError(
                f"ChatSubpass (modelpass connection '{self.connection}') needs the "
                f"last message to be a human turn, not {last['role']!r}: modelpass's "
                "stateless call takes the new user turn as message= and everything "
                "before it as history=, and there is no way to ask a runtime to "
                "continue its own half-finished answer."
            )
        return system_prompt, converted[:-1], last["content"]

    def _run(
        self,
        messages: Sequence[BaseMessage],
        state: dict,
        *,
        schema: dict | None = None,
        schema_name: str = "",
        sampling: Mapping[str, Any] | None = None,
    ) -> Iterator[str]:
        """Drive one modelpass call, yielding text deltas and filling ``state``.

        The generator is shared by :meth:`_generate` and :meth:`_stream` so
        there is exactly one place that knows the event vocabulary. Everything
        the caller needs after the fact -- usage, receipt, terminal status, the
        schema-bound answer, whether the call crossed onto metered billing --
        lands in ``state``.
        """
        bridge, kwargs = self._call_kwargs(
            messages, state, schema=schema, schema_name=schema_name, sampling=sampling
        )
        try:
            events = bridge.chat(**kwargs)
        except SubpassError as exc:
            raise self._cannot_run(exc) from exc

        for event in events:
            delta = self._absorb(event, state, schema_name)
            if delta:
                yield delta

        self._require_terminal(state)

    async def _arun_events(
        self,
        messages: Sequence[BaseMessage],
        state: dict,
        *,
        schema: dict | None = None,
        schema_name: str = "",
        sampling: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """:meth:`_run`, on a loop (ticket 1.13).

        The same three steps -- assemble, fold, check -- because all three are
        shared methods; what differs is ``achat`` and ``async for``. The leaf
        used to decline async on purpose, and the reason it gave was true at the
        time: ``BaseChatModel`` runs the sync path in an executor and a
        hand-rolled async path would have been a second way to do the same
        thing. It stopped being true when modelpass grew a real async face --
        running it in an executor now means a thread wrapping a loop-native
        call, which is precisely the marshalling R1 exists to delete.
        """
        bridge, kwargs = self._call_kwargs(
            messages, state, schema=schema, schema_name=schema_name, sampling=sampling
        )
        try:
            events = bridge.achat(**kwargs)
        except SubpassError as exc:
            raise self._cannot_run(exc) from exc

        async for event in events:
            delta = self._absorb(event, state, schema_name)
            if delta:
                yield delta

        self._require_terminal(state)

    def _cannot_run(self, exc: SubpassError) -> ValueError:
        """A refusal raised before the stream exists, in LangChain's own currency.

        Everything raised before the iterator exists is configuration or a
        caller mistake -- modelpass guarantees a vendor failure arrives as a
        terminal event instead. LangChain's contract for "you called me wrong"
        is ``ValueError``, and callers of a chat model catch that rather than a
        modelpass type they may not import.
        """
        return ValueError(f"modelpass connection '{self.connection}' cannot run: {exc}")

    def _call_kwargs(
        self,
        messages: Sequence[BaseMessage],
        state: dict,
        *,
        schema: dict | None = None,
        schema_name: str = "",
        sampling: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """The bridge and the call, assembled once for both faces."""
        bridge = self._get_bridge()
        connection = self._connection_record(bridge)
        self._announce_failover_stance(connection)

        system_prompt, history, message = self._split_for_chat(messages)
        kwargs: dict[str, Any] = {
            "connection": self.connection,
            "message": message,
            # Declining a configured quota failover is one keyword, not guard
            # surgery: modelpass provides it for exactly this caller, and it
            # composes with the threshold shorthand below because both start
            # from the connection's own guards.
            "allow_failover": self.allow_failover,
        }
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        if history:
            kwargs["history"] = history
        if self.model:
            kwargs["model"] = self.model
        if self.stop_at_tokens is not None:
            kwargs["stop_at_tokens"] = self.stop_at_tokens
        # Distinct from ``max_tokens`` above it and deliberately adjacent to it:
        # ``stop_at_tokens`` is the guard over the whole run's spend and
        # ``max_output_tokens`` is a ceiling on one answer. Two settings, two
        # keywords, and a caller that wants both sets both (DESIGN R5).
        resolved_sampling = self._sampling_for(sampling or {})
        if resolved_sampling is not None:
            kwargs["sampling"] = resolved_sampling
        if schema is not None:
            kwargs["schema"] = self._schema_for(connection, schema, state)
            if schema_name:
                kwargs["schema_name"] = schema_name

        return bridge, kwargs

    def _absorb(self, event: Any, state: dict, schema_name: str) -> str:
        """Fold one modelpass event into ``state``; return the text to emit, if any.

        One place that knows the event vocabulary, shared by the sync and async
        readers above -- the leaf's own small echo of why
        :mod:`modelpass._fold` exists.
        """
        if isinstance(event, ReceiptEvent):
            receipt = event.receipt
            state["receipt_summary"] = event.summary
            state["auth_mode"] = str(event.auth_mode)
            state["runtime"] = str(event.runtime)
            state["account"] = receipt.account
            state["plan_name"] = receipt.plan_name
            state["resolved_model"] = receipt.model
            # The honesty half of R5, off the pre-run receipt: what was
            # actually sent, and one sentence for every field that was not.
            state["sampling_applied"] = dict(receipt.sampling_applied)
            state["sampling_notes"] = list(receipt.sampling_notes)
        elif isinstance(event, TextDeltaEvent):
            if event.text:
                state["text"] += event.text
                return event.text
        elif isinstance(event, UsageEvent):
            # ``cumulative`` is the bridge's own running total, already
            # correct for delta-vs-run_total reporting. Never accumulate the
            # per-event figures here -- that is how a run_total gets counted
            # twice.
            if event.cumulative is not None:
                state["usage"] = _usage_dict(event.cumulative)
        elif isinstance(event, StructuredOutputEvent):
            # Exactly one of these arrives, immediately before the terminal,
            # and only because ``schema=`` was passed. A run that produced
            # nothing parseable never gets here -- modelpass ends it with a
            # terminal error naming what did come back, rather than
            # substituting an empty object.
            state["structured"] = event
            if not event.valid:
                # modelpass's own structural check, not the vendor's. The
                # answer is delivered as-is either way; this is the line
                # that makes a schema drift visible instead of turning into
                # a quietly thinner result.
                logger.warning(
                    "modelpass connection '%s' returned a schema-bound answer "
                    "for '%s' that failed modelpass's structural check: %s. "
                    "The answer is being used as returned.",
                    self.connection,
                    event.schema_name or schema_name,
                    "; ".join(event.problems) or "no detail given",
                )
        elif isinstance(event, GuardWarningEvent):
            logger.warning("modelpass guard warning: %s", event.message)
            state["guard_warnings"].append(event.message)
        elif isinstance(event, FailoverEvent):
            # A billing-mode change is provenance worth recording: which
            # connection ran out, which one is finishing, and whether that
            # crossed from an allowance onto a metered bill.
            state["failed_over_from"] = event.from_connection
            state["crosses_to_metered"] = bool(event.crosses_to_metered)
            logger.warning(
                "modelpass failed over from '%s' to '%s'%s: %s",
                event.from_connection,
                event.to_connection,
                " (ONTO METERED BILLING)" if event.crosses_to_metered else "",
                event.reason,
            )
        elif isinstance(event, TerminalEvent):
            state["usage"] = _usage_dict(event.usage)
            state["status"] = str(event.status)
            state["reason"] = event.reason or ""
            if event.failed_over_from:
                state["failed_over_from"] = event.failed_over_from
            _raise_for_terminal(event, self.connection, state)

        return ""

    def _require_terminal(self, state: dict) -> None:
        if not state.get("status"):
            # modelpass guarantees exactly one terminal event on every stream that
            # completes, so its absence is a contract violation rather than a
            # run outcome. Say so plainly instead of returning whatever text
            # happened to arrive as though the run had finished.
            raise SubpassRunError(
                "the modelpass stream ended without a terminal event",
                connection=self.connection,
                receipt_summary=state.get("receipt_summary", ""),
                usage=state.get("usage"),
            )

    # --- LangChain surface -----------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Run one call and return the whole answer.

        ``stop`` sequences are not supported by the agent runtimes and are
        ignored rather than emulated; a caller that depended on one would get a
        silently longer answer either way, and pretending otherwise here would
        hide it a layer further down.

        When a schema is bound, the answer also arrives as a LangChain
        ``tool_call``. That shape is not decoration: it is what a caller reading
        ``result.tool_calls[0]["args"]`` already expects, so the same reader
        works whether the JSON was constrained by a forced tool choice over an
        API key or by the runtime's own schema mechanism over a subscription.
        """
        schema, schema_name = self._bound_schema(kwargs)
        sampling = self._bound_sampling(kwargs)
        state = _new_state()
        for _ in self._run(
            messages, state, schema=schema, schema_name=schema_name, sampling=sampling
        ):
            pass
        message = AIMessage(
            content=state["text"],
            tool_calls=self._tool_calls_from(state, schema, schema_name),
            usage_metadata=_usage_metadata(state["usage"]),
            response_metadata=_response_metadata(state),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _bound_schema(kwargs: dict) -> tuple[dict | None, str]:
        """Take the bound schema out of the per-call kwargs, if there is one."""
        return kwargs.pop(_SCHEMA_KWARG, None), kwargs.pop(_SCHEMA_NAME_KWARG, "") or ""

    def _structured_args(self, state: dict, schema_name: str) -> dict:
        """The schema-bound answer as the dict a caller builds its object from."""
        event = state.get("structured")
        if event is None:
            # modelpass emits exactly one structured_output event per schema-bound
            # run, and ends a run that produced nothing parseable with a
            # terminal error instead -- which _raise_for_terminal has already
            # turned into a raise. Reaching here means the contract broke, and
            # returning ``{}`` would hand the caller an empty result that looks
            # exactly like a real one that found nothing.
            raise SubpassRunError(
                f"the modelpass run completed without the schema-bound answer for "
                f"'{schema_name}' that was asked for",
                connection=self.connection,
                receipt_summary=state.get("receipt_summary", ""),
                partial_text=state.get("text", ""),
                usage=state.get("usage"),
            )
        data = event.data
        if state.get("strict_schema"):
            data = _strip_nulls(data)
        return data if isinstance(data, dict) else {"value": data}

    def _tool_calls_from(
        self, state: dict, schema: dict | None, schema_name: str
    ) -> list[dict]:
        if schema is None:
            return []
        return [
            {
                "name": schema_name,
                "args": self._structured_args(state, schema_name),
                "id": f"modelpass-{uuid.uuid4()}",
                "type": "tool_call",
            }
        ]

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the answer, with usage and the receipt on the final chunk.

        Usage arrives late by construction -- on one runtime it is reported per
        turn, on the other only when the exec run ends -- so it rides the last
        chunk, which is where LangChain's chunk aggregation expects it.
        """
        schema, schema_name = self._bound_schema(kwargs)
        sampling = self._bound_sampling(kwargs)
        state = _new_state()
        for delta in self._run(
            messages, state, schema=schema, schema_name=schema_name, sampling=sampling
        ):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk
        tool_call_chunks = []
        if schema is not None:
            # A schema-bound answer is not incremental -- modelpass emits it once,
            # complete, just before the terminal -- so it rides the same final
            # chunk usage does, as one whole chunk rather than a fake stream of
            # partial JSON.
            tool_call_chunks = [
                create_tool_call_chunk(
                    name=schema_name,
                    args=json.dumps(self._structured_args(state, schema_name)),
                    id=f"modelpass-{uuid.uuid4()}",
                    index=0,
                )
            ]
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=tool_call_chunks,
                usage_metadata=_usage_metadata(state["usage"]),
                response_metadata=_response_metadata(state),
            )
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """:meth:`_generate`, awaited, over :meth:`~modelpass.Bridge.achat`.

        Overridden as of ticket 1.13. ``BaseChatModel``'s default would run
        :meth:`_generate` in an executor, which was the right answer while
        modelpass was synchronous and is the wrong one now: it would put a
        thread around a call that is already loop-native, and on the API
        runtimes it would push a caller's coroutine tool handlers back off the
        loop they live on.

        Same answer as the sync twin, asserted by the drift test.
        """
        schema, schema_name = self._bound_schema(kwargs)
        sampling = self._bound_sampling(kwargs)
        state = _new_state()
        async for _ in self._arun_events(
            messages, state, schema=schema, schema_name=schema_name, sampling=sampling
        ):
            pass
        message = AIMessage(
            content=state["text"],
            tool_calls=self._tool_calls_from(state, schema, schema_name),
            usage_metadata=_usage_metadata(state["usage"]),
            response_metadata=_response_metadata(state),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """:meth:`_stream`, awaited. Same chunks, same late usage, same final chunk."""
        schema, schema_name = self._bound_schema(kwargs)
        sampling = self._bound_sampling(kwargs)
        state = _new_state()
        async for delta in self._arun_events(
            messages, state, schema=schema, schema_name=schema_name, sampling=sampling
        ):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                await run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk
        tool_call_chunks = []
        if schema is not None:
            tool_call_chunks = [
                create_tool_call_chunk(
                    name=schema_name,
                    args=json.dumps(self._structured_args(state, schema_name)),
                    id=f"modelpass-{uuid.uuid4()}",
                    index=0,
                )
            ]
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=tool_call_chunks,
                usage_metadata=_usage_metadata(state["usage"]),
                response_metadata=_response_metadata(state),
            )
        )

    def bind_tools(
        self,
        tools: Sequence,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        """Bind the one schema this call must answer in.

        These "tools" are not tools. Forced tool choice is how an API-key
        provider is made to return schema-shaped JSON, and the tool *is* the
        schema. On a subscription the runtime has its own mechanism for that, so
        the binding is translated rather than emulated -- the class becomes a
        JSON Schema and rides ``Bridge.chat(schema=..., schema_name=...)``.

        ``tool_choice`` is the schema name, and it is **wire protocol, not a
        label**. On ``anthropic-sdk`` the runtime's structured-output mechanism
        *is* a tool call, so modelpass lets the caller own the name and never
        invents one (D13); prompts that refer to the schema by name keep
        working only because it is passed through unchanged.

        Exactly one schema may be bound: modelpass refuses ``schema=`` together
        with ``tools=`` on both runtimes, and there is no caller-executed tool
        loop here to choose between several. A list of more than one would be a
        caller mistake worth seeing.
        """
        bound = list(tools)
        if len(bound) != 1:
            raise ValueError(
                f"ChatSubpass (modelpass connection '{self.connection}') binds "
                f"exactly one structured-output schema per call, not "
                f"{len(bound)}. modelpass has no caller-executed tool loop to "
                "choose between them."
            )
        schema = to_json_schema(bound[0])
        name = tool_choice or schema.get("title") or getattr(bound[0], "__name__", "")
        if not name:
            raise ValueError(
                "A schema-bound modelpass call needs a schema name: it is the name "
                "the runtime's own structured-output tool travels under, and a "
                "prompt may refer to it (D13)."
            )
        return self.bind(**{_SCHEMA_KWARG: schema, _SCHEMA_NAME_KWARG: name})

    def with_structured_output(self, schema: Any, **kwargs: Any):
        """Bind ``schema``, by the same path as :meth:`bind_tools`.

        Note that this returns a model whose answer carries the schema on a
        ``tool_call``, exactly as :meth:`bind_tools` does -- **not** LangChain's
        "returns the parsed object" contract. That is deliberate: it is the
        shape a caller reading ``tool_calls[0]["args"]`` already handles for
        every other provider, and parsing into the class is the application's
        job, not this adapter's -- modelpass hands over ``data`` and never edits
        it, and an adapter that constructed the object would have to decide what
        to do with an answer that failed the structural check.
        """
        return self.bind_tools([schema], tool_choice=getattr(schema, "__name__", None))


def _new_state() -> dict:
    return {
        "text": "",
        "usage": {},
        "guard_warnings": [],
        "failed_over_from": None,
        "crosses_to_metered": False,
        # The modelpass StructuredOutputEvent, when a schema was bound, and
        # whether the schema was converted to OpenAI's strict subset on the way
        # out -- which is what says the answer needs its nulls stripped.
        "structured": None,
        "strict_schema": False,
        # R5. Empty until the receipt event arrives, and legitimately empty
        # afterwards on a call that asked for nothing.
        "sampling_applied": {},
        "sampling_notes": [],
    }


def _usage_dict(usage: TokenUsage) -> dict:
    """A modelpass ``TokenUsage`` as a plain dict.

    Every field, because this is the only producer of the dict
    :func:`_usage_metadata` consumes: a field missing here is silently zero
    there, and reads as a real count rather than an absent one.
    """
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


def _usage_metadata(usage: dict) -> dict:
    """Map modelpass token counts onto LangChain's ``usage_metadata``.

    This is the load-bearing part of the adapter. Cost accounting in LangChain
    applications reads exactly these keys -- a chat model that omits
    ``usage_metadata`` makes every cost metric report zero while looking like it
    worked.

    The arithmetic is a genuine convention change, not a copy.
    :class:`modelpass.TokenUsage` counts ``cached_input_tokens`` **separately
    from** ``input_tokens``; ``usage_metadata`` follows ``langchain-anthropic``,
    where ``input_tokens`` is the *true* total including cached reads and
    ``input_token_details.cache_read`` breaks out the cached part. Matching the
    incumbent matters more than matching modelpass's own arithmetic, because an
    application's stored totals have to mean the same thing whichever provider
    produced them.

    Cache *writes* count toward ``input_tokens`` too. They are billed as input,
    and ``langchain-anthropic``'s ``input_tokens`` is the true total, so omitting
    them under-reports every cold prefix -- silently, and by the whole size of
    the prefix. Until 2026-08-30 modelpass had no cache-creation concept and this
    reported a hardcoded zero; ``TokenUsage.cache_write_tokens`` now carries it.
    """
    cached = int(usage.get("cached_input_tokens", 0) or 0)
    created = int(usage.get("cache_write_tokens", 0) or 0)
    input_tokens = int(usage.get("input_tokens", 0) or 0) + cached + created
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": cached, "cache_creation": created},
    }


def _response_metadata(state: dict) -> dict:
    """The receipt-shaped context worth recording beside a call.

    Every key is namespaced except ``model_name``, which is LangChain's own and
    carries the model the receipt says actually applied -- not the one that was
    asked for, which may have been nothing at all.
    """
    return {
        "subpass_receipt": state.get("receipt_summary", ""),
        "subpass_auth_mode": state.get("auth_mode", ""),
        "subpass_runtime": state.get("runtime", ""),
        "subpass_account": state.get("account"),
        "subpass_plan_name": state.get("plan_name"),
        "model_name": state.get("resolved_model"),
        "subpass_status": state.get("status", ""),
        "subpass_failed_over_from": state.get("failed_over_from"),
        "subpass_crosses_to_metered": state.get("crosses_to_metered", False),
        "subpass_guard_warnings": list(state.get("guard_warnings", [])),
        # R5, ticket 1.7. Two keys rather than one: what went is the fact a
        # reproducibility check needs, and why the rest did not is the sentence
        # a person needs. Added under the existing prefix; nothing was renamed.
        "subpass_sampling_applied": dict(state.get("sampling_applied", {})),
        "subpass_sampling_notes": list(state.get("sampling_notes", [])),
    }


def _raise_for_terminal(event: TerminalEvent, connection: str, state: dict) -> None:
    """Turn a bad terminal status into the exception LangChain callers expect.

    modelpass reports a vendor failure as a normal completion carrying
    ``status="error"``; LangChain callers -- including every retry loop wrapped
    around a chat model -- expect a raise. The receipt and the partial text ride
    along so the raise is as diagnosable as the event was.
    """
    if event.status is TerminalStatus.OK:
        return
    shared = {
        "connection": connection,
        "receipt_summary": state.get("receipt_summary", ""),
        "partial_text": state.get("text", ""),
        "usage": state.get("usage"),
        # The verdict is read off the event's typed fields, never off its text,
        # and every class here carries it rather than only the timeout (1.12b).
        "retryable": event.retryable,
        "retry_after": event.retry_after,
        "status_code": event.status_code,
        "terminal": event,
    }
    reason = event.reason or str(event.status)
    if event.status is TerminalStatus.GUARD_STOP:
        raise SubpassGuardStopError(reason, **shared)
    if event.status is TerminalStatus.TIMED_OUT:
        raise SubpassTimeoutError(reason, **shared)
    if event.status is TerminalStatus.QUOTA_EXHAUSTED:
        raise SubpassQuotaExhaustedError(
            f"{reason}. Neither vendor exposes remaining-allowance state before "
            "a run, so this is the first point at which it is knowable; wait for "
            "the allowance to reset, or configure onQuotaExhausted.failover on "
            "the modelpass connection and set allow_failover=True on this model.",
            **shared,
        )
    raise SubpassRunError(reason, **shared)
