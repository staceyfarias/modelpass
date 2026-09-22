"""Core value types: messages, token usage, and the agent event union.

Two rules shape this module.

1. **Tokens, never dollars** (D7). Two of the three providers draw on an
   allowance rather than a price, so a normalized cost field would be a lie for
   most runs. Vendor cost fields, where they exist, stay inside ``vendor_event``.
2. **Serializable as protocol** (D11). Every type here round-trips through
   ``to_dict()`` into JSON-safe primitives with a string ``type`` discriminator,
   so a second-language port -- or an OpenAI-compatible HTTP server as a leaf
   package -- is a transport away rather than a redesign.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .runtimes import Runtime

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Imported for annotation only: preflight imports this module, so a real
    # import here would be a cycle. Nothing at runtime needs the class -- the
    # encoder asks any dataclass for its own to_dict.
    from .preflight import Receipt

__all__ = [
    "CACHE_TTLS",
    "CALLER_TOOL_SERVER",
    "EVENT_TYPES",
    "REASONING_EFFORTS",
    "SAMPLING_FIELDS",
    "TEMPERATURE_RANGE",
    "TOP_P_RANGE",
    "AgentEvent",
    "AuthMode",
    "CacheControl",
    "ContentBlock",
    "FailoverEvent",
    "GuardStopEvent",
    "GuardWarningEvent",
    "Message",
    "ReceiptEvent",
    "Role",
    "Sampling",
    "SessionInfo",
    "SessionKind",
    "StructuredOutputEvent",
    "TerminalEvent",
    "TerminalStatus",
    "TextBlock",
    "TextDeltaEvent",
    "ThinkingEvent",
    "TokenUsage",
    "ToolCallEvent",
    "ToolResultEvent",
    "UsageEvent",
    "UsageScope",
    "VendorEvent",
    "count_cache_breakpoints",
    "normalize_content",
    "normalize_messages",
    "parse_auth_mode",
]

#: ``ToolCallEvent.server`` for a tool the caller supplied in-process, as opposed
#: to one reached through a named MCP server.
CALLER_TOOL_SERVER = "caller"


class Role(StrEnum):
    """Message roles. v1 is the lowest-level Messages idiom (D7)."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class AuthMode(StrEnum):
    """How a run is paid for.

    ``subscription`` draws on the plan allowance the user already pays for;
    ``api_key`` is metered billing. modelpass never moves between them without an
    explicit instruction (D4, guaranteed layer).
    """

    SUBSCRIPTION = "subscription"
    API_KEY = "api_key"


class SessionKind(StrEnum):
    """Which of the two session faces a runtime is being asked to open (D15).

    The two public classes -- :class:`~modelpass.sessions.ChatSession` and
    :class:`~modelpass.sessions.WorkerSession` -- share one implementation and
    differ only in policy, and this is the value that carries that policy across
    the adapter boundary. An adapter reads it to decide three things, all of
    which the design record's table states outright:

    * ``chat`` -- the runtime's own tools are **off**, ``system_prompt``
      **replaces** whatever persona the runtime ships with, and
      ``project_folder`` defaults to a modelpass-owned scratch directory.
    * ``worker`` -- the runtime's own tools are **on**, ``system_prompt`` is
      **appended** to that persona rather than replacing it, and
      ``project_folder`` is required because a worker with tools pointed at a
      scratch directory is useless.

    It is an enum on the wire rather than two adapter methods because every
    other part of opening a session is identical, and a second method would have
    invited two implementations of the parts that must not differ.
    """

    CHAT = "chat"
    WORKER = "worker"


class UsageScope(StrEnum):
    """What a usage report is counting (Phase 5).

    Guards have to be able to stop a run *mid-loop*, not only at its end (D12's
    corollary), which means folding in usage reports as they arrive. Doing that
    safely needs one bit of information the runtimes do not agree about: whether
    a given report is an increment or a running total for the whole run.

    * ``delta`` -- this report covers work not covered by earlier reports.
      Accumulates. The default, because it is the assumption that under-counts
      rather than over-counts if an adapter gets it wrong.
    * ``run_total`` -- the runtime's own total for the run so far. Replaces the
      accumulated figure rather than adding to it, so a final authoritative
      total cannot double-count the interim reports that preceded it.
    """

    DELTA = "delta"
    RUN_TOTAL = "run_total"


class TerminalStatus(StrEnum):
    """How a run ended.

    ``TIMED_OUT`` joined the five on 2026-09-13 (ticket 1.12) rather than being
    folded into ``CANCELLED``, and the reason is that the two answer different
    questions about the same stopped run: *someone let go of the iterator* and
    *the bound this call was given ran out*. A retry loop wants the second and
    must not touch the first -- a run the user cancelled is not a run to try
    again -- and the verdict in :mod:`modelpass.retry` turns on exactly that
    distinction.

    **Adding a member was checked against the consumers first, because a status
    is a string on a wire.** The four apps reading this vocabulary either match
    ``ok`` and treat everything else as a failure (a downstream agent host),
    match the two budget statuses and fall through to a raise (a retrieval
    service), or render a per-status sentence with a default arm (a desktop
    agent app). That last app's two status tuples are documentation of the
    vocabulary and are never used to validate a value -- grepped 2026-09-13. So
    no consumer exhausts this enum, and a sixth member degrades to each app's
    existing "something went wrong" arm rather than crashing one.
    """

    OK = "ok"
    ERROR = "error"
    GUARD_STOP = "guard_stop"
    QUOTA_EXHAUSTED = "quota_exhausted"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class Retryable(StrEnum):
    """Whether trying this run again is sensible -- the library's verdict (R6).

    Three values and not a boolean, because "we do not know" is a real answer
    and the two-valued version has to guess which way to lie. A vendor that
    reported nothing typed gets :attr:`UNKNOWN`, and a caller decides what their
    own allowance is worth.

    **modelpass never retries for you.** This is a verdict, not a loop: the
    library has no idea what a second run would cost against a subscription
    allowance, and the consumer does.

    Computed only from typed facts -- HTTP status codes, exception classes,
    terminal statuses -- and never from message text. The rule exists because
    substring-matching a status code out of a message once matched ``"500"``
    inside ``"stopAtTokens threshold 500000 reached"`` and retried a
    deterministic guard stop four times against a live allowance (reported by a
    downstream agent host, 2026-08-18).

    The vocabulary lives here beside :class:`TerminalStatus` because it is a
    vocabulary; every verdict that uses it lives in :mod:`modelpass.retry`,
    which re-exports this name.
    """

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Timeout:
    """A wall-clock bound on one call (R6, ticket 1.12).

    ``total`` bounds the whole run: from the moment :meth:`~modelpass.Bridge.chat`
    starts driving the adapter to the terminal event. ``first_token`` bounds the
    narrower wait for the first event that is not the receipt -- the "is anything
    happening at all" question, which on a subprocess-driven runtime is the one
    that distinguishes a slow answer from a runtime that never came up.

    Either may be ``None``, which means "unbounded on this axis". Both ``None``
    is a :class:`Timeout` that does nothing, and is accepted rather than refused
    so a caller can build one from configuration without special-casing the
    empty case.

    On expiry the run is cancelled through the adapter's own ``cancel()`` hook
    -- the documented D10 floor -- and the stream still finishes the way every
    other stream does: one terminal event, stamped, carrying the usage that was
    accumulated before the bound ran out. The status is
    :attr:`TerminalStatus.TIMED_OUT`.

    Seconds, as a float. Zero and negatives are refused: a bound that has
    already expired before the call starts is a configuration mistake, not a
    request to spend nothing.
    """

    total: float | None = None
    first_token: float | None = None

    def __post_init__(self) -> None:
        for name in ("total", "first_token"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Timeout.{name} takes a number of seconds or None, got "
                    f"{type(value).__name__}"
                )
            if value <= 0:
                raise ValueError(
                    f"Timeout.{name} must be greater than zero, got {value!r}. A "
                    "bound that has already expired is a configuration mistake, "
                    "not a request to spend nothing"
                )

    def __bool__(self) -> bool:
        """False when neither axis is bounded, so ``if timeout:`` reads right."""
        return self.total is not None or self.first_token is not None

    @classmethod
    def coerce(cls, value: Timeout | float | int | None) -> Timeout | None:
        """Accept a :class:`Timeout`, a bare number of seconds, or ``None``.

        A bare number is ``total``: it is the bound everyone means when they say
        "time this out", and requiring the wrapper for the common case would buy
        nothing but typing.
        """
        if value is None:
            return None
        if isinstance(value, Timeout):
            return value
        return cls(total=float(value))

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "first_token": self.first_token}


def parse_auth_mode(value: str | AuthMode) -> AuthMode:
    """Parse an auth mode, raising ``ValueError`` with the valid set."""
    if isinstance(value, AuthMode):
        return value
    try:
        return AuthMode(value)
    except ValueError:
        valid = ", ".join(m.value for m in AuthMode)
        raise ValueError(f"unknown auth mode {value!r} (expected one of: {valid})") from None


def _encode(value: Any) -> Any:
    """Convert a value to a JSON-safe primitive."""
    if isinstance(value, StrEnum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        own = getattr(value, "to_dict", None)
        return own() if callable(own) else _as_dict(value)
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


def _as_dict(obj: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    type_name = getattr(obj, "type", None)
    if isinstance(type_name, str):
        data["type"] = type_name
    for f in dataclasses.fields(obj):
        data[f.name] = _encode(getattr(obj, f.name))
    return data


# --- content blocks (R3) -------------------------------------------------------

#: The prefix-cache lifetimes Anthropic's ``cache_control`` accepts. ``None``
#: means "pinned breakpoint, vendor default lifetime", which is a different
#: statement from either of these and is why the field is optional rather than
#: defaulted to ``"5m"``.
CACHE_TTLS: tuple[str, ...] = ("5m", "1h")


@dataclass(frozen=True, slots=True)
class CacheControl:
    """A prefix-cache breakpoint marker, Anthropic's spelling (R3).

    ``type`` has one member today and is still written down, because it is what
    goes on the wire and a vocabulary with an implicit discriminator is the one
    that breaks when a second member arrives. ``ttl`` is ``"5m"`` or ``"1h"``,
    validated here rather than at the adapter: a typo'd lifetime is a caller
    mistake, and hearing about it from a 400 several seconds and one HTTP round
    trip later is strictly worse than hearing about it from the constructor.
    """

    type: Literal["ephemeral"] = "ephemeral"
    ttl: str | None = None

    def __post_init__(self) -> None:
        if self.type != "ephemeral":
            raise ValueError(
                f"cache_control type must be 'ephemeral', got {self.type!r}"
            )
        if self.ttl is not None and self.ttl not in CACHE_TTLS:
            valid = ", ".join(repr(t) for t in CACHE_TTLS)
            raise ValueError(
                f"cache_control ttl must be one of: {valid} (or None for the "
                f"vendor default), got {self.ttl!r}"
            )

    @classmethod
    def coerce(cls, value: CacheControl | Mapping[str, Any]) -> CacheControl:
        if isinstance(value, CacheControl):
            return value
        if isinstance(value, Mapping):
            return cls(
                type=value.get("type", "ephemeral"),  # type: ignore[arg-type]
                ttl=value.get("ttl"),
            )
        raise TypeError(f"cannot read a cache_control from {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.ttl is not None:
            data["ttl"] = self.ttl
        return data


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One span of text, optionally ending a cacheable prefix (R3).

    **Text only, deliberately.** Images, documents and tool-result blocks stay
    out of v1 -- they were out before this type existed and adding a marker for
    prefix caching is not the occasion to let them in. What this vocabulary
    exists for is the one thing a ``str`` could not carry: *where the cacheable
    prefix ends*. A caller that splits its system prompt into four blocks and
    marks the boundaries is making a billing decision, and before this the
    library had nowhere to put it: a downstream agent host's four-breakpoint
    template markers were stripped at the door, and a retrieval service and a
    batch scraping tool lost the same thing.

    ``cache_control`` on a block means "everything up to and including this
    block is one cacheable prefix". modelpass never invents one and never moves
    one; whether a runtime *honours* it is
    :attr:`~modelpass.capabilities.Capability.CACHE_BREAKPOINTS`, and a runtime
    that does not says so on the receipt rather than dropping it silently.
    """

    type: ClassVar[str] = "text"
    text: str
    cache_control: CacheControl | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(
                f"TextBlock text must be str, got {type(self.text).__name__}"
            )
        if self.cache_control is not None:
            object.__setattr__(
                self, "cache_control", CacheControl.coerce(self.cache_control)
            )

    @classmethod
    def coerce(cls, value: TextBlock | Mapping[str, Any] | str) -> TextBlock:
        """Accept a block, a ``{"type": "text", "text": ...}`` dict, or bare text."""
        if isinstance(value, TextBlock):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, Mapping):
            kind = value.get("type", "text")
            if kind != "text":
                raise ValueError(
                    f"modelpass carries text content blocks only, got {kind!r}: "
                    "images and documents stay out of v1"
                )
            raw = value.get("cache_control")
            return cls(
                text=str(value.get("text", "")),
                cache_control=CacheControl.coerce(raw) if raw is not None else None,
            )
        raise TypeError(f"cannot read a content block from {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type, "text": self.text}
        if self.cache_control is not None:
            data["cache_control"] = self.cache_control.to_dict()
        return data


#: The block union. One member today; named so the annotations read as the
#: vocabulary rather than as the single type that currently fills it.
ContentBlock = TextBlock


def normalize_content(
    content: str | Iterable[TextBlock | Mapping[str, Any] | str],
) -> str | tuple[TextBlock, ...]:
    """Coerce message or system-prompt content into its stored shape.

    A ``str`` stays a ``str`` -- byte-for-byte, so nothing about an existing
    caller's prompt moves. Anything iterable becomes a tuple of
    :class:`TextBlock`, which is the only way a breakpoint survives the trip.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (TextBlock, Mapping)):
        return (TextBlock.coerce(content),)
    if isinstance(content, Iterable):
        return tuple(TextBlock.coerce(b) for b in content)
    raise TypeError(
        "content must be str or a sequence of text content blocks, got "
        f"{type(content).__name__}"
    )


def count_cache_breakpoints(
    content: str | Iterable[TextBlock] | None,
) -> int:
    """How many explicit cache breakpoints this content carries. ``0`` for text."""
    if content is None or isinstance(content, str):
        return 0
    return sum(1 for b in content if getattr(b, "cache_control", None) is not None)


# --- messages ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Message:
    """A single conversation message.

    ``content`` is **either** a ``str`` or a tuple of :class:`TextBlock`, which
    is the second accepted shape the original v1 note promised rather than a
    revision of the wire format: a string message serializes exactly as it
    always did. Blocks exist for one reason -- carrying ``cache_control``
    breakpoints (R3) -- and the vocabulary is text-only; image and document
    blocks were out of v1 before this and stay out.

    **Read the text with :attr:`text`, never by assuming ``content`` is one.**
    That property is what keeps every existing consumer working, and it is what
    every adapter in this package uses to render a message onto a wire that
    takes a string.
    """

    role: Role
    content: str | tuple[TextBlock, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", Role(self.role))
        if isinstance(self.content, str):
            return
        try:
            object.__setattr__(self, "content", normalize_content(self.content))
        except TypeError:
            raise TypeError(
                "message content must be str or a sequence of text content "
                f"blocks, got {type(self.content).__name__}"
            ) from None

    @property
    def text(self) -> str:
        """The message's text, blocks concatenated. Always a ``str``.

        Blocks are joined with nothing between them: a breakpoint is a marker
        *inside* one prompt, not a separator between two, so inserting anything
        here would change the bytes a caller wrote.
        """
        if isinstance(self.content, str):
            return self.content
        return "".join(block.text for block in self.content)

    @property
    def flat_text(self) -> str:
        """The message's text for a runtime that cannot carry blocks.

        Blocks are joined with one blank line, which is what the LangChain
        adapter's flattening produced before blocks existed (0.1) and therefore
        what every prompt on the agent runtimes has always read as. ``text``
        stays byte-faithful for the API runtimes, where the blocks travel
        intact and no separator is ever inserted. Two joins, one reason each;
        picking one for both would either glue words on the agent runtimes or
        change the bytes a caller wrote on the API ones (2026-09-13).
        """
        if isinstance(self.content, str):
            return self.content
        return "\n\n".join(block.text for block in self.content if block.text)

    @property
    def blocks(self) -> tuple[TextBlock, ...]:
        """The message as blocks, a plain string wrapped in one unmarked block."""
        if isinstance(self.content, str):
            return (TextBlock(self.content),)
        return self.content

    @property
    def cache_breakpoints(self) -> int:
        """How many explicit cache breakpoints this message carries."""
        return count_cache_breakpoints(
            None if isinstance(self.content, str) else self.content
        )

    @classmethod
    def coerce(cls, value: Message | Mapping[str, Any]) -> Message:
        """Accept either a ``Message`` or the ``{"role": ..., "content": ...}`` dict."""
        if isinstance(value, Message):
            return value
        if isinstance(value, Mapping):
            missing = {"role", "content"} - set(value)
            if missing:
                raise ValueError(f"message is missing key(s): {', '.join(sorted(missing))}")
            return cls(role=Role(value["role"]), content=value["content"])
        raise TypeError(f"cannot read a message from {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {"role": self.role.value, "content": self.content}
        return {
            "role": self.role.value,
            "content": [block.to_dict() for block in self.content],
        }


def normalize_messages(
    messages: Iterable[Message | Mapping[str, Any]],
) -> tuple[Message, ...]:
    """Coerce and validate a caller's message list."""
    normalized = tuple(Message.coerce(m) for m in messages)
    if not normalized:
        raise ValueError("messages must not be empty")
    if not any(m.role is Role.USER for m in normalized):
        raise ValueError("messages must contain at least one user message")
    return normalized


# --- sampling ------------------------------------------------------------------


#: The reasoning dial's three positions (R5). Three rather than a vendor's own
#: list, for the reason a RAG evaluation harness wrote down when it built the
#: same dial (read 2026-09-13): it is the coarsest
#: thing every provider that reasons at all can express, and an operator
#: choosing between five would be choosing noise. Vendors offer more --
#: ``anthropic`` 0.97.0 takes ``low|medium|high|xhigh|max`` and ``openai``
#: 2.32.0 takes ``none|minimal|low|medium|high|xhigh`` -- and a caller who needs
#: one of those is asking for a per-vendor passthrough, which is the thing one
#: abstract dial exists instead of.
#: The one reasoning vocabulary in this library, ascending. Widened from
#: ``("low", "medium", "high")`` on 2026-09-21, additively -- every value that
#: was valid still is.
#:
#: The rungs are the union of what the installed SDKs accept, read that day:
#: ``openai`` 2.32.0 ``ReasoningEffort`` is ``none|minimal|low|medium|high|
#: xhigh``; ``anthropic`` 0.97.0 ``OutputConfigParam.effort`` is
#: ``low|medium|high|xhigh|max``; ``claude-agent-sdk`` 0.2.148 ``EffortLevel``
#: is the same five; ``google-genai`` 1.73.1 ``ThinkingLevel`` is
#: ``MINIMAL|LOW|MEDIUM|HIGH``.
#:
#: ``max`` and ``ultra`` are deliberately **not** here even though two vendors
#: type them. :mod:`modelpass.reasoning` holds the reasons: ``ultra`` also turns
#: on agentic execution and is therefore a second axis, and ``max`` is a name
#: two vendors disagree about. A rung that means different things per vendor is
#: not a portable rung.
REASONING_EFFORTS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)

#: The widest range any supported vendor accepts for ``temperature``. Validated
#: here so a typo'd 20 is a constructor error rather than a 400 one HTTP round
#: trip later, the same bargain :class:`CacheControl` makes about ``ttl``.
#: Narrower per-runtime ceilings -- Anthropic's is 1.0 -- are *not* enforced
#: here: a range that differs by runtime is model knowledge, it lives in
#: :mod:`modelpass.sampling_rules`, and a value over the ceiling is coerced with
#: a note on the receipt rather than refused at the door.
TEMPERATURE_RANGE: tuple[float, float] = (0.0, 2.0)

#: ``top_p`` is a probability mass on every vendor that has one.
TOP_P_RANGE: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True, slots=True)
class Sampling:
    """How the model should pick its words, as a request (R5).

    Every field is optional and ``None`` means *say nothing about it*, which is
    distinct from any value a caller could pass: ``temperature=0.0`` asks for
    determinism, and ``temperature=None`` lets the model's own default stand.
    Nothing here is a promise. A request field is what the caller asked for; what
    was actually sent is on the receipt, because **support varies by model and
    not only by runtime** -- GPT-5 forces temperature 1.0, several Anthropic 4.x
    models accept one of temperature or top_p but not both, and the o-series
    refuses all three. That is R5's whole point and
    :mod:`modelpass.sampling_rules` is where the per-model knowledge lives.

    ``max_output_tokens`` is a **ceiling on the answer**, and it is deliberately
    a different thing from ``Guards.stop_at_tokens``, which the two are
    routinely confused. The ceiling is a request parameter the vendor enforces
    by truncating; the guard is modelpass's own tripwire over *total* spend
    across a whole run and it ends the stream. A caller wanting shorter answers
    wants this; a caller wanting a bounded bill wants the guard; a caller wanting
    both sets both (DESIGN R5; validation 2026-09-13, where a downstream agent
    host's own support module already says so in prose).

    ``reasoning_effort`` is one abstract dial across vendors -- see
    :data:`REASONING_EFFORTS`. What it becomes on the wire is per runtime and per
    model: ``output_config.effort`` on ``anthropic-api``, ``reasoning_effort`` on
    the OpenAI models that have one, nothing at all on a model that cannot
    reason, in which case it is reported dropped rather than quietly ignored.
    Some models make it *exclusive* with the sampling fields -- Anthropic's
    adaptive-thinking families remove temperature, top_p and top_k from a
    thinking request -- and that too is reported, because a judge configured at
    temperature 0 that starts thinking is no longer at temperature 0.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        self._check_fraction("temperature", self.temperature, TEMPERATURE_RANGE)
        self._check_fraction("top_p", self.top_p, TOP_P_RANGE)
        self._check_positive_int("top_k", self.top_k)
        self._check_positive_int("max_output_tokens", self.max_output_tokens)
        if self.reasoning_effort is not None:
            effort = str(self.reasoning_effort).strip().lower()
            if effort not in REASONING_EFFORTS:
                valid = ", ".join(repr(e) for e in REASONING_EFFORTS)
                raise ValueError(
                    f"reasoning_effort must be one of: {valid} (or None to say "
                    f"nothing about it), got {self.reasoning_effort!r}"
                )
            object.__setattr__(self, "reasoning_effort", effort)

    @staticmethod
    def _check_fraction(name: str, value: Any, bounds: tuple[float, float]) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number or None, got {type(value).__name__}")
        low, high = bounds
        if not low <= float(value) <= high:
            raise ValueError(f"{name} must be between {low} and {high}, got {value!r}")

    @staticmethod
    def _check_positive_int(name: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int or None, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value!r}")

    @property
    def is_empty(self) -> bool:
        """Whether this asks for nothing -- the same as having passed no sampling."""
        return not self.requested()

    def requested(self) -> dict[str, Any]:
        """The fields the caller actually set, in declaration order.

        The report's left-hand column. An unset field is absent rather than
        ``None``, so "asked for nothing" and "asked for temperature 0" never
        collapse into the same dict.
        """
        return {
            name: getattr(self, name)
            for name in SAMPLING_FIELDS
            if getattr(self, name) is not None
        }

    @classmethod
    def coerce(cls, value: Sampling | Mapping[str, Any] | None) -> Sampling | None:
        """A :class:`Sampling`, a mapping of its fields, or ``None``."""
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - set(SAMPLING_FIELDS))
            if unknown:
                known = ", ".join(SAMPLING_FIELDS)
                raise TypeError(
                    f"unknown sampling field(s) {unknown}: modelpass's sampling "
                    f"vocabulary is {known}"
                )
            return cls(**value)
        raise TypeError(f"cannot read sampling controls from {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form (D11). Only what was set, like :meth:`requested`."""
        return self.requested()


#: The vocabulary, in the order a report lists it. Named once so the request
#: type, the rules table and both honesty reports cannot drift apart.
SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "reasoning_effort",
)


# --- sessions ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """One session a runtime knows about, as :meth:`Bridge.list_sessions` reports it.

    Normalized to the **intersection** of what the runtimes agree on, with
    everything else in ``vendor`` -- the same division
    :class:`VendorEvent` draws, and for the same reason. Anthropic's listing
    carries a working directory, a summary and a leaf message id; Codex's
    ``thread/list`` carries a source kind, a provider and a title. Inventing a
    common shape for those would be the fiction D7 refused for thinking, and
    dropping them would make a caller re-read the transcript files modelpass just
    enumerated.

    ``id`` is the only field guaranteed to be present, because it is the only
    one :meth:`Bridge.resume_chat` needs. Every other normalized field is
    ``None`` when the runtime did not say, and ``None`` means *not reported*
    rather than *empty* -- a listing that guessed a timestamp would be worse
    than one that admits it does not have one.

    Timestamps are **ISO 8601 strings, not ``datetime``**. D11 asks that these
    types round-trip into JSON-safe primitives, and the runtimes hand over
    strings; parsing them into ``datetime`` here would mean re-encoding them on
    the way out and hoping the two spellings match.
    """

    id: str
    connection: str
    runtime: Runtime
    project_folder: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    title: str | None = None
    message_count: int | None = None
    vendor: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)


# --- usage ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for a run. Never dollars -- see the module docstring.

    **The convention, stated because shim authors kept having to derive it**
    (second-consumer feedback, 2026-08-17):

    * ``cached_input_tokens`` is counted **separately from** ``input_tokens``,
      not inside it. Cache *reads* land here.
    * ``cache_write_tokens`` is separate again, and is the field that makes
      "is my caching working" answerable. Cache writes are billed as input, so
      they used to be folded into ``input_tokens`` -- which meant a cold prefix
      and genuinely new content were indistinguishable in the record. A run
      reporting 6,222 input tokens where 6,220 of them were a cache write reads
      as heavy usage when it is really a cache miss, and no amount of querying
      the run log could tell the two apart (2026-08-30).
    * ``total_tokens`` is the sum of **all four** fields. A caller wanting
      "everything that went in" wants
      ``input_tokens + cache_write_tokens + cached_input_tokens``; one wanting
      "what did the prefix cost me this run" wants ``cache_write_tokens``
      against ``cached_input_tokens``.

    The split exists because the two halves are priced differently everywhere
    and drawn from an allowance differently on a subscription, so collapsing
    them would lose the one number a user can act on. Getting the convention
    wrong in either direction double-counts or under-counts a run, which is why
    it is written down here rather than left to be inferred from
    :meth:`__add__`.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    #: Output tokens the model spent thinking -- **a subset of**
    #: :attr:`output_tokens`, never a peer of it, and therefore **excluded from**
    #: :attr:`total_tokens`. This is the opposite convention from
    #: :attr:`cached_input_tokens`, which is counted separately from
    #: :attr:`input_tokens`, and the difference is stated here because getting
    #: it backwards double-counts every reasoning run.
    #:
    #: The relation was **checked, not assumed**, on each runtime that reports
    #: one (2026-09-22):
    #:
    #: * ``anthropic-sdk`` -- ``usage.output_tokens_details.thinking_tokens``,
    #:   driven live on claude-code 2.1.278: ``output_tokens 996`` of which
    #:   ``thinking_tokens 993``, answer text ``'5'``.
    #: * ``openai-sdk`` -- ``reasoningOutputTokens`` on ``TokenUsageBreakdown``.
    #:   66,406 ``token_count`` records in a real Codex history: **zero** where
    #:   reasoning exceeded output, and ``input + output == total`` throughout,
    #:   so the vendor does not add it in either.
    #: * ``openai-api`` / ``openai-compatible`` -- ``reasoning_tokens`` inside
    #:   the output-token details object, which is what "details" means.
    #: * ``google-api`` -- the **exception on the wire**:
    #:   ``thoughts_token_count`` is a *peer* of ``candidates_token_count`` and
    #:   the vendor's own total sums both. The adapter already folds thoughts
    #:   into :attr:`output_tokens` for cross-runtime comparability, so by the
    #:   time a count reaches this field the subset relation holds here too.
    #:
    #: ``None`` means **not reported**, and is not the same claim as ``0``.
    #: ``anthropic-api`` is ``None`` forever -- ``anthropic`` 0.97.0's ``Usage``
    #: has no details object at all, so a 996-token answer that thought for 993
    #: is indistinguishable there from 996 tokens of prose. A run that genuinely
    #: did no thinking reports ``0``. Collapsing the two would turn "the vendor
    #: cannot tell us" into "the model did not think", which is the single
    #: inference this field exists to prevent.
    reasoning_output_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        """Every token this run reported: input + output + cache write + read."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cached_input_tokens
            + self.cache_write_tokens
        )

    @property
    def prompt_tokens(self) -> int:
        """Everything that went *in*: fresh input, cache reads and cache writes.

        The denominator for any "what fraction of my prompt was cached" question
        (2026-09-22), and a property rather than a sum each caller writes for
        itself because writing it out is where the nesting mistakes happen --
        modelpass counts these three side by side, and one vendor's wire nests
        two of them. Getting it wrong reports a cache break as a hit.
        """
        return self.input_tokens + self.cached_input_tokens + self.cache_write_tokens

    @property
    def billable_input_tokens(self) -> int:
        """Input the run paid full or premium rate for: fresh plus cache writes.

        The number a guard has to bound. Cache writes are billed as input, so a
        guard that watched ``input_tokens`` alone after the split would silently
        stop counting them.
        """
        return self.input_tokens + self.cache_write_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_output_tokens=_add_optional(
                self.reasoning_output_tokens, other.reasoning_output_tokens
            ),
        )

    def at_least(self, other: TokenUsage) -> TokenUsage:
        """The component-wise maximum of two counts.

        Used to fold in a ``run_total`` report: the runtime's own total for the
        run is authoritative, but it must never drag the figure *down* below what
        was already observed. If a runtime's "total" turns out to be per-turn
        after all, this keeps the accumulated interim sum instead of quietly
        under-reporting a run's spend -- the direction a guard should err in.
        """
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=max(self.input_tokens, other.input_tokens),
            output_tokens=max(self.output_tokens, other.output_tokens),
            cached_input_tokens=max(self.cached_input_tokens, other.cached_input_tokens),
            cache_write_tokens=max(self.cache_write_tokens, other.cache_write_tokens),
            reasoning_output_tokens=_max_optional(
                self.reasoning_output_tokens, other.reasoning_output_tokens
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = _as_dict(self)
        data["total_tokens"] = self.total_tokens
        return data


def _add_optional(left: int | None, right: int | None) -> int | None:
    """Sum two reasoning counts where either may be *not reported*.

    ``None`` + ``None`` is ``None``: folding two silences must not manufacture a
    zero, which would read as "this run did no thinking". ``None`` + a number is
    that number, because discarding a count we actually hold is the worse of the
    two errors -- and the mixed case is an anomaly rather than a shape to design
    around, since a runtime either reports this metric or does not.
    """
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def _max_optional(left: int | None, right: int | None) -> int | None:
    """Component-wise maximum, with ``None`` meaning *no report* rather than zero.

    The same asymmetry :meth:`TokenUsage.at_least` has everywhere else: a
    ``run_total`` that reports nothing must not drag an observed count down, so
    a number always beats a silence.
    """
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


# --- events --------------------------------------------------------------------


class _Event:
    """Shared behavior for the event union. Not a dataclass itself."""

    __slots__ = ()

    type: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)


@dataclass(frozen=True, slots=True)
class ReceiptEvent(_Event):
    """The preflight receipt for the connection that is about to run.

    **The first event of every chat stream**, before anything an adapter
    produced, and emitted again -- for the second connection -- immediately
    after a :class:`FailoverEvent`. The invariant is one sentence: *a
    ``receipt`` event always precedes any event produced by the connection it
    describes.*

    Added on first-consumer feedback (2026-08-17). The stream contract already
    promised that holding an iterator means the auth check passed; what it did
    not do was hand over the *evidence*, so a caller that has to log "which
    subscription paid for this" ran the preflight a second time itself. That is
    two round trips for one fact, and worse, two chances for the answer to
    differ. Now the guarantee carries its own proof.

    ``receipt`` is the live :class:`~modelpass.preflight.Receipt` -- so
    ``event.receipt.summary()``, ``.account``, ``.plan_name``, ``.model`` are
    all right there -- and it serializes whole under D11, because a ``Receipt``
    knows how to turn itself into JSON.

    The three stamp fields repeat :class:`TerminalEvent`'s so that "which
    connection, which mode" can be matched at the *start* of a run as well as at
    its end, without reaching inside the receipt. ``auth_mode`` is the effective
    mode -- what the preflight detected, not merely what was declared.
    """

    type: ClassVar[str] = "receipt"
    connection: str
    runtime: Runtime
    auth_mode: AuthMode
    receipt: Receipt

    @property
    def summary(self) -> str:
        """The one human line -- the same one ``modelpass check`` prints."""
        return self.receipt.summary()


@dataclass(frozen=True, slots=True)
class TextDeltaEvent(_Event):
    """Incremental assistant output."""

    type: ClassVar[str] = "text_delta"
    text: str


@dataclass(frozen=True, slots=True)
class ThinkingEvent(_Event):
    """Reasoning output, passed through opaquely.

    Deliberately not structurally normalized (D7): the providers disagree about
    what thinking *is*, and inventing a common shape would be fiction.
    """

    type: ClassVar[str] = "thinking"
    text: str


@dataclass(frozen=True, slots=True)
class UsageEvent(_Event):
    """A usage report from the runtime, in tokens.

    ``usage`` is what the runtime reported; ``scope`` says how to read it (see
    :class:`UsageScope`); ``cumulative`` is the bridge's running total for the
    run, injected on the way past so a caller can render a live counter without
    doing the accounting itself.

    Runs may emit several of these. On ``anthropic-sdk`` one arrives per
    assistant turn, which is what lets a guard stop a tool loop between rounds;
    on ``openai-sdk`` the runtime reports only at the end of the exec run. The
    capability registry states that asymmetry rather than smoothing it over.
    """

    type: ClassVar[str] = "usage"
    usage: TokenUsage = field(default_factory=TokenUsage)
    cumulative: TokenUsage | None = None
    scope: UsageScope = UsageScope.DELTA


@dataclass(frozen=True, slots=True)
class GuardWarningEvent(_Event):
    """A configured ``warnAt`` threshold has been crossed (D4, offered layer)."""

    type: ClassVar[str] = "guard_warning"
    guard: str
    threshold: int
    observed: int
    connection: str
    message: str = ""


@dataclass(frozen=True, slots=True)
class GuardStopEvent(_Event):
    """A configured ``stopAt`` threshold has been crossed; the run is being stopped."""

    type: ClassVar[str] = "guard_stop"
    guard: str
    threshold: int
    observed: int
    connection: str
    message: str = ""


@dataclass(frozen=True, slots=True)
class ToolCallEvent(_Event):
    """The runtime is invoking a tool (D12).

    **Observed, not answered.** There is no response channel on this event and
    that is deliberate: the runtime executes the tool and continues its own loop
    inside the same run. A caller renders this, logs it, or ignores it. The
    caller-side round-trip loop -- where the caller would receive this and hand
    a result back -- is the deferred compatibility mode in D12, and giving this
    event a reply path now would be the first half of building it by accident.

    ``name`` is the tool's own short name; ``server`` says where it lives -- an
    MCP server name, or :data:`CALLER_TOOL_SERVER` for a function the caller
    supplied in-process. The pair is kept split rather than joined into the
    runtimes' ``mcp__server__tool`` spelling, because Codex reports them as two
    fields and a caller filtering by server should not have to parse a string.
    """

    type: ClassVar[str] = "tool_call"
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    id: str = ""
    server: str = ""


@dataclass(frozen=True, slots=True)
class ToolResultEvent(_Event):
    """What a tool returned, correlated to its call by ``id``.

    ``content`` is flattened to text: the runtimes' result payloads are MCP
    content-block arrays that may carry images and embedded resources, and
    normalizing those into a common shape would be the same fiction D7 refused
    for thinking. The unflattened payload stays available as a ``vendor_event``
    where the runtime emits one.

    ``name`` is best-effort. Anthropic's ``ToolResultBlock`` carries only
    ``tool_use_id``, so the adapter correlates it back to the call that produced
    it; if the call was never seen, this is empty rather than guessed.
    """

    type: ClassVar[str] = "tool_result"
    id: str = ""
    name: str = ""
    content: str = ""
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class StructuredOutputEvent(_Event):
    """The schema-bound answer, when the caller asked for one (D13).

    Emitted **once per run, immediately before the terminal event**, and only
    when ``chat(..., schema=...)`` was passed. Text deltas still stream as they
    occur: on both v1 runtimes the model produces prose, or the JSON itself, on
    its way to the structured answer, and suppressing that would hide a real
    part of what was paid for.

    ``data`` is the parsed object. ``raw`` is the JSON text it was parsed from,
    kept because the two runtimes hand over different halves -- ``anthropic-sdk``
    gives a parsed object *and* the text, ``openai-sdk`` gives only text -- and a
    caller that wants to log exactly what came back should not have to re-encode
    it and hope the encoding matches.

    ``valid`` is **modelpass's own structural check** (see
    :mod:`modelpass.schema` for exactly which keywords it covers), not the
    vendor's. Vendor-side validation is primary on both runtimes and is what
    actually constrains the model; this flag exists so a caller can branch
    without re-validating, and it never changes ``data`` -- a failed check
    reports the problems and hands over what arrived. modelpass does not invent
    fields to make a schema fit.

    ``data`` is ``None`` only when nothing parseable arrived at all, and that
    case also ends the run with a ``terminal`` carrying ``status="error"`` and a
    reason naming what did come back. An empty dict is never substituted for a
    missing answer.
    """

    type: ClassVar[str] = "structured_output"
    data: Any = None
    raw: str = ""
    valid: bool = False
    schema_name: str = ""
    problems: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FailoverEvent(_Event):
    """The run moved to a second connection after the first ran out of allowance.

    Emitted by the bridge -- never by an adapter -- immediately before the
    failover connection's first event, and only after that connection's own
    preflight has passed.

    **The receipt is not on this event; it follows it.** Until 2026-08-17 this
    event carried a serialized receipt dict, which was the only place in the
    library a receipt reached a caller. Making the primary connection's receipt
    a first-class :class:`ReceiptEvent` would have left two ways to be handed
    the same thing -- a dict here, an object there -- so the dict went and the
    failover connection's receipt arrives as a ``ReceiptEvent`` immediately
    after this event, exactly like the primary connection's. One vocabulary, one
    shape, and the D4(d) guarantee ("its receipt reaches the caller *before* it
    does any work") is unchanged.

    **Why this is a normalized event and not a ``vendor_event``.** A
    ``vendor_event`` is defined as something the *vendor runtime* reported that
    the normalized vocabulary does not cover, and its ``runtime`` field names
    whose runtime said it. A failover receipt is modelpass's own statement about a
    decision modelpass made, so putting it there would mean either inventing a
    runtime attribution or lying about one. More importantly, D3 makes "which
    connection and which auth mode actually ran" a guaranteed property rather
    than passthrough: a caller must be able to detect a billing-mode change by
    matching on the event vocabulary, not by string-matching a vendor payload.

    ``usage`` is what the *first* connection spent before it gave out, so nothing
    is lost when the terminal event goes on to report the second connection's
    total.
    """

    type: ClassVar[str] = "failover"
    from_connection: str
    to_connection: str
    from_auth_mode: AuthMode
    to_auth_mode: AuthMode
    to_runtime: Runtime
    reason: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def crosses_to_metered(self) -> bool:
        """Whether this failover moves a subscription run onto metered billing.

        Always the result of explicit configuration -- naming an ``api_key``
        connection as the failover target *is* the consent -- but a caller that
        wants to refuse, log loudly, or prompt gets one property to check
        instead of comparing two enums.
        """
        return (
            self.from_auth_mode is AuthMode.SUBSCRIPTION
            and self.to_auth_mode is AuthMode.API_KEY
        )


@dataclass(frozen=True, slots=True)
class TerminalEvent(_Event):
    """The last event of every stream that completes normally.

    Stamped with the connection name and the auth mode actually used, so a
    conversation that mixed billing modes is auditable after the fact (D3). The
    bridge owns this stamp; adapters may not forge it.

    ``failed_over_from`` names the connection that ran *first* when quota
    exhaustion moved the call to another one. Without it the terminal would name
    only the connection that finished, and a run that started on a subscription
    and ended on a metered key would be indistinguishable from one that was
    metered all along.

    ``retryable``, ``status_code`` and ``retry_after`` are the retryability
    verdict and the two typed facts it was computed from (R6, ticket 1.12).
    ``status_code`` is the HTTP status the vendor SDK carried on its exception,
    where there was one; ``retry_after`` is the seconds the vendor asked for,
    read off the response headers and never off the message. Both are ``None``
    on the agent runtimes, which report neither -- an absence, and the reason
    those families' verdicts are mostly :attr:`Retryable.UNKNOWN`.

    All three default to the shape a hand-built terminal already had, so an
    adapter that fills none of them is stamped by the bridge exactly as before.
    """

    type: ClassVar[str] = "terminal"
    status: TerminalStatus
    connection: str
    runtime: Runtime
    auth_mode: AuthMode
    reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    failed_over_from: str | None = None
    retryable: Retryable = Retryable.UNKNOWN
    status_code: int | None = None
    retry_after: float | None = None
    #: The effort level this run was told to use, in the runtime's own spelling,
    #: or ``None`` when the connection stated none (2026-09-22). Copied from the
    #: receipt onto the terminal so that *what we asked for* and *what it cost*
    #: arrive in the same object -- a caller comparing the two across a batch
    #: would otherwise have to keep the receipt and match it up by hand.
    reasoning_value: str | None = None
    #: The level the **vendor says** the run used, where a vendor says anything.
    #: ``openai-sdk`` answers ``thread/start`` with the thread's own
    #: ``reasoningEffort``; ``anthropic-sdk`` and ``anthropic-api`` echo nothing
    #: at all, which was driven rather than assumed (2026-09-22) -- so ``None``
    #: here is usually *no echo exists*, not *the run failed to report*. When it
    #: is present and differs from :attr:`reasoning_value`, the server did not
    #: take the level modelpass sent.
    reasoning_echo: str | None = None
    #: ``reported`` / ``unreported`` / ``unavailable`` -- see
    #: :class:`~modelpass.reasoning.ReasoningMetric`. The word that says how to
    #: read ``usage.reasoning_output_tokens``, so a consumer never has to infer
    #: *why* a count is missing from the fact that it is.
    reasoning_metric: str | None = None
    #: Whether *this turn* reused the cached prefix across an effort change
    #: (2026-09-22). ``None`` unless the run actually changed effort mid-session
    #: **and** reported cache counts -- which is a strictly narrower thing than
    #: the receipt's ``effort_cache_continuity``, and the narrowness is the
    #: point. The receipt says what the vendor allows; this says what happened.
    #: A capability of ``supported`` never implies a ``True`` here, and this
    #: field is never derived from it.
    #:
    #: ``None`` on every run today, because no adapter changes effort
    #: mid-session yet: the seam is a ``reasoning_effort_changed`` vendor event,
    #: which :class:`~modelpass._fold.RunFold` observes and nothing emits. That
    #: is a gap stated rather than hidden -- the alternative was a field that
    #: quietly reported ``False`` for every run that never changed anything.
    effort_change_cache_preserved: bool | None = None


@dataclass(frozen=True, slots=True)
class VendorEvent(_Event):
    """Anything the normalized vocabulary does not cover, passed through intact.

    Retries, permission requests, plan detection, vendor cost fields and every
    tool payload richer than :class:`ToolCallEvent` / :class:`ToolResultEvent`
    arrive here rather than being dropped or half-normalized.
    """

    type: ClassVar[str] = "vendor_event"
    runtime: Runtime
    name: str
    data: Mapping[str, Any] = field(default_factory=dict)


AgentEvent = (
    ReceiptEvent
    | TextDeltaEvent
    | ThinkingEvent
    | UsageEvent
    | GuardWarningEvent
    | GuardStopEvent
    | ToolCallEvent
    | ToolResultEvent
    | StructuredOutputEvent
    | FailoverEvent
    | TerminalEvent
    | VendorEvent
)

#: Every event class, keyed by wire discriminator.
EVENT_TYPES: dict[str, type] = {
    cls.type: cls
    for cls in (
        ReceiptEvent,
        TextDeltaEvent,
        ThinkingEvent,
        UsageEvent,
        GuardWarningEvent,
        GuardStopEvent,
        ToolCallEvent,
        ToolResultEvent,
        StructuredOutputEvent,
        FailoverEvent,
        TerminalEvent,
        VendorEvent,
    )
}
