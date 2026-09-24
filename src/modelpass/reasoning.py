"""Asking a model to think harder, in one vocabulary, and being told what that became.

The owner ask (2026-09-21): a caller states an effort level once, modelpass maps
it to the best equivalent on whatever runtime actually runs, and the receipt
reports the **runtime's own** setting rather than the abstract one. The mapping
is the convenience; the report is what keeps the convenience honest.

**This is the move prompt caching refused, and the difference is load-bearing.**
:mod:`modelpass.prompt_cache` records a standing constraint -- no house TTL
vocabulary, because ``'5m'``/``'1h'`` and ``'in-memory'``/``'24h'`` are disjoint
and any translation between them is an equivalence nobody published. A house
*effort* ladder is the same shape of move, so it needs its own justification
rather than inheriting that one's. It has one: **five of the six runtimes below
take an ordered ladder**, and they share most of their members. These are the
same quantity in the same units, which the two cache lifetimes were not.

Two vendors go further and publish, per model, which rungs that model actually
has -- ``anthropic`` 0.97.0 ``EffortCapability`` (a ``CapabilitySupport`` for
each of low/medium/high/xhigh/max) and ``openai-codex`` 0.154.0
``supportedReasoningEfforts`` on the model catalog entry. Which rungs exist is
therefore a thing to *ask* on those runtimes rather than a table to maintain and
watch go stale. :data:`RUNTIME_EFFORTS` is the static floor; asking is a later
refinement and the better answer.

**Two axes, not one ladder.** The ask arrived as a single scale ending in
``ultra``. It cannot be, and the vendors say so themselves:

* ``openai-codex`` 0.154.0 declares ``multiAgentVersion``
  (``disabled``/``v1``/``v2``) **on the model catalog entry** and
  ``multiAgentMode`` (``explicitRequestOnly``/``proactive``) as a separate
  per-turn setting -- alongside ``reasoningEffort``, not inside it.
* ``claude-agent-sdk`` 0.2.148 gives every ``AgentDefinition`` its own
  ``effort``. Effort is a property *of each agent*; orchestration is the thing
  that makes there be several.

So an ``ultra``-shaped setting is a **composite** -- a depth plus an execution
mode -- and a caller who asks to think harder must never be handed multi-agent
execution as a side effect: different cost shape, different latency, different
tool and permission surface, and on one runtime a capability the *model* has to
declare before it exists at all. :data:`EFFORT_LADDER` is depth only. The other
axis gets its own key when a runtime for it is established; until then asking
for it here is refused with a message that says which axis it belongs to.

**``max`` is left out too, and for a weaker reason -- which is why it is said
out loud.** ``claude-agent-sdk`` types ``max`` as a plain effort level and
documents it as "maximum effort", which reads like a rung. ``openai-codex``
types it beside ``ultra``, which does not. One vendor's rung and another's
composite cannot share a name in a portable vocabulary, so neither spelling is
offered until somebody establishes what Codex's ``max`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .capabilities import DEFAULT_REGISTRY, Capability, CapabilityRegistry, Support
from .errors import InvalidConnection
from .runtimes import Runtime
from .types import REASONING_EFFORTS

__all__ = [
    "EFFORT_LADDER",
    "OPTION_EFFORT",
    "OPTION_THINKING",
    "REASONING_METRIC_FIELDS",
    "RUNTIME_EFFORTS",
    "THINKING_OFF",
    "Effort",
    "ReasoningDisposition",
    "ReasoningMetric",
    "ReasoningPlan",
    "plan_reasoning",
    "reasoning_metric",
    "stated_reasoning",
]


class Effort(StrEnum):
    """How hard to think, on the model the caller already chose.

    Ordered, and the order is the whole point: :meth:`rank` is what lets a
    runtime that is missing a rung say *which way* it would have to move to
    honour the request, rather than only that it cannot.

    Deliberately absent: ``ultra`` and ``max``. See the module docstring -- one
    is a different axis and the other is a name two vendors disagree about.
    """

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"

    @property
    def rank(self) -> int:
        return EFFORT_LADDER.index(self)


#: The house ladder, ascending. Every member names a depth of thinking and
#: nothing else.
EFFORT_LADDER: tuple[Effort, ...] = tuple(Effort(e) for e in REASONING_EFFORTS)

#: Members a caller may ask for that are **not** depth, mapped to the sentence
#: that says so. Refused rather than silently accepted, because the failure this
#: prevents is a caller asking to think harder and being handed a different
#: execution mode.
_OTHER_AXIS: dict[str, str] = {
    "ultra": (
        "'ultra' is not a depth of thinking on the model you chose -- on the "
        "runtimes that have it, it also turns on agentic execution that can "
        "spin off subagents (openai-codex 0.154.0 declares multiAgentVersion "
        "on the model and multiAgentMode per turn, beside reasoningEffort "
        "rather than inside it). That is a different cost shape, latency "
        "profile and tool surface than a deeper think, so modelpass will not "
        "hand it to you under a key that says effort"
    ),
    "max": (
        "'max' is not portable: claude-agent-sdk 0.2.148 types it as a plain "
        "effort level, openai-codex 0.154.0 types it beside 'ultra'. One "
        "vendor's rung and another's composite cannot share a name here until "
        "somebody establishes what Codex's 'max' does"
    ),
}

#: What each runtime accepts, **in its own spelling**, mapped from the house
#: rung. Every entry was read off the SDK installed in this repository on
#: 2026-09-21, and the read is named beside it. Nothing here is remembered from
#: a documentation page, for the reason ``maxInputTokens`` ships no table of
#: model windows: a constant that reads as authoritative and goes stale
#: silently is worse than no constant.
#:
#: A runtime absent from this mapping carries no effort from modelpass. That is
#: not a claim the vendor has no such control -- it is the narrower, checkable
#: claim that none is established here, and the refusal says so.
RUNTIME_EFFORTS: dict[Runtime, dict[Effort, str]] = {
    # openai 2.32.0, openai/types/shared/reasoning_effort.py:
    # ``ReasoningEffort: TypeAlias = Optional[Literal["none", "minimal",
    # "low", "medium", "high", "xhigh"]]``, read 2026-09-21. The house ladder
    # is this vendor's ladder exactly, which is a fact about where the ladder
    # came from and not a reason to treat it as neutral.
    Runtime.OPENAI_API: {e: e.value for e in EFFORT_LADDER},
    # openai-codex 0.154.0, v2_all.py ``class ReasoningEffort``: the same six
    # plus ``max`` and ``ultra``, read 2026-09-21. The extra two are excluded
    # above rather than here, because the reason is about what they mean and
    # not about this runtime.
    Runtime.OPENAI_SDK: {e: e.value for e in EFFORT_LADDER},
    # claude-agent-sdk 0.2.148, types.py:
    # ``EffortLevel: TypeAlias = Literal["low", "medium", "high", "xhigh",
    # "max"]`` on ``ClaudeAgentOptions.effort``, read 2026-09-21. No ``none``
    # and no ``minimal``: this runtime's floor is ``low``, so the bottom two
    # rungs have nowhere to land and are refused rather than rounded up.
    Runtime.ANTHROPIC_SDK: {
        Effort.LOW: "low",
        Effort.MEDIUM: "medium",
        Effort.HIGH: "high",
        Effort.XHIGH: "xhigh",
    },
    # google-genai 1.73.1, ``types.ThinkingLevel``: ``MINIMAL``, ``LOW``,
    # ``MEDIUM``, ``HIGH`` (plus ``THINKING_LEVEL_UNSPECIFIED``), read
    # 2026-09-21. Stops at HIGH, so ``xhigh`` has nowhere to land here.
    Runtime.GOOGLE_API: {
        Effort.MINIMAL: "MINIMAL",
        Effort.LOW: "LOW",
        Effort.MEDIUM: "MEDIUM",
        Effort.HIGH: "HIGH",
    },
    # anthropic 0.97.0, ``anthropic/types/output_config_param.py``:
    # ``OutputConfigParam.effort: Optional[Literal["low", "medium", "high",
    # "xhigh", "max"]]``, read 2026-09-21. The same five as the agent SDK's
    # ``EffortLevel``, which is not a coincidence -- one vendor, one ladder.
    #
    # This runtime also takes ``thinking.budget_tokens``, an integer, and the
    # two are different questions: a level says how hard, a budget says how
    # many tokens. modelpass carries the level, because that is the portable
    # one; a rung-to-budget mapping would be modelpass choosing a token count
    # per model, which is the table ``maxInputTokens`` refuses to ship.
    Runtime.ANTHROPIC_API: {
        Effort.LOW: "low",
        Effort.MEDIUM: "medium",
        Effort.HIGH: "high",
        Effort.XHIGH: "xhigh",
    },
}

#: Runtimes where ``none`` is a **switch** rather than a rung: thinking turned
#: off through a separate option, in that option's own spelling. Consulted
#: only for :attr:`Effort.NONE`, and never as a rung to round to -- a caller
#: who asked for ``minimal`` on a runtime whose ladder starts at ``low`` has
#: asked for *some* thinking, and is moved up to ``low`` rather than down to
#: none at all.
#:
#: claude-agent-sdk 0.2.148, read 2026-09-23: ``ClaudeAgentOptions.thinking:
#: ThinkingConfig | None`` where ``ThinkingConfig`` includes
#: ``ThinkingConfigDisabled = {"type": "disabled"}`` (types.py), and
#: ``_internal/transport/subprocess_cli.py`` turns it into
#: ``--thinking disabled`` on the CLI. Before this entry, ``none`` on this
#: runtime was moved up to ``low`` and reported -- honest, and still a caller
#: who asked for no thinking getting thinking. Measured the day it was found
#: (RAGauge, 2026-09-22): one call at the default level spent 31,140 of its
#: 34,274 output tokens thinking.
THINKING_OFF: dict[Runtime, str] = {
    Runtime.ANTHROPIC_SDK: "disabled",
}

#: The runtime option each kind of plan travels on.
OPTION_EFFORT = "effort"
OPTION_THINKING = "thinking"


class ReasoningMetric(StrEnum):
    """Whether this run's reasoning-token count is a number, a silence, or a void.

    Three values rather than a count-or-``None``, because ``None`` alone cannot
    say *why* there is no number, and the two reasons call for different actions
    from a caller. A consumer sweeping effort levels can retry an
    :attr:`UNREPORTED` run and learn something; retrying an :attr:`UNAVAILABLE`
    one on the same connection will never produce a count no matter how many
    times they try.

    The distinction is the same one :class:`~modelpass.capabilities.Support`
    draws between ``unsupported`` and ``unverified``: an absence somebody
    established, against an absence nobody has looked into.
    """

    #: The runtime reported a count. ``0`` is a report -- the model thought for
    #: no tokens -- and is not the same answer as either member below.
    REPORTED = "reported"
    #: This runtime does report the metric, and this run carried none. A run
    #: stopped by a guard before usage arrived lands here, as does one on a
    #: transport that never asked.
    UNREPORTED = "unreported"
    #: No such field exists on this runtime, so no run of it will ever carry
    #: one. ``anthropic-api`` is the established case: ``anthropic`` 0.97.0's
    #: ``Usage`` carries ``input``, ``output``, the two cache counts,
    #: ``server_tool_use`` and ``service_tier``, and no details object of any
    #: kind (read 2026-09-22).
    UNAVAILABLE = "unavailable"


#: Where each runtime's reasoning-token count comes from, in the vendor's own
#: field name. Read off the installed SDKs and, for the two subscription
#: runtimes, off real traffic on 2026-09-22 -- the dates and the evidence are in
#: :attr:`~modelpass.types.TokenUsage.reasoning_output_tokens`, which is the one
#: place the subset convention is argued.
#:
#: A runtime absent from this mapping reports :attr:`ReasoningMetric.UNAVAILABLE`.
#: As with :data:`RUNTIME_EFFORTS`, that is the narrow checkable claim -- no
#: metric is established here -- and not a claim about what the vendor ships.
REASONING_METRIC_FIELDS: dict[Runtime, str] = {
    # Two carriers, and neither is in the SDK's typed surface: the CLI sends
    # ``thinkingTokens`` on ``modelUsage`` as well, and ``ModelUsage`` (a
    # TypedDict, passed through verbatim) declares no such key. A types-only
    # read of this runtime is a floor, not a fact -- which is why this cell
    # rests on a live drive.
    Runtime.ANTHROPIC_SDK: "usage.output_tokens_details.thinking_tokens",
    Runtime.OPENAI_SDK: "tokenUsage.last.reasoningOutputTokens",
    Runtime.OPENAI_API: "usage.output_tokens_details.reasoning_tokens",
    Runtime.OPENAI_COMPATIBLE: "usage.completion_tokens_details.reasoning_tokens",
    # A peer of ``candidates_token_count`` on the wire, folded into
    # ``output_tokens`` by the adapter before it reaches ``TokenUsage``.
    Runtime.GOOGLE_API: "usage_metadata.thoughts_token_count",
}


def reasoning_metric(runtime: Runtime, value: int | None) -> ReasoningMetric:
    """Which of the three answers this run's count is.

    Takes the value rather than reading it off a usage object, so the terminal
    stamp and the run log cannot disagree about a run they both describe.
    """
    if value is not None:
        return ReasoningMetric.REPORTED
    if runtime in REASONING_METRIC_FIELDS:
        return ReasoningMetric.UNREPORTED
    return ReasoningMetric.UNAVAILABLE


class ReasoningDisposition(StrEnum):
    """How a runtime meets a stated effort.

    Two members, because the third outcome is a refusal and a refusal is an
    exception rather than a value. A plan that exists has been met.
    """

    #: The runtime has this exact rung. What the caller asked for is what goes
    #: on the wire.
    EXACT = "exact"
    #: The runtime has an effort control but not this rung, and modelpass moved
    #: to the nearest one it does have. Always reported, never silent -- the
    #: vendor already does the silent version (``claude-agent-sdk`` documents
    #: ``xhigh`` as "Opus 4.7 only; falls back to ``high`` on other models"),
    #: and a caller who cannot see the substitution cannot cost it.
    ADJUSTED = "adjusted"


@dataclass(frozen=True, slots=True)
class ReasoningPlan:
    """What was asked, what the runtime will actually be told, and why.

    Shaped after :class:`~modelpass.prompt_cache.PromptCachePlan` and
    :class:`~modelpass.sampling_rules.SamplingPlan`: requested beside applied
    plus a sentence, so the interesting case -- where the two differ -- is
    readable rather than inferred.
    """

    #: The house rung the caller stated.
    requested: Effort
    #: The house rung actually honoured. Differs from :attr:`requested` only
    #: when :attr:`disposition` is ``ADJUSTED``.
    applied: Effort
    #: What goes on the wire, in the runtime's own spelling. This is the value
    #: the owner ask wanted reported back.
    runtime_value: str
    disposition: ReasoningDisposition
    #: One sentence, for a receipt or a listing. Always present.
    note: str
    #: Which runtime option carries :attr:`runtime_value`:
    #: :data:`OPTION_EFFORT` for a rung, :data:`OPTION_THINKING` for ``none``
    #: on a runtime in :data:`THINKING_OFF`. Last, with a default, so no
    #: existing construction moves (2026-09-23).
    option: str = OPTION_EFFORT

    @property
    def adjusted(self) -> bool:
        """Whether modelpass had to move off the requested rung."""
        return self.disposition is ReasoningDisposition.ADJUSTED

    @property
    def wire_value(self) -> str:
        """What a receipt says was sent: the effort in the runtime's spelling,
        or ``thinking=<value>`` when thinking was switched instead, so a bare
        ``disabled`` is never mistaken for an effort level."""
        if self.option == OPTION_EFFORT:
            return self.runtime_value
        return f"{self.option}={self.runtime_value}"


def plan_reasoning(
    value: str | Effort,
    runtime: Runtime,
    *,
    name: str,
    registry: CapabilityRegistry | None = None,
) -> ReasoningPlan:
    """Resolve a stated effort against a runtime, or refuse it.

    ``name`` is the connection's, and it is required rather than optional
    because every refusal in :mod:`modelpass.connections` names the connection
    it is about; one that does not is a message somebody has to go looking
    through a config file to act on.

    Raises :class:`~modelpass.errors.InvalidConnection` -- the existing class
    for "this connection says something it cannot say".
    """
    registry = registry or DEFAULT_REGISTRY
    effort = _coerce(value, name=name)

    if registry.support(runtime, Capability.REASONING_EFFORT) is not Support.SUPPORTED:
        raise InvalidConnection(
            f"connection {name!r}: reasoning effort is not established on "
            f"{runtime.value} -- reasoning_effort reads "
            f"{registry.support(runtime, Capability.REASONING_EFFORT).value} on "
            "that row. That is not a claim the vendor has no such control; it "
            "is that modelpass carries none to this runtime, so stating one "
            "here would be a setting that does nothing. Omit the key"
        )

    available = RUNTIME_EFFORTS.get(runtime) or {}
    if not available:  # pragma: no cover -- guarded by the capability cell
        raise InvalidConnection(
            f"connection {name!r}: {runtime.value} has no effort vocabulary in "
            "modelpass"
        )

    if effort is Effort.NONE and runtime in THINKING_OFF:
        switch = THINKING_OFF[runtime]
        return ReasoningPlan(
            requested=effort,
            applied=effort,
            runtime_value=switch,
            disposition=ReasoningDisposition.EXACT,
            option=OPTION_THINKING,
            note=(
                f"reasoning effort 'none' goes to {runtime.value} as "
                f"thinking={{'type': {switch!r}}}, not as an effort level: "
                "this runtime's effort ladder starts at 'low', and thinking is "
                "switched off by its own option"
            ),
        )

    if effort in available:
        return ReasoningPlan(
            requested=effort,
            applied=effort,
            runtime_value=available[effort],
            disposition=ReasoningDisposition.EXACT,
            note=(
                f"reasoning effort {effort.value!r} goes to {runtime.value} as "
                f"{available[effort]!r}"
            ),
        )

    nearest = min(available, key=lambda e: (abs(e.rank - effort.rank), e.rank))
    direction = "up" if nearest.rank > effort.rank else "down"
    offered = ", ".join(repr(e.value) for e in sorted(available, key=lambda e: e.rank))
    return ReasoningPlan(
        requested=effort,
        applied=nearest,
        runtime_value=available[nearest],
        disposition=ReasoningDisposition.ADJUSTED,
        note=(
            f"reasoning effort {effort.value!r} has no equivalent on "
            f"{runtime.value}, whose ladder is {offered}; moved {direction} to "
            f"{nearest.value!r}, which goes on the wire as "
            f"{available[nearest]!r}. modelpass reports the move rather than "
            "making it quietly, because a substitution a caller cannot see is "
            "one they cannot cost"
        ),
    )


def _coerce(value: str | Effort, *, name: str) -> Effort:
    """A house rung, or a refusal that says which axis the word belongs to."""
    if isinstance(value, Effort):
        return value
    if not isinstance(value, str) or not value:
        raise InvalidConnection(
            f"connection {name!r}: reasoning must be one of: "
            f"{_ladder_text()}. Omit the key to state nothing"
        )

    lowered = value.strip().lower()
    other = _OTHER_AXIS.get(lowered)
    if other is not None:
        raise InvalidConnection(f"connection {name!r}: {other}")
    try:
        return Effort(lowered)
    except ValueError:
        raise InvalidConnection(
            f"connection {name!r}: reasoning must be one of: {_ladder_text()}, "
            f"got {value!r}"
        ) from None


def _ladder_text() -> str:
    return ", ".join(repr(e.value) for e in EFFORT_LADDER)


def stated_reasoning(connection: object) -> ReasoningPlan | None:
    """The connection's standing effort, resolved, or ``None`` if it stated none.

    The lookup the agent adapters need, in one place. Duck-typed on purpose:
    :mod:`modelpass.connections` imports this module, so importing
    :class:`~modelpass.connections.Connection` back would be a cycle.

    Only the runtimes ``sampling_rules`` cannot reach use this. Everywhere else
    the connection's level is merged into ``Sampling.reasoning_effort`` before
    the request is built, and the existing per-model routing carries it.

    Never raises. Anything invalid was refused when the connection was built, so
    a failure here would mean a connection that should not exist -- and an
    adapter mid-run is the worst place to discover that.
    """
    stated = getattr(connection, "reasoning", None)
    if stated is None:
        return None
    try:
        return plan_reasoning(
            stated,
            connection.runtime,
            name=str(getattr(connection, "name", "?")),
        )
    except Exception:  # pragma: no cover - defensive: refused at build
        return None
