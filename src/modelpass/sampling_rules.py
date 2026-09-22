"""Which sampling controls a runtime *and a model* will actually take (R5).

One table, one function, no vendor imports. ``rules_for(runtime, model)``
answers "what would happen to a :class:`~modelpass.types.Sampling` request here"
and :func:`plan_sampling` carries it out, producing the applied-vs-requested
report the receipt and the run log both carry.

**Why this file exists rather than an ``if`` in each adapter.** R5 says the
table is "keyed by runtime and refined by model", and the three shipped
consumers had each grown their own copy of that refinement, in three different
shapes, none of which could be read by the others:

* **A downstream agent host** forces ``temperature=1.0`` on every ``gpt-5*``
  model and logs a warning for ``top_p`` / ``top_k``; keeps one of temperature
  or top_p on ``claude-sonnet-4-*`` and ``claude-haiku-4-*``; and re-states both
  as a form-validation message in its settings page. Three statements of one
  rule, two of them in the UI layer (read 2026-09-13).
* **A RAG evaluation harness** turns one effort dial into a per-provider
  parameter *plus the constraints that ride along*, and records "requested but
  not applied" rather than dropping it -- the design this module follows, and
  the anchored family matching below is lifted from it outright, including its
  reason (a substring match once priced ``gpt-4o-mini`` at the ``gpt-4o`` rate).
* **A desktop agent app** withholds ``temperature`` and ``max_tokens`` from its
  subscription chat class entirely because the leaf warned about any value it
  was given -- an app routing *around* the library to avoid being shouted at,
  which is the failure this module's honesty report is meant to end.

**A row is evidence, not a guess.** Every entry carries a dated ``source`` note
saying what was read, and ``verified`` says whether modelpass's own half has been
driven. ``verified=False`` rows are seeded from consumer code and installed SDK
signatures and are *still* better than nothing -- they say what modelpass
believes and where the belief came from -- but they are not a capability cell,
and the cells for those runtimes stay ``unverified`` until their adapter ticket
(1.10, 1.11) drives them.

**An unknown model gets the runtime's defaults and is told so.** Never a guessed
family: guessing that ``gpt-4.2`` is a reasoning model sends a parameter the
vendor rejects and the call stops working, while guessing the other way costs a
line of disclosure. The note is the cost; the refusal to guess is the point.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .runtimes import Runtime
from .types import SAMPLING_FIELDS, Sampling

__all__ = [
    "SAMPLING_RULES_READ",
    "SamplingPlan",
    "SamplingRules",
    "plan_sampling",
    "rules_for",
]

#: The day every row below was read off consumer code and installed SDK types.
#: Data rather than prose because the notes are evidence, and evidence names its
#: source -- the same reason ``anthropic_api.SDK_VERSION_READ`` is a constant.
SAMPLING_RULES_READ = "2026-09-13"

#: Separators a model family token may be followed by. Copied from the RAG
#: evaluation harness (read 2026-09-13) with its reason: a family matched by
#: ``in`` makes ``gpt-4o-mini`` a ``gpt-4o``, and the same defect here would
#: make it a reasoning model and break the call.
_SEPARATORS = ("-", "@", ":", ".", "_")


def _matches(model: str, families: tuple[str, ...]) -> str:
    """The family token ``model`` belongs to, or ``""``. Anchored, never ``in``.

    **The longest match wins, not the first.** Family tokens nest -- ``gpt-5``
    against ``gpt-5-pro``, ``claude-sonnet-4`` against ``claude-sonnet-4-6`` --
    and first-match made the answer depend on declaration order, so a more
    specific family silently inherited a more general one's rules unless
    somebody remembered to declare it first. That is the wrong rules applied to
    a real model with nothing said about it, and the table is the last place
    that should rely on being read in the right order.

    Verified 2026-09-21: every family token in the table already resolves to
    itself under this rule, so the tables were correct by ordering discipline
    and nothing moves. What changes is that the discipline is no longer load
    bearing.
    """
    name = str(model or "").strip().lower()
    best = ""
    for token in families:
        if name == token:
            return token
        if name.startswith(token) and name[len(token) : len(token) + 1] in _SEPARATORS:
            if len(token) > len(best):
                best = token
    return best


@dataclass(frozen=True, slots=True)
class SamplingRules:
    """What one ``(runtime, model)`` pair does with a sampling request.

    Pure data, and every field answers a question some consumer's code was
    already answering privately:

    ``accepted``
        Which of :data:`~modelpass.types.SAMPLING_FIELDS` reach the vendor at
        all. Everything else is dropped and named.
    ``one_of``
        Groups where the vendor takes *one* member. Ordered: the first member
        present wins and the rest are dropped with a note.
    ``forced``
        Fields the model accepts exactly one value for. GPT-5's temperature is
        the worked example and the reason R5 says "per model".
    ``ceilings``
        The top of a field's range on this runtime, where it is narrower than
        the request type's own validation. Anthropic's temperature stops at 1.0
        while OpenAI's goes to 2.0, so the request type cannot enforce either.
    ``required`` / ``defaults``
        Fields the wire protocol demands. ``anthropic-api`` has exactly one:
        ``max_tokens`` is not optional on the Messages API, so a caller who says
        nothing still gets a value and the report still says which.
    ``reasoning_parameter``
        What ``reasoning_effort`` becomes on the wire, or ``None`` for a model
        that cannot reason on request.
    ``reasoning_removes``
        Fields a *thinking* request takes off the call. Anthropic's adaptive
        families reject sampling parameters outright when thinking is on, which
        is why this is a per-family fact and not a runtime one.
    ``efforts``
        Which reasoning rungs **this model** has, where that is known. Empty
        means not known at this granularity, and the runtime's own ladder
        (:data:`~modelpass.reasoning.RUNTIME_EFFORTS`) stands -- which is the
        honest default, because a model nobody has recorded is not a model
        with no rungs.

        A static floor on purpose, and replaceable on purpose. Two vendors
        publish this per model -- ``anthropic``'s ``EffortCapability`` and
        Codex's ``supportedReasoningEfforts`` -- so the better answer is to
        *ask*, and the point of putting the question behind
        :func:`model_efforts` is that swapping the source does not move any
        caller. Until then this is a lookup somebody updates when a model
        lands, and an unrecorded model degrades to the runtime ladder with a
        note rather than to a guess.
    """

    runtime: Runtime
    model: str = ""
    family: str = ""
    model_known: bool = True
    has_model_rules: bool = False
    accepted: frozenset[str] = frozenset()
    one_of: tuple[tuple[str, ...], ...] = ()
    forced: Mapping[str, Any] = field(default_factory=dict)
    ceilings: Mapping[str, float] = field(default_factory=dict)
    required: frozenset[str] = frozenset()
    defaults: Mapping[str, Any] = field(default_factory=dict)
    reasoning_parameter: str | None = None
    reasoning_removes: frozenset[str] = frozenset()
    efforts: frozenset[str] = frozenset()
    verified: bool = False
    source: str = ""

    @property
    def who(self) -> str:
        """How a note names the thing that refused: the family, else the runtime."""
        return self.family or self.runtime.value

    def accepts(self, field_name: str) -> bool:
        return field_name in self.accepted

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form (D11)."""
        return {
            "runtime": self.runtime.value,
            "model": self.model,
            "family": self.family,
            "model_known": self.model_known,
            "accepted": sorted(self.accepted),
            "one_of": [list(group) for group in self.one_of],
            "forced": dict(self.forced),
            "ceilings": dict(self.ceilings),
            "required": sorted(self.required),
            "defaults": dict(self.defaults),
            "reasoning_parameter": self.reasoning_parameter,
            "reasoning_removes": sorted(self.reasoning_removes),
            "verified": self.verified,
            "source": self.source,
        }


# --- the table -------------------------------------------------------------------
#
# Read as: one ``SamplingRules`` per runtime holding that runtime's defaults,
# plus an ordered list of ``(family_token, overrides)`` refinements. A model
# matching no family gets the defaults and ``model_known=False``.

_ALL = frozenset(SAMPLING_FIELDS)

#: What a thinking request takes off an Anthropic adaptive-thinking call.
_ADAPTIVE_REMOVES = frozenset({"temperature", "top_p", "top_k"})

#: Agent runtimes take no *sampling* at all, and this is a *checked* absence
#: rather than an unexamined one: ``grep -n 'temperature|top_p|top_k|max_tokens'``
#: over ``adapters/anthropic.py``, ``adapters/openai.py`` and
#: ``adapters/codex_appserver.py`` returns nothing (2026-09-13), and neither CLI
#: takes such a flag. The two agent runtimes' capability cells moved to
#: ``unsupported`` on this evidence in ticket 1.7.
#:
#: **Amended 2026-09-22, and the amendment is the point of dating these.** The
#: sentence below once said reasoning effort was unreachable here too, and a
#: consumer read it, believed it -- correctly, on the date it carried -- and
#: concluded they needed an API key to vary effort on a Claude subscription.
#: They did not: ``claude-agent-sdk`` 0.2.148 has ``ClaudeAgentOptions.effort``
#: and Codex has ``effort`` on ``TurnStartParams``, both wired on 2026-09-21,
#: both reached from the connection's ``reasoning`` key rather than from
#: ``Sampling``. The citation did its job; the fact under it moved and this
#: string did not move with it.
_AGENT_SOURCE = (
    "the two agent runtimes drive a coding-agent CLI, not a completions "
    "endpoint: no sampling parameter exists anywhere in adapters/anthropic.py, "
    "adapters/openai.py or adapters/codex_appserver.py, and neither CLI accepts "
    f"one (grep, {SAMPLING_RULES_READ}). A Sampling on these runtimes is not an "
    "error -- every field is dropped and named on the receipt. "
    "**Reasoning effort is the exception and is NOT dropped**: it is reachable "
    "here, just not through Sampling. Set the connection's 'reasoning' key and "
    "it travels as ClaudeAgentOptions.effort or TurnStartParams.effort "
    "(claude-agent-sdk 0.2.148, codex 0.154.0, wired 2026-09-21); the receipt "
    "reports it on reasoning_requested / reasoning_applied / reasoning_value"
)

#: The same absence, one degree less certain: no adapter for these two exists,
#: so nothing has been driven and the capability cells stay ``unverified``. The
#: rules row still says "nothing accepted", which is the safe direction: it
#: drops a field and reports it rather than sending one nobody has checked.
_UNBUILT_SOURCE = (
    "no adapter exists for this runtime yet, so nothing about it has been "
    "driven. The row accepts nothing, which reports a dropped field rather than "
    "sending an unchecked one; its capability cells stay unverified"
)

_ANTHROPIC_API_SOURCE = (
    "anthropic 0.97.0 Messages.create takes temperature, top_p, top_k and "
    "max_tokens, and output_config.effort is "
    "Literal['low','medium','high','xhigh','max'] "
    f"(read from the installed SDK's typed surface, {SAMPLING_RULES_READ}); "
    "modelpass's half is tests/test_adapter_anthropic_api.py section 8"
)

_OPENAI_API_SOURCE = (
    "openai 2.32.0 responses.create takes temperature, top_p and "
    "max_output_tokens, and shared.ReasoningEffort is "
    "Literal['none','minimal','low','medium','high','xhigh'] behind "
    "reasoning.effort -- and there is no top_k anywhere in it, nor in "
    "chat.completions.create (read from the installed SDK's typed surface, "
    f"{SAMPLING_RULES_READ}). Ticket 1.9 chose the Responses API and drove this "
    "row: modelpass's half is tests/test_adapter_openai_api.py section 9"
)

_GOOGLE_API_SOURCE = (
    "google-genai 1.73.1's GenerateContentConfig carries temperature, top_p, "
    "top_k and max_output_tokens as fields of its own, and ThinkingConfig "
    "carries thinking_level, typed ThinkingLevel = "
    "MINIMAL | LOW | MEDIUM | HIGH -- which is where modelpass's three-position "
    "reasoning dial lands, word for word (read from the installed SDK's typed "
    f"surface, {SAMPLING_RULES_READ}). **top_k is accepted here and on no other "
    "API runtime**, which is the seeded half of this row holding up: "
    "a downstream agent host passes temperature, top_p, top_k and max_tokens "
    "through to this vendor unconditionally, with no per-model rule. Ticket 1.11 "
    "drove it: modelpass's half is tests/test_adapter_google_api.py section 9. "
    "**No thinking_budget** -- that field is a token count whose allowed range "
    "the SDK documents as model dependent, and ticket 1.7 refused to invent one "
    "for Anthropic for the same reason"
)

_TABLE: dict[Runtime, tuple[SamplingRules, tuple[tuple[str, dict[str, Any]], ...]]] = {
    # --- the one driven row ------------------------------------------------------
    Runtime.ANTHROPIC_API: (
        SamplingRules(
            runtime=Runtime.ANTHROPIC_API,
            has_model_rules=True,
            accepted=_ALL,
            ceilings={"temperature": 1.0},
            required=frozenset({"max_output_tokens"}),
            defaults={"max_output_tokens": 4096},
            reasoning_parameter="output_config.effort",
            verified=True,
            source=_ANTHROPIC_API_SOURCE,
        ),
        (
            # **Longest token first.** ``_matches`` takes the first family that
            # matches, and ``claude-sonnet-4-6`` is under both rules below, so an
            # order that put the generic ``claude-sonnet-4`` first would hide the
            # adaptive one. This is the ordering bug the RAG evaluation
            # harness's anchored matching prevents at the token level, one level
            # up.
            #
            # Adaptive thinking: the request is ``{"type": "adaptive"}`` with no
            # budget and no sampling parameters at all. From the RAG evaluation
            # harness's adaptive-family list (read 2026-09-13), whose comment
            # states that ``budget_tokens`` is rejected and temperature/top_p/top_k
            # are removed on these models. Sampling is accepted here when no
            # effort is asked for; asking for effort removes it, and says so.
            (
                "claude-sonnet-4-6",
                {
                    "one_of": (("temperature", "top_p"),),
                    "reasoning_removes": _ADAPTIVE_REMOVES,
                },
            ),
            *(
                (token, {"reasoning_removes": _ADAPTIVE_REMOVES})
                for token in (
                    "claude-fable-5",
                    "claude-mythos-5",
                    "claude-opus-5",
                    "claude-sonnet-5",
                    "claude-opus-4-8",
                    "claude-opus-4-7",
                    "claude-opus-4-6",
                )
            ),
            # One of temperature or top_p, never both. Read off a downstream
            # agent host's model factory and restated by its settings form,
            # which refuses the pair in the UI -- two places in one app, which is
            # one of the reasons this table exists. Deliberately *not* widened to
            # ``claude-opus-4-*``: that host's rule names sonnet and haiku, this
            # table copies evidence rather than extrapolating from it, and an opus 4.x
            # outside the adaptive list gets the defaults and the
            # "model rules unknown" note instead.
            (
                "claude-sonnet-4",
                {"one_of": (("temperature", "top_p"),), "family": "claude-sonnet-4-x"},
            ),
            (
                "claude-haiku-4",
                {"one_of": (("temperature", "top_p"),), "family": "claude-haiku-4-x"},
            ),
        ),
    ),
    # --- the second driven row (ticket 1.9) --------------------------------------
    Runtime.OPENAI_API: (
        SamplingRules(
            runtime=Runtime.OPENAI_API,
            has_model_rules=True,
            # No top_k: it is absent from both of the SDK's create signatures,
            # which is a fact about the request schema rather than a drive.
            accepted=frozenset({"temperature", "top_p", "max_output_tokens"}),
            ceilings={"temperature": 2.0},
            # No ``required`` and no ``defaults``, unlike anthropic-api:
            # max_output_tokens is optional on the Responses API, so a caller who
            # names no ceiling gets the model's own rather than a number modelpass
            # invented. The empty pair is the decision, not an omission.
            #
            # No runtime-level ``reasoning_parameter`` either: an effort dial is
            # a *model* fact here, not a runtime one -- gpt-4o cannot reason on
            # request and naming a parameter for it would say it could. The
            # o-series and gpt-5 rows below set it to "reasoning.effort", which
            # is the Responses spelling ticket 1.9 settled on.
            verified=True,
            source=_OPENAI_API_SOURCE,
        ),
        (
            # The o-series refuses every sampling control and takes an effort
            # dial instead. A downstream agent host warns and drops all three;
            # a RAG evaluation harness lists o1/o3/o4 as reasoning families and
            # sends no temperature with them.
            *(
                (
                    token,
                    {
                        "accepted": frozenset({"max_output_tokens", "reasoning_effort"}),
                        "reasoning_parameter": "reasoning.effort",
                    },
                )
                for token in ("o1", "o3", "o4")
            ),
            # GPT-5 accepts temperature and only the default value of it. A
            # downstream agent host sets temperature=1.0 unconditionally and
            # warns that top_p and top_k are ignored.
            (
                "gpt-5",
                {
                    "accepted": frozenset(
                        {"temperature", "max_output_tokens", "reasoning_effort"}
                    ),
                    "forced": {"temperature": 1.0},
                    "reasoning_parameter": "reasoning.effort",
                },
            ),
            # gpt-5-pro "defaults to (and only supports) high reasoning effort"
            # -- openai 2.32.0, the docstring on Reasoning.effort, read
            # 2026-09-21. The one model-specific rung set the installed SDKs
            # state outright, and the reason `efforts` exists: without it a
            # caller asking this model for 'low' is refused by the vendor after
            # a round trip instead of being moved and told here.
            (
                "gpt-5-pro",
                {
                    "accepted": frozenset(
                        {"temperature", "max_output_tokens", "reasoning_effort"}
                    ),
                    "forced": {"temperature": 1.0},
                    "reasoning_parameter": "reasoning.effort",
                    "efforts": frozenset({"high"}),
                },
            ),
            # The ordinary chat models: sampling yes, effort no. Named as
            # families rather than inferred, so an unlisted model gets the
            # runtime defaults and a note instead of a guessed capability.
            *(
                (token, {})
                for token in ("gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo")
            ),
        ),
    ),
    Runtime.OPENAI_COMPATIBLE: (
        SamplingRules(
            runtime=Runtime.OPENAI_COMPATIBLE,
            # Model rules exist in principle and are knowable in practice by
            # nobody: the endpoint is whatever the operator pointed it at.
            # Every model is therefore unknown, and every call carries the
            # "runtime defaults applied" note -- which is the honest answer for
            # an Ollama box serving a model modelpass has never heard of.
            has_model_rules=True,
            accepted=frozenset({"temperature", "top_p", "max_output_tokens"}),
            ceilings={"temperature": 2.0},
            verified=True,
            source=(
                "the OpenAI-compatible dialect carries temperature, top_p and a "
                "max-tokens field, and the adapter sends all three "
                f"(ticket 1.10, {SAMPLING_RULES_READ}). **top_k was removed from "
                "this row in that ticket**: it is an extension, it is in neither "
                "of openai 2.32.0's create signatures, and the only way to send "
                "it is extra_body -- which would put an unrecognised key in the "
                "request body of every server that does not take one. Some do "
                "(Ollama through its own options block, vLLM at the top level, "
                "and they disagree about where it goes), which is exactly why "
                "modelpass will not guess: a dropped field is named on the "
                "receipt, and a guessed one is a call that fails or, worse, "
                "silently samples differently from what was asked. Which model "
                "is behind the base URL is still the operator's business, so no "
                "model rule here could be anything but a guess"
            ),
        ),
        (),
    ),
    # --- the fourth driven row (ticket 1.11) -------------------------------------
    Runtime.GOOGLE_API: (
        SamplingRules(
            runtime=Runtime.GOOGLE_API,
            # Model rules exist and modelpass has driven none of them, so every
            # model is unknown and every call that asks for sampling carries the
            # "runtime defaults applied" note. That note is load-bearing on this
            # row: reasoning_effort is accepted at the runtime level because the
            # field is a runtime field, and whether a *given* Gemini model
            # thinks is exactly what the note says nobody here has checked.
            has_model_rules=True,
            accepted=frozenset(
                {"temperature", "top_p", "top_k", "max_output_tokens", "reasoning_effort"}
            ),
            ceilings={"temperature": 2.0},
            # No required field and no default: max_output_tokens is optional
            # here, as on openai-api, so a caller who names no ceiling gets the
            # model's own rather than a number modelpass invented.
            reasoning_parameter="thinking_config.thinking_level",
            verified=True,
            source=_GOOGLE_API_SOURCE,
        ),
        (),
    ),
    # --- the agent runtimes ------------------------------------------------------
    Runtime.ANTHROPIC_SDK: (
        SamplingRules(runtime=Runtime.ANTHROPIC_SDK, verified=True, source=_AGENT_SOURCE),
        (),
    ),
    Runtime.OPENAI_SDK: (
        SamplingRules(runtime=Runtime.OPENAI_SDK, verified=True, source=_AGENT_SOURCE),
        (),
    ),
    Runtime.GOOGLE_CLI: (
        SamplingRules(runtime=Runtime.GOOGLE_CLI, source=_UNBUILT_SOURCE),
        (),
    ),
    Runtime.GOOGLE_SDK: (
        SamplingRules(runtime=Runtime.GOOGLE_SDK, source=_UNBUILT_SOURCE),
        (),
    ),
}


def rules_for(runtime: Runtime | str, model: str | None = None) -> SamplingRules:
    """The rules for one runtime and model. Never raises, never guesses.

    A model matching no known family comes back with the runtime's defaults and
    ``model_known=False``, which :func:`plan_sampling` turns into the note R5
    asks for. A runtime with no model refinements at all -- the agent runtimes --
    reports ``model_known=True``, because there is nothing there a model could
    refine and a note saying otherwise would be noise on every call.
    """
    runtime = Runtime(runtime)
    base, refinements = _TABLE[runtime]
    name = str(model or "").strip()
    family = _matches(name, tuple(token for token, _ in refinements))
    overrides: dict[str, Any] = {}
    if family:
        for token, values in refinements:
            if token == family:
                overrides = dict(values)
                break
    return replace(
        base,
        model=name,
        model_known=bool(family) or not base.has_model_rules,
        family=overrides.pop("family", family),
        **overrides,
    )


def model_efforts(runtime: Runtime | str, model: str | None = None) -> tuple[str, ...]:
    """Which reasoning rungs this model has, ascending.

    **The interface, so that the source can change without a caller moving.**
    Today it answers from a static lookup: the per-model ``efforts`` set where
    one has been recorded, otherwise the runtime's ladder. Tomorrow it can
    answer from the vendor -- ``anthropic``'s ``EffortCapability`` and Codex's
    ``supportedReasoningEfforts`` both publish it per model -- and nothing that
    calls this has to know which happened.

    That is the whole reason the question is a function rather than a dict
    lookup at each call site. A table that reads as authoritative and goes stale
    silently is the thing ``maxInputTokens`` ships none of; a table behind a
    resolver is a *floor* that something better can replace, and an unrecorded
    model degrades to the runtime ladder rather than to a guess about a model
    nobody has looked at.

    Returns ``()`` for a runtime with no established effort control at all,
    which is a different answer from "this model has no rungs" and is why the
    empty tuple is not a refusal here -- the refusal already happened when the
    connection was built.
    """
    from .reasoning import EFFORT_LADDER, RUNTIME_EFFORTS

    runtime = Runtime(runtime)
    ladder = RUNTIME_EFFORTS.get(runtime)
    if not ladder:
        return ()
    recorded = rules_for(runtime, model).efforts
    order = {e.value: e.rank for e in EFFORT_LADDER}
    available = {e.value for e in ladder}
    if recorded:
        # A recorded model narrows the runtime's ladder; it never widens it,
        # because a rung the runtime has no spelling for cannot be sent.
        available &= set(recorded)
    return tuple(sorted(available, key=lambda v: order[v]))


# --- the plan --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    """What one sampling request becomes here, and what it cost on the way.

    ``requested`` is the caller's own words; ``applied`` is what modelpass will
    send, in modelpass's field names rather than any vendor's -- translating
    ``max_output_tokens`` into ``max_tokens`` is the adapter's job, and a report
    written in one vendor's spelling would not be readable beside another's.
    ``notes`` explains every difference between the two, one sentence per field,
    and is empty exactly when the two agree.
    """

    requested: Mapping[str, Any] = field(default_factory=dict)
    applied: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    rules: SamplingRules | None = None

    @property
    def honoured(self) -> bool:
        """Whether everything asked for was sent unchanged."""
        return all(
            name in self.applied and self.applied[name] == value
            for name, value in self.requested.items()
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form (D11)."""
        return {
            "requested": dict(self.requested),
            "applied": dict(self.applied),
            "notes": list(self.notes),
        }


def _fmt(value: Any) -> str:
    """A number as a note should read it: ``0.2``, ``1.0``, ``128``."""
    if isinstance(value, bool) or not isinstance(value, float):
        return str(value)
    return repr(value)


def _runtime_efforts_map():
    from .reasoning import RUNTIME_EFFORTS

    return RUNTIME_EFFORTS


def plan_sampling(
    sampling: Sampling | None,
    runtime: Runtime | str,
    model: str | None = None,
    *,
    rules: SamplingRules | None = None,
) -> SamplingPlan:
    """Resolve one request against the rules. Pure: no vendor, no network.

    The order matters and is the order a vendor would apply it in: drop what the
    model does not accept, coerce what it accepts only one value of, clamp what
    is out of range, resolve the one-of groups, let a thinking request take away
    what it takes away, then fill in whatever the wire protocol demands and
    nobody supplied.

    ``sampling`` of ``None`` -- or one that asks for nothing -- still produces a
    plan, because a required field with a default is something the caller should
    be able to read back: an ``anthropic-api`` call nobody set a ceiling on is
    still a call with a 4096-token ceiling on it.
    """
    resolved = rules or rules_for(runtime, model)
    requested = sampling.requested() if sampling is not None else {}
    applied: dict[str, Any] = {}
    notes: list[str] = []
    who = resolved.who

    if requested and not resolved.model_known:
        notes.append(
            f"model rules unknown for {resolved.model or 'an unnamed model'!r} on "
            f"{resolved.runtime.value}; runtime defaults applied"
        )

    for name, value in requested.items():
        if not resolved.accepts(name):
            # "Not supported" is true of *this path* and false of the runtime
            # when the field is reasoning_effort: both agent runtimes take an
            # effort, just not through Sampling. A consumer reported reading the
            # bare note, concluding effort was unreachable on a subscription
            # connection, and planning to buy an API key they did not need
            # (2026-09-22). Point at the key that works.
            if name == "reasoning_effort" and _runtime_efforts_map().get(resolved.runtime):
                notes.append(
                    f"reasoning_effort is not a Sampling field on {who}; set the "
                    "connection's 'reasoning' key instead and it travels as this "
                    "runtime's own effort option. Dropped from Sampling here"
                )
            else:
                notes.append(f"{name} not supported on {who}, dropped")
            continue
        if name in resolved.forced:
            forced = resolved.forced[name]
            if value != forced:
                notes.append(
                    f"{name} {_fmt(value)} requested; {who} accepts only "
                    f"{_fmt(forced)}, sent {_fmt(forced)}"
                )
            applied[name] = forced
            continue
        ceiling = resolved.ceilings.get(name)
        if ceiling is not None and isinstance(value, (int, float)) and value > ceiling:
            notes.append(
                f"{name} {_fmt(value)} requested; {who} accepts at most "
                f"{_fmt(ceiling)}, sent {_fmt(ceiling)}"
            )
            applied[name] = ceiling
            continue
        applied[name] = value

    for group in resolved.one_of:
        present = [name for name in group if name in applied]
        if len(present) < 2:
            continue
        kept, dropped = present[0], present[1:]
        for name in dropped:
            del applied[name]
        listed = " and ".join(present)
        notes.append(
            f"{listed} were both requested; {who} accepts one of them, kept "
            f"{kept} and dropped {', '.join(dropped)}"
        )

    if "reasoning_effort" in applied:
        rungs = model_efforts(resolved.runtime, resolved.model)
        asked = str(applied["reasoning_effort"])
        if rungs and asked not in rungs:
            from .reasoning import EFFORT_LADDER

            order = {e.value: e.rank for e in EFFORT_LADDER}
            nearest = min(
                rungs, key=lambda r: (abs(order[r] - order[asked]), order[r])
            )
            applied["reasoning_effort"] = nearest
            way = "up" if order[nearest] > order[asked] else "down"
            notes.append(
                f"reasoning_effort {asked!r} is not one this model takes "
                f"({who} accepts {', '.join(repr(r) for r in rungs)}); moved "
                f"{way} to {nearest!r}. Reported rather than sent, because a "
                "rung the vendor refuses costs a round trip to discover and a "
                "rung it silently downgrades costs nothing to discover and is "
                "worse"
            )

    if "reasoning_effort" in applied and resolved.reasoning_removes:
        removed = [
            name
            for name in SAMPLING_FIELDS
            if name in resolved.reasoning_removes and name in applied
        ]
        for name in removed:
            del applied[name]
        if removed:
            notes.append(
                f"reasoning_effort {applied['reasoning_effort']!r} is adaptive "
                f"thinking on {who}, which takes no sampling parameters; "
                f"{', '.join(removed)} dropped"
            )

    for name in SAMPLING_FIELDS:
        if name in resolved.required and name not in applied:
            default = resolved.defaults.get(name)
            if default is None:
                continue
            applied[name] = default
            notes.append(
                f"{name} not requested; {resolved.runtime.value} requires one, "
                f"sent the modelpass default {_fmt(default)}"
            )

    ordered = {name: applied[name] for name in SAMPLING_FIELDS if name in applied}
    return SamplingPlan(
        requested=requested, applied=ordered, notes=tuple(notes), rules=resolved
    )
