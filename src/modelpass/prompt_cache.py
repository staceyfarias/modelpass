"""Stating an intent to cache prompts, and reporting what a runtime did with it.

The connection-level half of prompt caching, added 2026-09-21. The per-call half
already existed and is not re-done here: :class:`~modelpass.types.CacheControl`
on a :class:`~modelpass.types.TextBlock` is how a caller says *where the
cacheable prefix ends*, which is inherently per call because it is a statement
about one payload's shape. What no caller could say anywhere was the other
thing -- *"I want prompt caching on for this connection, at this lifetime"* --
which is a standing policy and belongs beside ``retry`` and ``timeoutSeconds``.

Both halves are needed and neither replaces the other. A breakpoint with no
policy is what the library had before: a marker whose fate a caller learned from
a receipt line, after the fact, per call. A policy with no breakpoint is
meaningful on its own -- on every runtime but one, a caller *cannot* place a
breakpoint and the cache happens anyway -- which is the case this module exists
to name.

**Three outcomes, and conflating them is the mistake this module is built to
avoid.**

* :attr:`PromptCacheDisposition.EXPLICIT` -- the runtime takes an instruction
  about caching, so the request is honoured in the strong sense: put
  ``cache_control`` on your content and it reaches the vendor. Today this is
  ``anthropic-api`` alone.
* :attr:`PromptCacheDisposition.AUTOMATIC` -- the runtime caches without being
  asked and offers no way to stop it, so the request is **already satisfied**.
  This is not an error and must never be reported as one. It is also not the
  same answer as ``EXPLICIT``: a caller who needs to know whether its
  breakpoints mean anything reads the difference here.
* Refused -- the runtime has no prompt caching that modelpass has established,
  or the requested lifetime is not one it accepts. There is nothing to honour
  and nothing already happening, so an :class:`~modelpass.errors.InvalidConnection`
  says so rather than the key being quietly ignored. A setting that does nothing
  and says nothing is the failure mode every disclosure in this library exists
  to end.

**Refused at construction, not at the call.** The precedent is
:class:`~modelpass.types.CacheControl`, which validates its own ``ttl`` in
``__post_init__`` for the stated reason that "hearing about it from a 400 several
seconds and one HTTP round trip later is strictly worse than hearing about it
from the constructor", and :class:`~modelpass.connections.Connection`, which
already refuses an ``authMode`` the runtime does not allow. Every input this
module needs -- the runtime, the value -- is on the connection, so there is
nothing to wait for.

**No house TTL vocabulary, and that is a finding rather than a design taste.**
The two runtimes that take a lifetime spell it in disjoint words: ``anthropic``
0.97.0 types it ``Literal['5m', '1h']`` and ``openai`` 2.32.0 types it
``Literal['in-memory', '24h']``. A single modelpass vocabulary would have had to
either translate between them -- inventing an equivalence nobody published -- or
pick one vendor's words and mis-describe the other. So :data:`PROMPT_CACHE_TTLS`
is per runtime, every entry is a value read off the SDK installed in this
repository on a named date, and a runtime with no entry accepts
:data:`VENDOR_DEFAULT` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .capabilities import DEFAULT_REGISTRY, Capability, CapabilityRegistry, Support
from .errors import InvalidConnection
from .runtimes import Runtime

__all__ = [
    "PROMPT_CACHE_TTLS",
    "VENDOR_DEFAULT",
    "ContinuityMechanism",
    "EffortCacheContinuity",
    "EffortContinuity",
    "PromptCacheDisposition",
    "PromptCachePlan",
    "effort_cache_continuity",
    "observed_effort_continuity",
    "plan_prompt_cache",
]

#: The word for "caching on, at whatever lifetime the vendor uses". It is a
#: named value rather than an empty string, a ``True`` or a ``0``, because it has
#: to be distinguishable from *both* an explicit lifetime *and* from the key
#: being absent, and those are three states rather than two. Absent means nobody
#: has said anything and nothing changes; ``"default"`` means somebody asked for
#: caching and declined to name a lifetime; ``"1h"`` means somebody named one.
#: The same unset-versus-zero discipline ``maxInputTokens`` uses, one type along.
VENDOR_DEFAULT = "default"

#: The cache lifetimes each runtime accepts, in the **vendor's own spelling**.
#:
#: A runtime absent from this mapping accepts :data:`VENDOR_DEFAULT` and nothing
#: else. That is not a claim that the vendor has no lifetime lever -- it is the
#: narrower and checkable claim that *modelpass carries no lifetime to it from
#: this key*, and the refusal message says exactly that rather than the broader
#: thing.
#:
#: Every value here was read off the SDK installed in this repository on
#: 2026-09-21, and the read is named beside it. Nothing here is a number
#: remembered from a vendor's documentation page, for the reason ``maxInputTokens``
#: has no table of model window sizes: a constant that reads as authoritative and
#: goes stale silently is worse than no constant.
PROMPT_CACHE_TTLS: dict[Runtime, tuple[str, ...]] = {
    # anthropic 0.97.0, anthropic/types/cache_control_ephemeral_param.py:
    # ``CacheControlEphemeralParam.ttl: Literal["5m", "1h"]``, read 2026-09-21.
    # The same two values modelpass.types.CACHE_TTLS already validates for a
    # per-call breakpoint, which is not a coincidence -- on this runtime the
    # connection-level statement and the per-call marker are talking about the
    # same vendor field.
    Runtime.ANTHROPIC_API: ("5m", "1h"),
    # openai 2.32.0, openai/types/responses/response_create_params.py:
    # ``prompt_cache_retention: Optional[Literal["in-memory", "24h"]]``, read
    # 2026-09-21. Its own docstring is also the evidence for this runtime's
    # CACHE_AUTOMATIC cell: "Set to `24h` to enable extended prompt caching" is
    # a retention policy for a cache the request never asked to create.
    Runtime.OPENAI_API: ("in-memory", "24h"),
}


class PromptCacheDisposition(StrEnum):
    """What a runtime does with a request to cache prompts.

    Two members, not three: the third outcome is a refusal, and a refusal is an
    exception rather than a value. A plan that exists has been met.
    """

    #: The runtime takes an instruction. The caller's ``cache_control``
    #: breakpoints reach the vendor, and *where* the prefix ends is the caller's
    #: to decide.
    EXPLICIT = "explicit"
    #: The runtime caches on its own and offers the caller no lever. The request
    #: was already satisfied before it was made. A caller reading this knows its
    #: breakpoints, if it has any, are decoration here.
    AUTOMATIC = "automatic"


@dataclass(frozen=True, slots=True)
class PromptCachePlan:
    """What was asked for, what the runtime does about it, and in one sentence why.

    Shaped after :class:`~modelpass.sampling_rules.SamplingPlan`: requested
    beside applied, plus a note, so that the interesting case -- where the two
    differ -- is readable rather than inferred.
    """

    #: What the connection stated: :data:`VENDOR_DEFAULT` or a vendor lifetime.
    requested: str
    #: How this runtime meets it.
    disposition: PromptCacheDisposition
    #: The lifetime modelpass will actually put on the wire, or ``None`` when it
    #: puts none and the vendor's own default stands. ``None`` here is a fact
    #: about modelpass, never a claim that the vendor has no default.
    ttl: str | None
    #: One sentence, for a receipt or a listing. Always present.
    note: str

    @property
    def already_satisfied(self) -> bool:
        """Whether the runtime was doing this anyway."""
        return self.disposition is PromptCacheDisposition.AUTOMATIC


def plan_prompt_cache(
    value: str,
    runtime: Runtime,
    *,
    name: str,
    registry: CapabilityRegistry | None = None,
) -> PromptCachePlan:
    """Resolve a ``promptCache`` statement against a runtime, or refuse it.

    ``name`` is the connection's, and it is required rather than optional
    because every refusal in :mod:`modelpass.connections` names the connection it
    is about; a message that does not is a message somebody has to go looking
    through a config file to act on.

    Raises :class:`~modelpass.errors.InvalidConnection` -- the existing class for
    "this connection says something it cannot say", already used for an
    ``authMode`` a runtime does not allow. No new error class and no new
    user-facing vocabulary were needed for this, which is the answer to the
    question of whether one should have been invented.
    """
    registry = registry or DEFAULT_REGISTRY
    if not isinstance(value, str) or not value:
        raise InvalidConnection(
            f"connection {name!r}: promptCache must be a string -- "
            f"{VENDOR_DEFAULT!r} for the vendor's own cache lifetime, or a "
            f"lifetime this runtime accepts. Omit the key to state nothing"
        )

    breakpoints = registry.support(runtime, Capability.CACHE_BREAKPOINTS)
    automatic = registry.support(runtime, Capability.CACHE_AUTOMATIC)

    if breakpoints is Support.SUPPORTED:
        disposition = PromptCacheDisposition.EXPLICIT
    elif automatic is Support.SUPPORTED:
        disposition = PromptCacheDisposition.AUTOMATIC
    else:
        # Both unverified is the same refusal as both unsupported, deliberately.
        # supports() has treated unverified as unusable since the table was
        # written, and honouring a request on a cell nobody has checked would be
        # the guess this library exists not to make.
        raise InvalidConnection(
            f"connection {name!r}: promptCache asks for prompt caching, and "
            f"modelpass has not established that {runtime.value} caches prompts "
            f"-- cache_breakpoints reads {breakpoints.value} and cache_automatic "
            f"reads {automatic.value} on that row. That is not a claim the vendor "
            "does no caching; it is that nothing here can honour the request or "
            "tell you it was already met. Omit the key"
        )

    accepted = PROMPT_CACHE_TTLS.get(runtime, ())
    if value != VENDOR_DEFAULT:
        if not accepted:
            raise InvalidConnection(
                f"connection {name!r}: promptCache on {runtime.value} accepts only "
                f"{VENDOR_DEFAULT!r}, got {value!r}. modelpass carries no cache "
                f"lifetime to {runtime.value} from this key, so naming one here "
                "would be a setting that does nothing"
            )
        if value not in accepted:
            allowed = ", ".join(repr(t) for t in accepted)
            raise InvalidConnection(
                f"connection {name!r}: promptCache on {runtime.value} must be one "
                f"of: {allowed} (or {VENDOR_DEFAULT!r} for the vendor's own cache "
                f"lifetime), got {value!r}"
            )

    ttl = None if value == VENDOR_DEFAULT else value
    return PromptCachePlan(
        requested=value,
        disposition=disposition,
        ttl=ttl,
        note=_note(runtime, disposition, ttl),
    )


def _note(
    runtime: Runtime, disposition: PromptCacheDisposition, ttl: str | None
) -> str:
    """The one sentence. Written to be read on a receipt, once, by a person."""
    if disposition is PromptCacheDisposition.EXPLICIT:
        where = (
            f"at a {ttl} lifetime"
            if ttl
            else "at the vendor's own cache lifetime"
        )
        return (
            f"prompt caching requested {where}: {runtime.value} takes an "
            "instruction, so cache_control breakpoints on this call's content "
            "reach the vendor. modelpass places none of its own and moves none "
            "of yours -- where the prefix ends is yours to say"
        )
    if ttl:
        return (
            f"prompt caching requested at a {ttl} lifetime: {runtime.value} "
            "caches prompts without being asked and takes no breakpoint, so the "
            f"request was already met and the {ttl} retention is the one thing "
            "here that is yours to set"
        )
    return (
        f"prompt caching requested at the vendor's own cache lifetime: "
        f"{runtime.value} caches prompts without being asked and offers no lever "
        "to stop it or to place a breakpoint, so the request was already met "
        "before it was made. Nothing is sent for it and nothing needs to be"
    )


# --- changing effort without losing the prefix -----------------------------------


class EffortCacheContinuity(StrEnum):
    """Whether changing reasoning effort mid-conversation *can* keep the cache.

    **Three questions, three types, and collapsing any two of them is the bug
    this vocabulary exists to prevent:**

    1. *Does this runtime take an effort setting at all?* --
       :data:`~modelpass.reasoning.RUNTIME_EFFORTS` and
       :attr:`~modelpass.capabilities.Capability.REASONING_EFFORT`.
    2. *Can changing that setting mid-conversation preserve the cached prefix?*
       -- this type. A statement about a vendor and a model, knowable before a
       run, true whether or not anybody ever changes effort.
    3. *Did this particular turn actually reuse the prefix?* -- token telemetry,
       knowable only afterwards, and answered by
       :func:`observed_effort_continuity` from counts the runtime reported.

    A model can accept five effort levels and still restart the cache on every
    change (1 without 2); a capability can be ``SUPPORTED`` and a given turn
    still miss cache for reasons that have nothing to do with effort (2 without
    3). This is the same split :class:`~modelpass.preflight.CacheEligibility`
    already draws between *can* and *did*, for the same reason: no honest answer
    to "did it hit" exists before the bytes are sent.

    **Conservative by construction.** :attr:`SUPPORTED` requires the vendor to
    document a cache-preserving effort change; :attr:`UNSUPPORTED` requires the
    vendor to document that changing it restarts the cache. Everything else is
    :attr:`UNKNOWN`, which is not a hedge -- it is the difference between a
    consumer designing a long cached session around a guarantee and one who
    knows to measure first.
    """

    #: The vendor documents a mechanism that changes effort and keeps the prefix.
    SUPPORTED = "supported"
    #: A mechanism exists in the vendor's own protocol or client, but no vendor
    #: statement establishes the cache behaviour and nobody has characterized it
    #: here. Worth trying behind a flag; never worth relying on.
    EXPERIMENTAL = "experimental"
    #: The vendor documents that changing effort invalidates the cached prefix.
    UNSUPPORTED = "unsupported"
    #: Nothing is established either way. The default, and the answer modelpass
    #: gives about any pair nobody has checked.
    UNKNOWN = "unknown"


class ContinuityMechanism(StrEnum):
    """*How* a cache-preserving effort change is expressed, where one exists.

    Named rather than left implicit because the three mechanisms are genuinely
    different protocol moves, and a consumer deciding whether to attempt one
    needs to know which. They are also not interchangeable: each is documented
    for its own vendor and model, and sending one shape to a runtime that
    expects another is a 400 rather than a graceful ignore.
    """

    #: Anthropic Messages API: a ``role: "system"`` message with empty
    #: ``content`` carrying ``output_config.effort``, placed in ``messages``
    #: before the turn it should apply to. Beta header
    #: ``mid-conversation-output-config-2026-07-01``.
    ANTHROPIC_PER_MESSAGE_EFFORT = "anthropic_per_message_effort"
    #: OpenAI Responses API: a ``{"type": "configuration_update", "reasoning":
    #: {"effort": ...}}`` input item, with the request-level ``reasoning.effort``
    #: left unchanged.
    OPENAI_CONFIGURATION_UPDATE = "openai_configuration_update"
    #: The same item shape reached through Codex's app-server rather than
    #: through the Responses API directly.
    CODEX_CONFIGURATION_UPDATE = "codex_configuration_update"


@dataclass(frozen=True, slots=True)
class EffortContinuity:
    """What is established about one ``(runtime, model)`` pair, and by whom.

    :attr:`reachable` is the field that keeps this honest, and it exists because
    of a failure this repository has already shipped once: on 2026-09-22 a
    consumer read a dated note saying effort was unreachable on a subscription,
    believed it, and planned to buy an API key they did not need. The inverse
    error is available here and would be worse -- a receipt reporting
    ``supported`` for a mechanism modelpass has no code path to send, read by a
    consumer who then designs a long cached session around an effort change that
    never happens. So the vendor's capability and modelpass's ability to drive
    it are two fields, never one.
    """

    #: What the vendor establishes about this pair.
    capability: EffortCacheContinuity = EffortCacheContinuity.UNKNOWN
    #: Which protocol move carries it, or ``None`` when there is none to name.
    mechanism: ContinuityMechanism | None = None
    #: **Whether modelpass can actually perform that move today.** ``False``
    #: with a ``SUPPORTED`` capability is a real and current state: the vendor
    #: ships the mechanism, and the installed SDK or this library has no way to
    #: express it. Never inferred from :attr:`capability`.
    reachable: bool = False
    #: One sentence naming the evidence and its date, for a reader checking the
    #: cell rather than trusting it. Always present.
    detail: str = ""

    @property
    def claimed(self) -> bool:
        """Whether modelpass makes any positive claim about this pair at all."""
        return self.capability in (
            EffortCacheContinuity.SUPPORTED,
            EffortCacheContinuity.EXPERIMENTAL,
        )


_SUPPORTED = EffortCacheContinuity.SUPPORTED
_EXPERIMENTAL = EffortCacheContinuity.EXPERIMENTAL
_UNSUPPORTED = EffortCacheContinuity.UNSUPPORTED
_UNKNOWN = EffortCacheContinuity.UNKNOWN

#: The Anthropic models whose documentation states that a per-message effort
#: change keeps the cache, read off the vendor's own effort page on 2026-09-22:
#: *"Claude Fable 5.1 also supports changing effort mid-conversation with a
#: per-message ``output_config``, which preserves the prompt cache"*, the same
#: sentence for Claude Opus 5, and *"On Claude Fable 5.1, Claude Mythos 5.1, and
#: Claude Opus 5, use a per-message effort change, which keeps the prompt
#: cache."*
_ANTHROPIC_PER_MESSAGE_MODELS: tuple[str, ...] = (
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5",
)

#: Per ``(runtime, model family)``, what is established. The family token is
#: matched the way :mod:`modelpass.sampling_rules` matches one -- longest
#: prefix on a separator boundary -- because a second matcher that disagreed
#: with that one about ``claude-opus-5`` versus ``claude-opus-5-1`` would be a
#: bug nobody finds until a model ships.
#:
#: **A pair absent from this table is :attr:`EffortCacheContinuity.UNKNOWN`**,
#: and that is the design: modelpass makes no cache-continuity claim about a
#: combination nobody has established. Unknown is better than silently implying
#: that changing effort keeps a prefix.
_CONTINUITY: dict[Runtime, tuple[tuple[str, EffortContinuity], ...]] = {}


#: The models Anthropic's effort page lists as supporting effort **and** which
#: are not among the three above, read off that page's ``supportedModels`` front
#: matter on 2026-09-22. These are the pairs the documented negative actually
#: covers: they take an effort setting, they do not take a per-message one, and
#: the vendor says what changing the top-level value does to the cache.
#:
#: A model on neither list gets :attr:`EffortCacheContinuity.UNKNOWN`, and this
#: list is why there is no catch-all row for this runtime. An unreleased
#: ``claude-opus-5-1`` is not covered by a negative written about today's
#: models, and answering ``unsupported`` for it would be the same manufactured
#: certainty as answering ``supported`` -- just pointing the other way.
_ANTHROPIC_NO_PER_MESSAGE_MODELS: tuple[str, ...] = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)


def _anthropic_api_rows() -> tuple[tuple[str, EffortContinuity], ...]:
    """The three documented models, and the ones the documented negative covers.

    Both halves come from the vendor rather than from inference. The positive is
    *"On Claude Fable 5.1, Claude Mythos 5.1, and Claude Opus 5, use a
    per-message effort change, which keeps the prompt cache"*; the negative is
    *"Changing the ``output_config.effort`` value always invalidates message
    blocks"* on the prompt-caching page, together with the 400 the effort page
    quotes for models without per-message effort -- which names Claude Fable 5
    explicitly. Both read 2026-09-22.
    """
    supported = EffortContinuity(
        capability=_SUPPORTED,
        mechanism=ContinuityMechanism.ANTHROPIC_PER_MESSAGE_EFFORT,
        # The vendor ships it; the installed SDK cannot express it. anthropic
        # 0.97.0's BetaMessageParam types role as Literal["user", "assistant"]
        # with no per-message output_config, and the package carries no beta
        # literal for mid-conversation-output-config-2026-07-01 (read
        # 2026-09-22). modelpass would have to hand-roll the wire shape.
        reachable=False,
        detail=(
            "vendor-documented: a role='system' message with empty content and "
            "output_config.effort keeps the cached prefix (beta header "
            "mid-conversation-output-config-2026-07-01; platform.claude.com "
            "effort and prompt-caching pages, read 2026-09-22). modelpass "
            "cannot send it -- anthropic 0.97.0 types message roles as "
            "user|assistant with no per-message output_config"
        ),
    )
    unsupported = EffortContinuity(
        capability=_UNSUPPORTED,
        mechanism=None,
        reachable=False,
        detail=(
            "vendor-documented negative: 'Changing the output_config.effort "
            "value always invalidates message blocks' (prompt-caching page), "
            "and this model has no per-message effort -- those return 400 "
            "'output_config.effort requires a model that supports per-turn "
            "effort', with Claude Fable 5 named (effort page, read 2026-09-22). "
            "Setting effort to the model's own default is the documented "
            "exception and invalidates nothing"
        ),
    )
    return tuple(
        [(model, supported) for model in _ANTHROPIC_PER_MESSAGE_MODELS]
        + [(model, unsupported) for model in _ANTHROPIC_NO_PER_MESSAGE_MODELS]
    )


_CONTINUITY[Runtime.ANTHROPIC_API] = _anthropic_api_rows()

_CONTINUITY[Runtime.OPENAI_API] = (
    (
        "gpt-6-astra",
        EffortContinuity(
            capability=_SUPPORTED,
            mechanism=ContinuityMechanism.OPENAI_CONFIGURATION_UPDATE,
            # Not typed by the installed SDK: openai 2.32.0's
            # ResponseInputItemParam union has no configuration_update member
            # (read 2026-09-22), so sending one would be an untyped dict
            # smuggled into `input`. modelpass builds that list itself and has
            # no seam for it.
            reachable=False,
            detail=(
                "vendor-documented: append a "
                "{'type': 'configuration_update', 'reasoning': {'effort': ...}} "
                "input item and leave the request-level reasoning.effort "
                "unchanged -- 'this preserves the original prompt prefix for "
                "prompt caching'. Astra only, standard single-agent mode, no "
                "two adjacent updates, not with automatic compaction or "
                "truncation (developers.openai.com reasoning guide, read "
                "2026-09-22). Not in openai 2.32.0's typed input union"
            ),
        ),
    ),
)

_CONTINUITY[Runtime.OPENAI_SDK] = (
    (
        "gpt-6-astra",
        EffortContinuity(
            capability=_EXPERIMENTAL,
            mechanism=ContinuityMechanism.CODEX_CONFIGURATION_UPDATE,
            reachable=False,
            detail=(
                "protocol-typed, vendor-silent: openai-codex 0.154.0 types "
                "ConfigurationUpdateResponseItem {reasoning: {effort}} in the "
                "response/history item unions, but nothing in the package or "
                "the app-server model catalog states the cache behaviour -- "
                "model/list carries supportedReasoningEfforts and "
                "defaultReasoningEffort and no supports_reasoning_effort_updates "
                "(read 2026-09-22). The installed codex.exe 0.151.0 contains "
                "the string 'configuration_update' zero times against four hits "
                "for 'supportedReasoningEfforts', so this build cannot be "
                "emitting one. Characterize before relying on it: see "
                "tests/live/test_codex_effort_continuity_live.py"
            ),
        ),
    ),
)

_CONTINUITY[Runtime.ANTHROPIC_SDK] = (
    (
        "",
        EffortContinuity(
            capability=_UNKNOWN,
            mechanism=None,
            reachable=False,
            detail=(
                "no mid-session effort change exists to characterize: "
                "claude-agent-sdk 0.2.148 takes effort on ClaudeAgentOptions at "
                "session start and ClaudeSDKClient offers set_model and "
                "set_permission_mode but no set_effort (read 2026-09-22). The "
                "question this capability asks does not arise here until one does"
            ),
        ),
    ),
)


def _same_model(name: str, token: str) -> bool:
    """Whether ``name`` is the model ``token`` names -- and **only** that model.

    **Deliberately stricter than :func:`~modelpass.sampling_rules._matches`, and
    the difference is the point.** That matcher takes the longest family prefix
    on a separator boundary, which is right for sampling rules: a family shares
    its temperature ceiling with its descendants, and a new point release
    inheriting the family's rules is the desired behaviour.

    A cache-continuity guarantee is not a family property. It was documented for
    three named models, and the next model in the line is exactly the case where
    inheriting silently would produce the thing this whole type refuses -- a
    ``supported`` claim about a pair nobody established. Under the family
    matcher, an unreleased ``claude-opus-5-1`` would inherit ``claude-opus-5``'s
    guarantee on the strength of its name.

    So: an exact match, or the same id with a date stamp appended
    (``claude-opus-5-20260115``), which is the one suffix that names the *same*
    model rather than a different one. Anything else falls through to the
    runtime's default row, which for Anthropic is ``UNSUPPORTED`` on the
    vendor's own documented negative and elsewhere is ``UNKNOWN``.
    """
    if name == token:
        return True
    if not name.startswith(token + "-"):
        return False
    suffix = name[len(token) + 1 :]
    return len(suffix) == 8 and suffix.isdigit()


def effort_cache_continuity(
    runtime: Runtime | str, model: str | None = None
) -> EffortContinuity:
    """What is established about changing effort on this pair. Never raises.

    **A resolver rather than a dict lookup, for the reason
    :func:`~modelpass.sampling_rules.model_efforts` is one**: today the answer
    comes from a static table of vendor documentation, and tomorrow part of it
    can come from the vendor -- Codex's model catalog already carries
    ``supportedReasoningEfforts`` beside the field that would answer this, and
    Anthropic publishes ``EffortCapability`` per model. A caller that asks a
    function does not move when the source improves.

    Model-specific on purpose. Continuity differs *within* one vendor and
    runtime -- Anthropic documents it for three models and documents its absence
    for the rest -- so a runtime-level answer would be wrong for most pairs it
    covered.
    """
    runtime = Runtime(runtime)
    rows = _CONTINUITY.get(runtime)
    if not rows:
        return EffortContinuity(
            detail=f"nothing established about effort and cache on {runtime.value}"
        )
    name = str(model or "").strip().lower()
    for token, verdict in rows:
        if token and _same_model(name, token):
            return verdict
    for token, verdict in rows:
        if not token:  # the runtime's default row, where one is recorded
            return verdict
    name = str(model or "").strip() or "an unnamed model"
    return EffortContinuity(
        detail=(
            f"nothing established about effort and cache for {name!r} on "
            f"{runtime.value}"
        )
    )


def observed_effort_continuity(
    *,
    effort_changed: bool,
    cached_input_tokens: int | None,
    prompt_tokens: int | None,
    floor: float = 0.5,
) -> bool | None:
    """Whether *this turn's* telemetry shows the prefix survived an effort change.

    ``prompt_tokens`` is the **whole prompt** -- fresh input plus cache reads
    plus cache writes -- and is taken as an argument rather than read off a
    :class:`~modelpass.types.TokenUsage` because the one thing that must not
    happen here is the nesting confusion that inflated every Codex run by 65%
    in an earlier ticket. modelpass counts ``input_tokens`` *beside*
    ``cached_input_tokens``; Codex's wire counts them nested. A ratio computed
    against the wrong denominator would report a cache break as a hit.
    :meth:`TokenUsage.prompt_tokens` is the safe source.

    ``None`` unless there is enough to answer, and there are three ways there
    are not: no effort change happened, the runtime reported no cache counts, or
    the turn's prompt was empty. **A ``False`` is a measurement; a ``None`` is
    the absence of one**, and the two must not read alike -- which is why this
    returns three values rather than a bool.

    **The threshold is a majority of the prompt, not an equality.** The
    conversation grows every turn, so the cached fraction after a change is
    never the fraction before it. What distinguishes preserved from broken is
    not a delta of a few hundred tokens: a break reports cached ~0 with a large
    cache *write*, preservation reports most of the prefix still read. A test
    asserting exact token equality would be asserting something no vendor
    promises.

    Never call this to decide whether a mechanism works *in general*. One turn
    is one turn. The general claim lives in :func:`effort_cache_continuity` and
    moves only on vendor documentation or a characterization run.
    """
    if not effort_changed:
        return None
    if cached_input_tokens is None or prompt_tokens is None or prompt_tokens <= 0:
        return None
    return (cached_input_tokens / prompt_tokens) >= floor
