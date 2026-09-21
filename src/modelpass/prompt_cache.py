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
    "PromptCacheDisposition",
    "PromptCachePlan",
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
