"""The adapter contract.

An adapter is a leaf: it knows one vendor runtime and nothing about connection
storage, guards or the public API. The bridge hands it a fully prepared
:class:`RunRequest` -- messages normalized, environment already scrubbed,
directives computed -- and expects a stream of normalized events back.

Three rules the bridge relies on:

1. **The adapter never reads ``os.environ`` directly.** It launches with
   ``request.plan.env``. Anything else defeats the preflight.

   **The in-process clause** (2026-09-13), for the API runtimes, which launch
   nothing at all: *the credential is resolved explicitly and handed to a client
   constructor, and the vendor SDK must never read ``os.environ``.* Call
   :func:`~modelpass.preflight.resolve_credential` and pass the result --
   ``anthropic.Anthropic(api_key=...)``,
   ``openai.OpenAI(api_key=..., base_url=...)`` -- never the bare constructor,
   and never a constructor left to find its own key.

   This is the same rule, and it needs saying separately because the mechanism
   that used to enforce it is gone. There is no child environment to scrub, so
   ``request.plan.env`` is empty on these runtimes by design
   (:func:`~modelpass.preflight.plan_launch` says why), and the failure it
   prevents is silent: a machine with an ambient ``OPENAI_API_KEY`` and a
   connection naming a different credential would bill the wrong account, with
   nothing anywhere reporting it. ``tests/test_no_ambient_credentials.py`` is
   where that stays checked.
2. **The adapter never stamps a terminal event's identity.** It may report a
   terminal *status*; the bridge owns the connection/auth-mode stamp (D3), so a
   misbehaving adapter cannot claim a run was subscription-billed.
3. **Anything not in the normalized vocabulary becomes a ``vendor_event``**
   rather than being dropped. Retries, permission requests, plan detection and
   vendor cost fields all arrive that way in v1 (D7).

A fourth rule arrived with tools (D12):

4. **Tool events are observations.** An adapter emits ``tool_call`` /
   ``tool_result`` for tools the *runtime* executed; it never waits on the
   caller for a result. Whatever a runtime reports that is richer than those two
   events -- content blocks, partial output, permission prompts -- follows rule 3.

And a fifth on first-consumer feedback (2026-08-17):

5. **A vendor failure is reported, not raised.** When the vendor says no -- an
   API error, a rejected model, a runtime that will not start, a transport that
   dies -- the adapter either emits a ``terminal`` event with
   ``status=error`` or raises :class:`~modelpass.errors.VendorRunFailed`, and the
   bridge turns the latter into exactly that terminal, stamped and carrying the
   usage spent so far. Which of the two an adapter uses is a matter of where it
   is standing: inside its own event loop, emit the terminal; from a place that
   cannot yield, raise.

   Raise nothing else. Any other exception escaping :meth:`Adapter.run` is a bug
   and reaches the caller as :class:`~modelpass.errors.AdapterFailed`. The two
   deliberate exceptions are :class:`~modelpass.errors.AuthModeMismatch` and
   :class:`~modelpass.errors.UnsafeLaunch`, which are guaranteed-layer refusals
   (D4 layer 1) and must keep raising: a guarantee that can be downgraded into a
   status line is not a guarantee.

And a sixth with structured output (D13):

6. **A requested schema is answered or the run is an error.** When
   ``request.schema`` is set, the adapter emits exactly one
   ``structured_output`` event immediately before the terminal. If nothing
   parseable came back it emits the event only if there was *something* to show,
   and the terminal carries ``status=error`` with a reason naming what did
   arrive. An empty dict is never substituted for a missing answer, and
   ``valid=False`` never causes the adapter to alter ``data``.

And a seventh with sessions (D14-D17):

7. **A session is opened locally and created by running.** Neither runtime has
   a create-session call, so :meth:`Adapter.open_session` builds local state and
   nothing else -- no subprocess, no request, no spend. The vendor session comes
   into existence during the first :meth:`SessionHandle.send`, which is also
   when :attr:`SessionHandle.id` may first become non-``None``. An adapter that
   hands out an id before the session is durable is telling a lie the caller
   will discover only when a resume fails.
"""

from __future__ import annotations

import abc
import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol, runtime_checkable

from ..capabilities import Capability, Support, VerifyReport
from ..connections import Connection
from ..errors import AdapterNotImplemented
from ..preflight import CacheEligibility, ModelListProbe, PreflightPlan, Receipt
from ..runtimes import Runtime
from ..tools import ToolDef
from ..types import AgentEvent, Message, Sampling, SessionInfo, SessionKind, TextBlock

__all__ = [
    "Adapter",
    "AsyncRun",
    "RunRequest",
    "SessionHandle",
    "SessionRequest",
    "close_async_run",
    "native_run",
    "threaded_run",
]


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Everything an adapter needs for one stateless chat call.

    ``tools`` and ``mcp_servers`` are the Phase 6 additions (D12). Both are
    already capability-gated by the time an adapter sees them: the bridge
    refuses to build a request whose runtime cannot honour them, so an adapter
    receiving a non-empty value may assume it advertised support for it.

    ``schema`` / ``schema_name`` are the Phase 8 additions (D13) and follow the
    same rule -- gated, normalized and mutually exclusive with ``tools`` /
    ``mcp_servers`` before an adapter sees them. ``schema_name`` is the caller's
    name for the schema (falling back to its ``title``) and may be empty; it is
    a display and wire-protocol detail, never something modelpass invents.

    ``sampling`` is the ticket 1.7 addition (R5) and follows *neither* rule: it
    is not gated, and an adapter that cannot honour it reports rather than
    refuses. The field's own note says why.
    """

    connection: Connection
    messages: tuple[Message, ...]
    plan: PreflightPlan
    model: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[ToolDef, ...] = ()
    mcp_servers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    schema: Mapping[str, Any] | None = None
    schema_name: str = ""
    #: How the model should pick its words, as a *request* (R5, ticket 1.7).
    #: Unlike ``tools`` and ``schema`` this is **not** capability-gated before an
    #: adapter sees it, and the asymmetry is deliberate: asking a runtime to run
    #: a tool it cannot run has no sensible outcome but a refusal, whereas asking
    #: for a temperature a model will not take has an obvious one -- send what it
    #: takes, and say what happened to the rest. So a ``Sampling`` on an agent
    #: runtime is not an error; it is a receipt with notes on it.
    #:
    #: An adapter resolves it through
    #: :func:`~modelpass.sampling_rules.plan_sampling`, never by reading the
    #: fields directly, so the values it sends and the values the receipt reports
    #: cannot drift apart.
    sampling: Sampling | None = None

    @property
    def runtime(self) -> Runtime:
        return self.connection.runtime

    @property
    def env(self) -> Mapping[str, str]:
        return self.plan.env

    @property
    def wants_tools(self) -> bool:
        """Whether this run needs the runtime to execute a tool loop.

        The distinction matters to adapters: a run with no tools and no MCP
        servers is the D7 single-turn call and must keep behaving exactly as it
        did before Phase 6.
        """
        return bool(self.tools or self.mcp_servers)

    @property
    def native_tools(self) -> bool:
        """Whether the runtime's own toolbelt stays switched on (D23).

        ``False`` by default, because :meth:`~modelpass.Bridge.chat` is
        chat-shaped and says so in its own docstring: *"built-in tools are off
        by construction here, so ``tools=`` means the caller's own functions
        and nothing else."* That promise is unqualified by runtime, and until
        2026-08-31 only ``anthropic-sdk`` kept it -- ``openai-sdk`` ran every
        stateless call with Codex's full coding-agent toolbelt attached, worth
        a measured 5,621 prompt tokens (35.4% of a bare call's prefix) and an
        ``exec`` the caller never asked for. D23 is that repair.

        Deliberately the same name and the same meaning as
        :attr:`SessionRequest.native_tools`, so one word means one thing across
        the two request objects. What differs is where the answer comes from: a
        session reads it off its *kind* (a worker keeps the toolbelt, a chat
        does not), and a stateless call has no kind, so it reads an option.

        ``options={"native_tools": True}`` is the opt-in, for a caller who
        genuinely wants Codex driving its own shell -- the runtime as a coding
        agent rather than as a model. **It is read on ``openai-sdk`` only.**
        ``anthropic-sdk``'s stateless path passes ``tools=[]`` unconditionally
        and has since Phase 3; making the option work there would mean mapping
        to the ``claude_code`` preset, which is a session's job and reopens
        D22's preset question. The key is absent from
        :attr:`AnthropicAdapter.option_keys` (which is ``None``, so nothing
        warns), and that asymmetry is named here rather than left to be
        discovered.
        """
        return bool(self.options.get("native_tools", False))

    @property
    def effective_sampling(self) -> Sampling | None:
        """The sampling request, with ticket 1.6's option folded in (R5).

        ``options={"max_output_tokens": N}`` was the stopgap 1.6 shipped,
        because the Messages API cannot be called without a ceiling and there
        was nowhere else to put one. It keeps working as an **alias** rather
        than as a second channel: it is read here, once, into the same field
        ``sampling=`` fills, so exactly one value reaches the wire and exactly
        one appears on the receipt. Two live spellings of one field is how a
        report starts disagreeing with the request that produced it.

        ``sampling=`` wins where a caller sets both -- the request field beats
        the escape hatch that predates it. The option is documented as
        deprecated on :meth:`~modelpass.Bridge.chat` and warns about nothing
        yet; a warning belongs with the consumer migrations rather than ahead of
        them, which is the same courtesy the ``subpass`` shim gets.

        **Every reader of the sampling request goes through here**, the receipt
        included, which is what makes "what the adapter sent" and "what the
        receipt says" the same sentence rather than two implementations of it.
        """
        ceiling = self.options.get("max_output_tokens")
        if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling <= 0:
            return self.sampling
        if self.sampling is None:
            return Sampling(max_output_tokens=ceiling)
        if self.sampling.max_output_tokens is None:
            return replace(self.sampling, max_output_tokens=ceiling)
        return self.sampling

    @property
    def wants_schema(self) -> bool:
        """Whether this run must produce a schema-bound answer (D13)."""
        return self.schema is not None

    @property
    def system_blocks(self) -> tuple[TextBlock, ...]:
        """The system-role instruction as content blocks, in order (R3).

        **What an ``anthropic-api`` adapter passes straight to
        ``messages.create(system=[...])``**, ``cache_control`` intact. Every
        system message contributes its blocks; a plain-string system message
        contributes one unmarked block, so this is never a narrower rendering
        of the prompt than :attr:`system_prompt_chars` counted.

        A property rather than a stored field on purpose: the messages are the
        one source of truth for what this run sends, and a second copy of the
        system prompt sitting beside them is a copy that can disagree with them.

        Empty when the run carries no system message at all, which is the same
        answer ``system=`` being omitted gives.
        """
        blocks: list[TextBlock] = []
        for message in self.messages:
            if message.role.value == "system":
                blocks.extend(message.blocks)
        return tuple(blocks)

    @property
    def conversation_blocks(self) -> tuple[tuple[str, tuple[TextBlock, ...]], ...]:
        """The non-system turns as ``(role, blocks)`` pairs (R3).

        The per-message half of the same answer: an adapter with a real
        ``messages=`` array renders each turn's blocks -- breakpoints and all --
        rather than the flattened transcript the two CLI runtimes need.
        """
        return tuple(
            (m.role.value, m.blocks) for m in self.messages if m.role.value != "system"
        )

    @property
    def cache_breakpoints(self) -> int:
        """How many explicit ``cache_control`` breakpoints this run carries.

        Counted across every message, system included. ``0`` for a run whose
        content is all plain strings, which is every run written before R3 --
        so "did the caller ask for breakpoints at all" and "did modelpass
        honour them" stay two separate questions on the receipt.
        """
        return sum(m.cache_breakpoints for m in self.messages)

    @property
    def system_prompt_chars(self) -> int:
        """Characters of system-role instruction this run will send.

        ``0`` when the caller passed no ``system_prompt``. Counted from the
        message list because that is where :meth:`Bridge.chat` assembles it, and
        counted in characters because that is the unit modelpass can be honest
        about -- see :data:`~modelpass.preflight.CHARS_PER_TOKEN`.

        Adapters render the system parts slightly differently on the wire, so
        this is deliberately the *content* length and not a prediction of the
        serialized prefix. It feeds an estimate that already says it is one.
        """
        parts = [m.text for m in self.messages if m.role.value == "system"]
        if not parts:
            return 0
        return sum(len(part) for part in parts) + 2 * (len(parts) - 1)


@dataclass(frozen=True, slots=True)
class SessionRequest:
    """Everything an adapter needs to open one session (D14-D17).

    The session analogue of :class:`RunRequest`, and deliberately not the same
    object. A ``RunRequest`` carries a message list, because a stateless call
    *is* its history; a ``SessionRequest`` carries none, because the history
    lives where the runtime keeps it -- which is the whole point of D14. What it
    carries instead is the **prefix**: the tools and the system prompt, fixed
    once, for every turn this session will ever take.

    Everything here has already been gated by the bridge before an adapter sees
    it, exactly as with ``RunRequest``: a ``SessionRequest`` whose ``tools`` are
    non-empty reached a runtime that advertised ``tools_in_process``, and one
    whose ``kind`` is ``chat`` with a ``system_prompt`` reached a runtime that
    advertised ``system_prompt_replace``. An adapter does not re-check; it maps.

    ``project_folder`` is the process working directory, and it is never
    ``None`` by the time an adapter sees it -- the bridge substitutes a
    modelpass-owned directory for a ``ChatSession`` that did not name one, and
    refuses to build a ``WorkerSession`` without one. It is called
    ``project_folder`` and not ``project`` because that word already means
    something else in Claude Code, in a downstream agent host, and to whoever
    adopts modelpass next (D15).

    ``persist=False`` means the session must leave nothing on disk and cannot be
    resumed later. It is not merely a storage flag: on ``anthropic-sdk`` a live
    client still carries the conversation across turns, and on ``openai-sdk``
    there is no multi-turn at all without the rollout file -- which is why the
    bridge refuses the combination there rather than handing back an object that
    calls itself a session and behaves like a series of one-shots.

    ``resume_id`` is set only by :meth:`Adapter.resume_session` and is the id the
    caller asked to pick back up.
    """

    connection: Connection
    plan: PreflightPlan
    kind: SessionKind
    project_folder: str
    model: str | None = None
    system_prompt: str | None = None
    persist: bool = True
    tools: tuple[ToolDef, ...] = ()
    mcp_servers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    resume_id: str | None = None

    @property
    def runtime(self) -> Runtime:
        return self.connection.runtime

    @property
    def env(self) -> Mapping[str, str]:
        return self.plan.env

    @property
    def native_tools(self) -> bool:
        """Whether the runtime's own toolbelt stays switched on (D15).

        ``True`` for a worker, ``False`` for a chat. On ``anthropic-sdk`` the
        off position is ``tools=[]``; on ``openai-sdk`` it is
        ``-c features.shell_tool=false`` and its siblings, which was verified to
        work and to remove the tool definitions from the wire payload rather
        than merely hiding them (5,621 prompt tokens on app-server, D23).

        **Why a worker keeps them, stated plainly because it is the whole
        difference between the two objects.** A worker exists to *do* something,
        and doing it is iterative: run a command, read what came back, adjust,
        run the next one. Codex's ``exec`` / ``exec_command`` and Claude Code's
        Bash are what make that loop possible at all -- the agent is armed with
        them, and taking them away leaves an object that can only talk about
        work it cannot perform. That is what a ``ChatSession`` already is. So
        the toolbelt is not an oversight a worker tolerates; it is the
        capability a caller chose a worker in order to get.

        **What a worker is actually armed with.** Neither list is modelpass's --
        both are the vendor's own defaults, and both can grow without modelpass
        changing:

        * ``anthropic-sdk``: ``{"type": "preset", "preset": "claude_code"}``,
          the SDK's documented "all default Claude Code tools" switch. File
          reads and writes, edits, shell, search, web fetch, task management.
          Named in the options rather than left to the default so a reader can
          find it (``adapters/anthropic.py``).
        * ``openai-sdk``: Codex's own belt. Enumerated live 2026-08-31 by asking
          a real run what it could call, and it answered with fourteen:
          ``exec``, ``exec_command``, ``write_stdin``, ``apply_patch``,
          ``update_plan``, ``view_image``, ``image_gen.imagegen``, ``web.run``,
          ``wait``, ``request_user_input``, ``request_plugin_install``,
          ``list_mcp_resources``, ``list_mcp_resource_templates`` and
          ``read_mcp_resource``. Six of those are what
          :func:`~modelpass.adapters.openai.chat_tool_overrides` reaches; the rest
          are not separately switchable, which is why that function's docstring
          says "the toolbelt modelpass can reach is off" rather than "the toolbelt
          is off".

        **This is all-or-nothing today, and that is a real limitation rather
        than a design.** A caller who wants a worker with file access but no
        network, or shell but no image generation, has no way to say so: the
        switch is one boolean per session. See the decision log for why a
        per-tool subset is not obviously safe to add -- the short version is
        that the vendor's own instructions reference the belt they ship with,
        so removing tools underneath those instructions risks a model that
        confidently calls something that is no longer there.
        """
        return self.kind is SessionKind.WORKER

    @property
    def appends_system_prompt(self) -> bool:
        """Whether ``system_prompt`` is added to the runtime's persona or replaces it.

        The single fact that separates the two objects. A worker appends,
        because the runtime's persona is what you wanted; a chat replaces,
        because it is what you are trying to get rid of. On ``anthropic-sdk``
        that is ``{"type": "preset", "preset": "claude_code", "append": ...}``
        against a bare string; on ``openai-sdk`` it is
        ``ThreadStartParams.baseInstructions`` against the layered ``System: ``
        line :func:`~modelpass.adapters.openai.render_session_prompt` writes.

        **This used to say "on ``openai-sdk`` only the append half exists".**
        That was true of ``codex exec`` and stopped being true on 2026-08-31,
        when S7 made ``codex app-server`` the default and ``baseInstructions``
        -- which replaces -- became reachable. ``SYSTEM_PROMPT_REPLACE`` is
        ``supported`` for that runtime in the registry, and a ``ChatSession``
        opens there. The gate that remains is transport-scoped and correct: it
        still refuses on an explicit ``options={"transport": "exec"}``, where
        the append-only finding stands verbatim.

        A worker with **no** ``system_prompt`` still appends -- to nothing. It
        must map to the preset and never to omission, because on
        ``anthropic-sdk`` omitting the prompt does not give Claude Code's
        instructions, it gives the minimal tool-calling ones: a full toolbelt
        with no guidance for it.
        """
        return self.kind is SessionKind.WORKER

    @property
    def wants_tools(self) -> bool:
        """Whether the caller handed this session a tool loop of its own (D12)."""
        return bool(self.tools or self.mcp_servers)

    @property
    def system_prompt_chars(self) -> int:
        """Characters of caller-supplied system prompt; ``0`` when there is none.

        The caller's text only. On a worker the runtime's own preset sits in
        front of it and is not counted here, because modelpass does not have it --
        :attr:`appends_system_prompt` is the flag that says so, and the caching
        disclosure reads both rather than treating this number as the whole
        prefix.
        """
        return len(self.system_prompt or "")

    @property
    def cache_breakpoints(self) -> int:
        """Always ``0``: a session's system prompt is a string (R3).

        Stated here rather than left to an ``AttributeError`` because the
        receipt asks both request types the same question. Sessions carry no
        breakpoints because neither session-capable runtime takes them: the
        prefix is fixed at construction and cached by being byte-stable, which
        is the control D17 already describes. If an API runtime ever grows a
        session face, this is the property that changes.
        """
        return 0


@runtime_checkable
class SessionHandle(Protocol):
    """One live vendor session, owned by the adapter that opened it.

    The division of labour, stated because it is what keeps the two public
    session classes a single implementation: **the handle knows the vendor and
    nothing else.** It does not stamp terminal events, run guards, emit the
    receipt, decide whether an id may be shown, or know that ``ChatSession`` and
    ``WorkerSession`` are different words for it. All of that belongs to
    :mod:`modelpass.sessions`, which is why it is written once.

    A handle is not reusable across connections and is not thread-safe. One
    turn at a time; the session object serializes that.
    """

    @property
    def id(self) -> str | None:
        """The runtime's own id for this session, or ``None`` if it has none yet.

        ``None`` until the first turn **completes**, and ``None`` for the whole
        life of a ``persist=False`` session. This is not caution for its own
        sake: on ``openai-sdk`` a thread is durable only after its first turn
        finishes, and a run killed before that leaves an id that
        ``codex exec resume`` rejects outright. Handing that id to a caller who
        will store it is worse than handing over nothing.
        """
        ...

    def send(self, message: str) -> Iterator[AgentEvent]:
        """Run one turn and stream normalized events.

        The same event vocabulary and the same failure rules as
        :meth:`Adapter.run` -- rules 2 through 6 above apply unchanged, and a
        vendor failure is still reported rather than raised. The differences are
        that history is not passed in (the runtime has it) and that the terminal
        ends *this turn*, not the session: the handle stays usable, and the next
        ``send`` continues the same conversation against the same cached prefix.

        On the first call this is also what creates the session, so this is
        where an id first appears.
        """
        ...

    def history(self) -> tuple[Message, ...]:
        """The conversation so far, as the runtime has it.

        **Read-only introspection, and it cannot be fed back as input** --
        ``get_session_messages()`` on the Agent SDK is explicit about this, and
        it is the reason D14 exists. An adapter returns what it can read and an
        empty tuple when the runtime offers no way to read it; it never
        reconstructs a transcript from the events it happened to see, because a
        history assembled by modelpass and a history held by the runtime would
        drift and only one of them is what the model is actually being sent.
        """
        ...

    def close(self) -> None:
        """Release the session's local resources. Idempotent.

        Closing does **not** delete a persisted session -- a resumable session
        stays resumable. What it ends is modelpass's hold on it: the live
        subprocess, the client, the temp directory. Calling it twice is not an
        error, because a context manager and an explicit ``close()`` are both
        ordinary ways to reach it.
        """
        ...


class Adapter(abc.ABC):
    """What every runtime adapter implements."""

    #: The runtime this adapter drives.
    runtime: Runtime

    #: Per-connection :meth:`probe` results, owned by the base class. Not a
    #: class attribute by accident of assignment: it is created on first use in
    #: :meth:`cached_probe` and reset by :meth:`invalidate_identity_cache`.
    _probe_cache: dict[str, ModelListProbe | None]

    #: The ``options`` keys this adapter reads, or ``None`` when it forwards
    #: unrecognized keys to the vendor SDK and so cannot say what is unknown.
    #:
    #: A closed set is what lets an unknown key be *reported* rather than
    #: silently ignored. That matters more here than it looks: ``codex_bin`` is
    #: the documented lever for which binary actually runs, and a typo'd
    #: ``codex_bn`` or a Codex key on an Anthropic connection currently does
    #: nothing at all -- indistinguishable, from the outside, from a working
    #: override. It is the same failure shape as Codex silently ignoring an
    #: unknown ``-c`` key, which modelpass warns consumers about.
    option_keys: ClassVar[frozenset[str] | None] = None

    @classmethod
    def unknown_option_keys(cls, options: Mapping[str, Any] | None) -> tuple[str, ...]:
        """Keys this adapter will not read. Empty when it cannot know."""
        if cls.option_keys is None or not options:
            return ()
        return tuple(sorted(set(options) - cls.option_keys))

    @classmethod
    def is_available(cls) -> bool:
        """Whether the vendor package backing this runtime can be imported.

        Must not import the vendor package as a side effect of answering.
        """
        return True

    @abc.abstractmethod
    def preflight(self, request: RunRequest) -> Receipt:
        """Detect which auth mode this run would actually use, and report it.

        Should build on :meth:`Receipt.from_plan` so the vendor-independent facts
        (what was scrubbed, what was preserved, which directives apply) stay
        consistent across adapters.
        """

    def invalidate_identity_cache(self) -> None:
        """Drop any cached answer about *which account* this runtime is logged into.

        An adapter that probes the vendor for account identity during
        :meth:`preflight` may cache that answer -- the probes are subprocesses
        costing seconds, and every run, session open and bench page goes through
        a preflight that wants one. The cache is what keeps a chat call from
        paying for it.

        This is the escape hatch for the one caller who must not be served a
        cached answer: ``modelpass verify`` (and the bench's Verify button) exists
        precisely to re-read the live identity and pin it, so it invalidates
        first through :meth:`~modelpass.bridge.Bridge.refresh_identity`. A no-op
        here for identity, because an adapter that caches nothing has nothing to
        drop -- concrete rather than abstract on purpose, so caching stays an
        adapter's own optimization and not a method every implementation must
        write. It does clear the :meth:`probe` cache, which the base owns, so
        **an override must call ``super().invalidate_identity_cache()``**.
        """
        self._probe_cache = {}
        return None

    def probe(self, request: RunRequest) -> ModelListProbe | None:
        """Ask the endpoint whether this credential is live, without spending.

        The optional third check of the API preflight (see
        :func:`~modelpass.preflight.api_preflight`). A model-list call is
        token-free on every vendor and is the only thing that can prove a key
        works *before* a run; on ``openai-compatible`` it is also the only
        reliable way to learn what the configured endpoint actually serves, and
        the input :meth:`~modelpass.capabilities.CapabilityRegistry.refine`
        wants.

        ``None`` by default, and ``None`` means **nobody asked** -- the third
        state, not a failure. An adapter that has not implemented a probe says
        nothing rather than reporting a verdict it did not obtain, which is the
        same posture :meth:`cache_eligibility` takes.

        Call it through :meth:`cached_probe`, never directly: this is a network
        round trip and :meth:`~modelpass.Bridge.preflight` runs on every single
        call.
        """
        del request
        return None

    def cached_probe(self, request: RunRequest) -> ModelListProbe | None:
        """:meth:`probe`, memoized per connection until the cache is dropped.

        The pattern the identity probes already use, for the same reason: the
        answer changes when a key is rotated or an endpoint moves, which is
        monthly at best, and paying a round trip per ``chat()`` to re-learn it is
        a bad trade. ``modelpass verify`` is the caller that must never be served
        a cached answer, and it drops this through
        :meth:`~modelpass.Bridge.refresh_identity`.

        Keyed by connection name, because one adapter instance serves every
        connection on its runtime and two of them may hold different keys.
        """
        cache = getattr(self, "_probe_cache", None)
        if cache is None:
            cache = {}
            self._probe_cache = cache
        key = request.connection.name
        if key not in cache:
            cache[key] = self.probe(request)
        return cache[key]

    def verify_capabilities(self, request: RunRequest) -> VerifyReport | None:
        """Drive this *endpoint* and report which capability cells it answers.

        ``None`` by default, and ``None`` means **this runtime has nothing to
        verify per install** -- which is the honest answer for every runtime
        whose capability row is a fact about a vendor. ``anthropic-api`` talks to
        Anthropic; driving one install of it teaches nothing that is not already
        in the table with a date on it, and letting a user's local run move a
        shared cell is precisely what the tri-state exists to prevent.

        ``openai-compatible`` is the exception the hook exists for (ticket 1.10).
        Its row describes an API *shape*, so only the configured endpoint can say
        what it does, and its adapter implements this by driving one short chat,
        one tool round trip and one structured-output call and reporting what
        worked. ``modelpass verify`` writes the result onto the connection, where
        :meth:`~modelpass.Bridge.registry_for` folds it into a registry copy
        through :meth:`~modelpass.capabilities.CapabilityRegistry.refine`.

        **This method spends.** It is not part of the preflight and nothing calls
        it on the way to a run; it is the explicit verb a user types.
        """
        del request
        return None

    @abc.abstractmethod
    def run(self, request: RunRequest) -> Iterator[AgentEvent]:
        """Execute one stateless chat call and stream normalized events."""

    def arun(self, request: RunRequest) -> AsyncIterator[AgentEvent]:
        """:meth:`run`, awaited (R1, ticket 1.13). Concrete, so every adapter has it.

        The default drives :meth:`run` on one worker thread and hands its events
        to the caller's loop through a bounded queue, which is why an adapter
        that never heard of asyncio still answers ``async for``. The bound is
        what keeps a fast runtime from filling memory ahead of a slow consumer:
        the worker blocks on a full queue exactly as a sync caller's own
        ``next()`` would have.

        **The four API adapters override this natively**, with the vendor's
        async client, and their in-adapter tool loop awaits a coroutine handler
        on the caller's own loop. That is the point of the whole ticket: an
        async application stops marshalling its own handlers back across a
        thread boundary -- a downstream agent host carries about 80 lines of
        exactly that.

        **Their sync :meth:`run` is not re-implemented over this**, and the
        symmetry is deliberate: R1 rejects an async core with a sync wrapper at
        the library's edge, and the same reasoning holds at the adapter's -- a
        sync consumer calling from inside a running loop would break.

        **Cancellation is ``aclose()``** (R2, D10). It calls :meth:`cancel`, the
        graceful half the sync face calls when a caller abandons the iterator,
        and then joins the worker with a *bounded* wait -- bounded rather than
        indefinite because a runtime that ignores its cancel must not be able to
        hang the loop that asked.
        """
        return threaded_run(
            self, lambda: self.run(request), name=request.connection.name
        )

    def support_for(
        self, capability: Capability, options: Mapping[str, Any]
    ) -> Support | None:
        """Support for one capability **on this request**, or ``None`` for no opinion.

        There are two different questions here and the registry only answers one
        of them (S7, 2026-08-31):

        1. *What can I rely on from this runtime by default?* That is
           :meth:`~modelpass.capabilities.CapabilityRegistry.support` and it is what
           :meth:`~modelpass.bridge.Bridge.find` selects a connection with. Its
           answer must not depend on a request, because there is no request yet:
           a caller asking "give me a connection that can run my functions
           in-process" is picking a connection before they have written the call.
        2. *Will THIS request work?* That is the gate in
           :meth:`~modelpass.bridge.Bridge.chat`, and its answer legitimately can
           depend on the request's own ``options`` -- because on some runtimes
           the options decide which transport runs, and the transports differ in
           what they can do.

        Until S7a those two questions shared one answer and the narrower one won,
        so ``bridge.chat(tools=..., options={"transport": "app-server"})`` was
        refused by a cell that correctly described ``codex exec``. The refusal
        was about a run nobody was asking for.

        **The direction an adapter answers in is not fixed** (2026-08-31). When
        ``openai-sdk``'s default flipped to ``codex app-server`` its hook stopped
        widening and started narrowing: an explicit ``transport="exec"`` now
        takes capabilities away, and adds back the one only exec has. Both are
        the same statement -- *this request is not the default* -- and an adapter
        implementing this method should expect to say it in either direction.

        **This is not a license to contradict the registry casually.** An adapter
        may answer only for a capability whose truth *genuinely* depends on the
        request -- where the option in question changes which vendor surface is
        driven, and the difference has been verified on both settings. Everything
        else returns ``None``, which means "no opinion, use the registry", and
        that is the default here. Returning a cheerier answer than the table for
        a capability the request does not actually change is how a caller ends up
        spending tokens to discover a refusal the registry already knew about.

        Called before the preflight and before anything launches, so it must not
        spend anything or look at the network. It may raise
        :class:`~modelpass.errors.CapabilityNotSupported` if the options themselves
        are unusable -- resolving a bogus ``transport`` is a refusal either way,
        and hearing it here rather than from the run costs nothing.
        """
        del capability, options
        return None

    def cache_eligibility(
        self, request: RunRequest | SessionRequest
    ) -> CacheEligibility | None:
        """Whether this call's prefix *can* cache, for the receipt (D20).

        Takes either request type, because the question is the same one for a
        stateless call and for a session and the answer differs only in inputs.
        A session gets its own :class:`SessionRequest` rather than a synthesized
        run: its prompt semantics are on that object -- a worker's prefix begins
        with the runtime's own preset, and reporting a worker as sub-floor
        because the caller's *appended* text is short would be worse than saying
        nothing.

        Concrete and returning ``None`` by default, which is the honest answer
        for an adapter that has not worked out its runtime's caching rules:
        modelpass says nothing rather than reporting a verdict nobody verified.
        The bridge attaches whatever comes back to the receipt, so an adapter
        opting in gets the disclosure with no other wiring.

        Why the *adapter* answers a question the bridge could nearly answer
        itself: the two facts that decide it are vendor knowledge. The minimum
        cacheable prefix varies by model, and whether a TTL can be pinned at all
        depends on the runtime and, on ``anthropic-sdk``, on the resolved CLI's
        version. The bridge knows neither and should not learn them.

        **Must not spend anything and must not launch a run.** It is called
        during the preflight, on the same terms: everything here is either pure
        computation or a token-free probe the preflight already made.
        """
        return None

    # --- sessions (D14-D17) ----------------------------------------------------
    #
    # Concrete rather than abstract, and that is the decision: making them
    # abstract would break every adapter that has not implemented them yet at
    # *import* time, including in a caller who never touches a session. Leaving
    # them off entirely would surface as ``AttributeError: 'OpenAIAdapter' object
    # has no attribute 'open_session'`` -- a message about Python rather than
    # about modelpass. So the contract exists here, and an adapter that has not
    # landed its half says so in a sentence naming what is missing.
    #
    # ``AdapterNotImplemented`` is a ``NotImplementedError`` as well as a
    # ``SubpassError``, so both ``except NotImplementedError`` and the
    # library-wide ``except SubpassError`` catch it.

    def open_session(self, request: SessionRequest) -> SessionHandle:
        """Open a new session. **Local work only -- this must not spend anything.**

        Build whatever local state the runtime needs and return a handle. Do not
        launch a subprocess that bills, do not send a request, and above all do
        not invent an id: neither runtime has a create-session call, and the
        session comes into existence when the first turn runs (adapter contract,
        rule 7).

        Capability gating has already happened. A request arriving here with
        ``tools``, with ``persist=False``, or with a ``system_prompt`` on a
        ``chat`` kind reached a runtime the registry says can do that.
        """
        raise AdapterNotImplemented(
            f"the adapter for runtime {self.runtime.value!r} has not implemented "
            "open_session(); bridge.new_chat() / bridge.new_worker() need it. "
            "Stateless bridge.chat() calls are unaffected"
        )

    def resume_session(self, request: SessionRequest) -> SessionHandle:
        """Pick a persisted session back up. ``request.resume_id`` names it.

        Raise :class:`~modelpass.errors.SessionNotFound` when the runtime does not
        have that session -- and check for it here if the runtime can be asked
        cheaply, because "this id is gone" is worth learning before a turn runs
        rather than from a failed one. An adapter that cannot check without
        spending says so by letting the first ``send`` raise it instead; both
        are honest, and the session object surfaces either the same way.

        Like :meth:`open_session`, this must not run a turn. Unlike it, the
        returned handle may report an :attr:`SessionHandle.id` immediately: the
        caller supplied a durable id and the runtime accepted it, so there is
        nothing left to be provisional about.
        """
        raise AdapterNotImplemented(
            f"the adapter for runtime {self.runtime.value!r} has not implemented "
            "resume_session(); bridge.resume_chat() needs it"
        )

    def list_sessions(self, request: SessionRequest) -> tuple[SessionInfo, ...]:
        """Every session this connection can see, newest first where that is known.

        Gated on ``sessions_list`` before an adapter is called, so reaching this
        method means the registry said yes for this runtime.

        Scope is ``request.project_folder`` where the runtime scopes by it --
        Anthropic stores sessions at ``~/.claude/projects/<encoded-cwd>/`` and
        the working directory *is* the storage key, so a listing there is
        per-folder and cannot be otherwise. A runtime with a flat store ignores
        it. Either way this is enumeration, not inference: a session the runtime
        does not report is not reported.

        ``persist=False`` sessions never appear, on any runtime, because they
        were never written down.
        """
        raise AdapterNotImplemented(
            f"the adapter for runtime {self.runtime.value!r} has not implemented "
            "list_sessions(); bridge.list_sessions() needs it"
        )

    def cancel(self) -> None:
        """Best-effort cancellation (D10).

        The documented floor is "terminate the process". Closing the iterator
        returned by :meth:`run` is the transport-level equivalent and is what the
        bridge does; runtimes with a graceful interrupt override this and are
        marked ``graceful_cancel`` in the capability registry.
        """
        return None


class AsyncRun:
    """What every :meth:`Adapter.arun` returns: an async iterator that can be cancelled.

    One method more than a bare async generator, and it is the whole reason this
    class exists: ``aclose(cancel=False)`` tears the run down *without* calling
    :meth:`Adapter.cancel`, which is what a bridge whose timeout watchdog has
    already cancelled needs (R2, and the "cancelled exactly once" rule a
    consumer's own cancellation contract rests on). A caller who just wants to stop calls
    ``aclose()`` and gets the cancel.
    """

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        return self

    async def __anext__(self) -> AgentEvent:
        raise NotImplementedError

    async def aclose(self, *, cancel: bool = True) -> None:
        raise NotImplementedError


class _NativeRun(AsyncRun):
    """An adapter's own async generator, wearing the :class:`AsyncRun` contract.

    What the four API adapters return: they drive the vendor's async client
    directly, so the iteration is theirs and only the cancellation shape is
    borrowed.
    """

    def __init__(self, adapter: Adapter, events: AsyncIterator[AgentEvent]) -> None:
        self._adapter = adapter
        self._events = events
        self._closed = False

    async def __anext__(self) -> AgentEvent:
        return await self._events.__anext__()

    async def aclose(self, *, cancel: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if cancel:
            self._adapter.cancel()
        closer = getattr(self._events, "aclose", None)
        if callable(closer):
            await closer()


async def close_async_run(
    stream: AsyncIterator[AgentEvent], adapter: Adapter, *, cancel: bool
) -> None:
    """Tear one async run down, cancelling **at most once**.

    ``cancel=False`` is the timed-out call: the watchdog already cancelled from
    the timer thread, and a second cancel would terminate a process the runtime
    may by then have replaced -- the rule the sync face states in the same
    words, kept in one place so the two faces cannot disagree about it.

    An adapter that overrode ``arun`` with a bare async generator instead of an
    :class:`AsyncRun` is still handled: its ``aclose()`` tears the transport
    down and the cancel is made here.
    """
    if isinstance(stream, AsyncRun):
        await stream.aclose(cancel=cancel)
        return
    closer = getattr(stream, "aclose", None)
    if callable(closer):
        await closer()
    if cancel:
        adapter.cancel()


def native_run(adapter: Adapter, events: AsyncIterator[AgentEvent]) -> AsyncRun:
    """An adapter's own async stream, as the object :meth:`Adapter.arun` promises."""
    return _NativeRun(adapter, events)


def threaded_run(
    adapter: Adapter,
    start: Callable[[], Iterator[AgentEvent]],
    *,
    name: str,
) -> AsyncRun:
    """A synchronous stream driven on a worker thread, read as an async iterator.

    ``start`` is called **on the worker**, so an adapter whose ``run()`` does
    real work before it yields does that work off the caller's loop too.
    """
    return _ThreadedRun(adapter, start, name=name)


#: Put on an :class:`_ThreadedRun`'s queue when its stream ended normally. A
#: sentinel rather than ``None``, which is a legitimate thing to hand over.
_DONE = object()


def _drive(
    start: Callable[[], Iterator[AgentEvent]],
    out: queue.Queue[Any],
    stop: threading.Event,
) -> None:
    """One run, on one worker thread, feeding one bounded queue.

    A module-level function rather than a method, so the running thread does not
    keep the :class:`_ThreadedRun` that started it alive -- see
    :meth:`_ThreadedRun._start`.
    """

    def put(item: Any) -> None:
        # The timeout is what makes the bound safe: a worker blocked forever on
        # a full queue behind a consumer who walked away would never see
        # ``stop``, which is exactly the thread this must not leave spinning.
        while not stop.is_set():
            try:
                out.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    stream: Iterator[AgentEvent] | None = None
    try:
        stream = start()
        for event in stream:
            if stop.is_set():
                break
            put(event)
    except BaseException as exc:  # re-raised on the caller's side, traceback intact
        put(exc)
        return
    finally:
        if stream is not None:
            close = getattr(stream, "close", None)
            if callable(close):
                # Closed here, in the thread that drove it. A generator closed
                # from another thread while it is executing raises
                # ``ValueError`` -- the failure the sync face's own comment
                # about a consumer's watchdog names.
                close()
    put(_DONE)


class _ThreadedRun(AsyncRun):
    """A sync stream on a worker thread, read as an async iterator.

    The default :meth:`Adapter.arun`, and therefore the thing that gives the two
    agent runtimes an async face for free. One thread per run, one bounded
    queue, and three ways it ends: the stream runs out, the worker's exception
    is re-raised on the caller's side, or :meth:`aclose`.

    Written as an explicit class rather than an async generator because an async
    generator's cleanup runs whenever the loop gets round to finalising it, and
    "the worker thread has been joined" is a promise that has to be keepable at
    a known moment (R2).
    """

    #: How many events may sit between the worker and the consumer. Small, so a
    #: fast runtime cannot stream a hundred megabytes into memory ahead of a
    #: consumer rendering them one at a time.
    MAX_PENDING = 64

    #: How long a join waits. A runtime that has not noticed its cancel by then
    #: is one this process cannot make notice; the thread is a daemon and is
    #: left to its own teardown, and the fact is reported by :attr:`joined`
    #: rather than papered over.
    JOIN_TIMEOUT = 5.0

    #: Put on the queue by the worker when the stream ended normally. A sentinel
    #: rather than ``None``, which is a legitimate thing to hand over.
    _DONE: ClassVar[object] = _DONE

    def __init__(
        self,
        adapter: Adapter,
        start: Callable[[], Iterator[AgentEvent]],
        *,
        name: str,
    ) -> None:
        self._adapter = adapter
        self._start_stream = start
        self._name = name
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.MAX_PENDING)
        self._stop = threading.Event()
        self._started = False
        self._finished = False
        self._thread: threading.Thread | None = None
        #: Whether the worker was seen to end. ``False`` only after a bounded
        #: wait ran out, which is a fact a test can assert and a caller can be
        #: told -- never one to hide.
        self.joined = True

    async def __anext__(self) -> AgentEvent:
        if self._finished:
            raise StopAsyncIteration
        if not self._started:
            self._start()
        item = await asyncio.to_thread(self._queue.get)
        if item is self._DONE:
            self._end()
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            self._end()
            raise item
        return item

    async def aclose(self, *, cancel: bool = True) -> None:
        """Cancel the run and join the worker (R2, D10).

        Idempotent, and safe on a run that already ended: a second close cancels
        nothing and waits for nothing.
        """
        if self._finished or not self._started:
            self._finished = True
            return
        self._finished = True
        self._stop.set()
        # The graceful half, from the caller's thread rather than the worker's
        # -- the same call the sync face makes when an abandoned iterator is
        # closed. The worker closes the vendor stream itself, in the thread that
        # was driving it. ``cancel=False`` is the watchdog's case: the timer has
        # already cancelled, and a second cancel would terminate a process the
        # runtime may by then have replaced.
        if cancel:
            self._adapter.cancel()
        await asyncio.to_thread(self._drain_and_join)

    def __del__(self) -> None:  # pragma: no cover - timing depends on the collector
        """A consumer who walked away without ``aclose()`` still stops the worker.

        Best effort, and documented as best effort: this runs whenever the
        collector gets here, which may be long after the abandonment and is
        never a moment a test can rely on. What it can do is set the flag and
        cancel, so the thread stops at its next event rather than living as long
        as the process. It cannot join -- there is no loop to await on here, and
        blocking inside a finaliser is how an interpreter shutdown hangs.
        """
        if self._started and not self._finished:
            self._stop.set()
            try:
                self._adapter.cancel()
            except Exception:
                return

    # --- the worker -------------------------------------------------------------

    def _start(self) -> None:
        self._started = True
        # **The worker holds no reference back to this object**, which is not a
        # style choice: a running thread keeps its target alive, a bound method
        # would keep ``self`` alive with it, and ``__del__`` would then never
        # run for the abandoned consumer it exists for. The closure carries the
        # three things the worker needs and nothing else.
        self._thread = threading.Thread(
            target=_drive,
            args=(self._start_stream, self._queue, self._stop),
            name=f"modelpass-arun-{self._name}",
            daemon=True,
        )
        self._thread.start()

    def _drain_and_join(self) -> None:
        thread = self._thread
        if thread is None:
            return
        deadline = time.monotonic() + self.JOIN_TIMEOUT
        while thread.is_alive() and time.monotonic() < deadline:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                thread.join(0.05)
        self.joined = not thread.is_alive()

    def _end(self) -> None:
        self._finished = True
        thread = self._thread
        if thread is None:
            return
        thread.join(self.JOIN_TIMEOUT)
        self.joined = not thread.is_alive()
