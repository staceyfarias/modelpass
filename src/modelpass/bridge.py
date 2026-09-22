"""The public surface: sessions (D14), and the stateless :meth:`Bridge.chat` (D21).

**Two verbs, and the difference between them is who holds the history.**
:meth:`Bridge.chat` is stateless: the caller passes everything in, every time.
:meth:`Session.send` is stateful: history lives where the runtime keeps it, so
the prefix cache is native rather than reconstructed. :meth:`Bridge.new_chat`
and :meth:`Bridge.new_worker` open one; see :mod:`modelpass.sessions`.

::

    events = bridge.chat(
        connection="claude-sub",
        message="...",
        system_prompt="...",   # optional
        history=[...],         # optional, caller-owned
    )
    for event in events:
        ...

Each call runs an ephemeral runtime session with what it was given serialized
in. No hidden state, no session objects, and a known cost: whatever history the
caller passes is re-sent per call against the allowance. If that cost matters
later, a prefix-matching session cache can hide behind this same interface -- an
optimization, not a commitment.

Stream contract
---------------

* Every stream *begins* with exactly one ``receipt`` event per connection that
  runs, carrying that connection's preflight receipt. The rule is: a ``receipt``
  event always precedes any event produced by the connection it describes.
* A stream that completes normally ends with exactly one ``terminal`` event,
  stamped by the bridge with the connection name and the auth mode actually used
  (D3). Adapters cannot forge that stamp.
* Guard stops and quota exhaustion are *normal completions* with a terminal
  status, not exceptions -- unless the caller opts into ``raise_on_stop``.
* So is a vendor failure. A rejected model, an API error, a runtime that would
  not start: the stream ends with one ``terminal`` event carrying
  ``status=error``, the reason, and the usage spent up to that point. Exceptions
  after the iterator exists mean a caller mistake modelpass could not have caught
  earlier, or a bug -- never "the vendor said no".
* Configuration and preflight problems raise before any event is produced, so a
  caller that got an iterator has already passed the auth check -- and the
  ``receipt`` event is that guarantee's evidence, rather than something the
  caller has to go and recompute.
* Closing the iterator cancels the run, best-effort, with a floor of terminating
  the runtime process (D10).
* ``tools`` / ``mcp_servers`` let the runtime run a tool loop *inside* the one
  call (D12). ``tool_call`` / ``tool_result`` events are observations of what
  the runtime executed -- the caller never has to answer one.
* ``schema`` binds the run's final answer to a JSON Schema (D13), and adds
  exactly one ``structured_output`` event immediately before the terminal. Text
  deltas keep streaming as they occur. A run that was asked for a schema and
  produced nothing usable ends ``status=error`` with a reason naming what came
  back -- never a silently empty result.
* Every run that reaches a terminal appends one line to ``~/.modelpass/runs.jsonl``
  -- when, which connection, which auth mode actually paid, tokens, outcome.
  Best-effort and opt-out-able; see :mod:`modelpass.runlog`.
* A connection configured with ``onQuotaExhausted.failover`` may finish on a
  *second* connection. That is the only circumstance in which one
  :meth:`Bridge.chat` call touches two connections, it happens only because
  somebody wrote the second one down, and it is announced by a ``failover``
  event followed immediately by the second connection's own ``receipt`` event,
  both before it runs.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import (
    AsyncIterator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypeVar

from ._deadline import Deadline
from ._fold import CallFold
from .adapters import load_adapter
from .adapters.base import Adapter, RunRequest, SessionRequest, close_async_run
from .capabilities import Capability, CapabilityRegistry, Support, runtime_auth_modes
from .connections import (
    Connection,
    Group,
    GroupSelection,
    Guards,
    QuotaAction,
    QuotaPolicy,
    groups_of,
)
from .errors import (
    AdapterFailed,
    AuthModeMismatch,
    CapabilityNotSupported,
    ConnectionDisabled,
    GroupUnavailable,
    InvalidConnection,
    InvalidGuards,
    InvalidSchema,
    InvalidSession,
    NoSuchGroup,
    RuntimeNotAvailable,
    SubpassError,
    VendorRunFailed,
)
from .preflight import PreflightPlan, Receipt, plan_launch
from .prompt_cache import plan_prompt_cache
from .reasoning import stated_reasoning
from .runlog import RunLog, RunRecord, run_log_for
from .runtimes import API_RUNTIMES, Runtime
from .sampling_rules import plan_sampling, rules_for
from .schema import normalize_schema, resolve_schema_name
from .secrets import SecretStore
from .sessions import ChatSession, Session, WorkerSession, scratch_project_folder
from .store import ConnectionStore
from .tools import ToolDef, normalize_tools
from .types import (
    AgentEvent,
    AuthMode,
    FailoverEvent,
    Message,
    ReceiptEvent,
    Sampling,
    SessionInfo,
    SessionKind,
    StructuredOutputEvent,
    TerminalEvent,
    TerminalStatus,
    TextBlock,
    TextDeltaEvent,
    Timeout,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    normalize_messages,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; the import at use is a cycle
    from .manage import ConnectionManager

__all__ = ["Answer", "Bridge", "ValidationReport", "chat", "default_bridge"]

#: Bound so ``_open_session`` returns the concrete face it was handed, and
#: ``new_chat`` / ``new_worker`` keep their real return types.
_SessionT = TypeVar("_SessionT", bound=Session)

#: Added to every subscription receipt whose connection has no
#: :class:`~modelpass.models.AccountBinding`. One line, because every
#: pre-existing user sees it on every run until they pin, and it is advice
#: rather than a problem. Its exact text is also the de-duplication key --
#: see :meth:`Bridge._with_identity_verification`.
_UNPINNED_NOTE = (
    "account identity is not pinned; run 'modelpass verify <name>' to bind this "
    "profile to the vendor account it selects"
)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """What :meth:`Bridge.validate` found, offline (2026-08-17).

    A report rather than an exception, because the caller is usually validating
    a whole config file and wants every problem at once.

    ``problems`` are reasons this connection could not run as written.
    ``notes`` are true things worth saying that are not faults -- a disabled
    connection, an uninstalled vendor package, an absence of guards. ``ok`` is
    the one-line answer, and it is deliberately blind to ``notes``: a connection
    on a machine without the SDK is *valid*, it just cannot run here.
    """

    connection: str
    runtime: Runtime | None = None
    auth_mode: AuthMode | None = None
    enabled: bool = True
    runtime_installed: bool = False
    problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the configuration itself is sound. Not "this will run"."""
        return not self.problems

    def summary(self) -> str:
        """One human line, in the shape ``modelpass check`` uses."""
        head = f"{self.connection}: {'ok' if self.ok else 'invalid'}"
        if self.runtime is not None:
            head += f" ({self.runtime.value})"
        detail = list(self.problems) + list(self.notes)
        return head + ("" if not detail else " -- " + "; ".join(detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection": self.connection,
            "runtime": self.runtime.value if self.runtime else None,
            "auth_mode": self.auth_mode.value if self.auth_mode else None,
            "enabled": self.enabled,
            "runtime_installed": self.runtime_installed,
            "ok": self.ok,
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class Answer:
    """One run, drained: what :meth:`Bridge.ask` returns (R11).

    The iterator is the library's primary shape and stays that way. This is the
    shape for the other half of the call sites -- the ten one-shot ones in a
    scraper, the judge in an eval harness, the classify-this-line in a UI --
    where the caller wants a single blocking answer and would otherwise write
    the same drain loop ten times, each with its own idea of what "the text"
    was and its own hand-rolled reader for the structured half.

    Frozen, because it is a record of something that already happened.

    * ``text`` is every ``text_delta`` concatenated, in order. This is the same
      string ``ChatSubpass.invoke`` puts on its message, and
      ``tests/test_ask.py`` runs both over one script so the two cannot drift.
    * ``structured`` is the parsed schema-bound answer, and it is ``None`` only
      when no ``schema=`` was asked for. A run that was asked for one and ended
      ``ok`` without it raises rather than returning ``None`` here.
    * ``usage`` is the **final cumulative** total for the run -- the terminal
      event's figure, not a sum of the deltas, which would double-count a
      runtime that reports run totals rather than increments.
    * ``receipt`` is the receipt of the connection that *finished*. A run that
      failed over touched two; ``events`` holds both receipt events, and the
      terminal's ``failed_over_from`` names the one that ran first.
    * ``status`` is always ``ok`` by the time a caller holds this, because every
      other terminal raises. It is here anyway so the record is complete and so
      code that logs an :class:`Answer` does not have to special-case it.
    * ``tool_calls`` is the ``tool_call`` / ``tool_result`` observations in
      stream order, interleaved as they arrived. They are observations, not
      something the caller answers -- D12 is unchanged.
    * ``events`` is everything, in order, for a caller that wants the thinking,
      the vendor passthrough or the guard warnings after the fact.
    * ``sampling_applied`` / ``sampling_notes`` are ticket 1.7's honesty report
      lifted off the receipt onto the object a one-shot caller actually holds:
      what was really sent, and one sentence for each field that was not.
    """

    text: str
    receipt: Receipt
    status: TerminalStatus
    structured: Any = None
    usage: TokenUsage | None = None
    reason: str | None = None
    tool_calls: tuple[ToolCallEvent | ToolResultEvent, ...] = ()
    events: tuple[AgentEvent, ...] = ()
    sampling_applied: Mapping[str, Any] = field(default_factory=dict)
    sampling_notes: tuple[str, ...] = ()
    #: The effort dial, lifted here on the same argument as ``sampling_applied``
    #: (2026-09-22): a one-shot caller holds this object and nothing else, and
    #: the three questions about effort are answered in three different places
    #: otherwise -- the receipt, a vendor event, and a usage field.
    #:
    #: * ``reasoning_value`` -- the level sent, in the runtime's spelling.
    #: * ``reasoning_echo`` -- the level the vendor said it used, where any
    #:   vendor says: ``openai-sdk`` alone. ``None`` elsewhere because no echo
    #:   exists, not because the run failed to report one.
    #: * ``reasoning_metric`` -- how to read ``usage.reasoning_output_tokens``:
    #:   ``reported``, ``unreported`` or ``unavailable``.
    reasoning_value: str | None = None
    reasoning_echo: str | None = None
    reasoning_metric: str | None = None


#: The cells that together answer "does this runtime hold a conversation of its
#: own" (ticket 1.8, R11). All three unsupported is the capability table saying
#: the runtime keeps nothing between requests, which is the defining property of
#: an HTTP endpoint: every request carries its whole history. A runtime that can
#: resume, fork *or* list has something to hold a session in, and the existing
#: per-argument gates in :meth:`Bridge._open_session` take it from there.
#:
#: Read as a set rather than as one cell on purpose. Picking a single cell would
#: make the policy an accident of which one a future runtime happened to fill
#: in first.
_SESSION_CELLS: tuple[Capability, ...] = (
    Capability.SESSIONS_RESUME,
    Capability.SESSIONS_FORK,
    Capability.SESSIONS_LIST,
)


class _Collected:
    """:meth:`Bridge.chat`'s stream, drained into one :class:`Answer` (R11).

    Extracted in ticket 1.13 so :meth:`Bridge.ask` and :meth:`Bridge.aask` are
    the same reading of a stream rather than two readings that agree today. The
    loop that feeds it differs between the faces -- ``for`` and ``async for`` --
    and nothing else does.
    """

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.text: list[str] = []
        self.observations: list[ToolCallEvent | ToolResultEvent] = []
        self.structured: StructuredOutputEvent | None = None
        self.receipt: Receipt | None = None
        self.usage: TokenUsage | None = None
        self.terminal: TerminalEvent | None = None

    def absorb(self, event: AgentEvent) -> None:
        self.events.append(event)
        if isinstance(event, TextDeltaEvent):
            self.text.append(event.text)
        elif isinstance(event, ReceiptEvent):
            # Last one wins: a failover run carries two, and the one that
            # describes how this answer was actually paid for is the second.
            self.receipt = event.receipt
        elif isinstance(event, (ToolCallEvent, ToolResultEvent)):
            self.observations.append(event)
        elif isinstance(event, StructuredOutputEvent):
            self.structured = event
        elif isinstance(event, UsageEvent):
            # The bridge's own running total, never a sum of the events -- that
            # is how a runtime reporting run totals gets counted twice.
            if event.cumulative is not None:
                self.usage = event.cumulative
        elif isinstance(event, TerminalEvent):
            self.terminal = event
            self.usage = event.usage

    def answer(self, *, connection: str, schema: Mapping[str, Any] | None) -> Answer:
        """The collected answer, or the exception an unfinished run deserves."""
        terminal = self.terminal
        receipt = self.receipt
        if terminal is None or receipt is None:
            # Both are stream-contract guarantees, so their absence is a bug in
            # modelpass or an adapter rather than something that happened to a
            # run. Saying so beats returning whatever text arrived as though the
            # call had finished.
            missing = "terminal" if terminal is None else "receipt"
            raise AdapterFailed(
                f"the stream for connection {connection!r} "
                f"ended without a {missing} event, which the stream contract "
                "guarantees"
            )
        if terminal.status is not TerminalStatus.OK:
            # GUARD_STOP and QUOTA_EXHAUSTED have already raised from the pump,
            # and CANCELLED cannot happen here, so in practice this is the
            # vendor-failure arm. It is written to cover the whole enum anyway:
            # a status added later must not slip out of here as an ok answer.
            # The terminal rides along (ticket 1.12b): it holds the verdict,
            # the status code and the wait the vendor asked for, and an
            # exception raised in its place that knew none of them is how a
            # rate limit became UNKNOWN and was never retried.
            raise VendorRunFailed(
                terminal.reason
                or f"the run ended {terminal.status.value} with no reason given"
            ).with_terminal(terminal)
        if schema is not None and self.structured is None:
            raise AdapterFailed(
                f"the run for connection {terminal.connection!r} ended ok without "
                "the structured_output event that schema= asks for. modelpass ends "
                "a run that produced nothing parseable with a terminal error "
                "instead, so reaching this means the adapter broke the D13 "
                "contract -- and returning None here would look exactly like a "
                "schema-bound answer that legitimately found nothing"
            ).with_terminal(terminal)

        return Answer(
            text="".join(self.text),
            receipt=receipt,
            status=terminal.status,
            structured=None if self.structured is None else self.structured.data,
            usage=self.usage,
            reason=terminal.reason,
            tool_calls=tuple(self.observations),
            events=tuple(self.events),
            sampling_applied=dict(receipt.sampling_applied),
            sampling_notes=tuple(receipt.sampling_notes),
            reasoning_value=terminal.reasoning_value,
            reasoning_echo=terminal.reasoning_echo,
            reasoning_metric=terminal.reasoning_metric,
        )


@dataclass(frozen=True, slots=True)
class _CallPlan:
    """A call, decided but not yet priced: everything before the preflight.

    Ticket 1.13's first half. Frozen because a plan is an answer, not a
    scratchpad: the two faces share this pipeline and neither may edit the
    other's.
    """

    adapter: Adapter
    request: RunRequest
    guards: Guards
    failover: Connection | None
    expect_auth_mode: AuthMode | str | None
    timeout: Timeout | None
    raise_on_stop: bool

    def prepared(self, receipt: Receipt) -> _Prepared:
        return _Prepared(
            adapter=self.adapter,
            request=self.request,
            receipt=receipt,
            guards=self.guards,
            failover=self.failover,
            timeout=self.timeout,
            raise_on_stop=self.raise_on_stop,
        )


@dataclass(frozen=True, slots=True)
class _Prepared:
    """A call ready to run: the plan, plus the receipt that says what it costs.

    What :meth:`Bridge._prepare` returns and what both :meth:`Bridge.chat` and
    :meth:`Bridge.achat` pump. Nothing in here is face-specific, which is the
    whole point -- the sync and async doors differ in how they drive an
    iterator and in nothing else.
    """

    adapter: Adapter
    request: RunRequest
    receipt: Receipt
    guards: Guards
    failover: Connection | None
    timeout: Timeout | None
    raise_on_stop: bool


#: How a group is named where a connection name is expected (ticket 1.15).
#:
#: ``bridge.ask("group:cheap", "...")`` and ``chat(connection="group:cheap")``
#: both route through the group. See :meth:`Bridge._resolve` for why it is a
#: prefix and not a bare name.
_GROUP_PREFIX = "group:"



def _with_standing_reasoning(
    connection: Connection, sampling: Sampling | None
) -> Sampling | None:
    """Fill the per-call reasoning dial from the connection, when a call said nothing.

    The connection's ``reasoning`` is a **default for**
    :attr:`~modelpass.types.Sampling.reasoning_effort`, not a second dial
    pointing at the same wire field. Merging here -- at the one place every
    entry point assembles a request -- means the existing per-runtime routing,
    the drop reporting and the thinking-exclusivity rules in
    :mod:`modelpass.sampling_rules` all apply to it unchanged, instead of a
    parallel path that would have to learn them again and would disagree the
    first time one of them moved.

    **A call that names its own effort wins**, the same precedence ``model=``
    has over the connection's: the narrower statement is the later one, and a
    standing default that overrode a call would be a default nobody could turn
    off for one request.
    """
    stated = connection.reasoning
    if stated is None:
        return sampling
    if sampling is not None and sampling.reasoning_effort is not None:
        return sampling
    if "reasoning_effort" not in rules_for(connection.runtime, connection.model).accepted:
        # This runtime's sampling pipeline does not carry the dial -- both agent
        # runtimes report sampling_controls unsupported and drop everything.
        # Merging here anyway would put a "reasoning_effort not supported,
        # dropped" note on a receipt for a run where the adapter *did* apply it
        # (ClaudeAgentOptions.effort, TurnStartParams.effort), which is worse
        # than silence: it tells the caller the opposite of what happened.
        # Those adapters read the connection directly instead.
        return sampling
    base = sampling or Sampling()
    return replace(base, reasoning_effort=stated)

class Bridge:
    """Entry point. Owns a connection store, a capability registry and adapters.

    **Threading (R7, ticket 1.12).** One ``Bridge`` is safe to share across
    threads for :meth:`chat`, :meth:`ask`, :meth:`preflight` and
    :meth:`validate`. Each call builds its own request, its own
    :class:`~modelpass.guards.GuardTracker` and its own stream and shares
    nothing mutable with any other, so N workers on one bridge get N
    independent answers, receipts and usage figures. The connection store and
    the run log are guarded by their own locks. Adapters may be shared, and the
    fakes in :mod:`modelpass.testing` follow the same rules the real ones do.

    A :class:`~modelpass.sessions.Session` is the exception and takes **one
    caller at a time**: it holds a conversation, a tracker for that whole
    conversation, and a vendor handle whose turns are ordered. A second caller
    arriving mid-turn gets :class:`~modelpass.errors.SessionBusy` rather than an
    interleaved transcript or a deadlock. Give each thread its own session --
    opening one is local work by contract -- or take your own lock around the
    turn.

    This is written down because a consumer asked for it in a source comment and
    paid for the silence: a RAG evaluation harness runs a client per worker
    because nothing said whether one could be shared.
    """

    def __init__(
        self,
        *,
        store: ConnectionStore | None = None,
        registry: CapabilityRegistry | None = None,
        adapters: Mapping[Runtime, Adapter] | None = None,
        env: Mapping[str, str] | None = None,
        run_log: RunLog | None = None,
    ) -> None:
        self.store = store or ConnectionStore()
        self.registry = registry or CapabilityRegistry()
        self._adapters: dict[Runtime, Adapter] = dict(adapters or {})
        self._env = env
        self._run_log = run_log

    @property
    def run_log(self) -> RunLog:
        """The run-receipt log for this bridge's home (2026-08-17).

        Resolved from the store's ``[settings] runLog`` **on every access, and
        deliberately not cached**. The documented opt-out is editing
        ``connections.toml``, and a long-lived bridge -- ``default_bridge()`` in
        a host process, or the bench -- that had cached an enabled log would go
        on recording after the user switched it off, while every page telling
        them it was off. A logging switch that only takes effect on restart is
        not a switch. The cost is one small TOML parse per run, next to a call
        that contacts a model.

        An explicitly injected log wins and is used as given.
        """
        if self._run_log is not None:
            return self._run_log
        return run_log_for(self.store)

    @property
    def secrets(self) -> SecretStore:
        """The secrets file beside this bridge's connection file (R14).

        Resolved from ``self.store.root`` rather than from the ambient home, so
        a bridge pointed at a test root or an application's own directory reads
        the secrets that belong to *its* connections. Not cached, for the reason
        :attr:`run_log` is not: the file is edited out from under a long-lived
        process, and a stale answer here is a run charged to the wrong key.
        """
        return SecretStore(self.store.root)

    @property
    def manage(self) -> ConnectionManager:
        """Add, remove, rename and enable connections from Python (ticket 1.4).

        The write half of what ``modelpass connect`` does, with the same
        validation and the same receipts, so a desktop Settings page does not
        have to shell out to a command line or reimplement the rules. Kept on
        its own object rather than spread across this class: ``Bridge`` is the
        surface for *running*, and connection lifecycle is a different concern
        with its own vocabulary. See :mod:`modelpass.manage`.

        Not cached, for the reason :attr:`secrets` is not: the manager is a
        stateless facade over this bridge's store and secrets file.
        """
        from .manage import ConnectionManager

        return ConnectionManager(self)

    # --- connections -----------------------------------------------------------

    @property
    def env(self) -> Mapping[str, str]:
        return os.environ if self._env is None else self._env

    def connections(self) -> tuple[Connection, ...]:
        """Every configured connection. Empty on a fresh install, by design (D2)."""
        return self.store.list()

    def accounts(self) -> tuple[Connection, ...]:
        """Every configured account profile; account-oriented public façade."""
        return self.connections()

    def connection(self, name: str) -> Connection:
        return self.store.get(name)

    def account(self, name: str) -> Connection:
        """Resolve an account profile by its stable ID, not its nickname."""
        return self.connection(name)

    def groups(self) -> tuple[Group, ...]:
        """Every group, with its membership, in name order.

        Derived from the connections each time rather than stored: a group is
        the set of connections that claim it (see
        :attr:`~modelpass.connections.Connection.groups`), so there is nothing
        here that can be out of date with the file.
        """
        return groups_of(self.connections())

    def group(self, name: str) -> Group:
        """One group by name, or :class:`~modelpass.errors.NoSuchGroup`.

        Looking at a group is not the same as running through one: a group whose
        every member is disabled comes back from here intact, with the
        membership that explains why, and only :meth:`select` refuses it. That
        is the same split ``enabled = false`` already draws on a single
        connection -- it switches off spending, not looking.
        """
        for group in self.groups():
            if group.name == name:
                return group
        raise NoSuchGroup(name, tuple(g.name for g in self.groups()))

    def select(self, group: str) -> GroupSelection:
        """Which connection a run through ``group`` would go to, and who was skipped.

        **This spends nothing and starts nothing.** It is the honest answer to
        "if I say ``connection='group:cheap'``, what actually runs" -- available
        before the run rather than inferred from the receipt afterwards, which is
        the same bargain the preflight makes.

        Resolution is **name order among the enabled members**. That rule is
        arbitrary and the return type says so out loud: every member it walked
        past is in
        :attr:`~modelpass.connections.GroupSelection.passed_over` with the
        reason. A caller who needs a particular member names that member -- a
        group is a convenience for "any of these", not a router with a policy.
        Preference order *within* a group is a real want, and it is queued with
        throttling and metrics rather than guessed at here.

        Raises :class:`~modelpass.errors.NoSuchGroup` when nothing claims the
        name and :class:`~modelpass.errors.GroupUnavailable` when every member is
        disabled -- two findings that send a user to two different places, so
        they are two errors.
        """
        found = self.group(group)
        passed_over: list[tuple[str, str]] = []
        chosen: Connection | None = None
        for name in found.members:
            connection = self.store.get(name)
            if not connection.enabled:
                passed_over.append((name, "disabled"))
                continue
            if chosen is None:
                chosen = connection
                continue
            passed_over.append((name, f"{chosen.name} comes first in name order"))
        if chosen is None:
            raise GroupUnavailable(group, found.members)
        return GroupSelection(
            group=group, chosen=chosen, passed_over=tuple(passed_over)
        )

    def find(
        self,
        *,
        vendor: str | None = None,
        runtime: Runtime | str | None = None,
        capability: Capability | str | None = None,
        auth_mode: AuthMode | str | None = None,
        group: str | None = None,
        enabled: bool | None = None,
    ) -> tuple[Connection, ...]:
        """Accounts matching vendor, runtime, auth, group, or required capability.

        ``group`` filters and does not resolve: it answers "which connections are
        in this group", where :meth:`select` answers "which one would run". A
        group nothing claims yields nothing rather than raising, because that is
        what every other filter here does with a value nothing matches.

        ``enabled`` is the one filter that is **not** applied by default, and
        deliberately: ``find`` is how a caller inspects its store, and silently
        omitting the connections somebody switched off would make a disabled
        connection look deleted. Pass ``enabled=True`` to narrow to the ones a
        run could use.
        """
        want_vendor = vendor.strip().lower() if vendor is not None else None
        want_runtime = Runtime(runtime) if runtime is not None else None
        want_mode = AuthMode(auth_mode) if auth_mode is not None else None
        out = []
        for connection in self.connections():
            if want_vendor is not None and connection.vendor != want_vendor:
                continue
            if want_runtime is not None and connection.runtime is not want_runtime:
                continue
            if want_mode is not None and connection.auth_mode is not want_mode:
                continue
            if group is not None and not connection.in_group(group):
                continue
            if enabled is not None and connection.enabled is not enabled:
                continue
            if capability is not None and not self.registry.supports(
                connection.runtime, capability
            ):
                continue
            out.append(connection)
        return tuple(out)

    def adapter_for(self, runtime: Runtime) -> Adapter:
        """The adapter for a runtime; injected adapters win over the lazy registry.

        A lazily-loaded adapter is **bound to this bridge's resolution sources**
        before it is handed back (2026-09-13, ticket 1.6). The agent adapters do
        not need it -- their credential travels in ``PreflightPlan.env``, which
        the bridge already built from :attr:`env`. An API adapter has no such
        mapping by design: it calls
        :func:`~modelpass.preflight.resolve_credential` itself at the moment of
        use, and without this it would resolve against the *ambient* environment
        and the secrets file in the user's real home -- so a bridge pointed at a
        test root, or at an application's own directory, would read somebody
        else's keys and bill somebody else's account.

        Duck-typed rather than a contract method, because only the runtimes that
        resolve their own credential have anything to bind. Injected adapters are
        deliberately left alone: a caller who constructed the adapter chose its
        sources, and overwriting them here would silently undo that.
        """
        adapter = self._adapters.get(runtime)
        if adapter is None:
            adapter = load_adapter(runtime)
            bind = getattr(adapter, "bind_resolution", None)
            if callable(bind):
                bind(env=self.env, secrets=self.secrets)
            self._adapters[runtime] = adapter
        return adapter

    def registry_for(self, connection: Connection | str) -> CapabilityRegistry:
        """The capability registry as this *connection's* endpoint has been proved.

        :attr:`registry` answers "what can I rely on from this runtime", which is
        the right question everywhere the runtime's row is a fact about a vendor
        -- which is everywhere but one. ``openai-compatible`` names an API
        *shape*, its row is ``unverified`` by construction, and what a caller can
        rely on is whatever ``modelpass verify`` observed the configured endpoint
        doing. Those observations live on the connection
        (:attr:`~modelpass.connections.Connection.verified_capabilities`) and
        this is where they are folded in, through
        :meth:`~modelpass.capabilities.CapabilityRegistry.refine` -- the hook
        that has existed since the Anthropic init array and has had exactly one
        caller until now (ticket 1.10).

        **It returns :attr:`registry` itself when there is nothing to fold**, and
        a *copy* when there is. Both halves matter: no allocation on the path
        every other runtime takes, and no mutation of the shared read-mostly
        table when one connection's endpoint turns out to do something another's
        does not.
        """
        resolved = self._resolve(connection)
        observed = resolved.verified_capabilities.observed
        if not observed:
            return self.registry
        return self.registry.refined(resolved.runtime, observed)

    # --- preflight -------------------------------------------------------------

    def plan(self, connection: Connection | str) -> PreflightPlan:
        """The vendor-independent launch plan: scrubbed environment and directives."""
        resolved = self._resolve(connection)
        return plan_launch(resolved, self.env)

    def preflight(
        self,
        connection: Connection | str,
        *,
        messages: Iterable[Message | Mapping[str, Any]] | None = None,
        model: str | None = None,
        schema: Mapping[str, Any] | None = None,
        sampling: Sampling | Mapping[str, Any] | None = None,
    ) -> Receipt:
        """Run the full preflight and return the receipt, without starting a run.

        ``model`` is the model the caller intends to pass to :meth:`chat`. It is
        accepted here so that "which model will this run ask for" is answerable
        by the same call that answers "which subscription will pay for it" --
        the receipt reports it either way, but a receipt taken for a run that
        names a model should name the same one.

        ``schema`` is accepted for the same reason: on ``openai-sdk`` a receipt
        taken for a schema-bound run reports whether the schema is in the strict
        subset that runtime requires, which is a thing worth learning before the
        run rather than from its terminal.

        The cache disclosure is attached here too, and reads against whatever
        ``messages`` carries: a receipt taken with no messages says there is no
        system prompt, which is true of the call it describes. Pass the system
        message to ask "would *this* prompt be cacheable" without running it
        (D20).
        """
        resolved = self._resolve(connection)
        request = self._build_request(
            resolved,
            normalize_messages(messages) if messages else (),
            model=model,
            options={},
            sampling=Sampling.coerce(sampling),
            schema=normalize_schema(schema) if schema is not None else None,
        )
        adapter = self.adapter_for(resolved.runtime)
        receipt = self._with_identity_verification(
            resolved, adapter.preflight(request)
        )
        receipt = self._with_cache_disclosure(adapter, request, receipt)
        receipt = self._with_cache_breakpoints(request, receipt)
        receipt = self._with_reasoning_disclosure(request, receipt)
        receipt = self._with_prompt_cache(request, receipt)
        return self._with_sampling_disclosure(request, receipt)

    def refresh_identity(self, connection: Connection | str) -> None:
        """Drop this runtime's cached account-identity probe before the next preflight.

        Adapters cache the vendor identity probe (``claude auth status``,
        Codex's ``account/read``) for a short window, because it is a
        multi-second subprocess and every single run goes through a preflight
        that wants it. One caller must never be served that cache: ``modelpass
        verify``, whose entire job is to read the live identity and pin it. It
        calls this first.
        """
        resolved = self._resolve(connection)
        self.adapter_for(resolved.runtime).invalidate_identity_cache()

    def validate(self, connection: Connection | str) -> ValidationReport:
        """Check everything about a connection that can be checked **offline**.

        The half of preflight that needs no vendor package, no credential store
        and no subprocess: the connection exists and parses, its runtime is real
        and not gated, its auth mode is one that runtime can drive, its
        credential reference resolves against the environment, and its guards
        are coherent. Plus whether the vendor package is *installed* -- answered
        without importing it, exactly as :meth:`Adapter.is_available` promises.

        **Weaker than :meth:`preflight`, deliberately and by name.** It cannot
        tell you which auth mode a run would actually use, whether the login has
        expired, or what the runtime's default model is -- every one of those
        needs the vendor. A green :meth:`validate` means "this configuration is
        not obviously broken", not "this will run", and anything that treats it
        as the latter has made the mistake this method's docstring exists to
        prevent.

        It exists because a host application wants to validate its own config
        file on a machine where the SDK may not be installed, and until now the
        only way to ask was a call that imports a vendor package (second-consumer
        feedback, 2026-08-17). Returns a report rather than raising, because a
        config validator wants every problem at once, not the first one.
        """
        report = ValidationReport(connection=str(connection))
        try:
            resolved = self._resolve(connection)
        except SubpassError as exc:
            return replace(report, problems=(str(exc),))

        problems: list[str] = []
        notes: list[str] = []
        report = replace(
            report,
            connection=resolved.name,
            runtime=resolved.runtime,
            auth_mode=resolved.auth_mode,
            enabled=resolved.enabled,
        )
        if not resolved.enabled:
            notes.append("connection is disabled (enabled = false); it will refuse runs")
        # A key written by a newer build is not a fault of this connection, so it
        # is a note: the configuration is sound, and modelpass is saying out loud
        # which part of it it is carrying rather than reading.
        try:
            notes.extend(self.store.compatibility().notes_for(resolved.name))
        except SubpassError:
            pass

        # The connection's own drive, where it has one: on openai-compatible the
        # table's row is about an API shape and this connection's endpoint is the
        # thing being validated (ticket 1.10). Identical to self.registry
        # everywhere else.
        registry = self.registry_for(resolved)
        try:
            registry.require(resolved.runtime, Capability.CHAT)
        except SubpassError as exc:
            problems.append(str(exc))
        if resolved.auth_mode not in runtime_auth_modes(resolved.runtime, registry):
            problems.append(
                f"runtime {resolved.runtime.value!r} cannot drive auth mode "
                f"{resolved.auth_mode.value!r}"
            )

        try:
            plan_launch(resolved, self.env).require_credential()
        except SubpassError as exc:
            problems.append(str(exc))

        policy = resolved.guards.on_quota_exhausted
        if policy.action is QuotaAction.FAILOVER:
            try:
                self._resolve_failover(resolved, resolved.guards, None)
            except SubpassError as exc:
                problems.append(str(exc))
        if not resolved.guards.configured:
            notes.append("no spend guards configured")
        elif not registry.supports(resolved.runtime, Capability.INTERIM_USAGE):
            # A guard is only a guard where usage arrives mid-run. Saying "guards
            # configured" and stopping there lets a caller read a knob that can
            # only report after the fact as one that can interrupt -- and a
            # consumer did exactly that, then recommended setting a stop on a
            # runtime where it cannot fire in time.
            notes.append(
                f"spend guards are configured, but runtime {resolved.runtime.value!r} "
                "reports usage only when a run completes (interim_usage), so a stop "
                "bounds the NEXT run rather than interrupting this one"
            )

        # is_available() is contractually a probe, not an import (adapter
        # contract), which is what makes this whole method usable on a machine
        # without the SDK.
        available = True
        try:
            available = bool(self._adapter_class(resolved.runtime).is_available())
        except RuntimeNotAvailable as exc:
            available = False
            notes.append(str(exc))
        except SubpassError as exc:
            problems.append(str(exc))
        if not available:
            # "No adapter exists" and "the vendor package is not installed" both
            # arrive as RuntimeNotAvailable, and they are opposite verdicts: one
            # is a machine that has not run pip yet, the other is a runtime
            # modelpass cannot drive at all. Calling the second one "still valid"
            # tells a caller to go installing something that would not help.
            if not self._adapter_exists(resolved.runtime):
                problems.append(
                    f"modelpass has no adapter for runtime {resolved.runtime.value!r}; "
                    "this is not an install away"
                )
            else:
                notes.append(
                    f"the vendor package for runtime {resolved.runtime.value!r} is not "
                    "installed here; the configuration is still valid"
                )

        return replace(
            report,
            runtime_installed=available,
            problems=tuple(problems),
            notes=tuple(notes),
        )

    @staticmethod
    def _with_identity_verification(
        connection: Connection, receipt: Receipt
    ) -> Receipt:
        """Fail closed when the live vendor identity differs from the pin.

        Idempotent on purpose. A session's receipt goes through here twice --
        once inside the adapter-shaped preflight and once on the session request
        -- and the unpinned note below must not appear twice for it. Comparing
        against the note already present is the whole of the fix, and it costs
        nothing on the ordinary single-pass path.
        """
        binding = connection.account_binding
        if binding is None:
            if not connection.is_subscription or _UNPINNED_NOTE in receipt.notes:
                return receipt
            return replace(receipt, notes=(*receipt.notes, _UNPINNED_NOTE))
        if not receipt.ok:
            return receipt
        profile = receipt.account_profile
        if profile is None:
            return replace(
                receipt,
                identity_verified=False,
                ok=False,
                problem=(
                    f"{connection.display_name!r} is pinned to a verified account, but "
                    "the vendor identity probe returned nothing, so Subpass cannot "
                    "confirm the same account would pay. Check the runtime is logged "
                    f"in, then re-pin with 'modelpass verify {connection.name}'"
                ),
            )

        mismatches: list[str] = []
        actual_email = profile.email.strip().casefold() if profile.email else None
        if binding.email and actual_email != binding.email:
            mismatches.append(
                f"email expected {binding.email!r}, vendor reported {actual_email!r}"
            )
        if (
            binding.organization_id
            and profile.organization_id != binding.organization_id
        ):
            mismatches.append(
                "organization ID expected "
                f"{binding.organization_id!r}, vendor reported "
                f"{profile.organization_id!r}"
            )
        if mismatches:
            return replace(
                receipt,
                identity_verified=False,
                ok=False,
                problem=(
                    f"ACCOUNT IDENTITY MISMATCH for {connection.display_name!r}: "
                    + "; ".join(mismatches)
                    + ". No model request was started"
                ),
            )
        return replace(receipt, identity_verified=True)

    @staticmethod
    def _with_cache_disclosure(
        adapter: Adapter, request: RunRequest | SessionRequest, receipt: Receipt
    ) -> Receipt:
        """Attach the adapter's cache-eligibility answer to a receipt (D20).

        Attached here rather than inside each adapter's ``preflight`` because
        every adapter has several return points -- runtime missing, no login,
        expired token -- and a disclosure about caching on a receipt that failed
        before there was anything to cache is noise. One place, one rule: a
        receipt gets the line, whatever else it says.

        Best-effort in the same sense :meth:`_record_run` is. This is a
        disclosure, not a guarantee, and an adapter that raises while working
        out its own caching rules must not take down a run that would otherwise
        have gone ahead -- a receipt missing one line is a smaller failure than
        a call that never happened. ``SubpassError`` is deliberately *not*
        excepted: a guaranteed-layer refusal raised in here is still a refusal.
        """
        try:
            eligibility = adapter.cache_eligibility(request)
        except SubpassError:
            raise
        except Exception:
            return receipt
        if eligibility is None:
            return receipt
        return replace(receipt, cache=eligibility)

    def _with_cache_breakpoints(
        self, request: RunRequest | SessionRequest, receipt: Receipt
    ) -> Receipt:
        """Record what the caller asked for, and what this runtime will take (R3).

        The counterpart to :meth:`_with_cache_disclosure` and deliberately a
        separate step: that one asks an *adapter* a vendor question, and this one
        asks the *registry* a table question -- whether this runtime has a
        breakpoint channel at all
        (:attr:`~modelpass.capabilities.Capability.CACHE_BREAKPOINTS`). Putting
        it here rather than in each adapter is what makes the honesty uniform:
        an adapter that has never heard of content blocks still produces a
        receipt that says what happened to them.

        The note is the whole point. Before this, a caller's ``cache_control``
        markers were stripped at the door with nothing anywhere reporting it --
        a consumer built a four-breakpoint prompt, paid full price on every call,
        and the library said nothing. A dropped breakpoint is now a named line on
        the receipt that precedes the run, on the same terms as every other
        disclosure here: stated as a fact, not as a fault.
        """
        requested = request.cache_breakpoints
        if not requested:
            return receipt
        honours = self.registry.supports(
            request.connection.runtime, Capability.CACHE_BREAKPOINTS
        )
        notes = receipt.notes
        if not honours:
            notes = (
                *notes,
                "cache breakpoints were requested and dropped: "
                f"{request.connection.runtime.value} does not accept them",
            )
        cache = receipt.cache
        if cache is not None:
            cache = replace(
                cache,
                explicit_breakpoints=requested,
                breakpoints_honoured=honours,
            )
        return replace(
            receipt,
            notes=notes,
            cache=cache,
            cache_breakpoints_requested=requested,
            cache_breakpoints_honoured=requested if honours else 0,
        )

    @staticmethod
    def _with_prompt_cache(
        request: RunRequest | SessionRequest, receipt: Receipt
    ) -> Receipt:
        """Say what the connection asked about caching, and what it bought (2026-09-21).

        A fourth disclosure step beside :meth:`_with_cache_disclosure`,
        :meth:`_with_cache_breakpoints` and :meth:`_with_sampling_disclosure`,
        and here rather than in an adapter for the reason the middle two are:
        the answer comes from the *table*, so an adapter that has never heard of
        the key still produces a receipt that reports it.

        Silent on every connection that stated nothing, which is the difference
        from the sampling step: a required field nobody set is a fact about a
        run, but a caching request nobody made is not -- and a line reading
        "prompt caching: not requested" on every receipt ever printed would be
        noise standing where a disclosure should be.

        Cannot refuse. The refusal happened when the connection was built; by
        the time a request exists, the plan is known to be meetable.
        """
        stated = request.connection.prompt_cache
        if stated is None:
            return receipt
        try:
            plan = plan_prompt_cache(
                stated, request.connection.runtime, name=request.connection.name
            )
        except SubpassError:  # pragma: no cover - defensive: refused at build
            return receipt
        return replace(
            receipt,
            notes=(*receipt.notes, plan.note),
            prompt_cache_requested=plan.requested,
            prompt_cache_disposition=plan.disposition.value,
        )

    @staticmethod
    def _with_reasoning_disclosure(
        request: RunRequest, receipt: Receipt
    ) -> Receipt:
        """Say what effort was asked for and what the runtime is actually told.

        Here rather than in an adapter for the reason the cache and sampling
        steps are: the answer comes from a *table*, so every runtime reports it
        the same way and an adapter that has never heard of the key still
        produces a receipt that carries it.

        This is the only place the answer exists on two of the runtimes.
        ``anthropic-sdk`` sends the level and gets no echo -- anthropic 0.97.0's
        ``Message`` has no effort field and the effort documentation describes
        none -- so a caller who cannot read it here cannot read it anywhere.
        ``openai-sdk`` does echo it, on ``thread/start``, which is why
        :func:`~modelpass.adapters.openai.thread_reasoning_effort` exists to
        compare the two.

        Silent when nothing was stated. Cannot refuse: that happened when the
        connection was built.
        """
        plan = stated_reasoning(request.connection)
        if plan is None:
            return receipt
        return replace(
            receipt,
            notes=(*receipt.notes, plan.note),
            reasoning_requested=plan.requested.value,
            reasoning_applied=plan.applied.value,
            reasoning_value=plan.runtime_value,
        )

    @staticmethod
    def _with_sampling_disclosure(
        request: RunRequest, receipt: Receipt
    ) -> Receipt:
        """Say what this run asked about word choice, and what will be sent (R5).

        A third disclosure step beside :meth:`_with_cache_disclosure` and
        :meth:`_with_cache_breakpoints`, and here for the same reason the second
        one is: the answer comes from a *table* rather than from a vendor, so
        putting it here makes the honesty uniform. An adapter that has never
        heard of sampling -- both agent runtimes -- still produces a receipt
        that says, by name, which of the caller's fields went nowhere.

        It runs on every call, not only on calls that asked for something,
        because a required field nobody set is also a fact about the run: an
        ``anthropic-api`` call carries a 4096-token ceiling whether or not
        anyone chose it, and a receipt that mentioned it only when asked would
        be the silent default this whole ticket is about.
        """
        plan = plan_sampling(
            request.effective_sampling,
            request.connection.runtime,
            request.model or request.connection.model,
        )
        if not plan.requested and not plan.applied:
            return receipt
        return replace(
            receipt,
            sampling_requested=plan.requested,
            sampling_applied=plan.applied,
            sampling_notes=plan.notes,
        )

    @staticmethod
    def _option_note(adapter: Adapter, options: Mapping[str, Any] | None) -> str | None:
        """A note naming options this runtime will not read, or ``None``.

        Silently ignoring an unknown key is the worst behaviour available for
        ``codex_bin`` in particular: it is the documented lever for which binary
        actually runs, so an ignored override is indistinguishable from a
        working one. Adapters that forward unknown keys to their vendor SDK
        declare no closed set and are never second-guessed here.
        """
        unknown = adapter.unknown_option_keys(options)
        if not unknown:
            return None
        known = ", ".join(sorted(adapter.option_keys or ()))
        return (
            f"options {', '.join(repr(k) for k in unknown)} will be ignored: runtime "
            f"{adapter.runtime.value!r} reads only {known}"
        )

    @staticmethod
    def _adapter_exists(runtime: Runtime) -> bool:
        """Whether modelpass ships an adapter for this runtime at all.

        Distinct from whether its vendor package is installed. The first is a
        fact about modelpass; the second is a fact about this machine.
        """
        from .adapters import ADAPTER_REGISTRY

        return runtime in ADAPTER_REGISTRY

    def _adapter_class(self, runtime: Runtime) -> Any:
        """The adapter for availability probing, without constructing a run.

        An injected adapter answers for itself (a test double should be able to
        say it is unavailable); otherwise the registry's lazy loader is asked,
        which raises rather than importing a missing vendor package.
        """
        injected = self._adapters.get(runtime)
        if injected is not None:
            return injected
        return load_adapter(runtime)

    # --- chat ------------------------------------------------------------------

    def chat(
        self,
        *,
        connection: Connection | str,
        message: str | Sequence[TextBlock],
        system_prompt: str | Sequence[TextBlock] | None = None,
        history: Iterable[Message | Mapping[str, Any]] | None = None,
        model: str | None = None,
        expect_auth_mode: AuthMode | str | None = None,
        guards: Guards | None = None,
        stop_at_tokens: int | None = None,
        warn_at_tokens: int | None = None,
        allow_failover: bool | None = None,
        raise_on_stop: bool = False,
        options: Mapping[str, Any] | None = None,
        sampling: Sampling | Mapping[str, Any] | None = None,
        tools: Iterable[ToolDef | Mapping[str, Any]] | None = None,
        mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
        schema: Mapping[str, Any] | None = None,
        schema_name: str | None = None,
        timeout: Timeout | float | None = None,
    ) -> Iterator[AgentEvent]:
        """One stateless call: instructions, prior turns, a message, an answer.

        **You pass everything in.** This call keeps nothing between invocations:
        the new user turn is ``message``, the instructions are ``system_prompt``,
        and any prior turns are ``history``, which the *caller* owns. Underneath,
        modelpass renders those per runtime -- a lone user turn, or instructions
        above a lone user turn, are passed through as themselves, and only a real
        history becomes a transcript.

        There is exactly one other door, and it is the stateful one:
        :meth:`Session.send`, from :meth:`new_chat` or :meth:`new_worker`, where
        the *runtime* holds the history and the prefix cache is native. Two
        verbs, one distinction -- who is holding the conversation (D21).

        ``history`` takes the same message shapes a list-shaped API does --
        :class:`~modelpass.types.Message` objects or ``{"role": ..., "content":
        ...}`` dicts -- and is assembled ahead of ``message`` in the order given.
        Omitting it is the single-shot case and is what most callers want:
        score this, classify that, rewrite the other.

        ``system_prompt`` and ``message`` each take **either a string or a
        sequence of** :class:`~modelpass.types.TextBlock` (R3). The block form
        exists for one thing: putting a ``cache_control`` breakpoint where you
        want the cacheable prefix to end, instead of leaving modelpass and the
        runtime to guess. A string is unchanged in every respect -- same bytes,
        same rendering, same receipt -- so nothing about an existing call moves.
        Whether the breakpoints reach the vendor is
        :attr:`~modelpass.capabilities.Capability.CACHE_BREAKPOINTS`, and where
        they do not, the content is flattened to text and the receipt says so by
        name rather than dropping them quietly.

        **Keep ``system_prompt`` here rather than concatenating it into
        ``message``.** It is the front of the cached prefix, and holding it
        stable across a loop is the difference between every call after the
        first reading from cache and every call paying full price. Two
        runtime-specific facts about it are worth knowing, both in the
        capability registry: on ``anthropic-sdk`` it *replaces* the runtime's
        own prompt, and on ``openai-sdk`` it *layers on top of* Codex's
        hardcoded coding-agent persona, because that runtime has no
        system-prompt parameter to replace.

        **An absent ``system_prompt`` means the runtime's minimal prompt, with
        or without ``tools``, and that is now a decision rather than an
        oversight (D22, 2026-08-30).** Omitting it passes ``system_prompt=None``
        to the Agent SDK, which yields tool-calling support and nothing else --
        no coding-agent persona, no environment context, and none of the
        ``claude_code`` preset's safety instructions, which the vendor documents
        as tool-usage guidance bundled with that environment context rather than
        as anything the model's own behaviour rests on. modelpass writes no prompt
        of its own here: putting words in a model's mouth on a caller's behalf
        is not something this library does uninvited.

        The tempting alternative -- map ``tools=`` onto the preset, the way a
        ``WorkerSession`` does -- is wrong for this call, and the reason is that
        the two situations are not the same one. A worker switches the
        *runtime's own* toolbelt on and would otherwise hold a shell and a file
        editor with no instructions for either. A ``chat`` call never does:
        built-in tools are off by construction here, so ``tools=`` means the
        caller's own functions and nothing else. Handing that a prompt written
        for Bash, Read and Edit would be guidance for a toolbelt that is not
        present, wrapped around an identity the caller did not ask for -- and
        the preset also embeds the working directory, platform, shell and OS
        version, which would scope the cached prefix to one machine and one
        folder and inflate a measured 171-token floor into thousands, on exactly
        the scoring loops this call exists for.

        Passing a ``system_prompt`` is how you get a persona, on either kind of
        call. :meth:`new_worker` is how you get the runtime's.

        **"Off by construction" became true on both runtimes on 2026-08-31
        (D23), and the sentence above predates that.** ``anthropic-sdk`` had
        kept it since Phase 3 (``tools=[]``); ``openai-sdk`` had not -- every
        stateless call ran with Codex's whole coding-agent belt attached, on
        both transports, worth a measured 5,621 prompt tokens. It is now
        enforced on both, and ``options={"native_tools": True}`` is the opt-in
        for a caller who genuinely wants the runtime's own toolbelt on a
        one-shot call. That option is read on ``openai-sdk`` only; see
        :attr:`~modelpass.adapters.base.RunRequest.native_tools`.

        **``sampling=`` asks; the receipt answers** (R5, ticket 1.7).
        ``sampling=Sampling(temperature=0.0, max_output_tokens=256)`` carries
        ``temperature``, ``top_p``, ``top_k``, ``max_output_tokens`` and
        ``reasoning_effort``. What of it actually goes depends on the runtime
        **and on the model** -- GPT-5 accepts only ``temperature=1.0``, several
        Anthropic 4.x models accept one of temperature/top_p, the o-series
        accepts none -- so this call promises nothing and reports everything:
        ``sampling_requested``, ``sampling_applied`` and ``sampling_notes`` on
        the receipt, one sentence per field that was dropped or coerced, filled
        before the run from :mod:`modelpass.sampling_rules` rather than from a
        vendor.

        Passing it to a runtime that takes none of it -- both agent runtimes --
        is **not an error**: every field is dropped and named. That is the
        asymmetry with ``tools=``, which refuses, and the reason for it is that
        a tool has no partial answer and a temperature does.

        ``max_output_tokens`` is a ceiling on **one answer**, which the vendor
        enforces by truncating. ``stop_at_tokens`` is a guard over the **whole
        run's** total spend, which ends the stream. Set both if you want both.

        **``timeout=`` bounds the wall clock** (R6, ticket 1.12).
        ``timeout=Timeout(total=120, first_token=20)`` says the whole run gets
        two minutes and the wait for the first event that is not the receipt
        gets twenty seconds; a bare number is ``total``. On expiry the run is
        cancelled through the adapter's own ``cancel()`` hook -- the D10 floor --
        and the stream still ends the way every stream ends: one terminal event,
        stamped, carrying what was spent before the bound ran out, with
        ``status`` :attr:`~modelpass.types.TerminalStatus.TIMED_OUT`.
        ``raise_on_stop=True`` turns that into
        :class:`~modelpass.errors.RunTimedOut`, as it does for the other two
        stops. Passing nothing uses the connection's ``timeoutSeconds`` where it
        has one, and is otherwise unbounded, which is what every call did before
        this argument existed.

        A bound is not a retry. **modelpass never retries anything** -- every
        terminal and every error carries a
        :class:`~modelpass.types.Retryable` verdict and the caller decides,
        because the caller is the one whose allowance pays for the second
        attempt.

        .. deprecated::
           ``options={"max_output_tokens": N}``, ticket 1.6's stopgap, still
           works as an alias for ``sampling=Sampling(max_output_tokens=N)`` and
           does not warn. Prefer the request field; the option goes away with
           the consumer migrations.

        Caching is not guaranteed and not configurable here. It applies when the
        prompt is byte-stable between calls **and** clears the runtime's minimum
        cacheable prefix -- 512 tokens on Opus 5, 1024 on most others.

        **Read the reads, not the writes.** ``cached_input_tokens`` at zero
        across calls that should share a prefix means nothing is being reused,
        whatever the writes say. ``cache_write_tokens`` alone is ambiguous: it
        is zero when the whole prompt repeats byte-for-byte, and large when it
        does not, so both a perfect hit and a total miss can show a number a
        reader might take for health.

        A short system prompt does **not** on its own buy caching across varying
        messages. Measured on ``claude-sub`` / Sonnet, 2026-08-30: a 33-token
        prompt with items differing by 10-25 tokens paid a fresh ~1,850-token
        write on every call and never read; the same prompt with a *byte-identical*
        message read 1,846 and wrote nothing, three calls running. Once the system
        block cleared the floor -- 1,709 tokens -- reads of 2,790 appeared on
        later calls against *different* messages, which is the behaviour a scoring
        loop wants.

        So the practical rule for a loop: a rubric materially below the floor
        buys nothing from holding it stable, because the reusable unit is larger
        than the rubric. Growing it past the floor is what turns a stable prefix
        into a reused one -- and is usually something a judge prompt wants anyway.

        *Nine calls, one machine, one model.* The segment arithmetic does not
        fully reconcile across those arms, so treat "a breakpoint forms at the
        system block once it clears the floor" as the shape the data supports
        rather than a mechanism anyone has proven. What is solid is the signal:
        reads.

        ``expect_auth_mode`` is an assertion, not an override: a connection binds
        runtime, auth mode and credential together, so "switch to API for this
        call" means naming a different connection (D3). Asserting the mode lets a
        caller refuse to run if the connection is not what it thought.

        ``guards`` overrides the connection's thresholds for this call only and
        is never written back to config. If those guards carry an
        ``onQuotaExhausted`` failover, the target is validated here -- before
        anything runs -- so a typo'd or circular failover is a configuration
        error the caller sees straight away rather than a surprise sprung at the
        moment an allowance runs out.

        ``stop_at_tokens`` / ``warn_at_tokens`` are shorthand for the common
        case, which is "the connection's guards, but a tighter ceiling for this
        one call". They start from the connection's own guards, so an unnamed
        threshold and the quota policy carry across untouched -- lowering a
        ceiling never quietly disarms a configured failover -- and a warn left
        stranded above the new stop is clamped rather than refused
        (:meth:`Guards.for_call`). ``0`` means "no guard for this call".
        Passing them together with ``guards=`` is an error: two ways of saying
        the same thing, one of which would have to silently lose.

        ``allow_failover=False`` declines a connection's configured quota
        failover for this call only -- an exhausted allowance stops cleanly
        instead of crossing onto whatever the connection names next. ``None``,
        the default, respects the connection's configuration. It composes with
        the threshold shorthands (all three start from the connection's own
        guards) and is refused alongside an explicit ``guards=`` for the same
        reason they are: two ways of saying the same thing, one of which would
        have to silently lose. It exists because declining a failover per call
        was otherwise three imports deep -- ``Guards``, ``QuotaPolicy`` and
        ``dataclasses.replace`` -- for a boolean decision (second-consumer
        feedback, 2026-08-17). ``allow_failover=True`` is the default behavior
        stated out loud, not a way to *enable* a failover nobody configured.

        ``tools`` and ``mcp_servers`` hand the runtime a tool loop to run inside
        this one call (D12). The call stays stateless -- one ephemeral runtime
        session, everything serialized in -- but the runtime may take several
        internal turns to work through tool calls, and the caller *observes*
        them as ``tool_call`` / ``tool_result`` events rather than answering
        them. Both are capability-gated below, before the preflight and
        therefore before any spend: asking a runtime for something it cannot do
        must not cost a token to find out.

        ``schema`` binds the final answer to a JSON Schema (D13), using each
        runtime's *native* mechanism -- ``output_format`` on ``anthropic-sdk``,
        ``--output-schema`` on ``openai-sdk``. The stream gains one
        ``structured_output`` event before the terminal. ``schema_name`` is the
        name the schema travels under, defaulting to its ``title``; it matters
        because on ``anthropic-sdk`` the mechanism is a tool call, and a caller
        whose prompts refer to that tool by name is depending on a wire-protocol
        detail modelpass must not invent.

        **``schema`` and ``tools`` / ``mcp_servers`` cannot be combined in v1**,
        and that refusal is deliberate rather than an oversight -- see D13. The
        mechanics conflict on ``openai-sdk`` (``--output-schema`` is reported
        upstream as silently ignored when MCP servers are active,
        openai/codex#15451) and the interaction is unverified on
        ``anthropic-sdk``, where structured output is itself implemented as an
        end-turn tool. A combination that silently worked on one runtime and
        silently did not on the other is worse than one that refuses on both.
        """
        return self._stream(
            self._prepare(
                connection=connection,
                message=message,
                system_prompt=system_prompt,
                history=history,
                model=model,
                expect_auth_mode=expect_auth_mode,
                guards=guards,
                stop_at_tokens=stop_at_tokens,
                warn_at_tokens=warn_at_tokens,
                allow_failover=allow_failover,
                raise_on_stop=raise_on_stop,
                options=options,
                sampling=sampling,
                tools=tools,
                mcp_servers=mcp_servers,
                schema=schema,
                schema_name=schema_name,
                timeout=timeout,
            )
        )

    # --- the async face (R1, ticket 1.13) ---------------------------------------

    def achat(
        self,
        *,
        connection: Connection | str,
        message: str | Sequence[TextBlock],
        system_prompt: str | Sequence[TextBlock] | None = None,
        history: Iterable[Message | Mapping[str, Any]] | None = None,
        model: str | None = None,
        expect_auth_mode: AuthMode | str | None = None,
        guards: Guards | None = None,
        stop_at_tokens: int | None = None,
        warn_at_tokens: int | None = None,
        allow_failover: bool | None = None,
        raise_on_stop: bool = False,
        options: Mapping[str, Any] | None = None,
        sampling: Sampling | Mapping[str, Any] | None = None,
        tools: Iterable[ToolDef | Mapping[str, Any]] | None = None,
        mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
        schema: Mapping[str, Any] | None = None,
        schema_name: str | None = None,
        timeout: Timeout | float | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """:meth:`chat`, on a loop (R1)::

            async for event in bridge.achat(connection="claude-api", message="..."):
                ...

        **Every argument, every gate, every event and every receipt is
        :meth:`chat`'s** -- the two faces share one pre-run pipeline
        (:meth:`_prepare`) and one event fold (:mod:`modelpass._fold`), so they
        cannot drift into two dialects of one contract. Read :meth:`chat` for
        what any of it means; what follows is only what is different about being
        on a loop.

        **Nothing here makes the sync face async.** :meth:`chat` is untouched
        and is still the whole implementation for a sync caller, because an
        async core with a sync wrapper would break every consumer that calls
        from inside a running loop -- the alternative R1 rejects by name.

        **Cancellation is ``aclose()``** (R2, D10), where the sync contract is
        "close the iterator"::

            events = bridge.achat(connection="claude-api", message="...")
            try:
                async for event in events:
                    if enough(event):
                        break
            finally:
                await events.aclose()

        That cancels the run through the adapter's own ``cancel()`` and joins
        the worker thread where the default
        :meth:`~modelpass.adapters.base.Adapter.arun` is using one. Walking away
        without it is best effort: the worker is cancelled when the object is
        collected, which is a moment nobody controls, so a consumer who stops
        early should say so.

        **The refusals arrive in two places, and the split is deliberate.**
        Everything :meth:`chat` refuses without spending -- a disabled
        connection, an ungated capability, a schema beside tools, an auth-mode
        assertion that does not hold, a misconfigured failover -- is raised **by
        this call**, before you iterate, exactly as ``chat`` raises it. What a
        *preflight* discovers -- an unusable credential, a runtime that is not
        installed -- is raised on the **first step**, because the preflight is
        the one step this face hands to a thread: it is a subprocess on the
        agent runtimes, and blocking the loop on it is the bug this whole face
        exists to end.

        **Thread safety** (R7, extended by this ticket): a :class:`Bridge` is
        safe to share, one event loop may run many ``achat`` calls
        concurrently, and a :class:`~modelpass.sessions.Session` still takes one
        caller at a time whichever face that caller uses.
        """
        # Called eagerly, exactly where ``chat`` calls it: a caller mistake is a
        # caller mistake at the call site, not two awaits later.
        plan = self._plan(
            connection=connection,
            message=message,
            system_prompt=system_prompt,
            history=history,
            model=model,
            expect_auth_mode=expect_auth_mode,
            guards=guards,
            stop_at_tokens=stop_at_tokens,
            warn_at_tokens=warn_at_tokens,
            allow_failover=allow_failover,
            raise_on_stop=raise_on_stop,
            options=options,
            sampling=sampling,
            tools=tools,
            mcp_servers=mcp_servers,
            schema=schema,
            schema_name=schema_name,
            timeout=timeout,
        )
        return self._astream(plan)

    async def aask(
        self,
        connection: Connection | str,
        message: str | Sequence[TextBlock],
        *,
        system_prompt: str | Sequence[TextBlock] | None = None,
        history: Iterable[Message | Mapping[str, Any]] | None = None,
        model: str | None = None,
        expect_auth_mode: AuthMode | str | None = None,
        guards: Guards | None = None,
        stop_at_tokens: int | None = None,
        warn_at_tokens: int | None = None,
        allow_failover: bool | None = None,
        options: Mapping[str, Any] | None = None,
        sampling: Sampling | Mapping[str, Any] | None = None,
        tools: Iterable[ToolDef | Mapping[str, Any]] | None = None,
        mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
        schema: Mapping[str, Any] | None = None,
        schema_name: str | None = None,
        timeout: Timeout | float | None = None,
    ) -> Answer:
        """:meth:`ask`, awaited: one call, one :class:`Answer`, no drain loop::

            answer = await bridge.aask("claude-api", "Score this: ...", schema=SCHEMA)

        Same arguments, same gates, same receipts, same run-log line and the
        same three errors on a stop, because this *is* :meth:`achat`, drained by
        the reader :meth:`ask` uses. There is no ``raise_on_stop=`` here either,
        and for the same reason.
        """
        collected = _Collected()
        async for event in self.achat(
            connection=connection,
            message=message,
            system_prompt=system_prompt,
            history=history,
            model=model,
            expect_auth_mode=expect_auth_mode,
            guards=guards,
            stop_at_tokens=stop_at_tokens,
            warn_at_tokens=warn_at_tokens,
            allow_failover=allow_failover,
            raise_on_stop=True,
            options=options,
            sampling=sampling,
            tools=tools,
            mcp_servers=mcp_servers,
            schema=schema,
            schema_name=schema_name,
            timeout=timeout,
        ):
            collected.absorb(event)
        return collected.answer(
            connection=self._resolve(connection).name, schema=schema
        )

    # --- the collected door (R11) ----------------------------------------------

    def ask(
        self,
        connection: Connection | str,
        message: str | Sequence[TextBlock],
        *,
        system_prompt: str | Sequence[TextBlock] | None = None,
        history: Iterable[Message | Mapping[str, Any]] | None = None,
        model: str | None = None,
        expect_auth_mode: AuthMode | str | None = None,
        guards: Guards | None = None,
        stop_at_tokens: int | None = None,
        warn_at_tokens: int | None = None,
        allow_failover: bool | None = None,
        options: Mapping[str, Any] | None = None,
        sampling: Sampling | Mapping[str, Any] | None = None,
        tools: Iterable[ToolDef | Mapping[str, Any]] | None = None,
        mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
        schema: Mapping[str, Any] | None = None,
        schema_name: str | None = None,
        timeout: Timeout | float | None = None,
    ) -> Answer:
        """:meth:`chat`, drained: one call, one :class:`Answer` (R11)::

            answer = bridge.ask("claude-api", "Score this posting: ...", schema=SCHEMA)
            posting.score = answer.structured["score"]

        Same arguments, same gates, same receipts, same run-log line -- this
        *is* :meth:`chat`, with the loop written once here instead of once at
        every call site. Reach for :meth:`chat` when you want to render an
        answer as it arrives, show a tool call the moment it happens, or stop a
        run partway through; reach for this when what you want is the answer.

        **A terminal that is not ``ok`` raises.** A guard stop is
        :class:`~modelpass.errors.GuardStop`, an exhausted allowance is
        :class:`~modelpass.errors.QuotaExhausted`, and a vendor failure is
        :class:`~modelpass.errors.VendorRunFailed` -- the same three a
        ``chat(raise_on_stop=True)`` caller already handles. That is not a
        second error policy; it is the only one that makes sense here. A
        streaming caller who sees ``status=error`` on the terminal has already
        rendered the half answer above it and knows exactly what they have. A
        caller holding one returned object does not, and an :class:`Answer` with
        the first two sentences of a refusal in ``text`` would be indexed,
        scored and written to a database as though the run had finished.

        There is **no** ``raise_on_stop=`` here, deliberately. It would be a
        keyword whose ``False`` this method cannot honour.

        The one status that cannot reach this method is ``cancelled``:
        cancellation is what closing the iterator means (D10), and nothing
        closes this one early.

        ``schema=`` makes ``Answer.structured`` the parsed object. On an ``ok``
        run it is never ``None`` -- a run that produced nothing parseable ends
        ``status=error`` and has already raised by then, and the remaining case
        (an adapter that ended ``ok`` and emitted no ``structured_output``
        event) is a broken contract and is raised as
        :class:`~modelpass.errors.AdapterFailed` rather than handed over as an
        empty result that looks exactly like a real one.

        ``schema=`` with ``tools=`` is still refused (D13), because the refusal
        lives in :meth:`chat` and this method delegates rather than
        re-implementing.

        ``connection`` and ``message`` may be positional here, where every
        :meth:`chat` argument is keyword-only. The asymmetry is on purpose and
        it is the whole ergonomic: a one-shot call site reads as one line, and
        the two arguments in front of it are the two nobody ever mistakes for
        each other.
        """
        collected = _Collected()
        for event in self.chat(
            connection=connection,
            message=message,
            system_prompt=system_prompt,
            history=history,
            model=model,
            expect_auth_mode=expect_auth_mode,
            guards=guards,
            stop_at_tokens=stop_at_tokens,
            warn_at_tokens=warn_at_tokens,
            allow_failover=allow_failover,
            # Permanently on: the guard stop and the quota stop raise from the
            # one place that holds the tracker and the thresholds, so the errors
            # this method raises carry the same numbers a chat caller's do.
            raise_on_stop=True,
            options=options,
            sampling=sampling,
            tools=tools,
            mcp_servers=mcp_servers,
            schema=schema,
            schema_name=schema_name,
            timeout=timeout,
        ):
            collected.absorb(event)
        return collected.answer(
            connection=self._resolve(connection).name, schema=schema
        )

    # --- sessions (D14-D17) ----------------------------------------------------

    def new_chat(
        self,
        *,
        connection: Connection | str,
        model: str | None = None,
        system_prompt: str | None = None,
        persist: bool = True,
        project_folder: str | None = None,
        tools: Iterable[ToolDef | Mapping[str, Any]] | None = None,
        mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
        guards: Guards | None = None,
        stop_at_tokens: int | None = None,
        warn_at_tokens: int | None = None,
        expect_auth_mode: AuthMode | str | None = None,
        options: Mapping[str, Any] | None = None,
        raise_on_stop: bool = False,
    ) -> ChatSession:
        """Open a conversation with the runtime's own agent persona stripped out.

        Construction runs the preflight and every capability assertion, and
        returns a local object with a receipt already available. **Nothing is
        spent and no session exists yet**: neither runtime has a create-session
        call, so the session comes into existence when the first
        :meth:`Session.send` completes, which is also when :attr:`Session.id`
        stops being ``None``.

        ``system_prompt`` **replaces** whatever persona the runtime ships with,
        which is the point of this object and the one thing ``openai-sdk``
        cannot do -- Codex has no system-prompt parameter and layers a
        ``System:`` line on top of its hardcoded coding-agent persona instead.
        So this argument raises there rather than degrading, and
        :meth:`Session.help` leads with the reason.

        ``tools`` and ``system_prompt`` are settable **here and nowhere else**.
        They are the front of the cached prefix (``tools`` -> ``system`` ->
        ``messages``, matched exactly), so varying them per turn is the most
        cache-destructive thing available; making that unexpressible beats
        documenting it as discouraged (D17).

        ``persist=False`` means nothing is written down and the session cannot
        be resumed from another process. It is not merely a storage flag: on
        ``anthropic-sdk`` a live client still carries the conversation across
        turns, so multi-turn survives, while on ``openai-sdk`` continuation
        needs the rollout file and there would be no multi-turn at all. Hence
        the ``ephemeral_multi_turn`` gate below -- an object that called itself
        a chat and behaved like a series of one-shots would smuggle the
        transcript flattening D14 removes back in through the side door.

        ``project_folder`` defaults to a fresh modelpass-owned directory rather
        than the caller's process cwd. With ``persist=True`` as the default,
        inheriting the cwd would write agent transcripts into the consumer's own
        ``~/.claude/projects/<their-cwd>/``, where their ``claude --continue``
        would find them (D15).

        Guards are the session's envelope, not the turn's: ``stop_at_tokens``
        bounds everything this session will ever spend, which is what makes it a
        budget per surface (D18). The shorthands and ``guards=`` behave exactly
        as they do on :meth:`chat`, including the refusal to accept both.

        Refused outright on a runtime that holds no conversation of its own --
        the four API runtimes -- before an adapter is loaded. See
        :meth:`_require_sessions`.
        """
        self._require_sessions(self._resolve(connection).runtime, "bridge.new_chat()")
        return self._open_session(
            ChatSession,
            connection=connection,
            model=model,
            system_prompt=system_prompt,
            persist=persist,
            project_folder=project_folder,
            tools=tools,
            mcp_servers=mcp_servers,
            guards=guards,
            stop_at_tokens=stop_at_tokens,
            warn_at_tokens=warn_at_tokens,
            expect_auth_mode=expect_auth_mode,
            options=options,
            raise_on_stop=raise_on_stop,
        )

    def new_worker(
        self,
        *,
        connection: Connection | str,
        project_folder: str,
        model: str | None = None,
        system_prompt: str | None = None,
        persist: bool = True,
        tools: Iterable[ToolDef | Mapping[str, Any]] | None = None,
        mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
        guards: Guards | None = None,
        stop_at_tokens: int | None = None,
        warn_at_tokens: int | None = None,
        expect_auth_mode: AuthMode | str | None = None,
        options: Mapping[str, Any] | None = None,
        raise_on_stop: bool = False,
    ) -> WorkerSession:
        """Open a conversation with the runtime's own agent kept, pointed at a folder.

        The mirror of :meth:`new_chat`, differing only in policy: native tools
        stay on, ``system_prompt`` is **appended** to the runtime's persona
        instead of replacing it, and ``project_folder`` is required -- a worker
        whose toolbelt is aimed at a scratch directory is useless.

        This is where the two runtimes genuinely agree. Append is the only
        system-prompt semantics ``openai-sdk`` has, and here it is the semantics
        you wanted, so a worker is not capability-gated on
        ``system_prompt_replace`` and :meth:`Session.help` does not mention it.

        Everything else -- construction-time preflight, the id timing, immutable
        tools and prompt, session-wide guards -- is :meth:`new_chat`'s, because
        it is literally the same implementation.

        Two refusals guard it, both before anything is launched: a runtime that
        holds no conversation refuses every session door
        (:meth:`_require_sessions`), and a runtime with no toolbelt of its own
        refuses *this* door specifically (:meth:`_require_native_toolbelt`),
        because a worker without one would be a chat wearing a worker's name.
        """
        self._require_sessions(self._resolve(connection).runtime, "bridge.new_worker()")
        return self._open_session(
            WorkerSession,
            connection=connection,
            model=model,
            system_prompt=system_prompt,
            persist=persist,
            project_folder=project_folder,
            tools=tools,
            mcp_servers=mcp_servers,
            guards=guards,
            stop_at_tokens=stop_at_tokens,
            warn_at_tokens=warn_at_tokens,
            expect_auth_mode=expect_auth_mode,
            options=options,
            raise_on_stop=raise_on_stop,
        )

    def resume_chat(
        self,
        *,
        connection: Connection | str,
        session_id: str,
        system_prompt: str | None = None,
        project_folder: str | None = None,
        model: str | None = None,
        persist: bool = True,
        guards: Guards | None = None,
        stop_at_tokens: int | None = None,
        warn_at_tokens: int | None = None,
        expect_auth_mode: AuthMode | str | None = None,
        options: Mapping[str, Any] | None = None,
        raise_on_stop: bool = False,
    ) -> ChatSession:
        """Pick a persisted session back up by id.

        Raises :class:`~modelpass.errors.SessionNotFound` when the runtime does
        not have it -- which is an ordinary outcome, not a broken caller. Claude
        Code's ``cleanupPeriodDays`` sweep deletes old transcripts, and a Codex
        thread whose first turn never completed leaves an id that resume
        rejects. A stored id is a hint.

        **There is deliberately no ``tools=`` here.** Tools were fixed when the
        session was created, they are the very front of a cached prefix that has
        been warm ever since, and modelpass cannot check a re-declared value
        against what the session actually holds -- so accepting one would mean
        either silently ignoring it or silently changing the prefix, and D17 has
        no room for either.

        ``system_prompt`` is the exception, and the runtimes split on it because
        they store it in different places. Gated on
        ``resume_carries_system_prompt``:

        * On **anthropic-sdk** the prompt is a *launch option*, not part of the
          transcript. Resuming without one launches the session under the
          minimal tool-calling prompt rather than the persona it was created
          with, and sends a prefix the original never warmed -- wrong answers
          and a cold cache, silently. So it is **required** here, and ``""`` is
          how a caller says the session genuinely had none.
        * On **openai-sdk** the prompt was serialized into the first turn and
          lives in the rollout, so resuming replays it. Passing one here is
          **refused**, because it would inject a second persona mid-conversation
          rather than restore the first.

        D17 said "on resume either omitted or an error if they differ", which
        assumed the runtime keeps it. One of them does not, so the rule is now
        per-runtime and the capability cell says which.

        ``project_folder`` matters on ``anthropic-sdk``, where sessions live at
        ``~/.claude/projects/<encoded-cwd>/<id>.jsonl`` and the folder *is* the
        storage key: pass the same one the session reported, which is what
        :attr:`Session.project_folder` and :class:`SessionInfo` are for.

        ``persist`` exists here only so that ``persist=False`` alongside a
        ``session_id`` is refused out loud rather than quietly ignored: an
        ephemeral session was never written down, so there is nothing to resume.

        Refused outright on a runtime that holds no conversation of its own; see
        :meth:`_require_sessions`.
        """
        self._require_sessions(self._resolve(connection).runtime, "bridge.resume_chat()")
        return self._open_session(
            ChatSession,
            connection=connection,
            model=model,
            system_prompt=system_prompt,
            persist=persist,
            project_folder=project_folder,
            tools=None,
            mcp_servers=None,
            guards=guards,
            stop_at_tokens=stop_at_tokens,
            warn_at_tokens=warn_at_tokens,
            expect_auth_mode=expect_auth_mode,
            options=options,
            raise_on_stop=raise_on_stop,
            session_id=session_id,
        )

    def list_sessions(
        self,
        *,
        connection: Connection | str,
        project_folder: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> list[SessionInfo]:
        """Every session this connection can see. Capability-gated, and token-free.

        Gated on ``sessions_list``, **for this request** -- the adapter is asked
        before the table, exactly as :meth:`chat` and the session gates do. That
        matters on ``openai-sdk``, where the answer depends on which transport the
        call selects. Since the 2026-08-31 default flip it answers by default:
        ``codex app-server`` enumerates with ``thread/list``, and the registry's
        ``supported`` describes it. ``options={"transport": "exec"}`` opts out
        into a transport with no scriptable listing at all, and is refused there
        rather than answered empty -- the refusal names the default it opted out
        of rather than the cell that says no.

        ``project_folder`` scopes the listing where the runtime scopes by it. On
        ``anthropic-sdk`` that is not optional in practice -- the working
        directory is the storage key -- so a session opened with modelpass's
        default scratch folder will not appear in a listing of anywhere else.
        Keep :attr:`Session.project_folder` if you intend to come back for it.
        Omitting the argument lists the **process working directory**, which is
        the only default that is a fact rather than a guess. On ``openai-sdk`` the
        store is flat: the listing is account-wide and each row reports the
        directory its own thread ran in.

        ``persist=False`` sessions never appear here, on any runtime, because
        they were never written down.

        Refused outright on a runtime that holds no conversation of its own; see
        :meth:`_require_sessions`. That refusal comes first because it names the
        alternative that works, where the ``sessions_list`` cell alone would
        only name itself.
        """
        resolved = self._resolve(connection)
        self._require_sessions(resolved.runtime, "bridge.list_sessions()")
        adapter = self.adapter_for(resolved.runtime)
        listing_options = dict(options or {})
        self._require_for_request(
            adapter, resolved.runtime, Capability.SESSIONS_LIST, listing_options
        )
        request = self._build_session_request(
            resolved,
            kind=ChatSession.kind,
            project_folder=project_folder or os.getcwd(),
            model=None,
            system_prompt=None,
            persist=True,
            tools=(),
            mcp_servers={},
            options=listing_options,
        )
        return list(adapter.list_sessions(request))

    def _open_session(
        self,
        session_cls: type[_SessionT],
        *,
        connection: Connection | str,
        model: str | None,
        system_prompt: str | None,
        persist: bool,
        project_folder: str | None,
        tools: Iterable[ToolDef | Mapping[str, Any]] | None,
        mcp_servers: Mapping[str, Mapping[str, Any]] | None,
        guards: Guards | None,
        stop_at_tokens: int | None,
        warn_at_tokens: int | None,
        expect_auth_mode: AuthMode | str | None,
        options: Mapping[str, Any] | None,
        raise_on_stop: bool,
        session_id: str | None = None,
    ) -> _SessionT:
        """Everything the two faces share, which is everything except policy.

        Written once on purpose. ``ChatSession`` and ``WorkerSession`` differ in
        three class attributes and nothing else, and the part that must never
        differ between them is the part that decides how a run is billed.

        The order below is the same order :meth:`chat` uses and for the same
        reason: **every capability assertion happens before the preflight, and
        therefore before anything is launched.** Asking a runtime for something
        it cannot do must not cost a token to find out.
        """
        resolved = self._resolve(connection)
        if not resolved.enabled:
            raise ConnectionDisabled(resolved.name)
        runtime = resolved.runtime
        self.registry.require(runtime, Capability.CHAT)
        # Resolved before the gates rather than after them, because since S5 the
        # gates ask it: on a runtime with more than one transport, what a session
        # can do depends on the options this call carries (see
        # :meth:`_require_for_session`). An unusable ``transport`` value raises
        # from the first gate that asks, which is still before anything launches.
        adapter = self.adapter_for(runtime)
        session_options = dict(options or {})

        # First of the gates since ticket 1.8, because it is the one that asks
        # whether this object can exist at all rather than whether one of its
        # arguments is reachable. A worker on a runtime with no toolbelt is not
        # a worker, and hearing about persist= or system_prompt= first would
        # send a reader off to change an argument that was never the problem.
        if session_cls.kind is SessionKind.WORKER:
            self._require_native_toolbelt(adapter, runtime, session_options)

        if session_id is not None:
            if not persist:
                raise InvalidSession(
                    "persist=False cannot be combined with a session_id: an "
                    "ephemeral session is never written down, so there is nothing "
                    "on disk for a resume to find"
                )
            self._require_for_session(
                adapter,
                runtime,
                Capability.SESSIONS_RESUME,
                "resuming a session by id needs this runtime to be able to pick a "
                "stored conversation back up",
                session_options,
            )
        if session_id is not None:
            # The prompt lives in the transcript on one runtime and in the launch
            # on the other, so "do I re-supply the persona" has opposite answers
            # and both silent failures are bad ones: a chat resumed without it on
            # anthropic-sdk runs under the minimal prompt against a cold prefix,
            # and one resumed *with* it on openai-sdk gets a second persona
            # injected mid-conversation.
            carries = self.registry.supports(
                runtime, Capability.RESUME_CARRIES_SYSTEM_PROMPT
            )
            if carries and system_prompt is not None:
                raise InvalidSession(
                    f"resume_chat(system_prompt=...) is refused on {runtime.value}: "
                    "this runtime serialized the prompt into the first turn, so the "
                    "rollout already replays it. Supplying one here would add a "
                    "second persona mid-conversation rather than restore the first. "
                    "Omit it"
                )
            if not carries and system_prompt is None:
                raise InvalidSession(
                    f"resume_chat(system_prompt=...) is required on {runtime.value}: "
                    "the system prompt is a launch option here, not part of the "
                    "transcript, so resuming without one launches the session under "
                    "the minimal tool-calling prompt instead of the persona it was "
                    "created with -- and sends a prefix the original never warmed. "
                    'Pass the same prompt again, or "" if it genuinely had none'
                )

        if not persist:
            self._require_for_session(
                adapter,
                runtime,
                Capability.EPHEMERAL_MULTI_TURN,
                "persist=False asks for a multi-turn conversation that leaves "
                "nothing on disk. Where the runtime cannot hold one in memory, "
                "every turn would start over -- a series of one-shots wearing a "
                "session's name -- so this refuses instead of degrading. Use "
                "persist=True, or bridge.chat() if one-shots are what you want",
                session_options,
            )

        tool_defs = normalize_tools(tools)
        servers = dict(mcp_servers or {})
        if tool_defs:
            self._require_for_session(
                adapter,
                runtime,
                Capability.TOOLS_IN_PROCESS,
                "tools= runs the caller's own functions inside a runtime tool loop "
                "(D12), and this runtime offers no in-process tool mechanism a "
                "session can use. The capability row's note says what the runtime "
                "can do elsewhere, where there is somewhere else",
                session_options,
            )
        if servers:
            self._require_for_session(
                adapter,
                runtime,
                Capability.MCP_SERVERS,
                "mcp_servers= declares servers for this session only",
                session_options,
            )

        if system_prompt is not None and session_cls.kind is not SessionKind.WORKER:
            self._require_for_session(
                adapter,
                runtime,
                Capability.SYSTEM_PROMPT_REPLACE,
                "system_prompt= on a chat session replaces the runtime's own "
                "persona, and this runtime can only append to it -- the prompt "
                "would land on top of an agent identity you were trying to remove. "
                "Use bridge.new_worker(), where append is the semantics you want",
                session_options,
            )
        owns_folder = False
        folder = (project_folder or "").strip()
        if not folder:
            if session_cls.project_folder_required:
                raise InvalidSession(
                    "new_worker() needs a project_folder: a worker keeps the "
                    "runtime's tools switched on, and a toolbelt pointed at a "
                    "scratch directory is useless. Name the folder the work is in"
                )
            folder = scratch_project_folder()
            owns_folder = True

        if expect_auth_mode is not None:
            expected = AuthMode(expect_auth_mode)
            if resolved.auth_mode is not expected:
                raise AuthModeMismatch(str(expected), str(resolved.auth_mode), resolved.name)

        effective_guards = self._call_guards(
            resolved, guards, stop_at_tokens, warn_at_tokens, None
        )

        request = self._build_session_request(
            resolved,
            kind=session_cls.kind,
            project_folder=folder,
            model=model,
            system_prompt=system_prompt,
            persist=persist,
            tools=tool_defs,
            mcp_servers=servers,
            options=session_options,
            resume_id=session_id,
        )

        # The receipt is taken over the session's *own* launch plan, wrapped in
        # the run request shape ``Adapter.preflight`` already speaks -- so it
        # names the same model, reports the same scrub, and attributes the model
        # to the session rather than to a call nobody made. It is available on
        # the object from construction, which is the whole reason construction
        # runs a preflight at all (D15).
        # ``cwd`` is threaded in deliberately. An adapter's instruction-file
        # disclosure reads it to find what reaches the prompt unwritten by the
        # caller -- on Codex, every ``AGENTS.md`` from the session's folder up to
        # the drive root. Without it a WorkerSession pointed at a repository
        # would pull that whole chain in silently, which is the disclosure
        # failing in exactly the case it exists for.
        receipt = adapter.preflight(
            RunRequest(
                connection=resolved,
                messages=(),
                plan=request.plan,
                model=request.model,
                options={**request.options, "cwd": request.project_folder},
                tools=tool_defs,
                mcp_servers=servers,
            )
        )
        # Answered from the ``SessionRequest``, not from the run-shaped wrapper
        # above: a worker's prefix starts with the runtime's own preset, and
        # only the session request carries the prompt semantics that say so.
        receipt = self._with_identity_verification(resolved, receipt)
        receipt = self._with_cache_disclosure(adapter, request, receipt)
        receipt = self._with_cache_breakpoints(request, receipt)
        receipt = self._with_reasoning_disclosure(request, receipt)
        receipt = self._with_prompt_cache(request, receipt)
        receipt.require_ok()
        receipt.require_auth_mode(
            AuthMode(expect_auth_mode) if expect_auth_mode is not None else None
        )
        # The plan carries the *connection's* guards, and a session may have been
        # given tighter ones for its own envelope. Left alone, a session opened
        # with stop_at_tokens= would print a receipt saying "no spend guards
        # configured" next to a help() line naming the ceiling -- two true
        # halves that contradict each other on screen. The receipt reports the
        # guards this session will actually run under.
        receipt = replace(receipt, guards_configured=effective_guards.configured)

        return session_cls(
            bridge=self,
            adapter=adapter,
            request=request,
            receipt=receipt,
            guards=effective_guards,
            raise_on_stop=raise_on_stop,
            owns_project_folder=owns_folder,
        )

    def _require_sessions(self, runtime: Runtime, call: str) -> None:
        """The session policy, stated once and enforced before an adapter exists.

        A session rests on the **runtime** holding the conversation (D14). Where
        the capability table says it holds none -- every cell in
        :data:`_SESSION_CELLS` ``unsupported``, which is what an HTTP endpoint
        looks like from here -- all four session doors are shut, and shut
        *here*: before :meth:`adapter_for`, before the launch plan, before the
        preflight. Asking a runtime for something it structurally cannot do must
        not import a vendor package to find out, let alone spend a token.

        **This is deliberately the bridge's refusal and not an adapter's.**
        ``anthropic-api`` already refuses all three session methods with a good
        sentence (ticket 1.6), and the other three API adapters will each have
        to write one. A policy that lives in four places is a policy that will
        eventually differ in one of them -- and the failure mode of the one that
        forgets is not an error, it is
        :attr:`SessionRequest.native_tools` and
        :attr:`~modelpass.adapters.base.SessionRequest.appends_system_prompt`
        reading ``True`` off a ``kind`` and handing back a session against a
        runtime with no persona and no toolbelt to be true *of*.

        It comes before the ``enabled`` check for the same reason it comes
        before the adapter: re-enabling the connection would not change the
        answer, and pointing a reader at a flag that cannot help them is worse
        than saying nothing.

        :class:`~modelpass.errors.CapabilityNotSupported` rather than a new
        class, because it is the one that already means *a fact about the
        runtime, not a mistake in the call* -- its own docstring says so -- and
        it already carries the runtime, the cell and the tri-state a caller
        would otherwise parse out of English.
        """
        supports = [self.registry.supports(runtime, cell) for cell in _SESSION_CELLS]
        if any(supports):
            return
        raise CapabilityNotSupported(
            str(runtime),
            str(Capability.SESSIONS_RESUME),
            str(self.registry.support(runtime, Capability.SESSIONS_RESUME)),
            f"{call} needs the runtime itself to hold the conversation (D14), and "
            f"{runtime.value!r} holds none: every request carries its whole "
            "history. Use bridge.chat(history=[...]), the stateless alternative, "
            "or bridge.ask(...) when one blocking answer is what you want",
        )

    def _require_native_toolbelt(
        self, adapter: Adapter, runtime: Runtime, options: Mapping[str, Any]
    ) -> None:
        """Refuse a worker on a runtime with no toolbelt of its own (latent defect).

        A ``WorkerSession`` differs from a ``ChatSession`` in exactly two facts,
        and both are derived from ``kind is SessionKind.WORKER`` and nothing
        else: the runtime's own toolbelt stays on, and ``system_prompt`` is
        appended to the runtime's persona rather than replacing it. Neither
        derivation asks whether the runtime *has* a toolbelt or a persona. So on
        a runtime that has neither, ``new_worker()`` would have constructed
        successfully and handed back a chat wearing a worker's name -- the one
        outcome a session object is supposed to make impossible.

        The ``tools`` cell is the existing way to say it: *does this runtime have
        tools at all*. A tri-state that is not ``supported`` refuses, so
        ``unverified`` refuses too -- "nobody has checked whether this runtime
        has a toolbelt" is not a basis for arming one.
        """
        self._require_for_session(
            adapter,
            runtime,
            Capability.TOOLS,
            "bridge.new_worker() keeps the runtime's own toolbelt switched on and "
            "appends system_prompt= to the runtime's own persona -- and this "
            "runtime has neither, so a worker here would be a chat wearing a "
            "worker's name rather than an agent that can do anything. Use "
            "bridge.new_chat(), or bridge.chat(tools=[...]) to hand the runtime "
            "your own functions",
            options,
        )

    @staticmethod
    def _require_chat(registry: CapabilityRegistry, runtime: Runtime) -> None:
        """``registry.require(runtime, chat)``, carrying the table's own note.

        The bare refusal names a cell -- *runtime 'openai-compatible' capability
        'chat' is unverified* -- which is a true sentence that tells the reader
        nothing they can act on. The note is where the actionable half already
        lives, dated and sourced, and on this runtime it is the sentence naming
        ``modelpass verify`` (ticket 1.10). Same courtesy
        :meth:`_require_for_request` has always paid; the ``detail`` argument
        exists for exactly this and every other message is unchanged.
        """
        support = registry.support(runtime, Capability.CHAT)
        if support is Support.SUPPORTED:
            return
        raise CapabilityNotSupported(
            str(runtime),
            str(Capability.CHAT),
            str(support),
            registry.note(runtime, Capability.CHAT) or "",
        )

    def _require_for_request(
        self,
        adapter: Adapter,
        runtime: Runtime,
        capability: Capability,
        options: Mapping[str, Any],
        registry: CapabilityRegistry | None = None,
    ) -> None:
        """Assert a capability **for this request**, not merely for the default (S7).

        Two questions, one of which the registry cannot answer:

        * :meth:`find` and :meth:`CapabilityRegistry.support
          <modelpass.capabilities.CapabilityRegistry.support>` answer *what can I
          rely on from this runtime by default* -- asked before a request exists,
          so a conservative answer is the correct one and **neither changes
          here**.
        * This gate answers *will this request work*, and on a runtime with more
          than one transport that can legitimately depend on the request's own
          ``options``.

        Until S7a the two shared one answer and the narrower one won, which
        refused calls that would have succeeded: ``tools_in_process`` read
        ``unsupported`` on ``openai-sdk`` because the cell described ``codex
        exec``, and it refused ``chat(tools=..., options={"transport":
        "app-server"})`` -- a run whose tool loop had been driven live and was
        already implemented. So the adapter is asked first, and only its ``None``
        ("no opinion") falls through to the table.

        **The flip inverted which side needs the hook** (2026-08-31). The
        registry now describes ``codex app-server``, so a request that selects
        nothing needs no help from here; what needs it is a request that opts out
        into ``codex exec``, where six cells are genuinely false and
        ``mcp_servers`` is genuinely true. The mechanism did not change and
        neither did the two questions -- only the direction the second one
        travels.

        ``registry`` is ticket 1.10's addition and answers a *third* question the
        other two do not: what has this **endpoint** been proved to do. On
        ``openai-compatible`` the table's row describes an API shape and the
        connection carries the drive, so the caller passes
        :meth:`registry_for`'s answer and the cells that ``modelpass verify``
        moved are the ones this gate reads. ``None`` means the shared table,
        which is what every other runtime wants and gets.

        The refusal carries the registry's own note when there is one. The bare
        form names a cell (*capability 'tools_in_process' is unsupported*) and
        leaves a caller to go and read the table to find out that the thing they
        asked for exists one option away; the note is where that sentence already
        lives, dated and sourced, and it is the same courtesy
        :meth:`_require_for_session` pays with a hand-written detail.
        """
        table = registry if registry is not None else self.registry
        support = adapter.support_for(capability, options)
        if support is None:
            support = table.support(runtime, capability)
        if support is Support.SUPPORTED:
            return
        raise CapabilityNotSupported(
            str(runtime),
            str(capability),
            str(support),
            table.note(runtime, capability) or "",
        )

    def _require_for_session(
        self,
        adapter: Adapter,
        runtime: Runtime,
        capability: Capability,
        detail: str,
        options: Mapping[str, Any],
    ) -> None:
        """Assert a capability, saying which argument asked for it and why.

        The bare registry message names a cell (*capability
        'ephemeral_multi_turn' is unsupported*); this one names the keyword the
        caller typed. Same refusal, read by someone who has not memorised the
        capability table.

        **Asks the adapter first, exactly as** :meth:`_require_for_request`
        **does** (S5, 2026-08-31). Sessions were the half of the API where a
        transport-selectable request still met a gate that could only describe the
        default transport, so ``new_chat(system_prompt=..., options={"transport":
        "app-server"})`` was refused by a cell that correctly described ``codex
        exec`` -- for a session modelpass now knows how to open. Since the default
        flipped later that day that pairing reversed: ``new_chat(system_prompt=
        ...)`` opens with no options at all, and it is
        ``options={"transport": "exec"}`` that meets the refusal. A ``None`` from
        the adapter means *no opinion* and falls through to the table, which is
        what keeps every runtime that has one transport behaving exactly as
        before.
        """
        support = adapter.support_for(capability, options)
        if support is None:
            support = self.registry.support(runtime, capability)
        if support is not Support.SUPPORTED:
            raise CapabilityNotSupported(
                str(runtime), str(capability), str(support), detail
            )

    def _build_session_request(
        self,
        connection: Connection,
        *,
        kind: SessionKind,
        project_folder: str,
        model: str | None,
        system_prompt: str | None,
        persist: bool,
        tools: tuple[ToolDef, ...],
        mcp_servers: Mapping[str, Mapping[str, Any]],
        options: Mapping[str, Any],
        resume_id: str | None = None,
    ) -> SessionRequest:
        """The session analogue of :meth:`_build_request`, with the same guarantees.

        Auth mode checked against what the runtime can drive, the launch plan
        scrubbed, the credential required to exist, and a caller-supplied model
        recorded on the plan as *requested for this call* so the receipt never
        reports a name with no provenance.
        """
        if connection.auth_mode not in runtime_auth_modes(connection.runtime, self.registry):
            raise AuthModeMismatch(
                str(connection.auth_mode),
                "unsupported by " + connection.runtime.value,
                connection.name,
            )
        plan = plan_launch(connection, self.env)
        plan.require_credential()
        if model:
            plan = replace(plan, model=model, model_source="requested for this session")
        return SessionRequest(
            connection=connection,
            plan=plan,
            kind=kind,
            project_folder=project_folder,
            model=plan.model,
            system_prompt=system_prompt,
            persist=persist,
            tools=tools,
            mcp_servers=dict(mcp_servers),
            options=dict(options),
            resume_id=resume_id,
        )

    # --- internals -------------------------------------------------------------

    def _resolve(self, connection: Connection | str) -> Connection:
        """Accept a name from the store, a ``group:`` reference, or a built object.

        Passing a ``Connection`` still means the caller built one on purpose, so
        the no-ambient-credentials rule holds: nothing here reads the environment
        looking for something to connect with.

        **``"group:<name>"`` routes through a group** (ticket 1.15), and this is
        the only place that spelling is understood -- which is why every entry
        point gets it without a ``group=`` keyword of its own on each, and why
        there is no call shape where a caller can name a connection and a group
        at once and have to be told which won. The prefix is the spelling rather
        than a bare name being tried as both, because a bare name that is a
        connection *and* a group would route somewhere on a rule the caller
        never saw; connection names cannot contain a colon
        (:data:`~modelpass.connections._NAME_RE`), so the two can never collide.
        It follows the wire form ``credentialRef`` already uses -- ``env:NAME``,
        ``secret:entry`` -- for the same reason: a pointer says what kind of
        thing it points at.

        The resolved connection is what the receipt, the terminal event and the
        run log name. A group is how the run was *addressed*; what it was billed
        to is a connection, and the record says so.
        """
        if isinstance(connection, Connection):
            return connection
        if connection.startswith(_GROUP_PREFIX):
            return self.select(connection[len(_GROUP_PREFIX) :]).chosen
        return self.store.get(connection)

    @staticmethod
    def _call_guards(
        connection: Connection,
        guards: Guards | None,
        stop_at_tokens: int | None,
        warn_at_tokens: int | None,
        allow_failover: bool | None = None,
    ) -> Guards:
        """Which guards this one call runs under.

        Three spellings, in decreasing explicitness: a whole ``Guards`` object,
        the per-call shorthands, or nothing at all (the connection's own).
        Mixing the first two is refused rather than resolved by precedence --
        a caller that passed both meant something, and guessing which half to
        drop is how a ceiling silently goes missing.

        ``allow_failover`` is one of the shorthands and composes with the
        thresholds: all of them start from the connection's own guards, so
        declining a failover never disturbs a configured ceiling and lowering a
        ceiling never disarms a configured failover. ``False`` replaces the
        quota policy with a clean stop; ``True`` and ``None`` leave the
        connection's configuration exactly as written -- there is deliberately
        no spelling of this argument that can *create* a failover, because the
        consent for metered billing is the connection file naming a target
        (D4(d)), not a keyword argument on one call.
        """
        shorthand = (
            stop_at_tokens is not None
            or warn_at_tokens is not None
            or allow_failover is not None
        )
        if guards is not None and shorthand:
            raise InvalidGuards(
                "pass either guards= or the stop_at_tokens= / warn_at_tokens= / "
                "allow_failover= shorthands, not both: they set the same thing "
                "and one would have to be silently discarded"
            )
        if guards is not None:
            return guards
        if not shorthand:
            return connection.guards
        overrides: dict[str, Any] = {}
        if stop_at_tokens is not None:
            overrides["stop_at_tokens"] = stop_at_tokens
        if warn_at_tokens is not None:
            overrides["warn_at_tokens"] = warn_at_tokens
        effective = Guards.for_call(connection.guards, **overrides)
        if allow_failover is False:
            effective = replace(effective, on_quota_exhausted=QuotaPolicy())
        return effective

    @staticmethod
    def _call_timeout(
        connection: Connection, timeout: Timeout | float | None
    ) -> Timeout | None:
        """The bound for this call: the caller's, else the connection's, else none.

        **A call that names a bound replaces the connection's rather than being
        clamped by it.** The alternative -- take the smaller of the two -- looks
        safer and is worse: a caller who asked for 120 seconds on a connection
        configured at 30 would get 30 and no indication why, which is a working
        batch job turning flaky on a configuration change nobody connected to
        it. A caller who named a number has said what they want.
        """
        chosen = Timeout.coerce(timeout)
        if chosen is not None:
            return chosen if chosen else None
        if connection.timeout_seconds is None:
            return None
        return Timeout(total=connection.timeout_seconds)

    def _build_request(
        self,
        connection: Connection,
        messages: tuple[Message, ...],
        *,
        model: str | None,
        options: Mapping[str, Any],
        sampling: Sampling | None = None,
        tools: tuple[ToolDef, ...] = (),
        mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
        schema: Mapping[str, Any] | None = None,
        schema_name: str = "",
    ) -> RunRequest:
        if connection.auth_mode not in runtime_auth_modes(connection.runtime, self.registry):
            raise AuthModeMismatch(
                str(connection.auth_mode),
                "unsupported by " + connection.runtime.value,
                connection.name,
            )
        plan = plan_launch(connection, self.env)
        plan.require_credential()
        if model:
            # The caller's model wins over the connection's, and the receipt says
            # so rather than reporting a name with no provenance -- the whole
            # point of putting a model on the receipt is that a rejected model is
            # diagnosable from what was printed before the run.
            plan = replace(plan, model=model, model_source="requested for this call")
        sampling = _with_standing_reasoning(connection, sampling)
        return RunRequest(
            connection=connection,
            messages=messages,
            plan=plan,
            model=plan.model,
            options=dict(options),
            sampling=sampling,
            tools=tools,
            mcp_servers=dict(mcp_servers or {}),
            schema=dict(schema) if schema is not None else None,
            schema_name=schema_name,
        )

    def _resolve_failover(
        self,
        connection: Connection,
        guards: Guards,
        expect_auth_mode: AuthMode | str | None,
    ) -> Connection | None:
        """Validate a configured failover target, or return ``None`` if there is none.

        Everything checkable without spending anything is checked here, before
        the first run starts. The alternative -- discovering a typo'd failover
        name at the moment the allowance runs out -- turns a configuration
        mistake into a lost run.

        Three refusals worth naming, all raised rather than warned about:

        * **No chaining.** A failover target that is itself configured to fail
          over is refused. Two connections pointing at each other would be an
          infinite loop; a chain of three would be a spend path nobody
          deliberately drew.
        * **No self-failover**, which is the same run again with the same
          exhausted allowance.
        * **No contradicting an assertion.** If the caller passed
          ``expect_auth_mode``, a failover target that does not match it is
          refused up front -- the assertion is about how this call may be
          billed, and a failover is still this call.
        """
        policy = guards.on_quota_exhausted
        if policy.action is not QuotaAction.FAILOVER:
            return None

        name = policy.failover or ""
        if name == connection.name:
            raise InvalidConnection(
                f"connection {connection.name!r} names itself as its quota failover; "
                "the second run would draw on the same exhausted allowance"
            )
        # Raises NoSuchConnection, listing what is configured. Deliberately from
        # the store: a failover target is a written-down connection, not
        # something a caller can conjure inline for one call.
        target = self.store.get(name)

        if not target.enabled:
            raise ConnectionDisabled(target.name)
        if target.guards.on_quota_exhausted.action is QuotaAction.FAILOVER:
            raise InvalidConnection(
                f"failover target {target.name!r} is itself configured to fail over "
                f"to {target.guards.on_quota_exhausted.failover!r}; failover never "
                "chains, so this would be a spend path nobody drew on purpose"
            )
        if target.auth_mode not in runtime_auth_modes(target.runtime, self.registry):
            raise AuthModeMismatch(
                str(target.auth_mode),
                "unsupported by " + target.runtime.value,
                target.name,
            )
        # The failover target's own drive, for the same reason the primary got
        # one: a verified openai-compatible endpoint is a legitimate place to
        # fail over to, and the shared table would refuse it (ticket 1.10).
        self.registry_for(target).require(target.runtime, Capability.CHAT)

        if expect_auth_mode is not None:
            expected = AuthMode(expect_auth_mode)
            if target.auth_mode is not expected:
                raise AuthModeMismatch(str(expected), str(target.auth_mode), target.name)
        return target

    # --- the pre-run pipeline (ticket 1.13) ------------------------------------

    def _prepare(self, **kwargs: Any) -> _Prepared:
        """Everything a call does before an adapter runs, as one frozen answer.

        Extracted so the two faces share it rather than agreeing with each other
        (R1). The whole pipeline is CPU-bound except
        :meth:`~modelpass.adapters.base.Adapter.preflight`, which is a
        subprocess on the agent runtimes and a cheap probe on the API ones --
        which is why it is the one step :meth:`achat` runs off the loop, and why
        it is the seam the two halves below are split at.
        """
        plan = self._plan(**kwargs)
        return plan.prepared(self._receipt_for(plan))

    def _plan(
        self,
        *,
        connection: Connection | str,
        message: str | Sequence[TextBlock],
        system_prompt: str | Sequence[TextBlock] | None = None,
        history: Iterable[Message | Mapping[str, Any]] | None = None,
        model: str | None = None,
        expect_auth_mode: AuthMode | str | None = None,
        guards: Guards | None = None,
        stop_at_tokens: int | None = None,
        warn_at_tokens: int | None = None,
        allow_failover: bool | None = None,
        raise_on_stop: bool = False,
        options: Mapping[str, Any] | None = None,
        sampling: Sampling | Mapping[str, Any] | None = None,
        tools: Iterable[ToolDef | Mapping[str, Any]] | None = None,
        mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
        schema: Mapping[str, Any] | None = None,
        schema_name: str | None = None,
        timeout: Timeout | float | None = None,
    ) -> _CallPlan:
        """The half of the pipeline that touches nothing outside this process.

        Resolution, the enabled check, message assembly, tool normalisation, the
        capability gates, schema normalisation, the auth-mode assertion, the
        call's guards and its failover target, and the request itself. Every
        refusal here is a caller mistake, and a caller mistake should never
        reach the point of launching a runtime.
        """
        resolved = self._resolve(connection)
        # Before anything else: a connection taken out of service refuses runs,
        # not inspection. preflight() deliberately does not check this, so
        # `modelpass check` and the bench can still report on a disabled
        # connection -- what is switched off is spending, not looking.
        if not resolved.enabled:
            raise ConnectionDisabled(resolved.name)
        # The one place the three arguments become a list, in the only order
        # that reads as a conversation: instructions, what was already said,
        # then the turn being taken now. Assembling it here rather than asking
        # the caller for a list is what makes the shape of a stateless call
        # honest -- ``history`` is visibly the caller's to hold.
        turns: list[Message | Mapping[str, Any]] = []
        if system_prompt is not None:
            turns.append({"role": "system", "content": system_prompt})
        turns.extend(history or ())
        turns.append({"role": "user", "content": message})
        normalized = normalize_messages(turns)
        # Refined by this connection's own verify drive where it has one; the
        # shared table otherwise (ticket 1.10).
        registry = self.registry_for(resolved)
        self._require_chat(registry, resolved.runtime)

        # Validate and gate before the preflight: a malformed tool or an
        # unsupported runtime is a caller mistake, and a caller mistake should
        # never reach the point of launching a runtime.
        tool_defs = normalize_tools(tools)
        servers = dict(mcp_servers or {})
        call_options = dict(options or {})
        call_sampling = Sampling.coerce(sampling)
        # Resolved before the gates because the gates ask it a question (S7a):
        # the registry answers "what can I rely on from this runtime by default",
        # which is the right answer for find() and the wrong one for a request
        # that selected a different transport. Since the default flipped
        # (2026-08-31) the adapter answers by NARROWING rather than widening --
        # an explicit transport="exec" on openai-sdk takes six cells away and
        # gives mcp_servers back. See _require_for_request.
        adapter = self.adapter_for(resolved.runtime)
        if tool_defs:
            self._require_for_request(
                adapter,
                resolved.runtime,
                Capability.TOOLS_IN_PROCESS,
                call_options,
                registry,
            )
        if servers:
            self._require_for_request(
                adapter, resolved.runtime, Capability.MCP_SERVERS, call_options, registry
            )

        normalized_schema: dict[str, Any] | None = None
        resolved_schema_name = ""
        if schema is not None:
            if tool_defs or servers:
                raise InvalidSchema(
                    "schema= cannot be combined with tools= or mcp_servers= in v1. "
                    "On openai-sdk the two mechanisms are reported upstream to "
                    "conflict (--output-schema silently ignored while MCP servers "
                    "are active, openai/codex#15451), and on anthropic-sdk the "
                    "interaction is unverified because structured output is itself "
                    "an end-turn tool there. Refusing on both runtimes beats a "
                    "combination that quietly works on one of them (D13)"
                )
            normalized_schema = normalize_schema(schema)
            resolved_schema_name = resolve_schema_name(normalized_schema, schema_name)
            # Before the preflight, like every other gate: asking a runtime for
            # something it cannot do must not cost a token to find out. An
            # `unverified` cell refuses too -- the registry's tri-state exists so
            # "we did not check" is never silently spent against.
            self._require_for_request(
                adapter,
                resolved.runtime,
                Capability.STRUCTURED_OUTPUT,
                call_options,
                registry,
            )
        elif schema_name is not None:
            raise InvalidSchema(
                "schema_name= was passed without schema=; it names a schema, and "
                "on its own it does nothing"
            )

        if expect_auth_mode is not None:
            expected = AuthMode(expect_auth_mode)
            if resolved.auth_mode is not expected:
                raise AuthModeMismatch(
                    str(expected), str(resolved.auth_mode), resolved.name
                )

        effective_guards = self._call_guards(
            resolved, guards, stop_at_tokens, warn_at_tokens, allow_failover
        )
        # Resolved *before* the primary run so a misconfigured failover is a
        # configuration error the caller sees immediately, not a surprise
        # discovered at the worst possible moment -- halfway through a call whose
        # allowance has just run out.
        failover = self._resolve_failover(resolved, effective_guards, expect_auth_mode)

        request = self._build_request(
            resolved,
            normalized,
            model=model,
            options=call_options,
            sampling=call_sampling,
            tools=tool_defs,
            mcp_servers=servers,
            schema=normalized_schema,
            schema_name=resolved_schema_name,
        )

        return _CallPlan(
            adapter=adapter,
            request=request,
            guards=effective_guards,
            failover=failover,
            expect_auth_mode=expect_auth_mode,
            timeout=self._call_timeout(resolved, timeout),
            raise_on_stop=raise_on_stop,
        )

    def _receipt_for(self, plan: _CallPlan) -> Receipt:
        """The preflight, decorated and then asserted against.

        The one blocking step in the pipeline, alone in a method so
        :meth:`achat` can hand it to a thread without copying the decorations
        that follow it.
        """
        receipt = self._with_identity_verification(
            plan.request.connection, plan.adapter.preflight(plan.request)
        )
        _note = self._option_note(plan.adapter, plan.request.options)
        if _note is not None:
            receipt = replace(receipt, notes=(*receipt.notes, _note))
        receipt = self._with_cache_disclosure(plan.adapter, plan.request, receipt)
        receipt = self._with_cache_breakpoints(plan.request, receipt)
        receipt = self._with_reasoning_disclosure(plan.request, receipt)
        receipt = self._with_prompt_cache(plan.request, receipt)
        receipt = self._with_sampling_disclosure(plan.request, receipt)
        receipt.require_ok()
        receipt.require_auth_mode(
            AuthMode(plan.expect_auth_mode)
            if plan.expect_auth_mode is not None
            else None
        )

        return receipt

    def _stream(self, prepared: _Prepared) -> Iterator[AgentEvent]:
        """The sync face: a thin pump around :class:`~modelpass._fold.CallFold`.

        Everything this loop knows is how to drive an iterator. What the events
        *mean* -- the tracker, the cumulative, the guard events, the stamp, the
        allowance, the ledger, the failover decision -- is the fold's, so the
        async face folds the same events the same way rather than agreeing with
        this one by inspection (ticket 1.13).
        """
        # One deadline for the whole call. ``total`` is the run's wall clock,
        # and a failover is one call on two connections rather than two calls --
        # restarting the clock on leg 2 would let a bounded call take twice its
        # bound and still report that it honoured it.
        deadline = Deadline(prepared.timeout) if prepared.timeout else None
        call = self._call_fold(prepared, deadline)
        try:
            running = True
            while running:
                yield from call.begin()
                yield from self._pump(call)
                events, running = call.advance()
                yield from events
            # The durable half of D3's stamp, then the one terminal event.
            yield from call.finish()
            call.raise_for_stop()
        finally:
            # Always, on every exit -- a terminal, a raise, or the caller
            # letting go of the iterator. A timer left armed after the run would
            # cancel whatever adapter it still points at, minutes later, for a
            # call nobody is holding any more.
            if deadline is not None:
                deadline.stop()

    def _call_fold(self, prepared: _Prepared, deadline: Deadline | None) -> CallFold:
        """The fold for one prepared call, wired to this bridge's two seams."""
        return CallFold(
            adapter=prepared.adapter,
            request=prepared.request,
            receipt=prepared.receipt,
            guards=prepared.guards,
            failover=prepared.failover,
            raise_on_stop=prepared.raise_on_stop,
            deadline=deadline,
            record_run=self._record_run,
            begin_failover=self._begin_failover,
        )

    def _pump(self, call: CallFold) -> Iterator[AgentEvent]:
        """Drive one leg's adapter and feed the fold. Yields what the fold says.

        The three exception arms are the pump's because they are about driving
        an iterator; what each of them *means* is the fold's.
        """
        fold = call.fold
        leg = call.leg
        stream: Iterator[AgentEvent] | None = None
        try:
            # Inside the try: on ``anthropic-sdk`` this call is not lazy -- it
            # imports the SDK and starts the pump thread -- so a failure here is
            # a run failure like any other and must not escape the
            # classification below.
            stream = leg.adapter.run(leg.request)
            for event in stream:
                yield from fold.feed(event)
                if fold.done:
                    break
            else:
                fold.exhausted = True
        except VendorRunFailed as exc:
            fold.vendor_failed(exc)
        except SubpassError:
            # Guaranteed-layer refusals (AuthModeMismatch, UnsafeLaunch) and
            # caller mistakes still raise. A guarantee that can be downgraded to
            # a status line is not a guarantee.
            raise
        except Exception as exc:
            error = fold.failed(exc)
            if error is not None:
                raise error from exc
        finally:
            if not fold.exhausted:
                if stream is not None:
                    # Closed here, in the consuming thread, and never by the
                    # watchdog: a generator closed while another thread is
                    # executing it raises ``ValueError`` -- the failure a
                    # consumer's own watchdog comment names.
                    _close(stream)
                # **Cancelled once.** The timeout path has already called
                # ``cancel()`` from the timer thread; calling it again here
                # would terminate a second process on a runtime that had
                # already started a new one, and would make the "cancelled
                # exactly once" test a consumer's cancellation contract rests
                # on pass for the wrong reason.
                if fold.deadline is None or fold.deadline.fired is None:
                    leg.adapter.cancel()

    async def _astream(self, plan: _CallPlan) -> AsyncIterator[AgentEvent]:
        """The async face: the same fold, pumped with ``async for``.

        Line for line :meth:`_stream`, and that is the point -- what differs is
        how an iterator is driven, which is the only thing either pump knows.
        """
        receipt = await self._areceipt_for(plan)
        prepared = plan.prepared(receipt)
        deadline = Deadline(prepared.timeout) if prepared.timeout else None
        call = self._call_fold(prepared, deadline)
        try:
            running = True
            while running:
                for event in call.begin():
                    yield event
                pump = self._apump(call)
                try:
                    async for event in pump:
                        yield event
                finally:
                    # Closed here rather than left to the loop's async-generator
                    # finaliser: "the worker thread has been joined" is a
                    # promise with a moment attached (R2), and a generator
                    # finalised whenever the loop gets round to it has none.
                    #
                    # The suppression covers exactly one case and it is not this
                    # call failing: when the *loop* is tearing both generators
                    # down at once, ``aclose()`` on one already being closed
                    # raises "asynchronous generator is already running". The
                    # pump is being closed either way; raising over which of the
                    # two closers got there first would turn an orderly shutdown
                    # into an error nobody can act on.
                    with suppress(RuntimeError):
                        await pump.aclose()
                events, running = call.advance()
                for event in events:
                    yield event
            for event in call.finish():
                yield event
            call.raise_for_stop()
        finally:
            # Always, on every exit -- a terminal, a raise, or an ``aclose()``.
            if deadline is not None:
                deadline.stop()

    async def _areceipt_for(self, plan: _CallPlan) -> Receipt:
        """The preflight, off the loop where it is a subprocess.

        On the two agent runtimes ``preflight`` launches a CLI and waits for it,
        which is exactly the blocking call an async consumer came here to stop
        making (a batch scraping tool's report, section 6.A). On the four API
        runtimes it is
        a credential lookup and a URL check -- cheap, synchronous, and not worth
        a thread, so it runs inline.
        """
        if plan.request.connection.runtime in API_RUNTIMES:
            return self._receipt_for(plan)
        return await asyncio.to_thread(self._receipt_for, plan)

    async def _apump(self, call: CallFold) -> AsyncIterator[AgentEvent]:
        """Drive one leg's adapter on the loop and feed the fold.

        :meth:`_pump`'s three exception arms, its cancel-once rule and its
        close-on-every-exit, with ``await`` where the sync one blocks.
        """
        fold = call.fold
        leg = call.leg
        stream: AsyncIterator[AgentEvent] | None = None
        try:
            stream = leg.adapter.arun(leg.request)
            async for event in stream:
                for out in fold.feed(event):
                    yield out
                if fold.done:
                    break
            else:
                fold.exhausted = True
        except VendorRunFailed as exc:
            fold.vendor_failed(exc)
        except SubpassError:
            raise
        except Exception as exc:
            error = fold.failed(exc)
            if error is not None:
                raise error from exc
        finally:
            if not fold.exhausted:
                # **Cancelled once**, as on the sync face: where the bound has
                # already fired, the timer thread has already called
                # ``cancel()``, and the teardown below must not call it again.
                cancel = fold.deadline is None or fold.deadline.fired is None
                if stream is not None:
                    await close_async_run(stream, leg.adapter, cancel=cancel)
                elif cancel:
                    leg.adapter.cancel()

    def _record_run(
        self,
        request: RunRequest,
        terminal: TerminalEvent,
        guards: Guards,
        allowance: Mapping[str, Any] | None = None,
        binary: str | None = None,
        breakpoints: tuple[int, int] = (0, 0),
        sampling: tuple[Mapping[str, Any], Mapping[str, Any], tuple[str, ...]] = (
            {},
            {},
            (),
        ),
    ) -> None:
        """Append this run to the audit log, best-effort.

        Swallows everything. A run the user paid for must not be lost because
        the bookkeeping failed, and the log is deliberately not on the critical
        path of anything -- see :mod:`modelpass.runlog`.

        ``allowance`` is the last ``rate_limit`` payload this run produced. It
        is the vendor's own account of how the plan is doing, which is a better
        answer to "how am I doing against my allowance" than any arithmetic
        modelpass could invent, and until now it was normalized into an event and
        then dropped on the floor (D20). ``None`` where the runtime reports
        none, and ``None`` is written as absence rather than as zero.

        ``binary`` is the executable the *preflight for this leg* resolved, so a
        failover writes each leg's own. ``None`` on a runtime whose SDK owns its
        subprocess, which is an absence rather than a gap.

        ``breakpoints`` is ``(requested, honoured)`` off the same leg's receipt
        (R3). Both zero for every plain-string run, which is what makes the pair
        additive to a file nobody rewrites.

        ``sampling`` is ``(requested, applied, notes)`` off that same receipt
        (R5), and is what lets a stored run be asked whether the temperature its
        config names is the temperature it actually ran at.
        """
        try:
            self.run_log.append(
                RunRecord.from_terminal(
                    terminal,
                    model=request.model,
                    guards_configured=guards.configured,
                    allowance=allowance,
                    binary=binary,
                    cache_breakpoints_requested=breakpoints[0],
                    cache_breakpoints_honoured=breakpoints[1],
                    sampling_requested=sampling[0],
                    sampling_applied=sampling[1],
                    sampling_notes=sampling[2],
                )
            )
        except Exception:
            return

    def _begin_failover(
        self,
        target: Connection,
        primary: RunRequest,
        stopped: TerminalEvent,
        spent: TokenUsage,
    ) -> tuple[FailoverEvent, Adapter, RunRequest, Receipt, Guards] | str:
        """Get the failover connection ready to run, or say why it cannot.

        Returns the announcement event plus everything needed to run the second
        connection -- or a string explaining the refusal, which the caller folds
        into the first connection's terminal.

        **A refusal is reported, not raised.** The first run genuinely did stop
        cleanly on quota; turning that into an exception would throw away a true
        and useful outcome because a *second* connection was misconfigured. A
        refusal also costs nothing, because it happens before the second
        connection runs -- which is the whole reason its preflight is executed
        here rather than being assumed to pass.
        """
        origin = primary.connection
        # The failover connection's own guards, from its own config: the first
        # connection's thresholds were about the first connection, and a run that
        # has already spent an allowance should not arrive at a second one with
        # its budget half gone. The quota policy is forced back to a clean stop,
        # so failover cannot chain even if the store changed underneath us since
        # _resolve_failover checked it.
        second_guards = replace(target.guards, on_quota_exhausted=QuotaPolicy())

        # A model name belongs to a runtime. Carrying the caller's model across a
        # runtime boundary would ask Codex for a Claude model and fail for a
        # reason that has nothing to do with the failover.
        carried = primary.model if target.runtime is origin.runtime else target.model
        # Passed as an override only when it differs from what the target would
        # have used anyway, so the second receipt attributes the model to the
        # connection when that is where it really came from.
        model = None if carried == target.model else carried

        try:
            # Re-checked here, not only at _resolve_failover time, for the same
            # reason the quota policy is re-forced below: the primary run may
            # have taken minutes, and a connection somebody disabled during it
            # must not be run anyway on the strength of a stale read.
            if not self.store.get(target.name).enabled:
                raise ConnectionDisabled(target.name)
            # The same request-level gate ``chat()`` uses, and for the same
            # reason (S7): ``primary.options`` travels to the second connection
            # verbatim a few lines below, so the transport the caller selected is
            # the transport the failover will run -- and the gate has to be asked
            # about that run rather than about the target runtime's default. On a
            # target with one transport the adapter has no opinion and the
            # registry answers exactly as before.
            adapter = self.adapter_for(target.runtime)
            target_registry = self.registry_for(target)
            if primary.tools:
                self._require_for_request(
                    adapter,
                    target.runtime,
                    Capability.TOOLS_IN_PROCESS,
                    primary.options,
                    target_registry,
                )
            if primary.mcp_servers:
                self._require_for_request(
                    adapter,
                    target.runtime,
                    Capability.MCP_SERVERS,
                    primary.options,
                    target_registry,
                )
            if primary.schema is not None:
                # The second connection has to be able to answer the same
                # question the first one was asked. A failover that silently
                # dropped the schema would hand back prose to a caller that had
                # already decided how to parse the reply.
                self._require_for_request(
                    adapter,
                    target.runtime,
                    Capability.STRUCTURED_OUTPUT,
                    primary.options,
                    target_registry,
                )
            request = self._build_request(
                target,
                primary.messages,
                model=model,
                options=primary.options,
                tools=primary.tools,
                mcp_servers=primary.mcp_servers,
                schema=primary.schema,
                schema_name=primary.schema_name,
            )
            receipt = self._with_identity_verification(
                target, adapter.preflight(request)
            )
            receipt.require_ok()
            receipt.require_auth_mode()
        except SubpassError as exc:
            return f"failover to {target.name!r} could not run: {exc}"

        event = FailoverEvent(
            from_connection=origin.name,
            to_connection=target.name,
            from_auth_mode=stopped.auth_mode,
            to_auth_mode=receipt.effective_auth_mode,
            to_runtime=target.runtime,
            reason=stopped.reason or "subscription allowance exhausted",
            usage=spent,
        )
        return event, adapter, request, receipt, second_guards


def _close(stream: Iterator[AgentEvent]) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        close()


_DEFAULT_BRIDGE: Bridge | None = None


def default_bridge() -> Bridge:
    """A process-wide bridge over the user's ``~/.modelpass`` store."""
    global _DEFAULT_BRIDGE
    if _DEFAULT_BRIDGE is None:
        _DEFAULT_BRIDGE = Bridge()
    return _DEFAULT_BRIDGE


def chat(
    *,
    connection: Connection | str,
    message: str,
    **kwargs: Any,
) -> Iterator[AgentEvent]:
    """Module-level convenience for :meth:`Bridge.chat` on the default bridge."""
    return default_bridge().chat(connection=connection, message=message, **kwargs)
