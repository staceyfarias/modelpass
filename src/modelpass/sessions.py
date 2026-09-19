"""Sessions: the conversation is the primitive (D14-D17).

::

    session = bridge.new_chat(connection="claude-sub", system_prompt=RUBRIC)
    for item in items:
        for event in session.send(item):
            ...
    session.close()

Every runtime modelpass drives owns its own conversation history and **none of
them accepts a message list**. So the primitive here is a session that maps onto
the runtime's native session or thread, and history lives where the runtime
keeps it. The stateless ``bridge.chat()`` is not removed; it becomes a
wrapper over this, which is the right way round -- a prefix cache bolted
*underneath* a stateless interface oscillates between two prompt shapes and
every miss is cache-cold, while above a session object a miss simply means
starting a new session.

Two faces, one implementation
-----------------------------

:class:`ChatSession` and :class:`WorkerSession` are not parallel
implementations. :class:`Session` holds the lifecycle, the guards, the receipt,
the id timing and the event plumbing; the subclasses differ only in policy:

======================  ==============================  ==============================
                        ``ChatSession``                 ``WorkerSession``
======================  ==============================  ==============================
Native runtime tools    off                             on
``system_prompt``       **replaces** the persona        **appends** to it
``project_folder``      optional, defaults isolated     **required**
======================  ==============================  ==============================

One ``send()``. The difference is construction, not a verb.

Immutability is the caching contract
------------------------------------

``tools`` and ``system_prompt`` are set at construction and **absent from**
:meth:`Session.send`. That is not a stylistic preference. Cache render order is
``tools`` -> ``system`` -> ``messages`` and the match is exact, so varying tools
per turn is the single most cache-destructive thing available and varying the
system prompt is second. Making the wrong thing unexpressible beats documenting
it as discouraged, and there is no escape hatch to design around: Anthropic's
mid-conversation system messages are a Messages API feature that neither agent
runtime exposes, which is why ``midconversation_system`` is a dated
``unsupported`` cell rather than an open question (D17).

Construction validates; the first send creates
----------------------------------------------

Neither runtime has a create-session call -- the session comes into existence by
running. So construction runs the **preflight** and every capability assertion,
and :attr:`Session.id` is ``None`` until the first :meth:`Session.send`
completes. On ``openai-sdk`` a thread is durable only after its first turn
finishes; a run killed before that leaves an id that ``codex exec resume``
rejects outright, so handing one out early would be a lie.

Nothing is degraded silently
----------------------------

``persist=False`` on a runtime without ``ephemeral_multi_turn``, ``tools=`` on a
runtime without ``tools_in_process``, ``system_prompt=`` on a ``ChatSession``
whose runtime cannot replace a system prompt: each raises **at construction**,
before anything is spent. Degrading instead would hand back an object that calls
itself a chat and behaves like a series of one-shots, with the transcript
flattening D14 exists to remove reintroduced through the back door.
"""

from __future__ import annotations

import shutil
import tempfile
import textwrap
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from ._deadline import Deadline
from ._fold import TurnFold, raise_for
from .adapters.base import (
    Adapter,
    SessionHandle,
    SessionRequest,
    close_async_run,
    threaded_run,
)
from .capabilities import Capability, CapabilityRegistry, Support
from .connections import Guards, QuotaAction
from .errors import (
    GuardStop,
    SessionBusy,
    SessionClosed,
    SubpassError,
    VendorRunFailed,
)
from .guards import GuardTracker
from .preflight import Receipt
from .runlog import RunRecord
from .runtimes import Runtime
from .tools import ToolDef
from .types import (
    AgentEvent,
    Message,
    SessionKind,
    TerminalEvent,
    TerminalStatus,
    Timeout,
    TokenUsage,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    # The bridge imports this module to build sessions, so a real import here
    # would be a cycle. A session genuinely belongs to a bridge -- it reads that
    # bridge's registry for help() and its run log for the ledger -- so the
    # reference is kept and only the annotation is deferred.
    from .bridge import Bridge

__all__ = [
    "ChatSession",
    "Session",
    "WorkerSession",
    "scratch_project_folder",
]

#: The capability cells both session classes turn on, and therefore the ones
#: :meth:`Session.help` reports. Deliberately a subset of the full row: a
#: session's help should answer "what can *this object* do on *this connection*",
#: and a wall of cells it does not use is how a reader stops reading.
_SHARED_HELP_CAPABILITIES: tuple[Capability, ...] = (
    Capability.CHAT,
    Capability.STREAMING,
    Capability.INCREMENTAL_TEXT,
    Capability.USAGE_TOKENS,
    Capability.INTERIM_USAGE,
    Capability.EPHEMERAL_MULTI_TURN,
    Capability.SESSIONS_RESUME,
    Capability.SESSIONS_LIST,
    Capability.TOOLS_IN_PROCESS,
    Capability.MCP_SERVERS,
    Capability.TTL_CONTROL,
)

_HELP_WIDTH = 88


def scratch_project_folder() -> str:
    """A fresh modelpass-owned working directory for a session that named none.

    A working directory is mandatory on both runtimes; the only question is the
    default, and inheriting the caller's process cwd is the wrong answer. With
    ``persist=True`` as the default, that would write agent transcripts into the
    consumer's own ``~/.claude/projects/<their-cwd>/``, where their own
    ``claude --continue`` would find them -- the same category of surprise D2
    exists to prevent. It also matters on ``openai-sdk``, where every
    ``AGENTS.md`` on the cwd's ancestor chain joins the prompt: a scratch
    directory has no ancestors carrying somebody's repo instructions.

    Matches what the Codex adapter already does and closes the gap on the
    Anthropic side, where runs currently inherit the caller's cwd.
    """
    return tempfile.mkdtemp(prefix="modelpass-session-")


class Session:
    """A conversation held by the runtime, driven one turn at a time.

    Not constructed directly -- :meth:`Bridge.new_chat`,
    :meth:`Bridge.new_worker` and :meth:`Bridge.resume_chat` build the two
    concrete faces, because opening one means resolving a connection, asserting
    capabilities and running a preflight, and none of that is a session's job.

    What *is* a session's job is everything both faces share: the lifecycle, the
    guard envelope, the receipt, when an id may be shown, and the event
    plumbing. The subclasses below add policy and no behavior, which is the
    point -- two objects that differed in their implementation would drift, and
    the thing that must not drift is the part that decides how a run is billed.

    Usable as a context manager::

        with bridge.new_chat(connection="claude-sub") as session:
            for event in session.send("hello"):
                ...

    Not thread-safe, and deliberately so: one conversation is one ordered
    sequence of turns, and a session driven from two threads is two callers
    disagreeing about what the model was last told.
    """

    #: Which face this is. Carried into the adapter on every request (D15).
    kind: ClassVar[SessionKind]
    #: Whether ``project_folder`` must be named by the caller.
    project_folder_required: ClassVar[bool]
    #: The capability cells this class's behavior turns on, **the ones that
    #: define this face first**. Class-aware, because ``system_prompt_replace``
    #: is the whole story for one of these objects and irrelevant to the other,
    #: and ordered so that :meth:`help` leads with the cell the caller chose this
    #: class for rather than burying it under six shared ones.
    help_capabilities: ClassVar[tuple[Capability, ...]]

    def __init__(
        self,
        *,
        bridge: Bridge,
        adapter: Adapter,
        request: SessionRequest,
        receipt: Receipt,
        guards: Guards,
        raise_on_stop: bool = False,
        owns_project_folder: bool = False,
    ) -> None:
        self._bridge = bridge
        self._adapter = adapter
        self._request = request
        self._receipt = receipt
        self._guards = guards
        self._raise_on_stop = raise_on_stop
        self._owns_project_folder = owns_project_folder

        # **One tracker for the whole session, not one per turn.** D18 makes
        # ``stop_at_tokens`` a budget envelope per surface -- a long-running
        # worker gets a large ceiling, an unattended background task a small one
        # -- and a per-turn tracker on a fifty-turn worker never fires while the
        # allowance drains. The ceiling is the session's.
        self._tracker = GuardTracker(guards, request.connection.name)
        # **One caller at a time** (R7, ticket 1.12). Held for the whole of a
        # turn, from the first ``next()`` on the iterator to its terminal, and
        # taken without blocking so contention is a raise rather than a queue --
        # see :class:`~modelpass.errors.SessionBusy` for why waiting would be
        # worse than refusing. Acquired inside the generator rather than in
        # ``send()`` so a caller who builds an iterator and never drains it
        # holds nothing.
        self._busy = threading.Lock()
        self._closed = False
        self._stopped: str | None = None
        self._turns = 0

        # Opening the handle is local work by contract (adapter rule 7): no
        # subprocess, no request, no spend. Doing it here rather than lazily on
        # first send is what makes an adapter that has not implemented sessions
        # yet fail at ``new_chat()`` -- where the caller can read the message --
        # instead of three lines later inside a stream.
        if request.resume_id is not None:
            self._handle: SessionHandle = adapter.resume_session(request)
            # A resumed session's id is known immediately, and reporting ``None``
            # here would be the mirror of the lie the timing rule exists to
            # prevent: the caller supplied a durable id and the runtime accepted
            # it, so there is nothing provisional left.
            self._id: str | None = request.resume_id
        else:
            self._handle = adapter.open_session(request)
            self._id = None

    # --- identity ----------------------------------------------------------------

    @property
    def id(self) -> str | None:
        """The runtime's own id for this session, or ``None`` if there is not one yet.

        ``None`` until the first :meth:`send` **completes**, and ``None`` for the
        whole life of a ``persist=False`` session -- there is nothing on disk to
        name, and ``list_sessions()`` will never show it either.

        modelpass owns this timing rather than passing through whatever the
        adapter has, because the consequence of getting it wrong is a caller
        storing an id that a later resume rejects.
        """
        if not self._request.persist:
            return None
        return self._id

    @property
    def connection(self) -> str:
        """The name of the connection this session runs on."""
        return self._request.connection.name

    @property
    def runtime(self) -> Runtime:
        return self._request.runtime

    @property
    def receipt(self) -> Receipt:
        """The preflight receipt, available from the moment of construction.

        Which connection, which auth mode, what was scrubbed, which model. This
        is what makes the spend legible before the first turn rather than after
        the last (D18), and it is why construction runs the preflight.
        """
        return self._receipt

    @property
    def project_folder(self) -> str:
        """The working directory this session runs in.

        Worth keeping: on ``anthropic-sdk`` the folder *is* the storage key, so
        this is the argument :meth:`Bridge.list_sessions` needs to find this
        session again.
        """
        return self._request.project_folder

    @property
    def persists(self) -> bool:
        """Whether this session can be resumed after the process exits."""
        return self._request.persist

    @property
    def system_prompt(self) -> str | None:
        """The system prompt fixed at construction. Immutable by design (D17)."""
        return self._request.system_prompt

    @property
    def tools(self) -> tuple[ToolDef, ...]:
        """The in-process tools fixed at construction. Immutable by design (D17)."""
        return self._request.tools

    @property
    def usage(self) -> TokenUsage:
        """Everything this session has spent so far, across every turn.

        The same running total the guard is comparing against, which is the
        number a caller wants on screen: a per-turn counter next to a
        session-wide ceiling shows a figure with no relationship to when the run
        will stop.
        """
        return self._tracker.total

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:
        state = "closed" if self._closed else (self._id or "unsent")
        return f"<{type(self).__name__} {self.connection}:{state}>"

    # --- turns -------------------------------------------------------------------

    def send(
        self, message: str, *, timeout: Timeout | float | None = None
    ) -> Iterator[AgentEvent]:
        """Run one turn and stream normalized events.

        **There is no ``tools=`` or ``system_prompt=`` here, and that absence is
        the caching contract** (D17). Both are fixed at construction because
        they sit at the front of the cached prefix -- ``tools`` -> ``system`` ->
        ``messages``, matched exactly -- so changing either between turns
        recomputes everything after it, on every runtime, every time. On
        ``openai-sdk`` prefix stability is the *only* cache lever there is:
        caching is automatic, exact-prefix and has no configuration surface at
        all. Keeping ``system_prompt`` a construction argument rather than
        letting callers concatenate instructions into the message is what makes
        the prefix stable across a loop.

        The stream contract is :meth:`Bridge.chat`'s, per turn:

        * exactly one ``receipt`` event first -- repeated on every turn, because
          each ``send`` is its own stream and a caller holding turn five's
          iterator should not have had to keep turn one's;
        * exactly one ``terminal`` event last, stamped by modelpass with the
          connection and the auth mode actually used (D3), carrying **this
          turn's** usage;
        * a guard stop, a quota exhaustion and a vendor failure are all normal
          completions with a terminal status, never exceptions -- unless
          ``raise_on_stop`` was set at construction.

        A terminal ends the *turn*, not the session: the conversation stays open
        and the next ``send`` continues it against the same warm prefix. The two
        exceptions are a guard stop, which was a ceiling on the whole session
        and so ends it, and :meth:`close`.

        ``timeout=`` bounds **this turn**, not the session (R6, ticket 1.12): a
        conversation that has run for an hour has not used up anything a caller
        asked of this turn. On expiry the turn ends with one terminal event,
        ``status`` :attr:`~modelpass.types.TerminalStatus.TIMED_OUT`, carrying
        what the turn spent; the session stays open, because a bound that ran
        out is not a conversation that ended.

        **One caller at a time** (R7). A second thread sending into this session
        while a turn is running gets :class:`~modelpass.errors.SessionBusy`
        rather than an interleaved transcript. The lock is taken on the first
        ``next()``, so an iterator built and never drained holds nothing.

        Abandoning the returned iterator closes it, which is the transport-level
        cancel D10 describes. The session itself stays open -- one abandoned
        turn is not the end of a conversation -- but whether the runtime's own
        history records a partial turn is the runtime's business, and modelpass
        does not pretend to know.
        """
        self._require_sendable(message)
        return self._turn(message, Timeout.coerce(timeout))

    def asend(
        self, message: str, *, timeout: Timeout | float | None = None
    ) -> AsyncIterator[AgentEvent]:
        """:meth:`send`, on a loop (R1, ticket 1.13)::

            async for event in session.asend("and then?"):
                ...

        The same turn, the same stream contract, the same per-turn bound, the
        same ledger line and the same guard behaviour, because it is the same
        fold -- only the loop that drives the runtime's iterator differs.

        **One caller at a time, across both faces** (R7). The lock a sync
        ``send`` takes is the lock this takes: a session already running a turn
        answers :class:`~modelpass.errors.SessionBusy` whichever door the second
        caller came through. It is taken on the first step, so an iterator built
        and never driven holds nothing.

        **Cancellation is ``aclose()``**, where the sync contract is abandoning
        the iterator. The session stays open either way -- one abandoned turn is
        not the end of a conversation.

        The runtime's own handle is synchronous on both agent runtimes, so this
        drives it on a worker thread and joins that thread when the turn ends or
        is closed. There is no native async session anywhere to override it with
        yet: the API runtimes have no sessions at all (ticket 1.8), which is a
        policy about where a conversation lives rather than a gap here.
        """
        self._require_sendable(message)
        return self._aturn(message, Timeout.coerce(timeout))

    async def _aturn(
        self, message: str, timeout: Timeout | None
    ) -> AsyncIterator[AgentEvent]:
        """:meth:`_turn`, awaited: take the session for one turn, then give it back."""
        self._take()
        deadline = Deadline(timeout) if timeout else None
        try:
            async for event in self._arun_turn(message, deadline):
                yield event
        finally:
            if deadline is not None:
                deadline.stop()
            self._busy.release()

    async def _arun_turn(
        self, message: str, deadline: Deadline | None
    ) -> AsyncIterator[AgentEvent]:
        """:meth:`_run_turn` with ``await`` where it blocks. One fold, two pumps."""
        fold = TurnFold(
            connection=self._request.connection,
            receipt=self._receipt,
            guards=self._guards,
            tracker=self._tracker,
            deadline=deadline,
            reason_for=self._terminal_reason,
        )
        stream: AsyncIterator[AgentEvent] | None = None
        try:
            for event in fold.begin():
                yield event
            if deadline is not None:
                deadline.start(self._adapter)
            # The handle is synchronous on both agent runtimes, so the turn runs
            # on a worker thread and the loop waits on a queue rather than on a
            # subprocess. ``send`` is called **on that thread**: on
            # ``anthropic-sdk`` it is where the vendor session is created.
            stream = threaded_run(
                self._adapter,
                lambda: self._handle.send(message),
                name=self.connection,
            )
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
            if not fold.exhausted and stream is not None:
                # Cancelled once, and only where the sync face cancels: the
                # watchdog has already done it when the bound fired, and a turn
                # abandoned with no bound at all closes the transport without
                # interrupting a runtime the session intends to keep talking to.
                await close_async_run(
                    stream,
                    self._adapter,
                    cancel=deadline is not None and deadline.fired is None,
                )

        stopped = fold.finish()
        self._finish_turn(stopped, fold)
        yield stopped

        if self._raise_on_stop:
            raise_for(
                stopped,
                guards=self._guards,
                tracker=self._tracker,
                connection=self._request.connection.name,
                deadline=deadline,
                timeout_reason="the turn timed out",
            )

    def _require_sendable(self, message: str) -> None:
        """The three refusals every turn makes before it takes the session.

        Raised at the call, on both faces, because none of them is about a run:
        they are about a message, a closed session, and a session whose ceiling
        was already crossed.
        """
        if not isinstance(message, str):
            raise TypeError(
                f"send() takes the turn's text as a str, got {type(message).__name__}"
            )
        if not message.strip():
            raise ValueError("send() needs a non-empty message")
        if self._closed:
            raise SessionClosed(
                f"session on connection {self.connection!r} is closed; open a new "
                "one with bridge.new_chat() / bridge.new_worker(), or resume this "
                "one by id if it was persisted"
            )
        if self._stopped is not None:
            # Raised rather than answered with a terminal: nothing ran, nothing
            # was spent, and a terminal event for a turn that never happened
            # would put a line in the ledger for work nobody did.
            raise GuardStop(
                f"session on connection {self.connection!r} is over: {self._stopped}. "
                "The ceiling was for the whole session, so continuing means opening "
                "a new one",
                observed=self._tracker.total.total_tokens,
                threshold=self._guards.stop_at_tokens or 0,
            )

    def _take(self) -> None:
        """Hold the session for one turn, or say who is already holding it (R7)."""
        if not self._busy.acquire(blocking=False):
            raise SessionBusy(
                f"session on connection {self.connection!r} is already running a "
                "turn. A Session holds one conversation, one token tracker and "
                "one vendor handle, so it takes one caller at a time: give each "
                "thread its own session (bridge.new_chat() is local work), or "
                "serialize the turns yourself. A Bridge is safe to share"
            )

    def _finish_turn(self, stopped: TerminalEvent, fold: TurnFold) -> None:
        """What a finished turn leaves behind, on either face."""
        if fold.session_stopped is not None:
            # The ceiling was for the whole session, so the conversation ends
            # with the turn that crossed it.
            self._stopped = fold.session_stopped
        self._turns += 1
        # The id is read only now, because "the first send creates the session"
        # means the first send that *finishes*. A caller who walked away from the
        # iterator gets no id, which is exactly right -- on ``openai-sdk`` that
        # thread does not exist.
        if self._id is None:
            self._id = self._handle.id
        self._record(stopped, fold.allowance)

    def _turn(
        self, message: str, timeout: Timeout | None
    ) -> Iterator[AgentEvent]:
        """Take the session for the duration of one turn, then give it back.

        The bound is **per turn**, not per session: a conversation that has run
        for an hour has not used up anything a caller asked of this turn.
        """
        self._take()
        deadline = Deadline(timeout) if timeout else None
        try:
            yield from self._run_turn(message, deadline)
        finally:
            if deadline is not None:
                deadline.stop()
            self._busy.release()

    def _run_turn(
        self, message: str, deadline: Deadline | None
    ) -> Iterator[AgentEvent]:
        """One turn: a thin pump around :class:`~modelpass._fold.TurnFold`.

        The fold says what the events mean -- the tracker, this turn's usage,
        the guard events, the D3 stamp, the allowance, the timeout arms -- and
        this loop only drives the handle's iterator. The stateless call's pump
        is :meth:`modelpass.Bridge._pump` and folds the same events the same
        way, which is the whole reason the folding moved out of both of them
        (ticket 1.13).
        """
        fold = TurnFold(
            connection=self._request.connection,
            receipt=self._receipt,
            guards=self._guards,
            tracker=self._tracker,
            deadline=deadline,
            reason_for=self._terminal_reason,
        )
        stream: Iterator[AgentEvent] | None = None
        try:
            yield from fold.begin()
            if deadline is not None:
                deadline.start(self._adapter)
            stream = self._handle.send(message)
            for event in stream:
                yield from fold.feed(event)
                if fold.done:
                    break
            else:
                fold.exhausted = True
        except VendorRunFailed as exc:
            fold.vendor_failed(exc)
        except SubpassError:
            # Guaranteed-layer refusals and caller mistakes still raise. A
            # guarantee that can be downgraded to a status line is not one.
            raise
        except Exception as exc:
            error = fold.failed(exc)
            if error is not None:
                raise error from exc
        finally:
            if not fold.exhausted and stream is not None:
                _close(stream)
                if deadline is not None and deadline.fired is None:
                    # Cancelled once. The watchdog has already done it where the
                    # bound fired, and it never closes the iterator itself.
                    self._adapter.cancel()

        stopped = fold.finish()
        self._finish_turn(stopped, fold)
        yield stopped

        if self._raise_on_stop:
            raise_for(
                stopped,
                guards=self._guards,
                tracker=self._tracker,
                connection=self._request.connection.name,
                deadline=deadline,
                timeout_reason="the turn timed out",
            )

    def _terminal_reason(self, event: TerminalEvent) -> str | None:
        """The adapter's reason, plus the failover disclosure where one is owed.

        A connection may name a quota failover, and **a session does not take
        it.** Failing over means starting a fresh session on another connection
        with none of this conversation in it, which is not the same conversation
        by any reading (D14) -- so the run stops cleanly instead. What modelpass
        will not do is stay quiet about it: a caller who configured a failover
        and watched a session stop anyway is owed the sentence saying why, on
        the event where they will actually read it.
        """
        reason = event.reason
        if event.status is not TerminalStatus.QUOTA_EXHAUSTED:
            return reason
        policy = self._guards.on_quota_exhausted
        if policy.action is not QuotaAction.FAILOVER:
            return reason
        note = (
            f"the configured failover to {policy.failover!r} does not apply to a "
            "session: the conversation lives inside this runtime, and a second "
            "connection would start an empty one rather than continue this. Use "
            "bridge.chat() if a call needs to be able to cross connections"
        )
        return f"{reason}; {note}" if reason else note

    def _record(
        self, terminal: TerminalEvent, allowance: Mapping[str, Any] | None = None
    ) -> None:
        """Append this turn to the audit log, best-effort.

        One line per turn, because one turn is one run against the allowance.
        Swallows everything: a turn the user paid for must not be lost because
        the bookkeeping failed.

        ``allowance`` is the vendor's own report of where the plan stood during
        this turn, or ``None`` on a runtime that reports none. Absent rather
        than zero, throughout -- see :class:`~modelpass.runlog.RunRecord`.

        The binary comes from this session's own receipt: every turn launches
        the executable :meth:`Adapter.open_session` resolved once, so the whole
        conversation's lines name the same one, and a session's turns are as
        answerable as a stateless run when a defect turns up in one build.
        """
        try:
            self._bridge.run_log.append(
                RunRecord.from_terminal(
                    terminal,
                    model=self._request.model,
                    guards_configured=self._guards.configured,
                    allowance=allowance,
                    binary=self._receipt.binary,
                )
            )
        except Exception:
            return

    # --- history and lifecycle ---------------------------------------------------

    def get_history(self) -> list[Message]:
        """The conversation so far, as the runtime has it.

        **Read-only introspection.** It cannot be fed back as input on either
        runtime, which is the fact D14 is built on: history is not something a
        caller passes in, it is something the runtime holds and modelpass can
        read. A runtime with no way to read it answers with an empty list rather
        than a transcript modelpass reassembled from the events it happened to
        see -- two histories that drift, only one of which is what the model is
        being sent, is worse than one honest absence.

        On ``openai-sdk`` which of those two you get depends on the transport
        (2026-08-31), and the default answers: ``options={"transport": "exec"}``
        cannot read a thread back and answers empty;
        ``codex app-server`` pages the thread's own items and answers with the
        conversation, tool calls and compactions marked in place rather than
        dropped. Same method, and the receipt names which transport is behind it.
        """
        if self._closed:
            raise SessionClosed(
                f"session on connection {self.connection!r} is closed; its history "
                "belongs to the runtime and modelpass keeps no shadow copy. Resume it "
                "by id to read it back"
            )
        return list(self._handle.history())

    def close(self) -> None:
        """Release this session. Idempotent.

        Closing does **not** delete a persisted session -- it stays resumable by
        :attr:`id`. What ends is modelpass's hold on it. A ``persist=False``
        session is the exception in one respect: the scratch working directory
        modelpass created for it is removed here, because "nothing on disk" is the
        promise that flag makes.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._handle.close()
        finally:
            if self._owns_project_folder and not self._request.persist:
                # A persisted session keeps its folder: on ``anthropic-sdk`` the
                # working directory is the storage key, so deleting it would
                # leave a session that exists and cannot be found.
                shutil.rmtree(self._request.project_folder, ignore_errors=True)

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- help --------------------------------------------------------------------

    def help(self) -> str:
        """What this object can and cannot do **on this connection**, in prose.

        Better than static documentation, which has to hedge across every
        runtime, and better than the raw capability table, which answers about a
        runtime rather than about the object in the caller's hand. Three parts:
        the shape this session was constructed with, the capability cells its
        behavior turns on with the registry's own notes for them, and the
        preflight receipt for this connection.

        **Class-aware, and that is the reason it is a method rather than a
        module function.** A :class:`ChatSession` on a runtime that cannot
        replace a system prompt leads with exactly that, because replace is the
        point of the object. A :class:`WorkerSession` on the same connection
        does not mention it at all, because there the runtime's own persona is
        what you wanted and the two runtimes genuinely agree.
        """
        registry = self._bridge.registry
        runtime = self.runtime
        out: list[str] = [
            f"{type(self).__name__} on connection {self.connection!r} "
            f"({runtime.value})",
            "",
        ]
        out.extend(self._shape_lines())

        limits = [
            (cap, self._support(cap))
            for cap in self.help_capabilities
            if self._support(cap) is not Support.SUPPORTED
        ]
        if limits:
            out.append("")
            out.append(f"Limits on {runtime.value}")
            for cap, support in limits:
                out.append(f"  {cap.value} is {support.value}")
                out.extend(_note_lines(registry, runtime, cap))

        out.append("")
        out.append("Capabilities this object uses")
        for cap in self.help_capabilities:
            out.append(f"  {cap.value:<24} {self._support(cap).value}")

        notes = [
            cap
            for cap in self.help_capabilities
            if self._support(cap) is Support.SUPPORTED
            and registry.note(runtime, cap)
        ]
        if notes:
            out.append("")
            out.append("Notes")
            for cap in notes:
                out.append(f"  {cap.value}")
                out.extend(_note_lines(registry, runtime, cap))

        out.append("")
        out.append("Receipt")
        out.append(f"  {self._receipt.summary()}")
        for extra in (
            self._receipt.model_note,
            self._receipt.guard_note,
            self._receipt.cache_note,
        ):
            if extra:
                out.extend(_wrap(extra))
        return "\n".join(out)

    def _support(self, capability: Capability) -> Support:
        """What this cell means **for this session**, adapter first (S5).

        :meth:`help` answers *what can this object do on this connection*, and on
        a runtime with more than one transport the connection is only half the
        question -- the session's own ``options`` decide the rest. Reading the
        table alone produced a genuinely self-contradicting page: a
        ``ChatSession`` that opened on ``options={"transport": "app-server"}``,
        because ``baseInstructions`` replaces the persona, printed
        *system_prompt_replace is unsupported* underneath its own shape lines.

        Same rule the gates use, so ``help()`` and the refusals cannot disagree:
        the adapter's ``None`` means no opinion and the registry answers.
        """
        support = self._adapter.support_for(capability, self._request.options)
        if support is not None:
            return support
        return self._bridge.registry.support(self.runtime, capability)

    def _shape_lines(self) -> list[str]:
        """The construction facts, which is what a reader checks first."""
        request = self._request
        prompt = (
            "none"
            if request.system_prompt is None
            else (
                f"{len(request.system_prompt)} characters, "
                + (
                    "appended to the runtime's own persona"
                    if request.appends_system_prompt
                    else "replacing the runtime's own persona"
                )
            )
        )
        tools = "on -- the runtime's own toolbelt" if request.native_tools else "off"
        if request.tools:
            tools += f", plus {len(request.tools)} in-process"
        if request.mcp_servers:
            tools += f", plus {len(request.mcp_servers)} MCP server(s)"
        persist = (
            "yes -- resumable by id after this process exits"
            if request.persist
            else "no -- nothing is written down and session.id stays None for life"
        )
        stop = self._guards.stop_at_tokens
        ceiling = (
            f"{stop} tokens for the whole session"
            if stop
            else "none -- nothing bounds what this session may spend"
        )
        return [
            f"  system prompt   {prompt}",
            f"  native tools    {tools}",
            f"  project folder  {request.project_folder}",
            f"  persist         {persist}",
            f"  spend ceiling   {ceiling}",
            f"  session id      {self.id or 'not yet -- the first send creates it'}",
        ]


class ChatSession(Session):
    """A conversation with the runtime's own agent persona **stripped out**.

    ``system_prompt`` replaces whatever the runtime ships with, native tools are
    off, and ``project_folder`` defaults to a modelpass-owned scratch directory --
    a chat has no repository to be pointed at, and inheriting the caller's cwd
    would write transcripts into their own project history.

    This is the object for a scorer, a classifier, a rewriter, an assistant: any
    work where the runtime's built-in identity is something to remove rather
    than something to use. It is also the object with a runtime asymmetry worth
    knowing about, and on ``openai-sdk`` the asymmetry is now a *transport*
    asymmetry (2026-08-31): ``codex app-server`` has
    ``ThreadStartParams.baseInstructions``, which genuinely replaces, and it is
    the default -- so ``system_prompt=`` on a Codex connection opens a
    :class:`ChatSession` with no options at all, the first this runtime has been
    able to hold. ``options={"transport": "exec"}`` opts out into a transport
    with no system-prompt parameter, which layers a ``System:`` line on top of a
    hardcoded coding-agent persona instead, and there ``system_prompt=`` still
    raises with :meth:`help` leading on the reason.

    **What that replacement does and does not buy**, because it is easy to
    over-read: ``baseInstructions`` replaces the base instructions and nothing
    else. Tool definitions and environment context survive it, and a model reads
    its identity off its toolbelt too -- which is why a ``ChatSession`` also
    switches the runtime's own toolbelt off. Both halves together are what make
    this object a scorer rather than a coding agent wearing a scorer's prompt.

    Reuse one for repeated work. The economics differ by runtime and it is worth
    being straight about which argument applies: on ``openai-sdk`` the case is
    cost and it is overwhelming, because the harness is paid per invocation and
    nothing amortizes it. On ``anthropic-sdk``, where the measured floor is
    around 171 input tokens and the cache survives process death, the case is
    **fidelity** -- structured turns instead of a flattened ``Human:`` /
    ``Assistant:`` transcript -- not money.
    """

    kind: ClassVar[SessionKind] = SessionKind.CHAT
    project_folder_required: ClassVar[bool] = False
    help_capabilities: ClassVar[tuple[Capability, ...]] = (
        Capability.SYSTEM_PROMPT_REPLACE,
        *_SHARED_HELP_CAPABILITIES,
    )


class WorkerSession(Session):
    """A conversation with the runtime's own agent **kept**, pointed at a folder.

    ``system_prompt`` is appended to the runtime's persona rather than replacing
    it, native tools stay on, and ``project_folder`` is **required** -- a worker
    with a toolbelt aimed at a scratch directory is useless.

    This is where the two runtimes genuinely agree, and it is the object Codex
    is good at: append is the only system-prompt semantics it has, and here
    append is what you wanted. On ``anthropic-sdk`` it maps to the preset form,
    never to omission -- omitting the prompt there does not give Claude Code's
    instructions, it gives the minimal tool-calling ones, which is a full
    toolbelt with no guidance for using it.

    ``system_prompt_replace`` is deliberately absent from this class's
    :meth:`help`. It is not a limitation of a worker; it is a description of a
    different object.
    """

    kind: ClassVar[SessionKind] = SessionKind.WORKER
    project_folder_required: ClassVar[bool] = True
    help_capabilities: ClassVar[tuple[Capability, ...]] = (
        Capability.TOOLS,
        *_SHARED_HELP_CAPABILITIES,
    )


def _wrap(text: str, indent: str = "    ") -> list[str]:
    return textwrap.wrap(
        text,
        width=_HELP_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def _note_lines(
    registry: CapabilityRegistry, runtime: Runtime, capability: Capability
) -> list[str]:
    """The registry's own prose for a cell, wrapped.

    The notes carry the date and the evidence for every cell and are already
    written in the right voice; until now they had nowhere to be read.
    """
    note = registry.note(runtime, capability)
    return _wrap(note) if note else []


def _close(stream: Iterator[AgentEvent]) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        close()
