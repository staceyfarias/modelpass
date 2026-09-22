"""The event fold: one statement of what a run's events *mean*, driven by two faces.

Ticket 1.13. Nothing here is new behaviour -- every rule in this module was
lifted verbatim out of :meth:`modelpass.bridge.Bridge._run_one` and
:meth:`modelpass.sessions.Session._run_turn`, which between them held about 325
lines of folding: usage into the tracker, ``cumulative`` injected on the way
past, guard events, the D3 terminal stamp with its retryability verdict, the
allowance payload observed and never intercepted, ``VendorRunFailed`` turned
into a terminal rather than an exception, and the timeout arms.

**Why it moved.** The async face (R1) has to fold the same events the same way.
Writing that twice is how two dialects of one contract start -- and the fold is
precisely the part where a difference would be invisible until a consumer's
ledger disagreed with itself. So the folding is a state machine with no I/O in
it: :meth:`EventFold.feed` takes one adapter event and returns the events the
caller should see, and both faces are thin pumps around it. The pump owns the
loop, the ``try``/``finally``, and the cancel-once rule, because those are about
*driving* an iterator and differ between a ``for`` and an ``async for``. The
meaning does not differ, and is therefore written once.

Two folds, because two things are being folded:

* :class:`RunFold` -- one leg of a stateless :meth:`~modelpass.Bridge.chat`
  call. ``stateless=True`` for the retry verdict; its terminal carries the
  tracker's running total, and a guard stop names the call's threshold.
* :class:`TurnFold` -- one turn of a :class:`~modelpass.sessions.Session`.
  ``stateless=False``, because the turn left a conversation behind on the
  runtime; its terminal carries **this turn's** usage rather than the session's,
  and a guard stop ends the session as well as the turn.

:class:`CallFold` sits above :class:`RunFold` and holds the thing a single leg
cannot: the failover decision, the per-leg run record, and the raise-on-stop
policy that reads the *final* terminal. It reaches the bridge only through the
two callables it is constructed with, so it stays as testable as the folds are.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from ._deadline import Deadline
from .adapters.base import Adapter, RunRequest
from .connections import Connection, Guards
from .errors import (
    AdapterFailed,
    GuardStop,
    QuotaExhausted,
    RunTimedOut,
    VendorRunFailed,
)
from .guards import GuardTracker
from .preflight import Receipt
from .reasoning import reasoning_metric
from .retry import classify_terminal, vendor_error_facts
from .types import (
    AgentEvent,
    FailoverEvent,
    ReceiptEvent,
    TerminalEvent,
    TerminalStatus,
    TokenUsage,
    UsageEvent,
    UsageScope,
    VendorEvent,
)

__all__ = ["CallFold", "EventFold", "Leg", "RunFold", "TurnFold", "raise_for"]


class EventFold:
    """What one run's events mean. No I/O, no loop, no adapter driving.

    The caller pumps: :meth:`begin` once, :meth:`feed` per event until
    :attr:`done`, then :meth:`finish`. The three exception arms
    (:meth:`vendor_failed`, :meth:`failed`, and "the iterator ran out", which is
    :attr:`exhausted`) are the pump's to notice and this object's to interpret.
    """

    #: ``True`` once the adapter's iterator has been drained normally. The pump
    #: sets it, and it is what decides whether the stream still needs closing
    #: and the adapter still needs cancelling.
    exhausted: bool

    def __init__(
        self,
        *,
        connection: Connection,
        receipt: Receipt,
        guards: Guards,
        tracker: GuardTracker,
        deadline: Deadline | None,
    ) -> None:
        self.connection = connection
        self.receipt = receipt
        self.guards = guards
        self.tracker = tracker
        self.deadline = deadline
        self.exhausted = False
        self.stopped: TerminalEvent | None = None
        #: Last wins. A run may report the allowance several times; the most
        #: recent report is the one that describes where the plan stands now.
        self.allowance: Mapping[str, Any] | None = None
        #: The level the vendor said this run is using, where one says anything
        #: (2026-09-22). Observed the way the allowance is -- on the way past,
        #: never intercepted -- because the event it comes from is a passthrough
        #: the caller is entitled to see whole.
        self.reasoning_echo: str | None = None

    # --- what the pump asks -----------------------------------------------------

    @property
    def done(self) -> bool:
        """Whether a terminal has been decided and the pump should stop reading."""
        return self.stopped is not None

    def begin(self) -> list[AgentEvent]:
        """The evidence, before the work.

        Emitted per connection that runs, so a failover's second leg gets one on
        exactly the same terms as the primary: a receipt event always precedes
        any event produced by the connection it describes.
        """
        return [
            ReceiptEvent(
                connection=self.connection.name,
                runtime=self.connection.runtime,
                auth_mode=self.receipt.effective_auth_mode,
                receipt=self.receipt,
            )
        ]

    def feed(self, event: AgentEvent) -> list[AgentEvent]:
        """Fold one adapter event; return what the caller should see for it."""
        if self.deadline is not None:
            # Every event that is not the receipt retires the first-token bound,
            # including a usage event and a vendor passthrough: the question
            # that bound asks is "is anything happening at all", and all of
            # those are something.
            self.deadline.saw_event()
            if self.deadline.fired is not None:
                self.stopped = self.timed_out()
                return []

        if isinstance(event, UsageEvent):
            guard_events = self.tracker.observe(event.usage, event.scope)
            out: list[AgentEvent] = [
                replace(event, cumulative=self.tracker.total),
                *guard_events,
            ]
            self._observe_usage(event)
            if self.tracker.stopped:
                self.stopped = self.terminal(
                    TerminalStatus.GUARD_STOP, self._guard_stop_reason()
                )
            return out

        if isinstance(event, TerminalEvent):
            # The adapter reports a status and, on the API runtimes, the two
            # typed facts behind it; the bridge owns the stamp and the verdict
            # computed from them.
            self.stopped = self.terminal(
                event.status,
                self._adapter_reason(event),
                status_code=event.status_code,
                retry_after=event.retry_after,
            )
            return []

        if isinstance(event, VendorEvent) and event.name == "rate_limit":
            # Observed on the way past, never intercepted: the event still
            # reaches the caller exactly as before (D20).
            self.allowance = event.data

        if isinstance(event, VendorEvent) and event.name == "reasoning_echo":
            # Same treatment, same reason. ``openai-sdk`` is the only runtime
            # that answers with the effort it is running at, so this is the one
            # place a stamped terminal can say the vendor agreed -- or did not.
            echoed = event.data.get("echoed")
            if isinstance(echoed, str) and echoed:
                self.reasoning_echo = echoed

        return [event]

    def vendor_failed(self, exc: VendorRunFailed) -> None:
        """**A vendor-reported failure is a run outcome, not a caller mistake.**

        The run happened, it may well have spent tokens, and it ended badly -- so
        it ends the way every other outcome ends: one terminal event, stamped,
        carrying the usage accumulated up to this point (deviation 2 amendment,
        2026-08-17).

        A cancelled adapter often reports the teardown as a vendor failure.
        Where the bound is what cancelled it, the run timed out; saying "the
        transport closed" instead would name the symptom and hide the cause the
        caller configured.
        """
        if self.deadline is not None and self.deadline.fired is not None:
            self.stopped = self.timed_out()
            return
        self.stopped = self._vendor_terminal(exc)

    def failed(self, exc: Exception) -> Exception | None:
        """An exception that is not a vendor failure: the bug arm, or the bound.

        Returns the exception the pump should raise, or ``None`` where the fold
        absorbed it -- which happens only when the caller's own bound tore the
        adapter down, and calling that an adapter bug would be both wrong and
        unactionable.
        """
        if self.deadline is not None and self.deadline.fired is not None:
            self.stopped = self.timed_out()
            return None
        return self._bug(exc)

    def finish(self) -> TerminalEvent:
        """The terminal this run ended with; ``ok`` where nothing else decided."""
        if self.stopped is None:
            self.stopped = self.terminal(TerminalStatus.OK, None)
        return self.stopped

    # --- the stamp --------------------------------------------------------------

    def terminal(
        self,
        status: TerminalStatus,
        reason: str | None,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> TerminalEvent:
        # The retryability verdict is stamped here, beside the auth mode and for
        # the same reason (D3): it is a statement modelpass is making about this
        # run, so the bridge owns it and an adapter cannot forge it. Computed
        # from typed fields only -- the status, the code, the connection's
        # policy -- never from ``reason``.
        connection = self.connection
        verdict = classify_terminal(
            connection.runtime,
            status,
            status_code=status_code,
            retry_after=retry_after,
            first_token=(
                self.deadline is not None and self.deadline.fired == "first_token"
            ),
            stateless=self.stateless,
            connection_retry=connection.retry,
        )
        # The reasoning trio is stamped here for the reason everything else in
        # this method is: it is modelpass's statement about the run, assembled
        # from the receipt (what was sent), the observed echo (what the vendor
        # said) and the folded usage (what it cost). An adapter cannot forge it,
        # and a consumer gets all three in one object rather than having to
        # join a receipt to a usage event after the fact.
        usage = self._terminal_usage()
        return TerminalEvent(
            status=status,
            connection=connection.name,
            runtime=connection.runtime,
            auth_mode=self.receipt.effective_auth_mode,
            reason=reason,
            usage=usage,
            retryable=verdict.retryable,
            status_code=status_code,
            retry_after=verdict.retry_after,
            reasoning_value=self.receipt.reasoning_value,
            reasoning_echo=self.reasoning_echo,
            reasoning_metric=reasoning_metric(
                connection.runtime, usage.reasoning_output_tokens
            ).value,
        )

    def timed_out(self) -> TerminalEvent:
        """The terminal for an expired bound, carrying what was spent first.

        The usage is read at the moment this is built, so a run that streamed
        usage events for two minutes and then stalled reports those two minutes'
        tokens. A timed-out run is still a billed run.
        """
        assert self.deadline is not None
        return self.terminal(TerminalStatus.TIMED_OUT, self.deadline.reason())

    # --- what the two faces answer differently ----------------------------------

    #: Whether a repeat of this run would be a repeat of nothing. A stateless
    #: call left nothing behind; a session turn left a conversation.
    stateless: bool = True

    def _terminal_usage(self) -> TokenUsage:
        return self.tracker.total

    def _observe_usage(self, event: UsageEvent) -> None:
        """A hook for a fold that keeps a second total beside the tracker's."""

    def _guard_stop_reason(self) -> str:
        return f"stopAtTokens threshold {self.guards.stop_at_tokens} reached"

    def _adapter_reason(self, event: TerminalEvent) -> str | None:
        return event.reason

    def _vendor_terminal(self, exc: VendorRunFailed) -> TerminalEvent:
        return self.terminal(TerminalStatus.ERROR, str(exc))

    def _bug(self, exc: Exception) -> Exception:
        # Nothing should reach here. An adapter has a documented way to report
        # that the vendor failed; anything else escaping run() means modelpass
        # or the adapter is broken, which is a bug report rather than a run
        # outcome.
        return AdapterFailed(
            f"adapter for runtime {self.connection.runtime.value!r} failed on "
            f"connection {self.connection.name!r}: {exc}"
        )


class RunFold(EventFold):
    """One leg of a stateless call."""

    stateless = True

    def _vendor_terminal(self, exc: VendorRunFailed) -> TerminalEvent:
        code, after = vendor_error_facts(exc)
        return self.terminal(
            TerminalStatus.ERROR, str(exc), status_code=code, retry_after=after
        )


class TurnFold(EventFold):
    """One turn of a session.

    ``stateless=False``: this turn left a conversation behind on the runtime, so
    even a first-token timeout is UNKNOWN rather than YES -- repeating it might
    duplicate a turn the vendor already recorded.
    """

    stateless = False

    def __init__(
        self,
        *,
        reason_for: Callable[[TerminalEvent], str | None],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._reason_for = reason_for
        #: ``usage`` on a turn's terminal is **this turn's**, not the session's.
        #: The terminal is what the run log records, and a ledger whose every
        #: line repeated a growing cumulative would count the same tokens once
        #: per turn.
        self.turn_usage = TokenUsage()
        #: Set when this turn's usage crossed the session's ceiling. The session
        #: reads it: the guard was a ceiling on the whole session, so it ends
        #: the conversation and not only the turn.
        self.session_stopped: str | None = None

    def _terminal_usage(self) -> TokenUsage:
        return self.turn_usage

    def _observe_usage(self, event: UsageEvent) -> None:
        if event.scope is UsageScope.RUN_TOTAL:
            self.turn_usage = self.turn_usage.at_least(event.usage)
        else:
            self.turn_usage = self.turn_usage + event.usage

    def _guard_stop_reason(self) -> str:
        reason = (
            f"stopAtTokens threshold {self.guards.stop_at_tokens} reached for this "
            "session"
        )
        self.session_stopped = reason
        return reason

    def _adapter_reason(self, event: TerminalEvent) -> str | None:
        return self._reason_for(event)

    def _bug(self, exc: Exception) -> Exception:
        return AdapterFailed(
            f"the session adapter for runtime {self.connection.runtime.value!r} failed "
            f"on connection {self.connection.name!r}: {exc}"
        )


# --- the whole call, which may be two legs ------------------------------------------


@dataclass(frozen=True, slots=True)
class Leg:
    """One connection's turn at a call: what to run, and what it promised."""

    adapter: Adapter
    request: RunRequest
    receipt: Receipt
    guards: Guards


#: What :meth:`modelpass.Bridge._begin_failover` answers: the announcement plus
#: everything the second leg needs, or a sentence saying why it cannot run.
BeginFailover = Callable[
    [Connection, RunRequest, TerminalEvent, TokenUsage],
    "tuple[FailoverEvent, Adapter, RunRequest, Receipt, Guards] | str",
]

#: What :meth:`modelpass.Bridge._record_run` takes. Keyword-only there; the
#: alias exists to name the seam rather than to type it precisely.
RecordRun = Callable[..., None]


class CallFold:
    """One :meth:`~modelpass.Bridge.chat` call: one leg, or two across a failover.

    Holds what a single leg cannot -- which connection runs next, which leg's
    line goes in the ledger, and what the *final* terminal means for a caller
    who asked to be raised at. The bridge is reached only through
    ``record_run`` and ``begin_failover``.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        request: RunRequest,
        receipt: Receipt,
        guards: Guards,
        failover: Connection | None,
        raise_on_stop: bool,
        deadline: Deadline | None,
        record_run: RecordRun,
        begin_failover: BeginFailover,
    ) -> None:
        self.leg = Leg(adapter, request, receipt, guards)
        self.failover = failover
        self.raise_on_stop = raise_on_stop
        self.deadline = deadline
        self._record_run = record_run
        self._begin_failover = begin_failover
        self._failed_over_from: str | None = None
        self.fold = self._new_fold()

    def _new_fold(self) -> RunFold:
        leg = self.leg
        receipt = leg.receipt
        # Moved with the binary, and for the same reason: these are facts about
        # the leg that is about to run, and a failover's second leg may answer
        # them differently -- a subscription runtime that drops breakpoints
        # failing over to one that takes them is exactly the case the pair
        # exists to make visible.
        self._binary = receipt.binary
        self._breakpoints = (
            receipt.cache_breakpoints_requested,
            receipt.cache_breakpoints_honoured,
        )
        self._sampling = (
            receipt.sampling_requested,
            receipt.sampling_applied,
            receipt.sampling_notes,
        )
        return RunFold(
            connection=leg.request.connection,
            receipt=receipt,
            guards=leg.guards,
            tracker=GuardTracker(leg.guards, leg.request.connection.name),
            deadline=self.deadline,
        )

    # --- the pump's three questions ---------------------------------------------

    def begin(self) -> list[AgentEvent]:
        if self.deadline is not None:
            # Leg 1 starts the clock; leg 2 only changes the adapter it can
            # cancel. The clock keeps running across a failover: one call on two
            # connections is one call, and restarting it would let a bounded
            # call take twice its bound and still report that it honoured it.
            if self._failed_over_from is None:
                self.deadline.start(self.leg.adapter)
            else:
                self.deadline.bind(self.leg.adapter)
        return self.fold.begin()

    def advance(self) -> tuple[list[AgentEvent], bool]:
        """This leg is over. Say what the caller sees next, and whether to run again.

        A ``quota_exhausted`` on a connection with a configured failover is not
        the end of anything: the call continues on the second connection, and
        the stream must still finish with exactly one terminal event.
        """
        stopped = self.fold.finish()
        if stopped.status is not TerminalStatus.QUOTA_EXHAUSTED or self.failover is None:
            return [], False

        switch = self._begin_failover(
            self.failover, self.leg.request, stopped, self.fold.tracker.total
        )
        if isinstance(switch, str):
            # The target could not be made ready. The run still stopped cleanly
            # on quota -- that part is true and is reported as such; the failure
            # to fail over rides on the same terminal so it is impossible to
            # read one without the other.
            why = stopped.reason or "quota exhausted"
            self.fold.stopped = replace(stopped, reason=f"{why}; {switch}")
            return [], False

        event, adapter, request, receipt, guards = switch
        # The first connection's leg is recorded here, before the second one
        # starts. One call that touched two connections spent two allowances,
        # and a ledger that logged only the connection that *finished* would
        # report the subscription's tokens as the metered connection's -- or
        # lose them entirely. Two legs, two lines, each with the auth mode that
        # actually paid for it.
        self.record()
        self._failed_over_from = event.from_connection
        # Leg 2 resolves the same request against a different runtime and
        # usually a different model, so what it applied is its own answer. A
        # subscription failing over onto an API key is exactly where the two
        # legs disagree: leg 1 could apply nothing at all.
        self.leg = Leg(adapter, request, receipt, guards)
        # **A failover cannot chain.** Leg 2's own guards already force a clean
        # stop, and this is the same rule stated where the loop can see it: one
        # call crosses at most one connection boundary, so a second exhausted
        # allowance ends the call rather than starting a third leg.
        self.failover = None
        self.fold = self._new_fold()
        return [event], True

    def record(self) -> None:
        """Write this leg's line to the ledger."""
        self._record_run(
            self.leg.request,
            self.fold.finish(),
            self.leg.guards,
            self.fold.allowance,
            binary=self._binary,
            breakpoints=self._breakpoints,
            sampling=self._sampling,
        )

    def finish(self) -> list[AgentEvent]:
        """The durable half of D3's stamp, then the terminal the caller sees.

        The terminal event proves how this run was billed to whoever is holding
        the iterator right now; the log is what remains once they let go of it
        (2026-08-17).
        """
        stopped = self.fold.finish()
        if self._failed_over_from is not None:
            stopped = replace(stopped, failed_over_from=self._failed_over_from)
            self.fold.stopped = stopped
        self.record()
        return [stopped]

    def raise_for_stop(self) -> None:
        """``raise_on_stop``'s half of the contract, read off the final terminal."""
        if not self.raise_on_stop:
            return
        raise_for(
            self.fold.finish(),
            guards=self.leg.guards,
            tracker=self.fold.tracker,
            connection=self.leg.request.connection.name,
            deadline=self.deadline,
            timeout_reason="the run timed out",
        )


def raise_for(
    stopped: TerminalEvent,
    *,
    guards: Guards,
    tracker: GuardTracker,
    connection: str,
    deadline: Deadline | None,
    timeout_reason: str,
) -> None:
    """Turn a stop into the exception ``raise_on_stop`` promised, or return.

    One statement of the three, shared by the stateless call and the session
    turn, because a consumer matching on these classes must not learn which face
    produced them.

    Each carries the terminal it was raised for (ticket 1.12b), so the verdict
    the bridge already stamped survives the raise instead of being re-derived
    from the exception's class by whoever catches it.
    """
    if stopped.status is TerminalStatus.GUARD_STOP:
        raise GuardStop(
            stopped.reason or "guard stop",
            observed=tracker.total.total_tokens,
            threshold=guards.stop_at_tokens or 0,
        ).with_terminal(stopped)
    if stopped.status is TerminalStatus.QUOTA_EXHAUSTED:
        raise QuotaExhausted(connection, stopped.reason or "").with_terminal(stopped)
    if stopped.status is TerminalStatus.TIMED_OUT:
        # Carries the spend, like the two above: a bounded run that was cut off
        # still cost what it cost, and a caller deciding whether to try again
        # needs to know what the first attempt was worth.
        raise RunTimedOut(
            stopped.reason or timeout_reason,
            connection=connection,
            which=(deadline.fired or "total") if deadline is not None else "total",
            seconds=None if deadline is None else deadline.seconds(),
            usage=stopped.usage,
        ).with_terminal(stopped)
