"""The wall-clock watchdog behind ``timeout=`` (R6, ticket 1.12).

Its own module because two callers need it and they are on opposite sides of an
import cycle: :mod:`modelpass.bridge` bounds a stateless call, and
:mod:`modelpass.sessions` bounds one turn, and ``bridge`` already imports
``sessions``.
"""

from __future__ import annotations

import threading
import time

from .adapters.base import Adapter
from .types import Timeout

__all__ = ["Deadline"]


class Deadline:
    """The wall-clock watchdog behind ``timeout=`` (R6, ticket 1.12).

    **A timer thread that calls the adapter's own** ``cancel()`` **and nothing
    else.** That is the whole mechanism, and the shape was taken from the two
    consumers who had already built it by hand -- a retrieval service and a
    desktop agent app, both with a threading watchdog around a bounded call.
    Both learned the same thing the hard way, and one of them says it outright
    in a comment -- **the watchdog must not close the iterator**. A generator
    closed from one thread while another is executing it raises ``ValueError``,
    so the timer cancels the
    *adapter* (the D10 floor: close the stream, terminate the process) and the
    consuming thread, unblocked, leaves its own loop and does the closing.

    Why a thread at all: the bound has to fire while the consumer is blocked
    inside ``next()`` on a runtime that has stopped answering, which is the case
    the whole feature exists for. A clock checked between events never fires
    there, and that is precisely the hang a batch scraping tool works around
    today with ``multiprocessing`` and ``join(timeout)``.

    One deadline spans the whole call, failover included -- ``total`` means the
    run's wall clock, not a leg's -- so the adapter it points at is rebound when
    a second leg starts.
    """

    def __init__(self, timeout: Timeout) -> None:
        self._timeout = timeout
        self._lock = threading.Lock()
        self._adapter: Adapter | None = None
        self._timer: threading.Timer | None = None
        self._fired: str | None = None
        self._first_seen = False
        self._stopped = False
        self._started = time.monotonic()

    # --- what the bridge drives it with ---------------------------------------

    def start(self, adapter: Adapter) -> None:
        """Begin the clock, aimed at this leg's adapter."""
        with self._lock:
            self._adapter = adapter
            self._started = time.monotonic()
            self._schedule()

    def bind(self, adapter: Adapter) -> None:
        """Aim at a second leg's adapter without restarting the clock."""
        with self._lock:
            self._adapter = adapter

    def saw_event(self) -> None:
        """Record the first non-receipt event, retiring the first-token bound."""
        with self._lock:
            if self._first_seen:
                return
            self._first_seen = True
            self._schedule()

    def stop(self) -> None:
        """Cancel the timer. Idempotent, and always called from a ``finally``."""
        with self._lock:
            self._stopped = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    # --- what it reports -------------------------------------------------------

    @property
    def fired(self) -> str | None:
        """``"total"``, ``"first_token"``, or ``None`` if the bound still holds."""
        with self._lock:
            return self._fired

    def seconds(self) -> float | None:
        """The bound that fired, in seconds, for the error that reports it."""
        which = self.fired or "total"
        return self._timeout.first_token if which == "first_token" else self._timeout.total

    def reason(self) -> str:
        which = self.fired or "total"
        seconds = self._timeout.first_token if which == "first_token" else self._timeout.total
        if which == "first_token":
            return (
                f"timed out: nothing answered within the first_token bound of "
                f"{seconds}s"
            )
        return f"timed out: the total bound of {seconds}s expired"

    # --- the timer ------------------------------------------------------------

    def _schedule(self) -> None:
        """Arm the timer for whichever bound expires first. Lock held."""
        if self._stopped or self._fired is not None:
            return
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        elapsed = time.monotonic() - self._started
        candidates: list[tuple[float, str]] = []
        if self._timeout.total is not None:
            candidates.append((self._timeout.total - elapsed, "total"))
        if self._timeout.first_token is not None and not self._first_seen:
            candidates.append((self._timeout.first_token - elapsed, "first_token"))
        if not candidates:
            return
        delay, which = min(candidates)
        timer = threading.Timer(max(delay, 0.0), self._fire, args=(which,))
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _fire(self, which: str) -> None:
        with self._lock:
            if self._fired is not None or self._stopped:
                return
            self._fired = which
            adapter = self._adapter
        # Outside the lock, deliberately: ``cancel()`` on a real adapter
        # terminates a process or closes a transport, and holding a lock across
        # somebody else's teardown is how a watchdog becomes the hang.
        if adapter is not None:
            adapter.cancel()
