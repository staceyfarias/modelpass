"""Retryability verdicts: one table, typed inputs, no message matching (R6).

**modelpass never retries anything.** This module answers one question --
*would trying that again be sensible?* -- and hands the answer to the caller,
who is the only party that knows what a second run is worth against their own
allowance. There is no loop here, no backoff, and no configuration for either.

The whole reason it exists is a defect that has already happened. A consumer
classified retryability by substring-matching status codes out of the error
*message*; ``"500"`` matched inside ``"stopAtTokens threshold 500000
reached"``, so a deterministic guard stop was retried as a transient HTTP 500 --
four allowance-burning runs for one logical call (reported by a downstream
agent host, 2026-08-18; the fix pattern came from a retrieval service). Two
consumers have since reimplemented the same classifier by hand, one copied from
the other. It belongs here, once.

So the rule this module is built around, and the only rule that matters:

    **A verdict is computed from typed facts only** -- an HTTP status code, an
    exception class, a terminal status, a connection's declared policy. Never
    from the text of a message. There is no regex in this file, no ``in``
    against a string, and no place where one could be added without the tests
    noticing.

Every row below carries the date it was written and why it reads the way it
does, because a retry table without a rationale is a table nobody dares change.

``Retryable`` itself is defined in :mod:`modelpass.types`, next to
:class:`~modelpass.types.TerminalStatus` and for the same reason -- it is a
vocabulary that appears on an event -- and is re-exported here so a caller
reading a verdict imports the verdict and its vocabulary from one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .runtimes import Runtime
from .types import Retryable, TerminalStatus

__all__ = [
    "NEVER",
    "RetryVerdict",
    "Retryable",
    "classify_error",
    "classify_terminal",
    "family_of",
    "vendor_error_facts",
]

#: The connection-level policy value that forces every verdict to
#: :attr:`~modelpass.types.Retryable.NO` (config key ``retry``, ticket 1.12).
NEVER = "never"


@dataclass(frozen=True, slots=True)
class RetryVerdict:
    """A verdict, the seconds the vendor asked for, and why it reads that way.

    ``note`` is one sentence naming the rule that fired. It is for a human
    reading a log, never for a caller to match on -- the machine-readable half
    is :attr:`retryable` and :attr:`retry_after`, both typed.
    """

    retryable: Retryable = Retryable.UNKNOWN
    retry_after: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "retryable": self.retryable.value,
            "retry_after": self.retry_after,
            "note": self.note,
        }


# --- adapter families ----------------------------------------------------------
#
# A family, not a runtime, because the rows are about what a transport *reports*
# and the four API runtimes report the same three things: an HTTP status code on
# a typed SDK exception, a ``retry-after`` header, or nothing. Splitting the
# table four ways would be four copies of one row set, and the copy that drifted
# would be the one nobody noticed.

#: The four metered runtimes: a vendor SDK, typed exceptions, HTTP semantics.
API = "api"
#: The two subscription runtimes: a child process or an SDK that drives one.
#: They report no status codes at all, which is why their verdicts are mostly
#: :attr:`~modelpass.types.Retryable.UNKNOWN`.
AGENT = "agent"

_FAMILY: dict[Runtime, str] = {
    Runtime.ANTHROPIC_SDK: AGENT,
    Runtime.OPENAI_SDK: AGENT,
    Runtime.ANTHROPIC_API: API,
    Runtime.OPENAI_API: API,
    Runtime.GOOGLE_API: API,
    Runtime.OPENAI_COMPATIBLE: API,
    # Gated runtimes with no adapter (D5). Listed so the map is the whole enum
    # and a member added later is the only thing that falls to the default.
    Runtime.GOOGLE_CLI: AGENT,
    Runtime.GOOGLE_SDK: AGENT,
}


def family_of(runtime: Runtime) -> str:
    """Which classification family this runtime belongs to.

    Unknown members answer :data:`AGENT`, which is the conservative arm: it
    yields ``UNKNOWN`` rather than ``YES`` for anything it has no row for, and a
    runtime nobody has classified should never be told a retry is safe.
    """
    return _FAMILY.get(runtime, AGENT)


# --- the table -----------------------------------------------------------------
#
# Read as: (family, terminal status) -> verdict, with the HTTP status code
# refining the ERROR row on the API family. Each entry is dated.

_TERMINAL_ROWS: dict[tuple[str, TerminalStatus], tuple[Retryable, str]] = {
    # 2026-09-13. An ok run is not a candidate for anything.
    (API, TerminalStatus.OK): (Retryable.NO, "the run succeeded"),
    (AGENT, TerminalStatus.OK): (Retryable.NO, "the run succeeded"),
    # 2026-09-13. A guard stop is deterministic by construction: the same call
    # with the same ceiling reaches the same ceiling. This is the exact row the
    # 2026-08-18 incident above retried four times.
    (API, TerminalStatus.GUARD_STOP): (
        Retryable.NO,
        "a guard stop is deterministic; the same call hits the same ceiling",
    ),
    (AGENT, TerminalStatus.GUARD_STOP): (
        Retryable.NO,
        "a guard stop is deterministic; the same call hits the same ceiling",
    ),
    # 2026-09-13. An exhausted allowance comes back when the plan's window
    # resets and not before, so an immediate retry spends nothing but patience.
    # Where a failover is configured the bridge has already used it before this
    # terminal was stamped, so reaching here means there was none or it refused.
    (API, TerminalStatus.QUOTA_EXHAUSTED): (
        Retryable.NO,
        "the allowance is gone until the plan's window resets; failover, where "
        "one is configured, has already been tried",
    ),
    (AGENT, TerminalStatus.QUOTA_EXHAUSTED): (
        Retryable.NO,
        "the allowance is gone until the plan's window resets; failover, where "
        "one is configured, has already been tried",
    ),
    # 2026-09-13. Somebody let go of the iterator (D10). Retrying a run the
    # caller deliberately stopped is the library second-guessing the caller.
    (API, TerminalStatus.CANCELLED): (
        Retryable.NO,
        "the caller cancelled this run",
    ),
    (AGENT, TerminalStatus.CANCELLED): (
        Retryable.NO,
        "the caller cancelled this run",
    ),
}

#: HTTP status rows for the API family, 2026-09-13. Keys are exact codes; the
#: range fallbacks below handle everything else.
_STATUS_ROWS: dict[int, tuple[Retryable, str]] = {
    # 2026-09-13. 408 and 504 are the server saying the same thing the local
    # timeout says: nothing arrived in time. Nothing was completed, so nothing
    # is duplicated by asking again.
    408: (Retryable.YES, "HTTP 408 request timeout, nothing was completed"),
    409: (Retryable.YES, "HTTP 409 conflict, transient on these endpoints"),
    # 2026-09-13. 429 never reaches here as an ERROR: every API adapter types it
    # as QUOTA_EXHAUSTED off the status code, which is the row above. The entry
    # exists so an endpoint that returns 429 on some other path still lands
    # somewhere deliberate, and it is YES *only* with a retry-after -- a rate
    # limit that named a wait is transient, one that named nothing may be a
    # spent allowance wearing a rate limit's number.
    429: (Retryable.YES, "HTTP 429 rate limit"),
    500: (Retryable.YES, "HTTP 500, a server-side fault"),
    502: (Retryable.YES, "HTTP 502 bad gateway"),
    503: (Retryable.YES, "HTTP 503 service unavailable"),
    504: (Retryable.YES, "HTTP 504 gateway timeout"),
    # 2026-09-13. The 4xx rows are the ones that protect an allowance: a
    # malformed request, a bad key and a refused model are all deterministic,
    # and a loop around them is a loop that pays four times for one mistake.
    400: (Retryable.NO, "HTTP 400, the request itself is wrong"),
    401: (Retryable.NO, "HTTP 401, the credential is not accepted"),
    403: (Retryable.NO, "HTTP 403, this credential may not do that"),
    404: (Retryable.NO, "HTTP 404, no such endpoint or model"),
    413: (Retryable.NO, "HTTP 413, the request is too large as written"),
    422: (Retryable.NO, "HTTP 422, the request is not processable as written"),
}


def _status_verdict(code: int) -> tuple[Retryable, str] | None:
    row = _STATUS_ROWS.get(code)
    if row is not None:
        return row
    # 2026-09-13. Ranges, so an unlisted code still gets a considered answer:
    # every 4xx is the caller's request being wrong in some way the server has a
    # narrower word for, and every 5xx is the server's own fault.
    if 400 <= code < 500:
        return (Retryable.NO, f"HTTP {code}, a client-side refusal")
    if 500 <= code < 600:
        return (Retryable.YES, f"HTTP {code}, a server-side fault")
    return None


#: Exception classes whose *name* decides the verdict, 2026-09-13.
#:
#: Matched by name across the MRO rather than by ``isinstance``, deliberately: a
#: retrieval service and a desktop agent app both match modelpass's own error
#: classes that way so the library stays an optional import, and the vendor SDK
#: classes named here are not importable at all unless that extra is installed.
#: A name is what both ends can agree on.
_ERROR_NAME_ROWS: dict[str, tuple[Retryable, str]] = {
    # --- modelpass's own, the ones a consumer already matches on ---------------
    "GuardStop": (Retryable.NO, "a guard stop is deterministic"),
    "QuotaExhausted": (Retryable.NO, "the allowance is gone until it resets"),
    "RunTimedOut": (
        Retryable.UNKNOWN,
        "the bound ran out; whether the run was going to succeed is unknown",
    ),
    "SessionBusy": (
        Retryable.NO,
        "a Session takes one caller at a time; the fix is the call site, not a "
        "retry",
    ),
    "AuthModeMismatch": (Retryable.NO, "the connection would bill differently"),
    "UnsafeLaunch": (Retryable.NO, "a forbidden launch argument"),
    "CapabilityNotSupported": (Retryable.NO, "the runtime cannot do that"),
    "StructuredOutputRejected": (Retryable.NO, "the endpoint refuses this schema"),
    "InvalidSchema": (Retryable.NO, "the schema is malformed"),
    "InvalidTool": (Retryable.NO, "the tool definition is malformed"),
    "InvalidGuards": (Retryable.NO, "a guard argument is unusable"),
    "InvalidSession": (Retryable.NO, "the session arguments contradict each other"),
    "InvalidConnection": (Retryable.NO, "the connection definition is unusable"),
    "NoSuchConnection": (Retryable.NO, "no such connection is configured"),
    "NoSuchSecret": (Retryable.NO, "no such secret entry"),
    "ConnectionDisabled": (Retryable.NO, "the connection is out of service"),
    "RuntimeGated": (Retryable.NO, "the runtime is gated behind an opt-in"),
    "RuntimeNotAvailable": (Retryable.NO, "the vendor package is not installed"),
    "SessionClosed": (Retryable.NO, "the session is over"),
    "SessionNotFound": (Retryable.NO, "the runtime no longer has that session"),
    "PreflightFailed": (Retryable.NO, "the preflight could not vouch for the run"),
    "AdapterFailed": (
        Retryable.UNKNOWN,
        "an adapter broke in a way nobody accounted for; this is a bug report",
    ),
    "AdapterNotImplemented": (Retryable.NO, "that adapter method does not exist yet"),
    # --- the agent transports (2026-09-13) ------------------------------------
    # Deliberately UNKNOWN rather than YES, and this is the one place the table
    # departs from ordinary HTTP intuition. A closed transport *is* a connection
    # reset, which on a metered endpoint would be YES. On a subscription runtime
    # it is not: the child process may have spent allowance before the pipe
    # died, neither vendor reports how much, and a caller who retries on a YES
    # from here pays twice for a turn they cannot see. UNKNOWN hands that
    # judgement to the consumer, who at least knows what their plan is worth.
    "AppServerTransportClosed": (
        Retryable.UNKNOWN,
        "the app-server transport closed; a subscription run may already have "
        "spent allowance nobody can account for",
    ),
    "AppServerTimeout": (
        Retryable.UNKNOWN,
        "the app server did not answer in time; the run may still have spent",
    ),
    "AppServerRequestFailed": (
        Retryable.UNKNOWN,
        "the app server refused the request without a typed reason",
    ),
    # --- vendor SDK classes, by name (2026-09-13) -----------------------------
    # Named because the SDKs are optional imports. Every one of these also
    # carries a ``status_code``, which outranks the name below; the rows are the
    # fallback for the shapes that do not.
    "APIConnectionError": (Retryable.YES, "the connection to the endpoint failed"),
    "APITimeoutError": (Retryable.YES, "the SDK's own request timeout fired"),
    "ConnectionResetError": (Retryable.YES, "the connection was reset"),
    "ConnectionError": (Retryable.YES, "the connection failed"),
    "TimeoutError": (Retryable.YES, "a transport timeout"),
    "ServiceUnavailable": (Retryable.YES, "the service reported itself unavailable"),
    "InternalServerError": (Retryable.YES, "a server-side fault"),
    "RateLimitError": (Retryable.YES, "the endpoint reported a rate limit"),
    "AuthenticationError": (Retryable.NO, "the credential is not accepted"),
    "PermissionDeniedError": (Retryable.NO, "this credential may not do that"),
    "NotFoundError": (Retryable.NO, "no such endpoint or model"),
    "BadRequestError": (Retryable.NO, "the request itself is wrong"),
    "UnprocessableEntityError": (Retryable.NO, "the request is not processable"),
}


def classify_terminal(
    runtime: Runtime,
    status: TerminalStatus,
    *,
    status_code: int | None = None,
    retry_after: float | None = None,
    first_token: bool = False,
    stateless: bool = True,
    connection_retry: str | None = None,
) -> RetryVerdict:
    """The verdict for a terminal event, from typed facts only.

    ``first_token`` says a :attr:`~modelpass.types.TerminalStatus.TIMED_OUT`
    terminal came from the ``first_token`` bound rather than the ``total`` one,
    and ``stateless`` says the run held no server-side conversation. The pair is
    the one case the ticket calls transient outright: nothing answered at all,
    on a run that left nothing behind, so trying again repeats a request rather
    than duplicating a half-finished one. A ``total`` expiry is ``UNKNOWN``,
    because the run may well have been most of the way through an answer.

    ``connection_retry`` is the connection's declared policy and outranks every
    row: :data:`NEVER` forces :attr:`~modelpass.types.Retryable.NO`.
    """
    family = family_of(runtime)
    verdict = _terminal_verdict(
        family, status, status_code, retry_after, first_token, stateless
    )
    return _apply_policy(verdict, connection_retry)


def _terminal_verdict(
    family: str,
    status: TerminalStatus,
    status_code: int | None,
    retry_after: float | None,
    first_token: bool,
    stateless: bool,
) -> RetryVerdict:
    if status is TerminalStatus.TIMED_OUT:
        if first_token and stateless:
            return RetryVerdict(
                Retryable.YES,
                retry_after,
                "nothing answered before the first-token bound on a stateless "
                "run, so nothing was left half-done",
            )
        return RetryVerdict(
            Retryable.UNKNOWN,
            retry_after,
            "the run was cut off part-way; how far it got is not knowable here",
        )

    row = _TERMINAL_ROWS.get((family, status))
    if row is not None:
        verdict, note = row
        # 2026-09-13. A vendor that named a wait rides it out even on a NO row:
        # the number is the vendor's, and dropping it would lose information the
        # caller might want for a queue delay that is not a retry.
        return RetryVerdict(verdict, retry_after, note)

    if status is not TerminalStatus.ERROR:  # pragma: no cover - enum is closed
        return RetryVerdict(Retryable.UNKNOWN, retry_after, "unclassified status")

    if family == API and status_code is not None:
        code_row = _status_verdict(status_code)
        if code_row is not None:
            verdict, note = code_row
            if status_code == 429 and retry_after is None:
                return RetryVerdict(
                    Retryable.UNKNOWN,
                    None,
                    "HTTP 429 with no retry-after; a rate limit that named no "
                    "wait may be a spent allowance wearing a rate limit's number",
                )
            return RetryVerdict(verdict, retry_after, note)

    # 2026-09-13. The honest default. An agent runtime reports no status code at
    # all, and an API error the SDK did not type is one nobody here has
    # classified -- in both cases UNKNOWN is the true answer and YES would be a
    # guess made with somebody else's allowance.
    return RetryVerdict(
        Retryable.UNKNOWN,
        retry_after,
        "the vendor reported nothing typed to classify this by",
    )


def classify_error(
    exc: BaseException, *, connection_retry: str | None = None
) -> RetryVerdict:
    """The verdict for an exception, from its class and its status code.

    The status code, where the exception carries one, outranks the class name --
    the same precedence a consumer's own fix established, and for the same
    reason: a structured field is the vendor's own statement and a name is our
    reading of it.

    A terminal event carried on the exception (``exc.terminal``, ticket 1.12b)
    outranks both: it *is* this library's verdict for this run, stamped by the
    bridge from the same table, and re-deriving one from the exception's class
    would be a second reading of a question already answered -- the reading that
    answered ``UNKNOWN`` for a 429 and never retried it. Read by ``getattr`` so
    the LangChain leaf's errors, which are ``RuntimeError`` and not
    :class:`~modelpass.errors.SubpassError`, are classified by the same rule.
    """
    carried = _carried_verdict(exc)
    if carried is not None:
        return _apply_policy(carried, connection_retry)

    status_code, retry_after = vendor_error_facts(exc)
    verdict: RetryVerdict | None = None

    # 2026-09-13, amended 2026-09-15. The one exception whose own typed field
    # refines its row: a RunTimedOut names which bound fired, and a first-token
    # expiry is the only timeout this table calls transient outright. Matched by
    # class name for the reason the whole name table gives -- a consumer holding
    # this verdict may not have imported modelpass's errors module at all.
    #
    # **Reached only where no terminal rode along**, which since 1.12b is never
    # an error this library raised: _fold.raise_for stamps every RunTimedOut
    # with its terminal, and the carried verdict above outranks this row. What
    # is left for it is an exception somebody else built or one whose event did
    # not survive a process boundary -- and for those this row assumes the run
    # was stateless, because an exception alone cannot say otherwise. The
    # carried verdict can: a first-token expiry on a *session* turn reads
    # UNKNOWN, since the turn left a conversation behind and a repeat is not a
    # repeat of nothing. That difference is the 0.2.1 behaviour change.
    if type(exc).__name__ == "RunTimedOut" and getattr(exc, "which", None) == "first_token":
        return _apply_policy(
            RetryVerdict(
                Retryable.YES,
                retry_after,
                "nothing answered before the first-token bound, so nothing was "
                "left half-done",
            ),
            connection_retry,
        )

    if status_code is not None:
        row = _status_verdict(status_code)
        if row is not None:
            retryable, note = row
            if status_code == 429 and retry_after is None:
                verdict = RetryVerdict(
                    Retryable.UNKNOWN,
                    None,
                    "HTTP 429 with no retry-after; a rate limit that named no "
                    "wait may be a spent allowance wearing a rate limit's number",
                )
            else:
                verdict = RetryVerdict(retryable, retry_after, note)

    if verdict is None:
        for klass in type(exc).__mro__:
            row = _ERROR_NAME_ROWS.get(klass.__name__)
            if row is not None:
                retryable, note = row
                verdict = RetryVerdict(retryable, retry_after, note)
                break

    if verdict is None:
        verdict = RetryVerdict(
            Retryable.UNKNOWN,
            retry_after,
            f"{type(exc).__name__} is not in the retry table",
        )
    return _apply_policy(verdict, connection_retry)


def _carried_verdict(exc: BaseException) -> RetryVerdict | None:
    """The verdict on a terminal event this exception carries, if it carries one."""
    terminal = getattr(exc, "terminal", None)
    retryable = getattr(terminal, "retryable", None)
    if not isinstance(retryable, Retryable):
        return None
    return RetryVerdict(
        retryable,
        _seconds(getattr(terminal, "retry_after", None)),
        f"the run ended {getattr(terminal, 'status', '')!s} and the terminal "
        "event carried this verdict",
    )


def _apply_policy(verdict: RetryVerdict, connection_retry: str | None) -> RetryVerdict:
    """Force ``NO`` where the connection declared ``retry = "never"``.

    The override is applied last and names itself in the note, so a caller
    reading ``NO`` can tell a classified refusal from a policy one.
    """
    if connection_retry != NEVER:
        return verdict
    return RetryVerdict(
        Retryable.NO,
        verdict.retry_after,
        f'the connection declares retry = "never" (would otherwise be '
        f"{verdict.retryable.value}: {verdict.note})",
    )


def vendor_error_facts(exc: BaseException) -> tuple[int | None, float | None]:
    """The HTTP status and ``retry-after`` a vendor exception carries, if any.

    Both read off *attributes*, never off ``str(exc)``. ``anthropic``,
    ``openai`` and ``google-genai`` all name the status ``status_code``; Google
    also uses ``code``. ``retry-after`` arrives as a response header and is
    seconds, sometimes as a string, and a header that is an HTTP-date rather
    than a delta is reported as absent rather than guessed at.
    """
    status_code = _int_attr(exc, "status_code")
    if status_code is None:
        status_code = _int_attr(exc, "code")
    return status_code, _retry_after(exc)


def _int_attr(exc: BaseException, name: str) -> int | None:
    value = getattr(exc, name, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _retry_after(exc: BaseException) -> float | None:
    direct = _seconds(getattr(exc, "vendor_retry_after", None))
    if direct is not None:
        return direct
    # Read ``retry_after`` only where it is a plain attribute. On modelpass's own
    # errors it is a *property* that asks this module for a verdict, and reading
    # it here would be this function calling itself through the exception --
    # which it did, once, before this guard (2026-09-13).
    if not isinstance(getattr(type(exc), "retry_after", None), property):
        direct = _seconds(getattr(exc, "retry_after", None))
        if direct is not None:
            return direct
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-after"):
        if key in headers:
            seconds = _seconds(headers[key])
            if seconds is not None:
                return seconds
    return None


def _seconds(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            # An HTTP-date form raises here, which is the intended outcome: a
            # date is not a delta and converting one would be a guess.
            seconds = float(value.strip())
        except ValueError:
            return None
        return seconds if seconds >= 0 else None
    return None
