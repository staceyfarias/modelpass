"""Error taxonomy for modelpass.

Every error raised by modelpass core derives from :class:`SubpassError`, so a caller
can wrap the whole library in one ``except`` without catching unrelated failures.

The taxonomy is deliberately shallow. The distinctions that matter are the ones a
caller might *act* on: "you asked for a connection that does not exist", "the run
was about to bill differently than you asked", "this runtime cannot do that",
"a guard stopped the run", "the subscription allowance is gone".
"""

from __future__ import annotations

from typing import Any, Self

__all__ = [
    "AdapterFailed",
    "AdapterNotImplemented",
    "AuthModeMismatch",
    "CapabilityNotSupported",
    "ConfigError",
    "ConnectionDisabled",
    "CredentialRefIsSecret",
    "DuplicateConnection",
    "GroupUnavailable",
    "GuardStop",
    "InvalidConnection",
    "InvalidGuards",
    "InvalidSchema",
    "InvalidSession",
    "InvalidTool",
    "NoSuchConnection",
    "NoSuchGroup",
    "NoSuchSecret",
    "PreflightFailed",
    "QuotaExhausted",
    "RunTimedOut",
    "RuntimeGated",
    "RuntimeNotAvailable",
    "SecretStillReferenced",
    "SessionBusy",
    "SessionClosed",
    "SessionNotFound",
    "StructuredOutputRejected",
    "SubpassError",
    "UnsafeLaunch",
    "VendorRunFailed",
]


class SubpassError(Exception):
    """Base class for every error modelpass raises.

    Every error carries a retryability verdict (R6, ticket 1.12). It is a
    *property* rather than a stored field so it cannot drift from the table in
    :mod:`modelpass.retry`, and it is computed from this exception's class and
    its typed attributes -- never from its message. :attr:`retry_after` is the
    seconds the vendor asked for, where one said so.

    Where the error stands in for a terminal event -- everything
    :meth:`~modelpass.Bridge.ask` and ``chat(raise_on_stop=True)`` raise -- the
    event itself rides along on :attr:`terminal` and its verdict is the one
    these properties report (ticket 1.12b). See :meth:`with_terminal`.

    **modelpass does not act on the verdict.** It never retries anything; the
    caller decides, because the caller is the one who knows what a second run
    costs against their own allowance.
    """

    #: Set by the raiser where the connection declared ``retry = "never"``, so
    #: the verdict below can honour a policy the error itself came from. ``None``
    #: means "no policy was attached", not "default" -- an error raised outside
    #: any connection has no policy to report.
    connection_retry: str | None = None
    #: Set by the raiser where the vendor named a wait it could not put on the
    #: exception itself. ``None`` everywhere else.
    vendor_retry_after: float | None = None
    #: The :class:`~modelpass.types.TerminalEvent` this error was raised for,
    #: where one exists (ticket 1.12b). ``None`` for an error raised before any
    #: run, or beside one. It is the whole event and not a copy of three of its
    #: fields, so nothing the terminal knew is lost on the way out of
    #: :meth:`~modelpass.Bridge.ask`.
    terminal: Any = None
    #: The HTTP status the vendor reported, where the terminal carried one.
    #: ``None`` on the agent runtimes, which report none.
    status_code: int | None = None

    def with_terminal(self, terminal: Any) -> Self:
        """Carry ``terminal``'s verdict on this error, and return it (1.12b).

        **A verdict that does not survive the raise is not a verdict.** The
        stream's terminal event knows the three typed facts -- the status code,
        the wait the vendor asked for, and the verdict computed from them -- and
        without this the exception :meth:`~modelpass.Bridge.ask` raises in its
        place knew none of them, so a consumer classifying what they actually
        hold read ``UNKNOWN`` and a rate limit was never retried (reported by a
        RAG evaluation harness, 2026-09-13).

        Returns ``self`` so a raiser writes ``raise Err(...).with_terminal(t)``
        and cannot construct one half of the pair without the other.
        """
        self.terminal = terminal
        self.status_code = getattr(terminal, "status_code", None)
        after = getattr(terminal, "retry_after", None)
        if after is not None:
            self.vendor_retry_after = after
        return self

    def retry_verdict(self) -> Any:
        """This error's :class:`~modelpass.retry.RetryVerdict`."""
        from .retry import classify_error

        verdict = classify_error(self, connection_retry=self.connection_retry)
        if verdict.retry_after is None and self.vendor_retry_after is not None:
            from .retry import RetryVerdict

            return RetryVerdict(
                verdict.retryable, self.vendor_retry_after, verdict.note
            )
        return verdict

    @property
    def retryable(self) -> Any:
        """:class:`~modelpass.types.Retryable`: is trying this again sensible?"""
        return self.retry_verdict().retryable

    @property
    def retry_after(self) -> float | None:
        """Seconds the vendor asked us to wait, or ``None`` if it said nothing."""
        return self.retry_verdict().retry_after


# --- configuration -------------------------------------------------------------


class ConfigError(SubpassError):
    """The connection store or a connection definition is unusable."""


class NoSuchConnection(ConfigError):
    """A connection was requested by name and no such connection is configured.

    This is what an ambient credential gets you: nothing. Environment variables
    never create connections (D2).
    """

    def __init__(self, name: str, known: tuple[str, ...] = ()) -> None:
        self.name = name
        self.known = known
        if known:
            detail = "configured connections: " + ", ".join(known)
        else:
            detail = "no connections are configured; run 'modelpass connect' first"
        super().__init__(f"no connection named {name!r} ({detail})")


class NoSuchGroup(ConfigError):
    """A group was named and no connection declares it.

    Groups are not defined anywhere; they exist because connections claim them
    (ticket 1.15). So "no such group" and "the group is empty" are the same
    finding, and there is one error for it rather than two that a caller would
    have to tell apart for no gain. ``default`` is the exception and it is why
    :data:`~modelpass.connections.DEFAULT_GROUP` is special: it exists whenever
    any connection declares no groups at all, so the only way to get this error
    for ``default`` is a store where every connection has been put in a named
    group -- which is a real thing to be told.
    """

    def __init__(self, name: str, known: tuple[str, ...] = ()) -> None:
        self.name = name
        self.known = known
        if known:
            detail = "configured groups: " + ", ".join(known)
        else:
            detail = "no connections are configured, so there are no groups"
        super().__init__(f"no group named {name!r} ({detail})")


class GroupUnavailable(ConfigError):
    """A group exists, has members, and every one of them is disabled.

    Deliberately not :class:`NoSuchGroup`: "you named a group nobody is in" and
    "every member of this group is switched off" send a user to two different
    places, and collapsing them would send half of them to the wrong one. It is
    also deliberately not :class:`ConnectionDisabled`, because no single
    connection was named -- naming one of them in the message would suggest the
    group had picked it, and it had not.
    """

    def __init__(self, name: str, members: tuple[str, ...] = ()) -> None:
        self.name = name
        self.members = members
        listed = ", ".join(members)
        super().__init__(
            f"every connection in group {name!r} is disabled ({listed}). Re-enable "
            "one of them, or name a different group"
        )


class DuplicateConnection(ConfigError):
    """A connection with that name already exists and ``overwrite`` was not set."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"connection {name!r} already exists")


class NoSuchSecret(ConfigError):
    """A ``secret:<entry>`` reference names an entry the secrets file does not have.

    Deliberately says nothing about any value. The message names the entry, the
    file, and the command that would write it -- never a length, never a prefix,
    never a fragment of anything stored.
    """

    def __init__(self, entry: str, path: str, known: tuple[str, ...] = ()) -> None:
        self.entry = entry
        self.path = path
        self.known = known
        if known:
            detail = "entries in that file: " + ", ".join(known)
        else:
            detail = "that file holds no entries yet"
        super().__init__(
            f"no secret entry named {entry!r} in {path} ({detail}). Write one with "
            "'modelpass connect ... --api-key-stdin'"
        )


class SecretStillReferenced(ConfigError):
    """A secret entry was asked to be deleted while connections still point at it.

    An orphaned secret is inert and recoverable; a deleted key is neither. So the
    refusal names the connections, the same shape a missing failover target gets.
    """

    def __init__(self, entry: str, connections: tuple[str, ...]) -> None:
        self.entry = entry
        self.connections = connections
        named = ", ".join(repr(name) for name in connections)
        super().__init__(
            f"secret entry {entry!r} is still referenced by {named}. Remove or "
            "repoint those connections first"
        )


class InvalidConnection(ConfigError):
    """A connection definition is internally inconsistent or malformed."""


class CredentialRefIsSecret(InvalidConnection):
    """A ``credentialRef`` looks like an actual secret rather than a pointer.

    modelpass never stores secrets. A ``credentialRef`` names *where* a credential
    lives (native login, an environment variable, a keychain entry); it is never
    the credential itself.
    """


class InvalidGuards(SubpassError, ValueError):
    """A guard value is not usable as a number of tokens.

    **Deliberately not a** :class:`ConfigError` (2026-08-17). Guards arrive from
    two places: a connection file, and a per-call argument. When the caller
    lowered ``stop_at_tokens`` for one call and got ``InvalidConnection`` back,
    the error implicated a store nobody had edited and pointed at a file that
    was fine -- so the message sent the reader to the wrong place entirely.
    Per-call guard problems are argument problems and say so.

    Also a ``ValueError``, so ``except ValueError`` catches it like any other bad
    argument and ``except SubpassError`` still catches the whole library. A
    contradiction found while *reading a config file* is still an
    :class:`InvalidConnection`: there, the file really is wrong.
    """


class InvalidSession(SubpassError, ValueError):
    """A session was asked for in a way that contradicts itself (D14-D17).

    ``persist=False`` next to a ``session_id``; a ``WorkerSession`` with no
    ``project_folder``; a ``send()`` carrying tools or a system prompt that were
    fixed at construction. All the same class of mistake: the arguments describe
    an object that cannot exist, and the caller finds out where they wrote them.

    **Deliberately not a** :class:`ConfigError`, for the reason
    :class:`InvalidGuards` is not one: nothing is wrong with the connection
    file, and an error that implicates a store nobody edited sends the reader to
    the wrong place. Also a ``ValueError``, so ``except ValueError`` catches it
    like any other bad argument.

    A runtime that simply *cannot* do the thing asked for is
    :class:`CapabilityNotSupported` instead -- that is a fact about the runtime,
    not a mistake in the call.
    """


class SessionNotFound(SubpassError):
    """A session was named for resume and the runtime does not have it.

    Expected, not exceptional. Both runtimes delete sessions out from under a
    caller: Claude Code's ``cleanupPeriodDays`` retention sweep removes old
    transcripts, and a Codex thread whose first turn never completed leaves an
    id that ``codex exec resume`` rejects with ``no rollout found for thread
    id``. So a stored id is a *hint*, and code that keeps one must be ready for
    this.

    The prefix->session lookup that will sit under ``chat()``
    treats it as a cache miss and starts a fresh session; it is an error here
    only because :meth:`Bridge.resume_chat` was asked for one specific session
    and cannot silently hand back a different one.
    """

    def __init__(self, connection: str, session_id: str, detail: str = "") -> None:
        self.connection = connection
        self.session_id = session_id
        suffix = f": {detail}" if detail else ""
        super().__init__(
            f"connection {connection!r} has no session {session_id!r}{suffix}"
        )


class SessionBusy(SubpassError):
    """Two callers reached one :class:`~modelpass.sessions.Session` at once.

    **A** :class:`~modelpass.sessions.Session` **takes one caller at a time**
    (R7, ticket 1.12). It holds a conversation, a token tracker for the whole
    conversation, and a vendor handle whose turns are ordered; two threads
    sending into it produce interleaved turns, a tracker that double-counts, and
    a history neither caller can read back.

    Raised rather than serialized on purpose. Waiting would look like it worked
    -- the second caller's turn simply lands after the first, in an order
    nobody chose, against a prefix that changed underneath them -- and a lock
    that queues is a lock that can deadlock when one of the two callers is the
    thread draining the other's iterator. A raise names the mistake at the call
    site that made it, which is the only place it can be fixed.

    The fix is a session per caller (:meth:`~modelpass.Bridge.new_chat` is
    cheap: opening a handle is local work by contract) or the caller's own lock
    around the turn. :class:`~modelpass.Bridge` itself is safe to share and has
    no such rule.
    """


class RunTimedOut(SubpassError):
    """The wall-clock bound this call was given ran out (R6, ticket 1.12).

    Raised by :meth:`~modelpass.Bridge.ask` and by ``chat(raise_on_stop=True)``;
    a plain streaming caller sees it as a
    :attr:`~modelpass.types.TerminalStatus.TIMED_OUT` terminal event instead,
    which is the same rule a guard stop and an exhausted allowance already
    follow.

    ``usage`` is what the run had spent when the bound expired -- a timed-out
    run is still a billed run, and an exception that threw the spend away would
    be the one failure mode the receipt exists to prevent.

    ``which`` is ``"total"`` or ``"first_token"``, naming which bound fired, and
    the distinction is not cosmetic: nothing answering at all before the
    first-token bound on a stateless run is the one timeout this library calls
    outright transient.
    """

    def __init__(
        self,
        message: str,
        *,
        connection: str = "",
        which: str = "total",
        seconds: float | None = None,
        usage: Any = None,
    ) -> None:
        self.connection = connection
        self.which = which
        self.seconds = seconds
        self.usage = usage
        super().__init__(message)


class SessionClosed(SubpassError):
    """The session has been closed, or a guard stopped it, and cannot run again.

    Separate from :class:`GuardStop` because the two answer different questions.
    ``GuardStop`` is *this run hit the ceiling*; this is *you asked a session
    that is already over to do more work*, which is a caller-state mistake and
    never spends anything.
    """


class ConnectionDisabled(ConfigError):
    """The connection exists and is valid, but has been taken out of service.

    A configuration error rather than a run outcome: nothing was attempted and
    nothing was spent. The connection is still listed and its preflight still
    runs -- ``enabled = false`` refuses runs, not inspection.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"connection {name!r} is disabled (enabled = false in the connection "
            "file). Re-enable it there, or name a different connection"
        )

class RuntimeGated(ConfigError):
    """The runtime is design-complete but gated behind an explicit opt-in (D5)."""


class InvalidSchema(ConfigError):
    """A caller-supplied structured-output schema is malformed (D13).

    Raised at the ``chat(..., schema=)`` boundary, before anything is launched.
    The alternative is a vendor rejecting it: on ``openai-sdk`` that arrives as
    a 400 about a ``response_format`` named ``codex_output_schema``, which the
    caller never wrote and cannot search their own code for.

    Deliberately *not* raised for a schema that is merely narrower than one
    runtime accepts -- that is what :func:`modelpass.schema.openai_strict_issues`
    reports, and modelpass does not refuse a run over it.
    """


class InvalidTool(ConfigError):
    """A caller-supplied tool definition is malformed (D12).

    A configuration error rather than a run failure: the tool is wrong where it
    was written, and failing at construction beats failing three layers down
    inside a vendor SDK partway through a spend.
    """


# --- auth and preflight --------------------------------------------------------


class AuthModeMismatch(SubpassError):
    """The auth mode about to be used is not the auth mode that was asked for.

    Raised when a caller asserts an expected auth mode that the connection does
    not have, when a runtime does not support the connection's auth mode, or when
    the preflight detects that the runtime would resolve credentials differently
    from the connection's declaration -- the silent-metered-billing case.
    """

    def __init__(self, expected: str, actual: str, connection: str | None = None) -> None:
        self.expected = expected
        self.actual = actual
        self.connection = connection
        where = f" for connection {connection!r}" if connection else ""
        super().__init__(f"expected auth mode {expected!r} but run would use {actual!r}{where}")


class PreflightFailed(SubpassError):
    """The auth preflight could not confirm the connection is safe to run."""


class UnsafeLaunch(SubpassError):
    """A runtime launch was assembled with an argument modelpass forbids.

    Example: ``--bare`` on the Claude Code runtime never reads OAuth credentials
    and forces an API key, which would move a subscription connection onto
    metered billing.
    """


# --- capabilities and runtimes -------------------------------------------------


class CapabilityNotSupported(SubpassError):
    """The runtime behind this connection does not support the requested capability.

    ``detail`` is the sentence that says what the caller asked for and what they
    get to do about it, and it exists because the bare form of this message --
    *runtime 'openai-sdk' capability 'ephemeral_multi_turn' is unsupported* --
    names a registry cell rather than the argument that produced it. A session
    refused at construction (D15) says which keyword it refused and why, so the
    reader does not have to go and read the capability table to find out what
    they typed wrong. Empty by default: every existing caller keeps the message
    it already had.
    """

    def __init__(
        self,
        runtime: str,
        capability: str,
        support: str = "unsupported",
        detail: str = "",
    ) -> None:
        self.runtime = runtime
        self.capability = capability
        self.support = support
        self.detail = detail
        message = f"runtime {runtime!r} capability {capability!r} is {support}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


class StructuredOutputRejected(CapabilityNotSupported):
    """The endpoint refused the ``response_format`` this call sent (ticket 1.10).

    **A capability refusal that could not be made before the call, and a subclass
    rather than a new taxonomy branch for exactly that reason.** Everywhere else
    in modelpass, "this runtime cannot do what you asked" is decided from the
    capability table before a byte is sent. On ``openai-compatible`` it cannot
    be: the row describes an API *shape*, what answers is whatever the
    connection's ``baseUrl`` points at, and a great many OpenAI-compatible
    servers implement chat completions without ``response_format:
    {"type": "json_schema"}``. So the refusal arrives from the endpoint, and
    modelpass reports it as what it is -- a capability that is absent here -- with
    the same class name a consumer is already matching on (R9).

    It is deliberately **not** a :class:`VendorRunFailed`: the run did not fail,
    the request was never valid against this endpoint. And the message says what
    to try, because the two honest ways forward (record the absence with
    ``modelpass verify`` and ask for JSON in the prompt, or point the connection
    at a server that implements the field) are not obvious from a raw HTTP 400.
    """


class RuntimeNotAvailable(SubpassError):
    """The vendor package backing a runtime is not installed.

    Core installs with zero vendor dependencies; runtimes arrive through extras.
    """

    def __init__(self, runtime: str, extra: str, package: str) -> None:
        self.runtime = runtime
        self.extra = extra
        self.package = package
        super().__init__(
            f"runtime {runtime!r} needs the {package!r} package; install it with "
            f"'pip install modelpass[{extra}]'"
        )


class AdapterNotImplemented(SubpassError, NotImplementedError):
    """An adapter exists as a contract but its implementation phase has not landed."""


class AdapterFailed(SubpassError):
    """An adapter raised an unexpected error while running -- i.e. a bug.

    Narrowed 2026-08-17. This used to be where *every* mid-stream exception
    ended up, including the vendor runtime reporting that the run failed, which
    broke the "exactly one terminal event" contract for the most ordinary
    failure there is. Vendor-reported failures are now :class:`VendorRunFailed`
    and end the stream with a terminal event instead. What is left here is what
    the name always said: something inside modelpass or an adapter went wrong in a
    way nobody accounted for.

    Seeing this is a bug report, not a runtime condition to handle.
    """


class VendorRunFailed(SubpassError):
    """The vendor runtime reported that this run failed.

    **This is how an adapter says "the run failed", and it is not an exception
    a caller sees.** The bridge catches it and ends the stream with one terminal
    event (``status=error``), stamped and carrying the usage accumulated up to
    the failure -- because a run that spent tokens and then hit a vendor 400 is
    a run outcome, not a caller mistake, and the caller needs both the stamp and
    the spend (deviation 2 amendment, 2026-08-17).

    The boundary it draws, for adapter authors: raise this for anything the
    *vendor* did or failed to do -- an API error, a rejected model, a runtime
    that would not launch, a transport that died. Raise nothing else. An
    exception of any other type escaping :meth:`Adapter.run` is a bug and is
    reported as :class:`AdapterFailed`, with two deliberate exceptions that
    predate this rule and outrank it: :class:`AuthModeMismatch` and
    :class:`UnsafeLaunch` are guaranteed-layer refusals (D4 layer 1), and a
    guarantee that could be downgraded to a status line is not a guarantee.
    """


# --- run outcomes --------------------------------------------------------------


class GuardStop(SubpassError):
    """A configured guard stopped the run.

    The streaming interface reports a guard stop as ``guard_stop`` + ``terminal``
    events rather than an exception; this error exists for callers that opt into
    raising (``Bridge.chat(..., raise_on_stop=True)``) and for non-streaming
    helpers built on top later.
    """

    def __init__(self, message: str, *, observed: int = 0, threshold: int = 0) -> None:
        self.observed = observed
        self.threshold = threshold
        super().__init__(message)


class QuotaExhausted(SubpassError):
    """The subscription allowance behind this connection is exhausted.

    Default behavior is a clean stop, never an automatic move to metered billing
    (D4). Failover exists only as explicit per-connection configuration.
    """

    def __init__(self, connection: str, detail: str = "") -> None:
        self.connection = connection
        suffix = f": {detail}" if detail else ""
        super().__init__(f"quota exhausted for connection {connection!r}{suffix}")
