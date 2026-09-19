"""Typed retryability verdicts (R6, ticket 1.12).

The test this file exists for is
``test_a_token_count_containing_500_is_never_read_as_a_server_error``. Every
other test here defends the rule it states.

Background: a consumer classified retryability by substring-matching status
codes out of the error *message*, and ``"500"`` matched inside ``"stopAtTokens
threshold 500000 reached"`` -- so a deterministic guard stop was retried as a
transient HTTP 500, four times, against a live subscription allowance
(reported by a downstream agent host, 2026-08-18). The library now owns the
classification, and it reads typed fields only.
"""

from __future__ import annotations

import threading

import pytest

from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.errors import (
    AdapterFailed,
    AuthModeMismatch,
    CapabilityNotSupported,
    GuardStop,
    QuotaExhausted,
    RunTimedOut,
    SessionBusy,
    SubpassError,
    VendorRunFailed,
)
from modelpass.retry import (
    AGENT,
    API,
    NEVER,
    RetryVerdict,
    classify_error,
    classify_terminal,
    family_of,
    vendor_error_facts,
)
from modelpass.runtimes import Runtime
from modelpass.testing import (
    FakeSessionAdapter,
    FakeSessionHandle,
    fake_bridge,
    usage,
)
from modelpass.types import (
    AuthMode,
    Retryable,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    Timeout,
)


@pytest.fixture
def connection() -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        guards=Guards(stop_at_tokens=500_000),
    )


# --- the incident ----------------------------------------------------------------


def test_a_token_count_containing_500_is_never_read_as_a_server_error():
    """The regression test for the allowance-burning defect.

    The message says "500000". The typed facts say "a guard stopped this run".
    The verdict follows the typed facts, and there is no arrangement of this
    message that can make it say otherwise.
    """
    message = "stopAtTokens threshold 500000 reached"

    stop = GuardStop(message, observed=500_001, threshold=500_000)
    assert stop.retryable is Retryable.NO

    terminal = classify_terminal(
        Runtime.ANTHROPIC_SDK, TerminalStatus.GUARD_STOP, status_code=None
    )
    assert terminal.retryable is Retryable.NO

    # And the same message on a status nobody typed is UNKNOWN, not YES: an
    # unclassified failure is never told a retry is safe on the strength of
    # three digits sitting inside a token count.
    unclassified = classify_terminal(
        Runtime.ANTHROPIC_SDK, TerminalStatus.ERROR, status_code=None
    )
    assert unclassified.retryable is Retryable.UNKNOWN


def test_the_same_three_digits_as_an_actual_status_code_are_retryable():
    """The other half: a real 500 is still transient. The difference is *where*
    the number came from, not what it looks like."""
    verdict = classify_terminal(
        Runtime.ANTHROPIC_API, TerminalStatus.ERROR, status_code=500
    )
    assert verdict.retryable is Retryable.YES


# --- the table --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (TerminalStatus.OK, Retryable.NO),
        (TerminalStatus.GUARD_STOP, Retryable.NO),
        (TerminalStatus.QUOTA_EXHAUSTED, Retryable.NO),
        (TerminalStatus.CANCELLED, Retryable.NO),
        (TerminalStatus.ERROR, Retryable.UNKNOWN),
    ],
)
def test_terminal_statuses_on_an_agent_runtime(status, expected):
    assert classify_terminal(Runtime.ANTHROPIC_SDK, status).retryable is expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (400, Retryable.NO),
        (401, Retryable.NO),
        (403, Retryable.NO),
        (404, Retryable.NO),
        (413, Retryable.NO),
        (422, Retryable.NO),
        (418, Retryable.NO),
        (408, Retryable.YES),
        (500, Retryable.YES),
        (502, Retryable.YES),
        (503, Retryable.YES),
        (504, Retryable.YES),
        (599, Retryable.YES),
    ],
)
def test_http_status_rows(code, expected):
    verdict = classify_terminal(
        Runtime.OPENAI_API, TerminalStatus.ERROR, status_code=code
    )
    assert verdict.retryable is expected
    assert verdict.note


def test_a_429_is_transient_only_when_the_vendor_named_a_wait():
    """A rate limit that named a wait is transient. One that named nothing may
    be a spent allowance wearing a rate limit's number, and guessing costs the
    allowance."""
    with_wait = classify_terminal(
        Runtime.OPENAI_API, TerminalStatus.ERROR, status_code=429, retry_after=12.0
    )
    assert with_wait.retryable is Retryable.YES
    assert with_wait.retry_after == 12.0

    without = classify_terminal(
        Runtime.OPENAI_API, TerminalStatus.ERROR, status_code=429
    )
    assert without.retryable is Retryable.UNKNOWN


def test_a_status_code_on_an_agent_runtime_is_not_consulted():
    """Agent runtimes report no HTTP status, so nothing pretends they do."""
    assert family_of(Runtime.ANTHROPIC_SDK) == AGENT
    assert family_of(Runtime.OPENAI_COMPATIBLE) == API
    verdict = classify_terminal(
        Runtime.OPENAI_SDK, TerminalStatus.ERROR, status_code=503
    )
    assert verdict.retryable is Retryable.UNKNOWN


# --- exceptions --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (GuardStop("stopped"), Retryable.NO),
        (QuotaExhausted("claude-sub"), Retryable.NO),
        (SessionBusy("busy"), Retryable.NO),
        (AuthModeMismatch("subscription", "api_key"), Retryable.NO),
        (CapabilityNotSupported("openai-sdk", "tools"), Retryable.NO),
        (RunTimedOut("out of time"), Retryable.UNKNOWN),
        (RunTimedOut("out of time", which="first_token"), Retryable.YES),
    ],
)
def test_every_error_carries_a_verdict(error, expected):
    assert error.retryable is expected
    assert classify_error(error).retryable is expected


def test_an_error_nobody_classified_is_unknown_rather_than_a_guess():
    class Weird(SubpassError):
        pass

    assert Weird("what").retryable is Retryable.UNKNOWN


def test_a_status_code_outranks_the_class_name():
    """The precedence a consumer's own fix established: a structured field is the
    vendor's own statement, a name is our reading of it."""

    class BadRequestError(Exception):
        status_code = 503

    assert classify_error(BadRequestError("boom")).retryable is Retryable.YES


def test_retry_after_is_read_from_typed_places_only():
    class Headers(dict):
        pass

    class Response:
        headers = Headers({"retry-after": "30"})

    class Limited(Exception):
        status_code = 429
        response = Response()

    code, after = vendor_error_facts(Limited("slow down"))
    assert code == 429
    assert after == 30.0

    # An HTTP-date Retry-After is an absence, not a guess: a date is not a delta.
    Response.headers = Headers({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert vendor_error_facts(Limited("slow down"))[1] is None

    # And a message that says "retry after 30 seconds" and nothing typed gives
    # nothing, because nothing here reads a message.
    assert vendor_error_facts(Exception("retry after 30 seconds")) == (None, None)


# --- the connection-level stance ---------------------------------------------------


def test_retry_never_forces_no_and_says_so():
    verdict = classify_terminal(
        Runtime.OPENAI_API,
        TerminalStatus.ERROR,
        status_code=503,
        connection_retry=NEVER,
    )
    assert verdict.retryable is Retryable.NO
    # The note distinguishes a policy refusal from a classified one, so a reader
    # can tell why they are being told no.
    assert 'retry = "never"' in verdict.note
    assert "would otherwise be yes" in verdict.note


def test_retry_never_is_refused_when_misspelled():
    from modelpass.errors import InvalidConnection

    with pytest.raises(InvalidConnection, match="retry must be one of"):
        Connection(
            name="x",
            runtime=Runtime.ANTHROPIC_API,
            auth_mode=AuthMode.API_KEY,
            credential_ref=CredentialRef.parse("env:K"),
            retry="nver",
        )


def test_the_receipt_reports_the_stance(tmp_path):
    never = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        retry="never",
    )
    bridge, _, _ = fake_bridge(connections=[never], home=tmp_path / "modelpass")
    receipt = bridge.preflight("claude-sub")
    assert receipt.retry == "never"
    assert 'retry = "never"' in receipt.summary()
    assert "never retries on its own" in (receipt.retry_note or "")
    assert receipt.to_dict()["retry"] == "never"


def test_a_default_connection_says_nothing_new(tmp_path, connection):
    bridge, _, _ = fake_bridge(
        connections=[connection], home=tmp_path / "modelpass"
    )
    receipt = bridge.preflight("claude-sub")
    assert receipt.retry == "default"
    assert receipt.retry_note is None
    assert "retry" not in receipt.summary()


def test_a_run_on_a_never_connection_stamps_no_on_the_terminal(tmp_path):
    never = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        retry="never",
    )
    bridge, _, _ = fake_bridge(
        connections=[never],
        script=[
            TextDeltaEvent(text="partial"),
            usage(input_tokens=5),
            TerminalEvent(
                status=TerminalStatus.ERROR,
                connection="",
                runtime=Runtime.ANTHROPIC_SDK,
                auth_mode=AuthMode.SUBSCRIPTION,
                reason="the transport died",
            ),
        ],
        home=tmp_path / "modelpass",
    )
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    assert events[-1].retryable is Retryable.NO


def test_the_store_round_trips_both_new_keys(tmp_path):
    from modelpass.store import ConnectionStore

    store = ConnectionStore(tmp_path / "modelpass")
    store.add(
        Connection(
            name="bounded",
            runtime=Runtime.ANTHROPIC_API,
            auth_mode=AuthMode.API_KEY,
            credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
            retry="never",
            timeout_seconds=30,
        )
    )
    reread = ConnectionStore(tmp_path / "modelpass").get("bounded")
    assert reread.retry == "never"
    assert reread.timeout_seconds == 30.0


def test_a_default_connection_writes_neither_key(tmp_path):
    from modelpass.store import connection_to_dict

    plain = Connection(
        name="plain",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
    )
    data = connection_to_dict(plain)
    assert "retry" not in data
    assert "timeoutSeconds" not in data


# --- the shape of a verdict ---------------------------------------------------------


def test_every_verdict_carries_a_note_a_human_can_read():
    verdict = classify_terminal(Runtime.ANTHROPIC_API, TerminalStatus.GUARD_STOP)
    assert isinstance(verdict, RetryVerdict)
    assert verdict.note
    assert verdict.to_dict() == {
        "retryable": "no",
        "retry_after": None,
        "note": verdict.note,
    }


# --- the verdict survives the raise (ticket 1.12b) ---------------------------------
#
# A RAG evaluation harness's migration found the hole, 2026-09-13: an adapter
# meeting an HTTP 429
# ends the stream with a terminal carrying ``retryable=yes``, ``status_code=429``
# and the vendor's ``retry_after``, and ``Bridge.ask`` raised
# ``VendorRunFailed(reason)`` carrying none of the three -- so the consumer's
# ``classify_error`` answered UNKNOWN and a rate limit was never retried
# in that harness's own retry classifier. Every exception raised in a
# terminal's place now carries that terminal.


RATE_LIMITED = TerminalEvent(
    status=TerminalStatus.ERROR,
    connection="",
    runtime=Runtime.ANTHROPIC_API,
    auth_mode=AuthMode.API_KEY,
    reason="429: rate limited",
    status_code=429,
    retry_after=7.0,
)


@pytest.fixture
def metered() -> Connection:
    return Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
    )


def metered_bridge(tmp_path, metered, script, **kwargs):
    bridge, _store, _adapter = fake_bridge(
        connections=[metered],
        script=script,
        home=tmp_path / "modelpass",
        runtime=Runtime.ANTHROPIC_API,
        env={"ANTHROPIC_API_KEY": "sk-not-a-real-key"},
        **kwargs,
    )
    return bridge


def assert_rate_limited(error):
    """The three typed facts, plus the event they came from."""
    assert error.retryable is Retryable.YES
    assert error.status_code == 429
    assert error.retry_after == 7.0
    assert error.terminal is not None
    assert error.terminal.status is TerminalStatus.ERROR
    # And the consumer's own call answers the same thing, which is the whole
    # point: they hold the exception, not the event.
    verdict = classify_error(error)
    assert verdict.retryable is Retryable.YES
    assert verdict.retry_after == 7.0


def test_vendor_run_failed_from_ask_carries_the_429(tmp_path, metered):
    bridge = metered_bridge(
        tmp_path, metered, [TextDeltaEvent(text="half an ans"), RATE_LIMITED]
    )
    with pytest.raises(VendorRunFailed) as caught:
        bridge.ask("claude-api", "go")
    assert_rate_limited(caught.value)


def test_vendor_run_failed_from_aask_carries_the_same(tmp_path, metered):
    import asyncio

    bridge = metered_bridge(tmp_path, metered, [RATE_LIMITED])
    with pytest.raises(VendorRunFailed) as caught:
        asyncio.run(bridge.aask("claude-api", "go"))
    assert_rate_limited(caught.value)


def test_a_guard_stop_from_ask_carries_the_terminal_and_reads_no(tmp_path, metered):
    bridge = metered_bridge(
        tmp_path,
        metered.with_guards(Guards(stop_at_tokens=5)),
        [usage(input_tokens=10)],
    )
    with pytest.raises(GuardStop) as caught:
        bridge.ask("claude-api", "go")
    error = caught.value
    assert error.retryable is Retryable.NO
    assert error.terminal is not None
    assert error.terminal.status is TerminalStatus.GUARD_STOP
    # The numbers 1.8 put on it are untouched.
    assert error.threshold == 5
    assert classify_error(error).retryable is Retryable.NO


def test_quota_exhaustion_from_ask_carries_the_terminal(tmp_path, metered):
    from modelpass.testing import quota_exhausted

    bridge = metered_bridge(
        tmp_path, metered, [quota_exhausted(runtime=Runtime.ANTHROPIC_API)]
    )
    with pytest.raises(QuotaExhausted) as caught:
        bridge.ask("claude-api", "go")
    assert caught.value.retryable is Retryable.NO
    assert caught.value.terminal.status is TerminalStatus.QUOTA_EXHAUSTED
    assert classify_error(caught.value).retryable is Retryable.NO


def test_a_timed_out_ask_carries_the_terminal_s_verdict(tmp_path, metered):
    """Whatever 1.12 decided, and not a second opinion derived from the class.

    A ``total`` expiry is UNKNOWN -- the run may have been most of the way
    through an answer -- and a ``first_token`` expiry on a stateless run is the
    one timeout the table calls transient outright.
    """
    from modelpass.testing import StallingAdapter
    from modelpass.types import Timeout

    bridge = metered_bridge(
        tmp_path, metered, (), adapter=StallingAdapter(runtime=Runtime.ANTHROPIC_API)
    )
    with pytest.raises(RunTimedOut) as caught:
        bridge.ask("claude-api", "go", timeout=Timeout(first_token=0.05))
    error = caught.value
    assert error.terminal is not None
    assert error.terminal.status is TerminalStatus.TIMED_OUT
    assert error.retryable is error.terminal.retryable is Retryable.YES
    assert classify_error(error).retryable is Retryable.YES

    bridge = metered_bridge(
        tmp_path,
        metered,
        (),
        adapter=StallingAdapter(
            before=[TextDeltaEvent(text="thinking")], runtime=Runtime.ANTHROPIC_API
        ),
    )
    with pytest.raises(RunTimedOut) as caught:
        bridge.ask("claude-api", "go", timeout=0.1)
    assert caught.value.retryable is Retryable.UNKNOWN
    assert caught.value.terminal.status is TerminalStatus.TIMED_OUT


class _StallingSessionHandle(FakeSessionHandle):
    """A turn that answers nothing until the watchdog cancels it.

    The session-side shape of ``StallingAdapter``: blocked inside ``next()``, so
    only ``cancel()`` from the watchdog thread gets the consumer moving again.
    """

    def send(self, message):
        self.messages.append(message)
        self.adapter.released.wait(30.0)
        # Released by the watchdog's cancel, and what a cancelled runtime
        # reports is a cancel: the bridge is the one that knows it asked.
        yield TerminalEvent(
            status=TerminalStatus.CANCELLED,
            connection="",
            runtime=self.adapter.runtime,
            auth_mode=AuthMode.SUBSCRIPTION,
            reason="cancelled by caller",
        )


class _StallingSessionAdapter(FakeSessionAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.released = threading.Event()

    def open_session(self, request):
        handle = _StallingSessionHandle(self, request)
        self.session_requests.append(request)
        self.handles.append(handle)
        return handle

    def cancel(self) -> None:
        super().cancel()
        self.released.set()


def test_a_first_token_timeout_on_a_session_turn_reads_unknown(tmp_path, connection):
    """The verdict that changed in 0.2.1, pinned so it is a decision.

    Before the terminal rode on the exception, ``classify_error`` matched the
    class name and the ``which`` field and answered YES for every
    ``first_token`` expiry, session or not. The terminal it now carries knows
    something the exception's class cannot: this turn left a conversation
    behind on the runtime, so a repeat is not a repeat of nothing, and UNKNOWN
    is the honest answer. The stateless ``ask()`` above still reads YES.
    """
    adapter = _StallingSessionAdapter((), runtime=Runtime.ANTHROPIC_SDK)
    bridge, _store, _adapter = fake_bridge(
        connections=[connection],
        adapter=adapter,
        home=tmp_path / "modelpass",
    )
    session = bridge.new_chat(connection="claude-sub", raise_on_stop=True)

    with pytest.raises(RunTimedOut) as caught:
        list(session.send("go", timeout=Timeout(first_token=0.05)))

    error = caught.value
    assert error.which == "first_token"
    assert error.terminal is not None
    assert error.terminal.status is TerminalStatus.TIMED_OUT
    assert error.terminal.retryable is Retryable.UNKNOWN
    assert classify_error(error).retryable is Retryable.UNKNOWN


def test_the_structured_output_contract_violation_carries_its_terminal(
    tmp_path, metered
):
    """An ok terminal that brought no structured answer still raises with it.

    The verdict is the ok row's NO rather than the UNKNOWN ``AdapterFailed``
    used to read: the run finished, and repeating it repeats a contract
    violation rather than a transient failure.
    """
    from modelpass.testing import FakeAdapter

    class Terminal(FakeAdapter):
        def _with_structured(self, request, events):  # type: ignore[override]
            return list(events)

    bridge = metered_bridge(
        tmp_path,
        metered,
        (),
        adapter=Terminal(
            [TextDeltaEvent(text="{}")], runtime=Runtime.ANTHROPIC_API
        ),
    )
    with pytest.raises(AdapterFailed) as caught:
        bridge.ask(
            "claude-api",
            "score this",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
        )
    assert caught.value.terminal is not None
    assert caught.value.terminal.status is TerminalStatus.OK
    assert caught.value.retryable is Retryable.NO


def test_raise_on_stop_raises_the_same_carrying_errors(tmp_path, metered):
    """``chat(raise_on_stop=True)`` and ``ask`` share one raiser, so they agree."""
    bridge = metered_bridge(
        tmp_path,
        metered.with_guards(Guards(stop_at_tokens=5)),
        [usage(input_tokens=10)],
    )
    with pytest.raises(GuardStop) as caught:
        list(
            bridge.chat(
                connection="claude-api", message="go", raise_on_stop=True
            )
        )
    assert caught.value.terminal.status is TerminalStatus.GUARD_STOP
    assert caught.value.retryable is Retryable.NO


def test_a_never_connection_still_outranks_a_carried_verdict(tmp_path):
    """The policy is applied last, here as everywhere else."""
    never = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
        retry="never",
    )
    bridge = metered_bridge(tmp_path, never, [RATE_LIMITED])
    with pytest.raises(VendorRunFailed) as caught:
        bridge.ask("claude-api", "go")
    assert caught.value.retryable is Retryable.NO
    assert classify_error(caught.value, connection_retry=NEVER).retryable is (
        Retryable.NO
    )
