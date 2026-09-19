"""The session core (D14-D17), exercised without a vendor package anywhere.

The session-capable fake this file runs on -- :class:`~modelpass.testing.FakeSessionAdapter`
and its handle -- now ships in :mod:`modelpass.testing` rather than living here.
It was held back while the real adapters were unlanded, on the grounds that
anything shipped then would be a contract about to be discovered wrong; the
adapters have since landed and a consumer hit the cost of the arrangement,
having to spend subscription allowance to cover a seam that could not be driven
offline. So the fake moved, and this file is now one of its consumers rather
than its owner.

What the fake honours, and what these tests lean on throughout, is the adapter
contract's rule 7 -- opening a session is local work, the id appears only when a
turn completes -- because that timing is the thing most easily got wrong and
most expensive to get wrong.
"""

from __future__ import annotations

import json
import os

import pytest

from modelpass.adapters.base import RunRequest
from modelpass.bridge import Bridge
from modelpass.capabilities import Capability, CapabilityRegistry, Support
from modelpass.connections import Connection, CredentialRef, Guards, QuotaAction, QuotaPolicy
from modelpass.errors import (
    AdapterNotImplemented,
    AuthModeMismatch,
    CapabilityNotSupported,
    ConnectionDisabled,
    GuardStop,
    InvalidGuards,
    InvalidSession,
    SessionClosed,
    SessionNotFound,
)
from modelpass.runtimes import Runtime
from modelpass.sessions import ChatSession, WorkerSession
from modelpass.store import ConnectionStore
from modelpass.testing import FakeAdapter, FakeSessionAdapter, quota_exhausted, usage
from modelpass.tools import ToolDef
from modelpass.types import (
    AuthMode,
    ReceiptEvent,
    Role,
    SessionInfo,
    SessionKind,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    UsageEvent,
    VendorEvent,
)

# --- fixtures ------------------------------------------------------------------


def anthropic_connection(**overrides) -> Connection:
    base = {
        "name": "claude-sub",
        "runtime": Runtime.ANTHROPIC_SDK,
        "auth_mode": AuthMode.SUBSCRIPTION,
        "credential_ref": CredentialRef.native_login(),
    }
    return Connection(**{**base, **overrides})


class NarrowingSessionAdapter(FakeSessionAdapter):
    """A fake whose capabilities depend on the request, the way Codex's do.

    Added 2026-08-31 with S7. Until then ``openai-sdk`` was this file's stock
    example of a runtime that says no to ``tools_in_process``,
    ``system_prompt_replace`` and ``sessions_list``, and the registry alone was
    enough to drive those gates. Flipping the Codex default to ``codex
    app-server`` made all three true of the runtime, so the *example* had to
    move -- but the *gate* is exactly as load-bearing as it was, and it is now
    load-bearing in the narrowing direction: a request that opts out into
    ``codex exec`` must still be refused, by the adapter rather than the table.

    This mirrors :meth:`~modelpass.adapters.openai.OpenAIAdapter.support_for`
    without importing it, so the session core keeps being tested against the
    contract rather than against one adapter's implementation of it.
    """

    def support_for(self, capability, options):
        if options.get("transport") != "exec":
            return None
        return {
            Capability.TOOLS_IN_PROCESS: Support.UNSUPPORTED,
            Capability.SYSTEM_PROMPT_REPLACE: Support.UNSUPPORTED,
            Capability.SESSIONS_LIST: Support.UNSUPPORTED,
        }.get(capability)


#: What a request carries to opt out of the app-server default (S7).
EXEC = {"transport": "exec"}


def codex_connection(**overrides) -> Connection:
    base = {
        "name": "codex-sub",
        "runtime": Runtime.OPENAI_SDK,
        "auth_mode": AuthMode.SUBSCRIPTION,
        "credential_ref": CredentialRef.native_login(),
    }
    return Connection(**{**base, **overrides})


@pytest.fixture
def sessions(tmp_path):
    """A bridge whose connections are both session-capable fakes."""

    def make(*connections: Connection, script=None, adapter=None):
        store = ConnectionStore(tmp_path / "modelpass-home")
        adapters = {}
        made: dict[str, FakeSessionAdapter] = {}
        for connection in connections:
            store.add(connection, overwrite=True)
            if connection.runtime not in adapters:
                fake = adapter or FakeSessionAdapter(
                    script if script is not None else [TextDeltaEvent(text="hi")],
                    runtime=connection.runtime,
                )
                adapters[connection.runtime] = fake
            made[connection.name] = adapters[connection.runtime]
        bridge = Bridge(
            store=store,
            registry=CapabilityRegistry(),
            adapters=adapters,
            env={},
        )
        return bridge, made

    return make


def text(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDeltaEvent))


# --- lifecycle -----------------------------------------------------------------


def test_construction_runs_preflight_and_creates_nothing(sessions):
    bridge, made = sessions(anthropic_connection())
    session = bridge.new_chat(connection="claude-sub")

    assert isinstance(session, ChatSession)
    assert session.id is None
    assert session.receipt.connection == "claude-sub"
    assert session.receipt.effective_auth_mode is AuthMode.SUBSCRIPTION
    # A preflight ran; no turn did.
    assert made["claude-sub"].preflights
    assert made["claude-sub"].handles[0].messages == []


def test_the_id_appears_only_when_the_first_send_completes(sessions):
    bridge, _ = sessions(anthropic_connection())
    session = bridge.new_chat(connection="claude-sub")

    assert session.id is None
    stream = session.send("hello")
    assert session.id is None, "an unconsumed turn has not created anything"
    list(stream)
    first = session.id
    assert first == "sess-1"

    list(session.send("again"))
    assert session.id == first, "the id belongs to the session, not the turn"


def test_an_abandoned_first_turn_leaves_no_id(sessions):
    """A run killed mid-first-turn leaves an id a resume would reject (D15)."""
    bridge, _ = sessions(
        anthropic_connection(), script=[TextDeltaEvent(text="a"), TextDeltaEvent(text="b")]
    )
    session = bridge.new_chat(connection="claude-sub")

    stream = session.send("hello")
    next(stream)  # the receipt event
    next(stream)  # one delta, then walk away
    stream.close()

    assert session.id is None


def test_an_ephemeral_session_never_reports_an_id(sessions):
    bridge, _ = sessions(anthropic_connection())
    session = bridge.new_chat(connection="claude-sub", persist=False)

    list(session.send("one"))
    list(session.send("two"))

    assert session.id is None
    assert session.persists is False


def test_every_turn_is_a_stream_with_a_receipt_and_one_stamped_terminal(sessions):
    bridge, _ = sessions(
        anthropic_connection(),
        script=[TextDeltaEvent(text="hello "), TextDeltaEvent(text="world")],
    )
    session = bridge.new_chat(connection="claude-sub")

    for _ in range(2):
        events = list(session.send("hi"))
        assert isinstance(events[0], ReceiptEvent)
        assert text(events) == "hello world"
        assert sum(isinstance(e, TerminalEvent) for e in events) == 1
        terminal = events[-1]
        assert terminal.status is TerminalStatus.OK
        assert terminal.connection == "claude-sub"
        assert terminal.runtime is Runtime.ANTHROPIC_SDK
        assert terminal.auth_mode is AuthMode.SUBSCRIPTION


def test_the_terminal_carries_the_turn_and_the_session_carries_the_total(sessions):
    bridge, _ = sessions(anthropic_connection(), script=[usage(input_tokens=10, output_tokens=5)])
    session = bridge.new_chat(connection="claude-sub")

    first = list(session.send("one"))[-1]
    second = list(session.send("two"))[-1]

    assert first.usage.total_tokens == 15
    assert second.usage.total_tokens == 15, "a terminal reports its own turn"
    assert session.usage.total_tokens == 30, "the session reports everything"


def test_usage_events_carry_the_session_running_total(sessions):
    bridge, _ = sessions(anthropic_connection(), script=[usage(input_tokens=10)])
    session = bridge.new_chat(connection="claude-sub")

    list(session.send("one"))
    reported = [e for e in session.send("two") if isinstance(e, UsageEvent)]

    assert reported[0].cumulative.total_tokens == 20


def test_the_send_message_reaches_the_handle(sessions):
    bridge, made = sessions(anthropic_connection())
    session = bridge.new_chat(connection="claude-sub")
    list(session.send("the actual prompt"))
    assert made["claude-sub"].handles[0].messages == ["the actual prompt"]


def test_send_refuses_an_empty_or_non_string_turn(sessions):
    bridge, _ = sessions(anthropic_connection())
    session = bridge.new_chat(connection="claude-sub")
    with pytest.raises(ValueError):
        session.send("   ")
    with pytest.raises(TypeError):
        session.send(["not", "a", "string"])  # type: ignore[arg-type]


def test_get_history_comes_from_the_runtime(sessions):
    bridge, _ = sessions(anthropic_connection())
    session = bridge.new_chat(connection="claude-sub")
    list(session.send("hello"))

    history = session.get_history()
    assert [m.role for m in history] == [Role.USER, Role.ASSISTANT]
    assert history[0].content == "hello"


def test_close_is_idempotent_and_ends_the_session(sessions):
    bridge, made = sessions(anthropic_connection())
    session = bridge.new_chat(connection="claude-sub")
    list(session.send("hi"))

    session.close()
    session.close()

    assert made["claude-sub"].handles[0].closes == 1
    assert session.closed is True
    with pytest.raises(SessionClosed):
        session.send("more")
    with pytest.raises(SessionClosed):
        session.get_history()


def test_the_context_manager_closes(sessions):
    bridge, made = sessions(anthropic_connection())
    with bridge.new_chat(connection="claude-sub") as session:
        list(session.send("hi"))
    assert session.closed is True
    assert made["claude-sub"].handles[0].closes == 1


def test_a_persisted_session_keeps_its_scratch_folder_and_an_ephemeral_one_does_not(
    sessions,
):
    """The folder is the storage key on anthropic-sdk; deleting it orphans a session."""
    bridge, _ = sessions(anthropic_connection())

    kept = bridge.new_chat(connection="claude-sub")
    folder = kept.project_folder
    kept.close()
    assert os.path.isdir(folder)

    gone = bridge.new_chat(connection="claude-sub", persist=False)
    scratch = gone.project_folder
    gone.close()
    assert not os.path.isdir(scratch)


def test_a_named_project_folder_is_never_removed(sessions, tmp_path):
    bridge, _ = sessions(anthropic_connection())
    work = tmp_path / "work"
    work.mkdir()
    session = bridge.new_chat(
        connection="claude-sub", persist=False, project_folder=str(work)
    )
    session.close()
    assert work.is_dir()


def test_each_turn_writes_one_line_to_the_run_log(sessions, tmp_path):
    bridge, _ = sessions(anthropic_connection(), script=[usage(input_tokens=3)])
    session = bridge.new_chat(connection="claude-sub")
    list(session.send("one"))
    list(session.send("two"))

    records = bridge.run_log.read()
    assert len(records) == 2
    assert {r["connection"] for r in records} == {"claude-sub"}
    assert [r["total_tokens"] for r in records] == [3, 3], "one line per turn, not a total"


def test_each_turn_records_where_the_vendor_said_the_plan_stood(sessions):
    """A session is where this matters most (D20).

    The same allowance is drawn on turn after turn, so a log that kept the
    utilization is what lets a caller watch it climb rather than discover it at
    the stop. Per turn, because that is how it was reported.
    """
    bridge, _ = sessions(
        anthropic_connection(),
        script=[
            VendorEvent(
                Runtime.ANTHROPIC_SDK,
                "rate_limit",
                {"status": "allowed_warning", "rate_limit_type": "seven_day",
                 "utilization": 0.81, "resets_at": None},
            ),
            usage(input_tokens=3),
        ],
    )
    session = bridge.new_chat(connection="claude-sub")
    list(session.send("one"))

    record = bridge.run_log.read()[0]
    assert record["allowance_status"] == "allowed_warning"
    assert record["allowance_utilization"] == pytest.approx(0.81)
    # Nothing was said about a reset, so nothing is claimed about one.
    assert record["allowance_resets_at"] is None


def test_a_turn_the_runtime_said_nothing_about_records_absence(sessions):
    bridge, _ = sessions(anthropic_connection(), script=[usage(input_tokens=3)])
    session = bridge.new_chat(connection="claude-sub")
    list(session.send("one"))

    record = bridge.run_log.read()[0]
    assert record["allowance_status"] is None
    assert record["allowance_utilization"] is None


# --- guards, failure, quota -----------------------------------------------------


def test_the_spend_ceiling_is_the_sessions_not_the_turns(sessions):
    """D18: stop_at_tokens is a budget envelope per surface, not per call."""
    bridge, _ = sessions(anthropic_connection(), script=[usage(input_tokens=30)])
    session = bridge.new_chat(connection="claude-sub", stop_at_tokens=50)

    first = list(session.send("one"))[-1]
    assert first.status is TerminalStatus.OK, "one turn alone is under the ceiling"

    second = list(session.send("two"))[-1]
    assert second.status is TerminalStatus.GUARD_STOP
    assert "session" in (second.reason or "")


def test_a_stopped_session_refuses_further_turns_with_the_numbers(sessions):
    bridge, _ = sessions(anthropic_connection(), script=[usage(input_tokens=100)])
    session = bridge.new_chat(connection="claude-sub", stop_at_tokens=50)
    list(session.send("one"))

    with pytest.raises(GuardStop) as caught:
        session.send("two")
    assert caught.value.observed == 100
    assert caught.value.threshold == 50


def test_raise_on_stop_raises_after_the_terminal(sessions):
    bridge, _ = sessions(anthropic_connection(), script=[usage(input_tokens=100)])
    session = bridge.new_chat(
        connection="claude-sub", stop_at_tokens=50, raise_on_stop=True
    )
    with pytest.raises(GuardStop):
        list(session.send("one"))


def test_a_vendor_failure_ends_the_turn_and_does_not_raise(sessions):
    from modelpass.errors import VendorRunFailed

    adapter = FakeSessionAdapter([TextDeltaEvent(text="partial")])
    adapter.raise_at_end = VendorRunFailed("400: the model was rejected")
    bridge, _ = sessions(anthropic_connection(), adapter=adapter)
    session = bridge.new_chat(connection="claude-sub")

    events = list(session.send("hi"))
    assert events[-1].status is TerminalStatus.ERROR
    assert "400" in (events[-1].reason or "")
    assert text(events) == "partial"


def test_quota_exhaustion_discloses_that_a_session_does_not_fail_over(sessions):
    """Sessions do not cross connections; staying quiet about it would be worse."""
    connection = anthropic_connection(
        guards=Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api"))
    )
    bridge, _ = sessions(connection, script=[quota_exhausted()])
    session = bridge.new_chat(connection="claude-sub")

    terminal = list(session.send("hi"))[-1]
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED
    assert "claude-api" in terminal.reason
    assert "does not apply to a session" in terminal.reason


def test_guards_and_the_shorthands_still_refuse_each_other(sessions):
    bridge, _ = sessions(anthropic_connection())
    with pytest.raises(InvalidGuards):
        bridge.new_chat(
            connection="claude-sub", guards=Guards(stop_at_tokens=10), stop_at_tokens=20
        )


# --- raise at construction ------------------------------------------------------


def test_ephemeral_multi_turn_is_required_for_persist_false(sessions):
    bridge, _ = sessions(codex_connection())
    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.new_chat(connection="codex-sub", persist=False)
    assert caught.value.capability == "ephemeral_multi_turn"
    assert "persist=False" in str(caught.value)


def test_tools_in_process_is_required_for_tools(sessions):
    """And since 2026-08-31 the no can come from the request rather than the row.

    Codex gained ``tools_in_process`` when its default transport flipped, so
    the gate is exercised here in the direction that still refuses: a session
    that opted out into a transport without an in-process tool mechanism. The
    refusal must still name the keyword the caller typed.
    """
    tool = ToolDef(
        name="look_up",
        description="Look something up.",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "x",
    )
    bridge, _ = sessions(codex_connection(), adapter=NarrowingSessionAdapter())
    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.new_chat(connection="codex-sub", tools=[tool], options=EXEC)
    assert caught.value.capability == "tools_in_process"
    assert "tools=" in str(caught.value)

    # And with no option at all the registry answers, and it answers yes.
    assert bridge.new_chat(connection="codex-sub", tools=[tool]) is not None


def test_system_prompt_replace_is_required_for_a_chat_but_not_a_worker(sessions, tmp_path):
    """The one fact that separates the two objects (D16), still asked per request.

    A worker never needs the cell, on any transport, because append is the
    semantics it asked for. A chat needs it, and on 2026-08-31 Codex started
    answering yes by default -- so the refusal is exercised on the opt-out,
    which is the only place it is still true.
    """
    bridge, _ = sessions(codex_connection(), adapter=NarrowingSessionAdapter())

    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.new_chat(
            connection="codex-sub",
            system_prompt="You are a rubric scorer.",
            options=EXEC,
        )
    assert caught.value.capability == "system_prompt_replace"
    assert "new_worker" in str(caught.value)

    # The same chat with no transport option opens: baseInstructions replaces.
    chat = bridge.new_chat(
        connection="codex-sub", system_prompt="You are a rubric scorer."
    )
    assert isinstance(chat, ChatSession)
    assert chat.system_prompt == "You are a rubric scorer."

    # And a worker never asked for the cell in the first place -- on either.
    worker = bridge.new_worker(
        connection="codex-sub",
        project_folder=str(tmp_path),
        system_prompt="Follow the house style.",
        options=EXEC,
    )
    assert isinstance(worker, WorkerSession)
    assert worker.system_prompt == "Follow the house style."


def test_a_chat_without_a_system_prompt_is_fine_on_a_runtime_that_cannot_replace(sessions):
    bridge, _ = sessions(codex_connection())
    session = bridge.new_chat(connection="codex-sub")
    assert session.system_prompt is None


def test_a_worker_needs_a_project_folder(sessions):
    bridge, _ = sessions(anthropic_connection())
    with pytest.raises(InvalidSession) as caught:
        bridge.new_worker(connection="claude-sub", project_folder="  ")
    assert "project_folder" in str(caught.value)


def test_a_chat_gets_an_isolated_scratch_folder_by_default(sessions):
    bridge, made = sessions(anthropic_connection())
    session = bridge.new_chat(connection="claude-sub")
    assert os.path.isdir(session.project_folder)
    assert session.project_folder != os.getcwd()
    assert made["claude-sub"].session_requests[0].project_folder == session.project_folder


def test_persist_false_with_a_session_id_is_refused(sessions):
    bridge, _ = sessions(anthropic_connection())
    with pytest.raises(InvalidSession) as caught:
        bridge.resume_chat(
            connection="claude-sub",
            session_id="sess-9",
            system_prompt="be terse",
            persist=False,
        )
    assert "nothing on disk" in str(caught.value)


def test_mcp_servers_are_gated(sessions):
    bridge, _ = sessions(anthropic_connection())
    registry = CapabilityRegistry()
    registry.refine(Runtime.ANTHROPIC_SDK, {"mcp_servers": False})
    bridge.registry = registry
    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.new_chat(connection="claude-sub", mcp_servers={"docs": {"type": "stdio"}})
    assert caught.value.capability == "mcp_servers"


def test_a_disabled_connection_refuses_to_open_a_session(sessions):
    bridge, _ = sessions(anthropic_connection(enabled=False))
    with pytest.raises(ConnectionDisabled):
        bridge.new_chat(connection="claude-sub")


def test_expect_auth_mode_is_asserted_at_construction(sessions):
    bridge, _ = sessions(anthropic_connection())
    with pytest.raises(AuthModeMismatch):
        bridge.new_chat(connection="claude-sub", expect_auth_mode=AuthMode.API_KEY)


def test_capability_refusals_cost_nothing(sessions):
    """No preflight, no handle: asking for the impossible must not launch anything."""
    bridge, made = sessions(codex_connection())
    with pytest.raises(CapabilityNotSupported):
        bridge.new_chat(connection="codex-sub", persist=False)
    assert made["codex-sub"].preflights == []
    assert made["codex-sub"].handles == []


# --- the adapter seam -----------------------------------------------------------


def test_an_adapter_without_sessions_says_which_method_is_missing(sessions):
    """Not an AttributeError about Python -- a sentence about modelpass."""
    bridge, _ = sessions(anthropic_connection(), adapter=FakeAdapter())
    with pytest.raises(AdapterNotImplemented) as caught:
        bridge.new_chat(connection="claude-sub")
    assert "open_session" in str(caught.value)
    assert "anthropic-sdk" in str(caught.value)
    assert isinstance(caught.value, NotImplementedError)


def test_the_session_request_carries_the_policy_the_adapter_maps(sessions, tmp_path):
    bridge, made = sessions(anthropic_connection())

    bridge.new_chat(connection="claude-sub", system_prompt="be terse")
    chat_request = made["claude-sub"].session_requests[-1]
    assert chat_request.kind is SessionKind.CHAT
    assert chat_request.native_tools is False
    assert chat_request.appends_system_prompt is False

    bridge.new_worker(connection="claude-sub", project_folder=str(tmp_path))
    worker_request = made["claude-sub"].session_requests[-1]
    assert worker_request.kind is SessionKind.WORKER
    assert worker_request.native_tools is True
    assert worker_request.appends_system_prompt is True
    assert worker_request.project_folder == str(tmp_path)


def test_the_session_receipt_attributes_the_model_to_the_session(sessions):
    bridge, made = sessions(anthropic_connection())
    session = bridge.new_chat(connection="claude-sub", model="claude-opus-5")
    assert session.receipt.model == "claude-opus-5"
    assert "session" in session.receipt.model_source
    assert isinstance(made["claude-sub"].preflights[0], RunRequest)


# --- resume ---------------------------------------------------------------------


def test_resume_knows_its_id_immediately_and_passes_it_to_the_adapter(sessions):
    bridge, made = sessions(anthropic_connection())
    session = bridge.resume_chat(
        connection="claude-sub", session_id="sess-42", system_prompt="be terse"
    )

    assert session.id == "sess-42"
    assert made["claude-sub"].session_requests[0].resume_id == "sess-42"


def test_a_missing_session_raises_session_not_found(sessions):
    adapter = FakeSessionAdapter(missing=True)
    bridge, _ = sessions(anthropic_connection(), adapter=adapter)
    with pytest.raises(SessionNotFound) as caught:
        bridge.resume_chat(
            connection="claude-sub",
            session_id="sess-gone",
            system_prompt="be terse",
        )
    assert caught.value.session_id == "sess-gone"
    assert "no rollout found" in str(caught.value)


def test_resume_does_not_accept_tools(sessions):
    """Tools are the front of a warm prefix and cannot be re-declared (D17)."""
    bridge, _ = sessions(anthropic_connection())
    with pytest.raises(TypeError):
        bridge.resume_chat(
            connection="claude-sub", session_id="s", tools=[]
        )  # type: ignore[call-arg]


def test_resume_requires_the_system_prompt_where_the_runtime_drops_it(sessions):
    """anthropic-sdk keeps the prompt in the launch, not the transcript.

    Resuming without one would run under the minimal prompt against a cold
    prefix -- a wrong answer and a cache miss, both silent. So it is refused.
    """
    bridge, _ = sessions(anthropic_connection())
    with pytest.raises(InvalidSession) as excinfo:
        bridge.resume_chat(connection="claude-sub", session_id="s")
    assert "required" in str(excinfo.value)
    assert "launch option" in str(excinfo.value)


def test_resume_refuses_the_system_prompt_where_the_runtime_replays_it(sessions):
    """openai-sdk serialized it into the first turn; the rollout replays it.

    Supplying one here would inject a second persona mid-conversation rather
    than restore the first.
    """
    bridge, _ = sessions(codex_connection())
    with pytest.raises(InvalidSession) as excinfo:
        bridge.resume_chat(
            connection="codex-sub", session_id="s", system_prompt="new"
        )
    assert "refused" in str(excinfo.value)


# --- listing --------------------------------------------------------------------


def test_list_sessions_is_gated_and_normalized(sessions):
    listing = (
        SessionInfo(
            id="sess-1",
            connection="claude-sub",
            runtime=Runtime.ANTHROPIC_SDK,
            project_folder="/work",
            updated_at="2026-08-30T10:00:00Z",
            vendor={"leafUuid": "abc"},
        ),
    )
    adapter = FakeSessionAdapter(listing=listing)
    bridge, made = sessions(anthropic_connection(), adapter=adapter)

    found = bridge.list_sessions(connection="claude-sub", project_folder="/work")
    assert [s.id for s in found] == ["sess-1"]
    assert found[0].vendor == {"leafUuid": "abc"}
    assert made["claude-sub"].listed[0].project_folder == "/work"

    # And the round trip D11 asks of every value type.
    encoded = json.loads(json.dumps(found[0].to_dict()))
    assert encoded["runtime"] == "anthropic-sdk"
    assert encoded["created_at"] is None


def test_list_sessions_refuses_a_request_that_cannot_enumerate(sessions):
    """Codex enumerates by default since 2026-08-31; the opt-out still cannot."""
    bridge, _ = sessions(codex_connection(), adapter=NarrowingSessionAdapter())
    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.list_sessions(connection="codex-sub", options=EXEC)
    assert caught.value.capability == "sessions_list"

    # Without the opt-out the registry answers, and the listing happens.
    assert list(bridge.list_sessions(connection="codex-sub")) == []


def test_list_sessions_defaults_to_the_process_directory(sessions):
    adapter = FakeSessionAdapter()
    bridge, made = sessions(anthropic_connection(), adapter=adapter)
    bridge.list_sessions(connection="claude-sub")
    assert made["claude-sub"].listed[0].project_folder == os.getcwd()


# --- help -----------------------------------------------------------------------


def test_help_is_class_aware_about_the_system_prompt(sessions, tmp_path):
    """A chat leads with replace; a worker does not mention it (D15's closing note)."""
    bridge, _ = sessions(codex_connection())

    chat = bridge.new_chat(connection="codex-sub").help()
    assert "system_prompt_replace" in chat
    assert "Limits on openai-sdk" in chat
    # The registry's own prose, which until now had nowhere to be read.
    assert "no system-prompt parameter" in chat

    worker = bridge.new_worker(
        connection="codex-sub", project_folder=str(tmp_path)
    ).help()
    assert "system_prompt_replace" not in worker
    assert "no system-prompt parameter" not in worker


def test_help_has_no_limits_section_on_a_runtime_that_can_do_everything(sessions):
    bridge, _ = sessions(anthropic_connection())
    rendered = bridge.new_chat(connection="claude-sub").help()
    assert "Limits on" not in rendered


def test_help_carries_the_shape_the_receipt_and_the_id_timing(sessions):
    bridge, _ = sessions(anthropic_connection())
    session = bridge.new_chat(
        connection="claude-sub", system_prompt="be terse", stop_at_tokens=500
    )

    before = session.help()
    assert "ChatSession on connection 'claude-sub' (anthropic-sdk)" in before
    assert "replacing the runtime's own persona" in before
    assert "500 tokens for the whole session" in before
    assert "the first send creates it" in before
    assert session.receipt.summary() in before
    assert "Capabilities this object uses" in before

    list(session.send("hi"))
    assert "sess-1" in session.help()


def test_help_says_when_nothing_bounds_the_spend(sessions):
    bridge, _ = sessions(anthropic_connection())
    rendered = bridge.new_chat(connection="claude-sub").help()
    assert "nothing bounds what this session may spend" in rendered


def test_worker_help_reports_the_native_toolbelt(sessions, tmp_path):
    bridge, _ = sessions(anthropic_connection())
    rendered = bridge.new_worker(
        connection="claude-sub", project_folder=str(tmp_path)
    ).help()
    assert "the runtime's own toolbelt" in rendered
    assert str(tmp_path) in rendered


def test_the_receipt_reports_the_sessions_own_ceiling_not_the_connections(sessions):
    """Two true halves that contradicted each other on screen (the plan carries
    the connection's guards; the session may have tighter ones)."""
    bridge, _ = sessions(anthropic_connection())

    tightened = bridge.new_chat(connection="claude-sub", stop_at_tokens=500)
    assert tightened.receipt.guards_configured is True
    assert "no spend guards configured" not in tightened.help()

    unbounded = bridge.new_chat(connection="claude-sub")
    assert unbounded.receipt.guards_configured is False


def test_help_leads_with_the_cell_that_defines_the_class(sessions):
    """And it asks for THIS session, which is why the answer moved on 2026-08-31.

    ``help()`` consults the adapter the same way the gates do, so a chat that
    opted out into ``codex exec`` leads with ``system_prompt_replace`` -- the
    cell it was refused by, and the reason the class exists. The same chat on
    the default does not list it at all, because it is no longer a limit
    there: a ChatSession that opened BECAUSE replace works must not then print
    that replace is unsupported.
    """
    bridge, _ = sessions(codex_connection(), adapter=NarrowingSessionAdapter())

    on_exec = bridge.new_chat(connection="codex-sub", options=EXEC).help()
    limits = on_exec.split("Limits on openai-sdk")[1]
    assert limits.lstrip().startswith("system_prompt_replace is unsupported")

    on_default = bridge.new_chat(connection="codex-sub").help()
    default_limits = on_default.split("Limits on openai-sdk")[1]
    assert "system_prompt_replace is unsupported" not in default_limits
