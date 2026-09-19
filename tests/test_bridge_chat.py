from __future__ import annotations

from dataclasses import replace

import pytest

from modelpass.adapters import load_adapter
from modelpass.bridge import Bridge
from modelpass.capabilities import Capability, CapabilityRegistry, Support
from modelpass.connections import AccountBinding, Connection, Guards, QuotaAction, QuotaPolicy
from modelpass.errors import (
    AdapterFailed,
    AuthModeMismatch,
    CapabilityNotSupported,
    GuardStop,
    InvalidTool,
    NoSuchConnection,
    QuotaExhausted,
    RuntimeNotAvailable,
)
from modelpass.preflight import AccountProfile
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter, tool_exchange, usage
from modelpass.tools import ToolDef
from modelpass.types import (
    AuthMode,
    GuardStopEvent,
    GuardWarningEvent,
    Message,
    ReceiptEvent,
    Role,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    ThinkingEvent,
    UsageEvent,
    VendorEvent,
)


def text(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDeltaEvent))


def test_a_plain_run_streams_deltas_and_ends_with_one_terminal(
    bridge_factory, subscription_connection
):
    script = [
        ThinkingEvent(text="considering"),
        TextDeltaEvent(text="hello "),
        TextDeltaEvent(text="world"),
        UsageEvent(usage=usage(input_tokens=5, output_tokens=7).usage),
    ]
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter(script))

    events = list(bridge.chat(connection="claude-sub", message="hi"))

    assert text(events) == "hello world"
    assert sum(isinstance(e, TerminalEvent) for e in events) == 1
    terminal = events[-1]
    assert terminal.status is TerminalStatus.OK
    assert terminal.usage.total_tokens == 12
    assert fake.requests[0].messages[0].content == "hi"


def test_account_oriented_bridge_aliases_share_the_connection_source_of_truth(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(subscription_connection)
    assert bridge.accounts() == bridge.connections()
    assert bridge.account("claude-sub") == bridge.connection("claude-sub")


def test_find_can_select_accounts_by_vendor(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(subscription_connection)
    assert bridge.find(vendor="Anthropic") == (subscription_connection,)
    assert bridge.find(vendor="openai") == ()


def test_the_terminal_event_is_stamped_with_connection_and_auth_mode(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([TextDeltaEvent(text="x")]))
    terminal = list(bridge.chat(connection="claude-sub", message="hi"))[-1]
    assert terminal.connection == "claude-sub"
    assert terminal.runtime is Runtime.ANTHROPIC_SDK
    assert terminal.auth_mode is AuthMode.SUBSCRIPTION


def test_an_adapter_cannot_forge_the_stamp(bridge_factory, subscription_connection):
    """An adapter reports a status; the bridge owns the identity fields (D3)."""
    forged = TerminalEvent(
        status=TerminalStatus.OK,
        connection="something-else",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.API_KEY,
        reason="done",
    )
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([forged]))
    terminal = list(bridge.chat(connection="claude-sub", message="hi"))[-1]
    assert terminal.connection == "claude-sub"
    assert terminal.auth_mode is AuthMode.SUBSCRIPTION
    assert terminal.reason == "done"


def test_the_stamp_follows_the_detected_mode_not_the_declared_one(
    bridge_factory, api_connection
):
    bridge, _ = bridge_factory(
        api_connection,
        FakeAdapter([TextDeltaEvent(text="x")]),
        env={"ANTHROPIC_API_KEY": "sk-not-a-real-key"},
    )
    terminal = list(bridge.chat(connection="claude-api", message="hi"))[-1]
    assert terminal.auth_mode is AuthMode.API_KEY


def test_a_pinned_account_identity_is_verified_before_a_run(
    subscription_connection, bridge_factory
):
    connection = replace(
        subscription_connection,
        account_binding=AccountBinding(
            email="work@example.com", organization_id="org-work"
        ),
    )
    adapter = FakeAdapter(
        account_profile=AccountProfile(
            vendor="anthropic",
            source="test",
            email="work@example.com",
            organization_id="org-work",
        )
    )
    bridge, _ = bridge_factory(connection, adapter)
    receipt = bridge.preflight(connection.name)
    assert receipt.ok is True
    assert receipt.identity_verified is True


def test_account_drift_fails_closed_before_model_traffic(
    subscription_connection, bridge_factory
):
    connection = replace(
        subscription_connection,
        nickname="Anthropic Work",
        account_binding=AccountBinding(
            email="work@example.com", organization_id="org-work"
        ),
    )
    adapter = FakeAdapter(
        account_profile=AccountProfile(
            vendor="anthropic",
            source="test",
            email="personal@example.com",
            organization_id="org-personal",
        )
    )
    bridge, _ = bridge_factory(connection, adapter)
    receipt = bridge.preflight(connection.name)
    assert receipt.ok is False
    assert receipt.identity_verified is False
    assert "ACCOUNT IDENTITY MISMATCH" in (receipt.problem or "")
    assert adapter.requests == []


def test_a_pinned_account_whose_probe_says_nothing_fails_closed_with_advice(
    subscription_connection, bridge_factory
):
    """A pin nobody can check is not a pass -- but the text must say how to fix it."""
    connection = replace(
        subscription_connection,
        account_binding=AccountBinding(email="work@example.com"),
    )
    bridge, adapter = bridge_factory(connection, FakeAdapter(account_profile=None))
    receipt = bridge.preflight(connection.name)
    assert receipt.ok is False
    assert receipt.identity_verified is False
    problem = receipt.problem or ""
    assert "returned nothing" in problem
    assert f"modelpass verify {connection.name}" in problem
    assert adapter.requests == []


def test_the_unpinned_note_is_emitted_once_per_receipt(
    subscription_connection, bridge_factory
):
    """Wrapping a receipt twice (the session path does) must not duplicate it."""
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter())
    receipt = bridge.preflight(subscription_connection.name)
    unpinned = [n for n in receipt.notes if "not pinned" in n]
    assert len(unpinned) == 1

    twice = Bridge._with_identity_verification(subscription_connection, receipt)
    assert [n for n in twice.notes if "not pinned" in n] == unpinned


def test_refresh_identity_drops_the_adapters_cached_probe(
    subscription_connection, bridge_factory
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter())
    # The default FakeAdapter caches nothing, so this is the base-class contract:
    # every adapter answers the call, and nothing raises.
    bridge.refresh_identity(subscription_connection.name)


def test_usage_events_are_enriched_with_a_running_total(
    bridge_factory, subscription_connection
):
    script = [usage(input_tokens=10), usage(output_tokens=5)]
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter(script))
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    totals = [e.cumulative.total_tokens for e in events if isinstance(e, UsageEvent)]
    assert totals == [10, 15]


def test_warn_threshold_emits_one_warning(bridge_factory, subscription_connection):
    # guards: warn at 100, stop at 250
    script = [usage(input_tokens=60), usage(input_tokens=60), usage(input_tokens=10)]
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter(script))
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    warnings = [e for e in events if isinstance(e, GuardWarningEvent)]
    assert len(warnings) == 1
    assert warnings[0].threshold == 100
    assert warnings[0].observed == 120
    assert warnings[0].connection == "claude-sub"
    assert events[-1].status is TerminalStatus.OK


def test_stop_threshold_stops_the_run_and_cancels_the_adapter(
    bridge_factory, subscription_connection
):
    script = [
        TextDeltaEvent(text="before"),
        usage(input_tokens=300),
        TextDeltaEvent(text="after"),
    ]
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter(script))
    events = list(bridge.chat(connection="claude-sub", message="hi"))

    assert text(events) == "before"
    stops = [e for e in events if isinstance(e, GuardStopEvent)]
    assert len(stops) == 1
    assert stops[0].threshold == 250
    assert events[-1].status is TerminalStatus.GUARD_STOP
    assert events[-1].usage.total_tokens == 300
    assert fake.cancelled == 1
    assert fake.closed is True


def test_guard_stop_can_be_raised_instead(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([usage(input_tokens=300)]))
    stream = bridge.chat(
        connection="claude-sub",
        message="hi",
        raise_on_stop=True,
    )
    with pytest.raises(GuardStop):
        list(stream)


def test_per_call_guards_override_without_touching_config(
    bridge_factory, subscription_connection, store
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([usage(input_tokens=300)]))
    events = list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            guards=Guards.disabled(),
        )
    )
    assert not any(isinstance(e, GuardStopEvent) for e in events)
    assert events[-1].status is TerminalStatus.OK
    assert store.get("claude-sub").guards.stop_at_tokens == 250


def test_quota_exhaustion_stops_cleanly_and_never_fails_over(
    bridge_factory, subscription_connection
):
    script = [
        TerminalEvent(
            status=TerminalStatus.QUOTA_EXHAUSTED,
            connection="",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            reason="agent sdk credit exhausted",
        )
    ]
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter(script))
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED
    assert events[-1].auth_mode is AuthMode.SUBSCRIPTION

    with pytest.raises(QuotaExhausted):
        list(
            bridge.chat(
                connection="claude-sub",
                message="hi",
                raise_on_stop=True,
            )
        )


def test_a_failover_naming_an_unconfigured_connection_fails_before_any_spend(
    bridge_factory, subscription_connection
):
    """Failover behavior itself lives in test_failover.py; this pins the timing."""
    connection = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        guards=Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "not-configured")),
    )
    bridge, fake = bridge_factory(connection, FakeAdapter([]))
    with pytest.raises(NoSuchConnection):
        bridge.chat(connection="claude-sub", message="hi")
    assert fake.preflights == []


def test_vendor_events_pass_through_untouched(bridge_factory, subscription_connection):
    event = VendorEvent(
        runtime=Runtime.ANTHROPIC_SDK,
        name="system/api_retry",
        data={"attempt": 1, "total_cost_usd": 0.02},
    )
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([event]))
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    # events[0] is the stream's own ReceiptEvent; the adapter's output starts at 1.
    assert isinstance(events[0], ReceiptEvent)
    assert events[1] is event


def test_expect_auth_mode_is_an_assertion(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([]))
    with pytest.raises(AuthModeMismatch):
        bridge.chat(
            connection="claude-sub",
            message="hi",
            expect_auth_mode=AuthMode.API_KEY,
        )


def test_a_run_that_would_bill_differently_than_declared_is_refused(
    bridge_factory, subscription_connection
):
    """The silent-metered-billing case: the runtime resolved a key we did not expect."""
    fake = FakeAdapter([], detected_auth_mode=AuthMode.API_KEY)
    bridge, _ = bridge_factory(subscription_connection, fake)
    with pytest.raises(AuthModeMismatch):
        bridge.chat(connection="claude-sub", message="hi")


def test_the_adapter_receives_a_scrubbed_environment(bridge_factory, subscription_connection):
    bridge, fake = bridge_factory(
        subscription_connection,
        FakeAdapter([]),
        env={"ANTHROPIC_API_KEY": "sk-not-a-real-key", "PATH": "/usr/bin"},
    )
    list(bridge.chat(connection="claude-sub", message="hi"))
    assert dict(fake.requests[0].env) == {"PATH": "/usr/bin"}
    assert fake.requests[0].plan.scrubbed == ("ANTHROPIC_API_KEY",)


def test_closing_the_stream_cancels_the_run(bridge_factory, subscription_connection):
    script = [TextDeltaEvent(text="a"), TextDeltaEvent(text="b")]
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter(script))
    stream = bridge.chat(connection="claude-sub", message="hi")
    assert isinstance(next(stream), ReceiptEvent)
    assert next(stream).text == "a"
    stream.close()
    assert fake.cancelled == 1


def test_capability_gating_refuses_a_runtime_that_cannot_chat(
    store, subscription_connection
):
    registry = CapabilityRegistry()
    registry.refine(Runtime.ANTHROPIC_SDK, {"chat": Support.UNSUPPORTED})
    store.add(subscription_connection)
    bridge = Bridge(
        store=store,
        registry=registry,
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter([])},
        env={},
    )
    with pytest.raises(CapabilityNotSupported):
        bridge.chat(connection="claude-sub", message="hi")


# --- tools and MCP servers (D12) -----------------------------------------------


def _bridge_with_support(store, connection, support_by_capability):
    registry = CapabilityRegistry()
    registry.refine(connection.runtime, support_by_capability)
    store.add(connection)
    fake = FakeAdapter([])
    bridge = Bridge(
        store=store,
        registry=registry,
        adapters={connection.runtime: fake},
        env={},
    )
    return bridge, fake


def test_tools_and_mcp_servers_reach_the_adapter(bridge_factory, subscription_connection):
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter([]))
    tool = ToolDef(name="look_up", description="Look it up.", handler=lambda args: "ok")

    list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            tools=[tool],
            mcp_servers={"amt": {"command": "python", "args": ["-m", "amt"]}},
        )
    )

    request = fake.requests[0]
    assert request.tools == (tool,)
    assert request.mcp_servers == {"amt": {"command": "python", "args": ["-m", "amt"]}}
    assert request.wants_tools is True


def test_a_plain_chat_call_carries_no_tools_at_all(bridge_factory, subscription_connection):
    """Phase 6 must not change what a D7 chat call asks the runtime for."""
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter([]))
    list(bridge.chat(connection="claude-sub", message="hi"))
    request = fake.requests[0]
    assert request.tools == ()
    assert request.mcp_servers == {}
    assert request.wants_tools is False


def test_requesting_tools_on_a_runtime_without_them_raises_before_any_spend(
    store, subscription_connection
):
    bridge, fake = _bridge_with_support(
        store, subscription_connection, {"tools_in_process": Support.UNSUPPORTED}
    )
    with pytest.raises(CapabilityNotSupported) as excinfo:
        bridge.chat(
            connection="claude-sub",
            message="hi",
            tools=[ToolDef(name="t", description="d", handler=lambda a: "x")],
        )
    assert excinfo.value.capability == "tools_in_process"
    # Nothing was launched, and crucially the preflight never even ran: finding
    # out a runtime cannot do something must not cost a token.
    assert fake.requests == []
    assert fake.preflights == []


def test_requesting_mcp_servers_on_a_runtime_without_them_raises(
    store, subscription_connection
):
    bridge, fake = _bridge_with_support(
        store, subscription_connection, {"mcp_servers": Support.UNSUPPORTED}
    )
    with pytest.raises(CapabilityNotSupported) as excinfo:
        bridge.chat(
            connection="claude-sub",
            message="hi",
            mcp_servers={"amt": {"command": "python"}},
        )
    assert excinfo.value.capability == "mcp_servers"
    assert fake.preflights == []


def test_unverified_support_is_refused_the_same_as_unsupported(
    store, subscription_connection
):
    """"We did not check" is not a yes (D6). It must not become one for tools."""
    bridge, _ = _bridge_with_support(
        store, subscription_connection, {"mcp_servers": Support.UNVERIFIED}
    )
    with pytest.raises(CapabilityNotSupported) as excinfo:
        bridge.chat(
            connection="claude-sub",
            message="hi",
            mcp_servers={"amt": {"command": "python"}},
        )
    assert excinfo.value.support == "unverified"


def test_a_malformed_tool_is_refused_before_the_capability_check_or_a_launch(
    bridge_factory, subscription_connection
):
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter([]))
    with pytest.raises(InvalidTool):
        bridge.chat(
            connection="claude-sub",
            message="hi",
            tools=[{"name": "bad name", "description": "d"}],
        )
    assert fake.preflights == []


def test_gating_only_fires_when_the_option_is_actually_used(
    store, subscription_connection
):
    """A runtime with no tool support still runs a plain chat call."""
    bridge, _ = _bridge_with_support(
        store,
        subscription_connection,
        {"tools_in_process": Support.UNSUPPORTED, "mcp_servers": Support.UNSUPPORTED},
    )
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    assert events[-1].status is TerminalStatus.OK


def test_scripted_tool_events_stream_through_untouched(
    bridge_factory, subscription_connection
):
    """The bridge normalizes usage and terminals; tool events it just carries."""
    call, result = tool_exchange("look_up", {"topic": "x"}, "found it")
    bridge, _ = bridge_factory(
        subscription_connection,
        FakeAdapter([call, TextDeltaEvent(text="the answer is "), result]),
    )

    events = list(bridge.chat(connection="claude-sub", message="hi"))

    assert events[1] == call
    assert events[3] == result
    assert events[1].id == events[3].id
    assert isinstance(events[-1], TerminalEvent)


def test_a_failed_tool_does_not_stop_the_run(bridge_factory, subscription_connection):
    """A tool failure is the model's problem to work around, not a run outcome."""
    call, result = tool_exchange("look_up", result="boom", is_error=True)
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([call, result]))
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    assert events[2].is_error is True
    assert events[-1].status is TerminalStatus.OK


def test_guards_still_stop_a_tool_run(bridge_factory, subscription_connection):
    """A tool loop is exactly the shape that can run away; guards must reach it."""
    call, result = tool_exchange("look_up", result="ok")
    bridge, fake = bridge_factory(
        subscription_connection,
        FakeAdapter([call, usage(input_tokens=300), result, TextDeltaEvent(text="never")]),
    )
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    assert any(isinstance(e, GuardStopEvent) for e in events)
    assert events[-1].status is TerminalStatus.GUARD_STOP
    assert text(events) == ""
    assert fake.cancelled >= 1


def test_find_answers_which_connection_supports_what(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([]))
    assert bridge.find(capability=Capability.SUBAGENTS) == (subscription_connection,)
    assert bridge.find(runtime=Runtime.OPENAI_SDK) == ()
    assert bridge.find(auth_mode=AuthMode.API_KEY) == ()


def test_preflight_returns_a_receipt_without_running(bridge_factory, subscription_connection):
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter([]))
    receipt = bridge.preflight("claude-sub")
    assert receipt.connection == "claude-sub"
    assert receipt.detected_auth_mode is AuthMode.SUBSCRIPTION
    assert "fake-account" in receipt.summary()
    assert fake.requests == []


def test_history_is_validated_before_anything_runs(bridge_factory, subscription_connection):
    bridge, fake = bridge_factory(subscription_connection, FakeAdapter([]))
    with pytest.raises(ValueError):
        bridge.chat(
            connection="claude-sub",
            message="hi",
            history=[{"role": "user"}],
        )
    assert fake.preflights == []


def test_an_adapter_blowing_up_is_wrapped_and_cancelled(
    bridge_factory, subscription_connection
):
    fake = FakeAdapter([TextDeltaEvent(text="a")], error=RuntimeError("vendor exploded"))
    bridge, _ = bridge_factory(subscription_connection, fake)
    stream = bridge.chat(connection="claude-sub", message="hi")
    with pytest.raises(AdapterFailed, match="vendor exploded") as excinfo:
        list(stream)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert fake.cancelled == 1


def test_the_openai_adapter_is_implemented():
    """Phase 4 landed: the adapter must no longer raise AdapterNotImplemented.

    Its behavior is covered by ``test_adapter_openai.py``.
    """
    from modelpass.adapters.openai import OpenAIAdapter

    adapter = OpenAIAdapter()
    assert adapter.runtime is Runtime.OPENAI_SDK
    with pytest.raises(AttributeError):
        # A malformed request is a plain programming error now, not a stub.
        adapter.preflight(None)  # type: ignore[arg-type]


def test_the_anthropic_adapter_is_implemented():
    """Phase 3 landed: the adapter must no longer raise AdapterNotImplemented."""
    from modelpass.adapters.anthropic import AnthropicAdapter

    adapter = AnthropicAdapter()
    assert adapter.runtime is Runtime.ANTHROPIC_SDK
    assert not isinstance(adapter.preflight, type(None))
    with pytest.raises(AttributeError):
        # A malformed request is a plain programming error now, not a stub.
        adapter.preflight(None)  # type: ignore[arg-type]


def test_a_missing_vendor_package_names_the_extra_to_install(monkeypatch):
    from modelpass.adapters import anthropic as anthropic_module

    monkeypatch.setattr(
        anthropic_module.AnthropicAdapter,
        "is_available",
        classmethod(lambda cls: False),
    )
    with pytest.raises(RuntimeNotAvailable, match=r"modelpass\[anthropic\]"):
        load_adapter(Runtime.ANTHROPIC_SDK)


# --- assembly: system_prompt, history, message (D21) ---------------------------


def test_a_bare_message_sends_exactly_one_user_turn(bridge_factory, subscription_connection):
    """No history to flatten, because there is no history to pass."""
    bridge, adapter = bridge_factory(subscription_connection, FakeAdapter([]))
    list(bridge.chat(connection="claude-sub", message="Score this."))

    sent = adapter.requests[0].messages
    assert [m.role.value for m in sent] == ["user"]
    assert sent[0].content == "Score this."


def test_the_system_prompt_stays_a_separate_message(
    bridge_factory, subscription_connection
):
    """It is the front of the cached prefix, so it must not be concatenated.

    A caller who folds instructions into the message varies the prefix per call
    and caches nothing; keeping it a distinct message is what lets the stable
    half stay stable across a scoring loop.
    """
    bridge, adapter = bridge_factory(subscription_connection, FakeAdapter([]))
    list(
        bridge.chat(
            connection="claude-sub",
            message="Score this.",
            system_prompt="You are a rubric.",
        )
    )

    sent = adapter.requests[0].messages
    assert [m.role.value for m in sent] == ["system", "user"]
    assert sent[0].content == "You are a rubric."
    assert sent[1].content == "Score this."


def test_without_a_system_prompt_no_system_message_is_added(
    bridge_factory, subscription_connection
):
    """An absent prompt is absent, not an empty one that changes the prefix."""
    bridge, adapter = bridge_factory(subscription_connection, FakeAdapter([]))
    list(bridge.chat(connection="claude-sub", message="Hello."))
    assert all(m.role.value != "system" for m in adapter.requests[0].messages)


def test_history_is_assembled_between_the_prompt_and_the_new_turn(
    bridge_factory, subscription_connection
):
    """Instructions, what was already said, then the turn being taken now.

    The order is the whole contract: an adapter renders a transcript from this
    list, so a history assembled anywhere else would change what the model is
    asked -- and move the cached prefix with it.
    """
    bridge, adapter = bridge_factory(subscription_connection, FakeAdapter([]))
    list(
        bridge.chat(
            connection="claude-sub",
            message="and now?",
            system_prompt="You are terse.",
            history=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "an earlier answer"},
            ],
        )
    )

    sent = [m.to_dict() for m in adapter.requests[0].messages]
    assert sent == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "an earlier answer"},
        {"role": "user", "content": "and now?"},
    ]


def test_history_accepts_message_objects_as_well_as_dicts(
    bridge_factory, subscription_connection
):
    """The same shapes a message list took, because it is the same normalizer."""
    bridge, adapter = bridge_factory(subscription_connection, FakeAdapter([]))
    list(
        bridge.chat(
            connection="claude-sub",
            message="again",
            history=[Message(role=Role.USER, content="before")],
        )
    )

    sent = [m.to_dict() for m in adapter.requests[0].messages]
    assert sent == [
        {"role": "user", "content": "before"},
        {"role": "user", "content": "again"},
    ]


def test_an_empty_history_is_the_single_shot_case(bridge_factory, subscription_connection):
    """``history=[]`` and no history at all are the same call, not two shapes."""
    bridge, adapter = bridge_factory(subscription_connection, FakeAdapter([]))
    list(bridge.chat(connection="claude-sub", message="hi", history=[]))
    sent = [m.to_dict() for m in adapter.requests[0].messages]
    assert sent == [{"role": "user", "content": "hi"}]
