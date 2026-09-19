"""The second-consumer friction fixes (2026-08-17).

Four small things a retrieval service's integration report named, each of which
was
possible before and unreasonably indirect. Nothing here is a new capability;
it is the difference between a library that *can* do something and one that
lets you say it.
"""

from __future__ import annotations

import pytest

from modelpass.capabilities import CapabilityRegistry, Support
from modelpass.connections import (
    Connection,
    CredentialRef,
    Guards,
    QuotaAction,
    QuotaPolicy,
)
from modelpass.errors import InvalidGuards
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter, fake_bridge, quota_exhausted
from modelpass.types import AuthMode, FailoverEvent, TerminalStatus, TokenUsage


def anthropic(name: str = "claude-sub", **kwargs) -> Connection:
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        **kwargs,
    )


def metered(name: str = "claude-api") -> Connection:
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
    )


def failing_over_to(name: str, **guard_kwargs) -> Guards:
    return Guards(
        on_quota_exhausted=QuotaPolicy(action=QuotaAction.FAILOVER, failover=name),
        **guard_kwargs,
    )


# --- 1. declining a failover per call --------------------------------------------


def failover_bridge(tmp_path, **chat_kwargs):
    primary = anthropic(guards=failing_over_to("claude-api"))
    bridge, _, _ = fake_bridge(
        connections=[primary, metered()],
        home=tmp_path / "home",
        env={"ANTHROPIC_API_KEY": "sk-test"},
        adapter=FakeAdapter([quota_exhausted()]),
    )
    events = list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            **chat_kwargs,
        )
    )
    return events


def test_by_default_a_configured_failover_still_runs(tmp_path):
    events = failover_bridge(tmp_path)
    assert any(isinstance(e, FailoverEvent) for e in events)


def test_allow_failover_false_declines_it_for_this_call(tmp_path):
    events = failover_bridge(tmp_path, allow_failover=False)
    assert not any(isinstance(e, FailoverEvent) for e in events)
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED
    assert events[-1].connection == "claude-sub"


def test_allow_failover_none_respects_the_connection(tmp_path):
    events = failover_bridge(tmp_path, allow_failover=None)
    assert any(isinstance(e, FailoverEvent) for e in events)


def test_allow_failover_true_is_the_default_said_out_loud(tmp_path):
    events = failover_bridge(tmp_path, allow_failover=True)
    assert any(isinstance(e, FailoverEvent) for e in events)


def test_allow_failover_true_cannot_conjure_a_failover_nobody_configured(tmp_path):
    """The consent for metered billing is the connection file, not a keyword."""
    bridge, _, _ = fake_bridge(
        connections=[anthropic(), metered()],
        home=tmp_path / "home",
        env={"ANTHROPIC_API_KEY": "sk-test"},
        adapter=FakeAdapter([quota_exhausted()]),
    )
    events = list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            allow_failover=True,
        )
    )
    assert not any(isinstance(e, FailoverEvent) for e in events)
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED


def test_declining_a_failover_leaves_the_thresholds_alone(tmp_path):
    """It composes: it is a shorthand over the connection's own guards."""
    primary = anthropic(
        guards=failing_over_to("claude-api", warn_at_tokens=10, stop_at_tokens=20)
    )
    bridge, _, _ = fake_bridge(
        connections=[primary, metered()],
        home=tmp_path / "home",
        env={"ANTHROPIC_API_KEY": "sk-test"},
        adapter=FakeAdapter([]),
    )
    guards = bridge._call_guards(primary, None, None, None, False)
    assert guards.warn_at_tokens == 10 and guards.stop_at_tokens == 20
    assert guards.on_quota_exhausted.action is QuotaAction.STOP


def test_lowering_a_ceiling_still_does_not_disarm_a_failover(tmp_path):
    primary = anthropic(guards=failing_over_to("claude-api", stop_at_tokens=100))
    bridge, _, _ = fake_bridge(
        connections=[primary, metered()], home=tmp_path / "home", adapter=FakeAdapter([])
    )
    guards = bridge._call_guards(primary, None, 50, None, None)
    assert guards.stop_at_tokens == 50
    assert guards.on_quota_exhausted.action is QuotaAction.FAILOVER


def test_the_shorthands_compose_with_each_other(tmp_path):
    primary = anthropic(guards=failing_over_to("claude-api", stop_at_tokens=100))
    bridge, _, _ = fake_bridge(
        connections=[primary, metered()], home=tmp_path / "home", adapter=FakeAdapter([])
    )
    guards = bridge._call_guards(primary, None, 50, 10, False)
    assert (guards.stop_at_tokens, guards.warn_at_tokens) == (50, 10)
    assert guards.on_quota_exhausted.action is QuotaAction.STOP


def test_allow_failover_with_an_explicit_guards_object_is_refused(tmp_path):
    bridge, _, _ = fake_bridge(
        connections=[anthropic()], home=tmp_path / "home", adapter=FakeAdapter([])
    )
    with pytest.raises(InvalidGuards) as exc:
        bridge.chat(
            connection="claude-sub",
            message="hi",
            guards=Guards(stop_at_tokens=5),
            allow_failover=False,
        )
    assert "allow_failover" in str(exc.value)


# --- 2. the TokenUsage convention ------------------------------------------------


def test_total_tokens_sums_all_three_fields():
    """Written down because shim authors kept deriving it (and disagreeing)."""
    usage = TokenUsage(input_tokens=10, output_tokens=5, cached_input_tokens=3)
    assert usage.total_tokens == 18


def test_cached_is_counted_separately_from_input():
    usage = TokenUsage(input_tokens=10, cached_input_tokens=3)
    assert usage.input_tokens == 10, "cached is not inside input"
    assert usage.total_tokens == 13


# --- 3. Bridge.validate -- the offline half of preflight -------------------------


def test_a_good_connection_validates(tmp_path):
    bridge, _, _ = fake_bridge(connections=[anthropic()], home=tmp_path / "home")
    report = bridge.validate("claude-sub")
    assert report.ok and report.problems == ()
    assert report.runtime is Runtime.ANTHROPIC_SDK
    assert report.auth_mode is AuthMode.SUBSCRIPTION


def test_validate_never_touches_the_adapter(tmp_path):
    """The whole point: usable on a machine with no vendor package installed."""
    bridge, _, adapter = fake_bridge(connections=[anthropic()], home=tmp_path / "home")
    bridge.validate("claude-sub")
    assert adapter.preflights == [] and adapter.requests == []


def test_an_uninstalled_runtime_is_a_note_not_a_problem(tmp_path):
    bridge, _, _ = fake_bridge(
        connections=[anthropic()],
        home=tmp_path / "home",
        adapter=FakeAdapter([], available=False),
    )
    report = bridge.validate("claude-sub")
    assert report.ok, "the configuration is valid; it just cannot run here"
    assert not report.runtime_installed
    assert any("not installed" in note for note in report.notes)


def test_an_unknown_connection_is_reported_not_raised(tmp_path):
    bridge, _, _ = fake_bridge(connections=[anthropic()], home=tmp_path / "home")
    report = bridge.validate("nope")
    assert not report.ok
    assert "no connection named 'nope'" in report.problems[0]


def test_a_missing_credential_is_a_problem(tmp_path):
    bridge, _, _ = fake_bridge(
        connections=[metered()], home=tmp_path / "home", env={}
    )
    report = bridge.validate("claude-api")
    assert not report.ok
    assert any("ANTHROPIC_API_KEY" in problem for problem in report.problems)


def test_a_broken_failover_target_is_found_offline(tmp_path):
    """The whole reason a config validator is worth having."""
    bridge, _, _ = fake_bridge(
        connections=[anthropic(guards=failing_over_to("typo"))], home=tmp_path / "home"
    )
    report = bridge.validate("claude-sub")
    assert not report.ok
    assert "typo" in report.problems[0]


def test_an_unsupported_auth_mode_is_a_problem(tmp_path):
    registry = CapabilityRegistry()
    registry.refine(Runtime.ANTHROPIC_SDK, {"subscription_auth": Support.UNSUPPORTED})
    bridge, _, _ = fake_bridge(
        connections=[anthropic()], home=tmp_path / "home", registry=registry
    )
    report = bridge.validate("claude-sub")
    assert not report.ok
    assert "cannot drive auth mode" in report.problems[0]


def test_a_disabled_connection_is_valid_and_says_so(tmp_path):
    bridge, _, _ = fake_bridge(
        connections=[anthropic(enabled=False)], home=tmp_path / "home"
    )
    report = bridge.validate("claude-sub")
    assert report.ok and not report.enabled
    assert any("disabled" in note for note in report.notes)


def test_absent_guards_are_visible_rather_than_silent(tmp_path):
    bridge, _, _ = fake_bridge(connections=[anthropic()], home=tmp_path / "home")
    report = bridge.validate("claude-sub")
    assert any("no spend guards" in note for note in report.notes)


def test_the_report_serializes_and_summarizes(tmp_path):
    import json

    bridge, _, _ = fake_bridge(connections=[anthropic()], home=tmp_path / "home")
    report = bridge.validate("claude-sub")
    json.dumps(report.to_dict())
    assert report.summary().startswith("claude-sub: ok (anthropic-sdk)")


def test_validate_accepts_an_inline_connection(tmp_path):
    bridge, _, _ = fake_bridge(home=tmp_path / "home")
    assert bridge.validate(anthropic()).ok


# --- 4. fake_bridge --------------------------------------------------------------


def test_fake_bridge_wires_the_adapter_to_the_connections_runtime(tmp_path):
    bridge, store, adapter = fake_bridge(
        connections=[anthropic()], home=tmp_path / "home", script=[]
    )
    assert bridge.adapter_for(Runtime.ANTHROPIC_SDK) is adapter
    assert [c.name for c in store.list()] == ["claude-sub"]


def test_fake_bridge_takes_a_script_directly(tmp_path):
    from modelpass.types import TextDeltaEvent

    bridge, _, _ = fake_bridge(
        connections=[anthropic()], home=tmp_path / "home", script=[TextDeltaEvent("hi")]
    )
    events = list(bridge.chat(connection="claude-sub", message="x"))
    assert any(isinstance(e, TextDeltaEvent) and e.text == "hi" for e in events)


def test_fake_bridge_can_target_another_runtime(tmp_path):
    codex = Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    bridge, _, adapter = fake_bridge(
        connections=[codex], home=tmp_path / "home", runtime=Runtime.OPENAI_SDK
    )
    assert adapter.runtime is Runtime.OPENAI_SDK
    assert bridge.adapter_for(Runtime.OPENAI_SDK) is adapter


def test_fake_bridge_keeps_the_run_log_off_the_real_home(tmp_path):
    home = tmp_path / "home"
    bridge, _, _ = fake_bridge(
        connections=[anthropic()], home=home, script=[]
    )
    list(bridge.chat(connection="claude-sub", message="x"))
    assert (home / "runs.jsonl").is_file()
