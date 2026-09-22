"""Stating an effort once, and being told what it became (2026-09-21).

The ask was one ordered ladder ending in ``ultra``. It could not be one ladder:
``ultra`` turns on agentic execution as well as depth, so a caller asking to
think harder would be handed subagents. These hold the line between the two
axes, and hold the substitution reporting that keeps a portable ladder honest.
"""

from __future__ import annotations

import pytest

from modelpass.capabilities import CapabilityRegistry, Support
from modelpass.errors import InvalidConnection
from modelpass.reasoning import (
    EFFORT_LADDER,
    RUNTIME_EFFORTS,
    Effort,
    ReasoningDisposition,
    plan_reasoning,
)
from modelpass.runtimes import Runtime

# --- 1. the ladder is depth, and only depth ------------------------------------


def test_the_ladder_is_ordered_and_ascending():
    assert [e.rank for e in EFFORT_LADDER] == list(range(len(EFFORT_LADDER)))
    assert Effort.NONE.rank < Effort.LOW.rank < Effort.HIGH.rank < Effort.XHIGH.rank


@pytest.mark.parametrize("word", ["ultra", "ULTRA", " ultra "])
def test_ultra_is_refused_as_a_different_axis(word):
    """Not "unknown value" -- a sentence saying which axis it belongs to.

    A caller who asked to think harder and got multi-agent execution has had a
    different cost shape, latency profile and tool surface handed to them.
    """
    with pytest.raises(InvalidConnection) as excinfo:
        plan_reasoning(word, Runtime.OPENAI_SDK, name="c")

    message = str(excinfo.value)
    assert "agentic" in message
    assert "subagents" in message
    assert "multiAgentVersion" in message  # the evidence, named


def test_max_is_refused_because_two_vendors_disagree_about_it():
    with pytest.raises(InvalidConnection) as excinfo:
        plan_reasoning("max", Runtime.ANTHROPIC_SDK, name="c")
    assert "not portable" in str(excinfo.value)


def test_neither_composite_is_in_the_ladder():
    assert "ultra" not in {e.value for e in EFFORT_LADDER}
    assert "max" not in {e.value for e in EFFORT_LADDER}


# --- 2. an exact rung travels in the runtime's own spelling ---------------------


def test_an_exact_rung_reports_the_vendor_spelling():
    plan = plan_reasoning("high", Runtime.OPENAI_SDK, name="c")
    assert plan.disposition is ReasoningDisposition.EXACT
    assert plan.applied is Effort.HIGH
    assert plan.runtime_value == "high"
    assert not plan.adjusted


def test_google_reports_its_own_upper_case_spelling():
    """The owner ask was to report the *model-specific* setting, not the house one."""
    plan = plan_reasoning("medium", Runtime.GOOGLE_API, name="c")
    assert plan.runtime_value == "MEDIUM"
    assert plan.requested is Effort.MEDIUM


# --- 3. a missing rung moves, and says so --------------------------------------


def test_a_rung_below_a_runtimes_floor_is_adjusted_and_reported():
    """claude-agent-sdk's EffortLevel has no 'none' and no 'minimal'."""
    plan = plan_reasoning("none", Runtime.ANTHROPIC_SDK, name="c")
    assert plan.disposition is ReasoningDisposition.ADJUSTED
    assert plan.applied is Effort.LOW
    assert plan.runtime_value == "low"
    assert "moved up" in plan.note
    assert "'none'" in plan.note and "'low'" in plan.note


def test_a_rung_above_a_runtimes_ceiling_is_adjusted_and_reported():
    """google-genai's ThinkingLevel stops at HIGH."""
    plan = plan_reasoning("xhigh", Runtime.GOOGLE_API, name="c")
    assert plan.adjusted
    assert plan.applied is Effort.HIGH
    assert plan.runtime_value == "HIGH"
    assert "moved down" in plan.note


def test_the_note_lists_the_ladder_that_runtime_actually_has():
    plan = plan_reasoning("none", Runtime.ANTHROPIC_SDK, name="c")
    for rung in RUNTIME_EFFORTS[Runtime.ANTHROPIC_SDK]:
        assert repr(rung.value) in plan.note


# --- 4. refusals -----------------------------------------------------------------


def test_a_runtime_with_no_established_control_is_refused():
    """openai-compatible names an API *shape*, so its cells are unverified by
    construction -- including this one."""
    with pytest.raises(InvalidConnection) as excinfo:
        plan_reasoning("high", Runtime.OPENAI_COMPATIBLE, name="metered")
    message = str(excinfo.value)
    assert "metered" in message  # the connection, by name
    assert "setting that does nothing" in message


def test_an_unverified_cell_refuses_like_an_unsupported_one():
    registry = CapabilityRegistry()
    registry.refine(Runtime.OPENAI_SDK, {"reasoning_effort": Support.UNVERIFIED})
    with pytest.raises(InvalidConnection):
        plan_reasoning("high", Runtime.OPENAI_SDK, name="c", registry=registry)


def test_a_word_that_is_not_a_rung_lists_the_ladder():
    with pytest.raises(InvalidConnection) as excinfo:
        plan_reasoning("very-hard", Runtime.OPENAI_SDK, name="c")
    assert "'medium'" in str(excinfo.value)


def test_an_effort_member_is_accepted_as_well_as_its_spelling():
    assert plan_reasoning(Effort.LOW, Runtime.OPENAI_SDK, name="c").applied is Effort.LOW


# --- 5. the vocabularies are the installed SDKs', not invented -------------------


def test_every_runtime_vocabulary_is_a_subset_of_the_house_ladder():
    for runtime, mapping in RUNTIME_EFFORTS.items():
        assert set(mapping) <= set(EFFORT_LADDER), runtime


def test_every_runtime_with_a_supported_cell_has_a_vocabulary():
    """A cell that says 'supported' with nothing to send would be a promise."""
    registry = CapabilityRegistry()
    for runtime in Runtime:
        if registry.support(runtime, "reasoning_effort") is Support.SUPPORTED:
            assert RUNTIME_EFFORTS.get(runtime), runtime


def test_anthropic_api_takes_a_level_as_well_as_a_budget():
    """Recorded because it was got wrong once (2026-09-21).

    ``anthropic`` 0.97.0 has both ``thinking.budget_tokens`` (an integer) and
    ``OutputConfigParam.effort`` (a level). Finding only the first led to this
    runtime being marked unverified; the level is what modelpass carries, and
    sampling_rules had been routing to it since ticket 1.7.
    """
    plan = plan_reasoning("xhigh", Runtime.ANTHROPIC_API, name="c")
    assert plan.disposition is ReasoningDisposition.EXACT
    assert plan.runtime_value == "xhigh"


# --- 6. the connection key is a default for the per-call dial, not a second one --


def _api(runtime=Runtime.OPENAI_API, **kw):
    from modelpass.connections import Connection, CredentialRef
    from modelpass.types import AuthMode

    return Connection(
        name="c",
        runtime=runtime,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:K"),
        model="gpt-5",
        **kw,
    )


def test_a_stated_level_fills_the_per_call_dial():
    from modelpass.bridge import _with_standing_reasoning

    merged = _with_standing_reasoning(_api(reasoning="xhigh"), None)
    assert merged.reasoning_effort == "xhigh"


def test_a_call_that_names_its_own_effort_wins():
    """Same precedence ``model=`` has: the narrower statement is the later one,
    and a standing default that overrode a call could not be turned off for one
    request."""
    from modelpass.bridge import _with_standing_reasoning
    from modelpass.types import Sampling

    merged = _with_standing_reasoning(
        _api(reasoning="xhigh"), Sampling(reasoning_effort="low")
    )
    assert merged.reasoning_effort == "low"


def test_a_connection_that_states_nothing_changes_nothing():
    from modelpass.bridge import _with_standing_reasoning

    assert _with_standing_reasoning(_api(), None) is None


def test_the_standing_default_reaches_the_existing_per_model_routing():
    """The point of merging into Sampling rather than running beside it: the
    per-runtime routing, the drop reporting and the thinking-exclusivity rules
    all apply without being taught again."""
    from modelpass.bridge import _with_standing_reasoning
    from modelpass.sampling_rules import plan_sampling

    merged = _with_standing_reasoning(_api(reasoning="xhigh"), None)
    plan = plan_sampling(merged, Runtime.OPENAI_API, "gpt-5")
    # **Amended 2026-09-22: this asserted 'xhigh' passed through, and the vendor
    # says otherwise.** A live probe -- free, because a rejected request is not
    # billed -- established that `gpt-5` takes 'minimal', 'low', 'medium' and
    # 'high' and refuses 'xhigh'. So the routing now moves the standing level to
    # this model's ceiling and says so, which is the behaviour this test was
    # written to check; only the expected value was wrong, because the table it
    # trusted claimed six rungs for every gpt-5* model.
    assert plan.applied["reasoning_effort"] == "high"
    assert any("xhigh" in note and "high" in note for note in plan.notes)


def test_there_is_one_vocabulary_not_two():
    from modelpass.types import REASONING_EFFORTS

    assert tuple(e.value for e in EFFORT_LADDER) == tuple(REASONING_EFFORTS)


def test_the_connection_key_refuses_what_the_ladder_refuses():
    from modelpass.errors import InvalidConnection as IC

    with pytest.raises(IC):
        _api(reasoning="ultra")
    with pytest.raises(IC):
        _api(reasoning="max")


def test_a_runtime_without_the_control_refuses_the_key_at_build():
    from modelpass.connections import Connection, CredentialRef
    from modelpass.errors import InvalidConnection as IC
    from modelpass.types import AuthMode

    with pytest.raises(IC):
        Connection(
            name="c",
            runtime=Runtime.OPENAI_COMPATIBLE,
            auth_mode=AuthMode.API_KEY,
            credential_ref=CredentialRef.parse("env:K"),
            model="m",
            base_url="http://localhost:1234/v1",
            reasoning="high",
        )


# --- 7. which rungs a *model* has, behind an interface ---------------------------


def test_an_unrecorded_model_degrades_to_the_runtime_ladder():
    """Not to a guess, and not to nothing: a model nobody has recorded is not a
    model with no rungs."""
    from modelpass.sampling_rules import model_efforts

    assert model_efforts(Runtime.OPENAI_API, "gpt-6-unheard-of") == tuple(
        e.value for e in EFFORT_LADDER
    )


def test_a_runtime_with_no_effort_control_answers_empty():
    from modelpass.sampling_rules import model_efforts

    assert model_efforts(Runtime.OPENAI_COMPATIBLE, "anything") == ()


def test_a_recorded_model_narrows_the_ladder():
    """openai 2.32.0's Reasoning.effort docstring: gpt-5-pro "defaults to (and
    only supports) high reasoning effort"."""
    from modelpass.sampling_rules import model_efforts

    assert model_efforts(Runtime.OPENAI_API, "gpt-5-pro") == ("high",)
    # and the general family is untouched
    assert len(model_efforts(Runtime.OPENAI_API, "gpt-5")) > 1


def test_a_rung_the_model_lacks_is_moved_and_reported():
    from modelpass.sampling_rules import plan_sampling
    from modelpass.types import Sampling

    plan = plan_sampling(
        Sampling(reasoning_effort="low"), Runtime.OPENAI_API, "gpt-5-pro"
    )
    assert plan.applied["reasoning_effort"] == "high"
    assert any("not one this model takes" in n for n in plan.notes)


def test_a_model_family_resolves_to_its_own_rules_not_a_shorter_prefix():
    """Family tokens nest. First-match made the answer depend on declaration
    order, so a specific family silently inherited a general one's rules."""
    from modelpass.sampling_rules import _TABLE, _matches

    for _runtime, (_base, refinements) in _TABLE.items():
        tokens = tuple(token for token, _ in refinements)
        for token in tokens:
            assert _matches(token, tokens) == token, token


# --- 8. each runtime carries it exactly once ------------------------------------


def _sub(runtime):
    from modelpass.connections import Connection, CredentialRef
    from modelpass.types import AuthMode

    return Connection(
        name="c",
        runtime=runtime,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        reasoning="high",
    )


@pytest.mark.parametrize("runtime", [Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK])
def test_an_agent_runtime_is_not_told_through_sampling(runtime):
    """It would be reported dropped while the adapter applied it anyway.

    Both agent runtimes report ``sampling_controls`` unsupported and drop every
    field, so merging the standing default into Sampling there put a
    "reasoning_effort not supported, dropped" note on a receipt for a run where
    ``ClaudeAgentOptions.effort`` / ``TurnStartParams.effort`` carried it. That
    is worse than silence: it tells the caller the opposite of what happened.
    """
    from modelpass.bridge import _with_standing_reasoning
    from modelpass.reasoning import stated_reasoning

    assert _with_standing_reasoning(_sub(runtime), None) is None
    # ...and the adapter reads it directly instead, so it is not lost.
    assert stated_reasoning(_sub(runtime)).runtime_value == "high"


def test_an_api_runtime_is_told_through_sampling():
    from modelpass.bridge import _with_standing_reasoning

    merged = _with_standing_reasoning(_api(reasoning="high"), None)
    assert merged is not None and merged.reasoning_effort == "high"


def test_no_runtime_is_told_twice():
    """Exactly one carrier per runtime: sampling or the adapter, never both."""
    from modelpass.bridge import _with_standing_reasoning
    from modelpass.sampling_rules import rules_for

    for runtime in (Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK, Runtime.OPENAI_API):
        conn = (
            _api(runtime=runtime, reasoning="high")
            if runtime is Runtime.OPENAI_API
            else _sub(runtime)
        )
        via_sampling = _with_standing_reasoning(conn, None) is not None
        carried = "reasoning_effort" in rules_for(runtime, conn.model or "").accepted
        assert via_sampling is carried, runtime


# --- 9. what the runtime says it used -------------------------------------------


def test_the_receipt_carries_the_level_on_an_agent_runtime(tmp_path):
    """The only place the answer exists on anthropic-sdk.

    anthropic 0.97.0's Message response has no effort field and the effort
    documentation describes none, so a level sent to that runtime is never
    echoed. If the receipt does not say it, nothing does.
    """
    from modelpass.testing import fake_bridge
    from modelpass.types import ReceiptEvent, TextDeltaEvent

    bridge, *_ = fake_bridge(
        connections=[_sub(Runtime.ANTHROPIC_SDK)],
        script=[TextDeltaEvent(text="hi")],
        home=tmp_path / "home",
    )
    receipts = [e.receipt for e in bridge.chat(connection="c", message="x")
                if isinstance(e, ReceiptEvent)]
    assert receipts
    assert receipts[0].reasoning_requested == "high"
    assert receipts[0].reasoning_applied == "high"
    assert receipts[0].reasoning_value == "high"


def test_the_receipt_shows_an_adjustment_rather_than_hiding_it(tmp_path):
    from modelpass.connections import Connection, CredentialRef
    from modelpass.testing import fake_bridge
    from modelpass.types import AuthMode, ReceiptEvent, TextDeltaEvent

    conn = Connection(
        name="c",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        reasoning="none",
    )
    bridge, *_ = fake_bridge(
        connections=[conn], script=[TextDeltaEvent(text="hi")], home=tmp_path / "home"
    )
    receipt = next(
        e.receipt for e in bridge.chat(connection="c", message="x")
        if isinstance(e, ReceiptEvent)
    )
    assert receipt.reasoning_requested == "none"
    assert receipt.reasoning_applied == "low"  # this runtime's floor


def test_a_connection_that_states_nothing_puts_nothing_on_the_receipt(tmp_path):
    """A line reading "reasoning: not requested" on every receipt is noise."""
    from modelpass.connections import Connection, CredentialRef
    from modelpass.testing import fake_bridge
    from modelpass.types import AuthMode, ReceiptEvent, TextDeltaEvent

    conn = Connection(
        name="c",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    bridge, *_ = fake_bridge(
        connections=[conn], script=[TextDeltaEvent(text="hi")], home=tmp_path / "home"
    )
    receipt = next(
        e.receipt for e in bridge.chat(connection="c", message="x")
        if isinstance(e, ReceiptEvent)
    )
    assert receipt.reasoning_requested is None
    assert receipt.reasoning_value is None


def test_codex_echoes_the_level_and_a_disagreement_is_reported():
    """The one runtime that answers back. Read off the live capture, not a mock."""
    import json
    from pathlib import Path

    from modelpass.adapters.openai import reasoning_echo_note, thread_reasoning_effort

    capture = Path(__file__).parent / "fixtures/appserver/live-capture-2026-08-31.jsonl"
    started = next(
        json.loads(line)["payload"]
        for line in capture.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("_kind") == "thread_start_response"
    )
    echoed = thread_reasoning_effort(started)
    assert echoed == "medium"
    # agreement is silent; disagreement is not
    assert reasoning_echo_note("medium", echoed) is None
    assert "reports running at 'medium'" in reasoning_echo_note("high", echoed)
    # and nothing to compare is not a finding
    assert reasoning_echo_note(None, echoed) is None
    assert reasoning_echo_note("high", None) is None


def test_an_absent_echo_is_not_an_error():
    from modelpass.adapters.openai import thread_reasoning_effort

    assert thread_reasoning_effort({"thread": {"id": "t1"}}) is None
    assert thread_reasoning_effort({}) is None


# --- 10. the two surfaces do not contradict each other --------------------------


def test_dropping_reasoning_from_sampling_points_at_the_key_that_works():
    """Reported by a consumer, 2026-09-22.

    They read "reasoning_effort not supported on anthropic-sdk, dropped",
    concluded effort was unreachable on a subscription connection, and planned
    to buy an API key to run an effort experiment. Effort is reachable there --
    through the connection's ``reasoning`` key, not through ``Sampling``. The
    note was true of the path and false of the runtime.
    """
    from modelpass.sampling_rules import plan_sampling
    from modelpass.types import Sampling

    plan = plan_sampling(
        Sampling(reasoning_effort="medium"), Runtime.ANTHROPIC_SDK, "claude-sonnet-5"
    )
    note = next(n for n in plan.notes if "reasoning_effort" in n)
    assert "'reasoning' key" in note
    assert "not supported" not in note


def test_a_field_with_no_other_route_still_says_plainly_that_it_is_dropped():
    """temperature really is unreachable on this runtime; do not soften that."""
    from modelpass.sampling_rules import plan_sampling
    from modelpass.types import Sampling

    plan = plan_sampling(
        Sampling(temperature=0.2), Runtime.ANTHROPIC_SDK, "claude-sonnet-5"
    )
    assert any("temperature not supported" in n for n in plan.notes)


def test_the_rules_source_no_longer_claims_effort_is_unreachable():
    """The dated-citation convention worked; the fact under it moved.

    The source string cited a 2026-09-13 grep proving no sampling parameter
    reached the agent adapters. True then. ClaudeAgentOptions.effort and
    TurnStartParams.effort were wired on 2026-09-21 and the string did not move
    with them, so it read as authoritative and was stale -- the exact failure
    mode dating a claim is supposed to prevent.
    """
    from modelpass.sampling_rules import rules_for

    source = rules_for(Runtime.ANTHROPIC_SDK, "claude-sonnet-5").source
    assert "reasoning" in source.lower()
    assert "not dropped" in source.lower() or "NOT dropped" in source


def test_the_capability_cell_and_the_connection_key_agree():
    """The registry says effort is supported on the agent runtimes. It is --
    via the connection key. That is what the cell means, and this holds the two
    together so the next reader does not have to guess which to believe."""
    from modelpass.capabilities import DEFAULT_REGISTRY, Capability, Support
    from modelpass.reasoning import stated_reasoning

    for runtime in (Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK):
        assert (
            DEFAULT_REGISTRY.support(runtime, Capability.REASONING_EFFORT)
            is Support.SUPPORTED
        )
        assert stated_reasoning(_sub(runtime)) is not None


def _fold_for(connection):
    """A fold wired the way the pump wires one, for terminal-stamp assertions."""
    from modelpass._fold import RunFold
    from modelpass.connections import Guards
    from modelpass.guards import GuardTracker
    from modelpass.preflight import Receipt
    from modelpass.types import AuthMode

    guards = Guards()
    receipt = Receipt(
        connection=connection.name,
        runtime=connection.runtime,
        requested_auth_mode=AuthMode.SUBSCRIPTION,
        detected_auth_mode=AuthMode.SUBSCRIPTION,
        reasoning_value="high",
    )
    return RunFold(
        connection=connection,
        receipt=receipt,
        guards=guards,
        tracker=GuardTracker(guards, connection.name),
        deadline=None,
    )


# --- 11. what the run actually spent thinking -----------------------------------


def test_a_reasoning_count_is_a_subset_of_output_and_not_added_to_the_total():
    """The invariant every adapter's mapping owes this field.

    ``cached_input_tokens`` is a peer of ``input_tokens`` and counts toward the
    total; ``reasoning_output_tokens`` is a *part of* ``output_tokens`` and must
    not. Getting the two conventions the same way round double-counts every
    thinking run, which is a billing-shaped error rather than a cosmetic one.
    """
    from modelpass.types import TokenUsage

    usage = TokenUsage(input_tokens=2, output_tokens=996, reasoning_output_tokens=993)
    assert usage.total_tokens == 998
    assert usage.reasoning_output_tokens <= usage.output_tokens


def test_folding_two_silences_does_not_manufacture_a_zero():
    """``None`` + ``None`` is ``None``: a fold must not invent a measurement."""
    from modelpass.types import TokenUsage

    silent = TokenUsage(output_tokens=5)
    reported = TokenUsage(reasoning_output_tokens=7)
    assert (silent + silent).reasoning_output_tokens is None
    assert (silent + reported).reasoning_output_tokens == 7
    # at_least is the run_total fold: a number beats a silence in both
    # directions, for the reason the other four components take a max.
    assert silent.at_least(reported).reasoning_output_tokens == 7
    assert reported.at_least(silent).reasoning_output_tokens == 7


def test_zero_thinking_and_no_report_are_different_answers():
    """The distinction the whole field exists for, at every layer that has it."""
    from modelpass.adapters.anthropic import token_usage
    from modelpass.reasoning import ReasoningMetric, reasoning_metric

    thought_none = token_usage(
        {"output_tokens": 4, "output_tokens_details": {"thinking_tokens": 0}}
    )
    said_nothing = token_usage({"output_tokens": 4})
    assert thought_none.reasoning_output_tokens == 0
    assert said_nothing.reasoning_output_tokens is None
    assert reasoning_metric(Runtime.ANTHROPIC_SDK, 0) is ReasoningMetric.REPORTED
    assert reasoning_metric(Runtime.ANTHROPIC_SDK, None) is ReasoningMetric.UNREPORTED
    # ...and the runtime whose vendor has no such field at all is a third thing:
    # anthropic 0.97.0's Usage has no details object, so no run of it will ever
    # carry one and a consumer retrying to get a count is spending for nothing.
    assert reasoning_metric(Runtime.ANTHROPIC_API, None) is ReasoningMetric.UNAVAILABLE


def test_every_runtime_with_an_effort_ladder_declares_its_metric_or_lacks_one():
    """No runtime may quietly have neither answer.

    A runtime that accepts an effort level must either report what the thinking
    cost or be named as one that cannot. Otherwise a caller can set the dial and
    never find out whether it did anything -- which is the reporting hole this
    whole ticket started from.
    """
    from modelpass.reasoning import REASONING_METRIC_FIELDS, RUNTIME_EFFORTS

    for runtime in RUNTIME_EFFORTS:
        assert runtime in REASONING_METRIC_FIELDS or runtime is Runtime.ANTHROPIC_API


def _thinking_script():
    from modelpass.types import TextDeltaEvent, TokenUsage, UsageEvent

    return [
        TextDeltaEvent(text="hi"),
        UsageEvent(usage=TokenUsage(output_tokens=996, reasoning_output_tokens=993)),
    ]


def test_the_terminal_carries_what_was_sent_and_how_to_read_the_count(tmp_path):
    """Three answers in one object, so a batch can be read without a join."""
    from modelpass.testing import fake_bridge
    from modelpass.types import TerminalEvent

    bridge, *_ = fake_bridge(
        connections=[_sub(Runtime.ANTHROPIC_SDK)],
        script=_thinking_script(),
        home=tmp_path / "home",
    )
    terminal = next(
        e for e in bridge.chat(connection="c", message="x")
        if isinstance(e, TerminalEvent)
    )
    assert terminal.reasoning_value == "high"
    assert terminal.reasoning_metric == "reported"
    assert terminal.usage.reasoning_output_tokens == 993
    # No vendor on this runtime echoes the level back -- driven 2026-09-22 --
    # so the honest value is None rather than a repeat of what was sent.
    assert terminal.reasoning_echo is None


def test_a_runtime_that_reports_no_count_says_which_kind_of_nothing(tmp_path):
    from modelpass.testing import fake_bridge
    from modelpass.types import TerminalEvent, TextDeltaEvent

    bridge, *_ = fake_bridge(
        connections=[_sub(Runtime.ANTHROPIC_SDK)],
        script=[TextDeltaEvent(text="hi")],
        home=tmp_path / "home",
    )
    terminal = next(
        e for e in bridge.chat(connection="c", message="x")
        if isinstance(e, TerminalEvent)
    )
    assert terminal.usage.reasoning_output_tokens is None
    assert terminal.reasoning_metric == "unreported"


def test_the_run_log_keeps_the_dial_and_its_cost(tmp_path):
    """The durable half: a stored history can be asked what effort bought."""
    from modelpass.testing import fake_bridge

    bridge, *_ = fake_bridge(
        connections=[_sub(Runtime.ANTHROPIC_SDK)],
        script=_thinking_script(),
        home=tmp_path / "home",
    )
    bridge.ask("c", "x")
    (record,) = bridge.run_log.read(1)
    assert record["reasoning_value"] == "high"
    assert record["reasoning_output_tokens"] == 993
    assert record["reasoning_metric"] == "reported"
    assert record["reasoning_echo"] is None
    # The count is inside output_tokens, so the stored total must not have grown
    # by it: the ledger and the receipt agree, or the ledger is wrong.
    assert record["total_tokens"] == 996


def test_an_old_log_line_reads_back_as_unmeasured_rather_than_as_zero(tmp_path):
    """Additive keys, coerced toward None -- unlike the breakpoint pair.

    A line written before this existed says nothing about what the run reasoned.
    Reading it back as 0 would turn every historical run into evidence that the
    model never thought, which is exactly the inference this field forbids.
    """
    import json

    from modelpass.runlog import RunLog

    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-09-01T00:00:00+00:00",
                "connection": "c",
                "runtime": "anthropic-sdk",
                "auth_mode": "subscription",
                "status": "ok",
                "output_tokens": 996,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (record,) = RunLog(tmp_path).read(1)
    assert record["reasoning_output_tokens"] is None
    assert record["reasoning_metric"] is None


def test_the_answer_a_one_shot_caller_holds_carries_all_three(tmp_path):
    from modelpass.testing import fake_bridge

    bridge, *_ = fake_bridge(
        connections=[_sub(Runtime.ANTHROPIC_SDK)],
        script=_thinking_script(),
        home=tmp_path / "home",
    )
    answer = bridge.ask("c", "x")
    assert answer.reasoning_value == "high"
    assert answer.reasoning_metric == "reported"
    assert answer.usage is not None
    assert answer.usage.reasoning_output_tokens == 993


def test_codexs_echo_becomes_an_event_the_fold_can_stamp():
    """The one runtime that answers back, wired rather than merely available.

    ``thread/start`` reports the thread's own ``reasoningEffort``. The adapter
    emits it as a vendor event; the fold observes it the way it observes the
    allowance -- on the way past, never intercepted -- and stamps it on the
    terminal. A server that silently ignored the level now says so somewhere.
    """
    from modelpass.adapters.openai import _reasoning_echo_events
    from modelpass.types import VendorEvent

    class _Request:
        connection = _sub(Runtime.OPENAI_SDK)

    (disagreed,) = _reasoning_echo_events(
        _Request(), {"thread": {"id": "t1", "reasoningEffort": "medium"}}
    )
    assert isinstance(disagreed, VendorEvent)
    assert disagreed.name == "reasoning_echo"
    assert disagreed.data["sent"] == "high"
    assert disagreed.data["echoed"] == "medium"
    assert "medium" in disagreed.data["note"]
    # Agreement is silent: a note on every run is noise, and the two fields
    # already say it.
    (agreed,) = _reasoning_echo_events(
        _Request(), {"thread": {"id": "t1", "reasoningEffort": "high"}}
    )
    assert "note" not in agreed.data
    # No echo at all is not an error -- older servers, and the exec transport,
    # answer nothing.
    assert _reasoning_echo_events(_Request(), {"thread": {"id": "t1"}}) == ()


def test_the_echo_a_fold_observed_reaches_the_terminal():
    """The other half of the wiring: observed, then stamped."""
    from modelpass.types import Runtime as _Runtime
    from modelpass.types import VendorEvent

    fold = _fold_for(_sub(_Runtime.OPENAI_SDK))
    fold.feed(
        VendorEvent(
            runtime=_Runtime.OPENAI_SDK,
            name="reasoning_echo",
            data={"sent": "high", "echoed": "medium"},
        )
    )
    terminal = fold.finish()
    assert terminal.reasoning_value == "high"
    assert terminal.reasoning_echo == "medium"


# --- 5. the two routes that once sent nothing ------------------------------------


def test_the_exec_transport_carries_the_standing_level():
    """The gap that made a receipt name a level no process ever saw.

    `_with_reasoning_disclosure` is table-driven and stamps every runtime, so a
    run on `options={'transport': 'exec'}` reported `reasoning_value` while the
    argv carried nothing at all. `model_reasoning_effort` is the vendor's own
    config key, reached through the `-c` layer `codex exec --help` documents.
    """
    from modelpass.adapters.openai import effort_config_args

    assert effort_config_args(_sub(Runtime.OPENAI_SDK)) == [
        "-c",
        "model_reasoning_effort=high",
    ]


def test_a_connection_that_states_nothing_adds_no_config_override():
    """Silence stays silence: modelpass does not pick a level for anybody."""
    from modelpass.adapters.openai import effort_config_args
    from modelpass.connections import Connection, CredentialRef
    from modelpass.types import AuthMode

    quiet = Connection(
        name="c",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    assert effort_config_args(quiet) == []


def test_a_config_override_is_not_a_per_turn_field():
    """Why exec gets the standing level and still refuses a per-turn one.

    The two are different things: `-c` sets the thread's default before the
    process starts, and `TurnStartParams.effort` changes it mid-conversation.
    Translating a per-turn request into a config override would answer a
    question nobody asked and report it as the one they did.
    """
    from modelpass.adapters.openai import OpenAIAdapter
    from modelpass.capabilities import Capability, Support

    assert (
        OpenAIAdapter._EXEC_SUPPORT[Capability.REASONING_EFFORT_PER_TURN]
        is Support.UNSUPPORTED
    )


def test_the_per_turn_cell_answers_per_transport_not_per_runtime():
    """The reason this moved out of a private set in sessions.py.

    `openai-sdk` supports per-turn effort on its default transport and does not
    on `exec`. A runtime-level lookup cannot say that; the adapter's
    `support_for` hook can, and it is what the session refusal consults.
    """
    from modelpass.adapters.openai import OpenAIAdapter
    from modelpass.capabilities import DEFAULT_REGISTRY, Capability, Support

    adapter = OpenAIAdapter()
    assert (
        DEFAULT_REGISTRY.support(Runtime.OPENAI_SDK, Capability.REASONING_EFFORT_PER_TURN)
        is Support.SUPPORTED
    )
    assert (
        adapter.support_for(Capability.REASONING_EFFORT_PER_TURN, {"transport": "exec"})
        is Support.UNSUPPORTED
    )
    assert adapter.support_for(Capability.REASONING_EFFORT_PER_TURN, {}) is None


def test_a_session_thread_start_echo_reaches_the_stream():
    """The echo the *session* path never emitted.

    `thread/start` answers with the thread's own `reasoningEffort`, and the
    stateless run has compared the two since the echo was wired. A session --
    the object the whole continuity question is about -- sent a level and never
    looked at the answer. It is held on the handle because `thread/start`
    happens inside a non-generator, then drained onto the first turn's stream.
    """
    from modelpass.adapters.openai import _reasoning_echo_events

    class _Request:
        connection = _sub(Runtime.OPENAI_SDK)

    (event,) = _reasoning_echo_events(
        _Request(), {"thread": {"id": "t1", "reasoningEffort": "low"}}
    )
    assert event.name == "reasoning_echo"
    assert event.data["sent"] == "high"
    assert event.data["echoed"] == "low"
    assert "low" in event.data["note"]


# --- 6. what the vendors said when asked directly --------------------------------


def test_the_openai_api_echoes_the_level_it_was_sent():
    """Driven 2026-09-22: `Response.reasoning.effort` is populated, not just typed.

    `openai` 2.32.0 declares the field; a declaration is not a value. A live call
    to `gpt-5.4-mini` with `reasoning={'effort': 'low'}` answered
    `Reasoning(effort='low', ..., context='current_turn', mode='standard')`, and
    `'high'` answered `'high'` -- it tracks the request rather than repeating a
    constant, which is what makes an echo worth reading.
    """
    from modelpass.adapters.openai_api import (
        reasoning_echo_events,
        response_reasoning_echo,
    )

    class _Reasoning:
        effort = "low"

    class _Response:
        reasoning = _Reasoning()

    class _Sampling:
        reasoning_effort = "high"

    class _Request:
        sampling = _Sampling()

    assert response_reasoning_echo(_Response()) == "low"
    (event,) = reasoning_echo_events(_Request(), _Response(), Runtime.OPENAI_API)
    assert event.name == "reasoning_echo"
    assert event.data == {
        "sent": "high",
        "echoed": "low",
        "note": event.data["note"],
    }
    assert "low" in event.data["note"]


def test_an_agreeing_echo_says_nothing_and_a_missing_one_is_not_an_error():
    from modelpass.adapters.openai_api import reasoning_echo_events

    class _Response:
        reasoning = type("R", (), {"effort": "high"})()

    class _Request:
        sampling = type("S", (), {"reasoning_effort": "high"})()

    (agreed,) = reasoning_echo_events(_Request(), _Response(), Runtime.OPENAI_API)
    assert "note" not in agreed.data
    # A model with no reasoning surface answers without the field.
    assert reasoning_echo_events(_Request(), object(), Runtime.OPENAI_API) == []


def test_two_runtimes_now_echo_and_anthropic_still_does_not():
    """The shape of the evidence, as of 2026-09-22.

    `openai-api` and `openai-sdk` both answer with the level they are running at;
    Anthropic answers with nothing on either of its runtimes, which was driven
    rather than inferred. That asymmetry is why the receipt exists.
    """
    from modelpass.adapters.openai import thread_reasoning_effort
    from modelpass.adapters.openai_api import response_reasoning_echo

    assert thread_reasoning_effort({"thread": {"reasoningEffort": "high"}}) == "high"
    echoing = type("R", (), {"reasoning": type("X", (), {"effort": "high"})()})()
    assert response_reasoning_echo(echoing) == "high"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.4", ("none", "low", "medium", "high", "xhigh")),
        ("gpt-5.4-mini", ("none", "low", "medium", "high", "xhigh")),
        ("gpt-5.4-nano", ("none", "low", "medium", "high", "xhigh")),
        ("gpt-5.2", ("none", "low", "medium", "high", "xhigh")),
        ("gpt-5.1", ("none", "low", "medium", "high")),
        ("gpt-5", ("minimal", "low", "medium", "high")),
        ("gpt-5-nano", ("minimal", "low", "medium", "high")),
        ("gpt-5-pro", ("high",)),
    ],
)
def test_the_per_model_ladders_are_the_ones_the_vendor_accepts(model, expected):
    """Probed live on 2026-09-22, free, because a rejected request is not billed.

    The table previously claimed all six rungs for every one of these with
    `model_known=True` -- wrong in both directions. A caller setting 'minimal' on
    `gpt-5.4-mini` got a vendor 400 modelpass had said would not happen, and
    'none' on `gpt-5` the same.
    """
    from modelpass.sampling_rules import model_efforts

    assert model_efforts(Runtime.OPENAI_API, model) == expected


def test_the_gpt5_ladder_is_the_inverse_of_its_successors():
    """Why one family row could not have served both.

    `gpt-5` has 'minimal' and lacks 'none' and 'xhigh'; `gpt-5.4` has 'none' and
    'xhigh' and lacks 'minimal'. A single set covering both would be wrong for
    whichever model it was not written for.
    """
    from modelpass.sampling_rules import model_efforts

    five = set(model_efforts(Runtime.OPENAI_API, "gpt-5"))
    later = set(model_efforts(Runtime.OPENAI_API, "gpt-5.4"))
    assert "minimal" in five and "minimal" not in later
    assert {"none", "xhigh"} <= later and not {"none", "xhigh"} & five


def test_a_rung_the_model_lacks_is_moved_here_rather_than_refused_by_the_vendor():
    """The point of recording ladders at all: the 400 never happens.

    It is reported, not silent -- and the direction of the move is worth a look.
    'minimal' sits between 'none' and 'low', and the tie breaks *downward*, so a
    caller asking for minimal thinking on gpt-5.4-mini gets 'none', which
    disables reasoning rather than reducing it.
    """
    from modelpass.sampling_rules import plan_sampling
    from modelpass.types import Sampling

    plan = plan_sampling(
        Sampling(reasoning_effort="minimal"), Runtime.OPENAI_API, "gpt-5.4-mini"
    )
    assert plan.applied["reasoning_effort"] == "none"
    assert any("moved down to 'none'" in note for note in plan.notes)


def test_max_is_rejected_by_every_openai_model_probed():
    """Beside modelpass's own refusal of the word, the vendor refuses it too.

    Worth recording where the `max` question is decided: on `openai-api` there is
    nothing to allow. The case for `max` is Anthropic's alone.
    """
    from modelpass.sampling_rules import model_efforts

    for model in ("gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-pro"):
        assert "max" not in model_efforts(Runtime.OPENAI_API, model)
