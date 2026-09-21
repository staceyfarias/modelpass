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
    assert plan.applied["reasoning_effort"] == "xhigh"


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
