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
    """anthropic-api takes an integer budget, so no level is carried to it."""
    with pytest.raises(InvalidConnection) as excinfo:
        plan_reasoning("high", Runtime.ANTHROPIC_API, name="metered")
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
