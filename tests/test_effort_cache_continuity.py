"""Three questions about effort and cache, held apart.

This file exists because collapsing them is easy, tempting, and wrong:

1. *Does this runtime take an effort setting?* -- `tests/test_reasoning.py`.
2. *Could changing it mid-conversation keep the cached prefix?* -- a claim about
   a vendor and a model, knowable before any run. Here.
3. *Did this turn actually reuse the prefix?* -- telemetry, knowable only after.
   Also here, and never derived from (2).

The cells below are vendor documentation read on 2026-09-22, not inference. What
the tests hold is that modelpass says nothing it cannot cite, in either
direction: a `supported` it did not read, or an `unsupported` about a model the
vendor's negative never covered, are the same failure pointing opposite ways.
"""

from __future__ import annotations

import pytest

from modelpass.prompt_cache import (
    ContinuityMechanism,
    EffortCacheContinuity,
    effort_cache_continuity,
    observed_effort_continuity,
)
from modelpass.runtimes import Runtime
from modelpass.types import TokenUsage

# --- 1. the capability is model-specific, not runtime-specific -------------------


@pytest.mark.parametrize(
    "model",
    ["claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5"],
)
def test_the_three_documented_anthropic_models_are_supported(model):
    """The vendor's sentence, and nothing wider than it.

    "On Claude Fable 5.1, Claude Mythos 5.1, and Claude Opus 5, use a
    per-message effort change, which keeps the prompt cache."
    """
    verdict = effort_cache_continuity(Runtime.ANTHROPIC_API, model)
    assert verdict.capability is EffortCacheContinuity.SUPPORTED
    assert verdict.mechanism is ContinuityMechanism.ANTHROPIC_PER_MESSAGE_EFFORT


def test_a_model_the_vendor_names_as_lacking_it_is_unsupported():
    """Claude Fable 5 is named in the 400 the effort page quotes, by the vendor.

    This is the one direction `UNSUPPORTED` is allowed: documented, not inferred
    from the absence of a `SUPPORTED` row.
    """
    verdict = effort_cache_continuity(Runtime.ANTHROPIC_API, "claude-fable-5")
    assert verdict.capability is EffortCacheContinuity.UNSUPPORTED
    assert verdict.mechanism is None


def test_one_runtime_gives_three_different_answers():
    """The reason this is keyed by model and a runtime-level cell would be wrong.

    Same vendor, same runtime, same API: supported, unsupported, and unknown.
    """
    answers = {
        effort_cache_continuity(Runtime.ANTHROPIC_API, model).capability
        for model in ("claude-opus-5", "claude-sonnet-4-6", "claude-haiku-4-5")
    }
    assert answers == {
        EffortCacheContinuity.SUPPORTED,
        EffortCacheContinuity.UNSUPPORTED,
        EffortCacheContinuity.UNKNOWN,
    }


# --- 2. the conservative rule, in both directions --------------------------------


def test_an_unreleased_point_release_inherits_nothing():
    """The failure mode a family matcher would have shipped.

    `sampling_rules._matches` takes the longest family prefix on a separator
    boundary, which is right for temperature ceilings and wrong here: under it,
    an unreleased `claude-opus-5-1` would inherit Opus 5's cache guarantee on
    the strength of its name. A guarantee is not a family property.
    """
    assert (
        effort_cache_continuity(Runtime.ANTHROPIC_API, "claude-opus-5-1").capability
        is EffortCacheContinuity.UNKNOWN
    )


def test_a_dated_build_of_the_same_model_does_inherit():
    """The one suffix that names the same model rather than a different one."""
    assert (
        effort_cache_continuity(
            Runtime.ANTHROPIC_API, "claude-opus-5-20260115"
        ).capability
        is EffortCacheContinuity.SUPPORTED
    )


def test_an_unheard_of_pair_is_unknown_rather_than_either_answer():
    for runtime, model in (
        (Runtime.GOOGLE_API, "gemini-3-pro"),
        (Runtime.OPENAI_COMPATIBLE, "llama-4"),
        (Runtime.OPENAI_API, "gpt-5.4"),
        (Runtime.OPENAI_SDK, "gpt-5.1-codex"),
    ):
        verdict = effort_cache_continuity(runtime, model)
        assert verdict.capability is EffortCacheContinuity.UNKNOWN, (runtime, model)
        assert verdict.mechanism is None
        assert not verdict.claimed


def test_every_cell_carries_its_evidence():
    """A capability with no citation is the thing this repository does not ship."""
    for runtime in Runtime:
        for model in ("claude-opus-5", "gpt-6-astra", "whatever-3"):
            assert effort_cache_continuity(runtime, model).detail


# --- 3. the vendor's claim and modelpass's reach are two fields ------------------


def test_a_supported_capability_does_not_imply_modelpass_can_use_it():
    """The inverse of the stale-string failure of 2026-09-22, and worse.

    Anthropic ships per-message effort; `anthropic` 0.97.0 types message roles
    as user|assistant with no per-message `output_config`, so modelpass cannot
    send it. A receipt reporting only the vendor's half would have a consumer
    design a long cached session around an effort change that never happens.
    """
    verdict = effort_cache_continuity(Runtime.ANTHROPIC_API, "claude-opus-5")
    assert verdict.capability is EffortCacheContinuity.SUPPORTED
    assert verdict.reachable is False


def test_codex_is_experimental_rather_than_supported():
    """Protocol-typed, vendor-silent: the exact case `EXPERIMENTAL` is for.

    openai-codex 0.154.0 types `ConfigurationUpdateResponseItem`, but nothing
    states the cache behaviour, `model/list` carries no
    `supports_reasoning_effort_updates`, and the installed codex.exe 0.151.0
    contains the string `configuration_update` zero times. Enough to try behind
    a flag; not enough to promise.
    """
    verdict = effort_cache_continuity(Runtime.OPENAI_SDK, "gpt-6-astra")
    assert verdict.capability is EffortCacheContinuity.EXPERIMENTAL
    assert verdict.mechanism is ContinuityMechanism.CODEX_CONFIGURATION_UPDATE
    assert verdict.reachable is False


def test_the_two_configuration_update_mechanisms_are_not_one_mechanism():
    """Same item shape, different transports, different evidence.

    The Responses API's is vendor-documented as cache-preserving; Codex's is
    inferred from a type. Sharing a mechanism name would have merged a
    documented guarantee with a guess.
    """
    direct = effort_cache_continuity(Runtime.OPENAI_API, "gpt-6-astra")
    codex = effort_cache_continuity(Runtime.OPENAI_SDK, "gpt-6-astra")
    assert direct.mechanism is ContinuityMechanism.OPENAI_CONFIGURATION_UPDATE
    assert codex.mechanism is ContinuityMechanism.CODEX_CONFIGURATION_UPDATE
    assert direct.capability is not codex.capability


def test_the_agent_runtime_has_no_transition_to_characterize():
    """Not `unsupported`: the question does not arise until a mechanism does.

    claude-agent-sdk 0.2.148 takes effort at session start; ClaudeSDKClient
    offers set_model and set_permission_mode and no set_effort.
    """
    verdict = effort_cache_continuity(Runtime.ANTHROPIC_SDK, "claude-opus-5")
    assert verdict.capability is EffortCacheContinuity.UNKNOWN
    assert "set_effort" in verdict.detail


# --- 4. telemetry says what happened, and only that ------------------------------


def test_a_turn_that_changed_nothing_is_not_a_measurement():
    """`None`, never `False`. Most runs change no effort, and a `False` on each
    of them would read as a cache break that never happened."""
    assert (
        observed_effort_continuity(
            effort_changed=False, cached_input_tokens=48736, prompt_tokens=50281
        )
        is None
    )


def test_the_preserved_shape_and_the_broken_shape():
    """The two telemetry signatures, from the ticket's own worked example."""
    preserved = TokenUsage(input_tokens=1545, cached_input_tokens=48736)
    broken = TokenUsage(input_tokens=2000, cache_write_tokens=48000)
    assert (
        observed_effort_continuity(
            effort_changed=True,
            cached_input_tokens=preserved.cached_input_tokens,
            prompt_tokens=preserved.prompt_tokens,
        )
        is True
    )
    assert (
        observed_effort_continuity(
            effort_changed=True,
            cached_input_tokens=broken.cached_input_tokens,
            prompt_tokens=broken.prompt_tokens,
        )
        is False
    )


def test_a_growing_conversation_does_not_have_to_match_exactly():
    """The assertion the ticket asks for: a majority of the prefix, not equality.

    Each turn adds tokens, so the cached fraction after a change is never the
    fraction before it. Requiring equality would assert something no vendor
    promises.
    """
    # 48,736 cached of a prompt that has grown by 1,500 fresh tokens.
    assert observed_effort_continuity(
        effort_changed=True, cached_input_tokens=48736, prompt_tokens=50281
    )


def test_missing_telemetry_is_not_a_broken_cache():
    for cached, prompt in ((None, 50281), (48736, None), (48736, 0)):
        assert (
            observed_effort_continuity(
                effort_changed=True, cached_input_tokens=cached, prompt_tokens=prompt
            )
            is None
        )


def test_the_denominator_is_the_whole_prompt():
    """The nesting mistake that once inflated every Codex run by 65%.

    modelpass counts `input_tokens` beside `cached_input_tokens`; Codex's wire
    nests them. Dividing by `input_tokens` alone would report a 97%-cached turn
    as a cache break.
    """
    usage = TokenUsage(input_tokens=1545, cached_input_tokens=48736, cache_write_tokens=0)
    assert usage.prompt_tokens == 50281
    # The wrong denominator does not fail safe -- it yields 48736/1545, a
    # "fraction" above 31. It happens to answer True here, which is the right
    # answer arrived at by nonsense; on a turn with a large fresh input it would
    # answer False for a well-cached prompt. `prompt_tokens` exists so the
    # denominator is never assembled at a call site.
    assert usage.cached_input_tokens / usage.input_tokens > 1


# --- 5. capability never leaks into telemetry ------------------------------------


def test_a_supported_model_still_reports_nothing_without_a_measurement(tmp_path):
    """The rule the whole ticket turns on: a capability is not a cache hit.

    This run is on the runtime and model where the vendor documents continuity.
    It still reports `None`, because nothing changed effort and nothing measured
    a prefix.
    """
    from modelpass.connections import Connection, CredentialRef
    from modelpass.testing import fake_bridge
    from modelpass.types import AuthMode, TerminalEvent, TextDeltaEvent

    conn = Connection(
        name="c",
        runtime=Runtime.ANTHROPIC_API,
        model="claude-opus-5",
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:MODELPASS_TEST_KEY"),
        reasoning="high",
    )
    bridge, *_ = fake_bridge(
        connections=[conn],
        script=[TextDeltaEvent(text="hi")],
        home=tmp_path / "home",
        runtime=Runtime.ANTHROPIC_API,
        env={"MODELPASS_TEST_KEY": "k"},
    )
    events = list(bridge.chat(connection="c", message="x"))
    receipt = next(e.receipt for e in events if hasattr(e, "receipt"))
    terminal = next(e for e in events if isinstance(e, TerminalEvent))

    assert receipt.effort_cache_continuity == "supported"
    assert receipt.effort_cache_mechanism == "anthropic_per_message_effort"
    assert receipt.effort_cache_reachable is False
    assert terminal.effort_change_cache_preserved is None


def test_the_declared_capability_and_the_observed_answer_are_separate_columns(tmp_path):
    """A stored history has to be able to ask: where continuity was promised,
    did we get it? That needs both columns; either alone cannot answer it."""
    from modelpass.connections import Connection, CredentialRef
    from modelpass.testing import fake_bridge
    from modelpass.types import AuthMode, TextDeltaEvent

    conn = Connection(
        name="c",
        runtime=Runtime.ANTHROPIC_API,
        model="claude-opus-5",
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:MODELPASS_TEST_KEY"),
        reasoning="high",
    )
    bridge, *_ = fake_bridge(
        connections=[conn],
        script=[TextDeltaEvent(text="hi")],
        home=tmp_path / "home",
        runtime=Runtime.ANTHROPIC_API,
        env={"MODELPASS_TEST_KEY": "k"},
    )
    bridge.ask("c", "x")
    (record,) = bridge.run_log.read(1)
    assert record["effort_cache_continuity"] == "supported"
    assert record["effort_cache_reachable"] is False
    assert record["effort_change_cache_preserved"] is None


def test_a_run_that_states_no_effort_says_nothing_about_continuity(tmp_path):
    """No dial, no line. The same silence rule the reasoning fields follow."""
    from modelpass.connections import Connection, CredentialRef
    from modelpass.testing import fake_bridge
    from modelpass.types import AuthMode, TextDeltaEvent

    conn = Connection(
        name="c",
        runtime=Runtime.ANTHROPIC_API,
        model="claude-opus-5",
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:MODELPASS_TEST_KEY"),
    )
    bridge, *_ = fake_bridge(
        connections=[conn],
        script=[TextDeltaEvent(text="hi")],
        home=tmp_path / "home",
        runtime=Runtime.ANTHROPIC_API,
        env={"MODELPASS_TEST_KEY": "k"},
    )
    receipt = bridge.ask("c", "x").receipt
    assert receipt.effort_cache_continuity is None
    assert receipt.effort_cache_mechanism is None


def test_an_old_log_line_reads_back_unmeasured_in_both_columns(tmp_path):
    """`None`, not `False`: "nobody measured" and "the cache broke" are the two
    answers this pair exists to keep apart."""
    import json

    from modelpass.runlog import RunLog

    (tmp_path / "runs.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-09-01T00:00:00+00:00",
                "connection": "c",
                "runtime": "anthropic-api",
                "auth_mode": "api_key",
                "status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (record,) = RunLog(tmp_path).read(1)
    assert record["effort_cache_continuity"] is None
    assert record["effort_cache_reachable"] is None
    assert record["effort_change_cache_preserved"] is None


def test_the_fold_reports_preservation_only_when_told_effort_changed():
    """The seam a mid-session mechanism plugs into, held so it stays wired.

    Nothing emits `reasoning_effort_changed` yet. When something does, this is
    the path it takes to the terminal and the ledger -- observed on the way
    past, exactly as the allowance and the echo are.
    """
    from modelpass._fold import RunFold
    from modelpass.connections import Connection, CredentialRef, Guards
    from modelpass.guards import GuardTracker
    from modelpass.preflight import Receipt
    from modelpass.types import AuthMode, TokenUsage, UsageEvent, VendorEvent

    conn = Connection(
        name="c",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    guards = Guards()

    def _fold() -> RunFold:
        return RunFold(
            connection=conn,
            receipt=Receipt(
                connection="c",
                runtime=Runtime.OPENAI_SDK,
                requested_auth_mode=AuthMode.SUBSCRIPTION,
                detected_auth_mode=AuthMode.SUBSCRIPTION,
            ),
            guards=guards,
            tracker=GuardTracker(guards, "c"),
            deadline=None,
        )

    usage = UsageEvent(
        usage=TokenUsage(input_tokens=1545, cached_input_tokens=48736)
    )
    quiet = _fold()
    quiet.feed(usage)
    assert quiet.finish().effort_change_cache_preserved is None

    changed = _fold()
    changed.feed(
        VendorEvent(
            runtime=Runtime.OPENAI_SDK,
            name="reasoning_effort_changed",
            data={"from": "medium", "to": "high"},
        )
    )
    changed.feed(usage)
    assert changed.finish().effort_change_cache_preserved is True
