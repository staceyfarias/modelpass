"""The receipt discloses whether a call can cache its prefix (D20, built).

The failure this exists to surface is silent and free to fall into: a system
prompt below the runtime's minimum cacheable size, sent with varying messages,
reads *zero* and pays a full write on every single call. No error, no warning,
``cache_creation_input_tokens: 0`` in the usage. A consumer sat in that state
for a day without knowing.

So the receipt says, before the run, whether the call is *eligible*. Not whether
it will hit -- that depends on a prefix which has not been sent yet -- but
whether it can: is there a system prompt at all, does it plausibly clear the
floor, is a TTL lever available here.

Three things these tests hold the implementation to.

1. **The line renders either way.** A disclosure that only appeared when caching
   already worked would leave everybody else guessing, which is exactly where the
   consumer was.
2. **The estimate admits it is one.** Characters, with the ratio stated, rather
   than a token count modelpass would have had to invent.
3. **Information, not a warning.** A user near the end of a resetting window may
   rationally choose the expensive path, and a receipt that scolds them has
   misread its role.
"""

from __future__ import annotations

import pytest

from modelpass.adapters.base import RunRequest, SessionRequest
from modelpass.adapters.openai import OpenAIAdapter
from modelpass.preflight import (
    CHARS_PER_TOKEN,
    DEFAULT_CACHE_FLOOR_TOKENS,
    OPUS_5_CACHE_FLOOR_TOKENS,
    CacheEligibility,
    cache_floor_tokens,
    plan_launch,
)
from modelpass.testing import FakeAdapter
from modelpass.types import Message, Role, SessionKind

RUBRIC = "Score the answer from 1 to 5."


# --- the floor -------------------------------------------------------------------


def test_opus_5_gets_the_lower_floor_and_the_receipt_says_where_it_came_from():
    floor, source = cache_floor_tokens("claude-opus-5-20260101")
    assert floor == OPUS_5_CACHE_FLOOR_TOKENS
    assert "Opus 5" in source


def test_an_unrecognized_model_gets_the_conservative_floor():
    """The failure direction is chosen: under-promise eligibility, never over-promise.

    Model names arrive as whatever somebody called them -- a dated slug, a family
    alias, an alias of an alias. A spelling nobody anticipated must not be the
    reason a receipt claims a 300-token prompt will cache.
    """
    floor, _ = cache_floor_tokens("some-model-nobody-here-has-heard-of")
    assert floor == DEFAULT_CACHE_FLOOR_TOKENS


def test_no_model_named_takes_the_conservative_floor_and_says_so():
    floor, source = cache_floor_tokens(None)
    assert floor == DEFAULT_CACHE_FLOOR_TOKENS
    assert "no model is named" in source


# --- the verdict -----------------------------------------------------------------


def test_a_sub_floor_prompt_is_reported_as_such_rather_than_left_silent():
    eligibility = CacheEligibility(system_prompt_chars=len(RUBRIC))
    assert eligibility.has_prefix
    assert not eligibility.likely_eligible
    assert "below the caching floor" in eligibility.clause
    note = eligibility.note
    assert "full write and no read on every call" in note
    # The way out, stated, because a verdict with no next step is just a scold.
    assert "Growing it past the floor" in note


def test_a_prompt_past_the_floor_reads_as_eligible():
    eligibility = CacheEligibility(
        system_prompt_chars=DEFAULT_CACHE_FLOOR_TOKENS * CHARS_PER_TOKEN + 1
    )
    assert eligibility.likely_eligible
    assert "likely eligible" in eligibility.clause
    # Read the reads, not the writes -- the guidance that took two wrong guesses
    # to establish, put where the person about to run a loop will meet it.
    assert "cached_input_tokens" in eligibility.note


def test_no_system_prompt_still_produces_a_line():
    """The case the whole design turns on: silence is what people were left with."""
    eligibility = CacheEligibility(system_prompt_chars=0)
    assert not eligibility.has_prefix
    assert not eligibility.likely_eligible
    assert eligibility.clause == "cache: no system prompt, so no reusable prefix"
    assert "priced in full every time" in eligibility.note


def test_the_estimate_is_labelled_an_estimate_in_every_branch_that_uses_one():
    """modelpass does not tokenize, and must not sound as though it does."""
    for chars in (10, DEFAULT_CACHE_FLOOR_TOKENS * CHARS_PER_TOKEN * 2):
        note = CacheEligibility(system_prompt_chars=chars).note
        assert "not a token count" in note
        assert f"~{CHARS_PER_TOKEN} characters per token" in note


def test_the_note_never_recommends_a_number_or_calls_the_caller_wrong():
    """Information, not a warning (D20). Phrasing is the feature here."""
    note = CacheEligibility(system_prompt_chars=len(RUBRIC)).note.lower()
    for scold in ("you should", "warning", "mistake", "wrong", "must "):
        assert scold not in note


def test_a_floor_is_only_named_where_the_floor_decided_the_verdict():
    """Naming the floor under "there is no system prompt" answers nobody's question."""
    source = "the usual floor, from model 'x'"
    assert source not in CacheEligibility(system_prompt_chars=0, floor_source=source).note
    assert source in CacheEligibility(system_prompt_chars=8, floor_source=source).note


# --- the preset in front ---------------------------------------------------------


def test_a_preset_prefix_is_eligible_however_short_the_appended_text_is():
    """A worker judged by its append length would be reported as uncacheable.

    Its prefix begins with the runtime's own preset, which is a full agent prompt
    and clears any of these floors by itself. Reporting a short-instruction worker
    as sub-floor would be confidently and expensively wrong.
    """
    eligibility = CacheEligibility(system_prompt_chars=12, preset_prefix=True)
    assert eligibility.likely_eligible
    assert "runtime's own preset is the front of this prefix" in eligibility.note


def test_a_preset_prefix_with_nothing_appended_is_still_eligible():
    eligibility = CacheEligibility(system_prompt_chars=0, preset_prefix=True)
    assert eligibility.has_prefix
    assert eligibility.likely_eligible


# --- adapters ---------------------------------------------------------------------


def test_codex_always_reports_a_preset_prefix_because_it_has_no_prompt_parameter(
    codex_connection,
):
    """Its persona and tool definitions ride in front on every exec, unasked.

    "No system prompt, so no reusable prefix" would be a true statement about the
    caller and a false one about the request.
    """
    request = RunRequest(
        connection=codex_connection,
        messages=(Message(Role.USER, "hi"),),
        plan=plan_launch(codex_connection, {}),
    )
    eligibility = OpenAIAdapter().cache_eligibility(request)
    assert eligibility.preset_prefix
    assert eligibility.likely_eligible


def test_codex_reports_no_ttl_lever_and_says_why(codex_connection):
    """A checked absence, not an unchecked question (capability ``ttl_control``)."""
    request = RunRequest(
        connection=codex_connection,
        messages=(Message(Role.USER, "hi"),),
        plan=plan_launch(codex_connection, {}),
    )
    eligibility = OpenAIAdapter().cache_eligibility(request)
    assert eligibility.ttl is None
    assert "no TTL lever" in eligibility.ttl_detail


def test_a_session_request_is_answered_from_its_own_prompt_semantics(codex_connection):
    """Not from a synthesized run: only the session request knows append vs replace."""
    request = SessionRequest(
        connection=codex_connection,
        plan=plan_launch(codex_connection, {}),
        kind=SessionKind.WORKER,
        project_folder=".",
        system_prompt=RUBRIC,
    )
    assert request.system_prompt_chars == len(RUBRIC)
    assert OpenAIAdapter().cache_eligibility(request).preset_prefix


def test_an_adapter_that_has_not_answered_says_nothing_rather_than_no(codex_connection):
    """``None`` is a third state. A verdict nobody verified is not an improvement."""
    request = RunRequest(
        connection=codex_connection,
        messages=(Message(Role.USER, "hi"),),
        plan=plan_launch(codex_connection, {}),
    )
    assert FakeAdapter().cache_eligibility(request) is None


# --- reaching the receipt ---------------------------------------------------------


def test_the_receipt_carries_no_cache_clause_when_no_adapter_answered(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([]))
    receipt = bridge.preflight("claude-sub")
    assert receipt.cache is None
    assert receipt.cache_note is None
    assert "cache:" not in receipt.summary()


def test_an_adapters_verdict_reaches_the_summary_the_note_and_the_json(
    bridge_factory, subscription_connection
):
    adapter = FakeAdapter([], cache=CacheEligibility(system_prompt_chars=len(RUBRIC)))
    bridge, _ = bridge_factory(subscription_connection, adapter)
    receipt = bridge.preflight("claude-sub")

    assert "cache: prefix likely below the caching floor" in receipt.summary()
    assert receipt.cache_note is not None
    payload = receipt.to_dict()["cache"]
    assert payload["likely_eligible"] is False
    assert payload["chars_per_token"] == CHARS_PER_TOKEN
    assert payload["system_prompt_chars"] == len(RUBRIC)


def test_the_disclosure_reads_against_the_messages_the_receipt_was_taken_for(
    bridge_factory, subscription_connection
):
    """``preflight(messages=...)`` answers "would *this* prompt cache", unspent."""

    class Sized(FakeAdapter):
        def cache_eligibility(self, request):
            return CacheEligibility(system_prompt_chars=request.system_prompt_chars)

    bridge, _ = bridge_factory(subscription_connection, Sized([]))
    long_enough = "x" * (DEFAULT_CACHE_FLOOR_TOKENS * CHARS_PER_TOKEN + 1)
    receipt = bridge.preflight(
        "claude-sub",
        messages=[{"role": "system", "content": long_enough}, {"role": "user", "content": "hi"}],
    )
    assert receipt.cache is not None
    assert receipt.cache.likely_eligible


def test_a_broken_cache_answer_costs_the_line_and_not_the_run(
    bridge_factory, subscription_connection
):
    """A disclosure is not a guarantee. Losing it must not lose the call."""

    class Broken(FakeAdapter):
        def cache_eligibility(self, request):
            raise RuntimeError("boom")

    bridge, _ = bridge_factory(subscription_connection, Broken([]))
    receipt = bridge.preflight("claude-sub")
    assert receipt.cache is None
    assert receipt.ok


@pytest.fixture
def codex_connection():
    from modelpass.connections import Connection, CredentialRef
    from modelpass.runtimes import Runtime
    from modelpass.types import AuthMode

    return Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
