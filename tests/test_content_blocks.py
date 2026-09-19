"""Structured message content and cache breakpoints (R3, ticket 1.5).

The failure this file exists to end, with its evidence:

* A downstream agent host builds a four-breakpoint prompt and turns the markers
  into Anthropic ``cache_control`` blocks -- and then strips them again in its
  own runtime wrapper before the call, because the library had nowhere to put
  them.
* A retrieval service lost the same thing when it moved onto this library
  (validation 2026-09-13).
* A batch scraping tool, the same again.

None of those three got an error. They got a silent flattening, a full-price
write on every call, and no way to tell from anything the library reported.

So: ``Message.content`` carries blocks, the blocks carry markers, and where a
runtime will not take them the receipt **says the drop out loud** instead of
performing it quietly. Whether a given runtime takes them is one capability cell
with a dated note, not a guess.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from modelpass.adapters.anthropic import split_messages
from modelpass.adapters.base import RunRequest
from modelpass.adapters.openai import render_prompt
from modelpass.capabilities import Capability, CapabilityRegistry, Support
from modelpass.connections import Connection, Guards
from modelpass.preflight import CacheEligibility, plan_launch
from modelpass.runlog import RunLog, RunRecord
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter, fake_bridge
from modelpass.types import (
    AuthMode,
    CacheControl,
    Message,
    ReceiptEvent,
    Role,
    TextBlock,
    TextDeltaEvent,
    normalize_content,
)

MARKED = (
    TextBlock("system persona"),
    TextBlock("the long stable corpus", CacheControl()),
    TextBlock("today's variable tail"),
)


def build_bridge(tmp_path, runtime=Runtime.ANTHROPIC_SDK, **kwargs):
    adapter = FakeAdapter([TextDeltaEvent(text="ok")], runtime=runtime, **kwargs)
    bridge, _store, _adapter = fake_bridge(
        connections=[
            Connection(
                name="conn",
                runtime=runtime,
                auth_mode=AuthMode.SUBSCRIPTION,
                guards=Guards(),
            )
        ],
        home=tmp_path / "modelpass",
        adapter=adapter,
    )
    return bridge, adapter


# --- 1. the block vocabulary ------------------------------------------------------


def test_a_cache_control_takes_the_two_ttls_the_vendor_has_and_refuses_the_rest():
    assert CacheControl().ttl is None
    assert CacheControl(ttl="5m").ttl == "5m"
    assert CacheControl(ttl="1h").ttl == "1h"
    with pytest.raises(ValueError, match="ttl must be one of"):
        CacheControl(ttl="10m")
    with pytest.raises(ValueError, match="must be 'ephemeral'"):
        CacheControl(type="persistent")  # type: ignore[arg-type]


def test_the_block_vocabulary_is_text_only_and_says_so():
    """v1-out stays out: adding a marker is not the occasion to add images."""
    assert "text only" in (TextBlock.__doc__ or "").lower()
    with pytest.raises(ValueError, match="text content blocks only"):
        TextBlock.coerce({"type": "image", "source": {"data": "..."}})


def test_a_string_message_is_untouched_in_every_respect():
    """The whole compatibility promise in one test."""
    message = Message(Role.USER, "hi")
    assert message.content == "hi"
    assert message.text == "hi"
    assert message.cache_breakpoints == 0
    assert message.to_dict() == {"role": "user", "content": "hi"}


def test_text_reads_a_block_message_as_the_string_every_consumer_expected():
    message = Message(Role.SYSTEM, MARKED)
    assert message.text == "system personathe long stable corpustoday's variable tail"
    assert message.cache_breakpoints == 1
    assert message.blocks == MARKED


def test_a_string_message_still_answers_blocks_with_one_unmarked_block():
    assert Message(Role.USER, "hi").blocks == (TextBlock("hi"),)


def test_blocks_round_trip_through_to_dict_with_the_markers_intact():
    message = Message(Role.SYSTEM, MARKED)
    wire = json.loads(json.dumps(message.to_dict()))
    assert wire == {
        "role": "system",
        "content": [
            {"type": "text", "text": "system persona"},
            {
                "type": "text",
                "text": "the long stable corpus",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "today's variable tail"},
        ],
    }
    assert Message.coerce(wire) == message


def test_a_ttl_survives_the_round_trip_too():
    message = Message(Role.SYSTEM, (TextBlock("x", CacheControl(ttl="1h")),))
    assert Message.coerce(json.loads(json.dumps(message.to_dict()))) == message


def test_normalize_content_accepts_the_dict_form_langchain_and_anthropic_both_speak():
    blocks = normalize_content(
        [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
        ]
    )
    assert blocks == (TextBlock("a"), TextBlock("b", CacheControl(ttl="5m")))


def test_content_that_is_neither_a_string_nor_blocks_is_refused():
    with pytest.raises(TypeError):
        Message(Role.USER, 7)  # type: ignore[arg-type]


# --- 2. every adapter reads .text, so a block message still runs -------------------


def test_the_anthropic_splitter_flattens_blocks_rather_than_choking_on_them():
    system, prompt = split_messages(
        (Message(Role.SYSTEM, MARKED), Message(Role.USER, "q"))
    )
    assert system == "system persona\n\nthe long stable corpus\n\ntoday's variable tail"
    assert prompt == "q"


def test_the_codex_renderer_flattens_blocks_too():
    rendered = render_prompt((Message(Role.USER, (TextBlock("a"), TextBlock("b"))),))
    assert rendered == "a\n\nb"


def test_the_request_exposes_what_an_api_adapter_will_send(tmp_path):
    """The fields ticket 1.6 reads straight into ``messages.create``."""
    bridge, adapter = build_bridge(tmp_path)
    list(
        bridge.chat(
            connection="conn",
            system_prompt=MARKED,
            message=(TextBlock("q", CacheControl(ttl="5m")),),
        )
    )
    request: RunRequest = adapter.requests[0]
    assert request.system_blocks == MARKED
    assert request.conversation_blocks == (
        ("user", (TextBlock("q", CacheControl(ttl="5m")),)),
    )
    assert request.cache_breakpoints == 2
    # And the char count, which the caching heuristic reads, still counts the text.
    assert request.system_prompt_chars == len(
        "system personathe long stable corpustoday's variable tail"
    )


# --- 3. the capability cell -------------------------------------------------------


def test_the_two_shipped_runtimes_do_not_take_breakpoints_and_the_note_says_why():
    registry = CapabilityRegistry()
    for runtime in (Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK):
        assert (
            registry.support(runtime, Capability.CACHE_BREAKPOINTS)
            is Support.UNSUPPORTED
        )
        note = registry.note(runtime, Capability.CACHE_BREAKPOINTS) or ""
        assert "2026-09-13" in note


def test_the_anthropic_sdk_note_names_the_evidence_it_was_read_off():
    """A cell moves with evidence; so does a cell that stays put."""
    note = (
        CapabilityRegistry().note(Runtime.ANTHROPIC_SDK, Capability.CACHE_BREAKPOINTS)
        or ""
    )
    assert "claude_agent_sdk 0.2.148" in note
    assert "SystemPromptPreset" in note
    # And it refuses the misreading it would otherwise invite.
    assert "not 'no caching'" in note


def test_breakpoints_are_not_the_ttl_cell():
    """anthropic-sdk has a TTL lever and no breakpoint channel -- hence two cells."""
    registry = CapabilityRegistry()
    assert registry.supports(Runtime.ANTHROPIC_SDK, Capability.TTL_CONTROL)
    assert not registry.supports(Runtime.ANTHROPIC_SDK, Capability.CACHE_BREAKPOINTS)


# --- 4. the receipt says what happened to them ------------------------------------


def test_a_dropped_breakpoint_is_a_named_note_on_the_receipt(tmp_path):
    bridge, _ = build_bridge(tmp_path)
    receipt = bridge.preflight(
        "conn", messages=[Message(Role.SYSTEM, MARKED), Message(Role.USER, "q")]
    )
    assert receipt.cache_breakpoints_requested == 1
    assert receipt.cache_breakpoints_honoured == 0
    assert (
        "cache breakpoints were requested and dropped: anthropic-sdk does not "
        "accept them" in receipt.notes
    )


def test_a_plain_string_call_gets_no_note_and_two_zeroes(tmp_path):
    bridge, _ = build_bridge(tmp_path)
    receipt = bridge.preflight("conn", messages=[Message(Role.USER, "q")])
    assert receipt.cache_breakpoints_requested == 0
    assert receipt.cache_breakpoints_honoured == 0
    assert not any("cache breakpoints" in note for note in receipt.notes)


def test_a_runtime_that_honoured_them_would_report_them_honoured(tmp_path):
    """Ticket 1.6 flips one cell; nothing else here has to change."""
    registry = CapabilityRegistry()
    registry = registry.refined(
        Runtime.ANTHROPIC_SDK, {"cache_breakpoints": True}
    )
    bridge, _store, _adapter = fake_bridge(
        connections=[
            Connection(
                name="conn",
                runtime=Runtime.ANTHROPIC_SDK,
                auth_mode=AuthMode.SUBSCRIPTION,
            )
        ],
        home=tmp_path / "modelpass",
        adapter=FakeAdapter([TextDeltaEvent(text="ok")]),
        registry=registry,
    )
    receipt = bridge.preflight(
        "conn", messages=[Message(Role.SYSTEM, MARKED), Message(Role.USER, "q")]
    )
    assert receipt.cache_breakpoints_requested == 1
    assert receipt.cache_breakpoints_honoured == 1
    assert not any("were requested and dropped" in note for note in receipt.notes)


def test_the_receipt_dict_carries_both_counts(tmp_path):
    bridge, _ = build_bridge(tmp_path)
    receipt = bridge.preflight(
        "conn", messages=[Message(Role.SYSTEM, MARKED), Message(Role.USER, "q")]
    )
    wire = json.loads(json.dumps(receipt.to_dict()))
    assert wire["cache_breakpoints_requested"] == 1
    assert wire["cache_breakpoints_honoured"] == 0


def test_the_receipt_reaches_the_caller_before_the_run_does(tmp_path):
    """The drop is disclosed on the event that precedes any spend (D4d)."""
    bridge, _ = build_bridge(tmp_path)
    events = list(
        bridge.chat(connection="conn", system_prompt=MARKED, message="q")
    )
    first = events[0]
    assert isinstance(first, ReceiptEvent)
    assert first.receipt.cache_breakpoints_requested == 1
    assert first.receipt.cache_breakpoints_honoured == 0


# --- 5. eligibility reports a fact, not a heuristic, where breakpoints exist -------


def test_explicit_breakpoints_replace_the_estimate_in_the_clause():
    honoured = CacheEligibility(
        system_prompt_chars=40, explicit_breakpoints=4, breakpoints_honoured=True
    )
    assert honoured.clause == "cache: 4 explicit breakpoints on a runtime that honours them"
    dropped = replace(honoured, breakpoints_honoured=False)
    assert dropped.clause == "cache: 4 explicit breakpoints requested, will be dropped"


def test_one_breakpoint_is_singular():
    one = CacheEligibility(explicit_breakpoints=1, breakpoints_honoured=True)
    assert "1 explicit breakpoint on" in one.clause


def test_the_heuristic_is_kept_for_plain_string_prompts():
    """Nothing about the D20 disclosure moves for a caller who sent a string."""
    plain = CacheEligibility(system_prompt_chars=40)
    assert plain.clause == "cache: prefix likely below the caching floor"
    assert CacheEligibility().clause == "cache: no system prompt, so no reusable prefix"


def test_the_note_states_the_drop_and_keeps_the_floor_arithmetic_underneath():
    dropped = CacheEligibility(system_prompt_chars=40, explicit_breakpoints=2)
    note = dropped.note
    assert "2 explicit cache breakpoints" in note
    assert "does not accept them" in note
    # A breakpoint below the floor is still below the floor; it stops being the
    # headline, not the truth.
    assert "minimum cacheable prefix" in note


def test_the_eligibility_dict_carries_the_two_new_fields():
    data = CacheEligibility(explicit_breakpoints=3, breakpoints_honoured=True).to_dict()
    assert data["explicit_breakpoints"] == 3
    assert data["breakpoints_honoured"] is True


def test_the_bridge_stamps_the_counts_onto_the_adapters_eligibility(tmp_path):
    bridge, _ = build_bridge(tmp_path, cache=CacheEligibility(system_prompt_chars=40))
    receipt = bridge.preflight(
        "conn", messages=[Message(Role.SYSTEM, MARKED), Message(Role.USER, "q")]
    )
    assert receipt.cache is not None
    assert receipt.cache.explicit_breakpoints == 1
    assert receipt.cache.breakpoints_honoured is False


# --- 6. the run log ---------------------------------------------------------------


def test_a_run_writes_both_counts_to_the_ledger(tmp_path):
    bridge, _ = build_bridge(tmp_path)
    list(bridge.chat(connection="conn", system_prompt=MARKED, message="q"))
    record = bridge.run_log.read()[0]
    assert record["cache_breakpoints_requested"] == 1
    assert record["cache_breakpoints_honoured"] == 0


def test_an_old_record_written_before_these_keys_existed_still_parses(tmp_path):
    """runs.jsonl is append-only and never rewritten, so old lines must stay readable."""
    log = RunLog(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    legacy = {
        "timestamp": "2026-08-17T10:00:00+00:00",
        "connection": "claude-sub",
        "runtime": "anthropic-sdk",
        "auth_mode": "subscription",
        "status": "ok",
        "model": "claude-opus-4",
        "input_tokens": 10,
        "output_tokens": 5,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 15,
        "guards_configured": False,
        "failed_over_from": None,
    }
    log.path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    (record,) = log.read()
    assert record["connection"] == "claude-sub"
    assert record["total_tokens"] == 15
    # Absent means zero here, and that is not a guess: a line with no blocks
    # carried no breakpoints.
    assert record["cache_breakpoints_requested"] == 0
    assert record["cache_breakpoints_honoured"] == 0


def test_a_record_carries_the_counts_into_its_own_dict():
    from modelpass.types import TerminalEvent, TerminalStatus

    record = RunRecord.from_terminal(
        TerminalEvent(
            status=TerminalStatus.OK,
            connection="conn",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
        ),
        cache_breakpoints_requested=4,
        cache_breakpoints_honoured=4,
    )
    assert record.to_dict()["cache_breakpoints_requested"] == 4
    assert record.to_dict()["cache_breakpoints_honoured"] == 4


# --- 7. sessions are unaffected ---------------------------------------------------


def test_a_session_request_answers_the_same_question_with_zero(tmp_path):
    from modelpass.adapters.base import SessionRequest
    from modelpass.types import SessionKind

    connection = Connection(
        name="conn", runtime=Runtime.ANTHROPIC_SDK, auth_mode=AuthMode.SUBSCRIPTION
    )
    request = SessionRequest(
        connection=connection,
        plan=plan_launch(connection, {}),
        kind=SessionKind.CHAT,
        project_folder=str(tmp_path),
        system_prompt="persona",
    )
    assert request.cache_breakpoints == 0
