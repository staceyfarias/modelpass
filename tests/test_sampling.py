"""Sampling controls, and the honesty that comes with them (R5, ticket 1.7).

The requirement has two halves and this file is organized around them. The first
is ordinary: ``temperature`` and its four siblings are request fields and they
reach a runtime that can take them. The second is the one R5 exists for -- **the
receipt tells the truth about what was actually sent** -- and it is what most of
these tests are about, because it is the half that goes wrong silently.

Nothing here spends or opens a socket. Section 4 drives the ``anthropic-api``
adapter through the same fake transport ``tests/test_adapter_anthropic_api.py``
uses, and it is the test half of the evidence that moved that runtime's
``sampling_controls`` and ``max_output_tokens`` cells; the other half is the
installed SDK's own typed surface, read on 2026-09-13 and cited in
``sampling_rules._ANTHROPIC_API_SOURCE``.

The consumer code every rule below was read off is named in
:mod:`modelpass.sampling_rules`. Three apps had each grown a private copy of the
per-model knowledge; these tests are the assertion that one copy now answers for
all of them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from modelpass.adapters.anthropic_api import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    AnthropicAPIAdapter,
)
from modelpass.adapters.base import RunRequest
from modelpass.capabilities import Capability, CapabilityRegistry, Support
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.preflight import plan_launch
from modelpass.runlog import RunRecord, _normalize
from modelpass.runtimes import AGENT_RUNTIMES, API_RUNTIMES, Runtime
from modelpass.sampling_rules import SamplingPlan, plan_sampling, rules_for
from modelpass.testing import FakeAdapter, fake_bridge
from modelpass.types import (
    SAMPLING_FIELDS,
    AuthMode,
    Message,
    ReceiptEvent,
    Role,
    Sampling,
    TerminalEvent,
    TextDeltaEvent,
)
from test_adapter_anthropic_api import (
    ENV,
    KEY_VAR,
    Factory,
    adapter,
    drive,
    text_round,
)

MODEL = "claude-opus-5"


# --- 1. the request type ----------------------------------------------------------


def test_every_field_is_optional_and_none_means_say_nothing():
    empty = Sampling()
    assert empty.is_empty
    assert empty.requested() == {}
    assert empty.to_dict() == {}


def test_temperature_zero_is_a_request_and_not_an_absence():
    """The distinction the whole type exists to hold.

    ``temperature=0.0`` asks for determinism; ``temperature=None`` lets the
    model's own default stand. A dict that flattened them into one value would
    make a RAG evaluation harness's judges -- which run at exactly 0.0 --
    indistinguishable from
    judges nobody configured.
    """
    assert Sampling(temperature=0.0).requested() == {"temperature": 0.0}
    assert Sampling(temperature=None).requested() == {}


def test_the_requested_dict_lists_only_what_was_set():
    sampling = Sampling(temperature=0.2, max_output_tokens=512)
    assert sampling.requested() == {"temperature": 0.2, "max_output_tokens": 512}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 9},
        {"temperature": -1},
        {"top_p": 1.5},
        {"top_k": 0},
        {"max_output_tokens": 0},
        {"reasoning_effort": "extreme"},
    ],
)
def test_an_out_of_range_value_is_refused_at_the_constructor(kwargs):
    """The same bargain ``CacheControl`` makes about ``ttl``.

    A typo'd 20 heard from a constructor beats the same typo heard from a 400
    one HTTP round trip and one billed prompt later.
    """
    with pytest.raises(ValueError):
        Sampling(**kwargs)


@pytest.mark.parametrize("kwargs", [{"temperature": "hot"}, {"top_k": 1.5}])
def test_a_wrongly_typed_value_is_a_type_error(kwargs):
    with pytest.raises(TypeError):
        Sampling(**kwargs)


def test_reasoning_effort_is_normalized_to_lower_case():
    assert Sampling(reasoning_effort="HIGH").reasoning_effort == "high"


def test_a_mapping_coerces_and_an_unknown_field_is_named():
    assert Sampling.coerce({"temperature": 0.5}) == Sampling(temperature=0.5)
    assert Sampling.coerce(None) is None
    with pytest.raises(TypeError) as excinfo:
        Sampling.coerce({"temperatur": 0.5})
    assert "temperatur" in str(excinfo.value)


def test_the_vocabulary_is_named_once():
    """``SAMPLING_FIELDS`` is what keeps the type, the table and the report aligned."""
    assert set(SAMPLING_FIELDS) == set(Sampling().__dataclass_fields__)


# --- 2. the rules: per runtime AND per model --------------------------------------


def test_gpt5_forces_temperature_and_says_so_in_the_note():
    """R5's own worked example, read off a downstream agent host's model factory."""
    plan = plan_sampling(Sampling(temperature=0.2), Runtime.OPENAI_API, "gpt-5")
    assert plan.applied["temperature"] == 1.0
    assert "temperature 0.2 requested; gpt-5 accepts only 1.0, sent 1.0" in plan.notes
    assert not plan.honoured


def test_the_o_series_refuses_every_sampling_control():
    plan = plan_sampling(
        Sampling(temperature=0.2, top_p=0.9), Runtime.OPENAI_API, "o1-mini"
    )
    assert plan.applied == {}
    assert "temperature not supported on o1, dropped" in plan.notes
    assert "top_p not supported on o1, dropped" in plan.notes


def test_top_k_is_dropped_on_openai_because_the_schema_has_no_such_field():
    plan = plan_sampling(Sampling(top_k=40), Runtime.OPENAI_API, "gpt-4o")
    assert "top_k" not in plan.applied
    assert "top_k not supported on gpt-4o, dropped" in plan.notes


def test_an_anthropic_4x_model_takes_one_of_temperature_or_top_p():
    """A downstream agent host states this rule twice -- in the model factory
    and in the settings form. Here it is stated once."""
    plan = plan_sampling(
        Sampling(temperature=0.2, top_p=0.9), Runtime.ANTHROPIC_API, "claude-sonnet-4-5"
    )
    assert plan.applied["temperature"] == 0.2
    assert "top_p" not in plan.applied
    assert any("accepts one of them" in note for note in plan.notes)


def test_adaptive_thinking_takes_the_sampling_parameters_off_the_call():
    """A RAG evaluation harness's constraint, and its reason: a judge at
    temperature 0 that
    starts thinking is no longer at temperature 0, and a reader of a stored run
    has to be able to see that."""
    plan = plan_sampling(
        Sampling(temperature=0.0, reasoning_effort="high"),
        Runtime.ANTHROPIC_API,
        "claude-opus-5",
    )
    assert "temperature" not in plan.applied
    assert plan.applied["reasoning_effort"] == "high"
    assert any("adaptive thinking" in note for note in plan.notes)


def test_sampling_survives_on_an_adaptive_model_when_no_effort_was_asked_for():
    """The constraint rides *thinking*, not the model. Dropping temperature from
    a call that never asked to think would be a rule invented one step wider
    than the evidence for it."""
    plan = plan_sampling(
        Sampling(temperature=0.3), Runtime.ANTHROPIC_API, "claude-opus-5"
    )
    assert plan.applied["temperature"] == 0.3


def test_the_longest_family_wins_so_an_adaptive_model_is_not_read_as_a_generic_one():
    """``claude-sonnet-4-6`` is under both Anthropic rules. Order decides, and a
    table whose order was an accident would apply the wrong one."""
    rules = rules_for(Runtime.ANTHROPIC_API, "claude-sonnet-4-6")
    assert rules.one_of == (("temperature", "top_p"),)
    assert "temperature" in rules.reasoning_removes


def test_a_family_is_matched_at_a_token_boundary_and_never_by_substring():
    """A RAG evaluation harness's defect, inherited deliberately as its fix:
    ``gpt-4o-mini``
    matching ``gpt-4o`` is right, and matching a reasoning family would send a
    parameter the vendor rejects."""
    assert rules_for(Runtime.OPENAI_API, "gpt-4o-mini").family == "gpt-4o"
    assert rules_for(Runtime.OPENAI_API, "o1-preview").family == "o1"
    # Not a reasoning model, and nothing here may decide otherwise.
    assert rules_for(Runtime.OPENAI_API, "gpt-4o-mini").reasoning_parameter is None


def test_an_unknown_model_gets_runtime_defaults_and_the_receipt_says_so():
    plan = plan_sampling(Sampling(temperature=0.2), Runtime.OPENAI_API, "gpt-9-turbo")
    assert plan.applied["temperature"] == 0.2
    assert (
        "model rules unknown for 'gpt-9-turbo' on openai-api; runtime defaults applied"
        in plan.notes
    )


def test_an_unknown_model_is_never_guessed_into_a_family():
    """Guessing that an unrecognised model reasons sends a parameter the vendor
    rejects and the call stops working; guessing the other way costs one line of
    disclosure. The table extends, it does not infer."""
    rules = rules_for(Runtime.OPENAI_API, "gpt-9-turbo")
    assert not rules.model_known
    assert rules.reasoning_parameter is None


def test_a_temperature_over_the_runtime_ceiling_is_coerced_and_named():
    plan = plan_sampling(Sampling(temperature=1.8), Runtime.ANTHROPIC_API, MODEL)
    assert plan.applied["temperature"] == 1.0
    assert any("accepts at most 1.0" in note for note in plan.notes)


def test_a_required_field_nobody_set_is_supplied_and_disclosed():
    """A 4096-token ceiling used to be discoverable only from a truncated answer."""
    plan = plan_sampling(None, Runtime.ANTHROPIC_API, MODEL)
    assert plan.applied == {"max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS}
    assert plan.requested == {}
    assert any("requires one" in note for note in plan.notes)


def test_a_plan_that_changed_nothing_reports_honoured_and_no_notes():
    plan = plan_sampling(
        Sampling(temperature=0.4, max_output_tokens=256), Runtime.ANTHROPIC_API, MODEL
    )
    assert plan.applied == {"temperature": 0.4, "max_output_tokens": 256}
    assert plan.notes == ()
    assert plan.honoured


def test_every_runtime_has_a_rules_row():
    """The same guard ``STATIC_TABLE`` carries: a ``Runtime`` member with no row
    would raise from inside a call rather than say the table is incomplete."""
    for runtime in Runtime:
        assert rules_for(runtime).runtime is runtime


def test_every_rules_row_carries_a_dated_source_note():
    """A row is evidence or it is a guess, and evidence names its source."""
    for runtime in Runtime:
        rules = rules_for(runtime)
        assert rules.source
        assert len(rules.source) > 60


def test_only_the_driven_runtimes_claim_to_be_verified():
    """``verified`` follows the adapter, one ticket at a time.

    ``anthropic-api`` in 1.6-1.7, ``openai-api`` in 1.9, ``openai-compatible``
    in 1.10 and ``google-api`` in 1.11 -- which is every API runtime, so what is
    left unverified here is the four *agent* rows that never had a drive to have.
    """
    for runtime in sorted(API_RUNTIMES, key=str):
        assert rules_for(runtime).verified, runtime
    for runtime in (Runtime.GOOGLE_CLI, Runtime.GOOGLE_SDK):
        assert not rules_for(runtime).verified


def test_top_k_is_dropped_on_the_compatible_row_and_named():
    """Ticket 1.10 took ``top_k`` off this row, and a drop is never silent.

    The rule the row states: ``openai`` 2.32.0's ``chat.completions.create``
    has no ``top_k``, the servers that take one disagree about where it goes,
    and modelpass names a dropped field rather than guessing an ``extra_body``
    key that would ride along on every other endpoint's request.
    """
    rules = rules_for(Runtime.OPENAI_COMPATIBLE, "llama3.1")
    assert not rules.accepts("top_k")
    assert {"temperature", "top_p", "max_output_tokens"} <= rules.accepted
    plan = plan_sampling(
        Sampling(temperature=0.5, top_k=40), Runtime.OPENAI_COMPATIBLE, "llama3.1"
    )
    assert "top_k" not in plan.applied
    assert plan.applied["temperature"] == 0.5
    assert any("top_k" in note for note in plan.notes)


def test_a_rules_row_serializes():
    """D11, like everything else a caller may want to log."""
    data = rules_for(Runtime.ANTHROPIC_API, MODEL).to_dict()
    assert json.loads(json.dumps(data)) == data


# --- 3. the agent runtimes: dropped, named, never an error ------------------------


@pytest.mark.parametrize("runtime", sorted(AGENT_RUNTIMES, key=str))
def test_an_agent_runtime_takes_no_sampling_at_all(runtime):
    plan = plan_sampling(
        Sampling(temperature=0.2, max_output_tokens=99), runtime, "whatever"
    )
    assert plan.applied == {}
    assert len(plan.notes) == 2


def test_sampling_on_a_subscription_is_reported_rather_than_refused(tmp_path):
    """The asymmetry with ``tools=``, stated as a test.

    Asking a runtime to run a tool it cannot run has no sensible outcome but a
    refusal. Asking for a temperature a runtime will not take has an obvious
    one: send what it takes, and say what happened to the rest.
    """
    bridge, _store, _adapter = fake_bridge(
        connections=[
            Connection(
                name="claude-sub",
                runtime=Runtime.ANTHROPIC_SDK,
                auth_mode=AuthMode.SUBSCRIPTION,
                guards=Guards(),
            )
        ],
        home=tmp_path / "modelpass",
        adapter=FakeAdapter([TextDeltaEvent(text="ok")]),
    )
    events = list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            sampling=Sampling(temperature=0.0, max_output_tokens=4096),
        )
    )
    receipt = next(e for e in events if isinstance(e, ReceiptEvent)).receipt
    assert receipt.sampling_requested == {"temperature": 0.0, "max_output_tokens": 4096}
    assert receipt.sampling_applied == {}
    assert "temperature not supported on anthropic-sdk, dropped" in receipt.sampling_notes
    # And the run finished: nothing here is a refusal.
    assert any(isinstance(e, TerminalEvent) for e in events)


def test_the_capability_cells_say_the_absence_was_checked():
    registry = CapabilityRegistry()
    for runtime in (Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK):
        for capability in (Capability.SAMPLING_CONTROLS, Capability.MAX_OUTPUT_TOKENS):
            assert registry.support(runtime, capability) is Support.UNSUPPORTED
            note = (registry.note(runtime, capability) or "").replace("*", "")
            assert "checked absence" in note


def test_the_google_agent_runtimes_stay_unverified_because_no_adapter_exists():
    """An ``unsupported`` sourced from a family resemblance is the guess this
    table exists to refuse."""
    registry = CapabilityRegistry()
    for runtime in (Runtime.GOOGLE_CLI, Runtime.GOOGLE_SDK):
        for capability in (Capability.SAMPLING_CONTROLS, Capability.MAX_OUTPUT_TOKENS):
            assert registry.support(runtime, capability) is Support.UNVERIFIED


def test_the_pending_api_runtimes_keep_their_cells_unverified():
    """What is left pending is ``openai-compatible``, and it stays that way.

    The list shrank a runtime at a time -- ``anthropic-api`` in 1.7,
    ``openai-api`` in 1.9, ``google-api`` in 1.11 -- and stops here. This row's
    cells are unverified by construction, not by omission: the sampling a server
    behind somebody's baseUrl accepts is that server's business, which is what
    ``refine()`` is for.
    """
    registry = CapabilityRegistry()
    driven = {Runtime.ANTHROPIC_API, Runtime.OPENAI_API, Runtime.GOOGLE_API}
    for runtime in sorted(API_RUNTIMES - driven, key=str):
        for capability in (Capability.SAMPLING_CONTROLS, Capability.MAX_OUTPUT_TOKENS):
            assert registry.support(runtime, capability) is Support.UNVERIFIED


def test_the_google_api_cells_moved_and_name_what_moved_them():
    """Ticket 1.11, on the same two-part evidence 1.7 and 1.9 used."""
    registry = CapabilityRegistry()
    for capability in (Capability.SAMPLING_CONTROLS, Capability.MAX_OUTPUT_TOKENS):
        assert registry.support(Runtime.GOOGLE_API, capability) is Support.SUPPORTED
    note = registry.note(Runtime.GOOGLE_API, Capability.SAMPLING_CONTROLS) or ""
    assert "tests/test_adapter_google_api.py" in note
    # The one control this runtime has and the two OpenAI ones do not, and the
    # one field it has that modelpass refuses to fill in.
    assert "top_k" in note
    assert "thinking_budget" in note


def test_the_openai_api_cells_moved_and_name_what_moved_them():
    """Ticket 1.9, on the same two-part evidence 1.7 used for anthropic-api."""
    registry = CapabilityRegistry()
    for capability in (Capability.SAMPLING_CONTROLS, Capability.MAX_OUTPUT_TOKENS):
        assert registry.support(Runtime.OPENAI_API, capability) is Support.SUPPORTED
    note = registry.note(Runtime.OPENAI_API, Capability.SAMPLING_CONTROLS) or ""
    assert "tests/test_adapter_openai_api.py" in note
    # The one control this runtime has no field for at all.
    assert "top_k" in note


def test_the_anthropic_api_cells_moved_and_name_what_moved_them():
    registry = CapabilityRegistry()
    for capability in (Capability.SAMPLING_CONTROLS, Capability.MAX_OUTPUT_TOKENS):
        assert registry.support(Runtime.ANTHROPIC_API, capability) is Support.SUPPORTED
    note = registry.note(Runtime.ANTHROPIC_API, Capability.SAMPLING_CONTROLS) or ""
    assert "tests/test_sampling.py" in note


def test_the_ceiling_note_distinguishes_itself_from_the_spend_guard():
    """DESIGN R5, and a downstream agent host's validation §3 item 5:
    ``stop_at_tokens`` is a guard, not a knob, and that host's own support module
    already had to say so in prose."""
    registry = CapabilityRegistry()
    note = registry.note(Runtime.ANTHROPIC_SDK, Capability.MAX_OUTPUT_TOKENS) or ""
    assert "stop_at_tokens" in note


# --- 4. the anthropic-api adapter: what actually goes on the request --------------
#
# The test half of the evidence that moved this runtime's two cells. Each of
# these asserts a field on the *vendor call* the fake transport recorded, not on
# a modelpass object -- a report that agreed with itself and disagreed with the
# wire would be the failure, not the check.


def api_connection(model: str | None = MODEL) -> Connection:
    return Connection(
        name="claude-key",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(f"env:{KEY_VAR}"),
        model=model,
        guards=Guards(),
    )


def api_request(**kwargs: Any) -> RunRequest:
    conn = kwargs.pop("connection", None) or api_connection()
    return RunRequest(
        connection=conn,
        messages=(Message(role=Role.USER, content="hello"),),
        plan=plan_launch(conn, ENV),
        **kwargs,
    )


def sent(**kwargs: Any) -> dict[str, Any]:
    """One round's recorded vendor call."""
    factory = Factory([text_round("ok")])
    drive(api_request(**kwargs), factory)
    return factory.clients[0].calls[0]


def test_temperature_top_p_and_top_k_all_reach_the_messages_api():
    """All three are in ``anthropic`` 0.97.0's ``Messages.create`` signature."""
    call = sent(
        connection=api_connection("claude-3-haiku"),
        sampling=Sampling(temperature=0.3, top_p=0.8, top_k=40),
    )
    assert call["temperature"] == 0.3
    assert call["top_p"] == 0.8
    assert call["top_k"] == 40


def test_max_output_tokens_becomes_max_tokens():
    assert sent(sampling=Sampling(max_output_tokens=128))["max_tokens"] == 128


def test_a_call_that_asked_for_no_ceiling_still_carries_the_default_one():
    """``max_tokens`` is required on this API: there is no "the model decides"."""
    assert sent()["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


def test_the_anthropic_default_ceiling_is_one_number():
    """The constant and the rules table hold the same figure.

    Two spellings of one default is how one figure reaches the wire and a
    different one reaches the receipt.
    """
    rules = rules_for(Runtime.ANTHROPIC_API)
    assert rules.defaults["max_output_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


def test_reasoning_effort_rides_output_config_effort():
    """``OutputConfigParam.effort`` in anthropic 0.97.0, not the ``thinking``
    parameter -- which takes a token *budget* modelpass would have to invent."""
    call = sent(
        connection=api_connection("claude-3-haiku"),
        sampling=Sampling(reasoning_effort="high"),
    )
    assert call["output_config"] == {"effort": "high"}
    assert "thinking" not in call


def test_a_schema_and_an_effort_share_output_config_rather_than_overwriting():
    """Both live under one key in this SDK. Assignment would drop one silently."""
    call = sent(
        connection=api_connection("claude-3-haiku"),
        sampling=Sampling(reasoning_effort="low"),
        schema={"type": "object", "properties": {"a": {"type": "string"}}},
        schema_name="Answer",
    )
    assert call["output_config"]["effort"] == "low"
    assert "format" in call["output_config"]


def test_a_model_that_takes_one_of_the_pair_gets_one_of_the_pair_on_the_wire():
    call = sent(
        connection=api_connection("claude-sonnet-4-5"),
        sampling=Sampling(temperature=0.2, top_p=0.9),
    )
    assert call["temperature"] == 0.2
    assert "top_p" not in call


def test_an_adaptive_model_asked_to_think_sends_no_temperature():
    call = sent(sampling=Sampling(temperature=0.0, reasoning_effort="medium"))
    assert "temperature" not in call
    assert call["output_config"] == {"effort": "medium"}


def test_the_ticket_16_option_still_works_as_an_alias():
    """One source of truth, two doors. Deprecated in the docstring, warning-free
    until the consumer migrations."""
    assert sent(options={"max_output_tokens": 128})["max_tokens"] == 128


def test_the_request_field_wins_over_the_option_when_a_caller_sets_both():
    call = sent(
        options={"max_output_tokens": 128}, sampling=Sampling(max_output_tokens=512)
    )
    assert call["max_tokens"] == 512


def test_the_alias_is_folded_in_before_anything_reads_the_request():
    request = api_request(options={"max_output_tokens": 64})
    assert request.sampling is None
    assert request.effective_sampling == Sampling(max_output_tokens=64)


def test_the_adapter_sends_exactly_what_the_plan_applied():
    """The invariant the honesty rests on. If these two could disagree, the
    receipt would be a second opinion rather than a report."""
    sampling = Sampling(temperature=1.9, top_p=0.4, top_k=5, reasoning_effort="high")
    conn = api_connection("claude-sonnet-4-5")
    call = sent(connection=conn, sampling=sampling)
    plan = plan_sampling(sampling, Runtime.ANTHROPIC_API, conn.model)
    on_the_wire = {
        "temperature": call.get("temperature"),
        "top_p": call.get("top_p"),
        "top_k": call.get("top_k"),
        "max_output_tokens": call.get("max_tokens"),
        "reasoning_effort": (call.get("output_config") or {}).get("effort"),
    }
    assert {k: v for k, v in on_the_wire.items() if v is not None} == dict(plan.applied)


def test_an_adapter_driven_with_no_sampling_at_all_behaves_as_it_did_before():
    """Ticket 1.6's single-turn call is untouched: one required field, nothing else."""
    call = sent()
    assert set(call) == {"model", "messages", "max_tokens"}


# --- 5. the receipt and the run log -----------------------------------------------


def api_bridge(tmp_path, factory, model: str | None = MODEL):
    from modelpass.bridge import Bridge
    from modelpass.store import ConnectionStore

    store = ConnectionStore(tmp_path / "modelpass")
    store.add(api_connection(model))
    api = adapter(factory)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_API: api},
        env=dict(ENV),
    )
    return bridge


def test_the_receipt_reports_requested_and_applied_before_the_run(tmp_path):
    bridge = api_bridge(tmp_path, Factory([text_round("ok")]), "claude-sonnet-4-5")
    receipt = bridge.preflight(
        "claude-key", sampling=Sampling(temperature=0.2, top_p=0.9)
    )
    assert receipt.sampling_requested == {"temperature": 0.2, "top_p": 0.9}
    assert receipt.sampling_applied["temperature"] == 0.2
    assert "top_p" not in receipt.sampling_applied
    assert receipt.sampling_notes


def test_the_receipt_serializes_the_whole_report(tmp_path):
    bridge = api_bridge(tmp_path, Factory([text_round("ok")]))
    data = bridge.preflight("claude-key", sampling=Sampling(temperature=0.4)).to_dict()
    assert data["sampling_requested"] == {"temperature": 0.4}
    assert data["sampling_applied"]["temperature"] == 0.4
    assert isinstance(data["sampling_notes"], list)
    assert json.loads(json.dumps(data)) == data


def test_the_run_log_keeps_the_report_after_the_iterator_is_gone(tmp_path):
    bridge = api_bridge(tmp_path, Factory([text_round("ok")]), "gpt-ish")
    list(
        bridge.chat(
            connection="claude-key",
            message="hi",
            sampling=Sampling(temperature=0.2, max_output_tokens=64),
        )
    )
    record = bridge.run_log.read()[0]
    assert record["sampling_requested"] == {"temperature": 0.2, "max_output_tokens": 64}
    assert record["sampling_applied"]["max_output_tokens"] == 64


def test_a_record_written_before_these_keys_existed_still_parses():
    """Additive keys in an append-only file nobody rewrites (the R3 bargain)."""
    old = {
        "timestamp": "2026-08-01T00:00:00+00:00",
        "connection": "claude-sub",
        "runtime": "anthropic-sdk",
        "auth_mode": "subscription",
        "status": "ok",
    }
    normalized = _normalize(dict(old))
    assert normalized["sampling_requested"] == {}
    assert normalized["sampling_applied"] == {}
    assert normalized["sampling_notes"] == []


def test_a_hand_edited_record_is_repaired_rather_than_trusted():
    broken = {"timestamp": "t", "sampling_requested": "nonsense", "sampling_notes": 7}
    normalized = _normalize(broken)
    assert normalized["sampling_requested"] == {}
    assert normalized["sampling_notes"] == []


def test_a_record_round_trips_through_json():
    record = RunRecord(
        timestamp="t",
        connection="c",
        runtime="anthropic-api",
        auth_mode="api_key",
        status="ok",
        sampling_requested={"temperature": 0.2},
        sampling_applied={"temperature": 1.0},
        sampling_notes=("temperature 0.2 requested; sent 1.0",),
    )
    data = record.to_dict()
    assert json.loads(json.dumps(data)) == data


# --- 6. the LangChain leaf --------------------------------------------------------


langchain = pytest.importorskip(
    "langchain_core", reason="needs the modelpass[langchain] extra"
)


def chat_model(tmp_path, **kwargs):
    from modelpass.langchain_adapter import ChatSubpass

    bridge, _store, _adapter = fake_bridge(
        connections=[
            Connection(
                name="claude-sub",
                runtime=Runtime.ANTHROPIC_SDK,
                auth_mode=AuthMode.SUBSCRIPTION,
                guards=Guards(),
            )
        ],
        home=tmp_path / "modelpass",
        adapter=FakeAdapter([TextDeltaEvent(text="ok")]),
    )
    return ChatSubpass(connection="claude-sub", bridge=bridge, **kwargs)


def test_constructor_sampling_reaches_the_answers_metadata(tmp_path):
    """What a desktop agent app could not get, and stopped asking for."""
    model = chat_model(tmp_path, temperature=0.0, max_tokens=4096)
    answer = model.invoke("hi")
    assert answer.response_metadata["subpass_sampling_applied"] == {}
    notes = answer.response_metadata["subpass_sampling_notes"]
    assert any("temperature" in note for note in notes)
    assert any("max_output_tokens" in note for note in notes)


def test_a_bound_kwarg_overrides_the_instance_for_one_call(tmp_path):
    model = chat_model(tmp_path, temperature=0.0)
    bound = model.bind(temperature=0.9)
    notes = bound.invoke("hi").response_metadata["subpass_sampling_notes"]
    assert any("temperature" in note for note in notes)
    # The instance itself is untouched -- that is what ``bind`` means.
    assert model.temperature == 0.0


def test_the_leaf_renames_max_tokens_once_and_in_one_place(tmp_path):
    model = chat_model(tmp_path, max_tokens=256)
    assert model._sampling_fields()["max_output_tokens"] == 256
    assert model._sampling_for({}) == Sampling(max_output_tokens=256)


def test_a_model_with_no_sampling_sends_none(tmp_path):
    assert chat_model(tmp_path)._sampling_for({}) is None


def test_the_metadata_keys_are_additive_and_nothing_was_renamed(tmp_path):
    """Existing ``subpass_*`` keys are consumer contract."""
    metadata = chat_model(tmp_path).invoke("hi").response_metadata
    for key in (
        "subpass_receipt",
        "subpass_auth_mode",
        "subpass_runtime",
        "subpass_status",
        "model_name",
    ):
        assert key in metadata
    assert metadata["subpass_sampling_applied"] == {}
    assert metadata["subpass_sampling_notes"] == []


# --- 7. the plan object itself ----------------------------------------------------


def test_a_plan_serializes():
    plan = plan_sampling(Sampling(temperature=0.2), Runtime.OPENAI_API, "gpt-5")
    data = plan.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert set(data) == {"requested", "applied", "notes"}


def test_an_empty_plan_is_empty():
    plan = plan_sampling(None, Runtime.ANTHROPIC_SDK, "whatever")
    assert plan == SamplingPlan(requested={}, applied={}, notes=(), rules=plan.rules)


def test_the_adapter_class_still_reports_an_unknown_option_key():
    """The option alias did not open the closed set."""
    assert AnthropicAPIAdapter.unknown_option_keys(
        {"max_output_tokens": 1, "temprature": 2}
    ) == ("temprature",)
