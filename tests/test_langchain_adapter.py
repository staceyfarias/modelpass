"""``ChatSubpass`` over the scripted adapter -- no network, no vendor package, no key.

Every test here drives the real :class:`~modelpass.Bridge` through
:func:`modelpass.testing.fake_bridge`, so what is asserted is the actual event
vocabulary modelpass emits rather than a mock of it.

The one that matters most is ``usage_metadata``: LangChain cost accounting reads
it, so an adapter that forgets to populate it makes every cost metric in the
host application report zero while looking like it worked.

Skipped whole when ``langchain-core`` is absent -- it is the optional
``modelpass[langchain]`` extra, and core's suite has to stay green without it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import modelpass

pytest.importorskip("langchain_core", reason="needs the modelpass[langchain] extra")

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from modelpass.connections import (
    Connection,
    CredentialRef,
    Guards,
    QuotaAction,
    QuotaPolicy,
)
from modelpass.langchain_adapter import (
    ChatSubpass,
    SubpassGuardStopError,
    SubpassQuotaExhaustedError,
    SubpassRunError,
    to_json_schema,
)
from modelpass.runtimes import Runtime
from modelpass.schema import openai_strict_issues
from modelpass.testing import (
    FakeAdapter,
    fake_bridge,
    quota_exhausted,
    structured,
    usage,
    vendor_failure,
)
from modelpass.types import (
    AuthMode,
    CacheControl,
    TextBlock,
    TextDeltaEvent,
    ThinkingEvent,
)


@dataclass
class DocumentDecision:
    """Which sources to keep for an answer, and why."""

    keep_source_identities: list[str]
    exclude_source_identities: list[str]
    rationale: str = ""


@dataclass
class ReadDecision:
    """A second schema, only ever used to prove two cannot be bound at once."""

    read: bool


#: One schema-shaped answer, reused so the assertions read as a real payload.
DECISION = {
    "keep_source_identities": ["policy.md"],
    "exclude_source_identities": [],
    "rationale": "the policy doc is the only plausible source",
}


def build_bridge(
    tmp_path,
    script=(),
    *,
    guards=None,
    model=None,
    runtime=Runtime.ANTHROPIC_SDK,
    name="claude-sub",
    **adapter_kwargs,
):
    """A bridge over one subscription connection and a scripted adapter."""
    adapter = FakeAdapter(script, runtime=runtime, **adapter_kwargs)
    bridge, _store, _adapter = fake_bridge(
        connections=[
            Connection(
                name=name,
                runtime=runtime,
                auth_mode=AuthMode.SUBSCRIPTION,
                guards=guards or Guards(),
                model=model,
            )
        ],
        home=tmp_path / "modelpass",
        adapter=adapter,
    )
    return bridge, adapter


def codex_bridge(tmp_path, script=()):
    """The same, on the runtime that requires the strict schema subset."""
    return build_bridge(
        tmp_path, script, runtime=Runtime.OPENAI_SDK, name="codex-sub"
    )


def failover_bridge(tmp_path, script=(), *, adapter=None):
    """A subscription connection configured to fail over onto a metered one."""
    fake = adapter if adapter is not None else FakeAdapter(script)
    bridge, _store, _adapter = fake_bridge(
        connections=[
            Connection(
                name="claude-sub",
                runtime=Runtime.ANTHROPIC_SDK,
                auth_mode=AuthMode.SUBSCRIPTION,
                guards=Guards(
                    on_quota_exhausted=QuotaPolicy(
                        action=QuotaAction.FAILOVER, failover="claude-api"
                    )
                ),
            ),
            Connection(
                name="claude-api",
                runtime=Runtime.ANTHROPIC_SDK,
                auth_mode=AuthMode.API_KEY,
                credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
            ),
        ],
        home=tmp_path / "modelpass",
        adapter=fake,
        # Named explicitly rather than set on the process: the point of a fake
        # bridge is that the machine it runs on cannot change the answer.
        env={"ANTHROPIC_API_KEY": "sk-not-real"},
    )
    return bridge, fake


def merge(chunks):
    """Fold streamed chunks the way LangChain's own aggregation does."""
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    return merged


# --- the extra is a leaf, not a core dependency (D8) -----------------------------


def test_importing_modelpass_core_does_not_import_langchain():
    # Core keeps its zero runtime dependencies. Exporting the adapter eagerly
    # from ``modelpass/__init__.py`` would make every ``import modelpass`` on every
    # machine need langchain-core, which is exactly what the extra exists to
    # avoid -- so this is checked in a fresh interpreter rather than in this
    # process, which has already imported it.
    probe = "import modelpass, sys; print('langchain_core' in sys.modules)"
    # PYTHONPATH is named rather than inherited: pytest's own ``pythonpath``
    # setting does not reach a subprocess, so without this the probe could end
    # up importing an installed copy instead of the tree under test.
    source_root = Path(modelpass.__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(source_root)},
    )
    assert result.stdout.strip() == "False"


# --- message mapping -------------------------------------------------------------


def test_system_human_and_assistant_messages_map_to_modelpass_roles(tmp_path):
    bridge, adapter = build_bridge(tmp_path, [TextDeltaEvent(text="ok")])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    model.invoke(
        [
            SystemMessage(content="you are an editor"),
            HumanMessage(content="first"),
            AIMessage(content="an earlier answer"),
            HumanMessage(content="second"),
        ]
    )

    sent = [m.to_dict() for m in adapter.requests[0].messages]
    assert sent == [
        {"role": "system", "content": "you are an editor"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "an earlier answer"},
        {"role": "user", "content": "second"},
    ]


def test_the_full_history_is_resent_on_every_call(tmp_path):
    # modelpass is stateless: there is no session to hold, so a second call that
    # sent only the new turn would lose the conversation entirely.
    bridge, adapter = build_bridge(tmp_path, [TextDeltaEvent(text="ok")])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    model.invoke([SystemMessage(content="s"), HumanMessage(content="one")])
    model.invoke(
        [
            SystemMessage(content="s"),
            HumanMessage(content="one"),
            AIMessage(content="ok"),
            HumanMessage(content="two"),
        ]
    )

    assert len(adapter.requests[0].messages) == 2
    assert len(adapter.requests[1].messages) == 4


def test_anthropic_style_content_blocks_reach_the_request_as_blocks(tmp_path):
    """R3, ticket 1.5. This is the exact shape a downstream agent host and a
    retrieval service send.

    Before content blocks existed the two marked spans were joined into one
    string and the ``cache_control`` was dropped at the door -- no error, no
    warning, no caching. Now the leading block-bearing system message is hoisted
    into ``system_prompt=``, which is the front of the cached prefix and the
    only position where a breakpoint means anything, and the marker survives to
    ``RunRequest.system_blocks`` for an adapter that can use it.
    """
    bridge, adapter = build_bridge(tmp_path, [TextDeltaEvent(text="ok")])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    model.invoke(
        [
            SystemMessage(
                content=[
                    {"type": "text", "text": "instructions"},
                    {
                        "type": "text",
                        "text": "corpus",
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
            ),
            HumanMessage(content="q"),
        ]
    )

    request = adapter.requests[0]
    system = request.messages[0]
    assert system.role.value == "system"
    assert system.text == "instructionscorpus"
    assert request.system_blocks == (
        TextBlock("instructions"),
        TextBlock("corpus", CacheControl()),
    )
    assert request.cache_breakpoints == 1


def test_a_string_system_message_still_rides_history_unchanged(tmp_path):
    """The hoist is for block content only: a string caller's bytes do not move."""
    bridge, adapter = build_bridge(tmp_path, [TextDeltaEvent(text="ok")])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    model.invoke([SystemMessage(content="instructions"), HumanMessage(content="q")])

    request = adapter.requests[0]
    assert [m.content for m in request.messages] == ["instructions", "q"]
    assert request.cache_breakpoints == 0


def test_a_message_type_with_no_subscription_equivalent_is_refused(tmp_path):
    bridge, _ = build_bridge(tmp_path, [TextDeltaEvent(text="ok")])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    with pytest.raises(ValueError, match="cannot map a 'tool' message"):
        model.invoke(
            [HumanMessage(content="q"), ToolMessage(content="r", tool_call_id="1")]
        )


def test_a_list_ending_on_an_assistant_turn_is_refused(tmp_path):
    """``Bridge.chat`` takes the new *user* turn (D21); nothing else may pose as one.

    LangChain's list is split positionally -- everything but the last turn is
    ``history=`` -- so a trailing assistant message would be relabelled as
    something the user said. Refusing says what happened; relabelling would put
    words in the user's mouth.
    """
    bridge, adapter = build_bridge(tmp_path, [TextDeltaEvent(text="ok")])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    with pytest.raises(ValueError, match="last message to be a human turn"):
        model.invoke([HumanMessage(content="q"), AIMessage(content="half an answer")])
    assert adapter.requests == []


# --- usage_metadata: the one the cost reporting depends on -----------------------


def test_usage_metadata_is_populated_from_the_terminal_event(tmp_path):
    bridge, _ = build_bridge(
        tmp_path,
        [
            TextDeltaEvent(text="answer"),
            usage(input_tokens=1000, output_tokens=250, cached=400),
        ],
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    result = model.invoke([HumanMessage(content="q")])

    # The convention change is deliberate. modelpass counts cached_input_tokens
    # separately from input_tokens; langchain-anthropic reports input_tokens as
    # the TRUE total including cached reads, with the cached part broken out.
    # The adapter matches the incumbent so a host's stored totals mean the same
    # thing whichever provider produced them.
    assert result.usage_metadata["input_tokens"] == 1400
    assert result.usage_metadata["output_tokens"] == 250
    assert result.usage_metadata["total_tokens"] == 1650
    assert result.usage_metadata["input_token_details"]["cache_read"] == 400
    # Nothing was written to cache on this run, so this is genuinely zero.
    assert result.usage_metadata["input_token_details"]["cache_creation"] == 0


def test_cache_writes_reach_usage_metadata_through_the_whole_adapter(tmp_path):
    """End to end, because the isolated mapper is where this hid.

    ``_usage_metadata`` was taught to read ``cache_write_tokens`` while
    ``_usage_dict`` -- its only producer -- still emitted three keys, so the
    mapper was correct, its unit test passed, and every real call reported
    zero. A test that stops at the mapper cannot see that; this one goes
    through the same path ``_generate`` uses.

    The under-count mattered: writes are billed as input, so a cold prefix was
    missing from a host's totals by the whole size of the prefix. Measured
    live by a RAG evaluation harness on 2026-08-30 -- a ~7,000-word message
    reported
    ``input_tokens: 2`` with ``cache_creation: 0``, when the write was 7,851.
    """
    bridge, _ = build_bridge(
        tmp_path,
        [
            TextDeltaEvent(text="answer"),
            usage(input_tokens=2, output_tokens=4, cache_write=7851),
        ],
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    result = model.invoke([HumanMessage(content="a long, unique prompt")])

    assert result.usage_metadata["input_token_details"]["cache_creation"] == 7851
    # input_tokens is the TRUE total, so the write counts toward it -- which is
    # also why a bare input_tokens of 2 on a long payload is not a mystery.
    assert result.usage_metadata["input_tokens"] == 7853
    assert result.usage_metadata["total_tokens"] == 7857


def test_several_usage_reports_are_not_double_counted(tmp_path):
    # The bridge already folds delta-vs-run_total reports into one running
    # total. Accumulating the per-event figures here as well would report a
    # multi-turn run at roughly twice its real cost.
    bridge, _ = build_bridge(
        tmp_path,
        [
            usage(input_tokens=100, output_tokens=10),
            TextDeltaEvent(text="a"),
            usage(input_tokens=50, output_tokens=5),
        ],
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    result = model.invoke([HumanMessage(content="q")])

    assert result.usage_metadata["input_tokens"] == 150
    assert result.usage_metadata["output_tokens"] == 15


def test_the_receipt_reaches_the_caller_as_response_metadata(tmp_path):
    # The evidence of which subscription paid, without a second preflight.
    bridge, _ = build_bridge(tmp_path, [TextDeltaEvent(text="ok")])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    metadata = model.invoke([HumanMessage(content="q")]).response_metadata

    assert metadata["subpass_auth_mode"] == "subscription"
    assert metadata["subpass_runtime"] == "anthropic-sdk"
    assert metadata["subpass_account"] == "fake-account"
    assert metadata["subpass_plan_name"] == "Fake Plan"
    assert metadata["subpass_status"] == "ok"
    assert metadata["subpass_receipt"]


# --- streaming -------------------------------------------------------------------


def test_stream_yields_deltas_and_carries_usage_on_the_last_chunk(tmp_path):
    bridge, _ = build_bridge(
        tmp_path,
        [
            TextDeltaEvent(text="Hel"),
            TextDeltaEvent(text="lo"),
            usage(input_tokens=9, output_tokens=2),
        ],
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    chunks = list(model.stream([HumanMessage(content="q")]))

    assert "".join(str(chunk.content) for chunk in chunks) == "Hello"
    # Usage arrives late by construction -- one runtime reports it per turn, the
    # other only at the end -- so it rides a trailing chunk and reaches the
    # caller through LangChain's own chunk aggregation, which is how a streaming
    # consumer actually reads it.
    merged = merge(chunks)
    assert merged.usage_metadata["input_tokens"] == 9
    assert merged.usage_metadata["output_tokens"] == 2
    assert merged.response_metadata["subpass_auth_mode"] == "subscription"


def test_thinking_events_do_not_leak_into_the_answer(tmp_path):
    bridge, _ = build_bridge(
        tmp_path,
        [ThinkingEvent(text="hmm, the policy says"), TextDeltaEvent(text="Yes.")],
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    assert model.invoke([HumanMessage(content="q")]).content == "Yes."


# --- error and guard mapping -----------------------------------------------------


def test_a_vendor_failure_terminal_becomes_a_raise(tmp_path):
    # modelpass reports a vendor failure as a terminal event with status=error,
    # never an exception. LangChain callers -- including any retry loop wrapped
    # around the model -- expect a raise, and this is where that happens.
    bridge, _ = build_bridge(
        tmp_path,
        [TextDeltaEvent(text="partial")],
        error=vendor_failure("400: the model was rejected by the runtime"),
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    with pytest.raises(SubpassRunError) as excinfo:
        model.invoke([HumanMessage(content="q")])

    assert "400" in str(excinfo.value)
    # The receipt rides along, because a rejected model is only diagnosable from
    # what the receipt named.
    assert "claude-sub" in excinfo.value.receipt_summary
    assert excinfo.value.partial_text == "partial"


def test_a_guard_stop_raises_its_own_type_with_the_partial_text(tmp_path):
    bridge, _ = build_bridge(
        tmp_path,
        [
            TextDeltaEvent(text="as far as it got"),
            usage(input_tokens=500, output_tokens=100),
        ],
        guards=Guards(stop_at_tokens=100),
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    with pytest.raises(SubpassGuardStopError) as excinfo:
        model.invoke([HumanMessage(content="q")])

    assert excinfo.value.partial_text == "as far as it got"
    assert excinfo.value.usage["output_tokens"] == 100


def test_quota_exhaustion_is_its_own_exception_type(tmp_path):
    # Distinct because it is the one outcome a user can act on directly, and
    # because no vendor exposes remaining allowance before a run -- hitting this
    # is the only way an app finds out.
    bridge, _ = build_bridge(tmp_path, [quota_exhausted()])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    with pytest.raises(SubpassQuotaExhaustedError):
        model.invoke([HumanMessage(content="q")])


def test_stop_at_tokens_lowers_the_ceiling_for_one_call(tmp_path):
    bridge, _ = build_bridge(
        tmp_path,
        [TextDeltaEvent(text="x"), usage(input_tokens=60, output_tokens=0)],
        guards=Guards(stop_at_tokens=100_000),
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge, stop_at_tokens=50)

    with pytest.raises(SubpassGuardStopError):
        model.invoke([HumanMessage(content="q")])


# --- failover: opt-in, and recorded when it happens ------------------------------


def test_a_configured_failover_is_declined_unless_the_model_opts_in(tmp_path):
    bridge, adapter = failover_bridge(tmp_path, [quota_exhausted()])
    model = ChatSubpass(connection="claude-sub", bridge=bridge, allow_failover=False)

    # The run stops cleanly on quota instead of crossing onto the metered
    # connection. The decline goes through modelpass's own per-call keyword; the
    # adapter never implements a failover of its own.
    with pytest.raises(SubpassQuotaExhaustedError):
        model.invoke([HumanMessage(content="q")])
    assert [r.connection.name for r in adapter.requests] == ["claude-sub"]


def test_an_opted_in_failover_runs_and_is_stamped_as_crossing_to_metered(tmp_path):
    def script(request):
        if request.connection.name == "claude-sub":
            return [quota_exhausted()]
        return [
            TextDeltaEvent(text="finished on the key"),
            usage(input_tokens=5, output_tokens=1),
        ]

    bridge, _ = failover_bridge(tmp_path, script)
    model = ChatSubpass(connection="claude-sub", bridge=bridge, allow_failover=True)

    result = model.invoke([HumanMessage(content="q")])

    assert result.content == "finished on the key"
    assert result.response_metadata["subpass_failed_over_from"] == "claude-sub"
    assert result.response_metadata["subpass_crosses_to_metered"] is True


# --- schema-bound output ---------------------------------------------------------


def test_bind_tools_sends_the_schema_and_the_name_it_was_given(tmp_path):
    # The name is wire protocol, not a label: on anthropic-sdk the runtime's
    # structured-output mechanism IS a tool call, and a prompt may refer to the
    # schema by name. modelpass never invents one (D13), so what tool_choice says
    # has to arrive unchanged.
    bridge, adapter = build_bridge(
        tmp_path,
        [
            structured(DECISION, schema=to_json_schema(DocumentDecision)),
            usage(input_tokens=40, output_tokens=8),
        ],
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    model.bind_tools([DocumentDecision], tool_choice="DocumentDecision").invoke(
        [HumanMessage(content="q")]
    )

    request = adapter.requests[0]
    assert request.schema_name == "DocumentDecision"
    assert request.schema["properties"]["keep_source_identities"]["type"] == "array"


def test_the_structured_answer_arrives_as_a_langchain_tool_call(tmp_path):
    # The load-bearing shape of the whole swap: a caller reading
    # result.tool_calls[0]["args"] has to get the answer in exactly the form a
    # forced tool choice over an API key produces.
    bridge, _ = build_bridge(
        tmp_path,
        [
            TextDeltaEvent(text="thinking out loud"),
            structured(DECISION, schema=to_json_schema(DocumentDecision)),
            usage(input_tokens=40, output_tokens=8),
        ],
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    result = model.bind_tools(
        [DocumentDecision], tool_choice="DocumentDecision"
    ).invoke([HumanMessage(content="q")])

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "DocumentDecision"
    assert result.tool_calls[0]["args"] == DECISION
    assert result.tool_calls[0]["id"]
    # Text the run produced on the way is kept, not suppressed -- it was paid for.
    assert result.content == "thinking out loud"
    # ...and the usage the cost reporting reads still lands.
    assert result.usage_metadata["input_tokens"] == 40


def test_with_structured_output_takes_the_same_path_as_bind_tools(tmp_path):
    # Deliberately NOT LangChain's "returns the parsed object" contract: the
    # answer rides a tool_call, and building the class is the caller's job.
    bridge, adapter = build_bridge(
        tmp_path, [structured(DECISION, schema=to_json_schema(DocumentDecision))]
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    result = model.with_structured_output(DocumentDecision).invoke(
        [HumanMessage(content="q")]
    )

    assert adapter.requests[0].schema_name == "DocumentDecision"
    assert result.tool_calls[0]["args"] == DECISION


def test_an_unbound_call_carries_no_tool_calls_at_all(tmp_path):
    # Free-text call sites share this model. A stray empty tool call would send
    # a caller's "did it answer in schema?" branch down the wrong path.
    bridge, _ = build_bridge(tmp_path, [TextDeltaEvent(text="NAME: x")])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    assert model.invoke([HumanMessage(content="q")]).tool_calls == []


def test_a_schema_bound_stream_carries_the_answer_on_the_final_chunk(tmp_path):
    bridge, _ = build_bridge(
        tmp_path,
        [
            TextDeltaEvent(text="par"),
            TextDeltaEvent(text="tial"),
            structured(DECISION, schema=to_json_schema(DocumentDecision)),
        ],
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    chunks = list(
        model.bind_tools([DocumentDecision], tool_choice="DocumentDecision").stream(
            [HumanMessage(content="q")]
        )
    )

    merged = merge(chunks)
    assert merged.content == "partial"
    # Not a fake stream of partial JSON: modelpass emits the answer once, whole,
    # just before the terminal, so it rides one chunk.
    assert merged.tool_calls[0]["args"] == DECISION


def test_binding_more_than_one_schema_is_refused(tmp_path):
    # modelpass refuses schema= together with tools= on both runtimes, and there
    # is no caller-executed tool loop here to choose between several.
    bridge, _ = build_bridge(tmp_path)
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    with pytest.raises(ValueError, match="exactly one structured-output schema"):
        model.bind_tools([DocumentDecision, ReadDecision])


def test_a_run_that_produces_no_schema_bound_answer_raises(tmp_path):
    # modelpass never substitutes an empty object for a missing answer: a run that
    # produced nothing parseable ends with a terminal error naming what did come
    # back, and the adapter turns that into a raise like any other. The outcome
    # that must never happen is the quiet one -- an empty result that looks
    # exactly like a real one which found nothing.
    bridge, _ = build_bridge(tmp_path, [TextDeltaEvent(text="prose, no schema")])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    with pytest.raises(SubpassRunError, match="no structured answer"):
        model.bind_tools([DocumentDecision], tool_choice="DocumentDecision").invoke(
            [HumanMessage(content="q")]
        )


def test_an_invalid_answer_is_delivered_and_logged_rather_than_dropped(
    tmp_path, caplog
):
    # modelpass's `valid` is its own structural check, not the vendor's, and it
    # never edits `data`. The adapter keeps that posture: the answer is handed
    # over as returned, and the drift is made visible in the log rather than
    # turned into a quietly thinner result.
    wrong = {"keep_source_identities": "not-a-list", "exclude_source_identities": []}
    bridge, _ = build_bridge(
        tmp_path, [structured(wrong, schema=to_json_schema(DocumentDecision))]
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    with caplog.at_level("WARNING"):
        result = model.bind_tools(
            [DocumentDecision], tool_choice="DocumentDecision"
        ).invoke([HumanMessage(content="q")])

    assert result.tool_calls[0]["args"] == wrong
    assert "failed modelpass's structural check" in caplog.text


# --- the schema subset the two runtimes disagree about ---------------------------


def test_an_openai_connection_gets_the_strict_schema_and_anthropic_does_not(tmp_path):
    # A schema derived from a class with defaults is accepted by anthropic-sdk
    # as-is and 400s on openai-sdk before any generation. modelpass reports the
    # difference rather than converting silently; converting is the caller's
    # decision, and this adapter makes it from the connection's runtime.
    loose = to_json_schema(DocumentDecision)
    assert openai_strict_issues(loose)  # the schema really is loose

    codex, codex_adapter = codex_bridge(tmp_path, [structured(DECISION, schema=loose)])
    ChatSubpass(connection="codex-sub", bridge=codex).bind_tools(
        [DocumentDecision], tool_choice="DocumentDecision"
    ).invoke([HumanMessage(content="q")])

    assert openai_strict_issues(codex_adapter.requests[0].schema) == ()

    claude, claude_adapter = build_bridge(
        tmp_path / "anthropic", [structured(DECISION, schema=loose)]
    )
    ChatSubpass(connection="claude-sub", bridge=claude).bind_tools(
        [DocumentDecision], tool_choice="DocumentDecision"
    ).invoke([HumanMessage(content="q")])

    # Untouched: the loose form is what anthropic-sdk verified against, and
    # making an optional field mandatory-but-nullable changes the answer's shape.
    assert claude_adapter.requests[0].schema == loose


def test_strict_mode_nulls_are_stripped_from_the_answer(tmp_path):
    # to_openai_strict expresses "optional" as required-and-nullable, so an
    # openai-sdk run reports every field it had nothing to say about as null.
    # Handing that straight to a constructor replaces the class's own default --
    # a list field becomes None, and the next thing to iterate it raises three
    # layers from here.
    answer = {
        "keep_source_identities": ["policy.md"],
        "exclude_source_identities": [],
        "rationale": None,
    }
    bridge, _ = codex_bridge(tmp_path, [structured(answer, valid=True)])
    model = ChatSubpass(connection="codex-sub", bridge=bridge)

    result = model.bind_tools(
        [DocumentDecision], tool_choice="DocumentDecision"
    ).invoke([HumanMessage(content="q")])

    # The key is gone, not None: absent is what the loose schema produces, so
    # both runtimes hand the same dict to the same constructor.
    assert "rationale" not in result.tool_calls[0]["args"]
    assert result.tool_calls[0]["args"]["keep_source_identities"] == ["policy.md"]


def test_nulls_are_left_alone_on_an_anthropic_connection(tmp_path):
    # Nothing was converted there, so a null in the answer is the model's own
    # word and stripping it would be the adapter editing the answer.
    answer = {
        "keep_source_identities": ["policy.md"],
        "exclude_source_identities": [],
        "rationale": None,
    }
    bridge, _ = build_bridge(tmp_path, [structured(answer, valid=True)])
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    result = model.bind_tools(
        [DocumentDecision], tool_choice="DocumentDecision"
    ).invoke([HumanMessage(content="q")])

    assert result.tool_calls[0]["args"]["rationale"] is None


# --- sampling controls: forwarded, and reported (R5, ticket 1.7) ------------------


def test_a_configured_temperature_no_longer_warns_at_construction(tmp_path, caplog):
    """The warning this class used to emit is gone, and its premise with it.

    It fired at construction, before a connection was resolved -- so it could
    not know whether the runtime honours the value, and after 1.7 it would have
    shouted at an ``anthropic-api`` connection that honours all of it. The
    honesty moved to where the answer is: the receipt, and from there
    ``response_metadata["subpass_sampling_notes"]``.

    A desktop agent app is why this matters rather than being tidying: it
    stopped passing this class a temperature at all because of this warning, so
    a user who set one silently got nothing.
    """
    bridge, _ = build_bridge(tmp_path)
    with caplog.at_level("WARNING"):
        ChatSubpass(
            connection="claude-sub", bridge=bridge, temperature=0.0, max_tokens=4096
        )

    assert caplog.text == ""


def test_the_configured_sampling_is_still_visible_at_debug_level(tmp_path, caplog):
    """Not silence: "which knobs did this model think it had" is a real question."""
    bridge, _ = build_bridge(tmp_path)
    with caplog.at_level("DEBUG"):
        ChatSubpass(connection="claude-sub", bridge=bridge, temperature=0.0)

    assert "temperature" in caplog.text


def test_no_log_line_at_all_when_no_sampling_was_configured(tmp_path, caplog):
    bridge, _ = build_bridge(tmp_path)
    with caplog.at_level("DEBUG"):
        ChatSubpass(connection="claude-sub", bridge=bridge)

    assert "sampling" not in caplog.text


# --- configuration mistakes ------------------------------------------------------


def test_a_connection_that_does_not_exist_is_a_caller_error(tmp_path):
    # Everything raised before the event iterator exists is configuration or a
    # caller mistake; a vendor failure arrives as a terminal event instead.
    bridge, _ = build_bridge(tmp_path, [TextDeltaEvent(text="ok")])
    model = ChatSubpass(connection="typo-connection", bridge=bridge)

    with pytest.raises(ValueError, match="typo-connection"):
        model.invoke([HumanMessage(content="q")])


def test_the_leaf_s_errors_carry_the_terminal_s_verdict(tmp_path):
    """Ticket 1.12b: a 429 reaches a retry loop as a 429, through ``invoke``.

    The two consumers that match these classes by name across the MRO wrap a
    retry loop around ``invoke``; before this the verdict stopped at the
    terminal event and the raise carried a reason string, so a rate limit was
    classified UNKNOWN and never retried.
    """
    from modelpass.retry import classify_error
    from modelpass.types import Retryable, TerminalEvent, TerminalStatus

    rate_limited = TerminalEvent(
        status=TerminalStatus.ERROR,
        connection="",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        reason="429: rate limited",
        status_code=429,
        retry_after=7.0,
    )
    adapter = FakeAdapter(
        [TextDeltaEvent(text="half"), rate_limited], runtime=Runtime.ANTHROPIC_API
    )
    bridge, _store, _adapter = fake_bridge(
        connections=[
            Connection(
                name="claude-api",
                runtime=Runtime.ANTHROPIC_API,
                auth_mode=AuthMode.API_KEY,
                credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
            )
        ],
        home=tmp_path / "modelpass",
        adapter=adapter,
        env={"ANTHROPIC_API_KEY": "sk-not-a-real-key"},
    )
    model = ChatSubpass(connection="claude-api", bridge=bridge)

    with pytest.raises(SubpassRunError) as excinfo:
        model.invoke([HumanMessage(content="q")])

    error = excinfo.value
    assert error.retryable is Retryable.YES
    assert error.status_code == 429
    assert error.retry_after == 7.0
    # The event itself rides along -- the bridge's stamped one, not the
    # adapter's draft -- so nothing it knew is lost at the raise.
    assert error.terminal.status is TerminalStatus.ERROR
    assert error.terminal.connection == "claude-api"
    assert classify_error(error).retryable is Retryable.YES
    # Nothing 1.12 put here moved: the partial text and the receipt still ride.
    assert error.partial_text == "half"


def test_a_guard_stop_through_the_leaf_reads_no(tmp_path):
    from modelpass.retry import classify_error
    from modelpass.types import Retryable, TerminalStatus

    bridge, _ = build_bridge(
        tmp_path,
        [TextDeltaEvent(text="as far as it got"), usage(input_tokens=500)],
        guards=Guards(stop_at_tokens=100),
    )
    model = ChatSubpass(connection="claude-sub", bridge=bridge)

    with pytest.raises(SubpassGuardStopError) as excinfo:
        model.invoke([HumanMessage(content="q")])

    assert excinfo.value.retryable is Retryable.NO
    assert excinfo.value.terminal.status is TerminalStatus.GUARD_STOP
    assert classify_error(excinfo.value).retryable is Retryable.NO
