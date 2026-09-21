"""Structured output end to end, offline (D13).

The vendor-shaped halves are driven from the **live fixtures captured on
2026-08-17** (``tests/fixtures/structured_output/``) rather than from payloads
written to match the code, so a runtime changing its shape breaks a test instead
of quietly changing the answer. Provenance:
``docs/api-and-runtimes.md`` §2.2a.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from modelpass.adapters.anthropic import STRUCTURED_OUTPUT_TOOL, map_message
from modelpass.adapters.base import RunRequest
from modelpass.adapters.openai import OpenAIAdapter, _schema_notes
from modelpass.capabilities import CapabilityRegistry, Support
from modelpass.connections import (
    Connection,
    CredentialRef,
    Guards,
    QuotaAction,
    QuotaPolicy,
)
from modelpass.errors import CapabilityNotSupported, InvalidSchema
from modelpass.preflight import plan_launch
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter, fake_bridge, structured, usage
from modelpass.types import (
    AuthMode,
    Message,
    Role,
    StructuredOutputEvent,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    VendorEvent,
)

FIXTURES = Path(__file__).parent / "fixtures" / "structured_output"

SCHEMA = {
    "type": "object",
    "title": "Probe",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "high"]},
    },
    "required": ["answer"],
}


def load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def load_jsonl(name: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- the anthropic mapping, driven from the live capture -------------------------
#
# Stand-ins shaped like the SDK's dataclasses; the mapper dispatches on class
# name. Built *from the fixture* below, so the fixture is load-bearing.


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: bool | None = False


@dataclass
class AssistantMessage:
    content: list[Any] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    error: str | None = None
    stop_reason: str | None = None


@dataclass
class UserMessage:
    content: Any = field(default_factory=list)


@dataclass
class ResultMessage:
    subtype: str = "success"
    is_error: bool = False
    session_id: str = "s"
    usage: dict[str, Any] | None = None
    result: str | None = None
    structured_output: Any = None
    terminal_reason: str | None = "completed"
    api_error_status: int | None = None
    num_turns: int = 2
    duration_ms: int = 10
    total_cost_usd: float | None = None


_BLOCKS = {
    "TextBlock": lambda b: TextBlock(text=b["text"]),
    "ToolUseBlock": lambda b: ToolUseBlock(id=b["id"], name=b["name"], input=b["input"]),
    "ToolResultBlock": lambda b: ToolResultBlock(
        tool_use_id=b["tool_use_id"], content=b.get("content"), is_error=b.get("is_error")
    ),
}


def anthropic_messages() -> list[Any]:
    """Rebuild the live message sequence from the committed fixture."""
    out: list[Any] = []
    for entry in load_json("anthropic-structured-output-2026-08-17.json"):
        cls = entry["class"]
        if cls == "AssistantMessage":
            out.append(
                AssistantMessage(
                    content=[_BLOCKS[b["class"]](b) for b in entry["blocks"]],
                    usage=entry.get("usage"),
                )
            )
        elif cls == "UserMessage":
            out.append(UserMessage(content=[_BLOCKS[b["class"]](b) for b in entry["blocks"]]))
        elif cls == "ResultMessage":
            out.append(
                ResultMessage(
                    subtype=entry["subtype"],
                    is_error=entry["is_error"],
                    usage=entry.get("usage"),
                    result=entry.get("result"),
                    structured_output=entry.get("structured_output"),
                    terminal_reason=entry.get("terminal_reason"),
                    num_turns=entry.get("num_turns", 2),
                )
            )
    return out


def map_all(messages, **kwargs) -> list[Any]:
    tool_names: dict[str, str] = {}
    return [e for m in messages for e in map_message(m, tool_names=tool_names, **kwargs)]


def test_the_fixture_still_says_what_the_verification_recorded():
    """A guard on the evidence itself, not on the code that reads it."""
    log = load_json("anthropic-structured-output-2026-08-17.json")
    init = next(e for e in log if e.get("subtype") == "init")
    assert init["tools"] == [STRUCTURED_OUTPUT_TOOL], (
        "the runtime injects its submission tool even under tools=[]"
    )
    result = next(e for e in log if e["class"] == "ResultMessage")
    assert result["structured_output"] == {"answer": "4", "confidence": "high"}
    assert json.loads(result["result"]) == result["structured_output"]
    assert result["num_turns"] == 2 and result["terminal_reason"] == "completed", (
        "structured output costs a second turn and still completes under max_turns=1"
    )


def test_the_submission_tool_is_not_reported_as_a_caller_tool():
    events = map_all(anthropic_messages(), schema=SCHEMA, schema_name="Probe")
    assert not any(isinstance(e, (ToolCallEvent, ToolResultEvent)) for e in events), (
        "a caller who declared no tools must not see tool traffic"
    )
    names = [e.name for e in events if isinstance(e, VendorEvent)]
    assert "structured_output.call" in names
    assert "structured_output.result" in names


def test_nothing_is_dropped_when_the_submission_tool_is_suppressed():
    """Adapter rule 3 still holds: routed to vendor_event, never discarded."""
    events = map_all(anthropic_messages(), schema=SCHEMA)
    call = next(e for e in events if getattr(e, "name", "") == "structured_output.call")
    assert call.data["name"] == STRUCTURED_OUTPUT_TOOL
    assert call.data["id"] == "toolu_FIXTURE"


def test_prose_before_the_submission_still_streams():
    events = map_all(anthropic_messages(), schema=SCHEMA)
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["4"]


def test_the_structured_event_precedes_the_terminal():
    events = map_all(anthropic_messages(), schema=SCHEMA, schema_name="Probe")
    kinds = [type(e).__name__ for e in events]
    assert kinds.index("StructuredOutputEvent") == kinds.index("TerminalEvent") - 1
    event = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert event.data == {"answer": "4", "confidence": "high"}
    assert event.valid and event.problems == ()
    assert event.schema_name == "Probe"
    assert json.loads(event.raw) == event.data


def test_without_a_schema_the_same_messages_map_exactly_as_before():
    """Phase 6 behavior is untouched when nobody asked for a schema."""
    events = map_all(anthropic_messages())
    assert not any(isinstance(e, StructuredOutputEvent) for e in events)
    assert any(isinstance(e, ToolCallEvent) for e in events)
    assert any(isinstance(e, ToolResultEvent) for e in events)


def test_a_missing_structured_answer_ends_the_run_as_an_error():
    result = ResultMessage(result=None, structured_output=None, usage={})
    events = map_message(result, schema=SCHEMA)
    assert not any(isinstance(e, StructuredOutputEvent) for e in events)
    terminal = next(e for e in events if isinstance(e, TerminalEvent))
    assert terminal.status is TerminalStatus.ERROR
    assert "no structured answer" in terminal.reason


def test_an_unparseable_answer_shows_what_came_back():
    result = ResultMessage(result="sorry, no", structured_output=None, usage={})
    events = map_message(result, schema=SCHEMA)
    event = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert event.data is None and event.raw == "sorry, no" and not event.valid
    terminal = next(e for e in events if isinstance(e, TerminalEvent))
    assert terminal.status is TerminalStatus.ERROR and "sorry, no" in terminal.reason


def test_a_structurally_invalid_answer_does_not_fail_the_run():
    result = ResultMessage(
        result='{"answer": 4}', structured_output={"answer": 4}, usage={}
    )
    events = map_message(result, schema=SCHEMA)
    event = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert event.data == {"answer": 4} and not event.valid and event.problems
    terminal = next(e for e in events if isinstance(e, TerminalEvent))
    assert terminal.status is TerminalStatus.OK


def test_a_vendor_error_keeps_its_status_over_the_schema_complaint():
    """The more important truth wins; a quota stop is not relabelled 'error'."""
    result = ResultMessage(
        subtype="success", is_error=False, api_error_status=429, usage={}
    )
    events = map_message(result, schema=SCHEMA)
    terminal = next(e for e in events if isinstance(e, TerminalEvent))
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED


# --- the openai mapping, driven from the live fixtures ---------------------------


class FakeProc:
    def __init__(self, payloads, returncode: int = 0):
        self.stdout = iter(json.dumps(p) + "\n" for p in payloads)
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        return None


def codex_request(schema=None, **kwargs) -> RunRequest:
    connection = Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    # The Codex half of this module is the EXEC half: it asserts the temp-file
    # --output-schema mechanism and reads a scripted 'codex exec --json' stream.
    # S7 flipped the runtime default to app-server on 2026-08-31 (a change of
    # default, not a removal), where the same schema rides a per-turn
    # TurnStartParams.outputSchema field and no file is written at all -- covered
    # in tests/test_adapter_openai_appserver.py. The option is how these tests
    # say which transport they are about.
    kwargs.setdefault("options", {"transport": "exec"})
    return RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection, {"PATH": "/usr/bin"}),
        schema=schema,
        **kwargs,
    )


def run_codex(payloads, schema=None, schema_name="", returncode=0):
    captured: dict[str, Any] = {}

    def spawn(argv, env, cwd):
        captured["argv"] = argv
        return FakeProc(payloads, returncode)

    adapter = OpenAIAdapter(codex_bin="codex", spawn=spawn)
    request = codex_request(schema=schema, schema_name=schema_name)
    events = list(adapter.run(request))
    return events, captured


#: The schema the live run actually used, in the strict form codex requires.
#: Kept identical to the captured run so the fixture's answer really is an
#: answer to *this* schema.
OPENAI_SCHEMA = {
    "type": "object",
    "title": "ExtractionResult",
    "properties": {
        "context": {"type": "string"},
        "status": {"type": "string", "enum": ["found", "partial", "not_found"]},
        "excerpts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "text": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["chunk_id", "text", "score"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["context", "status", "excerpts"],
    "additionalProperties": False,
}


def test_the_answer_is_the_final_agent_message_text():
    payloads = load_jsonl("openai-strict-schema-2026-08-17.jsonl")
    events, _ = run_codex(payloads, schema=OPENAI_SCHEMA, schema_name="ExtractionResult")
    event = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert event.data["status"] == "found"
    assert event.data["excerpts"][0]["chunk_id"] == "c1"
    assert event.valid and event.schema_name == "ExtractionResult"


def test_the_answer_also_still_streams_as_text():
    """It is genuinely what the runtime produced; hiding it hides the spend."""
    payloads = load_jsonl("openai-strict-schema-2026-08-17.jsonl")
    events, _ = run_codex(payloads, schema=OPENAI_SCHEMA)
    deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(deltas) == 1
    assert json.loads(deltas[0].text)["status"] == "found"


def test_the_structured_event_precedes_the_terminal_on_codex():
    payloads = load_jsonl("openai-strict-schema-2026-08-17.jsonl")
    events, _ = run_codex(payloads, schema=OPENAI_SCHEMA)
    kinds = [type(e).__name__ for e in events]
    assert kinds[-2:] == ["StructuredOutputEvent", "TerminalEvent"]
    assert events[-1].status is TerminalStatus.OK


def test_the_schema_is_passed_as_a_file_that_is_cleaned_up():
    payloads = load_jsonl("openai-strict-schema-2026-08-17.jsonl")
    events, captured = run_codex(payloads, schema=OPENAI_SCHEMA)
    argv = captured["argv"]
    assert "--output-schema" in argv
    path = Path(argv[argv.index("--output-schema") + 1])
    assert not path.exists(), "the per-run schema file is removed"
    assert events  # the run really happened


def test_no_schema_means_no_output_schema_argument():
    payloads = load_jsonl("openai-strict-schema-2026-08-17.jsonl")
    _, captured = run_codex(payloads)
    assert "--output-schema" not in captured["argv"]


def test_a_run_that_produced_no_answer_is_an_error():
    payloads = [{"type": "thread.started", "thread_id": "x"}, {"type": "turn.completed"}]
    events, _ = run_codex(payloads, schema=OPENAI_SCHEMA)
    assert not any(isinstance(e, StructuredOutputEvent) for e in events)
    assert events[-1].status is TerminalStatus.ERROR
    assert "no structured answer" in events[-1].reason


def test_the_vendors_schema_rejection_reaches_the_caller_as_it_was_written():
    """Both live 400s: the error terminal carries the vendor's own words."""
    payloads = load_jsonl("openai-invalid-schema-2026-08-17.jsonl")
    events, _ = run_codex(payloads[:2], schema=OPENAI_SCHEMA, returncode=1)
    terminal = events[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert "invalid_json_schema" in terminal.reason or "required" in terminal.reason


def test_the_captured_rejections_are_what_the_strict_check_predicts():
    """The check and the evidence agree, which is the only reason to ship it."""
    messages = " ".join(
        json.dumps(p) for p in load_jsonl("openai-invalid-schema-2026-08-17.jsonl")
    )
    assert "additionalProperties" in messages
    assert "including every key in properties" in messages

    from modelpass.schema import openai_strict_issues

    loose = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a"],
    }
    issues = " | ".join(openai_strict_issues(loose))
    assert "additionalProperties" in issues and "'b'" in issues


def test_the_preflight_predicts_the_rejection_before_the_run():
    loose = {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}
    notes = _schema_notes(codex_request(schema=loose))
    assert notes and "strict subset" in notes[0]
    assert "to_openai_strict" in notes[0]


def test_the_preflight_says_nothing_when_the_schema_is_fine():
    assert _schema_notes(codex_request(schema=OPENAI_SCHEMA)) == ()
    assert _schema_notes(codex_request()) == ()


# --- the bridge surface ----------------------------------------------------------


def anthropic_connection(**kwargs) -> Connection:
    return Connection(
        name=kwargs.pop("name", "claude-sub"),
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        **kwargs,
    )


def test_a_schema_run_streams_the_event_before_the_terminal(tmp_path):
    bridge, _, _ = fake_bridge(
        connections=[anthropic_connection()],
        script=[TextDeltaEvent("..."), usage(input_tokens=10, output_tokens=5)],
        home=tmp_path / "home",
        adapter=FakeAdapter(
            [TextDeltaEvent("..."), usage(input_tokens=10, output_tokens=5)],
            structured_output={"answer": "4", "confidence": "high"},
        ),
    )
    events = list(
        bridge.chat(
            connection="claude-sub",
            message="2+2",
            schema=SCHEMA,
        )
    )
    kinds = [type(e).__name__ for e in events]
    assert kinds[-2:] == ["StructuredOutputEvent", "TerminalEvent"]
    assert events[-1].status is TerminalStatus.OK
    assert events[-2].data == {"answer": "4", "confidence": "high"}
    assert events[-2].valid


def test_the_adapter_receives_the_normalized_schema_and_the_name(tmp_path):
    bridge, _, adapter = fake_bridge(
        connections=[anthropic_connection()],
        home=tmp_path / "home",
        adapter=FakeAdapter(structured_output={"answer": "4"}),
    )
    list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            schema={"properties": {"answer": {"type": "string"}}},
            schema_name="CallerChosenName",
        )
    )
    request = adapter.requests[0]
    assert request.wants_schema
    assert request.schema["type"] == "object"
    assert request.schema_name == "CallerChosenName"


def test_an_unsupported_runtime_refuses_before_any_spend(tmp_path):
    registry = CapabilityRegistry()
    registry.refine(Runtime.ANTHROPIC_SDK, {"structured_output": Support.UNSUPPORTED})
    bridge, _, adapter = fake_bridge(
        connections=[anthropic_connection()], home=tmp_path / "home", registry=registry
    )
    with pytest.raises(CapabilityNotSupported):
        bridge.chat(
            connection="claude-sub",
            message="hi",
            schema=SCHEMA,
        )
    assert adapter.preflights == [], "the gate is before the preflight"
    assert adapter.requests == []


def test_unverified_is_not_a_yes(tmp_path):
    registry = CapabilityRegistry()
    registry.refine(Runtime.ANTHROPIC_SDK, {"structured_output": Support.UNVERIFIED})
    bridge, _, adapter = fake_bridge(
        connections=[anthropic_connection()], home=tmp_path / "home", registry=registry
    )
    with pytest.raises(CapabilityNotSupported) as exc:
        bridge.chat(
            connection="claude-sub",
            message="hi",
            schema=SCHEMA,
        )
    assert "unverified" in str(exc.value)
    assert adapter.preflights == []


def test_schema_and_tools_together_are_refused(tmp_path):
    bridge, _, adapter = fake_bridge(
        connections=[anthropic_connection()], home=tmp_path / "home"
    )
    with pytest.raises(InvalidSchema) as exc:
        bridge.chat(
            connection="claude-sub",
            message="hi",
            schema=SCHEMA,
            tools=[{"name": "t", "description": "d", "parameters": {}}],
        )
    assert "15451" in str(exc.value)
    assert adapter.preflights == []


def test_schema_and_mcp_servers_together_are_refused(tmp_path):
    bridge, _, _ = fake_bridge(connections=[anthropic_connection()], home=tmp_path / "home")
    with pytest.raises(InvalidSchema):
        bridge.chat(
            connection="claude-sub",
            message="hi",
            schema=SCHEMA,
            mcp_servers={"docs": {"command": "x"}},
        )


def test_a_malformed_schema_is_refused_at_the_boundary(tmp_path):
    bridge, _, adapter = fake_bridge(
        connections=[anthropic_connection()], home=tmp_path / "home"
    )
    with pytest.raises(InvalidSchema):
        bridge.chat(
            connection="claude-sub",
            message="hi",
            schema={"type": "array", "items": {}},
        )
    assert adapter.preflights == []


def test_a_schema_name_without_a_schema_is_a_mistake_worth_hearing_about(tmp_path):
    bridge, _, _ = fake_bridge(connections=[anthropic_connection()], home=tmp_path / "home")
    with pytest.raises(InvalidSchema):
        bridge.chat(
            connection="claude-sub",
            message="hi",
            schema_name="Thing",
        )


# This one drives a :class:`Bridge` whose failover target is ``openai-sdk``, and a
# Bridge refuses a runtime whose SDK does not import -- which would refuse the leg
# for the wrong reason and never reach the capability gate under test.
@pytest.mark.skipif(
    not OpenAIAdapter.is_available(),
    reason="needs the modelpass[openai] extra: openai-codex is not installed",
)
def test_a_schema_bearing_failover_needs_the_target_to_support_it(tmp_path):
    """A failover must be able to answer the same question the first leg was asked."""
    from modelpass.testing import quota_exhausted

    target = Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    primary = anthropic_connection(
        guards=Guards(
            on_quota_exhausted=QuotaPolicy(
                action=QuotaAction.FAILOVER, failover="codex-sub"
            )
        )
    )
    registry = CapabilityRegistry()
    registry.refine(
        Runtime.OPENAI_SDK, {"structured_output": Support.UNSUPPORTED}
    )
    bridge, _, _ = fake_bridge(
        connections=[primary, target],
        home=tmp_path / "home",
        registry=registry,
        adapter=FakeAdapter([quota_exhausted()], structured_output={"answer": "4"}),
    )
    events = list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            schema=SCHEMA,
        )
    )
    terminal = events[-1]
    assert terminal.status is TerminalStatus.QUOTA_EXHAUSTED
    assert "could not run" in terminal.reason and "structured_output" in terminal.reason


# --- the offline test kit --------------------------------------------------------


def test_fake_adapter_ignores_structured_output_without_a_schema(tmp_path):
    bridge, _, _ = fake_bridge(
        connections=[anthropic_connection()],
        home=tmp_path / "home",
        adapter=FakeAdapter([TextDeltaEvent("hi")], structured_output={"answer": "4"}),
    )
    events = list(bridge.chat(connection="claude-sub", message="hi"))
    assert not any(isinstance(e, StructuredOutputEvent) for e in events)


def test_fake_adapter_reproduces_the_missing_answer_failure(tmp_path):
    """Unset structured_output plus a schema is the real failure, on purpose."""
    bridge, _, _ = fake_bridge(
        connections=[anthropic_connection()],
        home=tmp_path / "home",
        adapter=FakeAdapter([TextDeltaEvent("prose, no schema")]),
    )
    events = list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            schema=SCHEMA,
        )
    )
    assert events[-1].status is TerminalStatus.ERROR
    assert "no structured answer" in events[-1].reason


def test_a_hand_written_script_stays_in_charge(tmp_path):
    scripted = structured({"answer": "7"}, schema=SCHEMA)
    bridge, _, _ = fake_bridge(
        connections=[anthropic_connection()],
        home=tmp_path / "home",
        adapter=FakeAdapter([scripted], structured_output={"answer": "ignored"}),
    )
    events = list(
        bridge.chat(
            connection="claude-sub",
            message="hi",
            schema=SCHEMA,
        )
    )
    assert events[-2].data == {"answer": "7"}


def test_structured_defaults_valid_to_the_real_check():
    good = structured({"answer": "4"}, schema=SCHEMA)
    bad = structured({"confidence": "high"}, schema=SCHEMA)
    assert good.valid and good.problems == ()
    assert not bad.valid and bad.problems
    assert json.loads(good.raw) == good.data


def test_structured_can_be_forced_for_a_deliberately_odd_case():
    event = structured({"anything": 1}, schema=SCHEMA, valid=True)
    assert event.valid


def test_fake_bridge_returns_the_triple_and_defaults_to_an_empty_environment(tmp_path):
    bridge, store, adapter = fake_bridge(
        connections=[anthropic_connection()], home=tmp_path / "home"
    )
    assert store.list()[0].name == "claude-sub"
    assert adapter.runtime is Runtime.ANTHROPIC_SDK
    assert dict(bridge.env) == {}, "the developer's shell cannot change the answer"


def test_fake_bridge_keeps_the_store_out_of_the_real_home(tmp_path):
    home = tmp_path / "home"
    _, _, _ = fake_bridge(connections=[anthropic_connection()], home=home)
    assert (home / "connections.toml").is_file()


def test_the_event_serializes_as_protocol():
    event = structured({"answer": "4"}, schema=SCHEMA, schema_name="Probe")
    data = event.to_dict()
    assert data["type"] == "structured_output"
    assert data["data"] == {"answer": "4"}
    assert data["valid"] is True and data["schema_name"] == "Probe"
    json.dumps(data)  # must round-trip through JSON (D11)
