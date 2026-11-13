"""The automatic strict-schema rewrite, held to its own consequences.

``openai-api`` and ``openai-compatible`` send every schema with ``strict: true``
and tighten it first with :func:`modelpass.schema.to_openai_strict`: an optional
property becomes *required and nullable*, because the strict subset has no
absent keys. Three things followed from that and were not handled:

* **The answer was checked against the schema that was not sent.** A model
  obeying the strict schema answers ``{"name": "a", "nick": null}``; the loose
  schema says ``nick`` is a string; the event came back ``valid=False`` for an
  answer that was exactly what was asked for.
* **``$defs`` were not rewritten.** A Pydantic model with a nested model keeps
  the nested one under ``$defs`` and points at it with ``$ref``. Only the root
  was tightened, so the nested object went out loose under ``strict: true``.
* **The LangChain leaf stripped the introduced nulls on ``openai-sdk`` only**,
  so on the two API runtimes a ``null`` overwrote a dataclass default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

import test_adapter_openai_api as responses_fakes
import test_adapter_openai_compatible as chat_fakes
import test_async_adapters as async_fakes
from modelpass.schema import openai_strict_issues, to_openai_strict, validate_instance
from modelpass.types import StructuredOutputEvent, TerminalStatus

LOOSE = {
    "title": "Person",
    "type": "object",
    "properties": {"name": {"type": "string"}, "nick": {"type": "string"}},
    "required": ["name"],
}

CASES = {"openai-api": responses_fakes, "openai-compatible": chat_fakes}


def _sync(fakes: Any, rounds: list[Any], **kwargs: Any) -> list[Any]:
    return fakes.drive(fakes.run_request(**kwargs), fakes.Factory(rounds))


def _async(fakes: Any, rounds: list[Any], **kwargs: Any) -> list[Any]:
    build = {module: case for case, module in async_fakes.CASES.values()}[fakes]
    adapter, _, _ = build(rounds)
    return async_fakes.run(async_fakes.drain(adapter.arun(fakes.run_request(**kwargs))))


@pytest.fixture(params=sorted(CASES))
def fakes(request):
    return CASES[request.param]


@pytest.fixture(params=[_sync, _async], ids=["sync", "async"])
def drive(request):
    return request.param


def _structured(events: list[Any]) -> StructuredOutputEvent:
    return next(e for e in events if isinstance(e, StructuredOutputEvent))


# --- the answer is checked against what was sent -----------------------------------


def test_a_null_for_a_formerly_optional_property_is_a_valid_answer(fakes, drive):
    answer = {"name": "a", "nick": None}
    events = drive(fakes, [fakes.text_round(json.dumps(answer))], schema=LOOSE)

    event = _structured(events)
    assert event.valid is True, event.problems
    assert event.data == answer  # never edited
    assert events[-1].status is TerminalStatus.OK


def test_an_answer_that_breaks_the_strict_schema_is_still_invalid(fakes, drive):
    """Checking against the sent schema is stricter, not looser: a missing key is a problem."""
    events = drive(fakes, [fakes.text_round(json.dumps({"name": "a"}))], schema=LOOSE)

    event = _structured(events)
    assert event.valid is False
    assert any("nick" in problem for problem in event.problems)


def test_a_wrong_type_is_still_invalid(fakes, drive):
    events = drive(
        fakes, [fakes.text_round(json.dumps({"name": 7, "nick": None}))], schema=LOOSE
    )
    assert _structured(events).valid is False


# --- $defs --------------------------------------------------------------------------

NESTED = {
    "title": "Order",
    "type": "object",
    "properties": {
        "customer": {"$ref": "#/$defs/Customer"},
        "lines": {"type": "array", "items": {"$ref": "#/$defs/Line"}},
    },
    "required": ["customer"],
    "$defs": {
        "Customer": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            "required": ["name"],
        },
        "Line": {
            "type": "object",
            "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
            "required": ["sku", "qty"],
        },
    },
}


@pytest.mark.parametrize("keyword", ["$defs", "definitions"])
def test_the_strict_rewrite_reaches_the_definitions(keyword):
    loose = {**{k: v for k, v in NESTED.items() if k != "$defs"}, keyword: NESTED["$defs"]}
    strict = to_openai_strict(loose)

    customer = strict[keyword]["Customer"]
    assert customer["additionalProperties"] is False
    assert customer["required"] == ["name", "email"]
    assert customer["properties"]["email"]["type"] == ["string", "null"]
    assert customer["properties"]["name"]["type"] == "string"
    assert strict[keyword]["Line"]["additionalProperties"] is False
    # The caller's schema is not edited in place.
    assert "additionalProperties" not in loose[keyword]["Customer"]


def test_the_diagnostic_sees_the_definitions_too():
    issues = openai_strict_issues(NESTED)
    assert any("$defs.Customer" in issue and "email" in issue for issue in issues)
    assert openai_strict_issues(to_openai_strict(NESTED)) == ()


def test_the_rewrite_is_idempotent_and_leaves_a_strict_schema_alone():
    strict = to_openai_strict(NESTED)
    assert to_openai_strict(strict) == strict


def test_a_pydantic_nested_model_goes_out_strict(fakes):
    pydantic = pytest.importorskip("pydantic")

    class Customer(pydantic.BaseModel):
        name: str
        email: str = ""

    class Order(pydantic.BaseModel):
        customer: Customer
        note: str = ""

    loose = Order.model_json_schema()
    assert "Customer" in loose["$defs"]

    factory = fakes.Factory([fakes.text_round("{}")])
    fakes.drive(fakes.run_request(schema=loose), factory)
    params = factory.clients[0].calls[0]
    sent = (
        params["text"]["format"]["schema"]
        if "text" in params
        else params["response_format"]["json_schema"]["schema"]
    )

    assert openai_strict_issues(sent) == ()
    assert sent["$defs"]["Customer"]["additionalProperties"] is False
    assert sorted(sent["$defs"]["Customer"]["required"]) == ["email", "name"]


def test_validation_is_unchanged_for_a_schema_validated_as_given():
    """``validate_instance`` itself did not move: the loose schema still rejects a null."""
    valid, _ = validate_instance({"name": "a", "nick": None}, LOOSE)
    assert valid is False
    valid, _ = validate_instance({"name": "a", "nick": None}, to_openai_strict(LOOSE))
    assert valid is True


# --- the LangChain leaf --------------------------------------------------------------


@dataclass
class Decision:
    """A schema with defaults, which is what a null must not overwrite."""

    keep: list[str]
    rationale: str = "none given"
    tags: list[str] = field(default_factory=list)


def _leaf(tmp_path, runtime, answer):
    pytest.importorskip("langchain_core")
    from modelpass.connections import Connection, CredentialRef
    from modelpass.langchain_adapter import ChatSubpass
    from modelpass.runtimes import Runtime
    from modelpass.testing import FakeAdapter, fake_bridge, structured
    from modelpass.types import AuthMode

    runtime = Runtime(runtime)
    api = runtime is not Runtime.OPENAI_SDK and runtime is not Runtime.ANTHROPIC_SDK
    kwargs: dict[str, Any] = {}
    if api:
        kwargs["credential_ref"] = CredentialRef.parse("env:MODELPASS_TEST_KEY")
        kwargs["model"] = "a-model"
    if runtime is Runtime.OPENAI_COMPATIBLE:
        kwargs["base_url"] = "http://localhost:11434/v1"
        kwargs["verified_capabilities"] = chat_fakes.all_verified()
    adapter = FakeAdapter([structured(answer, valid=True)], runtime=runtime)
    bridge, _, _ = fake_bridge(
        connections=[
            Connection(
                name="conn",
                runtime=runtime,
                auth_mode=AuthMode.API_KEY if api else AuthMode.SUBSCRIPTION,
                **kwargs,
            )
        ],
        home=tmp_path / "modelpass",
        adapter=adapter,
        # Named by the API connections and never used: the adapter is a fake.
        env={"MODELPASS_TEST_KEY": "not-a-real-key"},
    )
    return ChatSubpass(connection="conn", bridge=bridge), adapter


@pytest.mark.parametrize("runtime", ["openai-sdk", "openai-api", "openai-compatible"])
def test_the_leaf_strips_introduced_nulls_on_every_runtime_that_rewrites(tmp_path, runtime):
    from langchain_core.messages import HumanMessage

    answer = {"keep": ["a.md"], "rationale": None, "tags": None}
    model, _ = _leaf(tmp_path, runtime, answer)

    result = model.bind_tools([Decision], tool_choice="Decision").invoke(
        [HumanMessage(content="q")]
    )
    args = result.tool_calls[0]["args"]

    assert args == {"keep": ["a.md"]}
    decision = Decision(**args)
    assert decision.rationale == "none given"
    assert decision.tags == []


@pytest.mark.parametrize("runtime", ["anthropic-sdk", "anthropic-api", "google-api"])
def test_the_leaf_leaves_nulls_alone_where_nothing_was_rewritten(tmp_path, runtime):
    from langchain_core.messages import HumanMessage

    answer = {"keep": ["a.md"], "rationale": None, "tags": []}
    model, _ = _leaf(tmp_path, runtime, answer)

    result = model.bind_tools([Decision], tool_choice="Decision").invoke(
        [HumanMessage(content="q")]
    )
    assert result.tool_calls[0]["args"]["rationale"] is None
