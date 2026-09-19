"""The stdlib schema module: normalization, the structural check, and the one
place the two runtimes disagree (D13).

Every assertion here is offline and pure. The constants that describe *vendor*
behavior are checked against the committed live fixtures in
``test_structured_output.py``, not asserted from memory here.
"""

from __future__ import annotations

import pytest

from modelpass.errors import InvalidSchema
from modelpass.schema import (
    as_object_schema,
    build_structured_event,
    normalize_schema,
    openai_strict_issues,
    resolve_schema_name,
    schema_title,
    to_openai_strict,
    validate_instance,
)
from modelpass.types import StructuredOutputEvent

# The loose shape a Python dataclass with defaults converts to. Accepted by
# anthropic-sdk, rejected by openai-sdk -- both verified live 2026-08-17.
LOOSE = {
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
                "required": ["chunk_id", "text"],
            },
        },
    },
    "required": ["context", "status", "excerpts"],
}


# --- normalization ---------------------------------------------------------------


def test_a_bare_properties_mapping_is_wrapped():
    assert as_object_schema({"name": {"type": "string"}}) == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }


def test_an_empty_mapping_is_an_empty_object_schema():
    assert as_object_schema({}) == {"type": "object", "properties": {}}


def test_tooldef_and_schema_make_the_same_guess():
    """The wrapping rule is shared, not reimplemented (D13)."""
    from modelpass.tools import ToolDef

    bare = {"topic": {"type": "string"}}
    tool = ToolDef(name="t", description="d", parameters=bare)
    assert tool.json_schema() == as_object_schema(bare)


def test_normalize_fills_in_the_envelope():
    normalized = normalize_schema({"properties": {"a": {"type": "string"}}})
    assert normalized["type"] == "object"
    assert normalized["properties"] == {"a": {"type": "string"}}


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ({"type": "array", "items": {}}, "must be 'object'"),
        ({"type": "object", "properties": []}, "'properties' must be a mapping"),
        ({"type": "object", "properties": {}, "required": "a"}, "'required' must be a list"),
    ],
)
def test_normalize_refuses_unusable_schemas(bad, fragment):
    with pytest.raises(InvalidSchema) as exc:
        normalize_schema(bad)
    assert fragment in str(exc.value)


def test_normalize_refuses_a_non_mapping():
    with pytest.raises(InvalidSchema):
        normalize_schema(["not", "a", "schema"])  # type: ignore[arg-type]


def test_normalize_refuses_a_schema_that_cannot_be_json():
    """It is going to be serialized either way; better here than at spawn time."""
    with pytest.raises(InvalidSchema) as exc:
        normalize_schema({"type": "object", "properties": {"a": {"default": object()}}})
    assert "JSON-serializable" in str(exc.value)


# --- naming ----------------------------------------------------------------------


def test_the_callers_name_wins_over_the_title():
    """The tool name is a wire-protocol detail; modelpass never overrides it."""
    assert resolve_schema_name(LOOSE, "CallerChosenName") == "CallerChosenName"


def test_the_title_is_the_fallback():
    assert resolve_schema_name(LOOSE, None) == "ExtractionResult"
    assert resolve_schema_name(LOOSE, "   ") == "ExtractionResult"


def test_no_name_is_a_real_answer_not_an_invented_one():
    assert resolve_schema_name({"type": "object", "properties": {}}, None) == ""
    assert schema_title({"title": "   "}) is None


# --- the structural check --------------------------------------------------------


def test_a_conforming_instance_passes():
    instance = {
        "context": "c",
        "status": "found",
        "excerpts": [{"chunk_id": "c1", "text": "t", "score": 1}],
    }
    valid, problems = validate_instance(instance, LOOSE)
    assert valid and problems == ()


def test_a_missing_required_property_is_named_with_its_path():
    valid, problems = validate_instance({"context": "c", "status": "found"}, LOOSE)
    assert not valid
    assert problems == ("$: missing required property 'excerpts'",)


def test_a_wrong_type_is_named_with_its_path():
    instance = {"context": 7, "status": "found", "excerpts": []}
    valid, problems = validate_instance(instance, LOOSE)
    assert not valid
    assert problems == ("$.context: expected type string, got integer",)


def test_an_enum_violation_is_reported():
    instance = {"context": "c", "status": "maybe", "excerpts": []}
    valid, problems = validate_instance(instance, LOOSE)
    assert not valid
    assert "not one of" in problems[0]


def test_nested_array_items_are_checked_by_index():
    instance = {
        "context": "c",
        "status": "found",
        "excerpts": [{"chunk_id": "c1", "text": "t"}, {"text": "t2"}],
    }
    valid, problems = validate_instance(instance, LOOSE)
    assert not valid
    assert problems == ("$.excerpts[1]: missing required property 'chunk_id'",)


def test_additional_properties_false_is_enforced():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": False,
    }
    valid, problems = validate_instance({"a": "x", "b": 1}, schema)
    assert not valid
    assert problems == ("$: unexpected property 'b'",)


def test_additional_properties_is_permitted_by_default():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    valid, _ = validate_instance({"a": "x", "b": 1}, schema)
    assert valid


def test_a_type_list_admits_either():
    schema = {"type": "object", "properties": {"a": {"type": ["string", "null"]}}}
    assert validate_instance({"a": None}, schema)[0]
    assert validate_instance({"a": "x"}, schema)[0]
    assert not validate_instance({"a": 1}, schema)[0]


def test_a_bool_is_not_a_number():
    """JSON has no bool-is-a-number rule. Python does; do not inherit it."""
    schema = {"type": "object", "properties": {"n": {"type": "number"}}}
    assert not validate_instance({"n": True}, schema)[0]


def test_an_integer_satisfies_number():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}}
    assert validate_instance({"n": 3}, schema)[0]


def test_composition_subtrees_are_skipped_not_failed():
    """A subtree modelpass cannot check reports no violation, never a false one."""
    schema = {
        "type": "object",
        "properties": {"a": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }
    assert validate_instance({"a": [1, 2, 3]}, schema)[0]


def test_an_unrecognized_type_name_is_not_a_violation():
    schema = {"type": "object", "properties": {"a": {"type": "date-thing"}}}
    assert validate_instance({"a": "2026-08-17"}, schema)[0]


def test_constraint_keywords_are_not_checked_and_do_not_fail():
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 10}}}
    assert validate_instance({"n": 1}, schema)[0]


# --- the openai strict subset ----------------------------------------------------


def test_the_loose_schema_is_flagged_for_openai():
    issues = openai_strict_issues(LOOSE)
    joined = " | ".join(issues)
    assert "additionalProperties" in joined
    # the optional 'score' on the nested item, named with its path
    assert "$.excerpts[]" in joined
    assert "'score'" in joined


def test_a_strict_schema_has_no_issues():
    assert openai_strict_issues(to_openai_strict(LOOSE)) == ()


def test_to_openai_strict_makes_optional_fields_nullable_rather_than_mandatory():
    """OpenAI's own recipe: no notion of optional, so absence becomes null."""
    strict = to_openai_strict(LOOSE)
    item = strict["properties"]["excerpts"]["items"]
    assert item["required"] == ["chunk_id", "text", "score"]
    assert item["properties"]["score"]["type"] == ["number", "null"]
    # a property that was already required keeps its plain type
    assert item["properties"]["chunk_id"]["type"] == "string"
    assert item["additionalProperties"] is False


def test_to_openai_strict_does_not_mutate_the_callers_schema():
    before = repr(LOOSE)
    to_openai_strict(LOOSE)
    assert repr(LOOSE) == before


def test_to_openai_strict_leaves_composition_alone():
    schema = {
        "type": "object",
        "properties": {"a": {"anyOf": [{"type": "string"}]}},
        "required": [],
    }
    out = to_openai_strict(schema)
    assert out["properties"]["a"] == {"anyOf": [{"type": "string"}]}


# --- building the event ----------------------------------------------------------


def test_a_parsed_answer_becomes_a_valid_event_with_no_failure():
    data = {"context": "c", "status": "found", "excerpts": []}
    event, failure = build_structured_event(LOOSE, parsed=data, schema_name="X")
    assert failure is None
    assert isinstance(event, StructuredOutputEvent)
    assert event.data == data and event.valid and event.schema_name == "X"
    assert event.raw  # re-encoded when the runtime gave us only the object


def test_raw_text_is_parsed_when_the_runtime_gave_only_text():
    event, failure = build_structured_event(
        LOOSE, raw='{"context":"c","status":"found","excerpts":[]}'
    )
    assert failure is None
    assert event.data["status"] == "found"


def test_a_structurally_invalid_answer_is_still_an_answer():
    """The vendor constrained it; modelpass reports and never alters `data`."""
    data = {"context": "c", "status": "nope"}
    event, failure = build_structured_event(LOOSE, parsed=data)
    assert failure is None, "an invalid answer is not a failed run"
    assert event.data == data
    assert not event.valid
    assert event.problems


def test_unparseable_text_is_reported_and_fails_the_run():
    event, failure = build_structured_event(LOOSE, raw="I'm sorry, I can't do that.")
    assert event is not None and event.data is None
    assert event.raw == "I'm sorry, I can't do that."
    assert not event.valid
    assert failure is not None and "not JSON" in failure
    assert "I'm sorry" in failure


def test_nothing_at_all_yields_no_event_and_a_failure():
    """Never an empty dict: modelpass does not invent the answer."""
    event, failure = build_structured_event(LOOSE, raw="   ")
    assert event is None
    assert failure is not None and "no structured answer" in failure
