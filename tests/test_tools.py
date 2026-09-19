"""Offline tests for caller-supplied tool definitions (D12).

No network, no vendor package, no credential -- a ``ToolDef`` is stdlib all the
way down, which is most of the point of it.
"""

from __future__ import annotations

import pytest

from modelpass.errors import ConfigError, InvalidTool
from modelpass.tools import ToolDef, normalize_tools


def echo(args):
    return args.get("text", "")


SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def test_a_valid_tool_round_trips_to_the_wire_form_without_its_handler():
    tool = ToolDef(name="echo", description="Echo the text back.", parameters=SCHEMA, handler=echo)
    data = tool.to_dict()
    assert data == {"name": "echo", "description": "Echo the text back.", "parameters": SCHEMA}
    # A callable is not protocol (D11): the wire form must stay JSON-safe.
    assert "handler" not in data


@pytest.mark.parametrize(
    "name",
    [
        "",
        "9lives",  # must start with a letter
        "has space",
        "has.dot",  # a dot would be ambiguous in a qualified tool name
        "x" * 65,
        None,
    ],
)
def test_unusable_tool_names_are_refused_at_construction(name):
    with pytest.raises(InvalidTool):
        ToolDef(name=name, description="d", parameters={}, handler=echo)


def test_a_double_underscore_in_a_name_is_refused_with_the_reason():
    """``mcp__<server>__<tool>`` is how both runtimes address tools."""
    with pytest.raises(InvalidTool, match="mcp__<server>__<tool>"):
        ToolDef(name="my__tool", description="d", parameters={}, handler=echo)


def test_a_tool_needs_a_description_the_model_can_read():
    with pytest.raises(InvalidTool, match="non-empty description"):
        ToolDef(name="echo", description="   ", parameters={}, handler=echo)


def test_parameters_must_be_a_mapping():
    with pytest.raises(InvalidTool, match="JSON Schema mapping"):
        ToolDef(name="echo", description="d", parameters=["text"], handler=echo)


def test_a_non_callable_handler_is_refused():
    with pytest.raises(InvalidTool, match="must be callable"):
        ToolDef(name="echo", description="d", parameters={}, handler="not a function")


def test_invalid_tool_is_a_config_error_so_one_except_catches_setup_mistakes():
    assert issubclass(InvalidTool, ConfigError)


# --- schema normalization ------------------------------------------------------


def test_an_empty_schema_becomes_a_valid_empty_object():
    tool = ToolDef(name="ping", description="d", handler=echo)
    assert tool.json_schema() == {"type": "object", "properties": {}}


def test_a_bare_properties_mapping_is_wrapped_rather_than_rejected():
    """Callers assembling schemas from an existing registry often lack the envelope."""
    tool = ToolDef(
        name="echo", description="d", parameters={"text": {"type": "string"}}, handler=echo
    )
    assert tool.json_schema() == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
    }


def test_a_full_schema_is_left_alone_including_its_required_list():
    tool = ToolDef(name="echo", description="d", parameters=SCHEMA, handler=echo)
    assert tool.json_schema() == SCHEMA


def test_json_schema_does_not_mutate_the_caller_s_mapping():
    original = dict(SCHEMA)
    ToolDef(name="echo", description="d", parameters=original, handler=echo).json_schema()
    assert original == SCHEMA


# --- normalize_tools -----------------------------------------------------------


def test_normalize_accepts_the_dict_form_and_tool_defs_together():
    tools = normalize_tools(
        [
            {"name": "a", "description": "d", "parameters": {}, "handler": echo},
            ToolDef(name="b", description="d", handler=echo),
        ]
    )
    assert [t.name for t in tools] == ["a", "b"]
    assert tools[0].handler is echo


def test_normalize_of_none_is_no_tools():
    assert normalize_tools(None) == ()


def test_duplicate_names_are_an_error_not_a_last_one_wins_merge():
    with pytest.raises(InvalidTool, match="duplicate tool name"):
        normalize_tools(
            [
                ToolDef(name="echo", description="one", handler=echo),
                ToolDef(name="echo", description="two", handler=echo),
            ]
        )


def test_unknown_keys_in_the_dict_form_are_rejected_not_ignored():
    with pytest.raises(InvalidTool, match="unknown tool key"):
        normalize_tools([{"name": "a", "description": "d", "schema": {}}])


def test_something_that_is_not_a_tool_at_all_is_refused():
    with pytest.raises(InvalidTool, match="cannot read a tool"):
        normalize_tools([echo])
