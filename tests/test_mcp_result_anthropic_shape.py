"""A handler's MCP image reaches the Messages API as a Messages image block.

``ToolDef`` lets a handler return the MCP envelope, and an MCP image block is
``{"type": "image", "data": <base64>, "mimeType": ...}``. ``anthropic-api`` put
the handler's ``content`` list into ``tool_result.content`` unchanged, but the
Messages API spells an image ``{"type": "image", "source": {"type": "base64",
"media_type": ..., "data": ...}}`` (``ImageBlockParam`` / ``Base64ImageSourceParam``,
anthropic 0.97.0, read 2026-11-13). Same ``type``, different body: the request
was rejected, on the turn after the tool had already run.
"""

from __future__ import annotations

from typing import Any

import pytest

import test_adapter_anthropic_api as fakes
import test_async_adapters as async_fakes
from modelpass.adapters._mcp_result import anthropic_tool_result_content
from modelpass.tools import ToolDef
from modelpass.types import TerminalStatus, ToolResultEvent

PNG = "iVBORw0KGgo="
MCP_IMAGE = {"type": "image", "data": PNG, "mimeType": "image/png"}
MESSAGES_IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": PNG},
}


def test_an_mcp_image_block_becomes_a_messages_image_block():
    payload = {"content": [{"type": "text", "text": "the chart"}, MCP_IMAGE]}
    assert anthropic_tool_result_content(payload) == [
        {"type": "text", "text": "the chart"},
        MESSAGES_IMAGE,
    ]
    assert payload["content"][1] == MCP_IMAGE  # the handler's own result is not edited


def test_a_block_already_in_messages_shape_is_left_alone():
    assert anthropic_tool_result_content({"content": [MESSAGES_IMAGE]}) == [MESSAGES_IMAGE]


def test_text_keeps_only_what_the_messages_api_reads():
    block = {"type": "text", "text": "hi", "annotations": {"audience": ["user"]}}
    assert anthropic_tool_result_content({"content": [block, "bare"]}) == [
        {"type": "text", "text": "hi"},
        {"type": "text", "text": "bare"},
    ]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "audio", "data": "AAAA", "mimeType": "audio/wav"},
        {"type": "resource_link", "uri": "file:///x", "name": "x"},
        {"type": "image", "mimeType": "image/png"},
        {"type": "image", "data": PNG},
        17,
    ],
)
def test_a_block_the_messages_api_has_no_shape_for_becomes_a_text_marker(block):
    """Never dropped silently and never sent in a shape that fails the request."""
    (out,) = anthropic_tool_result_content({"content": [block]})
    assert out["type"] == "text"
    assert out["text"]


def test_no_content_is_an_empty_list():
    assert anthropic_tool_result_content({}) == []
    assert anthropic_tool_result_content({"content": None}) == []


def _image_tool() -> ToolDef:
    return ToolDef(
        name="look_up",
        description="Look something up.",
        parameters={"type": "object", "properties": {}},
        handler=lambda arguments: {"content": [{"type": "text", "text": "chart"}, MCP_IMAGE]},
    )


def _sent_tool_result(calls: list[dict[str, Any]]) -> dict[str, Any]:
    (block,) = calls[1]["messages"][-1]["content"]
    assert block["type"] == "tool_result"
    return block


def test_the_adapter_sends_the_converted_image_back_to_the_model():
    factory = fakes.Factory([fakes.tool_round("call_1", "look_up", {}), fakes.text_round("done")])
    events = fakes.drive(fakes.run_request(tools=(_image_tool(),)), factory)

    assert _sent_tool_result(factory.clients[0].calls)["content"] == [
        {"type": "text", "text": "chart"},
        MESSAGES_IMAGE,
    ]
    # The event is still flattened text with a marker, as it always was.
    assert next(e for e in events if isinstance(e, ToolResultEvent)).content == "chart\n[image]"
    assert events[-1].status is TerminalStatus.OK


def test_the_async_face_sends_the_same():
    build, _ = async_fakes.CASES["anthropic-api"]
    adapter, factory, _ = build(
        [fakes.tool_round("call_1", "look_up", {}), fakes.text_round("done")]
    )
    async_fakes.run(async_fakes.drain(adapter.arun(fakes.run_request(tools=(_image_tool(),)))))
    assert _sent_tool_result(factory.clients[0].calls)["content"][1] == MESSAGES_IMAGE
