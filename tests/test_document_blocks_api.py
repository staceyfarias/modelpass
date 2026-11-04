"""PDF documents on ``openai-api`` and ``google-api`` (2026-11-04).

The offline half of each row's ``document_input`` evidence. The shared promises
(refused elsewhere, user turns only, no bytes on disk) are held in
``test_document_blocks.py``; this file holds the two wire shapes.
"""

from __future__ import annotations

import base64
import typing

import pytest

from modelpass.adapters.google_api import content_items
from modelpass.adapters.openai_api import input_items
from modelpass.types import DocumentBlock, Message, Role, TextBlock
from test_adapter_anthropic_api import run_request

PDF = b"%PDF-1.4 not a real file, only bytes to carry"
PDF_B64 = base64.b64encode(PDF).decode("ascii")


def document(**kwargs) -> DocumentBlock:
    return DocumentBlock(data=PDF, **{"title": "handbook", **kwargs})


def test_openai_api_sends_the_document_as_an_input_file_in_order():
    messages = (
        Message(Role.USER, [document(), TextBlock("What does it say?")]),
        Message(Role.ASSISTANT, "It says hello."),
        Message(Role.USER, "And then?"),
    )
    assert input_items(run_request(messages=messages)) == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "handbook.pdf",
                    "file_data": f"data:application/pdf;base64,{PDF_B64}",
                },
                {"type": "input_text", "text": "What does it say?"},
            ],
        },
        {"role": "assistant", "content": "It says hello."},
        {"role": "user", "content": "And then?"},
    ]


def test_openai_api_names_an_untitled_document():
    messages = (Message(Role.USER, [document(title=None), TextBlock("q")]),)
    part = input_items(run_request(messages=messages))[0]["content"][0]
    assert part["filename"] == "document.pdf"


def test_google_api_sends_the_document_as_inline_data_in_order():
    messages = (Message(Role.USER, [document(), TextBlock("What does it say?")]),)
    items = content_items(run_request(messages=messages))
    assert items == [
        {
            "role": "user",
            "parts": [
                {"inline_data": {"data": PDF, "mime_type": "application/pdf"}},
                {"text": "What does it say?"},
            ],
        }
    ]
    genai_types = pytest.importorskip("google.genai.types")
    content = genai_types.Content.model_validate(items[0])
    assert content.parts[0].inline_data.data == PDF
    assert content.parts[1].text == "What does it say?"


def test_a_text_turn_is_unchanged_on_both_runtimes():
    messages = (Message(Role.USER, [TextBlock("a"), TextBlock("b")]),)
    request = run_request(messages=messages)
    assert input_items(request) == [{"role": "user", "content": "a\n\nb"}]
    assert content_items(request) == [{"role": "user", "parts": [{"text": "a\n\nb"}]}]


def test_the_installed_openai_sdk_types_the_shape_modelpass_sends():
    pytest.importorskip("openai")
    from openai.types.responses import ResponseInputFileParam

    hints = typing.get_type_hints(ResponseInputFileParam)
    assert {"type", "filename", "file_data"} <= set(hints)


def test_a_langchain_file_part_is_refused_rather_than_dropped():
    pytest.importorskip("langchain_core")
    from modelpass.langchain_adapter import _convert_content

    part = {"type": "file", "base64": PDF_B64, "mime_type": "application/pdf"}
    with pytest.raises(ValueError, match="document part"):
        _convert_content([part, {"type": "text", "text": "q"}])
