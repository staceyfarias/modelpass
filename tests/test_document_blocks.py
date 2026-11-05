"""PDF documents as message content (2026-11-03).

The consumer that asked: a RAG ingestion prototype sends a whole PDF to Claude,
marks it as the cacheable prefix, and asks several questions against it. Before
this, modelpass carried text and nothing else, so the only way to do that was to
step around the library -- and with it the receipt, the guards and the run log.

Four promises, each held here:

1. **The block is the vendor's own shape.** :class:`DocumentBlock` serializes to
   the Messages API's ``DocumentBlockParam`` with a ``Base64PDFSourceParam``
   source, and the ``anthropic-api`` adapter sends it in the caller's order with
   its ``cache_control`` intact.
2. **Refused, never dropped.** A runtime with no document channel renders content
   as text, so a document there would vanish from the prompt and the model would answer a
   question about a file it never saw. The bridge refuses such a call with
   ``CapabilityNotSupported`` before preflight and before any network, and the
   flattening adapters refuse again if one reaches them by another door.
3. **User turns only.** A document in a system prompt or an assistant turn is a
   ``ValueError`` at construction.
4. **No PDF bytes on disk or in logs.** The run log, the receipt, the event
   stream and ``repr`` carry counts, sizes and hashes; a durable conversation
   refuses documents rather than store one.

The ``openai-api`` and ``google-api`` wire shapes are held in
``test_document_blocks_api.py``.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from modelpass import DocumentBlock as ExportedDocumentBlock
from modelpass.adapters.anthropic import split_messages
from modelpass.adapters.base import RunRequest, refuse_documents
from modelpass.adapters.google_api import flatten as google_flatten
from modelpass.adapters.openai_api import flatten as openai_flatten
from modelpass.bridge import Bridge
from modelpass.capabilities import Capability, CapabilityRegistry, Support
from modelpass.connections import Connection, CredentialRef, Guards, QuotaAction, QuotaPolicy
from modelpass.errors import CapabilityNotSupported, ConversationError
from modelpass.preflight import plan_launch
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore
from modelpass.testing import FakeAdapter, usage
from modelpass.types import (
    AuthMode,
    CacheControl,
    ContentBlock,
    DocumentBlock,
    Message,
    Role,
    TerminalEvent,
    TerminalStatus,
    TextBlock,
    TextDeltaEvent,
    count_cache_breakpoints,
    normalize_content,
    normalize_messages,
)
from test_adapter_anthropic_api import (
    ENV,
    Factory,
    drive,
    run_request,
    text_round,
)

#: A string that appears in the PDF's own content stream, so a raw copy of the
#: file anywhere on disk is as findable as a base64 one.
MARKER = "MODELPASS-DOCUMENT-MARKER-7F3A"


def tiny_pdf(text: str = MARKER) -> bytes:
    """A minimal, valid one-page PDF showing ``text``, with a correct xref."""
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


PDF = tiny_pdf()
PDF_B64 = base64.b64encode(PDF).decode("ascii")
EPHEMERAL = {"type": "ephemeral"}
DOCUMENT_RUNTIMES = (Runtime.ANTHROPIC_API, Runtime.OPENAI_API, Runtime.GOOGLE_API)


def document(**kwargs: Any) -> DocumentBlock:
    return DocumentBlock(data=PDF, **{"title": "handbook", **kwargs})


# --- 1. the block -----------------------------------------------------------------


def test_the_block_is_exported_and_is_a_content_block():
    assert ExportedDocumentBlock is DocumentBlock
    assert DocumentBlock.type == "document"
    assert isinstance(document(), ContentBlock)
    assert isinstance(TextBlock("x"), ContentBlock)


def test_construction_validates_data_media_type_and_title():
    block = DocumentBlock(data=bytearray(PDF), cache_control=EPHEMERAL)
    assert isinstance(block.data, bytes)
    assert block.media_type == "application/pdf"
    assert block.title is None
    assert block.cache_control == CacheControl()
    assert block.size_bytes == len(PDF)
    assert len(block.sha256) == 64
    with pytest.raises(ValueError, match="must not be empty"):
        DocumentBlock(data=b"")
    with pytest.raises(TypeError, match="must be bytes"):
        DocumentBlock(data=PDF_B64)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="type/subtype"):
        DocumentBlock(data=PDF, media_type="pdf")
    with pytest.raises(TypeError, match="title"):
        DocumentBlock(data=PDF, title=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ttl"):
        DocumentBlock(data=PDF, cache_control={"type": "ephemeral", "ttl": "2h"})


def test_to_dict_is_the_messages_api_wire_shape_with_one_line_of_base64():
    """``DocumentBlockParam`` with a ``Base64PDFSourceParam`` source, anthropic 0.97.0."""
    big = DocumentBlock(data=bytes(range(256)) * 64, title="big", cache_control=EPHEMERAL)
    wire = big.to_dict()
    assert wire == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(big.data).decode("ascii"),
        },
        "title": "big",
        "cache_control": {"type": "ephemeral"},
    }
    assert "\n" not in wire["source"]["data"]
    assert base64.b64decode(wire["source"]["data"], validate=True) == big.data
    assert DocumentBlock.coerce(wire) == big
    untitled = DocumentBlock(data=PDF).to_dict()
    assert "title" not in untitled and "cache_control" not in untitled


def test_coerce_refuses_what_it_would_otherwise_have_to_drop():
    wire = document().to_dict()
    assert DocumentBlock.coerce(document()) == document()
    with pytest.raises(ValueError, match="citations"):
        DocumentBlock.coerce({**wire, "citations": {"enabled": True}})
    with pytest.raises(ValueError, match="'url'"):
        DocumentBlock.coerce(
            {"type": "document", "source": {"type": "url", "url": "https://x.example/a.pdf"}}
        )
    with pytest.raises(ValueError, match="not valid base64"):
        DocumentBlock.coerce(
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "%%%"},
            }
        )
    with pytest.raises(ValueError, match="type/subtype"):
        DocumentBlock.coerce(
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "plain", "data": PDF_B64},
            }
        )
    with pytest.raises(ValueError, match="type 'document'"):
        DocumentBlock.coerce({"type": "text", "text": "x"})
    with pytest.raises(TypeError):
        DocumentBlock.coerce(PDF)  # type: ignore[arg-type]


def test_from_path_reads_a_pdf_and_titles_it_by_its_stem(tmp_path):
    path = tmp_path / "Employee Handbook.PDF"
    path.write_bytes(PDF)
    block = DocumentBlock.from_path(path, cache_control=EPHEMERAL)
    assert block.data == PDF
    assert block.title == "Employee Handbook"
    assert block.cache_control == CacheControl()
    assert DocumentBlock.from_path(str(path), title="HB").title == "HB"
    other = tmp_path / "notes.zzz"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="pass media_type="):
        DocumentBlock.from_path(other)


def test_repr_and_summary_name_the_file_and_never_carry_it():
    block = document(cache_control=EPHEMERAL)
    text = repr(block)
    assert "handbook" in text and str(len(PDF)) in text and block.sha256 in text
    assert MARKER not in text and PDF_B64 not in text
    assert MARKER not in repr(Message(Role.USER, [block, TextBlock("q")]))
    assert block.summary() == {
        "type": "document",
        "media_type": "application/pdf",
        "title": "handbook",
        "bytes": len(PDF),
        "sha256": block.sha256,
    }


# --- 2. the block in a message ----------------------------------------------------


def test_normalize_content_dispatches_documents_and_still_refuses_images():
    blocks = normalize_content([document().to_dict(), "Summarize", {"type": "text", "text": "!"}])
    assert isinstance(blocks, tuple)
    assert [type(b) for b in blocks] == [DocumentBlock, TextBlock, TextBlock]
    assert normalize_content(document()) == (document(),)
    with pytest.raises(ValueError, match="text content blocks only"):
        normalize_content([{"type": "image", "source": {}}])
    with pytest.raises(ValueError, match="DocumentBlock"):
        TextBlock.coerce(document().to_dict())


def test_message_text_is_text_blocks_only_and_documents_lists_the_files():
    first, second = document(), document(title="appendix")
    message = Message(Role.USER, [first, TextBlock("What is "), second, TextBlock("it?")])
    assert message.text == "What is it?"
    assert message.flat_text == "What is \n\nit?"
    assert message.documents == (first, second)
    assert message.blocks == (first, TextBlock("What is "), second, TextBlock("it?"))
    assert Message(Role.USER, "plain").documents == ()
    assert Message.coerce(message.to_dict()) == message


def test_breakpoints_on_documents_are_counted():
    message = Message(Role.USER, [document(cache_control=EPHEMERAL), TextBlock("q")])
    assert message.cache_breakpoints == 1
    assert count_cache_breakpoints(message.content) == 1
    request = RunRequest(
        connection=_api_connection(),
        messages=(message,),
        plan=plan_launch(_api_connection(), ENV),
    )
    assert request.cache_breakpoints == 1
    assert request.documents == (document(cache_control=EPHEMERAL),)


@pytest.mark.parametrize("role", [Role.SYSTEM, Role.ASSISTANT])
def test_a_document_is_refused_outside_a_user_message(role):
    with pytest.raises(ValueError, match="only be sent in a user message"):
        Message(role, [document(), TextBlock("x")])
    with pytest.raises(ValueError, match="only be sent in a user message"):
        normalize_messages(
            [
                {"role": role.value, "content": [document().to_dict()]},
                {"role": "user", "content": "q"},
            ]
        )


def test_a_document_in_the_system_prompt_is_refused_before_preflight(tmp_path):
    bridge, fake = _bridge(tmp_path, _api_connection())
    with pytest.raises(ValueError, match="only be sent in a user message"):
        bridge.ask("api", "q", system_prompt=[document()])  # type: ignore[list-item]
    assert fake.preflights == [] and fake.requests == []


# --- 3. refused on every runtime that cannot carry one ------------------------------


def _connection_for(runtime: Runtime) -> Connection:
    if runtime in (Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK, Runtime.GOOGLE_CLI):
        return Connection(
            name="api",
            runtime=runtime,
            auth_mode=AuthMode.SUBSCRIPTION,
            credential_ref=CredentialRef.native_login(),
            experimental=runtime is Runtime.GOOGLE_CLI,
        )
    return Connection(
        name="api",
        runtime=runtime,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:VENDOR_KEY"),
        base_url="https://host.example/v1" if runtime is Runtime.OPENAI_COMPATIBLE else None,
        experimental=runtime is Runtime.GOOGLE_SDK,
    )


def _api_connection(**kwargs) -> Connection:
    return Connection(
        name="api",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:MODELPASS_TEST_ANTHROPIC_KEY"),
        model="claude-opus-5",
        **kwargs,
    )


def _bridge(tmp_path: Path, conn: Connection, script=()) -> tuple[Bridge, FakeAdapter]:
    store = ConnectionStore(tmp_path / "modelpass-home")
    store.add(conn)
    fake = FakeAdapter(script=script, runtime=conn.runtime)
    registry = CapabilityRegistry()
    if conn.runtime is Runtime.OPENAI_COMPATIBLE:
        # As if `modelpass verify` had driven a chat against this endpoint, so
        # the call reaches the document gate rather than stopping at `chat`.
        registry = registry.refined(conn.runtime, {Capability.CHAT: True})
    bridge = Bridge(
        store=store,
        registry=registry,
        adapters={conn.runtime: fake},
        env={**ENV, "VENDOR_KEY": "sk-not-a-real-key-0123456789"},
    )
    return bridge, fake


@pytest.mark.parametrize(
    "runtime", [r for r in Runtime if r not in DOCUMENT_RUNTIMES], ids=lambda r: r.value
)
def test_a_runtime_with_no_document_channel_refuses_before_any_network(
    tmp_path, runtime
):
    bridge, fake = _bridge(tmp_path, _connection_for(runtime))
    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(bridge.chat(connection="api", message=[document(), TextBlock("Summarize")]))
    assert excinfo.value.capability == "document_input"
    assert excinfo.value.runtime == runtime.value
    assert "never flattened, never dropped" in str(excinfo.value)
    # Refused in the plan: no preflight, no run, nothing in the run log.
    assert fake.preflights == [] and fake.requests == []
    assert not bridge.run_log.path.exists()
    # A document in history is the same document.
    with pytest.raises(CapabilityNotSupported):
        bridge.ask(
            "api",
            "and the second question?",
            history=[
                Message(Role.USER, [document(), TextBlock("q1")]),
                Message(Role.ASSISTANT, "a1"),
            ],
        )
    assert fake.requests == []


def test_a_failover_target_that_cannot_take_the_document_is_refused_before_it_runs(
    tmp_path,
):
    """The second connection must read the same file, or it does not run at all."""
    store = ConnectionStore(tmp_path / "modelpass-home")
    store.add(
        replace(
            _api_connection(),
            guards=Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "sub")),
        )
    )
    store.add(replace(_connection_for(Runtime.ANTHROPIC_SDK), name="sub"))
    exhausted = TerminalEvent(
        status=TerminalStatus.QUOTA_EXHAUSTED,
        connection="",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        reason="allowance exhausted",
    )
    api = FakeAdapter([exhausted], runtime=Runtime.ANTHROPIC_API)
    agent = FakeAdapter([], runtime=Runtime.ANTHROPIC_SDK)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_API: api, Runtime.ANTHROPIC_SDK: agent},
        env=dict(ENV),
    )
    events = list(bridge.chat(connection="api", message=[document(), TextBlock("q")]))
    assert len(api.requests) == 1
    assert agent.preflights == [] and agent.requests == []
    assert events[-1].status is TerminalStatus.QUOTA_EXHAUSTED
    assert "failover to 'sub' could not run" in (events[-1].reason or "")
    assert "document_input" in (events[-1].reason or "")


def test_the_flattening_adapters_refuse_a_document_handed_to_them_directly():
    """The second lock: an adapter that renders text never joins a file into nothing."""
    blocks = (document(), TextBlock("q"))
    with pytest.raises(CapabilityNotSupported, match="google-api"):
        google_flatten(blocks)
    with pytest.raises(CapabilityNotSupported, match="openai-api"):
        openai_flatten(blocks)
    with pytest.raises(CapabilityNotSupported, match="openai-compatible"):
        openai_flatten(blocks, runtime=Runtime.OPENAI_COMPATIBLE)
    with pytest.raises(CapabilityNotSupported, match="anthropic-sdk"):
        split_messages((Message(Role.USER, list(blocks)),))
    with pytest.raises(CapabilityNotSupported, match="document_input"):
        refuse_documents(Runtime.OPENAI_SDK, [Message(Role.USER, list(blocks))])
    # Text is untouched by the lock.
    assert google_flatten((TextBlock("a"), TextBlock("b"))) == "a\n\nb"
    refuse_documents(Runtime.OPENAI_SDK, [Message(Role.USER, "fine")])


def test_the_codex_adapter_refuses_a_document_before_it_launches_anything():
    from modelpass.adapters.openai import OpenAIAdapter

    conn = _connection_for(Runtime.OPENAI_SDK)
    request = RunRequest(
        connection=conn,
        messages=(Message(Role.USER, [document(), TextBlock("q")]),),
        plan=plan_launch(conn, {}),
    )
    with pytest.raises(CapabilityNotSupported, match="document_input"):
        OpenAIAdapter().run(request)


def test_langchain_document_parts_become_documents_and_are_never_flattened_away():
    pytest.importorskip("langchain_core")
    from modelpass.langchain_adapter import _convert_content, _flatten_content

    converted = _convert_content([document().to_dict(), {"type": "text", "text": "q"}])
    assert converted == (document(), TextBlock("q"))
    with pytest.raises(ValueError, match="document part"):
        _flatten_content([document().to_dict(), {"type": "image_url", "image_url": "x"}])
    with pytest.raises(ValueError, match="citations"):
        _convert_content([{**document().to_dict(), "citations": {"enabled": True}}])


# --- 4. anthropic-api: on the wire, in order, breakpoint intact ---------------------


def test_anthropic_api_sends_the_document_block_with_its_cache_control_in_order():
    """The offline half of the ``document_input`` cell's evidence (2026-11-03)."""
    doc = document(cache_control=EPHEMERAL)
    messages = (
        Message(Role.SYSTEM, "Answer from the document only."),
        Message(Role.USER, [doc, TextBlock("What does it say?")]),
    )
    factory = Factory([text_round("It says hello.")])
    drive(run_request(messages=messages), factory)
    sent = factory.clients[0].calls[0]
    assert sent["system"] == [{"type": "text", "text": "Answer from the document only."}]
    assert sent["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_B64},
                    "title": "handbook",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "What does it say?"},
            ],
        }
    ]


def test_a_question_after_a_cached_document_keeps_the_document_first_on_every_turn():
    """The prototype's loop: one marked document, then several questions."""
    doc = document(cache_control=EPHEMERAL)
    for question in ("first?", "second?"):
        factory = Factory([text_round("ok")])
        drive(run_request(messages=(Message(Role.USER, [doc, TextBlock(question)]),)), factory)
        content = factory.clients[0].calls[0]["messages"][0]["content"]
        assert content[0]["type"] == "document" and content[0]["cache_control"] == EPHEMERAL
        assert content[1] == {"type": "text", "text": question}


# --- 5. receipts, the run log, events: counts and hashes, never bytes ----------------


def test_the_receipt_counts_documents_and_bytes_and_nothing_written_carries_the_file(tmp_path):
    bridge, fake = _bridge(
        tmp_path,
        _api_connection(prompt_cache="custom"),
        script=[TextDeltaEvent("answer"), usage(input_tokens=10, output_tokens=2)],
    )
    events = list(
        bridge.chat(
            connection="api",
            message=[document(cache_control=EPHEMERAL), TextBlock("Summarize")],
        )
    )
    receipt = events[0].receipt
    assert receipt.documents == 1
    assert receipt.document_bytes == len(PDF)
    assert receipt.cache_breakpoints_requested == 1
    assert receipt.to_dict()["documents"] == 1
    assert receipt.to_dict()["document_bytes"] == len(PDF)
    # The adapter got the document; nothing modelpass wrote or emitted did.
    assert fake.requests[0].documents == (document(cache_control=EPHEMERAL),)
    serialized = json.dumps([e.to_dict() for e in events])
    assert PDF_B64 not in serialized and MARKER not in serialized
    written = [p for p in (tmp_path / "modelpass-home").rglob("*") if p.is_file()]
    assert bridge.run_log.path in written
    for path in written:
        content = path.read_bytes()
        assert PDF_B64.encode() not in content, path
        assert MARKER.encode() not in content, path


def test_a_receipt_for_a_text_call_reports_no_documents(tmp_path):
    bridge, _ = _bridge(tmp_path, _api_connection(), script=[TextDeltaEvent("x")])
    receipt = bridge.preflight("api")
    assert receipt.documents == 0 and receipt.document_bytes == 0


def test_a_managed_conversation_refuses_a_document_rather_than_store_it(tmp_path):
    bridge, _ = _bridge(tmp_path, _api_connection())
    conversations = bridge.conversations(tmp_path / "chats", application="rag-poc")
    with pytest.raises(ConversationError, match="cannot hold a document"):
        conversations.create(
            connection="api",
            seed_history=[
                Message(Role.USER, [document(), TextBlock("q")]),
                Message(Role.ASSISTANT, "a"),
            ],
        )
    for path in (tmp_path / "chats").rglob("*"):
        if path.is_file():
            assert PDF_B64.encode() not in path.read_bytes()
            assert MARKER.encode() not in path.read_bytes()


# --- 6. the capability cell ---------------------------------------------------------


def test_document_input_is_supported_on_the_three_api_rows_and_nowhere_else():
    registry = CapabilityRegistry()
    assert Capability.DOCUMENT_INPUT.value == "document_input"
    for runtime in Runtime:
        expected = Support.SUPPORTED if runtime in DOCUMENT_RUNTIMES else Support.UNSUPPORTED
        assert registry.support(runtime, Capability.DOCUMENT_INPUT) is expected, runtime
        note = registry.note(runtime, Capability.DOCUMENT_INPUT) or ""
        assert "2026-11-0" in note, runtime
    note = registry.note(Runtime.OPENAI_SDK, Capability.DOCUMENT_INPUT) or ""
    assert "never flattened, never dropped" in note
    assert "openai/codex#1797" in note
    note = registry.note(Runtime.OPENAI_API, Capability.DOCUMENT_INPUT) or ""
    assert "test_openai_api_sends_the_document_as_an_" in note
    assert "offline evidence only" in note
    note = registry.note(Runtime.GOOGLE_API, Capability.DOCUMENT_INPUT) or ""
    assert "test_google_api_sends_the_document_as_inline_" in note
    assert "offline evidence only" in note
    note = registry.note(Runtime.ANTHROPIC_API, Capability.DOCUMENT_INPUT) or ""
    assert "anthropic 0.97.0" in note
    assert "DocumentBlockParam" in note and "Base64PDFSourceParam" in note
    assert (
        "test_anthropic_api_sends_the_document_block_with_its_cache_control_in_order" in note
    )
    assert "test_one_pdf_document_question" in note


def test_the_installed_sdk_types_the_shape_modelpass_sends():
    """The SDK half of the cell, re-read on every run where the SDK is installed."""
    pytest.importorskip("anthropic")
    import typing

    from anthropic.types import Base64PDFSourceParam, DocumentBlockParam

    hints = typing.get_type_hints(DocumentBlockParam)
    assert {"source", "type", "cache_control", "title"} <= set(hints)
    source_hints = typing.get_type_hints(Base64PDFSourceParam)
    assert {"data", "media_type", "type"} <= set(source_hints)
    assert set(document(cache_control=EPHEMERAL).to_dict()) <= set(hints)
