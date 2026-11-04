"""Files on the API runtimes, and who decides which types are read (2026-11-04).

modelpass decides whether a runtime can carry a file at all. The vendor decides
which types it reads, so there is no list here to keep in step with anyone:

1. **Any well-formed media type travels.** Each adapter picks a wire shape by
   family; a type the vendor does not read is the vendor's to refuse.
2. **A refusal is typed.** A 4xx that rejects the request, on a call carrying a
   file, raises :class:`DocumentRejected` with the vendor's own words. Nothing
   else changes: no file, or an auth or quota status, keeps its ordinary path.
3. **Where a vendor publishes input types, they can be asked for.**
   :meth:`Bridge.input_support` reports what the vendor said, and says so plainly
   when no vendor answer exists.

The shared promises (refused elsewhere, user turns only, no bytes on disk) are in
``test_document_blocks.py``.
"""

from __future__ import annotations

import base64
import typing
from dataclasses import replace
from types import SimpleNamespace

import pytest

import test_adapter_anthropic_api as anthropic_fake
import test_adapter_google_api as google_fake
import test_adapter_openai_api as openai_fake
from modelpass.adapters.anthropic_api import AnthropicAPIAdapter, conversation_param
from modelpass.adapters.base import RunRequest, document_rejection
from modelpass.adapters.google_api import content_items
from modelpass.adapters.openai import OpenAIAdapter
from modelpass.adapters.openai_api import input_items
from modelpass.errors import CapabilityNotSupported, DocumentRejected
from modelpass.preflight import plan_launch
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter
from modelpass.types import DocumentBlock, Message, Role, TerminalEvent, TextBlock
from test_adapter_anthropic_api import run_request
from test_document_blocks import _api_connection, _bridge, _connection_for

PDF = b"%PDF-1.4 not a real file, only bytes to carry"
PDF_B64 = base64.b64encode(PDF).decode("ascii")
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def document(**kwargs) -> DocumentBlock:
    return DocumentBlock(data=PDF, **{"title": "handbook", **kwargs})


def turn(*blocks) -> tuple[Message, ...]:
    return (Message(Role.USER, [*blocks, TextBlock("q")]),)


# --- 1. the wire, by family ---------------------------------------------------------


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


def test_openai_api_names_a_file_by_its_type_and_never_invents_a_suffix():
    def filename(block: DocumentBlock) -> str:
        return input_items(run_request(messages=turn(block)))[0]["content"][0]["filename"]

    assert filename(document(title=None)) == "document.pdf"
    assert filename(document(title="notes.pdf")) == "notes.pdf"
    assert filename(document(media_type="text/csv")) == "handbook.csv"
    assert filename(document(media_type="application/x-unknown")) == "handbook"


def test_openai_api_sends_an_image_as_an_input_image():
    block = document(media_type="image/png")
    part = input_items(run_request(messages=turn(block)))[0]["content"][0]
    assert part == {
        "type": "input_image",
        "detail": "auto",
        "image_url": f"data:image/png;base64,{PDF_B64}",
    }


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


def test_google_api_carries_any_type_in_the_same_part():
    part = content_items(run_request(messages=turn(document(media_type=DOCX))))[0]["parts"][0]
    assert part == {"inline_data": {"data": PDF, "mime_type": DOCX}}


def test_anthropic_api_picks_its_three_shapes_by_family():
    image = document(media_type="image/png", cache_control={"type": "ephemeral"})
    text = DocumentBlock(data="héllo".encode(), media_type="text/plain", title="n")
    other = document(media_type=DOCX)
    content = conversation_param(run_request(messages=turn(image, text, other)))[0]["content"]
    assert content[0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PDF_B64},
        "cache_control": {"type": "ephemeral"},
    }
    assert content[1] == {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "héllo"},
        "title": "n",
    }
    assert content[2]["type"] == "document"
    assert content[2]["source"] == {"type": "base64", "media_type": DOCX, "data": PDF_B64}


def test_anthropic_api_refuses_a_text_file_that_is_not_utf8():
    block = DocumentBlock(data=b"\xff\xfe", media_type="text/plain")
    with pytest.raises(ValueError, match="UTF-8"):
        conversation_param(run_request(messages=turn(block)))


def test_a_text_turn_is_unchanged_on_both_runtimes():
    messages = (Message(Role.USER, [TextBlock("a"), TextBlock("b")]),)
    request = run_request(messages=messages)
    assert input_items(request) == [{"role": "user", "content": "a\n\nb"}]
    assert content_items(request) == [{"role": "user", "parts": [{"text": "a\n\nb"}]}]


def test_the_installed_sdks_type_the_shapes_modelpass_sends():
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    from anthropic.types import ImageBlockParam, PlainTextSourceParam
    from openai.types.responses import ResponseInputFileParam, ResponseInputImageParam

    assert {"type", "filename", "file_data"} <= set(
        typing.get_type_hints(ResponseInputFileParam)
    )
    assert {"type", "detail", "image_url"} <= set(
        typing.get_type_hints(ResponseInputImageParam)
    )
    assert {"source", "type", "cache_control"} == set(typing.get_type_hints(ImageBlockParam))
    assert {"data", "media_type", "type"} == set(typing.get_type_hints(PlainTextSourceParam))


# --- 2. the block --------------------------------------------------------------------


def test_any_well_formed_media_type_is_a_document_and_a_malformed_one_is_not():
    for media_type in ("text/csv", "image/png", DOCX, "application/vnd.x+json"):
        assert document(media_type=media_type).media_type == media_type
    for media_type in ("", "pdf", "text/", "/plain", "text/plain; charset=utf-8"):
        with pytest.raises(ValueError, match="type/subtype"):
            document(media_type=media_type)


def test_from_path_reads_the_suffix_from_pythons_table_and_never_guesses(tmp_path):
    table = tmp_path / "rows.csv"
    table.write_text("a,b", encoding="utf-8")
    # text/csv, not the application/vnd.ms-excel the Windows registry can answer.
    assert DocumentBlock.from_path(table).media_type == "text/csv"
    unknown = tmp_path / "notes.zzz"
    unknown.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="pass media_type="):
        DocumentBlock.from_path(unknown)
    assert DocumentBlock.from_path(unknown, media_type="text/x-z").media_type == "text/x-z"


# --- 3. a vendor refusal is typed ----------------------------------------------------


class VendorError(Exception):
    def __init__(self, status_code: int, text: str = "unsupported file type") -> None:
        super().__init__(text)
        self.status_code = status_code


def _anthropic(error, messages):
    factory = anthropic_fake.Factory(
        [anthropic_fake.Round(message=anthropic_fake.FakeMessage(content=[]), raises=error)]
    )
    return anthropic_fake.drive(anthropic_fake.run_request(messages=messages), factory)


def _openai(error, messages):
    factory = openai_fake.Factory(
        [openai_fake.Round(response=openai_fake.FakeResponse(), raises=error)]
    )
    return openai_fake.drive(openai_fake.run_request(messages=messages), factory)


def _google(error, messages):
    factory = google_fake.Factory([google_fake.Round(raises=error)])
    return google_fake.drive(google_fake.run_request(messages=messages), factory)


RUNTIMES = [
    pytest.param(_anthropic, "anthropic-api", id="anthropic-api"),
    pytest.param(_openai, "openai-api", id="openai-api"),
    pytest.param(_google, "google-api", id="google-api"),
]


@pytest.mark.parametrize("status_code", [400, 413, 415, 422])
@pytest.mark.parametrize(("run", "runtime"), RUNTIMES)
def test_a_vendor_refusal_of_a_call_carrying_a_file_is_typed(run, runtime, status_code):
    block = document(media_type=DOCX)
    with pytest.raises(DocumentRejected) as excinfo:
        run(VendorError(status_code), turn(block))
    error = excinfo.value
    assert isinstance(error, CapabilityNotSupported)
    assert error.runtime == runtime
    assert error.capability == "document_input"
    assert error.media_types == (DOCX,)
    assert error.status_code == status_code
    assert "unsupported file type" in str(error)
    # The vendor's words are quoted; the cause is not asserted on their behalf.
    assert "says whether the file was the cause" in str(error)


@pytest.mark.parametrize(("run", "runtime"), RUNTIMES)
def test_an_error_that_is_not_a_refusal_keeps_its_ordinary_terminal(run, runtime):
    events = run(VendorError(401, "bad key"), turn(document()))
    assert isinstance(events[-1], TerminalEvent)
    assert "bad key" in (events[-1].reason or "")
    assert events[-1].status_code == 401


@pytest.mark.parametrize(("run", "runtime"), RUNTIMES)
def test_a_refusal_with_no_file_on_the_call_is_not_about_a_file(run, runtime):
    events = run(VendorError(400, "bad parameter"), (Message(Role.USER, "hello"),))
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].status_code == 400


def test_a_rejection_lists_each_type_once_and_bounds_the_vendors_message():
    request = run_request(messages=turn(document(), document(), document(media_type=DOCX)))
    error = document_rejection(Runtime.OPENAI_API, request, 400, VendorError(400, "x" * 5000))
    assert error is not None
    assert error.media_types == ("application/pdf", DOCX)
    assert len(str(error)) < 1000


# --- 4. asking first -----------------------------------------------------------------


def _capabilities(*, image: bool, pdf: bool) -> SimpleNamespace:
    return SimpleNamespace(
        capabilities=SimpleNamespace(
            image_input=SimpleNamespace(supported=image),
            pdf_input=SimpleNamespace(supported=pdf),
        )
    )


def _anthropic_answering(monkeypatch, info) -> AnthropicAPIAdapter:
    client = SimpleNamespace(models=SimpleNamespace(retrieve=lambda model: info))
    monkeypatch.setattr(AnthropicAPIAdapter, "_client", lambda self, connection: client)
    return AnthropicAPIAdapter()


def test_anthropic_api_reports_what_the_model_record_says(monkeypatch):
    adapter = _anthropic_answering(monkeypatch, _capabilities(image=True, pdf=True))
    request = replace(run_request(), model="claude-x")
    assert adapter.input_modalities(request) == ("text", "image", "pdf")
    adapter = _anthropic_answering(monkeypatch, _capabilities(image=False, pdf=False))
    assert adapter.input_modalities(request) == ("text",)


def test_anthropic_api_says_nothing_when_the_record_has_no_capabilities(monkeypatch):
    adapter = _anthropic_answering(monkeypatch, SimpleNamespace(capabilities=None))
    assert adapter.input_modalities(replace(run_request(), model="claude-x")) is None


def test_the_installed_anthropic_sdk_still_carries_the_fields_read():
    pytest.importorskip("anthropic")
    from anthropic.types import ModelInfo
    from anthropic.types.model_capabilities import ModelCapabilities

    assert "capabilities" in ModelInfo.model_fields
    assert {"image_input", "pdf_input"} <= set(ModelCapabilities.model_fields)


class FakeCodex:
    """``AppServerClient`` with scripted ``model/list`` pages."""

    pages: typing.ClassVar[list[dict]] = []
    closed = 0

    def __init__(self, **kwargs) -> None:
        self.calls: list[tuple[str, dict]] = []

    def start(self) -> None:
        pass

    def request(self, method, params=None, *, timeout=None):
        assert method == "model/list"
        return type(self).pages.pop(0)

    def close(self) -> None:
        type(self).closed += 1


def _codex(monkeypatch, pages, model) -> tuple[OpenAIAdapter, RunRequest]:
    FakeCodex.pages = list(pages)
    FakeCodex.closed = 0
    monkeypatch.setattr("modelpass.adapters.openai.AppServerClient", FakeCodex)
    monkeypatch.setattr(OpenAIAdapter, "_resolve_binary", lambda self, request: "codex")
    connection = _connection_for(Runtime.OPENAI_SDK)
    request = RunRequest(
        connection=connection, messages=(), plan=plan_launch(connection, {}), model=model
    )
    return OpenAIAdapter(), request


SOL = {"id": "gpt-5.6-sol", "model": "gpt-5.6-sol", "inputModalities": ["text", "image"]}
DEFAULT = {"id": "gpt-6-astra", "isDefault": True, "inputModalities": ["text"]}


def test_codex_reads_model_list_across_pages_and_closes_the_child(monkeypatch):
    pages = [{"data": [DEFAULT], "nextCursor": "more"}, {"data": [SOL], "nextCursor": None}]
    adapter, request = _codex(monkeypatch, pages, "gpt-5.6-sol")
    assert adapter.input_modalities(request) == ("text", "image")
    assert FakeCodex.closed == 1


def test_codex_answers_for_the_default_model_when_none_is_named(monkeypatch):
    adapter, request = _codex(monkeypatch, [{"data": [SOL, DEFAULT]}], None)
    assert adapter.input_modalities(request) == ("text",)


def test_codex_reports_no_answer_rather_than_the_clients_default(monkeypatch):
    silent = {"id": "m", "model": "m"}
    adapter, request = _codex(monkeypatch, [{"data": [silent]}], "m")
    assert adapter.input_modalities(request) is None


def test_codex_refuses_to_answer_for_a_model_it_does_not_list(monkeypatch):
    adapter, request = _codex(monkeypatch, [{"data": [SOL]}], "nope")
    with pytest.raises(ValueError, match="does not name 'nope'"):
        adapter.input_modalities(request)


class Answering(FakeAdapter):
    answer: object = None

    def input_modalities(self, request):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _bridge_answering(tmp_path, conn, answer):
    bridge, _ = _bridge(tmp_path, conn)
    adapter = Answering(script=[], runtime=conn.runtime)
    adapter.answer = answer
    bridge.adapter_for = lambda runtime: adapter
    return bridge


def test_input_support_reports_what_the_vendor_said(tmp_path):
    bridge = _bridge_answering(tmp_path, _api_connection(), ("text", "image", "pdf"))
    support = bridge.input_support("api")
    assert support.queried and support.carries_files
    assert support.inputs == ("text", "image", "pdf")
    assert support.model == "claude-opus-5"
    assert support.to_dict()["runtime"] == "anthropic-api"
    assert bridge.input_support("api", model="other").model == "other"


def test_input_support_separates_carrying_a_file_from_what_the_model_takes(tmp_path):
    conn = _connection_for(Runtime.OPENAI_SDK)
    bridge = _bridge_answering(tmp_path, conn, ("text", "image"))
    support = bridge.input_support("api")
    assert support.queried and not support.carries_files
    assert support.inputs == ("text", "image")


def test_input_support_says_when_no_vendor_answer_exists(tmp_path):
    conn = _connection_for(Runtime.OPENAI_API)
    support = _bridge_answering(tmp_path, conn, None).input_support("api")
    assert support.carries_files and not support.queried
    assert support.inputs == ()
    assert "answers at send time" in support.detail


def test_input_support_reports_a_failed_query_instead_of_raising(tmp_path):
    bridge = _bridge_answering(tmp_path, _api_connection(), RuntimeError("no route"))
    support = bridge.input_support("api")
    assert not support.queried
    assert "no route" in support.detail


def test_a_langchain_file_part_is_refused_rather_than_dropped():
    pytest.importorskip("langchain_core")
    from modelpass.langchain_adapter import _convert_content

    part = {"type": "file", "base64": PDF_B64, "mime_type": "application/pdf"}
    with pytest.raises(ValueError, match="document part"):
        _convert_content([part, {"type": "text", "text": "q"}])
