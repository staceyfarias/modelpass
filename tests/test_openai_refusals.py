"""A refusal is not an empty success.

OpenAI models decline in-band. On the Responses API the answer is a ``message``
item whose content is a ``refusal`` part (``ResponseOutputRefusal``: ``type``
and ``refusal``) while ``status`` is still ``completed``; on Chat Completions
the words arrive in ``delta.refusal`` and ``finish_reason`` is still ``stop``
(openai 2.32.0, both types read 2026-11-13). Neither adapter read either field,
so a refused call ended ``ok`` with no text -- and with ``schema=`` the reason
given was that no structured answer arrived, which sends a caller looking for a
bug in their schema.

Now a refusal is ``status=error`` carrying the model's own words, the same
verdict ``anthropic-api`` gives ``stop_reason == "refusal"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import test_adapter_openai_api as responses_fakes
import test_adapter_openai_compatible as chat_fakes
import test_async_adapters as async_fakes
from modelpass.types import StructuredOutputEvent, TerminalEvent, TerminalStatus, TextDeltaEvent

REFUSAL = "I can't help with that request."
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


@dataclass
class RefusalPart:
    refusal: str
    type: str = "refusal"

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {"type": "refusal", "refusal": self.refusal}


def responses_round() -> Any:
    return responses_fakes.Round(
        response=responses_fakes.FakeResponse(
            output=[responses_fakes.MessageItem(content=[RefusalPart(REFUSAL)])]
        ),
        events=[responses_fakes.StreamEvent(type="response.refusal.delta", delta=REFUSAL)],
    )


def chat_round() -> Any:
    half = len(REFUSAL) // 2
    chunk, choice, delta = chat_fakes.FakeChunk, chat_fakes.FakeChoice, chat_fakes.FakeDelta
    return chat_fakes.Round(
        chunks=[
            chunk(choices=[choice(delta=delta(refusal=REFUSAL[:half]))]),
            chunk(choices=[choice(delta=delta(refusal=REFUSAL[half:]))]),
            chunk(choices=[choice(delta=delta(), finish_reason="stop")]),
        ]
    )


CASES = {
    "openai-api": (responses_fakes, responses_round),
    "openai-compatible": (chat_fakes, chat_round),
}


def _sync(fakes: Any, round_: Any, **kwargs: Any) -> list[Any]:
    return fakes.drive(fakes.run_request(**kwargs), fakes.Factory([round_]))


def _async(fakes: Any, round_: Any, **kwargs: Any) -> list[Any]:
    build = {module: case for case, module in async_fakes.CASES.values()}[fakes]
    adapter, _, _ = build([round_])
    return async_fakes.run(async_fakes.drain(adapter.arun(fakes.run_request(**kwargs))))


@pytest.fixture(params=sorted(CASES))
def case(request):
    return CASES[request.param]


@pytest.fixture(params=[_sync, _async], ids=["sync", "async"])
def drive(request):
    return request.param


def test_a_refusal_ends_the_run_as_an_error_carrying_the_models_words(case, drive):
    fakes, round_ = case
    events = drive(fakes, round_())

    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert terminal.status is TerminalStatus.ERROR
    assert "refused" in (terminal.reason or "")
    assert REFUSAL in (terminal.reason or "")
    # The refusal is the reason, not an answer: it is never passed off as text.
    assert not [e for e in events if isinstance(e, TextDeltaEvent)]


def test_a_refusal_under_a_schema_is_reported_as_a_refusal(case, drive):
    fakes, round_ = case
    events = drive(fakes, round_(), schema=SCHEMA)

    terminal = events[-1]
    assert terminal.status is TerminalStatus.ERROR
    assert REFUSAL in (terminal.reason or "")
    assert not [e for e in events if isinstance(e, StructuredOutputEvent)]
