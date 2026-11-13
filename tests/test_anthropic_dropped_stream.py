"""A Messages stream that stops before ``message_delta`` is not an answer.

``stop_reason`` arrives in ``message_delta``, near the end of the stream. A
server -- or a proxy in front of one -- that closes the SSE body cleanly after a
few content deltas leaves the SDK holding a message whose ``stop_reason`` is
still ``None`` (anthropic 0.97.0, ``get_final_message()``, driven here over an
``httpx.MockTransport`` on 2026-11-13). ``anthropic-api`` used to report that as
``ok``: half a sentence, recorded as a finished call.

Unlike a chat-completions chunk, where ``finish_reason`` is a field a server may
simply never set, the Messages API always states a stop reason on a message that
ended. Its absence is the stream not having ended.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import test_adapter_anthropic_api as fakes
from modelpass.adapters import anthropic_api
from modelpass.adapters.anthropic_api import terminal_status
from modelpass.types import TerminalEvent, TerminalStatus, TextDeltaEvent

httpx = pytest.importorskip("httpx")


def _sse(*events: dict[str, Any]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


START = {
    "type": "message_start",
    "message": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": fakes.MODEL,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
    },
}
TEXT = (
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "half a th"},
    },
)
END = (
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 4},
    },
    {"type": "message_stop"},
)


def _drive(body: bytes, monkeypatch) -> list[Any]:
    """The adapter over the real SDK client, answered from a scripted SSE body."""
    default = anthropic_api._default_client

    def scripted(**kwargs: Any) -> Any:
        client = default(**kwargs)
        client._client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, content=body
                )
            )
        )
        return client

    monkeypatch.setattr(anthropic_api, "_default_client", scripted)
    return list(fakes.adapter(None).run(fakes.run_request()))


@fakes.NEEDS_SDK
def test_a_stream_that_drops_before_message_delta_is_an_error(monkeypatch):
    events = _drive(_sse(START, *TEXT), monkeypatch)

    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["half a th"]
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert terminal.status is TerminalStatus.ERROR
    assert "stop_reason" in (terminal.reason or "")


@fakes.NEEDS_SDK
def test_the_same_stream_carried_to_its_end_is_ok(monkeypatch):
    events = _drive(_sse(START, *TEXT, *END), monkeypatch)
    assert events[-1].status is TerminalStatus.OK


def test_no_stop_reason_is_never_ok():
    class Message:
        stop_reason = None

    status, reason = terminal_status(Message())
    assert status is TerminalStatus.ERROR
    assert reason
