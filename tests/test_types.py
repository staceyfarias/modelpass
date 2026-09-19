from __future__ import annotations

import dataclasses
import json

import pytest

from modelpass.runtimes import Runtime
from modelpass.types import (
    EVENT_TYPES,
    AuthMode,
    GuardStopEvent,
    GuardWarningEvent,
    Message,
    Role,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    ThinkingEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    VendorEvent,
    normalize_messages,
)


def test_message_coerces_the_dict_form():
    message = Message.coerce({"role": "user", "content": "hi"})
    assert message == Message(Role.USER, "hi")
    assert message.to_dict() == {"role": "user", "content": "hi"}


def test_message_rejects_content_that_is_neither_text_nor_blocks():
    """v1 grew content blocks (R3); it did not grow arbitrary content."""
    with pytest.raises(TypeError):
        Message.coerce({"role": "user", "content": 7})
    with pytest.raises(ValueError, match="text content blocks only"):
        Message.coerce(
            {"role": "user", "content": [{"type": "image", "source": "..."}]}
        )


def test_message_rejects_missing_keys():
    with pytest.raises(ValueError, match="missing key"):
        Message.coerce({"role": "user"})


def test_normalize_messages_requires_a_user_turn():
    with pytest.raises(ValueError, match="at least one user message"):
        normalize_messages([{"role": "system", "content": "be nice"}])
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_messages([])


def test_token_usage_adds_and_totals():
    total = TokenUsage(10, 5, 2) + TokenUsage(1, 1, 1)
    assert (total.input_tokens, total.output_tokens, total.cached_input_tokens) == (11, 6, 3)
    assert total.total_tokens == 20


@pytest.mark.parametrize(
    "event",
    [
        TextDeltaEvent(text="hello"),
        ThinkingEvent(text="hmm"),
        UsageEvent(usage=TokenUsage(1, 2, 3)),
        GuardWarningEvent(
            guard="tokens", threshold=10, observed=11, connection="c", message="m"
        ),
        GuardStopEvent(guard="tokens", threshold=10, observed=11, connection="c", message="m"),
        TerminalEvent(
            status=TerminalStatus.OK,
            connection="c",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
        ),
        ToolCallEvent(name="look_up", arguments={"topic": "x"}, id="t1", server="caller"),
        ToolResultEvent(id="t1", name="look_up", content="found", is_error=False),
        VendorEvent(runtime=Runtime.OPENAI_SDK, name="turn.failed", data={"a": 1}),
    ],
)
def test_every_event_serializes_to_json_with_a_discriminator(event):
    data = event.to_dict()
    assert data["type"] == type(event).type
    assert EVENT_TYPES[data["type"]] is type(event)
    # Serializable as protocol (D11): no custom encoder needed.
    assert json.loads(json.dumps(data)) == data


def test_events_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TextDeltaEvent(text="a").text = "b"


def test_no_event_ever_reports_money():
    """Tokens, never dollars (D7): two of three providers draw on an allowance."""
    banned = ("cost", "usd", "dollar", "price", "spend", "credit")
    for cls in EVENT_TYPES.values():
        for field in dataclasses.fields(cls):
            assert not any(word in field.name.lower() for word in banned), (
                f"{cls.__name__}.{field.name} normalizes money; vendor cost fields "
                "belong in vendor_event"
            )


def test_tool_events_carry_no_reply_channel():
    """Runtime-executed semantics (D12): the caller observes, never responds.

    A field for handing a result back would be the first half of the deferred
    caller-side round-trip loop, built by accident. Guard the absence.
    """
    call_fields = {f.name for f in dataclasses.fields(ToolCallEvent)}
    assert call_fields == {"name", "arguments", "id", "server"}
    result_fields = {f.name for f in dataclasses.fields(ToolResultEvent)}
    assert result_fields == {"id", "name", "content", "is_error"}


def test_a_tool_call_and_its_result_correlate_by_id():
    call = ToolCallEvent(name="look_up", arguments={"topic": "x"}, id="t1", server="caller")
    result = ToolResultEvent(id="t1", name="look_up", content="found")
    assert call.id == result.id
    assert call.to_dict()["server"] == "caller"
    assert result.to_dict()["is_error"] is False


def test_a_tool_result_name_is_empty_rather_than_guessed():
    """Anthropic's ToolResultBlock carries only tool_use_id."""
    assert ToolResultEvent(id="t9", content="x").name == ""


def test_usage_event_reports_cumulative_totals():
    event = UsageEvent(usage=TokenUsage(1, 1), cumulative=TokenUsage(5, 5))
    assert event.to_dict()["cumulative"]["total_tokens"] == 10
