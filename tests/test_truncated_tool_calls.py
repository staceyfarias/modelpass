"""A tool call the model did not finish asking for is never run.

A caller's handler has side effects. Two ways an adapter used to run one on a
call that was not whole:

* **The turn was cut short.** On Chat Completions the arguments stream as
  fragments of a JSON string; a turn that hits the token ceiling ends
  ``finish_reason="length"`` with ``{"table": "us`` in hand. The loop entered on
  *any* assembled call, the fragment failed to parse, the fallback was ``{}``,
  and ``delete_rows({})`` ran -- after which the run could end ``ok``. The same
  shape exists on Responses (``status="incomplete"`` with a ``function_call``
  item) and on Gemini (a function call beside ``MAX_TOKENS`` or
  ``MALFORMED_FUNCTION_CALL``).
* **The arguments were not JSON at all**, on a turn that did end normally. ``{}``
  was substituted there too, and a handler was called with arguments nobody
  sent.

The rules held here, on both faces:

1. A turn that ended truncated, filtered or failed runs **no** tool, and ends
   on the terminal that turn deserves.
2. Arguments that do not parse as a JSON object are answered with a failed tool
   result **without calling the handler**, so the model can ask again.
3. A call with no arguments at all (``""``) is still a call with ``{}``.
"""

from __future__ import annotations

from typing import Any

import pytest

import test_adapter_anthropic_api as anthropic_fakes
import test_adapter_google_api as google_fakes
import test_adapter_openai_api as responses_fakes
import test_adapter_openai_compatible as chat_fakes
import test_async_adapters as async_fakes
from modelpass.tools import ToolDef
from modelpass.types import TerminalStatus, ToolCallEvent, ToolResultEvent


class Handler:
    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def __call__(self, arguments: Any) -> str:
        self.seen.append(dict(arguments))
        return "deleted"

    def tool(self) -> ToolDef:
        return ToolDef(
            name="delete_rows",
            description="Delete rows from a table.",
            parameters={"type": "object", "properties": {"table": {"type": "string"}}},
            handler=self,
        )


def _sync(fakes: Any, rounds: list[Any], handler: Handler) -> list[Any]:
    return fakes.drive(fakes.run_request(tools=[handler.tool()]), fakes.Factory(rounds))


def _async(fakes: Any, rounds: list[Any], handler: Handler) -> list[Any]:
    build = {module: case for case, module in async_fakes.CASES.values()}[fakes]
    adapter, _, _ = build(rounds)
    request = fakes.run_request(tools=[handler.tool()])
    return async_fakes.run(async_fakes.drain(adapter.arun(request)))


@pytest.fixture(params=[_sync, _async], ids=["sync", "async"])
def drive(request):
    return request.param


# --- the scripted turns -------------------------------------------------------------


def chat_call(arguments: str, finish_reason: str | None) -> Any:
    """One chat-completions tool call whose arguments are exactly ``arguments``."""
    f = chat_fakes
    half = len(arguments) // 2

    def fragment(**kwargs: Any) -> Any:
        return f.FakeChunk(
            choices=[f.FakeChoice(delta=f.FakeDelta(tool_calls=[f.FakeToolCallDelta(**kwargs)]))]
        )

    return f.Round(
        chunks=[
            fragment(
                index=0, id="call_1", function=f.FakeFunction(name="delete_rows", arguments="")
            ),
            fragment(index=0, function=f.FakeFunction(arguments=arguments[:half])),
            fragment(index=0, function=f.FakeFunction(arguments=arguments[half:])),
            f.FakeChunk(choices=[f.FakeChoice(delta=f.FakeDelta(), finish_reason=finish_reason)]),
            f.FakeChunk(
                choices=[],
                usage=f.FakeUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
            ),
        ]
    )


def responses_call(arguments: str, *, incomplete: str | None = None) -> Any:
    f = responses_fakes
    return f.Round(
        response=f.FakeResponse(
            output=[f.FunctionCallItem(call_id="call_1", name="delete_rows", arguments=arguments)],
            status="incomplete" if incomplete else "completed",
            incomplete_details=f.IncompleteDetails(reason=incomplete) if incomplete else None,
        )
    )


def google_call(finish_reason: str) -> Any:
    f = google_fakes
    return f.Round(
        chunks=[
            f.chunk(
                f.FakePart(
                    function_call=f.FakeFunctionCall(name="delete_rows", args={}, id="call_1")
                ),
                finish_reason=finish_reason,
                usage=f.FakeUsage(prompt_token_count=8, candidates_token_count=4),
            )
        ]
    )


def anthropic_call(stop_reason: str) -> Any:
    f = anthropic_fakes
    return f.Round(
        message=f.FakeMessage(
            content=[f.ToolUseBlockOut(id="call_1", name="delete_rows", input={})],
            stop_reason=stop_reason,
            usage=f.FakeUsage(input_tokens=8, output_tokens=4),
        )
    )


TRUNCATED = '{"table": "us'

CUT_SHORT = {
    "openai-compatible length": (chat_fakes, lambda: chat_call(TRUNCATED, "length")),
    "openai-compatible content_filter": (
        chat_fakes,
        lambda: chat_call('{"table": "users"}', "content_filter"),
    ),
    "openai-api incomplete": (
        responses_fakes,
        lambda: responses_call(TRUNCATED, incomplete="max_output_tokens"),
    ),
    "google-api MAX_TOKENS": (google_fakes, lambda: google_call("MAX_TOKENS")),
    "google-api MALFORMED_FUNCTION_CALL": (
        google_fakes,
        lambda: google_call("MALFORMED_FUNCTION_CALL"),
    ),
    "google-api SAFETY": (google_fakes, lambda: google_call("SAFETY")),
    "anthropic-api max_tokens": (anthropic_fakes, lambda: anthropic_call("max_tokens")),
}


@pytest.mark.parametrize("name", sorted(CUT_SHORT))
def test_a_turn_cut_short_runs_no_tool(name, drive):
    fakes, round_ = CUT_SHORT[name]
    handler = Handler()
    # A second round is scripted so that a loop which wrongly continues is seen
    # running the handler, rather than tripping over a missing round.
    events = drive(fakes, [round_(), fakes.text_round("done")], handler)

    assert handler.seen == []
    assert not [e for e in events if isinstance(e, (ToolCallEvent, ToolResultEvent))]
    terminal = events[-1]
    assert terminal.status is TerminalStatus.ERROR
    # ... and it says what happened to the turn, not that a tool was undeclared.
    assert "did not declare" not in (terminal.reason or "")


@pytest.mark.parametrize(
    "name", ["openai-compatible length", "openai-api incomplete", "anthropic-api max_tokens"]
)
def test_a_turn_truncated_at_the_ceiling_says_so(name, drive):
    fakes, round_ = CUT_SHORT[name]
    events = drive(fakes, [round_(), fakes.text_round("done")], Handler())
    assert "max_output_tokens" in (events[-1].reason or "")


UNPARSEABLE = {
    "openai-compatible": (chat_fakes, lambda raw: chat_call(raw, "tool_calls")),
    "openai-api": (responses_fakes, lambda raw: responses_call(raw)),
}


@pytest.mark.parametrize("name", sorted(UNPARSEABLE))
@pytest.mark.parametrize("raw", [TRUNCATED, "not json", '["a", "list"]', '"a string"'])
def test_arguments_that_are_not_a_json_object_never_reach_the_handler(name, raw, drive):
    fakes, round_ = UNPARSEABLE[name]
    handler = Handler()
    events = drive(fakes, [round_(raw), fakes.text_round("done")], handler)

    assert handler.seen == []
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert "delete_rows" in results[0].content
    # The model is told, and the run goes on: it can ask again.
    assert events[-1].status is TerminalStatus.OK


@pytest.mark.parametrize("name", sorted(UNPARSEABLE))
@pytest.mark.parametrize("raw", ["", "{}", '{"table": "users"}'])
def test_whole_arguments_still_reach_the_handler(name, raw, drive):
    fakes, round_ = UNPARSEABLE[name]
    handler = Handler()
    events = drive(fakes, [round_(raw), fakes.text_round("done")], handler)

    assert handler.seen == [{"table": "users"} if "users" in raw else {}]
    assert events[-1].status is TerminalStatus.OK
