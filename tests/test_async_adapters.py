"""The four API adapters' **native** ``arun`` (R1, ticket 1.13).

Native means two things and this file tests both: the vendor's own async client
drives the stream, and the in-adapter tool loop **awaits a coroutine handler on
the caller's own loop** -- no thread, no ``run_coroutine_threadsafe``, no
private event loop. That second one is the whole reason the ticket exists:
A downstream agent host carries about 80 lines of exactly that
marshalling, and an async application's tool handlers are coroutines over its
own session and pool.

Each adapter's scripted rounds are imported from its own sync test module rather
than re-invented here: the same ``Round``, the same blocks, the same usage
shapes, served through an async transport. That is deliberate. If the async
face read a *different* fake it could agree with a fake and disagree with the
vendor, and the sync files are where those shapes are pinned against the
installed SDKs.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

import test_adapter_anthropic_api as anthropic_fakes
import test_adapter_google_api as google_fakes
import test_adapter_openai_api as openai_fakes
import test_adapter_openai_compatible as compatible_fakes
from modelpass.adapters.anthropic_api import AnthropicAPIAdapter
from modelpass.adapters.base import AsyncRun
from modelpass.adapters.google_api import GoogleAPIAdapter
from modelpass.adapters.openai_api import OpenAIAPIAdapter
from modelpass.adapters.openai_compatible import OpenAICompatibleAdapter
from modelpass.tools import ToolDef
from modelpass.types import (
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)

TOOL_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def run(coro: Any) -> Any:
    return asyncio.run(coro)


async def drain(stream) -> list:
    return [event async for event in stream]


def kinds(events) -> list[str]:
    return [event.type for event in events]


# --- async transports over the sync files' own scripts ------------------------------


class AsyncRounds:
    """The shared half of all four fakes: hand out scripted rounds, in order."""

    def __init__(self, rounds: list[Any]) -> None:
        self.rounds = list(rounds)
        self.calls: list[dict[str, Any]] = []
        self.closed = 0
        #: Which thread each round was served on. The assertion that the native
        #: face never left the caller's loop.
        self.threads: list[int] = []

    def take(self, params: dict[str, Any]) -> Any:
        self.calls.append(params)
        self.threads.append(threading.get_ident())
        if not self.rounds:
            raise AssertionError("the adapter asked for more rounds than were scripted")
        round_ = self.rounds.pop(0)
        if round_.raises is not None:
            raise round_.raises
        return round_


class AnthropicAsyncStream:
    def __init__(self, client: AnthropicAsyncClient, round_: Any) -> None:
        self._client = client
        self._round = round_

    async def __aenter__(self) -> AnthropicAsyncStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._client.closed += 1

    async def __aiter__(self):
        for event in self._round.events:
            yield event

    async def get_final_message(self) -> Any:
        return self._round.message


class AnthropicAsyncClient(AsyncRounds):
    def __init__(self, rounds: list[Any]) -> None:
        super().__init__(rounds)
        self.messages = self

    def stream(self, **params: Any) -> AnthropicAsyncStream:
        return AnthropicAsyncStream(self, self.take(params))


class OpenAIAsyncStream:
    def __init__(self, client: OpenAIAsyncClient, round_: Any) -> None:
        self._client = client
        self._round = round_

    async def __aenter__(self) -> OpenAIAsyncStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._client.closed += 1

    async def __aiter__(self):
        for event in self._round.frames():
            yield event


class OpenAIAsyncClient(AsyncRounds):
    def __init__(self, rounds: list[Any]) -> None:
        super().__init__(rounds)
        self.responses = self

    def stream(self, **params: Any) -> OpenAIAsyncStream:
        return OpenAIAsyncStream(self, self.take(params))


class ChunkStream:
    """What both chunk-shaped SDKs hand back: an async iterator with ``aclose``."""

    def __init__(self, client: AsyncRounds, chunks: list[Any]) -> None:
        self._client = client
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self._client.closed += 1


class GoogleAsyncClient(AsyncRounds):
    """``client.aio`` -- the same attribute path as the sync client's."""

    def __init__(self, rounds: list[Any]) -> None:
        super().__init__(rounds)
        self.models = self

    async def generate_content_stream(self, **params: Any) -> ChunkStream:
        return ChunkStream(self, self.take(params).chunks)


class CompatibleAsyncClient(AsyncRounds):
    def __init__(self, rounds: list[Any]) -> None:
        super().__init__(rounds)
        self.chat = self
        self.completions = self

    async def create(self, **params: Any) -> ChunkStream:
        return ChunkStream(self, self.take(params).chunks)


class AsyncFactory:
    """The ``async_client_factory`` seam, recording how it was constructed."""

    def __init__(self, client_class: Any, rounds: list[Any] | None = None) -> None:
        self._client_class = client_class
        self.rounds = rounds or []
        self.kwargs: list[dict[str, Any]] = []
        self.clients: list[Any] = []

    def __call__(self, *, api_key: str, base_url: str | None) -> Any:
        self.kwargs.append({"api_key": api_key, "base_url": base_url})
        client = self._client_class(self.rounds)
        self.clients.append(client)
        return client


# --- one row per adapter ------------------------------------------------------------


def anthropic_case(rounds):
    factory = AsyncFactory(AnthropicAsyncClient, rounds)
    adapter = AnthropicAPIAdapter(async_client_factory=factory, env=anthropic_fakes.ENV)
    return adapter, factory, anthropic_fakes.run_request()


def openai_case(rounds):
    factory = AsyncFactory(OpenAIAsyncClient, rounds)
    adapter = OpenAIAPIAdapter(async_client_factory=factory, env=openai_fakes.ENV)
    return adapter, factory, openai_fakes.run_request()


def google_case(rounds):
    factory = AsyncFactory(GoogleAsyncClient, rounds)
    adapter = GoogleAPIAdapter(async_client_factory=factory, env=google_fakes.ENV)
    return adapter, factory, google_fakes.run_request()


def compatible_case(rounds):
    factory = AsyncFactory(CompatibleAsyncClient, rounds)
    adapter = OpenAICompatibleAdapter(
        async_client_factory=factory, env=compatible_fakes.ENV
    )
    return adapter, factory, compatible_fakes.run_request()


CASES = {
    "anthropic-api": (anthropic_case, anthropic_fakes),
    "openai-api": (openai_case, openai_fakes),
    "google-api": (google_case, google_fakes),
    "openai-compatible": (compatible_case, compatible_fakes),
}


@pytest.fixture(params=sorted(CASES))
def case(request):
    return CASES[request.param]


# --- what all four must do ----------------------------------------------------------


def test_every_api_adapter_streams_natively(case):
    build, fakes = case
    adapter, _factory, request = build([fakes.text_round("the answer")])
    events = run(drain(adapter.arun(request)))

    assert "text_delta" in kinds(events)
    assert "".join(e.text for e in events if isinstance(e, TextDeltaEvent)) == (
        "the answer"
    )
    assert any(isinstance(e, UsageEvent) for e in events)
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].status is TerminalStatus.OK


def test_arun_is_an_async_run_that_can_be_cancelled(case):
    build, fakes = case
    adapter, _, request = build([fakes.text_round("hello")])
    stream = adapter.arun(request)
    assert isinstance(stream, AsyncRun)

    async def go():
        first = await stream.__anext__()
        await stream.aclose()
        return first

    first = run(go())
    assert first is not None
    # ``aclose()`` is the cancel on this face, as closing the iterator is on the
    # other one. It cancels *this run's* state: until 2026-09-24 this read a
    # flag on the adapter, which every concurrent run shared -- the defect
    # tests/test_concurrent_runs.py holds. The run's own cancel is the one the
    # stream carries.
    assert stream._cancel.__self__.cancelled


def test_the_async_client_gets_the_connections_own_key_and_nothing_ambient(case):
    build, fakes = case
    adapter, factory, request = build([fakes.text_round("hi")])
    run(drain(adapter.arun(request)))
    (kwargs,) = factory.kwargs
    assert kwargs["api_key"] == fakes.KEY


def test_no_round_is_served_off_the_callers_loop(case):
    """Native, not the base class's worker: every round runs on the loop thread."""
    build, fakes = case
    adapter, factory, request = build([fakes.text_round("hi")])

    async def go():
        await drain(adapter.arun(request))
        return threading.get_ident()

    loop_thread = run(go())
    (client,) = factory.clients
    assert client.threads == [loop_thread]


def test_a_coroutine_tool_handler_is_awaited_on_the_callers_loop(case):
    """The line this whole ticket exists for.

    The handler is a coroutine that reads the running loop. If the adapter had
    driven it on a private loop or a worker thread -- which is exactly what the
    sync face must do -- the loop it saw would not be the caller's, and the two
    ids below would differ.
    """
    build, fakes = case
    seen: dict[str, Any] = {}

    async def handler(arguments: dict[str, Any]) -> str:
        seen["loop"] = asyncio.get_running_loop()
        seen["thread"] = threading.get_ident()
        seen["arguments"] = arguments
        return f"sunny in {arguments['city']}"

    tool = ToolDef(
        name="weather",
        description="the weather",
        parameters=TOOL_SCHEMA,
        handler=handler,
    )
    rounds = [
        tool_round_for(fakes, "call-1", "weather", {"city": "Lisbon"}),
        fakes.text_round("sunny"),
    ]
    adapter, _, _ = build(rounds)
    request = request_with_tools(fakes, (tool,))

    async def go():
        events = await drain(adapter.arun(request))
        return events, asyncio.get_running_loop(), threading.get_ident()

    events, loop, thread = run(go())

    assert seen["loop"] is loop, "the handler ran on a loop that was not the caller's"
    assert seen["thread"] == thread
    assert seen["arguments"] == {"city": "Lisbon"}
    call = next(e for e in events if isinstance(e, ToolCallEvent))
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert call.name == "weather"
    assert result.content == "sunny in Lisbon"
    assert not result.is_error


def test_a_raising_async_handler_becomes_a_failed_tool_result(case):
    build, fakes = case

    async def handler(arguments: dict[str, Any]) -> str:
        raise RuntimeError("the service is down")

    tool = ToolDef(
        name="weather",
        description="the weather",
        parameters=TOOL_SCHEMA,
        handler=handler,
    )
    rounds = [
        tool_round_for(fakes, "call-1", "weather", {"city": "Lisbon"}),
        fakes.text_round("sorry"),
    ]
    adapter, _, _ = build(rounds)
    request = request_with_tools(fakes, (tool,))
    events = run(drain(adapter.arun(request)))
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.is_error
    assert "the service is down" in result.content
    # The run continued rather than dying, which is ToolDef's promise.
    assert events[-1].status is TerminalStatus.OK


def test_a_sync_handler_still_works_on_the_async_face(case):
    build, fakes = case
    rounds = [
        tool_round_for(fakes, "call-1", "weather", {"city": "Lisbon"}),
        fakes.text_round("sunny"),
    ]
    tool = ToolDef(
        name="weather",
        description="the weather",
        parameters=TOOL_SCHEMA,
        handler=lambda arguments: f"sunny in {arguments['city']}",
    )
    adapter, _, _ = build(rounds)
    events = run(drain(adapter.arun(request_with_tools(fakes, (tool,)))))
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.content == "sunny in Lisbon"


def test_a_vendor_failure_is_a_terminal_on_the_async_face(case):
    build, fakes = case

    class Rejected(Exception):
        status_code = 400

    round_ = fakes.text_round("never streams")
    round_.raises = Rejected("the model was rejected")
    adapter, _, request = build([round_])
    events = run(drain(adapter.arun(request)))
    assert events[-1].status is TerminalStatus.ERROR
    assert "rejected" in (events[-1].reason or "")


def test_a_cancel_between_rounds_ends_the_run_as_cancelled(case):
    build, fakes = case
    adapter, _, request = build([fakes.text_round("hi")])
    adapter.cancel()
    events = run(drain(adapter.arun(request)))
    # ``arun`` clears the flag the way ``run`` does, so a cancel from *before*
    # the run does not stop it: the flag is per run, not per adapter.
    assert events[-1].status is TerminalStatus.OK


# --- helpers that paper over the four fakes' small differences ----------------------


def tool_round_for(fakes, call_id: str, name: str, arguments: dict[str, Any]):
    """``tool_round`` under each module's own signature."""
    if fakes is google_fakes:
        return fakes.tool_round(name, arguments, call_id=call_id)
    return fakes.tool_round(call_id, name, arguments)


def request_with_tools(fakes, tools):
    return fakes.run_request(tools=tools)
