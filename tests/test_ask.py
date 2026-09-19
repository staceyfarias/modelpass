"""Ticket 1.8: ``Bridge.ask`` -- one call, one answer (R11).

Ten one-shot call sites in a batch scraping tool each need a single blocking
answer, and
without this they each write their own drain loop plus a hand-rolled extractor
for whatever shape the answer arrived in. :meth:`~modelpass.Bridge.ask` drains
the stream once, in the library, and hands back a frozen
:class:`~modelpass.Answer`.

The contract these tests pin:

* ``ask`` never returns a half answer. A terminal that is not ``ok`` raises the
  same typed error ``chat``'s consumers already rely on -- ``GuardStop``,
  ``QuotaExhausted``, ``VendorRunFailed`` -- rather than returning whatever text
  happened to arrive first.
* ``schema=`` plus an ``ok`` terminal means ``structured`` is not ``None``,
  full stop. The absence of the ``structured_output`` event is a contract
  violation and is raised as one.
* ``ask`` and ``ChatSubpass.invoke`` are the library's two collected paths and
  they may not drift, so the last test in this file runs both over one script.
"""

from __future__ import annotations

import pytest

from modelpass.bridge import Answer
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.errors import (
    AdapterFailed,
    GuardStop,
    InvalidSchema,
    QuotaExhausted,
    VendorRunFailed,
)
from modelpass.runtimes import Runtime
from modelpass.testing import (
    FakeAdapter,
    fake_bridge,
    quota_exhausted,
    structured,
    tool_exchange,
    usage,
    vendor_failure,
)
from modelpass.tools import ToolDef
from modelpass.types import (
    AuthMode,
    Sampling,
    TerminalStatus,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)

SCHEMA = {
    "type": "object",
    "title": "Score",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
    "additionalProperties": False,
}


def connection(**kwargs) -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        **kwargs,
    )


def bridge_for(tmp_path, script=(), *, adapter=None, conn=None):
    return fake_bridge(
        connections=[conn or connection()],
        script=script,
        adapter=adapter,
        home=tmp_path / "modelpass",
    )


# --- 1. the collected shape -------------------------------------------------------


def test_ask_returns_the_whole_answer(tmp_path):
    bridge, _store, _adapter = bridge_for(
        tmp_path,
        [TextDeltaEvent(text="a git "), TextDeltaEvent(text="rebase"), usage(10, 4)],
    )
    answer = bridge.ask("claude-sub", "what is a rebase?")

    assert isinstance(answer, Answer)
    assert answer.text == "a git rebase"
    assert answer.status is TerminalStatus.OK
    assert answer.reason is None
    assert answer.structured is None
    assert answer.usage is not None
    assert answer.usage.total_tokens == 14
    assert answer.receipt.account == "fake-account"
    assert answer.tool_calls == ()
    assert answer.events[0].type == "receipt"
    assert answer.events[-1].type == "terminal"


def test_the_answer_is_frozen(tmp_path):
    bridge, _store, _adapter = bridge_for(tmp_path, [TextDeltaEvent(text="hi")])
    answer = bridge.ask("claude-sub", "hi")
    with pytest.raises(AttributeError):
        answer.text = "something else"  # type: ignore[misc]


def test_connection_and_message_may_be_passed_positionally(tmp_path):
    """The one-shot call site's whole point is that it is one line."""
    bridge, _store, _adapter = bridge_for(tmp_path, [TextDeltaEvent(text="ok")])
    assert bridge.ask("claude-sub", "go").text == "ok"
    assert bridge.ask(connection="claude-sub", message="go").text == "ok"


def test_ask_accepts_a_connection_object_like_chat_does(tmp_path):
    conn = connection()
    bridge, _store, _adapter = bridge_for(tmp_path, [TextDeltaEvent(text="ok")], conn=conn)
    assert bridge.ask(conn, "go").text == "ok"


def test_usage_is_the_final_cumulative_not_a_sum_of_deltas(tmp_path):
    bridge, _store, _adapter = bridge_for(
        tmp_path, [usage(10, 1), usage(5, 2), TextDeltaEvent(text="x")]
    )
    answer = bridge.ask("claude-sub", "go")
    assert answer.usage is not None
    assert answer.usage.input_tokens == 15
    assert answer.usage.output_tokens == 3


def test_tool_calls_are_collected_in_order(tmp_path):
    call, result = tool_exchange("search_notes", {"query": "rebase"}, result="found 3")
    second_call, second_result = tool_exchange("read", {"path": "a"}, call_id="call_2")

    def handler(_args):
        return "unused: the runtime runs the loop"

    tools = [
        ToolDef(
            name="search_notes",
            description="Search notes.",
            parameters={"type": "object", "properties": {}},
            handler=handler,
        )
    ]
    bridge, _store, _adapter = bridge_for(
        tmp_path, [call, result, second_call, second_result, TextDeltaEvent(text="done")]
    )
    answer = bridge.ask("claude-sub", "go", tools=tools)
    assert [type(e) for e in answer.tool_calls] == [
        ToolCallEvent,
        ToolResultEvent,
        ToolCallEvent,
        ToolResultEvent,
    ]
    assert answer.tool_calls[0].name == "search_notes"
    assert answer.tool_calls[1].content == "found 3"
    assert answer.text == "done"


def test_sampling_honesty_rides_along(tmp_path):
    """The 1.7 report, on the object a one-shot caller actually holds."""
    bridge, _store, _adapter = bridge_for(tmp_path, [TextDeltaEvent(text="hi")])
    answer = bridge.ask("claude-sub", "go", sampling=Sampling(temperature=0.2))
    assert answer.sampling_applied == dict(answer.receipt.sampling_applied)
    assert answer.sampling_notes == tuple(answer.receipt.sampling_notes)


# --- 2. a terminal that is not ok raises ------------------------------------------


def test_a_vendor_failure_raises_rather_than_returning_partial_text(tmp_path):
    adapter = FakeAdapter(
        [TextDeltaEvent(text="half an ans")], error=vendor_failure("400: bad model")
    )
    bridge, _store, _adapter = bridge_for(tmp_path, adapter=adapter)
    with pytest.raises(VendorRunFailed) as caught:
        bridge.ask("claude-sub", "go")
    assert "400: bad model" in str(caught.value)


def test_quota_exhaustion_raises_the_error_chat_callers_already_catch(tmp_path):
    bridge, _store, _adapter = bridge_for(tmp_path, [quota_exhausted()])
    with pytest.raises(QuotaExhausted):
        bridge.ask("claude-sub", "go")


def test_a_guard_stop_raises_with_the_numbers_on_it(tmp_path):
    conn = connection(guards=Guards(stop_at_tokens=5))
    bridge, _store, _adapter = bridge_for(
        tmp_path, [usage(10, 0), TextDeltaEvent(text="never seen")], conn=conn
    )
    with pytest.raises(GuardStop) as caught:
        bridge.ask("claude-sub", "go")
    assert caught.value.threshold == 5
    assert caught.value.observed >= 10


def test_ask_has_no_raise_on_stop_keyword(tmp_path):
    """It is permanently on, so offering the keyword would only offer a lie."""
    bridge, _store, _adapter = bridge_for(tmp_path, [quota_exhausted()])
    with pytest.raises(TypeError):
        bridge.ask("claude-sub", "go", raise_on_stop=False)


# --- 3. schema ---------------------------------------------------------------------


def test_a_schema_bound_ask_returns_the_parsed_object(tmp_path):
    adapter = FakeAdapter(
        [TextDeltaEvent(text='{"score": 7}')], structured_output={"score": 7}
    )
    bridge, _store, _adapter = bridge_for(tmp_path, adapter=adapter)
    answer = bridge.ask("claude-sub", "score this", schema=SCHEMA)
    assert answer.structured == {"score": 7}
    assert answer.status is TerminalStatus.OK


def test_a_run_that_produced_nothing_structured_raises(tmp_path):
    """The fake's unset ``structured_output`` is the real failure shape.

    modelpass ends such a run ``status=error``, so this arrives as the vendor
    failure it is rather than as an empty result that looks like a real one.
    """
    bridge, _store, _adapter = bridge_for(tmp_path, [TextDeltaEvent(text="sorry")])
    with pytest.raises(VendorRunFailed):
        bridge.ask("claude-sub", "score this", schema=SCHEMA)


def test_an_ok_run_missing_the_structured_event_is_a_contract_violation(tmp_path):
    """Belt and braces: ``structured`` is never ``None`` after an ok schema run.

    No shipped adapter can produce this -- the bridge turns a missing structured
    answer into a terminal error first -- so it is scripted by hand, because the
    alternative to raising here is handing a caller ``None`` from a call that
    said it succeeded.
    """

    class Terminal(FakeAdapter):
        def _with_structured(self, request, events):  # type: ignore[override]
            return list(events)

    adapter = Terminal([TextDeltaEvent(text="{}")])
    bridge, _store, _adapter = bridge_for(tmp_path, adapter=adapter)
    with pytest.raises(AdapterFailed) as caught:
        bridge.ask("claude-sub", "score this", schema=SCHEMA)
    assert "structured_output" in str(caught.value)


def test_schema_and_tools_are_still_refused_together(tmp_path):
    """D13's gate is ``chat``'s, and ``ask`` delegates rather than re-implementing."""

    def handler(_args):
        return ""

    tools = [
        ToolDef(
            name="t",
            description="t",
            parameters={"type": "object", "properties": {}},
            handler=handler,
        )
    ]
    bridge, _store, _adapter = bridge_for(tmp_path, [TextDeltaEvent(text="x")])
    with pytest.raises(InvalidSchema):
        bridge.ask("claude-sub", "go", schema=SCHEMA, tools=tools)


# --- 4. the two collected paths may not drift -------------------------------------


def test_ask_and_chatsubpass_invoke_agree_on_the_text(tmp_path):
    """One script, two collectors, one answer.

    The LangChain leaf already collects a stream into a single message and
    ``ask`` now does the same thing next door. Two implementations of one idea
    drift, so this is the test that notices.
    """
    pytest.importorskip("langchain_core")
    from modelpass.langchain_adapter import ChatSubpass

    script = [
        TextDeltaEvent(text="a git "),
        TextDeltaEvent(text="rebase "),
        TextDeltaEvent(text="replays commits"),
        usage(11, 5),
    ]
    bridge, _store, _adapter = bridge_for(tmp_path, script)

    answer = bridge.ask("claude-sub", "what is a rebase?")
    chat_model = ChatSubpass(connection="claude-sub", bridge=bridge)
    message = chat_model.invoke("what is a rebase?")

    assert answer.text == message.content
    assert answer.usage is not None
    assert answer.usage.total_tokens == message.usage_metadata["total_tokens"]


def test_ask_and_chatsubpass_invoke_agree_on_a_schema_bound_answer(tmp_path):
    pytest.importorskip("langchain_core")
    from modelpass.langchain_adapter import ChatSubpass

    adapter = FakeAdapter(
        [TextDeltaEvent(text="reasoning "), structured({"score": 7}, schema=SCHEMA)],
    )
    bridge, _store, _adapter = bridge_for(tmp_path, adapter=adapter)

    answer = bridge.ask("claude-sub", "score this", schema=SCHEMA)
    chat_model = ChatSubpass(connection="claude-sub", bridge=bridge)
    message = chat_model.bind_tools([SCHEMA]).invoke("score this")

    assert answer.structured == message.tool_calls[0]["args"]
    assert answer.text == message.content


def test_module_level_bridge_exports_answer():
    import modelpass

    assert modelpass.Answer is Answer
    assert "Answer" in modelpass.__all__


# --- 5. ...and the async twins may not drift from them either (ticket 1.13) --------


def test_aask_and_chatsubpass_ainvoke_agree_with_their_sync_twins(tmp_path):
    """Four collectors now, one answer.

    ``ask``/``invoke`` gained ``aask``/``ainvoke`` with ticket 1.13's async face,
    and the leaf stopped declining to override async. Four implementations of
    one idea is four chances to drift, so the same script is run through all of
    them and the four answers are compared to each other.
    """
    pytest.importorskip("langchain_core")
    import asyncio

    from modelpass.langchain_adapter import ChatSubpass

    def script(_request):
        return [
            TextDeltaEvent(text="a git "),
            TextDeltaEvent(text="rebase "),
            TextDeltaEvent(text="replays commits"),
            usage(11, 5),
        ]

    bridge, _store, _adapter = bridge_for(tmp_path, script)
    chat_model = ChatSubpass(connection="claude-sub", bridge=bridge)

    answer = bridge.ask("claude-sub", "what is a rebase?")
    aanswer = asyncio.run(bridge.aask("claude-sub", "what is a rebase?"))
    message = chat_model.invoke("what is a rebase?")
    amessage = asyncio.run(chat_model.ainvoke("what is a rebase?"))

    assert aanswer.text == answer.text == message.content == amessage.content
    assert aanswer.usage == answer.usage
    assert (
        amessage.usage_metadata["total_tokens"]
        == message.usage_metadata["total_tokens"]
        == answer.usage.total_tokens
    )
    assert aanswer.status is answer.status


def test_astream_and_stream_agree_on_the_chunks(tmp_path):
    pytest.importorskip("langchain_core")
    import asyncio

    from modelpass.langchain_adapter import ChatSubpass

    def script(_request):
        return [
            TextDeltaEvent(text="one "),
            TextDeltaEvent(text="two"),
            usage(4, 2),
        ]

    bridge, _store, _adapter = bridge_for(tmp_path, script)
    chat_model = ChatSubpass(connection="claude-sub", bridge=bridge)

    sync_chunks = [chunk.content for chunk in chat_model.stream("count")]

    async def collect():
        return [chunk.content async for chunk in chat_model.astream("count")]

    async_chunks = asyncio.run(collect())
    assert async_chunks == sync_chunks
    assert "".join(async_chunks) == "one two"


def test_aask_and_ainvoke_agree_on_a_schema_bound_answer(tmp_path):
    pytest.importorskip("langchain_core")
    import asyncio

    from modelpass.langchain_adapter import ChatSubpass

    def script(_request):
        return [TextDeltaEvent(text="reasoning "), structured({"score": 7}, schema=SCHEMA)]

    bridge, _store, _adapter = bridge_for(tmp_path, script)
    chat_model = ChatSubpass(connection="claude-sub", bridge=bridge)

    answer = asyncio.run(bridge.aask("claude-sub", "score this", schema=SCHEMA))
    message = asyncio.run(chat_model.bind_tools([SCHEMA]).ainvoke("score this"))

    assert answer.structured == message.tool_calls[0]["args"] == {"score": 7}
    assert answer.text == message.content
