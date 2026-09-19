"""Live drive of the ``openai-api`` adapter. **Spends real money.**

Four calls against the real Responses API, and they are the four things a fake
transport cannot answer no matter how carefully it is written:

1. a short chat -- the model really answers, and the usage really comes back;
2. one tool round trip -- the model really asks for the function, and really uses
   what the handler returned;
3. one structured output -- the vendor really constrains the answer to the strict
   schema, rather than modelpass parsing prose that happened to be JSON;
4. one reasoning-effort call with a summary asked for -- which is **the one that
   moves a capability cell**. ``thinking`` on this row is ``unverified``
   precisely because no fake can say whether a reasoning summary ever arrives: a
   summary is emitted only when the request asks for one, and raw reasoning text
   is gated on the organisation's verification status. A green
   :func:`test_one_reasoning_effort_call` is the drive that moves it.

Everything else about this adapter is pinned by
``tests/test_adapter_openai_api.py``, which drives a fake and runs by default.

Quarantined the same way ``tests/live/test_anthropic_api_live.py`` is, and for
the same reasons:

* marked ``live``, which ``addopts`` deselects by default;
* additionally gated on ``SUBPASS_LIVE_TESTS=1``, so ``-m live`` alone still
  skips rather than surprising anyone;
* refused outright when a CI environment is detected;
* gated on ``MODELPASS_LIVE_OPENAI_API_KEY`` -- **a variable of its own,
  deliberately not ``OPENAI_API_KEY``**, so that having a key on the machine is
  never on its own enough to start spending. Naming it explicitly is the consent.

It uses a temp connection store, never the user's ``~/.modelpass``, and asks for
a handful of tokens at a time. Test 4 is the expensive one -- a reasoning model
thinking at ``low`` effort -- and is still cents rather than dollars.

Run with::

    SUBPASS_LIVE_TESTS=1 MODELPASS_LIVE_OPENAI_API_KEY=sk-... \\
        python -m pytest -m live tests/live/test_openai_api_live.py

**What a green run moves.** ``thinking`` on ``openai-api``, and only that. The
other three tests re-check live what the fake already pins, which is worth doing
before the owner trusts the row but is not what moved those cells. ``tools`` and
``mcp`` stay ``unverified``: they are about the vendor's *own* server-side tools,
which this adapter never sends and this file never drives.
"""

from __future__ import annotations

import os

import pytest

from modelpass.bridge import Bridge
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore
from modelpass.types import (
    AuthMode,
    Sampling,
    StructuredOutputEvent,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)

pytestmark = pytest.mark.live

#: Environment variables that mean "this is CI". Live tests must never run there.
_CI_MARKERS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "JENKINS_URL", "TF_BUILD")

#: The key this file spends, under a name nothing else sets by accident.
KEY_VAR = "MODELPASS_LIVE_OPENAI_API_KEY"

#: Small, current, and cheap. Overridable for a drive against another model --
#: the sampling cells are per-model and the rules table says so.
MODEL = os.environ.get("MODELPASS_LIVE_OPENAI_MODEL", "gpt-4o-mini")

#: The reasoning model test 4 drives. Separate from MODEL because an effort dial
#: is a per-model fact here: gpt-4o takes no reasoning_effort at all, and
#: modelpass.sampling_rules drops it with a note rather than sending it.
REASONING_MODEL = os.environ.get("MODELPASS_LIVE_OPENAI_REASONING_MODEL", "o4-mini")


def _gate() -> None:
    present = [name for name in _CI_MARKERS if os.environ.get(name)]
    if present:
        pytest.skip(f"live tests never run in CI (found {', '.join(present)})")
    if os.environ.get("SUBPASS_LIVE_TESTS") != "1":
        pytest.skip("live tests are opt-in: set SUBPASS_LIVE_TESTS=1 to run them")
    if not os.environ.get(KEY_VAR):
        pytest.skip(f"this file spends a metered key: set {KEY_VAR} to run it")
    pytest.importorskip("openai")


@pytest.fixture
def bridge(tmp_path) -> Bridge:
    _gate()
    store = ConnectionStore(tmp_path / "modelpass-home")
    store.add(
        Connection(
            name="live-openai",
            runtime=Runtime.OPENAI_API,
            auth_mode=AuthMode.API_KEY,
            credential_ref=CredentialRef.parse(f"env:{KEY_VAR}"),
            model=MODEL,
            # A real ceiling, because this file spends: nothing here should be
            # able to run away, and every prompt below wants a short answer.
            guards=Guards(stop_at_tokens=20000),
        )
    )
    return Bridge(store=store)


def _terminal(events: list) -> TerminalEvent:
    terminals = [e for e in events if isinstance(e, TerminalEvent)]
    assert len(terminals) == 1, "exactly one terminal event ends a run"
    return terminals[0]


def test_one_short_chat(bridge):
    """chat, streaming, incremental_text, usage_tokens -- against the real thing."""
    events = list(
        bridge.chat(
            connection="live-openai",
            message="Reply with exactly the word: ready",
            sampling=Sampling(max_output_tokens=32),
        )
    )
    text = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert "ready" in text.lower()

    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert usage, "a real response reports usage"
    assert usage[-1].usage.input_tokens > 0
    assert usage[-1].usage.output_tokens > 0
    # The subtraction must not produce a negative or a double count: fresh plus
    # cached is exactly what the vendor called input_tokens.
    assert usage[-1].usage.cached_input_tokens >= 0
    assert usage[-1].usage.cache_write_tokens == 0

    terminal = _terminal(events)
    assert terminal.status is TerminalStatus.OK
    assert terminal.auth_mode is AuthMode.API_KEY
    assert terminal.usage.total_tokens > 0


def test_one_tool_round_trip(bridge):
    """tools_in_process and interim_usage: the loop runs inside the adapter."""
    from modelpass.tools import ToolDef

    called: list[dict] = []

    def lookup(args: dict) -> str:
        called.append(args)
        return "The password is HORIZON."

    tool = ToolDef(
        name="get_password",
        description="Return the current password. Call this to answer any question "
        "about the password.",
        parameters={"type": "object", "properties": {}},
        handler=lookup,
    )

    events = list(
        bridge.chat(
            connection="live-openai",
            message="What is the current password? Use your tool, then say the word.",
            tools=[tool],
            sampling=Sampling(max_output_tokens=256),
        )
    )

    assert called, "the model asked for the tool and the handler ran in this process"
    call = next(e for e in events if isinstance(e, ToolCallEvent))
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert call.name == "get_password"
    assert call.server == "caller"
    assert result.id == call.id
    assert result.is_error is False

    text = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert "HORIZON" in text.upper()

    # Usage per round, not once at the end -- the granularity a guard needs.
    assert len([e for e in events if isinstance(e, UsageEvent)]) >= 2
    assert _terminal(events).status is TerminalStatus.OK


def test_one_structured_output(bridge):
    """structured_output: the vendor constrains the answer against a strict schema.

    The schema below is deliberately *not* already strict -- ``confidence`` is
    optional -- so this also drives the ``to_openai_strict`` rewrite against the
    real endpoint. A vendor that rejected the rewritten document would fail here
    and nowhere else in the suite.
    """
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["sentiment"],
    }
    events = list(
        bridge.chat(
            connection="live-openai",
            message="Classify the sentiment of: 'this made my whole week'.",
            schema=schema,
            schema_name="SentimentVerdict",
            sampling=Sampling(max_output_tokens=256),
        )
    )
    structured = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert isinstance(structured.data, dict)
    assert structured.valid is True, structured.problems
    assert "sentiment" in structured.data
    assert structured.schema_name == "SentimentVerdict"
    assert _terminal(events).status is TerminalStatus.OK


def test_one_reasoning_effort_call(bridge):
    """**The cell-moving one.** reasoning.effort is accepted, and a summary arrives.

    Two claims in one call, and they are separable on purpose:

    * the effort dial reaches the vendor and the call succeeds -- if the model
      rejected ``reasoning.effort`` this would be a failed terminal, not an
      assertion error;
    * a reasoning summary really is emitted and really becomes a
      ``ThinkingEvent``, which is what ``thinking`` on this row is waiting for.

    ``reasoning_summary`` is passed explicitly because modelpass never asks for a
    summary on its own: it is billed output nobody requested. If the second
    assertion fails while the first passes, the honest reading is that this
    organisation is not permitted reasoning summaries -- not that the adapter is
    broken -- and the cell stays where it is.
    """
    events = list(
        bridge.chat(
            connection="live-openai",
            model=REASONING_MODEL,
            message="A farmer has 17 sheep; all but 9 run away. How many are left? "
            "Answer with the number alone.",
            sampling=Sampling(reasoning_effort="low", max_output_tokens=2048),
            options={"reasoning_summary": "auto"},
        )
    )
    terminal = _terminal(events)
    assert terminal.status is TerminalStatus.OK, terminal.reason

    text = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert "9" in text

    # The vendor counts reasoning tokens inside output_tokens, so a thinking
    # call's output is larger than its visible answer -- worth asserting, because
    # it is the convention openai_api.token_usage deliberately did not change.
    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert usage and usage[-1].usage.output_tokens > len(text)

    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    assert thinking, (
        "no reasoning summary arrived, so the thinking cell stays unverified. "
        "Check whether this organisation is verified for reasoning summaries "
        "before reading this as an adapter fault"
    )
