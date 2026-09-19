"""Live drive of the ``google-api`` adapter. **Spends real money.**

Four calls against the real Gemini API, and they are the four things a fake
transport cannot answer no matter how carefully it is written:

1. a short chat -- the model really answers, and the usage really comes back;
2. one tool round trip -- the model really asks for the function, really reads a
   ``function_response`` part back, and really uses what the handler returned;
3. one structured output -- the vendor really constrains the answer to the JSON
   schema, rather than modelpass parsing prose that happened to be JSON;
4. one thinking call with thoughts asked for -- which is **the one that moves a
   capability cell**. ``thinking`` on this row is ``unverified`` precisely
   because no fake can say whether a thought part ever arrives: thoughts are
   emitted only when ``thinking_config.include_thoughts`` asks for them, and only
   by a model that thinks at all. A green :func:`test_one_thinking_call` is the
   drive that moves it.

Everything else about this adapter is pinned by
``tests/test_adapter_google_api.py``, which drives a fake and runs by default.

Quarantined the same way ``tests/live/test_openai_api_live.py`` is, and for the
same reasons:

* marked ``live``, which ``addopts`` deselects by default;
* additionally gated on ``SUBPASS_LIVE_TESTS=1``, so ``-m live`` alone still
  skips rather than surprising anyone;
* refused outright when a CI environment is detected;
* gated on ``MODELPASS_LIVE_GOOGLE_API_KEY`` -- **a variable of its own,
  deliberately neither ``GEMINI_API_KEY`` nor ``GOOGLE_API_KEY``**, the two names
  this SDK would have picked up on its own, so that having a key on the machine
  is never enough to start spending. Naming it explicitly is the consent.

It uses a temp connection store, never the user's ``~/.modelpass``, and asks for
a handful of tokens at a time.

Run with::

    SUBPASS_LIVE_TESTS=1 MODELPASS_LIVE_GOOGLE_API_KEY=AIza... \\
        python -m pytest -m live tests/live/test_google_api_live.py

**What a green run moves.** ``thinking`` on ``google-api``, and only that. The
other three tests re-check live what the fake already pins, which is worth doing
before the owner trusts the row but is not what moved those cells. ``tools``,
``mcp`` and ``mcp_servers`` stay ``unverified``: they are about the vendor's own
server-side tools and its ``Tool.mcp_servers`` field, which this adapter never
sends and this file never drives.
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

#: The key this file spends, under a name the SDK does not discover by itself.
KEY_VAR = "MODELPASS_LIVE_GOOGLE_API_KEY"

#: Small, current, and cheap. Overridable for a drive against another model --
#: the sampling cells are per-model and the rules table says every Gemini model
#: is unknown to it, which is the note every call here will carry.
MODEL = os.environ.get("MODELPASS_LIVE_GOOGLE_MODEL", "gemini-2.5-flash")

#: The thinking model test 4 drives. Separate from MODEL because whether a model
#: thinks is a per-model fact modelpass has not driven: it is the reason the
#: thinking cell is open, and the reason this name is overridable.
THINKING_MODEL = os.environ.get("MODELPASS_LIVE_GOOGLE_THINKING_MODEL", "gemini-2.5-pro")


def _gate() -> None:
    present = [name for name in _CI_MARKERS if os.environ.get(name)]
    if present:
        pytest.skip(f"live tests never run in CI (found {', '.join(present)})")
    if os.environ.get("SUBPASS_LIVE_TESTS") != "1":
        pytest.skip("live tests are opt-in: set SUBPASS_LIVE_TESTS=1 to run them")
    if not os.environ.get(KEY_VAR):
        pytest.skip(f"this file spends a metered key: set {KEY_VAR} to run it")
    pytest.importorskip("google.genai")


@pytest.fixture
def bridge(tmp_path) -> Bridge:
    _gate()
    store = ConnectionStore(tmp_path / "modelpass-home")
    store.add(
        Connection(
            name="live-google",
            runtime=Runtime.GOOGLE_API,
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
            connection="live-google",
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
    # The subtraction must not produce a negative or a double count, and this
    # runtime reports no cache write at all.
    assert usage[-1].usage.cached_input_tokens >= 0
    assert usage[-1].usage.cache_write_tokens == 0

    terminal = _terminal(events)
    assert terminal.status is TerminalStatus.OK, terminal.reason
    assert terminal.auth_mode is AuthMode.API_KEY
    assert terminal.usage.total_tokens > 0


def test_one_tool_round_trip(bridge):
    """tools_in_process and interim_usage: the loop runs inside the adapter.

    This is also the only place the ``function_response`` shape is checked against
    a real reader. A fake will accept any mapping; the vendor will not.
    """
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
            connection="live-google",
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
    """structured_output: the vendor constrains the answer against the schema.

    The schema is deliberately *not* strict -- ``confidence`` is optional -- which
    is the difference from ``openai-api``: nothing is rewritten before it is sent,
    so a vendor that rejected an optional property would fail here and nowhere
    else in the suite.
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
            connection="live-google",
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


def test_one_thinking_call(bridge):
    """**The cell-moving one.** A thinking level is accepted, and thoughts arrive.

    Three claims in one call, separable on purpose:

    * ``thinking_config.thinking_level`` reaches the vendor and the call succeeds
      -- if the model rejected it this would be a failed terminal rather than an
      assertion error, which is the answer to "was mapping the effort dial onto a
      level honest";
    * thought parts really are emitted when ``include_thoughts`` asks, and become
      ``ThinkingEvent`` -- which is what ``thinking`` on this row is waiting for;
    * and the thoughts do **not** appear in the answer text, which is the failure
      this vendor's shape makes easy and the one a user would see.

    ``include_thoughts`` is passed explicitly because modelpass never asks for
    thoughts on its own: they are billed output nobody requested. If the second
    assertion fails while the first passes, the honest reading is that this model
    does not emit thought parts -- not that the adapter is broken -- and the cell
    stays where it is.
    """
    events = list(
        bridge.chat(
            connection="live-google",
            model=THINKING_MODEL,
            message="A farmer has 17 sheep; all but 9 run away. How many are left? "
            "Answer with the number alone.",
            sampling=Sampling(reasoning_effort="low", max_output_tokens=2048),
            options={"include_thoughts": True},
        )
    )
    terminal = _terminal(events)
    assert terminal.status is TerminalStatus.OK, terminal.reason

    text = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert "9" in text

    # Thinking tokens are counted beside the candidates on this vendor and are
    # added into output_tokens by the adapter, so a thinking call's output is
    # larger than its visible answer. That addition is the convention-preserving
    # half of google_api.token_usage and this is the only place it is observed.
    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert usage and usage[-1].usage.output_tokens > len(text)

    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    assert thinking, (
        "no thought part arrived, so the thinking cell stays unverified. Check "
        f"whether {THINKING_MODEL} emits thought parts before reading this as an "
        "adapter fault"
    )
    thoughts = "".join(e.text for e in thinking)
    assert thoughts not in text, "a thought must never reach the answer a user reads"
