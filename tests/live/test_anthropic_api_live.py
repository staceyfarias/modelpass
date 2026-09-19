"""Live drive of the ``anthropic-api`` adapter. **Spends real money.**

Four calls against the real Messages API, and they are the four things a fake
transport cannot answer no matter how carefully it is written:

1. a short chat -- the model really answers, and the usage really comes back;
2. one tool round trip -- the model really asks for the tool, and really uses
   what the handler returned;
3. one structured output -- the vendor really constrains the answer to the
   schema, rather than modelpass parsing prose that happened to be JSON;
4. one cache breakpoint, twice -- and the second call really reads the prefix
   back, which is the only observation that distinguishes "the marker was sent"
   from "the marker worked".

Everything else about this adapter is pinned by
``tests/test_adapter_anthropic_api.py``, which drives a fake and runs by
default. This file exists because four claims in the capability table are about
the *vendor's* behaviour, and the table's rule is that such a cell moves with a
drive rather than with a changelog.

Quarantined the same way ``tests/test_live_anthropic.py`` is, and for the same
reasons:

* marked ``live``, which ``addopts`` deselects by default;
* additionally gated on ``SUBPASS_LIVE_TESTS=1``, so ``-m live`` alone still
  skips rather than surprising anyone;
* refused outright when a CI environment is detected;
* gated on ``MODELPASS_LIVE_ANTHROPIC_API_KEY`` -- **a variable of its own,
  deliberately not ``ANTHROPIC_API_KEY``**, so that having a key on the machine
  is never on its own enough to start spending. Naming it explicitly is the
  consent.

It uses a temp connection store, never the user's ``~/.modelpass``, and asks for
a handful of tokens at a time. The whole file should cost a fraction of a cent.

Run with::

    SUBPASS_LIVE_TESTS=1 MODELPASS_LIVE_ANTHROPIC_API_KEY=sk-ant-... \\
        python -m pytest -m live tests/live/test_anthropic_api_live.py

**What a green run moves.** ``ttl_control`` on ``anthropic-api`` stays
``unverified`` even after this file passes, and that is not an oversight: a TTL
is a *lifetime*, and no assertion here waits five minutes to observe one.
Moving that cell needs a drive nobody should put in a test suite -- two calls
more than five minutes apart, the second still reading cache -- and the cell's
note says so.
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
    CacheControl,
    Message,
    Role,
    StructuredOutputEvent,
    TerminalEvent,
    TerminalStatus,
    TextBlock,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)

pytestmark = pytest.mark.live

#: Environment variables that mean "this is CI". Live tests must never run there.
_CI_MARKERS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "JENKINS_URL", "TF_BUILD")

#: The key this file spends, under a name nothing else sets by accident.
KEY_VAR = "MODELPASS_LIVE_ANTHROPIC_API_KEY"

#: Small, current, and cheap. Overridable for a drive against another model --
#: the sampling and thinking cells are per-model, and ticket 1.7 will want that.
MODEL = os.environ.get("MODELPASS_LIVE_ANTHROPIC_MODEL", "claude-haiku-4-5")

#: Long enough to clear the smallest published minimum cacheable prefix with
#: room to spare. Caching below the floor is silent, so a short block here would
#: make test 4 fail for a reason that has nothing to do with the adapter.
_CACHEABLE_PREFIX = ("This is a fixed instruction block used to exercise prompt caching. " * 220)


def _gate() -> None:
    present = [name for name in _CI_MARKERS if os.environ.get(name)]
    if present:
        pytest.skip(f"live tests never run in CI (found {', '.join(present)})")
    if os.environ.get("SUBPASS_LIVE_TESTS") != "1":
        pytest.skip("live tests are opt-in: set SUBPASS_LIVE_TESTS=1 to run them")
    if not os.environ.get(KEY_VAR):
        pytest.skip(f"this file spends a metered key: set {KEY_VAR} to run it")
    pytest.importorskip("anthropic")


@pytest.fixture
def bridge(tmp_path) -> Bridge:
    _gate()
    store = ConnectionStore(tmp_path / "modelpass-home")
    store.add(
        Connection(
            name="live-api",
            runtime=Runtime.ANTHROPIC_API,
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
            connection="live-api",
            message="Reply with exactly the word: ready",
            options={"max_output_tokens": 32},
        )
    )
    text = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert "ready" in text.lower()

    usage = [e for e in events if isinstance(e, UsageEvent)]
    assert usage, "a real response reports usage"
    assert usage[-1].usage.input_tokens > 0
    assert usage[-1].usage.output_tokens > 0

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
            connection="live-api",
            message="What is the current password? Use your tool, then say the word.",
            tools=[tool],
            options={"max_output_tokens": 256},
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
    """structured_output: the vendor constrains the answer, modelpass reports it."""
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["sentiment", "confidence"],
    }
    events = list(
        bridge.chat(
            connection="live-api",
            message="Classify the sentiment of: 'this made my whole week'.",
            schema=schema,
            schema_name="SentimentVerdict",
            options={"max_output_tokens": 256},
        )
    )
    structured = next(e for e in events if isinstance(e, StructuredOutputEvent))
    assert isinstance(structured.data, dict)
    assert structured.valid is True, structured.problems
    assert set(structured.data) >= {"sentiment", "confidence"}
    assert structured.schema_name == "SentimentVerdict"
    assert _terminal(events).status is TerminalStatus.OK


def test_cache_breakpoint_is_read_back_on_the_second_call(bridge):
    """cache_breakpoints: the marker is not merely sent, it works.

    Two calls with a byte-identical marked prefix and a different question. The
    first writes the prefix; the second must read it back. Asserting on the
    *read* is the whole point -- a breakpoint that reached the vendor and cached
    nothing looks exactly like one that was silently dropped, which is the
    failure R3 exists to end.
    """
    system = Message(
        role=Role.SYSTEM,
        content=[TextBlock(text=_CACHEABLE_PREFIX, cache_control=CacheControl(ttl="5m"))],
    )

    def ask(question: str) -> tuple:
        events = list(
            bridge.chat(
                connection="live-api",
                message=question,
                history=[system],
                options={"max_output_tokens": 32},
            )
        )
        usage = [e for e in events if isinstance(e, UsageEvent)]
        assert usage, "a real response reports usage"
        return _terminal(events), usage[-1]

    first_terminal, first_usage = ask("Reply with the single word: one")
    assert first_terminal.status is TerminalStatus.OK
    wrote = first_usage.usage.cache_write_tokens
    read_first = first_usage.usage.cached_input_tokens
    assert wrote > 0 or read_first > 0, (
        "the first call neither wrote nor read a cached prefix, so the breakpoint "
        "did not reach the vendor -- or the prefix is below this model's floor"
    )

    second_terminal, second_usage = ask("Reply with the single word: two")
    assert second_terminal.status is TerminalStatus.OK
    assert second_usage.usage.cached_input_tokens > 0, (
        "the second call did not read the prefix back; cache_read_input_tokens "
        f"was {second_usage.usage.cached_input_tokens}"
    )
