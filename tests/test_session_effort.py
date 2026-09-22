"""Changing reasoning effort within one session: the seam, and what it reports.

Two gaps closed on 2026-09-22, and they were different kinds of gap:

1. `_session_turn_start_params` never sent `effort` at all. The stateless path
   had carried it since the day effort was wired; the session path did not, so a
   session ran at the server's default while the receipt reported the level it
   had been told. A silent disagreement between the record and the wire.
2. `ChatSession.send` had no way to state a level, so no mid-session change was
   expressible on any runtime -- which is why
   `effort_change_cache_preserved` could only ever be `None`.

What this file holds is that closing them did not also quietly promise that a
change keeps the cache. The capability and the measurement stay separate.
"""

from __future__ import annotations

import pytest

from modelpass.errors import CapabilityNotSupported, InvalidConnection
from modelpass.runtimes import Runtime
from modelpass.types import TerminalEvent, TextDeltaEvent, TokenUsage, UsageEvent, VendorEvent


def _connection(runtime: Runtime = Runtime.OPENAI_SDK, reasoning: str | None = "medium"):
    from modelpass.connections import Connection, CredentialRef
    from modelpass.types import AuthMode

    return Connection(
        name="c",
        runtime=runtime,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        reasoning=reasoning,
    )


def _bridge(tmp_path, *, runtime=Runtime.OPENAI_SDK, reasoning="medium", script=None):
    """A session-capable fake, because the seam under test is a session's."""
    from modelpass.testing import fake_session_bridge

    bridge, _store, adapter = fake_session_bridge(
        connections=[_connection(runtime, reasoning)],
        script=script
        or [
            TextDeltaEvent(text="hi"),
            UsageEvent(usage=TokenUsage(input_tokens=1545, cached_input_tokens=48736)),
        ],
        home=tmp_path / "home",
        runtime=runtime,
    )
    return bridge, adapter


# --- 1. gap one: the standing level reaches a session turn -----------------------


def test_a_session_turn_carries_the_connections_standing_effort():
    """The gap that made the receipt disagree with the wire.

    The stateless path put the level on `TurnStartParams`; the session path
    built its params object without ever looking at the connection. A session
    therefore ran at the server's default while every receipt said `medium`.
    """
    from modelpass.adapters.openai import _session_turn_start_params

    class _Request:
        connection = _connection(reasoning="medium")
        system_prompt = None
        appends_system_prompt = False

    params = _session_turn_start_params(_Request(), "t1", "hello", first_turn=True)
    assert params["effort"] == "medium"


def test_a_session_on_a_connection_that_states_nothing_sends_no_effort():
    """Silence stays silence: absent is not `high`, and modelpass does not pick."""
    from modelpass.adapters.openai import _session_turn_start_params

    class _Request:
        connection = _connection(reasoning=None)
        system_prompt = None
        appends_system_prompt = False

    params = _session_turn_start_params(_Request(), "t1", "hello", first_turn=True)
    assert "effort" not in params


def test_a_turn_override_beats_the_standing_level():
    from modelpass.adapters.openai import _session_turn_start_params

    class _Request:
        connection = _connection(reasoning="medium")
        system_prompt = None
        appends_system_prompt = False

    params = _session_turn_start_params(
        _Request(), "t1", "hello", first_turn=False, effort="high"
    )
    assert params["effort"] == "high"


# --- 2. gap two: a turn can state its own level ----------------------------------


def test_the_level_reaches_the_handle_and_the_receipt(tmp_path):
    """What went on the wire, and what the turn's receipt says about it."""
    bridge, adapter = _bridge(tmp_path)
    session = bridge.new_chat(connection="c")
    try:
        list(session.send("one"))
        events = list(session.send("two", reasoning="high"))
    finally:
        session.close()

    assert adapter.handles[-1].efforts == [None, "high"]

    receipt = next(e.receipt for e in events if hasattr(e, "receipt"))
    assert receipt.reasoning_value == "high"
    terminal = next(e for e in events if isinstance(e, TerminalEvent))
    assert terminal.reasoning_value == "high"


def test_a_per_turn_level_goes_through_the_same_ladder(tmp_path):
    """One vocabulary, not two. A rung the runtime lacks is refused here exactly
    as it is on the connection -- a second spelling of the same dial would
    disagree with the first at the worst possible moment."""
    bridge, _adapter = _bridge(tmp_path)
    session = bridge.new_chat(connection="c")
    try:
        with pytest.raises(InvalidConnection):
            session.send("x", reasoning="ultra")
        with pytest.raises(InvalidConnection):
            session.send("x", reasoning="nonsense")
    finally:
        session.close()


def test_a_runtime_without_a_per_turn_lever_refuses_rather_than_drops(tmp_path):
    """The refusal that keeps this honest.

    `claude-agent-sdk` reads effort off `ClaudeAgentOptions` when the session is
    created and has no `set_effort`. Accepting the argument and dropping it
    would hand a caller a turn they believe ran at a different level.
    """
    bridge, _adapter = _bridge(tmp_path, runtime=Runtime.ANTHROPIC_SDK, reasoning="high")
    session = bridge.new_chat(connection="c")
    try:
        with pytest.raises(CapabilityNotSupported) as raised:
            session.send("x", reasoning="low")
    finally:
        session.close()
    assert "set_effort" in str(raised.value) or "session opens" in str(raised.value)


# --- 3. a change is a change, and repeating a level is not ----------------------


def test_changing_the_level_announces_it_and_measures_the_cache(tmp_path):
    """The whole point of the seam: a transition that can be measured.

    The change is announced as a vendor event, the fold observes it on the way
    past, and the terminal reports whether *this turn's* counts show the prefix
    survived. 48,736 cached of a 50,281-token prompt is a prefix that survived.
    """
    bridge, _adapter = _bridge(tmp_path)
    session = bridge.new_chat(connection="c")
    try:
        list(session.send("one"))
        events = list(session.send("two", reasoning="high"))
    finally:
        session.close()

    changed = [
        e
        for e in events
        if isinstance(e, VendorEvent) and e.name == "reasoning_effort_changed"
    ]
    assert len(changed) == 1
    assert changed[0].data["from"] == "medium"
    assert changed[0].data["to"] == "high"

    terminal = next(e for e in events if isinstance(e, TerminalEvent))
    assert terminal.effort_change_cache_preserved is True


def test_a_broken_prefix_is_reported_as_broken(tmp_path):
    """The other outcome, and the one worth more: a measured `False`.

    A large cache *write* beside a small read is what a restarted cache looks
    like. This is the shape that would move `openai-sdk` from `experimental` to
    `unsupported` if a live run produced it.
    """
    bridge, _adapter = _bridge(
        tmp_path,
        script=[
            TextDeltaEvent(text="hi"),
            UsageEvent(
                usage=TokenUsage(input_tokens=2000, cache_write_tokens=48000)
            ),
        ],
    )
    session = bridge.new_chat(connection="c")
    try:
        list(session.send("one"))
        events = list(session.send("two", reasoning="high"))
    finally:
        session.close()

    terminal = next(e for e in events if isinstance(e, TerminalEvent))
    assert terminal.effort_change_cache_preserved is False


def test_repeating_the_level_in_force_is_not_a_transition(tmp_path):
    """Otherwise every turn of a loop that passes the same level would log a
    measurement of a cache that was never asked to survive anything."""
    bridge, _adapter = _bridge(tmp_path)
    session = bridge.new_chat(connection="c")
    try:
        events = list(session.send("one", reasoning="medium"))
    finally:
        session.close()

    assert not [
        e
        for e in events
        if isinstance(e, VendorEvent) and e.name == "reasoning_effort_changed"
    ]
    terminal = next(e for e in events if isinstance(e, TerminalEvent))
    assert terminal.effort_change_cache_preserved is None


def test_a_turn_that_states_nothing_after_a_change_keeps_the_new_level(tmp_path):
    """`TurnStartParams.effort` is documented as applying to "this turn and
    subsequent turns", so the level in force moves and a later turn that states
    nothing is not a second transition."""
    bridge, _adapter = _bridge(tmp_path)
    session = bridge.new_chat(connection="c")
    try:
        list(session.send("one", reasoning="high"))
        events = list(session.send("two"))
    finally:
        session.close()

    assert not [
        e
        for e in events
        if isinstance(e, VendorEvent) and e.name == "reasoning_effort_changed"
    ]


# --- 4. the seam did not smuggle in a promise ------------------------------------


def test_the_capability_still_reads_experimental_on_codex(tmp_path):
    """Closing the gap made the change *expressible*. It did not make it safe.

    A consumer who sees a working `send(reasoning=...)` might reasonably assume
    modelpass would not offer it if it broke the cache. The receipt says
    otherwise on the same run, in the same object.
    """
    from modelpass.prompt_cache import EffortCacheContinuity, effort_cache_continuity

    verdict = effort_cache_continuity(Runtime.OPENAI_SDK, "gpt-6-astra")
    assert verdict.capability is EffortCacheContinuity.EXPERIMENTAL

    bridge, _adapter = _bridge(tmp_path)
    session = bridge.new_chat(connection="c")
    try:
        events = list(session.send("one", reasoning="high"))
    finally:
        session.close()
    receipt = next(e.receipt for e in events if hasattr(e, "receipt"))
    # The declared capability for an unnamed model is `unknown`, which is the
    # honest answer and not the `experimental` above: that cell is Astra's.
    assert receipt.effort_cache_continuity in {"unknown", "experimental"}
