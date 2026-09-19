"""The promise :mod:`modelpass.testing` makes, checked from outside the package.

Every other session test in this suite reaches for a local fixture. This one
deliberately does not: it imports nothing from ``tests/`` and touches nothing
private, so what it exercises is exactly the surface a downstream tool gets
after ``pip install modelpass`` -- and it fails if that surface stops being enough
to drive a session, which is what happened when the fakes lived in ``tests/``.

The gap was real and it cost real money. A consumer integrated
``bridge.new_chat()`` and found ``FakeAdapter`` had no ``open_session()``, so
the only way to cover their session handling was a live script spending
subscription allowance on every run. The module docstring in ``testing.py``
promises that every behavior of the core is reachable without a network, a
vendor SDK or a credential; sessions were the part where the promise was false.
This file is the test that would have caught it.
"""

from __future__ import annotations

import pytest

from modelpass import AuthMode, Connection, CredentialRef, Runtime
from modelpass.errors import AdapterNotImplemented
from modelpass.testing import (
    FakeAdapter,
    FakeSessionAdapter,
    FakeSessionHandle,
    fake_session_bridge,
    usage,
)
from modelpass.types import Role, SessionInfo, TerminalStatus, TextDeltaEvent


def connection(name: str = "claude-sub") -> Connection:
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


def offline(tmp_path, **kwargs):
    """A session-capable bridge assembled from the public surface only."""
    return fake_session_bridge(
        connections=[connection()],
        home=tmp_path / "modelpass-home",
        **kwargs,
    )


def text(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDeltaEvent))


def test_a_multi_turn_chat_runs_end_to_end_on_the_shipped_fakes(tmp_path):
    """Turn 2 continues the same session -- the assertion the seam exists for.

    A session that opened twice would still stream plausible-looking events and
    still pass every per-turn assertion; what it would have lost is the whole
    point of D14, the warm prefix and the runtime-held history. ``opened`` is
    the counter that tells the two apart, so it is the one asserted here.
    """
    bridge, _, adapter = offline(
        tmp_path,
        script=[TextDeltaEvent(text="hello "), TextDeltaEvent(text="world"), usage(input_tokens=7)],
    )

    with bridge.new_chat(connection="claude-sub", system_prompt="be terse") as session:
        assert session.id is None, "nothing exists until a turn completes"

        first = list(session.send("turn one"))
        assert text(first) == "hello world"
        assert first[-1].status is TerminalStatus.OK
        opened_after_first = session.id
        assert opened_after_first is not None

        second = list(session.send("turn two"))
        assert second[-1].status is TerminalStatus.OK
        assert session.id == opened_after_first, "turn two continued, it did not reopen"

        history = session.get_history()
        assert [m.role for m in history] == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]
        assert [m.content for m in history if m.role is Role.USER] == ["turn one", "turn two"]
        assert session.usage.total_tokens == 14, "the session totals both turns"

    assert adapter.opened == 1, "two turns, one vendor session"
    assert len(adapter.handles) == 1
    assert adapter.handles[0].messages == ["turn one", "turn two"]
    assert adapter.handles[0].closes == 1, "the context manager closed it"


def test_resume_and_listing_are_reachable_offline_too(tmp_path):
    """The rest of the session API, from the same public surface."""
    listing = (
        SessionInfo(
            id="sess-42",
            connection="claude-sub",
            runtime=Runtime.ANTHROPIC_SDK,
            project_folder="/work",
            updated_at="2026-08-30T10:00:00Z",
        ),
    )
    bridge, _, adapter = offline(tmp_path, adapter=FakeSessionAdapter(listing=listing))

    found = bridge.list_sessions(connection="claude-sub", project_folder="/work")
    assert [s.id for s in found] == ["sess-42"]

    resumed = bridge.resume_chat(
        connection="claude-sub", session_id="sess-42", system_prompt="be terse"
    )
    assert resumed.id == "sess-42", "a resumed session knows its id before it runs"
    assert adapter.session_requests[-1].resume_id == "sess-42"
    list(resumed.send("carry on"))
    assert adapter.opened == 0, "resuming creates nothing"
    resumed.close()


def test_the_session_bridge_is_a_drop_in_for_the_stateless_one(tmp_path):
    """``fake_session_bridge`` adds the session seam and subtracts nothing.

    The subclassing is the load-bearing part of the export: a consumer that
    already tests ``bridge.chat()`` against a scripted adapter must be able to
    add a session test without maintaining two adapters.
    """
    bridge, store, adapter = offline(tmp_path, script=[TextDeltaEvent(text="stateless")])

    assert isinstance(adapter, FakeAdapter)
    assert store.get("claude-sub").name == "claude-sub"
    stateless = bridge.chat(connection="claude-sub", message="hi")
    assert text(stateless) == "stateless"
    assert adapter.requests, "the stateless seam still records its runs"


def test_the_environment_stays_empty_so_the_machine_cannot_change_the_answer(tmp_path):
    """``env`` defaults to empty, exactly as :func:`fake_bridge` does.

    Inherited straight from ``fake_bridge`` rather than re-derived, but worth an
    assertion of its own: this is the default whose absence made a consumer's
    tests quietly read the developer's own ambient credentials.
    """
    bridge, _, adapter = offline(tmp_path)
    bridge.new_chat(connection="claude-sub")
    assert dict(adapter.session_requests[0].env) == {}


def test_the_plain_fake_adapter_still_refuses_sessions(tmp_path):
    """The refusal is a feature, and shipping the session fake must not erase it.

    A runtime that cannot open sessions has to be testable too, and the sentence
    it produces -- naming the method and the runtime -- is the one a consumer
    reads first when their own adapter is incomplete.
    """
    bridge, _, _ = offline(tmp_path, adapter=FakeAdapter())  # type: ignore[arg-type]
    with pytest.raises(AdapterNotImplemented) as caught:
        bridge.new_chat(connection="claude-sub")
    assert "open_session" in str(caught.value)


def test_the_handle_type_is_importable_for_consumers_that_subclass_it(tmp_path):
    """``FakeSessionHandle`` is exported, not an implementation detail.

    A consumer scripting a runtime quirk -- an id that never arrives, a history
    the runtime will not report -- subclasses the handle rather than rebuilding
    the adapter, so the name has to be reachable.
    """
    bridge, _, adapter = offline(tmp_path)
    session = bridge.new_chat(connection="claude-sub")
    list(session.send("hi"))
    assert isinstance(adapter.handles[0], FakeSessionHandle)
    session.close()
