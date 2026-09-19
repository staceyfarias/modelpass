"""Ticket 1.8: a session is a fact about the runtime, and the bridge says so.

Two refusals live here, and they are the same refusal read from two angles.

**Sessions on an API runtime.** :meth:`Bridge.new_chat`, ``new_worker``,
``resume_chat`` and ``list_sessions`` all rest on the *runtime* holding the
conversation (D14). An HTTP endpoint holds none -- every request carries its
whole history -- so all four are refused on all four API runtimes, before an
adapter is touched, with ``bridge.chat(history=[...])`` named as the thing that
does work. The refusal is the bridge's, not an adapter's: these tests register a
fully session-capable fake on an API runtime and the bridge still says no.

**A worker with no toolbelt.** ``SessionRequest.native_tools`` and
``.appends_system_prompt`` are derived from ``kind is SessionKind.WORKER`` and
nothing else (``adapters/base.py``), so a runtime with no native toolbelt and no
vendor persona would have handed back a ``WorkerSession`` that is a chat wearing
a worker's name. That is the ``new_worker()`` trap on the design's latent-defect
list, and the ``tools`` cell is what closes it.

Both refusals are :class:`~modelpass.errors.CapabilityNotSupported`, which is
the class its own docstring reserves for exactly this: *a runtime that simply
cannot do the thing asked for -- that is a fact about the runtime, not a mistake
in the call*.
"""

from __future__ import annotations

import pytest

from modelpass.capabilities import Capability, CapabilityRegistry, Support
from modelpass.connections import Connection, CredentialRef
from modelpass.errors import CapabilityNotSupported
from modelpass.runtimes import API_RUNTIMES, Runtime
from modelpass.testing import FakeSessionAdapter, fake_session_bridge
from modelpass.types import AuthMode

KEY = "sk-not-a-real-key-0123456789"


def api_connection(runtime: Runtime, *, name: str = "api") -> Connection:
    return Connection(
        name=name,
        runtime=runtime,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:VENDOR_KEY"),
        base_url="https://host.example/v1"
        if runtime is Runtime.OPENAI_COMPATIBLE
        else None,
    )


def api_bridge(runtime: Runtime, tmp_path):
    """A bridge whose adapter for an API runtime *can* open sessions.

    The point of handing the bridge a session-capable fake here is that nothing
    below can pass by accident: every refusal in this file is the bridge's
    policy, not an adapter that had not been written yet.
    """
    connection = api_connection(runtime)
    bridge, _store, adapter = fake_session_bridge(
        connections=[connection],
        runtime=runtime,
        home=tmp_path / "modelpass",
        env={"VENDOR_KEY": KEY},
    )
    return bridge, adapter


# --- 1. the policy, on all four API runtimes --------------------------------------


@pytest.mark.parametrize("runtime", sorted(API_RUNTIMES, key=str))
def test_new_chat_is_refused_on_every_api_runtime(runtime, tmp_path):
    bridge, adapter = api_bridge(runtime, tmp_path)
    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.new_chat(connection="api")
    message = str(caught.value)
    assert runtime.value in message
    assert "bridge.new_chat()" in message
    assert "bridge.chat(history=[...])" in message
    assert adapter.session_requests == [], "the adapter was never reached"


@pytest.mark.parametrize("runtime", sorted(API_RUNTIMES, key=str))
def test_new_worker_is_refused_on_every_api_runtime(runtime, tmp_path):
    bridge, adapter = api_bridge(runtime, tmp_path)
    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.new_worker(connection="api", project_folder=str(tmp_path))
    message = str(caught.value)
    assert runtime.value in message
    assert "bridge.new_worker()" in message
    assert "bridge.chat(history=[...])" in message
    assert adapter.opened == 0


@pytest.mark.parametrize("runtime", sorted(API_RUNTIMES, key=str))
def test_resume_chat_is_refused_on_every_api_runtime(runtime, tmp_path):
    bridge, adapter = api_bridge(runtime, tmp_path)
    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.resume_chat(connection="api", session_id="whatever", system_prompt="")
    message = str(caught.value)
    assert runtime.value in message
    assert "bridge.resume_chat()" in message
    assert adapter.session_requests == []


@pytest.mark.parametrize("runtime", sorted(API_RUNTIMES, key=str))
def test_list_sessions_is_refused_on_every_api_runtime(runtime, tmp_path):
    bridge, adapter = api_bridge(runtime, tmp_path)
    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.list_sessions(connection="api")
    message = str(caught.value)
    assert runtime.value in message
    assert "bridge.list_sessions()" in message
    assert adapter.listed == []


def test_the_refusal_names_the_cell_it_was_read_off(tmp_path):
    """``support`` and ``capability`` are the table's own words, not prose.

    A caller that wants to branch rather than read English gets the same two
    fields every other :class:`CapabilityNotSupported` carries.
    """
    bridge, _ = api_bridge(Runtime.ANTHROPIC_API, tmp_path)
    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.new_chat(connection="api")
    assert caught.value.runtime == Runtime.ANTHROPIC_API.value
    assert caught.value.capability == Capability.SESSIONS_RESUME.value
    assert caught.value.support == Support.UNSUPPORTED.value


def test_a_disabled_api_connection_still_reports_the_runtime_fact(tmp_path):
    """Re-enabling the connection would not help, so the runtime fact comes first.

    Deliberate ordering (ticket 1.8): ``ConnectionDisabled`` is a thing a caller
    can fix in the store, and offering it here would send somebody to flip a flag
    that changes nothing about whether this runtime holds a conversation.
    """
    connection = api_connection(Runtime.ANTHROPIC_API).with_enabled(False)
    bridge, _store, _adapter = fake_session_bridge(
        connections=[connection],
        runtime=Runtime.ANTHROPIC_API,
        home=tmp_path / "modelpass",
        env={"VENDOR_KEY": KEY},
    )
    with pytest.raises(CapabilityNotSupported):
        bridge.new_chat(connection="api")


def test_chat_is_untouched_on_an_api_runtime(tmp_path):
    """The alternative the refusal names has to actually be there."""
    from modelpass.types import TerminalEvent, TerminalStatus, TextDeltaEvent

    connection = api_connection(Runtime.ANTHROPIC_API)
    bridge, _store, _adapter = fake_session_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="hello")],
        runtime=Runtime.ANTHROPIC_API,
        home=tmp_path / "modelpass",
        env={"VENDOR_KEY": KEY},
    )
    events = list(bridge.chat(connection="api", message="hi"))
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert terminal.status is TerminalStatus.OK


# --- 2. the new_worker trap -------------------------------------------------------


def toolbelt_free_bridge(tmp_path):
    """An agent runtime whose ``tools`` cell reads ``unsupported``.

    Built by refining the registry rather than by inventing a runtime, because
    the cell is the thing under test: whatever runtime a future ticket adds, a
    ``tools`` cell that is not ``supported`` must refuse a worker.
    """
    registry = CapabilityRegistry()
    registry.refine(Runtime.ANTHROPIC_SDK, {Capability.TOOLS.value: Support.UNSUPPORTED})
    connection = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    adapter = FakeSessionAdapter(runtime=Runtime.ANTHROPIC_SDK)
    bridge, _store, adapter = fake_session_bridge(
        connections=[connection],
        adapter=adapter,
        home=tmp_path / "modelpass",
        registry=registry,
    )
    return bridge, adapter


def test_a_worker_is_refused_where_there_is_no_native_toolbelt(tmp_path):
    bridge, adapter = toolbelt_free_bridge(tmp_path)
    with pytest.raises(CapabilityNotSupported) as caught:
        bridge.new_worker(connection="claude-sub", project_folder=str(tmp_path))
    message = str(caught.value)
    assert caught.value.capability == Capability.TOOLS.value
    assert "bridge.new_worker()" in message
    assert "bridge.new_chat()" in message
    assert adapter.opened == 0, "no chat wearing a worker's name came back"
    assert adapter.session_requests == []


def test_a_chat_still_opens_on_that_same_runtime(tmp_path):
    """The refusal is about the worker, not about sessions on that runtime.

    Without this, a gate that refused everything would pass the test above and
    take ``new_chat`` down with it.
    """
    bridge, _adapter = toolbelt_free_bridge(tmp_path)
    session = bridge.new_chat(connection="claude-sub")
    assert session.kind.value == "chat"
    assert session.receipt.ok
