"""Live drive of the ``openai-compatible`` adapter, against a local Ollama.

**Spends nothing but electricity**, which is the one way this file differs in
kind from its two siblings and the reason its gate has no key variable in it.
``tests/live/test_anthropic_api_live.py`` and ``tests/live/test_openai_api_live.py``
are quarantined because they bill a metered account; this one is quarantined
because it needs *a server running on this machine*, which is a different and
equally real precondition for a suite whose rule is that nothing in it touches a
network by default.

What it is for: this runtime's capability row is ``unverified`` **by
construction** and moves per install through ``modelpass verify`` rather than by
ticket. So the thing worth driving live is not "do the cells move" --
``tests/test_adapter_openai_compatible.py`` pins that against a fake -- but
whether the three drives ``verify`` runs say true things about a *real* server
whose behaviour nobody wrote. A local Ollama is the reference install for that,
and it is where the interesting answers are: it honours
``stream_options.include_usage``, it will or will not call a tool depending on
the model, and its ``response_format`` support depends on its version. Every one
of those is a cell this file lets a person observe rather than assume.

Run with::

    SUBPASS_LIVE_TESTS=1 \\
        MODELPASS_LIVE_COMPATIBLE_BASE_URL=http://localhost:11434/v1 \\
        MODELPASS_LIVE_COMPATIBLE_MODEL=llama3.1 \\
        python -m pytest -m live tests/live/test_openai_compatible_live.py

Quarantined the same way the other two are: marked ``live``, which ``addopts``
deselects by default; additionally gated on ``SUBPASS_LIVE_TESTS=1``; refused
outright when a CI environment is detected; and gated on
``MODELPASS_LIVE_COMPATIBLE_BASE_URL``, which is the consent -- naming the
endpoint explicitly is how a person says "yes, drive that box".

**What a green run does not move.** Nothing in the shared capability table. The
whole point of this runtime is that a drive against one endpoint is evidence
about that endpoint, which is why the verdicts land on a connection in the store
and not in ``STATIC_TABLE``. If this file passes against your Ollama and fails
against your colleague's LM Studio, both runs were correct.
"""

from __future__ import annotations

import os

import pytest

from modelpass.adapters.base import RunRequest
from modelpass.adapters.openai_compatible import OpenAICompatibleAdapter
from modelpass.bridge import Bridge
from modelpass.capabilities import VERIFY_CELLS, Capability, Support
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore
from modelpass.types import (
    AuthMode,
    Message,
    Role,
    Sampling,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    UsageEvent,
)

pytestmark = pytest.mark.live

#: Environment variables that mean "this is CI". Live tests must never run there.
_CI_MARKERS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "JENKINS_URL", "TF_BUILD")

#: The endpoint this file drives. **Naming it is the consent**, which is what the
#: key variable is on the two metered files: having Ollama installed must never
#: be on its own enough for a test run to start talking to it.
BASE_URL_VAR = "MODELPASS_LIVE_COMPATIBLE_BASE_URL"

#: Which model to ask for. No default that is a real model name: an endpoint has
#: no default model, and guessing one here would produce a 404 that reads like an
#: adapter bug.
MODEL_VAR = "MODELPASS_LIVE_COMPATIBLE_MODEL"

#: Optional. A local box usually checks nothing, which is exactly the
#: ``credentialRef = "none"`` case, so this file drives that shape by default and
#: the variable is there for a gateway that does want a key.
KEY_VAR = "MODELPASS_LIVE_COMPATIBLE_KEY"

CONNECTION = "live-compatible"


def _gate() -> str:
    present = [name for name in _CI_MARKERS if os.environ.get(name)]
    if present:
        pytest.skip(f"live tests never run in CI (found {', '.join(present)})")
    if os.environ.get("SUBPASS_LIVE_TESTS") != "1":
        pytest.skip("live tests are opt-in: set SUBPASS_LIVE_TESTS=1 to run them")
    base_url = os.environ.get(BASE_URL_VAR)
    if not base_url:
        pytest.skip(
            f"this file drives a local model server: set {BASE_URL_VAR} "
            "(for example http://localhost:11434/v1) to run it"
        )
    if not os.environ.get(MODEL_VAR):
        pytest.skip(
            f"an OpenAI-compatible endpoint has no default model: set {MODEL_VAR} "
            "to one your server serves"
        )
    pytest.importorskip("openai")
    return base_url


def _connection(base_url: str) -> Connection:
    key = os.environ.get(KEY_VAR)
    return Connection(
        name=CONNECTION,
        runtime=Runtime.OPENAI_COMPATIBLE,
        auth_mode=AuthMode.API_KEY,
        credential_ref=(
            CredentialRef.parse(f"env:{KEY_VAR}") if key else CredentialRef.none()
        ),
        model=os.environ[MODEL_VAR],
        base_url=base_url,
        # A ceiling even though a local box is free: a runaway generation is
        # still minutes of somebody's afternoon.
        guards=Guards(stop_at_tokens=20000),
    )


@pytest.fixture
def store(tmp_path) -> ConnectionStore:
    base_url = _gate()
    store = ConnectionStore(tmp_path / "modelpass-home")
    store.add(_connection(base_url))
    return store


@pytest.fixture
def bridge(store) -> Bridge:
    return Bridge(store=store)


def _terminal(events: list) -> TerminalEvent:
    terminals = [e for e in events if isinstance(e, TerminalEvent)]
    assert len(terminals) == 1, "exactly one terminal event ends a run"
    return terminals[0]


def test_the_probe_reaches_a_real_server_and_says_what_it_serves(bridge):
    """The precondition for everything else, and a cell of its own on this
    runtime: the listing is the only reliable way to learn what the box has."""
    receipt = bridge.preflight(CONNECTION, options={"probe_credential": True})
    assert receipt.ok, receipt.problem
    assert any("credential probe: the endpoint answered" in n for n in receipt.notes)
    assert any("model(s) listed" in n for n in receipt.notes)


def test_an_unverified_connection_is_refused_before_verify_runs(bridge):
    """The honesty this runtime is built on, observed against a real endpoint:
    a server that is plainly working still cannot be chatted to until somebody
    has driven it, because the table has no way to know that it works."""
    from modelpass.errors import CapabilityNotSupported

    assert (
        bridge.registry.support(Runtime.OPENAI_COMPATIBLE, Capability.CHAT)
        is Support.UNVERIFIED
    )
    with pytest.raises(CapabilityNotSupported) as excinfo:
        list(bridge.chat(connection=CONNECTION, message="hello"))
    assert "modelpass verify" in str(excinfo.value)


def test_the_verify_drive_against_a_real_server(bridge, store):
    """The whole of ticket 1.10's per-install story, end to end and live.

    Deliberately asserts very little about *which* cells come back supported:
    that is the endpoint's answer, not modelpass's, and a test that demanded
    ``structured_output`` would fail against an older Ollama that is behaving
    correctly. What it does assert is that the drive reached the server, reached
    a verdict on every cell it claims to decide, and wrote something that reads
    back.
    """
    connection = bridge.connection(CONNECTION)
    adapter = bridge.adapter_for(Runtime.OPENAI_COMPATIBLE)
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="ok"),),
        plan=bridge.plan(connection),
    )
    report = adapter.verify_capabilities(request)
    assert report is not None
    assert report.ok, report.problem

    verified = report.verified
    decided = set(verified.supported) | set(verified.unsupported)
    # Every cell the drive claims to decide got a verdict, and chat got one
    # first -- a server that cannot chat skips the later two by design, so this
    # is only a full set where the chat succeeded.
    assert "chat" in decided
    if "chat" in verified.supported:
        assert decided == {str(cell) for cell in VERIFY_CELLS}
    assert verified.models, "the probe saw at least one model"
    assert verified.checked_at

    # It reads back off disk, and the connection can then chat.
    from dataclasses import replace

    store.add(replace(connection, verified_capabilities=verified), overwrite=True)
    reloaded = bridge.connection(CONNECTION)
    assert reloaded.verified_capabilities.supported == verified.supported
    if "chat" in verified.supported:
        assert bridge.registry_for(reloaded).supports(
            Runtime.OPENAI_COMPATIBLE, Capability.CHAT
        )


def test_one_short_chat_through_the_bridge(bridge, store):
    """The ordinary path, once the endpoint has been verified: chat, streaming,
    incremental text, and whatever the server chose to say about usage."""
    from dataclasses import replace

    connection = bridge.connection(CONNECTION)
    adapter = bridge.adapter_for(Runtime.OPENAI_COMPATIBLE)
    report = adapter.verify_capabilities(
        RunRequest(
            connection=connection,
            messages=(Message(role=Role.USER, content="ok"),),
            plan=bridge.plan(connection),
        )
    )
    if "chat" not in report.verified.supported:
        pytest.skip(f"this endpoint did not answer a plain chat: {report.notes}")
    store.add(
        replace(connection, verified_capabilities=report.verified), overwrite=True
    )

    events = list(
        bridge.chat(
            connection=CONNECTION,
            message="Reply with exactly the word: ready",
            sampling=Sampling(max_output_tokens=32, temperature=0.0),
        )
    )
    text = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert text.strip(), "a real server produced some text"
    assert _terminal(events).status is TerminalStatus.OK

    # Usage is the one thing a compatible server is genuinely free to withhold,
    # so this asserts the *consistency* rather than the presence: either a usage
    # event arrived with real numbers in it, or none arrived at all. What must
    # never happen is a usage event full of zeros.
    usage = [e for e in events if isinstance(e, UsageEvent)]
    for event in usage:
        assert event.usage.total_tokens > 0, (
            "a usage event with nothing in it is the zeroed report the adapter "
            "exists to avoid emitting"
        )


def test_a_no_credential_connection_really_works_against_a_local_box():
    """The ``credentialRef = "none"`` shape, driven rather than argued about.

    Skipped where the operator set a key, because then there is nothing to
    observe: the point is that a box which checks nothing is talked to with a
    placeholder and answers anyway.
    """
    base_url = _gate()
    if os.environ.get(KEY_VAR):
        pytest.skip(f"{KEY_VAR} is set, so this endpoint is not the unauthenticated case")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = ConnectionStore(Path(tmp) / "modelpass-home")
        store.add(_connection(base_url))
        bridge = Bridge(store=store)
        receipt = bridge.preflight(CONNECTION, options={"probe_credential": True})
        assert receipt.ok, receipt.problem
        assert receipt.account is None
        assert any("declares no credential" in note for note in receipt.notes)


def test_the_adapter_is_the_one_the_registry_hands_out(bridge):
    adapter = bridge.adapter_for(Runtime.OPENAI_COMPATIBLE)
    assert isinstance(adapter, OpenAICompatibleAdapter)
    assert adapter.runtime is Runtime.OPENAI_COMPATIBLE
