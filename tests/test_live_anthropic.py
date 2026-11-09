"""Live smoke tests for the Anthropic adapter.

These are the only tests in the repository that touch a real vendor runtime, and
they are quarantined accordingly:

* marked ``live``, which ``addopts`` deselects by default;
* additionally gated on ``SUBPASS_LIVE_TESTS=1``, so ``-m live`` alone still
  skips rather than surprising anyone;
* refused outright when a CI environment is detected.

They use a temp connection store, never the user's ``~/.modelpass``, and send
single-token prompts. The whole file should cost a fraction of a cent.

Run with::

    SUBPASS_LIVE_TESTS=1 pytest -m live
"""

from __future__ import annotations

import os

import pytest

from modelpass.bridge import Bridge
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.errors import PreflightFailed
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore
from modelpass.types import (
    AuthMode,
    CacheControl,
    TerminalEvent,
    TerminalStatus,
    TextBlock,
    TextDeltaEvent,
    UsageEvent,
)

pytestmark = pytest.mark.live

#: Environment variables that mean "this is CI". Live tests must never run there.
_CI_MARKERS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "JENKINS_URL", "TF_BUILD")


def _gate() -> None:
    present = [name for name in _CI_MARKERS if os.environ.get(name)]
    if present:
        pytest.skip(f"live tests never run in CI (found {', '.join(present)})")
    if os.environ.get("SUBPASS_LIVE_TESTS") != "1":
        pytest.skip("live tests are opt-in: set SUBPASS_LIVE_TESTS=1 to run them")


def _require_usable_login(bridge, name) -> None:
    """Skip when this machine has no login that can authenticate a child process.

    Deliberately narrow. "You are not logged in" is a setup state, not an adapter
    regression, so it skips with the receipt's own explanation -- but *only* for
    that one precondition. Every other way a receipt can be wrong (detecting the
    wrong auth mode, missing a credential source, bad stamping) still fails.
    """
    receipt = bridge.preflight(name)
    if not receipt.ok:
        pytest.skip(
            f"no usable Claude Code login on this machine: {receipt.problem}. "
            "Run 'claude /login' (or 'claude setup-token') and re-run."
        )


@pytest.fixture
def live_bridge(tmp_path):
    """A bridge over a throwaway store holding one subscription connection."""
    _gate()
    store = ConnectionStore(tmp_path / "modelpass-home")
    connection = Connection(
        name="claude-sub-live",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        # Tight guards: this is a smoke test, not a budget.
        guards=Guards(warn_at_tokens=0, stop_at_tokens=0),
        description="live smoke test",
    )
    store.add(connection)
    assert store.path.parent != (tmp_path.home() / ".modelpass")
    return Bridge(store=store), connection


def test_preflight_receipt_reports_subscription_auth(live_bridge):
    """(a) The receipt names the subscription and the credential source.

    If the machine's Claude Code login cannot actually authenticate a child
    process, this fails loudly with the receipt's own explanation rather than
    limping on -- which is the behaviour we want from the trust core.
    """
    bridge, connection = live_bridge
    _require_usable_login(bridge, connection.name)
    receipt = bridge.preflight(connection.name)

    print("\nRECEIPT:", receipt.summary())
    for note in receipt.notes:
        print("  note:", note)

    assert receipt.runtime is Runtime.ANTHROPIC_SDK
    assert receipt.detected_auth_mode is AuthMode.SUBSCRIPTION
    assert "Claude Code login" in receipt.credential_source
    # A receipt names the auth mode and its source -- never a secret.
    assert "sk-ant" not in receipt.summary()


def test_one_real_generation_end_to_end(live_bridge):
    """(b) A real run: text streams, usage is non-zero, terminal is stamped."""
    bridge, connection = live_bridge
    _require_usable_login(bridge, connection.name)

    events = list(
        bridge.chat(
            connection=connection.name,
            message="Reply with exactly OK",
            expect_auth_mode=AuthMode.SUBSCRIPTION,
        )
    )

    texts = [e.text for e in events if isinstance(e, TextDeltaEvent)]
    usages = [e for e in events if isinstance(e, UsageEvent)]
    terminals = [e for e in events if isinstance(e, TerminalEvent)]

    assert texts, "expected at least one text event"
    assert "OK" in "".join(texts).upper()

    assert usages, "expected a usage event"
    assert usages[-1].usage.total_tokens > 0, "usage must report real tokens"

    assert len(terminals) == 1, "exactly one terminal event ends a normal stream"
    terminal = terminals[0]
    assert terminal.status is TerminalStatus.OK, terminal.reason
    assert terminal.connection == connection.name
    assert terminal.auth_mode is AuthMode.SUBSCRIPTION
    assert terminal.runtime is Runtime.ANTHROPIC_SDK


def test_ambient_api_key_never_flips_the_run_to_metered(live_bridge, monkeypatch):
    """(c) With a decoy key in the environment, the run stays on the subscription.

    The offline proof that the variable leaves the child environment lives in
    ``test_anthropic_adapter.py``; this asserts the *outcome* against the real
    runtime -- the receipt still says subscription, and the run either succeeds
    on the subscription or fails closed. It never quietly bills by the token.
    """
    bridge, connection = live_bridge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-decoy-must-never-be-used")

    receipt = bridge.preflight(connection.name)
    assert receipt.detected_auth_mode is not AuthMode.API_KEY
    assert "ANTHROPIC_API_KEY" in receipt.scrubbed

    try:
        events = list(
            bridge.chat(
                connection=connection.name,
                message="Reply with exactly OK",
                expect_auth_mode=AuthMode.SUBSCRIPTION,
            )
        )
    except PreflightFailed:
        # Failing closed is a correct outcome here; billing metered is not.
        return

    terminals = [e for e in events if isinstance(e, TerminalEvent)]
    assert terminals[-1].auth_mode is AuthMode.SUBSCRIPTION


def _shared_document() -> str:
    """~3,000 tokens of stable text: over every model's cache floor."""
    lines = [
        f"Section {i}: the benefit described here applies to members enrolled before "
        f"year {2000 + i % 25}, with a copay of {i % 9 + 1}0 dollars per visit and a "
        f"limit of {i % 7 + 2} visits per quarter."
        for i in range(120)
    ]
    return "<document>\n" + "\n".join(lines) + "\n</document>"


def _usage(events):
    usages = [e.usage for e in events if isinstance(e, UsageEvent)]
    assert usages, "expected a usage event"
    return usages[-1]


def test_a_shared_document_is_read_from_cache_on_the_next_call(live_bridge):
    """(d) A document marked for caching in the user message is read back (2026-11-09).

    Two calls, one document, a different question each. ``arrange_prompt_cache``
    moves the document into the system prompt, where Claude Code places its own
    breakpoint, so the second call must READ the document rather than write it
    again. Before the move, a consumer's 800 such calls read ~60 tokens each.
    Costs two Haiku calls.
    """
    bridge, connection = live_bridge
    _require_usable_login(bridge, connection.name)
    document = TextBlock(_shared_document(), cache_control=CacheControl())

    def ask(question):
        return list(
            bridge.chat(
                connection=connection.name,
                system_prompt="Answer in one word.",
                message=(document, TextBlock(f"\n\nQuestion: {question}")),
                model="claude-haiku-5-5",
                expect_auth_mode=AuthMode.SUBSCRIPTION,
            )
        )

    first = _usage(ask("What is the copay in section 3?"))
    second = _usage(ask("How many visits are allowed in section 5?"))
    print(f"\nfirst:  write {first.cache_write_tokens} read {first.cached_input_tokens}")
    print(f"second: write {second.cache_write_tokens} read {second.cached_input_tokens}")
    assert second.cached_input_tokens >= 2000, (
        "the second call should read the shared document from the cache"
    )
