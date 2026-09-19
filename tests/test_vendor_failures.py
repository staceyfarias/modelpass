"""Vendor failures end the stream; they do not escape it (2026-08-17).

Reported live by the first consumer: a vendor 400 (a rejected model) surfaced as
``AdapterFailed`` raised mid-iteration. Two things broke at once.

* The stream contract's "exactly one terminal event" did not hold for the single
  most likely way a real run fails.
* The caller lost the D3 stamp *and* the accumulated usage of a run that
  genuinely spent tokens -- so the one thing it could not answer afterwards was
  "what did that cost me", about a run it is going to be billed for.

The ruling implemented here: **a vendor-reported failure is a run outcome, not a
caller mistake.** It ends the stream with one terminal, ``status=error``, the
reason, the usage, the stamp. Exceptions after the iterator exists are for
caller mistakes modelpass could not catch earlier and for genuine bugs -- and the
boundary between the two is a documented one, not a judgement call: an adapter
raises ``VendorRunFailed`` for anything the vendor did, and raising anything else
means the adapter is broken.
"""

from __future__ import annotations

import pytest

from modelpass.adapters.anthropic import classify_pump_failure
from modelpass.adapters.openai import OpenAIAdapter
from modelpass.connections import Guards
from modelpass.errors import (
    AdapterFailed,
    AuthModeMismatch,
    SubpassError,
    UnsafeLaunch,
    VendorRunFailed,
)
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter, usage, vendor_failure
from modelpass.types import (
    AuthMode,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
)

MESSAGE = "hi"

REJECTED = (
    "400: The 'gpt-5.6-sol' model requires a newer version of Codex. "
    "Please upgrade to the latest app or CLI and try again."
)


# --- the ruling -----------------------------------------------------------------


def test_a_vendor_failure_ends_the_stream_with_one_terminal(
    bridge_factory, subscription_connection
):
    fake = FakeAdapter(
        [TextDeltaEvent(text="partial answer")], error=vendor_failure(REJECTED)
    )
    bridge, _ = bridge_factory(subscription_connection, fake)

    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))

    terminals = [e for e in events if isinstance(e, TerminalEvent)]
    assert len(terminals) == 1
    assert terminals[0] is events[-1]
    assert terminals[0].status is TerminalStatus.ERROR
    assert "gpt-5.6-sol" in terminals[0].reason


def test_the_usage_of_a_run_that_spent_tokens_survives_the_failure(
    bridge_factory, subscription_connection
):
    """The part that actually hurt: a paid-for run reporting nothing about spend."""
    fake = FakeAdapter(
        [usage(input_tokens=1200, output_tokens=40), TextDeltaEvent(text="partial")],
        error=vendor_failure(REJECTED),
    )
    bridge, _ = bridge_factory(subscription_connection, fake)
    terminal = list(
        bridge.chat(
            connection="claude-sub", message=MESSAGE, guards=Guards.disabled()
        )
    )[-1]

    assert terminal.status is TerminalStatus.ERROR
    assert terminal.usage.input_tokens == 1200
    assert terminal.usage.output_tokens == 40
    assert terminal.usage.total_tokens == 1240


def test_the_stamp_survives_the_failure(bridge_factory, subscription_connection):
    """D3 is a guarantee for every run, and a failed run is still a run."""
    bridge, _ = bridge_factory(
        subscription_connection, FakeAdapter([], error=vendor_failure(REJECTED))
    )
    terminal = list(bridge.chat(connection="claude-sub", message=MESSAGE))[-1]
    assert terminal.connection == "claude-sub"
    assert terminal.runtime is Runtime.ANTHROPIC_SDK
    assert terminal.auth_mode is AuthMode.SUBSCRIPTION


def test_the_text_written_before_the_failure_is_kept(
    bridge_factory, subscription_connection
):
    """Throwing away paid-for words to report a clean failure is the worse trade."""
    fake = FakeAdapter(
        [TextDeltaEvent(text="as far as "), TextDeltaEvent(text="it got")],
        error=vendor_failure(REJECTED),
    )
    bridge, _ = bridge_factory(subscription_connection, fake)
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert "".join(e.text for e in events if isinstance(e, TextDeltaEvent)) == (
        "as far as it got"
    )


def test_the_adapter_is_torn_down_after_a_vendor_failure(
    bridge_factory, subscription_connection
):
    bridge, fake = bridge_factory(
        subscription_connection, FakeAdapter([], error=vendor_failure())
    )
    list(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert fake.cancelled == 1


def test_a_vendor_failure_does_not_trigger_a_configured_failover(
    bridge_factory, subscription_connection
):
    """Failover answers "the plan ran out", not "the vendor rejected the request"."""
    from modelpass.connections import Connection, CredentialRef, QuotaAction, QuotaPolicy

    connection = subscription_connection.with_guards(
        Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api"))
    )
    bridge, fake = bridge_factory(connection, FakeAdapter([], error=vendor_failure()))
    bridge.store.add(
        Connection(
            name="claude-api",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.API_KEY,
            credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
        )
    )
    events = list(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert events[-1].status is TerminalStatus.ERROR
    assert [r.connection.name for r in fake.requests] == ["claude-sub"]


# --- the other side of the boundary ---------------------------------------------


def test_a_genuine_adapter_bug_still_raises_adapter_failed(
    bridge_factory, subscription_connection
):
    """An adapter has a documented way to report a vendor failure. This is not it."""
    fake = FakeAdapter([TextDeltaEvent(text="a")], error=RuntimeError("vendor exploded"))
    bridge, _ = bridge_factory(subscription_connection, fake)
    with pytest.raises(AdapterFailed, match="vendor exploded") as excinfo:
        list(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert fake.cancelled == 1


@pytest.mark.parametrize(
    "error",
    [
        AuthModeMismatch("subscription", "api_key", "claude-sub"),
        UnsafeLaunch("--bare is never passed to this runtime"),
    ],
    ids=["auth_mode_mismatch", "unsafe_launch"],
)
def test_guaranteed_layer_refusals_keep_raising(
    bridge_factory, subscription_connection, error
):
    """D4 layer 1 is not configurable and not downgradable to a status line.

    A run that turned out to be billing the wrong way, or a launch assembled with
    an argument that forces metered billing, must stop the caller -- not arrive
    as one more terminal status the caller might not be matching on.
    """
    bridge, _ = bridge_factory(
        subscription_connection, FakeAdapter([TextDeltaEvent(text="a")], error=error)
    )
    with pytest.raises(type(error)):
        list(bridge.chat(connection="claude-sub", message=MESSAGE))


def test_vendor_run_failed_is_catchable_as_a_modelpass_error():
    """It should never reach a caller, but the taxonomy stays whole if it does."""
    assert issubclass(VendorRunFailed, SubpassError)


# --- the anthropic classification, which is where the leak actually was ----------


class _FakeSDKError(Exception):
    """Stands in for a claude_agent_sdk exception without importing the SDK."""


_FakeSDKError.__module__ = "claude_agent_sdk._errors"


def test_an_sdk_exception_is_classified_as_a_vendor_failure():
    """The exact path the live 400 escaped through: the SDK raising out of its
    async iterator instead of reporting a ResultMessage."""
    original = _FakeSDKError("Claude Code process exited with code 1")
    classified = classify_pump_failure(original)
    assert isinstance(classified, VendorRunFailed)
    assert "exited with code 1" in str(classified)


def test_anything_else_keeps_its_identity_and_stays_an_exception():
    for original in (
        TypeError("map_message got a surprise"),
        AttributeError("no such field"),
        KeyboardInterrupt(),
    ):
        assert classify_pump_failure(original) is original


def test_a_modelpass_error_out_of_the_pump_is_never_reclassified():
    """assert_expected_auth_mode's refusal travels this path and must keep raising."""
    refusal = AuthModeMismatch("subscription", "api_key")
    assert classify_pump_failure(refusal) is refusal

# --- the openai side: a runtime that will not launch -----------------------------
#
# Local fakes rather than an import from test_adapter_openai: these two files pin
# opposite halves of the same contract and should not be able to drift together.


def codex_request():
    from modelpass.adapters.base import RunRequest
    from modelpass.connections import Connection, CredentialRef
    from modelpass.preflight import plan_launch
    from modelpass.types import Message, Role

    connection = Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )
    # Exec, explicitly: these tests script a 'codex exec --json' stream and a
    # dying pipe on that transport. S7 made app-server the default on
    # 2026-08-31 without touching this path (see test_adapter_openai.py's EXEC).
    return RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection, {"PATH": "/usr/bin"}),
        options={"transport": "exec"},
    )


def test_a_codex_binary_that_cannot_be_launched_is_a_reported_failure():
    """WinError 193 on the npm shim, and every other spawn failure."""

    def refuse(argv, env, cwd):
        raise OSError(193, "%1 is not a valid Win32 application")

    adapter = OpenAIAdapter(codex_bin="codex", spawn=refuse)
    with pytest.raises(VendorRunFailed, match="could not be launched"):
        list(adapter.run(codex_request()))


def test_a_dying_pipe_mid_run_is_a_reported_failure():
    class DyingProc:
        stdout = iter(())
        returncode = 0

        def wait(self, timeout=None):
            raise OSError("the pipe went away")

        def poll(self):
            return 0

        def kill(self):
            pass

    adapter = OpenAIAdapter(codex_bin="codex", spawn=lambda a, e, c: DyingProc())
    with pytest.raises(VendorRunFailed, match="stream failed"):
        list(adapter.run(codex_request()))


def test_the_codex_json_failure_path_still_reports_a_terminal_not_an_exception():
    """It always did -- pinned here so both halves are visible in one place.

    Worth being explicit about, because it is what the investigation turned up:
    the documented `codex exec --json` failure vocabulary was never the leak. A
    400 that arrives as an `error` or `turn.failed` payload has always become a
    terminal event. What escaped was the transport around it.
    """
    import json as _json

    payloads = [
        {"type": "thread.started", "thread_id": "t"},
        {
            "type": "error",
            "message": (
                '{"type":"error","status":400,"error":{"type":"invalid_request_error",'
                '"message":"The \'gpt-5.6-sol\' model requires a newer version of '
                'Codex."}}'
            ),
        },
    ]

    class FakeProc:
        def __init__(self):
            self.stdout = iter(_json.dumps(p) + "\n" for p in payloads)
            self.returncode = 1

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return None

        def kill(self):
            pass

    adapter = OpenAIAdapter(codex_bin="codex", spawn=lambda a, e, c: FakeProc())
    events = list(adapter.run(codex_request()))
    assert events[-1].status is TerminalStatus.ERROR
    assert "400" in events[-1].reason
