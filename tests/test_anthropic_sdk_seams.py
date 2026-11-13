"""The places ``anthropic-sdk`` leans on the *installed* ``claude-agent-sdk`` (2026-11-13).

Every other test of this adapter runs against classes shaped like the vendor's,
so that a machine without the ``anthropic`` extra exercises the same code. This
file is the deliberate exception: it imports the installed package, because the
three things it holds are facts about that package which a fake can only repeat
back.

1. **The SDK still builds the child environment from** ``os.environ``. That is
   the only reason :func:`modelpass.adapters.anthropic._scrubbed_process_env`
   exists. If this fails, the SDK has changed how it composes the environment:
   re-read ``SubprocessCLITransport.connect`` before touching the scrub, and
   delete the scrub if an explicit environment can now be passed.
2. **A transport can be handed to** ``ClaudeSDKClient``, which is what lets the
   scrub cover the spawn and nothing after it.
3. **The private names** ``_cli_status`` **uses are still there.**

Nothing here starts Claude Code. ``anyio.open_process`` is replaced before any
transport connects, so no process is spawned, no login is read and nothing is
spent. Skipped where the package is not installed.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import threading

import pytest

sdk = pytest.importorskip("claude_agent_sdk")

from modelpass.adapters import anthropic as module  # noqa: E402
from modelpass.connections import Connection, CredentialRef  # noqa: E402
from modelpass.preflight import plan_launch  # noqa: E402
from modelpass.runtimes import Runtime  # noqa: E402
from modelpass.types import AuthMode  # noqa: E402

subprocess_cli = pytest.importorskip("claude_agent_sdk._internal.transport.subprocess_cli")


class _Spawned(RuntimeError):
    """Raised in place of a spawn, after recording what it would have been given."""


@pytest.fixture
def spawn(monkeypatch):
    """Replace the SDK's process spawn with a recorder. Nothing is launched."""
    seen: list[dict] = []

    async def open_process(command, **kwargs):
        seen.append(
            {
                "command": list(command),
                "env": dict(kwargs.get("env") or {}),
                "ambient_key_in_os_environ": "ANTHROPIC_API_KEY" in os.environ,
                "ambient_key_for_modelpass": module.process_env().get("ANTHROPIC_API_KEY"),
            }
        )
        raise _Spawned("no process is started by this test")

    monkeypatch.setattr(subprocess_cli.anyio, "open_process", open_process)
    # The transport's own version probe would be a second spawn of the binary.
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    return seen


def _options(**env: str):
    # ``cli_path`` only has to be a native executable the SDK will not refuse;
    # it is never run.
    return sdk.ClaudeAgentOptions(cli_path=sys.executable, env=dict(env))


def _subscription() -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


async def _empty_prompt():
    return
    yield {}  # pragma: no cover - makes this an async generator


# --- 1. why the scrub exists -------------------------------------------------------


def test_the_installed_sdk_still_builds_the_child_environment_from_os_environ(
    monkeypatch, spawn
):
    monkeypatch.setenv("MODELPASS_SEAM_AMBIENT", "from-the-process")
    monkeypatch.setenv("MODELPASS_SEAM_BOTH", "from-the-process")
    transport = subprocess_cli.SubprocessCLITransport(
        prompt=_empty_prompt(),
        options=_options(MODELPASS_SEAM_ADDED="from-options", MODELPASS_SEAM_BOTH="from-options"),
    )
    with pytest.raises(Exception, match="no process is started"):
        asyncio.run(transport.connect())

    (launch,) = spawn
    env = launch["env"]
    # Inherited: nothing in ``options`` asked for this variable, and it is there.
    # ``options.env`` can add and override -- and has no way to take it out.
    assert env["MODELPASS_SEAM_AMBIENT"] == "from-the-process"
    assert env["MODELPASS_SEAM_ADDED"] == "from-options"
    assert env["MODELPASS_SEAM_BOTH"] == "from-options"


def test_the_installed_options_still_have_no_way_to_replace_the_environment():
    """The day one of these appears, the scrub can go: read it and use it."""
    fields = set(inspect.signature(sdk.ClaudeAgentOptions).parameters)
    assert "env" in fields
    assert not fields & {"inherit_env", "replace_env", "clear_env", "env_replace"}


# --- 2. the scrub covers the spawn, and only the spawn ------------------------------


def test_a_client_built_for_a_plan_spawns_without_the_ambient_key(monkeypatch, spawn):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")
    plan = plan_launch(_subscription(), {})
    client, spawn_scrubbed = module._planned_client(sdk, _options(), plan)
    assert spawn_scrubbed is True, "the installed SDK takes a transport"

    with pytest.raises(Exception, match="no process is started"):
        asyncio.run(module._connect_planned(client, spawn_scrubbed, plan, None))

    (launch,) = spawn
    assert "ANTHROPIC_API_KEY" not in launch["env"]
    assert launch["ambient_key_in_os_environ"] is False
    # modelpass's own readers are not shown the hole the spawn needs.
    assert launch["ambient_key_for_modelpass"] == "sk-ant-ambient"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-ambient"
    assert module.process_env() is os.environ


def test_the_environment_is_whole_again_before_the_handshake(monkeypatch, spawn):
    """The scrub used to span ``connect()``: the spawn *and* the initialize handshake.

    A hung handshake therefore held every other ``anthropic-sdk`` spawn in the
    process, and kept the key out of ``os.environ``, for as long as it hung.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")
    plan = plan_launch(_subscription(), {})
    inside: dict[str, bool] = {}

    async def spawned(self):
        inside["key_during_spawn"] = "ANTHROPIC_API_KEY" in os.environ

    monkeypatch.setattr(subprocess_cli.SubprocessCLITransport, "connect", spawned)
    transport = module._planned_env_transport(sdk, _options(), plan)
    assert transport is not None

    async def connect_then_look():
        await transport.connect()
        # Where ``ClaudeSDKClient.connect`` goes on to ``query.initialize()``.
        return "ANTHROPIC_API_KEY" in os.environ, module._ENV_LOCK._is_owned()

    result: dict[str, object] = {}
    thread = threading.Thread(
        target=lambda: result.update(after=asyncio.run(connect_then_look()))
    )
    thread.start()
    thread.join(10)
    assert inside == {"key_during_spawn": False}
    # The key is back, and this thread no longer holds the lock other spawns
    # queue on.
    assert result["after"] == (True, False)


def test_options_the_sdk_only_applies_to_its_own_transport_keep_the_old_path():
    """``can_use_tool`` and ``session_store`` are configured by the client for the
    transport it builds itself, and skipped for one it is handed."""
    plan = plan_launch(_subscription(), {})

    async def allow(*args):  # pragma: no cover - never called
        return None

    options = sdk.ClaudeAgentOptions(cli_path=sys.executable, can_use_tool=allow)
    client, spawn_scrubbed = module._planned_client(sdk, options, plan)
    assert spawn_scrubbed is False
    assert client._custom_transport is None


def test_a_stand_in_sdk_keeps_the_old_path(subscription_connection):
    """The fakes the rest of the suite runs on take ``options`` and nothing else."""

    class Sdk:
        def ClaudeSDKClient(self, options):
            return ("client", options)

    plan = plan_launch(subscription_connection, {})
    assert module._planned_client(Sdk(), {"x": 1}, plan) == (("client", {"x": 1}), False)


# --- 3. the private names the binary lookup uses -------------------------------------


def test_the_private_names_the_binary_lookup_uses_are_still_there():
    """``_cli_status`` falls back to ``PATH`` if these go, which is quieter than
    it should be: a wheel with a bundled binary would stop being preferred."""
    transport = subprocess_cli.SubprocessCLITransport(
        prompt="", options=sdk.ClaudeAgentOptions()
    )
    assert callable(getattr(transport, "_find_cli", None))
    assert "transport" in inspect.signature(sdk.ClaudeSDKClient.__init__).parameters
    assert issubclass(subprocess_cli.SubprocessCLITransport, sdk.Transport)


def test_the_message_classes_matched_by_name_still_carry_those_names():
    """``map_message`` and the session handle dispatch on ``type(message).__name__``."""
    for name in ("SystemMessage", "ResultMessage", "AssistantMessage", "UserMessage",
                 "StreamEvent", "RateLimitEvent", "ToolUseBlock", "ToolResultBlock"):
        assert getattr(sdk, name).__name__ == name
