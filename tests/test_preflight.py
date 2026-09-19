from __future__ import annotations

import os
from dataclasses import replace

import pytest

from modelpass.connections import Connection, CredentialRef
from modelpass.errors import UnsafeLaunch
from modelpass.preflight import (
    FORBIDDEN_LAUNCH_ARGS,
    Receipt,
    check_launch_args,
    directives_for,
    env_names_to_scrub,
    plan_launch,
    preserved_env_names,
    scrub_env,
)
from modelpass.runtimes import Runtime
from modelpass.types import AuthMode


def test_scrub_matches_names_and_prefixes():
    env = {
        "ANTHROPIC_API_KEY": "x",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_USE_VERTEX": "1",
        "PATH": "/usr/bin",
    }
    assert env_names_to_scrub(Runtime.ANTHROPIC_SDK, env) == (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
    )
    assert scrub_env(Runtime.ANTHROPIC_SDK, env) == {"PATH": "/usr/bin"}


def test_scrub_only_reports_variables_that_are_actually_present():
    assert env_names_to_scrub(Runtime.ANTHROPIC_SDK, {"PATH": "/usr/bin"}) == ()


def test_scrub_is_per_runtime():
    env = {"CODEX_API_KEY": "x", "OPENAI_API_KEY": "y", "ANTHROPIC_API_KEY": "z"}
    assert env_names_to_scrub(Runtime.OPENAI_SDK, env) == ("CODEX_API_KEY", "OPENAI_API_KEY")
    assert env_names_to_scrub(Runtime.GOOGLE_CLI, env) == ()


def test_preserve_wins_over_scrub():
    env = {"OPENAI_API_KEY": "x", "CODEX_API_KEY": "y"}
    assert env_names_to_scrub(Runtime.OPENAI_SDK, env, preserve=["OPENAI_API_KEY"]) == (
        "CODEX_API_KEY",
    )
    assert scrub_env(Runtime.OPENAI_SDK, env, preserve=["OPENAI_API_KEY"]) == {
        "OPENAI_API_KEY": "x"
    }


def test_scrub_returns_a_new_mapping():
    env = {"ANTHROPIC_API_KEY": "x"}
    result = scrub_env(Runtime.ANTHROPIC_SDK, env)
    assert result == {}
    assert env == {"ANTHROPIC_API_KEY": "x"}


def test_preserved_env_names_only_covers_env_credential_refs(
    subscription_connection, api_connection
):
    assert preserved_env_names(subscription_connection) == ()
    assert preserved_env_names(api_connection) == ("ANTHROPIC_API_KEY",)


def test_plan_launch_defaults_to_the_process_environment(subscription_connection, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    plan = plan_launch(subscription_connection)
    assert "ANTHROPIC_API_KEY" in plan.scrubbed
    assert "ANTHROPIC_API_KEY" not in plan.env
    assert "ANTHROPIC_API_KEY" in os.environ


def test_an_ambient_claude_config_dir_is_scrubbed(subscription_connection):
    plan = plan_launch(
        subscription_connection,
        {"CLAUDE_CONFIG_DIR": "/wrong-account", "PATH": "/usr/bin"},
    )
    assert plan.scrubbed == ("CLAUDE_CONFIG_DIR",)
    assert "CLAUDE_CONFIG_DIR" not in plan.env


def test_a_connection_config_dir_replaces_the_ambient_account(
    subscription_connection, tmp_path
):
    selected = tmp_path / "claude-work"
    connection = Connection(
        name="claude-work",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        config_dir=str(selected),
    )
    plan = plan_launch(connection, {"CLAUDE_CONFIG_DIR": str(tmp_path / "wrong")})
    assert plan.env["CLAUDE_CONFIG_DIR"] == str(selected)
    assert plan.config_dir == str(selected)
    assert plan.scrubbed == ()
    assert plan.kept == ("CLAUDE_CONFIG_DIR",)


def test_an_openai_account_config_dir_maps_to_codex_home(tmp_path):
    selected = tmp_path / "codex-work"
    connection = Connection(
        name="openai-work",
        nickname="OpenAI Work",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        config_dir=str(selected),
    )
    plan = plan_launch(connection, {"CODEX_HOME": str(tmp_path / "wrong")})
    assert plan.env["CODEX_HOME"] == str(selected)
    assert plan.config_env_name == "CODEX_HOME"
    assert plan.kept == ("CODEX_HOME",)


def test_an_ambient_codex_home_is_scrubbed_from_the_default_account():
    connection = Connection(
        name="openai-default",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
    )
    plan = plan_launch(connection, {"CODEX_HOME": "/wrong-account", "PATH": "/bin"})
    assert plan.scrubbed == ("CODEX_HOME",)
    assert "CODEX_HOME" not in plan.env


def test_anthropic_directives_cover_the_non_environment_traps(subscription_connection):
    directives = {d.name: d.value for d in directives_for(subscription_connection)}
    assert directives["settings.apiKeyHelper"] == "disabled"
    assert directives["persistSession"] == "false"


def test_openai_directives_pin_the_login_method():
    subscription = Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
    )
    metered = Connection(
        name="codex-api",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:OPENAI_API_KEY"),
    )
    assert dict(
        (d.name, d.value) for d in directives_for(subscription)
    )["forced_login_method"] == "chatgpt"
    assert dict((d.name, d.value) for d in directives_for(metered))["forced_login_method"] == (
        "apikey"
    )


def test_bare_is_never_passed_to_the_claude_runtime():
    assert "--bare" in FORBIDDEN_LAUNCH_ARGS[Runtime.ANTHROPIC_SDK]
    check_launch_args(Runtime.ANTHROPIC_SDK, ["-p", "--output-format", "stream-json"])
    with pytest.raises(UnsafeLaunch, match="--bare"):
        check_launch_args(Runtime.ANTHROPIC_SDK, ["-p", "--bare"])
    with pytest.raises(UnsafeLaunch):
        check_launch_args(Runtime.ANTHROPIC_SDK, ["--bare=true"])


def test_plan_describes_itself_for_a_receipt(subscription_connection):
    plan = plan_launch(subscription_connection, {"ANTHROPIC_API_KEY": "x"})
    assert "claude-sub" in plan.describe()
    assert "scrubbed ANTHROPIC_API_KEY" in plan.describe()


def test_receipt_from_plan_carries_the_vendor_independent_facts(subscription_connection):
    plan = plan_launch(subscription_connection, {"ANTHROPIC_API_KEY": "x"})
    receipt = Receipt.from_plan(
        plan,
        detected_auth_mode=AuthMode.SUBSCRIPTION,
        credential_source="Claude Code login",
        account="someone@example.com",
        plan_name="Max 5x",
    )
    summary = receipt.summary()
    assert "claude-sub" in summary
    assert "subscription mode" in summary
    assert "Max 5x" in summary
    assert "scrubbed ANTHROPIC_API_KEY" in summary


def test_the_summary_names_the_config_dir_by_its_own_vendor_variable(
    subscription_connection, tmp_path
):
    """"Claude config <dir>" on an openai-sdk receipt named the wrong variable."""
    claude = replace(subscription_connection, config_dir=str(tmp_path / "claude-work"))
    summary = Receipt.from_plan(plan_launch(claude, {})).summary()
    assert f"CLAUDE_CONFIG_DIR {tmp_path / 'claude-work'}" in summary

    codex = Connection(
        name="openai-work",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        config_dir=str(tmp_path / "codex-work"),
    )
    summary = Receipt.from_plan(plan_launch(codex, {})).summary()
    assert f"CODEX_HOME {tmp_path / 'codex-work'}" in summary
    assert "Claude config" not in summary


def test_receipt_rejects_a_run_that_would_bill_differently(subscription_connection):
    plan = plan_launch(subscription_connection, {})
    receipt = Receipt.from_plan(plan, detected_auth_mode=AuthMode.API_KEY)
    with pytest.raises(Exception) as excinfo:
        receipt.require_auth_mode()
    assert "expected auth mode" in str(excinfo.value)


def test_receipt_require_ok_reports_the_problem(subscription_connection):
    plan = plan_launch(subscription_connection, {})
    receipt = Receipt.from_plan(plan, ok=False, problem="no login found")
    with pytest.raises(Exception, match="no login found"):
        receipt.require_ok()


# --- options that will not be read -----------------------------------------------


def test_a_closed_adapter_names_option_keys_it_will_ignore():
    """An ignored override must not look like a working one.

    ``codex_bin`` is the documented lever for which binary actually runs, so a
    typo there silently runs the default -- the same failure shape modelpass warns
    consumers about in Codex's own unknown ``-c`` keys.
    """
    from modelpass.adapters.openai import OpenAIAdapter

    assert OpenAIAdapter.unknown_option_keys({"codex_bn": "x"}) == ("codex_bn",)
    assert OpenAIAdapter.unknown_option_keys({"codex_bin": "x", "cwd": "."}) == ()
    # A Codex key on the wrong runtime is the other half of the same mistake.
    assert OpenAIAdapter.unknown_option_keys({"extra_args": {}}) == ("extra_args",)


def test_an_open_adapter_is_never_second_guessed():
    """Anthropic forwards unknown keys to ClaudeAgentOptions.

    Declaring a closed set there would turn a documented passthrough into a
    refusal, so it declares none and nothing is reported.
    """
    from modelpass.adapters.anthropic import AnthropicAdapter

    assert AnthropicAdapter.option_keys is None
    assert AnthropicAdapter.unknown_option_keys({"anything_at_all": 1}) == ()
