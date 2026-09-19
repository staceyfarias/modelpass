"""Which model is this run about to ask for? (first-consumer feedback, 2026-08-17)

The consumer hit a live vendor rejection -- ``400 ... The 'gpt-5.6-sol' model
requires a newer version of Codex`` -- and could not diagnose it, because
nothing produced *before* a run named a model. The receipt named the runtime,
the auth mode, the account, the plan and the scrub list, and was silent about
the one field the failure was about.

So the receipt names the model, and says where the name came from. Precedence,
most specific first:

1. what the caller passed to ``chat`` / ``preflight``;
2. the connection's own ``model``;
3. what the vendor runtime is configured to default to -- read from the
   vendor's config, never written to it;
4. nothing, which is a real answer: the runtime's own default applies and
   modelpass says so rather than guessing.
"""

from __future__ import annotations

import pytest

from modelpass.adapters.base import RunRequest
from modelpass.adapters.openai import OpenAIAdapter, codex_home, read_configured_model
from modelpass.connections import Connection, CredentialRef
from modelpass.preflight import plan_launch
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode, Message, ReceiptEvent, Role

MESSAGE = "hi"


# --- precedence, over the vendor-independent half --------------------------------


def test_a_connection_model_reaches_the_receipt_and_says_where_it_came_from(
    bridge_factory, subscription_connection
):
    pinned = Connection(
        name=subscription_connection.name,
        runtime=subscription_connection.runtime,
        auth_mode=subscription_connection.auth_mode,
        guards=subscription_connection.guards,
        model="claude-opus-4",
    )
    bridge, _ = bridge_factory(pinned, FakeAdapter([]))
    receipt = bridge.preflight("claude-sub")
    assert receipt.model == "claude-opus-4"
    assert receipt.model_source == "from connection 'claude-sub'"
    assert "model claude-opus-4 (from connection 'claude-sub')" in receipt.summary()


def test_the_callers_model_wins_and_the_receipt_says_so(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([]))
    receipt = bridge.preflight("claude-sub", model="claude-haiku-4")
    assert receipt.model == "claude-haiku-4"
    assert receipt.model_source == "requested for this call"


def test_the_receipt_event_of_a_chat_call_carries_the_model_the_call_asked_for(
    bridge_factory, subscription_connection
):
    """Which is the whole point: diagnosable *before* the run, from the stream."""
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([]))
    event = next(
        iter(bridge.chat(connection="claude-sub", message=MESSAGE, model="gpt-nope"))
    )
    assert isinstance(event, ReceiptEvent)
    assert event.receipt.model == "gpt-nope"
    assert event.receipt.model_source == "requested for this call"
    assert event.receipt.to_dict()["model"] == "gpt-nope"


def test_no_model_anywhere_is_a_stated_absence_not_a_blank(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([]))
    receipt = bridge.preflight("claude-sub")
    assert receipt.model is None
    assert receipt.model_source == ""
    assert "model not named; the runtime's own default applies" in receipt.summary()
    assert receipt.model_note is not None
    assert "the runtime's own default applies" in receipt.model_note


def test_a_named_model_has_no_note_to_print(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(subscription_connection, FakeAdapter([]))
    assert bridge.preflight("claude-sub", model="x").model_note is None


# --- the OpenAI half: reading the runtime's own configured default ----------------


def codex_connection(model: str | None = None) -> Connection:
    return Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        model=model,
    )


def request_for(connection: Connection, env: dict[str, str] | None = None) -> RunRequest:
    plan = plan_launch(connection, env if env is not None else {"PATH": "/usr/bin"})
    return RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan,
        model=plan.model,
    )


def logged_in(binary, env):
    return "Logged in using ChatGPT"


def write_codex_config(root, body: str):
    home = root / ".codex"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(body, encoding="utf-8")
    return home


def test_the_configured_codex_model_reaches_the_receipt(tmp_path):
    home = write_codex_config(tmp_path, 'model = "gpt-5.6-sol"\napproval_policy = "never"\n')
    adapter = OpenAIAdapter(codex_bin="codex", login_status=logged_in, codex_home=home)
    receipt = adapter.preflight(request_for(codex_connection()))

    assert receipt.model == "gpt-5.6-sol"
    assert receipt.model_source.startswith("from ")
    assert receipt.model_source.endswith("config.toml")
    assert "model gpt-5.6-sol (from " in receipt.summary()


def test_a_model_named_by_the_caller_beats_the_codex_config(tmp_path):
    home = write_codex_config(tmp_path, 'model = "gpt-5.6-sol"\n')
    adapter = OpenAIAdapter(codex_bin="codex", login_status=logged_in, codex_home=home)
    receipt = adapter.preflight(request_for(codex_connection(model="gpt-5.4")))

    assert receipt.model == "gpt-5.4"
    assert receipt.model_source == "from connection 'codex-sub'"


def test_a_missing_or_malformed_codex_config_is_silence_not_a_failure(tmp_path):
    """Not being able to name the model must never be why a run does not happen."""
    adapter = OpenAIAdapter(
        codex_bin="codex", login_status=logged_in, codex_home=tmp_path / "nope"
    )
    receipt = adapter.preflight(request_for(codex_connection()))
    assert receipt.ok is True
    assert receipt.model is None

    broken = write_codex_config(tmp_path, "model = this is not toml [[[\n")
    adapter = OpenAIAdapter(codex_bin="codex", login_status=logged_in, codex_home=broken)
    receipt = adapter.preflight(request_for(codex_connection()))
    assert receipt.ok is True
    assert receipt.model is None


def test_a_codex_config_without_a_model_key_names_nothing(tmp_path):
    home = write_codex_config(tmp_path, 'approval_policy = "never"\n')
    assert read_configured_model(home) == (None, "")


def test_reading_the_codex_config_never_writes_to_it(tmp_path):
    """modelpass reads a vendor's configuration; it does not own or edit one."""
    home = write_codex_config(tmp_path, 'model = "gpt-5.6-sol"\n')
    path = home / "config.toml"
    before = path.read_bytes(), path.stat().st_mtime_ns

    adapter = OpenAIAdapter(codex_bin="codex", login_status=logged_in, codex_home=home)
    adapter.preflight(request_for(codex_connection()))

    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert sorted(p.name for p in home.iterdir()) == ["config.toml"]


def test_codex_home_follows_the_runtimes_own_environment_variable(tmp_path):
    relocated = tmp_path / "elsewhere"
    assert codex_home({"CODEX_HOME": str(relocated)}) == relocated
    assert codex_home({}, relocated) == relocated
    # Read off the scrubbed plan environment, like everything else the adapter uses.
    assert codex_home({}).name == ".codex"


@pytest.mark.parametrize("option_key", ["codex_home"])
def test_the_codex_home_is_injectable_per_call(tmp_path, option_key):
    home = write_codex_config(tmp_path, 'model = "gpt-5.6-sol"\n')
    request = request_for(codex_connection())
    request = RunRequest(
        connection=request.connection,
        messages=request.messages,
        plan=request.plan,
        options={option_key: str(home)},
    )
    adapter = OpenAIAdapter(codex_bin="codex", login_status=logged_in)
    assert adapter.preflight(request).model == "gpt-5.6-sol"
