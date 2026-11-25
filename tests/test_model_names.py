"""Short series names as model names (2026-11-25).

A caller or a connection may write ``sol``, ``astra``, ``terra`` or ``luna`` on
an OpenAI runtime, or ``haiku``, ``sonnet``, ``opus`` or ``fable`` on an
Anthropic one, where a model id goes, in any case. Where modelpass has a model
list it sends the newest id of that series; the Claude subscription has none, so
there the lower-cased alias goes to the runtime to resolve. The receipt says
what was written and what was sent. Everything else is sent as typed.

All offline: the Codex list is a cache file in a temporary directory and the API
lists come from fake clients.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as Obj

import pytest

from modelpass import Bridge, Connection, CredentialRef, NoSuchModel, Runtime
from modelpass.adapters.anthropic_api import AnthropicAPIAdapter
from modelpass.adapters.openai_api import OpenAIAPIAdapter
from modelpass.connections import Guards, QuotaAction, QuotaPolicy
from modelpass.store import ConnectionStore
from modelpass.testing import FakeAdapter, FakeSessionAdapter, quota_exhausted

GPT = (
    "gpt-6.2-sol", "gpt-6.2-astra", "gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-astra",
    "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
)
NEWEST = [
    ("sol", "gpt-6.2-sol"),
    ("Astra", "gpt-6.2-astra"),
    ("TERRA", "gpt-6.1-terra"),
    ("lUnA", "gpt-6-luna"),
    ("gpt-sol", "gpt-6.2-sol"),
    ("GPT-Luna", "gpt-6-luna"),
    (" sol ", "gpt-6.2-sol"),
]


def codex(tmp_path, slugs=GPT, name="codex", **extra):
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    if slugs is not None:
        (root / "models_cache.json").write_text(
            json.dumps({"models": [{"slug": slug} for slug in slugs]})
        )
    return Connection(name=name, runtime=Runtime.OPENAI_SDK,
                      auth_mode="subscription", config_dir=str(root), **extra)


def bridge_with(tmp_path, adapter=None, runtime=Runtime.OPENAI_SDK, env=None):
    adapter = adapter or FakeAdapter(runtime=runtime)
    return Bridge(
        store=ConnectionStore(tmp_path / "config"),
        env=env or {},
        adapters={runtime: adapter},
    )


# --- the call and the connection ---------------------------------------------------


@pytest.mark.parametrize("name,expected", NEWEST)
def test_a_short_name_on_a_call_is_sent_as_the_newest_id(tmp_path, name, expected):
    adapter = FakeAdapter(runtime=Runtime.OPENAI_SDK)
    bridge = bridge_with(tmp_path, adapter)
    list(bridge.chat(connection=codex(tmp_path), message="x", model=name))
    assert adapter.requests[0].model == expected
    assert adapter.requests[0].plan.model == expected


@pytest.mark.parametrize("name,expected", NEWEST[:4])
def test_a_short_name_on_a_connection_is_sent_as_the_newest_id(tmp_path, name, expected):
    adapter = FakeAdapter(runtime=Runtime.OPENAI_SDK)
    bridge = bridge_with(tmp_path, adapter)
    connection = codex(tmp_path, model=name)
    list(bridge.chat(connection=connection, message="x"))
    assert adapter.requests[0].model == expected
    assert adapter.requests[0].connection.model == expected
    assert connection.model == name


def test_the_receipt_says_which_name_became_which_id(tmp_path):
    receipt = bridge_with(tmp_path).preflight(codex(tmp_path), model="Sol")
    assert receipt.model == "gpt-6.2-sol"
    assert any("'Sol'" in n and "gpt-6.2-sol" in n for n in receipt.notes)


@pytest.mark.parametrize("name,expected", [
    ("gpt-6.1-sol", "gpt-6.1-sol"),
    ("GPT-6.1-Sol", "gpt-6.1-sol"),
    ("Gpt-5.6-LUNA", "gpt-5.6-luna"),
])
def test_a_full_id_keeps_its_version_in_any_case(tmp_path, name, expected):
    # No cache file: a full id never needs the list.
    receipt = bridge_with(tmp_path).preflight(codex(tmp_path, slugs=None), model=name)
    assert receipt.model == expected


@pytest.mark.parametrize("name", ["gpt-6-sol-pro", "mini", "solar", "My-Model", "o4-mini"])
def test_any_other_name_is_sent_as_typed(tmp_path, name):
    receipt = bridge_with(tmp_path).preflight(codex(tmp_path, slugs=None), model=name)
    assert receipt.model == name
    assert not any("newest" in n for n in receipt.notes)


def test_another_runtime_keeps_the_name(tmp_path):
    adapter = FakeAdapter(runtime=Runtime.ANTHROPIC_SDK)
    bridge = bridge_with(tmp_path, adapter, runtime=Runtime.ANTHROPIC_SDK)
    assert bridge.preflight(claude(), model="Sol").model == "Sol"
    assert bridge_with(tmp_path).preflight(codex(tmp_path), model="Opus").model == "Opus"


@pytest.mark.parametrize("slugs", [None, ("gpt-6-astra",)])
def test_a_name_the_list_cannot_answer_is_refused_before_the_run(tmp_path, slugs):
    adapter = FakeAdapter(runtime=Runtime.OPENAI_SDK)
    bridge = bridge_with(tmp_path, adapter)
    with pytest.raises(NoSuchModel, match=r"'sol'.*full model id"):
        bridge.chat(connection=codex(tmp_path, slugs=slugs), message="x", model="sol")
    assert adapter.preflights == [] and adapter.requests == []


def test_input_support_reports_the_refusal_instead_of_raising(tmp_path):
    answer = bridge_with(tmp_path).input_support(codex(tmp_path, slugs=None), model="sol")
    assert not answer.queried and "full model id" in answer.detail


def test_a_failover_reads_the_name_against_its_own_list(tmp_path):
    def script(request):
        return [quota_exhausted(runtime=Runtime.OPENAI_SDK)] if (
            request.connection.name == "first"
        ) else []

    adapter = FakeAdapter(script, runtime=Runtime.OPENAI_SDK)
    bridge = bridge_with(tmp_path, adapter)
    bridge.store.add(codex(tmp_path, slugs=("gpt-6-sol",), name="second"))
    first = codex(
        tmp_path, name="first",
        guards=Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "second")),
    )
    list(bridge.chat(connection=first, message="x", model="sol"))
    assert [r.model for r in adapter.requests] == ["gpt-6.2-sol", "gpt-6-sol"]


# --- sessions and managed conversations --------------------------------------------


def test_a_session_is_opened_on_the_newest_id(tmp_path):
    adapter = FakeSessionAdapter(runtime=Runtime.OPENAI_SDK)
    bridge = bridge_with(tmp_path, adapter)
    with bridge.new_chat(connection=codex(tmp_path), model="Terra") as session:
        assert session.receipt.model == "gpt-6.1-terra"
        assert any("'Terra'" in n for n in session.receipt.notes)


def test_a_managed_conversation_pins_the_id_it_was_created_with(tmp_path):
    bridge = bridge_with(tmp_path)
    bridge.store.add(codex(tmp_path, model="sol"))
    conversations = bridge.conversations(tmp_path / "chats", application="app")
    assert conversations.create(connection="codex").info.model == "gpt-6.2-sol"
    named = conversations.create(connection="codex", model="LUNA")
    assert named.info.model == "gpt-6-luna"


# --- the API runtime ----------------------------------------------------------------


class _Client:
    """A fake ``openai`` client that only knows how to list models."""

    def __init__(self, model_ids=GPT):
        self.rows = [Obj(id=i) for i in model_ids]
        self.calls = 0
        self.models = self

    def with_options(self, **_kwargs):
        return self

    def list(self, **_kwargs):
        self.calls += 1
        return Obj(data=self.rows, has_next_page=lambda: False)

    def close(self):
        pass


class _Adapter(FakeAdapter):
    def __init__(self, client):
        super().__init__(runtime=Runtime.OPENAI_API)
        self.lister = OpenAIAPIAdapter(client_factory=lambda **_: client, env={"KEY": "k"})

    def list_models(self, connection, *, timeout):
        return self.lister.list_models(connection, timeout=timeout)


def api(**extra):
    return Connection(name="api", runtime=Runtime.OPENAI_API, auth_mode="api_key",
                      credential_ref=CredentialRef.parse("env:KEY"), **extra)


def api_bridge(tmp_path, client):
    adapter = _Adapter(client)
    return adapter, bridge_with(tmp_path, adapter, Runtime.OPENAI_API, env={"KEY": "k"})


def test_an_api_short_name_reads_the_model_list_once(tmp_path):
    client = _Client()
    adapter, bridge = api_bridge(tmp_path, client)
    for name in ("sol", "astra"):
        list(bridge.chat(connection=api(), message="x", model=name))
    assert [r.model for r in adapter.requests] == ["gpt-6.2-sol", "gpt-6.2-astra"]
    assert client.calls == 1


def test_an_api_full_id_asks_for_no_list(tmp_path):
    client = _Client()
    _, bridge = api_bridge(tmp_path, client)
    list(bridge.chat(connection=api(), message="x", model="gpt-6.1-sol"))
    assert client.calls == 0


def test_a_disabled_api_connection_is_inspected_without_a_request(tmp_path):
    client = _Client()
    _, bridge = api_bridge(tmp_path, client)
    assert bridge.preflight(api(enabled=False), model="sol").model == "sol"
    assert client.calls == 0


def test_achat_reads_the_api_list_off_the_event_loop(tmp_path):
    seen = []

    class Watching(_Client):
        def list(self, **kwargs):
            try:
                asyncio.get_running_loop()
                seen.append("on the loop")
            except RuntimeError:
                seen.append("off the loop")
            return super().list(**kwargs)

    adapter, bridge = api_bridge(tmp_path, Watching())

    async def drive():
        events = bridge.achat(connection=api(), message="x", model="sol")
        assert seen == []
        return [event async for event in events]

    events = asyncio.run(drive())
    assert seen == ["off the loop"]
    assert adapter.requests[0].model == "gpt-6.2-sol"
    assert events[0].receipt.model == "gpt-6.2-sol"


# --- the Claude names ----------------------------------------------------------------

CLAUDE = (
    "claude-fable-5-2", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
    "claude-fable-5-1", "claude-opus-5", "claude-sonnet-5",
    "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001",
)


def claude(**extra):
    return Connection(name="c", runtime=Runtime.ANTHROPIC_SDK,
                      auth_mode="subscription", **extra)


@pytest.mark.parametrize("name,sent", [
    ("Haiku", "haiku"),
    ("SONNET", "sonnet"),
    ("oPuS", "opus"),
    (" Fable ", "fable"),
    ("claude-opus", "opus"),
    ("Claude-Haiku", "haiku"),
    ("Claude-Opus-5-5", "claude-opus-5-5"),
    ("CLAUDE-HAIKU-4-5-20251001", "claude-haiku-4-5-20251001"),
    ("opus", "opus"),
    ("Claude-Next", "Claude-Next"),
])
def test_the_subscription_gets_the_lower_cased_alias(tmp_path, name, sent):
    adapter = FakeAdapter(runtime=Runtime.ANTHROPIC_SDK)
    bridge = bridge_with(tmp_path, adapter, runtime=Runtime.ANTHROPIC_SDK)
    list(bridge.chat(connection=claude(), message="x", model=name))
    assert adapter.requests[0].model == sent
    notes = bridge.preflight(claude(), model=name).notes
    assert not any("newest" in note for note in notes)
    assert any("was read as" in note for note in notes) is (name != sent)


def test_a_claude_name_on_a_subscription_connection_is_lower_cased(tmp_path):
    adapter = FakeAdapter(runtime=Runtime.ANTHROPIC_SDK)
    bridge = bridge_with(tmp_path, adapter, runtime=Runtime.ANTHROPIC_SDK)
    list(bridge.chat(connection=claude(model="Opus"), message="x"))
    assert adapter.requests[0].connection.model == "opus"


class _AnthropicAdapter(FakeAdapter):
    def __init__(self, client):
        super().__init__(runtime=Runtime.ANTHROPIC_API)
        self.lister = AnthropicAPIAdapter(
            client_factory=lambda **_: client, env={"KEY": "k"}
        )

    def list_models(self, connection, *, timeout):
        return self.lister.list_models(connection, timeout=timeout)


def anthropic_api(**extra):
    return Connection(name="a", runtime=Runtime.ANTHROPIC_API, auth_mode="api_key",
                      credential_ref=CredentialRef.parse("env:KEY"), **extra)


@pytest.mark.parametrize("name,expected", [
    ("Haiku", "claude-haiku-5-5"),
    ("sonnet", "claude-sonnet-5-5"),
    ("OPUS", "claude-opus-5-5"),
    ("Fable", "claude-fable-5-2"),
    ("claude-opus", "claude-opus-5-5"),
    ("Claude-Opus-5", "claude-opus-5"),
])
def test_a_claude_name_on_the_api_is_sent_as_the_newest_id(tmp_path, name, expected):
    adapter = _AnthropicAdapter(_Client(CLAUDE))
    bridge = bridge_with(tmp_path, adapter, Runtime.ANTHROPIC_API, env={"KEY": "k"})
    list(bridge.chat(connection=anthropic_api(), message="x", model=name))
    assert adapter.requests[0].model == expected


def test_achat_resolves_a_claude_api_name(tmp_path):
    adapter = _AnthropicAdapter(_Client(CLAUDE))
    bridge = bridge_with(tmp_path, adapter, Runtime.ANTHROPIC_API, env={"KEY": "k"})

    async def drive():
        return [e async for e in bridge.achat(
            connection=anthropic_api(model="Opus"), message="x"
        )]

    events = asyncio.run(drive())
    assert events[0].receipt.model == "claude-opus-5-5"
    assert any("'Opus'" in n and "newest" in n for n in events[0].receipt.notes)


def test_a_claude_name_the_api_list_lacks_is_refused(tmp_path):
    adapter = _AnthropicAdapter(_Client(("claude-opus-5-5",)))
    bridge = bridge_with(tmp_path, adapter, Runtime.ANTHROPIC_API, env={"KEY": "k"})
    with pytest.raises(NoSuchModel, match=r"'haiku'.*full model id"):
        bridge.chat(connection=anthropic_api(), message="x", model="haiku")
