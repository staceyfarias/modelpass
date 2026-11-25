"""``sol``, ``astra``, ``terra`` and ``luna`` as model names (2026-11-25).

A caller or a connection may name one of these OpenAI series where a model id
goes, in any case. modelpass sends the newest id of that series from the
connection's own model list, and the receipt says which name became which id.
Everything else is sent as typed.

All offline: the Codex list is a cache file in a temporary directory and the API
list comes from a fake client.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as Obj

import pytest

from modelpass import Bridge, Connection, CredentialRef, NoSuchModel, Runtime
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
    connection = Connection(name="c", runtime=Runtime.ANTHROPIC_SDK, auth_mode="subscription")
    assert bridge.preflight(connection, model="sol").model == "sol"


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
