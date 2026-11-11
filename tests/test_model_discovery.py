"""Metadata discovery uses real API adapters with fake SDK clients; no network."""
import json
from dataclasses import replace
from types import SimpleNamespace as Obj

import pytest

from modelpass import Bridge, Connection, ConnectionDisabled, CredentialRef, Runtime
from modelpass.adapters.anthropic_api import AnthropicAPIAdapter
from modelpass.adapters.google_api import GoogleAPIAdapter
from modelpass.adapters.openai_api import OpenAIAPIAdapter
from modelpass.adapters.openai_compatible import OpenAICompatibleAdapter
from modelpass.store import ConnectionStore


class Client:
    def __init__(self, runtime, pages):
        self.runtime, self.pages = runtime, list(pages)
        self.calls, self.options = [], []
        self.closed = False
        self.models = self

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def list(self, **kwargs):
        self.calls.append(kwargs)
        rows, more = self.pages.pop(0)
        if isinstance(rows, Exception):
            raise rows
        if self.runtime == Runtime.GOOGLE_API:
            return Obj(page=rows, config={"page_token": more})
        return Obj(data=rows, has_next_page=lambda: bool(more))

    def close(self):
        self.closed = True


@pytest.mark.parametrize("runtime,adapter_type", [
    (Runtime.OPENAI_API, OpenAIAPIAdapter),
    (Runtime.ANTHROPIC_API, AnthropicAPIAdapter),
    (Runtime.GOOGLE_API, GoogleAPIAdapter),
    (Runtime.OPENAI_COMPATIBLE, OpenAICompatibleAdapter),
])
def test_explicit_refresh_uses_named_credentials_and_endpoint_and_cache_is_offline(
    tmp_path, runtime, adapter_type,
):
    factory_calls = []
    client = Client(runtime, [([Obj(id="one", display_name="One")], None)])
    def factory(**kwargs):
        factory_calls.append(kwargs)
        return client
    adapter = adapter_type(client_factory=factory, env={"TEST_KEY": "named-test-key"})
    connection = Connection(name="named", runtime=runtime, auth_mode="api_key",
        credential_ref=CredentialRef.parse("env:TEST_KEY"), base_url="https://example.test/v1")
    bridge = Bridge(store=ConnectionStore(tmp_path), adapters={runtime: adapter}, env={})
    assert not bridge.models(connection).choices
    assert not factory_calls
    catalogue = bridge.models(connection, refresh=True, timeout=5)
    assert catalogue.problem is None and catalogue.exact
    assert [c.id for c in catalogue.choices] == ["one"]
    assert factory_calls == [{"api_key": "named-test-key", "base_url": "https://example.test/v1"}]
    assert client.closed and len(client.calls) == 1
    assert bridge.models(connection) == catalogue and len(client.calls) == 1
    assert not bridge.models(replace(connection, base_url="https://other.test/v1")).choices
    assert not (tmp_path / "runs.jsonl").exists()


@pytest.mark.parametrize("runtime,adapter_type", [(Runtime.ANTHROPIC_API, AnthropicAPIAdapter),
                                                 (Runtime.GOOGLE_API, GoogleAPIAdapter)])
def test_model_pages_follow_vendor_cursor_with_bounded_request_timeouts(
    tmp_path, runtime, adapter_type,
):
    client = Client(runtime, [([Obj(id="first")], "next"), ([Obj(name="models/second")], None)])
    adapter = adapter_type(client_factory=lambda **kw: client, env={"KEY": "test-only"})
    connection = Connection(name="test", runtime=runtime, auth_mode="api_key",
                            credential_ref=CredentialRef.parse("env:KEY"))
    bridge = Bridge(store=ConnectionStore(tmp_path), adapters={runtime: adapter})
    result = bridge.models(connection, refresh=True, timeout=7)
    assert result.problem is None
    assert len(result.choices) == 2 and len(client.calls) == 2
    if runtime == Runtime.GOOGLE_API:
        assert client.calls[1]["config"]["page_token"] == "next"
        assert 0 < client.calls[0]["config"]["http_options"]["timeout"] <= 7000
        assert client.calls[0]["config"]["http_options"]["retry_options"]["attempts"] == 1
    else:
        assert client.calls[1]["after_id"] == "first"
        assert 0 < client.calls[0]["timeout"] <= 7
        assert client.options == [{"max_retries": 0}]


def test_vendor_failure_does_not_expose_secrets_or_erase_previous_good_cache(tmp_path):
    client = Client(Runtime.OPENAI_API, [([Obj(id="one")], None),
        (RuntimeError("credential-with-secret-content"), None)])
    adapter = OpenAIAPIAdapter(client_factory=lambda **kw: client, env={"KEY": "test-only"})
    connection = Connection(name="test", runtime=Runtime.OPENAI_API, auth_mode="api_key",
                            credential_ref=CredentialRef.parse("env:KEY"))
    bridge = Bridge(store=ConnectionStore(tmp_path), adapters={Runtime.OPENAI_API: adapter})
    first = bridge.models(connection, refresh=True)
    failed = bridge.models(connection, refresh=True)
    assert failed.problem and "credential-with-secret-content" not in str(failed)
    assert bridge.models(connection) == first


def test_subscription_catalogue_uses_the_selected_account_directory_without_loading_an_adapter(
    tmp_path,
):
    root = tmp_path / "account"
    root.mkdir()
    (root / "models_cache.json").write_text(json.dumps({"models": [{"slug": "account-model"}]}))
    bridge = Bridge(store=ConnectionStore(tmp_path / "config"), env={})
    connection = Connection(name="account", runtime=Runtime.OPENAI_SDK,
                            auth_mode="subscription", config_dir=str(root))
    result = bridge.models(connection)
    assert [m.id for m in result.choices] == ["account-model"]
    assert not bridge._adapters


def test_disabled_connections_cannot_refresh_and_timeout_is_validated_before_any_sdk_request(
    tmp_path,
):
    bridge = Bridge(store=ConnectionStore(tmp_path), env={})
    connection = Connection(name="test", runtime=Runtime.OPENAI_API, auth_mode="api_key",
                            credential_ref=CredentialRef.parse("env:KEY"), enabled=False)
    assert not bridge.models(connection).choices
    with pytest.raises(ConnectionDisabled):
        bridge.models(connection, refresh=True)
    for timeout in (0, -1, float("inf"), float("nan"), True, 121):
        with pytest.raises(ValueError):
            bridge.models(replace(connection, enabled=True), refresh=True, timeout=timeout)
    assert not bridge._adapters
