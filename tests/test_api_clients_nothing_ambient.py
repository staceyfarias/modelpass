"""The vendor clients the four API adapters build, held to D2 on the wire.

``test_no_ambient_credentials.py`` holds the *key*: the connection's credential is
handed to the client rather than left for the SDK to find. This file holds
everything else a vendor SDK reads from the process environment when its
constructor is not told otherwise, because each of those decides something about
a run that the connection never stated:

* where the key is sent -- ``ANTHROPIC_BASE_URL``, ``OPENAI_BASE_URL``,
  ``GOOGLE_GEMINI_BASE_URL``;
* what else rides along -- ``ANTHROPIC_AUTH_TOKEN`` as a second credential,
  ``OPENAI_ORG_ID`` / ``OPENAI_PROJECT_ID`` as headers an ``openai-compatible``
  host was never meant to see;
* which service answers at all -- ``GOOGLE_GENAI_USE_VERTEXAI``, and
  ``GOOGLE_GENAI_CLIENT_MODE``, which swaps the client for a replay harness.

Every test here builds the **real** SDK client through the adapter's own default
factory, with all of those variables set, and reads the request the client would
actually send off an ``httpx.MockTransport`` -- no network, no vendor, nothing
spent. A fake client could not hold this: the defect is in what the real
constructor does with an argument it was not given.

SDK surfaces read 2026-11-13: anthropic 0.97.0 (``Anthropic.__init__``,
``auth_headers``, ``copy``), openai 2.32.0 (``OpenAI.__init__``,
``default_headers``, ``copy``), google-genai 1.73.1 (``Client.__init__``,
``BaseApiClient.__init__``, ``_base_url.get_base_url``, ``retry_args``).
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")

#: Everything ambient a vendor SDK is known to act on, set all at once: a machine
#: that has one of these usually has several.
AMBIENT = {
    "ANTHROPIC_API_KEY": "sk-ant-ambient-decoy",
    "ANTHROPIC_AUTH_TOKEN": "ambient-bearer-token",
    "ANTHROPIC_BASE_URL": "https://ambient-proxy.example",
    "OPENAI_API_KEY": "sk-ambient-decoy",
    "OPENAI_BASE_URL": "https://ambient-proxy.example/v1",
    "OPENAI_ORG_ID": "org-ambient",
    "OPENAI_PROJECT_ID": "proj-ambient",
    "OPENAI_WEBHOOK_SECRET": "whsec-ambient",
    "GEMINI_API_KEY": "ambient-gemini-decoy",
    "GOOGLE_API_KEY": "ambient-google-decoy",
    "GOOGLE_GEMINI_BASE_URL": "https://ambient-proxy.example",
    "GOOGLE_VERTEX_BASE_URL": "https://ambient-vertex.example",
    "GOOGLE_GENAI_USE_VERTEXAI": "true",
    "GOOGLE_CLOUD_PROJECT": "ambient-project",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "GOOGLE_GENAI_CLIENT_MODE": "replay",
}

NAMED = "the-key-the-connection-actually-names"
GATEWAY = "https://gateway.example/v1"


@pytest.fixture
def ambient(monkeypatch):
    for name, value in AMBIENT.items():
        monkeypatch.setenv(name, value)


def _factory(module: str, flavour: str) -> Any:
    adapter = importlib.import_module(f"modelpass.adapters.{module}")
    return adapter._default_client if flavour == "sync" else adapter._default_async_client


class _Wire:
    """Records what a real Stainless client (anthropic, openai) puts on the wire."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[Any] = []

    def _answer(self, request: Any) -> Any:
        self.requests.append(request)
        if self.status != 200:
            # A wait of one millisecond, so a client that does retry does it at
            # once and the count below is about the policy, not about a sleep.
            return httpx.Response(
                self.status,
                headers={"retry-after-ms": "1"},
                json={"error": {"message": "scripted", "type": "scripted"}},
            )
        return httpx.Response(200, json={"object": "list", "data": [], "has_more": False})

    def list_models(self, client: Any, flavour: str) -> Any:
        """One token-free ``models.list`` through ``client``'s own request path."""
        if flavour == "sync":
            client._client = httpx.Client(transport=httpx.MockTransport(self._answer))
            client.models.list()
        else:
            client._client = httpx.AsyncClient(transport=httpx.MockTransport(self._answer))

            async def go() -> None:
                await client.models.list()

            asyncio.run(go())
        return self.requests[0]


FLAVOURS = ("sync", "async")


# --- anthropic-api -------------------------------------------------------------


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_anthropic_client_sends_the_key_to_the_vendor_and_nothing_else(ambient, flavour):
    anthropic = pytest.importorskip("anthropic")
    # The failure being prevented: the bare constructor takes all three.
    bare = anthropic.Anthropic(api_key=NAMED)
    assert str(bare.base_url).startswith("https://ambient-proxy.example")
    assert bare.auth_headers.get("Authorization") == "Bearer ambient-bearer-token"

    client = _factory("anthropic_api", flavour)(api_key=NAMED, base_url=None)
    sent = _Wire().list_models(client, flavour)

    assert sent.url.host == "api.anthropic.com"
    assert sent.headers["x-api-key"] == NAMED
    assert "authorization" not in sent.headers
    assert "ambient" not in str(sent.url) + str(dict(sent.headers))


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_anthropic_client_stays_clean_through_with_options(ambient, flavour):
    """``with_options`` rebuilds the client, and the rebuild re-reads the environment.

    ``api_catalogue`` (``modelpass/models.py``) calls it on every model listing,
    so a fix that only cleaned the first construction would leak on the second.
    """
    pytest.importorskip("anthropic")
    client = _factory("anthropic_api", flavour)(api_key=NAMED, base_url=None)
    sent = _Wire().list_models(client.with_options(timeout=5), flavour)

    assert sent.url.host == "api.anthropic.com"
    assert "authorization" not in sent.headers


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_anthropic_client_keeps_the_connections_own_base_url(ambient, flavour):
    pytest.importorskip("anthropic")
    client = _factory("anthropic_api", flavour)(api_key=NAMED, base_url=GATEWAY)
    sent = _Wire().list_models(client, flavour)
    assert sent.url.host == "gateway.example"
    assert "authorization" not in sent.headers


# --- openai-api and openai-compatible --------------------------------------------


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_openai_client_sends_the_key_to_the_vendor_and_nothing_else(ambient, flavour):
    openai = pytest.importorskip("openai")
    bare = openai.OpenAI(api_key=NAMED)
    assert str(bare.base_url).startswith("https://ambient-proxy.example")
    assert bare.organization == "org-ambient"
    assert bare.project == "proj-ambient"

    client = _factory("openai_api", flavour)(api_key=NAMED, base_url=None)
    sent = _Wire().list_models(client, flavour)

    assert sent.url.host == "api.openai.com"
    assert sent.url.path == "/v1/models"
    assert sent.headers["authorization"] == f"Bearer {NAMED}"
    assert "openai-organization" not in sent.headers
    assert "openai-project" not in sent.headers
    assert "ambient" not in str(sent.url) + str(dict(sent.headers))
    assert client.webhook_secret is None


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_openai_client_stays_clean_through_with_options(ambient, flavour):
    pytest.importorskip("openai")
    client = _factory("openai_api", flavour)(api_key=NAMED, base_url=None)
    sent = _Wire().list_models(client.with_options(timeout=5), flavour)

    assert sent.url.host == "api.openai.com"
    assert "openai-organization" not in sent.headers
    assert "openai-project" not in sent.headers


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_openai_compatible_host_is_never_told_the_openai_organization(ambient, flavour):
    """An Ollama box or a gateway is not OpenAI, and gets no OpenAI account ids."""
    pytest.importorskip("openai")
    client = _factory("openai_compatible", flavour)(api_key=NAMED, base_url=GATEWAY)
    sent = _Wire().list_models(client, flavour)

    assert sent.url.host == "gateway.example"
    assert sent.headers["authorization"] == f"Bearer {NAMED}"
    assert "openai-organization" not in sent.headers
    assert "openai-project" not in sent.headers
    assert "ambient" not in str(sent.url) + str(dict(sent.headers))

    again = _Wire().list_models(client.with_options(timeout=5), flavour)
    assert "openai-organization" not in again.headers
    assert "openai-project" not in again.headers


# --- google-api ------------------------------------------------------------------


def _google_api_client(client: Any) -> Any:
    """The ``BaseApiClient`` behind a ``Client`` or its ``.aio`` -- one object, both faces."""
    return client._api_client


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_google_client_is_the_gemini_api_with_the_connections_key(ambient, flavour):
    genai = pytest.importorskip("google.genai")
    # The failure being prevented: Vertex, somebody's proxy, and a replay harness
    # in place of the HTTP client.
    bare = genai.Client(api_key=NAMED)
    assert bare.vertexai is True
    assert type(bare._api_client).__name__ == "ReplayApiClient"

    client = _factory("google_api", flavour)(api_key=NAMED, base_url=None)
    api = _google_api_client(client)

    assert type(api).__name__ == "BaseApiClient"
    assert api.vertexai is False
    assert api._http_options.base_url == "https://generativelanguage.googleapis.com/"
    assert api._http_options.api_version == "v1beta"
    assert api._http_options.headers["x-goog-api-key"] == NAMED
    assert "ambient" not in str(api._http_options.base_url) + str(api._http_options.headers)


def test_google_client_puts_the_key_on_the_gemini_wire(ambient):
    pytest.importorskip("google.genai")
    client = _factory("google_api", "sync")(api_key=NAMED, base_url=None)
    seen: list[Any] = []

    def answer(request: Any) -> Any:
        seen.append(request)
        return httpx.Response(200, json={"models": []})

    client._api_client._httpx_client = httpx.Client(transport=httpx.MockTransport(answer))
    list(client.models.list())

    assert seen[0].url.host == "generativelanguage.googleapis.com"
    assert seen[0].headers["x-goog-api-key"] == NAMED
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_google_client_keeps_the_connections_own_base_url(ambient, flavour):
    pytest.importorskip("google.genai")
    client = _factory("google_api", flavour)(api_key=NAMED, base_url="https://gateway.example/")
    api = _google_api_client(client)
    assert api.vertexai is False
    assert api._http_options.base_url == "https://gateway.example/"
