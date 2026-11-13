"""modelpass never retries -- and neither may the vendor client underneath it.

The README says it in a heading: *modelpass never retries for you*. The verdict
(``retryable``, ``retry_after``) goes to the caller, who knows what a second run
costs. A vendor SDK that quietly retries a 429 or a 5xx twice before reporting
it breaks that from below: the quota failover waits out two backoffs before it
is even told, a ``retry = "never"`` connection retries anyway, and a request the
vendor did process before the connection dropped is sent -- and billed -- again.

The anthropic and openai clients default to ``max_retries=2`` (``DEFAULT_MAX_RETRIES``
in each SDK's ``_constants.py``; anthropic 0.97.0 and openai 2.32.0, read
2026-11-13). google-genai 1.73.1 does not retry unless ``retry_options`` is
given (``_api_client.retry_args(None)`` is a one-attempt stop), which is pinned
here as well so a default that moves in a later SDK fails a test rather than
doubling a bill.

Every test builds the **real** client through the adapter's default factory and
counts the requests that reach an ``httpx.MockTransport`` answering with a
failure the SDK considers retryable. No network, nothing spent.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")

FLAVOURS = ("sync", "async")
RETRYABLE = (429, 500, 503)


def _client(module: str, flavour: str, base_url: str | None) -> Any:
    adapter = importlib.import_module(f"modelpass.adapters.{module}")
    factory = adapter._default_client if flavour == "sync" else adapter._default_async_client
    return factory(api_key="a-key", base_url=base_url)


def _requests_sent(client: Any, flavour: str, status: int) -> int:
    """How many requests one failing ``models.list`` puts on the wire."""
    sent: list[Any] = []

    def answer(request: Any) -> Any:
        sent.append(request)
        # A one-millisecond wait: a client that does retry does it at once, so
        # the count is about the policy and the test is not about a sleep.
        return httpx.Response(
            status,
            headers={"retry-after-ms": "1"},
            json={"error": {"message": "scripted", "type": "scripted"}},
        )

    transport = httpx.MockTransport(answer)
    if flavour == "sync":
        client._client = httpx.Client(transport=transport)
        with pytest.raises(Exception) as caught:
            client.models.list()
    else:
        client._client = httpx.AsyncClient(transport=transport)

        async def go() -> None:
            await client.models.list()

        with pytest.raises(Exception) as caught:
            asyncio.run(go())
    assert getattr(caught.value, "status_code", None) == status
    return len(sent)


@pytest.mark.parametrize("status", RETRYABLE)
@pytest.mark.parametrize("flavour", FLAVOURS)
@pytest.mark.parametrize(
    ("module", "package", "base_url"),
    [
        ("anthropic_api", "anthropic", None),
        ("openai_api", "openai", None),
        ("openai_compatible", "openai", "https://gateway.example/v1"),
    ],
)
def test_a_retryable_failure_is_sent_once(module, package, base_url, flavour, status):
    pytest.importorskip(package)
    client = _client(module, flavour, base_url)
    assert client.max_retries == 0
    assert _requests_sent(client, flavour, status) == 1


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_the_google_client_is_built_to_attempt_once(flavour):
    pytest.importorskip("google.genai")
    api = _client("google_api", flavour, None)._api_client
    assert api._http_options.retry_options is not None
    assert api._http_options.retry_options.attempts == 1
    assert api._retry.stop.max_attempt_number == 1
    assert api._async_retry.stop.max_attempt_number == 1


@pytest.mark.parametrize("status", RETRYABLE)
def test_the_google_client_sends_a_retryable_failure_once(status):
    pytest.importorskip("google.genai")
    client = _client("google_api", "sync", None)
    sent: list[Any] = []

    def answer(request: Any) -> Any:
        sent.append(request)
        return httpx.Response(status, json={"error": {"code": status, "message": "scripted"}})

    client._api_client._httpx_client = httpx.Client(transport=httpx.MockTransport(answer))
    with pytest.raises(Exception) as caught:
        list(client.models.list())
    assert getattr(caught.value, "code", None) == status
    assert len(sent) == 1
