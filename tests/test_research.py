"""Hosted job lifecycle, credential binding and no-retry submission semantics.

Fake clients exercise OpenAI Responses and Gemini Interactions independently of
vendor packages, accounts or networks. Both backends remain live-untested.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from modelpass import (
    AuthMode,
    Bridge,
    Connection,
    ConnectionStore,
    CredentialRef,
    InvalidConnection,
    ResearchJob,
    Runtime,
    VendorRunFailed,
    WebOptions,
)
from modelpass.adapters.google_api import GoogleAPIAdapter
from modelpass.adapters.openai_api import OpenAIAPIAdapter


class Resource:
    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = []
        self.error = None

    def create(self, **params):
        self.calls.append(("create", params))
        if self.error:
            raise self.error
        return {"id": "job-1", "status": "in_progress"}

    def retrieve(self, identifier, **params):
        self.calls.append(("get", identifier, params))
        if self.runtime is Runtime.OPENAI_API:
            return {
                "id": identifier,
                "status": "completed",
                "output": [
                    {
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {"type": "search"},
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "report",
                                "annotations": [
                                    {"type": "url_citation", "url": "https://example.org"}
                                ],
                            }
                        ],
                    },
                ],
            }
        return {
            "id": identifier,
            "status": "completed",
            "outputs": [
                # The shapes google-genai 1.73.1 dumps (read 2026-11-18): the
                # queries sit under ``arguments``, suggestions under each
                # ``result`` entry, and a text block with no citations carries
                # an explicit ``"annotations": None``.
                {"type": "google_search_call", "id": "s1", "arguments": {"queries": ["query"]}},
                {
                    "type": "google_search_result",
                    "call_id": "s1",
                    "result": [{"search_suggestions": "<div>Search</div>"}],
                },
                {"type": "text", "text": "A ", "annotations": None},
                {
                    "type": "text",
                    "text": "report",
                    "annotations": [
                        {"type": "url_citation", "url": "https://example.org"},
                    ],
                },
            ],
        }

    get = retrieve

    def cancel(self, identifier, **params):
        self.calls.append(("cancel", identifier, params))
        return {"id": identifier, "status": "cancelled"}


class GoogleClient:
    def __init__(self, resource):
        self.interactions = resource
        self._nextgen_client = self
        self.retries = None

    def with_options(self, *, max_retries):
        self.retries = max_retries
        return self

    def close(self):
        pass


def setup_bridge(tmp_path, runtime):
    resource = Resource(runtime)
    conn = Connection(
        name="research",
        runtime=runtime,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:TEST_KEY"),
        model="test-model",
    )
    client = (
        GoogleClient(resource)
        if runtime is Runtime.GOOGLE_API
        else SimpleNamespace(responses=resource, close=lambda: None)
    )
    adapter_cls = GoogleAPIAdapter if runtime is Runtime.GOOGLE_API else OpenAIAPIAdapter
    adapter = adapter_cls(client_factory=lambda **kwargs: client, env={"TEST_KEY": "fake-key"})
    adapter.is_available = lambda: True
    store = ConnectionStore(tmp_path)
    store.add(conn)
    bridge = Bridge(store=store, adapters={runtime: adapter}, env={"TEST_KEY": "fake-key"})
    return bridge, resource, client


@pytest.mark.parametrize("runtime", [Runtime.OPENAI_API, Runtime.GOOGLE_API])
def test_submit_restore_poll_and_cancel_original_job(tmp_path, runtime):
    bridge, resource, client = setup_bridge(tmp_path, runtime)
    selector = {"agent": "deep-research-preview-04-2026"} if runtime is Runtime.GOOGLE_API else {}
    job = bridge.start_research(
        "research", "brief", **selector, web=WebOptions(require_search=True, require_citations=True)
    )
    restored = ResearchJob.from_dict(job.to_dict())
    assert restored == job
    assert "fake-key" not in str(job.to_dict())
    snapshot = bridge.get_research(restored)
    assert snapshot.status == "completed"
    assert snapshot.result.text == ("A report" if runtime is Runtime.GOOGLE_API else "report")
    if runtime is Runtime.GOOGLE_API:
        assert snapshot.result.presentation == {"search_suggestions": ["<div>Search</div>"]}
    assert snapshot.result.citations[0].url == "https://example.org"
    assert bridge.cancel_research(restored).status == "cancelled"
    assert [call[0] for call in resource.calls] == ["create", "get", "cancel"]
    assert resource.calls[0][1]["background"] is True
    if runtime is Runtime.GOOGLE_API:
        assert client.retries == 0
        assert resource.calls[0][1]["tools"] == [
            {"type": "google_search"},
            {"type": "url_context"},
        ]


def test_changed_endpoint_refuses_retrieval_before_network(tmp_path):
    bridge, resource, _ = setup_bridge(tmp_path, Runtime.OPENAI_API)
    job = bridge.start_research("research", "brief")
    original = bridge.store.get("research")
    bridge.store.add(replace(original, base_url="https://other.example"), overwrite=True)
    with pytest.raises(InvalidConnection, match="identity changed"):
        bridge.get_research(job)
    assert len(resource.calls) == 1


def test_unknown_submission_is_never_retried(tmp_path):
    bridge, resource, _ = setup_bridge(tmp_path, Runtime.OPENAI_API)
    resource.error = TimeoutError()
    with pytest.raises(VendorRunFailed, match="outcome may be unknown"):
        bridge.start_research("research", "brief")
    assert len(resource.calls) == 1


def test_bad_timeout_cannot_submit(tmp_path):
    bridge, resource, _ = setup_bridge(tmp_path, Runtime.OPENAI_API)
    with pytest.raises(ValueError):
        bridge.start_research("research", "brief", timeout=float("nan"))
    assert not resource.calls


def test_a_completed_job_without_report_is_a_failed_snapshot(tmp_path):
    bridge, resource, _ = setup_bridge(tmp_path, Runtime.OPENAI_API)
    job = bridge.start_research("research", "brief")
    resource.retrieve = lambda identifier, **params: {"id": identifier, "status": "completed"}
    snapshot = bridge.get_research(job)
    assert snapshot.status == "failed"
    assert snapshot.result is None
    assert "no text report" in snapshot.reason


def test_a_client_with_no_job_interface_says_nothing_was_submitted(tmp_path):
    """A local failure is not an unknown remote outcome (2026-11-18)."""
    bridge, resource, client = setup_bridge(tmp_path, Runtime.GOOGLE_API)
    del client._nextgen_client
    with pytest.raises(VendorRunFailed, match="nothing was submitted"):
        bridge.start_research("research", "brief", agent="deep-research-preview-04-2026")
    assert not resource.calls


def test_installed_sdk_job_output_shapes_are_the_ones_the_snapshot_reads():
    """Ties the fixture above to the installed SDK instead of to the code."""
    types = pytest.importorskip("google.genai._interactions.types")
    assert "arguments" in types.GoogleSearchCallContent.model_fields
    assert "queries" not in types.GoogleSearchCallContent.model_fields
    assert "result" in types.GoogleSearchResultContent.model_fields
    assert "search_suggestions" in types.GoogleSearchResult.model_fields
    assert types.TextContent.model_fields["annotations"].default is None
