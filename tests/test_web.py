"""Native web opt-in, refusals, evidence and wire mappings without a network.

These tests hold the distinction between enabling a tool and observing its use,
including async calls, failed searches and a configured unsupported failover.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from modelpass import (
    CapabilityNotSupported,
    Connection,
    CredentialRef,
    Guards,
    QuotaAction,
    QuotaPolicy,
    Runtime,
    VendorRunFailed,
    WebCitation,
    WebEvidenceEvent,
    WebOptions,
    WebSource,
)
from modelpass.adapters.anthropic_api import AnthropicAPIAdapter
from modelpass.adapters.base import RunRequest
from modelpass.adapters.google_api import GoogleAPIAdapter
from modelpass.adapters.openai import web_config_overrides
from modelpass.adapters.openai_api import OpenAIAPIAdapter
from modelpass.preflight import plan_launch
from modelpass.testing import fake_bridge, tool_exchange
from modelpass.types import AuthMode, Message, TextDeltaEvent
from modelpass.web import anthropic_evidence, google_evidence, openai_evidence, validate_web


def connection(runtime=Runtime.ANTHROPIC_SDK, **kwargs):
    return Connection(
        name="web-test",
        runtime=runtime,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        **kwargs,
    )


def test_missing_search_is_an_error_even_when_model_answers(tmp_path):
    bridge, _, _ = fake_bridge(
        connections=[connection()],
        script=[TextDeltaEvent("an uncited guess")],
        home=tmp_path,
    )
    with pytest.raises(VendorRunFailed, match="no completed native search"):
        bridge.ask("web-test", "look it up", web=WebOptions(require_search=True))


@pytest.mark.parametrize("is_error", [False, True])
def test_subscription_requires_a_completed_result(tmp_path, is_error):
    bridge, _, _ = fake_bridge(
        connections=[connection()],
        script=[
            *tool_exchange("WebSearch", {"query": "Python"}, is_error=is_error),
            TextDeltaEvent("answer"),
        ],
        home=tmp_path,
    )
    if is_error:
        with pytest.raises(VendorRunFailed):
            bridge.ask("web-test", "look it up", web=WebOptions(require_search=True))
    else:
        answer = bridge.ask("web-test", "look it up", web=WebOptions(require_search=True))
        assert answer.text == "answer"
        assert answer.receipt.web["search"] == "live"
        assert answer.events[0].type == "receipt"
        assert answer.events[-1].type == "terminal"


def test_async_answer_has_the_same_evidence_and_deduplicates(tmp_path):
    evidence = WebEvidenceEvent(
        (WebSource("https://example.org", "Example"),),
        (WebCitation("https://example.org", "Example"),),
        searches=1,
    )
    bridge, _, _ = fake_bridge(
        connections=[connection()],
        script=[evidence, evidence, TextDeltaEvent("answer")],
        home=tmp_path,
    )
    answer = asyncio.run(bridge.aask("web-test", "question", web={"require_search": True}))
    assert len(answer.sources) == len(answer.citations) == 1
    assert answer.events[1].to_dict()["sources"][0]["url"] == "https://example.org"


@pytest.mark.parametrize(
    "runtime,options,web",
    [
        (Runtime.ANTHROPIC_SDK, {}, WebOptions(search="cached")),
        (Runtime.ANTHROPIC_SDK, {}, WebOptions(search_allowed_domains=("example.org",))),
        (Runtime.ANTHROPIC_SDK, {}, WebOptions(require_citations=True)),
        (Runtime.OPENAI_SDK, {"transport": "exec"}, WebOptions()),
        (Runtime.OPENAI_SDK, {}, WebOptions(require_citations=True)),
        (Runtime.OPENAI_COMPATIBLE, {}, WebOptions()),
        (Runtime.GOOGLE_API, {}, WebOptions(search_blocked_domains=("example.org",))),
    ],
)
def test_explicit_unsupported_controls_refuse(runtime, options, web):
    with pytest.raises(CapabilityNotSupported):
        validate_web(runtime, web, options)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"search": "invalid"},
        {"search": "off", "require_search": True},
        {"fetch_urls": "https://example.org"},
        {"fetch_urls": ("file:///tmp/a",)},
        {"search_allowed_domains": ("https://example.org",)},
        {"require_search": "yes"},
        {"search": "off"},
        {"fetch_urls": ("http://localhost:8080/admin",)},
        {"fetch_urls": ("http://127.0.0.1/",)},
        {"fetch_urls": ("http://169.254.169.254/latest/meta-data",)},
        {"fetch_urls": ("http://10.0.0.5/",)},
        {"fetch_urls": ("http://[::1]/",)},
        {"fetch_urls": ("http://intranet/",)},
        {"fetch_urls": ("https://build.internal/",)},
        {"search_allowed_domains": ("example.org",), "search_blocked_domains": ("other.org",)},
    ],
)
def test_bad_configuration_is_local(kwargs):
    with pytest.raises(ValueError):
        WebOptions(**kwargs)


def request(runtime, web):
    conn = Connection(
        name="api",
        runtime=runtime,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:TEST_KEY"),
        model="test-model",
    )
    return RunRequest(
        conn,
        (Message("user", "question"),),
        plan_launch(conn, {"TEST_KEY": "fake"}),
        model="test-model",
        web=web,
    )


def test_api_wire_tools_are_opt_in_and_keep_specific_controls():
    web = WebOptions(search_allowed_domains=("example.org",))
    adapter = AnthropicAPIAdapter()
    req = request(Runtime.ANTHROPIC_API, web)
    assert adapter.request_params(req, [])["tools"][0]["allowed_domains"] == ["example.org"]
    assert "tools" not in adapter.request_params(replace(req, web=None), [])
    req = request(Runtime.OPENAI_API, replace(web, search="cached"))
    params = OpenAIAPIAdapter().request_params(req, [])
    assert params["tools"][0]["external_web_access"] is False
    assert params["include"] == ["web_search_call.action.sources"]
    req = request(Runtime.GOOGLE_API, WebOptions(fetch_urls=("https://example.org",)))
    assert GoogleAPIAdapter().request_params(req, [])["config"]["tools"] == [
        {"google_search": {}},
        {"url_context": {}},
    ]


def test_codex_overrides_are_explicit_and_off_means_disabled():
    conn = connection(Runtime.OPENAI_SDK)
    req = RunRequest(conn, (), plan_launch(conn, {}), web=WebOptions(search="live"))
    assert web_config_overrides(req) == ["-c", 'web_search="live"']
    off = WebOptions(search="off", fetch_urls=("https://example.org",))
    assert web_config_overrides(replace(req, web=off)) == ["-c", 'web_search="disabled"']
    allowed = WebOptions(search_allowed_domains=("example.org", "docs.example.org"))
    assert web_config_overrides(replace(req, web=allowed)) == [
        "-c",
        'web_search="cached"',
        "-c",
        'tools.web_search.allowed_domains=["example.org", "docs.example.org"]',
    ]
    assert web_config_overrides(replace(req, web=None)) == []


def test_anthropic_native_evidence_distinguishes_errors_and_citations():
    events = anthropic_evidence(
        {
            "content": [
                {"type": "web_search_tool_result", "content": [{"url": "https://example.org"}]},
                {"type": "web_search_tool_result", "content": {"error_code": "max_uses_exceeded"}},
                {
                    "type": "text",
                    "citations": [{"url": "https://example.org", "cited_text": "proof"}],
                },
            ]
        }
    )
    evidence = events[-1]
    assert evidence.searches == 1
    assert evidence.errors == ("max_uses_exceeded",)
    assert evidence.citations[0].cited_text == "proof"


def test_openai_native_sources_and_citations_are_distinct():
    events = openai_evidence(
        {
            "output": [
                {
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "sources": [{"url": "https://read.example"}]},
                },
                {
                    "type": "message",
                    "content": [
                        {"annotations": [{"type": "url_citation", "url": "https://cited.example"}]}
                    ],
                },
            ]
        }
    )
    assert events[-1].sources[0].url == "https://read.example"
    assert events[-1].citations[0].url == "https://cited.example"


def test_google_metadata_keeps_search_presentation_and_citation_links():
    events = google_evidence(
        {
            "grounding_metadata": {
                "web_search_queries": ["query"],
                "grounding_chunks": [{"web": {"uri": "https://example.org"}}],
                "grounding_supports": [
                    {"grounding_chunk_indices": [0], "segment": {"text": "proof"}}
                ],
                "search_entry_point": {"rendered_content": "<div>Search</div>"},
            }
        }
    )
    assert events[-1].presentation["rendered_content"] == "<div>Search</div>"
    assert events[-1].citations[0].cited_text == "proof"


def test_unsupported_failover_is_refused_before_primary_runs(tmp_path):
    first = Connection(
        name="web-test",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:TEST_KEY"),
        guards=Guards(
            on_quota_exhausted=QuotaPolicy(
                action=QuotaAction.FAILOVER,
                failover="other",
            )
        ),
    )
    other = replace(connection(Runtime.OPENAI_SDK), name="other")
    bridge, _, adapter = fake_bridge(
        connections=[first, other],
        home=tmp_path,
        allow_real_adapters=True,
        runtime=Runtime.ANTHROPIC_API,
    )
    with pytest.raises(CapabilityNotSupported):
        bridge.chat(
            connection=first.name, message="question", web=WebOptions(require_citations=True)
        )
    assert not adapter.requests


def test_fetch_urls_reach_the_native_request_without_enabling_search(tmp_path):
    bridge, _, adapter = fake_bridge(
        connections=[connection()],
        script=[TextDeltaEvent("answer")],
        home=tmp_path,
    )
    answer = bridge.ask(
        "web-test",
        "read this",
        web=WebOptions(
            search="off",
            fetch_urls=("https://example.org",),
        ),
    )
    assert adapter.requests[0].messages[-1].text == "read this\nhttps://example.org"
    assert answer.receipt.web["search"] == "off"


def test_fetched_document_citations_resolve_native_document_indices():
    documents = [None]  # A user-provided PDF already occupies document index zero.
    evidence = anthropic_evidence(
        {
            "content": [
                {
                    "type": "web_fetch_tool_result",
                    "content": {
                        "type": "web_fetch_result",
                        "url": "https://example.org",
                        "content": {"title": "Page"},
                    },
                },
                {
                    "type": "text",
                    "citations": [
                        {"type": "char_location", "document_index": 1, "cited_text": "proof"}
                    ],
                },
            ]
        },
        documents,
    )[-1]
    assert evidence.citations[0].url == "https://example.org"
    assert documents == [None, "https://example.org"]


def test_installed_anthropic_sdk_web_surface():
    sdk = pytest.importorskip("claude_agent_sdk")
    assert {"tools", "allowed_tools", "disallowed_tools", "max_turns"} <= set(
        sdk.ClaudeAgentOptions.__dataclass_fields__
    )
    types = pytest.importorskip("anthropic.types")
    assert {"type", "name", "allowed_domains", "blocked_domains"} <= set(
        types.WebSearchTool20250305Param.__annotations__
    )
    assert "citations" in types.WebFetchTool20250910Param.__annotations__


def test_google_null_metadata_is_a_valid_ungrounded_answer():
    evidence = google_evidence(
        {
            "grounding_metadata": {
                "grounding_chunks": None,
                "grounding_supports": None,
                "web_search_queries": None,
            }
        }
    )[-1]
    assert evidence.searches == 0
    assert evidence.sources == evidence.citations == ()


def test_messages_server_pause_continues_without_losing_web_evidence(store):
    from test_adapter_anthropic_api import Factory, bridge_for, text_round
    from test_adapter_anthropic_api import connection as api_connection

    paused = text_round("Planning. ", stop_reason="pause_turn")
    paused.message.content.append(
        {
            "type": "web_search_tool_result",
            "tool_use_id": "search-1",
            "content": [{"type": "web_search_result", "url": "https://example.org"}],
        }
    )
    factory = Factory([paused, text_round("Answer.")])
    bridge, adapter = bridge_for(api_connection(), factory, store)
    adapter.is_available = lambda: True
    answer = bridge.ask("claude-key", "research", web=WebOptions(require_search=True))
    calls = factory.clients[0].calls
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["role"] == "assistant"
    assert calls[1]["messages"][-1]["content"][-1]["tool_use_id"] == "search-1"
    assert answer.text == "Planning. Answer."
    assert answer.sources[0].url == "https://example.org"


# --- review fixes, 2026-11-18 --------------------------------------------------------
#
# Each of these failed before the fix. The payloads are the shapes the installed
# SDKs produce -- ``model_dump`` writes an absent optional field as an explicit
# ``None`` -- which is what the first round of fixtures left out.


def test_google_evidence_survives_the_nulls_a_dumped_sdk_object_carries():
    evidence = google_evidence(
        {
            "grounding_metadata": {
                "grounding_chunks": [
                    {"web": None, "retrieved_context": {"uri": "gs://bucket/doc"}},
                    {"web": {"uri": "https://example.org", "title": "Example"}},
                ],
                "grounding_supports": [
                    {"grounding_chunk_indices": [0, 1], "segment": None},
                    {"grounding_chunk_indices": None, "segment": {"text": "unused"}},
                ],
            }
        }
    )[-1]
    assert evidence.sources == (WebSource("https://example.org", "Example"),)
    assert evidence.citations == (WebCitation("https://example.org", "Example", None),)


def test_a_google_chunk_with_no_web_metadata_emits_nothing():
    assert google_evidence({"content": {"parts": [{"text": "hello"}]}}) == []
    assert google_evidence({"grounding_metadata": None, "url_context_metadata": None}) == []


def test_a_fetched_document_citation_keeps_its_document_title():
    evidence = anthropic_evidence(
        {
            "content": [
                {
                    "type": "web_fetch_tool_result",
                    "content": {"url": "https://example.org", "content": {"title": "Page"}},
                },
                {
                    "type": "text",
                    "citations": [
                        {
                            "type": "char_location",
                            "document_index": 0,
                            "document_title": "Page",
                            "cited_text": "proof",
                        }
                    ],
                },
            ]
        }
    )[-1]
    assert evidence.citations == (WebCitation("https://example.org", "Page", "proof"),)


def test_a_failed_fetch_result_is_an_error_not_a_crash():
    error = {"type": "web_fetch_tool_result_error", "error_code": "url_not_allowed"}
    events = anthropic_evidence(
        {
            "content": [
                {"type": "web_fetch_tool_result", "tool_use_id": "f1", "content": error},
                {"type": "web_fetch_tool_result", "tool_use_id": "f2", "content": None},
            ]
        }
    )
    assert [e.status for e in events[:-1]] == ["failed", "failed"]
    assert events[-1].errors == ("url_not_allowed",)


def test_openai_says_nothing_about_access_unless_the_caller_named_a_mode():
    def tool(web):
        req = request(Runtime.OPENAI_API, web)
        return OpenAIAPIAdapter().request_params(req, [])["tools"][0]

    assert tool(WebOptions()) == {"type": "web_search"}
    assert tool(WebOptions(search="live"))["external_web_access"] is True
    assert tool(WebOptions(search="cached"))["external_web_access"] is False
    blocked = tool(WebOptions(search_blocked_domains=("spam.example",)))
    assert blocked["filters"] == {"blocked_domains": ["spam.example"]}


def test_anthropic_api_fetch_and_blocklist_reach_the_wire():
    web = WebOptions(
        search_blocked_domains=("spam.example",), fetch_urls=("https://example.org/a",)
    )
    tools = AnthropicAPIAdapter().request_params(request(Runtime.ANTHROPIC_API, web), [])["tools"]
    assert tools == [
        {"type": "web_search_20250305", "name": "web_search", "blocked_domains": ["spam.example"]},
        {"type": "web_fetch_20250910", "name": "web_fetch", "citations": {"enabled": True}},
    ]


def test_claude_code_search_results_printed_as_prose_still_yield_sources(tmp_path):
    """The CLI prints ``Links: [...]`` inside prose; the list is the evidence."""
    result = (
        'Web search results for query: "python release"\n\n'
        'Links: [{"title": "Python 3.14", "url": "https://www.python.org/downloads/"}, '
        '{"title": "PEP 745", "url": "https://peps.python.org/pep-0745/"}]\n\n'
        "The latest release is described at https://not-a-source.example/blog.\n\n"
        "REMINDER: You MUST include the sources above in your response."
    )
    bridge, _, _ = fake_bridge(
        connections=[connection()],
        script=[
            *tool_exchange("WebSearch", {"query": "python release"}, result),
            TextDeltaEvent("answer"),
        ],
        home=tmp_path,
    )
    answer = bridge.ask("web-test", "look it up", web=WebOptions(require_search=True))
    assert answer.sources == (
        WebSource("https://www.python.org/downloads/", "Python 3.14"),
        WebSource("https://peps.python.org/pep-0745/", "PEP 745"),
    )


def test_the_agent_sdk_run_allows_search_and_fetch_from_the_supplied_hosts_only(monkeypatch):
    """The options the real adapter launches with, not a mirror of them.

    WebFetch runs on this machine, so the permission rule is what keeps a run
    that was handed one URL from fetching any other host.
    """
    from test_anthropic_adapter import launch_options

    conn = connection()
    web = WebOptions(fetch_urls=("https://example.org/a", "https://docs.example.org/b"))
    req = RunRequest(conn, (Message("user", "read these"),), plan_launch(conn, {}), web=web)
    options = launch_options(monkeypatch, req)
    assert options["tools"] == ["WebSearch", "WebFetch"]
    assert options["allowed_tools"] == [
        "WebSearch",
        "WebFetch(domain:example.org)",
        "WebFetch(domain:docs.example.org)",
    ]
    assert "Bash" in options["disallowed_tools"]
    assert not {"WebSearch", "WebFetch"} & set(options["disallowed_tools"])
    assert options["max_turns"] is None

    fetch_only = replace(req, web=WebOptions(search="off", fetch_urls=("https://example.org/a",)))
    options = launch_options(monkeypatch, fetch_only)
    assert options["tools"] == ["WebFetch"]
    assert options["allowed_tools"] == ["WebFetch(domain:example.org)"]
    assert "WebSearch" in options["disallowed_tools"]
