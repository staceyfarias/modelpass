"""Native web opt-in, refusals, evidence and wire mappings without a network.

These tests hold the distinction between enabling a tool and observing its use,
including async calls, failed searches and a configured unsupported failover.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from modelpass import (
    AdapterFailed,
    Capability,
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
    WebEvidenceUnreadable,
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


# --- typed errors and the capability table, 2026-11-18 ----------------------------------
#
# These paths are live-unverified, so the failure to plan for is a vendor
# response in a shape nobody here has seen. What a caller gets then must be one
# typed error that says what could not be read -- never a bare TypeError.

JUNK = [1, "x", [None], [1], ["x"], [[]], {"a": None}]


def _malformed():
    for junk in JUNK:
        yield anthropic_evidence, {"content": junk}
        yield anthropic_evidence, {"content": [{"type": "web_search_tool_result", "content": junk}]}
        yield anthropic_evidence, {"content": [{"type": "text", "citations": junk}]}
        yield openai_evidence, {"output": junk}
        yield openai_evidence, {"output": [{"type": "web_search_call", "action": junk}]}
        yield openai_evidence, {"output": [{"type": "message", "content": junk}]}
        yield google_evidence, {"grounding_metadata": junk}
        yield google_evidence, {"grounding_metadata": {"grounding_chunks": junk}}
        yield google_evidence, {"grounding_metadata": {"grounding_supports": junk}}
        yield google_evidence, {"url_context_metadata": junk}


def test_a_malformed_vendor_response_is_never_a_bare_python_error():
    """Each reader either copes or raises the one typed error. Nothing else."""
    raised = 0
    for reader, payload in _malformed():
        try:
            reader(payload)
        except WebEvidenceUnreadable:
            raised += 1
    assert raised > 20


def test_the_typed_error_says_what_could_not_be_read_and_shows_no_content():
    payload = {
        "id": "resp_1",
        "output": [{"type": "web_search_call", "action": "secret query text"}],
    }
    with pytest.raises(WebEvidenceUnreadable) as caught:
        openai_evidence(payload)
    error = caught.value
    assert isinstance(error, AdapterFailed)
    assert error.runtime == "openai-api"
    message = str(error)
    assert "the web search items of a Responses response could not be read" in message
    assert "AttributeError" in message
    assert "Response keys: id, output" in message
    assert "may have completed and been billed" in message
    assert "secret query text" not in message


def test_an_unreadable_response_reaches_the_caller_of_ask_as_the_typed_error(store):
    """Through the real adapter and the real bridge, not just the reader."""
    from test_adapter_anthropic_api import Factory, bridge_for, text_round
    from test_adapter_anthropic_api import connection as api_connection

    broken = text_round("Answer.")
    broken.message.content.append({"type": "web_search_tool_result", "content": "not a list"})
    bridge, adapter = bridge_for(api_connection(), Factory([broken]), store)
    adapter.is_available = lambda: True
    with pytest.raises(WebEvidenceUnreadable, match="anthropic-api: the web search/fetch blocks"):
        bridge.ask("claude-key", "research", web=WebOptions())


def test_an_unreadable_tool_event_is_the_typed_error_too(tmp_path):
    from modelpass.types import VendorEvent

    broken = VendorEvent(
        runtime=Runtime.OPENAI_SDK,
        name="item/completed",
        data={"item": {"type": "webSearch", "id": "w1", "query": "q", "action": "search"}},
    )
    bridge, _, _ = fake_bridge(
        connections=[connection(Runtime.OPENAI_SDK)],
        script=[broken, TextDeltaEvent("answer")],
        home=tmp_path,
        runtime=Runtime.OPENAI_SDK,
    )
    with pytest.raises(WebEvidenceUnreadable, match="openai-sdk: the web activity in a vendor"):
        bridge.ask("web-test", "look it up", web=WebOptions())


def test_a_codex_web_search_item_is_read_as_its_schema_defines_it(tmp_path):
    """WebSearchThreadItem, from `codex app-server generate-json-schema` (0.151.0)."""
    from modelpass.types import VendorEvent

    def item(action, **extra):
        return VendorEvent(
            runtime=Runtime.OPENAI_SDK,
            name="item/completed",
            data={"item": {"type": "webSearch", "id": "w1", "query": "python", **extra,
                           "action": action}},
        )

    bridge, _, _ = fake_bridge(
        connections=[connection(Runtime.OPENAI_SDK)],
        script=[
            item({"type": "search", "query": "python", "queries": ["python"]},
                 results=[{"url": "https://www.python.org", "title": "Python"}]),
            item({"type": "openPage", "url": "https://www.python.org"}),
            item({"type": "findInPage", "url": "https://www.python.org", "pattern": "3.14"}),
            item(None, results=None),
            TextDeltaEvent("answer"),
        ],
        home=tmp_path,
        runtime=Runtime.OPENAI_SDK,
    )
    answer = bridge.ask("web-test", "look it up", web=WebOptions(require_search=True))
    activity = [e for e in answer.events if e.type == "web_activity"]
    assert [e.action for e in activity] == ["search", "open_page", "find_in_page", "search"]
    assert activity[1].url == "https://www.python.org"
    assert answer.sources == (WebSource("https://www.python.org", "Python"),)


def _every_option():
    yield "web.search", WebOptions()
    yield "web.cached", WebOptions(search="cached")
    yield "web.fetch_urls", WebOptions(search="off", fetch_urls=("https://example.org",))
    yield "web.domains", WebOptions(search_allowed_domains=("example.org",))
    yield "web.citations", WebOptions(require_citations=True)


@pytest.mark.parametrize("runtime", list(Runtime))
def test_the_capability_table_and_the_refusals_agree(runtime):
    """One source of truth: a cell that is not `supported` is always refused,
    and a refusal never contradicts a cell that says `supported`."""
    from modelpass.capabilities import STATIC_TABLE, Support
    from modelpass.web import web_capabilities

    for feature, web in _every_option():
        cells = [STATIC_TABLE[runtime][cap] for cap, _ in web_capabilities(web)]
        supported = all(cell is Support.SUPPORTED for cell in cells)
        try:
            validate_web(runtime, web, {})
            accepted = True
        except CapabilityNotSupported:
            accepted = False
        assert accepted == supported, (runtime.value, feature)


@pytest.mark.parametrize("runtime", [Runtime.GOOGLE_CLI, Runtime.GOOGLE_SDK])
def test_a_runtime_with_no_web_adapter_refuses_instead_of_ignoring(runtime):
    with pytest.raises(CapabilityNotSupported, match="web_search reads unsupported"):
        validate_web(runtime, WebOptions(), {})


def test_every_web_capability_cell_has_its_own_note():
    from modelpass.capabilities import CapabilityRegistry, Support

    registry = CapabilityRegistry()
    web_cells = [c for c in Capability if c.value.startswith("web_") or c.value == "research_jobs"]
    for runtime in Runtime:
        for capability in web_cells:
            note = registry.note(runtime, capability) or ""
            support = registry.support(runtime, capability)
            assert support.value in note, (runtime.value, capability.value)
            if support is Support.SUPPORTED:
                assert "Held by" in note
                assert ("live-verified 2026-11-18" in note) != ("live-untested" in note)
            elif support is Support.UNVERIFIED:
                assert capability is Capability.RESEARCH_JOBS
                assert "unimplemented" in note
            else:
                assert "refused" in note or "refuses" in note
    supported_notes = {
        registry.note(r, c)
        for r in Runtime
        for c in web_cells
        if registry.support(r, c) is Support.SUPPORTED
    }
    assert len(supported_notes) == sum(
        registry.support(r, c) is Support.SUPPORTED for r in Runtime for c in web_cells
    )


def test_a_mistyped_web_option_is_a_value_error_that_names_it():
    with pytest.raises(ValueError, match=r"unknown web option.*serach.*Known options: search"):
        WebOptions.coerce({"serach": "live"})
    with pytest.raises(ValueError, match=r"web.search must be"):
        WebOptions(search=["live"])
    with pytest.raises(ValueError, match="WebOptions, a mapping, or None"):
        WebOptions.coerce("live")


# --- after the first live drive, 2026-11-18 ------------------------------------------


@pytest.mark.parametrize(
    "runtime,status",
    [
        (Runtime.ANTHROPIC_API, "verified 2026-11-18"),
        (Runtime.ANTHROPIC_SDK, "verified 2026-11-18"),
        (Runtime.OPENAI_SDK, "verified 2026-11-18"),
        (Runtime.OPENAI_API, "untested"),
        (Runtime.GOOGLE_API, "untested"),
    ],
)
def test_the_receipt_says_whether_this_runtimes_web_path_has_met_a_vendor(runtime, status):
    from modelpass.web import web_receipt

    receipt = web_receipt(runtime, WebOptions())
    assert receipt["live_verification"] == status
    assert receipt["verification"] == ("offline" if status == "untested" else "live")


def test_only_the_driven_runtimes_have_live_verified_cells():
    """A cell says live-verified only on a runtime the receipt also calls verified."""
    from modelpass.capabilities import CapabilityRegistry
    from modelpass.web import WEB_LIVE_VERIFIED

    registry = CapabilityRegistry()
    for runtime in Runtime:
        for capability in Capability:
            note = registry.note(runtime, capability) or ""
            if "live-verified 2026-11-18" in note:
                assert runtime in WEB_LIVE_VERIFIED, (runtime.value, capability.value)
    for runtime in (Runtime.OPENAI_API, Runtime.GOOGLE_API):
        assert "live-untested" in registry.note(runtime, Capability.WEB_SEARCH)


def test_hosted_research_is_recorded_as_unimplemented_everywhere():
    """2026-11-18: a draft that has never run is not a supported capability."""
    from modelpass.capabilities import CapabilityRegistry, Support

    registry = CapabilityRegistry()
    for runtime in Runtime:
        support = registry.support(runtime, Capability.RESEARCH_JOBS)
        assert support is not Support.SUPPORTED, runtime.value
        drafted = runtime in (Runtime.OPENAI_API, Runtime.GOOGLE_API)
        assert support is (Support.UNVERIFIED if drafted else Support.UNSUPPORTED)
        if drafted:
            assert "unimplemented" in registry.note(runtime, Capability.RESEARCH_JOBS)


def test_a_completed_claude_code_fetch_names_its_url_as_a_source(tmp_path):
    def run(is_error):
        bridge, _, _ = fake_bridge(
            connections=[connection()],
            script=[
                *tool_exchange(
                    "WebFetch",
                    {"url": "https://www.python.org/downloads/", "prompt": "heading?"},
                    "The heading is Download Python.",
                    is_error=is_error,
                ),
                TextDeltaEvent("answer"),
            ],
            home=tmp_path / str(is_error),
        )
        web = WebOptions(search="off", fetch_urls=("https://www.python.org/downloads/",))
        return bridge.ask("web-test", "read it", web=web)

    assert run(False).sources == (WebSource("https://www.python.org/downloads/"),)
    assert run(True).sources is None


def test_dumping_a_vendor_object_does_not_print_serializer_warnings():
    """Seen live: one warning per block on stderr for every anthropic web call."""
    import warnings

    from modelpass.web import _dict

    class Dumped:
        def model_dump(self, *, mode, warnings=True):
            if warnings:
                import warnings as w

                w.warn("PydanticSerializationUnexpectedValue", UserWarning, stacklevel=1)
            return {"content": []}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _dict(Dumped()) == {"content": []}
