"""Portable native web options, wire mappings and evidence collection."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any
from urllib.parse import urlsplit

from .errors import CapabilityNotSupported
from .runtimes import Runtime
from .types import (
    AgentEvent,
    TerminalEvent,
    TerminalStatus,
    ToolCallEvent,
    ToolResultEvent,
    VendorEvent,
    WebActivityEvent,
    WebCitation,
    WebEvidenceEvent,
    WebSource,
)


@dataclass(frozen=True, slots=True)
class WebOptions:
    """Opt into native web tools. Unsupported explicit requests are refused."""

    search: str = "provider_default"
    fetch_urls: tuple[str, ...] = ()
    search_allowed_domains: tuple[str, ...] = ()
    search_blocked_domains: tuple[str, ...] = ()
    require_search: bool = False
    require_citations: bool = False

    def __post_init__(self) -> None:
        if self.search not in {"off", "provider_default", "live", "cached"}:
            raise ValueError("web.search must be off, provider_default, live or cached")
        for name in ("fetch_urls", "search_allowed_domains", "search_blocked_domains"):
            value = getattr(self, name)
            if isinstance(value, str) or not isinstance(value, (tuple, list)):
                raise ValueError(f"web.{name} must be a sequence of strings")
            if any(not isinstance(item, str) or not item for item in value):
                raise ValueError(f"web.{name} must contain nonempty strings")
            object.__setattr__(self, name, tuple(dict.fromkeys(value)))
        for name in ("require_search", "require_citations"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"web.{name} must be a boolean")
        if self.search_allowed_domains and self.search_blocked_domains:
            raise ValueError("choose web.search_allowed_domains or search_blocked_domains")
        if self.search == "off" and (
            self.require_search or self.search_allowed_domains or self.search_blocked_domains
        ):
            raise ValueError("web search options require search to be enabled")
        if self.search == "off" and not self.fetch_urls:
            # 2026-11-18: this used to reach the API runtimes as ``tools: []``.
            raise ValueError("web requires search or fetch_urls; pass web=None for neither")
        for url in self.fetch_urls:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
            ):
                raise ValueError("web.fetch_urls requires HTTP(S) URLs without credentials")
            if not _public_host(parsed.hostname):
                # 2026-11-18. On anthropic-sdk the URL tool runs on this
                # machine, so a loopback or private address is a request to
                # read the caller's own network, which this option is not for.
                raise ValueError(
                    "web.fetch_urls requires public hosts, not localhost or a "
                    "private, loopback or link-local address"
                )
        for domain in (*self.search_allowed_domains, *self.search_blocked_domains):
            parsed = urlsplit("https://" + domain)
            if (
                parsed.hostname != domain
                or parsed.path
                or parsed.query
                or parsed.fragment
                or parsed.port is not None
            ):
                raise ValueError("web search domains must be bare lowercase hostnames")

    @classmethod
    def coerce(cls, value: WebOptions | Mapping[str, Any] | None) -> WebOptions | None:
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**value)
        raise ValueError("web must be WebOptions, a mapping, or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


def _public_host(host: str) -> bool:
    """Whether a URL host is one a remote tool could be meant to read.

    A literal check only: a public name that *resolves* to a private address is
    not caught here, and nothing is resolved to find out.
    """
    name = host.lower().rstrip(".")
    if name == "localhost" or name.endswith((".localhost", ".local", ".internal")):
        return False
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return "." in name
    return address.is_global


def refuse(runtime: Runtime, feature: str, reason: str) -> None:
    raise CapabilityNotSupported(runtime.value, feature, "unsupported", reason)


def validate_web(runtime: Runtime, web: WebOptions | None, options: Mapping[str, Any]) -> None:
    """Validate controls before preflight, including the selected transport."""
    if web is None:
        return
    if runtime is Runtime.OPENAI_COMPATIBLE:
        refuse(runtime, "web", "OpenAI-compatible endpoints have no portable native web tool")
    if options.get("native_tools"):
        refuse(runtime, "web", "web= requires the isolated stateless tool configuration")
    if runtime is Runtime.OPENAI_SDK and options.get("transport", "app-server") != "app-server":
        refuse(runtime, "web", "web= currently requires the Codex app-server transport")
    if web.search == "cached" and runtime not in {Runtime.OPENAI_SDK, Runtime.OPENAI_API}:
        refuse(runtime, "web.cached", "this runtime has no cache-only search control")
    if runtime in {Runtime.ANTHROPIC_SDK, Runtime.GOOGLE_API} and (
        web.search_allowed_domains or web.search_blocked_domains
    ):
        refuse(runtime, "web.domains", "this adapter cannot enforce search domain restrictions")
    if runtime is Runtime.OPENAI_SDK:
        if web.search_blocked_domains:
            refuse(runtime, "web.domains", "Codex exposes an allowlist only")
        if web.fetch_urls:
            refuse(runtime, "web.fetch_urls", "standalone URL reading is not verified on Codex")
        if web.require_citations:
            refuse(runtime, "web.citations", "Codex has no verified native answer citation schema")
    if runtime is Runtime.OPENAI_API and web.fetch_urls:
        refuse(runtime, "web.fetch_urls", "standalone URL reading is not verified on Responses")
    if runtime is Runtime.ANTHROPIC_SDK and web.require_citations:
        refuse(
            runtime,
            "web.citations",
            "the Agent SDK exposes sources without native answer citations",
        )


def web_receipt(runtime: Runtime, web: WebOptions) -> dict[str, Any]:
    effective = web.search
    if effective == "provider_default":
        effective = "cached" if runtime is Runtime.OPENAI_SDK else "live"
    return {
        "requested": web.to_dict(),
        "search": effective,
        "verification": "offline",
        "live_verification": "untested",
    }


def native_web_tools(runtime: Runtime, web: WebOptions | None) -> list[dict[str, Any]]:
    if web is None:
        return []
    tools: list[dict[str, Any]] = []
    if web.search != "off":
        if runtime is Runtime.ANTHROPIC_API:
            tool: dict[str, Any] = {"type": "web_search_20250305", "name": "web_search"}
            if web.search_allowed_domains:
                tool["allowed_domains"] = list(web.search_allowed_domains)
            if web.search_blocked_domains:
                tool["blocked_domains"] = list(web.search_blocked_domains)
        elif runtime is Runtime.OPENAI_API:
            # ``external_web_access`` and ``filters.blocked_domains`` are in the
            # vendor's web search guide (read 2026-11-18: "Default is true (live
            # access) if you do not set it") and absent from openai 2.32.0's
            # WebSearchToolParam, which is a TypedDict and passes them through.
            # So the access field is sent only when the caller stated a mode.
            tool = {"type": "web_search"}
            if web.search in {"live", "cached"}:
                tool["external_web_access"] = web.search == "live"
            filters = {}
            if web.search_allowed_domains:
                filters["allowed_domains"] = list(web.search_allowed_domains)
            if web.search_blocked_domains:
                filters["blocked_domains"] = list(web.search_blocked_domains)
            if filters:
                tool["filters"] = filters
        else:
            tool = {"google_search": {}}
        tools.append(tool)
    if web.fetch_urls:
        tools.append(
            {"type": "web_fetch_20250910", "name": "web_fetch", "citations": {"enabled": True}}
            if runtime is Runtime.ANTHROPIC_API
            else {"url_context": {}}
        )
    return tools


def sdk_web_tools(web: WebOptions | None) -> list[str]:
    """The exact built-in tool names explicitly enabled on Claude Agent SDK."""
    if web is None:
        return []
    names = ["WebSearch"] if web.search != "off" else []
    if web.fetch_urls:
        names.append("WebFetch")
    return names


def sdk_web_permissions(web: WebOptions | None) -> list[str]:
    """Permission rules for those tools: fetch only from the supplied hosts.

    2026-11-18. Claude Code's WebFetch runs in the local process and takes any
    URL the model writes. Allowing the bare tool name let a run fetch anything;
    ``WebFetch(domain:<host>)`` is Claude Code's own rule syntax (read from the
    CLI bundled with claude-agent-sdk 0.2.180) and allows one host each. Whether
    a redirect off that host is followed is the CLI's behaviour, not checked.
    """
    if web is None:
        return []
    rules = ["WebSearch"] if web.search != "off" else []
    hosts = dict.fromkeys(urlsplit(url).hostname for url in web.fetch_urls)
    rules.extend(f"WebFetch(domain:{host})" for host in hosts)
    return rules


def _dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    return value


def anthropic_evidence(
    message: Any, document_urls: list[str | None] | None = None
) -> list[AgentEvent]:
    """Read completed server tool blocks, preserving tool errors and citations."""
    value = _dict(message)
    blocks = value.get("content", []) if isinstance(value, dict) else []
    sources: list[WebSource] = []
    citations: list[WebCitation] = []
    errors: list[str] = []
    searches = 0
    events: list[AgentEvent] = []
    documents = document_urls if document_urls is not None else []
    for block in blocks:
        kind = block.get("type")
        if kind == "server_tool_use" and block.get("name") in {"web_search", "web_fetch"}:
            args = block.get("input", {})
            events.append(
                WebActivityEvent(
                    action="search" if block["name"] == "web_search" else "fetch",
                    status="started",
                    id=block.get("id", ""),
                    query=args.get("query"),
                    url=args.get("url"),
                )
            )
        if kind == "web_search_tool_result":
            content = block.get("content", [])
            if isinstance(content, list):
                searches += 1
                sources.extend(
                    WebSource(item["url"], item.get("title")) for item in content if item.get("url")
                )
            else:
                errors.append(str(content.get("error_code", "web search failed")))
            events.append(
                WebActivityEvent(
                    "search",
                    "completed" if isinstance(content, list) else "failed",
                    block.get("tool_use_id", ""),
                )
            )
        if kind == "web_fetch_tool_result":
            content = block.get("content")
            content = content if isinstance(content, dict) else {}
            if content.get("url"):
                title = (content.get("content") or {}).get("title")
                sources.append(WebSource(content["url"], title))
                documents.append(content["url"])
            if content.get("error_code"):
                errors.append(str(content["error_code"]))
            events.append(
                WebActivityEvent(
                    "fetch",
                    "completed" if content.get("url") else "failed",
                    block.get("tool_use_id", ""),
                    url=content.get("url"),
                )
            )
        if kind == "text":
            for citation in block.get("citations") or []:
                url = citation.get("url")
                index = citation.get("document_index")
                if url is None and isinstance(index, int) and 0 <= index < len(documents):
                    url = documents[index]
                if url:
                    # A search citation names its page ``title``; a fetched
                    # document's citation names it ``document_title``.
                    title = citation.get("title") or citation.get("document_title")
                    citations.append(WebCitation(url, title, citation.get("cited_text")))
    events.append(WebEvidenceEvent(tuple(sources), tuple(citations), searches, tuple(errors)))
    return events


def openai_evidence(response: Any) -> list[AgentEvent]:
    value = _dict(response)
    if not isinstance(value, dict):
        return []
    sources: list[WebSource] = []
    citations: list[WebCitation] = []
    searches = 0
    events: list[AgentEvent] = []
    for item in value.get("output", []) or []:
        if item.get("type") == "web_search_call":
            action = item.get("action") or {}
            status = item.get("status", "unknown")
            kind = action.get("type", "search")
            searches += int(kind == "search" and status == "completed")
            events.append(
                WebActivityEvent(
                    kind, status, item.get("id", ""), action.get("query"), action.get("url")
                )
            )
            sources.extend(
                WebSource(s["url"], s.get("title"))
                for s in (action.get("sources", []) or [])
                if s.get("url")
            )
        if item.get("type") == "message":
            for content in item.get("content", []) or []:
                for annotation in content.get("annotations", []) or []:
                    if annotation.get("type") == "url_citation" and annotation.get("url"):
                        citations.append(WebCitation(annotation["url"], annotation.get("title")))
    events.append(WebEvidenceEvent(tuple(sources), tuple(citations), searches))
    return events


def google_evidence(candidate: Any) -> list[AgentEvent]:
    value = _dict(candidate)
    if not isinstance(value, dict):
        return []
    if value.get("grounding_metadata") is None and value.get("url_context_metadata") is None:
        # Most streamed chunks carry no web metadata at all. Saying nothing for
        # them keeps one answer from emitting an empty evidence event per chunk.
        return []
    meta = value.get("grounding_metadata") or {}
    # A dumped SDK object writes ``"web": None`` on a chunk that is not a web
    # chunk and ``"segment": None`` on a support without one (2026-11-18), so
    # every nested read here is ``or {}``, never a ``.get`` default.
    chunks = [(chunk or {}).get("web") or {} for chunk in meta.get("grounding_chunks") or []]
    sources = tuple(WebSource(web["uri"], web.get("title")) for web in chunks if web.get("uri"))
    citations = []
    for support in meta.get("grounding_supports") or []:
        support = support or {}
        for index in support.get("grounding_chunk_indices") or []:
            if 0 <= index < len(chunks) and chunks[index].get("uri"):
                source = chunks[index]
                citations.append(
                    WebCitation(
                        source["uri"],
                        source.get("title"),
                        (support.get("segment") or {}).get("text"),
                    )
                )
    queries = meta.get("web_search_queries") or []
    events: list[AgentEvent] = [
        WebActivityEvent("search", "completed", query=query) for query in queries
    ]
    fetched_sources = []
    url_context = value.get("url_context_metadata") or {}
    for item in url_context.get("url_metadata") or []:
        url = item.get("retrieved_url")
        status = item.get("url_retrieval_status")
        succeeded = status == "URL_RETRIEVAL_STATUS_SUCCESS"
        events.append(WebActivityEvent("fetch", "completed" if succeeded else "failed", url=url))
        if succeeded and url:
            fetched_sources.append(WebSource(url))
    events.append(
        WebEvidenceEvent(
            (*sources, *fetched_sources),
            tuple(citations),
            len(queries),
            presentation=meta.get("search_entry_point") or {},
        )
    )
    return events


class WebState:
    """Per-run web observations and required-evidence checks, shared by both faces."""

    def __init__(self, options: WebOptions) -> None:
        self.options = options
        self.calls: dict[str, ToolCallEvent] = {}
        self.searches = 0
        self.citations = 0

    def observe(self, event: AgentEvent) -> list[AgentEvent]:
        out: list[AgentEvent] = []
        if isinstance(event, WebEvidenceEvent):
            self.searches += event.searches
            self.citations += len(event.citations or ())
        elif isinstance(event, ToolCallEvent) and event.name in {"WebSearch", "WebFetch"}:
            self.calls[event.id or ""] = event
            out.append(
                WebActivityEvent(
                    "search" if event.name == "WebSearch" else "fetch",
                    "started",
                    event.id or "",
                    event.arguments.get("query"),
                    event.arguments.get("url"),
                )
            )
        elif isinstance(event, ToolResultEvent) and (event.id or "") in self.calls:
            call = self.calls[event.id or ""]
            success = not event.is_error
            self.searches += int(success and call.name == "WebSearch")
            out.append(
                WebActivityEvent(
                    "search" if call.name == "WebSearch" else "fetch",
                    "completed" if success else "failed",
                    event.id or "",
                )
            )
            try:
                content = json.loads(event.content)
            except (ValueError, TypeError):
                content = _links_values(event.content)
            sources = tuple(dict.fromkeys(_source_values(content)))
            if sources:
                out.append(WebEvidenceEvent(sources=sources))
        elif isinstance(event, VendorEvent) and event.runtime is Runtime.OPENAI_SDK:
            item = event.data.get("item", {})
            if item.get("type") == "webSearch" and event.name.startswith("item/completed"):
                action = item.get("action") or {}
                kind = action.get("type", "search")
                self.searches += int(kind == "search")
                out.append(
                    WebActivityEvent(
                        kind, "completed", item.get("id", ""), item.get("query"), action.get("url")
                    )
                )
        return out

    def failure(self, event: TerminalEvent) -> str | None:
        if event.status is not TerminalStatus.OK:
            return None
        if self.options.require_search and not self.searches:
            return "web.require_search was requested but no completed native search was observed"
        if self.options.require_citations and not self.citations:
            return "web.require_citations was requested but no native answer citation was returned"
        return None


def _links_values(text: Any) -> list[Any]:
    """The JSON lists Claude Code's WebSearch result prints after ``Links:``.

    The CLI bundled with claude-agent-sdk 0.2.180 returns the result as prose:
    a ``Web search results for query: "..."`` line, then ``Links: [...]`` with
    one JSON list of ``{"title", "url"}`` per result block (2026-11-18). Only
    those lists are read. A URL in the surrounding prose is the model's own
    writing and is not a source.
    """
    if not isinstance(text, str):
        return []
    found: list[Any] = []
    decoder = json.JSONDecoder()
    position = text.find("Links:")
    while position != -1:
        start = position + len("Links:")
        start += len(text[start:]) - len(text[start:].lstrip())
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            value = None
        if isinstance(value, list):
            found.append(value)
        position = text.find("Links:", start)
    return found


def _source_values(value: Any):
    """Extract structured SDK result URLs without interpreting prose links."""
    if isinstance(value, dict):
        if isinstance(value.get("url"), str):
            yield WebSource(value["url"], value.get("title"))
        for child in value.values():
            yield from _source_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _source_values(child)
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return
        if isinstance(decoded, (dict, list)):
            yield from _source_values(decoded)
