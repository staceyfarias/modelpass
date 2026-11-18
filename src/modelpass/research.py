"""Explicit hosted research jobs. OpenAI and Gemini backends are live-untested.

Submission, polling and remote cancellation are separate operations. No operation
retries, changes connection, or starts a local worker to impersonate a hosted job.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .errors import CapabilityNotSupported, InvalidConnection, SubpassError, VendorRunFailed
from .preflight import Receipt
from .runtimes import Runtime
from .types import WebCitation, WebEvidenceEvent, WebSource
from .web import WebOptions, openai_evidence, web_receipt

if TYPE_CHECKING:
    from .bridge import Bridge


@dataclass(frozen=True, slots=True)
class ResearchJob:
    """Persistable handle bound to the original connection and API identity."""

    id: str
    connection: str
    runtime: str
    identity: str
    web: Mapping[str, Any]
    model: str | None = None
    receipt: Receipt | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not all(
            isinstance(v, str) and v
            for v in (self.id, self.connection, self.runtime, self.identity)
        ):
            raise ValueError("research job identifiers must be nonempty strings")
        if self.runtime not in {Runtime.OPENAI_API.value, Runtime.GOOGLE_API.value}:
            raise ValueError("unknown hosted research runtime")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("research job model must be a nonempty string or None")
        if not isinstance(self.web, Mapping):
            raise ValueError("research job web options must be a mapping")
        object.__setattr__(self, "web", WebOptions.coerce(self.web).to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if key != "receipt"}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchJob:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ResearchResult:
    """Research output. The snapshot status identifies partial output."""

    text: str
    sources: tuple[WebSource, ...] | None = None
    citations: tuple[WebCitation, ...] | None = None
    presentation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResearchSnapshot:
    job: ResearchJob
    status: str
    vendor_status: str
    receipt: Receipt
    result: ResearchResult | None = None
    reason: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)


def _timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("research timeout must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("research timeout must be a positive finite number")
    return float(value)


def _identity(connection, receipt: Receipt) -> str:
    identity = {
        "connection": connection.name,
        "runtime": connection.runtime.value,
        "base_url": connection.base_url,
        "credential_ref": connection.credential_ref.to_dict(),
        "account": receipt.account,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _supported(runtime: Runtime) -> None:
    if runtime not in {Runtime.OPENAI_API, Runtime.GOOGLE_API}:
        raise CapabilityNotSupported(
            runtime.value,
            "research_jobs",
            "unsupported",
            "hosted research jobs require an OpenAI API or Gemini API connection",
        )


def _resource(client, runtime: Runtime):
    if runtime is Runtime.OPENAI_API:
        return client.responses
    # google-genai 1.73.1 maps attempts=1 to nextgen max_retries=1.
    # Override that retry explicitly on the resource used by Interactions.
    return client._nextgen_client.with_options(max_retries=0).interactions


def _client(adapter, connection):
    try:
        return adapter._client(connection)
    except SubpassError:
        raise
    except Exception as exc:
        raise VendorRunFailed(
            f"research client could not be constructed ({type(exc).__name__})"
        ) from exc


def start_research(
    bridge: Bridge,
    connection,
    message: str,
    *,
    model: str | None = None,
    agent: str | None = None,
    web: WebOptions | Mapping[str, Any] | None = None,
    timeout: float = 30,
) -> ResearchJob:
    """Submit once. A failed submission may have started a remote job."""
    timeout = _timeout(timeout)
    resolved = bridge._resolve(connection)
    _supported(resolved.runtime)
    if not isinstance(message, str) or not message.strip():
        raise ValueError("research requires a nonempty message")
    if resolved.runtime is Runtime.GOOGLE_API:
        if model is not None or not isinstance(agent, str) or not agent.strip():
            raise ValueError("Gemini research requires agent= and does not accept model=")
    elif agent is not None or not (model or resolved.model):
        raise ValueError("OpenAI research requires model= and does not accept agent=")
    options = WebOptions.coerce(web) or WebOptions()
    if (
        resolved.runtime is Runtime.GOOGLE_API
        and options.search == "off"
        and not options.fetch_urls
    ):
        raise ValueError("Gemini web research requires search or fetch_urls")
    plan = bridge._plan(
        connection=resolved,
        message=message,
        model=agent if resolved.runtime is Runtime.GOOGLE_API else model,
        web=options,
        allow_failover=False,
    )
    if plan.guards.stop_at_tokens or plan.guards.warn_at_tokens:
        raise CapabilityNotSupported(
            resolved.runtime.value,
            "research.guards",
            "unsupported",
            "detached jobs cannot enforce local token guards",
        )
    receipt = bridge._receipt_for(plan)
    receipt = replace(
        receipt,
        web={
            **(receipt.web or {}),
            "research_backend": resolved.runtime.value,
            "storage": "provider-managed background job",
            "live_verification": "untested",
        },
    )
    adapter = plan.adapter
    client = _client(adapter, resolved)
    try:
        resource = _resource(client, resolved.runtime)
        if resolved.runtime is Runtime.OPENAI_API:
            from .adapters.openai_api import input_items

            params = adapter.request_params(plan.request, input_items(plan.request))
            params.update(background=True, store=True, stream=False, timeout=timeout)
        else:
            tools = []
            if options.search != "off":
                tools.append({"type": "google_search"})
            # Research reads search results with URL Context as well as supplied URLs.
            if options.search != "off" or options.fetch_urls:
                tools.append({"type": "url_context"})
            params = {
                "agent": agent,
                "input": plan.request.messages[-1].text,
                "tools": tools,
                "background": True,
                "stream": False,
                "store": True,
                "timeout": timeout,
            }
        value = resource.create(**params)
    except Exception as exc:
        raise VendorRunFailed(
            f"research submission failed ({type(exc).__name__}); remote submission outcome "
            "may be unknown. No automatic retry was attempted"
        ) from exc
    finally:
        if getattr(adapter, "_owns_clients", True):
            with suppress(Exception):
                client.close()
    data = _data(value)
    if not isinstance(data.get("id"), str) or not data["id"]:
        raise VendorRunFailed("research submission returned no job ID; remote outcome is unknown")
    return ResearchJob(
        data["id"],
        resolved.name,
        resolved.runtime.value,
        _identity(resolved, receipt),
        options.to_dict(),
        plan.request.model,
        receipt,
    )


def get_research(
    bridge: Bridge, job: ResearchJob, *, timeout: float = 30, cancel: bool = False
) -> ResearchSnapshot:
    """Poll once or request remote cancellation through the original connection."""
    timeout = _timeout(timeout)
    if not isinstance(job, ResearchJob):
        raise ValueError("job must be a ResearchJob; use ResearchJob.from_dict to restore one")
    resolved = bridge._resolve(job.connection)
    _supported(resolved.runtime)
    if resolved.runtime.value != job.runtime:
        raise InvalidConnection("research connection runtime changed")
    plan = bridge._plan(connection=resolved, message="", model=job.model, allow_failover=False)
    receipt = bridge._receipt_for(plan)
    receipt = replace(
        receipt,
        web={
            **web_receipt(resolved.runtime, WebOptions.coerce(job.web)),
            "research_backend": resolved.runtime.value,
            "storage": "provider-managed background job",
            "live_verification": "untested",
        },
    )
    if _identity(resolved, receipt) != job.identity:
        raise InvalidConnection("research connection endpoint or credential identity changed")
    client = _client(plan.adapter, resolved)
    try:
        resource = _resource(client, resolved.runtime)
        if cancel:
            value = resource.cancel(job.id, timeout=timeout)
        elif resolved.runtime is Runtime.OPENAI_API:
            value = resource.retrieve(
                job.id, include=["web_search_call.action.sources"], timeout=timeout
            )
        else:
            value = resource.get(job.id, timeout=timeout)
    except Exception as exc:
        raise VendorRunFailed(
            f"research {'cancellation' if cancel else 'retrieval'} failed "
            f"({type(exc).__name__}); remote state is unknown"
        ) from exc
    finally:
        if getattr(plan.adapter, "_owns_clients", True):
            with suppress(Exception):
                client.close()
    return _snapshot(job, receipt, _data(value))


def _data(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value)


def _snapshot(job: ResearchJob, receipt: Receipt, data: dict[str, Any]) -> ResearchSnapshot:
    vendor_status = data.get("status", "unknown")
    status = {
        "queued": "pending",
        "in_progress": "running",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
        "cancelling": "running",
        "incomplete": "incomplete",
        "requires_action": "requires_action",
    }.get(vendor_status, "unknown")
    text = []
    if job.runtime == Runtime.OPENAI_API.value:
        events = openai_evidence(data)
        for item in data.get("output") or []:
            if item.get("type") == "message":
                text.extend(
                    c.get("text", "")
                    for c in item.get("content", [])
                    if c.get("type") == "output_text"
                )
    else:
        sources: list[WebSource] = []
        citations: list[WebCitation] = []
        searches = 0
        presentation = {}
        for item in data.get("outputs") or []:
            if item.get("type") == "google_search_result":
                presentation["search_suggestions"] = item.get("search_suggestions")
            if item.get("type") == "google_search_call":
                searches += len(item.get("queries", []))
            if item.get("type") == "text":
                text.append(item.get("text", ""))
                for citation in item.get("annotations", []):
                    if citation.get("type") == "url_citation" and citation.get("url"):
                        citations.append(WebCitation(citation["url"], citation.get("title")))
                        sources.append(WebSource(citation["url"], citation.get("title")))
        events = [
            WebEvidenceEvent(tuple(sources), tuple(citations), searches, presentation=presentation)
        ]
    evidence = [e for e in events if isinstance(e, WebEvidenceEvent)]
    sources = tuple(dict.fromkeys(s for e in evidence for s in e.sources or ()))
    citations = tuple(dict.fromkeys(c for e in evidence for c in e.citations or ()))
    reason = data.get("error") or data.get("incomplete_details")
    if status == "completed":
        options = WebOptions.coerce(job.web)
        if not any(text):
            status, reason = "failed", "completed research job returned no text report"
        if options.require_search and not any(e.searches for e in evidence):
            status, reason = "failed", "no completed native search was reported"
        if options.require_citations and not citations:
            status, reason = "failed", "no native answer citations were reported"
    result = (
        ResearchResult(
            "".join(text),
            sources,
            citations,
            {key: value for e in evidence for key, value in e.presentation.items()},
        )
        if text or sources or citations
        else None
    )
    return ResearchSnapshot(
        job,
        status,
        vendor_status,
        receipt,
        result,
        str(reason) if reason else None,
        data.get("usage") or {},
    )
