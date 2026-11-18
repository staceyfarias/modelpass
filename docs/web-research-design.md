# Web search and research in ModelPass

Implementation status as of 2026-11-17. The stateless web interface and hosted
OpenAI/Gemini job interfaces are implemented with offline tests. The requested
subscription and Anthropic API live checks encountered environment restrictions.
OpenAI API and Gemini remain live-untested. The implemented contract and exact
verification status are in [api-and-runtimes.md](api-and-runtimes.md#native-web-search-and-hosted-research-2026-11-17).
The original proposal below records design rationale and candidate extensions.

Design proposal based on vendor documentation and SDK surfaces reviewed on
2026-11-17. The table describes vendor capabilities and implementation
candidates. ModelPass's supported capabilities remain in
[the runtime contract](api-and-runtimes.md). No allowance-spending probes were
run for this design.

## Vendor surfaces

| ModelPass runtime | Web search in a turn | Longer research and output |
| --- | --- | --- |
| `anthropic-sdk` | Claude Agent SDK documents built-in `WebSearch` and `WebFetch`. Search accepts a query and domain allow/block lists as **tool-call inputs**. Fetch reads a URL for a prompt. ModelPass currently denies both in chat. | An ordinary agent turn can plan, search, fetch and synthesize. The SDK documents search results with URLs and titles, and fetch results with a URL and text. The SDK docs reviewed did not expose a guaranteed claim-to-source citation format or Claude Research as a job. Claude Research is documented in the consumer web, desktop and mobile clients. [Python SDK](https://code.claude.com/docs/en/agent-sdk/python#websearch) · [Claude Research](https://support.claude.com/en/articles/11088861-use-research-on-claude) · [Subscription usage](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan) · [Third-party use](https://code.claude.com/docs/en/legal-and-compliance) |
| `openai-sdk` | The Codex TypeScript SDK documents `webSearchMode` for disabled, cached and live. ModelPass already drives the same runtime directly through app-server or exec. Codex config also documents indexed access, domain allowlists, search context size and approximate location. ModelPass currently sends a disabled override on its chat toolbelt path. Verify requested controls on each transport. | App-server documents `webSearch` items with search, open-page and find-in-page actions. The documented item gives query or action data. A complete consulted-source list and typed answer citations need their own evidence. A Codex turn can do multi-step research. ChatGPT's consumer Deep Research is a separate product capability. [SDK](https://learn.chatgpt.com/docs/codex-sdk) · [TypeScript options](https://github.com/openai/codex/blob/main/sdk/typescript/src/threadOptions.ts) · [Codex config](https://learn.chatgpt.com/docs/config-file/config-reference) · [App-server](https://learn.chatgpt.com/docs/app-server) |
| `anthropic-api` | Messages offers versioned server-side `web_search` and `web_fetch`. Search has per-request `max_uses`, an allowlist **or** a blocklist, and location. Fetch has its own domain controls and use limit. It can read eligible URLs already seen in supported conversation inputs or results, with opt-in fetch citations. Some versions support dynamic filtering or cache bypass for fetch. | Messages can do multiple search and fetch steps and return citation and tool result blocks. Handle server-tool errors inside the response and the `pause_turn` continuation explicitly. Claude Managed Agents has hosted sessions with web tools through a separate API and session model. [Search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) · [Fetch](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool) · [Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) |
| `openai-api` | Responses `web_search` documents live/cache-only access, domain allow/block filters, context size, approximate location, inline URL annotations and optional consulted sources. The guide documents `blocked_domains` even where the generated SDK types omit it. Page opening and finding are actions of the search tool where available. | Use a current reasoning model with web search. The current guide recommends `gpt-6-astra` with higher reasoning for longer reports, optionally in Responses background mode. Both legacy Deep Research model families retired on July 23, 2026. Responses can poll and cancel background jobs. [Search](https://developers.openai.com/api/docs/guides/tools-web-search) · [Responses reference](https://developers.openai.com/api/reference/resources/responses/methods/create) · [Background mode](https://developers.openai.com/api/docs/guides/background) · [Deprecations](https://developers.openai.com/api/docs/deprecations) |
| `google-api` | ModelPass's existing GenerateContent transport has `google_search` grounding and a separate `url_context` tool for supplied URLs. Grounding metadata includes queries, source chunks, citation supports and search presentation metadata. The newer Interactions API returns citations as text annotations. The Gemini Developer API does not inherit Vertex or Gemini Enterprise domain exclusions. | Interactions provides the preview agents `deep-research-preview-04-2026` and `deep-research-max-preview-04-2026`. Requires `background=True`, with polling, streaming and an interaction cancel endpoint. Search, URL Context and Code Execution are defaults. The agent accepts explicit tool selection and supports collaborative planning. [GenerateContent grounding](https://ai.google.dev/gemini-api/docs/generate-content/google-search) · [URL Context](https://ai.google.dev/gemini-api/docs/url-context) · [SDK search type](https://googleapis.github.io/js-genai/release_docs/interfaces/types.GoogleSearch.html) · [Deep Research](https://ai.google.dev/gemini-api/docs/deep-research) · [Interactions reference](https://ai.google.dev/api/interactions-api) |
| `openai-compatible` | No portable web tool follows from OpenAI-shaped chat completions. | Refuse in the initial contract. Endpoint-specific web features would need their own evidence and contract. |

OpenAI also documents an open-source **Sign in with ChatGPT** preview for
eligible direct Responses requests. Web search is subject to model and
workspace policy. The preview excludes `background` and `max_tool_calls`.
This is a separate authentication path worth evaluating independently. It
should use its published OAuth flow and its own connection semantics.
Subscription-linked API credits from either vendor still use the API-key
path and its billing rules. [OpenAI sign-in](https://developers.openai.com/siwc/quickstart)
· [Preview limits](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
· [OpenAI API credits](https://developers.openai.com/api/docs/guides/chatgpt-api-credits)
· [Claude subscription and API credits](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)

## A web-enabled turn

Add an opt-in `web=WebOptions(...)` keyword to the stateless
`Bridge.ask/chat/aask/achat` family. Keep the model's answer, receipt and
event stream. These provider surfaces let a model decide what to search and
how to use it. Callers receive a grounded-answer interface, with no promise of
a fixed query, ranking, top-k list or complete set of raw hits.

```python
from modelpass import WebOptions

answer = bridge.ask(
    "my-openai-api",
    "Compare the current public SDK docs for the two runtimes.",
    web=WebOptions(
        search="live",
        search_allowed_domains=("developers.openai.com", "code.claude.com"),
        require_citations=True,
    ),
    allow_failover=False,
)
for citation in answer.citations:
    print(citation.url, citation.title)
```

| Proposed option | Contract |
| --- | --- |
| `search="off" \| "provider_default" \| "live" \| "cached"` | `web=None` preserves current behavior. An explicit `WebOptions` defaults to `provider_default`, enabling the runtime's normal search path and reporting the effective mode if known. `off` supports fetch-only requests where the runtime can isolate fetch. `live` allows external access without guaranteeing that each response bypasses a cache. `cached` requires a no-live-access guarantee. Codex's indexed mode can be considered separately. |
| `fetch_urls: tuple[str, ...]` | Explicit public HTTP(S) URLs offered to a supported native URL-reading tool. An empty tuple requests no extra URL tool; search can still open results when intrinsic to the runtime. URL input errors fail before execution. URLs that are inaccessible or unused do not become successful fetches. Report per-URL results only where the vendor provides them. |
| `search_allowed_domains`, `search_blocked_domains` | Search-result restrictions only, using bare hostnames. Validate invalid combinations and document subdomain semantics. Treat model-generated arguments as unenforced until the adapter can impose the constraint on every covered call. A guarantee covering fetches, redirects or other networking would require its own contract. |
| `require_search: bool = False` | A success condition checked against observed successful native search evidence. Tool availability and instructions alone cannot satisfy it. Reject it with `search="off"` and refuse before running if the adapter cannot observe the outcome. When a supported run skips search or the search fails, use the existing error terminal and collected-call exception behavior. |
| `require_citations: bool = False` | A success condition requiring at least one answer citation backed by native web evidence. Refuse on an adapter without a verified mechanism. A source list or a model-written Markdown URL alone cannot satisfy it. |

Start with those options. Add approximate location, search context size,
freshness filters and call limits when there is demand and a verifiable wire
mapping. Anthropic's search `max_uses` counts searches per Messages request.
OpenAI's `max_tool_calls` counts built-in tool calls for a response across
tools. Gemini's Python SDK has an API-specific date-range field whose model
and endpoint compatibility still needs verification. A local token guard or
timeout cannot guarantee a cap on remote search charges.
[Anthropic search options](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
· [OpenAI tool limit](https://developers.openai.com/api/reference/resources/responses/methods/create)
· [Google search options](https://googleapis.github.io/js-genai/release_docs/interfaces/types.GoogleSearch.html)

### Capabilities and results

- Resolve web support against the endpoint, transport, model and installed
  SDK/CLI where known. Expose that evidence through the capability registry and
  show the requested and effective settings on the receipt. Reject known
  unsupported requests and combinations with `CapabilityNotSupported` before
  execution. Account policy or tool failures discovered during a run use the
  existing terminal contract. An explicitly configured failover target must
  satisfy the same requested features before any run.
- Keep web tools outside `tools=`, which declares caller-provided functions.
  Preserve other built-in tool denials and the account's sandbox settings.
  Wire only verified combinations with custom tools, MCP and schemas. Start
  with stateless calls. Session support needs lifecycle evidence, especially
  for Codex's process or thread configuration and resumed sessions.
- Add proposed `WebActivityEvent` for observed search/open/fetch activity and
  `WebEvidenceEvent` for structured sources and citations after the relevant
  text is known. Fold the latter into `Answer.sources` and
  `Answer.citations`, using optional fields with backward-compatible defaults.
  A source records a vendor-returned URL and title when present. A citation
  points to a source and may carry a half-open Python string span in
  `Answer.text`. Convert only confirmed native offsets, including multipart
  output and Unicode. Otherwise retain the citation without an invented span.
- Distinguish unavailable metadata from a supported field that arrived empty.
  Preserve vendor links and attribution metadata, including Gemini search
  suggestions where returned, so consumers can meet the provider's
  [display terms](https://ai.google.dev/gemini-api/terms). Do not resolve Google
  redirect URLs or scrape pages locally to fill gaps. Keep queries and source
  URLs out of new run-log fields by default. Record tool usage separately
  from token usage when the provider supplies billable counts.

For the first implementation, refuse `cached` outside Codex and Responses.
Refuse strict search-domain restrictions on Gemini Developer API, and on
Claude Agent SDK until a ModelPass-controlled mechanism can enforce them.
Treat Responses' and Codex's model-driven page-opening actions as web
activity. Advertise `fetch_urls` on those runtimes only if a supplied URL
can be handed to a documented native surface with observable semantics.

## Hosted research jobs

An ordinary web-enabled turn can already take a research brief and produce a
report. Add a hosted-job interface when a caller needs to leave and come back
to a run that the provider continues executing. Start with Gemini Interactions
and OpenAI Responses background mode, subject to the model and storage policy.
Treat Anthropic Managed Agents as a later integration with its own configuration.

```python
job = bridge.start_research(
    "my-gemini-api",
    "Compare current royalty policies at three music distributors.",
    agent="deep-research-preview-04-2026",
)
snapshot = bridge.get_research(job)
# The app can persist job and poll again later.
if snapshot.status == "completed":
    print(snapshot.result.text)
# bridge.cancel_research(job) explicitly requests remote cancellation.
```

- Make `ResearchJob` a serializable handle with a provider job ID, backend
  and stable connection identity. Keep credentials out of the handle.
  Retrieval and cancellation use the original connection and refuse a
  changed endpoint or a mismatched identity where detectable. A job cannot
  fail over. If submission times out before an ID is received, report an
  unknown submission outcome so callers can avoid accidental duplicate jobs.
- Reuse `WebOptions` where the job backend supports them. Require explicit
  backend-specific model or agent selection in the first version. Gemini
  should request only search and URL Context for the initial web-only job,
  subject to proving the explicit selection removes the default code tool.
  The receipt identifies the API, model or agent, and storage requirements.
  A local token guard cannot enforce an in-flight budget on a detached job.
  Reject a requested guard unless the backend can honor the same limit.
- A poll returns a snapshot with a normalized status, available result/evidence
  and the vendor status. Final research output shares the text and citation
  semantics of `Answer`. Preserve incomplete, failed and cancelled statuses,
  with partial evidence distinguishable from a completed report. Stopping a
  poll or reaching its timeout leaves the job running. Cancellation is a
  separate provider request. Report provider retention limits and never promise
  the job remains retrievable indefinitely.
- Defer job streaming and collaborative plan approval until the submit,
  retrieve and cancel contracts are established. A provider may emit progress
  summaries, but status polling does not imply a meaningful completion percentage.

## Implementation sequence

1. Add the option validation, receipt, event and collection paths, and use
   OpenAI Responses and Anthropic Messages as the first adapter slices. Cover
   citations, sources and server-tool errors with offline wire fixtures.
2. Wire Gemini GenerateContent grounding and the two subscription runtimes.
   Read the exact installed SDK and protocol surfaces. Test opt-in isolation,
   policy restrictions, transport differences and cases where a search was
   skipped or returned no usable evidence.
3. Implement the two hosted job backends, including terminal-state and
   cancellation mapping and timeout-before-ID tests. Reuse selected connection
   auth and transport behavior. Add dated contract entries as each capability
   earns a test and SDK-surface evidence, with allowance-spending probes under
   the existing owner-run live test marker.
