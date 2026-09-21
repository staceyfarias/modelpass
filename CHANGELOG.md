# Changelog

Notable changes to modelpass (called `subpass` through 0.1.1). Dates are the day the
work landed.

## Unreleased

### Added

- **A connection can state its input window, and modelpass will never guess one**
  (2026-09-21). The optional config key `maxInputTokens` records the model's usable
  input window in tokens, read back as `Connection.max_input_tokens` (`int | None`).
  It exists because a consumer needs to decide whether a payload will fit *before*
  spending a call finding out, and the per-connection half was the part it could not
  express anywhere.

  **Nothing in modelpass reads it** — it bounds no call, truncates no message and
  refuses no run, in the same way `retry = "never"` is a stance rather than a retry
  loop. **And there is no default**: no table of model names to window sizes, now or
  later, because a vendor moves a window without moving the model string and a stale
  number that reads as authoritative is worse than no number. Omitted means
  *unknown*, which is what every existing connection means; unknown and zero are
  different facts, so `maxInputTokens = 0` is refused at construction rather than
  stored, as is any negative, fractional, boolean or non-numeric value.

  `modelpass list --verbose` prints the window and prints `unknown` when there is
  none, and the bench's accounts page says the same; the bench's edit form does not
  offer the field and now carries it through a save rather than dropping it. An
  additive config key: the file stays at version 1, and a connection written before
  the key existed loads unchanged and is not given one when the file is rewritten.

- **Connection groups, and a group is something a run can be addressed to**
  (2026-09-17). A connection may declare `groups = ["fast", "cheap"]`, and any entry
  point that takes a connection name takes `"group:<name>"` in its place —
  `bridge.ask("group:cheap", ...)`, `chat(connection="group:cheap")`, the session
  constructors, and an application's own `connection = "group:cheap"` config. The
  resolved connection is what the receipt, the terminal event and the run-log line
  name: a group is how a call was *addressed*, and what it was billed to is always a
  connection.

  `default` is **the name for the connections that declare no group**, not a universal
  membership: one in `fast` is not also in `default`. A group is derived from
  membership and never stored — there is no `[groups]` table, so a group exists while
  somebody is in it and moving the last member out is how it goes away. Resolution is
  name order among the enabled members, and because that is a rule nobody chose,
  `bridge.select(group)` returns the chosen connection *and* every member it walked
  past with the reason, which `modelpass groups` and the bench's Groups page both
  print. Preference order *within* a group is deliberately absent and is on the
  roadmap with throttling and metrics.

  New: `Connection.groups` / `.group_names` / `.in_group()` / `.with_groups()`,
  `Group`, `GroupSelection`, `DEFAULT_GROUP`, `groups_of()`, `Bridge.groups()` /
  `.group()` / `.select()`, `find(group=, enabled=)`,
  `bridge.manage.set_groups()` and `GroupsResult`, the errors `NoSuchGroup` and
  `GroupUnavailable`, the commands `modelpass groups` / `set-groups` /
  `list --group` / `connect --group`, and the bench's Groups page, group filter and
  groups field. `groups` is an additive config key: the file stays at version 1, and a
  build that has never heard of it preserves it verbatim and reports it.

- **`onQuotaExhausted.failover` refuses a `group:` target**, and is the one field that
  does. It is the only path in modelpass onto metered billing, and under D4(d) naming
  the target is the consent; a group names a set whose membership changes after that
  consent was given. Refused at construction rather than at quota exhaustion, so it is
  caught where the file is written. Previously this raised `NoSuchConnection` naming
  every configured connection, which said nothing about why a group was wrong there.

- **`modelpass enable` and `modelpass disable`.** `enabled = false` has existed since
  2026-08-17 and was reachable from the library and the bench, but a terminal user had
  to edit the file by hand. `disable` also says when it has just left a group with no
  enabled member, and `set-groups` says when a group has just lost its last member, so
  a consequence beyond the connection named is reported rather than discovered from a
  later refusal.

### Changed

- **`bridge.find()` takes `group=` and `enabled=`.** `enabled` is not applied unless
  asked for: `find` is how a caller inspects its store, and silently omitting a
  connection somebody switched off would make it look deleted.
- **`modelpass list` prints each connection's groups**, not behind `-v`: a group is
  somewhere a run can be sent, and a listing that hid it would let a user believe
  `group:cheap` reaches a connection that is not in `cheap`.
- **A plain link in the bench body is now themed.** There was no CSS rule for one at
  all, so it fell back to the browser's blue, which is unreadable on the dark theme.
  This affects the few body links that already existed on the accounts and playground
  pages.

## 0.2.1 — 2026-09-13

### Fixed

- **A vendor failure's retry verdict now survives `ask()`.** An adapter meeting an HTTP
  429 ends the stream with a terminal event carrying `retryable = yes`, the vendor's
  `retry_after` and `status_code = 429`; the exception `ask()` and `aask()` raised in
  that terminal's place carried none of the three, so a consumer classifying what they
  actually hold read `unknown` and a rate limit was never retried (found by a RAG
  evaluation harness's migration, 2026-09-13). Every exception raised for a non-ok terminal —
  `VendorRunFailed`, `GuardStop`, `QuotaExhausted`, `RunTimedOut`, and the
  `AdapterFailed` a schema-bound run raises when nothing structured came back — now
  carries `.retryable`, `.retry_after`, `.status_code` and the terminal event itself on
  `.terminal`, as do the errors `chat(raise_on_stop=True)` raises and the LangChain
  leaf's `SubpassRunError` family. `classify_error()` reads that carried verdict first,
  so it never answers `unknown` for a failure the terminal had already classified.
  Nothing was renamed and no class moved; every new name is an addition.

### Changed

- **A `first_token` timeout on a *session turn* now classifies as `unknown`, where it
  read `yes` in 0.2.0.** This follows from the fix above and is a deliberate
  correction, not a side effect. `classify_error()` used to derive the verdict from the
  exception's class and its `which` field, and an exception cannot say whether the run
  it ended left anything behind; the terminal it now carries can. A session turn holds
  a conversation on the runtime, so repeating it is not repeating nothing, and
  `unknown` is the honest answer — the same answer `chat()` and the terminal event
  itself already gave in 0.2.0 for that case. A `first_token` timeout on a **stateless**
  `ask()` or `chat()` still reads `yes`, unchanged. A caller whose loop retried on `yes`
  will now decline that one case; a caller who retries on `yes` *or* `unknown` sees no
  difference. Pinned by
  `test_a_first_token_timeout_on_a_session_turn_reads_unknown`.

## 0.2.0 — 2026-09-13

**subpass became modelpass, and the scope moved with the name.** 0.1.x was a way to
reach the AI subscriptions you already pay for. 0.2.0 is the shared LLM connectivity
library the rest of your programs can stand on: the two agent runtimes it always drove
(`anthropic-sdk`, `openai-sdk`) now sit beside four API-key runtimes with adapters of
their own — `anthropic-api`, `openai-api`, `google-api`, and `openai-compatible` for
anything OpenAI-shaped, a local Ollama included. One native interface reaches all six,
sync and async, with the same arguments, the same events, the same guards and the same
run log on both faces. Explicit keys live in a second file that is not the shareable
one, and a Settings page can write a connection through the same validation and the
same receipt `modelpass connect` gives a terminal.

**What holds it together is that the receipt tells the truth.** Before a run it says
which auth mode is about to pay, which sampling fields will actually be sent and which
are being dropped and why — per *model*, not just per runtime — whether the
prompt-cache breakpoints you marked reached the vendor or were flattened away, and what
bounds the call; after one, every terminal event and every error carries a typed
retryability verdict computed from status codes and exception classes and never from
the text of a message. Nothing here guesses: a capability cell still moves only with a
recorded drive, and `unverified` still means nobody checked. The `subpass` import
package, the `subpass` console script, `SUBPASS_HOME` and the `~/.subpass/` fallback
all ship alongside and keep working until 0.3.0, so no consumer has to move on this
release's schedule.

### Contract names

Every public name this release adds, in one place, because names are contract (R9).
Nothing was renamed and nothing was removed: `SubpassError` and its subclasses,
`ChatSubpass` and the `subpass_*` `response_metadata` keys all keep their spellings.

**Exported from `modelpass`** (31 new names in `__all__`):

| Group | Names |
| --- | --- |
| Async and timeouts (1.12, 1.13) | `Timeout`, `Retryable`, `RetryVerdict`, `RunTimedOut`, `SessionBusy`, `classify_terminal`, `classify_error` |
| The collected door (1.8) | `Answer` |
| Sampling (1.7) | `Sampling`, `SamplingPlan`, `SamplingRules`, `plan_sampling`, `rules_for` |
| Content blocks (1.5) | `TextBlock`, `ContentBlock`, `CacheControl` |
| Connection management (1.4) | `ConnectionManager`, `ConnectionPlan`, `Credential`, `AddResult`, `RemoveResult`, `RenameResult`, `EnabledResult` |
| Secrets (1.3) | `SecretStore`, `SecretPermissions`, `NoSuchSecret`, `SecretStillReferenced` |
| The per-install drive (1.10) | `VerifyReport`, `VerifiedCapabilities`, `VERIFY_CELLS`, `StructuredOutputRejected` |

**New members of existing enums**: `Capability.SAMPLING_CONTROLS`,
`Capability.MAX_OUTPUT_TOKENS`, `Capability.CACHE_BREAKPOINTS`;
`Runtime.ANTHROPIC_API`, `Runtime.OPENAI_API`, `Runtime.GOOGLE_API`,
`Runtime.OPENAI_COMPATIBLE`; `TerminalStatus.TIMED_OUT`; `CredentialKind.SECRET`,
`CredentialKind.NONE`.

**New methods and attributes**: `Bridge.ask`, `Bridge.achat`, `Bridge.aask`,
`Bridge.manage`, `Bridge.secrets`, `Bridge.registry_for`; `Session.asend` and
`aclose()` on the async iterator; `Adapter.arun`, `Adapter.probe`,
`Adapter.cached_probe`, `Adapter.bind_resolution`; `Message.flat_text`,
`Message.content` as blocks, `RunRequest.system_blocks`,
`RunRequest.conversation_blocks`, `RunRequest.effective_sampling`;
`ConnectionManager.plan_connection` / `.add_connection` / `.remove_connection` /
`.rename_connection` / `.set_enabled`; `Credential.native_login()` / `.env(NAME)` /
`.secret(value)` / `.stored_secret(entry)`; `modelpass.store.default_home()`.

**New receipt and run-log fields**: `sampling_requested`, `sampling_applied`,
`sampling_notes`, `cache_breakpoints_requested`, `cache_breakpoints_honoured`,
`retryable`, `retry_after`. All additive: a run-log line written before them reads
back empty or zero.

**New modules**: `modelpass.manage`, `modelpass.secrets`, `modelpass.retry`,
`modelpass.sampling_rules`, and the four adapters
`modelpass.adapters.anthropic_api`, `.openai_api`, `.google_api`,
`.openai_compatible`. `modelpass.testing.StallingAdapter` joins the testing leaf.

**On the LangChain leaf**: `SubpassTimeoutError`, a `SubpassRunError` subclass, so a
consumer matching these classes across the MRO keeps working.

**New store keys**, all additive under the 0.1.1 compatibility policy: `baseUrl`,
`timeoutSeconds`, `retry`, `verifiedCapabilities`, and `credentialRef = "none"` on
`openai-compatible`. Plus a second file, `~/.modelpass/secrets.toml`, with its own
`version` and `[secrets.<entry>]` tables.

**New command line**: `modelpass rename`, `modelpass remove`, `modelpass secrets`, and
on `connect` the flags `--api-key-stdin`, `--runtime NAME` and `--base-url URL`.
`modelpass verify` gained its second form, the capability drive.

### Added

- **`subpass` as a shim package.** `import subpass` still works and emits exactly one
  `DeprecationWarning`. It is not a copy: a meta-path finder maps every `subpass.X` onto
  the module object `modelpass.X` already is, so `subpass.Bridge is modelpass.Bridge`,
  `from subpass.langchain_adapter import ChatSubpass` yields the same class, and
  `importlib.import_module("subpass.langchain_adapter")` returns the same module. That
  identity is what keeps `isinstance` checks and `except` clauses working while half a
  program has migrated and half has not. Covered by `tests/test_shim.py`.

- `subpass` stays as a console-script alias for `modelpass.cli:main`. Help output names
  whichever of the two you actually invoked.

- **Four API-key runtimes: `anthropic-api`, `openai-api`, `google-api`,
  `openai-compatible`.** Plumbed, adapter pending — you can write and inspect a
  connection today, and you cannot run one: every capability that is not a
  checked absence reads `unverified`, `chat` included, so `bridge.chat()` refuses
  with a capability error naming the cell. A cell moves only with a recorded
  drive, and the drives are the adapter tickets.
  - **`google-api` ships ungated** while `google-cli` and `google-sdk` stay in
    `EXPERIMENTAL_RUNTIMES`. The asymmetry is the decision, not an oversight: the
    Antigravity terms govern the *subscription*, a Gemini API key falls under the
    Google Cloud terms instead, and Google's own SDK cannot reach the
    subscription at all. Recorded in `docs/legal/google.md` and in the
    capability note.
  - **`baseUrl` on a connection** (wire key `baseUrl`). Required on
    `openai-compatible`, optional on the other three API runtimes, refused on the
    agent runtimes. Must be `https`, or `http` to `localhost` / `127.0.0.1` /
    `::1`; may carry no credentials, query string or fragment. Older builds carry
    it through untouched under the 0.1.1 store policy.
  - **The API preflight is three cheap checks** — the credential resolves to a
    non-empty value, the base URL parses and is safe, and optionally a cached
    token-free model-list probe (`Adapter.probe()` / `cached_probe()`, dropped by
    `Bridge.refresh_identity`). Receipts on these runtimes report
    `detected_auth_mode = api_key`, no `plan_name`, no `binary`,
    `runtime_available` meaning "the vendor SDK imports", and an `account` that
    may carry a `sha256:` key fingerprint and never the key.

- `Capability.SAMPLING_CONTROLS` and `Capability.MAX_OUTPUT_TOKENS` (R5), added
  `unverified` on every runtime here and filled by ticket 1.7 once the request
  fields and the applied-vs-requested report existed to fill them with.

- **Explicit API keys, in a second file** (R14). `~/.modelpass/secrets.toml`, beside
  `connections.toml` and never inside it: the connection file stays the one that is safe
  to share, and a connection references a key as `credentialRef = "secret:<entry>"`.
  D2 is unchanged and the design decision says why — a key in that file does nothing at
  all until a connection names it, which is the property the environment case already
  had.
  - **`modelpass connect <vendor> --api-key-stdin`** reads the key from standard input.
    There is no option anywhere in the command line that takes a key as a *value*, and a
    test asserts that over the whole parser: argv is visible in process listings and
    lands in shell history. It refuses a terminal and says how to pipe instead, strips
    exactly one trailing newline, and refuses an empty value. The flag selects the
    vendor's API runtime, which is the runtime that bills a key you hand over.
  - **`secret:` is refused on the agent runtimes.** They launch a child process, and the
    only way a child gets a credential is an environment variable — which `env:NAME`
    already is. Accepting it there would write a connection whose child launches with no
    key and falls back to the vendor login, billing an account nobody named.
  - **Permissions are reported, never claimed.** POSIX `0600` is asserted by a test
    rather than assumed. Windows gets a best-effort
    `icacls <path> /inheritance:r /grant:r "<you>":F` once at creation whose real
    outcome is recorded in the file; `modelpass check` prints either "restricted to your
    account" or "could not restrict: `<reason>`; the profile directory's own ACL is the
    floor". A widened POSIX mode is reported with the `chmod` that fixes it, and never
    refused.
  - **Delete and rename rules**, so the two files cannot diverge quietly. The secret is
    written first and the connection second when adding, the connection first and the
    secret second when deleting — a crash between them leaves an inert orphan rather
    than a connection pointing at nothing. Deleting a connection removes its entry only
    when the entry is its own name and nothing else references it; deleting a referenced
    entry is refused, naming the connections. New **`modelpass rename OLD NEW`** moves
    both under the same rule and says so when it moves only one.
  - **`modelpass check` lists orphan entries** — entries no connection references. They
    are inert, they are reported, and they are never collected for you.
  - `SecretStore`, `SecretPermissions` and `NoSuchSecret` / `SecretStillReferenced` are
    public. The secrets file follows the 0.1.1 compatibility policy: any version reads,
    a newer version refuses writes, unknown keys are carried verbatim, and an unreadable
    file reads as "no secrets" while refusing writes.
  - No secret value reaches a receipt, the run log, a `to_dict()` or a vendor event; the
    `sha256:` fingerprint does. `tests/test_no_ambient_credentials.py` and
    `tests/test_serializable_surface.py` walk the whole serializable surface for a value
    that really is in the file, and the bench's connection view is checked too.

- **Connection management from Python: `bridge.manage`** (R10, and the one thing
  a desktop agent app's validation report called unanswered). `ConnectionManager` in the new
  `modelpass.manage` gives a Settings page what `modelpass connect` gives a
  terminal: `plan_connection(...)` builds the connection and returns the
  **pre-write receipt** without writing anything — a pasted key is held in memory
  as a `PendingSecret` so the preflight describes the shape that is about to be
  saved — and `add_connection(plan)` writes the secret and then the connection, in
  that order and for that reason. Plus `remove_connection(name)` (the R14 delete
  rules, returning what it removed and what it kept and why),
  `rename_connection(old, new)` and `set_enabled(name, bool)`. Credentials are
  stated as `Credential.native_login()` / `.env(NAME)` / `.secret(value)` /
  `.stored_secret(entry)`, so the auth mode is derived rather than passed
  alongside something that could disagree with it. Every result is a frozen
  dataclass with a `to_dict()` that is safe to print; no pasted value appears in
  any of them or in any `repr`, `PendingSecret` included. `modelpass connect`,
  `modelpass rename` and the bench's delete and enable routes are now callers of
  this surface, with their output unchanged.

- **Structured message content, so a prompt-cache breakpoint survives the trip**
  (R3). `Message.content` now takes either a `str` or a tuple of `TextBlock`,
  where a block may carry `CacheControl(type="ephemeral", ttl=None | "5m" |
  "1h")`. `system_prompt=` and `message=` on `Bridge.chat` take the same two
  shapes, and `Message.text` concatenates the blocks so anything that read
  `.content` as a string keeps working. A plain string is unchanged in every
  respect — same bytes, same rendering, same receipt.
  - **The failure this ends.** Three consumers build Anthropic `cache_control`
    prompts and all three lost them at the library boundary — a downstream agent
    host composes four breakpoints and strips them before the call because there
    was nowhere to put them, and a retrieval service and a batch scraping tool
    lose the same thing the same way. No error, no warning, a full-price write on every call.
  - **New capability cell `cache_breakpoints`**, distinct from `ttl_control`:
    one asks how long a cached prefix lives, the other whether you get to say
    where the prefix *ends*. `anthropic-sdk` and `openai-sdk` are both
    `unsupported`, each with a dated note naming what it was read off — the
    Anthropic one off the installed claude_agent_sdk 0.2.148, whose
    `system_prompt` is `str | SystemPromptPreset | SystemPromptFile | None` with
    no block list anywhere. Neither is "no caching": both runtimes cache, and
    `ttl_control` on `anthropic-sdk` stays `supported`. `anthropic-api` is
    `unverified` and moves with its adapter's drive; the other three API runtimes
    are `unsupported` with automatic vendor-side caching.
  - **A dropped breakpoint is now reported, not performed quietly.** The receipt
    that precedes the run carries `cache breakpoints were requested and dropped:
    <runtime> does not accept them`, plus `cache_breakpoints_requested` and
    `cache_breakpoints_honoured`; the same pair lands on each run-log record
    (additive keys — an older line reads back as `0`/`0`). `CacheEligibility`
    reports the fact rather than its floor heuristic when breakpoints are
    present, and keeps the heuristic for plain-string prompts.
  - **The LangChain leaf stops flattening.** `ChatSubpass` converts LangChain
    content lists into `TextBlock`s, and a leading block-bearing `SystemMessage`
    is hoisted into `system_prompt=` — the front of the cached prefix, which is
    the only position where a breakpoint means anything. A *string* system
    message still rides `history` exactly as before, so no existing caller's
    bytes move. This is the path the agent host and the retrieval service are on
    today.
  - `RunRequest.system_blocks` and `RunRequest.conversation_blocks` are the
    fields the `anthropic-api` adapter will pass straight into
    `messages.create` with `cache_control` intact.

- **The `anthropic-api` runtime can now run.** The first API-key adapter
  (`modelpass.adapters.anthropic_api.AnthropicAPIAdapter`, extra
  `anthropic-api`, written against `anthropic` 0.97.0): streaming text and
  thinking, per-response usage with the cache split mapped exactly as the agent
  runtime maps it, an in-adapter model↔tool loop, native structured output, and
  `cache_control` breakpoints that reach the vendor with their markers intact.
  `pip install "modelpass[anthropic-api]"`.
  - **Your functions still run in your process, and the loop still belongs to
    the runtime — it is just that here the runtime is the adapter.** `tools=`
    means the same thing it means on the other five runtimes: `tool_call` and
    `tool_result` are observations, there is no reply channel, and there is **no
    turn cap** — `stopAtTokens` is the bound. A handler that raises comes back
    to the model as a failed tool result instead of killing the run.
  - **A token guard is a real circuit breaker here.** Every response carries its
    own usage, so a `stopAtTokens` crossing stops the loop *between* rounds,
    before the next round's tools run.
  - **Sessions are refused**, with a message naming `bridge.chat(history=[...])`
    — an API endpoint holds no conversation, so there is nothing to resume.
  - **The key is the connection's, always.** The client is constructed with an
    explicit `api_key=` resolved at the moment of use; the SDK's own environment
    discovery never runs. `tests/test_no_ambient_credentials.py` sets a decoy
    `ANTHROPIC_API_KEY`, asserts it is not used, and asserts that the bare
    constructor really would have taken it.
  - **`max_tokens` is required by the Messages API**, so the adapter always
    sends one: `options={"max_output_tokens": N}`, else 4096. A stopgap until
    ticket 1.7 makes the ceiling a first-class request field.
  - Twelve capability cells on the `anthropic-api` row moved to `supported`,
    each with a dated note naming the test and the SDK version it was read off.
    `ttl_control` moved the other way — from `unsupported` to `unverified` —
    because the SDK types `cache_control.ttl` and the shared "no API has a TTL
    field" turned out to be false for this one. `midconversation_system`,
    `sampling_controls` and `max_output_tokens` stay `unverified`, and their
    notes say what would move them. See
    [docs/api-and-runtimes.md §2.0a](docs/api-and-runtimes.md).
  - New live file `tests/live/test_anthropic_api_live.py` (marker `live`,
    deselected by default) drives one chat, one tool round trip, one structured
    output and one cache breakpoint sent twice. It needs `SUBPASS_LIVE_TESTS=1`
    **and** `MODELPASS_LIVE_ANTHROPIC_API_KEY` — a variable of its own, so
    having a key on the machine is never enough on its own to start spending.

- **Sampling controls, with an honesty report** (R5, ticket 1.7). `Sampling` is a
  frozen request type carrying `temperature`, `top_p`, `top_k`,
  `max_output_tokens` and `reasoning_effort`, accepted by `Bridge.chat` and
  `Bridge.preflight` and validated at the constructor. Every receipt and every
  line of `runs.jsonl` gains `sampling_requested`, `sampling_applied` and
  `sampling_notes` — what you asked for, what is actually sent, and one sentence
  for each field that was dropped or coerced. The run-log keys are additive and
  a line written before them reads back as empty.

- **`modelpass/sampling_rules.py`: one table of the per-*model* rules.** Support
  varies by model and not only by runtime — GPT-5 accepts only
  `temperature=1.0`, `claude-sonnet-4-x` and `claude-haiku-4-x` accept one of
  `temperature`/`top_p`, the o-series accepts none, and Anthropic's
  adaptive-thinking families take no sampling parameters once an effort is
  asked for. Three shipped consumers had each grown a private copy of some of
  this; `rules_for(runtime, model)` is the one copy, every row carries a dated
  source note, and an unknown model gets the runtime's defaults plus a note
  saying so rather than a guessed family.

- **`anthropic-api` sends all five.** `temperature`/`top_p`/`top_k` as
  themselves, `max_output_tokens` as the API's required `max_tokens`, and
  `reasoning_effort` as `output_config={"effort": ...}` — read off `anthropic`
  0.97.0's own typed surface on 2026-09-13. Its `sampling_controls` and
  `max_output_tokens` capability cells move to `supported`; the two agent
  runtimes' move to `unsupported` as a checked absence; `google-cli`,
  `google-sdk` and the three pending API runtimes stay `unverified`.

- **`Bridge.ask(connection, message, ...) -> Answer`: one call, one answer**
  (R11, ticket 1.8). `chat()` drained, in the library, so a one-shot call site
  stops writing its own loop. `Answer` is a new frozen dataclass — exported as
  `modelpass.Answer` — carrying `text`, `structured` (the parsed schema-bound
  answer), `usage` (the final cumulative), `receipt`, `status`, `reason`,
  `tool_calls`, `events`, `sampling_applied` and `sampling_notes`. A terminal
  that is not `ok` **raises** the errors `chat()`'s consumers already handle —
  `GuardStop`, `QuotaExhausted`, `VendorRunFailed` — so `ask` never returns a
  half answer that gets indexed as a finished one. There is no `raise_on_stop=`
  keyword: it would be one whose `False` could not be honoured. `cancelled`
  cannot happen here, because nothing closes the iterator. With `schema=`,
  `structured` is never `None` on an `ok` run; an adapter that ends `ok` without
  the `structured_output` event raises `AdapterFailed` naming the missing event.
  `connection` and `message` may be positional, the one place in the API they
  may be. No new error class: everything `ask` raises is a name consumers
  already import.

- **The `openai-api` runtime has an adapter** (R4, ticket 1.9). `pip install
  "modelpass[openai-api]"` (pins `openai>=2,<3`; written against 2.32.0 — a
  different extra from `openai`, which installs `openai-codex` for the agent
  runtime). Streaming chat, an in-adapter tool loop with no turn cap, native
  structured output, per-round usage, a key-fingerprint preflight with an
  optional cached `models.list()` probe, and graceful cancellation between
  rounds — the same shape `anthropic-api` set in ticket 1.6, down to the test
  file's section order. The client is constructed with an explicit `api_key=`
  from the connection, so an ambient `OPENAI_API_KEY` can never bill an account
  nobody named. Eleven capability cells moved to `supported` on a fake-transport
  test plus the installed SDK's typed surface; `thinking` stayed `unverified`
  and names the live test that moves it.

- **It speaks the Responses API, not Chat Completions**, and the reason is the
  event vocabulary: reasoning text exists as a stream event only on Responses (Chat Completions reports a
  token count and no text), `status` + `incomplete_details.reason` maps onto the
  terminal vocabulary one-for-one, and `max_output_tokens` is the field's actual
  name there. There is no option to select the other transport; that shape is
  `openai-compatible`'s, in ticket 1.10.

- **Differences from `anthropic-api` that a consumer can see**, each deliberate:
  OpenAI's cached prefix is reported *inside* `input_tokens` and modelpass's
  convention keeps it beside, so the adapter subtracts and both runtimes land on
  the same `TokenUsage`; `cache_write_tokens` is always 0 and
  `reasoning_tokens` stays folded inside `output_tokens`; the system prompt is
  flattened into `instructions=` with `flat_text` semantics and its
  `cache_control` markers are dropped with the receipt naming the drop; a schema
  is sent `strict: true` after `schema.to_openai_strict()`, with
  `schema.openai_strict_issues()` reported as receipt notes before the call; and
  no default output ceiling is invented, because the field is optional here.

- **`options={"reasoning_summary": "auto"}`** asks the Responses API for the
  reasoning summary that becomes `thinking` events. modelpass does not ask on its
  own: a summary is billed output nobody requested.

- **`tests/live/test_openai_api_live.py`** — marker `live`, deselected by
  default, gated on `SUBPASS_LIVE_TESTS=1` and on `MODELPASS_LIVE_OPENAI_API_KEY`.
  Four calls: a short chat, a tool round trip, a structured output against a
  schema that is not already strict, and one reasoning-effort call whose green
  run is what moves the `thinking` cell.

- **`modelpass connect` takes `--runtime NAME` and `--base-url URL`** (ticket
  1.9b). `--runtime` names one runtime exactly and accepts only a runtime of the
  vendor already on the command line, with the error listing that vendor's own.
  It completes a cell that had no command line: `modelpass connect anthropic
  --runtime anthropic-api --api-key-env ANTHROPIC_API_KEY` writes a connection on
  the **API** runtime that reads a **named environment variable** — the library
  has planned that shape since ticket 1.4, but the CLI could only reach an API
  runtime by way of `--api-key-stdin`. Every default is unchanged: a bare
  `connect` is still the subscription, `--api-key-stdin` alone still selects the
  vendor's API runtime (ticket 1.3), and `--api-key-env` alone still means
  metered billing on the agent runtime, because a variable does reach a launched
  child. `--runtime anthropic-sdk --api-key-stdin` is refused with the same
  sentence `Connection` has always refused `secret:` on an agent runtime with.

- **`google` and `compatible` are vendors you can type.** `modelpass connect
  google` means `google-api` (its two agent runtimes are gated by D5 and this
  subcommand has no way to ungate them), and `modelpass connect compatible
  --base-url http://localhost:11434/v1` points a connection at any OpenAI-shaped
  endpoint. `--base-url` is validated by the connection's own rules, so plain
  `http` is accepted on loopback and nowhere else. Both take either credential
  form.

- **The `openai-compatible` runtime has an adapter, and it is the first runtime
  whose capability cells move per *install*** (R4, ticket 1.10). Same
  `openai-api` extra — it is the same `openai` client pointed at somebody else's
  server. Streaming chat over **Chat Completions**, an in-adapter tool loop with
  no turn cap assembled from streaming tool-call fragments, native structured
  output, usage through `stream_options.include_usage`, a `models.list()` probe,
  and graceful cancellation between rounds: the shape ticket 1.6 set, a third
  time.

- **Chat Completions by default, `options={"wire": "responses"}` per
  connection.** The *opposite* decision to ticket 1.9's, from the same evidence: `openai-api` asked which surface says
  more, and this runtime asks which surface the thing behind the base URL
  actually implements. Ollama, LM Studio, vLLM, llama.cpp, a LiteLLM proxy and
  OpenRouter all implement chat completions; Responses is a minority. The
  `wire` option runs ticket 1.9's code path **unchanged** — the adapter is a
  subclass — for a gateway that does implement it. `openai-api`'s behaviour and
  its tests are untouched: two helpers there gained a `runtime` argument that
  defaults to what it always was, and the no-model refusal became a class
  attribute.

- **`modelpass verify <connection>` drives an OpenAI-compatible endpoint and
  records what worked.** The per-install `refine()` the plan named, and the only
  command in modelpass that spends in order to learn something — it says so
  before it does. One token-free model listing, one short chat, one tool round
  trip and one structured-output call decide six cells (`chat`, `streaming`,
  `incremental_text`, `usage_tokens`, `tools_in_process`, `structured_output`),
  which are written to the connection under the additive store key
  `verifiedCapabilities` and folded into a registry **copy** by the new
  `Bridge.registry_for(connection)`. `STATIC_TABLE` is never touched: two
  installs of this runtime can legitimately disagree about every cell. An
  endpoint that cannot be reached writes nothing, because a record saying
  "driven, and it does nothing" about a box that was switched off is worse than
  no record. The subscription half of `verify` is unchanged.

- **Consequence, stated plainly: an unverified `openai-compatible` connection
  cannot chat.** `chat` reads `unverified`, the bridge refuses, and the refusal
  now carries the capability table's own note — which on this runtime names
  `modelpass verify`. That is the tri-state working rather than a gap.

- **`credentialRef = "none"`** — a connection that declares no credential at all,
  accepted on `openai-compatible` and refused by name on every other runtime. A
  local Ollama authenticates nobody, and saying so is more honest than pointing
  at an environment variable that will never be set; the receipt reports the two
  states differently. The client is still constructed explicitly, with a visible
  placeholder in the `api_key=` slot the SDK requires, so D2's in-process clause
  is untouched. `modelpass connect compatible` with no key flag writes this
  shape; `--api-key-env` and `--api-key-stdin` still work for a gateway.

- **`StructuredOutputRejected`**, a `CapabilityNotSupported` subclass raised when
  an endpoint refuses the `json_schema` response format — which many compatible
  servers do. It is a capability refusal that could not be made before the call,
  not a run failure, and its message says what to try. modelpass deliberately
  does **not** fall back to `response_format: {"type": "json_object"}`, which
  constrains the answer to be JSON without constraining it to be your schema.

- **Usage is `None` rather than zero where a server reports none.** A compatible
  server may ignore `include_usage`; the adapter then emits no usage event at all
  instead of one full of zeros, because "nobody told us" and "this run cost
  nothing" are different claims.

- **`tests/live/test_openai_compatible_live.py`** — marker `live`, deselected by
  default, gated on `SUBPASS_LIVE_TESTS=1` and on
  `MODELPASS_LIVE_COMPATIBLE_BASE_URL` (naming the endpoint is the consent) plus
  `MODELPASS_LIVE_COMPATIBLE_MODEL`. It spends nothing but electricity against a
  local Ollama, and a green run moves nothing in the shared table, by design.

- **The `google-api` runtime has an adapter, and every API runtime now has one**
  (R4, ticket 1.11). New `google-api` extra (`pip install
  "modelpass[google-api]"`, pinning `google-genai>=1,<2`; written against
  **1.73.1**). Streaming chat over `models.generate_content_stream`, an
  in-adapter tool loop with no turn cap, native structured output
  (`response_mime_type` + `response_json_schema`), usage through
  `usage_metadata`, an opt-in `models.list()` probe, and graceful cancellation
  between rounds: the shape ticket 1.6 set, a fourth and final time. The runtime
  ships **ungated**, and the two Google *subscription* runtimes stay gated — the
  D5 amendment, unchanged by this ticket.

- **Eleven `google-api` capability cells moved to `supported`**, each on the two
  pieces of evidence this table asks for: a fake-transport test in
  `tests/test_adapter_google_api.py` and the installed `google-genai` 1.73.1
  typed surface. `thinking` stays `unverified` — the adapter maps a thought part
  to a `ThinkingEvent`, but thoughts are emitted only when the request asks and
  only by a model that thinks, and `tests/live/test_google_api_live.py::
  test_one_thinking_call` is the drive that moves it.

- **`mcp_servers` on `google-api` moved the *other* way**, from a checked absence
  to `unverified`. The shared "an API endpoint runs no MCP client" note is false
  for this vendor: `Tool.mcp_servers` is `list[McpServer]`, a per-run declaration
  asking Gemini to connect to a server itself. modelpass sends none and has
  driven none, so the honest cell is the open one. Same correction ticket 1.6
  made to `anthropic-api`'s `ttl_control`.

- **The `google-api` sampling row is driven and `verified`.** `temperature`,
  `top_p`, **`top_k`** and `max_output_tokens` are sent under their own names —
  `top_k` is the control this runtime has and neither OpenAI runtime does — and
  `reasoning_effort` becomes `thinking_config.thinking_level`, whose
  `LOW | MEDIUM | HIGH` are modelpass's own three words. **No `thinking_budget`
  is ever sent**: it is a token count whose allowed range the SDK documents as
  model dependent, and ticket 1.7 refused to invent one for Anthropic on the same
  ground. No default ceiling either, as on `openai-api`.

- **Thoughts are opt-in**: `options={"include_thoughts": True}` fills
  `thinking_config.include_thoughts`. modelpass does not ask on its own, because
  thought output is billed output nobody requested.

- `tests/live/test_google_api_live.py`, marker `live`, deselected by default and
  gated on `SUBPASS_LIVE_TESTS=1` plus `MODELPASS_LIVE_GOOGLE_API_KEY` — a
  variable of its own, deliberately neither `GEMINI_API_KEY` nor
  `GOOGLE_API_KEY`, the **two** ambient names `genai.Client()` discovers by
  itself and which the adapter's explicit `api_key=` makes unreachable.

- Docs: `docs/api-and-runtimes.md` §2.0d, the runtime status table, the sampling
  table and the README install line.

- **Per-call timeouts, typed retryability, a connection-level "never retry", and
  a written thread-safety contract** (R6, R7, ticket 1.12). Four contracts that
  were previously four consumer workarounds.

  **New contract names**, all exported from `modelpass`:
  `Timeout`, `Retryable`, `RetryVerdict`, `RunTimedOut`, `SessionBusy`,
  `classify_terminal`, `classify_error`, plus `TerminalStatus.TIMED_OUT` and
  `modelpass.testing.StallingAdapter`. On the LangChain leaf:
  `SubpassTimeoutError` (a `SubpassRunError` subclass, so the consumers matching
  these classes by name across the MRO keep working).

  - `Bridge.chat`, `Bridge.ask` and `Session.send` take
    `timeout=Timeout(total=..., first_token=...)`, or a bare number meaning
    `total`. On expiry the run is cancelled through the adapter's own `cancel()`
    hook and the stream ends with one terminal event, `status = timed_out`,
    carrying the tokens spent before the bound ran out. `ask()` and
    `chat(raise_on_stop=True)` raise `RunTimedOut`. Connections carry a default
    with the additive `timeoutSeconds` key; a call that names its own bound
    replaces it rather than being clamped by it.
  - **`TerminalStatus.TIMED_OUT` is a new member** rather than a `CANCELLED`
    carrying a reason. A retry loop wants a timeout and must never touch a
    cancel. Checked against all four consumers first: none exhausts the enum,
    and the one app that keeps status tuples at all keeps them as documentation,
    never as validation.
  - **Every terminal event and every error carries `retryable: Retryable`**
    (`yes` / `no` / `unknown`) and `retry_after` where the vendor named a wait.
    The table lives in one module, `modelpass.retry`, with a family per adapter
    kind and a dated note per row, and it is computed **from typed fields only —
    never from message text**. A consumer once substring-matched `"500"` out of
    `"stopAtTokens threshold 500000 reached"` and retried a deterministic guard
    stop four times against a live allowance; there is now a regression test
    named after that.
  - **modelpass still never retries.** The verdict is for the caller's loop.
  - **A connection may declare `retry = "never"`** (additive key), forcing every
    verdict on it to `no` with a note saying the policy is why. The receipt
    reports the stance before the run.
  - **The thread-safety contract is stated and tested.** One `Bridge` is safe to
    share across threads for `chat` / `ask` / `preflight` / `validate`; stores
    and the run log are guarded by locks; adapters may be shared. A `Session`
    takes one caller at a time and raises `SessionBusy` on contention rather
    than interleaving turns or deadlocking. `tests/test_thread_safety.py` runs
    concurrent calls through one bridge and asserts every result is its own.
  - Cancelling by closing the iterator is **unchanged** and now has a regression
    test that it cancels exactly once, with and without a bound configured.

- **An async face: `Bridge.achat`, `Bridge.aask`, `Session.asend` and
  `Adapter.arun`** (R1, R2, ticket 1.13). Every argument, gate, event, receipt,
  guard, run-log line and typed error is the sync twin's, because there is one
  implementation of each underneath.

  ```python
  async for event in bridge.achat(connection="claude-api", message="..."):
      ...

  answer = await bridge.aask("claude-api", "Score this: ...", schema=SCHEMA)

  async for event in session.asend("and then?"):
      ...
  ```

  - **`chat`, `ask` and `send` are unchanged**, and are not wrappers over the
    async ones. An async core with a sync wrapper would break every consumer
    calling from inside a running loop, which is three shipped integrations. The
    same rule holds at the adapter: the API adapters' `run()` is not
    re-implemented over `arun()`.
  - **Cancelling is `aclose()`**, where the sync contract is closing the
    iterator. It calls the adapter's own `cancel()` and joins the worker thread
    with a bounded wait. **Cancelled exactly once** still holds where a timeout
    already cancelled from the watchdog thread. Abandoning an async iterator
    without closing it cancels the run when the object is collected — best
    effort, and documented as best effort. New file:
    `tests/test_async_cancel.py`.
  - **`Adapter.arun` is concrete on the base class.** The default drives `run()`
    on a worker thread through a bounded queue, so every adapter gains an async
    face, including the two agent runtimes. The four API adapters override it
    natively with the vendor's async client, reached through a new
    `async_client_factory=` seam beside the existing `client_factory=`
    (`anthropic.AsyncAnthropic`, `openai.AsyncOpenAI`, `google.genai`
    `Client(...).aio`). No new dependency and no new pin: each vendor ships its
    async client in the package that is already required.
  - **A coroutine tool handler is awaited on your own loop** on those four
    runtimes. That is the point of the ticket: an async application's tool
    handlers reach their own session and pool without `to_thread` or
    `run_coroutine_threadsafe` around them.
  - **The subprocess preflight runs off the loop.** On the agent runtimes it
    launches a CLI, so `achat` hands it to a thread; on the API runtimes it is a
    credential lookup and runs inline. One consequence worth knowing: caller
    mistakes still raise from the `achat(...)` call itself, while what the
    preflight discovers raises on the first step.
  - **`SessionBusy` holds across both faces.** A sync `send` and an async
    `asend` contend for one lock.
  - **The LangChain leaf overrides async now**: `ChatSubpass._agenerate` and
    `._astream` run over `achat`, so `ainvoke` and `astream` stop running the
    sync path in an executor. `_generate` and `_stream` are unchanged, and the
    drift test that pins `ask` against `invoke` gained an async twin.
  - **Thread-safety contract, extended**: one `Bridge` is safe to share across
    threads, *and* one event loop may run many `achat` calls concurrently.
  - Internal, but visible in a stack trace: the pre-run pipeline is now
    `Bridge._prepare()` and the post-run folding is `modelpass/_fold.py`, a pure
    state machine both faces pump. The store, CLI, bench, `validate` and
    `preflight` stay synchronous on purpose.

- **`modelpass remove NAME` and `modelpass secrets`, and a command line that
  covers all six configurable runtimes** (ticket 1.14). `connect` could write a
  connection and a secrets entry and nothing on the command line could take
  either back; the bench's delete button was the only delete path.

  - `remove` prints the connection -- runtime, credential, guards, and which way
    the secret rule will fall -- and asks before writing. The rule itself is
    `ConnectionManager.remove_connection`, so the bench and the command line
    share one answer to "does the key go with it".
  - `secrets` lists the entries and the connections referencing each. An entry
    nothing references is named an orphan and left alone. No value is printed,
    and no flag prints one.
  - `check` reports what a drive found an `openai-compatible` endpoint doing,
    and on an undriven one says every cell still reads `unverified` and names
    the verify command. `list` shows the base URL, and `-v` adds the timeout
    bound, a `retry = "never"` stance and the verified cells.
  - Every command is now tested against every runtime `connect` can reach, in
    both credential forms, plus the two gated Google runtimes' refusal and every
    read path against an empty store and an unreadable one.

- **The bench covers the runtimes and the fields added since 1.3** (ticket
  1.14). Six runtime columns instead of two. Each connection row carries the
  base URL, the timeout bound, the retry stance and the capability cells a drive
  recorded, with the store's compatibility notes above the listing and a stored-
  keys panel below it naming each secrets entry and what references it.

  - Adding a connection goes through the 1.4 manager, with four credential
    forms: the runtime's login, a named variable, a key pasted into a write-only
    field that is never echoed back into the page, or none at all. Renaming goes
    through `rename_connection`, so the stored entry moves with the connection.
    An edit no longer drops `retry`, `timeoutSeconds` or the verified cells.
  - The Verify button dispatches the way `modelpass verify` does: identity on a
    subscription, a capability drive on `openai-compatible`, confirming first
    because it spends and writing nothing when the endpoint could not be reached.
    The playground's model picker uses what that drive saw the endpoint serving.
  - The run log shows cache breakpoints requested against honoured, sampling
    requested against applied with the notes, the `timed_out` status in as many
    words, and the retry verdict -- computed from the row's own runtime and
    status by `classify_terminal`, never stored.
  - The R14 redaction sweep now walks the bench's rendered pages.

- **`docs/guards.md` rewritten for a majority-metered world** (ticket 1.14).
  Four of six configurable runtimes are metered now. The no-defaults rule and
  the no-dollar-thresholds ruling both survive, and the document says why each
  survived that change; the vendor-side hard cap gets a section of its own with
  a link per vendor; and "when guards actually fire" now covers the in-adapter
  tool loop, where thresholds are evaluated between rounds.

### Changed

- The import package is `modelpass`; the distribution is `modelpass`; the console script
  is `modelpass`. Repository URLs point at `staceyfarias/modelpass`.

- The config home is `~/.modelpass/`. **On first use, if `~/.modelpass/` does not exist
  and `~/.subpass/` does, the old directory is copied forward** — connections, run log,
  anything else in it — and one note on stderr says so. The old directory is never
  deleted or moved: a tool still running 0.1.x keeps reading the file it always read,
  and the two copies diverge from that point.

- `MODELPASS_HOME` is the config-location override. `SUBPASS_HOME` is still honoured when
  `MODELPASS_HOME` is unset, with a `DeprecationWarning`.

- `modelpass.store.default_home()` is public, so a caller can ask where the store will
  land without constructing one.

- `plan_launch` produces an **empty** environment on the API runtimes. Nothing is
  launched there, and a scrubbed copy of `os.environ` that nobody launches
  invites an adapter to hand it to an SDK constructor whose own discovery would
  read a key the connection never named. D2's rule survives as the in-process
  clause on adapter-contract rule 1 — the credential is resolved explicitly and
  handed to the client constructor, and the vendor SDK never reads the
  environment — and rides on the receipt as two directives.

- **An unreadable connection file now reads as zero connections instead of
  raising** (R10). The connection file has caught up with the secrets file: a
  file this build cannot open or parse, or one whose `version` is not a whole
  number, reads as "nothing configured" — the same legitimate state a fresh
  install is in — and every read path carries a note saying the file could not be
  read (`store.unreadable_reason()`, `store.compatibility().notes()`,
  `secrets.notes()`, printed by `modelpass check`). Writes are unchanged and still
  refuse, naming the file and the reason, because rewriting bytes nobody parsed
  would destroy them. Before this, `modelpass list` on a machine with a corrupt or
  unreadable file could not speak at all, which is the moment a user most needs it
  to. `tests/test_nothing_configured.py` walks every read path against five
  states: no home directory, an empty home, an empty file, an unreadable
  connection file and an unreadable secrets file.

- Flattening block content on the agent runtimes joins blocks with one blank line again (`Message.flat_text`), as 0.1 did; `Message.text` stays a byte-faithful join for the API runtimes. Regression from the content-blocks change, caught by a consumer suite.

- **`Bridge.adapter_for` binds a lazily-loaded adapter to that bridge's own
  environment and secrets file.** The agent adapters never needed it — their
  credential travels in `PreflightPlan.env`, which the bridge built. An API
  adapter has no such mapping by design and resolves its own credential at the
  moment of use, so without this a bridge pointed at a test root or at an
  application's own directory would read the keys in the user's real
  `~/.modelpass/`. Duck-typed (`bind_resolution`), so only the adapters that
  resolve their own credential are touched; an **injected** adapter is left
  alone, because whoever constructed it chose its sources.

- **`manage.receipt_for` stays on the offline preflight for API runtimes even
  once an adapter exists.** Setup is the one caller that must check a key it has
  not written down yet; going through the adapter would resolve the reference
  from the file on disk instead and fail a connection that is about to be
  perfectly good. Nothing is lost — an API adapter's preflight *is*
  `api_preflight`, plus an opt-in probe setup does not ask for.

- **`ChatSubpass` passes sampling through instead of warning about it.**
  `temperature`, `top_p`, `top_k`, `max_tokens` and `reasoning_effort` on the
  constructor — and the same names through `bind(...)` for one call — map onto
  `Sampling`, and `response_metadata` gains `subpass_sampling_applied` and
  `subpass_sampling_notes`. The constructor-time warning is gone (it is a debug
  line now): it fired before a connection was resolved, so it could not know
  whether the runtime honours the value, and it cost more than noise — a
  desktop agent app withheld temperature from this class entirely rather than be
  shouted at, so a user who set one silently got nothing. No existing
  `subpass_*` key was renamed.

- `options={"max_output_tokens": N}` is now an **alias** for
  `sampling=Sampling(max_output_tokens=N)`, folded in by
  `RunRequest.effective_sampling` so one value reaches the wire and one appears
  on the receipt. It keeps working and is deprecated in the docstring; it does
  not warn yet.

- **`top_k` came off the `openai-compatible` sampling row** (ticket 1.10) and is
  now dropped with a note like every other unaccepted field. It is in neither of
  `openai` 2.32.0's create signatures, and the servers that take one disagree
  about where it goes — Ollama inside its own options block, vLLM at the top
  level — so the only way to send it was an `extra_body` guess that would ride
  along on every endpoint that does not take one. A named drop beats a silently
  different sampling. The row is now marked `verified`, and `temperature`,
  `top_p` and `max_output_tokens` (sent as `max_tokens`) are unchanged.

### Fixed

- **`Connection.vendor` no longer derives a vendor from the runtime's name.**
  `VENDOR_OF` in `runtimes.py` is the single source, and `openai-compatible`'s
  vendor is `compatible`. The old derivation read `"openai"` for a connection
  pointed at a local Ollama box, so `bridge.find(vendor="openai")` would have
  returned it beside real OpenAI accounts.

- **A `Runtime` member with no `STATIC_TABLE` row now fails at import** with a
  message naming the runtime and the fix, instead of surfacing as a confusing
  `ValueError` from inside `Connection.__post_init__` the first time somebody
  configured one.

- **Sessions are refused on the four API runtimes, at the bridge, before an
  adapter is loaded.** `new_chat`, `new_worker`, `resume_chat` and
  `list_sessions` raise `CapabilityNotSupported` naming the runtime and naming
  `bridge.chat(history=[...])` as the stateless alternative, whenever the
  `sessions_resume` / `sessions_fork` / `sessions_list` cells all read
  `unsupported`. The policy is read off the capability table and lives in one
  place, rather than being re-stated in each API adapter where the one that
  forgets degrades silently. `anthropic-api`'s own three refusals and their
  wording are unchanged; nothing routed through the bridge reaches them now. The
  refusal deliberately precedes the `enabled` check, because re-enabling the
  connection would not change the answer.

- **`new_worker()` on a runtime with no native toolbelt is refused instead of
  returning a chat wearing a worker's name** (design latent-defect list).
  `SessionRequest.native_tools` and `.appends_system_prompt` are both derived
  from `kind is SessionKind.WORKER` and never asked whether the runtime has a
  toolbelt or a persona to be true of. The `tools` cell now gates the worker
  door, first among `_open_session`'s gates, and `unverified` refuses as well as
  `unsupported`.

### Not changed, on purpose

- **`SubpassError` and its 23 subclasses keep their names**, inside `modelpass`. So does
  `ChatSubpass`, and so do the `subpass_*` keys in the LangChain adapter's
  `response_metadata`. These are caught and read by name across four consumer projects;
  renaming them is a contract change, not part of a rename, and it is out of scope here
  (DESIGN R9). They get their own decision.

- `SUBPASS_LIVE_TESTS`, `SUBPASS_CODEX_ROOT` and `SUBPASS_BRIDGE` are developer- and
  bench-facing knobs, not part of the published contract; they are left for a later pass.

### Deprecated

- `subpass` (the import package, the console script), `SUBPASS_HOME`, and the
  `~/.subpass` fallback. All four are removed in 0.3.0, once every consumer has migrated.

---


## 0.1.1 — 2026-09-13

The forward-compatibility policy for `~/.subpass/connections.toml`, shipped on its own
and before anything writes a new file shape. Several tools on one machine share that
file, so they will not all be the same build of subpass, and until now an older build
meeting a newer file had no behaviour worth relying on.

**The policy.** It is stated in the header comment subpass writes into every
connections file, and in the README's configuration reference:

- The file stays **version 1** across the coming API-runtime extension. New keys and new
  runtime values are **additive**. The version bumps only when an existing key changes
  meaning or disappears.
- A build **reads any version**. It refuses to *write* a file whose version is newer than
  its own, with an error naming both versions and saying a newer build is needed.
- A key this build does not recognise is **preserved verbatim** and **reported** — never
  a failure, and never dropped when the file is rewritten.
- An **unknown runtime or auth mode** refuses **that connection**, names the values this
  build knows, and leaves every other connection usable.

### Changed

- The config version check is `<` ("newer than this build"), not `!=`. An older file
  version reads without complaint; a newer one reads everything parseable. Reading no
  longer has a path that leaves `subpass list` and `subpass check` unable to speak.
- Unknown keys on a connection — and inside its `guards`, `guards.onQuotaExhausted` and
  `verifiedIdentity` sub-tables — are no longer errors. They survive a load/save round
  trip byte-for-byte, following the carry-through `[settings]` already had.
- A connection naming an unknown `runtime` or `authMode` is left out of `load()` instead
  of raising for the whole file, and is written back untouched. `get()` on it raises
  `InvalidConnection` naming the reason rather than `NoSuchConnection`, and `remove()`
  can still delete it.
- Everything else invalid — a guard above its stop, `enabled = "no"` — still refuses the
  file. Those are mistakes in a hand-edited file, and dropping the connection would hide
  one.

### Added

- `ConnectionStore.compatibility()` returns a `StoreCompatibility` report: the file
  version, whether this build may write, the carried keys per connection, and the refused
  connections with reasons. `subpass accounts` and `subpass check` print its notes;
  `Bridge.validate()` carries the per-connection ones as notes, not problems.
- `subpass.store.unknown_connection_keys()`, for a host application that wants the same
  answer about a table it holds itself.
- `ConnectionStore.save(..., drop=...)`, the one way a refused connection leaves the file.
- `tests/test_store_compat.py`, and a test asserting `subpass.__version__` matches
  `pyproject.toml`.
