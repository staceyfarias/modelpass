# The API by contract, and the runtimes behind it

> Written 2026-08-31 against the source in this repository. Every capability claim
> below traces to the capability registry (`src/modelpass/capabilities.py`), a
> docstring, or a dated live finding — and where nobody has checked something, it
> says so. `unverified` means *nobody checked*. It is not a soft yes, and
> `supports()` treats it as unusable.
>
> The runtime facts here are dated because they go stale. Vendors ship; the
> measured numbers came off codex 0.151.0-alpha.7.1 and claude-agent-sdk 0.2.139
> on one Windows machine. Re-verify before you build on them.

Audience: a developer adopting modelpass. For what modelpass *is* and how to install
it, read [../README.md](../README.md) first. This document is the contract: what
each entry point promises, what it refuses, and what the two implemented runtimes
actually do with what you hand them.

---

## 1. The API by contract

Everything starts at a `Bridge`, which owns a connection store, a capability
registry and the adapters:

```python
import modelpass

bridge = modelpass.Bridge()          # reads ~/.modelpass/connections.toml
```

A connection binds a runtime, an auth mode and a credential together, and is
named. There is no per-call "use an API key this time": switching billing means
naming a different connection (D3).

### 1.1 The stream contract, stated once

Both doors — `Bridge.chat()` and `Session.send()` — return an
`Iterator[AgentEvent]` under the same contract:

* **Exactly one `ReceiptEvent` first**, per connection that runs. The rule is
  that a receipt always precedes any event produced by the connection it
  describes.
* **Exactly one `TerminalEvent` last** on a stream that completes, stamped by the
  bridge with the connection name and the auth mode that actually paid (D3).
  Adapters cannot forge that stamp.
* **A guard stop, a spent quota and a vendor failure are terminal *statuses*, not
  exceptions.** A rejected model, an API error, a runtime that would not start:
  the stream ends with one terminal carrying `status=error`, a reason, and the
  usage spent up to that point. The exception is `raise_on_stop=True`, which
  turns guard stops and quota exhaustion back into raises.
* **Configuration and preflight problems raise before any event exists.** A
  caller holding an iterator has already passed the auth check, and the receipt
  event is that guarantee's evidence.
* **Closing the iterator cancels the run**, best-effort, with a floor of
  terminating the runtime process (D10).
* `schema=` adds exactly one `StructuredOutputEvent` immediately before the
  terminal. `tools=` / `mcp_servers=` add `ToolCallEvent` / `ToolResultEvent`,
  which are **observations** — the runtime executed them; you never answer one.
* Every run reaching a terminal appends one line to `~/.modelpass/runs.jsonl`
  (best-effort, opt-out via `[settings] runLog`).
* A connection with an `onQuotaExhausted.failover` may finish on a *second*
  connection. That is the only case where one `chat()` call touches two, it
  happens only because somebody wrote the second one down, and it is announced by
  a `failover` event followed by the second connection's own receipt.

The full statement lives in `src/modelpass/bridge.py`'s module docstring and the
adapter-side rules in `src/modelpass/adapters/base.py`.

### 1.2 `chat()` — the stateless door

```python
def chat(
    *,
    connection: Connection | str,
    message: str,
    system_prompt: str | None = None,
    history: Iterable[Message | Mapping[str, Any]] | None = None,
    model: str | None = None,
    expect_auth_mode: AuthMode | str | None = None,
    guards: Guards | None = None,
    stop_at_tokens: int | None = None,
    warn_at_tokens: int | None = None,
    allow_failover: bool | None = None,
    raise_on_stop: bool = False,
    options: Mapping[str, Any] | None = None,
    sampling: Sampling | Mapping[str, Any] | None = None,
    tools: Iterable[ToolDef | Mapping[str, Any]] | None = None,
    mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
    schema: Mapping[str, Any] | None = None,
    schema_name: str | None = None,
) -> Iterator[AgentEvent]
```

**You pass everything in.** The call keeps nothing between invocations. The three
content arguments are assembled in exactly one order — instructions, what was
already said, then the turn being taken now:

```python
turns = []
if system_prompt is not None:
    turns.append({"role": "system", "content": system_prompt})
turns.extend(history or ())
turns.append({"role": "user", "content": message})
```

That assembly is done in the bridge rather than asked of the caller, which is
what makes the shape honest: `history` is visibly yours to hold, and it is
re-sent against your allowance on every call.

What each argument promises:

| Argument | Contract |
| --- | --- |
| `message` | The new user turn. Required. |
| `history` | `Message` objects or `{"role": ..., "content": ...}` dicts, in the order given, ahead of `message`. Omitting it is the single-shot case. |
| `system_prompt` | The front of the cached prefix. **Keep it here rather than concatenating it into `message`** — that is the difference between every call after the first reading from cache and every call paying full price. Absent means the runtime's *minimal* prompt, which is a decision (D22): modelpass writes no prompt of its own. |
| `model` | The model to ask for. The receipt names what was resolved. |
| `expect_auth_mode` | An **assertion, not an override**. Refuses to run if the connection is not what the caller thought. |
| `guards` | Overrides the connection's thresholds for this call only; never written back to config. A failover target named here is validated *before* anything runs. |
| `stop_at_tokens` / `warn_at_tokens` | Shorthand for "the connection's guards, but a tighter ceiling here". They start from the connection's own guards, so lowering a ceiling never quietly disarms a configured failover. `0` means no guard. **Refused alongside `guards=`** — two ways of saying the same thing, one of which would have to silently lose. **Omitting them is not the same as having none**: a call that mentions no guard is bounded by whatever the *connection* configures, and each guard is its own switch, so `stop_at_tokens=0` leaves a configured warning armed. Pass both zeros to run one call unguarded. modelpass ships no default thresholds at all, so a connection that configures none has none — see [guards.md](guards.md) for the full table. |
| `allow_failover=False` | Declines a configured quota failover for this call only. Same refusal rule against `guards=`. `True` states the default out loud; it does not *enable* a failover nobody configured. |
| `raise_on_stop` | Turns guard stop / quota exhaustion back into raises. |
| `options` | Per-call launch levers, e.g. `{"codex_bin": ...}`, `{"transport": "app-server"}`, `{"max_turns": N}`. Unknown keys are noted on the receipt rather than silently dropped. |
| `tools` / `mcp_servers` | Hand the runtime a tool loop inside this one call (D12). The call stays stateless — one ephemeral runtime session — but the runtime may take several internal turns, observed as `tool_call` / `tool_result`. |
| `schema` / `schema_name` | Binds the final answer to a JSON Schema using each runtime's **native** mechanism (D13). `schema_name` defaults to the schema's `title` and matters on `anthropic-sdk`, where the mechanism is a tool call a caller's prompt may refer to by name. |

**What it refuses**, all of it before the preflight and therefore before any
spend — asking a runtime for something it cannot do must not cost a token to find
out:

* A disabled connection → `ConnectionDisabled`. (`preflight()` deliberately does
  *not* check this: what is switched off is spending, not looking.)
* A capability the registry does not report as `supported` → `CapabilityNotSupported`,
  including for an `unverified` cell.
* `schema=` together with `tools=` or `mcp_servers=` → `InvalidSchema`. Deliberate
  (D13): the mechanics conflict on `openai-sdk` (`--output-schema` reported
  silently ignored while MCP servers are active, openai/codex#15451) and the
  interaction is unverified on `anthropic-sdk`. A combination that silently
  worked on one runtime and silently did not on the other is worse than one that
  refuses on both.
* `schema_name=` without `schema=` → `InvalidSchema`. On its own it does nothing.
* `expect_auth_mode` not matching → `AuthModeMismatch`.

```python
for event in bridge.chat(
    connection="claude-sub",
    system_prompt=RUBRIC,                     # stable across the loop
    message=f"Score this: {item}",
):
    if isinstance(event, modelpass.TextDeltaEvent):
        print(event.text, end="")
    elif isinstance(event, modelpass.TerminalEvent):
        print(event.status, event.usage.total_tokens)
```

### 1.3 The three session constructors

The other door is stateful, and the distinction is *who is holding the
conversation* (D21). With a session the **runtime** owns the history and the
prefix cache is native.

Construction runs the preflight and every capability assertion and hands back a
local object with a receipt already available. **Nothing is spent and no session
exists yet**: neither runtime has a create-session call, so the session comes
into existence when the first `send()` *completes* — which is also when
`Session.id` stops being `None`.

`codex app-server` is the one place that timing is a *choice* rather than a
constraint: `thread/start` answers with a thread id before the first turn runs.
The id is still withheld until a turn completes, because the rule is about
handing out an id a resume would reject and nobody has yet driven a
`thread/resume` of a thread whose first turn never finished. One experiment
settles it; until then the conservative timing holds on both transports.

`ChatSession` and `WorkerSession` share one implementation and differ in three
class attributes. The difference is **policy**, not behaviour:

| | `new_chat()` → `ChatSession` | `new_worker()` → `WorkerSession` |
| --- | --- | --- |
| The runtime's own agent persona | **replaced** by your `system_prompt` | **kept**; your `system_prompt` is appended to it |
| The runtime's native toolbelt | **off** | **on** |
| `project_folder` | optional — defaults to a fresh modelpass-owned scratch directory | **required** |
| Gated on `system_prompt_replace` | yes, when a `system_prompt` is passed — and on `openai-sdk` that gate depends on the transport: it opens by default and is refused on `options={"transport": "exec"}` (§3.5) | no — append is the semantics you wanted |
| What it is for | a scorer, a classifier, a rewriter, an assistant: work where the runtime's built-in identity is something to remove | work where the runtime's own agent, pointed at a real folder, is the point |

The scratch-directory default is not tidiness. With `persist=True` as the
default, inheriting the caller's cwd would write agent transcripts into the
consumer's own `~/.claude/projects/<their-cwd>/`, where their `claude --continue`
would find them (D15). It matters on `openai-sdk` too, where every `AGENTS.md` on
the cwd's ancestor chain joins the prompt — a scratch directory has no ancestors
carrying somebody's repo instructions.

`persist=False` is not merely a storage flag. On `anthropic-sdk` a live client
still carries the conversation across turns, so multi-turn survives; on
`openai-sdk` continuation needs the rollout file on disk and there would be no
multi-turn at all. Hence the `ephemeral_multi_turn` gate: `new_chat(persist=False)`
**raises at construction** on Codex rather than degrading into a series of
one-shots wearing a session's name.

Guards on a session are the **session's** envelope, not the turn's: `stop_at_tokens`
bounds everything this session will ever spend (D18). One tracker for the whole
session — a per-turn tracker on a fifty-turn worker never fires while the
allowance drains.

```python
with bridge.new_chat(
    connection="claude-sub",
    system_prompt=RUBRIC,
    stop_at_tokens=200_000,        # the whole session's ceiling
) as session:
    for item in items:
        for event in session.send(f"Score this: {item}"):
            ...
    print(session.id, session.usage.total_tokens)
```

#### `resume_chat()`

Picks a persisted session back up by id. `SessionNotFound` is an **ordinary
outcome, not a broken caller**: Claude Code's `cleanupPeriodDays` sweep deletes
old transcripts, and a Codex thread whose first turn never completed leaves an id
that resume rejects. A stored id is a hint.

* **There is deliberately no `tools=`.** Tools were fixed at creation, they are
  the very front of a prefix that has been warm ever since, and modelpass cannot
  check a re-declared value against what the session actually holds — so
  accepting one would mean either silently ignoring it or silently changing the
  prefix.
* **`system_prompt` splits by runtime**, gated on `resume_carries_system_prompt`,
  because the two store it in different places:
  * **`anthropic-sdk`: required.** The prompt is a *launch option*, not part of
    the transcript, so resuming without one launches under the minimal
    tool-calling prompt and sends a prefix the original never warmed — wrong
    answers and a cold cache, silently. `""` is how you say the session
    genuinely had none.
  * **`openai-sdk`: refused.** The prompt was serialized into the first turn and
    lives in the rollout, so resuming replays it. Passing one would inject a
    *second* persona mid-conversation rather than restore the first.
* `project_folder` matters on `anthropic-sdk`, where the folder **is** the
  storage key. Pass the one the session reported (`Session.project_folder`).
* `persist=False` alongside a `session_id` is refused out loud: an ephemeral
  session was never written down, so there is nothing to resume.

#### `list_sessions()`

```python
def list_sessions(*, connection, project_folder=None, options=None) -> list[SessionInfo]
```

Capability-gated on `sessions_list` and **token-free**. The gate asks the adapter
before the table, so on `openai-sdk` the answer depends on the transport this
call selects — and since the 2026-08-31 default flip the answer by default is
yes: `thread/list` on `codex app-server`. `options={"transport": "exec"}` opts
out into a transport with no scriptable listing at all, and is refused there
rather than answered empty. The refusal names the default it opted out of.

On `openai-sdk` the store is **flat**: the listing is account-wide, ignores
`project_folder`, and each row reports the working directory its own thread ran
in. `title` comes from the thread's own name and is `None` where it has none —
the first prompt is `preview`, and a listing whose titles were the opening words
of a message would be a different thing wearing the field's name, so it rides in
`vendor`. `message_count` is `None` because the listing does not count turns.

`project_folder` scopes the listing where the runtime scopes by it. On
`anthropic-sdk` that is not optional in practice — a session opened with
modelpass's default scratch folder will not appear in a listing of anywhere else.
Omitting it lists the **process working directory**, which is the only default
that is a fact rather than a guess. `persist=False` sessions never appear, on any
runtime.

`SessionInfo` is normalized to the **intersection** of what the runtimes agree
on: `id` is the only field guaranteed present, everything else
(`project_folder`, `created_at`, `updated_at`, `title`, `message_count`) is
`None` when the runtime did not say, and vendor-specific extras live in
`vendor`. `None` means *not reported*, never *empty*. Timestamps are ISO 8601
strings, not `datetime` (D11).

#### The session policy: which runtimes have sessions at all (ticket 1.8, R11)

A session rests on the **runtime** holding the conversation (D14). Where it holds
none, all four session doors are shut — `new_chat()`, `new_worker()`,
`resume_chat()`, `list_sessions()` — with one
`CapabilityNotSupported` that names the runtime and names the alternative:

```
runtime 'anthropic-api' capability 'sessions_resume' is unsupported:
bridge.new_chat() needs the runtime itself to hold the conversation (D14), and
'anthropic-api' holds none: every request carries its whole history. Use
bridge.chat(history=[...]), the stateless alternative, or bridge.ask(...) when
one blocking answer is what you want
```

Four facts about that refusal:

* **It is read off the table, not hard-coded per runtime.** The rule is *every*
  cell in `sessions_resume` / `sessions_fork` / `sessions_list` reading
  `unsupported`, which is what an HTTP request/response API looks like from
  here. A runtime that can resume, fork *or* list has somewhere to hold a
  session, and the per-argument gates take it from there. Today that catches the
  four API runtimes and nothing else.
* **It is raised at the bridge, before an adapter is loaded.** No vendor package
  is imported, no launch plan is built, no credential is resolved, nothing is
  spent. `anthropic-api`'s own three refusals are still there and still say the
  same thing, but nothing routed through the bridge reaches them.
* **It comes before the `enabled` check.** Re-enabling the connection would not
  change the answer, and pointing a reader at a flag that cannot help them is
  worse than saying nothing.
* **It is `CapabilityNotSupported`, not a new class.** That is the error whose
  whole job is *a fact about the runtime, not a mistake in the call*, and it
  already carries `.runtime`, `.capability` and `.support` for a caller that
  wants to branch rather than read English.

`new_worker()` carries a second refusal of the same class, on the `tools` cell,
and it is there because of a trap rather than a limitation.
`SessionRequest.native_tools` and `.appends_system_prompt` are both derived from
`kind is SessionKind.WORKER` and nothing else — neither asks whether the runtime
*has* a toolbelt or a persona. On a runtime with neither, `new_worker()` would
have constructed happily and handed back a chat wearing a worker's name, which is
the one outcome having two session classes exists to prevent. A `tools` cell that
is not `supported` refuses, and `unverified` refuses too: "nobody has checked
whether this runtime has a toolbelt" is not a basis for arming one.

### 1.4 `Session.send()`, and the absence that is the contract

```python
def send(self, message: str) -> Iterator[AgentEvent]
```

**There is no `tools=` and no `system_prompt=` here, and that absence is the
caching contract** (D17). Both are fixed at construction because they sit at the
front of the cached prefix — `tools` → `system` → `messages`, matched exactly —
so changing either between turns recomputes everything after it, on every
runtime, every time. On `openai-sdk` prefix stability is the *only* cache lever
there is. Making that unexpressible beats documenting it as discouraged.

Keeping `system_prompt` a construction argument, rather than letting callers
concatenate instructions into each message, is what makes the prefix stable
across a loop.

Per turn, the stream contract of §1.1 applies in full: one receipt first (repeated
every turn, because each `send` is its own stream), one terminal last carrying
**this turn's** usage. A terminal ends the *turn*, not the session — the next
`send` continues against the same warm prefix. Two things end the session: a
guard stop (the ceiling was for the whole session) and `close()`.

Refusals: a non-`str` message → `TypeError`; an empty one → `ValueError`; sending
on a closed session → `SessionClosed`; sending after a guard stop → `GuardStop`,
**raised rather than answered with a terminal**, because nothing ran and a
terminal for a turn that never happened would put a line in the ledger for work
nobody did.

Abandoning the returned iterator closes it (the transport-level cancel of D10).
The session stays open; whether the runtime's own history records a partial turn
is the runtime's business and modelpass does not pretend to know.

### 1.5 `get_history()`, `close()`, `help()`

**`get_history() -> list[Message]` is read-only introspection.** It cannot be fed
back as input on either runtime, which is the fact D14 is built on: history is
not something a caller passes to a session, it is something the runtime holds and
modelpass can *read*. A runtime with no way to read it **answers with an empty list
rather than a transcript modelpass reassembled from the events it happened to
see** — two histories that drift, only one of which is what the model is being
sent, is worse than one honest absence. Raises `SessionClosed` on a closed
session. See §3.4 for which runtime returns what.

**`close()`** releases modelpass's hold and is idempotent. It does **not** delete a
persisted session — that stays resumable by `id`. The one exception: a
`persist=False` session's scratch working directory is removed here, because
"nothing on disk" is the promise that flag makes.

**`help() -> str`** prints what this object can and cannot do **on this
connection**, in prose: the shape it was constructed with, the capability cells
its behaviour turns on with the registry's own dated notes, and the preflight
receipt. It is class-aware, and that is why it is a method rather than a module
function — a `ChatSession` on a runtime that cannot replace a system prompt leads
with exactly that, and a `WorkerSession` on the same connection does not mention
it at all, because there the runtime's persona is what you wanted.

Read-only properties worth knowing: `id`, `connection`, `runtime`, `receipt`
(available from construction), `project_folder`, `persists`, `system_prompt`,
`tools`, `usage` (the session's running total — the same number the guard is
comparing against), `closed`.

### 1.5a `ask()` — the collected door (R11)

```python
def ask(connection, message, *, system_prompt=None, history=None, model=None,
        expect_auth_mode=None, guards=None, stop_at_tokens=None, warn_at_tokens=None,
        allow_failover=None, options=None, sampling=None, tools=None,
        mcp_servers=None, schema=None, schema_name=None) -> Answer
```

`chat()`, drained. Same arguments, same gates, same receipt, same run-log line —
this *is* `chat()`, with the loop written once in the library instead of once at
every call site. Reach for `chat()` when you want to render an answer as it
arrives, show a tool call the moment it happens, or stop a run partway through;
reach for `ask()` when what you want is the answer.

`Answer` is frozen and carries `text`, `structured`, `usage`, `receipt`,
`status`, `reason`, `tool_calls`, `events`, `sampling_applied` and
`sampling_notes`:

| Field | What it is |
| --- | --- |
| `text` | every `text_delta` concatenated, in order — the same string `ChatSubpass.invoke` puts on its message, and a test runs both over one script so the two cannot drift |
| `structured` | the parsed schema-bound answer; `None` **only** when no `schema=` was given |
| `usage` | the final cumulative `TokenUsage`, never a sum of the deltas |
| `receipt` | the receipt of the connection that *finished* (a failover run touched two; both receipt events are in `events`) |
| `status` / `reason` | always `ok` by the time you hold one, because every other terminal raises |
| `tool_calls` | the `tool_call` / `tool_result` observations in stream order |
| `events` | everything, in order, for thinking, vendor passthrough and guard warnings |
| `sampling_applied` / `sampling_notes` | ticket 1.7's honesty report, lifted off the receipt |

**A terminal that is not `ok` raises**: `GuardStop`, `QuotaExhausted`,
`VendorRunFailed` — the same three a `chat(raise_on_stop=True)` caller already
handles. That is not a second error policy. A streaming caller who sees
`status=error` has already rendered the half answer above it and knows what they
have; a caller holding one returned object does not, and an `Answer` with the
first two sentences of a refusal in `text` would be indexed, scored and written
to a database as though the run had finished. There is consequently **no
`raise_on_stop=` keyword here** — it would be one whose `False` this method
cannot honour. The one status that cannot reach `ask()` is `cancelled`:
cancellation is what closing the iterator means (D10), and nothing closes this
one early.

With `schema=`, `structured` is never `None` on an `ok` run. A run that produced
nothing parseable ends `status=error` and has already raised by then; the
remaining case — an adapter that ended `ok` and emitted no `structured_output`
event — is a broken D13 contract and is raised as `AdapterFailed` rather than
handed over as an empty result that looks exactly like a real one. `schema=` with
`tools=` is still refused (D13), because that refusal lives in `chat()` and
`ask()` delegates rather than re-implementing.

`connection` and `message` may be positional here, where every `chat()` argument
is keyword-only. The asymmetry is the whole ergonomic: a one-shot call site reads
as one line, and those are the two arguments nobody mistakes for each other.

### 1.5b Timeouts, retryability, and the threading contract (R6, R7, ticket 1.12)

Three contracts that arrived together on 2026-09-13, because three of the four
consumer reviews asked for the same missing sentence in three different ways.

**The bound.** `chat()`, `ask()` and `Session.send()` take
`timeout=Timeout(total=..., first_token=...)`, or a bare number meaning `total`.

| field | bounds |
| --- | --- |
| `total` | the whole run's wall clock, failover legs included — one call, one clock |
| `first_token` | the wait for the first event that is **not** the receipt |

Mechanically it is a timer thread that calls the adapter's own `cancel()` and
nothing else. That shape is taken from the two consumers who had already built it
(a retrieval service and a desktop agent app, each with a threading watchdog
around a bounded call), including the part they learned the hard way: **the watchdog must not close the iterator**,
because a generator closed from one thread while another is executing it raises
`ValueError`. The watchdog cancels the adapter; the consuming thread, unblocked,
does the closing. A clock checked *between* events would never fire on the case
that matters, which is a runtime that came up and then said nothing — the hang
a batch scraping tool works around today with `multiprocessing` and
`join(timeout)`.

On expiry the stream ends with one terminal event, `status = timed_out`, carrying
the usage accumulated up to that point. `TIMED_OUT` is a new `TerminalStatus`
member rather than a `CANCELLED` with a reason, because a retry loop wants the
first and must never touch the second: a run the caller stopped is not a run to
try again. The addition was checked against every consumer that reads this
vocabulary first — none of them exhausts the enum, and the one app that keeps
status tuples at all keeps them as documentation, never as validation.

`ask()` and `chat(raise_on_stop=True)` raise `RunTimedOut`, which carries
`connection`, `which` (`"total"` or `"first_token"`), `seconds` and `usage`. The
LangChain leaf raises `SubpassTimeoutError`, a `SubpassRunError` subclass, so the
two consumers matching these classes **by name across the MRO** keep working.

A connection may carry `timeoutSeconds` as a default. A call that names its own
bound replaces that default rather than being clamped by it.

**The verdict.** Every terminal event and every error carries
`retryable: Retryable` (`YES` / `NO` / `UNKNOWN`) plus `retry_after` where the
vendor named a wait. **modelpass never retries** — there is no loop here, because
the library does not know what a second run costs against your allowance and you
do. The classification lives in `modelpass/retry.py`, one table with a family per
adapter kind and a dated note per row, and it reads **typed fields only**: HTTP
status codes, exception class names, terminal statuses, the connection's declared
policy. Never message text. The rule is a scar: `"500"` once matched inside
`"stopAtTokens threshold 500000 reached"` and retried a deterministic guard stop
four times against a live allowance.

Two rows are worth calling out because they depart from HTTP intuition:

* **A 429 with no `retry-after` is `UNKNOWN`, not `YES`.** A rate limit that named
  no wait may be a spent allowance wearing a rate limit's number.
* **A closed agent transport is `UNKNOWN`, not `YES`.** On a metered endpoint a
  connection reset is transient; on a subscription runtime the child process may
  already have spent allowance nobody can account for.

**The stance.** A connection may declare `retry = "never"`, which forces every
verdict on it to `NO` with a note saying the policy is why, and which the receipt
discloses before the run. A RAG evaluation harness asked for exactly this —
*"timeout 30s, 2
retries, and no retries on **this** connection"* — and the per-connection half was
the part it could not express anywhere.

**The threads.** One `Bridge` is safe to share across threads for `chat`, `ask`,
`preflight` and `validate`; each call owns its request, tracker and stream. The
store and the run log are guarded by their own locks. A `Session` takes one caller
at a time and raises `SessionBusy` on contention rather than interleaving or
deadlocking; the lock is taken on the first `next()` of a turn, so an iterator
that is built and never drained holds nothing. That same evaluation harness
carries a source comment asking for this paragraph, and it runs a client per
worker for want of it.

---

### 1.5c The async face: `achat`, `aask`, `asend` (R1, R2, ticket 1.13)

Added 2026-09-13. Three doors, and the sentence that matters about all of them is
that they are the *same* calls:

```python
async for event in bridge.achat(connection="claude-api", message="..."):
    ...

answer = await bridge.aask("claude-api", "Score this: ...", schema=SCHEMA)

async for event in session.asend("and then?"):
    ...
```

Every argument, gate, event, receipt, guard, run-log line and typed error is the
sync twin's, because there is one implementation of each underneath:

| shared | where it lives |
| --- | --- |
| the pre-run pipeline — resolve, gates, schema, guards, failover target, request, preflight | `Bridge._prepare()`, one frozen `_Prepared` |
| what a run's events *mean* — usage, cumulative, guard events, the D3 stamp, the retry verdict, the allowance, `VendorRunFailed`, the timeout arms, the ledger line, the failover decision | `modelpass/_fold.py` |

What is *not* shared is the pump: about forty lines each that know how to drive an
iterator and nothing else. That split is the decision, and it is written down as a
design amendment: async is a second face over one fold, not a second core.

**The sync face is untouched.** `chat`, `ask` and `send` are not wrappers over the
async ones. R1 rejects an async core with a sync wrapper by name — it breaks every
sync consumer calling from inside a running loop — and the same reasoning is applied
one layer down, at the adapter: the four API adapters' `run()` is not re-implemented
over their `arun()`.

**Where the two faces legitimately differ.**

* **Cancellation is `aclose()`**, where the sync contract is closing the iterator
  (D10). It calls the adapter's `cancel()` and joins the worker thread where one
  was used, with a bounded wait — bounded because a runtime that ignores its cancel
  must not be able to hang the loop that asked. Abandoning an async iterator without
  closing it is best effort: the run is cancelled when the object is collected, at a
  moment nobody controls. `tests/test_async_cancel.py` is where all of that is
  pinned.
* **Refusals arrive in two places.** Everything a caller got wrong — a disabled
  connection, an ungated capability, `schema=` beside `tools=`, a failed auth-mode
  assertion, a misconfigured failover — is raised **by the `achat(...)` call
  itself**, before you iterate. What the *preflight* discovers is raised on the
  **first step**, because the preflight is the one step that is handed to a thread:
  it is a subprocess on the agent runtimes, and blocking the loop on it is the bug
  this face exists to end (a batch scraping tool's report, §6.A). On the four
  API runtimes it is a
  credential lookup and a URL check, so it runs inline.
* **A timed-out call still ends cleanly.** The 1.12 timer thread is unchanged and
  cancels the adapter from off-loop; the teardown then joins the worker without
  blocking the loop doing the joining.

**`Adapter.arun(request)`** is concrete on the base class. The default drives
`run()` on one worker thread through a bounded queue, so every adapter has an async
face whether or not it knows what a loop is — the two agent runtimes get theirs that
way. The four API adapters override it natively with the vendor's own async client
(`anthropic.AsyncAnthropic`, `openai.AsyncOpenAI` for both OpenAI-shaped runtimes,
`google.genai` `Client(...).aio`), reached through an `async_client_factory=` seam
that mirrors the existing `client_factory=`.

The thing that buys is the **in-adapter tool loop**: a coroutine handler is awaited
on the caller's own loop, so an application whose tools are coroutines over its own
session and pool stops marshalling them across a thread boundary. That marshalling
is not hypothetical — a downstream agent host carries about eighty lines of
`to_thread` and `run_coroutine_threadsafe` doing exactly this by hand.

**Sessions.** `asend` drives the runtime's handle on a worker thread, because both
agent runtimes' session handles are synchronous and the API runtimes have no
sessions at all (ticket 1.8). The one-caller rule holds **across** the faces: a sync
`send` and an async `asend` contend for one lock, and the loser gets `SessionBusy`.

**Threads, restated.** One `Bridge` is safe to share across threads, and one event
loop may run many `achat` calls concurrently.

**What stays synchronous, deliberately:** the store, the CLI, the bench, `validate`,
`preflight` and the connection manager. They are file I/O and a subprocess, and they
do not belong on a loop.

**The LangChain leaf overrides async now.** `ChatSubpass._agenerate` and `._astream`
run over `achat`; `_generate` and `_stream` are unchanged. The leaf used to decline
async on purpose, and its reason was good while modelpass was synchronous —
`BaseChatModel` runs the sync path in an executor. Now that would be a thread
wrapping a loop-native call.

---

### 1.6 The entry points that spend nothing

| Method | What it answers |
| --- | --- |
| `find(runtime=, capability=, auth_mode=, group=, enabled=)` | "Give me connections whose runtime supports X" (D6). Returns a tuple of `Connection`. `enabled=` is not applied unless you ask: silently omitting a switched-off connection would make it look deleted. |
| `groups()` / `group(name)` | The groups and their membership, derived from the connections (ticket 1.15). |
| `select(group)` | Which member `connection="group:<name>"` would reach, **and every member it walked past, with why**. |
| `plan(connection)` | The vendor-independent launch plan: the scrubbed environment and the directives. |
| `preflight(connection, messages=, model=, schema=)` | The full preflight receipt without starting a run. Pass a system message to ask "would *this* prompt be cacheable" without running it (D20); pass a schema on `openai-sdk` to learn before the run whether it is in the strict subset. |
| `validate(connection)` | Everything checkable **offline** — no vendor package, no credential store, no subprocess. Returns a report rather than raising, because a config validator wants every problem at once. |
| `registry.support(runtime, capability)` | The tri-state cell. `registry.note(...)` gives the dated evidence behind it. |

`validate()` is **weaker than `preflight()`, deliberately and by name**. It
cannot tell you which auth mode a run would use, whether the login has expired,
or what the runtime's default model is — every one of those needs the vendor. A
green `validate()` means "this configuration is not obviously broken", not "this
will run".

Those are the read verbs. The **write** verbs live one attribute away, on
`bridge.manage` (`modelpass.manage.ConnectionManager`): `plan_connection(...)`
builds a connection and returns the pre-write receipt without writing anything,
`add_connection(plan)` writes the secret and then the connection,
`remove_connection(name)` applies the secret delete rules and says what it kept,
`rename_connection(old, new)` moves the entry with the name when it is
unambiguously that connection's, `set_enabled(name, bool)` switches runs off
without deleting anything, and `set_groups(name, groups)` replaces the groups a
connection is in. Every one returns a plain dataclass with a `to_dict()`
that is safe to print. See
[Managing connections from Python](../README.md#managing-connections-from-python).

### 1.7 Groups — addressing a set rather than a connection (ticket 1.15)

A connection may declare **groups**, and a group may then be named where a
connection name is expected:

```toml
[connections.local-ollama]
runtime = "openai-compatible"
authMode = "api_key"
baseUrl = "http://localhost:11434/v1"
credentialRef = "none"
groups = ["cheap", "offline"]
```

```python
answer = bridge.ask("group:cheap", "Classify this: ...")
```

**`default` is the name for the connections that declare no group**, and nothing
else. A connection with `groups = ["fast"]` is in `fast` and is *not* in
`default`; putting it in both is spelled by writing both. The alternative —
every connection always in `default` — would make the group carry no information
and make `group:default` mean "any connection at all", which is not a route
anybody chose.

**A group is derived, never stored.** There is no `[groups]` table and no group
object anywhere: a group exists while some connection claims it, and moving the
last member out is how it goes away. That is deliberate — a group with a
definition of its own would be a second place a connection could be taken out of
service, and there is already exactly one (`enabled = false`).

**Resolution is name order among the enabled members**, and modelpass does not
pretend that is a decision you made. `select(group)` returns the chosen
connection *and* every member it walked past with the reason, so
"`cheap` resolves to `local-ollama`; `a-gateway` was skipped because it is
disabled" is answerable before the run rather than inferred from the receipt
afterwards. `modelpass groups` prints exactly that, and so does the bench's
Groups page. A caller who needs a particular member names that member — a group
is a convenience for "any of these", not a router with a policy. **Preference
order within a group does not exist**; it is on the roadmap with throttling and
metrics, where it can be designed against the thing it will be used for.

Two refusals, and they are two on purpose:

| Situation | Error |
| --- | --- |
| Nothing claims that name | `NoSuchGroup`, naming the groups that do exist |
| It has members and every one is disabled | `GroupUnavailable`, naming them |

Both are raised before anything is planned or spent. The `group:` prefix is the
spelling rather than a bare name being tried as both, because a bare name that is
a connection *and* a group would route somewhere on a rule the caller never saw;
connection names cannot contain a colon, so the two can never collide.

**What a run is billed to is a connection.** A group is how the call was
*addressed*; the receipt, the terminal event and the run-log line all name the
connection that was actually reached.

**`onQuotaExhausted.failover` is the one field that refuses a group**, and the
refusal is at construction, where the file is written. Failover is the only path
in modelpass that moves a run onto metered billing, and under D4(d) naming the
target *is* the consent. A group names a set whose membership changes afterwards
— somebody adds a metered connection to `cheap` next month and a consent given
for one connection silently starts paying for another, which is the exact
surprise D4 exists to prevent. Name the connection you mean.

`groups` is an additive key under the store's 0.1 compatibility policy: a build
that has never heard of groups preserves it verbatim and reports it, and sees
every connection by name exactly as it did before.

### 1.8 `maxInputTokens` — a fact the connection carries, and nothing more (2026-09-21)

A connection may record the model's usable input window, in tokens:

```toml
[connections.claude-api]
runtime        = "anthropic-api"
authMode       = "api_key"
credentialRef  = "env:ANTHROPIC_API_KEY"
maxInputTokens = 200000    # optional; omit it and the window is unknown
```

It is read back as `Connection.max_input_tokens`, an `int | None`.

**Nothing in modelpass reads it.** It bounds no call, truncates no message,
refuses no run, and appears in no receipt or terminal event. It is a declaration
a *consumer* acts on, in the same way `retry = "never"` is a stance rather than a
retry loop. A retrieval evaluation harness asked for it so it can decide whether
a payload will fit before spending a call finding out; the part it could not
express anywhere was "how big is the window on **this** connection". If modelpass
ever enforces it, that is a separate contract with its own refusal, and this
paragraph is what it would have to change.

**It is supplied, never discovered.** There is no table of model names to window
sizes in this library and there is not going to be one. A vendor moves a window
without moving the model string, so a table is a stale number that reads as
authoritative — and a wrong window is worse than no window, because a consumer
that trusts it will split a payload that did not need splitting or send one that
does not fit.

| the file says | `max_input_tokens` | means |
| --- | --- | --- |
| nothing | `None` | **unknown** — nobody has told modelpass, and modelpass will not guess |
| `maxInputTokens = 200000` | `200000` | somebody stated this window |
| `maxInputTokens = 0` | — | refused at construction: `InvalidConnection` |

Unset and zero are different facts, which is why zero is refused rather than
stored. A consumer must read `is None`, not falsiness: a `None` collapsed to `0`
turns "unrecorded" into "the window is nothing", and every unconfigured
connection on the machine then looks too small for every payload. Anything that
is not a positive whole number — a negative, a float, a string, `true` — is an
`InvalidConnection` naming the connection.

`modelpass list --verbose` prints the window, and prints `unknown` when there is
none rather than leaving the row out; the bench's accounts page does the same.
An additive key under the store's 0.1 compatibility policy: a connection written
before it existed loads unchanged and is not given one when the file is
rewritten.

### 1.9 `promptCache` — asking for prompt caching, and being told what that bought (2026-09-21)

A connection may state that it wants prompt caching:

```toml
[connections.claude-api]
runtime       = "anthropic-api"
authMode      = "api_key"
credentialRef = "env:ANTHROPIC_API_KEY"
promptCache   = "default"   # optional; "default" = the vendor's own lifetime
```

Read back as `Connection.prompt_cache`, a `str | None`.

**This is the standing half of prompt caching. The per-call half already
exists** and is not replaced: `CacheControl` on a `TextBlock` (the ticket 1.5 content-block vocabulary) is
how a
caller says *where* the cacheable prefix ends, which is inherently per call
because it is a statement about one payload's shape. `promptCache` is the other
question — *do I want caching on this connection at all, and for how long* —
which is a standing policy and sits beside `retry` and `timeoutSeconds`. Both
are needed: a breakpoint with no policy was all the library had before, and on
every runtime but one a caller **cannot place a breakpoint** and the cache
happens anyway.

#### The three outcomes

Asking for caching has three possible answers, and conflating any two of them is
the way to misread this feature.

| outcome | when | what a caller sees |
| --- | --- | --- |
| **explicit** | the runtime takes an instruction (`cache_breakpoints` is `supported`) | the request is honoured in the strong sense: put `cache_control` on your content and it reaches the vendor. `Receipt.prompt_cache_disposition == "explicit"` |
| **automatic** | the runtime caches unasked and cannot be stopped (`cache_automatic` is `supported`) | **the request was already satisfied.** Not an error. Nothing is sent for it and nothing needs to be. `Receipt.prompt_cache_disposition == "automatic"` |
| **refused** | neither cell is `supported` | `InvalidConnection` when the connection is built |

`automatic` is the answer most connections get, and reporting it as a failure
would be wrong: most vendors cache prompts without markers, and on a
subscription runtime there is no way to turn it off. Reporting it as `explicit`
would be worse — a caller reading a single "caching: yes" could not tell whether
its breakpoints were doing anything.

The refusal is the case where there is nothing to honour *and* nothing already
happening, so the alternative is a setting that silently does nothing. It says
the narrow thing and not the broad one: *modelpass has not established that this
runtime caches prompts*, naming both cells and what each reads. It is **not** a
claim that the vendor does no caching.

#### Refused at construction

Everything needed to decide — the runtime, the value — is on the connection, so
the refusal happens in `Connection.__post_init__`, like an `authMode` a runtime
does not allow and like `CacheControl`'s own `ttl`. By the time a run exists,
the request is known to be meetable. The error class is the existing
`InvalidConnection`; no new error class and no new user-facing vocabulary were
introduced.

#### The lifetime, and what "unspecified" means

Three states, and they are all distinguishable:

| the file says | `prompt_cache` | `plan.ttl` | means |
| --- | --- | --- | --- |
| nothing | `None` | — | **nothing stated.** Nothing about the connection's behaviour changes. This is not "caching off" |
| `promptCache = "default"` | `"default"` | `None` | caching asked for, **lifetime unspecified** — the vendor's own applies |
| `promptCache = "1h"` | `"1h"` | `"1h"` | caching asked for at a named lifetime |

A consumer must read `is None`, not falsiness, for the same reason
`maxInputTokens` requires it.

**There is no modelpass TTL vocabulary, and that is a finding rather than a
taste.** The two runtimes that take a lifetime spell it in disjoint words —
`anthropic` 0.97.0 types it `Literal['5m', '1h']` and `openai` 2.32.0 types it
`Literal['in-memory', '24h']`. A single house vocabulary would have had to
invent an equivalence nobody published, or pick one vendor's words and
mis-describe the other. So `modelpass.prompt_cache.PROMPT_CACHE_TTLS` is per
runtime, every value in it was read off the SDK installed in this repository on
2026-09-21, and a test re-reads both literals so an SDK upgrade that moves one
fails rather than drifts. A runtime absent from that table accepts `"default"`
and nothing else, and its refusal says the checkable thing — *modelpass carries
no cache lifetime to this runtime from this key* — rather than the broader claim
that the vendor has no lever.

#### What reaches a wire

One thing: on `openai-api`, a named lifetime is sent as
`prompt_cache_retention`, which is a typed field of that SDK's Responses
request. `"default"` sends nothing, because the wire has no way to say "cache at
whatever you normally do" other than by not saying anything.

Everywhere else `promptCache` is a **declaration**, in the same way `retry =
"never"` is a stance rather than a retry loop. In particular, on `anthropic-api`
modelpass does not write the connection's lifetime onto the caller's
breakpoints: `TextBlock`'s own contract is that modelpass never invents a
breakpoint and never moves one, and filling in a `ttl` the caller left off would
be editing their content. Place it on the `CacheControl` where it belongs; what the connection
buys you is that an impossible lifetime is refused before a run rather than
after a 400.

No `prompt_cache_key` is sent either. Grouping requests so they land on the same
cache is a caching *strategy* — it depends on what a caller considers one
workload — and deriving a key from a connection name would be modelpass deciding
that for everybody.

#### Where it is reported

`Receipt.prompt_cache_requested` and `Receipt.prompt_cache_disposition`, both
`None` on every connection that stated nothing, plus one sentence in
`Receipt.notes`. `modelpass list --verbose` prints the request and its
disposition; the bench's accounts page prints the sentence, and says *nothing
stated* rather than *off* for a connection that asked for nothing. An additive
config key under the store's 0.1 compatibility policy.

---

## 2. The three vendors, and the matrix

`Runtime` has eight identities, because **capability identity is the runtime, not
the vendor** (D6):

| Runtime | Family | Adapter | Cells still awaiting the owner's live drive |
| --- | --- | --- | --- |
| `anthropic-sdk` | agent | implemented, validated live | none |
| `openai-sdk` | agent | implemented, validated live | `graceful_cancel`, `sessions_fork` (§3.2) |
| `anthropic-api` | API key | landed (§2.0a) | `ttl_control`, `midconversation_system` |
| `openai-api` | API key | landed (§2.0b) | `thinking` |
| `google-api` | API key | landed (§2.0d) | `thinking` |
| `openai-compatible` | API key, or none | landed (§2.0c) | **every cell, per install** — filled on the connection by `modelpass verify`, never in `STATIC_TABLE` |
| `google-cli` | agent | none — gated placeholder (§2.1) | the whole row |
| `google-sdk` | agent | none — gated placeholder (§2.1) | the whole row |

**The four live test files, in one place.** Each drives a real endpoint, each is
marked `live`, and all four are deselected by default. Running any of them needs
`SUBPASS_LIVE_TESTS=1` **and** that file's own variable, so having a vendor key on the
machine is never on its own enough to start a spend:

| File | Also needs | What it drives | What a green run moves |
| --- | --- | --- | --- |
| `tests/live/test_anthropic_api_live.py` | `MODELPASS_LIVE_ANTHROPIC_API_KEY` | one chat, one tool round trip, one structured output, one cache breakpoint sent twice | confirms the `anthropic-api` row; `ttl_control` needs two calls more than five minutes apart and stays open |
| `tests/live/test_openai_api_live.py` | `MODELPASS_LIVE_OPENAI_API_KEY` | one chat, one tool round trip, one structured output against a non-strict schema, one reasoning-effort call | `thinking` on `openai-api` |
| `tests/live/test_google_api_live.py` | `MODELPASS_LIVE_GOOGLE_API_KEY` — deliberately neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY`, the two names `genai.Client()` discovers by itself | the same four calls, with one thinking call | `thinking` on `google-api` |
| `tests/live/test_openai_compatible_live.py` | `MODELPASS_LIVE_COMPATIBLE_BASE_URL` (naming the endpoint is the consent) and `MODELPASS_LIVE_COMPATIBLE_MODEL` | one chat, one tool round trip, one structured-output call against your own endpoint | **nothing in this table**, by design — that runtime's cells are per connection |

Only the first three spend money; the fourth spends electricity.

### 2.0 The four API-key runtimes (2026-09-13)

An **agent runtime** is a vendor CLI or agent SDK driven as a child process: it
holds its own sessions, its own login store and its own toolbelt. An **API
runtime** is an HTTP client this process constructs with an explicit key. They
share a vendor and share nothing else.

"Plumbed, adapter pending" is a precise state, and it is worth knowing exactly
what you can and cannot do with one today:

* **You can** write the connection. The runtime parses, `baseUrl` validates,
  `modelpass check` reports it, the store round-trips it, and the preflight
  answers all three of its checks.
* **You cannot** run anything. Every capability that is not a checked absence
  reads `unverified`, including `chat`, so `bridge.chat()` refuses with a
  capability error naming the cell. That is deliberate: this table's rule is that
  **a cell moves only with a recorded drive**, no adapter has been written, and
  a `supported` cell sourced from a vendor's documentation is exactly the
  dishonesty the tri-state exists to prevent. The cells move in the adapter
  tickets — 1.10 and 1.11 for the two that are still waiting.

**All four left that state on 2026-09-13** and are described in §2.0a through
§2.0d below — `google-api` last, in ticket 1.11, which is why the paragraph above
is now a description of a state no shipped runtime is in. It is kept because it
is what the tri-state means and what the next runtime will start life as.
`anthropic-api` is the worked example of what
an adapter ticket costs and of where the line between "checked" and "looked
right" falls; `openai-api` is the same pattern applied a second time, and the
place to look for what that line does when the two vendors disagree.

Three things about them that are settled now rather than later:

* **`baseUrl` is a connection field**, wire key `baseUrl`. **Required** on
  `openai-compatible`, which names an API *shape* (Ollama, LM Studio, OpenRouter,
  a LiteLLM proxy, a gateway) and has no default endpoint. **Optional** on the
  other three, where it is the lever for a proxy or a regional endpoint.
  **Refused** on the agent runtimes, which read their endpoint from their own
  configuration — `ANTHROPIC_BASE_URL` through `allowEnv` is that path. It must
  be `https`, or `http` to `localhost` / `127.0.0.1` / `::1`; it may carry no
  credentials, no query string and no fragment.
* **`openai-compatible`'s vendor is `compatible`, not `openai`.** An Ollama box
  is not an OpenAI account, and `bridge.find(vendor="openai")` must not return
  one. `VENDOR_OF` in `runtimes.py` is the single source of that answer.
* **The preflight launches nothing, so it plans no environment.** `plan.env` is
  empty on these runtimes, on purpose: a scrubbed copy of `os.environ` that
  nobody launches invites an adapter to hand it to an SDK constructor, whose own
  discovery would then read a key the connection never named. What replaces it is
  D2's in-process clause — the credential is resolved explicitly and handed to
  the client constructor, and the SDK never reads the environment — carried on
  the receipt as two directives. The three checks are: the credential resolves to
  a non-empty value, the base URL parses and is safe, and optionally a cached
  token-free model-list probe.

`openai-compatible` is `unverified` **by construction** and stays that way: two
installs can legitimately disagree about every cell, because what answers is
whatever the base URL points at. `CapabilityRegistry.refine()` is the per-install
upgrade path, which is what that hook was built for, and `modelpass verify
<connection>` is the command that drives it (§2.0c).

### 2.0a `anthropic-api` — the first one with an adapter (ticket 1.6, 2026-09-13)

Status: **adapter landed; live cells pending owner drive.** Install it with the
`anthropic-api` extra (`pip install "modelpass[anthropic-api]"`, which pins
`anthropic>=0.40,<1`; the adapter was written against **0.97.0**).

```python
bridge.chat(connection="claude-api", message="...")            # streams
bridge.chat(connection="claude-api", message="...", tools=[t]) # loop runs in the adapter
bridge.chat(connection="claude-api", message="...", schema=s)  # native structured output
```

**What is the same as `anthropic-sdk`, deliberately.** The event vocabulary and
the token mapping. `cache_read_input_tokens` lands in `cached_input_tokens` and
`cache_creation_input_tokens` in `cache_write_tokens`, counted *beside*
`input_tokens` rather than inside it — the same four assignments, asserted side
by side in `tests/test_adapter_anthropic_api.py`, so a consumer's accounting, run
log and guards do not learn that a run changed runtimes.

**What is different, and why.**

* **The adapter owns the tool loop.** `while stop_reason == "tool_use"`, each
  `ToolDef.handler` called in *this* process, results fed back as `tool_result`
  blocks, `tool_call` / `tool_result` emitted as the observations they already
  are. **No turn cap** — the guard's `stopAtTokens` is the bound. A raising
  handler becomes a failed tool result, so the model can retry or explain rather
  than the run dying. D12 is unchanged: `ToolCallEvent` still has no reply path,
  because nothing outside the adapter answers one.
* **Usage arrives per response**, so a token guard stops the loop *between*
  rounds — before the next round's tools run. Better granularity than either
  agent runtime; `interim_usage` is `supported` here and means it.
* **`cache_control` breakpoints reach the vendor intact.** This is the runtime
  the content-block vocabulary of ticket 1.5 exists for. `cache_breakpoints` is
  the first `supported` cell for it in the table.
* **Structured output is native** (`output_config={"format": {"type":
  "json_schema", ...}}` in SDK 0.97.0), so the schema-as-forced-tool fallback was
  not built. No strict-subset restriction, unlike `openai-sdk`.
* **Sessions are refused**, naming `bridge.chat(history=[...])` as the multi-turn
  shape against a stateless runtime. There is no vendor-held history to resume.
* **`max_tokens` is required by the API**, so the adapter always sends one:
  `options={"max_output_tokens": N}`, else 4096. That option is a stopgap —
  ticket 1.7 makes the ceiling a first-class request field, and
  `AnthropicAPIAdapter.sampling_params` is the single hook it fills.

**What "driven" means on this row, because it is not what it meant on the agent
runtimes.** Every cell that moved was moved on two things held together: a
fake-transport test showing what modelpass does, and the installed SDK's own
typed surface showing the vendor takes it. Cells needing a *live* observation
neither half can produce stay `unverified`, and each names the live test that
moves it. The live file is `tests/live/test_anthropic_api_live.py` — marker
`live`, deselected by default, gated on `SUBPASS_LIVE_TESTS=1` **and** on
`MODELPASS_LIVE_ANTHROPIC_API_KEY` (a variable of its own, so merely having a key
on the machine never starts a spend). It drives four calls: one short chat, one
tool round trip, one structured output, and one cache breakpoint sent twice with
`cache_read_input_tokens > 0` asserted on the second.

Two cells are still open on purpose and are worth knowing about (`sampling_controls`
and `max_output_tokens` were the third pair, and ticket 1.7 moved both to `supported`
— see §2.3):

| Cell | Why it is still `unverified` |
| --- | --- |
| `ttl_control` | Changed *from* `unsupported`: SDK 0.97.0 types `cache_control.ttl` as `Literal['5m','1h']`, so the shared "no API has a TTL field" is false here, and modelpass carries a caller's ttl through. It is not `supported` because a lifetime cannot be observed by any test in this suite — proving a 1h pin needs two calls more than five minutes apart. |
| `midconversation_system` | The vendor feature is real and model-gated; **modelpass** does not offer it. `RunRequest.system_blocks` collects every system message wherever it sits and sends the lot as top-level `system=`, so a mid-conversation one is hoisted — which changes its meaning and invalidates the prefix it was meant to preserve. Needs a per-message split *and* a live drive. |

### 2.0b `openai-api` — the same pattern, a second vendor (ticket 1.9, 2026-09-13)

Status: **adapter landed; `thinking` pending owner drive.** Install it with the
`openai-api` extra (`pip install "modelpass[openai-api]"`, which pins
`openai>=2,<3`; the adapter was written against **2.32.0**). Note that this is a
*different* extra from `openai`, which installs `openai-codex` for the agent
runtime: two OpenAI runtimes, two packages, one name each.

```python
bridge.chat(connection="openai-key", message="...")            # streams
bridge.chat(connection="openai-key", message="...", tools=[t]) # loop runs in the adapter
bridge.chat(connection="openai-key", message="...", schema=s)  # strict json_schema
```

**It speaks the Responses API, and the reason is the event vocabulary.** The
short version is that `thinking`, the terminal mapping and `max_output_tokens` are all
answerable on Responses and the first of them is not answerable at all on Chat
Completions, which reports reasoning as a token count and no text. There is no
option to select the other transport. That shape belongs to `openai-compatible`
(ticket 1.10), which is a different runtime identity on purpose.

**What is the same as `anthropic-api`, deliberately.** The event vocabulary, the
tool loop's three rules (no turn cap, a raising handler becomes a failed tool
result, cancellation checked between rounds), the receipt fields, the session
refusals, and the credential handling — an explicit `api_key=` from the
connection, so an ambient `OPENAI_API_KEY` can never bill an account nobody
named. `tests/test_adapter_openai_api.py` is deliberately the same file, section
by section, as `tests/test_adapter_anthropic_api.py`.

**Where the two vendors differ, and where each difference lands.**

* **Usage arithmetic.** OpenAI reports `input_tokens` with the cached prefix
  *inside* it and `input_tokens_details.cached_tokens` beside it; `TokenUsage`
  counts `cached_input_tokens` **separately from** `input_tokens`. So the adapter
  **subtracts**, and a call both vendors would describe as "100 prompt tokens, 30
  of them cached" lands on the same `TokenUsage` either way.
  `cache_write_tokens` is always `0` — OpenAI's prefix cache is automatic and
  bills no write premium, so there is no number to report.
  `output_tokens_details.reasoning_tokens` stays folded **inside**
  `output_tokens`, where the vendor already counts it and where Anthropic counts
  thinking output.
* **No `cache_control`, so the prompt is flattened.** `instructions=` is a
  string, not a block array. System blocks are joined with `flat_text` semantics
  (one blank line between them) and their breakpoints are dropped — named on the
  receipt by the bridge's own R3 disclosure, exactly as ticket 1.5 defined.
  `cache_breakpoints` is `unsupported` here and has been since 1.5. Prefix
  *stability* is the whole of the caching control available.
* **Structured output must be strict.** `text={"format": {"type": "json_schema",
  "strict": true, ...}}`, and the strict subset has no optional properties. The
  caller's schema is run through `schema.to_openai_strict()` rather than refused,
  and `schema.openai_strict_issues()` is reported **as receipt notes before the
  call** — so "your optional `note` will come back present and possibly null" is
  something you read in the preflight rather than discover in the answer.
* **No default ceiling.** `max_output_tokens` is optional on this API, so
  modelpass sends none when nobody asked. The opposite of `anthropic-api`, where
  the field is required and 4096 is supplied and reported. Inventing one here
  would be modelpass truncating answers nobody asked it to truncate.
* **No `top_k` anywhere.** It is in neither of the SDK's create signatures, so
  `sampling_rules` does not accept it: a caller who sends one gets it dropped
  with a named note rather than a vendor error.
* **Reasoning summaries are opt-in.** `options={"reasoning_summary": "auto"}`
  fills `reasoning.summary`. modelpass does not ask on its own, because a summary
  is billed output nobody requested.

**The open cells, and the one that matters.**

| Cell | Why it is still `unverified` |
| --- | --- |
| `thinking` | **The one a live drive moves.** Half is checked: both reasoning deltas are typed members of `ResponseStreamEvent` and the adapter maps both to `ThinkingEvent`. The half a fake cannot supply is whether anything ever arrives — a summary is emitted only when asked for, and raw reasoning text is gated on the organisation's verification status. `tests/live/test_openai_api_live.py::test_one_reasoning_effort_call` is the drive. |
| `tools`, `mcp` | The *coarse* cells, about the vendor's own server-side tools (`web_search`, `file_search`, `code_interpreter`, the MCP tool type). Their stream events reach a caller as vendor events, but the adapter never sends them and nobody has driven them, so `unsupported` would be a claim about those too. The actionable cells — `tools_in_process` `supported`, `mcp_servers` `unsupported` — are both settled. |
| `midconversation_system` | modelpass's limitation, not the vendor's: `RunRequest.system_blocks` hoists every system message into `instructions=`. The Responses input array could carry one in place; the per-message split in modelpass is the missing half. |

The live file is `tests/live/test_openai_api_live.py` — marker `live`, deselected
by default, gated on `SUBPASS_LIVE_TESTS=1` **and** on
`MODELPASS_LIVE_OPENAI_API_KEY` (a variable of its own, so merely having a key on
the machine never starts a spend). It drives four calls: one short chat, one tool
round trip, one structured output against a schema that is *not* already strict,
and one reasoning-effort call that asks for a summary.

### 2.0c `openai-compatible` — one shape, many endpoints, one drive each (ticket 1.10, 2026-09-13)

Status: **adapter landed; every capability cell is a question about *your*
endpoint until you answer it.** Install it with the same `openai-api` extra
(`pip install "modelpass[openai-api]"`): it is the same `openai` HTTP client
pointed at somebody else's server, and a second extra would install one wheel
under two names.

```bash
# A local Ollama, which authenticates nobody.
modelpass connect compatible --base-url http://localhost:11434/v1 --model llama3.1
modelpass verify compatible-api        # <- the step this runtime cannot skip
```

```python
bridge.chat(connection="compatible-api", message="...")            # after verify
bridge.chat(connection="compatible-api", message="...", tools=[t]) # if verify said so
bridge.chat(connection="compatible-api", message="...", schema=s)  # if verify said so
```

**It speaks Chat Completions, and the reason is the opposite of `openai-api`'s.**
The short version is that §2.0b asked *which surface says more* and this section asks
*which surface is actually implemented by the thing behind the URL*. Ollama, LM
Studio, vLLM, llama.cpp, a LiteLLM proxy and OpenRouter all implement
`POST /v1/chat/completions`; Responses is OpenAI's and a minority of the rest.
An adapter defaulting to Responses would 404 against the commonest case.

**Responses is selectable per connection**: `options={"wire": "responses"}` runs
§2.0b's code path unchanged — the adapter is a subclass of `OpenAIAPIAdapter` and
that branch is one `super().run()`. Use it against a gateway that implements
Responses; drop it if the call 404s. The option says which of modelpass's paths
runs and claims nothing about the endpoint, so `support_for()` returns "no
opinion" for it.

**Three things that differ from every other runtime.**

* **A connection may declare no credential.** `credentialRef = "none"` — a
  credential kind accepted here and refused everywhere else — says the endpoint
  authenticates nobody, which is the plain truth about a local box. The receipt
  says so in those words, and the client is still constructed explicitly with a
  visible placeholder in the `api_key=` slot the SDK requires. `env:` and
  `secret:` work exactly as they do elsewhere, and a proxy or gateway will want
  one. `modelpass connect compatible` with no key flags writes this shape.
* **Usage may simply not arrive.** `stream_options={"include_usage": True}` is
  sent on every call, and a server is free to ignore it. Where none comes back
  the adapter emits **no usage event at all** rather than a zeroed one: "nobody
  told us" and "this run cost nothing" are different claims and only one is true.
* **`top_k` is dropped and named.** It is in neither of `openai` 2.32.0's create
  signatures; the servers that take one disagree about where it goes (Ollama in
  its own options block, vLLM at the top level). modelpass will not guess an
  `extra_body` key that would ride along on every endpoint that does not take
  one. `temperature`, `top_p` and `max_output_tokens` are sent —
  `max_output_tokens` as **`max_tokens`**, the spelling every compatible server
  has implemented since the beginning.

**Structured output is where endpoints differ most.** `response_format:
{"type": "json_schema", ...}` is the least widely implemented thing this adapter
sends. A server that refuses it raises `StructuredOutputRejected` — a
`CapabilityNotSupported` subclass, so a consumer already matching that name
catches it — whose message says what to try. modelpass deliberately does **not**
fall back to `response_format: {"type": "json_object"}`: that constrains the
answer to be JSON without constraining it to be your schema, and delivering
unvalidated JSON where a schema was asked for is the silent downgrade D13 exists
to refuse.

#### `modelpass verify` — the per-install drive

**`openai-compatible` is `unverified` by construction, so an unverified
connection cannot chat.** `bridge.chat()` refuses with a capability error naming
`modelpass verify`. That is the tri-state working, not a gap: the alternative is
modelpass claiming an arbitrary URL can hold a conversation because its scheme
parsed.

`modelpass verify <connection>` is the answer, and it is the only command in
modelpass that spends in order to learn something — it says so before it does.
Four steps in increasing order of cost:

| Step | What it costs | What it decides |
| --- | --- | --- |
| `models.list()` | nothing | is anything there, and what does it serve. A failure here ends the drive and **writes nothing**: a record saying "driven, and it can do nothing" about a box that was switched off is the worst lie this file could hold |
| one short chat | ~60 tokens | `chat`, `streaming`, `incremental_text`, `usage_tokens` |
| one tool round trip | ~2 rounds | `tools_in_process` |
| one structured-output call | ~60 tokens | `structured_output` — and `False` on a rejection, which is a *finding* |

The verdicts are written to the connection under the additive store key
`verifiedCapabilities` (store policy 0.1: an older build preserves it verbatim
and reports it rather than dropping the evidence):

```toml
[connections.compatible-api.verifiedCapabilities]
supported = ["chat", "incremental_text", "streaming", "tools_in_process", "usage_tokens"]
unsupported = ["structured_output"]
checkedAt = "2026-09-13T18:51:00+00:00"
models = ["llama3.1", "qwen2.5"]
```

`Bridge.registry_for(connection)` folds them into a registry **copy** through
`CapabilityRegistry.refine()` — the hook D6 has carried since Anthropic's init
array and which had exactly one caller until now. `STATIC_TABLE` is never
touched. Two installs of this runtime can legitimately disagree about every cell;
if a drive passes against your Ollama and fails against a colleague's LM Studio,
both runs were correct.

**What the drive deliberately does not decide**, and the line is 1.6's line
between "checked" and "looked right": `interim_usage` (one short call cannot tell
usage-during from usage-at-the-end), `thinking` (a server that emitted no
reasoning on one prompt has not been shown unable to), and the two sampling cells
(a server that accepted `temperature` has not shown that it applied it). Those
stay where the static table left them, and the `verify` output says so.

The live file is `tests/live/test_openai_compatible_live.py` — marker `live`,
deselected by default, gated on `SUBPASS_LIVE_TESTS=1` **and** on
`MODELPASS_LIVE_COMPATIBLE_BASE_URL` (naming the endpoint is the consent, the way
a key variable is on the metered files) plus `MODELPASS_LIVE_COMPATIBLE_MODEL`.
It spends nothing but electricity against a local Ollama, and a green run moves
nothing in the shared table — by design.

### 2.0d `google-api` — the fourth vendor, and the last one (ticket 1.11, 2026-09-13)

Status: **adapter landed; `thinking` pending owner drive.** Install it with the
`google-api` extra (`pip install "modelpass[google-api]"`, which pins
`google-genai>=1,<2`; the adapter was written against **1.73.1**). That is
`google-genai`, the current SDK — never the retired `google-generativeai`. There
is no `google` extra beside it, because the two Google *subscription* runtimes
have no adapter at all; §2.1 is why that asymmetry is the decision rather than an
oversight, and this ticket changed nothing about it.

```bash
modelpass connect google --api-key-stdin       # writes a google-api connection
```

```python
bridge.chat(connection="gemini-api", message="...")            # streams
bridge.chat(connection="gemini-api", message="...", tools=[t]) # loop runs in the adapter
bridge.chat(connection="gemini-api", message="...", schema=s)  # native json schema
```

**What is the same as the three before it, deliberately.** The event vocabulary,
the tool loop's three rules (no turn cap, a raising handler becomes a failed tool
result, cancellation checked between rounds), the receipt fields, the session
refusals, and the credential handling — an explicit `api_key=` from the
connection. That last one matters more here than anywhere else: `genai.Client()`
discovers **two** ambient names, `GEMINI_API_KEY` *and* `GOOGLE_API_KEY`, so an
adapter that let the SDK find its own key would have two ways to bill an account
nobody named. `tests/test_adapter_google_api.py` is the same file, section by
section, as `tests/test_adapter_openai_api.py`.

**Where this vendor differs, and where each difference lands.**

* **The stream is a sequence of whole responses, not typed events.**
  `models.generate_content_stream(...)` yields `GenerateContentResponse` chunks
  carrying whatever parts have arrived; there is no `type` string to match on, so
  the mapping reads *parts*. Anything without a word in the vocabulary becomes a
  vendor event named `part.<field>` rather than `stream.<type>`.
* **A thought is a flag on a text part.** `Part.thought` is a boolean and the
  thought itself arrives in the ordinary `text` field — so a reader that ignores
  the flag streams the model's private reasoning into the answer a user reads.
  modelpass separates them: a thought part becomes a `ThinkingEvent`, is never a
  `text_delta`, and is never the text a schema is validated against.
* **The assistant turn is called `model`.** A conversation sent with
  `role="assistant"` is not one this API understands.
* **Tool results are `function_response` parts in a `user` turn**, sent after the
  model's own turn is replayed verbatim — `thought_signature` included, for the
  same reason `openai-api` replays reasoning items. `FunctionCall.args` arrives
  already parsed (no JSON string to decode) and `FunctionCall.id` is optional and
  usually absent, so modelpass falls back to the function's name as the event id.
  Parallel calls are answered in **one** user turn.
* **Usage arithmetic, and this is the mapping worth reading.**
  `prompt_token_count` includes the cached content, so the adapter **subtracts**
  `cached_content_token_count` from it — the same correction `openai-api` makes.
  And `thoughts_token_count` is reported *beside* `candidates_token_count` rather
  than inside it (the SDK documents `total_token_count` as the sum of the four),
  so the adapter **adds** it into `output_tokens`: on every other runtime the
  thinking tokens are already inside the vendor's own output count, and leaving
  them out here would make a Gemini run's `output_tokens` mean something narrower
  than everyone else's. The result is that `TokenUsage.total_tokens` equals the
  vendor's own total. `cache_write_tokens` is always `0`.
* **`top_k` is accepted**, which neither OpenAI runtime does. It is a
  `GenerateContentConfig` field, it is sent, and nothing is dropped.
* **The reasoning dial becomes a *level*, never a budget.**
  `ThinkingConfig.thinking_level` is `MINIMAL | LOW | MEDIUM | HIGH` and
  modelpass's dial is `low | medium | high`, so the three positions map onto three
  of the vendor's own named values with nothing invented in between.
  `thinking_budget` — the other field — is a token count whose allowed range the
  SDK documents as *model dependent*, and ticket 1.7 refused to invent one of
  those for Anthropic. modelpass sends none, and that is the recorded decision.
* **Thoughts are opt-in.** `options={"include_thoughts": True}` fills
  `thinking_config.include_thoughts`. modelpass does not ask on its own, because
  thought output is billed output nobody requested — 1.9's `reasoning_summary`
  decision in this vendor's spelling.
* **Structured output needs no rewrite.** `response_mime_type:
  "application/json"` plus `response_json_schema: <your schema>` (the mime type is
  required when a schema is sent, and `response_schema` — the vendor's own
  `Schema` object — must then be omitted). There is no `strict` flag, so an
  optional property stays optional and there is nothing to disclose, which is the
  opposite of §2.0b.
* **No `cache_control`, and Gemini's reason is its own.** Implicit caching is
  automatic and unaddressable; explicit caching is a **separate resource** —
  `client.caches.create()` returns a named object with a TTL that a later request
  references through `config.cached_content`. An Anthropic-shaped breakpoint has
  nowhere to go in either, so the system prompt is flattened into
  `config.system_instruction` with `flat_text` semantics and the drop is named on
  the receipt by the bridge's own R3 disclosure. **modelpass does not create
  cached-content objects**: doing so would make the library the owner of a billed,
  expiring, server-side resource nobody asked it to create. That is why
  `ttl_control` is `unsupported` here for a different reason from the other rows.
* **The status code is spelled `code`.** `google.genai.errors.APIError` carries
  `code` (and `status`, the RPC name) rather than `status_code`. A 429 is still
  `quota_exhausted`, typed off the number and never matched on the message.
* **A blocked prompt arrives as a *successful* response with no candidates**, in
  `prompt_feedback.block_reason`. It is read separately from `finish_reason` and
  reported ahead of it, because a run that said `ok` there would be reporting an
  empty success.

**The open cells, and the one that matters.**

| Cell | Why it is still `unverified` |
| --- | --- |
| `thinking` | **The one a live drive moves.** Half is checked: `Part.thought` is typed, and the adapter separates thoughts from the answer. The half a fake cannot supply is whether one ever arrives — thoughts are emitted only when asked for and only by a model that thinks. `tests/live/test_google_api_live.py::test_one_thinking_call` is the drive. |
| `mcp_servers` | **A cell that moved the *other* way in this ticket**, from a checked absence back to an open question — the same correction ticket 1.6 made to `anthropic-api`'s `ttl_control`. The shared "an API endpoint runs no MCP client" note is false for this vendor: `Tool.mcp_servers` is `list[McpServer]`, a per-run declaration asking Gemini to connect to a server itself. modelpass never sends it and nobody has driven it, so it is neither a yes nor a checked no. |
| `tools`, `mcp` | The *coarse* cells, about the vendor's own server-side tools (`google_search`, `code_execution`, `url_context`, `file_search`) and its MCP tool type. The adapter never sends them, so `unsupported` would be a claim about those too. The actionable cell — `tools_in_process` — is `supported`. |
| `midconversation_system` | modelpass's limitation, not the vendor's: `RunRequest.system_blocks` hoists every system message into `config.system_instruction`. |

The live file is `tests/live/test_google_api_live.py` — marker `live`, deselected
by default, gated on `SUBPASS_LIVE_TESTS=1` **and** on
`MODELPASS_LIVE_GOOGLE_API_KEY` (a variable of its own, deliberately neither of
the two names the SDK would find by itself). It drives four calls: one short chat,
one tool round trip, one structured output against a schema that is not strict,
and one thinking call that asks for thoughts.

### 2.1 Google is excluded as a *subscription* runtime

**`google-api` is not.** The asymmetry is deliberate and is the D5 amendment of
2026-09-13: the Antigravity Additional Terms govern the *subscription*, while a
Gemini API key falls under the Google Cloud terms instead — the Additional Terms
say so themselves — and Google's own SDK cannot authenticate against the
subscription at all. Gating `google-api` would gate something nobody prohibited.
Do not "fix" the inconsistency in either direction without re-reading
[legal/google.md](legal/google.md) and the primary terms.

`GOOGLE_CLI` and `GOOGLE_SDK` exist in `src/modelpass/runtimes.py` and sit in
`EXPERIMENTAL_RUNTIMES`, the frozenset of runtimes that require an explicit
opt-in before use (D5). Requesting one raises.

**Read the scope of that exclusion precisely: it is about the *subscription*, not
about Google as a vendor and not about Gemini as a model family.** The gate is
the mechanism; the reason is the terms. What follows is the project's own
plain-language reading, **not legal advice** — Google's own terms are the only
authority and Google may change them at any time, without notice. The fuller
statement, with primary links, is [legal/google.md](legal/google.md).

The [Antigravity terms](https://antigravity.google/terms) state that using third
party software, tools or services to access the Service "is a breach of this
Agreement", with breaches grounds for suspension or termination of the account —
and the terms name a third-party tool as their example. **Re-verified against the
primary source on 2026-08-31: unchanged, still no carve-out.**

**"But Google ships an SDK — doesn't that contradict the ban?"** No, and the
resolution is that the SDK and the subscription are two different products under
two different agreements:

* The clause above lives in the Antigravity **Additional Terms**, which govern the
  **subscription** service (Google AI Pro / Ultra quota, reached by Antigravity
  OAuth). That is where the third-party-access ban applies.
* **API-key access is governed by different terms**, and the Additional Terms say
  so themselves: a Gemini Enterprise Agent Platform API key holder falls under
  the Google Cloud terms *instead of* those Additional Terms. Metered Gemini
  through an API key is an ordinary commercial product and is not implicated by
  that clause at all.
* Google's Python SDK targets **that** path — it requires `GEMINI_API_KEY` or
  Vertex ADC and **cannot authenticate against the subscription**. So the SDK's
  existence says nothing about subscription access, because the SDK cannot reach
  the subscription.

Why modelpass excludes the subscription runtime, in order:

1. **The only subscription surface would be subprocessing the `agy` CLI** — which
   is exactly the ambiguous third-party-access case. modelpass is unambiguously
   third-party software, and whether wrapping Google's own binary counts as
   "third party software ... accessing the Service" is at best unclear.
2. **The stakes are a stranger's Google account being terminated**, and that
   fails this project's bar. Ambiguity is survivable when the downside is a
   failed run; it is not when the downside is somebody else's account.
3. **Google's SDK structurally cannot use the subscription anyway**, so there is
   no second, cleaner route to build instead.

**Contrast Anthropic, which is supported precisely because it *does* have the
carve-out Google lacks**: Anthropic's terms prohibit a *specific shape* —
developers routing requests through other users' consumer credentials — while
explicitly covering individual Agent SDK use under one's own plan. The permitted
shape is written down affirmatively. Google's sentence is a blanket prohibition
with no equivalent.

The two identities stay in the codebase as gated placeholders whose capability
rows report honestly, so that **if Google ever publishes an Anthropic-style
carve-out**, the adapter work re-opens against a changed fact rather than a
changed design.

**A metered `api_key`-mode Gemini adapter is not prohibited — it is simply
unbuilt.** Nothing in the reading above stands against it; the blocker is demand,
not legality. Until one exists, use Google's metered APIs (Gemini API key, or
Vertex) under their own terms and billing, outside modelpass. What you should not
do is wire a *subscription* through community auth plugins: that pattern is the
named breach example.

### 2.2 The capability matrix

Reproduced from `STATIC_TABLE` in `src/modelpass/capabilities.py`. Query it in
code with `bridge.registry.support(runtime, capability)`; the dated evidence for
any cell is `bridge.registry.note(runtime, capability)`.

| Capability | `anthropic-sdk` | `openai-sdk` |
| --- | --- | --- |
| `chat` | supported | supported |
| `streaming` | supported | supported |
| `thinking` | supported | **supported** (summaries, not raw reasoning) |
| `usage_tokens` | supported | supported |
| `sessions_resume` | supported | supported |
| `sessions_fork` | supported | **unverified** |
| `subagents` | supported | unsupported |
| `mcp` | supported | supported |
| `tools` | supported | supported |
| `mcp_servers` | supported | **unsupported** (exec-only) |
| `tools_in_process` | supported | **supported** |
| `interim_usage` | supported | **supported** |
| `structured_output` | supported | supported |
| `incremental_text` | supported | **supported** |
| `ephemeral_multi_turn` | supported | unsupported |
| `system_prompt_replace` | supported | **supported** |
| `midconversation_system` | unsupported | unsupported |
| `sessions_list` | supported | **supported** |
| `ttl_control` | supported | unsupported |
| `cache_automatic` | **supported** | **supported** |
| `resume_carries_system_prompt` | unsupported | **supported** |
| `graceful_cancel` | supported | **unverified** |
| `subscription_auth` | supported | supported |
| `api_key_auth` | supported | supported |
| `sampling_controls` | **unsupported** | **unsupported** |
| `max_output_tokens` | **unsupported** | **unsupported** |

**The four API runtimes are not reproduced here**, because one of the four rows
is `unverified` by construction and a table of `unverified` would read as a
table.
`anthropic-api` (§2.0a), `openai-api` (§2.0b) and `google-api` (§2.0d) have
adapters and have their open cells listed in those sections.
`openai-compatible` (§2.0c) has an adapter and a row that stays `unverified` **by
construction**: its cells are filled per connection by `modelpass verify`, never
in this table. Ask the
registry — `bridge.registry.row(runtime)` and `bridge.registry.note(runtime,
capability)` — rather than this page.

`sampling_controls` and `max_output_tokens` are the R5 cells, added 2026-09-13
and filled by ticket 1.7 — see §2.3. `cache_automatic` is the 2026-09-21 cell —
see §1.9 for what it means and §2.4 for what moved it.

On the two agent runtimes they are **`unsupported`, as a checked absence**: no
`temperature`, `top_p`, `top_k` or output-length parameter appears anywhere in
`adapters/anthropic.py`, `adapters/openai.py` or `adapters/codex_appserver.py`,
and neither CLI protocol carries one. These are coding agents, not completions
endpoints. `anthropic-api`, `openai-api` and `google-api` are all `supported`;
`google-cli` and `google-sdk` stay `unverified` because nobody has driven them, and
`openai-compatible` stays `unverified` in this table by construction — ticket 1.10
drove its sampling row and what a given endpoint accepts is still a per-install answer.

A runtime-level `supported` means **the field exists on that runtime's wire
protocol** and never "your model will honour it". That second question is per
model, and it is answered by the receipt rather than by the table.

Two readings of that table that are easy to get wrong:

* **`resume_carries_system_prompt` inverts.** `unsupported` on Anthropic is not a
  deficiency — it is why `resume_chat(system_prompt=...)` is *required* there and
  *refused* on Codex. See §1.3.
* **Several `openai-sdk` rows are transport-dependent, and on 2026-08-31 they
  moved.** The cells describe the **default** transport, and the default became
  `codex app-server` that day. Six moved up — `thinking`, `tools_in_process`,
  `interim_usage`, `incremental_text`, `system_prompt_replace`, `sessions_list`
  — and **one moved down**: `mcp_servers`, which is a `codex exec` capability
  with no driven equivalent on the new default. None of that was new evidence
  about the vendor; it is the same 2026-08-31 evidence (§3.2) read against a
  different default, which is exactly why the rule is that **a row and the
  default flip together, in the same commit** — that commit *is* the
  re-verification, because it is the commit in which a cell becomes true (or
  false) of a caller who selects nothing.
* **`interim_usage` is the one with teeth.** `thread/tokenUsage/updated` arrives
  after each model response, mid-turn, so a `stopAtTokens` on Codex is now a real
  circuit breaker rather than a post-hoc report. If you chose a Codex ceiling
  while it could never fire, re-read the number ([guards.md](guards.md)).
* **`thinking` carries its caveat inside the yes.** What arrives is the reasoning
  *summary* stream; raw `item/reasoning/textDelta` did not fire even at
  `effort: "high"` with `summary: "auto"`. Promising your users "the model's
  reasoning" would overstate what the wire carries.
* **The cell is not the gate.** The registry answers *what can I rely on by
  default*, which is what `find()` and `registry.support()` are for;
  `bridge.chat()`, the session constructors and `list_sessions()` ask the adapter
  whether *this request* will work. Since the flip that hook answers by
  **narrowing**: `options={"transport": "exec"}` makes the six cells above false
  for that request and `mcp_servers` true, which is the mirror of what it did
  before. A caller who opted out gets a refusal naming the default they opted out
  of. The migration is recorded in
  the 2026-08-31 transport migration (recorded internally, not published).

Cells the README elaborates on with the consequences for what you can promise
your users — built-in tools, interim usage as a circuit breaker vs a report, the
structured-output schema subset — are in
[../README.md#capabilities-per-runtime](../README.md).

---

### 2.2a Where the cells came from: the verification record

No cell in this document moved on a reading of the vendor's marketing. Each one
was moved by a dated pass against an artifact — the installed SDK's own typed
surface, the shipped binary, the vendor's primary documentation — or by a live
drive that cost real allowance, and the evidence was recorded at the time.

The write-ups themselves are kept beside the code and not published: they quote
private consumer repositories, unshipped work and one machine's environment.
What follows is the part that supports a claim you can act on — **what was
checked, against which build, on what date, and the measurement the cell rests
on.** The per-cell version of the same thing is
`bridge.registry.note(runtime, capability)`, which is machine-readable and
travels with the install.

| Date | Pass | Against | What it settled |
| --- | --- | --- | --- |
| 2026-08-15 | Twelve fields per vendor | Primary vendor documentation | The static table's first fill, the three auth-mode traps, and why Google is not a subscription runtime |
| 2026-08-16 | Claude Agent SDK re-read | `claude-agent-sdk` 0.2.139 / Claude Code 2.1.63, installed | The env-scrub deviation, `graceful_cancel`, `RateLimitEvent`, `apiKeySource` |
| 2026-08-16 | Codex cancellation, driven live | `codex-cli` 0.117.0, ChatGPT login | `graceful_cancel: unsupported` on `openai-sdk`, and what a killed run leaves behind |
| 2026-08-16 | MCP and custom tools | Both installed artifacts, plus token-free live `codex mcp` runs | `mcp_servers` and `tools_in_process` on both runtimes |
| 2026-08-17 | Structured output, driven live | Both runtimes, on real logins | `structured_output` on both, and the schema subset they disagree about |
| 2026-08-30 | `command_execution` captured live | `codex exec --json` | The item's real field names, the fourth `status`, and one bug |
| 2026-08-31 | App-server transport, driven live | `codex` 0.151.0-alpha.7.1 | `system_prompt_replace`, the tool-belt switch, and the six cells the default flip moved (§3.2) |
| 2026-09-13 | The four API runtimes | `anthropic` 0.97.0, `openai` 2.32.0, `google-genai` 1.73.1, typed surfaces | §2.0a–§2.0d, including every cell those sections leave open |
| 2026-09-21 | `cache_automatic` | `openai` 2.32.0 typed surface; two recorded captures already in the tree | The one new cell, on three rows; five rows left `unverified` (§2.4) |

#### The auth-mode findings, which are the reason this library exists

**Anthropic fails *toward* metered.** Subscription OAuth from `/login` resolves
**last** in Claude Code's credential precedence, below cloud-provider variables,
`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, `apiKeyHelper` and
`CLAUDE_CODE_OAUTH_TOKEN` (2026-08-15). A stray API key in the environment
therefore wins silently and moves the run onto metered billing. Declining to set
a key is not enough; the adapter has to remove one.

**And it cannot remove one through the SDK.** Read off
`claude-agent-sdk` 0.2.139's own transport on 2026-08-16:
`ClaudeAgentOptions.env` is merged **on top of** a full `os.environ` snapshot, so
it can only add. That is why `adapters/anthropic.py` removes the planned names
from `os.environ` itself for the duration of the spawn, under a module lock,
restoring them in a `finally` — and why it recomputes them against the live
environment rather than trusting the plan. Setting a name to `""` was rejected:
an empty `ANTHROPIC_API_KEY` still occupies its precedence slot and
authenticates as an empty key. The residual risk is stated rather than hidden —
another thread mutating `os.environ` during the spawn window is a race this
design cannot close.

**OpenAI fails toward metered too**, for a different reason: the SDK injects
`CODEX_API_KEY` into the CLI environment and the vendor's own CI guidance
recommends setting it. Hence `forced_login_method` pinned to ChatGPT, plus the
scrub. **Google fails closed** — an unauthenticated headless run exits with an
authentication error rather than hanging.

**The runtime's own statement is better than any of it.** The Anthropic init
`SystemMessage` carries `apiKeySource`, which reads `"none"` under subscription
OAuth (observed 2026-08-16). The adapter cross-checks it mid-run and raises
`AuthModeMismatch` rather than inferring the billing mode beforehand. Two
related shapes from the same read: `is_error` is **not** implied by `subtype` —
a 401 arrives as `subtype='success'`, `is_error=True`, `api_error_status=401`,
so a mapper trusting `subtype` reports a failed run as a success — and the SDK
raises a bare `Exception` *after* yielding the error `ResultMessage`, which the
adapter swallows in favour of the result it already has.

#### Cancellation, and what a killed Codex run leaves behind

`ClaudeSDKClient.interrupt()` exists, which is the whole of
`graceful_cancel: supported` on `anthropic-sdk`. On `openai-sdk` there is no
documented cancel, so it was driven: `codex-cli` 0.117.0 under ChatGPT auth,
cancellation by hard process-tree kill, which is the floor Windows leaves (there
is no graceful SIGTERM). Findings, 2026-08-16:

* **No terminal event of any kind is emitted.** The stream ends after
  `thread.started` + `turn.started`, so the adapter synthesizes its own
  `canceled` terminal. This is why the cell is `unsupported` rather than
  `unverified`: it is a checked absence.
* **A thread is durable only after its first turn completes.** Killed during a
  thread's first turn, the whole thread is lost — no rollout is written and
  `codex exec resume <id>` fails with `no rollout found`. Killed during a later
  turn, the thread survives and resumes cleanly, and the in-flight turn
  (including the user message that started it) vanishes without trace.
* **Only a hard kill loses data.** A turn that failed with a server-side model
  error still wrote its rollout.
* **Whether the killed turn was billed is unknowable locally.** The generation
  was in flight server-side; assume tokens may still have been drawn.
* **The harness overhead is real and large.** A baseline "reply OK" run reported
  `input_tokens: 51618, output_tokens: 16` — roughly 50k tokens of harness
  prefix per `exec` invocation, which a stateless call pays every time. Read any
  token guard on this runtime against that number, not against your prompt.
* **Error payloads are JSON-encoded strings inside `error.message`**, so a
  status code needs a second parse. The same double-encoding shows up in the
  schema rejections below.

#### Tools and MCP

On `anthropic-sdk` (installed package, 2026-08-16): `mcp_servers` takes stdio,
SSE, HTTP and in-process SDK server configs; `@tool` + `create_sdk_mcp_server`
register caller functions in-process, which is `tools_in_process: supported`.
Two details the adapter depends on — `tools=[]` is the documented "remove every
built-in" switch and the transport really passes an empty list through
(`if self._options.tools is not None:` → `--tools ""`, so a falsy check there
would have left every built-in on); and `max_turns` counts **agentic** turns, so
`max_turns=1` forbids a tool result from ever coming back. `ToolResultBlock`
arrives in `UserMessage.content`, not in the assistant message.

On `openai-sdk`, per-run MCP declaration works — `-c
'mcp_servers.<name>={command=…}'` was driven token-free through `codex mcp list`
— but **`-c` merges, it does not replace**: assigning an empty table leaves the
user's configured servers in place, while a per-server `enabled=false` does
disable one. So exclusivity is enforced by enumerate-and-disable
(`codex mcp list --json`, then one `enabled=false` per configured server),
failing closed if enumeration fails. Relocating `CODEX_HOME` to an empty config
directory would isolate it and would also orphan `auth.json`, breaking the
subscription login.

`tools_in_process` is `unsupported` on `openai-sdk` as a **checked absence** for
the default transport: `codex exec` is an MCP client and a subprocess, with no
equivalent of `create_sdk_mcp_server`. The app-server transport does register
caller functions natively (`ThreadStartParams.dynamicTools`, an `item/tool/call`
request, a `{contentItems, success}` response — driven live 2026-08-31 and
captured at `tests/fixtures/appserver/live-capture-2026-08-31.jsonl`), and the
cell describes the default.

**A lesson worth keeping, because the record carried the wrong conclusion for
fifteen days.** "Codex has no `tools=[]`" is true; "so the toolbelt cannot be
switched off" does not follow, because Codex has a *config* layer. Six `-c` keys
(`features.shell_tool`, `features.view_image`, `features.browser_use`,
`features.image_generation`, `features.apps`, `web_search`) switch it off:
measured on app-server 2026-08-31, an otherwise identical bare call went from
**15,890 to 10,269 prompt tokens**, 5,621 removed, with the caller's own
`dynamicTools` unaffected. "The vendor has no parameter for X" is not "X is not
achievable" until the config layer has been checked too.

`command_execution` was captured live on 2026-08-30 — field names confirmed on
both `item.started` and `item.completed`, with no others, and a fourth `status`
nobody had documented: **`declined`**, emitted when the Windows sandbox refuses a
command. The bug it exposed is why the fixture is committed
(`tests/fixtures/tools/openai-command-execution-2026-08-30.jsonl`): the mapper
derived `is_error` from `status == "failed"` and a non-zero `exit_code`, so a
declined command reached `is_error=True` only incidentally, through its
`exit_code` of `-1`. A declined item with `exit_code: null` would have been
reported to the caller as a **success**, carrying the sandbox's refusal text as
its output. `mcp_tool_call`'s field shapes have still never been observed, and
the adapter falls back to `vendor_event` whenever expected keys are absent.

#### Structured output, and the subset the two runtimes disagree about

Driven live on both runtimes on 2026-08-17, on real logins, for five tiny runs —
three of which were schema rejections that land at request validation, before any
generation, and therefore cost nothing at all. Sanitized captures are committed
under `tests/fixtures/structured_output/` and are what the offline mapping tests
run against.

**Both runtimes have a native mechanism**, so modelpass builds no submission tool
by hand. On `anthropic-sdk` it is `ClaudeAgentOptions.output_format` →
`--json-schema`, and the answer arrives on `ResultMessage.structured_output`
already parsed. Three things that run settled and would otherwise have been
guesses: the **loose** schema form is accepted (an optional property absent from
`required`, no `additionalProperties` anywhere); `max_turns=1` does not need to
move, because the submission round trip comes back as `num_turns: 2` with
`terminal_reason: "completed"` rather than `"max_turns"`; and `tools=[]` does
not switch it off, because the runtime injects its own `StructuredOutput` tool
regardless. Mechanically it *is* a tool call, which is why the adapter routes
both blocks to `vendor_event` — a caller who declared no tools must not see tool
traffic — correlating by `tool_use_id` rather than by the result's wording,
which is the runtime's and can change.

On `openai-sdk` the mechanism is `--output-schema`, and **the answer is the
final `agent_message` item's `text`** — a JSON string in the ordinary place a
plain answer would be. There is no distinct item type for it and nothing on
`turn.completed`. That is the single most important mapping fact on this runtime
and it is documented nowhere; it was read off a real run.

**The subset is strict there, and two live 400s pin it.** One for a nested item
object with an optional property: *"'required' is required to be supplied and to
be an array including every key in properties"*. One for a schema with no
`additionalProperties`: *"'additionalProperties' is required to be supplied and
to be false"*. So a schema derived from a Python dataclass with defaults is
accepted by one runtime and rejected by the other. modelpass **reports rather
than refuses**: `schema.openai_strict_issues()` names every violation with its
path and the preflight receipt carries it, and `schema.to_openai_strict()`
applies the vendor's own recipe on request — not automatically, because turning
an optional field into a mandatory nullable one changes the shape of the answer
a caller gets back.

`schema=` and `tools=` together raise `InvalidSchema` on both runtimes: the
combination is reported broken upstream on one
([openai/codex#15451](https://github.com/openai/codex/issues/15451), where
`--output-schema` is silently ignored while MCP servers are active) and
unverified on the other, and one that quietly worked on one runtime and quietly
did not on the other would be worse than one that refuses on both.

Four things about structured output are **still** unverified, stated so they are
not mistaken for checked: behaviour on a schema the model cannot satisfy; very
large or deeply nested schemas; `$ref` / `anyOf` / `oneOf`, which are passed
through untouched and which the strict checker skips rather than fails; and a
live schema-bound failover across runtimes.

#### `system_prompt_replace` on `openai-sdk`: the measurement

`ThreadStartParams.baseInstructions` **substitutes**, and the arithmetic is what
proves it. Four fresh ephemeral threads, one identical question, `inputTokens`
read off the first `thread/tokenUsage/updated` (codex 0.151.0-alpha.7.1,
app-server, 2026-08-31):

| `baseInstructions` sent | `inputTokens` | vs none |
| --- | ---: | ---: |
| none | 15,946 | — |
| ~8 tokens | 12,403 | −3,543 |
| ~800 tokens | 13,195 | −2,751 |
| ~1,600 tokens | 13,995 | −1,951 |

**The slope is 1** — 792 tokens added between the second and third rows, 800
between the third and fourth — so every supplied token adds exactly one. **The
intercept names the block being replaced**: extrapolating to zero gives ~12,395
against 15,946 with the parameter absent, so supplying it removes a fixed
**~3,550-token** block, and supplying a *ten-token* one removes it just the
same. An append cannot go down by 3,543 when ten tokens are supplied.

**The caveat is the part product copy gets wrong.** Every case above still
answered "Yes" to *"Are you a coding agent?"*, including one whose entire
instruction set was "Answer the user's question directly and briefly." That is
not the instructions surviving: the remaining ~12.4k of prefix is not base
instructions at all — it is tool definitions, environment context and
`AGENTS.md`, untouched by this parameter. Identity is read off the toolbelt as
much as off the prompt, and the toolbelt is the separate switch described above.
The honest sentence is **"your persona instead of Codex's framing"**, not "Codex
stops being a coding agent".

---

### 2.3 Sampling controls, and the report that comes with them (R5, 2026-09-13)

`Sampling` is a frozen request type with five optional fields —
`temperature`, `top_p`, `top_k`, `max_output_tokens`, `reasoning_effort` — each
validated at the constructor, and `None` on each meaning *say nothing about it*
rather than any value a caller could pass. `bridge.chat(sampling=...)` and
`bridge.preflight(sampling=...)` both take it. `options={"max_output_tokens":
N}`, which ticket 1.6 shipped as a stopgap, is now an alias for the same field
and is folded in by `RunRequest.effective_sampling`, so one value reaches the
wire and one appears on the report.

**Support varies per model, not only per runtime, and that is the whole of R5.**
`modelpass/sampling_rules.py` holds one table, keyed by runtime and refined by
anchored model family:

| Runtime / family | Rule | Read from |
| --- | --- | --- |
| `anthropic-api` (default) | all five; `max_tokens` **required**, default 4096; temperature ceiling 1.0; effort → `output_config.effort` | `anthropic` 0.97.0 typed surface |
| `claude-sonnet-4-x`, `claude-haiku-4-x` | one of `temperature` / `top_p` | a downstream agent host's model factory, restated in its settings form |
| the adaptive-thinking families | an effort request removes `temperature`, `top_p`, `top_k` | a RAG evaluation harness's reasoning module |
| `openai-api` (default) | `temperature`, `top_p`, `max_output_tokens`; **no `top_k`** — the field is not in either create signature | `openai` 2.32.0 typed surface |
| `google-api` (default) | `temperature`, `top_p`, **`top_k`**, `max_output_tokens`; effort → `thinking_config.thinking_level`; **never a `thinking_budget`** | `google-genai` 1.73.1 typed surface |
| `o1` / `o3` / `o4` | no sampling at all; effort only | the same agent host and evaluation harness |
| `gpt-5` | `temperature` forced to 1.0; no `top_p` / `top_k`; effort accepted | the same agent host |
| `anthropic-sdk`, `openai-sdk` | nothing accepted | a grep over both adapters |

The `openai-api` row was driven by ticket 1.9, `openai-compatible`'s by 1.10 and
`google-api`'s by 1.11, so all four API rows are marked **verified**; the four
agent rows are not, and `google-cli` / `google-sdk` accept nothing because
nothing about them has been driven. Every row carries a dated source note
(`rules_for(runtime, model).source`).

An unknown model gets the runtime's defaults plus the note *"model rules unknown
for X; runtime defaults applied"*. Families are matched at a token boundary,
never by substring, so `gpt-4o-mini` is a `gpt-4o` and is never read as a
reasoning model.

**The report.** Every receipt and every `runs.jsonl` line carries
`sampling_requested`, `sampling_applied` and `sampling_notes` — and the notes are
the point:

```
temperature 0.2 requested; gpt-5 accepts only 1.0, sent 1.0
top_k not supported on openai-api, dropped
max_output_tokens not requested; anthropic-api requires one, sent the modelpass default 4096
```

It is filled **before the run**, from the table rather than from the vendor, so
it can be read before spending. What a vendor did with a value it accepted is not
knowable here and is not claimed. The run-log keys are additive; a line written
before they existed reads back as empty.

**Sampling on an agent runtime is not an error.** Every field is dropped, every
drop gets a sentence, and the run proceeds. That is the asymmetry with `tools=`,
which refuses: asking a runtime to run a tool it cannot run has no sensible
outcome but a refusal, whereas "send what it takes, say what happened to the
rest" is available here.

**`max_output_tokens` is not `stopAtTokens`.** A ceiling on one answer, enforced
by the vendor by truncating, versus a modelpass guard over a whole run's total
spend that ends the stream. Two settings, two keywords, set both if you want
both.

**Sessions take no sampling.** `Session.send` has no `sampling=`, deliberately:
sessions exist only on the two agent runtimes, whose cells are `unsupported`, so
the parameter could only ever be a field that is always dropped. If a future
runtime holds a history *and* takes sampling, it belongs at the session
constructor beside `system_prompt` — fixed for the conversation, for the
prefix-stability reason §1.4 gives — and not on `send`.

---

### 2.4 `cache_automatic` — does this runtime cache prompts unasked? (2026-09-21)

The third cache cell, added with `promptCache` (§1.9). `cache_breakpoints` asks
whether a caller may say *where* the prefix ends; `ttl_control` asks how *long*
one lives; neither of them answers *does it happen at all*. Before this cell the
table could not tell "this runtime ignores your caching request" from "this
runtime was already doing it", which are the two most different answers a caller
can get.

`supported` here is never "you can control this" — it is the opposite. It marks
the runtimes where a request to cache is already satisfied and there is nothing
to send.

| Runtime | Cell | What moved it |
| --- | --- | --- |
| `anthropic-sdk` | **supported** | `tests/fixtures/structured_output/anthropic-structured-output-2026-08-17.json`, a live capture whose four runs each report `cache_creation_input_tokens: 605` — modelpass sent no caching instruction and on this row cannot, since `cache_breakpoints` is `unsupported` and there is no field to send one in. SDK half: `claude_agent_sdk` 0.2.148 declares `cacheReadInputTokens` / `cacheCreationInputTokens` on the usage shape |
| `openai-sdk` | **supported** | `tests/fixtures/appserver/live-capture-2026-08-31.jsonl`: one thread whose first turn reports `cachedInputTokens: 0` and whose later turns report 12,032 and 24,064, with `cacheWriteInputTokens: 0` throughout and no cache field anywhere on the transport |
| `openai-api` | **supported** | `openai` 2.32.0 makes `ResponseUsage.input_tokens_details.cached_tokens` a **required** int on every response, and types `prompt_cache_retention` as `Optional[Literal['in-memory', '24h']]` with a docstring offering to *extend* a cache the request has no way to start |
| `anthropic-api` | unverified | Never consulted: `cache_breakpoints` is `supported`, so a request resolves to `explicit` before this cell is read. Left open rather than marked `unsupported`, because "this endpoint does no caching you did not ask for" is a claim about vendor behaviour that nobody has driven |
| `google-api` | unverified | **Deliberately not moved on this document's own prose.** The `cache_breakpoints` note says Gemini's implicit caching is "automatic and unaddressable", and that sentence was written to explain why a breakpoint has nowhere to go, not from a read of anything. What `google-genai` 1.73.1 actually shows is the *explicit* mechanism: `GenerateContentConfig.cached_content` names a resource the caller created, and `cached_content_token_count` is the read side of that resource. Moving the cell needs two live calls with an identical prefix and no `cached_content` |
| `openai-compatible` | unverified | By construction, like every cell on that row |
| `google-cli`, `google-sdk` | unverified | Experimental runtimes; nothing driven |

The consequence a caller feels: `promptCache` is **refused** on `google-api`,
`openai-compatible` and the two Google experimental runtimes. That is the
designed behaviour of an unverified cell rather than a gap — `supports()` has
treated `unverified` as unusable since the table was written, and honouring a
caching request on a cell nobody has checked would be the guess this table
exists to refuse.

---

## 3. The two implemented vendors, in depth

### 3.1 Anthropic (`anthropic-sdk`)

**Install and auth.** `pip install modelpass[anthropic]` brings `claude-agent-sdk`;
the native CLI comes from `npm install -g @anthropic-ai/claude-code` and you log
in with `claude /login`, which lands a credential in
`~/.claude/.credentials.json` (macOS: Keychain). modelpass never reads, copies,
replays or transmits it — it launches the runtime and the runtime authenticates
itself. Both subscription and API-key auth are supported; a *subscription*
connection **fails closed**, meaning that if the preflight cannot confirm a
usable subscription credential it reports `ok=False` and the bridge raises rather
than falling through to an API key.

**Multiple accounts.** [Claude Code's supported isolation
boundary](https://code.claude.com/docs/en/team#credential-management) is
`CLAUDE_CONFIG_DIR`: all credentials, settings, plugins and history move
beneath that root. A connection's `configDir` field (or `--config-dir PATH` at setup)
sets it per launch. The value is carried on the preflight plan and receipt, and the
credential probe reads `<configDir>/.credentials.json`, so the directory reported is
the directory the child receives. An ambient `CLAUDE_CONFIG_DIR` is scrubbed unless a
connection explicitly names its own; this prevents an IDE-launched process from
silently switching accounts.

**macOS.** Claude Code prefers the login Keychain there, and keys the Keychain entry to
`CLAUDE_CONFIG_DIR` — a session with a different directory reads a different entry — so
directory isolation is real on macOS too. When the Keychain refuses a write (locked, an
SSH session, a password out of sync) Claude Code writes the same plaintext
`.credentials.json` under that directory that Linux uses, and modelpass's probe checks for
that file first on every platform. Keychain contents are never inspected by modelpass; if
there is no file, `CredentialStatus.keychain_only` is set, the receipt names the entry
and the directory it is keyed to, and the token-free `claude auth status` probe — which
already runs with this connection's `CLAUDE_CONFIG_DIR` — becomes the evidence for which
account the directory selects. `loggedIn: false` there fails the subscription preflight
closed; a probe that returns nothing leaves the receipt usable and adds a note saying
identity could not be confirmed. Codex needs no macOS-specific handling: its keyring
entry is keyed by `CODEX_HOME` in the same way. **This macOS behaviour is implemented
from Claude Code's documentation and Codex's published source and has not yet been
exercised on a Mac**; every branch of it is covered by unit tests that simulate the
platform.

Modelpass exposes this as an account profile: `name` is the stable ID used by clients,
`nickname` is the human label, `vendor` is derived from the runtime, and `configDir`
is the optional isolation boundary. `Account` aliases `Connection`, so older callers
and files remain source-compatible; `Bridge.accounts()` and `Bridge.account(id)` expose
the higher-level terminology directly.

**Account identity.** Preflight runs the documented, token-free `claude auth status`
JSON command inside the selected `CLAUDE_CONFIG_DIR`. Its non-secret identity fields
become `Receipt.account_profile`: login state and method, API provider, email,
organization ID/name, and subscription type. The adapter normalizes an explicit
allow-list rather than retaining the vendor payload, so tokens and future unknown fields
cannot enter a receipt, CLI rendering, run log, or Bench page.

The probe is a subprocess costing seconds, and every `run`, `chat` and session open goes
through a preflight that wants its answer, so each adapter caches it for 60 seconds keyed
by *(binary, config directory)* — never by binary alone, since two profiles on one binary
are two different accounts. The same applies to Codex's `account/read`, whose probe
starts a whole app-server. `Bridge.refresh_identity(connection)` drops that cache;
`modelpass verify` and the Bench's Verify button call it first, because re-reading the live
identity is their entire job. `Adapter.invalidate_identity_cache()` is the adapter-side
hook, a no-op by default.

One install trap, from the README: `claude-agent-sdk` ships a bundled `claude`
binary and resolves it **ahead of anything on `PATH`**, so
`npm install -g @anthropic-ai/claude-code@latest` updates the CLI you type and
leaves modelpass's untouched. To move modelpass's, `pip install -U claude-agent-sdk`.
The receipt names the binary actually resolved, which is the only reliable way to
know which one a version verdict is about.

The legal position, hedged and dated 2026-08-16, is
[legal/anthropic.md](legal/anthropic.md): the prohibition targets a developer
mediating access through *other people's* consumer credentials, while the
Agent SDK credit article explicitly lists third-party apps authenticating with
your own subscription through the Agent SDK as covered. This is the strongest
footing of any vendor here — the permitted shape is written down affirmatively,
not merely un-prohibited.

**Design of the integration.** modelpass drives the Python `claude-agent-sdk`,
which owns its own subprocess. Three things are load-bearing:

* **The scrub cannot go through `ClaudeAgentOptions.env`.** The SDK builds the
  child environment as `{**os.environ, ..., **options.env}`, so `env` can only
  *add* — it can never remove. An ambient `ANTHROPIC_API_KEY` would survive and,
  because subscription OAuth resolves *last* in Claude Code's precedence, would
  silently win and move the run onto metered billing. modelpass therefore removes
  the planned names from `os.environ` for the duration of the spawn, under a
  lock, and restores them afterwards.
  `CLAUDE_CONFIG_DIR` is included in the same enforcement: scrub ambient selection,
  then add the connection's explicit directory through `options.env`, whose value wins
  in the SDK's child-environment merge.
* **Built-in tools are off by construction, not by list.** `tools=[]` is the
  SDK's documented "remove every built-in" switch; an explicit `disallowed_tools`
  list is kept as a second lock, but `tools=[]` is what survives Claude Code
  growing a tool nobody here has heard of.
* **Money never enters the normalized stream.** `total_cost_usd` rides in a
  `vendor_event` and nowhere else (D7).

A **session** here is one live `ClaudeSDKClient` held across turns, with its own
event loop on its own thread for the object's life, calling `query()` once per
turn. That is what makes history the runtime's rather than something modelpass
flattens back into a prompt. `persist=False` is a real mode, and the env var
`CLAUDE_CODE_SKIP_PROMPT_HISTORY=1` is what makes it one — proven 2026-08-30 by a
control run *without* the variable, which did write a transcript file.

### 3.2 OpenAI (`openai-sdk`), and its two transports

**Install and auth.** `pip install modelpass[openai]` brings `openai-codex`; the
native CLI comes from `npm install -g @openai/codex` and you log in with
`codex login` ("Sign in with ChatGPT"), landing a credential in
`~/.codex/auth.json`. Runs draw the Codex allowance included with the ChatGPT
plan. modelpass actively **scrubs** ambient `OPENAI_API_KEY` / `CODEX_API_KEY` from
every launch so a run can never silently land on metered billing.

**Multiple accounts.** Codex documents `CODEX_HOME` as the root for its state,
including `config.toml` and `auth.json`. A profile's `configDir` becomes `CODEX_HOME`
for preflight and for every exec or app-server child; an ambient `CODEX_HOME` is
scrubbed when the selected profile does not explicitly name one. No credential-store
setting is required: `cli_auth_credentials_store` defaults to `file`, `auto` falls back
to `auth.json` when no keyring is available, and the keyring entry is itself keyed by
`CODEX_HOME` — service `Codex Auth`, account key the SHA-256 of the canonicalized codex
home (`codex-rs/login/src/auth/storage.rs`, `compute_store_key` /
`compute_keyring_account`). A different directory therefore reads a different entry under
either store. Modelpass reports which one is in play as a receipt note
(`credential store: ...`) and refuses none of them; an earlier version refused `keyring`
and unresolved `auto` on the mistaken belief that only `auth.json` moved with the
directory. See OpenAI's [authentication storage documentation](https://learn.chatgpt.com/docs/auth)
and [advanced configuration documentation](https://learn.chatgpt.com/docs/config-file/config-advanced).

**Account identity.** After `codex login status` confirms the billing mode, preflight
uses the supported app-server `account/read` request inside the selected `CODEX_HOME`.
Codex reports account type, email, and ChatGPT plan type; those fields become the
normalized `Receipt.account_profile`. Tokens and the rest of the app-server payload are
not retained. `account/rateLimits/read` is a separate surface and is not presented as
identity metadata.

The legal position, dated 2026-08-16, is [legal/openai.md](legal/openai.md), and
it is honest that the footing here is **weaker than Anthropic's**: OpenAI
documents ChatGPT-authenticated Codex use and sells plans on it, and the
2026-08-15 verification found no prohibition on personal programmatic use — but
found no affirmative statement blessing third-party software that wraps the CLI
either. Silence is not permission; it is silence.

Watch for the **two-installs trap**: a vendor's desktop app may install its own
auto-updating copy of the CLI somewhere not on `PATH` (Codex does this, under
`AppData\Local\OpenAI\Codex\bin\<hash>\` on Windows), so your app runs the newest
models while `codex` in a shell is months behind. modelpass **reports** the
mismatch and deliberately does not resolve it — silently switching which
executable a chat call launches, because modelpass went looking for a config value,
is exactly the ambient influence this project refuses. Point it somewhere
specific with `options={"codex_bin": ...}` if you mean to.

**Two transports, one runtime identity, and `app-server` is what you get
without asking.** The default flipped on 2026-08-31 (S7). `exec` is fully
reachable, unchanged, and not on a removal path — see
[../README.md](../README.md) for the two behaviour changes the flip makes for
existing callers (`system_prompt` replaces instead of layering; `mcp_servers=`
needs the opt-out) and the one line that reverses both.

| | `exec` (opt-out) | `app-server` (**default** since 2026-08-31) |
| --- | --- | --- |
| How | `codex exec --json`, one process per turn, JSON Lines out | `codex app-server --listen stdio://`, JSON-RPC over one long-lived child |
| Selected by | `options={"transport": "exec"}` | nothing — it is the default |
| System prompt | a `System: ` line glued to the front of the prompt, **layered on** the vendor persona | `ThreadStartParams.baseInstructions`, which **replaces** the vendor persona |
| A conversation | flattened into one prompt string with a transcript preamble | prior turns injected as real role-bearing items (`thread/inject_items`); the flattened form is the announced fallback |
| Output schema | a temp file passed to `--output-schema` | a per-turn `outputSchema` field |
| Sessions | yes (the rollout file), one process per turn | yes (S5) — one long-lived child holding one thread: `thread/start`, `turn/start` per turn, `thread/resume` to pick one back up |
| `get_history()` on a session | `()` — the transport cannot read a thread back | the conversation, paged from `thread/items/list` |
| `list_sessions()` | refused — no scriptable listing exists | `thread/list`, token-free, account-wide |
| `ChatSession(system_prompt=...)` | refused — exec can only append | opens; `baseInstructions` replaces |
| `tools=` | refused — MCP over a command line is the only tool channel | `dynamicTools` registered natively, on `chat()` and on a session |
| `mcp_servers=` | `-c mcp_servers.<name>={...}`, exclusive by enumerate-and-disable | **refused** — no driven equivalent, which is the one capability the default gives up |
| Usage | once, at `turn.completed` — a token guard reports rather than interrupts | `thread/tokenUsage/updated` after **each** model response, so a guard can stop the run |
| Text | the whole answer in one `item.completed` | `item/agentMessage/delta`, token by token |
| Reasoning | undriven either way — `unverified` | the reasoning **summary** stream; raw `textDelta` never fired |

An unrecognized `transport` value is **refused rather than defaulted**. Silently
running the default for a caller who typed `"appserver"` would change what the
run does and say nothing. The receipt names the transport that actually ran, and
on an exec run it says in as many words that the capability rows describe a
default this call opted out of.

**Why modelpass speaks the app-server protocol itself rather than depending on the
`openai-codex` SDK** (decided 2026-08-31):

* Its `CodexClient.start()` builds the child environment from `os.environ.copy()`
  and merges the caller's overrides into it. modelpass hands a runtime a
  **scrubbed** environment and promises the receipt describes what the child
  actually got (D2); a client that re-adds the ambient environment underneath
  breaks that promise silently — the exact failure shape this library exists to
  prevent.
* Its typed parameter objects already lag the wire: `ThreadStartParams` has no
  `dynamicTools` field although the server accepts one, so modelpass would be
  passing raw dicts through a typed API anyway.
* Owning the transport is how every other adapter path stays offline-testable —
  one injectable spawn seam, and the whole protocol is exercisable from a
  scripted fake with no vendor package, no subprocess and no credential.

The SDK remains a perfectly good **binary source** through its pinned CLI,
exactly as it is for `codex exec` today.

One defect worth knowing about the exec transport, since it silently corrupted
runs before 2026-08-31: **a multi-line argv element does not survive the Windows
`.cmd` shim** — `cmd.exe` terminates its command line at the first newline, so
every line of a prompt after the first was dropped while the run reported `ok`.
The prompt now rides **stdin** with `-` in the argv slot. The run log from before
that fix cannot say which runs were truncated; Windows + `openai-sdk` +
multi-line prompt means affected.

### 3.3 What happens to your history on `chat()`

You pass history; modelpass assembles instructions → history → message (§1.2); then
each runtime renders that list its own way. **All three renderings flatten a real
conversation into labelled text.** Only the system prompt reliably becomes a
parameter, and only on two of the three paths.

**`anthropic-sdk`** — `split_messages()` in `src/modelpass/adapters/anthropic.py`:

* System messages are joined and become `ClaudeAgentOptions.system_prompt`, a
  real launch parameter.
* A lone user turn passes through **verbatim**. Wrapping "Reply with exactly OK"
  in a transcript envelope changes what the model is asked, so modelpass does not.
* Anything richer becomes a `Human: ` / `Assistant: ` transcript with a trailing
  `Assistant:` cue. This is the stateless path only — a **session** on this
  runtime sends structured turns, which is precisely the fidelity argument for
  reusing a `ChatSession` here.

**`openai-sdk` / exec** — `render_prompt()` in `src/modelpass/adapters/openai.py`:

* A lone user message passes through verbatim, and so does a system-plus-user
  pair — that is a *single shot with instructions*, not a conversation.
* Anything richer becomes a labelled transcript (`System: ` / `User: ` /
  `Assistant: `) prefixed with an explicit continuation instruction, because
  exec takes one prompt string and a real conversation has nowhere else to live.

**`openai-sdk` / app-server** — `app_server_base_instructions()` and
`app_server_input_items()`:

* System messages become `ThreadStartParams.baseInstructions` (joined, not
  last-wins — modelpass did not decide you meant only one of them).
* **Prior turns are injected with real roles.** After `thread/start`, one
  `thread/inject_items` call appends your `history` to the thread's model-visible
  history as *raw Responses API items* — `{"type": "message", "role":
  "assistant", "content": [{"type": "output_text", ...}]}` and the `input_text`
  counterpart for user turns. (The `input_text` / `output_text` asymmetry is the
  Responses API's own, not a typo.) The **current** message is not injected; it
  goes as the `turn/start` input, unlabelled. No history means no call at all.
  Driven live 2026-08-31: an injected exchange was answered from correctly, with
  no labels anywhere.
* **This is the one place a stateless call gets real roles on either vendor.**
  `chat()` on `anthropic-sdk` flattens into a `Human: ` / `Assistant: `
  transcript, and so does Codex on exec — see §3.3 above. The app-server
  transport is the exception.
* **The labels are the fallback, and it is announced.** If the server refuses
  `thread/inject_items`, modelpass falls back to the S4 labelled transcript rather
  than losing the conversation, and emits a `vendor_event` named
  `history/inject_items_refused` carrying the code, the detail and what the
  fallback costs — a silent downgrade is the thing this library exists to
  prevent. The fallback is not hypothetical: builds differ. Two traps found by
  driving it — the method is spelled **`thread/inject_items` (snake_case)** on
  the wire while the SDK's generated types imply `thread/injectItems` (the
  camelCase name is rejected as an unknown variant), and the response is `{}`,
  so success is the absence of an error rather than a payload.
* The labelled fallback exists because an input item carries **no role**. Every
  variant of the protocol's enum (`text`, `image`, `localImage`, `audio`,
  `localAudio`, `skill`, `mention`) is user-authored content, so without
  injection a previous assistant turn has nowhere role-shaped to live. Exec is
  untouched throughout; it has no such seam.

### 3.4 What happens to your history in a session

Here the **runtime owns the history** and modelpass holds no shadow copy.

* **`anthropic-sdk`**: the Agent SDK's own session, stored at
  `~/.claude/projects/<encoded-cwd>/<id>.jsonl`. The working directory *is* the
  storage key, which is why a session always sets one and why `list_sessions()`
  is directory-scoped. `get_history()` reads the transcript back via the SDK's
  `get_session_messages()` — and returns **empty for a `persist=False` session**,
  which is not a gap in the mapping: no transcript was written, and the SDK
  offers no way to read the live client's context back out.
* **`openai-sdk` / exec**: Codex's rollout file. The first `send` is a plain
  `codex exec`; every turn after it is `codex exec resume <id>`. `id` stays
  `None` until a first turn has **succeeded**, because `thread.started` arrives
  at the *start* of a turn while the rollout that makes the thread resumable is
  written at the end — publishing an id in between would hand a caller something
  a later resume rejects.
* **`openai-sdk` / app-server**: the same rollout, reached through a thread on a
  long-lived child. `thread/start` once, `turn/start` per turn — so turn two
  sends the new message and nothing else — and `thread/resume` picks a stored
  thread back up, which is also where "that thread is gone" is answered *before*
  anything is spent rather than by a turn that failed.

**`get_history()` returns empty on the Codex exec opt-out**, and honestly so:
that transport has no counterpart to `get_session_messages()`, and the rollout file under
`~/.codex/sessions/YYYY/MM/DD/` is a private on-disk format the CLI is actively
migrating. Reassembling a transcript from either that format or from the events
the adapter happened to see would produce a second history that drifts from the
one the model is actually being sent — and only one of the two is real.

**On app-server -- the default since 2026-08-31 -- it answers** (S5), which is
the headline difference between the two transports. The read is `thread/items/list`, paged in
conversation order until the cursor runs out — *not*
`thread/read(includeTurns=true)`, for two reasons worth stating because the
second one is a trap: the vendor's own schema calls full-history hydration
*deprecated for paginated threads* and every Codex thread is paginated, and the
turn summary that route returns carries only the assistant's final message. A
history read that way would have quietly lost the user's own turn.

Codex threads hold nineteen kinds of item and only two of them are messages, so
the mapping is explicit about the rest:

| Thread item | `get_history()` |
| --- | --- |
| `userMessage` | `Message(role="user", ...)`, content blocks flattened |
| `agentMessage` | `Message(role="assistant", ...)` |
| `commandExecution`, `dynamicToolCall`, `mcpToolCall`, `webSearch`, … | an assistant-role **marker**: `[commandExecution ls -la]` |
| `contextCompaction`, and every other variant | an assistant-role marker naming the type |
| `reasoning` with nothing in it | dropped — an artifact of the item stream, not a turn |

Marked rather than dropped, because a history that silently kept only the two
message types would tell a caller their conversation was three turns long when
the model ran four commands inside it — the same drift the exec answer refuses to
manufacture, arriving by a different route. It is the convention the Anthropic
adapter already uses, where a replayed tool result reads as `[tool_result]`, so
the two runtimes produce the same shape of transcript.

The method itself is **schema-sourced and undriven**: its params and response
come from the shipped binary's own generator (`codex app-server
generate-json-schema`, token-free), and no session has yet read a real thread
back. A build that refuses it raises the vendor's own error rather than answering
`()` — an empty history means *this conversation is empty*, and saying that about
a thread modelpass could not read would be exactly the lie the exec answer avoids.

### 3.5 The system-prompt asymmetry

This is the single fact that explains most of the Anthropic/Codex compatibility
story.

* **`anthropic-sdk` can replace.** `ClaudeAgentOptions.system_prompt` accepts a
  custom string that replaces the preset entirely — tools remain, but their
  guidance, the safety rules and the environment context are gone, which is
  exactly what a `ChatSession` wants. The preset object form
  (`{"type": "preset", "preset": "claude_code"}`, optionally with `append`) is
  the `WorkerSession` mapping. **The trap worth recording next to the yes**:
  omitting `system_prompt` does not give you Claude Code's prompt, it gives the
  *minimal* tool-calling one — a full toolbelt with no guidance for it. So a
  worker with no prompt maps to the preset and never to omission.
* **`openai-sdk` replaces by default since 2026-08-31.** A chat's system prompt
  is `ThreadStartParams.baseInstructions`, which goes in *instead of* the
  hardcoded "You are a deployed coding agent". So `new_chat(system_prompt=...)`
  opens on a Codex connection with no options at all — the first `ChatSession`
  this runtime has been able to hold.
* **On the exec opt-out it can only append**, and there
  `ChatSession(system_prompt=...)` still raises rather than degrading, with
  `Session.help()` leading on the reason. A `WorkerSession` keeps exec's layering
  on **both** transports — the prompt above the first turn's task — because
  append is the semantics a worker asked for and `baseInstructions` would delete
  the persona it chose to keep. `developerInstructions` is the vendor's undriven
  candidate for a native layered-above channel; nothing sends it yet.
* **This is the behaviour change the flip makes for existing callers**, and it is
  announced rather than buried: the same `system_prompt` on the same connection
  produced a layered run before 2026-08-31 and produces a replaced one after.
  Replacing is the better outcome for a caller who brought their own persona — it
  drops several thousand tokens of vendor framing — but it is a change, not a
  fix. `options={"transport": "exec"}` restores the old behaviour in one line.
* **The caveat that belongs next to the yes.** Replacing the instructions does
  *not* by itself erase the coding-agent character. `baseInstructions` names the
  base instructions and nothing else — tool definitions, environment context and
  `AGENTS.md` are untouched, which is why ~12.4k input tokens survive a
  replacement that removed ~3,543 — and a model infers its identity from its
  toolbelt as much as from its prompt. The toolbelt is a **separate switch**, and
  since D23 **every chat-shaped call throws both** — a `ChatSession` and a
  stateless `chat()` alike: `chat_tool_overrides()` on the command line
  alongside the replaced instructions. Driven on this transport 2026-08-31, the
  toolbelt half alone removed **5,621 prompt tokens** (15,890 → 10,269), and the
  caller's own `tools=` kept working across it. Promise "your persona instead of
  Codex's framing", not "Codex stops being a coding agent" — what modelpass
  switches off is what `-c` reaches, and `unified_exec` is the named remainder.

  A `WorkerSession` is the deliberate exception: it keeps the whole belt,
  because iterating with `exec` is what a worker is *for*. `options={"native_
  tools": True}` is the same opt-in on a stateless call.

`system_prompt` is applied on the **first** turn only in a session. The thread
carries it from then on, so restating it every turn would move a stable prefix
and pay for it twice.

### 3.6 Prompt caching, per vendor

The two vendors' caching models differ in the one way that matters to a caller
writing a loop: whether you can steer it, and whether writes cost anything.

**Codex: automatic, exact-prefix, no opt-in, no TTL lever, and no write fee.**
Caching engages at 1,024 tokens or more, whatever the model — the floor does not
vary — and there is no configuration surface at all. `-c` is a fully generic
config override, but there is no key here to override. That is a *checked
absence*, not an unchecked question (`ttl_control: unsupported`). Idle life is
roughly five to ten minutes against a one-hour ceiling.

Measured across one captured thread on 2026-08-31 (`last` breakdowns, in order;
fixture `tests/fixtures/appserver/live-capture-2026-08-31.jsonl`):

| model response | inputTokens | cached | cacheWrite | hit |
| --- | --- | --- | --- | --- |
| #1 (cold) | 13,038 | 0 | 0 | 0% |
| #2 | 13,123 | 12,032 | 0 | 91.7% |
| #3 | 13,165 | 12,032 | 0 | 91.4% |

Two things to read off that table:

* **`cacheWriteInputTokens` is always 0 on Codex. There is no write fee.**
* **The cached figure stays flat at 12,032 while input grows.** What caches is
  the stable *front* — base instructions plus tool definitions — and the growing
  conversation tail does not. That is `cache_eligibility`'s `preset_prefix=True`
  showing up as a number.

So **prefix stability is the only lever on Codex**, and that is what makes D17's
immutable `tools` and `system_prompt` load-bearing rather than merely tidy. For a
caller structuring a loop it means: put everything stable at the front and never
move it. Fix the rubric, the tool set and the persona at construction, vary only
the message, and reuse one session (or one byte-identical `system_prompt` across
`chat()` calls) rather than rebuilding the instructions per item. Anything that
edits the front — a timestamp in the system prompt, a tool list that grows,
instructions concatenated into the message — recomputes everything after it, on
every call.

One open question, honestly flagged: on exec the cached prefix includes Codex's
own coding-agent persona and toolbelt; with `baseInstructions` *replacing* the
persona and the toolbelt off, that prefix is a different and likely smaller
thing. These runs clear the 1,024-token floor comfortably, but the arithmetic
behind `cache_eligibility` on the app-server transport was re-derived in S7
(2026-08-31) and now branches by transport. On exec the cached prefix is the
persona, the tool definitions and the environment context. On app-server
`baseInstructions` replaces the persona, so the prefix is a **different and
smaller** thing: an identical turn cost 15,940 input tokens with no
`baseInstructions` and 12,397 with a ~10-token one, leaving ~12.4k of tool
definitions and environment context that the parameter does not name and does
not touch. The cached figure held flat at 12,032 across three responses while
input grew, which is the direct measurement that only the stable front caches.
`preset_prefix` stays `True` on both, and against a 1,024-token floor that is
roughly twelve times over — for a different reason and by a different margin
than on exec.

Those three figures were all taken with the runtime's toolbelt **on**, which
since D23 is no longer what a chat-shaped call runs. The toolbelt-off case was
measured on 2026-08-31: the same bare call fell from **15,890 to 10,269** prompt
tokens, 5,621 of tool definitions leaving the prefix. The floor survives that
comfortably — 10,269 against 1,024 is still an order of magnitude — but the
cached figure specifically has not been re-measured under the new launch, so the
receipt reports the floor as cleared rather than restating 12,032 as though it
were still the number.

**Anthropic: the opposite — you pay to write, and there is a TTL lever.** A live
copilot turn read `cache_write 7039`. modelpass models this explicitly:
`TokenUsage.cache_write_tokens` is its own field precisely so a cold prefix and
genuinely new content are distinguishable, and `billable_input_tokens` is
`input + cache_write`, because a guard watching `input_tokens` alone would
silently stop counting writes.

`ttl_control` is `supported`, via `CLAUDE_CODE_PROMPT_CACHE_TTL` (`'5m'` | `'1h'`)
— but **version-gated: it needs Claude Code v2.1.242 or later**, and the version
that decides it is the *SDK's* bundled CLI, which moves with pip rather than npm.
modelpass checks and falls back to the runtime default rather than exporting a
variable that is silently ignored. Never switch TTL on a live prefix: a change is
a new cache write at the new tier, not a timer update.

Practical guidance for `chat()` on Anthropic, measured on `claude-sub` / Sonnet,
2026-08-30, nine calls on one machine and one model:

* A 33-token system prompt with items differing by 10–25 tokens paid a fresh
  ~1,850-token write on **every** call and never read.
* The same prompt with a **byte-identical** message read 1,846 and wrote nothing,
  three calls running.
* Once the system block cleared the floor — 1,709 tokens — reads of 2,790 appeared
  on later calls against *different* messages, which is the behaviour a scoring
  loop wants.

The practical rule: **a rubric materially below the floor buys nothing from being
held stable**, because the reusable unit is larger than the rubric. Growing it
past the floor is what turns a stable prefix into a reused one — and is usually
something a judge prompt wants anyway. The segment arithmetic does not fully
reconcile across those arms, so treat "a breakpoint forms at the system block
once it clears the floor" as the shape the data supports rather than a mechanism
anyone has proven.

**Read the reads, not the writes.** `cached_input_tokens` at zero across calls
that should share a prefix means nothing is being reused, whatever the writes
say. `cache_write_tokens` alone is ambiguous: it is zero when the whole prompt
repeats byte-for-byte and large when it does not, so both a perfect hit and a
total miss can show a number a reader might take for health.

### 3.7 Token accounting differs between the vendors, and it is load-bearing

If you do your own arithmetic on `TokenUsage`, read this section.

**modelpass's convention** (`src/modelpass/types.py`): `cached_input_tokens` is
counted **separately from** `input_tokens`, not inside it. `cache_write_tokens`
is separate again. `total_tokens` is the sum of all four fields.

That convention is **correct for Anthropic natively** — a live row reads
`in 2, cached 1901`, which is impossible unless the two are parallel.

**It is wrong for Codex, which reports `input_tokens` INCLUSIVE of
`cached_input_tokens`.** Driven on both transports 2026-08-31:

* app-server `last`: `{inputTokens 13123, cachedInputTokens 12032, outputTokens 21,
  totalTokens 13144}` — the vendor's own total is `input + output`, so cached is
  demonstrably a **subset** of input.
* exec `turn.completed`: `{input_tokens 14228, cached_input_tokens 9984,
  output_tokens 5}` — same nesting.

Naively folding those into modelpass's parallel convention double-counts the cached
tokens. Measured on a real `~/.modelpass/runs.jsonl`, **every** Codex run was
inflated by roughly 65% — 22,655 reported against 13,695 true. That was live: it
corrupted the run log and fed `stop_at_tokens`, so guards fired early on Codex.

**The fix, and what it means for you:** both Codex mappers now subtract `cached`
out of `input`, clamped at zero, and `TokenUsage` keeps its parallel convention
untouched. A test in the Anthropic suite asserts that mapper is *not* "fixed" the
same way. So the numbers you read off `TokenUsage` follow one convention on both
runtimes — but if you are reading vendor payloads directly, or comparing a
modelpass figure against a vendor dashboard, the nesting difference is the thing
that will not reconcile. The related gap found at the same time: the exec mapper
had never read `cache_write_input_tokens` although the wire carries it; it does
now.

---

## 4. Post-MVP candidate: evaluate the rest of each vendor SDK's surface

**A candidate to evaluate, not a commitment.** Both vendors expose capabilities
modelpass does not model, and some of them may be worth exposing. What follows is
grounded in what has actually been observed, not in a survey of the docs.

**The observation, on the OpenAI side.** Enumerating the installed build's
supported methods (codex 0.151.0-alpha.7.1) turned up a substantial surface
modelpass drives none of:

| Method / family | What it appears to be |
| --- | --- |
| `thread/compact/start` | server-side history compaction |
| `turn/steer` | redirecting a turn while it is running |
| `thread/revert`, `thread/rollback` | undoing turns |
| `thread/items/list`, `thread/turns/list`, `thread/timeline/list` | reading a thread's contents back — directly relevant to `get_history()` on Codex (§3.4) |
| `thread/search` | search across threads |
| `thread/queue/*` | a queueing family |
| `thread/realtime/*` | an audio family |
| `account/rateLimits/updated` | a **native allowance figure**, arriving unprompted — modelpass currently infers allowance from quota markers |

Each is a candidate to *evaluate*: what it does, whether it survives a vendor
release, whether it maps onto anything in the normalized event vocabulary, and
whether a consumer wants it. Several would need capability cells of their own,
and the standing caution applies — this is experimental vendor surface behind
`capabilities.experimentalApi`, and the 0.117 → 0.151 move falsified three facts
in hours.

**The technique is worth keeping regardless: enumerating a build's methods is
token-free.** Send an unknown JSON-RPC method and the error lists every method
the build supports. That is a free capability probe for any future build, and it
is how the two `thread/inject_items` traps in §3.3 were found. Any re-verification
pass should start there rather than with the docs.

**The Anthropic side deserves the same audit, and nobody has done it.** modelpass
drives a deliberately narrow slice of `claude-agent-sdk` — the options it needs
for chat, sessions, tools, MCP and structured output. What else that SDK exposes,
and which of it maps onto something a modelpass caller could use, is an open
question that has not been investigated. Two already-visible loose threads sit in
the registry rather than in anyone's plan: `sessions_fork` is `supported` on
Anthropic (`ClaudeAgentOptions.fork_session`) and `unverified` on Codex, where
`codex exec fork <SESSION_ID>` shipped in 0.151.0 and only the subcommand's
*existence* has been checked — a cheap experiment, and the cell should move the
moment someone runs it. And on Anthropic, finer usage signals exist in the raw
`StreamEvent` payloads (`message_start` / `message_delta` usage) which modelpass
deliberately does not use yet because they are unverified against a live run.

The bar for anything in this section is the project's usual one: driven live,
dated, and recorded as a capability cell with its evidence — or left alone and
marked `unverified`, which is a real answer.

---

## See also

* [../README.md](../README.md) — what modelpass is, install, the preflight receipt,
  the scrubbed launch, the bench
* [legal/anthropic.md](legal/anthropic.md) · [legal/openai.md](legal/openai.md) ·
  [legal/google.md](legal/google.md) — the per-vendor positions, dated, not legal
  advice
* [guards.md](guards.md) — bounding what a run can spend
* the 2026-08-31 app-server transport migration record (internal)
  — the live-verified transport facts this document draws on, and the slice plan
* [§2.2a](#22a-where-the-cells-came-from-the-verification-record) above — the dated verification record behind the capability cells
* the decision log (D1–D23) and the per-phase verification write-ups are kept
  beside the code and not published: they cite private consumer repositories and
  work nobody outside can check. The reasoning they carry that a caller can act
  on is stated in this document and in the README.
