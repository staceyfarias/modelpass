# modelpass roadmap

Work proposed for modelpass, with the evidence behind it. An entry is a
proposal until it is built; measured facts and assumptions are kept apart.

## Throughput of many small calls on a subscription route (proposed 2026-10-19)

**Why.** RAGauge graded a Test Set's complete answer key through modelpass:
about 2,900 single-passage grading calls with Claude Sonnet 5.5 (the TEST 2
rules grader, a ~700-token prompt and a short structured form back). The calls
went over two subscription connections (`claude-sub`, `anthropic-work`), one
half each. This is a workload of many small, independent, stateless calls, and
on the subscription route it ran far slower than the model needs.

**Measured (2026-10-19, rag-eval `src/pipeline/complete_grading.py`):**

- **Pace:** about 8 grades a minute per connection.
- **More workers did not raise it.** Raising the caller's parallel workers
  from 4 to 8 left the pace at about 8.4 a minute. The calls did run at the
  same time (dozens of `claude` processes alive), so each call took about
  twice as long.
- **The caller was not the bottleneck.** The machine was at 35% load on 16
  cores, each grading process used under half a core, and RAGauge's
  Answer Map write took 0.16 s per grade.
- **Other work shared the cap.** Other agent sessions on the same
  subscription were running at the time, so part of the cap was theirs.
- **A bare call carries a large harness context.** One direct
  `claude -p "Reply with the single word ok." --model claude-sonnet-5-5 --output-format json`
  call (Claude Code 2.1.280; this was NOT through modelpass) reported
  25,441 cache-creation and 34,961 cache-read input tokens for a 4-token
  answer, `total_cost_usd` 0.21.
- **The CLI warns about the model id but serves it.** Claude Code 2.1.280
  prints `[claude-code:unrecognized_model] {"model":"claude-sonnet-5-5"}` on
  every call. `modelUsage` still names `claude-sonnet-5-5`
  (`canonicalModel` the same, `costBasis: "unknown"`), so the model served is
  the one asked for. The warning is the CLI's own model table lagging, not a
  substitution.

**Measured since (2026-10-20):** a modelpass subscription call is lean. One
structured relevance-filter call through `claude-sub` / `claude-haiku-5-5`,
with a 935-character prompt (~235 tokens), reported usage
`input_tokens 2, cache_write_tokens 1245, output_tokens 110`. So modelpass
adds about 1,000 tokens (the form and a minimal setup), not the ~60K a bare
`claude -p` carries. The per-call overhead is not what caps throughput here.

**Not yet known (measure first):**

- Whether the same ~1,000-token overhead holds on other runtimes and on
  unstructured calls, and how it splits between the form and the setup.
- Whether the cap is requests, tokens per minute, or concurrent sessions per
  account, and how it is shared with interactive Claude Code sessions on the
  same account.

**Proposal:**

1. **Measure it, before anything else.** Have `modelpass verify` (or a bench
   probe) report, per call, the input tokens sent (fresh, cache-write,
   cache-read), output tokens, wall time and the model served (from
   `modelUsage`), for a minimal stateless call on each subscription runtime.
   Record it in the receipt so callers can see the overhead they pay.
2. **A minimal-context mode for stateless calls**, if step 1 shows overhead.
   For calls that need no tools, no files and no session (graders, judges,
   classifiers), run the CLI with the least context it allows: no tools, no
   project instructions, no MCP servers, no memory, and the smallest system
   prompt the subscription terms allow. It is opt-in per request and recorded
   in the receipt, so a caller's certification can bind to it.
   - **Guard:** the payload the model sees must not change silently. A
     minimal-context call is a different route configuration, and callers that
     certify a grader on one route must be able to tell. RAGauge treats the
     access route as outside a grader's identity, so this change must be
     surfaced, never applied quietly to existing calls.
3. **Report throttling, never hide it.** When calls slow under concurrency,
   say so: queue time against model time per call, and a back-off signal the
   caller can act on (reduce workers), rather than letting latency double
   without saying why.
4. **Optional: a batch path for API-key connections.** Where the connection is
   an API key, the Message Batches API (half price, asynchronous) suits
   thousands of independent calls. A caller-visible option, not a silent
   switch.
5. **Keep the model-table warning legible.** When the CLI does not recognise a
   model id, modelpass should report "unrecognized by the CLI, served as
   requested" (from `modelUsage`) once per run. Today it is a raw line on
   every call, which reads like a failure.

**Done when:**

- A receipt shows a stateless call's real token overhead and wall time, and
  the served model.
- The minimal-context option exists, tested, with the payload difference
  stated.
- A RAGauge complete-grading run on a subscription connection reaches a pace
  set by the account's cap, with the overhead per call known rather than
  guessed.

## Keep the vendor SDKs and API libraries current (proposed 2026-11-05)

**Why.** Modelpass depends on packages it does not control, and nothing tells
anyone when they move. The desktop apps update themselves, but none of these
packages do. All three API libraries are now a major version behind, so their
caps in `pyproject.toml` block an upgrade until someone evaluates it.

**Measured (2026-11-05):**

| Package | Runtime | Declared | Installed | Latest |
| --- | --- | --- | --- | --- |
| `claude-agent-sdk` | `anthropic-sdk` | `>=0.2.159` | 0.2.178, with CLI 2.1.318 | 0.2.178 |
| `@openai/codex` (npm, on `PATH`) | `openai-sdk` | not declared | 0.151.0 | 0.166.0 |
| `openai-codex` (pip) | `openai-sdk` | no version | 0.154.0 | 0.163.0 |
| `anthropic` | `anthropic-api` | `>=0.40,<1` | 0.97.0 | 2.5.0 |
| `openai` | `openai-api`, `openai-compatible` | `>=2,<3` | 2.32.0 | 3.46.0 |
| `google-genai` | `google-api` | `>=1,<2` | 1.73.1 | 2.34.0 |

**Re-measured (2026-11-10):** `claude-agent-sdk` was upgraded to 0.2.180, whose
bundled CLI (2.1.321) carries `claude-fable-5-2`; the 0.2.178 bundle (CLI 2.1.318)
did not. Nothing else was upgraded. Two things moved since the table above:
`google-genai` is now at **3.0.0**, so its `<2` cap is two majors behind, and
`langchain-core` (declared `>=1,<2`, installed 1.3.2) was not in the table at all and
is at 1.7.7. A new Claude model reached the bundled CLI two releases after the
global Claude Code had it, which is what this entry exists to catch.

| Package | Declared | Installed | Latest (2026-11-10) |
| --- | --- | --- | --- |
| `claude-agent-sdk` | `>=0.2.159` | 0.2.180 | 0.2.180 |
| `@openai/codex` (npm) | not declared | 0.151.0 | 0.167.0 |
| `openai-codex` (pip) | no version | 0.154.0 | 0.167.0 |
| `anthropic` | `>=0.40,<1` | 0.97.0 | 2.7.0 |
| `openai` | `>=2,<3` | 2.32.0 | 3.49.0 |
| `google-genai` | `>=1,<2` | 1.73.1 | 3.0.0 |
| `langchain-core` | `>=1,<2` | 1.3.2 | 1.7.7 |

**What is known about `anthropic`, from reading the packages and Anthropic's
upgrade guide:**

- 1.x no longer accepts `temperature`, `top_p` or `top_k` as arguments, and moves
  the HTTP layer from `httpx` to `httpx2`.
- 2.x sends API version `2026-10-21`, where a request carrying any of those three
  fields returns a 400 on every model. `Sampling` on `anthropic-api` would have to
  drop them and say so, which is a change callers can see.
- Per-message effort is typed only in 2.x. 0.125.0 and 1.0.0 have neither the beta
  name nor the message type, and 1.17.0 has the beta name only.

**Not yet read:** the `openai` 3.x and `google-genai` 2.x and 3.x changes. Nothing here
says those upgrades are safe or unsafe.

**Proposed work:**

1. **Evaluate the subscription SDKs on each release that matters.** Move
   `claude-agent-sdk` and the Codex CLI forward, and confirm the receipt still names
   the binary and its version. Check for new controls, such as a `set_effort` on
   `ClaudeSDKClient`.
2. **Update the Anthropic API library first, then check the others.** `anthropic`
   2.7.0 is the one that matters most: it types per-message effort, which would make
   the Fable 5.2 and Mythos 5.2 `supported` rows `reachable`. Then work through
   `openai`, `google-genai`, `langchain-core` and the Codex packages in turn.
   Evaluate each API library across its major version. One library at a time:
   read its migration guide, list what breaks, and decide whether to raise the cap.
   Each is its own change, because each can alter what callers see.
3. **Re-read what Modelpass recorded from the old versions.** Every
   `SDK_VERSION_READ` and every dated capability note was read off a specific
   version, so each one moves with that package or is marked as older than it.
4. **Make the check repeatable.** One command or script that prints declared,
   installed and latest for every package above, so this table does not have to be
   rebuilt by hand.

**Done when:**

- Each package above has a recorded decision: upgraded, or held with the reason.
- The `pyproject.toml` caps match those decisions, and the suite passes on them.
- Live tests have been run by the owner on anything upgraded.
- Installed and latest versions can be listed with one command.

## Sessions for the API runtimes (proposed 2026-11-05)

**Why.** `new_chat`, `resume_chat` and `session.send` work on the subscription
runtimes, but an API connection such as `anthropic-api` refuses them, because the
vendor API holds no conversation between calls. Each caller then has to keep and
resend the history itself. The King's Unit harness hit this when moving its AI seats
from a subscription to API credits: it plays through chats, so its own
`anthropic-api` connection could not be used without a replay layer of its own.

A default session on the API runtimes would give every project that layer once.
It would also make prototyping AI chat apps quicker: the same chat code would run on
a subscription or an API key, and changing between them would mean naming a
different connection.

**Proposed work:**

1. **A default session mechanism for the API runtimes.** `new_chat` and `send` keep
   the history and replay it on each call, with the same events and terminal usage
   as a subscription session.
2. **Resume.** `resume_chat` reopens a saved session by its id, so the history has
   to be stored somewhere the caller can reach, or supplied by the caller.
3. **Prompt caching by default.** Place the breakpoints so a replayed history reads
   from the cache, with the lifetime set per connection or per block (for example a
   1-hour system prefix and a 5-minute conversation tail).
4. **Schemas.** `new_chat(schema=)` works as it does on a subscription session.
5. **History limits.** Decide what happens as a conversation nears the model's
   context limit, and leave the caller a way to choose.

**Done when:**

- A chat written for a subscription connection runs unchanged on an `anthropic-api`
  connection, including resume and a reply schema.
- A replayed conversation reads its earlier turns from the cache, and the usage
  shows it.
- Live tests have been run by the owner.
