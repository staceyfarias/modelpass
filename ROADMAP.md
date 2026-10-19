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

**Not yet known (measure first):**

- How many input tokens a modelpass subscription call actually sends around a
  ~1,000-token prompt. A RAGauge certification run earlier recorded 7,282
  tokens around a 10-word message on a "polluted" route, then re-certified on
  a clean one, so modelpass may already strip much of the harness. Unmeasured
  for this workload.
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
