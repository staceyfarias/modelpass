# Guards: bounding what a run can spend

Guards are D4's *offered* layer: configurable per connection, owned by the user.
This page is where the numbers live, because **modelpass ships none**.

Rewritten 2026-09-13, when the shape of the library changed under it. Four of the
six runtimes you can configure are now **metered** — `anthropic-api`,
`openai-api`, `google-api`, `openai-compatible` — and on those a run does not
draw down an allowance you have already paid for, it adds a line to a bill.
Everything below was written for a world where the metered case was the
exception. It is now the majority, and the rulings that survived that change
survived it for reasons, which is what most of this page is about.

## Why there are no defaults

An earlier draft shipped `warnAtTokens = 200000` and `stopAtTokens = 1000000` as
per-run defaults. Both were invented in the implementation plan — not derived
from a plan's allowance, not measured against a real run, not sourced from a
vendor. A made-up number presented as a product default reads as advice, and
modelpass is not in a position to give that advice: it does not know your plan,
your month, or what you are about to ask for.

So the rule (D4 amendment, 2026-08-17) is: **a guard exists when, and only when,
you configure it.** What modelpass does instead of defaulting is make the absence
visible —

* the preflight receipt carries `guards_configured` and its `summary()` ends
  with `-- no spend guards configured`;
* `modelpass connect` prints that prominently at setup time, next to the
  receipt, and names no number while it does it;
* `modelpass list` shows `guards    none (nothing bounds one run's spend)` per
  connection, and the bench tags the row `no spend guards`;
* the run log records `guards_configured` per run, so a ledger can tell a
  bounded run from an unbounded one afterwards.

Silence about spend was the failure mode worth designing against, not the
absence of a threshold. Going metered-by-majority did not change that: an
invented number is no better advice about a bill than it was about an allowance.

## Picking a number

Guards are **per run**, in **tokens**. Per run because that is what modelpass can
see: it observes one call at a time, and a monthly ledger kept locally would be a
second, lyingly authoritative accounting system sitting next to the vendor's.
Tokens because a token is the one unit every runtime reports and every vendor
bills against — see [Dollar thresholds](#dollar-thresholds-are-not-in-the-schema)
below for why the obvious second unit is still not here.

Three questions worth answering before writing a number down:

1. **What does one normal call cost you?** Run the work you actually do with no
   guards, watch the `usage` events, and take the observed total. Every runtime
   reports tokens; the terminal event carries the run total.
2. **What is the biggest run you would not want to be surprised by?** That is
   your `stopAtTokens`. It is a circuit breaker, not a budget — set it where you
   would rather the run died than continued.
3. **Where do you want to hear about it first?** That is `warnAtTokens`. A
   warning is free; it fires once and the run continues.

A workable starting shape, stated as a method rather than a number: set
`warnAtTokens` around 2–3× a typical run and `stopAtTokens` around 10× it, then
adjust the first time either one fires for a reason you disagree with.

**On a metered connection, do this twice.** Once here, and once at the vendor,
where the ceiling is enforced rather than observed. The two are not alternatives;
see [the hard cap](#the-hard-cap-belongs-at-the-vendor).

Tool-bearing runs (D12) are the case that most needs a guard. A tool loop has no
turn ceiling by default — it runs to completion, which is the vendor's own
behavior — so the token guard is the only thing bounding it. If you want a turn
ceiling as well, that is `options={"max_turns": N}`, set explicitly by you.

## Configuration

```toml
[connections.claude-api.guards]
warnAtTokens = 150000
stopAtTokens = 400000

[connections.claude-api.guards.onQuotaExhausted]
action = "stop"
```

`0` means disabled, and omitting a key means the same thing. Omitting the whole
`[guards]` table is what a connection with no guards looks like, which is what
`modelpass connect` writes unless you pass `--warn-at-tokens` /
`--stop-at-tokens`.

Per-call overrides never touch the file. The shorthand is usually what you want:

```python
bridge.chat(connection="claude-api", message=..., stop_at_tokens=50_000)
```

It starts from the connection's own guards, so anything you do not mention stays
as configured — including `onQuotaExhausted`, because tightening a ceiling for
one call should never quietly disarm a failover you set up. `0` switches a guard
off for the call. `guards=Guards(...)` is still there for replacing the lot at
once; passing both is an error rather than a precedence rule.

**Omitting is not disabling, and each guard is its own switch.** Both halves of
that trip people up, so concretely, against a connection configured with
`warnAtTokens = 100000` and `stopAtTokens = 200000`:

| The call | Effective guards |
| --- | --- |
| no guard arguments at all | warn 100,000 · stop 200,000 — *the connection's, not none* |
| `stop_at_tokens=0` | stop off · **warn still 100,000** |
| `stop_at_tokens=0, warn_at_tokens=0` | both off — the way to run one call unguarded |
| `stop_at_tokens=50_000` | stop 50,000 · warn **clamped to 50,000** |

So a call site that says nothing about guards is *bounded by the connection*,
which is the right default and is not always what the person reading that call
site expects. And switching the stop off leaves the warning armed, because a
warning that costs nothing to keep is not the thing you were disabling.

**A warn above a stop is clamped, not refused** (added 2026-08-17, on
first-consumer feedback). Lowering only the stop is the obvious gesture, and it
used to fail unless you remembered to lower a warn you had not mentioned — a
warning that can never fire is unreachable, not a contradiction, so it is pulled
down to the stop instead. In the *file* the same pair is still refused, loudly,
because there it is a typo you would otherwise trust. Bad per-call guard values
raise `InvalidGuards`, which is both a `SubpassError` and a `ValueError`, and
deliberately not a config error: it is your argument that is wrong, not your
connections file.

## When guards actually fire

A guard can only act on usage the runtime has reported, so *when* it can
interrupt is a per-runtime fact, not a promise. The capability registry reports
it as `interim_usage`:

| Runtime | Interim usage | What that means for a guard |
| --- | --- | --- |
| `anthropic-sdk` | supported | Usage arrives with each assistant turn (`AssistantMessage.usage`), so a guard can stop a tool loop **between rounds**. |
| `openai-sdk` | **supported** (2026-08-31) | `thread/tokenUsage/updated` arrives after **each** model response, mid-turn, on the default `codex app-server` transport — so a stop interrupts the run it is watching. On `options={"transport": "exec"}` it does not: `codex exec --json` carries no usage before `turn.completed`, which arrives once, at the end, and a guard there is a **post-hoc** report on that run and a bound on the *next* one. |
| `anthropic-api` · `openai-api` · `google-api` | **supported** (tickets 1.6, 1.9, 1.11) | Every vendor response carries its own usage, and the tool loop is **in this process** — so a guard is checked between rounds by code you can read, not by a subprocess you are asking nicely. This is the strongest position a guard has anywhere in modelpass. |
| `openai-compatible` | unverified | The row describes an API *shape*; what answers is whatever your `baseUrl` points at. `modelpass verify` drives your endpoint and records what it does. Until it has, treat a guard here as post-hoc. |
| `google-cli` / `google-sdk` | unverified | Not checked; the runtimes are gated anyway (D5). |

**Where the in-adapter tool loop is concerned, "between rounds" is the whole
promise.** On the four API runtimes a tool-bearing call is a loop this process
drives: send, read the response, run the handlers, send again. The usage on each
response is folded in as it arrives, and the thresholds are evaluated *there* —
between one round and the next. So a `stopAtTokens` ends the loop before the
round that would have crossed it is sent, and the terminal event is a
`guard_stop` carrying what was spent up to that point. It does not reach inside a
round: a single response that blows through the ceiling on its own is billed in
full, because the vendor had already produced it by the time anything could be
counted. That is why `max_output_tokens` and a guard are different instruments,
and why the first is worth setting on a connection whose single responses can be
large.

This is also why `stopAtTokens` is not a hard ceiling anywhere. It is the point
at which modelpass stops asking for more, as soon as it is in a position to know.

**The Codex row is transport-dependent, and on 2026-08-31 it moved.** The
default transport became `codex app-server`, which sends
`thread/tokenUsage/updated` after **each** model response, mid-turn, carrying
`last` and `total` breakdowns and `modelContextWindow` — driven live the same
day. So a `stopAtTokens` on a Codex connection is now a real circuit breaker.
`options={"transport": "exec"}` opts back out of that along with everything
else the older transport does differently, and the preflight receipt on such a
run says in as many words that the row you read describes a default this call
did not take. See [api-and-runtimes.md](api-and-runtimes.md) and
the 2026-08-31 transport migration record (internal).

**If you set a Codex ceiling while it could never fire, re-read the number.** A
threshold chosen on the understanding that it was a post-hoc report is a
different decision from one that will actually stop a run part-way, and the
generous number that was safe under the first reading may not be the number you
want under the second.

## When the allowance runs out

A spent subscription allowance is a **clean stop**, not an error: nothing
failed, the plan ran out. The stream ends with a `terminal` event carrying
`status = "quota_exhausted"`, and `raise_on_stop=True` turns that into
`QuotaExhausted`.

Stopping is the only default, and there is no way to make it implicit.

### Failover, if you ask for it by name

```toml
[connections.claude-sub.guards.onQuotaExhausted]
action = "failover"
failover = "claude-api"
```

That is the whole opt-in, and naming the second connection *is* the consent —
including when the second one is metered. **That sentence carries more weight
now than when it was written**, because most of the connections you might name
are metered: the common failover is a subscription that ran out handing the work
to a key that bills. Nothing about the mechanism changed; what changed is how
often the thing it consents to is the expensive one. Write the name deliberately.

What modelpass guarantees around it:

* **The second connection's own preflight runs first**, and its receipt reaches
  you before it does any work: a `failover` event announcing the switch,
  immediately followed by a `receipt` event carrying the second connection's
  receipt. Same evidence you would see at setup time — which credential, which
  account, which model, what was scrubbed. (Until 2026-08-17 the receipt was a
  dict on the `failover` event itself; it moved so that every receipt in a
  stream arrives the same way.) If that preflight fails, the failover does not
  happen — the first connection's `quota_exhausted` terminal is emitted with the
  reason extended to say why.
* **The terminal names both connections.** `connection` and `auth_mode` are the
  ones that finished; `failed_over_from` is the one that started. A run that
  began on a subscription and ended on a key cannot be mistaken for one that was
  metered throughout (D3), in the stream or in the run log.
* **It never chains.** A failover target that is itself configured to fail over
  is refused, as is a connection naming itself.
* **It never contradicts an assertion.** With `expect_auth_mode=` set, a
  failover target that does not match is refused before the first run starts.
* **The second connection uses its own guards**, from its own config. A run that
  has already spent one allowance does not arrive at the next one half-budgeted.
  On a metered target that is the guard that matters, and it is a separate
  number you have to have written down.

`FailoverEvent.crosses_to_metered` is the one property to check if you want to
log loudly, or refuse, when a call moves onto metered billing.

Failover is deliberately the only place in modelpass where one `chat()` call
touches two connections.

## The hard cap belongs at the vendor

A guard is an observation modelpass acts on. A **hard cap is enforcement**, and
it lives where the billing does. On a metered connection you want both, and the
cap is the one that is still true when this library is not in the call path —
when something else on the machine is using the same key, when a run is
interrupted in a way nothing gets to fold, when the vendor's accounting and yours
disagree. modelpass reports what it was told; the cap is what stops the bill.

Set one per vendor, in the vendor's own console. Their limits, their units, their
wording — so this page links rather than restates, and cannot go stale in a way
that costs you money:

* **Anthropic** (`anthropic-api`) — spend limits and usage credits on the
  organization: [console.anthropic.com/settings/limits](https://console.anthropic.com/settings/limits).
  Leaving usage credits disabled is a $0 ceiling, which is the strongest version
  of this and the right state for a key you only want a subscription to back up.
* **OpenAI** (`openai-api`) — per-project budget limits on the key:
  [platform.openai.com/settings/organization/limits](https://platform.openai.com/settings/organization/limits).
  A key scoped to a project with its own budget is the shape to aim for, because
  it caps *this* use of the API rather than everything you do.
* **Google** (`google-api`) — the key bills through Cloud, so the cap is a
  billing budget with a threshold action:
  [cloud.google.com/billing/docs/how-to/budgets](https://cloud.google.com/billing/docs/how-to/budgets).
  Note that a budget alert is not by itself a stop; the docs say which actions
  are enforcement.
* **`openai-compatible`** — there is no vendor to ask, because the vendor is
  whatever your `baseUrl` points at. A local Ollama or LM Studio costs
  electricity and the question does not arise. A hosted gateway is a vendor like
  any other and will have its own answer; find it before you point a key at it.

The two subscription runtimes have no equivalent because they have no bill: the
plan is the cap, and running out of it is the clean stop described above.

## Dollar thresholds are not in the schema

Decided 2026-08-17 as D4 amendment (b), recorded as decided rather than
deferred — and **re-examined on 2026-09-13, when the metered runtimes became the
majority, because that is exactly the change that could have overturned it.**

It did not. The original argument was that only one runtime reported a dollar
figure at all (`ResultMessage.total_cost_usd` on `anthropic-sdk`), so a
`stopAtUSD` would have silently worked on one runtime and silently done nothing
on the other — "precisely the kind of guard someone would set and then trust".
Four metered runtimes later the asymmetry has not gone away, it has changed
shape:

* A price is a **fact about your contract**, not about the response. The same
  token count costs different amounts on different tiers, under different
  discounts, with cached input priced differently again, and with batch and
  long-context multipliers that a client cannot see. Converting tokens to dollars
  locally means shipping a price table, and a price table in a library is wrong
  the week a vendor changes one and silently wrong until someone notices.
* `openai-compatible` has **no price at all** that modelpass could know. The same
  connection type covers a free local box and a metered gateway.
* A guard denominated in a unit modelpass cannot verify would be the one number
  on this page that is not sourced from something the runtime actually said.

So the unit stays tokens, which every runtime reports and every receipt and run
log carries, and **the dollar ceiling stays at the vendor**, where it is
enforced rather than estimated. That is not a smaller promise than `stopAtUSD`
would have been; it is a bigger one, kept in the only place it can be kept.

## `retry = "never"`, and why it lives near the guards

A guard bounds one run. `retry` bounds what happens *after* one, and on a metered
connection the two failure modes people actually hit are "one run cost more than
I meant" and "a loop somewhere ran that call four times".

```toml
[connections.claude-api]
retry = "never"
```

**It does not make modelpass retry anything — modelpass never does.** The library
produces a *verdict*, `Retryable.YES` / `NO` / `UNKNOWN`, computed only from
typed facts: HTTP status codes, exception classes, terminal statuses, never
message text. Your loop decides. What `retry = "never"` does is force every
verdict produced on this connection to `NO`, so a consumer's own loop stops here
without that consumer having to learn which of its connections must not be
repeated.

The stance is on the preflight receipt (`Receipt.retry`, and `retry_note` spells
it out), so it is visible **before** a run rather than inferred from a verdict
afterwards — the same bargain `guards_configured` makes. `modelpass list -v`
prints the line when a connection declares it, and the bench shows the stance on
every row.

Worth setting on: a connection whose calls are expensive enough that a
second attempt is a decision rather than a reflex; an endpoint that is not
idempotent in practice; and the metered half of a failover pair, where a retry
after the switch spends twice on the run you were already unhappy about.

The verdict vocabulary and the rules behind it are in the README's
*Timeouts, retries, and threads*; the per-run record of how a call ended — and
the verdict that follows from it — is on the bench's run-log page.

## What guards do not claim (D4, disclaimed layer)

Local accounting is best-effort token counting from the events the runtime
chooses to emit. **The vendor's bill is the truth.** modelpass reports what it was
told, folds it in as it arrives, and stops when you asked it to — nothing in
here is a billing system. On a metered connection that sentence is the reason the
vendor-side cap above is not optional advice.
