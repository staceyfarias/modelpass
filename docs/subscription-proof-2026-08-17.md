# How do I know it was the subscription?

**2026-08-17.** The permanent answer to the question the whole project rests on.

A tool that promises "this runs on the plan you already pay for" is making a claim
about somebody else's bill. The claim is worth exactly as much as the evidence behind
it, so this page records the evidence: a falsification experiment run on this machine
today, the control that keeps it honest, and the ledger you can check without trusting
anything written here.

---

## The short version

Three independent things, in increasing order of how much they depend on modelpass:

| | What it proves | What you have to trust |
| --- | --- | --- |
| **The vendor's usage page** | Nothing was billed to a key | Nothing. It is the vendor's own ledger. |
| **The falsification experiment** | A key could not have paid for it | The experiment's setup, reproduced below |
| **`~/.modelpass/runs.jsonl`** | Which mode paid for *each* run, months later | modelpass's own stamp (D3) |

The first is the one that settles arguments. The third is the one you will actually
use, because nobody reconstructs a month of runs from memory.

---

## The falsification experiment (2026-08-17, machine-verified)

The design: make it **impossible** for the API key to be the thing that worked, then
watch a real generation succeed anyway.

### Setup

The only `OPENAI_API_KEY` in the environment was set to a deliberately invalid string.
Not absent — *present and wrong*. An absent key proves little (a runtime might fall
back to something else and you would never know which); an invalid key cannot
authenticate anything at all, so any successful generation must have been paid for by
something else.

### Result

A real generation succeeded through modelpass:

* terminal event stamped `auth_mode=subscription`
* 54,074 total tokens

An invalid key cannot authenticate. The generation happened. Therefore the ChatGPT
login paid for it. That is the whole argument, and it does not depend on reading any
of modelpass's own code.

### The spawn-time audit

A separate check on the child process modelpass launched confirmed the scrub does what
the receipt says:

* `OPENAI_API_KEY` — **absent** from the child environment
* `OPENAI_API_KEY-STAGING` and other suffixed lookalikes — passed through, harmlessly

The lookalikes are worth naming rather than quietly filtering: the scrub matches exact
variable names and documented prefixes, so a variable whose name merely *contains* a
scrubbed one survives. That is correct — `OPENAI_API_KEY-STAGING` is not a variable the
Codex runtime reads — but it means the receipt's `scrubbed` list should be read as
"these exact names were removed", not "nothing resembling a key got through".

---

## The control, which is the part that keeps this honest

**`codex exec` was then run directly, with the same bogus key, unscrubbed. It also
succeeded.**

So the experiment above does *not* prove the scrub was load-bearing on this machine
today. codex-cli 0.117.0 with a ChatGPT login ignores an ambient `OPENAI_API_KEY`
entirely; API-key mode is entered only by an explicit `codex login --with-api-key`.
With that CLI, on that login, the key in the environment was never going to be used
whether modelpass removed it or not.

Reporting this is the point. The tidier story — "we scrubbed the key, therefore the
subscription paid" — would have been a post-hoc justification, and the correct
conclusion is narrower and more useful:

**The scrub is defense in depth against documented traps, not the only line of
defense.** The traps it is aimed at are real and dated:

* the **Codex SDK injects `CODEX_API_KEY`** into the CLI environment (verification doc,
  2026-08-15) — a path that does not go through `codex login`;
* **Anthropic's precedence is the opposite way round**: `ANTHROPIC_API_KEY`,
  `ANTHROPIC_AUTH_TOKEN`, `apiKeyHelper` and the cloud-provider switches all outrank
  subscription OAuth, so on that runtime an ambient key silently wins;
* **precedence changes between releases**, and this is a space where the CLI flags,
  package names and quota rules have all moved within six months.

A guarantee that holds because of today's CLI version's behavior is not a guarantee.
The scrub makes it hold because of modelpass's behavior, which is the version this
project can actually promise.

---

## The ledger you can check yourself

The strongest evidence is the one modelpass has no hand in.

**`platform.openai.com/usage`.** Each of these runs sends roughly 50k input tokens.
If the key had been billed, the usage page would show ~50k+ input tokens per run,
timestamped, on the account that owns the key. It should show none.

That is a check anybody can run, against the vendor's own records, without trusting
modelpass, this document, or the person who wrote it. If it ever shows those tokens, the
claim at the top of the README is false and should be treated as such.

The Anthropic equivalent is the account's usage view for the same window; the Anthropic
half of the live queue is still blocked on `claude /login` (see the roadmap), so it is
listed here as the check to run, not as a check that has been run.

---

## The durable answer: `~/.modelpass/runs.jsonl`

Everything above is a point-in-time experiment. The question the owner actually asked —
*how do I prove it was the subscription* — is asked in the present tense about runs
that already happened, and the honest answer needed to be something other than "you had
to be watching at the time".

Every run that reaches a terminal appends one line:

```json
{"timestamp":"2026-08-17T14:02:11+00:00","connection":"codex-sub","runtime":"openai-sdk",
 "auth_mode":"subscription","status":"ok","model":"gpt-5.6-sol","input_tokens":50102,
 "output_tokens":3972,"cached_input_tokens":0,"total_tokens":54074,
 "guards_configured":false,"failed_over_from":null}
```

What makes it evidence rather than decoration:

* **`auth_mode` is the mode the preflight detected, and the bridge stamps it.** An
  adapter may report a terminal *status*; it may not claim how the run was billed (D3,
  and the adapter contract's rule 2). The one field somebody would want to forge is the
  one field adapters cannot write.
* **`failed_over_from` keeps a mixed-billing run distinguishable — and accounted.** A
  call that started on a subscription and finished on a metered key is the single case
  where one `chat()` touches two connections. It writes **two lines**, one per leg, each
  with the auth mode and the tokens that belong to it. A single line naming the
  connection that finished would file the subscription's spend under the metered
  account, which is the exact confusion this file exists to prevent.
* **`guards_configured` records whether anything bounded the run.** A ledger that
  cannot distinguish a bounded run from an unbounded one is missing the thing you would
  go looking for.
* **It is a ledger, not a transcript.** No prompts, no responses, no credential, not
  even a pointer to one. Handing somebody this file tells them what was spent and on
  whose account, and nothing about what was said.
* **It is JSON Lines.** `jq`, a spreadsheet, five lines of Python. A proof you can only
  read with the tool making the claim is not much of a proof.

Cross-check it against the vendor's usage page and the two should agree: every
`auth_mode: "api_key"` line has a matching charge, and there is nothing on the bill
that does not correspond to one.

Turn it off with `[settings] runLog = false` in `~/.modelpass/connections.toml`. It is on
by default on purpose — evidence that only exists when you remembered to switch it on
is not evidence.

---

## What this page does not claim

* **Not a billing system.** modelpass reports what the runtime told it. The vendor's bill
  is the truth (D4, disclaimed layer).
* **Not a proof about your machine.** It is a proof about *this* one, on 2026-08-17,
  with codex-cli 0.117.0 and a ChatGPT login. Re-run it: set your key to garbage, run a
  generation, check your usage page.
* **Not settled for Anthropic.** The Anthropic live path is queued behind
  `claude /login`. The scrub, the preflight and the stamp are implemented and
  offline-tested there; the falsification experiment has only been run against OpenAI,
  and this page will say so until it has not.

## Re-verify before relying on any of this

Every fact here is dated 2026-08-17 and several are version-specific (codex-cli
0.117.0). Auth precedence is exactly the kind of thing that changes in a point release,
which is the argument for the scrub in the first place.
