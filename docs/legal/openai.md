# Using modelpass with OpenAI (ChatGPT Plus / Pro / Team / Edu / Enterprise)

> **Read this first.** This page is a plain-language summary written by the modelpass
> project, last verified against the sources below on **2026-08-16**. It is **not
> legal advice**, it is not authoritative, and it may be wrong or stale by the time
> you read it. **OpenAI's own terms are the only authority, and OpenAI may change
> them at any time, without notice.** Verify the primary sources yourself before
> relying on anything here. Your account is your responsibility; modelpass is
> MIT-licensed software provided AS IS, and its authors accept no liability for how
> you use it or for what a vendor does to your account.
>
> The library was called `subpass` until **2026-09-13**; only the name changed, and
> no reading on this page was re-verified when it did.

## What modelpass does with OpenAI

**This page is about the *subscription* runtime, `openai-sdk`.** modelpass also ships
`openai-api` and `openai-compatible`, which are metered API-key clients under
OpenAI's own API terms (or, for `openai-compatible`, whatever governs the endpoint
you point it at); nothing on this page constrains them, because they never touch a
ChatGPT login.

modelpass drives the **Codex CLI** — OpenAI's own first-party, Apache-2.0-licensed
runtime — under **your own "Sign in with ChatGPT" login** (`codex login`), on your
own machine. Runs draw the Codex allowance included with your ChatGPT plan. modelpass
never extracts or replays OAuth tokens, never calls OpenAI's API with subscription
credentials, and actively **scrubs** ambient `OPENAI_API_KEY` / `CODEX_API_KEY`
variables from every launch so a run can never silently land on metered billing.

## What OpenAI's terms and documentation say (as of 2026-08-16)

* [Using Codex with your ChatGPT plan](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan)
  documents ChatGPT-authenticated Codex use — including the CLI and SDK — as a
  feature of the plans, drawing the plan's included allowance, then credits.
* The Codex CLI and SDK are published by OpenAI for exactly this: local,
  programmatic, ChatGPT-authenticated use. Token-based accounting for that
  allowance has applied since 2026-04.
* Our 2026-08-15 verification **found no prohibition** on personal programmatic use
  of Codex under ChatGPT auth. OpenAI's guidance *prefers* API keys for automation
  and CI — steering language, not a stated ban.

## What that means for modelpass use — our reading, hedged

Individual, local, ChatGPT-authenticated use of OpenAI's own Codex runtime appears
to be an intended plan feature, and modelpass only ever drives that runtime. **Be
aware the footing here is weaker than Anthropic's**: OpenAI documents the behavior
and sells plans on it, but we found no affirmative statement blessing third-party
software that wraps the CLI, and no explicit prohibition either. Silence is not
permission; it is silence. If that changes in either direction, this page is wrong
until updated.

## What you must not do

* Do **not** proxy or share your ChatGPT-authenticated access — no hosted service
  where others' requests draw your plan, and no touching other people's logins.
* Do **not** extract tokens from `~/.codex/auth.json` or run OAuth-proxy shims that
  expose your ChatGPT auth as an API endpoint. Community packages exist that do
  this; that is the token-level pattern modelpass exists to avoid, and it lives at
  the vendor's tolerance, which can end without notice.
* For hosted or multi-user production workloads, use the OpenAI API under its own
  terms and billing.

## Primary sources

| Source | What it governs |
| --- | --- |
| [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/) | ChatGPT accounts |
| [Usage Policies](https://openai.com/policies/usage-policies/) | All use |
| [Using Codex with your ChatGPT plan](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan) | Plan allowance and eligibility |
| [Codex rate card](https://help.openai.com/en/articles/20001106-codex-rate-card) | Allowance accounting |
| [Codex SDK documentation](https://learn.chatgpt.com/docs/codex-sdk) | The runtime modelpass drives |

**These documents change.** Codex moved to token accounting in 2026-04 and swapped
its model lineup on 2026-08-31, after this page was last verified. Re-read them
before building anything that depends on this page.
