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
>
> **Updated 2026-11-08** for OpenAI's [Sign in with ChatGPT Terms](#the-sign-in-with-chatgpt-terms-read-2026-11-08).
> The rest of the page was not re-verified at that time.

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
* The [Terms of Use](https://openai.com/policies/terms-of-use/) (published and
  effective 2026-01-01; confirmed to load 2026-09-27) say: *"You may not share your
  account credentials or make your account available to anyone else."*
* Our 2026-08-15 verification **found no prohibition** on personal programmatic use
  of Codex under ChatGPT auth. OpenAI's guidance *prefers* API keys for automation
  and CI — steering language, not a stated ban.

## The Sign in with ChatGPT Terms (read 2026-11-08)

OpenAI now publishes [Sign in with ChatGPT Terms](https://openai.com/policies/sign-in-with-chatgpt-terms/).
A web search reports the English page as dated **2026-09-29**. We have not confirmed
that date ourselves, because the page refuses automated fetches. We read the text
on 2026-11-08, from a copy the maintainer took from that page.

**They do not bind modelpass directly, as far as we can tell.** They govern an
application that builds Sign in with ChatGPT ("SIWC") into itself. modelpass does
not: the sign-in is done by Codex, OpenAI's own application, and modelpass never
runs the sign-in flow or handles the tokens it produces. These are still the
clearest statement OpenAI has made about third-party software using a ChatGPT plan,
so we read modelpass against them. What they require of an app that uses a user's
plan (§3.1, the "ChatGPT Plan" scope):

* **Tokens only through OpenAI's flow, stored locally under the user's control**
  (§1). Never ask for ChatGPT passwords or session cookies.
* **No multiple accounts, split usage, rotated accounts, or other ways around
  usage limits.** No pooling, transferring, reselling, gifting or sharing plan
  usage or tokens. Never use one user's subscription to answer another user's
  requests.
* **User control.** Requests come from the user's own local runtime, or from a
  remote runtime that only that user controls.
* **User authorization.** Requests arise from the user's own activity or from
  automations they expressly authorized. Background use needs express consent.
* **Connected application only.** The plan is used only for the application the
  user connected. *"Do not provide general-purpose API access for other tools or
  unrelated requests."*
* **No charge.** Users must be able to use their plan without paying the app.
* Plan eligibility and usage limits still apply. SIWC grants no extra usage.

How modelpass sits against that, our reading:

| Requirement | modelpass |
| --- | --- |
| Tokens through OpenAI's flow, kept local | Yes. The login is `codex login`; the tokens stay in Codex's own store. modelpass reads `auth.json` only to report whether the access token has expired, and never sends or replays it. |
| Local, user-controlled runtime | Yes. Codex runs as a child process on your machine, and the bench listens on loopback only. |
| No charge | Yes. modelpass is MIT-licensed and free. |
| No sharing, pooling, or serving other users | Not done by modelpass, and forbidden below. It is your responsibility not to build it on top. |
| No rotating accounts or getting around limits | **Watch this one.** modelpass can fail over from one ChatGPT login to another when the first runs out of quota (`onQuotaExhausted.failover` between two `openai-sdk` connections). That is close to the account rotation these terms prohibit. Do not configure it. |
| Connected application only, no general-purpose access | **The closest fit to what is prohibited.** modelpass exists so that programs you write can use your plan. That is not a hosted proxy, but it is a general-purpose library, which is closer to what this clause describes than anything else modelpass does. |

## What that means for modelpass use — our reading, hedged

Individual, local, ChatGPT-authenticated use of OpenAI's own Codex runtime appears
to be an intended plan feature, and modelpass only ever drives that runtime. **Be
aware the footing here is weaker than Anthropic's**: OpenAI documents the behavior
and sells plans on it, but we found no statement blessing third-party software that
wraps the Codex CLI.

OpenAI has stopped being silent about third-party apps, though. Its Sign in with
ChatGPT Terms give such apps a sanctioned way to use a user's plan, and with it a
set of explicit limits, summarized above. They do not name modelpass's pattern,
which is driving OpenAI's own runtime rather than integrating SIWC. We read them as
the best available guide to where OpenAI draws the line, and we recommend staying
inside those limits even where they may not bind you. If OpenAI says more, in either
direction, this page is wrong until it is updated.

## What you must not do

* Do **not** proxy or share your ChatGPT-authenticated access — no hosted service
  where others' requests draw your plan, and no touching other people's logins.
* Do **not** extract tokens from `~/.codex/auth.json` or run OAuth-proxy shims that
  expose your ChatGPT auth as an API endpoint. Community packages exist that do
  this; that is the token-level pattern modelpass exists to avoid, and it lives at
  the vendor's tolerance, which can end without notice.
* Do **not** use more than one ChatGPT login to get around a plan's usage limits.
  In particular, do not set `onQuotaExhausted.failover` on an `openai-sdk`
  connection to point at another `openai-sdk` connection. Failing over to a
  metered `openai-api` connection is a different case: it moves the work onto API
  billing, which the SIWC Terms explicitly leave alone.
* Do **not** let anyone else's activity trigger runs on your login. That includes
  a tool or service other people use that runs modelpass behind it.
* For hosted or multi-user production workloads, use the OpenAI API under its own
  terms and billing.

## Primary sources

| Source | What it governs |
| --- | --- |
| [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/) | ChatGPT accounts |
| [Usage Policies](https://openai.com/policies/usage-policies/) | All use |
| [Sign in with ChatGPT Terms](https://openai.com/policies/sign-in-with-chatgpt-terms/) | Third-party apps that integrate Sign in with ChatGPT, including use of a user's plan (§3.1) |
| [Using Codex with your ChatGPT plan](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan) | Plan allowance and eligibility |
| [Codex rate card](https://help.openai.com/en/articles/20001106-codex-rate-card) | Allowance accounting |
| [Codex SDK documentation](https://learn.chatgpt.com/docs/codex-sdk) | The runtime modelpass drives |

**These documents change.** Codex moved to token accounting in 2026-04 and swapped
its model lineup on 2026-08-31, after this page was last verified. Re-read them
before building anything that depends on this page.
