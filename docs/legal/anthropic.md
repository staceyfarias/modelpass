# Using modelpass with Anthropic (Claude Pro / Max / Team / Enterprise)

> **Read this first.** This page is a plain-language summary written by the modelpass
> project, last verified against the sources below on **2026-08-16**. It is **not
> legal advice**, it is not authoritative, and it may be wrong or stale by the time
> you read it. **Anthropic's own terms are the only authority, and Anthropic may
> change them at any time, without notice.** Verify the primary sources yourself
> before relying on anything here. Your account is your responsibility; modelpass is
> MIT-licensed software provided AS IS, and its authors accept no liability for how
> you use it or for what a vendor does to your account.
>
> The library was called `subpass` until **2026-09-13**; only the name changed, and
> no reading on this page was re-verified when it did.

## What modelpass does with Anthropic

**This page is about the *subscription* runtime, `anthropic-sdk`.** modelpass also
ships `anthropic-api`, which is an ordinary metered API-key client under Anthropic's
commercial API terms; nothing on this page constrains it, because it never touches a
subscription credential.

modelpass drives the **Claude Agent SDK / Claude Code runtime** — Anthropic's own
first-party software — under **your own existing login** (`claude /login`), on your
own machine. It never extracts, stores, replays, or transmits your OAuth token; it
never calls Anthropic's API directly with subscription credentials; it has no server
component and never touches anyone else's account.

## What Anthropic's terms say (as of 2026-08-16)

From the [Claude Code legal and compliance page](https://code.claude.com/docs/en/legal-and-compliance):

* *"OAuth authentication is intended exclusively for purchasers of Claude Free, Pro,
  Max, Team, and Enterprise subscription plans and is designed to support ordinary
  use of Claude Code and other native Anthropic applications."*
* *"Anthropic does not permit third-party developers to offer Claude.ai login or to
  route requests through Free, Pro, or Max plan credentials **on behalf of their
  users**."*
* *"Advertised usage limits for Pro and Max plans assume ordinary, **individual**
  usage of Claude Code **and the Agent SDK**."*
* Developers building products or services "should use API key authentication" —
  steering language aimed at hosted products serving users.
* "Anthropic reserves the right to take measures to enforce these restrictions and
  may do so without prior notice." Server-side enforcement exists: consumer OAuth
  tokens error on direct API requests outside Claude Code.

The [Agent SDK credit support article](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)
explicitly lists **"third-party apps that authenticate with your Claude subscription
through the Agent SDK"** as a use covered by the plan's monthly Agent SDK credit.

## What that means for modelpass use — our reading, hedged

The prohibition targets a **developer mediating access through other people's
consumer credentials** (offering claude.ai login in your product; running a service
that spends your users' plans, or many users spending yours). The covered shape is an
**individual running software locally under their own login through the Agent SDK**
— which is precisely and exclusively what modelpass does. This is the strongest legal
footing of any vendor here: the permitted shape is written down affirmatively, not
merely un-prohibited.

## What you must not do

* Do **not** deploy modelpass (or anything built on it) as a hosted service where other
  people's requests run on your subscription, or your service touches their login.
* Do **not** extract the OAuth token from `~/.claude/.credentials.json` (or the OS
  keychain) and use it anywhere else. Anthropic restricted consumer-plan OAuth to its
  own apps in the 2026-02 update, and extracted tokens are refused server-side.
* For hosted or multi-user production workloads, use the Claude API under its
  commercial terms.

## Primary sources

| Source | What it governs |
| --- | --- |
| [Consumer Terms of Service](https://www.anthropic.com/legal/consumer-terms) | Free / Pro / Max accounts |
| [Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms) | Team / Enterprise / API |
| [Usage Policy](https://www.anthropic.com/legal/aup) | All use |
| [Claude Code: legal and compliance](https://code.claude.com/docs/en/legal-and-compliance) | Authentication and credential use |
| [Agent SDK credit article](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan) | What the plan's credit covers |

**These documents change.** The authentication section above was added in a 2026-02
update; the Agent SDK credit is a 2026-06 addition. Re-read them before building
anything that depends on this page.
