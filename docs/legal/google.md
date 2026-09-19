# modelpass and Google (Antigravity / Google AI Pro / Ultra): excluded

> **Read this first.** This page is a plain-language summary written by the modelpass
> project, last verified against the sources below on **2026-08-18**. It is **not
> legal advice**, it is not authoritative, and it may be wrong or stale by the time
> you read it. **Google's own terms are the only authority, and Google may change
> them at any time, without notice.** Verify the primary sources yourself before
> relying on anything here. Your account is your responsibility.
>
> The library was called `subpass` until **2026-09-13**; only the name changed, and
> no reading on this page was re-verified when it did.

## The short version

**modelpass does not support Google subscriptions, deliberately, and you should not
try to wire one up through third-party tooling of any kind.** This is the one
vendor where the answer is not "here is how to stay compliant" but "do not."

## What Google's terms say (as of 2026-08-18)

The [Antigravity terms](https://antigravity.google/terms) state:

> "Using third party software, tools, or services to access the Service (e.g. using
> OpenClaw with Antigravity OAuth) **is a breach of this Agreement.**"

with breaches "grounds for **suspension or termination of your account**." Note the
named example: the terms call out a third-party tool by name. Community reports of
real account blocks for third-party Antigravity auth plugins predate our read and
are consistent with it.

## "But Google ships an SDK — doesn't that contradict this?"

It looks like a contradiction and it is not, because **the SDK and the
subscription are two different products under two different agreements.**

* The clause above lives in the Antigravity **Additional Terms**, which govern
  the **subscription** service (Google AI Pro / Ultra quota, reached by
  Antigravity OAuth). That is where the third-party-access ban applies.
* **API-key access is governed by different terms.** The Additional Terms say so
  themselves: a holder of a Gemini Enterprise Agent Platform API key is subject
  to the Google Cloud terms *instead of* these Additional Terms. Metered Gemini
  through an API key is an ordinary commercial product and always was.
* Google's Python SDK is built for **that** path: it requires `GEMINI_API_KEY`
  or Vertex ADC and **cannot authenticate against the subscription at all**.

So the SDK's existence says nothing about subscription access, because the SDK
cannot reach the subscription. The ban is on third-party software reaching the
*subscription* Service — which is precisely, and only, what modelpass would want
to do.

**Re-verified 2026-08-31** against https://antigravity.google/terms: the quoted
sentence is unchanged, there is still no carve-out for third-party tools or
SDKs, and suspension or termination remains the stated consequence.

## Why modelpass excludes Google *as a subscription runtime*

* modelpass is unambiguously "third party software." Unlike Anthropic — whose terms
  prohibit a *specific shape* (developers routing through other users' consumer
  credentials) while explicitly covering individual Agent SDK use — Google's
  sentence is a blanket prohibition with **no carve-out for driving Google's own
  CLI**. Whether wrapping the first-party `agy` binary counts as "third party
  software ... accessing the Service" is at best ambiguous.
* Ambiguity, combined with account-termination stakes and a track record of
  enforcement, fails modelpass's bar: it cannot ship a feature whose realistic
  downside is a stranger's Google account being terminated.
* There is also a structural obstacle: Google's SDK cannot use the subscription at
  all (API key / Vertex ADC only), so the only subscription surface would have been
  the CLI-as-subprocess — the exact ambiguous case.

The `google-cli` / `google-sdk` runtime identities remain in the codebase as gated
placeholders whose capabilities report honestly, so that **if Google ever publishes
an Anthropic-style carve-out**, the adapter work re-opens against a changed fact
rather than a changed design. Until then, requesting them raises.

## If you want Gemini models anyway

Use Google's **metered APIs** under their own terms and billing. Since **2026-09-13**
that is a shipped, ungated runtime: `google-api` takes a Gemini API key through
`google-genai`, and the Antigravity clause quoted above does not reach it — an API
key is governed by the Google Cloud terms *instead of* those Additional Terms, as
they say themselves, and Google's own SDK cannot reach the subscription in any case.
The asymmetry is deliberate: `google-api` ships ungated while `google-cli` and
`google-sdk` stay gated, because gating the API-key path would gate something nobody
prohibited.

Do not use subscription-auth plugins from the community; the pattern is the named
breach example above.

## Primary sources

| Source | What it governs |
| --- | --- |
| [Antigravity terms](https://antigravity.google/terms) | The subscription service |
| [Google Terms of Service](https://policies.google.com/terms) | The account itself |
| [Antigravity docs](https://antigravity.google/docs/home) | The product surface |

**These documents change.** This exclusion was decided on the 2026-08-18 text; if
you are reading this much later, the terms may have moved in either direction —
check them, not this page.
