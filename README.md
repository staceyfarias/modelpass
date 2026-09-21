# modelpass

**The pass to the models you are entitled to — the subscription you already pay for, a
key you already have, or the box on your own desk. One interface over all of them.**

Install it with `pip install modelpass`, import it as `modelpass`.

> Developed under the working titles *agentbridge* and then *subpass*. The distribution
> and the import package are both `modelpass` as of 0.2.0; `import subpass` still works
> through a shim until 0.3.0. See [Migrating from subpass](#migrating-from-subpass).

## What this is

modelpass is a Python library for reaching language models from a program without the
program having to care *how the model is paid for*. Six runtimes, one calling
convention: the two vendor agent runtimes that run under a **subscription** you already
have — the Claude Agent SDK and the Codex CLI — and four **API-key** runtimes that bill
a key — Anthropic's Messages API, OpenAI's Responses API, the Gemini API, and
`openai-compatible` for anything that speaks the OpenAI shape, a local Ollama or LM
Studio or a gateway included.

Each of those is a **connection**: vendor, identity, runtime, billing mode and a
*pointer* to a credential, in one file that every modelpass tool on the machine reads
and that is safe to share, because no key is in it. Keys live in a second file that is
not shareable, or in an environment variable the connection names. Configuring a
connection is a deliberate act with a receipt in front of it, and nothing ambient can
become one on its own: an `ANTHROPIC_API_KEY` sitting in your environment does nothing
at all until a connection names it, and a fresh install has no AI connectivity
whatsoever.

Over the six runtimes there is **one interface**, sync and async: `bridge.ask(...)` for
a single answer, `bridge.chat(...)` for the event stream, `achat` / `aask` / `asend` for
the same things on your own loop — plus your own Python functions as tools, schema-bound
output, sessions where the runtime holds one, per-call timeouts and spend guards. And in
front of every run there is a **receipt**: which auth mode is actually about to pay,
which sampling fields will really be sent and which are being dropped and why, whether
the prompt-cache breakpoints you marked will reach the vendor, and what bounds the run.
Afterwards the terminal event carries the same answer — stamped by modelpass, not by the
adapter — and one line of it goes to `~/.modelpass/runs.jsonl`, so the question survives
the run that answered it.

### The subscription half, which is where this started

If you pay for Claude Max or a ChatGPT plan, you are already paying for a lot of model
capacity. But the moment you want to *build* something — a script, a small tool, an app
with a chat feature — the normal answer is "get an API key", and now you have a second,
metered bill for work your subscription would happily have done.

That is the half modelpass started as, and it is still the half nothing else does. It
drives the agent runtimes the vendors already ship — the Claude Agent SDK, the Codex
CLI — under **your own login**, so a tool can plug into whichever subscription its user
actually has, and the same code reaches a key when they have one instead.

It is not a trick, and it is not scraping anything. The vendors ship first-party
facilities for exactly this. What nobody ships is the *pluggability*: one calling
convention that works whether the person running your tool has a Claude plan or a
ChatGPT plan.

### Who it is for

**Your own local tools.** Scripts, dev utilities, one-off automations, the little
programs you write for yourself. These are the things where a metered bill is annoying
out of all proportion to the value, and where you already have a subscription sitting
right there.

**Tools you give to other people.** This is the audience that shaped the design. If you
distribute a local tool — a plugin, a CLI, a desktop app — the typical user has a
subscription and *no appetite whatsoever* for setting up API billing. Asking them for a
key is where your install numbers go to die. Letting them point your tool at the plan
they already pay for is a different proposition entirely.

That second audience is also the one that cannot absorb a surprise. Somebody running
your tool must never wake up to a metered bill they did not agree to. Most of what
follows exists because of that sentence.

**Not for:** hosted services, multi-user backends, anything where one subscription
serves many people. See [Boundary and compliance](#boundary-and-compliance).

### What it is not

A library with a shared on-disk store, and nothing else. It is **not a service, a
router, a gateway or a daemon**: there is no server component, no routing or
load-balancing between connections, and no HTTP face. It does not retry for you — it
gives your loop a typed verdict and lets you decide (see
[Timeouts, retries, and threads](#timeouts-retries-and-threads)). It is not a billing
system: guards count tokens from the events a runtime emits, and the vendor's bill is
the truth. And it does not do embeddings — no subscription runtime can embed, so the
one thing this library is for adds nothing there; embeddings stayed app-owned.

## Install

```
pip install modelpass
```

Until the first release reaches PyPI, install from source instead:
`pip install "modelpass @ git+https://github.com/staceyfarias/modelpass.git"`.

Core has **zero runtime dependencies**, on purpose. Every runtime, the bench and the
LangChain leaf arrive as extras, and modelpass never redistributes a vendor binary —
those come through the vendor's own distribution channel under the vendor's own license.

| Extra | Install | What it brings | Runtime it turns on |
| --- | --- | --- | --- |
| *(none)* | `pip install modelpass` | the store, the CLI, the event and receipt core | — |
| `anthropic` | `pip install "modelpass[anthropic]"` | `claude-agent-sdk` | `anthropic-sdk` (subscription) |
| `openai` | `pip install "modelpass[openai]"` | `openai-codex` | `openai-sdk` (subscription) |
| `anthropic-api` | `pip install "modelpass[anthropic-api]"` | `anthropic>=0.40,<1` | `anthropic-api` (metered key) |
| `openai-api` | `pip install "modelpass[openai-api]"` | `openai>=2,<3` | `openai-api` **and** `openai-compatible` |
| `google-api` | `pip install "modelpass[google-api]"` | `google-genai>=1,<2` | `google-api` (metered key) |
| `langchain` | `pip install "modelpass[langchain]"` | `langchain-core>=1,<2` | the `ChatSubpass` leaf |
| `bench` | `pip install "modelpass[bench]"` | `flask>=3` | `modelpass bench` |

Two of those are the same client pointed at two places: `openai-api` installs the
`openai` package, which serves both OpenAI's own endpoint and any OpenAI-shaped one. The
`openai` extra is a different package entirely — `openai-codex`, the agent runtime. Two
OpenAI runtimes, two packages, one name each.

The API-key runtimes need nothing beyond their extra and a key. The two **subscription**
runtimes need three things to be true on your machine — the **vendor's CLI is
installed**, you are **logged into it**, and its **version is current enough** —
and the rest of this section is about them. modelpass supplies none of the three; it
drives what you already have.

### The vendor CLI, on your `PATH`

The Python extras pull in each vendor's SDK, but the SDK drives a **native CLI binary**
that you install separately. modelpass resolves it from `PATH`.

| Runtime | Install | Login | Credential lands in |
| --- | --- | --- | --- |
| Anthropic | `npm install -g @anthropic-ai/claude-code` | `claude /login` | `~/.claude/.credentials.json` (macOS: Keychain) |
| Codex | `npm install -g @openai/codex` | `codex login` | `~/.codex/auth.json` |

> **On Anthropic, the CLI you install may not be the one modelpass runs.**
> `claude-agent-sdk` ships a bundled `claude` binary and resolves it **ahead of
> anything on `PATH`**, so `npm install -g @anthropic-ai/claude-code@latest` updates
> the CLI you type and leaves modelpass's untouched. To move modelpass's, upgrade the
> Python package: `pip install -U claude-agent-sdk`. You still want the npm install
> for `claude /login` and for your own use — the credential store is shared. The
> preflight receipt names the binary it actually resolved and its version, which is
> the only reliable way to know which one a version verdict is about.

The login is the vendor's own, in the vendor's own store. modelpass never reads, copies,
replays or transmits it — it launches the runtime and the runtime authenticates itself.
Log in the same way you would to use the CLI directly, because that is exactly what is
happening.

#### Account profiles and multiple subscriptions

An account with no `configDir` uses the vendor's normal default storage. An isolated
account names its own absolute `configDir`; modelpass maps that to the vendor's supported
root variable for every child process and scrubs any ambient root that the selected
account did not name.

For Claude, that variable is `CLAUDE_CONFIG_DIR`:

Claude Code [officially supports side-by-side accounts through
`CLAUDE_CONFIG_DIR`](https://code.claude.com/docs/en/team#credential-management).
Each directory contains that account's credentials, settings, plugins and session
history. modelpass makes the directory a property of the **connection**, so selecting a
connection also selects the subscription that will pay:

```powershell
# Log the second profile in once (PowerShell / Windows).
$env:CLAUDE_CONFIG_DIR = "$env:USERPROFILE\.claude-work"
claude /login
Remove-Item Env:CLAUDE_CONFIG_DIR

# Point two named modelpass connections at the two isolated Claude homes.
modelpass connect anthropic --name claude-home `
  --nickname "Anthropic Home" --yes
modelpass connect anthropic --name claude-work `
  --nickname "Anthropic Work" `
  --config-dir "$env:USERPROFILE\.claude-work" --yes

modelpass check claude-home
modelpass check claude-work
```

```bash
# The same thing on macOS or Linux.
CLAUDE_CONFIG_DIR="$HOME/.claude-work" claude /login

modelpass connect anthropic --name claude-home --nickname "Anthropic Home" --yes
modelpass connect anthropic --name claude-work --nickname "Anthropic Work" \
  --config-dir "$HOME/.claude-work" --yes
```

If an editor extension already launches Claude Code with `CLAUDE_CONFIG_DIR`, use that
same absolute directory for the matching modelpass connection. The preflight receipt
prints both `configDir` and the exact credential-file path it found, before any tokens
are spent.

An ambient `CLAUDE_CONFIG_DIR` is deliberately scrubbed when a connection has no
`configDir`; otherwise launching modelpass from an editor terminal could silently select
the editor's account instead of the default account.

**Migration note.** That scrub is newer than the feature it protects. If your only
Claude or Codex login lives in a custom directory that you have been exporting as an
ambient `CLAUDE_CONFIG_DIR` or `CODEX_HOME`, a connection without a `configDir` will no
longer see it — it now gets the vendor default (`~/.claude`, `~/.codex`). Set
`configDir` on that connection to the same absolute path, or run
`modelpass connect ... --config-dir PATH`, and the receipt will name the directory the
child actually receives.

**On macOS**, Claude Code prefers the login Keychain, and [the Keychain entry is keyed
to `CLAUDE_CONFIG_DIR`](https://code.claude.com/docs/en/team#credential-management), so
a session with a different directory reads a different entry — directory isolation
works there as it does elsewhere. When the Keychain refuses a write (locked, an SSH
session, a password out of sync) Claude Code falls back to a plaintext
`.credentials.json` under the same directory, and modelpass reads that file exactly as it
does on Linux. modelpass never inspects Keychain contents; when there is no file, the
receipt says so and the token-free `claude auth status` probe — run inside the selected
directory — is what answers *which account is logged in*. A profile whose
`claude auth status` reports no login fails closed before any tokens are spent.

Codex uses the same abstraction with `CODEX_HOME`. OpenAI documents that all Codex
state lives beneath this root, including `config.toml` and `auth.json`. The one-time
setup is a login with that variable set — nothing else:

```powershell
$work = "$env:USERPROFILE\.codex-work"
New-Item -ItemType Directory -Force $work | Out-Null
$env:CODEX_HOME = $work
codex login
Remove-Item Env:CODEX_HOME

modelpass connect openai --name openai-home `
  --nickname "OpenAI Home" --yes
modelpass connect openai --name openai-work `
  --nickname "OpenAI Work" --config-dir $work --yes
```

```bash
# macOS / Linux.
CODEX_HOME="$HOME/.codex-work" codex login

modelpass connect openai --name openai-home --nickname "OpenAI Home" --yes
modelpass connect openai --name openai-work --nickname "OpenAI Work" \
  --config-dir "$HOME/.codex-work" --yes
```

Either Codex credential store isolates correctly. `cli_auth_credentials_store` defaults
to `file`, and `auto` falls back to `auth.json` when a keyring is unavailable — but the
keyring entry is keyed by `CODEX_HOME` too (service `Codex Auth`, account key the
SHA-256 of the canonicalized path, `codex-rs/login/src/auth/storage.rs`), so a different
directory reads a different entry either way. modelpass reports which store is in play as
a receipt note rather than requiring one. The default account continues to use Codex's
normal default storage.

> **No part of modelpass has been run on a Mac.** The macOS behaviour described here and
> above — the account profiles, the Keychain reading, the fallback to `.credentials.json` —
> is implemented from Claude Code's documentation and Codex's published source, and is
> covered by unit tests that simulate the platform. Nobody has yet driven `modelpass
> connect`, a preflight or a chat on real hardware. If something reads wrong on a real
> Mac, that is a bug worth reporting rather than a documented limitation.

Prefer a UI? Install `modelpass[bench]`, run `modelpass bench`, and open the loopback URL it
prints. The Flask account manager can add, edit, disable, delete, inspect, and preflight
profiles without accepting or displaying a secret. Its playground lets a client select
the stable account ID while showing the nickname. The account list itself is **offline** —
it reads your connection file and computes each launch plan, and asks no vendor anything.
The token-free identity probe belongs to the two buttons that mean it: **Check** shows
the live receipt, and **Verify** re-reads the account and pins it. Claude supplies login
method, provider, email, organization, and `subscriptionType` through `claude auth
status`; Codex supplies account type, email, and `planType` through app-server
`account/read`. modelpass copies only those allow-listed fields into its `AccountProfile`
receipt field and ignores every other vendor field.

On Windows, a repository checkout has a one-step launcher: double-click `start.bat` or
run it from a terminal. It creates `.venv` when needed, installs the local project with
the Bench extra, opens `http://127.0.0.1:8765/`, and keeps the server attached to that
terminal so `Ctrl+C` stops it. The app binds to loopback only.

**There is no Gemini row, because there is no Google *subscription* to drive.** Google's
Antigravity terms state that "using third party software, tools, or services to access
the Service ... is a breach of this Agreement," with account suspension or termination as
the consequence — and unlike Anthropic, whose terms prohibit a specific shape while
explicitly covering individual Agent SDK use, Google's sentence has no carve-out for
driving Google's own CLI. Do not wire a Google subscription through modelpass or through
any other third-party tool.

**This is narrower than it sounds, and the asymmetry inside modelpass is deliberate.**
Google ships a Python SDK, which looks like a contradiction until you notice that these
are two products under two agreements: that clause lives in the Antigravity terms, which
govern the **subscription**, while an API key falls under the Google Cloud terms
*instead* — as those same Antigravity terms say. Google's SDK targets the API-key path
and **cannot authenticate against the subscription at all**. So metered Gemini through a
key is an ordinary commercial product that the clause does not touch, and it ships here
ungated as the `google-api` runtime, while `google-cli` and `google-sdk` — the two
*subscription* identities — stay in the codebase as gated placeholders. Somebody will
eventually read that asymmetry as an oversight, which is why it is written down in three
places: here, in the capability note (`bridge.registry.note(Runtime.GOOGLE_API,
Capability.API_KEY_AUTH)`), and in [docs/legal/google.md](docs/legal/google.md).
Re-verified against the primary source on 2026-08-31: the sentence is unchanged and
there is still no carve-out.

Then tell modelpass about it once, which is the consent step:

```
$ modelpass connect anthropic
$ modelpass list
```

### Versions — the part that bites

**modelpass drives binaries it does not ship, so their versions are yours to manage.** Two
failure modes follow, and neither is modelpass's to fix:

**A CLI too old for the server's current models** fails at request validation:

```
400 ... The 'gpt-5.6-luna' model requires a newer version of Codex.
```

Zero tokens are spent, but the run does not happen. Vendors retire model families on
their own schedule, and an old CLI can end up able to reach nothing at all.

**A CLI too old for a feature modelpass wants to use** fails *silently*. Claude Code's
prompt-cache TTL variables (`CLAUDE_CODE_PROMPT_CACHE_TTL`) need **v2.1.242 or later**;
on an older build the variable is passed, ignored, and nothing says so.

Check what you actually have:

```
$ claude --version
$ codex --version
$ modelpass check <connection>      # re-runs the preflight and prints the receipt
```

The receipt is where a stale CLI shows up before a run rather than as a confusing 400.
Both runtimes name the binary they resolved and its version, plus — on Codex — any
different build your config points at. On Anthropic that resolved binary is usually the
SDK's bundled one rather than whatever `claude --version` reports, which is why the
receipt is the answer and the shell is not.

> **The two-installs trap.** A vendor's desktop app may install its own auto-updating
> copy of the CLI somewhere that is **not on `PATH`** — Codex does this, under
> `AppData\Local\OpenAI\Codex\bin\<hash>\` on Windows. Your app then works fine on the
> newest models while `codex` in a shell is months behind, and modelpass gets the old one.
> Keep the `PATH` install current (`npm install -g @openai/codex@latest`) rather than
> assuming the app's version is the one in play. modelpass **reports** the mismatch and
> deliberately does not resolve it for you: silently switching which executable a chat
> call launches, because modelpass went looking for a config value, is the ambient
> influence this project refuses. Point it somewhere specific with
> `options={"codex_bin": ...}` if you mean to.

### Keeping modelpass's runs out of your own coding sessions

Claude Code stores each session as a transcript under
`~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`, where the directory name is your
absolute working directory with every non-alphanumeric character replaced by `-`. That
is also how `claude --continue` finds "the most recent session" and how `/resume`
builds its list — **both are scoped to the directory you are in**.

Which sets up a collision worth understanding. modelpass runs the Claude Code CLI as a
subprocess and does **not** set a working directory, so it inherits your application's.
An app launched from a repository you also work in would, if it wrote transcripts, drop
agent conversations into exactly the directory your own `claude --continue` reads from.

**Today it writes nothing.** Every Anthropic run sets `CLAUDE_CODE_SKIP_PROMPT_HISTORY=1`
unconditionally, so no transcript is created and none of your sessions gain a neighbour.
Verified live 2026-08-30: a run with the variable produced no file anywhere under
`~/.claude/projects/`, while an otherwise identical control run without it did.

To check for yourself, list that directory before and after a run and look for one named
after your app's working directory:

```
$ ls ~/.claude/projects/
```

Codex is isolated on two axes rather than one: `--ephemeral` (also unconditional) means
no rollout file, **and** the adapter defaults its working directory to a fresh temporary
one. The Anthropic side relies on the no-writes guarantee alone, which is why the
directory it inherits has not mattered so far.

> **A session is the other case**, and it is already shipped. Persistence is the default
> there — a conversation you can resume is the point of it — so the working directory
> stops being incidental, and `new_chat()` defaults to a fresh modelpass-owned scratch
> directory rather than inheriting yours. Pointing one at a real project folder is
> something you ask for explicitly with `project_folder=`, and `new_worker()` requires
> it. See [Sessions](#sessions-where-the-runtime-holds-the-conversation).

### What reaches the prompt that you did not write

Worth knowing when a run behaves oddly. Codex reads **`AGENTS.md`** from its home
directory *and* from every ancestor of the run's working directory, and injects it as
user instructions. It cannot be scrubbed the way MCP servers can, so the preflight
receipt names any that apply. Claude Code's equivalents (`CLAUDE.md`, `settings.json`)
are blocked outright — modelpass runs it with `setting_sources=[]`.

## Quickstart

Connect once (this is the consent step, and it prints a receipt before writing
anything):

```
$ modelpass connect anthropic
$ modelpass list
```

Then, from anywhere. **One call, one answer:**

```python
import modelpass

bridge = modelpass.Bridge()

answer = bridge.ask("claude-sub", "Explain what a git rebase does.")
print(answer.text)
print(f"[{answer.status} · {answer.receipt.effective_auth_mode} · {answer.usage.total_tokens} tokens]")
```

`ask` drains the run for you and returns a frozen `Answer` — the text, the parsed
structured answer when you passed a `schema=`, the usage, the receipt, the tool calls
that happened along the way, and every event, in case you want them later. A run that
ends badly **raises** rather than handing back half an answer: `GuardStop`,
`QuotaExhausted`, `VendorRunFailed`.

The same call, streamed, when you want to render the answer as it arrives:

```python
for event in bridge.chat(
    connection="claude-sub",
    message="Explain what a git rebase does.",
):
    if isinstance(event, modelpass.TextDeltaEvent):
        print(event.text, end="", flush=True)
    elif isinstance(event, modelpass.TerminalEvent):
        print(f"\n[{event.status} · {event.auth_mode} · {event.usage.total_tokens} tokens]")
```

Or with an API key instead of a subscription — same bridge, same events, same
receipt, and the billing mode is stamped on the terminal either way:

```
$ modelpass connect anthropic --api-key-stdin     # pip install "modelpass[anthropic-api]"
```

```python
for event in bridge.chat(connection="claude-api", message="Explain a git rebase."):
    ...  # identical loop; event.auth_mode on the terminal now reads "api_key"
```

**And the same call on your own event loop.** `achat`, `aask` and `asend` take every
argument their sync twins take and answer with the same events, receipts, guards and
ledger lines — one pre-run pipeline and one event fold underneath, two doors on top:

```python
answer = await bridge.aask("claude-api", "Explain what a git rebase does.")

async for event in bridge.achat(connection="claude-api", message="..."):
    ...
```

`chat`, `ask` and `send` are unchanged by any of that and are still the whole
implementation for a sync caller: an async core with a sync wrapper would have broken
every consumer that calls from inside a running loop. Cancelling an async run is
`aclose()` — see [Async](#async).

### With your own tools

Your functions run **in your process** — keeping their closures, database handles and UI
callbacks — while the runtime runs the model/tool loop inside the one call:

```python
from modelpass import Bridge, ToolDef, ToolCallEvent, ToolResultEvent

def search_notes(args):
    return my_index.search(args["query"])          # your code, your process

tools = [ToolDef(
    name="search_notes",
    description="Search the user's local notes.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    handler=search_notes,
)]

for event in Bridge().chat(
    connection="claude-sub",
    message="What did I write about rebasing?",
    tools=tools,
):
    if isinstance(event, ToolCallEvent):
        print(f"→ {event.name}({event.arguments})")
    elif isinstance(event, ToolResultEvent):
        print(f"← {event.content[:80]}")
```

You *observe* `tool_call` / `tool_result`; you never have to answer one. The runtime
executes the loop. (Tools are capability-gated before the preflight, so asking a runtime
for something it cannot do costs nothing to find out.)

### With a schema

Pass a JSON Schema and the run's final answer is bound to it, using each runtime's own
native mechanism — no prompt-and-hope, no parsing prose:

```python
from modelpass import Bridge, StructuredOutputEvent

SCHEMA = {
    "type": "object",
    "title": "Extraction",
    "properties": {
        "status": {"type": "string", "enum": ["found", "partial", "not_found"]},
        "excerpts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "excerpts"],
}

for event in Bridge().chat(
    connection="claude-sub",
    message="...",
    schema=SCHEMA,
):
    if isinstance(event, StructuredOutputEvent):
        result = event.data          # the parsed object
        print(result["status"], event.valid)
```

One `structured_output` event arrives immediately before the terminal, carrying the
parsed object (`data`), the JSON it was parsed from (`raw`), and `valid` — modelpass's own
structural check, with `problems` behind a false one. Text deltas still stream as they
occur.

Three things worth knowing, all of them deliberate:

* **`valid` is modelpass's check, not the vendor's.** The runtime is what actually
  constrains the model; `valid` exists so you can branch without re-validating. A
  structurally invalid answer is still delivered as-is — modelpass never edits `data` to
  make it fit, and it never substitutes an empty dict for a missing answer. A run that
  produced nothing usable ends `status="error"` with a reason naming what came back.
* **`schema_name=`** sets the name the schema travels under (default: its `title`). It
  matters because on `anthropic-sdk` the mechanism is a tool call, so if your prompts
  refer to that tool by name, the name is yours to control.
* **`schema=` and `tools=` cannot be combined** in v1. The mechanics conflict on
  `openai-sdk` and are unverified on `anthropic-sdk`; refusing on both beats working
  silently on one.

### From LangChain

If your application's LLM seam is already LangChain's `BaseChatModel`,
`modelpass.langchain_adapter` drops a modelpass connection into it:

```python
from modelpass.langchain_adapter import ChatSubpass

model = ChatSubpass(connection="claude-sub")
answer = model.invoke([HumanMessage(content="Explain what a git rebase does.")])

print(answer.content)
print(answer.usage_metadata)                          # tokens, cache reads broken out
print(answer.response_metadata["subpass_receipt"])    # which connection paid

# Schema-bound output, by the same mechanism as above. The answer arrives on a
# tool_call — the shape a forced tool choice over an API key produces — rather
# than as a parsed object, because building the object is the caller's job.
decision = model.bind_tools([MySchema], tool_choice="MySchema").invoke(messages)
decision.tool_calls[0]["args"]
```

`ainvoke` and `astream` run over `achat` rather than putting the sync path in an
executor.

It is a **convenience adapter, not the product's surface.** It covers modelpass's
chat-shaped subset — stateless chat and schema-bound output — and nothing else:
runtime-executed tools with their call/result pairs, MCP, cancellation and the run log
have no `BaseChatModel` shape, and bending them into one would mean inventing
behavior. `Bridge.chat` stays the real API. modelpass is not tied to LangChain; this is an
optional leaf over the same public bridge, and core still imports nothing from
LangChain.

Two things the adapter has to say out loud: `stop` sequences are ignored, and a vendor
failure — which modelpass reports as a terminal event, not an exception — is raised as
`SubpassRunError` (`SubpassGuardStopError` / `SubpassQuotaExhaustedError` for the two a
caller can act on, `SubpassTimeoutError` when a bound expired), carrying the receipt,
the partial text and what was spent. `temperature`, `top_p`, `top_k`, `max_tokens` and
`reasoning_effort` are **passed through** rather than warned about — they become a
`Sampling`, the runtime takes what it takes, and `response_metadata` gains
`subpass_sampling_applied` and `subpass_sampling_notes` saying what happened to the rest.

## Connections

A connection is the unit of consent and the unit of configuration. An application built
on modelpass stores **a connection name and nothing else**, and its user configures
their subscription or their key once, in one place, for every modelpass tool on the
machine.

### The four ways to point one at a model

Each of these prints a receipt and asks before it writes anything:

```bash
# Subscription. The runtime's own login pays for it and
# nothing is stored anywhere but the connection itself.
modelpass connect anthropic

# Metered, on the API runtime, reading a named variable.
# The variable's name is stored; its value never is.
modelpass connect anthropic --runtime anthropic-api --api-key-env ANTHROPIC_API_KEY

# Metered, on the API runtime, with the key handed over
# once and written to the secrets file described below.
cat key.txt | modelpass connect anthropic --api-key-stdin

# An OpenAI-shaped endpoint that is not OpenAI -- a local
# Ollama, say -- names its vendor `compatible` and its URL.
# With no key flag at all, because a local box authenticates
# nobody and saying so beats pointing at a variable nobody sets.
# (--api-key-env / --api-key-stdin still work, for a gateway.)
modelpass connect compatible --base-url http://localhost:11434/v1 --model llama3.1

# ...and then the step this runtime cannot skip. `openai-compatible`
# names an API *shape*, so its capability row is `unverified` by
# construction -- including `chat`, which means bridge.chat()
# refuses until somebody has driven the endpoint. This drives it:
# one short chat, one tool round trip, one structured-output call,
# and it records which of those actually worked on the connection.
modelpass verify compatible-api
```

`verify` is the only command that spends in order to learn something, and it says so
before it does. Against a local Ollama that is electricity; against a metered gateway
it is a few hundred tokens. What it writes is a per-connection record — two installs of
this runtime can legitimately disagree about every cell, so the verdicts live on your
connection and never in the shared table. Cells three short calls cannot decide
(`thinking`, `interim_usage`, the sampling pair) stay where the table left them, and
the output says so. Details in
[docs/api-and-runtimes.md §2.0c](docs/api-and-runtimes.md).

Omitting `--runtime` keeps the defaults unchanged: a bare `connect` is the subscription,
and `--api-key-stdin` on its own still moves to the API runtime. `--api-key-env` on its
own still means metered billing on the *agent* runtime, because an environment variable
does reach a launched child — which is why the combination needed a flag to say
otherwise rather than a change to what it already did.

### `~/.modelpass/connections.toml`

Written by `modelpass connect`, safe to edit by hand, and **it contains no secrets** —
only pointers, validated on the way in and on the way out.

```toml
version = 1

# --- store-wide settings (optional; omit the table entirely for defaults) ----
[settings]
# Every completed run appends one line to runs.jsonl beside this file. On by
# default; this is the opt-out.
runLog = true

# --- a subscription connection ----------------------------------------------
[connections.claude-sub]
runtime       = "anthropic-sdk"      # anthropic-sdk | openai-sdk
authMode      = "subscription"       # subscription | api_key
credentialRef = "native-login"       # the runtime's own login store
nickname      = "Anthropic Home"     # optional human-facing label
# configDir omitted: use this vendor's default account storage
model         = "sonnet"             # optional; omit for the runtime's default
description   = "Claude Code login"  # optional note to yourself
enabled       = true                 # optional, default true (see below)
groups        = ["fast"]             # optional; omit and it is in "default"
timeoutSeconds = 120                 # optional; default wall-clock bound per call
retry         = "default"            # optional; "default" | "never" (see below)
maxInputTokens = 200000              # optional; the model's usable input window, in
                                     # tokens, as YOU state it. Omit it and the window
                                     # is unknown -- there is no table of model sizes
                                     # here and nothing guesses one. Nothing in
                                     # modelpass reads it; it is for the tools that do.
promptCache   = "default"            # optional; ask for prompt caching. "default" =
                                     # the vendor's own lifetime, or name one the
                                     # runtime accepts. Omit it and nothing is stated
                                     # -- which is NOT "caching off" (see below).

  # Spend guards. NO DEFAULTS -- see "Guards" below. A connection with no
  # [guards] table genuinely has nothing bounding one run's spend, and the file
  # says nothing rather than writing a misleading "warnAtTokens = 0".
  [connections.claude-sub.guards]
  warnAtTokens = 150000
  stopAtTokens = 400000

    [connections.claude-sub.guards.onQuotaExhausted]
    action = "stop"                  # stop | failover

# --- a metered connection, for comparison -----------------------------------
[connections.claude-api]
runtime       = "anthropic-sdk"
authMode      = "api_key"
credentialRef = "env:ANTHROPIC_API_KEY"   # the NAME. Never the key.
# Non-credential variables this connection may see through the scrub. A closed
# per-runtime set: a credential can never be added here.
allowEnv      = ["ANTHROPIC_BASE_URL"]
groups        = ["fast", "metered"]       # a connection may be in several

# --- a connection that fails over when the allowance runs out ---------------
[connections.codex-sub]
runtime       = "openai-sdk"
authMode      = "subscription"
credentialRef = "native-login"
nickname      = "OpenAI Work"
configDir     = "C:\\Users\\you\\.codex-work"

  [connections.codex-sub.guards.onQuotaExhausted]
  action   = "failover"
  failover = "claude-api"            # naming it here IS the consent
```

**`credentialRef` forms.** It says *where* a credential lives, never what it is:

| Form | Meaning |
| --- | --- |
| `native-login` | The runtime's own login store (`claude /login`, `codex login`). The subscription path. |
| `env:NAME` | The environment variable called `NAME`. The name is stored; the value never is. |
| `secret:entry` | An entry in `~/.modelpass/secrets.toml`. The entry *name* is stored; the value lives in the other file. See [API keys](#api-keys-in-a-second-file) below. |
| `keychain:entry` | A keychain entry by name. |

Anything that looks like a secret is refused outright — paste an `sk-...` into a
`credentialRef` and modelpass raises `CredentialRefIsSecret` rather than storing it.

### API keys, in a second file

An explicit key goes in a **second file**, `~/.modelpass/secrets.toml`, beside
`connections.toml` and never inside it. The connection file is the one that is safe to
share and commit; one key in it would destroy that property for everybody, including
people who never wanted an explicit key. So the pointer lives in the shareable file and
the value lives in the other one:

```toml
# ~/.modelpass/secrets.toml -- NOT shareable.
version = 1

[secrets.claude-api]
apiKey = "..."
```

**Writing one.** The key is read from standard input and never from the command line:

```bash
# The key arrives on stdin. Nothing is written until the receipt is shown.
cat key.txt | modelpass connect anthropic --api-key-stdin
```

There is deliberately no option anywhere in `modelpass` that takes a key as a value: a
command line is visible in process listings, is written to shell history, and is echoed
by CI logs. `--api-key-stdin` also refuses to read from a terminal, and tells you how to
pipe instead. Exactly one trailing newline is stripped — what `echo`, a here-string and
a password manager's `--raw` each append — and an empty value is refused.

`--api-key-stdin` selects the vendor's **API runtime** (`anthropic-api`, `openai-api`),
because that is the one that bills a key you hand over. A stored key cannot reach the
child process that `anthropic-sdk` or `openai-sdk` launches, so `secret:` is refused on
those runtimes rather than accepted and quietly ignored; use `env:NAME` there.

**Permissions, stated honestly.** On macOS and Linux the file is created `0600`, which a
test asserts rather than assumes. If the mode is later widened so that group or other
can read it, `modelpass check` says so and tells you the `chmod` — it does not refuse,
because the file is still yours and refusing would only lock you out of your own
configuration.

On Windows there are two layers and modelpass claims only what it applied:

- The documented floor is the ACL your user profile directory already carries, which
  `%USERPROFILE%\.modelpass` inherits.
- On top of that, once at creation, modelpass runs
  `icacls <path> /inheritance:r /grant:r "<you>":F` and **records whether it worked**.
  `modelpass check` reports either "restricted to your account" or "could not restrict:
  `<reason>`; the profile directory's own ACL is the floor". A protection that was not
  applied is never described as though it were.

**Keeping the two files in step.** They are not written atomically together, so the
order is part of the design: the secret is written first and the connection second when
adding, and the connection first and the secret second when deleting. A crash in between
leaves an unreferenced secret — inert, and listed by `modelpass check` — rather than a
connection pointing at nothing.

- **Deleting a connection** removes its secret entry only when the entry is named after
  that connection *and* nothing else references it. Otherwise the entry stays and you
  are told. An orphan is recoverable; a deleted key is not.
- **Deleting a secret** that a connection still references is refused, naming them.
- **`modelpass rename <old> <new>`** moves the connection and, under the same rule,
  moves its secret entry with it. When the entry is shared, or is not named after the
  connection, it stays where it is and the command says so.
- **`modelpass check`** lists orphan entries — entries no connection references. They
  are reported and never collected for you.

No secret value ever reaches a receipt, the run log, a `to_dict()` or a vendor event; a
`sha256:` fingerprint of it may, and that is what the receipt's `account` field carries
on a key-backed connection.

### The rest of the keys, and the compatibility policy

**`guards`** — thresholds in tokens, per run. **There are no defaults.** A missing
`[guards]` table means nothing bounds that connection's spend, and modelpass makes that
absence loud rather than papering over it with an invented number. See
[Guards](#guards-and-where-the-responsibility-sits).

**`onQuotaExhausted`** — `stop` (clean stop, the only default) or `failover` to a named
connection. Failover is the *only* path in modelpass that can move a run onto metered
billing, and naming the second connection is the consent. It never chains, never
self-targets, announces itself as a `failover` event carrying the second connection's
own receipt, and the terminal event names both connections.

**`allowEnv`** — a closed, per-runtime set of *non-credential* variables that may
survive the scrub (currently: `ANTHROPIC_BASE_URL`, on `anthropic-sdk`, and nothing
else). Deliberately not a free-form passthrough: the first thing anyone would put in one
of those is an API key.

**`nickname`** — an optional human-facing label. It can contain spaces and can be
renamed safely. Clients, failover targets, receipts, and run logs continue to use the
stable table key (`claude-sub`, `codex-sub`, and so on).

**`configDir`** — an absolute vendor configuration directory. On `anthropic-sdk` it is
passed as `CLAUDE_CONFIG_DIR`; on `openai-sdk` it is passed as `CODEX_HOME`. Omitting it
uses the vendor default. An unconfigured ambient root is scrubbed. Both vendors key
their OS credential store to this directory — Claude Code's Keychain entry and Codex's
keyring entry are per-directory — so isolation holds on every platform, and no
credential-store setting is required of you. Keychain contents are never inspected by
modelpass; on macOS `claude auth status` supplies the identity instead.

**`enabled`** — default `true`. A disabled connection is still listed and still
preflights; what it refuses is runs, with a clear `ConnectionDisabled` error raised
before anything is attempted. It exists so that taking a connection out of service is
not "delete it and retype it later", which is how you lose the guard thresholds you
worked out.

**`groups`** — the groups this connection is in. Omit it and the connection is in
`default`, which is simply **the name for the connections that name no group**: one
with `groups = ["fast"]` is in `fast` and is *not* also in `default`. A group needs no
creating and cannot be deleted — it exists while some connection claims it, so moving
the last member out is how it goes away. See
[Groups](#groups-addressing-a-set-of-connections).

**Compatibility policy** (2026-09-13). Several tools on one machine share this file, so
they will not all be the same build of modelpass. What every build promises:

- The `version` above bumps **only** when an existing key changes meaning or disappears.
  New keys and new runtime values are **additive** and do not bump it. The file stays at
  version 1 across the API-runtime extension.
- A build **reads any version**. It refuses to **write** a file whose version is newer
  than its own, naming both versions and saying a newer build is needed. Reading a newer
  file never leaves `modelpass accounts` or `modelpass check` unable to say anything.
- A **key this build does not recognise** is preserved exactly as written and reported —
  never a failure, and never dropped when the file is rewritten. That applies inside
  `guards`, `onQuotaExhausted` and `verifiedIdentity` too.
- An **unknown `runtime` or `authMode`** refuses *that connection*, with the values this
  build does know named, and leaves every other connection usable. The refused connection
  is still written back untouched; `modelpass accounts` says which ones and why.

Anything else invalid — a warn threshold above its stop, `enabled = "no"` — still refuses
the file. Those are mistakes in a file a human edited, and quietly dropping the
connection would hide one.

**`MODELPASS_HOME`** moves the config location (`$MODELPASS_HOME/connections.toml`
instead of `~/.modelpass/connections.toml`). It moves *where the file is read from* and
nothing else — it can never turn an ambient credential into a connection. `SUBPASS_HOME`
is still honoured when `MODELPASS_HOME` is unset, with a deprecation warning, until 0.3.0.

### Per-app configuration

An application built on modelpass should store **a connection name and nothing else**:

```toml
# your-app's own config
[ai]
connection = "claude-sub"
```

Not a runtime, not an auth mode, and certainly not a credential. The connection is the
unit of consent, it lives in one place, and every modelpass tool on the machine reuses it.
Users configure their subscription once.

A group works in the same slot, and is the better answer when your app wants "whichever
of these the user has" rather than one named connection:

```toml
[ai]
connection = "group:cheap"
```

## Groups: addressing a set of connections

A connection may declare groups, and a group may be named anywhere a connection name is
expected:

```toml
[connections.local-ollama]
runtime       = "openai-compatible"
authMode      = "api_key"
baseUrl       = "http://localhost:11434/v1"
credentialRef = "none"
groups        = ["cheap", "offline"]
```

```python
answer = bridge.ask("group:cheap", "Classify this: ...")
```

Three things are worth knowing before you use one.

**`default` means "the ungrouped ones".** Not "all of them". A connection you put in
`fast` leaves `default`; say `groups = ["fast", "default"]` if you want both. The other
reading would make `default` carry no information and `group:default` mean "any
connection at all".

**A group resolves to the first enabled member in name order, and modelpass does not
pretend you chose that rule.** `bridge.select("cheap")` hands back the connection that
would run *and* every member it walked past with the reason, and `modelpass groups`
prints the same thing:

```
  cheap
      a-gateway   [disabled]
      local-ollama
      -> group:cheap resolves to local-ollama
         a-gateway passed over: disabled
```

If you need a particular connection, name that connection. A group is a convenience for
"any of these", not a router with a policy — **preference order within a group does not
exist yet**, and it is queued with throttling and metrics rather than guessed at.

**A group is not a thing you create.** There is no `[groups]` table; a group exists
while some connection claims it. Nothing is left behind pointing at nothing, and there
is no second place a connection can be taken out of service — `enabled = false` remains
the only one. A group whose every member is disabled raises `GroupUnavailable`, which is
a different error from `NoSuchGroup` because it sends you somewhere different.

What a run is *billed* to is always a connection. The group is how the call was
addressed; the receipt, the terminal event and the run-log line name the connection that
was actually reached.

**One field refuses a group: `onQuotaExhausted.failover`.** Failover is the only path
onto metered billing, and naming the target is the consent — a group's membership can
change after that consent was given, so a target added to the group next month would be
paid for under a decision nobody made about it. Name the connection you mean. The
refusal happens when the connection is constructed, not when an allowance runs out.

On the command line: `modelpass groups` lists them, `modelpass list --group fast` filters
the listing, and `modelpass set-groups <name> fast cheap` moves a connection (naming no
group returns it to `default`). In the bench, the **Groups** page shows the same
resolution, and each account's form has a groups box.

## What the receipt tells you

The receipt answers *what is about to happen, and who pays for it*. It exists before any
run and again at setup time, and nothing is written or spent before you have seen one.
`receipt.summary()` prints it; the fields are on the object.

| Field | What it answers |
| --- | --- |
| `effective_auth_mode`, `requested_auth_mode`, `detected_auth_mode` | which billing mode is really about to be used |
| `credential_source`, `account`, `plan_name`, `account_profile`, `identity_verified` | *whose* it is — a login, a named variable, a stored entry, or a `sha256:` fingerprint of a key, never a key |
| `runtime`, `runtime_available`, `binary`, `model`, `model_source` | what will run, and where the model came from |
| `scrubbed`, `preserved`, `passthrough`, `config_dir`, `directives` | what the child process will and will not see, or on an API runtime that the client is constructed with an explicit key and the SDK never reads the environment |
| `guards_configured`, `retry` | what bounds the run, including a `retry = "never"` stance |
| `sampling_requested`, `sampling_applied`, `sampling_notes` | what you asked for, what is actually being sent, and one sentence per field dropped or coerced |
| `cache_breakpoints_requested`, `cache_breakpoints_honoured` | whether the cache markers you set reach the vendor |
| `cache`, `ok`, `problem`, `notes` | the caching disclosure, whether this would run at all, and anything else worth saying |

### What one looks like

```
$ modelpass connect anthropic
claude-sub
  runtime      anthropic-sdk
  auth mode    subscription
  credential   Claude Code login (~/.claude/.credentials.json)
  plan         Max 5x
  scrubbed     ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, CLAUDE_CODE_USE_BEDROCK
  guards       no spend guards configured: nothing bounds what one run may spend...
  status       OK

Save this connection as 'claude-sub'? [y/N]
```

Nothing is written before you have seen that. The `scrubbed` line is the interesting
one — see the next section.

### A scrubbed launch

This part matters more than it sounds like it should, because **the vendor runtimes
scavenge ambient credentials, and two of the three fail toward metered billing**:

* Claude Code resolves subscription OAuth *last* — behind `ANTHROPIC_API_KEY`,
  `ANTHROPIC_AUTH_TOKEN`, `apiKeyHelper` and the cloud-provider switches. A stray key in
  your environment silently wins, and you find out on the bill.
* The Codex SDK injects `CODEX_API_KEY` into the CLI's environment, and OpenAI's own CI
  documentation encourages setting it.

So declining to *set* a key is not enough. modelpass computes the child process's
environment explicitly, removes every credential the connection did not name, and
reports what it removed. `ANTHROPIC_BASE_URL` is on that list too — not a credential,
but it silently changes where a subscription token gets sent, and nothing ambient may
shape a run.

### A normalized event stream

One call, one stream of events: text deltas, thinking, usage in tokens, guard warnings
and stops, tool calls and results, a schema-bound answer if you asked for one, and a
terminal event. Anything a runtime reports that the vocabulary does not cover arrives as
a `vendor_event` rather than being dropped.

### A stamped terminal, and a durable record

Every stream that completes normally ends with exactly one `terminal` event, stamped by
modelpass — not by the adapter — with the connection name and the **auth mode actually
used**. Adapters can report a status; they cannot claim how a run was billed.

That stamp is also written to `~/.modelpass/runs.jsonl`, one JSON line per connection the
call ran on, so the question survives the run that answered it. (A call that exhausts
its allowance and fails over spent two allowances, and writes two lines — one per leg,
each with the auth mode that paid for it.) See
[How do I know it was the subscription?](docs/subscription-proof-2026-08-17.md) — which
includes a live falsification experiment, its control, and the check that does not
require trusting modelpass at all.

## Guards, and where the responsibility sits

Guards are `warnAtTokens` / `stopAtTokens` per connection, in tokens, per run. Full
treatment: **[docs/guards.md](docs/guards.md)**.

**modelpass ships no default thresholds, and that is deliberate.** An earlier draft had
200k warn / 1M stop. Neither number was derived from a plan's allowance, measured
against a real run, or sourced from a vendor — and a made-up number presented as a
product default reads as advice this project is in no position to give. It does not know
your plan, your month, or what you are about to ask for.

What replaces the default is making the *absence* visible: the receipt says
`no spend guards configured`, `modelpass connect` prints it at the moment you form your
picture of what you just set up, `modelpass list` shows `guards: none`, and the bench tags
it. Silence about spend was the failure mode worth designing against.

Three things guards do not claim:

* **The vendor's bill is the truth.** Local accounting is best-effort token counting from
  the events a runtime chooses to emit. Nothing here is a billing system.
* **`stopAtTokens` is not a hard ceiling.** It stops the run as soon as modelpass is in a
  position to know. On the four API runtimes the tool loop is in this process and usage
  arrives with every response, so a guard is checked **between rounds**; on
  `openai-sdk` that holds on the default transport and not on `options={"transport":
  "exec"}`, where the only usage arrives at the end. The capability table's
  `interim_usage` row is the per-runtime answer.
* **There are no dollar thresholds**, decided rather than deferred — and re-examined on
  2026-09-13, when four of the six configurable runtimes became metered, without
  changing. A price is a fact about your contract rather than about the response, and
  `openai-compatible` has no price modelpass could know at all. The unit stays tokens,
  which every runtime reports.

**For metered (`api_key`) connections, set a vendor-side hard cap** — enforced where the
billing actually happens, not here. A guard is an observation modelpass acts on; a cap is
enforcement, and it is still true when this library is not in the call path.
[docs/guards.md](docs/guards.md) links the place to set one per vendor.

Do that in addition to token guards, not instead of them.

## Capabilities per runtime

These runtimes are not interchangeable, and modelpass reports the differences rather than
flattening them. `unverified` means *nobody checked* — it is not a soft yes, and modelpass
treats it as unusable.

Eight runtime identities, because **capability identity is the runtime, not the vendor**
(D6):

| Runtime | Family | State |
| --- | --- | --- |
| `anthropic-sdk` | agent | implemented, validated live |
| `openai-sdk` | agent | implemented, validated live |
| `anthropic-api` | API key | adapter landed; `ttl_control` and `midconversation_system` await a live drive |
| `openai-api` | API key | adapter landed; `thinking` awaits a live drive |
| `google-api` | API key | adapter landed; `thinking` awaits a live drive |
| `openai-compatible` | API key, or none | adapter landed; **every cell is per install**, filled by `modelpass verify` |
| `google-cli` | agent | gated placeholder — see the Google note under [Install](#install) |
| `google-sdk` | agent | gated placeholder — same reason |

"Awaits a live drive" is literal. Those cells moved as far as a fake transport and the
installed SDK's own typed surface can move them; the half that needs a real call to a
real endpoint is a `live`-marked test the owner runs, not something a changelog may
assert. The four live files and the variables they need are listed once in
[docs/api-and-runtimes.md](docs/api-and-runtimes.md).

The table below is the short form for the two agent runtimes — nine rows of
twenty-five.

| Capability | `anthropic-sdk` | `openai-sdk` |
| --- | --- | --- |
| `chat`, `streaming`, `usage_tokens` | supported | supported |
| `thinking` | supported | **supported** — but what arrives is the reasoning **summary** stream (`item/reasoning/summaryPartAdded` / `summaryTextDelta`), not raw reasoning text: `item/reasoning/textDelta` exists in the protocol and did not fire even at `effort: "high"` with `summary: "auto"` (driven 2026-08-31). Do not promise your users "the model's reasoning". `unverified` on `options={"transport": "exec"}`, where nobody has checked either way |
| `tools_in_process` — your Python functions as tools | **supported** | **supported since 2026-08-31**, when the default transport became `codex app-server`: each tool is registered as a `dynamicTools` entry and your function answers the server's `item/tool/call` in-process, driven live and then end-to-end through `bridge.chat(tools=[...])`. The vendor marks that surface experimental. **Unsupported on `options={"transport": "exec"}`**, whose only tool channel is MCP over a command line — a checked absence, not an open question. |
| `mcp_servers` — per-run MCP declaration | supported, and *exclusive*: only the servers you name are reachable | **unsupported on the default transport as of 2026-08-31**, and this is the one thing the app-server flip cost: per-run MCP servers have no driven equivalent there. **Works on `options={"transport": "exec"}`** — `-c mcp_servers.<name>={...}`, with exclusivity via enumerate-and-disable (fails closed if enumeration fails). Nothing about the vendor changed; the default did. |
| `interim_usage` — usage while the run is still going | **supported** — per assistant turn, so a guard can stop a tool loop **between rounds** | **supported since 2026-08-31** — `thread/tokenUsage/updated` arrives after **each** model response, mid-turn. So `stopAtTokens` on Codex is a real circuit breaker now, not a post-hoc report. **Unsupported on `options={"transport": "exec"}`**, where `codex exec --json` reports once, at the end. |
| `graceful_cancel` | supported (`interrupt()`) | unverified; the floor is terminating the process. modelpass sends `turn/interrupt` on the app-server transport — from `cancel()` and from an abandoned session turn — and a live drive on 2026-08-31 ended a turn with `status: "interrupted"`. The cell still does not move, because the half a caller would rely on is unchecked: modelpass does not wait for the terminal that interrupt produces, and nobody has shown a thread taking another turn afterwards |
| `structured_output` — a schema-bound answer | **supported**, native (`output_format` → `--json-schema`), and it accepts a **loose** schema: optional properties, no `additionalProperties` | **supported**, native — a per-turn `outputSchema` field on the default transport (no temp file at all) and `codex exec --output-schema` on the opt-out. Expect the Responses API's **strict** subset either way — every property in `required`, `additionalProperties: false` everywhere. That constraint is *driven* on exec and **not re-driven** on app-server, so modelpass predicts it on both rather than claiming it |
| `sampling_controls`, `max_output_tokens` | **unsupported** — a checked absence: no `temperature`, `top_p`, `top_k` or output-length parameter exists on either CLI protocol. Passing `sampling=` is still not an error; see [Sampling controls](#sampling-controls) | **unsupported**, same evidence |
| `subagents` | supported | unsupported |
| `sessions_resume` | supported | supported |
| `sessions_list` — enumerate stored conversations | supported, scoped by working directory | **supported since 2026-08-31** — `thread/list` on the default transport: token-free, paginated, account-wide. **Unsupported on `options={"transport": "exec"}`**, which has no scriptable listing at all, only an interactive picker — and is refused there rather than answered empty, because an empty list would be a false statement about an account that may have hundreds. |
| `system_prompt_replace` — a `ChatSession` with your persona instead of the runtime's | supported | **supported since 2026-08-31** — `baseInstructions` genuinely replaces Codex's coding-agent persona on the default transport. **See [the behaviour change](#a-behaviour-change-on-codex-2026-08-31) below: this changes runs you were already making.** **Unsupported on `options={"transport": "exec"}`**, where a system prompt is layered *on top of* the persona. A `WorkerSession` keeps the layering on both, because append is the semantics a worker asked for. |
| `get_history()` on a session — read the conversation back | supported (`persist=True`) | supported on the default transport, paged from `thread/items/list`, with tool calls and compactions **marked in place** rather than dropped. `()` on `options={"transport": "exec"}`, honestly: `codex exec` cannot read a thread back and modelpass will not reassemble one from the events it happened to see. |

This table describes the **default** transport on `openai-sdk`, which since 2026-08-31
is `codex app-server`; `options={"transport": "exec"}` narrows six of those cells and
widens `mcp_servers`, in the way the `tools_in_process` row spells out.
[docs/api-and-runtimes.md](docs/api-and-runtimes.md) carries the full matrix and says
which cells move with the transport.

Query it in code: `bridge.registry.support(runtime, Capability.TOOLS_IN_PROCESS)`, or
`bridge.find(capability="tools_in_process")` to pick a connection that can do the thing.
On an `openai-compatible` connection ask `bridge.registry_for(connection)` instead: that
runtime's cells live on the connection a drive wrote them to and never in the shared
table, because two installs of it can legitimately disagree about every one.

### A behaviour change on Codex (2026-08-31)

**`openai-sdk` now runs `codex app-server` by default.** It used to run `codex
exec`. Same runtime identity, same connection, same `openai-sdk` name — and two
things a caller can feel, so they are stated here rather than left in a
capability note:

1. **`system_prompt` replaces instead of layering.** On the default transport a
   system prompt becomes `ThreadStartParams.baseInstructions`, which goes in
   *instead of* Codex's built-in "deployed coding agent" persona. `codex exec`
   put it on a `System: ` line **on top of** that persona. If you pass
   `system_prompt=` to `chat()` or open a `ChatSession` on a Codex connection,
   **your runs changed**: several thousand tokens of vendor framing are gone and
   the model is no longer told it is a coding agent. For most callers who
   brought their own persona that is the better run — it is the whole reason
   `system_prompt_replace` is now `supported` — but it is a change, not a fix.
2. **`mcp_servers=` is refused on the default.** Per-run MCP declaration and its
   enumerate-and-disable exclusivity are `codex exec` features with no driven
   equivalent on app-server.

**The escape from both is one line:**

```python
bridge.chat(connection="codex-sub", message="...", options={"transport": "exec"})
```

`exec` is fully supported, unchanged, and not on a removal path. Every receipt
names which transport actually ran.

**One caveat worth carrying into your own copy.** Replacing the instructions
does *not* by itself stop Codex behaving like a coding agent. `baseInstructions`
replaces the base instructions and nothing else — tool definitions, environment
context and `AGENTS.md` are untouched, which is why ~12.4k input tokens survive a
replacement that removed ~3,543 — and a model reads its identity off its toolbelt
as much as off its prompt. Switching the toolbelt off is a separate switch, and
since the change below **every chat-shaped call throws both**. Say "your persona
instead of Codex's framing", not "Codex stops being a coding agent".

### Codex chat calls no longer carry Codex's toolbelt (2026-08-31)

A second change the same day, and this one is a **repair** rather than a
consequence. `bridge.chat()` has always documented that "built-in tools are off
by construction here, so `tools=` means the caller's own functions and nothing
else." `anthropic-sdk` kept that promise. `openai-sdk` did not: every stateless
call ran with Codex's whole coding-agent belt attached — `exec`, `exec_command`,
`write_stdin`, `apply_patch`, `web.run`, `image_gen` and eight more — on both
transports.

It now switches them off, and the difference is measurable: an otherwise
identical bare call fell from **15,890 to 10,269 prompt tokens — 5,621 removed,
35.4% of the prefix** (driven live 2026-08-31 on codex-cli 0.151.0). Two things
to know:

- **Your own `tools=` are unaffected.** Driven in the same session: the caller's
  `dynamicTools` still registered, the handler still ran in-process, the model
  still answered from its result. Switching off the runtime's toolbelt does not
  switch off yours.
- **If you were relying on Codex having a shell during a `chat()` call**, opt
  back in with `options={"native_tools": True}`. A `WorkerSession` keeps the belt
  by design — iterating with `exec` is what a worker is *for*.

Two things this does **not** mean. Codex is still not cheap next to Claude: a
bare call is ~10.3k tokens against ~2.4k, because what remains is environment
context and protocol overhead that no config key reaches. And "toolbelt off"
means *the toolbelt modelpass can reach* — `unified_exec` is a stable feature that
both `-c` and `--disable` fail to switch off.

One asymmetry worth reading twice, because it changes what you can promise your
users:

* **Interim usage** is what makes a `stopAtTokens` a circuit breaker rather than a
  report. It is a circuit breaker on both runtimes as of 2026-08-31 — and a report
  again on Codex the moment a call passes `options={"transport": "exec"}`, which the
  receipt says on that run.

> **Retired 2026-08-31.** This section used to carry a second asymmetry: that
> built-in tools stay off on `anthropic-sdk` but that this "is not achievable on
> `openai-sdk` — Codex has no `tools=[]` equivalent and its shell tool is core to
> the runtime." That was wrong, and believing it is what let a stateless Codex
> `chat()` keep a shell for as long as it did. Codex has no `tools=[]`, but it
> has a config layer: six `-c features.*` / `web_search` keys switch the belt off
> and take 5,621 prompt tokens off the wire with it. A chat call on either
> runtime now cannot suddenly read the filesystem. `command_execution` items are
> still mapped to ordinary tool events with server `"codex"`, which is what a
> `WorkerSession` — or an opted-in `native_tools=True` call — produces.
* **The structured-output schema subset differs.** A schema derived from a Python
  dataclass with defaults works as-is on `anthropic-sdk` and is rejected by
  `openai-sdk` with a 400 — before any generation, so it costs nothing to discover,
  but it will not run. `modelpass.schema.to_openai_strict(schema)` converts one;
  `openai_strict_issues(schema)` tells you whether you need to, and the OpenAI
  preflight receipt says so too. Both constraints were captured from live
  rejections on 2026-08-17 ([the verification record](docs/api-and-runtimes.md#22a-where-the-cells-came-from-the-verification-record)).

## Prompt caching

Two different questions live here, and conflating them is how a consumer ends up
paying full price on every call without knowing. `ttl_control` asks *how long a
cached prefix lives*. `cache_breakpoints` asks *whether you get to say where the
cacheable prefix ends* — an Anthropic `cache_control` marker on a content block.
A runtime can have one, both or neither.

Pass a marked prefix as content blocks:

```python
from modelpass import CacheControl, TextBlock

events = bridge.chat(
    connection="claude-sub",
    system_prompt=(
        TextBlock("You are a careful reviewer."),
        TextBlock(long_stable_corpus, CacheControl(ttl="1h")),
        TextBlock(todays_variable_tail),
    ),
    message="Review this.",
)
```

A plain string is unchanged in every respect — same bytes, same rendering, same
receipt — so nothing about a call you already have moves.

| Runtime | Explicit breakpoints | What caching you do get |
| --- | --- | --- |
| `anthropic-sdk` | **unsupported** — `ClaudeAgentOptions.system_prompt` is `str \| SystemPromptPreset \| SystemPromptFile`, with no block list anywhere (read off claude_agent_sdk 0.2.148, 2026-09-13) | the CLI caches on its own, and `ttl_control` **is** supported: a session pins `1h` |
| `openai-sdk` | **unsupported** — no `cache_control` in the Responses API, and `baseInstructions` is a string | automatic, exact-prefix, ~5–10 minutes idle. Prefix stability is the whole of the control |
| `anthropic-api` | **supported** — `cache_control` markers reach `messages.create` intact, and this is the runtime the content-block vocabulary exists for. `ttl_control` is `unverified`: a `5m`/`1h` lifetime is carried through, but no test in this suite can observe one expiring | explicit, where you put the breakpoints |
| `openai-api`, `openai-compatible` | **unsupported** — no such field in the request schema | automatic, vendor-side |
| `google-api` | **unsupported** — Gemini's explicit caching is a separate cached-content resource, not a marker inside a message | implicit caching, automatic |

**Where breakpoints are not honoured they are not silently dropped.** The content
is flattened to text and the receipt that precedes the run carries the line
`cache breakpoints were requested and dropped: <runtime> does not accept them`,
alongside two counts you can also read from the run log:

```python
receipt.cache_breakpoints_requested   # what you asked for
receipt.cache_breakpoints_honoured    # what reached the vendor
```

Two numbers that disagree is a run that paid for a prefix it thought it had
cached. That is the failure this exists to end: three consumers were building
four-breakpoint prompts and having the markers stripped at the door, with nothing
anywhere reporting it.

## Sampling controls

`temperature`, `top_p`, `top_k`, `max_output_tokens` and `reasoning_effort` are
request fields on `Bridge.chat`, carried by one frozen `Sampling` object:

```python
from modelpass import Bridge, Sampling

for event in bridge.chat(
    connection="claude-key",
    message="Summarise this in one sentence.",
    sampling=Sampling(temperature=0.0, max_output_tokens=256),
):
    ...
```

**The honesty rule: modelpass sends what the model takes, and the receipt says
what it sent.** Support varies *per model*, not only per runtime — GPT-5 accepts
only `temperature=1.0`, several Anthropic 4.x models accept one of
`temperature` or `top_p` but not both, the o-series accepts none of them, and an
Anthropic model asked to think adaptively takes no sampling parameters at all.
A capability cell cannot express that, so it does not try: the cell says whether
the *runtime's request schema* has the field, and every call carries its own
report.

Three fields on every receipt, and on every line of `runs.jsonl`:

| Field | What it holds |
| --- | --- |
| `sampling_requested` | what you asked for, in modelpass's field names |
| `sampling_applied` | what is actually being sent |
| `sampling_notes` | one sentence per field that was dropped or coerced |

```python
receipt = bridge.preflight("openai-key", model="gpt-5", sampling=Sampling(temperature=0.2))
receipt.sampling_applied   # {'temperature': 1.0}
receipt.sampling_notes     # ('temperature 0.2 requested; gpt-5 accepts only 1.0, sent 1.0',)
```

It is a *pre-run* report, from `modelpass/sampling_rules.py` rather than from a
vendor, so you can read it before you spend. A field the wire protocol requires
and you did not set is disclosed too: an `anthropic-api` call with no ceiling
carries a 4096-token one, and the note says so rather than leaving it to be
discovered in a truncated answer.

**On a subscription runtime, sampling is dropped — and named.** Passing
`sampling=` to `anthropic-sdk` or `openai-sdk` is not an error. Every field is
dropped, every drop gets a sentence, and the run proceeds. That is deliberate:
refusing would be the right answer for a tool a runtime cannot run, and the
wrong one here, where "send what it takes and say what happened to the rest" is
available.

**`max_output_tokens` is not `stopAtTokens`.** The first is a ceiling on one
answer, which the vendor enforces by truncating. The second is a modelpass guard
over a whole run's total spend, which ends the stream. Callers who want shorter
answers want the first; callers who want a bounded bill want the second; set
both if you want both.

An unknown model gets the runtime's defaults and a note saying so — modelpass
never guesses a model into a family, because guessing that something reasons
sends a parameter the vendor rejects and the call stops working, while guessing
the other way costs one line of disclosure.

## Timeouts, retries, and threads

Three contracts that used to be four apps' worth of workarounds. Added 2026-09-13
(ticket 1.12).

### A call can be bounded

```python
from modelpass import Timeout

answer = bridge.ask(
    "claude-sub", "Score this posting: ...",
    timeout=Timeout(total=120, first_token=20),
)
```

`total` bounds the whole run's wall clock. `first_token` bounds the narrower wait
for the first event that is not the receipt — the *is anything happening at all*
question, which on a subprocess-driven runtime is what separates a slow answer
from a runtime that never came up. A bare number is `total`.

When a bound expires, modelpass cancels the run through the adapter's own
`cancel()` hook and the stream still ends the way every stream ends: **one
terminal event, stamped, carrying the tokens spent before the bound ran out**,
with `status = timed_out`. A timed-out run is still a billed run, and the
terminal is where you find out what it cost. `ask()` and
`chat(raise_on_stop=True)` raise `RunTimedOut` instead; the LangChain leaf
raises `SubpassTimeoutError`.

A connection can carry a default with `timeoutSeconds`. A call that passes its
own `timeout=` **replaces** it rather than being clamped by it.

Cancelling by closing the iterator is unchanged and still the documented way to
stop a run early (D10). It cancels exactly once, with or without a bound
configured.

### modelpass never retries for you

Say that part first, because it is the part people assume otherwise. There is no
retry loop in this library, no backoff, and no configuration for either. What
modelpass provides is the **verdict**:

```python
for event in bridge.chat(connection="claude-sub", message="..."):
    if isinstance(event, TerminalEvent) and event.status is not TerminalStatus.OK:
        if event.retryable is Retryable.YES:
            schedule_again(after=event.retry_after)
```

`Retryable` has three values and not two, because *we do not know* is a real
answer and a boolean has to guess which way to lie:

| verdict | what it means |
|---|---|
| `YES` | transient: a 5xx, a 429 that named a `retry-after`, a connection reset, a `first_token` timeout on a stateless run |
| `NO` | deterministic: a caller mistake, auth, a guard stop, a spent allowance with no failover, a content filter, a rejected schema, a truncation |
| `UNKNOWN` | the vendor reported nothing typed to classify it by |

Every error modelpass raises carries the same verdict on `.retryable`, and
`.retry_after` where the vendor named a wait. That includes the errors `ask()`,
`aask()` and `chat(raise_on_stop=True)` raise in a terminal event's place: each one
carries that terminal's verdict — `.retryable`, `.retry_after`, `.status_code`, and
the event itself on `.terminal` — so a 429 reaches a retry loop as a 429 rather than
as an unclassified failure.

**The verdict is computed from typed fields only — status codes, exception
classes, terminal statuses — and never from the text of a message.** That rule is
not fastidiousness. A consumer once classified by substring-matching the message,
`"500"` matched inside `"stopAtTokens threshold 500000 reached"`, and a
deterministic guard stop was retried four times against a live subscription
allowance. The table lives in one module, `modelpass/retry.py`, with a dated note
on every row.

### A connection can say "never"

```toml
[connections.production-judge]
retry = "never"
```

Every verdict on that connection reads `no`, with a note saying the policy is why.
The receipt reports the stance before the run rather than leaving you to infer it
afterwards. It does not make modelpass retry the other connections — nothing makes
modelpass retry — it is how you tell *your* loop that this one is off limits.

### A connection can say how big its input window is

```toml
[connections.production-judge]
maxInputTokens = 200000
```

Read back as `connection.max_input_tokens`. It is **your** statement of the
model's usable input window, in tokens, for the tool that needs to decide whether
a payload will fit before it spends a call finding out.

**Nothing in modelpass reads it.** It bounds nothing, truncates nothing and
refuses nothing — it is a stance, like `retry = "never"`, not a mechanism.

**Omitting it means *unknown*, and unknown is the normal case.** There is no
table of model names to window sizes in this library: a vendor changes a window
without changing the model string, so a built-in number would go stale silently
while reading as authoritative, and a wrong window is worse than none. Unknown
and zero are different facts, so `maxInputTokens = 0` is refused rather than
stored, and anything that is not a positive whole number is refused with it.
Read it as `is None`, never as falsy.

`modelpass list --verbose` prints it, and prints `unknown` when nothing is set.

### A connection can ask for prompt caching, and be told what that bought

```toml
[connections.production-judge]
promptCache = "default"   # or a lifetime this runtime accepts, e.g. "1h"
```

Read back as `connection.prompt_cache`. It is the **standing** half of prompt
caching. The per-call half already exists and is unchanged: `CacheControl` on a
`TextBlock` is how you say *where* the cacheable prefix ends, which only a
caller holding one payload can say.

**Asking has three possible answers, and they are not the same answer.**

* **Honoured explicitly** — the runtime takes an instruction, so your
  `cache_control` breakpoints reach the vendor. `anthropic-api` today.
* **Already satisfied** — the runtime caches without being asked and gives you
  no way to stop it, which is true of most of them. This is **not an error**.
  Nothing is sent for it and nothing needs to be.
* **Refused** — modelpass has not established that this runtime caches prompts,
  so there is nothing to honour and nothing already happening. You get an
  `InvalidConnection` when the connection is built, rather than a setting that
  quietly does nothing.

You tell the first two apart on the receipt:
`receipt.prompt_cache_disposition` reads `"explicit"` or `"automatic"`, with one
explaining sentence in `receipt.notes`. `modelpass list --verbose` prints the
same thing.

**Omitting the key states nothing, and that is not "caching off."** Most of
these runtimes cache whether or not anyone asks; a connection with no
`promptCache` has simply not said anything about it.

**The lifetime is the vendor's word, not modelpass's.** `"default"` means the
vendor's own, and is distinguishable from every explicit value. An explicit one
is checked against what that runtime accepts, read off the SDK installed here —
`anthropic` types `'5m'`/`'1h'`, `openai` types `'in-memory'`/`'24h'`. There is
no house vocabulary and no translation between them, because nobody published
one. A lifetime a runtime does not take is refused when the connection is built,
not by a 400 halfway through a run.

### What is safe to share between threads

* **One `Bridge` is safe to share** across threads for `chat`, `ask`, `preflight`
  and `validate`. Each call builds its own request, its own guard tracker and its
  own stream, and shares nothing mutable, so N workers on one bridge get N
  independent answers, receipts and usage figures. The connection store and the
  run log are guarded by their own locks. Adapters may be shared.
* **A `Session` takes one caller at a time.** It holds a conversation, a token
  tracker for the whole conversation, and a vendor handle whose turns are
  ordered. A second caller arriving mid-turn gets `SessionBusy` — raised, not
  queued, because waiting would produce interleaved turns in an order nobody
  chose, and a lock that queues can deadlock when one of the two callers is the
  thread draining the other's iterator. Give each thread its own session
  (`new_chat()` is local work) or take your own lock around the turn.
* **One event loop may run many `achat` calls concurrently.** The same sharing
  rule, on the other face: the calls are independent, and the one blocking step
  in a call -- the agent runtimes' subprocess preflight -- is handed to a thread
  rather than run on the loop.

### Async

```python
async for event in bridge.achat(connection="claude-api", message="..."):
    ...

answer = await bridge.aask("claude-api", "Score this: ...", schema=SCHEMA)

async for event in session.asend("and then?"):
    ...
```

`achat`, `aask` and `asend` take every argument their sync twins take and answer
with the same events, receipts, guards and ledger lines: one pre-run pipeline and
one event fold underneath, two doors on top. `chat`, `ask` and `send` are
unchanged and are still the whole implementation for a sync caller.

**Cancelling is `aclose()`**, where the sync contract is closing the iterator:

```python
events = bridge.achat(connection="claude-api", message="...")
try:
    async for event in events:
        ...
finally:
    await events.aclose()
```

That cancels the run through the adapter's own `cancel()` and joins the worker
thread where one was used. On the four API runtimes there is no worker: their
adapters drive the vendor's async client on your loop, and an `async def` tool
handler is awaited there too -- no thread hop, no `run_coroutine_threadsafe`.
Added 2026-09-13 (ticket 1.13).

## Sessions, where the runtime holds the conversation

`chat()` is stateless: whatever `history=` you pass is re-sent every call, which is a
known cost because you can see it. A **session** is the other door — the runtime holds
the conversation, the prefix cache is native to it, and you send one turn at a time.

```python
with bridge.new_chat(
    connection="claude-sub",
    system_prompt=RUBRIC,
    stop_at_tokens=200_000,        # the whole session's ceiling, not the turn's
) as session:
    for item in items:
        for event in session.send(f"Score this: {item}"):
            ...
    print(session.id, session.usage.total_tokens)
```

Four doors: `new_chat()`, `new_worker(project_folder=...)`,
`resume_chat(session_id=...)` and `list_sessions()`. A `ChatSession` **replaces** the
runtime's own agent persona with your `system_prompt` and switches its native toolbelt
off — a scorer, a classifier, a rewriter, an assistant. A `WorkerSession` **keeps** both
and appends yours, and requires a `project_folder`, because work where the runtime's own
agent pointed at a real folder is the point is what it is for.

Three properties of the shape, each deliberate:

* **There is no `tools=` and no `system_prompt=` on `send()`.** Both are fixed at
  construction, because they sit at the very front of a prefix that has been warm since
  the first turn; re-declaring one would mean either ignoring it silently or changing
  that prefix silently.
* **Guards on a session are the session's envelope**, not the turn's. `stop_at_tokens`
  bounds everything this session will ever spend — a per-turn tracker on a fifty-turn
  worker never fires while the allowance drains.
* **A session takes one caller at a time.** A second arriving mid-turn gets `SessionBusy`
  rather than an interleaved conversation, raised instead of queued because a lock that
  queues can deadlock when one of the two callers is the thread draining the other's
  iterator. `asend()` is the async twin and contends for the same lock.

Construction runs the preflight and every capability assertion and hands back a local
object with a receipt already available — but **nothing is spent and no session exists
yet**. Neither runtime has a create-session call, so the session comes into being when
the first `send()` completes, which is also when `Session.id` stops being `None`. A
stored id is therefore a hint rather than a guarantee, and `SessionNotFound` from
`resume_chat()` is an ordinary outcome rather than a broken caller: Claude Code sweeps
old transcripts on its own schedule.

**Sessions exist on the two agent runtimes only.** The four API runtimes hold no
conversation — every request carries its whole history — so all four doors are shut
there, at the bridge, before an adapter is loaded or a credential is resolved:

```
runtime 'anthropic-api' capability 'sessions_resume' is unsupported:
bridge.new_chat() needs the runtime itself to hold the conversation (D14), and
'anthropic-api' holds none: every request carries its whole history. Use
bridge.chat(history=[...]), the stateless alternative, or bridge.ask(...) when
one blocking answer is what you want
```

That refusal is read off the capability table rather than hard-coded per runtime — the
rule is every one of `sessions_resume` / `sessions_fork` / `sessions_list` reading
`unsupported`, which is what an HTTP request/response API looks like from here. It is
`CapabilityNotSupported`, carrying `.runtime`, `.capability` and `.support` for a caller
who would rather branch than read English. `new_worker()` carries a second refusal of
the same class on the `tools` cell: a runtime with no native toolbelt would otherwise
have handed back a chat wearing a worker's name.

The two models are set side by side in
[docs/api-and-runtimes.md](docs/api-and-runtimes.md).

## Managing connections from Python

`modelpass connect` is a consent flow, and a desktop app's Settings page needs the
same one without a terminal. `bridge.manage` is that flow as a library: the same
validation, the same receipt, and the same rule that nothing is written before the
receipt exists.

```python
plan = bridge.manage.plan_connection(                  # writes nothing
    name="claude-api",
    runtime="anthropic-api",
    credential=modelpass.Credential.secret(pasted_key),  # or .env("MY_KEY"), or .native_login()
)
print(plan.receipt.summary(), plan.notes)              # show this before you save
bridge.manage.add_connection(plan)                     # secret first, then the connection
bridge.manage.remove_connection("claude-api").note     # "removed secret entry 'claude-api'"
```

`rename_connection(old, new)` moves the key with the name when it is unambiguously
that connection's, `set_enabled(name, False)` stops runs without deleting anything,
`set_groups(name, ["fast"])` replaces the groups a connection is in, and every result
is a dataclass with a `to_dict()` that is safe to print — the
pasted value is in none of them, in none of their `repr`s, and never reaches the
connection file.

## The bench

```
pip install modelpass[bench]
modelpass bench            # http://127.0.0.1:8765
```

A local page — loopback only, no authentication because it is not a service — that shows
the things this README claims, rather than describing them:

* **Connections** — everything configured, with each one's receipt one click away, and
  an **ambient-environment audit**: which credential variables are set on this machine
  right now (names only, never values) and the statement that runs will scrub them. It
  also notes that the audit reflects the environment of *that process* — a terminal
  opened before you cleaned your environment still carries stale copies, which is
  precisely why modelpass scrubs at launch instead of trusting what it inherits.
  Each row also carries what a run turns on: the runtime and auth mode, the credential
  *source* (never a value), the base URL where there is one, the timeout bound and the
  retry stance, and — on `openai-compatible` — the capability cells a drive recorded and
  when, or a plain statement that nothing has driven this endpoint yet. Below the
  listing, the **stored keys** panel names each entry in the secrets file and the
  connections referencing it; an entry nothing references is marked an orphan and there
  is deliberately no button here that deletes one.
* **Vendor setup** — create, edit, rename, enable/disable and delete connections, through
  the same manager `modelpass connect` uses, so the secrets file is written by one
  implementation rather than two. Four credential forms: the runtime's own login, a named
  environment variable, a key pasted once into a write-only field that is never echoed
  back into the page, or none at all for an endpoint that checks nothing. A pasted key in
  the *variable name* field is still refused by the validation that guards the config
  file. Renaming moves the stored entry with the connection.
* **Groups** — every group, its members, and which one `group:<name>` would reach, with
  the reason each other member was passed over. A group whose members are all disabled
  is shown as one that would be refused, because in a plain listing it reads exactly
  like a group that works. Each account's form has a groups box, and each row's group
  tags filter the listing.
* **Verify** — on a subscription, pin the vendor identity currently logged in; on an
  `openai-compatible` connection, drive the endpoint and record what answered. The
  second confirms first, because it spends.
* **Playground** — a real chat with a model picker (the Codex CLI's own model list where
  one exists; the models a drive saw your endpoint serving where one has run; labelled
  aliases where neither does), streaming the **entire event log** live: deltas, thinking,
  usage, guard events, tool calls, and the stamped terminal. There is a hello-world tool
  you can attach to watch a caller-supplied tool round trip end to end. It spends real
  allowance and says so.
* **Run log** — recent runs from `runs.jsonl`, newest first, with the three questions a
  ledger row could not previously answer: were the `cache_control` breakpoints honoured,
  was the run at the sampling its config says, and is this failure worth retrying — the
  last computed from the row's own runtime and terminal status by the same function the
  library gives your loop.

## Command line

Every command, in one table. `modelpass --help` and `<command> --help` carry the same
text, and the `subpass` alias invokes all of it identically until 0.3.0 — the help
output names whichever of the two you actually typed.

| Command | What it does |
| --- | --- |
| `modelpass connect <vendor>` | Run the preflight, print the receipt, and write a connection once confirmed. Vendors: `anthropic`, `openai`, `google`, `compatible`. `--name NAME` names it; `--nickname LABEL` adds the human name; `--model NAME` pins a model; `--description TEXT` leaves you a note; `--config-dir PATH` selects isolated vendor storage; `--runtime NAME` picks the vendor's runtime exactly; `--base-url URL` names the endpoint; `--api-key-env NAME` selects metered mode on a named variable; `--api-key-stdin` takes the key on a pipe and stores it in the secrets file; `--warn-at-tokens N` and `--stop-at-tokens N` set guards; `--allow-env NAME` lets one non-credential variable through the scrub; `--group NAME` puts it in a group (repeatable); `--force` overwrites an existing connection; `-y` writes without asking. |
| `modelpass list` | One line per connection: runtime, auth mode, credential source, base URL, groups, guards, disabled state. `--group NAME` narrows it to one group. `-v` adds the timeout bound, the retry stance and the capability cells a drive recorded. |
| `modelpass accounts` | The same connections read as account profiles — nickname, stable ID, guards, whether the identity is pinned, disabled state. `-v` adds the model, `allowEnv` and `configDir`. `list` is the alias-compatible legacy spelling of this view. |
| `modelpass check [NAME]` | Re-run the preflight and print the receipt, for one connection or for all of them. Exits non-zero if a connection would not run. Reports the secrets file's permissions and any orphan entries, and on an `openai-compatible` connection what a drive found — or that nothing has driven it yet. |
| `modelpass verify NAME` | Go and find out the live answer, and write it down. On a subscription: re-read the vendor's account identity, bypassing the preflight identity cache, and pin the connection to it. On an `openai-compatible` connection: drive the endpoint with one chat, one tool round trip and one structured-output call, and record which cells actually worked. **The second form spends**, and says so before it does. |
| `modelpass groups [NAME]` | The groups and who is in them, with the member `group:<name>` would actually reach and why the others were passed over. A group with members and no enabled one is reported as one that would be refused. |
| `modelpass set-groups NAME [GROUP...]` | Replace the groups a connection is in — the whole list, not one entry. Naming no group returns it to `default`. Says when a group has just lost its last member and no longer exists. |
| `modelpass enable NAME` | Put a connection back into service. |
| `modelpass disable NAME` | Take a connection out of service without deleting it: it stays listed, `check` still runs against it, and its guard thresholds survive. Says when this leaves a group with no enabled member. |
| `modelpass rename OLD NEW` | Rename a connection, moving its secret entry with it when the entry is that connection's own and nothing else references it — and saying so when it moves only one of the two. |
| `modelpass remove NAME` | Delete a connection, after printing what goes with it. Its stored key goes too when the entry was unambiguously that connection's; a key another connection references stays, and this says so. `-y` skips the confirmation. |
| `modelpass secrets` | The entries in the secrets file, and which connections reference each. An entry nothing references is named as an orphan and is never deleted for you. No value is printed, and no flag prints one. `-v` adds the file's permission note. |
| `modelpass bench` | Serve the local configuration and test page on `127.0.0.1:8765`. `--port N` moves it. Needs the `bench` extra. |

## Boundary and compliance

**Intended use:** personal development, local tooling, and workflows running under the
developer's own authenticated account — or, for a distributed tool, under *its user's*
own account.

**Explicitly not designed as:**

```
many external users -> one developer subscription -> shared model access
```

That shape is prohibited by at least one vendor's terms and unwise under all three. It
is also not something modelpass can be talked into: connections are local, credentials
stay in the vendor's own login store, and there is no server component. For hosted or
multi-user production workloads, use the vendors' official APIs and commercial terms.

Two more lines, held to throughout:

* **Never token-level.** modelpass drives vendor runtimes; it never extracts, stores or
  replays a subscription token. See [Runtime-level passthrough only](#runtime-level-passthrough-only--and-why-that-is-the-compliant-shape).
* **Never redistributes a runtime.** The vendor packages are dependencies, so proprietary
  bits arrive through the vendor's own channel under the vendor's own terms.

### Runtime-level passthrough only — and why that is the compliant shape

There are two ways to reach a subscription from code:

1. **Token-level** — extract the subscription's OAuth token and call the vendor's API
   with it.
2. **Runtime-level** — drive the vendor's own agent runtime, as the vendor intends.

**modelpass does (2) only, and will never do (1).** Not as a matter of taste:

* Anthropic restricted consumer-plan OAuth to Claude Code and claude.ai in a 2026-02
  legal update, with server-side enforcement — extracted consumer tokens now error on
  direct API requests.
* Token extraction is the pattern behind community-reported account blocks, and at
  least one vendor's terms now name a token-level tool as their worked example of a
  breach.
* Runtime-level use is explicitly covered — Anthropic's Agent SDK credit exists for it.

**modelpass is not that pattern.** Token-extraction tools lift the subscription's
OAuth token and impersonate the vendor's own app against the vendor's API — which is
what the vendors have moved against, and what they now block server-side. modelpass runs the
vendor's own runtime, under your own login, on your own machine, exactly as the vendor
ships it — and shows you a receipt proving which account paid. One is circumvention;
the other is using what your plan includes. The per-vendor legal pages below spell out
each vendor's written position with links to the primary texts.

It is also the only architecture that keeps working. A token-scraping tool breaks the
next time a vendor rotates a storage format; a tool that drives the vendor's own runtime
breaks when the vendor breaks their own runtime, which they have every reason not to do.

### The vendors' written positions, summarized

One page per vendor: what their terms say (quoted and dated), what that means for using
modelpass with that vendor, what you must never do, and links to the primary texts.

| Vendor | Position (summarized) | Details |
| --- | --- | --- |
| Anthropic | Individual Agent SDK use under your own plan is **explicitly covered**; mediating other people's credentials is prohibited | [docs/legal/anthropic.md](docs/legal/anthropic.md) |
| OpenAI | ChatGPT-authenticated Codex use is a **documented plan feature**; no prohibition found on wrapping the first-party CLI, but no affirmative blessing either | [docs/legal/openai.md](docs/legal/openai.md) |
| Google | **Excluded.** The terms name third-party tools as a breach, penalty account termination — do not wire a Google subscription through third-party tooling at all | [docs/legal/google.md](docs/legal/google.md) |

**The disclaimer that applies to all three:** those pages are summaries written by this
project, not legal advice, and not the terms themselves. **Vendors may update their
terms at any time, without notice**, and enforcement postures shift faster than
documentation. The dates on each page say when the primary sources were last read;
verify them yourself before relying on anything, and treat your account — and anything
a vendor does to it — as your own responsibility. modelpass is provided AS IS under the
[MIT license](LICENSE), without warranty of any kind.

### A note on freshness

**Every provider fact in this repository is dated, and this space moves fast.** Within
the last six months: Google replaced its entire CLI (June 2026), Anthropic introduced
the Agent SDK credit (June 2026), and OpenAI moved Codex to token-based accounting
(April 2026). Auth precedence is exactly the kind of thing that changes in a point
release — which is the argument for the scrub in the first place.

**Re-verify before implementing against any dated claim here**, including the compliance
ones. The verification documents below record what was checked, when, and against what.

## Migrating from subpass

The library was called `subpass` through 0.1.1. Everything about it is the same except
the name, and 0.2.0 ships the old name as a shim so nothing has to move on your schedule.

| What | Before | Now | Old spelling works until |
| --- | --- | --- | --- |
| Import | `import subpass` | `import modelpass` | 0.3.0, with one `DeprecationWarning` |
| Submodules | `from subpass.langchain_adapter import ChatSubpass` | `from modelpass.langchain_adapter import ChatSubpass` | 0.3.0 |
| Config directory | `~/.subpass/` | `~/.modelpass/` | 0.3.0 (copied forward on first use) |
| Env override | `SUBPASS_HOME` | `MODELPASS_HOME` | 0.3.0, with a `DeprecationWarning` |
| Command | `subpass connect anthropic` | `modelpass connect anthropic` | 0.3.0 (alias entry point) |

Two things deliberately did **not** change: the exception taxonomy is still
`SubpassError` and its subclasses, and the LangChain chat model is still `ChatSubpass`
with its `subpass_*` `response_metadata` keys. Renaming something a caller catches or
reads by name is a contract change, and it is not part of a rename. They get their own
decision later.

The shim is not a copy. `subpass.Bridge is modelpass.Bridge` — every name it hands out is
the object `modelpass` defines, so `isinstance` checks and `except` clauses still work
when one half of a program has migrated and the other has not.

On first use, if `~/.modelpass/` does not exist and `~/.subpass/` does, your connections
are copied forward and a one-line note says so. The old directory is never deleted, so a
tool still running 0.1.x keeps reading exactly what it always read.

## Contributing and development

```
git clone https://github.com/staceyfarias/modelpass
cd modelpass
python -m venv .venv
.venv/Scripts/activate            # source .venv/bin/activate on macOS or Linux
pip install -e ".[dev]"
python -m pytest -q
ruff check src tests
```

The `dev` extra installs pytest, ruff, Flask, `langchain-core`, and the three vendor
clients the API adapters are tested against. The two **subscription** runtimes need
their own extras and their vendor CLI on `PATH` (see [Install](#install)); their tests
use fakes and need neither.

You do not need any of it to run the suite. `pip install pytest` against a bare
checkout is enough, and that is exactly what CI does — nothing but the test runner, so
the run is also the check that modelpass still imports with zero dependencies. The few
cases that cannot be faked, because they resolve an adapter through the registry or
drive a `Bridge` that refuses a runtime whose SDK does not import, skip there with a
reason naming the extra they want; installing `.[dev]` runs them.

Four rules the repository holds itself to, spelled out in [AGENTS.md](AGENTS.md):

* **Core keeps zero runtime dependencies.** Vendor SDKs are extras, and the modules
  outside `adapters/`, `bench/` and `langchain_adapter.py` import none of them.
* **A capability cell moves only with recorded evidence** — a test *and* the SDK surface
  it was read off, each with a dated note. `unverified` means nobody checked; it is not
  a soft yes.
* **No test spends allowance.** Live tests carry the `live` marker, are deselected by
  default, and need `SUBPASS_LIVE_TESTS=1` plus their own per-file variable before they
  will run at all. Having a vendor key on the machine is never on its own enough.
* **User-visible wording, `event.type` strings, error class names and `pip install`
  hints are consumer contract**, and do not change outside a ticket that says so.

The roadmap, the ticket plan, the accepted design, the dated decision log and the
per-phase verification write-ups are **not published**. What is published is the contract and the
evidence behind it — including
[the verification record](docs/api-and-runtimes.md#22a-where-the-cells-came-from-the-verification-record),
which carries what each pass checked, against which build, on what date, and the
measurement each capability cell rests on — plus [CHANGELOG.md](CHANGELOG.md) for what
each release actually changed.

## Documentation

| Document | What it is |
| --- | --- |
| [docs/api-and-runtimes.md](docs/api-and-runtimes.md) | **The API by contract**: every entry point's promises and refusals, the full capability matrix, the dated verification record behind it (§2.2a — what was checked, against which build, on what date), and what each runtime does with your history, prompt caching and token accounting |
| [CHANGELOG.md](CHANGELOG.md) | What each release changed, with a `Contract names` list per release |
| [docs/subscription-proof-2026-08-17.md](docs/subscription-proof-2026-08-17.md) | **How do I know it was the subscription?** The falsification experiment, its control, and the ledger you can check without trusting modelpass |
| [docs/legal/anthropic.md](docs/legal/anthropic.md) · [openai.md](docs/legal/openai.md) · [google.md](docs/legal/google.md) | Per-vendor legal summaries: what the terms say, what you may and must not do, primary links — **not legal advice; terms change at any time** |
| [docs/guards.md](docs/guards.md) | Bounding what a run can spend: why there are no defaults, how to pick a number, when guards can actually fire on each runtime, where the vendor-side hard cap a metered connection needs is set, and why the unit is still tokens |

## License

[MIT](LICENSE). The runtimes modelpass drives are the vendors' own packages under their own
licenses and terms; modelpass never redistributes them.
