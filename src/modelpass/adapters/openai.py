"""OpenAI runtime adapter (``openai-sdk``): the Codex runtime under a ChatGPT login.

Implemented in ROADMAP Phase 4 against two sources of truth:

* the official headless surface -- ``codex exec --json`` emits the documented
  JSON Lines vocabulary (``thread.started`` / ``turn.started`` /
  ``item.completed`` / ``turn.completed`` / ``turn.failed`` / ``error``); and
* the live cancellation characterization of 2026-08-16 (``codex-cli`` 0.117.0,
  hard process-tree kill), summarized in ``docs/api-and-runtimes.md`` §2.2a: no
  terminal event of any kind is emitted, so this adapter synthesizes its own
  ``canceled`` terminal.

**Transport choice.** This adapter drives the Codex CLI binary directly rather
than going through the ``openai-codex`` Python SDK's thread objects. The SDK's
Python *streaming* surface is undocumented (verified 2026-08-16 against
learn.chatgpt.com/docs/codex-sdk, which documents ``thread.run()`` but no event
stream), while the CLI's ``--json`` stream is documented and now experimentally
characterized. For stateless one-turn runs (D7) the SDK adds nothing on top of
the CLI it bundles. The ``modelpass[openai]`` extra still installs ``openai-codex``
because its pinned CLI is a valid binary source; a ``codex`` on PATH works too.

Facts this adapter encodes (dated; re-verify on upgrade):

* Import name of the ``openai-codex`` distribution is ``openai_codex``
  (pinned 2026-08-16; settles implementation-plan open question 3).
* ``codex exec`` refuses to run outside a trusted directory without
  ``--skip-git-repo-check``; runs execute in a fresh temp dir by default.
* Error payloads arrive as JSON-encoded strings *inside* ``message`` fields --
  double-parse to get status codes out (experiment, 2026-08-16).
* A hard kill emits no terminal event and a thread is durable only after its
  first completed turn; stateless runs therefore simply evaporate when
  cancelled, and this adapter synthesizes the ``cancelled`` terminal (D10).
* No hard model pins: GPT-5.4 leaves ChatGPT-account Codex on 2026-08-31, and
  an out-of-date CLI against a newer server default fails with
  ``400 ... requires a newer version of Codex`` (observed live).
* **A multi-line argv element does not survive the Windows shim.** The resolved
  binary is usually ``codex.cmd``, a batch file, and ``cmd.exe`` terminates its
  command line at the first newline -- so every line of the prompt after the
  first was silently dropped while the run reported ``ok``. Reproduced
  2026-08-31 with a throwaway ``.cmd`` that echoed its arguments; no AI
  involved, no tokens spent.
* **So the prompt rides stdin**, with ``-`` in the ``[PROMPT]`` argv slot, which
  both ``codex exec`` and ``codex exec resume`` document as *read the prompt
  from stdin* (checked in the installed CLI's ``--help``, codex-cli 0.151.0,
  2026-08-31). It is one or the other and never both: given a real prompt
  argument *and* piped stdin, the CLI appends the stdin text as a separate
  ``<stdin>`` block rather than using it as the prompt.

How long that answer stays right, and what replaces it, is written down:
The 2026-08-31 migration record holds the dated transport
rationale and the migration onto ``codex app-server``, whose JSON-RPC surface
carries the prompt as a field and has no command line to truncate.

**Two transports, app-server is the default since S7 (2026-08-31), and the
receipt names which one ran.** The stateless :meth:`OpenAIAdapter.run` drives
either ``codex app-server`` (no option needed) or ``codex exec --json``
(``options={"transport": "exec"}``) -- see :func:`resolve_transport` for why the
lever is an option rather than a connection field, and why an unrecognized value
is refused instead of falling back. Same runtime identity, same connection:
what changes is that four things Phase 4 had to work around are parameters.
A system prompt is ``ThreadStartParams.baseInstructions`` rather than a
``System: `` line; a conversation is seeded into the thread's own history with
``thread/inject_items`` rather than flattened into a transcript with a preamble;
an output schema is a per-turn ``outputSchema`` field rather than a temp file
passed to ``--output-schema``; and (S6) ``tools=`` becomes ``dynamicTools``, so
the caller's own functions run in this process. **S5 adds sessions**: a
conversation here is one long-lived child holding one thread -- ``thread/start``
once, ``turn/start`` per turn, ``thread/resume`` to pick one back up -- which is
what makes ``get_history()``, ``list_sessions()`` and a ``ChatSession`` possible
on Codex at all. See :class:`CodexAppServerSession`.

**``exec`` is the opt-out, not a deprecation.** Its code path, its tests and its
behaviour are untouched by S7; only which one you get by default moved. Two
things still send a caller there, and both are named on the receipt:

* **``system_prompt`` semantics.** ``baseInstructions`` *replaces* Codex's
  built-in coding-agent persona where exec's ``System: `` line layered on top of
  it. A caller who passed ``system_prompt=`` to ``chat()`` or opened a
  ``ChatSession`` on a Codex connection before 2026-08-31 gets a **materially
  different run** now -- several thousand tokens of vendor framing gone, and a
  model no longer told it is a deployed coding agent. That is the better default
  for somebody who brought their own persona, and it is a change rather than a
  fix, so it is announced rather than buried. ``options={"transport": "exec"}``
  is the one line that restores the old behaviour.
* **``mcp_servers=``.** Per-run MCP servers and their enumerate-and-disable
  exclusivity are an exec capability with no driven equivalent on this path, so
  an ``mcp_servers=``-bearing call is refused on the default and works on the
  opt-out. The capability row moved to ``unsupported`` in the same commit for
  the same reason the others moved to ``supported``: a row describes the default.

**History on this transport is native, and the labels are the fallback**
(2026-08-31). ``thread/inject_items`` appends *raw Responses API items* -- real
``{"type": "message", "role": ..., "content": [...]}`` shapes -- to the thread's
model-visible history, so prior turns arrive with roles instead of with
``Assistant: `` written in front of them, and only the current message rides on
``turn/start``. Driven live on an ephemeral thread, which is the kind the
stateless call creates. S4's labelled transcript is kept and used when the
server refuses the method, with a ``vendor_event`` announcing the downgrade:
this surface moves (the method is snake_case on the wire while the vendor's own
generated types imply camelCase), and losing a conversation to a spelling would
be far worse than sending it the older way. ``codex exec`` is untouched -- it
has no such seam and :func:`render_prompt` is unchanged.

Phase 6 (D12) adds MCP passthrough, and lands *partial* on this runtime by
design -- the capability registry says so rather than pretending parity:

* **Per-run MCP servers: yes.** ``codex exec -c 'mcp_servers.<name>={...}'``
  registers a server for one invocation, verified live 2026-08-16 with
  ``codex mcp list``. No user config file is written.
* **Exclusivity: yes, by enumerate-and-disable** (added 2026-08-16, after the
  merge problem was first recorded as unsolvable). ``-c`` *merges* with
  ``~/.codex/config.toml`` and ``mcp_servers={}`` does not clear it — but
  ``codex mcp list --json`` enumerates the configured servers token-free, and a
  per-server ``-c mcp_servers.<name>.enabled=false`` switches each one off for
  the invocation. MCP-bearing runs do this automatically and **fail closed** if
  the enumeration fails; ``options={"allow_configured_mcp_servers": True}`` is
  the explicit opt-out. Plain chat runs are untouched (the runtime's own
  config behavior, as live-validated in Phase 4).
* **In-process caller tools: the default since S7, and absent only on the exec
  opt-out** (2026-08-31). ``codex exec``'s only tool channel is MCP, declared on
  a command line; there is no in-process registration there and Phase 6 recorded
  that. What Phase 6 got wrong was the scope, and the correction is the reason
  this bullet was rewritten three times: the absence belongs to *the transport*,
  not to the vendor. ``codex app-server`` registers caller functions natively --
  ``ThreadStartParams.dynamicTools``, the server's ``item/tool/call`` request,
  a ``contentItems`` response -- the loop was driven live and then driven again
  end-to-end through ``bridge.chat()``, so ``bridge.chat(..., tools=[...])``
  works with no options at all. The mapping lives in
  ``adapters/codex_appserver``'s dynamic-tools section. **The registry cell reads
  ``supported`` since S7**: it describes the DEFAULT transport, and that is now
  the one that runs the loop. Selecting exec narrows it back, which is what
  :meth:`OpenAIAdapter.support_for` answers and what the exec refusal says.
* **Built-in tools: off on a chat-shaped call since D23 (2026-08-31).** This
  bullet used to read "still on ... not achievable here", on the reasoning that
  Codex has no documented ``tools=[]`` equivalent and its shell tool is core to
  the runtime. The first half is true and the conclusion did not follow: Codex
  has no ``tools=[]``, but it has a **config layer**, and six ``-c`` keys
  (:func:`chat_tool_overrides`) take the belt off and 5,621 prompt tokens off
  the wire with it. So D12's "built-ins off by default" is honoured on both
  runtimes now, not only on Anthropic. A ``WorkerSession`` -- or
  ``options={"native_tools": True}`` -- keeps them, deliberately. Sandbox policy
  remains the lever for what a *worker's* shell may do; it is no longer the only
  lever for whether a chat has one.

Tool item shapes were verified on 2026-08-16 (``docs/api-and-runtimes.md``
§2.2a): the item *type names* were extracted from the shipped ``codex.exe``, and the
``command_execution`` *field* names were confirmed live on 2026-08-30 (fixture:
``tests/fixtures/tools/openai-command-execution-2026-08-30.jsonl``). The
``mcp_tool_call`` field names remain third-party-sourced, so
:func:`map_codex_event` reads every tool item defensively and falls back to
``vendor_event`` whenever the expected keys are absent.

That same capture turned up a ``status`` value the documentation does not list --
``declined``, emitted when the sandbox refuses a command. It is an error status
here. Until 2026-08-30 a declined command reached ``is_error=True`` only
incidentally, via its ``exit_code`` of ``-1``; one with ``exit_code: null`` would
have been reported as a **success**.

Two influences on a Codex run cannot be scrubbed, so the preflight receipt
**names** them instead (2026-08-30):

* **Which binary.** Two Codex installs can coexist -- an npm shim on ``PATH`` and
  a newer build the Desktop app uses, named by ``CODEX_CLI_PATH`` in
  ``config.toml`` -- and they do not accept the same models. The receipt reports
  the resolved path and its ``--version``, and says so when the config names a
  different one. modelpass does not *switch* binaries on a config value it went
  looking for; ``options={"codex_bin": ...}`` is the written-down lever (D2).
* **``AGENTS.md``.** Codex fills ``turn_context.user_instructions`` from
  ``<codex home>/AGENTS.md`` on every run and from ``AGENTS.md`` in the run's cwd
  and every ancestor directory. Relocating ``CODEX_HOME`` would isolate it and
  orphan ``auth.json``, so this is disclosure rather than a scrub: the receipt
  names every file that will join a prompt the caller did not write.

Phase 8 (D13) adds structured output, native here too and verified live on
2026-08-17 (``docs/api-and-runtimes.md`` §2.2a; fixtures committed under
``tests/fixtures/structured_output/``). Three facts shape the implementation:

* **``codex exec --output-schema <FILE>``** takes a JSON Schema *file*, so
  modelpass writes the caller's schema to a temporary file for the duration of the
  run and removes it afterwards. Nothing is written to the user's Codex home.
* **The answer arrives as the final ``agent_message`` item's ``text``** -- a JSON
  string, in the ordinary place a plain answer would be. There is no distinct
  item type and no field on ``turn.completed``. It still streams as a
  ``text_delta``, because that is genuinely what the runtime produced and
  suppressing it would hide part of what was paid for.
* **The schema goes to the Responses API's *strict* mode** (the 400 names
  ``response_format 'codex_output_schema'``), which is narrower than JSON
  Schema: every object needs ``additionalProperties: false`` and every property
  must be listed in ``required``. ``anthropic-sdk`` accepts the loose form. Both
  400s were captured live and are fixtures. modelpass does **not** refuse the run
  over it -- the vendor's rejection is precise and arrives before any generation
  -- but the preflight receipt names the issues, and
  :func:`modelpass.schema.to_openai_strict` converts a loose schema.
  **Re-checked for the app-server default (S7, 2026-08-31), and the honest
  answer is "unchanged, and not re-driven".** What moved is only the *delivery*:
  a per-turn ``TurnStartParams.outputSchema`` field instead of a temp file, and
  it was driven live -- a strict schema came back as
  ``{"lufs":-13.7,"verdict":"Pass"}`` with no file written. What was **not**
  driven is a *loose* schema on this path, so nobody has watched the strict
  subset be enforced here. The reason it is still predicted on both transports is
  structural rather than optimistic: the constraint belongs to the Responses
  API's ``response_format``, which is downstream of both, and the exec 400 named
  ``codex_output_schema`` rather than anything about ``--output-schema``. So
  :func:`_schema_notes` keeps warning on both and keeps calling it a prediction;
  it is one cheap 400 away from being confirmed on app-server and nobody has
  spent it.

Phase 9 (D14-D17) adds sessions, and here a session is a **thread**: on the exec
opt-out, one ``codex exec`` for the first turn and one ``codex exec resume
<id>`` for every turn after it; on the app-server default, one long-lived child
holding one thread (:class:`CodexAppServerSession`). The bullets below describe
the **exec** half and are unchanged by S7 -- they were re-verified against
**codex-cli 0.151.0** on
2026-08-30 -- the machine upgraded from 0.117.0 mid-Phase, and three facts
recorded that morning were already false by the afternoon, so every ``-c`` key
path this adapter sends is checked against the *installed* CLI rather than
against a note:

* **A thread is durable only after its first turn completes.** So
  :meth:`OpenAIAdapter.open_session` does local work only and the id appears
  after the first successful ``send`` (adapter contract, rule 7). A run killed
  before that leaves an id ``codex exec resume`` rejects outright.
* **Persisting means dropping ``--ephemeral``**, which
  :func:`modelpass.preflight.directives_for` adds unconditionally for every
  ``openai-sdk`` connection because it was written for the stateless call.
  Sessions never pass it. ``persist=False`` is refused before the adapter sees
  it (no ``ephemeral_multi_turn``), because on this runtime every turn is a
  separate process and continuation *is* the rollout file.
* **A missing thread is reported on stderr, not in the JSONL.** Driven live:
  ``codex exec resume <unknown-uuid> --json`` exits 1, writes **nothing** to
  stdout, and puts ``Error: thread/resume: thread/resume failed: no rollout
  found for thread id <id> (code -32600)`` on stderr. The stateless ``run()``
  path sends stderr to ``DEVNULL`` because it has no resume to fail; a session
  captures it, which is why sessions have their own spawn seam.
* **Resume by id is not scoped to a working directory.** Driven live on
  0.151.0: a thread recorded with one ``cwd`` resumed from an unrelated
  directory, found its rollout, and re-emitted ``thread.started`` carrying the
  same id. So ``project_folder`` is where a resumed turn *runs*, not where it is
  looked up -- which matters because ``bridge.resume_chat()`` defaults it to a
  fresh scratch directory. ``codex exec resume --all`` exists for the
  ``--last`` picker's cwd filtering and is not needed here.
* **``list_sessions`` is refused on the exec opt-out, not answered empty.**
  ``codex exec --help`` offers only ``resume [SESSION_ID]`` with ``--last`` /
  ``--all`` and ``codex resume`` is an interactive picker. An empty list would
  read as "this connection has no sessions", which is a different and false
  statement. ``thread/list`` lives on ``codex app-server`` and modelpass drives it there
  since S5 (2026-08-31), so the refusal names the option that works.
* **``codex exec fork <SESSION_ID> [PROMPT]`` exists** on 0.151.0 (checked in
  ``--help``, not driven). Nothing here uses it; ``sessions_fork`` stays
  ``unverified`` until somebody runs one.

**Switching Codex's own toolbelt off for a ``ChatSession``** is the part that
moves between releases, so the evidence is recorded per key. ``-c`` is a generic
dotted-path config override whose value is parsed as TOML, and **unknown keys
are silently ignored** -- ``-c totally_bogus_key_xyz=123`` exits 0 with no
warning -- so an override that does nothing is indistinguishable from one that
works unless it is read back. Two token-free readbacks exist and both were used:
``codex features list -c <override>`` prints the *effective* value of every
feature, and an invalid value on a known key returns a typed error naming the
key path.

==================================  ==========================================
``-c`` override                     Evidence on codex-cli 0.151.0 (2026-08-30)
==================================  ==========================================
``features.shell_tool=false``       readback flips ``true`` -> ``false``
``features.view_image=false``       readback flips ``true`` -> ``false``
``features.browser_use=false``      readback flips ``true`` -> ``false``
``features.image_generation=false`` readback flips ``true`` -> ``false``
``features.apps=false``             readback flips ``true`` -> ``false``
``web_search=disabled``             typed-validated, no readback surface
==================================  ==========================================

``web_search`` is the one sent without a readback: it is not a feature flag and
``features list`` does not print it, so the evidence is that
``-c web_search=__bogus__`` fails with *unknown variant ``__bogus__``, expected
one of ``disabled``, ``cached``, ``indexed``, ``live`` in ``web_search``* -- the
key path is live and ``disabled`` is a member of its enum. That enum gained
``indexed`` between 0.117.0 and 0.151.0 and its two ``web_search_*`` features
are now marked *deprecated*, so this is the key most likely to move next.

Two things that are **not** sent, both because they would be lies:

* ``tools.view_image`` -- validated on 0.117.0, and on 0.151.0 a bad value is
  silently ignored. The feature is spelled ``view_image`` in ``codex features
  list``, so ``features.view_image`` is the current path and the old one would
  now be a no-op that looks like a setting.
* ``features.unified_exec=false`` -- **accepted and ignored.** The readback
  stays ``true``, and so does ``--disable unified_exec``; the same is true of
  ``resize_all_images``, so some stable features are simply pinned on. This
  matters and is not papered over: ``unified_exec`` is an execution tool, it is
  on, and whether ``features.shell_tool=false`` removes every exec tool from
  the wire payload on 0.151.0 is **unverified**. What *was* verified, on
  0.117.0, is that disabling the shell tool dropped ~3.3k input tokens and
  produced zero ``command_execution`` items on a non-coding prompt, which is the
  tool definitions leaving the request rather than being hidden locally.

``system_prompt`` was **append-only** here, and that was the whole D16 story:
``codex exec`` has no system-prompt parameter, so a ``ChatSession`` carrying one
was refused by the core (no ``system_prompt_replace``) and a ``WorkerSession``'s
text was layered above the task on the **first** turn only. First is deliberate:
the thread keeps it thereafter, so it sits at the front of the cached prefix
instead of being restated every turn, which is exactly what D17 asks for on a
runtime whose only cache lever is prefix stability.

**S7 (2026-08-31) makes replace the default and keeps append reachable.** On the
app-server default a chat's prompt is ``ThreadStartParams.baseInstructions``,
which replaces; a ``WorkerSession`` still layers, because append is the
semantics a worker asked for and deleting the persona it was opened to keep
would be the silent kind of wrong. On ``options={"transport": "exec"}`` both
behave exactly as D16 described. The registry's ``system_prompt_replace`` cell
moved to ``supported`` in the same commit because it describes the default;
:meth:`OpenAIAdapter.support_for` narrows it back for a request that opts out.

**The caveat that belongs next to the yes**, because it is the thing a consumer
will otherwise over-promise: replacing the instructions does **not** by itself
erase the coding-agent character. ``baseInstructions`` names the base
instructions and nothing else -- tool definitions, environment context and
``AGENTS.md`` are untouched by it, which is why ~12.4k input tokens survive a
replacement that removed ~3,543 -- and a model reads its own identity off its
toolbelt as much as off its prompt. Switching the toolbelt off is a **separate
switch**, and since D23 **every chat-shaped call throws both** -- a
``ChatSession`` and a stateless :meth:`~modelpass.Bridge.chat` alike:
:func:`chat_tool_overrides` on the command line plus the replaced instructions.
Driven on app-server 2026-08-31, where the toolbelt half removed 5,621 prompt
tokens on its own (15,890 -> 10,269).

**What a caller can therefore promise, and what it still cannot.** After both
switches the model is running on the caller's persona with no shell, no
``apply_patch``, no ``web.run`` and no image generation that modelpass can reach.
What remains is environment context and whatever ``unified_exec`` retains --
see :func:`chat_tool_overrides` for why "the toolbelt modelpass can reach is off"
is the honest phrasing and "Codex has no execution tool left" is not. A
``WorkerSession`` deliberately keeps all of it; see
:func:`render_session_prompt` for what a worker is armed with and why.
"""

from __future__ import annotations

import base64
import binascii
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Callable, Container, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, ClassVar

from .._toml import dumps_inline
from ..capabilities import Capability, Support
from ..errors import (
    CapabilityNotSupported,
    InvalidSession,
    SessionClosed,
    SessionNotFound,
    VendorRunFailed,
)
from ..preflight import (
    DEFAULT_CACHE_FLOOR_TOKENS,
    AccountProfile,
    CacheEligibility,
    Receipt,
    check_launch_args,
)
from ..runtimes import Runtime
from ..schema import build_structured_event, openai_strict_issues
from ..types import (
    AgentEvent,
    AuthMode,
    Message,
    Role,
    SessionInfo,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    ThinkingEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    VendorEvent,
)
from ._mcp_result import flatten_mcp_result
from .base import Adapter, RunRequest, SessionRequest
from .codex_appserver import (
    THREAD_ITEMS_LIST_METHOD,
    THREAD_LIST_METHOD,
    THREAD_RESUME_METHOD,
    TURN_INTERRUPT_METHOD,
    AppServerClient,
    AppServerRequestFailed,
    AppServerSpawnFn,
    AppServerTransportClosed,
    AppServerTurn,
    CallerToolDispatcher,
    check_dynamic_tool_names,
    dynamic_tools_param,
    items_page,
    session_info_from_thread,
    thread_items_history,
    thread_items_list_params,
    thread_list_params,
    thread_resume_params,
    threads_from_list,
    turn_interrupt_params,
)

__all__ = [
    "DEFAULT_TRANSPORT",
    "INJECT_ITEMS_METHOD",
    "TRANSPORTS",
    "TRANSPORT_APP_SERVER",
    "TRANSPORT_EXEC",
    "CodexSession",
    "OpenAIAdapter",
    "agents_md_sources",
    "app_server_base_instructions",
    "app_server_history_items",
    "app_server_input_items",
    "app_server_turn_input",
    "chat_tool_overrides",
    "codex_home",
    "extract_error_detail",
    "inject_items_params",
    "interrupt_app_server_turn",
    "is_missing_thread",
    "is_quota_exhausted",
    "map_codex_event",
    "mcp_config_args",
    "read_configured_cli_path",
    "read_configured_model",
    "render_prompt",
    "render_session_prompt",
    "resolve_transport",
]

#: Import name of the 'openai-codex' PyPI distribution (pinned 2026-08-16).
_CODEX_MODULE = "openai_codex"

#: How long an ``account/read`` answer is reused for, in seconds.
#:
#: The Codex identity probe is not a cheap one: it starts a whole app-server
#: process and speaks JSON-RPC to it. Every run and every session open goes
#: through a preflight that wants the answer, so without this a chat call pays
#: for a process launch it did not ask for. It expires rather than being held
#: for the adapter's life because a user can ``codex login`` in another terminal
#: mid-process; :meth:`OpenAIAdapter.invalidate_identity_cache` drops it outright
#: for ``modelpass verify``.
_IDENTITY_CACHE_TTL_SECONDS = 60.0


# --- the configured model ------------------------------------------------------


def codex_home(env: Mapping[str, str], override: str | os.PathLike[str] | None = None) -> Path:
    """Where the Codex CLI keeps its own config on this machine.

    The same home the rest of this adapter talks about: ``~/.codex``, relocated
    by ``CODEX_HOME`` because the CLI itself honours it, so a user with a
    non-default layout gets a truthful receipt rather than a confident wrong one.
    Read from the *scrubbed* plan environment, like everything else the adapter
    consults. ``override`` exists so tests -- and a caller with an unusual setup
    -- can point this somewhere without a real home directory.
    """
    if override:
        return Path(override)
    configured = env.get("CODEX_HOME")
    return Path(configured) if configured else Path.home() / ".codex"


def _display_path(path: Path) -> str:
    """``~/.codex/config.toml`` where that is what it is, else the full path."""
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except (ValueError, OSError, RuntimeError):
        return str(path)


def _read_codex_config(home: Path) -> Mapping[str, Any]:
    """Parse ``<codex home>/config.toml``, or ``{}`` if it cannot be read.

    **Read-only, always.** modelpass never writes a vendor's configuration file;
    a chat call that edited someone's ``~/.codex/config.toml`` would be exactly
    the kind of surprise D2 exists to prevent, and this is the only place that
    file is opened at all.

    An unreadable, absent or malformed file is ``{}`` rather than an error: not
    being able to describe the runtime's own configuration is a normal outcome
    that the receipt reports honestly, and it must never be the reason a run
    does not happen.
    """
    try:
        with (home / "config.toml").open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, Mapping) else {}


def read_configured_model(home: Path) -> tuple[str | None, str]:
    """The ``model`` key from ``<codex home>/config.toml``, and where it came from."""
    data = _read_codex_config(home)
    model = data.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip(), f"from {_display_path(home / 'config.toml')}"
    return None, ""


def read_configured_cli_path(home: Path) -> str | None:
    """The ``CODEX_CLI_PATH`` key from ``<codex home>/config.toml``, or ``None``.

    Codex's own installer writes this when the newer Local\\OpenAI\\Codex build
    is installed alongside an npm shim, and the Desktop app runs *that* binary
    while ``PATH`` still points at the shim. The two are not interchangeable: a
    0.117.0 shim cannot drive a model a 0.150 build can (observed live
    2026-08-30, HTTP 400 at request validation).

    Read for the **receipt only**. modelpass does not run a binary because a config
    file it went looking for named one -- that is exactly the ambient influence
    D2 refuses, and it would mean a chat call silently changing which executable
    it launched. ``options={"codex_bin": ...}`` is the written-down override.
    """
    value = _read_codex_config(home).get("CODEX_CLI_PATH")
    return value.strip() if isinstance(value, str) and value.strip() else None


def agents_md_sources(home: Path, cwd: str | os.PathLike[str] | None = None) -> tuple[str, ...]:
    """Every ``AGENTS.md`` Codex will fold into this run's prompt.

    Codex populates ``turn_context.user_instructions`` from ``<codex home>/
    AGENTS.md`` on every run, and from ``AGENTS.md`` in the run's working
    directory and every ancestor of it (verified live 2026-08-30). Neither is
    scrubbable the way an environment variable or an MCP server is, and moving
    ``CODEX_HOME`` somewhere empty would orphan ``auth.json`` and break the
    subscription login. So they are *disclosed*: a caller is told what is in the
    prompt they did not write, rather than surprised by it later.

    Empty files are omitted, because an empty ``~/.codex/AGENTS.md`` is the
    common case and naming it would be a receipt describing an influence that is
    not happening -- the same rule ``passthrough`` follows in the preflight plan.
    ``cwd`` of ``None`` means the run gets a fresh temp directory, which has no
    chain worth walking.
    """
    candidates: list[Path] = [home / "AGENTS.md"]
    if cwd is not None:
        try:
            start = Path(cwd).resolve()
        except (OSError, ValueError):
            start = Path(cwd)
        candidates.extend(directory / "AGENTS.md" for directory in (start, *start.parents))
    found: list[str] = []
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 0:
                found.append(_display_path(path))
        except OSError:
            continue
    return tuple(found)


def _which_codex() -> str | None:
    """Locate a runnable Codex CLI.

    On Windows, npm installs an extensionless POSIX shim next to ``codex.cmd``;
    ``shutil.which("codex")`` returns the shim, which ``CreateProcess`` rejects
    with WinError 193. Prefer the executable extensions explicitly.

    This searches ``PATH`` only, which on a machine with two installs finds the
    npm shim and not the newer Local\\OpenAI\\Codex build. Deliberate: see
    :func:`read_configured_cli_path` for why the config's ``CODEX_CLI_PATH`` is
    reported rather than obeyed.
    """
    if os.name == "nt":
        for name in ("codex.cmd", "codex.exe", "codex.bat"):
            path = shutil.which(name)
            if path:
                return path
    return shutil.which("codex")

#: Event kinds that flip the run outcome to an error.
_FAILURE_KINDS = frozenset({"turn.failed", "error"})

#: Codex's own words for "the plan's allowance is gone", extracted from the
#: shipped ``codex.exe`` 0.117.0 on 2026-08-17 -- first-party evidence, since
#: OpenAI documents no error taxonomy for ``codex exec --json``. The binary
#: carries the error code in two spellings (the Rust-side ``usage_limit_exceeded``
#: and the app-server protocol's ``usageLimitExceeded``) plus the user-facing
#: message "You've hit your usage limit".
#:
#: **What is verified and what is not**, stated rather than blurred: these
#: strings exist in the binary, and HTTP 429 is known to reach modelpass because
#: the CLI double-encodes the server's error envelope into the ``message`` field
#: (characterized live in the Phase 1 experiment). What has *not* been observed
#: is one of these actually arriving in a ``turn.failed`` / ``error`` payload,
#: because doing that on purpose means exhausting a real ChatGPT plan. So the
#: detection is deliberately narrow and everything it does not recognize falls
#: back to ``error`` -- a misread quota as an error is a worse-labelled failure,
#: while a misread error as a quota stop would be a *silent* one, and could
#: trigger a configured failover onto metered billing for no reason.
#: TODO(live): capture a real exhausted-allowance payload and replace this with
#: a recorded fixture. Outstanding since the 2026-08-16 tools pass.
_QUOTA_MARKERS = (
    "usage_limit_exceeded",
    "usagelimitexceeded",
    "hit your usage limit",
)


def is_quota_exhausted(payload: Mapping[str, Any], detail: str) -> bool:
    """Whether a Codex failure payload is "the plan ran out" rather than a fault.

    Checked against the double-parsed detail *and* the raw payload, because the
    status code and the error code arrive by different routes: the former inside
    the JSON-encoded string the CLI puts in ``message``, the latter as a field
    name somewhere in the envelope.
    """
    haystack = f"{detail} {json.dumps(dict(payload), default=str)}".lower()
    if any(marker in haystack for marker in _QUOTA_MARKERS):
        return True
    # "429" alone is too loose to match on free-text; anchor it to the shapes the
    # envelope actually uses so a token count that happens to contain 429 does
    # not get read as a rate limit.
    return any(
        shape in haystack
        for shape in ('"status": 429', '"status":429', "status 429", "429:", "http 429")
    )

_ROLE_LABELS = {Role.SYSTEM: "System", Role.USER: "User", Role.ASSISTANT: "Assistant"}

_TRANSCRIPT_PREAMBLE = (
    "The following is a conversation transcript. Reply as the assistant to the "
    "final user message. Output only the assistant reply, nothing else."
)


def render_prompt(messages: tuple[Message, ...]) -> str:
    """Serialize the D7 message list into a single Codex prompt.

    A lone user message passes through verbatim, and so does a system-plus-user
    pair: that is a *single shot with instructions*, not a conversation, and
    wrapping it in a transcript envelope would ask the model to continue a
    dialogue that never happened. It is layered the way
    :func:`render_session_prompt` layers a session's first turn -- the same
    shape for the same reason, since Codex has no system-prompt parameter and
    the prompt is the only place instructions can go.

    Anything richer becomes a labelled transcript with an explicit continuation
    instruction: Codex has no native multi-message input in ``exec`` mode, so a
    real conversation has nowhere else to live.
    """
    if len(messages) == 1 and messages[0].role is Role.USER:
        return messages[0].flat_text
    if (
        len(messages) == 2
        and messages[0].role is Role.SYSTEM
        and messages[1].role is Role.USER
    ):
        return render_session_prompt(messages[0].flat_text, messages[1].flat_text)
    lines = [_TRANSCRIPT_PREAMBLE, ""]
    lines.extend(f"{_ROLE_LABELS[m.role]}: {m.flat_text}" for m in messages)
    return "\n".join(lines)


def render_session_prompt(system_prompt: str | None, message: str) -> str:
    """Layer a session's system prompt above the first turn's task (D15, D16).

    On ``codex exec`` there is no system-prompt parameter, so the only place
    instructions can go is the prompt, above the task and on top of the
    hardcoded ``"You are a deployed coding agent"`` persona. That is append
    semantics, and on **that transport** it is all this runtime has -- which is
    why a ``ChatSession`` carrying a system prompt is still refused under an
    explicit ``options={"transport": "exec"}``.

    **It is no longer all the runtime has.** Since S7 (2026-08-31) the default
    transport is ``codex app-server``, which carries
    ``ThreadStartParams.baseInstructions`` and *replaces*. A ``ChatSession``
    opens there and takes that path
    (:func:`_session_thread_start_params`). This function is still where a
    **worker's** prompt goes on both transports, because a worker appends by
    design: it keeps Codex's persona and adds to it, and the persona is what
    tells the model how to drive the ``exec`` toolbelt a worker exists to use.
    Deleting it while leaving the belt attached is the trade
    :attr:`~modelpass.adapters.base.SessionRequest.appends_system_prompt`
    declines to make on the caller's behalf.

    Applied on the **first** turn only. The thread carries it from then on, so
    restating it every turn would move a stable prefix and pay for it twice;
    prefix stability is the only cache lever this runtime has (D17). No prompt
    means the message passes through verbatim, the same way a lone user message
    does in :func:`render_prompt`.
    """
    if not system_prompt:
        return message
    return f"{_ROLE_LABELS[Role.SYSTEM]}: {system_prompt}\n\n{message}"


# --- transport selection (S4) ------------------------------------------------------


#: ``codex exec --json``, one process per turn. The Phase 4 transport, and the
#: **opt-out** since S7 (2026-08-31): fully reachable, unchanged, and selected
#: with ``options={"transport": "exec"}``. Not deprecated and not on a removal
#: path -- it is what carries per-run MCP servers, and it is the one line a
#: caller who wants their system prompt *layered on top of* Codex's persona
#: rather than replacing it needs to type.
TRANSPORT_EXEC = "exec"

#: ``codex app-server --listen stdio://``, JSON-RPC over one long-lived child.
#: **The default since S7** (2026-08-31): what ``openai-sdk`` runs when a caller
#: selects nothing.
TRANSPORT_APP_SERVER = "app-server"

TRANSPORTS = (TRANSPORT_EXEC, TRANSPORT_APP_SERVER)

#: What a call with no ``transport`` option gets. **Flipped to app-server in S7**
#: (2026-08-31), which is the commit the capability rows moved in: a row
#: describes the default transport, so the row and the default move together or
#: the registry is lying about the runtime a caller actually gets.
DEFAULT_TRANSPORT = TRANSPORT_APP_SERVER


def resolve_transport(options: Mapping[str, Any]) -> str:
    """Which Codex transport this call runs on. ``app-server`` unless asked otherwise.

    **The default flipped in S7 (2026-08-31)** and this is the function that
    says so. ``codex app-server`` is what ``openai-sdk`` runs when a caller
    selects nothing; ``options={"transport": "exec"}`` is the opt-out, and the
    exec path is untouched -- same argv, same stdin prompt, same event mapping,
    same tests. This is a change of default, **not a removal**: every sentence
    below about exec still describes a transport modelpass drives today.

    **What flipping costs an existing caller, said here because this is where
    the choice is made.** A ``system_prompt`` on a stateless ``chat()`` or a
    ``ChatSession`` becomes ``ThreadStartParams.baseInstructions``, which
    *replaces* Codex's built-in coding-agent persona where exec's ``System: ``
    line layered on top of it. Same request, materially different run -- and the
    escape is this one option. ``mcp_servers=`` is the other: it is an exec
    capability with no driven equivalent on this path, so a run carrying one is
    refused with the same one-line pointer rather than run without the servers.

    **An option, not a connection field** (S4, 2026-08-31; re-decided in S7).
    ``Connection`` has no free-form options bag -- every field on it is typed,
    parsed and round-tripped through the config store -- so a connection-level
    ``transport`` would mean schema work, a store migration and a
    serializable-surface change. S4 said the field would earn its place when the
    default flipped; flipping it is what *removed* the argument. The value a
    connection would carry is now the value it gets for free, so the only thing
    a stored field would buy is pinning a connection to the opt-out, which
    ``options=`` already does per call and which no consumer has asked for. It
    stays a written-down option, next to ``codex_bin``.

    An unrecognized value is **refused rather than defaulted**. Silently running
    the default for a caller who typed ``"appserver"`` is precisely the
    ignored-override failure the bridge's unknown-option note exists to prevent
    -- and here the two transports differ in what a run *does*, not only in how
    it is spelled.
    """
    requested = options.get("transport")
    if requested is None:
        return DEFAULT_TRANSPORT
    if isinstance(requested, str) and requested in TRANSPORTS:
        return requested
    raise CapabilityNotSupported(
        Runtime.OPENAI_SDK.value,
        "transport",
        detail=(
            f"{requested!r} is not a Codex transport modelpass drives; pass "
            f"{TRANSPORT_EXEC!r} or {TRANSPORT_APP_SERVER!r}. Refused rather than "
            "defaulted: the two transports render a system prompt and a "
            f"conversation differently, so a typo that fell back to "
            f"{DEFAULT_TRANSPORT!r} would change the run and say nothing"
        ),
    )


def app_server_base_instructions(messages: tuple[Message, ...]) -> str | None:
    """The system-role content for ``ThreadStartParams.baseInstructions``.

    **This is the fact that retires the exec workaround.**
    :func:`render_session_prompt`'s docstring says Codex has no system-prompt
    parameter; that is true of ``codex exec`` and false of the vendor, which has
    carried ``baseInstructions`` and ``developerInstructions`` on
    ``ThreadStartParams`` all along (verified 2026-08-31 -- present in the plan
    of record's live findings and in the shipped ``codex.exe`` 0.151.0's own
    parameter tables). So on this transport a system message is a parameter
    rather than a ``System: `` line glued to the front of the task.

    **Replace, not append, and that is a real behavioural difference** the
    receipt names rather than hides: ``baseInstructions`` has replace semantics,
    so a caller's system prompt goes in *instead of* Codex's built-in "deployed
    coding agent" persona, where the exec transport layers it on top. Same
    request, different runs. Naming it is the point -- this is the difference
    between the two transports a caller can actually feel.

    ``None`` when there is no system message, which leaves the vendor's own
    base instructions in place. Multiple system messages are joined rather than
    last-wins: modelpass did not decide the caller meant only one of them.
    """
    parts = [m.flat_text for m in messages if m.role is Role.SYSTEM and m.flat_text]
    return "\n\n".join(parts) if parts else None


def app_server_input_items(messages: tuple[Message, ...]) -> list[dict[str, str]]:
    """The D7 message list as ``TurnStartParams.input`` items. **The fallback.**

    One item per non-system message. ``{"type": "text", "text": ...}`` is the
    text variant of the protocol's input item and was **driven live** on
    2026-08-31; the sibling variants (``image``, ``localImage``, ``audio``,
    ``localAudio``, ``skill``, ``mention``) are read from the shipped
    ``codex.exe`` 0.151.0's own enum and modelpass sends none of them.

    **What this replaces**: ``_TRANSCRIPT_PREAMBLE``. On ``codex exec`` a
    conversation has to be flattened into one prompt string with an instruction
    telling the model it is reading a transcript, because exec takes one prompt
    and nothing else. Here each message is its own item, so the envelope goes.

    **What it cannot replace**: an input item has no ``role``. Every variant of
    the enum is user-authored content -- the wire has no assistant item to put a
    previous answer in -- so the ``Assistant: `` labels stay on the turns that
    need them. Dropping them along with the preamble would leave a model unable
    to tell its own past replies from the user's, which is a worse conversation
    than a labelled one. A lone user message is unlabelled, exactly as
    :func:`render_prompt` leaves it.

    **Superseded as the normal path on 2026-08-31, kept as the fallback.**
    ``thread/inject_items`` puts prior turns into the thread's own history as
    real role-bearing items and needs no labels at all
    (:func:`app_server_history_items`), so the ordinary run sends only the
    current message (:func:`app_server_turn_input`). This function is what
    :meth:`OpenAIAdapter._run_app_server` falls back to when the server refuses
    that method -- see :func:`inject_items_params` for why a build might.
    """
    conversation = [m for m in messages if m.role is not Role.SYSTEM]
    if len(conversation) == 1 and conversation[0].role is Role.USER:
        return [{"type": "text", "text": conversation[0].text}]
    return [
        {"type": "text", "text": f"{_ROLE_LABELS[m.role]}: {m.flat_text}"}
        for m in conversation
    ]


def _conversation(messages: tuple[Message, ...]) -> list[Message]:
    """Every non-system message, in order. Systems go to ``baseInstructions``."""
    return [m for m in messages if m.role is not Role.SYSTEM]


def app_server_turn_input(messages: tuple[Message, ...]) -> list[dict[str, str]]:
    """``TurnStartParams.input``: the **current** message and nothing else.

    The last non-system message is the turn being taken now -- ``Bridge.chat``
    assembles the list as instructions, then history, then the message -- and it
    is the one thing that must **not** be injected. Injected items are appended
    to the thread's history *before* the turn runs, so a current message put
    there would be part of the transcript the model is answering *from* rather
    than the question it is answering. The live 2026-08-31 run drew exactly that
    line: an injected user/assistant pair, then a fresh question on
    ``turn/start``.

    Unlabelled, because with the history injected there is nothing left to
    disambiguate: one item, from the user, which is what an input item already
    means.
    """
    conversation = _conversation(messages)
    if not conversation:
        return []
    return [{"type": "text", "text": conversation[-1].text}]


#: The content-type each role's text rides in, and the asymmetry is **the
#: Responses API's own, not a typo**: a user message carries ``input_text`` and
#: an assistant message carries ``output_text``, because one is what went in and
#: the other is what came out. Both shapes were driven live on 2026-08-31
#: (fixture: ``tests/fixtures/appserver/live-inject-items-2026-08-31.jsonl``).
_ITEM_CONTENT_TYPE = {Role.USER: "input_text", Role.ASSISTANT: "output_text"}


def app_server_history_items(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    """Prior turns as **raw Responses API items** for ``thread/inject_items``.

    Every non-system message except the last, which is the current turn's input
    (:func:`app_server_turn_input`). Each becomes
    ``{"type": "message", "role": ..., "content": [{"type": ..., "text": ...}]}``
    -- a real role-bearing item, which is the thing ``TurnStartParams.input``
    does not have and the whole reason S4 had to write ``Assistant: `` in front
    of the model's own past answers.

    **Driven live 2026-08-31**: an injected user/assistant exchange was answered
    from correctly, with no labels anywhere, on an **ephemeral** thread -- which
    is the kind the stateless call creates, so this is verified on the path that
    uses it rather than on a neighbouring one.

    Empty when there is nothing prior, and the caller sends no
    ``thread/inject_items`` at all in that case: a fresh single-turn call is
    byte-for-byte the run S4 shipped, minus the labels it never needed.
    """
    conversation = _conversation(messages)
    return [
        {
            "type": "message",
            "role": m.role.value,
            "content": [
                {"type": _ITEM_CONTENT_TYPE[m.role], "text": m.flat_text}
            ],
        }
        for m in conversation[:-1]
    ]


#: The method that seeds a thread's model-visible history, **snake_case**.
#:
#: A trap, and it cost a live run to find: the ``openai-codex`` SDK's generated
#: types imply ``thread/injectItems``, and the camelCase spelling is rejected as
#: an unknown method variant. The wire name is this one (driven 2026-08-31
#: against codex 0.151.0-alpha.7.1). Related and equally worth knowing: an
#: unknown method's error carries the build's **full supported-method list**,
#: which is a token-free capability probe for any future build and is how both
#: this and the response shape were found.
INJECT_ITEMS_METHOD = "thread/inject_items"


def inject_items_params(thread_id: str, messages: tuple[Message, ...]) -> dict[str, Any]:
    """Params for :data:`INJECT_ITEMS_METHOD`: ``{threadId, items}``.

    The response is ``{}`` -- **success is the absence of an error**, so nothing
    reads a payload back off this call (driven 2026-08-31). That is also why the
    failure mode is worth designing for rather than assuming away: the surface
    moves (the snake/camel spelling above is the proof), and a build that does
    not carry the method answers with a JSON-RPC error rather than doing nothing
    visible.
    """
    return {"threadId": thread_id, "items": app_server_history_items(messages)}


def _thread_start_params(request: RunRequest, cwd: str) -> dict[str, Any]:
    """``ThreadStartParams`` for one stateless call.

    Field names are the vendor's own. ``cwd``, ``model``, ``baseInstructions``
    and ``ephemeral`` are all members of the thread-parameter family in the
    shipped ``codex.exe`` 0.151.0 (read 2026-08-31 from the binary's own
    serialization tables, the same first-party source ``_QUOTA_MARKERS`` came
    from), and the live capture's ``thread/start`` response echoes ``cwd`` and
    ``model`` back. What was **driven**, rather than read, is ``cwd``: the
    2026-08-31 capture started a thread with it and got a working thread.

    ``ephemeral`` mirrors the ``--ephemeral`` directive the preflight plan adds
    for every stateless ``openai-sdk`` call, so the receipt's directive list and
    what the transport was actually told stay one statement. Sessions do not come
    through here: :func:`_session_thread_start_params` builds their thread, omits
    ``ephemeral`` so it persists, and :func:`thread_resume_params` picks one back
    up.

    ``dynamicTools`` (S6) is the caller's ``tools=``, and the key is **omitted
    entirely** when there are none rather than sent as ``[]`` --
    :func:`~modelpass.adapters.codex_appserver.dynamic_tools_param` records why the
    two are not the same statement.
    """
    params: dict[str, Any] = {"cwd": cwd}
    if any(directive.name == "--ephemeral" for directive in request.plan.directives):
        params["ephemeral"] = True
    instructions = app_server_base_instructions(request.messages)
    if instructions is not None:
        params["baseInstructions"] = instructions
    if request.model:
        params["model"] = request.model
    dynamic_tools = dynamic_tools_param(request.tools)
    if dynamic_tools is not None:
        params["dynamicTools"] = dynamic_tools
    return params


def _turn_start_params(
    request: RunRequest, thread_id: str, *, history_injected: bool = True
) -> dict[str, Any]:
    """``TurnStartParams`` for one turn on an already-started thread.

    ``outputSchema`` is a field on this params object (read from the shipped
    ``codex.exe`` 0.151.0's ``TurnStartParams`` table on 2026-08-31, alongside
    ``model``, ``effort`` and ``cwd``), and being **per turn** is what makes it
    better than exec's ``--output-schema <FILE>``: no temp file is written, none
    has to be removed in a ``finally``, and nothing lands near the user's Codex
    home. The schema is passed through verbatim -- modelpass predicts the strict-
    subset rejection on the receipt and never rewrites what the caller asked for.

    ``history_injected`` says which of the two input shapes this turn gets. When
    the thread's history was seeded with ``thread/inject_items``, the input is
    the current message alone; when that call was refused, it is S4's labelled
    transcript, because a conversation the model cannot read is worse than one
    it has to infer roles in.
    """
    params: dict[str, Any] = {
        "threadId": thread_id,
        "input": (
            app_server_turn_input(request.messages)
            if history_injected
            else app_server_input_items(request.messages)
        ),
    }
    if request.schema is not None:
        params["outputSchema"] = dict(request.schema)
    return params


def _started_thread_id(result: Mapping[str, Any]) -> str:
    """The thread id out of a ``thread/start`` response.

    ``{"thread": {"id": ...}}`` in the live capture (2026-08-31). A response
    without one is a protocol change, not a turn that can be started: every
    subsequent call is addressed by this id, so it fails here with the shape it
    got rather than sending ``turn/start`` to nothing.
    """
    thread = result.get("thread")
    thread_id = thread.get("id") if isinstance(thread, Mapping) else None
    if isinstance(thread_id, str) and thread_id.strip():
        return thread_id.strip()
    raise VendorRunFailed(
        "codex app-server answered thread/start without a thread id: "
        f"{json.dumps(dict(result), default=str)[:300]}"
    )


def interrupt_app_server_turn(
    client: Any, thread_id: str | None, turn_id: str | None, *, timeout: float = 5.0
) -> bool:
    """Best-effort ``turn/interrupt``. Returns whether the server accepted it.

    **The method is driven; modelpass's use of it is still best-effort.**
    ``TurnInterruptParams`` carries ``threadId`` and ``turnId``, both required,
    and ``TurnInterruptResponse`` is an empty object -- schema-sourced 2026-08-31
    from the shipped binary's own generator (``codex app-server
    generate-json-schema``). And it **works**: driven live the same day, the
    interrupt returned ``{}`` and the turn ended with ``status: "interrupted"``
    (``tests/fixtures/appserver/live-capability-evidence-2026-08-31.jsonl``,
    section ``graceful_cancel``).

    This is still written to be harmless when it is wrong -- no ids means no
    attempt, any failure is swallowed, and the caller terminates the child either
    way -- because what has *not* been driven is the part modelpass would have to
    rely on: nothing here waits for the ``turn/completed`` the interrupt produces,
    and whether the thread stays usable for a **next** turn afterwards is
    unchecked. That is the whole difference between a graceful cancel and a
    polite kill, so the capability registry still claims nothing;
    :meth:`CodexAppServerSession._abandon` records what is left to check.

    The short timeout is the point: an interrupt that has not been answered in
    five seconds is not the thing standing between the caller and a cancelled
    run. The kill behind it always works.
    """
    if not thread_id or not turn_id:
        return False
    try:
        client.request(
            TURN_INTERRUPT_METHOD,
            turn_interrupt_params(thread_id, turn_id),
            timeout=timeout,
        )
    except (VendorRunFailed, OSError, ValueError):
        return False
    return True


def extract_error_detail(payload: Mapping[str, Any]) -> str:
    """Pull a human-readable reason out of an ``error`` / ``turn.failed`` payload.

    The CLI wraps the server's JSON error envelope in a *string*:
    ``{"type": "error", "message": "{\\"status\\": 400, \\"error\\": {...}}"}``.
    """
    raw: Any = payload.get("message")
    if raw is None:
        err = payload.get("error")
        if isinstance(err, Mapping):
            raw = err.get("message")
    if not isinstance(raw, str):
        return json.dumps(dict(payload))[:500]
    try:
        inner = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(inner, Mapping):
        err = inner.get("error")
        if isinstance(err, Mapping) and isinstance(err.get("message"), str):
            status = inner.get("status")
            return f"{status}: {err['message']}" if status else err["message"]
    return raw


#: Item types that report a runtime-executed tool, and therefore map onto the
#: D12 tool events. Both are documented as emitting ``item.started`` *and*
#: ``item.completed``, which is why modelpass maps started -> ``tool_call`` and
#: completed -> ``tool_result`` rather than synthesizing a pair from one event:
#: doing the latter would double-report every call that reached completion.
_TOOL_ITEM_TYPES = frozenset({"mcp_tool_call", "command_execution"})

#: Server name reported for Codex's own built-in tools, which modelpass cannot
#: switch off on this runtime. Distinct from any MCP server name so a caller can
#: tell "the runtime ran its shell" from "the runtime called my server".
_BUILTIN_SERVER = "codex"

#: Item ``status`` values that mean the tool did not do what was asked.
#: ``failed`` is documented; ``declined`` is not, and was captured live on
#: 2026-08-30 when the Windows sandbox refused a command. It is listed here
#: rather than checked at one call site because ``status`` is one field shared by
#: both tool item types: a declined MCP call is no more a success than a
#: declined shell command, and reading it as one is silent, which is the failure
#: mode this library exists to avoid.
_ERROR_ITEM_STATUSES = frozenset({"failed", "declined"})


def _tool_call_from_item(item: Mapping[str, Any]) -> AgentEvent | None:
    """Build a ``tool_call`` from an ``item.started`` payload, or ``None``.

    Returning ``None`` on an unexpected shape is the point: the ``mcp_tool_call``
    field names are still third-party-sourced (see the module docstring), so an
    item that does not look the way it was documented falls back to
    ``vendor_event`` rather than being reported as a half-populated tool call.
    The defensive read stays on ``command_execution`` too now that its fields are
    confirmed -- a runtime is free to change them again.
    """
    item_id = item.get("id")
    item_id = item_id if isinstance(item_id, str) else ""

    if item.get("type") == "mcp_tool_call":
        tool = item.get("tool")
        server = item.get("server")
        if not isinstance(tool, str) or not tool:
            return None
        arguments = item.get("arguments")
        return ToolCallEvent(
            name=tool,
            arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
            id=item_id,
            server=server if isinstance(server, str) else "",
        )

    command = item.get("command")
    if not isinstance(command, str):
        return None
    return ToolCallEvent(
        name="command_execution",
        arguments={"command": command},
        id=item_id,
        server=_BUILTIN_SERVER,
    )


def _tool_result_from_item(item: Mapping[str, Any]) -> AgentEvent | None:
    """Build a ``tool_result`` from an ``item.completed`` payload, or ``None``."""
    item_id = item.get("id")
    item_id = item_id if isinstance(item_id, str) else ""
    status = item.get("status")

    if item.get("type") == "mcp_tool_call":
        tool = item.get("tool")
        if not isinstance(tool, str) or not tool:
            return None
        error = item.get("error")
        is_error = status in _ERROR_ITEM_STATUSES or bool(error)
        payload = error if is_error and error is not None else item.get("result")
        return ToolResultEvent(
            id=item_id,
            name=tool,
            content=flatten_mcp_result(payload),
            is_error=is_error,
        )

    if "aggregated_output" not in item and "exit_code" not in item:
        return None
    exit_code = item.get("exit_code")
    output = item.get("aggregated_output")
    return ToolResultEvent(
        id=item_id,
        name="command_execution",
        content=output if isinstance(output, str) else "",
        # A missing exit code is not evidence of success, but neither is it
        # evidence of failure; the reported status is the better signal -- and
        # the one that carries "declined", which the live 2026-08-30 capture
        # paired with exit_code -1 but which the CLI leaves null while a command
        # is still only in_progress.
        is_error=(
            status in _ERROR_ITEM_STATUSES
            or bool(isinstance(exit_code, int) and exit_code != 0)
        ),
    )


def map_codex_event(payload: Mapping[str, Any]) -> tuple[AgentEvent, ...]:
    """Map one Codex JSONL payload to normalized events. Pure; offline-testable.

    Anything unrecognized passes through as a ``vendor_event`` (adapter rule 3);
    nothing is dropped.

    **Why ``cached_input_tokens`` is subtracted out of ``input_tokens`` below.**
    It reads like an arithmetic bug and it is the opposite of one: the two
    vendors modelpass drives genuinely disagree about what this field counts, and
    that was measured on 2026-08-31 rather than assumed.

    * **Codex nests it.** A live ``turn.completed`` on this transport reported
      ``{input_tokens 14228, cached_input_tokens 9984,
      cache_write_input_tokens 0, output_tokens 5}``, and the app-server's
      ``last`` breakdown reported ``{inputTokens 13123, cachedInputTokens 12032,
      outputTokens 21, totalTokens 13144}`` -- the vendor's *own* total is
      ``input + output``, so ``cached`` is a **subset** of ``input``.
    * **Anthropic counts them side by side.** A live row there reads
      ``input_tokens 2`` beside ``cache_read_input_tokens 1901``, which is not a
      shape a nested field can produce.

    :class:`~modelpass.types.TokenUsage` keeps the parallel convention -- it is the
    one that can express both readings, and Anthropic satisfies it natively -- so
    the correction belongs in the *Codex* mappers rather than in a total that
    would have to mean different things per runtime. Until 2026-08-31 it was
    missing and ``total_tokens`` counted the cache read twice: every Codex run in
    ``~/.modelpass/runs.jsonl`` was inflated by roughly 65% (22,655 reported
    against 13,695 spent), which corrupted the run log *and* fed
    ``stop_at_tokens``, so guards fired early.
    :func:`modelpass.adapters.codex_appserver.token_usage_from_breakdown` carries
    the same correction for the app-server transport; the two move together.

    ``cache_write_input_tokens`` is read here as of the same date. The wire has
    carried it all along (seen live in the 2026-08-31 capture) and this mapper
    ignored it, so a cold prefix reported as free on the exec transport.
    """
    kind = payload.get("type")
    if kind in ("item.started", "item.completed"):
        item = payload.get("item")
        item = item if isinstance(item, Mapping) else {}
        item_type = item.get("type")
        text = item.get("text")
        if kind == "item.completed":
            if item_type == "agent_message" and isinstance(text, str):
                return (TextDeltaEvent(text=text),)
            if item_type == "reasoning" and isinstance(text, str):
                return (ThinkingEvent(text=text),)
        if item_type in _TOOL_ITEM_TYPES:
            build = _tool_call_from_item if kind == "item.started" else _tool_result_from_item
            event = build(item)
            if event is not None:
                return (event,)
        name = f"{kind}:{item_type}" if item_type else str(kind)
        return (VendorEvent(runtime=Runtime.OPENAI_SDK, name=name, data=dict(payload)),)
    if kind == "turn.completed":
        usage = payload.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        wire_input = int(usage.get("input_tokens", 0) or 0)
        cached = int(usage.get("cached_input_tokens", 0) or 0)
        return (
            UsageEvent(
                usage=TokenUsage(
                    # Codex's input is inclusive of its cache read; modelpass's
                    # convention counts them side by side (docstring above).
                    # Clamped at zero rather than trusted: if a release ever
                    # reports cached > input, a negative count here would flow
                    # into a run record and *subtract* from a guard's running
                    # total, which is a worse failure than an over-count.
                    input_tokens=max(0, wire_input - cached),
                    output_tokens=int(usage.get("output_tokens", 0) or 0),
                    cached_input_tokens=cached,
                    cache_write_tokens=int(usage.get("cache_write_input_tokens", 0) or 0),
                )
            ),
        )
    name = str(kind) if kind else "unknown"
    return (VendorEvent(runtime=Runtime.OPENAI_SDK, name=name, data=dict(payload)),)


class _TurnOutcome:
    """What one drained ``codex exec`` process turned out to be.

    A mutable companion to :func:`_drain_codex_jsonl`, because a generator can
    either yield events or return a summary and this needs both. It exists so
    the stateless ``run()`` and a session turn share one reading of the JSONL
    stream: two copies of "which payloads mean the run failed" would drift, and
    the half that drifts is the half that decides how a run is reported.
    """

    __slots__ = ("agent_messages", "reason", "status", "thread_id", "usage")

    def __init__(self) -> None:
        self.status: TerminalStatus = TerminalStatus.OK
        self.reason: str | None = None
        self.usage: TokenUsage = TokenUsage()
        #: **Every** ``agent_message`` in the turn, in arrival order. A live run
        #: on 2026-08-30 emitted two in one turn -- a full answer, then a shorter
        #: restatement -- so this is a list rather than a slot that the second
        #: one would silently overwrite. The event stream already accumulates:
        #: :func:`map_codex_event` turns each completed ``agent_message`` into
        #: its own ``text_delta``, so a caller concatenating deltas sees both.
        self.agent_messages: list[str] = []
        self.thread_id: str | None = None

    @property
    def final_answer(self) -> str | None:
        """The structured answer: the **last** ``agent_message``, not the join.

        The one place where last-wins is right rather than lossy. With
        ``--output-schema`` the answer arrives as the final ``agent_message``
        item's text -- a JSON string in the ordinary place a plain answer would
        be (verified live 2026-08-17) -- and concatenating two of them would
        produce something that parses as neither. What is *not* verified is a
        schema-bound turn emitting two messages at all; if one ever does, this
        is the line to revisit, which is why the choice is written down here
        instead of being a ``[-1]`` at the call site.
        """
        return self.agent_messages[-1] if self.agent_messages else None


def _drain_codex_jsonl(
    stdout: IO[str] | None, outcome: _TurnOutcome
) -> Iterator[AgentEvent]:
    """Read the CLI's JSONL stream into normalized events, recording the outcome.

    Everything that is the same for a one-shot and for a session turn: a line
    that is not JSON is reported rather than dropped (rule 3), a failure payload
    sets the status once, usage accumulates, and the thread id is picked up from
    ``thread.started`` -- which is the only place a session id ever comes from.
    """
    for raw in stdout or ():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            payload = None
        if not isinstance(payload, Mapping):
            yield VendorEvent(
                runtime=Runtime.OPENAI_SDK, name="stdout", data={"line": line[:2000]}
            )
            continue
        if payload.get("type") in _FAILURE_KINDS:
            reason = extract_error_detail(payload)
            # A spent allowance is a clean stop, not a failure: nothing went
            # wrong, the plan ran out (D4). It is also the signal a configured
            # failover keys off, which is why the detection errs narrow -- see
            # _QUOTA_MARKERS.
            if is_quota_exhausted(payload, reason):
                outcome.status = TerminalStatus.QUOTA_EXHAUSTED
                outcome.reason = f"ChatGPT plan allowance exhausted: {reason}"
            else:
                outcome.status = TerminalStatus.ERROR
                outcome.reason = reason
        if payload.get("type") == "thread.started":
            thread_id = payload.get("thread_id")
            if isinstance(thread_id, str) and thread_id.strip():
                outcome.thread_id = thread_id.strip()
        text = _agent_message_text(payload)
        if text is not None:
            outcome.agent_messages.append(text)
        for event in map_codex_event(payload):
            if isinstance(event, UsageEvent):
                outcome.usage = outcome.usage + event.usage
            yield event


def mcp_config_args(servers: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Per-invocation MCP server declarations as ``-c`` override arguments.

    ``codex exec -c <key>=<value>`` parses the value portion as TOML, so each
    server becomes one dotted-path assignment of an inline table::

        -c mcp_servers.docs={command = "npx", args = ["-y", "docs-mcp"]}

    Verified live against codex-cli 0.117.0 on 2026-08-16 through a token-free
    ``codex mcp list`` run (``docs/api-and-runtimes.md`` §2.2a). This writes nothing to the
    user's ``~/.codex/config.toml`` -- an important property, because a chat
    call that permanently added an MCP server to someone's CLI would be exactly
    the kind of surprise D2 exists to prevent.

    It also *merges* rather than replaces: the user's configured servers stay
    reachable. That limitation is real and is reported through the capability
    registry note rather than papered over here.

    Server names are validated because they are interpolated into a dotted TOML
    path, where a name containing ``.``, ``=`` or a quote would silently change
    which key is being assigned.
    """
    args: list[str] = []
    for name, config in servers.items():
        if not name or not all(ch.isalnum() or ch in "_-" for ch in name):
            raise ValueError(
                f"MCP server name {name!r} is not usable in a Codex '-c' override: "
                "use letters, digits, '_' and '-' only"
            )
        if not isinstance(config, Mapping):
            raise TypeError(
                f"MCP server {name!r} config must be a mapping, got "
                f"{type(config).__name__}"
            )
        args.extend(["-c", f"mcp_servers.{name}={dumps_inline(config)}"])
    return args


def _agent_message_text(payload: Mapping[str, Any]) -> str | None:
    """The text of a completed ``agent_message`` item, or ``None``.

    With ``--output-schema`` this text *is* the structured answer -- verified
    live 2026-08-17, and the reason there is no cleverer place to look: the CLI
    emits no distinct item type and puts nothing on ``turn.completed``.
    """
    if payload.get("type") != "item.completed":
        return None
    item = payload.get("item")
    if not isinstance(item, Mapping) or item.get("type") != "agent_message":
        return None
    text = item.get("text")
    return text if isinstance(text, str) else None


def _write_schema_file(schema: Mapping[str, Any]) -> str:
    """Write a schema to a private temp file for one run. Returns its path."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="modelpass-schema-",
        encoding="utf-8",
        delete=False,
    )
    try:
        json.dump(dict(schema), handle)
    finally:
        handle.close()
    return handle.name


def _remove_schema_file(path: str | None) -> None:
    """Remove the per-run schema file. Never the reason a run fails."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    """What the Codex ChatGPT login store looks like -- metadata only.

    Mirrors ``modelpass.adapters.anthropic.CredentialStatus``: reads
    ``auth.json`` for the stored access token's expiry and whether a
    refresh_token is present, never the token values themselves. The access
    token is a JWT (codex-rs's own ``token_data.rs`` parses the same ``exp``
    claim client-side), so the expiry comes from decoding that claim rather
    than from a plain field the way Claude Code's credential file has one.
    """

    source: str
    present: bool = False
    expires_at: float | None = None
    expired: bool = False
    refreshable: bool = True
    detail: str | None = None

    @property
    def usable(self) -> bool:
        """Whether this credential can plausibly authenticate a child process."""
        if not self.present:
            return False
        return not (self.expired and not self.refreshable)


def _jwt_exp(token: str) -> float | None:
    """Decode a JWT's ``exp`` claim without verifying the signature.

    modelpass never checks a token's signature or treats it as anything but an
    opaque string beyond this one claim -- the same "metadata only" boundary
    :class:`CredentialStatus` documents.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    padded = payload + "=" * (-len(payload) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return None
    exp = data.get("exp") if isinstance(data, Mapping) else None
    return float(exp) if isinstance(exp, (int, float)) else None


def read_credential_status(home: Path) -> CredentialStatus:
    """Detect the Codex ChatGPT login without reading any secret value.

    ``auth.json`` also carries an ``openai_api_key`` for API-key logins; that
    shape has no ``tokens`` entry and is reported as absent here rather than
    unusable, because it is simply not a subscription login to begin with.
    """
    path = home / "auth.json"
    source = f"Codex login ({path})"
    if not path.is_file():
        return CredentialStatus(source=source, present=False, detail="no credential file")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return CredentialStatus(
            source=source, present=True, detail=f"credential file unreadable ({type(exc).__name__})"
        )

    tokens = raw.get("tokens") if isinstance(raw, Mapping) else None
    if not isinstance(tokens, Mapping):
        return CredentialStatus(
            source=source, present=False, detail="no tokens entry in credential file"
        )

    access_token = str(tokens.get("access_token") or "")
    refreshable = bool(str(tokens.get("refresh_token") or ""))
    expires_at = _jwt_exp(access_token) if access_token else None
    expired = expires_at is not None and expires_at <= time.time()

    return CredentialStatus(
        source=source,
        present=bool(access_token),
        expires_at=expires_at,
        expired=expired,
        refreshable=refreshable,
    )


def _format_expiry(expires_at: float | None) -> str:
    if expires_at is None:
        return "at an unknown time"
    import datetime as _dt

    return _dt.datetime.fromtimestamp(expires_at, _dt.UTC).strftime("on %Y-%m-%d")


def _credential_store_notes(request: RunRequest, home: Path) -> tuple[str, ...]:
    """Where the login for this ``CODEX_HOME`` is kept, for an isolated profile.

    Reported, not enforced. modelpass used to refuse an isolated subscription
    profile configured for the OS keyring, on the theory that only ``auth.json``
    is deterministically relocated by ``CODEX_HOME``. That is not what Codex
    does: keyring storage is keyed by the codex home too -- service ``Codex
    Auth``, account key the SHA-256 of the canonicalized ``CODEX_HOME`` path
    (``codex-rs/login/src/auth/storage.rs``, ``compute_store_key`` and
    ``compute_keyring_account``) -- so a different directory reads a different
    entry under either store. Both isolate; the receipt says which one is in
    play, because "keyring" means there is nothing on disk for modelpass to point
    at and the identity probe is the whole of the evidence.

    ``cli_auth_credentials_store`` defaults to ``file``
    (``codex-rs/config/defaults.toml``); ``auto`` tries the keyring and falls
    back to ``auth.json`` when it is unavailable or empty.
    """
    connection = request.connection
    if not (connection.config_dir and connection.is_subscription):
        return ()
    configured = _read_codex_config(home).get("cli_auth_credentials_store")
    store = configured if isinstance(configured, str) else "file (default)"
    auth_file = home / "auth.json"
    if store == "keyring":
        where = "OS keyring, entry keyed to this CODEX_HOME"
    elif store == "auto":
        where = (
            "OS keyring if available, else "
            f"{auth_file}; both are keyed to this CODEX_HOME"
        )
    else:
        where = str(auth_file)
    return (f"credential store: {store} -- {where}",)


def _guard_timing_note(request: RunRequest) -> tuple[str, ...]:
    """When a token guard can fire on this run, which the transport decides.

    ``codex exec --json`` reports usage exactly once, at ``turn.completed``, so a
    threshold on that transport cannot stop a run -- only report on it after the
    fact, which is a different product and is said out loud rather than left in
    the capability table. The app-server sends
    ``thread/tokenUsage/updated`` after every model response (verified live
    2026-08-31, three updates across two turns in the committed capture), so the
    same threshold can fire mid-run there.

    **S7 flipped which of these is the surprise.** ``interim_usage`` now reads
    ``supported`` in the registry, because the row describes the default and the
    default reports mid-run. So the receipt's job on exec changed from *this is
    the runtime's limitation* to *this run opted out of the row you read*, which
    is a materially different sentence for somebody who set ``stop_at_tokens``
    and expects a circuit breaker.
    """
    if resolve_transport(request.options) == TRANSPORT_APP_SERVER:
        return (
            "token guards can fire mid-run on this transport: the app-server "
            "reports usage after every model response (thread/tokenUsage/updated), "
            "so a token stop interrupts this run rather than bounding the next one "
            "(capability interim_usage: supported, driven live 2026-08-31)",
        )
    return (
        "token guards check at the end of the run only: codex exec --json "
        "reports usage once, at turn.completed. The runtime's interim_usage row "
        "reads 'supported' and describes the DEFAULT transport, which this run "
        "opted out of with options={'transport': 'exec'} -- so a token stop here "
        "reports an overspend rather than preventing one, and bounds the NEXT run",
    )


def _schema_notes(request: RunRequest) -> tuple[str, ...]:
    """Receipt notes about a requested schema, before the run pays to find out.

    The 400 this predicts is cheap -- it lands at request validation, before any
    generation -- but it is also opaque: it names a ``response_format`` called
    ``codex_output_schema`` that the caller never wrote. Saying so on the receipt
    turns "why did my extraction fail" into one line read before the run.

    A *prediction*, not a gate: modelpass does not refuse the run, because the
    vendor is the authority on its own subset and this check is one release away
    from being wrong in either direction.
    """
    if request.schema is None:
        return ()
    issues = openai_strict_issues(request.schema)
    if not issues:
        return ()
    # Same Responses API strict mode on either transport -- only the way the
    # schema is handed over differs (a temp file for exec's --output-schema, a
    # per-turn TurnStartParams.outputSchema field for the app-server).
    where = (
        "TurnStartParams.outputSchema"
        if resolve_transport(request.options) == TRANSPORT_APP_SERVER
        else "codex exec --output-schema"
    )
    return (
        f"the requested schema is not in the strict subset {where} requires, so "
        "the runtime will likely reject it before "
        "generating (no tokens spent): "
        + "; ".join(issues[:4])
        + (" ..." if len(issues) > 4 else "")
        + ". modelpass.schema.to_openai_strict() converts a loose schema",
    )


SpawnFn = Callable[[list[str], dict[str, str], str], "subprocess.Popen[str] | Any"]
LoginStatusFn = Callable[[str, dict[str, str]], "str | None"]
AccountReadFn = Callable[[str, dict[str, str]], "Mapping[str, Any] | None"]
McpListFn = Callable[[str, dict[str, str]], "str | None"]
VersionFn = Callable[[str, dict[str, str]], "str | None"]


def exclusivity_disable_args(
    configured_json: str, requested: Mapping[str, Any]
) -> list[str]:
    """``-c`` overrides disabling every configured MCP server the caller did not name.

    This is what turns Codex's merge-only ``-c`` semantics into a real
    exclusivity guarantee: ``codex mcp list --json`` (a token-free config query,
    verified live 2026-08-16) enumerates what the user's ``config.toml`` would
    bring along, and per-server ``enabled=false`` overrides switch each one off
    for this invocation only. Nothing is written to the user's config.

    Raises ``ValueError`` if the JSON is not the expected shape — the caller
    fails closed rather than running without the guarantee it promised.
    """
    entries = json.loads(configured_json)
    if not isinstance(entries, list):
        raise ValueError("codex mcp list --json did not return a list")
    args: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            raise ValueError("codex mcp list --json entry has no usable name")
        name = entry["name"]
        if name in requested or entry.get("enabled") is False:
            continue
        args.extend(["-c", f"mcp_servers.{name}.enabled=false"])
    return args


def _default_mcp_list(binary: str, env: dict[str, str]) -> str | None:
    """Run ``codex mcp list --json`` with the scrubbed environment. Costs no tokens."""
    try:
        result = subprocess.run(
            [binary, "mcp", "list", "--json"],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout or ""


def _default_spawn(argv: list[str], env: dict[str, str], cwd: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # progress noise; real failures arrive as --json events
        # A pipe, because the prompt goes through it and never through argv:
        # the resolved binary is a .cmd shim on Windows and cmd.exe ends its
        # command line at the first newline, so an argv prompt loses every line
        # but the first (reproduced 2026-08-31). See _send_prompt.
        stdin=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _send_prompt(proc: Any, prompt: str) -> None:
    """Hand the child its prompt on stdin and close the pipe.

    **Why stdin and not argv.** On Windows the ``codex`` that resolves is
    ``codex.cmd``, a batch shim, and ``cmd.exe`` terminates its command line at
    the first newline: a multi-line prompt passed as an argument arrived with
    everything after line one missing, and the run reported ``ok`` anyway
    (reproduced 2026-08-31 with a throwaway ``.cmd``; no AI involved). ``-`` in
    the ``[PROMPT]`` slot is the CLI's own documented way to read the prompt
    from stdin, on ``codex exec`` and ``codex exec resume`` alike.

    **One or the other, never both.** A real prompt argument *and* piped stdin
    makes the CLI append the stdin text as a separate ``<stdin>`` block instead
    of treating it as the prompt, so the ``-`` and this write are a pair and
    every caller sends them together.

    Closing matters as much as writing: the CLI reads to EOF, so a stdin left
    open is a turn that never starts.

    An ``OSError`` here means the child is already gone -- a shim that failed to
    launch its target, a binary that exited on a bad flag. Swallowed on purpose:
    the drain and the exit code that follow report what actually happened, and a
    broken-pipe traceback would replace that account with a worse one.
    """
    stdin = getattr(proc, "stdin", None)
    if stdin is None:
        return
    try:
        stdin.write(prompt)
        stdin.close()
    except OSError:
        return


def _default_codex_version(binary: str, env: dict[str, str]) -> str | None:
    """``codex --version`` for the resolved binary. Costs no tokens.

    Probed only for a binary that is a concrete file on disk -- which is what
    :func:`_which_codex` returns and what ``options={"codex_bin": ...}`` should
    be. A bare command name is reported as it was given rather than run: a
    receipt describing a launch should not itself launch something to find out
    what it was handed.

    ``None`` on anything unexpected. An unnamed version is a smaller loss than a
    preflight that fails because a version query did.
    """
    if not os.path.isfile(binary):
        return None
    try:
        result = subprocess.run(
            [binary, "--version"],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    line = (result.stdout or "").strip().splitlines()
    return line[0].strip() if line and line[0].strip() else None


def _default_login_status(binary: str, env: dict[str, str]) -> str | None:
    """Run ``codex login status`` with the scrubbed environment. Costs no tokens."""
    try:
        result = subprocess.run(
            [binary, "login", "status"],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (result.stdout or "") + (result.stderr or "")


def _default_account_read(
    binary: str, env: dict[str, str]
) -> Mapping[str, Any] | None:
    """Read safe account metadata from Codex's supported app-server surface.

    ``account/read`` returns an account type, email and plan type; it does not
    return the stored OAuth credentials. As with the version probe, only launch
    a concrete binary path. Tests and embedders often pass the bare name
    ``codex`` as a sentinel and preflight must not unexpectedly resolve it.
    """
    if not os.path.isfile(binary):
        return None
    client = AppServerClient(binary=binary, env=env, cwd=tempfile.gettempdir())
    try:
        client.start()
        return client.request(
            "account/read", {"refreshToken": False}, timeout=20
        )
    except (OSError, VendorRunFailed):
        return None
    finally:
        client.close()


def _account_profile_from_account_read(
    data: Mapping[str, Any] | None,
) -> AccountProfile | None:
    """Normalize the allow-listed identity fields from ``account/read``."""
    if not isinstance(data, Mapping):
        return None
    account = data.get("account")
    if not isinstance(account, Mapping):
        return None

    def text(key: str) -> str | None:
        value = account.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else None

    return AccountProfile(
        vendor="openai",
        source="codex app-server account/read",
        logged_in=True,
        auth_method=text("type"),
        email=text("email"),
        subscription_type=text("planType"),
    )


def _default_session_spawn(
    argv: list[str], env: dict[str, str], cwd: str, stderr: IO[str]
) -> subprocess.Popen[str]:
    """Launch one turn of a session, with the CLI's stderr kept.

    The one difference from :func:`_default_spawn`, and the reason a session has
    its own seam rather than a fourth argument on the shared one: a stateless
    run has no resume to fail, so sending stderr to ``DEVNULL`` costs it
    nothing. A session does. ``codex exec resume`` on an unknown id writes
    nothing at all to stdout and puts the reason on stderr, so a session that
    discarded stderr could report only "exited with code 1" for the single
    failure a caller most needs named (verified live, codex-cli 0.151.0,
    2026-08-30).

    A **file**, not a second pipe: draining two pipes from one thread deadlocks
    the moment the child fills the one nobody is reading, and the child here is
    a long-lived agent turn. ``stdin`` is a pipe rather than a third file
    because it is written once and closed immediately -- the prompt goes through
    it rather than through argv, for the reason :func:`_send_prompt` records.
    """
    return subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=stderr,
        stdin=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


SessionSpawnFn = Callable[[list[str], dict[str, str], str, IO[str]], "subprocess.Popen[str] | Any"]

#: The ``-c`` overrides that switch Codex's own toolbelt off for a
#: ``ChatSession``, in the order they are sent. Every one of these was read back
#: against **codex-cli 0.151.0** on 2026-08-30; the module docstring carries the
#: evidence for each, including the two paths that are deliberately absent
#: because the CLI accepts and ignores them (``tools.view_image``,
#: ``features.unified_exec``). A silently-ignored override is indistinguishable
#: from a working one, so nothing goes in this table that was not read back or
#: type-checked by the installed binary.
_CHAT_TOOL_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("features.shell_tool", "false"),
    ("features.view_image", "false"),
    ("features.browser_use", "false"),
    ("features.image_generation", "false"),
    ("features.apps", "false"),
    ("web_search", "disabled"),
)


def chat_tool_overrides() -> list[str]:
    """``-c`` arguments that leave a Codex turn with no toolbelt of its own (D15).

    What a ``ChatSession`` means on this runtime -- **and, since D23, what a
    stateless** :meth:`~modelpass.Bridge.chat` **call means too.** Codex has no
    documented ``tools=[]`` equivalent, but ``-c`` is a fully generic
    config-layer override and the feature flags behind the built-in tools are
    reachable through it.

    **Measured twice, on both transports.** Disabling the shell tool was
    measured on codex-cli 0.117.0 to drop ~3.3k input tokens on an otherwise
    identical ``exec`` prompt. On 2026-08-31 the whole six-pair set was driven
    on ``app-server`` for the first time (D23): an otherwise identical bare
    stateless call went from **15,890 to 10,269 prompt tokens -- 5,621 removed,
    35.4% of the prefix**. Both numbers are tool definitions leaving the wire
    payload rather than being hidden locally, and the shell-less answers were
    near-identical on non-coding work.

    **The caller's own tools are unaffected**, which is the property D23
    depends on and which was driven in the same session: with these overrides
    in place, ``dynamicTools`` still registered, the server still sent
    ``item/tool/call``, the handler still ran in-process and the model still
    answered from what it returned. Switching off the runtime's toolbelt does
    not switch off the caller's.

    Not a guarantee that *nothing* executable remains: ``unified_exec`` is a
    stable feature on 0.151.0 that ``-c`` and ``--disable`` both fail to switch
    off, and whether it survives ``features.shell_tool=false`` on the wire has
    not been checked. Stated here rather than implied by an empty list, because
    the difference between "the toolbelt is off" and "the toolbelt modelpass can
    reach is off" is exactly the kind of thing that should not have to be
    rediscovered.
    """
    args: list[str] = []
    for key, value in _CHAT_TOOL_OVERRIDES:
        args.extend(["-c", f"{key}={value}"])
    return args


#: Codex's own words for "that thread is not on this disk", captured live on
#: 2026-08-30 against codex-cli 0.151.0 by resuming a UUID that was never a
#: thread: exit code 1, empty stdout, and
#: ``Error: thread/resume: thread/resume failed: no rollout found for thread id
#: <id> (code -32600)`` on stderr.
#:
#: Matched narrowly and on stderr text only, because the consequence of a false
#: positive is real: :class:`~modelpass.errors.SessionNotFound` is what the
#: prefix->session lookup under ``chat()`` will treat as a cache
#: miss, so reading an ordinary failure as a missing thread would quietly start
#: a fresh conversation instead of reporting that something broke.
_NO_THREAD_MARKERS = ("no rollout found for thread id", "thread/resume failed")


def is_missing_thread(stderr_text: str) -> bool:
    """Whether a failed ``codex exec resume`` failed because the thread is gone."""
    lowered = stderr_text.lower()
    return any(marker in lowered for marker in _NO_THREAD_MARKERS)


class CodexSession:
    """One Codex thread, driven one ``codex exec`` process per turn.

    A :class:`~modelpass.adapters.base.SessionHandle`, so it knows the vendor and
    nothing else: no stamping, no guards, no receipt, no opinion about whether
    it is backing a ``ChatSession`` or a ``WorkerSession``. All of that lives in
    :mod:`modelpass.sessions` and is written once.

    The thread is what persists, and it is created by running -- the first
    ``send`` is a plain ``codex exec``, every turn after it is
    ``codex exec resume <id>``, and :attr:`id` is ``None`` in between because a
    thread killed mid-first-turn leaves an id that resume rejects (adapter
    contract, rule 7). Nothing here passes ``--ephemeral``: that directive was
    written for the stateless call, and on this runtime the rollout file *is*
    the conversation.
    """

    def __init__(
        self,
        *,
        adapter: OpenAIAdapter,
        request: SessionRequest,
        binary: str | None,
        session_id: str | None = None,
    ) -> None:
        self._adapter = adapter
        self._request = request
        self._binary = binary
        self._id = session_id
        self._closed = False
        self._proc: Any = None
        #: MCP exclusivity is resolved once and reused for every turn. Not an
        #: optimization: recomputing it would let a change to the user's
        #: config.toml alter this session's launch mid-conversation, and the one
        #: thing a session owes its cache is a prefix that does not move (D17).
        self._exclusivity: list[str] | None = None

    # --- identity ----------------------------------------------------------------

    @property
    def id(self) -> str | None:
        """The thread id, or ``None`` until a first turn has **succeeded**.

        Two ways it is still ``None`` after a turn, both deliberate. A turn that
        ended in an error or a spent allowance publishes nothing, because
        ``thread.started`` arrives at the *start* of a turn and the rollout that
        makes the thread resumable is written at the end. And a turn that
        finished without a ``thread.started`` at all publishes nothing either --
        that is the only place the exec transport names a thread, so its absence
        is version drift worth noticing rather than a resume waiting to fail.
        """
        return self._id

    def __repr__(self) -> str:
        state = "closed" if self._closed else (self._id or "unsent")
        return f"<CodexSession {self._request.connection.name}:{state}>"

    # --- turns -------------------------------------------------------------------

    def send(self, message: str) -> Iterator[AgentEvent]:
        """Run one turn as its own ``codex exec`` process and stream its events.

        The same vocabulary and failure rules as :meth:`OpenAIAdapter.run`, with
        one addition the stateless path has no use for: a resume whose thread is
        gone raises :class:`~modelpass.errors.SessionNotFound` rather than
        reporting a terminal, because "this conversation no longer exists" is
        not an outcome of the turn -- the turn never started, and nothing was
        spent finding out.
        """
        if self._closed:
            raise SessionClosed(
                f"session on connection {self._request.connection.name!r} is closed"
            )
        if self._binary is None:
            # preflight would have refused first; belt and braces for a caller
            # driving the adapter directly, and the same words run() uses.
            yield self._terminal(TerminalStatus.ERROR, "codex CLI not found", TokenUsage())
            return

        resuming = self._id is not None
        # The system prompt goes in on the first turn only; the thread has held
        # it ever since, and restating it would move the cached prefix (D17).
        prompt = (
            message if resuming
            else render_session_prompt(self._request.system_prompt, message)
        )
        argv = [self._binary, *self._turn_args()]
        check_launch_args(self._adapter.runtime, argv)
        yield from self._stream_turn(argv, resuming, prompt)

    def _turn_args(self) -> list[str]:
        """The ``codex exec`` argument list for one turn. **No prompt in it.**

        ``codex exec resume [OPTIONS] [SESSION_ID] [PROMPT]`` on 0.151.0, which
        accepts ``--json``, ``--skip-git-repo-check`` and ``--model`` but **not**
        ``-C/--cd``: the working directory of a resumed turn is the process's,
        which is why ``project_folder`` is passed as the child's cwd rather than
        as a flag, exactly as the first turn does.

        The ``[PROMPT]`` slot holds ``-`` -- read it from stdin -- because a
        prompt with a newline in it does not survive the Windows ``.cmd`` shim
        (:func:`_send_prompt`, verified 2026-08-31). Position still matters: on
        a resume the session id comes first and the ``-`` after it, exactly
        where the prompt used to sit.
        """
        request = self._request
        args = ["exec"]
        if self._id is not None:
            args.append("resume")
        args.extend(["--json", "--skip-git-repo-check"])
        # No --ephemeral. directives_for() adds it for every openai-sdk
        # connection because it was written for the stateless call; a session
        # persists, and on this runtime dropping the rollout file would drop the
        # conversation with it.
        if not request.native_tools:
            args.extend(chat_tool_overrides())
        if request.model:
            args.extend(["--model", request.model])
        args.extend(mcp_config_args(request.mcp_servers))
        args.extend(self._exclusivity_args())
        if self._id is not None:
            args.append(self._id)
        args.append("-")
        return args

    def _exclusivity_args(self) -> list[str]:
        """Disable the user's configured MCP servers for this session's turns.

        The same enumerate-and-disable guarantee :meth:`OpenAIAdapter.run` makes,
        and it fails closed for the same reason: running without a guarantee
        that was promised is the silent surprise this library exists to prevent.
        Resolved on the first turn that needs it and reused thereafter.
        """
        request = self._request
        if not request.mcp_servers or request.options.get("allow_configured_mcp_servers"):
            return []
        if self._exclusivity is None:
            listing = self._adapter._mcp_list(self._binary or "", dict(request.plan.env))
            try:
                disable = (
                    None
                    if listing is None
                    else exclusivity_disable_args(listing, request.mcp_servers)
                )
            except ValueError:
                disable = None
            if disable is None:
                raise CapabilityNotSupported(
                    self._adapter.runtime.value,
                    "mcp_servers",
                    detail=(
                        "exclusivity could not be established for this session: "
                        "'codex mcp list --json' failed or returned an unexpected "
                        "shape, so the user's configured MCP servers cannot be "
                        "disabled for its turns. Pass options="
                        "{'allow_configured_mcp_servers': True} to run anyway with "
                        "those servers reachable"
                    ),
                )
            self._exclusivity = disable
        return list(self._exclusivity)

    def _stream_turn(
        self, argv: list[str], resuming: bool, prompt: str
    ) -> Iterator[AgentEvent]:
        """Launch one turn, drain it, and end with exactly one terminal event.

        ``prompt`` arrives separately from ``argv`` because that is where it
        travels: down the child's stdin, paired with the ``-`` the argument list
        carries in its place (:func:`_send_prompt`).
        """
        request = self._request
        outcome = _TurnOutcome()
        proc: Any = None
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
            try:
                try:
                    proc = self._adapter._session_spawn(
                        argv, dict(request.plan.env), request.project_folder, err
                    )
                except OSError as exc:
                    raise VendorRunFailed(
                        f"the codex CLI at {argv[0]!r} could not be launched: {exc}"
                    ) from exc
                self._proc = proc
                # After the handle is stored, so an abandoned turn still finds
                # a process to terminate.
                _send_prompt(proc, prompt)
                try:
                    yield from _drain_codex_jsonl(proc.stdout, outcome)
                    returncode = proc.wait()
                except OSError as exc:
                    raise VendorRunFailed(f"the codex CLI stream failed: {exc}") from exc
                status, reason = outcome.status, outcome.reason
                if returncode != 0:
                    stderr_text = _read_back(err)
                    if resuming and is_missing_thread(stderr_text):
                        # Not a turn that failed -- a conversation that is not
                        # there. Raised rather than reported, because the caller
                        # holding this id needs to stop using it, and a terminal
                        # event saying so would still leave the session looking
                        # usable.
                        raise SessionNotFound(
                            request.connection.name,
                            self._id or "",
                            "codex has no rollout for it: a thread is durable only "
                            "after its first turn completes, and Codex prunes its "
                            "own session files. Open a new session instead",
                        )
                    if status is TerminalStatus.OK:
                        status = TerminalStatus.ERROR
                        reason = _exit_reason(returncode, stderr_text)
                if (
                    self._id is None
                    and outcome.thread_id
                    and status is TerminalStatus.OK
                ):
                    # Only now, and only on a turn that actually finished.
                    # ``thread.started`` arrives at the *beginning* of the first
                    # turn, so publishing its id on a turn that then failed
                    # would hand over exactly the id resume rejects with "no
                    # rollout found" -- the lie rule 7 exists to prevent.
                    self._id = outcome.thread_id
                yield self._terminal(status, reason, outcome.usage)
            finally:
                OpenAIAdapter._terminate(proc)
                self._proc = None

    def _terminal(
        self, status: TerminalStatus, reason: str | None, usage: TokenUsage
    ) -> TerminalEvent:
        connection = self._request.connection
        return TerminalEvent(
            status=status,
            connection=connection.name,
            runtime=self._adapter.runtime,
            auth_mode=connection.auth_mode,
            reason=reason,
            usage=usage,
        )

    # --- history and lifecycle ---------------------------------------------------

    def history(self) -> tuple[Message, ...]:
        """Empty, and honestly so: ``codex exec`` offers no way to read a thread back.

        The transport has no counterpart to the Agent SDK's
        ``get_session_messages()``, and the rollout file under
        ``~/.codex/sessions/YYYY/MM/DD/`` is a private on-disk format the CLI is
        actively migrating (``codex migrate-rollouts``, and a
        ``background_paginated_rollout_migration`` feature flag). Reassembling a
        transcript from either that format or from the events this adapter
        happened to see would produce a second history that drifts from the one
        the model is actually being sent, and only one of the two is real.

        **The capability is real; this transport is wrong for it** (S5,
        2026-08-31). ``codex app-server`` pages a thread's own items back with
        ``thread/items/list`` and :meth:`CodexAppServerSession.history` returns
        them, so a caller who wants to read a conversation back opens the session
        with ``options={"transport": "app-server"}``. The empty tuple here is
        still the honest answer for a session that ran on ``codex exec``.
        """
        return ()

    def close(self) -> None:
        """Release modelpass's hold on the session. Idempotent.

        Does **not** delete the thread: a persisted Codex session stays
        resumable by :attr:`id`, which is the whole reason it was persisted.
        What ends here is the live subprocess, if a turn was abandoned
        mid-stream.
        """
        if self._closed:
            return
        self._closed = True
        proc, self._proc = self._proc, None
        OpenAIAdapter._terminate(proc)


def _read_back(handle: IO[str]) -> str:
    """Read a turn's captured stderr. Never the reason a turn fails."""
    try:
        handle.seek(0)
        return handle.read()
    except (OSError, ValueError):
        return ""


def _exit_reason(returncode: int, stderr_text: str) -> str:
    """``codex exited with code N``, plus what it said on the way out.

    The stateless path can only report the code, because it discards stderr. A
    session keeps it, and a non-zero exit with no JSONL behind it is precisely
    the case where the code alone says nothing.
    """
    detail = " ".join(stderr_text.split())[:500]
    base = f"codex exited with code {returncode}"
    return f"{base}: {detail}" if detail else base


def _session_thread_start_params(request: SessionRequest, cwd: str) -> dict[str, Any]:
    """``ThreadStartParams`` for a session's own thread.

    The stateless :func:`_thread_start_params` cannot be reused and the
    differences are all decisions rather than plumbing:

    * **No ``ephemeral``.** The stateless call passes ``ephemeral: true`` to
      leave nothing behind; a session's whole point is that it can be resumed,
      and the key is *omitted* rather than sent as ``false`` because omission is
      what was driven for a persisting thread. ``persist=False`` never reaches
      here -- see :meth:`OpenAIAdapter._refuse_unsupported_session`.
    * **``baseInstructions`` only for a chat.** It has *replace* semantics, which
      is exactly what a :class:`~modelpass.sessions.ChatSession` asks for and
      exactly what a :class:`~modelpass.sessions.WorkerSession` does not: a worker
      keeps Codex's persona and adds to it (``SessionRequest
      .appends_system_prompt``). A worker's prompt therefore goes where it goes
      on exec -- layered onto the first turn's text by
      :func:`render_session_prompt` -- because that is what append *is* on this
      runtime, and putting it here instead would silently delete the persona the
      caller chose a worker to keep. ``developerInstructions`` is the vendor's
      undriven candidate for a native layered-above channel
      (``ThreadStartParams``, schema-sourced 2026-08-31); nothing sends it yet.
    * ``dynamicTools`` is the caller's ``tools=``, omitted entirely when there
      are none -- and on a session that omission is load-bearing rather than
      tidy, because ``thread/resume`` restores a persisted registration when none
      is supplied and ``[]`` is a statement that could clear one.
    """
    params: dict[str, Any] = {"cwd": cwd}
    if request.system_prompt is not None and not request.appends_system_prompt:
        params["baseInstructions"] = request.system_prompt
    if request.model:
        params["model"] = request.model
    dynamic_tools = dynamic_tools_param(request.tools)
    if dynamic_tools is not None:
        params["dynamicTools"] = dynamic_tools
    return params


def _session_turn_start_params(
    request: SessionRequest, thread_id: str, message: str, *, first_turn: bool
) -> dict[str, Any]:
    """``TurnStartParams`` for one turn of a session. **Only the new message.**

    This is the difference between a session on this transport and the stateless
    call it replaces: the thread holds the conversation, so a second turn sends
    one input item and nothing else. Driven on 2026-08-31 -- the capture's second
    turn asked *"What was the integrated loudness again?"* with no history
    attached and was answered ``-13.7 LUFS`` from the first turn's tool result.

    No ``thread/inject_items`` here, and that is not an oversight: injection
    exists because a *stateless* call has an empty thread to seed
    (:meth:`OpenAIAdapter._inject_history`). A session's thread is the history.

    ``first_turn`` carries the one exec behaviour a worker keeps -- the system
    prompt layered above the first task and never restated, which is where D17
    wants it: at the front of a prefix that then stops moving.
    """
    text = (
        render_session_prompt(request.system_prompt, message)
        if first_turn and request.appends_system_prompt
        else message
    )
    return {"threadId": thread_id, "input": [{"type": "text", "text": text}]}


def _notification_turn_id(params: Mapping[str, Any]) -> str | None:
    """Which turn a notification belongs to, from either place it is written.

    **Two places, and missing the second one is a live bug the S5 tests caught.**
    Most notifications carry a flat ``turnId``; ``turn/started`` and
    ``turn/completed`` carry neither -- their id is inside ``turn.id``, next to
    the status. A stale-turn filter reading only the flat field therefore lets an
    abandoned turn's ``turn/completed`` through, which ends the *next* turn's
    drain the moment it starts and reports it finished before it produced
    anything.
    """
    turn_id = params.get("turnId")
    if isinstance(turn_id, str) and turn_id.strip():
        return turn_id.strip()
    turn = params.get("turn")
    nested = turn.get("id") if isinstance(turn, Mapping) else None
    return nested.strip() if isinstance(nested, str) and nested.strip() else None


def _started_turn_id(result: Mapping[str, Any]) -> str | None:
    """The turn id out of a ``turn/start`` response, or ``None``.

    ``{"turn": {"id": ..., "status": "inProgress"}}`` in the live capture. Unlike
    :func:`_started_thread_id` this is not fatal when absent: the id is also
    carried on every notification of the turn, and
    :meth:`~modelpass.adapters.codex_appserver.AppServerTurn._note_ids` picks it up
    there. Reading it from the response matters only because it is available
    *before* the first notification -- which is what lets an abandoned turn be
    named even if it produced nothing.
    """
    turn = result.get("turn")
    turn_id = turn.get("id") if isinstance(turn, Mapping) else None
    return turn_id.strip() if isinstance(turn_id, str) and turn_id.strip() else None


class CodexAppServerSession:
    """One Codex thread on **one long-lived** ``codex app-server`` child (S5).

    A :class:`~modelpass.adapters.base.SessionHandle`, so the same division of
    labour :class:`CodexSession` observes: it knows the vendor and nothing about
    stamping, guards, receipts or which of the two public session faces it is
    backing.

    What changes against the exec handle is the shape of a conversation. There,
    every turn is its own ``codex exec resume`` process and the thread is
    reconstructed from the rollout file each time; here one child holds the
    thread open, ``turn/start`` runs a turn on it, and three things exec cannot
    do at all become ordinary:

    * :meth:`history` reads the conversation back through
      ``thread/items/list`` -- exec answers ``()`` because it has no way to read
      a thread and will not reassemble one;
    * a :class:`~modelpass.sessions.ChatSession` works, because
      ``baseInstructions`` *replaces* the coding-agent persona rather than
      layering under it;
    * ``tools=`` works, through the same ``dynamicTools`` registration and
      ``item/tool/call`` dispatcher the stateless call uses (S6).

    **The id timing is exec's, deliberately, and the reasoning is written down
    because this transport genuinely differs.** ``thread/start`` answers
    immediately with a thread id *and* a rollout path, so modelpass knows the
    thread's name before the first turn -- which exec never does. It is still not
    published until a turn completes, because adapter contract rule 7 is about
    handing out an id a resume would reject, and *named* is not *durable*: the
    rollout under that path is written by the same binary that answers
    ``no rollout found for thread id`` for an exec thread whose first turn never
    finished (captured live 2026-08-30). Nobody has driven a ``thread/resume`` of
    a thread that never took a turn, so the conservative timing stands. One live
    experiment settles it -- start a thread, close the client, resume the id --
    and if it resumes, :attr:`id` can be published from ``thread/start`` on this
    transport alone.

    Not thread-safe: one conversation is one ordered sequence of turns, and
    :class:`~modelpass.sessions.Session` serializes them.
    """

    def __init__(
        self,
        *,
        adapter: OpenAIAdapter,
        request: SessionRequest,
        binary: str | None,
        session_id: str | None = None,
    ) -> None:
        self._adapter = adapter
        self._request = request
        self._binary = binary
        #: The id handed to the caller, under the rule in the class docstring.
        self._id = session_id
        #: The id the *wire* is addressed with, which a resume knows from the
        #: start and a new session learns from ``thread/start``. Two attributes
        #: rather than one because the whole id-timing decision lives in the gap.
        self._thread_id = session_id
        self._resume_id = session_id
        self._thread_ready = False
        self._closed = False
        self._client: AppServerClient | None = None
        #: Turns modelpass stopped watching -- abandoned iterators, mostly. Their
        #: notifications are still coming and must not be folded into the next
        #: turn's usage; see :meth:`_drain`.
        self._stale_turns: set[str] = set()
        self._turns = 0
        #: **Always present, even with no tools**, unlike the stateless path.
        #: ``thread/resume`` restores a thread's persisted ``dynamicTools`` and
        #: has no field to restate or clear them, so a resumed session can be
        #: asked to run a tool this process never declared. With a dispatcher the
        #: server gets a well-formed ``success: false`` naming what modelpass sent
        #: and the caller gets a ``tool_result`` saying so; without one it would
        #: get ``{}`` from :func:`~modelpass.adapters.codex_appserver.
        #: decline_approvals`, which the model is told is an invalid response.
        self._dispatcher = CallerToolDispatcher(request.tools)

    # --- identity ----------------------------------------------------------------

    @property
    def id(self) -> str | None:
        """The thread id, or ``None`` until a first turn has **succeeded**.

        A resumed session reports it immediately: the caller supplied a durable
        id and the runtime accepted it.
        """
        return self._id

    def __repr__(self) -> str:
        state = "closed" if self._closed else (self._id or "unsent")
        return f"<CodexAppServerSession {self._request.connection.name}:{state}>"

    # --- the child and the thread --------------------------------------------------

    def _ensure_client(self) -> AppServerClient:
        """The session's own app-server child, started on demand.

        Not started in :meth:`OpenAIAdapter.open_session`, because that is local
        work by contract (rule 7) and spawning a runtime is not. Started here
        rather than lazily inside the drain so that a child that will not launch
        is a failure of the turn that asked for it.

        A :class:`~modelpass.sessions.ChatSession` launches with
        :func:`chat_tool_overrides` on the command line -- this transport has no
        per-thread field for the runtime's own toolbelt, and ``codex app-server``
        accepts the same ``-c`` config layer ``codex exec`` does (checked
        token-free 2026-08-31; see
        :func:`~modelpass.adapters.codex_appserver.app_server_argv`).
        """
        client = self._client
        if client is not None and client.running:
            return client
        request = self._request
        client = AppServerClient(
            binary=self._binary or "",
            env=dict(request.plan.env),
            cwd=request.project_folder,
            spawn=self._adapter._app_server_spawn,
            server_request_handler=self._dispatcher,
            config_overrides=() if request.native_tools else chat_tool_overrides(),
        )
        client.start()
        self._client = client
        return client

    def _ensure_thread(self) -> AppServerClient:
        """Start or resume the thread this session runs on. Idempotent.

        Neither call runs a model turn, so both are free -- which is why a
        resume's *"this conversation is gone"* is answered before anything is
        spent, unlike on exec where the first turn is what finds out.
        """
        client = self._ensure_client()
        if self._thread_ready:
            return client
        if self._resume_id is not None:
            self._resume_thread(client, self._resume_id)
        else:
            started = self._start_thread(client)
            self._thread_id = _started_thread_id(started)
        self._thread_ready = True
        return client

    def _start_thread(self, client: AppServerClient) -> Mapping[str, Any]:
        """``thread/start``, with a tool-bearing refusal made legible.

        The same reading :meth:`OpenAIAdapter._start_thread` gives the stateless
        call: the *reserved dynamic tool name* rule compares against the thread's
        own toolbelt and cannot be pre-checked, so the server has the last word
        and its bare JSON-RPC code is turned into a sentence naming what modelpass
        sent.
        """
        request = self._request
        params = _session_thread_start_params(request, request.project_folder)
        try:
            return client.request("thread/start", params)
        except AppServerRequestFailed as exc:
            if not request.tools:
                raise
            names = ", ".join(tool.name for tool in request.tools)
            raise VendorRunFailed(
                f"codex app-server refused thread/start for a session declaring "
                f"tools [{names}]: {exc.detail} (code {exc.code}). A dynamic tool "
                "name may be refused as RESERVED because Codex's own toolbelt "
                "already uses it, which depends on the thread's configuration and "
                "cannot be known before the thread exists; rename the tool if that "
                "is the collision"
            ) from exc

    def _resume_thread(self, client: AppServerClient, thread_id: str) -> None:
        """``thread/resume``, turning the vendor's own refusal into SessionNotFound.

        **Undriven from this side and marked so** (see
        :data:`~modelpass.adapters.codex_appserver.THREAD_RESUME_METHOD`), with one
        real piece of evidence behind the failure branch: the ``no rollout found
        for thread id`` text ``codex exec resume`` prints on a dead id *is* this
        method's error, quoted by the exec CLI, captured live on 2026-08-30. So
        :func:`is_missing_thread` reads the same words here that it reads there,
        and a gone thread is raised rather than reported for the same reason --
        the turn never started, nothing was spent, and a terminal event would
        leave the session looking usable.

        Any other refusal is a vendor failure and keeps its own words.
        """
        request = self._request
        try:
            client.request(
                THREAD_RESUME_METHOD,
                thread_resume_params(
                    thread_id, cwd=request.project_folder, model=request.model
                ),
            )
        except AppServerRequestFailed as exc:
            if is_missing_thread(exc.detail):
                raise SessionNotFound(
                    request.connection.name,
                    thread_id,
                    "codex has no rollout for it: a thread is durable only after "
                    "its first turn completes, and Codex prunes its own session "
                    "files. Open a new session instead",
                ) from exc
            raise

    # --- turns -------------------------------------------------------------------

    def send(self, message: str) -> Iterator[AgentEvent]:
        """Run one turn on this thread and stream normalized events.

        The same vocabulary and failure rules as :meth:`CodexSession.send`,
        including the one addition sessions have over the stateless path: a
        resume whose thread is gone raises
        :class:`~modelpass.errors.SessionNotFound` rather than reporting a
        terminal. Here it raises *before* the turn is sent rather than after it
        failed, because ``thread/resume`` is a free call.
        """
        if self._closed:
            raise SessionClosed(
                f"session on connection {self._request.connection.name!r} is closed"
            )
        if self._binary is None:
            # preflight would have refused first; belt and braces for a caller
            # driving the adapter directly, and the same words run() uses.
            yield self._terminal(TerminalStatus.ERROR, "codex CLI not found", TokenUsage())
            return
        yield from self._stream_turn(message)

    def _stream_turn(self, message: str) -> Iterator[AgentEvent]:
        """One ``turn/start`` and its drain, ending in exactly one terminal.

        ``turn/start`` returns immediately with ``status: "inProgress"`` (live
        correction, 2026-08-31): its response is used to learn the turn's id and
        for nothing else, and the verdict comes from ``turn/completed`` on the
        notification stream.

        The ``finally`` is the cancel path. A caller who abandons this iterator
        gets D10's floor plus one politer step -- ``turn/interrupt`` -- and,
        unlike the stateless run, the child is **not** closed behind it: the turn
        ended, the conversation did not.
        """
        turn = AppServerTurn()
        turn.thread_id = self._thread_id
        finished = False
        try:
            client = self._ensure_thread()
            thread_id = self._thread_id
            if not thread_id:
                raise VendorRunFailed(
                    "codex app-server started a thread without an id, so there is "
                    "nothing to run this turn on"
                )
            turn.thread_id = thread_id
            started = client.request(
                "turn/start",
                _session_turn_start_params(
                    self._request, thread_id, message, first_turn=self._turns == 0
                ),
            )
            turn.turn_id = _started_turn_id(started) or turn.turn_id
            # The adapter's own drain, not a second one: see
            # :meth:`OpenAIAdapter._drain_app_server` for why one turn loop
            # serves both callers and what the two arguments below change.
            yield from self._adapter._drain_app_server(
                client, turn, self._dispatcher, stale_turns=self._stale_turns
            )
            finished = True
            self._turns += 1
            if self._id is None and turn.status is TerminalStatus.OK:
                # Only on a turn that actually finished: see the class docstring
                # for why "the server named the thread" is not "the thread is
                # durable" until somebody drives the resume that proves it.
                self._id = self._thread_id
            yield self._terminal(turn.status, turn.reason, turn.usage)
        finally:
            if not finished:
                self._abandon(turn)

    def _abandon(self, turn: AppServerTurn) -> None:
        """A turn nobody is watching any more: interrupt it, keep the session.

        **This is the session's ``graceful_cancel`` path**, and it is attempted
        rather than relied on. Half of it is now driven: on 2026-08-31 a live
        ``turn/interrupt`` returned ``{}`` and the turn ended with
        ``status: "interrupted"``
        (``tests/fixtures/appserver/live-capability-evidence-2026-08-31.jsonl``),
        so the mechanism is real and the status modelpass already maps to
        ``cancelled`` is one Codex actually sends. It is still written to be
        harmless when it is wrong: no ids means no attempt, any failure is
        swallowed, and the terminate floor is still :meth:`close`.

        **What is left to check, and it is the half this method depends on**: the
        drive interrupted a turn on a thread it then abandoned, so nobody has
        confirmed the thread is usable for a *next* ``turn/start`` afterwards --
        which is exactly the difference between a graceful cancel and a polite
        kill, and exactly what a session needs. Nothing waits here for the
        ``turn/completed`` the interrupt produces either, so a following ``send``
        meets it as a stale notification rather than as this turn's outcome. And
        whether an interrupted turn is billed is unknown. The capability registry
        claims nothing until those are answered.
        """
        turn_id = turn.turn_id
        if turn_id:
            self._stale_turns.add(turn_id)
        client = self._client
        if client is not None:
            interrupt_app_server_turn(client, turn.thread_id, turn_id)

    def _terminal(
        self, status: TerminalStatus, reason: str | None, usage: TokenUsage
    ) -> TerminalEvent:
        connection = self._request.connection
        return TerminalEvent(
            status=status,
            connection=connection.name,
            runtime=self._adapter.runtime,
            auth_mode=connection.auth_mode,
            reason=reason,
            usage=usage,
        )

    # --- history and lifecycle ---------------------------------------------------

    def history(self) -> tuple[Message, ...]:
        """The conversation as the thread has it, read back with ``thread/items/list``.

        **The headline difference between the two transports.**
        :meth:`CodexSession.history` returns ``()`` and says why: ``codex exec``
        has no way to read a thread back, and reassembling one from the events it
        happened to see would produce a second history that drifts from the one
        the model is being sent. This transport can simply ask.

        ``thread/items/list`` rather than ``thread/read(includeTurns=true)``, and
        the vendor chose for us: its own schema calls full-history hydration
        *deprecated for paginated threads*, every thread in the capture is
        ``historyMode: "paginated"``, and it names this method as the
        replacement. The two are not equivalent anyway -- the ``turn/completed``
        the capture holds carries ``itemsView: "summary"`` and, on a turn that
        ran a tool and answered a question, listed **only the assistant's final
        message**. A history read that way would have quietly lost the user's own
        turn. Items are paged in ascending order, which is conversation order,
        and every page is followed until ``nextCursor`` is ``null``: no cap, so
        nothing is truncated, and a repeated cursor stops the loop rather than
        spinning on a server that is not advancing.

        **What is undriven, marked because it is the part a live capture should
        pin next**: the method itself. Its params and response are schema-sourced
        from the shipped binary's own generator, and no session has yet read a
        real thread back. A build that refuses it raises the vendor's own error
        rather than answering ``()`` -- an empty history means *this conversation
        is empty*, and saying that about a thread modelpass could not read would be
        the exact lie the exec docstring refuses to tell.

        ``()`` before the first turn: there is no thread yet, and nothing is
        started to find that out.
        """
        if self._closed or self._thread_id is None:
            return ()
        client = self._ensure_thread()
        thread_id = self._thread_id
        history: list[Message] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            result = client.request(
                THREAD_ITEMS_LIST_METHOD,
                thread_items_list_params(thread_id, cursor=cursor),
            )
            entries, cursor = items_page(result)
            history.extend(thread_items_history(entries))
            if cursor is None or cursor in seen:
                return tuple(history)
            seen.add(cursor)

    def close(self) -> None:
        """Release modelpass's hold on the session. Idempotent.

        Does **not** delete the thread: a persisted Codex thread stays resumable
        by :attr:`id`. What ends here is the app-server child, which is also the
        terminate floor behind :meth:`_abandon`'s interrupt -- a runtime that
        will not stop must not be able to keep modelpass from returning.
        """
        if self._closed:
            return
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            client.close()


class OpenAIAdapter(Adapter):
    """Drives the Codex runtime under a ChatGPT subscription login."""

    runtime = Runtime.OPENAI_SDK

    #: Closed, because every key here is read by name below. An unknown one is
    #: a mistake worth naming rather than a passthrough worth honouring.
    option_keys = frozenset(
        {
            "allow_configured_mcp_servers",
            "codex_bin",
            "codex_home",
            "cwd",
            # D23. Read by RunRequest.native_tools, and it has to be listed
            # here or Bridge._option_note stamps "will be ignored" on the
            # receipt of the one call that honours it.
            "native_tools",
            "transport",
        }
    )

    def __init__(
        self,
        *,
        codex_bin: str | None = None,
        spawn: SpawnFn | None = None,
        session_spawn: SessionSpawnFn | None = None,
        app_server_spawn: AppServerSpawnFn | None = None,
        login_status: LoginStatusFn | None = None,
        account_read: AccountReadFn | None = None,
        mcp_list: McpListFn | None = None,
        codex_version: VersionFn | None = None,
        codex_home: str | os.PathLike[str] | None = None,
    ) -> None:
        self._codex_bin = codex_bin
        self._codex_home = codex_home
        self._spawn = spawn or _default_spawn
        self._session_spawn = session_spawn or _default_session_spawn
        #: The app-server transport's own spawn seam. ``None`` lets
        #: :class:`~modelpass.adapters.codex_appserver.AppServerClient` use its
        #: default; a test injects a scripted fake here and the whole JSON-RPC
        #: path runs with no process and no credential.
        self._app_server_spawn = app_server_spawn
        self._login_status = login_status or _default_login_status
        self._account_read = account_read or _default_account_read
        self._mcp_list = mcp_list or _default_mcp_list
        self._codex_version = codex_version or _default_codex_version
        self._proc: Any = None
        self._cancelled = False
        #: The live app-server client and turn, for :meth:`cancel`. Set for the
        #: length of one ``run()`` on that transport and ``None`` otherwise.
        self._client: AppServerClient | None = None
        self._app_turn: AppServerTurn | None = None
        #: ``account/read`` answers, keyed by binary *and* codex home -- two
        #: profiles on one binary are two different accounts and must never
        #: share an entry. Values are ``(monotonic timestamp, answer)``; see
        #: :data:`_IDENTITY_CACHE_TTL_SECONDS`.
        self._identity_cache: dict[
            tuple[str, str], tuple[float, Mapping[str, Any] | None]
        ] = {}

    def _cached_account_read(
        self, binary: str, env: dict[str, str], home: Path
    ) -> Mapping[str, Any] | None:
        """``account/read`` for one (binary, codex home), briefly cached."""
        key = (binary, str(home))
        now = time.monotonic()
        hit = self._identity_cache.get(key)
        if hit is not None and now - hit[0] < _IDENTITY_CACHE_TTL_SECONDS:
            return hit[1]
        answer = self._account_read(binary, env)
        self._identity_cache[key] = (now, answer)
        return answer

    def invalidate_identity_cache(self) -> None:
        """Force the next preflight to re-probe. See :meth:`Adapter.invalidate_identity_cache`."""
        super().invalidate_identity_cache()
        self._identity_cache.clear()

    @classmethod
    def is_available(cls) -> bool:
        if _which_codex() is not None:
            return True
        return importlib.util.find_spec(_CODEX_MODULE) is not None

    # --- preflight -------------------------------------------------------------

    def _resolve_binary(self, request: RunRequest | SessionRequest) -> str | None:
        override = request.options.get("codex_bin") or self._codex_bin
        if override:
            return str(override)
        return _which_codex()

    def _codex_home_for(self, request: RunRequest) -> Path:
        return codex_home(
            request.plan.env, request.options.get("codex_home") or self._codex_home
        )

    def _model_fields(self, request: RunRequest, home: Path) -> dict[str, Any]:
        """What the receipt should say about the model, when the plan cannot say.

        The plan already knows a model the caller asked for or the connection
        declared; both outrank this. Only when neither named one does the
        adapter go and read what the runtime would default to -- which is
        knowledge only the adapter has, and which is exactly the fact that was
        missing when a live run died on a model rejection with nothing pre-run
        naming a model (first-consumer feedback, 2026-08-17).
        """
        if request.plan.model:
            return {}
        model, source = read_configured_model(home)
        if model is None:
            return {}
        return {"model": model, "model_source": source}

    def _binary_notes(self, request: RunRequest, binary: str, home: Path) -> tuple[str, ...]:
        """Which Codex this run will launch, and whether another one is configured.

        A stale CLI is a failure mode the receipt is well placed to pre-announce:
        an 0.117.0 shim on ``PATH`` cannot drive a model a newer build can, and
        the rejection arrives as an opaque 400 mid-run rather than as a line read
        beforehand (operational finding, 2026-08-30).
        """
        version = self._codex_version(binary, dict(request.plan.env))
        notes = [f"codex binary: {binary}" + (f" ({version})" if version else "")]
        configured = read_configured_cli_path(home)
        if configured and os.path.normcase(configured) != os.path.normcase(binary):
            notes.append(
                f"{_display_path(home / 'config.toml')} names a different Codex build "
                f"in CODEX_CLI_PATH ({configured}); this run uses the one resolved on "
                "PATH above, and the two versions do not necessarily accept the same "
                "models. modelpass does not switch binaries on a config value -- pass "
                "options={'codex_bin': ...} to run that one"
            )
        return tuple(notes)

    @staticmethod
    def _transport_notes(request: RunRequest) -> tuple[str, ...]:
        """Which Codex transport this run will use, and what changes with it.

        Named on every receipt, not only the non-default one: the plan of record
        commits to the receipt naming the transport that ran, and "silence means
        the default" is a convention a reader has to already know. It sits beside
        the ``codex binary`` line because the two together are the whole answer
        to *what was actually launched*.

        **Since S7 (2026-08-31) the default is app-server and exec is the
        opt-out**, so the two lines swapped roles: the app-server line no longer
        names an option a caller must have typed, and the exec line does. The
        exec line says *opted out of* rather than merely naming the transport,
        because on a receipt read after the flip the interesting fact about an
        exec run is that somebody chose it.

        The app-server line carries the differences a caller can feel, all of
        them consequences of the transport having parameters where exec has only
        a prompt. **The prompt sentence names which shapes replace and which
        still append**, because a session receipt is printed for a worker as
        often as for a chat and a worker deliberately keeps exec's layering: the
        line would otherwise describe a run this one is not.
        """
        transport = resolve_transport(request.options)
        if transport == TRANSPORT_EXEC:
            return (
                "codex transport: exec ('codex exec --json', one process per turn) -- "
                "selected with options={'transport': 'exec'}, which is the opt-out "
                "from the app-server default. On this transport a system prompt is a "
                "'System: ' line layered ON TOP OF Codex's coding-agent persona "
                "rather than replacing it, usage arrives once at the end of the run, "
                "text arrives whole rather than in deltas, and tools= is refused -- "
                "and per-run mcp_servers= works, which is the one thing the default "
                "cannot do",
            )
        return (
            "codex transport: app-server ('codex app-server --listen stdio://', "
            "JSON-RPC over one child) -- the default since 2026-08-31; "
            "options={'transport': 'exec'} opts out",
            "on this transport a stateless call's or a chat session's system prompt "
            "is ThreadStartParams.baseInstructions, which *replaces* Codex's "
            "built-in coding-agent persona rather than layering above it as the "
            "exec transport's 'System: ' prefix does; a worker session keeps that "
            "layering, because append is the semantics it asked for. history= is "
            "injected into the thread as real role-bearing items "
            "(thread/inject_items) instead of being flattened into one labelled "
            "transcript, and a session's history simply lives in the thread. Same "
            "request, different run -- pass options={'transport': 'exec'} for the "
            "older behaviour",
        )

    def _instruction_notes(self, request: RunRequest, home: Path) -> tuple[str, ...]:
        """AGENTS.md files that will join this run's prompt (D2 disclosure).

        Codex reads these into ``turn_context.user_instructions`` and there is no
        switch for it, so the receipt names them. Silence means none were found,
        which is the ordinary case: an empty ``~/.codex/AGENTS.md`` and a fresh
        temp-directory cwd contribute nothing.
        """
        sources = agents_md_sources(home, request.options.get("cwd"))
        if not sources:
            return ()
        return (
            "AGENTS.md joins this run's prompt as Codex's user_instructions, from: "
            + ", ".join(sources)
            + ". modelpass cannot switch this off -- relocating CODEX_HOME would orphan "
            "auth.json and break the subscription login -- so it is disclosed rather "
            "than scrubbed",
        )

    def _verify_subscription_token(
        self,
        binary: str,
        env: dict[str, str],
        home: Path,
        notes: tuple[str, ...],
    ) -> tuple[bool, str | None, tuple[str, ...]]:
        """Confirm the ChatGPT login's access token is actually usable now.

        ``codex login status`` reporting "logged in" is not the same claim as
        "the stored access token still works": Codex records a *permanent*
        refresh failure (dead refresh_token, revoked grant) without logging
        the account out, so a stale-but-optimistic status line is exactly the
        shape of failure a preflight is supposed to catch before a run spends
        anything. ``auth.json``'s token expiry is the fact ``login status``
        does not expose, so this reads it directly (metadata only -- see
        :class:`CredentialStatus`).

        An expired-but-refreshable token gets one extra chance: Codex
        refreshes lazily on any command that needs a valid token, and
        ``codex login status`` is exactly such a command, so re-running it
        once and re-reading ``auth.json`` afterwards is the cheapest way to
        turn "the runtime will probably refresh it" into "it just did, or it
        didn't." No model tokens are spent either way.
        """
        status = read_credential_status(home)
        if not status.present:
            return True, None, notes
        if status.usable and not status.expired:
            return True, None, notes

        if not status.usable:
            when = _format_expiry(status.expires_at)
            return (
                False,
                f"the Codex ChatGPT login's access token expired {when} and carries "
                "no refresh token, so a modelpass-spawned runtime cannot authenticate. "
                "Re-run 'codex login' to log in again",
                notes,
            )

        # status.expired and status.refreshable: give the CLI one chance to
        # refresh it, then trust only what auth.json shows afterwards.
        self._login_status(binary, env)
        refreshed = read_credential_status(home)
        if refreshed.expired:
            when = _format_expiry(refreshed.expires_at)
            return (
                False,
                f"the Codex ChatGPT login's access token expired {when} and could not "
                "be refreshed (re-checked via 'codex login status'); the stored refresh "
                "token may be revoked or invalid. Re-run 'codex login' to log in again",
                notes,
            )
        return True, None, (*notes, "access token had expired and was refreshed during preflight")

    def preflight(self, request: RunRequest) -> Receipt:
        plan = request.plan
        home = self._codex_home_for(request)
        model_fields = self._model_fields(request, home)
        binary = self._resolve_binary(request)
        if binary is None:
            return Receipt.from_plan(
                plan,
                runtime_available=False,
                ok=False,
                problem=(
                    "codex CLI not found; install the Codex CLI (or 'pip install "
                    "modelpass[openai]' and pass options={'codex_bin': ...})"
                ),
                **model_fields,
            )

        notes = (
            *self._binary_notes(request, binary, home),
            *_credential_store_notes(request, home),
            *self._transport_notes(request),
            *self._instruction_notes(request, home),
            "forced_login_method is enforced by login-status detection plus env "
            "scrubbing in v1; the config.toml setting is not modified",
            # Where the user actually reads it, not only in the capability table:
            # a threshold that cannot fire until the run is over is a different
            # product from one that can, and saying which is which is the point.
            # It is also the one line that is transport-dependent: the app-server
            # streams thread/tokenUsage/updated after every model response, so
            # repeating exec's sentence there would be a receipt describing a
            # limitation the run does not have.
            *_guard_timing_note(request),
            *_schema_notes(request),
        )
        output = self._login_status(binary, dict(plan.env))
        if output is None:
            return Receipt.from_plan(
                plan,
                runtime_available=True,
                ok=False,
                problem="could not run 'codex login status' to determine the auth mode",
                notes=notes,
                binary=binary,
                **model_fields,
            )

        lower = output.lower()
        if "logged in using chatgpt" in lower:
            detected: AuthMode | None = AuthMode.SUBSCRIPTION
            source = f"Sign in with ChatGPT (codex login, {home / 'auth.json'})"
            ok, problem = True, None
        elif "logged in" in lower and "api key" in lower:
            detected = AuthMode.API_KEY
            source = "API key login (codex login --with-api-key)"
            ok, problem = True, None
        elif "not logged in" in lower:
            detected, source = None, ""
            # Name the directory when the connection chose one. `codex login`
            # with no CODEX_HOME logs in the *default* account, so a user who
            # follows the bare advice against a profile connection authenticates
            # the wrong account and the connection fails exactly as before --
            # having spent a browser round trip to change nothing.
            if request.connection.config_dir:
                problem = (
                    f"codex is not logged in for {home}; run 'codex login' with "
                    "CODEX_HOME set to that directory before using this "
                    "subscription connection"
                )
            else:
                problem = "codex is not logged in; run 'codex login' first"
            ok = False
        else:
            detected, source = None, ""
            ok = False
            problem = f"unrecognized 'codex login status' output: {output.strip()[:200]!r}"

        if detected is AuthMode.SUBSCRIPTION and ok:
            ok, problem, notes = self._verify_subscription_token(
                binary, dict(plan.env), home, notes
            )

        account_profile = None
        if detected is not None:
            account_profile = _account_profile_from_account_read(
                self._cached_account_read(binary, dict(plan.env), home)
            )

        return Receipt.from_plan(
            plan,
            detected_auth_mode=detected,
            credential_source=source,
            account=account_profile.email if account_profile else None,
            plan_name=(
                account_profile.subscription_type if account_profile else None
            ),
            account_profile=account_profile,
            runtime_available=True,
            ok=ok,
            problem=problem,
            notes=notes,
            binary=binary,
            **model_fields,
        )

    def cache_eligibility(
        self, request: RunRequest | SessionRequest
    ) -> CacheEligibility:
        """Whether this call's prefix can cache (D20). Pure; costs nothing.

        Two runtime facts shape the answer here, and both differ from
        ``anthropic-sdk``:

        * **The floor does not vary by model.** Codex caching is automatic at
          1,024 tokens or more, with no opt-in and no write fee, so the same
          number applies whatever is running.
        * **There is no TTL lever at all** -- a checked absence, not an
          unchecked question (capability ``ttl_control``). ``-c`` is a fully
          generic config override but there is no key here to override, so
          ``ttl`` is ``None`` and the note says why rather than leaving a reader
          to wonder whether modelpass simply did not look.

        And the floor is always cleared here, whatever the caller sends, which is
        why ``preset_prefix`` is unconditionally ``True``. **The reason differs by
        transport, and after S7 the default is the app-server one**, so the two
        are derived separately rather than one inheriting a sentence written
        about the other:

        * **exec.** Codex has no system-prompt parameter: its hardcoded
          coding-agent persona, its tool definitions and its environment context
          ride in front of everything on every ``exec``, measured at tens of
          thousands of input tokens before any content of the caller's.
        * **app-server (S7, 2026-08-31).** The persona no longer rides in front
          -- ``baseInstructions`` *replaces* it -- so the prefix is a **different
          and smaller** thing and the old arithmetic does not carry over. What it
          is instead was measured on this transport
          (``tests/fixtures/appserver/live-capability-evidence-2026-08-31.jsonl``):
          an identical turn cost **15,940** input tokens with no
          ``baseInstructions`` and **12,397** with a ~10-token one, so replacing
          the instructions removed ~3,543 tokens and left ~12.4k standing. That
          remainder is tool definitions and environment context, which
          ``baseInstructions`` does not name and does not touch. Separately, the
          cached figure held **flat at 12,032** across three responses while
          input grew 13,038 -> 13,123 -> 13,165, which is the direct measurement
          that what caches is the stable front and not the growing tail.
          12,032 against a 1,024-token floor is roughly twelve times over, so the
          floor is cleared comfortably -- for a different reason and by a
          different margin than on exec, which is the point of deriving it twice.

        Reporting "no system prompt, so no reusable prefix" for a bare call on
        either transport would be a true statement about the caller and a false
        one about the request.

        **The toolbelt-off case, measured in D23 (2026-08-31).** Switching
        Codex's own toolbelt off (:data:`_CHAT_TOOL_OVERRIDES`) takes tool
        definitions out of the remaining ~12.4k and shrinks the prefix again:
        an otherwise identical bare call went from **15,890 to 10,269** prompt
        tokens, **5,621 removed**. That configuration is no longer exotic --
        since D23 it is what *every* chat-shaped call on this runtime gets, so
        the numbers above (15,940 / 12,397 / 12,032, all taken with the toolbelt
        **on**) describe the pre-D23 launch and are kept as the derivation
        rather than as a description of today's traffic.

        **The floor claim survives the change, which is why it is still stated.**
        The cached front was 12,032 with the toolbelt on; removing 5,621 tokens
        of tool definitions cannot leave it near a 1,024-token floor, and the
        post-fix total prefix of 10,269 is itself ten times over. What is *not*
        re-measured is the cached figure specifically under the new launch --
        only the total is -- so this reports the floor as comfortably cleared and
        does not restate 12,032 as though it were still the number.
        """
        app_server = resolve_transport(request.options) == TRANSPORT_APP_SERVER
        floor_source = (
            "Codex caches automatically at this size, whatever the model"
            if not app_server
            else (
                "Codex caches automatically at this size, whatever the model. On this "
                "transport the cacheable front is Codex's environment context rather "
                "than its persona -- baseInstructions replaces the persona, and since "
                "D23 a chat-shaped call also runs with the runtime's own tool "
                "definitions switched off. The front was measured at 12,032 cached "
                "tokens flat across responses while input grew, with the toolbelt on "
                "(2026-08-31); the fix then removed 5,621 tokens of tool definitions "
                "from the prefix, so the floor is cleared by a smaller margin than "
                "that figure and still by an order of magnitude"
            )
        )
        return CacheEligibility(
            system_prompt_chars=request.system_prompt_chars,
            floor_tokens=DEFAULT_CACHE_FLOOR_TOKENS,
            floor_source=floor_source,
            tools_declared=request.wants_tools,
            preset_prefix=True,
            ttl=None,
            ttl_detail=(
                "this runtime has no TTL lever: caching is automatic, exact-prefix and "
                "not configurable, with roughly five to ten minutes of idle life "
                "against a one-hour ceiling. Prefix stability is the only cache "
                "control modelpass has here, which is what makes the immutable tools and "
                "system prompt load-bearing rather than merely tidy"
            ),
        )

    #: Capabilities whose answer on this runtime depends on which transport the
    #: request selected, and what an **explicit** ``transport="exec"`` answers
    #: for them.
    #:
    #: **This table was inverted in S7 (2026-08-31), not deleted, and the
    #: inversion is the whole point.** Until S7 the registry described ``codex
    #: exec`` and this constant carried the three cells app-server could do that
    #: exec could not. Flipping the default made most of that the registry's own
    #: answer: ``tools_in_process``, ``system_prompt_replace``, ``sessions_list``,
    #: ``interim_usage``, ``incremental_text`` and ``thinking`` are now
    #: ``supported`` in the table, because the table describes the default and the
    #: default is ``codex app-server``. What remains request-dependent is the
    #: mirror image: selecting exec **narrows** what a request can do, exactly as
    #: selecting app-server used to widen it. Same hook, same two questions, other
    #: direction.
    #:
    #: Each entry is a checked statement about ``codex exec`` and every one of
    #: them predates this slice -- nothing here was re-verified by flipping a
    #: default, because a default cannot change what a transport can do:
    #:
    #: * ``tools_in_process`` -- exec's only tool channel is MCP over a command
    #:   line; no in-process registration exists there (Phase 6, 2026-08-16).
    #: * ``system_prompt_replace`` -- exec has no system-prompt parameter, so a
    #:   system message becomes a ``System: `` line layered *on top of* the
    #:   vendor persona (D16, 2026-08-30).
    #: * ``sessions_list`` -- ``codex exec --help`` offers only ``resume
    #:   [SESSION_ID]``; ``codex resume`` is an interactive picker (2026-08-30).
    #: * ``interim_usage`` -- the exec JSON vocabulary carries no usage before
    #:   ``turn.completed``, and one exec invocation is one turn, so a token guard
    #:   there reports rather than interrupts (Phase 5, 2026-08-17).
    #: * ``incremental_text`` -- ``agent_message`` is absent from exec's lifecycle
    #:   enum, so the whole answer arrives in one ``item.completed`` (2026-08-30).
    #: * ``thinking`` -- ``unverified`` rather than ``unsupported``: nobody has
    #:   driven reasoning events on exec either way, and an absence of knowledge
    #:   is not a no. The app-server evidence (summary deltas, 2026-08-31) says
    #:   nothing about this transport.
    #:
    #: And one entry that **widens** instead, which is why this is a table of
    #: answers rather than a table of refusals:
    #:
    #: * ``mcp_servers`` -- ``supported`` on exec, where ``-c
    #:   mcp_servers.<name>={...}`` plus enumerate-and-disable exclusivity was
    #:   driven live (2026-08-16). The registry now reads ``unsupported`` because
    #:   the default cannot do it; a request that selects exec can, and gets to
    #:   hear so instead of being refused by a cell describing a transport it
    #:   opted out of.
    #:
    #: Cells this deliberately does **not** answer, in either direction:
    #:
    #: * ``sessions_fork`` -- ``thread/fork`` is in the build's method list and
    #:   nothing calls it. Undriven, unbuilt, ``unverified`` everywhere.
    #: * ``graceful_cancel`` -- the *vendor's* half is driven (a live
    #:   ``turn/interrupt`` ended a turn with ``status: "interrupted"``,
    #:   ``live-capability-evidence-2026-08-31.jsonl``). **modelpass's half is not**:
    #:   neither the run nor a session waits for the terminal that interrupt
    #:   produces, and nobody has confirmed a thread survives one and takes
    #:   another turn. Claiming the cell would promise the part that was not
    #:   checked -- see :meth:`CodexAppServerSession._abandon`.
    #: * ``ephemeral_multi_turn`` -- **driven** at the vendor the same day: two
    #:   turns on one ``ephemeral: true`` thread, the second recalling a number
    #:   given only in the first. Not claimed because modelpass has not *built* it:
    #:   a session always starts a persisting thread, so ``persist=False`` is
    #:   refused rather than degraded. An evidence-backed follow-up, not an open
    #:   question -- and the note says "not built" rather than implying the
    #:   runtime cannot.
    _EXEC_SUPPORT: ClassVar[dict[Capability, Support]] = {
        Capability.TOOLS_IN_PROCESS: Support.UNSUPPORTED,
        Capability.SYSTEM_PROMPT_REPLACE: Support.UNSUPPORTED,
        Capability.SESSIONS_LIST: Support.UNSUPPORTED,
        Capability.INTERIM_USAGE: Support.UNSUPPORTED,
        Capability.INCREMENTAL_TEXT: Support.UNSUPPORTED,
        Capability.THINKING: Support.UNVERIFIED,
        Capability.MCP_SERVERS: Support.SUPPORTED,
    }

    def support_for(
        self, capability: Capability, options: Mapping[str, Any]
    ) -> Support | None:
        """What *this request* can do, when ``options`` is what decides it (S7a).

        See :meth:`~modelpass.adapters.base.Adapter.support_for` for the two
        questions this separates. On this runtime the request-dependent fact is
        the **transport**: ``codex exec`` and ``codex app-server`` are the same
        binary and the same runtime identity, and they do not have the same
        abilities. The registry describes the default; this says what the caller
        actually selected.

        **Since S7 the default is ``codex app-server``, so the direction
        reversed.** A request that selects nothing, or selects app-server
        explicitly, gets ``None`` for everything -- no opinion, because the
        registry's own row is now the right answer for it, and that is exactly
        what keeps :meth:`~modelpass.Bridge.find` and ``registry.support()``
        answering the *default* question without this hook in the way. A request
        that selects ``transport="exec"`` gets :data:`_EXEC_SUPPORT`, which
        narrows the six cells exec genuinely cannot do and widens the one
        (``mcp_servers``) that only exec can.

        An unusable ``transport`` value raises here rather than being ignored,
        which is :func:`resolve_transport`'s rule applied one step earlier: a
        caller who typed ``"appserver"`` should hear about *that* and not about a
        capability they would have had.
        """
        if resolve_transport(options) != TRANSPORT_EXEC:
            return None
        return self._EXEC_SUPPORT.get(capability)

    # --- run -------------------------------------------------------------------

    def run(self, request: RunRequest) -> Iterator[AgentEvent]:
        """One stateless turn, on whichever transport ``options`` selected.

        The transport is resolved **first**, before the binary lookup, because an
        unusable ``transport`` value is a refusal and not a run outcome: nothing
        has been launched yet and nothing was spent finding out.
        """
        transport = resolve_transport(request.options)
        binary = self._resolve_binary(request)
        if binary is None:
            # preflight would have failed first; belt and braces for direct callers
            yield TerminalEvent(
                status=TerminalStatus.ERROR,
                connection=request.connection.name,
                runtime=self.runtime,
                auth_mode=request.connection.auth_mode,
                reason="codex CLI not found",
            )
            return

        if transport == TRANSPORT_APP_SERVER:
            yield from self._run_app_server(request, binary)
            return

        if request.tools:
            # The bridge gates this before the preflight, so reaching here means
            # a caller drove the adapter directly. Refuse rather than silently
            # dropping the tools and returning an answer the model reached
            # without them.
            #
            # An absence in THIS TRANSPORT, and the message says so (S6,
            # 2026-08-31). It used to claim "Codex is an MCP client only", which
            # was a statement about the vendor and is false: the app-server
            # transport registers caller functions natively and modelpass drives
            # that loop today.
            #
            # S7 rewrote the second half rather than the first: reaching here now
            # means the caller *opted out* of the transport that would have run
            # their tools, so the fix is to drop an option rather than to add one.
            raise CapabilityNotSupported(
                self.runtime.value,
                "tools_in_process",
                "'codex exec' has no in-process tool registration -- its only "
                "tool channel is MCP, over a command line. This run selected it "
                "with options={'transport': 'exec'}; the DEFAULT transport "
                "(app-server) runs caller functions in-process, so removing that "
                "option is the whole fix (verified live 2026-08-31; the vendor "
                "marks that surface experimental). Or stay on exec and use "
                "mcp_servers= with a server the caller runs",
            )

        args = ["exec", "--json", "--skip-git-repo-check"]
        # D23: a chat-shaped call leaves the runtime's own toolbelt off, which
        # is what Bridge.chat's docstring has always promised and what
        # anthropic-sdk has always done. Mirrors the session path's own guard
        # in _turn_args; opt back in with options={"native_tools": True}.
        if not request.native_tools:
            args.extend(chat_tool_overrides())
        if any(d.name == "--ephemeral" for d in request.plan.directives):
            args.append("--ephemeral")
        if request.model:
            args.extend(["--model", request.model])
        args.extend(mcp_config_args(request.mcp_servers))
        if request.mcp_servers and not request.options.get("allow_configured_mcp_servers"):
            # Exclusivity: an MCP-bearing run reaches ONLY the servers the caller
            # named. Codex -c config merges with the user's config.toml, so the
            # guarantee is built by enumerating what is configured (token-free)
            # and disabling everything not requested — failing closed if the
            # enumeration fails, because running without a promised guarantee is
            # the kind of silent surprise this library exists to prevent.
            # Explicit opt-out: options={"allow_configured_mcp_servers": True}.
            listing = self._mcp_list(binary, dict(request.plan.env))
            try:
                disable = None if listing is None else exclusivity_disable_args(
                    listing, request.mcp_servers
                )
            except ValueError:
                disable = None
            if disable is None:
                raise CapabilityNotSupported(
                    self.runtime.value,
                    "mcp_servers",
                    "exclusivity could not be established: 'codex mcp list --json' "
                    "failed or returned an unexpected shape, so the user's "
                    "configured MCP servers cannot be disabled for this run. Pass "
                    "options={'allow_configured_mcp_servers': True} to run anyway "
                    "with those servers reachable",
                )
            args.extend(disable)

        # --output-schema takes a *path*, so the schema needs somewhere to live
        # for the length of the run. A private temp file, removed in the finally
        # below: nothing is written to the user's Codex home, for the same
        # reason mcp_config_args uses -c overrides rather than config.toml.
        # Created last, so every refusal above happens before there is a file to
        # clean up.
        schema_file: str | None = None
        proc: Any = None
        try:
            if request.schema is not None:
                schema_file = _write_schema_file(request.schema)
                args.extend(["--output-schema", schema_file])
            # ``-`` is the prompt: the text itself goes down stdin below,
            # because a multi-line argv element is cut at its first newline by
            # the Windows .cmd shim (_send_prompt has the reproduction).
            argv = [binary, *args, "-"]
            check_launch_args(self.runtime, argv)

            # Launching the runtime is the vendor's half of the run, so a
            # failure to launch is a run outcome and not an exception the caller
            # has to handle separately (adapter contract rule 5, 2026-08-17).
            # The Windows npm shim documented on _which_codex is the concrete
            # case: CreateProcess rejects the extensionless shim with WinError
            # 193, which used to escape the stream as AdapterFailed with no
            # terminal event behind it.
            try:
                cwd = str(
                    request.options.get("cwd") or tempfile.mkdtemp(prefix="modelpass-codex-")
                )
                self._cancelled = False
                proc = self._spawn(argv, dict(request.plan.env), cwd)
            except OSError as exc:
                raise VendorRunFailed(
                    f"the codex CLI at {binary!r} could not be launched: {exc}"
                ) from exc
            self._proc = proc
            # After the handle is stored, so a cancel arriving mid-write still
            # finds a process to terminate.
            _send_prompt(proc, render_prompt(request.messages))
            yield from self._stream_run(request, proc)
        finally:
            self._terminate(proc)
            self._proc = None
            _remove_schema_file(schema_file)

    def _stream_run(self, request: RunRequest, proc: Any) -> Iterator[AgentEvent]:
        """Drain the CLI's JSONL stream into normalized events."""

        outcome = _TurnOutcome()
        try:
            stdout: IO[str] | None = proc.stdout
            yield from _drain_codex_jsonl(stdout, outcome)
            status, reason = outcome.status, outcome.reason
            returncode = proc.wait()
            if self._cancelled:
                status = TerminalStatus.CANCELLED
                reason = "run cancelled; the in-flight turn leaves no vendor-side record"
            elif returncode != 0 and status is TerminalStatus.OK:
                status = TerminalStatus.ERROR
                reason = f"codex exited with code {returncode}"
            if request.schema is not None and status is not TerminalStatus.CANCELLED:
                # The answer, then the outcome (adapter rule 6). Skipped on a
                # cancellation, where "no structured answer" is the expected
                # consequence of the cancel rather than a second failure worth
                # reporting on top of it.
                structured, failure = build_structured_event(
                    request.schema,
                    raw=outcome.final_answer,
                    schema_name=request.schema_name,
                )
                if structured is not None:
                    yield structured
                if failure is not None and status is TerminalStatus.OK:
                    status, reason = TerminalStatus.ERROR, failure
            yield TerminalEvent(
                status=status,
                connection=request.connection.name,
                runtime=self.runtime,
                auth_mode=request.connection.auth_mode,
                reason=reason,
                usage=outcome.usage,
            )
        except OSError as exc:
            # The pipe died mid-run. Still the vendor's half of the run failing,
            # so it ends the stream rather than escaping it; whatever usage the
            # run already reported is in the bridge's tracker and survives onto
            # the terminal it builds.
            raise VendorRunFailed(f"the codex CLI stream failed: {exc}") from exc

    # --- run: the app-server transport (S4) --------------------------------------

    def _run_app_server(self, request: RunRequest, binary: str) -> Iterator[AgentEvent]:
        """One stateless turn over ``codex app-server``: thread, turn, drain, close.

        The shape of this is dictated by one live correction (2026-08-31, plan of
        record): **``turn/start`` returns immediately** with
        ``status: "inProgress"`` and the turn ends when ``turn/completed`` arrives
        on the notification stream. Code that read the ``turn/start`` response as
        the outcome would report an unfinished turn as finished, so the response
        is used for nothing but confirming the request was accepted, and the
        verdict comes from the drain.

        Four things are native here that ``codex exec`` has to work around, and
        each of them is a workaround deleted rather than a feature added:

        * the system prompt is ``ThreadStartParams.baseInstructions`` rather than
          a ``System: `` line glued to the task (:func:`app_server_base_instructions`);
        * a conversation is seeded into the thread's own history with
          ``thread/inject_items`` as real role-bearing items, rather than
          flattened into one prompt with a preamble telling the model it is
          reading a transcript (:meth:`_inject_history`); only the current
          message rides on ``turn/start``;
        * ``request.schema`` is ``TurnStartParams.outputSchema``, a **per-turn
          field**, so there is no temp file to write and remove -- exec needs
          ``--output-schema <FILE>`` because its only channel is a command line,
          and a file that has to be cleaned up in a ``finally`` is a failure mode
          this path simply does not have;
        * the stateless call's ``--ephemeral`` directive is
          ``ThreadStartParams.ephemeral: true``, a parameter rather than a flag.

        And a fifth thing is native here that ``codex exec`` cannot do at all
        (S6): ``tools=`` becomes ``ThreadStartParams.dynamicTools`` and the
        caller's own functions run **in this process**, dispatched by
        :class:`~modelpass.adapters.codex_appserver.CallerToolDispatcher` when the
        server sends ``item/tool/call``. Names are checked before anything is
        launched, so a name the vendor would refuse costs no process.

        The client is closed in a ``finally``, which on a generator means when it
        is exhausted *or* when the caller abandons it: an app-server child that
        outlives its run is a stray ``codex`` process holding a rollout open.
        """
        self._refuse_on_app_server(request)
        check_dynamic_tool_names(request.tools)
        cwd = str(request.options.get("cwd") or tempfile.mkdtemp(prefix="modelpass-codex-"))
        # No tools means no dispatcher, which means the client keeps
        # ``decline_approvals`` verbatim: a plain chat run on this transport is
        # byte-for-byte the run S4 shipped.
        dispatcher = CallerToolDispatcher(request.tools) if request.tools else None
        client = AppServerClient(
            binary=binary,
            env=dict(request.plan.env),
            cwd=cwd,
            spawn=self._app_server_spawn,
            server_request_handler=dispatcher,
            # D23, and the same line the session path carries at
            # _ensure_client. Driven live 2026-08-31: this removed 5,621
            # prompt tokens from a bare call (15,890 -> 10,269) and the
            # caller's own dynamicTools kept working across it.
            config_overrides=() if request.native_tools else chat_tool_overrides(),
        )
        turn = AppServerTurn()
        self._cancelled = False
        self._client = client
        self._app_turn = turn
        try:
            client.start()
            started = self._start_thread(client, request, cwd)
            turn.thread_id = _started_thread_id(started)
            injected = True
            for event in self._inject_history(client, request, turn.thread_id):
                injected = False
                yield event
            # Accepted, not finished. See the docstring.
            client.request(
                "turn/start",
                _turn_start_params(request, turn.thread_id, history_injected=injected),
            )
            yield from self._drain_app_server(
                client, turn, dispatcher, cancelled=lambda: self._cancelled
            )

            status, reason = turn.status, turn.reason
            if self._cancelled:
                status = TerminalStatus.CANCELLED
                reason = (
                    "run cancelled; turn/interrupt was attempted and the app-server "
                    "child was then closed, so no vendor-side terminal was awaited"
                )
            if request.schema is not None and status is not TerminalStatus.CANCELLED:
                # The answer, then the outcome (adapter rule 6), exactly as the
                # exec path orders them.
                structured, failure = build_structured_event(
                    request.schema,
                    raw=turn.final_answer,
                    schema_name=request.schema_name,
                )
                if structured is not None:
                    yield structured
                if failure is not None and status is TerminalStatus.OK:
                    status, reason = TerminalStatus.ERROR, failure
            yield TerminalEvent(
                status=status,
                connection=request.connection.name,
                runtime=self.runtime,
                auth_mode=request.connection.auth_mode,
                reason=reason,
                usage=turn.usage,
            )
        finally:
            self._client = None
            self._app_turn = None
            client.close()

    def _start_thread(
        self, client: AppServerClient, request: RunRequest, cwd: str
    ) -> Mapping[str, Any]:
        """``thread/start``, with a tool-bearing refusal made legible.

        One vendor rule for dynamic tool names cannot be pre-checked -- *dynamic
        tool name is reserved* compares against the thread's own active toolbelt,
        which does not exist until the thread does
        (:func:`~modelpass.adapters.codex_appserver.check_dynamic_tool_names` has
        the reading from the binary). So the server gets the last word, and this
        is where its word is turned into a message that names what modelpass sent.
        Without it the caller sees a bare JSON-RPC code and has to guess that one
        of their tool names collided with Codex's own.
        """
        try:
            return client.request("thread/start", _thread_start_params(request, cwd))
        except AppServerRequestFailed as exc:
            if not request.tools:
                raise
            names = ", ".join(tool.name for tool in request.tools)
            raise VendorRunFailed(
                f"codex app-server refused thread/start for a run declaring tools "
                f"[{names}]: {exc.detail} (code {exc.code}). The vendor validates "
                "dynamic tool names on this call -- a name may be empty, duplicated, "
                "outside ^[a-zA-Z0-9_-]+$, or RESERVED because Codex's own toolbelt "
                "already uses it. modelpass checks every one of those but the last, "
                "which depends on the thread's configuration and cannot be known "
                "before the thread exists; rename the tool if that is the collision"
            ) from exc

    def _inject_history(
        self, client: AppServerClient, request: RunRequest, thread_id: str
    ) -> Iterator[AgentEvent]:
        """Seed the thread's history with ``thread/inject_items``. Yields on failure.

        **This is what retires S4's ``Assistant: `` labels on this transport.**
        ``thread/inject_items`` takes raw Responses API items -- real
        role-bearing messages -- and appends them to the thread's model-visible
        history, which is the seam a *stateless* ``chat(history=...)`` needs: the
        call creates a fresh ephemeral thread per turn, so there is nothing
        server-side to lean on until something is put there. Driven live
        2026-08-31 on an ephemeral thread, the kind this path creates
        (``tests/fixtures/appserver/live-inject-items-2026-08-31.jsonl``): an
        injected user/assistant exchange was answered from correctly with no
        labels anywhere.

        The current message is deliberately **not** injected --
        :func:`app_server_turn_input` says why -- and a run with no prior turns
        sends no ``thread/inject_items`` at all.

        **Yielding is the failure path, and the labels are why it exists.** An
        empty stream means the history was injected. One
        :class:`~modelpass.types.VendorEvent` means the server refused the method
        and the caller of this generator falls back to S4's labelled transcript
        (:func:`app_server_input_items`), which every build accepts because it is
        ordinary input text.

        Why keep a fallback for a method that was driven: this surface **moves**.
        The wire name is snake_case while the vendor's own generated types imply
        camelCase (:data:`INJECT_ITEMS_METHOD`), and the 0.117 -> 0.151 upgrade
        falsified three facts in hours. Losing a caller's conversation because a
        build spells a method differently would be a far worse outcome than
        sending it the older way.

        **What the fallback costs, so the vendor_event is not a shrug**: the
        labels consume prompt space on every turn, and the model has to infer
        from ``Assistant: `` that a line was its own -- an instruction it follows
        well rather than a structure it is given. Announcing the downgrade is the
        point: a silent one is exactly what this library exists to prevent.
        """
        items = app_server_history_items(request.messages)
        if not items:
            return
        try:
            client.request(
                INJECT_ITEMS_METHOD, inject_items_params(thread_id, request.messages)
            )
        except AppServerRequestFailed as exc:
            yield VendorEvent(
                runtime=self.runtime,
                name="history/inject_items_refused",
                data={
                    "method": INJECT_ITEMS_METHOD,
                    "code": exc.code,
                    "detail": exc.detail,
                    "items": len(items),
                    "fallback": (
                        "prior turns are being sent as TurnStartParams.input with "
                        "'Assistant: ' / 'User: ' labels instead (the S4 shape). The "
                        "conversation is intact; it costs prompt space and asks the "
                        "model to infer roles it would otherwise have been given"
                    ),
                },
            )

    def _drain_app_server(
        self,
        client: AppServerClient,
        turn: AppServerTurn,
        dispatcher: CallerToolDispatcher | None = None,
        *,
        cancelled: Callable[[], bool] | None = None,
        stale_turns: Container[str] = frozenset(),
    ) -> Iterator[AgentEvent]:
        """Read notifications through S3's fold until the turn ends.

        **One turn loop for both callers** -- the stateless run and a session
        turn -- for the same reason ``_TurnOutcome`` and ``AppServerTurn`` are
        each written once: two readings of "which notification means the run
        failed" would drift, and the half that drifts is the half that decides
        how a run is reported. The two parameters below are the whole difference
        between the callers, and both default to the stateless behaviour.

        The end-of-stream sentinel before ``turn/completed`` means the child died
        mid-turn. Reported as a vendor failure carrying its stderr tail rather
        than as a quietly successful empty run, unless ``cancelled`` says a
        :meth:`cancel` put it there, which is the one case where a stream that
        stops early is what was asked for. A session passes no ``cancelled``: its
        cancel is ``turn/interrupt`` plus an abandoned iterator, and its child
        going away really is a failure.

        ``stale_turns`` names turns this caller has stopped watching -- a session
        outlives its turns, so an abandoned one's notifications are still
        arriving. They are announced as a ``vendor_event`` rather than folded,
        because folding would add another turn's tokens to this one's usage and
        could end it on another turn's ``turn/completed``, and dropping them would
        be the silent alternative. A stateless run has no earlier turn and passes
        nothing.

        **Caller-tool failures join the stream here**, and the ordering is
        causal rather than lucky: the dispatcher records a failure on the reader
        thread *before* it answers the server, and the server's completed item
        cannot arrive until after that answer -- so draining the dispatcher on
        each pass puts a handler's ``is_error`` result ahead of the item it
        explains. Recording the ids on the turn is what stops the server's own
        item becoming a second result for the same call.
        """
        while True:
            if dispatcher is not None:
                for event in dispatcher.drain_failures():
                    turn.reported_tool_failures.add(event.id)
                    yield event
            notification = client.next_notification()
            if notification is None:
                if cancelled is not None and cancelled():
                    return
                tail = client.stderr_tail()
                raise AppServerTransportClosed(
                    "the codex app-server stopped before the turn completed"
                    + (f": {' '.join(tail.split())[:500]}" if tail else "")
                )
            stale = _notification_turn_id(notification.params)
            if stale is not None and stale in stale_turns:
                yield VendorEvent(
                    runtime=self.runtime,
                    name="turn/stale_notification",
                    data={
                        "method": notification.method,
                        "turnId": stale,
                        "note": (
                            "this belongs to an earlier turn of the same session "
                            "that was abandoned mid-stream. It is reported rather "
                            "than dropped, and deliberately not folded into this "
                            "turn's usage or outcome"
                        ),
                    },
                )
                continue
            yield from turn.observe(notification)
            if notification.method == "turn/completed":
                if dispatcher is not None:
                    # A failure recorded while the last notification was in
                    # flight would otherwise never be yielded. Nothing dropped.
                    for event in dispatcher.drain_failures():
                        turn.reported_tool_failures.add(event.id)
                        yield event
                return

    def _refuse_on_app_server(self, request: RunRequest) -> None:
        """The one request shape this transport cannot honour yet.

        ``tools=`` used to be the other one and is honoured since S6. What is
        left is gated by the bridge before the preflight, so reaching it means a
        caller drove the adapter directly. Refused rather than dropped: an answer
        the model reached without the servers it was told it had is the silent
        surprise this library exists to prevent.

        **S7 made this the transport a caller gets without asking**, which turns
        the refusal from an edge case into a behaviour change: an
        ``mcp_servers=``-bearing call that worked on 2026-08-30 needs one option
        added to keep working. That is why ``mcp_servers`` moved to
        ``unsupported`` in the registry in the same commit -- the row describes
        the default, and a ``supported`` verdict that then failed here is exactly
        the shape of lie ``sessions_list`` was kept honest about -- and why
        :data:`OpenAIAdapter._EXEC_SUPPORT` answers ``supported`` for a request
        that selects exec, so the capability is refused by the gate only for the
        callers it is actually absent for.
        """
        if request.mcp_servers:
            raise CapabilityNotSupported(
                self.runtime.value,
                "mcp_servers",
                detail=(
                    "the app-server transport does not carry per-run MCP servers "
                    "yet: exec declares them with '-c mcp_servers.<name>={...}' and "
                    "builds exclusivity from 'codex mcp list --json', and neither "
                    "has an equivalent on this path that has been driven. Refused "
                    "rather than run without the servers, or run without the "
                    "exclusivity guarantee. app-server is the default since "
                    "2026-08-31, so this is one option away: "
                    "options={'transport': 'exec'}"
                ),
            )

    # --- sessions (D14-D17) ------------------------------------------------------

    def open_session(
        self, request: SessionRequest
    ) -> CodexSession | CodexAppServerSession:
        """Open a new thread. **Local work only: nothing is launched and nothing spent.**

        There is no create-thread call on either transport that modelpass is
        willing to make here, so all this does is resolve which binary the turns
        will use and hand back a handle with no id. The thread comes into
        existence during the first ``send``.

        **The transport is selected exactly as it is on** :meth:`run`:
        ``options={"transport": "app-server"}``, defaulting to ``exec``, resolved
        first so an unusable value is a refusal rather than a run outcome (S5,
        2026-08-31). The receipt names which one a session runs on, because the
        session's preflight is taken over the same options.

        On **exec**, id timing is forced: a thread is durable only after its
        first turn finishes and a run killed before that leaves an id ``codex
        exec resume`` rejects with ``no rollout found for thread id``. On
        **app-server** ``thread/start`` would answer with a thread id
        immediately -- and it is still not called here, because rule 7 says this
        method builds local state and launches nothing, and because *named* is
        not *durable*:
        :class:`CodexAppServerSession` carries the full reasoning and the one
        experiment that would settle it.

        Resolving the binary is a ``PATH`` lookup, which spends nothing and
        makes every turn of this session launch the same executable. A missing
        one is reported by the first turn's terminal rather than raised: the
        preflight already refused before a caller could get here, and a runtime
        that will not start is a run outcome (rule 5).
        """
        transport = resolve_transport(request.options)
        self._refuse_unsupported_session(request, transport)
        binary = self._resolve_binary(request)
        if transport == TRANSPORT_APP_SERVER:
            # Before anything launches, exactly as the stateless path does it: a
            # name the vendor would refuse should not cost a process.
            check_dynamic_tool_names(request.tools)
            return CodexAppServerSession(adapter=self, request=request, binary=binary)
        return CodexSession(adapter=self, request=request, binary=binary)

    def resume_session(
        self, request: SessionRequest
    ) -> CodexSession | CodexAppServerSession:
        """Pick a persisted thread back up. ``request.resume_id`` names it.

        Transport-selected like :meth:`open_session`. On **app-server** the
        "does this thread exist" question is answered by ``thread/resume`` on the
        first turn, before any model call and therefore for free -- so the
        paragraph below describes exec, which is where the question has no cheap
        answer at all.

        **The check happens on the first turn, not here**, and that is the
        honest half of the choice the contract offers. ``codex exec`` has no
        token-free "does this thread exist" call: the id could be checked
        against ``~/.codex/sessions/YYYY/MM/DD/rollout-*-<uuid>.jsonl``, but
        that is a private on-disk layout the CLI is mid-migration on
        (``codex migrate-rollouts``), and a wrong *negative* there would refuse
        a session that resumes perfectly well. The first ``send`` finds out
        instead, and finds out for free -- a rejected resume fails before any
        model call, spending nothing, and raises
        :class:`~modelpass.errors.SessionNotFound` with the CLI's own words.

        A system prompt is refused here rather than applied. It belongs to the
        front of a conversation that already exists, so applying it now would
        inject instructions mid-thread -- ``midconversation_system``, which
        neither runtime supports -- and ignoring it would silently drop
        something the caller asked for. **This holds on app-server even though
        ``ThreadResumeParams`` has a ``baseInstructions`` field** (schema-sourced
        2026-08-31): the rollout already replays the prompt the thread was opened
        with, so a second one would be a persona arriving in the middle rather
        than the original being restored. That is the same answer
        ``resume_carries_system_prompt`` gives for this runtime, and it is a
        statement about the rollout, not about the transport.

        A *model* may differ, and the runtime says so rather than refusing:
        resuming with a name other than the one the thread was recorded with
        emits an advisory ``item.completed`` of type ``error`` -- *"This session
        was recorded with model X but is resuming with Y"* -- and carries on
        (observed live, 0.151.0). It reaches the caller as a ``vendor_event``,
        which is where a warning modelpass did not generate belongs (rule 3).
        """
        transport = resolve_transport(request.options)
        self._refuse_unsupported_session(request, transport)
        if request.system_prompt is not None:
            raise InvalidSession(
                "a resumed Codex session cannot take a system_prompt: this runtime "
                "can only layer one above the first turn's task, and that turn is "
                "already in the thread. Resuming with one would put instructions "
                "into the middle of a conversation, which no runtime modelpass drives "
                "supports. Omit it -- the session still has the one it was opened with"
            )
        binary = self._resolve_binary(request)
        if transport == TRANSPORT_APP_SERVER:
            return CodexAppServerSession(
                adapter=self,
                request=request,
                binary=binary,
                session_id=request.resume_id,
            )
        return CodexSession(
            adapter=self,
            request=request,
            binary=binary,
            session_id=request.resume_id,
        )

    def list_sessions(self, request: SessionRequest) -> tuple[SessionInfo, ...]:
        """Enumerate threads on the app-server transport; refuse on exec.

        **The transport is the whole answer here** (S5, 2026-08-31), which is why
        the two halves read so differently.

        On ``codex app-server`` this is ``thread/list``: paginated, token-free,
        driven live on 2026-08-31 with twenty-five real rows behind
        :func:`~modelpass.adapters.codex_appserver.session_info_from_thread`. It
        spawns a child, asks one question and closes it -- no model call, nothing
        billed. The listing is **account-wide**, not scoped to
        ``request.project_folder``: the store is flat, the capture's rows span
        seven working directories, and
        :func:`~modelpass.adapters.codex_appserver.thread_list_params` records why
        the schema's ``cwd`` filter is left alone.

        On ``codex exec`` it is refused, and the refusal is repeated here rather
        than left to the bridge's gate because the alternative -- returning
        ``()`` -- reads as *this connection has no sessions*, which is a different
        and false statement about an account that may have hundreds. ``codex exec
        --help`` offers only ``resume [SESSION_ID]`` with ``--last`` / ``--all``,
        and ``codex resume`` is an interactive picker.

        The registry cell now says ``supported`` (S7, 2026-08-31), because it
        describes the **default** transport and the default is app-server;
        :meth:`support_for` is what answers for a request that opted out into
        exec, where the refusal below still stands.
        """
        if resolve_transport(request.options) != TRANSPORT_APP_SERVER:
            raise CapabilityNotSupported(
                self.runtime.value,
                "sessions_list",
                detail=(
                    "codex exec offers no scriptable thread listing -- only 'resume "
                    "[SESSION_ID]' and an interactive picker. This call selected exec "
                    "with options={'transport': 'exec'}; the DEFAULT transport "
                    "enumerates threads (thread/list, driven live 2026-08-31, "
                    "token-free), so removing that option is the whole fix. Or keep "
                    "the id from session.id rather than expecting to enumerate them"
                ),
            )
        binary = self._resolve_binary(request)
        if binary is None:
            raise VendorRunFailed("codex CLI not found")
        client = AppServerClient(
            binary=binary,
            env=dict(request.plan.env),
            cwd=request.project_folder,
            spawn=self._app_server_spawn,
        )
        try:
            client.start()
            result = client.request(THREAD_LIST_METHOD, thread_list_params())
        finally:
            # One question, then the child goes: a listing that left a codex
            # process running would be a stray holding somebody's rollout open.
            client.close()
        return tuple(
            session_info_from_thread(thread, connection=request.connection.name)
            for thread in threads_from_list(result)
        )

    def _refuse_unsupported_session(
        self, request: SessionRequest, transport: str = DEFAULT_TRANSPORT
    ) -> None:
        """The session shapes this runtime cannot honour, per transport.

        All of these are already gated by the bridge, so reaching one means a
        caller built the request themselves. Refusing anyway is the difference
        between an object that cannot do what it says and an exception that names
        why -- and each message names the transport, because since S5 two of the
        three answers depend on it.

        The default argument follows :data:`DEFAULT_TRANSPORT` rather than
        naming a transport, so a direct caller who passes no transport is
        refused for the same reasons the bridge would refuse them (S7,
        2026-08-31).
        """
        app_server = transport == TRANSPORT_APP_SERVER
        if request.tools and not app_server:
            raise CapabilityNotSupported(
                self.runtime.value,
                "tools_in_process",
                detail=(
                    "a session on this transport is a 'codex exec resume' thread, "
                    "and 'codex exec' has no in-process tool registration -- its "
                    "only tool channel is MCP, over a command line. This session "
                    "selected exec with options={'transport': 'exec'}; the DEFAULT "
                    "transport runs caller functions in-process, sessions included "
                    "since 2026-08-31, so removing that option is the whole fix. "
                    "Or stay on exec and use mcp_servers= with a server the caller "
                    "runs"
                ),
            )
        if request.mcp_servers and app_server:
            raise CapabilityNotSupported(
                self.runtime.value,
                "mcp_servers",
                detail=(
                    "the app-server transport does not carry per-session MCP "
                    "servers yet: exec declares them with '-c mcp_servers.<name>="
                    "{...}' and builds exclusivity from 'codex mcp list --json', "
                    "and neither has an equivalent on this path that has been "
                    "driven. Refused rather than run without the servers, or "
                    "without the exclusivity guarantee. app-server is the default "
                    "since 2026-08-31, so this is one option away: "
                    "options={'transport': 'exec'}"
                ),
            )
        if not request.persist:
            raise CapabilityNotSupported(
                self.runtime.value,
                "ephemeral_multi_turn",
                detail=(
                    "every exec turn is a separate 'codex exec' process and "
                    "continuation needs the rollout file, so persist=False there is "
                    "not an ephemeral conversation -- it is a series of one-shots "
                    "wearing a session's name. On the app-server transport it is a "
                    "real capability that modelpass has not built yet: two turns on "
                    "one 'ephemeral: true' thread were driven live 2026-08-31 (the "
                    "second recalled a number given only in the first), but a "
                    "session here always starts a persisting thread. Refused rather "
                    "than silently persisting a conversation you asked to leave no "
                    "trace of. Use persist=True, or bridge.chat()"
                ),
            )

    def cancel(self) -> None:
        """D10 floor, as characterized by the 2026-08-16 experiment: terminate the
        process; no terminal event will arrive from the runtime, so ``run()``
        synthesizes the ``cancelled`` terminal.

        Stateless runs only. A session turn is cancelled by closing the iterator
        :meth:`CodexSession.send` returned, which is the same transport-level
        floor reached through the same ``finally``; one adapter instance backs
        every session on its runtime, so a shared ``cancel()`` here would kill
        whichever turn happened to be running.

        **On the app-server transport there is a politer step to try first**, and
        it is tried: ``turn/interrupt`` asks the server to stop the turn before
        the child is closed under it. It is attempted rather than relied on, and
        the reason narrowed on 2026-08-31: the method itself is driven (see
        :func:`interrupt_app_server_turn`), but this path does not wait for the
        ``turn/completed`` it produces -- the close follows regardless and the
        cancelled terminal is synthesized exactly as it is on exec. Nothing
        claims ``graceful_cancel`` on this runtime until modelpass's half is
        checked too."""
        self._cancelled = True
        client, self._client = self._client, None
        turn, self._app_turn = self._app_turn, None
        if client is not None:
            interrupt_app_server_turn(
                client,
                turn.thread_id if turn is not None else None,
                turn.turn_id if turn is not None else None,
            )
            client.close()
        proc, self._proc = self._proc, None
        self._terminate(proc)

    @staticmethod
    def _terminate(proc: Any) -> None:
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        except OSError:
            pass
