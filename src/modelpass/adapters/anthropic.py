"""Anthropic runtime adapter (``anthropic-sdk``) -- ROADMAP Phase 3.

Drives ``claude-agent-sdk`` under a Claude subscription login. Verified against
``claude-agent-sdk`` 0.2.139 / Claude Code 2.1.63 on 2026-08-16, which found the
installed dataclasses drifted well away from the vendor's own reference page --
dispatch on the class, not on a ``type`` attribute; ``cli_path``, not
``path_to_claude_code_executable``; ``StreamEvent.event`` a dict. Read the
installed package, never the docs page, when changing anything here.
``docs/api-and-runtimes.md`` §2.2a is the published record of that pass.

Three things in here are load-bearing and worth reading before changing:

1. **The scrub cannot go through ``ClaudeAgentOptions.env``.** The SDK builds
   the child environment as ``{**os.environ, ..., **options.env}``, so ``env``
   can only *add* variables -- it can never remove one. An ambient
   ``ANTHROPIC_API_KEY`` would survive and, because subscription OAuth resolves
   *last* in Claude Code's precedence, silently win and move the run onto
   metered billing. :func:`_scrubbed_process_env` therefore removes the planned
   names from ``os.environ`` for the duration of the spawn, under a lock, and
   restores them afterwards. See the deviation note in that function.
2. **Subscription connections fail closed.** If the preflight cannot confirm a
   usable subscription credential, it reports ``ok=False`` and the bridge raises
   rather than letting the runtime fall through to an API key.
3. **Money never enters the normalized stream.** ``total_cost_usd`` rides in a
   ``vendor_event`` and nowhere else (D7).

Phase 6 (D12) adds tools, verified against the same SDK build on 2026-08-16
(``docs/api-and-runtimes.md`` §2.2a). Two more things worth reading first:

4. **``max_turns`` is the only thing tools change about statelessness.** A run
   with no tools keeps ``max_turns=1`` exactly as Phase 3 left it -- that is not
   a budget, it is the definition of a single chat response. A run *with* tools
   removes the ceiling (``max_turns=None``, the vendor default: run to
   completion), because ``1`` forbids a tool result from ever coming back and
   any other number would be invented. The spend bound is the guard system's
   job, in tokens; a caller wanting a turn ceiling passes
   ``options={"max_turns": N}``. The run is still one ephemeral session with
   the message list serialized in. See :func:`_tool_options`.
5. **Built-in tools are off, by construction rather than by list.** ``tools=[]``
   is the SDK's documented "remove every built-in" switch, and the transport
   passes an empty list through as ``--tools ""`` rather than dropping it. The
   explicit ``disallowed_tools`` list is kept as a second lock, but ``tools=[]``
   is what survives Claude Code growing a tool nobody here has heard of.

Phase 8 (D13) adds structured output, and this runtime has a **native**
mechanism -- verified live on 2026-08-17 against the same SDK build
(``docs/api-and-runtimes.md`` §2.2a), with the message sequence committed as a
fixture. Two things about it are load-bearing:

6. **``ClaudeAgentOptions.output_format`` is the mechanism, not a submission
   tool modelpass builds.** ``{"type": "json_schema", "schema": {...}}`` becomes
   ``--json-schema <json>`` on the CLI, and the answer comes back on
   ``ResultMessage.structured_output`` (parsed) with ``.result`` carrying the
   same JSON as text. The **loose** schema form is accepted -- optional
   properties absent from ``required``, no ``additionalProperties`` -- which is
   exactly where this runtime differs from Codex.
7. **The runtime implements it as an end-turn tool called ``StructuredOutput``**,
   which appears in the init ``tools`` list *even with* ``tools=[]``, and shows
   up mid-stream as a ``ToolUseBlock`` plus a matching ``ToolResultBlock``. Those
   are the runtime's own plumbing, not a tool the caller declared, so
   :func:`map_message` routes them to ``vendor_event`` rather than emitting
   ``tool_call`` / ``tool_result`` -- reporting them as tool events would mean a
   caller who declared no tools sees tool traffic, which is precisely the
   surprise D12 promised not to spring. Nothing is dropped (adapter rule 3).
   ``num_turns`` comes back as 2 under ``max_turns=1`` and the terminal reason is
   ``completed``, so the ceiling does not need moving.

Phase 9 (D14-D17) adds sessions. The stateless path above is untouched: a
session is a *second* way to drive the same SDK, not a rewrite of the first.
Five things in that half are load-bearing:

8. **A session is one live ``ClaudeSDKClient``, held across turns.** ``run()``
   spawns a client, drives one exchange and lets the loop die with it; a session
   keeps its own event loop on its own thread for the object's whole life and
   calls ``query()`` once per turn (verified live 2026-08-30: stable
   ``session_id``, input growing 213 -> 260, correct recall of the earlier turn).
   That is what makes history the runtime's rather than something modelpass
   flattens back into a prompt.
9. **``persist=False`` is a real mode here, and it is the env var that makes it
   one.** ``CLAUDE_CODE_SKIP_PROMPT_HISTORY=1`` stops the transcript being
   written while the live client still carries the conversation -- proven by a
   control run *without* the variable, which did write a file. So ephemeral
   multi-turn is held in the subprocess, not faked by re-sending. Note that
   :meth:`AnthropicAdapter.run` sets that variable *unconditionally*, because a
   stateless call has nothing to resume; a ``persist=True`` session must not
   inherit it, which is why the session path builds its own additions rather
   than reusing the run path's.
10. **The working directory is the storage key, so a session always sets one.**
    ``cwd=request.project_folder`` maps a session onto
    ``~/.claude/projects/<encoded-cwd>/<id>.jsonl``, which is also what makes
    :meth:`AnthropicAdapter.list_sessions` directory-scoped. Without it a
    modelpass session would write its transcript into the *caller's* own project
    history, where their ``claude --continue`` would find it -- the surprise the
    core's scratch-directory default exists to prevent.
11. **The two session faces map onto two different ``system_prompt`` forms, and
    neither maps onto omission.** See :func:`session_system_prompt`: a chat gets
    a custom string (replace), a worker gets the ``claude_code`` preset with the
    caller's text appended, and omitting the option entirely would give the
    *minimal* tool-calling prompt -- a full toolbelt with no guidance for it.
12. **TTL pinning is version-gated and stays off where it would be ignored.**
    ``CLAUDE_CODE_PROMPT_CACHE_TTL`` is read from Claude Code v2.1.242 onward;
    older builds do not know the name. modelpass detects the resolved CLI's
    version and only sets it where it lands (:func:`supports_prompt_cache_ttl`),
    because setting an ignored variable and reporting the TTL as pinned is
    precisely the silent failure this library exists to refuse.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import inspect
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..connections import CredentialKind
from ..errors import (
    AuthModeMismatch,
    CapabilityNotSupported,
    InvalidSession,
    RuntimeNotAvailable,
    SessionNotFound,
    SubpassError,
    VendorRunFailed,
)
from ..preflight import (
    AccountProfile,
    CacheEligibility,
    PreflightPlan,
    Receipt,
    cache_floor_tokens,
    check_launch_args,
    env_names_to_scrub,
)
from ..reasoning import OPTION_THINKING, ReasoningPlan, stated_reasoning
from ..runtimes import Runtime
from ..schema import build_structured_event
from ..tools import ToolDef
from ..types import (
    CALLER_TOOL_SERVER,
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
    UsageScope,
    VendorEvent,
)
from ._mcp_result import mcp_result_payload
from .base import Adapter, RunRequest, SessionHandle, SessionRequest

__all__ = ["AnthropicAdapter", "AnthropicSessionHandle"]

_MODULE = "claude_agent_sdk"
_EXTRA = "anthropic"
_PACKAGE = "claude-agent-sdk"

#: Serializes the ``os.environ`` mutation in :func:`_scrubbed_process_env`.
_ENV_LOCK = threading.RLock()

#: How long a ``claude auth status`` answer is reused for, in seconds.
#:
#: The probe costs ~2.4s of subprocess, and *every* run, session open and bench
#: page goes through a preflight that wants it. Unlike ``claude --version`` the
#: answer can genuinely change while the process lives -- a user runs
#: ``claude /login`` in another terminal -- so this expires rather than being
#: held for the adapter's life, and :meth:`AnthropicAdapter.invalidate_identity_cache`
#: drops it outright for the one command whose whole job is to re-read it.
_IDENTITY_CACHE_TTL_SECONDS = 60.0

#: Set in the child so a stateless call writes no prompt-history transcript.
#: ``persistSession: false`` is the TypeScript spelling; Python opts out through
#: the environment (verification doc, "Sessions").
_SKIP_HISTORY_ENV = "CLAUDE_CODE_SKIP_PROMPT_HISTORY"

#: ``ResultMessage.terminal_reason`` values that mean "this turn was cancelled".
_CANCELLED_REASONS = frozenset({"aborted_streaming", "aborted_tools"})

#: ``ResultMessage.subtype`` values that are a clean, budget-driven stop rather
#: than a failure.
_BUDGET_SUBTYPES = frozenset({"error_max_budget_usd"})

#: Stated on every subscription receipt, because the alternative is a receipt
#: that is silent about the one thing it cannot tell you. The Agent SDK exposes
#: no pre-run credit or allowance query -- checked against 0.2.139's full public
#: surface -- and ``RateLimitEvent`` reports the allowance only once a run is
#: already under way. Saying "how much is left is not knowable before the run"
#: is the honest form of the ROADMAP's "allowance presence in the receipt".
_ALLOWANCE_NOTE = (
    "remaining allowance is not knowable before a run: the Agent SDK has no "
    "pre-run credit query, and the runtime reports allowance mid-run as "
    "rate_limit vendor events (status, utilization, resets_at)"
)


# --- credential detection ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    """What the Claude Code login store looks like -- metadata only.

    This type deliberately has no field that can hold a token. The detector
    reads the credential file for *expiry and plan* and nothing else, so a
    receipt can say "your login expired in March" without a secret ever entering
    a log, an event, or this process's memory beyond the parse.
    """

    source: str
    present: bool = False
    expires_at: float | None = None
    expired: bool = False
    refreshable: bool = True
    subscription_type: str | None = None
    rate_limit_tier: str | None = None
    detail: str | None = None
    #: True only on macOS when no ``.credentials.json`` exists and the answer
    #: therefore rests on the login Keychain, which modelpass does not open. The
    #: subscription receipt reads this to decide whether ``claude auth status``
    #: is the only evidence it has about which account a config directory holds.
    keychain_only: bool = False
    #: True when the credential file exists and parses but its token fields are
    #: blank -- what Claude Code leaves behind when the CLI is logged out: the
    #: plan metadata (``subscriptionType``, ``rateLimitTier``, ``scopes``) stays
    #: and the secrets are emptied. Indistinguishable from a missing file by
    #: :attr:`present` alone, and the two want the same instruction but not the
    #: same diagnosis -- a reader who sees a populated plan in the file and is
    #: told "no login found" reasonably concludes the token must live somewhere
    #: else, which on this platform it does not.
    logged_out: bool = False

    @property
    def usable(self) -> bool:
        """Whether this credential can plausibly authenticate a child process."""
        if not self.present:
            return False
        return not (self.expired and not self.refreshable)


def credentials_path(env: Mapping[str, str] | None = None) -> Path:
    """Where Claude Code keeps its login on this OS.

    ``CLAUDE_CONFIG_DIR`` relocates the whole config directory; honour it so a
    user with a non-default layout gets a truthful receipt.
    """
    source = os.environ if env is None else env
    configured = source.get("CLAUDE_CONFIG_DIR")
    root = Path(configured) if configured else Path.home() / ".claude"
    return root / ".credentials.json"


def read_credential_status(env: Mapping[str, str] | None = None) -> CredentialStatus:
    """Detect the Claude Code login without reading any secret value.

    The file is checked **first on every platform**, macOS included. Claude Code
    prefers the login Keychain there, but it falls back to the same plaintext
    ``<CLAUDE_CONFIG_DIR or ~/.claude>/.credentials.json`` whenever the Keychain
    refuses the write -- a locked keychain, an SSH session, a password out of
    sync (documented under "Authentication issues" in Claude Code's install
    troubleshooting). So on macOS the file may or may not exist, and when it
    does it is the same layout Linux writes and carries the same expiry.

    Only when there is no file does macOS fall back to the blind answer: the
    credential is present, and the Keychain entry it lives in is keyed to
    ``CLAUDE_CONFIG_DIR``, so a different directory is a different entry. modelpass
    does not open the Keychain, so it says that rather than claiming more.
    """
    path = credentials_path(env)
    source = f"Claude Code login ({path})"
    if not path.is_file():
        if sys.platform == "darwin":
            configured = (os.environ if env is None else env).get("CLAUDE_CONFIG_DIR")
            where = f", keyed to {configured}" if configured else ""
            return CredentialStatus(
                source=f"Claude Code login (macOS Keychain{where})",
                present=True,
                detail="keychain contents not inspected",
                keychain_only=True,
            )
        return CredentialStatus(source=source, present=False, detail="no credential file")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return CredentialStatus(
            source=source, present=True, detail=f"credential file unreadable ({type(exc).__name__})"
        )

    oauth = raw.get("claudeAiOauth") if isinstance(raw, Mapping) else None
    if not isinstance(oauth, Mapping):
        return CredentialStatus(
            source=source, present=True, detail="no claudeAiOauth entry in credential file"
        )

    expires_raw = oauth.get("expiresAt")
    expires_at: float | None = None
    # A zero or negative expiry is not a date. Claude Code writes ``0`` into
    # this field when it blanks the tokens, so reading it as a timestamp puts
    # the expiry at the epoch and every check downstream reports a login that
    # "expired in 1970" -- an artefact of the field being cleared, stated as a
    # fact about a token that was never there. No expiry recorded is `None`.
    if isinstance(expires_raw, (int, float)) and expires_raw > 0:
        # The file stores milliseconds; tolerate seconds from an older layout.
        expires_at = float(expires_raw) / 1000.0 if expires_raw > 1e11 else float(expires_raw)

    # Presence/length only -- the value itself is never read out of this scope.
    refreshable = bool(str(oauth.get("refreshToken") or ""))
    expired = expires_at is not None and expires_at <= time.time()
    present = bool(str(oauth.get("accessToken") or ""))

    return CredentialStatus(
        source=source,
        present=present,
        expires_at=expires_at,
        expired=expired,
        refreshable=refreshable,
        subscription_type=_as_str(oauth.get("subscriptionType")),
        rate_limit_tier=_as_str(oauth.get("rateLimitTier")),
        detail=None if present else "credential file holds no access token",
        logged_out=not present,
    )


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


# --- environment enforcement ---------------------------------------------------


@contextlib.contextmanager
def _scrubbed_process_env(plan: PreflightPlan) -> Iterator[tuple[str, ...]]:
    """Make ``os.environ`` match the preflight plan for the duration of a spawn.

    **Deviation from the adapter contract, recorded deliberately.**
    ``adapters/base.py`` says an adapter launches with ``request.plan.env`` and
    never touches ``os.environ``. That is not achievable through this SDK: the
    subprocess transport composes the child environment as
    ``{**os.environ, **options.env}``, so ``options.env`` is purely additive and
    a scrubbed dict passed there would leave the very variables we must remove
    in place. The only way to honour the plan is to take them out of
    ``os.environ`` across the spawn.

    Two consequences, both intentional:

    * The names removed are recomputed against the *live* ``os.environ`` rather
      than trusted from ``plan.scrubbed``, because the plan may have been built
      from an injected environment view. Scrubbing what is actually there is the
      only version of this that is a guarantee.
    * The mutation is serialized on a module lock and restored in ``finally``.
      A caller mutating ``os.environ`` from another thread during a spawn is
      still a race we cannot close from here; it is documented rather than
      hidden.

    Yields the names actually removed, so a caller (or a test) can assert on the
    guarantee rather than take it on trust.
    """
    live_names = env_names_to_scrub(plan.runtime, os.environ, preserve=plan.kept)
    remove = sorted(set(live_names) | set(plan.scrubbed))

    with _ENV_LOCK:
        saved: dict[str, str] = {}
        try:
            for name in remove:
                if name in os.environ:
                    saved[name] = os.environ[name]
                    del os.environ[name]
            yield tuple(saved)
        finally:
            for name, value in saved.items():
                os.environ[name] = value


def _contributed_args(request: RunRequest | SessionRequest) -> tuple[str, ...]:
    """Every CLI argument this run would add to the Claude Code command line.

    The SDK builds most of the command itself; the only place a caller can push
    a raw flag through is ``extra_args``. Those are checked against
    :data:`modelpass.preflight.FORBIDDEN_LAUNCH_ARGS` so that ``--bare`` -- which
    never reads OAuth credentials and forces an API key -- cannot be smuggled
    into a subscription run through ``options``.
    """
    extra = request.options.get("extra_args")
    if not isinstance(extra, Mapping):
        return ()
    args: list[str] = []
    for key, value in extra.items():
        flag = key if str(key).startswith("-") else f"--{key}"
        args.append(flag if value is None else f"{flag}={value}")
    return tuple(args)


def _explicit_env_additions(plan: PreflightPlan) -> dict[str, str]:
    """Variables explicitly selected by the connection, from the planned env.

    These must be present in ``ClaudeAgentOptions.env`` even when a different
    ambient value exists. The SDK merges options over ``os.environ``: scrubbing
    removes unselected values, while this mapping makes a selected configDir win.
    """
    additions: dict[str, str] = {}
    for name in plan.kept:
        value = plan.env.get(name)
        if value is not None:
            additions[name] = value
    return additions


# --- message serialization -----------------------------------------------------


def split_messages(messages: Sequence[Any]) -> tuple[str | None, str]:
    """Turn a modelpass message list into ``(system_prompt, prompt)``.

    v1 chat is stateless (D7): there is no session to carry history, so the
    history is serialized into the prompt. The common single-user-turn case is
    passed through verbatim -- wrapping "Reply with exactly OK" in a transcript
    envelope changes what the model is asked, so we do not.
    """
    system_parts = [m.flat_text for m in messages if m.role.value == "system"]
    turns = [m for m in messages if m.role.value != "system"]

    system_prompt = "\n\n".join(p for p in system_parts if p) or None

    if len(turns) == 1 and turns[0].role.value == "user":
        return system_prompt, turns[0].flat_text

    lines: list[str] = []
    for message in turns:
        label = "Human" if message.role.value == "user" else "Assistant"
        lines.append(f"{label}: {message.flat_text}")
    lines.append("Assistant:")
    return system_prompt, "\n\n".join(lines)


# --- event mapping -------------------------------------------------------------


def token_usage(usage: Mapping[str, Any] | None) -> TokenUsage:
    """Normalize the SDK's usage dict into modelpass tokens.

    Each of the three input kinds lands in its own field. Cache *writes* used to
    be folded into ``input_tokens`` -- billed as input, so a guard ignoring them
    would under-count -- but that made a cold prefix and genuinely new content
    indistinguishable in the run log, which is the one question a cache lever is
    for. ``TokenUsage.billable_input_tokens`` is the sum a guard wants;
    ``cache_write_tokens`` against ``cached_input_tokens`` is what tells a caller
    whether their prefix is stable. The untouched dict still passes through as a
    ``vendor_event``.

    **``output_tokens_details.thinking_tokens`` is read here and is the only
    thing on this runtime that says an effort setting did anything** (2026-09-22).
    The level goes out on ``ClaudeAgentOptions.effort`` and nothing comes back
    naming it -- driven on claude-code 2.1.278, where the string ``effort``
    appears in the whole ``stream-json`` transcript exactly once, as a
    *slash-command name*. What does come back is this count: one drive of a
    fixed prompt gave 993 thinking tokens of 996 output at ``xhigh`` against 365
    of 368 at ``low``. It is a subset of ``output_tokens``, so it is recorded as
    one.

    Note for anyone re-deriving this from the SDK: ``thinking_tokens`` is *not*
    in the typed surface. ``ResultMessage.usage`` is ``dict[str, Any]`` and the
    ``ModelUsage`` TypedDict -- which carries the same figure as
    ``thinkingTokens`` -- does not declare it either. The wire is ahead of the
    types here, which is why this cell rests on a live read.
    """
    if not isinstance(usage, Mapping):
        return TokenUsage()

    def count(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def reported(group: str, key: str) -> int | None:
        """``None`` where the vendor said nothing -- never ``0``, which is a claim."""
        inner = usage.get(group)
        value = inner.get(key) if isinstance(inner, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    return TokenUsage(
        input_tokens=count("input_tokens"),
        output_tokens=count("output_tokens"),
        cached_input_tokens=count("cache_read_input_tokens"),
        cache_write_tokens=count("cache_creation_input_tokens"),
        reasoning_output_tokens=reported("output_tokens_details", "thinking_tokens"),
    )


def _terminal_status(result: Any) -> tuple[TerminalStatus, str | None]:
    """Map a ``ResultMessage`` onto a terminal status and reason.

    ``subtype`` alone is not trustworthy: a 401 arrives as
    ``subtype='success'`` with ``is_error=True`` and ``api_error_status=401``,
    so ``is_error`` is checked before the subtype is believed.
    """
    subtype = getattr(result, "subtype", "") or ""
    terminal_reason = getattr(result, "terminal_reason", None)
    is_error = bool(getattr(result, "is_error", False))
    api_status = getattr(result, "api_error_status", None)

    if terminal_reason in _CANCELLED_REASONS:
        return TerminalStatus.CANCELLED, f"run cancelled ({terminal_reason})"

    if subtype in _BUDGET_SUBTYPES:
        return TerminalStatus.GUARD_STOP, "runtime budget limit reached (error_max_budget_usd)"

    if api_status == 429:
        return TerminalStatus.QUOTA_EXHAUSTED, "runtime reported HTTP 429 (rate limit / allowance)"

    if is_error or subtype.startswith("error"):
        detail = getattr(result, "result", None) or subtype or "unknown error"
        if api_status:
            detail = f"HTTP {api_status}: {detail}"
        return TerminalStatus.ERROR, str(detail)[:500]

    return TerminalStatus.OK, terminal_reason


def split_tool_name(qualified: str) -> tuple[str, str]:
    """Split ``mcp__<server>__<tool>`` into ``(server, tool)``.

    Anything that is not MCP-qualified is a built-in runtime tool, reported with
    server ``"builtin"`` -- which should not happen while ``tools=[]`` holds, and
    is therefore worth being visibly odd in the stream if it ever does.
    """
    if qualified.startswith("mcp__"):
        rest = qualified[len("mcp__") :]
        server, sep, tool = rest.partition("__")
        if sep and tool:
            return server, tool
    return "builtin", qualified


def flatten_tool_content(content: Any) -> str:
    """Flatten an MCP tool result into text.

    Result payloads are content-block arrays that may carry images and embedded
    resources. ``ToolResultEvent.content`` is text on purpose (see its
    docstring), so text blocks are joined and non-text blocks are named rather
    than dropped -- a caller reading only the normalized stream should be able
    to tell that an image came back, even though it cannot see the image.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return flatten_tool_content([content])
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            kind = _get(block, "type")
            text = _get(block, "text")
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(kind, str):
                parts.append(f"[{kind}]")
            else:
                parts.append(repr(block)[:200])
        return "\n".join(p for p in parts if p)
    return str(content)


def _get(obj: Any, attr: str) -> Any:
    """Read a field from either a mapping or an object. Blocks arrive as both."""
    if isinstance(obj, Mapping):
        return obj.get(attr)
    return getattr(obj, attr, None)


def _stream_event_to_events(event: Mapping[str, Any]) -> list[AgentEvent]:
    """Map one raw Anthropic stream event onto normalized events."""
    if event.get("type") != "content_block_delta":
        return [VendorEvent(Runtime.ANTHROPIC_SDK, f"stream.{event.get('type')}", dict(event))]

    delta = event.get("delta")
    if not isinstance(delta, Mapping):
        return [VendorEvent(Runtime.ANTHROPIC_SDK, "stream.content_block_delta", dict(event))]

    kind = delta.get("type")
    if kind == "text_delta":
        text = delta.get("text")
        return [TextDeltaEvent(text)] if isinstance(text, str) and text else []
    if kind == "thinking_delta":
        text = delta.get("thinking")
        return [ThinkingEvent(text)] if isinstance(text, str) and text else []
    return [VendorEvent(Runtime.ANTHROPIC_SDK, f"stream.{kind}", dict(event))]


#: The name the Claude runtime gives the end-turn tool it uses to submit
#: schema-bound output. Observed live 2026-08-17: it appears in the init
#: ``tools`` list even under ``tools=[]``, and its call/result blocks travel in
#: the ordinary assistant/user stream. modelpass recognizes it by name so that
#: runtime plumbing never reaches a caller as a ``tool_call``.
STRUCTURED_OUTPUT_TOOL = "StructuredOutput"


def map_message(
    message: Any,
    *,
    partial_text: bool = False,
    tool_names: MutableMapping[str, str] | None = None,
    schema: Mapping[str, Any] | None = None,
    schema_name: str = "",
) -> list[AgentEvent]:
    """Map one SDK message onto zero or more normalized events.

    A pure function of the vendor message, so the mapping is testable against
    recorded fixtures without a network, a credential, or the SDK installed.
    Anything outside the normalized vocabulary becomes a ``vendor_event`` --
    never a drop.

    ``partial_text`` says that token-level deltas are already arriving as
    ``StreamEvent``s, in which case the assembled ``AssistantMessage`` text is
    suppressed so callers do not receive every token twice.

    ``tool_names`` is the caller's correlation table, ``tool_use_id -> tool
    name``. Anthropic's ``ToolResultBlock`` carries only ``tool_use_id``, so
    naming the tool in a ``tool_result`` event requires remembering the call
    that produced it. Passing the table in rather than holding it in module
    state keeps this function pure with respect to its inputs and lets a test
    drive a whole call/result exchange by hand.

    ``schema`` says the caller asked for structured output (D13). It changes two
    things and nothing else: the runtime's own ``StructuredOutput`` tool traffic
    becomes ``vendor_event`` instead of ``tool_call`` / ``tool_result``, and the
    ``ResultMessage`` grows a ``structured_output`` event ahead of its terminal
    -- with the terminal turned into an ``error`` when no usable answer arrived.
    """
    runtime = Runtime.ANTHROPIC_SDK
    name = type(message).__name__
    events: list[AgentEvent] = []

    if name == "UserMessage":
        # The runtime replays tool results to the model as a user turn. Under
        # modelpass's stateless contract the *caller's* messages never come back
        # this way, so a UserMessage is tool traffic and nothing else.
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return []
        for block in content or []:
            if type(block).__name__ == "ToolResultBlock":
                if _is_structured_output_result(block, tool_names, schema):
                    events.append(
                        VendorEvent(runtime, "structured_output.result", _block_data(block))
                    )
                else:
                    events.append(_tool_result_event(block, tool_names))
            else:
                events.append(
                    VendorEvent(runtime, f"user.{type(block).__name__}", _block_data(block))
                )
        return events

    if name == "SystemMessage":
        subtype = getattr(message, "subtype", "") or "unknown"
        data = getattr(message, "data", {})
        return [VendorEvent(runtime, f"system.{subtype}", dict(data) if data else {})]

    if name == "AssistantMessage":
        error = getattr(message, "error", None)
        if error:
            events.append(
                VendorEvent(
                    runtime,
                    "assistant.error",
                    {"error": error, "stop_reason": getattr(message, "stop_reason", None)},
                )
            )
        for block in getattr(message, "content", []) or []:
            block_name = type(block).__name__
            if block_name == "TextBlock":
                if not partial_text and getattr(block, "text", ""):
                    events.append(TextDeltaEvent(block.text))
            elif block_name == "ThinkingBlock":
                if not partial_text and getattr(block, "thinking", ""):
                    events.append(ThinkingEvent(block.thinking))
            elif block_name == "ToolUseBlock":
                if schema is not None and _get(block, "name") == STRUCTURED_OUTPUT_TOOL:
                    # The runtime's own submission plumbing, not a caller tool.
                    call_id = _get(block, "id")
                    if tool_names is not None and isinstance(call_id, str) and call_id:
                        tool_names[call_id] = STRUCTURED_OUTPUT_TOOL
                    events.append(
                        VendorEvent(runtime, "structured_output.call", _block_data(block))
                    )
                else:
                    events.append(_tool_call_event(block, tool_names))
            else:
                events.append(
                    VendorEvent(runtime, f"assistant.{block_name}", _block_data(block))
                )
        # Interim usage: the cost of the turn that just finished, reported before
        # the next one starts (Phase 5). This is what lets a token guard stop a
        # tool loop *between rounds* rather than reading the bill afterwards --
        # which matters because Phase 6 removed the invented turn cap, so a tool
        # run is bounded by guards or by nothing. Verified against
        # claude-agent-sdk 0.2.139: the parser fills AssistantMessage.usage from
        # the API's per-message usage, so these are increments, not totals.
        interim = getattr(message, "usage", None)
        if isinstance(interim, Mapping):
            events.append(UsageEvent(usage=token_usage(interim), scope=UsageScope.DELTA))
        return events

    if name == "StreamEvent":
        event = getattr(message, "event", None)
        if isinstance(event, Mapping):
            return _stream_event_to_events(event)
        return [VendorEvent(runtime, "stream.unknown", {})]

    if name == "ResultMessage":
        raw_usage = getattr(message, "usage", None)
        # The runtime's own accounting for the whole run. Reported as a
        # ``run_total`` so it replaces the interim per-turn reports rather than
        # being added on top of the turns it already covers. Marking it this way
        # also makes the mapper safe against the one thing that could not be
        # settled offline -- whether this field is a run aggregate or the last
        # turn's -- because a run_total is folded in with a max, so either
        # reading produces the same (correct) figure.
        events.append(UsageEvent(usage=token_usage(raw_usage), scope=UsageScope.RUN_TOTAL))
        # Money leaves the normalized vocabulary here and only here (D7).
        events.append(
            VendorEvent(
                runtime,
                "result",
                {
                    "subtype": getattr(message, "subtype", None),
                    "is_error": getattr(message, "is_error", None),
                    "session_id": getattr(message, "session_id", None),
                    "terminal_reason": getattr(message, "terminal_reason", None),
                    "api_error_status": getattr(message, "api_error_status", None),
                    "num_turns": getattr(message, "num_turns", None),
                    "duration_ms": getattr(message, "duration_ms", None),
                    "total_cost_usd": getattr(message, "total_cost_usd", None),
                    "usage": dict(raw_usage) if isinstance(raw_usage, Mapping) else None,
                },
            )
        )
        status, reason = _terminal_status(message)
        if schema is not None:
            # The answer, then the outcome. One event, immediately before the
            # terminal, so a caller draining the stream has the structured
            # result in hand by the time it learns how the run ended.
            structured, failure = build_structured_event(
                schema,
                parsed=getattr(message, "structured_output", None),
                raw=getattr(message, "result", None),
                schema_name=schema_name,
            )
            if structured is not None:
                events.append(structured)
            if failure is not None and status is TerminalStatus.OK:
                # A run that was asked for a schema and produced nothing usable
                # did not succeed, whatever the runtime called it. Only an
                # otherwise-clean terminal is overridden: a quota stop or a
                # vendor error is the more important truth and keeps its status.
                status, reason = TerminalStatus.ERROR, failure
        events.append(_bare_terminal(status, reason))
        return events

    if name == "RateLimitEvent":
        return rate_limit_events(message)

    return [VendorEvent(runtime, f"message.{name}", {"repr": repr(message)[:500]})]


#: ``RateLimitInfo.status`` values. ``rejected`` is the allowance actually being
#: gone; ``allowed_warning`` is the runtime saying it is close.
_RATE_LIMIT_REJECTED = "rejected"
_RATE_LIMIT_WARNING = "allowed_warning"


def rate_limit_info_fields(info: Any) -> dict[str, Any]:
    """Everything ``RateLimitInfo`` carries, read defensively.

    Verified against ``claude-agent-sdk`` 0.2.139: ``status``, ``resets_at``,
    ``rate_limit_type``, ``utilization``, ``overage_status``,
    ``overage_resets_at``, ``overage_disabled_reason``, ``raw``. This is the
    only allowance signal either v1 runtime offers, and it exists **only during
    a run** -- there is no pre-run credit or allowance query on the Agent SDK,
    which is why the preflight says so rather than leaving the question open.
    """
    fields = (
        "status",
        "rate_limit_type",
        "resets_at",
        "utilization",
        "overage_status",
        "overage_resets_at",
        "overage_disabled_reason",
    )
    data: dict[str, Any] = {name: getattr(info, name, None) for name in fields}
    raw = getattr(info, "raw", None)
    if isinstance(raw, Mapping):
        data["raw"] = dict(raw)
    return data


def _allowance_phrase(data: Mapping[str, Any]) -> str:
    """The human part of a quota reason: which window, how full, when it resets."""
    parts: list[str] = []
    if data.get("rate_limit_type"):
        parts.append(str(data["rate_limit_type"]))
    utilization = data.get("utilization")
    if isinstance(utilization, (int, float)) and not isinstance(utilization, bool):
        parts.append(f"{utilization * 100:.0f}% used")
    resets_at = data.get("resets_at")
    if isinstance(resets_at, (int, float)) and not isinstance(resets_at, bool):
        parts.append(f"resets at {_format_timestamp(float(resets_at))}")
    if data.get("overage_disabled_reason"):
        parts.append(f"overage unavailable: {data['overage_disabled_reason']}")
    return ", ".join(parts)


def rate_limit_events(message: Any) -> list[AgentEvent]:
    """Map a ``RateLimitEvent`` onto normalized events.

    ``rejected`` is the vendor's own statement that the subscription allowance is
    gone, and is the signal Phase 5 normalizes to a ``quota_exhausted`` terminal:
    a *clean stop*, not an error, because nothing failed -- the plan ran out
    (D4). ``allowed_warning`` is passed through as information and deliberately
    does **not** become a ``guard_warning``: guard events mean a threshold the
    user configured was crossed, and quietly minting one from a vendor signal
    would make the guard vocabulary mean two different things.
    """
    runtime = Runtime.ANTHROPIC_SDK
    info = getattr(message, "rate_limit_info", None)
    data = rate_limit_info_fields(info)
    events: list[AgentEvent] = [VendorEvent(runtime, "rate_limit", data)]

    status = data.get("status")
    if status == _RATE_LIMIT_REJECTED:
        detail = _allowance_phrase(data)
        events.append(
            _bare_terminal(
                TerminalStatus.QUOTA_EXHAUSTED,
                "subscription allowance exhausted"
                + (f" ({detail})" if detail else ""),
            )
        )
    return events


#: ``apiKeySource`` values on the init ``SystemMessage`` that mean "no API key
#: was used". Anything else is the runtime telling us it found metered credentials.
_NO_API_KEY_SOURCES = frozenset({"none", "", "/login managed key"})


def assert_expected_auth_mode(message: Any, expected: AuthMode) -> None:
    """Fail closed if the runtime says it authenticated differently than asked.

    The init ``SystemMessage`` carries ``apiKeySource``: the runtime's own
    account of which credential it resolved. That is a far stronger signal than
    anything modelpass can infer beforehand, so it is checked on every run. A
    subscription connection that finds an API key in play raises rather than
    streams -- the whole point of the library is that this cannot happen quietly.
    """
    if type(message).__name__ != "SystemMessage":
        return
    if getattr(message, "subtype", None) != "init":
        return
    data = getattr(message, "data", None)
    if not isinstance(data, Mapping):
        return
    source = data.get("apiKeySource")
    if not isinstance(source, str):
        return

    used_api_key = source.strip().lower() not in _NO_API_KEY_SOURCES
    if expected is AuthMode.SUBSCRIPTION and used_api_key:
        raise AuthModeMismatch(
            AuthMode.SUBSCRIPTION.value, f"api_key (apiKeySource={source!r})"
        )
    if expected is AuthMode.API_KEY and not used_api_key:
        raise AuthModeMismatch(
            AuthMode.API_KEY.value, f"subscription (apiKeySource={source!r})"
        )


def _tool_call_event(
    block: Any, tool_names: MutableMapping[str, str] | None
) -> ToolCallEvent:
    """Map a ``ToolUseBlock`` (``id``, ``name``, ``input``) onto ``tool_call``."""
    qualified = _get(block, "name")
    qualified = qualified if isinstance(qualified, str) else ""
    server, tool = split_tool_name(qualified)
    call_id = _get(block, "id")
    call_id = call_id if isinstance(call_id, str) else ""
    arguments = _get(block, "input")
    if tool_names is not None and call_id:
        tool_names[call_id] = tool
    return ToolCallEvent(
        name=tool,
        arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
        id=call_id,
        server=server,
    )


def _tool_result_event(
    block: Any, tool_names: MutableMapping[str, str] | None
) -> ToolResultEvent:
    """Map a ``ToolResultBlock`` (``tool_use_id``, ``content``, ``is_error``)."""
    call_id = _get(block, "tool_use_id")
    call_id = call_id if isinstance(call_id, str) else ""
    return ToolResultEvent(
        id=call_id,
        # Empty rather than guessed when the call was never seen -- see the
        # ToolResultEvent docstring.
        name=(tool_names or {}).get(call_id, ""),
        content=flatten_tool_content(_get(block, "content")),
        is_error=bool(_get(block, "is_error")),
    )


def _is_structured_output_result(
    block: Any, tool_names: Mapping[str, str] | None, schema: Mapping[str, Any] | None
) -> bool:
    """Whether a ``ToolResultBlock`` answers the runtime's own submission tool.

    Correlated by ``tool_use_id`` rather than matched on the result's text,
    because the text (*"Structured output provided successfully"*) is the
    runtime's wording and could change; the id is the protocol.
    """
    if schema is None or not tool_names:
        return False
    call_id = _get(block, "tool_use_id")
    return (
        isinstance(call_id, str)
        and tool_names.get(call_id) == STRUCTURED_OUTPUT_TOOL
    )


def _block_data(block: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for attr in ("id", "name", "input", "tool_use_id", "content", "is_error", "signature"):
        if hasattr(block, attr):
            value = getattr(block, attr)
            simple = isinstance(value, (str, int, float, bool, type(None)))
            data[attr] = value if simple else repr(value)
    return data


def classify_pump_failure(exc: BaseException) -> BaseException:
    """Decide what an exception out of the SDK drive loop actually means.

    **The specific leak this closes** (first-consumer feedback, 2026-08-17). The
    Agent SDK reports some failures as a ``ResultMessage`` -- which maps to a
    terminal event, correctly -- and others by *raising* out of its async
    iterator: a rejected model, a CLI that exits non-zero, a transport that
    dies. Those used to be re-raised verbatim, wrapped by the bridge into
    ``AdapterFailed``, and delivered mid-iteration. The stream then had no
    terminal event at all, so a caller lost the D3 stamp and the accumulated
    usage of a run that genuinely spent tokens. It was the single most likely
    way for a real run to fail, and it was the one path that broke the contract.

    The rule, chosen so it can be checked rather than guessed at: **an exception
    whose type belongs to the vendor package is the vendor reporting a failed
    run**; anything else escaping is modelpass's own bug and keeps its old
    treatment. Testing the defining module rather than a list of SDK exception
    classes means the SDK can add one without opening the hole again -- and does
    not require importing the SDK to classify.

    ``SubpassError`` passes through untouched, which is how
    :func:`assert_expected_auth_mode`'s refusal keeps raising: that is a
    guaranteed-layer check (D4 layer 1), not a run outcome.
    """
    if isinstance(exc, SubpassError) or not isinstance(exc, Exception):
        return exc
    root = (type(exc).__module__ or "").split(".")[0]
    if root == _MODULE:
        return VendorRunFailed(
            f"the {_PACKAGE} runtime reported a failed run: "
            f"{type(exc).__name__}: {exc}"
        )
    return exc


def _bare_terminal(status: TerminalStatus, reason: str | None) -> TerminalEvent:
    """A terminal carrying a *status* only.

    The connection, runtime and auth-mode fields are placeholders: the bridge
    overwrites all three, and an adapter is not allowed to claim how a run was
    billed (adapter contract, rule 2).
    """
    return TerminalEvent(
        status=status,
        connection="",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        reason=reason,
    )


# --- caller tools as an in-process MCP server (D12) ------------------------------


#: Whatever a caller's handler returned, as an MCP tool result. Hoisted to
#: ``adapters/_mcp_result`` in S6 of the app-server migration, when the Codex
#: dynamic-tool dispatch needed the identical reading of the same
#: :class:`~modelpass.tools.ToolDef` contract. Kept under the old private name so
#: this module reads as it did; the definition is shared.
_tool_result_payload = mcp_result_payload


def _wrap_handler(tool: ToolDef) -> Any:
    """Adapt a caller's handler to the SDK's ``async (args) -> result`` contract.

    Two things happen here that are worth being explicit about:

    * **A sync handler runs off the event loop.** The SDK hosts the tool server
      inside the same loop that pumps the message stream, so a blocking handler
      -- a database call, a file read, anything a real caller tool does -- would
      stall the stream it is part of. ``asyncio.to_thread`` keeps the loop free.
    * **Exceptions become failed tool results, not run failures.** The model can
      act on a failed tool: retry it, try another, or explain. Killing the run
      instead throws away the turns already paid for. The exception's type and
      message reach the model, which is documented on :class:`ToolDef` so a
      caller knows not to raise anything carrying a secret.
    """
    handler = tool.handler
    is_async = handler is not None and inspect.iscoroutinefunction(handler)

    async def invoke(args: Any) -> dict[str, Any]:
        arguments = dict(args) if isinstance(args, Mapping) else {}
        if handler is None:
            return {
                "content": [
                    {"type": "text", "text": f"tool {tool.name!r} has no handler"}
                ],
                "is_error": True,
            }
        try:
            result = (
                await handler(arguments)
                if is_async
                else await asyncio.to_thread(handler, arguments)
            )
        except Exception as exc:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"tool {tool.name!r} failed: {type(exc).__name__}: {exc}",
                    }
                ],
                "is_error": True,
            }
        return _tool_result_payload(result)

    return invoke


def build_caller_tool_server(sdk: Any, tools: Sequence[ToolDef], name: str) -> Any:
    """Wrap caller tools as an in-process MCP server the runtime can call.

    ``sdk.tool(...)`` is a decorator factory; the handler it decorates must be
    async, so each :class:`ToolDef` goes through :func:`_wrap_handler` first.
    Schemas are passed as full JSON Schema rather than the SDK's Python-type
    shorthand, which cannot express enums, ranges or optional fields.
    """
    wrapped = [
        sdk.tool(tool.name, tool.description, tool.json_schema())(_wrap_handler(tool))
        for tool in tools
    ]
    return sdk.create_sdk_mcp_server(name=name, version="1.0.0", tools=wrapped)


#: MCP server name for caller-supplied tools. Matches
#: :data:`modelpass.types.CALLER_TOOL_SERVER`, so ``ToolCallEvent.server`` reads
#: ``"caller"`` for exactly the tools the caller handed in.
_CALLER_SERVER = CALLER_TOOL_SERVER


def _reasoning_options(plan: ReasoningPlan) -> dict[str, Any]:
    """The ClaudeAgentOptions a resolved reasoning plan sets. One place, both
    paths (the stateless call and the session).

    A rung goes on ``effort``. ``none`` goes on ``thinking`` as
    ``{"type": "disabled"}`` (claude-agent-sdk 0.2.148 ``ThinkingConfigDisabled``,
    sent as ``--thinking disabled``; read 2026-09-23) and ``effort`` is left
    unset, because this runtime has no effort below ``low`` and sending one
    beside thinking-off would be two statements about one dial.
    """
    if plan.option == OPTION_THINKING:
        return {"thinking": {"type": plan.runtime_value}}
    return {"effort": plan.runtime_value}


def _tool_options(sdk: Any, request: RunRequest | SessionRequest) -> dict[str, Any]:
    """The options a tool-bearing run adds. Empty for a plain D7 chat call.

    Returning nothing when there are no tools is the guarantee that Phase 6 did
    not change Phase 3's behavior: a chat call still runs with ``max_turns=1``,
    no MCP config, and no allowed tools at all.

    Shared with sessions rather than copied, because a session's caller tools
    are the same tools under the same rules -- the two request types agree on
    every field this reads (``tools``, ``mcp_servers``, ``wants_tools``). What a
    session adds on top is the *runtime's own* toolbelt, which is a different
    question and lives in :func:`session_options`.
    """
    if not request.wants_tools:
        return {}

    servers: dict[str, Any] = dict(request.mcp_servers)
    if _CALLER_SERVER in servers:
        raise CapabilityNotSupported(
            Runtime.ANTHROPIC_SDK.value,
            "mcp_servers",
            f"the MCP server name {_CALLER_SERVER!r} is reserved for caller-supplied "
            "tools; rename the server",
        )
    if request.tools:
        servers[_CALLER_SERVER] = build_caller_tool_server(
            sdk, request.tools, _CALLER_SERVER
        )

    return {
        "mcp_servers": servers,
        "allowed_tools": allowed_tool_patterns(request.tools, request.mcp_servers),
        # Only the servers passed here are used: no .mcp.json, no user config.
        # setting_sources=[] already blocks the file sources; this closes the
        # question rather than relying on that side effect.
        "strict_mcp_config": True,
        # Vendor default: run to completion. No invented turn budget (D12
        # amendment, 2026-08-16) -- the spend bound is the guard system's job,
        # in tokens, which is the unit the user actually owns. A caller that
        # wants a turn ceiling sets options={"max_turns": N} explicitly.
        "max_turns": None,
    }


def allowed_tool_patterns(
    tools: Sequence[ToolDef], mcp_servers: Mapping[str, Any]
) -> list[str]:
    """The exact set of tools this run may use.

    Caller tools are listed individually rather than by wildcard: the wildcard
    would also admit anything else that ends up on that server, and the point of
    this list is that it enumerates what the caller asked for. Named MCP servers
    *do* get a wildcard, because their tool lists are the server's business and
    are not knowable here.
    """
    patterns = [f"mcp__{_CALLER_SERVER}__{tool.name}" for tool in tools]
    patterns.extend(f"mcp__{name}__*" for name in mcp_servers)
    return patterns


# --- sessions (D14-D17) ---------------------------------------------------------


#: Pins the prompt-cache TTL for a session's prefix. **Only read by Claude Code
#: v2.1.242 and later** -- see :func:`supports_prompt_cache_ttl` for why that is
#: detected rather than assumed.
_TTL_ENV = "CLAUDE_CODE_PROMPT_CACHE_TTL"

#: The TTL both session faces pin (D17). A session is the long-lived prefix
#: class, so it takes the hour rather than the five minutes Claude Code gives
#: its own short-lived helpers. The reason for pinning at all is that the
#: subscription default silently drops from 1h to 5m once an account exceeds
#: plan usage and starts drawing on credits; inheriting that means a prefix
#: whose lifetime changes underneath the caller without anything saying so.
_SESSION_TTL = "1h"

#: The first Claude Code build that reads :data:`_TTL_ENV`. Below this a string
#: scan of the binary finds no occurrence of the name (verified 2026-08-30
#: against 2.1.236, and again against the 2.1.233 binary bundled with
#: ``claude-agent-sdk`` 0.2.139).
_TTL_MIN_CLI_VERSION = (2, 1, 242)

#: Above this many characters a custom system prompt travels as a file rather
#: than as ``--system-prompt <text>``. See :func:`session_system_prompt`.
_SYSTEM_PROMPT_ARG_LIMIT = 8_000

#: How long an abandoned turn is given to stop after an interrupt before the
#: session declares itself unusable. Generous, because the alternative to
#: waiting is a queue nobody is draining and a loop thread blocked on it.
_ABANDON_DRAIN_SECONDS = 15.0

#: ``ClaudeAgentOptions`` fields a session owns outright, with the reason each
#: one is refused rather than silently overridden. All six decide *which
#: conversation this object is*, and a caller reaching around the constructor to
#: change that produces an object whose own properties describe something else --
#: ``session.project_folder`` naming a directory the runtime never used,
#: ``session.id`` naming a transcript nobody wrote to.
_SESSION_RESERVED_OPTIONS: Mapping[str, str] = {
    "cwd": (
        "the working directory is project_folder, and on this runtime it is also "
        "the key the transcript is stored under, so a session pointed somewhere "
        "else could not be listed or resumed by the folder it reports"
    ),
    "system_prompt": (
        "the system prompt is fixed at construction and sits at the front of the "
        "cached prefix (D17); pass it as system_prompt= on new_chat()/new_worker(), "
        "where the session's face decides whether it replaces or appends"
    ),
    "resume": (
        "resuming is bridge.resume_chat(session_id=...), which checks the session "
        "exists and lets the object report its id honestly"
    ),
    "continue_conversation": (
        "this would pick up whatever conversation was last run in this directory, "
        "which is not a session the caller named"
    ),
    "session_id": (
        "the runtime issues the id and modelpass reports what it issued; a supplied "
        "one would make session.id a claim rather than an observation"
    ),
    "fork_session": (
        "a fork is a different conversation from the one that was resumed, and "
        "the object would keep reporting the id it was opened with"
    ),
}


def parse_cli_version(text: str | None) -> tuple[int, ...] | None:
    """The numeric version out of ``claude --version`` output.

    ``"2.1.233 (Claude Code)"`` -> ``(2, 1, 233)``. ``None`` when the output is
    not something this can read, which is treated everywhere as *unknown*
    rather than as *old* or *new*.
    """
    if not text:
        return None
    words = text.strip().split()
    head = words[0] if words else ""
    numbers: list[int] = []
    for part in head.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        numbers.append(int(digits))
        if digits != part:
            # A pre-release or build suffix (``2.2.0-beta.1``). Everything after
            # it orders by rules this cannot check, so the numeric release is
            # where the reading stops rather than where it starts guessing.
            break
    return tuple(numbers) if numbers else None


def supports_prompt_cache_ttl(version: str | None) -> bool:
    """Whether this Claude Code build reads :data:`_TTL_ENV`.

    **Unknown means no**, and that asymmetry is the point. Setting a variable an
    older CLI ignores costs nothing at the runtime and everything at the
    receipt: modelpass would report a pinned 1h TTL while the build silently used
    whatever it felt like. Leaving the runtime default alone where the lever is
    not there is the honest half of D17's "set the TTL explicitly".
    """
    parsed = parse_cli_version(version)
    if parsed is None:
        return False
    return parsed >= _TTL_MIN_CLI_VERSION


def _default_claude_version(binary: str, env: Mapping[str, str]) -> str | None:
    """``claude --version`` for the resolved binary. Costs no tokens.

    Probed only for a concrete file on disk, for the reason the Codex adapter
    states: a receipt describing a launch should not itself launch something to
    find out what it was handed. ``None`` on anything unexpected -- an unnamed
    version loses a receipt line and the TTL pin, and neither is worth failing a
    preflight over.
    """
    if not os.path.isfile(binary):
        return None
    try:
        result = subprocess.run(
            [binary, "--version"],
            env=dict(env),
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
    lines = (result.stdout or "").strip().splitlines()
    return lines[0].strip() if lines and lines[0].strip() else None


VersionFn = Callable[[str, Mapping[str, str]], "str | None"]
AuthStatusFn = Callable[[str, Mapping[str, str]], "Mapping[str, Any] | None"]


def _default_auth_status(
    binary: str, env: Mapping[str, str]
) -> Mapping[str, Any] | None:
    """Run the documented ``claude auth status`` JSON probe. Costs no tokens."""
    if not os.path.isfile(binary):
        return None
    try:
        result = subprocess.run(
            [binary, "auth", "status"],
            env=dict(env),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        parsed = json.loads(result.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _account_profile_from_auth_status(
    data: Mapping[str, Any] | None,
) -> AccountProfile | None:
    """Normalize only non-secret identity fields from Claude's status JSON."""
    if data is None:
        return None
    logged_in = data.get("loggedIn")
    return AccountProfile(
        vendor="anthropic",
        source="claude auth status",
        logged_in=logged_in if isinstance(logged_in, bool) else None,
        auth_method=_as_str(data.get("authMethod")),
        api_provider=_as_str(data.get("apiProvider")),
        email=_as_str(data.get("email")),
        organization_id=_as_str(data.get("orgId")),
        organization_name=_as_str(data.get("orgName")),
        subscription_type=_as_str(data.get("subscriptionType")),
    )


def file_backed_system_prompt(text: str | None) -> tuple[Any, str | None]:
    """Return a prompt option safe to pass through the SDK's CLI transport.

    The SDK sends string system prompts as one command-line argument. Keep that
    established path for ordinary prompts, but use the runtime's file form once
    the prompt is large enough to make process creation unreliable (especially
    on Windows). The returned directory belongs to the caller and must remain
    alive until the client has spawned.
    """
    if text is None or len(text) <= _SYSTEM_PROMPT_ARG_LIMIT:
        return text, None

    directory = tempfile.mkdtemp(prefix="modelpass-anthropic-prompt-")
    path = Path(directory) / "system_prompt.txt"
    path.write_text(text, encoding="utf-8")
    return {"type": "file", "path": str(path)}, directory


def session_system_prompt(request: SessionRequest) -> tuple[Any, str | None]:
    """The ``system_prompt`` option for a session, and any temp directory it needs.

    Anthropic offers four forms and a session uses two of them. Which one is
    decided by :attr:`SessionRequest.appends_system_prompt`, never by whether
    the caller supplied text:

    * **chat** -- a bare string, which replaces the runtime's persona entirely.
      That is the whole point of the object, and it is also what gives a chat a
      machine-independent prefix: the preset embeds working directory, platform,
      shell, OS version and auto-memory paths, and a caller's own string embeds
      none of them.
    * **worker** -- ``{"type": "preset", "preset": "claude_code", "append": ...}``,
      with ``append`` omitted when the caller supplied no text. **Never
      omission**: leaving ``system_prompt`` off does not give Claude Code's
      instructions, it gives the minimal tool-calling ones, which is a full
      toolbelt and no guidance for using it.

    ``exclude_dynamic_sections`` rides on the preset form only. It moves the
    per-machine context out of the system prompt and into the first user
    message, so two machines running the same worker share cache entries instead
    of writing one apiece. It has no string equivalent and needs none, which is
    exactly why it belongs to this object and not the other. Verified reachable
    on ``claude-agent-sdk`` 0.2.139: it does not become a CLI flag but is sent on
    the control-protocol ``initialize`` request (``excludeDynamicSections``), and
    the bundled 2.1.233 binary knows that name. An older CLI ignores it, which
    costs a cross-machine cache entry and claims nothing -- unlike the TTL lever,
    nothing reports this one as applied.

    **The file form, and why the string one is not always it.** The SDK passes a
    string prompt as a single command-line argument, so a large prompt fails at
    process spawn with ``Argument list too long`` before any request is made.
    Over :data:`_SYSTEM_PROMPT_ARG_LIMIT` characters -- well under the tightest
    platform ceiling (Windows' 32,767-character command line, which also has to
    carry the rest of the launch) and well over any ordinary rubric -- the prompt
    is written to a temp file and passed as ``{"type": "file", "path": ...}``,
    the same pattern ``--output-schema`` already uses. Small prompts keep the
    long-standing ``--system-prompt`` path rather than depending on a newer flag
    for the common case.

    One limitation, stated rather than worked around: **the append form has no
    file equivalent.** ``--append-system-prompt`` takes its text as an argument
    and the SDK's file form *replaces* the preset rather than appending to it, so
    a worker whose appended text exceeds the OS limit fails at spawn and modelpass
    cannot fix it from here. Shorten it, or put the bulk in the project folder.

    Returns the option value and the temp directory that must outlive the
    session, or ``None`` when no file was written.
    """
    text = request.system_prompt

    if request.appends_system_prompt:
        preset: dict[str, Any] = {
            "type": "preset",
            "preset": "claude_code",
            "exclude_dynamic_sections": True,
        }
        if text is not None:
            preset["append"] = text
        return preset, None

    if text is None:
        # A chat with no prompt of its own asked for the runtime's persona to be
        # gone and put nothing in its place. The empty replacement is what says
        # that; the preset would say the opposite.
        return "", None
    return file_backed_system_prompt(text)


def session_options(
    sdk: Any, request: SessionRequest, *, env_additions: Mapping[str, str]
) -> tuple[dict[str, Any], str | None]:
    """The ``ClaudeAgentOptions`` kwargs for a session, and its prompt directory.

    Every field of :class:`SessionRequest` lands somewhere here:

    ==========================  ================================================
    ``project_folder``          ``cwd`` -- and therefore the transcript's
                                storage key and the scope of ``list_sessions``
    ``kind`` / ``native_tools``  ``tools``: ``[]`` for a chat, the
                                ``claude_code`` preset for a worker
    ``kind`` / ``appends_...``  ``system_prompt``, per
                                :func:`session_system_prompt`
    ``system_prompt``           the string, or the preset's ``append``
    ``model``                   ``model``
    ``persist``                 ``env``: ``CLAUDE_CODE_SKIP_PROMPT_HISTORY``
                                when ``False``, nothing when ``True``
    ``tools`` / ``mcp_servers``  ``mcp_servers`` / ``allowed_tools`` via
                                :func:`_tool_options`
    ``resume_id``               ``resume``, set by the adapter after this
    ``options``                 merged last, minus the six a session owns
    ==========================  ================================================

    Two settings are deliberately the same as the stateless path and are worth
    reading as decisions rather than copies:

    * ``setting_sources=[]`` on **both** faces, including a worker. It is what
      keeps ``settings.json`` out of the run, and ``apiKeyHelper`` lives there
      and outranks OAuth -- so this is an auth guarantee, not a preference, and
      a worker does not get to trade it for project settings.
    * ``permission_mode="default"``. modelpass does not widen a worker's
      permissions on the caller's behalf: handing an unattended agent write
      access to a named project folder is a decision its owner makes, in
      ``options={"permission_mode": ...}``, where it is visible.
    """
    prompt, prompt_dir = session_system_prompt(request)
    native_tools = request.native_tools

    options_kwargs: dict[str, Any] = {
        "system_prompt": prompt,
        "model": request.model,
        # The isolation the core's scratch-directory default depends on: without
        # it the runtime inherits the caller's process cwd and writes modelpass
        # transcripts into their own ~/.claude/projects/<their-cwd>/.
        "cwd": request.project_folder,
        "env": dict(env_additions),
        "setting_sources": [],
        "include_partial_messages": True,
        "permission_mode": "default",
    }
    if native_tools:
        # The documented "all default Claude Code tools" switch. Named rather
        # than left to the default so that the worker's toolbelt is a statement
        # in the options a reader can find, not the absence of one.
        options_kwargs["tools"] = {"type": "preset", "preset": "claude_code"}
        options_kwargs["allowed_tools"] = []
        options_kwargs["disallowed_tools"] = []
        # A toolbelt needs somewhere to go. The ceiling is the guard system's
        # job, in tokens (D12 amendment); a caller wanting a turn cap sets one.
        options_kwargs["max_turns"] = None
    else:
        options_kwargs["tools"] = []
        options_kwargs["allowed_tools"] = []
        options_kwargs["disallowed_tools"] = list(_BUILTIN_TOOLS)
        # One turn is one chat response, exactly as in the stateless path.
        options_kwargs["max_turns"] = 1
    options_kwargs.update(_tool_options(sdk, request))
    # The stated effort, in this runtime's own spelling (2026-09-21).
    # claude-agent-sdk 0.2.148 types ClaudeAgentOptions.effort as
    # Literal["low","medium","high","xhigh","max"]. This runtime is wired
    # directly rather than through sampling_rules because its
    # sampling_controls cell reads unsupported -- the sampling pipeline
    # carries nothing here, so the standing default would reach no wire.
    _effort = stated_reasoning(request.connection)
    if _effort is not None:
        options_kwargs.update(_reasoning_options(_effort))

    for key, value in request.options.items():
        reason = _SESSION_RESERVED_OPTIONS.get(key)
        if reason is not None:
            raise InvalidSession(f"options[{key!r}] is not available on a session: {reason}")
        if key == "env":
            # Merged *underneath* modelpass's own additions rather than replacing
            # them. The variables computed above carry this object's promises --
            # the preserved credential, "no transcript" for an ephemeral session,
            # the pinned TTL -- and a caller's env dict is for the levers
            # alongside them (DISABLE_AUTO_COMPACT and friends), not a way to
            # quietly cancel one.
            merged = dict(value) if isinstance(value, Mapping) else {}
            merged.update(options_kwargs["env"])
            options_kwargs["env"] = merged
            continue
        options_kwargs[key] = value

    return options_kwargs, prompt_dir


def session_messages_to_history(messages: Sequence[Any]) -> tuple[Message, ...]:
    """Map ``get_session_messages()`` output onto modelpass messages.

    Read-only introspection of what the runtime holds -- it cannot be fed back
    as input, which is the fact D14 is built on. Tool traffic in the transcript
    is flattened by :func:`flatten_tool_content`, so a replayed tool result
    reads as ``[tool_result]`` rather than vanishing; a message that flattens to
    nothing at all is dropped, because an empty turn is an artifact of the
    transcript's shape and not something the model was told.
    """
    history: list[Message] = []
    for entry in messages:
        kind = _get(entry, "type")
        role = Role.ASSISTANT if kind == "assistant" else Role.USER
        payload = _get(entry, "message")
        content = flatten_tool_content(_get(payload, "content"))
        if content:
            history.append(Message(role, content))
    return tuple(history)


def session_info_from_sdk(
    info: Any, *, connection: str, project_folder: str
) -> SessionInfo:
    """Map one ``SDKSessionInfo`` onto the normalized listing type.

    The normalized fields are the ones both runtimes can answer; everything the
    Agent SDK carries that Codex has no counterpart for rides in ``vendor``
    rather than being dropped or renamed into a shape it does not have.

    ``cwd`` is reported as ``project_folder`` when the transcript names one,
    because the session's own record of where it ran outranks the directory the
    listing was asked about.
    """
    cwd = _get(info, "cwd")
    return SessionInfo(
        id=str(_get(info, "session_id") or ""),
        connection=connection,
        runtime=Runtime.ANTHROPIC_SDK,
        project_folder=cwd if isinstance(cwd, str) and cwd else project_folder,
        created_at=_iso_from_millis(_get(info, "created_at")),
        updated_at=_iso_from_millis(_get(info, "last_modified")),
        title=_as_str(_get(info, "custom_title")) or _as_str(_get(info, "summary")),
        # Not reported: the listing is a stat plus a head/tail read by design,
        # and counting messages would mean parsing every transcript in the
        # folder. ``None`` means "not reported", which is the truth here.
        message_count=None,
        vendor={
            "summary": _get(info, "summary"),
            "first_prompt": _get(info, "first_prompt"),
            "git_branch": _get(info, "git_branch"),
            "tag": _get(info, "tag"),
            "file_size": _get(info, "file_size"),
        },
    )


def _iso_from_millis(value: Any) -> str | None:
    """ISO 8601 UTC from the SDK's epoch milliseconds, or ``None`` if absent.

    :class:`SessionInfo` carries timestamps as strings (D11), so this is where
    the conversion happens rather than in a caller who would have to guess the
    unit.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    import datetime as _dt

    try:
        return _dt.datetime.fromtimestamp(value / 1000.0, _dt.UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


#: Sentinel putting "this turn produced its last event" on the queue.
_TURN_DONE = object()


class AnthropicSessionHandle:
    """One live ``ClaudeSDKClient``, driven a turn at a time.

    **The client is the session.** It is connected once, in streaming mode with
    no initial prompt, and every turn is a ``query()`` against it followed by a
    drain of ``receive_response()``. Nothing is re-sent: the conversation lives
    in the subprocess and, when ``persist=True``, in the transcript the runtime
    writes beside it.

    The threading is the same shape as :meth:`AnthropicAdapter._pump` with one
    difference that matters: the loop outlives the turn. ``run()`` can afford
    ``asyncio.run()`` because its loop dies with the exchange; a session holds
    its loop on a dedicated thread until :meth:`close`, because that is what
    keeps the client -- and therefore the conversation -- alive between turns.

    Not thread-safe, by contract. One turn at a time; a second ``send`` while a
    turn is still streaming is refused rather than interleaved, because two
    ``query()`` calls into one client is two callers disagreeing about what the
    model was last told.
    """

    def __init__(
        self,
        *,
        sdk: Any,
        request: SessionRequest,
        options_kwargs: Mapping[str, Any],
        prompt_dir: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._sdk = sdk
        self._request = request
        self._options_kwargs = dict(options_kwargs)
        self._prompt_dir = prompt_dir
        self._partial_text = bool(self._options_kwargs.get("include_partial_messages", True))
        self._session_id = session_id
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Any = None
        self._closed = False
        self._streaming = False
        self._broken: str | None = None

    # --- identity ----------------------------------------------------------------

    @property
    def id(self) -> str | None:
        """The runtime's own session id, once a turn has produced one.

        ``None`` for the whole life of an ephemeral session, because nothing was
        written down: with ``CLAUDE_CODE_SKIP_PROMPT_HISTORY=1`` there is no
        transcript for that id to name, so handing it out would be handing over
        something no resume and no listing will ever find.
        """
        if not self._request.persist:
            return None
        return self._session_id

    # --- turns -------------------------------------------------------------------

    def send(self, message: str, *, effort: str | None = None) -> Iterator[AgentEvent]:
        """Run one turn against the live client and stream normalized events."""
        if effort is not None:
            # Not a gap this adapter can close. claude-agent-sdk 0.2.148 takes
            # effort on ClaudeAgentOptions, which is read when the session is
            # created, and ClaudeSDKClient exposes set_model and
            # set_permission_mode and no set_effort (read 2026-09-22). There is
            # no per-turn lever to put this on, so it is refused rather than
            # accepted and dropped.
            raise CapabilityNotSupported(
                Runtime.ANTHROPIC_SDK.value,
                "per-turn reasoning effort",
                "claude-agent-sdk sets effort on ClaudeAgentOptions when the "
                "session is created and offers no way to change it afterwards "
                "(0.2.148: set_model and set_permission_mode exist, set_effort "
                "does not). State the level on the connection instead, or open "
                "a new session to run at a different one",
            )

        if self._closed:
            raise VendorRunFailed(
                "this session's client has been closed; open a new session, or "
                "resume this one by id if it was persisted"
            )
        if self._broken is not None:
            raise VendorRunFailed(self._broken)
        if self._streaming:
            raise InvalidSession(
                "a turn is already streaming on this session: the runtime holds one "
                "conversation and one ordered sequence of turns, so finish or "
                "abandon the previous send() before starting another"
            )

        # Unbounded on purpose, where the run path is bounded at 256. The
        # consumer of a turn may walk away mid-stream, and a full queue leaves
        # the loop thread blocked inside ``put`` -- which is precisely when the
        # interrupt that would end the turn cannot be delivered. One turn's
        # events are bounded by the turn itself; a deadlocked session is not.
        items: queue.Queue[Any] = queue.Queue()
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._drive_turn(message, items), loop)
        self._streaming = True
        return self._drain(items, future)

    def _drain(self, items: queue.Queue[Any], future: Any) -> Iterator[AgentEvent]:
        completed = False
        try:
            while True:
                item = items.get()
                if item is _TURN_DONE:
                    completed = True
                    break
                if isinstance(item, BaseException):
                    # Same rule as the run path: an exception whose type belongs
                    # to the vendor package is the vendor reporting a failed
                    # turn and becomes a reported failure; anything else is
                    # modelpass's own bug and keeps raising.
                    failure = classify_pump_failure(item)
                    if failure is item:
                        raise item
                    raise failure from item
                yield item
        finally:
            self._streaming = False
            if not completed:
                self._abandon(items, future)

    async def _drive_turn(self, message: str, items: queue.Queue[Any]) -> None:
        """One turn, inside the session's loop. Never raises out of here.

        A failure is put on the queue for the consumer to classify, which is
        what keeps a vendor error a *reported* outcome (adapter rule 5) instead
        of an exception escaping a thread nobody is waiting on.
        """
        # Per turn, not per session: a tool_use id correlates a call and its
        # result inside one exchange, and a table that outlived the turn would
        # only accumulate.
        tool_names: dict[str, str] = {}
        try:
            client = await self._ensure_client()
            await client.query(message)
            async for vendor_message in client.receive_response():
                assert_expected_auth_mode(vendor_message, self._request.connection.auth_mode)
                self._remember_session_id(vendor_message)
                for event in map_message(
                    vendor_message,
                    partial_text=self._partial_text,
                    tool_names=tool_names,
                ):
                    items.put(event)
        except BaseException as exc:
            # Handed to the consumer to classify, not swallowed: the drain
            # decides whether this was the vendor reporting a failed turn or a
            # bug of ours, and either way it reaches the caller.
            items.put(exc)
        finally:
            items.put(_TURN_DONE)

    def _remember_session_id(self, message: Any) -> None:
        """Take the id from the runtime's own ``ResultMessage``, every turn.

        Read on every result rather than only the first, because the runtime is
        the authority on which conversation this is: a resume that the CLI
        chose to continue under a different id would otherwise leave the handle
        reporting an id nobody can resume.
        """
        if type(message).__name__ != "ResultMessage":
            return
        session_id = getattr(message, "session_id", None)
        if isinstance(session_id, str) and session_id:
            self._session_id = session_id

    async def _ensure_client(self) -> Any:
        """Connect once, under the scrub, and keep it.

        ``connect(None)`` is the SDK's streaming mode with no initial prompt --
        the subprocess comes up and waits, and the first turn arrives through
        ``query()`` like every turn after it. The scrub wraps only the connect,
        because that is where the transport snapshots ``os.environ``; by the
        second turn the child is already running with the environment the plan
        described.
        """
        if self._client is not None:
            return self._client
        options = self._sdk.ClaudeAgentOptions(**self._options_kwargs)
        client = self._sdk.ClaudeSDKClient(options=options)
        with _scrubbed_process_env(self._request.plan):
            await client.connect(None)
        self._client = client
        return client

    def _abandon(self, items: queue.Queue[Any], future: Any) -> None:
        """The caller walked away mid-turn. End the turn, keep the session.

        Abandoning one turn is not the end of a conversation (the session
        contract says so), so this asks the runtime for its graceful interrupt
        and then *drains what is still coming* -- an abandoned queue with a
        producer still writing to it is how the next turn would hang.

        If the turn will not stop, the session says so rather than pretending:
        the next ``send`` reports the failure instead of queueing behind a turn
        that never ended.
        """
        if not future.done():
            self._interrupt()
        deadline = time.monotonic() + _ABANDON_DRAIN_SECONDS
        while time.monotonic() < deadline:
            try:
                item = items.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _TURN_DONE:
                return
        self._broken = (
            "the previous turn on this session did not stop after an interrupt, so "
            "the runtime and modelpass no longer agree on where the conversation is. "
            "Close this session and open a new one"
        )

    def _interrupt(self) -> None:
        loop, client = self._loop, self._client
        if loop is None or client is None or loop.is_closed():
            return
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(client.interrupt(), loop).result(timeout=5)

    # --- history -----------------------------------------------------------------

    def history(self) -> tuple[Message, ...]:
        """The conversation as the runtime has it, read back from the transcript.

        Empty for an ephemeral session, and that is not a gap in the mapping:
        ``persist=False`` means no transcript was written, and the SDK offers no
        way to read the live client's context back out. Reconstructing it from
        the events modelpass happened to see would produce a second history that
        drifts from the one the model is being sent, which the contract refuses
        outright.
        """
        session_id = self._session_id
        if not self._request.persist or not session_id:
            return ()
        messages = self._sdk.get_session_messages(
            session_id, directory=self._request.project_folder
        )
        return session_messages_to_history(messages or ())

    # --- lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        """Disconnect the client and stop the loop. Idempotent.

        A persisted session stays on disk and stays resumable; what ends here is
        modelpass's hold on it -- the subprocess, the loop thread, and the temp
        file a large system prompt was written to.
        """
        if self._closed:
            return
        self._closed = True
        loop, client, thread = self._loop, self._client, self._thread
        try:
            if loop is not None and not loop.is_closed():
                if client is not None:
                    with contextlib.suppress(Exception):
                        asyncio.run_coroutine_threadsafe(
                            client.disconnect(), loop
                        ).result(timeout=10)
                loop.call_soon_threadsafe(loop.stop)
                if thread is not None:
                    thread.join(timeout=10)
                with contextlib.suppress(Exception):
                    loop.close()
        finally:
            self._client = None
            self._loop = None
            self._thread = None
            if self._prompt_dir:
                shutil.rmtree(self._prompt_dir, ignore_errors=True)
                self._prompt_dir = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """The session's own event loop, on its own thread, for its whole life."""
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="modelpass-anthropic-session", daemon=True
        )
        thread.start()
        self._loop, self._thread = loop, thread
        return loop


# --- the adapter ---------------------------------------------------------------


class AnthropicAdapter(Adapter):
    """Drives the Claude Agent SDK under a Claude subscription login."""

    runtime = Runtime.ANTHROPIC_SDK

    #: Deliberately open. Unrecognized keys are forwarded to
    #: ``ClaudeAgentOptions``, so an unknown one here is very often a valid SDK
    #: option this adapter has no reason to know about. Claiming a closed set
    #: would turn that passthrough into a refusal.
    option_keys = None

    def __init__(
        self,
        *,
        claude_version: VersionFn | None = None,
        auth_status: AuthStatusFn | None = None,
    ) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None
        self._cancelled = threading.Event()
        self._claude_version = claude_version or _default_claude_version
        self._auth_status = auth_status or _default_auth_status
        # Keyed by binary path and held for the adapter's life. A bridge keeps
        # one adapter per runtime, so this is one ~2s subprocess per process
        # rather than one per preflight -- and every preflight and every session
        # open wants the same answer about the same file.
        self._version_cache: dict[str, str | None] = {}
        #: ``claude auth status`` answers, keyed by binary *and* by the config
        #: directory the probe ran under -- two profiles on one binary are two
        #: different accounts and must never share an entry. Values are
        #: ``(monotonic timestamp, answer)``; see
        #: :data:`_IDENTITY_CACHE_TTL_SECONDS` for why this one expires.
        self._identity_cache: dict[
            tuple[str, str | None], tuple[float, Mapping[str, Any] | None]
        ] = {}

    # --- availability ----------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Whether ``claude-agent-sdk`` can be imported -- without importing it."""
        return importlib.util.find_spec(_MODULE) is not None

    @staticmethod
    def _sdk() -> Any:
        try:
            return importlib.import_module(_MODULE)
        except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
            raise RuntimeNotAvailable(Runtime.ANTHROPIC_SDK.value, _EXTRA, _PACKAGE) from exc

    @staticmethod
    def _cli_status(sdk: Any) -> tuple[str | None, str | None]:
        """Locate the Claude Code binary, preferring the SDK's own resolution.

        Source distributions and ``npm ci --omit=optional`` installs ship without
        the bundled binary; a native install then has to be found instead.
        """
        try:
            from claude_agent_sdk._internal.transport.subprocess_cli import (
                SubprocessCLITransport,
            )

            transport = SubprocessCLITransport(prompt="", options=sdk.ClaudeAgentOptions())
            return transport._find_cli(), None
        except Exception as exc:
            native = shutil.which("claude") or shutil.which("claude.exe")
            if native:
                return native, f"bundled binary unavailable, using native install ({exc})"
            return None, f"Claude Code binary not found: {exc}"

    def _cli_version(self, binary: str, env: Mapping[str, str]) -> str | None:
        """``claude --version`` for a resolved binary, asked once per process."""
        if binary not in self._version_cache:
            self._version_cache[binary] = self._claude_version(binary, env)
        return self._version_cache[binary]

    def _cached_auth_status(
        self, binary: str, env: Mapping[str, str]
    ) -> Mapping[str, Any] | None:
        """``claude auth status`` for one (binary, config directory), briefly cached."""
        key = (binary, env.get("CLAUDE_CONFIG_DIR"))
        now = time.monotonic()
        hit = self._identity_cache.get(key)
        if hit is not None and now - hit[0] < _IDENTITY_CACHE_TTL_SECONDS:
            return hit[1]
        answer = self._auth_status(binary, env)
        self._identity_cache[key] = (now, answer)
        return answer

    def invalidate_identity_cache(self) -> None:
        """Force the next preflight to re-probe. See :meth:`Adapter.invalidate_identity_cache`."""
        super().invalidate_identity_cache()
        self._identity_cache.clear()

    def _force_auth_status(
        self, binary: str, env: Mapping[str, str]
    ) -> Mapping[str, Any] | None:
        """Bypass the identity cache for a fresh ``claude auth status`` answer.

        Used only once a stored login's access token is found expired: the
        answer already cached for this preflight predates that discovery, and
        a ``claude`` invocation is Claude Code's own trigger for the proactive
        refresh it performs lazily on any command that needs a valid token. A
        cached "yes, logged in" from up to a minute ago says nothing about
        whether that refresh just succeeded or failed.
        """
        key = (binary, env.get("CLAUDE_CONFIG_DIR"))
        self._identity_cache.pop(key, None)
        return self._cached_auth_status(binary, env)

    def _ttl_env(self, sdk: Any, plan: PreflightPlan) -> dict[str, str]:
        """Pin the session prefix's cache TTL, but only where the CLI reads it.

        D17 asks modelpass to set the TTL rather than inherit it, because the
        subscription default drops from an hour to five minutes once an account
        starts drawing on credits. The version gate is the other half of that
        instruction: below :data:`_TTL_MIN_CLI_VERSION` the variable is not read
        at all, and setting it anyway would leave modelpass reporting a pin the
        runtime never applied.
        """
        binary, _ = self._cli_status(sdk)
        if binary is None:
            return {}
        if not supports_prompt_cache_ttl(self._cli_version(binary, plan.env)):
            return {}
        return {_TTL_ENV: _SESSION_TTL}

    # --- preflight -------------------------------------------------------------

    def preflight(self, request: RunRequest) -> Receipt:
        """Report which auth mode this run would actually use, and why."""
        plan = request.plan
        connection = request.connection
        check_launch_args(self.runtime, _contributed_args(request))

        notes: list[str] = []
        available = self.is_available()
        if not available:
            return Receipt.from_plan(
                plan,
                runtime_available=False,
                ok=False,
                problem=(
                    f"the {_PACKAGE!r} package is not installed; "
                    f"install it with 'pip install modelpass[{_EXTRA}]'"
                ),
            )

        sdk = self._sdk()
        cli_path, cli_note = self._cli_status(sdk)
        if cli_note:
            notes.append(cli_note)
        if cli_path is None:
            return Receipt.from_plan(
                plan,
                runtime_available=False,
                ok=False,
                problem="Claude Code binary not found; install Claude Code natively",
                notes=tuple(notes),
            )
        # Which Claude Code this will actually launch, named the way the Codex
        # receipt names its binary. The version is the fact a user otherwise has
        # to go and run ``claude --version`` for, and it decides real behavior:
        # the prompt-cache TTL lever exists only from 2.1.242 (D17).
        version = self._cli_version(cli_path, plan.env)
        notes.append(f"claude binary: {cli_path}" + (f" ({version})" if version else ""))
        notes.append(f"claude-agent-sdk {getattr(sdk, '__version__', 'unknown')}")

        if connection.auth_mode is AuthMode.API_KEY:
            return self._api_key_receipt(request, notes)
        profile = _account_profile_from_auth_status(
            self._cached_auth_status(cli_path, plan.env)
        )
        return self._subscription_receipt(
            request, notes, account_profile=profile, cli_path=cli_path
        )

    def _api_key_receipt(self, request: RunRequest, notes: list[str]) -> Receipt:
        plan = request.plan
        ref = request.connection.credential_ref
        if ref.kind is CredentialKind.ENV:
            present = bool(plan.env.get(ref.locator or ""))
            return Receipt.from_plan(
                plan,
                detected_auth_mode=AuthMode.API_KEY,
                credential_source=ref.describe(),
                ok=present,
                problem=None if present else f"{ref.locator} is not set in the environment",
                notes=tuple(notes),
            )
        return Receipt.from_plan(
            plan,
            detected_auth_mode=AuthMode.API_KEY,
            credential_source=ref.describe(),
            notes=tuple((*notes, "keychain credentials are not inspected by modelpass")),
        )

    def _subscription_receipt(
        self,
        request: RunRequest,
        notes: list[str],
        *,
        account_profile: AccountProfile | None = None,
        cli_path: str | None = None,
    ) -> Receipt:
        """Confirm a usable subscription login, or fail closed.

        A subscription connection that cannot prove subscription auth must never
        silently proceed: the runtime would fall back down its precedence list
        and bill the user by the token. Reporting ``ok=False`` makes the bridge
        raise before the first token is spent.
        """
        plan = request.plan
        status = read_credential_status(plan.env)

        if status.keychain_only:
            # No file to read, so ``claude auth status`` -- already run with this
            # connection's CLAUDE_CONFIG_DIR in its environment, and keyed to the
            # same directory the Keychain entry is keyed to -- is the only
            # evidence available about which account this directory selects.
            where = request.connection.config_dir or "the default Claude config directory"
            if account_profile is not None and account_profile.logged_in is False:
                return Receipt.from_plan(
                    plan,
                    detected_auth_mode=None,
                    credential_source=status.source,
                    account_profile=account_profile,
                    ok=False,
                    problem=(
                        f"claude auth status reports no login for {where}; run "
                        "'claude /login' with CLAUDE_CONFIG_DIR set to that directory "
                        "before using this subscription connection"
                    ),
                    notes=tuple(notes),
                )
            if account_profile is None:
                notes.append(
                    "identity could not be confirmed: 'claude auth status' returned "
                    "nothing and modelpass does not inspect Keychain contents"
                )

        if not status.present:
            # Same instruction either way, different diagnosis. "No login
            # found" against a file that plainly holds a plan reads as modelpass
            # looking in the wrong place, and sends the reader hunting for an
            # external credential store instead of logging in.
            # A connection that named a directory gets told which one, and that
            # the login has to be run with the variable set. `claude /login`
            # with no CLAUDE_CONFIG_DIR authenticates the *default* account, so
            # the bare instruction against a profile connection logs in the
            # wrong account and leaves this one failing. A connection on the
            # default directory keeps the wording it has always had.
            profile = request.connection.config_dir
            at = f" for {profile}" if profile else ""
            how = " with CLAUDE_CONFIG_DIR set to that directory" if profile else ""
            if status.logged_out:
                problem = (
                    f"the Claude Code credential file{at} holds no access token -- "
                    "the CLI is logged out (the plan metadata it keeps is not a "
                    f"login); run 'claude /login'{how} before using a "
                    "subscription connection"
                )
            else:
                problem = (
                    f"no Claude Code login found{at}; run 'claude /login'{how} (or "
                    "'claude setup-token' for headless use) before using a "
                    "subscription connection"
                )
            return Receipt.from_plan(
                plan,
                detected_auth_mode=None,
                credential_source=status.source,
                ok=False,
                problem=problem,
                notes=tuple(notes),
            )

        if not status.usable:
            when = _format_expiry(status.expires_at)
            return Receipt.from_plan(
                plan,
                detected_auth_mode=None,
                credential_source=status.source,
                account=account_profile.email if account_profile else None,
                plan_name=(
                    account_profile.subscription_type
                    if account_profile and account_profile.subscription_type
                    else status.subscription_type
                ),
                account_profile=account_profile,
                ok=False,
                problem=(
                    f"the Claude Code login expired {when} and carries no refresh token, "
                    "so a modelpass-spawned runtime cannot authenticate. Re-run "
                    "'claude /login' (or 'claude setup-token') to refresh it"
                ),
                notes=tuple(notes),
            )

        if status.expired:
            # usable is True here, so a refresh token is present. Rather than
            # trusting "the runtime will refresh it on use", spend one cheap
            # CLI probe to find out whether it already did: a ``claude``
            # invocation is Claude Code's own trigger for the proactive
            # refresh it performs lazily, and a permanently-failed refresh
            # (dead refresh_token, revoked grant) leaves the account looking
            # "logged in" without ever clearing the stored tokens -- exactly
            # the failure a passive note would miss.
            if cli_path is not None:
                fresh_profile = _account_profile_from_auth_status(
                    self._force_auth_status(cli_path, plan.env)
                )
                if fresh_profile is not None:
                    account_profile = fresh_profile
                status = read_credential_status(plan.env)
                if status.expired:
                    when = _format_expiry(status.expires_at)
                    return Receipt.from_plan(
                        plan,
                        detected_auth_mode=None,
                        credential_source=status.source,
                        account=account_profile.email if account_profile else None,
                        plan_name=(
                            account_profile.subscription_type
                            if account_profile and account_profile.subscription_type
                            else status.subscription_type
                        ),
                        account_profile=account_profile,
                        ok=False,
                        problem=(
                            f"the Claude Code login's access token expired {when} and "
                            "could not be refreshed (re-checked via 'claude auth "
                            "status'); the stored refresh token may be revoked or "
                            "invalid. Re-run 'claude /login' (or 'claude setup-token') "
                            "to log in again"
                        ),
                        notes=tuple(notes),
                    )
                notes.append("access token had expired and was refreshed during preflight")
            else:
                notes.append(
                    "access token is past its expiry; the runtime will refresh it on use"
                )
        if status.detail:
            notes.append(status.detail)
        if status.rate_limit_tier:
            notes.append(f"rate limit tier: {status.rate_limit_tier}")
        notes.append(_ALLOWANCE_NOTE)

        return Receipt.from_plan(
            plan,
            detected_auth_mode=AuthMode.SUBSCRIPTION,
            credential_source=status.source,
            account=(
                account_profile.email
                if account_profile and account_profile.email
                else (
                    account_profile.organization_name
                    if account_profile
                    else None
                )
            ),
            plan_name=(
                account_profile.subscription_type
                if account_profile and account_profile.subscription_type
                else status.subscription_type
            ),
            account_profile=account_profile,
            notes=tuple(notes),
        )

    def cache_eligibility(
        self, request: RunRequest | SessionRequest
    ) -> CacheEligibility:
        """Whether this call's prefix can cache, and at what TTL (D20).

        Token-free: the model comes off the request, the floor is a lookup, and
        the CLI version is the one :meth:`preflight` already probed and cached.

        A worker is reported as ``preset_prefix``, because that is what
        :func:`session_system_prompt` actually sends -- the ``claude_code``
        preset with the caller's text appended -- and the preset clears any of
        these floors by itself. A chat and a stateless call send the caller's
        own string and nothing else, so for those the caller's length is the
        whole of the answer.

        The TTL half reports **what modelpass actually does**, which is not the
        same as what the runtime supports, and the difference is the point:

        * A session pins ``1h`` where the CLI reads the variable (D17), so a
          receipt taken for a session says ``1h``.
        * A stateless call pins nothing, so the runtime's own default applies --
          one hour on the subscription path today, but five minutes once the
          account exceeds plan usage and starts drawing on credits, and modelpass
          is not the thing that decides which. Saying "1h" there would be a
          claim about somebody else's billing state.
        * A CLI below :data:`_TTL_MIN_CLI_VERSION` does not read the variable at
          all, so no TTL is reported however the call was made. Reporting a pin
          the runtime silently ignores is the exact failure the version gate
          exists to prevent, and the receipt must not reintroduce it.
        """
        session = isinstance(request, SessionRequest)
        floor, floor_source = cache_floor_tokens(request.model)
        ttl, ttl_detail = self._ttl_disclosure(request, session=session)
        return CacheEligibility(
            system_prompt_chars=request.system_prompt_chars,
            floor_tokens=floor,
            floor_source=floor_source,
            tools_declared=request.wants_tools,
            preset_prefix=session and request.appends_system_prompt,
            ttl=ttl,
            ttl_detail=ttl_detail,
        )

    def _ttl_disclosure(
        self, request: RunRequest | SessionRequest, *, session: bool
    ) -> tuple[str | None, str]:
        """The TTL this call runs under, and the sentence explaining it.

        Returns no TTL and **no sentence** when the runtime cannot be inspected
        at all. A preflight that could not find the SDK or the binary has
        already said so in a way that matters more, and adding a line about
        cache lifetimes underneath it would be noise on top of a real problem.
        """
        if not self.is_available():
            return None, ""
        try:
            binary, _ = self._cli_status(self._sdk())
        except SubpassError:
            return None, ""
        if binary is None:
            return None, ""
        version = self._cli_version(binary, request.plan.env)
        if not supports_prompt_cache_ttl(version):
            named = version or "an unreadable version"
            wanted = ".".join(str(number) for number in _TTL_MIN_CLI_VERSION)
            return None, (
                "the prefix cache TTL is the runtime's own default: pinning it needs "
                f"Claude Code {wanted} or later and this build reports {named}, so "
                "modelpass sets nothing rather than setting a variable that would be "
                "silently ignored"
            )
        if session:
            return _SESSION_TTL, (
                f"the prefix cache TTL is pinned to {_SESSION_TTL} for this session, "
                "rather than inherited -- the subscription default drops from an hour "
                "to five minutes once an account starts drawing on credits"
            )
        return None, (
            "the prefix cache TTL is the runtime's own default for a stateless call: "
            "modelpass pins one per session (D17) and does not pin one here, so it is "
            "an hour on the subscription path and five minutes once the account is "
            "drawing on credits"
        )

    # --- run -------------------------------------------------------------------

    def run(self, request: RunRequest) -> Iterator[AgentEvent]:
        """Execute one stateless chat call and stream normalized events."""
        sdk = self._sdk()
        plan = request.plan
        check_launch_args(self.runtime, _contributed_args(request))
        self._cancelled.clear()

        partial_text = bool(request.options.get("include_partial_messages", True))
        system_prompt, prompt = split_messages(request.messages)

        additions = _explicit_env_additions(plan)
        additions[_SKIP_HISTORY_ENV] = "1"

        options_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "model": request.model,
            "env": additions,
            # Built-in tools are off in every mode (D7 for a plain chat call,
            # D12 for a tool run). ``tools=[]`` is the SDK's documented switch
            # for "remove every built-in"; the explicit deny list is a second
            # lock that does not survive Claude Code adding a tool, which is
            # exactly why it is not the primary mechanism.
            "tools": [],
            "allowed_tools": [],
            "disallowed_tools": list(_BUILTIN_TOOLS),
            # An empty source list keeps settings.json out of the run entirely,
            # which is how the apiKeyHelper directive is enforced: apiKeyHelper
            # outranks OAuth and lives in settings, not the environment.
            "setting_sources": [],
            "max_turns": 1,
            "include_partial_messages": partial_text,
            "permission_mode": "default",
        }
        options_kwargs.update(_tool_options(sdk, request))
        # The stated effort, in this runtime's own spelling (2026-09-21).
        # claude-agent-sdk 0.2.148 types ClaudeAgentOptions.effort as
        # Literal["low","medium","high","xhigh","max"]. This runtime is wired
        # directly rather than through sampling_rules because its
        # sampling_controls cell reads unsupported -- the sampling pipeline
        # carries nothing here, so the standing default would reach no wire.
        _effort = stated_reasoning(request.connection)
        if _effort is not None:
            options_kwargs.update(_reasoning_options(_effort))
        if request.schema is not None:
            # The runtime's own mechanism (D13). ``output_format`` becomes
            # ``--json-schema <json>`` in the transport and the answer comes back
            # on ResultMessage.structured_output. max_turns=1 is left alone: the
            # submission tool's round trip does not need a higher ceiling
            # (verified live -- num_turns=2, terminal_reason "completed").
            options_kwargs["output_format"] = {
                "type": "json_schema",
                "schema": dict(request.schema),
            }
        for key, value in request.options.items():
            if key == "include_partial_messages":
                continue
            if key == "env":
                # Caller levers may travel alongside the connection's explicit
                # account selection, but may not replace it. This mirrors the
                # session path: configDir and named credentials are properties
                # of the connection whose receipt the run carries.
                merged = dict(value) if isinstance(value, Mapping) else {}
                merged.update(options_kwargs["env"])
                options_kwargs["env"] = merged
                continue
            options_kwargs[key] = value

        # ClaudeAgentSDK transports a string system prompt as one CLI argument.
        # The stateless path needs the same large-prompt protection as sessions:
        # real scoring rubrics plus candidate profiles can exceed Windows'
        # process command-line ceiling before Claude is ever launched.
        prompt_dir: str | None = None
        prompt_option = options_kwargs.get("system_prompt")
        if isinstance(prompt_option, str):
            prompt_option, prompt_dir = file_backed_system_prompt(prompt_option)
            options_kwargs["system_prompt"] = prompt_option

        return self._pump(
            sdk,
            request,
            options_kwargs,
            prompt,
            partial_text,
            prompt_dir=prompt_dir,
        )

    def _pump(
        self,
        sdk: Any,
        request: RunRequest,
        options_kwargs: dict[str, Any],
        prompt: str,
        partial_text: bool,
        *,
        prompt_dir: str | None = None,
    ) -> Iterator[AgentEvent]:
        """Bridge the SDK's async iterator onto modelpass's sync one.

        v1 ships a sync surface (implementation plan, "The async story"), so the
        adapter owns the event loop: a worker thread runs the async iteration and
        posts normalized events to a queue that the returned generator drains.
        """
        items: queue.Queue[Any] = queue.Queue(maxsize=256)
        done = object()
        # tool_use_id -> tool name, so a tool_result can name the tool that
        # produced it. Owned by the run, not the adapter, so concurrent runs
        # cannot cross-contaminate each other's correlations.
        tool_names: dict[str, str] = {}

        async def drive() -> None:
            self._loop = asyncio.get_running_loop()
            client = sdk.ClaudeSDKClient(options=sdk.ClaudeAgentOptions(**options_kwargs))
            self._client = client
            try:
                # The scrub is only needed while the child is being spawned: the
                # transport snapshots os.environ inside connect().
                with _scrubbed_process_env(request.plan):
                    await client.connect(prompt)
                async for message in client.receive_response():
                    assert_expected_auth_mode(message, request.connection.auth_mode)
                    for event in map_message(
                        message,
                        partial_text=partial_text,
                        tool_names=tool_names,
                        schema=request.schema,
                        schema_name=request.schema_name,
                    ):
                        items.put(event)
            finally:
                with contextlib.suppress(Exception):
                    await client.disconnect()
                self._client = None
                self._loop = None

        def worker() -> None:
            try:
                asyncio.run(drive())
            except BaseException as exc:
                items.put(exc)
            finally:
                items.put(done)

        thread = threading.Thread(target=worker, name="modelpass-anthropic", daemon=True)
        thread.start()

        def generate() -> Iterator[AgentEvent]:
            saw_terminal = False
            try:
                while True:
                    item = items.get()
                    if item is done:
                        break
                    if isinstance(item, BaseException):
                        # The SDK raises after yielding an error ResultMessage.
                        # If we already emitted a terminal, that result is the
                        # story and the exception is noise; otherwise it is real
                        # -- and what "real" means is decided by
                        # classify_pump_failure: the vendor's own exception
                        # types become a reported run failure, everything else
                        # stays an exception.
                        if saw_terminal or self._cancelled.is_set():
                            break
                        failure = classify_pump_failure(item)
                        if failure is item:
                            raise item
                        raise failure from item
                    if isinstance(item, TerminalEvent):
                        saw_terminal = True
                    yield item
                if self._cancelled.is_set() and not saw_terminal:
                    yield _bare_terminal(TerminalStatus.CANCELLED, "cancelled by caller")
            finally:
                self.cancel()
                thread.join(timeout=10)
                if prompt_dir is not None:
                    shutil.rmtree(prompt_dir, ignore_errors=True)

        return generate()

    # --- sessions (D14-D17) ----------------------------------------------------

    def open_session(self, request: SessionRequest) -> SessionHandle:
        """Build the local half of a session. No subprocess, no request, no spend.

        Everything here is options assembly and, for a large system prompt, one
        temp file. The vendor session comes into existence on the first
        :meth:`AnthropicSessionHandle.send` -- there is no create-session call to
        make, and the id it will be known by is issued by that first turn
        (adapter contract, rule 7).

        The one thing that does run is a token-free ``claude --version`` probe,
        cached per process, because whether the TTL lever is real on this
        machine has to be known before the client is configured rather than
        discovered from a cache that behaved unexpectedly.
        """
        sdk = self._sdk()
        check_launch_args(self.runtime, _contributed_args(request))
        options_kwargs, prompt_dir = session_options(
            sdk, request, env_additions=self._session_env(sdk, request)
        )
        return AnthropicSessionHandle(
            sdk=sdk,
            request=request,
            options_kwargs=options_kwargs,
            prompt_dir=prompt_dir,
        )

    def resume_session(self, request: SessionRequest) -> SessionHandle:
        """Pick a stored session back up. ``resume=<id>`` is the whole mechanism.

        The existence check runs here rather than being left to the first turn,
        because it is cheap -- a stat and a head/tail read of one transcript, no
        subprocess and no tokens -- and "this id is gone" is worth learning
        before a turn is paid for.

        **One thing this runtime cannot carry across a resume, stated because it
        is a real loss rather than an omission:** the system prompt is a launch
        option, not part of the transcript. A resumed chat is launched with the
        empty replacement, so the persona the original session was constructed
        with is *not* restored and the prefix it warmed is not the one this
        conversation now sends. :meth:`Bridge.resume_chat` takes no
        ``system_prompt`` (D17 allows only omission or an exact match), so where
        that persona is load-bearing the honest move is a fresh session with the
        prompt supplied again.
        """
        sdk = self._sdk()
        check_launch_args(self.runtime, _contributed_args(request))
        session_id = request.resume_id or ""
        self._require_stored_session(sdk, request, session_id)
        options_kwargs, prompt_dir = session_options(
            sdk, request, env_additions=self._session_env(sdk, request)
        )
        # Set after the caller's options are merged: ``resume`` is one of the
        # fields a session owns, and this is the one place it may be written.
        options_kwargs["resume"] = session_id
        return AnthropicSessionHandle(
            sdk=sdk,
            request=request,
            options_kwargs=options_kwargs,
            prompt_dir=prompt_dir,
            session_id=session_id,
        )

    def list_sessions(self, request: SessionRequest) -> tuple[SessionInfo, ...]:
        """Every session stored under this request's project folder, newest first.

        Directory-scoped because the storage is: Claude Code writes transcripts
        to ``~/.claude/projects/<encoded-cwd>/``, so the working directory *is*
        the key and a listing cannot be anything but per-folder. The SDK already
        sorts by last-modified descending, and enumerating is enumerating -- a
        session the runtime does not report is not reported, and an ephemeral
        one never appears because it was never written down.

        ``include_worktrees=False``, which is not the SDK's default. A listing
        asked about one folder should answer about that folder: sessions from a
        sibling git worktree ran against different files, and offering them here
        would invite a resume that reads as continuing this work and does not.
        """
        sdk = self._sdk()
        folder = request.project_folder
        listed = sdk.list_sessions(directory=folder, include_worktrees=False)
        return tuple(
            session_info_from_sdk(
                info, connection=request.connection.name, project_folder=folder
            )
            for info in listed or ()
        )

    def _session_env(self, sdk: Any, request: SessionRequest) -> dict[str, str]:
        """The environment additions a session launches with.

        Three sources, and the ordering between them is not interesting because
        they do not overlap: the environment values the connection explicitly
        selected (credential variables, allowEnv and configDir), the skip-history
        switch that makes ``persist=False`` mean something, and the TTL pin where
        the CLI is new enough to read it.

        Note what is *absent*: :data:`_SKIP_HISTORY_ENV` for a persisted
        session. :meth:`run` sets it unconditionally, correctly, because a
        stateless call has nothing to resume -- but a session that inherited it
        would leave nothing on disk while telling the caller it could be resumed.
        """
        additions = _explicit_env_additions(request.plan)
        if not request.persist:
            additions[_SKIP_HISTORY_ENV] = "1"
        additions.update(self._ttl_env(sdk, request.plan))
        return additions

    @staticmethod
    def _require_stored_session(
        sdk: Any, request: SessionRequest, session_id: str
    ) -> None:
        """Refuse a resume the runtime cannot honour, before a turn runs.

        ``get_session_info()`` is the cheap question, but it answers ``None`` to
        two different ones: *there is no such transcript*, and *there is one and
        no summary could be extracted from it*. Refusing on the second would
        turn away a session that exists, so the transcript itself is the second
        opinion and only an id the runtime can show nothing at all for is
        reported missing.
        """
        directory = request.project_folder
        if sdk.get_session_info(session_id, directory=directory) is not None:
            return
        if sdk.get_session_messages(session_id, directory=directory):
            return
        raise SessionNotFound(
            request.connection.name,
            session_id,
            f"no transcript for it under {directory!r}. Claude Code stores sessions "
            "per working directory and sweeps old ones on cleanupPeriodDays, so "
            "resume from the project_folder the session was opened with, and treat "
            "a stored id as a hint rather than a guarantee",
        )

    # --- cancellation ----------------------------------------------------------

    def cancel(self) -> None:
        """Best-effort cancellation (D10).

        Claude Code advertises a graceful interrupt, so this asks for one and
        falls back to the documented floor -- tearing down the transport, which
        terminates the child process -- if the interrupt cannot be delivered.
        """
        self._cancelled.set()
        loop, client = self._loop, self._client
        if loop is None or client is None or loop.is_closed():
            return
        try:
            future = _run_coroutine_threadsafe(client.interrupt(), loop)
            future.result(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                _run_coroutine_threadsafe(client.disconnect(), loop).result(timeout=5)


def _run_coroutine_threadsafe(
    coroutine: Any, loop: asyncio.AbstractEventLoop
) -> Any:
    """Schedule a coroutine without leaking it if the loop closes in the race.

    ``run_coroutine_threadsafe`` takes ownership only after successful
    scheduling. A stateless run can finish and close its worker loop between a
    cancellation check and this call; explicitly close the still-local
    coroutine in that case so repeated batches do not emit misleading
    ``coroutine was never awaited`` warnings.
    """
    try:
        return asyncio.run_coroutine_threadsafe(coroutine, loop)
    except BaseException:
        coroutine.close()
        raise


def _format_expiry(expires_at: float | None) -> str:
    if expires_at is None:
        return "at an unknown time"
    import datetime as _dt

    return _dt.datetime.fromtimestamp(expires_at, _dt.UTC).strftime("on %Y-%m-%d")


def _format_timestamp(value: float) -> str:
    """Format an allowance reset time. Tolerates seconds or milliseconds."""
    import datetime as _dt

    seconds = value / 1000.0 if value > 1e11 else value
    try:
        return _dt.datetime.fromtimestamp(seconds, _dt.UTC).strftime("%Y-%m-%d %H:%MZ")
    except (OverflowError, OSError, ValueError):
        return str(value)


#: Claude Code's built-in tool names, denied for a v1 chat call.
_BUILTIN_TOOLS: tuple[str, ...] = (
    "Bash",
    "Edit",
    "Glob",
    "Grep",
    "NotebookEdit",
    "Read",
    "Task",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
)
