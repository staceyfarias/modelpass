"""The ``anthropic-api`` adapter: the Messages API, driven in this process (ticket 1.6).

The first of the four API-key adapters, and the one the other three copy. It
shares a vendor with ``anthropic-sdk`` and shares nothing else: no subprocess, no
login store, no vendor-held session. What it has instead is an HTTP client this
process constructs with an explicit key, which is the whole of the in-process
clause of the adapter contract's rule 1 -- see :mod:`modelpass.adapters.base`.

**What is the same as the agent runtime, on purpose.** The event vocabulary
(``text_delta``, ``thinking``, ``usage``, ``tool_call`` / ``tool_result``,
``structured_output``, ``terminal``, ``vendor_event``) and the token mapping.
``TokenUsage``'s four fields mean exactly what they mean on ``anthropic-sdk`` --
``cache_read_input_tokens`` into ``cached_input_tokens`` and
``cache_creation_input_tokens`` into ``cache_write_tokens``, counted beside
``input_tokens`` rather than inside it -- so a consumer's accounting, its run
log and its guards do not learn that a run changed runtimes.
``tests/test_adapter_anthropic_api.py`` pins that with the two adapters' mappings
side by side, because "the same" is a claim worth a test rather than a comment.

**What is different, and why each difference exists.**

* **The adapter owns the tool loop.** ``while stop_reason == "tool_use"``, each
  requested :class:`~modelpass.tools.ToolDef` handler called in this process,
  the results fed back as ``tool_result`` blocks, no turn cap (the guard's
  ``stopAtTokens`` is the bound -- the invented turn cap was deliberately
  removed on 2026-08-15 and is not coming back through a new adapter). A raising
  handler becomes a failed tool result, exactly as
  :class:`~modelpass.tools.ToolDef` promises, so the model can retry or explain
  rather than the run dying. Validation of 2026-09-13, §2.7, is where this is
  argued; the short version is that the only thing that differs between the six
  runtimes is *who owns the ``while``*, which is precisely what an adapter is
  for. D12 stays deferred: ``ToolCallEvent`` still has no reply path, because
  nothing outside this module ever answers one.
* **Usage arrives per response, so a guard can stop between rounds.** Each
  response carries its own counts, emitted as a ``delta``; the bridge folds them
  and breaks the stream the moment ``stopAtTokens`` is crossed, which happens
  *before* that round's tools run. Better granularity than either agent runtime
  and the reason ``interim_usage`` is ``supported`` here.
* **The caller's ``cache_control`` breakpoints reach the vendor intact.** This
  is the runtime the whole content-block vocabulary of ticket 1.5 exists for:
  ``system=`` and each message's ``content=`` are block arrays, and
  :attr:`~modelpass.adapters.base.RunRequest.system_blocks` /
  :attr:`~modelpass.adapters.base.RunRequest.conversation_blocks` are handed
  through with their markers on them. Nothing is flattened and nothing is
  invented.
* **Structured output is native.** ``output_config={"format": {"type":
  "json_schema", "schema": ...}}``, read off the installed SDK on 2026-09-13 (see
  :data:`SDK_VERSION_READ`). The schema-as-forced-tool fallback was written for
  an SDK that did not have this and is not built, because building an unused
  second path is how two dialects of one feature start. D13 holds: a schema and
  tools are never combined, and the bridge refuses the combination before an
  adapter sees it.

**What it deliberately does not do.** Sessions: an API endpoint holds no
history, so :meth:`open_session` and its siblings refuse with the base
contract's wording and name ``chat(history=...)``, which is the documented
multi-turn shape against a stateless runtime (validation §2.6, answer 1; ticket
1.8 formalises the policy). MCP servers: there is no client here to connect
anything. Subagents: there is no orchestrator.

**Sampling, and the honesty that comes with it (ticket 1.7, R5).**
``temperature`` / ``top_p`` / ``top_k`` / ``max_output_tokens`` / reasoning
effort are first-class request fields, and this is the first runtime that can
actually take them: all five have a home in ``anthropic`` 0.97.0's request
schema. :meth:`AnthropicAPIAdapter.sampling_params` translates them and decides
none of them -- which model accepts which control is
:mod:`modelpass.sampling_rules`' answer, given once and used both here and by the
receipt, so what is sent and what is reported are the same list. Two capability
cells moved on that evidence: ``sampling_controls`` and ``max_output_tokens``.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..capabilities import Capability, Support
from ..connections import Connection
from ..errors import (
    AdapterNotImplemented,
    PreflightFailed,
    RuntimeNotAvailable,
    SubpassError,
    VendorRunFailed,
)
from ..preflight import (
    CacheEligibility,
    ModelListProbe,
    Receipt,
    api_preflight,
    cache_floor_tokens,
    resolve_credential,
)
from ..retry import vendor_error_facts
from ..runtimes import Runtime
from ..sampling_rules import plan_sampling
from ..schema import build_structured_event, normalize_schema, resolve_schema_name
from ..tools import ToolDef
from ..types import (
    CALLER_TOOL_SERVER,
    AgentEvent,
    AuthMode,
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
from ._mcp_result import ainvoke_handler, flatten_mcp_result, mcp_result_payload
from .base import (
    Adapter,
    InFlight,
    RunCancel,
    RunRequest,
    RunStream,
    SessionHandle,
    SessionRequest,
    native_run,
)

__all__ = ["SDK_VERSION_READ", "AnthropicAPIAdapter", "token_usage"]

_MODULE = "anthropic"
_EXTRA = "anthropic-api"
_PACKAGE = "anthropic"

#: The installed SDK this adapter was written against and whose surface every
#: capability note cites. Recorded as data rather than as prose because the notes
#: are evidence, and evidence names its source. Read on 2026-09-13.
SDK_VERSION_READ = "0.97.0"

#: What ``max_tokens`` is sent as when nobody said. The Messages API requires the
#: field -- there is no "the model's default" to fall back to -- so an adapter
#: that sends nothing cannot make a call at all. 4096 is large enough for an
#: ordinary answer and small enough that a runaway costs a cent rather than a
#: dollar.
#:
#: No longer a stopgap (ticket 1.7): the ceiling is a request field
#: (``sampling=Sampling(max_output_tokens=N)``) and this is what
#: :mod:`modelpass.sampling_rules` supplies when nobody set one -- and, unlike
#: before, the receipt *says* it supplied it. The number is written in both
#: places and kept in step by
#: ``tests/test_sampling.py::test_the_anthropic_default_ceiling_is_one_number``,
#: because a silent disagreement between them would put one figure on the wire
#: and a different one on the receipt.
DEFAULT_MAX_OUTPUT_TOKENS = 4096

#: How many model ids :meth:`AnthropicAPIAdapter.probe` keeps. The probe answers
#: "is this credential live"; the list is a bonus, and an unbounded one would put
#: a hundred strings on a receipt nobody asked for.
_PROBE_LIMIT = 20


# --- usage -----------------------------------------------------------------------


def token_usage(usage: Any) -> TokenUsage:
    """Normalize the SDK's ``Usage`` onto modelpass tokens.

    **Deliberately the same four assignments as
    :func:`modelpass.adapters.anthropic.token_usage`**, against the same four
    wire names, because the two runtimes report the same fields and a consumer
    reading ``cache_write_tokens`` must not have to know which one produced the
    number. The difference is only in what arrives: the Agent SDK hands over a
    dict and the API SDK a model object, so this reads attributes first and keys
    second, and a missing or non-integer field counts as zero rather than
    raising inside an event mapping.
    """

    def count(key: str) -> int:
        if isinstance(usage, Mapping):
            value = usage.get(key)
        else:
            value = getattr(usage, key, None)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    return TokenUsage(
        input_tokens=count("input_tokens"),
        output_tokens=count("output_tokens"),
        cached_input_tokens=count("cache_read_input_tokens"),
        cache_write_tokens=count("cache_creation_input_tokens"),
    )


# --- request assembly ------------------------------------------------------------


def system_param(request: RunRequest) -> list[dict[str, Any]] | None:
    """The ``system=`` block array, ``cache_control`` intact (R3).

    ``None`` when the run carries no system message, which is what omitting the
    field means. Every block is the caller's own -- :meth:`TextBlock.to_dict` is
    already the vendor's wire shape, which is not a coincidence: ticket 1.5 chose
    Anthropic's spelling for the breakpoint marker precisely so this function
    would be a pass-through rather than a translation.
    """
    blocks = request.system_blocks
    if not blocks:
        return None
    return [block.to_dict() for block in blocks]


def conversation_param(request: RunRequest) -> list[dict[str, Any]]:
    """The ``messages=`` array: one entry per non-system turn, blocks and all."""
    return [
        {"role": role, "content": [block.to_dict() for block in blocks]}
        for role, blocks in request.conversation_blocks
    ]


def tool_params(tools: Sequence[ToolDef]) -> list[dict[str, Any]]:
    """Caller tools as ``tools=`` entries.

    Full JSON Schema, never the SDK's Python-type shorthand, for the reason
    :mod:`modelpass.tools` gives: the shorthand cannot express enums, ranges,
    optional fields or nested objects, and JSON Schema is what MCP speaks.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.json_schema(),
        }
        for tool in tools
    ]


def output_config(request: RunRequest) -> dict[str, Any] | None:
    """The native structured-output configuration, or ``None`` (D13).

    ``output_config={"format": {"type": "json_schema", "schema": ...}}`` is the
    current spelling, read off ``anthropic`` 0.97.0 on 2026-09-13; the older
    top-level ``output_format`` is deprecated in the same build and is not used.
    The schema is normalized first so the one modelpass validates the answer
    against and the one the vendor constrains it with are the same object.
    """
    if request.schema is None:
        return None
    return {"format": {"type": "json_schema", "schema": normalize_schema(request.schema)}}



def _block_param(block: Any) -> Any:
    """One response content block, in the shape it goes back up as.

    Assistant turns are replayed verbatim in a tool loop -- thinking blocks
    included, signatures included -- so this dumps whatever the SDK handed over
    rather than rebuilding a block from the fields this adapter happens to read.
    A block that cannot be dumped is passed through as itself, because the SDK
    accepts its own objects and guessing would be worse than not trying.
    """
    for name in ("to_dict", "model_dump"):
        dump = getattr(block, name, None)
        if callable(dump):
            try:
                return dump()
            except Exception:  # pragma: no cover - defensive: a vendor model quirk
                break
    if isinstance(block, Mapping):
        return dict(block)
    return block


# --- event mapping ---------------------------------------------------------------

#: Stream frames the helper events are *built from*. The SDK's stream yields both
#: the raw frame and the normalized helper event for the same text, so passing
#: these through as well would report every delta twice -- once as a
#: ``text_delta`` and once as a ``vendor_event`` carrying the same characters.
#: They are not dropped information: everything they accumulate into is read off
#: the final message at the end of the round.
_RAW_STREAM_TYPES = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    }
)


def stream_events(event: Any) -> list[AgentEvent]:
    """Map one SDK stream event onto the normalized vocabulary.

    ``text`` and ``thinking`` are the two the vocabulary has words for. Every
    other helper event becomes a ``vendor_event`` named ``stream.<type>`` (rule
    3) -- ``input_json`` while a tool's arguments stream, a ``citation``, a
    ``signature`` -- except the raw frames in :data:`_RAW_STREAM_TYPES`, which
    are the transport underneath the helper rather than anything new.
    """
    kind = getattr(event, "type", None)
    if kind == "text":
        text = getattr(event, "text", "")
        return [TextDeltaEvent(text)] if isinstance(text, str) and text else []
    if kind == "thinking":
        text = getattr(event, "thinking", "")
        return [ThinkingEvent(text)] if isinstance(text, str) and text else []
    if not isinstance(kind, str) or kind in _RAW_STREAM_TYPES:
        return []
    return [VendorEvent(Runtime.ANTHROPIC_API, f"stream.{kind}", _vendor_payload(event))]


def _vendor_payload(event: Any) -> dict[str, Any]:
    """A JSON-safe payload for a vendor event. Never raises (D11)."""
    dumped = _block_param(event)
    if isinstance(dumped, Mapping):
        try:
            json.dumps(dumped, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return {"repr": repr(event)[:500]}
        return dict(dumped)
    return {"repr": repr(event)[:500]}


def terminal_status(message: Any) -> tuple[TerminalStatus, str | None]:
    """Map a finished response's ``stop_reason`` onto a terminal status.

    The five documented reasons, and the decisions behind two of them:

    * ``end_turn`` and ``stop_sequence`` are ``ok``. A stop sequence is the
      caller's own instruction being obeyed, not a failure.
    * ``tool_use`` reaching here at all means the loop had nothing to run -- the
      model asked for a tool this run did not declare. That is ``error`` with a
      reason naming it, because the answer is incomplete and silently calling it
      ``ok`` is how a consumer ships a half-finished response.
    * ``max_tokens`` is ``error``, and this is the judgement call worth stating.
      The answer is truncated mid-sentence; nothing downstream can tell that from
      a complete one, and the run log would record a successful call that
      returned half a thought. The reason names the ceiling that was hit so the
      fix -- raise it -- is in the message rather than in a doc.
    * ``refusal`` is ``error``, carrying ``stop_details.category`` where the
      vendor supplied one.

    Anything else -- ``pause_turn`` today, whatever the vocabulary grows next --
    is ``ok`` with the raw reason recorded, because an adapter inventing a
    verdict for a value it does not know is rule 3 broken in the other direction.
    """
    reason = getattr(message, "stop_reason", None)
    if reason in (None, "end_turn", "stop_sequence"):
        return TerminalStatus.OK, reason
    if reason == "max_tokens":
        return (
            TerminalStatus.ERROR,
            "the answer was truncated at the max_tokens ceiling; raise "
            "max_output_tokens for this call",
        )
    if reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None)
        suffix = f" (category {category!r})" if category else ""
        return TerminalStatus.ERROR, f"the model refused this request{suffix}"
    if reason == "tool_use":
        return (
            TerminalStatus.ERROR,
            "the model asked for a tool this run did not declare, so the answer "
            "is unfinished",
        )
    return TerminalStatus.OK, str(reason)


def _bare_terminal(
    status: TerminalStatus,
    reason: str | None,
    *,
    status_code: int | None = None,
    retry_after: float | None = None,
) -> TerminalEvent:
    """A terminal carrying a *status* only; the bridge owns the stamp (rule 2).

    ``status_code`` and ``retry_after`` are the two typed facts the bridge needs
    to compute a retryability verdict without reading ``reason`` (R6, ticket
    1.12). They ride here rather than being parsed back out of the message,
    which is the mistake this whole path exists to avoid.
    """
    return TerminalEvent(
        status=status,
        connection="",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        reason=reason,
        status_code=status_code,
        retry_after=retry_after,
    )


# --- the in-process tool loop -----------------------------------------------------


def invoke_handler(tool: ToolDef, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Run one caller tool and return an MCP result envelope. Never raises.

    The sync counterpart of ``adapters/anthropic._wrap_handler``, and the same
    two promises, worded the same way so a consumer matching on the text sees one
    string across runtimes:

    * **A handler that raises becomes a failed tool result.** The model can retry
      it, try another tool, or explain -- killing the run instead throws away the
      turns already paid for. The exception's type and message reach the model,
      which :class:`~modelpass.tools.ToolDef` documents so a caller knows not to
      raise anything carrying a secret.
    * **An async handler is driven to completion here.** ``run()`` is a sync
      generator with no loop of its own, so a coroutine is run on a private one.
      Unlike the agent runtime there is nothing to stall: this adapter's stream
      is an HTTP response already fully consumed for this round, so a blocking
      handler blocks only the caller who asked for it.
    """
    handler = tool.handler
    if handler is None:
        return {
            "content": [{"type": "text", "text": f"tool {tool.name!r} has no handler"}],
            "is_error": True,
        }
    try:
        result = handler(dict(arguments))
        if inspect.isawaitable(result):
            result = asyncio.run(_await(result))
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
    return mcp_result_payload(result)


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _missing_tool(name: str) -> dict[str, Any]:
    """The result for a tool the model named and this run never declared."""
    return {
        "content": [{"type": "text", "text": f"tool {name!r} was not declared for this run"}],
        "is_error": True,
    }


# --- the adapter -----------------------------------------------------------------


class AnthropicAPIAdapter(Adapter):
    """Drives Anthropic's Messages API with an explicit key."""

    runtime = Runtime.ANTHROPIC_API

    #: A closed set, so an unknown key is *reported* rather than silently
    #: ignored. Unlike ``anthropic-sdk`` this adapter forwards nothing to a
    #: vendor options object, so it genuinely can say what it does not read.
    #: ``max_output_tokens`` became a request field in ticket 1.7
    #: (``sampling=Sampling(max_output_tokens=N)``); the option stays readable as
    #: an alias, folded in by
    #: :attr:`~modelpass.adapters.base.RunRequest.effective_sampling` so there is
    #: still only one value and one report. Deprecated in ``chat``'s docstring,
    #: warning-free until the consumer migrations.
    option_keys = frozenset({"max_output_tokens", "probe_credential"})

    def __init__(
        self,
        *,
        client_factory: Any = None,
        async_client_factory: Any = None,
        env: Mapping[str, str] | None = None,
        secrets: Any = None,
    ) -> None:
        """``client_factory`` is the seam every test in this suite runs through.

        It is called as ``client_factory(api_key=..., base_url=...)`` and must
        return something with ``messages.stream(...)`` and, for the probe,
        ``models.list(...)``. The default builds
        ``anthropic.Anthropic(api_key=..., base_url=...)`` -- **never the bare
        constructor**, which would let the SDK's own discovery read
        ``ANTHROPIC_API_KEY`` out of the process environment and bill an account
        nobody named (adapter contract rule 1, in-process clause;
        ``tests/test_no_ambient_credentials.py`` is where that stays checked).

        ``async_client_factory`` is the same seam for :meth:`arun` (ticket
        1.13), called with the same two arguments and expected to return
        something whose ``messages.stream(...)`` is an *async* context manager.
        The default builds ``anthropic.AsyncAnthropic(api_key=..., base_url=...)``
        -- the vendor's own async client, under the same never-ambient rule. Two
        factories rather than one that returns both, because a test injecting
        one of them should not have to supply the other.

        ``env`` and ``secrets`` are the resolution sources handed to
        :func:`~modelpass.preflight.resolve_credential`; ``None`` means the
        process environment and the store beside the connection file, which is
        what production wants. They exist because a host carrying its own
        environment -- a test, an app with its own store root -- must be able to
        resolve against it, and because the credential must reach the client as
        one string at the moment of use rather than through a mapping anything
        can copy.
        """
        self._client_factory = client_factory or _default_client
        self._async_client_factory = async_client_factory or _default_async_client
        self._env = env
        self._secrets = secrets
        #: The runs in flight, each with its own cancel state. Nothing about one
        #: run lives on the adapter: a bridge shares it across concurrent calls.
        self._runs = InFlight()

    def bind_resolution(
        self, *, env: Mapping[str, str] | None, secrets: Any = None
    ) -> None:
        """Point credential resolution at a bridge's own environment and secrets.

        Called by :meth:`~modelpass.Bridge.adapter_for` on an adapter it loaded
        itself, and the reason it exists is the same one that keeps the key off
        :class:`~modelpass.preflight.PreflightPlan`: this adapter resolves its
        own credential, so it has to be told *which* environment and *which*
        secrets file are this bridge's. Without it a bridge pointed at a test
        root or at an application's own directory would read the user's real
        ``~/.modelpass/secrets.toml``.
        """
        self._env = env
        self._secrets = secrets

    # --- availability ----------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Whether ``anthropic`` can be imported -- without importing it."""
        return importlib.util.find_spec(_MODULE) is not None

    def _client(self, connection: Connection) -> Any:
        """The vendor client for this connection, built with an explicit key."""
        key = resolve_credential(connection, self._env, secrets=self._secrets)
        return self._client_factory(api_key=key, base_url=connection.base_url)

    def _async_client(self, connection: Connection) -> Any:
        """The vendor's *async* client, built the same way and never ambient."""
        key = resolve_credential(connection, self._env, secrets=self._secrets)
        return self._async_client_factory(api_key=key, base_url=connection.base_url)

    # --- preflight -------------------------------------------------------------

    def preflight(self, request: RunRequest) -> Receipt:
        """The three API checks of :func:`~modelpass.preflight.api_preflight`.

        Nothing here launches, logs in, or spends. ``runtime_available`` means
        the vendor SDK imports; ``account`` carries a key fingerprint and never a
        key; ``plan_name`` and ``binary`` are ``None`` because an API key has no
        subscription plan and no executable.

        The model-list probe is **opt-in** (``options={"probe_credential":
        True}``) and cached per connection once taken. :meth:`Bridge.preflight`
        runs on every single call, and paying a network round trip per ``chat()``
        to re-learn a fact that changes when a key is rotated is the trade
        :meth:`Adapter.cached_probe` exists to refuse. ``modelpass verify`` drops
        the cache.

        The one thing this adapter adds to the three: **a note when no model
        resolves**. The Messages API has no "whatever the runtime defaults to" --
        every request names a model -- so a connection with no ``model`` can be
        written and checked, but a call against it must pass ``model=``. That is
        a note rather than a failure on purpose: a keyless-model connection is a
        perfectly good thing to configure (``modelpass connect`` writes one, and
        ``Bridge.chat(model=...)`` is the documented way to use it), and failing
        its preflight would make setup refuse a state setup is allowed to be in.
        The refusal happens at :meth:`run`, still before a single byte is sent.
        """
        probe = None
        if request.options.get("probe_credential"):
            probe = self.cached_probe(request)
        receipt = api_preflight(
            request.connection,
            request.plan,
            self._env,
            runtime_available=self.is_available(),
            probe=probe,
            secrets=self._secrets,
        )
        if receipt.ok and not (request.model or request.connection.model):
            return replace(receipt, notes=(*receipt.notes, _NO_MODEL.format(
                name=request.connection.name
            )))
        return receipt

    def probe(self, request: RunRequest) -> ModelListProbe | None:
        """``models.list()`` -- token-free, and the only pre-run proof a key is live.

        Call it through :meth:`Adapter.cached_probe`, which is what
        :meth:`preflight` does.
        """
        try:
            listing = self._client(request.connection).models.list(limit=_PROBE_LIMIT)
        except SubpassError:
            raise
        except Exception as exc:
            return ModelListProbe(ok=False, detail=f"{type(exc).__name__}: {exc}")
        ids = []
        for item in getattr(listing, "data", listing) or ():
            model_id = getattr(item, "id", None)
            if isinstance(model_id, str):
                ids.append(model_id)
        return ModelListProbe(ok=True, models=tuple(ids[:_PROBE_LIMIT]))

    def support_for(
        self, capability: Capability, options: Mapping[str, Any]
    ) -> Support | None:
        """No opinion: nothing in ``options`` changes which vendor surface runs.

        Stated rather than inherited, because the two adapters this one was
        written beside both answer here and a reader should not have to wonder
        whether the omission was a decision. On ``openai-sdk`` an option selects
        a transport and the transports differ in what they can do; there is one
        transport here.
        """
        del capability, options
        return None

    def cache_eligibility(self, request: RunRequest | SessionRequest) -> CacheEligibility:
        """Whether this call's prefix can cache -- a fact here, not an estimate.

        The difference from ``anthropic-sdk`` is worth naming, since the two
        answers now differ *in kind* (validation §2.3). There, modelpass guesses
        from prompt length against a model floor because the runtime gives it
        nothing better. Here the caller *says* where the cacheable prefix ends,
        the marker travels to the vendor untouched, and
        :attr:`CacheEligibility.explicit_breakpoints` turns the estimate into the
        thing the caller asked for. The floor still reports, because a breakpoint
        below the model's minimum cacheable prefix is still not cached.

        ``ttl`` is ``None``: modelpass pins no lifetime on this runtime. A caller
        may pin one itself -- ``cache_control={"type": "ephemeral", "ttl":
        "1h"}`` is carried through with the block -- and ``ttl_detail`` says so
        rather than modelpass claiming a pin it did not make.
        """
        floor, floor_source = cache_floor_tokens(request.model)
        pinned = _pinned_ttls(request)
        detail = "modelpass pins no TTL on anthropic-api; the endpoint's own default applies"
        if pinned:
            detail = (
                "modelpass pins no TTL on anthropic-api; the caller pinned "
                f"{', '.join(repr(t) for t in pinned)} on its own cache_control block(s), "
                "which travels "
                "with the block"
            )
        return CacheEligibility(
            system_prompt_chars=request.system_prompt_chars,
            floor_tokens=floor,
            floor_source=floor_source,
            tools_declared=request.wants_tools,
            ttl=None,
            ttl_detail=detail,
            explicit_breakpoints=request.cache_breakpoints,
        )

    # --- the run ---------------------------------------------------------------

    def sampling_params(self, request: RunRequest) -> dict[str, Any]:
        """The sampling half of the request, in the vendor's own spelling (R5).

        Called once per round and merged into everything else. **The per-model
        decisions are not taken here**: they are taken once, in
        :func:`~modelpass.sampling_rules.plan_sampling`, whose answer this method
        translates and does not revisit. That split is what makes the receipt
        true -- the bridge builds its applied-vs-requested report from the same
        call, so a field this method sends and a field the report names cannot
        be two different sets.

        What is left here is the translation, which is genuinely vendor
        knowledge and belongs nowhere else:

        * ``max_output_tokens`` -> ``max_tokens``, a **required** field on the
          Messages API. The rules table supplies
          :data:`DEFAULT_MAX_OUTPUT_TOKENS` when nobody set one, and the receipt
          says it did rather than leaving a 4096-token ceiling to be discovered
          in a truncated answer.
        * ``temperature`` / ``top_p`` / ``top_k`` -> themselves. All three are in
          ``anthropic`` 0.97.0's ``Messages.create`` signature (read
          2026-09-13; see :data:`SDK_VERSION_READ`).
        * ``reasoning_effort`` -> ``output_config={"effort": ...}``.
          ``OutputConfigParam.effort`` is
          ``Literal["low","medium","high","xhigh","max"]`` in that same SDK, so
          all three positions of modelpass's dial are accepted values. It shares
          ``output_config`` with structured output's ``format``, which is why
          :meth:`request_params` merges rather than assigns -- an effort that
          silently replaced a schema would be the worst kind of quiet.

          Note what this is *not*: the ``thinking`` parameter. 0.97.0 has both,
          and ``ThinkingConfigParam`` is ``{"type": "enabled", "budget_tokens":
          N}`` / ``{"type": "adaptive"}`` / ``{"type": "disabled"}`` -- a token
          budget, not an effort dial. Mapping three abstract positions onto
          three invented budgets is a number modelpass would be making up, and
          a RAG evaluation harness already pays for having had to -- its
          effort-to-budget table carries a ``max_tokens`` bump and a temperature
          override that ride along. ``effort`` is the vendor's own
          three-position dial and needs no invention, so it is the one used.
        """
        plan = plan_sampling(
            request.effective_sampling,
            self.runtime,
            request.model or request.connection.model,
        )
        params: dict[str, Any] = {}
        for name, value in plan.applied.items():
            if name == "max_output_tokens":
                params["max_tokens"] = value
            elif name == "reasoning_effort":
                params["output_config"] = {"effort": value}
            else:
                params[name] = value
        params.setdefault("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        return params

    def request_params(
        self, request: RunRequest, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """One round's request, assembled. Pure: no network, no client."""
        params: dict[str, Any] = {
            "model": request.model or request.connection.model,
            "messages": messages,
        }
        sampling = self.sampling_params(request)
        params.update(sampling)
        system = system_param(request)
        if system is not None:
            params["system"] = system
        if request.tools:
            params["tools"] = tool_params(request.tools)
        config = output_config(request)
        if config is not None:
            # D13: a schema and tools are never combined. The bridge refuses the
            # combination before an adapter sees it; the assert-by-construction
            # here is that the two branches cannot both fire on one request.
            #
            # Merged rather than assigned: ``output_config`` carries both the
            # schema's ``format`` and the reasoning dial's ``effort`` in
            # anthropic 0.97.0, and a schema-bound call that also asked to think
            # harder must send both. Assignment here would drop whichever of the
            # two was written first, silently, which is the failure R5 exists to
            # end wearing a different hat.
            params["output_config"] = {
                **sampling.get("output_config", {}),
                **config,
            }
        return params

    def run(self, request: RunRequest) -> Iterator[AgentEvent]:
        """One stateless call, plus however many tool rounds the model asks for.

        The two refusals here are both :class:`PreflightFailed`, raised before
        the client is built and therefore before anything is sent: a credential
        that no longer resolves, and a run with no model anywhere. Neither is a
        vendor failure, so neither is a terminal event -- rule 5's "report, do
        not raise" is about the vendor saying no, and nothing has been asked yet.
        """
        if not (request.model or request.connection.model):
            raise PreflightFailed(_NO_MODEL.format(name=request.connection.name))
        try:
            client = self._client(request.connection)
        except SubpassError:
            raise
        except Exception as exc:  # pragma: no cover - defensive: a client constructor
            raise VendorRunFailed(
                f"the {_PACKAGE} client could not be constructed: {type(exc).__name__}: {exc}"
            ) from exc
        # This run's own cancel state (2026-09-24). It used to be one flag on
        # the adapter, cleared by every run() and set by every teardown, so
        # one run ending stopped whichever other run reached a round boundary.
        cancel = RunCancel()
        self._runs.add(cancel)
        return RunStream(self._loop(client, request, cancel), cancel.cancel)

    def _loop(self, client: Any, request: RunRequest, cancel: RunCancel) -> Iterator[AgentEvent]:
        messages = conversation_param(request)
        by_name = {tool.name: tool for tool in request.tools}
        final_text = ""

        while True:
            if cancel.is_set():
                yield _bare_terminal(TerminalStatus.CANCELLED, "cancelled by caller")
                return

            params = self.request_params(request, messages)
            message = None
            try:
                with client.messages.stream(**params) as stream:
                    for event in stream:
                        yield from stream_events(event)
                    message = stream.get_final_message()
            except SubpassError:
                raise
            except Exception as exc:
                status, reason = _vendor_failure(exc)
                code, after = vendor_error_facts(exc)
                yield _bare_terminal(
                    status, reason, status_code=code, retry_after=after
                )
                return

            yield UsageEvent(
                usage=token_usage(getattr(message, "usage", None)),
                scope=UsageScope.DELTA,
            )

            blocks = list(getattr(message, "content", None) or ())
            text = "".join(
                getattr(block, "text", "")
                for block in blocks
                if getattr(block, "type", None) == "text"
            )
            if text:
                final_text = text

            calls = [block for block in blocks if getattr(block, "type", None) == "tool_use"]
            # A run that declared no tools has nothing to answer with, and
            # replying "not declared" round after round would be a loop with a
            # bill on it. The model should never ask -- it was sent no tools --
            # so if it does, the run ends and terminal_status() says why.
            stopped_for_tools = getattr(message, "stop_reason", None) == "tool_use"
            if not (request.tools and stopped_for_tools and calls):
                yield from self._finish(request, message, final_text)
                return

            # The assistant turn goes back verbatim -- thinking blocks and their
            # signatures included -- before the results that answer it.
            messages.append({"role": "assistant", "content": [_block_param(b) for b in blocks]})
            results: list[dict[str, Any]] = []
            for block in calls:
                call_id = getattr(block, "id", "") or ""
                name = getattr(block, "name", "") or ""
                raw = getattr(block, "input", None)
                arguments = dict(raw) if isinstance(raw, Mapping) else {}
                yield ToolCallEvent(
                    name=name,
                    arguments=arguments,
                    id=call_id,
                    server=CALLER_TOOL_SERVER,
                )
                tool = by_name.get(name)
                payload = (
                    invoke_handler(tool, arguments) if tool is not None else _missing_tool(name)
                )
                is_error = bool(payload.get("is_error"))
                yield ToolResultEvent(
                    id=call_id,
                    name=name,
                    content=flatten_mcp_result(payload),
                    is_error=is_error,
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": payload.get("content", []),
                        "is_error": is_error,
                    }
                )
            # Every result in **one** user message. Splitting them across
            # messages is what teaches the model to stop asking for tools in
            # parallel, which is a behaviour change nobody asked for.
            messages.append({"role": "user", "content": results})

    def arun(self, request: RunRequest) -> AsyncIterator[AgentEvent]:
        """:meth:`run`, on the caller's loop, with the vendor's async client (R1).

        **Native, not the base class's thread** -- and the difference a consumer
        can see is the tool loop: a coroutine handler is awaited right here, on
        the loop that called ``achat``, so an application's async tools reach
        their own session, pool and cache without a thread hop or a
        ``run_coroutine_threadsafe`` in sight.

        Everything else is :meth:`run`'s, statement for statement: the same
        request assembly, the same events, the same terminal mapping, the same
        two :class:`PreflightFailed` refusals before the client exists.
        :meth:`run` is **not** re-implemented over this (R1's rejected
        alternative applies to adapters too, for the same reason: a sync caller
        inside a running loop).
        """
        if not (request.model or request.connection.model):
            raise PreflightFailed(_NO_MODEL.format(name=request.connection.name))
        try:
            client = self._async_client(request.connection)
        except SubpassError:
            raise
        except Exception as exc:  # pragma: no cover - defensive: a client constructor
            raise VendorRunFailed(
                f"the {_PACKAGE} async client could not be constructed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        cancel = RunCancel()
        self._runs.add(cancel)
        return native_run(self, self._aloop(client, request, cancel), cancel.cancel)

    async def _aloop(
        self, client: Any, request: RunRequest, cancel: RunCancel
    ) -> AsyncIterator[AgentEvent]:
        """:meth:`_loop` with ``await`` where it blocks. Same rounds, same events."""
        messages = conversation_param(request)
        by_name = {tool.name: tool for tool in request.tools}
        final_text = ""

        while True:
            if cancel.is_set():
                yield _bare_terminal(TerminalStatus.CANCELLED, "cancelled by caller")
                return

            params = self.request_params(request, messages)
            message = None
            try:
                async with client.messages.stream(**params) as stream:
                    async for event in stream:
                        for out in stream_events(event):
                            yield out
                    message = await stream.get_final_message()
            except SubpassError:
                raise
            except Exception as exc:
                status, reason = _vendor_failure(exc)
                code, after = vendor_error_facts(exc)
                yield _bare_terminal(
                    status, reason, status_code=code, retry_after=after
                )
                return

            yield UsageEvent(
                usage=token_usage(getattr(message, "usage", None)),
                scope=UsageScope.DELTA,
            )

            blocks = list(getattr(message, "content", None) or ())
            text = "".join(
                getattr(block, "text", "")
                for block in blocks
                if getattr(block, "type", None) == "text"
            )
            if text:
                final_text = text

            calls = [block for block in blocks if getattr(block, "type", None) == "tool_use"]
            stopped_for_tools = getattr(message, "stop_reason", None) == "tool_use"
            if not (request.tools and stopped_for_tools and calls):
                for out in self._finish(request, message, final_text):
                    yield out
                return

            messages.append({"role": "assistant", "content": [_block_param(b) for b in blocks]})
            results: list[dict[str, Any]] = []
            for block in calls:
                call_id = getattr(block, "id", "") or ""
                name = getattr(block, "name", "") or ""
                raw = getattr(block, "input", None)
                arguments = dict(raw) if isinstance(raw, Mapping) else {}
                yield ToolCallEvent(
                    name=name,
                    arguments=arguments,
                    id=call_id,
                    server=CALLER_TOOL_SERVER,
                )
                tool = by_name.get(name)
                # The one line this whole face exists for: the caller's own
                # coroutine, awaited on the caller's own loop.
                payload = (
                    await ainvoke_handler(tool, arguments)
                    if tool is not None
                    else _missing_tool(name)
                )
                is_error = bool(payload.get("is_error"))
                yield ToolResultEvent(
                    id=call_id,
                    name=name,
                    content=flatten_mcp_result(payload),
                    is_error=is_error,
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": payload.get("content", []),
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

    def _finish(
        self, request: RunRequest, message: Any, final_text: str
    ) -> Iterator[AgentEvent]:
        """The structured answer, if one was asked for, and then the terminal."""
        status, reason = terminal_status(message)
        if request.schema is not None:
            event, failure = build_structured_event(
                request.schema,
                raw=final_text,
                schema_name=resolve_schema_name(request.schema, request.schema_name),
            )
            if event is not None:
                yield event
            if failure is not None and status is TerminalStatus.OK:
                status, reason = TerminalStatus.ERROR, failure
        yield _bare_terminal(status, reason)

    def cancel(self) -> None:
        """Stop the loop at the next round boundary (D10).

        Two halves, and only one of them is this method. The bridge closes the
        iterator, which exits the ``with`` around the stream and tears the
        in-flight HTTP response down -- the transport-level floor the contract
        documents. This flag is the graceful half: a loop that is between rounds
        stops there and reports ``cancelled`` rather than spending another round
        first, which is what a caller pressing stop during a long tool loop
        actually asked for.

        **Every run in flight on this adapter.** A bridge cancels one run through
        the stream :meth:`run` returned, never through this method; what calls
        this is a caller holding the adapter, for whom there is no other meaning.
        """
        self._runs.cancel_all()

    # --- sessions: refused, and the refusal says what to use instead ------------

    def open_session(self, request: SessionRequest) -> SessionHandle:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.new_chat()"))

    def resume_session(self, request: SessionRequest) -> SessionHandle:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.resume_chat()"))

    def list_sessions(self, request: SessionRequest) -> tuple[SessionInfo, ...]:
        raise AdapterNotImplemented(_NO_SESSIONS.format(call="bridge.list_sessions()"))


#: The one sentence all three session refusals share (validation §2.6, answer 1).
#: Same shape as the base contract's wording -- name the runtime, name the call,
#: say what still works -- with the substitution this runtime actually has.
_NO_SESSIONS = (
    f"the adapter for runtime {Runtime.ANTHROPIC_API.value!r} has not implemented "
    "{call}, and will not: a session rests on the runtime holding the "
    "conversation (D14) and an API endpoint holds none -- every request carries "
    "its whole history. Use bridge.chat(history=[...]), which is the documented "
    "multi-turn shape against a stateless runtime. Stateless bridge.chat() calls "
    "are unaffected"
)


def _pinned_ttls(request: RunRequest | SessionRequest) -> tuple[str, ...]:
    """The distinct ``ttl`` values the caller pinned on its own breakpoints."""
    seen: list[str] = []
    for message in getattr(request, "messages", ()):
        for block in message.blocks:
            control = block.cache_control
            ttl = getattr(control, "ttl", None)
            if isinstance(ttl, str) and ttl not in seen:
                seen.append(ttl)
    return tuple(seen)


#: Said once, because the receipt and the refusal must not drift apart.
_NO_MODEL = (
    "connection {name!r} names no model and this call passed none. The Messages "
    "API has no default model, so there is nothing to send: set a model on the "
    "connection, or pass model= to the call"
)


def _vendor_failure(exc: Exception) -> tuple[TerminalStatus, str]:
    """Classify a vendor exception into a terminal status and a reason.

    A 429 is :attr:`TerminalStatus.QUOTA_EXHAUSTED` and not an error, because it
    is the outcome a quota failover exists to act on -- typed off the status code
    the SDK carries rather than matched on the message, which is the substring
    matching that burned a real allowance once already (R6).
    """
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return (
            TerminalStatus.QUOTA_EXHAUSTED,
            f"the endpoint reported HTTP 429 (rate limit / quota): {exc}",
        )
    if isinstance(status_code, int):
        return TerminalStatus.ERROR, f"HTTP {status_code}: {exc}"
    return TerminalStatus.ERROR, f"{type(exc).__name__}: {exc}"


def _default_client(*, api_key: str, base_url: str | None) -> Any:
    """``anthropic.Anthropic``, constructed with an explicit key and nothing ambient.

    ``base_url`` is passed only when the connection carries one, so a connection
    without one gets the SDK's own default endpoint rather than a ``None`` the
    constructor has to interpret.
    """
    try:
        sdk = importlib.import_module(_MODULE)
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise RuntimeNotAvailable(Runtime.ANTHROPIC_API.value, _EXTRA, _PACKAGE) from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    return sdk.Anthropic(**kwargs)


def _default_async_client(*, api_key: str, base_url: str | None) -> Any:
    """``anthropic.AsyncAnthropic``, under exactly the rules of :func:`_default_client`.

    The vendor ships the async client in the same package and with the same
    surface, so ticket 1.13's native ``arun`` needs no second dependency and no
    second pin -- which is why the extra is unchanged.
    """
    try:
        sdk = importlib.import_module(_MODULE)
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise RuntimeNotAvailable(Runtime.ANTHROPIC_API.value, _EXTRA, _PACKAGE) from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    return sdk.AsyncAnthropic(**kwargs)
