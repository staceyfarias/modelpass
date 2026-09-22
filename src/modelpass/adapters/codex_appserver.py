"""The Codex ``app-server`` transport: JSON-RPC over the CLI's own stdio surface.

Slices 2, 3 and 6 of the 2026-08-31 app-server transport migration. **Wired
since S4**: :meth:`~modelpass.adapters.openai.OpenAIAdapter.run` drives this
transport when the caller passes ``options={"transport": "app-server"}``, and
``codex exec`` remains the default. **S6 added in-process caller tools** here --
see the dynamic-tools section below -- so this transport now does the one thing
``codex exec`` genuinely cannot. Sessions (S5) still run on exec, and flipping
the default is its own slice (S7).

**What it is.** ``codex app-server --listen stdio://`` is the same binary the
exec transport already launches, speaking newline-delimited JSON-RPC instead of
JSON Lines. It carries the prompt as a protocol *field*, which is the fact that
retires the S1 defect at the root: there is no command line for ``cmd.exe`` to
cut at the first newline (see ``_send_prompt`` in ``adapters/openai.py`` for the
reproduction, 2026-08-31). Argv here is four fixed, newline-free elements, so
the ``.cmd`` shim hazard cannot apply to it at all -- and the JSON-RPC content
rides stdin, where a newline only ever means *end of message* because JSON
escapes every other one.

**Why modelpass owns this seam rather than depending on ``openai-codex``**
(decided 2026-08-31, recorded in the plan of record):

* Its ``CodexClient.start()`` builds the child environment from
  ``os.environ.copy()`` and merges the caller's overrides into it. modelpass hands
  a runtime a **scrubbed** environment and promises the receipt describes what
  the child actually got (D2); a client that re-adds the ambient environment
  underneath breaks that promise silently, which is the exact failure shape this
  library exists to prevent.
* Its typed parameter objects already lag the wire -- ``ThreadStartParams`` has
  no ``dynamicTools`` field although the server accepts one (verified live
  2026-08-31), so S6 would be passing raw dicts through a typed API anyway.
* Owning the transport is how every other adapter path stays offline-testable:
  one injectable spawn seam and the whole protocol is exercisable from a
  scripted fake, with no vendor package, no subprocess and no credential.

The SDK remains a perfectly good *binary source* through its pinned CLI, exactly
as it is today for ``codex exec``.

**Wire format**, verified live on 2026-08-31 against codex 0.151.0-alpha.7.1 and
confirmed against ``codex-rs/app-server-protocol/schema/json`` in openai/codex:
one JSON object per line, in both directions, with no ``jsonrpc`` version field
(``JSONRPCRequest.json`` requires ``id`` and ``method`` only). Three shapes
arrive from the server and the discriminator is which of ``method`` and ``id``
is present:

===================  ==========================================================
Line carries         What it is
===================  ==========================================================
``method`` + ``id``  a **request to the client**, which must be answered with
                     ``{"id": <same>, "result": {...}}``
``method``, no id    a notification -- the whole event stream is these
``id``, no method    the response to one of our requests
===================  ==========================================================

**Handshake**, and it is ordered: ``initialize`` (a request) must be answered
before the ``initialized`` notification is sent, and nothing else may go before
either. ``capabilities.experimentalApi: true`` is what admits the dynamic-tools
surface S6 needs -- ``InitializeCapabilities.experimentalApi`` is documented as
*opt into receiving experimental API methods and fields*, so this is
**experimental vendor surface by the vendor's own label** and is re-verified per
release, not assumed (standing caution, plan of record).

**Threading model: one reader thread, fanning into queues, with a lock around
stdin writes.** The same shape the ``openai-codex`` SDK uses, adopted after
weighing the simpler synchronous alternative and rejecting it for a specific
reason: ``turn/start``'s response does not arrive until the turn is *over*, and
the entire event stream -- every text delta, every tool item -- arrives as
notifications in between. A client that only read the pipe while blocked inside
``request()`` could still answer a server request in time, but it could not hand
a caller a single delta before the turn ended, and it would sit on a stdout pipe
nobody drains for the length of a long turn. Streaming is the whole point of
adopting this transport, so the reader runs on its own thread and ``request()``
blocks on a per-id waiter it fills.

Two consequences, both load-bearing:

* **The server-request handler runs on the reader thread**, so it must not call
  :meth:`AppServerClient.request` -- the reader is the only thing that delivers
  responses, and waiting for one from inside it is a deadlock rather than a slow
  call. Since S6 that constraint is inherited by **the caller's own tool
  functions**, which :class:`CallerToolDispatcher` runs directly here; it is
  restated on that class, because a caller reads it there and not up here.
* A server request arriving *mid-*``request()`` is answered while the caller is
  still blocked, which is exactly the case approvals and ``item/tool/call``
  produce.

**stderr is drained into a bounded deque by a second thread**, and that is a
deliberate departure from ``_default_session_spawn``'s temp file. The reasoning
there was that draining two pipes from one thread deadlocks -- true, and this
module answers it by having two threads instead. What a file cannot answer is
lifetime: an exec turn is one short process and its stderr file dies with it,
while an app-server child lives for a whole session and a file behind it would
grow without limit. A ``deque(maxlen=...)`` keeps the tail, which is all a
failure report ever wants, at a fixed cost.

**Refusing to authorize by default.** The two approval requests the server can
send -- ``item/commandExecution/requestApproval`` and
``item/fileChange/requestApproval`` -- are answered ``{"decision": "decline"}``
by :func:`decline_approvals` unless the caller injects a handler of their own.
Decline rather than accept, because an unattended library must not authorize
execution or a file write that the caller never saw; the decision enum
(``CommandExecutionApprovalResponse.json``, schema-sourced) offers ``cancel``
too, which also interrupts the turn, and ``decline`` is chosen because the
agent is allowed to continue and report what it could not do. A tool-bearing run
injects :class:`CallerToolDispatcher` instead, which answers ``item/tool/call``
itself and **delegates everything else back to** :func:`decline_approvals`, so
adding tools never quietly stops declining approvals. Unknown request methods
are answered ``{}``, which is what the vendor's own SDK does and keeps a server
waiting on an unrecognized round trip from hanging the turn.

Everything here is offline-testable through ``spawn=``: ``tests/
test_codex_appserver.py`` drives the whole protocol against a scripted fake and
never launches a process.

--------------------------------------------------------------------------------

**The event mapping** (S3) lives in the second half of this module:
:func:`map_appserver_event` is pure and total in the spirit of
``map_codex_event``, and :class:`AppServerTurn` is the stateful fold over it,
exactly as ``_TurnOutcome`` is for the exec transport. Two semantic traps are
worth reading before the code, because both are the kind that produce a
plausible wrong number rather than an error.

**Trap one: ``total`` is thread-cumulative, not this turn's.**
``thread/tokenUsage/updated`` carries ``{tokenUsage: {last, total,
modelContextWindow}}`` (``ThreadTokenUsageUpdatedNotification.json``). ``total``
is the running total for the whole **thread**, so on a resumed conversation it
includes every previous turn -- reporting it as the turn's usage would bill a
tenth turn for all ten. ``last`` is the most recent model response's own
breakdown. modelpass emits one interim :class:`~modelpass.types.UsageEvent` per
update carrying ``last``, with ``scope=delta``, so the bridge's ordinary
accumulation lands on the turn's real usage and a guard can fire mid-turn --
which is the capability the exec transport does not have. The thread total is
not discarded: it is recorded on :attr:`AppServerTurn.thread_total_usage` and
carried in the accompanying ``vendor_event``, so S4 can report it without
re-deriving it. Field names are camelCase on the wire and are mapped by name
(driven live 2026-08-31; ``cacheWriteInputTokens`` was present in every captured
breakdown but only ever ``0``, so its *name* is verified and a non-zero value is
still undriven). ``inputTokens`` is **inclusive** of ``cachedInputTokens`` on
this vendor -- :func:`token_usage_from_breakdown` is where that measurement and
the correction it forces are written down.

**Trap two: the completed message repeats the deltas.**
``item/agentMessage/delta`` streams the answer and ``item/completed``
(``agentMessage``) then carries the *whole* text again. Emitting both would
double every answer for a caller concatenating ``text_delta`` events. So the
rule is: a completed ``agentMessage`` whose item id already streamed a delta
becomes a ``vendor_event`` carrying the full item -- nothing dropped, no text
counted twice -- and its text is recorded on
:attr:`AppServerTurn.agent_messages`, which is where the structured answer is
read from, the same place ``_TurnOutcome.agent_messages`` reads it. A completed
``agentMessage`` with **no** prior delta becomes a ``text_delta``, because a
server that did not stream still produced an answer and suppressing it would
lose the run's output entirely. Both paths are tested.

Two mapping choices are flagged for architect review rather than buried:

* **``server="codex"`` for ``commandExecution``, the MCP server's name for
  ``mcpToolCall``, and ``server="caller"`` for ``dynamicToolCall``** -- the last
  because a dynamic tool is answered by a function in *this* process, which is
  precisely what :data:`~modelpass.types.CALLER_TOOL_SERVER` means everywhere else
  in modelpass. The item's own ``namespace`` field is preserved in the tool call's
  arguments rather than being promoted to ``server``.
* **``name="command_execution"``**, the exec transport's spelling, not the
  app-server's ``commandExecution``. The plan keeps one runtime identity across
  the transport flip, so a consumer filtering tool events by name must not have
  to change when the transport underneath it does.

And one shape the brief expected is **not on this protocol**: there is no
``turn/failed`` notification in v2. ``ServerNotification.json`` lists
``turn/completed`` and ``error`` and nothing between them; a failed turn arrives
as ``turn/completed`` with ``turn.status == "failed"`` and a populated
``turn.error`` (``TurnCompletedNotification.json``: ``TurnStatus`` is
``completed | interrupted | failed | inProgress``). The separate ``error``
notification carries ``willRetry``, and a retryable one is deliberately *not*
read as a turn failure -- a retry follows it, and a run reported failed because
of a transient error it recovered from would be a worse lie than a slow one.
Both readings are schema-sourced, not yet driven.
"""

from __future__ import annotations

import asyncio
import datetime
import inspect
import json
import queue
import re
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import InvalidTool, VendorRunFailed
from ..preflight import check_launch_args
from ..runtimes import Runtime
from ..tools import ToolDef
from ..types import (
    CALLER_TOOL_SERVER,
    AgentEvent,
    Message,
    Role,
    SessionInfo,
    TerminalStatus,
    TextDeltaEvent,
    ThinkingEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    VendorEvent,
)
from ._mcp_result import flatten_mcp_result, mcp_result_payload

__all__ = [
    "APPROVAL_REQUEST_METHODS",
    "DYNAMIC_TOOL_CALL_METHOD",
    "RESPONSES_API_NAME_MAX",
    "RESPONSES_API_NAME_PATTERN",
    "THREAD_ITEMS_LIST_METHOD",
    "THREAD_LIST_METHOD",
    "THREAD_READ_METHOD",
    "THREAD_RESUME_METHOD",
    "TURN_INTERRUPT_METHOD",
    "AppServerClient",
    "AppServerRequestFailed",
    "AppServerSpawnFn",
    "AppServerTimeout",
    "AppServerTransportClosed",
    "AppServerTurn",
    "CallerToolDispatcher",
    "Notification",
    "ServerRequestHandler",
    "app_server_argv",
    "check_dynamic_tool_names",
    "decline_approvals",
    "dynamic_tool_response",
    "dynamic_tools_param",
    "items_page",
    "map_appserver_event",
    "session_info_from_thread",
    "thread_item_message",
    "thread_items_history",
    "thread_items_list_params",
    "thread_list_params",
    "thread_read_params",
    "thread_resume_params",
    "threads_from_list",
    "turn_error_detail",
    "turn_interrupt_params",
]


#: How many lines of the child's stderr are kept for a failure report. Bounded
#: because this process outlives many turns; see the module docstring.
STDERR_TAIL_LINES = 200

#: Seconds to wait for the ``initialize`` response before giving up. The
#: handshake is local work -- no model call, no network -- so a child that has
#: not answered in this long is a child that is not going to.
HANDSHAKE_TIMEOUT = 30.0

#: Seconds allowed for a polite terminate, and again for the kill after it.
_SHUTDOWN_GRACE = 5.0


# --- errors --------------------------------------------------------------------
#
# All three subclass VendorRunFailed rather than introducing a new branch of the
# taxonomy: adapter contract rule 5 says a transport that dies is the vendor's
# half of the run failing, and the bridge already turns VendorRunFailed into one
# stamped terminal event carrying the usage spent so far. A new base class would
# escape that handling and reach the caller as AdapterFailed -- a bug report for
# something that is an ordinary runtime condition.


class AppServerTransportClosed(VendorRunFailed):
    """The app-server child is gone: EOF on stdout, or a closed client.

    Carries the captured stderr tail where there is one, because a child that
    died on a bad flag or a missing login says so there and nowhere else -- the
    same lesson ``codex exec resume`` taught on 2026-08-30, which is why
    sessions started capturing stderr at all.
    """


class AppServerTimeout(VendorRunFailed):
    """A request went unanswered for longer than its caller allowed."""


class AppServerRequestFailed(VendorRunFailed):
    """The server answered one of our requests with a JSON-RPC error.

    ``code`` and ``message`` are the vendor's own (``JSONRPCErrorError.json``:
    both required, ``data`` free-form). They are kept as fields rather than
    flattened into the message so a caller can branch on them -- S5 needs
    exactly that to tell "no rollout found for thread id" (``-32600``) from any
    other failed ``thread/resume``.
    """

    def __init__(
        self, method: str, code: int, message: str, data: Any = None
    ) -> None:
        self.method = method
        self.code = code
        self.detail = message
        self.data = data
        super().__init__(f"codex app-server refused {method!r}: {message} (code {code})")


# --- the protocol ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Notification:
    """One server -> client notification: a method name and its params.

    Deliberately not normalized here. Turning these into modelpass events is S3's
    job and lives in its own pure function, so the transport can be tested
    without an opinion about the event vocabulary.
    """

    method: str
    params: Mapping[str, Any] = field(default_factory=dict)


#: A caller-injected answer to a server -> client request. Runs **on the reader
#: thread** -- see the module docstring -- and returns the ``result`` dict the
#: server will receive.
ServerRequestHandler = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]

#: The server -> client requests that ask permission to do something. Both are
#: answered with ``{"decision": ...}`` (``CommandExecutionRequestApprovalResponse
#: .json`` / ``FileChangeRequestApprovalResponse.json``, schema-sourced
#: 2026-08-31, not yet driven).
APPROVAL_REQUEST_METHODS = frozenset(
    {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }
)


def decline_approvals(method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
    """The default server-request handler: refuse approvals, answer the rest emptily.

    ``decline`` and not ``accept``, because modelpass is a library running
    unattended inside somebody else's process: authorizing a shell command or a
    file write that the caller never saw is not a default anything can opt out
    of afterwards. It is also not ``cancel``, the enum's other refusal, which
    interrupts the whole turn -- a declined command lets the agent carry on and
    report what it could not do, which is more information for the same safety.

    Anything else gets ``{}``. An unrecognized request left unanswered would
    hang whatever the server is waiting on, and an empty result is what the
    vendor's own SDK sends.
    """
    del params  # the decision does not depend on what is being asked
    if method in APPROVAL_REQUEST_METHODS:
        return {"decision": "decline"}
    return {}


# --- caller tools as the app-server's dynamic tools (S6) --------------------------
#
# The full loop, driven live on 2026-08-31 and recorded in
# tests/fixtures/appserver/live-capture-2026-08-31.jsonl:
#
#   1. register  thread/start  {"dynamicTools": [{"type": "function", "name",
#                              "description", "inputSchema"}]}
#   2. call      server -> client REQUEST "item/tool/call"
#                              {threadId, turnId, callId, namespace, tool,
#                               arguments}
#   3. answer    {"contentItems": [{"type": "inputText", "text": ...}],
#                 "success": true}
#   4. report    item/started + item/completed carrying a "dynamicToolCall"
#                item, mapped by the S3 half of this module.
#
# Registration requires ``capabilities.experimentalApi: true`` at ``initialize``,
# which :meth:`AppServerClient._handshake` already sends, and the vendor's own
# field description calls that surface experimental -- so every shape here is
# dated and re-verified per release rather than assumed.


#: The server -> client request that asks this process to run one caller tool.
#: Driven live 2026-08-31. Not in :data:`APPROVAL_REQUEST_METHODS`: it is not a
#: permission question, and its answer is a result rather than a decision.
DYNAMIC_TOOL_CALL_METHOD = "item/tool/call"

#: What the vendor accepts as a dynamic tool name, **its own regex**, read from
#: the shipped ``codex.exe`` 0.151.0-alpha.7.1 on 2026-08-31 next to the error it
#: formats: ``<field> must match ^[a-zA-Z0-9_-]+$ to match Responses API: ...``.
RESPONSES_API_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

#: Longest name modelpass will send. The vendor has its own ceiling -- the binary
#: carries ``<field> must be at most <N> characters to match Responses API`` --
#: but ``N`` is a formatted integer rather than a readable string, so this is
#: **modelpass's** cap, the same 64 :data:`~modelpass.tools.TOOL_NAME_PATTERN`
#: already enforces, and not a claim about the vendor's number.
RESPONSES_API_NAME_MAX = 64


def check_dynamic_tool_names(tools: Sequence[ToolDef]) -> None:
    """Refuse names ``thread/start`` would reject, **before anything launches**.

    Mirrors :func:`~modelpass.adapters.openai.mcp_config_args`'s validation of MCP
    server names, and for the same reason: a name the vendor will not take is a
    caller mistake, and a caller mistake should surface where it was written
    rather than as a JSON-RPC error from a child process that has already been
    spawned.

    Every rule here is the vendor's own, read from the shipped ``codex.exe``
    0.151.0-alpha.7.1 on 2026-08-31 -- the same first-party source
    ``_QUOTA_MARKERS`` came from. Its dynamic-tool validator formats:
    ``dynamic tool name must not be empty``, ``dynamic tool name has
    leading/trailing whitespace``, ``duplicate dynamic tool name``, ``dynamic
    tool name is reserved``, and ``<field> must match ^[a-zA-Z0-9_-]+$`` /
    ``must be at most <N> characters`` -- both suffixed *to match Responses API*,
    which is where the constraint actually comes from.

    **In practice this is a checked restatement, and that is the point.**
    :data:`~modelpass.tools.TOOL_NAME_PATTERN` is strictly narrower already (letter
    first, at most 64, no ``__``), so a name that reached a :class:`ToolDef` will
    pass here. Writing the vendor's rule down separately is what makes a future
    loosening of modelpass's pattern fail *here*, loudly, instead of at
    ``thread/start`` on somebody's machine. Duplicates are checked for the same
    reason -- :func:`~modelpass.tools.normalize_tools` already rejects them, and
    this path must not be the one that quietly stops.

    **One vendor rule cannot be pre-checked: ``dynamic tool name is reserved``.**
    The binary carries the message but no static list of reserved names, and the
    reserved set is the thread's own active toolbelt, which depends on the
    session's configuration and is not knowable before the thread exists. So a
    collision with a Codex built-in is refused *by the server*, and
    :meth:`~modelpass.adapters.openai.OpenAIAdapter._run_app_server` is where that
    refusal is turned into a message naming the tools modelpass sent.
    """
    seen: set[str] = set()
    for tool in tools:
        name = tool.name
        if not name:
            raise InvalidTool(
                "a dynamic tool name must not be empty: codex app-server rejects "
                "the whole thread/start over it"
            )
        if name != name.strip():
            raise InvalidTool(
                f"tool name {name!r} has leading or trailing whitespace, which "
                "codex app-server rejects ('dynamic tool name has "
                "leading/trailing whitespace')"
            )
        if not RESPONSES_API_NAME_PATTERN.match(name):
            raise InvalidTool(
                f"tool name {name!r} is not usable as a Codex dynamic tool: the "
                f"vendor requires {RESPONSES_API_NAME_PATTERN.pattern} to match the "
                "Responses API (read from codex.exe 0.151.0, 2026-08-31)"
            )
        if len(name) > RESPONSES_API_NAME_MAX:
            raise InvalidTool(
                f"tool name {name!r} is {len(name)} characters; the Responses API "
                f"caps a function name at {RESPONSES_API_NAME_MAX}"
            )
        if name in seen:
            raise InvalidTool(
                f"duplicate tool name {name!r}: codex app-server refuses a "
                "thread/start carrying two dynamic tools with one name, and the "
                "model's choice between them would be undefined anyway"
            )
        seen.add(name)


def dynamic_tools_param(tools: Sequence[ToolDef]) -> list[dict[str, Any]] | None:
    """``ThreadStartParams.dynamicTools``, or ``None`` when there are no tools.

    One entry per :class:`~modelpass.tools.ToolDef`, straight across: ``name``,
    ``description`` and the ``inputSchema`` that :meth:`ToolDef.json_schema`
    already normalizes, under ``{"type": "function"}``. The tag and the field
    names are the vendor's ``DynamicToolSpec::Function`` /
    ``DynamicToolFunctionSpec`` (read from codex.exe 0.151.0 on 2026-08-31,
    ``function`` / ``namespace`` variants, fields ``name``, ``description``,
    ``inputSchema``, ``deferLoading``), and the whole shape was **driven live**
    the same day.

    ``deferLoading`` is not sent. It is an optional field on the same struct, and
    the binary's ``deferred dynamic tool must include a namespace`` says it costs
    a namespace modelpass does not use -- so leaving it off is the default and not
    an omission.

    **``None`` rather than ``[]`` when the caller passed no tools**, and the
    caller of this function omits the key entirely on that answer. An empty list
    is a *statement* -- "this thread has zero dynamic tools" -- where absence is
    silence, and the two differ on ``thread/resume``: Codex persists the
    registration in the rollout and restores it when a resume supplies none
    (documented; not driven). A future S5 resume that sent ``[]`` could therefore
    clear a registration it only meant to leave alone. Sending nothing is the
    reading that cannot mean the wrong thing, and it is also what the live
    capture's tool-free ``thread/start`` did.
    """
    if not tools:
        return None
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.json_schema(),
        }
        for tool in tools
    ]


def dynamic_tool_response(text: str, *, success: bool) -> dict[str, Any]:
    """A ``DynamicToolCallResponse``: ``contentItems`` plus ``success``.

    Both fields are **required** by ``DynamicToolCallResponse.json`` in
    openai/codex, and the binary agrees (``struct DynamicToolCallResponse with 2
    elements``: ``contentItems``, ``success``). ``inputText`` is the text variant
    of ``DynamicToolCallOutputContentItem`` and is the one that was driven; its
    siblings ``inputImage`` (``imageUrl``) and ``inputAudio`` (``audioUrl``) are
    schema- and binary-sourced, and modelpass sends neither -- a
    :class:`~modelpass.tools.ToolDef` handler returns text or an MCP content list,
    and :func:`~modelpass.adapters._mcp_result.flatten_mcp_result` has already
    turned the latter into text by the time it reaches here.

    Sending a **well-formed** answer matters even for a failure. A malformed one
    does not kill the turn -- verified live 2026-08-31: Codex tells the model the
    dynamic tool response was invalid and the model may retry -- but "the model
    was told your tool is broken" is a much worse report than "your tool said
    no", and only the second one carries the caller's own error text.
    """
    return {"contentItems": [{"type": "inputText", "text": text}], "success": success}


class CallerToolDispatcher:
    """Runs the caller's own functions when the server asks. **On the reader thread.**

    Injected as :class:`AppServerClient`'s ``server_request_handler``, so this is
    the object that turns an ``item/tool/call`` request into a
    :func:`dynamic_tool_response`. Anything that is not a tool call falls through
    to ``fallback`` -- :func:`decline_approvals` by default -- because a
    dispatcher must not quietly stop declining the approvals it replaced.

    **Where the caller's function runs, and the one thing it must not do.** The
    module docstring records that the server-request handler runs on the reader
    thread and must never call :meth:`AppServerClient.request`; until S6 that was
    a note about modelpass's own code, and now it is a note about *the caller's*.
    A tool handler that reached back into this client's ``request()`` would wait
    for a response that only the thread it is blocking can deliver -- a deadlock,
    not a slow call. Handlers do ordinary work: compute, read a file, query a
    database, call the caller's own API. They do not drive the run they are part
    of. (They may also block for as long as they need: the reader thread is not
    the main thread, and the run's own notification drain keeps going.)

    **A handler that raises is a result, never a crash**, and that is a
    transport-level requirement rather than a kindness. This code runs on the
    reader thread; an exception escaping here would end the only thread that
    delivers responses and strand every pending request -- the run would hang or
    die on a bug in one tool. So every failure becomes a well-formed
    ``success: false`` answer carrying the error text, exactly the shape the
    model can read and retry from, and the caller hears about it as a
    ``tool_result`` with ``is_error=True``.

    That last event is emitted by modelpass rather than read off the wire, which is
    a deliberate choice: whether the server mirrors ``success: false`` onto the
    completed ``dynamicToolCall`` item is **undriven** (the live capture only
    ever succeeded), and a failure the caller never hears about is the silent
    drop this library exists to prevent. :meth:`drain_failures` hands those
    events to the run's drain, and the ids go on
    :attr:`AppServerTurn.reported_tool_failures` so the server's own completed
    item becomes a ``vendor_event`` instead of a second, duplicate result.
    Nothing dropped, nothing counted twice -- the same rule the ``agentMessage``
    dedup follows.
    """

    __slots__ = ("_failures", "_fallback", "_tools")

    def __init__(
        self,
        tools: Sequence[ToolDef],
        *,
        fallback: ServerRequestHandler | None = None,
    ) -> None:
        self._tools: dict[str, ToolDef] = {tool.name: tool for tool in tools}
        self._fallback: ServerRequestHandler = fallback or decline_approvals
        #: Failures observed on the reader thread, waiting for the run's drain.
        #: A queue rather than a list because the two threads meet here.
        self._failures: queue.SimpleQueue[ToolResultEvent] = queue.SimpleQueue()

    def __repr__(self) -> str:
        return f"<CallerToolDispatcher {sorted(self._tools)}>"

    def __call__(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if method != DYNAMIC_TOOL_CALL_METHOD:
            return self._fallback(method, params)
        return self._call(params)

    def drain_failures(self) -> list[ToolResultEvent]:
        """Every failure recorded since the last drain, in order. Never blocks."""
        drained: list[ToolResultEvent] = []
        while True:
            try:
                drained.append(self._failures.get_nowait())
            except queue.Empty:
                return drained

    # -- one call ------------------------------------------------------------------

    def _call(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Dispatch one ``item/tool/call``. Total: always answers, never raises.

        ``callId`` is the id the completed ``dynamicToolCall`` item carries --
        the live capture shows the same ``exec-<uuid>`` on the request and on
        both item notifications -- which is what lets a reported failure and the
        server's item be recognized as one call.
        """
        name = params.get("tool")
        name = name if isinstance(name, str) else ""
        call_id = params.get("callId")
        call_id = call_id if isinstance(call_id, str) else ""
        raw_arguments = params.get("arguments")
        arguments = dict(raw_arguments) if isinstance(raw_arguments, Mapping) else {}

        tool = self._tools.get(name)
        if tool is None:
            return self._failed(
                call_id,
                name,
                f"tool {name!r} was not registered by this run; modelpass sent "
                + (", ".join(sorted(self._tools)) or "no dynamic tools"),
            )
        if tool.handler is None:
            return self._failed(call_id, name, f"tool {name!r} has no handler")

        try:
            result = _invoke_handler(tool.handler, arguments)
        # Broad on purpose, and documented on ToolDef: the caller's own function
        # is running, its bugs are ordinary, and the exception's type and message
        # go to the model. See the class docstring for why this cannot be
        # narrowed -- the reader thread is what would die.
        except Exception as exc:
            return self._failed(
                call_id, name, f"tool {name!r} failed: {type(exc).__name__}: {exc}"
            )

        # The ToolDef contract, read once for every runtime: text, or an MCP
        # {"content": [...], "is_error": bool} mapping. A handler that reports
        # is_error itself took the same route a raise does -- it said no, and
        # saying so is not a crash.
        payload = mcp_result_payload(result)
        text = flatten_mcp_result(payload)
        if payload.get("is_error"):
            return self._failed(call_id, name, text)
        return dynamic_tool_response(text, success=True)

    def _failed(self, call_id: str, name: str, detail: str) -> Mapping[str, Any]:
        """Answer ``success: false``, and record the result the caller will see."""
        text = detail[:_TOOL_ERROR_MAX_CHARS]
        # No ``server`` field on a tool *result* -- ToolCallEvent carries the
        # attribution and the id correlates the pair, exactly as it does for the
        # results read off the wire.
        self._failures.put(
            ToolResultEvent(id=call_id, name=name, content=text, is_error=True)
        )
        return dynamic_tool_response(text, success=False)


#: Cap on a handler failure's text. It goes to the model *and* onto a
#: ``tool_result``, and a megabyte of traceback in either place is noise where a
#: reason belongs. Generous enough for a stack-free exception message.
_TOOL_ERROR_MAX_CHARS = 4000


def _invoke_handler(handler: Callable[..., Any], arguments: Mapping[str, Any]) -> Any:
    """Call a caller's handler, sync or async, from a thread with no event loop.

    :class:`~modelpass.tools.ToolDef` documents that a handler may be either, and
    that promise is not the Anthropic adapter's alone. There the SDK owns a
    running loop and a sync handler is pushed *off* it with
    ``asyncio.to_thread``; here the situation is the mirror image -- the reader
    thread is a plain thread with no loop at all -- so a sync handler is simply
    called and an async one gets a loop of its own for the length of the call.

    ``asyncio.run`` is safe precisely because this is never the main thread and
    never inside a loop: the reader thread does nothing but read, and a caller's
    own loop, if they have one, is elsewhere.
    """
    if inspect.iscoroutinefunction(handler):
        return asyncio.run(handler(dict(arguments)))
    return handler(dict(arguments))


def app_server_argv(binary: str, config_overrides: Sequence[str] = ()) -> list[str]:
    """The launch: ``<codex> app-server --listen stdio://`` plus any ``-c`` pairs.

    A function rather than a constant so the binary stays where every other
    launch in this package puts it -- resolved by the adapter, overridable with
    ``options={"codex_bin": ...}`` (D2) -- and so the newline-free property S1
    made load-bearing is pinnable by a test on one callable.

    ``config_overrides`` is the already-built ``-c key=value`` argument list.
    Both the session path (S5) and the stateless path (D23) pass
    :func:`~modelpass.adapters.openai.chat_tool_overrides` here, because a chat --
    held or one-shot -- means *the runtime's own toolbelt is off* and this
    transport has no per-thread field for that. It is empty only for a caller
    who opted back in with ``options={"native_tools": True}``.

    **This used to say "empty for every stateless run", and that was the
    defect.** Keeping S4's launch byte-for-byte through the migration was
    migration conservatism, not a decision that a chat-shaped call should carry
    a coding agent's shell -- and it left ``openai-sdk`` contradicting
    :meth:`~modelpass.Bridge.chat`'s own promise that "built-in tools are off by
    construction here", which ``anthropic-sdk`` had kept since Phase 3. D23 is
    the repair.

    **That ``-c`` reaches this subcommand at all was checked token-free on
    2026-08-31** against codex-cli 0.151.0: ``codex app-server --help`` documents
    ``-c/--config``, and ``codex app-server --listen off --strict-config -c
    <pair>`` accepts each of the six overrides in ``_CHAT_TOOL_OVERRIDES``
    (failing only at *no transport configured*, which is after the config layer)
    while rejecting ``-c features.not_a_real_flag=false`` with *unknown
    configuration field*. So the keys are live on this subcommand rather than
    accepted and ignored -- the failure mode the exec table exists to rule out.
    The *effect* is verified too, as of D23 (2026-08-31): an otherwise
    identical bare stateless call went from 15,890 to 10,269 prompt tokens with
    these six pairs in place -- 5,621 removed, 35.4% of the prefix -- driven
    through this exact argv. The earlier note here said no app-server run had
    ever been driven with the toolbelt off; that is no longer true, and the
    number above is that run.
    """
    return [binary, "app-server", *config_overrides, "--listen", "stdio://"]


AppServerSpawnFn = Callable[[list[str], dict[str, str], str], "subprocess.Popen[str] | Any"]


def _default_spawn(
    argv: list[str], env: dict[str, str], cwd: str
) -> subprocess.Popen[str]:
    """Launch the app-server child with the scrubbed plan environment, verbatim.

    ``env`` is passed straight through and never merged with ``os.environ`` --
    the property that made depending on ``openai-codex``'s client impossible
    (module docstring). Three real pipes: stdin carries our requests, stdout the
    server's lines, stderr the crash reason. Draining the last two needs two
    threads and gets them.
    """
    return subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )


class AppServerClient:
    """One ``codex app-server`` child and the JSON-RPC conversation with it.

    Not thread-safe for concurrent :meth:`request` calls from many callers in
    the sense of ordering, but safe in the sense that matters: writes are
    serialized under a lock and responses are correlated by id, so two threads
    issuing requests will each get their own answer back.

    Lifecycle is explicit -- :meth:`start` spawns and handshakes, :meth:`close`
    tears down and is idempotent -- and the object is a context manager, because
    a client whose child outlives the caller is a stray ``codex`` process.
    """

    def __init__(
        self,
        *,
        binary: str,
        env: Mapping[str, str],
        cwd: str,
        spawn: AppServerSpawnFn | None = None,
        server_request_handler: ServerRequestHandler | None = None,
        config_overrides: Sequence[str] = (),
        stderr_tail_lines: int = STDERR_TAIL_LINES,
        handshake_timeout: float = HANDSHAKE_TIMEOUT,
    ) -> None:
        self._binary = binary
        self._env = dict(env)
        self._cwd = cwd
        #: ``-c key=value`` arguments for the child's launch. Carries
        #: :func:`~modelpass.adapters.openai.chat_tool_overrides` for a chat --
        #: held (S5) or stateless (D23) -- and is empty only for a worker
        #: session or an explicit ``options={"native_tools": True}``. See
        #: :func:`app_server_argv`.
        self._config_overrides = tuple(config_overrides)
        self._spawn = spawn or _default_spawn
        self._handle_server_request = server_request_handler or decline_approvals
        self._handshake_timeout = handshake_timeout

        self._proc: Any = None
        self._started = False
        self._closed = False
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, queue.SimpleQueue[tuple[str, Any]]] = {}
        self._stderr: deque[str] = deque(maxlen=max(1, stderr_tail_lines))
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._ended = threading.Event()

        #: Server -> client notifications, in arrival order. ``None`` is the
        #: end-of-stream sentinel and is pushed exactly once, when the child's
        #: stdout reaches EOF or the client is closed, so a consumer blocked on
        #: ``get()`` wakes up instead of waiting on a process that is gone.
        self.notifications: queue.Queue[Notification | None] = queue.Queue()

    # --- lifecycle ---------------------------------------------------------------

    def __enter__(self) -> AppServerClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else ("running" if self._started else "unstarted")
        return f"<AppServerClient {self._binary}:{state}>"

    @property
    def running(self) -> bool:
        """Whether the child is up and the protocol is usable."""
        return self._started and not self._closed

    def start(self) -> None:
        """Spawn the child, start the readers, and complete the handshake.

        Idempotent by return rather than by exception: calling it twice is what
        a context manager plus an eager caller looks like, and re-spawning would
        orphan the first child.

        **Ordering is the contract.** ``initialize`` is issued and *answered*
        before the ``initialized`` notification leaves, and nothing else is
        written before either. A server that received a ``thread/start`` ahead
        of the handshake would be within its rights to reject it, and the
        failure would arrive later and somewhere else.
        """
        if self._started:
            return
        argv = app_server_argv(self._binary, self._config_overrides)
        # The same guard every launch in this package passes through, even
        # though this argv is fixed: a forbidden argument that only appears in
        # one code path is a forbidden argument that gets through.
        check_launch_args(Runtime.OPENAI_SDK, argv)
        try:
            proc = self._spawn(argv, dict(self._env), self._cwd)
        except OSError as exc:
            raise AppServerTransportClosed(
                f"the codex CLI at {self._binary!r} could not be launched as an "
                f"app-server: {exc}"
            ) from exc
        self._proc = proc
        self._started = True
        self._reader = threading.Thread(
            target=self._read_loop, name="modelpass-appserver-reader", daemon=True
        )
        self._reader.start()
        if getattr(proc, "stderr", None) is not None:
            self._stderr_reader = threading.Thread(
                target=self._drain_stderr, name="modelpass-appserver-stderr", daemon=True
            )
            self._stderr_reader.start()
        self._handshake()

    def _handshake(self) -> None:
        """``initialize`` then ``initialized``. See :meth:`start` for the ordering."""
        self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "modelpass",
                    "title": "modelpass",
                    "version": _modelpass_version(),
                },
                # Opt into the experimental surface. This is what makes the
                # dynamic-tools registration S6 needs visible at all, and the
                # vendor's own field description calls it experimental -- so it
                # is pinned with a date and re-verified per release rather than
                # assumed stable (plan of record, standing cautions).
                "capabilities": {"experimentalApi": True},
            },
            timeout=self._handshake_timeout,
        )
        self.notify("initialized")

    def close(self) -> None:
        """Terminate the child politely, then kill it. Idempotent.

        Politely first because the app-server may be holding a thread open, and
        a clean exit is how it finishes writing its rollout; then a kill,
        because a runtime that will not stop must not be able to keep modelpass
        from returning. Every pending request is failed rather than left to time
        out, and the notification stream gets its end sentinel, so nothing that
        was waiting on this client stays blocked.
        """
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            proc, self._proc = self._proc, None
        if proc is not None:
            self._shutdown(proc)
        self._fail_pending("the app-server client was closed")
        self._end_stream()

    def _shutdown(self, proc: Any) -> None:
        """Close stdin, terminate, then kill. Never raises."""
        stdin = getattr(proc, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except OSError:
                pass
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=_SHUTDOWN_GRACE)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=_SHUTDOWN_GRACE)
        except (OSError, subprocess.TimeoutExpired):
            pass

    # --- sending -----------------------------------------------------------------

    def request(
        self, method: str, params: Mapping[str, Any] | None = None, *, timeout: float | None = None
    ) -> Mapping[str, Any]:
        """Send a request and block until its response, correlated by id.

        ``timeout`` defaults to ``None`` -- wait indefinitely -- and that is the
        right default rather than a hazard: a turn legitimately takes minutes,
        and the failure this could hang on is the child *dying*, which the
        reader detects as EOF and turns into
        :class:`AppServerTransportClosed` carrying the stderr tail. A number is
        for calls that are local work and should be quick, like the handshake.

        A JSON-RPC error response raises :class:`AppServerRequestFailed` with
        the vendor's code and message intact.
        """
        if not self._started:
            raise AppServerTransportClosed(
                "the codex app-server client has not been started; call start() first"
            )
        with self._state_lock:
            if self._closed:
                raise AppServerTransportClosed(
                    f"the codex app-server client is closed; {method!r} was not sent"
                )
            self._next_id += 1
            request_id = self._next_id
            waiter: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
            self._pending[request_id] = waiter

        message: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = dict(params)
        try:
            self._write(message)
        except AppServerTransportClosed:
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise

        try:
            kind, payload = waiter.get(timeout=timeout)
        except queue.Empty:
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise AppServerTimeout(
                f"codex app-server did not answer {method!r} within {timeout}s"
            ) from None
        if kind == "result":
            return payload if isinstance(payload, Mapping) else {"result": payload}
        if kind == "error":
            error = payload if isinstance(payload, Mapping) else {}
            code = error.get("code")
            raise AppServerRequestFailed(
                method,
                code if isinstance(code, int) else 0,
                str(error.get("message", "")),
                error.get("data"),
            )
        raise AppServerTransportClosed(self._closed_reason(str(payload), method))

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        """Send a notification: no id, no response, no waiting.

        ``params`` is omitted from the message entirely when it is ``None``,
        because ``initialized`` is specified with ``method`` alone
        (``ClientNotification.json``) and sending an empty object where the
        schema has no field is a difference nobody needs to debug later.
        """
        message: dict[str, Any] = {"method": method}
        if params is not None:
            message["params"] = dict(params)
        self._write(message)

    def _write(self, message: Mapping[str, Any]) -> None:
        """One JSON object, one line, under the write lock.

        The lock is what makes two threads' messages interleave at message
        boundaries rather than mid-line. A dead pipe becomes
        :class:`AppServerTransportClosed` with whatever the child said on its
        way out.
        """
        proc = self._proc
        stdin = getattr(proc, "stdin", None) if proc is not None else None
        if stdin is None:
            raise AppServerTransportClosed(
                self._closed_reason("the app-server child has no stdin pipe", None)
            )
        line = json.dumps(dict(message), default=str) + "\n"
        with self._write_lock:
            try:
                stdin.write(line)
                flush = getattr(stdin, "flush", None)
                if callable(flush):
                    flush()
            except (OSError, ValueError) as exc:
                raise AppServerTransportClosed(
                    self._closed_reason(f"writing to the app-server failed: {exc}", None)
                ) from exc

    # --- receiving ---------------------------------------------------------------

    def next_notification(self, timeout: float | None = None) -> Notification | None:
        """The next server notification, or ``None`` once the stream has ended.

        Raises ``queue.Empty`` on ``timeout``, which is a different statement
        from ``None``: one says *nothing yet*, the other says *nothing ever
        again*. Collapsing them would let a slow turn read as a finished one.
        """
        return self.notifications.get(timeout=timeout)

    def stderr_tail(self) -> str:
        """The last of the child's stderr, as one string. Possibly empty."""
        return "".join(self._stderr).strip()

    def _read_loop(self) -> None:
        """The single reader: classify each line and route it. Never raises out.

        Ends on EOF -- the child exited -- which is the condition that turns
        every blocked :meth:`request` into a reported failure instead of a hang.
        """
        proc = self._proc
        stdout = getattr(proc, "stdout", None) if proc is not None else None
        try:
            for raw in stdout or ():
                line = raw.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    self._unroutable(line)
                    continue
                if isinstance(message, Mapping):
                    self._dispatch(message)
                else:
                    self._unroutable(line)
        except (OSError, ValueError):
            # The pipe died under us. Same outcome as EOF, and the finally
            # below reports it the same way.
            pass
        finally:
            self._fail_pending("the codex app-server exited")
            self._end_stream()

    def _end_stream(self) -> None:
        """Push the end-of-stream sentinel once, whichever half gets there first.

        Both the reader's EOF and an explicit :meth:`close` end the stream, and
        a consumer that read two sentinels would see the second as a second
        ending. One event, one ``None``.
        """
        if not self._ended.is_set():
            self._ended.set()
            self.notifications.put(None)

    def _dispatch(self, message: Mapping[str, Any]) -> None:
        """Route one parsed line by which of ``method`` and ``id`` it carries."""
        method = message.get("method")
        request_id = message.get("id")
        has_id = request_id is not None

        if isinstance(method, str) and has_id:
            self._answer_server_request(request_id, method, message.get("params"))
            return
        if isinstance(method, str):
            params = message.get("params")
            self.notifications.put(
                Notification(
                    method=method, params=dict(params) if isinstance(params, Mapping) else {}
                )
            )
            return
        if has_id:
            self._deliver_response(request_id, message)
            return
        self._unroutable(json.dumps(dict(message), default=str))

    def _answer_server_request(
        self, request_id: Any, method: str, params: Any
    ) -> None:
        """Run the caller's handler and write the result back under the same id.

        A handler that raises does not take the reader thread down and does not
        leave the server waiting: the request is answered ``{}`` -- the same
        answer an unknown method gets -- and the failure is surfaced as an
        unroutable line rather than swallowed. A turn stalled forever on an
        unanswered round trip would be a worse report than a recorded handler
        bug.
        """
        payload = dict(params) if isinstance(params, Mapping) else {}
        try:
            result = self._handle_server_request(method, payload)
        # Broad on purpose: this is the caller's handler running on our
        # reader thread, and its bugs must not end the transport.
        except Exception as exc:
            self._unroutable(f"server_request_handler failed for {method!r}: {exc!r}")
            result = {}
        try:
            self._write({"id": request_id, "result": dict(result or {})})
        except AppServerTransportClosed:
            # The child is gone; the reader's own EOF handling reports it.
            pass

    def _deliver_response(self, request_id: Any, message: Mapping[str, Any]) -> None:
        """Hand a response to whoever is blocked on that id.

        The id is normalized before the lookup because ``RequestId`` is
        ``string | integer`` on the wire (``JSONRPCRequest.json``): modelpass
        always sends an integer, but a server that echoed ``"3"`` would
        otherwise strand a caller who is waiting on ``3``.
        """
        with self._state_lock:
            waiter = self._pending.pop(_request_key(request_id), None)
        if waiter is None:
            # A response to an id we are not waiting on -- a duplicate, or one
            # whose caller timed out. Not dropped silently.
            self._unroutable(json.dumps(dict(message), default=str))
            return
        if "error" in message:
            waiter.put(("error", message.get("error")))
        else:
            waiter.put(("result", message.get("result")))

    def _drain_stderr(self) -> None:
        """Feed the bounded tail. Its own thread, because two pipes need two."""
        proc = self._proc
        stream = getattr(proc, "stderr", None) if proc is not None else None
        try:
            for line in stream or ():
                self._stderr.append(line)
        except (OSError, ValueError):
            pass

    def _unroutable(self, line: str) -> None:
        """Record a line the protocol could not place. Never dropped (rule 3).

        It goes to the stderr tail because that is the buffer a failure report
        already reads, and a malformed line is exactly the evidence someone
        debugging a protocol drift needs.
        """
        self._stderr.append(f"[unroutable] {line[:2000]}\n")

    def _fail_pending(self, reason: str) -> None:
        """Wake every blocked request with a transport-closed outcome."""
        with self._state_lock:
            pending, self._pending = self._pending, {}
            self._closed = True
        for waiter in pending.values():
            waiter.put(("closed", reason))

    def _closed_reason(self, reason: str, method: str | None) -> str:
        """A transport failure, plus what the child said on its way out."""
        where = f" while sending {method!r}" if method else ""
        tail = self.stderr_tail()
        detail = " ".join(tail.split())[:500]
        base = f"{reason}{where}"
        return f"{base}: {detail}" if detail else base


def _request_key(request_id: Any) -> int:
    """An id from the wire as the integer key ``_pending`` is keyed by.

    ``-1`` for anything unusable, which no outstanding request can be, so an
    unrecognizable id lands on the unroutable path rather than on some other
    caller's waiter.
    """
    if isinstance(request_id, bool):
        return -1
    if isinstance(request_id, int):
        return request_id
    if isinstance(request_id, str):
        try:
            return int(request_id)
        except ValueError:
            return -1
    return -1


def _modelpass_version() -> str:
    """modelpass's version, imported lazily to keep this module cycle-free.

    ``modelpass/__init__`` imports the bridge, which imports the adapters; once
    S4 has ``adapters/openai`` import this module, a top-level
    ``from .. import __version__`` here would close that loop. A function-local
    import costs one dict lookup per handshake and cannot.
    """
    from .. import __version__

    return __version__


# --- event mapping (S3) -----------------------------------------------------------


#: Item ``type`` values that report a tool the runtime executed, and so map onto
#: the D12 tool events. All three emit ``item/started`` *and* ``item/completed``,
#: which is why started -> ``tool_call`` and completed -> ``tool_result`` rather
#: than synthesizing a pair from one notification: the latter would double-report
#: every call that reached completion.
TOOL_ITEM_TYPES = frozenset({"commandExecution", "mcpToolCall", "dynamicToolCall"})

#: ``server`` for Codex's own built-in shell. The exec transport's
#: ``_BUILTIN_SERVER``, repeated deliberately: same runtime, same attribution,
#: so a consumer filtering on it survives the transport flip.
BUILTIN_SERVER = "codex"

#: The normalized ``name`` for a shell command. The **exec transport's**
#: spelling, not the app-server's ``commandExecution`` -- see the module
#: docstring for why one runtime identity means one tool name.
COMMAND_TOOL_NAME = "command_execution"

#: Item ``status`` values that mean the tool did not do what was asked.
#: ``failed`` is shared by all three item types; ``declined`` exists only on
#: ``CommandExecutionStatus`` (schema-sourced) and is the sandbox refusing a
#: command -- the value that was captured live on the exec transport on
#: 2026-08-30 and was, until then, reported as a success whenever the exit code
#: happened to be null.
ERROR_ITEM_STATUSES = frozenset({"failed", "declined"})

#: Turn statuses, mapped to how modelpass reports the run ending. ``inProgress``
#: is deliberately absent: it is a legal ``TurnStatus`` but not a legal way for
#: a ``turn/completed`` to arrive, so it falls through to the unknown-status
#: branch and is named rather than read as a success.
_TURN_STATUS = {
    "completed": TerminalStatus.OK,
    "failed": TerminalStatus.ERROR,
    "interrupted": TerminalStatus.CANCELLED,
}

#: The vendor's own typed code for "the plan's allowance is gone". A real
#: improvement on the exec transport, where the same condition had to be
#: detected by string-matching markers extracted from the binary: here
#: ``CodexErrorInfo`` is an enum and ``usageLimitExceeded`` is a member of it
#: (``TurnCompletedNotification.json``, schema-sourced 2026-08-31).
#:
#: Narrow on purpose, exactly as the exec transport's detection is. Neighbouring
#: members -- ``rateLimitExceeded``, ``sessionBudgetExceeded`` -- are *not* here:
#: a misread error is a worse-labelled failure, while a misread failure as a
#: spent allowance is a silent one that can trigger a configured failover onto
#: metered billing for no reason.
_QUOTA_ERROR_CODES = frozenset({"usageLimitExceeded"})


def turn_error_detail(error: Any) -> str:
    """A human-readable reason from a ``TurnError``. Defensive by design.

    ``message`` is the only required field (``TurnError`` in
    ``TurnCompletedNotification.json`` / ``ErrorNotification.json``);
    ``codexErrorInfo`` is either a bare enum string or a single-key object, and
    ``additionalDetails`` is free text. Every one of those shapes is
    schema-sourced and none has been driven, so anything unexpected keeps its
    raw text rather than being parsed into something prettier and wrong.
    """
    if isinstance(error, str):
        return error
    if not isinstance(error, Mapping):
        return json.dumps(error, default=str)[:500] if error is not None else ""
    parts: list[str] = []
    code = _error_code(error)
    if code:
        parts.append(code)
    message = error.get("message")
    parts.append(
        message
        if isinstance(message, str) and message
        else json.dumps(dict(error), default=str)[:500]
    )
    extra = error.get("additionalDetails")
    if isinstance(extra, str) and extra.strip():
        parts.append(extra.strip())
    return ": ".join(p for p in parts if p)


def _error_code(error: Mapping[str, Any]) -> str:
    """``codexErrorInfo`` as a name, whichever of its two spellings arrived.

    The enum has bare-string members (``"usageLimitExceeded"``) and object
    members keyed by the variant name (``{"httpConnectionFailed": {...}}``).
    Reading only the first would drop the code from every error that carries an
    HTTP status, which are the ones most worth naming.
    """
    info = error.get("codexErrorInfo")
    if isinstance(info, str):
        return info
    if isinstance(info, Mapping) and len(info) == 1:
        return next(iter(info))
    return ""


def _is_quota_exhausted(error: Any) -> bool:
    """Whether a turn failure is "the plan ran out" rather than a fault (D4)."""
    return isinstance(error, Mapping) and _error_code(error) in _QUOTA_ERROR_CODES


def token_usage_from_breakdown(breakdown: Any) -> TokenUsage:
    """One ``TokenUsageBreakdown`` as a :class:`~modelpass.types.TokenUsage`.

    Field names are camelCase on the wire and mapped one for one:
    ``inputTokens``, ``outputTokens``, ``cachedInputTokens`` (all required) and
    ``cacheWriteInputTokens`` (optional, default 0). ``totalTokens`` has no home
    on ``TokenUsage`` and is not invented into one -- it rides along in the
    accompanying ``vendor_event`` with the rest of the payload.

    **``reasoningOutputTokens`` now has one** (2026-09-22), and its relation to
    ``outputTokens`` was settled from real traffic rather than from the schema,
    which describes neither: **66,406** ``token_count`` records across a machine's
    ``~/.codex/sessions`` and ``archived_sessions``, 56,749 of them with a
    non-zero count, contain **zero** rows where reasoning exceeded output, and
    ``input_tokens + output_tokens == total_tokens`` holds in all but 151 -- and
    those 151 are a degenerate shape with every component zero and a non-zero
    total. So the vendor neither adds it in nor counts it apart: it is a subset
    of output, exactly as :attr:`~modelpass.types.TokenUsage
    .reasoning_output_tokens` requires.

    **``inputTokens`` is inclusive of ``cachedInputTokens``** -- verified live on
    2026-08-31, on this transport and on ``codex exec`` alike, which is why the
    subtraction below is not the arithmetic bug it looks like. The two vendors
    modelpass drives genuinely disagree about this field:

    * **Codex nests it.** The ``last`` breakdown from the live capture reads
      ``{inputTokens 13123, cachedInputTokens 12032, outputTokens 21,
      totalTokens 13144}``: the vendor's *own* total is ``input + output``, so
      the cache read is a **subset** of the input. ``codex exec``'s
      ``turn.completed`` nests the same way (``{input_tokens 14228,
      cached_input_tokens 9984, cache_write_input_tokens 0, output_tokens 5}``).
    * **Anthropic counts them side by side.** A live row there reads
      ``input_tokens 2`` beside ``cache_read_input_tokens 1901``, a shape a
      nested field cannot produce.

    :class:`~modelpass.types.TokenUsage` keeps the parallel convention, because it
    is the reading that can express both and the one Anthropic satisfies
    natively; the correction therefore lives in the Codex mappers rather than in
    a ``total_tokens`` that would mean something different per runtime. Before
    2026-08-31 it was missing on both transports and the cache read was counted
    twice: every Codex run in ``~/.modelpass/runs.jsonl`` was inflated by roughly
    65% (22,655 reported against 13,695 spent), which corrupted the run log
    *and* fed ``stop_at_tokens``, so guards fired early.
    :func:`modelpass.adapters.openai.map_codex_event` carries the same correction
    for the exec transport; the two move together.
    """
    if not isinstance(breakdown, Mapping):
        return TokenUsage()
    wire_input = _as_int(breakdown.get("inputTokens"))
    cached = _as_int(breakdown.get("cachedInputTokens"))
    return TokenUsage(
        # Clamped at zero rather than trusted: if a release ever reports
        # cachedInputTokens above inputTokens, a negative count here would reach
        # a run record and *subtract* from a guard's running total, which is a
        # worse failure than the over-count this line exists to fix.
        input_tokens=max(0, wire_input - cached),
        output_tokens=_as_int(breakdown.get("outputTokens")),
        reasoning_output_tokens=_reported_int(breakdown.get("reasoningOutputTokens")),
        cached_input_tokens=cached,
        cache_write_tokens=_as_int(breakdown.get("cacheWriteInputTokens")),
    )


def _reported_int(value: Any) -> int | None:
    """A wire integer, or ``None`` when the field was absent.

    Not :func:`_as_int`: a reasoning count of ``0`` is a report that the model
    did not think, and an absent field is the server not saying. Collapsing
    them would make a transport that never sends the field indistinguishable
    from a turn that spent no reasoning tokens.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_int(value: Any) -> int:
    """A wire integer, or 0. A token count is never the reason a run fails."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _vendor(method: str, params: Mapping[str, Any]) -> VendorEvent:
    """Rule 3's catch-all: the vendor said it, modelpass does not normalize it."""
    return VendorEvent(runtime=Runtime.OPENAI_SDK, name=method, data=dict(params))


def _item_of(params: Mapping[str, Any]) -> Mapping[str, Any]:
    item = params.get("item")
    return item if isinstance(item, Mapping) else {}


def _item_id(item: Mapping[str, Any]) -> str:
    value = item.get("id")
    return value if isinstance(value, str) else ""


def map_appserver_event(
    method: str,
    params: Mapping[str, Any],
    *,
    streamed_message_items: frozenset[str] | set[str] = frozenset(),
    reported_tool_failures: frozenset[str] | set[str] = frozenset(),
) -> tuple[AgentEvent, ...]:
    """Map one app-server notification to normalized events. Pure and total.

    Nothing is dropped: every notification this does not recognize -- and every
    half of one it recognizes but cannot normalize -- leaves as a
    ``vendor_event`` (adapter contract, rule 3). One notification maps to *two*
    events, which is not a special case but the same rule applied twice: a usage
    update carries both a number modelpass normalizes and a payload it does not.

    ``streamed_message_items`` is the set of ``agentMessage`` item ids that have
    already streamed a delta, and ``reported_tool_failures`` the set of dynamic
    tool ``callId``s whose failure modelpass has already reported (see
    :class:`CallerToolDispatcher`). Both are turn state, and both are passed in
    rather than held: it keeps the function pure and each dedup rule testable in
    isolation, and :class:`AppServerTurn` is what accumulates them in practice.
    """
    if method == "item/agentMessage/delta":
        delta = params.get("delta")
        if isinstance(delta, str):
            return (TextDeltaEvent(text=delta),)
        return (_vendor(method, params),)

    if method in ("item/reasoning/textDelta", "item/reasoning/summaryTextDelta"):
        # Both are reasoning: one the model's own text, one the summary stream.
        # Kept as a single normalized kind because D7 refuses to invent a
        # structure for thinking that the providers do not agree on -- and a
        # caller who needs the distinction still has the method name, since the
        # payload that carried it is the one modelpass did not normalize away.
        delta = params.get("delta")
        if isinstance(delta, str):
            return (ThinkingEvent(text=delta),)
        return (_vendor(method, params),)

    if method == "item/started":
        item = _item_of(params)
        if item.get("type") in TOOL_ITEM_TYPES:
            event = _tool_call_from_item(item)
            if event is not None:
                return (event,)
        return (_vendor(_item_method(method, item), params),)

    if method == "item/completed":
        return _map_item_completed(
            method, params, streamed_message_items, reported_tool_failures
        )

    if method == "thread/tokenUsage/updated":
        return _map_token_usage(method, params)

    return (_vendor(method, params),)


def _item_method(method: str, item: Mapping[str, Any]) -> str:
    """``item/completed:mcpToolCall`` -- the method plus which item it carried.

    A caller filtering vendor events should not have to open ``data`` to find
    out which of nineteen item types arrived, which is the same reason
    ``map_codex_event`` names its passthroughs ``item.completed:reasoning``.
    """
    item_type = item.get("type")
    return f"{method}:{item_type}" if isinstance(item_type, str) and item_type else method


def _map_item_completed(
    method: str,
    params: Mapping[str, Any],
    streamed_message_items: frozenset[str] | set[str],
    reported_tool_failures: frozenset[str] | set[str] = frozenset(),
) -> tuple[AgentEvent, ...]:
    """``item/completed``: the two dedup rules, the tool results, and the rest.

    See the module docstring for why a completed ``agentMessage`` that already
    streamed does not become a second ``text_delta``, and
    :class:`CallerToolDispatcher` for why a ``dynamicToolCall`` whose failure
    modelpass already reported does not become a second ``tool_result``.
    """
    item = _item_of(params)
    item_type = item.get("type")

    if item_type == "dynamicToolCall" and _item_id(item) in reported_tool_failures:
        # modelpass ran the handler, watched it fail, and has already emitted the
        # tool_result carrying the caller's own error text. The server's item
        # still reaches the caller whole -- status, durationMs, whatever the
        # vendor made of the success:false answer -- as a vendor_event.
        return (_vendor(_item_method(method, item), params),)

    if item_type == "agentMessage":
        text = item.get("text")
        if isinstance(text, str) and _item_id(item) not in streamed_message_items:
            return (TextDeltaEvent(text=text),)
        # Streamed already, or no readable text: the item still reaches the
        # caller whole, carrying the authoritative full text plus the fields
        # (phase, delivery, memoryCitation) the normalized event has no room
        # for. Nothing is dropped; nothing is counted twice.
        return (_vendor(_item_method(method, item), params),)

    if item_type in TOOL_ITEM_TYPES:
        event = _tool_result_from_item(item)
        if event is not None:
            return (event,)

    return (_vendor(_item_method(method, item), params),)


def _tool_call_from_item(item: Mapping[str, Any]) -> AgentEvent | None:
    """A ``tool_call`` from an ``item/started`` item, or ``None`` on a bad shape.

    ``None`` is the fallback to ``vendor_event``, and it is the point: these
    field names are schema-sourced (2026-08-31) rather than driven, so an item
    that does not look the way the schema describes reaches the caller intact
    instead of arriving as a half-populated tool call nobody can tell from a
    real one.
    """
    item_type = item.get("type")
    item_id = _item_id(item)

    if item_type == "commandExecution":
        command = item.get("command")
        if not isinstance(command, str):
            return None
        return ToolCallEvent(
            name=COMMAND_TOOL_NAME,
            arguments={"command": command},
            id=item_id,
            server=BUILTIN_SERVER,
        )

    tool = item.get("tool")
    if not isinstance(tool, str) or not tool:
        return None
    arguments = item.get("arguments")
    arguments = dict(arguments) if isinstance(arguments, Mapping) else {}

    if item_type == "mcpToolCall":
        server = item.get("server")
        return ToolCallEvent(
            name=tool,
            arguments=arguments,
            id=item_id,
            server=server if isinstance(server, str) else "",
        )

    # dynamicToolCall: a tool the caller registered, executed by a function in
    # this process. That is what CALLER_TOOL_SERVER means everywhere else in
    # modelpass, so it is what it means here. The item's own ``namespace`` is kept
    # in the arguments rather than promoted to ``server``: it is the vendor's
    # grouping for the tool, not a place the tool runs.
    namespace = item.get("namespace")
    if isinstance(namespace, str) and namespace:
        arguments = {**arguments, "__namespace": namespace}
    return ToolCallEvent(
        name=tool, arguments=arguments, id=item_id, server=CALLER_TOOL_SERVER
    )


def _tool_result_from_item(item: Mapping[str, Any]) -> AgentEvent | None:
    """A ``tool_result`` from an ``item/completed`` item, or ``None``."""
    item_type = item.get("type")
    item_id = _item_id(item)
    status = item.get("status")

    if item_type == "commandExecution":
        if "aggregatedOutput" not in item and "exitCode" not in item:
            return None
        output = item.get("aggregatedOutput")
        exit_code = item.get("exitCode")
        return ToolResultEvent(
            id=item_id,
            name=COMMAND_TOOL_NAME,
            content=output if isinstance(output, str) else "",
            # The status is the better signal and the only one that carries
            # ``declined``; a null exit code is not evidence either way, which
            # is what made a declined command read as a success on the exec
            # transport until 2026-08-30.
            is_error=(
                status in ERROR_ITEM_STATUSES
                or bool(isinstance(exit_code, int) and exit_code != 0)
            ),
        )

    tool = item.get("tool")
    if not isinstance(tool, str) or not tool:
        return None

    if item_type == "mcpToolCall":
        error = item.get("error")
        is_error = status in ERROR_ITEM_STATUSES or bool(error)
        payload = error if is_error and error is not None else item.get("result")
        return ToolResultEvent(
            id=item_id, name=tool, content=flatten_mcp_result(payload), is_error=is_error
        )

    # dynamicToolCall. ``success`` is a nullable boolean alongside ``status``,
    # so the two are read together: an explicit ``success: false`` is a failure
    # even where the status says the call completed, because the caller's own
    # function is what reported it.
    success = item.get("success")
    return ToolResultEvent(
        id=item_id,
        name=tool,
        content=_flatten_content_items(item.get("contentItems")),
        is_error=status in ERROR_ITEM_STATUSES or success is False,
    )


def _flatten_content_items(items: Any) -> str:
    """Flatten a dynamic tool's ``contentItems`` to text.

    ``DynamicToolCallOutputContentItem`` is one of ``inputText`` (``text``),
    ``inputImage`` (``imageUrl``) or ``inputAudio`` (``audioUrl``), and only the
    first is text. The other two become a marker rather than a URL: a
    ``tool_result``'s ``content`` is documented as flattened text, and pasting a
    data URI of an image into it would be a large, useless string where a caller
    expects an answer. The whole item is still on the ``vendor_event`` path for
    anyone who needs the URL.
    """
    if not isinstance(items, (list, tuple)):
        return ""
    parts: list[str] = []
    for block in items:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            text = block.get("text")
            parts.append(text if isinstance(text, str) else f"[{block.get('type')}]")
    return "\n".join(p for p in parts if p)


def _map_token_usage(method: str, params: Mapping[str, Any]) -> tuple[AgentEvent, ...]:
    """``thread/tokenUsage/updated`` -> one interim usage event, plus the payload.

    ``last`` and never ``total``: see trap one in the module docstring. The
    ``vendor_event`` beside it is not redundancy -- it is where ``total``,
    ``modelContextWindow``, ``reasoningOutputTokens`` and ``totalTokens`` reach a
    caller, none of which :class:`~modelpass.types.TokenUsage` has a field for and
    none of which should be silently dropped to make the normalized half tidy.
    """
    usage = params.get("tokenUsage")
    if not isinstance(usage, Mapping) or not isinstance(usage.get("last"), Mapping):
        return (_vendor(method, params),)
    return (
        UsageEvent(usage=token_usage_from_breakdown(usage["last"])),
        _vendor(method, params),
    )


class AppServerTurn:
    """What one turn on the app-server transport turned out to be.

    The counterpart of ``_TurnOutcome`` on the exec transport, and it exists for
    the same reason: a generator can either yield events or return a summary,
    and a turn needs both. :meth:`observe` is the single reading of the
    notification stream that the stateless call and a session turn will share in
    S4 and S5 -- two copies of "which notification means the run failed" would
    drift, and the half that drifts is the half that decides how a run is
    reported.

    Not thread-safe, and does not need to be: one turn is driven by one caller
    draining one notification queue.
    """

    __slots__ = (
        "agent_messages",
        "model_context_window",
        "reason",
        "reported_tool_failures",
        "status",
        "streamed_message_items",
        "thread_id",
        "thread_total_usage",
        "turn_id",
        "usage",
    )

    def __init__(self) -> None:
        self.status: TerminalStatus = TerminalStatus.OK
        self.reason: str | None = None
        #: This turn's usage: the sum of every update's ``last`` breakdown. See
        #: trap one in the module docstring for why it is not ``total``.
        self.usage: TokenUsage = TokenUsage()
        #: The thread's running total as of the last update seen, which on a
        #: resumed thread includes turns this object never watched. Kept because
        #: it is the vendor's own figure and re-deriving it downstream would be
        #: guesswork; never used as this turn's usage.
        self.thread_total_usage: TokenUsage = TokenUsage()
        self.model_context_window: int | None = None
        #: **Every** completed ``agentMessage`` in the turn, in arrival order --
        #: a list rather than a slot, because a live exec run on 2026-08-30
        #: emitted two in one turn and the second would silently overwrite the
        #: first. This is also where the structured answer is read from.
        self.agent_messages: list[str] = []
        #: Item ids that have streamed at least one delta. The dedup rule's
        #: whole state.
        self.streamed_message_items: set[str] = set()
        #: Dynamic tool ``callId``s whose failure modelpass reported itself, from
        #: :meth:`CallerToolDispatcher.drain_failures`. The second dedup rule's
        #: whole state, filled by the run's drain rather than by
        #: :meth:`observe`: the failure happens on the reader thread, before the
        #: server's completed item exists.
        self.reported_tool_failures: set[str] = set()
        self.thread_id: str | None = None
        self.turn_id: str | None = None

    def __repr__(self) -> str:
        return f"<AppServerTurn {self.status.value} turn={self.turn_id}>"

    @property
    def final_answer(self) -> str | None:
        """The structured answer: the **last** completed message, not the join.

        Last-wins is right here for the reason ``_TurnOutcome.final_answer``
        records: with an output schema the answer is a JSON string in the
        ordinary place a plain answer would be, and concatenating two of them
        produces something that parses as neither.
        """
        return self.agent_messages[-1] if self.agent_messages else None

    def observe(self, notification: Notification) -> tuple[AgentEvent, ...]:
        """Fold one notification in and return the events it produced.

        Order matters in one place and is easy to get wrong: a delta is recorded
        **before** the mapping runs, so a completed message whose deltas arrived
        earlier in the same turn is deduped, while the mapping of the delta
        itself is unaffected.
        """
        method, params = notification.method, notification.params
        self._note_ids(params)

        if method == "item/agentMessage/delta":
            item_id = params.get("itemId")
            if isinstance(item_id, str):
                self.streamed_message_items.add(item_id)
        elif method == "item/completed":
            item = _item_of(params)
            if item.get("type") == "agentMessage":
                text = item.get("text")
                if isinstance(text, str):
                    self.agent_messages.append(text)
        elif method == "thread/tokenUsage/updated":
            self._note_usage(params)
        elif method == "turn/completed":
            self._note_turn_completed(params)
        elif method == "error":
            self._note_error(params)

        events = map_appserver_event(
            method,
            params,
            streamed_message_items=frozenset(self.streamed_message_items),
            reported_tool_failures=frozenset(self.reported_tool_failures),
        )
        for event in events:
            if isinstance(event, UsageEvent):
                self.usage = self.usage + event.usage
        return events

    def _note_ids(self, params: Mapping[str, Any]) -> None:
        """Pick up ``threadId`` / ``turnId`` wherever they appear.

        Nearly every notification carries both, so there is no one notification
        to read them from -- and reading them everywhere means a turn knows its
        own ids even when the stream starts mid-turn, which is what a resumed
        thread looks like.
        """
        thread_id = params.get("threadId")
        if isinstance(thread_id, str) and thread_id.strip():
            self.thread_id = thread_id.strip()
        turn_id = params.get("turnId")
        if isinstance(turn_id, str) and turn_id.strip():
            self.turn_id = turn_id.strip()

    def _note_usage(self, params: Mapping[str, Any]) -> None:
        """Record the thread total and the context window.

        ``last`` is *not* folded here: it is folded in :meth:`observe`, through
        the ``UsageEvent`` the mapping produced, so there is exactly one place
        the turn's number is added up and no way for the two to disagree.
        """
        usage = params.get("tokenUsage")
        if not isinstance(usage, Mapping):
            return
        if isinstance(usage.get("total"), Mapping):
            self.thread_total_usage = token_usage_from_breakdown(usage["total"])
        window = usage.get("modelContextWindow")
        if isinstance(window, int) and not isinstance(window, bool):
            self.model_context_window = window

    def _note_turn_completed(self, params: Mapping[str, Any]) -> None:
        """Read the turn's own verdict off ``turn.status`` / ``turn.error``.

        There is no ``turn/failed`` notification on this protocol -- a failed
        turn is a ``turn/completed`` whose status says so (module docstring).
        An unrecognized status is reported as an error naming the raw value:
        modelpass did not observe a success, and claiming one because a future
        status was not in a lookup table is precisely the silent misreport this
        library exists to prevent.
        """
        turn = params.get("turn")
        turn = turn if isinstance(turn, Mapping) else {}
        turn_id = turn.get("id")
        if isinstance(turn_id, str) and turn_id.strip():
            self.turn_id = turn_id.strip()
        status = turn.get("status")
        error = turn.get("error")
        mapped = _TURN_STATUS.get(status) if isinstance(status, str) else None
        if mapped is None:
            self.status = TerminalStatus.ERROR
            self.reason = (
                f"codex reported turn status {status!r}, which modelpass does not "
                "recognize as an outcome"
            )
            return
        if mapped is TerminalStatus.OK:
            return
        detail = turn_error_detail(error)
        if mapped is TerminalStatus.CANCELLED:
            self.status = TerminalStatus.CANCELLED
            self.reason = detail or "the turn was interrupted"
            return
        if _is_quota_exhausted(error):
            # A spent allowance is a clean stop, not a failure: nothing went
            # wrong, the plan ran out (D4). It is also the signal a configured
            # failover keys off, which is why the detection errs narrow.
            self.status = TerminalStatus.QUOTA_EXHAUSTED
            self.reason = f"ChatGPT plan allowance exhausted: {detail}"
            return
        self.status = TerminalStatus.ERROR
        self.reason = detail or "codex reported the turn failed"

    def _note_error(self, params: Mapping[str, Any]) -> None:
        """An ``error`` notification, which is a turn failure only sometimes.

        ``willRetry`` is a required field on this notification, and a retryable
        error is one the runtime intends to recover from. Reporting the run
        failed because of an error it went on to retry successfully would be a
        worse lie than reporting nothing, so only ``willRetry: false`` sets the
        outcome -- and it does not overwrite a verdict ``turn/completed`` has
        already given, because the turn's own status outranks a notification
        that preceded it.
        """
        if params.get("willRetry") is True:
            return
        if self.status is not TerminalStatus.OK:
            return
        error = params.get("error")
        detail = turn_error_detail(error)
        if _is_quota_exhausted(error):
            self.status = TerminalStatus.QUOTA_EXHAUSTED
            self.reason = f"ChatGPT plan allowance exhausted: {detail}"
            return
        self.status = TerminalStatus.ERROR
        self.reason = detail or "codex reported an error"


# --- sessions on native threads (S5) ----------------------------------------------
#
# Where the shapes below come from, and it is a better source than S2-S4 had:
# **the shipped binary generates the protocol's own JSON Schema.** Token-free, no
# model call, no account touched::
#
#     codex app-server generate-json-schema --out <dir> --experimental
#
# Run on 2026-08-31 against codex-cli 0.151.0, it writes 366 files under
# ``<dir>/v2``, one per params/response/notification type -- including
# ``ThreadListParams``, ``ThreadReadParams``, ``ThreadItemsListParams``,
# ``ThreadResumeParams``, ``TurnInterruptParams`` and the full ``ThreadItem``
# variant list. That is first-party and complete where reading strings out of
# ``codex.exe`` was first-party and partial, so every "schema-sourced" note in
# this section names it. It is still not *driven*: a field that exists is not a
# field somebody has watched work, and the two are marked apart throughout.
#
# Three facts from that bundle that changed decisions here:
#
# * ``thread/list`` takes ``limit``, **not** ``pageSize`` -- which is why the
#   2026-08-31 capture asked for three threads and got twenty-five. It also
#   takes ``cwd``, ``sortKey`` and ``sortDirection``, and the last defaults to
#   descending, so the listing is newest-first without being asked.
# * ``thread/read``'s ``includeTurns`` is documented as **deprecated for
#   paginated threads** -- which every thread in the capture is
#   (``historyMode: "paginated"``) -- and the vendor's own advice is to page
#   ``thread/turns/list`` / ``thread/items/list`` instead. That is why
#   :func:`thread_items_list_params` exists and ``thread/read`` is used for
#   metadata only.
# * ``ThreadResumeParams`` has **no ``dynamicTools`` field at all**. A resumed
#   thread's caller-tool registration is restored from its rollout and cannot be
#   re-declared, which is the other half of why ``thread/start`` omits the key
#   rather than sending ``[]``.


#: Enumerate a connection's threads. Driven live 2026-08-31 (the capture's
#: ``probe:thread/list``): ``{"data": [Thread, ...], "nextCursor",
#: "backwardsCursor"}``.
THREAD_LIST_METHOD = "thread/list"

#: Pick a stored thread back up. **Not driven from here** -- but not unheard of
#: either: ``codex exec resume`` on a dead id answers *"thread/resume:
#: thread/resume failed: no rollout found for thread id <id> (code -32600)"*
#: (captured live 2026-08-30), which is this method's own error surfacing
#: through the exec CLI. So the name, the failure text and the code are known;
#: what has never been watched is a success.
THREAD_RESUME_METHOD = "thread/resume"

#: Read a thread's metadata. Driven live 2026-08-31 with
#: ``{"threadId", "includeTurns": false}`` -> ``{"thread": {...}}``.
THREAD_READ_METHOD = "thread/read"

#: Page a thread's items in conversation order. Schema-sourced, undriven.
THREAD_ITEMS_LIST_METHOD = "thread/items/list"

#: Ask the server to stop the turn in flight. Schema-sourced
#: (``TurnInterruptParams``: ``threadId`` and ``turnId``, both **required**;
#: ``TurnInterruptResponse`` is an empty object) and **driven live 2026-08-31**:
#: the call returned ``{}`` and the turn ended with ``status: "interrupted"``
#: (``tests/fixtures/appserver/live-capability-evidence-2026-08-31.jsonl``). What
#: that drive did *not* establish is whether the thread takes another turn
#: afterwards, which is the part a session depends on -- see
#: ``CodexAppServerSession._abandon``.
TURN_INTERRUPT_METHOD = "turn/interrupt"

#: How many threads one :func:`thread_list_params` page asks for. A ceiling
#: rather than a promise: ``limit`` is documented as optional with "a reasonable
#: server-side value" behind it, and modelpass reads exactly one page (see
#: :func:`thread_list_params`).
THREAD_LIST_LIMIT = 200

#: How many items one :func:`thread_items_list_params` page asks for. History is
#: paged to exhaustion, so this is a round-trip size and not a cap on what a
#: caller can read back.
THREAD_ITEMS_LIMIT = 200


def thread_list_params(limit: int = THREAD_LIST_LIMIT) -> dict[str, Any]:
    """Params for :data:`THREAD_LIST_METHOD`.

    **``limit``, not ``pageSize``** (``ThreadListParams``, schema-sourced
    2026-08-31). The distinction is not pedantry: the live probe sent
    ``{"pageSize": 3}`` and twenty-five threads came back, so an unknown key here
    is accepted and ignored rather than refused -- exactly the silent-override
    failure this package warns consumers about on ``codex -c``.

    **The ``cwd`` filter is deliberately not sent.** ``ThreadListParams`` has one,
    and using it would look like the obvious way to honour
    ``SessionRequest.project_folder``. It would be wrong here:
    :meth:`~modelpass.bridge.Bridge.list_sessions` defaults that argument to the
    *process* working directory, and on this runtime the working directory is not
    a storage key -- the capture's twenty-five threads span seven different
    ``cwd``s. Filtering by a default nobody chose would report "no sessions" to
    an account with hundreds. So the listing is account-wide and each row carries
    its own ``cwd`` as :attr:`~modelpass.types.SessionInfo.project_folder`.

    One page. ``nextCursor`` is in the response and is **not** followed:
    continuing a cursor has never been driven, and a listing that silently
    stopped somewhere in the middle of a second page would be worse than one that
    is honestly the newest ``limit``. Ordering comes from the server, whose
    ``sortDirection`` defaults to descending (schema-sourced), which is the
    newest-first the adapter contract asks for.
    """
    return {"limit": limit}


def thread_read_params(thread_id: str) -> dict[str, Any]:
    """Params for :data:`THREAD_READ_METHOD`: metadata only.

    ``includeTurns`` is sent as ``false`` -- the value that was **driven** on
    2026-08-31 -- rather than left out. Its ``true`` branch is what a first pass
    at ``get_history()`` would reach for, and the vendor's own schema says not
    to: *"Full-history hydration is deprecated for paginated threads; prefer a
    metadata-only read and page with ``thread/turns/list`` and
    ``thread/items/list``."* Every thread in the capture is
    ``historyMode: "paginated"``.
    """
    return {"threadId": thread_id, "includeTurns": False}


def thread_items_list_params(
    thread_id: str, *, cursor: str | None = None, limit: int = THREAD_ITEMS_LIMIT
) -> dict[str, Any]:
    """Params for :data:`THREAD_ITEMS_LIST_METHOD`: one page of a thread's items.

    Schema-sourced from ``ThreadItemsListParams`` on 2026-08-31 and **undriven**.
    ``threadId`` is the only required field; ``turnId`` would narrow it to one
    turn and is not sent, because a history is the whole thread.

    ``sortDirection`` is left out on purpose: it *defaults to ascending*, which
    is conversation order, and stating a default is how a default drifts without
    anybody noticing. ``cursor`` continues after the last item of the previous
    page.
    """
    params: dict[str, Any] = {"threadId": thread_id, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    return params


def thread_resume_params(
    thread_id: str,
    *,
    cwd: str | None = None,
    base_instructions: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Params for :data:`THREAD_RESUME_METHOD`.

    ``threadId`` is the only required field (``ThreadResumeParams``,
    schema-sourced 2026-08-31), and the vendor's own description says *"Prefer
    using thread_id whenever possible"* over the ``history`` and ``path`` routes.

    ``excludeTurns: true`` is sent because modelpass does not read the resume
    response's transcript: the schema calls full-history hydration deprecated for
    paginated threads and points at ``thread/turns/list`` /
    ``thread/items/list``, which is where :func:`thread_items_history` goes.

    **No ``dynamicTools``, because the params object has no such field.** Codex
    restores a thread's persisted caller-tool registration on resume, and there
    is no way to restate or clear it here -- which is why ``thread/start`` omits
    the key rather than sending ``[]`` (:func:`dynamic_tools_param`). A resumed
    session may therefore be asked to run a tool it never declared this time
    round; :class:`CallerToolDispatcher` answers that with a well-formed
    ``success: false`` naming what modelpass did send, rather than leaving the
    server without a response.

    ``baseInstructions`` and ``model`` are optional overrides and are sent only
    when the caller has one; ``cwd`` names where the resumed turns run.
    """
    params: dict[str, Any] = {"threadId": thread_id, "excludeTurns": True}
    if cwd:
        params["cwd"] = cwd
    if base_instructions is not None:
        params["baseInstructions"] = base_instructions
    if model:
        params["model"] = model
    return params


def turn_interrupt_params(thread_id: str, turn_id: str) -> dict[str, Any]:
    """Params for :data:`TURN_INTERRUPT_METHOD`: ``{threadId, turnId}``.

    Both **required** (``TurnInterruptParams``, schema-sourced 2026-08-31), which
    is why an interrupt with either id missing is not attempted at all.
    """
    return {"threadId": thread_id, "turnId": turn_id}


# --- thread/list -> SessionInfo ----------------------------------------------------


def session_info_from_thread(
    thread: Mapping[str, Any], *, connection: str
) -> SessionInfo:
    """One ``thread/list`` row as the normalized listing type.

    Every field below was **driven** on 2026-08-31 -- the capture's
    ``probe:thread/list`` carries twenty-five real rows -- so this mapping is
    read off wire text rather than off a schema.

    What the wire has no answer for stays ``None``, which
    :class:`~modelpass.types.SessionInfo` defines as *not reported* rather than
    *empty*:

    * ``message_count`` -- every row's ``turns`` is ``[]`` in the capture, on a
      thread that had just taken two turns. The listing does not count them, and
      counting them here would mean a ``thread/items/list`` round trip per row.
    * ``title`` comes from ``name``, the thread's own title, and is ``None``
      where the thread has none. **``preview`` is
      deliberately not promoted into it**: it is the first prompt verbatim, not a
      title, and a listing whose titles were the opening words of a message would
      be a different thing wearing the field's name. It rides in ``vendor``.

    ``project_folder`` is the row's own ``cwd``. On this runtime that is not a
    storage key -- the store is flat and ``thread/list`` is account-wide -- so it
    is a fact about where the thread ran, not the argument needed to find it
    again. The id alone is enough for :meth:`~modelpass.bridge.Bridge.resume_chat`.
    """
    cwd = thread.get("cwd")
    return SessionInfo(
        id=str(thread.get("id") or ""),
        connection=connection,
        runtime=Runtime.OPENAI_SDK,
        project_folder=cwd if isinstance(cwd, str) and cwd else None,
        created_at=_iso_from_epoch_seconds(thread.get("createdAt")),
        updated_at=_iso_from_epoch_seconds(thread.get("updatedAt")),
        title=_non_empty_str(thread.get("name")),
        message_count=None,
        vendor={
            "sessionId": thread.get("sessionId"),
            "forkedFromId": thread.get("forkedFromId"),
            "parentThreadId": thread.get("parentThreadId"),
            "preview": thread.get("preview"),
            "ephemeral": thread.get("ephemeral"),
            "recencyAt": _iso_from_epoch_seconds(thread.get("recencyAt")),
            "path": thread.get("path"),
            "cliVersion": thread.get("cliVersion"),
            "source": thread.get("source"),
            "modelProvider": thread.get("modelProvider"),
            "status": thread.get("status"),
        },
    )


def threads_from_list(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The rows out of a ``thread/list`` response, in the server's own order.

    ``{"data": [...]}`` was driven 2026-08-31, and ``data`` is the schema's only
    required field. Not re-sorted: ``ThreadListParams.sortDirection`` defaults to
    descending, so newest-first is the server's job and imposing a second
    ordering on top would hide the day it stops being true.
    """
    data = result.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        return ()
    return tuple(row for row in data if isinstance(row, Mapping))


def _non_empty_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _iso_from_epoch_seconds(value: Any) -> str | None:
    """ISO 8601 UTC from the thread record's epoch **seconds**, or ``None``.

    Seconds, not milliseconds, and the capture is what says so: ``createdAt
    1788193277`` is 2026-08-31, while the same number read as milliseconds is
    1970. The Anthropic listing carries milliseconds and has its own converter;
    the two runtimes genuinely differ and neither guesses.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        return datetime.datetime.fromtimestamp(value, datetime.UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


# --- thread/items/list -> history --------------------------------------------------


#: How a non-message thread item is written into a read-back history: the item's
#: own ``type``, plus the one field that says *which* one it was.
#:
#: **Marked, never dropped**, and that is the whole fidelity argument. Nineteen
#: ``ThreadItem`` variants exist (schema-sourced 2026-08-31) and only two of them
#: are messages; a history that silently kept those two would tell a caller their
#: conversation was three turns long when the model ran four commands inside it.
#: The Anthropic adapter already answers this way -- ``flatten_tool_content``
#: renders a replayed tool result as ``[tool_result]`` -- so the two runtimes'
#: ``get_history()`` produce the same *shape* of transcript rather than two
#: different conventions the caller has to learn.
_ITEM_DETAIL_FIELDS: dict[str, str] = {
    "dynamicToolCall": "tool",
    "mcpToolCall": "tool",
    "commandExecution": "command",
    "functionCallOutput": "name",
    "webSearch": "query",
    "collabAgentToolCall": "tool",
    "imageView": "path",
}

#: Cap on one marker. A ``commandExecution`` carries the whole command line and a
#: history is introspection, not a log file.
_HISTORY_MARKER_MAX_CHARS = 500


def thread_item_message(item: Mapping[str, Any]) -> Message | None:
    """One ``ThreadItem`` as a history message, or ``None`` if it carries nothing.

    The two message variants are mapped as themselves:

    * ``userMessage`` -> :attr:`~modelpass.types.Role.USER`, its ``content`` blocks
      flattened (driven: ``[{"type": "text", "text": ...}]``);
    * ``agentMessage`` -> :attr:`~modelpass.types.Role.ASSISTANT`, its ``text``.

    Everything else becomes an assistant-role **marker** -- ``[commandExecution
    ls -la]``, ``[reasoning]``, ``[contextCompaction]`` -- because those items
    happened inside the model's own turn and dropping them would hand back a
    different conversation from the one the model is holding. That is precisely
    the objection ``CodexSession.history()``'s docstring records against
    reassembled transcripts, and it applies to a faithful read-back too: the
    danger is not where the text came from, it is a history that is *quietly
    shorter than the truth*.

    ``None`` for an item that flattens to nothing at all -- an **empty
    ``reasoning``**, of which the live capture has one (``summary: []``,
    ``content: []``), is an artifact of the item stream rather than something the
    conversation contains. Same rule the Anthropic mapper applies, and it is
    narrow on purpose: ``contextCompaction`` carries no fields either and is kept,
    because *"the earlier history was compacted"* is a fact about the
    conversation and not an empty slot in it.
    """
    kind = item.get("type")
    if not isinstance(kind, str):
        return None
    if kind == "userMessage":
        text = _flatten_content_items(item.get("content"))
        return Message(Role.USER, text) if text else None
    if kind == "agentMessage":
        text = item.get("text")
        return Message(Role.ASSISTANT, text) if isinstance(text, str) and text else None
    if kind == "reasoning" and not (item.get("summary") or item.get("content")):
        return None
    detail: Any = item.get(_ITEM_DETAIL_FIELDS.get(kind, ""))
    if isinstance(detail, (list, tuple)):
        detail = " ".join(str(part) for part in detail)
    marker = f"[{kind} {detail}]" if isinstance(detail, str) and detail else f"[{kind}]"
    return Message(Role.ASSISTANT, marker[:_HISTORY_MARKER_MAX_CHARS])


def thread_items_history(entries: Sequence[Any]) -> tuple[Message, ...]:
    """A page of ``ThreadItemEntry`` objects as history messages.

    ``{"item": ThreadItem, "turnId": str}`` per entry (``ThreadItemEntry``,
    schema-sourced 2026-08-31), both required. The ``turnId`` is not used: the
    page is already in conversation order and grouping by turn would impose a
    structure :class:`~modelpass.types.Message` has nowhere to put.
    """
    history: list[Message] = []
    for entry in entries:
        item = entry.get("item") if isinstance(entry, Mapping) else None
        if not isinstance(item, Mapping):
            continue
        message = thread_item_message(item)
        if message is not None:
            history.append(message)
    return tuple(history)


def items_page(result: Mapping[str, Any]) -> tuple[tuple[Any, ...], str | None]:
    """One ``thread/items/list`` response as ``(entries, next_cursor)``.

    ``nextCursor`` is ``None`` when there are no more items
    (``ThreadItemsListResponse``, schema-sourced 2026-08-31), which is what ends
    the paging loop in ``CodexAppServerSession.history()``.
    """
    data = result.get("data")
    entries = (
        tuple(data)
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes))
        else ()
    )
    cursor = result.get("nextCursor")
    return entries, cursor if isinstance(cursor, str) and cursor else None
