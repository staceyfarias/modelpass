"""Flattening an MCP tool result to the text a ``tool_result`` event carries.

Hoisted here in S4 of the 2026-08-31 app-server transport migration, on the
architect's ruling. Until then the same function lived twice: once in
``adapters/openai`` for ``codex exec``'s ``mcp_tool_call`` items and once in
``adapters/codex_appserver`` for the app-server's ``mcpToolCall`` items. The
duplicate was deliberate and temporary -- the second module could not import the
first without a cycle *if* the first went on to import the second, which is
exactly what S4 made it do. A third module both can import costs nothing and
removes the drift, and the two transports must agree here for the reason the
whole migration exists: one runtime identity means a consumer reading a tool
result does not have to know which transport produced it.

:func:`mcp_result_payload` is the other half and joined it in S6, for the same
reason: it reads what a **caller's own handler returned** and normalizes it into
that envelope. It lived in ``adapters/anthropic`` as ``_tool_result_payload``
until the app-server transport needed the identical reading for
``item/tool/call``, and the :class:`~modelpass.tools.ToolDef` contract it
implements -- *text, or a mapping in the MCP ``{"content": [...], "is_error":
bool}`` shape* -- is one promise made to a caller who does not know or care which
runtime will run their function. Two copies of it would be two dialects of one
documented contract.

Deliberately not merged with ``adapters/anthropic``'s ``flatten_tool_content``.
That one flattens the Agent SDK's *content block* list and has its own
image/document handling; this one flattens an MCP ``result``/``error`` envelope.
They look alike at the ``[{"type": "text", "text": ...}]` layer and diverge
above it, and collapsing two vendor shapes into one function is how a mapping
starts quietly answering for a payload it never saw.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from typing import Any

from ..tools import ToolDef

__all__ = ["ainvoke_handler", "flatten_mcp_result", "mcp_result_payload"]

#: Cap on the JSON fallback. A ``tool_result``'s ``content`` is documented as
#: flattened text for a human or a model to read, not as a transport for an
#: arbitrarily large payload; the whole item still reaches a caller on the
#: ``vendor_event`` path, so nothing is lost by not repeating it here.
_MAX_FALLBACK_CHARS = 4000


def flatten_mcp_result(payload: Any) -> str:
    """An MCP ``result`` or ``error`` payload as text. Total: never raises.

    Three shapes, in the order they are checked: a bare string is already the
    answer; a mapping with ``content`` is the MCP content-block list, whose
    non-text blocks become a ``[<type>]`` marker rather than a URL or a base64
    image; a mapping with ``message`` is an error envelope. Anything else is
    JSON, truncated -- which is the branch that makes this total, and the reason
    a shape nobody anticipated arrives readable instead of raising inside an
    event mapping that is supposed to be pure.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        content = payload.get("content")
        if isinstance(content, (list, tuple)):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, Mapping):
                    text = block.get("text")
                    parts.append(text if isinstance(text, str) else f"[{block.get('type')}]")
            return "\n".join(p for p in parts if p)
        message = payload.get("message")
        if isinstance(message, str):
            return message
    try:
        return json.dumps(payload, default=str)[:_MAX_FALLBACK_CHARS]
    except (TypeError, ValueError):
        return str(payload)[:_MAX_FALLBACK_CHARS]


def mcp_result_payload(value: Any) -> dict[str, Any]:
    """Whatever a caller's handler returned, as an MCP tool-result envelope.

    The :class:`~modelpass.tools.ToolDef` contract, implemented once: a handler
    returns text, or a mapping in the MCP ``{"content": [...], "is_error":
    bool}`` shape for callers that need images or an explicit failure. A mapping
    carrying ``content`` is therefore passed through *unchanged* -- including its
    ``is_error`` flag, which is how a handler reports "I ran and the answer is
    no" without raising.

    Anything else is text: ``None`` is empty, a string is itself, and any other
    object is JSON with a ``str`` fallback. Never raises, because this runs
    where a tool result is being built and a caller's exotic return value must
    not become a second failure on top of whatever the first one was.
    """
    if isinstance(value, Mapping) and "content" in value:
        return dict(value)
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str)
        except (TypeError, ValueError):
            text = str(value)
    return {"content": [{"type": "text", "text": text}]}


async def ainvoke_handler(tool: ToolDef, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Run one caller tool **on the caller's own loop** and envelope the result.

    The async twin of each API adapter's ``invoke_handler``, and the reason
    ticket 1.13 exists in the shape it does. The sync version drives a coroutine
    handler with ``asyncio.run`` on a private loop, because ``run()`` has no loop
    of its own. Here there is one -- the caller's -- and a coroutine is simply
    awaited on it. That is what lets an application whose tool handlers are
    coroutines over its own database, its own session, its own connection pool
    stop marshalling them back across a thread boundary with
    ``run_coroutine_threadsafe``, which is what a downstream agent host does
    today.

    One function rather than one per adapter: the two promises below are made by
    :class:`~modelpass.tools.ToolDef` to a caller who does not know which runtime
    will run their function, so they are worded once.

    * **A handler that raises becomes a failed tool result.** The model can
      retry it, try another tool, or explain -- killing the run instead throws
      away the turns already paid for. The exception's type and message reach
      the model.
    * **A synchronous handler still works.** It runs inline, on the loop, which
      is the same place the sync face would have run it and is the caller's
      business: a handler that blocks blocks the loop that asked for it.
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
            result = await result
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
