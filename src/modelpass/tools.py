"""Caller-supplied tools, exposed to a runtime as in-process MCP (D12).

The shape a caller writes::

    def look_up(args):
        return f"the answer for {args['topic']}"

    tools = [ToolDef(
        name="look_up",
        description="Look a topic up in the local index.",
        parameters={
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
        handler=look_up,
    )]

    bridge.chat(connection="claude-sub", message="...", tools=tools)

The runtime then runs the model/tool loop itself, inside one modelpass call, while
``look_up`` executes **in the caller's own process** -- so it keeps its closures,
its database handle, its UI callbacks. That inversion is the whole point (D12):
the alternative is a caller-side loop that pays a fresh stateless run, plus the
runtime's per-call harness overhead, for every round trip.

Three choices worth naming:

* **JSON Schema, not Python type hints.** The Anthropic SDK accepts a shorthand
  dict of Python types, but it cannot express enums, ranges, optional fields or
  nested objects, and it is one vendor's convenience rather than a standard.
  JSON Schema is what MCP itself speaks, so it is what crosses this boundary.
* **Stdlib only.** No pydantic, no attrs, no decorator magic. ``ToolDef`` is a
  frozen dataclass a caller can build from a dict in a loop, which is what a
  bridge from an existing tool registry (LangChain's, say) actually needs.
* **Validated at construction.** A malformed tool should fail where it was
  written, not three layers down inside a vendor SDK during a spend.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import InvalidTool
from .schema import as_object_schema

__all__ = ["TOOL_NAME_PATTERN", "ToolDef", "ToolResult", "normalize_tools"]

#: What the runtimes accept as a tool name. Both v1 runtimes address tools as
#: ``mcp__<server>__<tool>``, so a name containing ``__`` or a dot would be
#: ambiguous to parse back out of an event; the pattern excludes both.
TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")

#: What a handler may return. A plain string is the common case; a mapping is
#: passed through for callers that want to shape MCP content blocks themselves.
ToolResult = str | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolDef:
    """One caller-supplied tool.

    ``handler`` is called with the model's arguments as a single mapping and may
    be sync or async; a sync handler is run off the event loop so a blocking
    call cannot stall the runtime's stream. It returns text, or a mapping in the
    MCP ``{"content": [...], "is_error": bool}`` shape for callers that need
    images or explicit failures.

    A handler that raises is reported back to the model as a failed tool result
    rather than killing the run -- the model can then retry, try another tool,
    or explain the failure, which is the behavior that makes an agent loop
    useful. The exception type and message reach the model, so a handler should
    not raise anything carrying a secret.
    """

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    handler: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not TOOL_NAME_PATTERN.match(self.name):
            raise InvalidTool(
                f"tool name {self.name!r} is not usable: names must start with a letter "
                "and contain only letters, digits, '_' and '-' (max 64 chars). Both v1 "
                "runtimes address tools as 'mcp__<server>__<tool>', so '__' and '.' "
                "cannot appear in a name"
            )
        if "__" in self.name:
            raise InvalidTool(
                f"tool name {self.name!r} contains '__', which is the separator the "
                "runtimes use in 'mcp__<server>__<tool>' and would make the tool's "
                "server ambiguous in a tool_call event"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise InvalidTool(
                f"tool {self.name!r} needs a non-empty description -- it is what the "
                "model reads to decide whether to call the tool at all"
            )
        if not isinstance(self.parameters, Mapping):
            raise InvalidTool(
                f"tool {self.name!r} parameters must be a JSON Schema mapping, got "
                f"{type(self.parameters).__name__}"
            )
        if self.handler is not None and not callable(self.handler):
            raise InvalidTool(
                f"tool {self.name!r} handler must be callable, got "
                f"{type(self.handler).__name__}"
            )

    def json_schema(self) -> dict[str, Any]:
        """The parameters as a JSON Schema object the runtimes will accept.

        A bare properties mapping is wrapped rather than rejected: callers
        assembling schemas from an existing registry routinely have the
        properties without the envelope, and guessing wrong here is cheap to
        correct while rejecting it is not.

        The wrapping rule moved to :func:`modelpass.schema.as_object_schema` when
        structured output arrived (D13), so ``schema=`` and ``tools=`` make the
        same guess about the same input rather than growing two dialects of it.
        """
        return as_object_schema(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        """The wire form -- no handler, because a callable is not protocol (D11)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema(),
        }


def normalize_tools(
    tools: Iterable[ToolDef | Mapping[str, Any]] | None,
) -> tuple[ToolDef, ...]:
    """Coerce and validate a caller's tool list, rejecting duplicate names.

    Duplicates are an error rather than a last-one-wins merge: two tools with
    one name means the model's choice silently routes to whichever the loop
    happened to keep, and that is a bug the caller should hear about here.
    """
    if tools is None:
        return ()
    normalized: list[ToolDef] = []
    for item in tools:
        if isinstance(item, ToolDef):
            normalized.append(item)
        elif isinstance(item, Mapping):
            unknown = set(item) - {"name", "description", "parameters", "handler"}
            if unknown:
                raise InvalidTool(
                    "unknown tool key(s): " + ", ".join(sorted(unknown))
                )
            normalized.append(
                ToolDef(
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    parameters=item.get("parameters", {}),
                    handler=item.get("handler"),
                )
            )
        else:
            raise InvalidTool(f"cannot read a tool from {type(item).__name__}")

    seen: set[str] = set()
    for tool in normalized:
        if tool.name in seen:
            raise InvalidTool(f"duplicate tool name {tool.name!r}")
        seen.add(tool.name)
    return tuple(normalized)
