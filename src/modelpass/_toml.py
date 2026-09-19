"""A minimal TOML writer for the shapes modelpass emits.

Reading TOML is stdlib (``tomllib``); writing is not, and core takes no runtime
dependencies (D8). Rather than fall back to JSON, this module serializes the
narrow subset the connection store actually produces -- nested tables of
strings, integers, floats, booleans and string arrays.

The reason to keep TOML rather than write JSON: the connection file is user
configuration that the product asks people to read and trust. Comments, plain
``key = value`` lines and no quoting noise are the point. The cost is roughly
sixty lines here, tested by round-tripping through ``tomllib``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["dumps", "dumps_inline"]

_BARE_KEY_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def dumps(data: Mapping[str, Any]) -> str:
    """Serialize a mapping to TOML text. ``None`` values are omitted."""
    lines: list[str] = []
    _dump_table(data, (), lines)
    text = "\n".join(lines)
    return text + "\n" if text else ""


def dumps_inline(value: Any) -> str:
    """Serialize one value as a TOML *inline* expression, tables included.

    Separate from :func:`dumps` because it serves a different consumer: the
    Codex CLI's ``-c key=value`` override parses the value portion as TOML, so
    an MCP server declaration has to arrive as ``{command = "x", args = ["y"]}``
    on a single line. ``dumps`` emits ``[table]`` headers, which a ``-c`` value
    cannot contain.

    ``None`` values inside a table are dropped, matching :func:`dumps`, so an
    unset optional key never becomes a literal.
    """
    if isinstance(value, Mapping):
        pairs = [
            f"{_format_key(str(k))} = {dumps_inline(v)}"
            for k, v in value.items()
            if v is not None
        ]
        return "{" + ", ".join(pairs) + "}"
    if not isinstance(value, (str, bool, int, float)) and isinstance(value, Sequence):
        return "[" + ", ".join(dumps_inline(item) for item in value) + "]"
    return _format_value(value)


def _dump_table(table: Mapping[str, Any], path: tuple[str, ...], lines: list[str]) -> None:
    scalars = {k: v for k, v in table.items() if not isinstance(v, Mapping) and v is not None}
    subtables = {k: v for k, v in table.items() if isinstance(v, Mapping)}
    if path:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("[" + ".".join(_format_key(part) for part in path) + "]")
    for key, value in scalars.items():
        lines.append(f"{_format_key(key)} = {_format_value(value)}")
    for key, value in subtables.items():
        _dump_table(value, (*path, key), lines)


def _format_key(key: str) -> str:
    if key and all(ch in _BARE_KEY_OK for ch in key):
        return key
    return _format_string(key)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _format_string(value)
    if isinstance(value, Sequence):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    raise TypeError(f"cannot serialize {type(value).__name__} to TOML")


def _format_string(value: str) -> str:
    out = []
    for ch in value:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'
