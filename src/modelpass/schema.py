"""JSON Schema handling for structured output (D13) -- stdlib only.

Three jobs, deliberately kept small:

1. **Normalize** what a caller passes as ``schema=`` into the object schema the
   runtimes expect, and refuse the shapes that would fail three layers down
   inside a vendor (:func:`normalize_schema`).
2. **Check** a result against that schema, so ``structured_output.valid`` means
   something (:func:`validate_instance`).
3. **Name the one place the two runtimes disagree** about what a schema may look
   like, in code a caller can act on rather than in a paragraph they have to
   remember (:func:`openai_strict_issues`, :func:`to_openai_strict`).

**Why not the ``jsonschema`` package.** Core depends on nothing but the vendor
runtimes (D8), and this is not a general validator: vendor-side validation is
primary on both v1 runtimes -- OpenAI rejects a non-conforming schema at the
request boundary and constrains generation server-side; Anthropic constrains the
tool call it uses to submit the answer. modelpass's check exists to answer *"did
what came back actually match what you asked for"* for the caller's own logging
and fallback logic, not to be the thing standing between the model and a bad
answer.

**What the checker covers, stated once so ``valid=True`` is readable.** ``type``
(including a type list), ``enum``, ``required``, ``properties`` (recursively),
``items`` (recursively), and ``additionalProperties: false``. Composition
keywords -- ``$ref``, ``anyOf``, ``oneOf``, ``allOf``, ``not`` -- and every
constraint keyword (``minimum``, ``pattern``, ``format``, ``minItems``, ...) are
**skipped, not failed**: a subtree modelpass cannot check is reported as no
violation rather than a false one. So ``valid=True`` means *"no violation found
among the keywords modelpass checks"*, which is why the schema travels alongside
the answer and the vendor's own enforcement is the primary line.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import InvalidSchema

__all__ = [
    "CHECKED_KEYWORDS",
    "as_object_schema",
    "build_structured_event",
    "normalize_schema",
    "openai_strict_issues",
    "resolve_schema_name",
    "schema_title",
    "to_openai_strict",
    "validate_instance",
]

#: The keywords :func:`validate_instance` actually checks. Everything else in a
#: schema is carried to the vendor untouched and skipped here -- see the module
#: docstring for why that is a deliberate position rather than a gap.
CHECKED_KEYWORDS = (
    "type",
    "enum",
    "required",
    "properties",
    "items",
    "additionalProperties",
)

#: Keywords that make a subtree uncheckable by this module. Their presence means
#: the subtree is skipped rather than reported as a violation.
_COMPOSITION_KEYWORDS = frozenset({"$ref", "anyOf", "oneOf", "allOf", "not"})

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


# --- normalization ---------------------------------------------------------------


def as_object_schema(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a bare properties mapping into a full object schema.

    Callers assembling schemas from an existing registry routinely have the
    properties without the envelope. Guessing wrong here is cheap to correct;
    rejecting it is not. Shared with :class:`~modelpass.tools.ToolDef`, which had
    this logic first -- structured output reuses it rather than growing a second
    dialect of the same guess.
    """
    schema = dict(parameters)
    if not schema:
        return {"type": "object", "properties": {}}
    if "type" not in schema and "properties" not in schema:
        return {"type": "object", "properties": schema}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def normalize_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a caller's ``schema=``, or raise :class:`InvalidSchema`.

    Refuses at the call boundary rather than at the vendor's: a malformed schema
    that reaches a runtime costs a request, and on one of the two v1 runtimes the
    rejection comes back as an opaque 400 about a ``response_format`` the caller
    never named.

    The schema must be JSON-serializable, because it is going to be serialized --
    into a CLI argument on ``anthropic-sdk`` and into a file on ``openai-sdk``.
    Finding that out here beats finding it out at spawn time.
    """
    if not isinstance(schema, Mapping):
        raise InvalidSchema(
            f"schema must be a JSON Schema mapping, got {type(schema).__name__}"
        )
    normalized = as_object_schema(schema)
    declared = normalized.get("type")
    if declared != "object":
        raise InvalidSchema(
            f"schema type must be 'object', got {declared!r}. Both v1 runtimes "
            "constrain the model's final answer to a JSON object; wrap a bare "
            "array or scalar in a one-property object"
        )
    if not isinstance(normalized.get("properties"), Mapping):
        raise InvalidSchema(
            "schema 'properties' must be a mapping of property name to subschema"
        )
    required = normalized.get("required", [])
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        raise InvalidSchema("schema 'required' must be a list of property names")
    try:
        json.dumps(normalized)
    except (TypeError, ValueError) as exc:
        raise InvalidSchema(
            f"schema is not JSON-serializable ({exc}); it has to be, because it is "
            "passed to the runtime as JSON"
        ) from None
    return normalized


def schema_title(schema: Mapping[str, Any]) -> str | None:
    """The schema's ``title``, when it has a usable one."""
    title = schema.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def resolve_schema_name(schema: Mapping[str, Any], schema_name: str | None) -> str:
    """The name this schema travels under: caller's, else the schema's ``title``.

    The caller's name wins because on a runtime that submits structured output
    through a tool call, **the tool name is a wire-protocol detail the caller may
    depend on** -- a retrieval service's schema-bound call sites name the tool
    after the schema class, and their prompts refer to it. modelpass therefore never invents
    that string when the caller supplied one, and falls back to the schema's own
    ``title`` rather than to something of its own devising.

    Empty when neither exists, which is a real answer: the runtimes' native
    mechanisms do not require a name.
    """
    if isinstance(schema_name, str) and schema_name.strip():
        return schema_name.strip()
    return schema_title(schema) or ""


# --- structural checking ---------------------------------------------------------


def validate_instance(
    instance: Any, schema: Mapping[str, Any], *, path: str = "$"
) -> tuple[bool, tuple[str, ...]]:
    """Check ``instance`` against the subset of ``schema`` this module understands.

    Returns ``(valid, problems)``. ``problems`` are human-readable and path-
    prefixed, because a caller looking at ``valid=False`` wants to know *which
    field*, and a boolean alone sends them to re-read their own schema.

    See the module docstring for exactly which keywords are checked. Anything
    else is skipped rather than failed, so this never reports a violation it
    cannot substantiate.
    """
    problems: list[str] = []
    _check(instance, schema, path, problems)
    return not problems, tuple(problems)


def _check(value: Any, schema: Any, path: str, problems: list[str]) -> None:
    if not isinstance(schema, Mapping):
        return
    if any(keyword in schema for keyword in _COMPOSITION_KEYWORDS):
        # Uncheckable here by design: reporting a violation modelpass cannot
        # substantiate would be worse than reporting nothing.
        return

    declared = schema.get("type")
    if declared is not None and not _type_matches(value, declared):
        problems.append(f"{path}: expected type {_type_label(declared)}, got {_kind(value)}")
        return

    enum = schema.get("enum")
    if isinstance(enum, Sequence) and not isinstance(enum, (str, bytes)):
        if not any(value == option for option in enum):
            problems.append(f"{path}: {value!r} is not one of {list(enum)!r}")

    if isinstance(value, Mapping):
        _check_object(value, schema, path, problems)
    elif isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                _check(item, items, f"{path}[{index}]", problems)


def _check_object(
    value: Mapping[str, Any], schema: Mapping[str, Any], path: str, problems: list[str]
) -> None:
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}

    required = schema.get("required")
    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        for name in required:
            if isinstance(name, str) and name not in value:
                problems.append(f"{path}: missing required property {name!r}")

    if schema.get("additionalProperties") is False:
        for name in value:
            if name not in properties:
                problems.append(f"{path}: unexpected property {name!r}")

    for name, subschema in properties.items():
        if name in value:
            _check(value[name], subschema, f"{path}.{name}", problems)


def _type_matches(value: Any, declared: Any) -> bool:
    names = declared if isinstance(declared, (list, tuple)) else [declared]
    for name in names:
        expected = _JSON_TYPES.get(name) if isinstance(name, str) else None
        if expected is None:
            # An unrecognized type name is not something to fail a value over.
            return True
        if name in ("number", "integer") and isinstance(value, bool):
            # JSON has no bool-is-a-number rule; Python does. Do not inherit it.
            continue
        if isinstance(value, expected):
            return True
    return False


def _type_label(declared: Any) -> str:
    if isinstance(declared, (list, tuple)):
        return " or ".join(str(name) for name in declared)
    return str(declared)


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return type(value).__name__


# --- turning a runtime's answer into the normalized event -------------------------


def build_structured_event(
    schema: Mapping[str, Any],
    *,
    parsed: Any = None,
    raw: str | None = None,
    schema_name: str = "",
) -> tuple[Any, str | None]:
    """Build the ``structured_output`` event, and the reason the run failed if it did.

    Both adapters end up here, from different starting points: ``anthropic-sdk``
    hands over a parsed object *and* the JSON text, ``openai-sdk`` only text. The
    shared rule is what matters and is stated once:

    * something parseable arrived -> an event whose ``valid`` is modelpass's
      structural check, and **no** failure reason. A structurally invalid answer
      is still an answer: the vendor constrained it, modelpass reports what it
      found, and the caller decides.
    * text arrived but is not JSON -> an event carrying the text with
      ``data=None``, **and** a reason, so the run's terminal says the answer was
      unusable rather than the stream implying it was fine.
    * nothing arrived -> no event at all, and a reason. Returning an empty dict
      here would be modelpass inventing the one thing it must never invent.

    Returns ``(event | None, reason | None)``.
    """
    from .types import StructuredOutputEvent  # local: types imports nothing from here

    text = raw if isinstance(raw, str) else ""

    if parsed is None and text.strip():
        try:
            parsed = json.loads(text)
        except ValueError:
            excerpt = text.strip()[:200]
            return (
                StructuredOutputEvent(
                    data=None,
                    raw=text,
                    valid=False,
                    schema_name=schema_name,
                    problems=("the runtime's final answer is not valid JSON",),
                ),
                f"structured output requested but the answer was not JSON: {excerpt!r}",
            )

    if parsed is None:
        return (
            None,
            "structured output requested but the runtime returned no structured "
            "answer and no final text to read one from",
        )

    if not text:
        try:
            text = json.dumps(parsed)
        except (TypeError, ValueError):
            text = str(parsed)

    valid, problems = validate_instance(parsed, schema)
    return (
        StructuredOutputEvent(
            data=parsed,
            raw=text,
            valid=valid,
            schema_name=schema_name,
            problems=problems,
        ),
        None,
    )


# --- the one place the runtimes disagree ------------------------------------------


def openai_strict_issues(schema: Mapping[str, Any], *, path: str = "$") -> tuple[str, ...]:
    """Why ``openai-sdk`` would reject this schema, in its own terms. Pure.

    ``codex exec --output-schema`` feeds the schema to the Responses API's
    **strict** structured-output mode, which is narrower than JSON Schema in two
    specific ways -- both observed live on 2026-08-17 as real 400s from
    ``codex-cli`` 0.117.0, and both recorded as fixtures:

    * every object needs ``additionalProperties: false``
      (*"'additionalProperties' is required to be supplied and to be false"*);
    * every key in ``properties`` must appear in ``required``
      (*"'required' ... to be an array including every key in properties.
      Missing 'score'"*).

    ``anthropic-sdk`` accepts the loose form -- verified live in the same pass,
    with an optional property absent from ``required`` and no
    ``additionalProperties`` anywhere. So a schema derived from a Python
    dataclass with defaults works on one runtime and 400s on the other, which is
    the kind of asymmetry this project reports rather than smooths over.

    This is a **diagnostic, not a gate**: modelpass does not refuse the run. The
    vendor's rejection is precise, arrives before any generation, and is the
    authority; this function exists so a caller can see it coming, and
    :func:`to_openai_strict` exists so they can do something about it.
    """
    issues: list[str] = []
    _strict_issues(schema, path, issues)
    return tuple(issues)


def _strict_issues(schema: Any, path: str, issues: list[str]) -> None:
    if not isinstance(schema, Mapping):
        return
    if any(keyword in schema for keyword in _COMPOSITION_KEYWORDS):
        return
    if schema.get("type") == "object":
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        if schema.get("additionalProperties") is not False:
            issues.append(f"{path}: needs \"additionalProperties\": false")
        required = schema.get("required")
        required = set(required) if isinstance(required, (list, tuple)) else set()
        missing = [name for name in properties if name not in required]
        if missing:
            issues.append(
                f"{path}: every property must be listed in 'required'; missing "
                + ", ".join(repr(name) for name in missing)
            )
        for name, subschema in properties.items():
            _strict_issues(subschema, f"{path}.{name}", issues)
    items = schema.get("items")
    if isinstance(items, Mapping):
        _strict_issues(items, f"{path}[]", issues)


def to_openai_strict(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite a schema into the strict subset ``openai-sdk`` requires.

    Applies OpenAI's own documented recipe for a subset with no notion of
    optional fields: **every property becomes required, and a property that was
    not required gains ``"null"`` in its type**, so "absent" is expressed as
    "present and null". Objects gain ``additionalProperties: false``.

    **Deliberately not applied automatically.** Making an optional field
    mandatory-but-nullable changes the shape of the answer the caller gets back,
    and doing that silently to somebody's schema is the kind of helpfulness that
    turns into a bug report about missing keys. A caller who wants it asks for
    it, once, where they can see what it did::

        schema = to_openai_strict(EXTRACTION_SCHEMA)

    Composition subtrees (``anyOf``, ``$ref``, ...) are returned untouched: they
    cannot be rewritten without changing meaning, and OpenAI's strict mode has
    its own rules for them that modelpass has not verified.
    """
    return _to_strict(schema)


def _to_strict(schema: Any) -> Any:
    if not isinstance(schema, Mapping):
        return schema
    out = dict(schema)
    if any(keyword in out for keyword in _COMPOSITION_KEYWORDS):
        return out
    if out.get("type") == "object":
        properties = out.get("properties")
        properties = dict(properties) if isinstance(properties, Mapping) else {}
        was_required = set(out.get("required") or ())
        rewritten: dict[str, Any] = {}
        for name, subschema in properties.items():
            sub = _to_strict(subschema)
            if name not in was_required:
                sub = _nullable(sub)
            rewritten[name] = sub
        out["properties"] = rewritten
        out["required"] = list(rewritten)
        out["additionalProperties"] = False
    items = out.get("items")
    if isinstance(items, Mapping):
        out["items"] = _to_strict(items)
    return out


def _nullable(schema: Any) -> Any:
    """Widen a subschema's type to admit ``null``, leaving everything else alone."""
    if not isinstance(schema, Mapping):
        return schema
    out = dict(schema)
    declared = out.get("type")
    if declared is None:
        return out
    names = list(declared) if isinstance(declared, (list, tuple)) else [declared]
    if "null" not in names:
        names.append("null")
    out["type"] = names if len(names) > 1 else names[0]
    return out
