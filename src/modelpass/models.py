"""What models a runtime will accept, read from the vendor's own state.

Written for the bench's playground, where "type the model id from memory" is a
bad answer, and kept in core because any consumer building a model picker needs
the same thing.

The honesty rule for this module is the one D6 established for capabilities:
**say where the list came from, and never present a guess as the vendor's
list.** So a catalogue carries a ``source`` phrase and an ``exact`` flag, and
the two subscription runtimes land on opposite sides of it.

``Bridge.models(connection, refresh=True)`` also lists API models using the
connection's explicit credential and endpoint through its vendor adapter.
Without refresh, API discovery reads only successful metadata cached in that
Bridge instance. No generations, fallback accounts, run receipts, or prompt data
are involved. A catalogue may include non-chat models; listing is not a claim
that every model supports a particular workload.

``openai-sdk`` -- exact
    The Codex CLI maintains ``~/.codex/models_cache.json`` and refreshes it
    itself. modelpass reads it **read-only, as plain JSON**, never through a
    vendor parser: the file is written by whichever CLI version last ran and can
    carry newer enum variants than the installed binary understands, so a strict
    parse would fail on exactly the machines that have the freshest list. Entries
    marked ``visibility: "hide"`` are the vendor's own "do not offer this to a
    user" signal and are filtered out.

    The cache records the ``client_version`` that wrote it, and it can be *newer*
    than the CLI you have installed. That gap is a real failure mode already hit
    on this project (codex-cli 0.117.0 rejecting ``gpt-5.6-sol``), so the
    catalogue reports the version rather than hiding it, and a rejected slug
    surfaces as an ordinary terminal error.

``anthropic-sdk`` -- not exact
    Checked 2026-08-17 against the installed artifacts: the Claude Code CLI
    (2.1.63) has no ``models`` subcommand, ``~/.claude`` holds no model
    manifest, and ``claude-agent-sdk`` exposes no enumeration API. What *is*
    verifiable is that the CLI resolves the aliases ``sonnet``, ``opus`` and
    ``haiku`` -- its own ``--model`` help documents the alias form ("an alias for
    the latest model (e.g. 'sonnet' or 'opus') or a model's full name") and all
    three appear in the shipped bundle's alias handling.

    So the catalogue offers those three, labelled as aliases the *runtime*
    resolves, and says plainly that this is not the vendor's list. Inventing a
    hardcoded table of ``claude-*-4-6`` ids and presenting it as one would be a
    made-up fact wearing a vendor's name -- the same thing D4's amendment threw
    out for guard defaults.

Series and versions (2026-11-18)
    ``Bridge.models`` shows the newest version of each series unless asked for
    ``versions="all"``. The grouping is :func:`parse_model_id`'s reading of the
    id string -- no vendor says which id supersedes which -- so it is timid: an
    id it cannot read is never grouped and never hidden.

Either way the caller keeps a free-text field, because model churn in this space
is constant, and the receipt plus the terminal event are what say which model
actually applied.

Facts here are dated 2026-08-17. Re-verify before relying on them.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import NoSuchModel
from .runtimes import Runtime

__all__ = [
    "ANTHROPIC_ALIASES",
    "CODEX_MODELS_CACHE",
    "MODEL_VERSIONS",
    "OPENAI_SHORT_NAMES",
    "ModelCatalogue",
    "ModelChoice",
    "ModelName",
    "codex_home",
    "models_for",
    "openai_full_id",
    "openai_series",
    "parse_model_id",
]

#: The two answers ``Bridge.models(..., versions=)`` takes. ``"latest"`` is the
#: default since 2026-11-18: one entry per series, the newest. ``"all"`` is the
#: vendor's whole list, which is what the default was before that date.
MODEL_VERSIONS: tuple[str, ...] = ("latest", "all")

#: Where the Codex CLI keeps its refreshed model list, relative to the codex home.
CODEX_MODELS_CACHE = "models_cache.json"

_CODEX_HOME_ENV = "CODEX_HOME"

#: Aliases the Claude Code runtime resolves to "the latest model of that family".
#: Not a model list -- see the module docstring for why there isn't one.
ANTHROPIC_ALIASES: tuple[tuple[str, str], ...] = (
    ("fable", "Fable (alias)"),
    ("sonnet", "Sonnet (alias)"),
    ("opus", "Opus (alias)"),
    ("haiku", "Haiku (alias)"),
)

#: The OpenAI series a caller may name where a model id goes (2026-11-25), in any
#: case, bare or as ``gpt-<name>``. A closed list on purpose: any other string is
#: a model id and is sent as typed.
OPENAI_SHORT_NAMES: tuple[str, ...] = ("sol", "astra", "terra", "luna")

_SHORT = "|".join(OPENAI_SHORT_NAMES)
_OPENAI_SHORT = re.compile(rf"(?:gpt-)?({_SHORT})", re.IGNORECASE)
_OPENAI_FULL = re.compile(
    rf"gpt-\d{{1,2}}(?:\.\d{{1,2}})*-(?:{_SHORT})(?:-\d{{8}})?", re.IGNORECASE
)


def openai_series(name: str) -> str | None:
    """The series a short name means (``"Sol"`` -> ``"gpt-sol"``), else ``None``."""
    match = _OPENAI_SHORT.fullmatch(name.strip())
    return f"gpt-{match.group(1).lower()}" if match else None


def openai_full_id(name: str) -> str | None:
    """A versioned id of one of those series, lower-cased, else ``None``."""
    wanted = name.strip()
    return wanted.lower() if _OPENAI_FULL.fullmatch(wanted) else None


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """One offerable model. ``id`` is what goes on the wire as ``model=``."""

    id: str
    label: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class ModelName:
    """A model id read as *series* + *version* (+ dated snapshot), 2026-11-18.

    ``gpt-6.2-sol`` is series ``gpt-sol`` at version ``6.2``;
    ``claude-opus-4-5-20251101`` is series ``claude-opus`` at ``4.5``, snapshot
    ``20251101``. This is a reading of the *string*, not vendor metadata: no
    models endpoint says which id supersedes which, so the grouping is
    modelpass's own and is deliberately timid. An id it cannot read --
    ``gpt-4o``, ``o4-mini``, ``llama3.1`` -- has an empty ``series`` and is
    never grouped with, or hidden behind, anything.
    """

    id: str
    #: The id with its version and snapshot removed, or ``""`` when unread.
    series: str = ""
    version: tuple[int, ...] = ()
    #: A trailing date, digits only (``20251101``), or ``""``.
    snapshot: str = ""

    @property
    def version_text(self) -> str:
        return ".".join(str(part) for part in self.version)

    @property
    def rank(self) -> tuple[tuple[int, ...], bool, str]:
        """Sort key, newest last: the highest version; then the undated id
        (the vendor's own moving name for that version) over a dated
        snapshot; then the newest snapshot."""
        return (self.version, not self.snapshot, self.snapshot)


#: A version token: one or two digits, optionally dotted (``6``, ``6.2``, ``5``).
#: Longer digit runs -- ``0613``, ``001`` -- are build tags, not versions, and
#: stay in the series so a tagged build is never mistaken for a newer release.
_VERSION_TOKEN = re.compile(r"^\d{1,2}(\.\d{1,2})*$")
_DATE8 = re.compile(r"^\d{8}$")


def parse_model_id(model_id: str) -> ModelName:
    """Read a model id as series + version. Never raises; unread ids say so."""
    tokens = model_id.split("-")
    snapshot = ""
    if len(tokens) > 1 and _DATE8.match(tokens[-1]):
        snapshot = tokens.pop()
    elif (
        len(tokens) > 3
        and re.fullmatch(r"\d{4}", tokens[-3])
        and re.fullmatch(r"\d{2}", tokens[-2])
        and re.fullmatch(r"\d{2}", tokens[-1])
    ):
        snapshot = "".join(tokens[-3:])
        tokens = tokens[:-3]
    start = next((i for i, t in enumerate(tokens) if _VERSION_TOKEN.match(t)), None)
    if start is None:
        return ModelName(id=model_id)
    end = start
    while end < len(tokens) and _VERSION_TOKEN.match(tokens[end]):
        end += 1
    rest = tokens[:start] + tokens[end:]
    if not any(rest):
        return ModelName(id=model_id)
    version = tuple(int(part) for t in tokens[start:end] for part in t.split("."))
    return ModelName(
        id=model_id, series="-".join(rest).lower(), version=version, snapshot=snapshot
    )


def _version_tuple(version: str | int | float) -> tuple[int, ...]:
    text = str(version).strip().lower().removeprefix("v")
    parts = [p for p in re.split(r"[.\-_ ]", text) if p]
    if not parts or not all(p.isdigit() for p in parts):
        raise NoSuchModel(f"version must look like '6.1' or '5', got {version!r}")
    return tuple(int(p) for p in parts)


@dataclass(frozen=True, slots=True)
class ModelCatalogue:
    """Models on offer for a runtime, plus an honest account of where they came from."""

    runtime: Runtime
    choices: tuple[ModelChoice, ...] = ()
    #: A human phrase naming the origin, shown next to the picker.
    source: str = ""
    #: ``True`` only when this is the vendor's own list, read from vendor state.
    #: ``False`` for aliases, and for anything modelpass assembled itself.
    exact: bool = False
    #: Why the list is empty or degraded, when it is.
    problem: str | None = None
    #: Anything worth saying alongside the list -- a stale-cache warning, a note
    #: that these are aliases rather than ids.
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return bool(self.choices)

    # --- series and versions (2026-11-18) -------------------------------------

    def latest(self) -> ModelCatalogue:
        """This catalogue with only the newest version of each series.

        The vendor's order is kept. A model whose id does not read as a series
        (see :class:`ModelName`) is always kept: hiding something on a guess
        would be the picker lying by omission. What was left out is counted in
        a note, with the way to get it back.
        """
        best: dict[str, ModelName] = {}
        for choice in self.choices:
            name = parse_model_id(choice.id)
            if not name.series:
                continue
            held = best.get(name.series)
            if held is None or name.rank > held.rank:
                best[name.series] = name
        keep = {name.id for name in best.values()}
        kept = tuple(
            c for c in self.choices if c.id in keep or not parse_model_id(c.id).series
        )
        hidden = len(self.choices) - len(kept)
        if not hidden:
            return self
        note = (
            f"Showing the latest version of each series; {hidden} older "
            f"version{'s' if hidden != 1 else ''} not shown. Ask for versions='all' "
            "for the full list."
        )
        return ModelCatalogue(
            runtime=self.runtime, choices=kept, source=self.source, exact=self.exact,
            problem=self.problem, notes=(*self.notes, note),
        )

    def series(self) -> tuple[str, ...]:
        """Every series name in this catalogue, in the vendor's order."""
        seen: dict[str, None] = {}
        for choice in self.choices:
            name = parse_model_id(choice.id).series
            if name:
                seen.setdefault(name)
        return tuple(seen)

    def versions(self, name: str) -> tuple[ModelChoice, ...]:
        """Every version of one series, newest first.

        ``name`` is a series (``gpt-sol``, ``claude-opus``) or its short form
        (``sol``, ``opus``). Raises :class:`~modelpass.errors.NoSuchModel` when
        it names none, or more than one.
        """
        series = self._series_named(name)
        if series is None:
            raise NoSuchModel(self._unknown(name))
        rows = [
            (parse_model_id(c.id), c)
            for c in self.choices
            if parse_model_id(c.id).series == series
        ]
        rows.sort(key=lambda row: row[0].rank, reverse=True)
        return tuple(choice for _, choice in rows)

    def resolve(
        self, name: str, version: str | int | float | None = None
    ) -> ModelChoice:
        """The one model ``name`` (and optionally ``version``) picks.

        * An exact id in the list is itself.
        * A series name with no version is that series' newest.
        * A series name with a version is exactly that version -- its undated
          id if there is one, otherwise its newest snapshot.

        When this catalogue is not the vendor's own complete list (``exact`` is
        false, or it carries a ``problem``), an unknown ``name`` with no
        ``version`` is passed through as typed, because the list cannot say it
        is wrong; with a ``version`` it is refused, because modelpass will not
        assemble an id it has not seen.
        """
        wanted = str(name).strip()
        if not wanted:
            raise NoSuchModel("a model name is required")
        authoritative = self.exact and self.problem is None
        for choice in self.choices:
            if choice.id.lower() == wanted.lower():
                if version is None:
                    return choice
                if authoritative:
                    raise NoSuchModel(
                        f"{choice.id!r} is already an exact model id; drop version="
                        f"{version!r}, or name the series instead"
                    )
                break
        series = self._series_named(wanted)
        if series is None:
            if authoritative:
                raise NoSuchModel(self._unknown(wanted))
            if version is None:
                return ModelChoice(
                    id=wanted,
                    label=wanted,
                    description="as typed; not checked against a vendor list",
                )
            why = self.problem or self.source or "this runtime has no model list"
            raise NoSuchModel(
                f"cannot pick version {version!r} of {wanted!r}: {why}. "
                "Pass the exact model id as the name instead"
            )
        candidates = self.versions(series)
        if version is None:
            return candidates[0]
        target = _version_tuple(version)
        for choice in candidates:
            if parse_model_id(choice.id).version == target:
                return choice
        offered = ", ".join(
            dict.fromkeys(parse_model_id(c.id).version_text for c in candidates)
        )
        raise NoSuchModel(
            f"{series!r} has no version {'.'.join(map(str, target))}; "
            f"this list offers: {offered}"
        )

    def _series_named(self, name: str) -> str | None:
        wanted = name.strip().lower()
        known = self.series()
        if wanted in known:
            return wanted
        matches = [s for s in known if s.endswith("-" + wanted)]
        if len(matches) > 1:
            raise NoSuchModel(
                f"{name!r} matches more than one series: {', '.join(matches)}. "
                "Name one of them in full"
            )
        return matches[0] if matches else None

    def _unknown(self, name: str) -> str:
        known = ", ".join(self.series()) or "none"
        return (
            f"no model or series named {name!r} in this list "
            f"({self.source or 'no source'}). Series on offer: {known}. To send an "
            "id this list does not carry, pass model= on the call itself"
        )


def codex_home(env: dict[str, str] | None = None) -> Path:
    """The Codex CLI's home directory, honouring ``CODEX_HOME``.

    Read from the *ambient* environment on purpose, and this is the one place in
    modelpass where that is right: it is a path lookup, not a credential, and the
    question being answered is "where does the CLI on this machine keep its
    state", which only the ambient environment can answer. Nothing read from
    here ever reaches a runtime launch -- the scrubbed plan does that.
    """
    source = os.environ if env is None else env
    configured = source.get(_CODEX_HOME_ENV)
    if configured:
        return Path(configured)
    return Path.home() / ".codex"


def models_for(
    runtime: Runtime | str,
    *,
    codex_root: str | os.PathLike[str] | None = None,
) -> ModelCatalogue:
    """The model catalogue for a runtime. Never raises; degrades and says so."""
    try:
        resolved = Runtime(runtime)
    except ValueError:
        # "Never raises" has to include the argument, not just the file being
        # read. A picker is a UI affordance; a caller asking about a runtime
        # this build has never heard of should get an empty list and a reason.
        return ModelCatalogue(
            runtime=Runtime.ANTHROPIC_SDK,
            source=f"unknown runtime {runtime!r}",
            problem=f"{runtime!r} is not a runtime this build of modelpass knows",
        )
    if resolved is Runtime.OPENAI_SDK:
        return _codex_catalogue(Path(codex_root) if codex_root else codex_home())
    if resolved is Runtime.ANTHROPIC_SDK:
        return _anthropic_catalogue()
    return ModelCatalogue(
        runtime=resolved,
        source="no model source is known for this runtime",
        problem="modelpass has not verified a model list for this runtime",
    )


def _anthropic_catalogue() -> ModelCatalogue:
    return ModelCatalogue(
        runtime=Runtime.ANTHROPIC_SDK,
        choices=tuple(
            ModelChoice(id=alias, label=label, description="resolved by the runtime")
            for alias, label in ANTHROPIC_ALIASES
        ),
        source="documented aliases, not a vendor model list",
        exact=False,
        notes=(
            "The Claude Code runtime exposes no way to enumerate models: no 'models' "
            "subcommand on the CLI (2.1.63), no manifest in ~/.claude, no enumeration "
            "API on claude-agent-sdk (checked 2026-08-17). These are aliases the "
            "runtime resolves to the current model of each family.",
            "For an exact model id -- 'claude-sonnet-4-6' and the like -- type it in "
            "the free-text field. modelpass will not guess one for you.",
        ),
    )


def _codex_catalogue(root: Path) -> ModelCatalogue:
    path = root / CODEX_MODELS_CACHE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ModelCatalogue(
            runtime=Runtime.OPENAI_SDK,
            source=f"{path} (not present)",
            problem=(
                "the Codex CLI has not written a model cache here yet. Run codex once, "
                "or type a model id in the free-text field"
            ),
        )
    except Exception as exc:
        return ModelCatalogue(
            runtime=Runtime.OPENAI_SDK,
            source=str(path),
            problem=f"could not read the Codex model cache: {exc}",
        )

    if not isinstance(raw, dict):
        return ModelCatalogue(
            runtime=Runtime.OPENAI_SDK,
            source=str(path),
            problem="the Codex model cache is not the expected JSON object",
        )

    entries = raw.get("models")
    if not isinstance(entries, list):
        return ModelCatalogue(
            runtime=Runtime.OPENAI_SDK,
            source=str(path),
            problem="the Codex model cache carries no 'models' array",
        )

    choices = _codex_choices(entries)
    written_by = raw.get("client_version")
    fetched = raw.get("fetched_at")
    detail = [str(path)]
    if fetched:
        detail.append(f"fetched {fetched}")
    if written_by:
        detail.append(f"written by codex {written_by}")

    notes = [
        "The Codex CLI maintains this file itself; modelpass only reads it, and only "
        "offers entries the vendor marks visible."
    ]
    if written_by:
        notes.append(
            f"This list was written by codex {written_by}. If your installed CLI is "
            "older it may reject a newer slug -- that arrives as an ordinary terminal "
            "error naming the model, not as a modelpass failure."
        )
    return ModelCatalogue(
        runtime=Runtime.OPENAI_SDK,
        choices=choices,
        source=", ".join(detail),
        exact=True,
        problem=None if choices else "the Codex model cache lists no visible models",
        notes=tuple(notes),
    )


def _codex_choices(entries: list[Any]) -> tuple[ModelChoice, ...]:
    """Visible entries, in the vendor's own priority order.

    Unknown fields are ignored rather than rejected: the file is written by a CLI
    that may be newer than anything modelpass was tested against, and a picker that
    empties itself the moment OpenAI adds a key would be worse than useless.
    """
    picked: list[tuple[int, int, ModelChoice]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        # A denylist, not an allowlist. "hide" is the vendor's own do-not-offer
        # signal and is honoured; anything else -- including a missing key, or a
        # visibility value invented after this code was written -- is offered.
        # Requiring an exact "list" would empty the picker on the first machine
        # whose CLI renamed the field, which is the failure this function's
        # docstring exists to rule out.
        if entry.get("visibility") == "hide":
            continue
        label = entry.get("display_name")
        description = entry.get("description")
        priority = entry.get("priority")
        picked.append(
            (
                priority if isinstance(priority, int) else 10_000,
                index,
                ModelChoice(
                    id=slug,
                    label=label if isinstance(label, str) and label else slug,
                    description=description if isinstance(description, str) else "",
                ),
            )
        )
    picked.sort(key=lambda item: (item[0], item[1]))
    return tuple(choice for _, _, choice in picked)


def api_catalogue(connection, client, *, timeout: float) -> ModelCatalogue:
    """Fetch model metadata only, via an already explicitly authenticated client.

    SDK response shapes verified from installed OpenAI 2.x, Anthropic 0.x and
    google-genai 1.x on 2026-11-01. Never invokes a generation endpoint.
    """
    runtime = connection.runtime
    source = f"{runtime.value} models endpoint"
    deadline = time.monotonic() + timeout
    found: dict[str, ModelChoice] = {}
    cursor = None
    target = client
    problem = None
    try:
        if runtime is not Runtime.GOOGLE_API:
            target = client.with_options(max_retries=0)
        for _ in range(20):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            if runtime is Runtime.GOOGLE_API:
                config = {"page_size": 1000, "http_options": {
                    "timeout": max(1, int(remaining * 1000)),
                    "retry_options": {"attempts": 1}}}
                if cursor:
                    config["page_token"] = cursor
                page = target.models.list(config=config)
                rows = page.page
                next_cursor = page.config.get("page_token")
            else:
                kwargs = {"timeout": remaining}
                if runtime is Runtime.ANTHROPIC_API:
                    kwargs["limit"] = 1000
                    if cursor:
                        kwargs["after_id"] = cursor
                page = target.models.list(**kwargs)
                rows = page.data
                more = page.has_next_page()
                next_cursor = (getattr(rows[-1], "id", None) if rows and more else None)
            for row in rows:
                identifier = getattr(row, "id", None) or getattr(row, "name", None)
                if not isinstance(identifier, str) or not identifier:
                    continue
                if runtime is Runtime.GOOGLE_API:
                    identifier = identifier.removeprefix("models/")
                label = getattr(row, "display_name", None) or identifier
                description = getattr(row, "description", None) or ""
                found[identifier] = ModelChoice(identifier, str(label), str(description))
            if not next_cursor:
                break
            if runtime not in (Runtime.ANTHROPIC_API, Runtime.GOOGLE_API):
                problem = "The model list is partial; this endpoint returned additional pages."
                break
            if next_cursor == cursor or len(found) >= 5000:
                problem = "The model list is partial; the endpoint exceeded discovery limits."
                break
            cursor = next_cursor
        else:
            problem = "The model list is partial; the endpoint exceeded discovery limits."
    except Exception as exc:
        problem = (f"Could not finish listing models ({type(exc).__name__}). "
                   "Check the selected account, endpoint and API access, then retry.")
    finally:
        for handle in (target,) if target is client else (target, client):
            with suppress(Exception):
                handle.close()
    return ModelCatalogue(
        runtime=runtime, choices=tuple(found.values()), source=source, exact=True,
        problem=problem,
        notes=("This is the account's model catalogue, not a promise that every model "
               "supports this runtime's chat or tools. No generation was requested.",))
