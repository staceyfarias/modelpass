"""What models a runtime will accept, read from the vendor's own state.

Written for the bench's playground, where "type the model id from memory" is a
bad answer, and kept in core because any consumer building a model picker needs
the same thing.

The honesty rule for this module is the one D6 established for capabilities:
**say where the list came from, and never present a guess as the vendor's
list.** So a catalogue carries a ``source`` phrase and an ``exact`` flag, and
the two v1 runtimes land on opposite sides of it.

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

Either way the caller keeps a free-text field, because model churn in this space
is constant, and the receipt plus the terminal event are what say which model
actually applied.

Facts here are dated 2026-08-17. Re-verify before relying on them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runtimes import Runtime

__all__ = [
    "ANTHROPIC_ALIASES",
    "CODEX_MODELS_CACHE",
    "ModelCatalogue",
    "ModelChoice",
    "codex_home",
    "models_for",
]

#: Where the Codex CLI keeps its refreshed model list, relative to the codex home.
CODEX_MODELS_CACHE = "models_cache.json"

_CODEX_HOME_ENV = "CODEX_HOME"

#: Aliases the Claude Code runtime resolves to "the latest model of that family".
#: Not a model list -- see the module docstring for why there isn't one.
ANTHROPIC_ALIASES: tuple[tuple[str, str], ...] = (
    ("sonnet", "Sonnet (alias)"),
    ("opus", "Opus (alias)"),
    ("haiku", "Haiku (alias)"),
)


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """One offerable model. ``id`` is what goes on the wire as ``model=``."""

    id: str
    label: str
    description: str = ""


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
            "API on claude-agent-sdk (checked 2026-08-17). These three are aliases the "
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
