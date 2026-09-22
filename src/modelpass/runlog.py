"""The run-receipt audit log: ``~/.modelpass/runs.jsonl`` (2026-08-17).

Why this exists, in the owner's words: *how do I prove it was the subscription?*

Every layer above this one answers that question and then throws the answer
away. The preflight receipt is printed and scrolls off. The ``terminal`` event
carries the auth mode actually used and then the iterator is exhausted. Both are
per-run ephemera, so a month later the honest answer to "which of these runs was
billed to a key" is a shrug. This module makes the answer durable: one JSON line
appended per completed run, in the same home as the connection file.

One line per *connection* a call actually ran on, which is the same thing except
in the one case where it is not: a call that exhausts its allowance and fails
over spends two allowances, so it writes two lines, each stamped with the auth
mode that paid for it. A single line naming only the connection that finished
would report the subscription's tokens as the metered connection's.

Three properties it holds to.

* **It is a ledger, not a transcript.** Timestamp, connection, runtime, the auth
  mode actually used, model, tokens, outcome, and where the vendor said the plan
  stood. No prompts, no responses, no credential -- not even a pointer to one.
  Somebody handed this file has learned what was spent and on whose account, and
  nothing about what was said.
* **It never breaks a run.** Every write is wrapped: a full disk, a read-only
  home, a permissions change mid-session all cost you the log line and nothing
  else. A logging failure that killed a generation would be a worse bug than the
  missing evidence it was trying to record.
* **It is append-only and boring.** JSON Lines, UTF-8, one ``\\n``-terminated
  object per run, no rewriting and no compaction. Anything can read it: ``jq``,
  a spreadsheet, five lines of Python, the bench's run-log page. A format the
  user can audit without modelpass is the only kind worth calling proof.

Opt out with ``[settings] runLog = false`` in ``connections.toml``. It is on by
default deliberately -- evidence that only exists when you remembered to switch
it on is not evidence.

The permanent write-up of what this proves, and the live falsification
experiment behind it, is ``docs/subscription-proof-2026-08-17.md``.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .store import ConnectionStore
from .types import AuthMode, TerminalEvent, TokenUsage

__all__ = [
    "RUNS_FILENAME",
    "RunLog",
    "RunRecord",
    "allowance_fields",
    "run_log_for",
]

RUNS_FILENAME = "runs.jsonl"

#: Serializes appends across threads in this process. See :meth:`RunLog.append`.
_APPEND_LOCK = threading.Lock()

#: Lines read from the tail of the file when rendering recent runs. A ceiling
#: rather than a promise: the file is append-only and unbounded, and a UI page
#: should not be the thing that decides to parse a year of it.
DEFAULT_READ_LIMIT = 50


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One completed run, as it lands on disk.

    Field choices worth naming:

    * ``auth_mode`` is the mode the preflight *detected*, not the one the
      connection declared. Those differ exactly when something went wrong, which
      is the case the log is for.
    * ``guards_configured`` records whether anything bounded this run's spend.
      A log of runs with no note of that would be a ledger that cannot tell an
      unbounded run from a bounded one.
    * ``failed_over_from`` is how a run that started on a subscription and
      finished on a metered key stays distinguishable from one that was metered
      throughout -- the same reason the terminal event carries it (D3).
    * The four ``allowance_*`` fields are **the vendor's own account of how the
      plan is doing**, not arithmetic modelpass performed. ``anthropic-sdk``
      reports them mid-run on ``RateLimitEvent``; the adapter already normalized
      that into a ``rate_limit`` vendor event and nothing consumed it, so the
      answer to "how am I doing against my allowance" died with the run that
      produced it. They are ``None`` on ``openai-sdk``, which reports no
      equivalent -- **absent, not zero**, because a zero utilization is a claim
      that the plan is untouched and modelpass has no basis for it.

      They are also the *only* window-position signal either runtime offers, and
      it arrives during a run rather than before one. So the honest reading of a
      stored value is "where the plan stood the last time a run asked", which is
      exactly what makes it worth keeping: the next call's decision is made
      against the previous call's report, and stale-but-labelled beats absent
      (D20).
    """

    timestamp: str
    connection: str
    runtime: str
    auth_mode: str
    status: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    guards_configured: bool = False
    failed_over_from: str | None = None
    #: Which executable actually ran, from the preflight receipt. Two Codex
    #: installs routinely coexist on one machine -- an npm shim on ``PATH`` and
    #: the build the Desktop app uses -- and they do not behave alike. When a
    #: transport-level defect turns up in one of them, "was this run affected"
    #: is answerable from the ledger or not at all; before this field a run
    #: truncated by the Windows ``.cmd`` shim was indistinguishable here from a
    #: clean one (2026-08-31). ``None`` where the runtime's own SDK owns its
    #: subprocess and no path was ever resolved.
    binary: str | None = None
    #: How many ``cache_control`` breakpoints this run's content carried, and
    #: how many reached the vendor (R3, 2026-09-13). Additive keys: a line
    #: written before they existed reads back as ``0`` and ``0``, which is the
    #: truth about it -- there were no breakpoints to carry. The pair is the
    #: durable half of the receipt's disclosure, and it answers the question a
    #: consumer could not previously ask of its own history: *were the
    #: breakpoints I wrote ever actually honoured?* Two numbers that disagree is
    #: a run that paid for a prefix it thought it had cached.
    cache_breakpoints_requested: int = 0
    cache_breakpoints_honoured: int = 0
    #: What this run asked about word choice and what was sent, off the same
    #: leg's receipt (R5, ticket 1.7). Additive keys, like the breakpoint pair
    #: above and for the same reason: a line written before they existed has
    #: neither, and :func:`_normalize` reads both back as ``{}`` -- which is the
    #: truth about it, since a run nobody set sampling on requested nothing.
    #:
    #: The durable half of the receipt's disclosure, and the question it makes
    #: answerable of a stored history is a RAG evaluation harness's: *was the
    #: judge that
    #: produced this run actually at the temperature its config says?* A stored
    #: run whose two dicts disagree measured with a different instrument from
    #: the one the config describes, and before this the difference left no
    #: trace anywhere.
    sampling_requested: Mapping[str, Any] = field(default_factory=dict)
    sampling_applied: Mapping[str, Any] = field(default_factory=dict)
    #: Why they disagree, one sentence per field. Empty when they agree.
    sampling_notes: tuple[str, ...] = ()
    #: The effort dial, as the durable half of the same disclosure (2026-09-22).
    #: All four come off the stamped terminal rather than the connection, for the
    #: reason :meth:`from_terminal` gives: the terminal is the one object an
    #: adapter cannot forge.
    #:
    #: The question they make answerable of a stored history is the one a
    #: consumer of this library asked and could not answer -- *did asking for
    #: more effort do anything?* Before them, a run at ``xhigh`` and a run at
    #: ``low`` left identical ledger lines.
    #:
    #: * ``reasoning_value`` -- the level sent, in the runtime's own spelling.
    #: * ``reasoning_echo`` -- the level the vendor said it used. ``None`` on
    #:   every runtime but ``openai-sdk``, which is a fact about the vendors and
    #:   not about the run: Anthropic echoes nothing, driven 2026-09-22.
    #: * ``reasoning_output_tokens`` -- what the thinking cost, a **subset** of
    #:   ``output_tokens`` and therefore already inside ``total_tokens``.
    #:   ``None`` means no count, which is not ``0``.
    #: * ``reasoning_metric`` -- which kind of ``None`` that is:
    #:   ``reported`` / ``unreported`` / ``unavailable``. Written out as a word
    #:   rather than left implicit, so a line can be read without a table of
    #:   which runtimes have the field.
    reasoning_value: str | None = None
    reasoning_echo: str | None = None
    reasoning_output_tokens: int | None = None
    reasoning_metric: str | None = None
    #: ``RateLimitInfo.status`` -- ``allowed``, ``allowed_warning``, ``rejected``.
    allowance_status: str | None = None
    #: ``rate_limit_type`` -- which window the figures below describe.
    allowance_window: str | None = None
    #: ``utilization`` as the vendor reported it: a fraction, not a percentage.
    allowance_utilization: float | None = None
    #: When that window resets, as ISO-8601 UTC. The SDK reports an epoch
    #: number; it is converted here so the field reads the same way as
    #: ``timestamp`` to ``jq``, a spreadsheet, or a person. A value that is
    #: already a string is kept verbatim rather than reinterpreted.
    allowance_resets_at: str | None = None

    @classmethod
    def from_terminal(
        cls,
        terminal: TerminalEvent,
        *,
        model: str | None = None,
        guards_configured: bool = False,
        allowance: Mapping[str, Any] | None = None,
        binary: str | None = None,
        cache_breakpoints_requested: int = 0,
        cache_breakpoints_honoured: int = 0,
        sampling_requested: Mapping[str, Any] | None = None,
        sampling_applied: Mapping[str, Any] | None = None,
        sampling_notes: tuple[str, ...] = (),
        when: datetime | None = None,
    ) -> RunRecord:
        """Build a record from the stamped terminal event.

        Sourced from the terminal rather than from the connection on purpose:
        the terminal is the one object in the system that the bridge stamps and
        an adapter cannot forge (D3), so it is the only honest source for "which
        auth mode actually paid for this".

        ``allowance`` is the payload of the last ``rate_limit`` vendor event the
        run produced, or ``None`` when it produced none -- which is every run on
        a runtime that does not report one, and any run that finished before the
        first report arrived. Missing and zero are kept distinct throughout.
        """
        usage: TokenUsage = terminal.usage
        moment = when or datetime.now(UTC)
        mode = terminal.auth_mode
        return cls(
            timestamp=moment.astimezone(UTC).isoformat(timespec="seconds"),
            connection=terminal.connection,
            runtime=str(terminal.runtime.value),
            auth_mode=str(mode.value if isinstance(mode, AuthMode) else mode),
            status=str(terminal.status.value),
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            total_tokens=usage.total_tokens,
            guards_configured=guards_configured,
            failed_over_from=terminal.failed_over_from,
            binary=binary,
            cache_breakpoints_requested=cache_breakpoints_requested,
            cache_breakpoints_honoured=cache_breakpoints_honoured,
            sampling_requested=dict(sampling_requested or {}),
            sampling_applied=dict(sampling_applied or {}),
            sampling_notes=tuple(sampling_notes),
            reasoning_value=terminal.reasoning_value,
            reasoning_echo=terminal.reasoning_echo,
            reasoning_output_tokens=usage.reasoning_output_tokens,
            reasoning_metric=terminal.reasoning_metric,
            **allowance_fields(allowance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "connection": self.connection,
            "runtime": self.runtime,
            "auth_mode": self.auth_mode,
            "status": self.status,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "guards_configured": self.guards_configured,
            "failed_over_from": self.failed_over_from,
            "binary": self.binary,
            "cache_breakpoints_requested": self.cache_breakpoints_requested,
            "cache_breakpoints_honoured": self.cache_breakpoints_honoured,
            "sampling_requested": dict(self.sampling_requested),
            "sampling_applied": dict(self.sampling_applied),
            "sampling_notes": list(self.sampling_notes),
            "reasoning_value": self.reasoning_value,
            "reasoning_echo": self.reasoning_echo,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "reasoning_metric": self.reasoning_metric,
            "allowance_status": self.allowance_status,
            "allowance_window": self.allowance_window,
            "allowance_utilization": self.allowance_utilization,
            "allowance_resets_at": self.allowance_resets_at,
        }

    @property
    def allowance_line(self) -> str | None:
        """The plan's position in words, or ``None`` when the run reported none.

        ``None`` is the honest answer for ``openai-sdk`` and for any run that
        ended before the runtime said anything about the allowance. It is not
        rendered as "0% used", which would be a claim rather than a silence.
        """
        if self.allowance_status is None and self.allowance_utilization is None:
            return None
        parts: list[str] = []
        if self.allowance_window:
            parts.append(str(self.allowance_window))
        if self.allowance_utilization is not None:
            parts.append(f"{self.allowance_utilization * 100:.0f}% used")
        if self.allowance_status:
            parts.append(str(self.allowance_status))
        if self.allowance_resets_at:
            parts.append(f"resets {self.allowance_resets_at}")
        return ", ".join(parts) if parts else None


def allowance_fields(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """The four ``allowance_*`` record fields out of a ``rate_limit`` payload.

    Reads defensively and drops anything it cannot type, for the same reason
    :func:`~modelpass.adapters.anthropic.rate_limit_info_fields` reads the SDK
    object defensively: the shape belongs to a vendor that may change it, and a
    field modelpass cannot vouch for is better absent than guessed. Every
    unreadable value becomes ``None``, never ``0`` or ``""``.
    """
    if not isinstance(data, Mapping):
        return {}
    out: dict[str, Any] = {}

    status = data.get("status")
    if isinstance(status, str) and status:
        out["allowance_status"] = status

    window = data.get("rate_limit_type")
    if isinstance(window, str) and window:
        out["allowance_window"] = window

    utilization = data.get("utilization")
    if isinstance(utilization, (int, float)) and not isinstance(utilization, bool):
        out["allowance_utilization"] = float(utilization)

    resets_at = data.get("resets_at")
    if isinstance(resets_at, str) and resets_at:
        # Already a string: kept verbatim. Parsing it to re-emit it would be
        # modelpass reinterpreting a vendor value it did not have to touch.
        out["allowance_resets_at"] = resets_at
    elif isinstance(resets_at, (int, float)) and not isinstance(resets_at, bool):
        try:
            out["allowance_resets_at"] = (
                datetime.fromtimestamp(float(resets_at), UTC).isoformat(timespec="seconds")
            )
        except (OverflowError, OSError, ValueError):
            # An epoch the platform cannot represent. The rest of the record is
            # still worth writing, so this one field simply does not appear.
            pass

    return out


class RunLog:
    """Append-only run receipts at ``<home>/runs.jsonl``.

    ``enabled=False`` makes every write a no-op while leaving reads working, so
    a user who switches the log off can still read what was recorded before.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        enabled: bool = True,
        filename: str = RUNS_FILENAME,
    ) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.filename = filename

    @property
    def path(self) -> Path:
        return self.root / self.filename

    def exists(self) -> bool:
        return self.path.is_file()

    def append(self, record: RunRecord) -> bool:
        """Append one record. Returns whether it was written; never raises.

        The bare ``except Exception`` is the point of this method rather than a
        shortcut in it. Everything reachable here -- disk full, read-only home,
        a directory where the file should be, a JSON encoder meeting something
        exotic -- has the same correct handling: lose the line, keep the run.
        The caller is mid-stream on a subscription the user is paying for.
        """
        if not self.enabled:
            return False
        try:
            line = json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=False)
            self.root.mkdir(parents=True, exist_ok=True)
            # One open-append-close per run, unbuffered by any handle we hold:
            # nothing here should keep a file descriptor alive across runs, and
            # a crashed process must not be able to lose a line it already had.
            #
            # **Serialized across threads** (R7, ticket 1.12). One ``Bridge`` is
            # documented as safe to share, so N workers on one bridge reach this
            # method at once, and on Windows two processes opening the same file
            # for append at the same moment is a sharing violation rather than
            # an interleaved write -- which ``append`` would swallow, losing a
            # line from the ledger that exists to prove what was spent. A
            # process-wide lock costs nothing here (the critical section is one
            # short write) and it is *not* a cross-process lock: a second
            # process writing the same file still relies on O_APPEND, which is
            # the pre-existing contract and unchanged.
            with _APPEND_LOCK:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
        except Exception:
            return False
        return True

    def read(self, limit: int | None = DEFAULT_READ_LIMIT) -> tuple[dict[str, Any], ...]:
        """Recent records, newest first, normalized. Never raises.

        Two kinds of damage are tolerated rather than fatal, because this file
        is append-only, hand-inspectable and explicitly hand-editable:

        * **A malformed line** -- a half-written last line after a power cut, or
          one somebody edited -- is skipped. One bad line must not hide the rest
          of the history.
        * **A well-formed record with the wrong field types** is repaired on
          read, not trusted. A hand-edited ``"total_tokens": "12"`` is still a
          readable record; letting a string reach a caller that formats it as a
          number would let one edited line take down the whole ledger view.
        """
        lines = self._tail(limit)
        out: list[dict[str, Any]] = []
        for line in reversed(lines):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                out.append(_normalize(parsed))
        return tuple(out)

    def _tail(self, limit: int | None) -> list[str]:
        """The last ``limit`` lines, read from the end of the file.

        The log is append-only and never rotated, so reading the whole thing to
        show the most recent fifty is a cost that grows without bound on exactly
        the machine that uses modelpass most. Reading backwards in blocks keeps a
        page load proportional to what it displays.
        """
        if limit is not None and limit <= 0:
            return []
        try:
            with self.path.open("rb") as handle:
                if limit is None:
                    return handle.read().decode("utf-8", "replace").splitlines()
                handle.seek(0, os.SEEK_END)
                end = handle.tell()
                block = 8192
                chunks: list[bytes] = []
                found = 0
                while end > 0 and found <= limit:
                    size = min(block, end)
                    end -= size
                    handle.seek(end)
                    chunk = handle.read(size)
                    chunks.append(chunk)
                    found += chunk.count(b"\n")
                data = b"".join(reversed(chunks))
        except Exception:
            return []
        return data.decode("utf-8", "replace").splitlines()[-limit:]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.read(limit=None))


#: Record fields that must be integers for a reader to be able to add or format
#: them, with the value substituted when the file says otherwise.
_INT_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "total_tokens",
    # Additive (R3, 2026-09-13). A line written before these existed has neither
    # key, and the coercion below fills both with 0 -- which is not a guess: a
    # run whose content was all plain strings carried no breakpoints. That is
    # what makes them safe to add to an append-only file nobody rewrites.
    "cache_breakpoints_requested",
    "cache_breakpoints_honoured",
)


def _normalize(record: dict[str, Any]) -> dict[str, Any]:
    """Coerce a parsed record into the types a reader may rely on.

    Deliberately not validation: an unrecognized field is kept as-is, because a
    line written by a newer modelpass is still a record and dropping what we do
    not understand would make the log less useful over time, not more.
    """
    out = dict(record)
    for field_name in _INT_FIELDS:
        value = out.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            try:
                out[field_name] = int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                out[field_name] = 0
    # Additive (R5, 2026-09-13), coerced toward *empty* rather than toward
    # ``None``: an absent key and a run that asked for nothing are the same
    # fact, unlike the allowance fields below where they are not. A hand-edited
    # non-mapping is repaired the way the token counts are.
    for field_name in ("sampling_requested", "sampling_applied"):
        value = out.get(field_name)
        out[field_name] = dict(value) if isinstance(value, dict) else {}
    notes = out.get("sampling_notes")
    out["sampling_notes"] = (
        [str(note) for note in notes] if isinstance(notes, list) else []
    )
    if not isinstance(out.get("guards_configured"), bool):
        out["guards_configured"] = bool(out.get("guards_configured"))
    for field_name in ("timestamp", "connection", "runtime", "auth_mode", "status"):
        value = out.get(field_name)
        out[field_name] = value if isinstance(value, str) else ""
    model = out.get("model")
    out["model"] = model if isinstance(model, str) else None
    origin = out.get("failed_over_from")
    out["failed_over_from"] = origin if isinstance(origin, str) else None
    # Toward None, like ``model``: a line written before this field existed says
    # nothing about which binary ran, and "" would read as an answer.
    binary = out.get("binary")
    out["binary"] = binary if isinstance(binary, str) else None
    # The allowance fields normalize toward ``None`` rather than toward a zero
    # value, which is the whole distinction they exist to hold: a line written
    # before these fields existed, a Codex run that reports no allowance, and a
    # run the vendor said was 0% used are three different facts, and only the
    # third one is a number. A hand-edited "0.42" is still readable and is
    # repaired the way the token counts are.
    for field_name in ("allowance_status", "allowance_window", "allowance_resets_at"):
        value = out.get(field_name)
        out[field_name] = value if isinstance(value, str) and value else None
    # The reasoning fields normalize toward ``None`` for the reason the
    # allowance ones do, and one of them harder than the rest: a line written
    # before these keys existed says nothing about what the run reasoned, and a
    # ``0`` there would claim the model thought for no tokens. A missing
    # ``reasoning_metric`` is left as ``None`` rather than guessed at from the
    # runtime, because the word describes a *run* and an old line was never
    # measured.
    for field_name in ("reasoning_value", "reasoning_echo", "reasoning_metric"):
        value = out.get(field_name)
        out[field_name] = value if isinstance(value, str) and value else None
    reasoning_tokens = out.get("reasoning_output_tokens")
    if isinstance(reasoning_tokens, bool) or not isinstance(reasoning_tokens, int):
        out["reasoning_output_tokens"] = None
    utilization = out.get("allowance_utilization")
    if isinstance(utilization, bool) or not isinstance(utilization, (int, float)):
        try:
            out["allowance_utilization"] = float(utilization)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            out["allowance_utilization"] = None
    else:
        out["allowance_utilization"] = float(utilization)
    return out


def run_log_for(store: ConnectionStore) -> RunLog:
    """The run log for a store's home, honouring ``[settings] runLog``.

    A broken or unreadable settings table leaves the log *enabled*: the failure
    mode to avoid is a config problem silently switching off the evidence.
    """
    try:
        enabled = store.settings().run_log
    except Exception:
        enabled = True
    return RunLog(store.root, enabled=enabled)
