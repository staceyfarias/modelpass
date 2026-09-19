"""Guard evaluation (D4, offered layer).

The tracker is deliberately dumb: it folds in the token counts the runtime
reports and compares them to the connection's thresholds. That is the honest
limit of what a local library can do -- usage events are best-effort and the
vendor's bill is the truth (D4, disclaimed layer). What it buys is a normalized
``guard_warning`` / ``guard_stop`` vocabulary that a caller can act on without
knowing whose runtime produced it.

**No default thresholds** (D4 amendment, 2026-08-17). A tracker built from an
unconfigured :class:`~modelpass.connections.Guards` never fires, because modelpass
has no basis for a number and will not invent one. Its absence is surfaced on
the preflight receipt instead.

**Mid-run, where the runtime allows it.** Phase 6 removed the invented turn cap
from tool runs, which made "the guard fires at the end" a real gap: a tool loop
runs to completion and the guard learns about it afterwards. So the tracker
folds in *every* usage report as it arrives, and :meth:`observe` is written to be
safe when a runtime mixes increments and running totals -- see
:class:`~modelpass.types.UsageScope`. The granularity that buys is per-runtime and
is reported honestly by the capability registry (``interim_usage``): per
assistant turn on ``anthropic-sdk``, and after every model response on
``openai-sdk`` since its default transport became ``codex app-server``
(2026-08-31), which sends ``thread/tokenUsage/updated``. A run that opts out
with ``options={"transport": "exec"}`` goes back to end-of-run only -- the
``codex exec --json`` vocabulary carries no usage before ``turn.completed`` --
and its preflight receipt says so, because the row it contradicts describes the
default rather than that run.
"""

from __future__ import annotations

from .connections import Guards
from .types import GuardStopEvent, GuardWarningEvent, TokenUsage, UsageScope

__all__ = ["TOKENS_GUARD", "GuardTracker"]

TOKENS_GUARD = "tokens"


class GuardTracker:
    """Accumulates usage for one run and emits guard events on threshold crossings."""

    def __init__(self, guards: Guards, connection: str) -> None:
        self._guards = guards
        self._connection = connection
        self._total = TokenUsage()
        self._warned = False
        self._stopped = False

    @property
    def total(self) -> TokenUsage:
        return self._total

    @property
    def stopped(self) -> bool:
        return self._stopped

    def observe(
        self,
        usage: TokenUsage,
        scope: UsageScope = UsageScope.DELTA,
    ) -> list[GuardWarningEvent | GuardStopEvent]:
        """Fold in a usage report and return any guard events it triggers.

        ``scope`` decides how the report is folded in: a ``delta`` adds, a
        ``run_total`` replaces without ever lowering the figure. That is what
        lets a runtime emit per-turn usage *and* a final authoritative total for
        the same run without the guard counting the work twice.
        """
        if scope is UsageScope.RUN_TOTAL:
            self._total = self._total.at_least(usage)
        else:
            self._total = self._total + usage
        observed = self._total.total_tokens
        events: list[GuardWarningEvent | GuardStopEvent] = []

        warn_at = self._guards.warn_at_tokens
        stop_at = self._guards.stop_at_tokens

        if warn_at is not None and not self._warned and observed >= warn_at:
            self._warned = True
            events.append(
                GuardWarningEvent(
                    guard=TOKENS_GUARD,
                    threshold=warn_at,
                    observed=observed,
                    connection=self._connection,
                    message=(
                        f"{observed} tokens used on connection {self._connection!r}, "
                        f"past the warnAtTokens threshold of {warn_at}"
                    ),
                )
            )

        if stop_at is not None and not self._stopped and observed >= stop_at:
            self._stopped = True
            events.append(
                GuardStopEvent(
                    guard=TOKENS_GUARD,
                    threshold=stop_at,
                    observed=observed,
                    connection=self._connection,
                    message=(
                        f"{observed} tokens used on connection {self._connection!r}, "
                        f"past the stopAtTokens threshold of {stop_at}; stopping the run"
                    ),
                )
            )

        return events
