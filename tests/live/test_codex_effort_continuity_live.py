"""Characterize whether changing effort mid-thread keeps Codex's cached prefix.

**Why this is a live test and not a fake one.** Everything a fake can establish
here is already established: `openai-codex` 0.154.0 types
`ConfigurationUpdateResponseItem`, the app-server takes `effort` on
`TurnStartParams`, and the cell in `modelpass.prompt_cache` reads
`EXPERIMENTAL` on exactly that evidence. What no fake can answer is the only
question worth asking -- whether a real server, handed a new effort level on a
thread with a large cached prefix, reuses that prefix or starts it over. A
scripted transport would answer whatever it was scripted to.

**What moves the cell.** A run of this file is a characterization, not a
promise. A `True` here on one model and one build moves `openai-sdk` +
`gpt-6-astra` from `EXPERIMENTAL` no further than `EXPERIMENTAL` with a dated
observation beside it; only a vendor statement makes it `SUPPORTED`. A `False`
is worth more: it is direct evidence for `UNSUPPORTED`, and one clean negative
settles more than three ambiguous positives.

**It spends allowance and the owner runs it** (AGENTS.md). Gated on
`MODELPASS_LIVE_CODEX_CONTINUITY=1` and skipped otherwise, because 30k-50k of
prefix across four turns is not something to run by accident.

Set:

* `MODELPASS_LIVE_CODEX_CONTINUITY=1` -- the opt-in.
* `MODELPASS_LIVE_CODEX_CONNECTION` -- a configured `openai-sdk` connection.
* `MODELPASS_LIVE_CODEX_MODEL` -- defaults to `gpt-6-astra`, the only model any
  vendor documents this mechanism for.

Read the printed table even when the assertions pass. The numbers are the
result; the assertion is only the coarsest reading of them.
"""

from __future__ import annotations

import os

import pytest

from modelpass import Bridge
from modelpass.prompt_cache import effort_cache_continuity, observed_effort_continuity
from modelpass.runtimes import Runtime

pytestmark = pytest.mark.live

_OPT_IN = "MODELPASS_LIVE_CODEX_CONTINUITY"
_CONNECTION = "MODELPASS_LIVE_CODEX_CONNECTION"
_MODEL = "MODELPASS_LIVE_CODEX_MODEL"

#: Big enough that a cache break is unmistakable in the counts. The ticket asks
#: for 30k-50k; this builds it from one stable block repeated, because what has
#: to be *identical* across turns is the prefix, and text assembled from a
#: generator is easier to keep byte-identical than prose.
_BLOCK = (
    "Section {n}. The following inventory line is stable across every turn of "
    "this thread and exists only to occupy prompt tokens: widget-{n:04d}, "
    "quantity {n}, warehouse {n} of the northern region, last audited in the "
    "fiscal quarter beginning on the first day of the month.\n"
)


def _prefix(sections: int = 900) -> str:
    return "".join(_BLOCK.format(n=n) for n in range(sections))


def _skip_unless_opted_in() -> tuple[str, str]:
    if os.environ.get(_OPT_IN) != "1":
        pytest.skip(f"set {_OPT_IN}=1 to run the effort/cache characterization")
    connection = os.environ.get(_CONNECTION)
    if not connection:
        pytest.skip(f"set {_CONNECTION} to a configured openai-sdk connection")
    return connection, os.environ.get(_MODEL) or "gpt-6-astra"


def _counts(events) -> dict[str, int | None]:
    """The six numbers the ticket asks to record, off one turn's terminal event.

    A turn is a stream, so the counts come from the terminal rather than from a
    returned object -- and from the terminal specifically, because that is the
    event modelpass stamps and an adapter cannot forge (D3).
    """
    from modelpass.types import TerminalEvent

    terminal = next(e for e in events if isinstance(e, TerminalEvent))
    usage = terminal.usage
    return {
        "input": usage.input_tokens,
        "cached_input": usage.cached_input_tokens,
        "cache_write": usage.cache_write_tokens,
        "prompt": usage.prompt_tokens,
        "output": usage.output_tokens,
        "reasoning_output": usage.reasoning_output_tokens,
    }


def _render(rows: list[tuple[str, dict[str, int | None]]]) -> str:
    head = (
        f"{'turn':<22}{'input':>10}{'cached':>10}{'write':>10}"
        f"{'prompt':>10}{'out':>8}{'reason':>8}"
    )
    lines = [head, "-" * len(head)]
    for label, c in rows:
        lines.append(
            f"{label:<22}{c['input']!s:>10}{c['cached_input']!s:>10}"
            f"{c['cache_write']!s:>10}{c['prompt']!s:>10}{c['output']!s:>8}"
            f"{c['reasoning_output']!s:>8}"
        )
    return "\n".join(lines)


def test_the_seam_this_characterization_needs_does_not_exist_yet():
    """**Run this first. It is the finding, not a placeholder.**

    The sequence the ticket describes -- build a 30k-50k prefix at ``medium``,
    baseline it, then change to ``high`` on the same thread -- cannot be
    expressed against modelpass today, and the reason is two specific gaps
    rather than a missing convenience:

    1. ``ChatSession.send`` takes ``message`` and ``timeout`` and nothing else.
       Effort comes from the connection, which is frozen for the session's life,
       so there is no turn at which a different level could be stated.
    2. ``_session_turn_start_params`` does not send ``effort`` at all. The
       stateless path does; the session path never did. So even a session whose
       connection changed underneath it would transmit nothing.

    Codex's own ``TurnStartParams.effort`` is documented as *"Override the
    reasoning effort for this turn and subsequent turns"* -- a per-turn field,
    which is exactly the lever this needs and is a vendor-typed one rather than
    a synthesized protocol message. Wiring it is a small change to a method
    whose docstring is a deliberate caching contract (``send`` excludes
    ``tools=`` and ``system_prompt=`` precisely because they move the prefix),
    so it is a decision rather than a fix.

    Until then this file establishes the *baseline* below, which is the half of
    the experiment that does not need the seam and is worth having ready.
    """
    import inspect

    from modelpass.sessions import ChatSession

    assert "reasoning" not in inspect.signature(ChatSession.send).parameters, (
        "ChatSession.send now takes a reasoning override -- the seam exists, so "
        "delete this test and enable the transition half of "
        "test_effort_change_within_one_thread"
    )


def test_baseline_cache_behaviour_of_a_persistent_thread(capsys):
    """What a normal cached turn looks like, at an unchanged effort.

    **This is not optional scaffolding for the experiment; it is half of it.**
    Without it, a large cached read after an effort change proves nothing --
    it could be the prefix any second turn would have hit. What a transition
    has to beat is this number, not zero.

    Run it now, keep the table, and compare against it when the seam lands.
    """
    connection, model = _skip_unless_opted_in()

    declared = effort_cache_continuity(Runtime.OPENAI_SDK, model)
    print(
        f"declared capability: {declared.capability.value} "
        f"({declared.mechanism.value if declared.mechanism else 'no mechanism'}), "
        f"reachable={declared.reachable}"
    )
    print(declared.detail)

    bridge = Bridge()
    rows: list[tuple[str, dict[str, int | None]]] = []
    session = bridge.new_chat(connection=connection, model=model)
    try:
        rows.append(
            (
                "1 build prefix",
                _counts(
                    session.send(_prefix() + "Reply with the single word: ready.")
                ),
            )
        )
        rows.append(
            (
                "2 baseline",
                _counts(session.send("Reply with the single word: still.")),
            )
        )
        rows.append(
            (
                "3 baseline",
                _counts(session.send("Reply with the single word: again.")),
            )
        )
    finally:
        session.close()

    print(_render(rows))
    baseline = rows[1][1]
    print(
        "baseline cached fraction: "
        f"{(baseline['cached_input'] or 0) / max(baseline['prompt'] or 1, 1):.3f}"
    )

    assert baseline["cached_input"], (
        "this thread cached nothing on turn 2, so it cannot serve as a baseline "
        "for an effort transition. Check the prefix is byte-identical across "
        "turns and large enough to clear the runtime's minimum cacheable prefix "
        "before concluding anything about effort"
    )
    # Deliberately not an assertion about the effort change: nothing changed
    # effort here, so observed_effort_continuity is asked with effort_changed
    # False and must answer None. Holding that keeps the capability from ever
    # leaking into the telemetry column by way of this file.
    assert (
        observed_effort_continuity(
            effort_changed=False,
            cached_input_tokens=baseline["cached_input"],
            prompt_tokens=baseline["prompt"],
        )
        is None
    )
