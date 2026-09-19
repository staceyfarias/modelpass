#!/usr/bin/env python3
"""The inverted agent loop: your Python functions, the runtime's loop (D12).

The interesting property is not that the model can call a tool -- every vendor
offers that. It is *where the tool runs*. ``inventory`` below is an ordinary
dict living in this process, and the handler closes over it. The model asks for
it, the Claude runtime calls back into this process, the handler answers, and
the runtime keeps going -- all inside **one** ``bridge.chat()`` call, on the
subscription you already pay for.

The alternative shape -- surface each tool call to the caller, run it, start a
fresh request with the result appended -- pays a full stateless run per round
trip. That is why modelpass inverted it (D12) and why the caller-side loop is a
deferred compatibility mode rather than the default.

    pip install -e ".[anthropic]"
    python examples/caller_tools.py

Requires a Claude Code login (``claude /login``, or ``claude setup-token`` for
headless use). Costs a fraction of a cent of your Agent SDK allowance --
somewhat more than the plain smoke test, because a tool loop takes several
model turns.

Like the other examples this is **not** part of the test suite: it spends real
allowance and needs a real login. ``pytest`` never runs it.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import modelpass
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore
from modelpass.tools import ToolDef
from modelpass.types import AuthMode

# Process-local state the model has no other way to reach. If the answer below
# mentions 7 mixers, the handler really ran here.
INVENTORY = {"mixer": 7, "cable": 42, "microphone": 3}
CALLS: list[dict] = []


def check_stock(args: dict) -> str:
    """Ordinary Python. No subprocess, no serialization, no MCP boilerplate."""
    CALLS.append(args)
    item = str(args.get("item", "")).lower()
    if item not in INVENTORY:
        # Returning the failure as text lets the model recover -- it can ask
        # about a different item instead of the run dying. Raising would work
        # too: modelpass turns an exception into a failed tool result.
        return f"No item named {item!r}. Known items: {', '.join(sorted(INVENTORY))}."
    return f"{INVENTORY[item]} units of {item} in stock."


STOCK_TOOL = ToolDef(
    name="check_stock",
    # The description is what the model reads to decide whether to call this at
    # all, so it is part of the prompt, not documentation.
    description="Look up how many units of an item are currently in stock.",
    parameters={
        "type": "object",
        "properties": {
            "item": {
                "type": "string",
                "description": "Item name, e.g. 'mixer', 'cable', 'microphone'.",
            }
        },
        "required": ["item"],
    },
    handler=check_stock,
)


def build_bridge(connection_name: str | None) -> tuple[modelpass.Bridge, str]:
    """Either the user's real store, or a throwaway one holding a demo connection."""
    if connection_name:
        return modelpass.Bridge(), connection_name

    root = Path(tempfile.mkdtemp(prefix="modelpass-tools-"))
    store = ConnectionStore(root)
    store.add(
        Connection(
            name="claude-sub",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            credential_ref=CredentialRef.native_login(),
            # A tool loop is exactly the shape that can run away, so a real
            # application should keep a stopAt here. Disabled in the demo only
            # because the prompt is tiny and a surprise stop would be confusing.
            guards=Guards(warn_at_tokens=0, stop_at_tokens=0),
            description="caller-tools example (temporary store)",
        )
    )
    return modelpass.Bridge(store=store), "claude-sub"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection",
        help="a connection from your real ~/.modelpass store (default: a temporary one)",
    )
    parser.add_argument(
        "--prompt",
        default="How many mixers and how many cables do we have? Answer in one sentence.",
        help="what to ask (keep it tiny: this spends your allowance)",
    )
    args = parser.parse_args()

    bridge, name = build_bridge(args.connection)

    # Ask the registry before spending anything. It answers honestly per
    # runtime: Codex, for instance, reports tools_in_process as unsupported.
    support = bridge.registry.support(
        bridge.connection(name).runtime, modelpass.Capability.TOOLS_IN_PROCESS
    )
    print(f"=== capability ===\nin-process tools: {support.value}")
    if note := bridge.registry.note(
        bridge.connection(name).runtime, modelpass.Capability.TOOLS_IN_PROCESS
    ):
        print(f"  note: {note}")

    print("\n=== preflight receipt ===")
    receipt = bridge.preflight(name)
    print(receipt.summary())
    if not receipt.ok:
        print(f"\nPreflight failed: {receipt.problem}", file=sys.stderr)
        return 1

    print("\n=== run ===")
    terminal = None
    for event in bridge.chat(
        connection=name,
        message=args.prompt,
        expect_auth_mode=AuthMode.SUBSCRIPTION,
        tools=[STOCK_TOOL],
    ):
        if isinstance(event, modelpass.TextDeltaEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, modelpass.ToolCallEvent):
            # Observed, not answered: the runtime already ran it (D12).
            print(f"\n[tool_call] {event.server}/{event.name} {dict(event.arguments)}")
        elif isinstance(event, modelpass.ToolResultEvent):
            flag = " (error)" if event.is_error else ""
            print(f"[tool_result]{flag} {event.content}")
        elif isinstance(event, modelpass.UsageEvent):
            total = (event.cumulative or event.usage).total_tokens
            print(f"\n[usage] {total} tokens so far")
        elif isinstance(event, modelpass.TerminalEvent):
            terminal = event

    print(f"\n=== handler ran in this process {len(CALLS)} time(s): {CALLS} ===")

    if terminal is None:
        print("no terminal event: the stream ended abnormally", file=sys.stderr)
        return 1
    print(
        f"[terminal] status={terminal.status.value} "
        f"connection={terminal.connection} auth_mode={terminal.auth_mode.value}"
    )
    return 0 if terminal.status is modelpass.TerminalStatus.OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
