#!/usr/bin/env python3
"""The hello world: print the preflight receipt, then stream one generation.

This is the shortest complete demonstration of what modelpass is for -- you see
*which subscription is about to be spent* before a single token is, and the
stream is stamped with the connection and auth mode actually used.

    pip install -e ".[anthropic]"
    python examples/smoke_anthropic.py

By default it uses a throwaway connection store under a temp directory, so it
neither reads nor writes your real ``~/.modelpass``. Point it at your own config
with ``--connection NAME``, which uses the real store instead.

Requires a Claude Code login (``claude /login``, or ``claude setup-token`` for
headless use). Costs a fraction of a cent of your Agent SDK allowance.
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
from modelpass.types import AuthMode


def build_bridge(connection_name: str | None) -> tuple[modelpass.Bridge, str]:
    """Either the user's real store, or a throwaway one holding a demo connection."""
    if connection_name:
        return modelpass.Bridge(), connection_name

    root = Path(tempfile.mkdtemp(prefix="modelpass-smoke-"))
    store = ConnectionStore(root)
    store.add(
        Connection(
            name="claude-sub",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            credential_ref=CredentialRef.native_login(),
            guards=Guards(warn_at_tokens=0, stop_at_tokens=0),
            description="smoke test (temporary store)",
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
        default="Reply with exactly OK",
        help="what to ask (keep it tiny: this spends your allowance)",
    )
    args = parser.parse_args()

    bridge, name = build_bridge(args.connection)

    # 1. The receipt. This is the informed-consent step: it says which account
    #    and which billing mode, and what was removed from the child environment.
    print("=== preflight receipt ===")
    receipt = bridge.preflight(name)
    print(receipt.summary())
    for note in receipt.notes:
        print(f"  note: {note}")
    if not receipt.ok:
        print(f"\nPreflight failed: {receipt.problem}", file=sys.stderr)
        print(
            "modelpass refuses to run rather than fall through to metered billing.",
            file=sys.stderr,
        )
        return 1

    # 2. The run. `expect_auth_mode` asserts what we expect to be billed; the
    #    call raises rather than quietly running on the wrong one.
    print("\n=== generation ===")
    terminal = None
    for event in bridge.chat(
        connection=name,
        message=args.prompt,
        expect_auth_mode=AuthMode.SUBSCRIPTION,
    ):
        if isinstance(event, modelpass.TextDeltaEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, modelpass.ThinkingEvent):
            print(f"\n[thinking] {event.text}", flush=True)
        elif isinstance(event, modelpass.UsageEvent):
            usage = event.cumulative or event.usage
            print(
                f"\n\n[usage] in={usage.input_tokens} out={usage.output_tokens} "
                f"cached={usage.cached_input_tokens} total={usage.total_tokens}"
            )
        elif isinstance(event, modelpass.TerminalEvent):
            terminal = event

    if terminal is not None:
        print(
            f"[terminal] status={terminal.status.value} "
            f"connection={terminal.connection} auth_mode={terminal.auth_mode.value}"
        )
        if terminal.reason:
            print(f"[terminal] reason: {terminal.reason}")
        return 0 if terminal.status is modelpass.TerminalStatus.OK else 1

    print("no terminal event: the stream ended abnormally", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
