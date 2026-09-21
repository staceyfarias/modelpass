"""``modelpass`` -- the command line for setting connections up and checking them.

D2 calls setup "a verb with a receipt". This is that verb. Three subcommands,
each doing one thing:

* ``modelpass connect anthropic`` -- run the preflight, print what it actually
  found, and write the connection **only** once you have seen it.
* ``modelpass list`` -- what is configured, in one line each.
* ``modelpass check`` -- re-run the preflight against a configured connection and
  exit non-zero if it would not run.
* ``modelpass remove`` -- delete one, and say what happened to its stored key.
* ``modelpass secrets`` -- what is in the secrets file and which connections point
  at each entry. Names only, never a value.
* ``modelpass bench`` -- serve the same facts as a local page, plus a playground
  that runs a real call and shows the whole event stream (needs
  ``pip install modelpass[bench]``).

Three rules this module holds to:

1. **No secret is ever printed.** Everything here deals in *pointers*: the name
   of an environment variable, the path a login store lives at, an account
   identifier the vendor already showed the user. A value read from the
   environment is checked for presence and then dropped on the floor.
2. **Nothing is written before the receipt is shown.** ``connect`` prints the
   preflight and then asks. ``--yes`` skips the asking, not the printing.
3. **The absence of spend guards is loud.** modelpass ships no default thresholds
   (D4 amendment, 2026-08-17), so a fresh connection has nothing bounding what
   one run can spend, and a setup tool that did not say so would be leaving the
   user with a false impression of what they had just configured.

Written against stdlib ``argparse`` -- core takes no dependencies (D8) -- and
structured so it is testable offline: :func:`main` takes an injected
:class:`~modelpass.bridge.Bridge` (hence an injected store path and adapter
registry) and its own input/output streams, so the whole surface runs against
``FakeAdapter`` with no vendor package, no network and no credential.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import IO

from . import __version__
from .adapters.base import RunRequest
from .bridge import Bridge
from .connections import Connection, CredentialKind, Guards
from .errors import ConfigError, SubpassError
from .manage import Credential, binding_from_receipt, receipt_for
from .preflight import Receipt
from .prompt_cache import plan_prompt_cache
from .runtimes import VENDOR_OF, Runtime
from .store import ConnectionStore
from .types import AuthMode, Message, Role

__all__ = ["main"]

#: Friendly names for the runtimes a user can connect to today. The runtime
#: identity is what the registry keys on (D6), but "modelpass connect anthropic"
#: is what someone types, so the mapping is explicit rather than derived.
#:
#: The key is the **vendor** (:data:`~modelpass.runtimes.VENDOR_OF`), and the
#: value is the runtime that vendor means when nothing else is said. For the two
#: subscription vendors that is the agent runtime, which is the 1.3 default and
#: does not change. ``google`` points at ``google-api`` instead, because its two
#: agent runtimes are gated by D5 and there is no ``--experimental`` on this
#: subcommand to ungate them; ``compatible`` has exactly one runtime to point at.
RUNTIME_ALIASES: dict[str, Runtime] = {
    "anthropic": Runtime.ANTHROPIC_SDK,
    "openai": Runtime.OPENAI_SDK,
    "google": Runtime.GOOGLE_API,
    "compatible": Runtime.OPENAI_COMPATIBLE,
}

#: The metered, in-process runtime each vendor alias means when a key is handed
#: over on stdin. ``--api-key-stdin`` says "bill this to a key I am giving you",
#: and the runtime that does that without a vendor login is the API one --
#: a stored key cannot reach the child process an agent runtime launches, which
#: is why ``Connection`` refuses ``secret:`` there outright.
API_RUNTIME_ALIASES: dict[str, Runtime] = {
    "anthropic": Runtime.ANTHROPIC_API,
    "openai": Runtime.OPENAI_API,
    "google": Runtime.GOOGLE_API,
    "compatible": Runtime.OPENAI_COMPATIBLE,
}

#: Default connection names, so the common case is one word of typing. Every
#: runtime has one, because ``--runtime`` can now name any of them.
DEFAULT_NAMES: dict[Runtime, str] = {
    Runtime.ANTHROPIC_SDK: "claude-sub",
    Runtime.OPENAI_SDK: "codex-sub",
    Runtime.GOOGLE_CLI: "gemini-sub",
    Runtime.GOOGLE_SDK: "gemini-sdk",
    Runtime.ANTHROPIC_API: "claude-api",
    Runtime.OPENAI_API: "openai-api",
    Runtime.GOOGLE_API: "gemini-api",
    Runtime.OPENAI_COMPATIBLE: "compatible-api",
}

EXIT_OK = 0
EXIT_FAILED = 1

#: The width the receipt and advice blocks wrap to. Fixed rather than read from
#: the terminal: this output gets pasted into issues and logs, and a block that
#: reflows differently on every machine is harder to compare than one that does
#: not.
_WIDTH = 88

#: The bench's bind address, repeated here so ``--help`` can state it without
#: importing the bench (and therefore without needing Flask installed).
_BENCH_HOST = "127.0.0.1"


def invoked_as() -> str:
    """``subpass`` when that alias entry point was used, else ``modelpass``.

    Both console scripts run this same ``main``. Help text that named the other
    one would send a user to a command they did not type, so the prog name
    follows the invocation while the alias exists (removed in 0.3.0).
    """
    argv0 = sys.argv[0] if sys.argv else ""
    return "subpass" if Path(argv0).stem == "subpass" else "modelpass"


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog or invoked_as(),
        description=(
            "Pluggable access to the AI subscriptions you already pay for. "
            "A fresh install has no AI connectivity: configuring a connection "
            "is the informed-consent step."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"{parser.prog} {__version__}"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    connect = subcommands.add_parser(
        "connect",
        help="run the preflight for a runtime and write the connection",
        description=(
            "Runs the auth preflight, prints what it found, and writes the "
            "connection once you confirm. Nothing is written before the receipt "
            "is shown."
        ),
    )
    connect.add_argument("vendor", choices=sorted(RUNTIME_ALIASES))
    defaults = ", ".join(
        f"{alias}={DEFAULT_NAMES[runtime]}"
        for alias, runtime in sorted(RUNTIME_ALIASES.items())
    )
    connect.add_argument("--name", help=f"connection name (default: {defaults})")
    connect.add_argument(
        "--runtime",
        dest="runtime_name",
        metavar="NAME",
        help=(
            "the exact runtime to connect, which must belong to this vendor "
            "(anthropic: anthropic-sdk, anthropic-api). Omit it for the default: "
            "the vendor's agent runtime, or its API runtime with --api-key-stdin"
        ),
    )
    connect.add_argument(
        "--base-url",
        metavar="URL",
        help=(
            "the API endpoint this connection talks to. Required by "
            "openai-compatible, which names an API shape rather than a vendor and "
            "so has no default endpoint"
        ),
    )
    connect.add_argument(
        "--api-key-env",
        metavar="NAME",
        help=(
            "use metered API-key billing, reading the key from this environment "
            "variable. The variable NAME is stored; its value never is. Omit this "
            "for subscription mode, which uses the runtime's own login"
        ),
    )
    connect.add_argument(
        "--api-key-stdin",
        action="store_true",
        help=(
            "use metered API-key billing with a key read from standard input. The "
            "key is written to the modelpass secrets file and the connection stores "
            "only a pointer to it. There is deliberately no option that takes a key "
            "on the command line: argv is visible in process listings and lands in "
            "shell history"
        ),
    )
    connect.add_argument("--model", help="pin a model for this connection")
    connect.add_argument(
        "--nickname",
        help="human display name for this account, for example 'OpenAI Work'",
    )
    connect.add_argument("--description", help="a note to yourself, stored in the config")
    connect.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "put this connection in a group. Repeatable. A connection that names "
            "no group is in 'default'; a group needs no creating, it exists while "
            "somebody is in it"
        ),
    )
    connect.add_argument(
        "--config-dir",
        metavar="PATH",
        help=(
            "vendor configuration directory for this account. Maps to "
            "CLAUDE_CONFIG_DIR on Anthropic and CODEX_HOME on OpenAI"
        ),
    )
    connect.add_argument(
        "--warn-at-tokens",
        type=int,
        metavar="N",
        help="emit guard_warning once a run passes N tokens",
    )
    connect.add_argument(
        "--stop-at-tokens",
        type=int,
        metavar="N",
        help="stop a run once it passes N tokens",
    )
    connect.add_argument(
        "--allow-env",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "let a named non-credential variable through the scrub "
            "(anthropic: ANTHROPIC_BASE_URL). Repeatable"
        ),
    )
    connect.add_argument(
        "--force", action="store_true", help="overwrite an existing connection of this name"
    )
    connect.add_argument(
        "-y", "--yes", action="store_true", help="write it without asking to confirm"
    )

    listing = subcommands.add_parser(
        "list", help="show configured connections", description="One line per connection."
    )
    listing.add_argument("--verbose", "-v", action="store_true", help="include guard detail")
    listing.add_argument(
        "--group", metavar="NAME", help="show only the connections in this group"
    )

    accounts = subcommands.add_parser(
        "accounts",
        help="show configured account profiles (account-oriented alias for list)",
        description="One block per vendor account profile.",
    )
    accounts.add_argument("--verbose", "-v", action="store_true", help="include detail")
    accounts.add_argument(
        "--group", metavar="NAME", help="show only the accounts in this group"
    )

    bench = subcommands.add_parser(
        "bench",
        help="serve the local configuration and test page",
        description=(
            "Serves a local page for inspecting connections, running preflights, "
            "watching a real event stream and reading the run log. Binds "
            f"{_BENCH_HOST} only -- it is a development tool for this machine, not "
            "a service. Needs Flask: pip install modelpass[bench]"
        ),
    )
    bench.add_argument(
        "--port", type=int, default=8765, help="port to serve on (default: 8765)"
    )

    check = subcommands.add_parser(
        "check",
        help="re-run the preflight for a configured connection",
        description=(
            "Re-runs the preflight and prints the receipt. Exits non-zero if the "
            "connection would not run."
        ),
    )
    check.add_argument("name", nargs="?", help="connection to check (default: all of them)")

    verify = subcommands.add_parser(
        "verify",
        help="go and find out the live answer for a connection, and write it down",
        description=(
            "On a subscription account: runs a token-free identity preflight and "
            "stores the reported email and organization ID, so later runs fail if "
            "that identity changes. On an openai-compatible connection: drives the "
            "configured endpoint with one short chat, one tool round trip and one "
            "structured-output call, and records which capability cells actually "
            "worked -- that runtime's row describes an API shape, so only your "
            "endpoint can answer for it. The second form SPENDS: a few hundred "
            "tokens against whatever the base URL points at."
        ),
    )
    verify.add_argument("name", help="account profile or connection to verify")

    rename = subcommands.add_parser(
        "rename",
        help="give a connection a new name, moving its secret entry with it",
        description=(
            "Renames a connection. When its secret entry is named after the "
            "connection and nothing else references it, the entry moves too; "
            "otherwise it is left where it is and this says so."
        ),
    )
    rename.add_argument("old", help="the connection's current name")
    rename.add_argument("new", help="the name to give it")

    remove = subcommands.add_parser(
        "remove",
        help="delete a connection, applying the secret rule to its key",
        description=(
            "Deletes a connection. Its stored key goes with it only when the entry "
            "was unambiguously this connection's -- named after it, and referenced "
            "by nothing else; every other case leaves the key where it is and says "
            "why, because an orphaned secret is recoverable and a deleted one is "
            "not. Prints what is about to go and asks before writing."
        ),
    )
    remove.add_argument("name", help="connection to delete")
    remove.add_argument(
        "-y", "--yes", action="store_true", help="delete it without asking to confirm"
    )

    groups = subcommands.add_parser(
        "groups",
        help="show the groups and who is in them",
        description=(
            "One block per group, with its members and which of them a run could "
            "actually go to. Groups are not defined anywhere -- a group exists "
            "while some connection declares it, and 'default' is the name for the "
            "connections that declare none. Says which member "
            "'group:<name>' would resolve to, and why the others were passed over."
        ),
    )
    groups.add_argument(
        "name", nargs="?", help="a single group to describe (default: all of them)"
    )

    set_groups = subcommands.add_parser(
        "set-groups",
        help="replace the groups a connection is in",
        description=(
            "Replaces the whole list rather than adding to it, because 'put this "
            "in cheap' and 'put this in cheap only' are different instructions. "
            "With no group named, the connection returns to the default group."
        ),
    )
    set_groups.add_argument("name", help="connection to move")
    set_groups.add_argument(
        "groups", nargs="*", metavar="GROUP", help="the groups it should be in"
    )

    enable = subcommands.add_parser(
        "enable",
        help="put a connection back into service",
        description="Sets enabled = true. Runs on this connection are accepted again.",
    )
    enable.add_argument("name", help="connection to enable")

    disable = subcommands.add_parser(
        "disable",
        help="take a connection out of service without deleting it",
        description=(
            "Sets enabled = false. The connection stays listed and its preflight "
            "still runs -- this switches off spending, not looking -- and its "
            "guard thresholds survive, which is what deleting and retyping it "
            "would lose. A group skips it when resolving."
        ),
    )
    disable.add_argument("name", help="connection to disable")

    secrets = subcommands.add_parser(
        "secrets",
        help="list the stored key entries and which connections reference them",
        description=(
            "One line per entry in the secrets file: its name, and the connections "
            "pointing at it. Never a value -- there is no flag anywhere in this "
            "command line that prints one. An entry nothing references is listed as "
            "an orphan and is never deleted for you."
        ),
    )
    secrets.add_argument(
        "--verbose", "-v", action="store_true", help="include the file's permission note"
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    bridge: Bridge | None = None,
    out: IO[str] | None = None,
    err: IO[str] | None = None,
    confirm: Callable[[str], bool] | None = None,
    stdin: IO[str] | None = None,
) -> int:
    """Run one CLI invocation and return its exit code.

    Every collaborator is injectable so the whole surface is testable offline:
    ``bridge`` carries the store path and the adapter registry, ``confirm``
    stands in for the interactive prompt.
    """
    args = build_parser().parse_args(argv)
    stdout = out if out is not None else sys.stdout
    stderr = err if err is not None else sys.stderr
    active = bridge if bridge is not None else Bridge()
    ask = confirm if confirm is not None else _prompt
    source = stdin if stdin is not None else sys.stdin

    try:
        if args.command == "connect":
            return _connect(args, active, stdout, ask, source)
        if args.command in {"list", "accounts"}:
            return _list(args, active, stdout)
        if args.command == "bench":
            return _bench(args, active, stdout)
        if args.command == "verify":
            return _verify(args.name, active, stdout)
        if args.command == "rename":
            return _rename(args.old, args.new, active, stdout)
        if args.command == "remove":
            return _remove(args, active, stdout, ask)
        if args.command == "groups":
            return _groups(args, active, stdout)
        if args.command == "set-groups":
            return _set_groups(args, active, stdout)
        if args.command in {"enable", "disable"}:
            return _set_enabled(args, active, stdout)
        if args.command == "secrets":
            return _secrets(args, active, stdout)
        return _check(args, active, stdout)
    except SubpassError as exc:
        # The taxonomy exists so a message is actionable; printing it beats a
        # traceback for the audience this command line is written for.
        print(f"modelpass: {exc}", file=stderr)
        return EXIT_FAILED


# --- connect ---------------------------------------------------------------------


def _connect(
    args: argparse.Namespace, bridge: Bridge, out: IO[str], ask, stdin: IO[str]
) -> int:
    """Plan, print, ask, write -- with the library doing the planning and writing.

    Every rule this subcommand holds to lives in
    :class:`~modelpass.manage.ConnectionManager` as of ticket 1.4, so a Settings
    page in a desktop app gets the same ones: the receipt is taken against the
    connection as it *would* be written, a pasted key is checked in memory and
    not on disk, and the secret is written before the connection. What stays
    here is the part that is genuinely a command line -- reading the key from a
    pipe, printing the receipt, and asking.
    """
    if args.api_key_stdin and args.api_key_env:
        raise ConfigError(
            "--api-key-stdin and --api-key-env name two different places for one "
            "credential; pick one"
        )

    runtime = _resolve_runtime(args)
    name = args.name or DEFAULT_NAMES[runtime]
    if args.api_key_stdin:
        # The entry is named after the connection -- Credential.secret defaults
        # it that way -- which is what makes the rename and delete rules able to
        # tell "this connection's own key" from "a key two connections share".
        credential = Credential.secret(_read_key_from_stdin(stdin))
    elif args.api_key_env:
        credential = Credential.env(args.api_key_env)
    elif runtime is Runtime.OPENAI_COMPATIBLE:
        # The one runtime where naming no credential is an answer rather than an
        # omission (ticket 1.10). Everywhere else a bare ``connect`` means "use
        # the runtime's own login", which is the subscription path; here there is
        # no login to use and the endpoint very often checks nothing, so the
        # honest default is to say so. ``--api-key-env`` and ``--api-key-stdin``
        # above still work exactly as they do on every other runtime, for the
        # gateway and proxy cases that do want a key.
        credential = Credential.none()
    else:
        credential = Credential.native_login()

    if bridge.store.has(name) and not args.force:
        print(
            f"Connection {name!r} already exists. Re-run with --force to replace it,\n"
            f"or pass --name to add a second one.",
            file=out,
        )
        return EXIT_FAILED

    plan = bridge.manage.plan_connection(
        name=name,
        runtime=runtime,
        credential=credential,
        # Strict: this is about to be written to the user's config file, where a
        # warn above a stop is a typo they would otherwise trust.
        warn_at_tokens=args.warn_at_tokens,
        stop_at_tokens=args.stop_at_tokens,
        model=args.model,
        base_url=args.base_url,
        description=args.description,
        nickname=args.nickname,
        config_dir=args.config_dir,
        allow_env=tuple(args.allow_env),
        groups=tuple(args.group),
        # The existence gate above is this command's own, with its own wording
        # and its own --force flag; by here the decision has been made.
        overwrite=True,
    )
    _print_receipt(plan.receipt, out)

    if not plan.ok:
        print(
            "\nNot writing the connection: the preflight could not confirm it is safe\n"
            "to run. Fix the problem above and try again.",
            file=out,
        )
        return EXIT_FAILED

    if plan.account_pinned is False:
        # A missing profile is not a broken account, so this is said rather than
        # enforced -- see ConnectionManager.plan_connection.
        print(
            f"\n  ! Saving {name!r} unpinned: the vendor reported no email or "
            f"organization ID. Run 'modelpass verify {name}' to pin it later.",
            file=out,
        )

    if not args.yes and not ask(f"\nSave this connection as {name!r}? [y/N] "):
        print("Nothing was written.", file=out)
        return EXIT_FAILED

    written = bridge.manage.add_connection(plan)
    if written.secret_entry is not None:
        print(
            f"\nWrote secret entry {written.secret_entry!r} to {written.secret_path}",
            file=out,
        )
        if written.secret_permissions is not None:
            _print_field("permissions", written.secret_permissions.note, out)
    print(f"\nWrote {written.connection!r} to {written.path}", file=out)
    if not written.runtime_available:
        # Said at the moment of setup rather than at the moment of the first
        # failed run. The preflight above is a real OK -- everything it checks
        # is checkable offline and it all passed -- but a connection that
        # nothing can drive yet is not what "OK" means to the person reading it.
        print("", file=out)
        print(
            f"  ! There is no adapter for runtime {plan.connection.runtime.value!r} in "
            "this build, so",
            file=out,
        )
        print(
            "    this connection can be inspected and checked, and not yet run.",
            file=out,
        )
    if not written.guards_configured:
        _print_guard_advice(args.vendor, name, bridge.store, out)
    return EXIT_OK


def _resolve_runtime(args: argparse.Namespace) -> Runtime:
    """Which runtime ``connect <vendor>`` means, with ``--runtime`` overriding.

    The defaults are the ones ticket 1.3 settled and they are unchanged: a bare
    ``connect anthropic`` is the subscription, and ``--api-key-stdin`` moves to
    the vendor's API runtime because a stored key cannot reach the child process
    an agent runtime launches. ``--api-key-env`` still means metered billing *on
    the default runtime* -- an environment variable does reach a child, so that
    combination was never broken and changing it would rewrite what existing
    users' commands do.

    What was missing is the third cell: an API runtime reading a named variable.
    ``--runtime`` is how that is asked for, and the only thing it is allowed to
    name is a runtime of the vendor already on the command line -- naming another
    vendor's runtime is a typo, not a shortcut, and this says which ones fit.
    """
    if not args.runtime_name:
        return (
            API_RUNTIME_ALIASES[args.vendor]
            if args.api_key_stdin
            else RUNTIME_ALIASES[args.vendor]
        )
    valid = tuple(r for r in Runtime if VENDOR_OF[r] == args.vendor)
    for runtime in valid:
        if runtime.value == args.runtime_name:
            return runtime
    raise ConfigError(
        f"--runtime {args.runtime_name!r} is not a runtime of vendor "
        f"{args.vendor!r}; valid runtimes for {args.vendor}: "
        f"{', '.join(r.value for r in valid)}"
    )


def _read_key_from_stdin(stdin: IO[str]) -> str:
    """Read one API key from standard input, and refuse everything else.

    Three rules, each answering a way keys get leaked:

    * **Never argv.** There is no option anywhere in this command line that
      takes a key as a value, because a command line is visible in ``ps``, is
      written to shell history, and is echoed by CI logs.
    * **Never a prompt.** If stdin is a terminal this refuses and shows how to
      pipe instead, rather than reading a typed key that the terminal has
      already put on screen and the shell may have kept.
    * **One trailing newline, and nothing else.** ``echo``, a here-string and a
      password manager's ``--raw`` all append exactly one. Stripping more than
      that would silently accept -- and silently alter -- a value whose
      whitespace is real.
    """
    if hasattr(stdin, "isatty") and stdin.isatty():
        raise ConfigError(
            "--api-key-stdin will not read from a terminal: a key typed at a prompt "
            "is on your screen and may be in your shell history. Pipe it in "
            "instead, for example:  cat key.txt | modelpass connect anthropic "
            "--api-key-stdin"
        )
    value = stdin.read()
    if value.endswith("\n"):
        value = value[:-1]
    if value.endswith("\r"):
        value = value[:-1]
    if not value.strip():
        raise ConfigError(
            "--api-key-stdin read an empty value from standard input; nothing was "
            "written"
        )
    return value


def _rename(old: str, new: str, bridge: Bridge, out: IO[str]) -> int:
    """Rename a connection, and move its secret entry when it is unambiguously its."""
    result = bridge.manage.rename_connection(old, new)
    print(f"Renamed {result.old!r} to {result.new!r} in {result.path}", file=out)
    if result.note:
        print(f"  {result.note}", file=out)
    return EXIT_OK


def _remove(args: argparse.Namespace, bridge: Bridge, out: IO[str], ask) -> int:
    """Delete a connection, after saying what goes with it.

    The bench has had a delete button since ticket 1.3 and the command line had
    nothing, so the only way to undo a ``connect`` was to edit the file by hand
    -- next to a secrets file the hand edit would leave behind. The rule about
    the key is not restated here: it lives in
    :meth:`~modelpass.manage.ConnectionManager.remove_connection` (ticket 1.4)
    and this prints the sentence that comes back.

    Printed before asking, like ``connect``: a user about to lose a pinned
    identity, a set of guard thresholds and possibly a stored key should be
    reading what those are, not remembering them.
    """
    connection = bridge.connection(args.name)
    print(f"About to delete {connection.display_name!r} from {bridge.store.path}", file=out)
    print(f"  runtime      {connection.runtime.value}", file=out)
    print(
        f"  auth         {connection.auth_mode.value} "
        f"({connection.credential_ref.describe()})",
        file=out,
    )
    print(f"  guards       {_guard_summary(connection.guards)}", file=out)
    if connection.credential_ref.kind is CredentialKind.SECRET:
        entry = connection.credential_ref.locator or ""
        others = bridge.secrets.referenced_by(
            entry, [c for c in bridge.connections() if c.name != connection.name]
        )
        if others:
            print(
                f"  key          entry {entry!r} stays: also referenced by "
                f"{', '.join(repr(name) for name in others)}",
                file=out,
            )
        else:
            print(f"  key          entry {entry!r} goes with it", file=out)

    if not args.yes and not ask(f"\nDelete {connection.name!r}? [y/N] "):
        print("Nothing was deleted.", file=out)
        return EXIT_FAILED

    result = bridge.manage.remove_connection(connection.name)
    print(f"\nDeleted {result.connection!r} from {bridge.store.path}", file=out)
    if result.note:
        print(f"  {result.note}", file=out)
    return EXIT_OK


def _secrets(args: argparse.Namespace, bridge: Bridge, out: IO[str]) -> int:
    """List the stored entries and who points at them. Never a value.

    ``check`` already reports the orphans as part of a health check; this is the
    other half of the question -- *what is in that file, and what is it for* --
    which previously had no answer short of opening a file full of keys in an
    editor. Entries are listed with their referencing connections, so the
    consequence of removing one is visible before anybody removes it.

    Degrades the way every read path in this library degrades (R10): no file is
    a legitimate state and says so, and a file this build could not read is
    reported as unreadable rather than rendered as an empty list.
    """
    secrets = bridge.secrets
    if not secrets.exists():
        print(
            f"No secrets file at {secrets.path}. Nothing is stored, which is the "
            "normal state\nfor an install whose connections use a subscription "
            "login or a named environment\nvariable.",
            file=out,
        )
        return EXIT_OK

    print(f"secrets file {secrets.path}", file=out)
    if args.verbose:
        _print_field("permissions", secrets.permissions().note, out)
    for note in secrets.notes():
        _print_field("note", note, out)
    entries = secrets.entries()
    if not entries:
        print("\nNo entries. The file exists and holds no keys.", file=out)
        return EXIT_OK

    connections = bridge.connections()
    print(f"\n{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}:\n", file=out)
    for entry in entries:
        holders = secrets.referenced_by(entry, connections)
        print(f"  {entry}", file=out)
        if holders:
            print(f"      used by  {', '.join(holders)}", file=out)
        else:
            print(
                "      used by  nothing -- an orphan. It is inert, it is listed "
                "here, and it is\n               never deleted for you.",
                file=out,
            )
    print(
        "\nNo value is printed by this command, and no flag makes it print one.",
        file=out,
    )
    return EXIT_OK


def _print_secret_notes(bridge: Bridge, out: IO[str]) -> None:
    """What ``modelpass check`` says about the secrets file, if there is one.

    Two facts, both of which are only useful said out loud. The permission note
    reports what was actually applied -- never what was intended. The orphan
    list is reported and never acted on: an unreferenced entry is inert, and
    deleting one on the user's behalf is how somebody loses a key they were
    about to point a connection at.
    """
    secrets = bridge.secrets
    if not secrets.exists():
        return
    print("", file=out)
    print("secrets", file=out)
    print(f"  file         {secrets.path}", file=out)
    _print_field("permissions", secrets.permissions().note, out)
    for note in secrets.notes():
        # A file this build could not read is reported here rather than raised:
        # the read degraded to "no secrets", and a user whose keys are invisible
        # must be told that rather than shown an empty list (R10).
        _print_field("note", note, out)
    orphans = secrets.orphans(bridge.connections())
    if orphans:
        _print_field(
            "orphans",
            ", ".join(orphans)
            + " -- no connection references these, so they are inert. They are "
            "listed, never deleted for you",
            out,
        )
    else:
        print("  orphans      none", file=out)


def _verify(name: str, bridge: Bridge, out: IO[str]) -> int:
    """Two verbs behind one word, and they are the same verb.

    On a subscription account, verify means *pin the vendor identity that is
    logged in now* -- a token-free probe, recorded on the connection, so a later
    run fails if the account drifts. On ``openai-compatible`` it means *drive
    this endpoint and record what it can do* -- three short calls, recorded on
    the connection, so a later run knows which capability cells are real. Both
    are "go and find out the live answer and write it down", which is why they
    share a subcommand rather than growing a second one nobody would find.

    Only these two. Every other API runtime has an identity that is a key
    fingerprint (nothing to pin) and a capability row that is a fact about a
    vendor (nothing an install could move), so there is nothing for this verb to
    do and it says so rather than doing something adjacent.
    """
    connection = bridge.connection(name)
    if connection.runtime is Runtime.OPENAI_COMPATIBLE:
        return _verify_capabilities(connection, bridge, out)
    if not connection.is_subscription:
        print(f"{name!r} uses an API key and has no subscription identity to pin.", file=out)
        return EXIT_FAILED
    # Deliberately clear the old pin for this one probe: verify is the explicit
    # re-binding action used after an intentional account change. And drop the
    # adapter's cached identity answer first -- a user who has just switched
    # accounts and run this must not be handed a minute-old probe.
    bridge.refresh_identity(connection)
    candidate = replace(connection, account_binding=None)
    receipt = bridge.preflight(candidate)
    _print_receipt(receipt, out)
    if not receipt.ok:
        return EXIT_FAILED
    binding = binding_from_receipt(receipt)
    if binding is None:
        print("Vendor did not report a stable identity; nothing was changed.", file=out)
        return EXIT_FAILED
    bridge.store.add(replace(connection, account_binding=binding), overwrite=True)
    print(f"\nVerified and pinned {connection.display_name!r}.", file=out)
    return EXIT_OK


def _verify_capabilities(connection: Connection, bridge: Bridge, out: IO[str]) -> int:
    """Drive an ``openai-compatible`` endpoint and record what answered.

    **The only command in modelpass that spends in order to learn something**, and
    it says so before it does. Three short calls -- a chat, a tool round trip, a
    structured-output request, sixty-odd tokens apiece -- against the endpoint the
    connection names. Against a local Ollama that costs nothing but electricity;
    against a metered gateway it costs a fraction of a cent, which is still a
    real number and is why this is a verb the user types rather than something
    the preflight does on their behalf.

    Nothing is written unless the drive reached the endpoint at all: a box that
    is switched off must not leave behind a record saying it can do nothing.
    """
    print(f"Verifying {connection.display_name!r} against {connection.base_url}", file=out)
    print(
        "  Three short calls -- one chat, one tool round trip, one structured-output\n"
        "  request. Against a metered endpoint this spends a few hundred tokens.\n",
        file=out,
    )
    adapter = bridge.adapter_for(connection.runtime)
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="ok"),),
        plan=bridge.plan(connection),
    )
    report = adapter.verify_capabilities(request)
    if report is None:
        # Reachable only if an adapter for this runtime is injected that does not
        # implement the hook. Said rather than crashed: the contract's default is
        # None and this command is the one caller that needs a real answer.
        print(
            f"The adapter for runtime {connection.runtime.value!r} does not implement a "
            "capability drive, so there is nothing to verify.",
            file=out,
        )
        return EXIT_FAILED
    for note in report.notes:
        _print_field("note", note, out)
    if not report.ok:
        print(f"\n  status       FAILED -- {report.problem}", file=out)
        print("Nothing was written.", file=out)
        return EXIT_FAILED
    bridge.store.add(
        replace(connection, verified_capabilities=report.verified), overwrite=True
    )
    verified = report.verified
    print("", file=out)
    print(f"  supported    {', '.join(verified.supported) or 'nothing'}", file=out)
    print(f"  unsupported  {', '.join(verified.unsupported) or 'nothing'}", file=out)
    print(f"  checked      {verified.checked_at}", file=out)
    print(f"\nRecorded on {connection.name!r} in {bridge.store.path}", file=out)
    print(
        "Cells this drive could not decide -- thinking, interim usage, sessions --\n"
        "still read as the capability table left them.",
        file=out,
    )
    return EXIT_OK


def _print_guard_advice(alias: str, name: str, store: ConnectionStore, out: IO[str]) -> None:
    """Say plainly that nothing bounds this connection's spend.

    Printed at the end of setup rather than buried in the receipt, because this
    is the moment a user forms their picture of what they just configured. The
    text names no number on purpose -- see docs/guards.md.
    """
    print("\n  ! No spend guards are configured for this connection.", file=out)
    _print_wrapped(
        "modelpass does not set a default threshold, because it has no basis for a "
        "number: it does not know your plan, your month, or what you are about to "
        "ask for. Pick one and add it:",
        out,
        indent="    ",
    )
    print(
        f"\n      modelpass connect {alias} --name {name} "
        "--warn-at-tokens N --stop-at-tokens N\n",
        file=out,
    )
    _print_wrapped(
        f"or edit {store.path} -- docs/guards.md walks through choosing N.",
        out,
        indent="    ",
    )


def _print_wrapped(text: str, out: IO[str], *, indent: str) -> None:
    import textwrap

    for line in textwrap.wrap(text, width=_WIDTH - len(indent)):
        print(f"{indent}{line}", file=out)


def _print_field(label: str, text: str, out: IO[str]) -> None:
    """A receipt line that may be a paragraph, kept aligned under its label."""
    import textwrap

    body = f"  {label:<13}"
    for index, line in enumerate(textwrap.wrap(text, width=_WIDTH - len(body))):
        print(f"{body if index == 0 else ' ' * len(body)}{line}", file=out)


# --- list ------------------------------------------------------------------------


def _print_store_notes(bridge: Bridge, out: IO[str]) -> None:
    """Anything in the file this build carried rather than read.

    Printed before the listing, not after: a connection missing from the rows
    below because this build cannot drive it is the first thing a reader needs,
    not a footnote.
    """
    try:
        notes = bridge.store.compatibility().notes()
    except SubpassError:
        return
    for note in notes:
        print(f"note: {note}", file=out)
    if notes:
        print("", file=out)


def _list(args: argparse.Namespace, bridge: Bridge, out: IO[str]) -> int:
    _print_store_notes(bridge, out)
    wanted_group = getattr(args, "group", None)
    connections = (
        bridge.connections()
        if wanted_group is None
        else bridge.find(group=wanted_group)
    )
    account_wording = args.command == "accounts"
    if connections and wanted_group is not None:
        print(f"group {wanted_group!r}\n", file=out)
    if not connections and wanted_group is not None:
        # Not NoSuchGroup: the user asked to *look*, and "nothing is in it" is
        # the answer to what they asked. Naming the groups that do exist is the
        # part that saves the next command.
        known = ", ".join(g.name for g in bridge.groups()) or "none"
        print(
            f"No connections in group {wanted_group!r}. Configured groups: {known}.",
            file=out,
        )
        return EXIT_OK
    if not connections:
        subject = "account profiles" if account_wording else "connections"
        print(
            f"No {subject} configured. A fresh install has no AI connectivity "
            "by design;\nrun 'modelpass connect anthropic' or 'modelpass connect openai' "
            "to add one.",
            file=out,
        )
        return EXIT_OK

    subject = "account profile(s)" if account_wording else "connection(s)"
    print(f"{len(connections)} {subject} in {bridge.store.path}\n", file=out)
    for connection in connections:
        state = "" if connection.enabled else "   [disabled -- runs are refused]"
        title = connection.display_name
        stable = f"   [id: {connection.name}]" if connection.nickname else ""
        print(f"  {title}{stable}{state}", file=out)
        print(f"      vendor    {connection.vendor}", file=out)
        print(
            f"      runtime   {connection.runtime.value}",
            file=out,
        )
        print(
            f"      auth      {connection.auth_mode.value} "
            f"({connection.credential_ref.describe()})",
            file=out,
        )
        # Not behind --verbose: a group is where a run can be sent, so a listing
        # that hides it lets a user believe 'group:cheap' would reach a
        # connection that is not in cheap.
        print(f"      groups    {', '.join(connection.group_names)}", file=out)
        print(f"      guards    {_guard_summary(connection.guards)}", file=out)
        # Not behind --verbose: "which account pays, and has anybody checked"
        # is the headline of an account profile, and a listing that only says it
        # when asked twice is a listing that lets a user assume the wrong one.
        if connection.is_subscription:
            if connection.account_binding:
                pinned = connection.account_binding.email or ""
                if connection.account_binding.organization_id:
                    pinned += (
                        (", " if pinned else "")
                        + f"org {connection.account_binding.organization_id}"
                    )
                print(f"      identity  VERIFIED: {pinned}", file=out)
            else:
                print("      identity  NOT VERIFIED", file=out)
        if connection.base_url:
            # Not behind --verbose, for the same reason the identity line is
            # not: on the one runtime that has a base URL it is the whole
            # answer to "what am I about to talk to", and a listing that hides
            # it lets a user mistake a local Ollama for a vendor endpoint.
            print(f"      baseUrl   {connection.base_url}", file=out)
        if args.verbose:
            if connection.model:
                print(f"      model     {connection.model}", file=out)
            bound = (
                "unbounded"
                if connection.timeout_seconds is None
                else f"{connection.timeout_seconds:g}s"
            )
            print(f"      timeout   {bound} (default for calls on this connection)", file=out)
            # Printed even when unset, and "unknown" is the honest word for it:
            # modelpass has no default input window and does not look one up, so
            # a blank line here would read as "small" or "fine" to a reader
            # sizing a payload.
            window = (
                "unknown -- nothing here guesses one"
                if connection.max_input_tokens is None
                else f"{connection.max_input_tokens:,} input tokens (as configured)"
            )
            print(f"      window    {window}", file=out)
            # Only when the connection stated something. Unlike the window
            # above, silence here is not a hazard: a connection that asked for
            # nothing is not at risk of misreading an answer it never got.
            if connection.prompt_cache is not None:
                plan = plan_prompt_cache(
                    connection.prompt_cache,
                    connection.runtime,
                    name=connection.name,
                )
                lifetime = plan.ttl or "the vendor's own lifetime"
                print(
                    f"      cache     prompt caching requested at {lifetime} "
                    f"-- {plan.disposition.value} on {connection.runtime.value}",
                    file=out,
                )
            if connection.retry == "never":
                print(
                    '      retry     never -- every retryable verdict here reads "no"',
                    file=out,
                )
            verified = connection.verified_capabilities
            if verified.supported or verified.unsupported:
                print(
                    f"      verified  {', '.join(verified.supported) or 'nothing'} "
                    f"(checked {verified.checked_at or 'at an unrecorded time'})",
                    file=out,
                )
            if connection.allow_env:
                print(f"      allowEnv  {', '.join(connection.allow_env)}", file=out)
            if connection.config_dir:
                print(f"      configDir {connection.config_dir}", file=out)
            if connection.description:
                print(f"      note      {connection.description}", file=out)
        print("", file=out)
    return EXIT_OK


def _guard_summary(guards: Guards) -> str:
    """One phrase per connection. "none" is a real and common answer."""
    parts: list[str] = []
    if guards.warn_at_tokens is not None:
        parts.append(f"warn at {guards.warn_at_tokens:,} tokens")
    if guards.stop_at_tokens is not None:
        parts.append(f"stop at {guards.stop_at_tokens:,} tokens")
    policy = guards.on_quota_exhausted
    if policy.failover:
        parts.append(f"on quota exhausted, fail over to {policy.failover!r}")
    else:
        parts.append("on quota exhausted, stop")
    if not guards.configured:
        return "none (nothing bounds one run's spend); " + parts[-1]
    return "; ".join(parts)


# --- groups ----------------------------------------------------------------------


def _groups(args: argparse.Namespace, bridge: Bridge, out: IO[str]) -> int:
    """Print the groups, or one of them, with the member a run would reach."""
    _print_store_notes(bridge, out)
    if args.name is not None:
        found = (bridge.group(args.name),)
    else:
        found = bridge.groups()
        if not found:
            print(
                "No connections are configured, so there are no groups. Run "
                "'modelpass connect anthropic' to add one.",
                file=out,
            )
            return EXIT_OK
        print(f"{len(found)} group(s) in {bridge.store.path}\n", file=out)

    for group in found:
        suffix = "   [the connections that name no group]" if group.is_default else ""
        print(f"  {group.name}{suffix}", file=out)
        for member in group.members:
            connection = bridge.connection(member)
            state = "" if connection.enabled else "   [disabled]"
            print(f"      {member}{state}", file=out)
        if group.available:
            selection = bridge.select(group.name)
            print(f"      -> group:{group.name} resolves to {selection.chosen.name}", file=out)
            for name, reason in selection.passed_over:
                print(f"         {name} passed over: {reason}", file=out)
        else:
            # The one state where naming the resolution is the whole point: a
            # group with members and no runnable one reads, in a plain listing,
            # exactly like a group that works.
            print(
                f"      -> group:{group.name} would be REFUSED: every member is disabled",
                file=out,
            )
        print("", file=out)
    return EXIT_OK


def _set_groups(args: argparse.Namespace, bridge: Bridge, out: IO[str]) -> int:
    """Move a connection between groups, and say what that did to the groups."""
    result = bridge.manage.set_groups(args.name, args.groups)
    where = ", ".join(result.effective)
    if not result.changed:
        print(f"{args.name} is already in {where}; nothing written.", file=out)
        return EXIT_OK
    print(f"{args.name} is now in {where}.", file=out)
    # A group that just lost its last member is gone, and a user who has been
    # told "moved" and not "and cheap no longer exists" will go looking for it.
    live = {group.name for group in bridge.groups()}
    emptied = [name for name in result.previous if name not in live]
    if emptied:
        print(
            f"Group(s) {', '.join(emptied)} now have no members and no longer exist.",
            file=out,
        )
    print(f"Written to {bridge.store.path}", file=out)
    return EXIT_OK


def _set_enabled(args: argparse.Namespace, bridge: Bridge, out: IO[str]) -> int:
    """``enable`` and ``disable``: one verb each, one write, one sentence back."""
    wanted = args.command == "enable"
    result = bridge.manage.set_enabled(args.name, wanted)
    state = "enabled" if wanted else "disabled"
    if not result.changed:
        print(f"{args.name} is already {state}; nothing written.", file=out)
        return EXIT_OK
    print(f"{args.name} is now {state}.", file=out)
    if not wanted:
        connection = bridge.connection(args.name)
        print(
            "It stays listed and 'modelpass check' still runs against it; what is "
            "refused is a run.",
            file=out,
        )
        # Where disabling has a consequence beyond this connection, say it here
        # rather than leaving it to be discovered by a refusal.
        stranded = [
            group.name
            for group in bridge.groups()
            if not group.available and connection.in_group(group.name)
        ]
        if stranded:
            print(
                f"Group(s) {', '.join(stranded)} now have no enabled member: "
                "'group:<name>' there will be refused.",
                file=out,
            )
    print(f"Written to {bridge.store.path}", file=out)
    return EXIT_OK


# --- bench -----------------------------------------------------------------------


def _bench(args: argparse.Namespace, bridge: Bridge, out: IO[str]) -> int:
    """Serve the bench, after saying plainly what is about to be exposed.

    The import is deferred to here rather than to module scope so that a modelpass
    without Flask installed still has a fully working command line -- the extra
    is only needed by the one subcommand that needs it, and the error names it.
    """
    from .bench import require_flask, serve

    # Before the banner, not after it: the announcement of a URL is a claim
    # that something is listening there, and the extra is missing often enough
    # that the claim has to be true when it is printed.
    require_flask()

    url = f"http://{_BENCH_HOST}:{args.port}/"
    print(f"modelpass bench on {url}", file=out)
    print(
        f"  Bound to {_BENCH_HOST} only. No authentication, because it is not a\n"
        "  service: anything that can reach this port can edit your connection\n"
        "  file and spend your allowance. Ctrl-C to stop.\n",
        file=out,
    )
    serve(port=args.port, bridge=bridge)
    return EXIT_OK


# --- check -----------------------------------------------------------------------


def _check(args: argparse.Namespace, bridge: Bridge, out: IO[str]) -> int:
    _print_store_notes(bridge, out)
    if args.name:
        targets = (bridge.connection(args.name),)
    else:
        targets = bridge.connections()
        if not targets:
            print("No connections configured; nothing to check.", file=out)
            _print_secret_notes(bridge, out)
            return EXIT_OK

    worst = EXIT_OK
    for index, connection in enumerate(targets):
        if index:
            print("", file=out)
        try:
            receipt = receipt_for(bridge, connection)
        except SubpassError as exc:
            # One unusable connection must not hide the state of the others.
            print(f"{connection.name}: FAILED -- {exc}", file=out)
            worst = EXIT_FAILED
            continue
        _print_receipt(receipt, out)
        _print_verified_capabilities(connection, out)
        if not receipt.ok:
            worst = EXIT_FAILED
    _print_secret_notes(bridge, out)
    return worst


def _print_verified_capabilities(connection: Connection, out: IO[str]) -> None:
    """What a drive found this *endpoint* doing, or that nobody has driven it.

    Only ever a line on ``openai-compatible``, and on it always a line. That
    runtime's row describes an API shape rather than a vendor, so the table
    alone cannot answer "will this work" -- an unverified endpoint reads
    ``unverified`` in every cell including ``chat``, which means ``bridge.chat``
    refuses it. Saying nothing here would leave the receipt reading OK beside a
    connection that cannot run (ticket 1.10).

    A connection on any other runtime carries no verified cells, so there is
    nothing to print and nothing is printed.
    """
    verified = connection.verified_capabilities
    if verified.supported or verified.unsupported:
        _print_field(
            "verified",
            f"{', '.join(verified.supported) or 'nothing'} supported; "
            f"{', '.join(verified.unsupported) or 'nothing'} unsupported; checked "
            f"{verified.checked_at or 'at an unrecorded time'}",
            out,
        )
        if verified.models:
            _print_field("models seen", ", ".join(verified.models), out)
    elif connection.runtime is Runtime.OPENAI_COMPATIBLE:
        _print_field(
            "verified",
            "nothing yet -- this runtime names an API shape, so every capability "
            f"cell reads 'unverified' until a drive answers for your endpoint. Run "
            f"'modelpass verify {connection.name}'",
            out,
        )


# --- shared output ---------------------------------------------------------------


def _print_receipt(receipt: Receipt, out: IO[str]) -> None:
    """The receipt, as a block rather than one line.

    ``Receipt.summary()`` is the one-liner every run recomputes; at setup time
    there is room to lay the same facts out so each is legible on its own --
    especially the scrub list, which is the part that says what this connection
    will *not* be allowed to see.
    """
    print(f"{receipt.connection}", file=out)
    print(f"  runtime      {receipt.runtime.value}", file=out)
    mode = receipt.effective_auth_mode.value
    detected = "" if receipt.detected_auth_mode else "  (declared; not confirmed)"
    print(f"  auth mode    {mode}{detected}", file=out)
    if receipt.credential_source:
        print(f"  credential   {receipt.credential_source}", file=out)
    if receipt.account:
        print(f"  account      {receipt.account}", file=out)
    if receipt.plan_name:
        print(f"  plan         {receipt.plan_name}", file=out)
    if receipt.identity_verified is True:
        print("  identity     VERIFIED (matches pinned account)", file=out)
    elif receipt.identity_verified is False:
        print("  identity     MISMATCH", file=out)
    elif receipt.requested_auth_mode is AuthMode.SUBSCRIPTION:
        print("  identity     NOT PINNED", file=out)
    profile = receipt.account_profile
    if profile is not None:
        if profile.auth_method:
            print(f"  login        {profile.auth_method}", file=out)
        if profile.email and profile.email != receipt.account:
            print(f"  email        {profile.email}", file=out)
        if profile.organization_name:
            print(f"  organization {profile.organization_name}", file=out)
        if profile.organization_id:
            print(f"  org ID       {profile.organization_id}", file=out)
        if profile.api_provider:
            print(f"  provider     {profile.api_provider}", file=out)
        print(f"  profile via  {profile.source}", file=out)
    if receipt.model:
        where = f"  ({receipt.model_source})" if receipt.model_source else ""
        print(f"  model        {receipt.model}{where}", file=out)
    else:
        # Said out loud rather than omitted. A blank line here is what left the
        # first consumer unable to diagnose a model rejection: nothing anywhere
        # before the run named a model, so nothing after it could be checked.
        _print_field("model", receipt.model_note or "", out)
    if receipt.scrubbed:
        print(f"  scrubbed     {', '.join(receipt.scrubbed)}", file=out)
    if receipt.preserved:
        print(f"  kept         {', '.join(receipt.preserved)}", file=out)
    if receipt.passthrough:
        print(f"  allowEnv     {', '.join(receipt.passthrough)}", file=out)
    if receipt.config_dir:
        print(f"  configDir    {receipt.config_dir}", file=out)
    for directive in receipt.directives:
        print(f"  directive    {directive.name} = {directive.value}", file=out)
    for note in receipt.notes:
        _print_field("note", note, out)
    guard_note = receipt.guard_note
    if guard_note:
        _print_field("guards", guard_note, out)
    else:
        print("  guards       configured", file=out)
    # Printed whichever way the answer falls, unlike the guard line above: a
    # sub-floor prefix costs a full write on every call, silently and with no
    # error, and a disclosure that only appeared when caching was already
    # working would leave exactly the people who need it guessing (D20).
    cache_note = receipt.cache_note
    if cache_note:
        _print_field("cache", cache_note, out)
    if receipt.ok:
        print("  status       OK", file=out)
    else:
        print(f"  status       FAILED -- {receipt.problem or 'unknown reason'}", file=out)


def _prompt(question: str) -> bool:
    try:
        answer = input(question)
    except EOFError:
        # A piped-in invocation with no --yes is a "no": defaulting the other way
        # would write config nobody looked at.
        return False
    return answer.strip().lower() in {"y", "yes"}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
