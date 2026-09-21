"""The bench's Flask application.

Server-rendered, no build step, no JavaScript framework, no CDN. The only script
on any page is the ~40 lines that read the playground's streaming response,
because a live event log is the one thing HTML cannot do on its own.

Three rules this module holds to, each tested:

1. **No secret is displayed or accepted.** The connection forms take
   *pointers* -- an environment variable name -- and a pasted ``sk-...`` is
   refused by the same :class:`CredentialRefIsSecret` validation the store uses.
   The ambient-environment audit shows variable **names** only; no value read
   from the environment reaches a template.
2. **Every mutation is a POST**, and a POST from another origin is refused. A
   page on the internet can reach ``127.0.0.1`` in your browser, and the thing
   behind this one edits your connection file.
3. **The playground spends real allowance.** It says so, on the page, next to
   the button -- and shows the usage per run rather than leaving you to guess.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ..adapters.base import RunRequest
from ..bridge import Bridge
from ..capabilities import Capability, Support
from ..connections import (
    ALLOWED_ENV_PASSTHROUGH,
    AccountBinding,
    Connection,
    CredentialKind,
    CredentialRef,
    Guards,
    QuotaAction,
    QuotaPolicy,
)
from ..errors import PreflightFailed, SubpassError
from ..manage import ConnectionPlan, Credential
from ..models import ModelChoice, models_for
from ..preflight import SCRUB_RULES, Receipt, env_names_to_scrub
from ..prompt_cache import plan_prompt_cache
from ..retry import classify_terminal
from ..runlog import run_log_for
from ..runtimes import Runtime
from ..tools import ToolDef
from ..types import AuthMode, Message, Role, TerminalStatus

__all__ = ["DEFAULT_PORT", "HOST", "create_app", "hello_world_tool", "serve"]

#: Loopback only, and not configurable. The bench renders credential *sources*,
#: an environment audit and a button that spends money; none of that should be
#: one mistyped flag away from the local network.
HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: Runtimes the bench offers, in the order the columns read.
#:
#: Every runtime except the two Google agent ones, which stay gated (D5) and
#: which a page with no ``--experimental`` affordance cannot write anyway. The
#: four API runtimes joined on 2026-09-13 (ticket 1.14): they are four of the six
#: a user can configure, and a bench that listed two of them was describing the
#: library as it stood before 1.6-1.11. ``google-api`` is here and its two agent
#: siblings are not, which is the D5 asymmetry recorded in ``runtimes.py``.
BENCH_RUNTIMES: tuple[Runtime, ...] = (
    Runtime.ANTHROPIC_SDK,
    Runtime.OPENAI_SDK,
    Runtime.ANTHROPIC_API,
    Runtime.OPENAI_API,
    Runtime.GOOGLE_API,
    Runtime.OPENAI_COMPATIBLE,
)

#: The capability cells the bench reports per runtime, in display order.
SHOWN_CAPABILITIES: tuple[Capability, ...] = (
    Capability.CHAT,
    Capability.STREAMING,
    Capability.THINKING,
    Capability.USAGE_TOKENS,
    Capability.INTERIM_USAGE,
    Capability.TOOLS_IN_PROCESS,
    Capability.MCP_SERVERS,
    Capability.GRACEFUL_CANCEL,
    Capability.STRUCTURED_OUTPUT,
)

#: How a stored credential reference reads back as a form choice. A stored key
#: comes back as ``keep`` rather than ``secret``: an edit that does not touch
#: the credential must not require the key to be typed again.
_CREDENTIAL_FORMS: dict[CredentialKind, str] = {
    CredentialKind.NATIVE_LOGIN: "native-login",
    CredentialKind.ENV: "env",
    CredentialKind.SECRET: "keep",
    CredentialKind.NONE: "none",
}

_FLASK_HINT = (
    "the bench needs Flask, which is not part of modelpass core (core has zero "
    "runtime dependencies on purpose). Install it with 'pip install modelpass[bench]'"
)


def require_flask():
    """Import Flask, or explain the extra. Lazy so core stays dependency-free.

    Public because the command line calls it *before* announcing a URL: a
    banner saying the bench is on http://127.0.0.1:8765 and then an error
    saying it could not start reads, for the half-second it is on screen, like
    a server that came up.
    """
    try:
        import flask
    except ImportError as exc:  # pragma: no cover - exercised by hand, not in CI
        raise SubpassError(f"{_FLASK_HINT} ({exc})") from exc
    return flask


# --- the hello-world tool ---------------------------------------------------------


def hello_world_tool() -> ToolDef:
    """One caller-supplied tool, for proving the round trip end to end.

    It returns a timestamp on purpose. A greeting alone could plausibly have been
    written by the model without ever calling anything; a timestamp generated in
    *this* process, at the moment of the call, could not. That is the difference
    between a demo and evidence.
    """

    def handler(args: Mapping[str, Any]) -> str:
        name = str(args.get("name") or "world")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        return f"Hello, {name}. This greeting was generated in the bench process at {now}."

    return ToolDef(
        name="hello_world",
        description=(
            "Greet someone by name. Returns a greeting stamped with the current time "
            "from the machine running modelpass. Call this whenever you are asked to."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Who to greet."}
            },
            "required": ["name"],
        },
        handler=handler,
    )


# --- view models ------------------------------------------------------------------


def _prompt_cache_note(connection: Connection) -> str | None:
    """The prompt-caching sentence for the accounts page, or ``None``.

    Never raises: a stored ``promptCache`` has already been through
    ``Connection.__post_init__``, but the page's job is to show a connection
    rather than to be the second place that refuses one.
    """
    if connection.prompt_cache is None:
        return None
    try:
        return plan_prompt_cache(
            connection.prompt_cache, connection.runtime, name=connection.name
        ).note
    except SubpassError as exc:  # pragma: no cover - defensive
        return str(exc)


def _connection_view(bridge: Bridge, connection: Connection) -> dict[str, Any]:
    """Everything the connections page shows about one connection.

    The launch plan is computed here rather than in the template: it is the
    thing that says what this connection will and will not be allowed to see,
    and it should be one function call away from the row that displays it.

    **Offline.** ``plan_launch`` and the stored connection, and nothing else. A
    preflight per row briefly lived here and it made loading the list a fan-out
    of multi-second vendor subprocesses -- one ``claude auth status`` and one
    Codex app-server launch per account, on a page whose job is to show what is
    configured. The live vendor answer belongs to the two places that ask for
    it: the Check receipt and the Verify button. What a row shows about identity
    comes from the binding on disk, which is what "verified" means here anyway.
    """
    guards = connection.guards
    try:
        plan = bridge.plan(connection)
    except SubpassError as exc:
        scrubbed, kept, credential_present = [], [], True
        problem = str(exc)
    else:
        scrubbed = list(plan.scrubbed)
        kept = list(plan.kept)
        credential_present = plan.credential_present
        problem = None

    thresholds: list[str] = []
    if guards.warn_at_tokens is not None:
        thresholds.append(f"warn at {guards.warn_at_tokens:,} tokens")
    if guards.stop_at_tokens is not None:
        thresholds.append(f"stop at {guards.stop_at_tokens:,} tokens")
    policy = guards.on_quota_exhausted
    config_env = {
        Runtime.ANTHROPIC_SDK: "CLAUDE_CONFIG_DIR",
        Runtime.OPENAI_SDK: "CODEX_HOME",
    }.get(connection.runtime)
    setup_note = None
    if connection.config_dir and config_env == "CLAUDE_CONFIG_DIR":
        setup_note = (
            f"Set CLAUDE_CONFIG_DIR to {connection.config_dir!r}, then run "
            "claude /login once for this account."
        )
    elif connection.config_dir and config_env == "CODEX_HOME":
        setup_note = (
            f"Set CODEX_HOME to {connection.config_dir!r}, then run codex login "
            "once for this account. Either credential store works: Codex keys its "
            "keyring entry to CODEX_HOME as well as its auth.json."
        )

    return {
        "name": connection.name,
        "nickname": connection.nickname,
        "display_name": connection.display_name,
        "vendor": connection.vendor,
        "runtime": connection.runtime.value,
        "auth_mode": connection.auth_mode.value,
        "credential_ref": connection.credential_ref.to_str(),
        "credential_describes": connection.credential_ref.describe(),
        "credential_present": credential_present,
        "model": connection.model,
        "description": connection.description,
        "enabled": connection.enabled,
        "guards_configured": guards.configured,
        "guard_thresholds": thresholds,
        "quota_action": policy.action.value,
        "quota_failover": policy.failover,
        "allow_env": list(connection.allow_env),
        "groups": list(connection.group_names),
        "declared_groups": list(connection.groups),
        "config_dir": connection.config_dir,
        "config_env": config_env,
        "setup_note": setup_note,
        # Driven by the binding on disk, not by a live probe: this page is
        # offline, and "verified" means "somebody pinned this", which is exactly
        # what the stored binding records.
        "identity_verified": connection.account_binding is not None,
        "verified_identity": (
            connection.account_binding.to_dict()
            if connection.account_binding is not None
            else None
        ),
        "scrubbed": scrubbed,
        "kept": kept,
        "plan_problem": problem,
        # --- ticket 1.14 -------------------------------------------------
        # The endpoint, the bounds and what a drive found. Each of these is a
        # fact a run turns on and none of them was on this page: a user could
        # not tell a connection pointed at a local Ollama from one pointed at a
        # gateway, nor a connection that refuses to be retried from one that
        # does not, without opening the file.
        "base_url": connection.base_url,
        "timeout_seconds": connection.timeout_seconds,
        "retry": connection.retry,
        # `None` reaches the template as the unknown it is; the template says so
        # in words rather than leaving the row blank.
        "max_input_tokens": connection.max_input_tokens,
        # The stated request, verbatim, plus the sentence that says what this
        # runtime does about it. The note is computed here rather than in the
        # template because it is the same answer the CLI and the receipt give,
        # and three renderings of one fact is how they drift apart.
        "prompt_cache": connection.prompt_cache,
        "prompt_cache_note": _prompt_cache_note(connection),
        "is_compatible": connection.runtime is Runtime.OPENAI_COMPATIBLE,
        "verified": {
            "supported": list(connection.verified_capabilities.supported),
            "unsupported": list(connection.verified_capabilities.unsupported),
            "checked_at": connection.verified_capabilities.checked_at,
            "models": list(connection.verified_capabilities.models),
        },
        "verified_any": bool(
            connection.verified_capabilities.supported
            or connection.verified_capabilities.unsupported
        ),
    }


def _store_notes(bridge: Bridge) -> list[str]:
    """Anything in the connection file this build carried rather than read (0.1).

    Shown above the listing rather than below it: a connection missing from the
    rows because this build cannot drive its runtime is the first thing a reader
    needs, not a footnote.
    """
    try:
        return list(bridge.store.compatibility().notes())
    except SubpassError as exc:
        return [str(exc)]


def _secrets_view(bridge: Bridge) -> dict[str, Any]:
    """The secrets file, as names and references. Never a value.

    The same three facts ``modelpass check`` and ``modelpass secrets`` print
    (R14, tickets 1.3 and 1.4): what the permissions actually are, anything that
    made the file unreadable, and which connections point at each entry. An
    entry nothing references is marked an orphan and is never collected here --
    this page has a delete button for connections and deliberately none for
    keys.
    """
    secrets = bridge.secrets
    if not secrets.exists():
        return {"exists": False, "path": str(secrets.path), "entries": [], "notes": []}
    connections = bridge.connections()
    entries = [
        {
            "entry": entry,
            "used_by": list(secrets.referenced_by(entry, connections)),
        }
        for entry in secrets.entries()
    ]
    return {
        "exists": True,
        "path": str(secrets.path),
        "permissions": secrets.permissions().note,
        "notes": list(secrets.notes()),
        "entries": entries,
        "orphans": list(secrets.orphans(connections)),
    }


def _ambient_audit(env: Mapping[str, str]) -> list[dict[str, Any]]:
    """Which scrub-list variables are set in *this process's* environment.

    Names only. A value read here would end up in a rendered page, and the whole
    argument of this section is that modelpass can see these variables without
    ever passing them on.
    """
    out = []
    for runtime in BENCH_RUNTIMES:
        present = list(env_names_to_scrub(runtime, env))
        watched: list[str] = []
        for rule in SCRUB_RULES.get(runtime, ()):
            watched.extend(rule.names)
            watched.extend(f"{prefix}*" for prefix in rule.prefixes)
        # Composed here rather than in the template so the sentence a user reads
        # is one string in one place, and so a test can assert on it verbatim.
        if present:
            plural = len(present) > 1
            sentence = (
                f"{', '.join(present)} {'are' if plural else 'is'} set in this "
                f"environment; runs on any {runtime.value} connection scrub "
                f"{'them' if plural else 'it'}."
            )
        else:
            sentence = ""
        out.append(
            {
                "runtime": runtime.value,
                "present": present,
                "watched": watched,
                "sentence": sentence,
            }
        )
    return out


def _capability_rows(bridge: Bridge) -> list[dict[str, Any]]:
    rows = []
    for capability in SHOWN_CAPABILITIES:
        cells = []
        for runtime in BENCH_RUNTIMES:
            support = bridge.registry.support(runtime, capability)
            cells.append(
                {
                    "runtime": runtime.value,
                    "support": support.value,
                    "note": bridge.registry.note(runtime, capability),
                }
            )
        rows.append({"capability": capability.value, "cells": cells})
    return rows


def _receipt_view(bridge: Bridge, name: str) -> dict[str, Any]:
    """Run a preflight and shape its receipt for display, or report the refusal.

    A preflight costs nothing and touches no allowance, which is why the check
    button is a plain GET and why it still runs for a *disabled* connection --
    ``enabled = false`` switches off spending, not looking.
    """
    try:
        connection = bridge.connection(name)
        receipt = bridge.preflight(connection)
    except SubpassError as exc:
        return {"connection": name, "failed": str(exc)}
    data = receipt.to_dict()
    data["failed"] = None
    data["guard_note"] = receipt.guard_note
    data["cache_note"] = receipt.cache_note
    data["enabled"] = connection.enabled
    return data


# --- the app ----------------------------------------------------------------------


def create_app(
    *,
    bridge: Bridge | None = None,
    codex_root: str | os.PathLike[str] | None = None,
) -> Any:
    """Build the bench application.

    ``bridge`` carries the store path, the adapter registry and the environment
    the audit reports on, so the whole surface is testable offline against
    ``FakeAdapter`` and a temp home -- the same injection seam the CLI uses.
    """
    flask = require_flask()
    from flask import Response, abort, redirect, render_template, request, url_for

    active = bridge if bridge is not None else Bridge()
    app = flask.Flask(__name__)
    app.config["SUBPASS_BRIDGE"] = active
    app.config["SUBPASS_CODEX_ROOT"] = codex_root

    @app.before_request
    def _refuse_cross_origin_writes():
        """A page on the internet can POST to 127.0.0.1 in your browser.

        Behind this one is a form that rewrites the connection file, so a
        mutating request carrying somebody else's Origin is refused. Requests
        with no Origin header at all (curl, the test client, an old browser
        doing a plain form post) are allowed -- this is a guard against a
        hostile *page*, not an authentication system.
        """
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            abort(403, "cross-origin request refused")
        return None

    def _flash(message: str, kind: str = "ok"):
        return redirect(url_for("connections", note=message, note_kind=kind))

    # --- 1. configured connections -------------------------------------------

    @app.route("/")
    def connections():
        checked = request.args.get("check")
        # A filter rather than a separate page: these are the same rows, and a
        # group that nothing is in shows an empty listing rather than an error,
        # because the user asked to look.
        group_filter = request.args.get("group") or None
        shown = (
            active.connections()
            if group_filter is None
            else active.find(group=group_filter)
        )
        return render_template(
            "connections.html",
            page="connections",
            store_path=str(active.store.path),
            run_log_path=str(run_log_for(active.store).path),
            run_log_enabled=run_log_for(active.store).enabled,
            group_filter=group_filter,
            connections=[_connection_view(active, c) for c in shown],
            store_notes=_store_notes(active),
            secrets=_secrets_view(active),
            audit=_ambient_audit(active.env),
            capabilities=_capability_rows(active),
            runtimes=[r.value for r in BENCH_RUNTIMES],
            receipt=_receipt_view(active, checked) if checked else None,
            note=request.args.get("note"),
            note_kind=request.args.get("note_kind", "ok"),
        )

    @app.route("/groups")
    def groups():
        """The groups, and which member each one would actually reach.

        A page of its own rather than a column on the listing, because the
        question it answers is not about any one connection: ``group:cheap``
        resolves to something, and the only place that was previously visible
        was a run's receipt, after the fact.
        """
        rows = []
        for group in active.groups():
            selection = None
            if group.available:
                selection = active.select(group.name).to_dict()
            rows.append(
                {
                    **group.to_dict(),
                    "selection": selection,
                    "disabled": [
                        name for name in group.members if name not in group.enabled
                    ],
                }
            )
        return render_template(
            "groups.html",
            page="groups",
            store_path=str(active.store.path),
            groups=rows,
            note=request.args.get("note"),
            note_kind=request.args.get("note_kind", "ok"),
        )

    # --- 2. vendor setup ------------------------------------------------------

    @app.route("/accounts/new")
    @app.route("/connections/new")
    def new_connection():
        return render_template(
            "connection_form.html",
            page="connections",
            mode="new",
            form=_blank_form(request.args.get("runtime")),
            runtimes=[r.value for r in BENCH_RUNTIMES],
            allow_env_options=_allow_env_options(),
            error=None,
        )

    @app.route("/accounts/<name>/edit")
    @app.route("/connections/<name>/edit")
    def edit_connection(name: str):
        try:
            connection = active.connection(name)
        except SubpassError as exc:
            return _flash(str(exc), "error")
        return render_template(
            "connection_form.html",
            page="connections",
            mode="edit",
            form=_form_from_connection(connection),
            runtimes=[r.value for r in BENCH_RUNTIMES],
            allow_env_options=_allow_env_options(),
            error=None,
        )

    @app.post("/accounts/save")
    @app.post("/connections/save")
    def save_connection():
        original = (request.form.get("original_name") or "").strip()
        mode = "edit" if original else "new"
        previous = None
        # What an *absent* enabled field means: for an edit, "leave it as it
        # was", so a scripted POST that omits the field cannot quietly bring a
        # disabled connection back into service. The rendered form always sends
        # the field, so this only ever applies to a hand-made request.
        default_enabled = True
        if original:
            try:
                previous = active.connection(original)
                default_enabled = previous.enabled
            except SubpassError:
                default_enabled = True
        form = _form_from_request(request.form, default_enabled=default_enabled)
        # Read once, off the request, and never put in ``form``: what a failed
        # save re-renders is ``form``, so a value that never enters it can never
        # be echoed back into a page (ticket 1.14).
        pasted_key = str(request.form.get("api_key") or "")

        def refuse(message: str):
            return render_template(
                "connection_form.html",
                page="connections",
                mode=mode,
                form=form,
                runtimes=[r.value for r in BENCH_RUNTIMES],
                allow_env_options=_allow_env_options(),
                error=message,
            ), 400

        # Overwriting is allowed in exactly one case: editing a connection in
        # place, under the name it already had. Creating a connection whose name
        # is taken, or renaming one onto a name that is taken, must fail loudly
        # -- silently replacing an existing connection would destroy somebody's
        # guard thresholds as a side effect of a typed name.
        new_name = str(form.get("name") or "").strip()
        editing_in_place = bool(original) and original == new_name
        try:
            credential = _credential_from_form(form, pasted_key, previous)
            if original and not editing_in_place:
                # A rename goes through the manager rather than through an add
                # and a remove, because those two orders each get the secrets
                # file wrong in their own way: one orphans the key, the other
                # destroys it. `rename_connection` moves the entry when it is
                # unambiguously this connection's and says so when it is not
                # (ticket 1.4).
                renamed = active.manage.rename_connection(original, new_name)
                previous = active.connection(new_name)
                if credential.kind is CredentialKind.SECRET and not credential.value:
                    # The entry moved with the connection, so "keep the stored
                    # key" now means the new name.
                    credential = Credential.stored_secret(
                        renamed.secret_entry or credential.locator or new_name
                    )
            # Everything a connection is made of, validated by the same manager
            # `modelpass connect` uses -- so the secrets file is written first
            # and by one implementation, not two (ticket 1.4).
            fields = dict(
                name=new_name,
                runtime=Runtime(form.get("runtime") or ""),
                credential=credential,
                guards=_guards_from_form(form),
                model=str(form.get("model") or "").strip() or None,
                base_url=str(form.get("base_url") or "").strip() or None,
                description=str(form.get("description") or "").strip() or None,
                nickname=str(form.get("nickname") or "").strip() or None,
                config_dir=str(form.get("config_dir") or "").strip() or None,
                allow_env=tuple(form.get("allow_env") or ()),
                groups=_groups_from_form(form),
                enabled=bool(form.get("enabled", True)),
                overwrite=bool(original),
            )
            try:
                plan = active.manage.plan_connection(**fields)
            except PreflightFailed as exc:
                # This page has never refused a connection whose credential is
                # not resolvable *yet* -- naming a variable you are about to set
                # is an ordinary thing to do at a Settings page, and the ambient
                # audit below the listing is what tells you it is not set. So the
                # refusal becomes a plan carrying a failing receipt, which is a
                # shape `add_connection(force=True)` already understands, and the
                # problem is shown on the way back. A pasted key is the one
                # exception: it resolves in memory, so a failure here is a real
                # one and writing the key anyway would store a secret for a
                # connection that does not work.
                if credential.value is not None:
                    raise
                plan = _unpreflighted_plan(active, fields, exc)
            connection = plan.connection
            if previous is not None:
                # Carried rather than re-derived: `retry`, `timeoutSeconds`,
                # `maxInputTokens`, `promptCache` and the verified cells are
                # connection state this form does not edit, and an edit that
                # dropped them would silently un-bound a call, throw away a
                # drive's evidence, turn a stated input window back into an
                # unknown one, or withdraw a caching request nobody withdrew.
                connection = replace(
                    connection,
                    retry=previous.retry,
                    timeout_seconds=previous.timeout_seconds,
                    max_input_tokens=previous.max_input_tokens,
                    prompt_cache=previous.prompt_cache,
                    verified_capabilities=previous.verified_capabilities,
                )
                if (
                    previous.runtime == connection.runtime
                    and previous.auth_mode == connection.auth_mode
                    and previous.credential_ref == connection.credential_ref
                    and previous.config_dir == connection.config_dir
                ):
                    connection = replace(
                        connection, account_binding=previous.account_binding
                    )
            # `force` because this page has never preflighted on save and must
            # not start refusing a connection whose variable is set later; the
            # receipt's problem is shown on the way back instead, and the Check
            # button is a click away.
            active.manage.add_connection(replace(plan, connection=connection), force=True)
        except (SubpassError, ValueError) as exc:
            return refuse(str(exc))
        verb = "Updated" if original else "Saved"
        message = f"{verb} account {connection.name!r}."
        if not plan.receipt.ok:
            return _flash(
                f"{message} Its preflight does not pass yet: "
                f"{plan.receipt.problem or 'unknown reason'}",
                "error",
            )
        return _flash(message)

    @app.post("/accounts/<name>/delete")
    @app.post("/connections/<name>/delete")
    def delete_connection(name: str):
        try:
            # The connection goes first and the secret rule is applied second,
            # so a crash between the two writes leaves an inert orphan rather
            # than a connection pointing at an entry that is gone. Both halves,
            # and the sentence explaining what happened to the key, live in
            # ConnectionManager.remove_connection (ticket 1.4).
            result = active.manage.remove_connection(name)
        except SubpassError as exc:
            return _flash(str(exc), "error")
        suffix = f" {result.note}." if result.note else ""
        return _flash(f"Deleted account {name!r}.{suffix}")

    @app.post("/accounts/<name>/enabled")
    @app.post("/connections/<name>/enabled")
    def set_enabled(name: str):
        wanted = request.form.get("enabled") == "true"
        try:
            active.manage.set_enabled(name, wanted)
        except SubpassError as exc:
            return _flash(str(exc), "error")
        state = "enabled" if wanted else "disabled"
        return _flash(f"Account {name!r} is now {state}.")

    @app.post("/accounts/<name>/verify")
    def verify_identity(name: str):
        try:
            connection = active.connection(name)
            if connection.runtime is Runtime.OPENAI_COMPATIBLE:
                return _verify_capabilities(name, connection)
            if not connection.is_subscription:
                raise ValueError("API-key accounts have no subscription identity to pin")
            # Verify is the one caller that must see a live probe, not the
            # adapter's short-lived cache of one.
            active.refresh_identity(connection)
            candidate = replace(connection, account_binding=None)
            receipt = active.preflight(candidate)
            if not receipt.ok:
                raise ValueError(receipt.problem or "identity preflight failed")
            profile = receipt.account_profile
            if profile is None or not (profile.email or profile.organization_id):
                raise ValueError(
                    "the vendor did not report an email or organization ID"
                )
            binding = AccountBinding(
                email=profile.email, organization_id=profile.organization_id
            )
            active.store.add(
                replace(connection, account_binding=binding), overwrite=True
            )
        except (SubpassError, ValueError) as exc:
            return _flash(f"Could not verify {name!r}: {exc}", "error")
        return _flash(
            f"Verified {connection.display_name!r}. Future account drift will be blocked."
        )

    def _verify_capabilities(name: str, connection: Connection):
        """Drive an ``openai-compatible`` endpoint and record what answered.

        The same verb as the identity probe above and the same button, for the
        reason ``modelpass verify`` gives them one subcommand: both are "go and
        find out the live answer and write it down". The difference is that this
        one **spends** -- three short calls against whatever the base URL points
        at -- which is why the button confirms first.

        Nothing is written unless the drive reached the endpoint at all: a box
        that is switched off must not leave behind a record saying it can do
        nothing (ticket 1.10).
        """
        try:
            adapter = active.adapter_for(connection.runtime)
            report = adapter.verify_capabilities(
                RunRequest(
                    connection=connection,
                    messages=(Message(role=Role.USER, content="ok"),),
                    plan=active.plan(connection),
                )
            )
        except SubpassError as exc:
            return _flash(f"Could not verify {name!r}: {exc}", "error")
        if report is None:
            return _flash(
                f"The adapter for runtime {connection.runtime.value!r} does not "
                "implement a capability drive, so there is nothing to verify.",
                "error",
            )
        if not report.ok:
            return _flash(
                f"Could not verify {name!r}: {report.problem or 'the drive failed'}. "
                "Nothing was written.",
                "error",
            )
        active.store.add(
            replace(connection, verified_capabilities=report.verified), overwrite=True
        )
        supported = ", ".join(report.verified.supported) or "nothing"
        return _flash(
            f"Verified {connection.display_name!r} against {connection.base_url}: "
            f"{supported}. Cells the drive could not decide still read as the "
            "capability table left them."
        )

    # --- 3. playground --------------------------------------------------------

    @app.route("/playground")
    def playground():
        rows = []
        for connection in active.connections():
            runtime = connection.runtime
            support = active.registry.support(runtime, Capability.TOOLS_IN_PROCESS)
            catalogue = models_for(
                runtime, codex_root=app.config["SUBPASS_CODEX_ROOT"]
            )
            verified_models = connection.verified_capabilities.models
            if verified_models:
                # The one runtime where a model list is a fact about *this*
                # connection: `modelpass verify` asked the endpoint what it
                # serves and wrote the answer down (ticket 1.10). Better than
                # the free-text box the catalogue degrades to, and honestly
                # sourced -- the picker says where it came from.
                catalogue = replace(
                    catalogue,
                    choices=tuple(
                        ModelChoice(id=name, label=name, description="")
                        for name in verified_models
                    ),
                    source=(
                        "reported by this endpoint when it was verified"
                        f" ({connection.verified_capabilities.checked_at or 'undated'})"
                    ),
                    exact=True,
                    problem=None,
                )
            rows.append(
                {
                    "name": connection.name,
                    "display_name": connection.display_name,
                    "runtime": runtime.value,
                    "auth_mode": connection.auth_mode.value,
                    "enabled": connection.enabled,
                    "model": connection.model,
                    "tools_supported": support is Support.SUPPORTED,
                    "tools_support": support.value,
                    "tools_note": active.registry.note(
                        runtime, Capability.TOOLS_IN_PROCESS
                    ),
                    "guards_configured": connection.guards.configured,
                    "models": [
                        {
                            "id": choice.id,
                            "label": choice.label,
                            "description": choice.description,
                        }
                        for choice in catalogue.choices
                    ],
                    "model_source": catalogue.source,
                    "model_exact": catalogue.exact,
                    "model_problem": catalogue.problem,
                    "model_notes": list(catalogue.notes),
                }
            )
        return render_template(
            "playground.html",
            page="playground",
            connections=rows,
            tool=hello_world_tool().to_dict(),
        )

    @app.post("/playground/run")
    def playground_run():
        payload = request.get_json(silent=True) or {}
        name = str(payload.get("connection") or "")
        message = str(payload.get("message") or "").strip()
        model = str(payload.get("model") or "").strip() or None
        with_tool = bool(payload.get("tool"))

        if not message:
            return {"error": "a message is required"}, 400

        # Everything the generator needs is read out of the request here: the
        # response body is produced after the request context is gone.
        def stream() -> Iterator[str]:
            started = datetime.now(UTC).isoformat(timespec="seconds")
            yield _line(
                {
                    "kind": "run_started",
                    "connection": name,
                    "model": model,
                    "tool": with_tool,
                    "at": started,
                }
            )
            try:
                events = active.chat(
                    connection=name,
                    message=message,
                    model=model,
                    tools=[hello_world_tool()] if with_tool else None,
                )
                for event in events:
                    yield _line({"kind": "event", "event": event.to_dict()})
            except SubpassError as exc:
                # Refusals are the interesting half of this page: a disabled
                # connection, a capability the runtime does not have, a model
                # the installed CLI does not know. They render as a labelled
                # error rather than a dead stream.
                yield _line(
                    {
                        "kind": "error",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive
                yield _line(
                    {"kind": "error", "error": str(exc), "error_type": type(exc).__name__}
                )
            yield _line({"kind": "run_finished"})

        return Response(stream(), mimetype="application/x-ndjson")

    # --- 4. run log -----------------------------------------------------------

    @app.route("/runs")
    def runs():
        log = run_log_for(active.store)
        return render_template(
            "runs.html",
            page="runs",
            records=[_run_view(record) for record in log.read()],
            path=str(log.path),
            enabled=log.enabled,
            exists=log.exists(),
        )

    return app


def _run_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """One stored run, with the questions tickets 1.5, 1.7 and 1.12 made askable.

    Three things a ledger row could not previously answer, all of them off keys
    the run log already writes:

    * **Were the cache breakpoints I wrote actually honoured?** Two numbers that
      disagree is a run that paid for a prefix it thought it had cached (R3).
    * **Was this run at the temperature its config says?** ``sampling_requested``
      against ``sampling_applied``, with one sentence per field that moved (R5).
    * **Is this failure worth trying again?** Computed here rather than stored,
      by the library's own :func:`~modelpass.retry.classify_terminal`, from the
      two typed facts the row carries -- runtime and terminal status. A verdict
      derived from stored facts is honest; a verdict stored at the time and read
      back as current would not be, because the connection's stance can have
      changed since.

    Everything else is passed through untouched. A hand-edited row that
    ``RunLog.read`` already normalized stays normalized.
    """
    view = dict(record)
    requested = dict(record.get("sampling_requested") or {})
    applied = dict(record.get("sampling_applied") or {})
    view["sampling_requested"] = requested
    view["sampling_applied"] = applied
    view["sampling_notes"] = list(record.get("sampling_notes") or ())
    view["sampling_moved"] = requested != applied
    view["cache_breakpoints_requested"] = int(record.get("cache_breakpoints_requested") or 0)
    view["cache_breakpoints_honoured"] = int(record.get("cache_breakpoints_honoured") or 0)
    view["cache_breakpoints_dropped"] = (
        view["cache_breakpoints_requested"] > view["cache_breakpoints_honoured"]
    )
    view["timed_out"] = record.get("status") == TerminalStatus.TIMED_OUT.value
    verdict = None
    try:
        verdict = classify_terminal(
            Runtime(record.get("runtime")), TerminalStatus(record.get("status"))
        )
    except ValueError:
        # A row naming a runtime or a status this build does not know. The rest
        # of the line is still worth rendering, which is the whole rule the run
        # log page already follows for a hand-edited line.
        pass
    view["retry_verdict"] = None if verdict is None else verdict.retryable.value
    view["retry_note"] = None if verdict is None else verdict.note
    return view


def _line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":")) + "\n"


# --- form handling ----------------------------------------------------------------


def _allow_env_options() -> dict[str, list[str]]:
    return {
        runtime.value: sorted(ALLOWED_ENV_PASSTHROUGH.get(runtime, frozenset()))
        for runtime in BENCH_RUNTIMES
    }


def _blank_form(runtime: str | None = None) -> dict[str, Any]:
    return {
        "original_name": "",
        "name": "",
        "runtime": runtime if runtime in {r.value for r in BENCH_RUNTIMES} else "",
        "auth_mode": AuthMode.SUBSCRIPTION.value,
        # Which of the credential forms this connection uses (ticket 1.14). The
        # pasted key that goes with ``secret`` is never a member of this dict:
        # it is read straight off the request, handed to the manager, and
        # dropped. What is re-rendered after a failed save is this form, so a
        # value that never enters it can never be echoed back into a page.
        "credential": "native-login",
        "secret_entry": "",
        "api_key_env": "",
        "base_url": "",
        "model": "",
        "description": "",
        "nickname": "",
        "config_dir": "",
        "warn_at_tokens": "",
        "stop_at_tokens": "",
        "quota_action": QuotaAction.STOP.value,
        "failover": "",
        "allow_env": [],
        "groups": "",
        "enabled": True,
    }


def _form_from_connection(connection: Connection) -> dict[str, Any]:
    guards = connection.guards
    ref = connection.credential_ref
    return {
        "original_name": connection.name,
        "name": connection.name,
        "runtime": connection.runtime.value,
        "auth_mode": connection.auth_mode.value,
        # "keep" rather than "secret" on a stored key: an edit that does not
        # touch the credential must not need the key typed again, and a form
        # that asked for it would teach people to paste keys into a web page
        # for no reason.
        "credential": _CREDENTIAL_FORMS[ref.kind],
        "secret_entry": ref.locator or "" if ref.kind is CredentialKind.SECRET else "",
        "api_key_env": ref.locator or "" if ref.kind.value == "env" else "",
        "base_url": connection.base_url or "",
        "model": connection.model or "",
        "description": connection.description or "",
        "nickname": connection.nickname or "",
        "config_dir": connection.config_dir or "",
        "warn_at_tokens": "" if guards.warn_at_tokens is None else str(guards.warn_at_tokens),
        "stop_at_tokens": "" if guards.stop_at_tokens is None else str(guards.stop_at_tokens),
        "quota_action": guards.on_quota_exhausted.action.value,
        "failover": guards.on_quota_exhausted.failover or "",
        "allow_env": list(connection.allow_env),
        # One text box, space or comma separated, rather than a repeated field:
        # groups are free-form names the user invents, so there is no list to
        # offer checkboxes for the way `allowEnv` has one.
        "groups": " ".join(connection.groups),
        "enabled": connection.enabled,
    }


def _form_from_request(
    source: Mapping[str, Any], *, default_enabled: bool = True
) -> dict[str, Any]:
    form = _blank_form()
    for key in (
        "original_name",
        "name",
        "runtime",
        "auth_mode",
        "credential",
        "secret_entry",
        "api_key_env",
        "base_url",
        "model",
        "description",
        "nickname",
        "config_dir",
        "warn_at_tokens",
        "stop_at_tokens",
        "quota_action",
        "failover",
        "groups",
    ):
        form[key] = str(source.get(key) or "").strip()
    getlist = getattr(source, "getlist", None)
    form["allow_env"] = list(getlist("allow_env")) if getlist else []
    # The checkbox submits "true" only when ticked, and a hidden "false" always
    # precedes it, so the *list* is what carries the answer. Reading it this way
    # means the form works with JavaScript switched off, and an unticked box is
    # a real "no" rather than a missing key.
    values = list(getlist("enabled")) if getlist else _as_list(source.get("enabled"))
    form["enabled"] = "true" in values if values else default_enabled
    return form


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _unpreflighted_plan(
    bridge: Bridge, fields: Mapping[str, Any], problem: BaseException
) -> ConnectionPlan:
    """The plan for a connection whose preflight could not run yet.

    Same shape :meth:`~modelpass.manage.ConnectionManager.plan_connection`
    returns, with a receipt that says plainly what stopped it, so the one write
    path -- ``add_connection`` -- still applies. Reached only where the
    credential is a *pointer*: a named variable that is not set in this process,
    which is the state a Settings page is routinely used to fix. A pasted key
    never arrives here, so this never builds a pending secret and never writes
    one.
    """
    credential: Credential = fields["credential"]
    locator = credential.locator
    ref = {
        CredentialKind.NATIVE_LOGIN: CredentialRef.native_login,
        CredentialKind.NONE: CredentialRef.none,
    }.get(credential.kind, lambda: CredentialRef(credential.kind, locator))()
    connection = Connection(
        name=fields["name"],
        runtime=fields["runtime"],
        auth_mode=credential.auth_mode,
        credential_ref=ref,
        guards=fields["guards"],
        model=fields["model"],
        base_url=fields["base_url"],
        description=fields["description"],
        nickname=fields["nickname"],
        config_dir=fields["config_dir"],
        allow_env=fields["allow_env"],
        groups=fields["groups"],
        enabled=fields["enabled"],
    )
    return ConnectionPlan(
        connection=connection,
        receipt=Receipt(
            connection=connection.name,
            runtime=connection.runtime,
            requested_auth_mode=connection.auth_mode,
            guards_configured=connection.guards.configured,
            retry=connection.retry,
            ok=False,
            problem=str(problem),
        ),
        pending_secret=None,
        exists=bridge.store.has(connection.name),
        overwrite=bool(fields.get("overwrite")),
    )


def _groups_from_form(form: Mapping[str, Any]) -> tuple[str, ...]:
    """Read the groups box: names separated by commas or spaces, in order.

    Both separators are accepted because both are what people type, and the
    order is kept as written -- it is not what resolution uses today, but
    silently sorting a list somebody typed would make a later preference order
    impossible to introduce without changing what an existing file means.
    Validation stays on :class:`~modelpass.connections.Connection`, so a bad
    name is refused by the same rule the CLI and the file are held to.
    """
    raw = str(form.get("groups") or "").replace(",", " ")
    return tuple(name for name in raw.split() if name)


def _guards_from_form(form: Mapping[str, Any]) -> Guards:
    """The thresholds and the quota policy, validated by the core type."""
    return Guards(
        warn_at_tokens=_optional_int(form.get("warn_at_tokens"), "warnAtTokens"),
        stop_at_tokens=_optional_int(form.get("stop_at_tokens"), "stopAtTokens"),
        on_quota_exhausted=QuotaPolicy(
            action=QuotaAction(form.get("quota_action") or QuotaAction.STOP.value),
            failover=(str(form.get("failover") or "").strip() or None),
        ),
    )


def _credential_from_form(
    form: Mapping[str, Any], api_key: str, previous: Connection | None
) -> Credential:
    """Which of the four credential forms this submission means (ticket 1.14).

    ``api_key`` is the pasted value, passed separately and deliberately: it is
    read off the request, it reaches :class:`~modelpass.manage.Credential`, and
    it is in no dict that a template can see. A ``Credential.secret`` excludes
    it from its own ``repr`` and has no ``to_dict``, so the value cannot reach a
    rendered page or a log line through the plan either -- only the entry name
    does.

    Every check is the core type's rather than a second copy written for a web
    form: a pasted ``sk-...`` typed into the *environment variable* field is
    refused by the same ``CredentialRefIsSecret`` rule that protects the config
    file, because ``Credential.env`` validates the pointer at construction.
    """
    choice = str(form.get("credential") or "").strip()
    if not choice:
        # A submission from before this field existed, or a hand-made POST:
        # the two shapes the bench has always accepted, read off auth_mode.
        mode = AuthMode(form.get("auth_mode") or AuthMode.SUBSCRIPTION.value)
        choice = "env" if mode is AuthMode.API_KEY else "native-login"
    if choice == "env":
        env_name = str(form.get("api_key_env") or "").strip()
        if not env_name:
            raise ValueError(
                "metered (api_key) mode needs the NAME of an environment variable "
                "holding the key. The name is stored; the value never is"
            )
        return Credential.env(env_name)
    if choice == "secret":
        if not api_key.strip():
            raise ValueError(
                "the pasted-key form was chosen and no key was given. The key is "
                "written to the secrets file and the connection stores only a "
                "pointer to it"
            )
        return Credential.secret(api_key)
    if choice == "keep":
        entry = str(form.get("secret_entry") or "").strip()
        if not entry:
            raise ValueError(
                "there is no stored key on this connection to keep; choose a "
                "credential form"
            )
        if previous is None:
            raise ValueError("a stored key can only be kept on a connection that has one")
        return Credential.stored_secret(entry)
    if choice == "none":
        return Credential.none()
    return Credential.native_login()


def _optional_int(value: Any, label: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{label} must be a whole number of tokens, got {text!r}") from None


# --- serving ----------------------------------------------------------------------


def serve(
    *,
    port: int = DEFAULT_PORT,
    bridge: Bridge | None = None,
    debug: bool = False,
) -> None:
    """Run the bench on the loopback interface. Blocks until interrupted."""
    app = create_app(bridge=bridge)
    app.run(host=HOST, port=port, debug=debug, threaded=True)
