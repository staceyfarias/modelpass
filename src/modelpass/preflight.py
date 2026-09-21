"""Auth preflight -- the trust core (D1, D2).

Everything in this module answers one question before a run starts: *which auth
mode is this runtime actually about to use?* Getting it wrong is silent and
expensive, because two of the three providers fail toward metered billing:

* Anthropic resolves subscription OAuth **last**, behind ``ANTHROPIC_API_KEY``,
  ``ANTHROPIC_AUTH_TOKEN``, ``apiKeyHelper`` and the cloud-provider switches. A
  stray key in the environment silently wins.
* The Codex SDK injects ``CODEX_API_KEY`` into the CLI environment and OpenAI's
  CI docs encourage setting it.
* Google alone fails closed: an unauthenticated headless run errors out.

So declining to *set* a key is not enough. modelpass computes the child
environment explicitly, removing every credential the connection did not name,
and reports what it removed. The computation is a pure function of (connection,
environment): no vendor package, no network, no credentials, fully unit-testable
offline. Vendor-specific *detection* (which account, which plan, is the runtime
even installed) is the adapter's half, and lands in the same :class:`Receipt`.

Provider facts here are dated 2026-08-15; re-verify before implementing against
them.

**The API runtimes changed the shape of that answer, not the rule** (2026-09-13).
``anthropic-api``, ``openai-api``, ``google-api`` and ``openai-compatible``
launch nothing: the adapter constructs an HTTP client inside this process. So
:func:`plan_launch` produces an *empty* environment for them -- a scrubbed
environment nobody launches is worse than none, because the obvious thing to do
with a dict of environment variables is hand it to something, and every vendor
SDK's own discovery would then read a key the connection never named. The rule
D2 states survives intact and gets sharper:

    An API adapter constructs its client with an **explicit** credential and
    must never permit the SDK's own environment discovery.
    ``anthropic.Anthropic(api_key=...)``,
    ``openai.OpenAI(api_key=..., base_url=...)`` -- never the bare constructor.

That is rule 1 of the adapter contract restated for a runtime with no child
process, and it rides on the receipt as a :class:`Directive` rather than living
only in prose. :func:`api_preflight` is the API-runtime entry point and
:func:`resolve_credential` is how the value reaches a constructor without ever
entering a mapping.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .connections import Connection, CredentialKind, parse_base_url
from .errors import (
    AdapterNotImplemented,
    AuthModeMismatch,
    InvalidConnection,
    NoSuchSecret,
    PreflightFailed,
    UnsafeLaunch,
)
from .runtimes import API_RUNTIMES, Runtime
from .types import AuthMode

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, not at type time
    from .secrets import SecretSource

__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_CACHE_FLOOR_TOKENS",
    "FORBIDDEN_LAUNCH_ARGS",
    "OPUS_5_CACHE_FLOOR_TOKENS",
    "SCRUB_RULES",
    "AccountProfile",
    "CacheEligibility",
    "Directive",
    "ModelListProbe",
    "PreflightPlan",
    "Receipt",
    "ScrubRule",
    "api_preflight",
    "cache_floor_tokens",
    "check_launch_args",
    "credential_fingerprint",
    "directives_for",
    "env_names_to_scrub",
    "passthrough_env_names",
    "plan_launch",
    "preserved_env_names",
    "resolve_credential",
    "scrub_env",
]


@dataclass(frozen=True, slots=True)
class AccountProfile:
    """Safe, vendor-reported account identity metadata.

    Deliberately incapable of holding access tokens, refresh tokens, API keys,
    cookies, or arbitrary vendor payloads. Adapters normalize only the fields a
    person needs to distinguish one configured account from another.
    """

    vendor: str
    source: str
    logged_in: bool | None = None
    auth_method: str | None = None
    api_provider: str | None = None
    email: str | None = None
    organization_id: str | None = None
    organization_name: str | None = None
    subscription_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "vendor": self.vendor,
            "source": self.source,
            "logged_in": self.logged_in,
            "auth_method": self.auth_method,
            "api_provider": self.api_provider,
            "email": self.email,
            "organization_id": self.organization_id,
            "organization_name": self.organization_name,
            "subscription_type": self.subscription_type,
        }


@dataclass(frozen=True, slots=True)
class ScrubRule:
    """Environment variables to remove from a runtime's child process."""

    names: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    reason: str = ""


#: What each runtime scavenges if we let it. Two of the three runtimes fail
#: *toward* metered billing when a stray variable is present, which is the
#: 2026-08-15 auth-mode finding this table exists to act on; see
#: docs/api-and-runtimes.md §2.2a.
SCRUB_RULES: dict[Runtime, tuple[ScrubRule, ...]] = {
    Runtime.ANTHROPIC_SDK: (
        ScrubRule(
            names=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
            reason="outranks subscription OAuth in the Claude Code auth precedence",
        ),
        ScrubRule(
            prefixes=("CLAUDE_CODE_USE_",),
            reason="cloud provider credentials (Bedrock/Vertex/Foundry) outrank everything",
        ),
        ScrubRule(
            names=("ANTHROPIC_PROFILE",),
            reason="profile/federation credentials outrank subscription OAuth",
        ),
        ScrubRule(
            names=("CLAUDE_CODE_OAUTH_TOKEN",),
            reason="a subscription token the connection did not reference is still an "
            "ambient credential (D2)",
        ),
        ScrubRule(
            names=("CLAUDE_CONFIG_DIR",),
            reason="selects a Claude Code credential store and therefore an account; "
            "honoured only when configDir is written on the connection",
        ),
        ScrubRule(
            names=("ANTHROPIC_BASE_URL",),
            reason="not a credential, but it silently redirects where a subscription "
            "token is sent; nothing ambient may shape a run (D2 extension, 2026-08-17). "
            "Honoured when a connection names it in allowEnv",
        ),
    ),
    Runtime.OPENAI_SDK: (
        ScrubRule(
            names=("CODEX_API_KEY", "OPENAI_API_KEY"),
            reason="the Codex SDK injects CODEX_API_KEY into the CLI environment",
        ),
        ScrubRule(
            names=("CODEX_HOME",),
            reason="selects a Codex credential store and therefore an account; "
            "honoured only when configDir is written on the account profile",
        ),
    ),
    Runtime.GOOGLE_CLI: (
        ScrubRule(
            names=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            reason="with modelProvider set, the agy CLI will use an API key instead of "
            "the Google account login",
        ),
        ScrubRule(
            names=("GOOGLE_APPLICATION_CREDENTIALS",),
            reason="Vertex ADC is metered billing",
        ),
    ),
    Runtime.GOOGLE_SDK: (
        ScrubRule(
            names=("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"),
            reason="only the credential the connection references may be used (D2)",
        ),
    ),
}

#: Arguments modelpass will never pass to a runtime.
FORBIDDEN_LAUNCH_ARGS: dict[Runtime, tuple[str, ...]] = {
    # --bare never reads OAuth credentials and requires an API key. The docs warn
    # it may become the default for -p in a future release: re-check on upgrade.
    Runtime.ANTHROPIC_SDK: ("--bare",),
    Runtime.OPENAI_SDK: (),
    Runtime.GOOGLE_CLI: (),
    Runtime.GOOGLE_SDK: (),
}


@dataclass(frozen=True, slots=True)
class Directive:
    """A non-environment instruction the adapter must apply at launch.

    Some of the enforcement surface is not environment variables: a settings-file
    key, an SDK option, a CLI flag. Modelling those as data keeps them visible in
    the receipt and testable without a vendor package.
    """

    name: str
    value: str
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11). Directives ride along inside a receipt."""
        return {"name": self.name, "value": self.value, "reason": self.reason}


def directives_for(connection: Connection) -> tuple[Directive, ...]:
    """Launch directives implied by a connection's runtime and auth mode."""
    runtime = connection.runtime
    subscription = connection.auth_mode is AuthMode.SUBSCRIPTION
    out: list[Directive] = []

    if runtime is Runtime.ANTHROPIC_SDK:
        out.append(
            Directive(
                "settings.apiKeyHelper",
                "disabled",
                "apiKeyHelper outranks subscription OAuth and lives in settings.json, "
                "not the environment",
            )
        )
        out.append(
            Directive(
                "persistSession",
                "false",
                "v1 chat is stateless: each call runs an ephemeral session (D7)",
            )
        )
    elif runtime is Runtime.OPENAI_SDK:
        out.append(
            Directive(
                "forced_login_method",
                "chatgpt" if subscription else "apikey",
                "pins the Codex runtime to one auth mode; the cleanest enforcement "
                "mechanism of the three providers",
            )
        )
        out.append(
            Directive("--ephemeral", "true", "no persisted session files for a stateless call")
        )
    elif runtime is Runtime.GOOGLE_CLI:
        out.append(
            Directive(
                "settings.modelProvider",
                "unset",
                "setting modelProvider = gemini moves the CLI onto an API key",
            )
        )
    elif runtime in API_RUNTIMES:
        # The in-process form of D2, stated as a directive so it is on the
        # receipt rather than in a docstring somebody may not have read. There is
        # no child process to scrub, so the enforcement moves to the one line of
        # adapter code that matters: how the client is constructed.
        out.append(
            Directive(
                "client.credential",
                "explicit",
                "the vendor client is constructed with the credential this "
                "connection names, passed as an argument -- anthropic.Anthropic("
                "api_key=...), openai.OpenAI(api_key=..., base_url=...), never the "
                "bare constructor",
            )
        )
        out.append(
            Directive(
                "sdk_environment_discovery",
                "never",
                "every vendor SDK falls back to reading its key from os.environ. "
                "A machine with an ambient OPENAI_API_KEY and a connection naming "
                "a different credential would bill the wrong account silently, "
                "which is the same failure the environment scrub prevents for a "
                "child process (D2, in-process clause 2026-09-13)",
            )
        )
        if connection.base_url is not None:
            out.append(
                Directive(
                    "client.base_url",
                    connection.base_url,
                    "the endpoint this connection names; safe to report because a "
                    "baseUrl may not carry credentials (see parse_base_url)",
                )
            )

    return tuple(out)


def preserved_env_names(connection: Connection) -> tuple[str, ...]:
    """Environment variables the connection explicitly references, so must survive.

    This is the whole of D2 in one function: a credential is used when, and only
    when, a connection names it.

    Only *credential* pointers count here, because the caller of this function is
    also the check for "the credential you named is missing". Non-credential
    passthroughs go through :func:`passthrough_env_names` instead, so an unset
    ``ANTHROPIC_BASE_URL`` is simply not passed rather than a failed preflight.
    """
    ref = connection.credential_ref
    if ref.kind is CredentialKind.ENV and ref.locator:
        return (ref.locator,)
    return ()


def passthrough_env_names(connection: Connection) -> tuple[str, ...]:
    """Non-credential variables the connection explicitly allowed through the scrub.

    Empty unless a connection names something in ``allowEnv``, which is the
    mechanism that keeps a variable like ``ANTHROPIC_BASE_URL`` from being either
    silently obeyed or silently dropped: it is scrubbed by default and honoured
    only on a written-down instruction (D2 extension, 2026-08-17).
    """
    return tuple(connection.allow_env)


def env_names_to_scrub(
    runtime: Runtime,
    env: Mapping[str, str],
    *,
    preserve: Sequence[str] = (),
) -> tuple[str, ...]:
    """Which variables present in ``env`` this runtime must not see."""
    keep = set(preserve)
    hits: list[str] = []
    for rule in SCRUB_RULES.get(runtime, ()):
        for name in rule.names:
            if name in env and name not in keep and name not in hits:
                hits.append(name)
        for prefix in rule.prefixes:
            for name in env:
                if name.startswith(prefix) and name not in keep and name not in hits:
                    hits.append(name)
    return tuple(sorted(hits))


def scrub_env(
    runtime: Runtime,
    env: Mapping[str, str],
    *,
    preserve: Sequence[str] = (),
) -> dict[str, str]:
    """The environment to launch this runtime with. Pure; does not touch os.environ."""
    removed = set(env_names_to_scrub(runtime, env, preserve=preserve))
    return {name: value for name, value in env.items() if name not in removed}


def check_launch_args(runtime: Runtime, args: Sequence[str]) -> None:
    """Raise :class:`UnsafeLaunch` if a forbidden argument is present."""
    forbidden = set(FORBIDDEN_LAUNCH_ARGS.get(runtime, ()))
    for arg in args:
        head = arg.split("=", 1)[0]
        if head in forbidden:
            raise UnsafeLaunch(
                f"argument {head!r} is never passed to runtime {runtime.value!r}: it "
                "bypasses subscription credentials and forces metered billing"
            )


@dataclass(frozen=True, slots=True)
class PreflightPlan:
    """The vendor-independent half of the preflight: what the launch will look like.

    Computed from the connection and an environment mapping alone.

    **Deliberately has no ``to_dict``**, and is the one public type that does not
    (D11 otherwise applies everywhere). ``env`` holds the *values* of the child
    environment, so a serialized plan is a credential dump waiting to be logged.
    Everything about a plan that is safe to publish is already on the
    :class:`Receipt`, which is the object built to be shown.
    """

    connection: str
    runtime: Runtime
    auth_mode: AuthMode
    env: Mapping[str, str] = field(default_factory=dict)
    scrubbed: tuple[str, ...] = ()
    preserved: tuple[str, ...] = ()
    #: Non-credential variables kept because ``allowEnv`` named them, and which
    #: of those were actually present in the environment.
    passthrough: tuple[str, ...] = ()
    #: Per-account vendor configuration root. Safe to report (it is a path, not
    #: a credential) and separate from ``passthrough`` because it is a
    #: first-class account selector, not an allowEnv escape hatch.
    config_dir: str | None = None
    directives: tuple[Directive, ...] = ()
    forbidden_args: tuple[str, ...] = ()
    credential_present: bool = True
    #: The model this run would ask for, and where that name came from. Both are
    #: carried on the plan so every adapter's receipt reports them without each
    #: adapter having to remember to (first-consumer feedback, 2026-08-17). An
    #: adapter may still *fill* them when the plan does not know -- reading the
    #: vendor's own configured default, which is knowledge only the adapter has.
    model: str | None = None
    model_source: str = ""
    #: Whether the connection has any spend threshold in force. Carried on the
    #: plan so the receipt can say "no spend guards configured" without needing
    #: the connection object (D4 amendment, 2026-08-17).
    guards_configured: bool = False
    #: This connection's retry stance (ticket 1.12, 2026-09-13). ``"default"``
    #: or ``"never"``; carried for the reason ``guards_configured`` is, so the
    #: receipt can disclose it before the run rather than leaving a caller to
    #: infer it from a verdict afterwards.
    retry: str = "default"

    @property
    def config_env_name(self) -> str | None:
        """Vendor variable carrying this account's isolated config root."""
        if self.config_dir is None:
            return None
        if self.runtime is Runtime.ANTHROPIC_SDK:
            return "CLAUDE_CONFIG_DIR"
        if self.runtime is Runtime.OPENAI_SDK:
            return "CODEX_HOME"
        return None

    @property
    def kept(self) -> tuple[str, ...]:
        """Every variable that survives the scrub by explicit instruction."""
        configured = (self.config_env_name,) if self.config_env_name else ()
        return (*self.preserved, *self.passthrough, *configured)

    @property
    def launches_a_process(self) -> bool:
        """Whether :attr:`env` describes a child process anybody will start.

        ``False`` on the API runtimes, where it is an empty mapping on purpose --
        see :func:`plan_launch`.
        """
        return self.runtime not in API_RUNTIMES

    def require_base_url(self, connection: Connection | None = None) -> None:
        """Re-run the base-URL check as a preflight, raising :class:`PreflightFailed`.

        The same validation :class:`~modelpass.connections.Connection` runs at
        construction, run again at the point of use and reported as a preflight
        failure rather than as an invalid connection. It is not redundant: the
        connection file is a text file people edit by hand and a store built by
        an older build carries keys it did not validate, so "this was checked
        when the object was made" is only true of objects this build made.
        """
        target = connection.base_url if connection is not None else None
        if target is None:
            return
        try:
            parse_base_url(target, where=f"connection {self.connection!r}: baseUrl")
        except InvalidConnection as exc:
            raise PreflightFailed(str(exc)) from exc

    def require_credential(self) -> None:
        """Raise if the connection references an environment variable that is absent."""
        if not self.credential_present:
            missing = ", ".join(self.preserved) or "the referenced credential"
            raise PreflightFailed(
                f"connection {self.connection!r} references {missing}, which is not set "
                "in the environment"
            )

    def describe(self) -> str:
        parts = [
            f"{self.connection}: {self.runtime.value} in {self.auth_mode.value} mode",
        ]
        if self.scrubbed:
            parts.append("scrubbed " + ", ".join(self.scrubbed))
        if self.kept:
            parts.append("kept " + ", ".join(self.kept))
        return "; ".join(parts)


def plan_launch(
    connection: Connection, env: Mapping[str, str] | None = None
) -> PreflightPlan:
    """Compute the launch plan for a connection. Pure, offline, vendor-free.

    **On an API runtime this plans no environment at all**, and the empty
    mapping is the decision rather than a gap (2026-09-13). Nothing is launched:
    the adapter constructs an HTTP client in this process. A scrubbed copy of
    ``os.environ`` would be worse than useless here -- it is a plausible-looking
    dict sitting on a request object, and the obvious thing to do with a dict of
    environment variables is hand it to something, at which point the vendor
    SDK's own discovery reads a key nobody referenced out of it. So the plan
    carries nothing to hand over, and the rule that replaces the scrub is a
    directive on the receipt: the client is constructed with an explicit
    credential and the SDK never reads the process environment.

    The *credential* is still resolved from the environment where the connection
    says so -- ``preserved`` and ``credential_present`` are filled exactly as
    before, which is what keeps :meth:`PreflightPlan.require_credential` working
    on both families. What changes is that the value reaches the adapter through
    :func:`resolve_credential`, one string at the moment of use, rather than
    riding along inside a mapping.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    preserved = preserved_env_names(connection)
    if connection.runtime in API_RUNTIMES:
        return PreflightPlan(
            connection=connection.name,
            runtime=connection.runtime,
            auth_mode=connection.auth_mode,
            env={},
            scrubbed=(),
            preserved=preserved,
            passthrough=(),
            config_dir=None,
            directives=directives_for(connection),
            forbidden_args=(),
            credential_present=all(name in source for name in preserved),
            guards_configured=connection.guards.configured,
            retry=connection.retry,
            model=connection.model,
            model_source=(
                f"from connection {connection.name!r}" if connection.model else ""
            ),
        )
    allowed = passthrough_env_names(connection)
    config_name = None
    if connection.config_dir is not None:
        if connection.runtime is Runtime.ANTHROPIC_SDK:
            config_name = "CLAUDE_CONFIG_DIR"
        elif connection.runtime is Runtime.OPENAI_SDK:
            config_name = "CODEX_HOME"
    configured = (config_name,) if config_name else ()
    keep = (*preserved, *allowed, *configured)
    scrubbed = env_names_to_scrub(connection.runtime, source, preserve=keep)
    child_env = scrub_env(connection.runtime, source, preserve=keep)
    if config_name and connection.config_dir is not None:
        child_env[config_name] = connection.config_dir
    return PreflightPlan(
        connection=connection.name,
        runtime=connection.runtime,
        auth_mode=connection.auth_mode,
        env=child_env,
        scrubbed=scrubbed,
        preserved=preserved,
        # Report only what was actually there: "kept ANTHROPIC_BASE_URL" on a
        # machine that has no ANTHROPIC_BASE_URL would be a receipt describing a
        # run that is not happening.
        passthrough=tuple(name for name in allowed if name in source),
        config_dir=connection.config_dir,
        directives=directives_for(connection),
        forbidden_args=FORBIDDEN_LAUNCH_ARGS.get(connection.runtime, ()),
        credential_present=all(name in source for name in preserved),
        guards_configured=connection.guards.configured,
        retry=connection.retry,
        model=connection.model,
        model_source=(
            f"from connection {connection.name!r}" if connection.model else ""
        ),
    )


def resolve_credential(
    connection: Connection,
    env: Mapping[str, str] | None = None,
    *,
    secrets: SecretSource | None = None,
) -> str:
    """The credential value this connection names. One string, at the moment of use.

    The in-process counterpart to the environment scrub: an API adapter calls
    this and passes the result straight into a client constructor, so the
    credential never sits in a mapping that something else may copy, serialize
    or hand to an SDK's own discovery. :class:`PreflightPlan` deliberately has no
    ``to_dict`` for that reason and this function is deliberately not a field on
    it for the same one.

    Raises :class:`PreflightFailed` when the reference resolves to nothing, which
    is the first of the three API preflight checks.

    ``secret:<entry>`` is read **here**, at the moment of use, from the
    :class:`~modelpass.secrets.SecretStore` (default: the one beside the
    connection file). It is deliberately not resolved earlier and cached: the
    value would then sit on some object for the lifetime of a run, and every
    object in this module that could hold one is the object something else
    serializes. ``secrets`` is injectable so a caller with a non-default home --
    a test, an app carrying its own store root -- resolves against its own file
    rather than the user's.

    ``keychain:`` remains declared and unimplemented; the message says so rather
    than guessing at a platform.
    """
    ref = connection.credential_ref
    if ref.kind is CredentialKind.NONE:
        # Refused here rather than answered with a placeholder, and the reason is
        # that this function's whole contract is "the credential this connection
        # names". A connection that names none has no answer to give, and handing
        # back a string that is not a credential would make every caller of this
        # function -- the receipt's fingerprint among them -- report a key that
        # does not exist. The one adapter that accepts such a connection checks
        # the kind before it asks (ticket 1.10).
        raise PreflightFailed(
            f"connection {connection.name!r} declares no credential "
            "(credentialRef 'none'), so there is none to resolve. That is a valid "
            "openai-compatible configuration and its adapter handles it; anything "
            "asking this function for a key on such a connection is asking the "
            "wrong question"
        )
    if ref.kind is CredentialKind.NATIVE_LOGIN:
        raise PreflightFailed(
            f"connection {connection.name!r} points at the runtime's own login "
            "store, which an API runtime does not have; an api_key connection "
            "must name a credential"
        )
    if ref.kind is CredentialKind.ENV:
        source: Mapping[str, str] = os.environ if env is None else env
        value = source.get(ref.locator or "", "")
        if not value.strip():
            raise PreflightFailed(
                f"connection {connection.name!r} references {ref.describe()}, which "
                "is not set in the environment (or is empty)"
            )
        return value
    if ref.kind is CredentialKind.SECRET:
        from .secrets import SecretStore

        store = secrets if secrets is not None else SecretStore()
        try:
            value = store.get(ref.locator or "")
        except NoSuchSecret as exc:
            raise PreflightFailed(str(exc)) from exc
        if not value.strip():
            raise PreflightFailed(
                f"connection {connection.name!r} references {ref.describe()}, which "
                "holds no value"
            )
        return value
    raise AdapterNotImplemented(
        f"connection {connection.name!r} references {ref.describe()}, and modelpass "
        "cannot resolve that yet: nothing in the package reads a platform keychain. "
        "Use env:NAME or secret:<entry> -- an unresolvable reference is reported here "
        "rather than guessed at"
    )


def credential_fingerprint(value: str) -> str:
    """A stable, non-reversible name for a key: ``sha256:`` plus eight hex digits.

    What the receipt's ``account`` field may carry on an API runtime, and the
    only thing about a credential that is ever allowed onto a receipt, into the
    run log or through a ``to_dict()``. Eight hex digits is enough to tell two
    configured keys apart and to notice that a key was rotated, which is the
    whole question a person asks of this field; it is not enough to be worth
    attacking, and sha256 is one-way regardless.

    R14's rule stated positively: **no secret value ever reaches a receipt, the
    run log, ``to_dict()`` or a vendor event; a sha256 fingerprint may.**
    """
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:8]}"


@dataclass(frozen=True, slots=True)
class ModelListProbe:
    """The optional third API preflight check: *is this credential live?*

    A model-list call (``GET /v1/models``, ``models.list()``) is token-free on
    every vendor and is the only pre-run check that can prove a key works before
    anything is spent. It is **optional** because it is not free of *time*: it is
    a network round trip, :meth:`~modelpass.Bridge.preflight` runs on every single
    call, and paying a round trip per chat to re-learn a fact that changes
    monthly is a bad trade.

    So the shape is the one the identity probes already use: the adapter caches
    it (:meth:`~modelpass.adapters.base.Adapter.cached_probe`) and
    ``modelpass verify`` drops the cache through
    :meth:`~modelpass.Bridge.refresh_identity`, which is the caller whose entire
    job is to re-read the live answer.

    ``ok=False`` is a real finding and belongs on the receipt; ``None`` from
    :meth:`~modelpass.adapters.base.Adapter.probe` means nobody asked, which is
    the third state and not a failure.
    """

    ok: bool
    detail: str = ""
    #: Model ids the endpoint reported, where it reported any. Useful on
    #: ``openai-compatible``, where the answer is also the only reliable way to
    #: find out what the thing on the other end actually serves.
    models: tuple[str, ...] = ()

    @property
    def note(self) -> str:
        """The receipt sentence for this probe."""
        if self.ok:
            count = f", {len(self.models)} model(s) listed" if self.models else ""
            return f"credential probe: the endpoint answered{count}"
        return f"credential probe FAILED: {self.detail or 'the endpoint did not answer'}"

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11). Carries no credential -- see the ``ok`` field."""
        return {"ok": self.ok, "detail": self.detail, "models": list(self.models)}


#: Characters per token, for the prefix-size estimate below. Four is the usual
#: rule of thumb for English prose and it is **stated everywhere it is used**,
#: because the alternative -- dividing by it and printing the quotient as a
#: token count -- would dress a guess up as a measurement. modelpass does not
#: tokenize: neither runtime exposes a tokenizer, the vendors' own counting
#: endpoints cost a request, and a receipt is a thing that runs before anything
#: is spent. So the honest unit here is the one modelpass can actually count.
CHARS_PER_TOKEN = 4

#: The minimum cacheable prefix, in tokens, on most models.
DEFAULT_CACHE_FLOOR_TOKENS = 1024

#: The minimum on Opus 5, which is half the usual figure.
OPUS_5_CACHE_FLOOR_TOKENS = 512

#: Model-name fragments that take the lower floor. Substring matching, because
#: a model arrives here as whatever the caller or the runtime called it -- a
#: dated slug, a family alias, an alias of an alias -- and an exact-match table
#: would silently fall back to the higher floor on every spelling nobody
#: anticipated. The failure direction is chosen deliberately: an unrecognized
#: name gets the *conservative* floor, so modelpass under-promises eligibility
#: rather than over-promising it.
_LOW_FLOOR_MODEL_FRAGMENTS = ("opus-5", "opus5")


def cache_floor_tokens(model: str | None) -> tuple[int, str]:
    """The minimum cacheable prefix for a model, and where the number came from.

    ``None`` -- no model named, so the runtime's own default applies and modelpass
    cannot know which it is -- takes the conservative floor and says so.
    """
    if not model:
        return (
            DEFAULT_CACHE_FLOOR_TOKENS,
            "no model is named for this run, so the higher of the two floors is "
            "assumed rather than guessed at",
        )
    lowered = model.lower()
    if any(fragment in lowered for fragment in _LOW_FLOOR_MODEL_FRAGMENTS):
        return OPUS_5_CACHE_FLOOR_TOKENS, f"the Opus 5 floor, from model {model!r}"
    return DEFAULT_CACHE_FLOOR_TOKENS, f"the usual floor, from model {model!r}"


@dataclass(frozen=True, slots=True)
class CacheEligibility:
    """Whether this call *can* cache its prefix -- not whether it will (D20).

    The distinction is the whole design of this type. Whether a call **hits**
    cache depends on a prefix that has not been sent yet, and on whether some
    earlier call sent the same bytes; no honest answer to that exists before the
    run. Whether a call is **eligible** is answerable now, from three facts that
    are all knowable pre-run: is there a system prompt at all, is it plausibly
    large enough to clear the runtime's minimum cacheable prefix, and is a TTL
    lever available on this runtime and this CLI build.

    **Why this is on the receipt at all.** The measured failure it exists to
    surface is silent and free to fall into: a sub-floor system prompt with
    varying messages reads *zero* and pays a full write on every single call,
    with no error and ``cache_creation_input_tokens: 0`` to look at. A consumer
    sat in exactly that state for a day without knowing. Nothing in the stream
    says so, which is precisely why it belongs to the object that runs before
    the stream.

    **Information, not a warning, and the phrasing is deliberate.** A user who
    drains an allowance without ever being told they paid full price every call
    is entitled to be angry -- and a user near the end of a resetting window,
    with allowance that expires unspent, may quite rationally choose the
    expensive path anyway. modelpass's job is to make that a decision rather than
    a discovery. A receipt that scolds the second user has misread its role, so
    nothing here is phrased as a fault, a recommendation, or a number to fix.

    **The line renders either way.** A disclosure that only appears when the
    answer is "yes" leaves everybody else guessing, which is the state this was
    built to end.
    """

    #: Characters in the system prompt this call will send; ``0`` when there is
    #: none. Characters, not tokens, because that is what modelpass can count.
    system_prompt_chars: int = 0
    #: The runtime's minimum cacheable prefix for this model, in tokens.
    floor_tokens: int = DEFAULT_CACHE_FLOOR_TOKENS
    #: Where :attr:`floor_tokens` came from, for a reader checking the reasoning.
    floor_source: str = ""
    #: Whether tools or MCP servers were declared. Tool definitions render
    #: *ahead* of the system block in the cached prefix, so they add to it -- by
    #: an amount modelpass cannot size, since only the runtime knows how it
    #: serializes them. Recorded so the estimate is never read as a ceiling.
    tools_declared: bool = False
    #: Whether the runtime's own preset sits at the front of this prefix, with
    #: the caller's text appended after it -- a ``WorkerSession``. When it does,
    #: eligibility does not turn on :attr:`system_prompt_chars` at all: the
    #: preset is a full agent prompt and clears any of these floors on its own.
    #: Judging such a session by the length of its *append* would report a
    #: short-instruction worker as uncacheable, which is both wrong and exactly
    #: the sort of confident-and-mistaken line this type exists to avoid.
    preset_prefix: bool = False
    #: The prefix cache TTL this call pins, or ``None`` when modelpass pins none
    #: and the runtime's own default applies.
    ttl: str | None = None
    #: Why the TTL reads the way it does -- no lever on this runtime, a CLI too
    #: old to read the one that exists, or the pin modelpass actually set.
    ttl_detail: str = ""
    #: How many ``cache_control`` breakpoints the caller placed on this call's
    #: content (R3). Non-zero turns every line below from an estimate into a
    #: **fact**: the heuristic exists to guess where a prefix probably ends, and
    #: a caller who said where it ends has removed the need to guess.
    explicit_breakpoints: int = 0
    #: Whether the runtime this call is going to takes those breakpoints, from
    #: :attr:`~modelpass.capabilities.Capability.CACHE_BREAKPOINTS`. ``False``
    #: with :attr:`explicit_breakpoints` non-zero is the case worth shouting
    #: about: the caller asked, and modelpass is flattening the request.
    breakpoints_honoured: bool = False

    @property
    def has_breakpoints(self) -> bool:
        """Whether the caller marked the prefix boundary explicitly."""
        return self.explicit_breakpoints > 0

    @property
    def breakpoint_clause(self) -> str:
        """The explicit-breakpoint fact, in one clause. ``""`` when there are none."""
        if not self.has_breakpoints:
            return ""
        count = self.explicit_breakpoints
        plural = "" if count == 1 else "s"
        if self.breakpoints_honoured:
            return (
                f"cache: {count} explicit breakpoint{plural} on a runtime that "
                "honours them"
            )
        return (
            f"cache: {count} explicit breakpoint{plural} requested, will be dropped"
        )

    @property
    def floor_chars(self) -> int:
        """:attr:`floor_tokens` restated in the unit this type can count."""
        return self.floor_tokens * CHARS_PER_TOKEN

    @property
    def has_prefix(self) -> bool:
        """Whether there is a system prompt to be stable *about*."""
        return self.preset_prefix or self.system_prompt_chars > 0

    @property
    def likely_eligible(self) -> bool:
        """Whether the system block plausibly clears the floor.

        **Likely**, and the word is load-bearing: this compares an estimate
        against a threshold, and a prompt near the line could fall either side
        of it. It is reliable at the two ends, which is where it matters -- a
        300-character rubric is not close, and a 20,000-character one is not
        close either.
        """
        if self.preset_prefix:
            return True
        return self.system_prompt_chars > 0 and self.system_prompt_chars >= self.floor_chars

    @property
    def clause(self) -> str:
        """The short form, for the one-line receipt summary.

        **The heuristic is for plain-string prompts only.** Where the caller
        placed breakpoints, this reports what they placed and whether it
        survives the trip -- a measured fact rather than a character count
        compared against a floor (R3). The floor arithmetic is still in
        :attr:`note` underneath, because a breakpoint below the floor is still
        below the floor; what it stops being is the *headline*.
        """
        if self.has_breakpoints:
            return self.breakpoint_clause
        if not self.has_prefix:
            return "cache: no system prompt, so no reusable prefix"
        if self.likely_eligible:
            ttl = f", TTL {self.ttl}" if self.ttl else ""
            return f"cache: prefix likely eligible{ttl}"
        return "cache: prefix likely below the caching floor"

    @property
    def note(self) -> str:
        """The full sentence, always available and never ``None``.

        Written to be read once by somebody deciding how to structure a loop,
        not skimmed every run -- which is why the short :attr:`clause` is what
        goes on the summary line and this is what sits underneath it.
        """
        estimate = (
            f"estimated at ~{CHARS_PER_TOKEN} characters per token, which is a "
            "rule of thumb and not a token count -- modelpass does not tokenize"
        )
        if self.preset_prefix:
            appended = (
                f", with {self.system_prompt_chars:,} characters of your own appended "
                "after it"
                if self.system_prompt_chars
                else ", and nothing of your own is appended to it"
            )
            body = (
                "the runtime's own preset is the front of this prefix"
                + appended
                + f". That is a full agent prompt and clears the {self.floor_tokens:,}"
                "-token minimum on its own, so eligibility here does not depend on how "
                "long your text is. What it does depend on is that front staying "
                "byte-identical between runs, which is what fixing the tools and the "
                "system prompt at construction is for"
            )
        elif not self.has_prefix:
            body = (
                "this call sends no system prompt, so there is no stable prefix for "
                "the runtime to reuse and its input is priced in full every time. "
                "That is the right shape for a one-off question and the wrong one "
                "for a loop: passing the instructions as system_prompt= rather than "
                "concatenating them into the message is what gives a loop something "
                "cacheable"
            )
        elif self.likely_eligible:
            body = (
                f"the system prompt is {self.system_prompt_chars:,} characters, which "
                f"should clear the ~{self.floor_chars:,} needed for this runtime's "
                f"{self.floor_tokens:,}-token minimum cacheable prefix ({estimate}). "
                "Hold it byte-identical across calls and later calls should read it "
                "back rather than paying for it again; read cached_input_tokens to "
                "confirm, not the writes"
            )
        else:
            body = (
                f"the system prompt is {self.system_prompt_chars:,} characters, below "
                f"the ~{self.floor_chars:,} needed for this runtime's "
                f"{self.floor_tokens:,}-token minimum cacheable prefix ({estimate}). "
                "Expect a full write and no read on every call, silently and with no "
                "error: below the floor there is nothing to reuse, so holding the "
                "prompt stable buys nothing on its own. Growing it past the floor is "
                "what turns a stable prefix into a reused one, and a judge or rubric "
                "prompt usually wants to be that long anyway"
            )
        parts = []
        if self.has_breakpoints:
            count = self.explicit_breakpoints
            plural = "" if count == 1 else "s"
            parts.append(
                f"you placed {count} explicit cache breakpoint{plural} on this "
                "call's content, so where the cacheable prefix ends is stated "
                "rather than estimated"
                if self.breakpoints_honoured
                else (
                    f"you placed {count} explicit cache breakpoint{plural} on this "
                    "call's content and this runtime does not accept them, so the "
                    "content is flattened to text and the markers do not reach the "
                    "vendor. Whatever caching happens is the runtime's own, decided "
                    "without your input"
                )
            )
        parts.append(body)
        if self.floor_source and self.has_prefix and not self.preset_prefix:
            # Only where the floor is what decided the verdict. Naming which
            # floor applies underneath "there is no system prompt" answers a
            # question nobody asked and buries the one sentence that mattered.
            parts.append(self.floor_source)
        if self.tools_declared:
            parts.append(
                "tools are declared, and their definitions render ahead of the system "
                "block in the same prefix, so the real prefix is larger than this by "
                "an amount only the runtime can measure"
            )
        if self.ttl_detail:
            parts.append(self.ttl_detail)
        return "; ".join(parts)

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11)."""
        return {
            "system_prompt_chars": self.system_prompt_chars,
            "chars_per_token": CHARS_PER_TOKEN,
            "floor_tokens": self.floor_tokens,
            "floor_chars": self.floor_chars,
            "floor_source": self.floor_source,
            "tools_declared": self.tools_declared,
            "preset_prefix": self.preset_prefix,
            "ttl": self.ttl,
            "ttl_detail": self.ttl_detail,
            "has_prefix": self.has_prefix,
            "likely_eligible": self.likely_eligible,
            "explicit_breakpoints": self.explicit_breakpoints,
            "breakpoints_honoured": self.breakpoints_honoured,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class Receipt:
    """What the preflight actually found -- the thing a user is asked to trust.

    ``modelpass connect`` prints this at setup time and every run recomputes it, so
    "which subscription am I about to spend" is answerable before the first token
    rather than on next month's bill.
    """

    connection: str
    runtime: Runtime
    requested_auth_mode: AuthMode
    detected_auth_mode: AuthMode | None = None
    credential_source: str = ""
    account: str | None = None
    plan_name: str | None = None
    account_profile: AccountProfile | None = None
    identity_verified: bool | None = None
    runtime_available: bool = True
    scrubbed: tuple[str, ...] = ()
    preserved: tuple[str, ...] = ()
    passthrough: tuple[str, ...] = ()
    config_dir: str | None = None
    directives: tuple[Directive, ...] = ()
    notes: tuple[str, ...] = ()
    ok: bool = True
    problem: str | None = None
    #: Whether any spend threshold is in force. ``False`` is the normal state
    #: for a fresh connection, because modelpass ships no default thresholds (D4
    #: amendment, 2026-08-17) -- which is exactly why it is a first-class field
    #: rather than a note string an adapter could overwrite.
    guards_configured: bool = False
    #: This connection's retry stance (ticket 1.12, 2026-09-13). ``"never"``
    #: forces every :class:`~modelpass.types.Retryable` verdict on this
    #: connection to ``no``. It does **not** make modelpass retry -- modelpass
    #: never does -- and it is on the receipt so a caller can see the stance
    #: before spending rather than deduce it from a terminal afterwards.
    retry: str = "default"
    #: Which model this run will ask for, when that is knowable before the run,
    #: and where the name came from (first-consumer feedback, 2026-08-17). The
    #: consumer hit a vendor model rejection and had nothing pre-run that named
    #: a model, so the receipt could not be used to diagnose the one thing that
    #: went wrong. Precedence, most specific first: what the caller passed to
    #: ``chat`` / ``preflight``, then the connection's own ``model``, then
    #: whatever the vendor runtime is configured to default to -- which only an
    #: adapter can read, and which modelpass reads and *never* writes.
    #:
    #: ``None`` is a real answer and means the runtime's own default applies;
    #: :attr:`model_note` is the sentence for it.
    model: str | None = None
    model_source: str = ""
    #: The executable this run resolves to launch, filled by the adapters that
    #: launch one by path. ``None`` where the runtime's own SDK owns its
    #: subprocess -- ``anthropic-sdk`` never sees a path to name -- and that
    #: absence is honest rather than missing data: nothing here went looking and
    #: failed. Recorded because two Codex builds routinely coexist on one
    #: machine, and a transport-level defect in one of them makes "was *this*
    #: run affected" a question only a named binary can answer (2026-08-31).
    binary: str | None = None
    #: Whether this call's prefix *can* cache, and at what TTL (D20). Filled by
    #: the adapter, because the two facts it needs -- the model's floor and
    #: whether a TTL lever exists on this runtime and this CLI build -- are
    #: vendor knowledge. ``None`` means no adapter answered, which is the third
    #: state and not a "no": a receipt that reported "no cache" for a runtime
    #: nobody asked would be inventing a verdict, and the whole point of this
    #: field is that a silence is what people were left with before.
    cache: CacheEligibility | None = None
    #: How many ``cache_control`` breakpoints this call's content carries, and
    #: how many of them reach the vendor (R3, 2026-09-13). Two numbers rather
    #: than one because they are two different facts and the interesting case is
    #: exactly where they disagree: a consumer that builds a four-breakpoint
    #: prompt against a runtime with no breakpoint channel reads ``4`` and ``0``,
    #: which is the answer it could not get before -- the markers used to be
    #: stripped at the door with nothing anywhere saying so. Both are ``0`` for
    #: every plain-string call, which is every call written before this existed.
    cache_breakpoints_requested: int = 0
    cache_breakpoints_honoured: int = 0
    #: What the connection's ``promptCache`` asked for, and how this runtime
    #: meets it (2026-09-21). Both ``None`` on every connection that stated
    #: nothing, which is every connection written before the key existed.
    #:
    #: Two fields rather than one, because the interesting thing is not that
    #: caching was requested but **which of two very different things that
    #: bought**: ``"explicit"`` means the runtime takes an instruction and a
    #: caller's ``cache_control`` breakpoints reach it, ``"automatic"`` means
    #: the runtime was caching anyway and the request was already met before it
    #: was made. A caller that cannot tell those apart cannot tell whether its
    #: breakpoints are doing anything. The third outcome -- the runtime can
    #: neither be instructed nor shown to cache -- never reaches a receipt: it
    #: is an ``InvalidConnection`` raised when the connection is built.
    prompt_cache_requested: str | None = None
    prompt_cache_disposition: str | None = None
    #: What this run asked the model to do about word choice, and what will
    #: actually be sent (R5, ticket 1.7). Two dicts rather than one flag,
    #: because the interesting case is exactly where they disagree and a
    #: boolean cannot say *which* field went missing: a caller that asked for
    #: ``temperature=0.2`` on a GPT-5 model reads ``{"temperature": 0.2}`` here
    #: and ``{"temperature": 1.0}`` there, which is the answer a RAG evaluation
    #: harness currently derives for itself and a desktop agent app gave up on
    #: getting.
    #:
    #: Both are in **modelpass's own field names**, never a vendor's, so one
    #: reader works across runtimes. Both are empty for every call that asked
    #: for nothing -- except where the wire protocol requires a field nobody
    #: set, which is disclosed rather than hidden: an ``anthropic-api`` call
    #: with no ceiling still carries a 4096-token one, and ``sampling_applied``
    #: says so.
    #:
    #: **Filled before the run, from the table rather than from the vendor.**
    #: The point is to be readable *before* spending, which is the same bargain
    #: :attr:`cache` and :attr:`model` make. What a vendor did with a value it
    #: accepted is not knowable here and is not claimed.
    sampling_requested: Mapping[str, Any] = field(default_factory=dict)
    sampling_applied: Mapping[str, Any] = field(default_factory=dict)
    #: One sentence per field that was dropped or coerced, and empty when
    #: nothing was. "temperature 0.2 requested; gpt-5 accepts only 1.0, sent
    #: 1.0"; "top_k not supported on openai-api, dropped". A silent drop is the
    #: failure this exists to end -- a desktop agent app stopped passing
    #: sampling to modelpass *at all* rather than route a value into a hole,
    #: which is what
    #: an unreported drop costs in practice (validation 2026-09-13, §1c).
    sampling_notes: tuple[str, ...] = ()

    @classmethod
    def from_plan(cls, plan: PreflightPlan, **overrides: object) -> Receipt:
        """Seed a receipt from the vendor-independent plan; adapters fill the rest."""
        base: dict[str, object] = {
            "connection": plan.connection,
            "runtime": plan.runtime,
            "requested_auth_mode": plan.auth_mode,
            "scrubbed": plan.scrubbed,
            "preserved": plan.preserved,
            "passthrough": plan.passthrough,
            "config_dir": plan.config_dir,
            "directives": plan.directives,
            "guards_configured": plan.guards_configured,
            "retry": plan.retry,
            "model": plan.model,
            "model_source": plan.model_source,
        }
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

    @property
    def effective_auth_mode(self) -> AuthMode:
        return self.detected_auth_mode or self.requested_auth_mode

    def require_ok(self) -> None:
        if not self.ok:
            raise PreflightFailed(
                f"preflight failed for connection {self.connection!r}: "
                f"{self.problem or 'unknown reason'}"
            )
        if not self.runtime_available:
            raise PreflightFailed(
                f"runtime {self.runtime.value!r} is not available for connection "
                f"{self.connection!r}"
            )

    def require_auth_mode(self, expected: AuthMode | None = None) -> None:
        """Raise :class:`AuthModeMismatch` unless the detected mode is the asked-for one."""
        want = expected or self.requested_auth_mode
        got = self.detected_auth_mode
        if got is not None and got is not want:
            raise AuthModeMismatch(str(want), str(got), self.connection)
        if expected is not None and self.requested_auth_mode is not expected:
            raise AuthModeMismatch(
                str(expected), str(self.requested_auth_mode), self.connection
            )

    @property
    def config_env_name(self) -> str | None:
        """The vendor variable this receipt's ``config_dir`` is passed as.

        Mirrors :attr:`PreflightPlan.config_env_name`. It exists because the
        summary line used to hard-code "Claude config", which is wrong on every
        ``openai-sdk`` receipt: there the directory is ``CODEX_HOME``.
        """
        if self.config_dir is None:
            return None
        if self.runtime is Runtime.ANTHROPIC_SDK:
            return "CLAUDE_CONFIG_DIR"
        if self.runtime is Runtime.OPENAI_SDK:
            return "CODEX_HOME"
        return None

    @property
    def guard_note(self) -> str | None:
        """The "no spend guards configured" line, or ``None`` when there are some.

        Phrased as a fact plus where to decide, never as a recommended number:
        modelpass has no basis for one, and inventing it was the thing D4's
        2026-08-17 amendment threw out.
        """
        if self.guards_configured:
            return None
        return (
            "no spend guards configured: nothing bounds what one run on "
            f"{self.connection!r} may spend. Set warnAtTokens / stopAtTokens on the "
            "connection to add one -- see docs/guards.md for how to pick a number"
        )

    @property
    def retry_note(self) -> str | None:
        """The ``retry = "never"`` disclosure, or ``None`` on the default stance.

        Written only for the connection that opted in, for the reason the guard
        note is written only when there are no guards: a line repeated on every
        receipt is a line nobody reads, and the one connection somebody
        deliberately fenced off is the one worth saying out loud.
        """
        if self.retry != "never":
            return None
        return (
            f'connection {self.connection!r} declares retry = "never": every '
            "retryable verdict on it reads 'no'. modelpass never retries on its "
            "own in any case -- the verdict is for your loop, not ours"
        )

    @property
    def model_line(self) -> str:
        """``model X (where X came from)``, or the honest absence."""
        if not self.model:
            return "model not named; the runtime's own default applies"
        where = f" ({self.model_source})" if self.model_source else ""
        return f"model {self.model}{where}"

    @property
    def model_note(self) -> str | None:
        """The sentence for a receipt that cannot name a model, else ``None``.

        Deliberately not phrased as a problem. Letting the runtime pick is a
        perfectly ordinary way to run; what is *not* ordinary is finding out
        after a rejected run that nothing anywhere said which model was asked
        for, which is the failure this field exists to prevent.
        """
        if self.model:
            return None
        return (
            "no model is named for this run, so the runtime's own default applies "
            "and modelpass cannot say in advance what it is. Pass model= to chat() "
            "or set model on the connection to pin it"
        )

    @property
    def cache_note(self) -> str | None:
        """The prefix-caching sentence, or ``None`` when no adapter answered.

        The companion to :attr:`guard_note` and :attr:`model_note`, and it
        differs from both in one way worth stating: it has no "everything is
        fine, nothing to say" branch. Whichever way the answer falls, there is a
        sentence, because the failure this exists to surface is a *silent* one
        and a disclosure that only appears when things are already good is no
        disclosure at all (D20).
        """
        return self.cache.note if self.cache is not None else None

    def summary(self) -> str:
        """One human line. This is the receipt users read."""
        who = f" for {self.account}" if self.account else ""
        plan = f" ({self.plan_name})" if self.plan_name else ""
        source = self.credential_source or "unknown credential source"
        line = (
            f"{self.connection}: {self.runtime.value} in "
            f"{self.effective_auth_mode.value} mode using {source}{who}{plan}"
        )
        line += f" -- {self.model_line}"
        if self.scrubbed:
            line += f" -- scrubbed {', '.join(self.scrubbed)}"
        if self.passthrough:
            # Loud rather than buried: these are variables that were scrubbed by
            # default precisely because they change where a run goes.
            line += f" -- allowEnv kept {', '.join(self.passthrough)}"
        if self.config_dir:
            line += f" -- {self.config_env_name or 'config root'} {self.config_dir}"
        if not self.guards_configured:
            line += " -- no spend guards configured"
        if self.retry == "never":
            # Only for the connection that opted in, so no existing receipt's
            # text moves: no connection could say this before today.
            line += ' -- retry = "never"'
        if self.cache is not None:
            # The short clause, not the paragraph: this line is read every run,
            # and the reasoning belongs to ``cache_note``, which is read once.
            line += f" -- {self.cache.clause}"
        if not self.ok:
            line += f" -- FAILED: {self.problem or 'unknown reason'}"
        return line

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form, for embedding a receipt in an event (D11)."""
        return {
            "connection": self.connection,
            "runtime": self.runtime.value,
            "requested_auth_mode": self.requested_auth_mode.value,
            "detected_auth_mode": (
                self.detected_auth_mode.value if self.detected_auth_mode else None
            ),
            "credential_source": self.credential_source,
            "account": self.account,
            "plan_name": self.plan_name,
            "account_profile": (
                self.account_profile.to_dict() if self.account_profile is not None else None
            ),
            "identity_verified": self.identity_verified,
            "runtime_available": self.runtime_available,
            "scrubbed": list(self.scrubbed),
            "preserved": list(self.preserved),
            "passthrough": list(self.passthrough),
            "config_dir": self.config_dir,
            "directives": [d.to_dict() for d in self.directives],
            "notes": list(self.notes),
            "guards_configured": self.guards_configured,
            "retry": self.retry,
            "model": self.model,
            "model_source": self.model_source,
            "binary": self.binary,
            "cache": self.cache.to_dict() if self.cache is not None else None,
            "cache_breakpoints_requested": self.cache_breakpoints_requested,
            "cache_breakpoints_honoured": self.cache_breakpoints_honoured,
            "prompt_cache_requested": self.prompt_cache_requested,
            "prompt_cache_disposition": self.prompt_cache_disposition,
            "sampling_requested": dict(self.sampling_requested),
            "sampling_applied": dict(self.sampling_applied),
            "sampling_notes": list(self.sampling_notes),
            "ok": self.ok,
            "problem": self.problem,
            "summary": self.summary(),
        }


def api_preflight(
    connection: Connection,
    plan: PreflightPlan,
    env: Mapping[str, str] | None = None,
    *,
    runtime_available: bool = True,
    probe: ModelListProbe | None = None,
    secrets: SecretSource | None = None,
) -> Receipt:
    """The preflight for an API runtime: three cheap checks and a receipt.

    The agent runtimes' preflight asks "which auth mode will this *launch*
    actually use", because two of the three providers fail toward metered
    billing and a stray environment variable decides it silently. None of that
    question survives here: an API runtime has exactly one auth mode, and the
    credential is whichever one the adapter passes to the constructor. What
    replaces it is the set of things that *can* still go wrong before a request
    is made, in increasing order of cost:

    1. **The credential resolves to a non-empty value** (:func:`resolve_credential`).
       Free, offline, and catches the commonest failure by a distance: a
       connection naming an environment variable that is not set in this process.
    2. **The base URL parses and is safe** (:meth:`PreflightPlan.require_base_url`).
       Free, offline. Catches a hand-edited endpoint and refuses plain ``http``
       to anywhere but loopback, where the key would travel in the clear.
    3. **The endpoint answers**, optionally (:class:`ModelListProbe`). One
       token-free round trip, supplied by the caller rather than performed here,
       because this function must stay offline and pure -- the adapter owns the
       network and the cache.

    Failures of (1) and (2) come back as ``ok=False`` with a ``problem``, not as
    an exception: they are the ordinary "your setup is not finished" answers and
    :meth:`Receipt.require_ok` is where a caller turns them into a raise. An
    unresolvable credential *kind* still raises, because that is a gap in
    modelpass rather than in the user's configuration.

    Receipt fields that differ from an agent runtime, all for the same reason --
    nothing is launched and nothing is logged into:

    * ``detected_auth_mode`` is always ``api_key``. There is no precedence order
      to lose a fight with.
    * ``plan_name`` is ``None``: an API key has no subscription plan. Not
      "unknown" -- there is no such thing to know.
    * ``binary`` is ``None``: no executable is resolved.
    * ``runtime_available`` means *the vendor SDK imports*, not *a CLI is on
      PATH*.
    * ``account`` may carry a :func:`credential_fingerprint` and never the key.
    """
    notes: list[str] = []
    problem: str | None = None
    account: str | None = None

    # A connection that declares no credential is not a connection whose
    # credential is missing, and the receipt must not make them look alike
    # (ticket 1.10). Only ``openai-compatible`` can be in this state --
    # ``Connection`` refuses 'none' everywhere else -- and there the endpoint
    # very often checks nothing, so the honest report is what was configured and
    # what will be sent in its place.
    declares_no_credential = connection.credential_ref.kind is CredentialKind.NONE
    if declares_no_credential:
        notes.append(
            "this connection declares no credential (credentialRef 'none'), which "
            "only openai-compatible accepts: a local Ollama or LM Studio endpoint "
            "authenticates nobody. The client is still constructed explicitly, with "
            "a placeholder the SDK requires the endpoint is free to ignore, so "
            "nothing ambient in this process's environment is read"
        )
        try:
            plan.require_base_url(connection)
        except PreflightFailed as exc:
            problem = str(exc)
    else:
        try:
            plan.require_base_url(connection)
            value = resolve_credential(connection, env, secrets=secrets)
        except PreflightFailed as exc:
            problem = str(exc)
        else:
            account = credential_fingerprint(value)
            notes.append(
                "the credential is resolved explicitly and handed to the client "
                "constructor; the vendor SDK never reads the process environment"
            )

    if connection.base_url is not None:
        notes.append(f"endpoint {connection.base_url}")
    if probe is not None:
        notes.append(probe.note)
        if not probe.ok and problem is None:
            problem = probe.detail or "the endpoint did not answer the model-list probe"

    return Receipt.from_plan(
        plan,
        detected_auth_mode=AuthMode.API_KEY,
        credential_source=connection.credential_ref.describe(),
        account=account,
        plan_name=None,
        binary=None,
        runtime_available=runtime_available,
        notes=tuple(notes),
        ok=problem is None,
        problem=problem,
    )
