"""The connection model -- the central object of the library (D2).

A connection is the informed-consent step. A fresh install has no AI
connectivity: an ``ANTHROPIC_API_KEY`` sitting in the environment does nothing at
all until the user creates a connection that explicitly references it. Everything
downstream (which runtime, which billing mode, which credential, which guards)
is read off this object and nothing else.

``credentialRef`` is never a secret. It says *where* the credential lives --
"native login", an environment variable name, a keychain entry -- and modelpass
refuses to store anything that looks like the credential itself.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from .capabilities import VerifiedCapabilities, runtime_auth_modes
from .errors import (
    CredentialRefIsSecret,
    InvalidConnection,
    InvalidGuards,
    RuntimeGated,
)
from .runtimes import API_RUNTIMES, EXPERIMENTAL_RUNTIMES, VENDOR_OF, Runtime, parse_runtime
from .types import AuthMode, parse_auth_mode

__all__ = [
    "ALLOWED_ENV_PASSTHROUGH",
    "BASE_URL_REQUIRED",
    "DEFAULT_GROUP",
    "LOCAL_HOSTS",
    "Account",
    "AccountBinding",
    "Connection",
    "CredentialKind",
    "CredentialRef",
    "Group",
    "GroupSelection",
    "Guards",
    "QuotaAction",
    "QuotaPolicy",
    "VerifiedCapabilities",
    "group_names",
    "groups_of",
    "parse_base_url",
]

#: Non-credential environment variables a connection may explicitly let through
#: the scrub, per runtime (D2 extension, 2026-08-17).
#:
#: These are variables that are *not* credentials but still change where a run
#: goes or how it is billed, so they are scrubbed by default and honoured only
#: when a connection names them in ``allowEnv``. The list is a closed set rather
#: than a free-form passthrough on purpose: a general "keep these variables"
#: escape hatch would be a hole straight through D2, since the first thing
#: someone would put in it is an API key.
ALLOWED_ENV_PASSTHROUGH: dict[Runtime, frozenset[str]] = {
    Runtime.ANTHROPIC_SDK: frozenset({"ANTHROPIC_BASE_URL"}),
    Runtime.OPENAI_SDK: frozenset(),
    Runtime.GOOGLE_CLI: frozenset(),
    Runtime.GOOGLE_SDK: frozenset(),
    # Nothing on the API runtimes, and the emptiness is the design rather than a
    # gap to be filled later: allowEnv exists so a *child process* may inherit a
    # non-credential variable that would otherwise be scrubbed. An API runtime
    # launches no child and its client is constructed with explicit arguments, so
    # a variable let through here would reach nothing. Where a base URL is the
    # thing being expressed, ``baseUrl`` on the connection is the written-down
    # place for it -- which is the same argument allowEnv made, one step further.
    Runtime.ANTHROPIC_API: frozenset(),
    Runtime.OPENAI_API: frozenset(),
    Runtime.GOOGLE_API: frozenset(),
    Runtime.OPENAI_COMPATIBLE: frozenset(),
}

#: Runtimes that cannot be used without a ``baseUrl``: there is no default
#: endpoint to fall back to, because the runtime names an API *shape* and not a
#: vendor.
BASE_URL_REQUIRED: frozenset[Runtime] = frozenset({Runtime.OPENAI_COMPATIBLE})

#: Hosts where plain ``http`` is accepted. Everything else must be ``https``.
#:
#: The exception is narrow on purpose. A local model server -- Ollama, LM
#: Studio, a LiteLLM proxy on the same machine -- genuinely has no certificate
#: and refusing it would make the ``openai-compatible`` runtime useless for its
#: commonest case. Traffic that never leaves the loopback interface cannot be
#: read off the wire by somebody else, which is the risk TLS is answering. A
#: private-network address (``192.168.x.x``, ``10.x.x.x``) is deliberately *not*
#: on this list: that traffic crosses a wire, the wire has other machines on it,
#: and "it's just my LAN" is exactly the reasoning that puts a key on it in
#: plaintext.
LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

#: The group a connection belongs to when it declares none (ticket 1.15).
#:
#: It always exists and it is never written to the file, because it is not a
#: thing that is configured -- it is the *name for the ungrouped ones*. A
#: connection with ``groups = ["fast"]`` is in ``fast`` and is **not** in
#: ``default``; putting it in both is spelled by saying both, which is the only
#: reading under which ``default`` carries any information at all. A group that
#: silently contained every connection would be a filter that filters nothing
#: and a route that routes anywhere.
DEFAULT_GROUP = "default"

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")

#: What ``retry`` may say (ticket 1.12). Two values and no more: this is a
#: stance, not a policy engine, and "how many times and how far apart" is the
#: consumer's question because the consumer is the one paying for the answer.
_RETRY_POLICIES: frozenset[str] = frozenset({"default", "never"})
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOCATOR_RE = re.compile(r"^[A-Za-z0-9_./@:-]{1,64}$")
_SECRET_HINTS = ("sk-", "sk_", "sess-", "ghp_", "-ant-", "bearer ")


@dataclass(frozen=True, slots=True)
class AccountBinding:
    """Non-secret vendor identity pinned to a named Subpass account."""

    email: str | None = None
    organization_id: str | None = None

    def __post_init__(self) -> None:
        email = self.email.strip().casefold() if isinstance(self.email, str) else None
        organization_id = (
            self.organization_id.strip()
            if isinstance(self.organization_id, str)
            else None
        )
        if not email and not organization_id:
            raise InvalidConnection(
                "verifiedIdentity needs an email or organizationId"
            )
        object.__setattr__(self, "email", email or None)
        object.__setattr__(self, "organization_id", organization_id or None)

    def to_dict(self) -> dict[str, str | None]:
        return {"email": self.email, "organization_id": self.organization_id}


class CredentialKind(StrEnum):
    """Where a credential lives. Never what it is."""

    NATIVE_LOGIN = "native-login"
    ENV = "env"
    KEYCHAIN = "keychain"
    #: No credential at all (ticket 1.10, 2026-09-13). Accepted on
    #: ``openai-compatible`` and **nowhere else**, because that runtime is the
    #: only one where "the endpoint authenticates nobody" is a real and common
    #: configuration: a local Ollama or LM Studio box accepts any bearer token,
    #: or none. It is a *stated* absence rather than an empty ``env:`` pointing
    #: at an unset variable, which is the shape people reach for otherwise and
    #: which the preflight is right to refuse -- "I did not configure a key" and
    #: "my key is missing" are opposite findings and must not look the same.
    #: The adapter still constructs its client with an explicit placeholder,
    #: because the SDK requires the argument; nothing ambient is ever read.
    NONE = "none"
    #: An entry in the modelpass secrets file (R14, 2026-09-13). The locator is
    #: the entry *name*, which is still a pointer -- the value lives in
    #: ``secrets.toml``, a separate, non-shareable file, and never here.
    SECRET = "secret"


@dataclass(frozen=True, slots=True)
class CredentialRef:
    """A pointer to a credential, in the form ``kind`` (+ ``locator``).

    Wire form is a single string: ``native-login``, ``env:ANTHROPIC_API_KEY``,
    ``secret:vendor-key``, ``keychain:modelpass/anthropic``.
    """

    kind: CredentialKind
    locator: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CredentialKind(self.kind))
        if self.kind is CredentialKind.NATIVE_LOGIN:
            if self.locator:
                raise InvalidConnection(
                    "credentialRef 'native-login' takes no locator; the vendor runtime "
                    "owns its own login store"
                )
            return
        if self.kind is CredentialKind.NONE:
            if self.locator:
                raise InvalidConnection(
                    "credentialRef 'none' takes no locator; it says there is no "
                    "credential, which is not a place a credential lives"
                )
            return
        if not self.locator:
            raise InvalidConnection(f"credentialRef {self.kind.value!r} requires a locator")
        _reject_secret(self.locator)
        if self.kind is CredentialKind.ENV and not _ENV_NAME_RE.match(self.locator):
            raise InvalidConnection(
                f"credentialRef env locator {self.locator!r} is not a valid variable name"
            )
        # A secret entry is addressed the same way a connection is, and on
        # purpose: the commonest entry name *is* a connection name, the rename
        # rule compares the two directly, and one character set means a name
        # that works in one file works in the other.
        if self.kind is CredentialKind.SECRET and not _NAME_RE.match(self.locator):
            raise InvalidConnection(
                f"credentialRef secret entry {self.locator!r} must be 1-64 characters "
                "of letters, digits, dot, dash or underscore -- the same rule "
                "connection names follow"
            )

    @classmethod
    def native_login(cls) -> CredentialRef:
        return cls(CredentialKind.NATIVE_LOGIN)

    @classmethod
    def none(cls) -> CredentialRef:
        """No credential -- ``openai-compatible`` only. See :attr:`CredentialKind.NONE`."""
        return cls(CredentialKind.NONE)

    @classmethod
    def parse(cls, raw: str | CredentialRef) -> CredentialRef:
        if isinstance(raw, CredentialRef):
            return raw
        if not isinstance(raw, str) or not raw.strip():
            raise InvalidConnection("credentialRef must be a non-empty string")
        text = raw.strip()
        kind_text, _, locator = text.partition(":")
        try:
            kind = CredentialKind(kind_text)
        except ValueError:
            valid = ", ".join(k.value for k in CredentialKind)
            raise InvalidConnection(
                f"unknown credentialRef kind {kind_text!r} (expected one of: {valid})"
            ) from None
        return cls(kind=kind, locator=locator or None)

    def to_str(self) -> str:
        return self.kind.value if self.locator is None else f"{self.kind.value}:{self.locator}"

    def describe(self) -> str:
        """A phrase for the preflight receipt."""
        if self.kind is CredentialKind.NATIVE_LOGIN:
            return "the runtime's own login store"
        if self.kind is CredentialKind.NONE:
            return "no credential (the endpoint is declared unauthenticated)"
        if self.kind is CredentialKind.ENV:
            return f"environment variable {self.locator}"
        if self.kind is CredentialKind.SECRET:
            return f"the modelpass secrets file (entry {self.locator})"
        return f"keychain entry {self.locator}"

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11). A pointer, never a secret -- that is enforced
        at construction, so serializing one cannot leak a credential."""
        return {
            "kind": self.kind.value,
            "locator": self.locator,
            "ref": self.to_str(),
            "describe": self.describe(),
        }


def _reject_secret(locator: str) -> None:
    """Refuse anything that looks like a credential rather than a pointer."""
    lowered = locator.lower()
    if any(hint in lowered for hint in _SECRET_HINTS):
        raise CredentialRefIsSecret(
            "credentialRef looks like a secret. It must name where the credential "
            "lives (env:NAME, keychain:entry), never the credential itself"
        )
    if not _LOCATOR_RE.match(locator):
        raise CredentialRefIsSecret(
            "credentialRef locator has an unexpected shape (whitespace, punctuation "
            "or length); it must be a short pointer, never a credential"
        )


def parse_base_url(value: object, *, where: str = "baseUrl") -> str:
    """Validate and normalize a connection's ``baseUrl``. Pure; makes no request.

    Four checks, and each one is a failure somebody has actually shipped:

    * **It parses, and it names a host.** ``api.example.com`` with no scheme is
      the commonest typo and every SDK treats it differently -- some prepend
      ``https://``, some treat it as a relative path, one raises. Refusing it
      here means the receipt says what is wrong instead of the vendor saying
      something else is.
    * **``https``, or ``http`` to an explicit loopback host** (:data:`LOCAL_HOSTS`).
      A key sent over plain ``http`` across a network is a key somebody else on
      that network has.
    * **No credentials in the URL.** ``https://user:key@host`` puts a secret in
      a file the store's whole contract says holds none -- the same rule
      :func:`_reject_secret` applies to ``credentialRef``, applied to the other
      field that can carry one.
    * **No query string and no fragment.** A base URL has no use for either, and
      the one thing people put in a query string is ``?api_key=...``. Refusing it
      keeps the "this file contains no secrets" promise checkable rather than
      hopeful.

    Normalization is whitespace-stripping and nothing else. A trailing slash, a
    ``/v1`` suffix, an explicit port: all preserved exactly as written, because
    which of those a given endpoint wants is the endpoint's business and a
    library that quietly rewrites a URL is a library whose user cannot tell what
    was actually requested.
    """
    if not isinstance(value, str) or not value.strip():
        raise InvalidConnection(f"{where} must be a non-empty URL")
    text = value.strip()
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise InvalidConnection(f"{where} is not a valid URL: {exc}") from exc

    if parsed.username or parsed.password:
        raise CredentialRefIsSecret(
            f"{where} carries credentials in the URL. The connection file holds "
            "pointers, never secrets -- put the key behind credentialRef and give "
            "baseUrl the bare endpoint"
        )
    if parsed.query or parsed.fragment:
        raise InvalidConnection(
            f"{where} must be a bare endpoint with no query string or fragment; a "
            "query string is where a key ends up by accident"
        )
    if parsed.scheme not in {"http", "https"}:
        scheme = parsed.scheme or "none"
        raise InvalidConnection(
            f"{where} must be an http or https URL (scheme was {scheme!r}); "
            f"{text!r} is not one -- a bare host name is the usual cause"
        )
    host = parsed.hostname
    if not host:
        raise InvalidConnection(f"{where} names no host: {text!r}")
    if parsed.scheme == "http" and host.lower() not in LOCAL_HOSTS:
        allowed = ", ".join(sorted(LOCAL_HOSTS))
        raise InvalidConnection(
            f"{where} uses plain http to {host!r}. http is accepted only for a "
            f"local endpoint ({allowed}); anything else must be https, because the "
            "credential travels with every request"
        )
    return text


class _Unset:
    """Sentinel for "this threshold was not mentioned".

    Needed because ``None`` is a *meaningful* value for a threshold -- it means
    "no guard" -- so it cannot double as "leave whatever was there".
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET = _Unset()


class QuotaAction(StrEnum):
    """What to do when the subscription allowance runs out."""

    STOP = "stop"
    FAILOVER = "failover"


@dataclass(frozen=True, slots=True)
class QuotaPolicy:
    """Quota-exhaustion behavior. Stopping cleanly is the only default (D4)."""

    action: QuotaAction = QuotaAction.STOP
    failover: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", QuotaAction(self.action))
        if self.action is QuotaAction.FAILOVER and not self.failover:
            raise InvalidConnection(
                "onQuotaExhausted.action = 'failover' requires a 'failover' connection "
                "name; failover is never implicit"
            )
        if self.action is QuotaAction.STOP and self.failover:
            raise InvalidConnection(
                "onQuotaExhausted.failover is set but action is 'stop'; set "
                "action = 'failover' to opt in"
            )
        # A group is refused here, by name, and it is the one place in the
        # library where a group reference is *not* accepted where a connection
        # name is (ticket 1.15). Failover is the only path in modelpass that can
        # move a run onto metered billing, and under D4(d) the consent for that
        # is this field naming the target. A group names a *set whose membership
        # changes afterwards*: somebody adds a metered connection to "cheap"
        # next month and a consent given to one connection silently starts
        # paying for another. That is exactly the surprise D4 exists to prevent,
        # so the refusal is at construction -- where the file is written and the
        # user is looking -- rather than at the moment an allowance runs out.
        if self.failover and self.failover.startswith("group:"):
            raise InvalidConnection(
                f"onQuotaExhausted.failover is {self.failover!r}, and a failover "
                "target must be one named connection rather than a group. "
                "Failover is the only path onto metered billing and naming the "
                "target is the consent; a group's membership can change after "
                "that consent was given. Name the connection you mean"
            )

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11)."""
        return {"action": self.action.value, "failover": self.failover}


@dataclass(frozen=True, slots=True)
class Guards:
    """Per-connection guard configuration (D4, offered layer).

    **There are no default thresholds** (D4 amendment, 2026-08-17). An earlier
    draft shipped 200k warn / 1M stop per run; both numbers were invented here
    rather than derived from anything, and a made-up number presented as a
    product default is the same class of dishonesty this library exists to
    prevent -- it reads as advice the project cannot actually give. So guards
    exist when, and only when, the user configures them.

    The consequence is handled rather than shrugged at: their absence is made
    *visible* instead of defaulted. :meth:`configured` drives a preflight-receipt
    line ("no spend guards configured") and a prominent note from
    ``modelpass connect``, and recommended starting values live in the docs, where
    they can be argued with, rather than in code, where they cannot.

    Thresholds are tokens, per run. **Dollar thresholds stay out of the v1
    schema** (decided 2026-08-17, not deferred): only one of the three providers
    reports a dollar figure at all, that figure lives in ``vendor_event`` under
    D7's tokens-never-dollars rule, and a dollar guard that silently worked on
    one runtime would be worse than none. Vendor-side hard caps are the honest
    answer for metered connections, and the docs say so.

    **A warn above a stop is clamped, not refused** (first-consumer feedback,
    2026-08-17). Constructing ``Guards`` used to raise when
    ``warn_at_tokens > stop_at_tokens``, which made the obvious per-call gesture
    -- lower the stop for this one call -- fail unless the caller also
    remembered to lower a warn it had not mentioned. A warn beyond a stop is not
    a contradiction, it is unreachable: the run stops before it can fire. So it
    is clamped down to the stop, which is the only reading with any behavior in
    it. Where the same pair *is* worth failing over is a config file, and
    :meth:`for_config` -- which is what the store and the CLI use -- still
    refuses it loudly, because there it is a typo somebody will otherwise trust.
    """

    warn_at_tokens: int | None = None
    stop_at_tokens: int | None = None
    on_quota_exhausted: QuotaPolicy = field(default_factory=QuotaPolicy)

    def __post_init__(self) -> None:
        for name in ("warn_at_tokens", "stop_at_tokens"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidGuards(f"{name} must be an integer number of tokens")
            if value < 0:
                raise InvalidGuards(
                    f"{name} is a number of tokens, so it cannot be negative; "
                    "pass 0 to run with no guard at all"
                )
            # 0 means "disabled" in config; normalize to None internally.
            if value == 0:
                object.__setattr__(self, name, None)
        if (
            self.warn_at_tokens is not None
            and self.stop_at_tokens is not None
            and self.warn_at_tokens > self.stop_at_tokens
        ):
            object.__setattr__(self, "warn_at_tokens", self.stop_at_tokens)

    @classmethod
    def for_config(
        cls,
        *,
        warn_at_tokens: int | None = None,
        stop_at_tokens: int | None = None,
        on_quota_exhausted: QuotaPolicy | None = None,
        where: str = "guards",
    ) -> Guards:
        """Guards read from, or about to be written to, a config file.

        The strict constructor. A warn above a stop is refused here rather than
        clamped, and refused as an :class:`InvalidConnection`, because a written
        file that says two contradictory things is a typo the user should be
        told about before it is saved and trusted -- not something to silently
        reinterpret behind their back.
        """
        if (
            isinstance(warn_at_tokens, int)
            and not isinstance(warn_at_tokens, bool)
            and isinstance(stop_at_tokens, int)
            and not isinstance(stop_at_tokens, bool)
            and warn_at_tokens > 0
            and stop_at_tokens > 0
            and warn_at_tokens > stop_at_tokens
        ):
            raise InvalidConnection(
                f"{where}: warnAtTokens must not exceed stopAtTokens "
                f"({warn_at_tokens} > {stop_at_tokens}); a warning that can never "
                "fire is not a warning"
            )
        try:
            return cls(
                warn_at_tokens=warn_at_tokens,
                stop_at_tokens=stop_at_tokens,
                on_quota_exhausted=on_quota_exhausted or QuotaPolicy(),
            )
        except InvalidGuards as exc:
            # In a file, a bad guard value *is* a bad connection definition.
            raise InvalidConnection(f"{where}: {exc}") from exc

    @classmethod
    def for_call(
        cls,
        base: Guards | None = None,
        *,
        warn_at_tokens: int | _Unset | None = _UNSET,
        stop_at_tokens: int | _Unset | None = _UNSET,
    ) -> Guards:
        """Guards for one call, derived from whatever the connection already has.

        The lenient constructor, and the one behind ``chat(stop_at_tokens=N)``.
        A threshold not mentioned keeps the connection's value; a threshold given
        as ``0`` or ``None`` switches that guard off for the call; a warn left
        above the new stop is clamped rather than refused. The quota policy is
        carried across untouched, so lowering a ceiling for one call never
        quietly disarms a configured failover.
        """
        base = base or cls()
        return cls(
            warn_at_tokens=(
                base.warn_at_tokens if isinstance(warn_at_tokens, _Unset) else warn_at_tokens
            ),
            stop_at_tokens=(
                base.stop_at_tokens if isinstance(stop_at_tokens, _Unset) else stop_at_tokens
            ),
            on_quota_exhausted=base.on_quota_exhausted,
        )

    @property
    def configured(self) -> bool:
        """Whether any spend threshold is actually in force for this run.

        The quota policy is deliberately not counted: stopping cleanly when the
        allowance runs out is what happens anyway, so a connection with only a
        quota policy still has nothing bounding its spend.
        """
        return self.warn_at_tokens is not None or self.stop_at_tokens is not None

    @classmethod
    def disabled(cls) -> Guards:
        return cls(warn_at_tokens=None, stop_at_tokens=None)

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11)."""
        return {
            "warn_at_tokens": self.warn_at_tokens,
            "stop_at_tokens": self.stop_at_tokens,
            "on_quota_exhausted": self.on_quota_exhausted.to_dict(),
            "configured": self.configured,
        }


@dataclass(frozen=True, slots=True)
class Connection:
    """A vendor account profile and its explicitly configured runtime route."""

    name: str
    runtime: Runtime
    auth_mode: AuthMode
    credential_ref: CredentialRef = field(default_factory=CredentialRef.native_login)
    guards: Guards = field(default_factory=Guards)
    model: str | None = None
    description: str | None = None
    experimental: bool = False
    #: Non-credential environment variables this connection explicitly permits
    #: through the scrub (config key ``allowEnv``). Restricted to
    #: :data:`ALLOWED_ENV_PASSTHROUGH` for the runtime -- see its docstring for
    #: why this is a closed set and not a general escape hatch.
    allow_env: tuple[str, ...] = ()
    #: Whether this connection may be used for a run (config key ``enabled``,
    #: default ``true``; added 2026-08-17).
    #:
    #: A disabled connection stays fully configured and fully *inspectable* --
    #: it is listed, and its preflight still runs -- but :meth:`Bridge.chat`
    #: refuses it. The point is to have a way of taking a connection out of
    #: service that is not "delete it and retype it later": deleting is how a
    #: careful user loses the guard thresholds they worked out, and a commented-
    #: out block in the config file is a connection modelpass can no longer say
    #: anything about. Disabling keeps the connection visible and keeps the
    #: refusal legible, which is the same reasoning D4 applied to absent guards.
    enabled: bool = True
    #: Runtime-owned configuration root (config key ``configDir``). Claude Code
    #: calls this ``CLAUDE_CONFIG_DIR``; Codex calls it ``CODEX_HOME``. Keeping
    #: the path on the account -- rather than inheriting one ambient process
    #: value -- lets named subscription accounts remain isolated.
    #:
    #: Kept after all older fields to preserve their positional constructor
    #: shape. New code should still use keyword arguments for connections.
    config_dir: str | None = None
    #: Human-facing account label (config key ``nickname``), for example
    #: ``"Anthropic Home"``. ``name`` remains the stable machine-facing ID used
    #: by callers and logs; nicknames may contain spaces and may be changed
    #: without breaking client configuration. Last for positional compatibility.
    nickname: str | None = None
    #: Vendor-reported identity captured during deliberate verification. Every
    #: later preflight compares the current login to this binding and fails
    #: before a run if the account has drifted.
    account_binding: AccountBinding | None = None
    #: The API endpoint this connection talks to (config key ``baseUrl``), for
    #: the API runtimes only (2026-09-13).
    #:
    #: **Required on ``openai-compatible``**, which names an API shape rather
    #: than a vendor: without an endpoint there is nothing to talk to, and
    #: defaulting it to OpenAI's would turn "compatible" into a silent alias for
    #: the real thing. **Optional on the other three**, where the vendor's own
    #: default applies and this is the lever for a proxy, a gateway or a regional
    #: endpoint. **Refused on the agent runtimes**, where the equivalent is
    #: ``ANTHROPIC_BASE_URL`` through ``allowEnv``: those launch a child process
    #: that reads its endpoint from its own configuration, and a field modelpass
    #: stored but never passed would be a setting that looks obeyed and is not.
    #:
    #: Validated by :func:`parse_base_url` at construction -- which is also the
    #: check the API preflight re-runs, so a connection that was edited by hand
    #: into an unusable state is caught before a run rather than by one. Last in
    #: the field order, to preserve every older field's positional shape.
    base_url: str | None = None
    #: What ``modelpass verify`` observed this *endpoint* doing (config key
    #: ``verifiedCapabilities``, ticket 1.10, 2026-09-13).
    #:
    #: Empty on every runtime whose capability row is a fact about a vendor,
    #: which is all of them but one. On ``openai-compatible`` the row can only
    #: describe an API *shape*, so the cells a caller can act on are filled from
    #: a drive against the configured endpoint and kept here --
    #: :meth:`~modelpass.Bridge.registry_for` folds them into a registry copy
    #: through :meth:`~modelpass.capabilities.CapabilityRegistry.refine`, per
    #: connection, for the life of that call. An additive key under the store's
    #: 0.1 compatibility policy: an older build preserves it verbatim and reports
    #: it rather than dropping the evidence. Last in the field order, for the same
    #: positional reason every field above it gives.
    verified_capabilities: VerifiedCapabilities = field(
        default_factory=VerifiedCapabilities
    )
    #: This connection's retry policy (config key ``retry``, ticket 1.12,
    #: 2026-09-13). ``"default"`` or ``"never"``.
    #:
    #: **It does not make modelpass retry anything -- modelpass never does.** It
    #: forces every :class:`~modelpass.types.Retryable` verdict produced on this
    #: connection to :attr:`~modelpass.types.Retryable.NO`, so a consumer's own
    #: loop stops here without that consumer having to learn which of its
    #: connections must never be repeated. A RAG evaluation harness asked for
    #: exactly this
    #: ("timeout 30s, 2 retries, and *no* retries on **this** connection") and
    #: the part it could not express anywhere was the per-connection stance.
    #:
    #: The receipt reports it, so the refusal is visible before a run rather
    #: than inferred from a verdict afterwards. An additive key under the store's
    #: 0.1 compatibility policy. Last in the field order, like every field above.
    retry: str = "default"
    #: Default wall-clock bound for calls on this connection, in seconds (config
    #: key ``timeoutSeconds``, ticket 1.12, 2026-09-13).
    #:
    #: Used as ``Timeout(total=...)`` when a call passes no ``timeout=`` of its
    #: own; a call that passes one replaces this entirely rather than being
    #: clamped by it, because a caller who named a bound has said what they want
    #: and a silent tightening is the kind of surprise that turns a working
    #: batch job into a flaky one. ``None`` means unbounded, which is what every
    #: connection written before this key existed means.
    timeout_seconds: float | None = None
    #: The groups this connection belongs to (config key ``groups``, ticket
    #: 1.15). Empty means :data:`DEFAULT_GROUP`.
    #:
    #: A group is a **selection unit**, not a label: ``bridge.select("cheap")``
    #: resolves to a member, and ``connection="group:cheap"`` on any entry point
    #: routes a run through one. That is why membership is declared here, on the
    #: connection, rather than in a ``[groups]`` table listing members: a group
    #: with a definition of its own would be a second place a connection can be
    #: taken out of service, and there is already exactly one
    #: (:attr:`enabled`). Groups exist because connections claim them and for no
    #: other reason, so deleting the last member deletes the group and nothing
    #: is left behind pointing at nothing.
    #:
    #: **Which member a group resolves to is name order among the enabled
    #: ones**, and the arbitrariness of that is why
    #: :meth:`~modelpass.bridge.Bridge.select` hands back the runners-up and the
    #: reason each was passed over rather than just the winner. A caller who
    #: wants a particular member names that member; a *preference* order within
    #: a group is a real want and it is on the roadmap next to throttling, where
    #: it can be designed with the thing it will be used for.
    #:
    #: An additive key under the store's 0.1 compatibility policy: an older
    #: build carries it through verbatim and reports it, so a grouped file stays
    #: usable by a build that has never heard of groups -- it just sees every
    #: connection by name, which is what it saw before.
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise InvalidConnection(
                f"connection name {self.name!r} must be 1-64 characters of letters, "
                "digits, dot, dash or underscore"
            )
        object.__setattr__(self, "runtime", parse_runtime(self.runtime))
        object.__setattr__(self, "auth_mode", parse_auth_mode(self.auth_mode))
        object.__setattr__(self, "credential_ref", CredentialRef.parse(self.credential_ref))

        if self.config_dir is not None:
            if not isinstance(self.config_dir, str) or not self.config_dir.strip():
                raise InvalidConnection("configDir must be a non-empty path")
            if self.runtime not in {Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK}:
                raise InvalidConnection(
                    "configDir is only supported by the Anthropic and OpenAI runtimes"
                )
            expanded = Path(self.config_dir).expanduser()
            if not expanded.is_absolute():
                raise InvalidConnection(
                    "configDir must be an absolute path (or start with '~'); a relative "
                    "credential directory could select a different account by working directory"
                )
            object.__setattr__(self, "config_dir", str(expanded))

        if self.nickname is not None:
            if not isinstance(self.nickname, str) or not self.nickname.strip():
                raise InvalidConnection("nickname must be a non-empty display name")
            object.__setattr__(self, "nickname", self.nickname.strip())

        if self.account_binding is not None:
            if not isinstance(self.account_binding, AccountBinding):
                raise InvalidConnection("verifiedIdentity must be an AccountBinding")
            if self.auth_mode is not AuthMode.SUBSCRIPTION:
                raise InvalidConnection(
                    "verifiedIdentity is only valid for subscription accounts"
                )

        if self.base_url is not None:
            if self.runtime not in API_RUNTIMES:
                raise InvalidConnection(
                    f"baseUrl is only supported by the API runtimes "
                    f"({', '.join(sorted(r.value for r in API_RUNTIMES))}); runtime "
                    f"{self.runtime.value!r} launches its own process and reads its "
                    "endpoint from its own configuration"
                )
            object.__setattr__(
                self,
                "base_url",
                parse_base_url(self.base_url, where=f"connection {self.name!r}: baseUrl"),
            )
        elif self.runtime in BASE_URL_REQUIRED:
            raise InvalidConnection(
                f"connection {self.name!r}: runtime {self.runtime.value!r} requires "
                "baseUrl -- it names an API shape rather than a vendor, so there is "
                "no default endpoint to fall back to"
            )

        if self.retry not in _RETRY_POLICIES:
            valid = ", ".join(sorted(_RETRY_POLICIES))
            raise InvalidConnection(
                f"connection {self.name!r}: retry must be one of {valid}, got "
                f"{self.retry!r}. It does not make modelpass retry -- modelpass "
                'never does -- it forces this connection\'s retryable verdicts to '
                '"no"'
            )

        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds, (int, float)
            ):
                raise InvalidConnection(
                    f"connection {self.name!r}: timeoutSeconds must be a number of "
                    f"seconds, got {type(self.timeout_seconds).__name__}"
                )
            if self.timeout_seconds <= 0:
                raise InvalidConnection(
                    f"connection {self.name!r}: timeoutSeconds must be greater than "
                    f"zero, got {self.timeout_seconds!r}"
                )
            object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

        allowed = runtime_auth_modes(self.runtime)
        if self.auth_mode not in allowed:
            allowed_text = ", ".join(sorted(m.value for m in allowed)) or "none"
            raise InvalidConnection(
                f"runtime {self.runtime.value!r} cannot run in auth mode "
                f"{self.auth_mode.value!r} (supported: {allowed_text})"
            )

        if (
            self.auth_mode is AuthMode.API_KEY
            and self.credential_ref.kind is CredentialKind.NATIVE_LOGIN
        ):
            raise InvalidConnection(
                "an api_key connection must point at a credential (env: or keychain:); "
                "'native-login' is the subscription path"
            )

        # 'none' is the one runtime-scoped credential kind, and the scope is the
        # whole of why it is safe. On openai-compatible the endpoint is a box the
        # user chose and very often authenticates nobody, so declaring that is
        # more honest than pointing at an environment variable nobody will set.
        # On any other runtime the same declaration would be a connection that
        # cannot pay for itself -- and worse, one whose vendor SDK would then
        # look for a key of its own and find whatever is ambient, which is D2's
        # exact failure. So it is refused there, by name, at construction.
        if (
            self.credential_ref.kind is CredentialKind.NONE
            and self.runtime is not Runtime.OPENAI_COMPATIBLE
        ):
            raise InvalidConnection(
                f"credentialRef 'none' is only supported by runtime "
                f"{Runtime.OPENAI_COMPATIBLE.value!r}, where an endpoint that checks "
                f"no key is an ordinary configuration; runtime {self.runtime.value!r} "
                "talks to a metered vendor that requires one -- name it with "
                "'env:NAME' or 'secret:<entry>'"
            )

        if not isinstance(self.verified_capabilities, VerifiedCapabilities):
            raise InvalidConnection(
                "verifiedCapabilities must be a VerifiedCapabilities record"
            )

        # A stored key can only be handed to something this process constructs.
        # The agent runtimes launch a child, and the only way a child gets a
        # credential is an environment variable -- which is the path `env:NAME`
        # already is. Accepting `secret:` there would write a connection whose
        # child launches with no key at all and quietly falls back to the
        # runtime's own login, billing an account nobody named. That is the
        # exact failure D2 exists to prevent, so it is refused at construction
        # rather than discovered from a bill.
        if (
            self.credential_ref.kind is CredentialKind.SECRET
            and self.runtime not in API_RUNTIMES
        ):
            raise InvalidConnection(
                f"credentialRef 'secret:' is only supported by the API runtimes "
                f"({', '.join(sorted(r.value for r in API_RUNTIMES))}); runtime "
                f"{self.runtime.value!r} launches a child process, which can only be "
                "handed a credential through the environment -- use 'env:NAME' there"
            )

        if not isinstance(self.enabled, bool):
            raise InvalidConnection(
                f"connection {self.name!r}: enabled must be true or false"
            )

        object.__setattr__(self, "groups", tuple(self.groups or ()))
        seen: set[str] = set()
        for group in self.groups:
            if not isinstance(group, str) or not _NAME_RE.match(group):
                raise InvalidConnection(
                    f"connection {self.name!r}: group {group!r} must be 1-64 "
                    "characters of letters, digits, dot, dash or underscore -- the "
                    "same rule connection names follow"
                )
            if group in seen:
                # Refused rather than de-duplicated: a list that names one group
                # twice is a typo in a file a human wrote, and the second entry
                # is very often the one that was meant to be a different group.
                raise InvalidConnection(
                    f"connection {self.name!r}: group {group!r} is listed twice"
                )
            seen.add(group)

        object.__setattr__(self, "allow_env", tuple(self.allow_env or ()))
        permitted = ALLOWED_ENV_PASSTHROUGH.get(self.runtime, frozenset())
        for name in self.allow_env:
            if not isinstance(name, str) or not _ENV_NAME_RE.match(name):
                raise InvalidConnection(
                    f"allowEnv entry {name!r} is not a valid environment variable name"
                )
            if name not in permitted:
                allowed = ", ".join(sorted(permitted)) or "nothing"
                raise InvalidConnection(
                    f"allowEnv may not carry {name!r} for runtime "
                    f"{self.runtime.value!r} (permitted: {allowed}). allowEnv exists "
                    "for non-credential variables that still change where a run goes; "
                    "a credential belongs in credentialRef, where it is checked"
                )

        if self.runtime in EXPERIMENTAL_RUNTIMES and not self.experimental:
            raise RuntimeGated(
                f"runtime {self.runtime.value!r} is gated: set experimental = true on the "
                "connection to opt in. Google stays gated until the Antigravity terms "
                "have been read from the primary source (D5)"
            )

    @property
    def is_subscription(self) -> bool:
        return self.auth_mode is AuthMode.SUBSCRIPTION

    @property
    def vendor(self) -> str:
        """The account vendor, independent of the adapter implementation name.

        Reads :data:`~modelpass.runtimes.VENDOR_OF`, which is the single source
        (2026-09-13). This used to special-case the two v1 runtimes and derive
        everything else from the runtime string, which was fine until a runtime
        arrived whose name does not contain its vendor: ``openai-compatible``
        read as ``"openai"``, so ``Bridge.find(vendor="openai")`` would have
        returned an Ollama box on localhost beside real OpenAI accounts. Deriving
        a vendor from a name is the bug; looking it up is the fix.
        """
        return VENDOR_OF[self.runtime]

    @property
    def display_name(self) -> str:
        """The label people see; the stable ID remains :attr:`name`."""
        return self.nickname or self.name

    @property
    def group_names(self) -> tuple[str, ...]:
        """The groups this connection is actually in, with the default filled in.

        :attr:`groups` is what the file says; this is what it *means*. The two
        differ for exactly one connection shape -- the ungrouped one -- and
        keeping them separate is what lets the file stay quiet about a default
        while every reader still gets a straight answer to "which groups is this
        in".
        """
        return self.groups or (DEFAULT_GROUP,)

    def in_group(self, group: str) -> bool:
        """Whether this connection is a member of ``group``."""
        return group in self.group_names

    def with_groups(self, groups: Sequence[str]) -> Connection:
        """A copy in different groups. Persisted by writing it back."""
        return replace(self, groups=tuple(groups))

    def with_guards(self, guards: Guards) -> Connection:
        """A copy with different guards -- for per-run overrides that never persist."""
        return replace(self, guards=guards)

    def with_enabled(self, enabled: bool) -> Connection:
        """A copy taken in or out of service. Persisted by writing it back."""
        return replace(self, enabled=bool(enabled))

    def summary(self) -> str:
        state = "" if self.enabled else " [disabled]"
        identity = f"{self.nickname} ({self.name})" if self.nickname else self.name
        # The default group is left out on purpose: printing "in default" on
        # every line of a store nobody has grouped is noise that hides the one
        # connection somebody did put somewhere.
        groups = f" in {', '.join(self.groups)}" if self.groups else ""
        return (
            f"{identity}: {self.runtime.value} "
            f"in {self.auth_mode.value} mode "
            f"via {self.credential_ref.describe()}{groups}{state}"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11).

        Field names are the Python ones, not the file's camelCase: this is the
        in-memory object serialized for a log or a wire, and
        :func:`modelpass.store.connection_to_dict` remains the on-disk shape. A
        connection holds no secret by construction, so this is safe to print.
        """
        return {
            "name": self.name,
            "runtime": self.runtime.value,
            "auth_mode": self.auth_mode.value,
            "credential_ref": self.credential_ref.to_dict(),
            "guards": self.guards.to_dict(),
            "model": self.model,
            "description": self.description,
            "nickname": self.nickname,
            "display_name": self.display_name,
            "vendor": self.vendor,
            "config_dir": self.config_dir,
            "base_url": self.base_url,
            "verified_identity": (
                self.account_binding.to_dict() if self.account_binding else None
            ),
            "experimental": self.experimental,
            "allow_env": list(self.allow_env),
            "groups": list(self.group_names),
            "declared_groups": list(self.groups),
            "verified_capabilities": self.verified_capabilities.to_dict(),
            "summary": self.summary(),
        }


# Public account-oriented name. Kept as an alias rather than a wrapper so every
# existing adapter and client that accepts ``Connection`` accepts ``Account``
# without translation or a second source of truth.
Account = Connection


@dataclass(frozen=True, slots=True)
class Group:
    """A group as it currently stands: who is in it, and who of those can run.

    Derived, never stored. There is no ``[groups]`` table and no group object on
    disk -- a group is the set of connections that claim it, computed on demand,
    which is why a group cannot drift out of step with its membership and why
    removing the last member removes the group.
    """

    name: str
    #: Every member, in name order, enabled or not.
    members: tuple[str, ...]
    #: The members a run could actually go to, in name order.
    enabled: tuple[str, ...]

    @property
    def is_default(self) -> bool:
        return self.name == DEFAULT_GROUP

    @property
    def available(self) -> bool:
        """Whether :func:`select` would find anything here."""
        return bool(self.enabled)

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11)."""
        return {
            "name": self.name,
            "members": list(self.members),
            "enabled": list(self.enabled),
            "is_default": self.is_default,
            "available": self.available,
        }


@dataclass(frozen=True, slots=True)
class GroupSelection:
    """Which member of a group a run is about to go to, and who was passed over.

    The runners-up are the point. Resolution is name order among the enabled
    members, which is a real rule but not one the user chose, so handing back
    only the winner would present an arbitrary pick as though it were a
    decision. A caller -- or a receipt, or the bench -- can say "``cheap``
    resolved to ``local-ollama``; ``a-gateway`` was skipped because it is
    disabled", which is a sentence somebody can act on.
    """

    group: str
    chosen: Connection
    #: ``(connection name, why it was not chosen)``, in name order.
    passed_over: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11)."""
        return {
            "group": self.group,
            "chosen": self.chosen.name,
            "passed_over": [
                {"connection": name, "reason": reason}
                for name, reason in self.passed_over
            ],
        }


def group_names(connections: Iterable[Connection]) -> tuple[str, ...]:
    """Every group these connections are in, in name order.

    ``default`` appears when, and only when, some connection declares no groups
    -- it is the name for the ungrouped ones, so a store where everything has
    been filed somewhere genuinely has no ``default`` group and saying otherwise
    would be inventing a member.
    """
    seen: set[str] = set()
    for connection in connections:
        seen.update(connection.group_names)
    return tuple(sorted(seen))


def groups_of(connections: Iterable[Connection]) -> tuple[Group, ...]:
    """Every group, with its membership, in name order."""
    members: dict[str, list[str]] = {}
    enabled: dict[str, list[str]] = {}
    for connection in sorted(connections, key=lambda c: c.name):
        for group in connection.group_names:
            members.setdefault(group, []).append(connection.name)
            if connection.enabled:
                enabled.setdefault(group, []).append(connection.name)
    return tuple(
        Group(
            name=name,
            members=tuple(members[name]),
            enabled=tuple(enabled.get(name, ())),
        )
        for name in sorted(members)
    )
