"""The connection store: ``~/.modelpass/connections.toml``.

Connections are global on purpose (D2). A user configures a subscription once and
every tool built on modelpass reuses it, rather than each tool inventing its own
credential handling.

Two properties this file must have, and the tests enforce both:

* **It is the only source of connections.** The store never reads the process
  environment looking for credentials. A missing file means zero connections,
  which is exactly what a fresh install should have.
* **It contains no secrets.** Only ``credentialRef`` pointers, validated on the
  way in and on the way out.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import tomllib
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _toml
from .connections import (
    AccountBinding,
    Connection,
    CredentialRef,
    Guards,
    QuotaAction,
    QuotaPolicy,
    VerifiedCapabilities,
)
from .errors import ConfigError, DuplicateConnection, InvalidConnection, NoSuchConnection
from .runtimes import parse_runtime
from .types import parse_auth_mode

__all__ = [
    "CONFIG_VERSION",
    "AccountStore",
    "ConnectionStore",
    "StoreCompatibility",
    "StoreSettings",
    "connection_from_dict",
    "connection_to_dict",
    "default_home",
    "unknown_connection_keys",
]

CONFIG_VERSION = 1
_FILENAME = "connections.toml"
_HOME_ENV = "MODELPASS_HOME"

#: The pre-rename home and its environment variable. Both are honoured for the
#: shim's lifetime (removed in 0.3.0) so a machine that configured connections
#: under the old name keeps working without the user being told to move files.
_LEGACY_HOME_ENV = "SUBPASS_HOME"
_HOME_DIRNAME = ".modelpass"
_LEGACY_HOME_DIRNAME = ".subpass"

#: Copy-forward and the old-env-var warning each say their piece once per
#: process. A store is constructed on nearly every call; a note repeated per
#: call would be noise, and noise is how a one-time migration gets ignored.
_said_legacy_env = False
_said_copy_forward = False


def default_home() -> Path:
    """Where connections live when the caller names no root.

    ``$MODELPASS_HOME`` wins. ``$SUBPASS_HOME`` is honoured when it does not,
    with a deprecation warning. Otherwise ``~/.modelpass``. The environment
    variable moves the *config location* only; it can never turn an ambient
    credential into a connection.
    """
    global _said_legacy_env

    env_root = os.environ.get(_HOME_ENV)
    if env_root:
        return Path(env_root)

    legacy_root = os.environ.get(_LEGACY_HOME_ENV)
    if legacy_root:
        if not _said_legacy_env:
            _said_legacy_env = True
            warnings.warn(
                f"{_LEGACY_HOME_ENV} is deprecated and will be ignored in modelpass 0.3.0. "
                f"Set {_HOME_ENV} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return Path(legacy_root)

    home = Path.home() / _HOME_DIRNAME
    _copy_forward_legacy_home(home, Path.home() / _LEGACY_HOME_DIRNAME)
    return home


def _copy_forward_legacy_home(home: Path, legacy: Path) -> bool:
    """Seed a first ``~/.modelpass`` from an existing ``~/.subpass``.

    Copy, never move: the old directory stays exactly as it was, so a tool still
    running the pre-rename build keeps reading the file it has always read. The
    two diverge from here, which is the honest trade -- a machine mid-migration
    has two builds with different ideas of the same config, and silently
    deleting one side's copy is the worse of the two failures.
    """
    global _said_copy_forward

    if home.exists() or not legacy.is_dir():
        return False
    try:
        shutil.copytree(legacy, home)
    except OSError as exc:
        # Best effort by design: a store that cannot be seeded is a store with
        # zero connections, which is a legitimate state and already handled.
        print(f"modelpass: could not copy {legacy} to {home}: {exc}", file=sys.stderr)
        return False
    if not _said_copy_forward:
        _said_copy_forward = True
        print(
            f"modelpass: copied your connections forward from {legacy} to {home}. "
            f"{legacy} was left untouched.",
            file=sys.stderr,
        )
    return True


_HEADER = f"""\
# modelpass connections
#
# Written by modelpass. Safe to edit by hand.
#
# This file contains no secrets. credentialRef says where a credential lives --
# "native-login" (the vendor runtime's own login), "env:NAME", or
# "keychain:entry" -- never the credential itself.
#
# A credential that is not referenced here is never used.
#
# Spend guards are opt-in and have no defaults: a connection with no [guards]
# table has nothing bounding what one run may spend. modelpass will not invent a
# number for you -- suggested starting points are in the docs (docs/guards.md).
#
# Every completed run appends one line to runs.jsonl beside this file: when,
# which connection, which auth mode actually paid for it, tokens, outcome. No
# prompts and no responses -- it is a billing ledger, not a transcript. Turn it
# off with [settings] runLog = false.
#
# Compatibility policy (2026-09-13). Several tools on one machine share this
# file, so they will not all be the same build of modelpass:
#
#   * The version above bumps only when an existing key changes meaning or
#     disappears. New keys and new runtime values are additive and do NOT bump
#     it.
#   * A build reads any version. A build refuses to *write* a file whose
#     version is newer than its own, and says which build is needed.
#   * A key this build does not recognise is preserved exactly as written and
#     reported -- never a failure, and never dropped when the file is rewritten.
#   * An unknown runtime or auth mode refuses that one connection, naming the
#     values this build knows, and leaves every other connection usable.

version = {CONFIG_VERSION}
"""

#: The keys each table understands, by path. A closed set is what lets an
#: unrecognised key be *carried through* rather than guessed at: this build does
#: not know what it means, which is precisely why it has no business rewriting
#: or dropping it.
_CONNECTION_KEYS = frozenset(
    {
        "runtime",
        "authMode",
        "credentialRef",
        "model",
        "description",
        "nickname",
        "configDir",
        "baseUrl",
        "verifiedIdentity",
        "verifiedCapabilities",
        "experimental",
        "allowEnv",
        "groups",
        "guards",
        "enabled",
        "retry",
        "timeoutSeconds",
    }
)
_IDENTITY_KEYS = frozenset({"email", "organizationId"})
#: The keys inside ``verifiedCapabilities`` (ticket 1.10). Note that the
#: *values* are capability names this build may not know, which is why the
#: record keeps them as strings and ``refine()`` ignores the ones it has never
#: heard of -- an unknown cell name is a newer build's evidence, not a fault.
_VERIFIED_CAPABILITY_KEYS = frozenset({"supported", "unsupported", "checkedAt", "models"})
_GUARD_KEYS = frozenset({"warnAtTokens", "stopAtTokens", "onQuotaExhausted"})
_QUOTA_KEYS = frozenset({"action", "failover"})


@dataclass(frozen=True, slots=True)
class StoreSettings:
    """Store-wide settings -- the ``[settings]`` table, one per modelpass home.

    Deliberately a *store* concern rather than a per-connection one: the run log
    answers "what has this machine spent, and on whose credential", which is a
    question about the home directory, not about any one connection. Splitting it
    per connection would let a user switch off logging for exactly the connection
    they later needed to account for.
    """

    #: Whether completed runs are appended to ``runs.jsonl`` (config key
    #: ``runLog``). On by default: the log is the durable answer to "prove it was
    #: the subscription", and a proof that only exists when you remembered to
    #: turn it on is not much of a proof.
    run_log: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StoreSettings:
        if not isinstance(data, Mapping):
            raise ConfigError("'settings' must be a table")
        unknown = set(data) - {"runLog"}
        if unknown:
            raise ConfigError(f"unknown setting(s): {', '.join(sorted(unknown))}")
        value = data.get("runLog", True)
        if not isinstance(value, bool):
            raise ConfigError(f"settings.runLog must be true or false, got {value!r}")
        return cls(run_log=value)

    def to_dict(self) -> dict[str, Any]:
        """Only non-default values, so the file stays as quiet as the defaults."""
        return {} if self.run_log else {"runLog": False}


@dataclass(frozen=True)
class StoreCompatibility:
    """What this build could not fully account for in the file on disk.

    A report rather than an exception, for the same reason
    :class:`~modelpass.bridge.ValidationReport` is one: the caller is usually
    looking at a whole file and wants every surprise at once. Empty on a file
    this build wrote itself, which is the common case.
    """

    #: The ``version`` the file declares. Defaults to this build's own when the
    #: file says nothing, which is how a pre-version file is read.
    file_version: int = CONFIG_VERSION
    #: The version this build understands.
    build_version: int = CONFIG_VERSION
    #: Connection name -> the dotted paths of keys this build does not know and
    #: will carry through untouched.
    carried_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: Connection name -> why this build cannot use it. Refused connections are
    #: left out of :meth:`ConnectionStore.load` and still written back verbatim.
    refused: Mapping[str, str] = field(default_factory=dict)
    #: Why this build could not read the file at all, if it could not. The file
    #: then reads as zero connections -- a legitimate state (R10) -- and writes
    #: refuse rather than rewriting a shape nobody parsed.
    unreadable: str | None = None

    @property
    def writable(self) -> bool:
        """Whether this build may rewrite the file at all."""
        return self.unreadable is None and self.file_version <= self.build_version

    def notes_for(self, name: str) -> tuple[str, ...]:
        """Everything worth saying about one connection."""
        carried = self.carried_keys.get(name)
        if not carried:
            return ()
        return (
            f"key(s) this build of modelpass does not understand are preserved "
            f"unchanged: {', '.join(carried)}",
        )

    def notes(self) -> tuple[str, ...]:
        """Everything worth saying about the file, in reading order."""
        out: list[str] = []
        if self.unreadable is not None:
            return (
                "the connection file could not be read, so modelpass is reading it "
                f"as no connections configured and will not rewrite it: {self.unreadable}",
            )
        if not self.writable:
            out.append(
                f"this file declares config version {self.file_version}; this build of "
                f"modelpass understands version {self.build_version}, so it can read the "
                "file but will not rewrite it"
            )
        for name in sorted(self.carried_keys):
            for note in self.notes_for(name):
                out.append(f"connection {name!r}: {note}")
        for name in sorted(self.refused):
            out.append(f"connection {name!r} is unusable here: {self.refused[name]}")
        return tuple(out)


def unknown_connection_keys(data: Mapping[str, Any]) -> tuple[str, ...]:
    """The dotted paths in a connection table this build does not recognise.

    Sorted, so a report reads the same way twice.
    """
    found: list[str] = []
    _collect_unknown(data, _CONNECTION_KEYS, "", found)
    identity = data.get("verifiedIdentity")
    if isinstance(identity, Mapping):
        _collect_unknown(identity, _IDENTITY_KEYS, "verifiedIdentity.", found)
    verified = data.get("verifiedCapabilities")
    if isinstance(verified, Mapping):
        _collect_unknown(
            verified, _VERIFIED_CAPABILITY_KEYS, "verifiedCapabilities.", found
        )
    guards = data.get("guards")
    if isinstance(guards, Mapping):
        _collect_unknown(guards, _GUARD_KEYS, "guards.", found)
        quota = guards.get("onQuotaExhausted")
        if isinstance(quota, Mapping):
            _collect_unknown(quota, _QUOTA_KEYS, "guards.onQuotaExhausted.", found)
    return tuple(sorted(found))


def _collect_unknown(
    table: Mapping[str, Any], known: frozenset[str], prefix: str, found: list[str]
) -> None:
    found.extend(f"{prefix}{key}" for key in table if key not in known)


def _carry_unknown_keys(raw: Mapping[str, Any], emitted: dict[str, Any]) -> dict[str, Any]:
    """Merge the keys this build did not read back into what it is about to write.

    The ``[settings]`` carry-through below argues the general case; this is it,
    applied to a connection. An older build rewriting a file a newer one wrote
    must not quietly delete the half it could not parse.
    """
    out = dict(emitted)
    _carry_level(raw, out, _CONNECTION_KEYS)
    identity = raw.get("verifiedIdentity")
    if isinstance(identity, Mapping):
        identity_out = dict(out.get("verifiedIdentity") or {})
        _carry_level(identity, identity_out, _IDENTITY_KEYS)
        if identity_out:
            out["verifiedIdentity"] = identity_out
    verified = raw.get("verifiedCapabilities")
    if isinstance(verified, Mapping):
        verified_out = dict(out.get("verifiedCapabilities") or {})
        _carry_level(verified, verified_out, _VERIFIED_CAPABILITY_KEYS)
        if verified_out:
            out["verifiedCapabilities"] = verified_out
    guards = raw.get("guards")
    if isinstance(guards, Mapping):
        guards_out = dict(out.get("guards") or {})
        _carry_level(guards, guards_out, _GUARD_KEYS)
        quota = guards.get("onQuotaExhausted")
        if isinstance(quota, Mapping):
            quota_out = dict(guards_out.get("onQuotaExhausted") or {})
            _carry_level(quota, quota_out, _QUOTA_KEYS)
            if quota_out:
                guards_out["onQuotaExhausted"] = quota_out
        if guards_out:
            out["guards"] = guards_out
    return out


def _carry_level(
    raw: Mapping[str, Any], out: dict[str, Any], known: frozenset[str]
) -> None:
    for key, value in raw.items():
        if key not in known and key not in out:
            out[key] = value


def connection_to_dict(connection: Connection) -> dict[str, Any]:
    """Serialize a connection to the on-disk shape (camelCase keys)."""
    guards = connection.guards
    data: dict[str, Any] = {
        "runtime": connection.runtime.value,
        "authMode": connection.auth_mode.value,
        "credentialRef": connection.credential_ref.to_str(),
    }
    if connection.model is not None:
        data["model"] = connection.model
    if connection.description is not None:
        data["description"] = connection.description
    if connection.nickname is not None:
        data["nickname"] = connection.nickname
    if connection.config_dir is not None:
        data["configDir"] = connection.config_dir
    if connection.base_url is not None:
        data["baseUrl"] = connection.base_url
    if connection.account_binding is not None:
        binding: dict[str, str] = {}
        if connection.account_binding.email:
            binding["email"] = connection.account_binding.email
        if connection.account_binding.organization_id:
            binding["organizationId"] = connection.account_binding.organization_id
        data["verifiedIdentity"] = binding
    # Written only when a drive actually happened, for the reason the guards
    # table is written only when a threshold was set: an empty record in the
    # file reads as "verified, and it does nothing", which is the opposite of
    # "nobody has run modelpass verify against this endpoint yet".
    verified = connection.verified_capabilities
    if verified:
        verified_table: dict[str, Any] = {"supported": list(verified.supported)}
        if verified.unsupported:
            verified_table["unsupported"] = list(verified.unsupported)
        if verified.checked_at:
            verified_table["checkedAt"] = verified.checked_at
        if verified.models:
            verified_table["models"] = list(verified.models)
        data["verifiedCapabilities"] = verified_table
    if connection.experimental:
        data["experimental"] = True
    if connection.allow_env:
        data["allowEnv"] = list(connection.allow_env)
    # Written only when the connection actually names a group. An ungrouped
    # connection is in `default` and the file says nothing, for the reason
    # `enabled = true` is not written: `groups = ["default"]` on every line is a
    # default spelled out, and it would also make the one connection somebody
    # deliberately put in `default` alongside another group indistinguishable
    # from the ones that just never moved.
    if connection.groups:
        data["groups"] = list(connection.groups)
    # Written only when it is false, for the same reason the guards table is
    # written only when it is configured: `enabled = true` on every connection
    # is noise that makes the one line that matters harder to spot.
    if not connection.enabled:
        data["enabled"] = False
    # Both written only when they say something (ticket 1.12), for the reason
    # `enabled = true` is not written: a default spelled out on every connection
    # is noise that hides the one line somebody actually chose.
    if connection.retry != "default":
        data["retry"] = connection.retry
    if connection.timeout_seconds is not None:
        data["timeoutSeconds"] = connection.timeout_seconds

    # Guards are written only when the user configured something. There are no
    # default thresholds (D4 amendment, 2026-08-17), so a connection with no
    # [guards] table genuinely has no spend guards -- and the file saying nothing
    # is more honest than the file saying "warnAtTokens = 0", which reads like a
    # deliberate disabling of a guard that never existed.
    quota = guards.on_quota_exhausted
    quota_table: dict[str, Any] = {"action": quota.action.value}
    if quota.failover:
        quota_table["failover"] = quota.failover
    guard_table: dict[str, Any] = {}
    if guards.warn_at_tokens is not None:
        guard_table["warnAtTokens"] = guards.warn_at_tokens
    if guards.stop_at_tokens is not None:
        guard_table["stopAtTokens"] = guards.stop_at_tokens
    if guards.configured or quota.action is not QuotaAction.STOP:
        guard_table["onQuotaExhausted"] = quota_table
        data["guards"] = guard_table
    return data


def connection_from_dict(name: str, data: Mapping[str, Any]) -> Connection:
    """Read a connection from the on-disk shape."""
    if not isinstance(data, Mapping):
        raise InvalidConnection(f"connection {name!r} must be a table")
    # Unknown keys are not read and not refused: see the compatibility policy in
    # the file header. They are reported by :func:`unknown_connection_keys` and
    # written back untouched by :func:`_carry_unknown_keys`.
    for required in ("runtime", "authMode"):
        if required not in data:
            raise InvalidConnection(f"connection {name!r} is missing {required!r}")

    return Connection(
        name=name,
        runtime=data["runtime"],
        auth_mode=data["authMode"],
        credential_ref=CredentialRef.parse(data.get("credentialRef", "native-login")),
        guards=_guards_from_dict(name, data.get("guards", {})),
        model=data.get("model"),
        description=data.get("description"),
        nickname=data.get("nickname"),
        config_dir=data.get("configDir"),
        base_url=data.get("baseUrl"),
        account_binding=_account_binding_from_dict(
            name, data.get("verifiedIdentity")
        ),
        verified_capabilities=_verified_capabilities_from_dict(
            name, data.get("verifiedCapabilities")
        ),
        experimental=bool(data.get("experimental", False)),
        allow_env=_allow_env_from_dict(name, data.get("allowEnv", ())),
        groups=_groups_from_dict(name, data.get("groups", ())),
        enabled=_enabled_from_dict(name, data.get("enabled", True)),
        retry=_retry_from_dict(name, data.get("retry", "default")),
        timeout_seconds=data.get("timeoutSeconds"),
    )


def _retry_from_dict(name: str, value: Any) -> str:
    """Read the ``retry`` key, refusing anything but the two policies.

    A misspelling is refused rather than defaulted, and this is the one place
    where that matters most: ``retry = "nver"`` silently meaning "default" would
    be a connection the user believes is protected and is not.
    """
    if not isinstance(value, str):
        raise InvalidConnection(
            f"connection {name!r}: retry must be a string, got "
            f"{type(value).__name__}"
        )
    return value


def _unknown_value_reason(data: Mapping[str, Any]) -> str | None:
    """Why this build cannot use this connection at all, or ``None``.

    Deliberately narrow: an unknown *value* for ``runtime`` or ``authMode``
    means a newer build wrote a connection this one has no engine for, so the
    honest answer is to set that connection aside with the known set named and
    leave the rest of the file working. Every other invalid connection -- a
    guard above its stop, ``enabled = "no"`` -- is a mistake in a file a human
    edited, and quietly dropping the connection would hide it.
    """
    for key, parse in (("runtime", parse_runtime), ("authMode", parse_auth_mode)):
        value = data.get(key)
        if value is None:
            continue
        try:
            parse(value)
        except ValueError as exc:
            return str(exc)
    return None


def _verified_capabilities_from_dict(name: str, value: Any) -> VerifiedCapabilities:
    """Read the ``verifiedCapabilities`` table, or an empty record (ticket 1.10).

    Absent is the normal state and reads as "nobody has driven this endpoint",
    which is different from "driven and it can do nothing" -- the second is what
    an empty ``supported`` list beside a populated ``unsupported`` one says, and
    both round-trip.
    """
    if value is None:
        return VerifiedCapabilities()
    if not isinstance(value, Mapping):
        raise InvalidConnection(
            f"connection {name!r}: verifiedCapabilities must be a table"
        )
    try:
        return VerifiedCapabilities(
            supported=tuple(value.get("supported", ()) or ()),
            unsupported=tuple(value.get("unsupported", ()) or ()),
            checked_at=str(value.get("checkedAt", "") or ""),
            models=tuple(value.get("models", ()) or ()),
        )
    except TypeError as exc:
        raise InvalidConnection(
            f"connection {name!r}: verifiedCapabilities holds a value of the wrong "
            f"shape ({exc})"
        ) from exc


def _account_binding_from_dict(name: str, value: Any) -> AccountBinding | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise InvalidConnection(
            f"connection {name!r}: verifiedIdentity must be a table"
        )
    return AccountBinding(
        email=value.get("email"), organization_id=value.get("organizationId")
    )


def _enabled_from_dict(name: str, value: Any) -> bool:
    """Read ``enabled``, refusing anything that is not a boolean.

    Not coerced with ``bool()``: ``enabled = "false"`` is a string, which is
    truthy, and silently enabling a connection its owner meant to switch off is
    the exact failure this field exists to prevent.
    """
    if not isinstance(value, bool):
        raise InvalidConnection(
            f"connection {name!r}: enabled must be true or false, got {value!r}"
        )
    return value


def _allow_env_from_dict(name: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise InvalidConnection(f"connection {name!r}: allowEnv must be an array of names")
    return tuple(str(item) for item in value)


def _groups_from_dict(name: str, value: Any) -> tuple[str, ...]:
    """Read the ``groups`` key. An array of names, and never a bare string.

    ``groups = "fast"`` is refused rather than read as a one-element list. TOML
    would accept the string happily and Python would iterate it into five
    single-character groups, which is the kind of quiet nonsense that shows up
    later as a group nobody can find.
    """
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise InvalidConnection(
            f"connection {name!r}: groups must be an array of group names, for "
            'example groups = ["fast", "cheap"]'
        )
    for item in value:
        if not isinstance(item, str):
            raise InvalidConnection(
                f"connection {name!r}: group {item!r} must be a string"
            )
    # The name rule and the duplicate rule live on Connection, which is the one
    # place they can also catch a connection built in code.
    return tuple(value)


def _guards_from_dict(name: str, data: Mapping[str, Any]) -> Guards:
    if not isinstance(data, Mapping):
        raise InvalidConnection(f"connection {name!r}: guards must be a table")
    quota = data.get("onQuotaExhausted", {})
    if not isinstance(quota, Mapping):
        raise InvalidConnection(f"connection {name!r}: onQuotaExhausted must be a table")
    policy = QuotaPolicy(
        action=quota.get("action", "stop"),
        failover=quota.get("failover"),
    )
    defaults = Guards()
    # for_config, not the plain constructor: a file that names a warn above its
    # stop is a typo worth failing on, where the same pair passed for one call is
    # clamped (first-consumer feedback, 2026-08-17).
    return Guards.for_config(
        warn_at_tokens=data.get("warnAtTokens", defaults.warn_at_tokens),
        stop_at_tokens=data.get("stopAtTokens", defaults.stop_at_tokens),
        on_quota_exhausted=policy,
        where=f"connection {name!r}",
    )


class ConnectionStore:
    """Reads and writes the connection file.

    ``root`` defaults to :func:`default_home` -- ``$MODELPASS_HOME`` if set,
    else the deprecated ``$SUBPASS_HOME``, else ``~/.modelpass`` (seeded once
    from ``~/.subpass`` if that is what this machine has).
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        if root is None:
            root = default_home()
        self.root = Path(root)

    @property
    def path(self) -> Path:
        return self.root / _FILENAME

    def exists(self) -> bool:
        return self.path.is_file()

    # --- reading ---------------------------------------------------------------

    def _read(self) -> dict[str, Any]:
        """The raw file. Missing *or unreadable* file means an empty document.

        **Reading never fails** (R10). Two checks that used to raise here now
        answer through :meth:`unreadable_reason` instead:

        * The version. The check is "newer than this build", and it gates
          *writes* only (:meth:`save`). A build that refused to read a file a
          newer build wrote could not run ``modelpass list`` or ``modelpass
          check`` either, which is the moment a user most needs their tools able
          to say something.
        * Whether the file parses and opens at all. A corrupt file, or one whose
          permissions this process cannot get past, reads as **zero
          connections** -- the same legitimate state a fresh install is in --
          and :meth:`compatibility` carries the reason so every surface can say
          out loud that it is not seeing the file. This is what the secrets file
          has always done; the connection file used to raise, which made
          ``modelpass list`` unable to speak on exactly the machine where
          somebody needed it to (2026-09-13, ticket 1.4).

        Writes are the opposite and deliberately so: :meth:`save` refuses,
        because rewriting a file this build could not read would destroy
        whatever is in it.
        """
        raw, reason = self._read_checked()
        return {} if reason is not None else raw

    def _read_checked(self) -> tuple[dict[str, Any], str | None]:
        """The raw file and, if this build could not read it, why."""
        if not self.exists():
            return {}, None
        try:
            with self.path.open("rb") as handle:
                raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            return {}, f"{self.path} is not valid TOML: {exc}"
        except OSError as exc:
            return {}, f"cannot read {self.path}: {exc}"
        version = raw.get("version", CONFIG_VERSION)
        if isinstance(version, bool) or not isinstance(version, int):
            return (
                {},
                f"{self.path}: config version must be a whole number, got {version!r}",
            )
        if not isinstance(raw.get("connections", {}), Mapping):
            return {}, f"{self.path}: 'connections' must be a table"
        return raw, None

    def unreadable_reason(self) -> str | None:
        """Why this build could not read the file, or ``None`` (R10).

        ``None`` for a file that is absent -- "nothing configured" is not a
        failure -- and for one whose declared version is newer than this build's,
        which reads fine and merely cannot be rewritten.
        """
        return self._read_checked()[1]

    def _file_version(self, raw: Mapping[str, Any]) -> int:
        version = raw.get("version", CONFIG_VERSION)
        # A version that is not a whole number is a malformed file, not a newer
        # one: nothing can be concluded from it, including that it is safe to
        # rewrite.
        if isinstance(version, bool) or not isinstance(version, int):
            raise ConfigError(
                f"{self.path}: config version must be a whole number, got {version!r}"
            )
        return version

    def compatibility(self) -> StoreCompatibility:
        """What this build could not fully account for in the file on disk."""
        raw, unreadable = self._read_checked()
        if unreadable is not None:
            return StoreCompatibility(unreadable=unreadable)
        connections = raw.get("connections", {})
        if not isinstance(connections, Mapping):  # pragma: no cover - _read_checked
            return StoreCompatibility()
        carried: dict[str, tuple[str, ...]] = {}
        refused: dict[str, str] = {}
        for name, data in connections.items():
            if not isinstance(data, Mapping):
                continue
            unknown = unknown_connection_keys(data)
            if unknown:
                carried[name] = unknown
            reason = _unknown_value_reason(data)
            if reason is not None:
                refused[name] = reason
        return StoreCompatibility(
            file_version=self._file_version(raw),
            build_version=CONFIG_VERSION,
            carried_keys=carried,
            refused=refused,
        )

    def settings(self) -> StoreSettings:
        """Store-wide settings, defaulted when the file says nothing."""
        raw = self._read()
        if "settings" not in raw:
            return StoreSettings()
        try:
            return StoreSettings.from_dict(raw["settings"])
        except ConfigError as exc:
            raise ConfigError(f"{self.path}: {exc}") from exc

    def save_settings(self, settings: StoreSettings) -> None:
        """Write store-wide settings, leaving the connections alone."""
        self.save(self.load().values(), settings=settings)

    def load(self) -> dict[str, Connection]:
        """Every *usable* connection, keyed by name. Missing file means none.

        A connection naming a runtime or auth mode this build does not know is
        left out rather than making the whole file unreadable; ask
        :meth:`compatibility` for the names and the reasons. It is still written
        back untouched by :meth:`save`.
        """
        raw = self._read()
        connections = raw.get("connections", {})
        if not isinstance(connections, Mapping):  # pragma: no cover - _read_checked
            return {}
        return {
            name: connection_from_dict(name, data)
            for name, data in connections.items()
            if not (isinstance(data, Mapping) and _unknown_value_reason(data))
        }

    def list(self) -> tuple[Connection, ...]:
        """Connections in name order."""
        return tuple(sorted(self.load().values(), key=lambda c: c.name))

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.load()))

    def has(self, name: str) -> bool:
        return name in self.load()

    def get(self, name: str) -> Connection:
        """One connection by name, or :class:`NoSuchConnection`."""
        connections = self.load()
        try:
            return connections[name]
        except KeyError:
            pass
        # "There is no such connection" and "this build cannot use that one" are
        # opposite answers, and offering the first for the second sends a user
        # looking for a typo in a name that is right there in the file.
        reason = self.compatibility().refused.get(name)
        if reason is not None:
            raise InvalidConnection(f"connection {name!r} is unusable here: {reason}")
        raise NoSuchConnection(name, tuple(sorted(connections))) from None

    # --- writing ---------------------------------------------------------------

    def save(
        self,
        connections: Iterable[Connection],
        *,
        settings: StoreSettings | None = None,
        drop: Iterable[str] = (),
    ) -> None:
        """Replace the whole file with these connections, atomically.

        ``drop`` names connections to forget even though this build could not
        parse them -- the one way a refused connection leaves the file, used by
        :meth:`remove`. Without it, "delete the connection my build does not
        understand" would have no answer but hand-editing.

        ``settings`` defaults to whatever the file already says, so adding a
        connection never silently reverts a store-wide setting somebody set.

        Refuses outright if the file on disk declares a config version newer
        than this build understands: writing it would mean rewriting a shape
        this build cannot read. Keys and whole connections this build could not
        parse are carried through verbatim.
        """
        raw, unreadable = self._read_checked()
        if unreadable is not None:
            # The read path treats this file as zero connections; the write path
            # must not. Emitting this build's shape over bytes nobody parsed is
            # how a user loses whatever was in there.
            raise ConfigError(unreadable)
        self._require_writable(raw)
        raw_connections = raw.get("connections", {})
        if not isinstance(raw_connections, Mapping):
            raw_connections = {}
        table = {
            connection.name: connection_to_dict(connection)
            for connection in sorted(connections, key=lambda c: c.name)
        }
        for name, emitted in table.items():
            existing = raw_connections.get(name)
            if isinstance(existing, Mapping):
                table[name] = _carry_unknown_keys(existing, emitted)
        # A connection this build refused is still the user's connection. It is
        # written back exactly as found, in name order with the rest.
        dropped = set(drop)
        for name, existing in raw_connections.items():
            if name in table or name in dropped or not isinstance(existing, Mapping):
                continue
            if _unknown_value_reason(existing):
                table[name] = dict(existing)
        table = {name: table[name] for name in sorted(table)}
        document: dict[str, Any] = {}
        settings_table = self._settings_table_to_write(settings)
        if settings_table:
            document["settings"] = settings_table
        if table:
            document["connections"] = table
        body = _toml.dumps(document) if document else ""
        text = _HEADER + ("\n" + body if body else "")
        self._write(text)

    def _require_writable(self, raw: Mapping[str, Any]) -> None:
        """Refuse to rewrite a file written by a newer build of modelpass.

        Read, yes -- everything parseable stays readable. But rewriting means
        emitting this build's shape over a shape it does not know, which is how
        a user loses the half of their config that their newer tool wrote.
        """
        version = self._file_version(raw)
        if version > CONFIG_VERSION:
            raise ConfigError(
                f"{self.path} declares config version {version}; this build of modelpass "
                f"writes version {CONFIG_VERSION}. It can still read the file, but a "
                "newer build of modelpass is needed to change it."
            )

    def _settings_table_to_write(self, settings: StoreSettings | None) -> dict[str, Any]:
        """The ``[settings]`` table to emit, preserving one this build cannot read.

        A settings table that fails validation -- a typo, or a key written by a
        newer modelpass -- must not make the file unwritable. Refusing to add or
        *delete* a connection until an unrelated optional setting is hand-fixed
        would strand a user inside their own config.

        So an unparseable table is carried through **verbatim** rather than
        being dropped or reinterpreted: this build does not understand it, which
        is precisely the reason it has no business rewriting it.
        """
        if settings is not None:
            return settings.to_dict()
        try:
            return self.settings().to_dict()
        except ConfigError:
            pass
        try:
            raw = self._read().get("settings")
        except ConfigError:
            return {}
        return dict(raw) if isinstance(raw, Mapping) else {}

    def add(self, connection: Connection, *, overwrite: bool = False) -> None:
        """Add or replace one connection."""
        connections = self.load()
        if connection.name in connections and not overwrite:
            raise DuplicateConnection(connection.name)
        connections[connection.name] = connection
        self.save(connections.values())

    def remove(self, name: str) -> None:
        connections = self.load()
        if name in connections:
            del connections[name]
            self.save(connections.values())
            return
        if name in self.compatibility().refused:
            self.save(connections.values(), drop=(name,))
            return
        raise NoSuchConnection(name, tuple(sorted(connections)))

    def _write(self, text: str) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            # Best effort on POSIX; a no-op on Windows, where the user profile
            # directory already carries the ACL.
            try:
                self.root.chmod(0o700)
            except (OSError, NotImplementedError):
                pass
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=self.root,
                prefix=".connections-",
                suffix=".tmp",
                delete=False,
            )
            try:
                with handle:
                    handle.write(text)
                os.replace(handle.name, self.path)
            except BaseException:
                Path(handle.name).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise ConfigError(f"cannot write {self.path}: {exc}") from exc


# Account-oriented public name. The on-disk filename and ``[connections]`` table
# remain unchanged for backward compatibility; records now represent named
# account profiles and ConnectionStore remains a supported alias.
AccountStore = ConnectionStore
