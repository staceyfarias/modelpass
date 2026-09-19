"""Connection management from Python, with the receipts the command line shows.

``modelpass connect`` is a consent flow: it builds a connection, runs the
preflight, shows the receipt, and writes only once somebody has looked. That is
the right shape, and until now it existed **only** as a command line. A desktop
app whose Settings page lets a user paste a key had two options, both bad: shell
out to a CLI that refuses to read a key from anywhere but a pipe, or write the
files itself and reinvent the validation, the ordering and the delete rules
(R10, and a desktop agent app's validation report, which calls this "the single
biggest unanswered question for this consumer").

This module is that flow as a library. The verbs are the CLI's verbs, split at
the one seam that matters:

* :meth:`ConnectionManager.plan_connection` builds the :class:`Connection` and
  returns the **pre-write receipt**. It writes nothing -- not the connection,
  and not the pasted key, which is held in memory as a
  :class:`~modelpass.secrets.PendingSecret` so the preflight describes the shape
  that is actually about to be saved.
* :meth:`ConnectionManager.add_connection` takes that plan and writes it, secret
  first and connection second, for the reason ``connect`` writes them in that
  order: the two files are not written atomically together, and a crash between
  them should leave an inert unreferenced secret rather than a connection
  pointing at nothing.

Why a separate object rather than five more methods on :class:`Bridge`
(2026-09-13): ``Bridge`` is the surface for *running* -- chat, sessions,
preflight, validation -- and is already two thousand lines of it. Connection
lifecycle is a different concern with its own vocabulary and its own result
types, and grouping it under ``bridge.manage`` keeps "what can I change" one
attribute away from "what can I run" without interleaving the two in one
namespace. The manager holds no state of its own; it is a facade over the
bridge's store and secrets, so a bridge pointed at a test home manages that
home's files.

Every result is a plain frozen dataclass with a ``to_dict()`` that is safe to
print, log or return as JSON. No secret **value** appears in any of them, in
their ``repr``, or in anything they carry -- the redaction tests in
``tests/test_secrets.py`` walk this module's results the same way they walk the
receipt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .connections import (
    DEFAULT_GROUP,
    AccountBinding,
    Connection,
    CredentialKind,
    CredentialRef,
    Guards,
)
from .errors import ConfigError, DuplicateConnection, PreflightFailed, SubpassError
from .preflight import Receipt, api_preflight, plan_launch
from .runtimes import API_RUNTIMES, Runtime
from .secrets import PendingSecret, SecretPermissions, SecretSource
from .types import AuthMode

if TYPE_CHECKING:  # pragma: no cover - typing only, and it would be a cycle
    from .bridge import Bridge

__all__ = [
    "AddResult",
    "ConnectionManager",
    "ConnectionPlan",
    "Credential",
    "EnabledResult",
    "GroupsResult",
    "RemoveResult",
    "RenameResult",
    "receipt_for",
]


def receipt_for(
    bridge: Bridge, connection: Connection, *, secrets: SecretSource | None = None
) -> Receipt:
    """The receipt for a connection, on either family of runtime.

    On an agent runtime this is :meth:`~modelpass.Bridge.preflight`, which needs
    the adapter because half the answer is the vendor's. On an API runtime the
    whole preflight is offline -- the credential resolves, the base URL parses --
    so this takes that path **whether or not an adapter is installed**, reporting
    ``runtime_available`` from whether one could be loaded.

    That it does so even when an adapter exists is the decision (2026-09-13,
    ticket 1.6). Setup is the one caller that must check a key it has not written
    down yet, and ``secrets`` is how it says so; going through the adapter would
    resolve the reference from the file on disk instead and fail a connection
    that is about to be perfectly good. Nothing is lost by staying offline here:
    an API adapter's preflight *is* :func:`api_preflight`, plus an opt-in network
    probe and a note about the model, neither of which setup asks for.

    ``secrets`` names where a ``secret:`` reference is resolved from, so a key
    that has been pasted but not yet written can be checked in memory.
    """
    if connection.runtime in API_RUNTIMES:
        try:
            bridge.adapter_for(connection.runtime)
        except SubpassError:
            available = False
        else:
            available = True
        source = secrets if secrets is not None else bridge.secrets
        return api_preflight(
            connection,
            plan_launch(connection, bridge.env),
            bridge.env,
            runtime_available=available,
            secrets=source,
        )
    return bridge.preflight(connection)


def binding_from_receipt(receipt: Receipt) -> AccountBinding | None:
    """The identity a subscription receipt pinned, or ``None`` if it named none."""
    profile = receipt.account_profile
    if profile is None or not (profile.email or profile.organization_id):
        return None
    return AccountBinding(email=profile.email, organization_id=profile.organization_id)


@dataclass(frozen=True, slots=True)
class Credential:
    """How a connection is paid for, as a caller states it before anything exists.

    Four shapes, matching the four the store understands, each a constructor
    so that the auth mode is derived rather than asked for -- an ``api_key``
    connection pointing at a native login, or a subscription one pointing at an
    environment variable, are both mistakes a caller should not be able to make
    by passing two arguments that disagree.

    :meth:`secret` carries the pasted value in memory for the length of one
    plan. It is excluded from ``repr`` and has no ``to_dict``; the only thing
    that ever leaves is the entry name.
    """

    kind: CredentialKind
    #: ``None`` on a secret means "the connection's own name", which is filled in
    #: by :meth:`ConnectionManager.plan_connection` once the name is known.
    locator: str | None = None
    value: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.locator is not None:
            # Validate the pointer now rather than at write time: an env name
            # that is really a pasted key is refused here by the same check the
            # store uses.
            CredentialRef(self.kind, self.locator)

    @classmethod
    def native_login(cls) -> Credential:
        """The runtime's own login: the subscription path, no credential stored."""
        return cls(CredentialKind.NATIVE_LOGIN)

    @classmethod
    def none(cls) -> Credential:
        """No credential at all -- ``openai-compatible`` only (ticket 1.10).

        Still ``api_key`` billing as far as the auth mode is concerned, because
        that is what the runtime *is*: an endpoint spoken to over HTTP with a
        bearer slot, rather than a vendor login. What this says is that the
        bearer slot is empty because the endpoint checks nothing, which is the
        ordinary state of a local Ollama or LM Studio box.
        :class:`~modelpass.connections.Connection` refuses it on every other
        runtime.
        """
        return cls(CredentialKind.NONE)

    @classmethod
    def env(cls, name: str) -> Credential:
        """An environment variable by **name**. The value is never stored."""
        return cls(CredentialKind.ENV, name)

    @classmethod
    def secret(cls, value: str, *, entry: str | None = None) -> Credential:
        """A pasted key, to be written to the secrets file under ``entry``.

        ``entry`` defaults to the connection's own name, which is what makes the
        rename and delete rules able to tell "this connection's own key" from "a
        key two connections share". :meth:`ConnectionManager.plan_connection`
        fills it in when it is left out.
        """
        if not isinstance(value, str) or not value.strip():
            raise ConfigError("a pasted credential is empty; there is nothing to store")
        return cls(CredentialKind.SECRET, entry, value)

    @classmethod
    def stored_secret(cls, entry: str) -> Credential:
        """An entry that is already in the secrets file; nothing new is written."""
        return cls(CredentialKind.SECRET, entry)

    @property
    def auth_mode(self) -> AuthMode:
        if self.kind is CredentialKind.NATIVE_LOGIN:
            return AuthMode.SUBSCRIPTION
        return AuthMode.API_KEY

    def _resolve(self, name: str) -> tuple[CredentialRef, PendingSecret | None]:
        """The reference to store, and the entry to write first, for connection ``name``."""
        if self.kind is CredentialKind.NATIVE_LOGIN:
            return CredentialRef.native_login(), None
        if self.kind is CredentialKind.NONE:
            return CredentialRef.none(), None
        locator = self.locator if self.locator is not None else name
        ref = CredentialRef(self.kind, locator)
        pending = None if self.value is None else PendingSecret(locator, self.value)
        return ref, pending

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form. A pointer and a flag, never a value."""
        return {
            "kind": self.kind.value,
            "locator": self.locator,
            "auth_mode": self.auth_mode.value,
            "has_pending_secret": self.value is not None,
        }


@dataclass(frozen=True, slots=True)
class ConnectionPlan:
    """A connection as it *would* be written, plus the receipt for that shape.

    The point of the split: a receipt taken for a different shape than the one
    about to be saved would be theatre, and a receipt taken after the write
    would be a report rather than a decision point. So this holds both, and
    nothing has been written when it is returned.
    """

    connection: Connection
    receipt: Receipt
    #: Held in memory for the length of the plan; excluded from ``repr`` and
    #: from :meth:`to_dict`, which report the entry name only.
    pending_secret: PendingSecret | None = field(default=None, repr=False)
    #: Whether a connection of this name is already in the store.
    exists: bool = False
    #: Whether :meth:`ConnectionManager.add_connection` may replace it.
    overwrite: bool = False
    #: ``True`` / ``False`` for a subscription connection whose vendor identity
    #: was or was not pinned; ``None`` where there is no identity to pin.
    account_pinned: bool | None = None
    notes: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.connection.name

    @property
    def ok(self) -> bool:
        """Whether the preflight says this is safe to run. Not "safe to write"."""
        return self.receipt.ok

    @property
    def secret_entry(self) -> str | None:
        """The entry a pasted key would be written to, if there is one."""
        return None if self.pending_secret is None else self.pending_secret.entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection": self.connection.to_dict(),
            "name": self.name,
            "ok": self.ok,
            "receipt": self.receipt.to_dict(),
            "secret_entry": self.secret_entry,
            "writes_secret": self.pending_secret is not None,
            "exists": self.exists,
            "overwrite": self.overwrite,
            "account_pinned": self.account_pinned,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class AddResult:
    """What :meth:`ConnectionManager.add_connection` actually wrote."""

    connection: str
    path: str
    replaced: bool = False
    secret_entry: str | None = None
    secret_path: str | None = None
    secret_permissions: SecretPermissions | None = None
    runtime_available: bool = True
    guards_configured: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection": self.connection,
            "path": self.path,
            "replaced": self.replaced,
            "secret_entry": self.secret_entry,
            "secret_path": self.secret_path,
            "secret_permissions": (
                None
                if self.secret_permissions is None
                else self.secret_permissions.to_dict()
            ),
            "runtime_available": self.runtime_available,
            "guards_configured": self.guards_configured,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class RemoveResult:
    """What :meth:`ConnectionManager.remove_connection` did, and what it kept.

    ``note`` is the sentence
    :meth:`~modelpass.secrets.SecretStore.forget_for_connection` returns: the
    one that says whether the key went with the connection and, when it did not,
    why it was left where it is. Empty when the connection referenced no stored
    secret at all.
    """

    connection: str
    removed: bool
    secret_entry: str | None = None
    secret_removed: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection": self.connection,
            "removed": self.removed,
            "secret_entry": self.secret_entry,
            "secret_removed": self.secret_removed,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class RenameResult:
    """What :meth:`ConnectionManager.rename_connection` did, including the secret."""

    old: str
    new: str
    path: str
    secret_entry: str | None = None
    secret_moved: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "old": self.old,
            "new": self.new,
            "path": self.path,
            "secret_entry": self.secret_entry,
            "secret_moved": self.secret_moved,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class EnabledResult:
    """The result of switching a connection on or off."""

    connection: str
    enabled: bool
    changed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection": self.connection,
            "enabled": self.enabled,
            "changed": self.changed,
        }


@dataclass(frozen=True, slots=True)
class GroupsResult:
    """The result of changing which groups a connection is in."""

    connection: str
    #: The groups now declared on the connection. Empty means the default group.
    groups: tuple[str, ...]
    previous: tuple[str, ...]
    changed: bool

    @property
    def effective(self) -> tuple[str, ...]:
        """What the connection is actually in, with the default filled in."""
        return self.groups or (DEFAULT_GROUP,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection": self.connection,
            "groups": list(self.groups),
            "effective": list(self.effective),
            "previous": list(self.previous),
            "changed": self.changed,
        }


class ConnectionManager:
    """The write verbs for connections and their secrets. Reached as ``bridge.manage``."""

    def __init__(self, bridge: Bridge) -> None:
        self._bridge = bridge

    # --- add -------------------------------------------------------------------

    def plan_connection(
        self,
        *,
        name: str,
        runtime: Runtime | str,
        credential: Credential | None = None,
        model: str | None = None,
        base_url: str | None = None,
        guards: Guards | None = None,
        warn_at_tokens: int | None = None,
        stop_at_tokens: int | None = None,
        description: str | None = None,
        nickname: str | None = None,
        config_dir: str | None = None,
        allow_env: Sequence[str] = (),
        experimental: bool = False,
        enabled: bool = True,
        groups: Sequence[str] = (),
        overwrite: bool = False,
    ) -> ConnectionPlan:
        """Build a connection and take its receipt. **Nothing is written.**

        ``credential`` defaults to :meth:`Credential.native_login`, the
        subscription path. The auth mode follows from it rather than being asked
        for separately. The vendor follows from ``runtime``
        (:data:`~modelpass.runtimes.VENDOR_OF` is the single source), so there is
        no vendor argument to disagree with it.

        Guards may be given either as a :class:`~modelpass.connections.Guards`
        or as the two thresholds, which are then validated by
        :meth:`Guards.for_config` -- the strict form, because this is about to
        be written to a config file where a warn above a stop is a typo the user
        would otherwise trust. Passing both is a :class:`ConfigError`.
        """
        if guards is not None and (
            warn_at_tokens is not None or stop_at_tokens is not None
        ):
            raise ConfigError(
                "pass either guards= or warn_at_tokens/stop_at_tokens, not both; "
                "two descriptions of one threshold cannot both be authoritative"
            )
        ref, pending = (credential or Credential.native_login())._resolve(name)
        if guards is None:
            guards = Guards.for_config(
                warn_at_tokens=warn_at_tokens,
                stop_at_tokens=stop_at_tokens,
                where=f"connection {name!r}",
            )
        connection = Connection(
            name=name,
            runtime=runtime,
            auth_mode=(credential or Credential.native_login()).auth_mode,
            credential_ref=ref,
            guards=guards,
            model=model,
            base_url=base_url,
            description=description,
            nickname=nickname,
            config_dir=config_dir,
            allow_env=tuple(allow_env),
            experimental=experimental,
            enabled=enabled,
            groups=tuple(groups),
        )
        receipt = receipt_for(self._bridge, connection, secrets=pending)

        notes: list[str] = []
        pinned: bool | None = None
        if connection.is_subscription and receipt.ok:
            # A missing profile is not a broken account: the probe returns
            # nothing whenever the resolved binary is not a plain file, the
            # subprocess times out, or the installed CLI predates the JSON
            # status command. None of that is a reason to refuse a connection
            # whose preflight just passed, so it is said rather than enforced.
            binding = binding_from_receipt(receipt)
            pinned = binding is not None
            if binding is None:
                notes.append(
                    f"saving {name!r} unpinned: the vendor reported no email or "
                    f"organization ID. Run 'modelpass verify {name}' to pin it later"
                )
            else:
                connection = replace(connection, account_binding=binding)
        if not receipt.runtime_available:
            notes.append(
                f"there is no adapter for runtime {connection.runtime.value!r} in "
                "this build, so this connection can be inspected and checked, and "
                "not yet run"
            )
        if not connection.guards.configured:
            notes.append(
                "no spend guards are configured for this connection; nothing bounds "
                "what one run can spend"
            )
        return ConnectionPlan(
            connection=connection,
            receipt=receipt,
            pending_secret=pending,
            exists=self._bridge.store.has(name),
            overwrite=overwrite,
            account_pinned=pinned,
            notes=tuple(notes),
        )

    def add_connection(self, plan: ConnectionPlan, *, force: bool = False) -> AddResult:
        """Write the plan: the secret first, then the connection.

        The order is the one ``modelpass connect`` uses and it is load-bearing.
        The two files are not written atomically together, so the order decides
        what a crash between them leaves behind: an unreferenced secret, which is
        inert and which ``modelpass check`` lists, rather than a connection
        pointing at an entry that does not exist.

        Refuses a plan whose preflight failed, which is what ``connect`` does
        when the receipt is not OK. ``force`` writes it anyway -- for the caller
        who is configuring a connection against an environment variable that
        will exist later, and who has seen the receipt say so.
        """
        connection = plan.connection
        if plan.exists and not plan.overwrite:
            raise DuplicateConnection(connection.name)
        if not plan.receipt.ok and not force:
            raise PreflightFailed(
                f"not writing connection {connection.name!r}: the preflight could "
                f"not confirm it is safe to run ({plan.receipt.problem or 'unknown reason'}). "
                "Fix it, or pass force=True to write it anyway"
            )
        secrets = self._bridge.secrets
        entry: str | None = None
        permissions: SecretPermissions | None = None
        if plan.pending_secret is not None:
            secrets.set(plan.pending_secret.entry, plan.pending_secret.value)
            entry = plan.pending_secret.entry
            permissions = secrets.permissions()
        self._bridge.store.add(connection, overwrite=True)
        return AddResult(
            connection=connection.name,
            path=str(self._bridge.store.path),
            replaced=plan.exists,
            secret_entry=entry,
            secret_path=None if entry is None else str(secrets.path),
            secret_permissions=permissions,
            runtime_available=plan.receipt.runtime_available,
            guards_configured=connection.guards.configured,
            notes=plan.notes,
        )

    # --- remove ----------------------------------------------------------------

    def remove_connection(self, name: str) -> RemoveResult:
        """Delete a connection, then apply the secret delete rule (R14).

        The connection goes first so that a crash between the two writes leaves
        an inert orphan rather than a connection pointing at an entry that is
        gone. The entry itself is removed only when it was unambiguously *this*
        connection's -- named after it, and referenced by nothing else. Every
        other case leaves the key where it is and says why, because an orphaned
        secret is recoverable and a deleted one is not.
        """
        try:
            connection: Connection | None = self._bridge.connection(name)
        except SubpassError:
            # A connection this build cannot parse can still be deleted; the
            # store's own `drop` path handles it. There is then no credential
            # reference to reason about, which is exactly what `connection=None`
            # means downstream.
            connection = None
        self._bridge.store.remove(name)

        entry: str | None = None
        removed_secret = False
        note = ""
        if connection is not None:
            ref = connection.credential_ref
            if ref.kind is CredentialKind.SECRET:
                entry = ref.locator
            secrets = self._bridge.secrets
            had = bool(entry) and secrets.has(entry or "")
            note = secrets.forget_for_connection(connection, self._bridge.connections())
            removed_secret = had and not secrets.has(entry or "")
        return RemoveResult(
            connection=name,
            removed=True,
            secret_entry=entry,
            secret_removed=removed_secret,
            note=note,
        )

    # --- rename ----------------------------------------------------------------

    def rename_connection(self, old: str, new: str) -> RenameResult:
        """Give a connection a new name, moving its secret entry when it is its.

        The two files can otherwise only be kept in step by a delete and an add
        in the right order, which orphans or destroys a key depending on which
        way round you do it. The rule is the delete rule: the entry moves when it
        is named after the connection and nothing else references it, and
        otherwise it stays and this says so.
        """
        bridge = self._bridge
        connection = bridge.connection(old)
        connections = dict(bridge.store.load())
        if new in connections:
            raise ConfigError(
                f"connection {new!r} already exists; pick another name or remove it first"
            )
        renamed = replace(connection, name=new)

        secrets = bridge.secrets
        ref = connection.credential_ref
        entry = ref.locator if ref.kind is CredentialKind.SECRET else None
        others = (
            secrets.referenced_by(entry, [c for c in connections.values() if c.name != old])
            if entry
            else ()
        )
        move_secret = bool(entry) and entry == old and not others and secrets.has(entry)
        note = ""
        if entry and not move_secret:
            if others:
                named = ", ".join(repr(name) for name in others)
                note = (
                    f"secret entry {entry!r} stays where it is: also referenced by {named}"
                )
            elif entry != old:
                note = (
                    f"secret entry {entry!r} stays where it is: it is a shared locator "
                    f"rather than the connection's own name"
                )
            else:
                note = f"there is no secret entry {entry!r} to move"

        if move_secret:
            # Secret first, then the connection, for the reason add writes them
            # in that order: a crash in between leaves an inert extra entry
            # rather than a connection pointing at a name that does not exist.
            secrets.set(new, secrets.get(entry))
            renamed = replace(renamed, credential_ref=CredentialRef.parse(f"secret:{new}"))

        del connections[old]
        connections[new] = renamed
        bridge.store.save(connections.values())
        if move_secret:
            secrets.remove(entry)
            note = f"moved secret entry {entry!r} to {new!r}"
        return RenameResult(
            old=old,
            new=new,
            path=str(bridge.store.path),
            secret_entry=new if move_secret else entry,
            secret_moved=move_secret,
            note=note,
        )

    # --- enable ----------------------------------------------------------------

    def set_enabled(self, name: str, enabled: bool) -> EnabledResult:
        """Switch a connection on or off. ``enabled = false`` refuses runs, not looking."""
        connection = self._bridge.connection(name)
        changed = connection.enabled != enabled
        self._bridge.store.add(connection.with_enabled(enabled), overwrite=True)
        return EnabledResult(connection=name, enabled=enabled, changed=changed)

    # --- groups ----------------------------------------------------------------

    def set_groups(self, name: str, groups: Sequence[str]) -> GroupsResult:
        """Replace the groups a connection declares.

        Replace rather than merge, and the whole list rather than one entry: a
        group is membership, and "put this in ``cheap``" and "put this in
        ``cheap`` *only*" are different instructions that an add-one verb cannot
        tell apart. An empty sequence returns the connection to the default
        group, which is the same gesture as deleting the ``groups`` key.

        The group itself needs no creating and no deleting: it exists while
        somebody is in it (:func:`~modelpass.connections.groups_of`). Moving the
        last member out of a group is how a group goes away, and there is
        nothing left over pointing at it.
        """
        connection = self._bridge.connection(name)
        wanted = tuple(groups)
        previous = connection.groups
        # Built before the write so an invalid group name is refused by
        # Connection's own validation rather than reaching the file.
        updated = connection.with_groups(wanted)
        changed = previous != updated.groups
        self._bridge.store.add(updated, overwrite=True)
        return GroupsResult(
            connection=name,
            groups=updated.groups,
            previous=previous,
            changed=changed,
        )
