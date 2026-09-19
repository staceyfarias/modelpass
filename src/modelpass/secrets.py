"""The secrets file: ``~/.modelpass/secrets.toml`` (R14).

A second versioned artifact, beside ``connections.toml`` and deliberately *not*
inside it. The whole value of the connection file is that it is shareable and
committable -- the header it writes says so, and the README repeats it. One key
in that file destroys the property for everybody, including the users who never
wanted an explicit key. So the pointer lives in the shareable file and the value
lives here, in a file that is never shared.

Three rules this module holds to, and the tests are their executable form:

1. **A value never leaves except through** :meth:`SecretStore.get`. No value
   reaches a log line, an exception message, a ``to_dict()``, a receipt or the
   run log. :func:`~modelpass.preflight.credential_fingerprint` is the only
   thing about a key that may.
2. **A protection is reported, never claimed.** POSIX gets ``0600`` and a read
   that notices group or other bits says so. Windows gets a best-effort
   ``icacls`` whose *actual* result -- success or the reason it failed -- is
   recorded in the file and printed by ``modelpass check``.
3. **An orphan is recoverable; a deleted key is not.** So the delete rules lean
   toward leaving an entry in place and saying so, and a secret that a
   connection still references is never removed.

The compatibility policy is the one ticket 0.1 wrote for the connection file,
applied here: this build reads any version, refuses to *write* a version newer
than its own, carries unknown keys through verbatim, and treats an unreadable
file as "no secrets" on read and as a refusal on write.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import _toml
from .connections import Connection, CredentialKind
from .errors import ConfigError, NoSuchSecret, SecretStillReferenced
from .store import default_home

__all__ = [
    "SECRETS_VERSION",
    "PendingSecret",
    "SecretPermissions",
    "SecretSource",
    "SecretStore",
]

SECRETS_VERSION = 1
_FILENAME = "secrets.toml"

#: The keys a ``[secrets.<entry>]`` table understands. Anything else is carried
#: through untouched, for the reason the connection file carries unknown keys:
#: this build does not know what it means, which is precisely why it has no
#: business rewriting or dropping it.
_ENTRY_KEYS = frozenset({"apiKey"})

#: The keys the ``[permissions]`` note understands.
_PERMISSION_KEYS = frozenset({"windowsAcl", "windowsAclDetail"})

_HEADER = f"""\
# modelpass secrets -- NOT shareable, NOT committable.
#
# This file holds credential *values*. Its sibling connections.toml holds only
# pointers; that one is safe to share and this one never is.
#
# A key in here does nothing on its own. It is used when, and only when, a
# connection names it as credentialRef = "secret:<entry>" -- an unreferenced
# entry is inert, which is the same property an unreferenced environment
# variable has (D2).
#
# Permissions. On macOS and Linux this file is created 0600 and modelpass check
# says so if the mode has since been widened. On Windows modelpass runs, once at
# creation, icacls <path> /inheritance:r /grant:r "<you>":F, and records below
# whether that actually worked -- the floor either way is the ACL on your user
# profile directory, which is what %USERPROFILE% carries.
#
# Compatibility: this build reads any version and refuses to write one newer
# than its own. Keys it does not recognise are preserved exactly as written.

version = {SECRETS_VERSION}
"""


class SecretSource(Protocol):
    """Anything that can answer "what is the value stored under this entry".

    :class:`SecretStore` is the one that reads the file; :class:`PendingSecret`
    is the one ``modelpass connect`` uses to take a receipt against a key that
    has been pasted but not yet written, so that the promise "nothing is written
    before the receipt is shown" survives the arrival of explicit keys.
    """

    def get(self, entry: str) -> str: ...  # pragma: no cover - protocol


@dataclass(frozen=True, slots=True)
class PendingSecret:
    """One entry that exists only in memory, for the length of one preflight."""

    entry: str
    value: str

    def __repr__(self) -> str:
        """Redacted, and deliberately not the generated one.

        This object is the one place in modelpass that holds a pasted key, and
        it is reachable from a plan a caller may well print while debugging. The
        dataclass-generated ``repr`` would put the key in that output.
        """
        return f"PendingSecret(entry={self.entry!r}, value=<redacted>)"

    def get(self, entry: str) -> str:
        if entry != self.entry:
            raise NoSuchSecret(entry, "<not yet written>", (self.entry,))
        return self.value


@dataclass(frozen=True, slots=True)
class SecretPermissions:
    """What is actually true about this file's permissions, and nothing more.

    Every field is an observation. ``note`` is the sentence ``modelpass check``
    prints, and it is written so that a protection that was not applied is never
    described as though it were.
    """

    path: str
    exists: bool
    mode: int | None = None
    group_or_other_readable: bool = False
    windows_acl: str | None = None
    windows_acl_detail: str = ""

    @property
    def note(self) -> str:
        if not self.exists:
            return f"no secrets file yet ({self.path})"
        if os.name == "nt":
            if self.windows_acl == "restricted":
                return "the secrets file is restricted to your account"
            if self.windows_acl == "failed":
                reason = self.windows_acl_detail or "no reason was recorded"
                return (
                    f"could not restrict the secrets file: {reason}; the profile "
                    "directory's own ACL is the floor"
                )
            return (
                "the secrets file carries no record of an ACL attempt; the profile "
                "directory's own ACL is the floor"
            )
        mode = "unknown" if self.mode is None else f"{self.mode:04o}"
        if self.group_or_other_readable:
            return (
                f"the secrets file is readable beyond your account (mode {mode}); "
                f"run: chmod 600 {self.path}"
            )
        return f"the secrets file is restricted to your account (mode {mode})"

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11). Permissions, never a value."""
        return {
            "path": self.path,
            "exists": self.exists,
            "mode": None if self.mode is None else f"{self.mode:04o}",
            "group_or_other_readable": self.group_or_other_readable,
            "windows_acl": self.windows_acl,
            "windows_acl_detail": self.windows_acl_detail,
            "note": self.note,
        }


class SecretStore:
    """Reads and writes ``secrets.toml`` under the modelpass home.

    ``root`` defaults to :func:`~modelpass.store.default_home`, the same
    resolution the connection store uses, so the two files are always siblings.
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
        """The raw file, or an empty document.

        A file that cannot be read or parsed reads as **no secrets** rather than
        raising. Every read path in modelpass degrades to "nothing configured"
        (R10), and a corrupt secrets file must not make ``modelpass list``
        unable to speak. Writes are the opposite: :meth:`_write_refusal`
        refuses, because rewriting a file this build could not read would
        destroy whatever is in it.
        """
        if not self.exists():
            return {}
        try:
            with self.path.open("rb") as handle:
                return tomllib.load(handle)
        except (tomllib.TOMLDecodeError, OSError):
            return {}

    def unreadable_reason(self) -> str | None:
        """Why this build could not read the file at all, or ``None`` (R10).

        Separate from :meth:`_write_refusal` because the two answers are not the
        same: a file written by a *newer* build reads fine and must not be
        rewritten, while a file that cannot be parsed or opened reads as no
        secrets and must not be rewritten either. Only the second is a reason to
        tell the user their secrets are not being seen.
        """
        if not self.exists():
            return None
        try:
            with self.path.open("rb") as handle:
                tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            return f"{self.path} is not valid TOML: {exc}"
        except OSError as exc:
            return f"cannot read {self.path}: {exc}"
        return None

    def notes(self) -> tuple[str, ...]:
        """Anything worth saying about the file, in reading order. Usually none."""
        reason = self.unreadable_reason()
        if reason is not None:
            return (
                "the secrets file could not be read, so modelpass is reading it as "
                f"no secrets and will not rewrite it: {reason}",
            )
        version = self._write_refusal()
        return () if version is None else (version,)

    def _write_refusal(self) -> str | None:
        """Why this build cannot rewrite the file, or ``None``."""
        reason = self.unreadable_reason()
        if reason is not None:
            return reason
        if not self.exists():
            return None
        try:
            with self.path.open("rb") as handle:
                raw = tomllib.load(handle)
        except (tomllib.TOMLDecodeError, OSError) as exc:  # pragma: no cover - raced
            return f"cannot read {self.path}: {exc}"
        version = raw.get("version", SECRETS_VERSION)
        if isinstance(version, bool) or not isinstance(version, int):
            return f"{self.path}: secrets version must be a whole number, got {version!r}"
        if version > SECRETS_VERSION:
            return (
                f"{self.path} declares secrets version {version}; this build of "
                f"modelpass writes version {SECRETS_VERSION}. It can still read the "
                "file, but a newer build of modelpass is needed to change it."
            )
        return None

    def _table(self) -> dict[str, Any]:
        raw = self._read().get("secrets", {})
        return dict(raw) if isinstance(raw, Mapping) else {}

    def entries(self) -> tuple[str, ...]:
        """Every entry name, in name order. A missing file means none."""
        return tuple(sorted(self._table()))

    def has(self, entry: str) -> bool:
        return entry in self._table()

    def get(self, entry: str) -> str:
        """The value stored under ``entry``.

        The **only** way a value leaves this module. Raises
        :class:`~modelpass.errors.NoSuchSecret` when the entry is absent or
        holds no usable value -- with a message that names the entry and the
        file, and says nothing whatever about any value.
        """
        table = self._table()
        record = table.get(entry)
        value = record.get("apiKey") if isinstance(record, Mapping) else None
        if not isinstance(value, str) or not value.strip():
            raise NoSuchSecret(entry, str(self.path), tuple(sorted(table)))
        return value

    def referenced_by(
        self, entry: str, connections: Iterable[Connection]
    ) -> tuple[str, ...]:
        """The names of the connections pointing at ``entry``, in name order."""
        return tuple(
            sorted(
                connection.name
                for connection in connections
                if connection.credential_ref.kind is CredentialKind.SECRET
                and connection.credential_ref.locator == entry
            )
        )

    def orphans(self, connections: Iterable[Connection]) -> tuple[str, ...]:
        """Entries no connection references. Reported, never auto-collected."""
        wanted = {
            connection.credential_ref.locator
            for connection in connections
            if connection.credential_ref.kind is CredentialKind.SECRET
        }
        return tuple(entry for entry in self.entries() if entry not in wanted)

    def permissions(self) -> SecretPermissions:
        """What is actually true about the file's permissions right now."""
        if not self.exists():
            return SecretPermissions(path=str(self.path), exists=False)
        acl, detail = self._recorded_acl()
        try:
            mode = stat.S_IMODE(self.path.stat().st_mode)
        except OSError:
            return SecretPermissions(
                path=str(self.path),
                exists=True,
                windows_acl=acl,
                windows_acl_detail=detail,
            )
        return SecretPermissions(
            path=str(self.path),
            exists=True,
            mode=mode,
            group_or_other_readable=bool(mode & 0o077),
            windows_acl=acl,
            windows_acl_detail=detail,
        )

    def _recorded_acl(self) -> tuple[str | None, str]:
        table = self._read().get("permissions")
        if not isinstance(table, Mapping):
            return None, ""
        acl = table.get("windowsAcl")
        detail = table.get("windowsAclDetail")
        return (
            acl if isinstance(acl, str) else None,
            detail if isinstance(detail, str) else "",
        )

    # --- writing ---------------------------------------------------------------

    def set(self, entry: str, api_key: str) -> None:
        """Write ``api_key`` under ``entry``, replacing whatever was there."""
        from .connections import CredentialRef  # local: validates the entry name

        CredentialRef(CredentialKind.SECRET, entry)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigError(
                f"secret entry {entry!r} was given an empty value; there is nothing "
                "to store"
            )
        table = self._table()
        existing = table.get(entry)
        record = dict(existing) if isinstance(existing, Mapping) else {}
        record["apiKey"] = api_key
        table[entry] = record
        self._save(table)

    def remove(self, entry: str, *, connections: Iterable[Connection] = ()) -> None:
        """Delete ``entry``, refusing while any connection still references it."""
        table = self._table()
        if entry not in table:
            raise NoSuchSecret(entry, str(self.path), tuple(sorted(table)))
        referencing = self.referenced_by(entry, connections)
        if referencing:
            raise SecretStillReferenced(entry, referencing)
        del table[entry]
        self._save(table)

    def rename(self, old: str, new: str) -> None:
        """Move an entry to a new name, value untouched."""
        from .connections import CredentialRef

        CredentialRef(CredentialKind.SECRET, new)
        table = self._table()
        if old not in table:
            raise NoSuchSecret(old, str(self.path), tuple(sorted(table)))
        if new in table:
            raise ConfigError(
                f"secret entry {new!r} already exists; remove it first or pick "
                "another entry name"
            )
        table[new] = table.pop(old)
        self._save(table)

    def forget_for_connection(
        self, connection: Connection, remaining: Iterable[Connection]
    ) -> str:
        """Apply the delete rule for a connection that has just been removed.

        The rule, and the reason it is this way round: an orphaned secret is
        inert and recoverable, a deleted key is neither. So the entry goes only
        when it was unambiguously *this* connection's -- the locator equals the
        connection name -- and nothing else points at it. Every other case
        leaves it and says so.

        Returns the sentence to show the user. The caller deletes the connection
        first and calls this second, so a crash between the two leaves an
        unreferenced secret rather than a connection pointing at nothing.
        """
        ref = connection.credential_ref
        if ref.kind is not CredentialKind.SECRET or not ref.locator:
            return ""
        entry = ref.locator
        if not self.has(entry):
            return f"no secret entry {entry!r} to remove"
        others = self.referenced_by(entry, remaining)
        if others:
            named = ", ".join(repr(name) for name in others)
            return (
                f"left secret entry {entry!r} in place: still referenced by {named}"
            )
        if entry != connection.name:
            return (
                f"left secret entry {entry!r} in place: it is a shared locator rather "
                f"than connection {connection.name!r}'s own name, so removing it is "
                "not this deletion's call. 'modelpass check' lists it as an orphan"
            )
        self.remove(entry)
        return f"removed secret entry {entry!r}"

    def _save(self, table: Mapping[str, Any]) -> None:
        reason = self._write_refusal()
        if reason is not None:
            raise ConfigError(reason)
        raw = self._read()
        document: dict[str, Any] = {}
        permissions = self._permissions_table_to_write()
        if permissions:
            document["permissions"] = permissions
        if table:
            raw_secrets = raw.get("secrets", {})
            out: dict[str, Any] = {}
            for entry in sorted(table):
                emitted = dict(table[entry])
                existing = (
                    raw_secrets.get(entry) if isinstance(raw_secrets, Mapping) else None
                )
                if isinstance(existing, Mapping):
                    for key, value in existing.items():
                        if key not in _ENTRY_KEYS and key not in emitted:
                            emitted[key] = value
                out[entry] = emitted
            document["secrets"] = out
        # Carry through any top-level key this build did not read.
        for key, value in raw.items():
            if key not in {"version", "secrets", "permissions"}:
                document.setdefault(key, value)
        body = _toml.dumps(document) if document else ""
        self._write(_HEADER + ("\n" + body if body else ""))

    def _permissions_table_to_write(self) -> dict[str, Any]:
        """The ``[permissions]`` note to re-emit, preserving what is recorded.

        Carried rather than recomputed: it records what happened *when the file
        was created*, and re-running ``icacls`` on every write would turn a
        one-time observation into a claim refreshed behind the user's back.
        """
        raw = self._read().get("permissions")
        if not isinstance(raw, Mapping):
            return {}
        # Unknown keys inside it are carried too, by the same policy the rest of
        # the file follows.
        return dict(raw)

    def _write(self, text: str) -> None:
        created = not self.exists()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            try:
                self.root.chmod(0o700)
            except (OSError, NotImplementedError):
                pass
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=self.root,
                prefix=".secrets-",
                suffix=".tmp",
                delete=False,
            )
            try:
                with handle:
                    handle.write(text)
                # NamedTemporaryFile already creates at 0600 and os.replace
                # preserves the source mode, so this is belt and braces -- but
                # the file it protects is the one file in modelpass that holds a
                # value, and a test asserts the result rather than trusting the
                # stdlib to keep behaving this way.
                try:
                    os.chmod(handle.name, 0o600)
                except (OSError, NotImplementedError):  # pragma: no cover - Windows
                    pass
                os.replace(handle.name, self.path)
            except BaseException:
                Path(handle.name).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise ConfigError(f"cannot write {self.path}: {exc}") from exc
        if created and os.name == "nt":
            self._record_windows_acl(restrict_windows_acl(self.path))

    def _record_windows_acl(self, outcome: tuple[str, str]) -> None:
        """Store what ``icacls`` actually did, so ``check`` can report it honestly."""
        status, detail = outcome
        raw = self._read()
        document: dict[str, Any] = {"permissions": {"windowsAcl": status}}
        if detail:
            document["permissions"]["windowsAclDetail"] = detail
        secrets = raw.get("secrets")
        if isinstance(secrets, Mapping) and secrets:
            document["secrets"] = dict(secrets)
        for key, value in raw.items():
            if key not in {"version", "secrets", "permissions"}:
                document.setdefault(key, value)
        body = _toml.dumps(document)
        try:
            self._write(_HEADER + "\n" + body)
        except ConfigError:  # pragma: no cover - best effort, never fatal
            pass


def restrict_windows_acl(path: Path) -> tuple[str, str]:
    """Best-effort ``icacls`` on a freshly created secrets file.

    Returns ``("restricted", "")`` or ``("failed", reason)``. It never raises and
    it never claims more than happened: the documented floor is the ACL that
    ``%USERPROFILE%`` already carries, and this is one layer on top of that
    which may or may not have been applied. ``modelpass check`` prints whichever
    of the two is true.
    """
    user = os.environ.get("USERNAME") or ""
    if not user:
        return "failed", "USERNAME is not set, so there is no account to grant to"
    domain = os.environ.get("USERDOMAIN")
    account = f"{domain}\\{user}" if domain else user
    try:
        completed = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{account}:F"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except OSError as exc:
        return "failed", f"icacls could not be run ({exc})"
    except subprocess.TimeoutExpired:
        return "failed", "icacls did not finish within 20 seconds"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        reason = detail[0] if detail else f"icacls exited {completed.returncode}"
        return "failed", reason
    return "restricted", ""
