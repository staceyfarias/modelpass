"""Durable, application-scoped conversations above stateless calls/native sessions.

This is an opt-in private content store, not the global connection/run-log store.
History mode replays completed text turns; native mode resumes the vendor thread.
Neither mode silently changes a conversation's connection, requested model or origin.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .capabilities import Capability
from .errors import ConversationBusy, ConversationError, ConversationNotFound
from .runtimes import Runtime
from .tools import ToolDef, normalize_tools
from .types import AgentEvent, Message, Role, TerminalEvent, TextDeltaEvent

if TYPE_CHECKING:
    from .bridge import Bridge
    from .connections import Connection

__all__ = [
    "Conversation",
    "ConversationBusy",
    "ConversationError",
    "ConversationInfo",
    "ConversationNotFound",
    "ConversationStore",
    "ConversationTurn",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _text(value: str, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ConversationError(f"{name} must be a {'possibly empty ' if empty else ''}string")
    return value


def _binding(connection: Connection) -> str:
    # Credential pointers only; rotation of a secret at the same pointer is allowed.
    identity = {
        "runtime": connection.runtime.value,
        "auth": connection.auth_mode.value,
        "credential": connection.credential_ref.to_str(),
        "config_dir": connection.config_dir,
        "base_url": connection.base_url,
        "account": connection.account_binding.to_dict() if connection.account_binding else None,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ConversationInfo:
    id: str
    application: str
    application_name: str
    scope: str | None
    scope_name: str | None
    title: str
    connection: str
    model: str
    runtime: str
    auth_mode: str
    mode: str
    created_at: str
    updated_at: str
    archived: bool = False
    native_session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationTurn:
    id: str
    conversation_id: str
    message: str
    text: str
    status: str
    created_at: str
    finished_at: str | None
    events: tuple[dict[str, Any], ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["events"] = list(self.events)
        return data


class ConversationStore:
    """Private SQLite store, scoped to one explicit application.

    Multiple applications may share ``root``. Every lookup, mutation and list
    filters by application; scope is an additional project/workspace filter.
    This is local namespacing, not authentication between untrusted programs.
    Construction, create/list/open/rename/archive spend no model allowance.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        bridge: Bridge,
        application: str,
        application_name: str | None = None,
        keep_alive: bool = False,
    ) -> None:
        self.application = _text(application, "application")
        self.application_name = _text(
            application if application_name is None else application_name, "application_name"
        )
        self.bridge = bridge
        self.keep_alive = keep_alive
        self._live: dict[str, dict[str, Any]] = {}
        self._live_lock = threading.RLock()
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "conversations.sqlite3"
        with self._db() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ConversationError(f"Unsupported conversation database version: {version}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, application TEXT NOT NULL,
                    updated_at TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS conversation_origin
                    ON conversations(application, updated_at);
                CREATE TABLE IF NOT EXISTS turns (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT UNIQUE NOT NULL, conversation_id TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS turn_conversation ON turns(conversation_id, seq);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_id TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS event_turn ON events(turn_id, seq);
                PRAGMA user_version = 1;
            """)
        if os.name != "nt":
            self.path.chmod(0o600)

    @contextmanager
    def _db(self):
        try:
            db = sqlite3.connect(self.path, timeout=10)
        except sqlite3.Error as exc:
            raise ConversationError("Conversation database could not be opened") from exc
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        except sqlite3.Error as exc:
            raise ConversationError("Conversation database operation failed") from exc
        finally:
            db.close()

    def _read(self, identifier: str) -> dict[str, Any]:
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ConversationNotFound("Conversation not found for this application")
        with self._db() as db:
            row = db.execute(
                "SELECT data FROM conversations WHERE id=? AND application=?",
                (identifier, self.application),
            ).fetchone()
        if row is None:
            raise ConversationNotFound("Conversation not found for this application")
        return json.loads(row[0])

    @contextmanager
    def _lock(self, identifier: str):
        self._read(identifier)  # Verify origin and id before creating a lock path.
        directory = self.root / "locks"
        directory.mkdir(exist_ok=True, mode=0o700)
        with (directory / identifier).open("a+b") as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ConversationBusy("This conversation already has an active operation") from exc
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _write(self, data: dict[str, Any]) -> None:
        with self._db() as db:
            db.execute(
                "UPDATE conversations SET updated_at=?, data=? WHERE id=? AND application=?",
                (data["updated_at"], json.dumps(data), data["id"], self.application),
            )

    def create(
        self,
        *,
        connection: str,
        model: str | None = None,
        title: str = "New conversation",
        scope: str | None = None,
        scope_name: str | None = None,
        system_prompt: str | None = None,
        mode: str = "history",
        seed_history: Sequence[Message | Mapping[str, Any]] = (),
    ) -> Conversation:
        """Create a separate conversation. History is empty unless explicitly seeded.

        ``history`` uses Bridge.chat/achat on any runtime; ``native`` explicitly
        uses new_chat/resume_chat and never falls back. Native sends can rebind
        handlers for fixed tool definitions; keep_alive retains the vendor handle.
        Schema sessions remain the existing Session API.
        """
        selected = self.bridge.store.get(_text(connection, "connection"))
        chosen_model = _text(model if model is not None else selected.model, "model")
        _text(title, "title")
        if scope is not None:
            _text(scope, "scope")
        if scope_name is not None:
            _text(scope_name, "scope_name", empty=True)
        if system_prompt is not None:
            _text(system_prompt, "system_prompt", empty=True)
        if mode not in ("history", "native"):
            raise ConversationError("mode must be 'history' or 'native'")
        seeds = [Message.coerce(m).to_dict() for m in seed_history]
        if any(m["role"] == "system" for m in seeds):
            raise ConversationError("Use system_prompt for instructions, not seed_history")
        if mode == "native":
            self.bridge._require_sessions(selected.runtime, "managed native conversation")
            if seeds:
                raise ConversationError("Native history cannot be seeded; use history mode")
        stamp, identifier = _now(), uuid4().hex
        info = ConversationInfo(
            identifier,
            self.application,
            self.application_name,
            scope,
            scope_name,
            title,
            selected.name,
            chosen_model,
            selected.runtime.value,
            selected.auth_mode.value,
            mode,
            stamp,
            stamp,
        )
        data = {
            **info.to_dict(),
            "binding": _binding(selected),
            "system_prompt": system_prompt,
            "seed_history": seeds,
        }
        with self._db() as db:
            db.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                (identifier, self.application, stamp, json.dumps(data)),
            )
        return Conversation(self, identifier)

    def close(self) -> None:
        """Release retained idle vendor sessions; never cancel another active turn.

        A store with keep_alive=True must be retained by its host and closed on
        shutdown. At most four native sessions are kept; idle ones are evicted.
        Persisted vendor ids and the app's conversation journal remain intact.
        """
        with self._live_lock:
            if any(entry["active"] for entry in self._live.values()):
                raise ConversationBusy("Cannot close a store with an active conversation turn")
            entries = list(self._live.values())
            self._live.clear()
        for entry in entries:
            entry["session"].close()

    def _release_native(self, identifier, session, data, successful):
        with self._live_lock:
            entry = self._live.get(identifier)
            if successful and entry and entry["session"] is session:
                entry["active"] = False
                entry["marker"] = data.get("native_revision")
                entry["handlers"] = {}
                return
            if entry and entry["session"] is session:
                self._live.pop(identifier)
        session.close()

    def open(self, identifier: str) -> Conversation:
        self._read(identifier)
        return Conversation(self, identifier)

    def list(
        self,
        *,
        scope: str | None = None,
        connection: str | None = None,
        model: str | None = None,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationInfo]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ConversationError("limit must be a positive integer")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ConversationError("offset must be a nonnegative integer")
        with self._db() as db:
            rows = db.execute(
                "SELECT data FROM conversations WHERE application=? "
                "ORDER BY updated_at DESC, id DESC",
                (self.application,),
            ).fetchall()
        result = []
        for row in rows:
            data = json.loads(row[0])
            if (
                (scope is not None and scope != data["scope"])
                or (connection is not None and connection != data["connection"])
                or (model is not None and model != data["model"])
                or (data["archived"] and not include_archived)
            ):
                continue
            result.append(self._info(data))
        return result[offset : offset + limit]

    @staticmethod
    def _info(data: dict[str, Any]) -> ConversationInfo:
        return ConversationInfo(**{key: data[key] for key in ConversationInfo.__dataclass_fields__})


class Conversation:
    """A durable app-owned chat, not a claim of a vendor-held session.

    Calling send/asend starts work only when the iterator is consumed. Each
    turn owns a nonblocking process-safe lock. Closing the iterator cancels only
    that run and archives its partial output, excluded from future replay.
    """

    def __init__(self, store: ConversationStore, identifier: str) -> None:
        self.store, self.id = store, identifier

    @property
    def info(self) -> ConversationInfo:
        return self.store._info(self.store._read(self.id))

    def rename(self, title: str) -> ConversationInfo:
        _text(title, "title")
        return self._edit(title=title)

    def archive(self, archived: bool = True) -> ConversationInfo:
        if not isinstance(archived, bool):
            raise ConversationError("archived must be a boolean")
        return self._edit(archived=archived)

    def _edit(self, **changes: Any) -> ConversationInfo:
        with self.store._lock(self.id):
            data = self.store._read(self.id)
            data.update(changes, updated_at=_now())
            self.store._write(data)
        return self.info

    def turns(self) -> list[ConversationTurn]:
        self.store._read(self.id)
        with self.store._db() as db:
            rows = db.execute(
                "SELECT data FROM turns WHERE conversation_id=? ORDER BY seq", (self.id,)
            ).fetchall()
            result = []
            for row in rows:
                data = json.loads(row[0])
                events = tuple(
                    json.loads(e[0])
                    for e in db.execute(
                        "SELECT data FROM events WHERE turn_id=? ORDER BY seq", (data["id"],)
                    )
                )
                # A process crash may have left the final turn unfinished; events
                # are durable before they are yielded, so partial text is visible.
                if data["status"] == "running":
                    data["text"] = "".join(e["text"] for e in events if e["type"] == "text_delta")
                result.append(ConversationTurn(**data, events=events))
        return result

    def history(self) -> list[Message]:
        """Replayable successful text turns; native mode reads the vendor history.

        turns() is the app's observed turn/event journal, including failures.
        Native history is never reconstructed from this journal.
        """
        with self.store._lock(self.id):
            data = self.store._read(self.id)
            if data["mode"] == "native":
                if not data["native_session_id"]:
                    return []
                signature = json.loads(data.get("native_signature", "{}"))
                tools = normalize_tools(signature.get("tools"))
                session = self._native(data, tools, signature.get("launch"))
                try:
                    return session.get_history()
                finally:
                    self.store._release_native(self.id, session, data, True)
            return self._history(data)

    def _history(self, data: dict[str, Any]) -> list[Message]:
        result = [Message.coerce(m) for m in data["seed_history"]]
        for turn in self.turns():
            if turn.status == "ok":
                result.extend(
                    (Message(Role.USER, turn.message), Message(Role.ASSISTANT, turn.text))
                )
        return result

    def _check_binding(self, data: dict[str, Any]) -> None:
        selected = self.store.bridge.store.get(data["connection"])
        if _binding(selected) != data["binding"]:
            raise ConversationError(
                "Conversation connection identity changed; create a new conversation"
            )

    def _native(self, data: dict[str, Any], tools=(), launch=None):
        self._check_binding(data)
        launch = launch or {}
        signature = json.dumps(
            {"tools": [tool.to_dict() for tool in tools], "launch": launch}, sort_keys=True
        )
        recorded = data.get("native_signature")
        if recorded is not None and recorded != signature:
            raise ConversationError("Native tools/guards changed; start a separate conversation")
        if recorded is None and data["native_session_id"] and tools:
            raise ConversationError(
                "This native conversation was created without tool bindings; "
                "start a separate conversation"
            )
        data["native_signature"] = signature
        with self.store._live_lock:
            entry = self.store._live.get(self.id)
            if entry and (
                entry["marker"] != data.get("native_revision") or entry["session"].closed
            ):
                self.store._live.pop(self.id)
                entry["session"].close()
                entry = None
            if entry:
                entry["handlers"] = {tool.name: tool.handler for tool in tools}
                entry["active"] = True
                return entry["session"]
            entry = {"handlers": {tool.name: tool.handler for tool in tools}, "active": True}

        def route(name):
            async def handler(args):
                function = entry["handlers"].get(name) if entry["active"] else None
                if function is None:
                    raise ConversationError("No active handler for this native turn")
                if inspect.iscoroutinefunction(function):
                    return await function(args)
                result = await asyncio.to_thread(function, args)
                return await result if inspect.isawaitable(result) else result

            return handler

        routed = tuple(
            ToolDef(tool.name, tool.description, tool.parameters, route(tool.name))
            for tool in tools
        )
        kwargs = {
            "connection": data["connection"],
            "model": data["model"],
            "expect_auth_mode": data["auth_mode"],
            "project_folder": str(self.store.root / "native" / self.id),
            "tools": routed,
            **launch,
        }
        Path(kwargs["project_folder"]).mkdir(parents=True, exist_ok=True)
        if data["native_session_id"]:
            carries = self.store.bridge.registry.supports(
                Runtime(data["runtime"]), Capability.RESUME_CARRIES_SYSTEM_PROMPT
            )
            if not carries:
                kwargs["system_prompt"] = data["system_prompt"] or ""
            session = self.store.bridge.resume_chat(session_id=data["native_session_id"], **kwargs)
        else:
            session = self.store.bridge.new_chat(system_prompt=data["system_prompt"], **kwargs)
        if self.store.keep_alive:
            with self.store._live_lock:
                if len(self.store._live) >= 4:
                    idle = next(
                        (key for key, item in self.store._live.items() if not item["active"]), None
                    )
                    if idle is not None:
                        self.store._live.pop(idle)["session"].close()
                if len(self.store._live) < 4:
                    entry.update(session=session, marker=data.get("native_revision"))
                    self.store._live[self.id] = entry
        return session

    def _begin(self, data: dict[str, Any], message: str) -> dict[str, Any]:
        self._check_binding(data)
        if data["archived"]:
            raise ConversationError("Unarchive this conversation before sending another turn")
        previous = self.turns()
        if (
            data["mode"] == "native"
            and previous
            and previous[-1].status != "ok"
            and previous[-1].events
        ):
            raise ConversationError(
                "Previous native turn did not finish successfully; its runtime "
                "state is uncertain. Start a separate conversation; no history "
                "will be silently replayed"
            )
        # Holding the OS lock proves no live managed turn is using the row. Any
        # orphaned history-mode turn was interrupted by an earlier process exit.
        with self.store._db() as db:
            for old in previous:
                if old.status == "running":
                    payload = old.to_dict()
                    payload.pop("events")
                    payload.update(status="interrupted", finished_at=_now())
                    db.execute("UPDATE turns SET data=? WHERE id=?", (json.dumps(payload), old.id))
            turn = {
                "id": uuid4().hex,
                "conversation_id": self.id,
                "message": message,
                "text": "",
                "status": "running",
                "created_at": _now(),
                "finished_at": None,
                "error": None,
            }
            db.execute(
                "INSERT INTO turns(id, conversation_id, data) VALUES (?, ?, ?)",
                (turn["id"], self.id, json.dumps(turn)),
            )
        return turn

    def _event(self, turn: dict[str, Any], event: AgentEvent) -> None:
        with self.store._db() as db:
            db.execute(
                "INSERT INTO events(turn_id, data) VALUES (?, ?)",
                (turn["id"], json.dumps(event.to_dict())),
            )
        if isinstance(event, TextDeltaEvent):
            turn["text"] += event.text

    def _finish(
        self,
        data: dict[str, Any],
        turn: dict[str, Any],
        status: str,
        error: str | None = None,
        session: Any = None,
    ) -> None:
        turn.update(status=status, finished_at=_now(), error=error)
        data["updated_at"] = turn["finished_at"]
        if session is not None:
            data["native_revision"] = turn["id"]
            if session.id is not None:
                data["native_session_id"] = session.id
        with self.store._db() as db:
            db.execute("UPDATE turns SET data=? WHERE id=?", (json.dumps(turn), turn["id"]))
            db.execute(
                "UPDATE conversations SET updated_at=?, data=? WHERE id=? AND application=?",
                (data["updated_at"], json.dumps(data), self.id, self.store.application),
            )

    def _request(self, data: dict[str, Any], message: str, controls: dict[str, Any]):
        forbidden = {
            "model",
            "connection",
            "history",
            "system_prompt",
            "allow_failover",
            "expect_auth_mode",
        } & controls.keys()
        if forbidden:
            raise ConversationError(
                "Conversation binding/history cannot be overridden on send: "
                + ", ".join(sorted(forbidden))
            )
        return {
            "connection": data["connection"],
            "model": data["model"],
            "message": message,
            "system_prompt": data["system_prompt"],
            "history": self._history(data) if data["mode"] == "history" else None,
            "expect_auth_mode": data["auth_mode"],
            "allow_failover": False,
            **controls,
        }

    def send(self, message: str, **controls: Any) -> Iterator[AgentEvent]:
        _text(message, "message")
        with self.store._lock(self.id):
            data = self.store._read(self.id)
            # Reject override keys before recording even an attempted turn.
            request = self._request(data, message, controls)
            tools, launch, turn_controls = (), {}, controls
            if data["mode"] == "native":
                if set(controls) - {
                    "timeout",
                    "reasoning",
                    "tools",
                    "stop_at_tokens",
                    "warn_at_tokens",
                }:
                    raise ConversationError("Unsupported native managed send control")
                tools = normalize_tools(controls.get("tools"))
                launch = {
                    key: controls[key]
                    for key in ("stop_at_tokens", "warn_at_tokens")
                    if key in controls
                }
                turn_controls = {
                    key: controls[key] for key in ("timeout", "reasoning") if key in controls
                }
            turn = self._begin(data, message)
            stream, session, terminal = None, None, None
            finished = False
            try:
                if data["mode"] == "native":
                    session = self._native(data, tools, launch)
                    stream = session.send(message, **turn_controls)
                else:
                    stream = self.store.bridge.chat(**request)
                for event in stream:
                    self._event(turn, event)
                    if isinstance(event, TerminalEvent):
                        terminal = event
                    else:
                        yield event
                status = terminal.status.value if terminal else "interrupted"
                self._finish(data, turn, status, session=session)
                finished = True
            except GeneratorExit:
                raise
            except BaseException as exc:
                if not finished:
                    status = (
                        "cancelled"
                        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                        else "error"
                    )
                    carried = getattr(exc, "terminal", None)
                    if isinstance(carried, TerminalEvent):
                        if terminal is None:
                            self._event(turn, carried)
                        status = carried.status.value
                    self._finish(data, turn, status, type(exc).__name__, session)
                    finished = True
                raise
            finally:
                try:
                    if stream is not None:
                        stream.close()
                finally:
                    if not finished:
                        self._finish(data, turn, "cancelled", session=session)
                    if session is not None:
                        self.store._release_native(
                            self.id, session, data, finished and turn["status"] == "ok"
                        )
        if terminal:
            yield terminal

    async def asend(self, message: str, **controls: Any) -> AsyncIterator[AgentEvent]:
        _text(message, "message")
        with self.store._lock(self.id):
            data = self.store._read(self.id)
            request = self._request(data, message, controls)
            tools, launch, turn_controls = (), {}, controls
            if data["mode"] == "native":
                if set(controls) - {
                    "timeout",
                    "reasoning",
                    "tools",
                    "stop_at_tokens",
                    "warn_at_tokens",
                }:
                    raise ConversationError("Unsupported native managed send control")
                tools = normalize_tools(controls.get("tools"))
                launch = {
                    key: controls[key]
                    for key in ("stop_at_tokens", "warn_at_tokens")
                    if key in controls
                }
                turn_controls = {
                    key: controls[key] for key in ("timeout", "reasoning") if key in controls
                }
            turn = self._begin(data, message)
            stream, session, terminal = None, None, None
            finished = False
            try:
                if data["mode"] == "native":
                    session = self._native(data, tools, launch)
                    stream = session.asend(message, **turn_controls)
                else:
                    stream = self.store.bridge.achat(**request)
                async for event in stream:
                    self._event(turn, event)
                    if isinstance(event, TerminalEvent):
                        terminal = event
                    else:
                        yield event
                status = terminal.status.value if terminal else "interrupted"
                self._finish(data, turn, status, session=session)
                finished = True
            except GeneratorExit:
                raise
            except BaseException as exc:
                if not finished:
                    status = (
                        "cancelled"
                        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                        else "error"
                    )
                    carried = getattr(exc, "terminal", None)
                    if isinstance(carried, TerminalEvent):
                        if terminal is None:
                            self._event(turn, carried)
                        status = carried.status.value
                    self._finish(data, turn, status, type(exc).__name__, session)
                    finished = True
                raise
            finally:
                try:
                    if stream is not None:
                        await stream.aclose()
                finally:
                    if not finished:
                        self._finish(data, turn, "cancelled", session=session)
                    if session is not None:
                        self.store._release_native(
                            self.id, session, data, finished and turn["status"] == "ok"
                        )
        if terminal:
            yield terminal
