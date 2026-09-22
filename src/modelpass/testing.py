"""Test support: scripted adapters that need no vendor package.

Shipped inside the package rather than kept in ``tests/`` so that downstream
tools built on modelpass can test their own event handling offline too. Nothing
here touches a network, a vendor SDK, or a credential -- which is the point:
every behavior of the core (guards, stamping, preflight, capability gating,
**and sessions**) must be reachable without either.

Two adapters, because modelpass has two seams:

* :class:`FakeAdapter` drives the stateless seam -- ``bridge.chat()`` and
  everything it carries. It deliberately does *not*
  implement ``open_session()``, so a caller that reaches for a session against
  it gets the same ``AdapterNotImplemented`` sentence a real runtime without
  session support would give.
* :class:`FakeSessionAdapter` subclasses it and adds the session seam --
  ``open_session()``, ``resume_session()``, ``list_sessions()`` -- so
  ``bridge.new_chat()`` / ``new_worker()`` / ``resume_chat()`` and the whole
  multi-turn lifecycle can be driven offline. It lived in ``tests/`` until a
  consumer had to push its session coverage into a live script that spends
  subscription allowance; a seam that can only be tested by paying for it is
  not a tested seam (third-consumer feedback, 2026-08-30).

:func:`fake_bridge` and :func:`fake_session_bridge` are the assembled
counterparts: a bridge, its store and its adapter, with the environment empty so
the machine a test runs on cannot change the answer.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar

from .adapters.base import Adapter, RunRequest, SessionRequest
from .capabilities import CapabilityRegistry
from .connections import Connection
from .errors import SessionNotFound, VendorRunFailed
from .preflight import AccountProfile, CacheEligibility, Receipt
from .runtimes import Runtime
from .schema import validate_instance
from .store import ConnectionStore
from .types import (
    CALLER_TOOL_SERVER,
    AgentEvent,
    AuthMode,
    Message,
    Role,
    SessionInfo,
    StructuredOutputEvent,
    TerminalEvent,
    TerminalStatus,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)

__all__ = [
    "FakeAdapter",
    "FakeSessionAdapter",
    "FakeSessionHandle",
    "StallingAdapter",
    "fake_bridge",
    "fake_session_bridge",
    "quota_exhausted",
    "structured",
    "tool_exchange",
    "usage",
    "vendor_failure",
]

Script = Sequence[AgentEvent] | Callable[[RunRequest], Iterable[AgentEvent]]


def usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached: int = 0,
    cache_write: int = 0,
) -> UsageEvent:
    """Shorthand for a usage event in a script."""
    return UsageEvent(
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached,
            cache_write_tokens=cache_write,
        )
    )


def quota_exhausted(
    reason: str = "subscription allowance exhausted",
    *,
    runtime: Runtime = Runtime.ANTHROPIC_SDK,
) -> TerminalEvent:
    """Script a run that ends because the plan's allowance ran out.

    Shipped here rather than left to each test to hand-roll because it is the
    trigger for the only path that can move a run onto metered billing, and a
    downstream tool that handles failover -- or deliberately does not -- needs to
    be able to exercise that without a spent subscription.

    The identity fields are placeholders, exactly as a real adapter emits them:
    the bridge overwrites connection, runtime and auth mode, and an adapter is
    not allowed to claim how a run was billed (adapter contract, rule 2).
    """
    return TerminalEvent(
        status=TerminalStatus.QUOTA_EXHAUSTED,
        connection="",
        runtime=runtime,
        auth_mode=AuthMode.SUBSCRIPTION,
        reason=reason,
    )


def vendor_failure(
    reason: str = "400: the model was rejected by the runtime",
) -> VendorRunFailed:
    """Script the vendor saying no partway through a run.

    Pass to ``FakeAdapter(script, error=vendor_failure())``. The run streams the
    script, then fails the way a real vendor failure fails: the stream ends with
    one terminal ``error`` event carrying the reason and the usage spent so far,
    rather than an exception escaping mid-iteration.

    Shipped here because it is the shape a downstream tool most needs to test
    and least wants to reproduce for real -- getting a live 400 out of a vendor
    means asking for a model that does not exist, on somebody's plan.
    """
    return VendorRunFailed(reason)


def tool_exchange(
    name: str,
    arguments: Mapping[str, Any] | None = None,
    result: str = "",
    *,
    call_id: str = "call_1",
    server: str = CALLER_TOOL_SERVER,
    is_error: bool = False,
) -> tuple[ToolCallEvent, ToolResultEvent]:
    """Script one runtime-executed tool round trip: a call and its result (D12).

    Emitted as a *pair* because that is what a caller actually sees -- the
    runtime executes the tool and reports both halves, and there is no point in
    the stream where the caller is expected to supply the result. A downstream
    tool testing its own event handling should be exercised against the shape it
    will really get, correlated ``id`` and all.
    """
    return (
        ToolCallEvent(
            name=name, arguments=dict(arguments or {}), id=call_id, server=server
        ),
        ToolResultEvent(id=call_id, name=name, content=result, is_error=is_error),
    )


def structured(
    data: Any,
    *,
    schema: Mapping[str, Any] | None = None,
    raw: str | None = None,
    valid: bool | None = None,
    schema_name: str = "",
) -> StructuredOutputEvent:
    """Script the schema-bound answer a run produced (D13).

    ``valid`` defaults to **the real structural check** against ``schema`` when
    one is given, so a test that scripts a deliberately wrong answer gets
    ``valid=False`` without having to say so -- and a test that scripts a right
    one cannot accidentally assert a passing flag over a failing payload. Pass
    ``valid=`` explicitly to force either.

    ``raw`` defaults to ``data`` re-encoded, which is what both real runtimes
    hand over alongside the parsed object.
    """
    text = raw if raw is not None else json.dumps(data, default=str)
    problems: tuple[str, ...] = ()
    if schema is not None:
        checked, problems = validate_instance(data, schema)
        valid = checked if valid is None else valid
    return StructuredOutputEvent(
        data=data,
        raw=text,
        valid=bool(valid),
        schema_name=schema_name,
        problems=problems if not valid else (),
    )


def _refuse_unserved_runtimes(
    connections: list[Connection], served: Runtime, allowed: bool
) -> None:
    """Refuse a connection this fake does not serve, instead of dialing out.

    A fake bridge injects its adapter for **one** runtime. A connection naming
    any other runtime is not unserved -- :meth:`Bridge.adapter_for` lazily loads
    the *real* adapter for it, and if that vendor's extra happens to be
    installed the "offline" test reaches the network. It does so quietly: on a
    bare CI runner the real adapter refuses at SDK import and the test passes,
    while on a developer machine with the extras installed the same test opens a
    socket. The failure is invisible exactly where it matters.

    That is the hazard ``env={}`` already exists to close, one layer along:
    there the machine's credentials leak in, here the machine's *installed
    packages* decide whether a test is offline.

    ``allow_real_adapters=True`` is the opt-out, for the tests that mean it --
    a failover whose target leg must be refused by the real adapter's own
    capability gate has to reach that adapter to be refused by it.
    """
    if allowed:
        return
    stray = {c.name: c.runtime for c in connections if c.runtime is not served}
    if not stray:
        return
    listed = ", ".join(f"{n!r} ({r.value})" for n, r in sorted(stray.items()))
    raise ValueError(
        f"fake_bridge serves {served.value!r} only, but these connections name "
        f"another runtime: {listed}. Left alone, each would resolve the real "
        "adapter for its runtime and reach the vendor if that extra is "
        "installed. Pass runtime= (or an adapter= built for it) to serve that "
        "runtime instead, or allow_real_adapters=True if a real adapter is the "
        "point of the test"
    )


def fake_bridge(
    *,
    connections: Iterable[Connection] = (),
    script: Script = (),
    home: Any = None,
    adapter: FakeAdapter | None = None,
    runtime: Runtime = Runtime.ANTHROPIC_SDK,
    env: Mapping[str, str] | None = None,
    registry: CapabilityRegistry | None = None,
    allow_real_adapters: bool = False,
) -> tuple[Any, ConnectionStore, FakeAdapter]:
    """A bridge, its store and its scripted adapter -- the six lines everyone writes.

    Returns ``(bridge, store, adapter)``. Every downstream consumer that tests
    its modelpass integration was assembling the same store/adapter/bridge triple
    by hand, and getting one detail wrong -- usually ``env={}``, without which a
    test picks up the developer's own ambient credentials and stops testing what
    it thinks it tests (second-consumer feedback, 2026-08-17)::

        bridge, store, adapter = fake_bridge(
            connections=[my_connection],
            script=[TextDeltaEvent("hi"), usage(input_tokens=10)],
            home=tmp_path / "modelpass",
        )

    ``home`` is where the store lives; pass a pytest ``tmp_path`` so nothing
    touches ``~/.modelpass`` -- including the run log, which follows the store.
    Leaving it unset uses the real home, which is almost never what a test
    wants, so it is worth passing even when it feels like boilerplate.

    ``env`` defaults to **empty**, not to ``os.environ``: the point of a fake
    bridge is that the machine it runs on cannot change the answer.
    """
    from .bridge import Bridge  # local: bridge imports this module's siblings

    store = ConnectionStore(home) if home is not None else ConnectionStore()
    connections = list(connections)
    for connection in connections:
        store.add(connection, overwrite=True)
    fake = adapter if adapter is not None else FakeAdapter(script, runtime=runtime)
    _refuse_unserved_runtimes(connections, fake.runtime, allow_real_adapters)
    bridge = Bridge(
        store=store,
        registry=registry or CapabilityRegistry(),
        adapters={fake.runtime: fake},
        env=dict(env or {}),
    )
    return bridge, store, fake


def fake_session_bridge(
    *,
    connections: Iterable[Connection] = (),
    script: Script = (),
    home: Any = None,
    adapter: FakeSessionAdapter | None = None,
    runtime: Runtime = Runtime.ANTHROPIC_SDK,
    env: Mapping[str, str] | None = None,
    registry: CapabilityRegistry | None = None,
    allow_real_adapters: bool = False,
) -> tuple[Any, ConnectionStore, FakeSessionAdapter]:
    """:func:`fake_bridge`, but the adapter can open sessions as well as run.

    Returns the same ``(bridge, store, adapter)`` triple with the same defaults,
    so the two are interchangeable at the call site -- the only difference is
    that ``bridge.new_chat()``, ``bridge.new_worker()``, ``bridge.resume_chat()``
    and ``bridge.list_sessions()`` work against what comes back::

        bridge, store, adapter = fake_session_bridge(
            connections=[my_connection],
            script=[TextDeltaEvent(text="hi"), usage(input_tokens=10)],
            home=tmp_path / "modelpass",
        )
        with bridge.new_chat(connection="my-connection") as session:
            list(session.send("first"))
            list(session.send("second"))
        assert adapter.opened == 1, "two turns, one session"

    The ``script`` is what **each turn** streams, not what the session streams
    once: a session is a sequence of turns and every one of them replays it.
    That is the difference worth knowing before writing an assertion, and it is
    why the interesting facts about a multi-turn test are usually read off the
    adapter (``adapter.opened``, ``adapter.handles[0].messages``) rather than off
    the events.

    ``home`` and ``env`` carry the same warnings as :func:`fake_bridge`: pass a
    ``tmp_path`` so neither the store nor the run log touches ``~/.modelpass``, and
    note that ``env`` defaults to **empty**, not ``os.environ``, so the machine
    the test runs on cannot change the answer.

    Pass ``adapter=FakeSessionAdapter(...)`` to reach the session-only knobs --
    ``listing=`` for what ``list_sessions()`` returns, ``missing=True`` to make
    every resume raise :class:`~modelpass.errors.SessionNotFound`. Assigning
    ``adapter.raise_at_end`` after the fact scripts a vendor failure partway
    through a turn, the session counterpart of ``FakeAdapter(error=...)``.
    """
    fake = adapter if adapter is not None else FakeSessionAdapter(script, runtime=runtime)
    bridge, store, _ = fake_bridge(
        connections=connections,
        home=home,
        adapter=fake,
        env=env,
        registry=registry,
        allow_real_adapters=allow_real_adapters,
    )
    return bridge, store, fake


#: What a ``FakeAdapter`` reports as the logged-in account unless a test says
#: otherwise. A module-level singleton rather than an inline default, because
#: :class:`~modelpass.preflight.AccountProfile` is frozen and one shared instance
#: is exactly what a default argument should be.
FAKE_ACCOUNT_PROFILE = AccountProfile(
    vendor="fake",
    source="FakeAdapter",
    logged_in=True,
    email="fake@example.com",
    subscription_type="Fake Plan",
)


class FakeAdapter(Adapter):
    """Emits a scripted event stream and records what it was asked to do.

    ``error`` is raised after the script is exhausted, which is how the two
    failure classes are exercised: :func:`vendor_failure` for "the vendor said
    no" (ends the stream with a terminal ``error``) and any other exception for
    "an adapter is broken" (surfaces as ``AdapterFailed``). The distinction is
    the adapter contract's rule 5, and a downstream tool should test against
    both because it will meet both.

    ``structured_output`` is what a schema-bound run answers with (D13). It is
    used **only when the request actually carries a schema**, so one
    ``FakeAdapter`` can serve both a plain call and a schema-bound one, and a
    consumer's schema path is exercised end to end -- gating, event order,
    validity flag -- with no vendor anywhere. Leaving it unset while the caller
    asks for a schema reproduces the other real behavior on purpose: the run
    ends ``status=error`` because nothing structured came back.
    """

    #: Sentinel: "no structured answer configured", distinct from ``None``,
    #: which is a caller deliberately scripting a run that produced nothing.
    _UNSET: ClassVar[object] = object()

    def __init__(
        self,
        script: Script = (),
        *,
        runtime: Runtime = Runtime.ANTHROPIC_SDK,
        detected_auth_mode: AuthMode | None = None,
        account: str | None = "fake-account",
        plan_name: str | None = "Fake Plan",
        account_profile: AccountProfile | None = FAKE_ACCOUNT_PROFILE,
        available: bool = True,
        ok: bool = True,
        problem: str | None = None,
        runtime_available: bool = True,
        error: BaseException | None = None,
        structured_output: Any = _UNSET,
        cache: CacheEligibility | None = None,
    ) -> None:
        self.runtime = runtime
        self.script = script
        self.detected_auth_mode = detected_auth_mode
        self.account = account
        self.plan_name = plan_name
        self.account_profile = account_profile
        self.available = available
        self.ok = ok
        self.problem = problem
        self.runtime_available = runtime_available
        self.error = error
        self.structured_output = structured_output
        self.cache = cache

        self.requests: list[RunRequest] = []
        self.preflights: list[RunRequest] = []
        self.cancelled = 0
        self.consumed: list[AgentEvent] = []
        self.closed = False

    def is_available(self) -> bool:  # type: ignore[override]
        return self.available

    def cache_eligibility(
        self, request: RunRequest | SessionRequest
    ) -> CacheEligibility | None:
        """Whatever ``cache=`` was constructed with; ``None`` by default.

        ``None`` matches the base adapter and is what a test that does not care
        about caching should get: no clause on the summary, no ``cache_note``,
        nothing new to assert around. Pass a
        :class:`~modelpass.preflight.CacheEligibility` to exercise the disclosure
        a real runtime produces -- rendering it, reacting to a sub-floor verdict
        -- without a subscription, a vendor package, or a CLI on the machine.
        """
        return self.cache

    def preflight(self, request: RunRequest) -> Receipt:
        self.preflights.append(request)
        return Receipt.from_plan(
            request.plan,
            detected_auth_mode=self.detected_auth_mode or request.connection.auth_mode,
            credential_source=request.connection.credential_ref.describe(),
            account=self.account,
            plan_name=self.plan_name,
            account_profile=self.account_profile,
            runtime_available=self.runtime_available,
            ok=self.ok,
            problem=self.problem,
        )

    def run(self, request: RunRequest) -> Iterator[AgentEvent]:
        self.requests.append(request)
        events = self.script(request) if callable(self.script) else self.script
        return self._generate(self._with_structured(request, events))

    def _with_structured(
        self, request: RunRequest, events: Iterable[AgentEvent]
    ) -> Iterable[AgentEvent]:
        """Append the schema-bound answer, mirroring what a real adapter does.

        Nothing happens unless the request carries a schema, and nothing happens
        if the script already produced a ``structured_output`` event -- a test
        that scripts the whole stream by hand stays in charge of it.
        """
        scripted = list(events)
        if not request.wants_schema:
            return scripted
        if any(isinstance(e, StructuredOutputEvent) for e in scripted):
            return scripted
        if self.structured_output is FakeAdapter._UNSET:
            # The real failure shape: asked for a schema, produced nothing.
            return [
                *scripted,
                TerminalEvent(
                    status=TerminalStatus.ERROR,
                    connection="",
                    runtime=self.runtime,
                    auth_mode=AuthMode.SUBSCRIPTION,
                    reason=(
                        "structured output requested but the runtime returned no "
                        "structured answer and no final text to read one from"
                    ),
                ),
            ]
        return [
            *scripted,
            structured(
                self.structured_output,
                schema=request.schema,
                schema_name=request.schema_name,
            ),
        ]

    def _generate(self, events: Iterable[AgentEvent]) -> Iterator[AgentEvent]:
        try:
            for event in events:
                self.consumed.append(event)
                yield event
            if self.error is not None:
                raise self.error
        finally:
            self.closed = True

    def cancel(self) -> None:
        self.cancelled += 1


class StallingAdapter(FakeAdapter):
    """A :class:`FakeAdapter` whose run stops answering until it is cancelled.

    The fake a timeout test needs, and the same shape the real hang has: the
    consuming thread is blocked inside ``next()`` on a stream that will never
    produce another event, so nothing a clock checked *between* events could do
    would ever bound it. Only ``cancel()`` -- from the watchdog thread -- gets
    the consumer moving again, which is exactly the contract
    :class:`~modelpass.errors.RunTimedOut` rests on.

    ``before`` is the script it plays first, so a test can choose between "stall
    at the very first event" (the ``first_token`` bound) and "stream some usage,
    then stall" (the ``total`` bound, with spend to account for).

    **It follows the same threading rules the real adapters do** (R7): the
    stream is driven from one thread, ``cancel()`` is safe to call from another,
    and ``cancel()`` never touches the generator -- it sets the event the
    generator is waiting on and returns.
    """

    def __init__(self, *args: Any, before: Script = (), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.before = before
        #: Set by :meth:`cancel`; the stalled run waits on it.
        self.released = threading.Event()
        #: How long the stalled run will wait before giving up on its own. Long
        #: enough that a test never reaches it, short enough that a broken test
        #: fails rather than hanging the suite.
        self.patience = 30.0
        self.stalled = threading.Event()

    def run(self, request: RunRequest) -> Iterator[AgentEvent]:
        self.requests.append(request)
        return self._stall(request)

    def _stall(self, request: RunRequest) -> Iterator[AgentEvent]:
        try:
            events = self.before(request) if callable(self.before) else self.before
            for event in events:
                self.consumed.append(event)
                yield event
            self.stalled.set()
            self.released.wait(self.patience)
            # Reached only when the watchdog released us. The adapter reports
            # what a cancelled runtime reports -- the bridge decides that a
            # cancel it asked for is a timeout, because the bridge is the one
            # that knows why it asked.
            yield TerminalEvent(
                status=TerminalStatus.CANCELLED,
                connection="",
                runtime=self.runtime,
                auth_mode=AuthMode.SUBSCRIPTION,
                reason="cancelled by caller",
            )
        finally:
            self.closed = True

    def cancel(self) -> None:
        self.cancelled += 1
        self.released.set()


class FakeSessionHandle:
    """One scripted vendor session: the handle a :class:`FakeSessionAdapter` hands out.

    It implements the :class:`~modelpass.adapters.base.SessionHandle` protocol and
    honours the one piece of timing the protocol cares about -- adapter contract
    rule 7, *a session is opened locally and created by running*. ``send`` is a
    generator, and the id is assigned at the **end** of it, after the script has
    been fully consumed. That is not a stylistic choice: it is what makes a turn
    abandoned mid-stream leave :attr:`id` at ``None``, the way a real
    ``codex exec`` killed before its first turn finishes leaves a rollout that
    ``resume`` rejects. A fake that assigned the id on entry would let a caller
    write a passing test for a resume that cannot work.

    ``persist=False`` sessions never get an id here either, for the same reason
    they never get one from a real runtime: there is nothing on disk to name.

    The recorded state is the point of the object. ``messages`` is every turn's
    text in order, ``closes`` counts ``close()`` calls (it must reach 1, not 2,
    for an idempotent close), and ``history()`` is the transcript the runtime
    would report -- alternating user/assistant, assembled by the *handle* rather
    than by modelpass, which is exactly the division D14 rests on.
    """

    def __init__(self, adapter: FakeSessionAdapter, request: SessionRequest) -> None:
        self.adapter = adapter
        self.request = request
        #: What each turn was told about effort, in order; ``None`` where a turn
        #: stated nothing. Parallel to :attr:`messages`.
        self.efforts: list[str | None] = []
        self.messages: list[str] = []
        self.closes = 0
        self._id: str | None = request.resume_id
        self._history: list[Message] = []

    @property
    def id(self) -> str | None:
        return self._id

    def send(self, message: str, *, effort: str | None = None) -> Iterator[AgentEvent]:
        """One scripted turn. ``efforts`` records what each turn was told.

        The recorded list is the point, the same way ``messages`` is: a test for
        a mid-session effort change has to be able to ask *what went on the
        wire*, and on the one runtime that has this the answer is a field on
        ``TurnStartParams`` that no fake can otherwise show. ``None`` is
        recorded for a turn that stated nothing, because "this turn said
        nothing" and "this turn said the standing level" are different turns.
        """
        self.efforts.append(effort)
        self.messages.append(message)
        self._history.append(Message(role=Role.USER, content=message))
        yield from self.adapter.script
        if self.adapter.raise_at_end is not None:
            raise self.adapter.raise_at_end
        self._history.append(Message(role=Role.ASSISTANT, content="ok"))
        if self._id is None and self.request.persist:
            self.adapter.opened += 1
            self._id = f"sess-{self.adapter.opened}"

    def history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    def close(self) -> None:
        self.closes += 1


class FakeSessionAdapter(FakeAdapter):
    """A :class:`FakeAdapter` that also knows how to open sessions.

    A subclass rather than a replacement, deliberately. Everything already
    written against ``FakeAdapter`` -- scripts, ``error=``, ``structured_output=``,
    the recorded ``requests`` and ``preflights`` -- keeps working, and the base
    class keeps its own value: a ``FakeAdapter`` still refuses ``open_session()``
    with the sentence a runtime without session support gives, which is the only
    way to test that refusal.

    What it adds is the session seam and the counters worth asserting on:

    * ``opened`` is how many vendor sessions were actually **created**, which is
      the number that answers "did turn 2 continue the conversation or start a
      new one" -- the question a multi-turn integration most needs pinned.
    * ``handles`` is every :class:`FakeSessionHandle` handed out, in order, so a
      test can read the turns that reached the runtime.
    * ``session_requests`` and ``listed`` are the
      :class:`~modelpass.adapters.base.SessionRequest` objects the bridge built,
      which is where the mapped policy shows up: ``kind``, ``native_tools``,
      ``appends_system_prompt``, ``project_folder``, ``resume_id``.

    ``listing`` is what ``list_sessions()`` returns; ``missing=True`` makes every
    resume raise :class:`~modelpass.errors.SessionNotFound`, which is how the
    "that id is gone" path is exercised without deleting anyone's real rollout.
    ``raise_at_end`` -- assigned after construction -- raises partway through
    every turn, the session counterpart of ``FakeAdapter(error=...)``; pass
    :func:`vendor_failure` to it for the shape a real vendor refusal has.
    """

    def __init__(
        self,
        *args: Any,
        listing: tuple[SessionInfo, ...] = (),
        missing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raise_at_end: BaseException | None = None
        self.listing = listing
        self.missing = missing
        self.opened = 0
        self.handles: list[FakeSessionHandle] = []
        self.session_requests: list[SessionRequest] = []
        self.listed: list[SessionRequest] = []

    def open_session(self, request: SessionRequest) -> FakeSessionHandle:
        self.session_requests.append(request)
        handle = FakeSessionHandle(self, request)
        self.handles.append(handle)
        return handle

    def resume_session(self, request: SessionRequest) -> FakeSessionHandle:
        if self.missing:
            raise SessionNotFound(
                request.connection.name, request.resume_id or "", "no rollout found"
            )
        return self.open_session(request)

    def list_sessions(self, request: SessionRequest) -> tuple[SessionInfo, ...]:
        self.listed.append(request)
        return self.listing
