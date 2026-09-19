"""The thread-safety contract, stated and enforced (R7, ticket 1.12).

A RAG evaluation harness asked for this in code -- a source comment beside the
client it builds per worker
is a comment saying nobody has told us whether a client may be shared, so this
one runs a client per worker. That is a measurable cost paid for a missing
sentence. The sentence is now in the README and in :class:`modelpass.Bridge`'s
docstring, and these tests are what make it true rather than aspirational.

The contract, in three sentences:

1. A :class:`~modelpass.Bridge` is safe to share across threads for ``chat``,
   ``ask``, ``preflight`` and ``validate``; each call builds its own request,
   its own guard tracker and its own stream, and shares nothing mutable.
2. A :class:`~modelpass.sessions.Session` takes one caller at a time and raises
   :class:`~modelpass.errors.SessionBusy` on contention rather than
   interleaving two conversations or deadlocking.
3. Stores are guarded by their existing locks; adapters may be shared, and
   the fakes in :mod:`modelpass.testing` follow the same rules the real ones do.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from modelpass.connections import Connection, CredentialRef
from modelpass.errors import SessionBusy
from modelpass.runtimes import Runtime
from modelpass.testing import (
    FakeSessionAdapter,
    fake_bridge,
    fake_session_bridge,
    usage,
)
from modelpass.types import AuthMode, TerminalStatus, TextDeltaEvent

WORKERS = 12


@pytest.fixture
def connection() -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


def _per_request_script(request):
    """A script that answers with what *this* request asked, plus its own usage.

    The whole point of the cross-talk test: if two concurrent calls shared any
    state, one worker would read the other's text or the other's tokens, and the
    only way to notice is to make every answer distinguishable.
    """
    body = request.messages[-1].content
    n = int(str(body).rsplit("-", 1)[-1])
    return [
        TextDeltaEvent(text=f"answer-{n}"),
        usage(input_tokens=n, output_tokens=n * 2),
    ]


def test_one_bridge_serves_many_concurrent_asks_without_cross_talk(
    tmp_path, connection
):
    bridge, _, _ = fake_bridge(
        connections=[connection],
        script=_per_request_script,
        home=tmp_path / "modelpass",
    )

    def one(n: int):
        return n, bridge.ask("claude-sub", f"question-{n}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(one, range(1, WORKERS + 1)))

    assert len(results) == WORKERS
    for n, answer in results:
        # Its own text...
        assert answer.text == f"answer-{n}"
        # ...its own usage, never a neighbour's and never a running total...
        assert answer.usage.input_tokens == n
        assert answer.usage.output_tokens == n * 2
        # ...and its own receipt and terminal.
        assert answer.receipt.connection == "claude-sub"
        assert answer.status is TerminalStatus.OK


def test_concurrent_streams_do_not_interleave_into_each_other(tmp_path, connection):
    """The streaming form of the same guarantee, driven by hand.

    Each thread holds its own iterator and drains it; every event it sees must
    belong to its own run.
    """
    bridge, _, _ = fake_bridge(
        connections=[connection],
        script=_per_request_script,
        home=tmp_path / "modelpass",
    )
    start = threading.Barrier(WORKERS)
    seen: dict[int, list[str]] = {}
    lock = threading.Lock()

    def one(n: int) -> None:
        stream = bridge.chat(connection="claude-sub", message=f"question-{n}")
        start.wait(timeout=10)
        texts = [e.text for e in stream if isinstance(e, TextDeltaEvent)]
        with lock:
            seen[n] = texts

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(one, range(1, WORKERS + 1)))

    assert seen == {n: [f"answer-{n}"] for n in range(1, WORKERS + 1)}


def test_concurrent_preflights_share_one_bridge(tmp_path, connection):
    bridge, _, _ = fake_bridge(
        connections=[connection], home=tmp_path / "modelpass"
    )
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        receipts = list(
            pool.map(lambda _: bridge.preflight("claude-sub"), range(WORKERS))
        )
    assert all(r.connection == "claude-sub" for r in receipts)
    assert len({r.summary() for r in receipts}) == 1


def test_concurrent_validate_shares_one_bridge(tmp_path, connection):
    bridge, _, _ = fake_bridge(
        connections=[connection], home=tmp_path / "modelpass"
    )
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        reports = list(
            pool.map(lambda _: bridge.validate("claude-sub"), range(WORKERS))
        )
    assert all(report.ok for report in reports)


def test_the_run_log_keeps_every_concurrent_run(tmp_path, connection):
    """A shared bridge shares a run log, and the ledger is the thing that must
    not lose a line: a run nobody recorded is a run nobody can account for."""
    bridge, _, _ = fake_bridge(
        connections=[connection],
        script=_per_request_script,
        home=tmp_path / "modelpass",
    )
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(
            pool.map(
                lambda n: bridge.ask("claude-sub", f"question-{n}"),
                range(1, WORKERS + 1),
            )
        )
    records = bridge.run_log.read()
    assert len(records) == WORKERS
    assert {r["status"] for r in records} == {"ok"}


# --- a session is single-threaded, and says so ------------------------------------


class _BlockingSessionAdapter(FakeSessionAdapter):
    """A session adapter whose turn holds until the test lets it go.

    Needed so the second caller reliably arrives while the first turn is still
    running; without it the race is real but unrepeatable, which is not a test.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()

    def open_session(self, request):
        handle = super().open_session(request)
        original = handle.send

        def send(message: str):
            self.entered.set()
            self.release.wait(10)
            return original(message)

        handle.send = send  # type: ignore[method-assign]
        return handle


def test_two_threads_sending_into_one_session_raises_rather_than_interleaving(
    tmp_path, connection
):
    adapter = _BlockingSessionAdapter(
        script=[TextDeltaEvent(text="hello"), usage(input_tokens=3)]
    )
    bridge, _, _ = fake_session_bridge(
        connections=[connection], adapter=adapter, home=tmp_path / "modelpass"
    )
    session = bridge.new_chat(connection="claude-sub", persist=False)

    first_done = threading.Event()

    def first() -> None:
        list(session.send("turn one"))
        first_done.set()

    worker = threading.Thread(target=first, daemon=True)
    worker.start()
    assert adapter.entered.wait(10), "the first turn never started"

    # The second caller arrives mid-turn and is refused, not queued and not
    # deadlocked -- it raises immediately rather than waiting on the release.
    with pytest.raises(SessionBusy, match="one caller at a time"):
        list(session.send("turn two"))

    adapter.release.set()
    assert first_done.wait(10)

    # And the session is perfectly usable afterwards: a refusal is not damage.
    events = list(session.send("turn three"))
    assert events[-1].status is TerminalStatus.OK


def test_a_session_per_thread_is_the_supported_shape(tmp_path, connection):
    """The fix the refusal points at: sessions are cheap to open (adapter rule
    7 makes opening a handle local work), so one per caller costs nothing."""
    bridge, _, _ = fake_session_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="hello"), usage(input_tokens=3)],
        home=tmp_path / "modelpass",
    )

    def one(_: int) -> str:
        session = bridge.new_chat(connection="claude-sub", persist=False)
        return "".join(
            e.text for e in session.send("hi") if isinstance(e, TextDeltaEvent)
        )

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        answers = list(pool.map(one, range(WORKERS)))

    assert answers == ["hello"] * WORKERS


def test_an_abandoned_iterator_does_not_hold_the_session(tmp_path, connection):
    """The lock is taken on the first ``next()``, not by ``send()`` itself.

    A caller who builds an iterator and never drains it holds nothing -- the
    alternative leaks the lock for the life of the session, and the leak would
    look exactly like the contention this is meant to report.
    """
    bridge, _, _ = fake_session_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="hello"), usage(input_tokens=3)],
        home=tmp_path / "modelpass",
    )
    session = bridge.new_chat(connection="claude-sub", persist=False)
    session.send("never drained")  # built and dropped on the floor
    events = list(session.send("drained"))
    assert events[-1].status is TerminalStatus.OK


def test_a_closed_iterator_gives_the_session_back(tmp_path, connection):
    bridge, _, _ = fake_session_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="a"), TextDeltaEvent(text="b")],
        home=tmp_path / "modelpass",
    )
    session = bridge.new_chat(connection="claude-sub", persist=False)
    stream = session.send("one")
    next(stream)
    stream.close()
    # Abandoning a turn is cancellation (D10); it is not damage to the session.
    events = list(session.send("two"))
    assert events[-1].status is TerminalStatus.OK
