"""A cancelled Codex run leaves no process behind -- driven with real processes.

Every other Codex test injects a fake child, which is exactly why this defect
went unseen: a fake has no children of its own. The ``codex`` that resolves on
Windows is ``codex.cmd``, so what modelpass launches is ``cmd.exe`` and the turn
runs two levels below it (cmd -> node -> the native binary). ``Popen.kill()``
ends the one process it holds, the turn carries on underneath, and because the
grandchild inherited stdout the run never even sees its stream end.

So these tests build a real chain of the same shape -- a shim, which starts
Python, which starts Python -- through the adapters' **default** spawn
functions, cancel, and assert that the innermost process has stopped. The
innermost process writes a heartbeat file, and "stopped" is read off that file
rather than off a pid, which could be reused.

No vendor binary, no network and no credential is involved. Every test kills
whatever it started even when it fails, and every wait is bounded, so a
regression fails here instead of hanging the suite.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelpass import _background
from modelpass.adapters.base import RunRequest, SessionRequest
from modelpass.adapters.codex_appserver import AppServerClient
from modelpass.adapters.openai import OpenAIAdapter
from modelpass.connections import Connection, CredentialRef
from modelpass.preflight import plan_launch
from modelpass.runtimes import Runtime
from modelpass.types import AuthMode, Message, Role, SessionKind, TerminalStatus

# --- the chain -------------------------------------------------------------------

#: The innermost process: the one that would be running the turn. It answers
#: JSON-RPC requests (enough for an app-server handshake), announces a thread on
#: ``exec``, and then never stops on its own -- not on stdin EOF, not on its
#: parent dying -- beating into a file the whole time.
LEAF = r'''
import json, os, sys, threading, time
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "leaf.pid"), "w") as handle:
    handle.write(str(os.getpid()))
exec_mode = "exec" in sys.argv

def answer():
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if isinstance(message, dict) and "id" in message and "method" in message:
            sys.stdout.write(json.dumps({"id": message["id"], "result": {}}) + "\n")
            sys.stdout.flush()

threading.Thread(target=answer, daemon=True).start()
if exec_mode:
    sys.stdout.write(json.dumps({"type": "thread.started", "thread_id": "t-1"}) + "\n")
    sys.stdout.flush()
beat = 0
while True:
    beat += 1
    with open(os.path.join(here, "beats"), "a") as handle:
        handle.write(".")
    if exec_mode:
        try:
            sys.stdout.write(json.dumps({"type": "beat", "n": beat}) + "\n")
            sys.stdout.flush()
        except OSError:
            pass
    time.sleep(0.05)
'''

#: The middle process, standing where ``node`` stands: it starts the leaf with
#: its own stdin and stdout and waits for it.
MIDDLE = r'''
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "middle.pid"), "w") as handle:
    handle.write(str(os.getpid()))
sys.exit(subprocess.Popen([sys.executable, os.path.join(here, "leaf.py"), *sys.argv[1:]]).wait())
'''


class Chain:
    """A fake ``codex`` that is three processes deep, and the means to check on it."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "leaf.py").write_text(LEAF, encoding="utf-8")
        (folder / "middle.py").write_text(MIDDLE, encoding="utf-8")
        middle = folder / "middle.py"
        if os.name == "nt":
            shim = folder / "codex.cmd"
            shim.write_text(
                f'@echo off\r\n"{sys.executable}" "{middle}" %*\r\n', encoding="utf-8"
            )
        else:
            shim = folder / "codex"
            shim.write_text(
                f'#!/bin/sh\n"{sys.executable}" "{middle}" "$@"\n', encoding="utf-8"
            )
            shim.chmod(0o755)
        self.binary = str(shim)

    def beats(self) -> int:
        try:
            return (self.folder / "beats").stat().st_size
        except OSError:
            return 0

    def wait_until_beating(self, timeout: float = 15.0) -> None:
        """The leaf is up: both levels below the shim exist and the turn is 'running'."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.beats() > 0 and (self.folder / "middle.pid").exists():
                return
            time.sleep(0.02)
        raise AssertionError("the fake codex chain never started")

    def wait_until_beating_leaf(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.beats() > 0:
                return
            time.sleep(0.02)
        raise AssertionError("the leaf never started")

    def stopped(self, timeout: float = 6.0, quiet: float = 0.6) -> bool:
        """Whether the leaf stops beating within ``timeout``.

        Quiet for twelve of its own intervals is stopped; a leaf that is still
        alive cannot be quiet that long.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            before = self.beats()
            time.sleep(quiet)
            if self.beats() == before:
                return True
        return False

    def pids(self) -> list[int]:
        found = []
        for name in ("leaf.pid", "middle.pid"):
            try:
                found.append(int((self.folder / name).read_text().strip()))
            except (OSError, ValueError):
                continue
        return found

    def destroy(self) -> None:
        """Kill whatever is left, by pid. The safety net, never the assertion."""
        for pid in self.pids():
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True,
                        timeout=10,
                    )
                else:
                    os.kill(pid, signal.SIGKILL)
            except (OSError, subprocess.SubprocessError):
                continue


@pytest.fixture
def chain(tmp_path):
    built = Chain(tmp_path / "fake-codex")
    yield built
    built.destroy()


@pytest.fixture(autouse=True)
def _hermetic_codex_home(tmp_path_factory, monkeypatch):
    """No test here may read this machine's real ``~/.codex``."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("CODEX_HOME", raising=False)


def codex_connection() -> Connection:
    return Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


def child_env() -> dict[str, str]:
    """The machine's own environment, scrubbed by the plan: a real child needs one."""
    return dict(plan_launch(codex_connection(), dict(os.environ)).env)


def finish(stream, timeout: float = 20.0) -> list:
    """Drain ``stream`` on a thread, so a run that never ends fails instead of hanging."""
    events: list = []
    failure: list[BaseException] = []

    def drain() -> None:
        try:
            events.extend(stream)
        except BaseException as exc:  # reported below, on the test's thread
            failure.append(exc)

    worker = threading.Thread(target=drain, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), (
        f"the stream was still producing {timeout}s after the cancel: the "
        "process holding its stdout was never killed"
    )
    if failure:
        raise failure[0]
    return events


# --- exec: the stateless run -------------------------------------------------------


def test_cancelling_an_exec_run_stops_every_process_under_the_shim(chain, tmp_path):
    connection = codex_connection()
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="Reply with exactly OK"),),
        plan=plan_launch(connection, dict(os.environ)),
        options={"transport": "exec", "cwd": str(tmp_path)},
    )
    adapter = OpenAIAdapter(codex_bin=chain.binary)  # the default spawn, on purpose

    stream = adapter.run(request)
    assert next(stream).type == "vendor_event"  # thread.started, from two levels down
    chain.wait_until_beating()

    stream.cancel()
    events = finish(stream)

    assert events[-1].status is TerminalStatus.CANCELLED
    assert chain.stopped(), "the turn kept running after the cancel"


# --- exec: a session turn ----------------------------------------------------------


def test_cancelling_an_exec_session_turn_stops_every_process_under_the_shim(
    chain, tmp_path
):
    connection = codex_connection()
    request = SessionRequest(
        connection=connection,
        plan=plan_launch(connection, dict(os.environ)),
        kind=SessionKind.WORKER,
        project_folder=str(tmp_path),
        options={"transport": "exec"},
    )
    adapter = OpenAIAdapter(codex_bin=chain.binary)
    handle = adapter.open_session(request)

    stream = handle.send("hello")
    try:
        assert next(stream).type == "vendor_event"
        chain.wait_until_beating()

        stream.cancel()
        finish(stream)

        assert chain.stopped(), "the session's turn kept running after the cancel"
    finally:
        handle.close()


# --- app-server --------------------------------------------------------------------


def test_closing_an_app_server_client_stops_a_child_that_ignores_stdin_eof(
    chain, tmp_path
):
    """The close has a floor: a server that will not leave is killed, all of it."""
    client = AppServerClient(binary=chain.binary, env=child_env(), cwd=str(tmp_path))
    try:
        client.start()  # the handshake is answered by the leaf
        chain.wait_until_beating()
        started = time.monotonic()
        client.close()
        assert time.monotonic() - started < 15
        assert chain.stopped(), "the app-server kept running after close()"
    finally:
        client.close()


# --- the short queries -------------------------------------------------------------


def test_a_query_that_times_out_returns_and_leaves_nothing_running(chain):
    """``codex login status`` and friends: the timeout has to actually bound the call.

    ``subprocess.run(timeout=...)`` kills the shim and then reads the pipes the
    leaf still holds, so it returns only when the leaf does.
    """
    outcome: list = []

    def query() -> None:
        try:
            _background.run_tree([chain.binary, "login", "status"], env=child_env(), timeout=2)
        except subprocess.TimeoutExpired as exc:
            outcome.append(exc)

    worker = threading.Thread(target=query, daemon=True)
    worker.start()
    worker.join(20)

    assert not worker.is_alive(), "the query outlived its own timeout"
    assert outcome, "a query that timed out must say so"
    assert chain.stopped()


def test_a_query_that_answers_is_returned_whole(tmp_path):
    script = tmp_path / "answer.py"
    script.write_text("import sys; print('ok'); sys.stderr.write('note')", encoding="utf-8")
    result = _background.run_tree(
        [sys.executable, str(script)], env=dict(os.environ), timeout=30
    )
    assert (result.returncode, result.stdout.strip(), result.stderr) == (0, "ok", "note")


# --- the Windows job itself --------------------------------------------------------

windows_only = pytest.mark.skipif(os.name != "nt", reason="Job Objects are Windows")

#: Starts the leaf detached and exits at once: a finished run that left
#: something running on purpose.
LEAVER = r"""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
subprocess.Popen(
    [sys.executable, os.path.join(here, "leaf.py")],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
"""

#: A stand-in for modelpass's own process: it launches the chain the way the
#: adapters do and then dies without running a single ``finally``.
DYING_PARENT = r"""
import os, subprocess, sys, time
from modelpass import _background
proc = _background.spawn_tree(
    [sys.argv[1], "app-server"],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
deadline = time.monotonic() + 15
while not os.path.exists(sys.argv[2]) and time.monotonic() < deadline:
    time.sleep(0.02)
os._exit(0 if os.path.exists(sys.argv[2]) else 3)
"""


@windows_only
def test_the_job_reaches_a_descendant_whose_parents_are_already_dead(chain):
    """Parent links are gone once the shim dies; the job still names the leaf."""
    proc = _background.spawn_tree(
        [chain.binary, "app-server"],
        env=child_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert isinstance(proc._modelpass_tree, _background._WindowsJob)
        chain.wait_until_beating()
        proc.kill()  # the shim alone, which is the defect in one line
        proc.wait(timeout=10)
        assert not chain.stopped(timeout=1.0, quiet=0.5), (
            "the leaf died with the shim, so this chain no longer shows the defect"
        )

        _background.kill_tree(proc)

        assert chain.stopped()
    finally:
        _background.kill_tree(proc)


@windows_only
def test_the_tree_dies_with_the_process_that_launched_it(chain, tmp_path):
    """Kill-on-close: no ``finally`` ran, and the turn still stopped."""
    script = tmp_path / "dying_parent.py"
    script.write_text(DYING_PARENT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}

    parent = subprocess.run(
        [sys.executable, str(script), chain.binary, str(chain.folder / "beats")],
        env=env,
        capture_output=True,
        timeout=60,
    )

    assert parent.returncode == 0, parent.stderr
    assert chain.stopped()


@windows_only
def test_a_child_that_exited_on_its_own_is_let_go_without_killing_what_it_left(chain):
    """A finished run is not a cancelled one: what it started on purpose survives."""
    leaver = chain.folder / "leaver.py"
    leaver.write_text(LEAVER, encoding="utf-8")
    proc = _background.spawn_tree([sys.executable, str(leaver)], env=child_env())
    try:
        assert proc.wait(timeout=30) == 0
        chain.wait_until_beating_leaf()

        OpenAIAdapter._terminate(proc)  # every run's finally

        assert not chain.stopped(timeout=1.5, quiet=0.6)
    finally:
        _background.kill_tree(proc)


# --- what cannot be run here -------------------------------------------------------
#
# The other platform's branch, held with fakes: this suite runs on one operating
# system at a time, and the half it is not running on would otherwise have no
# test at all. These pin the calls made, not what the kernel does with them.


class _Child:
    def __init__(self, returncode=None) -> None:
        self.pid = 4242
        self.returncode = returncode
        self.calls: list[str] = []

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.calls.append("kill")

    def terminate(self) -> None:
        self.calls.append("terminate")


def _posix(monkeypatch) -> list[tuple[int, int]]:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        _background,
        "os",
        SimpleNamespace(name="posix", killpg=lambda pid, sig: sent.append((pid, sig))),
    )
    monkeypatch.setattr(_background, "signal", SimpleNamespace(SIGKILL=9, SIGTERM=15))
    return sent


def test_off_windows_the_child_leads_a_session_and_the_group_is_signalled(monkeypatch):
    sent = _posix(monkeypatch)
    assert _background.tree_options() == {"start_new_session": True}
    child = _background.adopt_tree(_Child())

    _background.terminate_tree(child)
    _background.kill_tree(child)

    assert sent == [(4242, 15), (4242, 9)]


def test_off_windows_a_reaped_child_is_never_signalled_by_group(monkeypatch):
    """Its pid may name somebody else's group by now."""
    sent = _posix(monkeypatch)
    child = _background.adopt_tree(_Child(returncode=0))

    _background.terminate_tree(child)
    _background.kill_tree(child)

    assert sent == [] and child.calls == []


def test_a_process_nobody_adopted_is_killed_the_way_it_always_was():
    """A caller's own spawn, or a test's fake: its ``kill()`` is all there is."""
    child = _Child()
    _background.terminate_tree(child)
    _background.kill_tree(child)
    _background.release_tree(child)
    assert child.calls == ["terminate", "kill"]
