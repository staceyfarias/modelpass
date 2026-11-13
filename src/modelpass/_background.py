"""Hidden vendor children on Windows; never change the host event-loop policy.

Authentication and transport are separate: hiding a subprocess does not change
which subscription/account pays, refresh credentials, or introduce API billing.

Also where a vendor child's whole process tree is held and killed
(:func:`spawn_tree`, :func:`kill_tree`): a CLI behind a shim is several
processes, and a cancel has to stop all of them.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import signal
import subprocess
import threading
from typing import Any


def process_options() -> dict[str, Any]:
    """Suppress a console for noninteractive vendor runs and identity probes."""
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def runtime_loop() -> asyncio.AbstractEventLoop:
    """An SDK-owned loop that hides only its own Windows subprocesses.

    The Claude SDK starts its child via AnyIO/asyncio and exposes no creationflags
    option. Apply the flag at this private loop's subprocess boundary, preserving
    caller flags and leaving the application's loops and global policy untouched.
    """
    loop = asyncio.new_event_loop()
    if os.name == "nt":
        original = loop.subprocess_exec

        async def hidden_exec(*args: Any, **kwargs: Any) -> Any:
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
            return await original(*args, **kwargs)

        loop.subprocess_exec = hidden_exec
    return loop


# --- whole process trees -----------------------------------------------------------
#
# A vendor CLI is rarely one process. On Windows the ``codex`` that resolves is
# ``codex.cmd``, so the child modelpass holds is ``cmd.exe`` and the turn runs
# two levels below it (cmd -> node -> the native binary). ``Popen.kill()`` ends
# only the process it holds: reproduced 2026-11-13 with a ``.cmd`` -> python ->
# python chain, whose grandchild kept writing for five seconds after ``kill()``.
# So a cancel or a timeout that must stop a run has to reach the whole tree.

#: The attribute a tree handle rides on, set by :func:`adopt_tree`. A process
#: without it (a test's fake, an embedder's own spawn) is killed the way it
#: always was, through its own ``kill()`` and ``terminate()``.
_TREE_ATTR = "_modelpass_tree"

_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_TASKKILL_TIMEOUT = 10.0


class _WindowsJob:
    """A kill-on-close Job Object holding one child and everything it starts.

    Kill-on-close is what makes the tree die even when this process does not get
    to run a ``finally``: the operating system closes the handle with the
    process, and that ends the job. Breakaway stays allowed, so a descendant
    that asks to leave the job still launches exactly as it did before there was
    one.
    """

    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle
        self._lock = threading.Lock()

    @classmethod
    def holding(cls, proc: Any) -> _WindowsJob | None:
        """A new job with ``proc`` assigned to it, or ``None`` if that failed."""
        try:
            kernel32 = _kernel32()
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return None
            job = cls(handle)
            if not job._set_limits(
                _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | _JOB_OBJECT_LIMIT_BREAKAWAY_OK
            ) or not kernel32.AssignProcessToJobObject(handle, int(proc._handle)):
                job.release()
                return None
            return job
        except (OSError, AttributeError, TypeError, ValueError):
            return None

    def _set_limits(self, flags: int) -> bool:
        info = _job_limit_information()
        info.BasicLimitInformation.LimitFlags = flags
        return bool(
            _kernel32().SetInformationJobObject(
                self._handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
        )

    def kill(self) -> None:
        """End every process still in the job. Idempotent."""
        with self._lock:
            handle, self._handle = self._handle, None
            if handle:
                _kernel32().TerminateJobObject(handle, 1)
                _kernel32().CloseHandle(handle)

    def release(self) -> None:
        """Let go without killing: the child exited on its own. Idempotent.

        What a finished run left behind on purpose -- a server its shell tool
        started, say -- survived before there was a job, and still does.
        """
        with self._lock:
            if self._handle:
                self._set_limits(_JOB_OBJECT_LIMIT_BREAKAWAY_OK)
                _kernel32().CloseHandle(self._handle)
            self._handle = None

    def __del__(self) -> None:
        # Dropped without kill() or release(): nobody is watching this child
        # any more, so closing the handle ends the job, which is the point.
        handle, self._handle = self._handle, None
        if handle:
            try:
                _kernel32().CloseHandle(handle)
            except Exception:  # interpreter shutdown; the OS closes it anyway
                pass


_KERNEL32: Any = None


def _kernel32() -> Any:
    """``kernel32`` with the job functions typed. Handles are pointer-sized."""
    global _KERNEL32
    if _KERNEL32 is None:
        from ctypes import wintypes

        dll = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        dll.CreateJobObjectW.restype = wintypes.HANDLE
        dll.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        dll.SetInformationJobObject.restype = wintypes.BOOL
        dll.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        dll.AssignProcessToJobObject.restype = wintypes.BOOL
        dll.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        dll.TerminateJobObject.restype = wintypes.BOOL
        dll.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        dll.CloseHandle.restype = wintypes.BOOL
        dll.CloseHandle.argtypes = [wintypes.HANDLE]
        _KERNEL32 = dll
    return _KERNEL32


def _job_limit_information() -> Any:
    """A zeroed ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``."""
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return ExtendedLimitInformation()


def tree_options() -> dict[str, Any]:
    """:func:`process_options`, plus what lets :func:`kill_tree` reach a whole tree.

    Off Windows that is a session of the child's own, so its process group is
    exactly the tree and ``killpg`` cannot reach anything else. On Windows the
    tree is held by a Job Object instead, which :func:`adopt_tree` creates once
    the child exists.
    """
    options = process_options()
    if os.name != "nt":
        options["start_new_session"] = True
    return options


def adopt_tree(proc: Any) -> Any:
    """Take hold of the tree under a child just launched with :func:`tree_options`.

    Returns ``proc``. On Windows the child is assigned to a kill-on-close Job
    Object; where that cannot be done the kill falls back to ``taskkill /T``.
    """
    if os.name == "nt":
        proc._modelpass_tree = _WindowsJob.holding(proc) or "taskkill"
    else:
        proc._modelpass_tree = "group"
    return proc


def spawn_tree(argv: Any, **popen_options: Any) -> Any:
    """``subprocess.Popen`` for a vendor child whose whole tree can be killed."""
    return adopt_tree(subprocess.Popen(argv, **tree_options(), **popen_options))


def _taskkill(pid: int) -> None:
    """``taskkill /T /F``: the tree as its parent links describe it right now."""
    root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows"
    try:
        subprocess.run(
            [os.path.join(root, "System32", "taskkill.exe"), "/T", "/F", "/PID", str(pid)],
            **process_options(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_TASKKILL_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def kill_tree(proc: Any) -> None:
    """Kill a child and every process under it. Never raises; safe to repeat.

    **Windows**: ``taskkill /T`` while the child is still alive, then the job.
    Two steps because each covers what the other cannot. The job holds every
    descendant born after :func:`adopt_tree` ran, including ones whose parent
    has already exited; ``taskkill`` follows parent links, so it also finds a
    descendant born in the instant between launch and adoption.

    **Elsewhere**: ``SIGKILL`` to the child's process group, sent only while the
    child is unreaped, because that is when its pid is certain to still name the
    group.

    A process this module did not adopt gets its own ``kill()``.
    """
    if proc is None:
        return
    tree = getattr(proc, _TREE_ATTR, None)
    try:
        alive = proc.poll() is None
    except OSError:
        alive = False
    if isinstance(tree, _WindowsJob) or tree == "taskkill":
        if alive:
            _taskkill(proc.pid)
        if isinstance(tree, _WindowsJob):
            tree.kill()
    elif tree == "group":
        if alive:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        return
    if alive:
        try:
            proc.kill()
        except OSError:
            pass


def terminate_tree(proc: Any) -> bool:
    """Ask a child to stop: ``SIGTERM`` to its group, or its own ``terminate()``.

    Returns whether anything was sent. Windows has no polite signal --
    ``terminate()`` there is the same hard kill of one process that ``kill()``
    is -- so an adopted Windows child is sent nothing and left for
    :func:`kill_tree`: the polite step is whatever the caller did first (closing
    the child's stdin), and ``False`` tells the caller that is all there was.
    """
    if proc is None:
        return False
    tree = getattr(proc, _TREE_ATTR, None)
    try:
        if tree == "group":
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            return True
        if tree is None:
            proc.terminate()
            return True
    except OSError:
        return True
    return False


def release_tree(proc: Any) -> None:
    """The child exited on its own: stop holding its tree, kill nothing."""
    tree = getattr(proc, _TREE_ATTR, None) if proc is not None else None
    if isinstance(tree, _WindowsJob):
        tree.release()


def run_tree(
    argv: list[str], *, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run`` for a short vendor query, with a timeout that holds.

    ``subprocess.run(timeout=...)`` kills the one process it launched and then
    reads its pipes to the end. Behind a ``.cmd`` shim the process holding those
    pipes is a grandchild the kill never reached, so the read -- and the caller
    -- waits for as long as it runs. Here the whole tree is killed first.
    """
    proc = spawn_tree(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except BaseException:
        kill_tree(proc)
        try:
            proc.communicate(timeout=_TASKKILL_TIMEOUT)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        raise
    release_tree(proc)
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
