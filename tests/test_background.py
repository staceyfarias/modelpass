"""Vendor children are invisible without changing host-wide launch policy."""

import asyncio
from types import SimpleNamespace

from modelpass import _background


def test_non_windows_process_options_are_unchanged(monkeypatch):
    monkeypatch.setattr(_background, "os", SimpleNamespace(name="posix"))
    assert _background.process_options() == {}


def test_windows_flags_apply_only_to_the_owned_loop(monkeypatch):
    observed = []

    async def original(*args, **kwargs):
        observed.append(kwargs)

    loop = SimpleNamespace(subprocess_exec=original)
    monkeypatch.setattr(_background, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(_background.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(_background.asyncio, "new_event_loop", lambda: loop)
    assert _background.runtime_loop() is loop
    asyncio.run(loop.subprocess_exec("protocol", "child", creationflags=4))
    assert observed == [{"creationflags": 0x08000004}]
    assert _background.process_options() == {"creationflags": 0x08000000}


def test_owned_loop_keeps_child_pipes_working_without_vendor_or_tokens():
    import sys

    async def run():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "print('background pipe works')",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await process.communicate()
        assert process.returncode == 0
        assert out.strip() == b"background pipe works"
        assert err == b""

    with asyncio.Runner(loop_factory=_background.runtime_loop) as runner:
        runner.run(run())
