"""Hidden vendor children on Windows; never change the host event-loop policy.

Authentication and transport are separate: hiding a subprocess does not change
which subscription/account pays, refresh credentials, or introduce API billing.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
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
