"""Adapter resolution with lazy imports.

Core installs with zero vendor dependencies (D8). Importing ``modelpass`` must
therefore never import ``claude_agent_sdk`` or ``codex``; the vendor package is
touched only when a run over that runtime actually starts, and its absence is a
clear :class:`RuntimeNotAvailable` naming the extra to install.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from ..errors import RuntimeNotAvailable
from ..runtimes import Runtime
from .base import Adapter, RunRequest

__all__ = ["ADAPTER_REGISTRY", "Adapter", "AdapterEntry", "RunRequest", "load_adapter"]


@dataclass(frozen=True, slots=True)
class AdapterEntry:
    """Where an adapter lives and what it needs installed."""

    module: str
    attribute: str
    extra: str
    package: str


ADAPTER_REGISTRY: dict[Runtime, AdapterEntry] = {
    Runtime.ANTHROPIC_SDK: AdapterEntry(
        module="modelpass.adapters.anthropic",
        attribute="AnthropicAdapter",
        extra="anthropic",
        package="claude-agent-sdk",
    ),
    Runtime.OPENAI_SDK: AdapterEntry(
        module="modelpass.adapters.openai",
        attribute="OpenAIAdapter",
        extra="openai",
        package="openai-codex",
    ),
    # The first API-key runtime (ticket 1.6). Same discovery, same lazy import,
    # same "the extra is missing" message -- the only thing new is that the
    # vendor package here is an HTTP client rather than a CLI wrapper.
    Runtime.ANTHROPIC_API: AdapterEntry(
        module="modelpass.adapters.anthropic_api",
        attribute="AnthropicAPIAdapter",
        extra="anthropic-api",
        package="anthropic",
    ),
    # The second (ticket 1.9). The extra is ``openai-api`` and the package is
    # ``openai`` -- deliberately not the ``openai`` *extra*, which is already
    # ``openai-codex``, the CLI wrapper the agent runtime drives. Two OpenAI
    # runtimes, two different packages, and the names have to say which.
    Runtime.OPENAI_API: AdapterEntry(
        module="modelpass.adapters.openai_api",
        attribute="OpenAIAPIAdapter",
        extra="openai-api",
        package="openai",
    ),
    # The third (ticket 1.10), and it shares the second's extra on purpose: the
    # package is the same ``openai`` HTTP client, pointed at somebody else's
    # endpoint. A separate extra would install the same wheel under a second
    # name and would say, falsely, that this runtime has a dependency of its own.
    # The fourth and last (ticket 1.11). ``google-genai`` is the Gemini API
    # client; the extra is ``google-api`` and there is deliberately no ``google``
    # extra beside it, because the two Google *subscription* runtimes ship no
    # adapter at all -- see EXPERIMENTAL_RUNTIMES for why that asymmetry is the
    # decision rather than an oversight.
    Runtime.GOOGLE_API: AdapterEntry(
        module="modelpass.adapters.google_api",
        attribute="GoogleAPIAdapter",
        extra="google-api",
        package="google-genai",
    ),
    Runtime.OPENAI_COMPATIBLE: AdapterEntry(
        module="modelpass.adapters.openai_compatible",
        attribute="OpenAICompatibleAdapter",
        extra="openai-api",
        package="openai",
    ),
}


def load_adapter(runtime: Runtime) -> Adapter:
    """Instantiate the adapter for a runtime, importing it on demand."""
    entry = ADAPTER_REGISTRY.get(runtime)
    if entry is None:
        raise RuntimeNotAvailable(str(runtime), "unavailable", "an adapter")
    try:
        module = importlib.import_module(entry.module)
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeNotAvailable(str(runtime), entry.extra, entry.package) from exc
    adapter_cls: type[Adapter] = getattr(module, entry.attribute)
    if not adapter_cls.is_available():
        raise RuntimeNotAvailable(str(runtime), entry.extra, entry.package)
    return adapter_cls()
