"""Runtime identities.

Capability identity is the *runtime*, not the vendor (D6). ``anthropic-sdk`` and
``openai-sdk`` are genuinely different engines with different capabilities, auth
modes and session stores; Google ships two runtimes that cannot even share a
billing mode, so it gets two identities.

The same rule is what separates the **agent runtimes** -- a vendor CLI or agent
SDK driven as a child process, holding its own sessions and its own toolbelt --
from the **API runtimes** added on 2026-09-13, which are an HTTP client this
process constructs with an explicit key. They share a vendor and share nothing
else: no subprocess, no login store, no vendor-held history. Nothing in the
library asks "which vendor is this"; everything asks "which runtime", and
:data:`API_RUNTIMES` is how the two families are told apart where the answer
genuinely differs (the preflight launches nothing, so it plans no environment).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "AGENT_RUNTIMES",
    "API_RUNTIMES",
    "EXPERIMENTAL_RUNTIMES",
    "VENDOR_OF",
    "Runtime",
    "parse_runtime",
]


class Runtime(StrEnum):
    """The execution engines modelpass can drive."""

    ANTHROPIC_SDK = "anthropic-sdk"
    OPENAI_SDK = "openai-sdk"
    GOOGLE_CLI = "google-cli"
    GOOGLE_SDK = "google-sdk"
    # The API-key runtimes (2026-09-13). Plumbed here; their adapters land in
    # tickets 1.6 and 1.9-1.11, and until then every capability that is not a
    # checked absence reads ``unverified``.
    ANTHROPIC_API = "anthropic-api"
    OPENAI_API = "openai-api"
    GOOGLE_API = "google-api"
    OPENAI_COMPATIBLE = "openai-compatible"


#: The vendor behind each runtime, and the **single source** of that answer.
#: :attr:`~modelpass.connections.Connection.vendor` reads this map and nothing
#: else.
#:
#: ``openai-compatible`` gets its own vendor, ``"compatible"``, and that is the
#: whole reason this map became the single source (2026-09-13). The property it
#: replaced derived a vendor from the runtime string -- ``runtime.value.split
#: ("-", 1)[0]`` -- which reads ``"openai"`` for a connection pointed at an
#: Ollama box on localhost, so ``Bridge.find(vendor="openai")`` would have
#: handed back a local Llama alongside real OpenAI accounts. The endpoint is
#: whatever the user pointed at; claiming a vendor for it would be a guess, and
#: the honest name for "some OpenAI-shaped endpoint" is not "OpenAI".
VENDOR_OF: dict[Runtime, str] = {
    Runtime.ANTHROPIC_SDK: "anthropic",
    Runtime.OPENAI_SDK: "openai",
    Runtime.GOOGLE_CLI: "google",
    Runtime.GOOGLE_SDK: "google",
    Runtime.ANTHROPIC_API: "anthropic",
    Runtime.OPENAI_API: "openai",
    Runtime.GOOGLE_API: "google",
    Runtime.OPENAI_COMPATIBLE: "compatible",
}

#: Runtimes driven as an in-process HTTP client against a vendor API, paid for
#: by the token. No subprocess, no login store, no vendor-held session.
API_RUNTIMES: frozenset[Runtime] = frozenset(
    {
        Runtime.ANTHROPIC_API,
        Runtime.OPENAI_API,
        Runtime.GOOGLE_API,
        Runtime.OPENAI_COMPATIBLE,
    }
)

#: Runtimes driven as a vendor CLI or agent SDK in a child process.
AGENT_RUNTIMES: frozenset[Runtime] = frozenset(
    {
        Runtime.ANTHROPIC_SDK,
        Runtime.OPENAI_SDK,
        Runtime.GOOGLE_CLI,
        Runtime.GOOGLE_SDK,
    }
)

#: Runtimes that require an explicit opt-in before they may be used (D5).
#: Google stays gated until the Antigravity terms have been read from the primary
#: source; the elevated risk is acceptable personally but not in tools other
#: people run.
#:
#: **``google-api`` is deliberately absent, and that asymmetry is not a bug**
#: (D5 amendment, 2026-09-13). The gate is about the *subscription*: the
#: Antigravity Additional Terms prohibit third-party software accessing the
#: Service, penalty account termination, with no carve-out for driving the
#: first-party CLI -- which is exactly what ``google-cli`` and ``google-sdk``
#: would do. An API key is a different product under a different agreement: the
#: same Additional Terms say a key holder is subject to the Google Cloud terms
#: *instead of* them, and Google's own SDK cannot authenticate against the
#: subscription at all. So metered Gemini through a key was always an ordinary
#: commercial product, and the roadmap already recorded it as "unbuilt rather
#: than prohibited". Adding ``google-api`` here would gate a thing nobody
#: prohibited; removing the other two would ship the thing that is. See
#: ``docs/legal/google.md``.
EXPERIMENTAL_RUNTIMES: frozenset[Runtime] = frozenset(
    {Runtime.GOOGLE_CLI, Runtime.GOOGLE_SDK}
)


def parse_runtime(value: str | Runtime) -> Runtime:
    """Parse a runtime identity, raising ``ValueError`` with the valid set."""
    if isinstance(value, Runtime):
        return value
    try:
        return Runtime(value)
    except ValueError:
        valid = ", ".join(r.value for r in Runtime)
        raise ValueError(f"unknown runtime {value!r} (expected one of: {valid})") from None


def _check_vendor_coverage() -> None:
    """Every runtime must have a vendor, checked at import rather than at use.

    The companion to the capability-table check in
    :mod:`modelpass.capabilities`, and for the same reason: a member added to
    the enum without its row is a mistake that should surface the moment the
    package is imported, not the first time somebody constructs a connection.
    """
    missing = sorted(r.value for r in Runtime if r not in VENDOR_OF)
    if missing:  # pragma: no cover - a failure here breaks import deliberately
        raise RuntimeError(
            "VENDOR_OF is missing an entry for runtime(s): "
            f"{', '.join(missing)}. Every Runtime member needs a vendor -- it is "
            "the single source Connection.vendor reads."
        )
    uncategorised = sorted(
        r.value for r in Runtime if r not in API_RUNTIMES and r not in AGENT_RUNTIMES
    )
    if uncategorised:  # pragma: no cover - same
        raise RuntimeError(
            "runtime(s) in neither API_RUNTIMES nor AGENT_RUNTIMES: "
            f"{', '.join(uncategorised)}. The preflight branches on that "
            "distinction, so a runtime in neither set plans a launch nobody makes."
        )


_check_vendor_coverage()
