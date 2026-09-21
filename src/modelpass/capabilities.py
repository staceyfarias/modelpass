"""The capability registry (D6).

A flat table of runtimes, each with a static capability row, refined at runtime
where a vendor tells us something better (Anthropic's init ``capabilities``
array). Callers get one question to ask: *give me a connection whose runtime
supports X.*

The registry is tri-state on purpose. ``unverified`` is not ``unsupported``: the
provider verification of 2026-08-15 left real holes (Codex cancellation, session
forking outside Anthropic, Antigravity subagent APIs) and reporting a guess as a
fact is exactly the dishonesty this project exists to avoid. ``supports()``
treats ``unverified`` as unusable; ``support()`` tells you why.

Entries carry their own provenance in :data:`NOTES`, which is where the date and
the evidence for a given cell live. The original table is dated 2026-08-15, from a
twelve-field read of all three vendors' primary documentation; later cells were
added by their own verification passes — Phase 5/6 (2026-08-16), Phase 8
(2026-08-17), the ChatSession cells (2026-08-30) and the app-server transport
(2026-08-31). ``docs/api-and-runtimes.md`` §2.2a is the published record of what
each pass checked and when. Re-verify before implementing against any of it.

**A row describes the runtime's DEFAULT transport, and that is why rows move
when a default does.** On 2026-08-31 the ``openai-sdk`` default flipped from
``codex exec`` to ``codex app-server`` (S7 of
the 2026-08-31 transport migration), and seven cells moved with it
in the same commit — ``thinking``, ``tools_in_process``, ``interim_usage``,
``incremental_text``, ``system_prompt_replace`` and ``sessions_list`` up,
``mcp_servers`` down. None of that was new evidence about the vendor; it is the
same evidence, re-read against a different default, which is why that commit *is*
the re-verification. Where a request needs an answer about *itself* rather than
about the default — because it selected the other transport —
:meth:`~modelpass.adapters.base.Adapter.support_for` answers it, and
:meth:`~modelpass.Bridge.find` deliberately does not: somebody picking a connection
has not written the call yet, so the default is the right thing to tell them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .errors import CapabilityNotSupported, InvalidConnection
from .runtimes import API_RUNTIMES, Runtime
from .types import AuthMode

__all__ = [
    "DEFAULT_REGISTRY",
    "STATIC_TABLE",
    "VERIFY_CELLS",
    "Capability",
    "CapabilityRegistry",
    "Support",
    "VerifiedCapabilities",
    "VerifyReport",
    "runtime_auth_modes",
]


class Capability(StrEnum):
    """Things a runtime may or may not be able to do."""

    CHAT = "chat"
    STREAMING = "streaming"
    THINKING = "thinking"
    USAGE_TOKENS = "usage_tokens"
    SESSIONS_RESUME = "sessions_resume"
    SESSIONS_FORK = "sessions_fork"
    SUBAGENTS = "subagents"
    MCP = "mcp"
    TOOLS = "tools"
    # The two above are the coarse 2026-08-15 cells: "is this runtime an MCP
    # client / does it have tools at all". The two below are the Phase 6 cells,
    # and they answer the question a caller can act on: "will
    # bridge.chat(..., mcp_servers=) / (..., tools=) work here" (D12). They are
    # kept separate rather than folded in because a runtime can be an MCP client
    # without offering a *per-run* declaration, and can be an MCP client with no
    # in-process server mechanism on the transport it is driven over. Both cells
    # describe the DEFAULT transport, and 2026-08-31 is the day that stopped
    # being a footnote: flipping openai-sdk's default from 'codex exec' to
    # 'codex app-server' moved tools_in_process UP and mcp_servers DOWN in the
    # same commit, in opposite directions, on the same evidence. Read both notes
    # -- they say which transport can do which, and name the one option that
    # selects it.
    MCP_SERVERS = "mcp_servers"
    TOOLS_IN_PROCESS = "tools_in_process"
    # Phase 5. Distinct from USAGE_TOKENS, which only says the runtime reports
    # tokens *at all*. This one answers the question a guard depends on: does
    # usage arrive while the run is still going? Where it does not, a token guard
    # is a post-hoc report on the run that just happened rather than something
    # that can interrupt it -- and saying so is the difference between a guard
    # and the appearance of one.
    INTERIM_USAGE = "interim_usage"
    # Phase 8 (D13). "Will bridge.chat(..., schema=) work here", answered from a
    # live run on each runtime rather than from a flag in a changelog. Both v1
    # runtimes have a *native* mechanism, and the cell's note is where the one
    # thing they disagree about lives: openai-sdk requires the strict JSON Schema
    # subset, anthropic-sdk does not.
    STRUCTURED_OUTPUT = "structured_output"
    # The ChatSession cells (D14-D19), verified live 2026-08-30. Same move the
    # Phase 6 cells made: split a coarse cell when the fine one is what a caller
    # can act on. STREAMING already says "events arrive incrementally"; it does
    # not say "*text* arrives incrementally", and only the second one tells a UI
    # whether it can render an answer as it is produced. SESSIONS_RESUME says a
    # session can be picked back up if you already hold its id; SESSIONS_LIST
    # says a caller can find out which sessions exist at all. The remaining four
    # are the cells the session objects turn on: which system-prompt semantics a
    # runtime offers (replace vs append only), whether a system message can be
    # added mid-conversation without invalidating the cached prefix, whether
    # persist=False can honestly promise multi-turn, and whether the prefix
    # cache has a TTL lever or only a default.
    INCREMENTAL_TEXT = "incremental_text"
    EPHEMERAL_MULTI_TURN = "ephemeral_multi_turn"
    SYSTEM_PROMPT_REPLACE = "system_prompt_replace"
    MIDCONVERSATION_SYSTEM = "midconversation_system"
    SESSIONS_LIST = "sessions_list"
    TTL_CONTROL = "ttl_control"
    #: Does resuming a session bring its system prompt back with it? The
    #: question a caller can act on is not "can I resume" but "do I have to
    #: re-supply the persona", and the two v1 runtimes answer it opposite ways
    #: for the same underlying reason: whether the prompt lives in the
    #: transcript or in the launch.
    RESUME_CARRIES_SYSTEM_PROMPT = "resume_carries_system_prompt"
    GRACEFUL_CANCEL = "graceful_cancel"
    SUBSCRIPTION_AUTH = "subscription_auth"
    API_KEY_AUTH = "api_key_auth"
    #: The API-runtime cells (R5, 2026-09-13). Split from each other on the same
    #: principle every other split here followed: they are two different
    #: questions and a caller can act on each separately. SAMPLING_CONTROLS asks
    #: whether ``temperature`` / ``top_p`` / ``top_k`` / reasoning effort reach
    #: the model at all; MAX_OUTPUT_TOKENS asks whether a *ceiling on the answer*
    #: can be set, which is a spend lever rather than a style lever and exists on
    #: endpoints that accept none of the others.
    #:
    #: **Both are refined per model, not just per runtime**, which is why R5 says
    #: the table is "keyed by runtime and refined by model": GPT-5 forces
    #: temperature 1.0, some Anthropic models take one of temperature/top_p but
    #: not both, and the o-series refuses all of them. A runtime-level
    #: ``supported`` here means the request field exists on that runtime's wire
    #: protocol, never that every model on it will honour the value -- which is
    #: what the receipt's applied-vs-requested report is for (ticket 1.7).
    SAMPLING_CONTROLS = "sampling_controls"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    #: Does a ``cache_control`` breakpoint the caller placed reach the vendor
    #: (R3, 2026-09-13)? Distinct from :attr:`TTL_CONTROL`, and the two are
    #: routinely confused: TTL_CONTROL asks how *long* a cached prefix lives,
    #: this one asks whether the caller gets to say *where the prefix ends* at
    #: all. A runtime can have neither, either, or both -- ``anthropic-sdk`` has
    #: a TTL lever and no breakpoint channel, which is exactly the pairing that
    #: makes one cell insufficient.
    #:
    #: ``unsupported`` here is never "this runtime does not cache". Three of the
    #: four unsupported rows cache automatically and well; what they do not do
    #: is take an instruction about it. Where the cell is ``unsupported`` and a
    #: caller sent breakpoints anyway, modelpass flattens the content and the
    #: receipt says so by name -- the failure this cell exists to end is the
    #: silent strip, where a consumer builds a four-breakpoint prompt, pays full
    #: price on every call and is told nothing.
    CACHE_BREAKPOINTS = "cache_breakpoints"
    #: Does this runtime cache prompts **without being asked**, with no way for
    #: the caller to turn it off (2026-09-21)? The third cache cell, and the
    #: reason there are three: :attr:`CACHE_BREAKPOINTS` asks whether the caller
    #: may say *where* the prefix ends, :attr:`TTL_CONTROL` asks how *long* it
    #: lives, and neither of them answers *does it happen at all*. Before this
    #: cell existed the table could not distinguish "this runtime ignores your
    #: caching request" from "this runtime was already doing it", which are the
    #: two most different answers a caller can get.
    #:
    #: ``supported`` here is never "you can control this". It is the opposite:
    #: it is the runtimes where a request to cache is **already satisfied** and
    #: there is nothing to send. ``unverified`` is the honest state for a
    #: runtime whose caching modelpass has not read off an SDK surface or seen
    #: in a recorded drive -- including runtimes whose vendors are widely
    #: believed to cache, because a cell is not the place for a belief.
    CACHE_AUTOMATIC = "cache_automatic"
    # THINKING says the runtime *emits* thoughts; this says a caller can ask for
    # how hard it thinks. Separate for the reason MCP/TOOLS are separate from
    # MCP_SERVERS/TOOLS_IN_PROCESS above: one is "does this exist here", the
    # other is the question a caller can act on. A runtime can stream thinking
    # blocks and offer no dial, which is most of them.
    REASONING_EFFORT = "reasoning_effort"


class Support(StrEnum):
    """Tri-state support, because "we did not check" is a real answer."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


_S = Support.SUPPORTED
_N = Support.UNSUPPORTED
_U = Support.UNVERIFIED

#: Static capability table. First filled 2026-08-15 from the three vendors'
#: primary documentation; see docs/api-and-runtimes.md §2.2a.
STATIC_TABLE: dict[Runtime, dict[Capability, Support]] = {
    Runtime.ANTHROPIC_SDK: {
        Capability.CHAT: _S,
        Capability.STREAMING: _S,
        Capability.THINKING: _S,
        Capability.USAGE_TOKENS: _S,
        Capability.SESSIONS_RESUME: _S,
        Capability.SESSIONS_FORK: _S,
        Capability.SUBAGENTS: _S,
        Capability.MCP: _S,
        Capability.TOOLS: _S,
        Capability.MCP_SERVERS: _S,
        Capability.TOOLS_IN_PROCESS: _S,
        Capability.INTERIM_USAGE: _S,
        Capability.STRUCTURED_OUTPUT: _S,
        Capability.INCREMENTAL_TEXT: _S,
        Capability.EPHEMERAL_MULTI_TURN: _S,
        Capability.SYSTEM_PROMPT_REPLACE: _S,
        Capability.MIDCONVERSATION_SYSTEM: _N,
        Capability.SESSIONS_LIST: _S,
        Capability.TTL_CONTROL: _S,
        Capability.RESUME_CARRIES_SYSTEM_PROMPT: _N,
        Capability.GRACEFUL_CANCEL: _S,
        Capability.SUBSCRIPTION_AUTH: _S,
        Capability.API_KEY_AUTH: _S,
        Capability.SAMPLING_CONTROLS: _U,
        Capability.MAX_OUTPUT_TOKENS: _U,
        Capability.CACHE_BREAKPOINTS: _N,
        Capability.CACHE_AUTOMATIC: _S,
        Capability.REASONING_EFFORT: _S,
    },
    # Re-verified 2026-08-31 by S7, the commit that flipped this runtime's
    # default transport from ``codex exec`` to ``codex app-server``. A row
    # describes the DEFAULT transport, so the row and the default had to move
    # together: these cells are the answer to "what does a caller who selects
    # nothing get", and that caller now gets a different transport. Six cells
    # moved up and one moved down; every one of them has a dated note below
    # saying what was driven, and the ones that moved only because the default
    # moved say that too. Nothing here was inferred from the flip -- the
    # evidence predates it and is in the 2026-08-31 consumer-impact review
    # plus tests/fixtures/appserver/.
    Runtime.OPENAI_SDK: {
        Capability.CHAT: _S,
        Capability.STREAMING: _S,
        Capability.THINKING: _S,
        Capability.USAGE_TOKENS: _S,
        Capability.SESSIONS_RESUME: _S,
        Capability.SESSIONS_FORK: _U,
        Capability.SUBAGENTS: _N,
        Capability.MCP: _S,
        Capability.TOOLS: _S,
        Capability.MCP_SERVERS: _N,
        Capability.TOOLS_IN_PROCESS: _S,
        Capability.INTERIM_USAGE: _S,
        Capability.STRUCTURED_OUTPUT: _S,
        Capability.INCREMENTAL_TEXT: _S,
        Capability.EPHEMERAL_MULTI_TURN: _N,
        Capability.SYSTEM_PROMPT_REPLACE: _S,
        Capability.MIDCONVERSATION_SYSTEM: _N,
        Capability.SESSIONS_LIST: _S,
        Capability.TTL_CONTROL: _N,
        Capability.RESUME_CARRIES_SYSTEM_PROMPT: _S,
        Capability.GRACEFUL_CANCEL: _U,
        Capability.SUBSCRIPTION_AUTH: _S,
        Capability.API_KEY_AUTH: _S,
        Capability.SAMPLING_CONTROLS: _U,
        Capability.MAX_OUTPUT_TOKENS: _U,
        Capability.CACHE_BREAKPOINTS: _N,
        Capability.CACHE_AUTOMATIC: _S,
        Capability.REASONING_EFFORT: _S,
    },
    Runtime.GOOGLE_CLI: {
        Capability.CHAT: _S,
        Capability.STREAMING: _S,
        Capability.THINKING: _U,
        Capability.USAGE_TOKENS: _S,
        Capability.SESSIONS_RESUME: _S,
        Capability.SESSIONS_FORK: _U,
        Capability.SUBAGENTS: _U,
        Capability.MCP: _S,
        Capability.TOOLS: _S,
        Capability.MCP_SERVERS: _U,
        Capability.TOOLS_IN_PROCESS: _U,
        Capability.INTERIM_USAGE: _U,
        Capability.STRUCTURED_OUTPUT: _S,
        Capability.INCREMENTAL_TEXT: _U,
        Capability.EPHEMERAL_MULTI_TURN: _U,
        Capability.SYSTEM_PROMPT_REPLACE: _U,
        Capability.MIDCONVERSATION_SYSTEM: _U,
        Capability.SESSIONS_LIST: _U,
        Capability.TTL_CONTROL: _U,
        Capability.RESUME_CARRIES_SYSTEM_PROMPT: _U,
        Capability.GRACEFUL_CANCEL: _U,
        Capability.SUBSCRIPTION_AUTH: _S,
        Capability.API_KEY_AUTH: _N,
        Capability.SAMPLING_CONTROLS: _U,
        Capability.MAX_OUTPUT_TOKENS: _U,
        Capability.CACHE_BREAKPOINTS: _U,
        Capability.CACHE_AUTOMATIC: _U,
        Capability.REASONING_EFFORT: _U,
    },
    Runtime.GOOGLE_SDK: {
        Capability.CHAT: _S,
        Capability.STREAMING: _S,
        Capability.THINKING: _S,
        Capability.USAGE_TOKENS: _U,
        Capability.SESSIONS_RESUME: _S,
        Capability.SESSIONS_FORK: _U,
        Capability.SUBAGENTS: _S,
        Capability.MCP: _S,
        Capability.TOOLS: _S,
        Capability.MCP_SERVERS: _U,
        Capability.TOOLS_IN_PROCESS: _U,
        Capability.INTERIM_USAGE: _U,
        Capability.STRUCTURED_OUTPUT: _S,
        Capability.INCREMENTAL_TEXT: _U,
        Capability.EPHEMERAL_MULTI_TURN: _U,
        Capability.SYSTEM_PROMPT_REPLACE: _U,
        Capability.MIDCONVERSATION_SYSTEM: _U,
        Capability.SESSIONS_LIST: _U,
        Capability.TTL_CONTROL: _U,
        Capability.RESUME_CARRIES_SYSTEM_PROMPT: _U,
        Capability.GRACEFUL_CANCEL: _U,
        Capability.SUBSCRIPTION_AUTH: _N,
        Capability.API_KEY_AUTH: _S,
        Capability.SAMPLING_CONTROLS: _U,
        Capability.MAX_OUTPUT_TOKENS: _U,
        Capability.CACHE_BREAKPOINTS: _U,
        Capability.CACHE_AUTOMATIC: _U,
        Capability.REASONING_EFFORT: _U,
    },
}

#: The cells an API runtime answers with a **checked absence** rather than with
#: "nobody has looked" (2026-09-13). Every one of them is a statement about the
#: shape of an HTTP request/response API, not about a particular vendor: the
#: endpoint keeps no conversation, so there is nothing to resume, fork, list or
#: re-carry a system prompt into; it runs no MCP client and no subagent
#: orchestrator on the caller's behalf; and its prefix cache, where it has one,
#: offers no TTL lever through the request. Those are the same facts for all four
#: runtimes, which is why one set covers them. Each has its own dated note below.
_API_UNSUPPORTED: frozenset[Capability] = frozenset(
    {
        Capability.SUBSCRIPTION_AUTH,
        Capability.MCP_SERVERS,
        Capability.SESSIONS_RESUME,
        Capability.SESSIONS_FORK,
        Capability.SESSIONS_LIST,
        Capability.RESUME_CARRIES_SYSTEM_PROMPT,
        Capability.TTL_CONTROL,
        Capability.SUBAGENTS,
    }
)


def _api_runtime_row() -> dict[Capability, Support]:
    """The placeholder row every API runtime starts life with (ticket 1.2).

    **Built rather than written out, and that is the point.** This table's rule
    is that a cell moves only with evidence, and no adapter for these runtimes
    exists yet -- so there is no drive to record and nothing to read a cell off.
    Writing four hand-made rows of plausible verdicts would produce exactly the
    dishonesty the tri-state was invented to prevent: ``supported`` cells sourced
    from a vendor's documentation and a confident guess.

    So the row is: one ``supported`` cell that is structurally true (an API
    runtime authenticates with a key, which is what makes
    ``Connection.__post_init__`` accept an ``api_key`` connection on it), the
    checked absences in :data:`_API_UNSUPPORTED`, and ``unverified`` everywhere
    else. The unverified cells move in tickets 1.6 and 1.9-1.11, each with a
    recorded drive, exactly as every other cell in this file did.
    """
    row = dict.fromkeys(Capability, Support.UNVERIFIED)
    row.update(dict.fromkeys(_API_UNSUPPORTED, Support.UNSUPPORTED))
    row[Capability.API_KEY_AUTH] = Support.SUPPORTED
    return row


for _api_runtime in sorted(API_RUNTIMES):
    STATIC_TABLE[_api_runtime] = _api_runtime_row()

#: The one cell an API runtime can answer before its adapter exists, because the
#: answer is a fact about the vendor's request schema rather than about a drive
#: (R3, 2026-09-13). Three of the four vendors have no ``cache_control`` field at
#: all -- their prefix caching is automatic and unaddressable -- so a breakpoint
#: sent to them could only ever be dropped, and ``unverified`` there would be
#: pretending not to know something the published request shape states outright.
#: ``anthropic-api`` is left ``unverified`` and moves in ticket 1.6 with a drive,
#: like every other cell on that row: the field is in the schema, and whether
#: modelpass puts a caller's breakpoint through it correctly is the part only a
#: recorded run can say.
for _api_runtime in (Runtime.OPENAI_API, Runtime.OPENAI_COMPATIBLE, Runtime.GOOGLE_API):
    STATIC_TABLE[_api_runtime][Capability.CACHE_BREAKPOINTS] = Support.UNSUPPORTED

#: The ``anthropic-api`` cells the adapter of ticket 1.6 moved (2026-09-13).
#:
#: **What "driven" means on this row, said plainly, because it is not the same
#: thing it meant on the two agent runtimes.** Those rows were filled from live
#: runs against a real subscription, because there was no other way to learn what
#: a CLI does. Every cell below was moved on two pieces of evidence held
#: together: a **fake-transport test** in ``tests/test_adapter_anthropic_api.py``
#: showing what *modelpass* does -- which field it sends, which event it emits,
#: in which order -- and the **installed SDK's own typed surface**
#: (``anthropic`` 0.97.0, read the same day) showing that the vendor takes what
#: modelpass sends. Neither half is sufficient alone: a test against a fake proves
#: only that the adapter is self-consistent, and a signature proves only that a
#: parameter exists. Where the answer needs a *live* observation neither half can
#: produce -- a real cache read, a real refusal, a lifetime that outlives five
#: minutes -- the cell stays ``unverified`` and its note names the live test that
#: moves it. That line is where it is on purpose: it is the line between "this
#: was checked" and "this looked right".
_ANTHROPIC_API_SUPPORTED: tuple[Capability, ...] = (
    Capability.CHAT,
    Capability.STREAMING,
    Capability.INCREMENTAL_TEXT,
    Capability.USAGE_TOKENS,
    Capability.INTERIM_USAGE,
    Capability.TOOLS_IN_PROCESS,
    Capability.STRUCTURED_OUTPUT,
    Capability.THINKING,
    Capability.CACHE_BREAKPOINTS,
    Capability.SYSTEM_PROMPT_REPLACE,
    Capability.GRACEFUL_CANCEL,
    # Ticket 1.7 (R5), on the pair of evidence this row's rule asks for: the
    # fake-transport tests in tests/test_sampling.py section 4 show which fields
    # modelpass puts on the request and which it drops, and anthropic 0.97.0's
    # own Messages.create signature -- temperature, top_p, top_k, max_tokens,
    # output_config.effort -- shows the vendor takes them. Both cells move
    # together, which is what their 1.6 notes said would have to happen.
    #
    # ``supported`` here is a claim about the *runtime's request schema*, never
    # a promise that every model on it honours a value. That second question is
    # per model, it is answered by modelpass.sampling_rules, and the receipt's
    # applied-vs-requested report is where a caller reads the answer. A cell
    # cannot carry it: R5 says so outright.
    Capability.SAMPLING_CONTROLS,
    Capability.MAX_OUTPUT_TOKENS,
)
for _capability in _ANTHROPIC_API_SUPPORTED:
    STATIC_TABLE[Runtime.ANTHROPIC_API][_capability] = Support.SUPPORTED

#: The ``openai-api`` cells the adapter of ticket 1.9 moved (2026-09-13).
#:
#: **Same standard of evidence as the row above, deliberately.** Each cell was
#: moved on a fake-transport test in ``tests/test_adapter_openai_api.py`` showing
#: what modelpass does, held together with the installed ``openai`` 2.32.0 typed
#: surface showing that the Responses API takes it. Neither half alone would do.
#:
#: **``thinking`` is the cell that did *not* move**, and it is worth knowing why,
#: because it moved on ``anthropic-api``. The Responses stream really does carry
#: reasoning text (``response.reasoning_summary_text.delta`` and
#: ``response.reasoning_text.delta`` are typed members of ``ResponseStreamEvent``)
#: and the adapter really does map both to ``ThinkingEvent`` -- but a summary
#: arrives only when the request asks for one, and whether an account is
#: permitted to receive reasoning text at all is an org-level fact no fake can
#: answer. That is precisely the line this table draws between "checked" and
#: "looked right", so the cell stays ``unverified`` and its note names the live
#: test that moves it.
_OPENAI_API_SUPPORTED: tuple[Capability, ...] = (
    Capability.CHAT,
    Capability.STREAMING,
    Capability.INCREMENTAL_TEXT,
    Capability.USAGE_TOKENS,
    Capability.INTERIM_USAGE,
    Capability.TOOLS_IN_PROCESS,
    Capability.STRUCTURED_OUTPUT,
    Capability.SYSTEM_PROMPT_REPLACE,
    Capability.GRACEFUL_CANCEL,
    Capability.SAMPLING_CONTROLS,
    Capability.MAX_OUTPUT_TOKENS,
)
for _capability in _OPENAI_API_SUPPORTED:
    STATIC_TABLE[Runtime.OPENAI_API][_capability] = Support.SUPPORTED

#: The one ``cache_automatic`` cell that moved on 2026-09-21, and it is in its
#: own block rather than in the tuple above because it is not one of the cells
#: ticket 1.9 drove -- putting it there would backdate it.
#:
#: **Read off the installed SDK, and the evidence is two typed fields that only
#: make sense together.** ``openai`` 2.32.0 declares
#: ``ResponseUsage.input_tokens_details.cached_tokens`` as a **required** int on
#: every response ("The number of tokens that were retrieved from the cache"),
#: and ``ResponseCreateParamsBase`` declares
#: ``prompt_cache_retention: Optional[Literal["in-memory", "24h"]]`` whose own
#: docstring reads "Set to ``24h`` to enable **extended** prompt caching". A
#: retention policy that *extends* something, on a request that has no way to
#: *start* it, and a cache-read counter on every single response: that is a
#: cache the caller did not ask for. Held by
#: tests/test_prompt_cache.py::test_openai_api_caches_without_being_asked.
STATIC_TABLE[Runtime.OPENAI_API][Capability.CACHE_AUTOMATIC] = Support.SUPPORTED

# Reasoning effort, read off the installed SDKs on 2026-09-21. The two API
# runtimes that take a named level get the cell here rather than inline,
# because their rows are seeded by the loop above.
#
# openai 2.32.0, openai/types/shared/reasoning_effort.py: ``ReasoningEffort:
# TypeAlias = Optional[Literal["none", "minimal", "low", "medium", "high",
# "xhigh"]]``, carried on ``Reasoning.effort``.
STATIC_TABLE[Runtime.OPENAI_API][Capability.REASONING_EFFORT] = Support.SUPPORTED
# google-genai 1.73.1, ``types.ThinkingLevel``: MINIMAL/LOW/MEDIUM/HIGH on
# ``ThinkingConfig.thinking_level``. Stops at HIGH, which is a fact about the
# ladder rather than about the cell -- modelpass.reasoning reports the move.
STATIC_TABLE[Runtime.GOOGLE_API][Capability.REASONING_EFFORT] = Support.SUPPORTED
# anthropic 0.97.0, output_config_param.py: ``OutputConfigParam.effort`` is
# ``Optional[Literal["low","medium","high","xhigh","max"]]``. The runtime also
# takes ``thinking.budget_tokens``, an integer -- two different questions, and
# modelpass carries the level because that is the portable one. (Corrected
# 2026-09-21: first recorded as unverified on the strength of finding only the
# budget. sampling_rules has routed reasoning_effort to output_config.effort on
# this runtime since ticket 1.7.)
STATIC_TABLE[Runtime.ANTHROPIC_API][Capability.REASONING_EFFORT] = Support.SUPPORTED

#: The ``google-api`` cells the adapter of ticket 1.11 moved (2026-09-13).
#:
#: **Same standard of evidence as the two rows above, deliberately.** Each cell
#: was moved on a fake-transport test in ``tests/test_adapter_google_api.py``
#: showing what modelpass does, held together with the installed ``google-genai``
#: 1.73.1 typed surface showing that ``models.generate_content_stream`` takes it.
#: Neither half alone would do.
#:
#: **``thinking`` is again the cell that did not move**, and for a reason of the
#: same family as ``openai-api``'s: ``Part.thought`` is a typed field and the
#: adapter maps a thought part to a ``ThinkingEvent``, but the model emits one
#: only when the request asks (``options={"include_thoughts": True}`` fills
#: ``thinking_config.include_thoughts``) *and* only on a model that thinks at
#: all. Whether anything ever arrives is a live observation, and its note names
#: the test.
_GOOGLE_API_SUPPORTED: tuple[Capability, ...] = (
    Capability.CHAT,
    Capability.STREAMING,
    Capability.INCREMENTAL_TEXT,
    Capability.USAGE_TOKENS,
    Capability.INTERIM_USAGE,
    Capability.TOOLS_IN_PROCESS,
    Capability.STRUCTURED_OUTPUT,
    Capability.SYSTEM_PROMPT_REPLACE,
    Capability.GRACEFUL_CANCEL,
    Capability.SAMPLING_CONTROLS,
    Capability.MAX_OUTPUT_TOKENS,
)
for _capability in _GOOGLE_API_SUPPORTED:
    STATIC_TABLE[Runtime.GOOGLE_API][_capability] = Support.SUPPORTED

#: And one cell that moved the *other* way on this row, from a checked absence
#: back to an open question (ticket 1.11) -- the same correction ticket 1.6 made
#: to ``anthropic-api``'s ``ttl_control``, and made for the same reason: the
#: shared API note turned out to be false for one vendor, and a cell the SDK
#: contradicts must not keep saying ``unsupported``. ``google-genai`` 1.73.1
#: types ``Tool.mcp_servers`` as ``list[McpServer]`` with ``name`` and
#: ``streamable_http_transport``, which *is* a per-run declaration asking the
#: vendor to be the MCP client. modelpass neither sends it nor has driven it, so
#: the honest position is the open one.
STATIC_TABLE[Runtime.GOOGLE_API][Capability.MCP_SERVERS] = Support.UNVERIFIED

#: The two agent runtimes' sampling cells, moved to a **checked absence**
#: (ticket 1.7, R5). Not a drive: a drive is what you do when only running the
#: thing can tell you, and here reading can. ``grep`` for
#: ``temperature|top_p|top_k|max_tokens|max_output_tokens`` over
#: ``adapters/anthropic.py``, ``adapters/openai.py`` and
#: ``adapters/codex_appserver.py`` returns nothing at all (2026-09-13): there is
#: no parameter to send, on either transport, because neither runtime is a
#: completions endpoint -- they are coding agents driven over a CLI protocol.
#:
#: ``unsupported`` rather than ``unverified`` is the point of the move. The
#: tri-state's third position means "we did not check"; this was checked, and
#: leaving it open had a measurable cost -- a desktop agent app withholds
#: temperature and max_tokens from its subscription chat class entirely because
#: the leaf warned about any value it was given, so a user who sets a
#: temperature on a subscription configuration silently gets nothing. The cell
#: now says so before the call, the receipt says so on the call, and the leaf has
#: stopped shouting.
#:
#: ``google-cli`` and ``google-sdk`` are deliberately **not** moved with them:
#: modelpass ships no adapter for either, so there is nothing to have checked,
#: and an ``unsupported`` sourced from a family resemblance is the guess this
#: table exists to refuse.
for _agent_runtime in (Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK):
    for _capability in (Capability.SAMPLING_CONTROLS, Capability.MAX_OUTPUT_TOKENS):
        STATIC_TABLE[_agent_runtime][_capability] = Support.UNSUPPORTED

#: And one cell that moved the *other* way, from a checked absence back to an
#: open question (ticket 1.6). The shared API note says no request field selects
#: a prefix-cache lifetime on any of these APIs; reading the installed SDK made
#: that false for this one -- ``CacheControlEphemeralParam['ttl']`` is
#: ``Literal['5m', '1h']``. It is not ``supported``, because no test anybody can
#: write today observes a lifetime, and it must not stay ``unsupported``, because
#: that is now a statement the SDK contradicts. Its note says what would move it.
STATIC_TABLE[Runtime.ANTHROPIC_API][Capability.TTL_CONTROL] = Support.UNVERIFIED

#: The sampling note both agent runtimes carry (ticket 1.7, R5). One string
#: rather than two near-identical ones: the reason is the same on both, and two
#: copies of a reason drift.
_AGENT_SAMPLING_NOTE = (
    "unsupported (ticket 1.7, 2026-09-13), and a *checked* absence rather than "
    "an unexamined one: no temperature, top_p, top_k or reasoning parameter "
    "appears anywhere in adapters/anthropic.py, adapters/openai.py or "
    "adapters/codex_appserver.py, and neither CLI protocol carries one. These "
    "runtimes are coding agents, not completions endpoints. Passing sampling= "
    "here is nonetheless **not an error**: every field is dropped and named in "
    "the receipt's sampling_notes, which is the repair for the thing this cell "
    "used to cost -- a desktop agent app stopped passing a temperature to its "
    "subscription chat class at all because the leaf warned about any value it "
    "was given, so a user who set one silently got nothing"
)

#: And the ceiling note. Separate because the distinction it draws is the one
#: callers actually get wrong.
_AGENT_CEILING_NOTE = (
    "unsupported (ticket 1.7, 2026-09-13). Same checked absence as "
    "sampling_controls: there is no output-length parameter on either CLI. What "
    "these runtimes *do* have is Guards.stop_at_tokens, and the two are "
    "routinely confused -- a guard is modelpass's own tripwire over a whole "
    "run's total spend and it ends the stream, while this cell asks whether a "
    "*single answer* can be capped, which the vendor would enforce by "
    "truncating. A caller who wants a bounded bill here is already served; a "
    "caller who wants shorter answers is not, and will not be"
)

#: Why a given cell says what it says, where the reason is not obvious.
NOTES: dict[tuple[Runtime, Capability], str] = {
    (Runtime.ANTHROPIC_SDK, Capability.SAMPLING_CONTROLS): _AGENT_SAMPLING_NOTE,
    (Runtime.ANTHROPIC_SDK, Capability.MAX_OUTPUT_TOKENS): _AGENT_CEILING_NOTE,
    (Runtime.OPENAI_SDK, Capability.SAMPLING_CONTROLS): _AGENT_SAMPLING_NOTE,
    (Runtime.OPENAI_SDK, Capability.MAX_OUTPUT_TOKENS): _AGENT_CEILING_NOTE,
    (Runtime.ANTHROPIC_SDK, Capability.CACHE_BREAKPOINTS): (
        "**unsupported, read off the installed SDK on 2026-09-13** (R3): "
        "claude_agent_sdk 0.2.148 types ClaudeAgentOptions.system_prompt as "
        "'str | SystemPromptPreset | SystemPromptFile | None' (types.py:1967) -- three "
        "shapes, none of them a content-block list, so there is no field a "
        "cache_control could travel in. The prompt channel is no better: query() "
        "takes 'prompt: str | AsyncIterable[dict]' and modelpass's stateless path "
        "flattens the message list into one string anyway, because this runtime has "
        "no messages= array to put blocks in. Nothing in the installed package "
        "mentions cache_control; the sole match in the tree is inside the bundled "
        "claude.exe, which is the CLI making its own caching decisions on the far "
        "side of a boundary the caller cannot reach through. So a breakpoint sent "
        "here is flattened, and the receipt names the drop rather than swallowing "
        "it. **This is not 'no caching'**: the CLI caches, and TTL_CONTROL on this "
        "row is 'supported' because the lifetime is settable (D17). What is absent "
        "is the caller's say in where the prefix ends. Moving this cell is a "
        "one-line re-read of the SDK's options"
    ),
    (Runtime.ANTHROPIC_SDK, Capability.CACHE_AUTOMATIC): (
        "**supported, 2026-09-21**, and the evidence is a recorded drive rather "
        "than a vendor's word. tests/fixtures/structured_output/"
        "anthropic-structured-output-2026-08-17.json is a live capture of four "
        "runs through this adapter, and every one of them reports "
        "cache_creation_input_tokens 605 with cache_read_input_tokens 0 -- the "
        "CLI wrote 605 tokens into a prompt cache on a run where modelpass sent "
        "no caching instruction of any kind, because on this row it cannot: "
        "CACHE_BREAKPOINTS is 'unsupported' and there is no field to send one "
        "in. A cache that fills itself on a request that never mentioned it is "
        "what this cell names. The SDK half, read the same day: claude_agent_sdk "
        "0.2.148 types.py declares cacheReadInputTokens and "
        "cacheCreationInputTokens on the usage shape, so the two counters are "
        "part of the published surface and not an artefact of one build. Note "
        "how this row differs from every other 'supported' cell here: it is not "
        "something a caller can switch on, it is something a caller cannot "
        "switch off, which is why TTL_CONTROL sits beside it as the only lever "
        "there is (D17)"
    ),
    (Runtime.OPENAI_SDK, Capability.CACHE_AUTOMATIC): (
        "**supported, 2026-09-21**, on the live app-server capture of 2026-08-31. "
        "tests/fixtures/appserver/live-capture-2026-08-31.jsonl records one "
        "thread's thread/tokenUsage/updated notifications turn by turn: the first "
        "turn reports cachedInputTokens 0, the second reports 12032 and the third "
        "24064, with cacheWriteInputTokens 0 throughout. modelpass sends no "
        "caching instruction on this transport and has no way to -- "
        "CACHE_BREAKPOINTS is 'unsupported' for two independently checked "
        "reasons, and TTL_CONTROL is 'unsupported' too -- so a prefix that starts "
        "cold and is being read back by the next turn is the vendor caching "
        "unasked. This is the cell that makes 'unsupported' on the other two "
        "cache rows readable: nothing about this runtime's caching is addressable, "
        "and all of it is already happening"
    ),
    (Runtime.OPENAI_SDK, Capability.CACHE_BREAKPOINTS): (
        "unsupported, 2026-09-13 (R3), for two independent reasons either of which "
        "settles it. The vendor has no such lever -- OpenAI prefix caching is "
        "automatic on an exact-prefix match and there is no cache_control field in "
        "the request, the same absence TTL_CONTROL on this row already records. And "
        "the transport could not carry one if there were: 'codex app-server' takes "
        "baseInstructions as a string and its input items are {type: text, text: "
        "...}, with nowhere to hang a marker. Caching here is real and automatic; "
        "what a caller cannot do is address it"
    ),
    (Runtime.OPENAI_SDK, Capability.SUBAGENTS): (
        "no subagent primitive; OpenAI's documented pattern is to run Codex as an "
        "MCP server and orchestrate it from outside (D9)"
    ),
    (Runtime.OPENAI_SDK, Capability.GRACEFUL_CANCEL): (
        "undocumented; pending the Phase 1 cancellation experiment (D10). The floor "
        "is terminating the process. **Still 'unverified' after S7 flipped the "
        "default on 2026-08-31 -- deliberately not moved with the cells that did, "
        "because the flip changes which transport this describes and not what "
        "anybody has checked.** The reason is narrow and it is worth being precise "
        "about which half is missing: the VENDOR's half is driven and SUBPASS's is "
        "not. 'codex app-server' carries turn/interrupt, modelpass "
        "sends it -- from cancel() on a stateless run and from an abandoned "
        "session turn -- and a live drive the same day returned {} and ended the "
        "turn with status 'interrupted' "
        "(tests/fixtures/appserver/live-capability-evidence-2026-08-31.jsonl). "
        "What nobody has checked is the half a caller would rely on: modelpass does "
        "not wait for the terminal the interrupt produces, and no drive has "
        "confirmed the thread takes another turn afterwards -- which is the whole "
        "difference between a graceful cancel and a polite kill. The cell moves "
        "when that is answered, not when the method is sent"
    ),
    (Runtime.OPENAI_SDK, Capability.THINKING): (
        "**supported since 2026-08-31 (S7)**, moved with the default transport, "
        "**and the caveat is part of the verdict.** What arrives on 'codex "
        "app-server' is the reasoning **SUMMARY** stream: "
        "item/reasoning/summaryPartAdded and item/reasoning/summaryTextDelta both "
        "fired on a live turn at effort 'high' with summary 'auto', and the "
        "completed reasoning item carried a summary list with an empty content "
        "list (fixture: "
        "tests/fixtures/appserver/live-capability-evidence-2026-08-31.jsonl). "
        "Raw item/reasoning/textDelta exists in the protocol and did **not** fire "
        "-- not even at that effort and that summary setting. modelpass maps both "
        "summary events onto ThinkingEvent, which is right, but a consumer "
        "promising users 'the model's reasoning' would be overstating what the "
        "wire carries: it is the model's summary of its reasoning. Say so in "
        "product copy. **Narrowed by the exec opt-out to 'unverified', not to "
        "'unsupported'**: nobody has driven reasoning events on 'codex exec' in "
        "either direction, and an absence of knowledge is not a no -- the "
        "app-server evidence says nothing about that transport"
    ),
    (Runtime.OPENAI_SDK, Capability.SESSIONS_FORK): (
        "no forking primitive existed at 2026-08-15, but one shipped since: codex-cli "
        "0.151.0 carries 'codex exec fork <SESSION_ID> [PROMPT]', documented as "
        "forking a previous session by id or thread name into a new one. Still "
        "'unverified' rather than 'supported' because only the subcommand's existence "
        "has been checked (2026-08-30) -- nobody has driven it and confirmed the fork "
        "carries history the way ClaudeAgentOptions.fork_session does. A cheap "
        "experiment, and the cell should move the moment someone runs it"
    ),
    (Runtime.GOOGLE_CLI, Capability.API_KEY_AUTH): (
        "the agy CLI does accept GEMINI_API_KEY, but modelpass drives it only in "
        "subscription mode; API-key work belongs to the google-sdk runtime"
    ),
    (Runtime.GOOGLE_SDK, Capability.SUBSCRIPTION_AUTH): (
        "the Antigravity SDK takes an API key or Vertex ADC only -- no OAuth path "
        "to the subscription (antigravity-sdk-python#20)"
    ),
    # --- Phase 8 cells, verified live 2026-08-17 -------------------------------
    (Runtime.ANTHROPIC_SDK, Capability.STRUCTURED_OUTPUT): (
        "native: ClaudeAgentOptions.output_format={'type':'json_schema','schema':...} "
        "becomes --json-schema on the CLI, and the answer arrives on "
        "ResultMessage.structured_output (parsed) with .result carrying the same "
        "JSON as text. Verified live on a real subscription 2026-08-17 against "
        "claude-agent-sdk 0.2.139 / Claude Code 2.1.63, under modelpass's own launch "
        "options (max_turns=1, tools=[], setting_sources=[]) -- num_turns comes "
        "back as 2 and terminal_reason as 'completed', so the ceiling does not "
        "move. Accepts the LOOSE schema form: optional properties absent from "
        "'required', no additionalProperties. Mechanically it is an end-turn tool "
        "named 'StructuredOutput' (present in the init tools list even under "
        "tools=[]); modelpass routes that call/result pair to vendor_event so a "
        "caller who declared no tools never sees tool traffic. Not combinable "
        "with tools= / mcp_servers= in v1 (D13)"
    ),
    (Runtime.OPENAI_SDK, Capability.STRUCTURED_OUTPUT): (
        "supported on both transports; **stayed supported across the 2026-08-31 "
        "default flip (S7) but changed mechanism**, so the note moved even though "
        "the cell did not. On the DEFAULT transport ('codex app-server') the "
        "schema is a per-turn TurnStartParams.outputSchema field and **no temp "
        "file is written at all** -- driven live on codex 0.151.0-alpha.7.1, where "
        "a strict schema came back as {\"lufs\":-13.7,\"verdict\":\"Pass\"} "
        "(fixture: "
        "tests/fixtures/appserver/live-capability-evidence-2026-08-31.jsonl). On "
        "the exec opt-out it is 'codex exec --output-schema <FILE>', a JSON Schema "
        "*file* modelpass writes for the run and removes afterwards (verified live "
        "2026-08-17 against codex-cli 0.117.0). Either way the answer arrives as "
        "the final agent_message item's text -- no distinct item type, nothing on "
        "turn.completed. "
        "**On the STRICT subset, what was checked and what was not.** The exec "
        "constraint is driven: every object needs additionalProperties:false and "
        "every property must be listed in 'required', both observed as real 400s "
        "naming response_format 'codex_output_schema', captured as fixtures, and "
        "landing before any generation so they cost no tokens. anthropic-sdk does "
        "not require this. On app-server the constraint has **not** been "
        "re-driven: the schema that was sent was already strict, so nobody has "
        "watched a loose one be rejected on this path. modelpass keeps predicting it "
        "on both, and the reason is structural rather than optimistic -- the "
        "subset belongs to the Responses API's response_format, which is "
        "downstream of both transports, and the exec 400 named that rather than "
        "anything about --output-schema. Read it as 'expected to apply, one cheap "
        "400 away from being confirmed', which is what the preflight receipt says: "
        "it names the issues and calls itself a prediction rather than refusing "
        "the run. modelpass.schema.to_openai_strict() converts a loose schema. "
        "Not combinable with tools= / mcp_servers= in v1: openai/codex#15451 "
        "reports --output-schema being silently ignored while MCP servers are "
        "active (D13)"
    ),
    # --- Phase 6 cells, verified 2026-08-16 -----------------------------------
    (Runtime.ANTHROPIC_SDK, Capability.MCP_SERVERS): (
        "ClaudeAgentOptions.mcp_servers takes stdio/sse/http configs per run; "
        "strict_mcp_config plus setting_sources=[] means only the servers modelpass "
        "passes are reachable"
    ),
    (Runtime.ANTHROPIC_SDK, Capability.TOOLS_IN_PROCESS): (
        "create_sdk_mcp_server + @tool host caller functions inside this process; "
        "verified against claude-agent-sdk 0.2.139. The live round trip is still "
        "outstanding -- the dev machine's Claude login is expired"
    ),
    (Runtime.OPENAI_SDK, Capability.MCP_SERVERS): (
        "**moved DOWN to unsupported on 2026-08-31 (S7), and nothing about the "
        "vendor changed -- the default transport did.** This is the one cell the "
        "app-server flip costs, and it is recorded loudly rather than quietly: "
        "per-run MCP servers are a 'codex exec' capability with no driven "
        "equivalent on 'codex app-server', so a caller who selects nothing can no "
        "longer pass mcp_servers=. The row describes the default and a 'supported' "
        "verdict that then failed at the transport is exactly the shape of lie "
        "sessions_list was kept honest about, so the row follows the default down. "
        "**The capability is intact and one option away**: "
        "options={'transport': 'exec'} restores it, OpenAIAdapter.support_for() "
        "answers 'supported' for such a request so the gate does not refuse it, "
        "and the refusal on the default names the option. What exec does, verified "
        "live 2026-08-16 and unchanged: per-invocation declaration via 'codex exec "
        "-c mcp_servers.<name>={...}'. -c merges with ~/.codex/config.toml, so "
        "exclusivity is enforced by enumerate-and-disable: 'codex mcp list --json' "
        "(token-free) lists the configured servers and per-server enabled=false "
        "overrides switch off everything the caller did not name. Fails closed if "
        "enumeration fails; options={'allow_configured_mcp_servers': True} opts "
        "out. Neither half has an app-server equivalent that anybody has driven, "
        "which is why this is 'not built here' rather than 'the vendor lacks it'"
    ),
    (Runtime.OPENAI_SDK, Capability.TOOLS_IN_PROCESS): (
        "**supported since 2026-08-31 (S7)**, and it moved because the DEFAULT "
        "transport moved: this cell reports what a caller who selects nothing "
        "gets, and that is now 'codex app-server'. What was driven, on codex "
        "0.151.0-alpha.7.1: a full round trip -- each ToolDef registered as a "
        "ThreadStartParams.dynamicTools entry, the server's item/tool/call request "
        "answered from the caller's own Python function, the reply sent back as "
        "contentItems, and the model using the result -- and then again "
        "**end-to-end through bridge.chat(tools=[...])** with no options at all. "
        "Fixture: tests/fixtures/appserver/live-capture-2026-08-31.jsonl. The "
        "vendor gates the surface behind capabilities.experimentalApi and calls it "
        "experimental, so re-verify per release. **Narrowed by the exec opt-out**: "
        "'codex exec' has no in-process registration -- its only tool channel is "
        "MCP over a command line, a checked absence since 2026-08-16 -- so "
        "options={'transport': 'exec'} makes this unsupported for that request, "
        "which is what OpenAIAdapter.support_for() answers and what the refusal "
        "says. The two questions stay separate: this cell and find() answer 'what "
        "can I rely on by default', support_for() answers 'will THIS request work'"
    ),
    # --- Phase 5 cells, verified 2026-08-17 -----------------------------------
    (Runtime.ANTHROPIC_SDK, Capability.INTERIM_USAGE): (
        "AssistantMessage.usage carries the API's per-message usage and arrives "
        "once per assistant turn, so a token guard can stop a tool loop between "
        "rounds. Verified in claude-agent-sdk 0.2.139's own message parser, which "
        "fills it from data['message']['usage']; the docs do not cover it. "
        "Granularity is the turn: a single long generation is still interrupted "
        "only at its end. Finer signals exist in the raw StreamEvent payloads "
        "(message_start / message_delta usage) and are deliberately not used yet "
        "-- unverified against a live run"
    ),
    (Runtime.OPENAI_SDK, Capability.INTERIM_USAGE): (
        "**supported since 2026-08-31 (S7)**, and of every cell that moved with "
        "the default this is the one with the largest consumer consequence: a "
        "token ceiling on Codex stops being a post-hoc report and becomes a real "
        "circuit breaker. Driven live on codex 0.151.0-alpha.7.1 -- "
        "thread/tokenUsage/updated arrives after **each** model response, "
        "mid-turn, carrying 'last' and 'total' breakdowns (inputTokens, "
        "cachedInputTokens, cacheWriteInputTokens, outputTokens) and "
        "modelContextWindow; three updates across two turns are in "
        "tests/fixtures/appserver/live-capture-2026-08-31.jsonl. So stop_at_tokens "
        "on a Codex connection now interrupts the run it is watching rather than "
        "bounding the next one, and a consumer that degraded its own guard "
        "language for this runtime can stop. **Narrowed by the exec opt-out**, "
        "where the old finding stands unchanged: the 'codex exec --json' event "
        "vocabulary carries no usage before turn.completed and one exec "
        "invocation is one turn, so usage arrives once, at the end (verified "
        "2026-08-17 from the shipped codex.exe 0.117.0's own event-name table; the "
        "TokenCountEvent in that binary belongs to the app-server/TUI protocol, "
        "which is exactly where it turned out to be usable). A run carrying "
        "options={'transport': 'exec'} therefore still gets a guard that reports "
        "rather than interrupts, and the preflight receipt says so on that run"
    ),
    (Runtime.ANTHROPIC_SDK, Capability.RESUME_CARRIES_SYSTEM_PROMPT): (
        "no -- the system prompt is a *launch option* on the Agent SDK, not part "
        "of the transcript, so resume=<id> restores the conversation and not the "
        "persona it was held under. A resumed session launched without one gets "
        "the minimal tool-calling prompt, and its prefix is not the one the "
        "original warmed, so the cache misses too. Found 2026-08-30 while "
        "implementing resume_session. The consequence is a required argument "
        "rather than a documented trap: Bridge.resume_chat refuses on this "
        "runtime unless system_prompt= is supplied again, and '' is the explicit "
        "way to say the session genuinely had none"
    ),
    (Runtime.OPENAI_SDK, Capability.RESUME_CARRIES_SYSTEM_PROMPT): (
        "yes on both transports, and after the 2026-08-31 default flip (S7) for "
        "two different reasons -- so the note carries both rather than letting the "
        "exec one stand for a default it no longer describes. On the exec opt-out "
        "it is structural: 'codex exec' has no system-prompt parameter, so the "
        "text was serialized into the first turn's prompt and lives in the rollout "
        "with everything else, and resuming replays it (2026-08-30). On the "
        "app-server DEFAULT the prompt is a launch parameter "
        "(ThreadStartParams.baseInstructions), which on Anthropic is exactly the "
        "shape that makes this cell a NO -- but the thread's rollout replays the "
        "conversation the prompt was applied to, so modelpass answers yes and "
        "refuses a re-supplied prompt the same way on both. **Undriven on this "
        "transport**: nobody has resumed a thread and measured whether the "
        "original baseInstructions still reaches the model, and "
        "ThreadResumeParams' own optional baseInstructions field is the seam that "
        "would settle it either way. The mirror consequence of the anthropic-sdk "
        "cell holds regardless: passing system_prompt= to resume_chat here is "
        "*refused*, because it would inject a second persona mid-conversation "
        "rather than restore the first"
    ),
    # --- ChatSession cells (D14-D19), verified live 2026-08-30 ----------------
    (Runtime.ANTHROPIC_SDK, Capability.INCREMENTAL_TEXT): (
        "the Agent SDK streams token-level deltas: with include_partial_messages "
        "(modelpass's default) the raw content_block_delta / text_delta events arrive "
        "mid-generation and _stream_event_to_events maps them onto TextDeltaEvent, "
        "so a UI can render an answer while it is being produced. Split out from "
        "STREAMING because that cell conflates 'events arrive incrementally' with "
        "'text arrives incrementally', and only the second is something a UI can "
        "act on -- reading the coarse cell ships something that looks broken on "
        "half its connections (2026-08-30)"
    ),
    (Runtime.OPENAI_SDK, Capability.INCREMENTAL_TEXT): (
        "**supported since 2026-08-31 (S7)**, and it moved because the default "
        "moved rather than because anything was implemented: "
        "item/agentMessage/delta streams the answer token by token on 'codex "
        "app-server' and was driven live on codex 0.151.0-alpha.7.1 (208 delta "
        "notifications across the committed captures under "
        "tests/fixtures/appserver/), so a UI on a Codex connection can render an "
        "answer while it is being produced. The 2026-08-30 prediction that closing "
        "this cell 'means moving the runtime from exec to app-server' was exactly "
        "right, and that is what happened. **Narrowed by the exec opt-out**: "
        "'codex exec --json' emits the whole agent_message in one item.completed "
        "and there is no partial text there -- agent_message is absent from the "
        "exec lifecycle enum in the shipped codex.exe 0.117.0, which lists only "
        "command_execution, file_change, collab_tool_call and todo_list, so "
        "item.updated never fires for text, and the 0.150.0-alpha.8 binary has an "
        "identical table. The deltas were always in the core protocol enum -- "
        "agent_message_delta, agent_reasoning_delta and "
        "agent_reasoning_raw_content_delta -- and only 'codex app-server' ever "
        "exposed them. **That was a transport limitation, not a mapper gap**: "
        "map_codex_event()'s started/completed-only branch is correct as written "
        "for exec, item.updated can only ever carry tool/progress state there, and "
        "nothing was being discarded"
    ),
    (Runtime.ANTHROPIC_SDK, Capability.EPHEMERAL_MULTI_TURN): (
        "multi-turn with nothing written to disk: ClaudeSDKClient holds the session "
        "in the live subprocess across query() calls, and "
        "CLAUDE_CODE_SKIP_PROMPT_HISTORY=1 suppresses the transcript. Verified live "
        "2026-08-30 -- context carried across two query() calls on one client "
        "(stable session_id, input tokens 213 -> 260, and the second turn answered "
        "correctly from something only the first turn supplied) while "
        "~/.claude/projects/ gained no file; a control run without the env var DID "
        "write one, so the variable is the cause rather than a coincidence. What "
        "persist=False gives up here is cross-process resume, not multi-turn, which "
        "is what makes the flag meaningful on this runtime"
    ),
    (Runtime.OPENAI_SDK, Capability.EPHEMERAL_MULTI_TURN): (
        "**stays unsupported after the 2026-08-31 default flip (S7), and the "
        "reason is now 'not built' rather than 'the runtime cannot'.** The vendor "
        "half is DRIVEN: two turns were taken on one 'ephemeral: true' thread on "
        "codex app-server and the second recalled a number given only in the "
        "first, so an ephemeral multi-turn conversation is a real capability of "
        "the default transport. modelpass has not built it -- a session here always "
        "starts a persisting thread -- so new_chat(persist=False) is refused "
        "rather than silently persisting a conversation somebody asked to leave no "
        "trace of. Read this cell as 'modelpass does not offer this yet', never as "
        "'Codex cannot do this'. The follow-up is small and evidence-backed: send "
        "'ephemeral: true' from _session_thread_start_params when not "
        "request.persist, then check Session.id stays None for life and the thread "
        "never appears in thread/list. On the exec opt-out the older reason still "
        "holds and is a genuine absence: one 'codex exec' is one turn, continuing "
        "a thread requires the rollout file on disk, and there is no live client "
        "holding the conversation in memory (2026-08-30)"
    ),
    (Runtime.ANTHROPIC_SDK, Capability.SYSTEM_PROMPT_REPLACE): (
        "ClaudeAgentOptions.system_prompt accepts a custom string that replaces the "
        "preset entirely: tools remain, but their guidance, the safety rules and "
        "the environment context are gone -- which is precisely what a ChatSession "
        "wants. The preset object form ({'type':'preset','preset':'claude_code'}, "
        "optionally with 'append') is the WorkerSession mapping. The trap worth "
        "recording next to the yes: omitting system_prompt does not give Claude "
        "Code's prompt, it gives the minimal tool-calling one, so a WorkerSession "
        "with no prompt must map to the preset and never to omission (D15/D16, "
        "2026-08-30)"
    ),
    (Runtime.OPENAI_SDK, Capability.SYSTEM_PROMPT_REPLACE): (
        "**supported since 2026-08-31 (S7)**, moved with the default transport. "
        "ThreadStartParams.baseInstructions on 'codex app-server' has REPLACE "
        "semantics and modelpass sends a chat's system prompt there as a parameter, "
        "so bridge.chat(system_prompt=...) and bridge.new_chat(system_prompt=...) "
        "now open a Codex conversation carrying the caller's persona instead of "
        "the vendor's. The evidence is threefold and each part answers a different "
        "objection. (1) MEASURED substitution, not addition: an identical turn "
        "cost 15,940 input tokens with no baseInstructions and 12,397 with a "
        "~10-token one -- ~3,543 tokens REMOVED. Appending a ten-token instruction "
        "can only make a prefix bigger, so a prefix that got smaller is a "
        "replacement and cannot be anything else; read the direction, not the "
        "magnitude (fixture: "
        "tests/fixtures/appserver/live-capability-evidence-2026-08-31.jsonl). "
        "(2) The VENDOR SOURCE says the same thing: codex-rs/protocol/src/models.rs "
        "documents BaseInstructions as corresponding to the 'instructions' field "
        "in the Responses API, with BaseInstructionsProvenance::Custom meaning "
        "explicitly configured and surviving a model change -- so this is the "
        "request's instructions slot, not a prepend. (3) modelpass DRIVES it, on "
        "both the stateless call and a ChatSession. "
        "**The caveat, and it is load-bearing for anyone writing product copy:** "
        "replacing the instructions does NOT by itself erase the coding-agent "
        "character. baseInstructions names the base instructions and nothing else "
        "-- tool definitions, environment context and AGENTS.md are untouched, "
        "which is why ~12.4k input tokens survive a replacement that removed "
        "~3,543 -- and a model infers its identity from its toolbelt as much as "
        "from its prompt, so a replaced prompt can still answer 'yes' to 'are you "
        "a coding agent'. **The toolbelt is a separate switch**, and since D23 "
        "every chat-shaped call throws both -- a ChatSession and a stateless "
        "bridge.chat() alike: chat_tool_overrides() goes on the command line "
        "alongside the replaced instructions. Driven on app-server 2026-08-31: "
        "the toolbelt half alone removed 5,621 prompt tokens (15,890 -> 10,269), "
        "and the caller's own tools kept working across it. A WorkerSession is "
        "the deliberate exception and keeps the whole toolbelt. Claim 'your "
        "persona instead of Codex's framing', not 'Codex stops being a coding "
        "agent': what modelpass can switch off is what -c reaches, and "
        "unified_exec is named in chat_tool_overrides() as the known remainder. "
        "**Narrowed by the exec opt-out**, where D16's finding stands verbatim: "
        "'codex exec' has no system-prompt parameter, a system message becomes a "
        "'System:' line layered on top of the hardcoded 'You are a deployed coding "
        "agent', and a ChatSession there is refused (2026-08-30). A WorkerSession "
        "keeps that layering on BOTH transports, because append is the semantics a "
        "worker asked for and the persona is what it was opened to keep. "
        "developerInstructions remains the vendor's undriven candidate for a "
        "native layered-above channel"
    ),
    (Runtime.ANTHROPIC_SDK, Capability.MIDCONVERSATION_SYSTEM): (
        "appending {'role':'system',...} to messages[] mid-conversation without "
        "invalidating the cached prefix is a **Messages API** feature, model-gated "
        "to Opus 5 / Opus 4.8 / Fable 5 / Mythos 5 -- not Sonnet 5. The Agent SDK "
        "does not expose it: prior turns come from a session, and 'prompt' "
        "documents user-role items only. Logged as a dated, checked 'unsupported' "
        "specifically so nobody re-derives it while hunting for an escape hatch "
        "from D17's immutable tools and system prompt. There is no escape hatch "
        "(2026-08-30)"
    ),
    (Runtime.OPENAI_SDK, Capability.MIDCONVERSATION_SYSTEM): (
        "unsupported on both transports, and the 2026-08-31 default flip (S7) "
        "changed the reasoning without moving the cell -- recorded because the old "
        "sentence expired rather than being overruled. On the exec opt-out it is "
        "blunt: 'codex exec' takes a bare prompt string and has no roles at all, "
        "so there is no messages[] to append to (2026-08-30). On the app-server "
        "DEFAULT that is no longer true -- thread/inject_items takes raw Responses "
        "API items with real roles -- so the honest reason is narrower: injection "
        "APPENDS to a thread's history, which is not the same thing as inserting a "
        "system message mid-conversation, and modelpass sends only user and "
        "assistant roles there. A system-role injected item is **undriven** and no "
        "public API asks for one. So this is a statement about what modelpass "
        "offers, not a claim that the vendor cannot; the Anthropic cell's note has "
        "the API feature this capability is named after"
    ),
    (Runtime.ANTHROPIC_SDK, Capability.SESSIONS_LIST): (
        "list_sessions() / get_session_messages() on the Agent SDK. Scoped by "
        "working directory rather than global: sessions live at "
        "~/.claude/projects/<encoded-cwd>/<id>.jsonl and the cwd is the storage "
        "key, so a listing is per project_folder. get_session_messages() is "
        "read-only introspection -- it reads a transcript back, it cannot be fed "
        "back in as input (2026-08-30)"
    ),
    (Runtime.OPENAI_SDK, Capability.SESSIONS_LIST): (
        "**supported since 2026-08-31 (S7)**, moved with the default transport: "
        "'codex app-server' answers thread/list and modelpass drives it, so "
        "bridge.list_sessions() on a Codex connection enumerates real threads "
        "instead of refusing. Driven live the same day on codex "
        "0.151.0-alpha.7.1, token-free, paginated, account-wide. **One correction "
        "worth carrying**: the page-size parameter is 'limit', not 'pageSize' -- a "
        "probe sent pageSize: 3 and got twenty-five threads back, because an "
        "unknown key here is ignored rather than refused. "
        "**Narrowed by the exec opt-out**, where the 2026-08-30 finding stands and "
        "the refusal is still a refusal rather than an empty list -- supports() is "
        "the caller's gate and '()' would read as 'this connection has no "
        "sessions', a different and false statement about an account that may have "
        "hundreds. From exec "
        "there is no scriptable listing: 'codex exec --help' offers only 'resume "
        "[SESSION_ID]' with --last / --all, and 'codex resume' is an interactive "
        "TUI picker. app-server's stdio JSON-RPC ClientRequest enumerates "
        "thread/list among 58 methods, and it was driven live 2026-08-30: it "
        "returned real modelpass-created threads, paginated, filterable by cwd / "
        "source / provider / archived, with title search, and costs no tokens. The "
        "gotcha that will otherwise look like an empty account: the default "
        "sourceKinds returns interactive sources only, so modelpass's own runs are "
        "invisible unless the request passes sourceKinds: ['exec']. The "
        "zero-dependency fallback is the on-disk layout, "
        "~/.codex/sessions/YYYY/MM/DD/rollout-<ISO8601>-<uuid>.jsonl with a "
        "session_meta first record -- **not** ~/.codex/session_index.jsonl, which "
        "indexes named threads only (187 entries against 202 rollout files) and so "
        "omits every unnamed exec run. "
        "The sourceKinds gotcha above did not bite when it was re-driven on "
        "2026-08-31: a thread modelpass had just created through app-server came "
        "back in a default listing, reported as source 'vscode'. Whether a thread "
        "created by codex exec is still invisible without sourceKinds: ['exec'] "
        "has not been re-checked"
    ),
    (Runtime.ANTHROPIC_SDK, Capability.TTL_CONTROL): (
        "CLAUDE_CODE_PROMPT_CACHE_TTL and CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL "
        "take '5m' | '1h' and reach the CLI through options.env, which the SDK "
        "passes verbatim and lets win over the inherited environment -- so modelpass's "
        "existing additions channel carries them, and setting_sources=[] does not "
        "close this route the way it closes the settings one. **But the lever is "
        "version-gated: it requires Claude Code v2.1.242 or later**. On 2026-08-30 the "
        "resolved CLI was 2.1.233 -- the binary claude-agent-sdk 0.2.139 bundles, "
        "which _find_cli() prefers ahead of anything on PATH -- and a string scan "
        "found neither name there, consistent with the gate rather than evidence "
        "the variables are fictional. Upgrading the SDK to 0.2.148 the same day "
        "moved the bundled CLI to 2.1.251 and the gate opened. Which is the "
        "argument for checking rather than assuming in either direction: the "
        "version that decides this is the SDK's, moves with pip rather than npm, "
        "and crossed the threshold within hours. modelpass checks and falls back to "
        "the runtime default rather than exporting a variable that is silently "
        "ignored. Worth pinning "
        "rather than inheriting: the subscription default is 1 hour and holds in "
        "practice (observed cache_creation of ephemeral_1h_input_tokens: 6220, "
        "ephemeral_5m_input_tokens: 0), but it drops to 5 minutes once the account "
        "draws on credits. Never switch TTL on a live prefix -- a change is a new "
        "cache write at the new tier, not a timer update"
    ),
    (Runtime.OPENAI_SDK, Capability.TTL_CONTROL): (
        "Codex caching is automatic and not selectable: >=1024 tokens, no opt-in, "
        "no write fee, exact-prefix, ~5-10 minutes of idle life against a one-hour "
        "ceiling, and no configuration surface at all. '-c' is a fully generic "
        "config-layer override but there is no key here to override. A checked "
        "absence rather than an unchecked question, and the reason D17's "
        "immutability rules are load-bearing on this runtime: prefix stability is "
        "the only cache lever modelpass has (2026-08-30)"
    ),
}

# --- the API-runtime cells (ticket 1.2, 2026-09-13) --------------------------
#
# Written as one set of texts folded over the four runtimes rather than as forty
# hand-copied strings, because the evidence for each cell genuinely is the same
# evidence: it is a statement about what an HTTP request/response API is, and
# the four runtimes differ in their wire format rather than in that. Where a
# runtime's reason is its own -- google-api's gating, openai-compatible's whole
# row -- it is written out per runtime below and wins.

#: The date these cells were recorded. They are a *plumbing* verdict: no adapter
#: exists yet, nothing has been driven, and the honest half of this row is the
#: ``unverified`` half.
_API_CELLS_DATED = "2026-09-13"

#: What every ``unverified`` cell on an API runtime means, said once.
_API_UNVERIFIED_NOTE = (
    "unverified because nothing has been driven: ticket 1.2 plumbed the runtime "
    "identity, the vendor field and the preflight, and deliberately shipped no "
    "adapter. The cell moves in the adapter ticket with a recorded drive, the way "
    "every other cell in this table did -- a verdict read off a vendor changelog "
    "is exactly what the tri-state exists to refuse "
    f"({_API_CELLS_DATED})"
)

#: The checked absences, each with the fact it was read off.
_API_NOTES: dict[Capability, str] = {
    Capability.API_KEY_AUTH: (
        "an API runtime authenticates with a key and only with a key -- that is "
        "what the runtime *is*. Structural rather than driven, and it is the cell "
        "that makes Connection.__post_init__ accept an api_key connection here at "
        f"all, via runtime_auth_modes() ({_API_CELLS_DATED})"
    ),
    Capability.SUBSCRIPTION_AUTH: (
        "no consumer subscription reaches a raw API endpoint: the Anthropic, "
        "OpenAI and Google subscription products are billed against their own "
        "agent runtimes and their own login stores, and the API is the metered "
        "product sold beside them. A 'subscription' connection here would be a "
        f"connection that cannot resolve a credential ({_API_CELLS_DATED})"
    ),
    Capability.MCP_SERVERS: (
        "a per-run MCP server declaration is an agent-runtime feature: it asks the "
        "runtime to *be* an MCP client and connect to something on the caller's "
        "behalf. A chat-completions endpoint runs no client and connects to "
        "nothing; whatever tool loop happens here happens inside the adapter, in "
        "this process (validation 2026-09-13, §2.7). Read it as 'the endpoint has "
        "no such surface', not as 'modelpass has not built it' "
        f"({_API_CELLS_DATED})"
    ),
    Capability.SESSIONS_RESUME: (
        "the session abstraction rests on the runtime holding the history (D14). "
        "An API endpoint holds none: every request carries its whole conversation, "
        "which is why Bridge.chat(history=...) is the documented multi-turn shape "
        "here. There is no id to resume because there is nothing on the other end "
        f"to resume it from ({_API_CELLS_DATED})"
    ),
    Capability.SESSIONS_FORK: (
        "forking presupposes a stored conversation to fork from, and there is "
        "none -- see the sessions_resume note. Copying a Python list of messages "
        "is something the caller can already do and is not a runtime capability "
        f"({_API_CELLS_DATED})"
    ),
    Capability.SESSIONS_LIST: (
        "nothing to enumerate: no session is stored anywhere, so a listing could "
        "only be modelpass reporting on its own memory. Kept an explicit refusal "
        "rather than an empty tuple for the reason the openai-sdk note gives -- "
        "'()' reads as 'this account has no sessions', which is a different and "
        f"false statement ({_API_CELLS_DATED})"
    ),
    Capability.RESUME_CARRIES_SYSTEM_PROMPT: (
        "follows from sessions_resume: with no resume there is no question about "
        "what a resume carries. Recorded as unsupported rather than unverified so "
        "the cell is not read as an open question somebody should go and drive "
        f"({_API_CELLS_DATED})"
    ),
    Capability.TTL_CONTROL: (
        "no request field selects a prefix-cache lifetime on any of these APIs. "
        "Anthropic's cache_control breakpoints are a different lever -- they say "
        "*where* the cacheable prefix ends, which is a control surface worth "
        "having (R3, ticket 1.5) and is not a TTL. If a ttl field ever appears "
        f"alongside them this cell is the one that moves ({_API_CELLS_DATED})"
    ),
    Capability.SUBAGENTS: (
        "no subagent primitive: an API endpoint answers one request and has no "
        "orchestrator to delegate to. Anything subagent-shaped over an API is the "
        "caller's own second call, which is not this capability "
        f"({_API_CELLS_DATED})"
    ),
}

#: The sentence every ``anthropic-api`` note that leans on a fake-transport test
#: carries, so a reader meets the standard of evidence rather than inferring it.
_API_DRIVEN = (
    "moved by ticket 1.6 on two pieces of evidence, held together: a fake-transport "
    "test in tests/test_adapter_anthropic_api.py showing what modelpass does, and "
    "the installed anthropic 0.97.0 typed surface showing the vendor takes it "
    f"({_API_CELLS_DATED}). "
)

#: What the live file adds, for the cells where a fake genuinely cannot answer.
_API_LIVE_HALF = (
    "The live half is tests/live/test_anthropic_api_live.py (marker `live`, "
    "deselected by default, owner-run):"
)

#: The ``anthropic-api`` sentence, with ticket 1.9's file and SDK in it. Written
#: as its own constant rather than parameterised: the two rows cite two different
#: pieces of evidence and a shared template would hide that.
_OPENAI_API_DRIVEN = (
    "moved by ticket 1.9 on two pieces of evidence, held together: a "
    "fake-transport test in tests/test_adapter_openai_api.py showing what "
    "modelpass does, and the installed openai 2.32.0 typed surface showing the "
    f"Responses API takes it ({_API_CELLS_DATED}). "
)

#: Where ``openai-api``'s live file picks up.
_OPENAI_API_LIVE_HALF = (
    "The live half is tests/live/test_openai_api_live.py (marker `live`, "
    "deselected by default, owner-run):"
)

#: The ``google-api`` sentence, with ticket 1.11's file and SDK in it.
_GOOGLE_API_DRIVEN = (
    "moved by ticket 1.11 on two pieces of evidence, held together: a "
    "fake-transport test in tests/test_adapter_google_api.py showing what "
    "modelpass does, and the installed google-genai 1.73.1 typed surface showing "
    f"that models.generate_content_stream takes it ({_API_CELLS_DATED}). "
)

#: Where ``google-api``'s live file picks up.
_GOOGLE_API_LIVE_HALF = (
    "The live half is tests/live/test_google_api_live.py (marker `live`, "
    "deselected by default, owner-run):"
)

#: Where a runtime's own reason differs from the shared one.
_API_RUNTIME_NOTES: dict[tuple[Runtime, Capability], str] = {
    (Runtime.ANTHROPIC_API, Capability.CACHE_AUTOMATIC): (
        "**unverified, and never consulted, 2026-09-21.** CACHE_BREAKPOINTS is "
        "'supported' on this row -- ticket 1.6 moved it on 2026-09-13 with the "
        "SDK read and the fake-transport test its note names -- so a promptCache "
        "request here resolves to 'explicit' before this cell is read: a runtime "
        "that takes an "
        "instruction is answered by the instruction. Left unverified rather than "
        "moved to 'unsupported' on purpose: 'this endpoint does no caching you "
        "did not ask for' is a claim about vendor behaviour, nobody has driven "
        "it, and the absence of a request field is not evidence of the absence "
        "of a behaviour. Nothing in modelpass depends on the answer"
    ),
    (Runtime.GOOGLE_API, Capability.CACHE_AUTOMATIC): (
        "**unverified, 2026-09-21, and deliberately not moved on what is already "
        "written elsewhere in this file.** The CACHE_BREAKPOINTS note on this row, "
        "recorded by ticket 1.11 on 2026-09-13, says in prose that 'Gemini's "
        "implicit caching is automatic and unaddressable', and that sentence is "
        "not evidence: it was written to "
        "explain why a breakpoint has nowhere to go, not from a read of anything. "
        "What google-genai 1.73.1 actually shows is the *explicit* mechanism -- "
        "GenerateContentConfig.cached_content names a cached-content resource the "
        "caller created beforehand, and "
        "GenerateContentResponseUsageMetadata.cached_content_token_count is "
        "Optional and documents itself as 'the number of tokens in the cached "
        "content that was used', which is the read side of *that* resource. A "
        "non-zero count there is consistent with a cache somebody asked for. "
        "Moving this cell needs two live calls with an identical prefix and no "
        "cached_content, showing the second one read tokens back; until then a "
        "promptCache request on this runtime is refused rather than reported as "
        "already met"
    ),
    (Runtime.OPENAI_API, Capability.CACHE_AUTOMATIC): (
        "**supported, read off the installed SDK on 2026-09-21** -- see the block "
        "beside _OPENAI_API_SUPPORTED for the two typed fields it rests on. In "
        "short: openai 2.32.0 makes "
        "ResponseUsage.input_tokens_details.cached_tokens a required int on every "
        "response, and types prompt_cache_retention as "
        "Optional[Literal['in-memory', '24h']] with a docstring that offers to "
        "*extend* a cache the request never asked to create. That retention field "
        "is also the one cache lifetime modelpass carries to any runtime from the "
        "connection's promptCache key (modelpass.prompt_cache.PROMPT_CACHE_TTLS); "
        "what stays absent, and what CACHE_BREAKPOINTS has said 'unsupported' to "
        "since ticket 1.9 recorded it on 2026-09-13, is any say in where the "
        "prefix ends"
    ),
    (Runtime.ANTHROPIC_API, Capability.CHAT): (
        f"**supported**. {_API_DRIVEN}"
        "test_a_plain_call_streams_text_then_usage_then_a_terminal drives one "
        "stateless call from RunRequest to terminal; messages.stream is the vendor "
        f"surface. {_API_LIVE_HALF} test_one_short_chat"
    ),
    (Runtime.ANTHROPIC_API, Capability.STREAMING): (
        f"**supported**. {_API_DRIVEN}"
        "events arrive while the response is still being produced -- the adapter "
        "iterates the SDK's stream helper inside a `with` block and emits as it "
        "goes, rather than waiting for a finished message. Distinct from "
        "incremental_text, which is the narrower and more useful cell"
    ),
    (Runtime.ANTHROPIC_API, Capability.INCREMENTAL_TEXT): (
        f"**supported**. {_API_DRIVEN}"
        "the stream helper's `text` events carry the delta, and the adapter maps "
        "each to one text_delta "
        "(test_a_plain_call_streams_text_then_usage_then_a_terminal, and "
        "test_the_raw_frames_beneath_the_helper_events_are_not_reported_twice for "
        "the part that would otherwise double-count). This is the cell a UI needs: "
        "an answer can be rendered as it is produced"
    ),
    (Runtime.ANTHROPIC_API, Capability.USAGE_TOKENS): (
        f"**supported**. {_API_DRIVEN}"
        "every response carries a Usage with the four field names modelpass already "
        "maps, and anthropic_api.token_usage makes the same four assignments as the "
        "anthropic-sdk adapter's -- asserted side by side in "
        "test_usage_maps_the_cache_fields_exactly_as_the_agent_runtime_does, because "
        "a consumer's accounting must not learn that a run changed runtimes"
    ),
    (Runtime.ANTHROPIC_API, Capability.INTERIM_USAGE): (
        f"**supported, and better here than on either agent runtime**. {_API_DRIVEN}"
        "each response in a tool loop reports its own counts, the adapter emits them "
        "as a `delta` before the next round starts, and the bridge's guard folds "
        "them in time to stop the loop between rounds -- "
        "test_a_token_guard_stops_the_loop_between_rounds asserts the second round "
        "never happens and that round's tool never runs. That is the granularity "
        "validation 2026-09-13 §2.5 predicted"
    ),
    (Runtime.ANTHROPIC_API, Capability.TOOLS_IN_PROCESS): (
        f"**supported**. {_API_DRIVEN}"
        "the loop lives in the adapter (validation §2.7): ToolDef.handler is called "
        "in this process, results go back as tool_result blocks, tool_call / "
        "tool_result are emitted as the observations they already are, and there is "
        "no turn cap (test_there_is_no_turn_cap). A raising handler becomes a failed "
        "tool result rather than a dead run "
        "(test_a_raising_handler_becomes_a_failed_tool_result_and_the_run_continues). "
        "D12 is unchanged: ToolCallEvent still has no reply path, because nothing "
        f"outside the adapter answers one. {_API_LIVE_HALF} test_one_tool_round_trip"
    ),
    (Runtime.ANTHROPIC_API, Capability.STRUCTURED_OUTPUT): (
        f"**supported, natively**. {_API_DRIVEN}"
        "anthropic 0.97.0 takes output_config={'format': {'type': 'json_schema', "
        "'schema': ...}} on messages.stream, so the schema constrains the model "
        "vendor-side and the schema-as-forced-tool fallback was not built. modelpass "
        "still runs its own structural check and reports it as "
        "StructuredOutputEvent.valid without ever altering data "
        "(test_a_schema_is_sent_as_the_sdks_native_structured_output_config, "
        "test_an_answer_that_misses_the_schema_is_still_an_answer). D13 holds: a "
        "schema and tools are never combined. Unlike openai-sdk, no strict-subset "
        f"restriction applies. {_API_LIVE_HALF} test_one_structured_output"
    ),
    (Runtime.ANTHROPIC_API, Capability.THINKING): (
        f"**supported, with one thing worth knowing**. {_API_DRIVEN}"
        "the SDK types thinking blocks and a `thinking` request parameter, and the "
        "adapter maps a thinking stream event to a thinking event, passed through "
        "opaquely per D7 (test_thinking_blocks_become_thinking_events). What "
        "modelpass does not yet have is a lever to *ask* for it: sampling_params "
        "sends no `thinking` and no effort, so what arrives is whatever the model "
        "does by default -- which on the current Opus line is thinking, and on "
        "others is none. Ticket 1.7 adds the request field; this cell says the "
        "output reaches the caller when the model produces it"
    ),
    (Runtime.ANTHROPIC_API, Capability.SYSTEM_PROMPT_REPLACE): (
        f"**supported**. {_API_DRIVEN}"
        "there is no vendor persona to append to -- system= *is* the prompt -- so "
        "replace is the only semantics the runtime has, which is the opposite of "
        "the worry the cell exists for on the agent runtimes. The adapter sends the "
        "caller's blocks as system= and omits the field entirely when there is no "
        "system message (test_a_run_with_no_system_message_omits_the_field_entirely)"
    ),
    (Runtime.ANTHROPIC_API, Capability.GRACEFUL_CANCEL): (
        f"**supported, and here is exactly what 'graceful' means here**. {_API_DRIVEN}"
        "two halves: closing the iterator exits the `with` around the stream and "
        "tears the in-flight HTTP response down, which is the transport-level floor "
        "the adapter contract documents (test_closing_the_iterator_closes_the_vendor_"
        "stream); and Adapter.cancel() sets a flag the tool loop checks at every "
        "round boundary, so a caller who presses stop during a long loop gets a "
        "cancelled terminal instead of one more paid round "
        "(test_cancel_stops_the_loop_at_the_next_round_boundary). What it is not: "
        "there is no vendor-side interrupt, and a request already in flight is "
        "closed rather than drained"
    ),
    (Runtime.ANTHROPIC_API, Capability.API_KEY_AUTH): (
        "**supported**, and on this runtime the cell carries a second promise worth "
        "naming: the key is the *connection's*. The client is constructed with an "
        "explicit api_key from resolve_credential() and the SDK's own environment "
        "discovery never runs, so a machine with an ambient ANTHROPIC_API_KEY and a "
        "connection naming a different credential bills the account the connection "
        "named. tests/test_no_ambient_credentials.py sets a decoy and asserts it is "
        f"not used ({_API_CELLS_DATED})"
    ),
    (Runtime.ANTHROPIC_API, Capability.CACHE_BREAKPOINTS): (
        "**supported** (R3, ticket 1.6). This is the runtime the whole content-block "
        "vocabulary exists for: messages.create takes system= and per-message "
        "content= as block arrays with cache_control on them, which is where a "
        "downstream agent host's four template breakpoints were always meant to "
        "land. Both halves checked "
        f"on {_API_CELLS_DATED}: the adapter hands "
        "RunRequest.system_blocks / .conversation_blocks through untouched, marker "
        "and ttl intact, pinned by "
        "tests/test_adapter_anthropic_api.py::test_system_blocks_reach_the_vendor_"
        "with_their_cache_control_intact; and anthropic 0.97.0 types "
        f"TextBlockParam['cache_control'] as CacheControlEphemeralParam. {_API_LIVE_HALF} "
        "the cache *working* -- a second identical call reading tokens back -- is "
        "test_cache_breakpoint_is_read_back_on_the_second_call, which asserts "
        "cache_read_input_tokens > 0. This cell is about the breakpoint reaching the "
        "vendor, which is the part that used to fail silently"
    ),
    (Runtime.ANTHROPIC_API, Capability.TTL_CONTROL): (
        "**unverified, and this is a change of answer** (ticket 1.6, "
        f"{_API_CELLS_DATED}). The shared 'no request field selects a prefix-cache "
        "lifetime on any of these APIs' is wrong for this one: anthropic 0.97.0 "
        "types CacheControlEphemeralParam['ttl'] as Literal['5m', '1h'] -- the same "
        "two values modelpass.types.CACHE_TTLS already validates -- and the adapter "
        "carries a caller's ttl through with its block "
        "(tests/test_adapter_anthropic_api.py::test_cache_eligibility_reports_the_"
        "callers_own_ttl_rather_than_claiming_a_pin). It is not 'supported' because "
        "no test anyone can write today observes a *lifetime*: proving a 1h pin "
        "needs two live calls more than five minutes apart still reading cache, "
        "which tests/live/test_anthropic_api_live.py does not do and no fake can. "
        "Moving it is that drive. Note also that modelpass itself pins nothing here "
        "-- the lever is the caller's, which is a different shape from "
        "anthropic-sdk, where modelpass sets the TTL for a session"
    ),
    (Runtime.ANTHROPIC_API, Capability.MIDCONVERSATION_SYSTEM): (
        "**unverified, and the reason is modelpass's rather than the vendor's** "
        f"(ticket 1.6, {_API_CELLS_DATED}). The feature this cell is named after is "
        "real here -- appending {'role':'system',...} to messages[] mid-conversation "
        "without invalidating the cached prefix is a Messages API feature, and the "
        "anthropic-sdk note records it as the place it lives. modelpass does not "
        "offer it yet: RunRequest.system_blocks collects *every* system message, "
        "wherever it sits in the history, and the adapter sends the lot as the "
        "top-level system= block array. So a caller who appends a system message "
        "mid-conversation gets it hoisted to the front, which both changes its "
        "meaning and invalidates the prefix it was supposed to preserve. Moving the "
        "cell needs two things neither of which is this ticket: a per-message split "
        "in the adapter, and a live drive on a model that has the feature (it is "
        "model-gated -- Opus 5 / Opus 4.8 / Fable 5, not Sonnet 5), which is a "
        "runtime-level 'supported' this table cannot honestly give on its own"
    ),
    (Runtime.ANTHROPIC_API, Capability.EPHEMERAL_MULTI_TURN): (
        f"unverified (ticket 1.6, {_API_CELLS_DATED}). Not a vendor question: an API "
        "endpoint is stateless, so a multi-turn conversation that leaves nothing on "
        "disk is exactly what bridge.chat(history=[...]) already is. What the cell "
        "would be claiming is a library-held ChatSession -- validation 2026-09-13 "
        "§2.6 answer 2 -- which was deliberately deferred to its own decision "
        "record. Until that exists there is nothing to verify, and 'unsupported' "
        "would read as 'the runtime cannot', which is the wrong reason"
    ),
    (Runtime.ANTHROPIC_API, Capability.SAMPLING_CONTROLS): (
        f"supported (ticket 1.7, {_API_CELLS_DATED}). anthropic 0.97.0's "
        "Messages.create takes temperature, top_p and top_k, and "
        "OutputConfigParam.effort is Literal['low','medium','high','xhigh','max'], "
        "which is where modelpass's reasoning dial lands; "
        "AnthropicAPIAdapter.sampling_params puts them on the request and "
        "tests/test_sampling.py section 4 drives each one through a fake "
        "transport. **A runtime-level yes, and only that**: which controls a "
        "given model honours is per-model knowledge, it lives in "
        "modelpass.sampling_rules, and the receipt's sampling_requested / "
        "sampling_applied / sampling_notes is where a caller reads what actually "
        "went. claude-sonnet-4-x and claude-haiku-4-x accept one of "
        "temperature/top_p; the adaptive-thinking families take no sampling "
        "parameters at all once an effort is asked for. A cell cannot say that, "
        "which is exactly why R5 pairs it with a report"
    ),
    (Runtime.ANTHROPIC_API, Capability.MAX_OUTPUT_TOKENS): (
        f"supported (ticket 1.7, {_API_CELLS_DATED}). max_tokens is a *required* "
        "field on the Messages API -- there is no runtime default to fall back to "
        "-- so every call carries a ceiling, from sampling=Sampling("
        "max_output_tokens=N), from the options alias it replaced, or from "
        "DEFAULT_MAX_OUTPUT_TOKENS. What moved the cell is not that a number is "
        "sent (1.6 already sent one) but that the receipt now says which and why: "
        "a 4096-token ceiling nobody chose used to be discoverable only from a "
        "truncated answer. Distinct from Guards.stop_at_tokens, which is "
        "modelpass's own tripwire over a whole run's spend and ends the stream; "
        "this is a request parameter the vendor enforces by truncating"
    ),
    (Runtime.ANTHROPIC_API, Capability.TOOLS): (
        f"unverified (ticket 1.6, {_API_CELLS_DATED}). This is the *coarse* cell -- "
        "'does this runtime have a toolbelt of its own' -- and the answer for a "
        "plain API is no in the sense a worker session means it: there is no vendor "
        "persona, no shell, no file tools. What a caller can act on is "
        "tools_in_process, which moved to supported in this ticket. Left unverified "
        "rather than unsupported because the Messages API does carry vendor "
        "server-side tools (web search, code execution) that modelpass has not "
        "driven, and 'unsupported' would be a claim about those too"
    ),
    (Runtime.ANTHROPIC_API, Capability.MCP): (
        f"unverified (ticket 1.6, {_API_CELLS_DATED}). Same split as tools: the "
        "actionable cell is mcp_servers, which is an unsupported checked absence, "
        "and this coarse one would be answering for the vendor's MCP connector -- a "
        "beta the adapter neither sends nor has driven"
    ),
    # --- openai-api, ticket 1.9 -------------------------------------------------
    (Runtime.OPENAI_API, Capability.CHAT): (
        f"**supported**. {_OPENAI_API_DRIVEN}"
        "test_a_plain_call_streams_text_then_usage_then_a_terminal drives one "
        "stateless call from RunRequest to terminal; responses.stream is the "
        f"vendor surface. {_OPENAI_API_LIVE_HALF} test_one_short_chat"
    ),
    (Runtime.OPENAI_API, Capability.STREAMING): (
        f"**supported**. {_OPENAI_API_DRIVEN}"
        "the adapter consumes responses.stream(...) as an iterator and yields as "
        "it goes, so events reach the caller while the response is still being "
        "produced rather than after it; there is no collected-then-replayed path "
        "here at all"
    ),
    (Runtime.OPENAI_API, Capability.INCREMENTAL_TEXT): (
        f"**supported**. {_OPENAI_API_DRIVEN}"
        "response.output_text.delta carries one fragment at a time and becomes "
        "one TextDeltaEvent each, pinned by "
        "test_text_arrives_one_delta_at_a_time_in_the_order_it_was_produced. "
        "Distinct from `streaming`, which is about *when* events arrive: this "
        "cell is about text arriving in pieces rather than as one final blob"
    ),
    (Runtime.OPENAI_API, Capability.USAGE_TOKENS): (
        f"**supported**, with a mapping worth reading. {_OPENAI_API_DRIVEN}"
        "ResponseUsage reports input_tokens with the cached prefix *inside* it "
        "and input_tokens_details.cached_tokens beside it, while TokenUsage "
        "counts cached_input_tokens *separately from* input_tokens -- so "
        "openai_api.token_usage subtracts, and "
        "test_usage_puts_openais_cached_tokens_where_modelpass_keeps_them pins "
        "the arithmetic against anthropic-api's mapping of the same totals. "
        "cache_write_tokens is 0: OpenAI's prefix cache is automatic and bills no "
        "write premium, so there is no wire field. "
        "output_tokens_details.reasoning_tokens stays folded inside "
        "output_tokens, where the vendor already counts it and where Anthropic "
        "counts thinking output"
    ),
    (Runtime.OPENAI_API, Capability.INTERIM_USAGE): (
        f"**supported**. {_OPENAI_API_DRIVEN}"
        "each round's response carries its own usage and the adapter emits it as "
        "a DELTA the moment the round ends -- *before* that round's tools run -- "
        "so a Guards.stop_at_tokens tripwire fires between rounds rather than "
        "after the whole loop. test_a_token_guard_stops_the_loop_between_rounds "
        "drives it through the bridge and asserts the second round never happened"
    ),
    (Runtime.OPENAI_API, Capability.TOOLS_IN_PROCESS): (
        f"**supported**. {_OPENAI_API_DRIVEN}"
        "the loop is the adapter's: function_call items are executed by the "
        "caller's own ToolDef.handler in this process and answered with "
        "function_call_output items. No turn cap (the guard is the bound), a "
        "raising handler becomes a failed tool result, and an undeclared tool is "
        "answered rather than crashed -- the same three rules anthropic-api's "
        "loop follows, pinned by the same-named tests"
    ),
    (Runtime.OPENAI_API, Capability.STRUCTURED_OUTPUT): (
        f"**supported, with a restriction the cell cannot carry**. "
        f"{_OPENAI_API_DRIVEN}"
        "text={'format': {'type': 'json_schema', 'strict': True, ...}} is the "
        "vendor surface, and strict is what makes it *constrain* rather than "
        "suggest. The strict subset has no optional properties, so the adapter "
        "runs the caller's schema through schema.to_openai_strict() and reports "
        "schema.openai_strict_issues() as receipt notes before the call: a "
        "formerly optional property comes back present and possibly null, which "
        "is a change to the answer's shape and is disclosed rather than done "
        f"quietly. {_OPENAI_API_LIVE_HALF} test_one_structured_output"
    ),
    (Runtime.OPENAI_API, Capability.SYSTEM_PROMPT_REPLACE): (
        f"**supported**. {_OPENAI_API_DRIVEN}"
        "instructions= is the Responses API's system prompt and it replaces "
        "rather than appends -- there is no runtime persona underneath it to "
        "append to. RunRequest.system_blocks are flattened into it with flat_text "
        "semantics (one blank line between blocks), because instructions is typed "
        "str and not a block array; the cache_control markers that flattening "
        "drops are named on the receipt by the bridge's own R3 disclosure"
    ),
    (Runtime.OPENAI_API, Capability.GRACEFUL_CANCEL): (
        f"**supported**. {_OPENAI_API_DRIVEN}"
        "two halves, as on every runtime: closing the iterator exits the `with` "
        "around responses.stream and tears the in-flight HTTP response down (the "
        "documented floor, D10), and Adapter.cancel() sets a flag the loop checks "
        "at each round boundary so a long tool loop stops there and reports "
        "cancelled instead of paying for another round"
    ),
    (Runtime.OPENAI_API, Capability.SAMPLING_CONTROLS): (
        f"**supported** (ticket 1.9, {_API_CELLS_DATED}). openai 2.32.0's "
        "Responses.create takes temperature, top_p and max_output_tokens, and "
        "shared.ReasoningEffort is "
        "Literal['none','minimal','low','medium','high','xhigh'], which is where "
        "modelpass's reasoning dial lands as reasoning.effort; "
        "OpenAIAPIAdapter.sampling_params puts them on the request and "
        "tests/test_adapter_openai_api.py section 9 drives each through a fake "
        "transport. **There is no top_k on this runtime at all** -- it is absent "
        "from both of the SDK's create signatures, so modelpass.sampling_rules "
        "does not accept it and a caller who sends one gets it dropped with a "
        "named note rather than a vendor error. **A runtime-level yes, and only "
        "that**: the o-series takes no sampling controls and GPT-5 accepts only "
        "temperature=1.0, which is per-model knowledge living in "
        "modelpass.sampling_rules and reported per call as "
        "sampling_requested / sampling_applied / sampling_notes"
    ),
    (Runtime.OPENAI_API, Capability.MAX_OUTPUT_TOKENS): (
        f"**supported** (ticket 1.9, {_API_CELLS_DATED}). max_output_tokens is a "
        "first-class Responses request field under modelpass's own name for it -- "
        "one of the three reasons ticket 1.9 chose this transport over Chat "
        "Completions, which spells it max_completion_tokens and keeps a "
        "deprecated max_tokens beside it. **Unlike anthropic-api it is optional**, "
        "so modelpass sends no default: a caller who names no ceiling gets the "
        "model's own, and an invented 4096 here would be modelpass truncating "
        "answers nobody asked it to truncate "
        "(tests/test_adapter_openai_api.py::test_no_ceiling_is_invented_when_"
        "nobody_asked_for_one). When the ceiling *is* hit the run "
        "ends as an error naming it, because a truncated answer is "
        "indistinguishable from a finished one downstream"
    ),
    (Runtime.OPENAI_API, Capability.API_KEY_AUTH): (
        "**supported**, and on this runtime the cell carries a second promise "
        "worth naming: the key is the *connection's*. The client is constructed "
        "with an explicit api_key from resolve_credential() and the SDK's own "
        "environment discovery never runs, so a machine with an ambient "
        "OPENAI_API_KEY and a connection naming a different credential bills the "
        "account the connection named. tests/test_no_ambient_credentials.py sets "
        f"a decoy and asserts it is not used ({_API_CELLS_DATED})"
    ),
    (Runtime.OPENAI_API, Capability.THINKING): (
        f"**unverified, and the reason is an account fact rather than a missing "
        f"adapter** (ticket 1.9, {_API_CELLS_DATED}). Half of it is checked: "
        "response.reasoning_summary_text.delta and response.reasoning_text.delta "
        "are typed members of openai 2.32.0's ResponseStreamEvent union, and the "
        "adapter maps both to ThinkingEvent "
        "(tests/test_adapter_openai_api.py::test_reasoning_deltas_become_thinking"
        "_events). The half a fake cannot supply is whether anything ever arrives: "
        "a summary is emitted only when the request asks for one "
        "(options={'reasoning_summary': 'auto'} fills reasoning.summary, which "
        "modelpass does not send on its own because a summary is billed output "
        "nobody requested), and raw reasoning text depends on the model and on "
        "the organisation's verification status. Moving this cell is a live "
        f"observation. {_OPENAI_API_LIVE_HALF} "
        "test_one_reasoning_effort_call, which asks for a summary and asserts a "
        "ThinkingEvent arrives"
    ),
    (Runtime.OPENAI_API, Capability.TOOLS): (
        f"unverified (ticket 1.9, {_API_CELLS_DATED}). The *coarse* cell -- 'does "
        "this runtime have a toolbelt of its own' -- and the answer for a plain "
        "API is no in the sense a worker session means it: no vendor persona, no "
        "shell, no file tools. What a caller can act on is tools_in_process, "
        "which moved to supported in this ticket. Left unverified rather than "
        "unsupported because the Responses API does carry vendor server-side "
        "tools (web_search, file_search, code_interpreter, image_gen -- their "
        "stream events are in the union modelpass passes through as vendor "
        "events) that this adapter neither sends nor has driven, and "
        "'unsupported' would be a claim about those too"
    ),
    (Runtime.OPENAI_API, Capability.MCP): (
        f"unverified (ticket 1.9, {_API_CELLS_DATED}). Same split as tools: the "
        "actionable cell is mcp_servers, an unsupported checked absence, and this "
        "coarse one would be answering for the Responses API's own MCP tool type "
        "-- whose stream events (response.mcp_call.*) are in the union and reach "
        "a caller as vendor events, but which the adapter never sends and nobody "
        "has driven"
    ),
    (Runtime.OPENAI_API, Capability.EPHEMERAL_MULTI_TURN): (
        f"unverified (ticket 1.9, {_API_CELLS_DATED}). Not a vendor question: an "
        "API endpoint is stateless, so a multi-turn conversation that leaves "
        "nothing on disk is exactly what bridge.chat(history=[...]) already is. "
        "What the cell would be claiming is a library-held ChatSession, "
        "deliberately deferred to its own decision record. Note that this runtime "
        "*also* has a vendor-held history (conversation, previous_response_id) "
        "which is the opposite of ephemeral and is not a modelpass session "
        "either -- a response id is not resumable, listable or forkable in D14's "
        "sense, and the adapter sends neither field"
    ),
    (Runtime.OPENAI_API, Capability.MIDCONVERSATION_SYSTEM): (
        f"unverified, and the reason is modelpass's rather than the vendor's "
        f"(ticket 1.9, {_API_CELLS_DATED}). RunRequest.system_blocks collects "
        "*every* system message wherever it sits in the history, and the adapter "
        "sends the lot as top-level instructions= -- so a caller who appends a "
        "system message mid-conversation gets it hoisted to the front, which "
        "changes its meaning and invalidates the prefix it was meant to preserve. "
        "The Responses input array can carry a role='system' or role='developer' "
        "message in place, so the vendor side is not the obstacle; the per-message "
        "split in modelpass is, and it needs its own ticket and its own drive"
    ),
    (Runtime.OPENAI_API, Capability.CACHE_BREAKPOINTS): (
        "unsupported, and a checked absence rather than an open question (R3): "
        "OpenAI prefix caching is automatic and vendor-side -- an exact prefix match "
        "at or above the vendor's minimum is cached without being asked -- and the "
        "request schema carries no cache_control to ask with. Breakpoints sent here "
        "are flattened into text and the receipt names the drop. Prefix stability, "
        f"not a marker, is the whole of the control available ({_API_CELLS_DATED})"
    ),
    (Runtime.OPENAI_COMPATIBLE, Capability.CACHE_BREAKPOINTS): (
        "unsupported at the level this runtime can answer for: the OpenAI-shaped "
        "request schema it speaks carries no cache_control, whatever the server "
        "behind the base URL does with prefixes of its own accord. A server that "
        "adds a vendor extension is something refine() can record for that install "
        f"(ticket 1.10); the schema-level answer is no ({_API_CELLS_DATED})"
    ),
    # --- google-api, ticket 1.11 ------------------------------------------------
    (Runtime.GOOGLE_API, Capability.CHAT): (
        f"**supported**. {_GOOGLE_API_DRIVEN}"
        "test_a_plain_call_streams_text_then_usage_then_a_terminal drives one "
        "stateless call from RunRequest to terminal; "
        "models.generate_content_stream is the vendor surface. "
        f"{_GOOGLE_API_LIVE_HALF} test_one_short_chat"
    ),
    (Runtime.GOOGLE_API, Capability.STREAMING): (
        f"**supported**. {_GOOGLE_API_DRIVEN}"
        "the adapter iterates the chunks generate_content_stream yields and "
        "emits as it goes, so events reach the caller while the answer is still "
        "being produced. Note the shape difference from the other three: a chunk "
        "here is a whole GenerateContentResponse carrying whatever parts have "
        "arrived, not a typed event with a name on it, which is why the mapping "
        "reads parts rather than event types"
    ),
    (Runtime.GOOGLE_API, Capability.INCREMENTAL_TEXT): (
        f"**supported**. {_GOOGLE_API_DRIVEN}"
        "each chunk's text parts become one TextDeltaEvent each, in order, pinned "
        "by test_text_arrives_one_delta_at_a_time_in_the_order_it_was_produced. "
        "The cell a UI needs: an answer can be rendered as it is produced"
    ),
    (Runtime.GOOGLE_API, Capability.USAGE_TOKENS): (
        f"**supported**, with the most arithmetic of the four mappings. "
        f"{_GOOGLE_API_DRIVEN}"
        "usage_metadata reports prompt_token_count *including* the cached content "
        "(the SDK says so in the field's own description) and reports "
        "thoughts_token_count *beside* candidates_token_count rather than inside "
        "it (total_token_count is documented as the sum of the four). TokenUsage "
        "counts cached input separately and counts thinking output as output, so "
        "google_api.token_usage subtracts the first and adds the second -- and "
        "test_usage_puts_geminis_four_counts_where_modelpass_keeps_them pins the "
        "result against anthropic-api's mapping of the same real call, because a "
        "consumer's accounting must not learn that a run changed runtimes. "
        "cache_write_tokens is 0: a generateContent response reports no write "
        "count"
    ),
    (Runtime.GOOGLE_API, Capability.INTERIM_USAGE): (
        f"**supported**. {_GOOGLE_API_DRIVEN}"
        "each round's usage_metadata is emitted as a DELTA the moment the round "
        "ends -- before that round's tools run -- so a Guards.stop_at_tokens "
        "tripwire fires between rounds rather than after the whole loop. "
        "test_a_token_guard_stops_the_loop_between_rounds drives it through the "
        "bridge and asserts the second round never happened"
    ),
    (Runtime.GOOGLE_API, Capability.TOOLS_IN_PROCESS): (
        f"**supported**. {_GOOGLE_API_DRIVEN}"
        "the loop is the adapter's: function_call parts are executed by the "
        "caller's own ToolDef.handler in this process and answered with "
        "function_response parts in a user turn, after the model's own turn is "
        "replayed verbatim. No turn cap (the guard is the bound), a raising "
        "handler becomes a failed tool result, and an undeclared tool is answered "
        "rather than crashed -- the same three rules the other three loops "
        "follow, pinned by the same-named tests. Two vendor differences worth "
        "knowing: FunctionCall.args arrives already parsed (no JSON string), and "
        "FunctionCall.id is optional and usually absent, so the ToolCallEvent id "
        f"falls back to the function's name. {_GOOGLE_API_LIVE_HALF} "
        "test_one_tool_round_trip"
    ),
    (Runtime.GOOGLE_API, Capability.STRUCTURED_OUTPUT): (
        f"**supported, natively**. {_GOOGLE_API_DRIVEN}"
        "response_mime_type='application/json' plus response_json_schema=<the "
        "caller's schema> is the vendor surface, and the mime type is required "
        "when a schema is sent. **No strict rewrite happens here**: that was an "
        "OpenAI strict:true requirement rather than a universal one, so an "
        "optional property stays optional and no receipt note is needed. The SDK "
        "documents a JSON Schema subset; modelpass validates the answer it gets "
        "back either way and reports it as StructuredOutputEvent.valid without "
        f"ever altering data. D13 holds. {_GOOGLE_API_LIVE_HALF} "
        "test_one_structured_output"
    ),
    (Runtime.GOOGLE_API, Capability.SYSTEM_PROMPT_REPLACE): (
        f"**supported**. {_GOOGLE_API_DRIVEN}"
        "config.system_instruction *is* the prompt -- there is no runtime persona "
        "underneath it to append to -- and the adapter omits the field entirely "
        "when a run carries no system message. The caller's blocks are flattened "
        "into it with flat_text semantics (one blank line between blocks); the "
        "cache_control markers that flattening drops are named on the receipt by "
        "the bridge's own R3 disclosure"
    ),
    (Runtime.GOOGLE_API, Capability.GRACEFUL_CANCEL): (
        f"**supported**. {_GOOGLE_API_DRIVEN}"
        "two halves, as on every runtime: closing the iterator runs the adapter "
        "generator's finally and closes the vendor stream (the documented floor, "
        "D10 -- this SDK's stream is an iterator rather than a context manager, "
        "so the close is explicit here where the other two use a `with`), and "
        "Adapter.cancel() sets a flag the loop checks at each round boundary so a "
        "long tool loop stops there and reports cancelled instead of paying for "
        "another round"
    ),
    (Runtime.GOOGLE_API, Capability.SAMPLING_CONTROLS): (
        f"**supported** (ticket 1.11, {_API_CELLS_DATED}). google-genai 1.73.1's "
        "GenerateContentConfig takes temperature, top_p, **top_k** and "
        "max_output_tokens, and ThinkingConfig.thinking_level is "
        "MINIMAL | LOW | MEDIUM | HIGH, which is where modelpass's reasoning dial "
        "lands word for word; GoogleAPIAdapter.sampling_params puts them on the "
        "config and tests/test_adapter_google_api.py section 9 drives each "
        "through a fake transport. **top_k is the control this runtime has and "
        "neither OpenAI runtime does** -- it is dropped with a named note there "
        "and sent here, which is the whole reason the rules table is keyed by "
        "runtime. **No thinking_budget is ever sent**: that field is a token "
        "count whose allowed range the SDK documents as model dependent, and "
        "ticket 1.7 refused to invent one for Anthropic on the same ground. **A "
        "runtime-level yes, and only that**: which Gemini model honours which "
        "control is per-model knowledge nobody here has driven, so every call "
        "that asks for sampling carries the 'model rules unknown; runtime "
        "defaults applied' note and the receipt's sampling_requested / "
        "sampling_applied / sampling_notes is where a caller reads what went"
    ),
    (Runtime.GOOGLE_API, Capability.MAX_OUTPUT_TOKENS): (
        f"**supported** (ticket 1.11, {_API_CELLS_DATED}). max_output_tokens is a "
        "GenerateContentConfig field under modelpass's own name for it. **Unlike "
        "anthropic-api it is optional**, so modelpass sends no default: a caller "
        "who names no ceiling gets the model's own, and an invented 4096 here "
        "would be modelpass truncating answers nobody asked it to truncate. When "
        "the ceiling *is* hit the run ends as an error naming it (finish_reason "
        "MAX_TOKENS), because a truncated answer is indistinguishable from a "
        "finished one downstream (tests/test_adapter_google_api.py::"
        "test_no_ceiling_is_invented_when_nobody_asked_for_one and "
        "::test_a_truncated_answer_says_which_ceiling_it_hit). Distinct from "
        "Guards.stop_at_tokens, which is modelpass's own tripwire over a whole "
        "run's spend"
    ),
    (Runtime.GOOGLE_API, Capability.THINKING): (
        f"**unverified, and the reason is a request modelpass does not make on "
        f"its own** (ticket 1.11, {_API_CELLS_DATED}). Half of it is checked: "
        "Part.thought is a typed boolean on google-genai 1.73.1's Part, a thought "
        "part carries ordinary text, and the adapter separates the two so private "
        "reasoning becomes a ThinkingEvent and never leaks into the answer "
        "(tests/test_adapter_google_api.py::test_a_thought_part_becomes_thinking_"
        "and_never_text). The half a fake cannot supply is whether anything ever "
        "arrives: thoughts are emitted only when the request asks "
        "(options={'include_thoughts': True} fills "
        "thinking_config.include_thoughts, which modelpass does not send on its "
        "own because thought output is billed output nobody requested) and only "
        "on a model that thinks at all. Moving this cell is a live observation. "
        f"{_GOOGLE_API_LIVE_HALF} test_one_thinking_call, which asks for thoughts "
        "and asserts a ThinkingEvent arrives"
    ),
    (Runtime.GOOGLE_API, Capability.TTL_CONTROL): (
        f"unsupported, and this runtime's reason is its own (ticket 1.11, "
        f"{_API_CELLS_DATED}). Gemini *does* have a settable cache lifetime -- "
        "but it belongs to a **separate resource**: client.caches.create() "
        "returns a named cached-content object with a ttl, which a later request "
        "references through config.cached_content. That is not a lever on the "
        "call this adapter makes, and modelpass does not create such an object: "
        "doing so would make the library the owner of a billed, expiring, "
        "server-side resource nobody asked it to create. If this ever becomes a "
        "feature it is a decision record of its own, and this is the cell that "
        "moves with it"
    ),
    (Runtime.GOOGLE_API, Capability.MCP_SERVERS): (
        "**unverified, and this is a change of answer** (ticket 1.11, "
        f"{_API_CELLS_DATED}). The shared 'a chat endpoint runs no MCP client' "
        "absence is wrong for this one: google-genai 1.73.1 types "
        "Tool.mcp_servers as list[McpServer], each with a name and a "
        "streamable_http_transport, which *is* a per-run declaration asking the "
        "vendor to connect to a server on the caller's behalf (the SDK notes it "
        "is not supported on Vertex AI). It is not 'supported', because modelpass "
        "never sends the field and nobody has driven it -- and it must not stay "
        "'unsupported', because that is now a statement the SDK contradicts. "
        "Moving it is an adapter change plus a drive, not a re-read"
    ),
    (Runtime.GOOGLE_API, Capability.TOOLS): (
        f"unverified (ticket 1.11, {_API_CELLS_DATED}). The *coarse* cell -- 'does "
        "this runtime have a toolbelt of its own' -- and the answer for a plain "
        "API is no in the sense a worker session means it: no vendor persona, no "
        "shell, no file tools. What a caller can act on is tools_in_process, "
        "which moved to supported in this ticket. Left unverified rather than "
        "unsupported because Tool carries a row of vendor-side tools this adapter "
        "never sends and nobody has driven -- google_search, code_execution, "
        "url_context, file_search -- and 'unsupported' would be a claim about "
        "those too"
    ),
    (Runtime.GOOGLE_API, Capability.MCP): (
        f"unverified (ticket 1.11, {_API_CELLS_DATED}). Same split as tools, and "
        "here the actionable cell moved with it: mcp_servers on this row is no "
        "longer a checked absence, because Tool.mcp_servers exists. Neither cell "
        "is a yes -- the adapter sends no MCP tool and nothing has been driven"
    ),
    (Runtime.GOOGLE_API, Capability.EPHEMERAL_MULTI_TURN): (
        f"unverified (ticket 1.11, {_API_CELLS_DATED}). Not a vendor question: an "
        "API endpoint is stateless, so a multi-turn conversation that leaves "
        "nothing on disk is exactly what bridge.chat(history=[...]) already is. "
        "What the cell would be claiming is a library-held ChatSession, "
        "deliberately deferred to its own decision record. Note that this SDK "
        "*does* ship client.chats.create(), which is a client-side list of turns "
        "in the caller's own process -- the same thing history=[...] is, under "
        "another name, and not a vendor-held session either"
    ),
    (Runtime.GOOGLE_API, Capability.MIDCONVERSATION_SYSTEM): (
        f"unverified, and the reason is modelpass's rather than the vendor's "
        f"(ticket 1.11, {_API_CELLS_DATED}). RunRequest.system_blocks collects "
        "*every* system message wherever it sits in the history, and the adapter "
        "sends the lot as config.system_instruction -- so a caller who appends a "
        "system message mid-conversation gets it hoisted to the front, which "
        "changes its meaning. The per-message split in modelpass is the obstacle, "
        "and it needs its own ticket and its own drive"
    ),
    (Runtime.GOOGLE_API, Capability.CACHE_BREAKPOINTS): (
        "unsupported (R3): Gemini's implicit caching is automatic and unaddressable, "
        "and its explicit caching is a *separate resource* -- a cached-content "
        "object created ahead of time and referenced by name -- rather than a marker "
        "inside a message. An Anthropic-shaped breakpoint has nowhere to go in "
        "either mechanism, so it is flattened and reported rather than silently "
        "translated into a cache object whose lifetime modelpass would then own "
        f"({_API_CELLS_DATED})"
    ),
    (Runtime.GOOGLE_API, Capability.API_KEY_AUTH): (
        "**ungated on purpose, and the asymmetry with google-cli / google-sdk is "
        "the decision rather than an oversight** (D5 amendment, "
        f"{_API_CELLS_DATED}). Those two sit in EXPERIMENTAL_RUNTIMES and this one "
        "does not, because they are two different products under two different "
        "agreements. The Antigravity Additional Terms prohibit third-party "
        "software accessing the *subscription* Service -- penalty account "
        "termination, no carve-out for driving the first-party CLI -- which is "
        "precisely what a subscription runtime would do. The same Additional "
        "Terms say a Gemini API key holder is subject to the Google Cloud terms "
        "*instead of* them, and Google's own SDK cannot authenticate against the "
        "subscription at all, so metered Gemini through a key is an ordinary "
        "commercial product and always was; the roadmap already recorded it as "
        "'unbuilt rather than prohibited'. Gating this runtime would gate "
        "something nobody prohibited. Do not 'fix' the inconsistency in either "
        "direction without re-reading docs/legal/google.md and the primary terms. "
        "Since ticket 1.11 the cell carries the same second promise the other "
        "three API rows do, and it matters more here than anywhere: the key is "
        "the *connection's*. google.genai.Client() discovers **two** ambient "
        "names on its own -- GEMINI_API_KEY and GOOGLE_API_KEY -- so the client "
        "is built with an explicit api_key from resolve_credential() and that "
        "discovery never runs; tests/test_no_ambient_credentials.py sets a decoy "
        "for both and asserts neither is used"
    ),
    (Runtime.OPENAI_COMPATIBLE, Capability.API_KEY_AUTH): (
        "an OpenAI-shaped endpoint takes a bearer key, and modelpass sends the one "
        "the connection names. **Whether the endpoint checks it is not modelpass's "
        "to say**: a local Ollama or LM Studio box accepts anything, a LiteLLM "
        "proxy or a gateway enforces its own. The cell reports that this runtime "
        "authenticates by key rather than by login, which is the question "
        f"Connection validation asks ({_API_CELLS_DATED})"
    ),
}

#: The prefix on every ``openai-compatible`` cell. The runtime is a *shape*, not
#: a vendor, so no verdict about it can be a fact about the thing on the other
#: end of the base URL.
_COMPATIBLE_PREFIX = (
    "openai-compatible is unverified by construction: the runtime names an API "
    "*shape*, and what actually answers is whatever the connection's baseUrl "
    "points at -- Ollama, LM Studio, OpenRouter, a LiteLLM proxy, a gateway. Two "
    "installs of this runtime can legitimately disagree about every cell, so a "
    "static table can only report the shape. **refine() is the upgrade path, per "
    "install**: run 'modelpass verify <connection>', which drives one short chat, "
    "one tool round trip and one structured-output call against your endpoint and "
    "records what actually worked on the connection; every later call folds that "
    "into a registry copy (CapabilityRegistry.refine / .refined), which is what "
    "that hook was built for and what ticket 1.10 wired up. Until you have run it, "
    f"treat a cell here as a question about your endpoint ({_API_CELLS_DATED}). "
)


def _record_api_notes() -> None:
    """Fill :data:`NOTES` for every cell on every API runtime.

    Every cell gets one, including the ``unverified`` ones: this table's contract
    is that a reader can ask *why* and get an answer, and "nobody has driven it"
    is an answer worth writing down when the alternative is a reader assuming it
    was checked.
    """
    for runtime in sorted(API_RUNTIMES):
        compatible = runtime is Runtime.OPENAI_COMPATIBLE
        for capability in Capability:
            key = (runtime, capability)
            text = _API_RUNTIME_NOTES.get(key) or _API_NOTES.get(capability)
            if text is None:
                text = _API_UNVERIFIED_NOTE
            NOTES[key] = f"{_COMPATIBLE_PREFIX}{text}" if compatible else text


_record_api_notes()


def _check_table_coverage() -> None:
    """Every :class:`Runtime` member must have a row, checked at import.

    The latent defect this closes: a member added to the enum without a row here
    used to surface as ``ValueError: no capability row for runtime ...`` raised
    from :func:`runtime_auth_modes` during ``Connection.__post_init__`` -- so the
    first report of a missing row was a user's connection failing to construct,
    with a message about capabilities that did not say the table was incomplete.
    A table with a hole in it is a packaging mistake, and a packaging mistake
    should fail where packaging mistakes are found: at import, in the test suite,
    before anything ships.
    """
    missing = sorted(r.value for r in Runtime if r not in STATIC_TABLE)
    if missing:  # pragma: no cover - a failure here breaks import deliberately
        raise RuntimeError(
            "STATIC_TABLE is missing a capability row for runtime(s): "
            f"{', '.join(missing)}. Every Runtime member needs one -- "
            "runtime_auth_modes() reads it during Connection construction, so a "
            "missing row is a connection that cannot be built. Add the row with "
            "dated evidence in NOTES, or ``unverified`` cells if nothing has been "
            "driven yet."
        )


_check_table_coverage()


class CapabilityRegistry:
    """Answers "what can this runtime do", honestly.

    Instances own a copy of the table, so runtime refinement never mutates the
    verified static facts shared by other callers.
    """

    def __init__(
        self,
        table: Mapping[Runtime, Mapping[Capability, Support]] | None = None,
    ) -> None:
        source = STATIC_TABLE if table is None else table
        self._table: dict[Runtime, dict[Capability, Support]] = {
            runtime: dict(row) for runtime, row in source.items()
        }

    def __iter__(self) -> Iterator[Runtime]:
        return iter(self._table)

    def row(self, runtime: Runtime) -> Mapping[Capability, Support]:
        """The full capability row for a runtime."""
        try:
            return dict(self._table[runtime])
        except KeyError:
            raise ValueError(f"no capability row for runtime {runtime!r}") from None

    def support(self, runtime: Runtime, capability: Capability | str) -> Support:
        """Tri-state support for one capability."""
        cap = Capability(capability)
        return self.row(runtime).get(cap, Support.UNVERIFIED)

    def supports(self, runtime: Runtime, capability: Capability | str) -> bool:
        """``True`` only for verified support. ``unverified`` is not a yes."""
        return self.support(runtime, capability) is Support.SUPPORTED

    def require(self, runtime: Runtime, capability: Capability | str) -> None:
        """Raise :class:`CapabilityNotSupported` unless support is verified."""
        support = self.support(runtime, capability)
        if support is not Support.SUPPORTED:
            raise CapabilityNotSupported(str(runtime), str(capability), str(support))

    def note(self, runtime: Runtime, capability: Capability | str) -> str | None:
        """Why the table says what it says, when there is something to say."""
        return NOTES.get((runtime, Capability(capability)))

    def runtimes_supporting(self, capability: Capability | str) -> tuple[Runtime, ...]:
        """Every runtime with verified support for a capability."""
        cap = Capability(capability)
        return tuple(r for r in self._table if self.supports(r, cap))

    def refine(self, runtime: Runtime, observed: Mapping[str, bool | Support]) -> None:
        """Fold runtime-reported capabilities into this registry instance.

        This is the hook for Anthropic's init ``capabilities`` array and anything
        equivalent elsewhere: the vendor telling us, at session start, what this
        particular install can do. Unknown capability names are ignored rather
        than raising -- a vendor adding a feature we have never heard of is not
        an error.
        """
        row = self._table.setdefault(runtime, {})
        for name, value in observed.items():
            try:
                cap = Capability(name)
            except ValueError:
                continue
            if isinstance(value, Support):
                row[cap] = value
            else:
                row[cap] = Support.SUPPORTED if value else Support.UNSUPPORTED

    def refined(
        self, runtime: Runtime, observed: Mapping[str, bool | Support]
    ) -> CapabilityRegistry:
        """Non-mutating :meth:`refine`, returning a new registry."""
        clone = CapabilityRegistry(self._table)
        clone.refine(runtime, observed)
        return clone


#: Shared read-mostly registry. Adapters should refine a copy, not this.
DEFAULT_REGISTRY = CapabilityRegistry()


#: The cells ``modelpass verify`` can move on a per-install runtime, in the order
#: the drive observes them (ticket 1.10).
#:
#: Six, and the list is short on purpose. Each one is a thing a *single* short
#: exchange against the configured endpoint either does or does not do, watched
#: from the event stream modelpass already produces: text came back (``chat``),
#: events arrived while it was coming (``streaming``), the text itself arrived in
#: pieces (``incremental_text``), a usage report arrived with numbers in it
#: (``usage_tokens``), a declared tool was asked for and its result was used
#: (``tools_in_process``), and a json_schema request came back as an object
#: matching the schema (``structured_output``).
#:
#: **What is deliberately not here**, because the drive cannot see it: anything
#: about sessions or MCP (the shape has no such surface -- those stay checked
#: absences), ``interim_usage`` (one short call cannot tell "usage arrives while
#: the run is going" from "usage arrives at the end"), ``thinking`` (a server
#: that emits no reasoning on one prompt has not been shown unable to),
#: ``cache_breakpoints`` (a schema-level answer, already recorded), and the two
#: sampling cells (a server that accepted temperature on one call has not proved
#: it applied it). A cell a three-call drive cannot decide stays where the static
#: table left it, which is the whole reason this tuple is a closed set.
VERIFY_CELLS: tuple[Capability, ...] = (
    Capability.CHAT,
    Capability.STREAMING,
    Capability.INCREMENTAL_TEXT,
    Capability.USAGE_TOKENS,
    Capability.TOOLS_IN_PROCESS,
    Capability.STRUCTURED_OUTPUT,
)


@dataclass(frozen=True, slots=True)
class VerifiedCapabilities:
    """What one *install* of a runtime was observed to do (ticket 1.10).

    The persisted half of the ``refine()`` story, and the only place in modelpass
    where a capability verdict is a fact about a **connection** rather than about
    a runtime. That is not a loophole in the "cells move only with a recorded
    drive" rule -- it is that rule taken literally on the one runtime whose row
    cannot be a fact about a vendor: ``openai-compatible`` names an API *shape*,
    and what answers is whatever the connection's ``baseUrl`` points at. Two
    installs can legitimately disagree about every cell, so the drive is per
    install and so is its record.

    Written to the connection file under the additive key ``verifiedCapabilities``
    (store policy 0.1: a build that does not know a key preserves it verbatim and
    reports it, so an older modelpass reading a verified connection carries this
    through untouched rather than dropping the evidence).

    Two lists rather than one mapping, because that is what a person reading the
    file wants to see -- *this endpoint does these things and does not do these* --
    and because TOML renders it without a nested table per cell. ``checked_at``
    and ``models`` are the provenance: a verdict with no date is the thing this
    whole table exists to refuse, and the model list is what the probe actually
    saw the endpoint serving.
    """

    #: Capability names the drive observed working, sorted.
    supported: tuple[str, ...] = ()
    #: Capability names the drive asked for and did not get, sorted. A real
    #: finding, and the reason this is not "everything absent is unknown": an
    #: Ollama box that refuses ``response_format`` has *answered* the
    #: structured-output question, and recording the no is what keeps
    #: ``bridge.chat(schema=...)`` from spending a call to rediscover it.
    unsupported: tuple[str, ...] = ()
    #: When the drive ran, ISO-8601 in UTC. Prose elsewhere in this file carries
    #: its date in the sentence; here it is data, because a consumer may want to
    #: re-verify an endpoint that has not been checked this month.
    checked_at: str = ""
    #: Model ids the token-free probe reported, truncated by the adapter. The one
    #: piece of the answer that is about the endpoint rather than about modelpass.
    models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("supported", self.supported),
            ("unsupported", self.unsupported),
            ("models", self.models),
        ):
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise InvalidConnection(
                    f"verifiedCapabilities.{name} must be a list of strings"
                )
            cleaned = tuple(str(item).strip() for item in value if str(item).strip())
            object.__setattr__(
                self, name, tuple(sorted(set(cleaned))) if name != "models" else cleaned
            )
        both = set(self.supported) & set(self.unsupported)
        if both:
            raise InvalidConnection(
                "verifiedCapabilities lists "
                f"{', '.join(sorted(both))} as both supported and unsupported; a "
                "drive records one verdict per cell, so this file was edited into a "
                "state no verify run produces"
            )
        if not isinstance(self.checked_at, str):
            raise InvalidConnection("verifiedCapabilities.checkedAt must be a string")

    def __bool__(self) -> bool:
        """Whether this record says anything at all about any cell."""
        return bool(self.supported or self.unsupported)

    @property
    def observed(self) -> dict[str, bool]:
        """The mapping :meth:`CapabilityRegistry.refine` takes.

        Unknown names are *not* filtered here. ``refine`` ignores a capability it
        has never heard of, which is the behaviour a record written by a newer
        build needs: the cell it names is one this build has no question to ask
        about, and dropping the name on the way in would be this build editing
        evidence it does not understand.
        """
        out = dict.fromkeys(self.unsupported, False)
        out.update(dict.fromkeys(self.supported, True))
        return out

    def note(self) -> str:
        """One sentence for a receipt or a listing."""
        yes = ", ".join(self.supported) or "nothing"
        no = f"; not: {', '.join(self.unsupported)}" if self.unsupported else ""
        when = f" on {self.checked_at}" if self.checked_at else ""
        return (
            f"this endpoint was driven by 'modelpass verify'{when} and does: {yes}{no}. "
            "Cells this drive could not decide still read as the static table left "
            "them"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11). Holds no credential -- it is a list of verdicts."""
        return {
            "supported": list(self.supported),
            "unsupported": list(self.unsupported),
            "checked_at": self.checked_at,
            "models": list(self.models),
        }


@dataclass(frozen=True, slots=True)
class VerifyReport:
    """The result of one per-install capability drive (ticket 1.10).

    The adapter's answer to ``modelpass verify``: what was attempted, what came
    back, and the :class:`VerifiedCapabilities` that should be written to the
    connection if the caller wants to keep it. Separate from the record itself so
    a failed drive can still be *reported* -- ``ok=False`` with a problem naming
    the endpoint is the commonest outcome of pointing this at a box that is not
    running, and it must not write an empty record that reads as "verified, does
    nothing".
    """

    #: The verdicts, ready to persist.
    verified: VerifiedCapabilities = field(default_factory=lambda: VerifiedCapabilities())
    #: One line per step of the drive, in the order it ran.
    notes: tuple[str, ...] = ()
    #: Whether the endpoint could be reached and driven at all.
    ok: bool = True
    #: Why not, when not.
    problem: str | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form (D11)."""
        return {
            "verified": self.verified.to_dict(),
            "notes": list(self.notes),
            "ok": self.ok,
            "problem": self.problem,
        }


def runtime_auth_modes(
    runtime: Runtime, registry: CapabilityRegistry | None = None
) -> frozenset[AuthMode]:
    """Which auth modes a runtime can actually drive."""
    reg = registry or DEFAULT_REGISTRY
    modes = set()
    if reg.supports(runtime, Capability.SUBSCRIPTION_AUTH):
        modes.add(AuthMode.SUBSCRIPTION)
    if reg.supports(runtime, Capability.API_KEY_AUTH):
        modes.add(AuthMode.API_KEY)
    return frozenset(modes)
