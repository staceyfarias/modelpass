from __future__ import annotations

import pytest

from modelpass.adapters.openai import OpenAIAdapter
from modelpass.capabilities import (
    DEFAULT_REGISTRY,
    Capability,
    CapabilityRegistry,
    Support,
    runtime_auth_modes,
)
from modelpass.errors import CapabilityNotSupported
from modelpass.runtimes import AGENT_RUNTIMES, API_RUNTIMES, Runtime
from modelpass.types import AuthMode


def test_every_agent_runtime_can_chat():
    registry = CapabilityRegistry()
    for runtime in AGENT_RUNTIMES:
        assert registry.supports(runtime, Capability.CHAT)


def test_an_api_runtime_does_not_claim_chat_until_its_adapter_lands():
    """``unverified``, and that is the honest state until an adapter exists.

    This table's rule is that a cell moves only with evidence, and the
    consequence is deliberate and useful: because ``Bridge.chat`` requires
    ``chat``, a connection on a runtime with no adapter is refused with a
    capability error naming the unverified cell rather than getting as far as a
    missing import. ``anthropic-api`` moved in ticket 1.6, ``openai-api`` in 1.9
    and ``google-api`` in 1.11. ``openai-compatible`` has an adapter too (ticket
    1.10) and its cell stays ``unverified`` **by construction**: the runtime
    names a shape rather than a vendor, so what a cell says is a question about
    the endpoint behind the connection's baseUrl, answered per install by
    ``modelpass verify``.
    """
    registry = CapabilityRegistry()
    driven = (Runtime.ANTHROPIC_API, Runtime.OPENAI_API, Runtime.GOOGLE_API)
    for runtime in driven:
        assert registry.support(runtime, Capability.CHAT) is Support.SUPPORTED
    for runtime in API_RUNTIMES:
        if runtime in driven:
            continue
        assert registry.support(runtime, Capability.CHAT) is Support.UNVERIFIED
        assert "adapter ticket" in (registry.note(runtime, Capability.CHAT) or "")


def test_openai_reports_no_subagents_with_a_reason():
    registry = CapabilityRegistry()
    assert registry.support(Runtime.OPENAI_SDK, Capability.SUBAGENTS) is Support.UNSUPPORTED
    assert "MCP server" in (registry.note(Runtime.OPENAI_SDK, Capability.SUBAGENTS) or "")


def test_unverified_is_not_a_yes():
    registry = CapabilityRegistry()
    assert (
        registry.support(Runtime.OPENAI_SDK, Capability.GRACEFUL_CANCEL) is Support.UNVERIFIED
    )
    assert registry.supports(Runtime.OPENAI_SDK, Capability.GRACEFUL_CANCEL) is False
    with pytest.raises(CapabilityNotSupported, match="unverified"):
        registry.require(Runtime.OPENAI_SDK, Capability.GRACEFUL_CANCEL)


def test_the_phase_6_cells_report_the_asymmetry_rather_than_parity():
    """Still asymmetric after S7 -- the asymmetry just changed which way it runs.

    Anthropic does both. Codex does one or the other depending on the transport,
    and on 2026-08-31 flipping the default swapped which one a caller gets for
    free: ``tools_in_process`` became supported and ``mcp_servers`` became
    unsupported, in the same commit, on the same evidence. Both cells describe
    the default (D12), so both had to move with it.
    """
    registry = CapabilityRegistry()
    assert registry.supports(Runtime.ANTHROPIC_SDK, Capability.MCP_SERVERS)
    assert registry.supports(Runtime.ANTHROPIC_SDK, Capability.TOOLS_IN_PROCESS)

    assert registry.supports(Runtime.OPENAI_SDK, Capability.TOOLS_IN_PROCESS)
    # A *checked* absence on the DEFAULT transport, so unsupported rather than
    # unverified: 'codex app-server' has no driven per-run MCP declaration and
    # no equivalent of exec's enumerate-and-disable exclusivity.
    assert (
        registry.support(Runtime.OPENAI_SDK, Capability.MCP_SERVERS)
        is Support.UNSUPPORTED
    )


def test_the_openai_tools_note_names_the_transport_rather_than_blaming_the_vendor():
    """The retired claim must not survive in the note (S6, 2026-08-31).

    ``dynamicTools`` + ``item/tool/call`` were driven live, so "Codex is an MCP
    client only" is false about the vendor and was the sentence a caller read
    when the gate refused them. The cell stays ``unsupported`` because it
    describes the default transport; the note has to carry the difference, or
    the verdict reads as an absence that does not exist.
    """
    note = CapabilityRegistry().note(Runtime.OPENAI_SDK, Capability.TOOLS_IN_PROCESS) or ""
    assert "app-server" in note
    assert "2026-08-31" in note
    assert "experimental" in note
    assert "MCP client only" not in note


def test_the_registry_answers_the_default_and_the_default_flipped():
    """The table still describes the default; on 2026-08-31 the default moved.

    The rule did not change and neither did the two questions. What changed is
    the answer to the first one: ``registry.support()`` and ``find()`` report
    what a caller who selects nothing gets, and that caller now gets ``codex
    app-server``. ``Adapter.support_for()`` still answers the second -- *will
    THIS request work* -- and since S7 it answers by **narrowing** for a request
    that opts out into exec, which is the mirror of what it used to do.
    """
    registry = CapabilityRegistry()
    assert registry.supports(Runtime.OPENAI_SDK, Capability.TOOLS_IN_PROCESS)
    registry.require(Runtime.OPENAI_SDK, Capability.TOOLS_IN_PROCESS)
    note = registry.note(Runtime.OPENAI_SDK, Capability.TOOLS_IN_PROCESS) or ""
    assert "supported since 2026-08-31" in note
    # The cell moved because the default moved -- the note must say so, or a
    # reader cannot tell a re-verification from a new discovery.
    assert "DEFAULT transport moved" in note
    # And the narrowing direction is named where a caller on exec will read it.
    assert "exec opt-out" in note

    adapter = OpenAIAdapter(codex_bin="codex")
    assert adapter.support_for(Capability.TOOLS_IN_PROCESS, {}) is None
    assert (
        adapter.support_for(Capability.TOOLS_IN_PROCESS, {"transport": "exec"})
        is Support.UNSUPPORTED
    )


def test_the_one_cell_the_flip_cost_is_recorded_as_loudly_as_the_six_it_gained():
    """``mcp_servers`` moved DOWN on 2026-08-31, and burying that would be a lie.

    Per-run MCP servers are an exec capability with no driven equivalent on
    app-server, so the cell follows the default down -- a ``supported`` verdict
    that then failed at the transport is exactly the shape of lie
    ``sessions_list`` was kept honest about. The capability is intact one option
    away, and ``support_for()`` is what says so for a request that takes it.
    """
    registry = CapabilityRegistry()
    assert not registry.supports(Runtime.OPENAI_SDK, Capability.MCP_SERVERS)
    note = registry.note(Runtime.OPENAI_SDK, Capability.MCP_SERVERS) or ""
    assert "moved DOWN" in note
    assert "nothing about the vendor changed" in note
    assert "options={'transport': 'exec'}" in note

    adapter = OpenAIAdapter(codex_bin="codex")
    assert adapter.support_for(Capability.MCP_SERVERS, {}) is None
    assert (
        adapter.support_for(Capability.MCP_SERVERS, {"transport": "exec"})
        is Support.SUPPORTED
    )


def test_the_openai_mcp_caveat_is_recorded_where_a_caller_will_find_it():
    """`-c` merges with the user's config, so exclusivity cannot be promised."""
    note = DEFAULT_REGISTRY.note(Runtime.OPENAI_SDK, Capability.MCP_SERVERS) or ""
    assert "merges" in note


#: The cells added 2026-08-30 for the ChatSession work (D14-D19).
SESSION_CELLS = (
    Capability.INCREMENTAL_TEXT,
    Capability.EPHEMERAL_MULTI_TURN,
    Capability.SYSTEM_PROMPT_REPLACE,
    Capability.MIDCONVERSATION_SYSTEM,
    Capability.SESSIONS_LIST,
    Capability.TTL_CONTROL,
)


def test_incremental_text_is_split_out_because_streaming_hides_the_difference():
    """Both runtimes stream *events*; only one streams *text* (2026-08-30)."""
    registry = CapabilityRegistry()
    for runtime in (Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK):
        assert registry.supports(runtime, Capability.STREAMING)

    assert registry.supports(Runtime.ANTHROPIC_SDK, Capability.INCREMENTAL_TEXT)
    # Codex answered no until 2026-08-31, when the default became the transport
    # that carries item/agentMessage/delta. The split is still worth having:
    # STREAMING said yes on both throughout and never meant this.
    assert registry.supports(Runtime.OPENAI_SDK, Capability.INCREMENTAL_TEXT)


def test_the_codex_text_gap_is_recorded_as_a_transport_limit_not_a_modelpass_bug():
    """The gap closed by moving transports, which is what the note predicted.

    The reasoning has to survive the cell moving: it was never a mapper bug, and
    the exec half of the note still has to explain why ``map_codex_event()``'s
    started/completed-only branch is correct rather than lazy.
    """
    note = DEFAULT_REGISTRY.note(Runtime.OPENAI_SDK, Capability.INCREMENTAL_TEXT) or ""
    assert "not a mapper gap" in note
    # The deltas exist; only `exec` cannot see them.
    assert "agent_message_delta" in note
    assert "app-server" in note
    assert "supported since 2026-08-31" in note


def test_ephemeral_multi_turn_is_verified_on_anthropic_and_absent_on_codex():
    registry = CapabilityRegistry()
    assert registry.supports(Runtime.ANTHROPIC_SDK, Capability.EPHEMERAL_MULTI_TURN)
    # Codex needs the rollout file to continue a thread: a checked absence.
    assert (
        registry.support(Runtime.OPENAI_SDK, Capability.EPHEMERAL_MULTI_TURN)
        is Support.UNSUPPORTED
    )
    note = registry.note(Runtime.ANTHROPIC_SDK, Capability.EPHEMERAL_MULTI_TURN) or ""
    assert "CLAUDE_CODE_SKIP_PROMPT_HISTORY" in note
    # A live yes plus a control run that behaved differently, not an inference.
    assert "control run" in note


def test_codex_replaces_by_default_now_and_the_caveat_travels_with_the_yes():
    """D16's fact was true of ``codex exec``, and the default stopped being exec.

    ``baseInstructions`` replaces -- measured, and vendor-sourced. The cell moved
    on 2026-08-31 with the default. What the note must NOT let a reader conclude
    is that Codex stops behaving like a coding agent: the tool definitions stay,
    and the toolbelt is a separate switch. That caveat is the difference between
    honest product copy and an over-promise.
    """
    registry = CapabilityRegistry()
    assert registry.supports(Runtime.ANTHROPIC_SDK, Capability.SYSTEM_PROMPT_REPLACE)
    assert registry.supports(Runtime.OPENAI_SDK, Capability.SYSTEM_PROMPT_REPLACE)
    note = registry.note(Runtime.OPENAI_SDK, Capability.SYSTEM_PROMPT_REPLACE) or ""
    # The measurement, read for direction rather than magnitude.
    assert "3,543 tokens REMOVED" in note
    # The vendor's own source, so this does not rest on one A/B alone.
    assert "models.rs" in note
    # The caveat, and the separate switch that answers it.
    assert "toolbelt is a separate switch" in note
    # A worker still appends, on both transports.
    assert "WorkerSession keeps that layering on BOTH transports" in note


def test_midconversation_system_is_logged_so_it_is_not_re_derived():
    """An API feature neither agent runtime exposes; D17 has no escape hatch."""
    registry = CapabilityRegistry()
    for runtime in (Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK):
        assert (
            registry.support(runtime, Capability.MIDCONVERSATION_SYSTEM)
            is Support.UNSUPPORTED
        )
    note = registry.note(Runtime.ANTHROPIC_SDK, Capability.MIDCONVERSATION_SYSTEM) or ""
    assert "Messages API" in note


def test_sessions_list_on_codex_reports_the_transport_modelpass_actually_drives():
    """Which is ``codex app-server`` since 2026-08-31, so the cell says yes.

    The rule that kept this ``unsupported`` is the rule that moves it now:
    ``supports()`` is the caller's gate and must describe the transport a caller
    who selects nothing actually gets. The exec evidence stays in the note --
    the refusal there is still a refusal rather than an empty list.
    """
    registry = CapabilityRegistry()
    assert registry.supports(Runtime.ANTHROPIC_SDK, Capability.SESSIONS_LIST)
    assert registry.supports(Runtime.OPENAI_SDK, Capability.SESSIONS_LIST)
    assert Runtime.OPENAI_SDK in registry.runtimes_supporting(Capability.SESSIONS_LIST)

    note = registry.note(Runtime.OPENAI_SDK, Capability.SESSIONS_LIST) or ""
    assert "supported since 2026-08-31" in note
    assert "thread/list" in note
    # The correction that otherwise costs somebody an afternoon.
    assert "'limit', not" in note
    # The gotcha that otherwise looks like an empty account, kept for exec.
    assert "sourceKinds: ['exec']" in note


def test_ttl_control_records_the_version_gate_rather_than_promising_the_lever():
    """The vars exist, and whether they are read depends on the bundled CLI.

    The note has to carry the threshold and the fact that the deciding version
    is the one ``claude-agent-sdk`` bundles -- it moves with pip, not npm, and
    it crossed the gate within hours of being recorded below it.
    """
    registry = CapabilityRegistry()
    assert registry.supports(Runtime.ANTHROPIC_SDK, Capability.TTL_CONTROL)
    note = registry.note(Runtime.ANTHROPIC_SDK, Capability.TTL_CONTROL) or ""
    assert "CLAUDE_CODE_PROMPT_CACHE_TTL" in note
    assert "2.1.242" in note
    assert "bundles" in note
    assert "pip rather than npm" in note

    # Codex caching is automatic: there is no key to override.
    assert registry.support(Runtime.OPENAI_SDK, Capability.TTL_CONTROL) is Support.UNSUPPORTED


def test_every_session_cell_carries_a_note_on_both_v1_runtimes():
    """A cell without its evidence is a guess wearing a verdict's clothes."""
    registry = CapabilityRegistry()
    for runtime in (Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK):
        for capability in SESSION_CELLS:
            note = registry.note(runtime, capability)
            assert note, f"{runtime.value}/{capability.value} has no note"
            assert "2026-08-30" in note, (
                f"{runtime.value}/{capability.value} note carries no verification date"
            )


def test_google_runtimes_stay_unverified_for_the_session_cells():
    registry = CapabilityRegistry()
    for runtime in (Runtime.GOOGLE_CLI, Runtime.GOOGLE_SDK):
        for capability in SESSION_CELLS:
            assert registry.support(runtime, capability) is Support.UNVERIFIED


def test_every_runtime_has_a_value_for_every_capability():
    """A missing cell reads as 'unverified', which would hide a forgotten row."""
    registry = CapabilityRegistry()
    for runtime in Runtime:
        row = registry.row(runtime)
        missing = [c.value for c in Capability if c not in row]
        assert not missing, f"{runtime.value} has no entry for: {', '.join(missing)}"


def test_google_runtimes_stay_unverified_for_tools_rather_than_guessing():
    registry = CapabilityRegistry()
    for runtime in (Runtime.GOOGLE_CLI, Runtime.GOOGLE_SDK):
        for capability in (Capability.MCP_SERVERS, Capability.TOOLS_IN_PROCESS):
            assert registry.support(runtime, capability) is Support.UNVERIFIED


def test_runtimes_supporting_answers_the_caller_question():
    registry = CapabilityRegistry()
    assert registry.runtimes_supporting(Capability.SUBAGENTS) == (
        Runtime.ANTHROPIC_SDK,
        Runtime.GOOGLE_SDK,
    )


def test_google_runtimes_split_the_auth_modes():
    assert runtime_auth_modes(Runtime.GOOGLE_CLI) == frozenset({AuthMode.SUBSCRIPTION})
    assert runtime_auth_modes(Runtime.GOOGLE_SDK) == frozenset({AuthMode.API_KEY})
    assert runtime_auth_modes(Runtime.ANTHROPIC_SDK) == frozenset(
        {AuthMode.SUBSCRIPTION, AuthMode.API_KEY}
    )


def test_refinement_is_instance_local():
    registry = CapabilityRegistry()
    registry.refine(Runtime.OPENAI_SDK, {"graceful_cancel": True})
    assert registry.supports(Runtime.OPENAI_SDK, Capability.GRACEFUL_CANCEL)
    # The verified static table is untouched for everyone else.
    assert not DEFAULT_REGISTRY.supports(Runtime.OPENAI_SDK, Capability.GRACEFUL_CANCEL)
    assert not CapabilityRegistry().supports(Runtime.OPENAI_SDK, Capability.GRACEFUL_CANCEL)


def test_refined_returns_a_new_registry():
    # sessions_fork on openai-sdk is the standing 'unverified' cell. Two others
    # have stood here and stopped being unverified when they were driven --
    # structured_output (2026-08-17) and thinking (2026-08-31) -- which is
    # exactly the kind of drift this test should survive.
    base = CapabilityRegistry()
    refined = base.refined(Runtime.OPENAI_SDK, {"sessions_fork": Support.SUPPORTED})
    assert refined.supports(Runtime.OPENAI_SDK, Capability.SESSIONS_FORK)
    assert not base.supports(Runtime.OPENAI_SDK, Capability.SESSIONS_FORK)


def test_refinement_ignores_capabilities_we_have_never_heard_of():
    registry = CapabilityRegistry()
    registry.refine(Runtime.ANTHROPIC_SDK, {"time_travel": True})
    assert Capability.CHAT in registry.row(Runtime.ANTHROPIC_SDK)


def test_unknown_capability_names_are_a_programming_error():
    with pytest.raises(ValueError):
        CapabilityRegistry().support(Runtime.ANTHROPIC_SDK, "teleport")
