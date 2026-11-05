"""Where cache breakpoints go, and how a prompt can say so (2026-11-04).

``promptCache`` says how long. Two more keys and one placeholder say where:

1. **A marker is a breakpoint where one can travel, and nothing anywhere else.**
   :data:`PROMPT_CACHE_MARKER` in a prompt ends a cacheable prefix on
   ``anthropic-api``. Every other runtime gets the same prompt with the marker
   removed, so a prompt can move between vendors unedited.
2. **``promptCachePlacement`` places breakpoints nobody wrote.** ``"system"``
   marks the end of the system prompt and ``"turns"`` the end of the re-sent
   turns. A connection on ``anthropic-api`` gets both unless it says otherwise,
   so nobody has to think about it, and the caller's own breakpoints replace
   them.
3. **``promptCacheMarkers = "ignore"`` lets the placement alone decide.**
4. **``promptCache = "none"`` turns caching off** where it can be turned off.
5. **The lifetime reaches the wire.** It goes on every breakpoint that does not
   pin its own, and a call stays inside the vendor's rules: four breakpoints, and
   no longer lifetime after a shorter one.
"""

import pytest

from modelpass.adapters.anthropic_api import AnthropicAPIAdapter, turn_breakpoint
from modelpass.bridge import Bridge, RunRequest
from modelpass.capabilities import CapabilityRegistry
from modelpass.connections import Connection, CredentialRef
from modelpass.errors import InvalidConnection
from modelpass.preflight import plan_launch
from modelpass.prompt_cache import (
    PROMPT_CACHE_MARKER,
    arrange_prompt_cache,
    cache_rules,
    plan_prompt_cache,
)
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore, connection_from_dict, connection_to_dict
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode, CacheControl, Message, Role, SessionKind, TextBlock

M = PROMPT_CACHE_MARKER
ENV = {"ANTHROPIC_API_KEY": "sk-not-a-real-key", "OPENAI_API_KEY": "sk-not-a-real-key"}


def claude(**kwargs) -> Connection:
    return Connection(
        name="claude",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
        **kwargs,
    )


def gpt(**kwargs) -> Connection:
    return Connection(
        name="gpt",
        runtime=Runtime.OPENAI_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:OPENAI_API_KEY"),
        **kwargs,
    )


def subscription(**kwargs) -> Connection:
    return Connection(
        name="sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        **kwargs,
    )


def system(content) -> Message:
    return Message(role=Role.SYSTEM, content=content)


def user(content) -> Message:
    return Message(role=Role.USER, content=content)


def assistant(content) -> Message:
    return Message(role=Role.ASSISTANT, content=content)


def params(connection: Connection, *messages: Message) -> dict:
    """One Messages request as the bridge and the adapter build it. No network."""
    arranged, _ = arrange_prompt_cache(messages, connection)
    request = RunRequest(
        connection=connection,
        messages=arranged,
        plan=plan_launch(connection, ENV),
        model="claude-opus-5-5",
        cache_turns=cache_rules(connection, written=messages).turns,
    )
    from modelpass.adapters.anthropic_api import conversation_param

    return AnthropicAPIAdapter(env=ENV).request_params(request, conversation_param(request))


def controls(wire: dict) -> list[dict]:
    """Every ``cache_control`` on a request, in the order the vendor reads them."""
    blocks = [
        *(wire.get("system") or ()),
        *(b for message in wire["messages"] for b in message["content"]),
    ]
    return [b["cache_control"] for b in blocks if "cache_control" in b]


# --- 1. a marker is a breakpoint where one can travel ------------------------------


def test_a_marker_ends_a_cacheable_prefix_on_anthropic_api():
    wire = params(claude(), system(f"stable{M}volatile"), user("hi"))
    assert wire["system"] == [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "volatile"},
    ]


def test_a_marker_counts_with_no_cache_settings_at_all():
    assert cache_rules(claude()).use_markers is True
    assert len(controls(params(claude(), system(f"a{M}b"), user("hi")))) == 1


def test_a_marker_in_a_user_message_works_too():
    wire = params(claude(), user(f"shared context{M}the question"))
    assert wire["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "shared context",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "the question"},
    ]


def test_splitting_keeps_every_byte_but_the_marker():
    wire = params(claude(), system(f"one\n\n{M}\n\ntwo"), user("hi"))
    assert "".join(b["text"] for b in wire["system"]) == "one\n\n\n\ntwo"


def test_a_trailing_marker_caches_the_whole_prompt():
    wire = params(claude(), system(f"everything{M}"), user("hi"))
    assert wire["system"] == [
        {"type": "text", "text": "everything", "cache_control": {"type": "ephemeral"}}
    ]


def test_a_leading_marker_has_nothing_to_mark():
    connection = claude(prompt_cache_placement=[])
    wire = params(connection, system(f"{M}text"), user("hi"))
    assert wire["system"] == [{"type": "text", "text": "text"}]


def test_a_marker_that_marks_nothing_leaves_the_default_placement_alone():
    wire = params(claude(), system(f"{M}text"), user("hi"))
    assert wire["system"] == [
        {"type": "text", "text": "text", "cache_control": {"type": "ephemeral"}}
    ]


def test_a_block_keeps_its_own_breakpoint_when_a_marker_splits_it():
    block = TextBlock(f"a{M}b", cache_control=CacheControl(ttl="1h"))
    wire = params(claude(), system((block,)), user("hi"))
    assert wire["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert wire["system"][1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.parametrize("connection", [gpt(), subscription()])
def test_a_marker_disappears_on_every_other_runtime(connection):
    messages = (system(f"stable{M}volatile"), user(f"a{M}b"))
    arranged, _ = arrange_prompt_cache(messages, connection)
    assert arranged[0].content == "stablevolatile"
    assert arranged[1].content == "ab"
    assert arranged[0].cache_breakpoints == 0


def test_a_marker_in_a_block_disappears_without_leaving_a_breakpoint():
    arranged, _ = arrange_prompt_cache((system((TextBlock(f"a{M}b"),)),), gpt())
    assert arranged[0].content == (TextBlock("ab"),)


def test_the_receipt_says_a_marker_was_removed():
    _, notes = arrange_prompt_cache((system(f"a{M}b{M}c"),), gpt())
    assert notes == (
        "2 prompt cache marker(s) removed: openai-api takes no breakpoints, so the "
        "prompt was sent without them",
    )


def run_through_bridge(connection: Connection, **kwargs) -> tuple[RunRequest, object]:
    adapter = FakeAdapter(runtime=connection.runtime)
    bridge = Bridge(
        registry=CapabilityRegistry(), adapters={connection.runtime: adapter}, env=ENV
    )
    events = list(bridge.chat(connection=connection, **kwargs))
    return adapter.requests[-1], events[0]


def test_no_adapter_ever_sees_a_marker():
    request, _ = run_through_bridge(
        gpt(), message=f"a{M}b", system_prompt=f"c{M}d", history=[]
    )
    assert M not in "".join(m.text for m in request.messages)
    assert request.cache_notes


def test_the_bridge_splits_a_marker_for_anthropic_api():
    request, _ = run_through_bridge(claude(), message="hi", system_prompt=f"c{M}d")
    assert request.cache_breakpoints == 1


# --- 2. placement places breakpoints nobody wrote ----------------------------------


def test_asking_for_caching_places_the_system_and_turn_breakpoints():
    rules = cache_rules(claude(prompt_cache="auto"))
    assert (rules.system, rules.turns) == (True, True)


def test_a_connection_that_states_nothing_caches_by_default():
    rules = cache_rules(claude())
    assert (rules.system, rules.turns) == (True, True)
    wire = params(claude(), system("s"), user("a"), assistant("b"), user("c"))
    assert controls(wire) == [{"type": "ephemeral"}] * 2


def test_a_default_one_shot_with_no_system_prompt_sends_no_breakpoint():
    assert controls(params(claude(), user("only this"))) == []


@pytest.mark.parametrize("build", [gpt, subscription])
def test_the_default_applies_only_where_breakpoints_travel(build):
    rules = cache_rules(build())
    assert (rules.system, rules.turns) == (False, False)
    arranged, notes = arrange_prompt_cache((system("s"), user("hi")), build())
    assert arranged[0].cache_breakpoints == 0
    assert notes == ()


def test_the_bridge_gives_a_default_call_both_breakpoints():
    request, _ = run_through_bridge(
        claude(),
        message="c",
        system_prompt="s",
        history=[user("a"), assistant("b")],
    )
    assert request.cache_breakpoints == 1
    assert request.cache_turns is True


# --- your own breakpoints replace the automatic ones --------------------------------


def test_a_marker_replaces_the_automatic_placement():
    wire = params(claude(), system(f"stable{M}volatile"), user("a"), assistant("b"), user("c"))
    assert controls(wire) == [{"type": "ephemeral"}]
    assert "cache_control" not in wire["messages"][1]["content"][-1]


def test_a_marker_in_a_message_stops_the_system_breakpoint_too():
    wire = params(claude(), system("instructions"), user(f"shared{M}question"))
    assert wire["system"] == [{"type": "text", "text": "instructions"}]
    assert len(controls(wire)) == 1


def test_a_callers_cache_control_replaces_the_automatic_placement():
    block = TextBlock("text", cache_control=CacheControl())
    wire = params(claude(), system("s"), user((block,)), assistant("b"), user("c"))
    assert controls(wire) == [{"type": "ephemeral"}]


def test_a_marker_replaces_the_placement_when_a_lifetime_is_stated():
    connection = claude(prompt_cache="1h")
    wire = params(connection, system(f"a{M}b"), user("c"), assistant("d"), user("e"))
    assert controls(wire) == [{"type": "ephemeral", "ttl": "1h"}]


def test_a_stated_placement_keeps_its_breakpoints_beside_a_marker():
    connection = claude(prompt_cache_placement=["turns"])
    wire = params(connection, system(f"a{M}b"), user("c"), assistant("d"), user("e"))
    assert len(controls(wire)) == 2


def test_the_receipt_says_the_markers_replaced_the_placement():
    _, notes = arrange_prompt_cache((system(f"a{M}b"), user("hi")), claude())
    assert notes == (
        "modelpass placed no cache breakpoints of its own: this call carries yours. "
        "Set promptCachePlacement to have both",
    )


def test_a_call_with_no_markers_of_its_own_gets_no_such_note():
    _, notes = arrange_prompt_cache((system("s"), user("hi")), claude())
    assert notes == ()


def test_the_bridge_leaves_the_turn_alone_when_a_marker_replaces_it():
    request, _ = run_through_bridge(
        claude(),
        message="c",
        system_prompt=f"a{M}b",
        history=[user("a"), assistant("b")],
    )
    assert request.cache_turns is False


# --- where the breakpoints go, one at a time ----------------------------------------


def test_system_placement_marks_the_end_of_the_system_prompt():
    connection = claude(prompt_cache="auto", prompt_cache_placement=["system"])
    wire = params(connection, system("instructions"), user("hi"))
    assert wire["system"] == [
        {
            "type": "text",
            "text": "instructions",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_system_placement_gives_way_to_a_marker_the_caller_wrote():
    connection = claude(prompt_cache="auto", prompt_cache_placement=["system"])
    wire = params(connection, system(f"stable{M}volatile"), user("hi"))
    assert [b.get("cache_control") for b in wire["system"]] == [{"type": "ephemeral"}, None]


def test_system_placement_with_no_system_prompt_marks_nothing():
    connection = claude(prompt_cache="auto", prompt_cache_placement=["system"])
    assert controls(params(connection, user("hi"))) == []


def test_turn_placement_marks_the_message_before_the_current_turn():
    connection = claude(prompt_cache="auto", prompt_cache_placement=["turns"])
    wire = params(connection, user("a"), assistant("b"), user("c"))
    assert wire["messages"][1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in wire["messages"][2]["content"][-1]


def test_a_one_shot_call_pays_no_write_for_its_question():
    connection = claude(prompt_cache="auto", prompt_cache_placement=["turns"])
    assert controls(params(connection, user("only this"))) == []


def test_the_turn_breakpoint_skips_a_thinking_block():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t", "name": "n", "input": {}},
                {"type": "thinking", "thinking": "", "signature": "s"},
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "next"}]},
    ]
    marked = turn_breakpoint(messages, None, None)
    assert "cache_control" in marked[0]["content"][0]
    assert "cache_control" not in marked[0]["content"][1]


def test_the_turn_breakpoint_never_touches_the_list_a_tool_loop_keeps():
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
    ]
    turn_breakpoint(messages, None, "1h")
    assert "cache_control" not in messages[0]["content"][0]


def test_an_empty_placement_leaves_only_the_callers_own():
    connection = claude(prompt_cache="auto", prompt_cache_placement=[])
    wire = params(connection, system(f"a{M}b"), user("c"), assistant("d"), user("e"))
    assert len(controls(wire)) == 1


def test_placement_alone_is_enough_to_cache():
    connection = claude(prompt_cache_placement=["system"])
    assert len(controls(params(connection, system("s"), user("hi")))) == 1


def test_placement_on_a_runtime_with_no_breakpoints_is_a_note_and_not_a_refusal():
    connection = gpt(prompt_cache_placement=["system", "turns"])
    _, notes = arrange_prompt_cache((system("s"), user("hi")), connection)
    assert notes == ("promptCachePlacement has no effect on openai-api: it takes no breakpoints",)


# --- 3. ignoring the caller's markers ----------------------------------------------


def test_ignore_removes_markers_and_places_automatically():
    connection = claude(prompt_cache="auto", prompt_cache_markers="ignore")
    wire = params(connection, system(f"stable{M}volatile"), user("hi"))
    assert wire["system"] == [
        {
            "type": "text",
            "text": "stablevolatile",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_ignore_removes_a_callers_cache_control_as_well():
    connection = claude(prompt_cache_placement=[], prompt_cache_markers="ignore")
    block = TextBlock("text", cache_control=CacheControl())
    arranged, notes = arrange_prompt_cache((user((block,)),), connection)
    assert arranged[0].cache_breakpoints == 0
    assert notes == (
        "1 prompt cache marker(s) and breakpoint(s) removed: promptCacheMarkers is 'ignore'",
    )


# --- 4. turning caching off --------------------------------------------------------


def test_none_sends_no_cache_control_at_all():
    connection = claude(prompt_cache="none")
    block = TextBlock("text", cache_control=CacheControl())
    wire = params(connection, system(f"a{M}b"), user((block,)), assistant("c"), user("d"))
    assert controls(wire) == []
    assert wire["system"][0]["text"] == "ab"


def test_none_says_so_on_the_receipt():
    plan = plan_prompt_cache("none", Runtime.ANTHROPIC_API, name="c")
    assert plan.off is True
    assert plan.note.startswith("prompt caching off")


@pytest.mark.parametrize("build", [gpt, subscription])
def test_none_is_refused_where_caching_cannot_be_stopped(build):
    with pytest.raises(InvalidConnection, match="offers no way to stop it"):
        build(prompt_cache="none")


def test_none_beside_a_placement_contradicts_itself():
    with pytest.raises(InvalidConnection, match="turns caching off"):
        claude(prompt_cache="none", prompt_cache_placement=["system"])


def test_none_beside_an_empty_placement_is_fine():
    assert claude(prompt_cache="none", prompt_cache_placement=[]).prompt_cache == "none"


# --- 5. the lifetime, and the vendor's rules ---------------------------------------


@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_the_lifetime_reaches_the_wire(ttl):
    wire = params(claude(prompt_cache=ttl), system("s"), user("a"), assistant("b"), user("c"))
    assert controls(wire) == [{"type": "ephemeral", "ttl": ttl}] * 2


@pytest.mark.parametrize("word", ["auto", "default"])
def test_auto_names_no_lifetime(word):
    wire = params(claude(prompt_cache=word), system("s"), user("hi"))
    assert controls(wire) == [{"type": "ephemeral"}]


def test_auto_means_the_same_as_default_everywhere():
    assert plan_prompt_cache("auto", Runtime.OPENAI_API, name="c").ttl is None
    assert plan_prompt_cache("auto", Runtime.ANTHROPIC_SDK, name="c").already_satisfied


def test_a_lifetime_is_still_refused_on_a_subscription():
    with pytest.raises(InvalidConnection, match="accepts only"):
        subscription(prompt_cache="1h")


def test_a_lifetime_reaches_a_callers_unpinned_breakpoint():
    block = TextBlock("text", cache_control=CacheControl())
    wire = params(claude(prompt_cache="1h", prompt_cache_placement=[]), user((block,)))
    assert controls(wire) == [{"type": "ephemeral", "ttl": "1h"}]


def test_a_callers_pin_beats_the_connection():
    block = TextBlock("text", cache_control=CacheControl(ttl="1h"))
    wire = params(claude(prompt_cache="5m", prompt_cache_placement=[]), user((block,)))
    assert controls(wire) == [{"type": "ephemeral", "ttl": "1h"}]


def test_a_shorter_pin_caps_everything_after_it():
    pinned = TextBlock("text", cache_control=CacheControl(ttl="5m"))
    later = TextBlock("later", cache_control=CacheControl())
    connection = claude(prompt_cache="1h", prompt_cache_placement=["turns"])
    wire = params(connection, system((pinned,)), user((later,)), assistant("b"), user("c"))
    assert controls(wire) == [
        {"type": "ephemeral", "ttl": "5m"},
        {"type": "ephemeral"},
        {"type": "ephemeral"},
    ]


def test_a_call_never_passes_the_vendors_limit_of_four():
    prompt = M.join("abcdefg")
    wire = params(claude(), system(prompt), user("hi"))
    assert len(controls(wire)) == 4


def test_a_turn_keeps_a_slot_when_there_are_too_many_markers():
    prompt = M.join("abcdefg")
    connection = claude(prompt_cache_placement=["turns"])
    messages = (system(prompt), user("a"), assistant("b"), user("c"))
    arranged, notes = arrange_prompt_cache(messages, connection)
    assert arranged[0].cache_breakpoints == 3
    assert "3 cache breakpoint(s) dropped" in notes[0]
    assert len(controls(params(connection, *messages))) == 4


def test_breakpoints_a_caller_sent_with_no_settings_are_left_for_the_vendor_to_judge():
    blocks = tuple(TextBlock(c, cache_control=CacheControl()) for c in "abcde")
    arranged, _ = arrange_prompt_cache((user(blocks),), claude())
    assert arranged[0].cache_breakpoints == 5


# --- 6. the settings themselves ----------------------------------------------------


@pytest.mark.parametrize(
    ("placement", "shown"),
    [(["nowhere"], "'nowhere'"), ("system", "str")],
)
def test_a_bad_placement_is_refused_when_the_connection_is_built(placement, shown):
    with pytest.raises(InvalidConnection, match=shown):
        claude(prompt_cache_placement=placement)


def test_a_bad_marker_mode_is_refused_when_the_connection_is_built():
    with pytest.raises(InvalidConnection, match="promptCacheMarkers"):
        claude(prompt_cache_markers="sometimes")


def test_the_store_round_trips_both_keys():
    connection = claude(
        prompt_cache="1h",
        prompt_cache_placement=["turns"],
        prompt_cache_markers="ignore",
    )
    data = connection_to_dict(connection)
    assert data["promptCachePlacement"] == ["turns"]
    assert data["promptCacheMarkers"] == "ignore"
    read = connection_from_dict("claude", data)
    assert read.prompt_cache_placement == ("turns",)
    assert read.prompt_cache_markers == "ignore"


def test_a_connection_that_says_nothing_writes_neither_key():
    data = connection_to_dict(claude())
    assert "promptCachePlacement" not in data
    assert "promptCacheMarkers" not in data


def test_both_keys_are_known_keys():
    from modelpass.store import unknown_connection_keys

    keys = {"promptCachePlacement": [], "promptCacheMarkers": "use"}
    assert unknown_connection_keys(keys) == ()


# --- 7. sessions -------------------------------------------------------------------


def test_a_session_prompt_loses_its_markers(tmp_path):
    connection = subscription()
    bridge = Bridge(
        store=ConnectionStore(tmp_path / "c.toml"),
        registry=CapabilityRegistry(),
        adapters={connection.runtime: FakeAdapter(runtime=connection.runtime)},
        env=ENV,
    )
    request = bridge._build_session_request(
        connection,
        kind=SessionKind.CHAT,
        project_folder="",
        model=None,
        system_prompt=f"a{M}b",
        persist=False,
        tools=(),
        mcp_servers={},
        options={},
    )
    assert request.system_prompt == "ab"
    assert request.cache_notes
