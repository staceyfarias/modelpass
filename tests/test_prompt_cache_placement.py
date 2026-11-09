"""The three prompt cache modes, and where each puts its breakpoints (2026-11-04).

1. **Default.** Caching is on with no settings at all. modelpass marks the system
   prompt, which is the minimum, and the repeated turns, which can be turned off,
   at a 5-minute lifetime that can be changed.
2. **Custom.** The caller places every breakpoint with
   :data:`PROMPT_CACHE_MARKER`, at the vendor's own lifetime unless told
   otherwise. A call with no markers is not cached.
3. **None.** Nothing is marked and nothing is left marked.
4. **Every other runtime removes the markers,** so a prompt can move between
   vendors unedited -- except that on ``anthropic-sdk`` (a Claude subscription) a
   breakpoint in the first user message moves the text before it into the system
   prompt, where Claude Code caches on its own (2026-11-09, section 4a).
5. **A call stays inside the vendor's rules:** four breakpoints, and no longer
   lifetime after a shorter one.
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
    connection_cache_plan,
    plan_prompt_cache,
)
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore, connection_from_dict, connection_to_dict
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode, CacheControl, Message, Role, SessionKind, TextBlock

M = PROMPT_CACHE_MARKER
ENV = {"ANTHROPIC_API_KEY": "sk-not-a-real-key", "OPENAI_API_KEY": "sk-not-a-real-key"}
FIVE = {"type": "ephemeral", "ttl": "5m"}
HOUR = {"type": "ephemeral", "ttl": "1h"}
AUTO = {"type": "ephemeral"}


def claude(**kwargs) -> Connection:
    return Connection(
        name="claude",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
        **kwargs,
    )


def custom(**kwargs) -> Connection:
    return claude(prompt_cache="custom", **kwargs)


def gpt(**kwargs) -> Connection:
    return Connection(
        name="gpt",
        runtime=Runtime.OPENAI_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:OPENAI_API_KEY"),
        **kwargs,
    )


def gemini(**kwargs) -> Connection:
    return Connection(
        name="gemini",
        runtime=Runtime.GOOGLE_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:GOOGLE_API_KEY"),
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


def run_through_bridge(connection: Connection, **kwargs) -> tuple[RunRequest, object]:
    adapter = FakeAdapter(runtime=connection.runtime)
    bridge = Bridge(
        registry=CapabilityRegistry(), adapters={connection.runtime: adapter}, env=ENV
    )
    events = list(bridge.chat(connection=connection, **kwargs))
    return adapter.requests[-1], events[0]


# --- 1. the default mode -----------------------------------------------------------


def test_a_connection_with_no_settings_caches_the_system_prompt_and_the_turns():
    wire = params(claude(), system("s"), user("a"), assistant("b"), user("c"))
    assert controls(wire) == [FIVE, FIVE]


@pytest.mark.parametrize("stated", [{}, {"prompt_cache": "default"}])
def test_the_default_mode_says_the_same_thing_stated_or_not(stated):
    rules = cache_rules(claude(**stated))
    assert (rules.system, rules.turns, rules.ttl) == (True, True, "5m")


def test_the_system_breakpoint_goes_on_the_end_of_the_system_prompt():
    wire = params(claude(), system("instructions"), user("hi"))
    assert wire["system"] == [
        {"type": "text", "text": "instructions", "cache_control": FIVE}
    ]


def test_a_call_with_no_system_prompt_and_one_turn_sends_no_breakpoint():
    assert controls(params(claude(), user("only this"))) == []


def test_the_turn_breakpoint_goes_on_the_message_before_the_current_turn():
    wire = params(claude(), user("a"), assistant("b"), user("c"))
    assert wire["messages"][1]["content"][-1]["cache_control"] == FIVE
    assert "cache_control" not in wire["messages"][2]["content"][-1]


def test_turns_can_be_turned_off():
    connection = claude(prompt_cache_turns=False)
    wire = params(connection, system("s"), user("a"), assistant("b"), user("c"))
    assert controls(wire) == [FIVE]
    assert wire["system"][0]["cache_control"] == FIVE


def test_the_system_breakpoint_is_the_minimum_of_the_default_mode():
    assert cache_rules(claude(prompt_cache_turns=False)).system is True


def test_a_marker_is_removed_in_the_default_mode():
    wire = params(claude(), system(f"stable{M}volatile"), user("hi"))
    assert wire["system"] == [
        {"type": "text", "text": "stablevolatile", "cache_control": FIVE}
    ]


def test_the_receipt_says_a_marker_was_removed_in_the_default_mode():
    _, notes = arrange_prompt_cache((system(f"a{M}b"), user("hi")), claude())
    assert notes == (
        "1 prompt cache marker(s) removed: modelpass places the breakpoints in this "
        "mode. promptCache 'custom' uses your markers",
    )


def test_a_callers_cache_control_still_reaches_the_vendor_in_the_default_mode():
    block = TextBlock("text", cache_control=CacheControl())
    wire = params(claude(), system("s"), user((block,)), assistant("b"), user("c"))
    assert controls(wire) == [FIVE, FIVE, FIVE]


def test_a_system_prompt_the_caller_marked_is_not_marked_again():
    block = TextBlock("text", cache_control=CacheControl())
    wire = params(claude(), system((block, TextBlock("volatile"))), user("hi"))
    assert [b.get("cache_control") for b in wire["system"]] == [FIVE, None]


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


def test_the_bridge_arranges_a_default_call():
    request, _ = run_through_bridge(claude(), message="hi", system_prompt="s")
    assert request.cache_breakpoints == 1


# --- 2. the custom mode ------------------------------------------------------------


def test_a_marker_ends_a_cacheable_prefix():
    wire = params(custom(), system(f"stable{M}volatile"), user("hi"))
    assert wire["system"] == [
        {"type": "text", "text": "stable", "cache_control": AUTO},
        {"type": "text", "text": "volatile"},
    ]


def test_custom_places_nothing_by_itself():
    wire = params(custom(), system(f"a{M}b"), user("c"), assistant("d"), user("e"))
    assert controls(wire) == [AUTO]


def test_a_custom_call_with_no_markers_is_not_cached():
    wire = params(custom(), system("s"), user("a"), assistant("b"), user("c"))
    assert controls(wire) == []


def test_the_receipt_says_a_custom_call_with_no_markers_is_not_cached():
    _, notes = arrange_prompt_cache((system("s"), user("hi")), custom())
    assert notes == (
        "nothing on this call is cached: promptCache is 'custom' and the call carries "
        "no markers",
    )


def test_custom_uses_the_vendors_own_lifetime_unless_told_otherwise():
    assert cache_rules(custom()).ttl is None
    assert plan_prompt_cache("custom", Runtime.ANTHROPIC_API, name="c").ttl is None


def test_custom_can_set_a_lifetime():
    wire = params(custom(prompt_cache_time="1h"), system(f"a{M}b"), user("hi"))
    assert controls(wire) == [HOUR]


def test_a_marker_in_a_user_message_works_too():
    wire = params(custom(), user(f"shared context{M}the question"))
    assert wire["messages"][0]["content"] == [
        {"type": "text", "text": "shared context", "cache_control": AUTO},
        {"type": "text", "text": "the question"},
    ]


def test_splitting_keeps_every_byte_but_the_marker():
    wire = params(custom(), system(f"one\n\n{M}\n\ntwo"), user("hi"))
    assert "".join(b["text"] for b in wire["system"]) == "one\n\n\n\ntwo"


def test_a_trailing_marker_caches_the_whole_prompt():
    wire = params(custom(), system(f"everything{M}"), user("hi"))
    assert wire["system"] == [
        {"type": "text", "text": "everything", "cache_control": AUTO}
    ]


def test_a_leading_marker_has_nothing_to_mark():
    wire = params(custom(), system(f"{M}text"), user("hi"))
    assert wire["system"] == [{"type": "text", "text": "text"}]


def test_a_block_keeps_its_own_breakpoint_when_a_marker_splits_it():
    block = TextBlock(f"a{M}b", cache_control=CacheControl(ttl="1h"))
    wire = params(custom(), system((block,)), user("hi"))
    assert wire["system"][0]["cache_control"] == AUTO
    assert wire["system"][1]["cache_control"] == HOUR


def test_a_callers_cache_control_counts_in_custom():
    block = TextBlock("text", cache_control=CacheControl())
    arranged, notes = arrange_prompt_cache((user((block,)),), custom())
    assert arranged[0].cache_breakpoints == 1
    assert notes == ()


def test_the_bridge_splits_a_marker_in_custom():
    request, _ = run_through_bridge(custom(), message="hi", system_prompt=f"c{M}d")
    assert request.cache_breakpoints == 1


# --- 3. turning caching off --------------------------------------------------------


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


def test_none_counts_what_it_removed():
    block = TextBlock("text", cache_control=CacheControl())
    _, notes = arrange_prompt_cache((user((block,)),), claude(prompt_cache="none"))
    assert notes == (
        "1 prompt cache marker(s) and breakpoint(s) removed: promptCache is 'none'",
    )


@pytest.mark.parametrize("build", [gpt, subscription])
def test_none_is_refused_where_caching_cannot_be_stopped(build):
    with pytest.raises(InvalidConnection, match="offers no way to stop it"):
        build(prompt_cache="none")


# --- 4. every other runtime --------------------------------------------------------


@pytest.mark.parametrize("connection", [gpt(), gemini()])
def test_a_marker_disappears_on_every_other_runtime(connection):
    messages = (system(f"stable{M}volatile"), user(f"a{M}b"))
    arranged, _ = arrange_prompt_cache(messages, connection)
    assert arranged[0].content == "stablevolatile"
    assert arranged[1].content == "ab"
    assert arranged[0].cache_breakpoints == 0


def test_a_marker_in_the_system_prompt_disappears_on_a_subscription():
    arranged, _ = arrange_prompt_cache((system(f"stable{M}volatile"), user("q")), subscription())
    assert arranged[0].content == "stablevolatile"
    assert arranged[1].content == "q"


# --- 4a. a subscription: the cached prefix moves into the system prompt -----------
#
# Claude Code places its own breakpoint at the end of the system prompt and takes
# none from the caller, and a stateless call's message is flattened into one string.
# So a document marked for caching in the user message is moved into the system
# prompt, and calls that share the document share a cached prefix (2026-11-09).


DOC = TextBlock("<document>the whole document</document>", cache_control=CacheControl())


def test_a_breakpoint_in_the_user_message_moves_the_text_before_it():
    arranged, notes = arrange_prompt_cache(
        (system("instructions"), user((DOC, TextBlock("<chunk>one</chunk>")))), subscription()
    )
    assert [m.role for m in arranged] == [Role.SYSTEM, Role.SYSTEM, Role.USER]
    assert arranged[1].content == (TextBlock("<document>the whole document</document>"),)
    assert arranged[2].content == (TextBlock("<chunk>one</chunk>"),)
    assert sum(m.cache_breakpoints for m in arranged) == 0
    assert notes == (
        "1 cache breakpoint(s) honoured by moving the text before them into the system "
        "prompt: anthropic-sdk places its own cache breakpoint at the end of the system "
        "prompt and takes none from the caller",
    )


def test_the_adapter_sends_the_document_as_the_system_prompt():
    from modelpass.adapters.anthropic import split_messages

    arranged, _ = arrange_prompt_cache(
        (system("instructions"), user((DOC, TextBlock("<chunk>one</chunk>")))), subscription()
    )
    assert split_messages(arranged) == (
        "instructions\n\n<document>the whole document</document>",
        "<chunk>one</chunk>",
    )


def test_two_calls_on_one_document_share_the_system_prompt():
    from modelpass.adapters.anthropic import split_messages

    def call(chunk):
        arranged, _ = arrange_prompt_cache(
            (system("instructions"), user((DOC, TextBlock(chunk)))), subscription()
        )
        return split_messages(arranged)

    (system_a, prompt_a), (system_b, prompt_b) = call("first"), call("second")
    assert system_a == system_b
    assert (prompt_a, prompt_b) == ("first", "second")


def test_a_marker_in_the_user_message_moves_too():
    arranged, notes = arrange_prompt_cache(
        (user(f"<document>d</document>{M}<chunk>c</chunk>"),), subscription()
    )
    assert arranged[0].role is Role.SYSTEM
    assert arranged[0].text == "<document>d</document>"
    assert arranged[1].text == "<chunk>c</chunk>"
    assert M not in "".join(m.text for m in arranged)
    assert len(notes) == 1 and "honoured by moving" in notes[0]


def test_every_byte_but_the_marker_is_kept():
    arranged, _ = arrange_prompt_cache((system("s"), user(f"  a\nb {M} c\n")), subscription())
    assert "".join(m.text for m in arranged[1:]) == "  a\nb  c\n"


def test_only_the_last_breakpoint_counts_and_everything_before_it_moves():
    first = TextBlock("one", cache_control=CacheControl())
    second = TextBlock("two", cache_control=CacheControl())
    arranged, notes = arrange_prompt_cache(
        (user((first, second, TextBlock("question"))),), subscription()
    )
    assert arranged[0].content == (TextBlock("one"), TextBlock("two"))
    assert arranged[1].content == (TextBlock("question"),)
    assert notes[0].startswith("2 cache breakpoint(s) honoured")


def test_nothing_moves_when_no_text_follows_the_breakpoint():
    arranged, notes = arrange_prompt_cache((user((DOC,)),), subscription())
    assert [m.role for m in arranged] == [Role.USER]
    assert arranged[0].content == (DOC,)
    assert notes == ()


def test_a_trailing_marker_is_removed_and_nothing_moves():
    arranged, notes = arrange_prompt_cache((user(f"all of it{M}"),), subscription())
    assert [m.role for m in arranged] == [Role.USER]
    assert arranged[0].text == "all of it"
    assert arranged[0].cache_breakpoints == 0
    assert notes == (
        "1 prompt cache marker(s) removed: anthropic-sdk takes no breakpoints, so the "
        "prompt was sent without them",
    )


def test_a_message_with_no_breakpoint_is_left_alone():
    messages = (system("s"), user("just a question"))
    assert arrange_prompt_cache(messages, subscription()) == (messages, ())


def test_only_the_first_user_message_moves():
    messages = (
        user("earlier question"),
        assistant("answer"),
        user((DOC, TextBlock("follow-up"))),
    )
    arranged, notes = arrange_prompt_cache(messages, subscription())
    assert [m.role for m in arranged] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert notes == ()


def test_a_system_marker_alongside_a_moved_one_is_still_reported_removed():
    arranged, notes = arrange_prompt_cache(
        (system(f"a{M}b"), user(f"doc{M}question")), subscription()
    )
    assert arranged[0].text == "ab"
    assert arranged[1].text == "doc"
    assert arranged[2].text == "question"
    assert len(notes) == 2
    assert notes[1].startswith("1 prompt cache marker(s) removed")


@pytest.mark.parametrize("build", [gpt, gemini])
def test_no_other_runtime_moves_anything(build):
    arranged, _ = arrange_prompt_cache((system("s"), user((DOC, TextBlock("q")))), build())
    assert [m.role for m in arranged] == [Role.SYSTEM, Role.USER]


def test_the_api_runtime_still_gets_the_breakpoint_where_the_caller_put_it():
    arranged, _ = arrange_prompt_cache((system("s"), user((DOC, TextBlock("q")))), claude())
    assert [m.role for m in arranged] == [Role.SYSTEM, Role.USER]
    assert arranged[1].content[0].cache_control is not None


def test_a_marker_in_a_block_disappears_without_leaving_a_breakpoint():
    arranged, _ = arrange_prompt_cache((system((TextBlock(f"a{M}b"),)),), gpt())
    assert arranged[0].content == (TextBlock("ab"),)


def test_the_receipt_says_a_marker_was_removed():
    _, notes = arrange_prompt_cache((system(f"a{M}b{M}c"),), gpt())
    assert notes == (
        "2 prompt cache marker(s) removed: openai-api takes no breakpoints, so the "
        "prompt was sent without them",
    )


@pytest.mark.parametrize("build", [gpt, subscription])
def test_the_default_mode_marks_nothing_where_breakpoints_do_not_travel(build):
    rules = cache_rules(build())
    assert (rules.system, rules.turns, rules.ttl) == (False, False, None)
    arranged, notes = arrange_prompt_cache((system("s"), user("hi")), build())
    assert arranged[0].cache_breakpoints == 0
    assert notes == ()


@pytest.mark.parametrize("build", [gpt, subscription])
def test_custom_can_move_to_a_runtime_that_caches_on_its_own(build):
    plan = connection_cache_plan(build(prompt_cache="custom"))
    assert plan.already_satisfied is True
    assert "your markers are removed" in plan.note


def test_turns_on_a_runtime_with_no_breakpoints_is_a_note_and_not_a_refusal():
    _, notes = arrange_prompt_cache((user("hi"),), gpt(prompt_cache_turns=False))
    assert notes == ("promptCacheTurns has no effect on openai-api: it takes no breakpoints",)


def test_no_adapter_ever_sees_a_marker():
    request, _ = run_through_bridge(
        gpt(), message=f"a{M}b", system_prompt=f"c{M}d", history=[]
    )
    assert M not in "".join(m.text for m in request.messages)
    assert request.cache_notes


# --- 5. the lifetime, and the vendor's rules ---------------------------------------


@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_the_time_reaches_the_wire(ttl):
    connection = claude(prompt_cache_time=ttl)
    wire = params(connection, system("s"), user("a"), assistant("b"), user("c"))
    assert controls(wire) == [{"type": "ephemeral", "ttl": ttl}] * 2


def test_a_lifetime_in_promptcache_means_the_default_mode_at_that_lifetime():
    wire = params(claude(prompt_cache="1h"), system("s"), user("a"), assistant("b"), user("c"))
    assert controls(wire) == [HOUR, HOUR]


def test_auto_leaves_the_lifetime_to_the_vendor():
    wire = params(claude(prompt_cache_time="auto"), system("s"), user("hi"))
    assert controls(wire) == [AUTO]


def test_auto_is_accepted_where_no_lifetime_can_be_named():
    assert subscription(prompt_cache_time="auto").prompt_cache_time == "auto"


def test_a_lifetime_is_still_refused_on_a_subscription():
    with pytest.raises(InvalidConnection, match="accepts only"):
        subscription(prompt_cache_time="1h")


def test_a_lifetime_a_runtime_does_not_take_is_refused():
    with pytest.raises(InvalidConnection, match="must be one of"):
        claude(prompt_cache_time="24h")


def test_a_lifetime_on_openai_api_is_its_own_word():
    assert connection_cache_plan(gpt(prompt_cache_time="24h")).ttl == "24h"


def test_a_time_on_a_runtime_with_no_established_caching_is_refused():
    with pytest.raises(InvalidConnection, match="has not established"):
        gemini(prompt_cache_time="auto")


def test_two_lifetimes_are_refused():
    with pytest.raises(InvalidConnection, match="already names a lifetime"):
        claude(prompt_cache="1h", prompt_cache_time="5m")


def test_a_callers_pin_beats_the_connection():
    block = TextBlock("text", cache_control=CacheControl(ttl="1h"))
    wire = params(claude(prompt_cache="5m"), user((block,)))
    assert controls(wire) == [HOUR]


def test_a_shorter_pin_caps_everything_after_it():
    pinned = TextBlock("text", cache_control=CacheControl(ttl="5m"))
    later = TextBlock("later", cache_control=CacheControl())
    connection = claude(prompt_cache="1h")
    wire = params(connection, system((pinned,)), user((later,)), assistant("b"), user("c"))
    assert controls(wire) == [FIVE, AUTO, AUTO]


def test_a_longer_pin_raises_everything_before_it():
    pinned = TextBlock("text", cache_control=CacheControl(ttl="1h"))
    wire = params(claude(), system("s"), user((pinned,)), assistant("b"), user("c"))
    assert controls(wire) == [HOUR, HOUR, FIVE]


def test_a_longer_pin_on_the_current_turn_raises_the_turn_breakpoint():
    pinned = TextBlock("text", cache_control=CacheControl(ttl="1h"))
    wire = params(claude(), user("a"), assistant("b"), user((pinned,)))
    assert controls(wire) == [HOUR, HOUR]


def test_a_call_never_passes_the_vendors_limit_of_four():
    wire = params(custom(), system(M.join("abcdefg")), user("hi"))
    assert len(controls(wire)) == 4


def test_a_turn_keeps_a_slot_when_a_caller_sends_too_many():
    blocks = tuple(TextBlock(c, cache_control=CacheControl()) for c in "abcde")
    messages = (user(blocks), assistant("b"), user("c"))
    arranged, notes = arrange_prompt_cache(messages, claude())
    assert arranged[0].cache_breakpoints == 3
    assert "2 cache breakpoint(s) dropped" in notes[0]
    assert len(controls(params(claude(), *messages))) == 4


# --- 6. the settings themselves ----------------------------------------------------


def test_turns_must_be_a_boolean():
    with pytest.raises(InvalidConnection, match="promptCacheTurns must be true or false"):
        claude(prompt_cache_turns="yes")


@pytest.mark.parametrize("mode", ["none", "custom"])
def test_asking_for_turns_beside_a_mode_that_marks_nothing_contradicts_itself(mode):
    with pytest.raises(InvalidConnection, match="mark nothing"):
        claude(prompt_cache=mode, prompt_cache_turns=True)


@pytest.mark.parametrize("mode", ["none", "custom"])
def test_turning_turns_off_beside_those_modes_is_fine(mode):
    assert claude(prompt_cache=mode, prompt_cache_turns=False).prompt_cache == mode


def test_a_time_beside_none_contradicts_itself():
    with pytest.raises(InvalidConnection, match="nothing to set"):
        claude(prompt_cache="none", prompt_cache_time="1h")


def test_the_store_round_trips_both_keys():
    connection = claude(prompt_cache_turns=False, prompt_cache_time="1h")
    data = connection_to_dict(connection)
    assert data["promptCacheTurns"] is False
    assert data["promptCacheTime"] == "1h"
    read = connection_from_dict("claude", data)
    assert read.prompt_cache_turns is False
    assert read.prompt_cache_time == "1h"


def test_a_connection_that_says_nothing_writes_neither_key():
    data = connection_to_dict(claude())
    assert "promptCacheTurns" not in data
    assert "promptCacheTime" not in data


def test_both_keys_are_known_keys():
    from modelpass.store import unknown_connection_keys

    keys = {"promptCacheTurns": False, "promptCacheTime": "1h"}
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
