"""Connection groups: membership, and a group as something a run can be sent to.

Three properties this file holds, in the order they matter:

1. **``default`` is the name for the ungrouped ones, not "everything".** A
   connection in ``fast`` is not also in ``default``. The moment it were, the
   group would carry no information and ``group:default`` would mean "any
   connection at all", which is the opposite of a route somebody chose.
2. **A group is derived, never stored.** There is no ``[groups]`` table: a group
   exists while some connection claims it. So it cannot drift out of step with
   its membership, and moving the last member out is how it goes away.
3. **Resolution is visible before the run.** ``select()`` hands back the member
   that would run *and* everyone it walked past, because name order is a real
   rule that nobody chose and presenting only the winner would dress an
   arbitrary pick up as a decision.
"""

from __future__ import annotations

import asyncio
import io
import tomllib

import pytest

from modelpass.bridge import Bridge
from modelpass.capabilities import CapabilityRegistry
from modelpass.cli import main
from modelpass.connections import (
    DEFAULT_GROUP,
    Connection,
    Guards,
    QuotaAction,
    QuotaPolicy,
    groups_of,
)
from modelpass.errors import GroupUnavailable, InvalidConnection, NoSuchGroup
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode

MESSAGE = "hello"


def make(name: str, groups: tuple[str, ...] = (), *, enabled: bool = True) -> Connection:
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        groups=groups,
        enabled=enabled,
    )


def bridge_for(store, *connections: Connection) -> Bridge:
    for connection in connections:
        store.add(connection, overwrite=True)
    return Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter()},
        env={},
    )


# --- membership ------------------------------------------------------------------


def test_a_connection_naming_no_group_is_in_default():
    assert make("a").group_names == (DEFAULT_GROUP,)
    assert make("a").in_group(DEFAULT_GROUP) is True


def test_a_grouped_connection_is_not_also_in_default():
    """The whole reason `default` carries any information at all."""
    connection = make("a", ("fast",))
    assert connection.group_names == ("fast",)
    assert connection.in_group(DEFAULT_GROUP) is False


def test_default_can_be_named_explicitly_alongside_another_group():
    connection = make("a", ("fast", DEFAULT_GROUP))
    assert connection.in_group("fast") is True
    assert connection.in_group(DEFAULT_GROUP) is True


def test_declared_order_is_kept_rather_than_sorted():
    """Sorting a list somebody typed would make a later preference order
    impossible to introduce without changing what an existing file means."""
    assert make("a", ("zebra", "alpha")).groups == ("zebra", "alpha")


# --- validation ------------------------------------------------------------------


def test_a_group_name_follows_the_connection_name_rule():
    with pytest.raises(InvalidConnection, match="group 'not a group'"):
        make("a", ("not a group",))


def test_a_group_listed_twice_is_refused_rather_than_deduplicated():
    """The second entry is very often the one that was meant to be different."""
    with pytest.raises(InvalidConnection, match="listed twice"):
        make("a", ("fast", "fast"))


def test_a_bare_string_in_the_file_is_refused(store):
    """TOML takes the string happily and Python iterates it into four
    one-character groups, which surfaces later as a group nobody can find."""
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        "version = 1\n\n[connections.x]\nruntime = 'anthropic-sdk'\n"
        "authMode = 'subscription'\ngroups = 'fast'\n",
        encoding="utf-8",
    )
    with pytest.raises(InvalidConnection, match="groups must be an array"):
        store.load()


# --- the file --------------------------------------------------------------------


def test_groups_round_trip_through_the_store(store):
    store.add(make("a", ("fast", "cheap")))
    assert store.get("a").groups == ("fast", "cheap")
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert raw["connections"]["a"]["groups"] == ["fast", "cheap"]


def test_the_default_group_is_never_written(store):
    """`groups = ["default"]` on every line is a default spelled out, and it
    would also hide the one connection deliberately put in `default` beside
    another group."""
    store.add(make("a"))
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert "groups" not in raw["connections"]["a"]


# --- derivation ------------------------------------------------------------------


def test_groups_are_derived_from_membership():
    found = {g.name: g for g in groups_of([make("a"), make("b", ("fast",))])}
    assert set(found) == {DEFAULT_GROUP, "fast"}
    assert found["fast"].members == ("b",)
    assert found[DEFAULT_GROUP].members == ("a",)


def test_default_does_not_exist_when_everything_has_been_filed(store):
    """It is the name for the ungrouped ones, so a store with none of those
    genuinely has no `default` -- saying otherwise would invent a member."""
    bridge = bridge_for(store, make("a", ("fast",)), make("b", ("cheap",)))
    assert [g.name for g in bridge.groups()] == ["cheap", "fast"]
    with pytest.raises(NoSuchGroup, match="cheap, fast"):
        bridge.group(DEFAULT_GROUP)


def test_moving_the_last_member_out_removes_the_group(store):
    bridge = bridge_for(store, make("a", ("fast",)))
    assert [g.name for g in bridge.groups()] == ["fast"]
    bridge.manage.set_groups("a", ("cheap",))
    assert [g.name for g in bridge.groups()] == ["cheap"]


def test_a_disabled_member_is_still_a_member(store):
    """Looking is not running: `enabled = false` takes a connection out of
    service, not out of the group it is in."""
    bridge = bridge_for(store, make("a", ("fast",), enabled=False))
    group = bridge.group("fast")
    assert group.members == ("a",)
    assert group.enabled == ()
    assert group.available is False


# --- selection -------------------------------------------------------------------


def test_select_skips_the_disabled_member_and_says_so(store):
    bridge = bridge_for(store, make("a-gw", ("cheap",), enabled=False), make("local", ("cheap",)))
    selection = bridge.select("cheap")
    assert selection.chosen.name == "local"
    assert selection.passed_over == (("a-gw", "disabled"),)


def test_select_reports_the_runners_up_rather_than_only_the_winner(store):
    bridge = bridge_for(store, make("a", ("cheap",)), make("b", ("cheap",)))
    selection = bridge.select("cheap")
    assert selection.chosen.name == "a"
    assert selection.passed_over == (("b", "a comes first in name order"),)


def test_select_spends_nothing(store):
    bridge = bridge_for(store, make("a", ("cheap",)))
    adapter = bridge.adapter_for(Runtime.ANTHROPIC_SDK)
    bridge.select("cheap")
    assert adapter.requests == []
    assert adapter.preflights == []


def test_an_unknown_group_and_an_all_disabled_group_are_different_errors(store):
    """Two findings that send a user to two different places."""
    bridge = bridge_for(store, make("a", ("cheap",), enabled=False))
    with pytest.raises(NoSuchGroup):
        bridge.select("nope")
    with pytest.raises(GroupUnavailable, match="every connection in group 'cheap'"):
        bridge.select("cheap")


# --- addressing a run ------------------------------------------------------------


def test_a_group_reference_routes_a_run_to_a_member(store):
    bridge = bridge_for(store, make("a-gw", ("cheap",), enabled=False), make("local", ("cheap",)))
    events = list(bridge.chat(connection="group:cheap", message=MESSAGE))
    # A group is how the run was *addressed*; what it was billed to is a
    # connection, and every record says so.
    assert events[-1].connection == "local"


def test_ask_through_a_group_names_the_connection_it_reached(store):
    bridge = bridge_for(store, make("local", ("cheap",)))
    answer = bridge.ask("group:cheap", MESSAGE)
    assert answer.receipt.connection == "local"


def test_a_group_with_no_enabled_member_refuses_before_anything_runs(store):
    bridge = bridge_for(store, make("a", ("cheap",), enabled=False))
    adapter = bridge.adapter_for(Runtime.ANTHROPIC_SDK)
    with pytest.raises(GroupUnavailable):
        list(bridge.chat(connection="group:cheap", message=MESSAGE))
    assert adapter.requests == []
    assert adapter.preflights == []


def test_a_connection_named_like_a_group_is_not_reachable_by_a_bare_name(store):
    """The prefix is the spelling because a bare name that is both would route
    on a rule the caller never saw. Connection names cannot hold a colon, so
    the two can never collide."""
    bridge = bridge_for(store, make("cheap"))
    assert bridge.select(DEFAULT_GROUP).chosen.name == "cheap"
    with pytest.raises(NoSuchGroup):
        bridge.select("cheap")


# --- find ------------------------------------------------------------------------


def test_find_filters_by_group_without_hiding_the_disabled(store):
    """`find` is how a caller inspects its store; omitting a switched-off
    connection would make it look deleted."""
    bridge = bridge_for(store, make("a", ("cheap",), enabled=False), make("b", ("cheap",)))
    assert [c.name for c in bridge.find(group="cheap")] == ["a", "b"]
    assert [c.name for c in bridge.find(group="cheap", enabled=True)] == ["b"]


def test_find_on_a_group_nothing_claims_returns_nothing(store):
    bridge = bridge_for(store, make("a"))
    assert bridge.find(group="nope") == ()


# --- the write verb --------------------------------------------------------------


def test_set_groups_replaces_rather_than_merges(store):
    """"Put this in cheap" and "put this in cheap only" are different
    instructions that an add-one verb cannot tell apart."""
    bridge = bridge_for(store, make("a", ("fast", "work")))
    result = bridge.manage.set_groups("a", ("cheap",))
    assert result.groups == ("cheap",)
    assert result.previous == ("fast", "work")
    assert result.changed is True
    assert store.get("a").groups == ("cheap",)


def test_set_groups_with_nothing_returns_a_connection_to_default(store):
    bridge = bridge_for(store, make("a", ("fast",)))
    result = bridge.manage.set_groups("a", ())
    assert result.groups == ()
    assert result.effective == (DEFAULT_GROUP,)
    assert store.get("a").group_names == (DEFAULT_GROUP,)


def test_set_groups_refuses_a_bad_name_before_writing(store):
    bridge = bridge_for(store, make("a", ("fast",)))
    with pytest.raises(InvalidConnection):
        bridge.manage.set_groups("a", ("not a group",))
    assert store.get("a").groups == ("fast",)


# --- the command line ------------------------------------------------------------


def test_cli_list_names_the_groups(store):
    bridge = bridge_for(store, make("a", ("fast",)), make("b"))
    out = io.StringIO()
    assert main(["list"], bridge=bridge, out=out) == 0
    text = out.getvalue()
    assert "groups    fast" in text
    assert f"groups    {DEFAULT_GROUP}" in text


def test_cli_groups_reports_the_member_a_run_would_reach(store):
    bridge = bridge_for(store, make("a-gw", ("cheap",), enabled=False), make("local", ("cheap",)))
    out = io.StringIO()
    assert main(["groups"], bridge=bridge, out=out) == 0
    text = out.getvalue()
    assert "group:cheap resolves to local" in text
    assert "a-gw passed over: disabled" in text


def test_cli_groups_says_when_a_group_would_be_refused(store):
    """A group with members and no runnable one reads, in a plain listing,
    exactly like a group that works."""
    bridge = bridge_for(store, make("a", ("cheap",), enabled=False))
    out = io.StringIO()
    assert main(["groups", "cheap"], bridge=bridge, out=out) == 0
    assert "would be REFUSED" in out.getvalue()


def test_cli_list_on_an_empty_group_says_what_groups_exist(store):
    """The user asked to *look*, so "nothing is in it" is the answer -- and
    naming the real groups saves the next command."""
    bridge = bridge_for(store, make("a", ("fast",)))
    out = io.StringIO()
    assert main(["list", "--group", "nope"], bridge=bridge, out=out) == 0
    assert "Configured groups: fast" in out.getvalue()


def test_cli_set_groups_warns_that_a_group_has_gone(store):
    """A user told "moved" and not "and fast no longer exists" goes looking."""
    bridge = bridge_for(store, make("a", ("fast",)))
    out = io.StringIO()
    assert main(["set-groups", "a", "cheap"], bridge=bridge, out=out) == 0
    assert "fast now have no members and no longer exist" in out.getvalue()


def test_cli_disable_warns_when_it_strands_a_group(store):
    """Where disabling has a consequence beyond this connection, say it here
    rather than leaving it to be discovered by a refusal."""
    bridge = bridge_for(store, make("a", ("cheap",)))
    out = io.StringIO()
    assert main(["disable", "a"], bridge=bridge, out=out) == 0
    text = out.getvalue()
    assert "a is now disabled" in text
    assert "cheap now have no enabled member" in text


def test_cli_enable_is_idempotent_and_writes_nothing(store):
    bridge = bridge_for(store, make("a"))
    out = io.StringIO()
    assert main(["enable", "a"], bridge=bridge, out=out) == 0
    assert "already enabled; nothing written" in out.getvalue()


def test_cli_disable_then_enable_round_trips(store):
    bridge = bridge_for(store, make("a"))
    assert main(["disable", "a"], bridge=bridge, out=io.StringIO()) == 0
    assert store.get("a").enabled is False
    assert main(["enable", "a"], bridge=bridge, out=io.StringIO()) == 0
    assert store.get("a").enabled is True


# --- every other door that takes a connection name --------------------------------


def test_the_entry_points_that_spend_nothing_take_a_group(store):
    """`_resolve` is the one place a name becomes a connection, so these come
    along for free -- but a behaviour with no test here is not a behaviour of
    this library, and the contract doc now claims this one."""
    bridge = bridge_for(store, make("a-gw", ("cheap",), enabled=False), make("local", ("cheap",)))
    assert bridge.preflight("group:cheap").connection == "local"
    assert bridge.validate("group:cheap").connection == "local"
    assert bridge.plan("group:cheap") is not None


def test_a_session_constructor_takes_a_group(store):
    from modelpass.testing import FakeSessionAdapter

    store.add(make("a-gw", ("cheap",), enabled=False), overwrite=True)
    store.add(make("local", ("cheap",)), overwrite=True)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: FakeSessionAdapter()},
        env={},
    )
    session = bridge.new_chat(connection="group:cheap")
    # The session belongs to the connection the group reached, not to the group:
    # what holds the conversation and what pays for it are the same thing.
    assert session.connection == "local"


# --- the one door that refuses a group --------------------------------------------


def test_a_failover_target_may_not_be_a_group(store):
    """The only path onto metered billing, and naming the target is the consent
    (D4(d)). A group's membership changes after that consent was given."""
    with pytest.raises(InvalidConnection, match="rather than a group"):
        QuotaPolicy(QuotaAction.FAILOVER, "group:cheap")


def test_the_group_failover_refusal_happens_where_the_file_is_written(store):
    """At construction, not at the moment an allowance runs out -- by then the
    user is not looking at the file and a run has already been lost."""
    with pytest.raises(InvalidConnection, match="rather than a group"):
        Connection(
            name="primary",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            guards=Guards(
                on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "group:cheap")
            ),
        )


def test_a_plain_failover_target_is_untouched(store):
    policy = QuotaPolicy(QuotaAction.FAILOVER, "claude-api")
    assert policy.failover == "claude-api"


# --- the async face ---------------------------------------------------------------


def run(coro):
    """One loop, one call, no plugin -- the convention `tests/test_async.py` uses."""
    return asyncio.run(coro)


def test_achat_takes_a_group(store):
    """The async doors share the sync pre-run pipeline, so they inherit group
    resolution from `_resolve` -- but "it should follow" is what the sync doors
    were assumed to do too, and only the tested half is a behaviour."""
    bridge = bridge_for(store, make("a-gw", ("cheap",), enabled=False), make("local", ("cheap",)))

    async def drive():
        events = []
        async for event in bridge.achat(connection="group:cheap", message=MESSAGE):
            events.append(event)
        return events

    assert run(drive())[-1].connection == "local"


def test_aask_takes_a_group(store):
    bridge = bridge_for(store, make("a-gw", ("cheap",), enabled=False), make("local", ("cheap",)))
    answer = run(bridge.aask("group:cheap", MESSAGE))
    assert answer.receipt.connection == "local"


def test_an_unavailable_group_refuses_on_the_async_face_too(store):
    bridge = bridge_for(store, make("a", ("cheap",), enabled=False))
    adapter = bridge.adapter_for(Runtime.ANTHROPIC_SDK)

    async def drive():
        async for _ in bridge.achat(connection="group:cheap", message=MESSAGE):
            pass

    with pytest.raises(GroupUnavailable):
        run(drive())
    assert adapter.requests == []
