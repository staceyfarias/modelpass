"""Ticket 1.4: connection management from Python, with the CLI's receipts.

The consumer this exists for is a desktop Settings page (a desktop agent app's
validation report, §2b
item 2): it enumerates connections offline, validates them, checks them with a
preflight, and needs to **add** a connection with a key the user pasted into a
web form and **remove** one, programmatically, getting back exactly what
``modelpass connect`` and the bench would have shown.

So the properties pinned here are the CLI's properties, asserted against the
library instead of against stdout: nothing is written before the receipt exists,
the receipt describes the shape that is actually about to be saved, the secret
is written before the connection, the delete rules lean toward keeping a key,
and no result object carries a value.
"""

from __future__ import annotations

import pytest

from modelpass.bridge import Bridge
from modelpass.capabilities import CapabilityRegistry
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.errors import (
    ConfigError,
    CredentialRefIsSecret,
    DuplicateConnection,
    NoSuchConnection,
    PreflightFailed,
)
from modelpass.manage import Credential
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode

KEY = "sk-ant-this-value-must-never-be-written-anywhere-visible"


@pytest.fixture
def bridge(store):
    """A bridge over a temp home, with a scripted adapter for both SDK runtimes."""
    fake = FakeAdapter([])
    return Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: fake, Runtime.OPENAI_SDK: fake},
        env={"MY_KEY": KEY},
    )


def _api_plan(bridge, name="claude-api", **kwargs):
    return bridge.manage.plan_connection(
        name=name,
        runtime=Runtime.ANTHROPIC_API,
        credential=Credential.secret(KEY),
        **kwargs,
    )


# --- 1. plan: a receipt, and nothing written --------------------------------------


def test_a_plan_writes_neither_the_connection_nor_the_key(bridge):
    plan = _api_plan(bridge)
    assert plan.receipt.connection == "claude-api"
    assert plan.ok is True
    assert bridge.store.list() == ()
    assert bridge.secrets.exists() is False


def test_the_receipt_is_taken_against_the_key_that_is_about_to_be_written(bridge):
    """A receipt for a different shape than the one being saved would be theatre."""
    plan = _api_plan(bridge)
    from modelpass.preflight import credential_fingerprint

    assert plan.receipt.account == credential_fingerprint(KEY)
    assert plan.secret_entry == "claude-api"


def test_a_subscription_plan_pins_the_identity_the_receipt_reported(bridge):
    plan = bridge.manage.plan_connection(
        name="claude-sub", runtime=Runtime.ANTHROPIC_SDK
    )
    assert plan.connection.auth_mode is AuthMode.SUBSCRIPTION
    assert plan.account_pinned is not None
    if plan.account_pinned:
        assert plan.connection.account_binding is not None
    else:
        assert any("unpinned" in note for note in plan.notes)


def test_an_api_plan_has_no_identity_to_pin(bridge):
    assert _api_plan(bridge).account_pinned is None


def test_the_entry_defaults_to_the_connections_own_name(bridge):
    """Which is what makes the rename and delete rules able to tell whose key it is."""
    assert _api_plan(bridge, name="work").secret_entry == "work"
    plan = bridge.manage.plan_connection(
        name="work",
        runtime=Runtime.ANTHROPIC_API,
        credential=Credential.secret(KEY, entry="shared"),
    )
    assert plan.secret_entry == "shared"


def test_guards_may_be_given_as_thresholds_or_as_guards_but_not_both(bridge):
    plan = _api_plan(bridge, warn_at_tokens=100, stop_at_tokens=250)
    assert plan.connection.guards.stop_at_tokens == 250
    plan = _api_plan(bridge, guards=Guards(warn_at_tokens=1, stop_at_tokens=2))
    assert plan.connection.guards.warn_at_tokens == 1
    with pytest.raises(ConfigError, match="not both"):
        _api_plan(bridge, guards=Guards(), warn_at_tokens=5)


def test_an_unguarded_plan_says_so(bridge):
    assert any("no spend guards" in note for note in _api_plan(bridge).notes)


def test_a_runtime_with_no_adapter_is_reported_rather_than_refused(bridge, monkeypatch):
    """**Repointed in ticket 1.11, because every API runtime now has an adapter.**

    The stand-in used to be whichever runtime was still waiting for one --
    ``anthropic-api`` until 1.6, ``openai-api`` until 1.9, ``google-api`` until
    1.11 -- and there is no such runtime left. So the state is produced the way a
    *user* now reaches it, which is the only way left: the adapter exists and its
    vendor package is not installed, so ``is_available()`` is ``False`` and
    ``load_adapter`` refuses. That is a real install without the extra rather
    than an invented runtime.

    The behaviour under test is unchanged: a connection nothing can drive is
    still fully planned, fully checked and fully written, and the receipt says
    what is missing rather than setup refusing to speak.
    """
    from modelpass.adapters.google_api import GoogleAPIAdapter

    monkeypatch.setattr(GoogleAPIAdapter, "is_available", classmethod(lambda cls: False))
    plan = bridge.manage.plan_connection(
        name="gemini-key", runtime=Runtime.GOOGLE_API, credential=Credential.secret(KEY)
    )
    assert plan.receipt.runtime_available is False
    assert any("no adapter" in note for note in plan.notes)


def test_a_runtime_that_now_has_an_adapter_says_so(bridge, monkeypatch):
    """The mirror of the case above, and it fakes availability the same way:
    what is under test is what the plan says about a drivable runtime, not
    whether this machine happens to have the ``anthropic-api`` extra."""
    from modelpass.adapters.anthropic_api import AnthropicAPIAdapter

    monkeypatch.setattr(AnthropicAPIAdapter, "is_available", classmethod(lambda cls: True))
    plan = _api_plan(bridge)
    assert plan.receipt.runtime_available is True
    assert not any("no adapter" in note for note in plan.notes)


def test_an_environment_credential_stores_the_name_never_the_value(bridge):
    plan = bridge.manage.plan_connection(
        name="k", runtime=Runtime.ANTHROPIC_SDK, credential=Credential.env("MY_KEY")
    )
    assert plan.connection.auth_mode is AuthMode.API_KEY
    assert plan.connection.credential_ref.to_str() == "env:MY_KEY"
    assert plan.pending_secret is None


def test_a_pasted_key_in_an_environment_name_is_refused_at_construction(bridge):
    with pytest.raises(CredentialRefIsSecret):
        Credential.env(KEY)


def test_an_empty_pasted_key_is_refused(bridge):
    with pytest.raises(ConfigError, match="nothing to store"):
        Credential.secret("   ")


# --- 2. add: secret first, then the connection ------------------------------------


def test_add_writes_the_secret_and_then_the_connection(bridge):
    result = bridge.manage.add_connection(_api_plan(bridge))
    assert result.connection == "claude-api"
    assert result.secret_entry == "claude-api"
    assert bridge.secrets.get("claude-api") == KEY
    assert bridge.store.get("claude-api").credential_ref.to_str() == "secret:claude-api"
    assert KEY not in bridge.store.path.read_text(encoding="utf-8")
    assert result.secret_permissions is not None


def test_add_refuses_a_name_that_is_taken_unless_the_plan_says_overwrite(bridge):
    bridge.manage.add_connection(_api_plan(bridge))
    with pytest.raises(DuplicateConnection):
        bridge.manage.add_connection(_api_plan(bridge))
    result = bridge.manage.add_connection(_api_plan(bridge, overwrite=True))
    assert result.replaced is True


def test_add_refuses_a_plan_whose_preflight_failed_and_force_overrides(bridge):
    plan = bridge.manage.plan_connection(
        name="k", runtime=Runtime.ANTHROPIC_API, credential=Credential.env("NOT_SET")
    )
    assert plan.ok is False
    with pytest.raises(PreflightFailed, match="not writing connection"):
        bridge.manage.add_connection(plan)
    assert bridge.store.list() == ()
    bridge.manage.add_connection(plan, force=True)
    assert bridge.store.names() == ("k",)


def test_a_planned_connection_round_trips_through_the_file(bridge):
    plan = _api_plan(bridge, model="claude-opus-4", nickname="Work", stop_at_tokens=9)
    bridge.manage.add_connection(plan)
    stored = bridge.store.get("claude-api")
    assert stored.model == "claude-opus-4"
    assert stored.nickname == "Work"
    assert stored.guards.stop_at_tokens == 9


# --- 3. remove: the delete rules, and what they kept ------------------------------


def test_remove_takes_the_connections_own_unshared_entry_with_it(bridge):
    bridge.manage.add_connection(_api_plan(bridge))
    result = bridge.manage.remove_connection("claude-api")
    assert result.removed is True
    assert result.secret_removed is True
    assert "removed secret entry" in result.note
    assert bridge.store.list() == ()
    assert bridge.secrets.entries() == ()


def test_remove_leaves_an_entry_another_connection_shares_and_says_why(bridge):
    bridge.manage.add_connection(
        bridge.manage.plan_connection(
            name="one",
            runtime=Runtime.ANTHROPIC_API,
            credential=Credential.secret(KEY, entry="one"),
        )
    )
    bridge.manage.add_connection(
        bridge.manage.plan_connection(
            name="two",
            runtime=Runtime.ANTHROPIC_API,
            credential=Credential.stored_secret("one"),
        )
    )
    result = bridge.manage.remove_connection("one")
    assert result.secret_removed is False
    assert "still referenced by 'two'" in result.note
    assert bridge.secrets.get("one") == KEY


def test_remove_of_a_connection_with_no_stored_secret_says_nothing_about_keys(bridge):
    bridge.manage.add_connection(
        bridge.manage.plan_connection(
            name="k", runtime=Runtime.ANTHROPIC_SDK, credential=Credential.env("MY_KEY")
        )
    )
    result = bridge.manage.remove_connection("k")
    assert result.note == ""
    assert result.secret_entry is None


def test_removing_something_that_is_not_there_is_the_named_error(bridge):
    with pytest.raises(NoSuchConnection):
        bridge.manage.remove_connection("ghost")


# --- 4. rename and enable ---------------------------------------------------------


def test_rename_moves_the_entry_when_it_is_unambiguously_the_connections(bridge):
    bridge.manage.add_connection(_api_plan(bridge))
    result = bridge.manage.rename_connection("claude-api", "work")
    assert result.secret_moved is True
    assert result.note == "moved secret entry 'claude-api' to 'work'"
    assert bridge.secrets.get("work") == KEY
    assert bridge.secrets.entries() == ("work",)
    assert bridge.store.get("work").credential_ref.to_str() == "secret:work"


def test_rename_leaves_a_shared_locator_where_it_is(bridge):
    bridge.manage.add_connection(
        bridge.manage.plan_connection(
            name="one",
            runtime=Runtime.ANTHROPIC_API,
            credential=Credential.secret(KEY, entry="shared"),
        )
    )
    result = bridge.manage.rename_connection("one", "two")
    assert result.secret_moved is False
    assert "shared locator" in result.note
    assert bridge.secrets.entries() == ("shared",)


def test_rename_onto_a_taken_name_is_refused(bridge):
    bridge.store.add(_subscription("a"))
    bridge.store.add(_subscription("b"))
    with pytest.raises(ConfigError, match="already exists"):
        bridge.manage.rename_connection("a", "b")


def test_set_enabled_switches_runs_off_and_reports_whether_it_changed(bridge):
    bridge.store.add(_subscription("a"))
    first = bridge.manage.set_enabled("a", False)
    assert (first.enabled, first.changed) == (False, True)
    assert bridge.store.get("a").enabled is False
    assert bridge.manage.set_enabled("a", False).changed is False


def _subscription(name: str) -> Connection:
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
    )


# --- 5. nothing here ever carries a value -----------------------------------------


def test_no_result_or_repr_anywhere_in_this_surface_carries_the_key(bridge):
    """The blunt version of R14, over every object this module hands back."""
    plan = _api_plan(bridge)
    credential = Credential.secret(KEY)
    written = bridge.manage.add_connection(plan)
    renamed = bridge.manage.rename_connection("claude-api", "work")
    enabled = bridge.manage.set_enabled("work", False)
    removed = bridge.manage.remove_connection("work")
    for obj in (plan, credential, written, renamed, enabled, removed):
        assert KEY not in repr(obj), type(obj).__name__
        assert KEY not in str(obj.to_dict()), type(obj).__name__
    assert KEY not in repr(plan.pending_secret)


def test_every_result_is_json_safe(bridge):
    import json

    plan = _api_plan(bridge)
    written = bridge.manage.add_connection(plan)
    for obj in (plan, written, bridge.manage.remove_connection("claude-api")):
        json.dumps(obj.to_dict())
