"""D11 conformance: the public value types serialize, and the wire is JSON.

D11 asks that the event vocabulary be designed *as if* serializable protocol, so
a second-language port -- or an OpenAI-compatible HTTP server as a leaf package
-- is a transport away rather than a redesign. This file is the check that keeps
the rule from being true of most types and quietly false of the rest.

It exists because of first-consumer feedback (2026-08-17): the consumer reported
``Receipt`` as having no ``to_dict``, which had in fact landed in Phase 5 a few
days earlier -- but going looking turned up the real gap, which was
``Connection``, ``Guards``, ``QuotaPolicy``, ``CredentialRef`` and ``Directive``.
A rule enforced type-by-type is a rule that drifts, so the test walks the surface
rather than naming instances one at a time: a type added later is caught by the
same assertion rather than needing a new test somebody has to remember to write.
"""

from __future__ import annotations

import json

import pytest

from modelpass.connections import (
    Connection,
    CredentialRef,
    Guards,
    QuotaAction,
    QuotaPolicy,
)
from modelpass.preflight import Directive, PreflightPlan, Receipt, plan_launch
from modelpass.runtimes import Runtime
from modelpass.types import (
    EVENT_TYPES,
    AuthMode,
    Message,
    Role,
    Sampling,
    TokenUsage,
    UsageScope,
)

CONNECTION = Connection(
    name="claude-sub",
    runtime=Runtime.ANTHROPIC_SDK,
    auth_mode=AuthMode.SUBSCRIPTION,
    credential_ref=CredentialRef.native_login(),
    guards=Guards(
        warn_at_tokens=100,
        stop_at_tokens=250,
        on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api"),
    ),
    model="claude-opus-4",
    description="Claude Code login",
)

RECEIPT = Receipt.from_plan(
    plan_launch(CONNECTION, {"ANTHROPIC_API_KEY": "sk-not-a-real-key"}),
    detected_auth_mode=AuthMode.SUBSCRIPTION,
    credential_source="Claude Code login",
    account="someone@example.com",
    plan_name="Max 5x",
    notes=("a note",),
)

#: Every public value type, with an instance populated enough to be interesting.
#: Events are added below from ``EVENT_TYPES`` so a new event class joins the
#: sweep automatically.
VALUES = {
    "Connection": CONNECTION,
    "CredentialRef": CONNECTION.credential_ref,
    "Directive": Directive("persistSession", "false", "stateless (D7)"),
    "Guards": CONNECTION.guards,
    "Message": Message(Role.USER, "hi"),
    "QuotaPolicy": CONNECTION.guards.on_quota_exhausted,
    "Receipt": RECEIPT,
    "Sampling": Sampling(temperature=0.2, reasoning_effort="high"),
    "TokenUsage": TokenUsage(10, 5, 2),
}


def _sample_event(cls: type):
    """One instance of an event class, built from its declared fields."""
    from modelpass.types import TerminalStatus

    defaults = {
        "text": "hello",
        "usage": TokenUsage(1, 2, 3),
        "cumulative": TokenUsage(1, 2, 3),
        "scope": UsageScope.DELTA,
        "guard": "tokens",
        "threshold": 10,
        "observed": 11,
        "connection": "claude-sub",
        "message": "m",
        "name": "look_up",
        "arguments": {"topic": "x"},
        "id": "t1",
        "server": "caller",
        "content": "found",
        "is_error": False,
        "from_connection": "claude-sub",
        "to_connection": "claude-api",
        "from_auth_mode": AuthMode.SUBSCRIPTION,
        "to_auth_mode": AuthMode.API_KEY,
        "to_runtime": Runtime.ANTHROPIC_SDK,
        "reason": "allowance gone",
        "status": TerminalStatus.OK,
        "runtime": Runtime.ANTHROPIC_SDK,
        "auth_mode": AuthMode.SUBSCRIPTION,
        "failed_over_from": None,
        "data": {"a": 1},
        "receipt": RECEIPT,
    }
    import dataclasses

    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name in defaults:
            kwargs[field.name] = defaults[field.name]
    return cls(**kwargs)


EVENTS = {cls.__name__: _sample_event(cls) for cls in EVENT_TYPES.values()}


@pytest.mark.parametrize("name", sorted({**VALUES, **EVENTS}))
def test_every_public_value_type_round_trips_through_json(name):
    """``to_dict()`` exists, and what it returns survives ``json.dumps``."""
    value = {**VALUES, **EVENTS}[name]
    to_dict = getattr(value, "to_dict", None)
    assert callable(to_dict), f"{name} has no to_dict(); D11 asks that it does"
    data = to_dict()
    assert isinstance(data, dict)
    # No custom encoder, and no information lost on the way back.
    assert json.loads(json.dumps(data)) == data


def test_the_receipt_carries_everything_it_shows_a_human():
    """The consumer's actual need: log the receipt beside the result it explains."""
    data = RECEIPT.to_dict()
    assert data["connection"] == "claude-sub"
    assert data["runtime"] == "anthropic-sdk"
    assert data["detected_auth_mode"] == "subscription"
    assert data["account"] == "someone@example.com"
    assert data["plan_name"] == "Max 5x"
    assert data["scrubbed"] == ["ANTHROPIC_API_KEY"]
    assert data["notes"] == ["a note"]
    assert data["guards_configured"] is True
    assert data["summary"] == RECEIPT.summary()
    # Directives are part of "what this run will be launched with", so they are
    # in the serialized form rather than being summary-only.
    assert {d["name"] for d in data["directives"]} == {
        "settings.apiKeyHelper",
        "persistSession",
    }


def test_the_launch_plan_deliberately_does_not_serialize():
    """``PreflightPlan.env`` holds credential *values*; a to_dict would leak them."""
    plan = plan_launch(CONNECTION, {"PATH": "/usr/bin"})
    assert isinstance(plan, PreflightPlan)
    assert not hasattr(plan, "to_dict")


def test_a_receipt_event_serializes_its_receipt_whole():
    from modelpass.types import ReceiptEvent

    event = ReceiptEvent(
        connection="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        receipt=RECEIPT,
    )
    data = event.to_dict()
    assert data["type"] == "receipt"
    assert data["receipt"] == RECEIPT.to_dict()
    assert json.loads(json.dumps(data)) == data


# --- R14: the sweep is also a redaction sweep (2026-09-13) ----------------------

SECRET_VALUE = "sk-not-a-real-stored-key-0123456789"

#: The value that was in the environment ``RECEIPT`` was computed from. It is a
#: credential this sweep must never find in a serialized form -- and it is a
#: real chance to find one, because the plan that produced the receipt held it.
AMBIENT_VALUE = "sk-not-a-real-key"


@pytest.mark.parametrize("name", sorted({**VALUES, **EVENTS}))
def test_no_public_value_type_can_carry_a_credential_value(name):
    """D11 says these serialize; R14 says what they may not serialize.

    Walked over the same sweep rather than named one type at a time, so a type
    added later is covered by the same assertion instead of needing a test
    somebody has to remember to write.
    """
    value = {**VALUES, **EVENTS}[name]
    serialized = json.dumps(value.to_dict(), default=str)
    assert AMBIENT_VALUE not in serialized
    assert SECRET_VALUE not in serialized


def test_a_secret_backed_receipt_serializes_its_pointer_and_its_fingerprint(tmp_path):
    from modelpass.connections import CredentialRef
    from modelpass.preflight import api_preflight, credential_fingerprint
    from modelpass.secrets import SecretStore
    from modelpass.types import AuthMode as _AuthMode

    secrets = SecretStore(tmp_path / "modelpass-home")
    secrets.set("claude-api", SECRET_VALUE)
    connection = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=_AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("secret:claude-api"),
    )
    receipt = api_preflight(
        connection, plan_launch(connection, {}), {}, secrets=secrets
    )
    data = receipt.to_dict()
    assert data["credential_source"] == (
        "the modelpass secrets file (entry claude-api)"
    )
    assert data["account"] == credential_fingerprint(SECRET_VALUE)
    assert SECRET_VALUE not in json.dumps(data)
    assert json.loads(json.dumps(data)) == data
