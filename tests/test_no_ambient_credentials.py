"""The rule that makes the rest of the library trustworthy (D2).

An ambient credential is never a connection. These tests are the executable form
of that promise.
"""

from __future__ import annotations

import pytest

from modelpass.bridge import Bridge
from modelpass.connections import Connection
from modelpass.errors import InvalidConnection, NoSuchConnection, PreflightFailed
from modelpass.preflight import Receipt, plan_launch
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode

VENDOR_ENV = {
    "ANTHROPIC_API_KEY": "sk-ant-not-a-real-key",
    "ANTHROPIC_AUTH_TOKEN": "token",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "ANTHROPIC_PROFILE": "work",
    "CLAUDE_CODE_OAUTH_TOKEN": "oauth",
    "OPENAI_API_KEY": "sk-not-a-real-key",
    "CODEX_API_KEY": "sk-not-a-real-key",
    "ANTHROPIC_BASE_URL": "https://proxy.example/v1",
}


def test_environment_variables_do_not_create_connections(store, monkeypatch):
    for name, value in VENDOR_ENV.items():
        monkeypatch.setenv(name, value)
    assert store.list() == ()
    assert Bridge(store=store).connections() == ()


def test_chat_over_an_unconfigured_connection_fails_even_with_a_key_present(store):
    bridge = Bridge(
        store=store,
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter()},
        env=dict(VENDOR_ENV),
    )
    with pytest.raises(NoSuchConnection):
        bridge.chat(connection="claude-sub", message="hi")


def test_a_subscription_run_never_sees_an_ambient_key(subscription_connection):
    plan = plan_launch(subscription_connection, VENDOR_ENV)
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "ANTHROPIC_PROFILE",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        assert name not in plan.env
        assert name in plan.scrubbed
    # Another vendor's variables are none of this runtime's business, and are
    # left alone rather than silently rewriting the user's whole environment.
    assert plan.env["OPENAI_API_KEY"] == "sk-not-a-real-key"


def test_only_the_referenced_credential_survives(api_connection):
    plan = plan_launch(api_connection, VENDOR_ENV)
    assert plan.env["ANTHROPIC_API_KEY"] == "sk-ant-not-a-real-key"
    assert plan.preserved == ("ANTHROPIC_API_KEY",)
    assert "ANTHROPIC_AUTH_TOKEN" in plan.scrubbed
    assert "ANTHROPIC_API_KEY" not in plan.scrubbed


def test_a_referenced_credential_that_is_missing_fails_the_preflight(api_connection):
    plan = plan_launch(api_connection, {})
    assert plan.credential_present is False
    with pytest.raises(PreflightFailed, match="ANTHROPIC_API_KEY"):
        plan.require_credential()


def test_the_bridge_refuses_to_run_without_the_referenced_credential(
    bridge_factory, api_connection
):
    bridge, _ = bridge_factory(api_connection, env={})
    with pytest.raises(PreflightFailed):
        bridge.chat(connection="claude-api", message="hi")


def test_scrubbing_does_not_mutate_the_caller_environment(subscription_connection):
    env = dict(VENDOR_ENV)
    plan_launch(subscription_connection, env)
    assert env == VENDOR_ENV


def test_the_store_reads_no_environment_beyond_its_own_location(store, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    assert store.load() == {}


def test_an_api_connection_is_still_explicit(api_connection):
    """An api_key connection exists only because someone wrote it down."""
    assert api_connection.auth_mode is AuthMode.API_KEY
    assert api_connection.credential_ref.to_str() == "env:ANTHROPIC_API_KEY"


# --- ANTHROPIC_BASE_URL (D2 extension, 2026-08-17) -----------------------------


def test_an_ambient_base_url_is_scrubbed(subscription_connection):
    """Not a credential, but it decides where a subscription token gets sent."""
    plan = plan_launch(subscription_connection, VENDOR_ENV)
    assert "ANTHROPIC_BASE_URL" in plan.scrubbed
    assert "ANTHROPIC_BASE_URL" not in plan.env


def test_a_connection_may_allow_the_base_url_through_explicitly():
    connection = Connection(
        name="claude-proxy",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        allow_env=("ANTHROPIC_BASE_URL",),
    )
    plan = plan_launch(connection, VENDOR_ENV)
    assert plan.env["ANTHROPIC_BASE_URL"] == "https://proxy.example/v1"
    assert "ANTHROPIC_BASE_URL" not in plan.scrubbed
    assert plan.passthrough == ("ANTHROPIC_BASE_URL",)
    # Still not a credential, so it does not participate in the "you named a
    # credential that is missing" check.
    assert plan.preserved == ()
    assert plan.credential_present is True


def test_an_allowed_passthrough_that_is_absent_is_not_claimed_on_the_receipt():
    connection = Connection(
        name="claude-proxy",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        allow_env=("ANTHROPIC_BASE_URL",),
    )
    plan = plan_launch(connection, {})
    assert plan.passthrough == ()
    assert Receipt.from_plan(plan).summary().count("allowEnv") == 0


def test_the_receipt_says_loudly_when_a_base_url_override_is_in_play():
    connection = Connection(
        name="claude-proxy",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        allow_env=("ANTHROPIC_BASE_URL",),
    )
    summary = Receipt.from_plan(plan_launch(connection, VENDOR_ENV)).summary()
    assert "allowEnv kept ANTHROPIC_BASE_URL" in summary


def test_allow_env_is_a_closed_set_and_can_never_carry_a_credential():
    """The escape hatch cannot become a hole through D2."""
    with pytest.raises(InvalidConnection, match="allowEnv may not carry"):
        Connection(
            name="sneaky",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            allow_env=("ANTHROPIC_API_KEY",),
        )
    with pytest.raises(InvalidConnection, match="allowEnv may not carry"):
        Connection(
            name="sneaky",
            runtime=Runtime.OPENAI_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            allow_env=("ANTHROPIC_BASE_URL",),
        )


def test_allow_env_round_trips_through_the_store(store):
    connection = Connection(
        name="claude-proxy",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        allow_env=("ANTHROPIC_BASE_URL",),
    )
    store.add(connection)
    assert store.get("claude-proxy").allow_env == ("ANTHROPIC_BASE_URL",)


# --- explicit keys (R14, 2026-09-13) -------------------------------------------


SECRET_VALUE = "sk-not-a-real-stored-key-0123456789"


def _secret_connection(name: str = "claude-api") -> Connection:
    from modelpass.connections import CredentialRef

    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(f"secret:{name}"),
    )


def test_a_key_in_the_secrets_file_creates_no_connection(tmp_path):
    """An unreferenced secret is inert -- the property the env case already had."""
    from modelpass.secrets import SecretStore
    from modelpass.store import ConnectionStore

    root = tmp_path / "modelpass-home"
    SecretStore(root).set("claude-api", SECRET_VALUE)
    assert ConnectionStore(root).list() == ()
    assert Bridge(store=ConnectionStore(root)).connections() == ()


def test_the_connection_file_never_contains_the_value(tmp_path):
    from modelpass.secrets import SecretStore
    from modelpass.store import ConnectionStore

    root = tmp_path / "modelpass-home"
    store = ConnectionStore(root)
    SecretStore(root).set("claude-api", SECRET_VALUE)
    store.add(_secret_connection())
    assert SECRET_VALUE not in store.path.read_text(encoding="utf-8")


def test_no_serializable_surface_carries_a_stored_key(tmp_path):
    """The sweep R14 asks for: every ``to_dict()``, receipt, run record and
    vendor event, walked for a value that is actually in the secrets file."""
    import json

    from modelpass.preflight import api_preflight, credential_fingerprint
    from modelpass.runlog import RunRecord
    from modelpass.secrets import SecretStore
    from modelpass.store import ConnectionStore
    from modelpass.types import ReceiptEvent, TerminalEvent, TerminalStatus, VendorEvent

    root = tmp_path / "modelpass-home"
    secrets = SecretStore(root)
    secrets.set("claude-api", SECRET_VALUE)
    store = ConnectionStore(root)
    connection = _secret_connection()
    store.add(connection)

    plan = plan_launch(connection, {})
    receipt = api_preflight(connection, plan, {}, secrets=secrets)
    assert receipt.ok is True

    record = RunRecord(
        timestamp="2026-09-13T00:00:00Z",
        connection=connection.name,
        runtime=connection.runtime.value,
        auth_mode=receipt.effective_auth_mode.value,
        status="ok",
        model=receipt.model,
    )
    surfaces = [
        connection.to_dict(),
        connection.credential_ref.to_dict(),
        receipt.to_dict(),
        {"summary": receipt.summary()},
        record.to_dict(),
        ReceiptEvent(
            connection=connection.name,
            runtime=connection.runtime,
            auth_mode=AuthMode.API_KEY,
            receipt=receipt,
        ).to_dict(),
        TerminalEvent(
            status=TerminalStatus.OK,
            connection=connection.name,
            runtime=connection.runtime,
            auth_mode=AuthMode.API_KEY,
        ).to_dict(),
        VendorEvent(
            runtime=connection.runtime,
            name="rate_limit",
            data={"note": "whatever the vendor said"},
        ).to_dict(),
        secrets.permissions().to_dict(),
    ]
    for surface in surfaces:
        assert SECRET_VALUE not in json.dumps(surface, default=str)

    # The one thing about a key that may travel, and it is one-way.
    assert receipt.account == credential_fingerprint(SECRET_VALUE)
    assert SECRET_VALUE not in receipt.account


def test_the_bench_connection_view_shows_the_pointer_and_not_the_value(tmp_path):
    from modelpass.bench.app import _connection_view
    from modelpass.secrets import SecretStore
    from modelpass.store import ConnectionStore

    root = tmp_path / "modelpass-home"
    SecretStore(root).set("claude-api", SECRET_VALUE)
    store = ConnectionStore(root)
    connection = _secret_connection()
    store.add(connection)
    view = _connection_view(Bridge(store=store), connection)
    assert view["credential_ref"] == "secret:claude-api"
    assert view["credential_describes"] == (
        "the modelpass secrets file (entry claude-api)"
    )
    assert SECRET_VALUE not in repr(view)


# --- the in-process clause: an API adapter's client (2026-09-13, ticket 1.6) -------
#
# D2's rule is "nothing ambient may shape a run". Until the API runtimes it was
# enforced by the environment scrub, which is a mechanism a child process needs
# and an HTTP client in this process does not have. The rule did not go away; it
# changed shape, and validation 2026-09-13 §2.4 states the new form: *an API
# adapter constructs its client with an explicit key and never permits the SDK's
# own environment discovery.* The failure it prevents is silent -- a machine with
# an ambient ANTHROPIC_API_KEY and a connection naming a different credential
# would bill the wrong account, with nothing anywhere reporting it -- so it is
# checked here, in the file whose whole job is that rule.

DECOY = "sk-ant-the-ambient-key-that-must-never-be-used"
NAMED = "sk-ant-the-key-the-connection-actually-names"


def _api_key_connection(ref: str = "env:MODELPASS_NAMED_KEY") -> Connection:
    from modelpass.connections import CredentialRef

    return Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(ref),
        model="claude-opus-5",
    )


def test_the_api_adapter_hands_the_client_the_connections_key_not_the_ambient_one(
    monkeypatch,
):
    from modelpass.adapters.anthropic_api import AnthropicAPIAdapter
    from modelpass.adapters.base import RunRequest
    from modelpass.types import Message, Role

    monkeypatch.setenv("ANTHROPIC_API_KEY", DECOY)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", DECOY)
    monkeypatch.setenv("MODELPASS_NAMED_KEY", NAMED)

    seen: list[dict[str, object]] = []

    def factory(*, api_key: str, base_url: str | None) -> object:
        seen.append({"api_key": api_key, "base_url": base_url})
        raise _Stop()

    connection = _api_key_connection()
    adapter = AnthropicAPIAdapter(client_factory=factory)
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection),
    )
    from modelpass.errors import VendorRunFailed

    with pytest.raises(VendorRunFailed):
        list(adapter.run(request))

    assert seen == [{"api_key": NAMED, "base_url": None}]
    assert DECOY not in str(seen)


class _Stop(Exception):
    """Ends the run at the exact moment the client would have been built.

    The adapter reports a client that will not construct as a vendor failure, so
    that is what comes back out; what this test is after is the argument the
    factory was handed on the way in.
    """


def test_the_real_sdk_client_is_given_the_key_rather_than_left_to_find_one(monkeypatch):
    """The decoy is set, and the bare constructor really would have taken it.

    Both halves matter. Asserting only that ``_default_client`` carries the
    named key would pass even if the SDK ignored the argument; asserting only
    that a bare constructor reads the environment proves nothing about modelpass.
    Together they say: the thing modelpass does not do is a thing that would have
    worked, and it would have billed the wrong account.
    """
    anthropic = pytest.importorskip("anthropic")
    from modelpass.adapters.anthropic_api import _default_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", DECOY)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    assert anthropic.Anthropic().api_key == DECOY  # the failure being prevented
    client = _default_client(api_key=NAMED, base_url=None)
    assert client.api_key == NAMED


def test_the_api_preflight_reports_the_explicit_construction_as_a_fact(monkeypatch):
    from modelpass.adapters.anthropic_api import AnthropicAPIAdapter
    from modelpass.adapters.base import RunRequest
    from modelpass.types import Message, Role

    monkeypatch.setenv("ANTHROPIC_API_KEY", DECOY)
    monkeypatch.setenv("MODELPASS_NAMED_KEY", NAMED)
    connection = _api_key_connection()
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection),
    )
    receipt = AnthropicAPIAdapter().preflight(request)
    assert receipt.ok
    assert any("never reads the process environment" in note for note in receipt.notes)
    # The plan carries no environment to hand to anything (ticket 1.2), and the
    # receipt names the credential rather than carrying it.
    assert request.plan.env == {}
    assert NAMED not in repr(receipt)
    assert DECOY not in repr(receipt)


# --- the same rule on the second API adapter (ticket 1.9) --------------------------
#
# One rule, restated per adapter rather than parameterised, because the failure is
# per *vendor*: a machine with an ambient OPENAI_API_KEY and a connection naming a
# different credential must bill the account the connection named. The decoy here
# is an OpenAI-shaped one, and the second test proves the bare constructor really
# would have taken it -- which is what makes the first test a statement about a
# prevented failure rather than a statement about an argument.

OPENAI_DECOY = "sk-the-ambient-openai-key-that-must-never-be-used"
OPENAI_NAMED = "sk-the-openai-key-the-connection-actually-names"


def _openai_api_connection(ref: str = "env:MODELPASS_NAMED_OPENAI_KEY") -> Connection:
    from modelpass.connections import CredentialRef

    return Connection(
        name="openai-api",
        runtime=Runtime.OPENAI_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(ref),
        model="gpt-4o",
    )


def test_the_openai_api_adapter_hands_the_client_the_connections_key(monkeypatch):
    from modelpass.adapters.base import RunRequest
    from modelpass.adapters.openai_api import OpenAIAPIAdapter
    from modelpass.errors import VendorRunFailed
    from modelpass.types import Message, Role

    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_DECOY)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://not-the-endpoint.example")
    monkeypatch.setenv("MODELPASS_NAMED_OPENAI_KEY", OPENAI_NAMED)

    seen: list[dict[str, object]] = []

    def factory(*, api_key: str, base_url: str | None) -> object:
        seen.append({"api_key": api_key, "base_url": base_url})
        raise _Stop()

    connection = _openai_api_connection()
    adapter = OpenAIAPIAdapter(client_factory=factory)
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection),
    )
    with pytest.raises(VendorRunFailed):
        list(adapter.run(request))

    assert seen == [{"api_key": OPENAI_NAMED, "base_url": None}]
    assert OPENAI_DECOY not in str(seen)


def test_the_real_openai_client_is_given_the_key_rather_than_left_to_find_one(monkeypatch):
    """The decoy is set, and the bare constructor really would have taken it."""
    openai = pytest.importorskip("openai")
    from modelpass.adapters.openai_api import _default_client

    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_DECOY)

    assert openai.OpenAI().api_key == OPENAI_DECOY  # the failure being prevented
    client = _default_client(api_key=OPENAI_NAMED, base_url=None)
    assert client.api_key == OPENAI_NAMED


def test_the_openai_api_preflight_reports_the_explicit_construction_as_a_fact(monkeypatch):
    from modelpass.adapters.base import RunRequest
    from modelpass.adapters.openai_api import OpenAIAPIAdapter
    from modelpass.types import Message, Role

    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_DECOY)
    monkeypatch.setenv("MODELPASS_NAMED_OPENAI_KEY", OPENAI_NAMED)
    connection = _openai_api_connection()
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection),
    )
    receipt = OpenAIAPIAdapter().preflight(request)
    assert receipt.ok
    assert any("never reads the process environment" in note for note in receipt.notes)
    assert request.plan.env == {}
    assert OPENAI_NAMED not in repr(receipt)
    assert OPENAI_DECOY not in repr(receipt)


# --- the same rule on the fourth API adapter (ticket 1.11) -------------------------
#
# Restated once more, and this vendor is the one that most needed it: ``google.genai``
# discovers **two** ambient names rather than one. A machine with either set, and a
# connection naming a different credential, must bill the account the connection
# named -- and the second test proves the bare constructor really would have taken
# whichever decoy was there.

GOOGLE_DECOY = "ambient-gemini-key-that-must-never-be-used"
GOOGLE_DECOY_2 = "ambient-google-key-that-must-never-be-used"
GOOGLE_NAMED = "the-gemini-key-the-connection-actually-names"


def _google_api_connection(ref: str = "env:MODELPASS_NAMED_GOOGLE_KEY") -> Connection:
    from modelpass.connections import CredentialRef

    return Connection(
        name="gemini-api",
        runtime=Runtime.GOOGLE_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(ref),
        model="gemini-2.5-flash",
    )


def test_the_google_api_adapter_hands_the_client_the_connections_key(monkeypatch):
    from modelpass.adapters.base import RunRequest
    from modelpass.adapters.google_api import GoogleAPIAdapter
    from modelpass.errors import VendorRunFailed
    from modelpass.types import Message, Role

    monkeypatch.setenv("GEMINI_API_KEY", GOOGLE_DECOY)
    monkeypatch.setenv("GOOGLE_API_KEY", GOOGLE_DECOY_2)
    monkeypatch.setenv("MODELPASS_NAMED_GOOGLE_KEY", GOOGLE_NAMED)

    seen: list[dict[str, object]] = []

    def factory(*, api_key: str, base_url: str | None) -> object:
        seen.append({"api_key": api_key, "base_url": base_url})
        raise _Stop()

    connection = _google_api_connection()
    adapter = GoogleAPIAdapter(client_factory=factory)
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection),
    )
    with pytest.raises(VendorRunFailed):
        list(adapter.run(request))

    assert seen == [{"api_key": GOOGLE_NAMED, "base_url": None}]
    assert GOOGLE_DECOY not in str(seen)
    assert GOOGLE_DECOY_2 not in str(seen)


def test_the_real_gemini_client_is_given_the_key_rather_than_left_to_find_one(monkeypatch):
    """Both decoys are set, and the bare constructor really would have taken one."""
    genai = pytest.importorskip("google.genai")
    from modelpass.adapters.google_api import _default_client

    monkeypatch.setenv("GEMINI_API_KEY", GOOGLE_DECOY)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # The failure being prevented, read off the client the SDK would have built.
    assert genai.Client()._api_client.api_key == GOOGLE_DECOY
    assert _default_client(api_key=GOOGLE_NAMED, base_url=None)._api_client.api_key == (
        GOOGLE_NAMED
    )

    # And the other name, which is the half an adapter written against one
    # vendor's habits would have missed.
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", GOOGLE_DECOY_2)
    assert genai.Client()._api_client.api_key == GOOGLE_DECOY_2
    assert _default_client(api_key=GOOGLE_NAMED, base_url=None)._api_client.api_key == (
        GOOGLE_NAMED
    )


def test_a_base_url_travels_in_http_options_on_this_sdk():
    """There is no ``base_url`` argument here; a connection's endpoint would be
    silently dropped by an adapter that assumed there was."""
    pytest.importorskip("google.genai")
    from modelpass.adapters.google_api import _default_client

    client = _default_client(api_key=GOOGLE_NAMED, base_url="https://gateway.example")
    assert client._api_client._http_options.base_url == "https://gateway.example"


def test_the_google_api_preflight_reports_the_explicit_construction_as_a_fact(monkeypatch):
    from modelpass.adapters.base import RunRequest
    from modelpass.adapters.google_api import GoogleAPIAdapter
    from modelpass.types import Message, Role

    monkeypatch.setenv("GEMINI_API_KEY", GOOGLE_DECOY)
    monkeypatch.setenv("MODELPASS_NAMED_GOOGLE_KEY", GOOGLE_NAMED)
    connection = _google_api_connection()
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hi"),),
        plan=plan_launch(connection),
    )
    receipt = GoogleAPIAdapter().preflight(request)
    assert receipt.ok
    assert any("never reads the process environment" in note for note in receipt.notes)
    assert request.plan.env == {}
    assert GOOGLE_NAMED not in repr(receipt)
    assert GOOGLE_DECOY not in repr(receipt)
