"""The bench, driven entirely offline through Flask's test client.

Everything here runs against a ``FakeAdapter``, a temp home and an explicit
environment mapping: no network, no vendor package, no credential. The bench is
the page that *shows* modelpass's claims, so the tests are mostly about whether it
shows the uncomfortable ones -- an absent guard, a set API key, a disabled
connection, a capability a runtime does not have.
"""

from __future__ import annotations

import json

import pytest

from modelpass.bench.app import create_app, hello_world_tool
from modelpass.bridge import Bridge
from modelpass.capabilities import CapabilityRegistry, VerifiedCapabilities, VerifyReport
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.preflight import AccountProfile
from modelpass.runtimes import Runtime
from modelpass.store import StoreSettings
from modelpass.testing import FakeAdapter, tool_exchange, usage
from modelpass.types import AuthMode, TextDeltaEvent

pytest.importorskip("flask")


@pytest.fixture
def bench(store):
    """An app over a temp store, a scripted adapter and a stated environment."""

    def make(
        connections=(),
        *,
        script=(),
        env=None,
        adapters=None,
        codex_root=None,
    ):
        for connection in connections:
            store.add(connection, overwrite=True)
        fake = FakeAdapter(script=script)
        openai_fake = FakeAdapter(script=script, runtime=Runtime.OPENAI_SDK)
        bridge = Bridge(
            store=store,
            registry=CapabilityRegistry(),
            adapters=adapters
            or {Runtime.ANTHROPIC_SDK: fake, Runtime.OPENAI_SDK: openai_fake},
            env=dict(env or {}),
        )
        app = create_app(bridge=bridge, codex_root=codex_root)
        app.config.update(TESTING=True)
        return app, app.test_client(), bridge, fake

    return make


def anthropic(name="claude-sub", **kwargs) -> Connection:
    kwargs.setdefault("guards", Guards(warn_at_tokens=100, stop_at_tokens=250))
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        **kwargs,
    )


def openai(name="codex-sub", **kwargs) -> Connection:
    return Connection(
        name=name,
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        **kwargs,
    )


def gemini(name="gemini-sub", **kwargs) -> Connection:
    """A connection on a runtime whose tool cells are still ``unverified``.

    Added 2026-08-31, when ``openai-sdk`` stopped being able to play the part
    of "a runtime that cannot run in-process tools".
    """
    kwargs.setdefault("experimental", True)
    return Connection(
        name=name,
        runtime=Runtime.GOOGLE_CLI,
        auth_mode=AuthMode.SUBSCRIPTION,
        **kwargs,
    )


def lines(response) -> list[dict]:
    """Drain a streaming response into its parsed NDJSON lines.

    Draining matters: the response body is a generator, so a test that never
    reads it never runs the chat call it was asserting about.
    """
    return [
        json.loads(line)
        for line in response.get_data(as_text=True).splitlines()
        if line.strip()
    ]


def run(client, **payload):
    """POST a playground run and fully consume the stream."""
    payload.setdefault("message", "hi")
    return lines(client.post("/playground/run", json=payload))


# --- 1. configured connections ----------------------------------------------------


def test_a_fresh_home_says_so_rather_than_looking_broken(bench):
    _, client, _, _ = bench()
    body = client.get("/").get_data(as_text=True)
    assert "No accounts configured" in body
    assert "by design" in body or "the design" in body


def test_connections_are_listed_with_runtime_mode_and_credential(bench):
    _, client, _, _ = bench([anthropic()])
    body = client.get("/").get_data(as_text=True)
    assert "claude-sub" in body
    assert "anthropic-sdk" in body
    assert "native-login" in body


def test_a_connection_without_guards_says_so_prominently(bench):
    _, client, _, _ = bench([anthropic(guards=Guards.disabled())])
    body = client.get("/").get_data(as_text=True)
    assert "no spend guards configured" in body
    assert "nothing bounds" in body


def test_a_disabled_connection_is_shown_as_disabled(bench):
    _, client, _, _ = bench([anthropic(enabled=False)])
    body = client.get("/").get_data(as_text=True)
    assert "disabled" in body


def test_check_renders_the_preflight_receipt(bench):
    _, client, _, _ = bench([anthropic()])
    body = client.get("/?check=claude-sub").get_data(as_text=True)
    assert "Preflight receipt" in body
    assert "the runtime&#39;s own login store" in body or "own login store" in body
    assert "fake-account" in body


def test_the_check_receipt_shows_the_vendor_reported_subscription_and_profile(bench):
    """The live identity belongs to the Check receipt, not to every listed row.

    A probe per row turned loading this page into a fan-out of multi-second
    vendor subprocesses. GET / is offline again; ?check= is where a vendor is
    asked anything.
    """
    fake = FakeAdapter(
        account="person@example.com",
        plan_name="team",
        account_profile=AccountProfile(
            vendor="anthropic",
            source="claude auth status",
            logged_in=True,
            auth_method="claude.ai",
            api_provider="firstParty",
            email="person@example.com",
            organization_id="org-123",
            organization_name="Example Co",
            subscription_type="team",
        ),
    )
    _, client, _, _ = bench(
        [anthropic()], adapters={Runtime.ANTHROPIC_SDK: fake}
    )
    body = client.get("/?check=claude-sub").get_data(as_text=True)
    assert "subscription type" in body
    assert "person@example.com" in body
    assert "Example Co" in body
    assert "claude auth status" in body
    assert fake.preflights, "the check must actually run a preflight"


def test_listing_accounts_asks_no_vendor_anything(bench):
    """GET / is offline: plan_launch and the stored file, and nothing else."""
    fake = FakeAdapter()
    _, client, _, _ = bench(
        [anthropic()], adapters={Runtime.ANTHROPIC_SDK: fake}
    )
    assert client.get("/").status_code == 200
    assert fake.preflights == []


def test_verify_button_pins_the_current_vendor_identity(bench):
    fake = FakeAdapter(
        account_profile=AccountProfile(
            vendor="anthropic",
            source="claude auth status",
            email="work@example.com",
            organization_id="org-work",
            subscription_type="team",
        )
    )
    _, client, bridge, _ = bench(
        [anthropic(nickname="Anthropic Work")],
        adapters={Runtime.ANTHROPIC_SDK: fake},
    )
    response = client.post(
        "/accounts/claude-sub/verify", follow_redirects=True
    )
    assert response.status_code == 200
    assert bridge.connection("claude-sub").account_binding is not None
    assert "identity verified" in response.get_data(as_text=True)


def test_check_still_works_on_a_disabled_connection(bench):
    """enabled = false switches off spending, not looking."""
    _, client, _, _ = bench([anthropic(enabled=False)])
    body = client.get("/?check=claude-sub").get_data(as_text=True)
    assert "Preflight receipt" in body
    assert "switches off is spending, not looking" in body


def test_check_on_an_unknown_connection_reports_rather_than_500s(bench):
    _, client, _, _ = bench([anthropic()])
    response = client.get("/?check=nope")
    assert response.status_code == 200
    assert "no connection named" in response.get_data(as_text=True)


# --- the ambient-environment audit ------------------------------------------------


def test_a_set_api_key_is_named_and_the_scrub_is_stated(bench):
    _, client, _, _ = bench([anthropic()], env={"OPENAI_API_KEY": "sk-live-secret"})
    body = client.get("/").get_data(as_text=True)
    assert "OPENAI_API_KEY" in body
    assert "is set in this environment" in body
    assert "scrub" in body


def test_the_audit_never_prints_a_value(bench):
    """Names only. A value read here would end up in a rendered page."""
    _, client, _, _ = bench(
        [anthropic()],
        env={"OPENAI_API_KEY": "sk-live-secret", "ANTHROPIC_API_KEY": "sk-ant-secret"},
    )
    body = client.get("/").get_data(as_text=True)
    assert "sk-live-secret" not in body
    assert "sk-ant-secret" not in body


def test_a_clean_environment_reports_clean(bench):
    _, client, _, _ = bench([anthropic()], env={"PATH": "/usr/bin"})
    body = client.get("/").get_data(as_text=True)
    assert "clean" in body


def test_the_audit_says_it_reflects_this_process(bench):
    """A terminal opened before a cleanup still carries the old copies."""
    _, client, _, _ = bench([anthropic()])
    body = client.get("/").get_data(as_text=True)
    assert "environment of the process serving this page" in body
    assert "restarted" in body


def test_the_per_connection_row_shows_what_the_launch_will_scrub(bench):
    _, client, _, _ = bench([anthropic()], env={"ANTHROPIC_API_KEY": "x"})
    body = client.get("/").get_data(as_text=True)
    assert "scrubbed at launch" in body
    assert "ANTHROPIC_API_KEY" in body


# --- 2. vendor setup --------------------------------------------------------------


def test_creating_a_subscription_connection(bench):
    _, client, bridge, _ = bench()
    response = client.post(
        "/connections/save",
        data={
            "name": "claude-sub",
            "runtime": "anthropic-sdk",
            "auth_mode": "subscription",
            "quota_action": "stop",
            "enabled": ["false", "true"],
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    connection = bridge.store.get("claude-sub")
    assert connection.auth_mode is AuthMode.SUBSCRIPTION
    assert connection.credential_ref.to_str() == "native-login"
    assert connection.enabled is True


def test_creating_a_metered_connection_stores_the_pointer_not_the_key(bench):
    _, client, bridge, _ = bench()
    client.post(
        "/connections/save",
        data={
            "name": "claude-api",
            "runtime": "anthropic-sdk",
            "auth_mode": "api_key",
            "api_key_env": "ANTHROPIC_API_KEY",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    connection = bridge.store.get("claude-api")
    assert connection.credential_ref.to_str() == "env:ANTHROPIC_API_KEY"
    assert "sk-" not in bridge.store.path.read_text(encoding="utf-8")


def test_a_pasted_secret_in_the_env_name_field_is_refused(bench):
    """The same CredentialRefIsSecret rule that protects the config file."""
    _, client, bridge, _ = bench()
    response = client.post(
        "/connections/save",
        data={
            "name": "oops",
            "runtime": "openai-sdk",
            "auth_mode": "api_key",
            "api_key_env": "sk-proj-abcdefghijklmnop",
            "quota_action": "stop",
        },
    )
    assert response.status_code == 400
    assert "looks like a secret" in response.get_data(as_text=True)
    assert bridge.store.names() == ()
    assert not bridge.store.exists()


def test_api_key_mode_without_a_variable_name_is_refused(bench):
    _, client, _, _ = bench()
    response = client.post(
        "/connections/save",
        data={
            "name": "x",
            "runtime": "openai-sdk",
            "auth_mode": "api_key",
            "api_key_env": "",
            "quota_action": "stop",
        },
    )
    assert response.status_code == 400
    assert "NAME of an environment variable" in response.get_data(as_text=True)


def test_guards_can_be_set_from_the_form(bench):
    _, client, bridge, _ = bench()
    client.post(
        "/connections/save",
        data={
            "name": "claude-sub",
            "runtime": "anthropic-sdk",
            "auth_mode": "subscription",
            "warn_at_tokens": "1000",
            "stop_at_tokens": "5000",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    guards = bridge.store.get("claude-sub").guards
    assert guards.warn_at_tokens == 1000
    assert guards.stop_at_tokens == 5000


def test_a_nonsense_threshold_is_reported_not_swallowed(bench):
    _, client, _, _ = bench()
    response = client.post(
        "/connections/save",
        data={
            "name": "claude-sub",
            "runtime": "anthropic-sdk",
            "auth_mode": "subscription",
            "stop_at_tokens": "lots",
            "quota_action": "stop",
        },
    )
    assert response.status_code == 400
    assert "whole number of tokens" in response.get_data(as_text=True)


def test_editing_a_connection_keeps_its_name_and_updates_fields(bench):
    _, client, bridge, _ = bench([anthropic()])
    client.post(
        "/connections/save",
        data={
            "original_name": "claude-sub",
            "name": "claude-sub",
            "runtime": "anthropic-sdk",
            "auth_mode": "subscription",
            "model": "opus",
            "warn_at_tokens": "100",
            "stop_at_tokens": "250",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    assert bridge.store.get("claude-sub").model == "opus"


def test_creating_a_connection_whose_name_is_taken_is_refused(bench):
    """The new-connection form must not be a way to silently wipe a connection."""
    _, client, bridge, _ = bench([anthropic(description="worked this out carefully")])
    response = client.post(
        "/connections/save",
        data={
            "name": "claude-sub",
            "runtime": "anthropic-sdk",
            "auth_mode": "subscription",
            "quota_action": "stop",
        },
    )
    assert response.status_code == 400
    assert "already exists" in response.get_data(as_text=True)
    survivor = bridge.store.get("claude-sub")
    assert survivor.description == "worked this out carefully"
    assert survivor.guards.stop_at_tokens == 250


def test_a_post_that_omits_enabled_does_not_re_enable_a_disabled_connection(bench):
    """The rendered form always sends it; a scripted one might not."""
    _, client, bridge, _ = bench([anthropic(enabled=False)])
    client.post(
        "/connections/save",
        data={
            "original_name": "claude-sub",
            "name": "claude-sub",
            "runtime": "anthropic-sdk",
            "auth_mode": "subscription",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    assert bridge.store.get("claude-sub").enabled is False


def test_renaming_onto_an_existing_name_is_refused(bench):
    """Silently replacing another connection would destroy config as a side effect."""
    _, client, bridge, _ = bench([anthropic(), anthropic(name="claude-other")])
    response = client.post(
        "/connections/save",
        data={
            "original_name": "claude-sub",
            "name": "claude-other",
            "runtime": "anthropic-sdk",
            "auth_mode": "subscription",
            "quota_action": "stop",
        },
    )
    assert response.status_code == 400
    assert "already exists" in response.get_data(as_text=True)
    assert bridge.store.names() == ("claude-other", "claude-sub")


def test_renaming_a_connection_moves_it(bench):
    _, client, bridge, _ = bench([anthropic()])
    client.post(
        "/connections/save",
        data={
            "original_name": "claude-sub",
            "name": "claude-main",
            "runtime": "anthropic-sdk",
            "auth_mode": "subscription",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    assert bridge.store.names() == ("claude-main",)


def test_disable_and_enable_round_trip(bench):
    _, client, bridge, _ = bench([anthropic()])
    client.post(
        "/connections/claude-sub/enabled", data={"enabled": "false"}, follow_redirects=True
    )
    assert bridge.store.get("claude-sub").enabled is False
    client.post(
        "/connections/claude-sub/enabled", data={"enabled": "true"}, follow_redirects=True
    )
    assert bridge.store.get("claude-sub").enabled is True


def test_disabling_keeps_the_guards(bench):
    """The reason not to make 'disable' mean 'delete and retype later'."""
    _, client, bridge, _ = bench([anthropic()])
    client.post(
        "/connections/claude-sub/enabled", data={"enabled": "false"}, follow_redirects=True
    )
    assert bridge.store.get("claude-sub").guards.stop_at_tokens == 250


def test_delete_removes_the_connection(bench):
    _, client, bridge, _ = bench([anthropic()])
    client.post("/connections/claude-sub/delete", follow_redirects=True)
    assert bridge.store.names() == ()


def test_deleting_something_absent_reports_rather_than_500s(bench):
    _, client, _, _ = bench()
    response = client.post("/connections/nope/delete", follow_redirects=True)
    assert response.status_code == 200
    assert "no connection named" in response.get_data(as_text=True)


def test_the_edit_form_prefills_from_the_connection(bench):
    _, client, _, _ = bench([anthropic(model="sonnet", description="my note")])
    body = client.get("/connections/claude-sub/edit").get_data(as_text=True)
    assert 'value="sonnet"' in body
    assert "my note" in body


def test_accounts_page_shows_nickname_vendor_id_and_isolated_root(bench, tmp_path):
    root = tmp_path / "claude-home"
    account = anthropic(
        name="anthropic-home",
        nickname="Anthropic Home",
        config_dir=str(root),
    )
    _, client, _, _ = bench([account])
    body = client.get("/").get_data(as_text=True)
    assert "Configured accounts" in body
    assert "Anthropic Home" in body
    assert "anthropic-home" in body
    assert "anthropic" in body
    assert str(root) in body
    assert "CLAUDE_CONFIG_DIR" in body


def test_account_routes_are_first_class_while_connection_routes_remain_compatible(bench):
    _, client, bridge, _ = bench()
    assert client.get("/accounts/new").status_code == 200
    response = client.post(
        "/accounts/save",
        data={
            "name": "claude-home",
            "nickname": "Anthropic Home",
            "runtime": "anthropic-sdk",
            "auth_mode": "subscription",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert bridge.account("claude-home").nickname == "Anthropic Home"
    assert client.get("/accounts/claude-home/edit").status_code == 200


def test_openai_isolated_account_page_explains_the_one_time_login(bench, tmp_path):
    """The setup is a login with CODEX_HOME set -- no config.toml edit required.

    Codex keys its keyring entry to CODEX_HOME as well as its auth.json, so the
    page must not still be asking for cli_auth_credentials_store = "file".
    """
    account = Connection(
        name="openai-work",
        nickname="OpenAI Work",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        config_dir=str(tmp_path / "codex-work"),
    )
    _, client, _, _ = bench([account])
    body = client.get("/").get_data(as_text=True)
    assert "CODEX_HOME" in body
    assert "codex login" in body
    assert "cli_auth_credentials_store" not in body


def test_a_cross_origin_post_is_refused(bench):
    """A page on the internet can reach 127.0.0.1 in your browser."""
    _, client, bridge, _ = bench([anthropic()])
    response = client.post(
        "/connections/claude-sub/delete",
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
    assert bridge.store.names() == ("claude-sub",)


def test_a_same_origin_post_is_allowed(bench):
    _, client, bridge, _ = bench([anthropic()])
    response = client.post(
        "/connections/claude-sub/delete",
        headers={"Origin": "http://localhost"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert bridge.store.names() == ()


# --- 3. playground ----------------------------------------------------------------


def test_the_playground_says_it_spends_real_allowance(bench):
    _, client, _, _ = bench([anthropic()])
    body = client.get("/playground").get_data(as_text=True)
    assert "spends real allowance" in body
    assert "Nothing here is simulated" in body


def test_a_run_streams_the_whole_event_log(bench):
    _, client, _, _ = bench(
        [anthropic()],
        script=[TextDeltaEvent("hello "), TextDeltaEvent("world"), usage(input_tokens=12)],
    )
    response = client.post(
        "/playground/run", json={"connection": "claude-sub", "message": "hi"}
    )
    assert response.status_code == 200
    payloads = lines(response)
    kinds = [p["kind"] for p in payloads]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_finished"
    types = [p["event"]["type"] for p in payloads if p["kind"] == "event"]
    # The receipt opens every stream since the first-consumer feedback round
    # (2026-08-17) -- the bench log shows it like any other event.
    assert types == ["receipt", "text_delta", "text_delta", "usage", "terminal"]


def test_the_terminal_event_carries_its_stamp(bench):
    _, client, _, _ = bench([anthropic()], script=[usage(input_tokens=9)])
    response = client.post(
        "/playground/run", json={"connection": "claude-sub", "message": "hi"}
    )
    terminal = next(
        p["event"]
        for p in lines(response)
        if p["kind"] == "event" and p["event"]["type"] == "terminal"
    )
    assert terminal["connection"] == "claude-sub"
    assert terminal["auth_mode"] == "subscription"
    assert terminal["usage"]["total_tokens"] == 9


def test_the_chosen_model_reaches_the_adapter(bench):
    _, client, _, fake = bench([anthropic()])
    run(client, connection="claude-sub", model="gpt-5.6-sol")
    assert fake.requests[0].model == "gpt-5.6-sol"


def test_an_empty_model_falls_back_to_the_connections_own(bench):
    _, client, _, fake = bench([anthropic(model="sonnet")])
    run(client, connection="claude-sub")
    assert fake.requests[0].model == "sonnet"


def test_the_hello_world_tool_is_passed_when_the_box_is_ticked(bench):
    _, client, _, fake = bench(
        [anthropic()],
        script=list(tool_exchange("hello_world", {"name": "Stacey"}, "Hello, Stacey.")),
    )
    payloads = run(client, connection="claude-sub", message="greet me", tool=True)
    assert [t.name for t in fake.requests[0].tools] == ["hello_world"]
    types = [p["event"]["type"] for p in payloads if p["kind"] == "event"]
    assert "tool_call" in types
    assert "tool_result" in types


def test_no_tool_is_passed_when_the_box_is_clear(bench):
    _, client, _, fake = bench([anthropic()])
    run(client, connection="claude-sub")
    assert fake.requests[0].tools == ()


def test_the_hello_world_tool_returns_a_live_timestamp():
    """A greeting could have been invented by the model; a timestamp from this
    process at call time could not."""
    tool = hello_world_tool()
    result = tool.handler({"name": "Stacey"})
    assert "Hello, Stacey." in result
    assert "generated in the bench process at" in result


def test_the_tool_checkbox_is_gated_by_the_capability_registry(bench):
    """And on 2026-08-31 the registry started answering yes for openai-sdk.

    The checkbox reads the row rather than a hardcoded runtime list, which is
    the whole point of gating it there: flipping the Codex default transport
    lit the box up with no bench change at all. The page still shows the
    registry's own note, so the sentence a bench user reads carries the date
    and the evidence rather than a bare verdict -- and the retired claim from
    before S6 must not have survived anywhere in it.
    """
    _, client, _, _ = bench([openai()])
    body = client.get("/playground").get_data(as_text=True)
    assert 'data-tools="yes"' in body
    assert "app-server" in body
    assert "MCP client only" not in body


def test_the_tool_checkbox_is_offered_where_it_is_supported(bench):
    _, client, _, _ = bench([anthropic()])
    assert 'data-tools="yes"' in client.get("/playground").get_data(as_text=True)


def test_asking_an_unsupporting_runtime_for_tools_is_refused_before_any_spend(bench):
    """The example moved to google-cli on 2026-08-31; the gate did not move.

    ``openai-sdk`` was the stock no here until its default transport flipped,
    which is exactly the drift a registry-driven gate is supposed to absorb.
    ``google-cli`` reports ``unverified`` for this cell -- an absence of
    knowledge, which ``supports()`` treats as unusable -- so the refusal still
    lands before the adapter is touched and nothing is spent finding out.
    """
    gemini_fake = FakeAdapter(runtime=Runtime.GOOGLE_CLI)
    _, client, _, _ = bench(
        [gemini()], adapters={Runtime.GOOGLE_CLI: gemini_fake}
    )
    response = client.post(
        "/playground/run",
        json={"connection": "gemini-sub", "message": "hi", "tool": True},
    )
    payloads = lines(response)
    errors = [p for p in payloads if p["kind"] == "error"]
    assert errors and "tools_in_process" in errors[0]["error"]
    assert gemini_fake.requests == []


def test_running_a_disabled_connection_renders_the_refusal(bench):
    _, client, _, _ = bench([anthropic(enabled=False)])
    response = client.post(
        "/playground/run", json={"connection": "claude-sub", "message": "hi"}
    )
    errors = [p for p in lines(response) if p["kind"] == "error"]
    assert errors[0]["error_type"] == "ConnectionDisabled"
    assert "is disabled" in errors[0]["error"]


def test_an_empty_message_is_refused(bench):
    _, client, _, _ = bench([anthropic()])
    response = client.post(
        "/playground/run", json={"connection": "claude-sub", "message": "   "}
    )
    assert response.status_code == 400


def test_an_adapter_error_renders_as_a_labelled_error(bench):
    """The gpt-5.6-sol-on-an-old-CLI failure mode has to be legible."""
    _, client, _, _ = bench([anthropic()])
    fake = FakeAdapter(error=RuntimeError("model 'gpt-5.6-sol' is not supported"))
    app = create_app(
        bridge=Bridge(
            store=_store_of(client),
            registry=CapabilityRegistry(),
            adapters={Runtime.ANTHROPIC_SDK: fake},
            env={},
        )
    )
    response = app.test_client().post(
        "/playground/run", json={"connection": "claude-sub", "message": "hi"}
    )
    errors = [p for p in lines(response) if p["kind"] == "error"]
    assert errors and "gpt-5.6-sol" in errors[0]["error"]


def _store_of(client):
    return client.application.config["SUBPASS_BRIDGE"].store


# --- the model picker -------------------------------------------------------------


def test_codex_models_come_from_the_vendor_cache(bench, tmp_path):
    cache = tmp_path / "codex"
    cache.mkdir()
    (cache / "models_cache.json").write_text(
        json.dumps(
            {
                "client_version": "0.148.0",
                "models": [
                    {
                        "slug": "gpt-5.6-sol",
                        "display_name": "GPT-5.6-Sol",
                        "visibility": "list",
                        "priority": 1,
                    },
                    {
                        "slug": "codex-auto-review",
                        "display_name": "Codex Auto Review",
                        "visibility": "hide",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _, client, _, _ = bench([openai()], codex_root=cache)
    body = client.get("/playground").get_data(as_text=True)
    assert "gpt-5.6-sol" in body
    assert "codex-auto-review" not in body
    assert 'data-model-exact="yes"' in body


def test_a_missing_codex_cache_degrades_to_free_text(bench, tmp_path):
    _, client, _, _ = bench([openai()], codex_root=tmp_path / "absent")
    body = client.get("/playground").get_data(as_text=True)
    assert "free-text" in body
    assert 'data-model-exact="no"' in body or "not present" in body


def test_anthropic_models_are_labelled_as_aliases(bench):
    _, client, _, _ = bench([anthropic()])
    body = client.get("/playground").get_data(as_text=True)
    assert "not a vendor model list" in body
    assert 'data-model-exact="no"' in body


# --- 4. run log -------------------------------------------------------------------


def test_the_run_log_page_renders_records_newest_first(bench):
    _, client, bridge, _ = bench([anthropic()], script=[usage(input_tokens=11)])
    run(client, connection="claude-sub", message="one", model="first-model")
    run(client, connection="claude-sub", message="two", model="second-model")

    assert len(bridge.run_log.read()) == 2
    body = client.get("/runs").get_data(as_text=True)
    assert body.index("second-model") < body.index("first-model")
    assert "subscription" in body
    assert "11" in body


def test_one_hand_edited_line_does_not_take_down_the_ledger(bench):
    """The file is advertised as hand-inspectable, so hand-damaged is a real state."""
    _, client, bridge, _ = bench([anthropic()], script=[usage(input_tokens=11)])
    run(client, connection="claude-sub")
    with bridge.run_log.path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-08-17T00:00:00+00:00",
                    "connection": "hand-edited",
                    "runtime": "anthropic-sdk",
                    "auth_mode": "subscription",
                    "status": "ok",
                    "total_tokens": "12",
                    "guards_configured": "yes",
                }
            )
            + "\n{ not json at all\n"
        )
    response = client.get("/runs")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "hand-edited" in body
    assert "claude-sub" in body


def test_an_empty_run_log_says_so(bench):
    _, client, _, _ = bench([anthropic()])
    assert "Nothing recorded yet" in client.get("/runs").get_data(as_text=True)


def test_a_switched_off_run_log_is_stated_on_the_page(bench):
    _, client, bridge, _ = bench([anthropic()])
    bridge.store.save_settings(StoreSettings(run_log=False))
    body = client.get("/runs").get_data(as_text=True)
    assert "switched off" in body
    assert "cannot be recovered later" in body


def test_the_run_log_page_points_at_the_check_that_does_not_trust_modelpass(bench):
    _, client, _, _ = bench([anthropic()])
    body = client.get("/runs").get_data(as_text=True)
    assert "usage page" in body
    assert "subscription-proof-2026-08-17.md" in body


def test_the_connections_page_links_the_run_log(bench):
    _, client, _, _ = bench([anthropic()])
    assert "runs.jsonl" in client.get("/").get_data(as_text=True)


# --- 5. ticket 1.14: the bench covers what 1.3-1.13 added -------------------------
#
# The page described the library as it stood before the API runtimes existed:
# two runtimes in the capability table, two in the runtime picker, one credential
# form, and no sign anywhere of a base URL, a timeout, a retry stance, a drive's
# verdicts or the secrets file. Everything below is the page catching up, and the
# last test in the file is the rule none of it may break: no credential value
# reaches a rendered page, ever.

SECRET = "sk-ant-this-value-must-never-be-rendered"


def compatible(name="ollama", **kwargs) -> Connection:
    kwargs.setdefault("base_url", "http://localhost:11434/v1")
    return Connection(
        name=name,
        runtime=Runtime.OPENAI_COMPATIBLE,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.none(),
        **kwargs,
    )


def anthropic_api(name="claude-api", **kwargs) -> Connection:
    kwargs.setdefault("credential_ref", CredentialRef.parse("env:ANTHROPIC_API_KEY"))
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        **kwargs,
    )


class VerifyingAdapter(FakeAdapter):
    """A ``FakeAdapter`` that answers the one hook only one runtime implements."""

    report = VerifyReport(
        verified=VerifiedCapabilities(
            supported=("chat", "streaming"),
            unsupported=("structured_output",),
            checked_at="2026-09-13T09:00:00+00:00",
            models=("llama3.1",),
        ),
        notes=("chat: answered",),
    )

    def verify_capabilities(self, request):
        self.verified_request = request
        return self.report


class FailingVerifyAdapter(FakeAdapter):
    def verify_capabilities(self, request):
        return VerifyReport(ok=False, problem="connection refused to localhost:11434")


ALL_BENCH_RUNTIMES = (
    "anthropic-sdk",
    "openai-sdk",
    "anthropic-api",
    "openai-api",
    "google-api",
    "openai-compatible",
)


@pytest.mark.parametrize("runtime", ALL_BENCH_RUNTIMES)
def test_every_configurable_runtime_has_a_capability_column_and_a_picker_row(
    bench, runtime
):
    _, client, _, _ = bench()
    assert runtime in client.get("/").get_data(as_text=True)
    assert runtime in client.get("/connections/new").get_data(as_text=True)


def test_the_two_gated_google_runtimes_are_not_offered(bench):
    """D5: the gate is about the subscription, so google-api is here and these are not."""
    _, client, _, _ = bench()
    page = client.get("/connections/new").get_data(as_text=True)
    assert "google-cli" not in page
    assert "google-sdk" not in page


def test_a_row_shows_the_endpoint_the_bounds_and_the_stance(bench):
    _, client, _, _ = bench(
        [compatible(retry="never", timeout_seconds=30.0)],
        adapters={Runtime.OPENAI_COMPATIBLE: FakeAdapter(runtime=Runtime.OPENAI_COMPATIBLE)},
    )
    page = client.get("/").get_data(as_text=True)
    assert "http://localhost:11434/v1" in page
    assert "30.0s" in page
    assert "never" in page
    assert "openai-compatible" in page


def test_an_unbounded_connection_says_unbounded_rather_than_nothing(bench):
    _, client, _, _ = bench([anthropic()])
    page = client.get("/").get_data(as_text=True)
    assert "unbounded" in page


def test_a_row_shows_the_credential_source_and_never_a_value(bench):
    _, client, _, _ = bench(
        [anthropic_api()],
        env={"ANTHROPIC_API_KEY": SECRET},
        adapters={Runtime.ANTHROPIC_API: FakeAdapter(runtime=Runtime.ANTHROPIC_API)},
    )
    page = client.get("/").get_data(as_text=True)
    assert "env:ANTHROPIC_API_KEY" in page
    assert SECRET not in page


def test_a_driven_endpoint_shows_its_cells_and_when_they_were_checked(bench):
    _, client, _, _ = bench(
        [
            compatible(
                verified_capabilities=VerifiedCapabilities(
                    supported=("chat", "streaming"),
                    unsupported=("structured_output",),
                    checked_at="2026-09-13T09:00:00+00:00",
                    models=("llama3.1",),
                )
            )
        ],
        adapters={Runtime.OPENAI_COMPATIBLE: FakeAdapter(runtime=Runtime.OPENAI_COMPATIBLE)},
    )
    page = client.get("/").get_data(as_text=True)
    assert "chat, streaming" in page
    assert "structured_output" in page
    assert "2026-09-13T09:00:00+00:00" in page
    assert "llama3.1" in page


def test_an_undriven_compatible_endpoint_says_no_cell_is_usable_yet(bench):
    _, client, _, _ = bench(
        [compatible()],
        adapters={Runtime.OPENAI_COMPATIBLE: FakeAdapter(runtime=Runtime.OPENAI_COMPATIBLE)},
    )
    page = client.get("/").get_data(as_text=True)
    assert "Nothing yet" in page
    assert "unverified" in page


# --- the secrets panel ------------------------------------------------------------


def test_the_secrets_panel_says_so_when_there_is_no_file(bench):
    _, client, _, _ = bench([anthropic()])
    page = client.get("/").get_data(as_text=True)
    assert "No secrets file" in page
    assert "normal state" in page


def test_the_secrets_panel_lists_entries_and_references_never_values(bench, store):
    from modelpass.secrets import SecretStore

    SecretStore(store.root).set("claude-api", SECRET)
    _, client, _, _ = bench(
        [
            Connection(
                name="claude-api",
                runtime=Runtime.ANTHROPIC_API,
                auth_mode=AuthMode.API_KEY,
                credential_ref=CredentialRef.parse("secret:claude-api"),
            )
        ],
        adapters={Runtime.ANTHROPIC_API: FakeAdapter(runtime=Runtime.ANTHROPIC_API)},
    )
    page = client.get("/").get_data(as_text=True)
    assert "secret:claude-api" in page
    assert "used by" in page
    assert SECRET not in page


def test_an_orphaned_entry_is_named_and_left_alone(bench, store):
    from modelpass.secrets import SecretStore

    SecretStore(store.root).set("left-behind", SECRET)
    _, client, _, _ = bench([anthropic()])
    page = client.get("/").get_data(as_text=True)
    assert "left-behind" in page
    assert "orphan" in page
    assert SecretStore(store.root).has("left-behind")
    assert SECRET not in page


def test_a_store_note_is_shown_above_the_listing(bench, store):
    """A connection this build carried rather than read is the first thing to say."""
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        '[connections.mystery]\nruntime = "quantum-sdk"\nauthMode = "subscription"\n',
        encoding="utf-8",
    )
    _, client, _, _ = bench()
    page = client.get("/").get_data(as_text=True)
    assert "mystery" in page


# --- adding a connection in either credential form --------------------------------


def test_creating_a_connection_that_reads_a_named_variable(bench):
    _, client, bridge, _ = bench(env={"ANTHROPIC_API_KEY": SECRET})
    client.post(
        "/accounts/save",
        data={
            "name": "claude-api",
            "runtime": "anthropic-api",
            "credential": "env",
            "api_key_env": "ANTHROPIC_API_KEY",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    written = bridge.store.get("claude-api")
    assert written.credential_ref.to_str() == "env:ANTHROPIC_API_KEY"
    assert written.auth_mode is AuthMode.API_KEY
    assert SECRET not in bridge.store.path.read_text(encoding="utf-8")


def test_creating_a_connection_from_a_pasted_key_writes_it_to_the_secrets_file(bench):
    _, client, bridge, _ = bench()
    response = client.post(
        "/accounts/save",
        data={
            "name": "claude-api",
            "runtime": "anthropic-api",
            "credential": "secret",
            "api_key": SECRET,
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    written = bridge.store.get("claude-api")
    assert written.credential_ref.to_str() == "secret:claude-api"
    assert bridge.secrets.get("claude-api") == SECRET
    assert SECRET not in bridge.store.path.read_text(encoding="utf-8")
    assert SECRET not in response.get_data(as_text=True)


def test_the_pasted_key_field_is_never_echoed_back_into_the_form(bench):
    """A refused save re-renders the form, and the key must not be in it."""
    _, client, bridge, _ = bench()
    response = client.post(
        "/accounts/save",
        data={
            "name": "",  # invalid: forces the form to be rendered again
            "runtime": "anthropic-api",
            "credential": "secret",
            "api_key": SECRET,
            "quota_action": "stop",
        },
    )
    assert response.status_code == 400
    assert SECRET not in response.get_data(as_text=True)
    assert not bridge.secrets.exists()


def test_choosing_the_pasted_form_with_no_key_is_refused(bench):
    _, client, bridge, _ = bench()
    response = client.post(
        "/accounts/save",
        data={
            "name": "claude-api",
            "runtime": "anthropic-api",
            "credential": "secret",
            "api_key": "",
            "quota_action": "stop",
        },
    )
    assert response.status_code == 400
    assert "no key was given" in response.get_data(as_text=True)
    assert bridge.store.names() == ()


def test_an_endpoint_that_checks_nothing_can_be_saved_with_no_credential(bench):
    _, client, bridge, _ = bench()
    client.post(
        "/accounts/save",
        data={
            "name": "ollama",
            "runtime": "openai-compatible",
            "credential": "none",
            "base_url": "http://localhost:11434/v1",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    written = bridge.store.get("ollama")
    assert written.base_url == "http://localhost:11434/v1"
    assert written.credential_ref.to_str() == "none"


def test_editing_a_connection_keeps_its_stored_key_without_retyping_it(bench, store):
    from modelpass.secrets import SecretStore

    SecretStore(store.root).set("claude-api", SECRET)
    _, client, bridge, _ = bench(
        [
            Connection(
                name="claude-api",
                runtime=Runtime.ANTHROPIC_API,
                auth_mode=AuthMode.API_KEY,
                credential_ref=CredentialRef.parse("secret:claude-api"),
            )
        ],
        adapters={Runtime.ANTHROPIC_API: FakeAdapter(runtime=Runtime.ANTHROPIC_API)},
    )
    form = client.get("/accounts/claude-api/edit").get_data(as_text=True)
    assert SECRET not in form
    assert "keep the stored key" in form
    client.post(
        "/accounts/save",
        data={
            "original_name": "claude-api",
            "name": "claude-api",
            "runtime": "anthropic-api",
            "credential": "keep",
            "secret_entry": "claude-api",
            "nickname": "Work",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    written = bridge.store.get("claude-api")
    assert written.nickname == "Work"
    assert written.credential_ref.to_str() == "secret:claude-api"
    assert bridge.secrets.get("claude-api") == SECRET


def test_an_edit_does_not_drop_the_bounds_or_the_drives_evidence(bench):
    """`retry`, `timeoutSeconds` and the verified cells are not on this form."""
    _, client, bridge, _ = bench(
        [
            compatible(
                retry="never",
                timeout_seconds=45.0,
                verified_capabilities=VerifiedCapabilities(
                    supported=("chat",), checked_at="2026-09-13T09:00:00+00:00"
                ),
            )
        ],
        adapters={Runtime.OPENAI_COMPATIBLE: FakeAdapter(runtime=Runtime.OPENAI_COMPATIBLE)},
    )
    client.post(
        "/accounts/save",
        data={
            "original_name": "ollama",
            "name": "ollama",
            "runtime": "openai-compatible",
            "credential": "none",
            "base_url": "http://localhost:11434/v1",
            "nickname": "Desk box",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    written = bridge.store.get("ollama")
    assert written.nickname == "Desk box"
    assert written.retry == "never"
    assert written.timeout_seconds == 45.0
    assert written.verified_capabilities.supported == ("chat",)


def test_renaming_a_connection_takes_its_stored_key_with_it(bench, store):
    from modelpass.secrets import SecretStore

    SecretStore(store.root).set("claude-api", SECRET)
    _, client, bridge, _ = bench(
        [
            Connection(
                name="claude-api",
                runtime=Runtime.ANTHROPIC_API,
                auth_mode=AuthMode.API_KEY,
                credential_ref=CredentialRef.parse("secret:claude-api"),
            )
        ],
        adapters={Runtime.ANTHROPIC_API: FakeAdapter(runtime=Runtime.ANTHROPIC_API)},
    )
    client.post(
        "/accounts/save",
        data={
            "original_name": "claude-api",
            "name": "claude-work",
            "runtime": "anthropic-api",
            "credential": "keep",
            "secret_entry": "claude-api",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    written = bridge.store.get("claude-work")
    assert written.credential_ref.to_str() == "secret:claude-work"
    assert bridge.secrets.get("claude-work") == SECRET
    assert not bridge.secrets.has("claude-api")
    assert not bridge.store.has("claude-api")


# --- verify, on the one runtime that can be verified ------------------------------


def test_the_verify_button_drives_a_compatible_endpoint_and_records_it(bench):
    adapter = VerifyingAdapter(runtime=Runtime.OPENAI_COMPATIBLE)
    _, client, bridge, _ = bench(
        [compatible()], adapters={Runtime.OPENAI_COMPATIBLE: adapter}
    )
    response = client.post("/accounts/ollama/verify", follow_redirects=True)
    page = response.get_data(as_text=True)
    assert "chat, streaming" in page
    written = bridge.store.get("ollama")
    assert written.verified_capabilities.supported == ("chat", "streaming")
    assert written.verified_capabilities.unsupported == ("structured_output",)
    assert written.verified_capabilities.checked_at == "2026-09-13T09:00:00+00:00"


def test_a_drive_that_could_not_reach_the_endpoint_writes_nothing(bench):
    _, client, bridge, _ = bench(
        [compatible()],
        adapters={Runtime.OPENAI_COMPATIBLE: FailingVerifyAdapter(
            runtime=Runtime.OPENAI_COMPATIBLE
        )},
    )
    response = client.post("/accounts/ollama/verify", follow_redirects=True)
    page = response.get_data(as_text=True)
    assert "connection refused" in page
    assert "Nothing was written" in page
    assert bridge.store.get("ollama").verified_capabilities.supported == ()


def test_the_verify_button_on_a_metered_api_connection_still_refuses(bench):
    """Nothing to pin and nothing to drive: an API key's identity is a fingerprint."""
    _, client, _, _ = bench(
        [anthropic_api()],
        env={"ANTHROPIC_API_KEY": SECRET},
        adapters={Runtime.ANTHROPIC_API: FakeAdapter(runtime=Runtime.ANTHROPIC_API)},
    )
    response = client.post("/accounts/claude-api/verify", follow_redirects=True)
    assert "no subscription identity to pin" in response.get_data(as_text=True)


# --- the run log carries the 1.5 / 1.7 / 1.12 fields ------------------------------


def _write_run(store, **fields):
    from modelpass.runlog import run_log_for

    record = {
        "timestamp": "2026-09-13T09:00:00+00:00",
        "connection": "claude-api",
        "runtime": "anthropic-api",
        "auth_mode": "api_key",
        "status": "ok",
        "model": "claude-sonnet-4-5",
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "guards_configured": True,
    }
    record.update(fields)
    log = run_log_for(store)
    log.path.parent.mkdir(parents=True, exist_ok=True)
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_the_run_log_shows_breakpoints_requested_against_breakpoints_honoured(
    bench, store
):
    _write_run(store, cache_breakpoints_requested=4, cache_breakpoints_honoured=0)
    _, client, _, _ = bench()
    page = client.get("/runs").get_data(as_text=True)
    assert "0/4" in page
    assert "paid for a prefix it thought it had cached" in page


def test_the_run_log_shows_honoured_breakpoints_as_honoured(bench, store):
    _write_run(store, cache_breakpoints_requested=2, cache_breakpoints_honoured=2)
    _, client, _, _ = bench()
    page = client.get("/runs").get_data(as_text=True)
    assert "2/2" in page
    assert "breakpoints honoured" in page


def test_the_run_log_shows_what_sampling_was_asked_for_and_what_was_sent(bench, store):
    _write_run(
        store,
        sampling_requested={"temperature": 0.2},
        sampling_applied={"temperature": 1.0},
        sampling_notes=["temperature 0.2 requested; gpt-5 accepts only 1.0, sent 1.0"],
    )
    _, client, _, _ = bench()
    page = client.get("/runs").get_data(as_text=True)
    assert "temperature=1.0" in page
    assert "not as requested" in page
    assert "gpt-5 accepts only 1.0" in page


def test_a_timed_out_run_says_what_the_tokens_beside_it_mean(bench, store):
    _write_run(store, status="timed_out")
    _, client, _, _ = bench()
    page = client.get("/runs").get_data(as_text=True)
    assert "timed_out" in page
    assert "the bound this call was given ran out" in page


def test_the_run_log_carries_the_librarys_own_retry_verdict(bench, store):
    _write_run(store, status="timed_out")
    _write_run(store, status="ok")
    _, client, _, _ = bench()
    page = client.get("/runs").get_data(as_text=True)
    # Computed from runtime and status by classify_terminal, not stored: the
    # page shows the verdict the library would hand a consumer's own loop, and
    # "unknown" is a real verdict rather than a missing one.
    assert "unknown" in page
    assert "the run was cut off part-way" in page
    assert "the run succeeded" in page
    assert "modelpass never" in page


def test_a_row_naming_a_runtime_this_build_does_not_know_still_renders(bench, store):
    _write_run(store, runtime="quantum-sdk")
    _, client, _, _ = bench()
    response = client.get("/runs")
    assert response.status_code == 200
    assert "quantum-sdk" in response.get_data(as_text=True)


# --- the redaction walk, extended to every rendered page --------------------------


def test_no_page_the_bench_renders_contains_a_credential_value(bench, store):
    """R14's redaction rule, walked over the bench's whole surface.

    The value is in the environment, in the secrets file, and referenced by two
    connections, which is every way a key reaches this process. Every page is
    fetched, including the two that are rendered after a mutation.
    """
    from modelpass.secrets import SecretStore

    SecretStore(store.root).set("claude-api", SECRET)
    _, client, _, _ = bench(
        [
            anthropic(),
            Connection(
                name="claude-api",
                runtime=Runtime.ANTHROPIC_API,
                auth_mode=AuthMode.API_KEY,
                credential_ref=CredentialRef.parse("secret:claude-api"),
            ),
            compatible(),
            anthropic_api(name="from-env"),
        ],
        env={"ANTHROPIC_API_KEY": SECRET, "OPENAI_API_KEY": SECRET},
        adapters={
            Runtime.ANTHROPIC_SDK: FakeAdapter(),
            Runtime.ANTHROPIC_API: FakeAdapter(runtime=Runtime.ANTHROPIC_API),
            Runtime.OPENAI_COMPATIBLE: VerifyingAdapter(
                runtime=Runtime.OPENAI_COMPATIBLE
            ),
        },
    )
    _write_run(store, sampling_applied={"temperature": 0.2})
    pages = [
        "/",
        "/?check=claude-sub",
        "/?check=claude-api",
        "/?check=ollama",
        "/connections/new",
        "/accounts/claude-api/edit",
        "/accounts/ollama/edit",
        "/playground",
        "/runs",
    ]
    for path in pages:
        body = client.get(path, follow_redirects=True).get_data(as_text=True)
        assert SECRET not in body, path
    for path in ("/accounts/ollama/verify", "/accounts/claude-sub/enabled"):
        body = client.post(path, follow_redirects=True).get_data(as_text=True)
        assert SECRET not in body, path


def test_the_playground_offers_the_models_a_drive_saw_the_endpoint_serving(bench):
    """The one runtime whose model list is a fact about the connection (1.10)."""
    _, client, _, _ = bench(
        [
            compatible(
                verified_capabilities=VerifiedCapabilities(
                    supported=("chat",),
                    checked_at="2026-09-13T09:00:00+00:00",
                    models=("llama3.1", "qwen2.5-coder"),
                )
            )
        ],
        adapters={Runtime.OPENAI_COMPATIBLE: FakeAdapter(runtime=Runtime.OPENAI_COMPATIBLE)},
    )
    page = client.get("/playground").get_data(as_text=True)
    assert "llama3.1" in page
    assert "qwen2.5-coder" in page
    assert "reported by this endpoint when it was verified" in page


# --- groups (ticket 1.15) ---------------------------------------------------------


def test_the_groups_page_names_the_account_a_group_would_reach(bench):
    _, client, _, _ = bench(
        [
            anthropic("claude-work", groups=("fast",), enabled=False),
            anthropic("claude-home", groups=("fast",)),
        ]
    )
    page = client.get("/groups").get_data(as_text=True)
    assert "claude-home" in page
    assert "passed over" in page
    assert "disabled" in page


def test_the_groups_page_says_a_group_with_no_enabled_member_would_be_refused(bench):
    """It reads, in a plain listing, exactly like a group that works."""
    _, client, _, _ = bench([anthropic("claude-sub", groups=("fast",), enabled=False)])
    page = client.get("/groups").get_data(as_text=True)
    assert "no enabled member" in page
    assert "refused" in page.lower()


def test_the_accounts_page_filters_by_group(bench):
    _, client, _, _ = bench([anthropic("in-it", groups=("fast",)), anthropic("out-of-it")])
    page = client.get("/?group=fast").get_data(as_text=True)
    assert "in-it" in page
    assert "out-of-it" not in page
    assert "Show all accounts" in page


def test_an_ungrouped_account_reads_as_default_and_says_why(bench):
    _, client, _, _ = bench([anthropic()])
    page = client.get("/").get_data(as_text=True)
    assert "this account names no group" in page


def test_the_form_writes_groups_from_a_space_separated_box(bench, store):
    _, client, _, _ = bench([anthropic()])
    client.post(
        "/accounts/save",
        data={
            "original_name": "claude-sub",
            "name": "claude-sub",
            "runtime": "anthropic-sdk",
            "credential": "native-login",
            "groups": "fast, cheap",
            "quota_action": "stop",
            "enabled": "true",
        },
        follow_redirects=True,
    )
    assert store.get("claude-sub").groups == ("fast", "cheap")


def test_the_form_refuses_a_group_name_that_is_not_one(bench, store):
    """Validated by the same rule the file and the command line are held to."""
    _, client, _, _ = bench([anthropic()])
    body = client.post(
        "/accounts/save",
        data={
            "original_name": "claude-sub",
            "name": "claude-sub",
            "runtime": "anthropic-sdk",
            "credential": "native-login",
            "groups": "not/a/group!",
            "quota_action": "stop",
            "enabled": "true",
        },
        follow_redirects=True,
    ).get_data(as_text=True)
    assert "must be 1-64 characters" in body
    assert store.get("claude-sub").groups == ()
