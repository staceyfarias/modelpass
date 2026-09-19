"""The four API-key runtimes, as far as ticket 1.2 takes them (2026-09-13).

This file is the plumbing, not an adapter. It pins the things every API adapter
will stand on -- the runtime identities, the vendor map, the gating asymmetry,
the ``baseUrl`` field, the preflight that launches nothing -- and it pins them
*before* any of the four adapters exist, so that ticket 1.6 onward inherits a
contract rather than inventing one.

Two failures that made it into a shipped build are also fixed here and have
their own tests: ``Connection.vendor`` deriving a vendor from a runtime's name,
and a ``Runtime`` member with no capability row surfacing as a confusing error at
connection construction rather than at import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelpass import store as store_module
from modelpass.adapters.base import Adapter, RunRequest
from modelpass.bridge import Bridge
from modelpass.capabilities import (
    STATIC_TABLE,
    Capability,
    CapabilityRegistry,
    Support,
    runtime_auth_modes,
)
from modelpass.connections import Connection, CredentialRef, parse_base_url
from modelpass.errors import (
    AdapterNotImplemented,
    CredentialRefIsSecret,
    InvalidConnection,
    PreflightFailed,
    RuntimeGated,
)
from modelpass.preflight import (
    ModelListProbe,
    Receipt,
    api_preflight,
    credential_fingerprint,
    plan_launch,
    resolve_credential,
)
from modelpass.runtimes import (
    AGENT_RUNTIMES,
    API_RUNTIMES,
    VENDOR_OF,
    Runtime,
    parse_runtime,
)
from modelpass.types import AuthMode

KEY = "sk-not-a-real-key-0123456789"


def api_connection(
    runtime: Runtime = Runtime.ANTHROPIC_API,
    *,
    name: str = "api",
    base_url: str | None = None,
    credential_ref: str = "env:VENDOR_KEY",
) -> Connection:
    return Connection(
        name=name,
        runtime=runtime,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse(credential_ref),
        base_url=base_url,
    )


# --- 1. runtime identity, and the vendor map as the single source -----------------


def test_the_four_api_runtimes_exist_with_their_wire_values():
    assert Runtime.ANTHROPIC_API.value == "anthropic-api"
    assert Runtime.OPENAI_API.value == "openai-api"
    assert Runtime.GOOGLE_API.value == "google-api"
    assert Runtime.OPENAI_COMPATIBLE.value == "openai-compatible"
    assert API_RUNTIMES == {
        Runtime.ANTHROPIC_API,
        Runtime.OPENAI_API,
        Runtime.GOOGLE_API,
        Runtime.OPENAI_COMPATIBLE,
    }
    assert API_RUNTIMES.isdisjoint(AGENT_RUNTIMES)
    assert set(API_RUNTIMES) | set(AGENT_RUNTIMES) == set(Runtime)


def test_parse_runtime_names_the_full_set_in_its_error():
    with pytest.raises(ValueError) as excinfo:
        parse_runtime("anthropic-http")
    message = str(excinfo.value)
    for runtime in Runtime:
        assert runtime.value in message


def test_every_runtime_has_a_vendor_and_the_map_is_the_single_source():
    for runtime in Runtime:
        assert runtime in VENDOR_OF
        assert api_or_agent_vendor(runtime) == VENDOR_OF[runtime]


def api_or_agent_vendor(runtime: Runtime) -> str:
    """``Connection.vendor`` for a connection on this runtime, however built."""
    if runtime in API_RUNTIMES:
        base = "https://host.example/v1" if runtime is Runtime.OPENAI_COMPATIBLE else None
        return api_connection(runtime, base_url=base).vendor
    return Connection(
        name="agent",
        runtime=runtime,
        auth_mode=(
            AuthMode.API_KEY
            if runtime is Runtime.GOOGLE_SDK
            else AuthMode.SUBSCRIPTION
        ),
        credential_ref=CredentialRef.parse(
            "env:GEMINI_API_KEY" if runtime is Runtime.GOOGLE_SDK else "native-login"
        ),
        experimental=runtime in {Runtime.GOOGLE_CLI, Runtime.GOOGLE_SDK},
    ).vendor


def test_an_openai_compatible_connection_is_not_an_openai_account():
    """The latent defect, with the consequence it would have had.

    ``runtime.value.split("-", 1)[0]`` reads ``"openai"`` for a connection
    pointed at an Ollama box, so a caller asking for their OpenAI accounts would
    have been handed a local Llama beside them.
    """
    local = api_connection(
        Runtime.OPENAI_COMPATIBLE, name="ollama", base_url="http://localhost:11434/v1"
    )
    assert local.vendor == "compatible"
    assert api_connection(Runtime.OPENAI_API, name="oai").vendor == "openai"
    assert api_connection(Runtime.ANTHROPIC_API, name="ant").vendor == "anthropic"
    assert api_connection(Runtime.GOOGLE_API, name="gem").vendor == "google"


def test_find_by_vendor_excludes_the_compatible_endpoint(store):
    store.add(api_connection(Runtime.OPENAI_API, name="oai"))
    store.add(
        api_connection(
            Runtime.OPENAI_COMPATIBLE, name="ollama", base_url="http://localhost:11434/v1"
        )
    )
    bridge = Bridge(store=store)
    assert [c.name for c in bridge.find(vendor="openai")] == ["oai"]
    assert [c.name for c in bridge.find(vendor="compatible")] == ["ollama"]


# --- 2. the gating asymmetry, which exists on purpose -----------------------------


def test_google_api_ships_ungated_while_the_subscription_runtimes_stay_gated():
    """The asymmetry pinned, because the obvious "fix" is wrong in both directions.

    The Antigravity terms prohibit third-party software reaching the
    *subscription*; a Gemini API key falls under the Google Cloud terms instead,
    and Google's own SDK cannot reach the subscription at all. Gating
    ``google-api`` would gate something nobody prohibited. See
    ``docs/legal/google.md``.
    """
    assert api_connection(Runtime.GOOGLE_API, name="gemini").experimental is False

    for gated in (Runtime.GOOGLE_CLI, Runtime.GOOGLE_SDK):
        with pytest.raises(RuntimeGated):
            Connection(
                name="gated",
                runtime=gated,
                auth_mode=(
                    AuthMode.API_KEY
                    if gated is Runtime.GOOGLE_SDK
                    else AuthMode.SUBSCRIPTION
                ),
                credential_ref=CredentialRef.parse(
                    "env:GEMINI_API_KEY"
                    if gated is Runtime.GOOGLE_SDK
                    else "native-login"
                ),
            )


def test_the_gating_asymmetry_is_written_down_where_someone_would_undo_it():
    note = CapabilityRegistry().note(Runtime.GOOGLE_API, Capability.API_KEY_AUTH) or ""
    assert "2026-09-13" in note
    assert "google-cli" in note and "google-sdk" in note
    assert "docs/legal/google.md" in note

    # The other two places are the ones that ship, and they are the ones this
    # test can hold: the decision log that used to be the third lives in the
    # untracked docs_internal/ now, so reading it here would fail against a
    # source distribution rather than against a real regression.
    root = Path(__file__).resolve().parent.parent
    legal = (root / "docs" / "legal" / "google.md").read_text(encoding="utf-8")
    assert "google-cli" in legal and "google-sdk" in legal
    assert "gated" in legal

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "google-api" in readme and "gated placeholder" in readme


def test_the_decision_log_gloss_covers_every_decision_the_shipped_tree_cites():
    """The log itself does not ship, so its range is a claim nobody can check.

    It drifted once already: the See also line said D1-D22 while D23 was cited
    two dozen times, which tells a reader looking up D23 that they misread a
    number rather than that the index in front of them is stale.
    """
    import re

    number = re.compile(r"(?<![A-Za-z0-9])D([0-9]+)(?![0-9])")
    root = Path(__file__).resolve().parent.parent
    doc = (root / "docs" / "api-and-runtimes.md").read_text(encoding="utf-8")

    cited: set[int] = set()
    texts = [doc]
    for name in ("README.md", "CHANGELOG.md", "AGENTS.md"):
        texts.append((root / name).read_text(encoding="utf-8"))
    for path in (root / "src").rglob("*.py"):
        texts.append(path.read_text(encoding="utf-8"))
    for text in texts:
        cited.update(int(found) for found in number.findall(text))
    assert cited, "nothing in the shipped tree cites a decision any more"

    gloss = re.search(r"the decision log \(D([0-9]+).D([0-9]+)\)", doc)
    assert gloss, "the See also section no longer glosses the decision log"
    first, last = int(gloss.group(1)), int(gloss.group(2))
    assert first <= min(cited)
    assert last >= max(cited), f"D{max(cited)} is cited and the gloss stops at D{last}"


# --- 3. capability rows -----------------------------------------------------------


#: The cells an API runtime answers with a checked absence.
EXPECTED_UNSUPPORTED = {
    Capability.SUBSCRIPTION_AUTH,
    Capability.MCP_SERVERS,
    Capability.SESSIONS_RESUME,
    Capability.SESSIONS_FORK,
    Capability.SESSIONS_LIST,
    Capability.RESUME_CARRIES_SYSTEM_PROMPT,
    Capability.TTL_CONTROL,
    Capability.SUBAGENTS,
}


def test_the_two_new_capability_members_exist():
    assert Capability.SAMPLING_CONTROLS.value == "sampling_controls"
    assert Capability.MAX_OUTPUT_TOKENS.value == "max_output_tokens"


@pytest.mark.parametrize("runtime", sorted(API_RUNTIMES))
def test_an_api_runtime_takes_a_key_and_not_a_subscription(runtime):
    registry = CapabilityRegistry()
    assert registry.support(runtime, Capability.API_KEY_AUTH) is Support.SUPPORTED
    assert registry.support(runtime, Capability.SUBSCRIPTION_AUTH) is Support.UNSUPPORTED
    # Which is precisely what lets Connection accept an api_key connection here.
    assert runtime_auth_modes(runtime) == frozenset({AuthMode.API_KEY})


#: ``cache_breakpoints`` is the one cell that splits the four rows (R3, ticket
#: 1.5). Three of the vendors have no ``cache_control`` field in their request
#: schema at all, which is a checked absence and not an open question;
#: ``anthropic-api`` does have one, and whether modelpass puts a caller's
#: breakpoint through it correctly is what ticket 1.6's drive answers.
EXPECTED_UNSUPPORTED_BY_RUNTIME = {
    Runtime.OPENAI_API: {Capability.CACHE_BREAKPOINTS},
    Runtime.OPENAI_COMPATIBLE: {Capability.CACHE_BREAKPOINTS},
    Runtime.GOOGLE_API: {Capability.CACHE_BREAKPOINTS},
}


#: The three runtimes whose rows were moved by a drive, and are therefore
#: asserted cell by cell in their own files -- tests/test_adapter_anthropic_api.py
#: (ticket 1.6), tests/test_adapter_openai_api.py (ticket 1.9) and
#: tests/test_adapter_google_api.py (ticket 1.11) -- against the tests and the
#: SDK surface each cell was moved on.
DRIVEN = (Runtime.ANTHROPIC_API, Runtime.OPENAI_API, Runtime.GOOGLE_API)

#: What is left: ``openai-compatible``, which has an adapter (ticket 1.10) and
#: whose static row stays unverified **by construction** -- the runtime names a
#: shape, and what answers is whatever the connection's baseUrl points at, so the
#: cells move per install through ``refine()`` rather than in this table.
UNDRIVEN = tuple(r for r in sorted(API_RUNTIMES) if r not in DRIVEN)


@pytest.mark.parametrize("runtime", UNDRIVEN)
def test_the_row_is_checked_absences_and_otherwise_unverified(runtime):
    row = CapabilityRegistry().row(runtime)
    expected_unsupported = EXPECTED_UNSUPPORTED | EXPECTED_UNSUPPORTED_BY_RUNTIME.get(
        runtime, set()
    )
    assert set(row) == set(Capability)
    for capability, support in row.items():
        if capability is Capability.API_KEY_AUTH:
            assert support is Support.SUPPORTED
        elif capability in expected_unsupported:
            assert support is Support.UNSUPPORTED
        else:
            assert support is Support.UNVERIFIED


def test_anthropic_api_is_the_only_runtime_whose_breakpoint_cell_moved():
    """The one API runtime that *has* a cache_control field, now driven.

    This test said "waits for its drive" until ticket 1.6 produced one. The
    other three are checked absences for a reason that has not changed: their
    request schema carries no cache_control at all.
    """
    registry = CapabilityRegistry()
    assert (
        registry.support(Runtime.ANTHROPIC_API, Capability.CACHE_BREAKPOINTS)
        is Support.SUPPORTED
    )
    for runtime in (Runtime.OPENAI_API, Runtime.OPENAI_COMPATIBLE, Runtime.GOOGLE_API):
        assert (
            registry.support(runtime, Capability.CACHE_BREAKPOINTS)
            is Support.UNSUPPORTED
        )


@pytest.mark.parametrize("runtime", sorted(API_RUNTIMES))
def test_every_cell_that_is_not_unverified_carries_dated_evidence(runtime):
    registry = CapabilityRegistry()
    for capability in Capability:
        note = registry.note(runtime, capability) or ""
        assert note, f"{runtime.value}/{capability.value} has no note"
        assert "2026-09-13" in note
        if registry.support(runtime, capability) is not Support.UNVERIFIED:
            continue
        if runtime in DRIVEN:
            # These rows have an adapter, so "nobody has driven it" is no longer
            # the reason any of their cells is open. Each one names its own
            # ticket and says what would move it instead.
            assert "ticket 1." in note
        else:
            assert "adapter ticket" in note
            assert "recorded drive" in note


def test_openai_compatible_says_refine_is_the_upgrade_path():
    registry = CapabilityRegistry()
    for capability in Capability:
        note = registry.note(Runtime.OPENAI_COMPATIBLE, capability) or ""
        assert "refine()" in note
        assert "baseUrl" in note


def test_refine_actually_upgrades_a_compatible_row_without_touching_the_static_one():
    registry = CapabilityRegistry()
    refined = registry.refined(Runtime.OPENAI_COMPATIBLE, {"chat": True, "tools": False})
    assert refined.support(Runtime.OPENAI_COMPATIBLE, Capability.CHAT) is Support.SUPPORTED
    assert refined.support(Runtime.OPENAI_COMPATIBLE, Capability.TOOLS) is Support.UNSUPPORTED
    assert (
        registry.support(Runtime.OPENAI_COMPATIBLE, Capability.CHAT) is Support.UNVERIFIED
    )


def test_the_agent_runtimes_did_not_move(subscription_connection):
    """The new members must not have disturbed a verified row."""
    registry = CapabilityRegistry()
    assert registry.supports(Runtime.ANTHROPIC_SDK, Capability.CHAT)
    assert registry.supports(Runtime.OPENAI_SDK, Capability.TOOLS_IN_PROCESS)
    assert subscription_connection.vendor == "anthropic"
    for runtime in AGENT_RUNTIMES:
        for capability in (Capability.SAMPLING_CONTROLS, Capability.MAX_OUTPUT_TOKENS):
            if runtime in (Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK):
                # Moved in ticket 1.7 to a *checked* absence: no sampling
                # parameter exists in either adapter or on either CLI protocol.
                assert registry.support(runtime, capability) is Support.UNSUPPORTED
            else:
                # google-cli and google-sdk have no adapter, so nothing about
                # them has been checked and neither cell may claim it was.
                assert registry.support(runtime, capability) is Support.UNVERIFIED


# --- 4. the other latent defect: a runtime with no row ----------------------------


def test_every_runtime_member_has_a_capability_row():
    """The defect, closed at import rather than at connection construction.

    A missing row used to surface as ``ValueError: no capability row for runtime
    ...`` from inside ``Connection.__post_init__`` -- a user's connection failing
    to build, with a message that did not say the table was incomplete.
    ``capabilities.py`` now refuses to import at all, so this assertion is really
    a statement that the guard exists and has not been removed.
    """
    missing = [r.value for r in Runtime if r not in STATIC_TABLE]
    assert missing == []
    for runtime in Runtime:
        assert set(CapabilityRegistry().row(runtime)) >= {Capability.API_KEY_AUTH}


def test_the_import_guard_names_the_runtime_and_the_fix():
    from modelpass.capabilities import _check_table_coverage

    saved = STATIC_TABLE.pop(Runtime.GOOGLE_API)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _check_table_coverage()
    finally:
        STATIC_TABLE[Runtime.GOOGLE_API] = saved
    message = str(excinfo.value)
    assert "google-api" in message
    assert "Connection construction" in message


# --- 5. baseUrl -------------------------------------------------------------------


def test_base_url_is_required_on_openai_compatible():
    with pytest.raises(InvalidConnection, match="requires baseUrl"):
        api_connection(Runtime.OPENAI_COMPATIBLE, name="ollama")


def test_base_url_is_optional_on_the_vendor_api_runtimes():
    for runtime in (Runtime.ANTHROPIC_API, Runtime.OPENAI_API, Runtime.GOOGLE_API):
        assert api_connection(runtime).base_url is None
        proxied = api_connection(runtime, base_url="https://gateway.example/v1")
        assert proxied.base_url == "https://gateway.example/v1"


@pytest.mark.parametrize("runtime", sorted(AGENT_RUNTIMES))
def test_base_url_is_refused_on_an_agent_runtime(runtime):
    with pytest.raises(InvalidConnection, match="baseUrl is only supported"):
        Connection(
            name="agent",
            runtime=runtime,
            auth_mode=AuthMode.API_KEY,
            credential_ref=CredentialRef.parse("env:VENDOR_KEY"),
            experimental=True,
            base_url="https://gateway.example/v1",
        )


def test_http_is_accepted_only_for_an_explicit_local_host():
    for host in ("localhost", "127.0.0.1", "[::1]"):
        assert parse_base_url(f"http://{host}:11434/v1")
    with pytest.raises(InvalidConnection, match="plain http"):
        parse_base_url("http://models.example.com/v1")
    # A private LAN address is not loopback and gets no exemption.
    with pytest.raises(InvalidConnection, match="plain http"):
        parse_base_url("http://192.168.1.40:11434/v1")
    assert parse_base_url("https://models.example.com/v1")


def test_a_base_url_may_not_carry_credentials():
    with pytest.raises(CredentialRefIsSecret, match="credentials in the URL"):
        parse_base_url("https://user:sk-secret@models.example.com/v1")
    with pytest.raises(InvalidConnection, match="query string"):
        parse_base_url("https://models.example.com/v1?api_key=sk-secret")


def test_a_bare_host_is_refused_rather_than_guessed_at():
    with pytest.raises(InvalidConnection, match="http or https"):
        parse_base_url("models.example.com")
    with pytest.raises(InvalidConnection, match="non-empty URL"):
        parse_base_url("   ")


def test_a_base_url_is_preserved_exactly_as_written():
    for text in (
        "https://host.example/v1",
        "https://host.example/v1/",
        "https://host.example:8443/openai/v1",
    ):
        assert parse_base_url(f"  {text} ") == text


def test_base_url_round_trips_through_the_store(store):
    store.add(
        api_connection(
            Runtime.OPENAI_COMPATIBLE, name="ollama", base_url="http://localhost:11434/v1"
        )
    )
    reloaded = store.get("ollama")
    assert reloaded.base_url == "http://localhost:11434/v1"
    assert reloaded.to_dict()["base_url"] == "http://localhost:11434/v1"
    with store.path.open("rb") as handle:
        import tomllib

        raw = tomllib.load(handle)
    assert raw["connections"]["ollama"]["baseUrl"] == "http://localhost:11434/v1"


#: ``_CONNECTION_KEYS`` as 0.1.1 shipped it -- before ``baseUrl`` existed.
_ZERO_ONE_CONNECTION_KEYS = frozenset(
    {
        "runtime",
        "authMode",
        "credentialRef",
        "model",
        "description",
        "nickname",
        "configDir",
        "verifiedIdentity",
        "experimental",
        "allowEnv",
        "guards",
        "enabled",
    }
)


def test_a_zero_one_shaped_reader_carries_base_url_forward(monkeypatch):
    """The 0.1 compatibility policy, applied to the first key that tests it.

    R12 shipped alone so that *older builds carry what newer ones write*. This
    is that promise exercised against ``baseUrl``: a build whose key set predates
    the field reports it as carried and writes it back untouched, rather than
    silently dropping an endpoint from a connection it did not understand.
    """
    monkeypatch.setattr(store_module, "_CONNECTION_KEYS", _ZERO_ONE_CONNECTION_KEYS)
    raw = {
        "runtime": "openai-compatible",
        "authMode": "api_key",
        "credentialRef": "env:LOCAL_KEY",
        "baseUrl": "http://localhost:11434/v1",
    }
    assert store_module.unknown_connection_keys(raw) == ("baseUrl",)

    # What a 0.1 build's connection_to_dict would have emitted: everything it
    # knows, and nothing about the endpoint.
    emitted = {
        "runtime": "openai-compatible",
        "authMode": "api_key",
        "credentialRef": "env:LOCAL_KEY",
    }
    carried = store_module._carry_unknown_keys(raw, emitted)
    assert carried["baseUrl"] == "http://localhost:11434/v1"


# --- 6. the preflight for a runtime that launches nothing -------------------------


AMBIENT = {
    "ANTHROPIC_API_KEY": "sk-ant-ambient",
    "OPENAI_API_KEY": "sk-ambient",
    "GEMINI_API_KEY": "ambient",
    "VENDOR_KEY": KEY,
    "PATH": "/usr/bin",
}


@pytest.mark.parametrize("runtime", sorted(API_RUNTIMES))
def test_an_api_runtime_plans_no_environment_at_all(runtime):
    """Not a scrubbed environment -- *no* environment. See ``plan_launch``.

    A dict of environment variables on a request object is an invitation to hand
    it to an SDK constructor, and the SDK's own discovery would then read a key
    the connection never named.
    """
    base = "http://localhost:11434/v1" if runtime is Runtime.OPENAI_COMPATIBLE else None
    plan = plan_launch(api_connection(runtime, base_url=base), AMBIENT)
    assert dict(plan.env) == {}
    assert plan.scrubbed == ()
    assert plan.passthrough == ()
    assert plan.forbidden_args == ()
    assert plan.launches_a_process is False
    # The credential reference is still resolved the ordinary way.
    assert plan.preserved == ("VENDOR_KEY",)
    assert plan.credential_present is True


def test_an_agent_runtime_still_plans_and_scrubs_one(subscription_connection):
    plan = plan_launch(subscription_connection, AMBIENT)
    assert plan.launches_a_process is True
    assert "ANTHROPIC_API_KEY" in plan.scrubbed
    assert "ANTHROPIC_API_KEY" not in plan.env
    assert plan.env["PATH"] == "/usr/bin"


def test_the_in_process_rule_rides_on_the_receipt_as_a_directive():
    plan = plan_launch(api_connection(), AMBIENT)
    directives = {d.name: d for d in plan.directives}
    assert directives["client.credential"].value == "explicit"
    assert directives["sdk_environment_discovery"].value == "never"
    assert "os.environ" in directives["sdk_environment_discovery"].reason


def test_a_base_url_is_reported_as_a_directive_because_it_holds_no_secret():
    plan = plan_launch(
        api_connection(
            Runtime.OPENAI_COMPATIBLE, name="ollama", base_url="http://localhost:11434/v1"
        ),
        AMBIENT,
    )
    directives = {d.name: d.value for d in plan.directives}
    assert directives["client.base_url"] == "http://localhost:11434/v1"


def test_a_missing_credential_is_the_first_check_and_it_fails_the_receipt():
    connection = api_connection()
    plan = plan_launch(connection, {})
    assert plan.credential_present is False
    with pytest.raises(PreflightFailed, match="VENDOR_KEY"):
        plan.require_credential()

    receipt = api_preflight(connection, plan, {})
    assert receipt.ok is False
    assert "VENDOR_KEY" in (receipt.problem or "")
    assert receipt.account is None
    with pytest.raises(PreflightFailed):
        receipt.require_ok()


def test_an_empty_credential_counts_as_missing():
    connection = api_connection()
    with pytest.raises(PreflightFailed, match="not set in the environment"):
        resolve_credential(connection, {"VENDOR_KEY": "   "})
    assert resolve_credential(connection, AMBIENT) == KEY


def test_the_second_check_re_validates_a_hand_edited_base_url():
    """The connection object validated it; a file this build did not write did not."""
    connection = api_connection(Runtime.OPENAI_API, base_url="https://gateway.example/v1")
    plan = plan_launch(connection, AMBIENT)
    plan.require_base_url(connection)

    tampered = object.__new__(Connection)
    for field_name in (
        "name",
        "runtime",
        "auth_mode",
        "credential_ref",
        "guards",
        "model",
        "description",
        "experimental",
        "allow_env",
        "enabled",
        "config_dir",
        "nickname",
        "account_binding",
    ):
        object.__setattr__(tampered, field_name, getattr(connection, field_name))
    object.__setattr__(tampered, "base_url", "http://models.example.com/v1")
    with pytest.raises(PreflightFailed, match="plain http"):
        plan.require_base_url(tampered)


def test_the_receipt_fields_for_an_api_runtime():
    connection = api_connection()
    receipt = api_preflight(connection, plan_launch(connection, AMBIENT), AMBIENT)
    assert receipt.ok is True
    assert receipt.detected_auth_mode is AuthMode.API_KEY
    assert receipt.effective_auth_mode is AuthMode.API_KEY
    assert receipt.plan_name is None
    assert receipt.binary is None
    assert receipt.runtime_available is True
    assert receipt.credential_source == "environment variable VENDOR_KEY"
    assert receipt.account == credential_fingerprint(KEY)
    # Guard disclosure is unchanged, and matters more on a metered connection.
    assert receipt.guards_configured is False
    assert "no spend guards configured" in receipt.summary()


def test_runtime_available_means_the_vendor_sdk_imports():
    connection = api_connection()
    receipt = api_preflight(
        connection, plan_launch(connection, AMBIENT), AMBIENT, runtime_available=False
    )
    with pytest.raises(PreflightFailed, match="not available"):
        receipt.require_ok()


def test_the_key_never_reaches_the_receipt_and_the_fingerprint_does():
    """R14's redaction rule, checked over the whole serialized receipt."""
    connection = api_connection()
    receipt = api_preflight(connection, plan_launch(connection, AMBIENT), AMBIENT)
    serialized = repr(receipt.to_dict()) + receipt.summary() + repr(receipt)
    assert KEY not in serialized
    assert "sk-" not in serialized
    fingerprint = credential_fingerprint(KEY)
    assert fingerprint in serialized
    assert fingerprint.startswith("sha256:")
    assert len(fingerprint) == len("sha256:") + 8
    assert credential_fingerprint(KEY) == credential_fingerprint(KEY)
    assert credential_fingerprint(KEY) != credential_fingerprint(KEY + "x")


def test_a_receipt_is_still_a_receipt(api_connection_fixture_unused=None):
    connection = api_connection()
    receipt = api_preflight(connection, plan_launch(connection, AMBIENT), AMBIENT)
    assert isinstance(receipt, Receipt)
    assert receipt.to_dict()["runtime"] == "anthropic-api"


# --- 7. the optional probe --------------------------------------------------------


class _CountingAdapter(Adapter):
    runtime = Runtime.ANTHROPIC_API

    def __init__(self) -> None:
        self.calls = 0

    def preflight(self, request: RunRequest) -> Receipt:  # pragma: no cover - unused
        raise NotImplementedError

    def run(self, request: RunRequest):  # pragma: no cover - unused
        raise NotImplementedError

    def probe(self, request: RunRequest) -> ModelListProbe | None:
        self.calls += 1
        return ModelListProbe(ok=True, models=("model-a", "model-b"))


def _request(connection: Connection) -> RunRequest:
    return RunRequest(
        connection=connection,
        messages=(),
        plan=plan_launch(connection, AMBIENT),
    )


def test_the_base_adapter_probes_nothing_and_says_so():
    class Bare(_CountingAdapter):
        probe = Adapter.probe

    request = _request(api_connection())
    assert Bare().cached_probe(request) is None


def test_a_probe_is_cached_and_dropped_by_refresh_identity(store):
    adapter = _CountingAdapter()
    connection = api_connection()
    request = _request(connection)

    assert adapter.cached_probe(request).models == ("model-a", "model-b")
    adapter.cached_probe(request)
    assert adapter.calls == 1

    store.add(connection)
    bridge = Bridge(store=store, adapters={Runtime.ANTHROPIC_API: adapter})
    bridge.refresh_identity("api")
    adapter.cached_probe(request)
    assert adapter.calls == 2


def test_a_failed_probe_fails_the_receipt_and_names_why():
    connection = api_connection()
    plan = plan_launch(connection, AMBIENT)
    receipt = api_preflight(
        connection,
        plan,
        AMBIENT,
        probe=ModelListProbe(ok=False, detail="401 from the endpoint"),
    )
    assert receipt.ok is False
    assert "401" in (receipt.problem or "")
    assert any("probe FAILED" in note for note in receipt.notes)


def test_a_successful_probe_is_a_note_rather_than_a_verdict_change():
    connection = api_connection()
    receipt = api_preflight(
        connection,
        plan_launch(connection, AMBIENT),
        AMBIENT,
        probe=ModelListProbe(ok=True, models=("a",)),
    )
    assert receipt.ok is True
    assert any("1 model(s) listed" in note for note in receipt.notes)


# --- 8. the remaining credential kinds --------------------------------------------


def test_an_unresolvable_credential_kind_says_so_rather_than_guessing():
    """``keychain:`` is declared and unimplemented; the message says which."""
    connection = api_connection(credential_ref="keychain:modelpass/anthropic")
    with pytest.raises(AdapterNotImplemented, match="reads a platform keychain"):
        resolve_credential(connection, AMBIENT)


def test_a_native_login_is_not_a_credential_an_api_runtime_can_use():
    """Refused at construction, and refused again if one is somehow handed over."""
    with pytest.raises(InvalidConnection, match="must point at a credential"):
        api_connection(credential_ref="native-login")


def test_a_secret_reference_resolves(tmp_path, monkeypatch):
    """Landed in ticket 1.3. Written as an xfail in 1.2; the marker came off here."""
    from modelpass.secrets import SecretStore

    monkeypatch.setenv("MODELPASS_HOME", str(tmp_path))
    SecretStore(tmp_path).set("vendor-key", KEY)
    connection = api_connection(credential_ref="secret:vendor-key")
    # No ``secrets=`` argument, and no environment: the value comes from the
    # secrets file under the resolved home, read at the moment of use.
    assert resolve_credential(connection, {}) == KEY
