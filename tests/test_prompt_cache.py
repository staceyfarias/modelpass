"""Asking for prompt caching (``promptCache``, 2026-09-21).

A connection may state that it wants prompt caching, at the vendor's own
lifetime or at a named one. The per-call half of prompt caching already existed
-- ``CacheControl`` on a ``TextBlock`` says *where* the cacheable prefix ends --
and this is the other half: a standing policy, beside ``retry`` and
``timeoutSeconds``, which is the thing no caller could say anywhere.

What this file exists to hold is the **three-way outcome**, because collapsing
any two of them is the way to get this wrong:

* ``explicit`` -- the runtime takes an instruction, so the caller's breakpoints
  reach the vendor and where the prefix ends is theirs to decide.
* ``automatic`` -- the runtime caches unasked and cannot be stopped, so the
  request was **already met**. This must never be an error, and it must be
  distinguishable from ``explicit``: a caller that cannot tell them apart cannot
  tell whether its breakpoints mean anything.
* refused -- neither is established, so there is nothing to honour and nothing
  already happening. An ``InvalidConnection`` says so, because a setting that
  silently does nothing is the failure every disclosure in this library exists
  to end.

Also held here: that a connection which states nothing keeps behaving exactly as
it did before this key existed, and that no capability cell moved without the
evidence named in its note.
"""

from __future__ import annotations

import io
import json
import tomllib
from dataclasses import replace

import pytest

from modelpass.bridge import Bridge, RunRequest
from modelpass.capabilities import Capability, CapabilityRegistry, Support
from modelpass.cli import main
from modelpass.connections import Connection, CredentialRef
from modelpass.errors import InvalidConnection, SubpassError
from modelpass.prompt_cache import (
    PROMPT_CACHE_TTLS,
    VENDOR_DEFAULT,
    PromptCacheDisposition,
    plan_prompt_cache,
)
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore, connection_from_dict, connection_to_dict
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def api(name: str = "claude-api", **kwargs) -> Connection:
    kwargs.setdefault("credential_ref", CredentialRef.parse("env:ANTHROPIC_API_KEY"))
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        **kwargs,
    )


def openai(name: str = "gpt-api", **kwargs) -> Connection:
    kwargs.setdefault("credential_ref", CredentialRef.parse("env:OPENAI_API_KEY"))
    return Connection(
        name=name,
        runtime=Runtime.OPENAI_API,
        auth_mode=AuthMode.API_KEY,
        **kwargs,
    )


def subscription(name: str = "claude-sub", **kwargs) -> Connection:
    kwargs.setdefault("credential_ref", CredentialRef.native_login())
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        **kwargs,
    )


def _openai_params(connection: Connection) -> dict:
    """One Responses request, assembled. Pure: no client, no network."""
    from modelpass.adapters.openai_api import OpenAIAPIAdapter
    from modelpass.preflight import plan_launch
    from modelpass.types import Message, Role

    env = {"OPENAI_API_KEY": "sk-not-a-real-key"}
    request = RunRequest(
        connection=connection,
        messages=(Message(role=Role.USER, content="hello"),),
        plan=plan_launch(connection, env),
    )
    return OpenAIAPIAdapter(env=env).request_params(request, [])


def _bridge(store) -> Bridge:
    """Offline, against the fake adapter -- no vendor package, no credential."""
    return Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={
            Runtime.ANTHROPIC_API: FakeAdapter(runtime=Runtime.ANTHROPIC_API),
            Runtime.OPENAI_API: FakeAdapter(runtime=Runtime.OPENAI_API),
            Runtime.ANTHROPIC_SDK: FakeAdapter(runtime=Runtime.ANTHROPIC_SDK),
        },
        env={
            "ANTHROPIC_API_KEY": "sk-not-a-real-key",
            "OPENAI_API_KEY": "sk-not-a-real-key",
        },
    )


@pytest.fixture
def cli(store):
    def run(argv) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = main(argv, bridge=_bridge(store), out=out, err=err, confirm=lambda _q: True)
        return code, out.getvalue(), err.getvalue()

    return run


@pytest.fixture
def bench(store):
    def make(connections=()):
        flask_app = pytest.importorskip("modelpass.bench.app")
        for connection in connections:
            store.add(connection, overwrite=True)
        bridge = _bridge(store)
        app = flask_app.create_app(bridge=bridge)
        app.config.update(TESTING=True)
        return app, app.test_client(), bridge

    return make


# --- 1. stating nothing is the default, and it changes nothing --------------------


def test_a_connection_that_says_nothing_has_stated_nothing():
    assert api().prompt_cache is None


def test_unset_is_none_rather_than_a_caching_verdict():
    """``None`` is "nobody has said", never "caching off".

    The distinction matters more here than it does for a window: three of the
    runtimes in this library cache prompts whether or not anyone asks, so a
    consumer that read ``None`` as "off" would be reading the opposite of what
    is happening.
    """
    stated = api().prompt_cache
    assert stated is None
    assert stated != ""
    assert stated is not False


def test_a_connection_that_says_nothing_writes_no_key():
    assert "promptCache" not in connection_to_dict(api())


def test_a_connection_that_says_nothing_puts_nothing_on_the_receipt(store):
    store.add(api(name="plain"))
    receipt = _bridge(store).preflight("plain")
    assert receipt.prompt_cache_requested is None
    assert receipt.prompt_cache_disposition is None


def test_a_connection_that_says_nothing_adds_no_note(store):
    """No line at all, rather than "prompt caching: not requested".

    A disclosure printed on every receipt ever produced is noise standing where
    a disclosure should be.
    """
    store.add(api(name="plain"))
    receipt = _bridge(store).preflight("plain")
    assert not any("prompt caching" in note for note in receipt.notes)


# --- 2. outcome one: the runtime takes an instruction -----------------------------


def test_anthropic_api_honours_the_request_explicitly():
    plan = plan_prompt_cache(VENDOR_DEFAULT, Runtime.ANTHROPIC_API, name="c")
    assert plan.disposition is PromptCacheDisposition.EXPLICIT


def test_an_explicit_runtime_is_not_reported_as_already_satisfied():
    plan = plan_prompt_cache(VENDOR_DEFAULT, Runtime.ANTHROPIC_API, name="c")
    assert plan.already_satisfied is False


def test_the_explicit_note_tells_the_caller_the_breakpoints_are_theirs():
    plan = plan_prompt_cache("1h", Runtime.ANTHROPIC_API, name="c")
    assert "cache_control" in plan.note
    assert "reach the vendor" in plan.note


def test_explicit_is_decided_by_the_breakpoint_cell_and_not_by_the_vendor_name():
    """The rule is a cell, not a vendor: ``anthropic-sdk`` is the same vendor
    and gets the other answer, because it takes no instruction."""
    registry = CapabilityRegistry()
    assert (
        registry.support(Runtime.ANTHROPIC_API, Capability.CACHE_BREAKPOINTS)
        is Support.SUPPORTED
    )
    assert (
        registry.support(Runtime.ANTHROPIC_SDK, Capability.CACHE_BREAKPOINTS)
        is Support.UNSUPPORTED
    )
    assert (
        plan_prompt_cache(VENDOR_DEFAULT, Runtime.ANTHROPIC_SDK, name="c").disposition
        is PromptCacheDisposition.AUTOMATIC
    )


# --- 3. outcome two: already satisfied, and that is not an error ------------------


@pytest.mark.parametrize(
    "runtime",
    [Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK, Runtime.OPENAI_API],
)
def test_a_runtime_that_caches_unasked_reports_already_satisfied(runtime):
    plan = plan_prompt_cache(VENDOR_DEFAULT, runtime, name="c")
    assert plan.disposition is PromptCacheDisposition.AUTOMATIC
    assert plan.already_satisfied is True


@pytest.mark.parametrize(
    "runtime",
    [Runtime.ANTHROPIC_SDK, Runtime.OPENAI_SDK, Runtime.OPENAI_API],
)
def test_asking_a_runtime_that_was_doing_it_anyway_is_never_an_error(runtime):
    """The owner's case, stated as a test.

    Most vendors cache without markers and the subscription runtimes cannot be
    told to stop. Refusing there would make a correct request look like a
    mistake.
    """
    plan_prompt_cache(VENDOR_DEFAULT, runtime, name="c")


def test_automatic_and_explicit_are_different_answers_not_one_boolean():
    """The whole point of the enum. A caller reading a single "caching: yes"
    could not tell whether its breakpoints were doing anything."""
    explicit = plan_prompt_cache(VENDOR_DEFAULT, Runtime.ANTHROPIC_API, name="c")
    automatic = plan_prompt_cache(VENDOR_DEFAULT, Runtime.OPENAI_SDK, name="c")
    assert explicit.disposition is not automatic.disposition
    assert explicit.disposition.value == "explicit"
    assert automatic.disposition.value == "automatic"


def test_the_automatic_note_says_the_request_was_already_met():
    plan = plan_prompt_cache(VENDOR_DEFAULT, Runtime.OPENAI_SDK, name="c")
    assert "already met" in plan.note
    assert "without being asked" in plan.note


# --- 4. outcome three: refused rather than quietly ignored ------------------------


@pytest.mark.parametrize(
    "runtime",
    [
        Runtime.GOOGLE_API,
        Runtime.OPENAI_COMPATIBLE,
        Runtime.GOOGLE_CLI,
        Runtime.GOOGLE_SDK,
    ],
)
def test_a_runtime_with_no_established_caching_refuses_the_request(runtime):
    with pytest.raises(InvalidConnection, match="has not established"):
        plan_prompt_cache(VENDOR_DEFAULT, runtime, name="c")


def test_the_refusal_does_not_claim_the_vendor_does_no_caching():
    """The narrow claim, not the broad one.

    modelpass has not read Gemini's implicit caching off anything; that is a
    statement about modelpass's evidence, and a refusal that said "Google does
    not cache" would be the guess this whole table exists to refuse.
    """
    with pytest.raises(InvalidConnection) as caught:
        plan_prompt_cache(VENDOR_DEFAULT, Runtime.GOOGLE_API, name="c")
    message = str(caught.value)
    assert "not a claim the vendor does no caching" in message
    assert "cache_automatic reads unverified" in message


def test_the_refusal_names_the_connection_and_says_what_to_do():
    with pytest.raises(InvalidConnection, match="connection 'desk'") as caught:
        plan_prompt_cache(VENDOR_DEFAULT, Runtime.GOOGLE_API, name="desk")
    assert "Omit the key" in str(caught.value)


def test_unverified_is_refused_for_the_same_reason_supports_treats_it_as_unusable():
    registry = CapabilityRegistry()
    assert (
        registry.support(Runtime.GOOGLE_API, Capability.CACHE_AUTOMATIC)
        is Support.UNVERIFIED
    )
    assert not registry.supports(Runtime.GOOGLE_API, Capability.CACHE_AUTOMATIC)
    with pytest.raises(InvalidConnection):
        plan_prompt_cache(VENDOR_DEFAULT, Runtime.GOOGLE_API, name="c")


def test_the_refusal_is_the_repositorys_existing_error_class():
    """No new user-facing error class was invented for this."""
    with pytest.raises(InvalidConnection) as caught:
        plan_prompt_cache(VENDOR_DEFAULT, Runtime.GOOGLE_API, name="c")
    assert isinstance(caught.value, SubpassError)


# --- 5. the lifetime, and what "unspecified" means --------------------------------


def test_the_vendor_default_is_a_named_word_rather_than_an_absence():
    assert VENDOR_DEFAULT == "default"


def test_unspecified_lifetime_reads_back_as_no_ttl():
    """Three states, and this is the middle one.

    Key absent -> nothing stated. ``"default"`` -> caching asked for, lifetime
    not named, ``plan.ttl is None``. ``"1h"`` -> a lifetime named. The same
    unset-versus-zero discipline ``maxInputTokens`` uses.
    """
    plan = plan_prompt_cache(VENDOR_DEFAULT, Runtime.ANTHROPIC_API, name="c")
    assert plan.requested == VENDOR_DEFAULT
    assert plan.ttl is None


def test_a_named_lifetime_reads_back_exactly():
    plan = plan_prompt_cache("1h", Runtime.ANTHROPIC_API, name="c")
    assert plan.requested == "1h"
    assert plan.ttl == "1h"


def test_the_default_and_a_named_lifetime_are_distinguishable():
    default = plan_prompt_cache(VENDOR_DEFAULT, Runtime.ANTHROPIC_API, name="c")
    named = plan_prompt_cache("5m", Runtime.ANTHROPIC_API, name="c")
    assert default.ttl is None and named.ttl == "5m"
    assert default.requested != named.requested


def test_there_is_no_house_ttl_vocabulary_because_the_vendors_disagree():
    """The finding that settled the design.

    ``anthropic`` spells a lifetime ``'5m'``/``'1h'`` and ``openai`` spells one
    ``'in-memory'``/``'24h'``. A single modelpass vocabulary would have had to
    invent an equivalence nobody published, so the table is per runtime.
    """
    assert PROMPT_CACHE_TTLS[Runtime.ANTHROPIC_API] == ("5m", "1h")
    assert PROMPT_CACHE_TTLS[Runtime.OPENAI_API] == ("in-memory", "24h")
    assert not set(PROMPT_CACHE_TTLS[Runtime.ANTHROPIC_API]) & set(
        PROMPT_CACHE_TTLS[Runtime.OPENAI_API]
    )


def test_one_vendors_lifetime_is_refused_on_another_vendors_runtime():
    with pytest.raises(InvalidConnection, match="must be one of"):
        plan_prompt_cache("1h", Runtime.OPENAI_API, name="c")
    with pytest.raises(InvalidConnection, match="must be one of"):
        plan_prompt_cache("24h", Runtime.ANTHROPIC_API, name="c")


def test_an_unknown_lifetime_is_refused_and_the_message_lists_the_real_ones():
    with pytest.raises(InvalidConnection) as caught:
        plan_prompt_cache("30m", Runtime.ANTHROPIC_API, name="c")
    message = str(caught.value)
    assert "'5m', '1h'" in message
    assert "'default'" in message


def test_a_runtime_that_takes_no_lifetime_says_so_without_blaming_the_vendor():
    """``anthropic-sdk`` has a TTL lever (D17) that modelpass drives itself; what
    it does not have is one this key reaches. The refusal must say the narrow
    thing."""
    with pytest.raises(InvalidConnection) as caught:
        plan_prompt_cache("1h", Runtime.ANTHROPIC_SDK, name="c")
    message = str(caught.value)
    assert "accepts only 'default'" in message
    assert "carries no cache lifetime" in message


def test_an_empty_string_is_not_a_lifetime():
    with pytest.raises(InvalidConnection, match="must be a string"):
        plan_prompt_cache("", Runtime.ANTHROPIC_API, name="c")


def test_true_is_not_a_caching_request():
    """``promptCache = true`` in a hand-edited file is a plausible typo, and
    accepting it would mean guessing which of three states it meant."""
    with pytest.raises(InvalidConnection, match="must be a string"):
        plan_prompt_cache(True, Runtime.ANTHROPIC_API, name="c")  # type: ignore[arg-type]


# --- 6. the connection refuses at construction ------------------------------------


def test_an_unmeetable_request_is_refused_when_the_connection_is_built():
    """Not at the call. Everything needed to decide is on the connection, and
    ``CacheControl`` validates its own ttl in ``__post_init__`` for the same
    stated reason."""
    with pytest.raises(InvalidConnection, match="has not established"):
        Connection(
            name="gem",
            runtime=Runtime.GOOGLE_API,
            auth_mode=AuthMode.API_KEY,
            credential_ref=CredentialRef.parse("env:GEMINI_API_KEY"),
            prompt_cache=VENDOR_DEFAULT,
        )


def test_a_bad_lifetime_is_refused_when_the_connection_is_built():
    with pytest.raises(InvalidConnection, match="must be one of"):
        api(prompt_cache="24h")


def test_a_meetable_request_reads_back_off_the_connection():
    assert api(prompt_cache="1h").prompt_cache == "1h"
    assert subscription(prompt_cache=VENDOR_DEFAULT).prompt_cache == VENDOR_DEFAULT


def test_a_hand_edited_file_is_refused_at_load_rather_than_at_the_call():
    with pytest.raises(InvalidConnection, match="must be one of"):
        connection_from_dict(
            "claude-api",
            {
                "runtime": "anthropic-api",
                "authMode": "api_key",
                "credentialRef": "env:ANTHROPIC_API_KEY",
                "promptCache": "forever",
            },
        )


# --- 7. persistence ---------------------------------------------------------------


def test_the_store_round_trips_the_request(tmp_path):
    store = ConnectionStore(tmp_path / "modelpass")
    store.add(api(name="cached", prompt_cache="1h"))
    assert ConnectionStore(tmp_path / "modelpass").get("cached").prompt_cache == "1h"


def test_the_request_reaches_the_file_under_its_camelcase_key(tmp_path):
    store = ConnectionStore(tmp_path / "modelpass")
    store.add(api(name="cached", prompt_cache=VENDOR_DEFAULT))
    with store.path.open("rb") as handle:
        raw = tomllib.load(handle)
    assert raw["connections"]["cached"]["promptCache"] == "default"


_OLDER = """\
version = 1

[connections.claude-sub]
runtime = "anthropic-sdk"
authMode = "subscription"
credentialRef = "native-login"
model = "sonnet"
"""


def test_a_connection_written_before_the_key_existed_loads(store):
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(_OLDER, encoding="utf-8")
    connection = store.get("claude-sub")
    assert connection.prompt_cache is None
    assert connection.model == "sonnet"


def test_a_file_that_predates_the_key_is_not_given_one_when_rewritten(store):
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(_OLDER, encoding="utf-8")
    store.add(api(name="added"))
    with store.path.open("rb") as handle:
        raw = tomllib.load(handle)
    assert "promptCache" not in raw["connections"]["claude-sub"]
    assert "promptCache" not in raw["connections"]["added"]


def test_the_key_is_a_known_key_and_not_carried_as_unrecognised():
    from modelpass.store import unknown_connection_keys

    assert unknown_connection_keys({"promptCache": "default"}) == ()


# --- 8. the receipt, which is where a caller reads the outcome --------------------


def test_the_receipt_reports_what_was_asked_and_what_it_bought(store):
    store.add(api(name="cached", prompt_cache="1h"))
    receipt = _bridge(store).preflight("cached")
    assert receipt.prompt_cache_requested == "1h"
    assert receipt.prompt_cache_disposition == "explicit"


def test_the_receipt_distinguishes_already_satisfied_from_honoured(store):
    store.add(api(name="explicit-one", prompt_cache=VENDOR_DEFAULT))
    store.add(subscription(name="automatic-one", prompt_cache=VENDOR_DEFAULT))
    bridge = _bridge(store)
    assert bridge.preflight("explicit-one").prompt_cache_disposition == "explicit"
    assert bridge.preflight("automatic-one").prompt_cache_disposition == "automatic"


def test_the_receipt_carries_the_sentence_as_a_note(store):
    store.add(subscription(name="automatic-one", prompt_cache=VENDOR_DEFAULT))
    receipt = _bridge(store).preflight("automatic-one")
    assert any("already met" in note for note in receipt.notes)


def test_the_receipt_serialises_both_fields(store):
    store.add(api(name="cached", prompt_cache="5m"))
    data = json.loads(json.dumps(_bridge(store).preflight("cached").to_dict()))
    assert data["prompt_cache_requested"] == "5m"
    assert data["prompt_cache_disposition"] == "explicit"


# --- 9. the one place it reaches a wire -------------------------------------------


def test_an_openai_api_connection_sends_its_retention_policy():
    """``openai`` 2.32.0 types ``prompt_cache_retention`` on the Responses
    request, so a stated lifetime on this runtime is a lifetime that travels."""
    params = _openai_params(openai(prompt_cache="24h", model="gpt-5"))
    assert params["prompt_cache_retention"] == "24h"


def test_the_vendor_default_sends_no_retention_policy():
    """Nothing on the wire says "cache at whatever you normally do", so the
    honest way to say it is to say nothing."""
    params = _openai_params(openai(prompt_cache=VENDOR_DEFAULT, model="gpt-5"))
    assert "prompt_cache_retention" not in params


def test_a_connection_that_states_nothing_sends_no_retention_policy():
    params = _openai_params(openai(model="gpt-5"))
    assert "prompt_cache_retention" not in params


def test_every_accepted_openai_lifetime_is_one_the_installed_sdk_types():
    """The table is a read of the SDK, so it has to still match the SDK.

    This is the test that fails when somebody upgrades ``openai`` and the
    literal moves -- which is the whole reason the values are not remembered
    from a documentation page.
    """
    pytest.importorskip("openai")
    import typing

    from openai.types.responses import response_create_params

    hints = typing.get_type_hints(response_create_params.ResponseCreateParamsBase)
    accepted = set()
    for arg in typing.get_args(hints["prompt_cache_retention"]):
        accepted.update(a for a in typing.get_args(arg) if isinstance(a, str))
    assert set(PROMPT_CACHE_TTLS[Runtime.OPENAI_API]) == accepted


def test_every_accepted_anthropic_lifetime_is_one_the_installed_sdk_types():
    pytest.importorskip("anthropic")
    import typing

    from anthropic.types.cache_control_ephemeral_param import CacheControlEphemeralParam

    hints = typing.get_type_hints(CacheControlEphemeralParam)
    accepted = {
        a for a in typing.get_args(hints["ttl"]) if isinstance(a, str)
    }
    assert set(PROMPT_CACHE_TTLS[Runtime.ANTHROPIC_API]) == accepted


# --- 10. the capability cell, and the evidence under it ---------------------------


def test_openai_api_caches_without_being_asked():
    """The cell, and the two typed fields it was read off.

    A required cache-read counter on every response, beside a retention policy
    that offers to *extend* something the request has no way to *start*, is a
    cache the caller did not ask for.
    """
    pytest.importorskip("openai")
    from openai.types.responses.response_usage import InputTokensDetails, ResponseUsage

    cached = InputTokensDetails.model_fields["cached_tokens"]
    assert cached.annotation is int
    # Required, not optional: every response reports it, whether or not the
    # caller ever mentioned caching.
    assert cached.is_required()
    assert ResponseUsage.model_fields["input_tokens_details"].is_required()
    registry = CapabilityRegistry()
    assert (
        registry.support(Runtime.OPENAI_API, Capability.CACHE_AUTOMATIC)
        is Support.SUPPORTED
    )


def test_the_anthropic_sdk_cell_rests_on_a_capture_of_a_cache_nobody_asked_for():
    """The fixture the note names, re-read rather than taken on trust.

    modelpass cannot send a caching instruction on this runtime -- there is no
    field for one -- so tokens written into a prompt cache on these runs are the
    CLI caching unasked.
    """
    capture = FIXTURES / "structured_output"
    raw = (capture / "anthropic-structured-output-2026-08-17.json").read_text(
        encoding="utf-8"
    )
    written = [
        int(m)
        for m in __import__("re").findall(r'"cache_creation_input_tokens":\s*(\d+)', raw)
    ]
    assert written and all(n > 0 for n in written)
    registry = CapabilityRegistry()
    assert (
        registry.support(Runtime.ANTHROPIC_SDK, Capability.CACHE_BREAKPOINTS)
        is Support.UNSUPPORTED
    )
    assert (
        registry.support(Runtime.ANTHROPIC_SDK, Capability.CACHE_AUTOMATIC)
        is Support.SUPPORTED
    )


def test_the_openai_sdk_cell_rests_on_a_prefix_that_warmed_itself():
    """The app-server capture: a thread whose first turn reads nothing from
    cache and whose later turns read thousands, with nothing ever written on
    request and no field to request it with."""
    reads = []
    with (FIXTURES / "appserver" / "live-capture-2026-08-31.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            if "cachedInputTokens" not in line:
                continue
            total = json.loads(line)["params"]["tokenUsage"]["total"]
            reads.append((total["cachedInputTokens"], total["cacheWriteInputTokens"]))
    assert reads[0][0] == 0
    assert max(cached for cached, _ in reads) > 0
    assert all(written == 0 for _, written in reads)
    registry = CapabilityRegistry()
    assert (
        registry.support(Runtime.OPENAI_SDK, Capability.CACHE_AUTOMATIC)
        is Support.SUPPORTED
    )


def test_the_new_cell_carries_a_dated_note_wherever_it_is_not_unverified():
    """A cell without its evidence is a guess wearing a verdict's clothes."""
    registry = CapabilityRegistry()
    for runtime in Runtime:
        if registry.support(runtime, Capability.CACHE_AUTOMATIC) is Support.UNVERIFIED:
            continue
        note = registry.note(runtime, Capability.CACHE_AUTOMATIC) or ""
        assert note, f"{runtime.value} has no cache_automatic note"
        assert "2026-09-21" in note, f"{runtime.value} note carries no date"


def test_the_new_cell_did_not_move_any_other_cell():
    """Adding a question must not change an existing answer."""
    registry = CapabilityRegistry()
    assert (
        registry.support(Runtime.ANTHROPIC_API, Capability.CACHE_BREAKPOINTS)
        is Support.SUPPORTED
    )
    assert (
        registry.support(Runtime.ANTHROPIC_SDK, Capability.TTL_CONTROL)
        is Support.SUPPORTED
    )
    for runtime in (Runtime.OPENAI_SDK, Runtime.OPENAI_API, Runtime.GOOGLE_API):
        assert (
            registry.support(runtime, Capability.CACHE_BREAKPOINTS)
            is Support.UNSUPPORTED
        )


def test_automatic_caching_is_not_a_lever_and_ttl_control_still_says_so():
    """``cache_automatic = supported`` is the opposite of controllable.

    ``openai-sdk`` caches and offers nothing: no breakpoint, no lifetime. The
    three cells have to be readable together without one implying another.
    """
    registry = CapabilityRegistry()
    row = registry.row(Runtime.OPENAI_SDK)
    assert row[Capability.CACHE_AUTOMATIC] is Support.SUPPORTED
    assert row[Capability.CACHE_BREAKPOINTS] is Support.UNSUPPORTED
    assert row[Capability.TTL_CONTROL] is Support.UNSUPPORTED


# --- 11. the surfaces -------------------------------------------------------------


def test_the_verbose_listing_states_the_request(cli, store):
    store.add(api(name="cached", prompt_cache="1h"))
    _, out, _ = cli(["list", "--verbose"])
    assert "cache     prompt caching requested at 1h" in out
    assert "explicit on anthropic-api" in out


def test_the_verbose_listing_names_the_vendor_default(cli, store):
    store.add(subscription(name="sub", prompt_cache=VENDOR_DEFAULT))
    _, out, _ = cli(["list", "--verbose"])
    assert "the vendor's own lifetime" in out
    assert "automatic on anthropic-sdk" in out


def test_the_verbose_listing_is_silent_when_nothing_was_stated(cli, store):
    store.add(api(name="plain"))
    _, out, _ = cli(["list", "--verbose"])
    assert "prompt caching requested" not in out


def test_the_bench_detail_shows_the_request(bench):
    _, client, _ = bench([api(name="cached", prompt_cache="1h")])
    body = client.get("/").get_data(as_text=True)
    assert "prompt cache" in body
    assert "reach the vendor" in body


def test_the_bench_detail_says_nothing_stated_rather_than_off(bench):
    """A page reading "caching: off" would be wrong on most of these runtimes."""
    _, client, _ = bench([api(name="plain")])
    body = client.get("/").get_data(as_text=True)
    assert "nothing stated" in body
    assert "promptCache" in body


def test_a_bench_edit_does_not_withdraw_the_request(bench):
    """The accounts form does not edit this field, so it must carry it. An edit
    that dropped it would withdraw a caching request nobody withdrew."""
    _, client, bridge = bench([api(name="claude-api", prompt_cache="1h")])
    client.post(
        "/accounts/save",
        data={
            "original_name": "claude-api",
            "name": "claude-api",
            "runtime": "anthropic-api",
            "credential": "env",
            "api_key_env": "ANTHROPIC_API_KEY",
            "nickname": "Work",
            "quota_action": "stop",
        },
        follow_redirects=True,
    )
    written = bridge.store.get("claude-api")
    assert written.nickname == "Work"
    assert written.prompt_cache == "1h"


def test_editing_a_connection_in_python_keeps_the_request(tmp_path):
    store = ConnectionStore(tmp_path / "modelpass")
    store.add(api(name="cached", prompt_cache="1h"))
    store.add(replace(store.get("cached"), nickname="Work"), overwrite=True)
    assert ConnectionStore(tmp_path / "modelpass").get("cached").prompt_cache == "1h"
