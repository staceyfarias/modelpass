"""``modelpass`` command-line tests -- offline, against FakeAdapter.

The CLI is the surface where a user first decides whether to trust this thing,
so the properties worth pinning are about what it shows and when it writes:
nothing is written before the receipt is printed, a refused confirmation writes
nothing at all, and no credential *value* ever appears in the output.
"""

from __future__ import annotations

import io

import pytest

from modelpass.bridge import Bridge
from modelpass.capabilities import CapabilityRegistry
from modelpass.cli import main
from modelpass.connections import Connection, CredentialRef, Guards, QuotaAction, QuotaPolicy
from modelpass.runtimes import Runtime
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode

SECRET = "sk-ant-this-value-must-never-be-printed"


@pytest.fixture
def cli(store):
    """Run the CLI against a temp store and a scripted adapter, capturing output."""

    def run(
        argv,
        *,
        env=None,
        answer=True,
        adapter=None,
        registry=None,
        stdin=None,
    ) -> tuple[int, str, str]:
        fake = adapter or FakeAdapter([])
        bridge = Bridge(
            store=store,
            registry=registry or CapabilityRegistry(),
            adapters={Runtime.ANTHROPIC_SDK: fake, Runtime.OPENAI_SDK: fake},
            env=dict(env or {}),
        )
        out, err = io.StringIO(), io.StringIO()
        code = main(
            argv,
            bridge=bridge,
            out=out,
            err=err,
            confirm=lambda _question: answer,
            stdin=None if stdin is None else _Stdin(stdin),
        )
        return code, out.getvalue(), err.getvalue()

    return run


class _Stdin:
    """A non-terminal stdin stand-in, so a piped key can be scripted."""

    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text

    def isatty(self) -> bool:
        return False


# --- connect ---------------------------------------------------------------------


def test_connect_prints_the_receipt_and_writes_on_confirmation(cli, store):
    code, out, _ = cli(["connect", "anthropic"])
    assert code == 0
    assert "claude-sub" in out
    assert "anthropic-sdk" in out
    assert "subscription" in out
    assert "status       OK" in out
    assert store.get("claude-sub").runtime is Runtime.ANTHROPIC_SDK


def test_connect_writes_nothing_when_the_confirmation_is_declined(cli, store):
    code, out, _ = cli(["connect", "anthropic"], answer=False)
    assert code != 0
    assert "Nothing was written." in out
    assert store.list() == ()


def test_yes_skips_the_asking_not_the_printing(cli, store):
    code, out, _ = cli(["connect", "anthropic", "--yes"], answer=False)
    assert code == 0
    assert "runtime      anthropic-sdk" in out
    assert store.has("claude-sub")


def test_the_receipt_is_printed_before_anything_is_written(cli, store):
    """A receipt shown after the write would be a report, not a decision point."""
    code, out, _ = cli(["connect", "anthropic"])
    assert code == 0
    assert out.index("status       OK") < out.index("Wrote 'claude-sub'")


def test_a_failed_preflight_writes_nothing(cli, store):
    adapter = FakeAdapter([], ok=False, problem="no Claude Code login found")
    code, out, _ = cli(["connect", "anthropic", "--yes"], adapter=adapter)
    assert code != 0
    assert "no Claude Code login found" in out
    assert "Not writing the connection" in out
    assert store.list() == ()


def test_an_api_key_connection_stores_the_variable_name_never_its_value(cli, store):
    code, out, _ = cli(
        ["connect", "anthropic", "--name", "claude-api", "--api-key-env", "MY_KEY", "--yes"],
        env={"MY_KEY": SECRET},
    )
    assert code == 0
    assert SECRET not in out
    assert "MY_KEY" in out
    connection = store.get("claude-api")
    assert connection.auth_mode is AuthMode.API_KEY
    assert connection.credential_ref.to_str() == "env:MY_KEY"
    assert SECRET not in store.path.read_text(encoding="utf-8")


def test_no_command_ever_echoes_a_credential_value(cli, store):
    """The blunt version of the rule, across every subcommand."""
    env = {"MY_KEY": SECRET, "ANTHROPIC_API_KEY": SECRET}
    cli(["connect", "anthropic", "--name", "k", "--api-key-env", "MY_KEY", "--yes"], env=env)
    for argv in (["list", "--verbose"], ["check"], ["check", "k"]):
        _, out, err = cli(argv, env=env)
        assert SECRET not in out
        assert SECRET not in err


def test_connect_refuses_to_clobber_without_force(cli, store, subscription_connection):
    store.add(subscription_connection)
    code, out, _ = cli(["connect", "anthropic", "--yes"])
    assert code != 0
    assert "--force" in out
    # The original survived, guards and all.
    assert store.get("claude-sub").guards.stop_at_tokens == 250


def test_force_replaces_an_existing_connection(cli, store, subscription_connection):
    store.add(subscription_connection)
    code, _, _ = cli(["connect", "anthropic", "--yes", "--force"])
    assert code == 0
    assert store.get("claude-sub").guards.configured is False


# --- connect: which runtime, and which credential ---------------------------------
#
# One test per cell of the matrix ``--runtime`` completes (ticket 1.9b). The
# three default rows are the ones ticket 1.3 settled and must not move; the rows
# with ``--runtime`` are what was previously unreachable -- an API runtime
# reading a *named environment variable* rather than a pasted key.

OLLAMA = "http://localhost:11434/v1"


def test_no_flags_means_the_vendors_agent_runtime_on_its_own_login(cli, store):
    assert cli(["connect", "anthropic", "--yes"])[0] == 0
    written = store.get("claude-sub")
    assert written.runtime is Runtime.ANTHROPIC_SDK
    assert written.credential_ref.to_str() == "native-login"


def test_api_key_stdin_alone_still_means_the_vendors_api_runtime(cli, store):
    """Ticket 1.3's default: a stored key cannot reach a launched child."""
    assert cli(["connect", "anthropic", "--api-key-stdin", "--yes"], stdin=SECRET)[0] == 0
    written = store.get("claude-api")
    assert written.runtime is Runtime.ANTHROPIC_API
    assert written.credential_ref.to_str() == "secret:claude-api"


def test_api_key_env_alone_still_means_the_agent_runtime_metered(cli, store):
    """Unchanged, and deliberately: a variable *does* reach a launched child."""
    code, _, _ = cli(
        ["connect", "anthropic", "--name", "metered", "--api-key-env", "MY_KEY", "--yes"],
        env={"MY_KEY": SECRET},
    )
    assert code == 0
    written = store.get("metered")
    assert written.runtime is Runtime.ANTHROPIC_SDK
    assert written.auth_mode is AuthMode.API_KEY
    assert written.credential_ref.to_str() == "env:MY_KEY"


def test_runtime_plus_api_key_env_puts_the_variable_on_the_api_runtime(cli, store):
    """The cell that had no command line before this ticket."""
    code, out, _ = cli(
        [
            "connect",
            "anthropic",
            "--runtime",
            "anthropic-api",
            "--api-key-env",
            "MY_KEY",
            "--yes",
        ],
        env={"MY_KEY": SECRET},
    )
    assert code == 0
    written = store.get("claude-api")
    assert written.runtime is Runtime.ANTHROPIC_API
    assert written.credential_ref.to_str() == "env:MY_KEY"
    assert SECRET not in out


def test_runtime_plus_api_key_stdin_on_an_agent_runtime_is_refused(cli, store):
    """Asking for it explicitly does not make it possible -- 1.3's reason, verbatim."""
    code, _, err = cli(
        ["connect", "anthropic", "--runtime", "anthropic-sdk", "--api-key-stdin", "--yes"],
        stdin=SECRET,
    )
    assert code != 0
    assert "'secret:' is only supported by the API runtimes" in err
    assert "use 'env:NAME' there" in err
    assert store.list() == ()


def test_openai_compatible_takes_an_environment_variable_and_a_base_url(cli, store):
    code, _, _ = cli(
        ["connect", "compatible", "--base-url", OLLAMA, "--api-key-env", "MY_KEY", "--yes"],
        env={"MY_KEY": SECRET},
    )
    assert code == 0
    written = store.get("compatible-api")
    assert written.runtime is Runtime.OPENAI_COMPATIBLE
    assert written.credential_ref.to_str() == "env:MY_KEY"
    assert written.base_url == OLLAMA


def test_openai_compatible_takes_a_pasted_key_too(cli, store):
    code, out, _ = cli(
        ["connect", "compatible", "--base-url", OLLAMA, "--api-key-stdin", "--yes"],
        stdin=SECRET,
    )
    assert code == 0
    written = store.get("compatible-api")
    assert written.runtime is Runtime.OPENAI_COMPATIBLE
    assert written.credential_ref.to_str() == "secret:compatible-api"
    assert SECRET not in out


def test_openai_compatible_without_a_base_url_is_refused(cli, store):
    code, _, err = cli(
        ["connect", "compatible", "--api-key-env", "MY_KEY", "--yes"], env={"MY_KEY": SECRET}
    )
    assert code != 0
    assert "requires baseUrl" in err
    assert store.list() == ()


def test_google_api_takes_an_environment_variable(cli, store):
    code, _, _ = cli(
        ["connect", "google", "--api-key-env", "MY_KEY", "--yes"], env={"MY_KEY": SECRET}
    )
    assert code == 0
    written = store.get("gemini-api")
    assert written.runtime is Runtime.GOOGLE_API
    assert written.credential_ref.to_str() == "env:MY_KEY"


def test_google_api_takes_a_pasted_key(cli, store):
    code, out, _ = cli(["connect", "google", "--api-key-stdin", "--yes"], stdin=SECRET)
    assert code == 0
    written = store.get("gemini-api")
    assert written.runtime is Runtime.GOOGLE_API
    assert written.credential_ref.to_str() == "secret:gemini-api"
    assert SECRET not in out


def test_runtime_must_belong_to_the_vendor_on_the_command_line(cli, store):
    code, _, err = cli(["connect", "anthropic", "--runtime", "openai-api", "--yes"])
    assert code != 0
    assert "not a runtime of vendor 'anthropic'" in err
    assert "anthropic-sdk, anthropic-api" in err
    assert store.list() == ()


def test_an_unknown_runtime_is_refused_with_the_vendors_own_list(cli, store):
    code, _, err = cli(["connect", "google", "--runtime", "gemini-pro", "--yes"])
    assert code != 0
    assert "valid runtimes for google: google-cli, google-sdk, google-api" in err


def test_the_receipt_names_the_runtime_and_the_credential_it_is_about_to_write(
    cli, store
):
    """What is printed before "Save this connection?" is what lands in the file."""
    code, out, _ = cli(
        [
            "connect",
            "anthropic",
            "--runtime",
            "anthropic-api",
            "--api-key-env",
            "MY_KEY",
            "--yes",
        ],
        env={"MY_KEY": SECRET},
    )
    assert code == 0
    receipt = out[: out.index("Wrote 'claude-api'")]
    assert "runtime      anthropic-api" in receipt
    assert "auth mode    api_key" in receipt
    assert "credential   " in receipt
    assert "MY_KEY" in receipt


def test_connect_says_loudly_when_no_spend_guards_were_set(cli, store):
    _, out, _ = cli(["connect", "anthropic", "--yes"])
    assert "No spend guards are configured" in out
    assert "does not set a default threshold" in out
    assert "--stop-at-tokens" in out
    assert "docs/guards.md" in out


def test_the_guard_advice_names_no_number(cli, store):
    """It says how to decide, not what to decide (D4 amendment, 2026-08-17)."""
    _, out, _ = cli(["connect", "anthropic", "--yes"])
    advice = out[out.index("No spend guards are configured") :]
    assert "200000" not in advice
    assert "200,000" not in advice
    assert "1000000" not in advice


def test_setting_guards_at_connect_time_silences_the_warning(cli, store):
    code, out, _ = cli(
        [
            "connect",
            "anthropic",
            "--yes",
            "--warn-at-tokens",
            "150000",
            "--stop-at-tokens",
            "400000",
        ]
    )
    assert code == 0
    assert "No spend guards are configured" not in out
    guards = store.get("claude-sub").guards
    assert guards.warn_at_tokens == 150_000
    assert guards.stop_at_tokens == 400_000


def test_an_invalid_guard_pair_is_an_error_not_a_written_connection(cli, store):
    code, _, err = cli(
        ["connect", "anthropic", "--yes", "--warn-at-tokens", "10", "--stop-at-tokens", "5"]
    )
    assert code != 0
    assert "must not exceed" in err
    assert store.list() == ()


def test_connect_openai_uses_the_openai_runtime_and_name(cli, store):
    code, _, _ = cli(["connect", "openai", "--yes"])
    assert code == 0
    assert store.get("codex-sub").runtime is Runtime.OPENAI_SDK


def test_connect_can_pin_a_claude_account_directory(cli, store, tmp_path):
    root = tmp_path / "claude-work"
    code, out, _ = cli(
        ["connect", "anthropic", "--yes", "--config-dir", str(root)]
    )
    assert code == 0
    assert f"configDir    {root}" in out
    assert store.get("claude-sub").config_dir == str(root)


def test_account_nickname_is_stored_and_accounts_command_displays_it(cli, store):
    code, _, _ = cli(
        ["connect", "anthropic", "--yes", "--nickname", "Anthropic Home"]
    )
    assert code == 0
    code, out, _ = cli(["accounts"])
    assert code == 0
    assert "Anthropic Home" in out
    assert "id: claude-sub" in out
    assert "vendor    anthropic" in out
    assert "identity  VERIFIED" in out


def test_connect_saves_unpinned_when_the_vendor_reports_no_identity(cli, store):
    """A silent probe is not a broken account, and must not block setup.

    ``claude auth status`` returns nothing whenever the resolved binary is not a
    plain file, the subprocess times out, or the installed CLI predates the JSON
    command. Refusing to write the connection there would lock out every user
    whose CLI is a shim.
    """
    code, out, _ = cli(
        ["connect", "anthropic", "--yes"],
        adapter=FakeAdapter([], account_profile=None),
    )
    assert code == 0
    assert store.get("claude-sub").account_binding is None
    assert "unpinned" in out
    assert "modelpass verify claude-sub" in out


def test_verify_pins_an_existing_unverified_account(cli, store, subscription_connection):
    store.add(subscription_connection)
    code, out, _ = cli(["verify", subscription_connection.name])
    assert code == 0
    assert "Verified and pinned" in out
    assert store.get(subscription_connection.name).account_binding is not None


def test_accounts_uses_account_language_on_a_fresh_install(cli):
    code, out, _ = cli(["accounts"])
    assert code == 0
    assert "No account profiles configured" in out


def test_allow_env_is_recorded_and_shown(cli, store):
    code, out, _ = cli(
        ["connect", "anthropic", "--yes", "--allow-env", "ANTHROPIC_BASE_URL"],
        env={"ANTHROPIC_BASE_URL": "https://proxy.example/v1"},
    )
    assert code == 0
    assert "allowEnv     ANTHROPIC_BASE_URL" in out
    assert store.get("claude-sub").allow_env == ("ANTHROPIC_BASE_URL",)


def test_allow_env_still_refuses_a_credential(cli, store):
    code, _, err = cli(
        ["connect", "anthropic", "--yes", "--allow-env", "ANTHROPIC_API_KEY"],
        env={"ANTHROPIC_API_KEY": SECRET},
    )
    assert code != 0
    assert "allowEnv may not carry" in err
    assert store.list() == ()


def test_the_scrub_list_is_visible_at_setup_time(cli, store):
    """What this connection will *not* be allowed to see is part of the receipt."""
    _, out, _ = cli(
        ["connect", "anthropic", "--yes"],
        env={"ANTHROPIC_API_KEY": SECRET, "CLAUDE_CODE_USE_BEDROCK": "1"},
    )
    assert "scrubbed     ANTHROPIC_API_KEY, CLAUDE_CODE_USE_BEDROCK" in out
    assert SECRET not in out


# --- list ------------------------------------------------------------------------


def test_list_on_a_fresh_install_says_so_rather_than_printing_nothing(cli):
    code, out, _ = cli(["list"])
    assert code == 0
    assert "No connections configured" in out
    assert "by design" in out


def test_list_summarizes_runtime_auth_mode_and_guards(cli, store, subscription_connection):
    store.add(subscription_connection)
    code, out, _ = cli(["list"])
    assert code == 0
    assert "claude-sub" in out
    assert "anthropic-sdk" in out
    assert "subscription" in out
    assert "warn at 100 tokens" in out
    assert "stop at 250 tokens" in out


def test_list_says_none_when_a_connection_has_no_guards(cli, store):
    store.add(
        Connection(
            name="bare", runtime=Runtime.ANTHROPIC_SDK, auth_mode=AuthMode.SUBSCRIPTION
        )
    )
    _, out, _ = cli(["list"])
    assert "none (nothing bounds one run's spend)" in out


def test_list_shows_a_configured_failover(cli, store):
    store.add(
        Connection(
            name="claude-sub",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            guards=Guards(
                stop_at_tokens=100,
                on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api"),
            ),
        )
    )
    _, out, _ = cli(["list"])
    assert "fail over to 'claude-api'" in out


def test_verbose_list_adds_model_and_notes(cli, store):
    store.add(
        Connection(
            name="claude-sub",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            model="claude-sonnet-5",
            description="my laptop login",
        )
    )
    _, plain, _ = cli(["list"])
    _, verbose, _ = cli(["list", "--verbose"])
    assert "claude-sonnet-5" not in plain
    assert "claude-sonnet-5" in verbose
    assert "my laptop login" in verbose


# --- check -----------------------------------------------------------------------


def test_check_reruns_the_preflight_and_exits_zero_when_healthy(
    cli, store, subscription_connection
):
    store.add(subscription_connection)
    code, out, _ = cli(["check", "claude-sub"])
    assert code == 0
    assert "status       OK" in out


def test_check_names_the_model_or_says_that_nothing_named_one(
    cli, store, subscription_connection
):
    """A model rejection has to be diagnosable from what was printed beforehand
    (first-consumer feedback, 2026-08-17)."""
    store.add(subscription_connection)
    _, out, _ = cli(["check", "claude-sub"])
    assert "model        no model is named for this run" in out

    store.add(
        Connection(
            name="pinned",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            model="claude-opus-4",
        )
    )
    _, out, _ = cli(["check", "pinned"])
    assert "model        claude-opus-4  (from connection 'pinned')" in out


def test_check_exits_non_zero_on_a_failing_preflight(cli, store, subscription_connection):
    store.add(subscription_connection)
    adapter = FakeAdapter([], ok=False, problem="login expired 2026-03-02")
    code, out, _ = cli(["check", "claude-sub"], adapter=adapter)
    assert code == 1
    assert "login expired" in out


def test_check_without_a_name_checks_everything(cli, store, subscription_connection):
    store.add(subscription_connection)
    store.add(
        Connection(
            name="codex-sub", runtime=Runtime.OPENAI_SDK, auth_mode=AuthMode.SUBSCRIPTION
        )
    )
    code, out, _ = cli(["check"])
    assert code == 0
    assert "claude-sub" in out and "codex-sub" in out


def test_one_broken_connection_does_not_hide_the_others(cli, store, subscription_connection):
    store.add(subscription_connection)
    store.add(
        Connection(
            name="needs-a-key",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.API_KEY,
            credential_ref=CredentialRef.parse("env:NOT_SET_ANYWHERE"),
        )
    )
    code, out, _ = cli(["check"])
    assert code == 1
    assert "needs-a-key: FAILED" in out
    # The healthy one still got reported.
    assert "status       OK" in out


def test_check_names_an_unknown_connection_clearly(cli, store):
    code, _, err = cli(["check", "nope"])
    assert code == 1
    assert "no connection named 'nope'" in err


def test_check_on_a_fresh_install_is_not_a_failure(cli):
    code, out, _ = cli(["check"])
    assert code == 0
    assert "nothing to check" in out


def test_check_reports_that_the_guard_state_is_part_of_being_healthy(
    cli, store, subscription_connection
):
    store.add(subscription_connection.with_guards(Guards()))
    _, out, _ = cli(["check", "claude-sub"])
    assert "no spend guards configured" in out


# --- ticket 1.14: the whole surface, every runtime, both credential forms --------
#
# The commands below mostly existed before this ticket and were tested against
# two runtimes; `remove` and `secrets` are new here. What the matrix pins is
# that a user who configured an API runtime in 1.9-1.11 can do every lifecycle
# verb to it, and that the two gated runtimes refuse where a user meets them.

#: The metered form every API runtime needs before it will preflight at all: a
#: named variable, set in the environment the bridge is given.
KEY_ENV = ["--api-key-env", "MY_KEY"]

#: The environment that variable is set in, for the tests that use the matrix.
KEY_SET = {"MY_KEY": SECRET}

#: Every runtime a user can reach from `connect`. The two Google agent runtimes
#: are reachable by name and refused by the gate (D5), which is its own test
#: below rather than a row here.
CONNECT_MATRIX = (
    ("anthropic-sdk", ["connect", "anthropic"]),
    ("anthropic-api", ["connect", "anthropic", "--runtime", "anthropic-api", *KEY_ENV]),
    ("openai-sdk", ["connect", "openai"]),
    ("openai-api", ["connect", "openai", "--runtime", "openai-api", *KEY_ENV]),
    ("google-api", ["connect", "google", *KEY_ENV]),
    ("openai-compatible", ["connect", "compatible", "--base-url", OLLAMA]),
)

API_RUNTIME_MATRIX = tuple(
    (runtime, [part for part in argv if part not in KEY_ENV])
    for runtime, argv in CONNECT_MATRIX
    if runtime.endswith("-api") or runtime.endswith("-compatible")
)


def _connect_args(argv, *, name, form):
    """The same connect invocation with each of the two credential forms."""
    if form == "env":
        return [*argv, "--name", name, "--api-key-env", "MY_KEY", "--yes"]
    return [*argv, "--name", name, "--api-key-stdin", "--yes"]


def _fields(connection):
    """A connection's own fields, for building a second one pointing at its key."""
    from dataclasses import fields

    return {f.name: getattr(connection, f.name) for f in fields(connection)}


@pytest.mark.parametrize(("runtime", "argv"), API_RUNTIME_MATRIX)
@pytest.mark.parametrize("form", ["env", "stdin"])
def test_connect_covers_every_api_runtime_in_both_credential_forms(
    cli, store, runtime, argv, form
):
    code, out, err = cli(
        _connect_args(argv, name="acc", form=form),
        env={"MY_KEY": SECRET},
        stdin=SECRET,
    )
    assert code == 0, err
    written = store.get("acc")
    assert written.runtime.value == runtime
    assert written.auth_mode is AuthMode.API_KEY
    assert written.credential_ref.to_str() == (
        "env:MY_KEY" if form == "env" else "secret:acc"
    )
    assert SECRET not in out


@pytest.mark.parametrize(("runtime", "argv"), CONNECT_MATRIX)
def test_check_covers_every_runtime_connect_can_write(cli, store, runtime, argv):
    code, _, err = cli([*argv, "--name", "acc", "--yes"], env=KEY_SET)
    assert code == 0, err
    _, out, _ = cli(["check", "acc"], env=KEY_SET)
    assert f"runtime      {runtime}" in out
    assert "status" in out


@pytest.mark.parametrize(("runtime", "argv"), CONNECT_MATRIX)
def test_list_covers_every_runtime_connect_can_write(cli, store, runtime, argv):
    assert cli([*argv, "--name", "acc", "--yes"], env=KEY_SET)[0] == 0
    for command in ("list", "accounts"):
        _, out, _ = cli([command, "--verbose"])
        assert f"runtime   {runtime}" in out
        assert "timeout   unbounded" in out


@pytest.mark.parametrize(("runtime", "argv"), CONNECT_MATRIX)
def test_rename_covers_every_runtime_connect_can_write(cli, store, runtime, argv):
    assert cli([*argv, "--name", "acc", "--yes"], env=KEY_SET)[0] == 0
    code, out, _ = cli(["rename", "acc", "renamed"])
    assert code == 0
    assert "Renamed 'acc' to 'renamed'" in out
    assert store.get("renamed").runtime.value == runtime
    assert not store.has("acc")


def test_the_listing_shows_the_endpoint_a_compatible_connection_talks_to(cli, store):
    assert cli(["connect", "compatible", "--base-url", OLLAMA, "--yes"])[0] == 0
    _, out, _ = cli(["list"])
    assert f"baseUrl   {OLLAMA}" in out


def test_the_listing_says_when_a_connection_refuses_to_be_retried(cli, store):
    from dataclasses import replace as _replace

    assert cli(["connect", "anthropic", "--yes"])[0] == 0
    store.add(_replace(store.get("claude-sub"), retry="never"), overwrite=True)
    _, out, _ = cli(["list", "--verbose"])
    assert "retry     never" in out


@pytest.mark.parametrize(
    ("runtime", "credential"),
    # Each in the only billing mode it has: google-cli is the subscription CLI,
    # google-sdk is metered. The gate is checked after the mode, so asking the
    # wrong way would test the mode error instead of the gate.
    [("google-cli", []), ("google-sdk", KEY_ENV)],
)
def test_the_two_gated_google_runtimes_refuse_at_connect_time(
    cli, store, runtime, credential
):
    """Reachable by name, refused by the gate -- and the refusal names the key."""
    code, _, err = cli(
        ["connect", "google", "--runtime", runtime, *credential, "--yes"], env=KEY_SET
    )
    assert code != 0
    assert f"runtime {runtime!r} is gated" in err
    assert "experimental = true" in err
    assert store.list() == ()


# --- remove ----------------------------------------------------------------------


@pytest.mark.parametrize(("runtime", "argv"), CONNECT_MATRIX)
def test_remove_deletes_any_runtimes_connection(cli, store, runtime, argv):
    assert cli([*argv, "--name", "acc", "--yes"], env=KEY_SET)[0] == 0
    code, out, _ = cli(["remove", "acc"], env=KEY_SET)
    assert code == 0
    assert f"runtime      {runtime}" in out
    assert "Deleted 'acc'" in out
    assert store.list() == ()


def test_remove_prints_what_goes_before_it_asks(cli, store):
    assert cli(["connect", "anthropic", "--warn-at-tokens", "10", "--yes"])[0] == 0
    _, out, _ = cli(["remove", "claude-sub"])
    assert out.index("warn at 10 tokens") < out.index("Deleted 'claude-sub'")


def test_a_declined_removal_deletes_nothing(cli, store):
    assert cli(["connect", "anthropic", "--yes"])[0] == 0
    code, out, _ = cli(["remove", "claude-sub"], answer=False)
    assert code != 0
    assert "Nothing was deleted." in out
    assert store.has("claude-sub")


def test_remove_takes_the_connections_own_key_with_it(cli, store):
    code, _, err = cli(["connect", "anthropic", "--api-key-stdin", "--yes"], stdin=SECRET)
    assert code == 0, err
    code, out, _ = cli(["remove", "claude-api", "--yes"])
    assert code == 0
    assert "entry 'claude-api' goes with it" in out
    _, listing, _ = cli(["secrets"])
    assert "claude-api" not in listing
    assert SECRET not in out + listing


def test_remove_leaves_a_key_two_connections_share(cli, store):
    """The 1.4 rule, surfaced: an orphan is recoverable and a deleted key is not."""
    assert cli(["connect", "anthropic", "--api-key-stdin", "--yes"], stdin=SECRET)[0] == 0
    shared = store.get("claude-api")
    store.add(Connection(**{**_fields(shared), "name": "second"}))
    code, out, _ = cli(["remove", "claude-api", "--yes"])
    assert code == 0
    assert "stays" in out
    assert "second" in out
    assert SECRET not in out


def test_remove_names_an_unknown_connection_clearly(cli, store):
    code, _, err = cli(["remove", "nope", "--yes"])
    assert code != 0
    assert "nope" in err


def test_remove_on_an_empty_store_says_so_rather_than_tracing_back(cli):
    code, _, err = cli(["remove", "anything", "--yes"])
    assert code != 0
    assert "Traceback" not in err
    assert "anything" in err


# --- secrets ---------------------------------------------------------------------


def test_secrets_on_an_install_with_no_secrets_file_is_not_a_failure(cli):
    code, out, _ = cli(["secrets"])
    assert code == 0
    assert "No secrets file" in out
    assert "normal state" in out


def test_secrets_lists_entries_and_who_references_them_never_a_value(cli, store):
    assert cli(["connect", "anthropic", "--api-key-stdin", "--yes"], stdin=SECRET)[0] == 0
    code, out, _ = cli(["secrets"])
    assert code == 0
    assert "claude-api" in out
    assert "used by  claude-api" in out
    assert SECRET not in out
    assert "No value is printed by this command" in out


def test_secrets_names_an_orphan_without_deleting_it(cli, store):
    from modelpass.secrets import SecretStore

    assert cli(["connect", "anthropic", "--api-key-stdin", "--yes"], stdin=SECRET)[0] == 0
    SecretStore(store.root).set("left-behind", SECRET)
    code, out, _ = cli(["secrets"])
    assert code == 0
    assert "left-behind" in out
    assert "an orphan" in out
    assert "never deleted for you" in out
    assert SecretStore(store.root).has("left-behind")
    assert SECRET not in out


def test_verbose_secrets_reports_the_permissions_that_were_actually_applied(cli, store):
    assert cli(["connect", "anthropic", "--api-key-stdin", "--yes"], stdin=SECRET)[0] == 0
    code, out, _ = cli(["secrets", "--verbose"])
    assert code == 0
    assert "permissions" in out
    assert SECRET not in out


def test_an_unreadable_secrets_file_is_reported_rather_than_shown_empty(cli, store):
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "secrets.toml").write_text("this is not = = toml", encoding="utf-8")
    code, out, _ = cli(["secrets"])
    assert code == 0
    assert "note" in out


# --- verify, and what check says about a driven endpoint -------------------------


def test_check_says_an_unverified_compatible_endpoint_has_not_been_driven(cli, store):
    code, _, err = cli(["connect", "compatible", "--base-url", OLLAMA, "--yes"])
    assert code == 0, err
    _, out, _ = cli(["check", "compatible-api"])
    assert "verified" in out
    assert "modelpass verify compatible-api" in out


def test_check_reports_the_cells_a_drive_recorded(cli, store):
    from dataclasses import replace as _replace

    from modelpass.capabilities import VerifiedCapabilities

    assert cli(["connect", "compatible", "--base-url", OLLAMA, "--yes"])[0] == 0
    store.add(
        _replace(
            store.get("compatible-api"),
            verified_capabilities=VerifiedCapabilities(
                supported=("chat", "streaming"),
                unsupported=("structured_output",),
                checked_at="2026-09-13T00:00:00+00:00",
                models=("llama3.1",),
            ),
        ),
        overwrite=True,
    )
    _, out, _ = cli(["check", "compatible-api"])
    assert "chat, streaming supported" in out
    assert "structured_output unsupported" in out
    assert "2026-09-13" in out
    assert "llama3.1" in out


def test_verify_refuses_an_api_key_connection_with_no_identity_to_pin(cli, store):
    code, _, err = cli(
        [
            "connect",
            "anthropic",
            "--runtime",
            "anthropic-api",
            "--api-key-env",
            "MY_KEY",
            "--yes",
        ],
        env={"MY_KEY": SECRET},
    )
    assert code == 0, err
    code, out, _ = cli(["verify", "claude-api"], env={"MY_KEY": SECRET})
    assert code != 0
    assert "no subscription identity to pin" in out


def test_verify_names_an_unknown_connection_clearly(cli):
    code, _, err = cli(["verify", "nope"])
    assert code != 0
    assert "nope" in err


# --- R10: every read path degrades honestly --------------------------------------


@pytest.mark.parametrize("argv", [["list"], ["accounts"], ["check"], ["secrets"]])
def test_every_read_command_treats_an_empty_store_as_legitimate(cli, argv):
    code, out, _ = cli(argv)
    assert code == 0
    assert out.strip()


@pytest.mark.parametrize(
    "argv",
    [["list"], ["accounts"], ["check"], ["secrets"], ["remove", "x", "--yes"]],
)
def test_an_unreadable_store_is_reported_rather_than_crashed(cli, store, argv):
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text("[connections\nbroken = ", encoding="utf-8")
    _code, out, err = cli(argv)
    text = out + err
    assert "Traceback" not in text
    assert text.strip()


# --- bench -----------------------------------------------------------------------


def test_bench_without_flask_says_so_before_announcing_a_url(cli, monkeypatch):
    """A banner naming a URL is a claim that something is listening on it.

    The Flask import used to happen inside ``serve``, one line after the
    announcement, so the missing extra was reported underneath the address of a
    server that never came up.
    """
    from modelpass import bench
    from modelpass.errors import SubpassError

    def missing():
        raise SubpassError("the bench needs Flask ... 'pip install modelpass[bench]'")

    monkeypatch.setattr(bench, "require_flask", missing)
    code, out, err = cli(["bench"])
    assert code == 1
    assert out == ""
    assert "modelpass[bench]" in err
