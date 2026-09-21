"""The per-connection input window (``maxInputTokens``, 2026-09-21).

A connection may record how big the model's usable input window is, in tokens,
**because a person said so**. A retrieval evaluation harness asked for it so it
can decide whether a payload will fit before spending a call finding out.

Everything this file defends is about the *absence* of a number rather than the
presence of one:

* Unset is the default, it is valid, and it means **unknown** -- not "small",
  not "the vendor's published figure", not zero. There is deliberately no table
  of model names to window sizes anywhere in modelpass: a vendor moves a window
  without moving the model string, so a table is a stale number that reads as
  authoritative, and a wrong window is worse than no window.
* Unknown and zero are different facts. Zero is refused at construction, so a
  stored window is always a positive integer somebody chose and ``None`` is
  always "nobody has said".
* A file written before this key existed keeps working, unchanged, and stays
  unchanged when this build rewrites it.

Nothing in modelpass reads the value. It bounds no call and truncates nothing;
there is no test here asserting that it does, because asserting a mechanism
would be the first step towards building one nobody asked for.
"""

from __future__ import annotations

import io
import tomllib
from dataclasses import replace

import pytest

from modelpass.bridge import Bridge
from modelpass.capabilities import CapabilityRegistry
from modelpass.cli import main
from modelpass.connections import Connection, CredentialRef
from modelpass.errors import InvalidConnection
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore, connection_from_dict, connection_to_dict
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode


def api(name: str = "claude-api", **kwargs) -> Connection:
    kwargs.setdefault("credential_ref", CredentialRef.parse("env:ANTHROPIC_API_KEY"))
    return Connection(
        name=name,
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        **kwargs,
    )


def _bridge(store) -> Bridge:
    """Offline, against the fake adapter -- no vendor package, no credential."""
    return Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_API: FakeAdapter(runtime=Runtime.ANTHROPIC_API)},
        env={"ANTHROPIC_API_KEY": "sk-not-a-real-key"},
    )


@pytest.fixture
def cli(store):
    """Run the CLI against a temp store, capturing what it printed."""

    def run(argv) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = main(argv, bridge=_bridge(store), out=out, err=err, confirm=lambda _q: True)
        return code, out.getvalue(), err.getvalue()

    return run


@pytest.fixture
def bench(store):
    """The bench app over a temp store. Skips where flask is not installed."""

    def make(connections=()):
        pytest.importorskip("flask")
        from modelpass.bench import app as flask_app
        for connection in connections:
            store.add(connection, overwrite=True)
        bridge = _bridge(store)
        app = flask_app.create_app(bridge=bridge)
        app.config.update(TESTING=True)
        return app, app.test_client(), bridge

    return make


# --- 1. unset is the default, and it is a fact of its own -------------------------


def test_a_connection_that_says_nothing_has_an_unknown_window():
    assert api().max_input_tokens is None


def test_unset_is_none_and_not_zero():
    """The distinction the whole field turns on.

    ``None`` is "nobody has told modelpass", and a falsy check that collapsed it
    to ``0`` would let a consumer read "the window is nothing" out of "the window
    is unrecorded" -- which is the reading that makes a payload look too big for
    every unconfigured connection on the machine.
    """
    unknown = api().max_input_tokens
    assert unknown is None
    assert unknown != 0
    assert not isinstance(unknown, int)


def test_a_stated_window_reads_back_exactly():
    assert api(max_input_tokens=200_000).max_input_tokens == 200_000


def test_one_is_a_legal_window():
    """The smallest thing a person can truthfully say. Only zero is the refusal."""
    assert api(max_input_tokens=1).max_input_tokens == 1


# --- 2. validation ----------------------------------------------------------------


def test_zero_is_refused_rather_than_stored():
    with pytest.raises(InvalidConnection, match="maxInputTokens must be greater than zero"):
        api(max_input_tokens=0)


def test_the_refusal_says_to_omit_the_key_rather_than_offering_a_default():
    """No default is the design, so the refusal has to say what to do instead."""
    with pytest.raises(InvalidConnection) as caught:
        api(max_input_tokens=0)
    message = str(caught.value)
    assert "Omit the key" in message
    assert "will not guess" in message


def test_a_negative_window_is_refused():
    with pytest.raises(InvalidConnection, match="greater than zero"):
        api(max_input_tokens=-1)


def test_a_fractional_window_is_refused():
    with pytest.raises(InvalidConnection, match="whole number of tokens"):
        api(max_input_tokens=128_000.5)


def test_a_whole_float_is_still_refused():
    """``128000.0`` is a number of tokens nobody can have half of.

    Accepting it would mean the field silently changes type depending on how the
    file was written, and a consumer comparing an integer payload length against
    it would be comparing against a float on some machines and not others.
    """
    with pytest.raises(InvalidConnection, match="whole number of tokens"):
        api(max_input_tokens=128_000.0)


def test_a_string_window_is_refused_even_when_it_looks_numeric():
    with pytest.raises(InvalidConnection, match="whole number of tokens"):
        api(max_input_tokens="200000")


def test_a_non_numeric_string_is_refused():
    with pytest.raises(InvalidConnection, match="whole number of tokens"):
        api(max_input_tokens="big")


def test_true_is_not_a_window_of_one():
    """``bool`` is an ``int`` in Python, and ``maxInputTokens = true`` in a
    hand-edited file would otherwise be stored as a one-token window."""
    with pytest.raises(InvalidConnection, match="whole number of tokens"):
        api(max_input_tokens=True)


def test_the_refusal_is_the_repositorys_existing_error_class():
    with pytest.raises(InvalidConnection):
        api(max_input_tokens=0)


def test_the_refusal_names_the_connection():
    with pytest.raises(InvalidConnection, match="connection 'desk'"):
        api(name="desk", max_input_tokens=0)


# --- 3. persistence ---------------------------------------------------------------


def test_the_store_round_trips_the_window(tmp_path):
    store = ConnectionStore(tmp_path / "modelpass")
    store.add(api(name="sized", max_input_tokens=200_000))
    reread = ConnectionStore(tmp_path / "modelpass").get("sized")
    assert reread.max_input_tokens == 200_000


def test_the_round_trip_keeps_it_an_integer(tmp_path):
    store = ConnectionStore(tmp_path / "modelpass")
    store.add(api(name="sized", max_input_tokens=128_000))
    reread = ConnectionStore(tmp_path / "modelpass").get("sized")
    assert isinstance(reread.max_input_tokens, int)
    assert not isinstance(reread.max_input_tokens, bool)


def test_a_connection_with_no_window_writes_no_key():
    """An absent key is how the file says "unknown".

    ``maxInputTokens = 0`` in the file would read as a deliberate setting, which
    is the same reason there is no ``[guards]`` table on a connection nobody
    guarded.
    """
    assert "maxInputTokens" not in connection_to_dict(api())


def test_a_stated_window_reaches_the_file_under_its_camelcase_key(tmp_path):
    store = ConnectionStore(tmp_path / "modelpass")
    store.add(api(name="sized", max_input_tokens=200_000))
    with store.path.open("rb") as handle:
        raw = tomllib.load(handle)
    assert raw["connections"]["sized"]["maxInputTokens"] == 200_000


def test_an_unset_window_survives_a_rewrite_as_unset(tmp_path):
    """Saving a second connection must not invent a window for the first."""
    store = ConnectionStore(tmp_path / "modelpass")
    store.add(api(name="first"))
    store.add(api(name="second", max_input_tokens=64_000))
    reread = ConnectionStore(tmp_path / "modelpass")
    assert reread.get("first").max_input_tokens is None
    assert reread.get("second").max_input_tokens == 64_000


def test_editing_a_connection_keeps_its_window(tmp_path):
    store = ConnectionStore(tmp_path / "modelpass")
    store.add(api(name="sized", max_input_tokens=200_000))
    store.add(replace(store.get("sized"), nickname="Work"), overwrite=True)
    reread = ConnectionStore(tmp_path / "modelpass").get("sized")
    assert reread.nickname == "Work"
    assert reread.max_input_tokens == 200_000


# --- 4. a file that predates the key ----------------------------------------------


_OLDER = """\
version = 1

[connections.claude-sub]
runtime = "anthropic-sdk"
authMode = "subscription"
credentialRef = "native-login"
model = "sonnet"
timeoutSeconds = 120
"""


def test_a_connection_written_before_the_key_existed_loads(store):
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(_OLDER, encoding="utf-8")
    connection = store.get("claude-sub")
    assert connection.max_input_tokens is None
    assert connection.model == "sonnet"
    assert connection.timeout_seconds == 120.0


def test_a_file_that_predates_the_key_is_not_given_one_when_rewritten(store):
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(_OLDER, encoding="utf-8")
    store.add(api(name="added"))
    with store.path.open("rb") as handle:
        raw = tomllib.load(handle)
    assert "maxInputTokens" not in raw["connections"]["claude-sub"]
    assert "maxInputTokens" not in raw["connections"]["added"]


def test_the_on_disk_reader_treats_a_missing_key_as_unknown():
    connection = connection_from_dict(
        "claude-sub",
        {"runtime": "anthropic-sdk", "authMode": "subscription", "credentialRef": "native-login"},
    )
    assert connection.max_input_tokens is None


def test_a_hand_edited_zero_in_the_file_is_refused_not_read_as_unset():
    with pytest.raises(InvalidConnection, match="greater than zero"):
        connection_from_dict(
            "claude-sub",
            {
                "runtime": "anthropic-sdk",
                "authMode": "subscription",
                "credentialRef": "native-login",
                "maxInputTokens": 0,
            },
        )


def test_the_key_is_a_known_key_and_not_carried_as_unrecognised(store):
    """If it were unknown to the store it would be preserved *and reported*,
    which would put a note in front of every user who set one."""
    from modelpass.store import unknown_connection_keys

    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(_OLDER + 'maxInputTokens = 200000\n', encoding="utf-8")
    assert unknown_connection_keys({"maxInputTokens": 200_000}) == ()
    assert store.compatibility().notes() == ()


# --- 5. the surfaces --------------------------------------------------------------


def test_the_verbose_listing_states_the_window(cli, store):
    store.add(api(name="sized", max_input_tokens=200_000))
    _, out, _ = cli(["list", "--verbose"])
    assert "window    200,000 input tokens (as configured)" in out


def test_the_verbose_listing_calls_an_unset_window_unknown(cli, store):
    """Printed, not omitted: a blank row reads as "fine" to somebody sizing a
    payload, and modelpass genuinely does not know."""
    store.add(api(name="plain"))
    _, out, _ = cli(["list", "--verbose"])
    assert "window    unknown" in out


def test_the_bench_detail_shows_the_window(bench):
    _, client, _ = bench([api(name="sized", max_input_tokens=200_000)])
    body = client.get("/").get_data(as_text=True)
    assert "input window" in body
    assert "200,000 tokens" in body


def test_the_bench_detail_says_unknown_when_nothing_is_set(bench):
    _, client, _ = bench([api(name="plain")])
    body = client.get("/").get_data(as_text=True)
    assert "maxInputTokens" in body


def test_a_bench_edit_does_not_drop_the_window(bench):
    """The accounts form does not edit this field, so it must carry it.

    An edit that dropped it would turn a window somebody measured back into an
    unknown one, silently, on a page whose visible change was a nickname.
    """
    _, client, bridge = bench([api(name="claude-api", max_input_tokens=200_000)])
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
    assert written.max_input_tokens == 200_000
