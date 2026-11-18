"""Series and versions in a model catalogue (2026-11-18).

Three behaviours, all offline:

1. A catalogue shows the newest version of each series by default, and the
   whole list on request. The grouping is modelpass's own reading of the id
   string, so the property that matters most is timidity: an id it cannot read
   is never hidden, and a build tag is never mistaken for a newer release.
2. A name -- a series, optionally a version, or an exact id -- picks one model,
   and a miss says what the list does offer.
3. ``Bridge.with_model`` turns that pick plus a reasoning level into an
   ordinary in-memory connection, defaulting to ``medium`` only where the
   runtime has an effort control.

The ids are the ones the vendors listed on 2026-11-18: the Codex cache written
by codex 0.169.0, and ``anthropic-api``'s models endpoint.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as Obj

import pytest

from modelpass import Bridge, Connection, CredentialRef, NoSuchModel, Runtime
from modelpass.adapters.anthropic_api import AnthropicAPIAdapter
from modelpass.errors import InvalidConnection
from modelpass.models import ModelCatalogue, ModelChoice, parse_model_id
from modelpass.store import ConnectionStore

GPT = (
    "gpt-6.2-sol", "gpt-6.2-astra", "gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-astra",
    "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
)
CLAUDE = (
    "claude-fable-5-2", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
    "claude-fable-5-1", "claude-fable-5", "claude-opus-5", "claude-sonnet-5",
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6",
    "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
)


def catalogue(ids, *, runtime=Runtime.OPENAI_SDK, exact=True, problem=None):
    return ModelCatalogue(
        runtime=runtime,
        choices=tuple(ModelChoice(id=i, label=i) for i in ids),
        source="test list",
        exact=exact,
        problem=problem,
    )


def ids(cat):
    return [c.id for c in cat.choices]


# --- 1. reading an id --------------------------------------------------------------


@pytest.mark.parametrize("model_id,series,version,snapshot", [
    ("gpt-6.2-sol", "gpt-sol", (6, 2), ""),
    ("gpt-6-sol", "gpt-sol", (6,), ""),
    ("gpt-5.4-mini", "gpt-mini", (5, 4), ""),
    ("gpt-5", "gpt", (5,), ""),
    ("gpt-5-2025-08-07", "gpt", (5,), "20250807"),
    ("claude-opus-5-5", "claude-opus", (5, 5), ""),
    ("claude-opus-4-5-20251101", "claude-opus", (4, 5), "20251101"),
    ("claude-3-5-sonnet-20241022", "claude-sonnet", (3, 5), "20241022"),
    ("gemini-2.5-flash", "gemini-flash", (2, 5), ""),
    ("gemini-2.5-flash-lite", "gemini-flash-lite", (2, 5), ""),
])
def test_an_id_reads_as_series_version_and_snapshot(model_id, series, version, snapshot):
    name = parse_model_id(model_id)
    assert (name.series, name.version, name.snapshot) == (series, version, snapshot)


@pytest.mark.parametrize("model_id", ["gpt-4o", "o4-mini", "llama3.1", "fable", "5", ""])
def test_an_id_with_no_version_token_belongs_to_no_series(model_id):
    assert parse_model_id(model_id).series == ""


def test_a_build_tag_is_not_a_version():
    """``gpt-4-0613`` is gpt 4, build 0613 -- not version 4.613, and not the
    same series as plain ``gpt-4``, so neither can hide the other."""
    assert parse_model_id("gpt-4-0613").series == "gpt-0613"
    assert parse_model_id("gemini-2.0-flash-001").series == "gemini-flash-001"


# --- 2. latest by default, all on request -------------------------------------------


def test_latest_keeps_the_newest_of_each_gpt_series_in_the_vendors_order():
    latest = catalogue(GPT).latest()
    assert ids(latest) == ["gpt-6.2-sol", "gpt-6.2-astra", "gpt-6.1-terra", "gpt-6-luna"]
    assert "6 older versions not shown" in latest.notes[-1]
    assert "versions='all'" in latest.notes[-1]


def test_latest_keeps_the_newest_of_each_claude_series():
    latest = catalogue(CLAUDE, runtime=Runtime.ANTHROPIC_API).latest()
    assert ids(latest) == [
        "claude-fable-5-2", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
    ]


def test_latest_never_hides_an_id_it_cannot_read():
    mixed = ("gpt-4o", "gpt-6-sol", "o4-mini", "gpt-6.2-sol", "text-embedding-3-small")
    assert ids(catalogue(mixed).latest()) == [
        "gpt-4o", "o4-mini", "gpt-6.2-sol", "text-embedding-3-small",
    ]


def test_at_one_version_the_undated_id_wins_then_the_newest_snapshot():
    assert ids(catalogue(("claude-opus-4-5-20251101", "claude-opus-4-5")).latest()) == [
        "claude-opus-4-5"
    ]
    assert ids(catalogue(("gpt-5-2025-08-07", "gpt-5-2025-10-01")).latest()) == [
        "gpt-5-2025-10-01"
    ]


def test_latest_with_nothing_to_hide_is_the_same_catalogue():
    aliases = catalogue(("fable", "sonnet", "opus", "haiku"), exact=False)
    assert aliases.latest() is aliases


def test_latest_keeps_the_source_the_flags_and_the_problem():
    partial = catalogue(GPT, problem="The model list is partial")
    latest = partial.latest()
    assert (latest.source, latest.exact, latest.problem) == (
        partial.source, partial.exact, partial.problem,
    )


def test_series_and_versions_list_what_is_there():
    cat = catalogue(GPT)
    assert cat.series() == ("gpt-sol", "gpt-astra", "gpt-terra", "gpt-luna")
    assert ids_of(cat.versions("sol")) == [
        "gpt-6.2-sol", "gpt-6.1-sol", "gpt-6-sol", "gpt-5.6-sol",
    ]
    assert cat.versions("gpt-sol") == cat.versions("sol")


def ids_of(choices):
    return [c.id for c in choices]


# --- 3. picking one ----------------------------------------------------------------


@pytest.mark.parametrize("name,version,expected", [
    ("sol", None, "gpt-6.2-sol"),
    ("SOL", None, "gpt-6.2-sol"),
    ("gpt-sol", None, "gpt-6.2-sol"),
    ("sol", "6.1", "gpt-6.1-sol"),
    ("sol", "6", "gpt-6-sol"),
    ("sol", 6, "gpt-6-sol"),
    ("sol", 5.6, "gpt-5.6-sol"),
    ("sol", "v6.1", "gpt-6.1-sol"),
    ("terra", None, "gpt-6.1-terra"),
    ("gpt-6-astra", None, "gpt-6-astra"),
])
def test_a_name_and_an_optional_version_pick_one_gpt_model(name, version, expected):
    assert catalogue(GPT).resolve(name, version).id == expected


@pytest.mark.parametrize("name,version,expected", [
    ("opus", None, "claude-opus-5-5"),
    ("opus", "5", "claude-opus-5"),
    ("opus", "4.5", "claude-opus-4-5-20251101"),
    ("opus", "4-8", "claude-opus-4-8"),
    ("fable", None, "claude-fable-5-2"),
])
def test_a_name_and_an_optional_version_pick_one_claude_model(name, version, expected):
    assert catalogue(CLAUDE).resolve(name, version).id == expected


def test_a_version_the_series_does_not_have_says_which_it_does():
    with pytest.raises(NoSuchModel, match=r"no version 7\.0.*6\.2, 6\.1, 6, 5\.6"):
        catalogue(GPT).resolve("sol", "7.0")


def test_a_version_is_matched_exactly_not_by_prefix():
    """``6`` is the model called 6, not "the newest 6.x"."""
    with pytest.raises(NoSuchModel, match="no version 6;"):
        catalogue(GPT).resolve("terra", "6")


def test_an_unknown_name_on_the_vendors_own_list_is_refused_with_the_series():
    with pytest.raises(NoSuchModel, match="gpt-sol, gpt-astra, gpt-terra, gpt-luna"):
        catalogue(GPT).resolve("opus")


def test_a_short_name_that_fits_two_series_is_refused():
    with pytest.raises(NoSuchModel, match="more than one series"):
        catalogue(("gpt-5.4-mini", "codex-5-mini")).resolve("mini")


def test_an_exact_id_with_a_version_is_refused():
    with pytest.raises(NoSuchModel, match="already an exact model id"):
        catalogue(GPT).resolve("gpt-6-sol", "6.1")


@pytest.mark.parametrize("version", ["", "latest", "six", "6.x"])
def test_a_version_that_is_not_numbers_is_refused(version):
    with pytest.raises(NoSuchModel, match="version must look like"):
        catalogue(GPT).resolve("sol", version)


def test_a_list_that_is_not_the_vendors_passes_an_unknown_name_through_as_typed():
    """The alias table cannot say ``claude-opus-5`` is wrong, so it does not."""
    aliases = catalogue(("fable", "sonnet", "opus", "haiku"), exact=False)
    assert aliases.resolve("opus").id == "opus"
    assert aliases.resolve("claude-opus-5").id == "claude-opus-5"


def test_a_list_that_is_not_the_vendors_will_not_assemble_a_versioned_id():
    aliases = catalogue(("fable", "sonnet", "opus", "haiku"), exact=False)
    for name in ("opus", "claude-opus"):
        with pytest.raises(NoSuchModel, match="Pass the exact model id as the name"):
            aliases.resolve(name, "5")


def test_no_such_model_is_a_value_error():
    assert issubclass(NoSuchModel, ValueError)


# --- 4. Bridge.models(versions=) ----------------------------------------------------


def codex_connection(tmp_path, slugs=GPT, **extra):
    root = tmp_path / "codex"
    root.mkdir(exist_ok=True)
    (root / "models_cache.json").write_text(
        json.dumps({"models": [{"slug": slug} for slug in slugs]})
    )
    return Connection(name="codex", runtime=Runtime.OPENAI_SDK,
                      auth_mode="subscription", config_dir=str(root), **extra)


def offline_bridge(tmp_path, **kwargs):
    return Bridge(store=ConnectionStore(tmp_path / "config"), env={}, **kwargs)


def test_bridge_models_shows_the_latest_by_default_and_everything_on_request(tmp_path):
    bridge = offline_bridge(tmp_path)
    connection = codex_connection(tmp_path)
    assert ids(bridge.models(connection)) == [
        "gpt-6.2-sol", "gpt-6.2-astra", "gpt-6.1-terra", "gpt-6-luna",
    ]
    assert ids(bridge.models(connection, versions="latest")) == ids(bridge.models(connection))
    assert ids(bridge.models(connection, versions="all")) == list(GPT)


def test_bridge_models_refuses_an_unknown_versions_word(tmp_path):
    with pytest.raises(ValueError, match="latest, all"):
        offline_bridge(tmp_path).models(codex_connection(tmp_path), versions="newest")


# --- 5. Bridge.with_model -----------------------------------------------------------


def test_with_model_picks_the_newest_and_defaults_reasoning_to_medium(tmp_path):
    bridge = offline_bridge(tmp_path)
    picked = bridge.with_model(codex_connection(tmp_path), "sol")
    assert (picked.model, picked.reasoning, picked.name) == ("gpt-6.2-sol", "medium", "codex")


def test_with_model_takes_a_version_and_a_reasoning_level(tmp_path):
    bridge = offline_bridge(tmp_path)
    picked = bridge.with_model(
        codex_connection(tmp_path), "sol", version="6.1", reasoning="high"
    )
    assert (picked.model, picked.reasoning) == ("gpt-6.1-sol", "high")


def test_with_model_reasoning_none_states_nothing(tmp_path):
    bridge = offline_bridge(tmp_path)
    assert bridge.with_model(codex_connection(tmp_path), "sol", reasoning=None).reasoning is None


def test_with_model_overrides_what_the_connection_already_said(tmp_path):
    bridge = offline_bridge(tmp_path)
    connection = codex_connection(tmp_path, model="gpt-5.6-luna", reasoning="low")
    picked = bridge.with_model(connection, "terra")
    assert (picked.model, picked.reasoning) == ("gpt-6.1-terra", "medium")
    assert (connection.model, connection.reasoning) == ("gpt-5.6-luna", "low")


def test_with_model_saves_nothing(tmp_path):
    bridge = offline_bridge(tmp_path)
    bridge.with_model(codex_connection(tmp_path), "sol")
    assert bridge.connections() == ()


def test_with_model_refuses_a_miss_and_a_group(tmp_path):
    bridge = offline_bridge(tmp_path)
    with pytest.raises(NoSuchModel):
        bridge.with_model(codex_connection(tmp_path), "opus")
    with pytest.raises(ValueError, match="not a group"):
        bridge.with_model("group:cheap", "sol")


def test_with_model_states_no_default_reasoning_where_the_runtime_has_no_control(tmp_path):
    """``google-cli`` has no effort control and no model list: a typed id goes
    through, the unsaid default becomes nothing, and a stated level is refused
    by the connection as it always was."""
    bridge = offline_bridge(tmp_path)
    connection = Connection(name="g", runtime=Runtime.GOOGLE_CLI, auth_mode="subscription",
                            experimental=True)
    picked = bridge.with_model(connection, "gemini-2.5-pro")
    assert (picked.model, picked.reasoning) == ("gemini-2.5-pro", None)
    with pytest.raises(InvalidConnection, match="reasoning effort is not established"):
        bridge.with_model(connection, "gemini-2.5-pro", reasoning="medium")


def test_with_model_on_claude_subscription_uses_the_alias(tmp_path):
    bridge = offline_bridge(tmp_path)
    connection = Connection(name="c", runtime=Runtime.ANTHROPIC_SDK, auth_mode="subscription")
    assert bridge.with_model(connection, "opus").model == "opus"
    with pytest.raises(NoSuchModel, match="Pass the exact model id"):
        bridge.with_model(connection, "opus", version="5")


class _Client:
    """A fake ``anthropic`` client that only knows how to list models."""

    def __init__(self, model_ids):
        self.rows = [Obj(id=i, display_name=i) for i in model_ids]
        self.calls = 0
        self.models = self

    def with_options(self, **_kwargs):
        return self

    def list(self, **_kwargs):
        self.calls += 1
        return Obj(data=self.rows, has_next_page=lambda: False)

    def close(self):
        pass


def test_with_model_fetches_an_api_list_once_and_only_metadata(tmp_path):
    client = _Client(CLAUDE)
    adapter = AnthropicAPIAdapter(client_factory=lambda **_: client, env={"KEY": "k"})
    bridge = offline_bridge(tmp_path, adapters={Runtime.ANTHROPIC_API: adapter})
    connection = Connection(name="api", runtime=Runtime.ANTHROPIC_API, auth_mode="api_key",
                            credential_ref=CredentialRef.parse("env:KEY"))
    assert bridge.with_model(connection, "opus").model == "claude-opus-5-5"
    assert bridge.with_model(connection, "opus", version="4.5").model == (
        "claude-opus-4-5-20251101"
    )
    assert client.calls == 1
