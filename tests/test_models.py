"""Model catalogues (2026-08-17).

Offline throughout: the Codex cache is a fixture written into a temp directory,
never the real one. The properties under test are mostly about honesty --
whether a degraded list admits to being degraded, and whether an unfamiliar file
empties the picker.
"""

from __future__ import annotations

import json

from modelpass.models import ModelCatalogue, codex_home, models_for
from modelpass.runtimes import Runtime

CACHE = {
    "fetched_at": "2026-08-17T14:06:38Z",
    "client_version": "0.148.0",
    "models": [
        {
            "slug": "gpt-5.6-luna",
            "display_name": "GPT-5.6-Luna",
            "description": "Fast and affordable.",
            "visibility": "list",
            "priority": 3,
        },
        {
            "slug": "gpt-5.6-sol",
            "display_name": "GPT-5.6-Sol",
            "description": "Latest frontier agentic coding model.",
            "visibility": "list",
            "priority": 1,
        },
        {
            "slug": "codex-auto-review",
            "display_name": "Codex Auto Review",
            "visibility": "hide",
            "priority": 43,
        },
    ],
}


def write_cache(root, data=CACHE) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "models_cache.json").write_text(json.dumps(data), encoding="utf-8")


def test_codex_models_come_from_the_vendors_own_cache(tmp_path):
    write_cache(tmp_path)
    catalogue = models_for(Runtime.OPENAI_SDK, codex_root=tmp_path)
    assert catalogue.exact is True
    assert [c.id for c in catalogue.choices] == ["gpt-5.6-sol", "gpt-5.6-luna"]
    assert catalogue.choices[0].label == "GPT-5.6-Sol"
    assert catalogue.problem is None


def test_hidden_entries_are_the_vendors_own_do_not_offer_signal(tmp_path):
    write_cache(tmp_path)
    catalogue = models_for(Runtime.OPENAI_SDK, codex_root=tmp_path)
    assert "codex-auto-review" not in [c.id for c in catalogue.choices]


def test_the_writing_cli_version_is_reported_rather_than_hidden(tmp_path):
    """0.148.0 wrote the cache; an installed 0.117.0 would reject its newest slugs."""
    write_cache(tmp_path)
    catalogue = models_for(Runtime.OPENAI_SDK, codex_root=tmp_path)
    assert "0.148.0" in catalogue.source
    assert any("0.148.0" in note for note in catalogue.notes)


def test_unknown_fields_and_newer_enum_variants_do_not_empty_the_picker(tmp_path):
    """The cache is written by whichever CLI ran last, and may be newer than us."""
    write_cache(
        tmp_path,
        {
            "models": [
                {
                    "slug": "gpt-9",
                    "display_name": "GPT-9",
                    "visibility": "list",
                    "some_future_key": {"nested": ["whatever"]},
                    "tool_mode": "a_variant_we_have_never_heard_of",
                }
            ]
        },
    )
    catalogue = models_for(Runtime.OPENAI_SDK, codex_root=tmp_path)
    assert [c.id for c in catalogue.choices] == ["gpt-9"]


def test_a_missing_cache_degrades_with_a_reason(tmp_path):
    catalogue = models_for(Runtime.OPENAI_SDK, codex_root=tmp_path / "nope")
    assert catalogue.choices == ()
    assert catalogue.problem is not None
    assert "free-text" in catalogue.problem
    assert bool(catalogue) is False


def test_a_corrupt_cache_degrades_rather_than_raising(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models_cache.json").write_text("{not json", encoding="utf-8")
    catalogue = models_for(Runtime.OPENAI_SDK, codex_root=tmp_path)
    assert catalogue.choices == ()
    assert "could not read" in (catalogue.problem or "")


def test_a_cache_with_no_models_array_degrades(tmp_path):
    write_cache(tmp_path, {"fetched_at": "x"})
    catalogue = models_for(Runtime.OPENAI_SDK, codex_root=tmp_path)
    assert "no 'models' array" in (catalogue.problem or "")


def test_anthropic_offers_aliases_and_says_they_are_aliases():
    """No enumeration API exists there; presenting a made-up list as the vendor's
    would be the guard-default mistake with a vendor's name on it."""
    catalogue = models_for(Runtime.ANTHROPIC_SDK)
    assert [c.id for c in catalogue.choices] == ["sonnet", "opus", "haiku"]
    assert catalogue.exact is False
    assert "not a vendor model list" in catalogue.source
    assert any("enumerate" in note for note in catalogue.notes)


def test_a_runtime_with_no_known_source_says_so():
    catalogue = models_for(Runtime.GOOGLE_CLI)
    assert catalogue.choices == ()
    assert catalogue.problem is not None


def test_an_unknown_runtime_degrades_rather_than_raising():
    """"Never raises" has to include the argument, not just the file being read."""
    catalogue = models_for("not-a-runtime")
    assert catalogue.choices == ()
    assert "not a runtime" in (catalogue.problem or "")


def test_an_entry_with_no_visibility_key_is_still_offered(tmp_path):
    """A denylist, not an allowlist: a CLI that renames the field must not
    empty the picker while claiming to be the vendor's exact list."""
    write_cache(
        tmp_path,
        {"models": [{"slug": "gpt-9", "display_name": "GPT-9"}]},
    )
    catalogue = models_for(Runtime.OPENAI_SDK, codex_root=tmp_path)
    assert [c.id for c in catalogue.choices] == ["gpt-9"]


def test_codex_home_honours_the_environment_variable(tmp_path):
    assert codex_home({"CODEX_HOME": str(tmp_path)}) == tmp_path
    assert codex_home({}).name == ".codex"


def test_a_catalogue_is_falsy_when_empty():
    assert not ModelCatalogue(runtime=Runtime.OPENAI_SDK)
