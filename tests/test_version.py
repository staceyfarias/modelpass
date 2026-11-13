"""``modelpass.__version__`` is the number the distribution actually ships.

Two sources of truth is one too many: the packaging metadata in
``pyproject.toml`` and the attribute a consumer reads at runtime have to agree,
or a bug report names a version nobody can reproduce.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import modelpass

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_dunder_version_matches_pyproject():
    with _PYPROJECT.open("rb") as handle:
        metadata = tomllib.load(handle)
    assert modelpass.__version__ == metadata["project"]["version"]


def test_a_changelog_exists_and_names_the_current_version():
    changelog = _PYPROJECT.parent / "CHANGELOG.md"
    assert changelog.is_file()
    assert f"## {modelpass.__version__}" in changelog.read_text(encoding="utf-8")


def test_every_vendor_extra_declares_a_floor_and_a_cap():
    """An extra with no upper bound installs whatever the vendor ships next.

    The two agent-runtime extras had none (2026-11-13): ``claude-agent-sdk``
    had a floor only and ``openai-codex`` had no version at all, while the
    adapters built on them read private names and generated types that move
    between releases. Every vendor requirement now says both ends.
    """
    import re
    import tomllib

    extras = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    vendor = ("anthropic", "openai", "anthropic-api", "openai-api", "google-api")
    for extra in vendor:
        for requirement in extras[extra]:
            assert re.search(r">=\s*\d", requirement), f"{extra}: {requirement} has no floor"
            assert re.search(r"<\s*\d", requirement), f"{extra}: {requirement} has no cap"
    assert extras["anthropic"] == ["claude-agent-sdk>=0.2.159,<0.3"]
