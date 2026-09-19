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
