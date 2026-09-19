"""``import subpass`` still works, and gives back the very same objects.

The rename ships a shim for one release cycle. A shim that handed out *copies*
would be worse than no shim at all: two consumers in one process would end up
with two unrelated ``SubpassError`` classes, and a caller catching one would sail
straight past the other. So what these tests assert is identity -- ``is``, not
``==`` -- across every import shape the four consumer projects actually use:

* ``from subpass import Bridge, SubpassError``
* ``from subpass.types import Message``
* ``import subpass.models``
* ``importlib.import_module("subpass.langchain_adapter")`` -- a dynamic string
  import in one consumer, and the reason a rename fails there at *run* time
  rather than at import time.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import modelpass

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import subpass

_SRC = Path(__file__).resolve().parent.parent / "src"


def test_the_two_packages_agree_on_version_and_exports():
    assert subpass.__version__ == modelpass.__version__
    assert subpass.__all__ == modelpass.__all__


def test_every_exported_name_is_the_same_object():
    missing = [name for name in modelpass.__all__ if not hasattr(subpass, name)]
    assert missing == []
    different = [
        name
        for name in modelpass.__all__
        if getattr(subpass, name) is not getattr(modelpass, name)
    ]
    assert different == [], "the shim must re-export, never re-create"


def test_module_level_chat_and_default_bridge_come_through():
    assert subpass.chat is modelpass.chat
    assert subpass.default_bridge is modelpass.default_bridge


def test_from_subpass_import_bridge_and_error():
    from subpass import Bridge, SubpassError

    assert Bridge is modelpass.Bridge
    assert SubpassError is modelpass.SubpassError
    # The reason identity matters: an error raised by modelpass is caught by a
    # consumer that still spells the name the old way.
    with pytest.raises(SubpassError):
        raise modelpass.NoSuchConnection("nope")


def test_from_subpass_types_import_message():
    import modelpass.types
    from subpass.types import Message

    assert Message is modelpass.types.Message


def test_import_subpass_models():
    import modelpass.models
    import subpass.models

    assert subpass.models is modelpass.models
    assert sys.modules["subpass.models"] is sys.modules["modelpass.models"]


def test_dynamic_string_import_of_the_langchain_leaf():
    pytest.importorskip("langchain_core")
    shimmed = importlib.import_module("subpass.langchain_adapter")

    import modelpass.langchain_adapter

    assert shimmed is modelpass.langchain_adapter

    from subpass.langchain_adapter import ChatSubpass

    assert ChatSubpass is modelpass.langchain_adapter.ChatSubpass


@pytest.mark.parametrize(
    "name",
    [
        "adapters",
        "adapters.base",
        "bridge",
        "capabilities",
        "connections",
        "errors",
        "models",
        "preflight",
        "runlog",
        "runtimes",
        "schema",
        "sessions",
        "store",
        "testing",
        "tools",
        "types",
    ],
)
def test_every_public_submodule_resolves_to_the_real_one(name):
    shimmed = importlib.import_module(f"subpass.{name}")
    real = importlib.import_module(f"modelpass.{name}")
    assert shimmed is real


def test_importing_through_the_shim_leaves_the_real_module_identity_alone():
    """``module_from_spec`` would otherwise rewrite ``__spec__`` on the target.

    A module that thinks it is named ``subpass.types`` reports the wrong name to
    anything that reads ``__spec__`` -- reloads, packaging tools, tracebacks.
    """
    importlib.import_module("subpass.types")

    import modelpass.types

    assert modelpass.types.__name__ == "modelpass.types"
    assert modelpass.types.__spec__ is not None
    assert modelpass.types.__spec__.name == "modelpass.types"


def test_an_attribute_that_is_not_a_module_is_an_attribute_error():
    with pytest.raises(AttributeError):
        _ = subpass.no_such_thing_at_all


def _run(script: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(_SRC.parent),
        env={**_child_env()},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _child_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_SRC) + (os.pathsep + existing if existing else "")
    return env


def test_the_deprecation_warning_fires_exactly_once():
    """Once per process, no matter how many times the old name is imported.

    A warning per import would be noise in a consumer with fifty import sites,
    and noise is how a deprecation gets filtered out and then missed.
    """
    script = """
import json, warnings, importlib
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import subpass
    import subpass
    importlib.import_module("subpass")
    import subpass.types
    from subpass.errors import SubpassError
    import subpass.adapters.base
print(json.dumps([
    str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
]))
"""
    messages = json.loads(_run(script))
    assert len(messages) == 1, messages
    assert "subpass" in messages[0]
    assert "modelpass" in messages[0]


def test_the_shim_works_when_it_is_imported_before_modelpass():
    """The order consumers will actually hit: old name first, new name never.

    A fresh interpreter, so nothing in ``sys.modules`` has pre-seeded the real
    package for the shim to lean on.
    """
    script = """
import subpass, modelpass, sys
assert subpass.Bridge is modelpass.Bridge
assert sys.modules["subpass.types"] is modelpass.types if "subpass.types" in sys.modules else True
from subpass.types import Message
import modelpass.types
assert Message is modelpass.types.Message
print("ok")
"""
    assert _run(script) == "ok"
