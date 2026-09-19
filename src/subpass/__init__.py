"""``subpass`` -- the pre-rename name for :mod:`modelpass`.

The library was renamed in 0.2.0. This package is a shim, not a copy: every name
it hands out **is** the object ``modelpass`` defines, so ``isinstance`` checks
and ``except`` clauses keep working across the boundary::

    from subpass.langchain_adapter import ChatSubpass
    import modelpass.langchain_adapter

    ChatSubpass is modelpass.langchain_adapter.ChatSubpass  # True

That identity is the whole point. A shim that re-created classes would give two
consumers in one process two different ``SubpassError`` types, and a caller
catching one would sail past the other.

How it works: a meta-path finder maps any ``subpass.X`` import onto the already
imported ``modelpass.X`` module object and registers it under both names. That
covers the three shapes consumers use today -- ``import subpass.models``,
``from subpass.types import Message``, and the dynamic
``importlib.import_module("subpass.langchain_adapter")`` -- at any depth
(``subpass.adapters.anthropic`` included), without a hand-written file per
module and without importing optional extras that may not be installed.

The names inside ``modelpass`` did **not** change: it is still ``SubpassError``
with its 23 subclasses, and still ``ChatSubpass``. Renaming an exception a
caller catches by name is a contract change, and it is not part of the rename
(DESIGN R9).

This package, the ``subpass`` console script, ``SUBPASS_HOME`` and the
``~/.subpass`` fallback all go away in 0.3.0, once every consumer has migrated.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType
from typing import Any

import modelpass as modelpass
from modelpass import *  # noqa: F403  (re-export: __all__ is modelpass's)
from modelpass import __all__ as __all__
from modelpass import __version__ as __version__

# Explicit because ``import *`` skips nothing here but a reader should not have
# to check: these two are module-level state consumers reach for by name.
from modelpass import chat as chat
from modelpass import default_bridge as default_bridge

_REAL = "modelpass"
_SHIM = "subpass"


class _AliasFinder(MetaPathFinder, Loader):
    """Resolves ``subpass.X`` to the module object ``modelpass.X`` already is."""

    #: ``module_from_spec`` overwrites ``__spec__`` on whatever
    #: :meth:`create_module` returns, and what it returns here is the real
    #: module -- so the real spec is stashed on the way in and put back on the
    #: way out. Without this, importing through the shim would leave
    #: ``modelpass.X.__spec__`` claiming to be named ``subpass.X``.
    _saved: dict[str, ModuleSpec | None]

    def __init__(self) -> None:
        self._saved = {}

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if not fullname.startswith(_SHIM + "."):
            return None
        return spec_from_loader(fullname, self)

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        real = importlib.import_module(_REAL + spec.name[len(_SHIM) :])
        self._saved[spec.name] = getattr(real, "__spec__", None)
        return real

    def exec_module(self, module: ModuleType) -> None:
        saved = self._saved.pop(_SHIM + module.__name__[len(_REAL) :], None)
        if saved is not None:
            module.__spec__ = saved


def _install_finder() -> None:
    if any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name: str) -> Any:
    """``subpass.models`` and friends, without importing them up front.

    Submodule attribute access on a package normally only works after the
    submodule has been imported. ``modelpass`` has optional-extra modules
    (``langchain_adapter``, the vendor adapters) that must not be imported at
    package import time, so attribute access imports on demand instead.
    """
    target = f"{_REAL}.{name}"
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as exc:
        # Only "there is no such module" becomes an AttributeError. A module
        # that exists but whose optional extra is missing must keep saying so,
        # by name, rather than being flattened into "no attribute".
        if exc.name != target:
            raise
        raise AttributeError(f"module {_SHIM!r} has no attribute {name!r}") from None


_install_finder()

warnings.warn(
    "subpass has been renamed to modelpass. Importing 'subpass' still works "
    "through a shim and gives you the same objects, but the shim is removed in "
    "0.3.0 -- import 'modelpass' instead.",
    DeprecationWarning,
    stacklevel=2,
)
