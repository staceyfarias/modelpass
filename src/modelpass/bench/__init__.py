"""The bench: a local page for configuring modelpass and watching it run.

``modelpass bench`` serves this on ``127.0.0.1`` and nothing else. It is a
development tool for the machine it runs on -- not a service, not multi-user,
and deliberately not reachable from the network.

Why it exists at all, given a perfectly good command line: the product's central
claim is that you can *see* how a run is authenticated and what it spends, and a
claim like that is better shown than described. The bench puts the four things
that make it true on one page -- the connections you configured, the receipt a
preflight actually produces, the raw event stream as it arrives, and the run log
that outlives all of it.

Flask arrives through the optional ``modelpass[bench]`` extra; core keeps its zero
dependencies (D8). The import is lazy and a missing package names the extra.
"""

from __future__ import annotations

from .app import DEFAULT_PORT, create_app, require_flask, serve

__all__ = ["DEFAULT_PORT", "create_app", "require_flask", "serve"]
