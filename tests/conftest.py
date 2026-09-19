"""Shared fixtures. No test in this suite may touch a network, a vendor package,
or a real credential. This is D2: the whole suite must pass offline, with no
vendor package installed and no credentials on the machine."""

from __future__ import annotations

import pytest

from modelpass.bridge import Bridge
from modelpass.capabilities import CapabilityRegistry
from modelpass.connections import Connection, CredentialRef, Guards
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore
from modelpass.testing import FakeAdapter
from modelpass.types import AuthMode


@pytest.fixture
def store(tmp_path) -> ConnectionStore:
    return ConnectionStore(tmp_path / "modelpass-home")


@pytest.fixture
def subscription_connection() -> Connection:
    return Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        guards=Guards(warn_at_tokens=100, stop_at_tokens=250),
        description="Claude Code login",
    )


@pytest.fixture
def api_connection() -> Connection:
    return Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:ANTHROPIC_API_KEY"),
    )


@pytest.fixture
def bridge_factory(store):
    """Build a bridge wired to a scripted adapter and an explicit environment."""

    def make(
        connection: Connection,
        adapter: FakeAdapter | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[Bridge, FakeAdapter]:
        store.add(connection)
        fake = adapter or FakeAdapter()
        bridge = Bridge(
            store=store,
            registry=CapabilityRegistry(),
            adapters={connection.runtime: fake},
            env=dict(env or {}),
        )
        return bridge, fake

    return make
