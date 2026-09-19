"""Live smoke: one tiny generation on your ChatGPT subscription via the Codex CLI.

Run it by hand; it is not part of the test suite. Costs one trivial turn of
your plan's Codex allowance (note: codex exec carries ~50k input tokens of
harness overhead per call — that is the vendor's fixed cost, not the prompt).

    python examples/smoke_openai.py

Requires: `codex login` already done (Sign in with ChatGPT), and either a
`codex` CLI on PATH or `pip install modelpass[openai]`.
"""

import sys
import tempfile

if not sys.flags.dev_mode:
    sys.path.insert(0, "src")

from modelpass.bridge import Bridge
from modelpass.connections import Connection, CredentialRef
from modelpass.runtimes import Runtime
from modelpass.store import ConnectionStore
from modelpass.types import AuthMode

# A throwaway store: the example never touches your real ~/.modelpass.
store = ConnectionStore(tempfile.mkdtemp(prefix="modelpass-example-"))
store.add(
    Connection(
        name="codex-sub",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        credential_ref=CredentialRef.native_login(),
        description="ChatGPT login via codex CLI",
    )
)
bridge = Bridge(store=store)

receipt = bridge.preflight("codex-sub")
print(receipt.summary())
receipt.require_ok()
receipt.require_auth_mode()  # refuse to run if this would bill an API key

for event in bridge.chat(
    connection="codex-sub",
    message="In one short sentence, say hello from modelpass.",
):
    if event.type == "text_delta":
        print(event.text)
    elif event.type == "terminal":
        print(
            f"[{event.status.value}] connection={event.connection} "
            f"auth_mode={event.auth_mode.value} tokens={event.usage.total_tokens}"
        )
