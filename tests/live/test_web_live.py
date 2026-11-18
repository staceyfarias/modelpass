"""Owner-authorized native web smoke tests. Spends allowance, excluded by default.

Run with SUBPASS_LIVE_TESTS=1 and pytest -m live tests/live/test_web_live.py.
Connection and model names can be selected with MODELPASS_WEB_* variables.
OpenAI API and Gemini are intentionally absent from this live drive.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from modelpass import Bridge, WebActivityEvent, WebEvidenceEvent, WebOptions

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "vendor,default_connection,default_model",
    [
        ("ANTHROPIC_SUB", "claude-sub", "haiku"),
        # gpt-5.6-luna, not gpt-6-luna: on 2026-11-18 the latter came back
        # "not supported when using Codex with a ChatGPT account" through
        # codex-cli 0.151.0. Override with MODELPASS_WEB_OPENAI_SUB_MODEL.
        ("OPENAI_SUB", "openai-sub", "gpt-5.6-luna"),
        ("ANTHROPIC_API", "claude-api", "claude-haiku-4-5"),
    ],
)
def test_native_search(vendor, default_connection, default_model, tmp_path):
    if os.environ.get("SUBPASS_LIVE_TESTS") != "1":
        pytest.skip("set SUBPASS_LIVE_TESTS=1 to authorize live web calls")
    if any(os.environ.get(key) for key in ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "TF_BUILD")):
        pytest.skip("live web tests do not run in CI")
    connection = os.environ.get(f"MODELPASS_WEB_{vendor}_CONNECTION", default_connection)
    model = os.environ.get(f"MODELPASS_WEB_{vendor}_MODEL", default_model)
    answer = Bridge().ask(
        connection,
        "Search the web for the current stable Python release on python.org/downloads. "
        "Actually use web search. Give the version and the official source URL in one sentence.",
        model=model,
        web=WebOptions(require_search=True, require_citations=vendor == "ANTHROPIC_API"),
        allow_failover=False,
        timeout=120,
    )
    assert answer.text
    assert any(isinstance(e, WebActivityEvent) for e in answer.events)
    if vendor == "ANTHROPIC_API":
        assert answer.citations
    record = {
        "date": datetime.now(UTC).isoformat(),
        "runtime": answer.receipt.runtime.value,
        "model": model,
        "status": answer.status.value,
        "text": answer.text,
        "sources": [vars_source(s) for s in answer.sources or ()],
        "citations": [vars_source(s) for s in answer.citations or ()],
        "web_events": [
            e.to_dict()
            for e in answer.events
            if isinstance(e, (WebActivityEvent, WebEvidenceEvent))
        ],
        "usage": answer.usage.to_dict(),
    }
    evidence = Path(os.environ.get("MODELPASS_WEB_EVIDENCE_DIR", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / f"{vendor.lower()}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def vars_source(source):
    return {"url": source.url, "title": source.title}
