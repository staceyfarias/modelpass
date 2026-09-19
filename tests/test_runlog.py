"""The run-receipt audit log (2026-08-17).

The question these tests are really about is the owner's: *how do I prove it was
the subscription?* The proof has to survive the run that produced it, has to be
readable without modelpass, and must never be able to cost somebody a generation.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace

import pytest

from modelpass.bridge import Bridge
from modelpass.capabilities import CapabilityRegistry
from modelpass.connections import Connection, Guards, QuotaAction, QuotaPolicy
from modelpass.errors import ConfigError
from modelpass.runlog import RunLog, RunRecord, run_log_for
from modelpass.runtimes import Runtime
from modelpass.store import StoreSettings
from modelpass.testing import FakeAdapter, quota_exhausted, usage
from modelpass.types import (
    AuthMode,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
    TokenUsage,
    VendorEvent,
)

MESSAGE = "hello"


def drain(events) -> list:
    return list(events)


def test_a_completed_run_appends_one_line(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(
        subscription_connection,
        FakeAdapter(script=[TextDeltaEvent("hi"), usage(input_tokens=10, output_tokens=5)]),
    )
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))

    lines = bridge.run_log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["connection"] == "claude-sub"
    assert record["runtime"] == "anthropic-sdk"
    assert record["auth_mode"] == "subscription"
    assert record["status"] == "ok"
    assert record["input_tokens"] == 10
    assert record["output_tokens"] == 5
    assert record["total_tokens"] == 15
    assert record["guards_configured"] is True
    assert record["failed_over_from"] is None
    assert record["timestamp"].endswith("+00:00")


def test_the_log_lands_in_the_same_home_as_the_store(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(subscription_connection)
    assert bridge.run_log.path == bridge.store.root / "runs.jsonl"


def test_the_record_carries_no_prompt_and_no_response(
    bridge_factory, subscription_connection
):
    """A ledger, not a transcript. Handing somebody this file must not hand them the work."""
    bridge, _ = bridge_factory(
        subscription_connection,
        FakeAdapter(script=[TextDeltaEvent("the secret answer is 42")]),
    )
    drain(
        bridge.chat(
            connection="claude-sub",
            message="my confidential question",
        )
    )
    text = bridge.run_log.path.read_text(encoding="utf-8")
    assert "confidential" not in text
    assert "secret answer" not in text


def test_the_model_is_recorded_when_it_is_known(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(subscription_connection)
    drain(bridge.chat(connection="claude-sub", message=MESSAGE, model="some-model"))
    assert bridge.run_log.read()[0]["model"] == "some-model"


def test_a_run_with_no_model_records_null(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(subscription_connection)
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert bridge.run_log.read()[0]["model"] is None


def test_guards_absence_is_recorded_too(bridge_factory, subscription_connection):
    """A ledger that cannot tell a bounded run from an unbounded one is missing the point."""
    bridge, _ = bridge_factory(subscription_connection.with_guards(Guards.disabled()))
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert bridge.run_log.read()[0]["guards_configured"] is False


def test_every_terminal_status_is_logged(bridge_factory, subscription_connection):
    """Including the ones that are not 'ok' -- a guard stop is a run that spent something."""
    guarded = subscription_connection.with_guards(Guards(stop_at_tokens=5))
    bridge, _ = bridge_factory(
        guarded, FakeAdapter(script=[usage(input_tokens=50), TextDeltaEvent("never")])
    )
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    record = bridge.run_log.read()[0]
    assert record["status"] == "guard_stop"
    assert record["total_tokens"] == 50


def test_a_failover_records_the_connection_that_finished_and_the_one_that_started(store):
    primary = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        guards=Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api")),
    )
    target = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref="env:ANTHROPIC_API_KEY",
    )
    store.add(primary)
    store.add(target)

    calls: list[str] = []

    def script(request):
        calls.append(request.connection.name)
        if len(calls) == 1:
            yield quota_exhausted()
        else:
            yield usage(input_tokens=7)

    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter(script=script)},
        env={"ANTHROPIC_API_KEY": "x"},
    )
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))

    records = bridge.run_log.read()
    # One call, two connections, two allowances spent -- so two lines. Logging
    # only the connection that *finished* would report the subscription's
    # tokens as the metered connection's, or lose them entirely, which is the
    # one case this ledger exists to disambiguate.
    assert len(records) == 2
    second, first = records  # newest first
    assert first["connection"] == "claude-sub"
    assert first["auth_mode"] == "subscription"
    assert first["status"] == "quota_exhausted"
    assert second["connection"] == "claude-api"
    assert second["auth_mode"] == "api_key"
    assert second["total_tokens"] == 7
    assert second["failed_over_from"] == "claude-sub"


def test_records_accumulate_newest_first(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(subscription_connection)
    for _ in range(3):
        drain(bridge.chat(connection="claude-sub", message=MESSAGE, model="m"))
    assert len(bridge.run_log.read()) == 3
    assert len(bridge.run_log.path.read_text(encoding="utf-8").splitlines()) == 3


# --- opt-out ---------------------------------------------------------------------


def test_the_store_setting_switches_it_off(bridge_factory, subscription_connection):
    bridge, _ = bridge_factory(subscription_connection)
    bridge.store.save_settings(StoreSettings(run_log=False))
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert bridge.run_log.enabled is False
    assert bridge.run_log.exists() is False


def test_the_opt_out_survives_adding_another_connection(store, subscription_connection):
    """Adding a connection rewrites the whole file; it must not revert the setting."""
    store.save_settings(StoreSettings(run_log=False))
    store.add(subscription_connection)
    assert store.settings().run_log is False


def test_the_setting_round_trips_through_the_file(store, subscription_connection):
    store.add(subscription_connection)
    assert store.settings() == StoreSettings()
    # The default is written as silence, like the guards table: no [settings]
    # table at all until somebody changes something.
    assert "settings" not in tomllib.loads(store.path.read_text(encoding="utf-8"))

    store.save_settings(StoreSettings(run_log=False))
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert raw["settings"] == {"runLog": False}
    assert store.settings().run_log is False
    assert store.get("claude-sub") == subscription_connection


def test_an_unknown_setting_is_refused_rather_than_ignored(store):
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text("version = 1\n\n[settings]\nrunLogg = false\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown setting"):
        store.settings()


def test_a_non_boolean_setting_is_refused(store):
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text("version = 1\n\n[settings]\nrunLog = 'no'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be true or false"):
        store.settings()


def test_a_broken_settings_table_leaves_the_log_enabled(store):
    """The failure to avoid is a config problem silently switching off the evidence."""
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text("version = 1\n\n[settings]\nrunLog = 'no'\n", encoding="utf-8")
    assert run_log_for(store).enabled is True


def test_a_settings_table_this_build_cannot_read_does_not_block_writes(
    store, subscription_connection
):
    """Refusing to add or delete a connection until an unrelated optional
    setting is hand-fixed would strand a user inside their own config."""
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        "version = 1\n\n[settings]\nfromTheFuture = 1\n", encoding="utf-8"
    )
    store.add(subscription_connection)
    assert store.get("claude-sub") == subscription_connection
    # Carried through verbatim: this build does not understand the key, which is
    # exactly why it has no business rewriting or dropping it.
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert raw["settings"] == {"fromTheFuture": 1}

    store.remove("claude-sub")
    raw = tomllib.loads(store.path.read_text(encoding="utf-8"))
    assert raw["settings"] == {"fromTheFuture": 1}


def test_a_disabled_log_still_reads_what_was_already_written(tmp_path):
    log = RunLog(tmp_path)
    log.append(_record())
    assert len(RunLog(tmp_path, enabled=False).read()) == 1


def test_switching_the_setting_off_takes_effect_without_a_restart(
    bridge_factory, subscription_connection
):
    """The documented opt-out is editing the file; a switch that needs a restart
    is not a switch, and every page would meanwhile claim it was off."""
    bridge, _ = bridge_factory(subscription_connection)
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert len(bridge.run_log.read()) == 1

    bridge.store.save_settings(StoreSettings(run_log=False))
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert len(bridge.run_log.read()) == 1

    bridge.store.save_settings(StoreSettings(run_log=True))
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert len(bridge.run_log.read()) == 2


def test_an_injected_log_is_used_as_given(store, subscription_connection, tmp_path):
    from modelpass.bridge import Bridge
    from modelpass.capabilities import CapabilityRegistry

    store.add(subscription_connection)
    injected = RunLog(tmp_path / "elsewhere")
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter()},
        env={},
        run_log=injected,
    )
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert len(injected.read()) == 1


# --- damage tolerance on read ----------------------------------------------------


def test_a_record_with_the_wrong_field_types_is_repaired_not_trusted(tmp_path):
    """Hand-editable means hand-damageable; a reader formatting a string as a
    number would let one edited line take down the whole ledger view."""
    log = RunLog(tmp_path)
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-17T00:00:00+00:00",
                "connection": "c",
                "runtime": "anthropic-sdk",
                "auth_mode": "subscription",
                "status": "ok",
                "total_tokens": "12",
                "input_tokens": None,
                "guards_configured": "yes",
                "model": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = log.read()[0]
    assert record["total_tokens"] == 12
    assert record["input_tokens"] == 0
    assert record["guards_configured"] is True
    assert record["model"] is None


def test_an_unrecognized_field_is_kept(tmp_path):
    """A line from a newer modelpass is still a record."""
    log = RunLog(tmp_path)
    log.append(_record())
    text = log.path.read_text(encoding="utf-8").rstrip("\n")
    payload = json.loads(text)
    payload["something_new"] = {"nested": 1}
    log.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert log.read()[0]["something_new"] == {"nested": 1}


def test_the_tail_read_matches_a_whole_file_read(tmp_path):
    """The block-wise backwards read must agree with the obvious implementation."""
    log = RunLog(tmp_path)
    for index in range(400):
        log.append(_record(connection=f"c{index}"))
    assert [r["connection"] for r in log.read(limit=3)] == ["c399", "c398", "c397"]
    assert len(log.read(limit=None)) == 400
    assert log.read(limit=None)[0]["connection"] == "c399"
    assert log.read(limit=500)[-1]["connection"] == "c0"


# --- failure tolerance -----------------------------------------------------------


def test_a_write_failure_never_breaks_a_run(bridge_factory, subscription_connection, tmp_path):
    """The bookkeeping must not be able to cost somebody a generation."""
    bridge, _ = bridge_factory(subscription_connection)
    # A directory where the file should be: opening it for append raises, on
    # every platform, without needing to fake anything.
    (bridge.store.root).mkdir(parents=True, exist_ok=True)
    (bridge.store.root / "runs.jsonl").mkdir()

    events = drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].status is TerminalStatus.OK
    assert bridge.run_log.read() == ()


def test_append_reports_failure_without_raising(tmp_path):
    log = RunLog(tmp_path)
    (tmp_path / "runs.jsonl").mkdir()
    assert log.append(_record()) is False


def test_an_unreadable_log_reads_as_empty(tmp_path):
    assert RunLog(tmp_path / "nope").read() == ()


def test_a_malformed_line_does_not_hide_the_rest(tmp_path):
    log = RunLog(tmp_path)
    log.append(_record())
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("{ this is not json\n")
    log.append(_record())
    assert len(log.read()) == 2


def test_read_limit_takes_the_tail(tmp_path):
    log = RunLog(tmp_path)
    for index in range(5):
        log.append(_record(connection=f"c{index}"))
    assert [r["connection"] for r in log.read(limit=2)] == ["c4", "c3"]
    assert len(log.read(limit=None)) == 5


# --- record shape ----------------------------------------------------------------


def test_the_record_reads_the_auth_mode_off_the_stamped_terminal():
    """The terminal is the one object an adapter cannot forge (D3)."""
    terminal = TerminalEvent(
        status=TerminalStatus.OK,
        connection="c",
        runtime=Runtime.OPENAI_SDK,
        auth_mode=AuthMode.API_KEY,
        usage=TokenUsage(input_tokens=1, output_tokens=2, cached_input_tokens=3),
    )
    record = RunRecord.from_terminal(terminal, model="m", guards_configured=True)
    assert record.auth_mode == "api_key"
    assert record.runtime == "openai-sdk"
    assert record.total_tokens == 6
    assert json.loads(json.dumps(record.to_dict()))["model"] == "m"


def _record(connection: str = "claude-sub") -> RunRecord:
    return RunRecord.from_terminal(
        TerminalEvent(
            status=TerminalStatus.OK,
            connection=connection,
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
        )
    )


# --- the allowance: the vendor's own number, kept (D20) ----------------------------
#
# ``rate_limit_events()`` had been normalizing the runtime's own allowance
# reporting into a vendor event, and nothing consumed it. The best answer either
# runtime offers to "how am I doing against my plan" was computed and then
# dropped, once per run. Now it lands on the record beside the tokens.


def rate_limit(
    *,
    status: str = "allowed_warning",
    window: str | None = "seven_day",
    utilization: float | None = 0.62,
    resets_at: float | None = 1_800_000_000.0,
) -> VendorEvent:
    """A ``rate_limit`` event shaped exactly as the Anthropic adapter emits one."""
    return VendorEvent(
        Runtime.ANTHROPIC_SDK,
        "rate_limit",
        {
            "status": status,
            "rate_limit_type": window,
            "resets_at": resets_at,
            "utilization": utilization,
            "overage_status": None,
        },
    )


def test_the_vendors_allowance_report_survives_the_run_that_produced_it(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(
        subscription_connection,
        FakeAdapter(script=[rate_limit(), usage(input_tokens=10)]),
    )
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))

    record = bridge.run_log.read()[0]
    assert record["allowance_status"] == "allowed_warning"
    assert record["allowance_window"] == "seven_day"
    assert record["allowance_utilization"] == pytest.approx(0.62)
    # Converted to ISO so the field reads the same way as ``timestamp`` does to
    # jq, a spreadsheet, or a person -- the rest of this file is ISO.
    assert (record["allowance_resets_at"] or "").startswith("2027-01-15T")


def test_the_allowance_event_still_reaches_the_caller_unchanged(
    bridge_factory, subscription_connection
):
    """Observed on the way past, never intercepted."""
    bridge, _ = bridge_factory(
        subscription_connection,
        FakeAdapter(script=[rate_limit(), usage(input_tokens=1)]),
    )
    events = drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert any(isinstance(e, VendorEvent) and e.name == "rate_limit" for e in events)


def test_the_last_allowance_report_wins_because_it_is_the_current_one(
    bridge_factory, subscription_connection
):
    bridge, _ = bridge_factory(
        subscription_connection,
        FakeAdapter(
            script=[
                rate_limit(utilization=0.10, status="allowed"),
                rate_limit(utilization=0.91, status="allowed_warning"),
                usage(input_tokens=1),
            ]
        ),
    )
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))

    record = bridge.run_log.read()[0]
    assert record["allowance_utilization"] == pytest.approx(0.91)
    assert record["allowance_status"] == "allowed_warning"


def test_a_runtime_that_reports_no_allowance_records_absence_not_zero(
    bridge_factory, subscription_connection
):
    """Codex has no equivalent, and 0% used is a claim modelpass has no basis for."""
    bridge, _ = bridge_factory(
        subscription_connection, FakeAdapter(script=[usage(input_tokens=4)])
    )
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))

    record = bridge.run_log.read()[0]
    assert record["allowance_status"] is None
    assert record["allowance_utilization"] is None
    assert record["allowance_window"] is None
    assert record["allowance_resets_at"] is None


def test_a_line_written_before_these_fields_existed_reads_as_absence(tmp_path):
    """Repaired toward None, never toward 0 -- the distinction is the whole point."""
    log = RunLog(tmp_path)
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-30T00:00:00+00:00",
                "connection": "claude-sub",
                "runtime": "anthropic-sdk",
                "auth_mode": "subscription",
                "status": "ok",
                "total_tokens": 12,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = log.read()[0]
    assert record["allowance_utilization"] is None
    assert record["allowance_status"] is None
    assert record["total_tokens"] == 12


def test_a_hand_edited_utilization_is_repaired_like_the_token_counts_are(tmp_path):
    log = RunLog(tmp_path)
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-30T00:00:00+00:00",
                "connection": "claude-sub",
                "runtime": "anthropic-sdk",
                "auth_mode": "subscription",
                "status": "ok",
                "allowance_utilization": "0.5",
                "allowance_status": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = log.read()[0]
    assert record["allowance_utilization"] == pytest.approx(0.5)
    assert record["allowance_status"] is None


def test_an_unreadable_epoch_costs_the_field_and_not_the_record():
    record = RunRecord.from_terminal(
        TerminalEvent(
            status=TerminalStatus.OK,
            connection="claude-sub",
            runtime=Runtime.ANTHROPIC_SDK,
            auth_mode=AuthMode.SUBSCRIPTION,
            usage=TokenUsage(input_tokens=3),
        ),
        allowance={"status": "allowed", "resets_at": 10**30},
    )
    assert record.allowance_status == "allowed"
    assert record.allowance_resets_at is None
    assert record.input_tokens == 3


def test_the_allowance_line_is_none_rather_than_zero_percent_used():
    """A ledger view must not render an absence as a measurement."""
    blank = RunRecord(
        timestamp="", connection="c", runtime="r", auth_mode="a", status="ok"
    )
    assert blank.allowance_line is None
    reported = RunRecord(
        timestamp="",
        connection="c",
        runtime="r",
        auth_mode="a",
        status="ok",
        allowance_status="allowed_warning",
        allowance_window="seven_day",
        allowance_utilization=0.62,
    )
    assert "62% used" in (reported.allowance_line or "")


def test_each_failover_leg_carries_its_own_allowance_or_none(store):
    """One plan's position must never be reported as another plan's."""
    primary = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        guards=Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api")),
    )
    target = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref="env:ANTHROPIC_API_KEY",
    )
    store.add(primary)
    store.add(target)

    calls: list[str] = []

    def script(request):
        calls.append(request.connection.name)
        if len(calls) == 1:
            yield rate_limit(status="rejected", utilization=1.0)
            yield quota_exhausted()
        else:
            yield usage(input_tokens=7)

    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: FakeAdapter(script=script)},
        env={"ANTHROPIC_API_KEY": "x"},
    )
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))

    second, first = bridge.run_log.read()  # newest first
    assert first["allowance_status"] == "rejected"
    assert first["allowance_utilization"] == pytest.approx(1.0)
    # The metered connection reported nothing of its own, and inheriting the
    # subscription's exhausted allowance would describe the wrong plan.
    assert second["allowance_status"] is None
    assert second["allowance_utilization"] is None


# --- which executable ran (2026-08-31) ---------------------------------------------
#
# A transport-level defect in the Codex CLI -- a multi-line prompt truncated by
# the Windows .cmd shim -- was undiagnosable from this ledger, because the line
# never said which of the machine's Codex builds had run. The receipt knew; the
# record threw it away. Now it does not, and "was this run affected" is a query.


class BinaryNamingAdapter(FakeAdapter):
    """A FakeAdapter whose preflight names an executable, one per connection.

    Two connections, two binaries, deliberately: the failover path is the one
    place a single call has more than one, and mixing them up is the exact
    confusion the field exists to prevent.
    """

    def preflight(self, request):
        return replace(
            super().preflight(request), binary=f"/opt/{request.connection.name}/codex"
        )


def test_the_record_names_the_binary_the_receipt_resolved(store, subscription_connection):
    store.add(subscription_connection)
    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: BinaryNamingAdapter()},
        env={},
    )
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert bridge.run_log.read()[0]["binary"] == "/opt/claude-sub/codex"


def test_a_runtime_that_launches_nothing_by_path_records_none(
    bridge_factory, subscription_connection
):
    """Absence, not a gap: anthropic-sdk's own SDK owns its subprocess."""
    bridge, _ = bridge_factory(subscription_connection)
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))
    assert bridge.run_log.read()[0]["binary"] is None


def test_each_failover_leg_records_its_own_binary(store):
    """One call, two connections, two executables -- and two lines to say so."""
    primary = Connection(
        name="claude-sub",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        guards=Guards(on_quota_exhausted=QuotaPolicy(QuotaAction.FAILOVER, "claude-api")),
    )
    target = Connection(
        name="claude-api",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.API_KEY,
        credential_ref="env:ANTHROPIC_API_KEY",
    )
    store.add(primary)
    store.add(target)

    calls: list[str] = []

    def script(request):
        calls.append(request.connection.name)
        if len(calls) == 1:
            yield quota_exhausted()
        else:
            yield usage(input_tokens=7)

    bridge = Bridge(
        store=store,
        registry=CapabilityRegistry(),
        adapters={Runtime.ANTHROPIC_SDK: BinaryNamingAdapter(script=script)},
        env={"ANTHROPIC_API_KEY": "x"},
    )
    drain(bridge.chat(connection="claude-sub", message=MESSAGE))

    second, first = bridge.run_log.read()  # newest first
    assert first["connection"] == "claude-sub"
    assert first["binary"] == "/opt/claude-sub/codex"
    assert second["connection"] == "claude-api"
    assert second["binary"] == "/opt/claude-api/codex"


def test_the_binary_round_trips_through_the_record(tmp_path):
    log = RunLog(tmp_path)
    log.append(replace(_record(), binary="C:/npm/codex.cmd"))
    assert log.read()[0]["binary"] == "C:/npm/codex.cmd"


def test_a_line_written_before_the_binary_field_existed_still_reads(tmp_path):
    """Old logs stay readable, and say None rather than inventing an answer."""
    log = RunLog(tmp_path)
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-30T00:00:00+00:00",
                "connection": "codex-sub",
                "runtime": "openai-sdk",
                "auth_mode": "subscription",
                "status": "ok",
                "total_tokens": 12,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = log.read()[0]
    assert record["binary"] is None
    assert record["total_tokens"] == 12  # the rest of the line is untouched
