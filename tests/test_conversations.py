"""Managed conversations: durable app ownership without changing native sessions.

All runs use the real Bridge with scripted adapters. No vendor package,
network, credential, browser, or model allowance is used.
"""

import asyncio
import json
import multiprocessing
import os
from dataclasses import replace

import pytest

from modelpass import (
    AuthMode,
    Connection,
    ConversationBusy,
    ConversationError,
    ConversationNotFound,
    CredentialRef,
    Message,
    Role,
    Runtime,
    TerminalEvent,
    TerminalStatus,
    TextDeltaEvent,
)
from modelpass.testing import fake_bridge, fake_session_bridge, usage


@pytest.fixture
def setup(tmp_path):
    connection = Connection(
        name="one",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:TEST_KEY"),
        model="model-one",
    )
    bridge, configs, adapter = fake_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="answer"), usage(10, 2)],
        home=tmp_path / "config",
        runtime=Runtime.ANTHROPIC_API,
        env={"TEST_KEY": "not-a-real-key"},
    )
    root = tmp_path / "private-chats"
    store = bridge.conversations(root, application="soundbench", application_name="Soundbench")
    return bridge, configs, adapter, store, connection


def test_create_and_browse_are_local_and_origin_survives_reopen(setup):
    bridge, _, adapter, store, _ = setup
    chat = store.create(connection="one", title="Brainstorm", scope="song-1", scope_name="New Song")
    assert not adapter.requests
    info = chat.info
    assert info.application == "soundbench" and info.application_name == "Soundbench"
    assert info.scope == "song-1" and info.scope_name == "New Song"
    assert info.model == "model-one" and chat.turns() == []
    assert json.loads(json.dumps(info.to_dict()))["application"] == "soundbench"
    reopened = bridge.conversations(store.root, application="soundbench").open(chat.id)
    assert reopened.info == info and not adapter.requests
    assert store.list(scope="different-project") == []
    assert store.list(scope="song-1")[0].id == chat.id


def test_shared_root_never_lists_or_opens_another_apps_conversations(setup):
    bridge, _, _, store, _ = setup
    ours = store.create(connection="one")
    other = bridge.conversations(store.root, application="other-app")
    theirs = other.create(connection="one", title="Other app")
    assert [c.id for c in store.list()] == [ours.id]
    assert [c.id for c in other.list()] == [theirs.id]
    for target, identifier in ((store, theirs.id), (other, ours.id), (store, "../outside")):
        with pytest.raises(ConversationNotFound):
            target.open(identifier)
    assert ours.info.application != theirs.info.application


def test_reopened_history_is_passed_to_real_bridge_and_default_changes_do_not_rebind(setup):
    bridge, configs, adapter, store, connection = setup
    chat = store.create(connection="one")
    events = list(chat.send("first question"))
    assert events[0].type == "receipt" and isinstance(events[-1], TerminalEvent)
    assert chat.turns()[0].status == "ok" and chat.turns()[0].text == "answer"
    configs.add(replace(connection, model="new-config-default"), overwrite=True)
    reopened = bridge.conversations(store.root, application="soundbench").open(chat.id)
    list(reopened.send("follow-up"))
    request = adapter.requests[-1]
    assert request.model == "model-one"
    assert [(m.role, m.text) for m in request.messages][-3:] == [
        (Role.USER, "first question"),
        (Role.ASSISTANT, "answer"),
        (Role.USER, "follow-up"),
    ]
    assert [m.text for m in reopened.history()] == [
        "first question",
        "answer",
        "follow-up",
        "answer",
    ]
    assert chat.turns()[0].events[0]["type"] == "receipt"
    assert chat.turns()[0].events[-1]["type"] == "terminal"


def test_fresh_perspective_does_not_implicitly_copy_another_conversation(setup):
    _, configs, adapter, store, connection = setup
    first = store.create(connection="one")
    list(first.send("private previous opinion"))
    configs.add(replace(connection, name="two"))
    second = store.create(connection="two", model="another-model", title="Fresh eyes")
    list(second.send("judge only these lyrics"))
    assert [m.text for m in adapter.requests[-1].messages] == ["judge only these lyrics"]
    assert len(store.list(connection="one")) == 1
    assert store.list(model="another-model")[0].id == second.id
    assert first.info.model == "model-one"


def test_explicit_seed_and_instructions_survive_reopen(setup):
    _, _, adapter, store, _ = setup
    chat = store.create(
        connection="one",
        system_prompt="Keep the artist's unresolved ending",
        seed_history=[Message(Role.USER, "chosen context only")],
    )
    list(store.open(chat.id).send("continue"))
    messages = adapter.requests[-1].messages
    assert [m.text for m in messages] == [
        "Keep the artist's unresolved ending",
        "chosen context only",
        "continue",
    ]


def test_credentials_are_not_copied_and_identity_drift_is_refused_before_run(setup):
    _, configs, adapter, store, connection = setup
    chat = store.create(connection="one")
    list(chat.send("hello"))
    assert b"not-a-real-key" not in store.path.read_bytes()
    configs.add(
        replace(connection, credential_ref=CredentialRef.parse("env:OTHER_KEY")), overwrite=True
    )
    count = len(adapter.requests)
    with pytest.raises(ConversationError, match="identity changed"):
        list(chat.send("no wrong account"))
    assert len(adapter.requests) == count and len(chat.turns()) == 1


def test_cancelled_partial_is_durable_but_is_not_replayed(setup):
    _, _, adapter, store, _ = setup
    chat = store.create(connection="one")
    stream = chat.send("unfinished request")
    assert next(stream).type == "receipt"
    assert next(stream).type == "text_delta"
    stream.close()
    turn = store.open(chat.id).turns()[0]
    assert turn.status == "cancelled" and turn.text == "answer"
    assert chat.history() == []
    list(chat.send("new attempt"))
    assert [m.text for m in adapter.requests[-1].messages] == ["new attempt"]
    assert adapter.cancelled == 1


def test_same_conversation_busy_even_through_a_separate_store_instance(setup):
    bridge, _, _, store, _ = setup
    chat = store.create(connection="one")
    stream = chat.send("first")
    next(stream)
    other = bridge.conversations(store.root, application="soundbench").open(chat.id)
    try:
        with pytest.raises(ConversationBusy):
            next(other.send("overlap"))
        with pytest.raises(ConversationBusy):
            other.rename("overwrite")
    finally:
        stream.close()
    list(other.send("after cancellation"))


def test_rename_archive_restore_and_paging_preserve_transcripts(setup):
    _, _, _, store, _ = setup
    first = store.create(connection="one")
    list(first.send("first"))
    first.rename("My lyrics")
    second = store.create(connection="one")
    assert len(store.list(limit=1)) == 1
    assert len(store.list(limit=1, offset=1)) == 1
    first.archive()
    assert [i.id for i in store.list()] == [second.id]
    assert len(store.list(include_archived=True)) == 2
    assert store.open(first.id).turns()[0].message == "first"
    with pytest.raises(ConversationError, match="Unarchive"):
        list(first.send("blocked only because archived"))
    first.archive(False)
    assert first.info.title == "My lyrics" and len(first.turns()) == 1


@pytest.mark.parametrize(
    "override",
    [
        {"model": "other"},
        {"connection": "other"},
        {"system_prompt": "changed"},
        {"history": []},
        {"allow_failover": True},
    ],
)
def test_send_cannot_silently_change_bound_identity_or_context(setup, override):
    _, _, adapter, store, _ = setup
    chat = store.create(connection="one")
    with pytest.raises(ConversationError):
        list(chat.send("hello", **override))
    assert not chat.turns() and not adapter.requests


def test_native_mode_resumes_same_vendor_id_and_keeps_a_separate_observation_journal(tmp_path):
    connection = Connection(
        name="native",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        model="native-model",
    )
    bridge, _, adapter = fake_session_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="native reply")],
        home=tmp_path / "config",
    )
    store = bridge.conversations(tmp_path / "chats", application="soundbench")
    chat = store.create(connection="native", mode="native", system_prompt="song helper")
    assert chat.info.native_session_id is None
    list(chat.send("first"))
    native_id = chat.info.native_session_id
    assert native_id is not None
    list(store.open(chat.id).send("second"))
    assert chat.info.native_session_id == native_id
    assert len(chat.turns()) == 2
    assert [turn.text for turn in chat.turns()] == ["native reply", "native reply"]
    # No attempt to build a fake native transcript out of those observed replies.
    assert adapter.handles[-1].request.resume_id == native_id


def test_native_mode_on_api_is_not_silently_downgraded(setup):
    from modelpass import CapabilityNotSupported

    _, _, _, store, _ = setup
    with pytest.raises(CapabilityNotSupported):
        store.create(connection="one", mode="native")
    assert not store.list()


def test_async_uses_same_durable_history_and_can_resume_sync(setup):
    _, _, adapter, store, _ = setup
    chat = store.create(connection="one")

    async def run():
        return [e async for e in chat.asend("async question")]

    events = asyncio.run(run())
    assert events[-1].type == "terminal" and chat.turns()[0].status == "ok"
    list(store.open(chat.id).send("sync follow-up"))
    assert [m.text for m in adapter.requests[-1].messages][-3:] == [
        "async question",
        "answer",
        "sync follow-up",
    ]


def test_empty_application_and_missing_model_are_explicit_errors(setup):
    bridge, configs, _, store, connection = setup
    with pytest.raises(ConversationError):
        bridge.conversations(store.root, application=" ")
    configs.add(replace(connection, model=None), overwrite=True)
    with pytest.raises(ConversationError, match="model"):
        store.create(connection="one")


def test_terminal_is_durable_and_releases_lock_before_it_is_delivered(setup):
    _, _, _, store, _ = setup
    chat = store.create(connection="one")
    stream = chat.send("first")
    while next(stream).type != "terminal":
        pass
    # The caller need not advance past terminal or explicitly close a completed
    # iterator before the next UI action can address the same conversation.
    assert chat.turns()[0].status == "ok"
    list(store.open(chat.id).send("immediately after terminal"))
    stream.close()


def test_async_close_preserves_partial_and_releases_lock(setup):
    _, _, _, store, _ = setup
    chat = store.create(connection="one")

    async def run():
        stream = chat.asend("unfinished async")
        assert (await anext(stream)).type == "receipt"
        assert (await anext(stream)).type == "text_delta"
        await stream.aclose()

    asyncio.run(run())
    assert chat.turns()[0].status == "cancelled"
    assert chat.turns()[0].text == "answer"
    list(store.open(chat.id).send("after closing"))


def test_native_cancellation_is_not_silently_replayed_as_a_fresh_session(tmp_path):
    connection = Connection(
        name="native",
        runtime=Runtime.ANTHROPIC_SDK,
        auth_mode=AuthMode.SUBSCRIPTION,
        model="native-model",
    )
    bridge, _, _ = fake_session_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="partial native reply")],
        home=tmp_path / "config",
    )
    store = bridge.conversations(tmp_path / "chats", application="soundbench")
    chat = store.create(connection="native", mode="native")
    stream = chat.send("first")
    next(stream)
    stream.close()
    assert chat.turns()[0].status == "cancelled"
    with pytest.raises(ConversationError, match="runtime state is uncertain"):
        list(store.open(chat.id).send("do not fabricate continuity"))


def _crash_with_open_turn(root, config, identifier, ready):
    connection = Connection(
        name="one",
        runtime=Runtime.ANTHROPIC_API,
        auth_mode=AuthMode.API_KEY,
        credential_ref=CredentialRef.parse("env:TEST_KEY"),
        model="model-one",
    )
    bridge, _, _ = fake_bridge(
        connections=[connection],
        script=[TextDeltaEvent(text="before crash")],
        home=config,
        runtime=Runtime.ANTHROPIC_API,
        env={"TEST_KEY": "not-a-real-key"},
    )
    chat = bridge.conversations(root, application="soundbench").open(identifier)
    stream = chat.send("request interrupted by process exit")
    next(stream)  # receipt
    next(stream)  # output is durable before it reaches the caller
    ready.send("running")
    ready.recv()
    os._exit(0)  # no generator finally block, no explicit lock release


def test_separate_process_lock_and_crash_recovery_preserve_partial_output(setup, tmp_path):
    _, _, adapter, store, _ = setup
    chat = store.create(connection="one")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_crash_with_open_turn, args=(store.root, tmp_path / "child-config", chat.id, child)
    )
    process.start()
    child.close()
    try:
        assert parent.poll(20), "child did not start its offline turn"
        assert parent.recv() == "running"
        with pytest.raises(ConversationBusy):
            next(chat.send("overlapping process"))
        parent.send("exit without cleanup")
        process.join(timeout=20)
        assert process.exitcode == 0
        unfinished = chat.turns()[0]
        assert unfinished.status == "running" and unfinished.text == "before crash"
        list(chat.send("continue independently of failed request"))
        assert chat.turns()[0].status == "interrupted"
        assert chat.turns()[0].text == "before crash"
        assert [m.text for m in adapter.requests[-1].messages] == [
            "continue independently of failed request"
        ]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        parent.close()


def test_independent_conversations_can_progress_while_one_turn_is_open(setup):
    _, _, _, store, _ = setup
    first = store.create(connection="one")
    second = store.create(connection="one", scope="other-project")
    stream = first.send("held turn")
    next(stream)
    try:
        list(second.send("independent turn"))
        assert second.turns()[0].status == "ok"
    finally:
        stream.close()


def test_guard_stop_preserves_typed_terminal_and_is_not_a_successful_reply(setup):
    from modelpass import GuardStop

    _, _, _, store, _ = setup
    chat = store.create(connection="one")
    with pytest.raises(GuardStop):
        list(chat.send("bounded request", stop_at_tokens=1, raise_on_stop=True))
    turn = chat.turns()[0]
    assert turn.status == "guard_stop"
    assert sum(e["type"] == "terminal" for e in turn.events) == 1
    assert turn.events[-1]["status"] == "guard_stop"
    assert not chat.history()


def test_error_reply_and_context_remain_inspectable_but_are_not_replayed(setup):
    from modelpass.testing import vendor_failure

    _, _, adapter, store, _ = setup
    adapter.error = vendor_failure("vendor unavailable")
    chat = store.create(connection="one")
    events = list(chat.send("failed request"))
    assert events[-1].status == TerminalStatus.ERROR
    assert chat.turns()[0].status == "error"
    assert chat.turns()[0].message == "failed request"
    assert not chat.history()


def test_future_database_schema_is_refused_without_resetting_it(setup):
    import sqlite3

    bridge, _, _, store, _ = setup
    chat = store.create(connection="one")
    with sqlite3.connect(store.path) as db:
        db.execute("PRAGMA user_version = 999")
    with pytest.raises(ConversationError, match="database version"):
        bridge.conversations(store.root, application="soundbench")
    assert store.open(chat.id).info.id == chat.id
    with sqlite3.connect(store.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 999


def test_cross_app_constructor_cannot_bypass_origin_checks(setup):
    from modelpass import Conversation

    bridge, _, _, store, _ = setup
    chat = store.create(connection="one")
    other = bridge.conversations(store.root, application="other")
    forged = Conversation(other, chat.id)
    for operation in (
        lambda: forged.info,
        forged.turns,
        forged.history,
        lambda: forged.rename("wrong app"),
        lambda: forged.archive(),
        lambda: list(forged.send("wrong app")),
    ):
        with pytest.raises(ConversationNotFound):
            operation()
    assert chat.info.title == "New conversation" and not chat.turns()


def test_managed_failures_do_not_suggest_automatic_retry():
    from modelpass import Retryable

    for error in (
        ConversationError("binding changed"),
        ConversationBusy("busy"),
        ConversationNotFound("other app"),
    ):
        assert error.retryable == Retryable.NO


def test_explicit_seed_cache_markers_are_preserved_across_storage(setup):
    from modelpass import CacheControl, TextBlock

    _, _, adapter, store, _ = setup
    seed = Message(Role.USER, (TextBlock("selected source material", CacheControl()),))
    chat = store.create(connection="one", seed_history=[seed])
    reopened = store.open(chat.id)
    assert reopened.history()[0].to_dict() == seed.to_dict()
    list(reopened.send("consider this material"))
    assert adapter.requests[-1].messages[0].cache_breakpoints == 1


def test_keep_alive_reuses_native_session_and_refreshes_only_handlers(tmp_path):
    from modelpass import ToolDef

    connection = Connection(
        name="native", runtime=Runtime.ANTHROPIC_SDK, auth_mode="subscription", model="sonnet"
    )
    bridge, _, adapter = fake_session_bridge(
        connections=[connection], script=[TextDeltaEvent(text="reply")], home=tmp_path / "config"
    )
    store = bridge.conversations(tmp_path / "chats", application="soundbench", keep_alive=True)
    chat = store.create(connection="native", mode="native")

    def tools(value):
        return [ToolDef("current_page", "Read current page", {"type": "object"}, lambda _: value)]

    list(chat.send("first", tools=tools("first page")))
    session = store._live[chat.id]["session"]
    handle = adapter.handles[0]
    assert handle.closes == 0
    # The actual retained tool dispatcher gets the second turn's closure.
    original = handle.send
    observed = []

    def sending(message, **kwargs):
        observed.append(asyncio.run(handle.request.tools[0].handler({})))
        yield from original(message, **kwargs)

    handle.send = sending
    chat.rename("New title")  # metadata must not restart a native process
    list(store.open(chat.id).send("second", tools=tools("second page")))
    assert observed == ["second page"]
    assert len(adapter.handles) == 1
    assert store._live[chat.id]["session"] is session
    assert chat.info.native_session_id == "sess-1"
    store.close()
    assert handle.closes == 1
    list(store.open(chat.id).send("third", tools=tools("third page")))
    assert len(adapter.handles) == 2
    assert adapter.handles[-1].request.resume_id == "sess-1"
    store.close()


def test_native_tools_do_not_change_schema_or_leak_between_chats(tmp_path):
    from modelpass import ToolDef

    connection = Connection(
        name="native", runtime=Runtime.ANTHROPIC_SDK, auth_mode="subscription", model="sonnet"
    )
    bridge, _, adapter = fake_session_bridge(
        connections=[connection], script=[TextDeltaEvent(text="reply")], home=tmp_path / "config"
    )
    store = bridge.conversations(tmp_path / "chats", application="soundbench", keep_alive=True)
    one = store.create(connection="native", mode="native")
    two = store.create(connection="native", mode="native")
    tool = ToolDef("current_page", "Read page", {"type": "object"}, lambda _: "private")
    list(one.send("first", tools=[tool]))
    list(two.send("other", tools=[tool]))
    assert adapter.handles[0] is not adapter.handles[1]
    with pytest.raises(ConversationError, match="tools/guards changed"):
        list(one.send("changed", tools=[]))
    with pytest.raises(ConversationError, match="No active handler"):
        asyncio.run(adapter.handles[0].request.tools[0].handler({}))
    store.close()


def test_keep_alive_async_and_external_progress_resume_instead_of_stale_cache(tmp_path):
    connection = Connection(
        name="native", runtime=Runtime.ANTHROPIC_SDK, auth_mode="subscription", model="sonnet"
    )
    bridge, _, adapter = fake_session_bridge(
        connections=[connection], script=[TextDeltaEvent(text="reply")], home=tmp_path / "config"
    )
    store = bridge.conversations(tmp_path / "chats", application="soundbench", keep_alive=True)
    chat = store.create(connection="native", mode="native")

    async def turns():
        for question in ["first", "second"]:
            assert [event async for event in chat.asend(question)]

    asyncio.run(turns())
    assert len(adapter.handles) == 1
    other = bridge.conversations(tmp_path / "chats", application="soundbench")
    list(other.open(chat.id).send("external progress"))
    list(chat.send("after external progress"))
    assert len(adapter.handles) == 3
    assert adapter.handles[0].closes == 1
    store.close()


def test_keep_alive_is_bounded_and_cancelled_turn_is_not_reused(tmp_path):
    connection = Connection(
        name="native", runtime=Runtime.ANTHROPIC_SDK, auth_mode="subscription", model="sonnet"
    )
    bridge, _, adapter = fake_session_bridge(
        connections=[connection], script=[TextDeltaEvent(text="partial")], home=tmp_path / "config"
    )
    store = bridge.conversations(tmp_path / "chats", application="soundbench", keep_alive=True)
    for _ in range(5):
        list(store.create(connection="native", mode="native").send("first"))
    assert len(store._live) == 4
    assert adapter.handles[0].closes == 1
    chat = store.create(connection="native", mode="native")
    stream = chat.send("cancel me")
    next(stream)  # receipt
    next(stream)  # text
    with pytest.raises(ConversationBusy):
        store.close()
    stream.close()
    assert chat.id not in store._live
    assert chat.turns()[-1].status == "cancelled"
    store.close()
