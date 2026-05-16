from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.retrieval.chat_persistence import (
    chat_conversation_not_found_payload,
    persist_chat_turn,
    resolve_chat_conversation,
    schedule_chat_turn_persistence,
)
from engram.storage.sqlite.conversations import ConversationNotFoundError


@pytest.mark.asyncio
async def test_resolve_chat_conversation_validates_existing_group() -> None:
    store = AsyncMock()

    result = await resolve_chat_conversation(
        store,
        group_id="brain_a",
        conversation_id="conv_1",
        message="ignored",
        session_date=None,
    )

    assert result.conversation_id == "conv_1"
    assert result.not_found is False
    store.get_conversation.assert_awaited_once_with("conv_1", "brain_a")
    store.create_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_chat_conversation_creates_when_missing_id() -> None:
    store = AsyncMock()
    store.create_conversation.return_value = "conv_new"

    result = await resolve_chat_conversation(
        store,
        group_id="brain_a",
        conversation_id=None,
        message="A" * 80,
        session_date="2026-05-15",
    )

    assert result.conversation_id == "conv_new"
    assert result.not_found is False
    store.create_conversation.assert_awaited_once_with(
        group_id="brain_a",
        session_date="2026-05-15",
        title="A" * 60,
    )


@pytest.mark.asyncio
async def test_resolve_chat_conversation_reports_not_found_for_foreign_id() -> None:
    store = AsyncMock()
    store.get_conversation.side_effect = ConversationNotFoundError("conv_foreign")

    result = await resolve_chat_conversation(
        store,
        group_id="brain_a",
        conversation_id="conv_foreign",
        message="ignored",
        session_date=None,
    )

    assert result.conversation_id == "conv_foreign"
    assert result.not_found is True


@pytest.mark.asyncio
async def test_resolve_chat_conversation_noops_without_store() -> None:
    result = await resolve_chat_conversation(
        None,
        group_id="brain_a",
        conversation_id="conv_client",
        message="ignored",
        session_date=None,
    )

    assert result.conversation_id == "conv_client"
    assert result.not_found is False


def test_chat_conversation_not_found_payload() -> None:
    assert chat_conversation_not_found_payload() == {"detail": "Conversation not found"}


@pytest.mark.asyncio
async def test_persist_chat_turn_saves_messages_and_tags_unique_entities() -> None:
    store = AsyncMock()

    await persist_chat_turn(
        store,
        conversation_id="conv_1",
        group_id="brain_a",
        user_message="What changed?",
        assistant_message="The recall path changed.",
        recall_results=[
            {"result_type": "entity", "entity": {"id": "ent_a"}},
            {"result_type": "entity", "entity": {"id": "ent_a"}},
            {"result_type": "entity", "entity": {"id": "ent_b"}},
            {"result_type": "cue_episode", "cue": {"episode_id": "ep_1"}},
        ],
    )

    store.add_messages_bulk.assert_awaited_once_with(
        "conv_1",
        [
            {"role": "user", "content": "What changed?"},
            {"role": "assistant", "content": "The recall path changed."},
        ],
        group_id="brain_a",
    )
    assert store.tag_entity.await_args_list[0].args == ("conv_1", "ent_a")
    assert store.tag_entity.await_args_list[0].kwargs == {"group_id": "brain_a"}
    assert store.tag_entity.await_args_list[1].args == ("conv_1", "ent_b")
    assert store.tag_entity.await_args_list[1].kwargs == {"group_id": "brain_a"}


@pytest.mark.asyncio
async def test_persist_chat_turn_noops_without_store_conversation_or_assistant() -> None:
    store = AsyncMock()

    await persist_chat_turn(
        None,
        conversation_id="conv_1",
        group_id="brain_a",
        user_message="hello",
        assistant_message="hi",
        recall_results=[],
    )
    await persist_chat_turn(
        store,
        conversation_id=None,
        group_id="brain_a",
        user_message="hello",
        assistant_message="hi",
        recall_results=[],
    )
    await persist_chat_turn(
        store,
        conversation_id="conv_1",
        group_id="brain_a",
        user_message="hello",
        assistant_message="",
        recall_results=[],
    )

    store.add_messages_bulk.assert_not_called()
    store.tag_entity.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_chat_turn_persistence_schedules_completed_turn() -> None:
    store = AsyncMock()
    scheduled = []

    scheduled_result = schedule_chat_turn_persistence(
        store,
        conversation_id="conv_1",
        group_id="brain_a",
        user_message="What changed?",
        assistant_message="The recall path changed.",
        recall_results=[{"result_type": "entity", "entity": {"id": "ent_a"}}],
        create_task=scheduled.append,
    )

    assert scheduled_result is True
    assert len(scheduled) == 1
    await scheduled[0]
    store.add_messages_bulk.assert_awaited_once_with(
        "conv_1",
        [
            {"role": "user", "content": "What changed?"},
            {"role": "assistant", "content": "The recall path changed."},
        ],
        group_id="brain_a",
    )
    store.tag_entity.assert_awaited_once_with("conv_1", "ent_a", group_id="brain_a")


def test_schedule_chat_turn_persistence_noops_without_persistable_turn() -> None:
    store = AsyncMock()
    scheduled = []

    assert (
        schedule_chat_turn_persistence(
            None,
            conversation_id="conv_1",
            group_id="brain_a",
            user_message="hello",
            assistant_message="hi",
            recall_results=[],
            create_task=scheduled.append,
        )
        is False
    )
    assert (
        schedule_chat_turn_persistence(
            store,
            conversation_id=None,
            group_id="brain_a",
            user_message="hello",
            assistant_message="hi",
            recall_results=[],
            create_task=scheduled.append,
        )
        is False
    )
    assert (
        schedule_chat_turn_persistence(
            store,
            conversation_id="conv_1",
            group_id="brain_a",
            user_message="hello",
            assistant_message="",
            recall_results=[],
            create_task=scheduled.append,
        )
        is False
    )
    assert scheduled == []


@pytest.mark.asyncio
async def test_schedule_chat_turn_persistence_logs_and_swallows_failure(caplog) -> None:
    store = AsyncMock()
    store.add_messages_bulk.side_effect = RuntimeError("write failed")
    scheduled = []

    assert schedule_chat_turn_persistence(
        store,
        conversation_id="conv_1",
        group_id="brain_a",
        user_message="hello",
        assistant_message="hi",
        recall_results=[],
        create_task=scheduled.append,
    )

    await scheduled[0]
    assert "Failed to persist chat messages" in caplog.text
