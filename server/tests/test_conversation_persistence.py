from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.retrieval.conversation_persistence import (
    append_group_conversation_messages,
    create_group_conversation,
    delete_group_conversation,
    get_group_conversation_messages,
    list_group_conversations,
    update_group_conversation_title,
)
from engram.storage.sqlite.conversations import ConversationNotFoundError


@pytest.mark.asyncio
async def test_group_conversation_helpers_scope_store_calls() -> None:
    store = AsyncMock()
    store.list_conversations.return_value = [{"id": "conv_1"}]
    store.create_conversation.return_value = "conv_2"
    store.get_messages.return_value = [{"role": "user", "content": "hello"}]
    store.add_messages_bulk.return_value = ["msg_1"]
    store.update_conversation.return_value = True
    store.delete_conversation.return_value = True

    assert await list_group_conversations(store, group_id="brain_a", limit=5) == [
        {"id": "conv_1"}
    ]
    assert (
        await create_group_conversation(
            store,
            group_id="brain_a",
            session_date="2026-05-15",
            title="Daily",
        )
        == "conv_2"
    )
    assert await get_group_conversation_messages(
        store,
        conversation_id="conv_1",
        group_id="brain_a",
    ) == [{"role": "user", "content": "hello"}]
    assert await append_group_conversation_messages(
        store,
        conversation_id="conv_1",
        group_id="brain_a",
        messages=[{"role": "assistant", "content": "hi"}],
    ) == ["msg_1"]
    assert await update_group_conversation_title(
        store,
        conversation_id="conv_1",
        group_id="brain_a",
        title="Updated",
    )
    assert await delete_group_conversation(
        store,
        conversation_id="conv_1",
        group_id="brain_a",
    )

    store.list_conversations.assert_awaited_once_with("brain_a", limit=5)
    store.create_conversation.assert_awaited_once_with(
        group_id="brain_a",
        session_date="2026-05-15",
        title="Daily",
    )
    store.get_messages.assert_awaited_once_with("conv_1", "brain_a")
    store.add_messages_bulk.assert_awaited_once_with(
        "conv_1",
        [{"role": "assistant", "content": "hi"}],
        group_id="brain_a",
    )
    store.update_conversation.assert_awaited_once_with(
        "conv_1",
        "brain_a",
        title="Updated",
    )
    store.delete_conversation.assert_awaited_once_with("conv_1", "brain_a")


@pytest.mark.asyncio
async def test_group_conversation_helpers_return_none_for_missing_owned_conversation() -> None:
    store = AsyncMock()
    store.get_messages.side_effect = ConversationNotFoundError("conv_foreign")
    store.add_messages_bulk.side_effect = ConversationNotFoundError("conv_foreign")

    assert (
        await get_group_conversation_messages(
            store,
            conversation_id="conv_foreign",
            group_id="brain_a",
        )
        is None
    )
    assert (
        await append_group_conversation_messages(
            store,
            conversation_id="conv_foreign",
            group_id="brain_a",
            messages=[{"role": "user", "content": "blocked"}],
        )
        is None
    )
