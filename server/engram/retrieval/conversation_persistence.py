"""Group-scoped conversation persistence helpers."""

from __future__ import annotations

from typing import Any

from engram.storage.helix.conversations import (
    ConversationNotFoundError as HelixConversationNotFoundError,
)
from engram.storage.sqlite.conversations import (
    ConversationNotFoundError as SQLiteConversationNotFoundError,
)

CONVERSATION_NOT_FOUND_ERRORS = (
    SQLiteConversationNotFoundError,
    HelixConversationNotFoundError,
)


async def list_group_conversations(
    conversation_store: Any,
    *,
    group_id: str,
    limit: int,
) -> list[dict]:
    """List persisted conversations for one brain group."""
    return await conversation_store.list_conversations(group_id, limit=limit)


async def create_group_conversation(
    conversation_store: Any,
    *,
    group_id: str,
    session_date: str | None = None,
    title: str | None = None,
) -> str:
    """Create a persisted conversation for one brain group."""
    return await conversation_store.create_conversation(
        group_id=group_id,
        session_date=session_date,
        title=title,
    )


async def get_group_conversation_messages(
    conversation_store: Any,
    *,
    conversation_id: str,
    group_id: str,
) -> list[dict] | None:
    """Return messages for an owned conversation, or None when not found."""
    try:
        return await conversation_store.get_messages(conversation_id, group_id)
    except CONVERSATION_NOT_FOUND_ERRORS:
        return None


async def append_group_conversation_messages(
    conversation_store: Any,
    *,
    conversation_id: str,
    group_id: str,
    messages: list[dict],
) -> list[str] | None:
    """Append messages to an owned conversation, or None when not found."""
    try:
        return await conversation_store.add_messages_bulk(
            conversation_id,
            messages,
            group_id=group_id,
        )
    except CONVERSATION_NOT_FOUND_ERRORS:
        return None


async def update_group_conversation_title(
    conversation_store: Any,
    *,
    conversation_id: str,
    group_id: str,
    title: str | None,
) -> bool:
    """Update an owned conversation title."""
    return bool(
        await conversation_store.update_conversation(
            conversation_id,
            group_id,
            title=title,
        )
    )


async def delete_group_conversation(
    conversation_store: Any,
    *,
    conversation_id: str,
    group_id: str,
) -> bool:
    """Delete an owned conversation."""
    return bool(await conversation_store.delete_conversation(conversation_id, group_id))
