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


async def build_api_conversation_list_surface(
    conversation_store: Any,
    *,
    group_id: str,
    limit: int,
) -> dict:
    """Return the REST conversation list payload for one brain group."""
    conversations = await list_group_conversations(
        conversation_store,
        group_id=group_id,
        limit=limit,
    )
    return {"conversations": conversations}


async def build_api_conversation_create_surface(
    conversation_store: Any,
    *,
    group_id: str,
    session_date: str | None = None,
    title: str | None = None,
) -> dict:
    """Create a conversation and return the REST acknowledgement payload."""
    conversation_id = await create_group_conversation(
        conversation_store,
        group_id=group_id,
        session_date=session_date,
        title=title,
    )
    return {"id": conversation_id}


async def build_api_conversation_messages_surface(
    conversation_store: Any,
    *,
    conversation_id: str,
    group_id: str,
) -> dict | None:
    """Return the REST message-list payload for an owned conversation."""
    messages = await get_group_conversation_messages(
        conversation_store,
        conversation_id=conversation_id,
        group_id=group_id,
    )
    if messages is None:
        return None
    return {"messages": messages}


async def build_api_conversation_append_messages_surface(
    conversation_store: Any,
    *,
    conversation_id: str,
    group_id: str,
    messages: list[dict],
) -> dict | None:
    """Append messages and return the REST acknowledgement payload."""
    message_ids = await append_group_conversation_messages(
        conversation_store,
        conversation_id=conversation_id,
        group_id=group_id,
        messages=messages,
    )
    if message_ids is None:
        return None
    return {"ids": message_ids}


async def build_api_conversation_update_surface(
    conversation_store: Any,
    *,
    conversation_id: str,
    group_id: str,
    title: str | None,
) -> dict | None:
    """Update a conversation title and return the REST acknowledgement payload."""
    updated = await update_group_conversation_title(
        conversation_store,
        conversation_id=conversation_id,
        group_id=group_id,
        title=title,
    )
    if not updated:
        return None
    return {"status": "updated"}


async def build_api_conversation_delete_surface(
    conversation_store: Any,
    *,
    conversation_id: str,
    group_id: str,
) -> dict | None:
    """Delete a conversation and return the REST acknowledgement payload."""
    deleted = await delete_group_conversation(
        conversation_store,
        conversation_id=conversation_id,
        group_id=group_id,
    )
    if not deleted:
        return None
    return {"status": "deleted"}


def conversation_not_found_payload() -> dict:
    """Return the shared REST 404 payload for conversation surfaces."""
    return {"detail": "Not found"}


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
