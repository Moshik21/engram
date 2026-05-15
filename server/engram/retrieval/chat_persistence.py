"""Conversation persistence helpers for knowledge chat."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from engram.retrieval.conversation_persistence import CONVERSATION_NOT_FOUND_ERRORS


@dataclass(frozen=True)
class ChatConversationResolution:
    """Resolved chat conversation target for a REST chat request."""

    conversation_id: str | None
    not_found: bool = False


async def resolve_chat_conversation(
    conversation_store: Any | None,
    *,
    group_id: str,
    conversation_id: str | None,
    message: str,
    session_date: str | None,
) -> ChatConversationResolution:
    """Validate or create the persisted conversation for a chat request."""
    if conversation_store is None:
        return ChatConversationResolution(conversation_id=conversation_id)

    if conversation_id:
        try:
            await conversation_store.get_conversation(conversation_id, group_id)
        except CONVERSATION_NOT_FOUND_ERRORS:
            return ChatConversationResolution(conversation_id=conversation_id, not_found=True)
        return ChatConversationResolution(conversation_id=conversation_id)

    title = message[:60].strip()
    created_id = await conversation_store.create_conversation(
        group_id=group_id,
        session_date=session_date,
        title=title,
    )
    return ChatConversationResolution(conversation_id=created_id)


async def persist_chat_turn(
    conversation_store: Any | None,
    *,
    conversation_id: str | None,
    group_id: str,
    user_message: str,
    assistant_message: str | None,
    recall_results: Sequence[Mapping[str, Any]],
) -> None:
    """Persist a completed chat turn and tag recalled entity references."""
    if conversation_store is None or not conversation_id or not assistant_message:
        return

    await conversation_store.add_messages_bulk(
        conversation_id,
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        group_id=group_id,
    )

    for entity_id in _entity_ids_from_recall_results(recall_results):
        await conversation_store.tag_entity(
            conversation_id,
            entity_id,
            group_id=group_id,
        )


def _entity_ids_from_recall_results(
    recall_results: Sequence[Mapping[str, Any]],
) -> list[str]:
    entity_ids: set[str] = set()
    for result in recall_results:
        if result.get("result_type") != "entity":
            continue
        entity = result.get("entity")
        if not isinstance(entity, Mapping):
            continue
        entity_id = entity.get("id")
        if isinstance(entity_id, str) and entity_id:
            entity_ids.add(entity_id)
    return sorted(entity_ids)
