"""Route-facing helpers for Capture-stage observe and remember writes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from engram.models.episode import Attachment


def parse_conversation_date(value: str | None) -> datetime | None:
    """Parse optional ISO conversation dates without failing public writes."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def build_observation_attachment(
    *,
    mime_type: str,
    data_url: str,
    description: str | None = None,
) -> Attachment:
    """Build the attachment payload used by image/file observe surfaces."""
    return Attachment(
        mime_type=mime_type,
        data_url=data_url,
        description=description,
    )


async def store_observation(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str | None = None,
    session_id: str | None = None,
    conversation_date: datetime | None = None,
    attachments: list[Attachment] | None = None,
    pass_session_id: bool = False,
    pass_conversation_date: bool = False,
    pass_attachments: bool = False,
) -> str:
    """Store a raw Capture-stage observation through the manager facade."""
    kwargs: dict[str, Any] = {
        "content": content,
        "group_id": group_id,
        "source": source,
    }
    if session_id is not None or pass_session_id:
        kwargs["session_id"] = session_id
    if conversation_date is not None or pass_conversation_date:
        kwargs["conversation_date"] = conversation_date
    if attachments is not None or pass_attachments:
        kwargs["attachments"] = attachments
    return await manager.store_episode(**kwargs)


async def ingest_projecting_memory(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str | None = None,
    session_id: str | None = None,
    conversation_date: datetime | None = None,
    proposed_entities: list[dict] | None = None,
    proposed_relationships: list[dict] | None = None,
    model_tier: str = "default",
    attachments: list[Attachment] | None = None,
    pass_session_id: bool = False,
    pass_attachments: bool = False,
) -> str:
    """Run the Capture -> Project compatibility write through the manager facade."""
    kwargs: dict[str, Any] = {
        "content": content,
        "group_id": group_id,
        "source": source,
        "conversation_date": conversation_date,
        "proposed_entities": proposed_entities,
        "proposed_relationships": proposed_relationships,
        "model_tier": model_tier,
    }
    if session_id is not None or pass_session_id:
        kwargs["session_id"] = session_id
    if attachments is not None or pass_attachments:
        kwargs["attachments"] = attachments
    return await manager.ingest_episode(**kwargs)
