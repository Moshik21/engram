"""Route-facing helpers for Capture-stage observe and remember writes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from engram.config import ActivationConfig
from engram.ingestion.adjudication_surface import load_client_enabled_episode_adjudication_requests
from engram.ingestion.presenter import (
    memory_write_contract,
    present_api_memory_write,
    present_api_observe_skip,
    present_mcp_memory_write,
)
from engram.models.episode import Attachment
from engram.utils.dates import utc_now


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


async def build_api_auto_observe_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str,
    session_id: str | None = None,
    conversation_date: str | None = None,
    auto_observe_enabled: bool = True,
    dedup_check: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Run the REST auto-observe Capture policy behind an ingestion boundary."""
    if not auto_observe_enabled:
        return present_api_observe_skip("skipped", reason="disabled")

    if not content or len(content.strip()) < 10:
        return present_api_observe_skip("skipped", reason="too_short")

    if dedup_check is not None and dedup_check(content):
        return present_api_observe_skip("dedup_skipped")

    episode_id = await store_observation(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        session_id=session_id,
        conversation_date=parse_conversation_date(conversation_date),
        pass_conversation_date=True,
    )
    return present_api_memory_write(
        memory_write_contract("observe", episode_id),
        status="observed",
    )


async def build_api_observe_write_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str,
    conversation_date: str | None = None,
) -> dict[str, Any]:
    """Run the REST observe write path behind a Capture-stage surface boundary."""
    episode_id = await store_observation(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        conversation_date=parse_conversation_date(conversation_date),
        pass_conversation_date=True,
    )
    return present_api_memory_write(
        memory_write_contract("observe", episode_id),
        status="observed",
    )


async def build_api_attachment_observe_write_surface(
    manager: Any,
    *,
    data_url: str,
    mime_type: str,
    attachment_kind: str,
    fallback_content: str,
    group_id: str,
    description: str = "",
    source: str = "api",
) -> dict[str, Any]:
    """Run a REST image/file observe write behind a Capture-stage boundary."""
    attachment = build_observation_attachment(
        mime_type=mime_type,
        data_url=data_url,
        description=description,
    )
    episode_id = await store_observation(
        manager,
        content=description or fallback_content,
        group_id=group_id,
        source=source,
        attachments=[attachment],
    )
    return present_api_memory_write(
        memory_write_contract("observe", episode_id, attachment_kind=attachment_kind),
        status="stored",
        include_legacy_episode_id=True,
    )


async def build_api_remember_write_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str,
    conversation_date: str | None = None,
    proposed_entities: list[dict] | None = None,
    proposed_relationships: list[dict] | None = None,
    model_tier: str = "default",
) -> dict[str, Any]:
    """Run the REST remember write path behind a Capture -> Project boundary."""
    episode_id = await ingest_projecting_memory(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        conversation_date=parse_conversation_date(conversation_date),
        proposed_entities=proposed_entities,
        proposed_relationships=proposed_relationships,
        model_tier=model_tier,
    )
    adjudications = await load_client_enabled_episode_adjudication_requests(
        manager,
        episode_id=episode_id,
        group_id=group_id,
    )
    return present_api_memory_write(
        memory_write_contract(
            "remember",
            episode_id,
            adjudication_requests=adjudications,
        ),
        status="remembered",
    )


def record_mcp_memory_write_activity(session: Any) -> None:
    """Update MCP session activity after a successful Capture write."""
    session.episode_count += 1
    session.last_activity = utc_now()


async def build_mcp_remember_write_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    session: Any,
    source: str = "mcp",
    conversation_date: str | None = None,
    proposed_entities: list[dict] | None = None,
    proposed_relationships: list[dict] | None = None,
    model_tier: str = "default",
    image_data: str | None = None,
    image_mime: str = "image/png",
    activation_cfg: ActivationConfig | None = None,
    ingest_live_turn: Callable[..., Any],
    recall_middleware: Callable[..., Any],
) -> dict[str, Any]:
    """Run the MCP remember write path behind a Capture-stage surface boundary."""
    attachments = []
    if image_data:
        attachments.append(
            build_observation_attachment(
                mime_type=image_mime,
                data_url=image_data,
                description=content[:200] if content else "",
            )
        )

    episode_id = await ingest_projecting_memory(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        session_id=session.session_id,
        conversation_date=parse_conversation_date(conversation_date),
        proposed_entities=proposed_entities,
        proposed_relationships=proposed_relationships,
        model_tier=model_tier,
        attachments=attachments or None,
        pass_session_id=True,
        pass_attachments=True,
    )
    record_mcp_memory_write_activity(session)
    await ingest_live_turn(manager, content, source="remember")

    message = "Memory received. Entities and relationships extracted."
    adjudications: list[dict] = []
    if activation_cfg and activation_cfg.evidence_extraction_enabled:
        message = "Memory received. Evidence extracted and evaluated."
        adjudications = await load_client_enabled_episode_adjudication_requests(
            manager,
            episode_id=episode_id,
            group_id=group_id,
            activation_cfg=activation_cfg,
        )

    response = present_mcp_memory_write(
        memory_write_contract(
            "remember",
            episode_id,
            adjudication_requests=adjudications,
        ),
        message=message,
    )
    await recall_middleware(content, response, tool_name="remember")
    return response


async def build_mcp_observe_write_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    session: Any,
    source: str = "mcp",
    conversation_date: str | None = None,
    ingest_live_turn: Callable[..., Any],
    recall_middleware: Callable[..., Any],
) -> dict[str, Any]:
    """Run the MCP observe write path behind a Capture-stage surface boundary."""
    episode_id = await store_observation(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        session_id=session.session_id,
        conversation_date=parse_conversation_date(conversation_date),
        pass_session_id=True,
        pass_conversation_date=True,
    )
    record_mcp_memory_write_activity(session)
    await ingest_live_turn(manager, content, source="observe")
    response = present_mcp_memory_write(
        memory_write_contract("observe", episode_id),
        message="Stored for background processing.",
    )
    await recall_middleware(content, response, tool_name="observe")
    return response


async def build_mcp_attachment_observe_write_surface(
    manager: Any,
    *,
    data_url: str,
    mime_type: str,
    attachment_kind: str,
    fallback_content: str,
    group_id: str,
    session: Any,
    description: str = "",
    source: str = "mcp",
) -> dict[str, Any]:
    """Run the MCP image/file observe path behind a Capture-stage boundary."""
    content = description or fallback_content
    attachment = build_observation_attachment(
        mime_type=mime_type,
        data_url=data_url,
        description=description,
    )
    episode_id = await store_observation(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        session_id=session.session_id,
        attachments=[attachment],
        pass_session_id=True,
    )
    record_mcp_memory_write_activity(session)
    return present_mcp_memory_write(
        memory_write_contract("observe", episode_id, attachment_kind=attachment_kind),
        message=f"{attachment_kind.title()} stored for background processing.",
    )
