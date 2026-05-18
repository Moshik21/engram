"""Route-facing helpers for Capture-stage observe and remember writes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from json import JSONDecodeError
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


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _text_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return _text_value(value.get("content")) or _text_value(value.get("text"))
    if isinstance(value, list | tuple):
        parts = [part for item in value if (part := _text_value(item))]
        return "\n".join(parts) if parts else None
    return None


def _project_from_cwd(cwd: str | None) -> str | None:
    if not cwd:
        return None
    name = cwd.rstrip("/").rsplit("/", 1)[-1]
    return name or None


def _source_for_role(role: str | None) -> str:
    if role == "assistant":
        return "auto:response"
    if role == "user":
        return "auto:prompt"
    return "auto:hook"


def _tag_hook_content(role: str, project: str, content: str) -> str:
    return f"[{role}|{project}] {content}"


def normalize_auto_observe_payload(raw: Mapping[str, Any]) -> dict[str, str | None]:
    """Normalize installed-hook and raw Claude-hook payloads for auto-observe."""
    project = (
        _string_value(raw.get("project"))
        or _project_from_cwd(_string_value(raw.get("cwd")))
        or "unknown"
    )
    raw_role = _string_value(raw.get("role"))
    session_id = _string_value(raw.get("session_id"))
    conversation_date = _string_value(raw.get("conversation_date"))

    content = _text_value(raw.get("content"))
    if content is not None:
        role = raw_role or "user"
        return {
            "content": content,
            "source": _string_value(raw.get("source")) or _source_for_role(role),
            "project": project,
            "role": role,
            "session_id": session_id,
            "conversation_date": conversation_date,
        }

    prompt = _text_value(raw.get("prompt"))
    if prompt is not None:
        return {
            "content": _tag_hook_content("user", project, prompt),
            "source": _string_value(raw.get("source")) or "auto:prompt",
            "project": project,
            "role": "user",
            "session_id": session_id,
            "conversation_date": conversation_date,
        }

    assistant_response = _text_value(raw.get("last_assistant_message"))
    if assistant_response is not None:
        return {
            "content": _tag_hook_content("assistant", project, assistant_response),
            "source": _string_value(raw.get("source")) or "auto:response",
            "project": project,
            "role": "assistant",
            "session_id": session_id,
            "conversation_date": conversation_date,
        }

    message = raw.get("message")
    if isinstance(message, Mapping):
        role = (
            _string_value(message.get("role"))
            or raw_role
            or _string_value(raw.get("type"))
            or "system"
        )
        message_content = _text_value(message.get("content"))
        if message_content is not None:
            return {
                "content": _tag_hook_content(role, project, message_content),
                "source": _string_value(raw.get("source")) or _source_for_role(role),
                "project": project,
                "role": role,
                "session_id": session_id,
                "conversation_date": conversation_date,
            }

    return {
        "content": "",
        "source": _string_value(raw.get("source")) or "auto:hook",
        "project": project,
        "role": raw_role or "system",
        "session_id": session_id,
        "conversation_date": conversation_date,
    }


def build_api_auto_observe_skip_surface(reason: str) -> dict[str, Any]:
    """Return a non-throwing auto-observe skip response for hook compatibility."""
    return present_api_observe_skip("skipped", reason=reason)


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


async def build_api_auto_observe_request_surface(
    manager: Any,
    *,
    request: Any,
    group_id: str,
    auto_observe_enabled: bool = True,
    dedup_check: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Parse and run the REST auto-observe Capture policy behind one boundary."""
    try:
        raw_body = await request.json()
    except JSONDecodeError:
        return build_api_auto_observe_skip_surface("invalid_json")

    if not isinstance(raw_body, Mapping):
        return build_api_auto_observe_skip_surface("unsupported_payload")

    body = normalize_auto_observe_payload(raw_body)
    return await build_api_auto_observe_surface(
        manager,
        content=body["content"] or "",
        group_id=group_id,
        source=body["source"] or "auto:hook",
        session_id=body["session_id"],
        conversation_date=body["conversation_date"],
        auto_observe_enabled=auto_observe_enabled,
        dedup_check=dedup_check,
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
