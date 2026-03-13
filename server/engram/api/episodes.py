"""Episode listing endpoint with cursor pagination."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_graph_store
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api/episodes", tags=["episodes"])


def _enum_value(value: object) -> str | None:
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return value if isinstance(value, str) else None


def _iso_z(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return f"{value.isoformat()}Z"


def _serialize_cue(cue) -> dict | None:
    if cue is None:
        return None
    return {
        "cueText": cue.cue_text[:240] if cue.cue_text else None,
        "projectionState": _enum_value(cue.projection_state),
        "routeReason": cue.route_reason,
        "hitCount": cue.hit_count,
        "surfacedCount": cue.surfaced_count,
        "selectedCount": cue.selected_count,
        "usedCount": cue.used_count,
        "nearMissCount": cue.near_miss_count,
        "policyScore": cue.policy_score,
        "projectionAttempts": cue.projection_attempts,
        "lastHitAt": _iso_z(cue.last_hit_at),
        "lastFeedbackAt": _iso_z(cue.last_feedback_at),
        "lastProjectedAt": _iso_z(cue.last_projected_at),
    }


@router.get("")
async def list_episodes(
    request: Request,
    cursor: str | None = Query(None, description="Pagination cursor (ISO timestamp)"),
    limit: int = Query(50, ge=1, le=200, description="Page size"),
    source: str | None = Query(None, description="Filter by source"),
    status: str | None = Query(None, description="Filter by status"),
) -> JSONResponse:
    """List episodes with cursor-based pagination."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    graph_store = get_graph_store()

    episodes, next_cursor = await graph_store.get_episodes_paginated(
        group_id=group_id,
        cursor=cursor,
        limit=limit,
        source=source,
        status=status,
    )

    items = []
    get_episode_cue = getattr(graph_store, "get_episode_cue", None)
    for ep in episodes:
        cue = await get_episode_cue(ep.id, group_id) if get_episode_cue else None
        items.append(
            {
                "episodeId": ep.id,
                "content": ep.content[:200] if ep.content else None,
                "source": ep.source,
                "status": _enum_value(ep.status),
                "projectionState": _enum_value(getattr(ep, "projection_state", None)),
                "lastProjectionReason": getattr(ep, "last_projection_reason", None),
                "lastProjectedAt": _iso_z(getattr(ep, "last_projected_at", None)),
                "conversationDate": _iso_z(getattr(ep, "conversation_date", None)),
                "createdAt": _iso_z(ep.created_at),
                "updatedAt": _iso_z(ep.updated_at),
                "error": ep.error,
                "retryCount": ep.retry_count,
                "processingDurationMs": ep.processing_duration_ms,
                "entities": [],
                "factsCount": 0,
                "cue": _serialize_cue(cue),
            }
        )

    return JSONResponse(
        content={
            "items": items,
            "nextCursor": next_cursor,
            "total": len(items),
        }
    )
