"""Episode listing endpoint with cursor pagination."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_graph_store
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api/episodes", tags=["episodes"])


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
    for ep in episodes:
        items.append({
            "episodeId": ep.id,
            "content": ep.content[:200] if ep.content else None,
            "source": ep.source,
            "status": ep.status.value if hasattr(ep.status, "value") else ep.status,
            "createdAt": ep.created_at.isoformat() if ep.created_at else None,
            "updatedAt": ep.updated_at.isoformat() if ep.updated_at else None,
            "error": ep.error,
            "retryCount": ep.retry_count,
            "processingDurationMs": ep.processing_duration_ms,
            "entities": [],
            "factsCount": 0,
        })

    return JSONResponse(content={
        "items": items,
        "nextCursor": next_cursor,
        "total": len(items),
    })
