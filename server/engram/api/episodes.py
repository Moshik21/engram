"""Episode listing endpoint with cursor pagination."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_manager
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
    manager = get_manager()
    payload = await manager.list_episode_summaries(
        group_id=tenant.group_id,
        cursor=cursor,
        limit=limit,
        source=source,
        status=status,
    )
    return JSONResponse(content=payload)
