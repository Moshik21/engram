"""Stats endpoint for dashboard overview."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_manager
from engram.retrieval.graph_state import build_api_dashboard_stats_surface
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats")
async def get_stats(
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Days for growth timeline"),
) -> JSONResponse:
    """Return graph statistics, top-activated nodes, top-connected, and growth timeline."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    payload = await build_api_dashboard_stats_surface(manager, group_id=group_id, days=days)
    return JSONResponse(content=payload)
