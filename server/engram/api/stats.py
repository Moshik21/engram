"""Stats endpoint for dashboard overview."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_graph_store, get_manager
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
    graph_store = get_graph_store()

    result = await manager.get_graph_state(
        group_id=group_id, top_n=20, include_edges=False
    )

    # Convert top_activated to camelCase
    top_activated = []
    for item in result.get("top_activated", []):
        top_activated.append(
            {
                "id": item["id"],
                "name": item["name"],
                "entityType": item["entity_type"],
                "summary": item["summary"],
                "activationCurrent": item["activation"],
                "accessCount": item["access_count"],
            }
        )

    # Top connected entities by edge degree
    top_connected = await graph_store.get_top_connected(group_id=group_id, limit=10)

    # Daily growth timeline
    growth_timeline = await graph_store.get_growth_timeline(group_id=group_id, days=days)

    return JSONResponse(
        content={
            "stats": result["stats"],
            "topActivated": top_activated,
            "topConnected": top_connected,
            "growthTimeline": growth_timeline,
            "groupId": result["group_id"],
        }
    )
