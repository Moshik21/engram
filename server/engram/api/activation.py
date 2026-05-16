"""Activation monitor API endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_manager
from engram.retrieval.graph_state import build_api_activation_curve_surface
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api/activation", tags=["activation"])


@router.get("/snapshot")
async def get_activation_snapshot(request: Request, limit: int = 50):
    """Top activated entities with scores and access metadata."""
    tenant = get_tenant(request)
    manager = get_manager()
    return await manager.get_activation_snapshot(group_id=tenant.group_id, limit=limit)


@router.get("/{entity_id}/curve")
async def get_activation_curve(
    request: Request,
    entity_id: str,
    hours: int = 24,
    points: int = 48,
):
    """Simulated ACT-R decay curve over past N hours."""
    tenant = get_tenant(request)
    manager = get_manager()
    result = await build_api_activation_curve_surface(
        manager,
        group_id=tenant.group_id,
        entity_id=entity_id,
        hours=hours,
        points=points,
    )
    return JSONResponse(status_code=result.status_code, content=result.payload)
