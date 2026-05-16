"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from engram import __version__
from engram.api.deps import get_config, get_graph_store, get_mode
from engram.api.health_surface import (
    HealthResponse,
    ServiceStatus,
    build_api_health_surface,
)

__all__ = ["HealthResponse", "ServiceStatus", "health_check"]

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return health status. Exempt from auth middleware."""
    try:
        graph_store = get_graph_store()
    except RuntimeError:
        graph_store = None

    try:
        default_group_id = get_config().default_group_id
    except RuntimeError:
        default_group_id = "default"

    return await build_api_health_surface(
        graph_store=graph_store,
        default_group_id=default_group_id,
        version=__version__,
        mode=get_mode(),
    )
