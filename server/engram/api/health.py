"""Health check endpoint."""

from __future__ import annotations

from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel

from engram import __version__
from engram.api.deps import get_config, get_graph_store, get_mode

router = APIRouter()


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    status: ServiceStatus
    version: str
    mode: str
    services: dict[str, ServiceStatus]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return health status. Exempt from auth middleware."""
    services: dict[str, ServiceStatus] = {}

    try:
        graph_store = get_graph_store()
    except RuntimeError:
        graph_store = None

    if graph_store:
        try:
            try:
                group_id = get_config().default_group_id
            except RuntimeError:
                group_id = "default"
            await graph_store.get_stats(group_id=group_id)
            services["graph_store"] = ServiceStatus.HEALTHY
        except Exception:
            services["graph_store"] = ServiceStatus.UNHEALTHY
    else:
        services["graph_store"] = ServiceStatus.UNHEALTHY

    if all(s == ServiceStatus.HEALTHY for s in services.values()):
        status = ServiceStatus.HEALTHY
    elif any(s == ServiceStatus.UNHEALTHY for s in services.values()):
        status = ServiceStatus.UNHEALTHY
    else:
        status = ServiceStatus.DEGRADED

    return HealthResponse(
        status=status,
        version=__version__,
        mode=get_mode(),
        services=services,
    )
