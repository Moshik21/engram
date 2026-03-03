"""Health check endpoint."""

from __future__ import annotations

from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel

from engram import __version__

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
    from engram.main import _app_state

    services: dict[str, ServiceStatus] = {}

    if _app_state.get("graph_store"):
        try:
            await _app_state["graph_store"].get_stats()
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
        mode=_app_state.get("mode", "unknown"),
        services=services,
    )
