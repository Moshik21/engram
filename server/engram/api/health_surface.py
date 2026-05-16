"""Health response assembly for public API routes."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    status: ServiceStatus
    version: str
    mode: str
    services: dict[str, ServiceStatus]


def aggregate_service_statuses(services: dict[str, ServiceStatus]) -> ServiceStatus:
    """Return the overall health status from individual service statuses."""
    if all(status == ServiceStatus.HEALTHY for status in services.values()):
        return ServiceStatus.HEALTHY
    if any(status == ServiceStatus.UNHEALTHY for status in services.values()):
        return ServiceStatus.UNHEALTHY
    return ServiceStatus.DEGRADED


async def probe_graph_store_health(
    graph_store: Any | None,
    *,
    group_id: str,
) -> ServiceStatus:
    """Probe the graph store for the active default brain group."""
    if graph_store is None:
        return ServiceStatus.UNHEALTHY
    try:
        await graph_store.get_stats(group_id=group_id)
    except Exception:
        return ServiceStatus.UNHEALTHY
    return ServiceStatus.HEALTHY


async def build_api_health_surface(
    *,
    graph_store: Any | None,
    default_group_id: str = "default",
    mode: str,
    version: str,
) -> HealthResponse:
    """Build the public API health response."""
    services = {
        "graph_store": await probe_graph_store_health(
            graph_store,
            group_id=default_group_id,
        )
    }
    return HealthResponse(
        status=aggregate_service_statuses(services),
        version=version,
        mode=mode,
        services=services,
    )
