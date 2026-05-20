"""Health response assembly for public API routes."""

from __future__ import annotations

import asyncio
import logging
import os
from enum import Enum
from typing import Any

from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)
HEALTH_PROBE_TIMEOUT_ENV = "ENGRAM_HEALTH_PROBE_TIMEOUT_SECONDS"
DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS = 2.0


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
    timeout_seconds: float | None = None,
) -> ServiceStatus:
    """Probe the graph store for the active default brain group."""
    if graph_store is None:
        return ServiceStatus.UNHEALTHY
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else _health_probe_timeout_seconds()
    )
    health_check = getattr(graph_store, "health_check", None)
    try:
        if health_check is not None:
            probe = health_check(group_id=group_id)
        else:
            probe = graph_store.get_stats(group_id=group_id)
        if timeout <= 0:
            await probe
        else:
            await asyncio.wait_for(probe, timeout=timeout)
    except TimeoutError:
        LOGGER.warning(
            "Graph store health probe timed out after %.1f seconds",
            timeout,
        )
        return ServiceStatus.DEGRADED
    except Exception:
        return ServiceStatus.UNHEALTHY
    return ServiceStatus.HEALTHY


def _health_probe_timeout_seconds() -> float:
    raw = os.environ.get(
        HEALTH_PROBE_TIMEOUT_ENV,
        str(DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS),
    )
    try:
        return float(raw)
    except ValueError:
        LOGGER.warning(
            "Invalid %s=%r; using %.1f seconds",
            HEALTH_PROBE_TIMEOUT_ENV,
            raw,
            DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS,
        )
        return DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS


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
