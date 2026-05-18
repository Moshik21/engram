"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from engram import __version__
from engram.api.health_runtime import build_api_health_response
from engram.api.health_surface import (
    HealthResponse,
    ServiceStatus,
)

__all__ = ["HealthResponse", "ServiceStatus", "health_check"]

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return health status. Exempt from auth middleware."""
    return await build_api_health_response(version=__version__)
