"""Storage visibility endpoint for operator and dashboard surfaces."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_storage_diagnostics
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api/storage", tags=["storage"])


@router.get("")
async def storage_summary(request: Request) -> JSONResponse:
    """Return resolved local storage paths, disk usage, and graph-count growth."""
    tenant = get_tenant(request)
    diagnostics = get_storage_diagnostics()
    payload = await diagnostics.snapshot(group_id=tenant.group_id)
    return JSONResponse(content=payload)
