"""Storage visibility endpoint for operator and dashboard surfaces."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_storage_diagnostics
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api/storage", tags=["storage"])


@router.get("")
async def storage_summary(
    request: Request,
    live: bool = Query(
        True,
        description=(
            "Refresh graph counts and disk paths before returning the report. "
            "Default true so operator counts match lifecycle (not write-through deltas)."
        ),
    ),
    timeout_seconds: float | None = Query(
        8.0,
        alias="timeoutSeconds",
        ge=0,
        le=30,
        description="Optional live refresh budget in seconds.",
    ),
) -> JSONResponse:
    """Return resolved local storage paths, disk usage, and graph-count growth."""
    tenant = get_tenant(request)
    diagnostics = get_storage_diagnostics()
    payload = await diagnostics.snapshot(
        group_id=tenant.group_id,
        live=live,
        timeout_seconds=timeout_seconds,
    )
    return JSONResponse(content=payload)
