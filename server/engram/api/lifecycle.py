"""Lifecycle summary endpoint for the brain-loop dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from engram.api.deps import (
    get_consolidation_engine,
    get_consolidation_scheduler,
    get_manager,
    get_pressure_accumulator,
)
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api/lifecycle", tags=["lifecycle"])


@router.get("/summary")
async def lifecycle_summary(request: Request) -> JSONResponse:
    """Return a semantic Capture -> Cue -> Project -> Recall -> Consolidate summary."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    engine = get_consolidation_engine()
    scheduler = get_consolidation_scheduler()
    summary = await manager.get_lifecycle_summary(
        group_id=group_id,
        consolidation_engine=engine,
        consolidation_scheduler=scheduler,
        pressure_accumulator=get_pressure_accumulator(),
    )
    return JSONResponse(content=summary)
