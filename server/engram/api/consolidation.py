"""REST endpoints for memory consolidation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import (
    get_config,
    get_consolidation_engine,
    get_consolidation_scheduler,
    get_pressure_accumulator,
)
from engram.consolidation_trigger import (
    build_api_consolidation_cycle_detail_surface,
    build_api_consolidation_history_surface,
    build_api_consolidation_status_response_surface,
    build_api_consolidation_trigger_response_surface,
)
from engram.security.middleware import get_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/consolidation", tags=["consolidation"])


@router.post("/trigger")
async def trigger_consolidation(
    request: Request,
    background_tasks: BackgroundTasks,
    dry_run: bool = Query(
        True,
        description="Report what would change without modifying data",
    ),
) -> JSONResponse:
    """Trigger a consolidation cycle. Runs in the background."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    config = get_config()
    if not config.shell_runs_in_process_brain():
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "error": (
                    f"runtime_role={config.runtime_role}: consolidation runs in "
                    "the cold brain, not the hot shell. Use 'engram brain run'."
                ),
            },
        )
    engine = get_consolidation_engine()

    result = build_api_consolidation_trigger_response_surface(
        engine,
        group_id=group_id,
        dry_run=dry_run,
        background_tasks=background_tasks,
        logger=logger,
    )
    return JSONResponse(status_code=result.status_code, content=result.payload)


@router.get("/status")
async def consolidation_status(request: Request) -> JSONResponse:
    """Get the latest consolidation cycle status."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    engine = get_consolidation_engine()
    scheduler = get_consolidation_scheduler()
    pressure = get_pressure_accumulator()
    payload = await build_api_consolidation_status_response_surface(
        engine,
        group_id=group_id,
        scheduler=scheduler,
        pressure=pressure,
        config=get_config(),
    )
    return JSONResponse(content=payload)


@router.get("/history")
async def consolidation_history(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
) -> JSONResponse:
    """Get recent consolidation cycle history."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    engine = get_consolidation_engine()

    payload = await build_api_consolidation_history_surface(
        engine,
        group_id=group_id,
        limit=limit,
    )
    return JSONResponse(content=payload)


@router.get("/cycle/{cycle_id}")
async def consolidation_cycle_detail(
    request: Request,
    cycle_id: str,
) -> JSONResponse:
    """Get full detail for a specific consolidation cycle including audit records."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    engine = get_consolidation_engine()

    result = await build_api_consolidation_cycle_detail_surface(
        engine,
        group_id=group_id,
        cycle_id=cycle_id,
    )
    return JSONResponse(status_code=result.status_code, content=result.payload)
