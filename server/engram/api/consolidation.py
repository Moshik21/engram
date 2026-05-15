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
    build_api_consolidation_status_surface,
    build_api_consolidation_trigger_surface,
    run_api_consolidation_cycle,
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
    engine = get_consolidation_engine()

    result = build_api_consolidation_trigger_surface(
        engine,
        group_id=group_id,
        dry_run=dry_run,
    )
    if result.should_run:
        background_tasks.add_task(
            run_api_consolidation_cycle,
            engine,
            group_id=group_id,
            dry_run=dry_run,
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
    activation_cfg = get_config().activation if pressure is not None else None

    payload = await build_api_consolidation_status_surface(
        engine,
        group_id=group_id,
        scheduler=scheduler,
        pressure=pressure,
        activation_cfg=activation_cfg,
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
