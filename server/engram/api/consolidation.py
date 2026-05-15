"""REST endpoints for memory consolidation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import (
    get_consolidation_engine,
    get_consolidation_scheduler,
    get_pressure_accumulator,
)
from engram.consolidation.presenter import serialize_cycle_detail, serialize_cycle_summary
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

    if engine.is_running:
        return JSONResponse(
            status_code=409,
            content={"detail": "A consolidation cycle is already running"},
        )

    async def _run():
        try:
            await engine.run_cycle(group_id=group_id, trigger="manual", dry_run=dry_run)
        except Exception:
            logger.exception("Background consolidation cycle failed")

    background_tasks.add_task(_run)

    return JSONResponse(
        content={
            "status": "triggered",
            "group_id": group_id,
            "dry_run": dry_run,
        }
    )


@router.get("/status")
async def consolidation_status(request: Request) -> JSONResponse:
    """Get the latest consolidation cycle status."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    engine = get_consolidation_engine()

    scheduler = get_consolidation_scheduler()
    result: dict = {
        "is_running": engine.is_running,
        "scheduler_active": scheduler.is_active if scheduler else False,
    }

    # Include pressure snapshot if available
    pressure = get_pressure_accumulator()
    if pressure:
        snapshot = pressure.get_snapshot(group_id)
        if snapshot:
            from engram.api.deps import get_config

            cfg = get_config().activation
            result["pressure"] = {
                "value": round(pressure.get_pressure(group_id, cfg), 2),
                "threshold": cfg.consolidation_pressure_threshold,
                "episodes_since_last": snapshot.episodes_since_last,
                "entities_created": snapshot.entities_created,
                "last_cycle_time": snapshot.last_cycle_time,
            }

    latest_cycle = await engine.get_latest_cycle(group_id)
    if latest_cycle is not None:
        result["latest_cycle"] = serialize_cycle_summary(latest_cycle)

    return JSONResponse(content=result)


@router.get("/history")
async def consolidation_history(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
) -> JSONResponse:
    """Get recent consolidation cycle history."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    engine = get_consolidation_engine()

    cycles = await engine.get_recent_cycles(group_id, limit=limit)
    return JSONResponse(
        content={
            "cycles": [serialize_cycle_summary(c) for c in cycles],
        }
    )


@router.get("/cycle/{cycle_id}")
async def consolidation_cycle_detail(
    request: Request,
    cycle_id: str,
) -> JSONResponse:
    """Get full detail for a specific consolidation cycle including audit records."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    engine = get_consolidation_engine()

    if not engine.audit_store_available:
        return JSONResponse(
            status_code=404,
            content={"detail": "Consolidation store not available"},
        )

    detail = await engine.get_cycle_detail(cycle_id, group_id)
    if detail is None:
        return JSONResponse(status_code=404, content={"detail": "Cycle not found"})

    return JSONResponse(content=serialize_cycle_detail(detail))
