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
from engram.security.middleware import get_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/consolidation", tags=["consolidation"])


@router.post("/trigger")
async def trigger_consolidation(
    request: Request,
    background_tasks: BackgroundTasks,
    dry_run: bool = Query(
        True, description="Report what would change without modifying data",
    ),
) -> JSONResponse:
    """Trigger a consolidation cycle. Runs in the background."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    engine = get_consolidation_engine()

    if engine.is_running:
        return JSONResponse(
            status_code=409,
            content={"status": "conflict", "message": "A consolidation cycle is already running"},
        )

    async def _run():
        try:
            await engine.run_cycle(group_id=group_id, trigger="manual", dry_run=dry_run)
        except Exception:
            logger.exception("Background consolidation cycle failed")

    background_tasks.add_task(_run)

    return JSONResponse(content={
        "status": "triggered",
        "group_id": group_id,
        "dry_run": dry_run,
    })


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

    if engine._store:
        cycles = await engine._store.get_recent_cycles(group_id, limit=1)
        if cycles:
            latest = cycles[0]
            result["latest_cycle"] = {
                "id": latest.id,
                "status": latest.status,
                "dry_run": latest.dry_run,
                "trigger": latest.trigger,
                "started_at": latest.started_at,
                "completed_at": latest.completed_at,
                "total_duration_ms": latest.total_duration_ms,
                "phases": [
                    {
                        "phase": pr.phase,
                        "status": pr.status,
                        "items_processed": pr.items_processed,
                        "items_affected": pr.items_affected,
                    }
                    for pr in latest.phase_results
                ],
            }

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

    if not engine._store:
        return JSONResponse(content={"cycles": []})

    cycles = await engine._store.get_recent_cycles(group_id, limit=limit)
    return JSONResponse(content={
        "cycles": [
            {
                "id": c.id,
                "status": c.status,
                "dry_run": c.dry_run,
                "trigger": c.trigger,
                "started_at": c.started_at,
                "completed_at": c.completed_at,
                "total_duration_ms": c.total_duration_ms,
                "phases": [
                    {
                        "phase": pr.phase,
                        "status": pr.status,
                        "items_processed": pr.items_processed,
                        "items_affected": pr.items_affected,
                    }
                    for pr in c.phase_results
                ],
            }
            for c in cycles
        ],
    })


@router.get("/cycle/{cycle_id}")
async def consolidation_cycle_detail(
    request: Request,
    cycle_id: str,
) -> JSONResponse:
    """Get full detail for a specific consolidation cycle including audit records."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    engine = get_consolidation_engine()

    if not engine._store:
        return JSONResponse(status_code=404, content={"error": "Consolidation store not available"})

    cycle = await engine._store.get_cycle(cycle_id, group_id)
    if not cycle:
        return JSONResponse(status_code=404, content={"error": "Cycle not found"})

    merges = await engine._store.get_merge_records(cycle_id, group_id)
    inferred = await engine._store.get_inferred_edges(cycle_id, group_id)
    prunes = await engine._store.get_prune_records(cycle_id, group_id)
    reindexes = await engine._store.get_reindex_records(cycle_id, group_id)
    replays = await engine._store.get_replay_records(cycle_id, group_id)
    dreams = await engine._store.get_dream_records(cycle_id, group_id)

    return JSONResponse(content={
        "id": cycle.id,
        "status": cycle.status,
        "dry_run": cycle.dry_run,
        "trigger": cycle.trigger,
        "started_at": cycle.started_at,
        "completed_at": cycle.completed_at,
        "total_duration_ms": cycle.total_duration_ms,
        "error": cycle.error,
        "phases": [
            {
                "phase": pr.phase,
                "status": pr.status,
                "items_processed": pr.items_processed,
                "items_affected": pr.items_affected,
                "duration_ms": pr.duration_ms,
                "error": pr.error,
            }
            for pr in cycle.phase_results
        ],
        "merges": [
            {
                "id": m.id,
                "keep_id": m.keep_id,
                "remove_id": m.remove_id,
                "keep_name": m.keep_name,
                "remove_name": m.remove_name,
                "similarity": m.similarity,
                "relationships_transferred": m.relationships_transferred,
            }
            for m in merges
        ],
        "inferred_edges": [
            {
                "id": e.id,
                "source_id": e.source_id,
                "target_id": e.target_id,
                "source_name": e.source_name,
                "target_name": e.target_name,
                "co_occurrence_count": e.co_occurrence_count,
                "confidence": e.confidence,
                "infer_type": e.infer_type,
                "pmi_score": e.pmi_score,
                "llm_verdict": e.llm_verdict,
            }
            for e in inferred
        ],
        "prunes": [
            {
                "id": p.id,
                "entity_id": p.entity_id,
                "entity_name": p.entity_name,
                "entity_type": p.entity_type,
                "reason": p.reason,
            }
            for p in prunes
        ],
        "reindexes": [
            {
                "id": r.id,
                "entity_id": r.entity_id,
                "entity_name": r.entity_name,
                "source_phase": r.source_phase,
            }
            for r in reindexes
        ],
        "replays": [
            {
                "id": rp.id,
                "episode_id": rp.episode_id,
                "new_entities_found": rp.new_entities_found,
                "new_relationships_found": rp.new_relationships_found,
                "entities_updated": rp.entities_updated,
                "skipped_reason": rp.skipped_reason,
            }
            for rp in replays
        ],
        "dreams": [
            {
                "id": d.id,
                "source_entity_id": d.source_entity_id,
                "target_entity_id": d.target_entity_id,
                "weight_delta": d.weight_delta,
                "seed_entity_id": d.seed_entity_id,
            }
            for d in dreams
        ],
    })
