"""REST endpoints for local brain-loop evaluation labels and reports."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from engram.api.deps import (
    get_consolidation_engine,
    get_evaluation_store,
    get_manager,
)
from engram.evaluation.label_service import (
    build_recall_evaluation_write_surface,
    build_session_continuity_evaluation_write_surface,
)
from engram.evaluation.report_service import build_brain_loop_evaluation_surface
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class RecallSampleRequest(BaseModel):
    """Operator label for one recall decision."""

    recall_triggered: bool = Field(alias="recallTriggered")
    recall_helped: bool = Field(alias="recallHelped")
    recall_needed: bool | None = Field(default=None, alias="recallNeeded")
    packets_surfaced: int = Field(default=0, ge=0, alias="packetsSurfaced")
    packets_used: int = Field(default=0, ge=0, alias="packetsUsed")
    false_recalls: int = Field(default=0, ge=0, alias="falseRecalls")
    source: str = "manual"
    query: str | None = None
    notes: str | None = None

    model_config = {"populate_by_name": True}


class SessionSampleRequest(BaseModel):
    """Operator label for one multi-turn continuity task."""

    baseline_score: float = Field(alias="baselineScore")
    memory_score: float = Field(alias="memoryScore")
    open_loop_expected: bool = Field(default=False, alias="openLoopExpected")
    open_loop_recovered: bool = Field(default=False, alias="openLoopRecovered")
    temporal_expected: bool = Field(default=False, alias="temporalExpected")
    temporal_correct: bool = Field(default=False, alias="temporalCorrect")
    source: str = "manual"
    scenario: str | None = None
    notes: str | None = None

    model_config = {"populate_by_name": True}


@router.post("/recall-samples")
async def create_recall_sample(
    request: Request,
    body: RecallSampleRequest,
) -> JSONResponse:
    """Persist a labeled recall-quality sample for the active group."""
    tenant = get_tenant(request)
    store = get_evaluation_store()
    payload = await build_recall_evaluation_write_surface(
        store,
        group_id=tenant.group_id,
        surface="rest",
        recall_triggered=body.recall_triggered,
        recall_helped=body.recall_helped,
        recall_needed=body.recall_needed,
        packets_surfaced=body.packets_surfaced,
        packets_used=body.packets_used,
        false_recalls=body.false_recalls,
        source=body.source,
        query=body.query,
        notes=body.notes,
    )
    return JSONResponse(status_code=201, content=payload)


@router.post("/session-samples")
async def create_session_sample(
    request: Request,
    body: SessionSampleRequest,
) -> JSONResponse:
    """Persist a labeled session-continuity sample for the active group."""
    tenant = get_tenant(request)
    store = get_evaluation_store()
    payload = await build_session_continuity_evaluation_write_surface(
        store,
        group_id=tenant.group_id,
        surface="rest",
        baseline_score=body.baseline_score,
        memory_score=body.memory_score,
        open_loop_expected=body.open_loop_expected,
        open_loop_recovered=body.open_loop_recovered,
        temporal_expected=body.temporal_expected,
        temporal_correct=body.temporal_correct,
        source=body.source,
        scenario=body.scenario,
        notes=body.notes,
    )
    return JSONResponse(status_code=201, content=payload)


@router.get("/brain-loop/report")
async def brain_loop_evaluation_report(
    request: Request,
    cycle_limit: int = Query(10, ge=1, le=100, alias="cycleLimit"),
    sample_limit: int = Query(500, ge=1, le=5000, alias="sampleLimit"),
) -> JSONResponse:
    """Return the local brain-loop evaluation report for the active group."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    evaluation_store = get_evaluation_store()
    engine = get_consolidation_engine()

    recent_cycles, calibration_snapshots = await engine.get_recent_evaluation_context(
        group_id,
        cycle_limit=cycle_limit,
    )
    report = await build_brain_loop_evaluation_surface(
        manager,
        evaluation_store,
        group_id=group_id,
        recent_cycles=recent_cycles,
        calibration_snapshots=calibration_snapshots,
        sample_limit=sample_limit,
        snapshot_source="rest_report",
    )
    return JSONResponse(content=report)
