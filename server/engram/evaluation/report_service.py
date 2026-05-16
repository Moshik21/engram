"""Shared brain-loop evaluation report assembly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from engram.consolidation.audit_reader import ConsolidationAuditReader
from engram.evaluation.brain_loop_report import (
    build_brain_loop_report,
    has_recall_runtime_metrics,
    merge_recall_runtime_metrics,
)
from engram.evaluation.store import StoredRecallRuntimeMetricsSnapshot


async def load_consolidation_evaluation_inputs(
    consolidation_store: Any | None,
    *,
    group_id: str,
    cycle_limit: int,
) -> tuple[list[Any], list[Any]]:
    """Read recent consolidation context from an optional audit store."""
    return await ConsolidationAuditReader(consolidation_store).evaluation_context(
        group_id,
        cycle_limit=max(1, cycle_limit),
    )


async def build_mcp_evaluation_report_surface(
    manager: Any,
    evaluation_store: Any,
    *,
    consolidation_store: Any | None,
    group_id: str,
    cycle_limit: int,
    sample_limit: int,
) -> dict[str, Any]:
    """Build the MCP brain-loop report from active MCP stores."""
    recent_cycles, calibration_snapshots = await load_consolidation_evaluation_inputs(
        consolidation_store,
        group_id=group_id,
        cycle_limit=cycle_limit,
    )
    return await build_brain_loop_evaluation_surface(
        manager,
        evaluation_store,
        group_id=group_id,
        recent_cycles=recent_cycles,
        calibration_snapshots=calibration_snapshots,
        sample_limit=max(1, sample_limit),
        snapshot_source="mcp_report",
    )


async def build_api_brain_loop_evaluation_surface(
    manager: Any,
    evaluation_store: Any,
    consolidation_engine: Any,
    *,
    group_id: str,
    cycle_limit: int,
    sample_limit: int,
) -> dict[str, Any]:
    """Build the REST brain-loop report from active runtime services."""
    recent_cycles, calibration_snapshots = await load_engine_evaluation_context(
        consolidation_engine,
        group_id=group_id,
        cycle_limit=cycle_limit,
    )
    return await build_brain_loop_evaluation_surface(
        manager,
        evaluation_store,
        group_id=group_id,
        recent_cycles=recent_cycles,
        calibration_snapshots=calibration_snapshots,
        sample_limit=max(1, sample_limit),
        snapshot_source="rest_report",
    )


async def load_engine_evaluation_context(
    consolidation_engine: Any,
    *,
    group_id: str,
    cycle_limit: int,
) -> tuple[list[Any], list[Any]]:
    """Read evaluation context through the consolidation engine facade."""
    return await consolidation_engine.get_recent_evaluation_context(
        group_id,
        cycle_limit=max(1, cycle_limit),
    )


async def build_brain_loop_evaluation_surface(
    manager: Any,
    evaluation_store: Any,
    *,
    group_id: str,
    recent_cycles: Sequence[Any],
    calibration_snapshots: Sequence[Any],
    sample_limit: int,
    snapshot_source: str,
) -> dict[str, Any]:
    """Build the REST/MCP brain-loop report from live graph and saved labels."""
    graph_state = await manager.get_graph_state(
        group_id=group_id,
        top_n=10,
        include_edges=False,
    )
    stats = graph_state.get("stats") or {}
    recall_metrics = stats.get("recall_metrics") or {}
    if has_recall_runtime_metrics(recall_metrics):
        await evaluation_store.save_recall_metrics_snapshot(
            StoredRecallRuntimeMetricsSnapshot(
                group_id=group_id,
                metrics=dict(recall_metrics),
                source=snapshot_source,
            )
        )
    else:
        stats = merge_recall_runtime_metrics(
            stats,
            await evaluation_store.get_latest_recall_metrics_snapshot(group_id),
        )

    recall_samples = await evaluation_store.get_recall_samples(
        group_id,
        limit=sample_limit,
    )
    session_samples = await evaluation_store.get_session_samples(
        group_id,
        limit=sample_limit,
    )
    return build_brain_loop_report(
        stats,
        group_id=group_id,
        recent_cycles=recent_cycles,
        calibration_snapshots=calibration_snapshots,
        recall_samples=recall_samples,
        session_samples=session_samples,
    )
