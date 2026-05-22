"""Shared brain-loop evaluation report assembly."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from engram.consolidation.audit_reader import ConsolidationAuditReader
from engram.evaluation.brain_loop_report import (
    build_brain_loop_report,
    has_memory_operation_metrics,
    has_recall_runtime_metrics,
    merge_memory_operation_metrics,
    merge_recall_runtime_metrics,
)
from engram.evaluation.store import (
    StoredMemoryOperationMetricsSnapshot,
    StoredRecallRuntimeMetricsSnapshot,
)

REPORT_CONTEXT_TIMEOUT_SECONDS = 2.0
REPORT_GRAPH_STATE_TIMEOUT_SECONDS = 2.0


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
    degradations: list[dict[str, Any]] = []
    recent_cycles, calibration_snapshots = await _load_consolidation_context_bounded(
        load_consolidation_evaluation_inputs(
            consolidation_store,
            group_id=group_id,
            cycle_limit=cycle_limit,
        ),
        timeout_seconds=REPORT_CONTEXT_TIMEOUT_SECONDS,
        source="mcp_report",
        degradations=degradations,
    )
    return await build_brain_loop_evaluation_surface(
        manager,
        evaluation_store,
        group_id=group_id,
        recent_cycles=recent_cycles,
        calibration_snapshots=calibration_snapshots,
        sample_limit=max(1, sample_limit),
        snapshot_source="mcp_report",
        degradations=degradations,
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
    degradations: list[dict[str, Any]] = []
    recent_cycles, calibration_snapshots = await _load_consolidation_context_bounded(
        load_engine_evaluation_context(
            consolidation_engine,
            group_id=group_id,
            cycle_limit=cycle_limit,
        ),
        timeout_seconds=REPORT_CONTEXT_TIMEOUT_SECONDS,
        source="rest_report",
        degradations=degradations,
    )
    return await build_brain_loop_evaluation_surface(
        manager,
        evaluation_store,
        group_id=group_id,
        recent_cycles=recent_cycles,
        calibration_snapshots=calibration_snapshots,
        sample_limit=max(1, sample_limit),
        snapshot_source="rest_report",
        degradations=degradations,
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
    degradations: Sequence[dict[str, Any]] | None = None,
    graph_state_timeout_seconds: float = REPORT_GRAPH_STATE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Build the REST/MCP brain-loop report from live graph and saved labels."""
    report_degradations = [dict(item) for item in degradations or []]
    graph_state = await _load_graph_state_bounded(
        manager,
        group_id=group_id,
        timeout_seconds=graph_state_timeout_seconds,
        degradations=report_degradations,
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

    memory_operation_metrics = (
        stats.get("memory_operation_metrics") or stats.get("memoryOperationMetrics") or {}
    )
    if has_memory_operation_metrics(memory_operation_metrics):
        await evaluation_store.save_memory_operation_metrics_snapshot(
            StoredMemoryOperationMetricsSnapshot(
                group_id=group_id,
                metrics=dict(memory_operation_metrics),
                source=snapshot_source,
            )
        )
    else:
        stats = merge_memory_operation_metrics(
            stats,
            await evaluation_store.get_latest_memory_operation_metrics_snapshot(group_id),
        )

    recall_samples = await evaluation_store.get_recall_samples(
        group_id,
        limit=sample_limit,
    )
    session_samples = await evaluation_store.get_session_samples(
        group_id,
        limit=sample_limit,
    )
    report = build_brain_loop_report(
        stats,
        group_id=group_id,
        recent_cycles=recent_cycles,
        calibration_snapshots=calibration_snapshots,
        recall_samples=recall_samples,
        session_samples=session_samples,
    )
    if report_degradations:
        report["degraded"] = True
        report["degradations"] = report_degradations
    return report


async def _load_consolidation_context_bounded(
    awaitable,
    *,
    timeout_seconds: float,
    source: str,
    degradations: list[dict[str, Any]],
) -> tuple[list[Any], list[Any]]:
    try:
        return await asyncio.wait_for(awaitable, timeout=max(0.01, timeout_seconds))
    except TimeoutError:
        degradations.append(
            {
                "surface": source,
                "stage": "consolidate",
                "status": "degraded",
                "skip_reason": "evaluation_context_timeout",
                "timeout_ms": round(max(0.01, timeout_seconds) * 1000),
            }
        )
        return [], []


async def _load_graph_state_bounded(
    manager: Any,
    *,
    group_id: str,
    timeout_seconds: float,
    degradations: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(
            manager.get_graph_state(
                group_id=group_id,
                top_n=10,
                include_edges=False,
            ),
            timeout=max(0.01, timeout_seconds),
        )
    except TimeoutError:
        degradations.append(
            {
                "stage": "graph_state",
                "status": "degraded",
                "skip_reason": "graph_state_timeout",
                "timeout_ms": round(max(0.01, timeout_seconds) * 1000),
            }
        )
        return {"stats": _fallback_runtime_stats(manager, group_id)}


def _fallback_runtime_stats(manager: Any, group_id: str) -> dict[str, Any]:
    return {
        "recall_metrics": _call_optional(manager, "get_recall_metrics", group_id) or {},
        "memory_operation_metrics": _call_optional(
            manager,
            "get_memory_operation_metrics",
            group_id,
        )
        or {},
        "packet_cache": _call_optional(manager, "get_memory_packet_cache_summary", group_id)
        or {},
    }


def _call_optional(manager: Any, name: str, group_id: str) -> Any:
    method = getattr(manager, name, None)
    if not callable(method):
        return None
    try:
        return method(group_id)
    except Exception:
        return None
