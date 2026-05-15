"""Shared brain-loop evaluation report assembly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from engram.evaluation.brain_loop_report import (
    build_brain_loop_report,
    has_recall_runtime_metrics,
    merge_recall_runtime_metrics,
)
from engram.evaluation.store import StoredRecallRuntimeMetricsSnapshot


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
