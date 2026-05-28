"""Shared brain-loop evaluation report assembly."""

from __future__ import annotations

import asyncio
import inspect
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
REPORT_CONTEXT_CACHE_TTL_SECONDS = 300.0
REPORT_CONTEXT_TASK_MAX_SECONDS = 30.0
_CONSOLIDATION_CONTEXT_CACHE: dict[
    tuple[str, str],
    tuple[float, tuple[list[Any], list[Any]]],
] = {}
_CONSOLIDATION_CONTEXT_TASKS: dict[
    tuple[str, str],
    asyncio.Task[tuple[list[Any], list[Any]]],
] = {}
_CONSOLIDATION_CONTEXT_TASK_STARTED_AT: dict[tuple[str, str], float] = {}


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
        lambda: load_consolidation_evaluation_inputs(
            consolidation_store,
            group_id=group_id,
            cycle_limit=cycle_limit,
        ),
        timeout_seconds=REPORT_CONTEXT_TIMEOUT_SECONDS,
        source="mcp_report",
        group_id=group_id,
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
    live_memory_operation_cost: bool = False,
) -> dict[str, Any]:
    """Build the REST brain-loop report from active runtime services."""
    degradations: list[dict[str, Any]] = []
    if live_memory_operation_cost:
        recent_cycles, calibration_snapshots = [], []
    else:
        recent_cycles, calibration_snapshots = await _load_consolidation_context_bounded(
            lambda: load_engine_evaluation_context(
                consolidation_engine,
                group_id=group_id,
                cycle_limit=cycle_limit,
            ),
            timeout_seconds=REPORT_CONTEXT_TIMEOUT_SECONDS,
            source="rest_report",
            group_id=group_id,
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
        skip_graph_state=live_memory_operation_cost,
        merge_saved_memory_operation_metrics=not live_memory_operation_cost,
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
    skip_graph_state: bool = False,
    merge_saved_memory_operation_metrics: bool = True,
) -> dict[str, Any]:
    """Build the REST/MCP brain-loop report from live graph and saved labels."""
    report_degradations = [dict(item) for item in degradations or []]
    if skip_graph_state:
        graph_state = _load_graph_state_runtime_only(
            manager,
            group_id=group_id,
            degradations=report_degradations,
            reason="live_cost_runtime_only",
        )
    else:
        graph_state = await _load_graph_state_bounded(
            manager,
            group_id=group_id,
            timeout_seconds=graph_state_timeout_seconds,
            degradations=report_degradations,
        )
    stats = graph_state.get("stats") or {}
    if (
        not skip_graph_state
        and not _has_graph_count_stats(stats)
        and not _has_graph_state_degradation(report_degradations)
    ):
        report_degradations.append(
            {
                "stage": "graph_state",
                "status": "degraded",
                "skip_reason": "graph_state_unavailable",
            }
        )
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
    elif merge_saved_memory_operation_metrics:
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
    awaitable_factory,
    *,
    timeout_seconds: float,
    source: str,
    group_id: str,
    degradations: list[dict[str, Any]],
) -> tuple[list[Any], list[Any]]:
    cache_key = (source, group_id)
    cached = _cached_consolidation_context(cache_key)
    if cached is not None:
        return cached

    task = _warm_consolidation_context(cache_key, awaitable_factory)
    try:
        return await asyncio.wait_for(
            asyncio.shield(task),
            timeout=max(0.01, timeout_seconds),
        )
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
        cached = _cached_consolidation_context(cache_key)
        if cached is not None:
            return cached
        return [], []


def _cached_consolidation_context(
    cache_key: tuple[str, str],
) -> tuple[list[Any], list[Any]] | None:
    cached = _CONSOLIDATION_CONTEXT_CACHE.get(cache_key)
    if cached is None:
        return None
    captured_at, value = cached
    if (asyncio.get_running_loop().time() - captured_at) > REPORT_CONTEXT_CACHE_TTL_SECONDS:
        return None
    cycles, snapshots = value
    return list(cycles), list(snapshots)


def _warm_consolidation_context(
    cache_key: tuple[str, str],
    awaitable_factory,
) -> asyncio.Task[tuple[list[Any], list[Any]]]:
    existing = _CONSOLIDATION_CONTEXT_TASKS.get(cache_key)
    if existing is not None and not existing.done():
        started_at = _CONSOLIDATION_CONTEXT_TASK_STARTED_AT.get(cache_key, 0.0)
        if (asyncio.get_running_loop().time() - started_at) <= REPORT_CONTEXT_TASK_MAX_SECONDS:
            return existing
        existing.cancel()
        _CONSOLIDATION_CONTEXT_TASKS.pop(cache_key, None)
        _CONSOLIDATION_CONTEXT_TASK_STARTED_AT.pop(cache_key, None)

    async def _load_and_cache() -> tuple[list[Any], list[Any]]:
        cycles, snapshots = await awaitable_factory()
        result = (list(cycles), list(snapshots))
        _CONSOLIDATION_CONTEXT_CACHE[cache_key] = (
            asyncio.get_running_loop().time(),
            result,
        )
        return result

    task = asyncio.create_task(_load_and_cache())
    _CONSOLIDATION_CONTEXT_TASKS[cache_key] = task
    _CONSOLIDATION_CONTEXT_TASK_STARTED_AT[cache_key] = asyncio.get_running_loop().time()
    task.add_done_callback(lambda done: _finish_consolidation_context_task(cache_key, done))
    return task


def _finish_consolidation_context_task(
    cache_key: tuple[str, str],
    task: asyncio.Task,
) -> None:
    if _CONSOLIDATION_CONTEXT_TASKS.get(cache_key) is task:
        _CONSOLIDATION_CONTEXT_TASKS.pop(cache_key, None)
        _CONSOLIDATION_CONTEXT_TASK_STARTED_AT.pop(cache_key, None)
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:
        return


async def _load_graph_state_bounded(
    manager: Any,
    *,
    group_id: str,
    timeout_seconds: float,
    degradations: list[dict[str, Any]],
) -> dict[str, Any]:
    cached_stats = _call_optional(manager, "get_cached_graph_stats", group_id)
    if isinstance(cached_stats, dict) and cached_stats:
        return {"stats": _merge_runtime_stats(cached_stats, manager, group_id)}

    graph_state_task = _evaluation_graph_state_task(manager, group_id=group_id)
    graph_state_task.add_done_callback(_consume_background_task_exception)
    try:
        return await asyncio.wait_for(
            asyncio.shield(graph_state_task),
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
        cached_stats = _call_optional(manager, "get_cached_graph_stats", group_id)
        if isinstance(cached_stats, dict) and cached_stats:
            return {"stats": _merge_runtime_stats(cached_stats, manager, group_id)}
        return {"stats": _fallback_runtime_stats(manager, group_id)}


def _evaluation_graph_state_task(manager: Any, *, group_id: str) -> asyncio.Task:
    warm_reader = _declared_optional_callable(manager, "warm_graph_stats")
    if warm_reader is None:
        return asyncio.create_task(_read_evaluation_graph_state(manager, group_id=group_id))
    try:
        stats_task = warm_reader(group_id)
    except TypeError:
        stats_task = warm_reader(group_id=group_id)

    async def _wrap_stats_task() -> dict[str, Any]:
        stats = await stats_task if inspect.isawaitable(stats_task) else stats_task
        return {"stats": dict(stats or {})}

    return asyncio.create_task(_wrap_stats_task())


async def _read_evaluation_graph_state(manager: Any, *, group_id: str) -> dict[str, Any]:
    stats_reader = _declared_optional_callable(manager, "get_graph_stats")
    if stats_reader is not None:
        try:
            stats = stats_reader(group_id=group_id)
        except TypeError:
            stats = stats_reader(group_id)
        if inspect.isawaitable(stats):
            stats = await stats
        return {"stats": dict(stats or {})}

    return await manager.get_graph_state(
        group_id=group_id,
        top_n=10,
        include_edges=False,
    )


def _load_graph_state_runtime_only(
    manager: Any,
    *,
    group_id: str,
    degradations: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    cached_stats = _call_optional(manager, "get_cached_graph_stats", group_id)
    if isinstance(cached_stats, dict) and cached_stats:
        return {"stats": _merge_runtime_stats(cached_stats, manager, group_id)}
    degradations.append(
        {
            "stage": "graph_state",
            "status": "skipped",
            "skip_reason": reason,
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


def _has_graph_count_stats(stats: dict[str, Any]) -> bool:
    return any(key in stats for key in ("episodes", "entities", "relationships"))


def _has_graph_state_degradation(degradations: Sequence[dict[str, Any]]) -> bool:
    return any(item.get("stage") == "graph_state" for item in degradations)


def _merge_runtime_stats(stats: dict[str, Any], manager: Any, group_id: str) -> dict[str, Any]:
    merged = dict(stats)
    runtime_stats = _fallback_runtime_stats(manager, group_id)
    for key, value in runtime_stats.items():
        if value:
            merged[key] = value
    return merged


def _consume_background_task_exception(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:
        return


def _call_optional(manager: Any, name: str, group_id: str) -> Any:
    method = _declared_optional_callable(manager, name)
    if method is None:
        return None
    try:
        return method(group_id)
    except Exception:
        return None


def _declared_optional_callable(manager: Any, name: str) -> Any | None:
    if name in getattr(manager, "__dict__", {}):
        method = getattr(manager, name, None)
    elif hasattr(type(manager), name):
        method = getattr(manager, name, None)
    else:
        return None
    return method if callable(method) else None
