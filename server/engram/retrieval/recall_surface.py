"""Explicit recall response builders shared by REST and MCP."""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from engram.extraction.promotion import (
    is_relationship_triple_entity as _is_relationship_triple_entity,
)
from engram.models.recall import MemoryPacket
from engram.retrieval.budgets import (
    RecallBudget,
    budget_profile_for_source,
    recall_budget_for_profile,
    surface_for_source,
)
from engram.retrieval.context_builder import (
    _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    SESSION_RECENT_PACKET_SCOPE,
    _run_project_file_executor,
    cache_project_file_fallback_packet_payloads,
    project_file_fallback_packet_payloads,
)
from engram.retrieval.control import resolve_manager_recall_need_thresholds
from engram.retrieval.feedback import note_surfaced_texts_from_response
from engram.retrieval.memory_operations import (
    MemoryOperationSample,
    record_manager_memory_operation,
)
from engram.retrieval.need import analyze_memory_need
from engram.retrieval.packets import assemble_memory_packets
from engram.retrieval.presenter import (
    present_api_recall_response,
    present_mcp_recall_items,
    present_mcp_recall_response,
    recall_contract_item,
)

PacketSerializer = Callable[[MemoryPacket], dict[str, Any]]
ResolveNameFn = Callable[[str], Awaitable[str]]
AccessCountFn = Callable[[str], Awaitable[int]]
FAST_RECALL_FALLBACK_TIMEOUT_SECONDS = 0.15
FAST_RECALL_FALLBACK_MIN_MATCHES = 2
PROJECT_FILE_RECALL_FALLBACK_MAX_CANDIDATES = 40
PROJECT_FILE_RECALL_FALLBACK_READ_LIMIT = 16
PROJECT_FILE_RECALL_FALLBACK_SCAN_CHARS = 16_000
PROJECT_FILE_RECALL_FALLBACK_WAIT_SECONDS = 0.1
PROJECT_FILE_RECALL_EMPTY_SUCCESS_WAIT_SECONDS = 1.25
EXPLICIT_RECALL_PACKET_CACHE_SCOPE = "explicit_recall"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 4)


def _camel_stage_name(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _default_project_path_for_recall() -> str | None:
    configured = os.environ.get("ENGRAM_RECALL_PROJECT_PATH") or os.environ.get(
        "ENGRAM_PROJECT_PATH"
    )
    for candidate in (configured, os.getcwd()):
        if not candidate:
            continue
        try:
            path = Path(candidate).expanduser().resolve()
        except OSError:
            continue
        if _looks_like_project_directory(path):
            return str(path)
    return None


def _looks_like_project_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        (path / marker).exists()
        for marker in (".git", "pyproject.toml", "package.json", "README.md", "docs")
    )


async def build_api_recall_surface(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    project_path: str | None = None,
    operation_source: str = "api_recall",
) -> dict[str, Any]:
    """Build the REST explicit-recall payload."""
    packet_policy = manager.get_explicit_recall_packet_policy()
    cfg = manager.get_memory_need_config()
    pre_stage_timings: dict[str, float] = {}
    started = time.perf_counter()
    packet_started = time.perf_counter()
    packets = await cached_explicit_recall_packet_payloads(
        manager,
        group_id=group_id,
        query=query,
        max_packets=packet_policy.max_packets,
        cfg=cfg,
        operation_source=operation_source,
        enabled=packet_policy.enabled,
    )
    pre_stage_timings["packet_cache"] = _elapsed_ms(packet_started)
    context_packets: list[dict[str, Any]] = []
    if not packets:
        context_packets = await cached_context_recall_packet_payloads(
            manager,
            group_id=group_id,
            query=query,
            max_packets=packet_policy.max_packets,
            cfg=cfg,
            operation_source=operation_source,
            mode="context_packet_cache_preflight",
            project_path=project_path,
        )
    cache_packets = packets or context_packets
    if _packets_satisfy_explicit_query(cache_packets, query=query):
        recall_metadata = await _cache_satisfied_recall_metadata(
            manager,
            group_id=group_id,
            operation_source=operation_source,
            cfg=cfg,
            limit=limit,
            packet_count=len(cache_packets),
            duration_ms=_elapsed_ms(started),
            stage_timings_ms=pre_stage_timings,
        )
        response = present_api_recall_response(
            query=query,
            results=[],
            packets=cache_packets,
        )
        _attach_recall_budget_metadata(response, recall_metadata, camel_case=True)
        # The REST payload presents camelCase `items` the citation scan cannot
        # read, so the mask/cue register is fed the surface-neutral contract
        # shape here (mirrors the MCP surface; mask-only, byte-safe on output).
        note_surfaced_texts_from_response(
            group_id,
            {"results": [], "packets": list(cache_packets)},
            cfg,
        )
        return response
    project_file_fallback_path = (
        project_path or _default_project_path_for_recall() if not cache_packets else None
    )
    prefer_project_file_before_recent = bool(project_path)
    project_file_fallback_task = _start_project_file_recall_fallback_task(
        query=query,
        project_path=project_file_fallback_path,
        max_packets=packet_policy.max_packets,
    )
    results, recall_metadata = await _run_explicit_recall_with_budget(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
        cfg=cfg,
        operation_source=operation_source,
        project_path=project_path,
        context_packet_count=len(context_packets),
        project_file_fallback_task=project_file_fallback_task,
        project_file_fallback_path=project_file_fallback_path,
        project_file_max_packets=packet_policy.max_packets,
    )
    early_project_packets = _pop_early_project_file_packets(recall_metadata)
    if early_project_packets and not packets:
        packets = early_project_packets
        _mark_project_file_recall_fallback(recall_metadata, packet_count=len(packets))
        recall_metadata["duration_ms"] = _elapsed_ms(started)
        recall_metadata["budget_miss"] = _metadata_budget_exceeded(recall_metadata)
    recall_metadata.setdefault("stage_timings_ms", {}).update(pre_stage_timings)
    results = await _maybe_run_success_fast_fallback(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
        cfg=cfg,
        recall_metadata=recall_metadata,
        started=started,
        results=results,
        packets=packets,
        context_packets=context_packets,
        project_path=project_path,
    )
    if recall_metadata["status"] != "ok" and not packets:
        packets = context_packets
        fallback_project_path = (
            project_file_fallback_path
            if prefer_project_file_before_recent and not packets
            else None
        )
        used_project_file_fallback = False
        if fallback_project_path:
            project_file_started = time.perf_counter()
            packets, build_duration_ms = await _resolve_project_file_recall_fallback_task(
                project_file_fallback_task,
                manager,
                group_id=group_id,
                query=query,
                project_path=fallback_project_path,
                max_packets=packet_policy.max_packets,
                operation_source=operation_source,
            )
            stage_timings = recall_metadata.setdefault("stage_timings_ms", {})
            stage_timings["project_file_recall_fallback_wait"] = _elapsed_ms(project_file_started)
            if build_duration_ms is not None:
                stage_timings["project_file_recall_fallback"] = build_duration_ms
            used_project_file_fallback = bool(packets)
        if not packets:
            packets = await cached_context_recall_packet_payloads(
                manager,
                group_id=group_id,
                query=query,
                max_packets=packet_policy.max_packets,
                cfg=cfg,
                operation_source=operation_source,
                allow_recent_miss=True,
                project_path=project_path,
            )
        if not packets and not prefer_project_file_before_recent and project_file_fallback_path:
            project_file_started = time.perf_counter()
            packets, build_duration_ms = await _resolve_project_file_recall_fallback_task(
                project_file_fallback_task,
                manager,
                group_id=group_id,
                query=query,
                project_path=project_file_fallback_path,
                max_packets=packet_policy.max_packets,
                operation_source=operation_source,
            )
            stage_timings = recall_metadata.setdefault("stage_timings_ms", {})
            stage_timings["project_file_recall_fallback_wait"] = _elapsed_ms(project_file_started)
            if build_duration_ms is not None:
                stage_timings["project_file_recall_fallback"] = build_duration_ms
            used_project_file_fallback = bool(packets)
        if packets:
            if used_project_file_fallback:
                _mark_project_file_recall_fallback(
                    recall_metadata,
                    packet_count=len(packets),
                )
            else:
                _mark_context_packet_recall_fallback(
                    recall_metadata,
                    packet_count=len(packets),
                )
            recall_metadata["duration_ms"] = _elapsed_ms(started)
            recall_metadata["budget_miss"] = True
            recall_metadata["degraded"] = True
    if recall_metadata["status"] == "ok" and not results and not packets and context_packets:
        packets = context_packets
        _mark_context_packet_recall_fallback(
            recall_metadata,
            packet_count=len(context_packets),
        )
    if recall_metadata["status"] == "ok" and results and not packets:
        packet_started = time.perf_counter()
        packets = await assemble_explicit_recall_packet_payloads(
            manager,
            group_id=group_id,
            query=query,
            results=results,
            enabled=packet_policy.enabled,
            max_packets=packet_policy.max_packets,
            cfg=cfg,
            serializer=memory_packet_to_api_dict,
            operation_source=operation_source,
        )
        recall_metadata.setdefault("stage_timings_ms", {})["packet_assembly"] = _elapsed_ms(
            packet_started
        )
    if (
        recall_metadata["status"] == "ok"
        and not results
        and not packets
        and project_file_fallback_path
    ):
        project_file_started = time.perf_counter()
        packets, build_duration_ms = await _resolve_project_file_recall_fallback_task(
            project_file_fallback_task,
            manager,
            group_id=group_id,
            query=query,
            project_path=project_file_fallback_path,
            max_packets=packet_policy.max_packets,
            operation_source=operation_source,
            timeout_seconds=_project_file_empty_success_wait_seconds(
                recall_metadata,
                started,
            ),
        )
        stage_timings = recall_metadata.setdefault("stage_timings_ms", {})
        stage_timings["project_file_recall_fallback_wait"] = _elapsed_ms(project_file_started)
        if build_duration_ms is not None:
            stage_timings["project_file_recall_fallback"] = build_duration_ms
        if packets:
            _mark_project_file_recall_fallback(recall_metadata, packet_count=len(packets))
            recall_metadata["duration_ms"] = _elapsed_ms(started)
            recall_metadata["budget_miss"] = _metadata_budget_exceeded(recall_metadata)
    if not results and not packets and recall_metadata["status"] != "ok":
        packets = [_diagnostic_recall_packet(query=query, metadata=recall_metadata)]
    response = present_api_recall_response(query=query, results=results, packets=packets)
    _attach_recall_budget_metadata(response, recall_metadata, camel_case=True)
    note_surfaced_texts_from_response(
        group_id,
        {
            "results": [recall_contract_item(result) for result in results],
            "packets": list(packets),
        },
        cfg,
    )
    return response


async def build_mcp_recall_surface(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    cfg: Any,
    project_path: str | None = None,
    resolve_entity_name: ResolveNameFn | None = None,
    get_access_count: AccessCountFn | None = None,
) -> dict[str, Any]:
    """Build the MCP explicit-recall payload without transport-only metadata."""
    pre_stage_timings: dict[str, float] = {}
    packet_enabled = bool(getattr(cfg, "recall_packets_enabled", False))
    max_packets = int(getattr(cfg, "recall_packet_explicit_limit", 0) or 0)
    started = time.perf_counter()
    packet_started = time.perf_counter()
    packets = await cached_explicit_recall_packet_payloads(
        manager,
        group_id=group_id,
        query=query,
        max_packets=max_packets,
        cfg=cfg,
        operation_source="mcp_recall",
        enabled=packet_enabled,
    )
    pre_stage_timings["packet_cache"] = _elapsed_ms(packet_started)
    context_packets: list[dict[str, Any]] = []
    if not packets:
        context_packets = await cached_context_recall_packet_payloads(
            manager,
            group_id=group_id,
            query=query,
            max_packets=max_packets,
            cfg=cfg,
            operation_source="mcp_recall",
            mode="context_packet_cache_preflight",
            project_path=project_path,
        )
    cache_packets = packets or context_packets
    if _packets_satisfy_explicit_query(cache_packets, query=query):
        recall_metadata = await _cache_satisfied_recall_metadata(
            manager,
            group_id=group_id,
            operation_source="mcp_recall",
            cfg=cfg,
            limit=limit,
            packet_count=len(cache_packets),
            duration_ms=_elapsed_ms(started),
            stage_timings_ms=pre_stage_timings,
        )
        response = present_mcp_recall_response(
            query=query,
            results=[],
            packets=cache_packets,
        )
        _attach_recall_budget_metadata(response, recall_metadata, camel_case=False)
        await attach_mcp_explicit_recall_enrichment(
            manager,
            response,
            group_id=group_id,
        )
        note_surfaced_texts_from_response(group_id, response, cfg)
        return response
    project_file_fallback_path = (
        project_path or _default_project_path_for_recall() if not cache_packets else None
    )
    prefer_project_file_before_recent = bool(project_path)
    project_file_fallback_task = _start_project_file_recall_fallback_task(
        query=query,
        project_path=project_file_fallback_path,
        max_packets=max_packets,
    )
    results, recall_metadata = await _run_explicit_recall_with_budget(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
        cfg=cfg,
        operation_source="mcp_recall",
        project_path=project_path,
        context_packet_count=len(context_packets),
        project_file_fallback_task=project_file_fallback_task,
        project_file_fallback_path=project_file_fallback_path,
        project_file_max_packets=max_packets,
    )
    early_project_packets = _pop_early_project_file_packets(recall_metadata)
    if early_project_packets and not packets:
        packets = early_project_packets
        _mark_project_file_recall_fallback(recall_metadata, packet_count=len(packets))
        recall_metadata["duration_ms"] = _elapsed_ms(started)
        recall_metadata["budget_miss"] = _metadata_budget_exceeded(recall_metadata)
    recall_metadata.setdefault("stage_timings_ms", {}).update(pre_stage_timings)
    results = await _maybe_run_success_fast_fallback(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
        cfg=cfg,
        recall_metadata=recall_metadata,
        started=started,
        results=results,
        packets=packets,
        context_packets=context_packets,
        project_path=project_path,
    )
    if recall_metadata["status"] != "ok" and not packets:
        packets = context_packets
        fallback_project_path = (
            project_file_fallback_path
            if prefer_project_file_before_recent and not packets
            else None
        )
        used_project_file_fallback = False
        if fallback_project_path:
            project_file_started = time.perf_counter()
            packets, build_duration_ms = await _resolve_project_file_recall_fallback_task(
                project_file_fallback_task,
                manager,
                group_id=group_id,
                query=query,
                project_path=fallback_project_path,
                max_packets=max_packets,
                operation_source="mcp_recall",
            )
            stage_timings = recall_metadata.setdefault("stage_timings_ms", {})
            stage_timings["project_file_recall_fallback_wait"] = _elapsed_ms(project_file_started)
            if build_duration_ms is not None:
                stage_timings["project_file_recall_fallback"] = build_duration_ms
            used_project_file_fallback = bool(packets)
        if not packets:
            packets = await cached_context_recall_packet_payloads(
                manager,
                group_id=group_id,
                query=query,
                max_packets=max_packets,
                cfg=cfg,
                operation_source="mcp_recall",
                allow_recent_miss=True,
                project_path=project_path,
            )
        if not packets and not prefer_project_file_before_recent and project_file_fallback_path:
            project_file_started = time.perf_counter()
            packets, build_duration_ms = await _resolve_project_file_recall_fallback_task(
                project_file_fallback_task,
                manager,
                group_id=group_id,
                query=query,
                project_path=project_file_fallback_path,
                max_packets=max_packets,
                operation_source="mcp_recall",
            )
            stage_timings = recall_metadata.setdefault("stage_timings_ms", {})
            stage_timings["project_file_recall_fallback_wait"] = _elapsed_ms(project_file_started)
            if build_duration_ms is not None:
                stage_timings["project_file_recall_fallback"] = build_duration_ms
            used_project_file_fallback = bool(packets)
        if packets:
            if used_project_file_fallback:
                _mark_project_file_recall_fallback(
                    recall_metadata,
                    packet_count=len(packets),
                )
            else:
                _mark_context_packet_recall_fallback(
                    recall_metadata,
                    packet_count=len(packets),
                )
            recall_metadata["duration_ms"] = _elapsed_ms(started)
            recall_metadata["budget_miss"] = True
            recall_metadata["degraded"] = True
    if recall_metadata["status"] == "ok" and not results and not packets and context_packets:
        packets = context_packets
        _mark_context_packet_recall_fallback(
            recall_metadata,
            packet_count=len(context_packets),
        )
    if resolve_entity_name is None:
        resolve_entity_name = _mcp_recall_entity_name_resolver(manager, group_id)
    if get_access_count is None:
        get_access_count = _mcp_recall_access_count_resolver(manager)

    present_started = time.perf_counter()
    formatted = await present_mcp_recall_items(
        results,
        resolve_entity_name=resolve_entity_name,
        get_access_count=get_access_count,
    )
    recall_metadata.setdefault("stage_timings_ms", {})["recall_present"] = _elapsed_ms(
        present_started,
    )
    if recall_metadata["status"] == "ok" and results and not packets:
        packet_started = time.perf_counter()
        packets = await assemble_explicit_recall_packet_payloads(
            manager,
            group_id=group_id,
            query=query,
            results=results,
            enabled=packet_enabled,
            max_packets=max_packets,
            cfg=cfg,
            serializer=lambda packet: packet.to_dict(),
            operation_source="mcp_recall",
        )
        recall_metadata.setdefault("stage_timings_ms", {})["packet_assembly"] = _elapsed_ms(
            packet_started
        )
    if (
        recall_metadata["status"] == "ok"
        and not results
        and not packets
        and project_file_fallback_path
    ):
        project_file_started = time.perf_counter()
        packets, build_duration_ms = await _resolve_project_file_recall_fallback_task(
            project_file_fallback_task,
            manager,
            group_id=group_id,
            query=query,
            project_path=project_file_fallback_path,
            max_packets=max_packets,
            operation_source="mcp_recall",
            timeout_seconds=_project_file_empty_success_wait_seconds(
                recall_metadata,
                started,
            ),
        )
        stage_timings = recall_metadata.setdefault("stage_timings_ms", {})
        stage_timings["project_file_recall_fallback_wait"] = _elapsed_ms(project_file_started)
        if build_duration_ms is not None:
            stage_timings["project_file_recall_fallback"] = build_duration_ms
        if packets:
            _mark_project_file_recall_fallback(recall_metadata, packet_count=len(packets))
            recall_metadata["duration_ms"] = _elapsed_ms(started)
            recall_metadata["budget_miss"] = _metadata_budget_exceeded(recall_metadata)
    if not formatted and not packets and recall_metadata["status"] != "ok":
        packets = [_diagnostic_recall_packet(query=query, metadata=recall_metadata)]
    response = present_mcp_recall_response(
        query=query,
        results=formatted,
        packets=packets,
    )
    _attach_recall_budget_metadata(response, recall_metadata, camel_case=False)
    await attach_mcp_explicit_recall_enrichment(
        manager,
        response,
        group_id=group_id,
    )
    note_surfaced_texts_from_response(group_id, response, cfg)
    return response


async def _run_explicit_recall_with_budget(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    cfg: Any,
    operation_source: str,
    project_path: str | None = None,
    context_packet_count: int = 0,
    project_file_fallback_task: asyncio.Task[tuple[list[dict[str, Any]], float]] | None = None,
    project_file_fallback_path: str | None = None,
    project_file_max_packets: int = 0,
) -> tuple[list[dict], dict[str, Any]]:
    """Run the live recall stage under the shared explicit recall budget."""
    budget = recall_budget_for_profile(
        cfg,
        budget_profile_for_source(operation_source),
        surface=surface_for_source(operation_source),
        mode=operation_source,
        max_results=limit,
    )
    timeout_seconds = budget.stage_timeout_seconds(budget.max_search_ms)
    started = time.perf_counter()
    stage_timings: dict[str, float] = {}
    fallback_results: list[dict] = []
    fallback_status = "not_run"
    if timeout_seconds <= 0:
        duration_ms = round((time.perf_counter() - started) * 1000, 4)
        await _record_recall_budget_event(
            manager,
            group_id=group_id,
            operation_source=operation_source,
            budget=budget,
            status="skipped",
            skip_reason="skipped_budget",
            duration_ms=duration_ms,
            timeout=False,
            budget_miss=True,
            result_count=len(fallback_results),
        )
        return list(fallback_results), _recall_budget_metadata(
            budget,
            status="skipped",
            duration_ms=duration_ms,
            skip_reason="skipped_budget",
            timeout=False,
            budget_miss=True,
            stage_timings_ms=stage_timings,
            fallback_status=fallback_status,
            fallback_result_count=0,
        )

    # Product polish: Decision/identity must not wait on hybrid first.
    # Cheap durable exact-name / type rescue before preflight or BM25/HNSW thrash.
    #
    # INTENT-GATED short-circuit: returning rescue stubs for ANY query that
    # merely names a known entity ("Engram", the user) silently converted
    # explicit recall into exact-name lookup for most real queries (no
    # episodes, no relationships). The fast path only replaces the deep
    # pipeline when the query is durable-lookup shaped or the top hit is a
    # high-signal durable fact with strong overlap; otherwise the deep
    # pipeline runs as before (later rescue stages still cover miss/timeout).
    durable_first_started = time.perf_counter()
    durable_first = await _durable_entity_name_rescue(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
        timeout_seconds=min(0.75, max(0.2, timeout_seconds * 0.4)),
    )
    stage_timings["durable_entity_first"] = _elapsed_ms(durable_first_started)
    if durable_first and not _durable_first_short_circuit_allowed(query, durable_first):
        stage_timings["durable_entity_first_gated"] = 1.0
        durable_first = []
    if durable_first:
        duration_ms = round((time.perf_counter() - started) * 1000, 4)
        await record_manager_memory_operation(
            manager,
            group_id,
            MemoryOperationSample(
                operation="recall",
                source=operation_source,
                mode=operation_source,
                status="ok",
                duration_ms=duration_ms,
                timeout=False,
                degraded=False,
                budget_miss=budget.exceeded(duration_ms),
                budget_ms=budget.budget_ms,
                budget_tokens=budget.budget_tokens,
                result_count=len(durable_first),
                packet_count=0,
            ),
        )
        return list(durable_first), _recall_budget_metadata(
            budget,
            status="ok",
            duration_ms=duration_ms,
            budget_miss=budget.exceeded(duration_ms),
            stage_timings_ms=stage_timings,
            fallback_status="durable_entity_first",
            fallback_result_count=len(durable_first),
        )

    if _fast_recall_preflight_enabled(cfg):
        fallback_started = time.perf_counter()
        preflight_timeout_seconds = _fast_recall_preflight_timeout_seconds(cfg)
        if context_packet_count > 0:
            preflight_timeout_seconds = min(
                preflight_timeout_seconds,
                _fast_recall_fallback_timeout_seconds(cfg),
            )
        fallback_results, fallback_status = await _run_fast_recall_fallback(
            manager,
            group_id=group_id,
            query=query,
            limit=limit,
            project_path=project_path,
            timeout_seconds=preflight_timeout_seconds,
        )
        stage_timings["recall_fast_preflight"] = _elapsed_ms(fallback_started)
        if fallback_results:
            duration_ms = round((time.perf_counter() - started) * 1000, 4)
            await record_manager_memory_operation(
                manager,
                group_id,
                MemoryOperationSample(
                    operation="recall",
                    source=operation_source,
                    mode=operation_source,
                    status="ok",
                    duration_ms=duration_ms,
                    timeout=False,
                    degraded=False,
                    budget_miss=budget.exceeded(duration_ms),
                    budget_ms=budget.budget_ms,
                    budget_tokens=budget.budget_tokens,
                    result_count=len(fallback_results),
                    packet_count=0,
                ),
            )
            return list(fallback_results), _recall_budget_metadata(
                budget,
                status="ok",
                duration_ms=duration_ms,
                budget_miss=budget.exceeded(duration_ms),
                stage_timings_ms=stage_timings,
                fallback_status="fast_preflight_hit",
                fallback_result_count=len(fallback_results),
            )
        # Do NOT abort on context-packet fallback when preflight misses/timeouts.
        # Session-recent observe packets are not a substitute for durable graph
        # Decisions; continue into durable-entity rescue + deep recall.
        if (
            context_packet_count > 0
            and not fallback_results
            and fallback_status in {"miss", "filtered", "timeout"}
        ):
            stage_timings["context_packet_soft_hold"] = float(context_packet_count)
        if (
            fallback_status == "timeout"
            and project_file_fallback_task is not None
            and project_file_fallback_path
            and project_file_max_packets > 0
        ):
            # Soft-hold project-file packets only. Do NOT abort deep recall —
            # durable Decision/Preference facts live in the graph, and aborting
            # here is why committed high-signal remembers never surface.
            project_file_started = time.perf_counter()
            packets, build_duration_ms = await _resolve_project_file_recall_fallback_task(
                project_file_fallback_task,
                manager,
                group_id=group_id,
                query=query,
                project_path=project_file_fallback_path,
                max_packets=project_file_max_packets,
                operation_source=operation_source,
                timeout_seconds=PROJECT_FILE_RECALL_FALLBACK_WAIT_SECONDS,
            )
            stage_timings["project_file_recall_fallback_wait"] = _elapsed_ms(project_file_started)
            if build_duration_ms is not None:
                stage_timings["project_file_recall_fallback"] = build_duration_ms
            if packets:
                # Stash for use only if deep recall also misses.
                stage_timings["project_file_soft_hold"] = float(len(packets))
                # Continue into manager.recall() below with remaining budget.

        # Cheap durable-entity rescue after preflight miss/timeout: name lookup
        # for high-signal types without waiting on the full activation pipeline.
        if not fallback_results and fallback_status in {"miss", "filtered", "timeout"}:
            rescue_started = time.perf_counter()
            rescue = await _durable_entity_name_rescue(
                manager,
                group_id=group_id,
                query=query,
                limit=limit,
            )
            stage_timings["durable_entity_rescue"] = _elapsed_ms(rescue_started)
            if rescue:
                duration_ms = round((time.perf_counter() - started) * 1000, 4)
                await record_manager_memory_operation(
                    manager,
                    group_id,
                    MemoryOperationSample(
                        operation="recall",
                        source=operation_source,
                        mode=operation_source,
                        status="ok",
                        duration_ms=duration_ms,
                        timeout=False,
                        degraded=False,
                        budget_miss=budget.exceeded(duration_ms),
                        budget_ms=budget.budget_ms,
                        budget_tokens=budget.budget_tokens,
                        result_count=len(rescue),
                        packet_count=0,
                    ),
                )
                return list(rescue), _recall_budget_metadata(
                    budget,
                    status="ok",
                    duration_ms=duration_ms,
                    budget_miss=budget.exceeded(duration_ms),
                    stage_timings_ms=stage_timings,
                    fallback_status="durable_entity_rescue",
                    fallback_result_count=len(rescue),
                )

    try:
        # Honor BOTH search stage budget and overall wall budget. Previously only
        # max_search_ms was applied here, so a tight explicit wall (e.g. 100ms)
        # could still wait the full 1.5s search budget and never degrade.
        timeout_seconds = budget.stage_timeout_seconds(budget.max_search_ms)
        recall_started = time.perf_counter()
        if timeout_seconds <= 0:
            raise asyncio.TimeoutError()
        results = await asyncio.wait_for(
            manager.recall(
                query=_recall_query_with_project_context(query, project_path),
                group_id=group_id,
                limit=limit,
                interaction_type="surfaced",
                interaction_source=operation_source,
            ),
            timeout=timeout_seconds,
        )
        stage_timings["recall_search"] = _elapsed_ms(recall_started)
        stage_timings.update(_manager_recall_stage_timings(manager))
    except asyncio.TimeoutError:
        stage_timings["recall_search"] = _elapsed_ms(recall_started)
        stage_timings.update(_manager_recall_stage_timings(manager))
        # Salvage: the primary search found and materialized candidates before a
        # downstream stage exceeded the search budget. Return those candidates
        # (degraded, best-effort) instead of destroying them for the durable /
        # project-file rescue cascade. Rescue fires only on a genuine empty.
        if _recall_return_partial_on_timeout_enabled(cfg):
            partial_results = _manager_recall_partial_results(manager)
            if partial_results:
                partial_results = _prefer_project_context_results(
                    partial_results, project_path=project_path
                )
                stage_timings["recall_partial_on_timeout"] = float(len(partial_results))
                duration_ms = round((time.perf_counter() - started) * 1000, 4)
                await _record_recall_budget_event(
                    manager,
                    group_id=group_id,
                    operation_source=operation_source,
                    budget=budget,
                    status="degraded",
                    skip_reason=None,
                    duration_ms=duration_ms,
                    timeout=True,
                    budget_miss=True,
                    result_count=len(partial_results),
                )
                return list(partial_results), _recall_budget_metadata(
                    budget,
                    status="degraded",
                    duration_ms=duration_ms,
                    timeout=True,
                    budget_miss=True,
                    stage_timings_ms=stage_timings,
                    fallback_status="partial_on_timeout",
                    fallback_result_count=len(partial_results),
                )
        if fallback_status == "not_run":
            fallback_started = time.perf_counter()
            fallback_results, fallback_status = await _run_fast_recall_fallback(
                manager,
                group_id=group_id,
                query=query,
                limit=limit,
                project_path=project_path,
                timeout_seconds=_fast_recall_fallback_timeout_seconds(cfg),
            )
            stage_timings["recall_fallback"] = _elapsed_ms(fallback_started)
        if not fallback_results:
            rescue_started = time.perf_counter()
            rescue = await _durable_entity_name_rescue(
                manager,
                group_id=group_id,
                query=query,
                limit=limit,
                timeout_seconds=1.25,
            )
            stage_timings["durable_entity_rescue_after_timeout"] = _elapsed_ms(rescue_started)
            if rescue:
                duration_ms = round((time.perf_counter() - started) * 1000, 4)
                await _record_recall_budget_event(
                    manager,
                    group_id=group_id,
                    operation_source=operation_source,
                    budget=budget,
                    status="ok",
                    skip_reason=None,
                    duration_ms=duration_ms,
                    timeout=False,
                    budget_miss=budget.exceeded(duration_ms),
                    result_count=len(rescue),
                )
                return list(rescue), _recall_budget_metadata(
                    budget,
                    status="ok",
                    duration_ms=duration_ms,
                    budget_miss=budget.exceeded(duration_ms),
                    stage_timings_ms=stage_timings,
                    fallback_status="durable_entity_rescue_after_timeout",
                    fallback_result_count=len(rescue),
                )
        duration_ms = round((time.perf_counter() - started) * 1000, 4)
        await _record_recall_budget_event(
            manager,
            group_id=group_id,
            operation_source=operation_source,
            budget=budget,
            status="degraded",
            skip_reason="recall_timeout",
            duration_ms=duration_ms,
            timeout=True,
            budget_miss=True,
            result_count=len(fallback_results),
        )
        return list(fallback_results), _recall_budget_metadata(
            budget,
            status="degraded",
            duration_ms=duration_ms,
            skip_reason="recall_timeout",
            timeout=True,
            budget_miss=True,
            stage_timings_ms=stage_timings,
            fallback_status=fallback_status,
            fallback_result_count=len(fallback_results),
        )

    duration_ms = round((time.perf_counter() - started) * 1000, 4)
    results = _prefer_project_context_results(results, project_path=project_path)
    return list(results), _recall_budget_metadata(
        budget,
        status="ok",
        duration_ms=duration_ms,
        budget_miss=budget.exceeded(duration_ms),
        stage_timings_ms=stage_timings,
        fallback_status=fallback_status,
        fallback_result_count=len(fallback_results),
    )


async def _maybe_run_success_fast_fallback(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    cfg: Any,
    recall_metadata: dict[str, Any],
    started: float,
    results: list[dict],
    packets: Sequence[Mapping[str, Any]],
    context_packets: Sequence[Mapping[str, Any]],
    project_path: str | None = None,
) -> list[dict]:
    """Use fast episode/cue fallback when successful deep recall is empty or weak."""
    if (
        recall_metadata.get("status") != "ok"
        or packets
        or context_packets
        or recall_metadata.get("fallback_status") != "not_run"
    ):
        return results
    live_results_empty = not results
    live_results_low_overlap = bool(
        results and not _recall_results_satisfy_query(results, query=query)
    )
    if not live_results_empty and not live_results_low_overlap:
        return results

    fallback_started = time.perf_counter()
    fallback_results, fallback_status = await _run_fast_recall_fallback(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
        project_path=project_path,
        timeout_seconds=_fast_recall_fallback_timeout_seconds(cfg),
    )
    stage_timings = recall_metadata.setdefault("stage_timings_ms", {})
    stage_key = (
        "recall_empty_success_fallback" if live_results_empty else "recall_low_overlap_fallback"
    )
    stage_timings[stage_key] = _elapsed_ms(fallback_started)
    recall_metadata["fallback_status"] = (
        fallback_status if live_results_empty else f"quality_rescue_{fallback_status}"
    )
    recall_metadata["fallback_result_count"] = len(fallback_results)
    recall_metadata["duration_ms"] = _elapsed_ms(started)
    recall_metadata["budget_miss"] = _metadata_budget_exceeded(recall_metadata)
    return list(fallback_results) if fallback_results else results


async def _record_recall_budget_event(
    manager: Any,
    *,
    group_id: str,
    operation_source: str,
    budget: RecallBudget,
    status: str,
    skip_reason: str,
    duration_ms: float,
    timeout: bool,
    budget_miss: bool,
    result_count: int = 0,
) -> None:
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="recall",
            source=operation_source,
            mode=operation_source,
            status=status,
            duration_ms=duration_ms,
            skip_reason=skip_reason,
            timeout=timeout,
            degraded=bool(timeout or (budget.timeout_degrades and budget_miss)),
            budget_miss=budget_miss,
            budget_ms=budget.budget_ms,
            budget_tokens=budget.budget_tokens,
            result_count=result_count,
        ),
    )


async def _cache_satisfied_recall_metadata(
    manager: Any,
    *,
    group_id: str,
    operation_source: str,
    cfg: Any,
    limit: int,
    packet_count: int,
    duration_ms: float,
    stage_timings_ms: dict[str, float],
) -> dict[str, Any]:
    """Record and describe explicit recall satisfied by cached packets."""
    budget = recall_budget_for_profile(
        cfg,
        budget_profile_for_source(operation_source),
        surface=surface_for_source(operation_source),
        mode=operation_source,
        max_results=limit,
    )
    stage_timings = dict(stage_timings_ms)
    stage_timings["cache_satisfied"] = duration_ms
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="recall",
            source=operation_source,
            mode=operation_source,
            status="ok",
            duration_ms=duration_ms,
            skip_reason="cache_satisfied",
            timeout=False,
            degraded=False,
            budget_miss=budget.exceeded(duration_ms),
            budget_ms=budget.budget_ms,
            budget_tokens=budget.budget_tokens,
            cache_hit=True,
            result_count=0,
            packet_count=packet_count,
        ),
    )
    return _recall_budget_metadata(
        budget,
        status="ok",
        duration_ms=duration_ms,
        skip_reason="cache_satisfied",
        timeout=False,
        budget_miss=budget.exceeded(duration_ms),
        stage_timings_ms=stage_timings,
        fallback_status="cache_satisfied",
        fallback_result_count=0,
    )


def _recall_budget_metadata(
    budget: RecallBudget,
    *,
    status: str,
    duration_ms: float,
    skip_reason: str | None = None,
    timeout: bool = False,
    budget_miss: bool = False,
    stage_timings_ms: dict[str, float] | None = None,
    fallback_status: str | None = None,
    fallback_result_count: int = 0,
) -> dict[str, Any]:
    return {
        "status": status,
        "duration_ms": duration_ms,
        "skip_reason": skip_reason,
        "timeout": timeout,
        "degraded": bool(timeout or (budget.timeout_degrades and budget_miss)),
        "budget_miss": budget_miss,
        "budget": budget.to_dict(),
        "stage_timings_ms": dict(stage_timings_ms or {}),
        "fallback_status": fallback_status,
        "fallback_result_count": max(0, int(fallback_result_count)),
    }


def _mark_project_file_recall_fallback(
    metadata: dict[str, Any],
    *,
    packet_count: int,
) -> None:
    """Mark that local project packets rescued an otherwise empty/degraded recall.

    Project-file rescue is not a clean graph-memory win — keep status honest.
    """
    metadata["fallback_status"] = "project_file_recall_fallback"
    metadata["fallback_result_count"] = 0
    metadata["project_file_packet_count"] = max(0, int(packet_count))
    metadata["degraded"] = True
    # Do not present multi-second timeout-then-docs as unqualified success.
    if metadata.get("status") == "ok" or not metadata.get("status"):
        metadata["status"] = "degraded"
    if not metadata.get("skip_reason"):
        metadata["skip_reason"] = "project_file_fallback"


def _pop_early_project_file_packets(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Remove and return packets resolved before deep recall was needed."""
    raw_packets = metadata.pop("_project_file_packets", None)
    if not isinstance(raw_packets, list):
        return []
    return [dict(packet) for packet in raw_packets if isinstance(packet, Mapping)]


def _mark_context_packet_recall_fallback(
    metadata: dict[str, Any],
    *,
    packet_count: int,
) -> None:
    """Mark that cached context packets carried an otherwise empty recall.

    Soft-hold context packets after a graph miss remain a fallback path, not a
    clean primary hit (unless already marked cache_satisfied earlier).
    """
    metadata["fallback_status"] = "context_packet_fallback"
    metadata["fallback_result_count"] = 0
    metadata["context_packet_count"] = max(0, int(packet_count))
    # When primary search already degraded/timed out, keep degraded visible.
    if metadata.get("timeout") or metadata.get("budget_miss") or metadata.get("degraded"):
        metadata["status"] = "degraded"
        metadata["degraded"] = True


def _metadata_budget_exceeded(metadata: Mapping[str, Any]) -> bool:
    budget = metadata.get("budget")
    duration_ms = metadata.get("duration_ms")
    if not isinstance(budget, Mapping) or not isinstance(duration_ms, int | float):
        return bool(metadata.get("budget_miss"))
    max_wall_ms = budget.get("max_wall_ms")
    if not isinstance(max_wall_ms, int | float) or max_wall_ms <= 0:
        return False
    return float(duration_ms) > float(max_wall_ms)


def _project_file_empty_success_wait_seconds(
    metadata: Mapping[str, Any],
    started: float,
) -> float:
    budget = metadata.get("budget")
    if not isinstance(budget, Mapping):
        return PROJECT_FILE_RECALL_FALLBACK_WAIT_SECONDS
    max_wall_ms = budget.get("max_wall_ms")
    if not isinstance(max_wall_ms, int | float) or max_wall_ms <= 0:
        return PROJECT_FILE_RECALL_FALLBACK_WAIT_SECONDS
    elapsed_seconds = max(0.0, time.perf_counter() - started)
    remaining_seconds = (float(max_wall_ms) / 1000.0) - elapsed_seconds
    return max(
        0.0,
        min(PROJECT_FILE_RECALL_EMPTY_SUCCESS_WAIT_SECONDS, remaining_seconds),
    )


def _manager_recall_stage_timings(manager: Any) -> dict[str, float]:
    getter = getattr(manager, "get_last_recall_stage_timings", None)
    if not callable(getter):
        return {}
    timings = getter()
    if inspect.isawaitable(timings):
        close = getattr(timings, "close", None)
        if callable(close):
            close()
        return {}
    if not isinstance(timings, dict):
        return {}
    return {
        str(key): float(value) for key, value in timings.items() if isinstance(value, int | float)
    }


def _manager_recall_partial_results(manager: Any) -> list[dict]:
    getter = getattr(manager, "get_last_recall_partial_results", None)
    if not callable(getter):
        return []
    results = getter()
    if inspect.isawaitable(results):
        close = getattr(results, "close", None)
        if callable(close):
            close()
        return []
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


async def _run_fast_recall_fallback(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    project_path: str | None = None,
    timeout_seconds: float = FAST_RECALL_FALLBACK_TIMEOUT_SECONDS,
) -> tuple[list[dict], str]:
    fallback = getattr(manager, "fast_recall_fallback", None)
    if not callable(fallback):
        return [], "unavailable"
    if timeout_seconds <= 0:
        return [], "disabled"
    try:
        value = fallback(query=query, group_id=group_id, limit=limit)
        if inspect.isawaitable(value):
            value = await asyncio.wait_for(
                value,
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError:
        return [], "timeout"
    except Exception:
        return [], "error"
    if not isinstance(value, list):
        return [], "invalid"
    filtered = _filter_fast_recall_fallback_results(value, query=query)
    if value and not filtered:
        return [], "filtered"
    filtered = _prefer_project_context_results(filtered, project_path=project_path)
    # Post-merge clamp: preference/merge steps can grow the list past the
    # caller's limit (gate-rerun measured n=13 for limit=10).
    filtered = filtered[: max(1, int(limit or 1))]
    return filtered, "hit" if filtered else "miss"


_RESCUE_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "about",
        "what",
        "when",
        "where",
        "which",
        "product",
        "memory",
        "project",
        "north",
        "star",
        "make",
        "made",
        "making",
        "decision",
        "decisions",
        "strategy",
        "strategies",
        "about",
        "ingestion",
        "prefer",
        "we",
        "did",
        "the",
        "and",
        "for",
        "not",
        "our",
        "are",
        "was",
        "were",
    }
)


def _is_decision_statement_noise(name: str) -> bool:
    """True for bootstrap/Cadence decision_statement scrap entities."""
    from engram.extraction.promotion import is_decision_statement_noise

    return is_decision_statement_noise(name)


def _rescue_query_tokens(query: str) -> list[str]:
    """Distinctive tokens for name probes (exclude generic decision/strategy words)."""
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{3,}", query)
        if token.casefold() not in _RESCUE_STOPWORDS
    ]
    # Longest first — more specific probes.
    tokens = sorted(set(tokens), key=len, reverse=True)[:6]
    head = query.strip().split(".")[0].strip()
    if head and len(head) >= 8:
        tokens = [head[:120], *tokens]
    if not tokens:
        tokens = [query.strip()[:80]] if query.strip() else []
    return tokens


def _name_query_overlap_score(name: str, query: str, probe_token: str) -> float:
    """Score how well an entity name matches the query (0..1+).

    Requires real content overlap — not merely sharing the word "decision".
    """
    name_l = (name or "").casefold()
    query_l = (query or "").casefold()
    probe_l = (probe_token or "").casefold()
    if not name_l or not query_l:
        return 0.0

    score = 0.0
    # Exact / near-exact name containment.
    if name_l == query_l or name_l in query_l or query_l in name_l:
        score += 1.0
    if probe_l and len(probe_l) >= 5 and probe_l in name_l:
        score += 0.45 if len(probe_l) >= 8 else 0.25

    query_parts = {p for p in re.findall(r"[a-z0-9]{4,}", query_l) if p not in _RESCUE_STOPWORDS}
    name_parts = set(re.findall(r"[a-z0-9]{4,}", name_l))
    if query_parts and name_parts:
        overlap = query_parts & name_parts
        # Need at least one distinctive shared token beyond noise.
        if overlap:
            score += min(0.5, 0.15 * len(overlap))
            score += 0.2 * (len(overlap) / max(1, len(query_parts)))
        else:
            return 0.0
    return score


async def _durable_entity_name_rescue(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    timeout_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """Bounded name lookup for durable entities when preflight times out.

    Explicit golden-path success: a freshly remembered Decision/Preference must
    still surface even when episode/cue preflight and deep hybrid search blow
    the budget on a loaded brain.

    Hard rules:
    - Drop ``*:decision_statement:*`` bootstrap/Cadence noise.
    - Require real name↔query overlap (not generic "decision"/"strategy" alone).
    - Prefer high-overlap durable types over weak Decision scrap.
    """
    try:
        return await asyncio.wait_for(
            _durable_entity_name_rescue_inner(
                manager,
                group_id=group_id,
                query=query,
                limit=limit,
                timeout_seconds=timeout_seconds,
            ),
            # Aggregate wall bound: per-probe timeouts alone allowed up to
            # 4 probes x ~1.9s of stacked waits (~7.6s theoretical) before
            # the budgeted stages even started.
            timeout=max(0.2, timeout_seconds * 2.0),
        )
    except TimeoutError:
        return []


# Only these types justify replacing the whole deep pipeline with name-stub
# results. Person/Project/Organization are durable for RANKING but far too
# broad for a short-circuit ("Engram" or the user's name appears in most
# queries).
_HIGH_SIGNAL_SHORT_CIRCUIT_TYPES = frozenset(
    {"Decision", "Preference", "Correction", "Goal", "Commitment", "Intention"}
)

_DURABLE_LOOKUP_INTENT_TERMS = (
    "decision",
    "decide",
    "decided",
    "preference",
    "prefer",
    "goal",
    "commitment",
    "committed",
    "correction",
    "policy",
    "convention",
    "strategy",
    "north star",
    "agreed",
    "agreement",
    "rule we",
    "what did we",
    "did we settle",
    "remind me what",
)


def _durable_first_short_circuit_allowed(query: str, hits: list[dict[str, Any]]) -> bool:
    """Whether rescue hits may REPLACE the deep pipeline for this query."""
    q = " ".join((query or "").lower().split())
    if any(term in q for term in _DURABLE_LOOKUP_INTENT_TERMS):
        return True
    for hit in hits:
        entity = hit.get("entity") or {}
        entity_type = str(entity.get("type") or "")
        overlap = float((hit.get("score_breakdown") or {}).get("name_overlap") or 0.0)
        if entity_type in _HIGH_SIGNAL_SHORT_CIRCUIT_TYPES and overlap >= 0.6:
            return True
    return False


async def _durable_entity_name_rescue_inner(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    from engram.extraction.promotion import (
        durable_result_boost,
        is_durable_recall_entity_type,
    )

    graph = getattr(manager, "_graph", None) or getattr(manager, "graph_store", None)
    find_exact = getattr(graph, "find_entities_exact_name", None) if graph is not None else None
    find = getattr(graph, "find_entity_candidates", None) if graph is not None else None
    search_entities = getattr(manager, "search_entities", None)
    if not callable(find) and not callable(search_entities) and not callable(find_exact):
        return []

    _cfg = getattr(manager, "_cfg", None)
    _drop_triple_rescue = bool(getattr(_cfg, "recall_rescue_drop_triple_entities", True))

    tokens = _rescue_query_tokens(query)
    if not tokens:
        return []

    seen_ids: set[str] = set()
    hits: list[dict[str, Any]] = []

    async def _probe(name: str) -> list[Any]:
        probes: list[Any] = []
        # Prefer exact-name only (fast on large native brains). Full fuzzy candidate
        # search (BM25+CONTAINS) regularly burns 1s+ per probe and misses the 2s budget.
        try:
            if callable(find_exact):
                value = find_exact(name, group_id, limit=5)
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, timeout=min(timeout_seconds, 0.4))
                if isinstance(value, list):
                    probes.extend(value)
        except Exception:
            pass
        if probes:
            return probes
        try:
            if callable(find):
                try:
                    value = find(name, group_id, limit=5)
                except TypeError:
                    value = find(name, group_id)
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, timeout=timeout_seconds)
                if isinstance(value, list):
                    probes.extend(value)
        except Exception:
            pass
        try:
            if callable(search_entities) and not probes:
                value = search_entities(
                    name=name,
                    limit=max(limit * 3, 10),
                    group_id=group_id,
                )
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, timeout=timeout_seconds)
                if isinstance(value, dict):
                    probes.extend(list(value.get("entities") or value.get("items") or []))
                elif isinstance(value, list):
                    probes.extend(value)
        except Exception:
            pass
        return probes

    # Full-query first (exact Decision/Preference names). Early-exit when we
    # already have enough high-signal hits — probing every token costs ~1s each
    # on large native brains and blew the 2s product budget.
    probe_names = tokens[:3]
    full_query = " ".join((query or "").split())
    if full_query and full_query not in probe_names:
        probe_names = [full_query[:200], *probe_names]

    for token in probe_names:
        candidates = await _probe(token)
        for entity in candidates:
            if isinstance(entity, Mapping):
                entity_id = str(entity.get("id") or "")
                entity_type = str(
                    entity.get("type")
                    or entity.get("entity_type")
                    or entity.get("entityType")
                    or ""
                )
                name = str(entity.get("name") or "")
                summary = str(entity.get("summary") or "")
            else:
                entity_id = str(getattr(entity, "id", "") or "")
                entity_type = str(
                    getattr(entity, "entity_type", "") or getattr(entity, "type", "") or ""
                )
                name = str(getattr(entity, "name", "") or "")
                summary = str(getattr(entity, "summary", "") or "")
            if not entity_id or entity_id in seen_ids:
                continue
            if _is_decision_statement_noise(name):
                continue
            if _drop_triple_rescue and _is_relationship_triple_entity(name, summary):
                # Graph-edge triple, not an answer fact — must not short-circuit
                # the deep episode search on a common-word name match.
                continue

            overlap = _name_query_overlap_score(name, query, token)
            # Require real content overlap — durable type alone is not enough.
            if overlap < 0.35:
                continue

            durable = is_durable_recall_entity_type(entity_type)
            if not durable and overlap < 0.7:
                continue

            seen_ids.add(entity_id)
            score = 0.4 + overlap
            if durable:
                score += durable_result_boost(entity_type) * 0.08
            hits.append(
                {
                    "result_type": "entity",
                    "score": min(0.99, score),
                    "entity": {
                        "id": entity_id,
                        "name": name,
                        "type": entity_type,
                        "summary": summary,
                    },
                    "relationships": [],
                    "score_breakdown": {
                        "relevance_confidence": min(0.99, score),
                        "planner_support": 0.0,
                        "rescue": "durable_entity_name",
                        "name_overlap": round(overlap, 4),
                    },
                    "source": "durable_entity_rescue",
                }
            )
        # Early exit: enough durable hits after full-query / first probes.
        durable_hits = [
            h
            for h in hits
            if is_durable_recall_entity_type(str((h.get("entity") or {}).get("type") or ""))
        ]
        if len(durable_hits) >= max(1, limit):
            break

    hits.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            float((item.get("score_breakdown") or {}).get("name_overlap") or 0.0),
        ),
        reverse=True,
    )
    return hits[: max(1, limit)]


def _fast_recall_fallback_timeout_seconds(cfg: Any) -> float:
    raw_timeout_ms = getattr(
        cfg,
        "recall_fast_fallback_timeout_ms",
        int(FAST_RECALL_FALLBACK_TIMEOUT_SECONDS * 1000),
    )
    try:
        timeout_ms = int(raw_timeout_ms)
    except (TypeError, ValueError):
        timeout_ms = int(FAST_RECALL_FALLBACK_TIMEOUT_SECONDS * 1000)
    return max(0, timeout_ms) / 1000.0


def _fast_recall_preflight_timeout_seconds(cfg: Any) -> float:
    raw_timeout_ms = getattr(
        cfg,
        "recall_fast_preflight_timeout_ms",
        None,
    )
    if raw_timeout_ms is None:
        raw_timeout_ms = getattr(
            cfg,
            "recall_fast_fallback_timeout_ms",
            int(FAST_RECALL_FALLBACK_TIMEOUT_SECONDS * 1000),
        )
    try:
        timeout_ms = int(raw_timeout_ms)
    except (TypeError, ValueError):
        timeout_ms = int(FAST_RECALL_FALLBACK_TIMEOUT_SECONDS * 1000)
    return max(0, timeout_ms) / 1000.0


def _fast_recall_preflight_enabled(cfg: Any) -> bool:
    return bool(getattr(cfg, "recall_fast_preflight_enabled", True)) and (
        _fast_recall_preflight_timeout_seconds(cfg) > 0
    )


def _recall_return_partial_on_timeout_enabled(cfg: Any) -> bool:
    return bool(getattr(cfg, "recall_return_partial_on_timeout", True))


def _filter_fast_recall_fallback_results(
    results: Sequence[Mapping[str, Any]],
    *,
    query: str,
) -> list[dict]:
    """Keep fallback hits only when their visible text overlaps the query.

    The fast fallback is a timeout rescue path. It should be conservative because
    vector-only nearest-neighbor hits can otherwise look like confident results
    for nonsense or unrelated queries.
    """
    tokens = _query_tokens(query)
    if not tokens:
        return [dict(result) for result in results]
    required_matches = 1 if len(tokens) == 1 else FAST_RECALL_FALLBACK_MIN_MATCHES
    filtered: list[dict] = []
    for result in results:
        matches = _recall_result_query_matches(result, tokens)
        if len(matches) >= required_matches:
            filtered.append(dict(result))
    return filtered


def _recall_results_satisfy_query(
    results: Sequence[Mapping[str, Any]],
    *,
    query: str,
) -> bool:
    tokens = _query_tokens(query)
    if not tokens:
        return True
    required_matches = 1 if len(tokens) == 1 else FAST_RECALL_FALLBACK_MIN_MATCHES
    return any(
        len(_recall_result_query_matches(result, tokens)) >= required_matches for result in results
    )


def _recall_result_query_matches(
    result: Mapping[str, Any],
    tokens: set[str],
) -> set[str]:
    text = _recall_result_search_text(result)
    return {token for token in tokens if token in text}


def _recall_result_search_text(result: Mapping[str, Any]) -> str:
    parts: list[str] = []
    episode = result.get("episode")
    if isinstance(episode, Mapping):
        for key in ("content", "source", "id"):
            parts.append(str(episode.get(key) or ""))
    cue = result.get("cue")
    if isinstance(cue, Mapping):
        parts.append(str(cue.get("cue_text") or ""))
        spans = cue.get("supporting_spans")
        if isinstance(spans, list | tuple):
            parts.extend(str(span) for span in spans)
    for linked in result.get("linked_entities") or []:
        if isinstance(linked, Mapping):
            parts.extend(str(linked.get(key) or "") for key in ("name", "summary", "id"))
        else:
            parts.append(str(linked))
    return " ".join(parts).lower()


def _recall_query_with_project_context(query: str, project_path: str | None) -> str:
    project_name = _project_name(project_path)
    if not project_name:
        return query
    query_text = query.strip()
    if project_name.lower() in query_text.lower():
        return query_text
    return f"{project_name} {query_text}" if query_text else project_name


def _prefer_project_context_results(
    results: Sequence[Mapping[str, Any]],
    *,
    project_path: str | None,
) -> list[dict]:
    result_list = [dict(result) for result in results]
    project_name = _project_name(project_path)
    if not project_name:
        return result_list
    project_terms = {
        project_name.lower(),
        str(Path(project_path).expanduser()).lower() if project_path else "",
    }
    project_terms.discard("")
    project_results = [
        result
        for result in result_list
        if any(term in _recall_result_search_text(result) for term in project_terms)
    ]
    return project_results or result_list


def _project_name(project_path: str | None) -> str | None:
    if not project_path:
        return None
    name = Path(project_path).expanduser().name.strip()
    return name or None


def _attach_recall_budget_metadata(
    response: dict[str, Any],
    metadata: dict[str, Any],
    *,
    camel_case: bool,
) -> None:
    response["status"] = metadata["status"]
    budget = metadata["budget"]
    stage_timings = dict(metadata.get("stage_timings_ms") or {})
    if camel_case:
        response["budget"] = {
            "profile": budget["profile"],
            "surface": budget["surface"],
            "mode": budget["mode"],
            "maxWallMs": budget["max_wall_ms"],
            "maxSearchMs": budget["max_search_ms"],
            "maxResults": budget["max_results"],
            "durationMs": metadata["duration_ms"],
            "budgetMiss": metadata["budget_miss"],
            "timeout": metadata["timeout"],
            "degraded": metadata["degraded"],
            "skipReason": metadata["skip_reason"],
        }
        lifecycle = response.setdefault("lifecycle", {})
        lifecycle["degraded"] = metadata["degraded"]
        lifecycle["skipReason"] = metadata["skip_reason"]
        lifecycle["timeout"] = metadata["timeout"]
        lifecycle["fallbackStatus"] = metadata.get("fallback_status")
        lifecycle["fallbackResultCount"] = metadata.get("fallback_result_count", 0)
        response.setdefault("diagnostics", {})["stageTimingsMs"] = {
            _camel_stage_name(key): value for key, value in stage_timings.items()
        }
        return
    response["budget"] = {
        "profile": budget["profile"],
        "surface": budget["surface"],
        "mode": budget["mode"],
        "max_wall_ms": budget["max_wall_ms"],
        "max_search_ms": budget["max_search_ms"],
        "max_results": budget["max_results"],
        "duration_ms": metadata["duration_ms"],
        "budget_miss": metadata["budget_miss"],
        "timeout": metadata["timeout"],
        "degraded": metadata["degraded"],
        "skip_reason": metadata["skip_reason"],
    }
    lifecycle = response.setdefault("lifecycle", {})
    lifecycle["degraded"] = metadata["degraded"]
    lifecycle["skip_reason"] = metadata["skip_reason"]
    lifecycle["timeout"] = metadata["timeout"]
    lifecycle["fallback_status"] = metadata.get("fallback_status")
    lifecycle["fallback_result_count"] = metadata.get("fallback_result_count", 0)
    response.setdefault("diagnostics", {})["stage_timings_ms"] = stage_timings


async def build_mcp_explicit_recall_tool_surface(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    cfg: Any,
    session: Any,
    recall_middleware: Callable[..., Awaitable[None]],
    project_path: str | None = None,
    lookup_kind: str = "general",
    entity_name: str | None = None,
    entity_type: str | None = None,
    subject: str | None = None,
    predicate: str | None = None,
    include_expired: bool = False,
    include_epistemic: bool = False,
    perf_counter: Callable[[], float] = time.perf_counter,
    time_source: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Build the MCP recall tool payload and update recall session state."""
    started = perf_counter()
    response = await build_mcp_recall_surface(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
        cfg=cfg,
        project_path=project_path,
    )
    lookup_kind_normalized = (lookup_kind or "general").strip().lower()
    if lookup_kind_normalized == "entities":
        from engram.retrieval.lookup import build_mcp_entity_search_surface

        lookup = await build_mcp_entity_search_surface(
            manager,
            group_id=group_id,
            name=entity_name or query,
            entity_type=entity_type,
            limit=limit,
        )
        if lookup.get("status") != "error":
            response["entities"] = lookup.get("entities", [])
            response["entityTotal"] = lookup.get("total", 0)
            response["lookupKind"] = "entities"
    elif lookup_kind_normalized == "facts":
        from engram.retrieval.lookup import build_mcp_fact_search_surface

        lookup = await build_mcp_fact_search_surface(
            manager,
            group_id=group_id,
            query=query,
            subject=subject,
            predicate=predicate,
            include_expired=include_expired,
            include_epistemic=include_epistemic,
            limit=limit,
        )
        response["facts"] = lookup.get("facts", [])
        response["factTotal"] = lookup.get("total", 0)
        response["lookupKind"] = "facts"
    response["query_time_ms"] = round((perf_counter() - started) * 1000, 1)
    session.last_recall_time = time_source()
    session.auto_recall_primed = True
    await recall_middleware(query, response, tool_name="recall")
    return response


def _mcp_recall_entity_name_resolver(manager: Any, group_id: str) -> ResolveNameFn:
    """Return the MCP recall entity-name resolver through the manager facade."""

    async def resolve_entity_name(entity_id: str) -> str:
        return await manager.resolve_entity_name(entity_id, group_id)

    return resolve_entity_name


def _mcp_recall_access_count_resolver(manager: Any) -> AccessCountFn:
    """Return the MCP recall access-count resolver through the manager facade."""

    async def get_access_count(entity_id: str) -> int:
        if not entity_id:
            return 0
        value = await manager.get_recall_item_access_count(entity_id)
        return value if isinstance(value, int) else 0

    return get_access_count


async def attach_mcp_explicit_recall_enrichment(
    manager: Any,
    response: dict[str, Any],
    *,
    group_id: str,
    now: float | None = None,
) -> None:
    """Attach MCP explicit-recall near-miss and surprise views when available."""
    near_misses = manager.get_last_near_miss_views()
    if inspect.isawaitable(near_misses):
        near_misses = await near_misses
    if isinstance(near_misses, list) and near_misses:
        response["near_misses"] = near_misses

    surprises = manager.get_surprise_connection_views(
        group_id,
        now=time.time() if now is None else now,
        limit=3,
    )
    if inspect.isawaitable(surprises):
        surprises = await surprises
    if isinstance(surprises, list) and surprises:
        results = response.get("results") or response.get("items") or []
        degraded = bool(
            response.get("status") == "degraded"
            or (response.get("lifecycle") or {}).get("timeout")
            or (response.get("lifecycle") or {}).get("degraded")
        )
        if not results and degraded:
            # Do not present surprise edges as a soft success when primary
            # recall failed — agents treat them as memory hits.
            response["surprise_connections_suppressed"] = True
            response["surprise_connections_note"] = (
                "Surprise edges omitted because primary recall returned no results "
                "(timeout/degraded). Graph edges may still exist; search failed."
            )
        else:
            response["surprise_connections"] = surprises


async def assemble_explicit_recall_packet_payloads(
    manager: Any,
    *,
    group_id: str,
    query: str,
    results: Sequence[dict],
    enabled: bool,
    max_packets: int,
    cfg: Any,
    serializer: PacketSerializer,
    operation_source: str = "recall",
) -> list[dict[str, Any]]:
    """Assemble explicit-recall packets and serialize them for a public surface."""
    if not enabled:
        return []

    budget = recall_budget_for_profile(
        cfg,
        budget_profile_for_source(operation_source),
        surface=surface_for_source(operation_source),
        mode=operation_source,
        max_packets=max_packets,
    )
    if max_packets <= 0 or budget.max_packets <= 0:
        return []

    cache_scope = EXPLICIT_RECALL_PACKET_CACHE_SCOPE
    cache_hit = _get_cached_packets(
        manager,
        group_id=group_id,
        scope=cache_scope,
        topic_hint=query,
    )
    if cache_hit is not None:
        await record_manager_memory_operation(
            manager,
            group_id,
            MemoryOperationSample(
                operation="packet_cache",
                source=operation_source,
                mode=cache_scope,
                status="ok",
                duration_ms=0.0,
                cache_hit=True,
                packet_count=len(cache_hit.packets),
                budget_ms=budget.budget_ms,
                budget_tokens=budget.budget_tokens,
            ),
        )
        return cache_hit.packets

    started = time.perf_counter()
    timeout_seconds = budget.stage_timeout_seconds(budget.max_packet_ms)
    if timeout_seconds <= 0:
        await _record_packet_assembly_budget_event(
            manager,
            group_id=group_id,
            operation_source=operation_source,
            cache_scope=cache_scope,
            budget=budget,
            status="skipped",
            skip_reason="skipped_budget",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
            timeout=False,
            budget_miss=True,
        )
        return []

    try:
        payloads = await asyncio.wait_for(
            _assemble_live_explicit_packet_payloads(
                manager,
                group_id=group_id,
                query=query,
                results=results,
                max_packets=max_packets,
                cfg=cfg,
                serializer=serializer,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        duration_ms = round((time.perf_counter() - started) * 1000, 4)
        await _record_packet_assembly_budget_event(
            manager,
            group_id=group_id,
            operation_source=operation_source,
            cache_scope=cache_scope,
            budget=budget,
            status="degraded",
            skip_reason="packet_timeout",
            duration_ms=duration_ms,
            timeout=True,
            budget_miss=True,
        )
        return []

    duration_ms = round((time.perf_counter() - started) * 1000, 4)
    _cache_packets(
        manager,
        group_id=group_id,
        scope=cache_scope,
        topic_hint=query,
        packets=payloads,
        build_duration_ms=duration_ms,
    )
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="packet_cache",
            source=operation_source,
            mode=cache_scope,
            status="ok",
            duration_ms=duration_ms,
            cache_hit=False,
            packet_count=len(payloads),
            budget_ms=budget.budget_ms,
            budget_tokens=budget.budget_tokens,
            budget_miss=budget.exceeded(duration_ms),
            degraded=bool(budget.timeout_degrades and budget.exceeded(duration_ms)),
        ),
    )
    return payloads


async def cached_explicit_recall_packet_payloads(
    manager: Any,
    *,
    group_id: str,
    query: str,
    max_packets: int,
    cfg: Any,
    operation_source: str,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """Return only already-cached explicit-recall packets for degraded paths."""
    if not enabled:
        return []
    budget = recall_budget_for_profile(
        cfg,
        budget_profile_for_source(operation_source),
        surface=surface_for_source(operation_source),
        mode=operation_source,
        max_packets=max_packets,
    )
    if max_packets <= 0 or budget.max_packets <= 0:
        return []
    cache_scope = EXPLICIT_RECALL_PACKET_CACHE_SCOPE
    cache_hit = _get_cached_packets(
        manager,
        group_id=group_id,
        scope=cache_scope,
        topic_hint=query,
    )
    if cache_hit is None:
        return []
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="packet_cache",
            source=operation_source,
            mode=cache_scope,
            status="ok",
            duration_ms=0.0,
            cache_hit=True,
            packet_count=len(cache_hit.packets),
            budget_ms=budget.budget_ms,
            budget_tokens=budget.budget_tokens,
        ),
    )
    return cache_hit.packets


async def cached_context_recall_packet_payloads(
    manager: Any,
    *,
    group_id: str,
    query: str,
    max_packets: int,
    cfg: Any,
    operation_source: str,
    mode: str = "context_packet_fallback",
    allow_recent_miss: bool = False,
    project_path: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent context packets when explicit recall degrades."""
    if max_packets <= 0:
        return []
    getter = getattr(manager, "get_recent_cached_memory_packets", None)
    if not callable(getter):
        return []
    started = time.perf_counter()
    packets = _scope_prioritized_context_packets(
        getter,
        group_id=group_id,
        max_packets=max_packets,
        sync_persistent=bool(project_path),
    )
    if not packets:
        return []
    packets = _filter_cached_context_packets_for_project(
        packets,
        project_path=project_path,
    )
    filtered = _filter_packets_for_query(packets, query=query, limit=max_packets)
    if not filtered:
        if not allow_recent_miss:
            filtered = _fallback_context_packets_for_project_or_identity(
                packets,
                project_path=project_path,
                limit=max_packets,
            )
            if filtered:
                mode = "context_packet_project_fallback"
            else:
                return []
        else:
            filtered = _dedupe_recent_packets(packets, limit=max_packets)
            if not filtered:
                return []
            mode = "context_packet_recent_fallback"
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="packet_cache",
            source=operation_source,
            mode=mode,
            status="ok",
            duration_ms=_elapsed_ms(started),
            cache_hit=True,
            packet_count=len(filtered),
        ),
    )
    return filtered


def _scope_prioritized_context_packets(
    getter: Callable[..., Any],
    *,
    group_id: str,
    max_packets: int,
    sync_persistent: bool,
) -> list[dict[str, Any]]:
    """Collect context-cache candidates without letting project packets starve recents."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scopes, limit, should_sync in (
        ((SESSION_RECENT_PACKET_SCOPE,), max_packets, False),
        (("identity_core",), max_packets, sync_persistent),
        (("project_home",), max_packets * 2, sync_persistent),
    ):
        for packet in _recent_packets_for_scopes(
            getter,
            group_id=group_id,
            scopes=scopes,
            limit_packets=limit,
            sync_persistent=should_sync,
        ):
            fingerprint = _packet_fingerprint(packet)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append(packet)
    return candidates


def _recent_packets_for_scopes(
    getter: Callable[..., Any],
    *,
    group_id: str,
    scopes: tuple[str, ...],
    limit_packets: int,
    sync_persistent: bool,
) -> list[dict[str, Any]]:
    try:
        packets = getter(
            group_id,
            scopes=scopes,
            limit_packets=limit_packets,
            sync_persistent=sync_persistent,
        )
    except Exception:
        return []
    if inspect.isawaitable(packets):
        close = getattr(packets, "close", None)
        if callable(close):
            close()
        return []
    if not isinstance(packets, list):
        return []
    return [dict(packet) for packet in packets if isinstance(packet, Mapping)]


async def project_file_recall_packet_payloads(
    manager: Any,
    *,
    group_id: str,
    query: str,
    project_path: str,
    max_packets: int,
    operation_source: str,
) -> list[dict[str, Any]]:
    if max_packets <= 0:
        return []
    started = time.perf_counter()
    packets = project_file_fallback_packet_payloads(
        manager,
        group_id=group_id,
        topic_hint=query,
        project_path=project_path,
        max_packets=max_packets,
        reason=(
            "Live explicit recall returned no usable memory under budget; this "
            "packet was synthesized from local project files without loaded-store reads."
        ),
        max_candidates=PROJECT_FILE_RECALL_FALLBACK_MAX_CANDIDATES,
        topic_scan_chars=PROJECT_FILE_RECALL_FALLBACK_SCAN_CHARS,
    )
    if not packets:
        return []
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="packet_cache",
            source=operation_source,
            mode="project_file_recall_fallback",
            status="ok",
            duration_ms=_elapsed_ms(started),
            cache_hit=False,
            packet_count=len(packets),
        ),
    )
    return packets


def _start_project_file_recall_fallback_task(
    *,
    query: str,
    project_path: str | None,
    max_packets: int,
) -> asyncio.Task[tuple[list[dict[str, Any]], float]] | None:
    if not project_path or max_packets <= 0:
        return None
    task = asyncio.create_task(
        _run_project_file_executor(
            _build_project_file_recall_fallback_packets,
            query=query,
            project_path=project_path,
            max_packets=max_packets,
        )
    )
    task.add_done_callback(_consume_project_file_recall_fallback_task)
    return task


def _build_project_file_recall_fallback_packets(
    *,
    query: str,
    project_path: str,
    max_packets: int,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    packets = project_file_fallback_packet_payloads(
        None,
        group_id="",
        topic_hint=query,
        project_path=project_path,
        max_packets=max_packets,
        reason=(
            "Live explicit recall returned no usable memory under budget; this "
            "packet was synthesized from local project files without loaded-store reads."
        ),
        max_candidates=PROJECT_FILE_RECALL_FALLBACK_MAX_CANDIDATES,
        topic_scan_chars=PROJECT_FILE_RECALL_FALLBACK_SCAN_CHARS,
        candidate_read_limit=PROJECT_FILE_RECALL_FALLBACK_READ_LIMIT,
        cache=False,
    )
    return packets, _elapsed_ms(started)


async def _resolve_project_file_recall_fallback_task(
    task: asyncio.Task[tuple[list[dict[str, Any]], float]] | None,
    manager: Any,
    *,
    group_id: str,
    query: str,
    project_path: str,
    max_packets: int,
    operation_source: str,
    timeout_seconds: float = PROJECT_FILE_RECALL_FALLBACK_WAIT_SECONDS,
) -> tuple[list[dict[str, Any]], float | None]:
    if task is None:
        task = _start_project_file_recall_fallback_task(
            query=query,
            project_path=project_path,
            max_packets=max_packets,
        )
    if task is None:
        return [], None
    if timeout_seconds <= 0:
        return [], None
    try:
        packets, build_duration_ms = await asyncio.wait_for(
            asyncio.shield(task),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        return [], None
    except Exception:
        return [], None
    if not packets:
        return [], build_duration_ms
    cache_project_file_fallback_packet_payloads(
        manager,
        group_id=group_id,
        topic_hint=query,
        project_path=project_path,
        packets=packets,
    )
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="packet_cache",
            source=operation_source,
            mode="project_file_recall_fallback",
            status="ok",
            duration_ms=build_duration_ms,
            cache_hit=False,
            packet_count=len(packets),
        ),
    )
    return packets, build_duration_ms


def _consume_project_file_recall_fallback_task(
    task: asyncio.Task[tuple[list[dict[str, Any]], float]],
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        return


def _default_project_path_for_recall() -> str | None:
    """Return a conservative local project fallback for recall without project_path."""
    if str(os.environ.get("ENGRAM_RECALL_PROJECT_FALLBACK") or "").lower() in {
        "0",
        "false",
        "off",
    }:
        return None
    raw_path = os.environ.get("ENGRAM_RECALL_PROJECT_PATH") or os.environ.get("ENGRAM_PROJECT_PATH")
    if raw_path:
        return str(Path(raw_path).expanduser())
    cwd = Path.cwd()
    if _looks_like_project_dir(cwd):
        return str(cwd)
    return None


def _looks_like_project_dir(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_dir():
            return False
    except OSError:
        return False
    markers = (
        ".git",
        "README.md",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "docs",
    )
    return any((path / marker).exists() for marker in markers)


def _dedupe_recent_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for packet in packets:
        fingerprint = _packet_fingerprint(packet)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(dict(packet))
        if len(selected) >= max(1, limit):
            break
    return selected


def _fallback_context_packets_for_project_or_identity(
    packets: Sequence[Mapping[str, Any]],
    *,
    project_path: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Return weak-but-useful packets when a project recall query has no hits."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for packet in packets:
        if not isinstance(packet, Mapping):
            continue
        if not (
            _packet_is_identity_core(packet)
            or _packet_is_same_project_home(packet, project_path=project_path)
        ):
            continue
        fingerprint = _packet_fingerprint(packet)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(dict(packet))
        if len(selected) >= max(1, limit):
            break
    return selected


def _packet_is_identity_core(packet: Mapping[str, Any]) -> bool:
    scope = str(packet.get("_cache_scope") or "")
    packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
    return scope == "identity_core" or packet_type in {"identity_core", "identityCore"}


def _packet_is_durable_fact(packet: Mapping[str, Any]) -> bool:
    """Graph-backed durable memory packets (not session recap / latent cues)."""
    packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
    if packet_type in {
        "cue_packet",
        "recent_observation",
        "recentObservation",
        "episode_packet",
        "recall_diagnostic",
    }:
        return False
    entity_ids = packet.get("entity_ids") or packet.get("entityIds") or []
    if entity_ids and packet_type in {
        "fact_packet",
        "state_packet",
        "open_loop_packet",
        "intention_packet",
        "identity_core",
        "identityCore",
    }:
        return True
    provenance = packet.get("provenance") or []
    if any(str(item).startswith("entity:") for item in provenance):
        return True
    trust = packet.get("trust")
    if isinstance(trust, Mapping) and str(trust.get("source") or "") == "entity":
        return True
    title = str(packet.get("title") or "").lower()
    return any(
        token in title
        for token in ("decision:", "preference:", "correction:", "person:", "fact:", "goal:")
    )


def _packet_is_same_project_home(
    packet: Mapping[str, Any],
    *,
    project_path: str | None,
) -> bool:
    scope = str(packet.get("_cache_scope") or "")
    packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
    if scope != "project_home" and packet_type not in {"project_home", "projectHome"}:
        return False
    if not project_path:
        return False
    if not _packet_has_project_file_fallback_source(packet):
        return True
    return _project_file_packet_matches_project(packet, project_path=project_path)


def _filter_cached_context_packets_for_project(
    packets: Sequence[Mapping[str, Any]],
    *,
    project_path: str | None,
) -> list[Mapping[str, Any]]:
    if not project_path:
        return list(packets)
    selected: list[Mapping[str, Any]] = []
    for packet in packets:
        if not isinstance(packet, Mapping):
            continue
        if not _packet_has_project_file_fallback_source(packet):
            if _packet_has_stale_project_file_fallback_source(packet):
                continue
            selected.append(packet)
            continue
        if _project_file_packet_matches_project(packet, project_path=project_path):
            selected.append(packet)
    return selected


def _project_file_packet_matches_project(
    packet: Mapping[str, Any],
    *,
    project_path: str,
) -> bool:
    expected = _normalize_project_path(project_path)
    if not expected:
        return False
    for key in (
        "_project_file_fallback_project_path",
        "_cache_project_path",
        "project_path",
        "projectPath",
    ):
        actual = _normalize_project_path(packet.get(key))
        if actual and actual == expected:
            return True
    return False


def _normalize_project_path(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return str(Path(text).expanduser().resolve())
    except OSError:
        return str(Path(text).expanduser())


def _packets_satisfy_explicit_query(
    packets: Sequence[Mapping[str, Any]],
    *,
    query: str,
) -> bool:
    """Return true when cached packets are specific enough to skip deep search.

    Project-file-only packets never satisfy explicit recall on their own. Explicit
    recall's product job is graph memory (Decisions, Preferences, people). Docs
    remain a last-resort fallback after graph search/rescue miss — not a
    short-circuit success that hides missing graph hits.
    """
    if not packets:
        return False
    tokens = _query_tokens(query)
    if not tokens:
        return False
    best_score = 0
    covered_tokens: set[str] = set()
    for packet in packets:
        is_graph = _packet_has_loaded_store_source(packet)
        is_project_file = _packet_has_project_file_fallback_source(packet)
        if not is_graph and not is_project_file:
            continue
        # Project-file alone never short-circuits explicit recall.
        if is_project_file and not is_graph:
            continue
        if _project_file_packet_matches_query(packet, query=query) and is_graph:
            return True
        matches = _packet_query_matches(packet, tokens)
        if not matches:
            continue
        # Session-recent observe recap also must not skip graph rescue alone.
        if _packet_is_session_recent_only(packet) and not is_graph:
            continue
        covered_tokens.update(matches)
        best_score = max(best_score, len(matches))
    return best_score >= 3 or (best_score >= 2 and len(covered_tokens) >= 4)


def _packet_is_session_recent_only(packet: Mapping[str, Any]) -> bool:
    scope = str(packet.get("_cache_scope") or "")
    packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
    return scope == SESSION_RECENT_PACKET_SCOPE or packet_type in {
        "recent_observation",
        "recentObservation",
    }


def _packet_has_loaded_store_source(packet: Mapping[str, Any]) -> bool:
    """Return whether a packet came from stored memory, not project-file fallback."""
    for key in (
        "entity_ids",
        "entityIds",
        "episode_ids",
        "episodeIds",
        "relationship_ids",
        "relationshipIds",
    ):
        value = packet.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes) and value:
            return True
    provenance = packet.get("provenance") or packet.get("sources") or []
    if isinstance(provenance, str):
        provenance_items: Sequence[Any] = (provenance,)
    elif isinstance(provenance, Sequence):
        provenance_items = provenance
    else:
        provenance_items = ()
    return any(
        str(item).startswith(("episode:", "entity:", "relationship:")) for item in provenance_items
    )


def _packet_has_project_file_fallback_source(packet: Mapping[str, Any]) -> bool:
    """Return whether a packet came from the bounded project-file fallback."""
    if packet.get("_project_file_fallback_version") != _PROJECT_FILE_FALLBACK_PACKET_VERSION:
        return False
    trust = packet.get("trust")
    if isinstance(trust, Mapping) and str(trust.get("source") or "") == "project_file":
        return True
    return str(packet.get("source") or "") == "project_file"


def _packet_has_stale_project_file_fallback_source(packet: Mapping[str, Any]) -> bool:
    trust = packet.get("trust")
    is_project_file = (
        isinstance(trust, Mapping) and str(trust.get("source") or "") == "project_file"
    ) or str(packet.get("source") or "") == "project_file"
    if not is_project_file:
        return False
    return packet.get("_project_file_fallback_version") != _PROJECT_FILE_FALLBACK_PACKET_VERSION


def _project_file_packet_matches_query(
    packet: Mapping[str, Any],
    *,
    query: str,
) -> bool:
    if not _packet_has_project_file_fallback_source(packet):
        return False
    return _normalize_cache_topic(packet.get("_project_file_fallback_topic_hint")) == (
        _normalize_cache_topic(query)
    )


def _normalize_cache_topic(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    return text or None


def _diagnostic_recall_packet(
    *,
    query: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    skip_reason = str(metadata.get("skip_reason") or "recall_degraded")
    duration_ms = metadata.get("duration_ms")
    fallback_status = metadata.get("fallback_status")
    fallback_result_count = metadata.get("fallback_result_count")
    stage_timings = metadata.get("stage_timings_ms")
    budget_miss = bool(metadata.get("budget_miss"))
    timeout = bool(metadata.get("timeout"))
    summary = (
        "No memory was surfaced for this explicit recall before the bounded recall "
        "path degraded. Treat this as no recalled evidence under the current budget, "
        "not as proof that no relevant memory exists."
    )
    evidence_lines = [
        f"query={query}",
        f"skip_reason={skip_reason}",
        f"timeout={timeout}",
        f"budget_miss={budget_miss}",
    ]
    if isinstance(duration_ms, int | float):
        evidence_lines.append(f"duration_ms={round(float(duration_ms), 4)}")
    if fallback_status:
        evidence_lines.append(f"fallback_status={fallback_status}")
    if isinstance(fallback_result_count, int):
        evidence_lines.append(f"fallback_result_count={fallback_result_count}")
    if isinstance(stage_timings, Mapping):
        for key in (
            "packet_cache",
            "recall_fallback",
            "recall_search",
            "recall_retrieve_cancelled",
        ):
            value = stage_timings.get(key)
            if isinstance(value, int | float):
                evidence_lines.append(f"{key}_ms={round(float(value), 4)}")
    return {
        "packet_type": "recall_diagnostic",
        "title": "No recalled evidence under budget",
        "summary": summary,
        "why_now": (
            "The live recall path degraded and neither cache nor fast fallback "
            "returned usable memory."
        ),
        "confidence": 1.0,
        "entity_ids": [],
        "relationship_ids": [],
        "episode_ids": [],
        "evidence_lines": evidence_lines,
        "provenance": ["runtime:recall_budget"],
        "supporting_intents": ["recall_timeout_diagnostic"],
        "trust": {
            "freshness": "runtime",
            "source": "recall_budget",
            "confidence": 1.0,
            "why_now": ("The packet reports recall runtime state; it is not a memory claim."),
            "provenance_count": 1,
            "evidence_count": len(evidence_lines),
            "belief_status": "diagnostic",
            "confirmed_count": 0,
            "corrected_count": 0,
            "dismissed_count": 0,
            "last_confirmed_at": None,
            "last_corrected_at": None,
            "last_dismissed_at": None,
        },
        "_cache_scope": "recall_diagnostic",
    }


def _filter_packets_for_query(
    packets: Sequence[Mapping[str, Any]],
    *,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    tokens = _query_tokens(query)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, packet in enumerate(packets):
        fingerprint = _packet_fingerprint(packet)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        matches = _packet_query_matches(packet, tokens)
        score = len(matches)
        exact_project_file_match = _project_file_packet_matches_query(
            packet,
            query=query,
        )
        if (
            _packet_has_project_file_fallback_source(packet)
            and not exact_project_file_match
            and not _project_file_packet_covers_required_query_tokens(
                packet,
                tokens=tokens,
                matches=matches,
            )
        ):
            continue
        if score <= 0 and tokens and not exact_project_file_match:
            continue
        if _weak_packet_query_match(matches):
            continue
        if _weak_session_recent_match(packet, matches):
            continue
        if exact_project_file_match:
            score = max(score, len(tokens))
        # Durable graph facts outrank transcript recap packets.
        durable_bonus = 3 if _packet_is_durable_fact(packet) else 0
        scope = str(packet.get("_cache_scope") or "")
        packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
        if scope == "session_recent" or packet_type in {
            "recent_observation",
            "recentObservation",
            "cue_packet",
        }:
            durable_bonus -= 2
        scored.append((score + durable_bonus, -index, dict(packet)))
    scored.sort(reverse=True)
    return [packet for _score, _index, packet in scored[: max(1, limit)]]


def _query_tokens(query: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", query.lower())
        if token
        not in {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "what",
            "where",
            "when",
            "how",
            "trace",
            "traces",
        }
    }


def _required_project_file_query_tokens(tokens: set[str]) -> set[str]:
    """Return distinctive terms required before project-file cache can satisfy recall."""
    ignored = {
        "cache",
        "cached",
        "codex",
        "context",
        "current",
        "dogfood",
        "final",
        "first",
        "live",
        "matrix",
        "nearby",
        "packet",
        "packets",
        "probe",
        "project",
        "recall",
        "reuse",
        "runtime",
        "startup",
        "trace",
    }
    return {
        token
        for token in tokens
        if not token.isdigit()
        and token not in ignored
        and (len(token) >= 6 or "-" in token or "_" in token)
    }


def _project_file_packet_covers_required_query_tokens(
    packet: Mapping[str, Any],
    *,
    tokens: set[str],
    matches: set[str],
) -> bool:
    if not _packet_has_project_file_fallback_source(packet):
        return True
    required_tokens = _required_project_file_query_tokens(tokens)
    return not required_tokens or required_tokens.issubset(matches)


def _packet_query_score(packet: Mapping[str, Any], tokens: set[str]) -> int:
    return len(_packet_query_matches(packet, tokens))


def _packet_query_matches(packet: Mapping[str, Any], tokens: set[str]) -> set[str]:
    text = _packet_search_text(packet)
    return {token for token in tokens if _token_matches_search_text(token, text)}


def _weak_session_recent_match(
    packet: Mapping[str, Any],
    matches: set[str],
) -> bool:
    if not matches:
        return False
    scope = str(packet.get("_cache_scope") or "")
    packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
    trust = packet.get("trust")
    trust_source = ""
    if isinstance(trust, Mapping):
        trust_source = str(trust.get("source") or trust.get("sourceType") or "")
    is_session_recent = scope == SESSION_RECENT_PACKET_SCOPE or packet_type in {
        "recent_observation",
        "recentObservation",
    }
    if not is_session_recent and trust_source not in {"mcp_observe", "api_auto_observe"}:
        return False
    if len(matches) >= 2:
        return False
    return not any(_high_signal_query_token(token) for token in matches)


def _weak_packet_query_match(matches: set[str]) -> bool:
    """Avoid treating a lone date/id token as useful recalled context."""
    if len(matches) != 1:
        return False
    token = next(iter(matches))
    return token.isdigit()


def _high_signal_query_token(token: str) -> bool:
    return any(char.isdigit() for char in token) or "-" in token or "_" in token or len(token) >= 10


def _token_matches_search_text(token: str, text: str) -> bool:
    if token in text:
        return True
    if "_" not in token and "-" not in token:
        return False
    parts = [part for part in re.split(r"[_-]+", token) if part]
    if len(parts) < 2:
        return False
    normalized_text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return " ".join(parts) in normalized_text


def _packet_search_text(packet: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "title",
        "summary",
        "packet_type",
        "packetType",
        "provenance",
        "supporting_intents",
        "supportingIntents",
        "evidence_lines",
        "evidenceLines",
    ):
        value = packet.get(key)
        if isinstance(value, list | tuple):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value or ""))
    return " ".join(parts).lower()


def _packet_fingerprint(packet: Mapping[str, Any]) -> str:
    provenance = packet.get("provenance") or packet.get("sources") or []
    if isinstance(provenance, list | tuple):
        provenance_text = "|".join(str(item) for item in provenance)
    else:
        provenance_text = str(provenance or "")
    if provenance_text:
        return f"provenance:{provenance_text}"
    title = str(packet.get("title") or "")
    summary = str(packet.get("summary") or "")
    packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
    return f"{packet_type}:{title}:{summary}"


async def _assemble_live_explicit_packet_payloads(
    manager: Any,
    *,
    group_id: str,
    query: str,
    results: Sequence[dict],
    max_packets: int,
    cfg: Any,
    serializer: PacketSerializer,
) -> list[dict[str, Any]]:
    packet_need = await analyze_memory_need(
        query,
        mode="explicit_recall",
        group_id=group_id,
        cfg=cfg,
        thresholds=await resolve_manager_recall_need_thresholds(manager, group_id),
    )
    packets = await assemble_memory_packets(
        list(results),
        query,
        mode="explicit_recall",
        memory_need=packet_need,
        max_packets=max_packets,
        resolve_entity_name=lambda entity_id: manager.resolve_entity_name(
            entity_id,
            group_id,
        ),
        feedback_lookup=_get_packet_feedback_lookup(manager, group_id, results),
    )
    return [serializer(packet) for packet in packets]


async def _record_packet_assembly_budget_event(
    manager: Any,
    *,
    group_id: str,
    operation_source: str,
    cache_scope: str,
    budget: RecallBudget,
    status: str,
    skip_reason: str,
    duration_ms: float,
    timeout: bool,
    budget_miss: bool,
) -> None:
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="packet_cache",
            source=operation_source,
            mode=cache_scope,
            status=status,
            duration_ms=duration_ms,
            skip_reason=skip_reason,
            timeout=timeout,
            degraded=bool(timeout or (budget.timeout_degrades and budget_miss)),
            budget_miss=budget_miss,
            budget_ms=budget.budget_ms,
            budget_tokens=budget.budget_tokens,
            cache_hit=False,
        ),
    )


def _get_cached_packets(
    manager: Any,
    *,
    group_id: str,
    scope: str,
    topic_hint: str,
) -> Any:
    get_cached = getattr(manager, "get_cached_memory_packets", None)
    if not callable(get_cached):
        return None
    hit = get_cached(
        group_id,
        scope=scope,
        topic_hint=topic_hint,
        sync_persistent=False,
    )
    if inspect.isawaitable(hit):
        close = getattr(hit, "close", None)
        if callable(close):
            close()
        return None
    packets = getattr(hit, "packets", None)
    if hit is not None and not isinstance(packets, list):
        return None
    return hit


def _cache_packets(
    manager: Any,
    *,
    group_id: str,
    scope: str,
    topic_hint: str,
    packets: Sequence[dict[str, Any]],
    build_duration_ms: float,
) -> None:
    cache = getattr(manager, "cache_memory_packets", None)
    if not callable(cache) or inspect.iscoroutinefunction(cache):
        return
    result = cache(
        group_id,
        scope=scope,
        topic_hint=topic_hint,
        packets=packets,
        build_duration_ms=build_duration_ms,
    )
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        return


def memory_packet_to_api_dict(packet: MemoryPacket) -> dict[str, Any]:
    """Convert a packet model to camelCase REST API shape."""
    result = {
        "packetType": packet.packet_type,
        "title": packet.title,
        "summary": packet.summary,
        "whyNow": packet.why_now,
        "confidence": round(packet.confidence, 4),
        "entityIds": packet.entity_ids,
        "relationshipIds": packet.relationship_ids,
        "episodeIds": packet.episode_ids,
        "evidenceLines": packet.evidence_lines,
        "provenance": packet.provenance,
        "supportingIntents": packet.supporting_intents,
    }
    if packet.trust:
        result["trust"] = {
            "freshness": packet.trust.get("freshness"),
            "source": packet.trust.get("source"),
            "confidence": packet.trust.get("confidence"),
            "whyNow": packet.trust.get("why_now"),
            "provenanceCount": packet.trust.get("provenance_count"),
            "evidenceCount": packet.trust.get("evidence_count"),
            "beliefStatus": packet.trust.get("belief_status"),
            "confirmedCount": packet.trust.get("confirmed_count"),
            "correctedCount": packet.trust.get("corrected_count"),
            "dismissedCount": packet.trust.get("dismissed_count"),
            "lastConfirmedAt": packet.trust.get("last_confirmed_at"),
            "lastCorrectedAt": packet.trust.get("last_corrected_at"),
            "lastDismissedAt": packet.trust.get("last_dismissed_at"),
        }
    return result


def _get_packet_feedback_lookup(
    manager: Any,
    group_id: str,
    results: Sequence[dict],
) -> dict[str, dict[str, Any]]:
    memory_ids = _packet_feedback_ids(results)
    if not memory_ids:
        return {}
    getter = getattr(manager, "get_recall_feedback_summary", None)
    if not callable(getter):
        return {}
    lookup = getter(group_id=group_id, memory_ids=memory_ids)
    if inspect.isawaitable(lookup):
        # Packet assembly call sites are already async, but this surface expects
        # the manager feedback lookup to be an in-memory sync read.
        close = getattr(lookup, "close", None)
        if callable(close):
            close()
        return {}
    return dict(lookup or {})


def _packet_feedback_ids(results: Sequence[dict]) -> list[str]:
    memory_ids: list[str] = []
    for result in results:
        result_type = result.get("result_type")
        if result_type == "cue_episode":
            cue = result.get("cue") if isinstance(result.get("cue"), dict) else {}
            episode = result.get("episode") if isinstance(result.get("episode"), dict) else {}
            episode_id = cue.get("episode_id") or episode.get("id")
            if episode_id:
                memory_ids.extend([f"cue:{episode_id}", episode_id, f"episode:{episode_id}"])
            continue
        if result_type == "episode":
            episode = result.get("episode") if isinstance(result.get("episode"), dict) else {}
            episode_id = episode.get("id")
            if episode_id:
                memory_ids.extend([episode_id, f"episode:{episode_id}"])
            continue
        entity = result.get("entity") if isinstance(result.get("entity"), dict) else {}
        entity_id = entity.get("id")
        if entity_id:
            memory_ids.append(entity_id)
    return list(dict.fromkeys(memory_ids))
