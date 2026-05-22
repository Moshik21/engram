"""Explicit recall response builders shared by REST and MCP."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from engram.models.recall import MemoryPacket
from engram.retrieval.budgets import (
    RecallBudget,
    budget_profile_for_source,
    recall_budget_for_profile,
    surface_for_source,
)
from engram.retrieval.control import resolve_manager_recall_need_thresholds
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
)

PacketSerializer = Callable[[MemoryPacket], dict[str, Any]]
ResolveNameFn = Callable[[str], Awaitable[str]]
AccessCountFn = Callable[[str], Awaitable[int]]
FAST_RECALL_FALLBACK_TIMEOUT_SECONDS = 0.2


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 4)


def _camel_stage_name(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


async def build_api_recall_surface(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    operation_source: str = "api_recall",
) -> dict[str, Any]:
    """Build the REST explicit-recall payload."""
    packet_policy = manager.get_explicit_recall_packet_policy()
    cfg = manager.get_memory_need_config()
    pre_stage_timings: dict[str, float] = {}
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
    results, recall_metadata = await _run_explicit_recall_with_budget(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
        cfg=cfg,
        operation_source=operation_source,
    )
    recall_metadata.setdefault("stage_timings_ms", {}).update(pre_stage_timings)
    if recall_metadata["status"] == "ok" and not packets:
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
        recall_metadata.setdefault("stage_timings_ms", {})["packet_assembly"] = (
            _elapsed_ms(packet_started)
        )
    response = present_api_recall_response(query=query, results=results, packets=packets)
    _attach_recall_budget_metadata(response, recall_metadata, camel_case=True)
    return response


async def build_mcp_recall_surface(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    cfg: Any,
    resolve_entity_name: ResolveNameFn | None = None,
    get_access_count: AccessCountFn | None = None,
) -> dict[str, Any]:
    """Build the MCP explicit-recall payload without transport-only metadata."""
    pre_stage_timings: dict[str, float] = {}
    packet_enabled = bool(getattr(cfg, "recall_packets_enabled", False))
    max_packets = int(getattr(cfg, "recall_packet_explicit_limit", 0) or 0)
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
    results, recall_metadata = await _run_explicit_recall_with_budget(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
        cfg=cfg,
        operation_source="mcp_recall",
    )
    recall_metadata.setdefault("stage_timings_ms", {}).update(pre_stage_timings)
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
    if recall_metadata["status"] == "ok" and not packets:
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
        recall_metadata.setdefault("stage_timings_ms", {})["packet_assembly"] = (
            _elapsed_ms(packet_started)
        )
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
    return response


async def _run_explicit_recall_with_budget(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    cfg: Any,
    operation_source: str,
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
    fallback_started = time.perf_counter()
    fallback_results, fallback_status = await _run_fast_recall_fallback(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
    )
    stage_timings["recall_fallback"] = _elapsed_ms(fallback_started)
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
            fallback_result_count=len(fallback_results),
        )

    try:
        recall_started = time.perf_counter()
        results = await asyncio.wait_for(
            manager.recall(
                query=query,
                group_id=group_id,
                limit=limit,
                interaction_type="used",
                interaction_source=operation_source,
            ),
            timeout=timeout_seconds,
        )
        stage_timings["recall_search"] = _elapsed_ms(recall_started)
        stage_timings.update(_manager_recall_stage_timings(manager))
    except asyncio.TimeoutError:
        stage_timings["recall_search"] = _elapsed_ms(recall_started)
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
    return list(results), _recall_budget_metadata(
        budget,
        status="ok",
        duration_ms=duration_ms,
        budget_miss=budget.exceeded(duration_ms),
        stage_timings_ms=stage_timings,
        fallback_status=fallback_status,
        fallback_result_count=len(fallback_results),
    )


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
        str(key): float(value)
        for key, value in timings.items()
        if isinstance(value, int | float)
    }


async def _run_fast_recall_fallback(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
) -> tuple[list[dict], str]:
    fallback = getattr(manager, "fast_recall_fallback", None)
    if not callable(fallback):
        return [], "unavailable"
    try:
        value = fallback(query=query, group_id=group_id, limit=limit)
        if inspect.isawaitable(value):
            value = await asyncio.wait_for(
                value,
                timeout=FAST_RECALL_FALLBACK_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        return [], "timeout"
    except Exception:
        return [], "error"
    if not isinstance(value, list):
        return [], "invalid"
    return list(value), "hit" if value else "miss"


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
    )
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

    cache_scope = f"explicit_recall:{operation_source}"
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
    cache_scope = f"explicit_recall:{operation_source}"
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
    hit = get_cached(group_id, scope=scope, topic_hint=topic_hint)
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
