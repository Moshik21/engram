"""Explicit recall response builders shared by REST and MCP."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from engram.models.recall import MemoryPacket
from engram.retrieval.control import resolve_manager_recall_need_thresholds
from engram.retrieval.need import analyze_memory_need
from engram.retrieval.packets import assemble_memory_packets
from engram.retrieval.presenter import (
    present_api_recall_items,
    present_mcp_recall_items,
)

PacketSerializer = Callable[[MemoryPacket], dict[str, Any]]
ResolveNameFn = Callable[[str], Awaitable[str]]
AccessCountFn = Callable[[str], Awaitable[int]]


async def build_api_recall_surface(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
) -> dict[str, Any]:
    """Build the REST explicit-recall payload."""
    packet_policy = manager.get_explicit_recall_packet_policy()
    results = await manager.recall(
        query=query,
        group_id=group_id,
        limit=limit,
        interaction_type="used",
        interaction_source="api_recall",
    )
    packets = await assemble_explicit_recall_packet_payloads(
        manager,
        group_id=group_id,
        query=query,
        results=results,
        enabled=packet_policy.enabled,
        max_packets=packet_policy.max_packets,
        cfg=manager.get_memory_need_config(),
        serializer=memory_packet_to_api_dict,
    )
    return {
        "items": present_api_recall_items(results),
        "packets": packets,
        "query": query,
    }


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
    results = await manager.recall(
        query=query,
        group_id=group_id,
        limit=limit,
        interaction_type="used",
        interaction_source="mcp_recall",
    )
    if resolve_entity_name is None:
        resolve_entity_name = _mcp_recall_entity_name_resolver(manager, group_id)
    if get_access_count is None:
        get_access_count = _mcp_recall_access_count_resolver(manager)

    formatted = await present_mcp_recall_items(
        results,
        resolve_entity_name=resolve_entity_name,
        get_access_count=get_access_count,
    )
    packets = await assemble_explicit_recall_packet_payloads(
        manager,
        group_id=group_id,
        query=query,
        results=results,
        enabled=bool(getattr(cfg, "recall_packets_enabled", False)),
        max_packets=int(getattr(cfg, "recall_packet_explicit_limit", 0) or 0),
        cfg=cfg,
        serializer=lambda packet: packet.to_dict(),
    )
    response = {
        "packets": packets,
        "results": formatted,
        "total_candidates": len(formatted),
    }
    await attach_mcp_explicit_recall_enrichment(
        manager,
        response,
        group_id=group_id,
    )
    return response


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
) -> list[dict[str, Any]]:
    """Assemble explicit-recall packets and serialize them for a public surface."""
    if not enabled:
        return []

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
    )
    return [serializer(packet) for packet in packets]


def memory_packet_to_api_dict(packet: MemoryPacket) -> dict[str, Any]:
    """Convert a packet model to camelCase REST API shape."""
    return {
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
