"""Knowledge-chat tool execution helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from engram.retrieval.control import resolve_manager_recall_need_thresholds
from engram.retrieval.need import analyze_memory_need
from engram.retrieval.packets import assemble_memory_packets
from engram.retrieval.presenter import present_chat_recall_items

logger = logging.getLogger(__name__)


async def execute_chat_tool(
    manager: Any,
    *,
    group_id: str,
    tool_name: str,
    tool_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute a knowledge-chat tool call and return the LLM-facing payload."""
    logger.info("Chat tool call: %s(%s)", tool_name, dict(tool_input))

    if tool_name == "recall":
        return await _execute_recall_tool(manager, group_id=group_id, tool_input=tool_input)
    if tool_name == "search_entities":
        return await _execute_search_entities_tool(
            manager,
            group_id=group_id,
            tool_input=tool_input,
        )
    if tool_name == "search_facts":
        return await _execute_search_facts_tool(manager, group_id=group_id, tool_input=tool_input)
    return {"error": f"Unknown tool: {tool_name}"}


async def _execute_recall_tool(
    manager: Any,
    *,
    group_id: str,
    tool_input: Mapping[str, Any],
) -> dict[str, Any]:
    policy = manager.get_chat_tool_recall_policy()
    query = str(tool_input["query"])

    results = await manager.recall(
        query=query,
        group_id=group_id,
        limit=_bounded_limit(tool_input.get("limit"), default=5, maximum=20),
        record_access=policy.record_access,
        interaction_type=policy.interaction_type,
        interaction_source=policy.interaction_source,
    )

    packets: list[dict[str, Any]] = []
    if policy.packets_enabled:
        packet_need = await analyze_memory_need(
            query,
            mode="chat",
            group_id=group_id,
            cfg=manager.get_memory_need_config(),
            thresholds=await resolve_manager_recall_need_thresholds(manager, group_id),
        )
        packets = [
            _packet_to_chat_tool_dict(packet)
            for packet in await assemble_memory_packets(
                results,
                query,
                mode="chat_tool_use",
                memory_need=packet_need,
                max_packets=policy.packet_limit,
                resolve_entity_name=lambda entity_id: manager.resolve_entity_name(
                    entity_id,
                    group_id,
                ),
            )
        ]

    items = present_chat_recall_items(results)
    return {"packets": packets, "results": items, "total": len(items)}


async def _execute_search_entities_tool(
    manager: Any,
    *,
    group_id: str,
    tool_input: Mapping[str, Any],
) -> dict[str, Any]:
    results = await manager.search_entities(
        group_id=group_id,
        name=tool_input.get("name"),
        entity_type=tool_input.get("entity_type"),
        limit=_bounded_limit(tool_input.get("limit"), default=10, maximum=20),
    )
    items = [
        {
            "name": entity.get("name", ""),
            "entityType": entity.get("type", ""),
            "summary": entity.get("summary"),
            "id": entity.get("id", ""),
        }
        for entity in results
    ]
    return {"entities": items, "total": len(items)}


async def _execute_search_facts_tool(
    manager: Any,
    *,
    group_id: str,
    tool_input: Mapping[str, Any],
) -> dict[str, Any]:
    requested_limit = _bounded_limit(tool_input.get("limit"), default=10, maximum=20)
    results = await manager.search_facts(
        group_id=group_id,
        query=tool_input.get("query", ""),
        subject=tool_input.get("subject"),
        predicate=tool_input.get("predicate"),
        include_epistemic=bool(tool_input.get("include_epistemic", False)),
        limit=requested_limit * 2,
    )

    seen: set[tuple[str, str, str]] = set()
    items = []
    for fact in results:
        key = (fact["subject"], fact["predicate"], fact["object"])
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "subject": fact["subject"],
                "predicate": fact["predicate"],
                "object": fact["object"],
                "confidence": fact.get("confidence"),
            }
        )
        if len(items) >= requested_limit:
            break

    logger.info(
        "Chat search_facts returned %d unique facts (from %d raw)",
        len(items),
        len(results),
    )
    return {"facts": items, "total": len(items)}

def _bounded_limit(value: Any, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, 1), maximum)


def _packet_to_chat_tool_dict(packet: Any) -> dict[str, Any]:
    return {
        "packetType": packet.packet_type,
        "title": packet.title,
        "summary": packet.summary,
        "whyNow": packet.why_now,
        "confidence": round(packet.confidence, 3),
        "evidence": packet.evidence_lines[:3],
        "provenance": packet.provenance[:4],
    }
