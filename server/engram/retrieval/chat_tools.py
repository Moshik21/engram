"""Knowledge-chat tool execution helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from engram.retrieval.chat_events import (
    accumulate_chat_tool_result,
    build_chat_tool_result_message,
)
from engram.retrieval.chat_feedback import build_memory_grounding_retry_system_prompt
from engram.retrieval.control import resolve_manager_recall_need_thresholds
from engram.retrieval.need import analyze_memory_need
from engram.retrieval.packets import assemble_memory_packets
from engram.retrieval.presenter import present_chat_recall_items

logger = logging.getLogger(__name__)

CHAT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "recall",
        "description": (
            "Search memories by semantic similarity + activation. "
            "Returns scored entities with summaries and relationships."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {
                    "type": "integer",
                    "description": "Max results (1-20)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_entities",
        "description": (
            "Find specific entities by name (fuzzy match) or type. "
            "Use when you know or suspect the name of an entity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Entity name to search for"},
                "entity_type": {
                    "type": "string",
                    "description": "Filter by type (Person, Technology, etc.)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "search_facts",
        "description": (
            "Search for relationships/facts in the knowledge graph. "
            "Can filter by subject entity name and/or predicate type "
            "(e.g., PARENT_OF, WORKS_AT, LIVES_IN). Internal epistemic "
            "decision/artifact edges stay hidden unless include_epistemic=true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "default": "",
                },
                "subject": {
                    "type": "string",
                    "description": "Filter by subject entity name",
                },
                "predicate": {
                    "type": "string",
                    "description": "Filter by relationship type",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 10,
                },
                "include_epistemic": {
                    "type": "boolean",
                    "description": (
                        "Debug-only: include decision/artifact graph facts. "
                        "Do not use this as the primary path for project reconcile answers."
                    ),
                    "default": False,
                },
            },
        },
    },
]


def extract_message_text(blocks: object) -> str:
    """Join text-bearing Anthropic blocks without assuming concrete block types."""
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


async def retry_memory_grounded_response(
    client: Any,
    *,
    system_prompt: list[dict[str, Any]],
    loop_messages: list[dict[str, Any]],
    chat_need: Any,
    prior_response: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
) -> str:
    """Retry once with a stronger memory-grounding instruction."""
    retry_system = build_memory_grounding_retry_system_prompt(
        system_prompt,
        chat_need=chat_need,
        prior_response=prior_response,
    )
    retry_response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=retry_system,
        messages=loop_messages,
    )
    retry_text = extract_message_text(retry_response.content)
    return retry_text or prior_response


@dataclass(frozen=True)
class ChatToolUseLoopResult:
    """Result of a knowledge-chat tool-use loop."""

    response: Any
    loop_messages: list[dict[str, Any]]
    recall_results: list[dict[str, Any]]
    facts: list[dict[str, Any]]


async def run_chat_tool_use_loop(
    client: Any,
    *,
    manager: Any,
    group_id: str,
    system_prompt: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    initial_recall_results: list[dict[str, Any]] | None = None,
    max_tool_turns: int = 3,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
) -> ChatToolUseLoopResult:
    """Run the non-streaming Anthropic tool loop for knowledge chat."""
    all_recall_results: list[dict[str, Any]] = list(initial_recall_results or [])
    all_facts: list[dict[str, Any]] = []
    loop_messages = list(messages)

    response = None
    for _turn in range(max_tool_turns):
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=loop_messages,
            tools=tools,
        )

        if getattr(response, "stop_reason", None) != "tool_use":
            break

        assistant_content = response.content
        tool_results = []
        for block in assistant_content:
            if getattr(block, "type", None) != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            try:
                result_str = await execute_chat_tool_json(
                    manager,
                    group_id=group_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                )
            except Exception as exc:
                logger.warning("Chat tool %s failed: %s", tool_name, exc)
                result_str = json.dumps({"error": str(exc)})

            tool_results.append(build_chat_tool_result_message(block.id, result_str))
            accumulate_chat_tool_result(
                tool_name=tool_name,
                result=result_str,
                recall_results=all_recall_results,
                facts=all_facts,
            )

        loop_messages.append({"role": "assistant", "content": assistant_content})
        loop_messages.append({"role": "user", "content": tool_results})
    else:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=loop_messages,
        )

    return ChatToolUseLoopResult(
        response=response,
        loop_messages=loop_messages,
        recall_results=all_recall_results,
        facts=all_facts,
    )


async def execute_chat_tool_json(
    manager: Any,
    *,
    group_id: str,
    tool_name: str,
    tool_input: Mapping[str, Any],
) -> str:
    """Execute a knowledge-chat tool call and return the JSON string payload."""
    payload = await execute_chat_tool(
        manager,
        group_id=group_id,
        tool_name=tool_name,
        tool_input=tool_input,
    )
    return json.dumps(payload)


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
