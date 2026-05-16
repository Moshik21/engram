"""Chat recall event presentation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChatToolEvent:
    """Route-neutral rich tool event for knowledge-chat UI components."""

    name: str
    input: dict[str, Any]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _score(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _round_score(value: Any) -> float:
    return round(_score(value), 3)


def _score_breakdown(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(result.get("score_breakdown"))


def build_chat_tool_events(
    recall_results: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
) -> list[ChatToolEvent]:
    """Build rich chat UI tool-event payloads from recall and fact results."""
    events: list[ChatToolEvent] = []

    entities = _entity_events(recall_results)
    if entities:
        events.append(ChatToolEvent("show_entities", {"entities": entities}))

    graph_event = _relationship_graph_event(recall_results, entities)
    if graph_event is not None:
        events.append(graph_event)

    if facts:
        events.append(
            ChatToolEvent(
                "show_facts",
                {
                    "facts": [
                        {
                            "subject": fact["subject"],
                            "predicate": fact["predicate"],
                            "object": fact["object"],
                            "confidence": fact.get("confidence"),
                        }
                        for fact in facts[:10]
                    ]
                },
            )
        )

    if len(entities) >= 3:
        events.append(
            ChatToolEvent(
                "show_activation_chart",
                {
                    "entities": [
                        {
                            "name": entity["name"],
                            "entityType": entity["entityType"],
                            "activation": entity["activation"],
                        }
                        for entity in sorted(
                            entities,
                            key=lambda item: item["activation"],
                            reverse=True,
                        )[:8]
                    ]
                },
            )
        )

    episodes = _episode_events(recall_results)
    if episodes:
        events.append(ChatToolEvent("show_timeline", {"episodes": episodes}))

    return events


def build_chat_tool_result_message(tool_use_id: str, result: str) -> dict[str, str]:
    """Build the Anthropic tool_result message for a chat tool call."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": result,
    }


def accumulate_chat_tool_result(
    *,
    tool_name: str,
    result: str,
    recall_results: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> None:
    """Accumulate chat tool JSON output for rich memory UI events."""
    try:
        parsed = json.loads(result)
    except Exception:
        return

    if not isinstance(parsed, Mapping):
        return

    if tool_name == "recall":
        for item in parsed.get("results", []) or []:
            if isinstance(item, Mapping):
                recall_results.append(raw_recall_from_chat_item(item))
    elif tool_name == "search_facts":
        for fact in parsed.get("facts", []) or []:
            if isinstance(fact, Mapping):
                facts.append(dict(fact))


def raw_recall_from_chat_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Convert summarized chat recall output back to the raw recall event shape."""
    item_type = item.get("type")
    if item_type == "episode":
        return {
            "result_type": "episode",
            "episode": {
                "id": "",
                "content": item.get("content", ""),
                "source": item.get("source"),
                "created_at": None,
            },
            "score": item.get("score", 0),
            "score_breakdown": _empty_breakdown(),
        }

    if item_type == "cue_episode":
        return {
            "result_type": "cue_episode",
            "cue": {
                "episode_id": item.get("episodeId", ""),
                "cue_text": item.get("cueText", ""),
                "supporting_spans": item.get("supportingSpans", []),
                "projection_state": item.get("projectionState"),
            },
            "episode": {
                "id": item.get("episodeId", ""),
                "source": item.get("source"),
                "created_at": None,
            },
            "score": item.get("score", 0),
            "score_breakdown": _empty_breakdown(),
        }

    return {
        "result_type": "entity",
        "entity": {
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "type": item.get("entityType", "Other"),
            "summary": item.get("summary"),
        },
        "score": item.get("score", 0),
        "score_breakdown": {
            **_empty_breakdown(),
            "activation": item.get("activation", 0),
        },
        "relationships": [
            {
                "predicate": rel.get("predicate"),
                "target_id": rel.get("target", ""),
                "source_id": rel.get("source", ""),
                "polarity": rel.get("polarity", "positive"),
            }
            for rel in item.get("relationships", []) or []
            if isinstance(rel, Mapping)
        ],
    }


def _empty_breakdown() -> dict[str, float]:
    return {
        "semantic": 0,
        "activation": 0,
        "edge_proximity": 0,
        "exploration_bonus": 0,
    }


def _entity_events(recall_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for result in recall_results:
        if result.get("result_type") != "entity":
            continue
        entity = _mapping(result.get("entity"))
        entities.append(
            {
                "id": entity["id"],
                "name": entity["name"],
                "entityType": entity.get("type", "Other"),
                "summary": entity.get("summary"),
                "score": _round_score(result.get("score")),
                "activation": _round_score(_score_breakdown(result).get("activation", 0)),
            }
        )
    return entities


def _relationship_graph_event(
    recall_results: Sequence[Mapping[str, Any]],
    entities: Sequence[Mapping[str, Any]],
) -> ChatToolEvent | None:
    for result in recall_results:
        if result.get("result_type") != "entity":
            continue
        relationships = result.get("relationships", []) or []
        if len(relationships) < 3:
            continue

        entity = _mapping(result.get("entity"))
        nodes = [
            {
                "id": entity["id"],
                "name": entity["name"],
                "type": entity.get("type", "Other"),
            }
        ]
        edges = []
        seen_ids = {entity["id"]}
        for relationship in relationships[:12]:
            rel = _mapping(relationship)
            target_id = rel.get("target_id", "")
            source_id = rel.get("source_id", "")
            other_id = target_id if source_id == entity["id"] else source_id
            if other_id and other_id not in seen_ids:
                other_name = other_id
                for known_entity in entities:
                    if known_entity["id"] == other_id:
                        other_name = known_entity["name"]
                        break
                nodes.append({"id": other_id, "name": other_name, "type": "Other"})
                seen_ids.add(other_id)
            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "predicate": rel.get("predicate", "RELATED"),
                    "weight": rel.get("weight", 1.0),
                }
            )

        return ChatToolEvent(
            "show_relationship_graph",
            {
                "centralEntity": entity["name"],
                "nodes": nodes,
                "edges": edges,
            },
        )
    return None


def _episode_events(recall_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for result in recall_results:
        if result.get("result_type") == "episode":
            episode = _mapping(result.get("episode"))
            episodes.append(
                {
                    "id": episode["id"],
                    "content": str(episode.get("content", ""))[:200],
                    "source": episode.get("source"),
                    "createdAt": episode.get("created_at"),
                    "score": _round_score(result.get("score")),
                }
            )
        elif result.get("result_type") == "cue_episode":
            cue = _mapping(result.get("cue"))
            episode = _mapping(result.get("episode"))
            episodes.append(
                {
                    "id": cue.get("episode_id") or episode.get("id"),
                    "content": str(cue.get("cue_text") or "")[:200],
                    "source": episode.get("source"),
                    "createdAt": episode.get("created_at"),
                    "score": _round_score(result.get("score")),
                    "latent": True,
                }
            )
    return episodes
