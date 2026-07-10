"""Memory packet assembly for recall surfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from engram.extraction.promotion import (
    durable_result_boost,
    is_durable_recall_entity_type,
)
from engram.models.recall import MemoryNeed, MemoryPacket

ResolveNameFn = Callable[[str], Awaitable[str]]
FeedbackLookup = Mapping[str, Mapping[str, Any]]

_PROJECT_ENTITY_TYPES = {"Project", "Task", "Issue", "Goal", "Organization"}


def _result_assembly_priority(result: dict) -> tuple:
    """Prefer durable entities over cue/episode recap when assembling packets."""
    result_type = str(result.get("result_type") or "")
    entity = result.get("entity") or {}
    entity_type = str(entity.get("type") or entity.get("entity_type") or "")
    score = float(result.get("score") or 0.0)
    boost = durable_result_boost(entity_type) if result_type == "entity" else 0.0
    if result_type == "entity" and is_durable_recall_entity_type(entity_type):
        type_rank = 3
    elif result_type == "entity":
        type_rank = 2
    elif result_type == "episode":
        type_rank = 1
    else:
        type_rank = 0
    return (type_rank, score + boost)


async def assemble_memory_packets(
    results: list[dict],
    query: str,
    *,
    mode: str = "explicit_recall",
    memory_need: MemoryNeed | None = None,
    max_packets: int = 3,
    resolve_entity_name: ResolveNameFn | None = None,
    feedback_lookup: FeedbackLookup | None = None,
) -> list[MemoryPacket]:
    """Assemble compact packets from raw recall results."""
    if not results or max_packets <= 0:
        return []

    packets: list[MemoryPacket] = []
    used_entities: set[str] = set()
    used_episodes: set[str] = set()

    ordered_results = sorted(results, key=_result_assembly_priority, reverse=True)

    for result in ordered_results:
        if len(packets) >= max_packets:
            break

        if result.get("result_type") == "episode":
            episode = result.get("episode", {})
            episode_id = episode.get("id")
            if not episode_id or episode_id in used_episodes:
                continue

            packet_type = _episode_packet_type(query, memory_need)
            if packet_type == "timeline_packet":
                timeline_packet, episode_ids = _build_timeline_packet(
                    results,
                    query,
                    memory_need,
                    resolve_entity_name=resolve_entity_name,
                    feedback_lookup=feedback_lookup,
                )
                if timeline_packet is not None:
                    packets.append(timeline_packet)
                    used_episodes.update(episode_ids)
                continue

            packets.append(
                _build_episode_packet(
                    result,
                    query,
                    memory_need,
                    feedback_lookup=feedback_lookup,
                )
            )
            used_episodes.add(episode_id)
            continue

        if result.get("result_type") == "cue_episode":
            cue = result.get("cue", {})
            episode = result.get("episode", {})
            episode_id = cue.get("episode_id") or episode.get("id")
            if not episode_id or episode_id in used_episodes:
                continue

            packets.append(
                _build_cue_packet(
                    result,
                    query,
                    memory_need,
                    feedback_lookup=feedback_lookup,
                )
            )
            used_episodes.add(episode_id)
            continue

        entity = result.get("entity", {})
        entity_id = entity.get("id")
        if not entity_id or entity_id in used_entities:
            continue

        packet = await _build_entity_packet(
            result,
            query,
            memory_need,
            resolve_entity_name=resolve_entity_name,
            feedback_lookup=feedback_lookup,
        )
        packets.append(packet)
        used_entities.add(entity_id)

    return packets[:max_packets]


def _episode_packet_type(query: str, memory_need: MemoryNeed | None) -> str:
    need_type = memory_need.need_type if memory_need is not None else ""
    lowered = query.lower()
    if need_type in {"temporal_update", "open_loop", "broad_context"}:
        return "timeline_packet"
    if any(token in lowered for token in ("latest", "recent", "last", "changed", "timeline")):
        return "timeline_packet"
    return "episode_packet"


def _build_episode_packet(
    result: dict,
    query: str,
    memory_need: MemoryNeed | None,
    *,
    feedback_lookup: FeedbackLookup | None = None,
) -> MemoryPacket:
    episode = result["episode"]
    content = (episode.get("content") or "").strip()
    created_at = episode.get("created_at") or "unknown time"
    score = float(result.get("score", 0.0))
    relevance = float(result.get("score_breakdown", {}).get("relevance_confidence", 0.0))
    confidence = _confidence(score, 0.0, relevance)
    why_now = _why_now(query, memory_need, "episode_packet")
    provenance = [f"episode:{episode['id']}", f"score:{round(score, 4)}"]
    evidence_lines = [content[:180]]
    return MemoryPacket(
        packet_type="episode_packet",
        title=f"Episode: {created_at}",
        summary=content[:180],
        why_now=why_now,
        confidence=confidence,
        episode_ids=[episode["id"]],
        evidence_lines=evidence_lines,
        provenance=provenance,
        trust=_trust_summary(
            source="episode",
            confidence=confidence,
            why_now=why_now,
            evidence_lines=evidence_lines,
            provenance=provenance,
            feedback=_feedback_for(feedback_lookup, episode["id"], f"episode:{episode['id']}"),
        ),
    )


def _build_cue_packet(
    result: dict,
    query: str,
    memory_need: MemoryNeed | None,
    *,
    feedback_lookup: FeedbackLookup | None = None,
) -> MemoryPacket:
    cue = result.get("cue", {})
    episode = result.get("episode", {})
    episode_id = cue.get("episode_id") or episode.get("id") or "unknown"
    cue_text = (cue.get("cue_text") or "").strip()
    supporting_spans = cue.get("supporting_spans") or []
    score = float(result.get("score", 0.0))
    relevance = float(result.get("score_breakdown", {}).get("relevance_confidence", 0.0))
    summary = cue_text[:180] if cue_text else f"Latent memory from {episode_id}"
    evidence_lines = [span[:180] for span in supporting_spans[:2]] or [summary]
    projection_state = cue.get("projection_state") or "cue_only"
    confidence = _confidence(score, 0.0, relevance)
    why_now = _why_now(query, memory_need, "cue_packet")
    provenance = [
        f"cue:{episode_id}",
        f"projection_state:{projection_state}",
        f"score:{round(score, 4)}",
    ]
    return MemoryPacket(
        packet_type="cue_packet",
        title=f"Latent Memory: {episode_id}",
        summary=summary,
        why_now=why_now,
        confidence=confidence,
        episode_ids=[episode_id],
        evidence_lines=evidence_lines,
        provenance=provenance,
        trust=_trust_summary(
            source="cue",
            confidence=confidence,
            why_now=why_now,
            evidence_lines=evidence_lines,
            provenance=provenance,
            feedback=_feedback_for(feedback_lookup, f"cue:{episode_id}", episode_id),
        ),
    )


async def _build_entity_packet(
    result: dict,
    query: str,
    memory_need: MemoryNeed | None,
    *,
    resolve_entity_name: ResolveNameFn | None = None,
    feedback_lookup: FeedbackLookup | None = None,
) -> MemoryPacket:
    entity = result["entity"]
    entity_type = entity.get("type") or "Entity"
    packet_type = _entity_packet_type(entity_type, query, memory_need)
    relationships = result.get("relationships", [])
    evidence_lines = await _build_relationship_lines(
        entity["name"],
        relationships,
        resolve_entity_name=resolve_entity_name,
    )
    if not evidence_lines and entity.get("summary"):
        evidence_lines = [entity["summary"][:160]]

    summary = _packet_summary(packet_type, entity, evidence_lines)
    score = float(result.get("score", 0.0))
    breakdown = result.get("score_breakdown", {})
    planner_support = float(breakdown.get("planner_support", 0.0))
    relevance = float(breakdown.get("relevance_confidence", 0.0))
    relationship_ids = [rel.get("id") for rel in relationships if rel.get("id")]
    intents = result.get("supporting_intents", [])
    provenance = [f"entity:{entity['id']}", f"score:{round(score, 4)}"]
    provenance.extend(f"intent:{intent}" for intent in intents)
    confidence = _confidence(score, planner_support, relevance)
    why_now = _why_now(query, memory_need, packet_type, intents=intents)
    source = "entity"

    return MemoryPacket(
        packet_type=packet_type,
        title=_packet_title(packet_type, entity["name"]),
        summary=summary,
        why_now=why_now,
        confidence=confidence,
        belief_map=result.get("belief_map"),
        entity_ids=[entity["id"]],
        relationship_ids=relationship_ids,
        evidence_lines=evidence_lines[:3],
        provenance=provenance,
        supporting_intents=intents,
        trust=_trust_summary(
            source=source,
            confidence=confidence,
            why_now=why_now,
            evidence_lines=evidence_lines[:3],
            provenance=provenance,
            belief_map=result.get("belief_map"),
            feedback=_feedback_for(feedback_lookup, entity["id"]),
        ),
    )


def _build_timeline_packet(
    results: list[dict],
    query: str,
    memory_need: MemoryNeed | None,
    *,
    resolve_entity_name: ResolveNameFn | None = None,
    feedback_lookup: FeedbackLookup | None = None,
) -> tuple[MemoryPacket | None, list[str]]:
    del resolve_entity_name  # reserved for future episode/entity enrichment
    episodes = [result for result in results if result.get("result_type") == "episode"][:2]
    if not episodes:
        return None, []

    evidence_lines: list[str] = []
    episode_ids: list[str] = []
    provenance: list[str] = []
    max_score = 0.0
    max_relevance = 0.0

    for result in episodes:
        episode = result["episode"]
        episode_ids.append(episode["id"])
        max_score = max(max_score, float(result.get("score", 0.0)))
        max_relevance = max(
            max_relevance,
            float(result.get("score_breakdown", {}).get("relevance_confidence", 0.0)),
        )
        created_at = episode.get("created_at") or "unknown time"
        content = (episode.get("content") or "").strip()
        evidence_lines.append(f"{created_at}: {content[:140]}")
        provenance.append(f"episode:{episode['id']}")

    summary = evidence_lines[0]
    if len(evidence_lines) > 1:
        summary = f"{evidence_lines[0]} | {evidence_lines[1]}"
    confidence = _confidence(max_score, 0.0, max_relevance)
    why_now = _why_now(query, memory_need, "timeline_packet")

    return (
        MemoryPacket(
            packet_type="timeline_packet",
            title=f"Timeline: {query[:60]}",
            summary=summary[:220],
            why_now=why_now,
            confidence=confidence,
            episode_ids=episode_ids,
            evidence_lines=evidence_lines,
            provenance=provenance,
            trust=_trust_summary(
                source="episode",
                confidence=confidence,
                why_now=why_now,
                evidence_lines=evidence_lines,
                provenance=provenance,
                feedback=_aggregate_feedback(
                    [
                        _feedback_for(feedback_lookup, episode_id, f"episode:{episode_id}")
                        for episode_id in episode_ids
                    ]
                ),
            ),
        ),
        episode_ids,
    )


def _entity_packet_type(
    entity_type: str,
    query: str,
    memory_need: MemoryNeed | None,
) -> str:
    need_type = memory_need.need_type if memory_need is not None else ""
    if entity_type == "Intention" or need_type == "prospective":
        return "intention_packet"
    if need_type == "open_loop":
        return "open_loop_packet"
    if need_type == "temporal_update":
        return "timeline_packet"
    if need_type == "project_state" or entity_type in _PROJECT_ENTITY_TYPES:
        return "state_packet"
    if "decide" in query.lower() or "next step" in query.lower():
        return "open_loop_packet"
    return "fact_packet"


def _packet_title(packet_type: str, entity_name: str) -> str:
    labels = {
        "fact_packet": "Fact",
        "state_packet": "State",
        "timeline_packet": "Timeline",
        "open_loop_packet": "Open Loop",
        "intention_packet": "Intention",
        "episode_packet": "Episode",
        "cue_packet": "Latent Memory",
    }
    return f"{labels.get(packet_type, 'Memory')}: {entity_name}"


def _packet_summary(packet_type: str, entity: dict, evidence_lines: list[str]) -> str:
    summary = (entity.get("summary") or "").strip()
    if packet_type == "open_loop_packet" and evidence_lines:
        return f"Pending thread around {entity['name']}. {evidence_lines[0]}"
    if packet_type == "state_packet" and summary:
        return summary[:180]
    if packet_type == "timeline_packet" and evidence_lines:
        return evidence_lines[0][:180]
    if packet_type == "intention_packet" and summary:
        return summary[:180]
    if summary:
        return summary[:180]
    if evidence_lines:
        return evidence_lines[0][:180]
    return str(entity.get("name") or "")


async def _build_relationship_lines(
    entity_name: str,
    relationships: list[dict],
    *,
    resolve_entity_name: ResolveNameFn | None = None,
) -> list[str]:
    lines: list[str] = []
    for rel in relationships[:3]:
        source_id = rel.get("source_id")
        target_id = rel.get("target_id")
        source_name = rel.get("source_name") or source_id or "unknown"
        target_name = rel.get("target_name") or target_id or "unknown"
        if resolve_entity_name is not None:
            if source_id and source_name == source_id:
                source_name = await resolve_entity_name(source_id)
            if target_id and target_name == target_id:
                target_name = await resolve_entity_name(target_id)
        predicate = (rel.get("predicate") or "RELATED_TO").replace("_", " ").lower()
        polarity = rel.get("polarity") or "positive"
        if source_name == entity_name:
            base = f"{entity_name} {predicate} {target_name}"
        elif target_name == entity_name:
            base = f"{source_name} {predicate} {entity_name}"
        else:
            base = f"{source_name} {predicate} {target_name}"
        if polarity == "negative":
            lines.append(f"Negated: {base}")
        elif polarity == "uncertain":
            lines.append(f"Uncertain: {base}")
        else:
            lines.append(base)
    return lines


def _why_now(
    query: str,
    memory_need: MemoryNeed | None,
    packet_type: str,
    *,
    intents: list[str] | None = None,
) -> str:
    if memory_need is not None and memory_need.need_type != "none":
        return f"Relevant to {memory_need.need_type} for this turn."
    if intents:
        return f"Supported by planner intents: {', '.join(intents)}."
    return f"Relevant to the recall query: {query[:80]}"


def _confidence(
    score: float,
    planner_support: float,
    relevance: float = 0.0,
) -> float:
    if relevance > 0:
        return min(0.99, relevance)
    # Fallback for results without embeddings
    clamped_score = max(0.0, min(score, 1.0))
    clamped_support = max(0.0, min(planner_support, 1.0))
    return min(0.99, max(clamped_score, (clamped_score * 0.7) + (clamped_support * 0.3)))


def _trust_summary(
    *,
    source: str,
    confidence: float,
    why_now: str,
    evidence_lines: list[str],
    provenance: list[str],
    belief_map: dict | None = None,
    feedback: Mapping[str, Any] | None = None,
) -> dict:
    return {
        "freshness": "unknown",
        "source": source,
        "confidence": round(confidence, 4),
        "why_now": why_now,
        "provenance_count": len(provenance),
        "evidence_count": len(evidence_lines),
        "belief_status": _belief_status(belief_map),
        "confirmed_count": _feedback_count(feedback, "confirmed_count"),
        "corrected_count": _feedback_count(feedback, "corrected_count"),
        "dismissed_count": _feedback_count(feedback, "dismissed_count"),
        "last_confirmed_at": _feedback_text(feedback, "last_confirmed_at"),
        "last_corrected_at": _feedback_text(feedback, "last_corrected_at"),
        "last_dismissed_at": _feedback_text(feedback, "last_dismissed_at"),
    }


def _belief_status(belief_map: dict | None) -> str:
    if not belief_map:
        return "unknown"
    status = str(belief_map.get("status") or belief_map.get("belief_status") or "")
    if status:
        return status
    if belief_map.get("conflicts") or belief_map.get("conflicting"):
        return "conflicting"
    if belief_map.get("confidence") is not None:
        return "supported"
    return "unknown"


def _feedback_for(
    feedback_lookup: FeedbackLookup | None,
    *memory_ids: str,
) -> Mapping[str, Any] | None:
    if feedback_lookup is None:
        return None
    for memory_id in memory_ids:
        feedback = feedback_lookup.get(memory_id)
        if feedback is not None:
            return feedback
    return None


def _aggregate_feedback(feedback_items: list[Mapping[str, Any] | None]) -> dict[str, Any] | None:
    items = [item for item in feedback_items if item]
    if not items:
        return None
    return {
        "confirmed_count": sum(_feedback_count(item, "confirmed_count") for item in items),
        "corrected_count": sum(_feedback_count(item, "corrected_count") for item in items),
        "dismissed_count": sum(_feedback_count(item, "dismissed_count") for item in items),
        "last_confirmed_at": _latest_feedback_text(items, "last_confirmed_at"),
        "last_corrected_at": _latest_feedback_text(items, "last_corrected_at"),
        "last_dismissed_at": _latest_feedback_text(items, "last_dismissed_at"),
    }


def _feedback_count(feedback: Mapping[str, Any] | None, key: str) -> int:
    if feedback is None:
        return 0
    value = feedback.get(key)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _feedback_text(feedback: Mapping[str, Any] | None, key: str) -> str | None:
    if feedback is None:
        return None
    value = feedback.get(key)
    text = str(value).strip() if value is not None else ""
    return text or None


def _latest_feedback_text(items: list[Mapping[str, Any]], key: str) -> str | None:
    values = [_feedback_text(item, key) for item in items]
    values = [value for value in values if value]
    return max(values) if values else None
