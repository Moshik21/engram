"""Shared recall result presenters for REST, MCP, and chat surfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

ResolveEntityName = Callable[[str], Awaitable[str]]
GetAccessCount = Callable[[str], Awaitable[int]]


def _score(result: Mapping[str, Any]) -> float:
    value = result.get("score", 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _breakdown(result: Mapping[str, Any]) -> dict[str, Any]:
    raw = result.get("score_breakdown", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def _api_score_breakdown(result: Mapping[str, Any]) -> dict[str, float]:
    bd = _breakdown(result)
    return {
        "semantic": float(bd.get("semantic", 0.0) or 0.0),
        "activation": float(bd.get("activation", 0.0) or 0.0),
        "edgeProximity": float(bd.get("edge_proximity", 0.0) or 0.0),
        "explorationBonus": float(bd.get("exploration_bonus", 0.0) or 0.0),
    }


def _rounded_numeric_breakdown(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: round(value, 4)
        for key, value in _breakdown(result).items()
        if isinstance(value, int | float)
    }


def _linked_entity_names(result: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for item in result.get("linked_entities", []) or []:
        if isinstance(item, Mapping):
            name = item.get("name")
            if isinstance(name, str):
                names.append(name)
        elif isinstance(item, str):
            names.append(item)
    return names


def recall_contract_item(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a raw GraphManager recall result into a surface-neutral contract."""
    result_type = str(result.get("result_type", "entity"))
    score = _score(result)
    score_breakdown = _breakdown(result)

    if result_type == "episode":
        episode = result.get("episode", {})
        ep = episode if isinstance(episode, Mapping) else {}
        return {
            "result_type": "episode",
            "score": score,
            "score_breakdown": score_breakdown,
            "episode_id": ep.get("id"),
            "content": ep.get("content", ""),
            "source": ep.get("source"),
            "created_at": ep.get("created_at"),
            "conversation_date": ep.get("conversation_date"),
            "linked_entities": _linked_entity_names(result),
        }

    if result_type == "cue_episode":
        cue_raw = result.get("cue", {})
        cue = cue_raw if isinstance(cue_raw, Mapping) else {}
        episode_raw = result.get("episode", {})
        episode = episode_raw if isinstance(episode_raw, Mapping) else {}
        return {
            "result_type": "cue_episode",
            "score": score,
            "score_breakdown": score_breakdown,
            "episode_id": cue.get("episode_id") or episode.get("id"),
            "cue_text": cue.get("cue_text"),
            "supporting_spans": cue.get("supporting_spans", []),
            "projection_state": cue.get("projection_state"),
            "route_reason": cue.get("route_reason"),
            "hit_count": cue.get("hit_count"),
            "surfaced_count": cue.get("surfaced_count"),
            "selected_count": cue.get("selected_count"),
            "used_count": cue.get("used_count"),
            "near_miss_count": cue.get("near_miss_count"),
            "policy_score": cue.get("policy_score"),
            "last_feedback_at": cue.get("last_feedback_at"),
            "last_projected_at": cue.get("last_projected_at"),
            "source": episode.get("source"),
            "created_at": episode.get("created_at"),
            "conversation_date": episode.get("conversation_date"),
        }

    entity_raw = result.get("entity", {})
    entity = entity_raw if isinstance(entity_raw, Mapping) else {}
    return {
        "result_type": "entity",
        "score": score,
        "score_breakdown": score_breakdown,
        "entity_id": entity.get("id"),
        "entity_name": entity.get("name", ""),
        "entity_type": entity.get("type") or entity.get("entity_type", ""),
        "summary": entity.get("summary"),
        "relationships": list(result.get("relationships", []) or []),
    }


def present_api_recall_item(result: Mapping[str, Any]) -> dict[str, Any]:
    """Format one recall result for the REST API camelCase contract."""
    item = recall_contract_item(result)
    result_type = item["result_type"]

    if result_type == "episode":
        return {
            "resultType": "episode",
            "episode": {
                "id": item["episode_id"],
                "content": item["content"],
                "source": item["source"],
                "createdAt": item["created_at"],
            },
            "score": item["score"],
            "scoreBreakdown": _api_score_breakdown(result),
        }

    if result_type == "cue_episode":
        return {
            "resultType": "cue_episode",
            "cue": {
                "episodeId": item["episode_id"],
                "cueText": item["cue_text"],
                "supportingSpans": item["supporting_spans"] or [],
                "projectionState": item["projection_state"],
                "routeReason": item["route_reason"],
                "hitCount": item["hit_count"],
                "surfacedCount": item["surfaced_count"],
                "selectedCount": item["selected_count"],
                "usedCount": item["used_count"],
                "nearMissCount": item["near_miss_count"],
                "policyScore": item["policy_score"],
                "lastFeedbackAt": item["last_feedback_at"],
                "lastProjectedAt": item["last_projected_at"],
            },
            "episode": {
                "id": item["episode_id"],
                "source": item["source"],
                "createdAt": item["created_at"],
            },
            "score": item["score"],
            "scoreBreakdown": _api_score_breakdown(result),
        }

    return {
        "resultType": "entity",
        "entity": {
            "id": item["entity_id"],
            "name": item["entity_name"],
            "entityType": item["entity_type"],
            "summary": item["summary"],
        },
        "score": item["score"],
        "scoreBreakdown": _api_score_breakdown(result),
        "relationships": item["relationships"],
    }


def present_api_recall_items(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [present_api_recall_item(result) for result in results]


def recall_response_contract(
    *,
    query: str,
    result_count: int,
    packet_count: int,
) -> dict[str, Any]:
    """Normalize explicit Recall response metadata across public surfaces."""
    return {
        "operation": "recall",
        "lifecycle_stage": "recall",
        "recall_mode": "explicit",
        "query": query,
        "result_count": max(0, int(result_count)),
        "packet_count": max(0, int(packet_count)),
    }


def results_from_packets(
    packets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Mirror packet sources into naive-client `results`/`items` shapes.

    Cache-satisfied recall historically returned packets with ``results: []``,
    which made agents that only read ``results`` conclude "no memory". Packets
    remain authoritative; these rows are a presentation convenience.
    """
    mirrored: list[dict[str, Any]] = []
    seen: set[str] = set()

    for packet in packets:
        if not isinstance(packet, Mapping):
            continue
        packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
        # Diagnostic "no memory" packets must not become fake hits.
        if packet_type in {"recall_diagnostic", "recallDiagnostic"}:
            continue
        # Project-file/degraded fallback packets still mirror (naive clients
        # should see the fallback), but never as clean score=1.0 memory —
        # that made an empty graph look like strong recall of the repo's own
        # files.
        is_fallback = packet_type in {"project_home", "projectHome"} or (
            bool(packet.get("degraded")) or str(packet.get("status") or "") == "degraded"
        )
        default_score = 0.35 if is_fallback else 1.0
        score = packet.get("score")
        try:
            score_f = float(score) if score is not None else default_score
        except (TypeError, ValueError):
            score_f = default_score
        if is_fallback:
            score_f = min(score_f, 0.5)
        source_label = "project_file_fallback" if is_fallback else "packet_cache"
        title = str(packet.get("title") or "")
        summary = str(packet.get("summary") or "")
        name_from_title = title.split(":", 1)[-1].strip() if ":" in title else title.strip()

        entity_ids = list(packet.get("entity_ids") or packet.get("entityIds") or [])
        for prov in packet.get("provenance") or []:
            text = str(prov)
            if text.startswith("entity:") and text[7:] not in entity_ids:
                entity_ids.append(text[7:])

        for eid in entity_ids:
            eid_s = str(eid or "").strip()
            if not eid_s or eid_s in seen:
                continue
            seen.add(eid_s)
            entity_type = str(packet.get("entity_type") or packet.get("entityType") or "")
            mirrored.append(
                {
                    "result_type": "entity",
                    "entity": {
                        "id": eid_s,
                        "name": name_from_title or eid_s,
                        "entity_type": entity_type,
                        "type": entity_type,
                        "summary": summary or None,
                    },
                    "score": score_f,
                    "score_breakdown": {"packet_cache": score_f},
                    "source": source_label,
                }
            )

        episode_ids = list(packet.get("episode_ids") or packet.get("episodeIds") or [])
        for prov in packet.get("provenance") or []:
            text = str(prov)
            if text.startswith("episode:") and text[8:] not in episode_ids:
                episode_ids.append(text[8:])

        for epid in episode_ids:
            epid_s = str(epid or "").strip()
            key = f"episode:{epid_s}"
            if not epid_s or key in seen:
                continue
            seen.add(key)
            mirrored.append(
                {
                    "result_type": "episode",
                    "episode": {
                        "id": epid_s,
                        "content": summary or title or epid_s,
                        "source": source_label,
                        "created_at": packet.get("created_at") or packet.get("createdAt"),
                    },
                    "score": score_f,
                    "score_breakdown": {"packet_cache": score_f},
                    "source": source_label,
                }
            )

        # Title-only packet with no entity/episode ids still needs a hit row
        # so naive clients do not see an empty results array.
        if not entity_ids and not episode_ids and (title or summary):
            synthetic = f"packet:{title}:{summary}"
            if synthetic not in seen:
                seen.add(synthetic)
                mirrored.append(
                    {
                        "result_type": "entity",
                        "entity": {
                            "id": synthetic,
                            "name": name_from_title or title or "memory",
                            "entity_type": str(
                                packet.get("packet_type") or packet.get("packetType") or "packet"
                            ),
                            "type": str(
                                packet.get("packet_type") or packet.get("packetType") or "packet"
                            ),
                            "summary": summary or None,
                        },
                        "score": score_f,
                        "score_breakdown": {"packet_cache": score_f},
                        "source": source_label,
                    }
                )

    return mirrored


def _mirror_empty_results_from_packets(
    results: Sequence[Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], bool]:
    """When packets have content and results are empty, mirror packet hits."""
    result_list = list(results)
    packet_list = list(packets)
    if result_list or not packet_list:
        return result_list, False
    mirrored = results_from_packets(packet_list)
    if not mirrored:
        return result_list, False
    return mirrored, True


def present_api_recall_response(
    *,
    query: str,
    results: Sequence[Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Format the REST explicit-recall response with shared lifecycle semantics."""
    packet_payloads = list(packets)
    effective_results, mirrored = _mirror_empty_results_from_packets(results, packet_payloads)
    items = present_api_recall_items(effective_results)
    contract = recall_response_contract(
        query=query,
        result_count=len(items),
        packet_count=len(packet_payloads),
    )
    payload: dict[str, Any] = {
        "operation": contract["operation"],
        "lifecycle": {
            "stage": contract["lifecycle_stage"],
            "recallMode": contract["recall_mode"],
            "resultCount": contract["result_count"],
            "packetCount": contract["packet_count"],
        },
        "items": items,
        "packets": packet_payloads,
        "query": contract["query"],
    }
    if mirrored:
        payload["resultsSource"] = "packets"
        payload["packetsAuthoritative"] = True
        payload["resultsNote"] = (
            "items mirrored from packets (cache hit); packets are authoritative"
        )
    return payload


def present_mcp_recall_response(
    *,
    query: str,
    results: Sequence[Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Format the MCP explicit-recall response with shared lifecycle semantics."""
    packet_payloads = list(packets)
    effective_results, mirrored = _mirror_empty_results_from_packets(results, packet_payloads)
    result_payloads = list(effective_results)
    contract = recall_response_contract(
        query=query,
        result_count=len(result_payloads),
        packet_count=len(packet_payloads),
    )
    payload: dict[str, Any] = {
        "operation": contract["operation"],
        "lifecycle": {
            "stage": contract["lifecycle_stage"],
            "recall_mode": contract["recall_mode"],
            "result_count": contract["result_count"],
            "packet_count": contract["packet_count"],
        },
        "packets": packet_payloads,
        "query": contract["query"],
        "results": result_payloads,
        "total_candidates": contract["result_count"],
    }
    if mirrored:
        payload["results_source"] = "packets"
        payload["packets_authoritative"] = True
        payload["results_note"] = (
            "results mirrored from packets (cache hit); packets are authoritative"
        )
    return payload


async def present_mcp_recall_item(
    result: Mapping[str, Any],
    *,
    resolve_entity_name: ResolveEntityName,
    get_access_count: GetAccessCount,
) -> dict[str, Any]:
    """Format one recall result for the MCP snake_case contract."""
    item = recall_contract_item(result)
    result_type = item["result_type"]
    bd = _breakdown(result)

    if result_type == "episode":
        return {
            "result_type": "episode",
            "episode_id": item["episode_id"],
            "content": item["content"],
            "source": item["source"],
            "created_at": item["created_at"],
            "score": round(item["score"], 4),
            "relevance_confidence": round(bd.get("relevance_confidence", 0.0) or 0.0, 4),
            "score_breakdown": _rounded_numeric_breakdown(result),
            "linked_entities": item["linked_entities"],
        }

    if result_type == "cue_episode":
        return {
            "result_type": "cue_episode",
            "cue_text": item["cue_text"],
            "supporting_spans": item["supporting_spans"] or [],
            "projection_state": item["projection_state"],
            "route_reason": item["route_reason"],
            "episode_id": item["episode_id"],
            "source": item["source"],
            "created_at": item["created_at"],
            "hit_count": item["hit_count"],
            "surfaced_count": item["surfaced_count"],
            "selected_count": item["selected_count"],
            "used_count": item["used_count"],
            "near_miss_count": item["near_miss_count"],
            "policy_score": item["policy_score"],
            "last_feedback_at": item["last_feedback_at"],
            "last_projected_at": item["last_projected_at"],
            "score": round(item["score"], 4),
            "relevance_confidence": round(bd.get("relevance_confidence", 0.0) or 0.0, 4),
            "score_breakdown": _rounded_numeric_breakdown(result),
        }

    related_facts = []
    for rel in item["relationships"]:
        if not isinstance(rel, Mapping):
            continue
        source_id = str(rel.get("source_id", ""))
        target_id = str(rel.get("target_id", ""))
        related_facts.append(
            {
                "subject": await resolve_entity_name(source_id),
                "predicate": rel.get("predicate"),
                "object": await resolve_entity_name(target_id),
                "polarity": rel.get("polarity", "positive"),
            }
        )

    entity_id = str(item["entity_id"] or "")
    return {
        "result_type": "entity",
        "entity_id": entity_id,
        "entity": item["entity_name"],
        "entity_type": item["entity_type"],
        "summary": item["summary"],
        "composite_score": round(item["score"], 4),
        "relevance_confidence": round(bd.get("relevance_confidence", 0.0) or 0.0, 4),
        "score_breakdown": _rounded_numeric_breakdown(result),
        "related_facts": related_facts,
        "access_count": await get_access_count(entity_id),
    }


async def present_mcp_recall_items(
    results: Sequence[Mapping[str, Any]],
    *,
    resolve_entity_name: ResolveEntityName,
    get_access_count: GetAccessCount,
) -> list[dict[str, Any]]:
    return [
        await present_mcp_recall_item(
            result,
            resolve_entity_name=resolve_entity_name,
            get_access_count=get_access_count,
        )
        for result in results
    ]


def present_chat_recall_item(result: Mapping[str, Any]) -> dict[str, Any]:
    """Format one recall result for knowledge-chat tool output."""
    item = recall_contract_item(result)
    result_type = item["result_type"]

    if result_type == "episode":
        return {
            "type": "episode",
            "content": str(item["content"])[:300],
            "source": item["source"],
            "score": round(item["score"], 3),
        }

    if result_type == "cue_episode":
        return {
            "type": "cue_episode",
            "cueText": str(item["cue_text"] or "")[:240],
            "supportingSpans": (item["supporting_spans"] or [])[:2],
            "projectionState": item["projection_state"],
            "policyScore": item["policy_score"],
            "episodeId": item["episode_id"],
            "source": item["source"],
            "score": round(item["score"], 3),
        }

    bd = item["score_breakdown"]
    return {
        "type": "entity",
        "name": item["entity_name"],
        "entityType": item["entity_type"],
        "summary": item["summary"],
        "id": item["entity_id"] or "",
        "score": round(item["score"], 3),
        "activation": round(bd.get("activation", 0) or 0, 3),
        "relationships": [
            {
                "predicate": rel.get("predicate"),
                "target": rel.get("target_name", rel.get("target_id", "")),
                "source": rel.get("source_name", rel.get("source_id", "")),
                "polarity": rel.get("polarity", "positive"),
            }
            for rel in item["relationships"][:10]
            if isinstance(rel, Mapping)
        ],
    }


def present_chat_recall_items(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [present_chat_recall_item(result) for result in results]
