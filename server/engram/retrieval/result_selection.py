"""Recall result selection helpers."""

from __future__ import annotations

import re
from typing import Any

from engram.extraction.promotion import (
    durable_result_boost,
    is_durable_recall_entity_type,
)

_CURRENT_STATE_TOKENS = {"now", "current", "currently"}


def query_prefers_current_state(query: str) -> bool:
    """Detect narrow queries asking for the current/latest state."""
    tokens = {match.group(0) for match in re.finditer(r"[a-z]+", query.lower())}
    if not tokens:
        return False
    return bool(tokens & _CURRENT_STATE_TOKENS)


def _query_relevant_entity(result: dict[str, Any], query_tokens: set[str]) -> bool:
    """True when an entity result is named after something the query mentions.

    A significant name token (len >= 4, excluding the current-state cue words)
    overlapping the query means the surfaced entity is about the query's subject
    and can plausibly carry its current state.
    """
    if result.get("result_type") != "entity":
        return False
    entity = result.get("entity") or {}
    name = entity.get("name") or ""
    name_tokens = {
        match.group(0) for match in re.finditer(r"[a-z]+", name.lower()) if len(match.group(0)) >= 4
    }
    return bool((name_tokens & query_tokens) - _CURRENT_STATE_TOKENS)


def filter_current_state_results(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer entity state over historical episodes for current-state queries.

    Episodes are suppressed ONLY when a surfaced entity is actually relevant to
    the query (its name overlaps the query subject) and can therefore plausibly
    carry the current value. When no surfaced entity matches the subject, the
    entity layer cannot answer the current-state question, so the historical
    episodes are the only source of the answer and are kept. Dropping episodes
    unconditionally (the old behavior) discarded the answer-bearing episode
    whenever extraction left the relevant entity incomplete or recall surfaced
    unrelated entities — making graph-on strictly worse than episode-only.
    """
    if not query_prefers_current_state(query):
        return results
    query_tokens = {match.group(0) for match in re.finditer(r"[a-z]+", query.lower())}
    if not any(_query_relevant_entity(result, query_tokens) for result in results):
        return results
    return [
        result for result in results if result.get("result_type") not in {"episode", "cue_episode"}
    ]


def prefer_durable_facts(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-rank recall results so durable graph facts beat session recap/cues.

    Primary success metric: a cold session should surface Decision / Preference /
    Person entities before api_auto_observe transcript packets.
    """
    if not results:
        return results

    def _sort_key(item: tuple[int, dict[str, Any]]) -> tuple:
        index, result = item
        result_type = str(result.get("result_type") or "")
        entity = result.get("entity") or {}
        entity_type = str(entity.get("type") or entity.get("entity_type") or "")
        base_score = float(result.get("score") or 0.0)
        durable_boost = durable_result_boost(entity_type) if result_type == "entity" else 0.0

        # Type priority: durable entity > other entity > episode > cue
        if result_type == "entity" and is_durable_recall_entity_type(entity_type):
            type_rank = 3
        elif result_type == "entity":
            type_rank = 2
        elif result_type == "episode":
            type_rank = 1
        elif result_type == "cue_episode":
            type_rank = 0
        else:
            type_rank = 1

        # Recency-only transcript dumps sort last when durable facts exist.
        source = str(result.get("source") or "")
        trust = result.get("trust") if isinstance(result.get("trust"), dict) else {}
        trust_source = str(trust.get("source") or "")
        is_auto_recap = (
            source.startswith("auto:")
            or trust_source in {"api_auto_observe", "mcp_observe", "auto:prompt"}
            or result_type == "cue_episode"
        )
        recap_penalty = 1.0 if is_auto_recap and durable_boost == 0.0 else 0.0

        return (type_rank, base_score + durable_boost - recap_penalty, -index)

    indexed = list(enumerate(results))
    indexed.sort(key=_sort_key, reverse=True)
    return [result for _index, result in indexed]
