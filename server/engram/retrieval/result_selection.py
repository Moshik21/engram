"""Recall result selection helpers."""

from __future__ import annotations

import re
from typing import Any

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
        match.group(0)
        for match in re.finditer(r"[a-z]+", name.lower())
        if len(match.group(0)) >= 4
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
        result
        for result in results
        if result.get("result_type") not in {"episode", "cue_episode"}
    ]
