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


def filter_current_state_results(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer entity state over historical episodes for current-state queries."""
    if not query_prefers_current_state(query):
        return results
    if not any(result.get("result_type") == "entity" for result in results):
        return results
    return [
        result
        for result in results
        if result.get("result_type") not in {"episode", "cue_episode"}
    ]
