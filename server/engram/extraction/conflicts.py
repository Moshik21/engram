"""Exclusive predicate definitions for conflict detection."""

from __future__ import annotations

# Predicates where only one active target is valid at a time per source entity.
# e.g., a person can only LIVE_IN one place at a time.
EXCLUSIVE_PREDICATES: set[str] = {
    "LIVES_IN",
    "WORKS_AT",
    "MARRIED_TO",
    "BASED_IN",
    "CEO_OF",
    "CTO_OF",
    "EMPLOYED_BY",
    "RESIDES_IN",
    "LOCATED_IN",
    "HEADQUARTERED_IN",
}


def is_exclusive_predicate(predicate: str) -> bool:
    """Check if a predicate is exclusive (only one active target allowed)."""
    return predicate.upper() in EXCLUSIVE_PREDICATES
