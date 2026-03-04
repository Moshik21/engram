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


# Predicate pairs that contradict each other for the same source-target.
# e.g., a person cannot both LIKE and DISLIKE the same thing.
CONTRADICTORY_PAIRS: set[frozenset[str]] = {
    frozenset({"LIKES", "DISLIKES"}),
    frozenset({"AIMS_FOR", "AVOIDS"}),
}

# Build a lookup dict for fast contradiction checks.
_CONTRADICTS: dict[str, set[str]] = {}
for _pair in CONTRADICTORY_PAIRS:
    _items = list(_pair)
    _CONTRADICTS.setdefault(_items[0], set()).add(_items[1])
    _CONTRADICTS.setdefault(_items[1], set()).add(_items[0])


def is_exclusive_predicate(predicate: str) -> bool:
    """Check if a predicate is exclusive (only one active target allowed)."""
    return predicate.upper() in EXCLUSIVE_PREDICATES


def get_contradictory_predicates(predicate: str) -> set[str]:
    """Return predicates that contradict the given one for the same source-target."""
    return _CONTRADICTS.get(predicate.upper(), set())
