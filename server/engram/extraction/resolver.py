"""Entity deduplication and name resolution with fuzzy matching."""

from __future__ import annotations

from thefuzz import fuzz

from engram.models.entity import Entity

FUZZY_MATCH_THRESHOLD = 85  # 0-100 scale (thefuzz uses integers)


def normalize_name(name: str) -> str:
    """Normalize entity name for dedup comparison."""
    return name.strip().lower().replace("_", " ").replace("-", " ")


def compute_similarity(name_a: str, name_b: str) -> float:
    """Compute similarity between two entity names (0.0 to 1.0).

    Uses multiple strategies and returns the max score:
    - Exact normalized match → 1.0
    - Token sort ratio (handles word reordering)
    - Partial ratio * 0.9 (handles substring containment)
    """
    norm_a = normalize_name(name_a)
    norm_b = normalize_name(name_b)

    # Exact normalized match
    if norm_a == norm_b:
        return 1.0

    # Token sort ratio — handles word reordering
    # e.g., "ACT-R Spreading Activation" ↔ "Spreading Activation ACT-R"
    token_sort = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0

    # Partial ratio — handles substring containment
    # e.g., "ACT-R" in "ACT-R Spreading Activation"
    partial = fuzz.partial_ratio(norm_a, norm_b) / 100.0 * 0.9

    return max(token_sort, partial)


async def resolve_entity(
    name: str,
    entity_type: str,
    existing_entities: list[Entity],
) -> Entity | None:
    """Find an existing entity that matches the given name using fuzzy matching.

    Returns the matching entity or None if no match found.
    Scores all existing entities and picks the best above threshold.
    Same-type matches get a 0.05 boost.
    """
    best_match: Entity | None = None
    best_score = 0.0

    for entity in existing_entities:
        score = compute_similarity(name, entity.name)

        # Boost same-type matches
        if entity.entity_type == entity_type:
            score = min(score + 0.05, 1.0)

        if score > best_score:
            best_score = score
            best_match = entity

    threshold = FUZZY_MATCH_THRESHOLD / 100.0
    if best_score >= threshold and best_match is not None:
        return best_match

    return None
