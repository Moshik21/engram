"""Entity deduplication and name resolution with fuzzy matching."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from rapidfuzz import fuzz

from engram.entity_dedup_policy import policy_aware_similarity
from engram.models.entity import Entity

FUZZY_MATCH_THRESHOLD = 85  # 0-100 scale (rapidfuzz uses integers)


def validate_entity_name(name: str) -> bool:
    """Validate that a name is plausible as an entity name.

    Rejects:
    - Names shorter than 2 characters
    - Names longer than 5 words (likely sentence fragments)
    - All-lowercase names (unless they contain dots/slashes indicating tech tokens)
    """
    stripped = name.strip()
    if len(stripped) < 2:
        return False
    if len(stripped.split()) > 5:
        return False
    # All-lowercase names are not proper nouns — except tech tokens with dots/slashes
    if stripped == stripped.lower() and "." not in stripped and "/" not in stripped:
        return False
    return True


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
    token_sort = cast(float, fuzz.token_sort_ratio(norm_a, norm_b)) / 100.0

    # Partial ratio — handles substring containment
    # e.g., "ACT-R" in "ACT-R Spreading Activation"
    partial = cast(float, fuzz.partial_ratio(norm_a, norm_b)) / 100.0 * 0.9

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
        decision, score = policy_aware_similarity(name, entity.name, compute_similarity)
        if not decision.allowed:
            continue

        # Boost same-type matches
        if entity.entity_type == entity_type and not decision.exact_identifier_match:
            score = min(score + 0.05, 1.0)

        if score > best_score:
            best_score = score
            best_match = entity

    threshold = FUZZY_MATCH_THRESHOLD / 100.0
    if best_score >= threshold and best_match is not None:
        return best_match

    return None


async def resolve_entity_fast(
    name: str,
    entity_type: str,
    get_candidates: Callable[[str, str], Awaitable[list[Entity]]],
    group_id: str,
    session_entities: dict[str, Entity] | None = None,
) -> Entity | None:
    """Resolve entity using indexed candidate retrieval instead of O(N) scan.

    Checks session_entities first (within-episode dedup), then retrieves
    ~30 candidates from the DB and fuzzy-matches only those.
    """
    # Check session cache first (entities created earlier in this episode)
    if session_entities:
        for entity in session_entities.values():
            decision, score = policy_aware_similarity(name, entity.name, compute_similarity)
            if not decision.allowed:
                continue
            if entity.entity_type == entity_type and not decision.exact_identifier_match:
                score = min(score + 0.05, 1.0)
            if score >= FUZZY_MATCH_THRESHOLD / 100.0:
                return entity

    # Retrieve candidates from DB
    candidates = await get_candidates(name, group_id)

    # Run same fuzzy matching logic on candidate set
    best_match: Entity | None = None
    best_score = 0.0

    for entity in candidates:
        decision, score = policy_aware_similarity(name, entity.name, compute_similarity)
        if not decision.allowed:
            continue
        if entity.entity_type == entity_type and not decision.exact_identifier_match:
            score = min(score + 0.05, 1.0)
        if score > best_score:
            best_score = score
            best_match = entity

    threshold = FUZZY_MATCH_THRESHOLD / 100.0
    if best_score >= threshold and best_match is not None:
        return best_match

    return None
