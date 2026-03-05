"""Multi-signal deterministic scorer for inferred edge validation.

Replaces LLM judge with a weighted ensemble of:
  1. Embedding coherence (0.30)
  2. Type compatibility (0.20)
  3. Statistical confidence (0.20)
  4. Ubiquity penalty (0.15)
  5. Graph structural plausibility (0.10)
  6. Graph embedding similarity (0.05)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain groups (mirrored from config defaults)
# ---------------------------------------------------------------------------

DEFAULT_DOMAIN_GROUPS: dict[str, list[str]] = {
    "personal": ["Person", "Event", "Emotion", "Goal", "Preference", "Habit", "Intention"],
    "technical": ["Technology", "Software", "Project"],
    "creative": ["CreativeWork", "Article"],
    "knowledge": ["Concept"],
    "health": ["HealthCondition", "BodyPart"],
    "spatial": ["Organization", "Location"],
}

# Cross-domain compatibility for MENTIONED_WITH relationships
DOMAIN_COMPATIBILITY: dict[tuple[str, str], float] = {
    # Same domain: always compatible
    ("personal", "personal"): 1.0,
    ("technical", "technical"): 1.0,
    ("knowledge", "knowledge"): 1.0,
    ("creative", "creative"): 1.0,
    ("health", "health"): 1.0,
    ("spatial", "spatial"): 1.0,
    # Cross-domain meaningful pairs (keys are alphabetically sorted)
    ("personal", "technical"): 0.8,
    ("personal", "spatial"): 0.7,
    ("knowledge", "personal"): 0.7,
    ("health", "personal"): 0.8,
    ("creative", "personal"): 0.7,
    ("knowledge", "technical"): 0.9,
    ("spatial", "technical"): 0.5,
    ("creative", "technical"): 0.6,
    ("creative", "knowledge"): 0.7,
    # Low compatibility (noise-prone)
    ("health", "spatial"): 0.3,
    ("health", "knowledge"): 0.4,
    ("knowledge", "spatial"): 0.3,
    ("creative", "health"): 0.3,
}


# ---------------------------------------------------------------------------
# Signal 1: Type compatibility
# ---------------------------------------------------------------------------


def _entity_type_to_domain(
    entity_type: str,
    domain_groups: dict[str, list[str]] | None = None,
) -> str:
    groups = domain_groups or DEFAULT_DOMAIN_GROUPS
    for domain, types in groups.items():
        if entity_type in types:
            return domain
    return "knowledge"  # default for unmapped types


def compute_type_compatibility(
    type_a: str,
    type_b: str,
    domain_groups: dict[str, list[str]] | None = None,
) -> float:
    """Return compatibility score [0, 1] for an entity-type pair."""
    domain_a = _entity_type_to_domain(type_a, domain_groups)
    domain_b = _entity_type_to_domain(type_b, domain_groups)
    pair = tuple(sorted([domain_a, domain_b]))
    return DOMAIN_COMPATIBILITY.get(pair, 0.5)


# ---------------------------------------------------------------------------
# Signal 2: Ubiquity penalty
# ---------------------------------------------------------------------------


def compute_ubiquity_score(
    ep_count_a: int,
    ep_count_b: int,
    co_occurrence_count: int,
    total_episodes: int,
) -> float:
    """Penalize ubiquitous entities that co-occur by chance."""
    if total_episodes <= 0:
        return 0.5

    freq_a = ep_count_a / total_episodes
    freq_b = ep_count_b / total_episodes
    max_freq = max(freq_a, freq_b)

    # Jaccard on episodes
    union_eps = ep_count_a + ep_count_b - co_occurrence_count
    jaccard = co_occurrence_count / max(union_eps, 1)

    if max_freq > 0.5:
        return 0.2  # very likely noise
    elif max_freq > 0.3:
        return 0.5 * jaccard + 0.3
    else:
        return 0.7 + 0.3 * jaccard


# ---------------------------------------------------------------------------
# Signal 3: Structural plausibility (shared neighbors / triangle closure)
# ---------------------------------------------------------------------------


async def compute_structural_score(
    entity_a_id: str,
    entity_b_id: str,
    graph_store: object,
    group_id: str,
) -> float:
    """Score based on shared graph neighbors (triangle closure)."""
    try:
        neighbors_a = await graph_store.get_active_neighbors_with_weights(  # type: ignore[union-attr]
            entity_a_id, group_id,
        )
        neighbors_b = await graph_store.get_active_neighbors_with_weights(  # type: ignore[union-attr]
            entity_b_id, group_id,
        )
    except Exception:
        return 0.5

    nid_a = {nid for nid, _, _, _ in neighbors_a}
    nid_b = {nid for nid, _, _, _ in neighbors_b}
    union = nid_a | nid_b

    if not union:
        return 0.5  # both isolated, neutral

    shared = nid_a & nid_b
    if shared:
        return min(1.0, 0.6 + 0.4 * len(shared) / max(1, len(union)))
    else:
        return 0.3  # connected but no overlap


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


async def score_infer_pair(
    entity_a_id: str,
    entity_b_id: str,
    entity_a_name: str,  # noqa: ARG001
    entity_b_name: str,  # noqa: ARG001
    entity_a_type: str,
    entity_b_type: str,
    co_occurrence_count: int,
    pmi_confidence: float,
    ep_count_a: int,
    ep_count_b: int,
    total_episodes: int,
    search_index: object,
    graph_store: object,
    group_id: str,
    domain_groups: dict[str, list[str]] | None = None,
    approve_threshold: float = 0.65,
    reject_threshold: float = 0.40,
) -> tuple[str, float, dict[str, float]]:
    """Score an inferred edge using multi-signal ensemble.

    Returns ``(verdict, score, signal_breakdown)`` where verdict is one of
    ``"approved"``, ``"rejected"``, or ``"uncertain"``.
    """
    # Signal 1: Embedding coherence (weight 0.30)
    emb_score = 0.5  # neutral default
    try:
        embeddings = await search_index.get_entity_embeddings(  # type: ignore[union-attr]
            [entity_a_id, entity_b_id],
            group_id=group_id,
        )
        if entity_a_id in embeddings and entity_b_id in embeddings:
            vec_a = np.array(embeddings[entity_a_id], dtype=np.float32)
            vec_b = np.array(embeddings[entity_b_id], dtype=np.float32)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a > 0 and norm_b > 0:
                emb_score = max(0.0, float(np.dot(vec_a, vec_b) / (norm_a * norm_b)))
    except Exception:
        pass

    # Signal 2: Type compatibility (weight 0.20)
    type_score = compute_type_compatibility(entity_a_type, entity_b_type, domain_groups)

    # Signal 3: Statistical confidence (weight 0.20)
    stat_score = pmi_confidence
    if co_occurrence_count >= 10:
        stat_score = min(1.0, stat_score + 0.15)
    elif co_occurrence_count >= 7:
        stat_score = min(1.0, stat_score + 0.08)

    # Signal 4: Ubiquity penalty (weight 0.15)
    ubiquity_score = compute_ubiquity_score(
        ep_count_a, ep_count_b, co_occurrence_count, total_episodes,
    )

    # Signal 5: Graph structural plausibility (weight 0.10)
    structural_score = await compute_structural_score(
        entity_a_id, entity_b_id, graph_store, group_id,
    )

    # Signal 6: Graph embedding similarity (weight 0.05)
    graph_sim = 0.5  # neutral default
    try:
        graph_embs = await search_index.get_graph_embeddings(  # type: ignore[union-attr]
            [entity_a_id, entity_b_id],
            method="node2vec",
            group_id=group_id,
        )
        if entity_a_id in graph_embs and entity_b_id in graph_embs:
            gv_a = np.array(graph_embs[entity_a_id], dtype=np.float32)
            gv_b = np.array(graph_embs[entity_b_id], dtype=np.float32)
            gn_a = np.linalg.norm(gv_a)
            gn_b = np.linalg.norm(gv_b)
            if gn_a > 0 and gn_b > 0:
                graph_sim = max(0.0, float(np.dot(gv_a, gv_b) / (gn_a * gn_b)))
    except Exception:
        pass

    # Composite score
    score = (
        0.30 * emb_score
        + 0.20 * type_score
        + 0.20 * stat_score
        + 0.15 * ubiquity_score
        + 0.10 * structural_score
        + 0.05 * graph_sim
    )

    signals = {
        "embedding": round(emb_score, 4),
        "type_compat": round(type_score, 4),
        "statistical": round(stat_score, 4),
        "ubiquity": round(ubiquity_score, 4),
        "structural": round(structural_score, 4),
        "graph_emb": round(graph_sim, 4),
    }

    if score >= approve_threshold:
        verdict = "approved"
    elif score < reject_threshold:
        verdict = "rejected"
    else:
        verdict = "uncertain"  # keep edge but downgrade confidence

    return verdict, round(score, 4), signals
