"""Inhibitory spreading: predicate suppression + lateral inhibition."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from engram.config import ActivationConfig
from engram.extraction.conflicts import CONTRADICTORY_PAIRS

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def apply_predicate_inhibition(
    bonuses: dict[str, float],
    seed_node_ids: set[str],
    graph_store,
    group_id: str,
    cfg: ActivationConfig,
    relationships: list | None = None,
) -> dict[str, float]:
    """Suppress bonuses for entities on the weaker side of contradictory predicates.

    For each contradictory pair (e.g. LIKES/DISLIKES), if both predicates exist
    from seed entities, suppress the weaker group's bonuses.

    This is synchronous -- works on pre-fetched relationship data.
    """
    if not cfg.inhibition_predicate_suppression or not relationships:
        return bonuses

    # Group relationships by predicate
    pred_groups: dict[str, list[tuple[str, str, float]]] = {}
    for rel in relationships:
        # rel is expected to be (source_id, target_id, predicate, weight) or similar
        if len(rel) >= 4:
            src, tgt, pred, weight = rel[0], rel[1], rel[2], rel[3]
        elif len(rel) >= 3:
            src, tgt, pred = rel[0], rel[1], rel[2]
            weight = 1.0
        else:
            continue
        # Only consider relationships from seeds
        if src in seed_node_ids or tgt in seed_node_ids:
            pred_groups.setdefault(pred, []).append((src, tgt, weight))

    # Check contradictory pairs
    for pair in CONTRADICTORY_PAIRS:
        preds = list(pair)
        if len(preds) != 2:
            continue
        p1, p2 = preds[0], preds[1]
        if p1 not in pred_groups or p2 not in pred_groups:
            continue

        # Compute aggregate weight for each group
        w1 = sum(w for _, _, w in pred_groups[p1])
        w2 = sum(w for _, _, w in pred_groups[p2])

        # Suppress the weaker group
        weaker_pred = p1 if w1 < w2 else p2
        weaker_targets = {
            tgt if src in seed_node_ids else src for src, tgt, _ in pred_groups[weaker_pred]
        }

        for nid in weaker_targets:
            if nid in bonuses:
                bonuses[nid] *= 1.0 - cfg.inhibit_strength

    return bonuses


async def apply_lateral_inhibition(
    bonuses: dict[str, float],
    hop_distances: dict[str, int],
    seed_node_ids: set[str],
    search_index,
    group_id: str,
    cfg: ActivationConfig,
) -> dict[str, float]:
    """Inhibit semantically similar but graph-disconnected entities.

    Graph-connected entities get reduced inhibition (protected by topology).
    Graph-disconnected entities with high semantic similarity to seeds get
    full inhibition.
    """
    if not seed_node_ids or not bonuses:
        return bonuses

    # Get seed entity embeddings
    seed_list = list(seed_node_ids)[: cfg.inhibit_max_seed_anchors]

    if not hasattr(search_index, "get_entity_embeddings"):
        return bonuses

    try:
        seed_embs = await search_index.get_entity_embeddings(
            seed_list,
            group_id=group_id,
        )
    except Exception:
        return bonuses

    if not seed_embs:
        return bonuses

    # Get candidate embeddings
    candidate_ids = [nid for nid in bonuses if nid not in seed_node_ids]
    if not candidate_ids:
        return bonuses

    try:
        cand_embs = await search_index.get_entity_embeddings(
            candidate_ids,
            group_id=group_id,
        )
    except Exception:
        return bonuses

    if not cand_embs:
        return bonuses

    # Compute average seed embedding
    import numpy as np

    seed_vecs = list(seed_embs.values())
    avg_seed = np.mean(seed_vecs, axis=0)
    avg_seed_norm = np.linalg.norm(avg_seed)
    if avg_seed_norm == 0:
        return bonuses
    avg_seed = avg_seed / avg_seed_norm

    for nid, emb in cand_embs.items():
        if nid not in bonuses or bonuses[nid] <= 0:
            continue

        emb_arr = np.array(emb)
        emb_norm = np.linalg.norm(emb_arr)
        if emb_norm == 0:
            continue

        cos_sim = float(np.dot(avg_seed, emb_arr / emb_norm))
        if cos_sim < cfg.inhibit_similarity_threshold:
            continue

        # Graph-disconnected: full inhibition
        if nid not in hop_distances:
            inhibition = cfg.inhibit_strength * cos_sim * 1.0
        else:
            # Graph-connected: reduced inhibition based on distance
            hops = hop_distances[nid]
            inhibition = cfg.inhibit_strength * cos_sim * (1.0 - 0.5**hops)

        bonuses[nid] = max(0.0, bonuses[nid] - inhibition)

    return bonuses


async def apply_inhibition(
    bonuses: dict[str, float],
    hop_distances: dict[str, int],
    seed_node_ids: set[str],
    graph_store,
    search_index,
    group_id: str,
    cfg: ActivationConfig,
    relationships: list | None = None,
) -> dict[str, float]:
    """Orchestrator: predicate suppression, then lateral inhibition."""
    if not cfg.inhibitory_spreading_enabled:
        return bonuses

    # Predicate suppression (synchronous)
    bonuses = apply_predicate_inhibition(
        bonuses,
        seed_node_ids,
        graph_store,
        group_id,
        cfg,
        relationships=relationships,
    )

    # Lateral inhibition (async -- needs embeddings)
    bonuses = await apply_lateral_inhibition(
        bonuses,
        hop_distances,
        seed_node_ids,
        search_index,
        group_id,
        cfg,
    )

    return bonuses
