"""Spreading activation dispatcher and seed identification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from engram.activation.engine import compute_activation
from engram.activation.strategy import create_strategy
from engram.config import ActivationConfig
from engram.models.activation import ActivationState

if TYPE_CHECKING:
    from engram.activation.context_gate import ContextGate
    from engram.retrieval.working_memory import WorkingMemoryBuffer


def identify_seeds(
    candidates: list[tuple[str, float]],  # (node_id, semantic_score)
    activation_states: dict[str, ActivationState],
    now: float,
    cfg: ActivationConfig,
    temporal_mode: bool = False,
) -> list[tuple[str, float]]:  # (node_id, initial_energy)
    """Identify seed nodes and assign initial energy for spreading.

    Seeds are candidates with semantic_similarity >= seed_threshold.
    Energy = semantic * max(activation, 0.15).

    When ``temporal_mode=True`` (for recency queries), the seed threshold
    is ignored and energy is based purely on activation, allowing
    activation-only candidates (sem_sim=0.0) to become seeds.
    """
    threshold = 0.0 if temporal_mode else cfg.seed_threshold
    seeds = []
    for node_id, sem_sim in candidates:
        if sem_sim >= threshold or temporal_mode:
            state = activation_states.get(node_id)
            if state and state.access_history:
                act = compute_activation(state.access_history, now, cfg)
            else:
                act = 0.0
            if temporal_mode:
                energy = max(act, 0.15)
            else:
                energy = sem_sim * max(act, 0.15)
            seeds.append((node_id, energy))
    return seeds


def identify_actr_seeds(
    working_memory: WorkingMemoryBuffer,
    now: float,
    cfg: ActivationConfig,
) -> list[tuple[str, float]]:
    """Identify ACT-R spreading seeds from working memory.

    Filters to entity-type items only (episodes have no graph edges),
    sorts by recency, and caps at ``actr_max_sources``.

    Returns ``[(item_id, 1.0)]`` — energy is a placeholder since
    ACTRStrategy computes W_j internally.
    """
    candidates = working_memory.get_candidates(now)
    # Filter to entities only
    entities = [
        (item_id, recency) for item_id, recency, item_type in candidates if item_type == "entity"
    ]
    # Sort by recency descending (most recent first)
    entities.sort(key=lambda x: x[1], reverse=True)
    # Cap at Miller's number
    entities = entities[: cfg.actr_max_sources]
    return [(item_id, 1.0) for item_id, _recency in entities]


async def spread_activation(
    seed_nodes: list[tuple[str, float]],  # (node_id, initial_energy)
    neighbor_provider,  # has get_active_neighbors_with_weights()
    cfg: ActivationConfig,
    group_id: str | None = None,
    community_store=None,
    context_gate: ContextGate | None = None,
    seed_entity_types: dict[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    """Spread activation from seed nodes through the graph.

    Dispatches to the configured strategy (BFS or PPR).

    Returns (bonuses, hop_distances):
      - bonuses: {node_id: spreading_bonus} for all reached nodes
      - hop_distances: {node_id: min_hops_from_seed}
    """
    if cfg.community_spreading_enabled and community_store is not None and group_id:
        seed_ids = [nid for nid, _ in seed_nodes]
        await community_store.ensure_fresh(
            group_id,
            neighbor_provider,
            entity_ids=seed_ids,
        )

    strategy = create_strategy(cfg.spreading_strategy)
    return await strategy.spread(
        seed_nodes,
        neighbor_provider,
        cfg,
        group_id=group_id,
        community_store=community_store,
        context_gate=context_gate,
        seed_entity_types=seed_entity_types,
    )
