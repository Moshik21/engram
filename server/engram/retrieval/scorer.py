"""Composite retrieval scorer with three orthogonal signals."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from engram.activation.engine import compute_activation
from engram.config import ActivationConfig
from engram.models.activation import ActivationState


@dataclass
class ScoredResult:
    """A scored retrieval result with per-signal breakdown."""

    node_id: str
    score: float
    semantic_similarity: float
    activation: float
    spreading: float
    edge_proximity: float
    exploration_bonus: float = 0.0
    graph_structural: float = 0.0
    emotional_boost: float = 0.0
    state_boost: float = 0.0
    hop_distance: int | None = None
    result_type: str = "entity"
    planner_support: float = 0.0
    planner_intents: list[str] = field(default_factory=list)
    recall_trace: list[dict] = field(default_factory=list)


def _tier_to_decay(mat_tier: str | None, cfg: ActivationConfig) -> float | None:
    """Map memory tier to decay exponent override."""
    if not cfg.memory_maturation_enabled:
        return None
    if mat_tier == "semantic":
        return cfg.decay_exponent_semantic
    if mat_tier == "transitional":
        return (cfg.decay_exponent_episodic + cfg.decay_exponent_semantic) / 2.0
    return None  # use default


def score_candidates(
    candidates: list[tuple[str, float]],  # (node_id, semantic_similarity)
    spreading_bonuses: dict[str, float],
    hop_distances: dict[str, int],
    seed_node_ids: set[str],
    activation_states: dict[str, ActivationState],
    now: float,
    cfg: ActivationConfig,
    conv_fingerprint_sim: dict[str, float] | None = None,
    priming_boosts: dict[str, float] | None = None,
    graph_similarities: dict[str, float] | None = None,
    entity_attributes: dict[str, dict] | None = None,
    state_biases: dict[str, float] | None = None,
) -> list[ScoredResult]:
    """Score and rank candidate nodes.

    score = w_sem * semantic + w_act * activation + w_spread * spreading
            + w_edge * edge_proximity + exploration

    activation = compute_activation(history), clamped [0, 1]
    spreading = spreading_bonus, clamped [0, 1] (independent signal)
    edge_proximity = 1.0 for seeds, 0.5^hops for reached, 0.0 for unreachable
    """
    results = []

    for node_id, sem_sim in candidates:
        # Compute activation lazily from access_history
        state = activation_states.get(node_id)
        # Differential decay: semantic memories decay slower
        decay = None
        if entity_attributes is not None:
            decay = _tier_to_decay(entity_attributes.get(node_id, {}).get("mat_tier"), cfg)
        if state and state.access_history:
            base_act = compute_activation(
                state.access_history,
                now,
                cfg,
                state.consolidated_strength,
                decay,
            )
        else:
            base_act = 0.0

        # Spreading as independent signal, clamped to [0, 1]
        spread = min(1.0, spreading_bonuses.get(node_id, 0.0))

        # Edge proximity
        if node_id in seed_node_ids:
            edge_prox = 1.0
        elif node_id in hop_distances:
            edge_prox = 0.5 ** hop_distances[node_id]
        else:
            edge_prox = 0.0

        # Exploration bonus: smooth novelty (no hard threshold gate)
        access_count = state.access_count if state else 0
        if sem_sim > 0:
            novelty = 1.0 / (1.0 + math.log1p(access_count))
            exploration = cfg.exploration_weight * sem_sim * novelty
        else:
            exploration = 0.0

        # Rediscovery bonus: exponential decay for dormant entities
        if sem_sim > 0 and state and state.access_history and cfg.rediscovery_weight > 0:
            last_access = max(state.access_history)
            days_since = (now - last_access) / 86400.0
            halflife = cfg.rediscovery_halflife_days
            rediscovery = (
                cfg.rediscovery_weight
                * sem_sim
                * (1.0 - math.exp(-math.log(2) * days_since / halflife))
            )
            exploration += rediscovery

        # Conversation context boost
        ctx_boost = 0.0
        if conv_fingerprint_sim is not None:
            ctx_boost = cfg.conv_context_rerank_weight * conv_fingerprint_sim.get(node_id, 0.0)

        # Retrieval priming boost (Wave 3)
        prime_boost = 0.0
        if priming_boosts is not None:
            prime_boost = priming_boosts.get(node_id, 0.0)

        # Graph structural similarity
        graph_sim = graph_similarities.get(node_id, 0.0) if graph_similarities else 0.0

        # Emotional retrieval boost
        emo_boost = 0.0
        if entity_attributes is not None and cfg.emotional_salience_enabled:
            attrs = entity_attributes.get(node_id, {})
            emo_boost = cfg.emotional_retrieval_boost * attrs.get("emo_composite", 0.0)

        # State-dependent retrieval boost
        s_boost = state_biases.get(node_id, 0.0) if state_biases else 0.0

        # Composite score
        score = (
            cfg.weight_semantic * sem_sim
            + cfg.weight_activation * base_act
            + cfg.weight_spreading * spread
            + cfg.weight_edge_proximity * edge_prox
            + cfg.weight_graph_structural * graph_sim
            + exploration
            + ctx_boost
            + prime_boost
            + emo_boost
            + s_boost
        )

        # Hop distance: 0 for seeds, from hop_distances dict, or None
        if node_id in seed_node_ids:
            hop_dist: int | None = 0
        elif node_id in hop_distances:
            hop_dist = hop_distances[node_id]
        else:
            hop_dist = None

        results.append(
            ScoredResult(
                node_id=node_id,
                score=score,
                semantic_similarity=sem_sim,
                activation=base_act,
                spreading=spread,
                edge_proximity=edge_prox,
                exploration_bonus=exploration,
                graph_structural=graph_sim,
                emotional_boost=emo_boost,
                state_boost=s_boost,
                hop_distance=hop_dist,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def score_candidates_thompson(
    candidates: list[tuple[str, float]],
    spreading_bonuses: dict[str, float],
    hop_distances: dict[str, int],
    seed_node_ids: set[str],
    activation_states: dict[str, ActivationState],
    now: float,
    cfg: ActivationConfig,
    rng_seed: int | None = None,
    conv_fingerprint_sim: dict[str, float] | None = None,
    priming_boosts: dict[str, float] | None = None,
    graph_similarities: dict[str, float] | None = None,
    entity_attributes: dict[str, dict] | None = None,
    state_biases: dict[str, float] | None = None,
) -> list[ScoredResult]:
    """Score candidates using Thompson Sampling for exploration.

    Same as score_candidates but replaces the deterministic exploration
    bonus with a sample from Beta(ts_alpha, ts_beta) per entity.
    """
    rng = random.Random(rng_seed)
    results = []

    for node_id, sem_sim in candidates:
        state = activation_states.get(node_id)
        # Differential decay: semantic memories decay slower
        decay = None
        if entity_attributes is not None:
            decay = _tier_to_decay(entity_attributes.get(node_id, {}).get("mat_tier"), cfg)
        if state and state.access_history:
            base_act = compute_activation(
                state.access_history,
                now,
                cfg,
                state.consolidated_strength,
                decay,
            )
        else:
            base_act = 0.0

        # Spreading as independent signal, clamped to [0, 1]
        spread = min(1.0, spreading_bonuses.get(node_id, 0.0))

        if node_id in seed_node_ids:
            edge_prox = 1.0
        elif node_id in hop_distances:
            edge_prox = 0.5 ** hop_distances[node_id]
        else:
            edge_prox = 0.0

        # Thompson Sampling exploration: sample from Beta posterior
        ts_alpha = state.ts_alpha if state else 1.0
        ts_beta_val = state.ts_beta if state else 1.0
        if sem_sim > 0:
            sample = rng.betavariate(ts_alpha, ts_beta_val)
            exploration = cfg.ts_weight * sem_sim * sample
        else:
            exploration = 0.0

        # Rediscovery bonus (same as deterministic scorer)
        if sem_sim > 0 and state and state.access_history and cfg.rediscovery_weight > 0:
            last_access = max(state.access_history)
            days_since = (now - last_access) / 86400.0
            halflife = cfg.rediscovery_halflife_days
            rediscovery = (
                cfg.rediscovery_weight
                * sem_sim
                * (1.0 - math.exp(-math.log(2) * days_since / halflife))
            )
            exploration += rediscovery

        # Conversation context boost
        ctx_boost = 0.0
        if conv_fingerprint_sim is not None:
            ctx_boost = cfg.conv_context_rerank_weight * conv_fingerprint_sim.get(node_id, 0.0)

        # Retrieval priming boost (Wave 3)
        prime_boost = 0.0
        if priming_boosts is not None:
            prime_boost = priming_boosts.get(node_id, 0.0)

        # Graph structural similarity
        graph_sim = graph_similarities.get(node_id, 0.0) if graph_similarities else 0.0

        # Emotional retrieval boost
        emo_boost = 0.0
        if entity_attributes is not None and cfg.emotional_salience_enabled:
            attrs = entity_attributes.get(node_id, {})
            emo_boost = cfg.emotional_retrieval_boost * attrs.get("emo_composite", 0.0)

        # State-dependent retrieval boost
        s_boost = state_biases.get(node_id, 0.0) if state_biases else 0.0

        score = (
            cfg.weight_semantic * sem_sim
            + cfg.weight_activation * base_act
            + cfg.weight_spreading * spread
            + cfg.weight_edge_proximity * edge_prox
            + cfg.weight_graph_structural * graph_sim
            + exploration
            + ctx_boost
            + prime_boost
            + emo_boost
            + s_boost
        )

        if node_id in seed_node_ids:
            hop_dist_t: int | None = 0
        elif node_id in hop_distances:
            hop_dist_t = hop_distances[node_id]
        else:
            hop_dist_t = None

        results.append(
            ScoredResult(
                node_id=node_id,
                score=score,
                semantic_similarity=sem_sim,
                activation=base_act,
                spreading=spread,
                edge_proximity=edge_prox,
                exploration_bonus=exploration,
                graph_structural=graph_sim,
                emotional_boost=emo_boost,
                state_boost=s_boost,
                hop_distance=hop_dist_t,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def extract_near_misses(
    all_scored: list[ScoredResult],
    top_n: int,
    window: int = 5,
) -> list[ScoredResult]:
    """Return candidates just outside top-N (positions [top_n, top_n+window))."""
    if len(all_scored) <= top_n:
        return []
    return all_scored[top_n : top_n + window]
