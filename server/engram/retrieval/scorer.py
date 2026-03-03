"""Composite retrieval scorer with three orthogonal signals."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

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
    result_type: str = "entity"


def score_candidates(
    candidates: list[tuple[str, float]],  # (node_id, semantic_similarity)
    spreading_bonuses: dict[str, float],
    hop_distances: dict[str, int],
    seed_node_ids: set[str],
    activation_states: dict[str, ActivationState],
    now: float,
    cfg: ActivationConfig,
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
        if state and state.access_history:
            base_act = compute_activation(state.access_history, now, cfg)
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
        if (
            sem_sim > 0
            and state
            and state.access_history
            and cfg.rediscovery_weight > 0
        ):
            last_access = max(state.access_history)
            days_since = (now - last_access) / 86400.0
            halflife = cfg.rediscovery_halflife_days
            rediscovery = (
                cfg.rediscovery_weight
                * sem_sim
                * (1.0 - math.exp(-math.log(2) * days_since / halflife))
            )
            exploration += rediscovery

        # Composite score with 4 weighted signals
        score = (
            cfg.weight_semantic * sem_sim
            + cfg.weight_activation * base_act
            + cfg.weight_spreading * spread
            + cfg.weight_edge_proximity * edge_prox
            + exploration
        )

        results.append(
            ScoredResult(
                node_id=node_id,
                score=score,
                semantic_similarity=sem_sim,
                activation=base_act,
                spreading=spread,
                edge_proximity=edge_prox,
                exploration_bonus=exploration,
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
) -> list[ScoredResult]:
    """Score candidates using Thompson Sampling for exploration.

    Same as score_candidates but replaces the deterministic exploration
    bonus with a sample from Beta(ts_alpha, ts_beta) per entity.
    """
    rng = random.Random(rng_seed)
    results = []

    for node_id, sem_sim in candidates:
        state = activation_states.get(node_id)
        if state and state.access_history:
            base_act = compute_activation(state.access_history, now, cfg)
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
        if (
            sem_sim > 0
            and state
            and state.access_history
            and cfg.rediscovery_weight > 0
        ):
            last_access = max(state.access_history)
            days_since = (now - last_access) / 86400.0
            halflife = cfg.rediscovery_halflife_days
            rediscovery = (
                cfg.rediscovery_weight
                * sem_sim
                * (1.0 - math.exp(-math.log(2) * days_since / halflife))
            )
            exploration += rediscovery

        score = (
            cfg.weight_semantic * sem_sim
            + cfg.weight_activation * base_act
            + cfg.weight_spreading * spread
            + cfg.weight_edge_proximity * edge_prox
            + exploration
        )

        results.append(
            ScoredResult(
                node_id=node_id,
                score=score,
                semantic_similarity=sem_sim,
                activation=base_act,
                spreading=spread,
                edge_proximity=edge_prox,
                exploration_bonus=exploration,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results
