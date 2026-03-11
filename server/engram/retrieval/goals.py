"""Goal-relevance gating: identify active goals and prime their neighborhoods."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from engram.config import ActivationConfig

logger = logging.getLogger(__name__)


@dataclass
class ActiveGoal:
    """An active goal entity with its neighborhood."""

    entity_id: str
    name: str
    activation: float
    neighbor_ids: list[str] = field(default_factory=list)


class GoalPrimingCache:
    """TTL cache for active goals to avoid re-querying every retrieval."""

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, list[ActiveGoal]]] = {}

    def get(self, group_id: str) -> list[ActiveGoal] | None:
        if group_id not in self._cache:
            return None
        ts, goals = self._cache[group_id]
        if time.time() - ts > self._ttl:
            del self._cache[group_id]
            return None
        return goals

    def set(self, group_id: str, goals: list[ActiveGoal]) -> None:
        self._cache[group_id] = (time.time(), goals)

    def invalidate(self, group_id: str | None = None) -> None:
        if group_id:
            self._cache.pop(group_id, None)
        else:
            self._cache.clear()


async def identify_active_goals(
    graph_store,
    activation_store,
    group_id: str,
    cfg: ActivationConfig,
    cache: GoalPrimingCache | None = None,
) -> list[ActiveGoal]:
    """Find Goal/Intention entities above activation threshold.

    Excludes completed/abandoned goals. Gets 1-hop neighbors for each.
    """
    if not cfg.goal_priming_enabled:
        return []

    # Check cache
    if cache is not None:
        cached = cache.get(group_id)
        if cached is not None:
            return cached

    goals: list[ActiveGoal] = []

    # Find Goal and Intention type entities
    goal_types = ("Goal", "Intention")
    now = time.time()

    for goal_type in goal_types:
        try:
            entities = await graph_store.find_entities(
                entity_type=goal_type,
                group_id=group_id,
                limit=cfg.goal_priming_max_goals * 2,
            )
        except Exception:
            continue

        for entity in entities:
            # Skip soft-deleted
            if getattr(entity, "deleted_at", None) is not None:
                continue

            # Skip completed/abandoned goals (check attributes)
            attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
            status = attrs.get("status", "").lower()
            if status in ("completed", "abandoned", "done", "cancelled"):
                continue

            # Check activation level
            state = await activation_store.get_activation(entity.id)
            if state and state.access_history:
                from engram.activation.engine import compute_activation

                act_level = compute_activation(state.access_history, now, cfg)
            else:
                act_level = 0.0

            if act_level < cfg.goal_priming_activation_floor:
                continue

            # Get 1-hop neighbors
            neighbor_ids: list[str] = []
            if hasattr(graph_store, "get_active_neighbors_with_weights"):
                try:
                    neighbors = await graph_store.get_active_neighbors_with_weights(
                        entity_id=entity.id,
                        group_id=group_id,
                    )
                    neighbor_ids = [n[0] for n in neighbors[: cfg.goal_priming_max_neighbors]]
                except Exception:
                    pass

            goals.append(
                ActiveGoal(
                    entity_id=entity.id,
                    name=entity.name,
                    activation=act_level,
                    neighbor_ids=neighbor_ids,
                )
            )

            if len(goals) >= cfg.goal_priming_max_goals:
                break

        if len(goals) >= cfg.goal_priming_max_goals:
            break

    # Cache results
    if cache is not None:
        cache.set(group_id, goals)

    return goals


def compute_goal_priming_seeds(
    goals: list[ActiveGoal],
    cfg: ActivationConfig,
) -> list[tuple[str, float]]:
    """Convert active goals to spreading activation seeds."""
    seeds: list[tuple[str, float]] = []
    for goal in goals:
        # Goal entity itself as seed
        energy = cfg.goal_priming_boost * goal.activation
        seeds.append((goal.entity_id, energy))
        # 1-hop neighbors with reduced energy
        for nid in goal.neighbor_ids:
            seeds.append((nid, energy * 0.5))
    return seeds


def compute_goal_triage_boost(
    content: str,
    goals: list[ActiveGoal],
    cfg: ActivationConfig,
) -> float:
    """Keyword-match goal names in episode content for triage boost."""
    if not goals or not content:
        return 0.0
    content_lower = content.lower()
    matched = 0.0
    for goal in goals:
        # Check if goal name appears in content (case-insensitive)
        name_lower = goal.name.lower()
        if name_lower in content_lower:
            matched += 1
        else:
            # Check individual words (for multi-word goal names)
            words = [w for w in name_lower.split() if len(w) > 3]
            if words and any(w in content_lower for w in words):
                matched += 0.5
    if matched > 0:
        return min(cfg.goal_triage_weight, cfg.goal_triage_weight * matched)
    return 0.0
