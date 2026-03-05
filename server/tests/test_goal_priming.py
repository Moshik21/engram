"""Tests for goal-relevance gating (Phase 1D: Brain Architecture)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.retrieval.goals import (
    ActiveGoal,
    GoalPrimingCache,
    compute_goal_priming_seeds,
    compute_goal_triage_boost,
    identify_active_goals,
)

# --- Mock helpers ---


@dataclass
class MockEntity:
    id: str
    name: str
    entity_type: str
    summary: str = ""
    attributes: dict = field(default_factory=dict)
    deleted_at: object = None
    identity_core: bool = False


@dataclass
class MockActivationState:
    access_history: list[float] = field(default_factory=list)
    access_count: int = 0
    consolidated_strength: float = 0.0


def make_cfg(**overrides) -> ActivationConfig:
    defaults = {
        "goal_priming_enabled": True,
        "goal_priming_boost": 0.10,
        "goal_priming_activation_floor": 0.15,
        "goal_priming_max_goals": 5,
        "goal_priming_max_neighbors": 10,
        "goal_priming_cache_ttl_seconds": 60.0,
        "goal_triage_weight": 0.10,
        "goal_prune_protection": True,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


def make_graph_store(entities_by_type=None, neighbors=None):
    store = AsyncMock()

    async def find_entities(entity_type=None, group_id=None, limit=10, **kwargs):
        if entities_by_type and entity_type in entities_by_type:
            return entities_by_type[entity_type][:limit]
        return []

    store.find_entities = AsyncMock(side_effect=find_entities)

    async def get_active_neighbors(entity_id=None, group_id=None, **kwargs):
        if neighbors and entity_id in neighbors:
            return neighbors[entity_id]
        return []

    store.get_active_neighbors_with_weights = AsyncMock(side_effect=get_active_neighbors)
    return store


def make_activation_store(states=None):
    store = AsyncMock()

    async def get_activation(entity_id):
        if states and entity_id in states:
            return states[entity_id]
        return None

    store.get_activation = AsyncMock(side_effect=get_activation)
    return store


# --- Tests ---


class TestIdentifyActiveGoals:
    @pytest.mark.asyncio
    async def test_finds_goals_above_threshold(self):
        now = time.time()
        cfg = make_cfg(goal_priming_activation_floor=0.01)
        goal_entity = MockEntity(id="g1", name="Learn Python", entity_type="Goal")
        graph_store = make_graph_store(
            entities_by_type={"Goal": [goal_entity]},
            neighbors={"g1": [("n1", 0.5, "RELATED_TO"), ("n2", 0.3, "USES")]},
        )
        activation_store = make_activation_store(
            states={"g1": MockActivationState(access_history=[now - 10, now - 5], access_count=2)}
        )

        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)
        assert len(goals) == 1
        assert goals[0].entity_id == "g1"
        assert goals[0].name == "Learn Python"
        assert len(goals[0].neighbor_ids) == 2

    @pytest.mark.asyncio
    async def test_excludes_completed_goals(self):
        now = time.time()
        cfg = make_cfg(goal_priming_activation_floor=0.01)
        completed = MockEntity(
            id="g1", name="Done Goal", entity_type="Goal",
            attributes={"status": "completed"},
        )
        graph_store = make_graph_store(entities_by_type={"Goal": [completed]})
        activation_store = make_activation_store(
            states={"g1": MockActivationState(access_history=[now - 1], access_count=1)}
        )

        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)
        assert len(goals) == 0

    @pytest.mark.asyncio
    async def test_excludes_abandoned_goals(self):
        now = time.time()
        cfg = make_cfg(goal_priming_activation_floor=0.01)
        abandoned = MockEntity(
            id="g1", name="Old Goal", entity_type="Goal",
            attributes={"status": "abandoned"},
        )
        graph_store = make_graph_store(entities_by_type={"Goal": [abandoned]})
        activation_store = make_activation_store(
            states={"g1": MockActivationState(access_history=[now - 1], access_count=1)}
        )

        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)
        assert len(goals) == 0

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        cfg = make_cfg(goal_priming_enabled=False)
        graph_store = make_graph_store()
        activation_store = make_activation_store()

        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)
        assert goals == []
        # Should not even call graph_store
        graph_store.find_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_soft_deleted(self):
        now = time.time()
        cfg = make_cfg(goal_priming_activation_floor=0.01)
        deleted = MockEntity(
            id="g1", name="Deleted Goal", entity_type="Goal",
            deleted_at="2025-01-01",
        )
        graph_store = make_graph_store(entities_by_type={"Goal": [deleted]})
        activation_store = make_activation_store(
            states={"g1": MockActivationState(access_history=[now - 1], access_count=1)}
        )

        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)
        assert len(goals) == 0

    @pytest.mark.asyncio
    async def test_respects_max_goals(self):
        now = time.time()
        cfg = make_cfg(goal_priming_activation_floor=0.01, goal_priming_max_goals=2)
        entities = [
            MockEntity(id=f"g{i}", name=f"Goal {i}", entity_type="Goal")
            for i in range(5)
        ]
        graph_store = make_graph_store(entities_by_type={"Goal": entities})
        states = {
            f"g{i}": MockActivationState(access_history=[now - 1], access_count=1)
            for i in range(5)
        }
        activation_store = make_activation_store(states=states)

        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)
        assert len(goals) == 2

    @pytest.mark.asyncio
    async def test_finds_intention_type(self):
        now = time.time()
        cfg = make_cfg(goal_priming_activation_floor=0.01)
        intention = MockEntity(id="i1", name="Buy groceries", entity_type="Intention")
        graph_store = make_graph_store(
            entities_by_type={"Intention": [intention]},
        )
        activation_store = make_activation_store(
            states={"i1": MockActivationState(access_history=[now - 1], access_count=1)}
        )

        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)
        assert len(goals) == 1
        assert goals[0].name == "Buy groceries"


class TestGoalPrimingCache:
    def test_cache_stores_and_retrieves(self):
        cache = GoalPrimingCache(ttl_seconds=60.0)
        goals = [ActiveGoal(entity_id="g1", name="Goal", activation=0.5)]
        cache.set("default", goals)
        result = cache.get("default")
        assert result is not None
        assert len(result) == 1
        assert result[0].entity_id == "g1"

    def test_cache_expires(self):
        cache = GoalPrimingCache(ttl_seconds=0.01)
        goals = [ActiveGoal(entity_id="g1", name="Goal", activation=0.5)]
        cache.set("default", goals)
        time.sleep(0.02)
        result = cache.get("default")
        assert result is None

    def test_cache_miss(self):
        cache = GoalPrimingCache(ttl_seconds=60.0)
        assert cache.get("default") is None

    def test_invalidate_specific(self):
        cache = GoalPrimingCache(ttl_seconds=60.0)
        goals = [ActiveGoal(entity_id="g1", name="Goal", activation=0.5)]
        cache.set("default", goals)
        cache.set("other", goals)
        cache.invalidate("default")
        assert cache.get("default") is None
        assert cache.get("other") is not None

    def test_invalidate_all(self):
        cache = GoalPrimingCache(ttl_seconds=60.0)
        goals = [ActiveGoal(entity_id="g1", name="Goal", activation=0.5)]
        cache.set("default", goals)
        cache.set("other", goals)
        cache.invalidate()
        assert cache.get("default") is None
        assert cache.get("other") is None

    @pytest.mark.asyncio
    async def test_cache_used_by_identify(self):
        cfg = make_cfg(goal_priming_activation_floor=0.01)
        cache = GoalPrimingCache(ttl_seconds=60.0)
        cached_goals = [ActiveGoal(entity_id="g1", name="Cached", activation=0.5)]
        cache.set("default", cached_goals)

        graph_store = make_graph_store()
        activation_store = make_activation_store()

        goals = await identify_active_goals(
            graph_store, activation_store, "default", cfg, cache=cache,
        )
        assert len(goals) == 1
        assert goals[0].name == "Cached"
        # Should not query graph_store since cache hit
        graph_store.find_entities.assert_not_called()


class TestComputeGoalPrimingSeeds:
    def test_creates_seeds_with_energy(self):
        cfg = make_cfg(goal_priming_boost=0.10)
        goals = [
            ActiveGoal(
                entity_id="g1", name="Goal", activation=0.8,
                neighbor_ids=["n1", "n2"],
            ),
        ]
        seeds = compute_goal_priming_seeds(goals, cfg)
        # g1 + 2 neighbors = 3 seeds
        assert len(seeds) == 3
        # Goal seed energy = boost * activation = 0.10 * 0.8
        assert seeds[0] == ("g1", pytest.approx(0.08))
        # Neighbor energy = half of goal
        assert seeds[1] == ("n1", pytest.approx(0.04))
        assert seeds[2] == ("n2", pytest.approx(0.04))

    def test_empty_goals(self):
        cfg = make_cfg()
        seeds = compute_goal_priming_seeds([], cfg)
        assert seeds == []

    def test_goal_without_neighbors(self):
        cfg = make_cfg(goal_priming_boost=0.10)
        goals = [ActiveGoal(entity_id="g1", name="Goal", activation=1.0)]
        seeds = compute_goal_priming_seeds(goals, cfg)
        assert len(seeds) == 1
        assert seeds[0] == ("g1", pytest.approx(0.10))


class TestComputeGoalTriageBoost:
    def test_keyword_match(self):
        cfg = make_cfg(goal_triage_weight=0.10)
        goals = [ActiveGoal(entity_id="g1", name="Learn Python", activation=0.5)]
        boost = compute_goal_triage_boost("I am learning python programming", goals, cfg)
        assert boost > 0

    def test_partial_word_match(self):
        cfg = make_cfg(goal_triage_weight=0.10)
        goals = [ActiveGoal(entity_id="g1", name="Learn Python", activation=0.5)]
        boost = compute_goal_triage_boost("python is great for scripting", goals, cfg)
        assert boost > 0

    def test_no_match_returns_zero(self):
        cfg = make_cfg(goal_triage_weight=0.10)
        goals = [ActiveGoal(entity_id="g1", name="Learn Python", activation=0.5)]
        boost = compute_goal_triage_boost("The weather is nice today", goals, cfg)
        assert boost == 0.0

    def test_empty_content_returns_zero(self):
        cfg = make_cfg()
        goals = [ActiveGoal(entity_id="g1", name="Goal", activation=0.5)]
        assert compute_goal_triage_boost("", goals, cfg) == 0.0

    def test_empty_goals_returns_zero(self):
        cfg = make_cfg()
        assert compute_goal_triage_boost("some content", [], cfg) == 0.0

    def test_boost_capped_at_weight(self):
        cfg = make_cfg(goal_triage_weight=0.10)
        goals = [
            ActiveGoal(entity_id="g1", name="Python", activation=0.5),
            ActiveGoal(entity_id="g2", name="Rust", activation=0.5),
            ActiveGoal(entity_id="g3", name="Go", activation=0.5),
        ]
        boost = compute_goal_triage_boost("python rust go all day", goals, cfg)
        assert boost <= cfg.goal_triage_weight


class TestConfigFields:
    def test_default_values(self):
        cfg = ActivationConfig()
        assert cfg.goal_priming_enabled is False
        assert cfg.goal_priming_boost == 0.10
        assert cfg.goal_priming_activation_floor == 0.15
        assert cfg.goal_priming_max_goals == 5
        assert cfg.goal_priming_max_neighbors == 10
        assert cfg.goal_priming_cache_ttl_seconds == 60.0
        assert cfg.goal_triage_weight == 0.10
        assert cfg.goal_prune_protection is True

    def test_conservative_enables(self):
        cfg = ActivationConfig(consolidation_profile="conservative")
        assert cfg.goal_priming_enabled is True

    def test_standard_enables(self):
        cfg = ActivationConfig(consolidation_profile="standard")
        assert cfg.goal_priming_enabled is True

    def test_off_profile_disabled(self):
        cfg = ActivationConfig(consolidation_profile="off")
        assert cfg.goal_priming_enabled is False


class TestPruneProtection:
    @pytest.mark.asyncio
    async def test_prune_skips_goal_entities(self):
        """Goal-related entities should be protected from pruning."""
        from engram.consolidation.phases.prune import PrunePhase

        now = time.time()
        cfg = make_cfg(
            goal_priming_enabled=True,
            goal_prune_protection=True,
            goal_priming_activation_floor=0.01,
            consolidation_prune_activation_floor=0.05,
            consolidation_prune_min_age_days=30,
            consolidation_prune_max_per_cycle=100,
            consolidation_prune_min_access_count=0,
        )

        # The "dead" entity is a goal neighbor
        dead_entity = MockEntity(id="n1", name="Related", entity_type="Concept")
        goal_entity = MockEntity(id="g1", name="Learn ML", entity_type="Goal")

        graph_store = AsyncMock()
        graph_store.get_dead_entities = AsyncMock(return_value=[dead_entity])
        graph_store.find_entities = AsyncMock(side_effect=lambda entity_type=None, **kw: (
            [goal_entity] if entity_type == "Goal" else []
        ))
        graph_store.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("n1", 0.5, "RELATED_TO")],
        )

        activation_store = AsyncMock()
        # Goal has high activation
        async def get_act(entity_id):
            if entity_id == "g1":
                return MockActivationState(access_history=[now - 1], access_count=1)
            # Dead entity has low activation
            return MockActivationState(access_history=[], access_count=0)

        activation_store.get_activation = AsyncMock(side_effect=get_act)

        phase = PrunePhase()
        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="test",
            dry_run=False,
        )

        # n1 should be protected (goal neighbor), so no prunes
        assert result.items_affected == 0
        assert len(records) == 0
