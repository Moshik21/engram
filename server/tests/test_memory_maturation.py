"""Tests for Brain Architecture Phase 2A: Memory Maturation."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.maturation import MaturationPhase, compute_maturity_score
from engram.consolidation.phases.semantic_transition import SemanticTransitionPhase
from engram.models.consolidation import CycleContext
from engram.models.episode import Episode
from engram.storage.protocols import EPISODE_UPDATABLE_FIELDS

# --- Episode model tests ---


def test_episode_has_memory_tier_default():
    ep = Episode(id="ep1", content="hello")
    assert ep.memory_tier == "episodic"


def test_episode_has_consolidation_cycles_default():
    ep = Episode(id="ep1", content="hello")
    assert ep.consolidation_cycles == 0


def test_episode_has_entity_coverage_default():
    ep = Episode(id="ep1", content="hello")
    assert ep.entity_coverage == 0.0


def test_episode_accepts_custom_memory_tier():
    ep = Episode(id="ep1", content="hello", memory_tier="semantic")
    assert ep.memory_tier == "semantic"


def test_episode_updatable_fields_has_new_fields():
    assert "memory_tier" in EPISODE_UPDATABLE_FIELDS
    assert "consolidation_cycles" in EPISODE_UPDATABLE_FIELDS
    assert "entity_coverage" in EPISODE_UPDATABLE_FIELDS


# --- Config tests ---


def test_config_defaults_maturation_off():
    cfg = ActivationConfig()
    assert cfg.memory_maturation_enabled is False
    assert cfg.episode_transition_enabled is False
    assert cfg.reconsolidation_enabled is False


def test_config_standard_enables_maturation():
    cfg = ActivationConfig(consolidation_profile="standard")
    assert cfg.memory_maturation_enabled is True
    assert cfg.episode_transition_enabled is True
    assert cfg.reconsolidation_enabled is True


def test_config_conservative_enables_maturation():
    cfg = ActivationConfig(consolidation_profile="conservative")
    assert cfg.memory_maturation_enabled is True
    assert cfg.episode_transition_enabled is True
    assert cfg.reconsolidation_enabled is False  # only standard


def test_config_decay_exponents():
    cfg = ActivationConfig()
    assert cfg.decay_exponent_episodic == 0.5
    assert cfg.decay_exponent_semantic == 0.3


# --- compute_maturity_score tests ---


def test_maturity_score_source_diversity():
    cfg = ActivationConfig()
    # 10 episodes saturates at 1.0
    score = compute_maturity_score(10, 0.0, 0, [], cfg)
    assert abs(score - cfg.maturation_source_weight * 1.0) < 0.01


def test_maturity_score_temporal_span():
    cfg = ActivationConfig()
    score = compute_maturity_score(0, 90.0, 0, [], cfg)
    assert abs(score - cfg.maturation_temporal_weight * 1.0) < 0.01


def test_maturity_score_richness():
    cfg = ActivationConfig()
    score = compute_maturity_score(0, 0.0, 8, [], cfg)
    assert abs(score - cfg.maturation_richness_weight * 1.0) < 0.01


def test_maturity_score_regularity_perfect():
    cfg = ActivationConfig()
    # Perfectly regular intervals: CV=0, regularity=1.0
    intervals = [100.0, 100.0, 100.0, 100.0]
    score = compute_maturity_score(0, 0.0, 0, intervals, cfg)
    assert abs(score - cfg.maturation_regularity_weight * 1.0) < 0.01


def test_maturity_score_regularity_irregular():
    cfg = ActivationConfig()
    # Very irregular intervals: high CV
    intervals = [1.0, 1000.0, 1.0, 1000.0]
    score = compute_maturity_score(0, 0.0, 0, intervals, cfg)
    assert score < cfg.maturation_regularity_weight * 0.5


def test_maturity_score_combined():
    cfg = ActivationConfig()
    score = compute_maturity_score(10, 90.0, 8, [100.0] * 4, cfg)
    assert score >= 0.9


# --- MaturationPhase tests ---


@pytest.fixture
def mat_cfg():
    return ActivationConfig(
        consolidation_profile="off",
        memory_maturation_enabled=True,
        maturation_min_age_days=0,  # disable age gate for testing
    )


@pytest.fixture
def mat_cfg_with_age():
    return ActivationConfig(
        consolidation_profile="off",
        memory_maturation_enabled=True,
        maturation_min_age_days=7,
    )


def _make_entity(
    entity_id="ent1", name="Test", attrs=None,
    created_days_ago=30, identity_core=False,
):
    entity = MagicMock()
    entity.id = entity_id
    entity.name = name
    entity.entity_type = "Concept"
    entity.deleted_at = None
    entity.created_at = datetime.utcnow() - timedelta(days=created_days_ago)
    entity.identity_core = identity_core
    entity.attributes = attrs or {}
    return entity


@pytest.mark.asyncio
async def test_maturation_phase_skipped_when_disabled():
    cfg = ActivationConfig()
    phase = MaturationPhase()
    result, records = await phase.execute(
        "default", AsyncMock(), AsyncMock(), AsyncMock(),
        cfg, "cyc1", dry_run=False,
    )
    assert result.status == "skipped"
    assert records == []


@pytest.mark.asyncio
async def test_maturation_phase_identity_core_auto_semantic(mat_cfg):
    entity = _make_entity(identity_core=True)
    graph = AsyncMock()
    graph.find_entities.return_value = [entity]
    activation = AsyncMock()
    activation.get_activation.return_value = None
    search = AsyncMock()

    phase = MaturationPhase()
    context = CycleContext()
    result, records = await phase.execute(
        "default", graph, activation, search,
        mat_cfg, "cyc1", dry_run=False, context=context,
    )
    assert len(records) == 1
    assert records[0].new_tier == "semantic"
    assert entity.id in context.matured_entity_ids


@pytest.mark.asyncio
async def test_maturation_phase_transitional_threshold(mat_cfg):
    entity = _make_entity(attrs={})
    graph = AsyncMock()
    graph.find_entities.return_value = [entity]
    graph.get_entity_episode_count.return_value = 5
    graph.get_entity_temporal_span.return_value = (
        (datetime.utcnow() - timedelta(days=45)).isoformat(),
        datetime.utcnow().isoformat(),
    )
    graph.get_entity_relationship_types.return_value = ["WORKS_AT", "KNOWS", "USES"]
    activation = AsyncMock()
    state = MagicMock()
    state.access_history = [time.time() - i * 86400 for i in range(5)]
    activation.get_activation.return_value = state
    search = AsyncMock()

    phase = MaturationPhase()
    result, records = await phase.execute(
        "default", graph, activation, search,
        mat_cfg, "cyc1", dry_run=False,
    )
    assert len(records) >= 1
    assert records[0].new_tier in ("transitional", "semantic")


@pytest.mark.asyncio
async def test_maturation_phase_min_age_gate(mat_cfg_with_age):
    # Entity created 1 day ago should NOT be promoted
    entity = _make_entity(created_days_ago=1)
    graph = AsyncMock()
    graph.find_entities.return_value = [entity]
    activation = AsyncMock()
    search = AsyncMock()

    phase = MaturationPhase()
    result, records = await phase.execute(
        "default", graph, activation, search,
        mat_cfg_with_age, "cyc1", dry_run=False,
    )
    assert len(records) == 0


@pytest.mark.asyncio
async def test_maturation_phase_dry_run(mat_cfg):
    entity = _make_entity(identity_core=True)
    graph = AsyncMock()
    graph.find_entities.return_value = [entity]
    activation = AsyncMock()
    activation.get_activation.return_value = None
    search = AsyncMock()

    phase = MaturationPhase()
    result, records = await phase.execute(
        "default", graph, activation, search,
        mat_cfg, "cyc1", dry_run=True,
    )
    assert len(records) == 1
    graph.update_entity.assert_not_called()


@pytest.mark.asyncio
async def test_maturation_phase_already_semantic_skipped(mat_cfg):
    entity = _make_entity(attrs={"mat_tier": "semantic"})
    graph = AsyncMock()
    graph.find_entities.return_value = [entity]
    activation = AsyncMock()
    search = AsyncMock()

    phase = MaturationPhase()
    result, records = await phase.execute(
        "default", graph, activation, search,
        mat_cfg, "cyc1", dry_run=False,
    )
    assert len(records) == 0


# --- Differential decay tests ---


def test_differential_decay_semantic_higher():
    from engram.activation.engine import compute_activation

    cfg = ActivationConfig(memory_maturation_enabled=True)
    now = time.time()
    history = [now - 86400 * 7]  # 7 days ago

    act_episodic = compute_activation(history, now, cfg, decay_override=cfg.decay_exponent_episodic)
    act_semantic = compute_activation(history, now, cfg, decay_override=cfg.decay_exponent_semantic)
    assert act_semantic > act_episodic


def test_decay_override_none_uses_default():
    from engram.activation.engine import compute_activation

    cfg = ActivationConfig()
    now = time.time()
    history = [now - 3600]

    act_default = compute_activation(history, now, cfg)
    act_none_override = compute_activation(history, now, cfg, decay_override=None)
    assert act_default == act_none_override


# --- SemanticTransitionPhase tests ---


def _make_episode(ep_id="ep1", status="completed", memory_tier="episodic", cycles=0):
    return Episode(
        id=ep_id,
        content="test content",
        status=status,
        memory_tier=memory_tier,
        consolidation_cycles=cycles,
    )


@pytest.mark.asyncio
async def test_semantic_transition_skipped_when_disabled():
    cfg = ActivationConfig()
    phase = SemanticTransitionPhase()
    result, records = await phase.execute(
        "default", AsyncMock(), AsyncMock(), AsyncMock(),
        cfg, "cyc1",
    )
    assert result.status == "skipped"


@pytest.mark.asyncio
async def test_semantic_transition_increments_cycles():
    cfg = ActivationConfig(
        consolidation_profile="off",
        episode_transition_enabled=True,
    )
    ep = _make_episode(cycles=2)
    graph = AsyncMock()
    graph.get_episodes.return_value = [ep]
    graph.get_episode_entities.return_value = []

    phase = SemanticTransitionPhase()
    result, records = await phase.execute(
        "default", graph, AsyncMock(), AsyncMock(),
        cfg, "cyc1", dry_run=False,
    )
    # Should update consolidation_cycles to 3
    graph.update_episode.assert_called_once()
    call_args = graph.update_episode.call_args
    assert call_args[0][1]["consolidation_cycles"] == 3


@pytest.mark.asyncio
async def test_semantic_transition_promotes_on_coverage():
    cfg = ActivationConfig(
        consolidation_profile="off",
        episode_transition_enabled=True,
        episode_transitional_coverage=0.50,
        episode_transitional_min_cycles=1,
    )
    ep = _make_episode(cycles=1)
    # 2 entities, both mature
    ent1 = _make_entity("ent1", attrs={"mat_tier": "semantic"})
    ent2 = _make_entity("ent2", attrs={"mat_tier": "transitional"})

    graph = AsyncMock()
    graph.get_episodes.return_value = [ep]
    graph.get_episode_entities.return_value = ["ent1", "ent2"]
    graph.get_entity.side_effect = lambda eid, gid: {"ent1": ent1, "ent2": ent2}.get(eid)

    phase = SemanticTransitionPhase()
    context = CycleContext()
    result, records = await phase.execute(
        "default", graph, AsyncMock(), AsyncMock(),
        cfg, "cyc1", dry_run=False, context=context,
    )
    assert len(records) == 1
    assert records[0].new_tier == "transitional"
    assert ep.id in context.transitioned_episode_ids


# --- Prune resistance tests ---


@pytest.mark.asyncio
async def test_prune_semantic_entity_survives():
    """Semantic entities should survive pruning if younger than semantic_prune_age_days."""
    from engram.consolidation.phases.prune import PrunePhase

    cfg = ActivationConfig(
        consolidation_profile="off",
        memory_maturation_enabled=True,
        semantic_prune_age_days=180,
    )

    # Entity is 30 days old, semantic tier — should survive 180-day cutoff
    entity = _make_entity(created_days_ago=30, attrs={"mat_tier": "semantic"})

    graph = AsyncMock()
    graph.get_dead_entities.return_value = [entity]
    graph.get_entity.return_value = entity

    activation = AsyncMock()
    activation.get_activation.return_value = None  # No activation state

    phase = PrunePhase()
    result, records = await phase.execute(
        "default", graph, activation, AsyncMock(),
        cfg, "cyc1", dry_run=False,
    )
    assert len(records) == 0  # Not pruned


# --- Engine phase order ---


def test_engine_phase_order():
    from engram.consolidation.engine import ConsolidationEngine

    cfg = ActivationConfig()
    e = ConsolidationEngine(AsyncMock(), AsyncMock(), AsyncMock(), cfg)
    phases = [p.name for p in e._phases]
    assert phases == [
        "triage", "merge", "infer", "replay", "prune", "compact",
        "mature", "semanticize", "schema", "reindex", "graph_embed", "dream",
    ]


# --- Scheduler PHASE_TIERS ---


def test_scheduler_phase_tiers():
    from engram.consolidation.scheduler import PHASE_TIERS

    assert "mature" in PHASE_TIERS
    assert PHASE_TIERS["mature"] == "warm"
    assert "semanticize" in PHASE_TIERS
    assert PHASE_TIERS["semanticize"] == "warm"


# --- CycleContext fields ---


def test_cycle_context_has_maturation_fields():
    ctx = CycleContext()
    assert hasattr(ctx, "matured_entity_ids")
    assert hasattr(ctx, "transitioned_episode_ids")
    assert isinstance(ctx.matured_entity_ids, set)
    assert isinstance(ctx.transitioned_episode_ids, set)
