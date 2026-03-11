"""Tests for Dream Associations: cross-domain creative connection discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.dream import (
    DreamSpreadingPhase,
    _compute_surprise_score,
)
from engram.models.activation import ActivationState
from engram.models.consolidation import CycleContext, DreamAssociationRecord
from engram.models.entity import Entity


def _make_state(access_history: list[float], node_id: str = "test") -> ActivationState:
    return ActivationState(
        node_id=node_id,
        access_history=access_history,
        access_count=len(access_history),
        last_accessed=max(access_history) if access_history else 0.0,
    )


def _dream_cfg(**overrides) -> ActivationConfig:
    defaults = {
        "consolidation_dream_enabled": True,
        "consolidation_dream_max_seeds": 5,
        "consolidation_dream_activation_floor": 0.0,
        "consolidation_dream_activation_ceiling": 1.0,
        "consolidation_dream_activation_midpoint": 0.5,
        "consolidation_dream_weight_increment": 0.05,
        "consolidation_dream_max_boost_per_edge": 0.15,
        "consolidation_dream_max_edge_weight": 3.0,
        "consolidation_dream_min_boost": 0.001,
        "consolidation_dream_associations_enabled": True,
        "consolidation_dream_assoc_max_per_cycle": 10,
        "consolidation_dream_assoc_min_surprise": 0.2,
        "consolidation_dream_assoc_ttl_days": 30,
        "consolidation_dream_assoc_weight": 0.1,
        "consolidation_dream_assoc_max_per_domain_pair": 3,
        "consolidation_dream_assoc_min_summary_len": 5,
        "consolidation_dream_assoc_structural_max_hops": 3,
        "consolidation_dream_assoc_max_duration_ms": 5000,
        "consolidation_dream_assoc_top_n_per_domain": 20,
        "cross_domain_penalty_enabled": True,
        "cross_domain_penalty_factor": 0.3,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


def _make_entity(
    eid: str,
    name: str,
    etype: str,
    summary: str = "A sufficiently long summary",
) -> Entity:
    return Entity(
        id=eid,
        name=name,
        entity_type=etype,
        summary=summary,
        group_id="default",
    )


class TestSurpriseScoring:
    def test_high_similarity_disconnected(self):
        """High similarity + no structural connection = high surprise."""
        score = _compute_surprise_score(0.9, 0.0)
        assert score == pytest.approx(0.9)

    def test_high_similarity_connected(self):
        """High similarity + structural connection = zero surprise."""
        score = _compute_surprise_score(0.9, 1.0)
        assert score == pytest.approx(0.0)

    def test_low_similarity(self):
        """Low similarity → low surprise regardless of structure."""
        score = _compute_surprise_score(0.1, 0.0)
        assert score == pytest.approx(0.1)

    def test_partial_proximity(self):
        """Partial structural proximity reduces surprise proportionally."""
        score = _compute_surprise_score(0.8, 0.5)
        assert score == pytest.approx(0.4)


class TestDomainPartitioning:
    def test_entities_grouped_by_type(self):
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()
        entities = [
            _make_entity("e1", "Python", "Technology"),
            _make_entity("e2", "React", "Technology"),
            _make_entity("e3", "John", "Person"),
            _make_entity("e4", "Conference", "Event"),
        ]
        buckets = phase._partition_by_domain(entities, cfg)

        assert "technical" in buckets
        assert "personal" in buckets
        assert len(buckets["technical"]) == 2
        assert len(buckets["personal"]) == 2  # Person + Event both → personal

    def test_uncategorized_entities(self):
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()
        entities = [
            _make_entity("e1", "Unknown", "MysteryType"),
        ]
        buckets = phase._partition_by_domain(entities, cfg)

        assert "uncategorized" in buckets
        assert len(buckets["uncategorized"]) == 1

    def test_top_n_per_domain_limit(self):
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(consolidation_dream_assoc_top_n_per_domain=5)
        entities = [_make_entity(f"e{i}", f"Tech{i}", "Technology") for i in range(10)]
        buckets = phase._partition_by_domain(entities, cfg)

        assert len(buckets["technical"]) == 5


class TestCrossDomainSimilarities:
    def test_matrix_multiplication_produces_candidates(self):
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        entities_tech = [_make_entity("t1", "Python", "Technology")]
        entities_person = [_make_entity("p1", "John", "Person")]

        domain_buckets = {
            "technical": entities_tech,
            "personal": entities_person,
        }

        # Create similar embeddings
        embeddings = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "p1": [0.9, 0.1, 0.0, 0.0],  # Similar to t1
        }

        candidates = phase._compute_cross_domain_similarities(domain_buckets, embeddings, cfg)

        assert len(candidates) > 0
        src, tgt, d1, d2, sim = candidates[0]
        assert {src, tgt} == {"t1", "p1"}
        assert sim > 0.8  # High similarity

    def test_no_candidates_for_dissimilar_entities(self):
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        entities_tech = [_make_entity("t1", "Python", "Technology")]
        entities_person = [_make_entity("p1", "John", "Person")]

        domain_buckets = {
            "technical": entities_tech,
            "personal": entities_person,
        }

        # Orthogonal embeddings
        embeddings = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "p1": [0.0, 1.0, 0.0, 0.0],  # Orthogonal to t1
        }

        candidates = phase._compute_cross_domain_similarities(domain_buckets, embeddings, cfg)

        # Below 0.2 threshold
        assert len(candidates) == 0

    def test_cross_domain_only(self):
        """Same-domain pairs should NOT appear in candidates."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        entities_tech = [
            _make_entity("t1", "Python", "Technology"),
            _make_entity("t2", "Java", "Technology"),
        ]

        domain_buckets = {
            "technical": entities_tech,
        }

        embeddings = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "t2": [0.95, 0.05, 0.0, 0.0],
        }

        candidates = phase._compute_cross_domain_similarities(domain_buckets, embeddings, cfg)

        # Only one domain → no cross-domain pairs
        assert len(candidates) == 0


class TestHebbianExclusion:
    @pytest.mark.asyncio
    async def test_dream_associated_edges_get_zero_boost(self):
        """DREAM_ASSOCIATED edges should not receive Hebbian boost."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        graph_store = AsyncMock()
        # Return a DREAM_ASSOCIATED neighbor
        graph_store.get_active_neighbors_with_weights.return_value = [
            ("n2", 0.1, "DREAM_ASSOCIATED", "Technology"),
        ]

        bonuses = {"n2": 0.5}
        boosts = await phase._accumulate_edge_boosts("seed1", bonuses, graph_store, "default", cfg)

        # Should be empty because DREAM_ASSOCIATED is excluded
        assert len(boosts) == 0

    @pytest.mark.asyncio
    async def test_regular_edges_get_boost(self):
        """Non-DREAM_ASSOCIATED edges should receive normal Hebbian boost."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights.return_value = [
            ("n2", 1.0, "RELATED_TO", "Concept"),
        ]

        bonuses = {"n2": 0.5, "seed1": 0.0}
        boosts = await phase._accumulate_edge_boosts("seed1", bonuses, graph_store, "default", cfg)

        # Should have at least one boost
        assert len(boosts) > 0


class TestCrossDomainExemption:
    @pytest.mark.asyncio
    async def test_bfs_skips_penalty_for_dream_associated(self):
        """BFS should not apply cross-domain penalty for DREAM_ASSOCIATED edges."""
        from engram.activation.bfs import BFSStrategy

        # Use equal predicate weights so we isolate the penalty effect
        cfg = _dream_cfg(
            spread_max_hops=1,
            spread_energy_budget=100.0,
            spread_firing_threshold=0.001,
            spread_decay_per_hop=0.5,
            predicate_weights={
                "DREAM_ASSOCIATED": 0.5,
                "RELATED_TO": 0.5,
            },
            predicate_weight_default=0.5,
        )

        neighbor_provider = AsyncMock()
        neighbor_provider.get_active_neighbors_with_weights.return_value = [
            ("tech_node", 1.0, "DREAM_ASSOCIATED", "Technology"),
            ("person_node", 1.0, "RELATED_TO", "Technology"),
        ]

        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread(
            seed_nodes=[("seed", 1.0)],
            neighbor_provider=neighbor_provider,
            cfg=cfg,
            group_id="default",
            seed_entity_types={"seed": "Person"},
        )

        # DREAM_ASSOCIATED: no penalty (exempt) → full spread
        # RELATED_TO: Person→Technology cross-domain → penalized by 0.3
        assert "tech_node" in bonuses
        assert "person_node" in bonuses
        assert bonuses["tech_node"] > bonuses["person_node"]

    @pytest.mark.asyncio
    async def test_ppr_skips_penalty_for_dream_associated(self):
        """PPR should not apply cross-domain penalty for DREAM_ASSOCIATED edges."""
        from engram.activation.ppr import PPRStrategy

        cfg = _dream_cfg(
            ppr_expansion_hops=1,
            ppr_alpha=0.15,
            ppr_max_iterations=20,
            ppr_epsilon=1e-6,
            spread_firing_threshold=0.001,
            spread_energy_budget=100.0,
            predicate_weights={"DREAM_ASSOCIATED": 0.5, "RELATED_TO": 0.5},
            predicate_weight_default=0.5,
        )

        neighbor_provider = AsyncMock()
        # Both edges go to "Technology" type — RELATED_TO gets cross-domain penalty,
        # DREAM_ASSOCIATED is exempt from it
        neighbor_provider.get_active_neighbors_with_weights.return_value = [
            ("tech_node", 1.0, "DREAM_ASSOCIATED", "Technology"),
            ("person_node", 1.0, "RELATED_TO", "Technology"),
        ]

        strategy = PPRStrategy()
        bonuses, _ = await strategy.spread(
            seed_nodes=[("seed", 1.0)],
            neighbor_provider=neighbor_provider,
            cfg=cfg,
            group_id="default",
            seed_entity_types={"seed": "Person"},
        )

        # DREAM_ASSOCIATED should get higher score than penalized RELATED_TO
        assert "tech_node" in bonuses
        assert "person_node" in bonuses
        assert bonuses["tech_node"] > bonuses["person_node"]


class TestDreamAssociationPhase:
    @pytest.mark.asyncio
    async def test_associations_disabled(self):
        """When dream_associations_enabled=False, no associations are created."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(consolidation_dream_associations_enabled=False)

        graph_store = AsyncMock()
        graph_store.find_entities.return_value = []
        activation_store = AsyncMock()
        activation_store.get_top_activated.return_value = []
        search_index = AsyncMock()
        search_index.get_entity_embeddings.return_value = {}

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_1",
            dry_run=False,
        )

        # No DreamAssociationRecord in output
        assoc_records = [r for r in records if isinstance(r, DreamAssociationRecord)]
        assert len(assoc_records) == 0

    @pytest.mark.asyncio
    async def test_associations_with_dry_run(self):
        """In dry_run mode, records are created but no relationships."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        entities = [
            _make_entity("t1", "Python", "Technology"),
            _make_entity("p1", "John", "Person"),
        ]

        graph_store = AsyncMock()
        graph_store.find_entities.return_value = entities
        graph_store.path_exists_within_hops.return_value = False

        activation_store = AsyncMock()
        activation_store.get_top_activated.return_value = []

        search_index = AsyncMock()
        # Embeddings with high similarity
        search_index.get_entity_embeddings.return_value = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "p1": [0.95, 0.05, 0.0, 0.0],
        }

        context = CycleContext()

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_dry",
            dry_run=True,
            context=context,
        )

        assoc_records = [r for r in records if isinstance(r, DreamAssociationRecord)]
        assert len(assoc_records) > 0
        # No relationship created in dry_run
        graph_store.create_relationship.assert_not_called()
        # Context should have IDs
        assert len(context.dream_association_ids) > 0

    @pytest.mark.asyncio
    async def test_associations_create_relationships(self):
        """Non-dry-run creates DREAM_ASSOCIATED relationships."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        entities = [
            _make_entity("t1", "Python", "Technology"),
            _make_entity("p1", "John", "Person"),
        ]

        graph_store = AsyncMock()
        graph_store.find_entities.return_value = entities
        graph_store.path_exists_within_hops.return_value = False
        graph_store.create_relationship.return_value = "rel_id"

        activation_store = AsyncMock()
        activation_store.get_top_activated.return_value = []

        search_index = AsyncMock()
        search_index.get_entity_embeddings.return_value = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "p1": [0.95, 0.05, 0.0, 0.0],
        }

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_live",
            dry_run=False,
        )

        assoc_records = [r for r in records if isinstance(r, DreamAssociationRecord)]
        assert len(assoc_records) > 0
        graph_store.create_relationship.assert_called()

        # Check relationship properties
        call_args = graph_store.create_relationship.call_args
        rel = call_args[0][0]
        assert rel.predicate == "DREAM_ASSOCIATED"
        assert rel.valid_to is not None  # TTL set

    @pytest.mark.asyncio
    async def test_max_per_cycle_respected(self):
        """Only max_per_cycle associations are created."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(consolidation_dream_assoc_max_per_cycle=1)

        # Create multiple cross-domain entities
        entities = [
            _make_entity("t1", "Python", "Technology"),
            _make_entity("t2", "Java", "Software"),
            _make_entity("p1", "John", "Person"),
            _make_entity("p2", "Jane", "Person"),
        ]

        graph_store = AsyncMock()
        graph_store.find_entities.return_value = entities
        graph_store.path_exists_within_hops.return_value = False

        activation_store = AsyncMock()
        activation_store.get_top_activated.return_value = []

        search_index = AsyncMock()
        search_index.get_entity_embeddings.return_value = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "t2": [0.98, 0.02, 0.0, 0.0],
            "p1": [0.95, 0.05, 0.0, 0.0],
            "p2": [0.93, 0.07, 0.0, 0.0],
        }

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_max",
            dry_run=True,
        )

        assoc_records = [r for r in records if isinstance(r, DreamAssociationRecord)]
        assert len(assoc_records) == 1

    @pytest.mark.asyncio
    async def test_pruned_entities_excluded(self):
        """Entities in context.pruned_entity_ids are skipped."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        entities = [
            _make_entity("t1", "Python", "Technology"),
            _make_entity("p1", "John", "Person"),
        ]

        graph_store = AsyncMock()
        graph_store.find_entities.return_value = entities
        graph_store.path_exists_within_hops.return_value = False

        activation_store = AsyncMock()
        activation_store.get_top_activated.return_value = []

        search_index = AsyncMock()
        search_index.get_entity_embeddings.return_value = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "p1": [0.95, 0.05, 0.0, 0.0],
        }

        # Mark one entity as pruned
        context = CycleContext()
        context.pruned_entity_ids.add("t1")

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_prune",
            dry_run=True,
            context=context,
        )

        # With t1 pruned, only p1 remains — not enough for cross-domain
        assoc_records = [r for r in records if isinstance(r, DreamAssociationRecord)]
        assert len(assoc_records) == 0

    @pytest.mark.asyncio
    async def test_summary_gating(self):
        """Entities with short summaries are excluded."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(consolidation_dream_assoc_min_summary_len=100)

        entities = [
            _make_entity("t1", "Python", "Technology", summary="Short"),
            _make_entity("p1", "John", "Person", summary="Also short"),
        ]

        graph_store = AsyncMock()
        graph_store.find_entities.return_value = entities

        activation_store = AsyncMock()
        activation_store.get_top_activated.return_value = []

        search_index = AsyncMock()
        search_index.get_entity_embeddings.return_value = {}

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_gate",
            dry_run=True,
        )

        assoc_records = [r for r in records if isinstance(r, DreamAssociationRecord)]
        assert len(assoc_records) == 0

    @pytest.mark.asyncio
    async def test_domain_pair_quota(self):
        """max_per_domain_pair limits associations per domain pair."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(
            consolidation_dream_assoc_max_per_domain_pair=1,
            consolidation_dream_assoc_max_per_cycle=10,
        )

        entities = [
            _make_entity("t1", "Python", "Technology"),
            _make_entity("t2", "Java", "Technology"),
            _make_entity("p1", "John", "Person"),
            _make_entity("p2", "Jane", "Person"),
        ]

        graph_store = AsyncMock()
        graph_store.find_entities.return_value = entities
        graph_store.path_exists_within_hops.return_value = False

        activation_store = AsyncMock()
        activation_store.get_top_activated.return_value = []

        search_index = AsyncMock()
        search_index.get_entity_embeddings.return_value = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "t2": [0.98, 0.02, 0.0, 0.0],
            "p1": [0.95, 0.05, 0.0, 0.0],
            "p2": [0.93, 0.07, 0.0, 0.0],
        }

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_quota",
            dry_run=True,
        )

        assoc_records = [r for r in records if isinstance(r, DreamAssociationRecord)]
        # Only 1 per domain pair (technical ↔ personal)
        assert len(assoc_records) == 1

    @pytest.mark.asyncio
    async def test_structurally_connected_entities_low_surprise(self):
        """Entities within structural hops get surprise=0, filtered out."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(consolidation_dream_assoc_min_surprise=0.1)

        entities = [
            _make_entity("t1", "Python", "Technology"),
            _make_entity("p1", "John", "Person"),
        ]

        graph_store = AsyncMock()
        graph_store.find_entities.return_value = entities
        # They are connected
        graph_store.path_exists_within_hops.return_value = True

        activation_store = AsyncMock()
        activation_store.get_top_activated.return_value = []

        search_index = AsyncMock()
        search_index.get_entity_embeddings.return_value = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "p1": [0.95, 0.05, 0.0, 0.0],
        }

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_conn",
            dry_run=True,
        )

        # Surprise = sim × (1 - 1.0) = 0, filtered out
        assoc_records = [r for r in records if isinstance(r, DreamAssociationRecord)]
        assert len(assoc_records) == 0
