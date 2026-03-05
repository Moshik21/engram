"""Tests for inhibitory spreading: predicate suppression + lateral inhibition."""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig
from engram.retrieval.inhibition import (
    apply_inhibition,
    apply_lateral_inhibition,
    apply_predicate_inhibition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> ActivationConfig:
    defaults = {
        "inhibitory_spreading_enabled": True,
        "inhibit_strength": 0.3,
        "inhibit_similarity_threshold": 0.6,
        "inhibit_max_seed_anchors": 5,
        "inhibition_predicate_suppression": True,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


class _FakeSearchIndex:
    """Mock search index with get_entity_embeddings."""

    def __init__(self, embeddings: dict[str, list[float]]):
        self._embeddings = embeddings

    async def get_entity_embeddings(
        self, entity_ids: list[str], *, group_id: str = "default",
    ) -> dict[str, list[float]]:
        return {eid: self._embeddings[eid] for eid in entity_ids if eid in self._embeddings}


class _NoEmbeddingsIndex:
    """Search index without get_entity_embeddings."""
    pass


# ---------------------------------------------------------------------------
# Predicate inhibition tests
# ---------------------------------------------------------------------------

class TestPredicateInhibition:
    def test_weaker_group_suppressed(self):
        """LIKES vs DISLIKES: weaker side gets suppressed."""
        bonuses = {"pizza": 0.5, "broccoli": 0.5}
        seeds = {"user1"}
        cfg = _cfg()

        # user1 LIKES pizza (weight 2.0), user1 DISLIKES broccoli (weight 1.0)
        relationships = [
            ("user1", "pizza", "LIKES", 2.0),
            ("user1", "broccoli", "DISLIKES", 1.0),
        ]

        result = apply_predicate_inhibition(
            bonuses, seeds, None, "default", cfg, relationships=relationships,
        )

        # DISLIKES is weaker (1.0 < 2.0), broccoli should be suppressed
        assert result["broccoli"] == pytest.approx(0.5 * (1.0 - 0.3))
        # LIKES is stronger, pizza unchanged
        assert result["pizza"] == pytest.approx(0.5)

    def test_noop_when_no_contradictory_pairs(self):
        """No suppression when predicates are not contradictory."""
        bonuses = {"a": 0.5, "b": 0.5}
        seeds = {"user1"}
        cfg = _cfg()

        relationships = [
            ("user1", "a", "WORKS_AT", 1.0),
            ("user1", "b", "KNOWS", 1.0),
        ]

        result = apply_predicate_inhibition(
            bonuses, seeds, None, "default", cfg, relationships=relationships,
        )

        assert result["a"] == pytest.approx(0.5)
        assert result["b"] == pytest.approx(0.5)

    def test_disabled_when_config_false(self):
        """No suppression when inhibition_predicate_suppression=False."""
        bonuses = {"pizza": 0.5, "broccoli": 0.5}
        seeds = {"user1"}
        cfg = _cfg(inhibition_predicate_suppression=False)

        relationships = [
            ("user1", "pizza", "LIKES", 2.0),
            ("user1", "broccoli", "DISLIKES", 1.0),
        ]

        result = apply_predicate_inhibition(
            bonuses, seeds, None, "default", cfg, relationships=relationships,
        )

        assert result["pizza"] == pytest.approx(0.5)
        assert result["broccoli"] == pytest.approx(0.5)

    def test_noop_with_no_relationships(self):
        """No suppression when relationships is None or empty."""
        bonuses = {"a": 0.5}
        seeds = {"user1"}
        cfg = _cfg()

        result = apply_predicate_inhibition(
            bonuses, seeds, None, "default", cfg, relationships=None,
        )
        assert result["a"] == pytest.approx(0.5)

        result = apply_predicate_inhibition(
            bonuses, seeds, None, "default", cfg, relationships=[],
        )
        assert result["a"] == pytest.approx(0.5)

    def test_aims_for_avoids_pair(self):
        """AIMS_FOR vs AVOIDS contradictory pair works."""
        bonuses = {"goal": 0.8, "antigol": 0.6}
        seeds = {"user1"}
        cfg = _cfg()

        relationships = [
            ("user1", "goal", "AIMS_FOR", 3.0),
            ("user1", "antigol", "AVOIDS", 1.0),
        ]

        result = apply_predicate_inhibition(
            bonuses, seeds, None, "default", cfg, relationships=relationships,
        )

        # AVOIDS is weaker
        assert result["antigol"] < 0.6
        assert result["goal"] == pytest.approx(0.8)

    def test_three_field_relationships(self):
        """Relationships with only 3 fields (no weight) default to 1.0."""
        bonuses = {"pizza": 0.5, "broccoli": 0.5}
        seeds = {"user1"}
        cfg = _cfg()

        # Same predicate count but LIKES has 2 entries, DISLIKES has 1
        relationships = [
            ("user1", "pizza", "LIKES"),
            ("user1", "pizza2", "LIKES"),
            ("user1", "broccoli", "DISLIKES"),
        ]

        result = apply_predicate_inhibition(
            bonuses, seeds, None, "default", cfg, relationships=relationships,
        )

        # DISLIKES weaker (1.0 vs 2.0)
        assert result["broccoli"] < 0.5


# ---------------------------------------------------------------------------
# Lateral inhibition tests
# ---------------------------------------------------------------------------

class TestLateralInhibition:
    @pytest.mark.asyncio
    async def test_graph_disconnected_suppressed(self):
        """High-sim, graph-disconnected entities get full inhibition."""
        # Create embeddings: seed and candidate are very similar
        seed_emb = [1.0, 0.0, 0.0]
        cand_emb = [0.95, 0.05, 0.0]  # high similarity to seed

        search_index = _FakeSearchIndex({
            "seed1": seed_emb,
            "cand1": cand_emb,
        })

        bonuses = {"cand1": 0.5}
        hop_distances: dict[str, int] = {}  # cand1 is graph-disconnected
        seeds = {"seed1"}
        cfg = _cfg(inhibit_similarity_threshold=0.5)

        result = await apply_lateral_inhibition(
            bonuses, hop_distances, seeds, search_index, "default", cfg,
        )

        # cand1 should be suppressed (disconnected + high similarity)
        assert result["cand1"] < 0.5

    @pytest.mark.asyncio
    async def test_graph_connected_reduced_inhibition(self):
        """Graph-connected entities get reduced inhibition based on hop distance."""
        seed_emb = [1.0, 0.0, 0.0]
        cand_emb = [0.95, 0.05, 0.0]

        search_index = _FakeSearchIndex({
            "seed1": seed_emb,
            "cand_near": cand_emb,
            "cand_far": cand_emb,
        })

        bonuses = {"cand_near": 0.5, "cand_far": 0.5}
        hop_distances = {"cand_near": 1, "cand_far": 3}
        seeds = {"seed1"}
        cfg = _cfg(inhibit_similarity_threshold=0.5)

        result = await apply_lateral_inhibition(
            bonuses, hop_distances, seeds, search_index, "default", cfg,
        )

        # 1-hop gets less inhibition than 3-hop
        assert result["cand_near"] > result["cand_far"]

    @pytest.mark.asyncio
    async def test_low_similarity_not_inhibited(self):
        """Entities below similarity threshold are not inhibited."""
        seed_emb = [1.0, 0.0, 0.0]
        cand_emb = [0.0, 1.0, 0.0]  # orthogonal = 0 similarity

        search_index = _FakeSearchIndex({
            "seed1": seed_emb,
            "cand1": cand_emb,
        })

        bonuses = {"cand1": 0.5}
        hop_distances: dict[str, int] = {}
        seeds = {"seed1"}
        cfg = _cfg(inhibit_similarity_threshold=0.6)

        result = await apply_lateral_inhibition(
            bonuses, hop_distances, seeds, search_index, "default", cfg,
        )

        assert result["cand1"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_graceful_with_missing_embeddings(self):
        """Returns bonuses unchanged when search index has no embeddings."""
        search_index = _NoEmbeddingsIndex()

        bonuses = {"a": 0.5, "b": 0.3}
        result = await apply_lateral_inhibition(
            bonuses, {}, {"seed1"}, search_index, "default", _cfg(),
        )

        assert result == bonuses

    @pytest.mark.asyncio
    async def test_graceful_with_empty_seed_embeddings(self):
        """Returns bonuses unchanged when seed embeddings are not found."""
        search_index = _FakeSearchIndex({
            "cand1": [1.0, 0.0],
        })

        bonuses = {"cand1": 0.5}
        result = await apply_lateral_inhibition(
            bonuses, {}, {"seed_missing"}, search_index, "default", _cfg(),
        )

        assert result["cand1"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_bonuses_never_negative(self):
        """Inhibition should not push bonuses below zero."""
        seed_emb = [1.0, 0.0, 0.0]
        cand_emb = [0.99, 0.01, 0.0]  # very high similarity

        search_index = _FakeSearchIndex({
            "seed1": seed_emb,
            "cand1": cand_emb,
        })

        # Start with a very small bonus
        bonuses = {"cand1": 0.01}
        hop_distances: dict[str, int] = {}  # disconnected
        seeds = {"seed1"}
        cfg = _cfg(inhibit_strength=0.9, inhibit_similarity_threshold=0.5)

        result = await apply_lateral_inhibition(
            bonuses, hop_distances, seeds, search_index, "default", cfg,
        )

        assert result["cand1"] >= 0.0

    @pytest.mark.asyncio
    async def test_empty_bonuses_noop(self):
        """No crash with empty bonuses dict."""
        search_index = _FakeSearchIndex({"seed1": [1.0, 0.0]})
        result = await apply_lateral_inhibition(
            {}, {}, {"seed1"}, search_index, "default", _cfg(),
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_seeds_noop(self):
        """No crash with empty seeds."""
        search_index = _FakeSearchIndex({})
        result = await apply_lateral_inhibition(
            {"a": 0.5}, {}, set(), search_index, "default", _cfg(),
        )
        assert result["a"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

class TestApplyInhibition:
    @pytest.mark.asyncio
    async def test_feature_gated(self):
        """Returns original bonuses when feature is disabled."""
        cfg = _cfg(inhibitory_spreading_enabled=False)
        bonuses = {"a": 0.5, "b": 0.3}

        result = await apply_inhibition(
            bonuses=bonuses,
            hop_distances={},
            seed_node_ids={"seed1"},
            graph_store=None,
            search_index=None,
            group_id="default",
            cfg=cfg,
        )

        assert result == bonuses

    @pytest.mark.asyncio
    async def test_orchestrator_runs_both(self):
        """Orchestrator runs predicate suppression then lateral inhibition."""
        seed_emb = [1.0, 0.0, 0.0]
        cand_emb = [0.95, 0.05, 0.0]

        search_index = _FakeSearchIndex({
            "seed1": seed_emb,
            "pizza": cand_emb,
            "broccoli": cand_emb,
        })

        bonuses = {"pizza": 0.5, "broccoli": 0.5}
        relationships = [
            ("seed1", "pizza", "LIKES", 2.0),
            ("seed1", "broccoli", "DISLIKES", 1.0),
        ]

        cfg = _cfg(inhibit_similarity_threshold=0.5)

        result = await apply_inhibition(
            bonuses=bonuses,
            hop_distances={},
            seed_node_ids={"seed1"},
            graph_store=None,
            search_index=search_index,
            group_id="default",
            cfg=cfg,
            relationships=relationships,
        )

        # broccoli should be double-hit: predicate suppression + lateral inhibition
        # pizza should only get lateral inhibition (not predicate-suppressed)
        assert result["broccoli"] < result["pizza"]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigFields:
    def test_defaults_exist(self):
        cfg = ActivationConfig()
        assert cfg.inhibitory_spreading_enabled is False
        assert cfg.inhibit_strength == pytest.approx(0.3)
        assert cfg.inhibit_similarity_threshold == pytest.approx(0.6)
        assert cfg.inhibit_max_seed_anchors == 5
        assert cfg.inhibition_predicate_suppression is True

    def test_conservative_enables(self):
        cfg = ActivationConfig(consolidation_profile="conservative")
        assert cfg.inhibitory_spreading_enabled is True

    def test_standard_enables(self):
        cfg = ActivationConfig(consolidation_profile="standard")
        assert cfg.inhibitory_spreading_enabled is True

    def test_off_profile_disabled(self):
        cfg = ActivationConfig(consolidation_profile="off")
        assert cfg.inhibitory_spreading_enabled is False

    def test_observe_profile_disabled(self):
        cfg = ActivationConfig(consolidation_profile="observe")
        assert cfg.inhibitory_spreading_enabled is False
