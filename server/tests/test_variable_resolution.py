"""Tests for Feature 5: Variable-Resolution Context Rendering."""

from __future__ import annotations

import pytest

from engram.retrieval.scorer import ScoredResult


class TestScoredResultHopDistance:
    """Test hop_distance field on ScoredResult."""

    def test_default_hop_distance_is_none(self):
        sr = ScoredResult(
            node_id="ent_1",
            score=0.8,
            semantic_similarity=0.5,
            activation=0.3,
            spreading=0.1,
            edge_proximity=1.0,
        )
        assert sr.hop_distance is None

    def test_seed_has_hop_distance_zero(self):
        sr = ScoredResult(
            node_id="ent_1",
            score=0.8,
            semantic_similarity=0.5,
            activation=0.3,
            spreading=0.1,
            edge_proximity=1.0,
            hop_distance=0,
        )
        assert sr.hop_distance == 0

    def test_hop_distance_preserved(self):
        sr = ScoredResult(
            node_id="ent_2",
            score=0.5,
            semantic_similarity=0.3,
            activation=0.2,
            spreading=0.1,
            edge_proximity=0.25,
            hop_distance=2,
        )
        assert sr.hop_distance == 2


class TestScoreCandidatesHopDistance:
    """Test that score_candidates populates hop_distance correctly."""

    def test_seed_gets_hop_zero(self):
        from engram.config import ActivationConfig
        from engram.retrieval.scorer import score_candidates

        cfg = ActivationConfig()
        results = score_candidates(
            candidates=[("ent_1", 0.8)],
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids={"ent_1"},
            activation_states={},
            now=1000.0,
            cfg=cfg,
        )
        assert len(results) == 1
        assert results[0].hop_distance == 0

    def test_spreading_discovered_gets_hop_distance(self):
        from engram.config import ActivationConfig
        from engram.retrieval.scorer import score_candidates

        cfg = ActivationConfig()
        results = score_candidates(
            candidates=[("ent_1", 0.8), ("ent_2", 0.3)],
            spreading_bonuses={"ent_2": 0.5},
            hop_distances={"ent_2": 2},
            seed_node_ids={"ent_1"},
            activation_states={},
            now=1000.0,
            cfg=cfg,
        )
        by_id = {r.node_id: r for r in results}
        assert by_id["ent_1"].hop_distance == 0
        assert by_id["ent_2"].hop_distance == 2

    def test_unreachable_gets_none(self):
        from engram.config import ActivationConfig
        from engram.retrieval.scorer import score_candidates

        cfg = ActivationConfig()
        results = score_candidates(
            candidates=[("ent_1", 0.8), ("ent_3", 0.1)],
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids={"ent_1"},
            activation_states={},
            now=1000.0,
            cfg=cfg,
        )
        by_id = {r.node_id: r for r in results}
        assert by_id["ent_1"].hop_distance == 0
        assert by_id["ent_3"].hop_distance is None


class TestEntityToContextDataDetailLevel:
    """Test _entity_to_context_data with different detail levels."""

    @pytest.fixture
    def graph_manager(self):
        """Create a minimal GraphManager for testing."""
        from unittest.mock import AsyncMock, MagicMock

        from engram.config import ActivationConfig
        from engram.graph_manager import GraphManager

        graph = AsyncMock()
        activation = AsyncMock()
        search = AsyncMock()
        extractor = MagicMock()
        cfg = ActivationConfig()

        gm = GraphManager(graph, activation, search, extractor, cfg)
        return gm

    @pytest.mark.asyncio
    async def test_mention_returns_minimal(self, graph_manager):
        """Mention level should return only name and type."""
        ed = await graph_manager._entity_to_context_data(
            "ent_1",
            "Python",
            "Technology",
            "A programming language",
            "default",
            1000.0,
            detail_level="mention",
        )
        assert ed["name"] == "Python"
        assert ed["type"] == "Technology"
        assert ed["detail_level"] == "mention"
        assert ed["facts"] == []
        assert ed["summary"] is None
        assert ed["attributes"] is None

    @pytest.mark.asyncio
    async def test_summary_returns_limited_facts(self, graph_manager):
        """Summary level should return up to 2 facts, no attributes."""
        from engram.models.activation import ActivationState
        from engram.models.relationship import Relationship

        graph_manager._activation.get_activation.return_value = ActivationState(
            node_id="ent_1",
            access_history=[900.0, 950.0],
            access_count=2,
        )
        # Return 5 relationships - summary should only use 2
        rels = [
            Relationship(
                id=f"r{i}",
                source_id="ent_1",
                target_id=f"ent_{i + 10}",
                predicate=f"PRED_{i}",
                group_id="default",
            )
            for i in range(5)
        ]
        graph_manager._graph.get_relationships.return_value = rels
        graph_manager._graph.get_entity.return_value = None

        async def resolve_name(eid, gid):
            return eid

        graph_manager.resolve_entity_name = resolve_name

        ed = await graph_manager._entity_to_context_data(
            "ent_1",
            "Python",
            "Technology",
            "A programming language",
            "default",
            1000.0,
            detail_level="summary",
        )
        assert ed["detail_level"] == "summary"
        assert len(ed["facts"]) == 2
        assert ed["attributes"] is None
        assert ed["summary"] == "A programming language"

    @pytest.mark.asyncio
    async def test_full_returns_everything(self, graph_manager):
        """Full level should return attributes and up to 5 facts."""
        from engram.models.activation import ActivationState
        from engram.models.entity import Entity
        from engram.models.relationship import Relationship

        graph_manager._activation.get_activation.return_value = ActivationState(
            node_id="ent_1",
            access_history=[900.0],
            access_count=1,
        )
        rels = [
            Relationship(
                id=f"r{i}",
                source_id="ent_1",
                target_id=f"ent_{i + 10}",
                predicate=f"PRED_{i}",
                group_id="default",
            )
            for i in range(7)
        ]
        graph_manager._graph.get_relationships.return_value = rels
        entity = Entity(
            id="ent_1",
            name="Achilles Injury",
            entity_type="HealthCondition",
            attributes={"status": "recovering", "duration": "3 weeks"},
        )
        graph_manager._graph.get_entity.return_value = entity

        async def resolve_name(eid, gid):
            return eid

        graph_manager.resolve_entity_name = resolve_name

        ed = await graph_manager._entity_to_context_data(
            "ent_1",
            "Achilles Injury",
            "HealthCondition",
            "Tweaked Achilles tendon",
            "default",
            1000.0,
            detail_level="full",
        )
        assert ed["detail_level"] == "full"
        assert len(ed["facts"]) == 5
        assert ed["attributes"] == {"status": "recovering", "duration": "3 weeks"}


class TestRenderTierVariableResolution:
    """Test _render_tier with variable resolution entities."""

    def test_mention_renders_name_type_only(self):
        from engram.graph_manager import GraphManager

        entities = [
            {
                "name": "Python",
                "type": "Technology",
                "detail_level": "mention",
                "activation": 0.0,
                "facts": [],
                "attributes": None,
                "summary": None,
            },
        ]
        text = GraphManager._render_tier("## Test", entities, [])
        assert "Python (Technology)" in text
        assert "act=" not in text

    def test_summary_renders_without_attributes(self):
        from engram.graph_manager import GraphManager

        entities = [
            {
                "name": "FastAPI",
                "type": "Technology",
                "detail_level": "summary",
                "activation": 0.75,
                "summary": "Web framework",
                "facts": ["FastAPI USES Python"],
                "attributes": None,
            },
        ]
        text = GraphManager._render_tier("## Test", entities, [])
        assert "FastAPI (Technology, act=0.75) — Web framework" in text
        assert "FastAPI USES Python" in text

    def test_full_renders_with_attributes(self):
        from engram.graph_manager import GraphManager

        entities = [
            {
                "name": "Alex",
                "type": "Person",
                "detail_level": "full",
                "activation": 0.95,
                "summary": "Software engineer",
                "facts": ["Alex WORKS_AT Company"],
                "attributes": {"status": "active", "role": "engineer"},
            },
        ]
        text = GraphManager._render_tier("## Test", entities, [])
        assert "Alex (Person, act=0.95)" in text
        assert "status: active" in text
        assert "role: engineer" in text
        assert "Alex WORKS_AT Company" in text

    def test_mixed_resolution_levels(self):
        from engram.graph_manager import GraphManager

        entities = [
            {
                "name": "Alex",
                "type": "Person",
                "detail_level": "full",
                "activation": 0.95,
                "summary": "Software engineer",
                "facts": ["Alex WORKS_AT Acme"],
                "attributes": {"role": "lead"},
            },
            {
                "name": "FastAPI",
                "type": "Technology",
                "detail_level": "summary",
                "activation": 0.6,
                "summary": "Web framework",
                "facts": ["FastAPI USES Python"],
                "attributes": None,
            },
            {
                "name": "Redis",
                "type": "Technology",
                "detail_level": "mention",
                "activation": 0.0,
                "facts": [],
                "attributes": None,
                "summary": None,
            },
        ]
        text = GraphManager._render_tier("## Test", entities, [])
        # Full: has attributes and facts inline
        assert "role: lead" in text
        assert "Alex WORKS_AT Acme" in text
        # Summary: has summary but no attributes
        assert "Web framework" in text
        assert "FastAPI USES Python" in text
        # Mention: just name + type
        assert "Redis (Technology)" in text
        # Mention should NOT have activation
        lines = text.split("\n")
        redis_line = [line for line in lines if "Redis" in line][0]
        assert "act=" not in redis_line

    def test_default_detail_level_is_full(self):
        """Entities without explicit detail_level default to full rendering."""
        from engram.graph_manager import GraphManager

        entities = [
            {
                "name": "Test",
                "type": "Concept",
                "activation": 0.5,
                "summary": "A test",
                "facts": [],
                "attributes": None,
            },
        ]
        text = GraphManager._render_tier("## Test", entities, [])
        # Should render with activation (full mode)
        assert "act=0.50" in text
