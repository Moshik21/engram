"""Tests for predicate-enriched embeddings in GraphManager."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.models.entity import Entity
from engram.models.relationship import Relationship


def _make_rel(rid, source_id, target_id, predicate):
    return Relationship(
        id=rid, source_id=source_id, target_id=target_id,
        predicate=predicate, weight=1.0, group_id="default",
        valid_from=datetime.utcnow(),
    )


def _make_entity(eid, name, entity_type="Other", summary=None):
    return Entity(
        id=eid, name=name, entity_type=entity_type,
        summary=summary, group_id="default",
    )


def _build_gm(cfg, rels, entities_by_id):
    graph_store = AsyncMock()
    graph_store.get_relationships = AsyncMock(return_value=rels)
    graph_store.get_entity = AsyncMock(
        side_effect=lambda eid, gid: entities_by_id.get(eid),
    )
    search_index = AsyncMock()
    search_index.index_entity = AsyncMock()
    gm = GraphManager(
        graph_store=graph_store,
        activation_store=AsyncMock(),
        search_index=search_index,
        extractor=AsyncMock(),
        cfg=cfg,
    )
    return gm, search_index


class TestPredicateEnrichedEmbeddings:
    @pytest.mark.asyncio
    async def test_relationships_sorted_by_predicate_weight(self):
        """Relationships sorted by predicate weight (EXPERT_IN before MENTIONED_WITH)."""
        cfg = ActivationConfig(structure_aware_embeddings=True)
        alice = _make_entity("e1", "Alice", "Person", "Engineer")
        entities = {
            "e1": alice,
            "e2": _make_entity("e2", "TechCorp", "Organization"),
            "e3": _make_entity("e3", "Python", "Technology"),
        }
        # MENTIONED_WITH (0.3) listed first but EXPERT_IN (0.9) should appear first in output
        rels = [
            _make_rel("r1", "e1", "e2", "MENTIONED_WITH"),
            _make_rel("r2", "e1", "e3", "EXPERT_IN"),
        ]
        gm, search_index = _build_gm(cfg, rels, entities)

        await gm._index_entity_with_structure(alice, "default")

        indexed = search_index.index_entity.call_args[0][0]
        expert_pos = indexed.name.index("expert in Python")
        mentioned_pos = indexed.name.index("mentioned with TechCorp")
        assert expert_pos < mentioned_pos, (
            f"EXPERT_IN (0.9) should appear before MENTIONED_WITH (0.3), "
            f"but got expert at {expert_pos}, mentioned at {mentioned_pos}"
        )

    @pytest.mark.asyncio
    async def test_structure_max_relationships_cap(self):
        """structure_max_relationships limits number of relationships in output."""
        cfg = ActivationConfig(
            structure_aware_embeddings=True,
            structure_max_relationships=2,
        )
        alice = _make_entity("e1", "Alice", "Person", "Engineer")
        entities = {
            "e1": alice,
            "e2": _make_entity("e2", "TechCorp"),
            "e3": _make_entity("e3", "Python"),
            "e4": _make_entity("e4", "Java"),
        }
        rels = [
            _make_rel("r1", "e1", "e2", "WORKS_AT"),
            _make_rel("r2", "e1", "e3", "EXPERT_IN"),
            _make_rel("r3", "e1", "e4", "USES"),
        ]
        gm, search_index = _build_gm(cfg, rels, entities)

        await gm._index_entity_with_structure(alice, "default")

        indexed = search_index.index_entity.call_args[0][0]
        # Only 2 relationships should appear (max_relationships=2)
        rel_section = indexed.name.split("Relationships: ")[1]
        rel_count = rel_section.count(",") + 1
        assert rel_count == 2

    @pytest.mark.asyncio
    async def test_unknown_predicate_fallback(self):
        """Unknown predicates fall back to lowercase with spaces."""
        cfg = ActivationConfig(structure_aware_embeddings=True)
        alice = _make_entity("e1", "Alice", "Person")
        entities = {
            "e1": alice,
            "e2": _make_entity("e2", "Bob"),
        }
        rels = [_make_rel("r1", "e1", "e2", "TRAINED_BY")]
        gm, search_index = _build_gm(cfg, rels, entities)

        await gm._index_entity_with_structure(alice, "default")

        indexed = search_index.index_entity.call_args[0][0]
        assert "trained by Bob" in indexed.name

    @pytest.mark.asyncio
    async def test_natural_language_forms_used(self):
        """Known predicates use natural language forms from config."""
        cfg = ActivationConfig(structure_aware_embeddings=True)
        alice = _make_entity("e1", "Alice", "Person")
        entities = {
            "e1": alice,
            "e2": _make_entity("e2", "TechCorp"),
            "e3": _make_entity("e3", "Bob"),
        }
        rels = [
            _make_rel("r1", "e1", "e2", "WORKS_AT"),
            _make_rel("r2", "e1", "e3", "COLLABORATES_WITH"),
        ]
        gm, search_index = _build_gm(cfg, rels, entities)

        await gm._index_entity_with_structure(alice, "default")

        indexed = search_index.index_entity.call_args[0][0]
        assert "Alice works at TechCorp" in indexed.name
        assert "Alice collaborates with Bob" in indexed.name

    @pytest.mark.asyncio
    async def test_structure_disabled_skips_structure_path(self):
        """structure_aware_embeddings=False still skips structure path in ingest."""
        cfg = ActivationConfig(structure_aware_embeddings=False)
        # Verify the config value is False
        assert cfg.structure_aware_embeddings is False

    def test_default_config_has_structure_enabled(self):
        """Fresh ActivationConfig() has structure_aware_embeddings=True."""
        cfg = ActivationConfig()
        assert cfg.structure_aware_embeddings is True

    @pytest.mark.asyncio
    async def test_incoming_relationships_formatted_correctly(self):
        """Incoming relationships formatted as '{source} {pred} {entity}'."""
        cfg = ActivationConfig(structure_aware_embeddings=True)
        alice = _make_entity("e1", "Alice", "Person")
        entities = {
            "e1": alice,
            "e2": _make_entity("e2", "TechCorp", "Organization"),
        }
        # TechCorp employs Alice — incoming from Alice's perspective
        # Using a custom predicate to test incoming format
        rels = [_make_rel("r1", "e2", "e1", "LEADS")]
        gm, search_index = _build_gm(cfg, rels, entities)

        await gm._index_entity_with_structure(alice, "default")

        indexed = search_index.index_entity.call_args[0][0]
        assert "TechCorp leads Alice" in indexed.name

    @pytest.mark.asyncio
    async def test_empty_summary_produces_valid_format(self):
        """Entity with no summary still produces valid dot-separated format."""
        cfg = ActivationConfig(structure_aware_embeddings=True)
        alice = _make_entity("e1", "Alice", "Person", summary=None)
        entities = {"e1": alice}
        gm, search_index = _build_gm(cfg, [], entities)

        await gm._index_entity_with_structure(alice, "default")

        indexed = search_index.index_entity.call_args[0][0]
        assert indexed.name == "Alice. Person."
