"""Tests for structure-aware embeddings in GraphManager."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.graph_manager import GraphManager
from engram.models.entity import Entity
from engram.models.relationship import Relationship


class TestStructureAwareEmbeddings:
    @pytest.mark.asyncio
    async def test_structure_text_with_relationships(self):
        """Structure-aware text includes relationship predicates."""
        cfg = ActivationConfig(structure_aware_embeddings=True)

        graph_store = AsyncMock()
        entity = Entity(
            id="e1",
            name="Alice",
            entity_type="Person",
            summary="Engineer at TechCorp",
            group_id="default",
        )
        graph_store.get_entity = AsyncMock(
            side_effect=lambda eid, gid: {
                "e1": entity,
                "e2": Entity(
                    id="e2",
                    name="TechCorp",
                    entity_type="Organization",
                    group_id="default",
                ),
                "e3": Entity(
                    id="e3",
                    name="Python",
                    entity_type="Technology",
                    group_id="default",
                ),
            }.get(eid),
        )

        rels = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                predicate="WORKS_AT",
                weight=1.0,
                group_id="default",
                valid_from=datetime.utcnow(),
            ),
            Relationship(
                id="r2",
                source_id="e1",
                target_id="e3",
                predicate="EXPERT_IN",
                weight=1.0,
                group_id="default",
                valid_from=datetime.utcnow(),
            ),
        ]
        graph_store.get_relationships = AsyncMock(return_value=rels)

        search_index = AsyncMock()
        search_index.index_entity = AsyncMock()

        gm = GraphManager(
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=search_index,
            extractor=AsyncMock(),
            cfg=cfg,
        )

        await gm._index_entity_with_structure(entity, "default")

        search_index.index_entity.assert_called_once()
        indexed_entity = search_index.index_entity.call_args[0][0]
        assert "Alice" in indexed_entity.name
        assert "Person" in indexed_entity.name
        assert "works at TechCorp" in indexed_entity.name
        assert "expert in Python" in indexed_entity.name
        # Verify sentence format with periods
        assert "Alice." in indexed_entity.name
        assert "Person." in indexed_entity.name

    @pytest.mark.asyncio
    async def test_structure_text_no_relationships(self):
        """Structure-aware text works with no relationships."""
        cfg = ActivationConfig(structure_aware_embeddings=True)

        graph_store = AsyncMock()
        entity = Entity(
            id="e1",
            name="Alice",
            entity_type="Person",
            summary="Engineer",
            group_id="default",
        )
        graph_store.get_relationships = AsyncMock(return_value=[])

        search_index = AsyncMock()
        search_index.index_entity = AsyncMock()

        gm = GraphManager(
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=search_index,
            extractor=AsyncMock(),
            cfg=cfg,
        )

        await gm._index_entity_with_structure(entity, "default")

        indexed_entity = search_index.index_entity.call_args[0][0]
        assert indexed_entity.name == "Alice. Person. Engineer."

    @pytest.mark.asyncio
    async def test_canonicalization_in_ingest(self):
        """Predicates are canonicalized during ingestion."""
        canonicalizer = PredicateCanonicalizer()
        assert canonicalizer.canonicalize("EMPLOYED_BY") == "WORKS_AT"
        assert canonicalizer.canonicalize("SKILLED_IN") == "EXPERT_IN"

    def test_default_canonicalizer_created(self):
        """GraphManager creates a default canonicalizer if none provided."""
        gm = GraphManager(
            graph_store=AsyncMock(),
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            extractor=AsyncMock(),
        )
        assert gm._canonicalizer is not None

    def test_custom_canonicalizer_used(self):
        """GraphManager uses provided canonicalizer."""
        custom = PredicateCanonicalizer(extra_mappings={"FOO": "BAR"})
        gm = GraphManager(
            graph_store=AsyncMock(),
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            extractor=AsyncMock(),
            canonicalizer=custom,
        )
        assert gm._canonicalizer is custom
