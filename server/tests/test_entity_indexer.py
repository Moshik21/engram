"""Tests for structure-aware entity indexing."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.ingestion.entity_indexer import StructureAwareEntityIndexer
from engram.models.entity import Entity
from engram.models.relationship import Relationship


class FakeGraphStore:
    def __init__(self) -> None:
        self.entities = {
            "ent_alice": Entity(
                id="ent_alice",
                name="Alice",
                entity_type="Person",
                summary="Engineer",
                group_id="test",
            ),
            "ent_engram": Entity(
                id="ent_engram",
                name="Engram",
                entity_type="Project",
                summary="Memory runtime",
                group_id="test",
            ),
        }
        self.relationships = [
            Relationship(
                id="rel_knows",
                source_id="ent_alice",
                target_id="ent_engram",
                predicate="KNOWS_ABOUT",
                group_id="test",
            )
        ]

    async def get_relationships(self, entity_id: str, **_kwargs):
        return [
            rel
            for rel in self.relationships
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]

    async def get_entity(self, entity_id: str, _group_id: str):
        return self.entities.get(entity_id)


class FakeSearchIndex:
    def __init__(self) -> None:
        self.indexed: list[Entity] = []

    async def index_entity(self, entity: Entity) -> None:
        self.indexed.append(entity)


async def test_structure_aware_entity_indexer_adds_relationship_text():
    graph = FakeGraphStore()
    search = FakeSearchIndex()
    indexer = StructureAwareEntityIndexer(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(),
    )

    await indexer.index_entity(graph.entities["ent_alice"], "test")

    indexed = search.indexed[0]
    assert indexed.id == "ent_alice"
    assert indexed.entity_type == "Person"
    assert indexed.summary is None
    assert "Alice. Person. Engineer." in indexed.name
    assert "Alice knows about Engram" in indexed.name
