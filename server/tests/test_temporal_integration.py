"""Integration tests for temporal contradiction pipeline."""

from uuid import uuid4

import pytest

from engram.config import HelixDBConfig
from engram.extraction.extractor import ExtractionResult
from engram.graph_manager import GraphManager
from engram.models.entity import Entity
from tests.conftest import _test_helix_config


@pytest.fixture
def helix_config(tmp_path):
    """Per-test native data dir.

    These tests write to group ``default`` with recurring entity names; on the
    shared session-wide native data dir they collide with sibling tests and
    across runs. Isolate each test on its own dir so projection sees a clean
    graph. Falls back to the shared HTTP lane when native is unavailable.
    """
    base = _test_helix_config()
    if base.transport != "native":
        return base
    data_dir = tmp_path / "helix"
    data_dir.mkdir(parents=True, exist_ok=True)
    return HelixDBConfig(transport="native", data_dir=str(data_dir), verbose=False)


class MockTemporalExtractor:
    """Extractor that returns canned results for temporal testing."""

    def __init__(self):
        self._call_count = 0
        self._results = []

    def add_result(self, result: ExtractionResult):
        self._results.append(result)

    async def extract(self, text: str) -> ExtractionResult:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._results):
            return self._results[idx]
        return ExtractionResult(entities=[], relationships=[])


@pytest.mark.asyncio
class TestTemporalContradictionPipeline:
    async def test_location_contradiction(self, graph_store, activation_store, search_index):
        """'lives in Mesa' then 'moved to Denver' should invalidate Mesa."""
        # A per-run unique person keeps a persistent native store from accruing
        # cross-run LOCATED_IN residue on a shared "Alex". The relationship
        # endpoints must already exist as committed entities or projection drops
        # the edge as missing_entities (narrow-extracted proper nouns defer below
        # the commit threshold), so seed Mesa/Denver up front.
        gid = "default"
        person = f"Alex_{uuid4().hex[:8]}"
        for city in ("Mesa", "Denver"):
            await graph_store.create_entity(
                Entity(
                    id=f"ent_{city.lower()}_{uuid4().hex[:8]}",
                    name=city,
                    entity_type="Location",
                    group_id=gid,
                )
            )

        extractor = MockTemporalExtractor()

        # First episode: lives in Mesa
        extractor.add_result(
            ExtractionResult(
                entities=[
                    {"name": person, "entity_type": "Person", "summary": "Lives in Mesa"},
                    {"name": "Mesa", "entity_type": "Location", "summary": "City in Arizona"},
                ],
                relationships=[
                    {
                        "source": person,
                        "target": "Mesa",
                        "predicate": "LIVES_IN",
                        "weight": 1.0,
                    },
                ],
            )
        )

        # Second episode: moved to Denver
        extractor.add_result(
            ExtractionResult(
                entities=[
                    {"name": person, "entity_type": "Person", "summary": "Moved to Denver"},
                    {"name": "Denver", "entity_type": "Location", "summary": "City in Colorado"},
                ],
                relationships=[
                    {
                        "source": person,
                        "target": "Denver",
                        "predicate": "LIVES_IN",
                        "weight": 1.0,
                        "temporal_hint": "last month",
                    },
                ],
            )
        )

        manager = GraphManager(graph_store, activation_store, search_index, extractor)

        await manager.ingest_episode("lives in Mesa", group_id=gid)
        await manager.ingest_episode("moved to Denver", group_id=gid)

        # Find the person's entity
        entities = await graph_store.find_entities(name=person, group_id=gid)
        assert len(entities) == 1
        alex_id = entities[0].id

        # Check LOCATED_IN relationships (LIVES_IN canonicalized to LOCATED_IN)
        all_rels = await graph_store.get_relationships(
            alex_id, direction="outgoing", predicate="LOCATED_IN", active_only=False
        )
        active_rels = [r for r in all_rels if r.valid_to is None]
        invalidated_rels = [r for r in all_rels if r.valid_to is not None]

        # Should have 2 total, 1 active (Denver), 1 invalidated (Mesa)
        assert len(all_rels) == 2
        assert len(active_rels) == 1
        assert len(invalidated_rels) == 1

        # Active should be Denver
        denver = await graph_store.get_entity(active_rels[0].target_id, gid)
        assert denver is not None
        assert denver.name == "Denver"

    async def test_temporal_hint_resolved_in_pipeline(
        self, graph_store, activation_store, search_index
    ):
        """Temporal hints should be resolved to dates in the pipeline."""
        extractor = MockTemporalExtractor()
        extractor.add_result(
            ExtractionResult(
                entities=[
                    {"name": "Alice", "entity_type": "Person"},
                    {"name": "Acme", "entity_type": "Organization"},
                ],
                relationships=[
                    {
                        "source": "Alice",
                        "target": "Acme",
                        "predicate": "WORKS_AT",
                        "weight": 1.0,
                        "valid_from": "2024-01-15",
                    },
                ],
            )
        )

        manager = GraphManager(graph_store, activation_store, search_index, extractor)
        await manager.ingest_episode("Alice works at Acme since Jan 2024", group_id="default")

        # Check that valid_from was set
        entities = await graph_store.find_entities(name="Alice", group_id="default")
        assert len(entities) == 1
        rels = await graph_store.get_relationships(entities[0].id, direction="outgoing")
        assert len(rels) >= 1
        works_at = [r for r in rels if r.predicate == "WORKS_AT"]
        assert len(works_at) == 1
        assert works_at[0].valid_from is not None
        assert works_at[0].valid_from.year == 2024
