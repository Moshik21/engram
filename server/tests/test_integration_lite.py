"""Integration test: end-to-end lite mode flow."""

from uuid import uuid4

import pytest

from engram.graph_manager import GraphManager


@pytest.fixture
def gid():
    return f"test_{uuid4().hex[:8]}"


@pytest.mark.asyncio
class TestLiteModeIntegration:
    async def test_ingest_and_recall(self, graph_manager: GraphManager, gid):
        """Ingest episodes and verify entities are searchable."""
        # Ingest an episode (uses mock extractor returning Python + FastAPI)
        episode_id = await graph_manager.ingest_episode(
            content="I'm building a web API using FastAPI and Python.",
            group_id=gid,
            source="test",
        )
        assert episode_id.startswith("ep_")

        # Recall should find the entities
        results = await graph_manager.recall(query="Python", group_id=gid)
        assert len(results) >= 1
        entity_names = [r["entity"]["name"] for r in results if "entity" in r]
        assert "Python" in entity_names

    async def test_multiple_episodes(self, graph_manager: GraphManager, gid):
        """Ingest multiple episodes, verify dedup and relationships."""
        await graph_manager.ingest_episode(
            content="First conversation about Python and FastAPI",
            group_id=gid,
        )
        await graph_manager.ingest_episode(
            content="Second conversation about the same topics",
            group_id=gid,
        )
        await graph_manager.ingest_episode(
            content="Third episode with more detail",
            group_id=gid,
        )

        # Should have entities from the mock extractor
        results = await graph_manager.recall(query="FastAPI", group_id=gid)
        assert len(results) >= 1

    async def test_entity_dedup(self, graph_manager: GraphManager, gid):
        """Same entity name from multiple episodes should not duplicate."""
        await graph_manager.ingest_episode(content="Working with Python", group_id=gid)
        await graph_manager.ingest_episode(content="More Python work", group_id=gid)

        # Check that we don't have duplicate entities
        store = graph_manager._graph
        entities = await store.find_entities(name="Python", group_id=gid)
        assert len(entities) == 1

    async def test_stats_after_ingestion(self, graph_manager: GraphManager, gid):
        """Stats should reflect ingested data."""
        await graph_manager.ingest_episode(content="Testing stats", group_id=gid)
        stats = await graph_manager._graph.get_stats(group_id=gid)
        assert stats["entities"] >= 1
        assert stats["episodes"] >= 1

    async def test_fuzzy_dedup_across_episodes(
        self, graph_store, activation_store, search_index, gid,
    ):
        """Fuzzy dedup should merge 'Python' and 'python' across episodes."""
        from engram.extraction.extractor import ExtractionResult
        from tests.conftest import MockExtractor

        # First episode creates "Python"
        ext1 = MockExtractor(
            ExtractionResult(
                entities=[
                    {"name": "Python", "entity_type": "Technology", "summary": "Language v1"},
                ],
                relationships=[],
            )
        )
        manager1 = GraphManager(graph_store, activation_store, search_index, ext1)
        await manager1.ingest_episode(content="ep1", group_id=gid)

        # Second episode references "python" (lowercase)
        ext2 = MockExtractor(
            ExtractionResult(
                entities=[
                    {"name": "python", "entity_type": "Technology", "summary": "Language v2"},
                ],
                relationships=[],
            )
        )
        manager2 = GraphManager(graph_store, activation_store, search_index, ext2)
        await manager2.ingest_episode(content="ep2", group_id=gid)

        # Should be one entity, not two
        entities = await graph_store.find_entities(group_id=gid)
        python_entities = [e for e in entities if "python" in e.name.lower()]
        assert len(python_entities) == 1
        # Summary should be merged
        assert "v1" in python_entities[0].summary
        assert "v2" in python_entities[0].summary

    async def test_recall_with_activation_scores(self, graph_manager: GraphManager, gid):
        """Recall results include score_breakdown."""
        await graph_manager.ingest_episode(
            content="Testing activation scores",
            group_id=gid,
        )
        results = await graph_manager.recall(query="Python", group_id=gid)
        assert len(results) >= 1
        r = results[0]
        assert "score_breakdown" in r
        assert "semantic" in r["score_breakdown"]
        assert "activation" in r["score_breakdown"]
        assert "edge_proximity" in r["score_breakdown"]

    async def test_ingest_records_access(self, graph_manager: GraphManager, gid):
        """After ingest, entities have activation state."""
        await graph_manager.ingest_episode(
            content="Testing access recording",
            group_id=gid,
        )
        # Python and FastAPI entities should have activation state
        entities = await graph_manager._graph.find_entities(group_id=gid)
        for entity in entities:
            state = await graph_manager._activation.get_activation(entity.id)
            assert state is not None
            assert len(state.access_history) >= 1

    async def test_recall_records_access(self, graph_manager: GraphManager, gid):
        """After recall, returned entities have updated access history."""
        await graph_manager.ingest_episode(
            content="Testing recall access",
            group_id=gid,
        )
        results = await graph_manager.recall(query="Python", group_id=gid)
        assert len(results) >= 1

        entity_id = results[0]["entity"]["id"]
        state = await graph_manager._activation.get_activation(entity_id)
        # Should have at least 2 accesses: one from ingest, one from recall
        assert state.access_count >= 2

    async def test_activation_boosts_frequently_mentioned(
        self, graph_store, activation_store, search_index, gid
    ):
        """Entity mentioned in 3 episodes should have higher activation than one mentioned once."""
        from engram.extraction.extractor import ExtractionResult
        from tests.conftest import MockExtractor

        # Ingest "Python" 3 times
        for _ in range(3):
            ext = MockExtractor(
                ExtractionResult(
                    entities=[
                        {"name": "Python", "entity_type": "Technology", "summary": "Language"},
                    ],
                    relationships=[],
                )
            )
            mgr = GraphManager(graph_store, activation_store, search_index, ext)
            await mgr.ingest_episode(content="ep", group_id=gid)

        # Ingest "Rust" once
        ext2 = MockExtractor(
            ExtractionResult(
                entities=[
                    {"name": "Rust", "entity_type": "Technology", "summary": "Language"},
                ],
                relationships=[],
            )
        )
        mgr2 = GraphManager(graph_store, activation_store, search_index, ext2)
        await mgr2.ingest_episode(content="ep", group_id=gid)

        # Python should have more accesses
        entities = await graph_store.find_entities(group_id=gid)
        python_ent = next(e for e in entities if e.name == "Python")
        rust_ent = next(e for e in entities if e.name == "Rust")

        py_state = await activation_store.get_activation(python_ent.id)
        rust_state = await activation_store.get_activation(rust_ent.id)

        assert py_state.access_count >= 3
        assert rust_state.access_count == 1

    async def test_recall_empty_returns_empty(self, graph_manager: GraphManager, gid):
        """Empty query on empty store still works."""
        results = await graph_manager.recall(query="nonexistent", group_id=gid)
        assert results == []

    async def test_recall_scores_descending(self, graph_manager: GraphManager, gid):
        """Results sorted by score descending."""
        await graph_manager.ingest_episode(
            content="Testing score ordering",
            group_id=gid,
        )
        results = await graph_manager.recall(query="Python FastAPI", group_id=gid)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]
