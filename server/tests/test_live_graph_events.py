"""Tests for live neural graph events: activation.access + enhanced graph.nodes_added."""

import pytest

from engram.events.bus import EventBus
from engram.extraction.extractor import ExtractionResult
from engram.graph_manager import GraphManager
from tests.conftest import MockExtractor


@pytest.mark.asyncio
class TestActivationAccessEvent:
    """Test that activation.access events fire during ingestion."""

    async def test_ingest_publishes_activation_access_events(
        self, graph_store, activation_store, search_index
    ):
        bus = EventBus()
        collected: list[dict] = []

        queue = bus.subscribe("default")

        extractor = MockExtractor(
            ExtractionResult(
                entities=[
                    {"name": "Alice", "entity_type": "Person", "summary": "A person"},
                    {"name": "Bob", "entity_type": "Person", "summary": "Another person"},
                ],
                relationships=[
                    {
                        "source": "Alice",
                        "target": "Bob",
                        "predicate": "KNOWS",
                        "weight": 1.0,
                    },
                ],
            )
        )

        manager = GraphManager(
            graph_store, activation_store, search_index, extractor, event_bus=bus
        )
        await manager.ingest_episode(
            content="Alice knows Bob", group_id="default", source="test"
        )

        # Drain events
        while not queue.empty():
            event = await queue.get()
            collected.append(event)

        # Filter activation.access events
        access_events = [e for e in collected if e["type"] == "activation.access"]

        # Should have one per entity (Alice + Bob)
        assert len(access_events) == 2

        # Check payload shape
        for ae in access_events:
            payload = ae["payload"]
            assert "entityId" in payload
            assert "name" in payload
            assert "entityType" in payload
            assert "activation" in payload
            assert payload["accessedVia"] == "ingest"

        # Check entity names
        names = {ae["payload"]["name"] for ae in access_events}
        assert names == {"Alice", "Bob"}

    async def test_recall_publishes_activation_access_events(
        self, graph_store, activation_store, search_index
    ):
        bus = EventBus()

        extractor = MockExtractor(
            ExtractionResult(
                entities=[
                    {"name": "Python", "entity_type": "Technology", "summary": "Language"},
                ],
                relationships=[],
            )
        )

        manager = GraphManager(
            graph_store, activation_store, search_index, extractor, event_bus=bus
        )
        await manager.ingest_episode(
            content="Using Python", group_id="default", source="test"
        )

        # Subscribe after ingest to only capture recall events
        queue = bus.subscribe("default")

        await manager.recall(query="Python", group_id="default")

        collected: list[dict] = []
        while not queue.empty():
            event = await queue.get()
            collected.append(event)

        access_events = [e for e in collected if e["type"] == "activation.access"]
        recall_events = [
            e for e in access_events if e["payload"]["accessedVia"] == "recall"
        ]
        assert len(recall_events) >= 1
        assert recall_events[0]["payload"]["name"] == "Python"


@pytest.mark.asyncio
class TestEnhancedGraphNodesAdded:
    """Test that graph.nodes_added includes full node/edge data."""

    async def test_graph_nodes_added_includes_nodes_and_edges(
        self, graph_store, activation_store, search_index
    ):
        bus = EventBus()

        queue = bus.subscribe("default")

        extractor = MockExtractor(
            ExtractionResult(
                entities=[
                    {"name": "React", "entity_type": "Technology", "summary": "UI library"},
                    {"name": "TypeScript", "entity_type": "Technology", "summary": "Language"},
                ],
                relationships=[
                    {
                        "source": "React",
                        "target": "TypeScript",
                        "predicate": "USES",
                        "weight": 0.9,
                    },
                ],
            )
        )

        manager = GraphManager(
            graph_store, activation_store, search_index, extractor, event_bus=bus
        )
        await manager.ingest_episode(
            content="React with TypeScript", group_id="default", source="test"
        )

        collected: list[dict] = []
        while not queue.empty():
            event = await queue.get()
            collected.append(event)

        graph_events = [e for e in collected if e["type"] == "graph.nodes_added"]
        assert len(graph_events) == 1

        payload = graph_events[0]["payload"]

        # Backward-compatible fields
        assert payload["entity_count"] == 2
        assert payload["relationship_count"] == 1
        assert set(payload["new_entities"]) == {"React", "TypeScript"}

        # New full node data
        assert "nodes" in payload
        assert len(payload["nodes"]) == 2
        node_names = {n["name"] for n in payload["nodes"]}
        assert node_names == {"React", "TypeScript"}

        # Verify node shape
        for node in payload["nodes"]:
            assert "id" in node
            assert "name" in node
            assert "entityType" in node
            assert "summary" in node
            assert "activationCurrent" in node

        # New full edge data
        assert "edges" in payload
        assert len(payload["edges"]) == 1
        edge = payload["edges"][0]
        assert edge["predicate"] == "USES"
        assert "id" in edge
        assert "source" in edge
        assert "target" in edge
        assert "weight" in edge
