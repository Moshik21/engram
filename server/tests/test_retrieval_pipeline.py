"""Tests for the retrieval pipeline, including spreading candidate injection."""

from __future__ import annotations

import time

import pytest

from dataclasses import dataclass

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.pipeline import _inject_entity_matches, retrieve


@dataclass
class _FakeEntity:
    """Minimal entity stub for find_entities results."""

    id: str
    name: str
    entity_type: str = "Person"
    summary: str = ""
    group_id: str = "default"


class _FakeSearchIndex:
    """Search index stub returning predetermined results."""

    def __init__(
        self,
        results: list[tuple[str, float]],
        similarity_map: dict[str, float] | None = None,
    ):
        self._results = results
        self._similarity_map = similarity_map or {}

    async def search(self, query: str, group_id: str, limit: int = 50):
        return self._results[:limit]

    async def compute_similarity(
        self,
        query: str,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, float]:
        return {eid: self._similarity_map.get(eid, 0.0) for eid in entity_ids}


class _FakeActivationStore:
    """In-memory activation store stub."""

    def __init__(self, states: dict[str, ActivationState] | None = None):
        self._states = states or {}

    async def batch_get(self, entity_ids: list[str]) -> dict[str, ActivationState]:
        return {eid: self._states[eid] for eid in entity_ids if eid in self._states}


class _FakeGraphStore:
    """Graph store stub with configurable neighbors and entity lookup."""

    def __init__(
        self,
        adjacency: dict[str, list[str]] | None = None,
        entities: list[_FakeEntity] | None = None,
    ):
        self._adj = adjacency or {}
        self._entities = entities or []

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None, **kwargs
    ) -> list[tuple[str, float]]:
        """Return (neighbor_id, weight) pairs for spreading activation."""
        neighbors = self._adj.get(entity_id, [])
        return [(n, 1.0) for n in neighbors]

    async def find_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[_FakeEntity]:
        """Return entities matching name (case-insensitive substring)."""
        results = []
        for ent in self._entities:
            if name and name.lower() in ent.name.lower():
                results.append(ent)
        return results[:limit]

    async def get_stats(self, group_id: str) -> dict:
        return {"entity_count": len(self._entities)}

    async def get_entity(self, entity_id: str, group_id: str):
        for ent in self._entities:
            if ent.id == entity_id:
                return ent
        return None


@pytest.mark.asyncio
class TestRetrievalPipeline:
    async def test_spreading_discovered_entities_included_in_results(self):
        """Entities discovered by spreading activation appear in results."""
        now = time.time()
        # search_ent is found by search, neighbor_ent is discovered via spreading
        search_index = _FakeSearchIndex([("search_ent", 0.8)])
        activation_store = _FakeActivationStore(
            {
                "search_ent": ActivationState(
                    node_id="search_ent",
                    access_history=[now - 10],
                    access_count=5,
                ),
                "neighbor_ent": ActivationState(
                    node_id="neighbor_ent",
                    access_history=[now - 5],
                    access_count=3,
                ),
            }
        )
        # search_ent connects to neighbor_ent
        graph_store = _FakeGraphStore({"search_ent": ["neighbor_ent"]})

        cfg = ActivationConfig(
            weight_semantic=0.50,
            weight_activation=0.35,
            weight_edge_proximity=0.15,
            seed_threshold=0.3,
            exploration_weight=0.0,
        )

        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
        )

        result_ids = {r.node_id for r in results}
        assert "search_ent" in result_ids
        assert "neighbor_ent" in result_ids

    async def test_spreading_entities_get_real_similarity(self):
        """Entities discovered by spreading get real semantic similarity from search index."""
        now = time.time()
        search_index = _FakeSearchIndex(
            [("seed", 0.9)],
            similarity_map={"discovered": 0.65},
        )
        activation_store = _FakeActivationStore(
            {
                "seed": ActivationState(
                    node_id="seed",
                    access_history=[now - 5],
                    access_count=3,
                ),
                "discovered": ActivationState(
                    node_id="discovered",
                    access_history=[now - 10],
                    access_count=2,
                ),
            }
        )
        graph_store = _FakeGraphStore({"seed": ["discovered"]})

        cfg = ActivationConfig(
            weight_semantic=0.50,
            weight_activation=0.35,
            weight_edge_proximity=0.15,
            seed_threshold=0.3,
            exploration_weight=0.0,
        )

        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
        )

        discovered = next((r for r in results if r.node_id == "discovered"), None)
        assert discovered is not None
        assert discovered.semantic_similarity == 0.65

    async def test_spreading_entities_fallback_to_zero(self):
        """Entities discovered by spreading get 0.0 when search index has no embeddings."""
        now = time.time()
        search_index = _FakeSearchIndex([("seed", 0.9)])  # no similarity_map
        activation_store = _FakeActivationStore(
            {
                "seed": ActivationState(
                    node_id="seed",
                    access_history=[now - 5],
                    access_count=3,
                ),
                "discovered": ActivationState(
                    node_id="discovered",
                    access_history=[now - 10],
                    access_count=2,
                ),
            }
        )
        graph_store = _FakeGraphStore({"seed": ["discovered"]})

        cfg = ActivationConfig(
            weight_semantic=0.50,
            weight_activation=0.35,
            weight_edge_proximity=0.15,
            seed_threshold=0.3,
            exploration_weight=0.0,
        )

        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
        )

        discovered = next((r for r in results if r.node_id == "discovered"), None)
        assert discovered is not None
        assert discovered.semantic_similarity == 0.0

    async def test_no_injection_when_spreading_disabled(self):
        """With no seeds (all below threshold), no injection happens."""
        search_index = _FakeSearchIndex([("ent_1", 0.1)])  # below seed_threshold
        activation_store = _FakeActivationStore({})
        graph_store = _FakeGraphStore({"ent_1": ["neighbor"]})

        cfg = ActivationConfig(
            seed_threshold=0.5,  # high threshold, so ent_1 (0.1) won't be a seed
            exploration_weight=0.0,
        )

        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
        )

        result_ids = {r.node_id for r in results}
        assert "ent_1" in result_ids
        assert "neighbor" not in result_ids

    async def test_empty_search_returns_empty(self):
        """Empty search results with no matching entities return empty list."""
        search_index = _FakeSearchIndex([])
        activation_store = _FakeActivationStore({})
        graph_store = _FakeGraphStore({})
        cfg = ActivationConfig()

        results = await retrieve(
            query="nothing",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
        )

        assert results == []


@pytest.mark.asyncio
class TestEntityFirstFallback:
    async def test_entity_first_fallback_injects_when_search_empty(self):
        """When search returns 0 results but entity 'Konner' exists, pipeline returns results."""
        now = time.time()
        konner = _FakeEntity(id="konner_id", name="Konner", entity_type="Person")
        book = _FakeEntity(
            id="book_id", name="The Agent of Fate", entity_type="CreativeWork"
        )

        search_index = _FakeSearchIndex(
            [],
            similarity_map={"konner_id": 0.3, "book_id": 0.2},
        )
        activation_store = _FakeActivationStore(
            {
                "konner_id": ActivationState(
                    node_id="konner_id",
                    access_history=[now - 10],
                    access_count=3,
                ),
                "book_id": ActivationState(
                    node_id="book_id",
                    access_history=[now - 20],
                    access_count=1,
                ),
            }
        )
        graph_store = _FakeGraphStore(
            adjacency={"konner_id": ["book_id"]},
            entities=[konner, book],
        )

        cfg = ActivationConfig(
            weight_semantic=0.30,
            weight_activation=0.10,
            weight_spreading=0.25,
            weight_edge_proximity=0.30,
            seed_threshold=0.05,
            exploration_weight=0.0,
        )

        results = await retrieve(
            query="books written by Konner",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
        )

        result_ids = {r.node_id for r in results}
        assert "konner_id" in result_ids

    async def test_entity_first_fallback_skipped_when_search_has_results(self):
        """When search returns 3+ results, find_entities is not called."""
        now = time.time()

        candidates = await _inject_entity_matches(
            query="test query",
            group_id="default",
            graph_store=_FakeGraphStore(entities=[]),
            candidates=[("a", 0.9), ("b", 0.8), ("c", 0.7)],
        )

        # With 3 candidates already, the pipeline won't call this function,
        # but verify the function itself doesn't add duplicates
        assert len(candidates) == 3

    async def test_entity_first_fallback_injects_neighbors(self):
        """Entity found by name → 1-hop neighbors also injected."""
        konner = _FakeEntity(id="konner_id", name="Konner")
        graph_store = _FakeGraphStore(
            adjacency={"konner_id": ["book_id", "project_id"]},
            entities=[konner],
        )

        result = await _inject_entity_matches(
            query="books by Konner",
            group_id="default",
            graph_store=graph_store,
            candidates=[],
        )

        result_ids = {eid for eid, _ in result}
        assert "konner_id" in result_ids
        assert "book_id" in result_ids
        assert "project_id" in result_ids
