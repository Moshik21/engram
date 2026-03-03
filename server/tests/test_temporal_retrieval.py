"""Tests for temporal bypass in retrieval and structure-aware indexing."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from engram.activation.spreading import identify_seeds
from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.relationship import Relationship
from engram.retrieval.pipeline import retrieve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_search_index(results=None):
    idx = AsyncMock()
    idx.search = AsyncMock(return_value=results or [])
    idx.compute_similarity = AsyncMock(return_value={})
    idx._embeddings_enabled = False
    return idx


def _mock_graph_store(neighbors=None):
    store = AsyncMock()
    store.get_active_neighbors_with_weights = AsyncMock(
        return_value=neighbors or [],
    )
    store.get_entity = AsyncMock(
        return_value=Entity(
            id="e1",
            name="Test",
            entity_type="Thing",
            summary="A test entity",
            group_id="default",
        )
    )
    return store


def _mock_activation_store(states=None, top_activated=None):
    store = AsyncMock()
    store.batch_get = AsyncMock(return_value=states or {})
    store.get_activation = AsyncMock(return_value=None)
    store.set_activation = AsyncMock()
    store.get_top_activated = AsyncMock(
        return_value=top_activated or [],
    )
    return store


def _make_state(entity_id: str, timestamps: list[float]) -> ActivationState:
    state = ActivationState(node_id=entity_id)
    state.access_history = list(timestamps)
    state.access_count = len(timestamps)
    if timestamps:
        state.last_accessed = max(timestamps)
    return state


# ---------------------------------------------------------------------------
# Phase 2: identify_seeds temporal_mode
# ---------------------------------------------------------------------------


class TestIdentifySeedsTemporal:
    def test_temporal_mode_seeds_with_zero_sem_sim(self):
        """In temporal_mode, candidates with sem_sim=0.0 become seeds."""
        now = time.time()
        cfg = ActivationConfig(seed_threshold=0.3)
        candidates = [("e1", 0.0), ("e2", 0.0)]
        states = {
            "e1": _make_state("e1", [now - 10]),
            "e2": _make_state("e2", [now - 100]),
        }

        seeds = identify_seeds(candidates, states, now, cfg, temporal_mode=True)
        seed_ids = {nid for nid, _ in seeds}
        assert "e1" in seed_ids
        assert "e2" in seed_ids

    def test_default_mode_rejects_low_sem_sim(self):
        """Default mode filters out candidates below seed_threshold."""
        now = time.time()
        cfg = ActivationConfig(seed_threshold=0.3)
        candidates = [("e1", 0.0), ("e2", 0.5)]
        states = {
            "e1": _make_state("e1", [now - 10]),
            "e2": _make_state("e2", [now - 10]),
        }

        seeds = identify_seeds(candidates, states, now, cfg, temporal_mode=False)
        seed_ids = {nid for nid, _ in seeds}
        assert "e1" not in seed_ids
        assert "e2" in seed_ids

    def test_temporal_energy_from_activation_only(self):
        """In temporal_mode, energy = max(act, 0.15), not sem * max(act, 0.15)."""
        now = time.time()
        cfg = ActivationConfig(seed_threshold=0.3)
        candidates = [("e1", 0.0)]
        states = {
            "e1": _make_state("e1", [now - 1]),  # very recent -> high activation
        }

        seeds = identify_seeds(candidates, states, now, cfg, temporal_mode=True)
        assert len(seeds) == 1
        _nid, energy = seeds[0]
        # Energy should be > 0 even though sem_sim=0
        assert energy > 0.15

    def test_default_energy_uses_sem_sim(self):
        """Default mode: energy = sem * max(act, 0.15)."""
        now = time.time()
        cfg = ActivationConfig(seed_threshold=0.3)
        candidates = [("e1", 0.5)]
        states = {
            "e1": _make_state("e1", [now - 1]),
        }

        seeds = identify_seeds(candidates, states, now, cfg, temporal_mode=False)
        assert len(seeds) == 1
        _nid, energy = seeds[0]
        # Energy should be sem_sim * max(act, 0.15) < 1.0
        assert energy < 1.0
        assert energy > 0.0


# ---------------------------------------------------------------------------
# Phase 2: Temporal bypass in pipeline
# ---------------------------------------------------------------------------


class TestTemporalBypassPipeline:
    @pytest.mark.asyncio
    async def test_temporal_query_merges_activation_candidates(self):
        """Temporal queries merge activation-based candidates with search results."""
        now = time.time()
        search_results = [("e1", 0.8)]
        top_activated = [
            ("e2", _make_state("e2", [now - 5])),
            ("e3", _make_state("e3", [now - 60])),
        ]

        search_index = _mock_search_index(search_results)
        activation_store = _mock_activation_store(
            states={
                "e1": _make_state("e1", [now - 100]),
                "e2": _make_state("e2", [now - 5]),
                "e3": _make_state("e3", [now - 60]),
            },
            top_activated=top_activated,
        )

        results = await retrieve(
            query="What was I working on recently?",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=activation_store,
            search_index=search_index,
            cfg=ActivationConfig(),
        )
        result_ids = {r.node_id for r in results}
        # All three should appear (e1 from search, e2/e3 from activation)
        assert "e1" in result_ids
        assert "e2" in result_ids or "e3" in result_ids

    @pytest.mark.asyncio
    async def test_temporal_bypass_deduplicates(self):
        """Temporal bypass doesn't create duplicate entity_ids."""
        now = time.time()
        search_results = [("e1", 0.8)]
        # e1 appears in both search and top_activated
        top_activated = [
            ("e1", _make_state("e1", [now - 5])),
            ("e2", _make_state("e2", [now - 10])),
        ]

        search_index = _mock_search_index(search_results)
        activation_store = _mock_activation_store(
            states={
                "e1": _make_state("e1", [now - 5]),
                "e2": _make_state("e2", [now - 10]),
            },
            top_activated=top_activated,
        )

        results = await retrieve(
            query="What did I do lately?",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=activation_store,
            search_index=search_index,
            cfg=ActivationConfig(),
        )
        result_ids = [r.node_id for r in results]
        # No duplicates
        assert len(result_ids) == len(set(result_ids))

    @pytest.mark.asyncio
    async def test_non_temporal_no_bypass(self):
        """Non-temporal queries do not trigger activation bypass."""
        activation_store = _mock_activation_store()
        search_index = _mock_search_index([("e1", 0.9)])

        await retrieve(
            query="Tell me about Python",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=activation_store,
            search_index=search_index,
            cfg=ActivationConfig(),
        )
        # get_top_activated should NOT be called for non-temporal queries
        activation_store.get_top_activated.assert_not_called()

    @pytest.mark.asyncio
    async def test_temporal_empty_search_still_works(self):
        """Temporal queries return results even when search returns nothing."""
        now = time.time()
        top_activated = [
            ("e1", _make_state("e1", [now - 5])),
            ("e2", _make_state("e2", [now - 10])),
        ]

        search_index = _mock_search_index([])  # empty search results
        activation_store = _mock_activation_store(
            states={
                "e1": _make_state("e1", [now - 5]),
                "e2": _make_state("e2", [now - 10]),
            },
            top_activated=top_activated,
        )

        results = await retrieve(
            query="What was I working on recently?",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=activation_store,
            search_index=search_index,
            cfg=ActivationConfig(),
        )
        # Should still get results from activation bypass
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Phase 2: Temporal bypass in benchmark run_retrieval()
# ---------------------------------------------------------------------------


class TestBenchmarkTemporalBypass:
    @pytest.mark.asyncio
    async def test_benchmark_temporal_bypass(self):
        """run_retrieval() applies temporal bypass for recency queries."""
        from engram.benchmark.methods import METHOD_FULL_ENGRAM, run_retrieval

        now = time.time()
        top_activated = [
            ("e_recent", _make_state("e_recent", [now - 5])),
        ]

        search_index = _mock_search_index([])
        activation_store = _mock_activation_store(
            states={"e_recent": _make_state("e_recent", [now - 5])},
            top_activated=top_activated,
        )

        results = await run_retrieval(
            query="What did I do recently?",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=activation_store,
            search_index=search_index,
            method=METHOD_FULL_ENGRAM,
        )
        # Should pick up activation candidates
        result_ids = {r.node_id for r in results}
        assert "e_recent" in result_ids


# ---------------------------------------------------------------------------
# Phase 3: Structure-aware indexing
# ---------------------------------------------------------------------------


class TestStructureAwareIndexing:
    @pytest.mark.asyncio
    async def test_load_structure_aware_enriches_text(self):
        """Structure-aware loading re-indexes entities with predicate text."""
        from engram.benchmark.corpus import CorpusGenerator, CorpusSpec

        entities = [
            Entity(
                id="e1",
                name="Alice",
                entity_type="person",
                summary="A developer",
                group_id="benchmark",
            ),
            Entity(
                id="e2",
                name="TechCorp",
                entity_type="organization",
                summary="A company",
                group_id="benchmark",
            ),
        ]
        relationships = [
            Relationship(id="r1", source_id="e1", target_id="e2", predicate="WORKS_AT"),
        ]
        corpus = CorpusSpec(
            entities=entities,
            relationships=relationships,
            access_events=[],
            ground_truth=[],
        )

        graph_store = AsyncMock()
        graph_store.create_entity = AsyncMock()
        graph_store.create_relationship = AsyncMock()

        activation_store = AsyncMock()

        search_index = AsyncMock()
        search_index.index_entity = AsyncMock()
        search_index._embeddings_enabled = True

        gen = CorpusGenerator(seed=42)
        await gen.load(corpus, graph_store, activation_store, search_index, structure_aware=True)

        # index_entity called initially (2) + re-indexed (2) = 4 calls
        assert search_index.index_entity.call_count == 4

        # Check that re-indexed calls have enriched text in name field
        # (new format: "{name}. {type}. {summary}. Relationships: ...")
        re_indexed_calls = search_index.index_entity.call_args_list[2:]
        for call in re_indexed_calls:
            entity = call.args[0]
            # Enriched text is in name, summary is None
            assert entity.summary is None
            assert "Relationships:" in entity.name or "." in entity.name

    @pytest.mark.asyncio
    async def test_load_default_no_enrichment(self):
        """Default loading (structure_aware=False) does not re-index."""
        from engram.benchmark.corpus import CorpusGenerator, CorpusSpec

        entities = [
            Entity(
                id="e1",
                name="Alice",
                entity_type="person",
                summary="A developer",
                group_id="benchmark",
            ),
        ]
        corpus = CorpusSpec(
            entities=entities,
            relationships=[],
            access_events=[],
            ground_truth=[],
        )

        graph_store = AsyncMock()
        graph_store.create_entity = AsyncMock()
        graph_store.create_relationship = AsyncMock()

        activation_store = AsyncMock()

        search_index = AsyncMock()
        search_index.index_entity = AsyncMock()
        search_index._embeddings_enabled = True

        gen = CorpusGenerator(seed=42)
        await gen.load(corpus, graph_store, activation_store, search_index, structure_aware=False)

        # Only initial indexing, no re-indexing
        assert search_index.index_entity.call_count == 1

    @pytest.mark.asyncio
    async def test_structure_aware_no_embeddings_noop(self):
        """Structure-aware is no-op when embeddings are disabled."""
        from engram.benchmark.corpus import CorpusGenerator, CorpusSpec

        entities = [
            Entity(
                id="e1",
                name="Alice",
                entity_type="person",
                summary="A developer",
                group_id="benchmark",
            ),
            Entity(
                id="e2",
                name="TechCorp",
                entity_type="organization",
                summary="A company",
                group_id="benchmark",
            ),
        ]
        relationships = [
            Relationship(id="r1", source_id="e1", target_id="e2", predicate="WORKS_AT"),
        ]
        corpus = CorpusSpec(
            entities=entities,
            relationships=relationships,
            access_events=[],
            ground_truth=[],
        )

        graph_store = AsyncMock()
        graph_store.create_entity = AsyncMock()
        graph_store.create_relationship = AsyncMock()

        activation_store = AsyncMock()

        search_index = AsyncMock()
        search_index.index_entity = AsyncMock()
        search_index._embeddings_enabled = False  # No embeddings

        gen = CorpusGenerator(seed=42)
        await gen.load(corpus, graph_store, activation_store, search_index, structure_aware=True)

        # Only initial indexing (2), no re-indexing because embeddings disabled
        assert search_index.index_entity.call_count == 2
