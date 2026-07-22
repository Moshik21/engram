"""AX M3.1 — durable candidate feeder tests.

Contract:
- Flag OFF (default): the candidate pool is byte-identical to the pre-feeder
  pool and the durable listings are never queried (drift probe).
- Flag ON: identity_core / durable-class entities are ALWAYS present in the
  candidate pool even with zero lexical/semantic overlap to the query; the
  feeder is bounded at 64, cached, and injects no rank/score (scoring decides).
"""

import time
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.extraction.promotion import is_durable_recall_entity_type
from engram.models.entity import Entity
from engram.retrieval.candidate_pool import (
    _DURABLE_FEEDER_LIMIT,
    _durable_feeder_ids,
    clear_durable_feeder_cache,
    generate_candidates,
)

_SEARCH = [("e1", 0.9), ("e2", 0.7)]


def _mock_search_index(results=_SEARCH, similarity=None):
    idx = AsyncMock()
    idx.search = AsyncMock(return_value=list(results))
    idx.compute_similarity = AsyncMock(return_value=similarity or {})
    idx._embeddings_enabled = False
    return idx


def _mock_activation_store():
    store = AsyncMock()
    store.get_top_activated = AsyncMock(return_value=[])
    store.batch_get = AsyncMock(return_value={})
    return store


def _entity(eid: str, entity_type: str = "Person", name: str | None = None) -> Entity:
    return Entity(id=eid, name=name or eid, entity_type=entity_type, group_id="default")


def _mock_graph_store(identity=(), by_type=None):
    store = AsyncMock()
    store.get_active_neighbors_with_weights = AsyncMock(return_value=[])
    store.find_entity_candidates = AsyncMock(return_value=[])
    store.get_identity_core_entities = AsyncMock(return_value=list(identity))

    async def _find_by_type(entity_type, group_id, limit=100):
        return list((by_type or {}).get(entity_type, []))[:limit]

    store.find_entities_by_type = AsyncMock(side_effect=_find_by_type)
    return store


async def _run(cfg, graph, search_idx=None, query="ranking pipeline latency"):
    return await generate_candidates(
        query=query,
        group_id="default",
        search_index=search_idx or _mock_search_index(),
        activation_store=_mock_activation_store(),
        graph_store=graph,
        cfg=cfg,
        now=time.time(),
    )


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_durable_feeder_cache()
    yield
    clear_durable_feeder_cache()


class TestFlagOffByteIdentity:
    @pytest.mark.asyncio
    async def test_default_flag_is_off(self):
        assert ActivationConfig().durable_candidate_feeder_enabled is False

    @pytest.mark.asyncio
    async def test_flag_off_pool_is_byte_identical_and_listings_untouched(self):
        """Drift probe: with durable entities present, flag-off output equals
        the pre-feeder pool exactly and never touches the durable listings."""
        identity = [_entity("id_core", "Person", "Konner Moshier")]
        graph = _mock_graph_store(identity=identity)
        cfg = ActivationConfig()

        results = await _run(cfg, graph)

        assert results == [("e1", 0.9), ("e2", 0.7)]
        graph.get_identity_core_entities.assert_not_awaited()
        graph.find_entities_by_type.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flag_off_continuity_style_probe_unaffected(self):
        """Continuity-style exact-name probe: flag-off ordering/scores stable
        across repeated runs (no feeder nondeterminism leaks in)."""
        cfg = ActivationConfig()
        runs = [
            await _run(cfg, _mock_graph_store(identity=[_entity("x", "Decision")]))
            for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2] == [("e1", 0.9), ("e2", 0.7)]


class TestFlagOnFeeder:
    @pytest.mark.asyncio
    async def test_identity_entity_with_zero_overlap_appears_in_candidates(self):
        """An identity_core entity with no lexical/semantic overlap to the
        query (absent from search, cosine 0) is still in the candidate pool,
        and its durable type means the durable result lane can surface it."""
        identity = [_entity("id_core", "Person", "Konner Moshier")]
        graph = _mock_graph_store(identity=identity)
        cfg = ActivationConfig(durable_candidate_feeder_enabled=True)

        results = await _run(cfg, graph, query="helix bm25 rrf fusion weights")

        ids = [eid for eid, _score in results]
        assert "id_core" in ids
        assert is_durable_recall_entity_type("Person")

    @pytest.mark.asyncio
    async def test_no_rank_injection_no_score_floor(self):
        """Feeder-only candidates append after ranked candidates with their
        real (backfilled) similarity — 0.0 here, no floor, no rank boost."""
        identity = [_entity("id_core", "Person")]
        graph = _mock_graph_store(identity=identity)
        cfg = ActivationConfig(durable_candidate_feeder_enabled=True)

        results = await _run(cfg, graph)

        assert results[0] == ("e1", 0.9)
        assert results[1] == ("e2", 0.7)
        score = dict(results)["id_core"]
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_search_scores_unchanged_flag_on(self):
        graph = _mock_graph_store(identity=[_entity("id_core", "Person")])
        on = await _run(ActivationConfig(durable_candidate_feeder_enabled=True), graph)
        off = await _run(ActivationConfig(), _mock_graph_store())
        assert [r for r in on if r[0] in {"e1", "e2"}] == off

    @pytest.mark.asyncio
    async def test_feeder_works_when_all_other_pools_empty(self):
        graph = _mock_graph_store(identity=[_entity("id_core", "Preference")])
        cfg = ActivationConfig(durable_candidate_feeder_enabled=True)
        results = await _run(cfg, graph, search_idx=_mock_search_index(results=[]))
        assert [eid for eid, _ in results] == ["id_core"]

    @pytest.mark.asyncio
    async def test_pool_budget_bounded_at_64(self):
        identity = [_entity(f"idc_{i}", "Person") for i in range(40)]
        by_type = {"Decision": [_entity(f"dec_{i}", "Decision") for i in range(80)]}
        graph = _mock_graph_store(identity=identity, by_type=by_type)

        ids = await _durable_feeder_ids("default", graph, time.time())

        assert len(ids) == _DURABLE_FEEDER_LIMIT == 64
        # identity_core listed first, then durable types up to the bound.
        assert ids[:40] == [f"idc_{i}" for i in range(40)]
        assert all(eid.startswith("dec_") for eid in ids[40:])

    @pytest.mark.asyncio
    async def test_listing_is_cached_within_ttl(self):
        graph = _mock_graph_store(identity=[_entity("id_core", "Person")])
        now = time.time()

        first = await _durable_feeder_ids("default", graph, now)
        second = await _durable_feeder_ids("default", graph, now + 1.0)

        assert first == second == ["id_core"]
        assert graph.get_identity_core_entities.await_count == 1

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self):
        graph = _mock_graph_store(identity=[_entity("id_core", "Person")])
        now = time.time()

        await _durable_feeder_ids("default", graph, now)
        await _durable_feeder_ids("default", graph, now + 61.0)

        assert graph.get_identity_core_entities.await_count == 2

    @pytest.mark.asyncio
    async def test_listing_failure_is_non_fatal(self):
        graph = _mock_graph_store()
        graph.get_identity_core_entities = AsyncMock(side_effect=RuntimeError("native down"))
        cfg = ActivationConfig(durable_candidate_feeder_enabled=True)

        results = await _run(cfg, graph)

        assert results == [("e1", 0.9), ("e2", 0.7)]

    @pytest.mark.asyncio
    async def test_feeder_dedupes_against_ranked_pool(self):
        graph = _mock_graph_store(identity=[_entity("e1", "Person")])
        cfg = ActivationConfig(durable_candidate_feeder_enabled=True)

        results = await _run(cfg, graph)

        ids = [eid for eid, _ in results]
        assert ids.count("e1") == 1
