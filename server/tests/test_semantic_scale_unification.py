"""Tests for the RRF-vs-cosine semantic scale unification.

Search candidates carry RRF rank-normalized scores (top hit always 1.0) while
graph-discovered/backfilled candidates carry raw cosine. When
``entity_episode_traversal_source="candidates"`` (the flip stack):

- search hits keep their RRF scores byte-identical;
- cosine-scored candidates (graph pool, backfill, spreading) are rank-mapped
  onto the RRF scale at the position their true cosine merits within the pool
  (``map_cosine_to_pool_scale``);
- seed selection gates on TRUE cosine, so seed_threshold is a relevance test
  rather than a rank cutoff.

The default ("results") path must be byte-identical.
"""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig
from engram.retrieval.candidate_pool import map_cosine_to_pool_scale
from engram.retrieval.pipeline import retrieve

# RRF rank-normalized scores: score(rank k) = (rrf_k + 1) / (rrf_k + k),
# so the top hit is always 1.0 regardless of true relevance (rrf_k = 60).
_RRF_K = 60
_N_HITS = 40


def _rrf_score(rank: int) -> float:
    return (_RRF_K + 1) / (_RRF_K + rank)


def _hit_id(rank: int) -> str:
    return f"hit_{rank:02d}"


class _RecordingSearchIndex:
    """Search index stub that records compute_similarity calls."""

    def __init__(
        self,
        results: list[tuple[str, float]],
        similarity_map: dict[str, float],
    ):
        self._results = results
        self._similarity_map = similarity_map
        self.similarity_calls: list[list[str]] = []

    async def search(self, query: str, group_id: str, limit: int = 50):
        return self._results[:limit]

    async def compute_similarity(
        self,
        query: str,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, float]:
        self.similarity_calls.append(list(entity_ids))
        return {eid: self._similarity_map.get(eid, 0.0) for eid in entity_ids}


class _FakeActivationStore:
    async def batch_get(self, entity_ids: list[str]):
        return {}

    async def get_activation(self, entity_id: str):
        return None

    async def set_activation(self, entity_id: str, state) -> None:
        pass


class _FakeGraphStore:
    def __init__(self, adjacency: dict[str, list[str]] | None = None):
        self._adj = adjacency or {}

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None, **kwargs
    ) -> list[tuple[str, float]]:
        return [(n, 1.0) for n in self._adj.get(entity_id, [])]

    async def find_entities(self, **kwargs):
        return []

    async def get_stats(self, group_id: str) -> dict:
        return {"entity_count": 0}

    async def get_entity(self, entity_id: str, group_id: str):
        return None


def _make_cfg(**overrides) -> ActivationConfig:
    """Semantic-only scoring so rank order is decided by the semantic term."""
    base = dict(
        weight_semantic=1.0,
        weight_activation=0.0,
        weight_spreading=0.0,
        weight_edge_proximity=0.0,
        exploration_weight=0.0,
        rediscovery_weight=0.0,
        seed_threshold=0.3,
        mmr_enabled=False,
    )
    base.update(overrides)
    return ActivationConfig(**base)


def _make_stores():
    """40 RRF-scored search hits + one graph-discovered bridge entity.

    ``discovered`` is a 1-hop neighbor of the top hit (never returned by
    search); its true cosine is 0.45 — a genuinely good neighbor. ``hit_40``
    is a lexical near-miss: rank 40 (RRF-normalized 0.61) but true cosine 0.10.
    """
    results = [(_hit_id(k), _rrf_score(k)) for k in range(1, _N_HITS + 1)]
    similarity_map = {
        _hit_id(1): 0.9,
        _hit_id(40): 0.10,
        "discovered": 0.45,
    }
    search_index = _RecordingSearchIndex(results, similarity_map)
    graph_store = _FakeGraphStore({_hit_id(1): ["discovered"]})
    return search_index, graph_store


async def _run(cfg: ActivationConfig, search_index, graph_store):
    return await retrieve(
        query="latest news about the hobby",
        group_id="default",
        graph_store=graph_store,
        activation_store=_FakeActivationStore(),
        search_index=search_index,
        cfg=cfg,
        limit=10,
    )


@pytest.mark.asyncio
class TestDefaultPathUnchanged:
    async def test_search_hits_keep_rrf_scores_and_no_rescale_call(self):
        """source="results" (default): scores untouched, rescale never runs."""
        search_index, graph_store = _make_stores()
        cfg = _make_cfg()
        assert cfg.entity_episode_traversal_source == "results"

        results = await _run(cfg, search_index, graph_store)

        # Search-scored ids are never sent to compute_similarity — only the
        # backfill for non-search ids (and step 4.5 spread-discovered) may be.
        for call in search_index.similarity_calls:
            assert not any(eid.startswith("hit_") for eid in call)

        # Returned hits keep their exact RRF rank-normalized scores.
        by_id = {r.node_id: r for r in results}
        for rank in range(1, 11):
            assert by_id[_hit_id(rank)].semantic_similarity == pytest.approx(_rrf_score(rank))

        # The 0.45-cosine graph-discovered entity is buried below all 40 hits
        # (every hit carries >= 0.61 rank-normalized) — today's defect.
        assert "discovered" not in by_id

    async def test_default_every_returned_hit_is_a_seed(self):
        """RRF scores >= 0.61 make seed_threshold=0.3 a rank cutoff."""
        search_index, graph_store = _make_stores()
        results = await _run(_make_cfg(), search_index, graph_store)

        for r in results:
            if r.node_id.startswith("hit_"):
                assert r.edge_proximity == 1.0
                assert r.hop_distance == 0


@pytest.mark.asyncio
class TestCandidatesModeUnifiedScale:
    async def test_graph_discovered_entity_not_buried(self):
        """A 0.45-cosine spread/graph-discovered entity outranks a rank-40 hit."""
        search_index, graph_store = _make_stores()
        cfg = _make_cfg(entity_episode_traversal_source="candidates")

        results = await _run(cfg, search_index, graph_store)
        ids = [r.node_id for r in results]

        assert "discovered" in ids
        by_id = {r.node_id: r for r in results}
        # Rank-mapped onto the RRF scale: above every 0.0-cosine hit, below
        # the 0.9-cosine top hit (i.e. no longer the raw 0.45 that buried it).
        assert by_id["discovered"].semantic_similarity > _rrf_score(40)
        assert by_id["discovered"].semantic_similarity < 1.0
        # True cosine order: hit_01 (0.9) > discovered (0.45) > hit_40 (0.10).
        assert ids.index(_hit_id(1)) < ids.index("discovered")

    async def test_search_hits_keep_rrf_scores_under_gate(self):
        """Scale unification never changes a search hit's own score."""
        search_index, graph_store = _make_stores()
        cfg = _make_cfg(entity_episode_traversal_source="candidates")

        results = await _run(cfg, search_index, graph_store)
        by_id = {r.node_id: r for r in results}
        for node_id, result in by_id.items():
            if node_id.startswith("hit_"):
                rank = int(node_id.split("_")[1])
                assert result.semantic_similarity == pytest.approx(_rrf_score(rank))

    async def test_seed_set_shrinks_to_genuinely_similar(self):
        """seed_threshold=0.3 against true cosine keeps only real matches."""
        search_index, graph_store = _make_stores()
        cfg = _make_cfg(entity_episode_traversal_source="candidates")

        results = await _run(cfg, search_index, graph_store)

        seeds = {r.node_id for r in results if r.hop_distance == 0}
        # Only hit_01 (cosine 0.9) and discovered (0.45) clear the threshold;
        # under the default scale all 40 hits would have seeded.
        assert seeds == {_hit_id(1), "discovered"}
        by_id = {r.node_id: r for r in results}
        if _hit_id(40) in by_id:
            assert by_id[_hit_id(40)].edge_proximity < 1.0

    async def test_scale_map_degrades_to_mixed_scale_on_failure(self):
        """A failing compute_similarity leaves scores unchanged, not zeroed."""

        class _FailingRescale(_RecordingSearchIndex):
            async def compute_similarity(self, query, entity_ids, group_id=None):
                raise RuntimeError("embeddings dead")

        results_in = [(_hit_id(k), _rrf_score(k)) for k in range(1, 6)]
        search_index = _FailingRescale(results_in, {})
        cfg = _make_cfg(entity_episode_traversal_source="candidates")

        results = await _run(cfg, search_index, _FakeGraphStore())
        by_id = {r.node_id: r for r in results}
        for rank in range(1, 6):
            assert by_id[_hit_id(rank)].semantic_similarity == pytest.approx(_rrf_score(rank))


class TestMapCosineToPoolScale:
    LADDER = [(0.9, 1.0), (0.5, 0.8), (0.2, 0.6)]

    def test_empty_ladder_returns_cosine(self):
        assert map_cosine_to_pool_scale(0.45, []) == 0.45

    def test_above_top_clamps_to_top_score(self):
        assert map_cosine_to_pool_scale(0.95, self.LADDER) == 1.0

    def test_exact_ladder_points(self):
        assert map_cosine_to_pool_scale(0.9, self.LADDER) == 1.0
        assert map_cosine_to_pool_scale(0.5, self.LADDER) == pytest.approx(0.8)
        assert map_cosine_to_pool_scale(0.2, self.LADDER) == pytest.approx(0.6)

    def test_interpolates_between_points(self):
        # Midway between (0.5, 0.8) and (0.9, 1.0) -> 0.9
        assert map_cosine_to_pool_scale(0.7, self.LADDER) == pytest.approx(0.9)

    def test_below_bottom_interpolates_toward_zero(self):
        # Half the bottom cosine -> half the bottom score
        assert map_cosine_to_pool_scale(0.1, self.LADDER) == pytest.approx(0.3)
        assert map_cosine_to_pool_scale(0.0, self.LADDER) == pytest.approx(0.0)

    def test_monotone_in_cosine(self):
        values = [map_cosine_to_pool_scale(c / 100.0, self.LADDER) for c in range(0, 101, 5)]
        assert values == sorted(values)
