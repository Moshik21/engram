"""Tests for embedding-based relevance confidence scoring."""

from __future__ import annotations

import pytest

from engram.retrieval.relevance import (
    RelevanceScorer,
    compute_answer_containment,
    cosine_similarity,
)
from engram.retrieval.scorer import ScoredResult

# ── Helpers ───────────────────────────────────────────────────────────

def _unit_vec(dim: int, idx: int) -> list[float]:
    """Create a unit vector with 1.0 at position idx."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


def _make_sr(node_id: str, result_type: str = "entity", sem_sim: float = 0.0) -> ScoredResult:
    return ScoredResult(
        node_id=node_id,
        score=0.5,
        semantic_similarity=sem_sim,
        activation=0.0,
        spreading=0.0,
        edge_proximity=0.0,
        result_type=result_type,
    )


class FakeProvider:
    """Fake embedding provider for testing."""

    def __init__(self, dim: int = 8, vectors: dict[str, list[float]] | None = None):
        self._dim = dim
        self._vectors = vectors or {}
        self._query_vec: list[float] | None = None

    def dimension(self) -> int:
        return self._dim

    async def embed_query(self, text: str) -> list[float]:
        if text in self._vectors:
            return self._vectors[text]
        if self._query_vec is not None:
            return self._query_vec
        if self._dim == 0:
            return []
        # Default: unit vector at position 0
        return _unit_vec(self._dim, 0)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for t in texts:
            if t in self._vectors:
                result.append(self._vectors[t])
            else:
                # Default: all zeros (no similarity)
                result.append([0.0] * self._dim)
        return result


# ── cosine_similarity tests ──────────────────────────────────────────

def test_cosine_same_vector():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ── RelevanceScorer tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_entity_uses_semantic_similarity():
    """Entity relevance should equal its semantic_similarity from search."""
    provider = FakeProvider(dim=4)
    scorer = RelevanceScorer(provider)

    sr = _make_sr("e1", "entity", sem_sim=0.85)
    await scorer.score_results(
        query="test",
        results=[sr],
        entity_summaries={"e1": "some summary"},
        episode_contents={},
        chunk_texts={},
    )
    assert sr.relevance_confidence == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_episode_with_chunk_gets_embedded():
    """Episode with chunk text should get relevance from embedding."""
    dim = 4
    query_vec = [1.0, 0.0, 0.0, 0.0]
    chunk_vec = [0.9, 0.1, 0.0, 0.0]  # high similarity to query

    provider = FakeProvider(dim=dim, vectors={
        "test query": query_vec,
        "relevant chunk text": chunk_vec,
    })
    scorer = RelevanceScorer(provider)

    sr = _make_sr("ep1", "episode", sem_sim=0.3)
    await scorer.score_results(
        query="test query",
        results=[sr],
        entity_summaries={},
        episode_contents={"ep1": "full episode content"},
        chunk_texts={"ep1": "relevant chunk text"},
        query_vec=query_vec,
    )
    expected = cosine_similarity(query_vec, chunk_vec)
    assert sr.relevance_confidence == pytest.approx(expected, abs=1e-4)


@pytest.mark.asyncio
async def test_episode_without_chunk_falls_back_to_semantic():
    """Episode without chunk text should use semantic_similarity."""
    provider = FakeProvider(dim=4)
    scorer = RelevanceScorer(provider)

    sr = _make_sr("ep1", "episode", sem_sim=0.6)
    await scorer.score_results(
        query="test",
        results=[sr],
        entity_summaries={},
        episode_contents={},  # no content
        chunk_texts={},  # no chunk
    )
    # No text to embed, falls back to semantic_similarity
    assert sr.relevance_confidence == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_reuses_query_vec():
    """Should reuse provided query_vec instead of re-embedding."""
    call_count = 0

    class TrackingProvider(FakeProvider):
        async def embed_query(self, text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            return await super().embed_query(text)

    provider = TrackingProvider(dim=4)
    scorer = RelevanceScorer(provider)

    sr = _make_sr("e1", "entity", sem_sim=0.5)
    await scorer.score_results(
        query="test",
        results=[sr],
        entity_summaries={},
        episode_contents={},
        chunk_texts={},
        query_vec=[1.0, 0.0, 0.0, 0.0],  # pre-computed
    )
    assert call_count == 0  # Should not call embed_query


@pytest.mark.asyncio
async def test_empty_results_noop():
    """Empty results should not raise."""
    provider = FakeProvider(dim=4)
    scorer = RelevanceScorer(provider)

    await scorer.score_results(
        query="test",
        results=[],
        entity_summaries={},
        episode_contents={},
        chunk_texts={},
    )


@pytest.mark.asyncio
async def test_no_embeddings_leaves_zero():
    """NoopProvider (dim=0) should leave relevance at 0.0."""
    provider = FakeProvider(dim=0)
    scorer = RelevanceScorer(provider)

    sr = _make_sr("e1", "entity", sem_sim=0.8)
    await scorer.score_results(
        query="test",
        results=[sr],
        entity_summaries={"e1": "summary"},
        episode_contents={},
        chunk_texts={},
    )
    assert sr.relevance_confidence == 0.0


@pytest.mark.asyncio
async def test_multiple_results_mixed_types():
    """Score a mix of entities and episodes."""
    dim = 4
    query_vec = [1.0, 0.0, 0.0, 0.0]

    provider = FakeProvider(dim=dim, vectors={
        "test": query_vec,
        "episode text": [0.8, 0.2, 0.0, 0.0],
    })
    scorer = RelevanceScorer(provider)

    entity = _make_sr("e1", "entity", sem_sim=0.9)
    episode = _make_sr("ep1", "episode", sem_sim=0.4)

    await scorer.score_results(
        query="test",
        results=[entity, episode],
        entity_summaries={"e1": "entity summary"},
        episode_contents={"ep1": "episode text"},
        chunk_texts={},
        query_vec=query_vec,
    )

    assert entity.relevance_confidence == pytest.approx(0.9)
    assert episode.relevance_confidence > 0.0


@pytest.mark.asyncio
async def test_cue_episode_scored():
    """cue_episode results should also get relevance scoring."""
    dim = 4
    query_vec = [1.0, 0.0, 0.0, 0.0]
    chunk_vec = [0.7, 0.3, 0.0, 0.0]

    provider = FakeProvider(dim=dim, vectors={
        "cue chunk": chunk_vec,
    })
    scorer = RelevanceScorer(provider)

    sr = _make_sr("ep1", "cue_episode", sem_sim=0.2)
    await scorer.score_results(
        query="test",
        results=[sr],
        entity_summaries={},
        episode_contents={},
        chunk_texts={"ep1": "cue chunk"},
        query_vec=query_vec,
    )
    expected = cosine_similarity(query_vec, chunk_vec)
    assert sr.relevance_confidence == pytest.approx(expected, abs=1e-4)


# ── compute_answer_containment tests ─────────────────────────────────

def test_containment_identical():
    """Identical vectors should give 1.0."""
    v = [1.0, 2.0, 3.0, 4.0]
    assert compute_answer_containment(v, [v]) == pytest.approx(1.0, abs=1e-6)


def test_containment_picks_best():
    """Should return max similarity across evidence."""
    gold = [1.0, 0.0, 0.0, 0.0]
    bad = [0.0, 1.0, 0.0, 0.0]
    good = [0.9, 0.1, 0.0, 0.0]
    perfect = [1.0, 0.0, 0.0, 0.0]

    score = compute_answer_containment(gold, [bad, good, perfect])
    assert score == pytest.approx(1.0, abs=1e-6)


def test_containment_empty_evidence():
    assert compute_answer_containment([1.0, 0.0], []) == 0.0


def test_containment_empty_gold():
    assert compute_answer_containment([], [[1.0, 0.0]]) == 0.0


def test_containment_orthogonal():
    """Orthogonal vectors should give 0.0."""
    gold = [1.0, 0.0]
    evidence = [[0.0, 1.0]]
    assert compute_answer_containment(gold, evidence) == pytest.approx(0.0, abs=1e-6)


# ── packets._confidence integration ──────────────────────────────────

def test_confidence_with_relevance():
    """_confidence should prefer relevance when available."""
    from engram.retrieval.packets import _confidence

    # With relevance
    assert _confidence(0.3, 0.0, 0.85) == pytest.approx(0.85)

    # Without relevance (0.0), falls back to score-based
    assert _confidence(0.6, 0.0, 0.0) == pytest.approx(0.6)


def test_confidence_capped_at_099():
    from engram.retrieval.packets import _confidence

    assert _confidence(0.0, 0.0, 1.0) == 0.99


# ── ScoredResult.relevance_confidence field ──────────────────────────

def test_scored_result_has_relevance():
    sr = ScoredResult(
        node_id="x",
        score=1.0,
        semantic_similarity=0.5,
        activation=0.0,
        spreading=0.0,
        edge_proximity=0.0,
    )
    assert sr.relevance_confidence == 0.0  # default

    sr.relevance_confidence = 0.87
    assert sr.relevance_confidence == 0.87
