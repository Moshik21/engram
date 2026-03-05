"""Tests for the multi-signal infer scorer."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import numpy as np
import pytest

from engram.consolidation.scorers.infer_scorer import (
    compute_structural_score,
    compute_type_compatibility,
    compute_ubiquity_score,
    score_infer_pair,
)

# ---------------------------------------------------------------------------
# compute_type_compatibility
# ---------------------------------------------------------------------------


class TestTypeCompatibility:
    def test_same_domain_personal(self):
        assert compute_type_compatibility("Person", "Event") == 1.0

    def test_same_domain_technical(self):
        assert compute_type_compatibility("Technology", "Software") == 1.0

    def test_cross_domain_personal_technical(self):
        assert compute_type_compatibility("Person", "Technology") == 0.8

    def test_cross_domain_technical_knowledge(self):
        assert compute_type_compatibility("Software", "Concept") == 0.9

    def test_cross_domain_low_compat(self):
        assert compute_type_compatibility("Organization", "HealthCondition") == 0.3

    def test_unknown_type_defaults_to_knowledge(self):
        # Unknown types map to "knowledge" domain
        score = compute_type_compatibility("UnknownType", "Concept")
        assert score == 1.0  # knowledge + knowledge

    def test_both_unknown_types(self):
        score = compute_type_compatibility("FooType", "BarType")
        assert score == 1.0  # both map to "knowledge"

    def test_custom_domain_groups(self):
        custom = {"custom": ["Alpha", "Beta"], "other": ["Gamma"]}
        assert compute_type_compatibility("Alpha", "Beta", custom) == 0.5  # not in DOMAIN_COMPAT
        assert compute_type_compatibility("Alpha", "Gamma", custom) == 0.5

    def test_symmetry(self):
        # Order of types should not matter
        assert compute_type_compatibility("Person", "Technology") == compute_type_compatibility(
            "Technology", "Person"
        )


# ---------------------------------------------------------------------------
# compute_ubiquity_score
# ---------------------------------------------------------------------------


class TestUbiquityScore:
    def test_zero_total_episodes(self):
        assert compute_ubiquity_score(5, 3, 2, 0) == 0.5

    def test_rare_entities_high_score(self):
        # Both entities appear in 5/100 episodes, co-occur in 3
        score = compute_ubiquity_score(5, 5, 3, 100)
        assert score > 0.7  # rare → high score

    def test_ubiquitous_entity_low_score(self):
        # One entity appears in 60/100 episodes
        score = compute_ubiquity_score(60, 5, 3, 100)
        assert score == 0.2  # max_freq > 0.5

    def test_moderately_frequent(self):
        # One entity in 35/100 episodes
        score = compute_ubiquity_score(35, 10, 5, 100)
        assert 0.3 <= score <= 0.8  # moderate range

    def test_perfect_jaccard(self):
        # Both entities in same 10 episodes, co-occur in all 10
        score = compute_ubiquity_score(10, 10, 10, 100)
        # max_freq = 0.1, jaccard = 10/10 = 1.0
        assert score == pytest.approx(1.0)  # 0.7 + 0.3 * 1.0

    def test_no_jaccard(self):
        # Co-occur 1 time, each appear many times
        score = compute_ubiquity_score(20, 20, 1, 100)
        # max_freq = 0.2, jaccard = 1/39
        assert score > 0.7  # still decent since max_freq < 0.3


# ---------------------------------------------------------------------------
# compute_structural_score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structural_score_shared_neighbors():
    """Shared neighbors produce a high structural score."""
    graph_store = AsyncMock()
    # Neighbors: (entity_id, predicate, weight, direction)
    shared_neighbor = ("n1", "USES", 0.5, "out")
    graph_store.get_active_neighbors_with_weights.side_effect = [
        [shared_neighbor, ("n2", "KNOWS", 0.3, "out")],
        [shared_neighbor, ("n3", "WORKS_AT", 0.4, "out")],
    ]
    score = await compute_structural_score("a", "b", graph_store, "g")
    # shared = {n1}, union = {n1, n2, n3}
    assert score == pytest.approx(0.6 + 0.4 * 1 / 3, abs=0.01)


@pytest.mark.asyncio
async def test_structural_score_no_overlap():
    """Connected but no shared neighbors → 0.3."""
    graph_store = AsyncMock()
    graph_store.get_active_neighbors_with_weights.side_effect = [
        [("n1", "USES", 0.5, "out")],
        [("n2", "KNOWS", 0.3, "out")],
    ]
    score = await compute_structural_score("a", "b", graph_store, "g")
    assert score == 0.3


@pytest.mark.asyncio
async def test_structural_score_both_isolated():
    """Both isolated → neutral 0.5."""
    graph_store = AsyncMock()
    graph_store.get_active_neighbors_with_weights.side_effect = [[], []]
    score = await compute_structural_score("a", "b", graph_store, "g")
    assert score == 0.5


@pytest.mark.asyncio
async def test_structural_score_exception_returns_neutral():
    """Graph store failure → neutral 0.5."""
    graph_store = AsyncMock()
    graph_store.get_active_neighbors_with_weights.side_effect = RuntimeError("db down")
    score = await compute_structural_score("a", "b", graph_store, "g")
    assert score == 0.5


# ---------------------------------------------------------------------------
# score_infer_pair (full async integration)
# ---------------------------------------------------------------------------


@dataclass
class _FakeEntity:
    entity_type: str
    name: str = "test"


def _make_mocks(
    *,
    emb_a: list[float] | None = None,
    emb_b: list[float] | None = None,
    graph_emb_a: list[float] | None = None,
    graph_emb_b: list[float] | None = None,
    neighbors_a: list | None = None,
    neighbors_b: list | None = None,
):
    search_index = AsyncMock()
    graph_store = AsyncMock()

    entity_embs = {}
    if emb_a is not None:
        entity_embs["a"] = emb_a
    if emb_b is not None:
        entity_embs["b"] = emb_b
    search_index.get_entity_embeddings.return_value = entity_embs

    graph_embs = {}
    if graph_emb_a is not None:
        graph_embs["a"] = graph_emb_a
    if graph_emb_b is not None:
        graph_embs["b"] = graph_emb_b
    search_index.get_graph_embeddings.return_value = graph_embs

    graph_store.get_active_neighbors_with_weights.side_effect = [
        neighbors_a or [],
        neighbors_b or [],
    ]

    return search_index, graph_store


@pytest.mark.asyncio
async def test_score_infer_pair_approved():
    """High signals across the board should produce an approved verdict."""
    # Identical embeddings → emb_score = 1.0
    vec = [1.0, 0.0, 0.0]
    search_index, graph_store = _make_mocks(
        emb_a=vec,
        emb_b=vec,
        graph_emb_a=vec,
        graph_emb_b=vec,
        neighbors_a=[("shared", "USES", 0.5, "out")],
        neighbors_b=[("shared", "USES", 0.5, "out")],
    )

    verdict, score, signals = await score_infer_pair(
        entity_a_id="a",
        entity_b_id="b",
        entity_a_name="Python",
        entity_b_name="Django",
        entity_a_type="Technology",
        entity_b_type="Software",
        co_occurrence_count=10,
        pmi_confidence=0.8,
        ep_count_a=5,
        ep_count_b=5,
        total_episodes=100,
        search_index=search_index,
        graph_store=graph_store,
        group_id="default",
    )

    assert verdict == "approved"
    assert score >= 0.65
    assert signals["embedding"] == pytest.approx(1.0, abs=0.01)
    assert signals["type_compat"] == 1.0  # same domain (technical)


@pytest.mark.asyncio
async def test_score_infer_pair_rejected():
    """Low signals → rejected."""
    # Orthogonal embeddings → emb_score = 0.0
    search_index, graph_store = _make_mocks(
        emb_a=[1.0, 0.0, 0.0],
        emb_b=[0.0, 1.0, 0.0],
        neighbors_a=[("n1", "X", 0.1, "out")],
        neighbors_b=[("n2", "Y", 0.1, "out")],
    )

    verdict, score, signals = await score_infer_pair(
        entity_a_id="a",
        entity_b_id="b",
        entity_a_name="Python",
        entity_b_name="Hospital",
        entity_a_type="Software",
        entity_b_type="Location",
        co_occurrence_count=3,
        pmi_confidence=0.2,
        ep_count_a=60,
        ep_count_b=5,
        total_episodes=100,
        search_index=search_index,
        graph_store=graph_store,
        group_id="default",
    )

    assert verdict == "rejected"
    assert score < 0.40
    assert signals["embedding"] == pytest.approx(0.0, abs=0.01)
    assert signals["ubiquity"] == 0.2  # ep_count_a/100 = 0.6 > 0.5


@pytest.mark.asyncio
async def test_score_infer_pair_uncertain():
    """Middling signals → uncertain."""
    search_index, graph_store = _make_mocks(
        emb_a=[1.0, 0.0, 0.0],
        emb_b=[0.5, 0.5, 0.707],
    )

    verdict, score, signals = await score_infer_pair(
        entity_a_id="a",
        entity_b_id="b",
        entity_a_name="Alice",
        entity_b_name="Hospital",
        entity_a_type="Person",
        entity_b_type="Location",
        co_occurrence_count=4,
        pmi_confidence=0.35,
        ep_count_a=25,
        ep_count_b=20,
        total_episodes=100,
        search_index=search_index,
        graph_store=graph_store,
        group_id="default",
    )

    assert verdict == "uncertain"
    assert 0.40 <= score < 0.65


@pytest.mark.asyncio
async def test_score_infer_pair_no_embeddings():
    """Missing embeddings default to 0.5 (neutral), should not crash."""
    search_index, graph_store = _make_mocks()

    verdict, score, signals = await score_infer_pair(
        entity_a_id="a",
        entity_b_id="b",
        entity_a_name="X",
        entity_b_name="Y",
        entity_a_type="Concept",
        entity_b_type="Concept",
        co_occurrence_count=5,
        pmi_confidence=0.6,
        ep_count_a=10,
        ep_count_b=10,
        total_episodes=100,
        search_index=search_index,
        graph_store=graph_store,
        group_id="default",
    )

    assert verdict in ("approved", "uncertain", "rejected")
    assert signals["embedding"] == 0.5
    assert signals["graph_emb"] == 0.5


@pytest.mark.asyncio
async def test_score_infer_pair_custom_thresholds():
    """Custom approve/reject thresholds shift verdicts."""
    vec = [1.0, 0.0, 0.0]
    search_index, graph_store = _make_mocks(emb_a=vec, emb_b=vec)

    verdict, score, _ = await score_infer_pair(
        entity_a_id="a",
        entity_b_id="b",
        entity_a_name="X",
        entity_b_name="Y",
        entity_a_type="Technology",
        entity_b_type="Technology",
        co_occurrence_count=5,
        pmi_confidence=0.7,
        ep_count_a=5,
        ep_count_b=5,
        total_episodes=100,
        search_index=search_index,
        graph_store=graph_store,
        group_id="default",
        approve_threshold=0.95,  # very high bar
        reject_threshold=0.90,
    )

    # Even good signals won't hit 0.95 with neutral structural/ubiquity
    assert verdict in ("rejected", "uncertain")


@pytest.mark.asyncio
async def test_score_infer_pair_co_occurrence_boost():
    """High co-occurrence count boosts statistical signal."""
    search_index, graph_store = _make_mocks()

    _, score_low, signals_low = await score_infer_pair(
        entity_a_id="a",
        entity_b_id="b",
        entity_a_name="X",
        entity_b_name="Y",
        entity_a_type="Concept",
        entity_b_type="Concept",
        co_occurrence_count=3,
        pmi_confidence=0.5,
        ep_count_a=10,
        ep_count_b=10,
        total_episodes=100,
        search_index=search_index,
        graph_store=graph_store,
        group_id="default",
    )

    # Reset mock side_effect for second call
    graph_store.get_active_neighbors_with_weights.side_effect = [[], []]

    _, score_high, signals_high = await score_infer_pair(
        entity_a_id="a",
        entity_b_id="b",
        entity_a_name="X",
        entity_b_name="Y",
        entity_a_type="Concept",
        entity_b_type="Concept",
        co_occurrence_count=12,
        pmi_confidence=0.5,
        ep_count_a=10,
        ep_count_b=10,
        total_episodes=100,
        search_index=search_index,
        graph_store=graph_store,
        group_id="default",
    )

    assert signals_high["statistical"] > signals_low["statistical"]


@pytest.mark.asyncio
async def test_score_signals_are_rounded():
    """All signal values should be rounded to 4 decimal places."""
    vec_a = list(np.random.default_rng(42).random(3).astype(float))
    vec_b = list(np.random.default_rng(99).random(3).astype(float))
    search_index, graph_store = _make_mocks(emb_a=vec_a, emb_b=vec_b)

    _, _, signals = await score_infer_pair(
        entity_a_id="a",
        entity_b_id="b",
        entity_a_name="X",
        entity_b_name="Y",
        entity_a_type="Person",
        entity_b_type="Technology",
        co_occurrence_count=5,
        pmi_confidence=0.55,
        ep_count_a=8,
        ep_count_b=12,
        total_episodes=100,
        search_index=search_index,
        graph_store=graph_store,
        group_id="default",
    )

    for key, val in signals.items():
        # Check that val has at most 4 decimal places
        assert val == round(val, 4), f"{key} not rounded: {val}"
