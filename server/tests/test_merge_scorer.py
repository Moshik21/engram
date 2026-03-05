"""Tests for the multi-signal deterministic merge scorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import numpy as np
import pytest

from engram.consolidation.scorers.merge_scorer import (
    COMPATIBLE_CROSS_TYPES,
    acronym_match,
    canonical_match,
    compute_name_score,
    containment_match,
    numeronym_match,
    score_merge_pair,
    summary_overlap,
    type_compatible,
)


# ---------------------------------------------------------------------------
# Mock entity
# ---------------------------------------------------------------------------


@dataclass
class MockEntity:
    id: str
    name: str
    entity_type: str
    summary: str | None = None
    access_count: int = 0
    created_at: str = "2025-01-01T00:00:00"


# ---------------------------------------------------------------------------
# acronym_match
# ---------------------------------------------------------------------------


class TestAcronymMatch:
    def test_js_javascript(self):
        assert acronym_match("JS", "JavaScript") == 0.95

    def test_ml_machine_learning(self):
        assert acronym_match("ML", "Machine Learning") == 0.95

    def test_api_application_programming_interface(self):
        assert acronym_match("API", "Application Programming Interface") == 0.95

    def test_no_match(self):
        assert acronym_match("XYZ", "Python") == 0.0

    def test_same_length(self):
        assert acronym_match("abc", "def") == 0.0

    def test_case_insensitive(self):
        assert acronym_match("ml", "machine learning") == 0.95

    def test_camel_case_split(self):
        assert acronym_match("ML", "MachineLearning") == 0.95


# ---------------------------------------------------------------------------
# numeronym_match
# ---------------------------------------------------------------------------


class TestNumeronymMatch:
    def test_k8s_kubernetes(self):
        assert numeronym_match("k8s", "kubernetes") == 0.95

    def test_i18n_internationalization(self):
        assert numeronym_match("i18n", "internationalization") == 0.95

    def test_no_match(self):
        assert numeronym_match("abc", "python") == 0.0

    def test_non_numeronym_format(self):
        assert numeronym_match("react", "reactjs") == 0.0

    def test_same_length(self):
        assert numeronym_match("k8s", "abc") == 0.0


# ---------------------------------------------------------------------------
# containment_match
# ---------------------------------------------------------------------------


class TestContainmentMatch:
    def test_react_reactjs(self):
        score = containment_match("React", "React.js")
        assert score >= 0.88

    def test_python_python_language(self):
        score = containment_match("Python", "Python Language")
        assert score >= 0.88

    def test_suffix_strip(self):
        score = containment_match("Python", "Python.py")
        assert score >= 0.95

    def test_framework_suffix(self):
        score = containment_match("Django", "Django framework")
        assert score >= 0.95

    def test_no_overlap(self):
        assert containment_match("Rust", "Python") == 0.0

    def test_empty_names(self):
        assert containment_match("", "Python") == 0.0


# ---------------------------------------------------------------------------
# canonical_match
# ---------------------------------------------------------------------------


class TestCanonicalMatch:
    def test_javascript_js(self):
        assert canonical_match("JavaScript", "JS") == 0.98

    def test_kubernetes_k8s(self):
        assert canonical_match("Kubernetes", "K8s") == 0.98

    def test_aws(self):
        assert canonical_match("Amazon Web Services", "AWS") == 0.98

    def test_ai(self):
        assert canonical_match("Artificial Intelligence", "AI") == 0.95

    def test_no_alias(self):
        assert canonical_match("Rust", "Go") == 0.0

    def test_same_name(self):
        # Same name should not match (na != nb guard)
        assert canonical_match("react", "react") == 0.0


# ---------------------------------------------------------------------------
# compute_name_score
# ---------------------------------------------------------------------------


class TestComputeNameScore:
    def test_returns_max_of_sub_matchers(self):
        # canonical_match("JavaScript", "JS") = 0.98 should dominate
        score = compute_name_score("JavaScript", "JS")
        assert score >= 0.98

    def test_exact_match(self):
        score = compute_name_score("Python", "python")
        assert score == 1.0

    def test_fuzzy_match(self):
        # compute_similarity handles word reordering
        score = compute_name_score("Machine Learning", "Machine Learning")
        assert score == 1.0

    def test_numeronym(self):
        score = compute_name_score("k8s", "kubernetes")
        assert score >= 0.95

    def test_low_similarity(self):
        score = compute_name_score("Rust", "Django")
        assert score < 0.5


# ---------------------------------------------------------------------------
# type_compatible
# ---------------------------------------------------------------------------


class TestTypeCompatible:
    def test_same_type(self):
        assert type_compatible("Technology", "Technology") is True

    def test_compatible_cross_type(self):
        assert type_compatible("Technology", "Software") is True

    def test_other_wildcard(self):
        assert type_compatible("Other", "Technology") is True
        assert type_compatible("Person", "Other") is True

    def test_incompatible(self):
        assert type_compatible("Person", "Technology") is False

    def test_all_declared_cross_types(self):
        for pair in COMPATIBLE_CROSS_TYPES:
            types = list(pair)
            assert type_compatible(types[0], types[1]) is True


# ---------------------------------------------------------------------------
# summary_overlap
# ---------------------------------------------------------------------------


class TestSummaryOverlap:
    def test_overlapping(self):
        a = "Python is a popular programming language for web development"
        b = "Python programming language used for data science and web"
        score = summary_overlap(a, b)
        assert score > 0.0

    def test_no_overlap(self):
        a = "Cooking pasta with tomato sauce"
        b = "Quantum physics wave function"
        score = summary_overlap(a, b)
        assert score == 0.0

    def test_none_summary(self):
        assert summary_overlap(None, "Some summary") == 0.0
        assert summary_overlap("Some summary", None) == 0.0
        assert summary_overlap(None, None) == 0.0

    def test_identical(self):
        text = "Python is a popular programming language"
        score = summary_overlap(text, text)
        assert score == 1.0

    def test_short_words_excluded(self):
        # Words < 3 chars are excluded by the regex
        a = "an is to"
        b = "an is to"
        assert summary_overlap(a, b) == 0.0


# ---------------------------------------------------------------------------
# score_merge_pair (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScoreMergePair:
    async def _make_mocks(self, embeddings=None, neighbors_a=None, neighbors_b=None):
        search_index = AsyncMock()
        graph_store = AsyncMock()

        if embeddings is not None:
            search_index.get_entity_embeddings.return_value = embeddings
        else:
            search_index.get_entity_embeddings.return_value = {}

        if neighbors_a is not None or neighbors_b is not None:
            async def _get_neighbors(eid, gid):
                if eid == "e1":
                    return neighbors_a or []
                return neighbors_b or []
            graph_store.get_active_neighbors_with_weights.side_effect = _get_neighbors
        else:
            graph_store.get_active_neighbors_with_weights.return_value = []

        return search_index, graph_store

    async def test_merge_identical_names(self):
        ea = MockEntity(id="e1", name="Python", entity_type="Technology")
        eb = MockEntity(id="e2", name="python", entity_type="Technology")

        # High embedding similarity too
        vec = np.random.randn(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        embeddings = {"e1": vec.tolist(), "e2": vec.tolist()}

        search_index, graph_store = await self._make_mocks(embeddings=embeddings)

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        assert verdict == "merge"
        assert conf >= 0.82

    async def test_reject_incompatible_types(self):
        ea = MockEntity(id="e1", name="John", entity_type="Person")
        eb = MockEntity(id="e2", name="Python", entity_type="Technology")

        search_index, graph_store = await self._make_mocks()

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        assert verdict == "keep_separate"
        assert signals.get("reason") == "incompatible_types"

    async def test_reject_different_names(self):
        ea = MockEntity(id="e1", name="Rust", entity_type="Technology")
        eb = MockEntity(id="e2", name="Django", entity_type="Technology")

        search_index, graph_store = await self._make_mocks()

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        assert verdict == "keep_separate"

    async def test_merge_canonical_alias(self):
        ea = MockEntity(id="e1", name="JavaScript", entity_type="Technology")
        eb = MockEntity(id="e2", name="JS", entity_type="Technology")

        # Add modest embedding similarity
        rng = np.random.default_rng(42)
        base = rng.standard_normal(64).astype(np.float32)
        base = base / np.linalg.norm(base)
        noise = rng.standard_normal(64).astype(np.float32) * 0.05
        vec_b = base + noise
        vec_b = vec_b / np.linalg.norm(vec_b)
        embeddings = {"e1": base.tolist(), "e2": vec_b.tolist()}

        search_index, graph_store = await self._make_mocks(embeddings=embeddings)

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        # canonical_match gives 0.98 + 0.03 type boost = 1.0 name score
        # 0.40 * 1.0 = 0.40 from name alone; with high embedding should exceed 0.82
        assert verdict == "merge"

    async def test_compatible_cross_types(self):
        ea = MockEntity(id="e1", name="React", entity_type="Technology")
        eb = MockEntity(id="e2", name="React", entity_type="Software")

        vec = np.ones(64, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        embeddings = {"e1": vec.tolist(), "e2": vec.tolist()}

        search_index, graph_store = await self._make_mocks(embeddings=embeddings)

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        # Same name (exact) + same embeddings, compatible types
        assert verdict == "merge"

    async def test_booster_name_and_embedding(self):
        """Test that high name + high embedding triggers booster to 0.95."""
        ea = MockEntity(id="e1", name="TensorFlow", entity_type="Technology")
        eb = MockEntity(id="e2", name="tensorflow", entity_type="Technology")

        vec = np.ones(64, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        embeddings = {"e1": vec.tolist(), "e2": vec.tolist()}

        search_index, graph_store = await self._make_mocks(embeddings=embeddings)

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        assert verdict == "merge"
        assert conf >= 0.95

    async def test_embedding_failure_graceful(self):
        """Embedding lookup failure should not crash the scorer."""
        ea = MockEntity(id="e1", name="Python", entity_type="Technology")
        eb = MockEntity(id="e2", name="python", entity_type="Technology")

        search_index = AsyncMock()
        search_index.get_entity_embeddings.side_effect = RuntimeError("no embeddings")
        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights.return_value = []

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        # Should still work based on name alone
        assert signals["embedding"] == 0.0
        # name_score for "Python"/"python" = 1.0 + 0.03 type boost = 1.0
        # 0.40 * 1.0 = 0.40 < 0.82, but booster: name>=0.93 requires emb>=0.80
        # So without embeddings, low confidence but not a crash
        assert isinstance(verdict, str)

    async def test_neighbor_overlap_boosts_confidence(self):
        ea = MockEntity(id="e1", name="React", entity_type="Technology")
        eb = MockEntity(id="e2", name="ReactJS", entity_type="Technology")

        neighbors_a = [("shared1", "USES", 1.0, "Entity"), ("shared2", "DEPENDS_ON", 0.8, "Entity")]
        neighbors_b = [("shared1", "USES", 1.0, "Entity"), ("shared2", "DEPENDS_ON", 0.8, "Entity")]

        vec = np.ones(64, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        embeddings = {"e1": vec.tolist(), "e2": vec.tolist()}

        search_index, graph_store = await self._make_mocks(
            embeddings=embeddings,
            neighbors_a=neighbors_a,
            neighbors_b=neighbors_b,
        )

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        assert verdict == "merge"
        assert signals["neighbor_overlap"] == 1.0
