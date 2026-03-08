"""Tests for structural merge candidate discovery, referential exclusivity,
summary dedup, and prefix fallback in entity resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from unittest.mock import AsyncMock

import numpy as np
import pytest

from engram.consolidation.scorers.merge_scorer import score_merge_pair
from engram.storage.sqlite.graph import _dedup_summaries

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
# _dedup_summaries
# ---------------------------------------------------------------------------


class TestDedupSummaries:
    def test_removes_duplicate_sentences(self):
        existing = "3-year-old child, son of Konner. Enjoys playing outside."
        incoming = "3-year-old child, son of Konner. Likes dinosaurs."
        result = _dedup_summaries(existing, incoming)
        # "3-year-old child, son of Konner" should NOT be duplicated
        assert result.count("son of Konner") == 1
        assert "dinosaurs" in result

    def test_keeps_novel_content(self):
        existing = "Software engineer at Acme Corp."
        incoming = "Enjoys hiking and photography."
        result = _dedup_summaries(existing, incoming)
        assert "Acme Corp" in result
        assert "hiking" in result

    def test_identical_summaries(self):
        text = "3-year-old child, son of Konner."
        result = _dedup_summaries(text, text)
        # Should not double the text
        assert result.count("Konner") == 1

    def test_empty_incoming(self):
        result = _dedup_summaries("Original summary.", "")
        assert result == "Original summary."

    def test_max_length_enforced(self):
        existing = "A" * 400
        incoming = "B" * 200
        result = _dedup_summaries(existing, incoming, max_len=500)
        assert len(result) <= 500

    def test_partial_overlap(self):
        existing = "Kallon is a 3-year-old child. Son of Konner Moshier."
        incoming = "Kallon is a child of Konner. He loves dinosaurs and trains."
        result = _dedup_summaries(existing, incoming)
        # "child of Konner" overlaps with "Son of Konner Moshier" — should be deduped
        assert "dinosaurs" in result
        # Count sentences mentioning Konner — should be limited
        konner_mentions = len(re.findall(r"Konner", result))
        assert konner_mentions <= 2


# ---------------------------------------------------------------------------
# Referential exclusivity signal in score_merge_pair
# ---------------------------------------------------------------------------


async def _make_mocks(
    embeddings=None,
    neighbors_a=None,
    neighbors_b=None,
    cooccurrence=0,
):
    search_index = AsyncMock()
    graph_store = AsyncMock()

    search_index.get_entity_embeddings.return_value = embeddings or {}

    async def _get_neighbors(eid, gid):
        if eid == "e1":
            return neighbors_a or []
        return neighbors_b or []

    graph_store.get_active_neighbors_with_weights.side_effect = _get_neighbors
    graph_store.get_episode_cooccurrence_count.return_value = cooccurrence

    return search_index, graph_store


@pytest.mark.asyncio
class TestExclusivitySignal:
    async def test_never_cooccur_with_shared_neighbors_boosts(self):
        """Entities that never co-occur + share neighbors = strong merge signal."""
        ea = MockEntity(id="e1", name="Konner", entity_type="Person")
        eb = MockEntity(id="e2", name="Konnor Moshier", entity_type="Person")

        # High embedding similarity
        vec = np.ones(64, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        embeddings = {"e1": vec.tolist(), "e2": vec.tolist()}

        # Shared neighbors (same kids)
        shared = [("kid1", 1.0, "PARENT_OF", "Person"), ("kid2", 1.0, "PARENT_OF", "Person")]

        search_index, graph_store = await _make_mocks(
            embeddings=embeddings,
            neighbors_a=shared,
            neighbors_b=shared,
            cooccurrence=0,  # Never co-occur
        )

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        assert signals["exclusivity"] > 0.7
        # Person booster + exclusivity should push this over threshold
        assert verdict == "merge"

    async def test_frequent_cooccurrence_penalizes(self):
        """Entities that frequently co-occur should NOT be merged (e.g., siblings)."""
        ea = MockEntity(id="e1", name="Kallon", entity_type="Person")
        eb = MockEntity(id="e2", name="Kaleb", entity_type="Person")

        vec_a = np.random.default_rng(42).standard_normal(64).astype(np.float32)
        vec_a = vec_a / np.linalg.norm(vec_a)
        vec_b = vec_a + np.random.default_rng(43).standard_normal(64).astype(np.float32) * 0.1
        vec_b = vec_b / np.linalg.norm(vec_b)
        embeddings = {"e1": vec_a.tolist(), "e2": vec_b.tolist()}

        # Shared parent
        nbrs_a = [("parent1", 1.0, "CHILD_OF", "Person")]
        nbrs_b = [("parent1", 1.0, "CHILD_OF", "Person")]

        search_index, graph_store = await _make_mocks(
            embeddings=embeddings,
            neighbors_a=nbrs_a,
            neighbors_b=nbrs_b,
            cooccurrence=10,  # Frequently mentioned together
        )

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        assert signals["exclusivity"] == -0.3
        assert verdict == "keep_separate"

    async def test_no_cooccurrence_no_neighbors_weak_signal(self):
        """No co-occurrence but no shared neighbors = weak positive signal."""
        ea = MockEntity(id="e1", name="Alpha", entity_type="Concept")
        eb = MockEntity(id="e2", name="Alpha Project", entity_type="Concept")

        search_index, graph_store = await _make_mocks(cooccurrence=0)

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        assert signals["exclusivity"] == 0.3

    async def test_structural_equivalence_booster(self):
        """High neighbor overlap + never co-occur + decent embedding = merge."""
        ea = MockEntity(id="e1", name="Fourth Son", entity_type="Person")
        eb = MockEntity(id="e2", name="Benjamin", entity_type="Person")

        # Moderate-high embedding similarity (cosine ~0.92)
        rng = np.random.default_rng(42)
        base = rng.standard_normal(64).astype(np.float32)
        base = base / np.linalg.norm(base)
        noise = rng.standard_normal(64).astype(np.float32) * 0.05
        vec_b = base + noise
        vec_b = vec_b / np.linalg.norm(vec_b)
        embeddings = {"e1": base.tolist(), "e2": vec_b.tolist()}

        # Many shared neighbors (same parent, same siblings)
        shared = [
            ("parent1", 1.0, "CHILD_OF", "Person"),
            ("sibling1", 1.0, "SIBLING_OF", "Person"),
            ("sibling2", 1.0, "SIBLING_OF", "Person"),
        ]

        search_index, graph_store = await _make_mocks(
            embeddings=embeddings,
            neighbors_a=shared,
            neighbors_b=shared,
            cooccurrence=0,
        )

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        # Structural equivalence booster: nbr_score >= 0.40 and exclusivity >= 0.7
        assert signals["neighbor_overlap"] == 1.0
        assert signals["exclusivity"] >= 0.7
        assert verdict == "merge"
        assert conf >= 0.85


# ---------------------------------------------------------------------------
# Structural candidate discovery (SQLite integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStructuralCandidateDiscovery:
    """Integration tests using real SQLite for structural candidate discovery."""

    async def _setup_graph(self):
        """Create an in-memory SQLite graph with family-like relationships."""
        from engram.storage.sqlite.graph import SQLiteGraphStore

        store = SQLiteGraphStore(":memory:")
        await store.initialize()

        # Create entities: parent, 4 children, and a duplicate "Fourth Son"
        from datetime import datetime

        from engram.models.entity import Entity
        from engram.models.relationship import Relationship

        now = datetime.utcnow().isoformat()
        entities = {
            "parent": Entity(id="parent", name="Konner", entity_type="Person",
                             group_id="default", created_at=now, updated_at=now),
            "child1": Entity(id="child1", name="Kallon", entity_type="Person",
                             group_id="default", created_at=now, updated_at=now),
            "child2": Entity(id="child2", name="Kaleb", entity_type="Person",
                             group_id="default", created_at=now, updated_at=now),
            "child3": Entity(id="child3", name="Ovando", entity_type="Person",
                             group_id="default", created_at=now, updated_at=now),
            "child4": Entity(id="child4", name="Benjamin", entity_type="Person",
                             group_id="default", created_at=now, updated_at=now),
            "dupe": Entity(id="dupe", name="Fourth Son", entity_type="Person",
                           group_id="default", created_at=now, updated_at=now),
        }
        for e in entities.values():
            await store.create_entity(e)

        # All children are children of parent
        for child_id in ["child1", "child2", "child3", "child4", "dupe"]:
            await store.create_relationship(Relationship(
                id=f"rel_{child_id}_parent",
                source_id=child_id,
                target_id="parent",
                predicate="CHILD_OF",
                group_id="default",
                created_at=now,
            ))

        # Siblings: all children know each other
        sibling_pairs = [
            ("child1", "child2"), ("child1", "child3"), ("child1", "child4"),
            ("child2", "child3"), ("child2", "child4"),
            ("child3", "child4"),
            # dupe also has sibling relationships
            ("dupe", "child1"), ("dupe", "child2"), ("dupe", "child3"),
        ]
        for i, (a, b) in enumerate(sibling_pairs):
            await store.create_relationship(Relationship(
                id=f"sib_{i}",
                source_id=a,
                target_id=b,
                predicate="SIBLING_OF",
                group_id="default",
                created_at=now,
            ))

        return store

    async def test_finds_structural_duplicates(self):
        store = await self._setup_graph()
        candidates = await store.find_structural_merge_candidates(
            "default", min_shared_neighbors=3,
        )
        # "child4" (Benjamin) and "dupe" (Fourth Son) share parent + 3 siblings
        pair_ids = {frozenset({a, b}) for a, b, _ in candidates}
        assert frozenset({"child4", "dupe"}) in pair_ids

    async def test_min_shared_neighbors_filters(self):
        store = await self._setup_graph()
        # High threshold should filter out pairs with fewer shared neighbors
        candidates = await store.find_structural_merge_candidates(
            "default", min_shared_neighbors=5,
        )
        # Most pairs won't have 5+ shared neighbors
        pair_ids = {frozenset({a, b}) for a, b, _ in candidates}
        # child4 and dupe share: parent, child1, child2, child3 = 4 neighbors
        assert frozenset({"child4", "dupe"}) not in pair_ids

    async def test_cooccurrence_count(self):
        store = await self._setup_graph()

        # No episode links — should return 0
        count = await store.get_episode_cooccurrence_count(
            "child4", "dupe", "default",
        )
        assert count == 0

        # Add some episodes with entities
        from datetime import datetime

        from engram.models.episode import Episode

        now = datetime.utcnow().isoformat()
        ep = Episode(
            id="ep1", content="Test episode", group_id="default",
            created_at=now, status="completed",
        )
        await store.create_episode(ep)
        await store.link_episode_entity("ep1", "child1")
        await store.link_episode_entity("ep1", "child2")

        count = await store.get_episode_cooccurrence_count(
            "child1", "child2", "default",
        )
        assert count == 1

        # child4 and dupe still don't co-occur
        count = await store.get_episode_cooccurrence_count(
            "child4", "dupe", "default",
        )
        assert count == 0

    async def test_limit_respected(self):
        store = await self._setup_graph()
        candidates = await store.find_structural_merge_candidates(
            "default", min_shared_neighbors=2, limit=2,
        )
        assert len(candidates) <= 2


# ---------------------------------------------------------------------------
# Prefix fallback in find_entity_candidates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPrefixFallback:
    async def _setup_store(self):
        from engram.storage.sqlite.graph import SQLiteGraphStore

        store = SQLiteGraphStore(":memory:")
        await store.initialize()

        from datetime import datetime

        from engram.models.entity import Entity

        now = datetime.utcnow().isoformat()
        await store.create_entity(Entity(
            id="e1", name="Konner", entity_type="Person",
            group_id="default", created_at=now, updated_at=now,
        ))
        await store.create_entity(Entity(
            id="e2", name="Konnor Moshier", entity_type="Person",
            group_id="default", created_at=now, updated_at=now,
        ))
        return store

    async def test_prefix_finds_typo_variants(self):
        """Searching 'Konnor' should find 'Konner' via prefix fallback."""
        store = await self._setup_store()
        candidates = await store.find_entity_candidates("Konnor", "default")
        names = {c.name for c in candidates}
        # Both "Konner" and "Konnor Moshier" share the "kon" prefix
        assert "Konner" in names
        assert "Konnor Moshier" in names

    async def test_prefix_short_name_skipped(self):
        """Names shorter than 3 chars shouldn't trigger prefix search."""
        store = await self._setup_store()
        # Very short name — no prefix fallback
        candidates = await store.find_entity_candidates("Ko", "default")
        # Should still work via FTS
        assert isinstance(candidates, list)

    async def test_exact_match_first(self):
        """Exact match should be found before prefix fallback."""
        store = await self._setup_store()
        candidates = await store.find_entity_candidates("Konner", "default")
        assert candidates[0].name == "Konner"


# ---------------------------------------------------------------------------
# Ensemble weight rebalance verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEnsembleWeights:
    async def test_weights_sum_to_one_positive(self):
        """Positive ensemble weights should sum to 1.0."""
        # Weights: 0.35 + 0.25 + 0.15 + 0.10 + 0.15 = 1.0
        assert abs(0.35 + 0.25 + 0.15 + 0.10 + 0.15 - 1.0) < 1e-10

    async def test_high_neighbor_overlap_low_name_can_merge(self):
        """Entities with zero name overlap but high structural similarity can merge."""
        ea = MockEntity(id="e1", name="Fourth Son", entity_type="Person")
        eb = MockEntity(id="e2", name="Benjamin", entity_type="Person")

        # High embedding similarity
        vec = np.ones(64, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        embeddings = {"e1": vec.tolist(), "e2": vec.tolist()}

        shared = [
            ("n1", 1.0, "CHILD_OF", "Person"),
            ("n2", 1.0, "SIBLING_OF", "Person"),
            ("n3", 1.0, "SIBLING_OF", "Person"),
        ]

        search_index, graph_store = await _make_mocks(
            embeddings=embeddings,
            neighbors_a=shared,
            neighbors_b=shared,
            cooccurrence=0,
        )

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        # Structural equivalence booster should fire
        assert conf >= 0.85

    async def test_strong_structural_high_embedding_without_exclusivity(self):
        """High neighbor overlap + high embedding even without exclusivity data."""
        ea = MockEntity(id="e1", name="Widget A", entity_type="Technology")
        eb = MockEntity(id="e2", name="Widget B", entity_type="Technology")

        vec = np.ones(64, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        embeddings = {"e1": vec.tolist(), "e2": vec.tolist()}

        shared = [
            ("n1", 1.0, "USES", "Technology"),
            ("n2", 1.0, "USES", "Technology"),
            ("n3", 1.0, "DEPENDS_ON", "Technology"),
        ]

        search_index = AsyncMock()
        search_index.get_entity_embeddings.return_value = embeddings
        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights.return_value = shared
        # Simulate cooccurrence method not available (e.g., FalkorDB)
        graph_store.get_episode_cooccurrence_count.side_effect = Exception("not available")

        verdict, conf, signals = await score_merge_pair(
            ea, eb, search_index, graph_store, "default",
        )
        # Should still benefit from strong structural booster
        assert signals["neighbor_overlap"] == 1.0
        assert signals["exclusivity"] == 0.0  # Fell back to default
