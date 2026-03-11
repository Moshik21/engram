"""Tests for graph health monitoring."""

from __future__ import annotations

import uuid

import pytest

from engram.consolidation.health import GraphHealthMetrics, compute_graph_health


@pytest.fixture
async def graph_store():
    from engram.storage.sqlite.graph import SQLiteGraphStore

    store = SQLiteGraphStore(":memory:")
    await store.initialize()
    yield store
    await store.close()


class TestGraphHealthMetrics:
    def test_defaults(self):
        m = GraphHealthMetrics()
        assert m.entity_count == 0
        assert m.relationship_count == 0
        assert m.deferred_evidence_count == 0
        assert m.evidence_commit_rate == 0.0


class TestComputeGraphHealth:
    @pytest.mark.asyncio
    async def test_empty_graph(self, graph_store):
        health = await compute_graph_health(graph_store, "default")
        assert health.entity_count == 0
        assert health.relationship_count == 0
        assert health.deferred_evidence_count == 0

    @pytest.mark.asyncio
    async def test_with_entities(self, graph_store):
        from engram.models.entity import Entity
        from engram.utils.dates import utc_now

        now = utc_now()
        for i in range(3):
            e = Entity(
                id=f"e{i}",
                name=f"Entity{i}",
                entity_type="Concept",
                group_id="default",
                created_at=now,
                updated_at=now,
            )
            await graph_store.create_entity(e)
        health = await compute_graph_health(graph_store, "default")
        assert health.entity_count == 3

    @pytest.mark.asyncio
    async def test_with_evidence(self, graph_store):
        from engram.models.episode import Episode
        from engram.utils.dates import utc_now

        ep = Episode(
            id="ep_health",
            content="Test.",
            source="test",
            group_id="default",
            created_at=utc_now(),
        )
        await graph_store.create_episode(ep)

        evidence = [
            {
                "evidence_id": f"evi_{uuid.uuid4().hex[:12]}",
                "episode_id": "ep_health",
                "fact_class": "entity",
                "confidence": 0.85,
                "source_type": "test",
                "extractor_name": "test",
                "payload": {},
                "corroborating_signals": [],
                "created_at": "2026-03-09T00:00:00",
            },
        ]
        await graph_store.store_evidence(evidence, group_id="default")
        health = await compute_graph_health(graph_store, "default")
        assert health.deferred_evidence_count == 1
        assert health.evidence_commit_rate == 0.0

    @pytest.mark.asyncio
    async def test_commit_rate(self, graph_store):
        from engram.models.episode import Episode
        from engram.utils.dates import utc_now

        ep = Episode(
            id="ep_rate",
            content="Test.",
            source="test",
            group_id="default",
            created_at=utc_now(),
        )
        await graph_store.create_episode(ep)

        evidence = []
        for i in range(4):
            evidence.append(
                {
                    "evidence_id": f"evi_rate_{i}",
                    "episode_id": "ep_rate",
                    "fact_class": "entity",
                    "confidence": 0.85,
                    "source_type": "test",
                    "extractor_name": "test",
                    "payload": {},
                    "corroborating_signals": [],
                    "created_at": "2026-03-09T00:00:00",
                },
            )
        await graph_store.store_evidence(evidence, group_id="default")
        # Commit 2 of 4
        await graph_store.update_evidence_status(
            "evi_rate_0", "committed", group_id="default",
        )
        await graph_store.update_evidence_status(
            "evi_rate_1", "committed", group_id="default",
        )

        health = await compute_graph_health(graph_store, "default")
        assert health.evidence_commit_rate == 0.5
        assert health.deferred_evidence_count == 2

    @pytest.mark.asyncio
    async def test_approved_evidence_counts_as_unresolved(self, graph_store):
        from engram.models.episode import Episode
        from engram.utils.dates import utc_now

        ep = Episode(
            id="ep_approved",
            content="Test.",
            source="test",
            group_id="default",
            created_at=utc_now(),
        )
        await graph_store.create_episode(ep)
        await graph_store.store_evidence(
            [
                {
                    "evidence_id": f"evi_{uuid.uuid4().hex[:12]}",
                    "episode_id": "ep_approved",
                    "fact_class": "entity",
                    "confidence": 0.8,
                    "source_type": "test",
                    "extractor_name": "test",
                    "payload": {"name": "Alice", "entity_type": "Person"},
                    "corroborating_signals": [],
                    "status": "approved",
                    "commit_reason": "promoted_by_adjudication",
                    "created_at": "2026-03-09T00:00:00",
                },
            ],
            group_id="default",
            default_status="approved",
        )

        health = await compute_graph_health(graph_store, "default")
        assert health.deferred_evidence_count == 1
