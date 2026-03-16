"""Tests for graph health monitoring."""

from __future__ import annotations

import uuid
from uuid import uuid4

import pytest

from engram.consolidation.health import GraphHealthMetrics, compute_graph_health


@pytest.fixture
async def graph_store():
    from engram.config import HelixDBConfig
    from engram.storage.helix.graph import HelixGraphStore


    store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def gid():
    return f"test_{uuid4().hex[:8]}"


class TestGraphHealthMetrics:
    def test_defaults(self):
        m = GraphHealthMetrics()
        assert m.entity_count == 0
        assert m.relationship_count == 0
        assert m.deferred_evidence_count == 0
        assert m.evidence_commit_rate == 0.0


class TestComputeGraphHealth:
    @pytest.mark.asyncio
    async def test_empty_graph(self, graph_store, gid):
        health = await compute_graph_health(graph_store, gid)
        assert health.entity_count == 0
        assert health.relationship_count == 0
        assert health.deferred_evidence_count == 0

    @pytest.mark.asyncio
    async def test_with_entities(self, graph_store, gid):
        from engram.models.entity import Entity
        from engram.utils.dates import utc_now

        now = utc_now()
        for i in range(3):
            e = Entity(
                id=f"e{i}_{gid}",
                name=f"Entity{i}",
                entity_type="Concept",
                group_id=gid,
                created_at=now,
                updated_at=now,
            )
            await graph_store.create_entity(e)
        health = await compute_graph_health(graph_store, gid)
        assert health.entity_count == 3

    @pytest.mark.asyncio
    async def test_with_evidence(self, graph_store, gid):
        from engram.models.episode import Episode
        from engram.utils.dates import utc_now

        ep = Episode(
            id=f"ep_health_{gid}",
            content="Test.",
            source="test",
            group_id=gid,
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
        await graph_store.store_evidence(evidence, group_id=gid)
        health = await compute_graph_health(graph_store, gid)
        assert health.deferred_evidence_count == 1
        assert health.evidence_commit_rate == 0.0

    @pytest.mark.asyncio
    async def test_commit_rate(self, graph_store, gid):
        from engram.models.episode import Episode
        from engram.utils.dates import utc_now

        ep = Episode(
            id=f"ep_rate_{gid}",
            content="Test.",
            source="test",
            group_id=gid,
            created_at=utc_now(),
        )
        await graph_store.create_episode(ep)

        evidence = []
        for i in range(4):
            evidence.append(
                {
                    "evidence_id": f"evi_rate_{gid}_{i}",
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
        await graph_store.store_evidence(evidence, group_id=gid)
        # Commit 2 of 4
        await graph_store.update_evidence_status(
            f"evi_rate_{gid}_0",
            "committed",
            group_id=gid,
        )
        await graph_store.update_evidence_status(
            f"evi_rate_{gid}_1",
            "committed",
            group_id=gid,
        )

        health = await compute_graph_health(graph_store, gid)
        assert health.evidence_commit_rate == 0.5
        assert health.deferred_evidence_count == 2

    @pytest.mark.asyncio
    async def test_approved_evidence_counts_as_unresolved(self, graph_store, gid):
        from engram.models.episode import Episode
        from engram.utils.dates import utc_now

        ep = Episode(
            id=f"ep_approved_{gid}",
            content="Test.",
            source="test",
            group_id=gid,
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
            group_id=gid,
            default_status="approved",
        )

        health = await compute_graph_health(graph_store, gid)
        assert health.deferred_evidence_count == 1
