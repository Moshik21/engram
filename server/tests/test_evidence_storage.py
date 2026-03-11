"""Tests for evidence storage (Phase 1 of v2 extractor)."""

from __future__ import annotations

import uuid

import pytest

from engram.extraction.evidence import (
    CommitDecision,
    EvidenceBundle,
    EvidenceCandidate,
)

# ── Data model tests ──


class TestEvidenceCandidate:
    def test_default_id_prefix(self):
        ev = EvidenceCandidate()
        assert ev.evidence_id.startswith("evi_")
        assert len(ev.evidence_id) == 16  # "evi_" + 12 hex

    def test_fields(self):
        ev = EvidenceCandidate(
            episode_id="ep_1",
            group_id="test",
            fact_class="entity",
            confidence=0.85,
            source_type="narrow_extractor",
            extractor_name="identity",
            payload={"name": "Alex", "entity_type": "Person"},
            source_span="My name is Alex",
        )
        assert ev.fact_class == "entity"
        assert ev.confidence == 0.85
        assert ev.payload["name"] == "Alex"

    def test_default_empty_signals(self):
        ev = EvidenceCandidate()
        assert ev.corroborating_signals == []

    def test_default_payload(self):
        ev = EvidenceCandidate()
        assert ev.payload == {}


class TestEvidenceBundle:
    def test_empty_bundle(self):
        bundle = EvidenceBundle(episode_id="ep_1")
        assert bundle.candidates == []
        assert bundle.total_ms == 0.0

    def test_bundle_with_candidates(self):
        c1 = EvidenceCandidate(fact_class="entity", confidence=0.9)
        c2 = EvidenceCandidate(fact_class="relationship", confidence=0.7)
        bundle = EvidenceBundle(
            episode_id="ep_1",
            candidates=[c1, c2],
            extractor_stats={"identity": {"count": 1, "duration_ms": 2.0}},
            total_ms=5.0,
        )
        assert len(bundle.candidates) == 2
        assert bundle.total_ms == 5.0


class TestCommitDecision:
    def test_commit_action(self):
        d = CommitDecision(
            evidence_id="evi_abc",
            action="commit",
            reason="high_confidence",
            effective_confidence=0.9,
        )
        assert d.action == "commit"
        assert d.committed_id is None

    def test_defer_action(self):
        d = CommitDecision(
            evidence_id="evi_abc",
            action="defer",
            reason="borderline",
            effective_confidence=0.62,
        )
        assert d.action == "defer"


# ── Storage tests ──


@pytest.fixture
async def graph_store():
    """Create and initialize an in-memory SQLiteGraphStore."""
    from engram.storage.sqlite.graph import SQLiteGraphStore

    store = SQLiteGraphStore(":memory:")
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def sample_evidence():
    """Sample evidence dicts for storage."""
    return [
        {
            "evidence_id": f"evi_{uuid.uuid4().hex[:12]}",
            "episode_id": "ep_test_1",
            "fact_class": "entity",
            "confidence": 0.85,
            "source_type": "narrow_extractor",
            "extractor_name": "identity",
            "payload": {"name": "Alex", "entity_type": "Person"},
            "source_span": "My name is Alex",
            "corroborating_signals": ["identity_pattern", "proper_name"],
            "created_at": "2026-03-09T00:00:00",
        },
        {
            "evidence_id": f"evi_{uuid.uuid4().hex[:12]}",
            "episode_id": "ep_test_1",
            "fact_class": "relationship",
            "confidence": 0.72,
            "source_type": "narrow_extractor",
            "extractor_name": "relationship",
            "payload": {
                "subject": "Alex",
                "predicate": "WORKS_AT",
                "object": "Anthropic",
            },
            "corroborating_signals": ["verb_pattern"],
            "created_at": "2026-03-09T00:00:00",
        },
    ]


@pytest.fixture
async def store_with_episode(graph_store):
    """Graph store with a test episode already inserted."""
    from engram.models.episode import Episode
    from engram.utils.dates import utc_now

    ep = Episode(
        id="ep_test_1",
        content="My name is Alex. I work at Anthropic.",
        source="test",
        group_id="default",
        created_at=utc_now(),
    )
    await graph_store.create_episode(ep)
    return graph_store


class TestStoreEvidence:
    @pytest.mark.asyncio
    async def test_store_empty_list(self, store_with_episode):
        await store_with_episode.store_evidence([], group_id="default")

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, store_with_episode, sample_evidence):
        await store_with_episode.store_evidence(sample_evidence, group_id="default")
        results = await store_with_episode.get_episode_evidence(
            "ep_test_1",
            group_id="default",
        )
        assert len(results) == 2
        # ordered by confidence DESC
        assert results[0]["confidence"] >= results[1]["confidence"]

    @pytest.mark.asyncio
    async def test_pending_evidence(self, store_with_episode, sample_evidence):
        await store_with_episode.store_evidence(sample_evidence, group_id="default")
        pending = await store_with_episode.get_pending_evidence(group_id="default")
        assert len(pending) == 2
        assert all(p["status"] == "pending" for p in pending)

    @pytest.mark.asyncio
    async def test_pending_evidence_limit(self, store_with_episode, sample_evidence):
        await store_with_episode.store_evidence(sample_evidence, group_id="default")
        pending = await store_with_episode.get_pending_evidence(
            group_id="default",
            limit=1,
        )
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_pending_includes_deferred_status(
        self,
        store_with_episode,
        sample_evidence,
    ):
        deferred = [dict(sample_evidence[0], status="deferred", deferred_cycles=2)]
        await store_with_episode.store_evidence(
            deferred,
            group_id="default",
            default_status="deferred",
        )
        pending = await store_with_episode.get_pending_evidence(group_id="default")
        assert len(pending) == 1
        assert pending[0]["status"] == "deferred"
        assert pending[0]["deferred_cycles"] == 2

    @pytest.mark.asyncio
    async def test_pending_includes_approved_status(
        self,
        store_with_episode,
        sample_evidence,
    ):
        approved = [
            dict(
                sample_evidence[0],
                status="approved",
                commit_reason="promoted_by_adjudication",
            ),
        ]
        await store_with_episode.store_evidence(
            approved,
            group_id="default",
            default_status="approved",
        )
        pending = await store_with_episode.get_pending_evidence(group_id="default")
        assert len(pending) == 1
        assert pending[0]["status"] == "approved"
        assert pending[0]["commit_reason"] == "promoted_by_adjudication"

    @pytest.mark.asyncio
    async def test_pending_excludes_superseded_status(
        self,
        store_with_episode,
        sample_evidence,
    ):
        superseded = [
            dict(
                sample_evidence[0],
                status="superseded",
                commit_reason="superseded_by_adjudication:adj_123",
            ),
        ]
        await store_with_episode.store_evidence(
            superseded,
            group_id="default",
            default_status="superseded",
        )
        pending = await store_with_episode.get_pending_evidence(group_id="default")
        assert pending == []

    @pytest.mark.asyncio
    async def test_store_committed_status_excluded_from_pending(
        self,
        store_with_episode,
        sample_evidence,
    ):
        committed = [
            dict(
                sample_evidence[0],
                status="committed",
                commit_reason="committed_on_hot_path",
            ),
        ]
        await store_with_episode.store_evidence(
            committed,
            group_id="default",
            default_status="committed",
        )
        episode_rows = await store_with_episode.get_episode_evidence(
            "ep_test_1",
            group_id="default",
        )
        assert episode_rows[0]["status"] == "committed"
        assert episode_rows[0]["resolved_at"] is not None
        pending = await store_with_episode.get_pending_evidence(group_id="default")
        assert pending == []

    @pytest.mark.asyncio
    async def test_update_evidence_status_commit(
        self,
        store_with_episode,
        sample_evidence,
    ):
        await store_with_episode.store_evidence(sample_evidence, group_id="default")
        evi_id = sample_evidence[0]["evidence_id"]
        await store_with_episode.update_evidence_status(
            evi_id,
            "committed",
            updates={"commit_reason": "high_confidence", "committed_id": "ent_123"},
            group_id="default",
        )
        results = await store_with_episode.get_episode_evidence(
            "ep_test_1",
            group_id="default",
        )
        committed = [r for r in results if r["evidence_id"] == evi_id][0]
        assert committed["status"] == "committed"
        assert committed["commit_reason"] == "high_confidence"
        assert committed["committed_id"] == "ent_123"
        assert committed["resolved_at"] is not None

    @pytest.mark.asyncio
    async def test_update_evidence_status_defer(
        self,
        store_with_episode,
        sample_evidence,
    ):
        await store_with_episode.store_evidence(sample_evidence, group_id="default")
        evi_id = sample_evidence[0]["evidence_id"]
        await store_with_episode.update_evidence_status(
            evi_id,
            "deferred",
            updates={"deferred_cycles": 1},
            group_id="default",
        )
        results = await store_with_episode.get_episode_evidence(
            "ep_test_1",
            group_id="default",
        )
        deferred = [r for r in results if r["evidence_id"] == evi_id][0]
        assert deferred["status"] == "deferred"
        assert deferred["deferred_cycles"] == 1
        # deferred does NOT set resolved_at
        assert deferred["resolved_at"] is None

    @pytest.mark.asyncio
    async def test_pending_excludes_committed(
        self,
        store_with_episode,
        sample_evidence,
    ):
        await store_with_episode.store_evidence(sample_evidence, group_id="default")
        evi_id = sample_evidence[0]["evidence_id"]
        await store_with_episode.update_evidence_status(
            evi_id,
            "committed",
            group_id="default",
        )
        pending = await store_with_episode.get_pending_evidence(group_id="default")
        assert len(pending) == 1
        assert pending[0]["evidence_id"] != evi_id

    @pytest.mark.asyncio
    async def test_duplicate_evidence_ignored(
        self,
        store_with_episode,
        sample_evidence,
    ):
        await store_with_episode.store_evidence(sample_evidence, group_id="default")
        # INSERT OR IGNORE
        await store_with_episode.store_evidence(sample_evidence, group_id="default")
        results = await store_with_episode.get_episode_evidence(
            "ep_test_1",
            group_id="default",
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_payload_roundtrip(self, store_with_episode, sample_evidence):
        await store_with_episode.store_evidence(sample_evidence, group_id="default")
        results = await store_with_episode.get_episode_evidence(
            "ep_test_1",
            group_id="default",
        )
        entity_ev = [r for r in results if r["fact_class"] == "entity"][0]
        assert entity_ev["payload"]["name"] == "Alex"
        assert entity_ev["payload"]["entity_type"] == "Person"

    @pytest.mark.asyncio
    async def test_ambiguity_fields_roundtrip(
        self,
        store_with_episode,
        sample_evidence,
    ):
        rows = [
            dict(
                sample_evidence[0],
                ambiguity_tags=["negation_scope"],
                ambiguity_score=0.66,
                adjudication_request_id="adj_123",
                status="pending",
                commit_reason="needs_adjudication",
            ),
        ]
        await store_with_episode.store_evidence(rows, group_id="default")
        stored = await store_with_episode.get_episode_evidence(
            "ep_test_1",
            group_id="default",
        )
        assert stored[0]["ambiguity_tags"] == ["negation_scope"]
        assert stored[0]["ambiguity_score"] == 0.66
        assert stored[0]["adjudication_request_id"] == "adj_123"


class TestAdjudicationRequests:
    @pytest.mark.asyncio
    async def test_store_and_fetch_requests(self, store_with_episode):
        await store_with_episode.store_adjudication_requests(
            [
                {
                    "request_id": "adj_123",
                    "episode_id": "ep_test_1",
                    "ambiguity_tags": ["negation_scope"],
                    "evidence_ids": ["evi_1"],
                    "selected_text": "I work at Google, but maybe not anymore.",
                    "request_reason": "needs_adjudication:negation_scope",
                    "created_at": "2026-03-09T00:00:00",
                },
            ],
            group_id="default",
        )

        episode_requests = await store_with_episode.get_episode_adjudications(
            "ep_test_1",
            group_id="default",
        )
        request = await store_with_episode.get_adjudication_request(
            "adj_123",
            group_id="default",
        )
        pending = await store_with_episode.get_pending_adjudication_requests(
            group_id="default",
        )

        assert len(episode_requests) == 1
        assert request is not None
        assert request["request_id"] == "adj_123"
        assert request["ambiguity_tags"] == ["negation_scope"]
        assert pending[0]["request_id"] == "adj_123"

    @pytest.mark.asyncio
    async def test_update_request_terminal_state(self, store_with_episode):
        await store_with_episode.store_adjudication_requests(
            [
                {
                    "request_id": "adj_456",
                    "episode_id": "ep_test_1",
                    "ambiguity_tags": ["coreference"],
                    "evidence_ids": ["evi_2"],
                    "selected_text": "She reminded me about the dentist.",
                    "request_reason": "needs_adjudication:coreference",
                    "created_at": "2026-03-09T00:00:00",
                },
            ],
            group_id="default",
        )

        await store_with_episode.update_adjudication_request(
            "adj_456",
            {
                "status": "materialized",
                "resolution_source": "client_adjudication",
                "resolution_payload": {"relationships": []},
            },
            group_id="default",
        )
        updated = await store_with_episode.get_adjudication_request(
            "adj_456",
            group_id="default",
        )

        assert updated is not None
        assert updated["status"] == "materialized"
        assert updated["resolution_source"] == "client_adjudication"
        assert updated["resolved_at"] is not None


class TestGetEntityCount:
    @pytest.mark.asyncio
    async def test_empty_group(self, graph_store):
        count = await graph_store.get_entity_count(group_id="default")
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_excludes_deleted(self, graph_store):
        from engram.models.entity import Entity
        from engram.utils.dates import utc_now

        now = utc_now()
        e1 = Entity(
            id="e1",
            name="Alice",
            entity_type="Person",
            group_id="default",
            created_at=now,
            updated_at=now,
        )
        e2 = Entity(
            id="e2",
            name="Bob",
            entity_type="Person",
            group_id="default",
            created_at=now,
            updated_at=now,
        )
        await graph_store.create_entity(e1)
        await graph_store.create_entity(e2)
        # Soft-delete e2
        await graph_store.delete_entity("e2", soft=True, group_id="default")
        count = await graph_store.get_entity_count(group_id="default")
        assert count == 1
