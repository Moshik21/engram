"""Tests for EvidenceAdjudicationPhase materialization behavior."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.phases.evidence_adjudication import EvidenceAdjudicationPhase
from engram.graph_manager import EvidenceMaterializationFailure, GraphManager
from engram.models.episode import Episode
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.utils.dates import utc_now
from tests.conftest import MockExtractor


def _evidence_row(
    episode_id: str,
    fact_class: str,
    payload: dict,
    *,
    confidence: float,
    status: str = "pending",
    commit_reason: str | None = None,
    deferred_cycles: int = 0,
) -> dict:
    return {
        "evidence_id": f"evi_{uuid.uuid4().hex[:12]}",
        "episode_id": episode_id,
        "fact_class": fact_class,
        "confidence": confidence,
        "source_type": "narrow_extractor",
        "extractor_name": "test",
        "payload": payload,
        "source_span": None,
        "corroborating_signals": [],
        "status": status,
        "commit_reason": commit_reason,
        "deferred_cycles": deferred_cycles,
        "created_at": "2026-03-09T00:00:00",
    }


async def _create_episode(
    graph_store: SQLiteGraphStore,
    episode_id: str,
    content: str,
) -> Episode:
    episode = Episode(
        id=episode_id,
        content=content,
        source="test",
        group_id="default",
        created_at=utc_now(),
    )
    await graph_store.create_episode(episode)
    return episode


@pytest_asyncio.fixture
async def graph_store(tmp_path) -> SQLiteGraphStore:
    store = SQLiteGraphStore(str(tmp_path / "evidence_adjudication.db"))
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def evidence_cfg() -> ActivationConfig:
    return ActivationConfig(
        evidence_extraction_enabled=True,
        evidence_store_deferred=True,
    )


@pytest_asyncio.fixture
async def graph_manager(
    graph_store: SQLiteGraphStore,
    evidence_cfg: ActivationConfig,
) -> GraphManager:
    search_index = SimpleNamespace(
        index_entity=AsyncMock(),
        index_episode=AsyncMock(),
    )
    activation_store = MemoryActivationStore(cfg=ActivationConfig())
    return GraphManager(
        graph_store=graph_store,
        activation_store=activation_store,
        search_index=search_index,
        extractor=MockExtractor(),
        cfg=evidence_cfg,
    )


class TestEvidenceAdjudicationPhase:
    @pytest.mark.asyncio
    async def test_skipped_when_disabled(self, graph_store):
        phase = EvidenceAdjudicationPhase()
        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=None,
            search_index=None,
            cfg=ActivationConfig(evidence_extraction_enabled=False),
            cycle_id="cyc_test",
        )
        assert result.status == "skipped"
        assert records == []

    @pytest.mark.asyncio
    async def test_skipped_when_no_pending(self, graph_store, evidence_cfg):
        phase = EvidenceAdjudicationPhase()
        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=None,
            search_index=None,
            cfg=evidence_cfg,
            cycle_id="cyc_test",
        )
        assert result.status == "skipped"
        assert records == []

    @pytest.mark.asyncio
    async def test_dry_run_no_mutations(self, graph_store, graph_manager, evidence_cfg):
        await _create_episode(graph_store, "ep_dry", "Alice works at Google.")
        await graph_store.store_evidence(
            [
                _evidence_row(
                    "ep_dry",
                    "entity",
                    {"name": "Alice", "entity_type": "Person"},
                    confidence=0.9,
                ),
            ],
            group_id="default",
        )
        phase = EvidenceAdjudicationPhase(graph_manager=graph_manager)

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=None,
            search_index=None,
            cfg=evidence_cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )

        pending = await graph_store.get_pending_evidence(group_id="default")
        assert result.status == "success"
        assert result.items_processed == 1
        assert result.items_affected == 0
        assert records == []
        assert pending[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_materializes_promotable_evidence_into_graph(
        self,
        graph_store,
        graph_manager,
        evidence_cfg,
    ):
        await _create_episode(
            graph_store,
            "ep_materialize",
            "Alice works at Google since 2026-01-15.",
        )
        rows = [
            _evidence_row(
                "ep_materialize",
                "entity",
                {"name": "Alice", "entity_type": "Person"},
                confidence=0.82,
            ),
            _evidence_row(
                "ep_materialize",
                "entity",
                {"name": "Google", "entity_type": "Organization"},
                confidence=0.82,
            ),
            _evidence_row(
                "ep_materialize",
                "relationship",
                {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
                confidence=0.84,
            ),
            _evidence_row(
                "ep_materialize",
                "temporal",
                {"temporal_marker": "2026-01-15", "nearby_entity": "Alice"},
                confidence=0.75,
            ),
        ]
        await graph_store.store_evidence(rows, group_id="default")
        phase = EvidenceAdjudicationPhase(graph_manager=graph_manager)

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=None,
            search_index=None,
            cfg=evidence_cfg,
            cycle_id="cyc_test",
        )

        stored = await graph_store.get_episode_evidence("ep_materialize", group_id="default")
        by_class = {row["fact_class"]: row for row in stored}
        alice_id = next(
            row["committed_id"] for row in stored if row["payload"].get("name") == "Alice"
        )
        google_id = next(
            row["committed_id"] for row in stored if row["payload"].get("name") == "Google"
        )
        rel_id = by_class["relationship"]["committed_id"]

        assert result.status == "success"
        assert result.items_processed == 4
        assert result.items_affected == 4
        assert {row["status"] for row in stored} == {"committed"}
        assert by_class["temporal"]["committed_id"] == rel_id
        assert by_class["relationship"]["committed_id"] == rel_id

        relationships = await graph_store.get_relationships(
            alice_id,
            direction="outgoing",
            group_id="default",
        )
        materialized = [rel for rel in relationships if rel.id == rel_id]
        assert len(materialized) == 1
        assert materialized[0].predicate == "WORKS_AT"
        assert materialized[0].target_id == google_id
        assert any(record.action == "materialized" for record in records)

    @pytest.mark.asyncio
    async def test_retries_approved_rows_after_crash(
        self,
        graph_store,
        graph_manager,
        evidence_cfg,
    ):
        await _create_episode(graph_store, "ep_retry", "Alice is a person.")
        await graph_store.store_evidence(
            [
                _evidence_row(
                    "ep_retry",
                    "entity",
                    {"name": "Alice", "entity_type": "Person"},
                    confidence=0.9,
                ),
            ],
            group_id="default",
        )
        phase = EvidenceAdjudicationPhase(graph_manager=graph_manager)
        original = graph_manager.materialize_stored_evidence
        graph_manager.materialize_stored_evidence = AsyncMock(
            side_effect=RuntimeError("boom"),
        )

        failed, _ = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=None,
            search_index=None,
            cfg=evidence_cfg,
            cycle_id="cyc_test",
        )

        after_fail = await graph_store.get_episode_evidence("ep_retry", group_id="default")
        assert failed.status == "error"
        assert after_fail[0]["status"] == "approved"

        graph_manager.materialize_stored_evidence = original
        succeeded, _ = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=None,
            search_index=None,
            cfg=evidence_cfg,
            cycle_id="cyc_retry",
        )

        after_retry = await graph_store.get_episode_evidence("ep_retry", group_id="default")
        assert succeeded.status == "success"
        assert after_retry[0]["status"] == "committed"
        assert after_retry[0]["committed_id"] is not None

    @pytest.mark.asyncio
    async def test_materialization_failure_reverts_rows_to_deferred(
        self,
        graph_store,
        graph_manager,
        evidence_cfg,
    ):
        await _create_episode(graph_store, "ep_fail", "Alice is a person.")
        await graph_store.store_evidence(
            [
                _evidence_row(
                    "ep_fail",
                    "entity",
                    {"name": "Alice", "entity_type": "Person"},
                    confidence=0.9,
                ),
            ],
            group_id="default",
        )
        phase = EvidenceAdjudicationPhase(graph_manager=graph_manager)
        graph_manager.materialize_stored_evidence = AsyncMock(
            side_effect=EvidenceMaterializationFailure("handled_failure"),
        )

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=None,
            search_index=None,
            cfg=evidence_cfg,
            cycle_id="cyc_test",
        )

        stored = await graph_store.get_episode_evidence("ep_fail", group_id="default")
        assert result.status == "success"
        assert result.items_affected == 0
        assert stored[0]["status"] == "deferred"
        assert stored[0]["deferred_cycles"] == 1
        assert any(record.action == "materialization_failed" for record in records)

    @pytest.mark.asyncio
    async def test_temporal_only_evidence_stays_deferred(
        self,
        graph_store,
        graph_manager,
        evidence_cfg,
    ):
        await _create_episode(graph_store, "ep_temporal", "Yesterday.")
        await graph_store.store_evidence(
            [
                _evidence_row(
                    "ep_temporal",
                    "temporal",
                    {"temporal_marker": "yesterday", "nearby_entity": "Alice"},
                    confidence=0.9,
                ),
            ],
            group_id="default",
        )
        phase = EvidenceAdjudicationPhase(graph_manager=graph_manager)

        result, _ = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=None,
            search_index=None,
            cfg=evidence_cfg,
            cycle_id="cyc_test",
        )

        stored = await graph_store.get_episode_evidence("ep_temporal", group_id="default")
        assert result.status == "success"
        assert result.items_affected == 0
        assert stored[0]["status"] == "deferred"
        assert stored[0]["committed_id"] is None
        assert stored[0]["deferred_cycles"] == 1

    @pytest.mark.asyncio
    async def test_cross_episode_corroboration_commits_per_episode(
        self,
        graph_store,
        graph_manager,
    ):
        cfg = ActivationConfig(
            evidence_extraction_enabled=True,
            evidence_store_deferred=True,
            evidence_commit_entity_threshold=0.65,
        )
        await _create_episode(graph_store, "ep_cross_1", "Alice mentioned once.")
        await _create_episode(graph_store, "ep_cross_2", "Alice mentioned twice.")
        await graph_store.store_evidence(
            [
                _evidence_row(
                    "ep_cross_1",
                    "entity",
                    {"name": "Alice", "entity_type": "Person"},
                    confidence=0.62,
                ),
                _evidence_row(
                    "ep_cross_2",
                    "entity",
                    {"name": "Alice", "entity_type": "Person"},
                    confidence=0.62,
                ),
            ],
            group_id="default",
        )
        phase = EvidenceAdjudicationPhase(graph_manager=graph_manager)

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=None,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_test",
        )

        ep1 = await graph_store.get_episode_evidence("ep_cross_1", group_id="default")
        ep2 = await graph_store.get_episode_evidence("ep_cross_2", group_id="default")
        assert result.status == "success"
        assert ep1[0]["status"] == "committed"
        assert ep2[0]["status"] == "committed"
        assert ep1[0]["committed_id"] == ep2[0]["committed_id"]
        assert ep1[0]["confidence"] == pytest.approx(0.67)
        assert ep2[0]["confidence"] == pytest.approx(0.67)
        assert len([record for record in records if record.action == "corroborated"]) == 2

    @pytest.mark.asyncio
    async def test_offline_materialization_does_not_run_runtime_hooks(
        self,
        graph_store,
        graph_manager,
        evidence_cfg,
    ):
        await _create_episode(graph_store, "ep_offline", "Alice is a person.")
        await graph_store.store_evidence(
            [
                _evidence_row(
                    "ep_offline",
                    "entity",
                    {"name": "Alice", "entity_type": "Person"},
                    confidence=0.9,
                ),
            ],
            group_id="default",
        )
        graph_manager._run_surprise_detection = AsyncMock(
            side_effect=AssertionError("surprise detection should not run"),
        )
        graph_manager._run_prospective_memory = AsyncMock(
            side_effect=AssertionError("prospective memory should not run"),
        )
        graph_manager._publish_projection_graph_changes = AsyncMock(
            side_effect=AssertionError("projection publish should not run"),
        )
        graph_manager._store_emotional_encoding_context = AsyncMock(
            side_effect=AssertionError("emotional encoding should not run"),
        )
        graph_manager._update_episode_status = AsyncMock(
            side_effect=AssertionError("episode status transitions should not run"),
        )
        phase = EvidenceAdjudicationPhase(graph_manager=graph_manager)

        result, _ = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=None,
            search_index=None,
            cfg=evidence_cfg,
            cycle_id="cyc_test",
        )

        assert result.status == "success"
        assert graph_manager._search.index_entity.await_count == 1
        assert graph_manager._search.index_episode.await_count == 1

    def test_evidence_key_grouping(self):
        phase = EvidenceAdjudicationPhase()
        ev1 = {
            "fact_class": "entity",
            "payload": {"name": "Alice", "entity_type": "Person"},
            "evidence_id": "e1",
        }
        ev2 = {
            "fact_class": "entity",
            "payload": {"name": "alice", "entity_type": "person"},
            "evidence_id": "e2",
        }
        assert phase._evidence_key(ev1) == phase._evidence_key(ev2)

    def test_phase_name_and_required_methods(self):
        phase = EvidenceAdjudicationPhase()
        cfg = ActivationConfig(
            evidence_extraction_enabled=True,
            evidence_store_deferred=True,
        )
        assert phase.name == "evidence_adjudication"
        methods = phase.required_graph_store_methods(cfg)
        assert "get_pending_evidence" in methods
        assert "update_evidence_status" in methods
        assert "get_entity_count" in methods
