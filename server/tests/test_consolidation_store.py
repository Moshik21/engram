"""Tests for SQLiteConsolidationStore."""

import time

import pytest
import pytest_asyncio

from engram.consolidation.store import SQLiteConsolidationStore
from engram.models.consolidation import (
    CalibrationSnapshot,
    ConsolidationCycle,
    DecisionOutcomeLabel,
    DecisionTrace,
    DistillationExample,
    IdentifierReviewRecord,
    InferredEdge,
    MergeRecord,
    PhaseResult,
    PruneRecord,
)


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteConsolidationStore(str(tmp_path / "consol.db"))
    await s.initialize()
    yield s
    await s.close()


class TestConsolidationStore:
    """CRUD operations on consolidation store."""

    @pytest.mark.asyncio
    async def test_save_and_get_cycle(self, store):
        cycle = ConsolidationCycle(group_id="test", trigger="manual")
        cycle.phase_results.append(PhaseResult(phase="merge", items_processed=10, items_affected=2))
        await store.save_cycle(cycle)

        fetched = await store.get_cycle(cycle.id, "test")
        assert fetched is not None
        assert fetched.id == cycle.id
        assert fetched.trigger == "manual"
        assert len(fetched.phase_results) == 1
        assert fetched.phase_results[0].items_affected == 2

    @pytest.mark.asyncio
    async def test_update_cycle(self, store):
        cycle = ConsolidationCycle(group_id="test")
        await store.save_cycle(cycle)

        cycle.status = "completed"
        cycle.completed_at = time.time()
        cycle.total_duration_ms = 123.4
        await store.update_cycle(cycle)

        fetched = await store.get_cycle(cycle.id, "test")
        assert fetched.status == "completed"
        assert fetched.total_duration_ms == 123.4

    @pytest.mark.asyncio
    async def test_get_recent_cycles(self, store):
        for i in range(5):
            c = ConsolidationCycle(group_id="test")
            c.started_at = time.time() + i
            await store.save_cycle(c)

        recent = await store.get_recent_cycles("test", limit=3)
        assert len(recent) == 3
        # Should be newest first
        assert recent[0].started_at >= recent[1].started_at

    @pytest.mark.asyncio
    async def test_group_id_filtering(self, store):
        c1 = ConsolidationCycle(group_id="group_a")
        c2 = ConsolidationCycle(group_id="group_b")
        await store.save_cycle(c1)
        await store.save_cycle(c2)

        result_a = await store.get_recent_cycles("group_a")
        result_b = await store.get_recent_cycles("group_b")
        assert len(result_a) == 1
        assert len(result_b) == 1
        assert result_a[0].group_id == "group_a"

    @pytest.mark.asyncio
    async def test_save_merge_record(self, store):
        record = MergeRecord(
            cycle_id="cyc_test",
            group_id="test",
            keep_id="e1",
            remove_id="e2",
            keep_name="Alice",
            remove_name="alice",
            similarity=0.92,
            decision_confidence=0.97,
            decision_source="multi_signal",
            decision_reason="identifier_exact_match",
            relationships_transferred=3,
        )
        await store.save_merge_record(record)

        records = await store.get_merge_records("cyc_test", "test")
        assert len(records) == 1
        assert records[0].keep_name == "Alice"
        assert records[0].decision_confidence == 0.97
        assert records[0].decision_source == "multi_signal"
        assert records[0].decision_reason == "identifier_exact_match"
        assert records[0].relationships_transferred == 3

    @pytest.mark.asyncio
    async def test_save_identifier_review_record(self, store):
        record = IdentifierReviewRecord(
            cycle_id="cyc_test",
            group_id="test",
            entity_a_id="e1",
            entity_b_id="e2",
            entity_a_name="1712061",
            entity_b_name="1712018",
            entity_a_type="Identifier",
            entity_b_type="Identifier",
            raw_similarity=0.86,
            adjusted_similarity=0.89,
            decision_source="fuzzy_threshold",
            decision_reason="identifier_mismatch",
            canonical_identifier_a="1712061",
            canonical_identifier_b="1712018",
        )
        await store.save_identifier_review_record(record)

        records = await store.get_identifier_review_records("cyc_test", "test")
        assert len(records) == 1
        assert records[0].entity_a_name == "1712061"
        assert records[0].decision_reason == "identifier_mismatch"
        assert records[0].review_status == "quarantined"

    @pytest.mark.asyncio
    async def test_save_inferred_edge(self, store):
        edge = InferredEdge(
            cycle_id="cyc_test",
            group_id="test",
            source_id="e1",
            target_id="e2",
            source_name="Python",
            target_name="FastAPI",
            co_occurrence_count=5,
            confidence=0.75,
        )
        await store.save_inferred_edge(edge)

        edges = await store.get_inferred_edges("cyc_test", "test")
        assert len(edges) == 1
        assert edges[0].source_name == "Python"

    @pytest.mark.asyncio
    async def test_save_prune_record(self, store):
        record = PruneRecord(
            cycle_id="cyc_test",
            group_id="test",
            entity_id="e1",
            entity_name="Dead Entity",
            entity_type="Concept",
            reason="dead_entity",
        )
        await store.save_prune_record(record)

        records = await store.get_prune_records("cyc_test", "test")
        assert len(records) == 1
        assert records[0].entity_name == "Dead Entity"

    @pytest.mark.asyncio
    async def test_cleanup_old_records(self, store):
        old_cycle = ConsolidationCycle(group_id="test")
        old_cycle.started_at = time.time() - (100 * 86400)  # 100 days ago
        await store.save_cycle(old_cycle)

        new_cycle = ConsolidationCycle(group_id="test")
        await store.save_cycle(new_cycle)

        # Add records to old cycle
        await store.save_merge_record(
            MergeRecord(
                cycle_id=old_cycle.id,
                group_id="test",
                keep_id="e1",
                remove_id="e2",
                keep_name="A",
                remove_name="B",
                similarity=0.9,
            )
        )

        deleted = await store.cleanup(ttl_days=90)
        assert deleted == 1

        remaining = await store.get_recent_cycles("test")
        assert len(remaining) == 1
        assert remaining[0].id == new_cycle.id

    @pytest.mark.asyncio
    async def test_save_decision_trace(self, store):
        trace = DecisionTrace(
            cycle_id="cyc_test",
            group_id="test",
            phase="merge",
            candidate_type="entity_pair",
            candidate_id="a:b",
            decision="merge",
            decision_source="multi_signal",
            confidence=0.91,
            threshold_band="accepted",
            features={"name": 0.95},
            constraints_hit=["same_type"],
            metadata={"origin": "ann"},
        )
        await store.save_decision_trace(trace)

        traces = await store.get_decision_traces("cyc_test", "test")
        assert len(traces) == 1
        assert traces[0].candidate_id == "a:b"
        assert traces[0].features["name"] == 0.95
        assert traces[0].constraints_hit == ["same_type"]

    @pytest.mark.asyncio
    async def test_save_decision_outcome_label(self, store):
        label = DecisionOutcomeLabel(
            cycle_id="cyc_test",
            group_id="test",
            phase="infer",
            decision_trace_id="dtr_1",
            outcome_type="validation",
            label="accept",
            value=1.0,
            metadata={"infer_type": "auto_validated"},
        )
        await store.save_decision_outcome_label(label)

        labels = await store.get_decision_outcome_labels("cyc_test", "test")
        assert len(labels) == 1
        assert labels[0].decision_trace_id == "dtr_1"
        assert labels[0].metadata["infer_type"] == "auto_validated"

    @pytest.mark.asyncio
    async def test_save_distillation_example(self, store):
        example = DistillationExample(
            cycle_id="cyc_test",
            group_id="test",
            phase="merge",
            candidate_type="entity_pair",
            candidate_id="a:b",
            decision_trace_id="dtr_1",
            teacher_label="merge",
            teacher_source="oracle:llm",
            student_decision="merge",
            student_confidence=0.87,
            threshold_band="accepted",
            features={"name_similarity": 0.91},
            correct=True,
            metadata={"policy_version": "identity_v1"},
        )

        await store.save_distillation_example(example)

        examples = await store.get_distillation_examples("cyc_test", "test")
        assert len(examples) == 1
        assert examples[0].teacher_source == "oracle:llm"
        assert examples[0].correct is True
        assert examples[0].features["name_similarity"] == 0.91

    @pytest.mark.asyncio
    async def test_save_calibration_snapshot(self, store):
        snapshot = CalibrationSnapshot(
            cycle_id="cyc_test",
            group_id="test",
            phase="infer",
            window_cycles=5,
            total_traces=12,
            labeled_examples=8,
            oracle_examples=2,
            abstain_count=1,
            accuracy=0.875,
            mean_confidence=0.79,
            expected_calibration_error=0.08,
            summary={"teacher_sources": {"outcome:materialization": 8}},
        )

        await store.save_calibration_snapshot(snapshot)

        snapshots = await store.get_calibration_snapshots("cyc_test", "test")
        assert len(snapshots) == 1
        assert snapshots[0].phase == "infer"
        assert snapshots[0].accuracy == 0.875
        assert snapshots[0].summary["teacher_sources"]["outcome:materialization"] == 8
