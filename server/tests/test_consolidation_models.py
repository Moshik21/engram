"""Tests for consolidation data models."""

from engram.models.consolidation import (
    ConsolidationCycle,
    ConsolidationStatus,
    InferredEdge,
    MergeRecord,
    PhaseResult,
    PruneRecord,
)


class TestConsolidationModels:
    """Verify model creation and defaults."""

    def test_phase_result_defaults(self):
        pr = PhaseResult(phase="merge")
        assert pr.phase == "merge"
        assert pr.status == "success"
        assert pr.items_processed == 0
        assert pr.items_affected == 0
        assert pr.duration_ms == 0.0
        assert pr.error is None

    def test_consolidation_cycle_auto_id(self):
        c = ConsolidationCycle(group_id="test")
        assert c.id.startswith("cyc_")
        assert len(c.id) == 16  # "cyc_" + 12 hex chars
        assert c.status == "pending"
        assert c.dry_run is True
        assert c.trigger == "manual"

    def test_merge_record_auto_id(self):
        m = MergeRecord(
            cycle_id="cyc_abc",
            group_id="test",
            keep_id="e1",
            remove_id="e2",
            keep_name="Alice",
            remove_name="alice",
            similarity=0.92,
        )
        assert m.id.startswith("mrg_")
        assert m.relationships_transferred == 0
        assert m.timestamp > 0

    def test_inferred_edge_auto_id(self):
        e = InferredEdge(
            cycle_id="cyc_abc",
            group_id="test",
            source_id="e1",
            target_id="e2",
            source_name="Python",
            target_name="FastAPI",
            co_occurrence_count=5,
            confidence=0.75,
        )
        assert e.id.startswith("inf_")
        assert e.co_occurrence_count == 5

    def test_prune_record_auto_id(self):
        p = PruneRecord(
            cycle_id="cyc_abc",
            group_id="test",
            entity_id="e1",
            entity_name="Old Thing",
            entity_type="Concept",
            reason="dead_entity",
        )
        assert p.id.startswith("prn_")
        assert p.reason == "dead_entity"

    def test_consolidation_status_enum(self):
        assert ConsolidationStatus.PENDING.value == "pending"
        assert ConsolidationStatus.RUNNING.value == "running"
        assert ConsolidationStatus.COMPLETED.value == "completed"
        assert ConsolidationStatus.FAILED.value == "failed"
        assert ConsolidationStatus.CANCELLED.value == "cancelled"

    def test_cycle_unique_ids(self):
        c1 = ConsolidationCycle(group_id="test")
        c2 = ConsolidationCycle(group_id="test")
        assert c1.id != c2.id

    def test_inferred_edge_default_infer_type(self):
        e = InferredEdge(
            cycle_id="cyc_abc",
            group_id="test",
            source_id="e1",
            target_id="e2",
            source_name="A",
            target_name="B",
            co_occurrence_count=5,
            confidence=0.75,
        )
        assert e.infer_type == "co_occurrence"

    def test_inferred_edge_transitivity_type(self):
        e = InferredEdge(
            cycle_id="cyc_abc",
            group_id="test",
            source_id="e1",
            target_id="e2",
            source_name="A",
            target_name="B",
            co_occurrence_count=0,
            confidence=0.64,
            infer_type="transitivity",
        )
        assert e.infer_type == "transitivity"
        assert e.co_occurrence_count == 0

    def test_inferred_edge_new_field_defaults(self):
        e = InferredEdge(
            cycle_id="cyc_abc",
            group_id="test",
            source_id="e1",
            target_id="e2",
            source_name="A",
            target_name="B",
            co_occurrence_count=5,
            confidence=0.75,
        )
        assert e.pmi_score is None
        assert e.llm_verdict is None
        assert e.relationship_id is None

    def test_inferred_edge_pmi_llm_fields(self):
        e = InferredEdge(
            cycle_id="cyc_abc",
            group_id="test",
            source_id="e1",
            target_id="e2",
            source_name="A",
            target_name="B",
            co_occurrence_count=5,
            confidence=0.82,
            infer_type="llm_validated",
            pmi_score=2.5,
            llm_verdict="approved",
            relationship_id="rel_abc123",
        )
        assert e.infer_type == "llm_validated"
        assert e.pmi_score == 2.5
        assert e.llm_verdict == "approved"
        assert e.relationship_id == "rel_abc123"
