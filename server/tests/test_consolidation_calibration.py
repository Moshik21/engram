"""Tests for consolidation distillation and calibration helpers."""

from engram.consolidation.calibration import (
    build_calibration_snapshots,
    build_distillation_examples,
)
from engram.models.consolidation import DecisionOutcomeLabel, DecisionTrace


def test_build_distillation_examples_from_outcomes_and_oracles():
    trace = DecisionTrace(
        cycle_id="cyc_test",
        group_id="test",
        phase="infer",
        candidate_type="relationship",
        candidate_id="a:MENTIONED_WITH:b",
        decision="accept",
        decision_source="llm",
        confidence=0.82,
        threshold_band="accepted",
        features={"score": 0.82},
    )
    label = DecisionOutcomeLabel(
        cycle_id="cyc_test",
        group_id="test",
        phase="infer",
        decision_trace_id=trace.id,
        outcome_type="materialization",
        label="created",
        value=1.0,
    )

    examples = build_distillation_examples(
        "cyc_test",
        "test",
        [trace],
        [label],
    )

    assert len(examples) == 2
    by_source = {example.teacher_source: example for example in examples}
    assert by_source["outcome:materialization"].teacher_label == "created"
    assert by_source["outcome:materialization"].correct is True
    assert by_source["oracle:llm"].teacher_label == "accept"
    assert by_source["oracle:llm"].correct is None


def test_build_calibration_snapshots_reports_accuracy_and_ece():
    accepted = DecisionTrace(
        cycle_id="cyc_test",
        group_id="test",
        phase="merge",
        candidate_type="entity_pair",
        candidate_id="a:b",
        decision="merge",
        decision_source="rule",
        confidence=0.9,
        threshold_band="accepted",
    )
    rejected = DecisionTrace(
        cycle_id="cyc_test",
        group_id="test",
        phase="merge",
        candidate_type="entity_pair",
        candidate_id="c:d",
        decision="keep_separate",
        decision_source="rule",
        confidence=0.6,
        threshold_band="rejected",
    )
    abstained = DecisionTrace(
        cycle_id="cyc_test",
        group_id="test",
        phase="merge",
        candidate_type="entity_pair",
        candidate_id="e:f",
        decision="abstain",
        decision_source="cross_encoder",
        confidence=0.5,
        threshold_band="uncertain_band",
    )
    labels = [
        DecisionOutcomeLabel(
            cycle_id="cyc_test",
            group_id="test",
            phase="merge",
            decision_trace_id=accepted.id,
            outcome_type="materialization",
            label="applied",
            value=1.0,
        ),
        DecisionOutcomeLabel(
            cycle_id="cyc_test",
            group_id="test",
            phase="merge",
            decision_trace_id=rejected.id,
            outcome_type="regret",
            label="rejected",
            value=0.0,
        ),
    ]

    snapshots = build_calibration_snapshots(
        "cyc_test",
        "test",
        [accepted, rejected, abstained],
        labels,
        window_cycles=3,
        min_examples=2,
        bins=2,
    )

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.phase == "merge"
    assert snapshot.total_traces == 3
    assert snapshot.labeled_examples == 2
    assert snapshot.oracle_examples == 1
    assert snapshot.abstain_count == 1
    assert snapshot.accuracy == 1.0
    assert snapshot.mean_confidence == 0.75
    assert snapshot.expected_calibration_error == 0.25
    assert snapshot.summary["teacher_sources"]["outcome:materialization"] == 1
    assert snapshot.summary["teacher_sources"]["outcome:regret"] == 1
    assert snapshot.summary["teacher_sources"]["oracle:cross_encoder"] == 1
