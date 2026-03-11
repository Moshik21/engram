"""Distillation and calibration helpers for consolidation decisioning."""

from __future__ import annotations

from collections import Counter, defaultdict

from engram.models.consolidation import (
    CalibrationSnapshot,
    DecisionOutcomeLabel,
    DecisionTrace,
    DistillationExample,
)

ORACLE_DECISION_SOURCES = frozenset(
    {
        "llm",
        "llm_escalation",
        "escalation",
        "cross_encoder",
    }
)

_POSITIVE_DECISIONS = frozenset(
    {
        "accept",
        "extract",
        "merge",
        "merge_applied",
        "created",
        "updated_existing",
        "applied",
    }
)
_NEGATIVE_DECISIONS = frozenset(
    {
        "reject",
        "skip",
        "keep_separate",
        "abstain",
        "rejected",
        "duplicate_skipped",
    }
)
_POSITIVE_OUTCOMES = frozenset(
    {
        "applied",
        "created",
        "updated_existing",
        "useful",
        "supported",
    }
)
_NEGATIVE_OUTCOMES = frozenset(
    {
        "duplicate_skipped",
        "rejected",
        "abstained",
        "empty",
        "failed",
        "not_applied",
    }
)
_DISTILLATION_OUTCOME_TYPES = frozenset(
    {
        "projection_yield",
        "materialization",
        "regret",
        "reuse",
    }
)


def build_distillation_examples(
    cycle_id: str,
    group_id: str,
    traces: list[DecisionTrace],
    labels: list[DecisionOutcomeLabel],
) -> list[DistillationExample]:
    """Build teacher/student examples from cycle audit artifacts."""
    labels_by_trace: dict[str, list[DecisionOutcomeLabel]] = defaultdict(list)
    for label in labels:
        labels_by_trace[label.decision_trace_id].append(label)

    examples: list[DistillationExample] = []
    seen: set[tuple[str, str, str]] = set()

    for trace in traces:
        for label in labels_by_trace.get(trace.id, []):
            if label.outcome_type not in _DISTILLATION_OUTCOME_TYPES:
                continue
            correct = _derive_correctness(trace, label)
            key = (trace.id, f"outcome:{label.outcome_type}", label.label)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                DistillationExample(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    phase=trace.phase,
                    candidate_type=trace.candidate_type,
                    candidate_id=trace.candidate_id,
                    decision_trace_id=trace.id,
                    teacher_label=label.label,
                    teacher_source=f"outcome:{label.outcome_type}",
                    student_decision=trace.decision,
                    student_confidence=trace.confidence,
                    threshold_band=trace.threshold_band,
                    features=dict(trace.features),
                    correct=correct,
                    metadata={
                        "outcome_value": label.value,
                        "outcome_metadata": label.metadata,
                        "decision_source": trace.decision_source,
                    },
                )
            )

        if trace.decision_source not in ORACLE_DECISION_SOURCES:
            continue
        key = (trace.id, f"oracle:{trace.decision_source}", trace.decision)
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            DistillationExample(
                cycle_id=cycle_id,
                group_id=group_id,
                phase=trace.phase,
                candidate_type=trace.candidate_type,
                candidate_id=trace.candidate_id,
                decision_trace_id=trace.id,
                teacher_label=trace.decision,
                teacher_source=f"oracle:{trace.decision_source}",
                student_decision=trace.decision,
                student_confidence=trace.confidence,
                threshold_band=trace.threshold_band,
                features=dict(trace.features),
                correct=None,
                metadata={"policy_version": trace.policy_version},
            )
        )

    return examples


def build_calibration_snapshots(
    cycle_id: str,
    group_id: str,
    traces: list[DecisionTrace],
    labels: list[DecisionOutcomeLabel],
    *,
    window_cycles: int,
    min_examples: int,
    bins: int,
) -> list[CalibrationSnapshot]:
    """Compute rolling phase calibration summaries from recent audit history."""
    examples = build_distillation_examples(cycle_id, group_id, traces, labels)
    traces_by_phase: dict[str, list[DecisionTrace]] = defaultdict(list)
    for trace in traces:
        traces_by_phase[trace.phase].append(trace)

    examples_by_phase: dict[str, list[DistillationExample]] = defaultdict(list)
    for example in examples:
        examples_by_phase[example.phase].append(example)

    snapshots: list[CalibrationSnapshot] = []
    for phase, phase_traces in traces_by_phase.items():
        phase_examples = examples_by_phase.get(phase, [])
        labeled = [
            example
            for example in phase_examples
            if example.correct is not None and example.student_confidence is not None
        ]
        oracle_examples = [
            example for example in phase_examples if example.teacher_source.startswith("oracle:")
        ]
        abstain_count = sum(
            1
            for trace in phase_traces
            if trace.decision in {"abstain", "defer"}
            or trace.threshold_band in {"abstained", "uncertain_band"}
        )

        accuracy = None
        mean_confidence = None
        ece = None
        bucket_summary: list[dict] = []
        if len(labeled) >= min_examples:
            accuracy = round(sum(1.0 for example in labeled if example.correct) / len(labeled), 4)
            mean_confidence = round(
                sum(float(example.student_confidence or 0.0) for example in labeled) / len(labeled),
                4,
            )
            bucket_summary, ece = _bucketize_examples(labeled, bins)

        snapshots.append(
            CalibrationSnapshot(
                cycle_id=cycle_id,
                group_id=group_id,
                phase=phase,
                window_cycles=window_cycles,
                total_traces=len(phase_traces),
                labeled_examples=len(labeled),
                oracle_examples=len(oracle_examples),
                abstain_count=abstain_count,
                accuracy=accuracy,
                mean_confidence=mean_confidence,
                expected_calibration_error=ece,
                summary={
                    "min_examples": min_examples,
                    "teacher_sources": dict(
                        Counter(example.teacher_source for example in phase_examples)
                    ),
                    "bucket_metrics": bucket_summary,
                },
            )
        )

    return snapshots


def _derive_correctness(
    trace: DecisionTrace,
    label: DecisionOutcomeLabel,
) -> bool | None:
    if label.outcome_type == "projection_yield":
        if trace.decision == "extract":
            return label.label == "useful"
        if trace.decision == "skip":
            return label.label == "empty"
        return None

    if label.label in _POSITIVE_OUTCOMES:
        return trace.decision in _POSITIVE_DECISIONS
    if label.label in _NEGATIVE_OUTCOMES:
        return trace.decision in _NEGATIVE_DECISIONS
    return None


def _bucketize_examples(
    examples: list[DistillationExample],
    bins: int,
) -> tuple[list[dict], float]:
    total = len(examples)
    bucket_width = 1.0 / bins
    bucketed: list[list[DistillationExample]] = [[] for _ in range(bins)]

    for example in examples:
        confidence = float(example.student_confidence or 0.0)
        index = min(bins - 1, max(0, int(confidence / bucket_width)))
        bucketed[index].append(example)

    summary: list[dict] = []
    ece = 0.0
    for index, bucket in enumerate(bucketed):
        low = round(index * bucket_width, 4)
        high = round((index + 1) * bucket_width, 4)
        if not bucket:
            summary.append(
                {
                    "low": low,
                    "high": high,
                    "count": 0,
                    "accuracy": None,
                    "mean_confidence": None,
                }
            )
            continue

        accuracy = sum(1.0 for example in bucket if example.correct) / len(bucket)
        mean_confidence = sum(float(example.student_confidence or 0.0) for example in bucket) / len(
            bucket
        )
        ece += abs(mean_confidence - accuracy) * (len(bucket) / total)
        summary.append(
            {
                "low": low,
                "high": high,
                "count": len(bucket),
                "accuracy": round(accuracy, 4),
                "mean_confidence": round(mean_confidence, 4),
            }
        )

    return summary, round(ece, 4)
