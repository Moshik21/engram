from __future__ import annotations

from datetime import datetime, timezone

from engram.benchmark.metrics import RecallEvalSample, SessionContinuitySample
from engram.evaluation.brain_loop_report import (
    EVALUATION_SIGNAL_ORDER,
    build_brain_loop_report,
    evaluation_signal_failure_message,
    format_brain_loop_report_markdown,
    is_brain_loop_report_payload,
    looks_like_partial_brain_loop_report,
    merge_recall_runtime_metrics,
    missing_brain_loop_report_sections,
    unmeasured_evaluation_signals,
)
from engram.models.consolidation import CalibrationSnapshot, ConsolidationCycle, PhaseResult


def test_evaluation_package_exports_report_artifact_helpers() -> None:
    import engram.evaluation as evaluation

    for name in (
        "evaluation_signal_failure_message",
        "is_brain_loop_report_payload",
        "looks_like_partial_brain_loop_report",
        "missing_brain_loop_report_sections",
        "unmeasured_evaluation_signals",
    ):
        assert name in evaluation.__all__
        assert getattr(evaluation, name) is not None


def test_brain_loop_report_summarizes_full_loop() -> None:
    stats = {
        "episodes": 4,
        "entities": 7,
        "relationships": 3,
        "active_entities": 2,
        "cue_metrics": {
            "cue_count": 3,
            "episodes_without_cues": 1,
            "cue_coverage": 0.75,
            "cue_hit_count": 12,
            "cue_surfaced_count": 10,
            "cue_selected_count": 4,
            "cue_used_count": 3,
            "cue_near_miss_count": 2,
            "avg_policy_score": 0.81,
            "cue_to_projection_conversion_rate": 0.6667,
        },
        "projection_metrics": {
            "state_counts": {
                "queued": 1,
                "cued": 0,
                "scheduled": 0,
                "projecting": 0,
                "projected": 2,
                "failed": 1,
                "dead_letter": 0,
            },
            "attempted_episode_count": 3,
            "total_attempts": 4,
            "failure_rate": 0.3333,
            "avg_processing_duration_ms": 42.5,
            "avg_time_to_projection_ms": 1500,
            "yield": {
                "linked_entity_count": 6,
                "relationship_count": 3,
                "avg_linked_entities_per_projected_episode": 3.0,
                "avg_relationships_per_projected_episode": 1.5,
            },
        },
        "adjudication_metrics": {
            "evidence_status_counts": {"pending": 1, "deferred": 1, "approved": 0},
            "request_status_counts": {"pending": 1, "deferred": 0, "error": 1},
            "open_evidence_count": 2,
            "pending_evidence_count": 1,
            "deferred_evidence_count": 1,
            "approved_evidence_count": 0,
            "open_request_count": 2,
            "pending_request_count": 1,
            "deferred_request_count": 0,
            "error_request_count": 1,
            "open_work_count": 4,
        },
        "recall_metrics": {
            "total_analyses": 5,
            "trigger_count": 3,
            "false_recall_rate": 0.3333,
            "graph_lift_rate": 0.25,
            "probe_trigger_rate": 0.4,
            "used_count": 3,
            "dismissed_count": 1,
            "surfaced_count": 5,
            "selected_count": 2,
            "confirmed_count": 1,
            "corrected_count": 1,
            "graph_override_count": 2,
            "adaptive_thresholds_enabled": True,
            "thresholds": {"linguistic": 0.32, "borderline": 0.18, "resonance": 0.5},
            "analyzer_latency_ms": {"avg": 12.5, "p95": 31.2},
            "probe_latency_ms": {"avg_ms": 8.1, "p95_ms": 22.7},
            "family_contributions": {"linguistic": 2, "graph": 1},
        },
    }
    cycle = ConsolidationCycle(
        id="cyc_test",
        group_id="default",
        trigger="manual",
        status="completed",
        dry_run=False,
        started_at=10.0,
        completed_at=12.0,
        phase_results=[
            PhaseResult(phase="triage", status="success", items_processed=5, items_affected=3),
            PhaseResult(
                phase="edge_adjudication",
                status="error",
                items_processed=2,
                items_affected=0,
                error="judge unavailable",
            ),
        ],
    )

    report = build_brain_loop_report(
        stats,
        recent_cycles=[cycle],
        calibration_snapshots=[
            CalibrationSnapshot(
                cycle_id="cyc_test",
                group_id="default",
                phase="triage",
                window_cycles=3,
                total_traces=12,
                labeled_examples=8,
                oracle_examples=2,
                abstain_count=1,
                accuracy=0.75,
                mean_confidence=0.8,
                expected_calibration_error=0.1,
            )
        ],
        recall_samples=[
            {
                "recallTriggered": True,
                "recallHelped": True,
                "recallNeeded": True,
                "packetsSurfaced": 4,
                "packetsUsed": 2,
                "falseRecalls": 1,
            },
            RecallEvalSample(
                recall_triggered=True,
                recall_helped=False,
                packets_surfaced=2,
                packets_used=0,
                false_recalls=1,
                recall_needed=True,
            ),
            RecallEvalSample(
                recall_triggered=False,
                recall_helped=False,
                packets_surfaced=1,
                packets_used=0,
                false_recalls=0,
                recall_needed=True,
            ),
        ],
        session_samples=[
            {
                "baselineScore": 0.2,
                "memoryScore": 0.7,
                "openLoopExpected": True,
                "openLoopRecovered": True,
            },
            {
                "baseline_score": 0.5,
                "memory_score": 0.6,
                "temporal_expected": True,
                "temporal_correct": True,
            },
        ],
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert report["generated_at"] == "2026-01-01T00:00:00Z"
    assert report["cue"]["coverage"] == 0.75
    assert report["cue"]["used_rate"] == 0.3
    assert report["cue"]["projection_conversion_rate"] == 0.6667
    assert report["project"]["active_count"] == 1
    assert report["project"]["tracked_count"] == 4
    assert report["project"]["projected_rate"] == 0.5
    assert report["project"]["backlog_rate"] == 0.25
    assert report["project"]["avg_processing_duration_ms"] == 42.5
    assert report["project"]["avg_time_to_projection_ms"] == 1500
    assert report["project"]["yield"]["avg_linked_entities_per_projected_episode"] == 3.0
    assert report["recall"]["evaluation"]["memory_need_precision"] == 0.5
    assert report["recall"]["evaluation"]["memory_need_recall"] == 0.6667
    assert report["recall"]["evaluation"]["missed_recall_rate"] == 0.3333
    assert report["recall"]["evaluation"]["needed_count"] == 3
    assert report["recall"]["evaluation"]["missed_count"] == 1
    assert report["recall"]["evaluation"]["useful_packet_rate"] == 0.2857
    assert report["recall"]["evaluation"]["false_recall_rate"] == 0.2857
    assert report["recall"]["evaluation"]["surfaced_to_used_ratio"] == 3.5
    assert report["recall"]["latency"]["analyzer_ms"] == {"avg_ms": 12.5, "p95_ms": 31.2}
    assert report["recall"]["latency"]["probe_ms"] == {"avg_ms": 8.1, "p95_ms": 22.7}
    assert report["recall"]["control"] == {
        "used_count": 3,
        "dismissed_count": 1,
        "surfaced_count": 5,
        "selected_count": 2,
        "confirmed_count": 1,
        "corrected_count": 1,
        "graph_override_count": 2,
        "adaptive_thresholds_enabled": True,
        "thresholds": {"linguistic": 0.32, "borderline": 0.18, "resonance": 0.5},
    }
    assert report["recall"]["continuity"]["session_continuity_lift"] == 0.3
    assert report["recall"]["continuity"]["open_loop_recovery_rate"] == 1.0
    assert report["consolidate"]["status"] == "attention"
    assert report["consolidate"]["latest_status"] == "completed"
    assert report["consolidate"]["latest_cycle"]["error"] is None
    assert (
        report["consolidate"]["latest_cycle"]["phase_issue"]
        == "edge_adjudication: judge unavailable"
    )
    assert report["consolidate"]["phase_status_counts"] == {"success": 1, "error": 1}
    assert report["consolidate"]["phase_totals"]["triage"]["items_affected"] == 3
    assert report["consolidate"]["phase_totals"]["triage"]["effect_rate"] == 0.6
    assert report["consolidate"]["adjudication"]["status"] == "attention"
    assert report["consolidate"]["adjudication"]["runs"] == 1
    assert report["consolidate"]["adjudication"]["items_processed"] == 2
    assert report["consolidate"]["adjudication"]["items_affected"] == 0
    assert report["consolidate"]["adjudication"]["items_unaffected"] == 2
    assert report["consolidate"]["adjudication"]["effect_rate"] == 0.0
    assert report["consolidate"]["adjudication"]["error_count"] == 1
    assert report["consolidate"]["adjudication"]["open_evidence_count"] == 2
    assert report["consolidate"]["adjudication"]["open_request_count"] == 2
    assert report["consolidate"]["adjudication"]["open_work_count"] == 4
    assert report["consolidate"]["adjudication"]["evidence_status_counts"] == {
        "pending": 1,
        "deferred": 1,
        "approved": 0,
    }
    assert report["consolidate"]["adjudication"]["request_status_counts"] == {
        "pending": 1,
        "deferred": 0,
        "error": 1,
    }
    assert report["consolidate"]["effect_rate"] == 0.4286
    assert report["consolidate"]["calibration"]["status"] == "measured"
    assert report["consolidate"]["calibration"]["phase_totals"]["triage"] == {
        "snapshots": 1,
        "total_traces": 12,
        "labeled_examples": 8,
        "oracle_examples": 2,
        "abstain_count": 1,
        "accuracy": 0.75,
        "mean_confidence": 0.8,
        "expected_calibration_error": 0.1,
    }
    assert report["consolidate"]["error_count"] == 1
    assert report["evaluation_signals"]["cue_usefulness"] == {
        "status": "measured",
        "evidence_count": 10,
        "metric": 0.3,
        "gap": None,
    }
    assert report["evaluation_signals"]["projection_yield"] == {
        "status": "measured",
        "evidence_count": 2,
        "metric": 3.0,
        "gap": None,
    }
    assert report["evaluation_signals"]["recall_quality"] == {
        "status": "measured",
        "evidence_count": 3,
        "metric": 0.5,
        "gap": None,
    }
    assert report["evaluation_signals"]["false_recall"] == {
        "status": "measured",
        "evidence_count": 7,
        "metric": 0.2857,
        "gap": None,
    }
    assert report["evaluation_signals"]["triage_calibration"] == {
        "status": "measured",
        "evidence_count": 8,
        "metric": 0.1,
        "gap": None,
    }
    assert report["evaluation_signals"]["consolidation_effect"] == {
        "status": "measured",
        "evidence_count": 1,
        "metric": 0.4286,
        "gap": None,
    }
    assert report["coverage_gaps"] == []

    markdown = format_brain_loop_report_markdown(report)
    assert "projection 66.7%" in markdown
    assert "backlog 25.0%" in markdown
    assert "projection lag 1.50s" in markdown
    assert "processing 42ms" in markdown
    assert "need recall 66.7%" in markdown
    assert "missed recall 33.3%" in markdown
    assert "analyzer p95 31ms" in markdown
    assert "probe p95 23ms" in markdown
    assert "Runtime control: surfaced 5 | used 3 | dismissed 1 | graph overrides 2" in markdown
    assert "resonance threshold 0.5000" in markdown
    assert "triage accuracy 75.0%, ECE 0.1000" in markdown
    assert "Effect 42.9%" in markdown
    assert "Phase issue: edge_adjudication: judge unavailable" in markdown
    assert "Adjudication: 1 runs, effect 0.0%, unaffected 2, errors 1" in markdown
    assert "open work 4 (evidence 2, requests 2)" in markdown
    assert "Evaluation Signals" in markdown
    assert "False Recall: measured (7 evidence)" in markdown


def test_brain_loop_report_empty_data_surfaces_gaps() -> None:
    report = build_brain_loop_report({}, generated_at="2026-01-01T00:00:00Z")

    assert report["capture"]["status"] == "empty"
    assert report["cue"]["coverage"] == 0.0
    assert report["project"]["projected_count"] == 0
    assert report["recall"]["evaluation"]["status"] == "needs_samples"
    assert report["consolidate"]["status"] == "needs_cycles"
    assert report["evaluation_signals"]["false_recall"] == {
        "status": "needs_labels",
        "evidence_count": 0,
        "metric": None,
        "gap": "recall quality needs labeled recall_samples input",
    }
    assert report["evaluation_signals"]["triage_calibration"]["status"] == "needs_snapshots"
    assert "capture has no stored episodes yet" in report["coverage_gaps"]
    assert "recall quality needs labeled recall_samples input" in report["coverage_gaps"]
    assert "recall gate needs runtime analyses" in report["coverage_gaps"]

    markdown = format_brain_loop_report_markdown(report)

    assert "Engram Brain Loop Report" in markdown
    assert "Coverage Gaps" in markdown


def test_unmeasured_evaluation_signals_accepts_measured_report() -> None:
    assert unmeasured_evaluation_signals(_measured_signal_report()) == []


def test_unmeasured_evaluation_signals_reports_ordered_missing_signals() -> None:
    report = _measured_signal_report()
    report["evaluation_signals"].pop("projection_yield")
    report["evaluation_signals"].pop("false_recall")

    assert unmeasured_evaluation_signals(report) == [
        "projection_yield:missing",
        "false_recall:missing",
    ]


def test_unmeasured_evaluation_signals_reports_signal_quality_failures() -> None:
    report = _measured_signal_report()
    report["evaluation_signals"]["cue_usefulness"]["status"] = "needs_feedback"
    report["evaluation_signals"]["recall_quality"]["evidence_count"] = 0
    report["evaluation_signals"]["triage_calibration"]["metric"] = None

    assert unmeasured_evaluation_signals(report) == [
        "cue_usefulness:needs_feedback",
        "recall_quality:no_evidence",
        "triage_calibration:no_metric",
    ]


def test_unmeasured_evaluation_signals_supports_minimum_evidence_gate() -> None:
    report = _measured_signal_report()
    for signal in report["evaluation_signals"].values():
        signal["evidence_count"] = 3
    report["evaluation_signals"]["cue_usefulness"]["evidence_count"] = 2

    assert unmeasured_evaluation_signals(report, min_evidence_count=3) == [
        "cue_usefulness:insufficient_evidence(2<3)"
    ]


def test_evaluation_signal_failure_message_formats_failures() -> None:
    report = _measured_signal_report()
    report["evaluation_signals"].pop("projection_yield")

    assert evaluation_signal_failure_message(report, prefix="Operator gate") == (
        "Operator gate: ['projection_yield:missing']"
    )
    threshold_report = _measured_signal_report()
    for signal in threshold_report["evaluation_signals"].values():
        signal["evidence_count"] = 2
    threshold_report["evaluation_signals"]["recall_quality"]["evidence_count"] = 1
    assert evaluation_signal_failure_message(
        threshold_report,
        prefix="Operator gate",
        min_evidence_count=2,
    ) == "Operator gate: ['recall_quality:insufficient_evidence(1<2)']"
    assert (
        evaluation_signal_failure_message(_measured_signal_report(), prefix="Operator gate")
        is None
    )


def test_brain_loop_report_artifact_shape_helpers() -> None:
    complete = {
        "totals": {},
        "capture": {},
        "cue": {},
        "project": {},
        "recall": {},
        "consolidate": {},
        "evaluation_signals": {},
    }
    partial = {
        "loop": ["capture", "cue", "project", "recall", "consolidate"],
        "totals": {},
        "capture": {},
        "evaluation_signals": {},
    }

    assert is_brain_loop_report_payload(complete) is True
    assert looks_like_partial_brain_loop_report(complete) is False
    assert missing_brain_loop_report_sections(complete) == []
    assert is_brain_loop_report_payload(partial) is False
    assert looks_like_partial_brain_loop_report(partial) is True
    assert missing_brain_loop_report_sections(partial) == [
        "cue",
        "project",
        "recall",
        "consolidate",
    ]
    assert looks_like_partial_brain_loop_report({"stats": {"episodes": 1}}) is False


def test_brain_loop_report_flags_cues_without_feedback() -> None:
    report = build_brain_loop_report(
        {
            "episodes": 1,
            "cue_metrics": {"cue_count": 1},
            "projection_metrics": {"state_counts": {"projected": 1}},
        }
    )

    assert "cue usefulness needs surfaced cue feedback" in report["coverage_gaps"]


def test_brain_loop_report_flags_missing_recall_gate_runtime_coverage() -> None:
    stats = {
        "episodes": 1,
        "cue_metrics": {"cue_count": 1, "cue_surfaced_count": 1},
        "projection_metrics": {"state_counts": {"projected": 1}},
        "recall_metrics": {},
    }
    cycle = ConsolidationCycle(group_id="default", status="completed")
    calibration = CalibrationSnapshot(
        cycle_id=cycle.id,
        group_id="default",
        phase="triage",
        window_cycles=1,
        total_traces=1,
        labeled_examples=1,
        oracle_examples=0,
        abstain_count=0,
        accuracy=1.0,
        mean_confidence=0.9,
        expected_calibration_error=0.0,
    )
    recall_sample = RecallEvalSample(
        recall_triggered=True,
        recall_helped=True,
        recall_needed=True,
        packets_surfaced=1,
        packets_used=1,
    )
    session_sample = SessionContinuitySample(
        baseline_score=0.2,
        memory_score=0.8,
    )

    no_gate_report = build_brain_loop_report(
        stats,
        recent_cycles=[cycle],
        calibration_snapshots=[calibration],
        recall_samples=[recall_sample],
        session_samples=[session_sample],
    )

    assert "recall quality needs labeled recall_samples input" not in no_gate_report[
        "coverage_gaps"
    ]
    assert "recall gate needs runtime analyses" in no_gate_report["coverage_gaps"]

    missing_latency_report = build_brain_loop_report(
        {
            **stats,
            "recall_metrics": {"total_analyses": 1, "trigger_count": 1},
        },
        recent_cycles=[cycle],
        calibration_snapshots=[calibration],
        recall_samples=[recall_sample],
        session_samples=[session_sample],
    )

    assert "recall gate needs runtime analyses" not in missing_latency_report[
        "coverage_gaps"
    ]
    assert (
        "recall gate latency needs analyzer samples"
        in missing_latency_report["coverage_gaps"]
    )


def test_brain_loop_report_flags_false_recall_without_surfaced_packets() -> None:
    report = build_brain_loop_report(
        {
            "episodes": 1,
            "cue_metrics": {"cue_count": 1, "cue_surfaced_count": 1},
            "projection_metrics": {"state_counts": {"projected": 1}},
            "recall_metrics": {
                "total_analyses": 1,
                "trigger_count": 1,
                "analyzer_latency_ms": {"avg": 3.0, "p95": 5.0},
            },
        },
        recent_cycles=[
            ConsolidationCycle(
                group_id="default",
                status="completed",
                phase_results=[
                    PhaseResult(
                        phase="triage",
                        status="success",
                        items_processed=1,
                        items_affected=1,
                    )
                ],
            )
        ],
        calibration_snapshots=[
            CalibrationSnapshot(
                cycle_id="cyc_false_recall_gap",
                group_id="default",
                phase="triage",
                window_cycles=1,
                total_traces=1,
                labeled_examples=1,
                oracle_examples=0,
                abstain_count=0,
                accuracy=1.0,
                mean_confidence=0.9,
                expected_calibration_error=0.0,
            )
        ],
        recall_samples=[
            RecallEvalSample(
                recall_triggered=True,
                recall_helped=True,
                recall_needed=True,
                packets_surfaced=0,
                packets_used=0,
                false_recalls=0,
            )
        ],
        session_samples=[SessionContinuitySample(baseline_score=0.2, memory_score=0.8)],
    )

    assert report["evaluation_signals"]["false_recall"] == {
        "status": "needs_surfaced_packets",
        "evidence_count": 0,
        "metric": 0.0,
        "gap": "false recall needs labeled surfaced packet counts",
    }
    assert (
        "false recall needs labeled surfaced packet counts"
        in report["coverage_gaps"]
    )


def test_brain_loop_report_flags_unscored_calibration_snapshots() -> None:
    report = build_brain_loop_report(
        {
            "episodes": 1,
            "cue_metrics": {"cue_count": 1, "cue_surfaced_count": 1},
            "projection_metrics": {"state_counts": {"projected": 1}},
            "recall_metrics": {
                "total_analyses": 1,
                "trigger_count": 1,
                "analyzer_latency_ms": {"avg": 3.0, "p95": 5.0},
            },
        },
        recent_cycles=[ConsolidationCycle(group_id="default", status="completed")],
        calibration_snapshots=[
            CalibrationSnapshot(
                cycle_id="cyc_unscored",
                group_id="default",
                phase="triage",
                window_cycles=1,
                total_traces=4,
                labeled_examples=0,
                oracle_examples=0,
                abstain_count=0,
            )
        ],
        recall_samples=[
            RecallEvalSample(
                recall_triggered=True,
                recall_helped=True,
                recall_needed=True,
                packets_surfaced=1,
                packets_used=1,
            )
        ],
        session_samples=[SessionContinuitySample(baseline_score=0.2, memory_score=0.8)],
    )

    assert report["consolidate"]["status"] == "attention"
    assert report["consolidate"]["calibration"]["status"] == "needs_quality"
    assert (
        "consolidation calibration needs saved calibration snapshots"
        not in report["coverage_gaps"]
    )
    assert (
        "consolidation calibration quality needs labeled decision outcomes"
        in report["coverage_gaps"]
    )

    markdown = format_brain_loop_report_markdown(report)
    assert "Calibration snapshots: 1 (needs_quality); needs labeled decisions" in markdown


def test_brain_loop_report_marks_consolidate_attention_without_calibration_snapshots() -> None:
    report = build_brain_loop_report(
        {
            "episodes": 1,
            "cue_metrics": {"cue_count": 1, "cue_surfaced_count": 1},
            "projection_metrics": {"state_counts": {"projected": 1}},
            "recall_metrics": {
                "total_analyses": 1,
                "trigger_count": 1,
                "analyzer_latency_ms": {"avg": 3.0, "p95": 5.0},
            },
        },
        recent_cycles=[ConsolidationCycle(group_id="default", status="completed")],
        recall_samples=[
            RecallEvalSample(
                recall_triggered=True,
                recall_helped=True,
                recall_needed=True,
                packets_surfaced=1,
                packets_used=1,
            )
        ],
        session_samples=[SessionContinuitySample(baseline_score=0.2, memory_score=0.8)],
    )

    assert report["consolidate"]["status"] == "attention"
    assert report["consolidate"]["calibration"]["status"] == "needs_snapshots"
    assert (
        "consolidation calibration needs saved calibration snapshots"
        in report["coverage_gaps"]
    )


def test_brain_loop_report_marks_open_adjudication_work_as_attention() -> None:
    report = build_brain_loop_report(
        {
            "adjudication_metrics": {
                "evidence_status_counts": {"pending": 2, "deferred": 1},
                "request_status_counts": {"pending": 1, "error": 1},
            }
        },
        generated_at="2026-01-01T00:00:00Z",
    )

    adjudication = report["consolidate"]["adjudication"]
    assert report["consolidate"]["status"] == "attention"
    assert adjudication["status"] == "attention"
    assert adjudication["runs"] == 0
    assert adjudication["open_evidence_count"] == 3
    assert adjudication["open_request_count"] == 2
    assert adjudication["open_work_count"] == 5
    assert adjudication["pending_evidence_count"] == 2
    assert adjudication["deferred_evidence_count"] == 1
    assert adjudication["error_request_count"] == 1


def test_merge_recall_runtime_metrics_prefers_saved_gate_coverage() -> None:
    stats = {
        "episodes": 1,
        "recall_metrics": {"total_analyses": 0, "thresholds": {"resonance": 0.45}},
    }
    saved = {
        "total_analyses": 2,
        "trigger_count": 1,
        "analyzer_latency_ms": {"avg": 8.0, "p95": 16.0},
        "surfaced_count": 3,
    }

    merged = merge_recall_runtime_metrics(stats, saved)

    assert merged["recall_metrics"]["total_analyses"] == 2
    assert merged["recall_metrics"]["analyzer_latency_ms"]["p95"] == 16.0

    stronger_live = merge_recall_runtime_metrics(
        {
            "recall_metrics": {
                "total_analyses": 3,
                "analyzer_latency_ms": {"avg": 9.0, "p95": 20.0},
            }
        },
        saved,
    )

    assert stronger_live["recall_metrics"]["total_analyses"] == 3
    assert stronger_live["recall_metrics"]["analyzer_latency_ms"]["p95"] == 20.0


def test_brain_loop_report_accepts_graph_state_and_lifecycle_cycle_shape() -> None:
    report = build_brain_loop_report(
        {
            "stats": {
                "episodes": 1,
                "cue_metrics": {
                    "cue_count": 1,
                    "cue_coverage": 1.0,
                    "cue_surfaced_count": 1,
                },
                "projection_metrics": {
                    "state_counts": {"cueOnly": 1, "deadLetter": 1},
                    "failure_rate": 1.0,
                },
            }
        },
        recent_cycles=[
            {
                "id": "cyc_lifecycle",
                "status": "failed",
                "error": "calibration failed",
                "dryRun": True,
                "phases": [
                    {
                        "phase": "calibrate",
                        "status": "skipped",
                        "itemsProcessed": 0,
                        "itemsAffected": 0,
                    }
                ],
            }
        ],
    )

    assert report["project"]["state_counts"]["cue_only"] == 1
    assert report["project"]["dead_letter_count"] == 1
    assert report["consolidate"]["latest_cycle"]["dry_run"] is True
    assert report["consolidate"]["latest_cycle"]["error"] == "calibration failed"
    assert report["consolidate"]["latest_cycle"]["phase_issue"] is None
    assert report["consolidate"]["error_count"] == 1
    assert report["consolidate"]["phase_status_counts"] == {"skipped": 1}

    markdown = format_brain_loop_report_markdown(report)
    assert "Error: calibration failed" in markdown


def test_markdown_includes_benchmark_evidence() -> None:
    report = {
        "group_id": "operator_brain",
        "generated_at": "2026-05-18T22:00:00Z",
        "totals": {},
        "capture": {},
        "cue": {},
        "project": {},
        "recall": {},
        "consolidate": {},
        "benchmark_evidence": {
            "status": "measured",
            "benchmark": "showcase",
            "baseline": "engram_full",
            "mode": "quick",
            "scenario_count": 6,
            "passed_count": 5,
            "scenario_pass_rate": 5 / 6,
            "min_pass_rate": 0.8,
            "fairness": {
                "baseline_contract_present": True,
                "transcript_hash_count": 6,
            },
            "failures": [],
        },
    }

    markdown = format_brain_loop_report_markdown(report)

    assert "## Benchmark Evidence" in markdown
    assert "engram_full on showcase (quick): measured" in markdown
    assert "Scenarios: 6 available, 5 passed | pass rate 83.3% (minimum 80.0%)" in markdown
    assert "Fairness: baseline contract present, 6 transcript hashes" in markdown


def test_markdown_includes_human_label_evidence() -> None:
    report = {
        "group_id": "operator_brain",
        "generated_at": "2026-05-18T23:10:00Z",
        "totals": {},
        "capture": {},
        "cue": {},
        "project": {},
        "recall": {},
        "consolidate": {},
        "human_label_evidence": {
            "status": "measured",
            "source": "staging_harness",
            "client": "Cursor",
            "captured_at": "2026-05-18T23:00:00Z",
            "labeler": "operator",
            "human_labeled": True,
            "recall_sample_count": 5,
            "session_sample_count": 2,
            "min_recall_samples": 5,
            "min_session_samples": 2,
            "failures": [],
        },
    }

    markdown = format_brain_loop_report_markdown(report)

    assert "## Human Label Evidence" in markdown
    assert "Cursor from staging_harness: measured" in markdown
    assert "Labels: 5 recall samples (minimum 5), 2 session samples (minimum 2)" in markdown
    assert "Review: human labeled yes, labeler operator" in markdown


def _measured_signal_report() -> dict:
    return {
        "evaluation_signals": {
            signal: {
                "status": "measured",
                "evidence_count": 1,
                "metric": 1.0,
                "gap": None,
            }
            for signal in EVALUATION_SIGNAL_ORDER
        }
    }
