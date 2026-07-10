"""Operator report for the Capture -> Cue -> Project -> Recall -> Consolidate loop."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from engram.benchmark.metrics import (
    RecallEvalSample,
    SessionContinuitySample,
    false_recall_rate,
    memory_need_precision,
    memory_need_recall,
    missed_recall_rate,
    open_loop_recovery_rate,
    session_continuity_lift,
    temporal_correctness,
    useful_packet_rate,
)

LOOP = ["capture", "cue", "project", "recall", "consolidate"]
PROJECT_ACTIVE_STATES = {"queued", "cued", "scheduled", "projecting"}
ADJUDICATION_PHASES = {"evidence_adjudication", "edge_adjudication"}
BRAIN_LOOP_REPORT_SECTION_KEYS = (
    "totals",
    "capture",
    "cue",
    "project",
    "recall",
    "consolidate",
)
BRAIN_LOOP_REPORT_MARKER_KEYS = BRAIN_LOOP_REPORT_SECTION_KEYS + (
    "loop",
    "memory_value",
    "evaluation_signals",
    "coverage_gaps",
)
EVALUATION_SIGNAL_ORDER = (
    "cue_usefulness",
    "projection_yield",
    "recall_quality",
    "false_recall",
    "triage_calibration",
    "consolidation_effect",
)
REQUIRED_EVALUATION_SIGNALS = set(EVALUATION_SIGNAL_ORDER)


def has_recall_runtime_metrics(metrics: Mapping[str, Any] | None) -> bool:
    """Return whether recall metrics include real runtime gate coverage."""
    total_analyses, _analyzer_p95, _surfaced = _recall_runtime_score(_mapping(metrics or {}))
    return total_analyses > 0


def has_memory_operation_metrics(metrics: Mapping[str, Any] | None) -> bool:
    """Return whether metrics include real memory operation cost coverage."""
    operation_count, _p95_ms, signal_count = _memory_operation_score(_mapping(metrics or {}))
    return operation_count > 0 or signal_count > 0


def unmeasured_evaluation_signals(
    report: Mapping[str, Any],
    *,
    min_evidence_count: int = 1,
) -> list[str]:
    """Return required evaluation signals that are missing or not measured."""
    evaluation_signals = _mapping(report.get("evaluation_signals"))
    min_evidence_count = max(1, int(min_evidence_count))
    missing_signals = [name for name in EVALUATION_SIGNAL_ORDER if name not in evaluation_signals]
    failures = [f"{name}:missing" for name in missing_signals]
    for name in EVALUATION_SIGNAL_ORDER:
        if name in missing_signals:
            continue
        signal = _mapping(evaluation_signals.get(name))
        if signal.get("status") != "measured":
            failures.append(f"{name}:{signal.get('status', 'missing')}")
            continue
        evidence_count = _int(signal.get("evidence_count"))
        if evidence_count <= 0:
            failures.append(f"{name}:no_evidence")
        elif evidence_count < min_evidence_count:
            failures.append(f"{name}:insufficient_evidence({evidence_count}<{min_evidence_count})")
        elif signal.get("metric") is None:
            failures.append(f"{name}:no_metric")
    return failures


def evaluation_signal_failure_message(
    report: Mapping[str, Any],
    *,
    prefix: str,
    min_evidence_count: int = 1,
) -> str | None:
    """Return a human-readable failure message for unmeasured evaluation signals."""
    failures = unmeasured_evaluation_signals(
        report,
        min_evidence_count=min_evidence_count,
    )
    if not failures:
        return None
    return f"{prefix}: {failures}"


def unmeasured_memory_value(report: Mapping[str, Any]) -> list[str]:
    """Return memory-value gate failures for cost/benefit evidence."""
    memory_value = _mapping(_get(report, "memory_value", "memoryValue"))
    if not memory_value:
        return ["memory_value:missing"]

    failures: list[str] = []
    status = str(memory_value.get("status") or "missing")
    if status != "measured":
        failures.append(f"memory_value:{status}")

    cost = _mapping(memory_value.get("cost"))
    cost_status = str(cost.get("status") or "missing")
    if cost_status != "measured":
        failures.append(f"memory_value.cost:{cost_status}")

    benefit = _mapping(memory_value.get("benefit"))
    benefit_status = str(benefit.get("status") or "missing")
    if benefit_status != "measured":
        failures.append(f"memory_value.benefit:{benefit_status}")
    return failures


def memory_value_failure_message(
    report: Mapping[str, Any],
    *,
    prefix: str,
) -> str | None:
    """Return a human-readable failure message for memory-value gates."""
    failures = unmeasured_memory_value(report)
    if not failures:
        return None
    return f"{prefix}: {failures}"


def build_release_evidence_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the shared release-readiness contract for report consumers."""
    evaluation_component = _release_evaluation_signal_component(report)
    human_label_component = _release_evidence_component(
        report.get("human_label_evidence"),
        missing_key="human_label_evidence",
    )
    adoption_component = _release_evidence_component(
        report.get("adoption_evidence"),
        missing_key="adoption_evidence",
    )
    adoption_client_component = _release_adoption_client_component(
        report.get("adoption_client_evidence")
    )

    components = {
        "evaluation_signals": evaluation_component,
        "human_labels": human_label_component,
        "adoption": adoption_component,
        "adoption_clients": adoption_client_component,
    }
    missing = [
        missing_key
        for component in components.values()
        for missing_key in component.get("missing", [])
    ]
    failures = [
        str(failure)
        for component in components.values()
        for failure in component.get("failures", [])
    ]

    if any(component.get("status") == "failed" for component in components.values()):
        status = "failed"
    elif evaluation_component.get("status") != "measured":
        status = "needs_signals"
    elif missing:
        status = "needs_evidence"
    else:
        status = "measured"

    return {
        "status": status,
        "components": components,
        "missing": missing,
        "failures": failures,
    }


def with_release_evidence_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a report copy with a freshly computed release evidence summary."""
    payload = dict(report)
    payload["release_evidence"] = build_release_evidence_summary(payload)
    return payload


def is_brain_loop_report_payload(payload: Any) -> bool:
    """Return whether a JSON payload is a complete brain-loop report artifact."""
    if not isinstance(payload, Mapping):
        return False
    return all(
        isinstance(payload.get(section), Mapping) for section in BRAIN_LOOP_REPORT_SECTION_KEYS
    )


def looks_like_partial_brain_loop_report(payload: Any) -> bool:
    """Return whether a payload looks report-shaped but is incomplete."""
    if not isinstance(payload, Mapping):
        return False
    if is_brain_loop_report_payload(payload):
        return False
    if any(key in payload for key in ("stats", "graph_state", "graphState")):
        return False
    return any(key in payload for key in BRAIN_LOOP_REPORT_MARKER_KEYS)


def missing_brain_loop_report_sections(payload: Mapping[str, Any]) -> list[str]:
    """Return required report sections missing from a report-shaped payload."""
    return [
        section
        for section in BRAIN_LOOP_REPORT_SECTION_KEYS
        if not isinstance(payload.get(section), Mapping)
    ]


def merge_recall_runtime_metrics(
    stats: Mapping[str, Any],
    saved_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Overlay saved Recall Gate metrics when live stats have less coverage."""
    stats_payload = dict(_mapping(stats))
    saved = _mapping(saved_metrics or {})
    if not has_recall_runtime_metrics(saved):
        return stats_payload

    current = _mapping(_get(stats_payload, "recall_metrics", "recallMetrics", default={}))
    if _recall_runtime_score(saved) > _recall_runtime_score(current):
        stats_payload["recall_metrics"] = dict(saved)
    return stats_payload


def merge_memory_operation_metrics(
    stats: Mapping[str, Any],
    saved_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Overlay saved memory operation metrics when live stats have less coverage."""
    stats_payload = dict(_mapping(stats))
    saved = _mapping(saved_metrics or {})
    if not has_memory_operation_metrics(saved):
        return stats_payload

    current = _mapping(
        _get(
            stats_payload,
            "memory_operation_metrics",
            "memoryOperationMetrics",
            default={},
        )
    )
    if _memory_operation_score(saved) > _memory_operation_score(current):
        stats_payload["memory_operation_metrics"] = dict(saved)
    return stats_payload


def build_brain_loop_report(
    stats: Mapping[str, Any],
    *,
    group_id: str = "default",
    recent_cycles: Sequence[Any] | None = None,
    calibration_snapshots: Sequence[Any] | None = None,
    recall_samples: Sequence[RecallEvalSample | Mapping[str, Any]] | None = None,
    session_samples: Sequence[SessionContinuitySample | Mapping[str, Any]] | None = None,
    generated_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable local report from graph/runtime stats.

    The report intentionally accepts plain dictionaries so it can be used from tests,
    SQLite/lite scripts, API exports, or future dashboard endpoints without booting the
    full FastAPI app.
    """
    stats_payload = _stats_payload(stats)
    cue_metrics = _mapping(stats_payload.get("cue_metrics"))
    projection_metrics = _mapping(stats_payload.get("projection_metrics"))
    recall_metrics = _mapping(stats_payload.get("recall_metrics"))
    memory_operation_metrics = _mapping(
        _get(
            stats_payload,
            "memory_operation_metrics",
            "memoryOperationMetrics",
            default={},
        )
    )
    adjudication_metrics = _mapping(stats_payload.get("adjudication_metrics"))

    episode_count = _int(stats_payload.get("episodes"))
    entity_count = _int(stats_payload.get("entities"))
    relationship_count = _int(stats_payload.get("relationships"))
    active_entity_count = _int(stats_payload.get("active_entities"))

    capture = _capture_summary(stats_payload, episode_count)
    cue = _cue_summary(cue_metrics, episode_count)
    project = _project_summary(projection_metrics)
    recall = _recall_summary(recall_metrics, recall_samples, session_samples)
    memory_value = _memory_value_summary(recall, memory_operation_metrics)
    consolidate = _consolidation_summary(
        recent_cycles or [],
        calibration_snapshots or [],
        adjudication_metrics,
    )

    coverage_gaps = _coverage_gaps(
        episode_count=episode_count,
        cue=cue,
        project=project,
        recall=recall,
        memory_value=memory_value,
        consolidate=consolidate,
    )
    evaluation_signals = _evaluation_signals(
        cue=cue,
        project=project,
        recall=recall,
        consolidate=consolidate,
    )

    report = {
        "group_id": group_id,
        "generated_at": _generated_at(generated_at),
        "loop": LOOP,
        "totals": {
            "episodes": episode_count,
            "entities": entity_count,
            "relationships": relationship_count,
            "active_entities": active_entity_count,
        },
        "capture": capture,
        "cue": cue,
        "project": project,
        "recall": recall,
        "memory_value": memory_value,
        "consolidate": consolidate,
        "evaluation_signals": evaluation_signals,
        "coverage_gaps": coverage_gaps,
    }
    return with_release_evidence_summary(report)


def format_brain_loop_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown view for operators and local CLI usage."""
    totals = _mapping(report.get("totals"))
    capture = _mapping(report.get("capture"))
    cue = _mapping(report.get("cue"))
    project = _mapping(report.get("project"))
    recall = _mapping(report.get("recall"))
    recall_eval = _mapping(recall.get("evaluation"))
    continuity = _mapping(recall.get("continuity"))
    recall_latency = _mapping(recall.get("latency"))
    analyzer_latency = _mapping(recall_latency.get("analyzer_ms"))
    probe_latency = _mapping(recall_latency.get("probe_ms"))
    recall_control = _mapping(recall.get("control"))
    thresholds = _mapping(recall_control.get("thresholds"))
    memory_value = _mapping(_get(report, "memory_value", "memoryValue"))
    memory_cost = _mapping(memory_value.get("cost"))
    memory_benefit = _mapping(memory_value.get("benefit"))
    consolidate = _mapping(report.get("consolidate"))
    latest_cycle = _mapping(_get(consolidate, "latest_cycle", "latestCycle"))
    latest_cycle_error = _get(latest_cycle, "error")
    latest_cycle_phase_issue = _get(latest_cycle, "phase_issue", "phaseIssue")
    latest_cycle_issue_text = ""
    if isinstance(latest_cycle_error, str) and latest_cycle_error.strip():
        latest_cycle_issue_text = f" | Error: {latest_cycle_error}"
    elif isinstance(latest_cycle_phase_issue, str) and latest_cycle_phase_issue.strip():
        latest_cycle_issue_text = f" | Phase issue: {latest_cycle_phase_issue}"
    adjudication = _mapping(consolidate.get("adjudication"))
    calibration = _mapping(consolidate.get("calibration"))
    calibration_detail = _calibration_markdown_detail(calibration)
    gaps = list(report.get("coverage_gaps") or [])
    evaluation_signals = _mapping(report.get("evaluation_signals"))

    lines = [
        "# Engram Brain Loop Report",
        "",
        f"Group: `{report.get('group_id', 'default')}`",
        f"Generated: `{report.get('generated_at', '')}`",
        "",
        "## Totals",
        "",
        (
            f"- Episodes: {totals.get('episodes', 0)} | Entities: "
            f"{totals.get('entities', 0)} | Relationships: {totals.get('relationships', 0)}"
        ),
        "",
        "## Capture -> Cue -> Project",
        "",
        (
            f"- Capture: {capture.get('episode_count', 0)} episodes "
            f"({capture.get('status', 'unknown')})"
        ),
        (
            f"- Cue: {cue.get('cue_count', 0)} cues, coverage "
            f"{_pct(cue.get('coverage'))}, used rate {_pct(cue.get('used_rate'))}, "
            f"projection {_pct(cue.get('projection_conversion_rate'))}"
        ),
        (
            f"- Project: {project.get('projected_count', 0)} projected, "
            f"failure rate {_pct(project.get('failure_rate'))}, "
            f"backlog {_pct(project.get('backlog_rate'))}, "
            f"projection lag {_duration(project.get('avg_time_to_projection_ms'))}, "
            f"processing {_duration(project.get('avg_processing_duration_ms'))}, "
            f"{project.get('yield', {}).get('linked_entity_count', 0)} linked entities"
        ),
        "",
        "## Recall",
        "",
        (
            f"- Runtime triggers: {recall.get('trigger_count', 0)} from "
            f"{recall.get('total_analyses', 0)} analyses | analyzer p95 "
            f"{_duration(analyzer_latency.get('p95_ms'))} | probe p95 "
            f"{_duration(probe_latency.get('p95_ms'))}"
        ),
        (
            f"- Runtime control: surfaced {recall_control.get('surfaced_count', 0)} | "
            f"used {recall_control.get('used_count', 0)} | dismissed "
            f"{recall_control.get('dismissed_count', 0)} | graph overrides "
            f"{recall_control.get('graph_override_count', 0)} | resonance threshold "
            f"{_number(thresholds.get('resonance'))}"
        ),
        (
            f"- Labeled quality: precision {_pct(recall_eval.get('memory_need_precision'))}, "
            f"need recall {_pct(recall_eval.get('memory_need_recall'))}, "
            f"useful packet rate {_pct(recall_eval.get('useful_packet_rate'))}, "
            f"stale packet rate {_pct(recall_eval.get('stale_packet_rate'))}, "
            f"corrected packet rate {_pct(recall_eval.get('corrected_packet_rate'))}, "
            f"false recall {_pct(recall_eval.get('false_recall_rate'))}, "
            f"missed recall {_pct(recall_eval.get('missed_recall_rate'))}"
        ),
        (
            f"- Continuity: lift {_number(continuity.get('session_continuity_lift'))}, "
            f"open-loop recovery {_pct(continuity.get('open_loop_recovery_rate'))}, "
            f"temporal correctness {_pct(continuity.get('temporal_correctness'))}"
        ),
        "",
        "## Memory Value",
        "",
        (
            f"- Status: {memory_value.get('status', 'needs_samples')} | operations "
            f"{memory_cost.get('operation_count', 0)} | avg added "
            f"{_duration(memory_cost.get('avg_added_latency_ms'))} | p95 added "
            f"{_duration(memory_cost.get('p95_added_latency_ms'))}"
        ),
        (
            f"- Cost controls: timeout {_pct(memory_cost.get('timeout_rate'))}, "
            f"budget miss {_pct(memory_cost.get('budget_miss_rate'))}, "
            f"cache hit {_pct(memory_cost.get('cache_hit_rate'))}, "
            f"skipped {memory_cost.get('skipped_count', 0)}"
        ),
        (
            f"- Benefit: precision {_pct(memory_benefit.get('memory_need_precision'))}, "
            f"need recall {_pct(memory_benefit.get('memory_need_recall'))}, "
            f"useful packet rate {_pct(memory_benefit.get('useful_packet_rate'))}, "
            f"stale packet rate {_pct(memory_benefit.get('stale_packet_rate'))}, "
            f"corrected packet rate {_pct(memory_benefit.get('corrected_packet_rate'))}, "
            f"continuity lift {_number(memory_benefit.get('session_continuity_lift'))}"
        ),
        "",
        "## Consolidate",
        "",
        (
            f"- Recent cycles: {consolidate.get('cycle_count', 0)} | Latest: "
            f"{consolidate.get('latest_status') or 'none'} | Processed: "
            f"{consolidate.get('items_processed', 0)} | Affected: "
            f"{consolidate.get('items_affected', 0)} | Effect "
            f"{_pct(consolidate.get('effect_rate'))}{latest_cycle_issue_text}"
        ),
        (
            f"- Calibration snapshots: {calibration.get('snapshot_count', 0)} "
            f"({calibration.get('status', 'unknown')}){calibration_detail}"
        ),
        (
            f"- Adjudication: {adjudication.get('runs', 0)} runs, "
            f"effect {_pct(adjudication.get('effect_rate'))}, "
            f"unaffected {adjudication.get('items_unaffected', 0)}, "
            f"errors {adjudication.get('error_count', 0)}, "
            f"open work {adjudication.get('open_work_count', 0)} "
            f"(evidence {adjudication.get('open_evidence_count', 0)}, "
            f"requests {adjudication.get('open_request_count', 0)})"
        ),
    ]

    if evaluation_signals:
        lines.extend(["", "## Evaluation Signals", ""])
        for signal in EVALUATION_SIGNAL_ORDER:
            payload = _mapping(evaluation_signals.get(signal))
            if not payload:
                continue
            gap = payload.get("gap")
            gap_text = f" | {gap}" if isinstance(gap, str) and gap else ""
            lines.append(
                f"- {signal.replace('_', ' ').title()}: "
                f"{payload.get('status', 'unknown')} "
                f"({payload.get('evidence_count', 0)} evidence)"
                f"{gap_text}"
            )

    release_evidence = _mapping(report.get("release_evidence"))
    if release_evidence:
        release_missing = list(release_evidence.get("missing") or [])
        release_failures = list(release_evidence.get("failures") or [])
        missing_text = ", ".join(str(item) for item in release_missing) or "none"
        failure_text = ", ".join(str(item) for item in release_failures) or "none"
        lines.extend(
            [
                "",
                "## Release Evidence",
                "",
                f"- Status: {release_evidence.get('status', 'unknown')}",
                f"- Missing: {missing_text}",
                f"- Failures: {failure_text}",
            ]
        )
        release_components = _mapping(release_evidence.get("components"))
        for component_name in (
            "evaluation_signals",
            "human_labels",
            "adoption",
            "adoption_clients",
        ):
            component = _mapping(release_components.get(component_name))
            if not component:
                continue
            component_failures = list(component.get("failures") or [])
            component_failure_text = (
                f" | failures: {', '.join(str(item) for item in component_failures)}"
                if component_failures
                else ""
            )
            lines.append(
                f"- {component_name.replace('_', ' ').title()}: "
                f"{component.get('status', 'unknown')}{component_failure_text}"
            )

    benchmark_evidence = _mapping(report.get("benchmark_evidence"))
    if benchmark_evidence:
        benchmark_failures = list(benchmark_evidence.get("failures") or [])
        failure_text = (
            f" | failures: {', '.join(str(failure) for failure in benchmark_failures)}"
            if benchmark_failures
            else ""
        )
        fairness = _mapping(benchmark_evidence.get("fairness"))
        lines.extend(
            [
                "",
                "## Benchmark Evidence",
                "",
                (
                    f"- {benchmark_evidence.get('baseline', 'unknown')} on "
                    f"{benchmark_evidence.get('benchmark', 'unknown')} "
                    f"({benchmark_evidence.get('mode') or 'unknown'}): "
                    f"{benchmark_evidence.get('status', 'unknown')}{failure_text}"
                ),
                (
                    f"- Scenarios: {benchmark_evidence.get('scenario_count', 0)} "
                    f"available, {benchmark_evidence.get('passed_count', 0)} passed | "
                    f"pass rate {_pct(benchmark_evidence.get('scenario_pass_rate'))} "
                    f"(minimum {_pct(benchmark_evidence.get('min_pass_rate'))})"
                ),
                (
                    f"- Fairness: baseline contract "
                    f"{'present' if fairness.get('baseline_contract_present') else 'missing'}, "
                    f"{fairness.get('transcript_hash_count', 0)} transcript hashes"
                ),
            ]
        )

    human_label_evidence = _mapping(report.get("human_label_evidence"))
    if human_label_evidence:
        human_label_failures = list(human_label_evidence.get("failures") or [])
        failure_text = (
            f" | failures: {', '.join(str(failure) for failure in human_label_failures)}"
            if human_label_failures
            else ""
        )
        lines.extend(
            [
                "",
                "## Human Label Evidence",
                "",
                (
                    f"- {human_label_evidence.get('client') or 'unknown client'} from "
                    f"{human_label_evidence.get('source') or 'unknown source'}: "
                    f"{human_label_evidence.get('status', 'unknown')}{failure_text}"
                ),
                (
                    f"- Labels: {human_label_evidence.get('recall_sample_count', 0)} "
                    f"recall samples (minimum "
                    f"{human_label_evidence.get('min_recall_samples', 0)}), "
                    f"{human_label_evidence.get('session_sample_count', 0)} "
                    f"session samples (minimum "
                    f"{human_label_evidence.get('min_session_samples', 0)})"
                ),
                (
                    f"- Review: human labeled "
                    f"{'yes' if human_label_evidence.get('human_labeled') else 'no'}, "
                    f"labeler {human_label_evidence.get('labeler') or 'unknown'}, "
                    f"captured {human_label_evidence.get('captured_at') or 'unknown'}"
                ),
                (
                    f"- Artifact: {human_label_evidence.get('artifact_path') or 'unknown'} "
                    f"| sha256 {human_label_evidence.get('artifact_sha256') or 'unknown'}"
                ),
            ]
        )

    adoption_evidence = _mapping(report.get("adoption_evidence"))
    if adoption_evidence:
        adoption_failures = list(adoption_evidence.get("failures") or [])
        adoption_blockers = [str(blocker) for blocker in adoption_evidence.get("blockers") or []]
        adoption_blocker_details = [
            str(detail) for detail in adoption_evidence.get("blocker_details") or []
        ]
        adoption_mcp_failures = [
            str(server) for server in adoption_evidence.get("mcp_server_failures") or []
        ]
        failure_text = (
            f" | failures: {', '.join(str(failure) for failure in adoption_failures)}"
            if adoption_failures
            else ""
        )
        lines.extend(
            [
                "",
                "## Adoption Evidence",
                "",
                (
                    f"- {adoption_evidence.get('client') or 'unknown client'}: "
                    f"{adoption_evidence.get('status', 'unknown')}{failure_text}"
                ),
                (
                    f"- Calls: {adoption_evidence.get('call_count', 0)} | "
                    f"captured {adoption_evidence.get('captured_at') or 'unknown'} | "
                    f"session {adoption_evidence.get('session_id') or 'unknown'}"
                ),
                (
                    f"- Artifact: {adoption_evidence.get('artifact_path') or 'unknown'} "
                    f"| sha256 {adoption_evidence.get('artifact_sha256') or 'unknown'}"
                ),
            ]
        )
        if adoption_blockers:
            lines.append(f"- Blockers: {', '.join(adoption_blockers)}")
        if adoption_mcp_failures:
            lines.append(f"- MCP server failures: {', '.join(adoption_mcp_failures)}")
        if adoption_blocker_details:
            lines.append(f"- Blocker details: {'; '.join(adoption_blocker_details)}")

    adoption_client_evidence = _mapping(report.get("adoption_client_evidence"))
    if adoption_client_evidence:
        client_failures = list(adoption_client_evidence.get("failures") or [])
        failure_text = (
            f" | failures: {', '.join(str(failure) for failure in client_failures)}"
            if client_failures
            else ""
        )
        required_clients = (
            ", ".join(
                str(client) for client in adoption_client_evidence.get("required_clients") or []
            )
            or "none"
        )
        observed_clients = (
            ", ".join(
                str(client) for client in adoption_client_evidence.get("observed_clients") or []
            )
            or "none"
        )
        client_blockers = [
            str(blocker) for blocker in adoption_client_evidence.get("blockers") or []
        ]
        client_mcp_failures = [
            str(server) for server in adoption_client_evidence.get("mcp_server_failures") or []
        ]
        lines.extend(
            [
                "",
                "## Adoption Client Evidence",
                "",
                (f"- Status: {adoption_client_evidence.get('status', 'unknown')}{failure_text}"),
                (f"- Required clients: {required_clients} | observed clients: {observed_clients}"),
                (f"- Reports: {adoption_client_evidence.get('report_count', 0)}"),
            ]
        )
        if client_blockers:
            lines.append(f"- Blockers: {', '.join(client_blockers)}")
        if client_mcp_failures:
            lines.append(f"- MCP server failures: {', '.join(client_mcp_failures)}")
        for evidence_report in adoption_client_evidence.get("reports") or []:
            if not isinstance(evidence_report, Mapping):
                continue
            report_blockers = [str(blocker) for blocker in evidence_report.get("blockers") or []]
            blocker_text = f" | blockers {', '.join(report_blockers)}" if report_blockers else ""
            lines.append(
                "- Report: "
                f"{evidence_report.get('client') or 'unknown client'} "
                f"{evidence_report.get('status') or 'unknown'} | "
                f"{evidence_report.get('artifact_path') or 'unknown artifact'} "
                f"| sha256 {evidence_report.get('artifact_sha256') or 'unknown'}"
                f"{blocker_text}"
            )

    if gaps:
        lines.extend(["", "## Coverage Gaps", ""])
        lines.extend(f"- {gap}" for gap in gaps)

    return "\n".join(lines).strip() + "\n"


def format_memory_value_markdown(report: Mapping[str, Any]) -> str:
    """Render just the memory value section for CLI value gates."""
    memory_value = _mapping(_get(report, "memory_value", "memoryValue"))
    cost = _mapping(memory_value.get("cost"))
    benefit = _mapping(memory_value.get("benefit"))
    lines = [
        "# Engram Memory Value",
        "",
        f"Group: `{report.get('group_id', 'default')}`",
        f"Generated: `{report.get('generated_at', '')}`",
        "",
        f"- Status: {memory_value.get('status', 'needs_samples')}",
        (
            f"- Cost: operations {cost.get('operation_count', 0)} | "
            f"avg added {_duration(cost.get('avg_added_latency_ms'))} | "
            f"p95 added {_duration(cost.get('p95_added_latency_ms'))} | "
            f"timeout {_pct(cost.get('timeout_rate'))} | "
            f"budget miss {_pct(cost.get('budget_miss_rate'))} | "
            f"cache hit {_pct(cost.get('cache_hit_rate'))}"
        ),
        (
            f"- Benefit: precision {_pct(benefit.get('memory_need_precision'))} | "
            f"need recall {_pct(benefit.get('memory_need_recall'))} | "
            f"useful packets {_pct(benefit.get('useful_packet_rate'))} | "
            f"continuity lift {_number(benefit.get('session_continuity_lift'))}"
        ),
        (
            f"- Trust: stale packets {_pct(benefit.get('stale_packet_rate'))} | "
            f"corrected packets {_pct(benefit.get('corrected_packet_rate'))}"
        ),
    ]
    failures = unmeasured_memory_value(report)
    if failures:
        lines.extend(["", "## Gate", "", f"- Failures: {', '.join(failures)}"])
    return "\n".join(lines).strip() + "\n"


def _release_evaluation_signal_component(report: Mapping[str, Any]) -> dict[str, Any]:
    evaluation_signals = _mapping(report.get("evaluation_signals"))
    if not evaluation_signals:
        return {
            "status": "missing",
            "missing": ["evaluation_signals"],
            "failures": ["evaluation_signals:missing"],
        }
    failures = unmeasured_evaluation_signals(report)
    return {
        "status": "measured" if not failures else "needs_signals",
        "missing": [],
        "failures": failures,
    }


def _release_evidence_component(
    evidence: Any,
    *,
    missing_key: str,
) -> dict[str, Any]:
    payload = _mapping(evidence)
    if not payload:
        return {
            "status": "missing",
            "missing": [missing_key],
            "failures": [],
        }
    status = str(payload.get("status") or "missing")
    failures = [str(failure) for failure in payload.get("failures") or []]
    if status != "measured":
        failures.append(f"{missing_key}:{status}")
    component = {
        "status": "measured" if status == "measured" and not failures else "failed",
        "missing": [],
        "failures": failures,
    }
    _copy_nonempty_list(component, payload, "blockers")
    _copy_nonempty_list(component, payload, "blocker_details")
    _copy_nonempty_list(component, payload, "mcp_server_failures")
    return component


def _release_adoption_client_component(evidence: Any) -> dict[str, Any]:
    payload = _mapping(evidence)
    if not payload:
        return {
            "status": "not_required",
            "missing": [],
            "failures": [],
            "required_clients": [],
            "observed_clients": [],
        }
    status = str(payload.get("status") or "missing")
    failures = [str(failure) for failure in payload.get("failures") or []]
    if status != "measured":
        failures.append(f"adoption_client_evidence:{status}")
    component = {
        "status": "measured" if status == "measured" and not failures else "failed",
        "missing": [],
        "failures": failures,
        "required_clients": [str(client) for client in payload.get("required_clients") or []],
        "observed_clients": [str(client) for client in payload.get("observed_clients") or []],
    }
    _copy_nonempty_list(component, payload, "blockers")
    _copy_nonempty_list(component, payload, "mcp_server_failures")
    return component


def _copy_nonempty_list(
    target: dict[str, Any],
    source: Mapping[str, Any],
    key: str,
) -> None:
    values = [str(value) for value in source.get(key) or []]
    if values:
        target[key] = values


def _capture_summary(stats: Mapping[str, Any], episode_count: int) -> dict[str, Any]:
    active_count = _int(stats.get("capture_active_count", stats.get("active_episodes")))
    return {
        "status": "empty" if episode_count == 0 else "active" if active_count else "ready",
        "episode_count": episode_count,
        "active_count": active_count,
    }


def _cue_summary(cue_metrics: Mapping[str, Any], episode_count: int) -> dict[str, Any]:
    cue_count = _int(cue_metrics.get("cue_count"))
    surfaced_count = _int(cue_metrics.get("cue_surfaced_count"))
    selected_count = _int(cue_metrics.get("cue_selected_count"))
    used_count = _int(cue_metrics.get("cue_used_count"))
    near_miss_count = _int(cue_metrics.get("cue_near_miss_count"))
    episodes_without_cues = _int(
        cue_metrics.get("episodes_without_cues", max(episode_count - cue_count, 0))
    )

    return {
        "status": "attention" if episodes_without_cues else "ready",
        "cue_count": cue_count,
        "episodes_without_cues": episodes_without_cues,
        "coverage": _float(cue_metrics.get("cue_coverage", _ratio(cue_count, episode_count))),
        "hit_count": _int(cue_metrics.get("cue_hit_count")),
        "hit_episode_count": _int(cue_metrics.get("cue_hit_episode_count")),
        "hit_episode_rate": _float(cue_metrics.get("cue_hit_episode_rate")),
        "surfaced_count": surfaced_count,
        "selected_count": selected_count,
        "used_count": used_count,
        "near_miss_count": near_miss_count,
        "selected_rate": _ratio(selected_count, surfaced_count),
        "used_rate": _ratio(used_count, surfaced_count),
        "near_miss_rate": _ratio(near_miss_count, surfaced_count),
        "avg_policy_score": _float(cue_metrics.get("avg_policy_score")),
        "projection_conversion_rate": _float(cue_metrics.get("cue_to_projection_conversion_rate")),
    }


def _project_summary(projection_metrics: Mapping[str, Any]) -> dict[str, Any]:
    state_counts = _state_counts(_mapping(projection_metrics.get("state_counts")))
    projected_count = state_counts["projected"]
    active_count = sum(state_counts[state] for state in PROJECT_ACTIVE_STATES)
    failure_count = state_counts["failed"] + state_counts["dead_letter"]
    tracked_count = _int(projection_metrics.get("tracked_episode_count")) or sum(
        state_counts.values()
    )
    yield_metrics = _mapping(projection_metrics.get("yield"))

    return {
        "status": "attention" if failure_count else "active" if active_count else "ready",
        "state_counts": state_counts,
        "tracked_count": tracked_count,
        "projected_count": projected_count,
        "active_count": active_count,
        "projected_rate": _ratio(projected_count, tracked_count),
        "backlog_rate": _ratio(active_count, tracked_count),
        "failed_count": state_counts["failed"],
        "dead_letter_count": state_counts["dead_letter"],
        "attempted_episode_count": _int(projection_metrics.get("attempted_episode_count")),
        "total_attempts": _int(projection_metrics.get("total_attempts")),
        "failure_rate": _float(projection_metrics.get("failure_rate")),
        "avg_processing_duration_ms": _float(projection_metrics.get("avg_processing_duration_ms")),
        "avg_time_to_projection_ms": _float(projection_metrics.get("avg_time_to_projection_ms")),
        "yield": {
            "linked_entity_count": _int(yield_metrics.get("linked_entity_count")),
            "relationship_count": _int(yield_metrics.get("relationship_count")),
            "avg_linked_entities_per_projected_episode": _float(
                yield_metrics.get("avg_linked_entities_per_projected_episode")
            ),
            "avg_relationships_per_projected_episode": _float(
                yield_metrics.get("avg_relationships_per_projected_episode")
            ),
        },
    }


def _recall_summary(
    recall_metrics: Mapping[str, Any],
    recall_samples: Sequence[RecallEvalSample | Mapping[str, Any]] | None,
    session_samples: Sequence[SessionContinuitySample | Mapping[str, Any]] | None,
) -> dict[str, Any]:
    recall_eval_samples = [_recall_sample(sample) for sample in recall_samples or []]
    continuity_samples = [_session_sample(sample) for sample in session_samples or []]
    surfaced = sum(max(0, sample.packets_surfaced) for sample in recall_eval_samples)
    used = sum(
        min(max(0, sample.packets_used), max(0, sample.packets_surfaced))
        for sample in recall_eval_samples
    )
    stale = sum(
        min(max(0, sample.stale_packets), max(0, sample.packets_surfaced))
        for sample in recall_eval_samples
    )
    corrected = sum(
        min(max(0, sample.corrected_packets), max(0, sample.packets_surfaced))
        for sample in recall_eval_samples
    )
    need_labeled = [sample for sample in recall_eval_samples if sample.recall_needed is not None]
    needed = [sample for sample in need_labeled if sample.recall_needed is True]
    missed = [sample for sample in needed if not sample.recall_triggered]

    evaluation = {
        "status": "measured" if recall_eval_samples else "needs_samples",
        "sample_count": len(recall_eval_samples),
        "need_status": "measured" if need_labeled else "needs_samples",
        "need_labeled_count": len(need_labeled),
        "needed_count": len(needed),
        "missed_count": len(missed),
        "memory_need_precision": (
            round(memory_need_precision(recall_eval_samples), 4) if recall_eval_samples else None
        ),
        "memory_need_recall": (
            round(memory_need_recall(recall_eval_samples), 4) if need_labeled else None
        ),
        "missed_recall_rate": (
            round(missed_recall_rate(recall_eval_samples), 4) if need_labeled else None
        ),
        "useful_packet_rate": (
            round(useful_packet_rate(recall_eval_samples), 4) if recall_eval_samples else None
        ),
        "false_recall_rate": (
            round(false_recall_rate(recall_eval_samples), 4) if recall_eval_samples else None
        ),
        "surfaced_count": surfaced,
        "used_count": used,
        "stale_packet_count": stale,
        "stale_packet_rate": _ratio_or_none(stale, surfaced),
        "corrected_packet_count": corrected,
        "corrected_packet_rate": _ratio_or_none(corrected, surfaced),
        "surfaced_to_used_ratio": _ratio_or_none(surfaced, used),
    }
    continuity = {
        "status": "measured" if continuity_samples else "needs_samples",
        "sample_count": len(continuity_samples),
        "session_continuity_lift": (
            round(session_continuity_lift(continuity_samples), 4) if continuity_samples else None
        ),
        "open_loop_recovery_rate": (
            round(open_loop_recovery_rate(continuity_samples), 4) if continuity_samples else None
        ),
        "temporal_correctness": (
            round(temporal_correctness(continuity_samples), 4) if continuity_samples else None
        ),
    }

    return {
        "status": "active" if _int(recall_metrics.get("trigger_count")) else "ready",
        "total_analyses": _int(recall_metrics.get("total_analyses")),
        "trigger_count": _int(recall_metrics.get("trigger_count")),
        "runtime_false_recall_rate": _float(recall_metrics.get("false_recall_rate")),
        "runtime_surfaced_to_used_ratio": recall_metrics.get("surfaced_to_used_ratio"),
        "graph_lift_rate": _float(recall_metrics.get("graph_lift_rate")),
        "probe_trigger_rate": _float(recall_metrics.get("probe_trigger_rate")),
        "latency": {
            "analyzer_ms": _latency_summary(
                _get(
                    recall_metrics,
                    "analyzer_latency_ms",
                    "analyzerLatencyMs",
                    default={},
                )
            ),
            "probe_ms": _latency_summary(
                _get(recall_metrics, "probe_latency_ms", "probeLatencyMs", default={})
            ),
        },
        "control": _recall_control_summary(recall_metrics),
        "family_contributions": dict(_mapping(recall_metrics.get("family_contributions"))),
        "evaluation": evaluation,
        "continuity": continuity,
    }


def _memory_value_summary(
    recall: Mapping[str, Any],
    operation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    cost = _memory_operation_cost_summary(operation_metrics)
    benefit = _memory_operation_benefit_summary(recall)
    cost_measured = cost.get("status") == "measured"
    benefit_measured = benefit.get("status") == "measured"
    if cost_measured and benefit_measured:
        status = "measured"
    elif cost_measured:
        status = "needs_benefit_labels"
    elif benefit_measured:
        status = "needs_cost_samples"
    else:
        status = "needs_samples"
    return {
        "status": status,
        "cost": cost,
        "benefit": benefit,
    }


def _memory_operation_cost_summary(
    metrics: Mapping[str, Any],
    *,
    include_modes: bool = True,
) -> dict[str, Any]:
    operation_count = _int(
        _get(
            metrics,
            "operation_count",
            "operationCount",
            "total_operations",
            "totalOperations",
            "count",
        )
    )
    duration = _latency_summary(
        _get(
            metrics,
            "added_latency_ms",
            "addedLatencyMs",
            "duration_ms",
            "durationMs",
            "latency_ms",
            "latencyMs",
            default={},
        )
    )
    budget = _latency_summary(_get(metrics, "budget_ms", "budgetMs", default={}))
    status_counts = dict(_mapping(_get(metrics, "status_counts", "statusCounts")))
    skip_reason_counts = dict(_mapping(_get(metrics, "skip_reason_counts", "skipReasonCounts")))
    operation_counts = dict(_mapping(_get(metrics, "operation_counts", "operationCounts")))
    source_counts = dict(_mapping(_get(metrics, "source_counts", "sourceCounts")))
    raw_recent_problem_samples = _get(
        metrics,
        "recent_problem_samples",
        "recentProblemSamples",
        default=[],
    )
    recent_problem_samples = (
        [dict(item) for item in raw_recent_problem_samples if isinstance(item, Mapping)]
        if isinstance(raw_recent_problem_samples, list)
        else []
    )
    completed_count = _int(
        _get(
            metrics,
            "completed_count",
            "completedCount",
            default=status_counts.get("ok", 0) + status_counts.get("completed", 0),
        )
    )
    skipped_count = _int(
        _get(
            metrics,
            "skipped_count",
            "skippedCount",
            default=status_counts.get("skipped", 0),
        )
    )
    error_count = _int(
        _get(metrics, "error_count", "errorCount", default=status_counts.get("error", 0))
    )
    timeout_count = _int(
        _get(metrics, "timeout_count", "timeoutCount", default=status_counts.get("timeout", 0))
    )
    degraded_count = _int(
        _get(
            metrics,
            "degraded_count",
            "degradedCount",
            default=status_counts.get("degraded", 0),
        )
    )
    budget_miss_count = _int(_get(metrics, "budget_miss_count", "budgetMissCount"))
    cache_hit_count = _int(_get(metrics, "cache_hit_count", "cacheHitCount"))
    cache_miss_count = _int(_get(metrics, "cache_miss_count", "cacheMissCount"))
    cache_total = cache_hit_count + cache_miss_count
    explicit_cache_hit_rate = _get(metrics, "cache_hit_rate", "cacheHitRate")
    explicit_timeout_rate = _get(metrics, "timeout_rate", "timeoutRate")
    explicit_degraded_rate = _get(metrics, "degraded_rate", "degradedRate")
    explicit_budget_miss_rate = _get(metrics, "budget_miss_rate", "budgetMissRate")

    summary = {
        "status": "measured"
        if operation_count > 0 or duration["p95_ms"] > 0 or duration["avg_ms"] > 0
        else "needs_samples",
        "operation_count": operation_count,
        "avg_added_latency_ms": duration["avg_ms"],
        "p95_added_latency_ms": duration["p95_ms"],
        "avg_budget_ms": budget["avg_ms"],
        "p95_budget_ms": budget["p95_ms"],
        "avg_budget_tokens": _int(_get(metrics, "avg_budget_tokens", "avgBudgetTokens")),
        "completed_count": completed_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "status_counts": status_counts,
        "skip_reason_counts": skip_reason_counts,
        "operation_counts": operation_counts,
        "source_counts": source_counts,
        "recent_problem_samples": recent_problem_samples,
        "timeout_count": timeout_count,
        "timeout_rate": _rate(explicit_timeout_rate, timeout_count, operation_count),
        "degraded_count": degraded_count,
        "degraded_rate": _rate(explicit_degraded_rate, degraded_count, operation_count),
        "budget_miss_count": budget_miss_count,
        "budget_miss_rate": _rate(
            explicit_budget_miss_rate,
            budget_miss_count,
            operation_count,
        ),
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": cache_miss_count,
        "cache_hit_rate": _rate(
            explicit_cache_hit_rate,
            cache_hit_count,
            cache_total or operation_count,
        ),
    }
    if include_modes:
        raw_modes = _mapping(
            _get(
                metrics,
                "modes",
                "by_mode",
                "byMode",
                "mode_metrics",
                "modeMetrics",
                default={},
            )
        )
        summary["by_mode"] = {
            str(mode): _memory_operation_cost_summary(
                _mapping(mode_metrics),
                include_modes=False,
            )
            for mode, mode_metrics in raw_modes.items()
        }
    return summary


def _memory_operation_benefit_summary(recall: Mapping[str, Any]) -> dict[str, Any]:
    recall_eval = _mapping(recall.get("evaluation"))
    continuity = _mapping(recall.get("continuity"))
    recall_measured = recall_eval.get("status") == "measured"
    continuity_measured = continuity.get("status") == "measured"
    return {
        "status": "measured" if recall_measured or continuity_measured else "needs_samples",
        "recall_sample_count": _int(recall_eval.get("sample_count")),
        "session_sample_count": _int(continuity.get("sample_count")),
        "memory_need_precision": recall_eval.get("memory_need_precision"),
        "memory_need_recall": recall_eval.get("memory_need_recall"),
        "missed_recall_rate": recall_eval.get("missed_recall_rate"),
        "useful_packet_rate": recall_eval.get("useful_packet_rate"),
        "stale_packet_rate": recall_eval.get("stale_packet_rate"),
        "corrected_packet_rate": recall_eval.get("corrected_packet_rate"),
        "stale_packet_count": _int(recall_eval.get("stale_packet_count")),
        "corrected_packet_count": _int(recall_eval.get("corrected_packet_count")),
        "false_recall_rate": recall_eval.get("false_recall_rate"),
        "session_continuity_lift": continuity.get("session_continuity_lift"),
        "open_loop_recovery_rate": continuity.get("open_loop_recovery_rate"),
        "temporal_correctness": continuity.get("temporal_correctness"),
    }


def _consolidation_summary(
    recent_cycles: Sequence[Any],
    calibration_snapshots: Sequence[Any],
    adjudication_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    cycle_summaries = [_cycle_summary(cycle) for cycle in recent_cycles]
    phase_status_counts: Counter[str] = Counter()
    phase_totals: dict[str, dict[str, int]] = {}
    calibration = _calibration_summary(calibration_snapshots)

    for cycle in cycle_summaries:
        phase_status_counts.update(cycle["phase_status_counts"])
        for phase, totals in cycle["phase_totals"].items():
            aggregate = phase_totals.setdefault(
                phase,
                {"runs": 0, "items_processed": 0, "items_affected": 0},
            )
            aggregate["runs"] += totals["runs"]
            aggregate["items_processed"] += totals["items_processed"]
            aggregate["items_affected"] += totals["items_affected"]

    latest = cycle_summaries[0] if cycle_summaries else None
    items_processed = sum(cycle["items_processed"] for cycle in cycle_summaries)
    items_affected = sum(cycle["items_affected"] for cycle in cycle_summaries)
    error_count = sum(cycle["error_count"] for cycle in cycle_summaries)
    normalized_phase_totals = _phase_totals_with_effect_rates(phase_totals)
    calibration_status = str(calibration.get("status") or "")
    adjudication = _adjudication_summary(
        normalized_phase_totals,
        cycle_summaries,
        adjudication_metrics,
    )
    return {
        "status": "attention"
        if error_count
        or _int(adjudication.get("open_work_count"))
        or (cycle_summaries and calibration_status != "measured")
        else "ready"
        if cycle_summaries
        else "needs_cycles",
        "cycle_count": len(cycle_summaries),
        "latest_status": latest["status"] if latest else None,
        "latest_cycle": latest,
        "phase_status_counts": dict(phase_status_counts),
        "phase_totals": normalized_phase_totals,
        "adjudication": adjudication,
        "calibration": calibration,
        "items_processed": items_processed,
        "items_affected": items_affected,
        "effect_rate": _ratio(items_affected, items_processed),
        "error_count": error_count,
    }


def _cycle_summary(cycle: Any) -> dict[str, Any]:
    phases = list(_get(cycle, "phase_results", "phases", default=[]) or [])
    phase_status_counts: Counter[str] = Counter()
    phase_totals: dict[str, dict[str, int]] = {}
    errors: list[dict[str, str]] = []
    cycle_error = _get(cycle, "error", default=None)
    phase_issue = _get(cycle, "phase_issue", "phaseIssue", default=None)
    if not (isinstance(phase_issue, str) and phase_issue.strip()):
        phase_issue = None

    for phase_result in phases:
        phase = str(_get(phase_result, "phase", default="unknown") or "unknown")
        status = str(_get(phase_result, "status", default="unknown") or "unknown")
        processed = _int(_get(phase_result, "items_processed", "itemsProcessed"))
        affected = _int(_get(phase_result, "items_affected", "itemsAffected"))
        error = _get(phase_result, "error", default=None)
        phase_status_counts[status] += 1
        totals = phase_totals.setdefault(
            phase,
            {"runs": 0, "items_processed": 0, "items_affected": 0},
        )
        totals["runs"] += 1
        totals["items_processed"] += processed
        totals["items_affected"] += affected
        if error:
            errors.append({"phase": phase, "error": str(error)})

    if phase_issue is None and errors:
        first_error = errors[0]
        phase_issue = f"{first_error['phase']}: {first_error['error']}"

    items_processed = sum(total["items_processed"] for total in phase_totals.values())
    items_affected = sum(total["items_affected"] for total in phase_totals.values())
    inferred_phase_issue_count = 1 if phase_issue and not errors else 0

    return {
        "id": _get(cycle, "id"),
        "status": _get(cycle, "status"),
        "error": cycle_error,
        "phase_issue": phase_issue,
        "trigger": _get(cycle, "trigger"),
        "dry_run": bool(_get(cycle, "dry_run", "dryRun", default=False)),
        "started_at": _get(cycle, "started_at", "startedAt"),
        "completed_at": _get(cycle, "completed_at", "completedAt"),
        "total_duration_ms": _float(_get(cycle, "total_duration_ms", "totalDurationMs")),
        "phase_count": len(phases),
        "phase_status_counts": dict(phase_status_counts),
        "phase_totals": _phase_totals_with_effect_rates(phase_totals),
        "items_processed": items_processed,
        "items_affected": items_affected,
        "effect_rate": _ratio(items_affected, items_processed),
        "error_count": len(errors) + (1 if cycle_error else 0) + inferred_phase_issue_count,
        "errors": errors,
    }


def _phase_totals_with_effect_rates(
    phase_totals: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        phase: {
            "runs": _int(totals.get("runs")),
            "items_processed": _int(totals.get("items_processed")),
            "items_affected": _int(totals.get("items_affected")),
            "effect_rate": _ratio(
                _int(totals.get("items_affected")),
                _int(totals.get("items_processed")),
            ),
        }
        for phase, totals in phase_totals.items()
    }


def _calibration_summary(snapshots: Sequence[Any]) -> dict[str, Any]:
    phase_totals: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        phase = str(_get(snapshot, "phase", default="unknown") or "unknown")
        totals = phase_totals.setdefault(
            phase,
            {
                "snapshots": 0,
                "total_traces": 0,
                "labeled_examples": 0,
                "oracle_examples": 0,
                "abstain_count": 0,
                "accuracy_values": [],
                "mean_confidence_values": [],
                "expected_calibration_error_values": [],
            },
        )
        totals["snapshots"] += 1
        totals["total_traces"] += _int(_get(snapshot, "total_traces", "totalTraces"))
        totals["labeled_examples"] += _int(_get(snapshot, "labeled_examples", "labeledExamples"))
        totals["oracle_examples"] += _int(_get(snapshot, "oracle_examples", "oracleExamples"))
        totals["abstain_count"] += _int(_get(snapshot, "abstain_count", "abstainCount"))
        _append_optional_float(totals["accuracy_values"], _get(snapshot, "accuracy"))
        _append_optional_float(
            totals["mean_confidence_values"],
            _get(snapshot, "mean_confidence", "meanConfidence"),
        )
        _append_optional_float(
            totals["expected_calibration_error_values"],
            _get(
                snapshot,
                "expected_calibration_error",
                "expectedCalibrationError",
            ),
        )

    normalized: dict[str, dict[str, Any]] = {}
    for phase, totals in phase_totals.items():
        normalized[phase] = {
            "snapshots": totals["snapshots"],
            "total_traces": totals["total_traces"],
            "labeled_examples": totals["labeled_examples"],
            "oracle_examples": totals["oracle_examples"],
            "abstain_count": totals["abstain_count"],
            "accuracy": _average_or_none(totals["accuracy_values"]),
            "mean_confidence": _average_or_none(totals["mean_confidence_values"]),
            "expected_calibration_error": _average_or_none(
                totals["expected_calibration_error_values"]
            ),
        }

    status = "needs_snapshots"
    if snapshots:
        status = (
            "measured"
            if _calibration_quality_measured({"phase_totals": normalized})
            else "needs_quality"
        )

    return {
        "status": status,
        "snapshot_count": len(snapshots),
        "phase_totals": normalized,
    }


def _coverage_gaps(
    *,
    episode_count: int,
    cue: Mapping[str, Any],
    project: Mapping[str, Any],
    recall: Mapping[str, Any],
    memory_value: Mapping[str, Any],
    consolidate: Mapping[str, Any],
) -> list[str]:
    gaps: list[str] = []
    recall_eval = _mapping(recall.get("evaluation"))
    continuity = _mapping(recall.get("continuity"))
    recall_latency = _mapping(recall.get("latency"))
    analyzer_latency = _mapping(recall_latency.get("analyzer_ms"))

    if episode_count == 0:
        gaps.append("capture has no stored episodes yet")
    if episode_count and _int(cue.get("cue_count")) == 0:
        gaps.append("cue usefulness cannot be measured until episodes have cues")
    elif _int(cue.get("cue_count")) > 0 and _int(cue.get("surfaced_count")) == 0:
        gaps.append("cue usefulness needs surfaced cue feedback")
    if _int(project.get("projected_count")) == 0:
        gaps.append("projection yield cannot be measured until episodes are projected")
    if recall_eval.get("status") != "measured":
        gaps.append("recall quality needs labeled recall_samples input")
    elif recall_eval.get("need_status") != "measured":
        gaps.append("missed recall needs recall_needed labels")
    elif _int(recall_eval.get("surfaced_count")) == 0:
        gaps.append("false recall needs labeled surfaced packet counts")
    if _int(recall.get("total_analyses")) == 0:
        gaps.append("recall gate needs runtime analyses")
    elif _float(analyzer_latency.get("p95_ms")) <= 0:
        gaps.append("recall gate latency needs analyzer samples")
    if continuity.get("status") != "measured":
        gaps.append("session continuity needs session_samples input")
    memory_cost = _mapping(memory_value.get("cost"))
    if memory_cost.get("status") != "measured":
        gaps.append("memory value needs memory operation cost samples")
    calibration = _mapping(consolidate.get("calibration"))
    calibration_status = calibration.get("status")
    if _int(consolidate.get("cycle_count")) == 0:
        gaps.append("consolidation effects need at least one recent cycle")
    elif calibration_status == "needs_snapshots" or calibration_status not in {
        "measured",
        "needs_quality",
    }:
        gaps.append("consolidation calibration needs saved calibration snapshots")
    elif calibration_status == "needs_quality" or not _calibration_quality_measured(calibration):
        gaps.append("consolidation calibration quality needs labeled decision outcomes")
    return gaps


def _evaluation_signals(
    *,
    cue: Mapping[str, Any],
    project: Mapping[str, Any],
    recall: Mapping[str, Any],
    consolidate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    recall_eval = _mapping(recall.get("evaluation"))
    calibration = _mapping(consolidate.get("calibration"))
    calibration_phases = _mapping(calibration.get("phase_totals"))
    triage_calibration = _mapping(calibration_phases.get("triage"))

    cue_count = _int(cue.get("cue_count"))
    cue_surfaced = _int(cue.get("surfaced_count"))
    projected_count = _int(project.get("projected_count"))
    recall_samples = _int(recall_eval.get("sample_count"))
    recall_surfaced = _int(recall_eval.get("surfaced_count"))
    triage_labels = _int(triage_calibration.get("labeled_examples"))
    cycle_count = _int(consolidate.get("cycle_count"))
    consolidation_processed = _int(consolidate.get("items_processed"))

    return {
        "cue_usefulness": _signal_readiness(
            status=(
                "measured" if cue_surfaced > 0 else "needs_feedback" if cue_count else "needs_data"
            ),
            evidence_count=cue_surfaced,
            metric=cue.get("used_rate"),
            gap=None
            if cue_surfaced > 0
            else "cue usefulness needs surfaced cue feedback"
            if cue_count
            else "cue usefulness cannot be measured until episodes have cues",
        ),
        "projection_yield": _signal_readiness(
            status="measured" if projected_count > 0 else "needs_data",
            evidence_count=projected_count,
            metric=_get(
                _mapping(project.get("yield")),
                "avg_linked_entities_per_projected_episode",
                "avgLinkedEntitiesPerProjectedEpisode",
            ),
            gap=None
            if projected_count > 0
            else "projection yield cannot be measured until episodes are projected",
        ),
        "recall_quality": _signal_readiness(
            status="measured" if recall_eval.get("status") == "measured" else "needs_labels",
            evidence_count=recall_samples,
            metric=recall_eval.get("memory_need_precision"),
            gap=None
            if recall_eval.get("status") == "measured"
            else "recall quality needs labeled recall_samples input",
        ),
        "false_recall": _signal_readiness(
            status="measured"
            if recall_eval.get("status") == "measured" and recall_surfaced > 0
            else "needs_labels"
            if recall_samples == 0
            else "needs_surfaced_packets",
            evidence_count=recall_surfaced,
            metric=recall_eval.get("false_recall_rate"),
            gap=None
            if recall_eval.get("status") == "measured" and recall_surfaced > 0
            else "recall quality needs labeled recall_samples input"
            if recall_samples == 0
            else "false recall needs labeled surfaced packet counts",
        ),
        "triage_calibration": _signal_readiness(
            status="measured"
            if _calibration_phase_quality_measured(triage_calibration)
            else "needs_snapshots"
            if _int(calibration.get("snapshot_count")) == 0
            else "needs_quality",
            evidence_count=triage_labels,
            metric=triage_calibration.get("expected_calibration_error"),
            gap=None
            if _calibration_phase_quality_measured(triage_calibration)
            else "consolidation calibration needs saved calibration snapshots"
            if _int(calibration.get("snapshot_count")) == 0
            else "consolidation calibration quality needs labeled decision outcomes",
        ),
        "consolidation_effect": _signal_readiness(
            status="measured"
            if cycle_count > 0 and consolidation_processed > 0
            else "needs_cycles"
            if cycle_count == 0
            else "needs_processed_items",
            evidence_count=cycle_count,
            metric=consolidate.get("effect_rate"),
            gap=None
            if cycle_count > 0 and consolidation_processed > 0
            else "consolidation effects need at least one recent cycle"
            if cycle_count == 0
            else "consolidation effects need processed cycle items",
        ),
    }


def _signal_readiness(
    *,
    status: str,
    evidence_count: int,
    metric: Any,
    gap: str | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "evidence_count": evidence_count,
        "metric": _optional_float(metric),
        "gap": gap,
    }


def _recall_runtime_score(metrics: Mapping[str, Any]) -> tuple[int, float, int]:
    analyzer_latency = _mapping(
        _get(metrics, "analyzer_latency_ms", "analyzerLatencyMs", default={})
    )
    control = _mapping(_get(metrics, "control", default={}))
    surfaced = _int(
        _get(
            metrics,
            "surfaced_count",
            "surfacedCount",
            default=_get(control, "surfaced_count", "surfacedCount", default=0),
        )
    )
    return (
        _int(_get(metrics, "total_analyses", "totalAnalyses")),
        _float(_get(analyzer_latency, "p95", "p95_ms", "p95Ms")),
        surfaced,
    )


def _memory_operation_score(metrics: Mapping[str, Any]) -> tuple[int, float, int]:
    duration = _mapping(
        _get(
            metrics,
            "added_latency_ms",
            "addedLatencyMs",
            "duration_ms",
            "durationMs",
            "latency_ms",
            "latencyMs",
            default={},
        )
    )
    signal_count = (
        _int(_get(metrics, "timeout_count", "timeoutCount"))
        + _int(_get(metrics, "degraded_count", "degradedCount"))
        + _int(_get(metrics, "cache_hit_count", "cacheHitCount"))
        + _int(_get(metrics, "cache_miss_count", "cacheMissCount"))
        + _int(_get(metrics, "budget_miss_count", "budgetMissCount"))
    )
    return (
        _int(
            _get(
                metrics,
                "operation_count",
                "operationCount",
                "total_operations",
                "totalOperations",
                "count",
            )
        ),
        _float(_get(duration, "p95", "p95_ms", "p95Ms")),
        signal_count,
    )


def _adjudication_summary(
    phase_totals: Mapping[str, Mapping[str, Any]],
    cycle_summaries: Sequence[Mapping[str, Any]],
    adjudication_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    adjudication_phase_totals = {
        phase: totals for phase, totals in phase_totals.items() if phase in ADJUDICATION_PHASES
    }
    runs = sum(_int(totals.get("runs")) for totals in adjudication_phase_totals.values())
    items_processed = sum(
        _int(totals.get("items_processed", totals.get("itemsProcessed")))
        for totals in adjudication_phase_totals.values()
    )
    items_affected = sum(
        _int(totals.get("items_affected", totals.get("itemsAffected")))
        for totals in adjudication_phase_totals.values()
    )
    error_count = sum(
        1
        for cycle in cycle_summaries
        for error in list(cycle.get("errors") or [])
        if _get(error, "phase") in ADJUDICATION_PHASES
    )
    evidence_status_counts = _status_count_summary(
        _mapping(
            _get(
                adjudication_metrics,
                "evidence_status_counts",
                "evidenceStatusCounts",
                default={},
            )
        ),
        ("pending", "deferred", "approved"),
    )
    request_status_counts = _status_count_summary(
        _mapping(
            _get(
                adjudication_metrics,
                "request_status_counts",
                "requestStatusCounts",
                default={},
            )
        ),
        ("pending", "deferred", "error"),
    )
    open_evidence_count = _int(
        _get(
            adjudication_metrics,
            "open_evidence_count",
            "openEvidenceCount",
            default=sum(evidence_status_counts.values()),
        )
    )
    open_request_count = _int(
        _get(
            adjudication_metrics,
            "open_request_count",
            "openRequestCount",
            default=sum(request_status_counts.values()),
        )
    )
    open_work_count = _int(
        _get(
            adjudication_metrics,
            "open_work_count",
            "openWorkCount",
            default=open_evidence_count + open_request_count,
        )
    )

    return {
        "status": "attention"
        if error_count or open_work_count
        else "active"
        if items_processed
        else "ready"
        if runs
        else "needs_cycles",
        "phase_count": len(adjudication_phase_totals),
        "runs": runs,
        "items_processed": items_processed,
        "items_affected": items_affected,
        "items_unaffected": max(items_processed - items_affected, 0),
        "effect_rate": _ratio(items_affected, items_processed),
        "error_count": error_count,
        "open_evidence_count": open_evidence_count,
        "open_request_count": open_request_count,
        "open_work_count": open_work_count,
        "pending_evidence_count": _int(
            _get(
                adjudication_metrics,
                "pending_evidence_count",
                "pendingEvidenceCount",
                default=evidence_status_counts["pending"],
            )
        ),
        "deferred_evidence_count": _int(
            _get(
                adjudication_metrics,
                "deferred_evidence_count",
                "deferredEvidenceCount",
                default=evidence_status_counts["deferred"],
            )
        ),
        "approved_evidence_count": _int(
            _get(
                adjudication_metrics,
                "approved_evidence_count",
                "approvedEvidenceCount",
                default=evidence_status_counts["approved"],
            )
        ),
        "pending_request_count": _int(
            _get(
                adjudication_metrics,
                "pending_request_count",
                "pendingRequestCount",
                default=request_status_counts["pending"],
            )
        ),
        "deferred_request_count": _int(
            _get(
                adjudication_metrics,
                "deferred_request_count",
                "deferredRequestCount",
                default=request_status_counts["deferred"],
            )
        ),
        "error_request_count": _int(
            _get(
                adjudication_metrics,
                "error_request_count",
                "errorRequestCount",
                default=request_status_counts["error"],
            )
        ),
        "evidence_status_counts": evidence_status_counts,
        "request_status_counts": request_status_counts,
        "phase_totals": adjudication_phase_totals,
    }


def _status_count_summary(
    counts: Mapping[str, Any],
    statuses: tuple[str, ...],
) -> dict[str, int]:
    return {status: _int(counts.get(status)) for status in statuses}


def _calibration_quality_measured(calibration: Mapping[str, Any]) -> bool:
    phase_totals = _mapping(_get(calibration, "phase_totals", "phaseTotals"))
    for totals in phase_totals.values():
        if _calibration_phase_quality_measured(_mapping(totals)):
            return True
    return False


def _calibration_phase_quality_measured(totals: Mapping[str, Any]) -> bool:
    has_labels = _int(_get(totals, "labeled_examples", "labeledExamples")) > 0
    has_quality = (
        _get(totals, "accuracy") is not None
        or _get(totals, "expected_calibration_error", "expectedCalibrationError") is not None
    )
    return has_labels and has_quality


def _calibration_markdown_detail(calibration: Mapping[str, Any]) -> str:
    if _get(calibration, "status") == "needs_quality":
        return "; needs labeled decisions"

    phase_totals = _mapping(_get(calibration, "phase_totals", "phaseTotals"))
    phases: list[tuple[str, Mapping[str, Any]]] = [
        (str(phase), _mapping(totals)) for phase, totals in phase_totals.items()
    ]
    if not phases:
        return ""

    phase, totals = max(
        phases,
        key=lambda item: _int(_get(item[1], "labeled_examples", "labeledExamples")),
    )
    accuracy = _get(totals, "accuracy")
    ece = _get(totals, "expected_calibration_error", "expectedCalibrationError")
    if accuracy is None and ece is None:
        return ""
    return f"; {phase} accuracy {_pct(accuracy)}, ECE {_number(ece)}"


def _stats_payload(stats: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = stats.get("stats")
    if isinstance(nested, Mapping):
        return nested
    return stats


def _state_counts(raw: Mapping[str, Any]) -> dict[str, int]:
    return {
        "queued": _int(raw.get("queued")),
        "cued": _int(raw.get("cued")),
        "cue_only": _int(raw.get("cue_only", raw.get("cueOnly"))),
        "scheduled": _int(raw.get("scheduled")),
        "projecting": _int(raw.get("projecting")),
        "projected": _int(raw.get("projected")),
        "merged": _int(raw.get("merged")),
        "failed": _int(raw.get("failed")),
        "dead_letter": _int(raw.get("dead_letter", raw.get("deadLetter"))),
    }


def _latency_summary(value: Any) -> dict[str, float]:
    data = _mapping(value)
    return {
        "avg_ms": _float(_get(data, "avg_ms", "avgMs", "avg")),
        "p95_ms": _float(_get(data, "p95_ms", "p95Ms", "p95")),
    }


def _recall_control_summary(recall_metrics: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = _mapping(_get(recall_metrics, "thresholds", default={}))
    return {
        "used_count": _int(_get(recall_metrics, "used_count", "usedCount")),
        "dismissed_count": _int(_get(recall_metrics, "dismissed_count", "dismissedCount")),
        "surfaced_count": _int(_get(recall_metrics, "surfaced_count", "surfacedCount")),
        "selected_count": _int(_get(recall_metrics, "selected_count", "selectedCount")),
        "confirmed_count": _int(_get(recall_metrics, "confirmed_count", "confirmedCount")),
        "corrected_count": _int(_get(recall_metrics, "corrected_count", "correctedCount")),
        "graph_override_count": _int(
            _get(recall_metrics, "graph_override_count", "graphOverrideCount")
        ),
        "adaptive_thresholds_enabled": bool(
            _get(
                recall_metrics,
                "adaptive_thresholds_enabled",
                "adaptiveThresholdsEnabled",
                default=False,
            )
        ),
        "thresholds": {
            "linguistic": _float(_get(thresholds, "linguistic")),
            "borderline": _float(_get(thresholds, "borderline")),
            "resonance": _float(_get(thresholds, "resonance")),
        },
    }


def _recall_sample(sample: RecallEvalSample | Mapping[str, Any]) -> RecallEvalSample:
    if isinstance(sample, RecallEvalSample):
        return sample
    recall_needed = _get(sample, "recall_needed", "recallNeeded", default=None)
    return RecallEvalSample(
        recall_triggered=bool(_get(sample, "recall_triggered", "recallTriggered")),
        recall_helped=bool(_get(sample, "recall_helped", "recallHelped")),
        packets_surfaced=_int(_get(sample, "packets_surfaced", "packetsSurfaced")),
        packets_used=_int(_get(sample, "packets_used", "packetsUsed")),
        false_recalls=_int(_get(sample, "false_recalls", "falseRecalls")),
        stale_packets=_int(_get(sample, "stale_packets", "stalePackets")),
        corrected_packets=_int(_get(sample, "corrected_packets", "correctedPackets")),
        recall_needed=None if recall_needed is None else bool(recall_needed),
    )


def _session_sample(
    sample: SessionContinuitySample | Mapping[str, Any],
) -> SessionContinuitySample:
    if isinstance(sample, SessionContinuitySample):
        return sample
    return SessionContinuitySample(
        baseline_score=_float(_get(sample, "baseline_score", "baselineScore")),
        memory_score=_float(_get(sample, "memory_score", "memoryScore")),
        open_loop_expected=bool(_get(sample, "open_loop_expected", "openLoopExpected")),
        open_loop_recovered=bool(_get(sample, "open_loop_recovered", "openLoopRecovered")),
        temporal_expected=bool(_get(sample, "temporal_expected", "temporalExpected")),
        temporal_correct=bool(_get(sample, "temporal_correct", "temporalCorrect")),
    )


def _get(source: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for key in keys:
            if key in source:
                return source[key]
        return default
    for key in keys:
        if hasattr(source, key):
            return getattr(source, key)
    return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return round(float(value or 0.0), 4)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _rate(value: Any, numerator: int, denominator: int) -> float | None:
    explicit = _optional_float(value)
    if explicit is not None:
        return explicit
    return _ratio_or_none(numerator, denominator)


def _append_optional_float(values: list[float], value: Any) -> None:
    if value is None:
        return
    try:
        values.append(float(value))
    except (TypeError, ValueError):
        return


def _average_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _generated_at(value: datetime | str | None) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        value = datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _duration(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        milliseconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if milliseconds < 0:
        return "n/a"
    if milliseconds < 1000:
        return f"{milliseconds:.0f}ms"
    if milliseconds < 60_000:
        return f"{milliseconds / 1000:.2f}s"
    return f"{milliseconds / 60_000:.1f}m"


def report_to_dict(report: Any) -> dict[str, Any]:
    """Compatibility helper for dataclass-based callers."""
    return asdict(report) if hasattr(report, "__dataclass_fields__") else dict(report)
