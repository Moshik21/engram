"""Local evaluation helpers for Engram runtime health."""

from engram.evaluation.brain_loop_report import (
    build_brain_loop_report,
    evaluation_signal_failure_message,
    format_brain_loop_report_markdown,
    has_recall_runtime_metrics,
    is_brain_loop_report_payload,
    looks_like_partial_brain_loop_report,
    merge_recall_runtime_metrics,
    missing_brain_loop_report_sections,
    unmeasured_evaluation_signals,
)
from engram.evaluation.presenter import (
    present_recall_sample,
    present_recall_sample_write,
    present_session_sample,
    present_session_sample_write,
)
from engram.evaluation.smoke import (
    format_smoke_report,
    run_projected_consolidated_smoke,
    run_projected_consolidated_smoke_for_args,
)
from engram.evaluation.store import (
    SQLiteEvaluationStore,
    StoredRecallEvalSample,
    StoredRecallRuntimeMetricsSnapshot,
    StoredSessionContinuitySample,
)

__all__ = [
    "SQLiteEvaluationStore",
    "StoredRecallEvalSample",
    "StoredRecallRuntimeMetricsSnapshot",
    "StoredSessionContinuitySample",
    "build_brain_loop_report",
    "evaluation_signal_failure_message",
    "format_brain_loop_report_markdown",
    "format_smoke_report",
    "has_recall_runtime_metrics",
    "is_brain_loop_report_payload",
    "looks_like_partial_brain_loop_report",
    "merge_recall_runtime_metrics",
    "missing_brain_loop_report_sections",
    "present_recall_sample",
    "present_recall_sample_write",
    "present_session_sample",
    "present_session_sample_write",
    "run_projected_consolidated_smoke",
    "run_projected_consolidated_smoke_for_args",
    "unmeasured_evaluation_signals",
]
