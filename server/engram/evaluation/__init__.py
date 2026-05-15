"""Local evaluation helpers for Engram runtime health."""

from engram.evaluation.brain_loop_report import (
    build_brain_loop_report,
    format_brain_loop_report_markdown,
    has_recall_runtime_metrics,
    merge_recall_runtime_metrics,
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
    "format_brain_loop_report_markdown",
    "format_smoke_report",
    "has_recall_runtime_metrics",
    "merge_recall_runtime_metrics",
    "present_recall_sample",
    "present_recall_sample_write",
    "present_session_sample",
    "present_session_sample_write",
    "run_projected_consolidated_smoke",
    "run_projected_consolidated_smoke_for_args",
]
