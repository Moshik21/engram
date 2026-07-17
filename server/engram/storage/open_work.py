"""Shared status sets and metrics for live consolidation work queues."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

OPEN_EVIDENCE_STATUSES = ("pending", "deferred", "approved")
OPEN_ADJUDICATION_STATUSES = ("pending", "deferred", "error")


def build_adjudication_metrics(
    evidence_status_counts: Mapping[str, Any],
    request_status_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize open evidence/request status counts for stats/reporting."""
    evidence_counts = _status_counts(evidence_status_counts, OPEN_EVIDENCE_STATUSES)
    request_counts = _status_counts(request_status_counts, OPEN_ADJUDICATION_STATUSES)
    open_evidence_count = sum(evidence_counts.values())
    open_request_count = sum(request_counts.values())
    return {
        "evidence_status_counts": evidence_counts,
        "request_status_counts": request_counts,
        "open_evidence_count": open_evidence_count,
        "pending_evidence_count": evidence_counts["pending"],
        "deferred_evidence_count": evidence_counts["deferred"],
        "approved_evidence_count": evidence_counts["approved"],
        "open_request_count": open_request_count,
        "pending_request_count": request_counts["pending"],
        "deferred_request_count": request_counts["deferred"],
        "error_request_count": request_counts["error"],
        "open_work_count": open_evidence_count + open_request_count,
    }


def _status_counts(raw: Mapping[str, Any], statuses: tuple[str, ...]) -> dict[str, int]:
    return {status: _int(raw.get(status)) for status in statuses}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        # silent-ok: count coercion; a non-numeric status value normalizes to 0.
        return 0
