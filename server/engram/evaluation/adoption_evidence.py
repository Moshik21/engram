"""Adoption-verifier evidence for production brain-loop gates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_adoption_evidence(artifact_path: Path) -> dict[str, Any]:
    """Load and summarize an `engram adoption --format json` report."""
    raw = artifact_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("adoption report must be a JSON object")
    return build_adoption_evidence(
        payload,
        artifact_path=artifact_path,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_adoption_evidence(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a release-gate summary from an adoption validation report."""
    evidence = _mapping(payload.get("evidence"))
    validation = _mapping(payload.get("validation"))
    required_tools = _mapping(validation.get("required_tools_before_answer"))
    capture = _mapping(validation.get("capture"))
    file_memory = _mapping(validation.get("file_memory"))

    client = _string(evidence.get("client"))
    captured_at = _string(evidence.get("captured_at") or evidence.get("capturedAt"))
    session_id = _string(evidence.get("session_id") or evidence.get("sessionId"))
    failures: list[str] = []
    if payload.get("status") != "passed":
        failures.append("adoption_status_not_passed")
    if not client:
        failures.append("missing_adoption_client")
    if not captured_at:
        failures.append("missing_adoption_captured_at")
    if evidence.get("required") is not True:
        failures.append("missing_required_live_evidence_gate")
    if required_tools.get("missing"):
        failures.append("missing_adoption_required_tools")
    if required_tools.get("in_order") is False:
        failures.append("adoption_required_tools_out_of_order")
    if capture.get("missing") is True:
        failures.append("missing_adoption_capture")
    if file_memory.get("substituted_for_engram") is True:
        failures.append("file_memory_substituted_for_engram")
    validation_failures = [str(failure) for failure in validation.get("failures") or []]
    if validation_failures:
        failures.append("adoption_validation_failures(" + ",".join(validation_failures) + ")")

    return {
        "status": "failed" if failures else "measured",
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": artifact_sha256,
        "adoption_status": payload.get("status"),
        "authority_path": payload.get("authorityPath"),
        "calls_path": payload.get("callsPath"),
        "call_count": _int(payload.get("callCount")),
        "client": client,
        "required_client": evidence.get("required_client"),
        "captured_at": captured_at,
        "session_id": session_id,
        "session_filter": evidence.get("session_filter"),
        "source": evidence.get("source"),
        "required_live_evidence": bool(evidence.get("required")),
        "required_tools": {
            "expected": list(required_tools.get("expected") or []),
            "observed": list(required_tools.get("observed") or []),
            "missing": list(required_tools.get("missing") or []),
            "in_order": bool(required_tools.get("in_order")),
        },
        "capture": {
            "destination": capture.get("destination"),
            "expected_tool": capture.get("expected_tool"),
            "observed_tools": list(capture.get("observed_tools") or []),
            "missing": bool(capture.get("missing")),
        },
        "file_memory": {
            "present": file_memory.get("present"),
            "substituted_for_engram": bool(
                file_memory.get("substituted_for_engram")
            ),
        },
        "failures": failures,
    }


def adoption_evidence_failure_message(
    evidence: Mapping[str, Any] | None,
    *,
    prefix: str,
) -> str | None:
    """Return a human-readable failure if adoption evidence did not pass gates."""
    if not evidence:
        return f"{prefix}: ['missing_adoption_evidence']"
    failures = list(evidence.get("failures") or [])
    if failures:
        return f"{prefix}: {failures}"
    if evidence.get("status") != "measured":
        return f"{prefix}: ['adoption_evidence:{evidence.get('status', 'missing')}']"
    return None


def link_adoption_to_human_label_evidence(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Fail adoption evidence when attached human labels point at a different run."""
    human = report.get("human_label_evidence")
    adoption = report.get("adoption_evidence")
    if not isinstance(human, Mapping) or not isinstance(adoption, Mapping):
        return report

    failures = list(adoption.get("failures") or [])
    _append_mismatch(
        failures,
        "human_label_client_mismatch",
        left=_string(human.get("client")),
        right=_string(adoption.get("client")),
    )
    _append_mismatch(
        failures,
        "human_label_captured_at_mismatch",
        left=_string(human.get("captured_at")),
        right=_string(adoption.get("captured_at")),
    )
    adoption_session = _string(adoption.get("session_id"))
    human_session = _string(human.get("session_id"))
    if adoption_session and not human_session:
        failures.append("missing_human_label_session_id")
    else:
        _append_mismatch(
            failures,
            "human_label_session_id_mismatch",
            left=human_session,
            right=adoption_session,
        )

    updated_adoption = dict(adoption)
    updated_adoption["failures"] = failures
    updated_adoption["status"] = "failed" if failures else adoption.get("status")
    updated_report = dict(report)
    updated_report["adoption_evidence"] = updated_adoption
    return updated_report


def _append_mismatch(
    failures: list[str],
    label: str,
    *,
    left: str | None,
    right: str | None,
) -> None:
    if left and right and left != right:
        failures.append(f"{label}({left}!={right})")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
