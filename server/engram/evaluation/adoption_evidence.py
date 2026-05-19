"""Adoption-verifier evidence for production brain-loop gates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

ADOPTION_VALIDATION_REPORT_KIND = "engram_adoption_validation_report"
_SYNTHETIC_SOURCE_TOKENS = {
    "benchmark",
    "deterministic",
    "fixture",
    "showcase",
    "simulated",
    "smoke",
    "synthetic",
}


def load_adoption_evidence(
    artifact_path: Path,
    *,
    required_client: str | None = None,
) -> dict[str, Any]:
    """Load and summarize an `engram adoption --format json` report."""
    raw = artifact_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("adoption report must be a JSON object")
    return build_adoption_evidence(
        payload,
        artifact_path=artifact_path,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        required_client=required_client,
    )


def build_adoption_evidence(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
    artifact_sha256: str | None = None,
    required_client: str | None = None,
) -> dict[str, Any]:
    """Build a release-gate summary from an adoption validation report."""
    evidence = _mapping(payload.get("evidence"))
    validation = _mapping(payload.get("validation"))
    required_tools = _mapping(validation.get("required_tools_before_answer"))
    capture = _mapping(validation.get("capture"))
    file_memory = _mapping(validation.get("file_memory"))

    kind = _string(payload.get("kind"))
    client = _string(evidence.get("client"))
    report_required_client = _string(evidence.get("required_client"))
    gate_required_client = _string(required_client)
    captured_at = _string(evidence.get("captured_at") or evidence.get("capturedAt"))
    session_id = _string(evidence.get("session_id") or evidence.get("sessionId"))
    source = _string(evidence.get("source"))
    blockers = _string_list(evidence.get("blockers"))
    blocker_details = _string_list(
        evidence.get("blocker_details") or evidence.get("blockerDetails")
    )
    mcp_server_failures = _string_list(
        evidence.get("mcp_server_failures") or evidence.get("mcpServerFailures")
    )
    call_count = _int(payload.get("callCount"))
    expected_before_answer = _string_list(required_tools.get("expected"))
    observed_before_answer = _string_list(required_tools.get("observed"))
    observed_capture_tools = _string_list(capture.get("observed_tools"))
    failures: list[str] = []
    if kind != ADOPTION_VALIDATION_REPORT_KIND:
        failures.append(
            "invalid_adoption_report_kind"
            f"({kind or 'missing'}!={ADOPTION_VALIDATION_REPORT_KIND})"
        )
    if payload.get("status") != "passed":
        failures.append("adoption_status_not_passed")
    if call_count <= 0:
        failures.append("missing_adoption_calls")
    if not client:
        failures.append("missing_adoption_client")
    elif _looks_placeholder(client):
        failures.append("placeholder_adoption_client")
    if not captured_at:
        failures.append("missing_adoption_captured_at")
    elif not _is_iso_timestamp(captured_at):
        failures.append(f"invalid_adoption_captured_at({captured_at})")
    if not source:
        failures.append("missing_adoption_source")
    elif _looks_placeholder(source):
        failures.append("placeholder_adoption_source")
    elif _looks_synthetic(source):
        failures.append(f"synthetic_adoption_source({source})")
    if evidence.get("required") is not True:
        failures.append("missing_required_live_evidence_gate")
    if not expected_before_answer:
        failures.append("missing_adoption_required_tool_expectations")
    if not observed_before_answer:
        failures.append("missing_adoption_observed_before_answer_tools")
    else:
        missing_core_tools = [
            tool
            for tool in ("get_context", "recall")
            if tool not in observed_before_answer
        ]
        if missing_core_tools:
            failures.append(
                "missing_adoption_core_before_answer_tools("
                + ",".join(missing_core_tools)
                + ")"
            )
    if required_tools.get("missing"):
        failures.append("missing_adoption_required_tools")
    if required_tools.get("in_order") is False:
        failures.append("adoption_required_tools_out_of_order")
    if capture.get("missing") is True:
        failures.append("missing_adoption_capture")
    if not observed_capture_tools:
        failures.append("missing_adoption_observed_capture_tools")
    if file_memory.get("substituted_for_engram") is True:
        failures.append("file_memory_substituted_for_engram")
    validation_failures = [str(failure) for failure in validation.get("failures") or []]
    if validation_failures:
        failures.append("adoption_validation_failures(" + ",".join(validation_failures) + ")")
    if gate_required_client:
        if not _client_matches(client, gate_required_client):
            failures.append(
                "adoption_client_mismatch"
                f"({_client_label(client)}!={gate_required_client})"
            )
        if not report_required_client:
            failures.append("missing_required_adoption_client_gate")
        elif not _client_matches(report_required_client, gate_required_client):
            failures.append(
                "required_adoption_client_mismatch"
                f"({report_required_client}!={gate_required_client})"
            )

    return {
        "status": "failed" if failures else "measured",
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": artifact_sha256,
        "kind": kind,
        "adoption_status": payload.get("status"),
        "authority_path": payload.get("authorityPath"),
        "calls_path": payload.get("callsPath"),
        "call_count": call_count,
        "client": client,
        "required_client": report_required_client,
        "gate_required_client": gate_required_client,
        "captured_at": captured_at,
        "session_id": session_id,
        "session_filter": evidence.get("session_filter"),
        "source": source,
        "required_live_evidence": bool(evidence.get("required")),
        "blockers": blockers,
        "blocker_details": blocker_details,
        "mcp_server_failures": mcp_server_failures,
        "required_tools": {
            "expected": expected_before_answer,
            "observed": observed_before_answer,
            "missing": list(required_tools.get("missing") or []),
            "in_order": bool(required_tools.get("in_order")),
        },
        "capture": {
            "destination": capture.get("destination"),
            "expected_tool": capture.get("expected_tool"),
            "observed_tools": observed_capture_tools,
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


def build_adoption_client_set_evidence(
    evidences: list[Mapping[str, Any]],
    *,
    required_clients: list[str],
) -> dict[str, Any]:
    """Summarize adoption evidence across multiple live MCP clients."""
    required_labels = _dedupe_labels(required_clients)
    observed_clients = _dedupe_labels(
        [
            str(evidence.get("client"))
            for evidence in evidences
            if evidence.get("client")
        ]
    )
    blockers = _dedupe_labels(
        [
            str(blocker)
            for evidence in evidences
            for blocker in evidence.get("blockers") or []
        ]
    )
    mcp_server_failures = _dedupe_labels(
        [
            str(server)
            for evidence in evidences
            for server in evidence.get("mcp_server_failures") or []
        ]
    )
    failures: list[str] = []
    if not evidences:
        failures.append("missing_adoption_evidence")
    for evidence in evidences:
        if evidence.get("status") != "measured":
            client_or_path = _string(evidence.get("client")) or _string(
                evidence.get("artifact_path")
            )
            failures.append(
                "adoption_report_failed"
                f"({_client_label(client_or_path)})"
            )
    for required_client in required_labels:
        matching = [
            evidence
            for evidence in evidences
            if _client_matches(_string(evidence.get("client")), required_client)
        ]
        if not matching:
            failures.append(f"missing_required_adoption_client({required_client})")
            continue
        if not any(
            _client_matches(_string(evidence.get("required_client")), required_client)
            for evidence in matching
        ):
            failures.append(
                f"missing_required_adoption_client_gate({required_client})"
            )

    return {
        "status": "failed" if failures else "measured",
        "required_clients": required_labels,
        "observed_clients": observed_clients,
        "report_count": len(evidences),
        "reports": [
            {
                "client": evidence.get("client"),
                "required_client": evidence.get("required_client"),
                "status": evidence.get("status"),
                "artifact_path": evidence.get("artifact_path"),
                "artifact_sha256": evidence.get("artifact_sha256"),
                "captured_at": evidence.get("captured_at"),
                "session_id": evidence.get("session_id"),
                "blockers": list(evidence.get("blockers") or []),
                "blocker_details": list(evidence.get("blocker_details") or []),
                "mcp_server_failures": list(
                    evidence.get("mcp_server_failures") or []
                ),
                "failures": list(evidence.get("failures") or []),
            }
            for evidence in evidences
        ],
        "blockers": blockers,
        "mcp_server_failures": mcp_server_failures,
        "failures": failures,
    }


def adoption_client_set_failure_message(
    evidence: Mapping[str, Any] | None,
    *,
    prefix: str,
) -> str | None:
    """Return a human-readable failure if multi-client adoption evidence failed."""
    if not evidence:
        return f"{prefix}: ['missing_adoption_client_set_evidence']"
    failures = list(evidence.get("failures") or [])
    if failures:
        return f"{prefix}: {failures}"
    if evidence.get("status") != "measured":
        return f"{prefix}: ['adoption_client_set:{evidence.get('status', 'missing')}']"
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


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item)]
    return []


def _is_iso_timestamp(value: str) -> bool:
    if "T" not in value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _looks_placeholder(value: str) -> bool:
    stripped = value.strip()
    return (
        stripped.startswith("<")
        and stripped.endswith(">")
        or stripped.lower() in {"todo", "tbd", "replace me", "placeholder"}
    )


def _looks_synthetic(source: str) -> bool:
    return any(
        token.lower() in _SYNTHETIC_SOURCE_TOKENS
        for token in source.replace("-", " ").replace("_", " ").split()
    )


def _client_matches(observed_client: str | None, required_client: str) -> bool:
    return _normalize_client_label(observed_client) == _normalize_client_label(
        required_client
    )


def _normalize_client_label(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def _client_label(value: str | None) -> str:
    return value if value else "missing"


def _dedupe_labels(values: list[str]) -> list[str]:
    labels: list[str] = []
    normalized_seen: set[str] = set()
    for value in values:
        label = _string(value)
        if not label:
            continue
        normalized = _normalize_client_label(label)
        if normalized in normalized_seen:
            continue
        normalized_seen.add(normalized)
        labels.append(label)
    return labels


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
