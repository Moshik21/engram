"""Human-labeled harness evidence for production brain-loop gates."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SYNTHETIC_SOURCE_TOKENS = {
    "benchmark",
    "deterministic",
    "fixture",
    "showcase",
    "simulated",
    "smoke",
    "synthetic",
}


def build_human_label_evidence_template(
    *,
    adoption_evidence: Mapping[str, Any] | None = None,
    adoption_report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return the operator template for real human-reviewed harness evidence."""
    adoption_report = _adoption_report_template_metadata(
        adoption_evidence,
        adoption_report_path=adoption_report_path,
    )
    client = (
        adoption_report.get("client")
        or "<Claude Code | Cursor | Windsurf | other MCP client>"
    )
    captured_at = (
        adoption_report.get("capturedAt")
        or "<ISO-8601 timestamp from live harness run>"
    )
    session_id = (
        adoption_report.get("sessionId")
        or "<client session/thread id from adoption report if available>"
    )
    adoption_report_arg = _shell_arg(
        adoption_report_path
        or adoption_report.get("path")
        or "adoption-report.json"
    )
    instructions = [
        "Collect this from a real staging or production harness run.",
        "Keep source/client/capturedAt/labeler as observed metadata, not placeholders.",
        (
            "Do not use smoke, showcase, benchmark, fixture, deterministic, "
            "simulated, or synthetic data for this release gate."
        ),
        "Add enough recallSamples and sessionSamples to satisfy the chosen gate thresholds.",
    ]
    if adoption_report:
        instructions.insert(
            1,
            (
                "Keep client/capturedAt/sessionId aligned with the attached "
                "adoptionReport metadata; the release gate cross-checks them."
            ),
        )
    template = {
        "kind": "engram_human_label_evidence",
        "humanLabeled": True,
        "source": "<staging_harness_or_production_harness_name>",
        "client": client,
        "capturedAt": captured_at,
        "sessionId": session_id,
        "labeler": "<human reviewer name or handle>",
        "recallSamples": [
            {
                "source": "<same real harness source>",
                "query": "<user turn or recall probe>",
                "recallTriggered": True,
                "recallHelped": True,
                "recallNeeded": True,
                "packetsSurfaced": 2,
                "packetsUsed": 1,
                "falseRecalls": 0,
                "notes": "<why the recalled packet was useful or misleading>",
            },
            {
                "source": "<same real harness source>",
                "query": "<memory-needed turn where recall did not help>",
                "recallTriggered": False,
                "recallHelped": False,
                "recallNeeded": True,
                "packetsSurfaced": 0,
                "packetsUsed": 0,
                "falseRecalls": 0,
                "notes": "<what was missing>",
            },
        ],
        "sessionSamples": [
            {
                "source": "<same real harness source>",
                "scenario": "<multi-turn task or open loop being evaluated>",
                "baselineScore": 0.2,
                "memoryScore": 0.8,
                "openLoopExpected": True,
                "openLoopRecovered": True,
                "temporalExpected": True,
                "temporalCorrect": True,
                "notes": "<what Engram preserved across turns>",
            }
        ],
        "validationCommand": (
            "engram evaluate --from-json brain-loop-report.json "
            "--require-release-evidence "
            "--human-label-artifact human-labels.json "
            f"--adoption-report {adoption_report_arg} "
            "--min-human-recall-samples 10 "
            "--min-human-session-samples 3 "
            "--evidence-bundle brain-loop-release-evidence.json "
            "--format json"
        ),
        "instructions": instructions,
    }
    if adoption_report:
        template["adoptionReport"] = adoption_report
    return template


def render_human_label_evidence_template_markdown(template: Mapping[str, Any]) -> str:
    """Render the human-label evidence template for operators."""
    lines = [
        "# Engram Human Label Evidence Template",
        "",
        "Use this for real staging or production harness review evidence.",
        "",
        "## Required Metadata",
        "",
        f"- Source: `{template.get('source')}`",
        f"- Client: `{template.get('client')}`",
        f"- Captured at: `{template.get('capturedAt')}`",
        f"- Labeler: `{template.get('labeler')}`",
        f"- Human labeled: `{template.get('humanLabeled')}`",
        "",
        "## Sample Counts In Template",
        "",
        f"- Recall samples: {len(_extract_samples(template, 'recallSamples'))}",
        f"- Session samples: {len(_extract_samples(template, 'sessionSamples'))}",
        "",
        "## Validation",
        "",
        f"```bash\n{template.get('validationCommand')}\n```",
        "",
        "## JSON",
        "",
        "```json",
        json.dumps(dict(template), indent=2, sort_keys=True),
        "```",
    ]
    instructions = _list_payload(template.get("instructions"))
    if instructions:
        lines.extend(["", "## Instructions", ""])
        lines.extend(f"- {instruction}" for instruction in instructions)
    adoption_report = _mapping(template.get("adoptionReport"))
    if adoption_report:
        failures = _list_payload(adoption_report.get("failures"))
        failure_text = ", ".join(str(failure) for failure in failures) or "none"
        lines.extend(
            [
                "",
                "## Adoption Report Prefill",
                "",
                f"- Path: `{adoption_report.get('path')}`",
                f"- Status: `{adoption_report.get('status')}`",
                f"- Client: `{adoption_report.get('client')}`",
                f"- Captured at: `{adoption_report.get('capturedAt')}`",
                f"- Session: `{adoption_report.get('sessionId')}`",
                f"- Failures: `{failure_text}`",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def load_human_label_evidence(
    artifact_path: Path,
    *,
    min_recall_samples: int = 1,
    min_session_samples: int = 1,
) -> dict[str, Any]:
    """Load and summarize a human-labeled harness artifact."""
    raw = artifact_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("human label artifact must be a JSON object")
    return build_human_label_evidence(
        payload,
        artifact_path=artifact_path,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        min_recall_samples=min_recall_samples,
        min_session_samples=min_session_samples,
    )


def build_human_label_evidence(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
    artifact_sha256: str | None = None,
    min_recall_samples: int = 1,
    min_session_samples: int = 1,
) -> dict[str, Any]:
    """Build a JSON-serializable summary for real labeled harness data."""
    min_recall_samples = max(0, int(min_recall_samples))
    min_session_samples = max(0, int(min_session_samples))
    recall_samples = _extract_samples(payload, "recall_samples", "recallSamples")
    session_samples = _extract_samples(payload, "session_samples", "sessionSamples")
    source = _string(
        _first(payload, "source", "label_source", "labelSource", "harness_source")
    )
    client = _string(_first(payload, "client", "harness", "client_label", "clientLabel"))
    captured_at = _string(_first(payload, "captured_at", "capturedAt"))
    session_id = _string(_first(payload, "session_id", "sessionId"))
    labeler = _string(_first(payload, "labeler", "reviewed_by", "reviewedBy"))
    human_labeled = _truthy(_first(payload, "human_labeled", "humanLabeled"))
    sample_sources = sorted(
        {
            sample_source
            for sample in recall_samples + session_samples
            if (sample_source := _string(_mapping(sample).get("source")))
        }
    )

    failures: list[str] = []
    if not human_labeled:
        failures.append("missing_human_labeled_flag")
    if not source:
        failures.append("missing_human_label_source")
    elif _looks_placeholder(source):
        failures.append("placeholder_human_label_source")
    elif _looks_synthetic(source):
        failures.append(f"synthetic_human_label_source({source})")
    synthetic_sample_sources = [
        sample_source for sample_source in sample_sources if _looks_synthetic(sample_source)
    ]
    placeholder_sample_sources = [
        sample_source for sample_source in sample_sources if _looks_placeholder(sample_source)
    ]
    if placeholder_sample_sources:
        failures.append(
            "placeholder_sample_sources(" + ",".join(placeholder_sample_sources) + ")"
        )
    if synthetic_sample_sources:
        failures.append(
            "synthetic_sample_sources(" + ",".join(synthetic_sample_sources) + ")"
        )
    if not client:
        failures.append("missing_harness_client")
    elif _looks_placeholder(client):
        failures.append("placeholder_harness_client")
    if not captured_at:
        failures.append("missing_harness_captured_at")
    elif _looks_placeholder(captured_at):
        failures.append("placeholder_harness_captured_at")
    if not labeler:
        failures.append("missing_human_labeler")
    elif _looks_placeholder(labeler):
        failures.append("placeholder_human_labeler")
    if len(recall_samples) < min_recall_samples:
        failures.append(
            f"insufficient_human_recall_samples({len(recall_samples)}<{min_recall_samples})"
        )
    if len(session_samples) < min_session_samples:
        failures.append(
            f"insufficient_human_session_samples({len(session_samples)}<{min_session_samples})"
        )

    return {
        "status": "failed" if failures else "measured",
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": artifact_sha256,
        "kind": payload.get("kind"),
        "source": source,
        "client": client,
        "captured_at": captured_at,
        "session_id": session_id,
        "labeler": labeler,
        "human_labeled": human_labeled,
        "recall_sample_count": len(recall_samples),
        "session_sample_count": len(session_samples),
        "min_recall_samples": min_recall_samples,
        "min_session_samples": min_session_samples,
        "sample_sources": sample_sources,
        "failures": failures,
    }


def human_label_evidence_failure_message(
    evidence: Mapping[str, Any] | None,
    *,
    prefix: str,
) -> str | None:
    """Return a human-readable failure if human label evidence did not pass gates."""
    if not evidence:
        return f"{prefix}: ['missing_human_label_evidence']"
    failures = list(evidence.get("failures") or [])
    if failures:
        return f"{prefix}: {failures}"
    if evidence.get("status") != "measured":
        return f"{prefix}: ['human_label_evidence:{evidence.get('status', 'missing')}']"
    return None


def _extract_samples(payload: Mapping[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return _list_payload(value)
    samples = payload.get("samples")
    if isinstance(samples, Mapping):
        for key in keys:
            value = samples.get(key)
            if value is not None:
                return _list_payload(value)
    return []


def _list_payload(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        samples = value.get("samples")
        if isinstance(samples, list):
            return samples
    return []


def _adoption_report_template_metadata(
    adoption_evidence: Mapping[str, Any] | None,
    *,
    adoption_report_path: Path | str | None,
) -> dict[str, Any]:
    evidence = _mapping(adoption_evidence)
    if not evidence:
        return {}
    path = adoption_report_path or evidence.get("artifact_path")
    return {
        "path": str(path) if path is not None else None,
        "sha256": evidence.get("artifact_sha256"),
        "status": evidence.get("status"),
        "client": evidence.get("client"),
        "capturedAt": evidence.get("captured_at"),
        "sessionId": evidence.get("session_id"),
        "source": evidence.get("source"),
        "failures": list(evidence.get("failures") or []),
    }


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _shell_arg(value: Path | str) -> str:
    return shlex.quote(str(value))


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _looks_synthetic(source: str) -> bool:
    tokens = {
        token.strip(" _-/.:").lower()
        for token in source.replace("-", " ").replace("_", " ").split()
    }
    return bool(tokens & _SYNTHETIC_SOURCE_TOKENS)


def _looks_placeholder(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")
