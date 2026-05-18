"""Human-labeled harness evidence for production brain-loop gates."""

from __future__ import annotations

import json
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


def load_human_label_evidence(
    artifact_path: Path,
    *,
    min_recall_samples: int = 1,
    min_session_samples: int = 1,
) -> dict[str, Any]:
    """Load and summarize a human-labeled harness artifact."""
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("human label artifact must be a JSON object")
    return build_human_label_evidence(
        payload,
        artifact_path=artifact_path,
        min_recall_samples=min_recall_samples,
        min_session_samples=min_session_samples,
    )


def build_human_label_evidence(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
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
    elif _looks_synthetic(source):
        failures.append(f"synthetic_human_label_source({source})")
    synthetic_sample_sources = [
        sample_source for sample_source in sample_sources if _looks_synthetic(sample_source)
    ]
    if synthetic_sample_sources:
        failures.append(
            "synthetic_sample_sources(" + ",".join(synthetic_sample_sources) + ")"
        )
    if not client:
        failures.append("missing_harness_client")
    if not captured_at:
        failures.append("missing_harness_captured_at")
    if not labeler:
        failures.append("missing_human_labeler")
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
        "kind": payload.get("kind"),
        "source": source,
        "client": client,
        "captured_at": captured_at,
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


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
