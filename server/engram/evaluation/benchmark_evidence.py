"""Benchmark artifact evidence for production brain-loop evaluation gates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_benchmark_evidence(
    artifact_path: Path,
    *,
    baseline: str = "engram_full",
    min_scenarios: int = 1,
    min_pass_rate: float = 0.0,
) -> dict[str, Any]:
    """Load and summarize a benchmark artifact for release/evaluation gating."""
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark artifact must be a JSON object")
    return build_benchmark_evidence(
        payload,
        artifact_path=artifact_path,
        baseline=baseline,
        min_scenarios=min_scenarios,
        min_pass_rate=min_pass_rate,
    )


def build_benchmark_evidence(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
    baseline: str = "engram_full",
    min_scenarios: int = 1,
    min_pass_rate: float = 0.0,
) -> dict[str, Any]:
    """Build a JSON-serializable benchmark evidence summary."""
    min_scenarios = max(1, int(min_scenarios))
    min_pass_rate = max(0.0, min(1.0, float(min_pass_rate)))
    summaries = _list(payload.get("baseline_summaries"))
    summary = _find_baseline_summary(summaries, baseline)
    scenario_results = [
        item
        for item in _list(payload.get("scenario_results"))
        if _mapping(item).get("baseline_name") == baseline
    ]
    available_scenarios = [
        item for item in scenario_results if bool(_mapping(item).get("available", True))
    ]
    scenario_count = len(available_scenarios)
    passed_count = sum(1 for item in available_scenarios if bool(_mapping(item).get("passed")))
    observed_pass_rate = (
        _float(_mapping(summary).get("scenario_pass_rate"))
        if summary is not None
        else (passed_count / scenario_count if scenario_count else None)
    )
    fairness = _mapping(payload.get("fairness_contract"))
    transcript_hashes = _mapping(fairness.get("transcript_hashes"))
    baseline_contracts = _mapping(fairness.get("baseline_contracts"))

    failures: list[str] = []
    if summary is None:
        failures.append("missing_baseline_summary")
    elif not bool(_mapping(summary).get("available", True)):
        failures.append("baseline_unavailable")
    if scenario_count < min_scenarios:
        failures.append(f"insufficient_benchmark_scenarios({scenario_count}<{min_scenarios})")
    if observed_pass_rate is None:
        failures.append("missing_benchmark_pass_rate")
    elif observed_pass_rate < min_pass_rate:
        failures.append(
            f"benchmark_pass_rate_below_threshold({observed_pass_rate:.3f}<{min_pass_rate:.3f})"
        )
    if not fairness:
        failures.append("missing_fairness_contract")
    if not transcript_hashes:
        failures.append("missing_transcript_hashes")
    if baseline not in baseline_contracts:
        failures.append("missing_baseline_fairness_contract")

    return {
        "status": "failed" if failures else "measured",
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "benchmark": "showcase" if "baseline_summaries" in payload else "unknown",
        "baseline": baseline,
        "track": payload.get("track"),
        "mode": payload.get("mode"),
        "generated_at": payload.get("generated_at"),
        "seeds": list(_list(payload.get("seeds"))),
        "scenario_count": scenario_count,
        "passed_count": passed_count,
        "scenario_pass_rate": observed_pass_rate,
        "false_recall_rate": _float(_mapping(summary).get("false_recall_rate")),
        "min_scenarios": min_scenarios,
        "min_pass_rate": min_pass_rate,
        "fairness": {
            "strict": bool(fairness.get("strict_fairness")),
            "transcript_hash_count": len(transcript_hashes),
            "baseline_contract_present": baseline in baseline_contracts,
        },
        "failures": failures,
    }


def benchmark_evidence_failure_message(
    evidence: Mapping[str, Any] | None,
    *,
    prefix: str,
) -> str | None:
    """Return a human-readable failure if benchmark evidence did not pass gates."""
    if not evidence:
        return f"{prefix}: ['missing_benchmark_evidence']"
    failures = list(evidence.get("failures") or [])
    if failures:
        return f"{prefix}: {failures}"
    if evidence.get("status") != "measured":
        return f"{prefix}: ['benchmark_evidence:{evidence.get('status', 'missing')}']"
    return None


def _find_baseline_summary(
    summaries: list[Any],
    baseline: str,
) -> Mapping[str, Any] | None:
    for summary in summaries:
        item = _mapping(summary)
        if item.get("baseline_name") == baseline:
            return item
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
