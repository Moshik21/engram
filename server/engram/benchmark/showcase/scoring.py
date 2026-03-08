"""Scoring helpers for showcase, answer, and summary reporting."""

from __future__ import annotations

from collections import defaultdict

from engram.benchmark.showcase.adapters import latency_percentiles
from engram.benchmark.showcase.models import (
    AnswerResult,
    AnswerSummary,
    BaselineSummary,
    EvidenceItem,
    ProbeResult,
    ScenarioProbe,
    ScenarioResult,
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def score_probe(
    probe: ScenarioProbe,
    evidence: list[EvidenceItem],
    latency_ms: float,
) -> ProbeResult:
    """Score a probe from surfaced evidence only."""
    normalized_evidence = [_normalize(item.text) for item in evidence]
    eligible_required = [
        _normalize(item.text)
        for item in evidence
        if not probe.required_evidence_result_types
        or item.result_type in probe.required_evidence_result_types
    ]

    required_hits: list[str] = []
    missing_required: list[str] = []
    for snippet in probe.required_evidence:
        normalized = _normalize(snippet)
        if any(normalized in item for item in eligible_required):
            required_hits.append(snippet)
        else:
            missing_required.append(snippet)

    forbidden_hits: list[str] = []
    for snippet in probe.forbidden_evidence:
        normalized = _normalize(snippet)
        if any(normalized in item for item in normalized_evidence):
            forbidden_hits.append(snippet)

    returned_types = sorted({item.result_type for item in evidence})
    type_candidates = probe.allowed_result_types or probe.expected_result_types
    expected_type_match = (
        True
        if not type_candidates
        else any(result_type in type_candidates for result_type in returned_types)
    )

    disallowed_type_hits = sorted(
        {
            item.result_type
            for item in evidence
            if item.result_type in set(probe.disallowed_result_types)
        }
    )
    historical_violation = (
        not probe.historical_evidence_allowed
        and any(item.result_type in {"episode", "cue_episode"} for item in evidence)
    )

    tokens_surfaced = sum(item.tokens for item in evidence)
    required_hit_rate = (
        len(required_hits) / len(probe.required_evidence)
        if probe.required_evidence
        else 1.0
    )
    forbidden_hit_rate = (
        len(forbidden_hits) / len(probe.forbidden_evidence)
        if probe.forbidden_evidence
        else 0.0
    )
    token_efficiency = (
        0.0
        if probe.max_tokens <= 0
        else max(0.0, min(1.0, 1.0 - (tokens_surfaced / probe.max_tokens)))
    )

    passed = (
        not missing_required
        and not forbidden_hits
        and expected_type_match
        and not disallowed_type_hits
        and not historical_violation
    )

    return ProbeResult(
        probe_id=probe.id,
        passed=passed,
        required_hits=required_hits,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
        expected_type_match=expected_type_match,
        returned_types=returned_types,
        latency_ms=latency_ms,
        tokens_surfaced=tokens_surfaced,
        required_hit_rate=required_hit_rate,
        forbidden_hit_rate=forbidden_hit_rate,
        token_efficiency=token_efficiency,
        disallowed_type_hits=disallowed_type_hits,
        historical_violation=historical_violation,
        evidence=evidence,
    )


def summarize_baseline(
    baseline_name: str,
    baseline_family: str,
    is_ablation: bool,
    scenario_results: list[ScenarioResult],
) -> BaselineSummary:
    """Aggregate showcase-track metrics for one baseline."""
    available_results = [result for result in scenario_results if result.available]
    available = bool(available_results)
    availability_reason = None
    if not available:
        for result in scenario_results:
            if result.availability_reason:
                availability_reason = result.availability_reason
                break

    scenario_pass_rate = (
        sum(1 for result in available_results if result.passed) / len(available_results)
        if available_results
        else 0.0
    )

    capability_totals: dict[str, int] = defaultdict(int)
    capability_passes: dict[str, int] = defaultdict(int)
    latencies: list[float] = []
    passed_tokens = 0
    passed_scenarios = 0
    false_recall_probes = 0
    total_probes = 0
    observed_turns = 0
    projected_turns = 0
    extraction_calls = 0
    embedding_calls = 0
    consolidation_cycles = 0
    required_hit_total = 0.0
    forbidden_hit_total = 0.0
    token_efficiency_total = 0.0

    for result in available_results:
        observed_turns += result.cost_stats.observed_turns
        projected_turns += result.cost_stats.projected_turns
        extraction_calls += result.cost_stats.extraction_calls
        embedding_calls += result.cost_stats.embedding_calls
        consolidation_cycles += result.cost_stats.consolidation_cycles
        if result.passed:
            passed_scenarios += 1
            passed_tokens += sum(probe.tokens_surfaced for probe in result.probe_results)
        for tag in result.capability_tags:
            capability_totals[tag] += 1
            if result.passed:
                capability_passes[tag] += 1
        for probe in result.probe_results:
            total_probes += 1
            latencies.append(probe.latency_ms)
            required_hit_total += probe.required_hit_rate
            forbidden_hit_total += probe.forbidden_hit_rate
            token_efficiency_total += probe.token_efficiency
            if probe.forbidden_hits:
                false_recall_probes += 1

    capability_pass_rates = {
        tag: (capability_passes[tag] / capability_totals[tag])
        for tag in sorted(capability_totals)
        if capability_totals[tag] > 0
    }
    latency_p50, latency_p95 = latency_percentiles(latencies)

    def _tag_rate(tag: str) -> float:
        total = capability_totals.get(tag, 0)
        if total == 0:
            return 0.0
        return capability_passes.get(tag, 0) / total

    tokens_per_passed_scenario = (
        passed_tokens / passed_scenarios if passed_scenarios > 0 else 0.0
    )
    selective_ratio = (
        projected_turns / observed_turns if observed_turns > 0 else 0.0
    )

    return BaselineSummary(
        baseline_name=baseline_name,
        baseline_family=baseline_family,
        is_ablation=is_ablation,
        available=available,
        availability_reason=availability_reason,
        scenario_pass_rate=scenario_pass_rate,
        capability_pass_rates=capability_pass_rates,
        false_recall_rate=(
            false_recall_probes / total_probes if total_probes > 0 else 0.0
        ),
        temporal_correctness=_tag_rate("temporal"),
        negation_correctness=_tag_rate("negation"),
        open_loop_recovery=_tag_rate("open_loop"),
        prospective_trigger_rate=_tag_rate("prospective"),
        required_hit_rate=(
            required_hit_total / total_probes if total_probes > 0 else 0.0
        ),
        forbidden_hit_rate=(
            forbidden_hit_total / total_probes if total_probes > 0 else 0.0
        ),
        token_efficiency=(
            token_efficiency_total / total_probes if total_probes > 0 else 0.0
        ),
        tokens_per_passed_scenario=tokens_per_passed_scenario,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        cost_proxies={
            "observed_turns": float(observed_turns),
            "projected_turns": float(projected_turns),
            "extraction_calls": float(extraction_calls),
            "embedding_calls": float(embedding_calls),
            "consolidation_cycles": float(consolidation_cycles),
            "selective_extraction_ratio": selective_ratio,
        },
    )


def summarize_answer_results(
    baseline_name: str,
    baseline_family: str,
    results: list[AnswerResult],
) -> AnswerSummary:
    """Aggregate answer-track metrics for a baseline."""
    available_results = [result for result in results if result.available]
    available = bool(available_results)
    availability_reason = None
    if not available:
        for result in results:
            if result.availability_reason:
                availability_reason = result.availability_reason
                break

    latencies = [result.latency_ms for result in available_results]
    latency_p50, latency_p95 = latency_percentiles(latencies)
    answer_pass_rate = (
        sum(1 for result in available_results if result.passed) / len(available_results)
        if available_results
        else 0.0
    )
    average_score = (
        sum(result.score for result in available_results) / len(available_results)
        if available_results
        else 0.0
    )
    tokens_per_passed_answer = (
        sum(result.tokens_surfaced for result in available_results if result.passed)
        / max(1, sum(1 for result in available_results if result.passed))
        if available_results
        else 0.0
    )

    return AnswerSummary(
        baseline_name=baseline_name,
        baseline_family=baseline_family,
        available=available,
        availability_reason=availability_reason,
        answer_pass_rate=answer_pass_rate,
        average_score=average_score,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        tokens_per_passed_answer=tokens_per_passed_answer,
    )
