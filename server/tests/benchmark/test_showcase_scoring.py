"""Scoring and summary tests for the showcase benchmark."""

from engram.benchmark.showcase.models import (
    AdapterCostStats,
    AnswerResult,
    EvidenceItem,
    ScenarioProbe,
    ScenarioResult,
)
from engram.benchmark.showcase.scoring import (
    score_probe,
    summarize_answer_results,
    summarize_baseline,
)


def test_score_probe_detects_required_and_forbidden_evidence():
    probe = ScenarioProbe(
        id="probe",
        after_turn_index=0,
        operation="recall",
        query="current base url",
        required_evidence=["api.v2.internal"],
        forbidden_evidence=["api.v1.internal"],
        expected_result_types=["context"],
    )
    evidence = [
        EvidenceItem(result_type="context", text="base_url: api.v2.internal"),
    ]

    result = score_probe(probe, evidence, latency_ms=12.0)

    assert result.passed is True
    assert result.required_hits == ["api.v2.internal"]
    assert result.forbidden_hits == []
    assert result.expected_type_match is True


def test_score_probe_fails_on_forbidden_match():
    probe = ScenarioProbe(
        id="probe",
        after_turn_index=0,
        operation="recall",
        query="frontend",
        required_evidence=["USES Svelte"],
        forbidden_evidence=["USES React"],
        expected_result_types=["entity"],
    )
    evidence = [
        EvidenceItem(result_type="entity", text="Falcon Dashboard USES React"),
        EvidenceItem(result_type="entity", text="Falcon Dashboard USES Svelte"),
    ]

    result = score_probe(probe, evidence, latency_ms=10.0)

    assert result.passed is False
    assert result.forbidden_hits == ["USES React"]


def test_summarize_baseline_rolls_up_metrics():
    scenario_results = [
        ScenarioResult(
            scenario_id="temporal_override",
            scenario_title="Temporal Override",
            why_it_matters="Latest fact should win.",
            baseline_name="engram_full",
            baseline_family="engram",
            seed=7,
            capability_tags=["temporal"],
            available=True,
            passed=True,
            probe_results=[
                score_probe(
                    ScenarioProbe(
                        id="probe1",
                        after_turn_index=0,
                        operation="get_context",
                        expected_result_types=["context"],
                    ),
                    [EvidenceItem(result_type="context", text="base_url: api.v2.internal")],
                    latency_ms=9.0,
                )
            ],
            cost_stats=AdapterCostStats(observed_turns=2, projected_turns=2),
        ),
        ScenarioResult(
            scenario_id="open_loop_recovery",
            scenario_title="Open Loop Recovery",
            why_it_matters="Unfinished work should return.",
            baseline_name="engram_full",
            baseline_family="engram",
            seed=7,
            capability_tags=["open_loop"],
            available=True,
            passed=False,
            probe_results=[
                score_probe(
                    ScenarioProbe(
                        id="probe2",
                        after_turn_index=0,
                        operation="recall",
                        forbidden_evidence=["wrong detail"],
                    ),
                    [EvidenceItem(result_type="episode", text="wrong detail")],
                    latency_ms=15.0,
                )
            ],
            cost_stats=AdapterCostStats(observed_turns=1, projected_turns=0),
        ),
    ]

    summary = summarize_baseline(
        "engram_full",
        "engram",
        False,
        scenario_results,
    )

    assert summary.available is True
    assert summary.scenario_pass_rate == 0.5
    assert summary.temporal_correctness == 1.0
    assert summary.open_loop_recovery == 0.0
    assert summary.false_recall_rate == 0.5
    assert summary.cost_proxies["selective_extraction_ratio"] == 2 / 3


def test_score_probe_flags_disallowed_types_and_historical_violations():
    probe = ScenarioProbe(
        id="probe",
        after_turn_index=0,
        operation="recall",
        disallowed_result_types=["episode"],
        historical_evidence_allowed=False,
        expected_result_types=["entity"],
    )
    evidence = [EvidenceItem(result_type="episode", text="old raw history")]

    result = score_probe(probe, evidence, latency_ms=8.0)

    assert result.passed is False
    assert result.disallowed_type_hits == ["episode"]
    assert result.historical_violation is True


def test_summarize_answer_results_rolls_up_scores():
    summary = summarize_answer_results(
        "engram_full",
        "engram",
        [
            AnswerResult(
                scenario_id="temporal_override",
                scenario_title="Temporal Override",
                baseline_name="engram_full",
                baseline_family="engram",
                seed=7,
                available=True,
                passed=True,
                answer_task_question="Question",
                score=1.0,
                latency_ms=12.0,
                tokens_surfaced=20,
            ),
            AnswerResult(
                scenario_id="open_loop_recovery",
                scenario_title="Open Loop Recovery",
                baseline_name="engram_full",
                baseline_family="engram",
                seed=7,
                available=True,
                passed=False,
                answer_task_question="Question",
                score=0.5,
                latency_ms=18.0,
                tokens_surfaced=30,
            ),
        ],
    )

    assert summary.available is True
    assert summary.answer_pass_rate == 0.5
    assert summary.average_score == 0.75
    assert summary.tokens_per_passed_answer == 20.0
