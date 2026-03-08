"""Report rendering tests for the showcase benchmark."""

from engram.benchmark.showcase.models import (
    AnswerSummary,
    BaselineSummary,
    ExternalTrackResult,
    FairnessContract,
    ScenarioResult,
    ShowcaseRunResult,
    TrackSummary,
)
from engram.benchmark.showcase.report import render_markdown_report


def _baseline_summary(name: str, *, pass_rate: float, available: bool = True) -> BaselineSummary:
    return BaselineSummary(
        baseline_name=name,
        baseline_family="engram" if name.startswith("engram") else "alternative",
        is_ablation=name.startswith("engram_") and name != "engram_full",
        available=available,
        availability_reason=None,
        scenario_pass_rate=pass_rate,
        capability_pass_rates={"temporal": pass_rate},
        false_recall_rate=0.0 if pass_rate > 0 else 1.0,
        temporal_correctness=pass_rate,
        negation_correctness=pass_rate,
        open_loop_recovery=pass_rate,
        prospective_trigger_rate=pass_rate,
        required_hit_rate=pass_rate,
        forbidden_hit_rate=0.0,
        token_efficiency=0.5,
        tokens_per_passed_scenario=40.0,
        latency_p50_ms=5.0,
        latency_p95_ms=8.0,
        cost_proxies={"selective_extraction_ratio": 1.0},
    )


def test_report_contains_fairness_executive_and_appendix_sections():
    run_result = ShowcaseRunResult(
        track="all",
        mode="quick",
        seeds=[7],
        generated_at="2026-03-08T00:00:00+00:00",
        output_dir="/tmp/showcase",
        fairness_contract=FairnessContract(
            track="all",
            strict_fairness=True,
            scenario_budgets={"temporal_override": {"retrieval_limit": 5, "evidence_max_tokens": 80, "answer_budget_tokens": 70}},
            vector_provider_family="local",
            answer_model="deterministic",
            answer_provider="deterministic",
            answer_prompt="prompt",
            transcript_invariant=True,
            baseline_contracts={"engram_full": {"answer_prompt_id": "showcase_answer_v2"}},
            transcript_hashes={"temporal_override": "abc"},
        ),
        primary_baselines=["engram_full", "context_summary"],
        appendix_baselines=["context_window"],
        ablation_baselines=["engram_no_cues"],
        scenario_results=[
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
            ),
            ScenarioResult(
                scenario_id="temporal_override",
                scenario_title="Temporal Override",
                why_it_matters="Latest fact should win.",
                baseline_name="context_summary",
                baseline_family="alternative",
                seed=7,
                capability_tags=["temporal"],
                available=True,
                passed=False,
            ),
        ],
        baseline_summaries=[
            _baseline_summary("engram_full", pass_rate=1.0),
            _baseline_summary("context_summary", pass_rate=0.0),
            _baseline_summary("context_window", pass_rate=0.0),
            _baseline_summary("engram_no_cues", pass_rate=0.5),
        ],
        answer_summaries=[
            AnswerSummary(
                baseline_name="engram_full",
                baseline_family="engram",
                available=True,
                availability_reason=None,
                answer_pass_rate=1.0,
                average_score=1.0,
                latency_p50_ms=4.0,
                latency_p95_ms=4.0,
                tokens_per_passed_answer=12.0,
            )
        ],
        external_track_results=[
            ExternalTrackResult(
                name="retrieval_ab",
                available=True,
                executed=False,
                summary_metrics={"purpose": "Controlled retrieval A/B comparison"},
                recommended_command="uv run python scripts/benchmark_ab.py",
            )
        ],
        track_summaries=[TrackSummary(track="showcase", executed=True, available=True)],
        supporting_artifacts={"benchmark_ab": "server/scripts/benchmark_ab.py"},
        artifact_paths={},
        readme_snippet="Benchmark results measured against equal retrieval budgets.",
    )

    report = render_markdown_report(run_result)

    assert "# Engram Benchmark Suite" in report
    assert "## Fairness Contract" in report
    assert "## Executive Table" in report
    assert "## Appendix Baselines" in report
    assert "## External And Supporting Tracks" in report
    assert "## README Snippet" in report
    assert "Context + Summary" in report
    assert "server/scripts/benchmark_ab.py" in report


def test_report_marks_no_winner_when_all_primary_baselines_fail():
    run_result = ShowcaseRunResult(
        track="showcase",
        mode="full",
        seeds=[7],
        generated_at="2026-03-08T00:00:00+00:00",
        output_dir="/tmp/showcase",
        fairness_contract=FairnessContract(
            track="showcase",
            strict_fairness=False,
            scenario_budgets={"negation_correction": {"retrieval_limit": 5, "evidence_max_tokens": 120, "answer_budget_tokens": 70}},
            vector_provider_family="none",
            answer_model=None,
            answer_provider=None,
            answer_prompt=None,
            transcript_invariant=True,
        ),
        primary_baselines=["engram_full", "context_summary"],
        appendix_baselines=[],
        ablation_baselines=[],
        scenario_results=[
            ScenarioResult(
                scenario_id="negation_correction",
                scenario_title="Negation And Correction",
                why_it_matters="Negative polarity should suppress stale relationships.",
                baseline_name="engram_full",
                baseline_family="engram",
                seed=7,
                capability_tags=["temporal", "negation"],
                available=True,
                passed=False,
            ),
            ScenarioResult(
                scenario_id="negation_correction",
                scenario_title="Negation And Correction",
                why_it_matters="Negative polarity should suppress stale relationships.",
                baseline_name="context_summary",
                baseline_family="alternative",
                seed=7,
                capability_tags=["temporal", "negation"],
                available=True,
                passed=False,
            ),
        ],
        baseline_summaries=[
            _baseline_summary("engram_full", pass_rate=0.0),
            _baseline_summary("context_summary", pass_rate=0.0),
        ],
    )

    report = render_markdown_report(run_result)

    assert "| Negation And Correction | No Baseline Passed |" in report
    assert "- Why it matters: Negative polarity should suppress stale relationships." in report
