"""Integration tests for the showcase benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram.benchmark.showcase.adapters import create_primary_adapter
from engram.benchmark.showcase.models import ExtractionSpec, to_serializable
from engram.benchmark.showcase.runner import run_showcase_benchmark
from engram.benchmark.showcase.scenarios import build_showcase_scenarios


def _normalize_run_result(result) -> dict:
    data = to_serializable(result)
    return {
        "track": data["track"],
        "mode": data["mode"],
        "seeds": data["seeds"],
        "primary_baselines": data["primary_baselines"],
        "appendix_baselines": data["appendix_baselines"],
        "ablation_baselines": data["ablation_baselines"],
        "fairness_contract": {
            "track": data["fairness_contract"]["track"],
            "strict_fairness": data["fairness_contract"]["strict_fairness"],
            "scenario_budgets": data["fairness_contract"]["scenario_budgets"],
            "vector_provider_family": data["fairness_contract"]["vector_provider_family"],
            "answer_model": data["fairness_contract"]["answer_model"],
            "answer_provider": data["fairness_contract"]["answer_provider"],
            "transcript_hashes": data["fairness_contract"]["transcript_hashes"],
        },
        "scenario_results": [
            {
                "scenario_id": scenario["scenario_id"],
                "baseline_name": scenario["baseline_name"],
                "available": scenario["available"],
                "passed": scenario["passed"],
                "capability_tags": scenario["capability_tags"],
                "cost_stats": {
                    "observed_turns": scenario["cost_stats"]["observed_turns"],
                    "projected_turns": scenario["cost_stats"]["projected_turns"],
                    "extraction_calls": scenario["cost_stats"]["extraction_calls"],
                    "embedding_calls": scenario["cost_stats"]["embedding_calls"],
                    "consolidation_cycles": scenario["cost_stats"]["consolidation_cycles"],
                    "method_calls": scenario["cost_stats"]["method_calls"],
                },
                "probe_results": [
                    {
                        "probe_id": probe["probe_id"],
                        "passed": probe["passed"],
                        "required_hits": probe["required_hits"],
                        "missing_required": probe["missing_required"],
                        "forbidden_hits": probe["forbidden_hits"],
                        "expected_type_match": probe["expected_type_match"],
                        "returned_types": probe["returned_types"],
                        "tokens_surfaced": probe["tokens_surfaced"],
                        "disallowed_type_hits": probe["disallowed_type_hits"],
                        "historical_violation": probe["historical_violation"],
                    }
                    for probe in scenario["probe_results"]
                ],
            }
            for scenario in data["scenario_results"]
        ],
        "baseline_summaries": [
            {
                "baseline_name": summary["baseline_name"],
                "available": summary["available"],
                "scenario_pass_rate": summary["scenario_pass_rate"],
                "false_recall_rate": summary["false_recall_rate"],
                "required_hit_rate": summary["required_hit_rate"],
                "forbidden_hit_rate": summary["forbidden_hit_rate"],
                "token_efficiency": summary["token_efficiency"],
                "cost_proxies": summary["cost_proxies"],
            }
            for summary in data["baseline_summaries"]
        ],
        "answer_results": [
            {
                "scenario_id": result["scenario_id"],
                "baseline_name": result["baseline_name"],
                "available": result["available"],
                "passed": result["passed"],
                "score": result["score"],
            }
            for result in data["answer_results"]
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_id",
    [
        "temporal_override",
        "prospective_trigger",
        "negation_correction",
        "correction_chain",
        "latent_open_loop_cue",
        "summary_drift_resistance",
    ],
)
async def test_engram_full_golden_scenarios(tmp_path: Path, scenario_id: str):
    result = await run_showcase_benchmark(
        mode="full",
        seeds=[7],
        output_dir=tmp_path / scenario_id,
        scenario_ids=[scenario_id],
        baseline_names=["engram_full"],
        include_ablations=False,
    )

    assert len(result.scenario_results) == 1
    scenario_result = result.scenario_results[0]
    assert scenario_result.baseline_name == "engram_full"
    assert scenario_result.passed is True


@pytest.mark.asyncio
async def test_quick_mode_smoke_and_artifacts(tmp_path: Path):
    output_dir = tmp_path / "quick"
    result = await run_showcase_benchmark(
        mode="quick",
        seeds=[7],
        output_dir=output_dir,
        strict_fairness=True,
        emit_readme_snippet=True,
        website_export_path=tmp_path / "website" / "latest.json",
    )

    assert result.primary_baselines == [
        "engram_full",
        "langgraph_store_memory",
        "mem0_style_memory",
        "graphiti_temporal_graph",
        "context_summary",
        "markdown_canonical",
        "hybrid_rag_temporal",
    ]
    assert result.headline_baselines == [
        "engram_full",
        "langgraph_store_memory",
        "mem0_style_memory",
        "graphiti_temporal_graph",
    ]
    assert result.control_baselines == [
        "context_summary",
        "markdown_canonical",
        "hybrid_rag_temporal",
    ]
    assert result.appendix_baselines == [
        "context_window",
        "markdown_memory",
        "vector_rag",
    ]
    assert (output_dir / "results.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "scenario_details.json").exists()
    assert (output_dir / "fairness_contract.json").exists()
    assert (output_dir / "website_summary.json").exists()
    assert (output_dir / "readme_snippet.md").exists()
    assert (tmp_path / "website" / "latest.json").exists()
    assert any(summary.baseline_name == "engram_full" for summary in result.baseline_summaries)

    scenario_catalog = {
        scenario.id: scenario
        for scenario in build_showcase_scenarios(mode="quick", seed=7)
    }
    for scenario_result in result.scenario_results:
        if not scenario_result.available:
            continue
        probe_specs = {
            probe.id: probe
            for probe in scenario_catalog[scenario_result.scenario_id].probes
        }
        for probe_result in scenario_result.probe_results:
            assert probe_result.tokens_surfaced <= probe_specs[probe_result.probe_id].max_tokens

    assert result.fairness_contract.baseline_contracts
    assert result.readme_snippet is not None


@pytest.mark.asyncio
async def test_cue_ablation_requires_real_cue_path(tmp_path: Path):
    result = await run_showcase_benchmark(
        mode="quick",
        seeds=[7],
        output_dir=tmp_path / "cue_ablation",
        scenario_ids=["cue_delayed_relevance"],
        baseline_names=["engram_full", "engram_no_cues"],
    )

    by_name = {scenario.baseline_name: scenario for scenario in result.scenario_results}
    assert by_name["engram_full"].passed is True
    assert by_name["engram_no_cues"].passed is False


@pytest.mark.asyncio
async def test_search_only_fails_latent_open_loop_cue_scenario(tmp_path: Path):
    result = await run_showcase_benchmark(
        mode="full",
        seeds=[7],
        output_dir=tmp_path / "latent_open_loop_cue",
        scenario_ids=["latent_open_loop_cue"],
        baseline_names=["engram_full", "engram_search_only"],
        include_ablations=False,
    )

    by_name = {scenario.baseline_name: scenario for scenario in result.scenario_results}
    assert by_name["engram_full"].passed is True
    assert by_name["engram_search_only"].passed is False


@pytest.mark.asyncio
async def test_quick_mode_is_deterministic_for_fixed_seed(tmp_path: Path):
    first = await run_showcase_benchmark(
        mode="quick",
        seeds=[11],
        output_dir=tmp_path / "run_a",
        baseline_names=["engram_full", "context_summary", "markdown_canonical"],
        include_ablations=False,
    )
    second = await run_showcase_benchmark(
        mode="quick",
        seeds=[11],
        output_dir=tmp_path / "run_b",
        baseline_names=["engram_full", "context_summary", "markdown_canonical"],
        include_ablations=False,
    )

    assert json.dumps(_normalize_run_result(first), sort_keys=True) == json.dumps(
        _normalize_run_result(second),
        sort_keys=True,
    )


@pytest.mark.asyncio
async def test_answer_track_skips_cleanly_without_model(tmp_path: Path):
    result = await run_showcase_benchmark(
        track="answer",
        mode="quick",
        seeds=[7],
        output_dir=tmp_path / "answer_skip",
        baseline_names=["engram_full"],
        include_ablations=False,
    )

    assert result.answer_results
    assert all(answer.available is False for answer in result.answer_results)
    assert all(
        answer.availability_reason == "answer model not configured"
        for answer in result.answer_results
    )
    answer_track = next(summary for summary in result.track_summaries if summary.track == "answer")
    assert answer_track.available is False
    assert answer_track.availability_reason == "answer model not configured"


@pytest.mark.asyncio
async def test_answer_track_runs_with_deterministic_provider(tmp_path: Path):
    result = await run_showcase_benchmark(
        track="all",
        mode="quick",
        seeds=[7],
        output_dir=tmp_path / "answer_det",
        baseline_names=["engram_full"],
        include_ablations=False,
        answer_model="deterministic",
    )

    assert result.answer_results
    assert all(answer.available for answer in result.answer_results)
    assert (tmp_path / "answer_det" / "answer_outputs.json").exists()


@pytest.mark.asyncio
async def test_langgraph_store_memory_is_more_token_efficient_than_context_summary(tmp_path: Path):
    result = await run_showcase_benchmark(
        mode="full",
        seeds=[7],
        output_dir=tmp_path / "langgraph_vs_context",
        scenario_ids=["multi_session_continuity", "context_budget_compression"],
        baseline_names=["langgraph_store_memory", "context_summary"],
        include_ablations=False,
    )

    by_name = {
        name: [
            scenario
            for scenario in result.scenario_results
            if scenario.baseline_name == name
        ]
        for name in ["langgraph_store_memory", "context_summary"]
    }
    assert all(scenario.passed for scenario in by_name["langgraph_store_memory"])
    assert all(scenario.passed for scenario in by_name["context_summary"])

    langgraph_tokens = sum(
        probe.tokens_surfaced
        for scenario in by_name["langgraph_store_memory"]
        for probe in scenario.probe_results
    )
    context_tokens = sum(
        probe.tokens_surfaced
        for scenario in by_name["context_summary"]
        for probe in scenario.probe_results
    )
    assert langgraph_tokens < context_tokens


@pytest.mark.asyncio
async def test_mem0_style_memory_beats_markdown_canonical_on_current_state(
    tmp_path: Path,
):
    result = await run_showcase_benchmark(
        mode="full",
        seeds=[7],
        output_dir=tmp_path / "mem0_vs_markdown",
        scenario_ids=[
            "temporal_override",
            "negation_correction",
            "correction_chain",
            "summary_drift_resistance",
        ],
        baseline_names=["mem0_style_memory", "markdown_canonical"],
        include_ablations=False,
    )

    by_name = {
        name: [
            scenario
            for scenario in result.scenario_results
            if scenario.baseline_name == name
        ]
        for name in ["mem0_style_memory", "markdown_canonical"]
    }
    assert all(scenario.passed for scenario in by_name["mem0_style_memory"])
    assert not any(scenario.passed for scenario in by_name["markdown_canonical"])


@pytest.mark.asyncio
async def test_graphiti_temporal_graph_beats_hybrid_rag_on_graph_temporal(
    tmp_path: Path,
):
    result = await run_showcase_benchmark(
        mode="full",
        seeds=[7],
        output_dir=tmp_path / "graphiti_vs_hybrid",
        scenario_ids=[
            "cross_cluster_association",
            "temporal_override",
            "negation_correction",
        ],
        baseline_names=["graphiti_temporal_graph", "hybrid_rag_temporal"],
        include_ablations=False,
    )

    by_name = {
        name: [
            scenario
            for scenario in result.scenario_results
            if scenario.baseline_name == name
        ]
        for name in ["graphiti_temporal_graph", "hybrid_rag_temporal"]
    }
    assert all(scenario.passed for scenario in by_name["graphiti_temporal_graph"])
    assert not any(scenario.passed for scenario in by_name["hybrid_rag_temporal"])


@pytest.mark.asyncio
async def test_external_proxy_baselines_stay_below_engram_on_cue_prospective(
    tmp_path: Path,
):
    result = await run_showcase_benchmark(
        mode="full",
        seeds=[7],
        output_dir=tmp_path / "engram_vs_external_limits",
        scenario_ids=[
            "cue_delayed_relevance",
            "prospective_trigger",
            "latent_open_loop_cue",
        ],
        baseline_names=[
            "engram_full",
            "langgraph_store_memory",
            "mem0_style_memory",
            "graphiti_temporal_graph",
        ],
        include_ablations=False,
    )

    assert all(
        scenario.passed
        for scenario in result.scenario_results
        if scenario.baseline_name == "engram_full"
    )
    for baseline_name in [
        "langgraph_store_memory",
        "mem0_style_memory",
        "graphiti_temporal_graph",
    ]:
        assert not any(
            scenario.passed
            for scenario in result.scenario_results
            if scenario.baseline_name == baseline_name
        )


@pytest.mark.asyncio
async def test_website_summary_includes_grouped_baselines_and_spec_targets(tmp_path: Path):
    output_dir = tmp_path / "website_summary"
    await run_showcase_benchmark(
        mode="quick",
        seeds=[7],
        output_dir=output_dir,
        include_ablations=False,
        website_export_path=tmp_path / "website" / "latest.json",
    )

    website_summary = json.loads((output_dir / "website_summary.json").read_text())
    assert [item["name"] for item in website_summary["headline_baselines"]] == [
        "engram_full",
        "langgraph_store_memory",
        "mem0_style_memory",
        "graphiti_temporal_graph",
    ]
    assert [item["name"] for item in website_summary["control_baselines"]] == [
        "context_summary",
        "markdown_canonical",
        "hybrid_rag_temporal",
    ]
    assert [item["baseline_id"] for item in website_summary["spec_only_baselines"]] == [
        "letta",
        "llamaindex_memory",
        "crewai_memory",
    ]


@pytest.mark.asyncio
async def test_answer_track_uses_scenario_probe_retrieval_cue(tmp_path: Path):
    result = await run_showcase_benchmark(
        track="answer",
        mode="quick",
        seeds=[7],
        output_dir=tmp_path / "answer_probe",
        scenario_ids=["prospective_trigger"],
        baseline_names=["engram_full"],
        include_ablations=False,
        answer_model="deterministic",
    )

    assert len(result.answer_results) == 1
    answer = result.answer_results[0]
    assert answer.available is True
    assert answer.passed is True


@pytest.mark.asyncio
async def test_external_track_status_writes_appendix_artifact(tmp_path: Path):
    result = await run_showcase_benchmark(
        track="external",
        mode="quick",
        seeds=[7],
        output_dir=tmp_path / "external",
    )

    assert result.external_track_results
    assert (tmp_path / "external" / "external_tracks.json").exists()
    external_track = next(
        summary for summary in result.track_summaries
        if summary.track == "external"
    )
    assert external_track.executed is True


@pytest.mark.asyncio
async def test_hybrid_voyage_baseline_is_reported_unavailable_without_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    result = await run_showcase_benchmark(
        mode="quick",
        seeds=[7],
        output_dir=tmp_path / "voyage_unavailable",
        scenario_ids=["temporal_override"],
        baseline_names=["engram_full_hybrid"],
        include_ablations=False,
        engram_vector_provider="voyage",
    )

    assert len(result.scenario_results) == 1
    scenario_result = result.scenario_results[0]
    assert scenario_result.baseline_name == "engram_full_hybrid"
    assert scenario_result.available is False
    assert scenario_result.availability_reason == "VOYAGE_API_KEY not set"


def test_primary_adapters_expose_fair_budget_contracts():
    extraction_map = {"hello": ExtractionSpec()}

    for baseline_name in [
        "engram_full",
        "context_summary",
        "markdown_canonical",
        "hybrid_rag_temporal",
    ]:
        adapter = create_primary_adapter(baseline_name, extraction_map)
        contract = adapter.budget_contract()
        assert contract["evidence_budget_source"] == "scenario_probe"
        assert contract["retrieval_limit_source"] == "scenario_probe"
        assert contract["answer_prompt_id"] == "showcase_answer_v2"
