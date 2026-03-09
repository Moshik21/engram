"""Runner for the showcase benchmark suite."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from engram.benchmark.showcase.adapters import (
    build_extraction_map,
    create_ablation_adapter,
    create_primary_adapter,
)
from engram.benchmark.showcase.answering import grade_answer, shared_answer_prompt
from engram.benchmark.showcase.catalog import (
    BASELINE_CATALOG,
    DEFAULT_ABLATION_BASELINES,
    DEFAULT_APPENDIX_BASELINES,
    DEFAULT_CONTROL_BASELINES,
    DEFAULT_HEADLINE_BASELINES,
    DEFAULT_PRIMARY_BASELINES,
    DEFAULT_SPEC_ONLY_BASELINES,
)
from engram.benchmark.showcase.external import collect_external_track_results
from engram.benchmark.showcase.models import (
    AnswerResult,
    FairnessContract,
    ScenarioProbe,
    ScenarioResult,
    ShowcaseRunResult,
    TrackSummary,
    to_serializable,
)
from engram.benchmark.showcase.report import render_markdown_report
from engram.benchmark.showcase.scenarios import build_showcase_scenarios
from engram.benchmark.showcase.scoring import (
    score_probe,
    summarize_answer_results,
    summarize_baseline,
)


def _default_seeds(mode: str) -> list[int]:
    if mode == "quick":
        return [7]
    return [7, 19, 31]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(__file__).resolve().parents[3] / ".benchmarks" / "showcase" / timestamp


def _resolve_baseline_groups(
    *,
    baseline_names: list[str] | None,
    primary_baselines: list[str] | None,
    appendix_baselines: list[str] | None,
    include_ablations: bool,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    if baseline_names is not None:
        selected = _dedupe(list(baseline_names))
        primary = [
            baseline
            for baseline in DEFAULT_PRIMARY_BASELINES + ["engram_full_hybrid"]
            if baseline in selected
        ]
        headline = [baseline for baseline in DEFAULT_HEADLINE_BASELINES if baseline in primary]
        control = [baseline for baseline in DEFAULT_CONTROL_BASELINES if baseline in primary]
        appendix = [baseline for baseline in DEFAULT_APPENDIX_BASELINES if baseline in selected]
        ablations = [baseline for baseline in DEFAULT_ABLATION_BASELINES if baseline in selected]
        return selected, headline, control, appendix, ablations

    primary = _dedupe(list(primary_baselines or DEFAULT_PRIMARY_BASELINES))
    headline = [baseline for baseline in DEFAULT_HEADLINE_BASELINES if baseline in primary]
    control = [baseline for baseline in DEFAULT_CONTROL_BASELINES if baseline in primary]
    appendix = _dedupe(list(appendix_baselines or DEFAULT_APPENDIX_BASELINES))
    ablations = list(DEFAULT_ABLATION_BASELINES if include_ablations else [])
    selected = _dedupe(primary + appendix + ablations)
    return selected, headline, control, appendix, ablations


def _supports_showcase(track: str) -> bool:
    return track in {"showcase", "all"}


def _supports_answer(track: str) -> bool:
    return track in {"answer", "all"}


def _supports_external(track: str) -> bool:
    return track in {"external", "all"}


def _scenario_lookup(
    *,
    mode: str,
    seed: int,
    scenario_ids: list[str] | None,
):
    scenarios = build_showcase_scenarios(mode=mode, seed=seed)
    if scenario_ids:
        requested = set(scenario_ids)
        scenarios = [scenario for scenario in scenarios if scenario.id in requested]
    return scenarios


def _scenario_budgets(scenarios) -> dict[str, dict[str, int]]:
    return {
        scenario.id: {
            "retrieval_limit": scenario.budget_profile.retrieval_limit,
            "evidence_max_tokens": scenario.budget_profile.evidence_max_tokens,
            "answer_budget_tokens": scenario.budget_profile.answer_budget_tokens,
        }
        for scenario in scenarios
    }


def _transcript_hashes(scenarios) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for scenario in scenarios:
        payload = json.dumps(
            to_serializable(scenario.turns),
            sort_keys=True,
            separators=(",", ":"),
        )
        hashes[scenario.id] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hashes


def _overall_vector_provider_family(
    selected_baselines: list[str], engram_vector_provider: str,
) -> str:
    vector_families: set[str] = set()
    for baseline in selected_baselines:
        if baseline in {"hybrid_rag_temporal", "vector_rag", "graphiti_temporal_graph"}:
            vector_families.add("local")
        elif baseline == "engram_full_hybrid":
            if engram_vector_provider != "none":
                vector_families.add(engram_vector_provider)
    if not vector_families:
        return "none"
    if len(vector_families) == 1:
        return next(iter(vector_families))
    return "mixed"


def _answer_track_reason(answer_model: str | None, answer_provider: str | None) -> str | None:
    if not answer_model:
        return "answer model not configured"
    effective_provider = answer_provider or "deterministic"
    if effective_provider != "deterministic":
        return f"unsupported answer provider: {effective_provider}"
    return None


def _build_answer_probe(scenario) -> ScenarioProbe:
    first_probe = scenario.probes[0]
    task = scenario.answer_task
    if task is None:
        raise ValueError(f"Scenario {scenario.id} has no answer task")
    return ScenarioProbe(
        id=f"{scenario.id}_answer_probe",
        after_turn_index=first_probe.after_turn_index,
        operation=first_probe.operation,
        query=first_probe.query or task.question,
        topic_hint=first_probe.topic_hint or first_probe.query or task.question,
        limit=scenario.budget_profile.retrieval_limit,
        max_tokens=scenario.budget_profile.evidence_max_tokens,
        capability_tags=list(scenario.capability_tags),
        note=task.note,
    )


def _create_adapter(
    baseline_name: str,
    extraction_map,
    *,
    engram_vector_provider: str,
    budget_profile,
):
    if baseline_name in DEFAULT_PRIMARY_BASELINES or baseline_name in DEFAULT_APPENDIX_BASELINES:
        return create_primary_adapter(
            baseline_name,
            extraction_map,
            engram_vector_provider=engram_vector_provider,
            budget_profile=budget_profile,
        )
    if baseline_name == "engram_full_hybrid":
        return create_primary_adapter(
            baseline_name,
            extraction_map,
            engram_vector_provider=engram_vector_provider,
            budget_profile=budget_profile,
        )
    if baseline_name in DEFAULT_ABLATION_BASELINES:
        return create_ablation_adapter(baseline_name, extraction_map)
    raise ValueError(f"Unknown baseline: {baseline_name}")


def _build_showcase_summaries(
    scenario_results: list[ScenarioResult],
    selected_baselines: list[str],
    ablation_baselines: list[str],
):
    summaries = []
    by_baseline: dict[str, list[ScenarioResult]] = {}
    for baseline_name in selected_baselines:
        by_baseline[baseline_name] = [
            result for result in scenario_results if result.baseline_name == baseline_name
        ]
    for baseline_name in selected_baselines:
        results = by_baseline.get(baseline_name, [])
        if results:
            family = results[0].baseline_family
        else:
            family = (
                BASELINE_CATALOG[baseline_name].family
                if baseline_name in BASELINE_CATALOG
                else "ablation" if baseline_name in ablation_baselines else "alternative"
            )
        summaries.append(
            summarize_baseline(
                baseline_name,
                family,
                baseline_name in ablation_baselines,
                results,
            )
        )
    return summaries


def _build_answer_summaries(answer_results: list[AnswerResult], selected_baselines: list[str]):
    summaries = []
    for baseline_name in selected_baselines:
        results = [result for result in answer_results if result.baseline_name == baseline_name]
        family = (
            results[0].baseline_family
            if results
            else BASELINE_CATALOG[baseline_name].family
            if baseline_name in BASELINE_CATALOG
            else "alternative"
        )
        summaries.append(summarize_answer_results(baseline_name, family, results))
    return summaries


def _validate_fairness(contract: FairnessContract) -> None:
    prompt_ids = {
        str(details.get("answer_prompt_id"))
        for details in contract.baseline_contracts.values()
        if details.get("answer_prompt_id") is not None
    }
    if len(prompt_ids) > 1:
        raise ValueError(f"Fairness violation: answer prompt ids differ: {sorted(prompt_ids)}")

    for baseline_name, details in contract.baseline_contracts.items():
        if details.get("evidence_budget_source") != "scenario_probe":
            raise ValueError(
                f"Fairness violation: {baseline_name} does not honor"
                " scenario probe evidence budgets"
            )
        if details.get("retrieval_limit_source") != "scenario_probe":
            raise ValueError(
                f"Fairness violation: {baseline_name} does not honor"
                " scenario probe retrieval limits"
            )

    vector_families = {
        str(details.get("vector_provider_family"))
        for details in contract.baseline_contracts.values()
        if details.get("vector_provider_family") not in {None, "none"}
    }
    if len(vector_families) > 1:
        raise ValueError(
            "Fairness violation: multiple vector provider families"
            f" in comparable baselines: {sorted(vector_families)}"
        )


def _track_summaries(
    *,
    track: str,
    baseline_summaries,
    answer_summaries,
    external_track_results,
    answer_reason: str | None,
) -> list[TrackSummary]:
    summaries: list[TrackSummary] = []

    if _supports_showcase(track):
        engram_summary = next(
            (summary for summary in baseline_summaries if summary.baseline_name == "engram_full"),
            None,
        )
        summaries.append(
            TrackSummary(
                track="showcase",
                executed=True,
                available=bool(baseline_summaries),
                headline_metric=(
                    None if engram_summary is None else engram_summary.scenario_pass_rate
                ),
                notes=["Deterministic evidence scoring over flagship memory scenarios."],
            )
        )

    if _supports_answer(track):
        engram_answer = next(
            (summary for summary in answer_summaries if summary.baseline_name == "engram_full"),
            None,
        )
        summaries.append(
            TrackSummary(
                track="answer",
                executed=answer_reason is None,
                available=answer_reason is None and bool(answer_summaries),
                availability_reason=answer_reason,
                headline_metric=(
                    None if engram_answer is None else engram_answer.answer_pass_rate
                ),
                notes=["Shared answer prompt + deterministic grader."],
            )
        )

    if _supports_external(track):
        summaries.append(
            TrackSummary(
                track="external",
                executed=True,
                available=bool(external_track_results),
                headline_metric=None,
                notes=["Supporting benchmark status only; not blended into the headline score."],
            )
        )

    return summaries


def _supporting_artifacts(project_root: Path) -> dict[str, str]:
    return {
        "benchmark_ab": str(project_root / "scripts" / "benchmark_ab.py"),
        "benchmark_working_memory": str(project_root / "scripts" / "benchmark_working_memory.py"),
        "benchmark_echo_chamber": str(project_root / "scripts" / "benchmark_echo_chamber.py"),
        "benchmark_locomo": str(project_root / "scripts" / "benchmark_locomo.py"),
    }


def _build_readme_snippet(run_result: ShowcaseRunResult) -> str | None:
    if not run_result.baseline_summaries:
        return None
    summary_by_name = {
        summary.baseline_name: summary for summary in run_result.baseline_summaries
    }
    engram = summary_by_name.get("engram_full")
    primary_competitors = [
        summary_by_name[name]
        for name in run_result.headline_baselines
        if name != "engram_full" and name in summary_by_name
    ]
    if engram is None or not primary_competitors:
        return None
    competitor_clause = ", ".join(
        f"{BASELINE_CATALOG[name].display_name} {summary_by_name[name].scenario_pass_rate:.3f}"
        for name in run_result.headline_baselines
        if name != "engram_full" and name in summary_by_name
    )
    return (
        f"Benchmark results ({run_result.mode}, measured against equal retrieval budgets): "
        f"`engram_full` passed {engram.scenario_pass_rate:.3f} of showcase scenarios with "
        f"false recall {engram.false_recall_rate:.3f}, versus {competitor_clause}."
    )


def _summary_payload(summary, baseline_id: str) -> dict[str, object]:
    entry = BASELINE_CATALOG.get(baseline_id)
    return {
        "name": summary.baseline_name,
        "display_name": (
            entry.display_name
            if entry is not None
            else baseline_id.replace("_", " ").title()
        ),
        "family": summary.baseline_family,
        "comparison_group": None if entry is None else entry.comparison_group,
        "status": None if entry is None else entry.status,
        "external_technology_label": None if entry is None else entry.external_technology_label,
        "accent": None if entry is None else entry.accent,
        "archetype": None if entry is None else entry.archetype,
        "description": None if entry is None else entry.description,
        "fairness_notes": None if entry is None else entry.fairness_notes,
        "known_limitations": None if entry is None else entry.known_limitations,
        "why_included": None if entry is None else entry.why_included,
        "scenario_pass_rate": summary.scenario_pass_rate,
        "false_recall_rate": summary.false_recall_rate,
        "temporal_correctness": summary.temporal_correctness,
        "negation_correctness": summary.negation_correctness,
        "open_loop_recovery": summary.open_loop_recovery,
        "prospective_trigger_rate": summary.prospective_trigger_rate,
        "latency_p50_ms": summary.latency_p50_ms,
        "latency_p95_ms": summary.latency_p95_ms,
    }


def _build_website_summary(run_result: ShowcaseRunResult) -> dict[str, object]:
    summary_by_name = {
        summary.baseline_name: summary for summary in run_result.baseline_summaries
    }
    headline_summaries = [
        _summary_payload(summary_by_name[name], name)
        for name in run_result.headline_baselines
        if name in summary_by_name
    ]
    control_summaries = [
        _summary_payload(summary_by_name[name], name)
        for name in run_result.control_baselines
        if name in summary_by_name
    ]
    primary_summaries = headline_summaries + control_summaries
    appendix_summaries = [
        _summary_payload(summary_by_name[name], name)
        for name in run_result.appendix_baselines
        if name in summary_by_name
    ]

    grouped: dict[tuple[str, str], list[bool]] = {}
    titles: dict[str, str] = {}
    for result in run_result.scenario_results:
        if result.baseline_name not in run_result.primary_baselines or not result.available:
            continue
        grouped.setdefault((result.scenario_id, result.baseline_name), []).append(result.passed)
        titles.setdefault(result.scenario_id, result.scenario_title)

    scenario_winners: list[dict[str, object]] = []
    for scenario_id, title in sorted(titles.items()):
        entries: list[tuple[str, float]] = []
        for baseline_name in run_result.primary_baselines:
            passes = grouped.get((scenario_id, baseline_name), [])
            if passes:
                entries.append((baseline_name, sum(1 for passed in passes if passed) / len(passes)))
        if not entries:
            continue
        winner_name, winner_score = max(entries, key=lambda item: item[1])
        scenario_winners.append(
            {
                "scenario_id": scenario_id,
                "title": title,
                "winner": winner_name if winner_score > 0.0 else None,
                "winner_score": winner_score,
            }
        )

    return {
        "generated_at": run_result.generated_at,
        "track": run_result.track,
        "mode": run_result.mode,
        "seeds": run_result.seeds,
        "headline": {
            "engram_full_pass_rate": summary_by_name.get("engram_full").scenario_pass_rate
            if "engram_full" in summary_by_name
            else None,
            "engram_full_false_recall": summary_by_name.get("engram_full").false_recall_rate
            if "engram_full" in summary_by_name
            else None,
            "best_headline_competitor_pass_rate": max(
                (
                    summary_by_name[name].scenario_pass_rate
                    for name in run_result.headline_baselines
                    if name != "engram_full" and name in summary_by_name
                ),
                default=None,
            ),
        },
        "headline_baselines": headline_summaries,
        "control_baselines": control_summaries,
        "primary_baselines": primary_summaries,
        "appendix_baselines": appendix_summaries,
        "ablations": [
            {
                "name": summary.baseline_name,
                "display_name": BASELINE_CATALOG[summary.baseline_name].display_name
                if summary.baseline_name in BASELINE_CATALOG
                else summary.baseline_name.replace("_", " ").title(),
                "scenario_pass_rate": summary.scenario_pass_rate,
                "false_recall_rate": summary.false_recall_rate,
            }
            for summary in run_result.baseline_summaries
            if summary.baseline_name in run_result.ablation_baselines
        ],
        "spec_only_baselines": [
            to_serializable(run_result.baseline_catalog[baseline_id])
            for baseline_id in run_result.spec_only_baselines
            if baseline_id in run_result.baseline_catalog
        ],
        "baseline_catalog": {
            baseline_id: to_serializable(entry)
            for baseline_id, entry in run_result.baseline_catalog.items()
        },
        "scenario_winners": scenario_winners,
    }


def _write_artifacts(
    run_result: ShowcaseRunResult,
    *,
    output_path: Path,
    scenario_catalog,
    emit_readme_snippet: bool,
    website_export_path: str | Path | None,
) -> dict[str, str]:
    results_path = output_path / "results.json"
    report_path = output_path / "report.md"
    scenario_details_path = output_path / "scenario_details.json"
    fairness_path = output_path / "fairness_contract.json"
    website_summary_path = output_path / "website_summary.json"
    artifact_paths: dict[str, str] = {
        "results": str(results_path),
        "report": str(report_path),
        "scenario_details": str(scenario_details_path),
        "fairness_contract": str(fairness_path),
        "website_summary": str(website_summary_path),
    }

    if run_result.answer_results:
        answer_outputs_path = output_path / "answer_outputs.json"
        artifact_paths["answer_outputs"] = str(answer_outputs_path)

    if run_result.external_track_results:
        external_path = output_path / "external_tracks.json"
        artifact_paths["external_tracks"] = str(external_path)

    if emit_readme_snippet and run_result.readme_snippet:
        snippet_path = output_path / "readme_snippet.md"
        artifact_paths["readme_snippet"] = str(snippet_path)

    run_result.artifact_paths = artifact_paths
    results_path.write_text(
        json.dumps(to_serializable(run_result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown_report(run_result), encoding="utf-8")
    scenario_details_path.write_text(
        json.dumps(
            {
                "scenario_catalog": to_serializable(scenario_catalog),
                "scenario_results": to_serializable(run_result.scenario_results),
                "answer_results": to_serializable(run_result.answer_results),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    fairness_path.write_text(
        json.dumps(to_serializable(run_result.fairness_contract), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    website_summary = _build_website_summary(run_result)
    website_summary_path.write_text(
        json.dumps(website_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if run_result.answer_results:
        answer_outputs_path = output_path / "answer_outputs.json"
        answer_outputs_path.write_text(
            json.dumps(to_serializable(run_result.answer_results), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if run_result.external_track_results:
        external_path = output_path / "external_tracks.json"
        external_path.write_text(
            json.dumps(
                to_serializable(run_result.external_track_results),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if emit_readme_snippet and run_result.readme_snippet:
        snippet_path = output_path / "readme_snippet.md"
        snippet_path.write_text(run_result.readme_snippet + "\n", encoding="utf-8")

    if website_export_path is not None:
        export_path = Path(website_export_path).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(
            json.dumps(website_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_paths["website_export"] = str(export_path)

    return artifact_paths


async def run_showcase_benchmark(
    *,
    mode: str = "quick",
    track: str = "showcase",
    seeds: list[int] | None = None,
    output_dir: str | Path | None = None,
    scenario_ids: list[str] | None = None,
    baseline_names: list[str] | None = None,
    primary_baselines: list[str] | None = None,
    appendix_baselines: list[str] | None = None,
    include_ablations: bool = True,
    engram_vector_provider: str = "none",
    answer_model: str | None = None,
    answer_provider: str | None = None,
    strict_fairness: bool = False,
    emit_readme_snippet: bool = False,
    locomo_dataset_path: str | None = None,
    website_export_path: str | Path | None = None,
) -> ShowcaseRunResult:
    """Execute the benchmark suite and write JSON + Markdown artifacts."""
    seeds = list(seeds or _default_seeds(mode))
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    output_path = _resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (
        selected_baselines,
        selected_headline,
        selected_control,
        selected_appendix,
        selected_ablations,
    ) = (
        _resolve_baseline_groups(
            baseline_names=baseline_names,
            primary_baselines=primary_baselines,
            appendix_baselines=appendix_baselines,
            include_ablations=include_ablations,
        )
    )
    selected_primary = selected_headline + selected_control
    project_root = Path(__file__).resolve().parents[3]
    scenario_catalog = _scenario_lookup(mode=mode, seed=seeds[0], scenario_ids=scenario_ids)

    effective_answer_provider = answer_provider or ("deterministic" if answer_model else None)
    answer_reason = _answer_track_reason(answer_model, effective_answer_provider)
    answer_prompt = shared_answer_prompt() if answer_reason is None else None

    fairness_contract = FairnessContract(
        track=track,
        strict_fairness=strict_fairness,
        scenario_budgets=_scenario_budgets(scenario_catalog),
        vector_provider_family=_overall_vector_provider_family(
            selected_baselines,
            engram_vector_provider,
        ),
        answer_model=answer_model,
        answer_provider=effective_answer_provider,
        answer_prompt=answer_prompt,
        transcript_invariant=True,
        transcript_hashes=_transcript_hashes(scenario_catalog),
    )

    scenario_results: list[ScenarioResult] = []
    answer_results: list[AnswerResult] = []
    baseline_contracts: dict[str, dict[str, object]] = {}

    run_showcase = _supports_showcase(track)
    run_answer = _supports_answer(track)
    run_external = _supports_external(track)

    if run_showcase or run_answer:
        for seed in seeds:
            scenarios = _scenario_lookup(mode=mode, seed=seed, scenario_ids=scenario_ids)
            for scenario in scenarios:
                extraction_map = build_extraction_map(scenario.turns)
                probes_by_turn: dict[int, list[ScenarioProbe]] = {}
                if run_showcase:
                    for probe in scenario.probes:
                        probes_by_turn.setdefault(probe.after_turn_index, []).append(probe)

                for baseline_name in selected_baselines:
                    adapter = _create_adapter(
                        baseline_name,
                        extraction_map,
                        engram_vector_provider=engram_vector_provider,
                        budget_profile=scenario.budget_profile,
                    )
                    await adapter.initialize()
                    baseline_contracts.setdefault(
                        baseline_name,
                        dict(adapter.budget_contract()),
                    )
                    try:
                        if not adapter.available:
                            if run_showcase:
                                scenario_results.append(
                                    ScenarioResult(
                                        scenario_id=scenario.id,
                                        scenario_title=scenario.title,
                                        why_it_matters=scenario.why_it_matters,
                                        baseline_name=adapter.name,
                                        baseline_family=adapter.family,
                                        seed=seed,
                                        capability_tags=list(scenario.capability_tags),
                                        available=False,
                                        passed=False,
                                        cost_stats=adapter.stats,
                                        availability_reason=adapter.availability_reason,
                                    )
                                )
                            if run_answer and scenario.answer_task is not None:
                                answer_results.append(
                                    AnswerResult(
                                        scenario_id=scenario.id,
                                        scenario_title=scenario.title,
                                        baseline_name=adapter.name,
                                        baseline_family=adapter.family,
                                        seed=seed,
                                        available=False,
                                        passed=False,
                                        answer_task_question=scenario.answer_task.question,
                                        availability_reason=adapter.availability_reason,
                                    )
                                )
                            continue

                        probe_results = []
                        answer_probe = (
                            _build_answer_probe(scenario)
                            if run_answer
                            and scenario.answer_task is not None
                            and answer_reason is None
                            else None
                        )
                        answer_recorded = False
                        for turn_index, turn in enumerate(scenario.turns):
                            await adapter.apply_turn(turn)
                            if run_showcase:
                                for probe in probes_by_turn.get(turn_index, []):
                                    started = time.perf_counter()
                                    evidence = await adapter.retrieve_evidence(probe)
                                    latency_ms = (time.perf_counter() - started) * 1000.0
                                    probe_results.append(score_probe(probe, evidence, latency_ms))
                            if (
                                answer_probe is not None
                                and not answer_recorded
                                and turn_index == answer_probe.after_turn_index
                            ):
                                started = time.perf_counter()
                                evidence = await adapter.retrieve_evidence(answer_probe)
                                answer = await adapter.answer_question(
                                    scenario.answer_task,
                                    evidence,
                                    answer_prompt=answer_prompt or "",
                                    answer_model=answer_model or "",
                                    answer_provider=effective_answer_provider or "",
                                )
                                latency_ms = (time.perf_counter() - started) * 1000.0
                                (
                                    score,
                                    passed,
                                    matched_fields,
                                    missing_fields,
                                    incorrect_fields,
                                    normalized_answer,
                                ) = grade_answer(scenario.answer_task, answer)
                                answer_results.append(
                                    AnswerResult(
                                        scenario_id=scenario.id,
                                        scenario_title=scenario.title,
                                        baseline_name=adapter.name,
                                        baseline_family=adapter.family,
                                        seed=seed,
                                        available=True,
                                        passed=passed,
                                        answer_task_question=scenario.answer_task.question,
                                        answer=answer,
                                        normalized_answer=normalized_answer,
                                        score=score,
                                        matched_fields=matched_fields,
                                        missing_fields=missing_fields,
                                        incorrect_fields=incorrect_fields,
                                        latency_ms=latency_ms,
                                        tokens_surfaced=sum(item.tokens for item in evidence),
                                    )
                                )
                                answer_recorded = True

                        if run_showcase:
                            scenario_results.append(
                                ScenarioResult(
                                    scenario_id=scenario.id,
                                    scenario_title=scenario.title,
                                    why_it_matters=scenario.why_it_matters,
                                    baseline_name=adapter.name,
                                    baseline_family=adapter.family,
                                    seed=seed,
                                    capability_tags=list(scenario.capability_tags),
                                    available=True,
                                    passed=all(probe.passed for probe in probe_results),
                                    probe_results=probe_results,
                                    cost_stats=adapter.stats,
                                )
                            )

                        if run_answer and scenario.answer_task is not None:
                            if answer_reason is not None:
                                answer_results.append(
                                    AnswerResult(
                                        scenario_id=scenario.id,
                                        scenario_title=scenario.title,
                                        baseline_name=adapter.name,
                                        baseline_family=adapter.family,
                                        seed=seed,
                                        available=False,
                                        passed=False,
                                        answer_task_question=scenario.answer_task.question,
                                        availability_reason=answer_reason,
                                    )
                                )
                            elif not answer_recorded:
                                answer_probe = _build_answer_probe(scenario)
                                started = time.perf_counter()
                                evidence = await adapter.retrieve_evidence(answer_probe)
                                answer = await adapter.answer_question(
                                    scenario.answer_task,
                                    evidence,
                                    answer_prompt=answer_prompt or "",
                                    answer_model=answer_model or "",
                                    answer_provider=effective_answer_provider or "",
                                )
                                latency_ms = (time.perf_counter() - started) * 1000.0
                                (
                                    score,
                                    passed,
                                    matched_fields,
                                    missing_fields,
                                    incorrect_fields,
                                    normalized_answer,
                                ) = grade_answer(scenario.answer_task, answer)
                                answer_results.append(
                                    AnswerResult(
                                        scenario_id=scenario.id,
                                        scenario_title=scenario.title,
                                        baseline_name=adapter.name,
                                        baseline_family=adapter.family,
                                        seed=seed,
                                        available=True,
                                        passed=passed,
                                        answer_task_question=scenario.answer_task.question,
                                        answer=answer,
                                        normalized_answer=normalized_answer,
                                        score=score,
                                        matched_fields=matched_fields,
                                        missing_fields=missing_fields,
                                        incorrect_fields=incorrect_fields,
                                        latency_ms=latency_ms,
                                        tokens_surfaced=sum(item.tokens for item in evidence),
                                    )
                                )
                    finally:
                        await adapter.close()

    fairness_contract.baseline_contracts = baseline_contracts
    if strict_fairness and baseline_contracts:
        _validate_fairness(fairness_contract)

    baseline_summaries = _build_showcase_summaries(
        scenario_results,
        selected_baselines,
        selected_ablations,
    )
    answer_summaries = _build_answer_summaries(answer_results, selected_baselines)
    external_track_results = (
        collect_external_track_results(
            project_root=project_root,
            locomo_dataset_path=locomo_dataset_path,
        )
        if run_external
        else []
    )
    track_summaries = _track_summaries(
        track=track,
        baseline_summaries=baseline_summaries,
        answer_summaries=answer_summaries,
        external_track_results=external_track_results,
        answer_reason=answer_reason if run_answer else None,
    )

    run_result = ShowcaseRunResult(
        track=track,
        mode=mode,
        seeds=seeds,
        generated_at=generated_at,
        output_dir=str(output_path),
        fairness_contract=fairness_contract,
        primary_baselines=selected_primary,
        appendix_baselines=selected_appendix,
        ablation_baselines=selected_ablations,
        scenario_results=scenario_results,
        baseline_summaries=baseline_summaries,
        answer_results=answer_results,
        answer_summaries=answer_summaries,
        external_track_results=external_track_results,
        track_summaries=track_summaries,
        supporting_artifacts=_supporting_artifacts(project_root),
        headline_baselines=selected_headline,
        control_baselines=selected_control,
        spec_only_baselines=list(DEFAULT_SPEC_ONLY_BASELINES),
        baseline_catalog=dict(BASELINE_CATALOG),
    )
    run_result.readme_snippet = _build_readme_snippet(run_result) if emit_readme_snippet else None
    run_result.artifact_paths = _write_artifacts(
        run_result,
        output_path=output_path,
        scenario_catalog=scenario_catalog,
        emit_readme_snippet=emit_readme_snippet,
        website_export_path=website_export_path,
    )
    return run_result
