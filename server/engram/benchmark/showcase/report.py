"""Markdown reporting for the showcase benchmark."""

from __future__ import annotations

from collections import defaultdict

from engram.benchmark.showcase.catalog import display_name
from engram.benchmark.showcase.models import ShowcaseRunResult


def _display_name(name: str) -> str:
    return display_name(name)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _summary_by_name(run_result: ShowcaseRunResult) -> dict:
    return {
        summary.baseline_name: summary for summary in run_result.baseline_summaries
    }


def _answer_summary_by_name(run_result: ShowcaseRunResult) -> dict:
    return {
        summary.baseline_name: summary for summary in run_result.answer_summaries
    }


def _aggregate_primary_scenarios(run_result: ShowcaseRunResult) -> list[dict]:
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    answer_grouped: dict[tuple[str, str], list] = defaultdict(list)
    scenario_meta: dict[str, tuple[str, str, tuple[str, ...]]] = {}

    for result in run_result.scenario_results:
        if result.baseline_name not in run_result.primary_baselines or not result.available:
            continue
        grouped[(result.scenario_id, result.baseline_name)].append(result)
        scenario_meta.setdefault(
            result.scenario_id,
            (result.scenario_title, result.why_it_matters, tuple(result.capability_tags)),
        )

    for answer_result in run_result.answer_results:
        if (
            answer_result.baseline_name not in run_result.primary_baselines
            or not answer_result.available
        ):
            continue
        answer_grouped[(answer_result.scenario_id, answer_result.baseline_name)].append(
            answer_result,
        )

    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for (scenario_id, baseline_name), results in grouped.items():
        answer_results = answer_grouped.get((scenario_id, baseline_name), [])
        probe_count = sum(len(result.probe_results) for result in results)
        by_scenario[scenario_id].append(
            {
                "baseline_name": baseline_name,
                "evidence_pass_rate": sum(1 for result in results if result.passed) / len(results),
                "avg_tokens": (
                    sum(
                        sum(probe.tokens_surfaced for probe in result.probe_results)
                        for result in results
                    )
                    / max(1, probe_count)
                ),
                "avg_latency": (
                    sum(
                        sum(probe.latency_ms for probe in result.probe_results)
                        for result in results
                    )
                    / max(1, probe_count)
                ),
                "answer_pass_rate": (
                    sum(1 for result in answer_results if result.passed) / len(answer_results)
                    if answer_results
                    else None
                ),
                "answer_score": (
                    sum(result.score for result in answer_results) / len(answer_results)
                    if answer_results
                    else None
                ),
            }
        )

    rows: list[dict] = []
    for scenario_id, entries in by_scenario.items():
        title, why, tags = scenario_meta[scenario_id]
        top_pass_rate = max(entry["evidence_pass_rate"] for entry in entries)
        winner_name: str | None = None
        if top_pass_rate > 0.0:
            winner = sorted(
                entries,
                key=lambda entry: (
                    entry["evidence_pass_rate"],
                    0.0 if entry["answer_score"] is None else entry["answer_score"],
                    -entry["avg_tokens"],
                    -entry["avg_latency"],
                ),
                reverse=True,
            )[0]
            winner_name = winner["baseline_name"]
            explanation = _winner_note(winner_name, set(tags), why)
        else:
            explanation = "No primary baseline passed this scenario."
        rows.append(
            {
                "scenario_id": scenario_id,
                "title": title,
                "why_it_matters": why,
                "tags": tags,
                "winner": winner_name,
                "explanation": explanation,
                "entries": sorted(entries, key=lambda entry: entry["baseline_name"]),
            }
        )
    return rows


def _winner_note(winner: str, tags: set[str], why: str) -> str:
    if winner == "engram_full":
        if "prospective" in tags:
            return (
                "Prospective retrieval surfaced the right intention"
                " from related entity activity."
            )
        if "temporal" in tags or "negation" in tags:
            return "Current-state memory won without leaking stale or negated facts."
        if "association" in tags or "graph" in tags:
            return "Associative retrieval connected lexically distant but linked entities."
        if "compression" in tags:
            return "Structured memory preserved the right facts under the fixed budget."
        if "cue" in tags or "open_loop" in tags or "continuity" in tags:
            return "Latent memory stayed available without carrying raw history in prompt."
        if "meta" in tags:
            return "Canonical memory excluded system chatter and paraphrase drift."
    if winner == "context_summary":
        return "Rolling summaries kept enough durable state to stay competitive."
    if winner == "markdown_canonical":
        return "A deterministic latest-win notebook stayed strong on this structured query."
    if winner == "hybrid_rag_temporal":
        return "Temporal filtering kept hybrid retrieval competitive without a graph layer."
    if winner == "context_window":
        return "Recent context alone was enough for this local task."
    if winner == "markdown_memory":
        return "A lexical notebook was sufficient for this mostly text-match scenario."
    if winner == "vector_rag":
        return "Raw retrieval stayed competitive on this mostly retrieval-shaped query."
    return why


def _capability_tags(run_result: ShowcaseRunResult) -> list[str]:
    tags: set[str] = set()
    for summary in run_result.baseline_summaries:
        tags.update(summary.capability_pass_rates)
    return sorted(tags)


def _render_fairness(lines: list[str], run_result: ShowcaseRunResult) -> None:
    fairness = run_result.fairness_contract
    lines.extend(
        [
            "## Fairness Contract",
            "",
            f"- Track: `{fairness.track}`",
            f"- Strict fairness: `{fairness.strict_fairness}`",
            f"- Transcript invariant: `{fairness.transcript_invariant}`",
            f"- Vector provider family: `{fairness.vector_provider_family}`",
            f"- Answer model: `{fairness.answer_model or 'not configured'}`",
            f"- Answer provider: `{fairness.answer_provider or 'not configured'}`",
            "",
            "| Scenario | Top-k | Evidence Tokens | Answer Tokens |",
            "|---|---:|---:|---:|",
        ]
    )
    for scenario_id, budget in fairness.scenario_budgets.items():
        lines.append(
            f"| {scenario_id} | {budget['retrieval_limit']}"
            f" | {budget['evidence_max_tokens']}"
            f" | {budget['answer_budget_tokens']} |"
        )


def _render_headline_measured(lines: list[str], run_result: ShowcaseRunResult) -> None:
    summaries = _summary_by_name(run_result)
    answer_summaries = _answer_summary_by_name(run_result)
    lines.extend(
        [
            "",
            "## Headline Measured Competitors",
            "",
            "| Baseline | Available | Scenario Pass | False Recall"
            " | Temporal | Negation | Open Loop | Prospective"
            " | Answer Pass | Answer Score | p50 ms | p95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline_name in run_result.headline_baselines:
        summary = summaries.get(baseline_name)
        if summary is None:
            continue
        answer_summary = answer_summaries.get(baseline_name)
        availability = "yes" if summary.available else f"no ({summary.availability_reason})"
        lines.append(
            "| "
            + " | ".join(
                [
                    _display_name(baseline_name),
                    availability,
                    _fmt(summary.scenario_pass_rate),
                    _fmt(summary.false_recall_rate),
                    _fmt(summary.temporal_correctness),
                    _fmt(summary.negation_correctness),
                    _fmt(summary.open_loop_recovery),
                    _fmt(summary.prospective_trigger_rate),
                    _fmt(None if answer_summary is None else answer_summary.answer_pass_rate),
                    _fmt(None if answer_summary is None else answer_summary.average_score),
                    _fmt(summary.latency_p50_ms),
                    _fmt(summary.latency_p95_ms),
                ]
            )
            + " |"
        )


def _render_control_baselines(lines: list[str], run_result: ShowcaseRunResult) -> None:
    if not run_result.control_baselines:
        return
    summaries = _summary_by_name(run_result)
    lines.extend(
        [
            "",
            "## Measured Control Baselines",
            "",
            "| Baseline | Available | Scenario Pass | False Recall "
            "| Temporal | Negation | Open Loop | Prospective | p50 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline_name in run_result.control_baselines:
        summary = summaries.get(baseline_name)
        if summary is None:
            continue
        availability = "yes" if summary.available else f"no ({summary.availability_reason})"
        lines.append(
            "| "
            + " | ".join(
                [
                    _display_name(baseline_name),
                    availability,
                    _fmt(summary.scenario_pass_rate),
                    _fmt(summary.false_recall_rate),
                    _fmt(summary.temporal_correctness),
                    _fmt(summary.negation_correctness),
                    _fmt(summary.open_loop_recovery),
                    _fmt(summary.prospective_trigger_rate),
                    _fmt(summary.latency_p50_ms),
                ]
            )
            + " |"
        )


def _render_capability_scorecard(lines: list[str], run_result: ShowcaseRunResult) -> None:
    summaries = _summary_by_name(run_result)
    tags = _capability_tags(run_result)
    if not tags:
        return
    lines.extend(
        [
            "",
            "## Capability Scorecard",
            "",
            "| Baseline | " + " | ".join(tag.replace("_", " ").title() for tag in tags) + " |",
            "|---|" + "|".join("---:" for _ in tags) + "|",
        ]
    )
    for baseline_name in run_result.primary_baselines:
        summary = summaries.get(baseline_name)
        if summary is None:
            continue
        lines.append(
            "| "
            + " | ".join(
                [_display_name(baseline_name)]
                + [_fmt(summary.capability_pass_rates.get(tag, 0.0)) for tag in tags]
            )
            + " |"
        )


def _render_scenario_rows(lines: list[str], run_result: ShowcaseRunResult) -> None:
    rows = _aggregate_primary_scenarios(run_result)
    lines.extend(
        [
            "",
            "## Scenario Winners",
            "",
            "| Scenario | Winner | Why |",
            "|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['title']} | "
            f"{_display_name(row['winner']) if row['winner'] else 'No Baseline Passed'} | "
            f"{row['explanation']} |"
        )

    lines.extend(
        [
            "",
            "## Scenario Matrix",
            "",
        ]
    )
    for row in rows:
        lines.append(f"### {row['title']}")
        lines.append("")
        lines.append(f"- Why it matters: {row['why_it_matters']}")
        lines.append("")
        lines.append(
            "| Baseline | Evidence Pass | Answer Pass"
            " | Answer Score | Avg Tokens | Avg Latency ms |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for entry in row["entries"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _display_name(entry["baseline_name"]),
                        _fmt(entry["evidence_pass_rate"]),
                        _fmt(entry["answer_pass_rate"]),
                        _fmt(entry["answer_score"]),
                        _fmt(entry["avg_tokens"]),
                        _fmt(entry["avg_latency"]),
                    ]
                )
                + " |"
            )
        lines.append("")


def _render_cost_and_errors(lines: list[str], run_result: ShowcaseRunResult) -> None:
    summaries = _summary_by_name(run_result)
    lines.extend(
        [
            "## Cost And Error Summary",
            "",
            "| Baseline | False Recall | Token Efficiency | Tokens / Success | p50 ms | p95 ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline_name in run_result.primary_baselines:
        summary = summaries.get(baseline_name)
        if summary is None:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _display_name(baseline_name),
                    _fmt(summary.false_recall_rate),
                    _fmt(summary.token_efficiency),
                    _fmt(summary.tokens_per_passed_scenario),
                    _fmt(summary.latency_p50_ms),
                    _fmt(summary.latency_p95_ms),
                ]
            )
            + " |"
        )


def _render_ablations(lines: list[str], run_result: ShowcaseRunResult) -> None:
    summaries = _summary_by_name(run_result)
    if not run_result.ablation_baselines:
        return
    lines.extend(
        [
            "",
            "## Engram Ablations",
            "",
            "| Ablation | Available | Scenario Pass | False Recall | Cue/Planning Signal |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for baseline_name in run_result.ablation_baselines:
        summary = summaries.get(baseline_name)
        if summary is None:
            continue
        availability = "yes" if summary.available else f"no ({summary.availability_reason})"
        lines.append(
            "| "
            + " | ".join(
                [
                    _display_name(baseline_name),
                    availability,
                    _fmt(summary.scenario_pass_rate),
                    _fmt(summary.false_recall_rate),
                    _fmt(summary.prospective_trigger_rate),
                ]
            )
            + " |"
        )


def _render_spec_only(lines: list[str], run_result: ShowcaseRunResult) -> None:
    if not run_result.spec_only_baselines:
        return
    lines.extend(
        [
            "",
            "## Spec-Only Comparison Targets",
            "",
            "| System | Technology | Archetype | Why Tracked | Current Limitation |",
            "|---|---|---|---|---|",
        ]
    )
    for baseline_id in run_result.spec_only_baselines:
        entry = run_result.baseline_catalog.get(baseline_id)
        if entry is None:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    entry.display_name,
                    entry.external_technology_label or "-",
                    entry.archetype or "-",
                    entry.why_included or "-",
                    entry.known_limitations or "-",
                ]
            )
            + " |"
        )


def _render_appendix(lines: list[str], run_result: ShowcaseRunResult) -> None:
    summaries = _summary_by_name(run_result)
    if run_result.appendix_baselines:
        lines.extend(
            [
                "",
                "## Appendix Baselines",
                "",
                "| Baseline | Available | Scenario Pass | False Recall | p50 ms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for baseline_name in run_result.appendix_baselines:
            summary = summaries.get(baseline_name)
            if summary is None:
                continue
            availability = "yes" if summary.available else f"no ({summary.availability_reason})"
            lines.append(
                "| "
                + " | ".join(
                    [
                        _display_name(baseline_name),
                        availability,
                        _fmt(summary.scenario_pass_rate),
                        _fmt(summary.false_recall_rate),
                        _fmt(summary.latency_p50_ms),
                    ]
                )
                + " |"
            )

    if run_result.external_track_results:
        lines.extend(
            [
                "",
                "## External And Supporting Tracks",
                "",
                "| Track | Available | Executed | Summary | Recommended Command |",
                "|---|---:|---:|---|---|",
            ]
        )
        for result in run_result.external_track_results:
            summary = ", ".join(
                f"{key}={value}" for key, value in sorted(result.summary_metrics.items())
            )
            lines.append(
                f"| {result.name} | "
                f"{'yes' if result.available else f'no ({result.availability_reason})'} | "
                f"{'yes' if result.executed else 'no'} | "
                f"{summary or '-'} | "
                f"`{result.recommended_command or ''}` |"
            )


def _render_takeaways(lines: list[str], run_result: ShowcaseRunResult) -> None:
    summaries = _summary_by_name(run_result)
    engram = summaries.get("engram_full")
    competitors = [
        summaries[baseline_name]
        for baseline_name in run_result.primary_baselines
        if baseline_name != "engram_full" and baseline_name in summaries
    ]
    if engram is None:
        return

    lines.extend(
        [
            "",
            "## Where Engram Wins",
            "",
        ]
    )
    if competitors:
        higher_pass = [
            _display_name(summary.baseline_name)
            for summary in competitors
            if engram.scenario_pass_rate > summary.scenario_pass_rate
        ]
        lower_false = [
            _display_name(summary.baseline_name)
            for summary in competitors
            if engram.false_recall_rate <= summary.false_recall_rate
        ]
        lines.append(
            f"- Headline showcase pass rate: `{_fmt(engram.scenario_pass_rate)}` for Engram Full."
        )
        lines.append(
            "- Lower or equal false recall versus: "
            f"{', '.join(lower_false) if lower_false else 'none'}."
        )
        lines.append(
            "- Higher scenario pass rate versus: "
            f"{', '.join(higher_pass) if higher_pass else 'none'}."
        )

    primary_rows = _aggregate_primary_scenarios(run_result)
    scenario_wins = [row["title"] for row in primary_rows if row["winner"] == "engram_full"]
    if scenario_wins:
        lines.append(f"- Primary scenario wins: {', '.join(scenario_wins)}.")

    lines.extend(
        [
            "",
            "## Where Competitors Stay Competitive",
            "",
        ]
    )
    non_engram_wins = [
        f"{row['title']} ({_display_name(row['winner'])})"
        for row in primary_rows
        if row["winner"] and row["winner"] != "engram_full"
    ]
    if non_engram_wins:
        for win in non_engram_wins:
            lines.append(f"- {win}.")
    else:
        lines.append("- No primary competitor won a showcase scenario in this run.")


def render_markdown_report(run_result: ShowcaseRunResult) -> str:
    """Render the benchmark result as a decision-ready Markdown report."""
    lines = [
        "# Engram Benchmark Suite",
        "",
        f"- Track: `{run_result.track}`",
        f"- Mode: `{run_result.mode}`",
        f"- Seeds: `{', '.join(str(seed) for seed in run_result.seeds)}`",
        f"- Generated: `{run_result.generated_at}`",
        f"- Output: `{run_result.output_dir}`",
    ]

    _render_fairness(lines, run_result)

    if run_result.primary_baselines and run_result.baseline_summaries:
        _render_headline_measured(lines, run_result)
        _render_control_baselines(lines, run_result)
        _render_capability_scorecard(lines, run_result)
        _render_scenario_rows(lines, run_result)
        _render_cost_and_errors(lines, run_result)
        _render_ablations(lines, run_result)
        _render_spec_only(lines, run_result)
        _render_appendix(lines, run_result)
        _render_takeaways(lines, run_result)
    else:
        lines.extend(
            [
                "",
                "## Track Status",
                "",
            ]
        )
        for summary in run_result.track_summaries:
            lines.append(
                f"- `{summary.track}`: executed={summary.executed}, "
                f"available={summary.available}, "
                f"reason={summary.availability_reason or 'n/a'}"
            )

    if run_result.supporting_artifacts:
        lines.extend(
            [
                "",
                "## Supporting Artifacts",
                "",
            ]
        )
        for name, path in sorted(run_result.supporting_artifacts.items()):
            lines.append(f"- `{name}`: `{path}`")

    if run_result.readme_snippet:
        lines.extend(
            [
                "",
                "## README Snippet",
                "",
                run_result.readme_snippet,
            ]
        )

    return "\n".join(lines) + "\n"
