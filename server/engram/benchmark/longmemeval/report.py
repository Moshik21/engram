"""LongMemEval benchmark report generation."""

from __future__ import annotations

from engram.benchmark.longmemeval.runner import LongMemEvalResult

# Published baselines for comparison (LongMemEval_S with GPT-4o reader)
PUBLISHED_BASELINES: dict[str, dict[str, float]] = {
    "Observational Memory (gpt-5-mini)": {"accuracy": 94.87},
    "Observational Memory (gpt-4o)": {"accuracy": 84.23},
    "EmergenceMem Internal": {"accuracy": 86.0},
    "EmergenceMem Simple": {"accuracy": 82.4},
    "Oracle GPT-4o": {"accuracy": 82.4},
    "Supermemory": {"accuracy": 81.6},
    "TiMem (GPT-4o-mini)": {"accuracy": 76.88},
    "Zep/Graphiti": {"accuracy": 71.2},
    "Full-context GPT-4o": {"accuracy": 60.2},
    "Naive RAG": {"accuracy": 52.0},
    "Best guess (no context)": {"accuracy": 18.8},
}


def format_report(result: LongMemEvalResult) -> str:
    """Generate a markdown report from benchmark results."""
    lines = [
        f"# LongMemEval Benchmark Report ({result.variant.upper()})",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Dataset variant | {result.variant} |",
        f"| Extraction mode | {result.extraction_mode} |",
        f"| Embedding provider | {result.embedding_provider} |",
        f"| Reranker | {result.reranker_provider} |",
        f"| Graph in ranking | {'yes' if result.use_graph else 'no'} |",
        f"| Consolidation | {'yes' if result.consolidation_used else 'no'} |",
        "| Evaluation method | embedding containment (no LLM reader) |",
        f"| Avg containment | {result.avg_containment * 100:.1f}% |",
        f"| Total instances | {result.total_instances} |",
        f"| Elapsed time | {result.elapsed_seconds:.0f}s |",
        "",
        "## Overall Results",
        "",
        f"**Overall accuracy: {result.overall_accuracy * 100:.1f}%** "
        f"({result.total_correct}/{result.total_instances})",
        "",
        f"**Category accuracy (official): {result.category_accuracy * 100:.1f}%** "
        f"(unweighted avg across question types)",
        "",
        "## Per-Type Breakdown",
        "",
        "| Type | Count | Correct | Acc | Contain | Latency | R@1 | R@5 | NDCG@5 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for tm in result.type_metrics:
        containment_pct = tm.avg_containment * 100 if hasattr(tm, "avg_containment") else 0.0
        recall1_pct = tm.avg_recall_at_1 * 100 if hasattr(tm, "avg_recall_at_1") else 0.0
        lines.append(
            f"| {tm.question_type} | {tm.count} | {tm.correct} | "
            f"{tm.accuracy * 100:.1f}% | {containment_pct:.1f}% | {tm.avg_latency_ms:.0f}ms | "
            f"{recall1_pct:.1f}% | {tm.avg_recall_at_5 * 100:.1f}% | "
            f"{tm.avg_ndcg_at_5 * 100:.1f}% |"
        )

    # Engram's own result, shown with provenance. Deliberately NOT ranked into
    # the published numbers below: those use a different LongMemEval variant
    # (LongMemEval_S), reader (GPT-4o), and sample, so a head-to-head leaderboard
    # would be misleading.
    lines.extend(
        [
            "",
            "## Engram Result",
            "",
            "| Config | Accuracy |",
            "|---|---|",
            (
                f"| Engram ({result.extraction_mode}/{result.embedding_provider}, "
                f"{result.total_instances}q) | {result.category_accuracy * 100:.1f}% |"
            ),
            "",
            "## Reference Numbers (not directly comparable)",
            "",
            "> Published baselines below use the LongMemEval_S variant with a GPT-4o "
            "reader and their own samples. They are listed for context only; "
            "differences in variant, reader, and sample size make a direct ranking "
            "against Engram's number unsound.",
            "",
            "| System | Accuracy |",
            "|---|---|",
        ]
    )
    for name, info in sorted(
        PUBLISHED_BASELINES.items(), key=lambda x: x[1]["accuracy"], reverse=True
    ):
        lines.append(f"| {name} | {info['accuracy']:.1f}% |")

    lines.extend(
        [
            "",
            "## Adapter Statistics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Sessions ingested | {result.adapter_stats.sessions_ingested} |",
            f"| Episodes stored | {result.adapter_stats.episodes_stored} |",
            f"| Episodes extracted | {result.adapter_stats.episodes_extracted} |",
            f"| Extraction calls | {result.adapter_stats.extraction_calls} |",
            f"| Embedding calls | {result.adapter_stats.embedding_calls} |",
            f"| Recall calls | {result.adapter_stats.recall_calls} |",
            f"| Total ingest time | {result.adapter_stats.total_ingest_ms / 1000:.1f}s |",
            f"| Total query time | {result.adapter_stats.total_query_ms / 1000:.1f}s |",
        ]
    )

    # Error analysis: which types are we worst at?
    if result.type_metrics:
        worst = min(result.type_metrics, key=lambda tm: tm.accuracy)
        best = max(result.type_metrics, key=lambda tm: tm.accuracy)
        lines.extend(
            [
                "",
                "## Error Analysis",
                "",
                f"- **Best category**: {best.question_type} ({best.accuracy * 100:.1f}%)",
                f"- **Worst category**: {worst.question_type} ({worst.accuracy * 100:.1f}%)",
            ]
        )

        # List some incorrect predictions
        incorrect = [r for r in result.instance_results if not r.correct]
        if incorrect:
            lines.extend(
                [
                    "",
                    f"### Sample Errors ({min(len(incorrect), 5)} of {len(incorrect)})",
                    "",
                ]
            )
            for r in incorrect[:5]:
                lines.extend(
                    [
                        f"**{r.question_id}** ({r.question_type})",
                        f"- Q: {r.question}",
                        f"- Gold: {r.gold_answer}",
                        f"- Predicted: {r.hypothesis[:200]}",
                        f"- Evidence retrieved: {len(r.evidence)} items, "
                        f"{r.num_entities} entities, {r.num_episodes} episodes",
                        "",
                    ]
                )

    return "\n".join(lines)
