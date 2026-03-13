#!/usr/bin/env python3
"""LongMemEval benchmark CLI.

Runs Engram against the LongMemEval benchmark (ICLR 2025) for
industry-standard long-term memory evaluation.

Usage:
    # Download dataset
    cd server && uv run python scripts/benchmark_longmemeval.py download --variant oracle

    # Quick test (5 per type, narrow extraction, no embeddings)
    cd server && uv run python scripts/benchmark_longmemeval.py run \
        --dataset data/longmemeval/longmemeval_oracle.json \
        --n-per-type 5 --extraction narrow --embeddings none

    # Full oracle run with local embeddings
    cd server && uv run python scripts/benchmark_longmemeval.py run \
        --dataset data/longmemeval/longmemeval_oracle.json \
        --extraction narrow --embeddings local

    # Full S variant with consolidation
    cd server && uv run python scripts/benchmark_longmemeval.py run \
        --dataset data/longmemeval/longmemeval_s_cleaned.json \
        --extraction narrow --embeddings local --consolidation \
        --output results/longmemeval_s.json

    # Report from saved results
    cd server && uv run python scripts/benchmark_longmemeval.py report \
        --results results/longmemeval_s.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def cmd_download(args: argparse.Namespace) -> None:
    """Download LongMemEval dataset from HuggingFace."""
    from engram.benchmark.longmemeval.dataset import download_dataset

    path = await download_dataset(variant=args.variant, output_dir=args.output_dir)
    print(f"Dataset downloaded to: {path}")


async def cmd_run(args: argparse.Namespace) -> None:
    """Run the LongMemEval benchmark."""
    from engram.benchmark.longmemeval.report import format_report
    from engram.benchmark.longmemeval.runner import run_longmemeval

    result = await run_longmemeval(
        dataset_path=args.dataset,
        extraction_mode=args.extraction,
        embedding_provider=args.embeddings,
        consolidation=args.consolidation,
        reader_model=args.reader_model,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        top_k=args.top_k,
        max_instances=args.max_instances,
        n_per_type=args.n_per_type,
        question_types=args.types.split(",") if args.types else None,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        verbose=args.verbose,
    )

    # Print report
    report = format_report(result)
    print(report)

    if args.output:
        print(f"\nResults saved to: {args.output}")

        # Also save markdown report
        md_path = args.output.replace(".json", ".md")
        if md_path != args.output:
            with open(md_path, "w") as f:
                f.write(report)
            print(f"Report saved to: {md_path}")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate a report from saved results."""
    from engram.benchmark.longmemeval.adapter import AdapterStats
    from engram.benchmark.longmemeval.report import format_report
    from engram.benchmark.longmemeval.runner import InstanceResult, LongMemEvalResult, TypeMetrics

    with open(args.results) as f:
        data = json.load(f)

    # Reconstruct enough of the result object for reporting
    type_metrics = [
        TypeMetrics(
            question_type=tm["question_type"],
            count=tm["count"],
            correct=tm["correct"],
            accuracy=tm["accuracy"],
            avg_latency_ms=tm.get("avg_latency_ms", 0),
            avg_recall_at_5=tm.get("avg_recall_at_5", 0),
            avg_ndcg_at_5=tm.get("avg_ndcg_at_5", 0),
        )
        for tm in data.get("type_metrics", [])
    ]

    stats_data = data.get("adapter_stats", {})
    adapter_stats = AdapterStats(
        sessions_ingested=stats_data.get("sessions_ingested", 0),
        episodes_stored=stats_data.get("episodes_stored", 0),
        episodes_extracted=stats_data.get("episodes_extracted", 0),
        extraction_calls=stats_data.get("extraction_calls", 0),
        embedding_calls=stats_data.get("embedding_calls", 0),
        recall_calls=stats_data.get("recall_calls", 0),
        reader_calls=stats_data.get("reader_calls", 0),
        total_ingest_ms=stats_data.get("total_ingest_ms", 0),
        total_query_ms=stats_data.get("total_query_ms", 0),
    )

    instance_results = [
        InstanceResult(
            question_id=inst["question_id"],
            question_type=inst["question_type"],
            question=inst.get("question", ""),
            gold_answer=inst.get("gold_answer", ""),
            hypothesis=inst.get("hypothesis", ""),
            correct=inst["correct"],
            judge_raw="",
            evidence=[],
            evidence_scores=[],
            retrieved_session_ids=[],
            answer_session_ids=[],
            retrieval_metrics=inst.get("retrieval_metrics", {}),
            query_latency_ms=inst.get("query_latency_ms", 0),
            ingest_sessions=inst.get("ingest_sessions", 0),
            num_entities=inst.get("num_entities", 0),
            num_episodes=inst.get("num_episodes", 0),
        )
        for inst in data.get("instances", [])
    ]

    result = LongMemEvalResult(
        variant=data.get("variant", "unknown"),
        extraction_mode=data.get("extraction_mode", "unknown"),
        embedding_provider=data.get("embedding_provider", "unknown"),
        consolidation_used=data.get("consolidation_used", False),
        reader_model=data.get("reader_model", "unknown"),
        judge_model=data.get("judge_model", "unknown"),
        total_instances=data.get("total_instances", 0),
        total_correct=data.get("total_correct", 0),
        overall_accuracy=data.get("overall_accuracy", 0),
        category_accuracy=data.get("category_accuracy", 0),
        type_metrics=type_metrics,
        instance_results=instance_results,
        adapter_stats=adapter_stats,
        elapsed_seconds=data.get("elapsed_seconds", 0),
    )

    print(format_report(result))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="LongMemEval Benchmark for Engram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    dl = subparsers.add_parser("download", help="Download LongMemEval dataset")
    dl.add_argument(
        "--variant",
        choices=["oracle", "s", "m"],
        default="oracle",
        help="Dataset variant (default: oracle)",
    )
    dl.add_argument(
        "--output-dir",
        default="data/longmemeval",
        help="Output directory (default: data/longmemeval)",
    )

    # Run command
    run = subparsers.add_parser("run", help="Run the benchmark")
    run.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset JSON file",
    )
    run.add_argument(
        "--extraction",
        choices=["none", "narrow", "full", "auto"],
        default="narrow",
        help="Extraction mode (default: narrow)",
    )
    run.add_argument(
        "--embeddings",
        choices=["none", "local", "voyage", "auto"],
        default="local",
        help="Embedding provider (default: local)",
    )
    run.add_argument(
        "--consolidation",
        action="store_true",
        help="Run consolidation after ingestion",
    )
    run.add_argument(
        "--reader-model",
        default="claude-haiku-4-5-20251001",
        help="Model for answer composition",
    )
    run.add_argument(
        "--judge-model",
        default="claude-haiku-4-5-20251001",
        help="Model for answer evaluation",
    )
    run.add_argument(
        "--judge-provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Judge provider (default: anthropic)",
    )
    run.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    run.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Max total instances",
    )
    run.add_argument(
        "--n-per-type",
        type=int,
        default=None,
        help="Stratified sampling: N instances per question type",
    )
    run.add_argument(
        "--types",
        default=None,
        help="Comma-separated question types to include",
    )
    run.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file",
    )
    run.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file for resume support",
    )
    run.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Report command
    rpt = subparsers.add_parser("report", help="Generate report from saved results")
    rpt.add_argument(
        "--results",
        required=True,
        help="Path to results JSON file",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "download":
        asyncio.run(cmd_download(args))
    elif args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "report":
        cmd_report(args)
