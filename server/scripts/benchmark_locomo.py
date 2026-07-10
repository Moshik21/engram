#!/usr/bin/env python3
"""LoCoMo benchmark CLI.

Runs Engram against the LoCoMo dataset for industry-standard memory
evaluation. Requires a LoCoMo JSON dataset file.

Usage:
    cd server && uv run python scripts/benchmark_locomo.py \
        --dataset-path data/locomo.json --max-conversations 50
    cd server && uv run python scripts/benchmark_locomo.py \
        --dataset-path data/locomo.json --json results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json

from engram.benchmark.locomo.runner import run_locomo


async def main(args: argparse.Namespace) -> None:
    """Run LoCoMo benchmark through Engram's real ingest + recall pipeline."""
    result = await run_locomo(
        dataset_path=args.dataset_path,
        max_conversations=args.max_conversations,
        max_questions=args.max_questions,
        limit=args.limit,
        reader=args.reader,
        judge=args.judge,
        reader_model=args.reader_model,
        use_graph=args.use_graph,
        extraction_mode=args.extraction,
    )

    primary = "accuracy" if result.overall_llm_accuracy is not None else "em"
    print(f"\n{'=' * 60}")
    print("LoCoMo Benchmark Results")
    print(f"{'=' * 60}")
    print(f"Conversations:  {result.total_conversations}")
    print(f"Total probes:   {result.total_probes}")
    print(
        f"Config:         reader={result.reader_mode} judge={result.judge_mode} "
        f"graph={result.use_graph} extraction={args.extraction}"
    )
    if result.overall_llm_accuracy is not None:
        print(f"LLM-judge acc:  {result.overall_llm_accuracy:.3f}")
    print(f"Overall F1:     {result.overall_f1:.3f}")
    print(f"Overall EM:     {result.overall_em:.3f}")
    print(f"Elapsed:        {result.elapsed_seconds:.1f}s")

    if result.category_scores:
        print("\nPer-category:")
        for cat, scores in sorted(result.category_scores.items()):
            pm = scores.get("accuracy", scores.get("em", 0.0))
            print(f"  {cat:14s}: {primary}={pm:.3f} F1={scores['f1']:.3f} (n={scores['count']})")

    if args.json:
        output = {
            "total_conversations": result.total_conversations,
            "total_probes": result.total_probes,
            "reader_mode": result.reader_mode,
            "judge_mode": result.judge_mode,
            "reader_model": result.reader_model,
            "use_graph": result.use_graph,
            "overall_llm_accuracy": result.overall_llm_accuracy,
            "overall_em": result.overall_em,
            "overall_f1": result.overall_f1,
            "elapsed_seconds": result.elapsed_seconds,
            "category_scores": result.category_scores,
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to LoCoMo JSON dataset",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to process",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap questions per conversation (for light slices)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Recall top-k per question")
    parser.add_argument(
        "--reader",
        choices=["none", "llm"],
        default="none",
        help="Answer generation: none (join evidence) or llm (generate from evidence)",
    )
    parser.add_argument(
        "--judge",
        choices=["f1", "llm"],
        default="f1",
        help="Grading: f1 (token F1/EM) or llm (grade generated answer vs gold)",
    )
    parser.add_argument(
        "--reader-model",
        default="claude-sonnet-4-6",
        help="Claude model for the LLM reader/judge",
    )
    parser.add_argument(
        "--extraction",
        choices=["none", "narrow", "auto"],
        default="narrow",
        help="Entity extraction mode for ingest (default: narrow, zero-LLM)",
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Include the knowledge graph in recall (default: episode-vector only)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    asyncio.run(main(parser.parse_args()))
