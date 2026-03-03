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
from engram.config import ActivationConfig
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


async def main(args: argparse.Namespace) -> None:
    """Run LoCoMo benchmark."""
    # Initialize stores (in-memory SQLite)
    graph_store = SQLiteGraphStore(":memory:")
    await graph_store.initialize()
    search_index = FTS5SearchIndex(":memory:")
    await search_index.initialize(db=graph_store._db)

    cfg = ActivationConfig()
    activation_store = MemoryActivationStore(cfg)

    # Run benchmark
    result = await run_locomo(
        dataset_path=args.dataset_path,
        graph_store=graph_store,
        activation_store=activation_store,
        search_index=search_index,
        cfg=cfg,
        max_conversations=args.max_conversations,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("LoCoMo Benchmark Results")
    print(f"{'=' * 60}")
    print(f"Conversations:  {result.total_conversations}")
    print(f"Total probes:   {result.total_probes}")
    print(f"Overall EM:     {result.overall_em:.3f}")
    print(f"Overall F1:     {result.overall_f1:.3f}")
    print(f"Elapsed:        {result.elapsed_seconds:.1f}s")

    if result.category_scores:
        print("\nPer-category:")
        for cat, scores in sorted(result.category_scores.items()):
            print(f"  {cat:20s}: EM={scores['em']:.3f} F1={scores['f1']:.3f} (n={scores['count']})")

    # Save JSON if requested
    if args.json:
        output = {
            "total_conversations": result.total_conversations,
            "total_probes": result.total_probes,
            "overall_em": result.overall_em,
            "overall_f1": result.overall_f1,
            "elapsed_seconds": result.elapsed_seconds,
            "category_scores": result.category_scores,
            "conversations": [
                {
                    "id": cr.conversation_id,
                    "turns": cr.num_turns,
                    "probes": cr.num_probes,
                    "avg_em": cr.avg_em,
                    "avg_f1": cr.avg_f1,
                }
                for cr in result.conversation_results
            ],
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json}")

    await graph_store.close()


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
        "--json",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    asyncio.run(main(parser.parse_args()))
