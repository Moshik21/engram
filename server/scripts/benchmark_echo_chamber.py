#!/usr/bin/env python3
"""Echo chamber benchmark CLI.

Measures whether activation creates filter bubbles by running sequential
queries and tracking coverage, Gini coefficient, top-10 stability, and
surfaced-vs-used recall behavior.

Usage:
    cd server && uv run python scripts/benchmark_echo_chamber.py --queries 200
    cd server && uv run python scripts/benchmark_echo_chamber.py --queries 200 --json results.json
    cd server && uv run python scripts/benchmark_echo_chamber.py --ts-enabled
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from engram.benchmark.corpus import generate_corpus
from engram.benchmark.echo_chamber import run_echo_chamber
from engram.config import ActivationConfig
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


async def main(args: argparse.Namespace) -> None:
    """Run echo chamber benchmark."""
    start = time.perf_counter()

    # Build corpus
    corpus = generate_corpus(seed=42)
    print(f"Corpus: {len(corpus.entities)} entities, {len(corpus.relationships)} relationships")

    # Initialize stores (in-memory SQLite)
    graph_store = SQLiteGraphStore(":memory:")
    await graph_store.initialize()
    search_index = FTS5SearchIndex(":memory:")
    await search_index.initialize(db=graph_store._db)

    cfg = ActivationConfig(
        ts_enabled=args.ts_enabled,
        exploration_weight=0.05,
        rediscovery_weight=0.02,
    )
    activation_store = MemoryActivationStore(cfg)

    # Ingest corpus
    for entity in corpus.entities:
        await graph_store.create_entity(entity)
        await search_index.index_entity(entity)
    for rel in corpus.relationships:
        await graph_store.create_relationship(rel)

    # Build query pools from ground truth
    hot_queries = []
    diverse_queries = []
    for q in corpus.ground_truth:
        if q.category in ("semantic", "direct_lookup", "direct"):
            hot_queries.append(q.query_text)
        else:
            diverse_queries.append(q.query_text)
    if not hot_queries:
        hot_queries = [q.query_text for q in corpus.ground_truth[:5]]
    if not diverse_queries:
        diverse_queries = [q.query_text for q in corpus.ground_truth[5:]]

    corpus_entity_ids = [e.id for e in corpus.entities]
    benchmark_group_id = corpus.entities[0].group_id if corpus.entities else "benchmark"

    # Run benchmark
    result = await run_echo_chamber(
        hot_queries=hot_queries,
        diverse_queries=diverse_queries,
        corpus_entity_ids=corpus_entity_ids,
        graph_store=graph_store,
        activation_store=activation_store,
        search_index=search_index,
        cfg=cfg,
        group_id=benchmark_group_id,
        total_queries=args.queries,
        snapshot_interval=args.snapshot_interval,
    )

    elapsed = time.perf_counter() - start

    # Print results
    print(f"\n{'=' * 60}")
    print("Echo Chamber Benchmark Results")
    print(f"{'=' * 60}")
    print(f"Total queries:       {result.total_queries}")
    print(f"Thompson Sampling:   {'enabled' if result.ts_enabled else 'disabled'}")
    print(f"Final coverage:      {result.final_coverage:.1%}")
    print(f"Final Gini:          {result.final_gini:.3f}")
    print(f"Final top-10 Jaccard:{result.final_top10_jaccard:.3f}")
    print(f"Surfaced results:    {result.final_surfaced_count}")
    print(f"Used results:        {result.final_used_count}")
    print(f"Surfaced/used ratio: {result.final_surfaced_to_used_ratio:.3f}")
    print(f"Coverage target:     {'PASS' if result.pass_coverage else 'FAIL'} (> 40%)")
    print(f"Gini target:         {'PASS' if result.pass_gini else 'FAIL'} (< 0.70)")
    print(f"Elapsed:             {elapsed:.1f}s")

    if result.snapshots:
        print("\nSnapshots:")
        for s in result.snapshots:
            print(
                f"  Q{s.query_index:>4d}: "
                f"coverage={s.coverage:.1%} "
                f"gini={s.gini:.3f} "
                f"jaccard={s.top10_jaccard:.3f} "
                f"surfaced={s.surfaced_count} "
                f"used={s.used_count}"
            )

    # Save JSON if requested
    if args.json:
        output = {
            "total_queries": result.total_queries,
            "ts_enabled": result.ts_enabled,
            "final_coverage": result.final_coverage,
            "final_gini": result.final_gini,
            "final_top10_jaccard": result.final_top10_jaccard,
            "final_surfaced_count": result.final_surfaced_count,
            "final_used_count": result.final_used_count,
            "final_surfaced_to_used_ratio": result.final_surfaced_to_used_ratio,
            "pass_coverage": result.pass_coverage,
            "pass_gini": result.pass_gini,
            "elapsed_seconds": elapsed,
            "snapshots": [
                {
                    "query_index": s.query_index,
                    "coverage": s.coverage,
                    "gini": s.gini,
                    "top10_jaccard": s.top10_jaccard,
                    "surfaced_count": s.surfaced_count,
                    "used_count": s.used_count,
                    "surfaced_to_used_ratio": s.surfaced_to_used_ratio,
                    "top10_ids": s.top10_ids,
                }
                for s in result.snapshots
            ],
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json}")

    await graph_store.close()
    sys.exit(0 if (result.pass_coverage and result.pass_gini) else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo Chamber Benchmark")
    parser.add_argument(
        "--queries",
        type=int,
        default=200,
        help="Total number of queries to simulate",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=50,
        help="Take snapshot every N queries",
    )
    parser.add_argument(
        "--ts-enabled",
        action="store_true",
        help="Enable Thompson Sampling exploration",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    asyncio.run(main(parser.parse_args()))
