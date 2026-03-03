#!/usr/bin/env python3
"""Working memory benchmark for Engram retrieval.

Runs multi-query conversation scenarios to measure how working memory
bridges sequential queries. Each scenario executes 3 queries:
  Q1: about cluster A's hub entity
  Q2: about cluster B's hub entity
  Q3: a bridging question — should benefit from WM seeding

Compares bridge recall with and without working memory to compute WM lift.

Usage:
    python benchmark_working_memory.py [--seed 42] [--verbose] [--json path]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Ensure engram package is importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from engram.benchmark.corpus import ConversationScenario, CorpusGenerator, CorpusSpec
from engram.benchmark.methods import METHOD_FULL_ENGRAM, run_retrieval
from engram.config import ActivationConfig
from engram.retrieval.working_memory import WorkingMemoryBuffer
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


async def _run_scenario(
    scenario: ConversationScenario,
    graph_store: SQLiteGraphStore,
    activation_store: MemoryActivationStore,
    search_index: FTS5SearchIndex,
    now: float = 0.0,
    use_wm: bool = True,
    verbose: bool = False,
) -> dict:
    """Run a single conversation scenario and return bridge recall metrics."""
    retrieval_method = METHOD_FULL_ENGRAM

    wm_buffer = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0) if use_wm else None

    all_results: dict[int, list[str]] = {}

    for qi, query_text in enumerate(scenario.queries):
        results = await run_retrieval(
            query_text,
            "benchmark",
            graph_store,
            activation_store,
            search_index,
            retrieval_method,
            limit=10,
            now=now,
            working_memory=wm_buffer,
        )

        ranked_ids = [r.node_id for r in results]
        all_results[qi] = ranked_ids

        # Populate working memory with results (mirroring graph_manager.recall)
        if wm_buffer is not None:
            for r in results:
                wm_buffer.add(
                    item_id=r.node_id,
                    item_type=r.result_type,
                    score=r.score,
                    query=query_text,
                    now=now,
                )
            wm_buffer.add_query(query_text, now)

        if verbose:
            mode = "WM" if use_wm else "no-WM"
            print(f"  [{mode}] Q{qi + 1}: {query_text[:50]}... -> {len(results)} results")

    # Compute bridge recall for each query index with expected bridges
    bridge_recalls: dict[int, float] = {}
    for query_idx, expected_ids in scenario.expected_bridge.items():
        if query_idx not in all_results or not expected_ids:
            continue
        retrieved = set(all_results[query_idx])
        found = retrieved & expected_ids
        recall = len(found) / len(expected_ids) if expected_ids else 0.0
        bridge_recalls[query_idx] = recall

    avg_bridge_recall = (
        sum(bridge_recalls.values()) / len(bridge_recalls) if bridge_recalls else 0.0
    )

    return {
        "scenario": scenario.name,
        "bridge_recalls": bridge_recalls,
        "avg_bridge_recall": avg_bridge_recall,
        "num_queries": len(scenario.queries),
    }


async def run_benchmark(args: argparse.Namespace) -> dict:
    """Execute the working memory benchmark."""

    # 1. Generate corpus
    print(
        f"Generating corpus with seed={args.seed}, entities={getattr(args, 'entities', 1000)} ..."
    )
    corpus_gen = CorpusGenerator(
        seed=args.seed,
        total_entities=getattr(args, "entities", 1000),
    )
    corpus: CorpusSpec = corpus_gen.generate()

    scenarios = corpus.conversation_scenarios
    if not scenarios:
        print("ERROR: No conversation scenarios generated. Check corpus generator.")
        sys.exit(1)

    print(
        f"Corpus: {len(corpus.entities)} entities, "
        f"{len(corpus.relationships)} relationships, "
        f"{len(corpus.episodes)} episodes, "
        f"{len(scenarios)} conversation scenarios"
    )

    # 2. Create temp stores
    tmp_dir = tempfile.mkdtemp(prefix="engram_wm_bench_")
    db_path = str(Path(tmp_dir) / "bench.db")

    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()

    activation_store = MemoryActivationStore(cfg=ActivationConfig())

    search_index = FTS5SearchIndex(db_path)
    await search_index.initialize(db=graph_store._db)

    # 3. Load corpus
    print("Loading corpus into stores ...")
    load_elapsed = await corpus_gen.load(
        corpus,
        graph_store,
        activation_store,
        search_index,
    )
    print(f"Loaded in {load_elapsed:.2f}s")

    benchmark_now = corpus.metadata.get("generated_at", time.time())

    # 4. Run each scenario with and without WM
    print(f"\nRunning {len(scenarios)} scenarios (with WM + without WM) ...\n")

    results_with_wm: list[dict] = []
    results_without_wm: list[dict] = []

    for si, scenario in enumerate(scenarios):
        if args.verbose:
            print(f"--- Scenario {si + 1}: {scenario.name} ---")

        result_wm = await _run_scenario(
            scenario,
            graph_store,
            activation_store,
            search_index,
            now=benchmark_now,
            use_wm=True,
            verbose=args.verbose,
        )
        results_with_wm.append(result_wm)

        result_no_wm = await _run_scenario(
            scenario,
            graph_store,
            activation_store,
            search_index,
            now=benchmark_now,
            use_wm=False,
            verbose=args.verbose,
        )
        results_without_wm.append(result_no_wm)

        if args.verbose:
            lift = result_wm["avg_bridge_recall"] - result_no_wm["avg_bridge_recall"]
            print(
                f"  Bridge recall: "
                f"WM={result_wm['avg_bridge_recall']:.3f}  "
                f"no-WM={result_no_wm['avg_bridge_recall']:.3f}  "
                f"lift={lift:+.3f}"
            )
            print()

    # 5. Aggregate results
    avg_wm = (
        sum(r["avg_bridge_recall"] for r in results_with_wm) / len(results_with_wm)
        if results_with_wm
        else 0.0
    )
    avg_no_wm = (
        sum(r["avg_bridge_recall"] for r in results_without_wm) / len(results_without_wm)
        if results_without_wm
        else 0.0
    )
    wm_lift = avg_wm - avg_no_wm

    # 6. Print results
    _print_results(
        scenarios,
        results_with_wm,
        results_without_wm,
        avg_wm,
        avg_no_wm,
        wm_lift,
    )

    # 7. Build output dict
    output = {
        "seed": args.seed,
        "num_scenarios": len(scenarios),
        "aggregate": {
            "bridge_recall_with_wm": avg_wm,
            "bridge_recall_without_wm": avg_no_wm,
            "wm_lift": wm_lift,
        },
        "per_scenario": [
            {
                "name": r_wm["scenario"],
                "with_wm": r_wm["avg_bridge_recall"],
                "without_wm": r_no["avg_bridge_recall"],
                "lift": (r_wm["avg_bridge_recall"] - r_no["avg_bridge_recall"]),
            }
            for r_wm, r_no in zip(results_with_wm, results_without_wm)
        ],
    }

    # 8. Write JSON if requested
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, indent=2))
        print(f"\nJSON results written to {json_path}")

    # 9. Cleanup
    await graph_store.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return output


def _print_results(
    scenarios: list[ConversationScenario],
    results_with_wm: list[dict],
    results_without_wm: list[dict],
    avg_wm: float,
    avg_no_wm: float,
    wm_lift: float,
) -> None:
    """Print formatted results."""
    print("=" * 60)
    print("=== Working Memory Benchmark Results ===")
    print("=" * 60)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Bridge recall (with WM):    {avg_wm:.3f}")
    print(f"Bridge recall (without WM): {avg_no_wm:.3f}")
    print(f"WM lift:                    {wm_lift:+.3f}")
    print()

    # Per-scenario breakdown
    print("--- Per-Scenario Breakdown ---")
    header = f"{'Scenario':<40}| {'WM':>5} | {'noWM':>5} | {'Lift':>6}"
    sep = f"{'-' * 40}|{'-' * 7}|{'-' * 7}|{'-' * 8}"
    print(header)
    print(sep)
    for r_wm, r_no in zip(results_with_wm, results_without_wm):
        name = r_wm["scenario"][:39]
        lift = r_wm["avg_bridge_recall"] - r_no["avg_bridge_recall"]
        print(
            f"{name:<40}| {r_wm['avg_bridge_recall']:>5.3f} | "
            f"{r_no['avg_bridge_recall']:>5.3f} | {lift:>+6.3f}"
        )
    print()

    positive_lift = sum(
        1
        for r_wm, r_no in zip(results_with_wm, results_without_wm)
        if r_wm["avg_bridge_recall"] > r_no["avg_bridge_recall"]
    )
    print(f"Scenarios with positive WM lift: {positive_lift}/{len(scenarios)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Working memory benchmark for Engram retrieval",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for corpus generation (default: 42)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path for JSON output file (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-scenario results as they run",
    )
    parser.add_argument(
        "--entities",
        type=int,
        default=1000,
        help="Total entities in corpus (default: 1000). Try 5000, 10000, 50000.",
    )

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
