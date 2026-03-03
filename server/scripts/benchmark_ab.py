#!/usr/bin/env python3
"""A/B benchmark CLI runner for Engram retrieval methods.

Generates a deterministic corpus, loads it into temp stores, runs each
retrieval method against ground-truth queries, and reports precision,
recall, MRR, nDCG, latency, per-category breakdowns, and bootstrap
confidence intervals for pairwise comparisons.

Usage:
    python benchmark_ab.py [--seed 42] [--methods ...]
        [--json path] [--verbose] [--bootstrap-n 1000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

# Ensure engram package is importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv

# Load .env from server/ directory
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from engram.activation.community import CommunityStore
from engram.benchmark.corpus import CorpusGenerator, CorpusSpec
from engram.benchmark.methods import ALL_METHODS, RetrievalMethod, run_retrieval
from engram.benchmark.metrics import (
    bootstrap_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from engram.config import ActivationConfig
from engram.embeddings.provider import VoyageProvider
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.hybrid_search import HybridSearchIndex
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.storage.sqlite.vectors import SQLiteVectorStore

# ---------------------------------------------------------------------------
# Latency percentile helpers (no numpy)
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Return the p-th percentile from a pre-sorted list (0 <= p <= 100)."""
    if not sorted_vals:
        return 0.0
    k = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


async def run_benchmark(args: argparse.Namespace) -> dict:
    """Execute the full A/B benchmark and return the results dict."""

    # 1. Generate corpus
    print(
        f"Generating corpus with seed={args.seed}, "
        f"entities={args.entities} ..."
    )
    corpus_gen = CorpusGenerator(seed=args.seed, total_entities=args.entities)
    corpus: CorpusSpec = corpus_gen.generate()

    n_entities = len(corpus.entities)
    n_relationships = len(corpus.relationships)
    n_access_events = len(corpus.access_events)
    n_queries = len(corpus.ground_truth)
    n_episodes = len(corpus.episodes)

    print(
        f"Corpus: {n_entities} entities, {n_relationships} relationships, "
        f"{n_access_events} access events, {n_episodes} episodes, "
        f"{n_queries} ground-truth queries"
    )

    # 2. Create temp stores
    tmp_dir = tempfile.mkdtemp(prefix="engram_bench_")
    db_path = str(Path(tmp_dir) / "bench.db")

    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()

    activation_store = MemoryActivationStore(cfg=ActivationConfig())

    fts_index = FTS5SearchIndex(db_path)
    await fts_index.initialize(db=graph_store._db)

    # Wire embeddings if requested
    if args.embeddings:
        api_key = os.environ.get("VOYAGE_API_KEY", "")
        if not api_key:
            print("ERROR: --embeddings requires VOYAGE_API_KEY env var")
            sys.exit(1)
        print("Using Voyage AI embeddings (voyage-3-lite, 512d)")
        provider = VoyageProvider(api_key=api_key)
        vector_store = SQLiteVectorStore(db_path)
        await vector_store.initialize(db=graph_store._db)
        search_index = HybridSearchIndex(
            fts=fts_index,
            vector_store=vector_store,
            provider=provider,
            fts_weight=0.3,
            vec_weight=0.7,
        )
    else:
        search_index = fts_index

    # 3. Load corpus into stores
    structure_aware = getattr(args, 'structure_aware', False)
    bench_cfg = ActivationConfig()
    if structure_aware:
        print("Loading corpus into stores (structure-aware indexing) ...")
    else:
        print("Loading corpus into stores ...")
    load_elapsed = await corpus_gen.load(
        corpus, graph_store, activation_store, search_index,
        structure_aware=structure_aware,
        cfg=bench_cfg,
    )
    print(f"Loaded in {load_elapsed:.2f}s")

    # 3.5. Build predicate embedding cache (if embeddings enabled)
    predicate_cache = None
    if args.embeddings:
        from engram.activation.context_gate import PredicateEmbeddingCache

        predicate_cache = PredicateEmbeddingCache()
        await predicate_cache.initialize(bench_cfg, provider)
        n_pred = len(predicate_cache.get_embeddings())
        print(f"Predicate embedding cache: {n_pred} predicates embedded")

    # 3.6. Build community store from corpus clusters
    community_store = CommunityStore(seed=args.seed)
    cluster_assignments: dict[str, str] = {}
    for cluster_info in corpus.metadata.get("clusters", []):
        for member_id in cluster_info["members"]:
            cluster_assignments[member_id] = cluster_info["name"]
    if cluster_assignments:
        community_store.set_assignments("benchmark", cluster_assignments)
        print(f"Community store: {len(cluster_assignments)} entities in "
              f"{len(corpus.metadata.get('clusters', []))} clusters")

    # 4. Filter methods
    method_names = {m.name for m in ALL_METHODS}
    selected_methods: list[RetrievalMethod] = []
    if args.methods:
        for name in args.methods:
            matched = [m for m in ALL_METHODS if m.name == name]
            if not matched:
                print(f"WARNING: Unknown method '{name}'. Available: {sorted(method_names)}")
            else:
                selected_methods.append(matched[0])
        if not selected_methods:
            print("ERROR: No valid methods selected.")
            sys.exit(1)
    else:
        selected_methods = list(ALL_METHODS)

    corpus.metadata["search_mode"] = (
        "Voyage AI + FTS5 hybrid" if args.embeddings else "FTS5 only"
    )

    # 4.5. Partition methods into regular and consolidation-requiring
    regular_methods = [m for m in selected_methods if not m.requires_consolidation]
    consolidation_methods = [m for m in selected_methods if m.requires_consolidation]

    print(f"Running {len(regular_methods)} regular + {len(consolidation_methods)} consolidation methods x {n_queries} queries ...\n")

    # 5. Run retrieval for each method x query
    # Structure: method_name -> list of per-query dicts
    method_results: dict[str, list[dict]] = {}
    method_latencies: dict[str, list[float]] = {}

    # Use the corpus reference time so activation decay matches access events
    benchmark_now = corpus.metadata.get("generated_at", time.time())

    # Track original config for swapping
    original_cfg = getattr(search_index, '_cfg', None)

    for method in regular_methods:
        method_results[method.name] = []
        method_latencies[method.name] = []

        # Swap search index config for methods that need it (RRF vs Linear)
        if hasattr(search_index, '_cfg'):
            search_index._cfg = method.config

        for qi, query in enumerate(corpus.ground_truth):
            t0 = time.perf_counter()
            results = await run_retrieval(
                query.query_text,
                "benchmark",
                graph_store,
                activation_store,
                search_index,
                method,
                limit=10,
                now=benchmark_now,
                community_store=community_store,
                predicate_cache=predicate_cache,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            ranked_ids = [r.node_id for r in results]
            # Merge episode ground truth into entity ground truth
            relevant = dict(query.relevant_entities)
            if query.relevant_episodes:
                relevant.update(query.relevant_episodes)

            p5 = precision_at_k(ranked_ids, relevant, k=5)
            r10 = recall_at_k(ranked_ids, relevant, k=10)
            mrr = reciprocal_rank(ranked_ids, relevant)
            ndcg5 = ndcg_at_k(ranked_ids, relevant, k=5)

            per_query = {
                "query_id": query.query_id,
                "category": query.category,
                "p_at_5": p5,
                "r_at_10": r10,
                "mrr": mrr,
                "ndcg_at_5": ndcg5,
                "latency_ms": elapsed_ms,
            }
            method_results[method.name].append(per_query)
            method_latencies[method.name].append(elapsed_ms)

            if args.verbose:
                print(
                    f"  [{method.name}] Q{qi + 1:02d} ({query.category:13s}) "
                    f"P@5={p5:.3f}  R@10={r10:.3f}  MRR={mrr:.3f}  "
                    f"nDCG@5={ndcg5:.3f}  {elapsed_ms:.1f}ms"
                )

        if args.verbose:
            print()

    # Restore original search config
    if hasattr(search_index, '_cfg') and original_cfg is not None:
        search_index._cfg = original_cfg

    # 5.5. Run consolidation methods (after running a consolidation cycle)
    if consolidation_methods:
        from engram.consolidation.engine import ConsolidationEngine

        consol_engine = ConsolidationEngine(
            graph_store, activation_store, search_index, cfg=bench_cfg,
        )
        print("Running consolidation cycle ...")
        cycle = await consol_engine.run_cycle(
            group_id="benchmark", trigger="benchmark", dry_run=False,
        )
        affected = sum(r.items_affected for r in cycle.phase_results)
        print(f"Consolidation: {cycle.status}, {affected} items affected\n")

        for method in consolidation_methods:
            method_results[method.name] = []
            method_latencies[method.name] = []

            if hasattr(search_index, '_cfg'):
                search_index._cfg = method.config

            for qi, query in enumerate(corpus.ground_truth):
                t0 = time.perf_counter()
                results = await run_retrieval(
                    query.query_text,
                    "benchmark",
                    graph_store,
                    activation_store,
                    search_index,
                    method,
                    limit=10,
                    now=benchmark_now,
                    community_store=community_store,
                    predicate_cache=predicate_cache,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                ranked_ids = [r.node_id for r in results]
                relevant = dict(query.relevant_entities)
                if query.relevant_episodes:
                    relevant.update(query.relevant_episodes)

                p5 = precision_at_k(ranked_ids, relevant, k=5)
                r10 = recall_at_k(ranked_ids, relevant, k=10)
                mrr = reciprocal_rank(ranked_ids, relevant)
                ndcg5 = ndcg_at_k(ranked_ids, relevant, k=5)

                per_query = {
                    "query_id": query.query_id,
                    "category": query.category,
                    "p_at_5": p5,
                    "r_at_10": r10,
                    "mrr": mrr,
                    "ndcg_at_5": ndcg5,
                    "latency_ms": elapsed_ms,
                }
                method_results[method.name].append(per_query)
                method_latencies[method.name].append(elapsed_ms)

                if args.verbose:
                    print(
                        f"  [{method.name}] Q{qi + 1:02d} ({query.category:13s}) "
                        f"P@5={p5:.3f}  R@10={r10:.3f}  MRR={mrr:.3f}  "
                        f"nDCG@5={ndcg5:.3f}  {elapsed_ms:.1f}ms"
                    )

            if args.verbose:
                print()

        # Restore original search config again
        if hasattr(search_index, '_cfg') and original_cfg is not None:
            search_index._cfg = original_cfg

    # 6. Aggregate results
    categories = sorted({q.category for q in corpus.ground_truth})

    overall: dict[str, dict[str, float]] = {}
    by_category: dict[str, dict[str, dict[str, float]]] = {}
    latency_stats: dict[str, dict[str, float]] = {}

    for method_name, per_query_list in method_results.items():
        # Overall means
        n = len(per_query_list)
        overall[method_name] = {
            "p_at_5": sum(q["p_at_5"] for q in per_query_list) / n if n else 0.0,
            "r_at_10": sum(q["r_at_10"] for q in per_query_list) / n if n else 0.0,
            "mrr": sum(q["mrr"] for q in per_query_list) / n if n else 0.0,
            "ndcg_at_5": sum(q["ndcg_at_5"] for q in per_query_list) / n if n else 0.0,
        }

        # Per category
        by_category[method_name] = {}
        for cat in categories:
            cat_queries = [q for q in per_query_list if q["category"] == cat]
            nc = len(cat_queries)
            by_category[method_name][cat] = {
                "p_at_5": sum(q["p_at_5"] for q in cat_queries) / nc if nc else 0.0,
                "r_at_10": sum(q["r_at_10"] for q in cat_queries) / nc if nc else 0.0,
                "mrr": sum(q["mrr"] for q in cat_queries) / nc if nc else 0.0,
                "ndcg_at_5": sum(q["ndcg_at_5"] for q in cat_queries) / nc if nc else 0.0,
            }

        # Latency
        times = sorted(method_latencies[method_name])
        latency_stats[method_name] = {
            "mean_ms": sum(times) / len(times) if times else 0.0,
            "p50_ms": _percentile(times, 50),
            "p95_ms": _percentile(times, 95),
            "p99_ms": _percentile(times, 99),
        }

    # 7. Pairwise bootstrap comparisons on P@5
    comparisons: list[dict] = []
    method_p5_scores: dict[str, list[float]] = {}
    for method_name, per_query_list in method_results.items():
        method_p5_scores[method_name] = [q["p_at_5"] for q in per_query_list]

    for name_a, name_b in combinations(method_results.keys(), 2):
        scores_a = method_p5_scores[name_a]
        scores_b = method_p5_scores[name_b]
        mean_diff, ci_lower, ci_upper = bootstrap_ci(
            scores_a, scores_b, n_resamples=args.bootstrap_n,
        )
        significant = (ci_lower > 0.0) or (ci_upper < 0.0)
        comparisons.append({
            "a": name_a,
            "b": name_b,
            "metric": "p_at_5",
            "mean_diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": significant,
        })

    # 7.5. Independent subset analysis (non-circular queries)
    independent_categories = {"semantic", "graph_traversal", "cross_cluster"}
    independent_overall: dict[str, dict[str, float]] = {}
    independent_comparisons: list[dict] = []
    independent_p5_scores: dict[str, list[float]] = {}

    for method_name, per_query_list in method_results.items():
        ind_queries = [q for q in per_query_list if q["category"] in independent_categories]
        ni = len(ind_queries)
        independent_overall[method_name] = {
            "p_at_5": sum(q["p_at_5"] for q in ind_queries) / ni if ni else 0.0,
            "r_at_10": sum(q["r_at_10"] for q in ind_queries) / ni if ni else 0.0,
            "mrr": sum(q["mrr"] for q in ind_queries) / ni if ni else 0.0,
            "ndcg_at_5": sum(q["ndcg_at_5"] for q in ind_queries) / ni if ni else 0.0,
            "n_queries": ni,
        }
        independent_p5_scores[method_name] = [q["p_at_5"] for q in ind_queries]

    for name_a, name_b in combinations(method_results.keys(), 2):
        scores_a = independent_p5_scores[name_a]
        scores_b = independent_p5_scores[name_b]
        if scores_a and scores_b and len(scores_a) == len(scores_b):
            mean_diff, ci_lower, ci_upper = bootstrap_ci(
                scores_a, scores_b, n_resamples=args.bootstrap_n,
            )
            significant = (ci_lower > 0.0) or (ci_upper < 0.0)
            independent_comparisons.append({
                "a": name_a,
                "b": name_b,
                "metric": "p_at_5",
                "mean_diff": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "significant": significant,
            })

    # 8. Print console output
    _print_results(
        overall, by_category, latency_stats, comparisons,
        selected_methods, categories, corpus,
        independent_overall=independent_overall,
        independent_comparisons=independent_comparisons,
    )

    # 9. Build JSON output
    output = {
        "seed": args.seed,
        "corpus": {
            "entities": n_entities,
            "relationships": n_relationships,
            "access_events": n_access_events,
            "episodes": n_episodes,
            "queries": n_queries,
        },
        "methods": {},
        "comparisons": comparisons,
        "independent_subset": {
            "categories": sorted(independent_categories),
            "methods": independent_overall,
            "comparisons": independent_comparisons,
        },
    }
    for method_name in method_results:
        output["methods"][method_name] = {
            "overall": overall[method_name],
            "by_category": by_category[method_name],
            "latency": latency_stats[method_name],
            "per_query": method_results[method_name],
        }

    # 10. Write JSON if requested
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, indent=2))
        print(f"\nJSON results written to {json_path}")

    # 11. Cleanup
    await graph_store.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return output


# ---------------------------------------------------------------------------
# Console output formatting
# ---------------------------------------------------------------------------


def _print_results(
    overall: dict[str, dict[str, float]],
    by_category: dict[str, dict[str, dict[str, float]]],
    latency_stats: dict[str, dict[str, float]],
    comparisons: list[dict],
    methods: list[RetrievalMethod],
    categories: list[str],
    corpus: CorpusSpec,
    independent_overall: dict[str, dict[str, float]] | None = None,
    independent_comparisons: list[dict] | None = None,
) -> None:
    """Print the formatted results table to stdout."""

    # Count queries per category
    cat_counts: dict[str, int] = {}
    for q in corpus.ground_truth:
        cat_counts[q.category] = cat_counts.get(q.category, 0) + 1

    cat_summary = ", ".join(
        f"{cat_counts.get(c, 0)} {c}" for c in categories
    )

    print("=" * 60)
    print("=== A/B Benchmark Results ===")
    print("=" * 60)
    print(
        f"Corpus: {len(corpus.entities)} entities, "
        f"{len(corpus.relationships)} relationships, "
        f"{len(corpus.access_events)} access events, "
        f"{len(corpus.episodes)} episodes"
    )
    print(f"Queries: {len(corpus.ground_truth)} ground truth ({cat_summary})")
    search_mode = corpus.metadata.get("search_mode", "FTS5")
    print(f"Search: {search_mode}")
    print()

    # --- Overall Results ---
    print("--- Overall Results ---")
    header = (
        f"{'Method':<16}| {'P@5':>5} | {'R@10':>5} | {'MRR':>5} "
        f"| {'nDCG@5':>6} | {'Latency(ms)':>11}"
    )
    sep = (
        f"{'-' * 16}|{'-' * 7}|{'-' * 7}|{'-' * 7}"
        f"|{'-' * 8}|{'-' * 12}"
    )
    print(header)
    print(sep)
    for method in methods:
        name = method.name
        o = overall[name]
        lat = latency_stats[name]
        print(
            f"{name:<16}| {o['p_at_5']:>5.3f} | {o['r_at_10']:>5.3f} | "
            f"{o['mrr']:>5.3f} | {o['ndcg_at_5']:>6.3f} | "
            f"{lat['mean_ms']:>8.1f}"
        )
    print()

    # --- Per-Category Breakdown (P@5) ---
    print("--- Per-Category Breakdown (P@5) ---")
    method_names = [m.name for m in methods]
    col_width = max(len(n) for n in method_names) + 2
    col_width = max(col_width, 12)

    cat_header = f"{'Category':<15}"
    for name in method_names:
        cat_header += f"| {name:^{col_width}}"
    print(cat_header)

    cat_sep = f"{'-' * 15}"
    for _ in method_names:
        cat_sep += f"|{'-' * (col_width + 1)}"
    print(cat_sep)

    for cat in categories:
        row = f"{cat:<15}"
        for name in method_names:
            val = by_category[name][cat]["p_at_5"]
            row += f"| {val:^{col_width}.3f}"
        print(row)
    print()

    # --- Pairwise Comparisons ---
    if comparisons:
        print("--- Pairwise Comparisons (P@5, 95% CI) ---")
        for comp in comparisons:
            sig_marker = " *" if comp["significant"] else ""
            print(
                f"{comp['a']} vs {comp['b']}:  "
                f"{comp['mean_diff']:+.3f} "
                f"[{comp['ci_lower']:+.3f}, {comp['ci_upper']:+.3f}]"
                f"{sig_marker}"
            )
        print("(* = significant at 95% level, CI excludes 0)")
        print()

    # --- Independent Subset Analysis ---
    if independent_overall:
        n_ind = next(
            (v.get("n_queries", 0) for v in independent_overall.values()), 0
        )
        cats = "semantic, graph_traversal, cross_cluster"
        print(f"--- Independent Subset ({n_ind} queries: {cats}) ---")
        method_names = [m.name for m in methods]
        ind_header = (
            f"{'Method':<16}| {'P@5':>5} | {'R@10':>5} | {'MRR':>5} | {'nDCG@5':>6}"
        )
        ind_sep = f"{'-' * 16}|{'-' * 7}|{'-' * 7}|{'-' * 7}|{'-' * 8}"
        print(ind_header)
        print(ind_sep)
        for name in method_names:
            if name in independent_overall:
                o = independent_overall[name]
                print(
                    f"{name:<16}| {o['p_at_5']:>5.3f} | {o['r_at_10']:>5.3f} | "
                    f"{o['mrr']:>5.3f} | {o['ndcg_at_5']:>6.3f}"
                )
        print()

    if independent_comparisons:
        print("--- Independent Subset Pairwise Comparisons (P@5, 95% CI) ---")
        for comp in independent_comparisons:
            sig_marker = " *" if comp["significant"] else ""
            print(
                f"{comp['a']} vs {comp['b']}:  "
                f"{comp['mean_diff']:+.3f} "
                f"[{comp['ci_lower']:+.3f}, {comp['ci_upper']:+.3f}]"
                f"{sig_marker}"
            )
        print("(* = significant at 95% level, CI excludes 0)")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B benchmark for Engram retrieval methods",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for corpus generation (default: 42)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help=(
            "Method names to benchmark (default: all). "
            "Available: " + ", ".join(f'"{m.name}"' for m in ALL_METHODS)
        ),
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Path for JSON output file (optional)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-query results as they run",
    )
    parser.add_argument(
        "--bootstrap-n", type=int, default=1000,
        help="Number of bootstrap resamples for CIs (default: 1000)",
    )
    parser.add_argument(
        "--embeddings", action="store_true",
        help="Use Voyage AI embeddings (requires VOYAGE_API_KEY)",
    )
    parser.add_argument(
        "--structure-aware", action="store_true",
        dest="structure_aware",
        help="Re-index entities with predicate-enriched text for semantic queries",
    )
    parser.add_argument(
        "--entities", type=int, default=1000,
        help="Total entities in corpus (default: 1000). Try 5000, 10000, 50000.",
    )

    args = parser.parse_args()
    # Default structure-aware when embeddings are enabled
    if args.embeddings and not args.structure_aware:
        args.structure_aware = True
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
