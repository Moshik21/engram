#!/usr/bin/env python3
"""3-backend benchmark comparison: Lite (SQLite) vs Full (FalkorDB+Redis) vs Helix (HelixDB).

Generates a deterministic corpus and runs identical operations against each
available backend, measuring write throughput, search latency, retrieval
quality (nDCG@10, MRR, precision@10), and memory footprint.

Usage:
    cd server && uv run python scripts/benchmark_backends.py [--entities 200] [--seed 42]

Prerequisites:
    - Lite:  always available (SQLite in-memory)
    - Full:  FalkorDB + Redis running (docker compose up -d)
    - Helix: HelixDB running (helix push dev or docker)
"""

from __future__ import annotations

import asyncio
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from engram.benchmark.corpus import CorpusGenerator, CorpusSpec
from engram.benchmark.methods import RetrievalMethod, run_retrieval
from engram.benchmark.metrics import ndcg_at_k, precision_at_k, reciprocal_rank
from engram.config import ActivationConfig
from engram.storage.memory.activation import MemoryActivationStore

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BackendResult:
    name: str
    available: bool = False
    startup_ms: float = 0.0
    load_ms: float = 0.0
    entities_loaded: int = 0
    relationships_loaded: int = 0
    episodes_loaded: int = 0
    search_latencies_ms: list[float] = field(default_factory=list)
    ndcg_scores: list[float] = field(default_factory=list)
    mrr_scores: list[float] = field(default_factory=list)
    precision_scores: list[float] = field(default_factory=list)
    category_ndcg: dict[str, list[float]] = field(default_factory=dict)
    rss_mb: float = 0.0
    error: str = ""
    write_eps: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    avg_lat: float = 0.0


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _rss_mb() -> float:
    """Current process RSS in MB (macOS returns bytes, Linux returns KB)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024


# ---------------------------------------------------------------------------
# Backend setup helpers
# ---------------------------------------------------------------------------


async def _setup_lite(cfg: ActivationConfig):
    """Create SQLite stores."""
    import tempfile

    from engram.storage.sqlite.graph import SQLiteGraphStore
    from engram.storage.sqlite.hybrid_search import HybridSearchIndex
    from engram.storage.sqlite.search import FTS5SearchIndex
    from engram.storage.sqlite.vectors import SQLiteVectorStore

    tmp = tempfile.mkdtemp(prefix="engram_bench_lite_")
    db_path = str(Path(tmp) / "bench.db")

    graph = SQLiteGraphStore(db_path)
    await graph.initialize()
    activation = MemoryActivationStore(cfg=cfg)

    fts = FTS5SearchIndex(db_path)
    await fts.initialize(db=graph._db)

    try:
        from engram.embeddings.provider import FastEmbedProvider

        provider = FastEmbedProvider(model="nomic-ai/nomic-embed-text-v1.5")
        vectors = SQLiteVectorStore(db_path)
        await vectors.initialize(db=graph._db)
        search = HybridSearchIndex(
            fts=fts,
            vector_store=vectors,
            provider=provider,
            fts_weight=0.3,
            vec_weight=0.7,
        )
    except ImportError:
        search = fts

    return graph, activation, search


async def _setup_full(cfg: ActivationConfig):
    """Create FalkorDB + Redis stores."""
    import redis.asyncio as aioredis

    from engram.config import EmbeddingConfig, FalkorDBConfig
    from engram.storage.falkordb.graph import FalkorDBGraphStore
    from engram.storage.vector.redis_search import RedisSearchIndex

    falkordb_cfg = FalkorDBConfig(
        host=os.environ.get("ENGRAM_FALKORDB__HOST", "localhost"),
        port=int(os.environ.get("ENGRAM_FALKORDB__PORT", "6380")),
        password=os.environ.get("ENGRAM_FALKORDB__PASSWORD", "engram_dev"),
    )
    redis_url = os.environ.get("ENGRAM_REDIS__URL", "redis://:engram_dev@localhost:6381/0")

    graph = FalkorDBGraphStore(falkordb_cfg)
    await graph.initialize()
    activation = MemoryActivationStore(cfg=cfg)

    redis_client = aioredis.from_url(redis_url, decode_responses=False)

    try:
        from engram.embeddings.provider import FastEmbedProvider

        provider = FastEmbedProvider(model="nomic-ai/nomic-embed-text-v1.5")
    except ImportError:
        from engram.embeddings.provider import NoopProvider

        provider = NoopProvider()

    search = RedisSearchIndex(
        redis_client,
        provider=provider,
        config=EmbeddingConfig(),
    )
    await search.initialize()

    return graph, activation, search


async def _setup_helix(cfg: ActivationConfig):
    """Create HelixDB stores with shared async client."""
    from engram.config import EmbeddingConfig, HelixDBConfig
    from engram.storage.helix.client import HelixClient
    from engram.storage.helix.graph import HelixGraphStore
    from engram.storage.helix.search import HelixSearchIndex

    helix_cfg = HelixDBConfig(
        host=os.environ.get("ENGRAM_HELIX__HOST", "localhost"),
        port=int(os.environ.get("ENGRAM_HELIX__PORT", "6969")),
        max_workers=8,
    )

    # Shared async client — one connection pool for all stores
    helix_client = HelixClient(helix_cfg)
    await helix_client.initialize()

    graph = HelixGraphStore(helix_cfg, client=helix_client)
    await graph.initialize()
    activation = MemoryActivationStore(cfg=cfg)

    try:
        from engram.embeddings.provider import FastEmbedProvider

        provider = FastEmbedProvider(model="nomic-ai/nomic-embed-text-v1.5")
    except ImportError:
        from engram.embeddings.provider import NoopProvider

        provider = NoopProvider()

    search = HelixSearchIndex(
        helix_cfg,
        provider=provider,
        embed_config=EmbeddingConfig(),
        storage_dim=0,
        embed_provider="local",
        embed_model="nomic-ai/nomic-embed-text-v1.5",
        client=helix_client,
    )
    await search.initialize()

    return graph, activation, search


async def _setup_helix_native(cfg: ActivationConfig):
    """Create HelixDB stores with native in-process PyO3 transport."""
    from engram.config import EmbeddingConfig, HelixDBConfig
    from engram.storage.helix.client import HelixClient
    from engram.storage.helix.graph import HelixGraphStore
    from engram.storage.helix.search import HelixSearchIndex

    helix_cfg = HelixDBConfig(
        transport="native",
        max_workers=8,
    )

    helix_client = HelixClient(helix_cfg)
    await helix_client.initialize()

    graph = HelixGraphStore(helix_cfg, client=helix_client)
    await graph.initialize()
    activation = MemoryActivationStore(cfg=cfg)

    try:
        from engram.embeddings.provider import FastEmbedProvider

        provider = FastEmbedProvider(model="nomic-ai/nomic-embed-text-v1.5")
    except ImportError:
        from engram.embeddings.provider import NoopProvider

        provider = NoopProvider()

    search = HelixSearchIndex(
        helix_cfg,
        provider=provider,
        embed_config=EmbeddingConfig(),
        storage_dim=0,
        embed_provider="local",
        embed_model="nomic-ai/nomic-embed-text-v1.5",
        client=helix_client,
    )
    await search.initialize()

    return graph, activation, search


# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------


async def _check_full() -> bool:
    try:
        import socket

        host = os.environ.get("ENGRAM_FALKORDB__HOST", "localhost")
        port = int(os.environ.get("ENGRAM_FALKORDB__PORT", "6380"))
        socket.create_connection((host, port), timeout=2)
        return True
    except Exception:
        return False


async def _check_helix() -> bool:
    try:
        import socket

        host = os.environ.get("ENGRAM_HELIX__HOST", "localhost")
        port = int(os.environ.get("ENGRAM_HELIX__PORT", "6969"))
        socket.create_connection((host, port), timeout=2)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


async def benchmark_backend(
    name: str,
    setup_fn,
    corpus: CorpusSpec,
    corpus_gen: CorpusGenerator,
    method: RetrievalMethod,
    group_id: str = "benchmark",
) -> BackendResult:
    """Run the full benchmark suite against one backend."""
    result = BackendResult(name=name, available=True)

    cfg = method.config
    try:
        # 1. Startup
        t0 = time.perf_counter()
        graph, activation, search = await setup_fn(cfg)
        result.startup_ms = (time.perf_counter() - t0) * 1000

        # 2. Load corpus
        t0 = time.perf_counter()
        await corpus_gen.load(corpus, graph, activation, search, structure_aware=False, cfg=cfg)
        result.load_ms = (time.perf_counter() - t0) * 1000
        result.entities_loaded = len(corpus.entities)
        result.relationships_loaded = len(corpus.relationships)
        result.episodes_loaded = len(corpus.episodes)

        # 3. Run ground-truth queries
        ref_time = corpus.metadata.get("reference_time", time.time())

        for gt in corpus.ground_truth:
            t_q = time.perf_counter()
            results = await run_retrieval(
                query=gt.query_text,
                group_id=group_id,
                graph_store=graph,
                activation_store=activation,
                search_index=search,
                method=method,
                limit=10,
                now=ref_time,
                total_entities=len(corpus.entities),
            )
            elapsed_ms = (time.perf_counter() - t_q) * 1000
            result.search_latencies_ms.append(elapsed_ms)

            # Score against ground truth
            returned_ids = [r.node_id for r in results]
            relevant = gt.relevant_entities  # {entity_id: relevance_grade}

            ndcg = ndcg_at_k(returned_ids, relevant, k=10)
            mrr = reciprocal_rank(returned_ids, relevant)
            prec = precision_at_k(returned_ids, relevant, k=10)

            result.ndcg_scores.append(ndcg)
            result.mrr_scores.append(mrr)
            result.precision_scores.append(prec)

            cat = gt.category
            result.category_ndcg.setdefault(cat, []).append(ndcg)

        # 4. Memory
        result.rss_mb = _rss_mb()

        # Compute derived metrics
        if result.load_ms > 0:
            result.write_eps = result.entities_loaded / (result.load_ms / 1000)
        s = sorted(result.search_latencies_ms)
        result.p50 = _percentile(s, 50)
        result.p95 = _percentile(s, 95)
        result.avg_lat = _avg(result.search_latencies_ms)

        # Cleanup
        if hasattr(graph, "close"):
            await graph.close()

    except Exception as e:
        import traceback

        result.error = str(e)
        result.available = False
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_results(results: list[BackendResult], corpus: CorpusSpec) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 78)
    print("  ENGRAM BACKEND BENCHMARK COMPARISON")
    print("=" * 78)
    print(f"  Corpus: {len(corpus.entities)} entities, {len(corpus.relationships)} rels, "
          f"{len(corpus.episodes)} episodes, {len(corpus.ground_truth)} queries")
    print("=" * 78)

    # Header
    names = [r.name for r in results]
    col_w = 18
    header = f"{'Metric':<28}" + "".join(f"{n:>{col_w}}" for n in names)
    print(f"\n{header}")
    print("-" * (28 + col_w * len(names)))

    def row(label: str, values: list[str]) -> None:
        print(f"{label:<28}" + "".join(f"{v:>{col_w}}" for v in values))

    # Availability
    row("Available", [("YES" if r.available else "NO") for r in results])

    active = [r for r in results if r.available]
    if not active:
        print("\nNo backends available for comparison!")
        return

    # Timing
    row("Startup (ms)", [f"{r.startup_ms:.0f}" if r.available else "-" for r in results])
    row("Corpus load (ms)", [f"{r.load_ms:.0f}" if r.available else "-" for r in results])
    row("Write throughput (ent/s)", [f"{r.write_eps:.0f}" if r.available else "-" for r in results])

    # Latency
    row("Search p50 (ms)", [f"{r.p50:.1f}" if r.available else "-" for r in results])
    row("Search p95 (ms)", [f"{r.p95:.1f}" if r.available else "-" for r in results])
    row("Search avg (ms)", [f"{r.avg_lat:.1f}" if r.available else "-" for r in results])

    # Quality
    row("nDCG@10", [f"{_avg(r.ndcg_scores):.3f}" if r.available else "-" for r in results])
    row("MRR", [f"{_avg(r.mrr_scores):.3f}" if r.available else "-" for r in results])
    row("Precision@10", [
        f"{_avg(r.precision_scores):.3f}" if r.available else "-" for r in results
    ])

    # Memory
    row("RSS (MB)", [f"{r.rss_mb:.0f}" if r.available else "-" for r in results])

    # Per-category breakdown
    all_cats: set[str] = set()
    for r in active:
        all_cats.update(r.category_ndcg.keys())

    if all_cats:
        print(f"\n{'--- nDCG@10 by query category ---':^{28 + col_w * len(names)}}")
        print("-" * (28 + col_w * len(names)))
        for cat in sorted(all_cats):
            vals = []
            for r in results:
                if r.available and cat in r.category_ndcg:
                    vals.append(f"{_avg(r.category_ndcg[cat]):.3f}")
                else:
                    vals.append("-")
            row(f"  {cat}", vals)

    print("\n" + "=" * 78)

    # Winner summary
    if len(active) > 1:
        best_ndcg = max(active, key=lambda r: _avg(r.ndcg_scores))
        best_lat = min(active, key=lambda r: r.avg_lat if r.avg_lat > 0 else float("inf"))
        best_write = max(active, key=lambda r: r.write_eps)
        ndcg_val = _avg(best_ndcg.ndcg_scores)
        print(f"  Best retrieval quality:  {best_ndcg.name} (nDCG={ndcg_val:.3f})")
        print(f"  Fastest search:          {best_lat.name} (avg={best_lat.avg_lat:.1f}ms)")
        print(f"  Fastest writes:          {best_write.name} ({best_write.write_eps:.0f} ent/s)")
        print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Engram storage backends")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entities", type=int, default=200,
                        help="Corpus size (default 200 for fast runs, use 1000+ for production)")
    parser.add_argument("--backends", nargs="+", default=["lite", "helix", "full"],
                        help="Backends to test (default: all available)")
    args = parser.parse_args()

    # Generate corpus once
    print(f"Generating corpus (seed={args.seed}, entities={args.entities})...")
    gen = CorpusGenerator(seed=args.seed, total_entities=args.entities)
    corpus = gen.generate()
    print(f"  {len(corpus.entities)} entities, {len(corpus.relationships)} rels, "
          f"{len(corpus.episodes)} episodes, {len(corpus.ground_truth)} queries\n")

    # Use a simple retrieval method for fair comparison
    method = RetrievalMethod(
        name="benchmark",
        description="Balanced retrieval for backend comparison",
        config=ActivationConfig(),
        spreading_enabled=True,
    )

    results: list[BackendResult] = []

    # --- Lite ---
    if "lite" in args.backends:
        print("[LITE] SQLite backend...")
        r = await benchmark_backend("Lite (SQLite)", _setup_lite, corpus, gen, method)
        results.append(r)
        if r.available:
            print(f"  Done: {r.load_ms:.0f}ms load, {r.avg_lat:.1f}ms avg search\n")
        else:
            print(f"  FAILED: {r.error}\n")

    # --- Helix Native (PyO3 in-process) ---
    if "native" in args.backends:
        try:
            import helix_native  # noqa: F401

            print("[NATIVE] HelixDB PyO3 in-process backend...")
            r = await benchmark_backend(
                "Native (PyO3)", _setup_helix_native, corpus, gen, method
            )
            results.append(r)
            if r.available:
                print(f"  Done: {r.load_ms:.0f}ms load, {r.avg_lat:.1f}ms avg search\n")
            else:
                print(f"  FAILED: {r.error}\n")
        except ImportError:
            print("[NATIVE] helix_native not installed -- skipping\n")
            results.append(BackendResult(name="Native (PyO3)", error="not installed"))

    # --- Helix ---
    if "helix" in args.backends:
        if await _check_helix():
            print("[HELIX] HelixDB backend...")
            r = await benchmark_backend("Helix (HelixDB)", _setup_helix, corpus, gen, method)
            results.append(r)
            if r.available:
                print(f"  Done: {r.load_ms:.0f}ms load, {r.avg_lat:.1f}ms avg search\n")
            else:
                print(f"  FAILED: {r.error}\n")
        else:
            print("[HELIX] HelixDB not available -- skipping\n")
            results.append(BackendResult(name="Helix (HelixDB)", error="not running"))

    # --- Full ---
    if "full" in args.backends:
        if await _check_full():
            print("[FULL] FalkorDB + Redis backend...")
            r = await benchmark_backend("Full (FalkorDB)", _setup_full, corpus, gen, method)
            results.append(r)
            if r.available:
                print(f"  Done: {r.load_ms:.0f}ms load, {r.avg_lat:.1f}ms avg search\n")
            else:
                print(f"  FAILED: {r.error}\n")
        else:
            print("[FULL] FalkorDB + Redis not available -- skipping\n")
            results.append(BackendResult(name="Full (FalkorDB)", error="not running"))

    # Print comparison
    print_results(results, corpus)


if __name__ == "__main__":
    asyncio.run(main())
