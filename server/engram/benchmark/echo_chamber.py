"""Echo chamber benchmark: measures whether activation creates filter bubbles."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from engram.benchmark.metrics import gini_coefficient
from engram.config import ActivationConfig
from engram.retrieval.pipeline import retrieve


@dataclass
class EchoChamberSnapshot:
    """State snapshot at a point during the echo chamber simulation."""

    query_index: int
    coverage: float  # fraction of corpus entities retrievable at P@5 > 0
    gini: float  # Gini coefficient of access counts
    top10_ids: list[str]  # top-10 entity IDs by access count
    top10_jaccard: float  # Jaccard similarity with previous snapshot


@dataclass
class EchoChamberResult:
    """Full echo chamber benchmark result."""

    total_queries: int
    snapshots: list[EchoChamberSnapshot]
    final_coverage: float
    final_gini: float
    final_top10_jaccard: float
    pass_coverage: bool  # coverage > 40%
    pass_gini: bool  # gini < 0.70
    ts_enabled: bool = False


async def run_echo_chamber(
    hot_queries: list[str],
    diverse_queries: list[str],
    corpus_entity_ids: list[str],
    graph_store,
    activation_store,
    search_index,
    cfg: ActivationConfig,
    group_id: str = "default",
    total_queries: int = 200,
    hot_ratio: float = 0.6,
    snapshot_interval: int = 50,
    seed: int = 42,
) -> EchoChamberResult:
    """Run echo chamber simulation.

    Alternates between hot-topic queries (echo chamber pressure) and
    diverse queries. Records access via the pipeline and measures
    coverage, Gini, and top-10 stability at intervals.
    """
    rng = random.Random(seed)
    snapshots: list[EchoChamberSnapshot] = []
    prev_top10: set[str] = set()

    for i in range(total_queries):
        # Select query
        if rng.random() < hot_ratio:
            query = rng.choice(hot_queries)
        else:
            query = rng.choice(diverse_queries)

        # Run retrieval (records access via pipeline)
        results = await retrieve(
            query=query,
            group_id=group_id,
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=5,
            enable_routing=False,
        )

        # Record access for returned entities
        now = time.time()
        for r in results:
            await activation_store.record_access(
                r.node_id, now, group_id=group_id
            )

        # Take snapshot at intervals
        if (i + 1) % snapshot_interval == 0 or i == total_queries - 1:
            snapshot = await _take_snapshot(
                i + 1,
                corpus_entity_ids,
                activation_store,
                group_id,
                prev_top10,
            )
            snapshots.append(snapshot)
            prev_top10 = set(snapshot.top10_ids)

    final = snapshots[-1] if snapshots else EchoChamberSnapshot(
        query_index=0, coverage=0, gini=0, top10_ids=[], top10_jaccard=0
    )

    return EchoChamberResult(
        total_queries=total_queries,
        snapshots=snapshots,
        final_coverage=final.coverage,
        final_gini=final.gini,
        final_top10_jaccard=final.top10_jaccard,
        pass_coverage=final.coverage > 0.40,
        pass_gini=final.gini < 0.70,
        ts_enabled=cfg.ts_enabled,
    )


async def _take_snapshot(
    query_index: int,
    corpus_entity_ids: list[str],
    activation_store,
    group_id: str,
    prev_top10: set[str],
) -> EchoChamberSnapshot:
    """Compute echo chamber metrics at a point in time."""
    # Gather access counts
    states = await activation_store.batch_get(corpus_entity_ids)
    access_counts = [
        float(states[eid].access_count) if eid in states else 0.0
        for eid in corpus_entity_ids
    ]

    # Coverage: fraction of entities with access_count > 0
    total = len(corpus_entity_ids)
    accessed = sum(1 for c in access_counts if c > 0)
    coverage = accessed / total if total > 0 else 0.0

    # Gini coefficient
    gini = gini_coefficient(access_counts)

    # Top-10 by access count
    sorted_entities = sorted(
        zip(corpus_entity_ids, access_counts),
        key=lambda x: x[1],
        reverse=True,
    )
    top10_ids = [eid for eid, _ in sorted_entities[:10]]

    # Jaccard similarity with previous top-10
    current_set = set(top10_ids)
    if prev_top10:
        intersection = len(current_set & prev_top10)
        union = len(current_set | prev_top10)
        jaccard = intersection / union if union > 0 else 0.0
    else:
        jaccard = 0.0  # First snapshot has no comparison

    return EchoChamberSnapshot(
        query_index=query_index,
        coverage=coverage,
        gini=gini,
        top10_ids=top10_ids,
        top10_jaccard=jaccard,
    )
