#!/usr/bin/env python3
"""Diagnose what spreading activation actually produces during benchmark queries.

Generates the standard benchmark corpus (1k entities, seed=42), runs the
Full Engram retrieval pipeline for every ground-truth query, and instruments
the spreading activation step to capture hard numbers on:

  - How many seeds are selected and what energy they carry
  - How many entities receive spreading bonuses, and the bonus values
  - How many "discovered" entities come from spreading (not in original pool)
  - Whether any discovered entities appear in the ground truth
  - The spreading bonus values for ground-truth vs non-ground-truth entities
  - The final score breakdown (semantic, activation, spreading, edge) for top-10

Usage:
    cd server && uv run python scripts/diagnose_spreading.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

# Ensure engram package is importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.activation.spreading import identify_seeds, spread_activation
from engram.benchmark.corpus import CorpusGenerator, CorpusSpec
from engram.config import ActivationConfig
from engram.retrieval.candidate_pool import generate_candidates
from engram.retrieval.router import QueryType, apply_route, classify_query
from engram.retrieval.scorer import ScoredResult, score_candidates
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex

# ---------------------------------------------------------------------------
# Per-query diagnostic container
# ---------------------------------------------------------------------------


class QueryDiagnostic:
    """Captures all spreading-related measurements for one query."""

    def __init__(self, query_id: str, category: str, query_text: str):
        self.query_id = query_id
        self.category = category
        self.query_text = query_text

        # Seeds
        self.num_seeds: int = 0
        self.seed_energies: list[float] = []

        # Spreading output
        self.num_bonuses: int = 0  # entities that got any spreading bonus
        self.bonus_values: list[float] = []  # all bonus values
        self.bonus_for_gt: list[float] = []  # bonuses for ground-truth entities
        self.bonus_for_non_gt: list[float] = []  # bonuses for non-ground-truth entities

        # Discovered entities (found by spreading, not in original candidate pool)
        self.num_discovered: int = 0
        self.discovered_in_gt: int = 0  # KEY METRIC: discovered AND in ground truth
        self.discovered_ids: list[str] = []

        # Candidate pool
        self.num_candidates_before_spread: int = 0
        self.num_candidates_after_spread: int = 0
        self.num_ground_truth: int = 0
        self.gt_in_candidates_before: int = 0
        self.gt_in_candidates_after: int = 0

        # Top-10 score breakdowns
        self.top10: list[ScoredResult] = []

        # Overlap
        self.gt_in_top10: int = 0


# ---------------------------------------------------------------------------
# Instrumented retrieval
# ---------------------------------------------------------------------------


async def instrumented_retrieve(
    query_text: str,
    ground_truth_ids: set[str],
    group_id: str,
    graph_store,
    activation_store,
    search_index,
    cfg: ActivationConfig,
    now: float,
    total_entities: int,
    diag: QueryDiagnostic,
) -> list[ScoredResult]:
    """Run Full Engram retrieval with instrumentation at every step."""

    # Step 1: Generate candidates (multi-pool, same as Full Engram)
    pre_query_type = await classify_query(query_text)

    candidates = await generate_candidates(
        query=query_text,
        group_id=group_id,
        search_index=search_index,
        activation_store=activation_store,
        graph_store=graph_store,
        cfg=cfg,
        now=now,
        total_entities=total_entities,
        query_type=pre_query_type,
    )

    query_type = await classify_query(query_text, search_results=candidates or [])
    routed_cfg = apply_route(query_type, cfg)
    temporal_mode = query_type == QueryType.TEMPORAL

    if not candidates:
        return []

    diag.num_candidates_before_spread = len(candidates)
    candidate_ids_before = {eid for eid, _ in candidates}
    diag.gt_in_candidates_before = len(ground_truth_ids & candidate_ids_before)

    # Step 2: Batch get activation states
    entity_ids = [eid for eid, _ in candidates]
    activation_states = await activation_store.batch_get(entity_ids)

    # Step 3: Identify seeds
    seeds = identify_seeds(
        candidates,
        activation_states,
        now,
        routed_cfg,
        temporal_mode=temporal_mode,
    )
    seed_node_ids = {nid for nid, _ in seeds}

    diag.num_seeds = len(seeds)
    diag.seed_energies = [e for _, e in seeds]

    # Step 4: Spread activation
    bonuses, hop_distances = await spread_activation(
        seeds,
        graph_store,
        routed_cfg,
        group_id=group_id,
    )

    # Record spreading diagnostics
    diag.num_bonuses = len(bonuses)
    diag.bonus_values = list(bonuses.values())

    for eid, bonus in bonuses.items():
        if eid in ground_truth_ids:
            diag.bonus_for_gt.append(bonus)
        else:
            diag.bonus_for_non_gt.append(bonus)

    # Step 4.5: Merge spreading-discovered entities
    existing_ids = {eid for eid, _ in candidates}
    new_ids = [nid for nid in bonuses if nid not in existing_ids and bonuses[nid] > 0.0]

    diag.num_discovered = len(new_ids)
    diag.discovered_ids = new_ids
    diag.discovered_in_gt = len(set(new_ids) & ground_truth_ids)

    if new_ids:
        new_states = await activation_store.batch_get(new_ids)
        activation_states.update(new_states)
        # FTS5 compute_similarity returns empty dict, so discovered entities get sem_sim=0.0
        discovered_sims = await search_index.compute_similarity(
            query=query_text,
            entity_ids=new_ids,
            group_id=group_id,
        )
        candidates = candidates + [(nid, discovered_sims.get(nid, 0.0)) for nid in new_ids]

    diag.num_candidates_after_spread = len(candidates)
    candidate_ids_after = {eid for eid, _ in candidates}
    diag.gt_in_candidates_after = len(ground_truth_ids & candidate_ids_after)

    # Step 5: Score all candidates
    scored = score_candidates(
        candidates=candidates,
        spreading_bonuses=bonuses,
        hop_distances=hop_distances,
        seed_node_ids=seed_node_ids,
        activation_states=activation_states,
        now=now,
        cfg=routed_cfg,
    )

    # Record top-10 results
    top10 = scored[:10]
    diag.top10 = top10
    diag.gt_in_top10 = sum(1 for r in top10 if r.node_id in ground_truth_ids)

    return top10


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def print_summary(diagnostics: list[QueryDiagnostic]) -> None:
    """Print summary tables grouped by query category."""

    categories = sorted(set(d.category for d in diagnostics))

    # -----------------------------------------------------------------------
    # Table 1: Spreading mechanics per category
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLE 1: SPREADING ACTIVATION MECHANICS PER QUERY CATEGORY")
    print("=" * 100)
    header = (
        f"{'Category':<18s} {'#Q':>3s} "
        f"{'Avg Seeds':>10s} {'Avg Seed E':>10s} "
        f"{'Avg Bonus#':>10s} {'Avg Bonus':>10s} "
        f"{'Avg Disc':>9s} {'Disc in GT':>10s}"
    )
    print(header)
    print("-" * 100)

    all_diags = diagnostics  # for total row

    for cat in categories:
        cat_diags = [d for d in diagnostics if d.category == cat]
        n = len(cat_diags)

        avg_seeds = _mean([d.num_seeds for d in cat_diags])
        avg_seed_energy = _mean(
            [_mean(d.seed_energies) if d.seed_energies else 0.0 for d in cat_diags]
        )
        avg_bonuses = _mean([d.num_bonuses for d in cat_diags])
        avg_bonus_val = _mean([_mean(d.bonus_values) if d.bonus_values else 0.0 for d in cat_diags])
        avg_discovered = _mean([d.num_discovered for d in cat_diags])
        total_disc_in_gt = sum(d.discovered_in_gt for d in cat_diags)

        print(
            f"{cat:<18s} {n:>3d} "
            f"{avg_seeds:>10.1f} {avg_seed_energy:>10.4f} "
            f"{avg_bonuses:>10.1f} {avg_bonus_val:>10.4f} "
            f"{avg_discovered:>9.1f} {total_disc_in_gt:>10d}"
        )

    # Total row
    n = len(all_diags)
    print("-" * 100)
    avg_seeds = _mean([d.num_seeds for d in all_diags])
    avg_seed_energy = _mean([_mean(d.seed_energies) if d.seed_energies else 0.0 for d in all_diags])
    avg_bonuses = _mean([d.num_bonuses for d in all_diags])
    avg_bonus_val = _mean([_mean(d.bonus_values) if d.bonus_values else 0.0 for d in all_diags])
    avg_discovered = _mean([d.num_discovered for d in all_diags])
    total_disc_in_gt = sum(d.discovered_in_gt for d in all_diags)
    print(
        f"{'TOTAL':<18s} {n:>3d} "
        f"{avg_seeds:>10.1f} {avg_seed_energy:>10.4f} "
        f"{avg_bonuses:>10.1f} {avg_bonus_val:>10.4f} "
        f"{avg_discovered:>9.1f} {total_disc_in_gt:>10d}"
    )

    # -----------------------------------------------------------------------
    # Table 2: Ground truth coverage
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLE 2: GROUND TRUTH COVERAGE (does spreading help find relevant entities?)")
    print("=" * 100)
    header = (
        f"{'Category':<18s} {'#Q':>3s} "
        f"{'Avg GT':>7s} {'GT in Pool':>10s} {'GT+Spread':>10s} {'GT Gained':>10s} "
        f"{'GT Top10':>8s} "
        f"{'Avg Bonus GT':>12s} {'Avg Bonus !GT':>13s}"
    )
    print(header)
    print("-" * 100)

    for cat in categories:
        cat_diags = [d for d in diagnostics if d.category == cat]
        n = len(cat_diags)

        avg_gt = _mean([d.num_ground_truth for d in cat_diags])
        avg_gt_in_pool = _mean([d.gt_in_candidates_before for d in cat_diags])
        avg_gt_after = _mean([d.gt_in_candidates_after for d in cat_diags])
        avg_gt_gained = _mean(
            [d.gt_in_candidates_after - d.gt_in_candidates_before for d in cat_diags]
        )
        avg_gt_top10 = _mean([d.gt_in_top10 for d in cat_diags])
        avg_bonus_gt = _mean([_mean(d.bonus_for_gt) if d.bonus_for_gt else 0.0 for d in cat_diags])
        avg_bonus_non_gt = _mean(
            [_mean(d.bonus_for_non_gt) if d.bonus_for_non_gt else 0.0 for d in cat_diags]
        )

        print(
            f"{cat:<18s} {n:>3d} "
            f"{avg_gt:>7.1f} {avg_gt_in_pool:>10.1f} {avg_gt_after:>10.1f} {avg_gt_gained:>10.2f} "
            f"{avg_gt_top10:>8.1f} "
            f"{avg_bonus_gt:>12.4f} {avg_bonus_non_gt:>13.4f}"
        )

    # Total row
    print("-" * 100)
    avg_gt = _mean([d.num_ground_truth for d in all_diags])
    avg_gt_in_pool = _mean([d.gt_in_candidates_before for d in all_diags])
    avg_gt_after = _mean([d.gt_in_candidates_after for d in all_diags])
    avg_gt_gained = _mean([d.gt_in_candidates_after - d.gt_in_candidates_before for d in all_diags])
    avg_gt_top10 = _mean([d.gt_in_top10 for d in all_diags])
    avg_bonus_gt = _mean([_mean(d.bonus_for_gt) if d.bonus_for_gt else 0.0 for d in all_diags])
    avg_bonus_non_gt = _mean(
        [_mean(d.bonus_for_non_gt) if d.bonus_for_non_gt else 0.0 for d in all_diags]
    )
    print(
        f"{'TOTAL':<18s} {len(all_diags):>3d} "
        f"{avg_gt:>7.1f} {avg_gt_in_pool:>10.1f} {avg_gt_after:>10.1f} {avg_gt_gained:>10.2f} "
        f"{avg_gt_top10:>8.1f} "
        f"{avg_bonus_gt:>12.4f} {avg_bonus_non_gt:>13.4f}"
    )

    # -----------------------------------------------------------------------
    # Table 3: Score component breakdown for top-10 results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLE 3: AVERAGE SCORE COMPONENTS IN TOP-10 RESULTS")
    print("=" * 100)
    header = (
        f"{'Category':<18s} {'#Q':>3s} "
        f"{'Semantic':>10s} {'Activation':>10s} {'Spreading':>10s} "
        f"{'EdgeProx':>10s} {'Explore':>10s} {'Total':>10s}"
    )
    print(header)
    print("-" * 100)

    for cat in categories:
        cat_diags = [d for d in diagnostics if d.category == cat]
        n = len(cat_diags)

        all_sem, all_act, all_spread, all_edge, all_explore, all_total = [], [], [], [], [], []
        for d in cat_diags:
            for r in d.top10:
                all_sem.append(r.semantic_similarity)
                all_act.append(r.activation)
                all_spread.append(r.spreading)
                all_edge.append(r.edge_proximity)
                all_explore.append(r.exploration_bonus)
                all_total.append(r.score)

        print(
            f"{cat:<18s} {n:>3d} "
            f"{_mean(all_sem):>10.4f} {_mean(all_act):>10.4f} {_mean(all_spread):>10.4f} "
            f"{_mean(all_edge):>10.4f} {_mean(all_explore):>10.4f} {_mean(all_total):>10.4f}"
        )

    # Total row
    print("-" * 100)
    all_sem, all_act, all_spread, all_edge, all_explore, all_total = [], [], [], [], [], []
    for d in all_diags:
        for r in d.top10:
            all_sem.append(r.semantic_similarity)
            all_act.append(r.activation)
            all_spread.append(r.spreading)
            all_edge.append(r.edge_proximity)
            all_explore.append(r.exploration_bonus)
            all_total.append(r.score)
    print(
        f"{'TOTAL':<18s} {len(all_diags):>3d} "
        f"{_mean(all_sem):>10.4f} {_mean(all_act):>10.4f} {_mean(all_spread):>10.4f} "
        f"{_mean(all_edge):>10.4f} {_mean(all_explore):>10.4f} {_mean(all_total):>10.4f}"
    )

    # -----------------------------------------------------------------------
    # Table 4: Weighted score contribution breakdown (component * weight)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLE 4: WEIGHTED SCORE CONTRIBUTIONS (component * weight) IN TOP-10")
    print("  Full Engram weights: sem=0.40, act=0.25, spread=0.20, edge=0.15")
    print("  (Routing may change these per-query; values below are post-routing)")
    print("=" * 100)

    for cat in categories:
        cat_diags = [d for d in diagnostics if d.category == cat]
        n = len(cat_diags)

        # For weighted contributions we need to re-estimate from final scores
        # Since score = w_sem*sem + w_act*act + w_spread*spread + w_edge*edge + exploration,
        # and the weights may have been routed, we can back out approximate contributions
        # by using the raw component values * default Full Engram weights
        w_sem, w_act, w_spread, w_edge = 0.40, 0.25, 0.20, 0.15

        weighted_sem, weighted_act, weighted_spread, weighted_edge = [], [], [], []
        for d in cat_diags:
            for r in d.top10:
                weighted_sem.append(w_sem * r.semantic_similarity)
                weighted_act.append(w_act * r.activation)
                weighted_spread.append(w_spread * r.spreading)
                weighted_edge.append(w_edge * r.edge_proximity)

        total_contribution = (
            _mean(weighted_sem)
            + _mean(weighted_act)
            + _mean(weighted_spread)
            + _mean(weighted_edge)
        )
        if total_contribution > 0:
            pct_sem = _mean(weighted_sem) / total_contribution * 100
            pct_spread = _mean(weighted_spread) / total_contribution * 100
        else:
            pct_sem = pct_spread = 0.0

        print(
            f"  {cat:<18s} sem={_mean(weighted_sem):.4f} ({pct_sem:.1f}%)  "
            f"act={_mean(weighted_act):.4f}  "
            f"spread={_mean(weighted_spread):.4f} ({pct_spread:.1f}%)  "
            f"edge={_mean(weighted_edge):.4f}"
        )

    # -----------------------------------------------------------------------
    # Table 5: Detailed per-query dump for interesting cases
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLE 5: QUERIES WHERE SPREADING DISCOVERED GROUND-TRUTH ENTITIES")
    print("=" * 100)

    found_any = False
    for d in diagnostics:
        if d.discovered_in_gt > 0:
            found_any = True
            print(
                f'\n  [{d.category}] {d.query_id}: "{d.query_text[:80]}..."'
                if len(d.query_text) > 80
                else f'\n  [{d.category}] {d.query_id}: "{d.query_text}"'
            )
            print(
                f"    Seeds: {d.num_seeds}, Bonuses: {d.num_bonuses}, "
                f"Discovered: {d.num_discovered}, Discovered in GT: {d.discovered_in_gt}"
            )
            # Show top-3 results
            for i, r in enumerate(d.top10[:3]):
                print(
                    f"    Top {i + 1}: {r.node_id[:30]:<30s} "
                    f"score={r.score:.4f} sem={r.semantic_similarity:.4f} "
                    f"spread={r.spreading:.4f} edge={r.edge_proximity:.4f}"
                )

    if not found_any:
        print("  >> NO queries had spreading discover ground-truth entities <<")
        print("  This means spreading NEVER finds relevant entities that search missed.")

    # -----------------------------------------------------------------------
    # Table 6: Spreading bonus distribution
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLE 6: SPREADING BONUS VALUE DISTRIBUTION")
    print("=" * 100)

    all_bonuses = []
    for d in diagnostics:
        all_bonuses.extend(d.bonus_values)

    if all_bonuses:
        sorted_b = sorted(all_bonuses)
        print(f"  Total bonus entries across all queries: {len(all_bonuses)}")
        print(f"  Min:    {sorted_b[0]:.6f}")
        print(f"  P25:    {sorted_b[len(sorted_b) // 4]:.6f}")
        print(f"  Median: {_median(all_bonuses):.6f}")
        print(f"  P75:    {sorted_b[3 * len(sorted_b) // 4]:.6f}")
        print(f"  P95:    {sorted_b[int(0.95 * len(sorted_b))]:.6f}")
        print(f"  Max:    {sorted_b[-1]:.6f}")
        print(f"  Mean:   {_mean(all_bonuses):.6f}")

        # How many bonuses are > 0.1 (i.e. would contribute meaningfully at weight=0.20)?
        significant = [b for b in all_bonuses if b > 0.1]
        high = [b for b in all_bonuses if b > 0.5]
        capped = [b for b in all_bonuses if b >= 1.0]
        sig_pct = 100 * len(significant) / len(all_bonuses)
        high_pct = 100 * len(high) / len(all_bonuses)
        cap_pct = 100 * len(capped) / len(all_bonuses)
        print(f"  Bonuses > 0.1 (meaningful): {len(significant)} ({sig_pct:.1f}%)")
        print(f"  Bonuses > 0.5 (strong):     {len(high)} ({high_pct:.1f}%)")
        print(f"  Bonuses >= 1.0 (capped):    {len(capped)} ({cap_pct:.1f}%)")
    else:
        print("  No spreading bonuses were generated at all!")

    # -----------------------------------------------------------------------
    # Table 7: Seed energy analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLE 7: SEED ENERGY ANALYSIS")
    print("=" * 100)

    all_energies = []
    for d in diagnostics:
        all_energies.extend(d.seed_energies)

    if all_energies:
        sorted_e = sorted(all_energies)
        print(f"  Total seeds across all queries: {len(all_energies)}")
        print(f"  Min:    {sorted_e[0]:.6f}")
        print(f"  Median: {_median(all_energies):.6f}")
        print(f"  Mean:   {_mean(all_energies):.6f}")
        print(f"  P95:    {sorted_e[int(0.95 * len(sorted_e))]:.6f}")
        print(f"  Max:    {sorted_e[-1]:.6f}")

        # Energy is sem_sim * max(act, 0.15)
        # If most seeds have low energy, spreading starts weak
        weak = [e for e in all_energies if e < 0.1]
        moderate = [e for e in all_energies if 0.1 <= e < 0.3]
        strong = [e for e in all_energies if e >= 0.3]
        weak_pct = 100 * len(weak) / len(all_energies)
        mod_pct = 100 * len(moderate) / len(all_energies)
        str_pct = 100 * len(strong) / len(all_energies)
        print(f"  Seeds with energy < 0.1 (weak):      {len(weak)} ({weak_pct:.1f}%)")
        print(f"  Seeds with energy 0.1-0.3 (moderate): {len(moderate)} ({mod_pct:.1f}%)")
        print(f"  Seeds with energy >= 0.3 (strong):    {len(strong)} ({str_pct:.1f}%)")
    else:
        print("  No seeds were generated!")

    # -----------------------------------------------------------------------
    # Final diagnosis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DIAGNOSIS SUMMARY")
    print("=" * 100)

    total_discovered_gt = sum(d.discovered_in_gt for d in diagnostics)
    total_discovered = sum(d.num_discovered for d in diagnostics)
    total_queries = len(diagnostics)
    avg_bonuses_count = _mean([d.num_bonuses for d in diagnostics])
    avg_bonus_val_all = _mean(
        [_mean(d.bonus_values) if d.bonus_values else 0.0 for d in diagnostics]
    )

    # Compute average weighted spreading contribution
    all_spread_weighted = []
    all_total_scores = []
    for d in diagnostics:
        for r in d.top10:
            all_spread_weighted.append(0.20 * r.spreading)
            all_total_scores.append(r.score)

    avg_spread_contribution = _mean(all_spread_weighted)
    avg_total_score = _mean(all_total_scores)
    spread_pct = (avg_spread_contribution / avg_total_score * 100) if avg_total_score > 0 else 0

    print(f"""
  Across {total_queries} queries:
    - Spreading produced bonuses for {avg_bonuses_count:.0f} entities/query on average
    - Average bonus value: {avg_bonus_val_all:.4f} (clamped to [0,1] in scoring)
    - Total discovered entities (not in search pool): {total_discovered}
    - Discovered entities that were in ground truth: {total_discovered_gt}
    - Average spreading contribution to final score: \
{avg_spread_contribution:.4f} ({spread_pct:.1f}% of total)

  KEY FINDINGS:
""")

    if total_discovered_gt == 0:
        print("    1. Spreading NEVER discovers ground-truth entities that search missed.")
        print(
            "       => The graph traversal does not reach relevant entities"
            " outside the search pool."
        )
    else:
        print(
            f"    1. Spreading discovered {total_discovered_gt} "
            "ground-truth entities across all queries."
        )

    if avg_bonus_val_all < 0.1:
        print(f"    2. Average spreading bonus ({avg_bonus_val_all:.4f}) is very small.")
        print("       => Even with weight=0.20, spreading contributes < 2% of total score.")
    else:
        print(f"    2. Average spreading bonus ({avg_bonus_val_all:.4f}) is moderate.")

    gt_bonus_vals = []
    non_gt_bonus_vals = []
    for d in diagnostics:
        gt_bonus_vals.extend(d.bonus_for_gt)
        non_gt_bonus_vals.extend(d.bonus_for_non_gt)

    avg_gt_bonus = _mean(gt_bonus_vals) if gt_bonus_vals else 0.0
    avg_non_gt_bonus = _mean(non_gt_bonus_vals) if non_gt_bonus_vals else 0.0

    if avg_gt_bonus <= avg_non_gt_bonus:
        print(
            f"    3. Ground-truth entities get LOWER spreading bonuses ({avg_gt_bonus:.4f}) "
            f"than non-GT ({avg_non_gt_bonus:.4f})."
        )
        print("       => Spreading does not preferentially boost relevant entities.")
    else:
        ratio = avg_gt_bonus / avg_non_gt_bonus if avg_non_gt_bonus > 0 else float("inf")
        print(
            f"    3. Ground-truth entities get higher spreading bonuses ({avg_gt_bonus:.4f}) "
            f"than non-GT ({avg_non_gt_bonus:.4f}), ratio={ratio:.2f}x."
        )

    # Check if seeds are mostly already top-ranked by search
    seeds_total = 0
    for d in diagnostics:
        for _, _ in zip(range(d.num_seeds), d.seed_energies):
            seeds_total += 1
    print(
        f"    4. Average seeds per query: {_mean([d.num_seeds for d in diagnostics]):.0f}. "
        f"Spreading radiates from these through the graph."
    )

    # Check how many unique entities spreading reaches
    queries_with_zero_bonus = sum(1 for d in diagnostics if d.num_bonuses == 0)
    if queries_with_zero_bonus > 0:
        print(
            f"    5. {queries_with_zero_bonus}/{total_queries} queries had ZERO spreading bonuses "
            "(no neighbors found in graph)."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 100)
    print("SPREADING ACTIVATION DIAGNOSTIC")
    print("Generates 1k entity corpus, runs Full Engram pipeline, instruments spreading")
    print("=" * 100)

    # 1. Generate corpus
    print("\n1. Generating corpus (seed=42, 1000 entities) ...")
    corpus_gen = CorpusGenerator(seed=42, total_entities=1000)
    corpus: CorpusSpec = corpus_gen.generate()
    print(
        f"   Corpus: {len(corpus.entities)} entities, "
        f"{len(corpus.relationships)} relationships, "
        f"{len(corpus.access_events)} access events, "
        f"{len(corpus.ground_truth)} ground-truth queries"
    )

    # 2. Create temp stores
    print("\n2. Creating temp stores ...")
    tmp_dir = tempfile.mkdtemp(prefix="engram_diag_")
    db_path = str(Path(tmp_dir) / "diag.db")

    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()

    activation_store = MemoryActivationStore(cfg=ActivationConfig())

    search_index = FTS5SearchIndex(db_path)
    await search_index.initialize(db=graph_store._db)

    # 3. Load corpus
    print("   Loading corpus into stores ...")
    load_elapsed = await corpus_gen.load(
        corpus,
        graph_store,
        activation_store,
        search_index,
    )
    print(f"   Loaded in {load_elapsed:.2f}s")

    # 4. Run instrumented retrieval for each query
    cfg = ActivationConfig(
        multi_pool_enabled=True,
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    )

    benchmark_now = corpus.metadata.get("generated_at", time.time())
    n_entities = len(corpus.entities)

    diagnostics: list[QueryDiagnostic] = []

    print(f"\n3. Running instrumented retrieval on {len(corpus.ground_truth)} queries ...")
    for qi, query in enumerate(corpus.ground_truth):
        gt_ids = set(query.relevant_entities.keys())
        if query.relevant_episodes:
            gt_ids.update(query.relevant_episodes.keys())

        diag = QueryDiagnostic(
            query_id=query.query_id,
            category=query.category,
            query_text=query.query_text,
        )
        diag.num_ground_truth = len(gt_ids)

        await instrumented_retrieve(
            query_text=query.query_text,
            ground_truth_ids=gt_ids,
            group_id="benchmark",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            now=benchmark_now,
            total_entities=n_entities,
            diag=diag,
        )

        diagnostics.append(diag)

        # Progress indicator every 10 queries
        if (qi + 1) % 10 == 0 or qi == len(corpus.ground_truth) - 1:
            print(f"   {qi + 1}/{len(corpus.ground_truth)} queries done")

    # 5. Print summary
    print_summary(diagnostics)

    # Cleanup
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
