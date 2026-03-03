"""Entity merge phase: find and merge near-duplicate entities."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.extraction.resolver import compute_similarity
from engram.models.consolidation import CycleContext, MergeRecord, PhaseResult

logger = logging.getLogger(__name__)


def _compare_block(
    entities: list,
    threshold: float,
    same_type_boost: float,
    require_same_type: bool,
    union_fn,
) -> int:
    """Run pairwise fuzzy comparisons within a block and union matches.

    Returns the number of pairs checked.
    """
    pairs_checked = 0
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            ea, eb = entities[i], entities[j]
            sim = compute_similarity(ea.name, eb.name)
            if require_same_type and ea.entity_type == eb.entity_type:
                sim += same_type_boost
            pairs_checked += 1
            if sim >= threshold:
                union_fn(ea.id, eb.id)
    return pairs_checked


class EntityMergePhase(ConsolidationPhase):
    """Find near-duplicate entities via fuzzy matching and merge them."""

    @property
    def name(self) -> str:
        return "merge"

    async def execute(
        self,
        group_id: str,
        graph_store,
        activation_store,
        search_index,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool = False,
        context: CycleContext | None = None,
    ) -> PhaseResult:
        t0 = time.perf_counter()
        threshold = cfg.consolidation_merge_threshold
        max_merges = cfg.consolidation_merge_max_per_cycle
        require_same_type = cfg.consolidation_merge_require_same_type
        block_size_limit = cfg.consolidation_merge_block_size

        # Load all active entities
        entities = await graph_store.find_entities(group_id=group_id, limit=100000)
        if not entities:
            return PhaseResult(phase=self.name, items_processed=0, items_affected=0,
                               duration_ms=_elapsed_ms(t0)), []

        # Group by type for blocking
        type_blocks: dict[str, list] = defaultdict(list)
        if require_same_type:
            for e in entities:
                type_blocks[e.entity_type].append(e)
        else:
            type_blocks["_all"] = list(entities)

        # Find merge candidates via pairwise comparison within blocks
        # Union-find for transitive merges
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        pairs_checked = 0
        same_type_boost = 0.03

        for block_type, block_entities in type_blocks.items():
            if len(block_entities) > block_size_limit:
                # Prefix sub-blocking: partition by first 2 chars of name
                prefix_blocks: dict[str, list] = defaultdict(list)
                for e in block_entities:
                    prefix = e.name[:2].lower() if e.name else ""
                    prefix_blocks[prefix].append(e)

                for sub_block in prefix_blocks.values():
                    pairs_checked += _compare_block(
                        sub_block, threshold, same_type_boost,
                        require_same_type, union,
                    )
            else:
                pairs_checked += _compare_block(
                    block_entities, threshold, same_type_boost,
                    require_same_type, union,
                )

        # Collect merge groups
        groups: dict[str, list] = defaultdict(list)
        for e in entities:
            root = find(e.id)
            groups[root].append(e)

        # Execute merges
        merge_records: list[MergeRecord] = []
        for root, members in groups.items():
            if len(members) < 2:
                continue
            if len(merge_records) >= max_merges:
                break

            # Survivor: highest access_count, tiebreak earliest created_at
            members.sort(key=lambda e: (-e.access_count, e.created_at))
            survivor = members[0]

            for loser in members[1:]:
                if len(merge_records) >= max_merges:
                    break

                sim = compute_similarity(survivor.name, loser.name)
                if require_same_type and survivor.entity_type == loser.entity_type:
                    sim += same_type_boost

                rels_transferred = 0
                if not dry_run:
                    rels_transferred = await graph_store.merge_entities(
                        survivor.id, loser.id, group_id,
                    )
                    # Merge activation histories
                    surv_state = await activation_store.get_activation(survivor.id)
                    loser_state = await activation_store.get_activation(loser.id)
                    if surv_state and loser_state:
                        merged_history = sorted(
                            set(surv_state.access_history + loser_state.access_history),
                            reverse=True,
                        )[:cfg.max_history_size]
                        surv_state.access_history = merged_history
                        surv_state.access_count += loser_state.access_count
                        surv_state.consolidated_strength += loser_state.consolidated_strength
                        await activation_store.set_activation(survivor.id, surv_state)

                    # Clean up loser from activation + search
                    await activation_store.clear_activation(loser.id)
                    await search_index.remove(loser.id)

                    # Track affected entities for reindex
                    if context is not None:
                        context.merge_survivor_ids.add(survivor.id)
                        context.affected_entity_ids.add(survivor.id)

                merge_records.append(MergeRecord(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    keep_id=survivor.id,
                    remove_id=loser.id,
                    keep_name=survivor.name,
                    remove_name=loser.name,
                    similarity=round(sim, 4),
                    relationships_transferred=rels_transferred,
                ))

        return PhaseResult(
            phase=self.name,
            items_processed=pairs_checked,
            items_affected=len(merge_records),
            duration_ms=_elapsed_ms(t0),
        ), merge_records


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
