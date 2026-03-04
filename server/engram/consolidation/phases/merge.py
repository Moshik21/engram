"""Entity merge phase: find and merge near-duplicate entities."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.extraction.resolver import compute_similarity
from engram.models.consolidation import CycleContext, MergeRecord, PhaseResult

logger = logging.getLogger(__name__)

_MERGE_JUDGE_PROMPT = (
    "You are a knowledge graph entity deduplication judge.\n"
    "Given two entities, determine if they refer to the same real-world entity.\n"
    "Respond with JSON only: "
    '{"verdict": "merge"|"keep_separate"|"uncertain", "reason": "brief explanation"}'
)

_MERGE_JUDGE_SYSTEM_CACHED = [
    {
        "type": "text",
        "text": _MERGE_JUDGE_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]

_MERGE_ESCALATION_PROMPT = (
    "You are a senior knowledge graph entity deduplication reviewer.\n"
    "A lower-tier model was uncertain about merging these entities.\n"
    "Apply strict judgment. Respond with JSON only:\n"
    '{"verdict": "merge"|"keep_separate", "reason": "brief explanation"}\n'
    "You MUST choose merge or keep_separate. Do not respond with uncertain."
)

_MERGE_ESCALATION_SYSTEM_CACHED = [
    {
        "type": "text",
        "text": _MERGE_ESCALATION_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


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

    def __init__(self, llm_client=None):
        self._llm_client = llm_client

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
            return PhaseResult(
                phase=self.name, items_processed=0, items_affected=0, duration_ms=_elapsed_ms(t0)
            ), []

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
                        sub_block,
                        threshold,
                        same_type_boost,
                        require_same_type,
                        union,
                    )
            else:
                pairs_checked += _compare_block(
                    block_entities,
                    threshold,
                    same_type_boost,
                    require_same_type,
                    union,
                )

        # --- LLM-assisted merge for soft-zone pairs ---
        if cfg.consolidation_merge_llm_enabled and not dry_run:
            soft_zone_pairs = self._collect_soft_zone_pairs(
                type_blocks,
                block_size_limit,
                cfg.consolidation_merge_soft_threshold,
                threshold,
                require_same_type,
            )
            if soft_zone_pairs:
                llm_merges = self._run_llm_merge_pass(
                    soft_zone_pairs, cfg, dry_run,
                )
                for ea_id, eb_id in llm_merges:
                    union(ea_id, eb_id)

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
                        survivor.id,
                        loser.id,
                        group_id,
                    )
                    # Merge activation histories
                    surv_state = await activation_store.get_activation(survivor.id)
                    loser_state = await activation_store.get_activation(loser.id)
                    if surv_state and loser_state:
                        merged_history = sorted(
                            set(surv_state.access_history + loser_state.access_history),
                            reverse=True,
                        )[: cfg.max_history_size]
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

                merge_records.append(
                    MergeRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        keep_id=survivor.id,
                        remove_id=loser.id,
                        keep_name=survivor.name,
                        remove_name=loser.name,
                        similarity=round(sim, 4),
                        relationships_transferred=rels_transferred,
                    )
                )

        return PhaseResult(
            phase=self.name,
            items_processed=pairs_checked,
            items_affected=len(merge_records),
            duration_ms=_elapsed_ms(t0),
        ), merge_records

    @staticmethod
    def _collect_soft_zone_pairs(
        type_blocks: dict[str, list],
        block_size_limit: int,
        soft_threshold: float,
        hard_threshold: float,
        require_same_type: bool,
    ) -> list[tuple]:
        """Find pairs in [soft_threshold, hard_threshold) similarity range."""
        same_type_boost = 0.03
        soft_pairs: list[tuple] = []

        for block_type, block_entities in type_blocks.items():
            entities = block_entities
            if len(entities) > block_size_limit:
                # Only check first block_size_limit to avoid O(n²) explosion
                entities = entities[:block_size_limit]

            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    ea, eb = entities[i], entities[j]
                    sim = compute_similarity(ea.name, eb.name)
                    if require_same_type and ea.entity_type == eb.entity_type:
                        sim += same_type_boost
                    if soft_threshold <= sim < hard_threshold:
                        soft_pairs.append((ea, eb, sim))

        return soft_pairs

    def _run_llm_merge_pass(
        self,
        soft_pairs: list[tuple],
        cfg: ActivationConfig,
        dry_run: bool,
    ) -> list[tuple[str, str]]:
        """Judge soft-zone pairs via LLM, return pairs to merge."""
        client = self._llm_client
        if client is None:
            try:
                import anthropic

                client = anthropic.Anthropic()
            except Exception:
                logger.warning("Could not create Anthropic client for merge LLM")
                return []

        approved_merges: list[tuple[str, str]] = []

        for ea, eb, sim in soft_pairs:
            try:
                user_msg = (
                    f"Entity A: {ea.name} (type: {ea.entity_type})\n"
                    f"Entity B: {eb.name} (type: {eb.entity_type})\n"
                    f"String similarity: {sim:.4f}"
                )
                response = client.messages.create(
                    model=cfg.consolidation_merge_llm_model,
                    max_tokens=256,
                    system=_MERGE_JUDGE_SYSTEM_CACHED,
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = response.content[0].text.strip()
                parsed = json.loads(text)
                verdict = parsed.get("verdict", "keep_separate")

                if verdict == "merge":
                    approved_merges.append((ea.id, eb.id))
                elif verdict == "uncertain" and cfg.consolidation_merge_escalation_enabled:
                    # Escalate to Sonnet
                    esc_verdict = self._escalate_merge(
                        ea, eb, sim, cfg, client,
                    )
                    if esc_verdict == "merge":
                        approved_merges.append((ea.id, eb.id))
            except Exception as exc:
                logger.warning("Merge LLM judge failed for %s/%s: %s", ea.name, eb.name, exc)

        return approved_merges

    def _escalate_merge(
        self,
        ea,
        eb,
        sim: float,
        cfg: ActivationConfig,
        client,
    ) -> str:
        """Escalate uncertain merge verdict to Sonnet."""
        try:
            user_msg = (
                f"Entity A: {ea.name} (type: {ea.entity_type})\n"
                f"Entity B: {eb.name} (type: {eb.entity_type})\n"
                f"String similarity: {sim:.4f}\n"
                f"Previous verdict: uncertain"
            )
            response = client.messages.create(
                model=cfg.consolidation_merge_escalation_model,
                max_tokens=256,
                system=_MERGE_ESCALATION_SYSTEM_CACHED,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text.strip()
            parsed = json.loads(text)
            verdict = parsed.get("verdict", "keep_separate")
            if verdict not in ("merge", "keep_separate"):
                verdict = "keep_separate"
            return verdict
        except Exception as exc:
            logger.warning("Merge escalation failed for %s/%s: %s", ea.name, eb.name, exc)
            return "keep_separate"


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
