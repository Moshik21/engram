"""Entity merge phase: find and merge near-duplicate entities."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import cast

import numpy as np

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.entity_dedup_policy import (
    IDENTIFIER_ENTITY_TYPE,
    DedupPolicyDecision,
    dedup_policy,
    policy_aware_similarity,
    policy_features,
    should_enqueue_identifier_review,
    should_promote_entity_type_to_identifier,
)
from engram.extraction.resolver import compute_similarity
from engram.models.consolidation import (
    CycleContext,
    DecisionOutcomeLabel,
    DecisionTrace,
    IdentifierReviewRecord,
    MergeRecord,
    PhaseResult,
)

logger = logging.getLogger(__name__)

_MERGE_JUDGE_PROMPT = (
    "You are a knowledge graph entity deduplication judge.\n"
    "Given one or more entity pairs, determine if each pair refers to the same real-world entity.\n"
    "Respond with a JSON array of verdicts, one per pair:\n"
    '[{"pair": 1, "verdict": "merge"|"keep_separate"|"uncertain", '
    '"reason": "brief explanation"}, ...]'
)

_MERGE_BATCH_SIZE = 5

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
    merge_decisions: dict[tuple[str, str], dict],
    path: str,
    context: CycleContext | None,
    cycle_id: str,
    group_id: str,
    identifier_review_records: list[IdentifierReviewRecord],
    identifier_review_pairs: set[tuple[str, str]],
    identifier_review_enabled: bool,
    identifier_review_min_similarity: float,
    identifier_review_max_per_cycle: int,
) -> int:
    """Run pairwise fuzzy comparisons within a block and union matches.

    Returns the number of pairs checked.
    """
    pairs_checked = 0
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            ea, eb = entities[i], entities[j]
            decision, sim = _policy_similarity(ea, eb, same_type_boost)
            raw_similarity = compute_similarity(ea.name, eb.name)
            pairs_checked += 1
            if not decision.allowed:
                _maybe_queue_identifier_review(
                    identifier_review_records,
                    identifier_review_pairs,
                    enabled=identifier_review_enabled,
                    max_records=identifier_review_max_per_cycle,
                    min_similarity=identifier_review_min_similarity,
                    cycle_id=cycle_id,
                    group_id=group_id,
                    entity_a=ea,
                    entity_b=eb,
                    decision=decision,
                    raw_similarity=raw_similarity,
                    adjusted_similarity=sim,
                    decision_source=path,
                )
                _add_policy_trace(
                    context,
                    cycle_id,
                    group_id,
                    ea,
                    eb,
                    decision,
                    path=path,
                    decision_label="keep_separate",
                    threshold_band="hard_reject",
                    confidence=0.0,
                    extra_features={"name_similarity": 0.0},
                )
                continue

            if decision.exact_identifier_match:
                _add_policy_trace(
                    context,
                    cycle_id,
                    group_id,
                    ea,
                    eb,
                    decision,
                    path=path,
                    decision_label="merge",
                    threshold_band="accepted",
                    confidence=1.0,
                    extra_features={
                        "name_similarity": round(_name_similarity(ea, eb, same_type_boost), 4),
                    },
                )

            if sim >= threshold:
                union_fn(ea.id, eb.id)
                _remember_merge_decision(
                    merge_decisions,
                    ea.id,
                    eb.id,
                    confidence=sim,
                    source=(
                        "identifier_policy"
                        if decision.exact_identifier_match
                        else "fuzzy_threshold"
                    ),
                    reason=_decision_reason(decision, "threshold_met"),
                )
    return pairs_checked


def _types_allowed(entity_a, entity_b, require_same_type: bool) -> bool:
    """Apply same-type enforcement consistently across all candidate paths."""
    return (not require_same_type) or entity_a.entity_type == entity_b.entity_type


def _pair_key(entity_a_id: str, entity_b_id: str) -> tuple[str, str]:
    return cast(tuple[str, str], tuple(sorted((entity_a_id, entity_b_id))))


def _name_similarity(entity_a, entity_b, same_type_boost: float) -> float:
    sim = compute_similarity(entity_a.name, entity_b.name)
    if entity_a.entity_type == entity_b.entity_type:
        sim = min(sim + same_type_boost, 1.0)
    return sim


def _policy_similarity(
    entity_a,
    entity_b,
    same_type_boost: float,
) -> tuple[DedupPolicyDecision, float]:
    decision, sim = policy_aware_similarity(entity_a.name, entity_b.name, compute_similarity)
    if (
        decision.allowed
        and not decision.exact_identifier_match
        and entity_a.entity_type == entity_b.entity_type
    ):
        sim = min(sim + same_type_boost, 1.0)
    return decision, sim


def _decision_reason(decision: DedupPolicyDecision, fallback: str) -> str:
    return decision.reason if decision.reason != "natural_language_fallback" else fallback


def _remember_merge_decision(
    merge_decisions: dict[tuple[str, str], dict],
    entity_a_id: str,
    entity_b_id: str,
    *,
    confidence: float | None,
    source: str,
    reason: str | None,
) -> None:
    key = _pair_key(entity_a_id, entity_b_id)
    existing = merge_decisions.get(key)
    next_confidence = round(confidence, 4) if confidence is not None else None
    if (
        existing is not None
        and existing.get("confidence") is not None
        and next_confidence is not None
    ):
        if existing["confidence"] > next_confidence:
            return
    merge_decisions[key] = {
        "confidence": next_confidence,
        "source": source,
        "reason": reason,
    }


def _append_identifier_review(
    records: list[IdentifierReviewRecord],
    seen_pairs: set[tuple[str, str]],
    *,
    max_records: int,
    cycle_id: str,
    group_id: str,
    entity_a,
    entity_b,
    decision: DedupPolicyDecision,
    raw_similarity: float,
    adjusted_similarity: float | None,
    decision_source: str,
) -> None:
    if len(records) >= max_records:
        return
    key = _pair_key(entity_a.id, entity_b.id)
    if key in seen_pairs:
        return
    seen_pairs.add(key)
    records.append(
        IdentifierReviewRecord(
            cycle_id=cycle_id,
            group_id=group_id,
            entity_a_id=entity_a.id,
            entity_b_id=entity_b.id,
            entity_a_name=entity_a.name,
            entity_b_name=entity_b.name,
            entity_a_type=entity_a.entity_type,
            entity_b_type=entity_b.entity_type,
            raw_similarity=round(raw_similarity, 4),
            adjusted_similarity=(
                round(adjusted_similarity, 4) if adjusted_similarity is not None else None
            ),
            decision_source=decision_source,
            decision_reason=decision.reason,
            entity_a_regime=decision.left.regime.value,
            entity_b_regime=decision.right.regime.value,
            canonical_identifier_a=decision.left.canonical_code,
            canonical_identifier_b=decision.right.canonical_code,
            metadata=policy_features(decision),
        )
    )


def _add_policy_trace(
    context: CycleContext | None,
    cycle_id: str,
    group_id: str,
    entity_a,
    entity_b,
    decision: DedupPolicyDecision,
    *,
    path: str,
    decision_label: str,
    threshold_band: str,
    confidence: float | None,
    extra_features: dict | None = None,
) -> None:
    if context is None:
        return
    features = policy_features(decision)
    if extra_features:
        features.update(extra_features)
    context.add_decision_trace(
        DecisionTrace(
            cycle_id=cycle_id,
            group_id=group_id,
            phase="merge",
            candidate_type="entity_pair",
            candidate_id=_pair_candidate_id(entity_a.id, entity_b.id),
            decision=decision_label,
            decision_source="constraint",
            confidence=confidence,
            threshold_band=threshold_band,
            features=features,
            constraints_hit=[decision.reason],
            metadata={"path": path},
        )
    )


def _maybe_queue_identifier_review(
    records: list[IdentifierReviewRecord],
    seen_pairs: set[tuple[str, str]],
    *,
    enabled: bool,
    max_records: int,
    min_similarity: float,
    cycle_id: str,
    group_id: str,
    entity_a,
    entity_b,
    decision: DedupPolicyDecision,
    raw_similarity: float,
    adjusted_similarity: float | None,
    decision_source: str,
) -> None:
    if not enabled:
        return
    if not should_enqueue_identifier_review(
        decision,
        raw_similarity,
        min_similarity=min_similarity,
    ):
        return
    _append_identifier_review(
        records,
        seen_pairs,
        max_records=max_records,
        cycle_id=cycle_id,
        group_id=group_id,
        entity_a=entity_a,
        entity_b=entity_b,
        decision=decision,
        raw_similarity=raw_similarity,
        adjusted_similarity=adjusted_similarity,
        decision_source=decision_source,
    )


class EntityMergePhase(ConsolidationPhase):
    """Find near-duplicate entities via fuzzy matching and merge them."""

    def __init__(self, llm_client=None):
        self._llm_client = llm_client
        # Cache LLM "keep_separate" verdicts across cycles: frozenset({id_a, id_b})
        self._keep_separate_cache: set[frozenset[str]] = set()

    @property
    def name(self) -> str:
        return "merge"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        methods = {"find_entities", "merge_entities", "update_entity"}
        if cfg.consolidation_merge_multi_signal_enabled:
            methods.update({"get_active_neighbors_with_weights", "get_episode_cooccurrence_count"})
            if cfg.consolidation_merge_structural_min_neighbors > 0:
                methods.add("find_structural_merge_candidates")
        return methods

    def required_activation_store_methods(self, cfg: ActivationConfig) -> set[str]:
        return {"get_activation", "set_activation", "clear_activation"}

    def required_search_index_methods(self, cfg: ActivationConfig) -> set[str]:
        methods = {"remove"}
        if cfg.consolidation_merge_use_embeddings or cfg.consolidation_merge_multi_signal_enabled:
            methods.add("get_entity_embeddings")
        return methods

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
    ) -> tuple[PhaseResult, list[object]]:
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
        merge_decisions: dict[tuple[str, str], dict] = {}
        identifier_review_enabled = getattr(cfg, "consolidation_identifier_review_enabled", True)
        identifier_review_min_similarity = getattr(
            cfg,
            "consolidation_identifier_review_min_similarity",
            0.8,
        )
        identifier_review_max_per_cycle = getattr(
            cfg,
            "consolidation_identifier_review_max_per_cycle",
            100,
        )
        identifier_review_records: list[IdentifierReviewRecord] = []
        identifier_review_pairs: set[tuple[str, str]] = set()

        pairs_checked += self._merge_exact_identifier_aliases(
            entities,
            same_type_boost,
            union,
            merge_decisions,
            context,
            cycle_id,
            group_id,
        )

        # --- Try embedding-based ANN candidate pre-filtering ---
        used_embeddings = False
        ann_llm_candidates: list[tuple] = []  # Semantic dupes needing LLM review
        if cfg.consolidation_merge_use_embeddings:
            ann_pairs = await self._find_candidates_via_embeddings(
                entities,
                search_index,
                group_id,
                cfg,
                require_same_type=require_same_type,
            )
            if ann_pairs is not None:
                used_embeddings = True
                # Only fuzzy-match the ANN candidates (not all N² pairs)
                for ea, eb in ann_pairs:
                    if not _types_allowed(ea, eb, require_same_type):
                        if context is not None:
                            context.add_decision_trace(
                                DecisionTrace(
                                    cycle_id=cycle_id,
                                    group_id=group_id,
                                    phase=self.name,
                                    candidate_type="entity_pair",
                                    candidate_id=_pair_candidate_id(ea.id, eb.id),
                                    decision="keep_separate",
                                    decision_source="constraint",
                                    threshold_band="hard_reject",
                                    constraints_hit=["require_same_type"],
                                    metadata={"path": "ann"},
                                )
                            )
                        continue
                    decision, sim = _policy_similarity(ea, eb, same_type_boost)
                    raw_similarity = compute_similarity(ea.name, eb.name)
                    pairs_checked += 1
                    if not decision.allowed:
                        _maybe_queue_identifier_review(
                            identifier_review_records,
                            identifier_review_pairs,
                            enabled=identifier_review_enabled,
                            max_records=identifier_review_max_per_cycle,
                            min_similarity=identifier_review_min_similarity,
                            cycle_id=cycle_id,
                            group_id=group_id,
                            entity_a=ea,
                            entity_b=eb,
                            decision=decision,
                            raw_similarity=raw_similarity,
                            adjusted_similarity=sim,
                            decision_source="ann",
                        )
                        _add_policy_trace(
                            context,
                            cycle_id,
                            group_id,
                            ea,
                            eb,
                            decision,
                            path="ann",
                            decision_label="keep_separate",
                            threshold_band="hard_reject",
                            confidence=0.0,
                            extra_features={"name_similarity": 0.0},
                        )
                        continue
                    if decision.exact_identifier_match:
                        _add_policy_trace(
                            context,
                            cycle_id,
                            group_id,
                            ea,
                            eb,
                            decision,
                            path="ann",
                            decision_label="merge",
                            threshold_band="accepted",
                            confidence=1.0,
                            extra_features={
                                "name_similarity": round(
                                    _name_similarity(ea, eb, same_type_boost),
                                    4,
                                ),
                            },
                        )
                    if sim >= threshold:
                        union(ea.id, eb.id)
                        _remember_merge_decision(
                            merge_decisions,
                            ea.id,
                            eb.id,
                            confidence=sim,
                            source=(
                                "identifier_policy"
                                if decision.exact_identifier_match
                                else "ann_fuzzy"
                            ),
                            reason=_decision_reason(decision, "threshold_met"),
                        )
                    else:
                        # Embedding says similar but names don't match well
                        # — route to LLM judge for semantic dedup
                        ann_llm_candidates.append((ea, eb, sim))

        # --- Fallback: O(N²) type-blocked pairwise comparison ---
        if not used_embeddings:
            type_blocks: dict[str, list] = defaultdict(list)
            if require_same_type:
                for e in entities:
                    type_blocks[e.entity_type].append(e)
            else:
                type_blocks["_all"] = list(entities)

            for block_type, block_entities in type_blocks.items():
                if len(block_entities) > block_size_limit:
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
                            merge_decisions,
                            "fuzzy_threshold",
                            context,
                            cycle_id,
                            group_id,
                            identifier_review_records,
                            identifier_review_pairs,
                            identifier_review_enabled,
                            identifier_review_min_similarity,
                            identifier_review_max_per_cycle,
                        )
                else:
                    pairs_checked += _compare_block(
                        block_entities,
                        threshold,
                        same_type_boost,
                        require_same_type,
                        union,
                        merge_decisions,
                        "fuzzy_threshold",
                        context,
                        cycle_id,
                        group_id,
                        identifier_review_records,
                        identifier_review_pairs,
                        identifier_review_enabled,
                        identifier_review_min_similarity,
                        identifier_review_max_per_cycle,
                    )

        # --- Multi-signal scoring (replaces LLM judge when enabled) ---
        if cfg.consolidation_merge_multi_signal_enabled:
            from engram.consolidation.scorers.merge_scorer import score_merge_pair

            multi_signal_candidates: list[tuple] = list(ann_llm_candidates)
            if not used_embeddings:
                type_blocks_ms: dict[str, list] = defaultdict(list)
                if require_same_type:
                    for e in entities:
                        type_blocks_ms[e.entity_type].append(e)
                else:
                    type_blocks_ms["_all"] = list(entities)

                soft_zone_pairs = self._collect_soft_zone_pairs(
                    type_blocks_ms,
                    block_size_limit,
                    cfg.consolidation_merge_soft_threshold,
                    threshold,
                    require_same_type,
                )
                multi_signal_candidates.extend(soft_zone_pairs)

            # Filter cached keep_separate
            uncached = [
                (ea, eb, sim)
                for ea, eb, sim in multi_signal_candidates
                if frozenset({ea.id, eb.id}) not in self._keep_separate_cache
            ]

            ce_enabled = getattr(cfg, "consolidation_cross_encoder_enabled", True)
            for ea, eb, _sim in uncached:
                verdict, conf, signals = await score_merge_pair(
                    ea,
                    eb,
                    search_index,
                    graph_store,
                    group_id,
                    cross_encoder_enabled=ce_enabled,
                )
                if context is not None:
                    reason = signals.get("reason")
                    constraints = [reason] if reason else []
                    context.add_decision_trace(
                        DecisionTrace(
                            cycle_id=cycle_id,
                            group_id=group_id,
                            phase=self.name,
                            candidate_type="entity_pair",
                            candidate_id=_pair_candidate_id(ea.id, eb.id),
                            decision=verdict,
                            decision_source="multi_signal",
                            confidence=conf,
                            threshold_band=_decision_band(verdict),
                            features=signals,
                            constraints_hit=constraints,
                        )
                    )
                if verdict == "merge":
                    union(ea.id, eb.id)
                    _remember_merge_decision(
                        merge_decisions,
                        ea.id,
                        eb.id,
                        confidence=conf,
                        source="multi_signal",
                        reason=signals.get("reason") or "multi_signal_accepted",
                    )
                else:
                    self._keep_separate_cache.add(frozenset({ea.id, eb.id}))

        # --- LLM-assisted merge for soft-zone + ANN candidates ---
        elif cfg.consolidation_merge_llm_enabled and not dry_run:
            all_llm_candidates: list[tuple] = []

            # ANN candidates that had high embedding similarity but low name match
            # Cap at consolidation_merge_ann_llm_max to control LLM costs
            if ann_llm_candidates:
                ann_cap = cfg.consolidation_merge_ann_llm_max
                all_llm_candidates.extend(ann_llm_candidates[:ann_cap])
                if len(ann_llm_candidates) > ann_cap:
                    logger.info(
                        "Merge: capped ANN→LLM candidates from %d to %d",
                        len(ann_llm_candidates),
                        ann_cap,
                    )

            # Also collect name-based soft-zone pairs (traditional path)
            if not used_embeddings:
                # type_blocks already built in fallback path above
                soft_zone_pairs = self._collect_soft_zone_pairs(
                    type_blocks,
                    block_size_limit,
                    cfg.consolidation_merge_soft_threshold,
                    threshold,
                    require_same_type,
                )
                all_llm_candidates.extend(soft_zone_pairs)

            if all_llm_candidates:
                llm_merges = self._run_llm_merge_pass(
                    all_llm_candidates,
                    cfg,
                    dry_run,
                )
                for ea_id, eb_id in llm_merges:
                    union(ea_id, eb_id)
                    _remember_merge_decision(
                        merge_decisions,
                        ea_id,
                        eb_id,
                        confidence=None,
                        source="llm",
                        reason="llm_merge",
                    )

        # --- Structural candidate discovery (neighbor-based, no name needed) ---
        if cfg.consolidation_merge_multi_signal_enabled and hasattr(
            graph_store, "find_structural_merge_candidates"
        ):
            try:
                structural_pairs = await graph_store.find_structural_merge_candidates(
                    group_id,
                    min_shared_neighbors=cfg.consolidation_merge_structural_min_neighbors,
                    limit=100,
                )
                # Build entity lookup for structural candidates
                id_to_entity = {e.id: e for e in entities}
                structural_uncached = [
                    (id_to_entity[a], id_to_entity[b], 0.0)
                    for a, b, _count in structural_pairs
                    if a in id_to_entity
                    and b in id_to_entity
                    and _types_allowed(id_to_entity[a], id_to_entity[b], require_same_type)
                    and frozenset({a, b}) not in self._keep_separate_cache
                    and find(a) != find(b)  # Not already merged
                ]
                if structural_uncached:
                    logger.info(
                        "Merge: %d structural candidates (shared neighbors)",
                        len(structural_uncached),
                    )
                    for ea, eb, _sim in structural_uncached:
                        verdict, conf, signals = await score_merge_pair(
                            ea,
                            eb,
                            search_index,
                            graph_store,
                            group_id,
                            cross_encoder_enabled=ce_enabled,
                        )
                        reason = signals.get("reason")
                        if isinstance(reason, str):
                            _maybe_queue_identifier_review(
                                identifier_review_records,
                                identifier_review_pairs,
                                enabled=identifier_review_enabled,
                                max_records=identifier_review_max_per_cycle,
                                min_similarity=identifier_review_min_similarity,
                                cycle_id=cycle_id,
                                group_id=group_id,
                                entity_a=ea,
                                entity_b=eb,
                                decision=dedup_policy(ea.name, eb.name),
                                raw_similarity=compute_similarity(ea.name, eb.name),
                                adjusted_similarity=conf,
                                decision_source="structural_multi_signal",
                            )
                        if verdict == "merge":
                            union(ea.id, eb.id)
                            _remember_merge_decision(
                                merge_decisions,
                                ea.id,
                                eb.id,
                                confidence=conf,
                                source="structural_multi_signal",
                                reason=signals.get("reason") or "multi_signal_accepted",
                            )
                        else:
                            self._keep_separate_cache.add(frozenset({ea.id, eb.id}))
            except Exception as exc:
                logger.warning("Structural candidate discovery failed: %s", exc)

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
                decision_meta = merge_decisions.get(_pair_key(survivor.id, loser.id))
                allow_exact_identifier_cross_type = (
                    decision_meta is not None
                    and decision_meta.get("source") == "identifier_policy"
                    and decision_meta.get("reason")
                    in {
                        "identifier_exact_match",
                        "hybrid_code_match",
                    }
                )
                if (
                    not _types_allowed(survivor, loser, require_same_type)
                    and not allow_exact_identifier_cross_type
                ):
                    continue

                name_similarity = _name_similarity(survivor, loser, same_type_boost)
                if decision_meta is None:
                    decision_meta = {
                        "confidence": round(name_similarity, 4),
                        "source": "transitive_union",
                        "reason": "transitive_union",
                    }

                rels_transferred = 0
                if not dry_run:
                    if (
                        allow_exact_identifier_cross_type
                        and IDENTIFIER_ENTITY_TYPE in {survivor.entity_type, loser.entity_type}
                        and survivor.entity_type != IDENTIFIER_ENTITY_TYPE
                        and should_promote_entity_type_to_identifier(survivor.entity_type)
                    ):
                        await graph_store.update_entity(
                            survivor.id,
                            {"entity_type": IDENTIFIER_ENTITY_TYPE},
                            group_id=group_id,
                        )
                        survivor.entity_type = IDENTIFIER_ENTITY_TYPE
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
                        similarity=round(name_similarity, 4),
                        decision_confidence=decision_meta.get("confidence"),
                        decision_source=decision_meta.get("source"),
                        decision_reason=decision_meta.get("reason"),
                        relationships_transferred=rels_transferred,
                    )
                )
                if context is not None:
                    trace = DecisionTrace(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        phase=self.name,
                        candidate_type="entity_pair",
                        candidate_id=_pair_candidate_id(survivor.id, loser.id),
                        decision="merge_applied",
                        decision_source=decision_meta.get("source") or "union_find",
                        confidence=decision_meta.get("confidence"),
                        threshold_band="applied",
                        features={
                            "name_similarity": round(name_similarity, 4),
                            "decision_reason": decision_meta.get("reason"),
                            "relationships_transferred": rels_transferred,
                            "survivor_access_count": survivor.access_count,
                            "loser_access_count": loser.access_count,
                        },
                    )
                    context.add_decision_trace(trace)
                    context.add_decision_outcome_label(
                        DecisionOutcomeLabel(
                            cycle_id=cycle_id,
                            group_id=group_id,
                            phase=self.name,
                            decision_trace_id=trace.id,
                            outcome_type="materialization",
                            label="applied",
                            value=1.0,
                            metadata={"relationships_transferred": rels_transferred},
                        )
                    )

        return PhaseResult(
            phase=self.name,
            items_processed=pairs_checked,
            items_affected=len(merge_records),
            duration_ms=_elapsed_ms(t0),
        ), [*merge_records, *identifier_review_records]

    @staticmethod
    def _merge_exact_identifier_aliases(
        entities: list,
        same_type_boost: float,
        union_fn,
        merge_decisions: dict[tuple[str, str], dict],
        context: CycleContext | None,
        cycle_id: str,
        group_id: str,
    ) -> int:
        """Safely union entities that share the exact same canonical identifier."""
        blocks: dict[str, list] = defaultdict(list)
        for entity in entities:
            canonical = getattr(entity, "canonical_identifier", None)
            if canonical:
                blocks[canonical].append(entity)

        pairs_checked = 0
        for block in blocks.values():
            if len(block) < 2:
                continue
            for i in range(len(block)):
                for j in range(i + 1, len(block)):
                    ea, eb = block[i], block[j]
                    decision = dedup_policy(ea.name, eb.name)
                    pairs_checked += 1
                    if not decision.exact_identifier_match:
                        continue
                    union_fn(ea.id, eb.id)
                    _remember_merge_decision(
                        merge_decisions,
                        ea.id,
                        eb.id,
                        confidence=1.0,
                        source="identifier_policy",
                        reason=decision.reason,
                    )
                    _add_policy_trace(
                        context,
                        cycle_id,
                        group_id,
                        ea,
                        eb,
                        decision,
                        path="identifier_exact_alias",
                        decision_label="merge",
                        threshold_band="accepted",
                        confidence=1.0,
                        extra_features={
                            "name_similarity": round(
                                _name_similarity(ea, eb, same_type_boost),
                                4,
                            ),
                        },
                    )
        return pairs_checked

    @staticmethod
    async def _find_candidates_via_embeddings(
        entities: list,
        search_index,
        group_id: str,
        cfg: ActivationConfig,
        require_same_type: bool = False,
    ) -> list[tuple] | None:
        """Find merge candidates using embedding cosine similarity.

        Returns list of (entity_a, entity_b) pairs above the embedding
        threshold, or None if embeddings aren't available/sufficient.
        """
        if not hasattr(search_index, "get_entity_embeddings"):
            return None

        entity_ids = [e.id for e in entities]
        embeddings = await search_index.get_entity_embeddings(entity_ids, group_id=group_id)

        if not embeddings:
            return None

        # Check coverage — need enough entities with embeddings
        coverage = len(embeddings) / len(entities)
        if coverage < cfg.consolidation_merge_embedding_min_coverage:
            logger.info(
                "Merge ANN: low embedding coverage (%.1f%% < %.1f%%), using fallback",
                coverage * 100,
                cfg.consolidation_merge_embedding_min_coverage * 100,
            )
            return None

        # Build ID-indexed entity lookup and embedding matrix
        id_to_entity = {e.id: e for e in entities}
        embedded_ids = [eid for eid in entity_ids if eid in embeddings]
        if len(embedded_ids) < 2:
            return None

        mat = np.array([embeddings[eid] for eid in embedded_ids], dtype=np.float64)

        # L2-normalize rows for cosine similarity via dot product
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        mat = mat / norms

        # Compute cosine similarity matrix (upper triangle only via chunking)
        emb_threshold = cfg.consolidation_merge_embedding_threshold
        candidates: list[tuple] = []

        # Process in chunks to keep memory bounded for large graphs
        chunk_size = 500
        for start in range(0, len(embedded_ids), chunk_size):
            end = min(start + chunk_size, len(embedded_ids))
            chunk = mat[start:end]  # (chunk_size, dim)

            # Similarity of this chunk against all entities from start onward
            # (avoids duplicate pairs)
            rest = mat[start:]  # (N-start, dim)
            sim_block = chunk @ rest.T  # (chunk_size, N-start)

            # Find pairs above threshold
            for ci in range(sim_block.shape[0]):
                # Only look at j > i to avoid duplicates
                for cj in range(ci + 1, sim_block.shape[1]):
                    if sim_block[ci, cj] >= emb_threshold:
                        i_abs = start + ci
                        j_abs = start + cj
                        ea = id_to_entity[embedded_ids[i_abs]]
                        eb = id_to_entity[embedded_ids[j_abs]]
                        if not _types_allowed(ea, eb, require_same_type):
                            continue
                        candidates.append((ea, eb))

        logger.info(
            "Merge ANN: %d entities with embeddings → %d candidate pairs (threshold=%.2f)",
            len(embedded_ids),
            len(candidates),
            emb_threshold,
        )
        return candidates

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
                    decision, sim = _policy_similarity(ea, eb, same_type_boost)
                    if not decision.allowed or decision.exact_identifier_match:
                        continue
                    if soft_threshold <= sim < hard_threshold:
                        soft_pairs.append((ea, eb, sim))

        return soft_pairs

    def _run_llm_merge_pass(
        self,
        soft_pairs: list[tuple],
        cfg: ActivationConfig,
        dry_run: bool,
    ) -> list[tuple[str, str]]:
        """Judge soft-zone pairs via LLM in batches, return pairs to merge."""
        client = self._llm_client
        if client is None:
            try:
                import anthropic

                client = anthropic.Anthropic()
            except Exception:
                logger.warning("Could not create Anthropic client for merge LLM")
                return []

        approved_merges: list[tuple[str, str]] = []

        # Filter out pairs previously judged "keep_separate"
        uncached_pairs = [
            (ea, eb, sim)
            for ea, eb, sim in soft_pairs
            if frozenset({ea.id, eb.id}) not in self._keep_separate_cache
        ]
        if len(uncached_pairs) < len(soft_pairs):
            logger.info(
                "Merge: skipped %d cached keep_separate pairs",
                len(soft_pairs) - len(uncached_pairs),
            )

        # Process pairs in batches of _MERGE_BATCH_SIZE
        for batch_start in range(0, len(uncached_pairs), _MERGE_BATCH_SIZE):
            batch = uncached_pairs[batch_start : batch_start + _MERGE_BATCH_SIZE]

            try:
                # Build numbered user message for the batch
                parts: list[str] = []
                for idx, (ea, eb, sim) in enumerate(batch, start=1):
                    parts.append(
                        f"Pair {idx}:\n"
                        f"Entity A: {ea.name} (type: {ea.entity_type})\n"
                        f"Entity B: {eb.name} (type: {eb.entity_type})\n"
                        f"String similarity: {sim:.4f}"
                    )
                user_msg = "\n\n".join(parts)

                response = client.messages.create(
                    model=cfg.consolidation_merge_llm_model,
                    max_tokens=256 * len(batch),
                    system=_MERGE_JUDGE_SYSTEM_CACHED,
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = response.content[0].text.strip()
                verdicts = json.loads(text)

                # Normalise: if the model returns a single dict, wrap it
                if isinstance(verdicts, dict):
                    verdicts = [verdicts]

                # Build a lookup from pair number to verdict
                verdict_map: dict[int, str] = {}
                for v in verdicts:
                    pair_num = v.get("pair", 0)
                    verdict_map[pair_num] = v.get("verdict", "keep_separate")

                # Process each pair's verdict
                for idx, (ea, eb, sim) in enumerate(batch, start=1):
                    verdict = verdict_map.get(idx, "keep_separate")
                    if verdict == "merge":
                        approved_merges.append((ea.id, eb.id))
                    elif verdict == "uncertain" and cfg.consolidation_merge_escalation_enabled:
                        # Escalate to Sonnet (one at a time)
                        esc_verdict = self._escalate_merge(
                            ea,
                            eb,
                            sim,
                            cfg,
                            client,
                        )
                        if esc_verdict == "merge":
                            approved_merges.append((ea.id, eb.id))
                        else:
                            self._keep_separate_cache.add(frozenset({ea.id, eb.id}))
                    elif verdict == "keep_separate":
                        self._keep_separate_cache.add(frozenset({ea.id, eb.id}))

            except Exception as exc:
                # Log batch failure with all pair names
                pair_names = ", ".join(f"{ea.name}/{eb.name}" for ea, eb, _sim in batch)
                logger.warning("Merge LLM judge failed for batch [%s]: %s", pair_names, exc)

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
            verdict_raw = parsed.get("verdict", "keep_separate")
            verdict = verdict_raw if isinstance(verdict_raw, str) else "keep_separate"
            if verdict not in ("merge", "keep_separate"):
                verdict = "keep_separate"
            return verdict
        except Exception as exc:
            logger.warning("Merge escalation failed for %s/%s: %s", ea.name, eb.name, exc)
            return "keep_separate"


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _pair_candidate_id(entity_a_id: str, entity_b_id: str) -> str:
    a_id, b_id = sorted((entity_a_id, entity_b_id))
    return f"{a_id}:{b_id}"


def _decision_band(verdict: str) -> str:
    if verdict == "merge":
        return "accepted"
    if verdict == "keep_separate":
        return "rejected"
    return "uncertain"
