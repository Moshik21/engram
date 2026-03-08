"""Edge inference phase: create MENTIONED_WITH edges for co-occurring entities."""

from __future__ import annotations

import json
import logging
import math
import time

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.graph_manager import GraphManager
from engram.models.consolidation import (
    CycleContext,
    DecisionOutcomeLabel,
    DecisionTrace,
    InferredEdge,
    PhaseResult,
)

logger = logging.getLogger(__name__)

_LLM_VALIDATION_PROMPT = (
    "You are a knowledge graph quality validator.\n"
    "Given one or more proposed relationships, determine if each makes semantic sense.\n"
    "Respond with a JSON array of verdicts, one per relationship:\n"
    '[{"rel": 1, "verdict": "approved"|"rejected"|"uncertain", "reason": "brief explanation"}, ...]'
)

_LLM_VALIDATION_SYSTEM_CACHED = [
    {
        "type": "text",
        "text": _LLM_VALIDATION_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]

_INFER_BATCH_SIZE = 5

_LLM_ESCALATION_PROMPT = (
    "You are a senior knowledge graph quality reviewer.\n"
    "A lower-tier model was uncertain about the following relationship.\n"
    "Apply strict judgment. Respond with JSON only:\n"
    '{"verdict": "approved"|"rejected", "reason": "brief explanation"}\n'
    "You MUST choose approved or rejected. Do not respond with uncertain."
)

_LLM_ESCALATION_SYSTEM_CACHED = [
    {
        "type": "text",
        "text": _LLM_ESCALATION_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


# ---------------------------------------------------------------------------
# PMI / tf-idf helpers
# ---------------------------------------------------------------------------


def _compute_pmi(co_count: int, ep_count_a: int, ep_count_b: int, total_episodes: int) -> float:
    """PMI(a,b) = log2(P(a,b) / (P(a) * P(b))). Returns 0.0 on invalid input."""
    if total_episodes <= 0 or ep_count_a <= 0 or ep_count_b <= 0 or co_count <= 0:
        return 0.0
    p_ab = co_count / total_episodes
    p_a = ep_count_a / total_episodes
    p_b = ep_count_b / total_episodes
    denom = p_a * p_b
    if denom <= 0:
        return 0.0
    return math.log2(p_ab / denom)


def _compute_tfidf_importance(entity_ep_count: int, total_episodes: int) -> float:
    """IDF-based importance: log(N/df) normalized to [0,1]. Rare entities → higher."""
    if total_episodes <= 0 or entity_ep_count <= 0:
        return 0.0
    raw = math.log(total_episodes / entity_ep_count)
    max_idf = math.log(total_episodes) if total_episodes > 1 else 1.0
    if max_idf <= 0:
        return 0.0
    return min(1.0, raw / max_idf)


def _pmi_confidence(
    pmi: float,
    importance_a: float,
    importance_b: float,
    tfidf_weight: float,
    floor: float,
) -> float:
    """Sigmoid-maps PMI to [floor, 0.95], blends with tf-idf importance."""
    # Sigmoid: 1 / (1 + exp(-pmi)) mapped to [floor, 0.95]
    sigmoid = 1.0 / (1.0 + math.exp(-pmi))
    pmi_conf = floor + (0.95 - floor) * sigmoid
    # Blend with tf-idf importance (average of both entities)
    avg_importance = (importance_a + importance_b) / 2.0
    blended = (1.0 - tfidf_weight) * pmi_conf + tfidf_weight * avg_importance
    return round(min(0.95, max(floor, blended)), 4)


def _get_total_episode_count(stats: dict | None) -> int:
    """Read the canonical episode-count key from store stats.

    SQLite and FalkorDB expose ``episodes``. ``total_episodes`` is accepted as a
    legacy fallback for mocks and older callers.
    """
    if not stats:
        return 0
    value = stats.get("episodes")
    if value is None:
        value = stats.get("total_episodes", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class EdgeInferencePhase(ConsolidationPhase):
    """Find entity pairs that co-occur across episodes and create edges."""

    def __init__(self, llm_client=None, escalation_client=None, canonicalizer=None):
        self._llm_client = llm_client
        self._escalation_client = escalation_client
        self._canonicalizer = canonicalizer or PredicateCanonicalizer()

    @property
    def name(self) -> str:
        return "infer"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        methods = {
            "get_co_occurring_entity_pairs",
            "get_entity",
            "create_relationship",
            "get_relationships",
            "find_existing_relationship",
            "find_conflicting_relationships",
            "update_relationship_weight",
            "invalidate_relationship",
        }
        if cfg.consolidation_infer_pmi_enabled or cfg.consolidation_infer_auto_validation_enabled:
            methods.update({"get_entity_episode_counts", "get_stats"})
        if cfg.consolidation_infer_transitivity_enabled:
            methods.add("get_relationships_by_predicate")
        return methods

    def required_search_index_methods(self, cfg: ActivationConfig) -> set[str]:
        if cfg.consolidation_infer_auto_validation_enabled:
            return {"get_entity_embeddings"}
        return set()

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
    ) -> tuple[PhaseResult, list[InferredEdge]]:
        t0 = time.perf_counter()
        min_co = cfg.consolidation_infer_cooccurrence_min
        confidence_floor = cfg.consolidation_infer_confidence_floor
        max_edges = cfg.consolidation_infer_max_per_cycle

        pairs = await graph_store.get_co_occurring_entity_pairs(
            group_id=group_id,
            min_co_occurrence=min_co,
            limit=max_edges,
        )

        # --- PMI pre-computation (Tier 2) ---
        ep_counts: dict[str, int] = {}
        total_episodes = 0
        if cfg.consolidation_infer_pmi_enabled and pairs:
            all_ids: set[str] = set()
            for a_id, b_id, _ in pairs:
                all_ids.add(a_id)
                all_ids.add(b_id)
            ep_counts = await graph_store.get_entity_episode_counts(
                group_id,
                list(all_ids),
            )
            stats = await graph_store.get_stats(group_id)
            total_episodes = _get_total_episode_count(stats)

        records: list[InferredEdge] = []
        for entity_a_id, entity_b_id, count in pairs:
            if len(records) >= max_edges:
                break

            pmi_score = None
            infer_type = "co_occurrence"

            if cfg.consolidation_infer_pmi_enabled and total_episodes > 0:
                ep_a = ep_counts.get(entity_a_id, 0)
                ep_b = ep_counts.get(entity_b_id, 0)
                pmi_val = _compute_pmi(count, ep_a, ep_b, total_episodes)
                if pmi_val < cfg.consolidation_infer_pmi_min:
                    continue
                imp_a = _compute_tfidf_importance(ep_a, total_episodes)
                imp_b = _compute_tfidf_importance(ep_b, total_episodes)
                confidence = _pmi_confidence(
                    pmi_val,
                    imp_a,
                    imp_b,
                    cfg.consolidation_infer_tfidf_weight,
                    confidence_floor,
                )
                pmi_score = round(pmi_val, 4)
                infer_type = "co_occurrence_pmi"
            else:
                # Original linear formula
                confidence = min(0.9, confidence_floor * (1 + (count - min_co) * 0.05))

            entity_a = await graph_store.get_entity(entity_a_id, group_id)
            entity_b = await graph_store.get_entity(entity_b_id, group_id)
            if not entity_a or not entity_b:
                continue

            records.append(
                InferredEdge(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    source_id=entity_a_id,
                    target_id=entity_b_id,
                    source_name=entity_a.name,
                    target_name=entity_b.name,
                    co_occurrence_count=count,
                    confidence=round(confidence, 4),
                    predicate="MENTIONED_WITH",
                    infer_type=infer_type,
                    pmi_score=pmi_score,
                )
            )

        # --- Transitivity pass ---
        if cfg.consolidation_infer_transitivity_enabled:
            trans_records = await _run_transitivity_pass(
                group_id,
                graph_store,
                cfg,
                cycle_id,
                remaining=max_edges - len(records),
            )
            records.extend(trans_records)

        # --- Multi-signal auto-validation (replaces LLM when enabled) ---
        if cfg.consolidation_infer_auto_validation_enabled:
            await self._run_auto_validation_pass(
                records, graph_store, search_index, cfg, group_id, dry_run,
            )
        elif cfg.consolidation_infer_llm_enabled:
            await self._run_llm_validation_pass(
                records,
                cfg,
                group_id,
                dry_run,
            )

        # --- Sonnet escalation pass (Tier 4) ---
        if cfg.consolidation_infer_escalation_enabled:
            await self._run_escalation_pass(
                records,
                cfg,
                group_id,
                dry_run,
            )

        affected = await self._materialize_records(
            records,
            graph_store,
            cfg,
            group_id,
            cycle_id,
            dry_run,
            context,
        )

        if context is not None:
            for rec in records:
                trace = DecisionTrace(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    phase=self.name,
                    candidate_type="relationship",
                    candidate_id=_relationship_candidate_id(
                        rec.source_id,
                        rec.target_id,
                        rec.predicate,
                    ),
                    decision=_infer_decision(rec),
                    decision_source=_infer_decision_source(rec),
                    confidence=rec.confidence,
                    threshold_band=_infer_threshold_band(rec, dry_run),
                    features={
                        "predicate": rec.predicate,
                        "co_occurrence_count": rec.co_occurrence_count,
                        "pmi_score": rec.pmi_score,
                        "infer_type": rec.infer_type,
                        "validation_score": rec.validation_score,
                        "validation_signals": rec.validation_signals,
                        "llm_verdict": rec.llm_verdict,
                        "escalation_verdict": rec.escalation_verdict,
                        "materialization_action": rec.materialization_action,
                    },
                    metadata={"relationship_id": rec.relationship_id},
                    policy_version="typed_link_v1",
                )
                context.add_decision_trace(trace)
                if trace.decision in {"accept", "reject"}:
                    context.add_decision_outcome_label(
                        DecisionOutcomeLabel(
                            cycle_id=cycle_id,
                            group_id=group_id,
                            phase=self.name,
                            decision_trace_id=trace.id,
                            outcome_type="validation",
                            label=trace.decision,
                            value=1.0,
                            metadata={"infer_type": rec.infer_type},
                        )
                    )
                if rec.materialization_action is not None:
                    context.add_decision_outcome_label(
                        DecisionOutcomeLabel(
                            cycle_id=cycle_id,
                            group_id=group_id,
                            phase=self.name,
                            decision_trace_id=trace.id,
                            outcome_type="materialization",
                            label=rec.materialization_action,
                            value=1.0 if rec.materialization_action in {
                                "created",
                                "updated_existing",
                            } else 0.0,
                            metadata={"predicate": rec.predicate},
                        )
                    )

        return PhaseResult(
            phase=self.name,
            items_processed=len(pairs) + (len(records) - len(pairs)),
            items_affected=affected,
            duration_ms=_elapsed_ms(t0),
        ), records

    async def _run_auto_validation_pass(
        self,
        records: list[InferredEdge],
        graph_store,
        search_index,
        cfg: ActivationConfig,
        group_id: str,
        dry_run: bool,
    ) -> None:
        """Validate inferred edges using multi-signal scoring (no LLM)."""
        from engram.consolidation.scorers.infer_scorer import score_infer_pair

        # Same candidate selection as LLM path
        threshold = cfg.consolidation_infer_llm_confidence_threshold
        max_validations = cfg.consolidation_infer_llm_max_per_cycle
        candidates = [
            r
            for r in records
            if r.confidence >= threshold
            and r.infer_type in (
                "co_occurrence", "co_occurrence_pmi", "transitivity",
            )
        ][:max_validations]

        if not candidates:
            return

        # Need episode counts for ubiquity scoring
        all_ids: set[str] = set()
        for rec in candidates:
            all_ids.add(rec.source_id)
            all_ids.add(rec.target_id)
        ep_counts = await graph_store.get_entity_episode_counts(group_id, list(all_ids))
        stats = await graph_store.get_stats(group_id)
        total_episodes = _get_total_episode_count(stats)

        for rec in candidates:
            try:
                # Get entity objects for type info
                entity_a = await graph_store.get_entity(rec.source_id, group_id)
                entity_b = await graph_store.get_entity(rec.target_id, group_id)
                if not entity_a or not entity_b:
                    continue

                verdict, _score, _signals = await score_infer_pair(
                    entity_a_id=rec.source_id,
                    entity_b_id=rec.target_id,
                    entity_a_name=rec.source_name,
                    entity_b_name=rec.target_name,
                    entity_a_type=entity_a.entity_type,
                    entity_b_type=entity_b.entity_type,
                    co_occurrence_count=rec.co_occurrence_count,
                    pmi_confidence=rec.confidence,
                    ep_count_a=ep_counts.get(rec.source_id, 0),
                    ep_count_b=ep_counts.get(rec.target_id, 0),
                    total_episodes=total_episodes,
                    search_index=search_index,
                    graph_store=graph_store,
                    group_id=group_id,
                    domain_groups=cfg.domain_groups if hasattr(cfg, "domain_groups") else None,
                    approve_threshold=cfg.consolidation_infer_auto_approve_threshold,
                    reject_threshold=cfg.consolidation_infer_auto_reject_threshold,
                )
                rec.validation_score = round(_score, 4)
                rec.validation_signals = dict(_signals)

                if verdict == "approved":
                    rec.infer_type = "auto_validated"
                    rec.llm_verdict = "auto_approved"
                elif verdict == "rejected":
                    rec.infer_type = "auto_rejected"
                    rec.llm_verdict = "auto_rejected"
                else:
                    # Uncertain: try cross-encoder (Tier 1)
                    ce_on = getattr(cfg, "consolidation_cross_encoder_enabled", True)
                    if not ce_on:
                        rec.llm_verdict = "auto_uncertain"
                        continue
                    try:
                        from engram.consolidation.scorers.cross_encoder import (
                            refine_infer_verdict,
                        )

                        ce_verdict, ce_score = await refine_infer_verdict(
                            entity_a, entity_b,
                            "MENTIONED_WITH",
                            _score,
                            approve_threshold=cfg.consolidation_infer_auto_approve_threshold,
                            reject_threshold=cfg.consolidation_infer_auto_reject_threshold,
                        )
                        rec.validation_signals["cross_encoder_score"] = round(ce_score, 4)
                        if ce_verdict == "approved":
                            rec.infer_type = "cross_encoder_approved"
                            rec.llm_verdict = "ce_approved"
                        elif ce_verdict == "rejected":
                            rec.infer_type = "cross_encoder_rejected"
                            rec.llm_verdict = "ce_rejected"
                        else:
                            rec.llm_verdict = "auto_uncertain"
                    except Exception as ce_exc:
                        logger.warning("Cross-encoder refinement failed: %s", ce_exc)
                        rec.llm_verdict = "auto_uncertain"
            except Exception as exc:
                logger.warning("Auto-validation failed for %s: %s", rec.source_name, exc)
                rec.llm_verdict = "error"

    async def _run_llm_validation_pass(
        self,
        records: list[InferredEdge],
        cfg: ActivationConfig,
        group_id: str,
        dry_run: bool,
    ) -> None:
        """Validate high-confidence inferred edges via LLM (batched)."""
        threshold = cfg.consolidation_infer_llm_confidence_threshold
        max_validations = cfg.consolidation_infer_llm_max_per_cycle

        candidates = [
            r
            for r in records
            if r.confidence >= threshold and r.infer_type in ("co_occurrence", "co_occurrence_pmi")
        ][:max_validations]

        if not candidates:
            return

        if dry_run:
            for rec in candidates:
                rec.llm_verdict = "dry_run_skipped"
            return

        # Get or create client
        client = self._llm_client
        if client is None:
            try:
                import anthropic

                client = anthropic.Anthropic()
            except Exception:
                logger.warning("Could not create Anthropic client for LLM validation")
                for rec in candidates:
                    rec.llm_verdict = "abstain_unavailable"
                return

        # Process candidates in batches
        for batch_start in range(0, len(candidates), _INFER_BATCH_SIZE):
            batch = candidates[batch_start : batch_start + _INFER_BATCH_SIZE]
            try:
                # Build numbered user message for the batch
                parts: list[str] = []
                for idx, rec in enumerate(batch, 1):
                    parts.append(
                        f"Relationship {idx}:\n"
                        f"Entity A: {rec.source_name} (ID: {rec.source_id})\n"
                        f"Entity B: {rec.target_name} (ID: {rec.target_id})\n"
                        f"Proposed relationship: MENTIONED_WITH\n"
                        f"Co-occurrence count: {rec.co_occurrence_count}\n"
                        f"Statistical confidence: {rec.confidence}"
                    )
                user_msg = "\n\n".join(parts)

                response = client.messages.create(
                    model=cfg.consolidation_infer_llm_model,
                    max_tokens=256 * len(batch),
                    system=_LLM_VALIDATION_SYSTEM_CACHED,
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = response.content[0].text.strip()
                parsed = json.loads(text)

                # Normalise: accept both a bare list and a dict wrapping one
                if isinstance(parsed, dict):
                    # Handle single-item response for batch of 1
                    parsed = [parsed]

                # Build lookup by rel number (1-indexed)
                verdict_map: dict[int, dict] = {}
                for item in parsed:
                    rel_num = item.get("rel", 0)
                    verdict_map[rel_num] = item

                # Apply verdicts to corresponding records
                for idx, rec in enumerate(batch, 1):
                    item = verdict_map.get(idx)
                    if item is None:
                        rec.llm_verdict = "error"
                        continue
                    verdict = item.get("verdict", "uncertain")
                    if verdict == "approved":
                        rec.infer_type = "llm_validated"
                        rec.llm_verdict = "approved"
                    elif verdict == "rejected":
                        rec.infer_type = "llm_rejected"
                        rec.llm_verdict = "rejected"
                    else:
                        rec.llm_verdict = "uncertain"

            except Exception as exc:
                batch_ids = [r.id for r in batch]
                logger.warning("LLM batch validation failed for %s: %s", batch_ids, exc)
                for rec in batch:
                    rec.llm_verdict = "error"

    async def _run_escalation_pass(
        self,
        records: list[InferredEdge],
        cfg: ActivationConfig,
        group_id: str,
        dry_run: bool,
    ) -> None:
        """Re-validate uncertain edges via Sonnet escalation model."""
        max_escalations = cfg.consolidation_infer_escalation_max_per_cycle

        candidates = [
            r for r in records if r.llm_verdict == "uncertain"
        ][:max_escalations]

        if not candidates:
            return

        if dry_run:
            for rec in candidates:
                rec.escalation_verdict = "dry_run_skipped"
            return

        client = self._escalation_client or self._llm_client
        if client is None:
            try:
                import anthropic

                client = anthropic.Anthropic()
            except Exception:
                logger.warning("Could not create Anthropic client for escalation")
                for rec in candidates:
                    rec.escalation_verdict = "abstain_unavailable"
                return

        for rec in candidates:
            try:
                user_msg = (
                    f"Entity A: {rec.source_name} (ID: {rec.source_id})\n"
                    f"Entity B: {rec.target_name} (ID: {rec.target_id})\n"
                    f"Proposed relationship: MENTIONED_WITH\n"
                    f"Co-occurrence count: {rec.co_occurrence_count}\n"
                    f"Statistical confidence: {rec.confidence}\n"
                    f"Previous verdict: uncertain"
                )
                response = client.messages.create(
                    model=cfg.consolidation_infer_escalation_model,
                    max_tokens=256,
                    system=_LLM_ESCALATION_SYSTEM_CACHED,
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = response.content[0].text.strip()
                parsed = json.loads(text)
                verdict = parsed.get("verdict", "rejected")

                # Sonnet must not return uncertain
                if verdict not in ("approved", "rejected"):
                    verdict = "rejected"

                rec.escalation_verdict = verdict

                if verdict == "approved":
                    rec.infer_type = "escalation_approved"
                    rec.llm_verdict = "escalation_approved"
                elif verdict == "rejected":
                    rec.infer_type = "escalation_rejected"
                    rec.llm_verdict = "escalation_rejected"
            except Exception as exc:
                logger.warning("Escalation failed for %s: %s", rec.id, exc)
                rec.escalation_verdict = "error"

    async def _materialize_records(
        self,
        records: list[InferredEdge],
        graph_store,
        cfg: ActivationConfig,
        group_id: str,
        cycle_id: str,
        dry_run: bool,
        context: CycleContext | None,
    ) -> int:
        """Apply accepted inferred edges through the shared relationship path."""
        affected = 0
        for rec in records:
            decision = _infer_decision(rec)
            if decision != "accept":
                rec.materialization_action = "rejected" if decision == "reject" else "abstained"
                continue

            if dry_run:
                rec.materialization_action = "dry_run"
                affected += 1
                continue

            source_episode = f"consolidation:{cycle_id}"
            if rec.infer_type == "transitivity":
                source_episode = f"consolidation:{cycle_id}:transitivity"
            weight = rec.confidence if rec.infer_type == "transitivity" else min(
                1.0, rec.co_occurrence_count / 10.0,
            )
            apply_result = await GraphManager._apply_relationship_fact(
                graph_store,
                self._canonicalizer,
                cfg,
                {
                    "source_id": rec.source_id,
                    "target_id": rec.target_id,
                    "source_name": rec.source_name,
                    "target_name": rec.target_name,
                    "predicate": rec.predicate,
                    "weight": weight,
                    "confidence": rec.confidence,
                },
                {
                    rec.source_name: rec.source_id,
                    rec.target_name: rec.target_id,
                },
                group_id,
                source_episode,
            )
            rec.materialization_action = apply_result.action
            rec.relationship_id = (
                apply_result.metadata.get("relationship_id")
                or apply_result.metadata.get("existing_relationship_id")
            )
            if apply_result.created or apply_result.action == "updated_existing":
                affected += 1
                if context is not None:
                    context.inferred_edge_entity_ids.add(rec.source_id)
                    context.inferred_edge_entity_ids.add(rec.target_id)
                    context.affected_entity_ids.add(rec.source_id)
                    context.affected_entity_ids.add(rec.target_id)
        return affected


async def _run_transitivity_pass(
    group_id: str,
    graph_store,
    cfg: ActivationConfig,
    cycle_id: str,
    remaining: int,
) -> list[InferredEdge]:
    """Infer transitive edges: A→B + B→C ⟹ A→C for configured predicates."""
    if remaining <= 0:
        return []

    decay = cfg.consolidation_infer_transitivity_decay
    records: list[InferredEdge] = []

    for predicate in cfg.consolidation_infer_transitive_predicates:
        if len(records) >= remaining:
            break

        rels = await graph_store.get_relationships_by_predicate(
            group_id,
            predicate,
        )
        if not rels:
            continue

        # Build adjacency: source → [(target, confidence)]
        adjacency: dict[str, list[tuple[str, float]]] = {}
        direct_pairs: set[tuple[str, str]] = set()
        for r in rels:
            adjacency.setdefault(r.source_id, []).append(
                (r.target_id, r.confidence),
            )
            # Track both directions for dedup
            direct_pairs.add((r.source_id, r.target_id))
            direct_pairs.add((r.target_id, r.source_id))

        # Walk 2-hop paths: A→B→C
        for a_id, a_neighbors in adjacency.items():
            if len(records) >= remaining:
                break
            for b_id, conf_ab in a_neighbors:
                if len(records) >= remaining:
                    break
                for c_id, conf_bc in adjacency.get(b_id, []):
                    if len(records) >= remaining:
                        break
                    if c_id == a_id:
                        continue
                    if (a_id, c_id) in direct_pairs:
                        continue

                    confidence = round(min(conf_ab, conf_bc) * decay, 4)

                    # Resolve names
                    entity_a = await graph_store.get_entity(a_id, group_id)
                    entity_c = await graph_store.get_entity(c_id, group_id)
                    if not entity_a or not entity_c:
                        continue

                    records.append(
                        InferredEdge(
                            cycle_id=cycle_id,
                            group_id=group_id,
                            source_id=a_id,
                            target_id=c_id,
                            source_name=entity_a.name,
                            target_name=entity_c.name,
                            co_occurrence_count=0,
                            confidence=confidence,
                            predicate=predicate,
                            infer_type="transitivity",
                        )
                    )

                    # Prevent duplicate within this pass
                    direct_pairs.add((a_id, c_id))
                    direct_pairs.add((c_id, a_id))

    return records


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _relationship_candidate_id(source_id: str, target_id: str, predicate: str) -> str:
    a_id, b_id = sorted((source_id, target_id))
    return f"{a_id}:{b_id}:{predicate}"


def _infer_decision(rec: InferredEdge) -> str:
    if "rejected" in rec.infer_type:
        return "reject"
    if rec.llm_verdict in {"uncertain", "error", "auto_uncertain", "abstain_unavailable"}:
        return "abstain"
    if rec.escalation_verdict in {"error", "abstain_unavailable"}:
        return "abstain"
    return "accept"


def _infer_decision_source(rec: InferredEdge) -> str:
    if rec.infer_type.startswith("auto_"):
        return "auto_validation"
    if rec.infer_type.startswith("cross_encoder"):
        return "cross_encoder"
    if rec.infer_type.startswith("llm_"):
        return "llm"
    if rec.infer_type.startswith("escalation"):
        return "escalation"
    if rec.infer_type == "transitivity":
        return "transitivity"
    return "statistical"


def _infer_threshold_band(rec: InferredEdge, dry_run: bool) -> str:
    decision = _infer_decision(rec)
    if decision == "reject":
        return "rejected"
    if decision == "abstain":
        return "abstained"
    if dry_run and rec.relationship_id is None:
        return "proposed"
    return "accepted"
