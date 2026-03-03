"""Edge inference phase: create MENTIONED_WITH edges for co-occurring entities."""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from datetime import datetime

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, InferredEdge, PhaseResult
from engram.models.relationship import Relationship

logger = logging.getLogger(__name__)

_LLM_VALIDATION_PROMPT = (
    "You are a knowledge graph quality validator.\n"
    "Given two entities and a proposed relationship, determine if it makes semantic sense.\n"
    "Respond with JSON only: "
    '{"verdict": "approved"|"rejected"|"uncertain", "reason": "brief explanation"}'
)


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


class EdgeInferencePhase(ConsolidationPhase):
    """Find entity pairs that co-occur across episodes and create edges."""

    def __init__(self, llm_client=None):
        self._llm_client = llm_client

    @property
    def name(self) -> str:
        return "infer"

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
                group_id, list(all_ids),
            )
            stats = await graph_store.get_stats(group_id)
            total_episodes = stats.get("total_episodes", 0)

        records: list[InferredEdge] = []
        for entity_a_id, entity_b_id, count in pairs:
            if len(records) >= max_edges:
                break

            rel_id = None
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
                    pmi_val, imp_a, imp_b,
                    cfg.consolidation_infer_tfidf_weight,
                    confidence_floor,
                )
                pmi_score = round(pmi_val, 4)
                infer_type = "co_occurrence_pmi"
            else:
                # Original linear formula
                confidence = min(0.9, confidence_floor * (1 + (count - min_co) * 0.05))

            weight = min(1.0, count / 10.0)

            # Resolve names for audit
            entity_a = await graph_store.get_entity(entity_a_id, group_id)
            entity_b = await graph_store.get_entity(entity_b_id, group_id)
            if not entity_a or not entity_b:
                continue

            if not dry_run:
                rel = Relationship(
                    id=f"rel_{uuid.uuid4().hex[:12]}",
                    source_id=entity_a_id,
                    target_id=entity_b_id,
                    predicate="MENTIONED_WITH",
                    weight=weight,
                    confidence=confidence,
                    source_episode=f"consolidation:{cycle_id}",
                    group_id=group_id,
                )
                await graph_store.create_relationship(rel)
                rel_id = rel.id

                # Track affected entities for reindex
                if context is not None:
                    context.inferred_edge_entity_ids.add(entity_a_id)
                    context.inferred_edge_entity_ids.add(entity_b_id)
                    context.affected_entity_ids.add(entity_a_id)
                    context.affected_entity_ids.add(entity_b_id)

            records.append(InferredEdge(
                cycle_id=cycle_id,
                group_id=group_id,
                source_id=entity_a_id,
                target_id=entity_b_id,
                source_name=entity_a.name,
                target_name=entity_b.name,
                co_occurrence_count=count,
                confidence=round(confidence, 4),
                infer_type=infer_type,
                pmi_score=pmi_score,
                relationship_id=rel_id,
            ))

        # --- Transitivity pass ---
        if cfg.consolidation_infer_transitivity_enabled:
            trans_records = await _run_transitivity_pass(
                group_id, graph_store, cfg, cycle_id, dry_run,
                remaining=max_edges - len(records),
                context=context,
            )
            records.extend(trans_records)

        # --- LLM validation pass (Tier 3) ---
        if cfg.consolidation_infer_llm_enabled:
            await self._run_llm_validation_pass(
                records, graph_store, cfg, group_id, dry_run,
            )

        return PhaseResult(
            phase=self.name,
            items_processed=len(pairs) + (len(records) - len(pairs)),
            items_affected=len(records),
            duration_ms=_elapsed_ms(t0),
        ), records

    async def _run_llm_validation_pass(
        self,
        records: list[InferredEdge],
        graph_store,
        cfg: ActivationConfig,
        group_id: str,
        dry_run: bool,
    ) -> None:
        """Validate high-confidence inferred edges via LLM."""
        threshold = cfg.consolidation_infer_llm_confidence_threshold
        max_validations = cfg.consolidation_infer_llm_max_per_cycle

        candidates = [
            r for r in records
            if r.confidence >= threshold
            and r.infer_type in ("co_occurrence", "co_occurrence_pmi")
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
                return

        for rec in candidates:
            try:
                user_msg = (
                    f"Entity A: {rec.source_name} (ID: {rec.source_id})\n"
                    f"Entity B: {rec.target_name} (ID: {rec.target_id})\n"
                    f"Proposed relationship: MENTIONED_WITH\n"
                    f"Co-occurrence count: {rec.co_occurrence_count}\n"
                    f"Statistical confidence: {rec.confidence}"
                )
                response = client.messages.create(
                    model=cfg.consolidation_infer_llm_model,
                    max_tokens=256,
                    system=_LLM_VALIDATION_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = response.content[0].text.strip()
                parsed = json.loads(text)
                verdict = parsed.get("verdict", "uncertain")

                if verdict == "approved":
                    rec.infer_type = "llm_validated"
                    rec.llm_verdict = "approved"
                elif verdict == "rejected":
                    rec.infer_type = "llm_rejected"
                    rec.llm_verdict = "rejected"
                    # Invalidate the created relationship
                    if rec.relationship_id:
                        await graph_store.invalidate_relationship(
                            rec.relationship_id, datetime.utcnow(), group_id,
                        )
                else:
                    rec.llm_verdict = "uncertain"
            except Exception as exc:
                logger.warning("LLM validation failed for %s: %s", rec.id, exc)
                rec.llm_verdict = "error"


async def _run_transitivity_pass(
    group_id: str,
    graph_store,
    cfg: ActivationConfig,
    cycle_id: str,
    dry_run: bool,
    remaining: int,
    context: CycleContext | None = None,
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
            group_id, predicate,
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

                    if not dry_run:
                        rel = Relationship(
                            id=f"rel_{uuid.uuid4().hex[:12]}",
                            source_id=a_id,
                            target_id=c_id,
                            predicate=predicate,
                            weight=confidence,
                            confidence=confidence,
                            source_episode=f"consolidation:{cycle_id}:transitivity",
                            group_id=group_id,
                        )
                        await graph_store.create_relationship(rel)

                        if context is not None:
                            context.inferred_edge_entity_ids.add(a_id)
                            context.inferred_edge_entity_ids.add(c_id)
                            context.affected_entity_ids.add(a_id)
                            context.affected_entity_ids.add(c_id)

                    records.append(InferredEdge(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        source_id=a_id,
                        target_id=c_id,
                        source_name=entity_a.name,
                        target_name=entity_c.name,
                        co_occurrence_count=0,
                        confidence=confidence,
                        infer_type="transitivity",
                    ))

                    # Prevent duplicate within this pass
                    direct_pairs.add((a_id, c_id))
                    direct_pairs.add((c_id, a_id))

    return records


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
