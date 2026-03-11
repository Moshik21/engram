"""Microglia phase: complement-mediated graph immune surveillance."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from engram.activation.engine import compute_activation
from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, MicrogliaRecord, PhaseResult
from engram.utils.text_guards import is_meta_summary

logger = logging.getLogger(__name__)

# Type pairs considered incompatible for edges
_INCOMPATIBLE_TYPE_PAIRS: set[frozenset[str]] = {
    frozenset({"Person", "Software"}),
    frozenset({"Person", "File"}),
    frozenset({"Person", "Repository"}),
    frozenset({"Person", "CodeModule"}),
    frozenset({"Person", "Package"}),
    frozenset({"Person", "Library"}),
    frozenset({"Person", "Framework"}),
    frozenset({"Person", "API"}),
    frozenset({"Person", "Endpoint"}),
}

# Predicates that legitimately connect incompatible type pairs
_ALLOWED_PREDICATES: set[str] = {
    "DEVELOPS",
    "USES",
    "CREATED",
    "MAINTAINS",
    "WORKS_ON",
    "WORKS_AT",
    "EXPERT_IN",
    "CONTRIBUTES_TO",
    "AUTHORED",
    "MANAGES",
    "OWNS",
    "DESIGNED",
}

# Generic predicates that indicate contamination
_GENERIC_PREDICATES: set[str] = {
    "RELATES_TO",
    "MENTIONED_WITH",
    "CO_OCCURS_WITH",
    "ASSOCIATED_WITH",
    "CONNECTED_TO",
}


class MicrogliaPhase(ConsolidationPhase):
    """Graph immune surveillance inspired by the brain's microglia cells.

    Uses a complement-mediated tagging system (C1q/C3/C4) to identify
    contaminated edges and summaries, then soft-demotes them through a
    multi-cycle safety process: Tag → Confirm → Demote.

    All detectors are deterministic (zero LLM cost).
    """

    @property
    def name(self) -> str:
        return "microglia"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        if not cfg.microglia_enabled:
            return set()
        return {
            "sample_edges",
            "get_entity",
            "get_active_neighbors_with_weights",
            "update_relationship_weight",
            "get_identity_core_entities",
            "find_entities",
            "update_entity",
        }

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
    ) -> tuple[PhaseResult, list[Any]]:
        t0 = time.perf_counter()

        if not cfg.microglia_enabled:
            return PhaseResult(phase=self.name, status="skipped", duration_ms=0.0), []

        # We need a consolidation store for complement tags
        # The engine passes stores via graph_store; we need the consolidation store
        # which is accessed through the engine. For now, we use a simpler approach:
        # the phase manages its own tag state via the graph_store's consolidation_store
        # attribute if available, otherwise skips tag lifecycle.
        consolidation_store = getattr(graph_store, "_consolidation_store", None)
        if consolidation_store is None:
            # Try to find it from the engine context - fallback to stateless mode
            consolidation_store = None

        records: list[MicrogliaRecord] = []
        now = time.time()

        # Load identity-core entities for protection
        identity_core_ids: set[str] = set()
        try:
            core_entities = await graph_store.get_identity_core_entities(group_id)
            identity_core_ids = {e.id for e in core_entities}
        except Exception:
            pass

        # Get cycle number (approximate from context)
        cycle_number = _extract_cycle_number(cycle_id)

        # --- Step 1: Clear tags where "don't eat me" signals appeared ---
        if consolidation_store is not None:
            cleared = await self._clear_protected_tags(
                consolidation_store=consolidation_store,
                graph_store=graph_store,
                activation_store=activation_store,
                group_id=group_id,
                identity_core_ids=identity_core_ids,
                cfg=cfg,
                now=now,
                cycle_id=cycle_id,
                records=records,
            )
        else:
            cleared = 0

        # --- Step 2: Demote confirmed + aged tags ---
        demoted = 0
        if consolidation_store is not None and not dry_run:
            demoted = await self._demote_confirmed_tags(
                consolidation_store=consolidation_store,
                graph_store=graph_store,
                group_id=group_id,
                cfg=cfg,
                cycle_number=cycle_number,
                cycle_id=cycle_id,
                context=context,
                records=records,
            )

        # --- Step 3: Scan sampled edges for new contamination ---
        tagged_edges = await self._scan_edges(
            graph_store=graph_store,
            search_index=search_index,
            consolidation_store=consolidation_store,
            group_id=group_id,
            cfg=cfg,
            cycle_number=cycle_number,
            cycle_id=cycle_id,
            dry_run=dry_run,
            records=records,
        )

        # --- Step 4: Re-evaluate unconfirmed tags ---
        confirmed = 0
        if consolidation_store is not None:
            confirmed = await self._reevaluate_unconfirmed(
                consolidation_store=consolidation_store,
                graph_store=graph_store,
                search_index=search_index,
                group_id=group_id,
                cfg=cfg,
                cycle_number=cycle_number,
                cycle_id=cycle_id,
                records=records,
            )

        # --- Step 5: Scan entity summaries for contamination ---
        repaired = await self._scan_summaries(
            graph_store=graph_store,
            consolidation_store=consolidation_store,
            group_id=group_id,
            cfg=cfg,
            cycle_number=cycle_number,
            cycle_id=cycle_id,
            dry_run=dry_run,
            context=context,
            records=records,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        total_affected = demoted + tagged_edges + confirmed + repaired + cleared
        return PhaseResult(
            phase=self.name,
            status="success",
            items_processed=(
                cfg.microglia_scan_edges_per_cycle + cfg.microglia_scan_entities_per_cycle
            ),
            items_affected=total_affected,
            duration_ms=round(elapsed, 1),
        ), records

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _score_c1q_domain(
        self,
        source_type: str,
        target_type: str,
        predicate: str,
        weight: float,
        has_episode_evidence: bool,
        cfg: ActivationConfig,
    ) -> float:
        """C1q Domain: type-incompatibility detector."""
        pair = frozenset({source_type, target_type})
        if pair not in _INCOMPATIBLE_TYPE_PAIRS:
            return 0.0

        # Allowed predicates are OK
        if predicate in _ALLOWED_PREDICATES:
            return 0.0

        score = 0.3  # Base score for incompatible type pair

        # Generic predicates boost score
        if predicate in _GENERIC_PREDICATES:
            score += 0.25

        # Low weight boosts score
        if weight < 0.3:
            score += 0.15

        # No episode evidence boosts score
        if not has_episode_evidence:
            score += 0.2

        return min(score, 1.0)

    async def _score_c1q_embedding(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        search_index,
        group_id: str,
    ) -> float:
        """C1q Embedding: cosine similarity < 0.15 with generic predicates."""
        if predicate not in _GENERIC_PREDICATES:
            return 0.0

        try:
            embeddings = await search_index.get_entity_embeddings(
                [source_id, target_id],
                group_id=group_id,
            )
            if source_id not in embeddings or target_id not in embeddings:
                return 0.0

            import numpy as np

            vec_a = np.array(embeddings[source_id], dtype=np.float32)
            vec_b = np.array(embeddings[target_id], dtype=np.float32)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

            if cosine < 0.15:
                return 0.7 - cosine  # Higher score for lower similarity
            return 0.0
        except Exception:
            return 0.0

    def _score_c3_summary(self, summary: str) -> tuple[float, str | None]:
        """C3 Summary: detect meta-contamination in entity summaries.

        Returns (score, cleaned_summary_or_none).
        """
        if not summary:
            return 0.0, None

        segments = [s.strip() for s in re.split(r"[;.]", summary) if s.strip()]
        if not segments:
            return 0.0, None

        meta_count = 0
        clean_segments: list[str] = []
        for seg in segments:
            if is_meta_summary(seg):
                meta_count += 1
            else:
                clean_segments.append(seg)

        if meta_count == 0:
            return 0.0, None

        score = meta_count / len(segments)

        # Dedup remaining segments (token-set Jaccard)
        deduped = _dedup_segments(clean_segments)
        cleaned = "; ".join(deduped) if deduped else ""

        return min(score, 1.0), cleaned if cleaned != summary else None

    # ------------------------------------------------------------------
    # Phase steps
    # ------------------------------------------------------------------

    async def _clear_protected_tags(
        self,
        consolidation_store,
        graph_store,
        activation_store,
        group_id: str,
        identity_core_ids: set[str],
        cfg: ActivationConfig,
        now: float,
        cycle_id: str,
        records: list[MicrogliaRecord],
    ) -> int:
        """Clear tags where 'don't eat me' signals appeared since tagging."""
        active_tags = await consolidation_store.get_active_complement_tags(group_id)
        cleared = 0

        for tag in active_tags:
            target_id = tag["target_id"]
            should_clear = False
            reason = ""

            # Check identity-core
            if target_id in identity_core_ids:
                should_clear = True
                reason = "identity_core"

            # For edge tags, check endpoint protection
            if not should_clear and tag["target_type"] == "edge":
                # We store edge target_id as "source_id:target_id:predicate"
                parts = target_id.split(":", 2)
                if len(parts) >= 2:
                    src_id, tgt_id = parts[0], parts[1]

                    # Check identity-core endpoints
                    if src_id in identity_core_ids or tgt_id in identity_core_ids:
                        should_clear = True
                        reason = "endpoint_identity_core"

                    # Check high activation
                    if not should_clear:
                        for eid in (src_id, tgt_id):
                            state = await activation_store.get_activation(eid)
                            if state:
                                act = compute_activation(state.access_history, now, cfg)
                                if act > 0.3:
                                    should_clear = True
                                    reason = f"high_activation_{act:.2f}"
                                    break

                    # Check semantic tier
                    if not should_clear:
                        for eid in (src_id, tgt_id):
                            entity = await graph_store.get_entity(eid, group_id)
                            if entity and getattr(entity, "mat_tier", None) == "semantic":
                                should_clear = True
                                reason = "semantic_tier"
                                break

            if should_clear:
                await consolidation_store.clear_complement_tag(tag["id"])
                cleared += 1
                records.append(
                    MicrogliaRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        target_type=tag["target_type"],
                        target_id=target_id,
                        action="cleared",
                        tag_type=tag["tag_type"],
                        score=tag["score"],
                        detail=f"Protected: {reason}",
                    )
                )

        return cleared

    async def _demote_confirmed_tags(
        self,
        consolidation_store,
        graph_store,
        group_id: str,
        cfg: ActivationConfig,
        cycle_number: int,
        cycle_id: str,
        context: CycleContext | None,
        records: list[MicrogliaRecord],
    ) -> int:
        """Soft-demote confirmed and aged tags."""
        confirmed_tags = await consolidation_store.get_confirmed_tags(
            min_age_cycles=cfg.microglia_min_cycles_to_demote,
            current_cycle=cycle_number,
            group_id=group_id,
        )

        demoted = 0
        for tag in confirmed_tags:
            if demoted >= cfg.microglia_max_demotions_per_cycle:
                break

            target_id = tag["target_id"]
            tag_type = tag["tag_type"]

            if tag["target_type"] == "edge":
                # Soft-demote edge: reduce weight to 10% and set uncertain polarity
                parts = target_id.split(":", 2)
                if len(parts) >= 3:
                    src_id, tgt_id, predicate = parts[0], parts[1], parts[2]

                    # Get current weight
                    neighbors = await graph_store.get_active_neighbors_with_weights(
                        src_id,
                        group_id=group_id,
                    )
                    current_weight = None
                    for nid, w, pred, *_ in neighbors:
                        if nid == tgt_id and pred == predicate:
                            current_weight = w
                            break

                    if current_weight is not None and current_weight > 0:
                        # Reduce to 10% of current weight
                        delta = -(current_weight * 0.9)
                        await graph_store.update_relationship_weight(
                            src_id,
                            tgt_id,
                            delta,
                            max_weight=3.0,
                            group_id=group_id,
                            predicate=predicate,
                        )
                        demoted += 1

                        if context is not None:
                            context.microglia_demoted_edge_ids.add(target_id)

                        records.append(
                            MicrogliaRecord(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                target_type="edge",
                                target_id=target_id,
                                action="demoted",
                                tag_type=tag_type,
                                score=tag["score"],
                                detail=f"Weight {current_weight:.3f} → {current_weight * 0.1:.3f}",
                            )
                        )

            # Clear the tag after demotion
            await consolidation_store.clear_complement_tag(tag["id"])

        if demoted:
            logger.info("Microglia: demoted %d confirmed contaminated edges", demoted)
        return demoted

    async def _scan_edges(
        self,
        graph_store,
        search_index,
        consolidation_store,
        group_id: str,
        cfg: ActivationConfig,
        cycle_number: int,
        cycle_id: str,
        dry_run: bool,
        records: list[MicrogliaRecord],
    ) -> int:
        """Scan sampled edges for contamination and create tags."""
        edges = await graph_store.sample_edges(
            group_id=group_id,
            limit=cfg.microglia_scan_edges_per_cycle,
        )

        tagged = 0
        for edge in edges:
            # Get entity types
            source_entity = await graph_store.get_entity(edge.source_id, group_id)
            target_entity = await graph_store.get_entity(edge.target_id, group_id)
            if not source_entity or not target_entity:
                continue

            source_type = source_entity.entity_type or ""
            target_type = target_entity.entity_type or ""
            has_evidence = bool(edge.source_episode)

            # C1q Domain score
            domain_score = self._score_c1q_domain(
                source_type,
                target_type,
                edge.predicate,
                edge.weight,
                has_evidence,
                cfg,
            )

            # C1q Embedding score (only for generic predicates)
            embedding_score = 0.0
            if domain_score < cfg.microglia_tag_threshold and edge.predicate in _GENERIC_PREDICATES:
                embedding_score = await self._score_c1q_embedding(
                    edge.source_id,
                    edge.target_id,
                    edge.predicate,
                    search_index,
                    group_id,
                )

            # Take the max score
            best_score = max(domain_score, embedding_score)
            if best_score < cfg.microglia_tag_threshold:
                continue

            tag_type = "c1q_domain" if domain_score >= embedding_score else "c1q_embedding"
            edge_key = f"{edge.source_id}:{edge.target_id}:{edge.predicate}"

            if not dry_run and consolidation_store is not None:
                await consolidation_store.create_complement_tag(
                    target_type="edge",
                    target_id=edge_key,
                    tag_type=tag_type,
                    score=best_score,
                    cycle_tagged=cycle_number,
                    group_id=group_id,
                )

            tagged += 1
            records.append(
                MicrogliaRecord(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    target_type="edge",
                    target_id=edge_key,
                    action="tagged",
                    tag_type=tag_type,
                    score=best_score,
                    detail=f"{source_type}→{edge.predicate}→{target_type} w={edge.weight:.2f}",
                )
            )

        if tagged:
            logger.info("Microglia: tagged %d suspicious edges from %d scanned", tagged, len(edges))
        return tagged

    async def _reevaluate_unconfirmed(
        self,
        consolidation_store,
        graph_store,
        search_index,
        group_id: str,
        cfg: ActivationConfig,
        cycle_number: int,
        cycle_id: str,
        records: list[MicrogliaRecord],
    ) -> int:
        """Re-evaluate unconfirmed tags and promote to confirmed if still suspicious."""
        unconfirmed = await consolidation_store.get_unconfirmed_tags(
            max_cycle=cycle_number,
            group_id=group_id,
        )

        confirmed = 0
        for tag in unconfirmed:
            target_id = tag["target_id"]

            if tag["target_type"] == "edge":
                parts = target_id.split(":", 2)
                if len(parts) < 3:
                    await consolidation_store.clear_complement_tag(tag["id"])
                    continue

                src_id, tgt_id, predicate = parts[0], parts[1], parts[2]

                # Re-check: does the edge still exist and still look suspicious?
                source_entity = await graph_store.get_entity(src_id, group_id)
                target_entity = await graph_store.get_entity(tgt_id, group_id)

                if not source_entity or not target_entity:
                    # Entity was deleted — clear the tag
                    await consolidation_store.clear_complement_tag(tag["id"])
                    continue

                # Re-score
                source_type = source_entity.entity_type or ""
                target_type = target_entity.entity_type or ""

                # Find current weight
                current_weight = 0.0
                neighbors = await graph_store.get_active_neighbors_with_weights(
                    src_id,
                    group_id=group_id,
                )
                for nid, w, pred, *_ in neighbors:
                    if nid == tgt_id and pred == predicate:
                        current_weight = w
                        break

                # Check if edge has evidence now
                has_evidence = False
                try:
                    rels = await graph_store.get_relationships(
                        src_id,
                        direction="outgoing",
                        predicate=predicate,
                        group_id=group_id,
                    )
                    for r in rels:
                        if r.target_id == tgt_id and r.source_episode:
                            has_evidence = True
                            break
                except Exception:
                    pass

                rescore = self._score_c1q_domain(
                    source_type,
                    target_type,
                    predicate,
                    current_weight,
                    has_evidence,
                    cfg,
                )

                if rescore >= cfg.microglia_confirm_threshold:
                    await consolidation_store.confirm_complement_tag(tag["id"], cycle_number)
                    confirmed += 1
                    records.append(
                        MicrogliaRecord(
                            cycle_id=cycle_id,
                            group_id=group_id,
                            target_type="edge",
                            target_id=target_id,
                            action="confirmed",
                            tag_type=tag["tag_type"],
                            score=rescore,
                            detail=f"Re-scored {rescore:.2f} >= {cfg.microglia_confirm_threshold}",
                        )
                    )
                else:
                    await consolidation_store.clear_complement_tag(tag["id"])
                    records.append(
                        MicrogliaRecord(
                            cycle_id=cycle_id,
                            group_id=group_id,
                            target_type="edge",
                            target_id=target_id,
                            action="cleared",
                            tag_type=tag["tag_type"],
                            score=rescore,
                            detail=f"Re-scored {rescore:.2f} < {cfg.microglia_confirm_threshold}",
                        )
                    )

            elif tag["target_type"] == "entity":
                # Re-check summary contamination
                entity = await graph_store.get_entity(target_id, group_id)
                if not entity or not entity.summary:
                    await consolidation_store.clear_complement_tag(tag["id"])
                    continue

                score, _ = self._score_c3_summary(entity.summary)
                if score >= cfg.microglia_confirm_threshold:
                    await consolidation_store.confirm_complement_tag(tag["id"], cycle_number)
                    confirmed += 1
                else:
                    await consolidation_store.clear_complement_tag(tag["id"])

        if confirmed:
            logger.info(
                "Microglia: confirmed %d tags from %d unconfirmed",
                confirmed,
                len(unconfirmed),
            )
        return confirmed

    async def _scan_summaries(
        self,
        graph_store,
        consolidation_store,
        group_id: str,
        cfg: ActivationConfig,
        cycle_number: int,
        cycle_id: str,
        dry_run: bool,
        context: CycleContext | None,
        records: list[MicrogliaRecord],
    ) -> int:
        """Scan entity summaries for meta-contamination."""
        entities = await graph_store.find_entities(
            group_id=group_id,
            limit=cfg.microglia_scan_entities_per_cycle,
        )

        repaired = 0
        for entity in entities:
            if not entity.summary:
                continue

            score, cleaned = self._score_c3_summary(entity.summary)
            if score < cfg.microglia_tag_threshold:
                continue

            if cleaned is not None and not dry_run:
                # Direct repair for summary contamination (no multi-cycle needed)
                await graph_store.update_entity(
                    entity.id,
                    {"summary": cleaned},
                    group_id=group_id,
                )
                repaired += 1

                if context is not None:
                    context.microglia_repaired_entity_ids.add(entity.id)

                records.append(
                    MicrogliaRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        target_type="entity_summary",
                        target_id=entity.id,
                        action="repaired",
                        tag_type="c3_summary",
                        score=score,
                        detail=f"Removed {int(score * 100)}% meta segments from '{entity.name}'",
                    )
                )
            elif cleaned is None and score >= cfg.microglia_tag_threshold:
                # Entirely contaminated summary — tag for review
                if consolidation_store is not None and not dry_run:
                    await consolidation_store.create_complement_tag(
                        target_type="entity",
                        target_id=entity.id,
                        tag_type="c3_summary",
                        score=score,
                        cycle_tagged=cycle_number,
                        group_id=group_id,
                    )
                records.append(
                    MicrogliaRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        target_type="entity_summary",
                        target_id=entity.id,
                        action="tagged",
                        tag_type="c3_summary",
                        score=score,
                        detail=f"Fully contaminated summary for '{entity.name}'",
                    )
                )

        if repaired:
            logger.info("Microglia: repaired %d contaminated summaries", repaired)
        return repaired


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_cycle_number(cycle_id: str) -> int:
    """Extract a numeric cycle identifier for tag lifecycle tracking."""
    # cycle_id format: "cyc_<hex>" — use hash for deterministic numbering
    try:
        return int(cycle_id.replace("cyc_", ""), 16) % 1_000_000
    except (ValueError, AttributeError):
        return int(time.time())


def _dedup_segments(segments: list[str], threshold: float = 0.6) -> list[str]:
    """Remove near-duplicate segments using token-set Jaccard similarity."""
    if not segments:
        return []

    def _tokens(text: str) -> set[str]:
        return {w.lower() for w in re.findall(r"\b\w{3,}\b", text)}

    result: list[str] = []
    result_token_sets: list[set[str]] = []

    for seg in segments:
        seg_tokens = _tokens(seg)
        if not seg_tokens:
            continue

        is_dup = False
        for existing_tokens in result_token_sets:
            intersection = seg_tokens & existing_tokens
            union = seg_tokens | existing_tokens
            if union and len(intersection) / len(union) >= threshold:
                is_dup = True
                break

        if not is_dup:
            result.append(seg)
            result_token_sets.append(seg_tokens)

    return result
