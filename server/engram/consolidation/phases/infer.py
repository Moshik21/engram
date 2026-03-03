"""Edge inference phase: create MENTIONED_WITH edges for co-occurring entities."""

from __future__ import annotations

import logging
import time
import uuid

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, InferredEdge, PhaseResult
from engram.models.relationship import Relationship

logger = logging.getLogger(__name__)


class EdgeInferencePhase(ConsolidationPhase):
    """Find entity pairs that co-occur across episodes and create edges."""

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

        records: list[InferredEdge] = []
        for entity_a_id, entity_b_id, count in pairs:
            if len(records) >= max_edges:
                break

            # Confidence scaled by co-occurrence count
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
                infer_type="co_occurrence",
            ))

        # --- Transitivity pass ---
        if cfg.consolidation_infer_transitivity_enabled:
            trans_records = await _run_transitivity_pass(
                group_id, graph_store, cfg, cycle_id, dry_run,
                remaining=max_edges - len(records),
                context=context,
            )
            records.extend(trans_records)

        return PhaseResult(
            phase=self.name,
            items_processed=len(pairs) + (len(records) - len(pairs)),
            items_affected=len(records),
            duration_ms=_elapsed_ms(t0),
        ), records


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
