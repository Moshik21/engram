"""Schema Formation phase: detect recurring motifs and promote to Schema entities."""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import defaultdict
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.maturity_features import get_cached_maturity_features
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, PhaseResult, SchemaRecord
from engram.models.entity import Entity

logger = logging.getLogger(__name__)


def compute_fingerprint(
    entity_type: str,
    relationships: list,
    entity_cache: dict[str, Entity],
    entity_id: str,
) -> frozenset[tuple[str, str, str]]:
    """Compute structural fingerprint for an entity given its relationships.

    Returns a frozenset of (source_type, predicate, target_type) triples
    describing the entity's local neighborhood shape.
    """
    triples: set[tuple[str, str, str]] = set()
    for rel in relationships:
        if rel.source_id == entity_id:
            target = entity_cache.get(rel.target_id)
            if target:
                triples.add((entity_type, rel.predicate, target.entity_type))
        else:
            source = entity_cache.get(rel.source_id)
            if source:
                triples.add((source.entity_type, rel.predicate, entity_type))
    return canonicalize_fingerprint(frozenset(triples))


def canonicalize_fingerprint(
    fingerprint: frozenset[tuple[str, str, str]],
) -> frozenset[tuple[str, str, str]]:
    """Normalize case and whitespace so equivalent motifs collapse together."""
    normalized = {
        (
            str(src_type).strip(),
            str(predicate).strip().upper(),
            str(tgt_type).strip(),
        )
        for src_type, predicate, tgt_type in fingerprint
    }
    return frozenset(normalized)


def _generate_schema_name(fingerprint: frozenset[tuple[str, str, str]]) -> str:
    """Generate a human-readable name from a fingerprint."""
    sorted_triples = sorted(fingerprint)
    if not sorted_triples:
        return "EmptySchema"
    # Use the source type of the first triple as the root
    root_type = sorted_triples[0][0]
    parts = []
    for src_type, predicate, tgt_type in sorted_triples:
        parts.append(f"{predicate}-{tgt_type}")
    return f"{root_type}: {', '.join(parts)}"


def _fingerprint_to_members(
    fingerprint: frozenset[tuple[str, str, str]],
) -> list[dict]:
    """Convert fingerprint triples to schema_members rows."""
    members = []
    for i, (src_type, predicate, tgt_type) in enumerate(sorted(fingerprint)):
        members.append({
            "role_label": f"r{i}_{src_type}_{predicate}_{tgt_type}",
            "member_type": tgt_type,
            "member_predicate": predicate,
        })
    return members


def _schema_matches_fingerprint(
    schema_members: list[dict],
    fingerprint: frozenset[tuple[str, str, str]],
) -> bool:
    """Check if existing schema members match a fingerprint."""
    expected = _fingerprint_to_members(canonicalize_fingerprint(fingerprint))
    if len(schema_members) != len(expected):
        return False
    existing_set = {
        (m["role_label"], m["member_type"], m["member_predicate"])
        for m in schema_members
    }
    expected_set = {
        (m["role_label"], m["member_type"], m["member_predicate"])
        for m in expected
    }
    return existing_set == expected_set


def _get_schema_support(
    entity: Entity,
    context: CycleContext | None,
    cfg: ActivationConfig,
) -> dict[str, Any]:
    attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
    bundle = get_cached_maturity_features(entity, context)
    mature = False
    if attrs.get("mat_tier") in {"transitional", "semantic"}:
        mature = True
    elif context is not None and entity.id in context.matured_entity_ids:
        mature = True
    elif isinstance(bundle, dict):
        try:
            mature = (
                float(bundle.get("maturity_score", 0.0))
                >= cfg.maturation_transitional_threshold
            )
        except (TypeError, ValueError):
            mature = False

    source_diverse = False
    time_recurrent = False
    if isinstance(bundle, dict):
        try:
            source_diverse = int(bundle.get("episode_count", 0)) >= 2
            time_recurrent = int(bundle.get("support_windows", 0)) >= 2
            maturity_score = float(bundle.get("maturity_score", 0.0))
        except (TypeError, ValueError):
            source_diverse = False
            time_recurrent = False
            maturity_score = 0.0
    else:
        maturity_score = 0.0

    return {
        "has_bundle": isinstance(bundle, dict),
        "mature": mature,
        "source_diverse": source_diverse,
        "time_recurrent": time_recurrent,
        "maturity_score": maturity_score,
    }


def _summarize_candidate_support(
    instance_ids: list[str],
    entity_cache: dict[str, Entity],
    context: CycleContext | None,
    cfg: ActivationConfig,
) -> dict[str, Any]:
    mature_instances = 0
    source_diverse_instances = 0
    time_recurrent_instances = 0
    bundle_instances = 0
    available_instances = 0

    for entity_id in instance_ids:
        entity = entity_cache.get(entity_id)
        if entity is None:
            continue
        support = _get_schema_support(entity, context, cfg)
        if support["mature"]:
            mature_instances += 1
            available_instances += 1
        elif support["has_bundle"]:
            available_instances += 1
        if support["has_bundle"]:
            bundle_instances += 1
        if support["source_diverse"]:
            source_diverse_instances += 1
        if support["time_recurrent"]:
            time_recurrent_instances += 1

    stable_threshold = max(1, math.ceil(cfg.schema_min_instances / 2))
    recurrence_instances = max(source_diverse_instances, time_recurrent_instances)
    support_available = available_instances > 0
    recurrence_available = bundle_instances >= stable_threshold
    passes = len(instance_ids) >= cfg.schema_min_instances
    if support_available and mature_instances < stable_threshold:
        passes = False
    if passes and recurrence_available and recurrence_instances < stable_threshold:
        passes = False

    return {
        "support_available": support_available,
        "recurrence_available": recurrence_available,
        "passes": passes,
        "stable_threshold": stable_threshold,
        "mature_instances": mature_instances,
        "source_diverse_instances": source_diverse_instances,
        "time_recurrent_instances": time_recurrent_instances,
        "recurrence_instances": recurrence_instances,
        "bundle_instances": bundle_instances,
        "instance_count": len(instance_ids),
    }


def _support_reason(summary: dict[str, Any]) -> str:
    return (
        f"stable motif across {summary['instance_count']} instances "
        f"({summary['mature_instances']} mature, "
        f"{summary['source_diverse_instances']} multi-source, "
        f"{summary['time_recurrent_instances']} multi-window)"
    )


class SchemaFormationPhase(ConsolidationPhase):
    """Detect recurring structural motifs and promote to Schema entities."""

    @property
    def name(self) -> str:
        return "schema"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        if not cfg.schema_formation_enabled:
            return set()
        return {"find_entities_by_type", "get_schema_members", "save_schema_members"}

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

        if not cfg.schema_formation_enabled:
            return PhaseResult(
                phase=self.name, status="skipped", duration_ms=_elapsed_ms(t0),
            ), []

        # 1. Scan entities — prefer mature entities if available
        entities: list[Entity] = await graph_store.find_entities(
            group_id=group_id, limit=cfg.schema_max_entities_scan,
        )

        if not entities:
            return PhaseResult(
                phase=self.name, status="skipped",
                items_processed=0, duration_ms=_elapsed_ms(t0),
            ), []

        # Filter to non-deleted, non-Schema entities
        entities = [
            e for e in entities
            if e.deleted_at is None and e.entity_type != "Schema"
        ]

        # Build entity cache
        entity_cache: dict[str, Entity] = {e.id: e for e in entities}

        # 2. Compute fingerprints
        motif_counter: dict[frozenset, list[str]] = defaultdict(list)
        scanned = 0

        for entity in entities:
            rels = await graph_store.get_relationships(
                entity.id, direction="both", group_id=group_id,
            )
            if len(rels) < cfg.schema_min_edges:
                scanned += 1
                continue

            fp = compute_fingerprint(
                entity.entity_type, rels, entity_cache, entity.id,
            )
            if len(fp) >= cfg.schema_min_edges:
                motif_counter[fp].append(entity.id)
            scanned += 1

        # 3. Filter candidates
        candidates: list[tuple[frozenset, list[str], dict[str, Any]]] = []
        for fp, eids in motif_counter.items():
            support = _summarize_candidate_support(eids, entity_cache, context, cfg)
            if support["passes"]:
                candidates.append((fp, eids, support))
        # Prefer stable, recurrent motifs over raw instance count.
        candidates.sort(
            key=lambda item: (
                item[2]["mature_instances"],
                item[2]["recurrence_instances"],
                len(item[1]),
            ),
            reverse=True,
        )

        # 4. Check existing schemas and create/reinforce
        records: list[SchemaRecord] = []
        created = 0

        existing_schemas = await graph_store.find_entities_by_type(
            "Schema", group_id, limit=200,
        )

        for fp, instance_ids, support_summary in candidates:
            if created >= cfg.schema_max_per_cycle:
                break

            # Check if a matching schema already exists
            matched_schema = None
            for schema_entity in existing_schemas:
                members = await graph_store.get_schema_members(
                    schema_entity.id, group_id,
                )
                if _schema_matches_fingerprint(members, fp):
                    matched_schema = schema_entity
                    break

            if matched_schema:
                # Reinforce existing schema
                if not dry_run:
                    await activation_store.record_access(
                        matched_schema.id, time.time(), group_id,
                    )
                    attrs = (
                        matched_schema.attributes
                        if isinstance(matched_schema.attributes, dict)
                        else {}
                    )
                    attrs["instance_count"] = len(instance_ids)
                    attrs["promotion_reason"] = _support_reason(support_summary)
                    attrs["support_policy_version"] = "schema_support_v2"
                    attrs["support_summary"] = support_summary
                    await graph_store.update_entity(
                        matched_schema.id,
                        {"attributes": json.dumps(attrs)},
                        group_id,
                    )
                records.append(SchemaRecord(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    schema_entity_id=matched_schema.id,
                    schema_name=matched_schema.name,
                    instance_count=len(instance_ids),
                    predicate_count=len(fp),
                    action="reinforced",
                ))
                continue

            # Create new schema
            schema_id = f"schema_{uuid.uuid4().hex[:12]}"
            schema_name = _generate_schema_name(fp)

            if not dry_run:
                schema_entity = Entity(
                    id=schema_id,
                    name=schema_name,
                    entity_type="Schema",
                    summary=(
                        f"Structural pattern shared by {len(instance_ids)} entities; "
                        f"{_support_reason(support_summary)}"
                    ),
                    attributes={
                        "schema_fingerprint": sorted([list(t) for t in fp]),
                        "instance_count": len(instance_ids),
                        "promotion_reason": _support_reason(support_summary),
                        "support_policy_version": "schema_support_v2",
                        "support_summary": support_summary,
                    },
                    group_id=group_id,
                )
                await graph_store.create_entity(schema_entity)

                # Save schema members
                members = _fingerprint_to_members(fp)
                await graph_store.save_schema_members(schema_id, members, group_id)

                # Create INSTANCE_OF edges
                from engram.models.relationship import Relationship

                for inst_id in instance_ids:
                    rel = Relationship(
                        id=f"rel_{uuid.uuid4().hex[:12]}",
                        source_id=inst_id,
                        target_id=schema_id,
                        predicate="INSTANCE_OF",
                        weight=1.0,
                        group_id=group_id,
                    )
                    await graph_store.create_relationship(rel)

                if context is not None:
                    context.schema_entity_ids.add(schema_id)
                    context.affected_entity_ids.add(schema_id)

            records.append(SchemaRecord(
                cycle_id=cycle_id,
                group_id=group_id,
                schema_entity_id=schema_id,
                schema_name=schema_name,
                instance_count=len(instance_ids),
                predicate_count=len(fp),
                action="created",
            ))
            created += 1

        return PhaseResult(
            phase=self.name,
            items_processed=scanned,
            items_affected=len(records),
            duration_ms=_elapsed_ms(t0),
        ), records


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
