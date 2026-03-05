"""Schema Formation phase: detect recurring motifs and promote to Schema entities."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Any

from engram.config import ActivationConfig
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
    return frozenset(triples)


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
    expected = _fingerprint_to_members(fingerprint)
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


class SchemaFormationPhase(ConsolidationPhase):
    """Detect recurring structural motifs and promote to Schema entities."""

    @property
    def name(self) -> str:
        return "schema"

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
        candidates: list[tuple[frozenset, list[str]]] = [
            (fp, eids)
            for fp, eids in motif_counter.items()
            if len(eids) >= cfg.schema_min_instances
        ]
        # Sort by instance count descending for deterministic selection
        candidates.sort(key=lambda x: len(x[1]), reverse=True)

        # 4. Check existing schemas and create/reinforce
        records: list[SchemaRecord] = []
        created = 0

        existing_schemas = await graph_store.find_entities_by_type(
            "Schema", group_id, limit=200,
        )

        for fp, instance_ids in candidates:
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
                    summary=f"Structural pattern shared by {len(instance_ids)} entities",
                    attributes={
                        "schema_fingerprint": sorted([list(t) for t in fp]),
                        "instance_count": len(instance_ids),
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
