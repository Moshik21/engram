"""Builds stable, materialized atlas snapshots from the live graph."""

from __future__ import annotations

import math
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from engram.activation.engine import compute_activation
from engram.models.atlas import AtlasBridge, AtlasRegion, AtlasSnapshot


@dataclass
class _CandidateRegion:
    seed_id: str
    member_ids: list[str]


class AtlasBuilder:
    """Build atlas snapshots using deterministic graph summaries."""

    def __init__(
        self,
        graph_store,
        activation_store,
        cfg,
        *,
        community_store=None,
        max_regions: int = 48,
        bridge_limit: int = 128,
        overlap_threshold: float = 0.45,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._cfg = cfg
        self._community_store = community_store
        self._max_regions = max_regions
        self._bridge_limit = bridge_limit
        self._overlap_threshold = overlap_threshold

    async def build(
        self,
        group_id: str,
        *,
        previous_snapshot: AtlasSnapshot | None = None,
    ) -> AtlasSnapshot:
        now = time.time()
        stats = await self._graph.get_stats(group_id=group_id)
        total_entities = int(stats.get("entities") or 0)
        total_relationships = int(stats.get("relationships") or 0)

        if total_entities <= 0:
            return AtlasSnapshot(
                group_id=group_id,
                represented_entity_count=0,
                represented_edge_count=0,
                displayed_node_count=0,
                displayed_edge_count=0,
                total_entities=0,
                total_relationships=0,
                total_regions=0,
            )

        all_entities = await self._graph.find_entities(
            group_id=group_id,
            limit=max(total_entities, 1),
        )
        if not all_entities:
            return AtlasSnapshot(
                group_id=group_id,
                represented_entity_count=0,
                represented_edge_count=0,
                displayed_node_count=0,
                displayed_edge_count=0,
                total_entities=0,
                total_relationships=0,
                total_regions=0,
            )

        entity_map = {entity.id: entity for entity in all_entities}
        entity_ids = list(entity_map)
        states = await self._activation.batch_get(entity_ids)
        node_map = {
            entity.id: self._build_node(entity, states.get(entity.id), now)
            for entity in all_entities
        }

        candidate_regions = await self._build_candidate_regions(group_id, all_entities)
        stable_region_ids = self._assign_region_ids(candidate_regions, previous_snapshot)

        region_members: dict[str, list[Any]] = defaultdict(list)
        entity_region: dict[str, str] = {}
        region_member_ids: dict[str, list[str]] = {}
        for candidate in candidate_regions:
            region_id = stable_region_ids[candidate.seed_id]
            members = [
                entity_map[entity_id]
                for entity_id in candidate.member_ids
                if entity_id in entity_map
            ]
            region_members[region_id].extend(members)
            ids = [entity.id for entity in members]
            region_member_ids[region_id] = ids
            for entity_id in ids:
                entity_region[entity_id] = region_id

        degree_counter: Counter[str] = Counter()
        internal_edge_counts: Counter[str] = Counter()
        bridge_weights: dict[tuple[str, str], dict[str, float | int]] = {}

        all_edges = await self._graph.get_all_edges(
            group_id=group_id,
            entity_ids=set(entity_ids),
            limit=max(total_relationships, len(entity_ids) * 4, 1),
        )
        for rel in all_edges:
            degree_counter[rel.source_id] += 1
            degree_counter[rel.target_id] += 1
            src_region = entity_region.get(rel.source_id)
            dst_region = entity_region.get(rel.target_id)
            if not src_region or not dst_region:
                continue
            if src_region == dst_region:
                internal_edge_counts[src_region] += 1
                continue
            source_region, target_region = sorted((src_region, dst_region))
            key = (source_region, target_region)
            bucket = bridge_weights.setdefault(
                key,
                {"weight": 0.0, "relationshipCount": 0},
            )
            bucket["weight"] = float(bucket["weight"]) + float(rel.weight or 0.0)
            bucket["relationshipCount"] = int(bucket["relationshipCount"]) + 1

        week_cutoff = now - (7 * 86400)
        month_cutoff = now - (30 * 86400)
        ranked_region_ids = sorted(
            region_members,
            key=lambda region_id: (-len(region_members[region_id]), region_id),
        )

        positions = self._layout_regions(
            ranked_region_ids,
            region_members,
            previous_snapshot,
        )

        regions: list[AtlasRegion] = []
        hottest_region_id = None
        hottest_activation = -1.0
        fastest_growing_region_id = None
        fastest_growth = -1

        for region_id in ranked_region_ids:
            members = region_members[region_id]
            dominant_entity_types = Counter(
                entity.entity_type or "Other" for entity in members
            )
            scored_members = sorted(
                members,
                key=lambda entity: (
                    -(node_map[entity.id]["activationCurrent"] * 4.0 + degree_counter[entity.id]),
                    entity.name or entity.id,
                ),
            )
            hub_entities = scored_members[:5]
            label, subtitle = self._region_label(
                members,
                dict(dominant_entity_types),
                [entity.name for entity in hub_entities if entity.name],
            )
            activation_score = round(
                max((node_map[entity.id]["activationCurrent"] for entity in members), default=0.0),
                4,
            )
            growth_7d = 0
            growth_30d = 0
            latest_entity_created_at = None
            for entity in members:
                entity_ts = self._entity_timestamp(entity)
                if entity_ts is None:
                    continue
                if entity_ts >= week_cutoff:
                    growth_7d += 1
                if entity_ts >= month_cutoff:
                    growth_30d += 1
                entity_created_at = getattr(entity, "created_at", None)
                if entity_created_at is None:
                    continue
                entity_created_iso = (
                    entity_created_at.isoformat()
                    if hasattr(entity_created_at, "isoformat")
                    else str(entity_created_at)
                )
                if (
                    latest_entity_created_at is None
                    or entity_created_iso > latest_entity_created_at
                ):
                    latest_entity_created_at = entity_created_iso

            if activation_score > hottest_activation:
                hottest_activation = activation_score
                hottest_region_id = region_id
            if growth_30d > fastest_growth:
                fastest_growth = growth_30d
                fastest_growing_region_id = region_id

            x, y, z = positions.get(region_id, (0.0, 0.0, 0.0))
            regions.append(
                AtlasRegion(
                    id=region_id,
                    label=label,
                    subtitle=subtitle,
                    kind=self._region_kind(members),
                    member_count=len(members),
                    represented_edge_count=int(internal_edge_counts[region_id]),
                    activation_score=activation_score,
                    growth_7d=growth_7d,
                    growth_30d=growth_30d,
                    dominant_entity_types=dict(dominant_entity_types),
                    hub_entity_ids=[entity.id for entity in hub_entities],
                    center_entity_id=hub_entities[0].id if hub_entities else None,
                    latest_entity_created_at=latest_entity_created_at,
                    x=x,
                    y=y,
                    z=z,
                )
            )

        bridges = [
            AtlasBridge(
                id=f"{source}::{target}",
                source=source,
                target=target,
                weight=round(float(meta["weight"]), 4),
                relationship_count=int(meta["relationshipCount"]),
            )
            for (source, target), meta in sorted(
                bridge_weights.items(),
                key=lambda item: (-float(item[1]["weight"]), item[0]),
            )[: self._bridge_limit]
        ]

        return AtlasSnapshot(
            group_id=group_id,
            represented_entity_count=total_entities,
            represented_edge_count=total_relationships,
            displayed_node_count=len(regions),
            displayed_edge_count=len(bridges),
            total_entities=total_entities,
            total_relationships=total_relationships,
            total_regions=len(regions),
            hottest_region_id=hottest_region_id,
            fastest_growing_region_id=fastest_growing_region_id,
            regions=regions,
            bridges=bridges,
            region_members=region_member_ids,
        )

    async def _build_candidate_regions(
        self,
        group_id: str,
        all_entities: list[Any],
    ) -> list[_CandidateRegion]:
        raw_assignments: dict[str, str | None] = {}
        if self._community_store and hasattr(self._community_store, "ensure_fresh"):
            entity_ids = [entity.id for entity in all_entities]
            await self._community_store.ensure_fresh(group_id, self._graph, entity_ids)
            raw_assignments = {
                entity_id: self._community_store.get_community(entity_id, group_id)
                for entity_id in entity_ids
            }

        community_counts = Counter(
            label for label in raw_assignments.values() if label is not None
        )
        buckets: dict[str, list[str]] = defaultdict(list)
        for entity in all_entities:
            raw_label = raw_assignments.get(entity.id)
            if raw_label and community_counts[raw_label] >= 4:
                candidate_id = f"community:{raw_label}"
            else:
                candidate_id = f"type:{entity.entity_type or 'Other'}"
            buckets[candidate_id].append(entity.id)

        if len(buckets) > self._max_regions:
            ranked = sorted(buckets, key=lambda bucket_id: (-len(buckets[bucket_id]), bucket_id))
            overflow_ids: list[str] = []
            for bucket_id in ranked[self._max_regions - 1 :]:
                overflow_ids.extend(buckets.pop(bucket_id))
            if overflow_ids:
                buckets["overflow:misc"] = overflow_ids

        return [
            _CandidateRegion(seed_id=seed_id, member_ids=member_ids)
            for seed_id, member_ids in sorted(
                buckets.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ]

    def _assign_region_ids(
        self,
        candidate_regions: list[_CandidateRegion],
        previous_snapshot: AtlasSnapshot | None,
    ) -> dict[str, str]:
        previous_members = previous_snapshot.region_members if previous_snapshot else {}
        unused_previous = set(previous_members)
        assigned: dict[str, str] = {}

        for candidate in candidate_regions:
            candidate_set = set(candidate.member_ids)
            best_region_id = None
            best_overlap = 0.0
            for region_id in list(unused_previous):
                prev_set = set(previous_members.get(region_id, []))
                if not prev_set:
                    continue
                overlap = len(candidate_set & prev_set) / len(candidate_set | prev_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_region_id = region_id
            if best_region_id and best_overlap >= self._overlap_threshold:
                assigned[candidate.seed_id] = best_region_id
                unused_previous.remove(best_region_id)
            else:
                assigned[candidate.seed_id] = f"region_{uuid.uuid4().hex[:12]}"
        return assigned

    def _layout_regions(
        self,
        region_ids: list[str],
        region_members: dict[str, list[Any]],
        previous_snapshot: AtlasSnapshot | None,
    ) -> dict[str, tuple[float, float, float]]:
        previous_positions = {}
        if previous_snapshot:
            previous_positions = {
                region.id: (region.x, region.y, region.z)
                for region in previous_snapshot.regions
            }

        identity_region_id = next(
            (
                region_id
                for region_id in region_ids
                if self._region_kind(region_members[region_id]) == "identity"
            ),
            None,
        )

        positions: dict[str, tuple[float, float, float]] = {}
        for region_id in region_ids:
            if region_id in previous_positions:
                positions[region_id] = previous_positions[region_id]

        new_region_ids = [region_id for region_id in region_ids if region_id not in positions]
        if not new_region_ids:
            return positions

        start_index = 0
        if identity_region_id and identity_region_id in new_region_ids:
            positions[identity_region_id] = (0.0, 0.0, 0.18)
            new_region_ids.remove(identity_region_id)
            start_index = 1

        count = len(new_region_ids)
        for index, region_id in enumerate(new_region_ids):
            angle = (-math.pi / 2.0) + ((2.0 * math.pi * (index + start_index)) / max(count, 1))
            radius = 0.82 if previous_positions else 0.72
            positions[region_id] = (
                round(math.cos(angle) * radius, 4),
                round(math.sin(angle) * radius, 4),
                0.0,
            )
        return positions

    def _build_node(self, entity: Any, state: Any, now: float) -> dict[str, Any]:
        activation_current = 0.0
        access_count = 0
        last_accessed = None
        if state and state.access_history:
            activation_current = compute_activation(state.access_history, now, self._cfg)
            access_count = state.access_count
            if state.last_accessed:
                last_accessed = state.last_accessed
        else:
            activation_current = getattr(entity, "activation_current", 0.0) or 0.0
            access_count = getattr(entity, "access_count", 0) or 0
            last_accessed = getattr(entity, "last_accessed", None)
        return {
            "id": entity.id,
            "activationCurrent": round(activation_current, 4),
            "accessCount": access_count,
            "lastAccessed": last_accessed,
        }

    def _entity_timestamp(self, entity: Any) -> float | None:
        created_at = getattr(entity, "created_at", None)
        if created_at is None:
            return None
        if hasattr(created_at, "timestamp"):
            return float(created_at.timestamp())
        return None

    def _region_kind(self, members: list[Any]) -> str:
        if any(bool(getattr(entity, "identity_core", False)) for entity in members):
            return "identity"
        return "mixed"

    def _region_label(
        self,
        members: list[Any],
        dominant_entity_types: dict[str, int],
        hub_names: list[str],
    ) -> tuple[str, str | None]:
        top_types = sorted(
            dominant_entity_types.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if top_types:
            top_type, top_count = top_types[0]
            if top_count >= max(3, math.ceil(len(members) * 0.6)):
                if top_type == "Person":
                    return "People", "Dominant entity type: Person"
                return top_type, f"Dominant entity type: {top_type}"

        if hub_names:
            if len(hub_names) == 1:
                return hub_names[0], f"{len(members)} memories in focus"
            return " / ".join(hub_names[:2]), f"{len(members)} memories in focus"

        return "Memory Region", f"{len(members)} memories in focus"
