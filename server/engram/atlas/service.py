"""Atlas snapshot lifecycle and region drill-down service."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from engram.models.atlas import AtlasSnapshot


class AtlasService:
    """Loads, refreshes, and materializes atlas read models."""

    def __init__(
        self,
        atlas_store,
        builder,
        graph_store,
        *,
        rebuild_threshold: float = 0.05,
        max_snapshot_age_seconds: int = 300,
    ) -> None:
        self._store = atlas_store
        self._builder = builder
        self._graph = graph_store
        self._rebuild_threshold = rebuild_threshold
        self._max_snapshot_age_seconds = max_snapshot_age_seconds

    async def get_snapshot(
        self,
        group_id: str,
        *,
        force: bool = False,
        snapshot_id: str | None = None,
    ) -> AtlasSnapshot:
        if snapshot_id:
            snapshot = await self._store.get_snapshot(snapshot_id, group_id)
            if snapshot is None:
                raise LookupError(f"Snapshot '{snapshot_id}' not found")
            return snapshot

        latest = await self._store.get_latest_snapshot(group_id)
        if force or latest is None or await self._needs_rebuild(latest, group_id):
            snapshot = await self._builder.build(group_id, previous_snapshot=latest)
            await self._store.save_snapshot(snapshot)
            return snapshot
        return latest

    async def list_snapshots(
        self,
        group_id: str,
        limit: int = 24,
    ) -> list:
        return await self._store.list_snapshots(group_id, limit=limit)

    async def get_region_payload(
        self,
        group_id: str,
        region_id: str,
        *,
        force: bool = False,
        snapshot_id: str | None = None,
    ) -> dict | None:
        snapshot = await self.get_snapshot(
            group_id,
            force=force,
            snapshot_id=snapshot_id,
        )
        region = next((item for item in snapshot.regions if item.id == region_id), None)
        if region is None:
            return None

        member_ids = await self._store.get_region_members(snapshot.id, region_id, group_id)
        connected_regions = self._connected_regions(snapshot, region_id)
        hub_details = []
        for entity_id in region.hub_entity_ids[:6]:
            entity = await self._graph.get_entity(entity_id, group_id)
            if entity is None:
                continue
            activation_current = (
                region.activation_score
                if entity.id == region.center_entity_id
                else max(region.activation_score * 0.75, 0.05)
            )
            hub_details.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entityType": entity.entity_type,
                    "activationCurrent": activation_current,
                }
            )

        nodes = [
            {
                "id": region.id,
                "kind": "cluster",
                "label": region.label,
                "representedEntityCount": region.member_count,
                "activationScore": region.activation_score,
                "growth30d": region.growth_30d,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "regionId": region.id,
            }
        ]
        edges = []

        hub_count = len(hub_details)
        for index, hub in enumerate(hub_details):
            angle = (-math.pi / 2.0) + ((2.0 * math.pi * index) / max(hub_count, 1))
            nodes.append(
                {
                    "id": f"hub:{hub['id']}",
                    "kind": "hub",
                    "label": hub["name"],
                    "representedEntityCount": 1,
                    "activationScore": hub["activationCurrent"],
                    "growth30d": 0,
                    "x": round(math.cos(angle) * 0.66, 4),
                    "y": round(math.sin(angle) * 0.42, 4),
                    "z": 0.0,
                    "entityId": hub["id"],
                    "entityType": hub["entityType"],
                }
            )
            edges.append(
                {
                    "id": f"{region.id}::hub:{hub['id']}",
                    "source": region.id,
                    "target": f"hub:{hub['id']}",
                    "weight": 1.0,
                    "predicateHint": "represents",
                }
            )

        bridge_count = len(connected_regions)
        for index, bridge in enumerate(connected_regions[:4]):
            if bridge_count > 1:
                angle = (-math.pi / 2.0) + (
                    (math.pi * index) / max(bridge_count - 1, 1)
                )
            else:
                angle = 0.0
            nodes.append(
                {
                    "id": f"bridge:{bridge['region'].id}",
                    "kind": "bridge",
                    "label": bridge["region"].label,
                    "representedEntityCount": bridge["region"].member_count,
                    "activationScore": bridge["region"].activation_score,
                    "growth30d": bridge["region"].growth_30d,
                    "x": round(math.cos(angle) * 0.86, 4),
                    "y": round(math.sin(angle) * 0.52, 4),
                    "z": 0.0,
                    "regionId": bridge["region"].id,
                }
            )
            edges.append(
                {
                    "id": f"{region.id}::bridge:{bridge['region'].id}",
                    "source": region.id,
                    "target": f"bridge:{bridge['region'].id}",
                    "weight": bridge["weight"],
                    "predicateHint": "bridge",
                }
            )

        return {
            "representation": {
                "scope": "region",
                "layout": "precomputed",
                "representedEntityCount": region.member_count,
                "representedEdgeCount": region.represented_edge_count,
                "displayedNodeCount": len(nodes),
                "displayedEdgeCount": len(edges),
                "truncated": False,
                "snapshotId": snapshot.id,
            },
            "generatedAt": snapshot.generated_at,
            "region": {
                "id": region.id,
                "label": region.label,
                "subtitle": region.subtitle,
                "kind": region.kind,
                "memberCount": region.member_count,
                "activationScore": region.activation_score,
                "growth7d": region.growth_7d,
                "growth30d": region.growth_30d,
                "latestEntityCreatedAt": region.latest_entity_created_at,
            },
            "nodes": nodes,
            "edges": edges,
            "topEntities": hub_details,
            "memberIds": member_ids,
        }

    async def _needs_rebuild(self, snapshot: AtlasSnapshot, group_id: str) -> bool:
        if not snapshot.generated_at:
            return True
        generated_at = datetime.fromisoformat(snapshot.generated_at)
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)
        age_seconds = (datetime.now(UTC) - generated_at).total_seconds()
        if age_seconds > self._max_snapshot_age_seconds:
            return True

        stats = await self._graph.get_stats(group_id=group_id)
        entity_delta = self._relative_delta(
            snapshot.total_entities,
            int(stats.get("entities") or 0),
        )
        rel_delta = self._relative_delta(
            snapshot.total_relationships,
            int(stats.get("relationships") or 0),
        )
        return entity_delta > self._rebuild_threshold or rel_delta > self._rebuild_threshold

    def _relative_delta(self, baseline: int, current: int) -> float:
        if baseline <= 0:
            return 1.0 if current > 0 else 0.0
        return abs(current - baseline) / baseline

    def _connected_regions(
        self,
        snapshot: AtlasSnapshot,
        region_id: str,
    ) -> list[dict]:
        region_map = {region.id: region for region in snapshot.regions}
        connected: list[dict] = []
        for bridge in snapshot.bridges:
            if bridge.source == region_id:
                target = region_map.get(bridge.target)
            elif bridge.target == region_id:
                target = region_map.get(bridge.source)
            else:
                continue
            if target is None:
                continue
            connected.append({"region": target, "weight": bridge.weight})
        connected.sort(key=lambda item: (-item["weight"], item["region"].label))
        return connected
