"""Redis-backed atlas snapshot store."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from engram.models.atlas import (
    AtlasBridge,
    AtlasRegion,
    AtlasSnapshot,
    AtlasSnapshotSummary,
)


class RedisAtlasStore:
    """Stores atlas snapshots as Redis JSON blobs."""

    def __init__(self, redis) -> None:
        self._redis = redis

    async def initialize(self, db=None) -> None:
        return None

    async def close(self) -> None:
        return None

    def _latest_key(self, group_id: str) -> str:
        return f"atlas:{group_id}:latest"

    def _snapshot_key(self, group_id: str, snapshot_id: str) -> str:
        return f"atlas:{group_id}:snapshot:{snapshot_id}"

    def _history_key(self, group_id: str) -> str:
        return f"atlas:{group_id}:history"

    async def get_latest_snapshot(self, group_id: str) -> AtlasSnapshot | None:
        snapshot_id = await self._redis.get(self._latest_key(group_id))
        if not snapshot_id:
            return None
        if isinstance(snapshot_id, bytes):
            snapshot_id = snapshot_id.decode()
        payload = await self._redis.get(self._snapshot_key(group_id, snapshot_id))
        if not payload:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode()
        return self._deserialize(payload)

    async def get_snapshot(
        self,
        snapshot_id: str,
        group_id: str,
    ) -> AtlasSnapshot | None:
        payload = await self._redis.get(self._snapshot_key(group_id, snapshot_id))
        if not payload:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode()
        return self._deserialize(payload)

    async def list_snapshots(
        self,
        group_id: str,
        limit: int = 24,
    ) -> list[AtlasSnapshotSummary]:
        snapshot_ids = await self._redis.zrevrange(self._history_key(group_id), 0, limit - 1)
        summaries: list[AtlasSnapshotSummary] = []
        for raw_snapshot_id in snapshot_ids:
            snapshot_id = (
                raw_snapshot_id.decode()
                if isinstance(raw_snapshot_id, bytes)
                else raw_snapshot_id
            )
            snapshot = await self.get_snapshot(snapshot_id, group_id)
            if snapshot is None:
                continue
            summaries.append(
                AtlasSnapshotSummary(
                    id=snapshot.id,
                    group_id=snapshot.group_id,
                    generated_at=snapshot.generated_at,
                    represented_entity_count=snapshot.represented_entity_count,
                    represented_edge_count=snapshot.represented_edge_count,
                    displayed_node_count=snapshot.displayed_node_count,
                    displayed_edge_count=snapshot.displayed_edge_count,
                    total_entities=snapshot.total_entities,
                    total_relationships=snapshot.total_relationships,
                    total_regions=snapshot.total_regions,
                    hottest_region_id=snapshot.hottest_region_id,
                    fastest_growing_region_id=snapshot.fastest_growing_region_id,
                    truncated=snapshot.truncated,
                )
            )
        return summaries

    async def save_snapshot(self, snapshot: AtlasSnapshot) -> None:
        payload = self._serialize(snapshot)
        generated_at = datetime.fromisoformat(snapshot.generated_at)
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        pipe = self._redis.pipeline()
        pipe.set(self._snapshot_key(snapshot.group_id, snapshot.id), payload)
        pipe.set(self._latest_key(snapshot.group_id), snapshot.id)
        pipe.zadd(
            self._history_key(snapshot.group_id),
            {snapshot.id: generated_at.timestamp()},
        )
        await pipe.execute()

    async def get_region_members(
        self,
        snapshot_id: str,
        region_id: str,
        group_id: str,
    ) -> list[str]:
        payload = await self._redis.get(self._snapshot_key(group_id, snapshot_id))
        if not payload:
            return []
        if isinstance(payload, bytes):
            payload = payload.decode()
        snapshot = self._deserialize(payload)
        return snapshot.region_members.get(region_id, [])

    def _serialize(self, snapshot: AtlasSnapshot) -> str:
        return json.dumps(asdict(snapshot))

    def _deserialize(self, payload: str) -> AtlasSnapshot:
        raw = json.loads(payload)
        regions = [AtlasRegion(**region) for region in raw.get("regions", [])]
        bridges = [AtlasBridge(**bridge) for bridge in raw.get("bridges", [])]
        return AtlasSnapshot(
            id=raw["id"],
            group_id=raw["group_id"],
            generated_at=raw["generated_at"],
            represented_entity_count=raw["represented_entity_count"],
            represented_edge_count=raw["represented_edge_count"],
            displayed_node_count=raw["displayed_node_count"],
            displayed_edge_count=raw["displayed_edge_count"],
            total_entities=raw["total_entities"],
            total_relationships=raw["total_relationships"],
            total_regions=raw["total_regions"],
            hottest_region_id=raw.get("hottest_region_id"),
            fastest_growing_region_id=raw.get("fastest_growing_region_id"),
            truncated=raw.get("truncated", False),
            regions=regions,
            bridges=bridges,
            region_members=raw.get("region_members", {}),
        )
