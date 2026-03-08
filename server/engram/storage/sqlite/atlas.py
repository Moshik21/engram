"""SQLite-backed atlas snapshot store."""

from __future__ import annotations

import json

import aiosqlite

from engram.models.atlas import (
    AtlasBridge,
    AtlasRegion,
    AtlasSnapshot,
    AtlasSnapshotSummary,
)


class SQLiteAtlasStore:
    """Stores materialized atlas snapshots in SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._owns_db = False

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("AtlasStore not initialized")
        return self._db

    async def initialize(self, db: aiosqlite.Connection | None = None) -> None:
        if db:
            self._db = db
            self._owns_db = False
        elif not self._db:
            self._db = await aiosqlite.connect(self._db_path)
            self._owns_db = True
        if self._db is not None:
            self._db.row_factory = aiosqlite.Row

        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS atlas_snapshots (
                id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                represented_entity_count INTEGER NOT NULL,
                represented_edge_count INTEGER NOT NULL,
                displayed_node_count INTEGER NOT NULL,
                displayed_edge_count INTEGER NOT NULL,
                total_entities INTEGER NOT NULL,
                total_relationships INTEGER NOT NULL,
                total_regions INTEGER NOT NULL,
                hottest_region_id TEXT,
                fastest_growing_region_id TEXT,
                truncated INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_atlas_snapshots_group_generated "
            "ON atlas_snapshots(group_id, generated_at DESC)"
        )
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS atlas_regions (
                snapshot_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                region_id TEXT NOT NULL,
                label TEXT NOT NULL,
                subtitle TEXT,
                kind TEXT NOT NULL,
                member_count INTEGER NOT NULL,
                represented_edge_count INTEGER NOT NULL,
                activation_score REAL NOT NULL,
                growth_7d INTEGER NOT NULL,
                growth_30d INTEGER NOT NULL,
                dominant_entity_types_json TEXT NOT NULL,
                hub_entity_ids_json TEXT NOT NULL,
                center_entity_id TEXT,
                latest_entity_created_at TEXT,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                PRIMARY KEY (snapshot_id, region_id)
            )
            """
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_atlas_regions_snapshot "
            "ON atlas_regions(snapshot_id, group_id)"
        )
        try:
            await self.db.execute(
                "ALTER TABLE atlas_regions ADD COLUMN latest_entity_created_at TEXT"
            )
        except Exception:
            pass
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS atlas_region_edges (
                snapshot_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                edge_id TEXT NOT NULL,
                source_region_id TEXT NOT NULL,
                target_region_id TEXT NOT NULL,
                weight REAL NOT NULL,
                relationship_count INTEGER NOT NULL,
                PRIMARY KEY (snapshot_id, edge_id)
            )
            """
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_atlas_region_edges_snapshot "
            "ON atlas_region_edges(snapshot_id, group_id)"
        )
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS atlas_region_members (
                snapshot_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                region_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY (snapshot_id, region_id, entity_id)
            )
            """
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_atlas_region_members_snapshot "
            "ON atlas_region_members(snapshot_id, group_id, region_id)"
        )
        await self.db.commit()

    async def close(self) -> None:
        if self._db and self._owns_db:
            await self._db.close()
        self._db = None
        self._owns_db = False

    async def save_snapshot(self, snapshot: AtlasSnapshot) -> None:
        await self.db.execute(
            """
            INSERT OR REPLACE INTO atlas_snapshots (
                id, group_id, generated_at, represented_entity_count,
                represented_edge_count, displayed_node_count, displayed_edge_count,
                total_entities, total_relationships, total_regions,
                hottest_region_id, fastest_growing_region_id, truncated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.id,
                snapshot.group_id,
                snapshot.generated_at,
                snapshot.represented_entity_count,
                snapshot.represented_edge_count,
                snapshot.displayed_node_count,
                snapshot.displayed_edge_count,
                snapshot.total_entities,
                snapshot.total_relationships,
                snapshot.total_regions,
                snapshot.hottest_region_id,
                snapshot.fastest_growing_region_id,
                1 if snapshot.truncated else 0,
            ),
        )
        await self.db.execute(
            "DELETE FROM atlas_regions WHERE snapshot_id = ? AND group_id = ?",
            (snapshot.id, snapshot.group_id),
        )
        await self.db.execute(
            "DELETE FROM atlas_region_edges WHERE snapshot_id = ? AND group_id = ?",
            (snapshot.id, snapshot.group_id),
        )
        await self.db.execute(
            "DELETE FROM atlas_region_members WHERE snapshot_id = ? AND group_id = ?",
            (snapshot.id, snapshot.group_id),
        )

        for region in snapshot.regions:
            await self.db.execute(
                """
                INSERT INTO atlas_regions (
                    snapshot_id, group_id, region_id, label, subtitle, kind,
                    member_count, represented_edge_count, activation_score,
                    growth_7d, growth_30d, dominant_entity_types_json,
                    hub_entity_ids_json, center_entity_id, latest_entity_created_at,
                    x, y, z
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.id,
                    snapshot.group_id,
                    region.id,
                    region.label,
                    region.subtitle,
                    region.kind,
                    region.member_count,
                    region.represented_edge_count,
                    region.activation_score,
                    region.growth_7d,
                    region.growth_30d,
                    json.dumps(region.dominant_entity_types),
                    json.dumps(region.hub_entity_ids),
                    region.center_entity_id,
                    region.latest_entity_created_at,
                    region.x,
                    region.y,
                    region.z,
                ),
            )
        for bridge in snapshot.bridges:
            await self.db.execute(
                """
                INSERT INTO atlas_region_edges (
                    snapshot_id, group_id, edge_id, source_region_id, target_region_id,
                    weight, relationship_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.id,
                    snapshot.group_id,
                    bridge.id,
                    bridge.source,
                    bridge.target,
                    bridge.weight,
                    bridge.relationship_count,
                ),
            )
        for region_id, entity_ids in snapshot.region_members.items():
            for entity_id in entity_ids:
                await self.db.execute(
                    """
                    INSERT INTO atlas_region_members (
                        snapshot_id, group_id, region_id, entity_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (snapshot.id, snapshot.group_id, region_id, entity_id),
                )
        await self.db.commit()

    async def get_latest_snapshot(self, group_id: str) -> AtlasSnapshot | None:
        cursor = await self.db.execute(
            """
            SELECT *
            FROM atlas_snapshots
            WHERE group_id = ?
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (group_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return await self._load_snapshot(row["id"], group_id)

    async def get_snapshot(
        self,
        snapshot_id: str,
        group_id: str,
    ) -> AtlasSnapshot | None:
        return await self._load_snapshot(snapshot_id, group_id)

    async def list_snapshots(
        self,
        group_id: str,
        limit: int = 24,
    ) -> list[AtlasSnapshotSummary]:
        cursor = await self.db.execute(
            """
            SELECT *
            FROM atlas_snapshots
            WHERE group_id = ?
            ORDER BY generated_at DESC
            LIMIT ?
            """,
            (group_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            AtlasSnapshotSummary(
                id=row["id"],
                group_id=row["group_id"],
                generated_at=row["generated_at"],
                represented_entity_count=row["represented_entity_count"],
                represented_edge_count=row["represented_edge_count"],
                displayed_node_count=row["displayed_node_count"],
                displayed_edge_count=row["displayed_edge_count"],
                total_entities=row["total_entities"],
                total_relationships=row["total_relationships"],
                total_regions=row["total_regions"],
                hottest_region_id=row["hottest_region_id"],
                fastest_growing_region_id=row["fastest_growing_region_id"],
                truncated=bool(row["truncated"]),
            )
            for row in rows
        ]

    async def get_region_members(
        self,
        snapshot_id: str,
        region_id: str,
        group_id: str,
    ) -> list[str]:
        cursor = await self.db.execute(
            """
            SELECT entity_id
            FROM atlas_region_members
            WHERE snapshot_id = ? AND region_id = ? AND group_id = ?
            ORDER BY entity_id ASC
            """,
            (snapshot_id, region_id, group_id),
        )
        rows = await cursor.fetchall()
        return [row["entity_id"] for row in rows]

    async def _load_snapshot(self, snapshot_id: str, group_id: str) -> AtlasSnapshot | None:
        snapshot_cursor = await self.db.execute(
            """
            SELECT *
            FROM atlas_snapshots
            WHERE id = ? AND group_id = ?
            """,
            (snapshot_id, group_id),
        )
        snapshot_row = await snapshot_cursor.fetchone()
        if snapshot_row is None:
            return None

        regions_cursor = await self.db.execute(
            """
            SELECT *
            FROM atlas_regions
            WHERE snapshot_id = ? AND group_id = ?
            ORDER BY member_count DESC, region_id ASC
            """,
            (snapshot_id, group_id),
        )
        region_rows = await regions_cursor.fetchall()
        regions = [
            AtlasRegion(
                id=row["region_id"],
                label=row["label"],
                subtitle=row["subtitle"],
                kind=row["kind"],
                member_count=row["member_count"],
                represented_edge_count=row["represented_edge_count"],
                activation_score=row["activation_score"],
                growth_7d=row["growth_7d"],
                growth_30d=row["growth_30d"],
                dominant_entity_types=json.loads(row["dominant_entity_types_json"] or "{}"),
                hub_entity_ids=json.loads(row["hub_entity_ids_json"] or "[]"),
                center_entity_id=row["center_entity_id"],
                latest_entity_created_at=row["latest_entity_created_at"],
                x=row["x"],
                y=row["y"],
                z=row["z"],
            )
            for row in region_rows
        ]

        bridge_cursor = await self.db.execute(
            """
            SELECT *
            FROM atlas_region_edges
            WHERE snapshot_id = ? AND group_id = ?
            ORDER BY weight DESC, edge_id ASC
            """,
            (snapshot_id, group_id),
        )
        bridge_rows = await bridge_cursor.fetchall()
        bridges = [
            AtlasBridge(
                id=row["edge_id"],
                source=row["source_region_id"],
                target=row["target_region_id"],
                weight=row["weight"],
                relationship_count=row["relationship_count"],
            )
            for row in bridge_rows
        ]

        member_cursor = await self.db.execute(
            """
            SELECT region_id, entity_id
            FROM atlas_region_members
            WHERE snapshot_id = ? AND group_id = ?
            ORDER BY region_id ASC, entity_id ASC
            """,
            (snapshot_id, group_id),
        )
        member_rows = await member_cursor.fetchall()
        region_members: dict[str, list[str]] = {}
        for row in member_rows:
            region_members.setdefault(row["region_id"], []).append(row["entity_id"])

        return AtlasSnapshot(
            id=snapshot_row["id"],
            group_id=snapshot_row["group_id"],
            generated_at=snapshot_row["generated_at"],
            represented_entity_count=snapshot_row["represented_entity_count"],
            represented_edge_count=snapshot_row["represented_edge_count"],
            displayed_node_count=snapshot_row["displayed_node_count"],
            displayed_edge_count=snapshot_row["displayed_edge_count"],
            total_entities=snapshot_row["total_entities"],
            total_relationships=snapshot_row["total_relationships"],
            total_regions=snapshot_row["total_regions"],
            hottest_region_id=snapshot_row["hottest_region_id"],
            fastest_growing_region_id=snapshot_row["fastest_growing_region_id"],
            truncated=bool(snapshot_row["truncated"]),
            regions=regions,
            bridges=bridges,
            region_members=region_members,
        )
