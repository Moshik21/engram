"""HelixDB-backed atlas snapshot store."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from engram.config import HelixDBConfig
from engram.models.atlas import (
    AtlasBridge,
    AtlasRegion,
    AtlasSnapshot,
    AtlasSnapshotSummary,
)

logger = logging.getLogger(__name__)


def _safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict returned by Helix."""
    v = d.get(key, default)
    return v if v is not None else default


class HelixAtlasStore:
    """Stores materialized atlas snapshots in HelixDB."""

    def __init__(self, config: HelixDBConfig, client=None) -> None:
        self._config = config
        self._client: Any | None = None
        self._helix_client = client  # Shared HelixClient (async httpx)
        # logical snapshot id -> Helix internal node ID
        self._snapshot_id_cache: dict[str, int] = {}
        # logical region key -> Helix internal node ID
        self._region_id_cache: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Query helper
    # ------------------------------------------------------------------

    async def _query(self, endpoint: str, payload: dict) -> list[dict]:
        """Execute a Helix query.

        Fast path: shared async HelixClient (httpx connection pool).
        Legacy fallback: synchronous helix-py SDK via thread pool.
        """
        # Fast path: shared async client
        if self._helix_client is not None:
            return await self._helix_client.query(endpoint, payload)

        # Legacy fallback: synchronous helix-py SDK
        client = self._client
        if client is None:
            raise RuntimeError("HelixAtlasStore not initialized")
        try:
            result = await asyncio.to_thread(client.query, endpoint, payload)
            if result is None:
                return []
            from engram.storage.helix import unwrap_helix_results

            return unwrap_helix_results(result)
        except Exception as exc:
            exc_name = type(exc).__name__
            if "NoValue" in exc_name or "NotFound" in exc_name:
                return []
            raise

    @staticmethod
    def _extract_helix_id(item: dict):
        """Extract the Helix-assigned internal ID from a response dict."""
        for key in ("id", "_id", "node_id", "edge_id"):
            if key in item and item[key] is not None:
                return item[key]
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, db: Any = None) -> None:
        """Connect to HelixDB."""
        if self._helix_client is None:
            from engram.storage.helix.client import HelixClient

            self._helix_client = HelixClient(self._config)
        if not self._helix_client.is_connected:
            await self._helix_client.initialize()

        transport = getattr(self._config, "transport", "http")
        if transport == "native":
            logger.info("HelixDB atlas store initialized (native transport)")
            return

        from helix import Client  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {
            "port": self._config.port,
            "verbose": self._config.verbose,
        }
        if self._config.api_endpoint:
            kwargs["url"] = self._config.api_endpoint
            kwargs["local"] = False
            if self._config.api_key:
                kwargs["api_key"] = self._config.api_key
        else:
            kwargs["local"] = True

        self._client = await asyncio.to_thread(Client, **kwargs)
        logger.info(
            "HelixDB atlas store initialized (host=%s, port=%d)",
            self._config.host,
            self._config.port,
        )

    async def close(self) -> None:
        """No-op -- Helix sync client has no close method."""
        self._client = None

    # ------------------------------------------------------------------
    # save_snapshot
    # ------------------------------------------------------------------

    async def save_snapshot(self, snapshot: AtlasSnapshot) -> None:
        """Persist a full atlas snapshot (upsert semantics).

        Steps:
        1. Delete old regions, edges, members for this snapshot ID.
        2. Delete the old snapshot node if it exists.
        3. Create the snapshot node.
        4. Create region nodes.
        5. Create bridge (region edge) nodes.
        6. Create region member nodes.
        """
        # -- 1. Delete old child data for this snapshot --
        try:
            await self._query(
                "delete_atlas_region_members_by_snapshot",
                {"snap_id": snapshot.id, "gid": snapshot.group_id},
            )
        except Exception:
            pass

        try:
            await self._query(
                "delete_atlas_region_edges_by_snapshot",
                {"snap_id": snapshot.id, "gid": snapshot.group_id},
            )
        except Exception:
            pass

        try:
            await self._query(
                "delete_atlas_regions_by_snapshot",
                {"snap_id": snapshot.id, "gid": snapshot.group_id},
            )
        except Exception:
            pass

        # -- 2. Delete old snapshot node --
        helix_id = self._snapshot_id_cache.pop(snapshot.id, None)
        if helix_id is not None:
            try:
                await self._query("delete_atlas_snapshot", {"id": helix_id})
            except Exception:
                pass
        else:
            # Try to find existing snapshot by query and delete
            # Try the group lookup to find and delete existing
            try:
                group_snaps = await self._query(
                    "find_atlas_snapshots_by_group",
                    {"gid": snapshot.group_id},
                )
                for s in group_snaps:
                    if _safe_get(s, "snapshot_id") == snapshot.id:
                        hid = self._extract_helix_id(s)
                        if hid is not None:
                            try:
                                await self._query("delete_atlas_snapshot", {"id": hid})
                            except Exception:
                                pass
                        break
            except Exception:
                pass

        # -- 3. Create snapshot node --
        results = await self._query(
            "create_atlas_snapshot",
            {
                "snapshot_id": snapshot.id,
                "group_id": snapshot.group_id,
                "generated_at": snapshot.generated_at,
                "represented_entity_count": snapshot.represented_entity_count,
                "represented_edge_count": snapshot.represented_edge_count,
                "displayed_node_count": snapshot.displayed_node_count,
                "displayed_edge_count": snapshot.displayed_edge_count,
                "total_entities": snapshot.total_entities,
                "total_relationships": snapshot.total_relationships,
                "total_regions": snapshot.total_regions,
                "hottest_region_id": snapshot.hottest_region_id or "",
                "fastest_growing_region_id": snapshot.fastest_growing_region_id or "",
                "truncated": snapshot.truncated,
            },
        )
        if results:
            hid = self._extract_helix_id(results[0])
            if hid is not None:
                self._snapshot_id_cache[snapshot.id] = hid

        # -- 4. Create region nodes --
        for region in snapshot.regions:
            region_results = await self._query(
                "create_atlas_region",
                {
                    "snapshot_id": snapshot.id,
                    "group_id": snapshot.group_id,
                    "region_id": region.id,
                    "label": region.label,
                    "subtitle": region.subtitle or "",
                    "kind": region.kind,
                    "member_count": region.member_count,
                    "represented_edge_count": region.represented_edge_count,
                    "activation_score": region.activation_score,
                    "growth_7d": region.growth_7d,
                    "growth_30d": region.growth_30d,
                    "dominant_entity_types_json": json.dumps(
                        region.dominant_entity_types
                    ),
                    "hub_entity_ids_json": json.dumps(region.hub_entity_ids),
                    "center_entity_id": region.center_entity_id or "",
                    "latest_entity_created_at": region.latest_entity_created_at or "",
                    "x": region.x,
                    "y": region.y,
                    "z": region.z,
                },
            )
            if region_results:
                hid = self._extract_helix_id(region_results[0])
                cache_key = f"{snapshot.id}:{region.id}"
                if hid is not None:
                    self._region_id_cache[cache_key] = hid

        # -- 5. Create bridge (region edge) nodes --
        for bridge in snapshot.bridges:
            await self._query(
                "create_atlas_region_edge",
                {
                    "snapshot_id": snapshot.id,
                    "group_id": snapshot.group_id,
                    "edge_id": bridge.id,
                    "source_region_id": bridge.source,
                    "target_region_id": bridge.target,
                    "weight": bridge.weight,
                    "relationship_count": bridge.relationship_count,
                },
            )

        # -- 6. Create region member nodes --
        for region_id, entity_ids in snapshot.region_members.items():
            for entity_id in entity_ids:
                await self._query(
                    "create_atlas_region_member",
                    {
                        "snapshot_id": snapshot.id,
                        "group_id": snapshot.group_id,
                        "region_id": region_id,
                        "entity_id": entity_id,
                    },
                )

    # ------------------------------------------------------------------
    # get_latest_snapshot
    # ------------------------------------------------------------------

    async def get_latest_snapshot(self, group_id: str) -> AtlasSnapshot | None:
        """Return the most recent snapshot for a group, or None."""
        results = await self._query(
            "find_atlas_snapshots_by_group",
            {"gid": group_id},
        )
        if not results:
            return None
        # Already ordered by generated_at DESC; take the first
        first = results[0]
        snapshot_id = _safe_get(first, "snapshot_id", "")
        if not snapshot_id:
            return None
        return await self._load_snapshot(snapshot_id, group_id)

    # ------------------------------------------------------------------
    # get_snapshot
    # ------------------------------------------------------------------

    async def get_snapshot(
        self,
        snapshot_id: str,
        group_id: str,
    ) -> AtlasSnapshot | None:
        """Load a specific snapshot by ID."""
        return await self._load_snapshot(snapshot_id, group_id)

    # ------------------------------------------------------------------
    # list_snapshots
    # ------------------------------------------------------------------

    async def list_snapshots(
        self,
        group_id: str,
        limit: int = 24,
    ) -> list[AtlasSnapshotSummary]:
        """Return lightweight summaries for recent snapshots."""
        results = await self._query(
            "find_atlas_snapshots_by_group",
            {"gid": group_id},
        )
        summaries: list[AtlasSnapshotSummary] = []
        for d in results[:limit]:
            truncated_raw = _safe_get(d, "truncated", False)
            if isinstance(truncated_raw, int):
                truncated_val = bool(truncated_raw)
            else:
                truncated_val = bool(truncated_raw)
            summaries.append(
                AtlasSnapshotSummary(
                    id=_safe_get(d, "snapshot_id", ""),
                    group_id=_safe_get(d, "group_id", group_id),
                    generated_at=_safe_get(d, "generated_at", ""),
                    represented_entity_count=int(
                        _safe_get(d, "represented_entity_count", 0)
                    ),
                    represented_edge_count=int(
                        _safe_get(d, "represented_edge_count", 0)
                    ),
                    displayed_node_count=int(
                        _safe_get(d, "displayed_node_count", 0)
                    ),
                    displayed_edge_count=int(
                        _safe_get(d, "displayed_edge_count", 0)
                    ),
                    total_entities=int(_safe_get(d, "total_entities", 0)),
                    total_relationships=int(
                        _safe_get(d, "total_relationships", 0)
                    ),
                    total_regions=int(_safe_get(d, "total_regions", 0)),
                    hottest_region_id=_safe_get(d, "hottest_region_id") or None,
                    fastest_growing_region_id=_safe_get(
                        d, "fastest_growing_region_id"
                    )
                    or None,
                    truncated=truncated_val,
                )
            )
        return summaries

    # ------------------------------------------------------------------
    # get_region_members
    # ------------------------------------------------------------------

    async def get_region_members(
        self,
        snapshot_id: str,
        region_id: str,
        group_id: str,
    ) -> list[str]:
        """Return entity IDs belonging to a region in a snapshot."""
        results = await self._query(
            "find_atlas_region_members",
            {
                "snap_id": snapshot_id,
                "region_id": region_id,
                "gid": group_id,
            },
        )
        entity_ids: list[str] = []
        for d in results:
            eid = _safe_get(d, "entity_id", "")
            if eid:
                entity_ids.append(eid)
        entity_ids.sort()
        return entity_ids

    # ------------------------------------------------------------------
    # _load_snapshot (private helper)
    # ------------------------------------------------------------------

    async def _load_snapshot(
        self,
        snapshot_id: str,
        group_id: str,
    ) -> AtlasSnapshot | None:
        """Fetch snapshot + regions + bridges + members and assemble the full model."""
        # -- Find the snapshot record --
        all_snaps = await self._query(
            "find_atlas_snapshots_by_group",
            {"gid": group_id},
        )
        snapshot_dict: dict | None = None
        for s in all_snaps:
            if _safe_get(s, "snapshot_id") == snapshot_id:
                snapshot_dict = s
                # Cache the helix ID
                hid = self._extract_helix_id(s)
                if hid is not None:
                    self._snapshot_id_cache[snapshot_id] = hid
                break

        if snapshot_dict is None:
            return None

        # -- Load regions --
        region_results = await self._query(
            "find_atlas_regions",
            {"snap_id": snapshot_id, "gid": group_id},
        )
        regions: list[AtlasRegion] = []
        for d in region_results:
            dominant_raw = _safe_get(d, "dominant_entity_types_json", "{}")
            hub_raw = _safe_get(d, "hub_entity_ids_json", "[]")
            try:
                dominant = json.loads(dominant_raw) if dominant_raw else {}
            except (json.JSONDecodeError, TypeError):
                dominant = {}
            try:
                hub_ids = json.loads(hub_raw) if hub_raw else []
            except (json.JSONDecodeError, TypeError):
                hub_ids = []

            subtitle = _safe_get(d, "subtitle", "")
            center_entity_id = _safe_get(d, "center_entity_id", "")
            latest_created = _safe_get(d, "latest_entity_created_at", "")

            regions.append(
                AtlasRegion(
                    id=_safe_get(d, "region_id", ""),
                    label=_safe_get(d, "label", ""),
                    subtitle=subtitle if subtitle else None,
                    kind=_safe_get(d, "kind", ""),
                    member_count=int(_safe_get(d, "member_count", 0)),
                    represented_edge_count=int(
                        _safe_get(d, "represented_edge_count", 0)
                    ),
                    activation_score=float(_safe_get(d, "activation_score", 0.0)),
                    growth_7d=int(_safe_get(d, "growth_7d", 0)),
                    growth_30d=int(_safe_get(d, "growth_30d", 0)),
                    dominant_entity_types=dominant,
                    hub_entity_ids=hub_ids,
                    center_entity_id=center_entity_id if center_entity_id else None,
                    latest_entity_created_at=latest_created
                    if latest_created
                    else None,
                    x=float(_safe_get(d, "x", 0.0)),
                    y=float(_safe_get(d, "y", 0.0)),
                    z=float(_safe_get(d, "z", 0.0)),
                )
            )
        # Sort: largest regions first, then by region_id for stability
        regions.sort(key=lambda r: (-r.member_count, r.id))

        # -- Load bridges (region edges) --
        bridge_results = await self._query(
            "find_atlas_region_edges",
            {"snap_id": snapshot_id, "gid": group_id},
        )
        bridges: list[AtlasBridge] = []
        for d in bridge_results:
            bridges.append(
                AtlasBridge(
                    id=_safe_get(d, "edge_id", ""),
                    source=_safe_get(d, "source_region_id", ""),
                    target=_safe_get(d, "target_region_id", ""),
                    weight=float(_safe_get(d, "weight", 0.0)),
                    relationship_count=int(_safe_get(d, "relationship_count", 0)),
                )
            )
        # Sort: heaviest bridges first, then by edge_id for stability
        bridges.sort(key=lambda b: (-b.weight, b.id))

        # -- Load region members --
        region_members: dict[str, list[str]] = {}
        # Collect region IDs from loaded regions
        for region in regions:
            members = await self._query(
                "find_atlas_region_members",
                {
                    "snap_id": snapshot_id,
                    "region_id": region.id,
                    "gid": group_id,
                },
            )
            entity_ids = []
            for m in members:
                eid = _safe_get(m, "entity_id", "")
                if eid:
                    entity_ids.append(eid)
            if entity_ids:
                entity_ids.sort()
                region_members[region.id] = entity_ids

        # -- Assemble the full snapshot --
        truncated_raw = _safe_get(snapshot_dict, "truncated", False)
        if isinstance(truncated_raw, int):
            truncated_val = bool(truncated_raw)
        else:
            truncated_val = bool(truncated_raw)

        hottest = _safe_get(snapshot_dict, "hottest_region_id", "")
        fastest = _safe_get(snapshot_dict, "fastest_growing_region_id", "")

        return AtlasSnapshot(
            id=_safe_get(snapshot_dict, "snapshot_id", ""),
            group_id=_safe_get(snapshot_dict, "group_id", group_id),
            generated_at=_safe_get(snapshot_dict, "generated_at", ""),
            represented_entity_count=int(
                _safe_get(snapshot_dict, "represented_entity_count", 0)
            ),
            represented_edge_count=int(
                _safe_get(snapshot_dict, "represented_edge_count", 0)
            ),
            displayed_node_count=int(
                _safe_get(snapshot_dict, "displayed_node_count", 0)
            ),
            displayed_edge_count=int(
                _safe_get(snapshot_dict, "displayed_edge_count", 0)
            ),
            total_entities=int(_safe_get(snapshot_dict, "total_entities", 0)),
            total_relationships=int(
                _safe_get(snapshot_dict, "total_relationships", 0)
            ),
            total_regions=int(_safe_get(snapshot_dict, "total_regions", 0)),
            hottest_region_id=hottest if hottest else None,
            fastest_growing_region_id=fastest if fastest else None,
            truncated=truncated_val,
            regions=regions,
            bridges=bridges,
            region_members=region_members,
        )
