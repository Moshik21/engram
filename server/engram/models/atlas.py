"""Data models for persisted atlas snapshots and region drill-down."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from engram.utils.dates import utc_now_iso


@dataclass
class AtlasBridge:
    """Bridge edge between two atlas regions."""

    source: str
    target: str
    weight: float
    relationship_count: int
    id: str = field(default_factory=lambda: f"abr_{uuid.uuid4().hex[:12]}")


@dataclass
class AtlasRegion:
    """A stable, abstracted region in the user's brain atlas."""

    id: str
    label: str
    subtitle: str | None
    kind: str
    member_count: int
    represented_edge_count: int
    activation_score: float
    growth_7d: int
    growth_30d: int
    dominant_entity_types: dict[str, int] = field(default_factory=dict)
    hub_entity_ids: list[str] = field(default_factory=list)
    center_entity_id: str | None = None
    latest_entity_created_at: str | None = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class AtlasSnapshot:
    """Materialized atlas summary for one group."""

    group_id: str
    represented_entity_count: int
    represented_edge_count: int
    displayed_node_count: int
    displayed_edge_count: int
    total_entities: int
    total_relationships: int
    total_regions: int
    hottest_region_id: str | None = None
    fastest_growing_region_id: str | None = None
    truncated: bool = False
    generated_at: str = field(default_factory=utc_now_iso)
    id: str = field(default_factory=lambda: f"atlas_{uuid.uuid4().hex[:12]}")
    regions: list[AtlasRegion] = field(default_factory=list)
    bridges: list[AtlasBridge] = field(default_factory=list)
    region_members: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class AtlasSnapshotSummary:
    """Lightweight atlas snapshot metadata for history scrubbing."""

    id: str
    group_id: str
    generated_at: str
    represented_entity_count: int
    represented_edge_count: int
    displayed_node_count: int
    displayed_edge_count: int
    total_entities: int
    total_relationships: int
    total_regions: int
    hottest_region_id: str | None = None
    fastest_growing_region_id: str | None = None
    truncated: bool = False
