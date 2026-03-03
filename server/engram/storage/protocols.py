"""Protocol ABCs for storage backends. Both lite and full mode implement these."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship

ENTITY_UPDATABLE_FIELDS = frozenset({
    "name", "summary", "updated_at", "pii_detected", "pii_categories",
    "access_count", "last_accessed", "deleted_at",
})

EPISODE_UPDATABLE_FIELDS = frozenset({
    "status", "updated_at", "error", "retry_count", "processing_duration_ms",
})


@runtime_checkable
class GraphStore(Protocol):
    """Persistent storage for entities, relationships, and episodes."""

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    async def create_entity(self, entity: Entity) -> str: ...
    async def get_entity(self, entity_id: str, group_id: str) -> Entity | None: ...
    async def update_entity(self, entity_id: str, updates: dict, group_id: str) -> None: ...
    async def delete_entity(self, entity_id: str, soft: bool = True, *, group_id: str) -> None: ...
    async def find_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[Entity]: ...
    async def create_relationship(self, rel: Relationship) -> str: ...
    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        predicate: str | None = None,
        active_only: bool = True,
        group_id: str = "default",
    ) -> list[Relationship]: ...
    async def invalidate_relationship(
        self, rel_id: str, valid_to: datetime, group_id: str
    ) -> None: ...
    async def find_conflicting_relationships(
        self,
        source_id: str,
        predicate: str,
        group_id: str,
    ) -> list[Relationship]: ...
    async def get_relationships_at(
        self,
        entity_id: str,
        at_time: datetime,
        direction: str = "both",
        group_id: str = "default",
    ) -> list[Relationship]: ...
    async def get_neighbors(
        self, entity_id: str, hops: int = 1, group_id: str | None = None
    ) -> list[tuple[Entity, Relationship]]: ...
    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None
    ) -> list[tuple[str, float, str]]: ...  # (neighbor_id, weight, predicate)
    async def create_episode(self, episode: Episode) -> str: ...
    async def update_episode(
        self, episode_id: str, updates: dict, group_id: str = "default",
    ) -> None: ...
    async def get_episodes(
        self,
        group_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Episode]: ...
    async def get_episode_by_id(self, episode_id: str, group_id: str) -> Episode | None: ...
    async def get_episode_entities(self, episode_id: str) -> list[str]: ...
    async def link_episode_entity(self, episode_id: str, entity_id: str) -> None: ...
    async def get_stats(self, group_id: str | None = None) -> dict: ...
    async def get_episodes_paginated(
        self, group_id: str | None = None, cursor: str | None = None,
        limit: int = 50, source: str | None = None, status: str | None = None,
    ) -> tuple[list[Episode], str | None]: ...
    async def get_top_connected(
        self, group_id: str | None = None, limit: int = 10,
    ) -> list[dict]: ...
    async def get_growth_timeline(
        self, group_id: str | None = None, days: int = 30,
    ) -> list[dict]: ...
    async def get_entity_type_counts(self, group_id: str | None = None) -> dict[str, int]: ...

    # --- Consolidation methods ---
    async def get_co_occurring_entity_pairs(
        self, group_id: str, since: datetime | None = None,
        min_co_occurrence: int = 3, limit: int = 100,
    ) -> list[tuple[str, str, int]]: ...

    async def get_entity_episode_counts(
        self, group_id: str, entity_ids: list[str],
    ) -> dict[str, int]: ...

    async def get_dead_entities(
        self, group_id: str, min_age_days: int = 30, limit: int = 100,
    ) -> list[Entity]: ...

    async def merge_entities(
        self, keep_id: str, remove_id: str, group_id: str,
    ) -> int: ...

    async def get_relationships_by_predicate(
        self, group_id: str, predicate: str,
        active_only: bool = True, limit: int = 10000,
    ) -> list[Relationship]: ...

    async def update_relationship_weight(
        self, source_id: str, target_id: str, weight_delta: float,
        max_weight: float = 3.0, group_id: str = "default",
    ) -> float | None: ...


@runtime_checkable
class ActivationStore(Protocol):
    """Hot-path activation state storage."""

    async def get_activation(self, entity_id: str) -> ActivationState | None: ...
    async def set_activation(self, entity_id: str, state: ActivationState) -> None: ...
    async def batch_get(self, entity_ids: list[str]) -> dict[str, ActivationState]: ...
    async def batch_set(self, states: dict[str, ActivationState]) -> None: ...
    async def record_access(
        self, entity_id: str, timestamp: float, group_id: str | None = None,
    ) -> None: ...
    async def clear_activation(self, entity_id: str) -> None: ...
    async def get_top_activated(
        self, group_id: str | None = None, limit: int = 20,
        now: float | None = None,
    ) -> list[tuple[str, ActivationState]]: ...


@runtime_checkable
class SearchIndex(Protocol):
    """Text/semantic search over entities and facts."""

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    async def index_entity(self, entity: Entity) -> None: ...
    async def index_episode(self, episode: Episode) -> None: ...
    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]: ...
    async def remove(self, entity_id: str) -> None: ...
    async def compute_similarity(
        self,
        query: str,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, float]: ...
    async def search_episodes(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]: ...
