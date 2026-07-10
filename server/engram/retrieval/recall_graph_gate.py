"""Suppress secondary graph reads after recall preflight probe timeouts."""

from __future__ import annotations

from typing import Any

from engram.config import ActivationConfig

GATED_GRAPH_METHODS: tuple[str, ...] = (
    "get_entity",
    "find_entities",
    "find_entity_candidates",
    "get_relationships",
    "get_episode_by_id",
    "get_active_neighbors_with_weights",
    "get_identity_core_entities",
)


def graph_probe_timed_out(stage_timings_ms: dict[str, float] | None) -> bool:
    return bool(
        stage_timings_ms
        and (
            "recall_stats_timeout" in stage_timings_ms or "graph_expand_timeout" in stage_timings_ms
        )
    )


def skip_secondary_graph_after_probe_timeout(
    cfg: ActivationConfig,
    stage_timings_ms: dict[str, float] | None,
) -> bool:
    return bool(
        cfg.retrieval_skip_secondary_graph_after_probe_timeout
        and graph_probe_timed_out(stage_timings_ms)
    )


class GatedGraphStore:
    """Proxy that blocks secondary graph reads after probe timeout."""

    def __init__(
        self,
        graph_store: Any,
        cfg: ActivationConfig,
        stage_timings_ms: dict[str, float] | None,
    ) -> None:
        self._graph_store = graph_store
        self._cfg = cfg
        self._stage_timings_ms = stage_timings_ms

    @property
    def underlying(self) -> Any:
        return self._graph_store

    def _blocked(self) -> bool:
        return skip_secondary_graph_after_probe_timeout(
            self._cfg,
            self._stage_timings_ms,
        )

    async def get_entity(self, *args: Any, **kwargs: Any) -> Any:
        if self._blocked():
            return None
        return await self._graph_store.get_entity(*args, **kwargs)

    async def find_entities(self, *args: Any, **kwargs: Any) -> list[Any]:
        if self._blocked():
            return []
        return await self._graph_store.find_entities(*args, **kwargs)

    async def find_entity_candidates(self, *args: Any, **kwargs: Any) -> list[Any]:
        if self._blocked():
            return []
        return await self._graph_store.find_entity_candidates(*args, **kwargs)

    async def get_relationships(self, *args: Any, **kwargs: Any) -> list[Any]:
        if self._blocked():
            return []
        return await self._graph_store.get_relationships(*args, **kwargs)

    async def get_episode_by_id(self, *args: Any, **kwargs: Any) -> Any:
        if self._blocked():
            return None
        return await self._graph_store.get_episode_by_id(*args, **kwargs)

    async def get_active_neighbors_with_weights(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        if self._blocked():
            return []
        return await self._graph_store.get_active_neighbors_with_weights(
            *args,
            **kwargs,
        )

    async def get_identity_core_entities(self, *args: Any, **kwargs: Any) -> list[Any]:
        if self._blocked():
            return []
        return await self._graph_store.get_identity_core_entities(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph_store, name)
