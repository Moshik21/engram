"""Retrieval priming updates for follow-up recall queries."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from engram.config import ActivationConfig
from engram.storage.protocols import GraphStore

PrimingBuffer = dict[str, tuple[float, float]]
TimeFn = Callable[[], float]


class RecallPrimingUpdater:
    """Apply short-lived one-hop priming boosts from recalled entity results."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        cfg: ActivationConfig,
        time_fn: TimeFn = time.time,
    ) -> None:
        self._graph = graph_store
        self._cfg = cfg
        self._time_fn = time_fn

    async def update(
        self,
        results: list[dict[str, Any]],
        *,
        group_id: str,
        priming_buffer: PrimingBuffer,
    ) -> None:
        if not self._cfg.retrieval_priming_enabled or not results:
            return

        expiry = self._time_fn() + self._cfg.retrieval_priming_ttl_seconds
        for result in results[: self._cfg.retrieval_priming_top_n]:
            if result.get("result_type") != "entity":
                continue
            entity_id = self._entity_id(result)
            if entity_id is None:
                continue

            try:
                neighbors = await self._graph.get_active_neighbors_with_weights(
                    entity_id,
                    group_id,
                )
            except Exception:
                continue

            for neighbor_info in neighbors[: self._cfg.retrieval_priming_max_neighbors]:
                neighbor_id = neighbor_info[0]
                weight = neighbor_info[1]
                priming_buffer[neighbor_id] = (
                    self._cfg.retrieval_priming_boost * weight,
                    expiry,
                )

    @staticmethod
    def _entity_id(result: dict[str, Any]) -> str | None:
        entity_payload = result.get("entity")
        entity_id = entity_payload.get("id") if isinstance(entity_payload, dict) else None
        return entity_id if isinstance(entity_id, str) and entity_id else None
