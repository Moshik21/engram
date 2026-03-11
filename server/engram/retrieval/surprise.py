"""Surprise connection detection for Wave 3 proactive intelligence."""

from __future__ import annotations

import time
from dataclasses import dataclass

from engram.activation.engine import compute_activation
from engram.config import ActivationConfig


@dataclass
class SurpriseConnection:
    """A dormant but strongly-connected entity that may be relevant."""

    entity_id: str
    entity_name: str
    connected_to_id: str
    connected_to_name: str
    predicate: str
    edge_weight: float
    activation_score: float
    surprise_score: float  # edge_weight * (1 - activation)


async def detect_surprises(
    entity_ids: list[str],
    graph_store,
    activation_store,
    cfg: ActivationConfig,
    group_id: str,
    now: float,
) -> list[SurpriseConnection]:
    """Find dormant but strongly-connected neighbors of the given entities.

    A surprise is a 1-hop neighbor with:
    - Low activation (below surprise_activation_floor)
    - Strong edge weight (above surprise_edge_weight_min)
    - Dormant (last accessed > surprise_dormancy_days ago)

    Score: edge_weight * (1.0 - activation)
    """
    dormancy_cutoff = now - (cfg.surprise_dormancy_days * 86400.0)
    surprises: list[SurpriseConnection] = []
    seen_pairs: set[tuple[str, str]] = set()

    for entity_id in entity_ids:
        try:
            neighbors = await graph_store.get_active_neighbors_with_weights(
                entity_id,
                group_id,
            )
        except Exception:
            continue

        # Get entity name for context
        try:
            source_entity = await graph_store.get_entity(entity_id, group_id)
            source_name = source_entity.name if source_entity else entity_id
        except Exception:
            source_name = entity_id

        for neighbor_info in neighbors:
            nid = neighbor_info[0]
            weight = neighbor_info[1]
            predicate = neighbor_info[2] if len(neighbor_info) > 2 else "RELATED_TO"

            # Skip weak edges
            if weight < cfg.surprise_edge_weight_min:
                continue

            # Skip already-seen pairs (in either direction)
            pair = tuple(sorted((entity_id, nid)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Get activation state
            state = await activation_store.get_activation(nid)
            if state and state.access_history:
                activation = compute_activation(state.access_history, now, cfg)
                last_accessed = max(state.access_history)
            else:
                activation = 0.0
                last_accessed = 0.0

            # Filter: must be dormant and low-activation
            if activation >= cfg.surprise_activation_floor:
                continue
            if last_accessed > dormancy_cutoff and last_accessed > 0:
                continue

            # Get neighbor name
            try:
                neighbor_entity = await graph_store.get_entity(nid, group_id)
                neighbor_name = neighbor_entity.name if neighbor_entity else nid
            except Exception:
                neighbor_name = nid

            surprise_score = weight * (1.0 - activation)
            surprises.append(
                SurpriseConnection(
                    entity_id=nid,
                    entity_name=neighbor_name,
                    connected_to_id=entity_id,
                    connected_to_name=source_name,
                    predicate=predicate,
                    edge_weight=weight,
                    activation_score=activation,
                    surprise_score=surprise_score,
                )
            )

    surprises.sort(key=lambda s: s.surprise_score, reverse=True)
    return surprises


class SurpriseCache:
    """TTL-based cache for surprise connections per group."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[float, list[SurpriseConnection]]] = {}

    def put(
        self,
        group_id: str,
        surprises: list[SurpriseConnection],
        now: float | None = None,
    ) -> None:
        ts = now if now is not None else time.time()
        self._entries[group_id] = (ts, list(surprises))

    def get(self, group_id: str, now: float | None = None) -> list[SurpriseConnection]:
        entry = self._entries.get(group_id)
        if entry is None:
            return []
        ts, surprises = entry
        ts_now = now if now is not None else time.time()
        if ts_now - ts > self._ttl:
            del self._entries[group_id]
            return []
        return surprises

    def clear(self) -> None:
        self._entries.clear()
