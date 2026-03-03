"""In-memory activation store for lite mode."""

from __future__ import annotations

import asyncio

from engram.config import ActivationConfig
from engram.models.activation import ActivationState


class MemoryActivationStore:
    """Dict-backed activation state. Acceptable for personal-scale graphs."""

    def __init__(self, cfg: ActivationConfig | None = None) -> None:
        self._states: dict[str, ActivationState] = {}
        self._group_map: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._cfg = cfg or ActivationConfig()

    async def get_activation(self, entity_id: str) -> ActivationState | None:
        return self._states.get(entity_id)

    async def set_activation(self, entity_id: str, state: ActivationState) -> None:
        self._states[entity_id] = state

    async def batch_get(self, entity_ids: list[str]) -> dict[str, ActivationState]:
        return {eid: self._states[eid] for eid in entity_ids if eid in self._states}

    async def batch_set(self, states: dict[str, ActivationState]) -> None:
        self._states.update(states)

    async def record_access(
        self, entity_id: str, timestamp: float, group_id: str | None = None,
    ) -> None:
        """Record an access event for an entity, creating state if needed."""
        from engram.activation.engine import record_access as _record_access

        state = self._states.get(entity_id)
        if state is None:
            state = ActivationState(node_id=entity_id)
            self._states[entity_id] = state
        _record_access(state, timestamp, self._cfg)
        if group_id:
            self._group_map[entity_id] = group_id

    async def clear_activation(self, entity_id: str) -> None:
        """Remove all activation state for an entity."""
        self._states.pop(entity_id, None)
        self._group_map.pop(entity_id, None)

    async def get_top_activated(
        self, group_id: str | None = None, limit: int = 20,
        now: float | None = None,
    ) -> list[tuple[str, ActivationState]]:
        import time

        from engram.activation.engine import compute_activation

        now = now if now is not None else time.time()
        scored = []
        for eid, state in self._states.items():
            if group_id and self._group_map.get(eid) != group_id:
                continue
            act = compute_activation(
                state.access_history, now, self._cfg,
                state.consolidated_strength,
            )
            scored.append((eid, state, act))
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(eid, state) for eid, state, _ in scored[:limit]]

    async def snapshot_to_graph(self, graph_store) -> None:
        """Persist current activation state to SQLite entity rows."""
        for eid, state in self._states.items():
            await graph_store.update_entity(
                eid,
                {
                    "access_count": state.access_count,
                    "last_accessed": state.last_accessed if state.last_accessed else None,
                },
            )
