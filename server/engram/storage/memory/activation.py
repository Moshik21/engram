"""In-memory activation store for lite mode."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from engram.config import ActivationConfig
from engram.models.activation import ActivationState

logger = logging.getLogger(__name__)


def activation_snapshot_path() -> Path:
    """Where the shell persists the ACT-R activation snapshot at shutdown.

    The shell (engram serve) owns writes to this file; other runtimes
    (brain, one-shot CLI) load it read-only so a stale save can never
    clobber a newer shell save.
    """
    home = Path(os.environ.get("ENGRAM_HOME", Path.home() / ".engram")).expanduser()
    return home / "activation-snapshot.json"


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
        self,
        entity_id: str,
        timestamp: float,
        group_id: str | None = None,
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
        self,
        group_id: str | None = None,
        limit: int = 20,
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
                state.access_history,
                now,
                self._cfg,
                state.consolidated_strength,
            )
            scored.append((eid, state, act))
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(eid, state) for eid, state, _ in scored[:limit]]

    def save_to_file(self, path: Path) -> int:
        """Persist activation states (incl. access_history) across restarts.

        ACT-R activation is computed from access_history, which lives only in
        this dict — without persistence every shell restart (12+/day under
        the 2h brain cadence) silently wiped all recency/frequency signal.
        """
        states = {}
        for eid, state in list(self._states.items())[:50000]:
            entry = asdict(state)
            entry["group_id"] = self._group_map.get(eid)
            states[eid] = entry
        payload = {"saved_at": time.time(), "states": states}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            # silent-ok: best-effort shutdown snapshot; failure is logged with a
            # traceback and returning 0 avoids aborting the rest of shutdown cleanup.
            logger.warning("Activation snapshot write failed: %s", path, exc_info=True)
            return 0
        return len(states)

    def load_from_file(self, path: Path, max_age_days: float = 14.0) -> int:
        """Restore a prior snapshot; stale snapshots are ignored."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # silent-ok: no readable snapshot to restore; start empty (next save rewrites it).
            return 0
        saved_at = payload.get("saved_at")
        if not isinstance(saved_at, (int, float)):
            return 0
        if (time.time() - float(saved_at)) > max_age_days * 86400.0:
            return 0
        loaded = 0
        for eid, entry in (payload.get("states") or {}).items():
            if eid in self._states:
                continue  # live state wins over the snapshot
            group_id = entry.pop("group_id", None)
            try:
                state = ActivationState(
                    node_id=entry.get("node_id") or eid,
                    access_history=[float(t) for t in entry.get("access_history") or []],
                    spreading_bonus=float(entry.get("spreading_bonus") or 0.0),
                    last_accessed=float(entry.get("last_accessed") or 0.0),
                    access_count=int(entry.get("access_count") or 0),
                    consolidated_strength=float(entry.get("consolidated_strength") or 0.0),
                    last_compacted=float(entry.get("last_compacted") or 0.0),
                    ts_alpha=float(entry.get("ts_alpha") or 1.0),
                    ts_beta=float(entry.get("ts_beta") or 1.0),
                )
            except (TypeError, ValueError):
                # silent-ok: skip a single malformed snapshot entry; others still restore.
                continue
            self._states[eid] = state
            if group_id:
                self._group_map[eid] = group_id
            loaded += 1
        return loaded

    async def snapshot_to_graph(self, graph_store) -> None:
        """Persist current activation state to graph entity rows."""
        for eid, state in self._states.items():
            group_id = self._group_map.get(eid, "default")
            last_accessed = (
                datetime.fromtimestamp(state.last_accessed, tz=timezone.utc).replace(tzinfo=None)
                if state.last_accessed
                else None
            )
            await graph_store.update_entity(
                eid,
                {
                    "access_count": state.access_count,
                    "last_accessed": last_accessed,
                },
                group_id=group_id,
            )
