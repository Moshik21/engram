"""Redis-backed activation store for full mode."""

from __future__ import annotations

import json
import logging
import time

from engram.config import ActivationConfig
from engram.models.activation import ActivationState

logger = logging.getLogger(__name__)


class RedisActivationStore:
    """Activation state stored in Redis hashes with TTL expiry.

    Key pattern: act:{entity_id} — entity IDs are globally unique (ULID-based),
    providing implicit tenant isolation. The hash stores group_id as a field
    for get_top_activated filtering.
    """

    def __init__(self, redis, cfg: ActivationConfig | None = None) -> None:
        self._redis = redis
        self._cfg = cfg or ActivationConfig()

    @property
    def _ttl_seconds(self) -> int:
        return self._cfg.activation_ttl_days * 24 * 3600

    def _key(self, entity_id: str) -> str:
        return f"act:{entity_id}"

    def _serialize(self, state: ActivationState, group_id: str = "") -> dict:
        return {
            "node_id": state.node_id,
            "group_id": group_id,
            "access_history": json.dumps(state.access_history),
            "spreading_bonus": str(state.spreading_bonus),
            "last_accessed": str(state.last_accessed),
            "access_count": str(state.access_count),
            "consolidated_strength": str(state.consolidated_strength),
            "last_compacted": str(state.last_compacted),
            "ts_alpha": str(state.ts_alpha),
            "ts_beta": str(state.ts_beta),
        }

    def _deserialize(self, data: dict) -> ActivationState | None:
        if not data:
            return None
        # Handle both bytes and string keys (decode_responses may vary)
        if isinstance(next(iter(data.keys()), ""), bytes):
            data = {k.decode(): v.decode() if isinstance(v, bytes) else v for k, v in data.items()}
        elif isinstance(next(iter(data.values()), ""), bytes):
            data = {k: v.decode() if isinstance(v, bytes) else v for k, v in data.items()}

        return ActivationState(
            node_id=data.get("node_id", ""),
            access_history=json.loads(data.get("access_history", "[]")),
            spreading_bonus=float(data.get("spreading_bonus", "0.0")),
            last_accessed=float(data.get("last_accessed", "0.0")),
            access_count=int(data.get("access_count", "0")),
            consolidated_strength=float(data.get("consolidated_strength", "0.0")),
            last_compacted=float(data.get("last_compacted", "0.0")),
            ts_alpha=float(data.get("ts_alpha", "1.0")),
            ts_beta=float(data.get("ts_beta", "1.0")),
        )

    async def get_activation(self, entity_id: str) -> ActivationState | None:
        data = await self._redis.hgetall(self._key(entity_id))
        return self._deserialize(data)

    async def set_activation(self, entity_id: str, state: ActivationState) -> None:
        key = self._key(entity_id)
        pipe = self._redis.pipeline()
        pipe.hset(key, mapping=self._serialize(state))
        pipe.expire(key, self._ttl_seconds)
        await pipe.execute()

    async def batch_get(self, entity_ids: list[str]) -> dict[str, ActivationState]:
        if not entity_ids:
            return {}
        pipe = self._redis.pipeline()
        for eid in entity_ids:
            pipe.hgetall(self._key(eid))
        results = await pipe.execute()

        out: dict[str, ActivationState] = {}
        for eid, data in zip(entity_ids, results):
            state = self._deserialize(data)
            if state:
                out[eid] = state
        return out

    async def batch_set(self, states: dict[str, ActivationState]) -> None:
        if not states:
            return
        from engram.activation.engine import compute_activation

        now = time.time()
        pipe = self._redis.pipeline()
        for eid, state in states.items():
            key = self._key(eid)
            serialized = self._serialize(state)
            group_id = serialized.get("group_id", "")
            pipe.hset(key, mapping=serialized)
            pipe.expire(key, self._ttl_seconds)
            # Update sorted set index for get_top_activated
            if group_id:
                act = compute_activation(
                    state.access_history, now, self._cfg, state.consolidated_strength,
                )
                zkey = f"act_group:{group_id}"
                pipe.zadd(zkey, {eid: act})
                pipe.expire(zkey, self._ttl_seconds)
        await pipe.execute()

    async def record_access(
        self,
        entity_id: str,
        timestamp: float,
        group_id: str | None = None,
    ) -> None:
        """Record an access event, creating state if needed."""
        from engram.activation.engine import record_access as _record_access

        state = await self.get_activation(entity_id)
        if state is None:
            state = ActivationState(node_id=entity_id)
        _record_access(state, timestamp, self._cfg)

        key = self._key(entity_id)
        pipe = self._redis.pipeline()
        pipe.hset(key, mapping=self._serialize(state, group_id=group_id or ""))
        pipe.expire(key, self._ttl_seconds)
        await pipe.execute()

    async def clear_activation(self, entity_id: str) -> None:
        await self._redis.delete(self._key(entity_id))

    async def get_top_activated(
        self,
        group_id: str | None = None,
        limit: int = 20,
        now: float | None = None,
    ) -> list[tuple[str, ActivationState]]:
        """Get top activated entities, using sorted set index when possible."""
        from engram.activation.engine import compute_activation

        now = now if now is not None else time.time()

        # Fast path: use sorted set index if group_id is specified
        if group_id:
            zkey = f"act_group:{group_id}"
            # Get top candidates from sorted set (fetch extra for re-scoring)
            candidates = await self._redis.zrevrange(zkey, 0, limit * 2 - 1)
            if candidates:
                entity_ids = [
                    c.decode() if isinstance(c, bytes) else c for c in candidates
                ]
                states = await self.batch_get(entity_ids)
                scored = []
                for eid, state in states.items():
                    act = compute_activation(
                        state.access_history, now, self._cfg, state.consolidated_strength,
                    )
                    scored.append((eid, state, act))
                scored.sort(key=lambda x: x[2], reverse=True)
                return [(eid, state) for eid, state, _ in scored[:limit]]

        # Fallback: SCAN all keys (original behavior for no group_id)
        scored = []

        async for key in self._redis.scan_iter(match="act:*", count=100):
            data = await self._redis.hgetall(key)
            if not data:
                continue

            key_str = key.decode() if isinstance(key, bytes) else key

            if group_id:
                raw_gid = data.get(b"group_id", data.get("group_id", b""))
                gid = raw_gid.decode() if isinstance(raw_gid, bytes) else raw_gid
                if gid and gid != group_id:
                    continue

            state = self._deserialize(data)
            if state:
                entity_id = key_str.removeprefix("act:")
                act = compute_activation(
                    state.access_history, now, self._cfg, state.consolidated_strength,
                )
                scored.append((entity_id, state, act))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [(eid, state) for eid, state, _ in scored[:limit]]

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._redis.aclose()
