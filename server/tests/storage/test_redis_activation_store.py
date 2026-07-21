"""Tests for RedisActivationStore without a live Redis service."""

from __future__ import annotations

import fnmatch

import pytest

from engram.models.activation import DEFAULT_USAGE_TIER_WEIGHTS, ActivationState
from engram.storage.redis.activation import RedisActivationStore


class _FakeRedisPipeline:
    def __init__(self, redis: _FakeRedis) -> None:
        self._redis = redis
        self._ops: list[tuple[str, tuple, dict]] = []

    def hset(self, key: str, mapping: dict) -> _FakeRedisPipeline:
        self._ops.append(("hset", (key,), {"mapping": mapping}))
        return self

    def hgetall(self, key: str) -> _FakeRedisPipeline:
        self._ops.append(("hgetall", (key,), {}))
        return self

    def expire(self, key: str, ttl: int) -> _FakeRedisPipeline:
        self._ops.append(("expire", (key, ttl), {}))
        return self

    def zadd(self, key: str, mapping: dict[str, float]) -> _FakeRedisPipeline:
        self._ops.append(("zadd", (key, mapping), {}))
        return self

    def zrem(self, key: str, member: str) -> _FakeRedisPipeline:
        self._ops.append(("zrem", (key, member), {}))
        return self

    def delete(self, key: str) -> _FakeRedisPipeline:
        self._ops.append(("delete", (key,), {}))
        return self

    async def execute(self) -> list:
        results = []
        for name, args, kwargs in self._ops:
            if name == "hset":
                key = args[0]
                self._redis.hashes[key] = dict(kwargs["mapping"])
                results.append(True)
            elif name == "hgetall":
                results.append(dict(self._redis.hashes.get(args[0], {})))
            elif name == "expire":
                self._redis.expirations.append(args)
                results.append(True)
            elif name == "zadd":
                key, mapping = args
                self._redis.zsets.setdefault(key, {}).update(mapping)
                results.append(True)
            elif name == "zrem":
                key, member = args
                self._redis.zsets.setdefault(key, {}).pop(member, None)
                results.append(True)
            elif name == "delete":
                self._redis.hashes.pop(args[0], None)
                results.append(True)
        return results


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.expirations: list[tuple] = []
        self.closed = False

    def pipeline(self) -> _FakeRedisPipeline:
        return _FakeRedisPipeline(self)

    async def hgetall(self, key: str) -> dict:
        return dict(self.hashes.get(key, {}))

    async def zrevrange(self, key: str, start: int, stop: int) -> list[str]:
        items = sorted(
            self.zsets.get(key, {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [member for member, _score in items[start : stop + 1]]

    async def scan_iter(self, match: str, count: int):
        del count
        for key in list(self.hashes):
            if fnmatch.fnmatch(key, match):
                yield key

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_set_activation_preserves_existing_group_index() -> None:
    redis = _FakeRedis()
    store = RedisActivationStore(redis)
    await store.record_access("ent_a", 1_000.0, group_id="brain_a")

    state = await store.get_activation("ent_a")
    assert state is not None
    state.consolidated_strength = 0.5
    await store.set_activation("ent_a", state)

    assert redis.hashes["act:ent_a"]["group_id"] == "brain_a"
    assert "ent_a" in redis.zsets["act_group:brain_a"]
    top = await store.get_top_activated(group_id="brain_a", now=1_100.0)
    assert [entity_id for entity_id, _state in top] == ["ent_a"]


@pytest.mark.asyncio
async def test_batch_set_preserves_existing_group_indexes() -> None:
    redis = _FakeRedis()
    store = RedisActivationStore(redis)
    await store.record_access("ent_a", 1_000.0, group_id="brain_a")
    await store.record_access("ent_b", 1_005.0, group_id="brain_b")

    states = await store.batch_get(["ent_a", "ent_b"])
    states["ent_a"].consolidated_strength = 0.3
    states["ent_b"].consolidated_strength = 0.7
    await store.batch_set(states)

    assert redis.hashes["act:ent_a"]["group_id"] == "brain_a"
    assert redis.hashes["act:ent_b"]["group_id"] == "brain_b"
    assert "ent_a" in redis.zsets["act_group:brain_a"]
    assert "ent_b" in redis.zsets["act_group:brain_b"]
    brain_b_top = await store.get_top_activated(group_id="brain_b", now=1_100.0)
    assert [entity_id for entity_id, _state in brain_b_top] == ["ent_b"]


@pytest.mark.asyncio
async def test_get_top_activated_ignores_stale_wrong_group_index() -> None:
    redis = _FakeRedis()
    store = RedisActivationStore(redis)
    redis.hashes["act:ent_a"] = store._serialize(
        ActivationState(node_id="ent_a", access_history=[1_000.0], access_count=1),
        group_id="brain_a",
    )
    redis.zsets["act_group:brain_b"] = {"ent_a": 1.0}

    top = await store.get_top_activated(group_id="brain_b", now=1_100.0)

    assert top == []


@pytest.mark.asyncio
async def test_clear_activation_removes_group_index_member() -> None:
    redis = _FakeRedis()
    store = RedisActivationStore(redis)
    await store.record_access("ent_a", 1_000.0, group_id="brain_a")

    await store.clear_activation("ent_a")

    assert "act:ent_a" not in redis.hashes
    assert "ent_a" not in redis.zsets["act_group:brain_a"]


@pytest.mark.asyncio
async def test_record_access_all_tiers_hygiene_only_no_usage_side_effects() -> None:
    """RF M4.3 pin: the full-mode Redis lane is DOCUMENTED-INERT for usage.

    record_access must accept every M1 tier without error and append hygiene
    history (access_history/access_count/last_accessed), but the usage-event
    mechanism stays inert: no usage_events, no usage caches, nothing
    usage-shaped in the persisted hash. If someone half-wires usage into this
    store (silent divergence from the documented choice), this test fails.
    """
    redis = _FakeRedis()
    store = RedisActivationStore(redis)
    # Sourced from the canonical tier table so a newly added tier is
    # automatically covered by this pin.
    tiers = sorted(DEFAULT_USAGE_TIER_WEIGHTS)
    assert set(tiers) == {"surfaced", "mentioned", "used", "corrected", "confirmed"}

    for i, tier in enumerate(tiers):
        await store.record_access("ent_a", 1_000.0 + i, group_id="brain_a", tier=tier)

    state = await store.get_activation("ent_a")
    assert state is not None
    # Hygiene history recorded for every tier.
    assert state.access_count == len(tiers)
    assert state.access_history == [1_000.0 + i for i in range(len(tiers))]
    assert state.last_accessed == 1_000.0 + len(tiers) - 1
    # Zero usage side-effects in the round-tripped state...
    assert state.usage_events == []
    assert state.usage_weight_sum == 0.0
    assert state.usage_last_ts == 0.0
    assert state.n_eff == 0.0
    # ...and nothing usage-shaped persisted anywhere in the store.
    for key, stored_hash in redis.hashes.items():
        assert not any("usage" in field for field in stored_hash), (key, stored_hash)
