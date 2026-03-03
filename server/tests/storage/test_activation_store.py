"""Tests for MemoryActivationStore."""

import time

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.storage.memory.activation import MemoryActivationStore


@pytest.mark.asyncio
class TestMemoryActivationStore:
    async def test_set_and_get(self, activation_store: MemoryActivationStore):
        state = ActivationState(node_id="ent_1", access_count=5)
        await activation_store.set_activation("ent_1", state)
        result = await activation_store.get_activation("ent_1")
        assert result is not None
        assert result.access_count == 5

    async def test_get_nonexistent(self, activation_store: MemoryActivationStore):
        result = await activation_store.get_activation("nonexistent")
        assert result is None

    async def test_batch_get(self, activation_store: MemoryActivationStore):
        for i in range(3):
            await activation_store.set_activation(
                f"ent_{i}",
                ActivationState(node_id=f"ent_{i}", access_count=i),
            )
        result = await activation_store.batch_get(["ent_0", "ent_1", "ent_2", "ent_99"])
        assert len(result) == 3
        assert "ent_99" not in result

    async def test_batch_set(self, activation_store: MemoryActivationStore):
        states = {
            f"ent_{i}": ActivationState(node_id=f"ent_{i}", access_count=1)
            for i in range(5)
        }
        await activation_store.batch_set(states)
        result = await activation_store.batch_get([f"ent_{i}" for i in range(5)])
        assert len(result) == 5

    async def test_get_top_activated(self, activation_store: MemoryActivationStore):
        """Top activated sorts by lazily computed activation (recent > old)."""
        now = time.time()
        for i in range(5):
            state = ActivationState(
                node_id=f"ent_{i}",
                access_history=[now - (i * 3600)],  # i hours ago
                access_count=1,
                last_accessed=now - (i * 3600),
            )
            await activation_store.set_activation(f"ent_{i}", state)
        top = await activation_store.get_top_activated(limit=3)
        assert len(top) == 3
        # ent_0 (most recent) should be first
        assert top[0][0] == "ent_0"

    async def test_record_access_creates_state(self, activation_store: MemoryActivationStore):
        """record_access creates state for unknown entity."""
        now = time.time()
        await activation_store.record_access("new_ent", now)
        state = await activation_store.get_activation("new_ent")
        assert state is not None
        assert len(state.access_history) == 1
        assert state.access_count == 1
        assert state.last_accessed == now

    async def test_record_access_appends(self, activation_store: MemoryActivationStore):
        """record_access appends to existing entity history."""
        now = time.time()
        await activation_store.record_access("ent_a", now - 100)
        await activation_store.record_access("ent_a", now)
        state = await activation_store.get_activation("ent_a")
        assert len(state.access_history) == 2
        assert state.access_count == 2

    async def test_record_access_caps_history(self):
        """record_access caps at max_history_size."""
        cfg = ActivationConfig(max_history_size=10)
        store = MemoryActivationStore(cfg=cfg)
        now = time.time()
        for i in range(15):
            await store.record_access("ent_cap", now + i)
        state = await store.get_activation("ent_cap")
        assert len(state.access_history) == 10
        assert state.access_count == 15

    async def test_get_top_activated_empty(self, activation_store: MemoryActivationStore):
        """Empty store returns empty list."""
        top = await activation_store.get_top_activated(limit=5)
        assert top == []

    async def test_record_access_same_timestamp(self, activation_store: MemoryActivationStore):
        """Multiple accesses at same timestamp handled gracefully."""
        now = time.time()
        await activation_store.record_access("ent_dup", now)
        await activation_store.record_access("ent_dup", now)
        state = await activation_store.get_activation("ent_dup")
        assert len(state.access_history) == 2
        assert state.access_count == 2

    async def test_get_top_activated_filters_by_group(
        self, activation_store: MemoryActivationStore,
    ):
        """get_top_activated filters by group_id when provided."""
        now = time.time()
        await activation_store.record_access("ent_a", now, group_id="g1")
        await activation_store.record_access("ent_b", now, group_id="g2")
        top = await activation_store.get_top_activated(group_id="g1")
        assert all(eid == "ent_a" for eid, _ in top)

    async def test_clear_activation_removes_group_map(
        self, activation_store: MemoryActivationStore,
    ):
        """clear_activation also removes from group map."""
        now = time.time()
        await activation_store.record_access("ent_grp", now, group_id="g1")
        await activation_store.clear_activation("ent_grp")
        assert "ent_grp" not in activation_store._group_map
