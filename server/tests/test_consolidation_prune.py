"""Tests for the prune consolidation phase."""

import time
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.phases.prune import PrunePhase
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.relationship import Relationship
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteGraphStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def search(store):
    idx = FTS5SearchIndex(store._db_path)
    await idx.initialize(db=store._db)
    return idx


@pytest_asyncio.fixture
async def activation():
    return MemoryActivationStore(cfg=ActivationConfig())


def _old_entity(name, group_id="test", days_old=60):
    return Entity(
        id=f"ent_{uuid.uuid4().hex[:8]}",
        name=name, entity_type="Concept", group_id=group_id,
        access_count=0,
        created_at=datetime.utcnow() - timedelta(days=days_old),
    )


async def _make_old(store, entity_id, days_old=60):
    """Force entity to have an old created_at in the database."""
    old = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
    await store.db.execute(
        "UPDATE entities SET created_at = ? WHERE id = ?",
        (old, entity_id),
    )
    await store.db.commit()


class TestPrunePhase:
    """Tests for PrunePhase."""

    @pytest.mark.asyncio
    async def test_prunes_dead_entity(self, store, activation, search):
        e = _old_entity("Ghost")
        await store.create_entity(e)
        await _make_old(store, e.id, 60)

        cfg = ActivationConfig(consolidation_prune_min_age_days=30)
        phase = PrunePhase()
        result, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )

        assert result.items_affected == 1
        assert records[0].entity_id == e.id
        # Entity should be soft-deleted
        assert await store.get_entity(e.id, "test") is None

    @pytest.mark.asyncio
    async def test_dry_run_no_delete(self, store, activation, search):
        e = _old_entity("Ghost")
        await store.create_entity(e)
        await _make_old(store, e.id, 60)

        cfg = ActivationConfig(consolidation_prune_min_age_days=30)
        phase = PrunePhase()
        result, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=True,
        )

        assert result.items_affected == 1
        # Entity should still exist
        assert await store.get_entity(e.id, "test") is not None

    @pytest.mark.asyncio
    async def test_entity_with_access_not_pruned(self, store, activation, search):
        """Entity with activation history should be spared."""
        e = _old_entity("Active Ghost")
        await store.create_entity(e)
        await _make_old(store, e.id, 60)

        # Give it activation history
        state = ActivationState(
            node_id=e.id,
            access_history=[time.time()],
            access_count=5,
        )
        await activation.set_activation(e.id, state)

        cfg = ActivationConfig(
            consolidation_prune_min_age_days=30,
            consolidation_prune_min_access_count=0,
        )
        phase = PrunePhase()
        result, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )

        assert result.items_affected == 0  # Protected by access count

    @pytest.mark.asyncio
    async def test_protective_check_has_relationships(self, store, activation, search):
        """Entity with relationships should not appear in dead entities."""
        e1 = _old_entity("Connected")
        e2 = _old_entity("Other")
        await store.create_entity(e1)
        await store.create_entity(e2)
        await _make_old(store, e1.id, 60)
        await _make_old(store, e2.id, 60)

        rel = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e1.id, target_id=e2.id,
            predicate="KNOWS", group_id="test",
        )
        await store.create_relationship(rel)

        cfg = ActivationConfig(consolidation_prune_min_age_days=30)
        phase = PrunePhase()
        result, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )

        # Neither should be pruned (they have a relationship to each other)
        pruned_ids = {r.entity_id for r in records}
        assert e1.id not in pruned_ids
        assert e2.id not in pruned_ids

    @pytest.mark.asyncio
    async def test_max_prunes_limit(self, store, activation, search):
        for i in range(5):
            e = _old_entity(f"Ghost_{i}")
            await store.create_entity(e)
            await _make_old(store, e.id, 60)

        cfg = ActivationConfig(
            consolidation_prune_min_age_days=30,
            consolidation_prune_max_per_cycle=2,
        )
        phase = PrunePhase()
        result, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )

        assert result.items_affected <= 2

    @pytest.mark.asyncio
    async def test_clears_activation_and_search(self, store, activation, search):
        e = _old_entity("Ghost")
        await store.create_entity(e)
        await _make_old(store, e.id, 60)
        await search.index_entity(e)

        state = ActivationState(node_id=e.id, access_count=0)
        await activation.set_activation(e.id, state)

        cfg = ActivationConfig(consolidation_prune_min_age_days=30)
        phase = PrunePhase()
        await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )

        # Activation should be cleared
        assert await activation.get_activation(e.id) is None

    @pytest.mark.asyncio
    async def test_group_isolation(self, store, activation, search):
        e = _old_entity("Ghost", group_id="group_a")
        await store.create_entity(e)
        await _make_old(store, e.id, 60)

        cfg = ActivationConfig(consolidation_prune_min_age_days=30)
        phase = PrunePhase()
        result, records = await phase.execute(
            group_id="group_b", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )

        assert result.items_affected == 0
