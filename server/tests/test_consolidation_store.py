"""Tests for SQLiteConsolidationStore."""

import time

import pytest
import pytest_asyncio

from engram.consolidation.store import SQLiteConsolidationStore
from engram.models.consolidation import (
    ConsolidationCycle,
    InferredEdge,
    MergeRecord,
    PhaseResult,
    PruneRecord,
)


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteConsolidationStore(str(tmp_path / "consol.db"))
    await s.initialize()
    yield s
    await s.close()


class TestConsolidationStore:
    """CRUD operations on consolidation store."""

    @pytest.mark.asyncio
    async def test_save_and_get_cycle(self, store):
        cycle = ConsolidationCycle(group_id="test", trigger="manual")
        cycle.phase_results.append(PhaseResult(phase="merge", items_processed=10, items_affected=2))
        await store.save_cycle(cycle)

        fetched = await store.get_cycle(cycle.id, "test")
        assert fetched is not None
        assert fetched.id == cycle.id
        assert fetched.trigger == "manual"
        assert len(fetched.phase_results) == 1
        assert fetched.phase_results[0].items_affected == 2

    @pytest.mark.asyncio
    async def test_update_cycle(self, store):
        cycle = ConsolidationCycle(group_id="test")
        await store.save_cycle(cycle)

        cycle.status = "completed"
        cycle.completed_at = time.time()
        cycle.total_duration_ms = 123.4
        await store.update_cycle(cycle)

        fetched = await store.get_cycle(cycle.id, "test")
        assert fetched.status == "completed"
        assert fetched.total_duration_ms == 123.4

    @pytest.mark.asyncio
    async def test_get_recent_cycles(self, store):
        for i in range(5):
            c = ConsolidationCycle(group_id="test")
            c.started_at = time.time() + i
            await store.save_cycle(c)

        recent = await store.get_recent_cycles("test", limit=3)
        assert len(recent) == 3
        # Should be newest first
        assert recent[0].started_at >= recent[1].started_at

    @pytest.mark.asyncio
    async def test_group_id_filtering(self, store):
        c1 = ConsolidationCycle(group_id="group_a")
        c2 = ConsolidationCycle(group_id="group_b")
        await store.save_cycle(c1)
        await store.save_cycle(c2)

        result_a = await store.get_recent_cycles("group_a")
        result_b = await store.get_recent_cycles("group_b")
        assert len(result_a) == 1
        assert len(result_b) == 1
        assert result_a[0].group_id == "group_a"

    @pytest.mark.asyncio
    async def test_save_merge_record(self, store):
        record = MergeRecord(
            cycle_id="cyc_test",
            group_id="test",
            keep_id="e1",
            remove_id="e2",
            keep_name="Alice",
            remove_name="alice",
            similarity=0.92,
            relationships_transferred=3,
        )
        await store.save_merge_record(record)

        records = await store.get_merge_records("cyc_test", "test")
        assert len(records) == 1
        assert records[0].keep_name == "Alice"
        assert records[0].relationships_transferred == 3

    @pytest.mark.asyncio
    async def test_save_inferred_edge(self, store):
        edge = InferredEdge(
            cycle_id="cyc_test",
            group_id="test",
            source_id="e1",
            target_id="e2",
            source_name="Python",
            target_name="FastAPI",
            co_occurrence_count=5,
            confidence=0.75,
        )
        await store.save_inferred_edge(edge)

        edges = await store.get_inferred_edges("cyc_test", "test")
        assert len(edges) == 1
        assert edges[0].source_name == "Python"

    @pytest.mark.asyncio
    async def test_save_prune_record(self, store):
        record = PruneRecord(
            cycle_id="cyc_test",
            group_id="test",
            entity_id="e1",
            entity_name="Dead Entity",
            entity_type="Concept",
            reason="dead_entity",
        )
        await store.save_prune_record(record)

        records = await store.get_prune_records("cyc_test", "test")
        assert len(records) == 1
        assert records[0].entity_name == "Dead Entity"

    @pytest.mark.asyncio
    async def test_cleanup_old_records(self, store):
        old_cycle = ConsolidationCycle(group_id="test")
        old_cycle.started_at = time.time() - (100 * 86400)  # 100 days ago
        await store.save_cycle(old_cycle)

        new_cycle = ConsolidationCycle(group_id="test")
        await store.save_cycle(new_cycle)

        # Add records to old cycle
        await store.save_merge_record(
            MergeRecord(
                cycle_id=old_cycle.id,
                group_id="test",
                keep_id="e1",
                remove_id="e2",
                keep_name="A",
                remove_name="B",
                similarity=0.9,
            )
        )

        deleted = await store.cleanup(ttl_days=90)
        assert deleted == 1

        remaining = await store.get_recent_cycles("test")
        assert len(remaining) == 1
        assert remaining[0].id == new_cycle.id
