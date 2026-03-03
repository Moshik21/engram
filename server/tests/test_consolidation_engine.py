"""Tests for the ConsolidationEngine."""

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.store import SQLiteConsolidationStore
from engram.events.bus import EventBus
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


@pytest_asyncio.fixture
async def consol_store(store):
    s = SQLiteConsolidationStore(store._db_path)
    await s.initialize(db=store._db)
    return s


@pytest_asyncio.fixture
async def engine(store, activation, search, consol_store):
    bus = EventBus()
    return ConsolidationEngine(
        store, activation, search,
        cfg=ActivationConfig(),
        consolidation_store=consol_store,
        event_bus=bus,
    )


class TestConsolidationEngine:
    """Tests for ConsolidationEngine orchestration."""

    @pytest.mark.asyncio
    async def test_full_cycle_completes(self, engine):
        cycle = await engine.run_cycle(group_id="test", dry_run=True)

        assert cycle.status == "completed"
        assert len(cycle.phase_results) == 5
        assert cycle.total_duration_ms > 0
        phase_names = [pr.phase for pr in cycle.phase_results]
        assert phase_names == ["merge", "infer", "prune", "compact", "reindex"]

    @pytest.mark.asyncio
    async def test_empty_graph_completes(self, engine):
        cycle = await engine.run_cycle(group_id="test", dry_run=False)

        assert cycle.status == "completed"
        for pr in cycle.phase_results:
            assert pr.items_affected == 0

    @pytest.mark.asyncio
    async def test_phase_ordering(self, engine):
        """Phases should run in order: merge → infer → prune → compact."""
        cycle = await engine.run_cycle(group_id="test")

        names = [pr.phase for pr in cycle.phase_results]
        assert names == ["merge", "infer", "prune", "compact", "reindex"]

    @pytest.mark.asyncio
    async def test_prevents_concurrent_cycles(self, engine):
        # Simulate running state
        engine._running = True

        with pytest.raises(RuntimeError, match="already running"):
            await engine.run_cycle(group_id="test")

        engine._running = False

    @pytest.mark.asyncio
    async def test_cancel_between_phases(self, engine):
        """Cancel should stop execution between phases."""
        original_execute = engine._phases[0].execute

        async def slow_merge(*args, **kwargs):
            engine.cancel()
            return await original_execute(*args, **kwargs)

        engine._phases[0].execute = slow_merge

        cycle = await engine.run_cycle(group_id="test")

        assert cycle.status == "cancelled"
        # Should have merge result but not all 5 phases
        assert len(cycle.phase_results) <= 2  # merge completes, then cancelled

    @pytest.mark.asyncio
    async def test_non_fatal_phase_errors(self, store, activation, search, consol_store):
        """A failing phase should not prevent other phases from running."""
        bus = EventBus()
        engine = ConsolidationEngine(
            store, activation, search,
            cfg=ActivationConfig(),
            consolidation_store=consol_store,
            event_bus=bus,
        )

        # Make merge phase raise an error
        async def failing_merge(*args, **kwargs):
            raise ValueError("Simulated merge failure")

        engine._phases[0].execute = failing_merge

        cycle = await engine.run_cycle(group_id="test")

        assert cycle.status == "completed"
        assert len(cycle.phase_results) == 5
        assert cycle.phase_results[0].status == "error"
        assert "Simulated merge failure" in cycle.phase_results[0].error
        # Other phases should succeed
        for pr in cycle.phase_results[1:]:
            assert pr.status == "success"

    @pytest.mark.asyncio
    async def test_event_publishing(self, store, activation, search, consol_store):
        bus = EventBus()
        q = bus.subscribe("test")
        engine = ConsolidationEngine(
            store, activation, search,
            cfg=ActivationConfig(),
            consolidation_store=consol_store,
            event_bus=bus,
        )

        await engine.run_cycle(group_id="test")

        events = []
        while not q.empty():
            events.append(await q.get())

        event_types = [e["type"] for e in events]
        assert "consolidation.started" in event_types
        assert "consolidation.completed" in event_types
        assert "consolidation.phase.merge.started" in event_types
        assert "consolidation.phase.merge.completed" in event_types

    @pytest.mark.asyncio
    async def test_cycle_persisted_to_store(self, engine, consol_store):
        cycle = await engine.run_cycle(group_id="test")

        fetched = await consol_store.get_cycle(cycle.id, "test")
        assert fetched is not None
        assert fetched.status == "completed"
        assert len(fetched.phase_results) == 5

    @pytest.mark.asyncio
    async def test_is_running_property(self, engine):
        assert engine.is_running is False
        # After cycle completes, should be false again
        await engine.run_cycle(group_id="test")
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_dry_run_default_from_config(self, store, activation, search, consol_store):
        """dry_run=None should use config default."""
        cfg = ActivationConfig(consolidation_dry_run=True)
        engine = ConsolidationEngine(
            store, activation, search, cfg=cfg,
            consolidation_store=consol_store,
        )

        cycle = await engine.run_cycle(group_id="test", dry_run=None)
        assert cycle.dry_run is True
