"""Tests for CycleContext flow through consolidation phases."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.phases.compact import AccessHistoryCompactionPhase
from engram.consolidation.phases.infer import EdgeInferencePhase
from engram.consolidation.phases.merge import EntityMergePhase
from engram.consolidation.phases.prune import PrunePhase
from engram.consolidation.store import SQLiteConsolidationStore
from engram.events.bus import EventBus
from engram.models.consolidation import CycleContext
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


class TestCycleContextDefaults:
    def test_context_defaults_empty(self):
        ctx = CycleContext()
        assert ctx.affected_entity_ids == set()
        assert ctx.merge_survivor_ids == set()
        assert ctx.inferred_edge_entity_ids == set()
        assert ctx.pruned_entity_ids == set()
        assert ctx.replay_new_entity_ids == set()

    def test_context_sets_are_independent(self):
        ctx = CycleContext()
        ctx.affected_entity_ids.add("a")
        ctx.merge_survivor_ids.add("b")
        assert "a" not in ctx.merge_survivor_ids
        assert "b" not in ctx.affected_entity_ids


class TestMergePopulatesContext:
    @pytest.mark.asyncio
    async def test_merge_adds_survivor_ids_to_context(self):
        """EntityMergePhase should populate merge_survivor_ids and affected_entity_ids."""
        phase = EntityMergePhase()
        cfg = ActivationConfig(
            consolidation_merge_threshold=0.50,
            consolidation_merge_require_same_type=False,
        )

        # Mock stores
        entity_a = MagicMock()
        entity_a.id = "ent_a"
        entity_a.name = "Alice"
        entity_a.entity_type = "person"
        entity_a.access_count = 5
        entity_a.created_at = 1000.0

        entity_b = MagicMock()
        entity_b.id = "ent_b"
        entity_b.name = "alice"  # Near-dupe
        entity_b.entity_type = "person"
        entity_b.access_count = 1
        entity_b.created_at = 2000.0

        graph_store = AsyncMock()
        graph_store.find_entities = AsyncMock(return_value=[entity_a, entity_b])
        graph_store.merge_entities = AsyncMock(return_value=2)

        activation_store = AsyncMock()
        activation_store.get_activation = AsyncMock(return_value=None)

        search_index = AsyncMock()
        ctx = CycleContext()

        await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
            context=ctx,
        )

        assert "ent_a" in ctx.merge_survivor_ids
        assert "ent_a" in ctx.affected_entity_ids


class TestInferPopulatesContext:
    @pytest.mark.asyncio
    async def test_infer_adds_edge_entity_ids_to_context(self):
        """EdgeInferencePhase should populate inferred_edge_entity_ids."""
        phase = EdgeInferencePhase()
        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=2,
            consolidation_infer_auto_validation_enabled=False,
            consolidation_infer_pmi_enabled=False,
        )

        entity_a = MagicMock()
        entity_a.name = "Alice"
        entity_b = MagicMock()
        entity_b.name = "Bob"

        graph_store = AsyncMock()
        graph_store.get_co_occurring_entity_pairs = AsyncMock(
            return_value=[("ent_a", "ent_b", 3)],
        )
        graph_store.get_entity = AsyncMock(
            side_effect=lambda eid, gid: entity_a if eid == "ent_a" else entity_b,
        )
        graph_store.create_relationship = AsyncMock()
        graph_store.get_relationships = AsyncMock(return_value=[])
        graph_store.find_existing_relationship = AsyncMock(return_value=None)
        graph_store.find_conflicting_relationships = AsyncMock(return_value=[])
        graph_store.update_relationship_weight = AsyncMock(return_value=None)
        graph_store.invalidate_relationship = AsyncMock()

        activation_store = AsyncMock()
        search_index = AsyncMock()
        ctx = CycleContext()

        await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
            context=ctx,
        )

        assert "ent_a" in ctx.inferred_edge_entity_ids
        assert "ent_b" in ctx.inferred_edge_entity_ids
        assert "ent_a" in ctx.affected_entity_ids
        assert "ent_b" in ctx.affected_entity_ids


class TestPrunePopulatesContext:
    @pytest.mark.asyncio
    async def test_prune_adds_pruned_ids_not_affected(self):
        """PrunePhase should populate pruned_entity_ids but NOT affected_entity_ids."""
        phase = PrunePhase()
        cfg = ActivationConfig(
            consolidation_prune_min_age_days=1,
            consolidation_prune_min_access_count=0,
            consolidation_prune_activation_floor=0.5,
        )

        dead_entity = MagicMock()
        dead_entity.id = "ent_dead"
        dead_entity.name = "Dead"
        dead_entity.entity_type = "concept"

        graph_store = AsyncMock()
        graph_store.get_dead_entities = AsyncMock(return_value=[dead_entity])
        graph_store.delete_entity = AsyncMock()

        activation_store = AsyncMock()
        activation_store.get_activation = AsyncMock(return_value=None)

        search_index = AsyncMock()
        ctx = CycleContext()

        await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
            context=ctx,
        )

        assert "ent_dead" in ctx.pruned_entity_ids
        assert "ent_dead" not in ctx.affected_entity_ids


class TestEnginePhaseOrder:
    @pytest_asyncio.fixture
    async def store(self, tmp_path):
        s = SQLiteGraphStore(str(tmp_path / "test.db"))
        await s.initialize()
        yield s
        await s.close()

    @pytest_asyncio.fixture
    async def search(self, store):
        idx = FTS5SearchIndex(store._db_path)
        await idx.initialize(db=store._db)
        return idx

    @pytest_asyncio.fixture
    async def activation(self):
        return MemoryActivationStore(cfg=ActivationConfig())

    @pytest_asyncio.fixture
    async def consol_store(self, store):
        s = SQLiteConsolidationStore(store._db_path)
        await s.initialize(db=store._db)
        return s

    @pytest_asyncio.fixture
    async def engine(self, store, activation, search, consol_store):
        bus = EventBus()
        return ConsolidationEngine(
            store,
            activation,
            search,
            cfg=ActivationConfig(),
            consolidation_store=consol_store,
            event_bus=bus,
        )

    @pytest.mark.asyncio
    async def test_engine_runs_15_phases(self, engine):
        """Engine should now run 15 phases in correct order."""
        cycle = await engine.run_cycle(group_id="test", dry_run=True)
        assert len(cycle.phase_results) == 15
        names = [pr.phase for pr in cycle.phase_results]
        assert names == [
            "triage", "merge", "infer", "evidence_adjudication",
            "edge_adjudication",
            "replay", "prune", "compact", "mature", "semanticize",
            "schema", "reindex", "graph_embed", "microglia", "dream",
        ]

    @pytest.mark.asyncio
    async def test_existing_phases_work_with_context_none(self):
        """Existing phases should work fine when context is not passed."""
        phase = AccessHistoryCompactionPhase()
        activation_store = AsyncMock()
        activation_store.get_top_activated = AsyncMock(return_value=[])

        result, records = await phase.execute(
            group_id="test",
            graph_store=AsyncMock(),
            activation_store=activation_store,
            search_index=AsyncMock(),
            cfg=ActivationConfig(),
            cycle_id="cyc_test",
            dry_run=True,
        )
        assert result.phase == "compact"
        assert result.status == "success"
