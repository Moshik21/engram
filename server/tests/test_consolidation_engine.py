"""Tests for the ConsolidationEngine."""

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.phases.base import ConsolidationPhase
from engram.consolidation.store import SQLiteConsolidationStore
from engram.events.bus import EventBus
from engram.models.consolidation import DecisionOutcomeLabel, DecisionTrace, PhaseResult
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
        store,
        activation,
        search,
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
        assert len(cycle.phase_results) == 15
        assert cycle.total_duration_ms > 0
        phase_names = [pr.phase for pr in cycle.phase_results]
        assert phase_names == [
            "triage",
            "merge",
            "infer",
            "evidence_adjudication",
            "edge_adjudication",
            "replay",
            "prune",
            "compact",
            "mature",
            "semanticize",
            "schema",
            "reindex",
            "graph_embed",
            "microglia",
            "dream",
        ]

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
        assert names == [
            "triage",
            "merge",
            "infer",
            "evidence_adjudication",
            "edge_adjudication",
            "replay",
            "prune",
            "compact",
            "mature",
            "semanticize",
            "schema",
            "reindex",
            "graph_embed",
            "microglia",
            "dream",
        ]

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
            store,
            activation,
            search,
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
        assert len(cycle.phase_results) == 15
        assert cycle.phase_results[0].status == "error"
        assert "Simulated merge failure" in cycle.phase_results[0].error
        # Other phases should succeed or be skipped (dream is disabled by default)
        for pr in cycle.phase_results[1:]:
            assert pr.status in ("success", "skipped")

    @pytest.mark.asyncio
    async def test_event_publishing(self, store, activation, search, consol_store):
        bus = EventBus()
        q = bus.subscribe("test")
        engine = ConsolidationEngine(
            store,
            activation,
            search,
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
        assert len(fetched.phase_results) == 15

    @pytest.mark.asyncio
    async def test_is_running_property(self, engine):
        assert engine.is_running is False
        # After cycle completes, should be false again
        await engine.run_cycle(group_id="test")
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_graph_embed_record_persisted(self, store, activation, search, consol_store):
        """GraphEmbedRecord should be persisted to audit tables after a cycle."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=1,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        bus = EventBus()

        # Create enough entities for node2vec
        from engram.models.entity import Entity

        for i in range(20):
            entity = Entity(
                id=f"ent_{i}",
                name=f"Entity_{i}",
                entity_type="Concept",
                summary=f"Test entity {i}",
                group_id="test",
            )
            await store.create_entity(entity)

        engine = ConsolidationEngine(
            store,
            activation,
            search,
            cfg=cfg,
            consolidation_store=consol_store,
            event_bus=bus,
        )

        cycle = await engine.run_cycle(group_id="test", dry_run=True)

        assert cycle.status == "completed"
        # Find the graph_embed phase result
        ge_result = next(
            (pr for pr in cycle.phase_results if pr.phase == "graph_embed"),
            None,
        )
        assert ge_result is not None

    @pytest.mark.asyncio
    async def test_replay_after_merge_infer(self, engine):
        """Replay phase should run after merge and infer phases."""
        cycle = await engine.run_cycle(group_id="test", dry_run=True)
        names = [pr.phase for pr in cycle.phase_results]
        merge_idx = names.index("merge")
        infer_idx = names.index("infer")
        replay_idx = names.index("replay")
        assert merge_idx < replay_idx, "merge must run before replay"
        assert infer_idx < replay_idx, "infer must run before replay"

    @pytest.mark.asyncio
    async def test_replay_skipped_without_graph_changes(self, engine):
        """Replay should be skipped when merge/infer produce no changes."""
        cfg = ActivationConfig(consolidation_replay_enabled=True)
        engine._cfg = cfg
        cycle = await engine.run_cycle(group_id="test", dry_run=True)
        replay_result = next(pr for pr in cycle.phase_results if pr.phase == "replay")
        assert replay_result.status == "skipped"

    @pytest.mark.asyncio
    async def test_dry_run_default_from_config(self, store, activation, search, consol_store):
        """dry_run=None should use config default."""
        cfg = ActivationConfig(consolidation_dry_run=True)
        engine = ConsolidationEngine(
            store,
            activation,
            search,
            cfg=cfg,
            consolidation_store=consol_store,
        )

        cycle = await engine.run_cycle(group_id="test", dry_run=None)
        assert cycle.dry_run is True

    @pytest.mark.asyncio
    async def test_phase_capability_validation_fails_cycle(
        self,
        store,
        activation,
        search,
        consol_store,
    ):
        class MissingCapabilityPhase(ConsolidationPhase):
            @property
            def name(self) -> str:
                return "missing_cap"

            def required_graph_store_methods(self, cfg):
                return {"definitely_missing_method"}

            async def execute(self, *args, **kwargs):
                return PhaseResult(phase=self.name), []

        engine = ConsolidationEngine(
            store,
            activation,
            search,
            cfg=ActivationConfig(),
            consolidation_store=consol_store,
        )
        engine._phases = [MissingCapabilityPhase()]

        cycle = await engine.run_cycle(group_id="test", dry_run=True)

        assert cycle.status == "failed"
        assert "definitely_missing_method" in (cycle.error or "")

    @pytest.mark.asyncio
    async def test_decision_traces_persisted_from_context(
        self,
        store,
        activation,
        search,
        consol_store,
    ):
        class TracePhase(ConsolidationPhase):
            @property
            def name(self) -> str:
                return "trace_phase"

            async def execute(
                self,
                group_id,
                graph_store,
                activation_store,
                search_index,
                cfg,
                cycle_id,
                dry_run=False,
                context=None,
            ):
                trace = DecisionTrace(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    phase=self.name,
                    candidate_type="entity_pair",
                    candidate_id="a:b",
                    decision="merge",
                    decision_source="unit_test",
                )
                label = DecisionOutcomeLabel(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    phase=self.name,
                    decision_trace_id=trace.id,
                    outcome_type="materialization",
                    label="applied",
                    value=1.0,
                )
                context.add_decision_trace(trace)
                context.add_decision_outcome_label(label)
                return PhaseResult(phase=self.name, items_affected=1), []

        engine = ConsolidationEngine(
            store,
            activation,
            search,
            cfg=ActivationConfig(),
            consolidation_store=consol_store,
        )
        engine._phases = [TracePhase()]

        cycle = await engine.run_cycle(group_id="test", dry_run=True)

        traces = await consol_store.get_decision_traces(cycle.id, "test")
        labels = await consol_store.get_decision_outcome_labels(cycle.id, "test")
        assert len(traces) == 1
        assert traces[0].decision_source == "unit_test"
        assert len(labels) == 1
        assert labels[0].decision_trace_id == traces[0].id

    @pytest.mark.asyncio
    async def test_stage3_learning_artifacts_persisted(
        self,
        store,
        activation,
        search,
        consol_store,
    ):
        class LearningPhase(ConsolidationPhase):
            @property
            def name(self) -> str:
                return "learning_phase"

            async def execute(
                self,
                group_id,
                graph_store,
                activation_store,
                search_index,
                cfg,
                cycle_id,
                dry_run=False,
                context=None,
            ):
                trace = DecisionTrace(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    phase=self.name,
                    candidate_type="entity_pair",
                    candidate_id="a:b",
                    decision="merge",
                    decision_source="llm",
                    confidence=0.84,
                    threshold_band="accepted",
                    features={"name_similarity": 0.92},
                )
                label = DecisionOutcomeLabel(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    phase=self.name,
                    decision_trace_id=trace.id,
                    outcome_type="materialization",
                    label="applied",
                    value=1.0,
                )
                context.add_decision_trace(trace)
                context.add_decision_outcome_label(label)
                return PhaseResult(phase=self.name, items_affected=1), []

        bus = EventBus()
        q = bus.subscribe("test")
        engine = ConsolidationEngine(
            store,
            activation,
            search,
            cfg=ActivationConfig(
                consolidation_distillation_enabled=True,
                consolidation_calibration_enabled=True,
                consolidation_calibration_window_cycles=5,
                consolidation_calibration_min_examples=1,
                consolidation_calibration_bins=2,
            ),
            consolidation_store=consol_store,
            event_bus=bus,
        )
        engine._phases = [LearningPhase()]

        cycle = await engine.run_cycle(group_id="test", dry_run=True)

        examples = await consol_store.get_distillation_examples(cycle.id, "test")
        snapshots = await consol_store.get_calibration_snapshots(cycle.id, "test")
        assert len(examples) == 2
        assert {example.teacher_source for example in examples} == {
            "oracle:llm",
            "outcome:materialization",
        }
        assert len(snapshots) == 1
        assert snapshots[0].phase == "learning_phase"
        assert snapshots[0].labeled_examples == 1
        assert snapshots[0].oracle_examples == 1
        assert snapshots[0].accuracy == 1.0
        assert snapshots[0].summary["cycles_observed"] == 1

        event_types = []
        while not q.empty():
            event_types.append((await q.get())["type"])
        assert "consolidation.learning.updated" in event_types


class TestShutdownTrigger:
    """Tests for shutdown consolidation trigger in main._shutdown()."""

    @pytest.mark.asyncio
    async def test_shutdown_trigger_calls_run_cycle(self, store, activation, search, consol_store):
        """Shutdown should run a consolidation cycle when enabled and not running."""
        from unittest.mock import AsyncMock

        from engram.config import EngramConfig
        from engram.main import _app_state, _shutdown

        bus = EventBus()
        engine = ConsolidationEngine(
            store,
            activation,
            search,
            cfg=ActivationConfig(consolidation_enabled=True),
            consolidation_store=consol_store,
            event_bus=bus,
        )
        config = EngramConfig()
        config.activation.consolidation_enabled = True

        _app_state.update(
            {
                "config": config,
                "consolidation_engine": engine,
                "consolidation_scheduler": None,
                "pressure_accumulator": None,
                "embedding_provider": None,
                "activation_store": None,
                "graph_store": None,
            }
        )

        engine.run_cycle = AsyncMock(return_value=None)

        await _shutdown()

        engine.run_cycle.assert_called_once_with(
            group_id=config.default_group_id,
            trigger="shutdown",
            dry_run=False,
        )

        _app_state.clear()

    @pytest.mark.asyncio
    async def test_shutdown_skips_when_disabled(self, store, activation, search, consol_store):
        """Shutdown should NOT run consolidation when consolidation_enabled=False."""
        from unittest.mock import AsyncMock

        from engram.config import EngramConfig
        from engram.main import _app_state, _shutdown

        bus = EventBus()
        engine = ConsolidationEngine(
            store,
            activation,
            search,
            cfg=ActivationConfig(consolidation_enabled=False),
            consolidation_store=consol_store,
            event_bus=bus,
        )
        config = EngramConfig()
        config.activation.consolidation_enabled = False

        _app_state.update(
            {
                "config": config,
                "consolidation_engine": engine,
                "consolidation_scheduler": None,
                "pressure_accumulator": None,
                "embedding_provider": None,
                "activation_store": None,
                "graph_store": None,
            }
        )

        engine.run_cycle = AsyncMock(return_value=None)

        await _shutdown()

        engine.run_cycle.assert_not_called()

        _app_state.clear()
