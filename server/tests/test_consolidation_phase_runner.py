from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phase_runner import ConsolidationPhaseRunner
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import (
    CycleContext,
    DecisionOutcomeLabel,
    DecisionTrace,
    MergeRecord,
    PhaseResult,
    PruneRecord,
)


class RecordingPhase(ConsolidationPhase):
    @property
    def name(self) -> str:
        return "recording"

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
            candidate_id="keep:drop",
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
        )
        context.add_decision_trace(trace)
        context.add_decision_outcome_label(label)
        return PhaseResult(phase=self.name, items_processed=2, items_affected=2), [
            MergeRecord(
                cycle_id=cycle_id,
                group_id=group_id,
                keep_id="keep",
                remove_id="drop",
                keep_name="Keep",
                remove_name="Drop",
                similarity=0.95,
            ),
            PruneRecord(
                cycle_id=cycle_id,
                group_id=group_id,
                entity_id="stale",
                entity_name="Stale",
                entity_type="concept",
                reason="dead_entity",
            ),
        ]


@pytest.mark.asyncio
async def test_phase_runner_persists_records_and_new_context_decisions_only():
    store = SimpleNamespace(
        save_merge_record=AsyncMock(),
        save_prune_record=AsyncMock(),
        save_decision_trace=AsyncMock(),
        save_decision_outcome_label=AsyncMock(),
    )
    context = CycleContext(trigger="manual")
    context.add_decision_trace(
        DecisionTrace(
            cycle_id="cyc_existing",
            group_id="test",
            phase="existing",
            candidate_type="entity",
            candidate_id="old",
            decision="skip",
            decision_source="seed",
        )
    )

    runner = ConsolidationPhaseRunner(
        graph_store=object(),
        activation_store=object(),
        search_index=object(),
        cfg=ActivationConfig(),
        consolidation_store=store,
    )

    outcome = await runner.run_phase(
        RecordingPhase(),
        group_id="test",
        cycle_id="cyc_test",
        dry_run=False,
        context=context,
    )

    assert outcome.result.phase == "recording"
    assert outcome.result.items_affected == 2
    assert len(outcome.records) == 2
    assert outcome.removed_node_ids == ("drop", "stale")
    store.save_merge_record.assert_awaited_once()
    store.save_prune_record.assert_awaited_once()
    store.save_decision_trace.assert_awaited_once()
    store.save_decision_outcome_label.assert_awaited_once()
    persisted_trace = store.save_decision_trace.await_args.args[0]
    assert persisted_trace.cycle_id == "cyc_test"
    assert persisted_trace.decision_source == "unit_test"


@pytest.mark.asyncio
async def test_phase_runner_without_store_returns_removed_nodes():
    runner = ConsolidationPhaseRunner(
        graph_store=object(),
        activation_store=object(),
        search_index=object(),
        cfg=ActivationConfig(),
        consolidation_store=None,
    )

    outcome = await runner.run_phase(
        RecordingPhase(),
        group_id="test",
        cycle_id="cyc_test",
        dry_run=True,
        context=CycleContext(trigger="manual"),
    )

    assert outcome.removed_node_ids == ("drop", "stale")
