from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.learning import ConsolidationLearningService
from engram.models.consolidation import (
    ConsolidationCycle,
    CycleContext,
    DecisionOutcomeLabel,
    DecisionTrace,
)


def _trace(
    *,
    cycle_id: str = "cyc_test",
    group_id: str = "test",
    phase: str = "merge",
    trace_id_suffix: str = "",
) -> DecisionTrace:
    return DecisionTrace(
        cycle_id=cycle_id,
        group_id=group_id,
        phase=phase,
        candidate_type="entity_pair",
        candidate_id=f"a:b{trace_id_suffix}",
        decision="merge",
        decision_source="llm",
        confidence=0.84,
        threshold_band="accepted",
        features={"name_similarity": 0.92},
    )


def _label(trace: DecisionTrace) -> DecisionOutcomeLabel:
    return DecisionOutcomeLabel(
        cycle_id=trace.cycle_id,
        group_id=trace.group_id,
        phase=trace.phase,
        decision_trace_id=trace.id,
        outcome_type="materialization",
        label="applied",
        value=1.0,
    )


@pytest.mark.asyncio
async def test_learning_service_persists_current_cycle_artifacts():
    cycle = ConsolidationCycle(group_id="test", id="cyc_test")
    trace = _trace()
    label = _label(trace)
    context = CycleContext(trigger="manual")
    context.add_decision_trace(trace)
    context.add_decision_outcome_label(label)
    store = SimpleNamespace(
        save_distillation_example=AsyncMock(),
        get_recent_cycles=AsyncMock(return_value=[cycle]),
        get_decision_traces=AsyncMock(),
        get_decision_outcome_labels=AsyncMock(),
        save_calibration_snapshot=AsyncMock(),
    )
    service = ConsolidationLearningService(
        cfg=ActivationConfig(
            consolidation_distillation_enabled=True,
            consolidation_calibration_enabled=True,
            consolidation_calibration_window_cycles=5,
            consolidation_calibration_min_examples=1,
            consolidation_calibration_bins=2,
        ),
        consolidation_store=store,
    )

    result = await service.analyze_cycle(cycle, context)

    assert result.updated is True
    assert result.distillation_examples == 2
    assert result.calibration_snapshots == 1
    assert store.save_distillation_example.await_count == 2
    store.get_recent_cycles.assert_awaited_once_with("test", limit=5)
    store.get_decision_traces.assert_not_awaited()
    store.get_decision_outcome_labels.assert_not_awaited()
    store.save_calibration_snapshot.assert_awaited_once()
    snapshot = store.save_calibration_snapshot.await_args.args[0]
    assert snapshot.phase == "merge"
    assert snapshot.summary["requested_window_cycles"] == 5
    assert snapshot.summary["cycles_observed"] == 1


@pytest.mark.asyncio
async def test_learning_service_collects_prior_cycle_calibration_history():
    current_cycle = ConsolidationCycle(group_id="test", id="cyc_current")
    prior_cycle = ConsolidationCycle(group_id="test", id="cyc_prior")
    prior_trace = _trace(cycle_id=prior_cycle.id, trace_id_suffix="-prior")
    prior_label = _label(prior_trace)
    store = SimpleNamespace(
        save_distillation_example=AsyncMock(),
        get_recent_cycles=AsyncMock(return_value=[prior_cycle]),
        get_decision_traces=AsyncMock(return_value=[prior_trace]),
        get_decision_outcome_labels=AsyncMock(return_value=[prior_label]),
        save_calibration_snapshot=AsyncMock(),
    )
    service = ConsolidationLearningService(
        cfg=ActivationConfig(
            consolidation_distillation_enabled=False,
            consolidation_calibration_enabled=True,
            consolidation_calibration_window_cycles=3,
            consolidation_calibration_min_examples=1,
        ),
        consolidation_store=store,
    )

    result = await service.analyze_cycle(
        current_cycle,
        CycleContext(trigger="manual"),
    )

    assert result.distillation_examples == 0
    assert result.calibration_snapshots == 1
    store.get_decision_traces.assert_awaited_once_with(prior_cycle.id, "test")
    store.get_decision_outcome_labels.assert_awaited_once_with(prior_cycle.id, "test")
    snapshot = store.save_calibration_snapshot.await_args.args[0]
    assert snapshot.cycle_id == current_cycle.id
    assert snapshot.phase == prior_trace.phase
    assert snapshot.summary["cycles_observed"] == 1


@pytest.mark.asyncio
async def test_learning_service_without_store_noops():
    service = ConsolidationLearningService(
        cfg=ActivationConfig(
            consolidation_distillation_enabled=True,
            consolidation_calibration_enabled=True,
        ),
        consolidation_store=None,
    )

    result = await service.analyze_cycle(
        ConsolidationCycle(group_id="test", id="cyc_test"),
        CycleContext(trigger="manual"),
    )

    assert result.updated is False
    assert result.distillation_examples == 0
    assert result.calibration_snapshots == 0
