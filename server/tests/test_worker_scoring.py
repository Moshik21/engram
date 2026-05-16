from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.ingestion.worker_scoring import EpisodeWorkerScoringService


def _cfg(**overrides) -> ActivationConfig:
    defaults = {
        "triage_enabled": True,
        "triage_multi_signal_enabled": False,
        "worker_extract_threshold": 0.6,
        "worker_skip_threshold": 0.2,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


@pytest.mark.asyncio
async def test_worker_scoring_heuristic_routes_rich_content_to_extract() -> None:
    service = EpisodeWorkerScoringService(
        graph=SimpleNamespace(),
        activation=SimpleNamespace(),
        search=SimpleNamespace(),
        cfg=_cfg(triage_multi_signal_enabled=False, triage_min_score=0.2),
    )

    decision, signals = await service.score(
        "Alice and Bob discussed Python at Anthropic in San Francisco.",
        "brain",
    )

    assert decision.action == "extract"
    assert decision.decision_source == "heuristic"
    assert signals is None


@pytest.mark.asyncio
async def test_worker_scoring_multi_signal_uses_runtime_stores() -> None:
    graph = SimpleNamespace()
    activation = SimpleNamespace()
    search = SimpleNamespace()
    service = EpisodeWorkerScoringService(
        graph=graph,
        activation=activation,
        search=search,
        cfg=_cfg(
            triage_multi_signal_enabled=True,
            worker_extract_threshold=0.5,
            worker_skip_threshold=0.1,
        ),
    )
    scorer = MagicMock()
    signals = SimpleNamespace(composite=0.8)
    scorer.score = AsyncMock(return_value=signals)
    service._scorer = scorer

    decision, returned_signals = await service.score("Alice works at Anthropic", "brain")

    assert decision.action == "extract"
    assert decision.decision_source == "multi_signal"
    assert returned_signals is signals
    scorer.score.assert_awaited_once_with(
        content="Alice works at Anthropic",
        search_index=search,
        graph_store=graph,
        activation_store=activation,
        group_id="brain",
    )


@pytest.mark.asyncio
async def test_worker_scoring_records_projection_outcome_in_group() -> None:
    graph = SimpleNamespace(get_episode_entities=AsyncMock(return_value=["ent_worker"]))
    service = EpisodeWorkerScoringService(
        graph=graph,
        activation=SimpleNamespace(),
        search=SimpleNamespace(),
        cfg=_cfg(triage_multi_signal_enabled=True),
    )
    scorer = MagicMock()
    service._scorer = scorer
    signals = object()

    await service.record_projection_outcome("ep_worker", "worker_brain", signals)

    graph.get_episode_entities.assert_awaited_once_with(
        "ep_worker",
        group_id="worker_brain",
    )
    scorer.record_outcome.assert_called_once_with(signals, 1)
