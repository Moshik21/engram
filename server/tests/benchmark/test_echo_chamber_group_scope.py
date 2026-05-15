"""Echo chamber benchmark group-scope contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from engram.benchmark.echo_chamber import (
    ECHO_CHAMBER_BENCHMARK_GROUP_ID,
    run_echo_chamber,
)
from engram.config import ActivationConfig


@dataclass
class _FakeRetrievalResult:
    node_id: str
    result_type: str = "entity"


@dataclass
class _FakeActivationState:
    access_count: int = 0


class _FakeActivationStore:
    def __init__(self) -> None:
        self.states: dict[str, _FakeActivationState] = {}
        self.recorded_group_ids: list[str] = []

    async def record_access(self, entity_id: str, _timestamp: float, *, group_id: str) -> None:
        self.recorded_group_ids.append(group_id)
        state = self.states.setdefault(entity_id, _FakeActivationState())
        state.access_count += 1

    async def batch_get(self, entity_ids: list[str]) -> dict[str, _FakeActivationState]:
        return {
            entity_id: self.states[entity_id]
            for entity_id in entity_ids
            if entity_id in self.states
        }


@pytest.mark.asyncio
async def test_echo_chamber_defaults_to_benchmark_group(monkeypatch) -> None:
    from engram.benchmark import echo_chamber

    retrieve_group_ids: list[str] = []

    async def fake_retrieve(**kwargs):
        retrieve_group_ids.append(kwargs["group_id"])
        return [_FakeRetrievalResult("ent_alpha")]

    monkeypatch.setattr(echo_chamber, "retrieve", fake_retrieve)
    activation_store = _FakeActivationStore()

    result = await run_echo_chamber(
        hot_queries=["alpha"],
        diverse_queries=["beta"],
        corpus_entity_ids=["ent_alpha", "ent_beta"],
        graph_store=object(),
        activation_store=activation_store,
        search_index=object(),
        cfg=ActivationConfig(),
        total_queries=2,
        hot_ratio=1.0,
        snapshot_interval=1,
    )

    assert result.total_queries == 2
    assert retrieve_group_ids == [
        ECHO_CHAMBER_BENCHMARK_GROUP_ID,
        ECHO_CHAMBER_BENCHMARK_GROUP_ID,
    ]
    assert activation_store.recorded_group_ids == [
        ECHO_CHAMBER_BENCHMARK_GROUP_ID,
        ECHO_CHAMBER_BENCHMARK_GROUP_ID,
    ]
