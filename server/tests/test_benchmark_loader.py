from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.benchmark_loader import BenchmarkLoadService, build_api_benchmark_load_surface


@pytest.mark.asyncio
async def test_api_benchmark_load_surface_forwards_route_options() -> None:
    manager = SimpleNamespace(
        load_benchmark_corpus=AsyncMock(return_value={"loaded": True, "seed": 11})
    )

    result = await build_api_benchmark_load_surface(
        manager,
        group_id="native_brain",
        seed=11,
        structure_aware=True,
    )

    assert result == {"loaded": True, "seed": 11}
    manager.load_benchmark_corpus.assert_awaited_once_with(
        group_id="native_brain",
        seed=11,
        structure_aware=True,
    )


@pytest.mark.asyncio
async def test_benchmark_load_service_scopes_corpus_to_active_group(monkeypatch) -> None:
    class FakeCorpusGenerator:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def generate(self):
            return SimpleNamespace(
                entities=[SimpleNamespace(group_id="stale")],
                relationships=[SimpleNamespace(group_id="stale")],
                access_events=[object(), object()],
                ground_truth=[object()],
            )

        async def load(
            self,
            corpus,
            graph_store,
            activation_store,
            search_index,
            *,
            structure_aware: bool = False,
        ) -> float:
            assert self.seed == 7
            assert graph_store == "graph"
            assert activation_store == "activation"
            assert search_index == "search"
            assert structure_aware is True
            assert {entity.group_id for entity in corpus.entities} == {"brain"}
            assert {rel.group_id for rel in corpus.relationships} == {"brain"}
            return 0.424

    monkeypatch.setattr("engram.benchmark.corpus.CorpusGenerator", FakeCorpusGenerator)
    service = BenchmarkLoadService(
        graph_store="graph",
        activation_store="activation",
        search_index="search",
    )

    result = await service.load_benchmark(
        group_id="brain",
        seed=7,
        structure_aware=True,
    )

    assert result == {
        "loaded": True,
        "seed": 7,
        "group_id": "brain",
        "entities": 1,
        "relationships": 1,
        "access_events": 2,
        "queries": 1,
        "elapsed_seconds": 0.42,
        "structure_aware": True,
    }
