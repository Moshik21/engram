"""LongMemEval adapter contract tests."""

from __future__ import annotations

import pytest

from engram.benchmark.longmemeval.adapter import EngramLongMemEvalAdapter


class _FakeClosable:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self._closed = closed

    async def close(self) -> None:
        self._closed.append(self.name)


@pytest.mark.asyncio
async def test_longmemeval_adapter_closes_runtime_store_triple() -> None:
    closed: list[str] = []
    adapter = EngramLongMemEvalAdapter()
    adapter._search_index = _FakeClosable("search", closed)
    adapter._activation_store = _FakeClosable("activation", closed)
    adapter._graph_store = _FakeClosable("graph", closed)
    adapter._initialized = True

    await adapter.close()

    assert closed == ["search", "activation", "graph"]
    assert adapter._search_index is None
    assert adapter._activation_store is None
    assert adapter._graph_store is None
    assert adapter._manager is None
    assert adapter._initialized is False
