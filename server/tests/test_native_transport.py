from __future__ import annotations

import logging

import pytest

from engram.storage.helix.native_transport import NativeTransport


class _FailingEngine:
    def __init__(self, message: str) -> None:
        self._message = message

    def query(self, endpoint: str, body_json: str) -> str:
        raise RuntimeError(self._message)


@pytest.mark.asyncio
async def test_native_transport_treats_missing_hnsw_index_as_empty(caplog) -> None:
    transport = object.__new__(NativeTransport)
    transport._engine = _FailingEngine(
        "Query 'search_graph_embed_vectors' failed: "
        "Vector error: no entry point found for hnsw index"
    )
    transport._executor = None

    with caplog.at_level(logging.ERROR, logger="engram.storage.helix.native_transport"):
        rows = await transport.query("search_graph_embed_vectors", {"vec": [], "k": 10})

    assert rows == []
    assert "search_graph_embed_vectors failed" not in caplog.text
