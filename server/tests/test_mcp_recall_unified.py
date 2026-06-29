"""MCP recall-unified retrieval and deprecated search compat shims."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.mcp.server import SessionState
from engram.retrieval.lookup import (
    MCP_SEARCH_ENTITIES_DEPRECATION,
    build_mcp_entity_search_tool_surface,
)
from engram.retrieval.recall_surface import build_mcp_explicit_recall_tool_surface
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex
from tests.conftest import MockExtractor
from tests.test_consolidation_profiles import _quiet_sqlite_recall_config


@pytest.mark.asyncio
async def test_entity_search_compat_primes_recall_path() -> None:
    manager = MagicMock()
    manager.search_entities = AsyncMock(return_value=[])
    recall_middleware = AsyncMock()
    cfg = ActivationConfig()

    with patch(
        "engram.retrieval.recall_surface.build_mcp_recall_surface",
        new_callable=AsyncMock,
        return_value={"results": [], "total": 0},
    ) as recall_surface:
        payload = await build_mcp_entity_search_tool_surface(
            manager,
            group_id="default",
            name="Helix",
            limit=3,
            recall_middleware=recall_middleware,
            cfg=cfg,
        )

    recall_surface.assert_awaited_once()
    assert payload["preferRecall"] is True
    assert payload["deprecationNotice"] == MCP_SEARCH_ENTITIES_DEPRECATION
    assert payload["recallCompanion"]["query"] == "Helix"
    recall_middleware.assert_awaited_once()


@pytest.mark.asyncio
async def test_mcp_explicit_recall_surface_runs_real_recall_and_middleware(
    tmp_path,
) -> None:
    """Drive build_mcp_explicit_recall_tool_surface on SQLite data without mocks."""
    cfg = _quiet_sqlite_recall_config()
    graph_store = SQLiteGraphStore(str(tmp_path / "mcp_recall_e2e.db"))
    await graph_store.initialize()
    search_index = FTS5SearchIndex(graph_store._db_path)
    await search_index.initialize(db=graph_store._db)
    activation_store = MemoryActivationStore(cfg=cfg)
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        MockExtractor(),
        cfg=cfg,
    )

    episode_id = await manager.store_episode(
        "Spreading activation strengthens graph pathways over time.",
        group_id="default",
        source="mcp_observe",
    )
    assert episode_id.startswith("ep_")

    middleware_calls: list[str | None] = []

    async def capture_middleware(
        query: str,
        result: dict,
        *,
        tool_name: str | None = None,
    ) -> None:
        middleware_calls.append(tool_name)
        assert result.get("operation") == "recall"

    session = SessionState(group_id="default")
    response = await build_mcp_explicit_recall_tool_surface(
        manager,
        group_id="default",
        query="spreading activation graph",
        limit=5,
        cfg=cfg,
        session=session,
        recall_middleware=capture_middleware,
    )

    assert response["operation"] == "recall"
    assert response["query"] == "spreading activation graph"
    assert middleware_calls == ["recall"]
    assert session.last_recall_time > 0
    await graph_store.close()


@pytest.mark.asyncio
async def test_mcp_recall_tool_entrypoint_on_sqlite(tmp_path, monkeypatch) -> None:
    """Full mcp_server.recall() entry on a real lite brain."""
    from engram.mcp import server as mcp_server

    cfg = _quiet_sqlite_recall_config()
    graph_store = SQLiteGraphStore(str(tmp_path / "mcp_tool_recall.db"))
    await graph_store.initialize()
    search_index = FTS5SearchIndex(graph_store._db_path)
    await search_index.initialize(db=graph_store._db)
    activation_store = MemoryActivationStore(cfg=cfg)
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        MockExtractor(),
        cfg=cfg,
    )
    await manager.store_episode(
        "Engram recall routing uses activation-aware graph search.",
        group_id="default",
        source="mcp_observe",
    )

    monkeypatch.setattr(mcp_server, "_manager", manager)
    monkeypatch.setattr(mcp_server, "_session", SessionState(group_id="default"))
    monkeypatch.setattr(mcp_server, "_group_id", "default")
    monkeypatch.setattr(mcp_server, "_activation_cfg", cfg)

    raw = await mcp_server.recall("Engram recall routing", limit=5)
    payload = json.loads(raw)
    assert payload["operation"] == "recall"
    assert payload["lifecycle"]["stage"] == "recall"
    assert payload["lifecycle"]["recall_mode"] == "explicit"
    await graph_store.close()
