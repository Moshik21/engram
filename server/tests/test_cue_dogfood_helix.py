"""Helix parity for cue dogfood — skipped when HelixDB is unavailable."""

from __future__ import annotations

import pytest

from tests.conftest import _helix_available


@pytest.mark.requires_helix
@pytest.mark.asyncio
async def test_helix_observe_recall_advances_cue_metrics() -> None:
    """Mirror SQLite dogfood on Helix when a native instance is running."""
    if not _helix_available():
        pytest.skip("HelixDB not available on localhost:6969")

    from engram.config import EmbeddingConfig, HelixDBConfig
    from engram.embeddings.provider import NoopProvider
    from engram.graph_manager import GraphManager
    from engram.mcp.server import SessionState
    from engram.retrieval.recall_surface import build_mcp_explicit_recall_tool_surface
    from engram.storage.helix.graph import HelixGraphStore
    from engram.storage.helix.search import HelixSearchIndex
    from engram.storage.memory.activation import MemoryActivationStore
    from tests.conftest import MockExtractor
    from tests.test_consolidation_profiles import _quiet_sqlite_recall_config

    cfg = _quiet_sqlite_recall_config()
    helix_config = HelixDBConfig(host="localhost", port=6969)
    graph_store = HelixGraphStore(helix_config)
    await graph_store.initialize()
    search_index = HelixSearchIndex(
        helix_config=helix_config,
        provider=NoopProvider(),
        embed_config=EmbeddingConfig(),
        storage_dim=0,
        embed_provider="noop",
        embed_model="noop",
    )
    await search_index.initialize()
    activation_store = MemoryActivationStore(cfg=cfg)
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        MockExtractor(),
        cfg=cfg,
    )
    group_id = "cue_dogfood_helix"

    episode_id = await manager.store_episode(
        "The migration to native Helix finished on Tuesday.",
        group_id=group_id,
        source="mcp_observe",
    )
    cue = await graph_store.get_episode_cue(episode_id, group_id)
    assert cue is not None

    session = SessionState(group_id=group_id)

    async def noop_middleware(*_args, **_kwargs) -> None:
        return None

    await build_mcp_explicit_recall_tool_surface(
        manager,
        group_id=group_id,
        query="native Helix migration",
        limit=5,
        cfg=cfg,
        session=session,
        recall_middleware=noop_middleware,
    )
    await build_mcp_explicit_recall_tool_surface(
        manager,
        group_id=group_id,
        query="native Helix migration",
        limit=5,
        cfg=cfg,
        session=session,
        recall_middleware=noop_middleware,
    )

    stats = await graph_store.get_stats(group_id=group_id)
    cue_metrics = stats.get("cue_metrics") or {}
    assert int(cue_metrics.get("cue_hit_count") or 0) > 0

    await search_index.close()
    await graph_store.close()
