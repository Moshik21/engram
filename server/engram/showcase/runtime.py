"""Open a read-only lite GraphManager against a showcase database."""

from __future__ import annotations

from pathlib import Path

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.showcase.extractor import ShowcaseExtractor
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


async def open_showcase_manager(
    db_path: Path,
    *,
    group_id: str = "showcase",
) -> tuple[GraphManager, SQLiteGraphStore]:
    cfg = ActivationConfig()
    graph_store = SQLiteGraphStore(str(db_path))
    await graph_store.initialize()
    activation_store = MemoryActivationStore(cfg=cfg)
    search_index = FTS5SearchIndex(str(db_path))
    await search_index.initialize(db=graph_store._db)
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        ShowcaseExtractor(),
        cfg=cfg,
        runtime_mode="showcase_run",
    )
    # Showcase runs pin a group via runtime_mode metadata when available; fall
    # back to attributes bag without private GraphManager fields.
    attrs = getattr(manager, "runtime_attributes", None)
    if isinstance(attrs, dict):
        attrs["showcase_group_id"] = group_id
    else:
        object.__setattr__(manager, "showcase_group_id", group_id)
    return manager, graph_store
