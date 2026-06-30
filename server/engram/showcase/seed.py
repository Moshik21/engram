"""Seed a lite SQLite demo database for the public showcase."""

from __future__ import annotations

import os
from pathlib import Path

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.showcase.beats import SHOWCASE_SEED_EPISODES
from engram.showcase.extractor import ShowcaseExtractor
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


async def seed_demo_db(
    output_path: Path,
    *,
    group_id: str = "showcase",
    overwrite: bool = True,
) -> Path:
    """Populate a lite sqlite database with fixed showcase episodes."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{output_path}{suffix}")
        if overwrite and candidate.exists():
            candidate.unlink()

    cfg = ActivationConfig()
    graph_store = SQLiteGraphStore(str(output_path))
    await graph_store.initialize()
    activation_store = MemoryActivationStore(cfg=cfg)
    search_index = FTS5SearchIndex(str(output_path))
    await search_index.initialize(db=graph_store._db)

    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        ShowcaseExtractor(),
        cfg=cfg,
        runtime_mode="showcase_seed",
    )

    seeded: list[dict[str, str]] = []
    for episode in SHOWCASE_SEED_EPISODES:
        episode_id = await manager.ingest_episode(
            episode.content,
            group_id=group_id,
            source=episode.source,
        )
        seeded.append(
            {
                "label": episode.label,
                "episode_id": episode_id,
                "source": episode.source,
            }
        )

    liam_entities = await graph_store.find_entities(name="Liam", group_id=group_id)
    if liam_entities:
        await graph_store.update_entity(
            liam_entities[0].id,
            {"identity_core": True, "summary": liam_entities[0].summary},
            group_id,
        )

    await graph_store.close()
    return output_path


def default_seed_output() -> Path:
    env_override = os.environ.get("ENGRAM_SHOWCASE_DEMO_DB")
    if env_override:
        return Path(env_override).expanduser()
    return Path(__file__).resolve().parent.parent / "data" / "demo.db"