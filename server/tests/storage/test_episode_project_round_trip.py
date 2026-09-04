"""The project field survives a store round trip (SQLite; the Helix path shares the codec)."""

from __future__ import annotations

import pytest

from engram.models.episode import Episode
from engram.storage.sqlite.graph import SQLiteGraphStore

pytestmark = pytest.mark.asyncio


async def test_project_round_trips_through_sqlite() -> None:
    store = SQLiteGraphStore(":memory:")
    await store.initialize()
    await store.create_episode(Episode(id="ep_p", content="x", project="server"))
    await store.create_episode(Episode(id="ep_none", content="y"))
    assert (await store.get_episode_by_id("ep_p", "default")).project == "server"
    assert (await store.get_episode_by_id("ep_none", "default")).project is None
