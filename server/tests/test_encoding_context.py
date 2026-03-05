"""Tests for encoding_context Episode field and infrastructure."""

from __future__ import annotations

import json

import pytest

from engram.models.episode import Episode, EpisodeStatus
from engram.storage.protocols import EPISODE_UPDATABLE_FIELDS


def test_episode_model_accepts_encoding_context():
    ep = Episode(
        id="ep_test",
        content="hello",
        encoding_context=json.dumps({"arousal": 0.5}),
    )
    assert ep.encoding_context == json.dumps({"arousal": 0.5})


def test_episode_model_defaults_encoding_context_none():
    ep = Episode(id="ep_test", content="hello")
    assert ep.encoding_context is None


def test_updatable_fields_contains_encoding_context():
    assert "encoding_context" in EPISODE_UPDATABLE_FIELDS


@pytest.mark.asyncio
async def test_sqlite_create_episode_with_encoding_context(tmp_path):
    from engram.storage.sqlite.graph import SQLiteGraphStore

    db_path = str(tmp_path / "test.db")
    store = SQLiteGraphStore(db_path)
    await store.initialize()

    ep = Episode(
        id="ep_enc1",
        content="test content",
        status=EpisodeStatus.QUEUED,
        group_id="default",
        encoding_context=json.dumps({"arousal": 0.7, "self_ref": 0.3}),
    )
    await store.create_episode(ep)

    fetched = await store.get_episode_by_id("ep_enc1", "default")
    assert fetched is not None
    assert fetched.encoding_context == json.dumps({"arousal": 0.7, "self_ref": 0.3})
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_update_episode_encoding_context(tmp_path):
    from engram.storage.sqlite.graph import SQLiteGraphStore

    db_path = str(tmp_path / "test.db")
    store = SQLiteGraphStore(db_path)
    await store.initialize()

    ep = Episode(
        id="ep_enc2",
        content="test content",
        status=EpisodeStatus.QUEUED,
        group_id="default",
    )
    await store.create_episode(ep)

    await store.update_episode(
        "ep_enc2",
        {"encoding_context": json.dumps({"composite": 0.8})},
        group_id="default",
    )
    fetched = await store.get_episode_by_id("ep_enc2", "default")
    assert fetched is not None
    ctx = json.loads(fetched.encoding_context)
    assert ctx["composite"] == 0.8
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_create_episode_without_encoding_context(tmp_path):
    from engram.storage.sqlite.graph import SQLiteGraphStore

    db_path = str(tmp_path / "test.db")
    store = SQLiteGraphStore(db_path)
    await store.initialize()

    ep = Episode(
        id="ep_enc3",
        content="no context",
        status=EpisodeStatus.QUEUED,
        group_id="default",
    )
    await store.create_episode(ep)

    fetched = await store.get_episode_by_id("ep_enc3", "default")
    assert fetched is not None
    assert fetched.encoding_context is None
    await store.close()


@pytest.mark.asyncio
async def test_migration_idempotent(tmp_path):
    """ALTER TABLE migration should not fail if column already exists."""
    from engram.consolidation.store import SQLiteConsolidationStore

    db_path = str(tmp_path / "consol.db")
    store = SQLiteConsolidationStore(db_path)
    await store.initialize()
    # Second init should not raise
    await store.initialize()
    await store.close()
