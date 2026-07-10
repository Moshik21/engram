"""Tests for encoding_context Episode field and infrastructure."""

from __future__ import annotations

import json
import sqlite3

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


@pytest.mark.asyncio
async def test_sqlite_graph_initialize_migrates_legacy_entity_table_before_indexes(
    tmp_path,
):
    """Existing lite DBs missing entity facets should start cleanly."""
    from engram.storage.sqlite.graph import SQLiteGraphStore

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            summary TEXT,
            attributes TEXT,
            group_id TEXT NOT NULL DEFAULT 'default',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT,
            activation_base REAL NOT NULL DEFAULT 0.5,
            activation_current REAL NOT NULL DEFAULT 0.5,
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            group_id TEXT NOT NULL DEFAULT 'default',
            session_id TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE relationships (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            valid_from TEXT,
            valid_to TEXT,
            created_at TEXT NOT NULL,
            source_episode TEXT,
            group_id TEXT NOT NULL DEFAULT 'default'
        )
        """
    )
    conn.commit()
    conn.close()

    store = SQLiteGraphStore(str(db_path))
    await store.initialize()
    try:
        entity_cursor = await store.db.execute("PRAGMA table_info(entities)")
        entity_columns = {row[1] for row in await entity_cursor.fetchall()}
        relationship_cursor = await store.db.execute("PRAGMA table_info(relationships)")
        relationship_columns = {row[1] for row in await relationship_cursor.fetchall()}
        index_cursor = await store.db.execute("PRAGMA index_list(entities)")
        indexes = {row[1] for row in await index_cursor.fetchall()}
    finally:
        await store.close()

    assert "lexical_regime" in entity_columns
    assert "canonical_identifier" in entity_columns
    assert "identifier_label" in entity_columns
    assert "source_episode_ids" in entity_columns
    assert "evidence_count" in entity_columns
    assert "evidence_span_start" in entity_columns
    assert "evidence_span_end" in entity_columns
    assert "polarity" in relationship_columns
    assert "idx_entities_lexical_regime" in indexes
