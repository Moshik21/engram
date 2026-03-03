"""Integration tests for encryption with SQLite storage."""

import os
from pathlib import Path

import pytest
import pytest_asyncio

from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.security.encryption import FieldEncryptor
from engram.storage.sqlite.graph import SQLiteGraphStore


def _hex_key() -> str:
    return os.urandom(32).hex()


@pytest_asyncio.fixture
async def encrypted_graph_store(tmp_path: Path):
    key = _hex_key()
    encryptor = FieldEncryptor(key)
    store = SQLiteGraphStore(str(tmp_path / "encrypted.db"), encryptor=encryptor)
    await store.initialize()
    yield store, encryptor, key
    await store.close()


@pytest.mark.asyncio
class TestEncryptedStorage:
    async def test_encrypted_entity_summary(self, encrypted_graph_store):
        store, encryptor, key = encrypted_graph_store
        entity = Entity(
            id="ent_enc1",
            name="Secret Person",
            entity_type="Person",
            summary="Phone: 555-1234, SSN: 123-45-6789",
            group_id="default",
        )
        await store.create_entity(entity)

        # Read back — should be decrypted transparently
        result = await store.get_entity("ent_enc1", "default")
        assert result is not None
        assert result.summary == "Phone: 555-1234, SSN: 123-45-6789"

        # Check raw DB value is encrypted
        cursor = await store.db.execute(
            "SELECT summary FROM entities WHERE id = ?", ("ent_enc1",)
        )
        row = await cursor.fetchone()
        raw_summary = row[0]
        assert raw_summary.startswith("enc::")
        assert "555-1234" not in raw_summary

    async def test_encrypted_episode_content(self, encrypted_graph_store):
        store, encryptor, key = encrypted_graph_store
        episode = Episode(
            id="ep_enc1",
            content="Sensitive conversation about health records",
            group_id="default",
        )
        await store.create_episode(episode)

        # Read back — should be decrypted
        episodes = await store.get_episodes(group_id="default")
        assert len(episodes) == 1
        assert episodes[0].content == "Sensitive conversation about health records"

        # Check raw DB value is encrypted
        cursor = await store.db.execute(
            "SELECT content FROM episodes WHERE id = ?", ("ep_enc1",)
        )
        row = await cursor.fetchone()
        raw_content = row[0]
        assert raw_content.startswith("enc::")

    async def test_unencrypted_fallback(self, tmp_path: Path):
        """Store without encryptor should work normally with plaintext."""
        store = SQLiteGraphStore(str(tmp_path / "plain.db"))
        await store.initialize()

        entity = Entity(
            id="ent_plain",
            name="Plain",
            entity_type="Test",
            summary="Not encrypted",
            group_id="default",
        )
        await store.create_entity(entity)

        result = await store.get_entity("ent_plain", "default")
        assert result.summary == "Not encrypted"

        # Raw value should be plaintext
        cursor = await store.db.execute(
            "SELECT summary FROM entities WHERE id = ?", ("ent_plain",)
        )
        row = await cursor.fetchone()
        assert row[0] == "Not encrypted"

        await store.close()

    async def test_pii_flags_stored_and_retrieved(self, encrypted_graph_store):
        store, _, _ = encrypted_graph_store
        entity = Entity(
            id="ent_pii1",
            name="Jake",
            entity_type="Person",
            summary="Has PII",
            group_id="default",
            pii_detected=True,
            pii_categories=["phone", "email", "health"],
        )
        await store.create_entity(entity)

        result = await store.get_entity("ent_pii1", "default")
        assert result is not None
        assert result.pii_detected is True
        assert set(result.pii_categories) == {"phone", "email", "health"}

    async def test_confidence_stored_and_retrieved(self, encrypted_graph_store):
        store, _, _ = encrypted_graph_store
        await store.create_entity(
            Entity(id="ent_c1", name="A", entity_type="Test", group_id="default")
        )
        await store.create_entity(
            Entity(id="ent_c2", name="B", entity_type="Test", group_id="default")
        )

        from engram.models.relationship import Relationship

        rel = Relationship(
            id="rel_conf1",
            source_id="ent_c1",
            target_id="ent_c2",
            predicate="KNOWS",
            confidence=0.7,
            group_id="default",
        )
        await store.create_relationship(rel)

        rels = await store.get_relationships("ent_c1")
        assert len(rels) == 1
        assert rels[0].confidence == 0.7
