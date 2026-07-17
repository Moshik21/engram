"""mat_tier/recon_count single source of truth: model fields + helix round-trip."""

from __future__ import annotations

import json

import pytest

from engram.config import HelixDBConfig
from engram.models.entity import Entity
from engram.storage.helix.graph import HelixGraphStore


class _FakeHelix:
    """In-memory fake for the native query transport."""

    def __init__(self) -> None:
        self.rows: dict[int, dict] = {}
        self._next_id = 1

    async def query(self, endpoint: str, payload: dict) -> list[dict]:
        if endpoint == "create_entity":
            hid = self._next_id
            self._next_id += 1
            row = dict(payload)
            row["id"] = hid
            self.rows[hid] = row
            return [row]
        if endpoint == "get_entity":
            row = self.rows.get(payload["id"])
            return [row] if row else []
        if endpoint == "find_entity_by_entity_id":
            for row in self.rows.values():
                if row["entity_id"] == payload["eid"] and row["group_id"] == payload["gid"]:
                    return [row]
            return []
        if endpoint == "update_entity_full":
            row = self.rows[payload["id"]]
            row.update({k: v for k, v in payload.items() if k != "id"})
            return [row]
        raise AssertionError(f"unexpected endpoint: {endpoint}")


@pytest.fixture
def fake_store(monkeypatch) -> tuple[HelixGraphStore, _FakeHelix]:
    store = HelixGraphStore(HelixDBConfig())
    fake = _FakeHelix()
    monkeypatch.setattr(store, "_query", fake.query)
    return store, fake


# ======================================================================
# Model fields
# ======================================================================


class TestEntityModelFields:
    def test_defaults(self) -> None:
        entity = Entity(id="e1", name="Alpha", entity_type="Concept")
        assert entity.mat_tier == "episodic"
        assert entity.recon_count == 0

    def test_attributes_fallback_for_old_rows(self) -> None:
        entity = Entity(
            id="e1",
            name="Alpha",
            entity_type="Concept",
            attributes={"mat_tier": "semantic", "recon_count": 3},
        )
        assert entity.mat_tier == "semantic"
        assert entity.recon_count == 3

    def test_field_authoritative_over_attributes(self) -> None:
        entity = Entity(
            id="e1",
            name="Alpha",
            entity_type="Concept",
            mat_tier="transitional",
            recon_count=2,
            attributes={"mat_tier": "semantic", "recon_count": 9},
        )
        assert entity.mat_tier == "transitional"
        assert entity.recon_count == 2

    def test_garbage_attribute_tier_ignored(self) -> None:
        entity = Entity(
            id="e1",
            name="Alpha",
            entity_type="Concept",
            attributes={"mat_tier": "bogus", "recon_count": "many"},
        )
        assert entity.mat_tier == "episodic"
        assert entity.recon_count == 0


# ======================================================================
# Helix native-lane round-trip (fake transport)
# ======================================================================


@pytest.mark.asyncio
class TestHelixMatTierRoundtrip:
    async def test_create_read_roundtrip(self, fake_store) -> None:
        store, fake = fake_store
        entity = Entity(
            id="ent_semantic",
            name="Konner",
            entity_type="Person",
            group_id="g1",
            mat_tier="semantic",
            recon_count=2,
        )
        await store.create_entity(entity)

        # Column (not attributes JSON) carries the tier natively.
        row = next(iter(fake.rows.values()))
        assert row["mat_tier"] == "semantic"
        assert row["recon_count"] == 2

        fetched = await store.get_entity("ent_semantic", "g1")
        assert fetched is not None
        assert fetched.mat_tier == "semantic"
        assert fetched.recon_count == 2

    async def test_update_preserves_tier_column(self, fake_store) -> None:
        store, _ = fake_store
        await store.create_entity(
            Entity(
                id="ent_keep",
                name="Alpha",
                entity_type="Concept",
                group_id="g1",
                mat_tier="transitional",
                recon_count=1,
            )
        )
        await store.update_entity("ent_keep", {"summary": "updated"}, group_id="g1")

        fetched = await store.get_entity("ent_keep", "g1")
        assert fetched is not None
        assert fetched.summary == "updated"
        assert fetched.mat_tier == "transitional"
        assert fetched.recon_count == 1

    async def test_old_row_tier_only_in_attributes_resolves(self, fake_store) -> None:
        store, fake = fake_store
        # Simulate a pre-fix row: column stuck at the 'episodic' default,
        # real tier only in attributes JSON.
        fake.rows[7] = {
            "id": 7,
            "entity_id": "ent_old",
            "name": "Legacy",
            "entity_type": "Concept",
            "group_id": "g1",
            "mat_tier": "episodic",
            "recon_count": 0,
            "attributes_json": json.dumps({"mat_tier": "semantic", "recon_count": 4}),
        }

        fetched = await store.get_entity("ent_old", "g1")
        assert fetched is not None
        assert fetched.mat_tier == "semantic"
        assert fetched.recon_count == 4


@pytest.mark.asyncio
async def test_identity_core_new_entity_promotes_to_semantic_tier():
    """M2.5 relocation: creating an identity-core-protected entity sets
    mat_tier='semantic' in the same write (the graduation path that survived
    the tier-system collapse)."""
    from unittest.mock import AsyncMock

    from engram.config import ActivationConfig
    from engram.extraction.apply import ApplyEngine
    from engram.extraction.canonicalize import PredicateCanonicalizer
    from engram.extraction.models import EntityCandidate
    from engram.models.episode import Episode

    graph = AsyncMock()
    graph.find_entity_candidates = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.link_episode_entity = AsyncMock()
    graph.update_entity = AsyncMock()

    engine = ApplyEngine(
        graph_store=graph,
        activation_store=AsyncMock(),
        cfg=ActivationConfig(identity_core_enabled=True),
        canonicalizer=PredicateCanonicalizer(),
    )
    episode = Episode(id="ep_idc", content="My name is Konner.", group_id="default")

    await engine.apply_entities(
        [
            EntityCandidate(
                name="Konner Moshier",
                entity_type="Person",
                raw_payload={"signals": ["client_proposal", "identity_pattern"]},
            )
        ],
        episode,
        "default",
    )

    created = graph.create_entity.call_args[0][0]
    assert created.identity_core
    assert created.mat_tier == "semantic"
