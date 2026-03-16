"""GraphStore contract mixin -- backend-agnostic tests for protocol conformance.

Any backend's test class can inherit from ``GraphStoreContractTests`` and
provide two abstract properties (``store`` and ``group_id``) to get the full
suite of contract assertions for free.

Example usage::

    @pytest.mark.asyncio
    class TestMyBackend(GraphStoreContractTests):
        @property
        def store(self):
            return self._store

        @property
        def group_id(self):
            return self._group_id

        @pytest.fixture(autouse=True)
        def _inject(self, my_store_fixture, my_group_fixture):
            self._store = my_store_fixture
            self._group_id = my_group_fixture
"""

from __future__ import annotations

import abc
from uuid import uuid4

from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship


def _uid() -> str:
    return uuid4().hex[:12]


class GraphStoreContractTests(abc.ABC):
    """Mixin that verifies the core GraphStore protocol contract.

    Subclasses must provide ``store`` and ``group_id`` via fixtures or
    properties so that each test method can interact with an initialised
    store instance.
    """

    @property
    @abc.abstractmethod
    def store(self):  # noqa: ANN201
        """Return an initialised GraphStore implementation."""
        ...

    @property
    @abc.abstractmethod
    def group_id(self) -> str:
        """Return the tenant group_id to use for this test run."""
        ...

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    async def test_create_and_get_entity(self):
        eid = f"ent_{_uid()}"
        entity = Entity(
            id=eid,
            name="Python",
            entity_type="Technology",
            summary="Programming language",
            group_id=self.group_id,
        )
        result_id = await self.store.create_entity(entity)
        assert result_id == eid

        fetched = await self.store.get_entity(eid, self.group_id)
        assert fetched is not None
        assert fetched.name == "Python"
        assert fetched.entity_type == "Technology"
        assert fetched.summary == "Programming language"

    async def test_entity_not_found_returns_none(self):
        result = await self.store.get_entity(f"ent_{_uid()}", self.group_id)
        assert result is None

    async def test_tenant_isolation(self):
        eid = f"ent_{_uid()}"
        entity = Entity(
            id=eid,
            name="Secret",
            entity_type="Concept",
            group_id=self.group_id,
        )
        await self.store.create_entity(entity)

        # Different group must NOT see the entity
        result = await self.store.get_entity(eid, "other_group_xyz")
        assert result is None

        # Correct group should see it
        result = await self.store.get_entity(eid, self.group_id)
        assert result is not None

    async def test_update_entity(self):
        eid = f"ent_{_uid()}"
        await self.store.create_entity(
            Entity(id=eid, name="OldName", entity_type="Person", group_id=self.group_id)
        )
        await self.store.update_entity(eid, {"name": "NewName"}, group_id=self.group_id)

        result = await self.store.get_entity(eid, self.group_id)
        assert result is not None
        assert result.name == "NewName"

    async def test_soft_delete_entity(self):
        eid = f"ent_{_uid()}"
        await self.store.create_entity(
            Entity(id=eid, name="ToDelete", entity_type="Test", group_id=self.group_id)
        )
        await self.store.delete_entity(eid, soft=True, group_id=self.group_id)

        result = await self.store.get_entity(eid, self.group_id)
        assert result is None  # Soft-deleted entities are hidden

    # ------------------------------------------------------------------
    # Relationship CRUD
    # ------------------------------------------------------------------

    async def _create_entity_pair(self) -> tuple[str, str]:
        """Helper: create source + target entities, return their IDs."""
        src = f"ent_src_{_uid()}"
        tgt = f"ent_tgt_{_uid()}"
        await self.store.create_entity(
            Entity(id=src, name="Source", entity_type="Test", group_id=self.group_id)
        )
        await self.store.create_entity(
            Entity(id=tgt, name="Target", entity_type="Test", group_id=self.group_id)
        )
        return src, tgt

    async def test_create_and_get_relationship(self):
        src, tgt = await self._create_entity_pair()
        rel = Relationship(
            id=f"rel_{_uid()}",
            source_id=src,
            target_id=tgt,
            predicate="CONNECTS",
            group_id=self.group_id,
        )
        await self.store.create_relationship(rel)

        rels = await self.store.get_relationships(
            src, direction="outgoing", group_id=self.group_id
        )
        assert len(rels) >= 1
        assert any(r.predicate == "CONNECTS" for r in rels)

    # ------------------------------------------------------------------
    # Episode CRUD
    # ------------------------------------------------------------------

    async def test_create_and_list_episodes(self):
        ep_id = f"ep_{_uid()}"
        ep = Episode(
            id=ep_id,
            content="Test episode content for contract",
            group_id=self.group_id,
        )
        result_id = await self.store.create_episode(ep)
        assert result_id == ep_id

        episodes = await self.store.get_episodes(group_id=self.group_id)
        assert len(episodes) >= 1
        assert any(e.id == ep_id for e in episodes)

    # ------------------------------------------------------------------
    # Episode-Entity linking
    # ------------------------------------------------------------------

    async def test_link_episode_entity(self):
        ent_id = f"ent_{_uid()}"
        ep_id = f"ep_{_uid()}"
        await self.store.create_entity(
            Entity(id=ent_id, name="Linked", entity_type="Test", group_id=self.group_id)
        )
        await self.store.create_episode(
            Episode(id=ep_id, content="Episode with link", group_id=self.group_id)
        )
        await self.store.link_episode_entity(ep_id, ent_id)

        entity_ids = await self.store.get_episode_entities(ep_id)
        assert ent_id in entity_ids

    # ------------------------------------------------------------------
    # Find entity candidates (text search)
    # ------------------------------------------------------------------

    async def test_find_entity_candidates(self):
        eid = f"ent_{_uid()}"
        await self.store.create_entity(
            Entity(
                id=eid,
                name="FastAPI Framework",
                entity_type="Technology",
                group_id=self.group_id,
            )
        )
        candidates = await self.store.find_entity_candidates(
            "FastAPI Framework", self.group_id
        )
        candidate_ids = [c.id for c in candidates]
        assert eid in candidate_ids

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def test_get_stats(self):
        await self.store.create_entity(
            Entity(
                id=f"ent_{_uid()}",
                name="StatEntity",
                entity_type="Test",
                group_id=self.group_id,
            )
        )
        stats = await self.store.get_stats(group_id=self.group_id)
        assert isinstance(stats, dict)
        assert "entities" in stats
        assert "relationships" in stats
        assert "episodes" in stats
        assert stats["entities"] >= 1
