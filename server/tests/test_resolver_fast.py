"""Tests for resolve_entity_fast — indexed candidate-based resolution."""

import pytest

from engram.extraction.resolver import resolve_entity_fast
from engram.models.entity import Entity


def _make_entity(name, entity_type="Technology", entity_id="ent_1"):
    return Entity(id=entity_id, name=name, entity_type=entity_type)


@pytest.mark.asyncio
class TestResolveEntityFast:
    async def test_session_entity_priority(self):
        """Session entities should be checked before DB candidates."""
        session_ent = _make_entity("Python", entity_id="session_1")

        async def mock_retriever(name, group_id):
            return [_make_entity("Python", entity_id="db_1")]

        result = await resolve_entity_fast(
            "Python",
            "Technology",
            mock_retriever,
            "default",
            session_entities={"session_1": session_ent},
        )
        assert result is not None
        assert result.id == "session_1"

    async def test_db_candidate_match(self):
        """Should match from DB candidates when no session match."""

        async def mock_retriever(name, group_id):
            return [_make_entity("Python", entity_id="db_1")]

        result = await resolve_entity_fast(
            "Python",
            "Technology",
            mock_retriever,
            "default",
        )
        assert result is not None
        assert result.id == "db_1"

    async def test_no_match_returns_none(self):
        """Should return None when no candidates match."""

        async def mock_retriever(name, group_id):
            return [_make_entity("Redis", entity_id="db_1")]

        result = await resolve_entity_fast(
            "Python",
            "Technology",
            mock_retriever,
            "default",
        )
        assert result is None

    async def test_empty_candidates(self):
        """Should return None when no candidates returned."""

        async def mock_retriever(name, group_id):
            return []

        result = await resolve_entity_fast(
            "Python",
            "Technology",
            mock_retriever,
            "default",
        )
        assert result is None

    async def test_type_boost(self):
        """Same-type candidates should get a boost."""

        async def mock_retriever(name, group_id):
            return [
                _make_entity("Mesa", entity_type="Location", entity_id="loc_1"),
                _make_entity("Mesa", entity_type="Concept", entity_id="con_1"),
            ]

        result = await resolve_entity_fast(
            "Mesa",
            "Location",
            mock_retriever,
            "default",
        )
        assert result is not None
        assert result.id == "loc_1"

    async def test_fuzzy_match_from_candidates(self):
        """Should fuzzy-match candidates like the original resolver."""

        async def mock_retriever(name, group_id):
            return [_make_entity("React", entity_id="db_1")]

        result = await resolve_entity_fast(
            "ReactJS",
            "Technology",
            mock_retriever,
            "default",
        )
        assert result is not None
        assert result.id == "db_1"

    async def test_session_entities_none(self):
        """Should work when session_entities is None."""

        async def mock_retriever(name, group_id):
            return [_make_entity("Python", entity_id="db_1")]

        result = await resolve_entity_fast(
            "Python",
            "Technology",
            mock_retriever,
            "default",
            session_entities=None,
        )
        assert result is not None
        assert result.id == "db_1"
