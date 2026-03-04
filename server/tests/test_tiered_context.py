"""Tests for tiered get_context() and briefing format."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.models.entity import Entity
from engram.models.relationship import Relationship
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex
from tests.conftest import MockExtractor

GROUP = "test_group"


def _entity(eid, name, etype, summary, identity_core=False):
    return Entity(
        id=eid,
        name=name,
        entity_type=etype,
        summary=summary,
        group_id=GROUP,
        identity_core=identity_core,
    )


def _rel(rid, src, tgt, pred, vf):
    return Relationship(
        id=rid,
        source_id=src,
        target_id=tgt,
        predicate=pred,
        weight=1.0,
        valid_from=vf,
        group_id=GROUP,
    )


@pytest_asyncio.fixture
async def tiered_manager(tmp_path):
    """GraphManager with identity core + regular entities for tiered context testing."""
    db_path = str(tmp_path / "tiered_test.db")
    gs = SQLiteGraphStore(db_path)
    await gs.initialize()
    acts = MemoryActivationStore(cfg=ActivationConfig())
    si = FTS5SearchIndex(db_path)
    await si.initialize(db=gs._db)

    entities = [
        _entity("ent_konner", "Konner", "Person", "Software engineer"),
        _entity("ent_engram", "Engram", "Project", "Memory system"),
        _entity("ent_python", "Python", "Technology", "Programming language"),
        _entity("ent_fastapi", "FastAPI", "Technology", "Web framework"),
    ]
    for e in entities:
        await gs.create_entity(e)
    # Mark Konner as identity core (mimics auto-detection from identity predicates)
    await gs.update_entity("ent_konner", {"identity_core": 1}, group_id=GROUP)

    now_dt = entities[0].created_at
    rels = [
        _rel("rel_builds", "ent_konner", "ent_engram", "BUILDS", now_dt),
        _rel("rel_uses", "ent_engram", "ent_python", "USES", now_dt),
    ]
    for r in rels:
        await gs.create_relationship(r)

    now = time.time()
    await acts.record_access("ent_konner", now - 10, group_id=GROUP)
    await acts.record_access("ent_engram", now - 20, group_id=GROUP)
    await acts.record_access("ent_python", now - 100, group_id=GROUP)
    await acts.record_access("ent_fastapi", now - 200, group_id=GROUP)

    mgr = GraphManager(gs, acts, si, MockExtractor())
    yield mgr
    await gs.close()


def _make_mock_anthropic(text="Briefing text.", *, raises=False):
    """Create a mock anthropic module with a working Anthropic client."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=text)]
    mock_client = MagicMock()
    if raises:
        mock_client.messages.create.side_effect = RuntimeError("API down")
    else:
        mock_client.messages.create.return_value = mock_resp
    mock_mod = MagicMock()
    mock_mod.Anthropic.return_value = mock_client
    return mock_mod, mock_client


class TestTieredSections:
    @pytest.mark.asyncio
    async def test_get_context_tiered_sections(self, tiered_manager):
        """Output has Identity and Recent Activity sections."""
        result = await tiered_manager.get_context(group_id=GROUP)
        ctx = result["context"]
        assert "## Identity" in ctx
        assert "## Recent Activity" in ctx
        assert result["format"] == "structured"

    @pytest.mark.asyncio
    async def test_get_context_project_path_derives_hint(self, tiered_manager):
        """project_path='/foo/Engram' derives topic_hint='Engram' and finds project entities."""
        result = await tiered_manager.get_context(
            group_id=GROUP, project_path="/foo/Engram",
        )
        ctx = result["context"]
        # Should have a Project Context section with derived hint
        assert "## Project Context (Engram)" in ctx

    @pytest.mark.asyncio
    async def test_get_context_home_dir_skips_project(self, tiered_manager):
        """project_path=Path.home() should not produce a Project Context section."""
        result = await tiered_manager.get_context(
            group_id=GROUP, project_path=str(Path.home()),
        )
        ctx = result["context"]
        assert "## Project Context" not in ctx

    @pytest.mark.asyncio
    async def test_get_context_budget_rollover(self, tiered_manager):
        """No project → Layer 3 gets Layer 2's budget (no crash, reasonable output)."""
        result = await tiered_manager.get_context(group_id=GROUP)
        # Should work fine, no project context section
        assert "## Project Context" not in result["context"]
        assert result["entity_count"] > 0

    @pytest.mark.asyncio
    async def test_get_context_deduplication(self, tiered_manager):
        """Entities in Layer 1 (identity) don't appear again in Layer 3 entity list."""
        result = await tiered_manager.get_context(group_id=GROUP)
        ctx = result["context"]
        # Split into sections, check Konner only in Identity (may appear in facts elsewhere)
        sections = ctx.split("## Recent Activity")
        if len(sections) > 1:
            recent = sections[1]
            # Konner should NOT be listed as an entity line in Recent Activity
            # Entity lines start with "- Name (Type,"
            entity_lines = [
                ln for ln in recent.split("\n")
                if ln.startswith("- Konner (")
            ]
            assert len(entity_lines) == 0, "Konner should not be duplicated in Recent Activity"


class TestBriefingFormat:
    @pytest.mark.asyncio
    async def test_briefing_format(self, tiered_manager):
        """Briefing format calls Anthropic with structured context."""
        mock_mod, mock_client = _make_mock_anthropic("Konner builds Engram, a memory system.")

        with patch.dict(sys.modules, {"anthropic": mock_mod}):
            result = await tiered_manager.get_context(
                group_id=GROUP, format="briefing",
            )
            assert result["format"] == "briefing"
            assert "Konner builds Engram" in result["context"]
            mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_briefing_cache_hit(self, tiered_manager):
        """Second briefing call uses cache, doesn't invoke Haiku."""
        mock_mod, mock_client = _make_mock_anthropic("Cached briefing.")

        with patch.dict(sys.modules, {"anthropic": mock_mod}):
            r1 = await tiered_manager.get_context(group_id=GROUP, format="briefing")
            r2 = await tiered_manager.get_context(group_id=GROUP, format="briefing")
            assert r1["format"] == "briefing"
            assert r2["format"] == "briefing"
            # Only one Haiku call — second used cache
            assert mock_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_briefing_cache_invalidation(self, tiered_manager):
        """invalidate_briefing_cache() clears cache entries."""
        tiered_manager._briefing_cache[("test_group", None)] = (time.time(), "old")
        tiered_manager.invalidate_briefing_cache("test_group")
        assert ("test_group", None) not in tiered_manager._briefing_cache

    @pytest.mark.asyncio
    async def test_briefing_fallback_on_error(self, tiered_manager):
        """If Haiku raises, falls back to structured format."""
        mock_mod, _ = _make_mock_anthropic(raises=True)

        with patch.dict(sys.modules, {"anthropic": mock_mod}):
            result = await tiered_manager.get_context(
                group_id=GROUP, format="briefing",
            )
            # Should fall back — context is structured markdown, not an error
            assert "##" in result["context"]

    @pytest.mark.asyncio
    async def test_briefing_disabled_config(self, tiered_manager):
        """briefing_enabled=False returns structured even when format='briefing'."""
        tiered_manager._cfg.briefing_enabled = False
        result = await tiered_manager.get_context(
            group_id=GROUP, format="briefing",
        )
        assert result["format"] == "structured"

    @pytest.mark.asyncio
    async def test_get_context_empty_graph(self, tmp_path):
        """Empty graph returns valid context with zero counts."""
        db_path = str(tmp_path / "empty.db")
        gs = SQLiteGraphStore(db_path)
        await gs.initialize()
        acts = MemoryActivationStore(cfg=ActivationConfig())
        si = FTS5SearchIndex(db_path)
        await si.initialize(db=gs._db)
        mgr = GraphManager(gs, acts, si, MockExtractor())

        result = await mgr.get_context(group_id="empty")
        assert result["entity_count"] == 0
        assert result["fact_count"] == 0
        assert result["format"] == "structured"
        await gs.close()
