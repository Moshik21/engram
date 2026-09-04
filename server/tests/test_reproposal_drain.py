"""Resident-agent step 5: narrow-projected episodes are visible and re-proposable."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from engram.retrieval.presenter import present_api_recall_item
from engram.retrieval.result_builder import extraction_label


def test_extraction_label_reads_the_projection_reason():
    assert extraction_label("projected_agent") == "agent"
    assert extraction_label("projected_narrow") == "narrow"
    assert extraction_label("projected") == "narrow"  # legacy rows: the external rungs never ran
    assert extraction_label("auto_capture_cue_only") is None
    assert extraction_label(None) is None


def test_api_items_say_who_extracted_the_episode():
    raw = {
        "result_type": "episode",
        "episode": {"id": "ep_1", "content": "x", "extraction": "narrow"},
        "score": 0.5,
        "score_breakdown": {},
    }
    assert present_api_recall_item(raw)["episode"]["extractedBy"] == "narrow"
    raw["episode"].pop("extraction")
    assert present_api_recall_item(raw)["episode"]["extractedBy"] is None


@pytest.mark.asyncio
async def test_manager_lists_narrow_projected_episodes_newest_first():
    from engram.graph_manager import GraphManager

    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 2, tzinfo=timezone.utc)
    eps = [
        SimpleNamespace(
            id="old_narrow",
            created_at=t0,
            source="hook",
            content="a" * 3000,
            last_projection_reason="projected",
        ),
        SimpleNamespace(
            id="agent",
            created_at=t1,
            source="mcp",
            content="b",
            last_projection_reason="projected_agent",
        ),
        SimpleNamespace(
            id="new_narrow",
            created_at=t1,
            source="hook",
            content="c",
            last_projection_reason="projected_narrow",
        ),
        SimpleNamespace(
            id="cue",
            created_at=t1,
            source="hook",
            content="d",
            last_projection_reason="auto_capture_cue_only",
        ),
    ]

    class _Graph:
        async def get_episodes(self, group_id="default", limit=50, offset=0):
            return eps

    manager = GraphManager.__new__(GraphManager)
    manager._graph = _Graph()
    rows = await GraphManager.list_unstructured_episodes(manager, "default", limit=10)
    assert [r["episodeId"] for r in rows] == ["new_narrow", "old_narrow"]
    assert len(rows[1]["content"]) == 2000  # bounded payload


def test_remember_can_target_an_existing_episode_and_the_list_tool_is_operator_only():
    from engram.mcp import surface
    from engram.mcp.server import list_unstructured_episodes, remember

    assert "episode_id" in inspect.signature(remember).parameters
    assert callable(list_unstructured_episodes)
    assert "list_unstructured_episodes" in surface.OPERATOR_TOOLS
    assert "list_unstructured_episodes" not in surface.PUBLIC_TOOLS
