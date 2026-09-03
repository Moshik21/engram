"""A demoted cue (blank cue_text) surfaces its episode instead of an empty row.

2026-09-03: cue hygiene blanks cue_text but the cue vector stays searchable, so
the cue lane returned rows whose only visible text was empty. Three of them
made a "full" preflight page for a pair question and the agent got nothing.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from engram.graph_manager import GraphManager
from engram.models.episode import EpisodeProjectionState

pytestmark = pytest.mark.asyncio


class _Builder:
    def __init__(self) -> None:
        self.kind: str | None = None

    def cue_episode_result(self, episode, cue, scored, linked_entities=None, hit_increment=0):
        self.kind = "cue"
        return {"result_type": "cue_episode"}

    def episode_result(self, episode, scored, linked_entities=None):
        self.kind = "episode"
        return {"result_type": "episode", "episode": {"content": episode.content}}


def _manager(cue_text: str | None):
    episode = SimpleNamespace(id="ep_1", content="Thompson sampling removed: noise")
    cue = None if cue_text is None else SimpleNamespace(cue_text=cue_text)

    async def get_episode_by_id(eid, gid):
        return episode

    async def get_episode_entities(eid, group_id=None):
        return []

    async def get_cue(eid, gid):
        return cue

    builder = _Builder()
    m = SimpleNamespace(
        _graph=SimpleNamespace(
            get_episode_by_id=get_episode_by_id, get_episode_entities=get_episode_entities
        ),
        _get_episode_cue=get_cue,
        _recall_result_builder=builder,
        _episode_projection_state_value=lambda ep: EpisodeProjectionState.PROJECTED.value,
        builder=builder,
    )
    m._fallback_episode_recall_result = MethodType(GraphManager._fallback_episode_recall_result, m)
    return m


async def test_blank_cue_falls_through_to_the_episode() -> None:
    m = _manager(cue_text="   ")
    row = await m._fallback_episode_recall_result(
        "ep_1", 0.5, group_id="default", result_type="cue_episode"
    )
    assert m.builder.kind == "episode"
    assert row["episode"]["content"].startswith("Thompson")


async def test_real_cue_still_returns_a_cue_row() -> None:
    m = _manager(cue_text="mentions: Thompson sampling")
    row = await m._fallback_episode_recall_result(
        "ep_1", 0.5, group_id="default", result_type="cue_episode"
    )
    assert m.builder.kind == "cue"
    assert row["result_type"] == "cue_episode"
