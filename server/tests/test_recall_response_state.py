from __future__ import annotations

from types import SimpleNamespace

import pytest

from engram.retrieval.response_state import RecallResponseStateService


def test_triggered_intention_views_serializes_optional_fields() -> None:
    match = SimpleNamespace(
        trigger_text="meeting",
        action_text="summarize decisions",
        similarity=0.93456,
        matched_via="embedding",
        context={"episode_id": "ep_1"},
        see_also=["ent_project"],
    )

    views = RecallResponseStateService().triggered_intention_views([match])

    assert views == [
        {
            "trigger": "meeting",
            "action": "summarize decisions",
            "similarity": 0.9346,
            "matched_via": "embedding",
            "context": {"episode_id": "ep_1"},
            "see_also": ["ent_project"],
        }
    ]


def test_near_miss_views_returns_copy() -> None:
    near_misses = [{"result_type": "cue_episode"}]

    views = RecallResponseStateService().near_miss_views(near_misses)

    assert views == near_misses
    assert views is not near_misses


@pytest.mark.asyncio
async def test_get_access_count_reads_activation_store() -> None:
    class Activation:
        async def get_activation(self, entity_id: str):
            if entity_id == "ent_1":
                return SimpleNamespace(access_count=4)
            return None

    activation = Activation()

    assert await RecallResponseStateService().get_access_count(activation, "ent_1") == 4
    assert await RecallResponseStateService().get_access_count(activation, "missing") == 0
    assert await RecallResponseStateService().get_access_count(activation, "") == 0


def test_surprise_connection_views_serializes_cache_entries() -> None:
    surprise = SimpleNamespace(
        entity_name="GraphManager",
        connected_to_name="GraphStateService",
        predicate="DELEGATES_TO",
        surprise_score=0.87654,
    )
    cache = SimpleNamespace(get=lambda group_id, now: [surprise] if group_id == "brain" else [])

    views = RecallResponseStateService().surprise_connection_views(
        cache,
        group_id="brain",
        now=123.0,
        limit=3,
    )

    assert views == [
        {
            "entity": "GraphManager",
            "connected_to": "GraphStateService",
            "relationship": "DELEGATES_TO",
            "surprise_score": 0.8765,
        }
    ]


def test_surprise_connection_views_handles_missing_cache() -> None:
    assert (
        RecallResponseStateService().surprise_connection_views(
            None,
            group_id="brain",
            now=123.0,
        )
        == []
    )
