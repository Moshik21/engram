"""Shared recall presenter contract tests."""

from __future__ import annotations

import pytest

from engram.retrieval.presenter import (
    present_api_recall_item,
    present_chat_recall_item,
    present_mcp_recall_item,
    recall_contract_item,
)


async def _resolve_entity_name(entity_id: str) -> str:
    return {
        "ent_alice": "Alice",
        "ent_google": "Google",
    }.get(entity_id, entity_id)


async def _get_access_count(entity_id: str) -> int:
    return 7 if entity_id == "ent_alice" else 0


@pytest.mark.asyncio
async def test_entity_recall_presenters_share_contract():
    raw = {
        "result_type": "entity",
        "entity": {
            "id": "ent_alice",
            "name": "Alice",
            "type": "Person",
            "summary": "Alice works on search.",
        },
        "score": 0.81234,
        "score_breakdown": {
            "semantic": 0.7,
            "activation": 0.2,
            "edge_proximity": 0.1,
            "exploration_bonus": 0.0,
            "relevance_confidence": 0.91,
        },
        "relationships": [
            {
                "id": "rel_1",
                "predicate": "WORKS_AT",
                "source_id": "ent_alice",
                "target_id": "ent_google",
                "weight": 1.0,
                "polarity": "positive",
            }
        ],
    }

    contract = recall_contract_item(raw)
    api = present_api_recall_item(raw)
    mcp = await present_mcp_recall_item(
        raw,
        resolve_entity_name=_resolve_entity_name,
        get_access_count=_get_access_count,
    )
    chat = present_chat_recall_item(raw)

    assert contract["result_type"] == api["resultType"] == mcp["result_type"]
    assert api["entity"]["name"] == mcp["entity"] == chat["name"] == "Alice"
    assert api["entity"]["entityType"] == mcp["entity_type"] == chat["entityType"] == "Person"
    assert api["scoreBreakdown"]["edgeProximity"] == pytest.approx(0.1)
    assert mcp["related_facts"][0] == {
        "subject": "Alice",
        "predicate": "WORKS_AT",
        "object": "Google",
        "polarity": "positive",
    }
    assert mcp["access_count"] == 7


@pytest.mark.asyncio
async def test_episode_recall_presenters_share_contract():
    raw = {
        "result_type": "episode",
        "episode": {
            "id": "ep_1",
            "content": "Discussed the recall presentation contract.",
            "source": "test",
            "created_at": "2026-05-11T12:00:00",
        },
        "score": 0.61,
        "score_breakdown": {
            "semantic": 0.6,
            "activation": 0.0,
            "edge_proximity": 0.0,
            "exploration_bonus": 0.01,
        },
        "linked_entities": [{"name": "Recall Contract"}],
    }

    contract = recall_contract_item(raw)
    api = present_api_recall_item(raw)
    mcp = await present_mcp_recall_item(
        raw,
        resolve_entity_name=_resolve_entity_name,
        get_access_count=_get_access_count,
    )
    chat = present_chat_recall_item(raw)

    assert contract["result_type"] == api["resultType"] == mcp["result_type"]
    assert contract["episode_id"] == api["episode"]["id"] == mcp["episode_id"] == "ep_1"
    assert api["episode"]["content"] == mcp["content"] == raw["episode"]["content"]
    assert chat["type"] == "episode"
    assert mcp["linked_entities"] == ["Recall Contract"]


@pytest.mark.asyncio
async def test_cue_episode_recall_presenters_share_contract():
    raw = {
        "result_type": "cue_episode",
        "cue": {
            "episode_id": "ep_cue",
            "cue_text": "Latent cue for brain loop discussion",
            "supporting_spans": ["Capture -> Cue -> Project"],
            "projection_state": "cue_only",
            "route_reason": "cue_recall",
            "hit_count": 3,
            "surfaced_count": 2,
            "selected_count": 1,
            "used_count": 1,
            "near_miss_count": 0,
            "policy_score": 0.74,
            "last_feedback_at": "2026-05-11T12:05:00",
            "last_projected_at": None,
        },
        "episode": {
            "id": "ep_cue",
            "source": "observe",
            "created_at": "2026-05-11T12:00:00",
        },
        "score": 0.74,
        "score_breakdown": {
            "semantic": 0.7,
            "activation": 0.0,
            "edge_proximity": 0.0,
            "exploration_bonus": 0.04,
        },
    }

    contract = recall_contract_item(raw)
    api = present_api_recall_item(raw)
    mcp = await present_mcp_recall_item(
        raw,
        resolve_entity_name=_resolve_entity_name,
        get_access_count=_get_access_count,
    )
    chat = present_chat_recall_item(raw)

    assert contract["result_type"] == api["resultType"] == mcp["result_type"]
    assert api["cue"]["episodeId"] == mcp["episode_id"] == chat["episodeId"] == "ep_cue"
    assert api["cue"]["cueText"] == mcp["cue_text"] == chat["cueText"]
    assert api["cue"]["projectionState"] == mcp["projection_state"] == chat["projectionState"]
    assert api["cue"]["usedCount"] == 1
    assert chat["policyScore"] == pytest.approx(0.74)
