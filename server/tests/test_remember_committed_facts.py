"""Remember responses surface committed entity/relationship ids."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.ingestion.capture_surface import load_remember_committed_facts
from engram.models.entity import Entity


@pytest.mark.asyncio
async def test_load_remember_committed_facts_from_links_and_evidence():
    entity = Entity(
        id="ent_dec_1",
        name="LongMemEval is not Engram north star",
        entity_type="Decision",
        summary="continuity metric",
        identity_core=True,
        group_id="default",
    )
    graph = SimpleNamespace(
        get_episode_entities=AsyncMock(return_value=["ent_dec_1"]),
        get_episode_evidence=AsyncMock(
            return_value=[
                {
                    "status": "committed",
                    "fact_class": "entity",
                    "committed_id": "ent_dec_1",
                    "payload": {"name": entity.name, "entity_type": "Decision"},
                },
                {
                    "status": "committed",
                    "fact_class": "relationship",
                    "committed_id": "rel_1",
                    "payload": {
                        "subject": "Engram",
                        "predicate": "DECIDED",
                        "object": entity.name,
                    },
                },
                {
                    "status": "deferred",
                    "fact_class": "entity",
                    "committed_id": "ent_skip",
                    "payload": {},
                },
            ]
        ),
        batch_get_entities=AsyncMock(return_value={"ent_dec_1": entity}),
    )
    manager = SimpleNamespace(_graph=graph)

    entities, relationships = await load_remember_committed_facts(
        manager,
        episode_id="ep_1",
        group_id="default",
    )

    assert len(entities) == 1
    assert entities[0]["id"] == "ent_dec_1"
    assert entities[0]["entity_type"] == "Decision"
    assert entities[0]["identity_core"] is True
    assert len(relationships) == 1
    assert relationships[0]["id"] == "rel_1"
    assert relationships[0]["predicate"] == "DECIDED"


@pytest.mark.asyncio
async def test_load_remember_committed_facts_empty_without_graph():
    entities, relationships = await load_remember_committed_facts(
        SimpleNamespace(),
        episode_id="ep_1",
        group_id="default",
    )
    assert entities == []
    assert relationships == []
