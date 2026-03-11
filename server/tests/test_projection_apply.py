"""Tests for the shared projection apply engine."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.extraction.apply import ApplyEngine
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.models import ClaimCandidate, EntityCandidate
from engram.models.entity import Entity
from engram.models.episode import Episode


@pytest.mark.asyncio
async def test_apply_engine_creates_and_links_entities():
    graph = AsyncMock()
    graph.find_entity_candidates = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.link_episode_entity = AsyncMock()
    graph.update_entity = AsyncMock()

    activation = AsyncMock()
    publish_access_event = AsyncMock()
    engine = ApplyEngine(
        graph_store=graph,
        activation_store=activation,
        cfg=ActivationConfig(),
        canonicalizer=PredicateCanonicalizer(),
        publish_access_event=publish_access_event,
    )
    episode = Episode(
        id="ep_apply",
        content="Alex moved to Phoenix.",
        group_id="default",
    )

    outcome = await engine.apply_entities(
        [
            EntityCandidate(
                name="Phoenix",
                entity_type="Location",
                summary="City in Arizona",
            )
        ],
        episode,
        "default",
        recall_content=episode.content,
    )

    created_id = outcome.entity_map["Phoenix"]
    assert outcome.new_entity_names == ["Phoenix"]
    graph.create_entity.assert_called_once()
    graph.link_episode_entity.assert_called_once_with("ep_apply", created_id)
    activation.record_access.assert_called_once()
    publish_access_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_engine_creates_relationships_from_claims():
    graph = AsyncMock()
    graph.get_relationships = AsyncMock(return_value=[])
    graph.find_conflicting_relationships = AsyncMock(return_value=[])
    graph.find_existing_relationship = AsyncMock(return_value=None)
    graph.create_relationship = AsyncMock()
    graph.update_entity = AsyncMock()

    engine = ApplyEngine(
        graph_store=graph,
        activation_store=AsyncMock(),
        cfg=ActivationConfig(),
        canonicalizer=PredicateCanonicalizer(),
    )

    results = await engine.apply_relationships(
        [
            ClaimCandidate(
                subject_text="Alex",
                object_text="Phoenix",
                predicate="lives in",
                raw_payload={
                    "source": "Alex",
                    "target": "Phoenix",
                    "predicate": "lives in",
                },
            )
        ],
        entity_map={"Alex": "ent_alex", "Phoenix": "ent_phoenix"},
        meta_entity_names=set(),
        group_id="default",
        source_episode="ep_apply",
    )

    assert len(results) == 1
    assert results[0].created is True
    assert results[0].predicate == PredicateCanonicalizer().canonicalize("LIVES_IN")
    graph.create_relationship.assert_called_once()


@pytest.mark.asyncio
async def test_apply_engine_promotes_code_like_entities_to_identifier_type():
    graph = AsyncMock()
    graph.find_entity_candidates = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.link_episode_entity = AsyncMock()
    graph.update_entity = AsyncMock()

    engine = ApplyEngine(
        graph_store=graph,
        activation_store=AsyncMock(),
        cfg=ActivationConfig(),
        canonicalizer=PredicateCanonicalizer(),
    )
    episode = Episode(id="ep_identifier", content="The SKU is 1712061.", group_id="default")

    await engine.apply_entities(
        [
            EntityCandidate(
                name="1712061",
                entity_type="Technology",
                summary="A catalog part number",
            )
        ],
        episode,
        "default",
        recall_content=episode.content,
    )

    created = graph.create_entity.await_args.args[0]
    assert isinstance(created, Entity)
    assert created.entity_type == "Identifier"
    assert created.canonical_identifier == "1712061"


@pytest.mark.asyncio
async def test_apply_engine_upgrades_existing_exact_identifier_alias_to_identifier():
    existing = Entity(
        id="ent_existing",
        name="1712061",
        entity_type="Technology",
        group_id="default",
    )
    graph = AsyncMock()
    graph.find_entity_candidates = AsyncMock(return_value=[existing])
    graph.create_entity = AsyncMock()
    graph.link_episode_entity = AsyncMock()
    graph.update_entity = AsyncMock()

    engine = ApplyEngine(
        graph_store=graph,
        activation_store=AsyncMock(),
        cfg=ActivationConfig(),
        canonicalizer=PredicateCanonicalizer(),
    )
    episode = Episode(id="ep_identifier_alias", content="Use SKU 1712061.", group_id="default")

    await engine.apply_entities(
        [
            EntityCandidate(
                name="SKU 1712061",
                entity_type="Identifier",
                summary="A part code alias",
            )
        ],
        episode,
        "default",
        recall_content=episode.content,
    )

    assert graph.create_entity.await_count == 0
    update_payloads = [call.args[1] for call in graph.update_entity.await_args_list]
    assert any(payload.get("entity_type") == "Identifier" for payload in update_payloads)
