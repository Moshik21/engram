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
    graph.link_episode_entity.assert_called_once_with(
        "ep_apply",
        created_id,
        group_id="default",
    )
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
async def test_apply_engine_blocks_identity_core_summary_merge_from_client_proposal():
    """Ship path: contradictory client proposal must not append into identity_core.

    merge_entity_attributes builds ``old; new``. Protection must compare the
    *proposed* summary to existing, then strip summary from updates — otherwise
    protected facts silently grow with conflicting text.
    """
    from engram.extraction.harness_metrics import get_harness_metrics, reset_harness_metrics

    reset_harness_metrics()
    existing = Entity(
        id="ent_pref",
        name="Prefer markdown handoffs",
        entity_type="Preference",
        summary="User prefers markdown handoffs",
        identity_core=True,
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
        cfg=ActivationConfig(identity_core_enabled=True),
        canonicalizer=PredicateCanonicalizer(),
    )
    episode = Episode(
        id="ep_conflict",
        content="User prefers JSON APIs only for agent memory.",
        group_id="default",
    )
    proposed = "User prefers JSON APIs only for agent memory"
    await engine.apply_entities(
        [
            EntityCandidate(
                name="Prefer markdown handoffs",
                entity_type="Preference",
                summary=proposed,
                raw_payload={
                    "signals": ["client_proposal", "high_signal_type", "span_verified"],
                },
            )
        ],
        episode,
        "default",
        recall_content=episode.content,
    )

    # Must resolve to existing entity (not create a second one).
    graph.create_entity.assert_not_called()
    assert graph.update_entity.await_count >= 1
    for call in graph.update_entity.await_args_list:
        updates = call.args[1] if len(call.args) > 1 else call.kwargs.get("updates", {})
        summary_update = updates.get("summary") if isinstance(updates, dict) else None
        # Contradictory proposed text must never land as "old; new" merge.
        assert summary_update is None or proposed not in str(summary_update)
        assert "; " not in str(summary_update or "")
        if summary_update is not None:
            assert "JSON APIs" not in str(summary_update)

    snap = get_harness_metrics()
    assert snap.identity_core_conflicts >= 1


@pytest.mark.asyncio
async def test_apply_engine_allows_identity_core_summary_compatible_expansion():
    """Compatible expansion of protected summary may still update."""
    existing = Entity(
        id="ent_pref2",
        name="Prefer sparse promotion",
        entity_type="Decision",
        summary="Prefer sparse promotion",
        identity_core=True,
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
        cfg=ActivationConfig(identity_core_enabled=True),
        canonicalizer=PredicateCanonicalizer(),
    )
    episode = Episode(
        id="ep_expand",
        content="Prefer sparse promotion of durable Decisions only.",
        group_id="default",
    )
    await engine.apply_entities(
        [
            EntityCandidate(
                name="Prefer sparse promotion",
                entity_type="Decision",
                summary="Prefer sparse promotion of durable Decisions only",
                raw_payload={"signals": ["client_proposal", "high_signal_type"]},
            )
        ],
        episode,
        "default",
        recall_content=episode.content,
    )
    # Expansion contains existing phrase → not a conflict; merge may apply.
    assert graph.update_entity.await_count >= 1


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
