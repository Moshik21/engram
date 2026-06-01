"""Unit tests for the write-side ObserverReflectPhase (no network, no HelixDB)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.observer.clustering import cluster_episodes_by_entity
from engram.consolidation.phases.reflect import (
    ObserverReflectPhase,
    cluster_key,
)
from engram.models.consolidation import CycleContext, ObservationRecord
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus

# --- Factories ---


def _ep(eid: str, content: str = "Some content with Capitalized Words here.", date=None) -> Episode:
    return Episode(
        id=eid,
        content=content,
        source="gt:s1",
        status=EpisodeStatus.COMPLETED,
        group_id="default",
        conversation_date=date,
    )


def _entity(eid: str, name: str, role: str | None = None) -> Entity:
    attrs = {"role": role} if role else {}
    return Entity(id=eid, name=name, entity_type="Person", group_id="default", attributes=attrs)


def _make_deps(
    *,
    episodes=None,
    episode_entities=None,
    entities=None,
    rels_by_entity=None,
):
    graph_store = AsyncMock()
    graph_store.get_episodes = AsyncMock(return_value=episodes or [])

    ent_map = episode_entities or {}

    async def get_episode_entities(episode_id, group_id="default"):
        return ent_map.get(episode_id, [])

    graph_store.get_episode_entities = get_episode_entities

    entity_lookup = {e.id: e for e in (entities or [])}

    async def get_entity(entity_id, group_id):
        return entity_lookup.get(entity_id)

    graph_store.get_entity = get_entity

    rels = rels_by_entity or {}

    async def get_relationships(entity_id, direction="both", group_id="default"):
        return rels.get(entity_id, [])

    graph_store.get_relationships = get_relationships

    graph_store.create_episode = AsyncMock()
    graph_store.link_episode_entity = AsyncMock()

    activation_store = AsyncMock()
    search_index = AsyncMock()
    return graph_store, activation_store, search_index


async def _run(graph_store, activation_store, search_index, cfg, *, context=None, dry_run=False):
    phase = ObserverReflectPhase()
    return await phase.execute(
        group_id="default",
        graph_store=graph_store,
        activation_store=activation_store,
        search_index=search_index,
        cfg=cfg,
        cycle_id="cyc1",
        context=context,
        dry_run=dry_run,
    )


# --- Config defaults / ship-dark ---


def test_config_observer_reflect_off_by_default():
    cfg = ActivationConfig()
    assert cfg.observer_reflect_enabled is False
    assert cfg.observer_reflect_llm_enabled is False


def test_config_no_profile_enables_observer_reflect():
    for profile in ("observe", "conservative", "standard"):
        cfg = ActivationConfig(consolidation_profile=profile)
        assert cfg.observer_reflect_enabled is False, profile
        assert cfg.observer_reflect_llm_enabled is False, profile


def test_config_defaults():
    cfg = ActivationConfig()
    assert cfg.observer_reflect_min_cluster == 3
    assert cfg.observer_reflect_min_importance == 0.4
    assert cfg.observer_reflect_max_observations_per_cycle == 5
    assert cfg.observer_reflect_max_episodes_scan == 500


@pytest.mark.asyncio
async def test_phase_skipped_when_disabled():
    graph_store, activation_store, search_index = _make_deps(episodes=[_ep("ep_a")])
    cfg = ActivationConfig()
    assert cfg.observer_reflect_enabled is False

    result, records = await _run(graph_store, activation_store, search_index, cfg)
    assert result.status == "skipped"
    assert records == []
    # Ship-dark: zero reads/writes while OFF.
    graph_store.get_episodes.assert_not_called()
    graph_store.create_episode.assert_not_called()
    search_index.index_episode.assert_not_called()


# --- Clustering determinism ---


def test_clusters_by_shared_entity_deterministic():
    episodes = [_ep("ep_3"), _ep("ep_1"), _ep("ep_2"), _ep("ep_solo")]
    episode_entities = {
        "ep_1": ["e1"],
        "ep_2": ["e1"],
        "ep_3": ["e1"],
        "ep_solo": ["e9"],  # singleton -> dropped
    }
    out1 = cluster_episodes_by_entity(episodes, episode_entities, min_cluster_size=3)
    out2 = cluster_episodes_by_entity(episodes, episode_entities, min_cluster_size=3)
    assert out1 == out2
    assert len(out1) == 1
    # Stable sort by episode_id.
    assert out1[0].episode_ids == ("ep_1", "ep_2", "ep_3")
    assert "ep_solo" not in out1[0].episode_ids


def test_singletons_below_min_cluster_dropped():
    episodes = [_ep("ep_1"), _ep("ep_2")]
    episode_entities = {"ep_1": ["e1"], "ep_2": ["e1"]}
    out = cluster_episodes_by_entity(episodes, episode_entities, min_cluster_size=3)
    assert out == []


# --- Importance gate ---


@pytest.mark.asyncio
async def test_low_importance_cluster_skipped():
    # Tiny, lowercase content -> importance below default 0.4 floor.
    episodes = [_ep("ep_1", "a"), _ep("ep_2", "b"), _ep("ep_3", "c")]
    episode_entities = {"ep_1": ["e1"], "ep_2": ["e1"], "ep_3": ["e1"]}
    graph_store, activation_store, search_index = _make_deps(
        episodes=episodes, episode_entities=episode_entities,
        entities=[_entity("e1", "Alice", role="Engineer")],
    )
    cfg = ActivationConfig(observer_reflect_enabled=True)

    result, records = await _run(graph_store, activation_store, search_index, cfg)
    assert len(records) == 1
    assert records[0].action == "skipped_low_importance"
    assert records[0].observation_episode_id == ""
    graph_store.create_episode.assert_not_called()


# --- Observation episode shape ---


def _high_signal_episodes():
    content = (
        "Alice Johnson was promoted to Senior Staff Engineer at Acme Corporation "
        "leading the Platform Reliability initiative across multiple teams this quarter."
    )
    return [
        _ep("ep_1", content, date=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        _ep("ep_2", content, date=datetime(2026, 3, 1, tzinfo=timezone.utc)),
        _ep("ep_3", content, date=datetime(2026, 2, 1, tzinfo=timezone.utc)),
    ]


@pytest.mark.asyncio
async def test_creates_observation_episode_shape():
    episodes = _high_signal_episodes()
    episode_entities = {"ep_1": ["e1"], "ep_2": ["e1"], "ep_3": ["e1"]}
    alice = _entity("e1", "Alice Johnson", role="Senior Staff Engineer")
    graph_store, activation_store, search_index = _make_deps(
        episodes=episodes, episode_entities=episode_entities, entities=[alice],
    )
    cfg = ActivationConfig(observer_reflect_enabled=True)
    context = CycleContext()

    result, records = await _run(
        graph_store, activation_store, search_index, cfg, context=context
    )

    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, ObservationRecord)
    assert rec.action == "created"
    assert rec.cluster_size == 3
    assert rec.synthesizer == "template"

    # Episode written with the expected shape.
    graph_store.create_episode.assert_awaited_once()
    obs: Episode = graph_store.create_episode.await_args.args[0]
    assert obs.memory_tier == "observation"
    assert obs.source == "observer:reflector"
    assert obs.status == EpisodeStatus.COMPLETED
    assert obs.projection_state == EpisodeProjectionState.PROJECTED
    # conversation_date == max() of cluster dates (newest).
    assert obs.conversation_date == datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert "Current role: Senior Staff Engineer." in obs.content

    # Indexed for vector retrievability + entity-linked.
    search_index.index_episode.assert_awaited_once()
    graph_store.link_episode_entity.assert_awaited()
    assert obs.id in context.observation_episode_ids


@pytest.mark.asyncio
async def test_idempotent_rerun():
    episodes = _high_signal_episodes()
    episode_entities = {"ep_1": ["e1"], "ep_2": ["e1"], "ep_3": ["e1"]}
    alice = _entity("e1", "Alice Johnson", role="Senior Staff Engineer")
    graph_store, activation_store, search_index = _make_deps(
        episodes=episodes, episode_entities=episode_entities, entities=[alice],
    )
    cfg = ActivationConfig(observer_reflect_enabled=True)

    # First run creates one observation.
    result, records = await _run(graph_store, activation_store, search_index, cfg)
    assert records[0].action == "created"
    obs: Episode = graph_store.create_episode.await_args.args[0]

    # Second run: the prior observation is now in the store; the cluster key
    # already exists, so no new observation is created.
    episodes_with_obs = list(episodes) + [obs]
    graph_store.get_episodes = AsyncMock(return_value=episodes_with_obs)
    graph_store.create_episode.reset_mock()

    result2, records2 = await _run(graph_store, activation_store, search_index, cfg)
    assert records2 == []
    graph_store.create_episode.assert_not_called()


@pytest.mark.asyncio
async def test_dry_run_no_writes():
    episodes = _high_signal_episodes()
    episode_entities = {"ep_1": ["e1"], "ep_2": ["e1"], "ep_3": ["e1"]}
    alice = _entity("e1", "Alice Johnson", role="Senior Staff Engineer")
    graph_store, activation_store, search_index = _make_deps(
        episodes=episodes, episode_entities=episode_entities, entities=[alice],
    )
    cfg = ActivationConfig(observer_reflect_enabled=True)

    result, records = await _run(graph_store, activation_store, search_index, cfg, dry_run=True)
    # Audit record emitted but no writes.
    assert len(records) == 1
    assert records[0].action == "created"
    graph_store.create_episode.assert_not_called()
    search_index.index_episode.assert_not_called()
    graph_store.link_episode_entity.assert_not_called()


def test_cluster_key_stable_and_order_independent():
    assert cluster_key(("ep_2", "ep_1")) == cluster_key(("ep_1", "ep_2"))
    assert cluster_key(("ep_1",)).startswith("reflect_cluster:")
