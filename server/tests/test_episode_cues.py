"""Tests for deterministic episode cue generation and storage."""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig
from engram.extraction.cues import build_episode_cue
from engram.graph_manager import GraphManager
from engram.models.episode import Episode, EpisodeProjectionState
from tests.conftest import MockExtractor


def test_build_episode_cue_skips_system_discourse():
    ep = Episode(
        id="ep_meta",
        content="Entity ent_abc123 has activation score 0.91 in the retrieval pipeline",
        group_id="default",
    )
    cue = build_episode_cue(ep, ActivationConfig(cue_layer_enabled=True))
    assert cue is None


def test_build_episode_cue_marks_contradiction_as_scheduled():
    ep = Episode(
        id="ep_world",
        content="I live in Phoenix now, not Mesa anymore. My name is Konner.",
        group_id="default",
    )
    cue = build_episode_cue(ep, ActivationConfig(cue_layer_enabled=True))
    assert cue is not None
    assert cue.projection_state == EpisodeProjectionState.SCHEDULED
    assert cue.route_reason in {"contradiction_hint", "identity_hint"}
    assert any(m["text"] == "Phoenix" for m in cue.entity_mentions)


def test_build_episode_cue_applies_source_priority_boost():
    ep = Episode(
        id="ep_source_boost",
        content="Follow up on the migration plan and capture the latest decision.",
        source="remember",
        group_id="default",
    )
    cue = build_episode_cue(
        ep,
        ActivationConfig(
            cue_layer_enabled=True,
            cue_policy_learning_enabled=True,
            cue_policy_source_boosts={"remember": 0.75},
        ),
    )
    assert cue is not None
    assert cue.projection_state == EpisodeProjectionState.SCHEDULED
    assert cue.route_reason == "source_priority:remember"
    assert cue.policy_score >= cue.projection_priority


@pytest.mark.asyncio
async def test_sqlite_graph_store_persists_episode_cue(graph_store):
    ep = Episode(id="ep_cue_store", content="Alice moved to Berlin last month", group_id="default")
    await graph_store.create_episode(ep)

    cue = build_episode_cue(ep, ActivationConfig(cue_layer_enabled=True))
    assert cue is not None
    await graph_store.upsert_episode_cue(cue)

    stored = await graph_store.get_episode_cue("ep_cue_store", "default")
    assert stored is not None
    assert stored.episode_id == "ep_cue_store"
    assert stored.cue_text
    assert stored.discourse_class == "world"


@pytest.mark.asyncio
async def test_store_episode_generates_and_indexes_cue(graph_store, activation_store, search_index):
    cfg = ActivationConfig(cue_layer_enabled=True, cue_vector_index_enabled=False)
    manager = GraphManager(
        graph_store=graph_store,
        activation_store=activation_store,
        search_index=search_index,
        extractor=MockExtractor(),
        cfg=cfg,
    )

    episode_id = await manager.store_episode(
        "Konner moved to Phoenix in 2024 and is working on Engram extraction redesign",
        group_id="default",
        source="test",
    )

    cue = await graph_store.get_episode_cue(episode_id, "default")
    assert cue is not None
    assert cue.cue_text

    episode = await graph_store.get_episode_by_id(episode_id, "default")
    assert episode is not None
    assert episode.projection_state in {
        EpisodeProjectionState.CUED,
        EpisodeProjectionState.SCHEDULED,
    }

    results = await search_index.search_episode_cues("Phoenix redesign", group_id="default")
    assert results
    assert results[0][0] == episode_id
