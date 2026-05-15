"""Tests for raw recall result assembly."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.models.relationship import Relationship
from engram.retrieval.result_builder import RecallResultBuilder
from engram.retrieval.scorer import ScoredResult
from engram.utils.dates import utc_now


def _scored_result(
    node_id: str = "ep_1",
    *,
    result_type: str = "episode",
    chunk_context: str | None = None,
) -> ScoredResult:
    return ScoredResult(
        node_id=node_id,
        score=0.82,
        semantic_similarity=0.7,
        activation=0.08,
        spreading=0.0,
        edge_proximity=0.2,
        exploration_bonus=0.04,
        hop_distance=2,
        result_type=result_type,
        chunk_context=chunk_context,
        planner_support=0.12,
        planner_intents=["answer"],
        recall_trace=[{"stage": "seed"}],
    )


def test_episode_result_uses_tier_aware_truncation_and_chunk_context() -> None:
    cfg = ActivationConfig(
        recall_tier_aware_truncation_enabled=True,
        recall_semantic_content_limit=9,
    )
    builder = RecallResultBuilder(cfg)
    episode = Episode(
        id="ep_1",
        content="Semantic episode content should be truncated",
        source="test",
        status=EpisodeStatus.COMPLETED,
        projection_state=EpisodeProjectionState.CUE_ONLY,
        memory_tier="semantic",
        created_at=utc_now(),
    )
    cue = EpisodeCue(
        episode_id="ep_1",
        projection_state=EpisodeProjectionState.CUE_ONLY,
        cue_text="Semantic cue summary",
    )

    result = builder.episode_result(
        episode,
        _scored_result(chunk_context="chunk around the match"),
        linked_entities=["ent_1"],
    )

    assert builder.truncate_episode_content(episode, cue) == "Semantic "
    assert result["episode"]["content"] == "Semantic "
    assert result["episode"]["id"] == "ep_1"
    assert result["linked_entities"] == ["ent_1"]
    assert result["chunk_context"] == "chunk around the match"
    assert result["score_breakdown"]["exploration_bonus"] == 0.04


def test_cue_episode_result_normalizes_state_and_feedback_counts() -> None:
    builder = RecallResultBuilder(ActivationConfig())
    episode = Episode(
        id="ep_1",
        content="Latent episode",
        source="observe",
        status=EpisodeStatus.COMPLETED,
        projection_state=EpisodeProjectionState.CUE_ONLY,
        created_at=utc_now(),
    )
    cue = EpisodeCue(
        episode_id="ep_1",
        projection_state=EpisodeProjectionState.CUE_ONLY,
        cue_text="mentions: native Helix",
        first_spans=["native Helix path"],
        hit_count=2,
        route_reason="entity_dense",
    )

    result = builder.cue_episode_result(
        episode,
        cue,
        _scored_result(result_type="cue_episode"),
        linked_entities=["ent_native"],
        hit_increment=1,
    )

    assert result["result_type"] == "cue_episode"
    assert result["cue"]["projection_state"] == "cue_only"
    assert result["cue"]["hit_count"] == 3
    assert result["episode"] == {
        "id": "ep_1",
        "source": "observe",
        "created_at": episode.created_at.isoformat(),
        "conversation_date": None,
    }
    assert result["linked_entities"] == ["ent_native"]


def test_entity_result_preserves_relationship_and_planner_metadata() -> None:
    builder = RecallResultBuilder(ActivationConfig())
    entity = Entity(
        id="ent_1",
        name="Engram",
        entity_type="Project",
        summary="AI memory runtime.",
    )
    relationship = Relationship(
        id="rel_1",
        source_id="ent_1",
        target_id="ent_2",
        predicate="USES",
        polarity="uncertain",
    )

    result = builder.entity_result(
        entity,
        [relationship],
        _scored_result(node_id="ent_1", result_type="entity"),
    )

    assert result["result_type"] == "entity"
    assert result["entity"]["name"] == "Engram"
    assert result["relationships"][0]["polarity"] == "uncertain"
    assert result["score_breakdown"]["hop_distance"] == 2
    assert result["supporting_intents"] == ["answer"]
    assert result["recall_trace"] == [{"stage": "seed"}]
