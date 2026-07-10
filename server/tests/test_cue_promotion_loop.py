"""B13: cue->promotion feedback fires when an episode outscores its cue.

An unprojected episode (CUE_ONLY/QUEUED) can surface from both the episode and
the cue search. When the episode-typed candidate outscores its cue candidate,
the survivor is surfaced as a plain ``episode`` (full content, not the cue
snippet). The cue hit must still drive promotion feedback. These tests prove
the suppressed cue is collected at merge time and that the decoupled feedback
records the hit (and promotes after ``cue_recall_hit_threshold`` hits) without
mutating the surfaced episode payload.
"""

from __future__ import annotations

from typing import Any

import pytest

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.extraction.policy import ProjectionPolicy
from engram.models.episode import Episode, EpisodeProjectionState
from engram.models.episode_cue import EpisodeCue
from engram.retrieval.control import RecallNeedController
from engram.retrieval.feedback import (
    RecallCueFeedbackRecorder,
    RecallEntityAccessRecorder,
    RecallInteractionRecorder,
)
from engram.retrieval.pipeline import _merge_special_results, retrieve
from engram.retrieval.primary_results import RecallPrimaryResultMaterializer
from engram.retrieval.result_builder import RecallResultBuilder
from engram.retrieval.scorer import ScoredResult
from engram.retrieval.service import RecallService
from engram.retrieval.working_memory import RecallWorkingMemoryUpdater

EPISODE_ID = "ep_collide"


def _episode_candidate(node_id: str, score: float) -> ScoredResult:
    return ScoredResult(
        node_id=node_id,
        score=score,
        semantic_similarity=score,
        activation=0.0,
        spreading=0.0,
        edge_proximity=0.0,
        result_type="episode",
    )


def _cue_candidate(node_id: str, score: float) -> ScoredResult:
    return ScoredResult(
        node_id=node_id,
        score=score,
        semantic_similarity=score,
        activation=0.0,
        spreading=0.0,
        edge_proximity=0.0,
        result_type="cue_episode",
    )


def test_merge_collects_suppressed_cue_when_episode_outscores() -> None:
    """The dropped cue (episode outscored it) is stashed for decoupled feedback."""
    cfg = ActivationConfig(cue_recall_enabled=True, episode_retrieval_enabled=True)
    episode_candidates = [_episode_candidate(EPISODE_ID, score=0.9)]
    cue_candidates = [_cue_candidate(EPISODE_ID, score=0.5)]
    suppressed: dict[str, float] = {}

    special = _merge_special_results(episode_candidates, cue_candidates, cfg, suppressed)

    # Survivor stays surfaced as a plain episode — content NOT swapped for cue.
    assert len(special) == 1
    assert special[0].node_id == EPISODE_ID
    assert special[0].result_type == "episode"
    # Suppressed cue collected with its (discounted) cue score for feedback.
    assert suppressed == {EPISODE_ID: 0.5}


def test_merge_does_not_collect_when_cue_wins() -> None:
    """When the cue outscores the episode, the cue survives and nothing leaks."""
    cfg = ActivationConfig(cue_recall_enabled=True, episode_retrieval_enabled=True)
    episode_candidates = [_episode_candidate(EPISODE_ID, score=0.4)]
    cue_candidates = [_cue_candidate(EPISODE_ID, score=0.9)]
    suppressed: dict[str, float] = {}

    special = _merge_special_results(episode_candidates, cue_candidates, cfg, suppressed)

    assert len(special) == 1
    assert special[0].result_type == "cue_episode"
    assert suppressed == {}


class _FakeSearchIndex:
    """Single-pool search index where episode beats its colliding cue."""

    async def search(self, *, query: str, group_id: str, limit: int) -> list:
        return [("ent_1", 0.2)]

    async def search_episodes(self, *, query: str, group_id: str, limit: int) -> list:
        return [(EPISODE_ID, 0.9)]

    async def search_episode_cues(self, *, query: str, group_id: str, limit: int) -> list:
        return [(EPISODE_ID, 0.5)]

    async def compute_similarity(
        self, *, query: str, entity_ids: list, group_id: str, **kwargs: Any
    ) -> dict:
        return {eid: 0.0 for eid in entity_ids}


class _FakeActivationStore:
    async def batch_get(self, entity_ids: list) -> dict:
        return {}

    async def get_top_activated(self, **kwargs: Any) -> list:
        return []

    async def record_access(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def get_activation(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FakeGraphStore:
    """Minimal graph store holding one unprojected episode plus its cue."""

    def __init__(self) -> None:
        self.episode = Episode(
            id=EPISODE_ID,
            content="The migration to native Helix finished on Tuesday.",
            group_id="g1",
            projection_state=EpisodeProjectionState.CUE_ONLY,
        )
        self.cue = EpisodeCue(
            episode_id=EPISODE_ID,
            group_id="g1",
            projection_state=EpisodeProjectionState.CUE_ONLY,
            cue_text="native Helix migration",
            hit_count=0,
        )
        self.episode_updates: list[dict] = []
        self.cue_updates: list[dict] = []

    async def get_stats(self, group_id: str, **kwargs: Any) -> dict:
        return {"entity_count": 1}

    async def get_active_neighbors_with_weights(self, **kwargs: Any) -> list:
        return []

    async def get_entity(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def get_relationships(self, *args: Any, **kwargs: Any) -> list:
        return []

    async def get_episode_by_id(self, episode_id: str, group_id: str) -> Episode | None:
        return self.episode if episode_id == EPISODE_ID else None

    async def get_episode_entities(self, episode_id: str, *, group_id: str) -> list:
        return []

    async def get_episode_cue(self, episode_id: str, group_id: str) -> EpisodeCue | None:
        return self.cue if episode_id == EPISODE_ID else None

    async def update_episode_cue(self, episode_id: str, updates: dict, *, group_id: str) -> None:
        self.cue_updates.append(dict(updates))
        if "hit_count" in updates:
            self.cue.hit_count = int(updates["hit_count"])
        state = updates.get("projection_state")
        if state is not None:
            self.cue.projection_state = state

    async def update_episode(self, episode_id: str, updates: dict, *, group_id: str) -> None:
        self.episode_updates.append(dict(updates))
        new_state = updates.get("projection_state")
        if new_state is not None:
            self.episode.projection_state = EpisodeProjectionState(new_state)


def _build_service(graph: _FakeGraphStore, cfg: ActivationConfig) -> RecallService:
    event_bus = EventBus()
    recorder = RecallCueFeedbackRecorder(
        cfg=cfg,
        graph_store=graph,
        projection_policy=ProjectionPolicy(cfg),
        recall_need_controller=RecallNeedController(cfg),
        event_bus=event_bus,
    )
    materializer = RecallPrimaryResultMaterializer(
        graph_store=graph,
        result_builder=RecallResultBuilder(cfg),
        cue_feedback_recorder=recorder,
        entity_access_recorder=RecallEntityAccessRecorder(
            cfg=cfg,
            activation_store=_FakeActivationStore(),
            event_bus=event_bus,
            labile_tracker=None,
        ),
        interaction_recorder=RecallInteractionRecorder(
            cfg=cfg,
            event_bus=event_bus,
            recall_need_controller=RecallNeedController(cfg),
        ),
        working_memory_updater=RecallWorkingMemoryUpdater(),
    )

    class _PassThroughPostProcessor:
        async def process(self, results: list, **kwargs: Any):
            from engram.retrieval.post_process import RecallPostProcessResult

            return RecallPostProcessResult(results=results, near_misses=[])

    return RecallService(
        graph_store=graph,
        activation_store=_FakeActivationStore(),
        search_index=_FakeSearchIndex(),
        cfg=cfg,
        primary_materializer=materializer,
        post_processor=_PassThroughPostProcessor(),
    )


def _quiet_recall_config() -> ActivationConfig:
    return ActivationConfig(
        multi_pool_enabled=False,
        graph_query_expansion_enabled=False,
        template_reformulation_enabled=False,
        query_decomposition_enabled=False,
        recall_planner_enabled=False,
        reranker_enabled=False,
        mmr_enabled=False,
        gc_mmr_enabled=False,
        ts_enabled=False,
        working_memory_enabled=False,
        goal_priming_enabled=False,
        inhibitory_spreading_enabled=False,
        cross_domain_penalty_enabled=False,
        emotional_salience_enabled=False,
        state_dependent_retrieval_enabled=False,
        preference_directed_enabled=False,
        conv_near_miss_enabled=False,
        chunk_search_enabled=False,
        weight_graph_structural=0.0,
        cue_recall_enabled=True,
        episode_retrieval_enabled=True,
        cue_recall_hit_threshold=2,
    )


@pytest.mark.asyncio
async def test_recall_records_cue_feedback_when_episode_outscores_cue() -> None:
    """End-to-end: episode surfaced unchanged, suppressed cue hit still recorded."""
    cfg = _quiet_recall_config()
    graph = _FakeGraphStore()
    service = _build_service(graph, cfg)

    result = await service.recall(
        query="native Helix migration",
        group_id="g1",
        limit=5,
        record_access=False,
        interaction_type="surfaced",
        interaction_source="auto_recall",
        conv_context=None,
        working_memory=None,
        priming_buffer={},
        goal_cache=None,
        memory_need=None,
    )

    # The episode is surfaced as a plain episode with full content (NOT a cue).
    episode_results = [r for r in result.results if r.get("result_type") == "episode"]
    assert episode_results, "episode should surface as a plain episode result"
    surfaced = next(r for r in episode_results if r["episode"]["id"] == EPISODE_ID)
    assert surfaced["result_type"] == "episode"
    assert "content" in surfaced["episode"]
    assert "cue" not in surfaced
    # The suppressed cue hit was recorded (decoupled from surfacing).
    assert graph.cue.hit_count == 1


@pytest.mark.asyncio
async def test_two_recalls_promote_episode_to_scheduled() -> None:
    """Two cue hits reach cue_recall_hit_threshold=2 and promote to SCHEDULED."""
    cfg = _quiet_recall_config()
    graph = _FakeGraphStore()
    service = _build_service(graph, cfg)

    async def _recall() -> None:
        await service.recall(
            query="native Helix migration",
            group_id="g1",
            limit=5,
            record_access=False,
            interaction_type="surfaced",
            interaction_source="auto_recall",
            conv_context=None,
            working_memory=None,
            priming_buffer={},
            goal_cache=None,
            memory_need=None,
        )

    await _recall()
    assert graph.episode.projection_state == EpisodeProjectionState.CUE_ONLY
    await _recall()
    assert graph.cue.hit_count == 2
    assert graph.episode.projection_state == EpisodeProjectionState.SCHEDULED


@pytest.mark.asyncio
async def test_retrieve_pipeline_stashes_suppressed_cue() -> None:
    """The real retrieve() pipeline populates suppressed_cue_out at merge time."""
    cfg = _quiet_recall_config()
    graph = _FakeGraphStore()
    suppressed: dict[str, float] = {}

    results = await retrieve(
        query="native Helix migration",
        group_id="g1",
        graph_store=graph,
        activation_store=_FakeActivationStore(),
        search_index=_FakeSearchIndex(),
        cfg=cfg,
        limit=5,
        enable_routing=False,
        suppressed_cue_out=suppressed,
    )

    assert EPISODE_ID in suppressed
    surfaced = [r for r in results if r.node_id == EPISODE_ID]
    assert surfaced and surfaced[0].result_type == "episode"
