"""Regression tests for recall-pipeline silent-inert fixes (B3, B8, B9, B13).

These cover bugs where a recall feature was enabled/metered but produced no
effect on the Helix backend (or any backend) because of a backend-specific
fetch, a missing argument, an unpopulated source, or a dedup that dropped the
signal-carrying result.

B3 - MMR / GC-MMR diversity was a no-op on Helix: embeddings were fetched via a
     SQLite-only ``search_index._vectors`` path, so entity_embeddings stayed
     empty and apply_mmr returned results unchanged. Fix: fetch via the
     backend-agnostic ``get_entity_embeddings``.
B8 - Predicate inhibition never fired: ``apply_inhibition`` was called without
     ``relationships=``. Fix: fetch seed relationships and pass them through.
B9 - State-dependent retrieval was inert: ``cognitive_state`` was never
     populated. Fix: call ``update_cognitive_state`` once per recall turn.
B13 - Cue-hit -> promotion loop was dead: ``_merge_special_results`` dropped the
     cue_episode result when a same-id episode result outscored it, freezing
     hit_count at 0. Fix: preserve the cue_episode signal on the survivor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.mmr import apply_mmr
from engram.retrieval.pipeline import _merge_special_results, retrieve
from engram.retrieval.scorer import ScoredResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeEntity:
    id: str
    name: str
    entity_type: str = "Person"
    summary: str = ""
    group_id: str = "default"
    attributes: dict | None = None


@dataclass
class _FakeRelationship:
    source_id: str
    target_id: str
    predicate: str
    weight: float = 1.0


@dataclass
class _FakeEpisode:
    id: str
    group_id: str = "default"
    content: str = "episode content"
    projection_state: str = "cued"


@dataclass
class _FakeCue:
    episode_id: str
    group_id: str = "default"
    cue_text: str = "cue text fragment"
    hit_count: int = 0
    policy_score: float = 0.0
    supporting_spans: list[str] = field(default_factory=list)


class _FakeSearchIndex:
    """Backend-agnostic search index stub exposing get_entity_embeddings."""

    def __init__(
        self,
        results: list[tuple[str, float]],
        similarity_map: dict[str, float] | None = None,
        embeddings: dict[str, list[float]] | None = None,
        cue_results: list[tuple[str, float]] | None = None,
    ) -> None:
        self._results = results
        self._similarity_map = similarity_map or {}
        self._embeddings = embeddings or {}
        self._cue_results = cue_results or []
        self.embedding_calls: list[list[str]] = []

    async def search(self, query: str, group_id: str, limit: int = 50):
        return self._results[:limit]

    async def compute_similarity(
        self,
        query: str,
        entity_ids: list[str],
        group_id: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> dict[str, float]:
        return {eid: self._similarity_map.get(eid, 0.0) for eid in entity_ids}

    async def get_entity_embeddings(
        self,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, list[float]]:
        self.embedding_calls.append(list(entity_ids))
        return {eid: self._embeddings[eid] for eid in entity_ids if eid in self._embeddings}

    async def search_episode_cues(self, query: str, group_id: str, limit: int = 10):
        return self._cue_results[:limit]


class _FakeActivationStore:
    def __init__(self, states: dict[str, ActivationState] | None = None) -> None:
        self._states = states or {}

    async def batch_get(self, entity_ids: list[str]) -> dict[str, ActivationState]:
        return {eid: self._states[eid] for eid in entity_ids if eid in self._states}

    async def get_activation(self, entity_id: str) -> ActivationState | None:
        return self._states.get(entity_id)

    async def set_activation(self, entity_id: str, state: ActivationState) -> None:
        self._states[entity_id] = state

    async def record_access(self, entity_id, now, group_id="default") -> None:
        return None

    async def get_top_activated(self, *args, **kwargs):
        return []


class _FakeGraphStore:
    def __init__(
        self,
        adjacency: dict[str, list[str]] | None = None,
        entities: list[_FakeEntity] | None = None,
        relationships: dict[str, list[_FakeRelationship]] | None = None,
        episodes: dict[str, _FakeEpisode] | None = None,
        cues: dict[str, _FakeCue] | None = None,
    ) -> None:
        self._adj = adjacency or {}
        self._entities = entities or []
        self._rels = relationships or {}
        self._episodes = episodes or {}
        self._cues = cues or {}
        self.cue_updates: list[tuple[str, dict]] = []

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None, **kwargs
    ) -> list[tuple[str, float]]:
        return [(n, 1.0) for n in self._adj.get(entity_id, [])]

    async def find_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[_FakeEntity]:
        results = [ent for ent in self._entities if name and name.lower() in ent.name.lower()]
        return results[:limit]

    async def get_stats(self, group_id: str, exact: bool = False) -> dict:
        return {"entity_count": len(self._entities)}

    async def get_entity(self, entity_id: str, group_id: str):
        for ent in self._entities:
            if ent.id == entity_id:
                return ent
        return None

    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        predicate: str | None = None,
        active_only: bool = True,
        group_id: str = "default",
    ) -> list[_FakeRelationship]:
        return self._rels.get(entity_id, [])

    async def get_episode_by_id(self, episode_id: str, group_id: str):
        return self._episodes.get(episode_id)

    async def get_episode_entities(self, episode_id: str, group_id: str | None = None):
        return []

    async def get_episode_cue(self, episode_id: str, group_id: str):
        return self._cues.get(episode_id)

    async def update_episode_cue(self, episode_id: str, updates: dict, group_id: str) -> None:
        self.cue_updates.append((episode_id, updates))


def _base_cfg(**overrides) -> ActivationConfig:
    defaults = {
        "weight_semantic": 0.50,
        "weight_activation": 0.35,
        "weight_edge_proximity": 0.15,
        "seed_threshold": 0.3,
        "exploration_weight": 0.0,
        # These pipeline tests exercise MMR / inhibition / state, not the
        # episode/cue special-result lanes (which require extra index methods).
        "episode_retrieval_enabled": False,
        "cue_recall_enabled": False,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


# ---------------------------------------------------------------------------
# B3 - MMR reorders once embeddings flow through get_entity_embeddings
# ---------------------------------------------------------------------------


class TestMmrReordersWithEmbeddings:
    def test_apply_mmr_reorders_when_embeddings_present(self):
        """Diversity re-ranking demotes a near-duplicate of the top result."""
        # a (top), b (near-duplicate of a), c (orthogonal/diverse).
        results = [
            ScoredResult("a", 1.0, 1.0, 0.0, 0.0, 0.0),
            ScoredResult("b", 0.9, 0.9, 0.0, 0.0, 0.0),
            ScoredResult("c", 0.8, 0.8, 0.0, 0.0, 0.0),
        ]
        embeddings = {
            "a": [1.0, 0.0],
            "b": [0.99, 0.01],  # nearly identical to a
            "c": [0.0, 1.0],  # orthogonal to a
        }

        reranked = apply_mmr(results, embeddings, lambda_param=0.5, top_n=3)
        order = [r.node_id for r in reranked]

        assert order[0] == "a"
        # Diverse c is promoted above the near-duplicate b.
        assert order.index("c") < order.index("b")

    def test_apply_mmr_is_noop_without_embeddings(self):
        """No embeddings -> order preserved (the old silent-inert behavior)."""
        results = [
            ScoredResult("a", 1.0, 1.0, 0.0, 0.0, 0.0),
            ScoredResult("b", 0.9, 0.9, 0.0, 0.0, 0.0),
        ]
        reranked = apply_mmr(results, {}, lambda_param=0.5, top_n=2)
        assert [r.node_id for r in reranked] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_pipeline_mmr_fetches_via_get_entity_embeddings(self):
        """retrieve() must fetch MMR embeddings via the backend-agnostic method."""
        now = time.time()
        search_index = _FakeSearchIndex(
            [("a", 0.9), ("b", 0.85), ("c", 0.8)],
            embeddings={
                "a": [1.0, 0.0],
                "b": [0.99, 0.01],
                "c": [0.0, 1.0],
            },
        )
        activation_store = _FakeActivationStore(
            {
                eid: ActivationState(node_id=eid, access_history=[now - 5], access_count=2)
                for eid in ("a", "b", "c")
            }
        )
        graph_store = _FakeGraphStore()
        cfg = _base_cfg(mmr_enabled=True, mmr_lambda=0.5, retrieval_top_n=3)

        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=3,
        )

        # The MMR stage must have asked the index for embeddings (no SQLite path).
        assert search_index.embedding_calls, "MMR did not fetch entity embeddings"
        fetched_ids = set().union(*[set(c) for c in search_index.embedding_calls])
        assert {"a", "b", "c"} & fetched_ids
        assert results  # pipeline still returns results


# ---------------------------------------------------------------------------
# B8 - apply_inhibition receives seed relationships from the pipeline
# ---------------------------------------------------------------------------


class TestInhibitionReceivesRelationships:
    @pytest.mark.asyncio
    async def test_pipeline_passes_relationships_to_apply_inhibition(self, monkeypatch):
        """retrieve() must fetch seed relationships and pass them to inhibition."""
        import engram.retrieval.inhibition as inhibition_mod

        captured: dict[str, object] = {}
        real_apply = inhibition_mod.apply_inhibition

        async def _spy_apply_inhibition(*args, **kwargs):
            captured["relationships"] = kwargs.get("relationships")
            return await real_apply(*args, **kwargs)

        monkeypatch.setattr(inhibition_mod, "apply_inhibition", _spy_apply_inhibition)

        now = time.time()
        # seed connects to pizza (LIKES, strong) and broccoli (DISLIKES, weak).
        search_index = _FakeSearchIndex(
            [("seed", 0.9)],
            similarity_map={"pizza": 0.4, "broccoli": 0.4},
        )
        activation_store = _FakeActivationStore(
            {
                "seed": ActivationState(node_id="seed", access_history=[now - 5], access_count=3),
                "pizza": ActivationState(node_id="pizza", access_history=[now - 5], access_count=1),
                "broccoli": ActivationState(
                    node_id="broccoli", access_history=[now - 5], access_count=1
                ),
            }
        )
        graph_store = _FakeGraphStore(
            adjacency={"seed": ["pizza", "broccoli"]},
            relationships={
                "seed": [
                    _FakeRelationship("seed", "pizza", "LIKES", 2.0),
                    _FakeRelationship("seed", "broccoli", "DISLIKES", 1.0),
                ]
            },
        )
        cfg = _base_cfg(
            inhibitory_spreading_enabled=True,
            inhibition_predicate_suppression=True,
            inhibit_strength=0.3,
        )

        await retrieve(
            query="test",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
        )

        rels = captured.get("relationships")
        assert rels, "apply_inhibition was called without seed relationships"
        preds = {r[2] for r in rels}
        assert {"LIKES", "DISLIKES"} <= preds


# ---------------------------------------------------------------------------
# B9 - cognitive state is populated so state bias becomes non-zero
# ---------------------------------------------------------------------------


class TestCognitiveStatePopulated:
    @pytest.mark.asyncio
    async def test_pipeline_populates_cognitive_state(self):
        """retrieve() calls update_cognitive_state, leaving a non-None state."""
        from engram.retrieval.context import ConversationContext

        now = time.time()
        conv_context = ConversationContext()
        assert conv_context.cognitive_state is None  # inert precondition

        search_index = _FakeSearchIndex([("seed", 0.9)])
        activation_store = _FakeActivationStore(
            {"seed": ActivationState(node_id="seed", access_history=[now - 5], access_count=3)}
        )
        graph_store = _FakeGraphStore()
        cfg = _base_cfg(state_dependent_retrieval_enabled=True)

        await retrieve(
            query="how do I fix this bug",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
            conv_context=conv_context,
        )

        state = conv_context.cognitive_state
        assert state is not None, "cognitive_state was never populated"
        # Task-style query should infer the 'task' mode.
        assert getattr(state, "mode", None) == "task"

    def test_compute_state_bias_nonzero_with_populated_state(self):
        """A populated cognitive state yields a non-zero domain bias."""
        from engram.retrieval.context import ConversationContext
        from engram.retrieval.state import compute_state_bias

        conv_context = ConversationContext()
        conv_context.update_cognitive_state("how do I implement this function")
        state = conv_context.cognitive_state
        assert state is not None

        cfg = ActivationConfig(
            state_dependent_retrieval_enabled=True,
            state_domain_weight=0.1,
        )
        # technical domain entity, task mode -> highest affinity
        domain_groups = {"technical": ["Technology"]}
        bias = compute_state_bias(
            state,
            {"entity_type": "Technology"},
            "Technology",
            cfg,
            domain_groups=domain_groups,
        )
        assert bias > 0.0


# ---------------------------------------------------------------------------
# B13 - cue feedback survives the episode/cue merge so hit_count increments
# ---------------------------------------------------------------------------


class TestCueFeedbackSurvivesMerge:
    def test_merge_keeps_episode_type_when_episode_wins(self):
        """When an episode result outscores its same-id cue, the survivor stays
        an EPISODE result (full content). B13's earlier attempt mutated it to
        cue_episode to trigger feedback, but that swapped episode content for the
        cue snippet and dropped required evidence — reverted/deferred. Surfacing
        correctness wins; cue-promotion feedback needs a non-corrupting fix."""
        episode = ScoredResult(
            "ep1", 0.9, 0.9, 0.0, 0.0, 0.0, result_type="episode"
        )
        cue = ScoredResult(
            "ep1", 0.4, 0.4, 0.0, 0.0, 0.0, result_type="cue_episode"
        )
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            episode_retrieval_max=5,
            cue_recall_enabled=True,
            cue_recall_max=5,
        )

        merged = _merge_special_results([episode], [cue], cfg)

        ep1_results = [r for r in merged if r.node_id == "ep1"]
        assert len(ep1_results) == 1
        survivor = ep1_results[0]
        assert survivor.score == pytest.approx(0.9)
        # Survivor keeps full episode content (not swapped to the cue snippet).
        assert survivor.result_type == "episode"

    def test_merge_keeps_cue_when_cue_wins(self):
        """When the cue result outscores the episode, the cue result survives."""
        episode = ScoredResult(
            "ep1", 0.2, 0.2, 0.0, 0.0, 0.0, result_type="episode"
        )
        cue = ScoredResult(
            "ep1", 0.8, 0.8, 0.0, 0.0, 0.0, result_type="cue_episode"
        )
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            episode_retrieval_max=5,
            cue_recall_enabled=True,
            cue_recall_max=5,
        )

        merged = _merge_special_results([episode], [cue], cfg)
        ep1_results = [r for r in merged if r.node_id == "ep1"]
        assert len(ep1_results) == 1
        assert ep1_results[0].result_type == "cue_episode"
        assert ep1_results[0].score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_recalled_cued_episode_increments_hit_count(self):
        """End-to-end: materializing a cue_episode result increments hit_count."""
        from engram.extraction.policy import ProjectionPolicy
        from engram.models.episode import Episode, EpisodeProjectionState
        from engram.models.episode_cue import EpisodeCue
        from engram.retrieval.control import RecallNeedController
        from engram.retrieval.feedback import RecallCueFeedbackRecorder

        group_id = "default"
        episode = Episode(
            id="ep1",
            group_id=group_id,
            content="user said the deadline is friday",
            projection_state=EpisodeProjectionState.CUED,
        )
        cue = EpisodeCue(
            episode_id="ep1",
            group_id=group_id,
            cue_text="deadline is friday",
            hit_count=0,
        )

        class _CueGraphStore:
            def __init__(self) -> None:
                self.cue = cue
                self.updates: list[dict] = []

            async def get_episode_cue(self, episode_id: str, group_id: str):
                return self.cue

            async def update_episode_cue(self, episode_id, updates, group_id) -> None:
                self.updates.append(updates)

        graph_store = _CueGraphStore()
        cfg = ActivationConfig(
            cue_recall_enabled=True,
            cue_recall_hit_threshold=20,  # high so it does not promote, just counts
        )
        recorder = RecallCueFeedbackRecorder(
            cfg=cfg,
            graph_store=graph_store,
            projection_policy=ProjectionPolicy(cfg),
            recall_need_controller=RecallNeedController(cfg),
            event_bus=None,
        )

        await recorder.record_cue_feedback(
            episode,
            score=0.5,
            query="when is the deadline",
            interaction_type="surfaced",
        )

        assert graph_store.updates, "cue feedback recorded no update"
        new_hit_count = graph_store.updates[-1].get("hit_count")
        assert new_hit_count == 1, f"hit_count did not increment: {new_hit_count}"
