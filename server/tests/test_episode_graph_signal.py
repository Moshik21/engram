"""GAP A + GAP B: graph signal on the answer-bearing (episode) channel.

Two mechanisms, deliberately tested as separate arms:

* GAP B — the FORWARD map (``episode -> HasEntity -> entity signal``), a
  RERANK mechanism that runs inside the top-N truncation and can reorder
  episodes the vector/BM25 lanes already found.
* GAP A — ``entity_episode_traversal_source='candidates'``, a RECALL mechanism
  that runs past the truncation and can only append.

Every "it works" test here is paired with an anti-inertness assertion: a value
that is computed and then discarded, or a term that fires and changes no
ordering, is the failure mode this codebase has paid for six times. A test that
passes while the feature is inert is worse than no test.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.retrieval.episode_graph_signal import (
    EntityGraphSignal,
    derive_episode_signal,
    snapshot_entity_signal,
)
from engram.retrieval.episode_traversal import RecallEpisodeTraversal
from engram.retrieval.pipeline import retrieve
from engram.retrieval.result_builder import RecallResultBuilder
from engram.retrieval.scorer import ScoredResult
from engram.utils.dates import utc_now

# ── Fixtures ────────────────────────────────────────────────────────


def _active_state(node_id: str, accesses: int) -> ActivationState:
    now = time.time()
    history = [now - (i + 1) * 3600.0 for i in range(accesses)]
    return ActivationState(
        node_id=node_id,
        access_history=history,
        last_accessed=max(history),
        access_count=accesses,
    )


def _activation_store(states: dict[str, ActivationState] | None = None):
    store = AsyncMock()
    store.batch_get = AsyncMock(return_value=states or {})
    store.get_activation = AsyncMock(return_value=None)
    store.set_activation = AsyncMock()
    store.record_access = AsyncMock()
    store.get_top_activated = AsyncMock(return_value=[])
    return store


def _graph_store(episode_entities: dict[str, list[str]] | None = None):
    """Graph store whose only interesting behaviour is the forward HasEntity read."""
    store = AsyncMock()
    store.get_active_neighbors_with_weights = AsyncMock(return_value=[])
    store.update_episode = AsyncMock()
    store.update_episode_cue = AsyncMock()
    store.get_entity = AsyncMock(
        return_value=Entity(
            id="e1",
            name="Test",
            entity_type="Thing",
            summary="A test entity",
            group_id="default",
        )
    )
    store.get_episode_by_id = AsyncMock(
        return_value=Episode(
            id="ep_1",
            content="Test episode content that is quite long",
            source="test",
            status=EpisodeStatus.COMPLETED,
            projection_state=EpisodeProjectionState.PROJECTED,
            group_id="default",
            created_at=utc_now(),
        )
    )
    mapping = episode_entities if episode_entities is not None else {}

    async def _get_episode_entities(episode_id, group_id=None):
        return list(mapping.get(episode_id, []))

    store.get_episode_entities = AsyncMock(side_effect=_get_episode_entities)
    store.get_episodes_for_entity = AsyncMock(return_value=[])
    return store


def _search_index(entity_results=None, episode_results=None, cue_results=None):
    idx = AsyncMock()
    idx.search = AsyncMock(
        return_value=entity_results if entity_results is not None else [("e1", 0.9), ("e2", 0.7)],
    )
    idx.search_episodes = AsyncMock(
        return_value=(
            episode_results if episode_results is not None else [("ep_1", 0.8), ("ep_2", 0.79)]
        ),
    )
    idx.search_episode_cues = AsyncMock(return_value=cue_results if cue_results is not None else [])
    idx.compute_similarity = AsyncMock(return_value={})
    idx._embeddings_enabled = False
    return idx


def _cfg(**overrides) -> ActivationConfig:
    base = {
        "episode_retrieval_enabled": True,
        "episode_retrieval_max": 5,
        "mmr_enabled": False,
    }
    base.update(overrides)
    return ActivationConfig(**base)


async def _run(cfg, *, graph_store=None, activation_store=None, search_index=None, timings=None):
    return await retrieve(
        query="test query",
        group_id="default",
        graph_store=graph_store or _graph_store(),
        activation_store=activation_store or _activation_store(),
        search_index=search_index or _search_index(),
        cfg=cfg,
        stage_timings_ms=timings if timings is not None else {},
    )


# ── Derivation unit tests ───────────────────────────────────────────


class TestDerivation:
    def test_no_matched_entity_returns_none(self):
        """Additive-only: an episode linking to nothing scored keeps its score."""
        cfg = _cfg()
        assert (
            derive_episode_signal(["unknown"], {"e1": EntityGraphSignal(0.9, 0.5, 0)}, cfg) is None
        )

    def test_activation_is_max_over_linked_times_hop_decay(self):
        cfg = _cfg(episode_graph_signal_hop_decay=0.5)
        signal = derive_episode_signal(
            ["e1", "e2"],
            {
                "e1": EntityGraphSignal(0.2, 0.0, None),
                "e2": EntityGraphSignal(0.8, 0.0, None),
            },
            cfg,
        )
        assert signal is not None
        assert signal.activation == pytest.approx(0.4)

    def test_edge_proximity_is_derived_from_hop_not_inherited(self):
        """The single most dangerous line in the design.

        ``score_candidates`` gives every SEED entity ``edge_proximity = 1.0``,
        and seeds are picked by ``sem_sim >= seed_threshold`` — so inheriting
        the parent's edge_proximity would hand episodes a *semantic* signal
        wearing a *graph* label. The derived value must come from hop distance
        plus the membership hop instead.
        """
        cfg = _cfg(episode_graph_signal_hop_decay=0.5)
        seed_parent = derive_episode_signal(["e1"], {"e1": EntityGraphSignal(0.0, 0.0, 0)}, cfg)
        assert seed_parent is not None
        # NOT 1.0 (the parent's own seed indicator): one real HasEntity hop.
        assert seed_parent.edge_proximity == pytest.approx(0.5)
        assert seed_parent.min_hop == 0

        two_hops = derive_episode_signal(["e1"], {"e1": EntityGraphSignal(0.0, 0.0, 2)}, cfg)
        assert two_hops is not None
        assert two_hops.edge_proximity == pytest.approx(0.125)

    def test_edge_proximity_zero_when_no_hop_information(self):
        """Spreading dead (hop_distances empty) and the entity was not a seed."""
        cfg = _cfg()
        signal = derive_episode_signal(["e1"], {"e1": EntityGraphSignal(0.7, 0.0, None)}, cfg)
        assert signal is not None
        assert signal.edge_proximity == 0.0
        assert signal.min_hop is None
        assert signal.activation > 0.0

    def test_snapshot_captures_entities_only(self):
        scored = [
            ScoredResult("e1", 0.9, 0.9, 0.5, 0.2, 1.0, hop_distance=0),
            ScoredResult("ep_1", 0.4, 1.0, 0.0, 0.0, 0.0, result_type="episode"),
        ]
        snap = snapshot_entity_signal(scored)
        assert set(snap) == {"e1"}
        assert snap["e1"] == EntityGraphSignal(0.5, 0.2, 0)


# ── GAP B: the pipeline stage ───────────────────────────────────────


class TestEpisodeGraphSignalFlagOff:
    @pytest.mark.asyncio
    async def test_default_is_off(self):
        assert ActivationConfig().episode_graph_signal_enabled is False
        assert ActivationConfig().episode_graph_signal_weight == 0.0
        assert ActivationConfig().episode_graph_signal_source == "activation"

    @pytest.mark.asyncio
    async def test_flag_off_leaves_literal_zeros_and_emits_no_metrics(self):
        """Reproduces today's shipped behaviour: 0.0 on every episode row."""
        timings: dict[str, float] = {}
        results = await _run(
            _cfg(),
            graph_store=_graph_store({"ep_1": ["e1"], "ep_2": ["e2"]}),
            activation_store=_activation_store({"e1": _active_state("e1", 12)}),
            timings=timings,
        )
        episodes = [r for r in results if r.result_type == "episode"]
        assert episodes
        for sr in episodes:
            assert sr.activation == 0.0
            assert sr.spreading == 0.0
            assert sr.edge_proximity == 0.0
        assert not [k for k in timings if k.startswith("recall_episode_graph_signal")]


class TestEpisodeGraphSignalFlagOn:
    @pytest.mark.asyncio
    async def test_writes_nonzero_activation_onto_episodes(self):
        """The direct falsifier of GRAPH_THESIS M3/M4 (0.0 on 55/55 live rows)."""
        timings: dict[str, float] = {}
        results = await _run(
            _cfg(episode_graph_signal_enabled=True, episode_graph_signal_weight=0.25),
            graph_store=_graph_store({"ep_1": ["e1"], "ep_2": ["e2"]}),
            activation_store=_activation_store(
                {"e1": _active_state("e1", 20), "e2": _active_state("e2", 20)}
            ),
            timings=timings,
        )
        episodes = [r for r in results if r.result_type == "episode"]
        assert episodes
        assert any(sr.activation > 0.0 for sr in episodes)
        assert timings["recall_episode_graph_signal_covered"] > 0
        assert timings["recall_episode_graph_signal_applied"] > 0
        assert timings["recall_episode_graph_signal_max"] > 0

    @pytest.mark.asyncio
    async def test_graph_signal_reorders_episodes_and_edgeless_control_does_not(self):
        """THE anti-inertness test.

        Assertion (a) — some episode carries non-zero graph signal — passes for
        a value that is computed and then discarded by any of the seven known
        discard paths. Assertion (b) — the returned ORDER differs from an
        edge-free control run on identical inputs — cannot.
        """
        ep_hits = [("ep_lead", 0.80), ("ep_trail", 0.79)]
        states = {"e_hot": _active_state("e_hot", 40)}

        edgeless_timings: dict[str, float] = {}
        edgeless = await _run(
            _cfg(episode_graph_signal_enabled=True, episode_graph_signal_weight=0.25),
            graph_store=_graph_store({}),  # brain with no HasEntity edges
            activation_store=_activation_store(states),
            search_index=_search_index(entity_results=[("e_hot", 0.9)], episode_results=ep_hits),
            timings=edgeless_timings,
        )

        edged_timings: dict[str, float] = {}
        edged = await _run(
            _cfg(episode_graph_signal_enabled=True, episode_graph_signal_weight=0.25),
            # Only the TRAILING episode is linked into the graph.
            graph_store=_graph_store({"ep_trail": ["e_hot"]}),
            activation_store=_activation_store(states),
            search_index=_search_index(entity_results=[("e_hot", 0.9)], episode_results=ep_hits),
            timings=edged_timings,
        )

        edgeless_order = [r.node_id for r in edgeless if r.result_type == "episode"]
        edged_order = [r.node_id for r in edged if r.result_type == "episode"]

        assert edgeless_order == ["ep_lead", "ep_trail"]
        assert edged_order == ["ep_trail", "ep_lead"], "graph term did not reorder"
        assert edged_order != edgeless_order

        # The edge-free control proves the edges were load-bearing.
        assert edgeless_timings["recall_episode_graph_signal_covered"] == 0
        assert edgeless_timings["recall_episode_graph_signal_pool"] > 0
        assert edged_timings["recall_episode_graph_signal_covered"] == 1
        assert edged_timings["recall_episode_graph_signal_reorders"] > 0

    @pytest.mark.asyncio
    async def test_edgeless_brain_leaves_scores_untouched(self):
        """Additive-only: no linked entity means byte-identical to flag-off."""
        off = await _run(
            _cfg(),
            graph_store=_graph_store({}),
            activation_store=_activation_store({"e1": _active_state("e1", 20)}),
        )
        on = await _run(
            _cfg(episode_graph_signal_enabled=True, episode_graph_signal_weight=0.25),
            graph_store=_graph_store({}),
            activation_store=_activation_store({"e1": _active_state("e1", 20)}),
        )
        assert [(r.node_id, r.score) for r in off] == [(r.node_id, r.score) for r in on]

    @pytest.mark.asyncio
    async def test_unlinked_episode_is_never_penalised(self):
        """46.7% of live episodes have no HasEntity edge; they must not lose."""
        cfg = _cfg(episode_graph_signal_enabled=True, episode_graph_signal_weight=0.25)
        baseline = await _run(
            _cfg(),
            graph_store=_graph_store({"ep_1": ["e1"]}),
            activation_store=_activation_store({"e1": _active_state("e1", 20)}),
        )
        boosted = await _run(
            cfg,
            graph_store=_graph_store({"ep_1": ["e1"]}),
            activation_store=_activation_store({"e1": _active_state("e1", 20)}),
        )
        base_scores = {r.node_id: r.score for r in baseline}
        for sr in boosted:
            assert sr.score >= base_scores.get(sr.node_id, 0.0) - 1e-9

    @pytest.mark.asyncio
    async def test_source_gate_excludes_terms_from_score_but_still_writes_them(self):
        """An inert signal must be VISIBLE as a zero, not absent."""
        graph = _graph_store({"ep_1": ["e1"]})
        states = {"e1": _active_state("e1", 30)}
        idx = _search_index(entity_results=[("e1", 0.95)], episode_results=[("ep_1", 0.8)])

        activation_only = await _run(
            _cfg(
                episode_graph_signal_enabled=True,
                episode_graph_signal_weight=0.25,
                episode_graph_signal_source="activation",
            ),
            graph_store=graph,
            activation_store=_activation_store(states),
            search_index=idx,
        )
        full = await _run(
            _cfg(
                episode_graph_signal_enabled=True,
                episode_graph_signal_weight=0.25,
                episode_graph_signal_source="full",
            ),
            graph_store=graph,
            activation_store=_activation_store(states),
            search_index=idx,
        )
        a_ep = next(r for r in activation_only if r.result_type == "episode")
        f_ep = next(r for r in full if r.result_type == "episode")
        # edge_proximity is WRITTEN in both, but only scored under 'full'.
        assert a_ep.edge_proximity == f_ep.edge_proximity
        assert f_ep.score > a_ep.score

    @pytest.mark.asyncio
    async def test_repeat_recall_is_score_stable(self):
        """Episode-lane determinism guard (the entity lane's guard disables it)."""
        kwargs = {
            "graph_store": _graph_store({"ep_1": ["e1"], "ep_2": ["e1"]}),
            "activation_store": _activation_store({"e1": _active_state("e1", 20)}),
        }
        cfg = _cfg(episode_graph_signal_enabled=True, episode_graph_signal_weight=0.25)
        first = await _run(cfg, **kwargs)
        second = await _run(cfg, **kwargs)
        assert [(r.node_id, r.score) for r in first] == [(r.node_id, r.score) for r in second]


class TestEpisodeGraphSignalDegradation:
    @pytest.mark.asyncio
    async def test_probe_timeout_gate_records_a_distinct_metric(self):
        """A GATED zero must never be confused with an EDGELESS zero."""
        timings: dict[str, float] = {"recall_stats_timeout": 1.0}
        results = await _run(
            _cfg(episode_graph_signal_enabled=True, episode_graph_signal_weight=0.25),
            graph_store=_graph_store({"ep_1": ["e1"], "ep_2": ["e1"]}),
            activation_store=_activation_store({"e1": _active_state("e1", 20)}),
            timings=timings,
        )
        assert "recall_episode_graph_signal_skipped_probe_timeout" in timings
        assert "recall_episode_graph_signal_covered" not in timings
        for sr in results:
            if sr.result_type == "episode":
                assert sr.activation == 0.0

    @pytest.mark.asyncio
    async def test_read_timeout_leaves_ranking_unchanged(self):
        graph = _graph_store({"ep_1": ["e1"], "ep_2": ["e1"]})

        async def _slow(_episode_id, group_id=None):
            await asyncio.sleep(0.2)
            return ["e1"]

        graph.get_episode_entities = AsyncMock(side_effect=_slow)
        timings: dict[str, float] = {}
        results = await _run(
            _cfg(
                episode_graph_signal_enabled=True,
                episode_graph_signal_weight=0.25,
                episode_graph_signal_timeout_ms=20,
            ),
            graph_store=graph,
            activation_store=_activation_store({"e1": _active_state("e1", 20)}),
            timings=timings,
        )
        assert "recall_episode_graph_signal_timeout" in timings
        assert "recall_episode_graph_signal_applied" not in timings
        for sr in results:
            if sr.result_type == "episode":
                assert sr.activation == 0.0

    @pytest.mark.asyncio
    async def test_empty_entity_channel_records_a_number_not_silence(self):
        """The `if not candidates:` early return (open task #20)."""
        timings: dict[str, float] = {}
        await _run(
            _cfg(episode_graph_signal_enabled=True, episode_graph_signal_weight=0.25),
            search_index=_search_index(entity_results=[]),
            timings=timings,
        )
        assert "recall_episode_graph_signal_no_entity_candidates" in timings


class TestEntitySignalSnapshotPlacement:
    @pytest.mark.asyncio
    async def test_snapshot_survives_mmr_truncation(self):
        """MMR REPLACES `scored` with a top_n=10 diversity-reordered list.

        A snapshot taken after MMR would silently shrink the signal source and
        measure a weaker effect for a reason unrelated to the graph.
        """
        entity_hits = [(f"e{i}", 0.9 - i * 0.01) for i in range(24)]
        signal_out: dict[str, EntityGraphSignal] = {}
        await retrieve(
            query="test query",
            group_id="default",
            graph_store=_graph_store({}),
            activation_store=_activation_store({}),
            search_index=_search_index(entity_results=entity_hits),
            cfg=_cfg(mmr_enabled=True, retrieval_top_n=10),
            stage_timings_ms={},
            entity_signal_out=signal_out,
        )
        assert len(signal_out) == 24


# ── GAP A: entity -> episode traversal ──────────────────────────────


def _traversal(cfg, *, episodes_for_entity, episode_entities):
    graph = AsyncMock()
    graph.get_episodes_for_entity = AsyncMock(return_value=list(episodes_for_entity))
    graph.get_episode_by_id = AsyncMock(
        return_value=Episode(
            id="ep_linked",
            content="An episode reachable only through the graph",
            source="test",
            status=EpisodeStatus.COMPLETED,
            projection_state=EpisodeProjectionState.PROJECTED,
            group_id="default",
            created_at=utc_now(),
        )
    )
    graph.get_episode_entities = AsyncMock(return_value=list(episode_entities))
    return RecallEpisodeTraversal(
        graph_store=graph,
        cfg=cfg,
        result_builder=RecallResultBuilder(cfg),
    )


class TestEntityEpisodeTraversalSource:
    def test_default_source_is_still_results(self):
        """Do NOT silently flip this: 'candidates' is an unmeasured default."""
        assert ActivationConfig().entity_episode_traversal_source == "results"

    @pytest.mark.asyncio
    async def test_default_source_is_inert_when_no_entity_wins_a_slot(self):
        """Pins the inertness of the shipped default.

        ``passage_first_entity_budget=0`` leaves the final results with zero
        entity rows, so 'results' mode has nothing to traverse FROM — the
        traversal runs as a no-op loop even though the graph has episodes.
        """
        cfg = ActivationConfig(entity_episode_traversal_source="results")
        traversal = _traversal(cfg, episodes_for_entity=["ep_linked"], episode_entities=["e_hot"])
        results: list[dict] = [
            {"result_type": "episode", "episode": {"id": "ep_seen"}, "score": 0.4}
        ]
        await traversal.append_entity_linked_episodes(
            results,
            group_id="default",
            seen_episode_ids={"ep_seen"},
            candidate_entity_scores=[("e_hot", 0.9)],
        )
        assert len(results) == 1
        traversal._graph.get_episodes_for_entity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_candidates_source_appends_graph_only_episodes(self):
        """Fails loudly if anyone re-breaks the 'candidates' path."""
        cfg = ActivationConfig(entity_episode_traversal_source="candidates")
        traversal = _traversal(cfg, episodes_for_entity=["ep_linked"], episode_entities=["e_hot"])
        results: list[dict] = [
            {"result_type": "episode", "episode": {"id": "ep_seen"}, "score": 0.4}
        ]
        await traversal.append_entity_linked_episodes(
            results,
            group_id="default",
            seen_episode_ids={"ep_seen"},
            candidate_entity_scores=[("e_hot", 0.9)],
        )
        assert len(results) == 2
        appended = results[1]
        assert appended["episode"]["id"] == "ep_linked"
        assert appended["score_breakdown"]["entity_traversal"] is True
        assert appended["score_breakdown"]["parent_entity_id"] == "e_hot"

    @pytest.mark.asyncio
    async def test_traversal_breakdown_carries_derived_signal_not_hardcoded_zeros(self):
        """The traversal used to report edge_proximity 0.0 on its OWN output.

        That is a false negative by construction: the mechanism whose entire
        purpose is surfacing graph-linked episodes claimed no graph signal
        reached them.
        """
        cfg = ActivationConfig(
            entity_episode_traversal_source="candidates",
            episode_graph_signal_hop_decay=0.5,
        )
        traversal = _traversal(cfg, episodes_for_entity=["ep_linked"], episode_entities=["e_hot"])
        results: list[dict] = []
        await traversal.append_entity_linked_episodes(
            results,
            group_id="default",
            seen_episode_ids=set(),
            candidate_entity_scores=[("e_hot", 0.9)],
            entity_signal={"e_hot": EntityGraphSignal(0.8, 0.3, 0)},
        )
        breakdown = results[0]["score_breakdown"]
        assert breakdown["activation"] == pytest.approx(0.4)
        assert breakdown["spreading"] == pytest.approx(0.15)
        assert breakdown["edge_proximity"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_traversal_breakdown_falls_back_to_explicit_zeros(self):
        cfg = ActivationConfig(entity_episode_traversal_source="candidates")
        traversal = _traversal(cfg, episodes_for_entity=["ep_linked"], episode_entities=["e_hot"])
        results: list[dict] = []
        await traversal.append_entity_linked_episodes(
            results,
            group_id="default",
            seen_episode_ids=set(),
            candidate_entity_scores=[("e_hot", 0.9)],
        )
        breakdown = results[0]["score_breakdown"]
        assert breakdown["activation"] == 0.0
        assert breakdown["spreading"] == 0.0
        assert breakdown["edge_proximity"] == 0.0


# ── Output surfaces: a value nobody can read is not built ───────────


class TestGraphSignalReachesOutput:
    def test_result_builder_emits_spreading(self):
        cfg = ActivationConfig()
        builder = RecallResultBuilder(cfg)
        sr = ScoredResult("ep_1", 0.5, 0.8, 0.4, 0.15, 0.5, result_type="episode")
        episode = Episode(
            id="ep_1",
            content="content",
            source="test",
            status=EpisodeStatus.COMPLETED,
            group_id="default",
            created_at=utc_now(),
        )
        result = builder.episode_result(episode, sr, linked_entities=["e1", "e2"])
        breakdown = result["score_breakdown"]
        assert breakdown["spreading"] == pytest.approx(0.15)
        assert breakdown["edge_proximity"] == pytest.approx(0.5)
        assert breakdown["activation"] == pytest.approx(0.4)

    def test_rest_item_exposes_spreading_and_linked_entity_count(self):
        """`spreading` did not exist as an output key on ANY surface."""
        from engram.retrieval.presenter import present_api_recall_item

        item = present_api_recall_item(
            {
                "result_type": "episode",
                "score": 0.47,
                "score_breakdown": {
                    "semantic": 0.8,
                    "activation": 0.4,
                    "spreading": 0.15,
                    "edge_proximity": 0.5,
                    "exploration_bonus": 0.0,
                },
                "episode": {"id": "ep_1", "content": "c", "source": "s", "created_at": None},
                "linked_entities": [{"name": "Alpha"}, {"name": "Beta"}],
            }
        )
        assert item["scoreBreakdown"]["spreading"] == pytest.approx(0.15)
        assert item["scoreBreakdown"]["edgeProximity"] == pytest.approx(0.5)
        assert item["linkedEntityCount"] == 2
