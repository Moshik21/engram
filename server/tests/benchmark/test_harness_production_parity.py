"""The benchmark harness must measure the SAME scoring code production runs.

`benchmark/methods.py::run_retrieval` is the retrieval path every A/B script in
`server/scripts/benchmark_*.py` executes. It has its own episode lane, and that
lane used to construct episode ``ScoredResult``s with the same
``activation=0.0, spreading=0.0, edge_proximity=0.0`` literals that the
production pipeline was fixed for in commit 8ff4297 (Step 5.01 snapshot +
Step 5.8 ``apply_episode_graph_signal``).

The consequence, which is the reason this file exists: an A/B run through the
harness would exercise code that CANNOT receive any graph signal, and would
report "the graph does not help" when the truth is "the harness bypassed the
graph". That is the sixth instance of the project's dominant bug class, and it
would silently void the pre-registered graph experiment
(`docs/product/GRAPH_THESIS.md` §5).

These tests run the production entry point (`retrieval.pipeline.retrieve`) and
the harness entry point (`benchmark.methods.run_retrieval`) over the SAME fake
stores and the SAME config, and compare the derived graph signal on the
answer-bearing episode rows. They fail if the two disagree — including if the
harness forks a private copy of the derivation that later drifts.

A parity assertion alone can be satisfied by two paths that are both dead
(0.0 == 0.0), so every parity assertion here is paired with a positive probe
that the signal is non-zero and load-bearing on the fixture.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from engram.benchmark.methods import RetrievalMethod, run_retrieval
from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.retrieval.pipeline import retrieve
from engram.utils.dates import utc_now

# ── Fixtures (shape copied from tests/test_episode_graph_signal.py so both
#    suites exercise the same stores) ──────────────────────────────────


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
    store = AsyncMock()
    store.get_active_neighbors_with_weights = AsyncMock(return_value=[])
    store.update_episode = AsyncMock()
    store.update_episode_cue = AsyncMock()
    store.get_entity = AsyncMock(
        return_value=Entity(
            id="e_hot",
            name="Test",
            entity_type="Thing",
            summary="A test entity",
            group_id="default",
        )
    )
    store.get_episode_by_id = AsyncMock(
        return_value=Episode(
            id="ep_lead",
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


def _search_index(entity_results, episode_results):
    idx = AsyncMock()
    idx.search = AsyncMock(return_value=list(entity_results))
    idx.search_episodes = AsyncMock(return_value=list(episode_results))
    idx.search_episode_cues = AsyncMock(return_value=[])
    idx.compute_similarity = AsyncMock(return_value={})
    idx._embeddings_enabled = False
    return idx


ENTITY_HITS = [("e_hot", 0.9)]
EPISODE_HITS = [("ep_lead", 0.80), ("ep_trail", 0.79)]
# Only the TRAILING episode is linked into the graph, so a live graph signal
# must be able to invert the pair.
EPISODE_ENTITIES = {"ep_trail": ["e_hot"]}
STATES = {"e_hot": _active_state("e_hot", 40)}


def _cfg(**overrides) -> ActivationConfig:
    base = {
        "episode_retrieval_enabled": True,
        "episode_retrieval_max": 5,
        "mmr_enabled": False,
        "episode_graph_signal_enabled": True,
        "episode_graph_signal_weight": 0.25,
    }
    base.update(overrides)
    return ActivationConfig(**base)


def _method(cfg: ActivationConfig) -> RetrievalMethod:
    return RetrievalMethod(
        name="parity",
        description="same config as the production run under test",
        config=cfg,
        spreading_enabled=True,
    )


def _episode_signal(results) -> dict[str, tuple[float, float, float]]:
    return {
        sr.node_id: (sr.activation, sr.spreading, sr.edge_proximity)
        for sr in results
        if sr.result_type == "episode"
    }


async def _production(cfg: ActivationConfig, timings: dict[str, float] | None = None):
    return await retrieve(
        query="test query",
        group_id="default",
        graph_store=_graph_store(EPISODE_ENTITIES),
        activation_store=_activation_store(STATES),
        search_index=_search_index(ENTITY_HITS, EPISODE_HITS),
        cfg=cfg,
        stage_timings_ms=timings if timings is not None else {},
    )


async def _harness(cfg: ActivationConfig, timings: dict[str, float] | None = None):
    return await run_retrieval(
        "test query",
        "default",
        _graph_store(EPISODE_ENTITIES),
        _activation_store(STATES),
        _search_index(ENTITY_HITS, EPISODE_HITS),
        _method(cfg),
        limit=10,
        stage_timings_ms=timings,
    )


# ── The parity contract ─────────────────────────────────────────────


class TestHarnessProductionParity:
    @pytest.mark.asyncio
    async def test_harness_episode_lane_carries_the_same_graph_signal_as_production(self):
        """THE divergence test.

        RED against the pre-fix harness: production writes a derived
        (activation, spreading, edge_proximity) onto ``ep_trail`` while the
        harness writes literal zeros, so the two dicts differ. It stays red for
        any future fork of the derivation that drifts from production.
        """
        cfg = _cfg()
        production = _episode_signal(await _production(cfg))
        harness = _episode_signal(await _harness(cfg))

        assert set(production) == set(harness), "harness and production saw different episodes"

        # Positive probe FIRST: a parity assertion between two dead paths is
        # satisfied by 0.0 == 0.0, which is exactly the failure this file exists
        # to prevent.
        assert production["ep_trail"][0] > 0.0, (
            "fixture is inert: production itself derived no activation, so parity "
            "below would prove nothing"
        )
        assert harness == production

    @pytest.mark.asyncio
    async def test_harness_graph_term_reorders_episodes_exactly_as_production_does(self):
        """Signal parity is not enough — the SCORE has to move the ordering.

        A harness that copied the three fields onto the result and forgot the
        score delta would pass the test above and still measure a null.
        """
        cfg = _cfg()
        prod_order = [sr.node_id for sr in await _production(cfg) if sr.result_type == "episode"]
        harness_order = [sr.node_id for sr in await _harness(cfg) if sr.result_type == "episode"]

        # ep_trail loses on semantics (0.79 < 0.80) and wins only via the graph.
        assert prod_order[0] == "ep_trail", "fixture is inert: production did not reorder"
        assert harness_order == prod_order

    @pytest.mark.asyncio
    async def test_harness_emits_the_same_positive_probe_metrics_as_production(self):
        """An A/B run must be able to PROVE the mechanism fired, not assume it.

        Without these counters a null result is indistinguishable from a
        harness that silently skipped the graph — the exact ambiguity that
        makes past graph results uninterpretable.
        """
        cfg = _cfg()
        prod_timings: dict[str, float] = {}
        harness_timings: dict[str, float] = {}
        await _production(cfg, prod_timings)
        await _harness(cfg, harness_timings)

        for key in (
            "recall_episode_graph_signal_covered",
            "recall_episode_graph_signal_applied",
            "recall_episode_graph_signal_reorders",
        ):
            assert prod_timings.get(key, 0.0) > 0.0, f"production probe {key} inert"
            assert harness_timings.get(key) == prod_timings.get(key), (
                f"harness/production disagree on {key}"
            )

    @pytest.mark.asyncio
    async def test_flag_off_harness_output_is_byte_identical_to_before(self):
        """Additive-only contract: the fix must not move any default A/B number."""
        off = _cfg(episode_graph_signal_enabled=False, episode_graph_signal_weight=0.0)
        results = await _harness(off)
        episodes = [sr for sr in results if sr.result_type == "episode"]
        assert episodes
        for sr in episodes:
            assert (sr.activation, sr.spreading, sr.edge_proximity) == (0.0, 0.0, 0.0)
        assert [sr.node_id for sr in episodes] == ["ep_lead", "ep_trail"]
