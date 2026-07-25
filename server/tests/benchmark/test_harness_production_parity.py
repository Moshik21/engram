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

import ast
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from engram.benchmark.methods import RetrievalMethod, run_retrieval
from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.retrieval.pipeline import retrieve
from engram.retrieval.spread_injection import select_spread_injections
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


# ── Ticket #32: the injection cap must be ONE cap ────────────────────
#
# `spread_candidate_injection_max` bounds how many graph-discovered entities may
# enter the recall candidate pool. Production applied it; the harness's forked
# copy of the same step did not. Measured on the dogfood corpus: production
# injected exactly 32 on 100% of 51 completions out of 453-487 discovered, while
# the harness injected all ~450 — a 12x pool divergence in the one wrapper that
# exists to guarantee the A/B measures production's scoring code.
#
# Harmless while spreading always returned {} live. Not harmless after 390866b.

_FANOUT = 8
_GRAPH_SIZE = 4000
_SEED_COUNT = 15
_INJECTION_CAP = 8


def _fanout_graph_store():
    """A hub-and-spoke graph wide enough that spreading discovers >> the cap.

    Shape taken from ``tests/test_spreading_reaches_graph.py``, which calibrated
    it against the live brain (~488 traversal reads on a real recall). Size is
    load-bearing: on a graph small enough that the traversal discovers fewer
    entities than the cap, an uncapped harness and a capped production agree by
    accident and the parity test passes on the bug.
    """

    async def neighbors(*args, **kwargs):
        node_id = args[0] if args else (kwargs.get("node_id") or kwargs.get("entity_id"))
        try:
            idx = int(str(node_id).removeprefix("e"))
        except ValueError:
            return []
        # Disjoint neighbourhoods, so each hop opens onto fresh nodes instead of
        # rediscovering the same handful.
        return [
            (f"e{(idx * _FANOUT + step) % _GRAPH_SIZE}", 0.9, "RELATES_TO", "Thing")
            for step in range(1, _FANOUT + 1)
        ]

    store = AsyncMock()
    store.get_active_neighbors_with_weights = neighbors
    store.get_entity = AsyncMock(return_value=None)
    store.get_relationships = AsyncMock(return_value=[])
    store.get_episode_entities = AsyncMock(return_value=[])
    store.get_episodes_for_entity = AsyncMock(return_value=[])
    store.update_episode = AsyncMock()
    store.update_episode_cue = AsyncMock()
    return store


def _fanout_search_index():
    idx = AsyncMock()
    # Descending seed strength: with every seed pinned equal the seed block fills
    # the result cap and no discovered entity is observable.
    idx.search = AsyncMock(return_value=[(f"e{i}", 0.9 - 0.03 * i) for i in range(_SEED_COUNT)])
    idx.search_episodes = AsyncMock(return_value=[])
    idx.search_episode_cues = AsyncMock(return_value=[])

    async def similarity(query="", entity_ids=None, group_id=None, **_kw):
        return {eid: 0.8 for eid in (entity_ids or [])}

    idx.compute_similarity = similarity
    idx._embeddings_enabled = False
    return idx


def _spread_cfg(**overrides) -> ActivationConfig:
    base = dict(
        episode_retrieval_enabled=False,
        cue_recall_enabled=False,
        chunk_search_enabled=False,
        mmr_enabled=False,
        gc_mmr_enabled=False,
        retrieval_spread_timeout_ms=5000,
        spread_candidate_injection_max=_INJECTION_CAP,
        # Keep the 1-hop candidate pool narrower than the traversal's reach, so
        # discovered entities are genuinely traversal-only.
        pool_graph_seed_count=1,
        pool_graph_max_neighbors=1,
        pool_graph_limit=5,
    )
    base.update(overrides)
    return ActivationConfig(**base)


async def _spread_production(cfg, timings):
    return await retrieve(
        query="spreading parity probe",
        group_id="default",
        graph_store=_fanout_graph_store(),
        activation_store=_activation_store(),
        search_index=_fanout_search_index(),
        cfg=cfg,
        stage_timings_ms=timings,
        limit=50,
    )


async def _spread_harness(cfg, timings):
    return await run_retrieval(
        "spreading parity probe",
        "default",
        _fanout_graph_store(),
        _activation_store(),
        _fanout_search_index(),
        _method(cfg),
        limit=50,
        stage_timings_ms=timings,
    )


def _is_traversal_discovered(entity_id: str) -> bool:
    """Seeds are e0..e14; anything else can only have come from the traversal."""
    try:
        return int(entity_id.removeprefix("e")) >= _SEED_COUNT
    except ValueError:
        return False


class TestSpreadInjectionParity:
    @pytest.mark.asyncio
    async def test_harness_injects_the_same_bounded_pool_production_injects(self):
        """THE divergence test for #32.

        RED against the pre-fix harness two ways at once: it emitted neither
        counter (KeyError on ``recall_spread_injected``) and it injected every
        discovery rather than the cap.
        """
        cfg = _spread_cfg()
        prod: dict[str, float] = {}
        harness: dict[str, float] = {}
        await _spread_production(cfg, prod)
        await _spread_harness(cfg, harness)

        # Positive probes FIRST. Equality between two paths that discovered
        # nothing is 0 == 0 — exactly the vacuous pass this file exists to stop.
        assert prod.get("recall_spread_discovered", 0) > _INJECTION_CAP, (
            "fixture is inert: production's traversal found nothing to cap"
        )
        assert harness.get("recall_spread_discovered", 0) > _INJECTION_CAP, (
            "fixture is inert: the harness's traversal found nothing to cap"
        )

        assert prod["recall_spread_injected"] == _INJECTION_CAP, (
            "production stopped applying the cap; this test no longer measures parity"
        )
        assert harness["recall_spread_injected"] == prod["recall_spread_injected"]

    @pytest.mark.asyncio
    async def test_harness_pool_is_not_swamped_by_the_traversal(self):
        """Independent of the counter: the pool the scorer saw must be bounded.

        A harness that emitted the right number and still appended every
        discovery would pass the test above. This one reads the candidate pool
        through the scored results instead of through the metric.
        """
        cfg = _spread_cfg()
        harness: dict[str, float] = {}
        results = await _spread_harness(cfg, harness)
        entity_ids = {sr.node_id for sr in results if sr.result_type != "episode"}
        discovered_in_pool = {eid for eid in entity_ids if _is_traversal_discovered(eid)}

        assert harness["recall_spread_discovered"] > _INJECTION_CAP
        assert len(discovered_in_pool) <= _INJECTION_CAP, (
            f"{len(discovered_in_pool)} traversal-discovered entities reached the "
            f"scored pool with a cap of {_INJECTION_CAP}"
        )

    def test_the_injection_cap_has_exactly_one_read_site(self):
        """A duplicated constant drifts — that is how #32 was born.

        Attribute reads only, so the kill rig's provenance list (which names the
        field as a string) is not counted as a second implementation.
        """
        engram_root = Path(__file__).resolve().parents[2] / "engram"
        readers = set()
        for path in engram_root.rglob("*.py"):
            source = path.read_text()
            if "spread_candidate_injection_max" not in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "spread_candidate_injection_max"
                ):
                    readers.add(path.relative_to(engram_root).as_posix())
        assert readers == {"retrieval/spread_injection.py"}, (
            f"the injection cap is read in {sorted(readers)}; it must have one "
            "implementation that every caller shares"
        )


class TestSpreadInjectionSelector:
    """The shared rule itself. Cheap, and it pins the properties both callers rely on."""

    def test_cap_keeps_the_strongest_bonuses(self):
        bonuses = {f"e{i}": (i + 1) / 100.0 for i in range(50)}
        chosen, discovered = select_spread_injections(
            bonuses, set(), ActivationConfig(spread_candidate_injection_max=3)
        )
        assert discovered == 50
        assert chosen == ["e49", "e48", "e47"]

    def test_zero_cap_is_unbounded(self):
        bonuses = {f"e{i}": 0.5 for i in range(50)}
        chosen, discovered = select_spread_injections(
            bonuses, set(), ActivationConfig(spread_candidate_injection_max=0)
        )
        assert discovered == 50
        assert len(chosen) == 50

    def test_ties_break_on_id_so_the_pool_is_deterministic(self):
        bonuses = {"eb": 0.5, "ea": 0.5, "ec": 0.5}
        cfg = ActivationConfig(spread_candidate_injection_max=2)
        assert select_spread_injections(bonuses, set(), cfg)[0] == ["ea", "eb"]

    def test_already_pooled_entities_are_not_re_injected(self):
        bonuses = {"ea": 0.5, "eb": 0.5}
        cfg = ActivationConfig(spread_candidate_injection_max=0)
        assert select_spread_injections(bonuses, {"ea"}, cfg) == (["eb"], 1)
