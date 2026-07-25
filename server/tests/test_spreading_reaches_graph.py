"""Positive probe: spreading activation must actually traverse the graph.

This area had no test that fails when spreading is DEAD, which is why it stayed
dead: on the live shell the stage timed out on ~100% of recalls, returned {},
and every existing assertion still passed (they assert ``bonuses == {}`` on
empty graphs, or assert the timeout path).

The probe is ``recall_spread_reached`` — the number of nodes at hop >= 1.
It deliberately does NOT assert "hop_distances is non-empty": every seed is
inserted at hop 0 before the traversal reads anything, so that check is true
even when zero edges are walked and would pass on a corpse.

The rest of the file guards the collateral damage a traversal bound can do:

* the bound must not leak out of recall into the offline (dream) and
  write-path (prospective memory) callers, which have no latency budget;
* the stage must not be allowed to outrun its own wall clock on a cold store,
  because the time it overruns by is taken from the recall stages after it;
* the probe must be emitted on the FAILING paths, not only the happy one;
* ``recall_spread_injected`` must report what reached the candidate pool, not
  what spreading hoped to put there.
"""

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from engram.activation.spreading import spread_activation
from engram.config import ActivationConfig
from engram.retrieval.pipeline import retrieve

# Per-neighbor-read latency. The live native store measured ~0.1ms warm and up
# to ~150ms cold; 0.8ms keeps the test fast while still making an unbounded
# traversal (~135 reads here, ~488 on the live brain) unable to fit a 75ms
# budget — the exact condition that killed the stage in production.
_READ_LATENCY_S = 0.0008
_GRAPH_SIZE = 8000
# Fan-out stays moderate on purpose: fan_factor is max(fan_s_min, fan_s_max -
# ln(degree+1)), so a very wide node damps each edge below
# spread_firing_threshold and the traversal fires nothing at all. 8 keeps every
# hop live while 15 seeds still push the read count past the budget.
_FANOUT = 8
_SEED_COUNT = 15


def _graph_store(*, reachable: bool = True, read_latency_s: float = _READ_LATENCY_S):
    """A hub-and-spoke graph big enough that an unbounded traversal overruns.

    ``reachable=False`` severs every edge, which is what a dead traversal looks
    like from the pipeline's point of view: the stage still runs and still
    returns, it just never leaves the seeds.
    """
    reads: list[str] = []

    async def neighbors(*args, **kwargs):
        # Several recall lanes read neighbours. Only the traversal calls with a
        # single positional node id plus a keyword group_id (bfs.py), so count
        # those alone against the traversal's budget.
        from_traversal = len(args) == 1 and "group_id" in kwargs and "entity_id" not in kwargs
        node_id = args[0] if args else (kwargs.get("node_id") or kwargs.get("entity_id"))
        if from_traversal:
            reads.append(node_id)
        await asyncio.sleep(read_latency_s)
        if not reachable:
            return []
        try:
            idx = int(str(node_id).removeprefix("e"))
        except ValueError:
            return []
        # Disjoint neighbourhoods: every node opens onto a fresh block, so the
        # traversal must genuinely read ~135 nodes rather than rediscovering the
        # same handful. That is what makes an unbounded traversal overrun the
        # 75ms budget here, exactly as it did on the live brain (~488 reads).
        return [
            (f"e{(idx * _FANOUT + step) % _GRAPH_SIZE}", 0.9, "RELATES_TO", "Thing")
            for step in range(1, _FANOUT + 1)
        ]

    store = AsyncMock()
    store.get_active_neighbors_with_weights = neighbors
    store.reads = reads
    store.get_entity = AsyncMock(return_value=None)
    store.get_relationships = AsyncMock(return_value=[])
    return store


def _activation_store():
    store = AsyncMock()
    store.batch_get = AsyncMock(return_value={})
    store.get_activation = AsyncMock(return_value=None)
    store.get_top_activated = AsyncMock(return_value=[])
    store.record_access = AsyncMock()
    return store


def _search_index(seed_count: int = _SEED_COUNT, similarity=None):
    idx = AsyncMock()
    # Descending seed strength. With every seed pinned at 0.9 the seed block
    # fills the result cap and no graph-discovered row can ever be observed —
    # seeds are hop 0 by definition, so the hop signal would be invisible.
    idx.search = AsyncMock(return_value=[(f"e{i}", 0.9 - 0.03 * i) for i in range(seed_count)])
    idx.search_episodes = AsyncMock(return_value=[])
    idx.search_episode_cues = AsyncMock(return_value=[])

    # Graph-discovered entities get a real cosine in production; without one
    # they score 0.0, never reach the top-K, and their hop distances cannot be
    # observed on any returned row.
    async def default_similarity(query="", entity_ids=None, group_id=None, **_kw):
        return {eid: 0.8 for eid in (entity_ids or [])}

    idx.compute_similarity = similarity or default_similarity
    idx._embeddings_enabled = False
    return idx


def _cfg(**overrides):
    base = dict(
        episode_retrieval_enabled=False,
        cue_recall_enabled=False,
        chunk_search_enabled=False,
        retrieval_spread_timeout_ms=75,
        # Other recall lanes also read neighbours; switch off the one that runs
        # after scoring so store.reads counts the traversal's budget alone.
        gc_mmr_enabled=False,
        # Keep the 1-hop candidate pool narrower than the traversal's reach.
        # That pool injects neighbours as CANDIDATES, and any candidate over
        # seed_threshold becomes a seed — which the scorer then pins at hop 0
        # by definition. Narrowing it leaves nodes that only the traversal can
        # find, so hop distances above 0 are observable on real rows.
        pool_graph_seed_count=1,
        pool_graph_max_neighbors=1,
        pool_graph_limit=5,
    )
    base.update(overrides)
    return ActivationConfig(**base)


async def _recall(store, cfg, search_index=None):
    stage_timings: dict[str, float] = {}
    results = await retrieve(
        query="spreading probe",
        group_id="default",
        graph_store=store,
        activation_store=_activation_store(),
        search_index=search_index or _search_index(),
        cfg=cfg,
        stage_timings_ms=stage_timings,
        limit=50,
    )
    return results, stage_timings


class TestSpreadingReachesTheGraph:
    @pytest.mark.asyncio
    async def test_spreading_traverses_real_edges_within_budget(self):
        """The stage completes AND walks at least one edge inside its budget."""
        store = _graph_store()
        results, timings = await _recall(store, _cfg())

        assert "recall_spread" in timings, (
            f"spreading did not complete; stage keys={sorted(timings)}"
        )
        assert "recall_spread_timeout" not in timings

        # THE PROBE: nodes at hop >= 1. Zero here means the traversal walked no
        # edges, which is the failure this test exists to catch.
        assert timings["recall_spread_reached"] > 0, (
            "spreading returned no nodes beyond the seeds — the traversal is dead"
        )
        assert timings["recall_spread_max_hop"] >= 1

        # ...and the hop distances must survive into the scored results.
        assert any((r.hop_distance or 0) >= 1 for r in results), (
            "no result carried a hop distance > 0"
        )

    @pytest.mark.asyncio
    async def test_probe_goes_red_when_the_traversal_cannot_leave_the_seeds(self):
        """Guard the guard: a severed graph must drive the probe to zero.

        If this ever passes with reached > 0 the probe is measuring something
        other than real traversal.
        """
        store = _graph_store(reachable=False)
        _results, timings = await _recall(store, _cfg())

        assert timings.get("recall_spread_reached") == 0
        assert timings.get("recall_spread_max_hop") == 0

    @pytest.mark.asyncio
    async def test_read_budget_bounds_how_much_graph_one_recall_touches(self):
        """retrieval_spread_max_reads is the traversal's budget, and it is enforced."""
        store = _graph_store()
        _results, timings = await _recall(store, _cfg(retrieval_spread_max_reads=16))

        assert len(store.reads) <= 16, f"read cap ignored: {len(store.reads)} reads"
        assert timings["recall_spread_reached"] > 0

    @pytest.mark.asyncio
    async def test_slow_store_degrades_to_shallow_results_not_to_nothing(self):
        """A store too slow to finish must still return the frontier it reached.

        Being cancelled by the caller's timeout discarded every completed read,
        so a slow store degraded the stage to {} rather than to "less depth".
        """
        store = _graph_store()
        # Traversal budget far below the traversal's cost, but still under the
        # stage's hard wall clock, so the traversal gets to return.
        _results, timings = await _recall(store, _cfg(retrieval_spread_traversal_budget_ms=10))

        assert "recall_spread" in timings, "traversal was cancelled instead of returning"
        assert timings["recall_spread_reached"] > 0, (
            "a slow store produced no traversal at all instead of a shallow one"
        )

    @pytest.mark.asyncio
    async def test_spreading_does_not_swamp_the_candidate_pool(self):
        """Graph-discovered entities supplement the pool; they must not flood it."""
        store = _graph_store()
        _results, timings = await _recall(store, _cfg(spread_candidate_injection_max=8))

        assert timings["recall_spread_reached"] > 8
        assert timings["recall_spread_discovered"] > 8
        assert timings["recall_spread_injected"] <= 8


class TestTheRecallBudgetStaysInRecall:
    """The traversal bound belongs to the caller that owes a user a latency.

    ``spread_activation`` has four callers and only one is recall. When the
    strategy read the bound off the config instead of taking it from the
    caller, the offline dream phase and the write-path prospective-memory
    caller silently inherited it: a single-hub dream spread went from 141 reads
    / 700 nodes reached to 64 reads / 392 nodes — a 44% cut in pathway
    strengthening, with no test and no way to see it.
    """

    @pytest.mark.asyncio
    async def test_spread_activation_is_unbounded_by_default(self):
        """No caller-supplied bound means no bound, whatever the recall knobs say."""
        cfg = ActivationConfig()
        assert cfg.retrieval_spread_max_reads > 0, "the recall cap must be ON for this to prove"

        store = _graph_store()
        # Seed energy mirrors the pipeline's sem * max(activation, 0.15) floor.
        # At energy 1.0 the ENERGY budget binds first (37 reads) and the read
        # cap would never be exercised, so the test would prove nothing.
        seeds = [(f"e{i}", 0.135) for i in range(_SEED_COUNT)]
        _bonuses, hops = await spread_activation(seeds, store, cfg, group_id="default")

        assert len(store.reads) > cfg.retrieval_spread_max_reads, (
            "the recall read cap leaked into an unbounded caller: "
            f"{len(store.reads)} reads vs cap {cfg.retrieval_spread_max_reads}"
        )
        assert max(hops.values()) >= 2

    @pytest.mark.asyncio
    async def test_a_slow_store_does_not_truncate_an_unbounded_caller(self):
        """Offline spreading has no wall clock, so a slow store must not cut it short."""
        cfg = ActivationConfig()
        assert cfg.retrieval_spread_traversal_budget_ms > 0

        # 9 reads at 10ms each is ~90ms — well past the recall traversal budget
        # this caller must not be subject to.
        slow = _graph_store(read_latency_s=0.01)
        fast = _graph_store(read_latency_s=0.0)
        seeds = [("e1", 0.135)]

        await spread_activation(seeds, slow, cfg, group_id="default")
        await spread_activation(seeds, fast, cfg, group_id="default")

        assert len(slow.reads) == len(fast.reads), (
            "a recall wall clock leaked into an unbounded caller: "
            f"slow store did {len(slow.reads)} reads vs {len(fast.reads)} fast"
        )

    def test_only_the_recall_caller_supplies_a_traversal_bound(self):
        """Static guard: the bound may be passed from exactly one call site.

        The leak was invisible precisely because it was a config read, not an
        argument. Now that it is an argument, this fails the moment a second
        caller starts bounding a traversal that has no latency budget.
        """
        server_root = Path(__file__).resolve().parents[1]
        bounded: set[str] = set()
        call_sites: set[str] = set()

        for path in sorted((server_root / "engram").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name != "spread_activation":
                    continue
                rel = str(path.relative_to(server_root))
                call_sites.add(rel)
                if {kw.arg for kw in node.keywords} & {"max_reads", "deadline"}:
                    bounded.add(rel)

        # Canary: if the walker stops finding call sites it stops finding leaks.
        assert len(call_sites) >= 4, f"spread_activation call sites not found: {call_sites}"
        assert bounded == {"engram/retrieval/pipeline.py"}, (
            f"traversal bound passed outside recall: {sorted(bounded)}"
        )


class TestTheStageCannotOutrunItsWallClock:
    """Overrun is not free: it is taken from the recall stages downstream.

    Widening the stage's cancel to make room for a slow traversal made cold
    recalls DROP episodes they used to return — the spreading stage ran ~2x its
    nominal deadline and pushed episode materialisation and similarity backfill
    past their own budgets. The stage's wall clock is therefore unchanged from
    the pre-fix value, and the traversal's own budget is set below it instead.
    """

    @pytest.mark.asyncio
    async def test_a_cold_store_cannot_stretch_the_stage_past_its_timeout(self):
        """One pathologically slow read must not cost more than the stage cap."""
        timeout_ms = 75
        # A single read far longer than the whole stage budget: the traversal
        # cannot bail early out of a read it has already started, so this is the
        # worst case the outer cancel exists for.
        store = _graph_store(read_latency_s=0.2)
        cfg = _cfg(retrieval_spread_timeout_ms=timeout_ms)

        _results, timings = await _recall(store, cfg)

        spent = timings.get("recall_spread", 0.0) + timings.get("recall_spread_timeout", 0.0)
        assert spent > 0.0, f"the stage did not run at all; keys={sorted(timings)}"
        # 2x the cap is generous for a loaded CI box and still fails loudly on a
        # backstop multiplier (the 4x version spent ~300ms here).
        assert spent <= timeout_ms * 2, (
            f"spreading spent {spent:.1f}ms against a {timeout_ms}ms stage cap — "
            "that overrun comes out of the stages after it"
        )

    @pytest.mark.asyncio
    async def test_the_probe_is_emitted_when_the_stage_times_out(self):
        """A gate on 'reached == 0' must fire on failure, not skip it.

        The probe ran only on the success path, so it was ABSENT — not zero —
        on exactly the failures it exists to catch.
        """
        store = _graph_store(read_latency_s=0.2)
        _results, timings = await _recall(store, _cfg(retrieval_spread_timeout_ms=20))

        assert "recall_spread_timeout" in timings, f"expected a timeout; keys={sorted(timings)}"
        assert timings.get("recall_spread_reached") == 0.0, (
            "recall_spread_reached is absent on the timeout path"
        )
        assert timings.get("recall_spread_max_hop") == 0.0

    @pytest.mark.asyncio
    async def test_a_ppr_install_is_bounded_too(self):
        """The bound is the caller's, so it must not depend on the strategy.

        A backstop applied only where BFS self-limits made a PPR install wait
        longer and still get nothing.
        """
        store = _graph_store()
        cfg = _cfg(spreading_strategy="ppr", retrieval_spread_max_reads=16)
        await _recall(store, cfg)

        assert len(store.reads) <= 16, f"PPR ignored the read cap: {len(store.reads)} reads"


class TestInjectedReportsOutcomeNotIntent:
    @pytest.mark.asyncio
    async def test_a_degraded_rescore_reports_zero_injected(self):
        """The count was written before the block that can throw it all away.

        It therefore read healthiest at the exact moment the mechanism failed.
        """

        class NativeQueryError(RuntimeError):
            """Named to match the pipeline's degrade check."""

        async def exploding_similarity(query="", entity_ids=None, group_id=None, **_kw):
            raise NativeQueryError("native store timed out")

        store = _graph_store()
        _results, timings = await _recall(
            store,
            _cfg(),
            search_index=_search_index(similarity=exploding_similarity),
        )

        assert timings["recall_spread_discovered"] > 0, "nothing was discovered to inject"
        assert "recall_spreading_rescore_degraded" in timings
        assert timings["recall_spread_injected"] == 0, (
            "injected reported entities that never reached the candidate pool"
        )


class TestCommunityFreshnessCannotRetryForever:
    @pytest.mark.asyncio
    async def test_a_cancelled_refresh_still_stamps_the_clock(self):
        """ensure_fresh runs inside the caller's budget, so it must back off.

        Recording freshness only after compute() returned meant a compute that
        was cancelled left the group stale forever: every later recall repeated
        the identical doomed work inside the identical budget.
        """
        from engram.activation.community import CommunityStore

        store = CommunityStore(stale_seconds=300.0)
        provider = AsyncMock()

        async def never_finishes(*_args, **_kwargs):
            await asyncio.sleep(3600)

        provider.get_active_neighbors_with_weights = never_finishes

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                store.ensure_fresh("g1", provider, entity_ids=["a", "b"]),
                timeout=0.02,
            )

        assert store.is_stale("g1") is False, (
            "a cancelled refresh left the group stale — the next recall repeats it"
        )


class TestBoundsAreCheckedBeforeTheReadNotAfter:
    @pytest.mark.asyncio
    async def test_the_traversal_bails_before_a_read_it_cannot_afford(self):
        """Checking the clock only after a read overshoots by one read, every time.

        The live cold stage ran ~2x its nominal deadline for exactly this
        reason. The traversal refuses to start a read that the slowest read it
        has seen says cannot finish in time.

        The FIRST read is not covered — nothing can be known about a store
        before it has been read once, so a single read slower than the whole
        budget still overshoots. That case is what the stage's outer wall clock
        is for; this test covers every read after it.
        """
        from engram.activation.bfs import BFSStrategy

        read_s = 0.03
        budget_s = 0.05
        store = _graph_store(read_latency_s=read_s)
        cfg = ActivationConfig()

        started = time.monotonic()
        await BFSStrategy().spread(
            [("e1", 1.0)],
            store,
            cfg,
            group_id="default",
            deadline=started + budget_s,
        )
        elapsed = time.monotonic() - started

        # With the predictive bail the traversal stops after the read that
        # leaves too little budget for another (30ms here). Without it, it
        # starts one more and lands at 60ms — past the budget it was given.
        assert elapsed < budget_s, (
            f"traversal overshot its budget by a full read: {elapsed * 1000:.1f}ms "
            f"against a {budget_s * 1000:.0f}ms budget"
        )
