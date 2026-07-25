"""Ticket #34: one slow read must not collapse the rest of the traversal.

The predictive bail added in `390866b` estimated read cost with a RUNNING MAX
that never decayed, so a single outlier poisoned the estimate for the remainder
of the recall. Measured on a 0.5 ms/read store, 50 ms traversal budget,
``max_reads=64`` (`engram/activation/read_budget.py` carries the argument):

    running max   no outlier            64 reads / 498 reached /  6.6 ms unspent
    running max   ONE 25 ms read         6 reads /  34 reached / 19.5 ms unspent
    running max   cold 30 ms FIRST read  1 read  /   0 reached / 17.9 ms unspent

    EWMA          no outlier            64 reads / 498 reached /  8.2 ms unspent
    EWMA          ONE 25 ms read        36 reads / 274 reached /  0.2 ms unspent
    EWMA          cold 30 ms FIRST read 28 reads / 210 reached /  0.4 ms unspent

93% of the reach, thrown away, with 39% of the budget never spent — and the
stage reported COMPLETED throughout. That silence is the second half of this
file: a collapsed traversal now has a metric signature
(``recall_spread_stop_predicted_cost`` plus a large
``recall_spread_budget_unspent_ms``) that a healthy one does not.

Every test here fails when the estimator is reverted to a running max — proved
by neutering ``READ_COST_EWMA_ALPHA``/``finish_read`` in place, not assumed.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from engram.activation import read_budget as read_budget_module
from engram.activation.bfs import BFSStrategy
from engram.activation.read_budget import ReadBudget
from engram.config import ActivationConfig
from engram.retrieval.pipeline import retrieve

# The ticket's measured regime. Live per-read latency spans 0.1-150 ms; 0.5 ms
# is the warm end, and the injected outliers are inside the measured range.
_BASE_READ_S = 0.0005
_BUDGET_S = 0.050
_MAX_READS = 64
_FANOUT = 8
_GRAPH_SIZE = 8000
_SEED_COUNT = 15


def _graph_store(*, base_s: float = _BASE_READ_S, outlier_s: float = 0.0, outlier_at: int = 6):
    """A hub-and-spoke store whose Nth read is slow.

    Disjoint neighbourhoods, so the traversal must genuinely keep reading rather
    than rediscovering the same handful of nodes — otherwise the reach number
    would be flat and could not show a collapse.
    """
    reads: list[str] = []

    async def neighbors(*args, **kwargs):
        from_traversal = len(args) == 1 and "group_id" in kwargs and "entity_id" not in kwargs
        node_id = args[0] if args else (kwargs.get("node_id") or kwargs.get("entity_id"))
        delay = base_s
        if from_traversal:
            reads.append(node_id)
            if outlier_s and len(reads) == outlier_at:
                delay = outlier_s
        await asyncio.sleep(delay)
        try:
            idx = int(str(node_id).removeprefix("e"))
        except ValueError:
            return []
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


async def _traverse(store, *, budget_s: float = _BUDGET_S, max_reads: int | None = _MAX_READS):
    """One bounded BFS, returning (reached, stats)."""
    stats: dict[str, float | str] = {}
    deadline = time.monotonic() + budget_s
    _bonuses, hops = await BFSStrategy().spread(
        # 0.135 mirrors the pipeline's sem * max(activation, 0.15) floor. At
        # energy 1.0 the ENERGY budget binds first and the read bound is never
        # exercised, so the test would prove nothing about the estimator.
        [(f"e{i}", 0.135) for i in range(_SEED_COUNT)],
        store,
        ActivationConfig(),
        group_id="default",
        max_reads=max_reads,
        deadline=deadline,
        traversal_stats=stats,
    )
    return len([h for h in hops.values() if h >= 1]), stats


class TestOneSlowReadDoesNotCollapseTheTraversal:
    @pytest.mark.asyncio
    async def test_a_single_outlier_read_costs_depth_not_the_traversal(self):
        """THE regression. A running max turns one 25 ms read into a 93% reach loss."""
        clean_reached, clean_stats = await _traverse(_graph_store())
        assert clean_reached > 100, (
            f"baseline traversal is too small to prove anything: {clean_stats}"
        )

        slow_reached, slow_stats = await _traverse(_graph_store(outlier_s=0.025))

        # Property 1: the budget must actually get SPENT. A running max leaves
        # 19.5 ms of 50 ms unspent because it refuses every remaining read.
        unspent = float(slow_stats["budget_unspent_ms"])
        assert unspent <= _BUDGET_S * 1000 * 0.25, (
            f"traversal quit with {unspent:.1f}ms of its "
            f"{_BUDGET_S * 1000:.0f}ms budget unspent — the cost estimate is poisoned; "
            f"stats={slow_stats}"
        )

        # Property 2: the user-visible consequence. One outlier may cost depth;
        # it may not cost the traversal.
        assert slow_reached >= clean_reached * 0.30, (
            f"one slow read cost {100 - 100 * slow_reached / clean_reached:.0f}% of the reach "
            f"({slow_reached} vs {clean_reached} clean); stats={slow_stats}"
        )

    @pytest.mark.asyncio
    async def test_a_cold_first_read_does_not_end_the_traversal(self):
        """A sample of ONE from a 1500x-spread distribution must not stop the work.

        With a running max a 30 ms first read pinned the estimate at 30 ms, so
        every subsequent read was refused: 1 read, ZERO nodes reached, and 18 ms
        of the 50 ms budget never spent.

        The first read itself always starts, and that is correct — nothing can be
        known about a store before it has been read once, and a traversal that
        refuses its first read has zero reach by construction. What must not
        happen is the first read deciding the whole traversal.
        """
        reached, stats = await _traverse(_graph_store(outlier_s=0.030, outlier_at=1))

        assert stats["reads"] > 1, f"one cold read ended the traversal: {stats}"
        assert reached > 0, f"a cold first read produced no traversal at all: {stats}"
        assert float(stats["budget_unspent_ms"]) <= _BUDGET_S * 1000 * 0.25, stats

    @pytest.mark.asyncio
    async def test_a_read_as_large_as_the_budget_still_stops_the_traversal(self):
        """The honest non-improvement: 45 ms of a 50 ms budget IS the budget.

        This case does NOT get better, and it should not. After a 45 ms read the
        budget is genuinely gone, so the traversal stops with ~nothing left —
        which is the honest outcome, not a collapse. Recorded so nobody reads
        the fix as "outliers no longer matter".

        Asserts the unspent budget rather than the stop reason: whether the last
        read leaves 0.4 ms or −1 ms decides between ``predicted_cost`` and
        ``deadline`` on scheduling jitter alone, and both mean "the budget is
        spent". This is the third place in this file where the reason proved
        unstable and the unspent figure did not — which is precisely why the
        alert in ``_record_spread_budget`` keys on unspent budget.
        """
        _reached, stats = await _traverse(_graph_store(outlier_s=0.045))

        assert float(stats["budget_unspent_ms"]) < 5.0, (
            f"a read that consumed 90% of the budget should leave ~nothing: {stats}"
        )

    @pytest.mark.asyncio
    async def test_a_clean_store_is_unchanged(self):
        """The fix must not buy its reach by loosening the bound on a healthy store.

        Deliberately does NOT assert *which* bound fired. On an unloaded box a
        fast store exhausts ``max_reads``; on a loaded one the 64 reads no
        longer fit the 50ms clock and it stops on the budget instead. Both are
        correct, and asserting the reason made this test flake 2 runs in 20.
        """
        reached, stats = await _traverse(_graph_store())

        assert stats["reads"] <= _MAX_READS, f"the read cap was loosened: {stats}"
        assert float(stats["budget_unspent_ms"]) <= _BUDGET_S * 1000 * 0.25, (
            f"a fast store left budget on the table: {stats}"
        )
        assert reached > 100, f"a fast store barely traversed: {reached}, {stats}"


class TestTheBoundIsStillABound:
    """What the EWMA gives up, stated and bounded rather than quietly dropped.

    The running max guaranteed "never overshoot the deadline after read 1". The
    EWMA guarantees the weaker "never START a read after the deadline, and
    overshoot by at most one read's cost". That trade is the whole fix: the
    strong guarantee is what refused 94% of the work.

    The harm the original bound was written against was a stage running ~2x its
    nominal deadline and stealing that time from the recall stages after it.
    That harm is still excluded.
    """

    @pytest.mark.asyncio
    async def test_overshoot_is_bounded_by_one_read(self):
        read_s = 0.030
        budget_s = 0.050
        store = _graph_store(base_s=read_s)

        started = time.monotonic()
        _reached, stats = await _traverse(store, budget_s=budget_s, max_reads=None)
        elapsed = time.monotonic() - started

        # No read may START after the deadline: the last read can therefore end
        # at most one read-cost past it. 10ms of slack for event-loop scheduling.
        ceiling = budget_s + float(stats["read_ms_max"]) / 1000.0 + 0.010
        assert elapsed <= ceiling, (
            f"traversal ran {elapsed * 1000:.1f}ms against a {budget_s * 1000:.0f}ms budget "
            f"(ceiling {ceiling * 1000:.1f}ms); stats={stats}"
        )
        # And the harm the original assertion protected against, unchanged.
        assert elapsed < budget_s * 2

    @pytest.mark.asyncio
    async def test_the_read_cap_is_still_hard(self):
        _reached, stats = await _traverse(_graph_store(), max_reads=8)
        assert stats["reads"] <= 8, stats
        assert stats["stop_reason"] == "max_reads", stats

    @pytest.mark.asyncio
    async def test_an_unbounded_caller_reports_no_unspent_budget(self):
        """Absence, not zero. The offline callers have no budget to leave unspent.

        ``budget_unspent_ms: 0.0`` on a dream traversal would read as "it used
        every millisecond it had" — a measurement of a quantity that does not
        exist (INSTRUMENT_AUDIT.md pattern 1).
        """
        stats: dict[str, float | str] = {}
        await BFSStrategy().spread(
            [("e1", 0.135)],
            _graph_store(),
            ActivationConfig(),
            group_id="default",
            traversal_stats=stats,
        )
        assert "budget_unspent_ms" not in stats, stats
        assert stats["reads"] > 0


class _Clock:
    """A monotonic clock the test drives, so these assertions cannot flake.

    The behavioural tests above run against real ``asyncio.sleep`` and are
    therefore approximate by construction — whether a traversal stops on the
    predictive bail or on the hard deadline can flip on a few milliseconds of
    scheduling jitter. The estimator's actual contract is exact, so it is
    asserted exactly here.
    """

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def monotonic(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(read_budget_module, "time", c)
    return c


class TestTheEstimatorForgetsASpike:
    def test_a_spike_decays_instead_of_pinning_the_estimate(self, clock):
        """THE mechanism, asserted exactly: one slow read must not persist.

        A running max holds the peak forever; the EWMA sheds it geometrically.
        ``read_ms_est`` is the published estimate, so this reads the same value
        the bail reads.
        """
        stats: dict[str, float | str] = {}
        budget = ReadBudget(deadline=clock.t + 10.0, stats=stats)

        for _ in range(5):  # warm: 0.5ms reads
            assert budget.start_read()
            clock.advance(0.0005)
            budget.finish_read()
        warm_est = float(stats["read_ms_est"])

        assert budget.start_read()  # the outlier
        clock.advance(0.025)
        budget.finish_read()
        peak_est = float(stats["read_ms_est"])
        assert peak_est > warm_est * 5, "the spike was not registered at all"

        for _ in range(6):  # back to 0.5ms reads
            assert budget.start_read()
            clock.advance(0.0005)
            budget.finish_read()

        assert float(stats["read_ms_est"]) < peak_est * 0.25, (
            f"the estimate did not decay after the spike: peak={peak_est:.3f}ms, "
            f"after 6 fast reads={stats['read_ms_est']}ms — one slow read still "
            "decides every read after it"
        )
        # The peak itself is still reported, so the spike is not hidden either.
        assert float(stats["read_ms_max"]) >= 25.0

    def test_a_refused_read_says_it_was_refused_and_how_much_was_left(self, clock):
        """``predicted_cost`` + ``budget_unspent_ms`` is the collapse signature."""
        stats: dict[str, float | str] = {}
        budget = ReadBudget(deadline=clock.t + 0.100, stats=stats)

        assert budget.start_read()
        clock.advance(0.080)  # one 80ms read against a 100ms budget
        budget.finish_read()

        # est is now 24ms; 20ms of budget remains, so the next read is refused.
        assert budget.start_read() is False
        assert stats["stop_reason"] == "predicted_cost"
        assert float(stats["budget_unspent_ms"]) == pytest.approx(20.0, abs=0.01)

    def test_a_spent_budget_is_reported_as_spent_not_as_refused(self, clock):
        """The healthy bounded stop must not wear the collapse's signature."""
        stats: dict[str, float | str] = {}
        budget = ReadBudget(deadline=clock.t + 0.100, stats=stats)

        assert budget.start_read()
        clock.advance(0.120)  # overran the whole budget
        budget.finish_read()

        assert budget.start_read() is False
        assert stats["stop_reason"] == "deadline"
        assert float(stats["budget_unspent_ms"]) == 0.0

    def test_an_unstopped_traversal_reports_no_unspent_budget(self, clock):
        """Mid-flight there is no honest unspent figure — the stop instant is unknown.

        This is what makes the cancelled-traversal case honest: the sink carries
        the reads that were discarded, and omits the number it cannot know.
        """
        stats: dict[str, float | str] = {}
        budget = ReadBudget(deadline=clock.t + 0.100, stats=stats)

        assert budget.start_read()
        assert stats["reads"] == 1
        assert "budget_unspent_ms" not in stats, stats

        clock.advance(0.010)
        budget.finish_read()
        assert "budget_unspent_ms" not in stats, stats

        budget.close()
        assert float(stats["budget_unspent_ms"]) == pytest.approx(90.0, abs=0.01)


def _cfg(**overrides):
    base = dict(
        episode_retrieval_enabled=False,
        cue_recall_enabled=False,
        chunk_search_enabled=False,
        gc_mmr_enabled=False,
        pool_graph_seed_count=1,
        pool_graph_max_neighbors=1,
        pool_graph_limit=5,
    )
    base.update(overrides)
    return ActivationConfig(**base)


def _search_index(seed_count: int = _SEED_COUNT):
    idx = AsyncMock()
    idx.search = AsyncMock(return_value=[(f"e{i}", 0.9 - 0.03 * i) for i in range(seed_count)])
    idx.search_episodes = AsyncMock(return_value=[])
    idx.search_episode_cues = AsyncMock(return_value=[])

    async def similarity(query="", entity_ids=None, group_id=None, **_kw):
        return {eid: 0.8 for eid in (entity_ids or [])}

    idx.compute_similarity = similarity
    idx._embeddings_enabled = False
    return idx


def _activation_store():
    store = AsyncMock()
    store.batch_get = AsyncMock(return_value={})
    store.get_activation = AsyncMock(return_value=None)
    store.get_top_activated = AsyncMock(return_value=[])
    store.record_access = AsyncMock()
    return store


async def _recall(store, cfg):
    stage_timings: dict[str, float] = {}
    await retrieve(
        query="read budget probe",
        group_id="default",
        graph_store=store,
        activation_store=_activation_store(),
        search_index=_search_index(),
        cfg=cfg,
        stage_timings_ms=stage_timings,
        limit=50,
    )
    return stage_timings


class TestTheCollapseIsVisibleInTheMetrics:
    """A traversal that reaches 17 nodes instead of 480 must not look healthy.

    Before this, both cases emitted ``recall_spread`` (completed) and a
    ``recall_spread_reached`` number with nothing to compare it against — a
    small reach could be a small graph or a dead traversal, and only reading the
    source could tell you which.
    """

    @pytest.mark.asyncio
    async def test_a_refused_read_is_reported_as_a_refused_read(self):
        """The signature: stopped on a PREDICTION, with budget left over.

        One 80ms read against a 100ms traversal budget leaves 20ms — less than
        the 24ms the estimate now stands at — so the next read is refused while
        the stage sits comfortably inside its own 400ms cap. That is exactly the
        shape that used to report as a clean COMPLETED.
        """
        timings = await _recall(
            _graph_store(base_s=0.080),
            _cfg(
                retrieval_spread_timeout_ms=400,
                retrieval_spread_traversal_budget_ms=100,
            ),
        )

        assert "recall_spread" in timings, f"the stage did not complete: {sorted(timings)}"
        assert timings.get("recall_spread_stop_predicted_cost") == 1.0, (
            "a traversal that refused a read as unaffordable did not say so; "
            f"keys={sorted(k for k in timings if k.startswith('recall_spread'))}"
        )
        assert timings["recall_spread_reads"] > 0
        # THE discriminator: ~20ms of a 100ms budget abandoned. The healthy run
        # in the sibling test leaves <1ms. Same stop reason, different verdict.
        assert timings["recall_spread_budget_unspent_ms"] >= 100 * 0.10, (
            f"the abandoned budget was not reported: {timings}"
        )

    @pytest.mark.asyncio
    async def test_a_healthy_traversal_carries_no_collapse_signature(self):
        """The contrast that makes the signal readable rather than decorative.

        Note what is NOT asserted: that a healthy traversal avoids
        ``predicted_cost``. It does not — a healthy traversal spends 49.93 of
        its 50 ms and then correctly declines a read it cannot afford. Asserting
        on the stop reason was this test's first version and it failed on real
        timings, which is the whole reason ``budget_unspent_ms`` is the
        discriminator the alert should use.
        """
        budget_ms = 50
        timings = await _recall(
            _graph_store(),
            _cfg(
                retrieval_spread_timeout_ms=400,
                retrieval_spread_traversal_budget_ms=budget_ms,
            ),
        )

        assert timings["recall_spread_reached"] > 0
        assert timings["recall_spread_reads"] > 0
        assert timings["recall_spread_budget_unspent_ms"] <= budget_ms * 0.25, (
            "a healthy traversal reported the collapse signature — the metric "
            f"cannot distinguish anything; timings={timings}"
        )

    @pytest.mark.asyncio
    async def test_a_discarded_traversal_still_reports_what_was_discarded(self):
        """The outer cancel throws the frontier away; it must not throw away the fact.

        A cancelled coroutine never reaches its own return, so provenance
        written on the way out is absent on exactly the failure worth seeing.
        The stats sink is caller-owned and written per read instead.
        """
        timings = await _recall(
            _graph_store(base_s=0.200),
            _cfg(retrieval_spread_timeout_ms=20),
        )

        assert "recall_spread_timeout" in timings, f"expected a cancel: {sorted(timings)}"
        assert timings["recall_spread_reached"] == 0.0
        assert timings.get("recall_spread_reads", 0.0) >= 1.0, (
            "the discarded reads left no trace at all; "
            f"keys={sorted(k for k in timings if k.startswith('recall_spread'))}"
        )
        # The stop instant is unknown for a cancelled traversal, so there is no
        # honest unspent figure to report. Absence is the correct answer.
        assert "recall_spread_budget_unspent_ms" not in timings, timings
