"""Ticket 28 — the candidate-pool depth cap must BE the config knob, not shadow it.

`compute_dynamic_limits` states its own contract in its docstring: "At 1k
entities the limits equal cfg defaults." Six of its seven keys honoured that.
`pool_total_limit` did not — it was recomputed as the SUM of the four
contributing pool caps (`pool_search + pool_activation + pool_graph + pool_wm`),
so `cfg.pool_total_limit` had zero read sites anywhere in the repo.

Two consequences, both bad:

1. **The knob lied.** `ENGRAM_ACTIVATION__POOL_TOTAL_LIMIT` at 20, 80, 400 or
   1000 all produced the identical cap (85 on the live brain). An operator
   raising it to widen the recall pool observed no change and would reasonably
   conclude depth is not the bottleneck — a false negative manufactured by a
   config field, on the exact lever of the "recall returns only 5-7 rows"
   investigation.

2. **The cap could barely bind.** Being the sum of its own four contributing
   pools, it was >= the union bound of those pools by construction. It could
   only ever trim candidates unique to the *entity-query* pool (a fifth pool,
   not in the sum). A depth cap that structurally cannot cap is a shipped
   mechanism whose output nothing consumes.

The clamp bounds on the shadow expression -- `_clamp(..., 20, 1000)` -- are
copied verbatim from the config field's own `ge=20, le=1000`, which is the tell
that `cfg.pool_total_limit` was always meant to be the base and the sum was a
placeholder that outlived its author.
"""

from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.retrieval.candidate_pool import compute_dynamic_limits, generate_candidates
from engram.retrieval.router import QueryType

# Derived, not hardcoded, so this is a ratchet: any key added to
# `compute_dynamic_limits` later is automatically held to the same contract and
# cannot be introduced as another shadow value.
_LIMIT_KEYS = tuple(compute_dynamic_limits(1000, ActivationConfig()))


def test_every_returned_key_names_a_real_config_field():
    """Guards the ratchet above: the key set must stay a config-field mapping."""
    assert _LIMIT_KEYS
    unknown = [k for k in _LIMIT_KEYS if k not in ActivationConfig.model_fields]
    assert not unknown, f"returned keys with no matching config field: {unknown}"


class TestEveryLimitReadsItsConfigField:
    def test_at_1k_every_limit_equals_its_config_field(self):
        """The function's own docstring contract, applied to ALL seven keys.

        The pre-existing test asserted this for three of them and quietly
        omitted `pool_total_limit` — the only one that would have failed.
        """
        cfg = ActivationConfig()
        limits = compute_dynamic_limits(1000, cfg, QueryType.DEFAULT)

        mismatched = {
            key: (limits[key], getattr(cfg, key))
            for key in _LIMIT_KEYS
            if limits[key] != getattr(cfg, key)
        }
        assert not mismatched, (
            "at 1k entities with no query-type multiplier every limit must equal "
            f"its config field; these did not (got, expected): {mismatched}"
        )

    @pytest.mark.parametrize("key", _LIMIT_KEYS)
    def test_raising_the_config_field_raises_the_limit(self, key):
        """A knob that cannot move its own limit is a knob that lies.

        Uses the field's own declared min/max so the assertion is about the
        read site, not about clamp headroom.
        """
        field = ActivationConfig.model_fields[key]
        lo_bound = next(m.ge for m in field.metadata if hasattr(m, "ge"))
        hi_bound = next(m.le for m in field.metadata if hasattr(m, "le"))

        lo = compute_dynamic_limits(1000, ActivationConfig(**{key: lo_bound}))[key]
        hi = compute_dynamic_limits(1000, ActivationConfig(**{key: hi_bound}))[key]

        assert lo < hi, (
            f"cfg.{key} moved from {lo_bound} to {hi_bound} and the computed "
            f"limit did not move ({lo} -> {hi}); the config field has no read site"
        )

    def test_cap_is_not_the_sum_of_the_pools_it_caps(self):
        """The cap is read from its own field, not derived from its inputs.

        With the shadow expression the cap moved whenever a pool moved, which
        is why it was inert rather than merely mis-valued. Parity of the
        default VALUES is deliberate (config.py: 85 keeps the pre-ticket-28
        lane width); parity of the EXPRESSION is the defect.
        """
        base = ActivationConfig()
        widened = ActivationConfig(pool_search_limit=base.pool_search_limit * 2)
        for total_entities in (1000, 5000, 9399, 50000):
            lo = compute_dynamic_limits(total_entities, base)
            hi = compute_dynamic_limits(total_entities, widened)
            if lo["pool_search_limit"] < 200:  # below the clamp the pool itself must move
                assert hi["pool_search_limit"] > lo["pool_search_limit"]
            assert hi["pool_total_limit"] == lo["pool_total_limit"], (
                f"at {total_entities} entities widening pool_search_limit moved the cap "
                f"({lo['pool_total_limit']} -> {hi['pool_total_limit']}); it is being "
                "derived from the pools again"
            )

    def test_cap_still_scales_with_corpus_size(self):
        """Making the knob live must not flatten the sqrt(N/1000) scaling.

        Without scaling, a 50k-entity brain would build 450 candidates' worth of
        pools and then truncate them to 80 — a new silent-inert lever pointed at
        the six sibling limits.
        """
        cfg = ActivationConfig()
        at_1k = compute_dynamic_limits(1000, cfg)["pool_total_limit"]
        at_5k = compute_dynamic_limits(5000, cfg)["pool_total_limit"]
        at_50k = compute_dynamic_limits(50000, cfg)["pool_total_limit"]

        assert at_1k == cfg.pool_total_limit
        assert at_5k > at_1k
        assert at_50k > at_5k


# ---------------------------------------------------------------------------
# End-to-end: the cap must actually truncate the merged candidate pool.
# The arithmetic test above would still pass if line ~1055 stopped reading the
# key at all, so this one drives the real orchestration.
# ---------------------------------------------------------------------------

_QUERY = "Who is Alice Smith"
_SEARCH_HITS = [(f"s{i:02d}", 0.9 - i * 0.01) for i in range(30)]
_NAMED_ENTITIES = [
    Entity(id=f"n{i:02d}", name="Alice Smith", entity_type="person") for i in range(20)
]


def _stores():
    search_index = AsyncMock()
    search_index.search = AsyncMock(return_value=list(_SEARCH_HITS))
    search_index.compute_similarity = AsyncMock(return_value={})
    search_index.search_episodes = AsyncMock(return_value=[])

    activation_store = AsyncMock()
    activation_store.get_top_activated = AsyncMock(return_value=[])
    activation_store.batch_get = AsyncMock(return_value={})

    graph_store = AsyncMock()
    graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])
    graph_store.find_entity_candidates = AsyncMock(return_value=list(_NAMED_ENTITIES))
    return search_index, activation_store, graph_store


async def _run(cfg):
    search_index, activation_store, graph_store = _stores()
    return await generate_candidates(
        query=_QUERY,
        group_id="default",
        search_index=search_index,
        activation_store=activation_store,
        graph_store=graph_store,
        cfg=cfg,
        now=1000.0,
        total_entities=1000,
        query_type=QueryType.DEFAULT,
    )


class TestCapTruncatesTheMergedPool:
    @pytest.mark.asyncio
    async def test_uncapped_pool_is_the_full_union(self):
        """Control: with the cap wide open both pools survive the merge."""
        candidates = await _run(ActivationConfig(pool_total_limit=1000))
        assert len(candidates) == len(_SEARCH_HITS) + len(_NAMED_ENTITIES) == 50

    @pytest.mark.asyncio
    async def test_lowering_the_knob_truncates_the_pool(self):
        """The knob, set below the union, must cut the pool to exactly its value."""
        candidates = await _run(ActivationConfig(pool_total_limit=20))
        assert len(candidates) == 20

    @pytest.mark.asyncio
    async def test_knob_sweep_is_monotone_end_to_end(self):
        """Three settings, three distinct pool depths — measured through the
        real orchestration, not through `compute_dynamic_limits` alone."""
        depths = [len(await _run(ActivationConfig(pool_total_limit=v))) for v in (20, 35, 1000)]
        assert depths == [20, 35, 50], depths
