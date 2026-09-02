"""Ticket #21 — the keyword lane's veto in the hybrid fusion, and its fix.

WHAT WAS MEASURED (live dogfood brain, read-only LMDB forensics, 2026-07-24)
---------------------------------------------------------------------------
The three episodes the ticket called "unretrievable by their own verbatim text"
are NOT missing from any index:

    ep_293a18033a09  Episode node present, group_id="default",
                     BM25 doc_length=317, 223 unique terms,
                     EpisodeVec + CueVec + 2 EpisodeChunk vectors, deleted=0.
                     For its own opening sentence  -> BM25 global rank 2,
                     rank 1 among Episode-label docs.
                     For "what is the flip condition for usage ranking"
                     -> BM25 global rank 4, rank 1 among Episode-label docs.
    ep_103d89337b0c  same shape (doc_length 323)
    ep_bb61718b8e60  same shape (doc_length 262)

So missing/corrupt embeddings, a BM25 doc-id collision and a group/scope
mismatch are all falsified. The document is found by BM25 at the top of the
episode lane and still does not come back.

THE MECHANISM (unchanged by the fix — this arithmetic is still true)
--------------------------------------------------------------------
RRF gives rank *r* (0-based) the weight ``w / (60 + r + 1)``, so a lane weight
is really a RANK HANDICAP. At the shipped ``fts_weight=0.3 / vec_weight=0.7``
and ``k=60``:

    best   BM25-only contribution = 0.3 / 61      = 0.004918
    worst  vector-lane   contribution at rank R-1 = 0.7 / (60 + R)

and ``0.7/(60+R) > 0.3/61`` for every page size ``R <= 82``. The keyword lane's
#1 document ties the vector lane's document at rank **80**. Every shipped page
is far shorter than that — the episode lane asks for ``episode_retrieval_max``
(5) x 3 x 2 (``retrieval_strategy`` defaults to ``passage_first``) = **30**, and
``episode_retrieval_max`` is ``Field(le=20)`` so the knob cannot reach the
crossover. Lane *overlap* — the normal case — widens the band further.

Net, pre-fix: ``fused[:limit]`` returned exactly the vector lane's page, the
BM25 lane was fetched 3x deep and could only ever REORDER, and a document whose
distinguishing feature is a rare literal (an id, an error string, a function
name) was findable only if the embedding happened to agree.

THE FIX (this file now pins the fixed contract)
-----------------------------------------------
``search.py`` :func:`fts_lane_reserve` / :func:`_apply_fts_lane_reserve`. The
weights are read the way an operator would read them: ``fts_weight=0.3`` means
the keyword lane is worth 30% of the page, so its top ``round(limit * 0.3)``
documents are seated and the remaining slots fill by fused rank. Deliberately
NOT changed: the weights, ``k``, and the fused score itself — the reserve
decides who is on the page, never what a document is worth, so the "final score
is pure reciprocal rank" property recorded in earlier work is neither worsened
nor secretly patched here.

Note the fix is confined to the Helix path. The lite/SQLite twin
(``storage/sqlite/hybrid_search.py`` ``_merge_rrf``) fuses UNWEIGHTED at
``1/(k+rank)`` per lane and never had the defect — the weights are
``_merge_linear``'s, and they leaked into a rank-fusion formula on the Helix
side only.
"""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig, EmbeddingConfig, HelixDBConfig
from engram.storage.helix.search import (
    _OVERFETCH_FACTOR,
    _RRF_K,
    HelixSearchIndex,
    fts_lane_reserve,
    get_rrf_lane_stats,
)

GROUP = "default"
TARGET = "ep_293a18033a09"


def _index(
    bm25_ids: list[str],
    vector_ids: list[str],
    embed_config: EmbeddingConfig | None = None,
) -> HelixSearchIndex:
    """A search index whose two lanes return exactly the given ranked ids."""

    class Provider:
        def dimension(self) -> int:
            return 8

    class FakeClient:
        _native_transport = None

        async def query(self, endpoint: str, payload: dict):
            k = int(payload["k"])
            if endpoint == "search_episodes_bm25":
                return [
                    {"episode_id": eid, "group_id": GROUP, "score": float(len(bm25_ids) - i)}
                    for i, eid in enumerate(bm25_ids[:k])
                ]
            if endpoint == "search_episode_vectors_filtered":
                return [
                    {"episode_id": eid, "group_id": GROUP, "distance": 0.01 * (i + 1)}
                    for i, eid in enumerate(vector_ids[:k])
                ]
            raise AssertionError(f"unexpected endpoint {endpoint}")

    index = HelixSearchIndex(
        HelixDBConfig(),
        Provider(),
        embed_config or EmbeddingConfig(),
        client=FakeClient(),
        bm25_breaker_enabled=False,
    )

    async def _embed(_text: str) -> list[float]:
        return [0.1] * 8

    index._embed_text = _embed  # type: ignore[method-assign]
    return index


def _disjoint_lanes(limit: int, target_in_vectors: bool = False) -> tuple[list[str], list[str]]:
    """Cleanest statement of the inequality: the two lanes share no documents.

    Every vector-lane document therefore carries ONLY its vector contribution,
    ``0.7 / (60 + rank + 1)``, so the crossover is exactly where the arithmetic
    in the module docstring puts it. Real lanes overlap heavily.
    """
    vector_ids = [f"ep_vec_{i:04d}" for i in range(limit)]
    bm25_ids = [TARGET, *[f"ep_bm_{i:04d}" for i in range(limit * _OVERFETCH_FACTOR)]]
    if target_in_vectors:
        vector_ids[-1] = TARGET
    return bm25_ids, vector_ids


def _overlapping_lanes(limit: int, target_in_vectors: bool = False) -> tuple[list[str], list[str]]:
    """Realistic shape: the ANN page is also near the top of the BM25 lane."""
    filler = [f"ep_filler_{i:04d}" for i in range(limit * _OVERFETCH_FACTOR)]
    vector_ids = list(filler[:limit])
    if target_in_vectors:
        vector_ids[-1] = TARGET
    bm25_ids = [TARGET, *filler]
    return bm25_ids, vector_ids


def _shipped_episode_lane_limit(cfg: ActivationConfig) -> int:
    """The limit pipeline.py Step 1.1 passes to search_episodes."""
    mult = 2 if cfg.retrieval_strategy == "passage_first" else 1
    return cfg.episode_retrieval_max * 3 * mult


def _veto_crossover() -> int:
    """Smallest page size at which the BM25 lane's #1 outranks the vector tail."""
    emb = EmbeddingConfig()
    best_fts_only = emb.fts_weight / (_RRF_K + 1)
    limit = 1
    while emb.vec_weight / (_RRF_K + limit) > best_fts_only:
        limit += 1
    return limit


# ---------------------------------------------------------------------------
# The arithmetic — UNCHANGED by the fix, and the reason the reserve exists
# ---------------------------------------------------------------------------


def test_the_crossover_is_where_the_arithmetic_says_it_is():
    """The weight ratio really is an 80-rank handicap. Still true post-fix."""
    emb = EmbeddingConfig()
    assert (emb.fts_weight, emb.vec_weight) == (0.3, 0.7)
    assert _RRF_K == 60
    crossover = _veto_crossover()
    assert crossover == 83
    # one below: the whole vector page outranks the best BM25-only document
    assert emb.vec_weight / (_RRF_K + crossover - 1) > emb.fts_weight / (_RRF_K + 1)
    # at the crossover: it no longer does
    assert emb.vec_weight / (_RRF_K + crossover) < emb.fts_weight / (_RRF_K + 1)


def test_shipped_defaults_sit_inside_the_veto_band():
    """Shipped DEFAULTS (not the live process — see STANDING_GOAL 2.11).

    The reserve is the only thing standing between these defaults and a keyword
    lane that cannot place. If this assertion ever flips because the page grew
    past 83, the reserve becomes a no-op rather than wrong.
    """
    cfg = ActivationConfig()
    assert cfg.retrieval_strategy == "passage_first"
    assert cfg.episode_retrieval_max == 5
    assert _shipped_episode_lane_limit(cfg) == 30 < _veto_crossover()


def test_the_reserve_is_the_weight_read_as_a_share_of_the_page():
    """``fts_weight`` now means what an operator would assume it means."""
    assert fts_lane_reserve(30, 0.3, 0.7) == 9
    assert fts_lane_reserve(20, 0.3, 0.7) == 6
    assert fts_lane_reserve(10, 0.3, 0.7) == 3
    # monotone in the knob, and 0.0 is exactly the pre-fix behaviour
    assert fts_lane_reserve(30, 0.0, 1.0) == 0
    assert fts_lane_reserve(30, 0.5, 0.5) == 15
    assert fts_lane_reserve(30, 1.0, 0.0) == 30
    # a tiny weight still buys one slot rather than silently rounding to zero
    assert fts_lane_reserve(30, 0.01, 0.99) == 1
    assert fts_lane_reserve(0, 0.3, 0.7) == 0


# ---------------------------------------------------------------------------
# The fix, at the shipped limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bm25_rank_one_episode_reaches_the_caller_at_the_shipped_limit():
    """The measured live case: BM25 rank 1, absent from the ANN page.

    This is ticket #21's episode. Pre-fix it was rank 1 in the BM25 episode
    lane on the live index and still could not reach the caller.
    """
    limit = _shipped_episode_lane_limit(ActivationConfig())
    assert limit == 30, "shipped episode-lane limit moved; re-derive the band"
    bm25_ids, vector_ids = _disjoint_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("verbatim self query", group_id=GROUP, limit=limit)
    returned = [eid for eid, _ in fused]

    assert TARGET in returned
    assert returned != vector_ids, "the fused page is no longer just the vector lane's page"
    assert len(returned) == limit


@pytest.mark.asyncio
async def test_the_bm25_lane_contributes_its_weight_share_of_new_documents():
    """Not just the target: the keyword lane places its share, and no more.

    The bound matters as much as the capability — an unbounded keyword reserve
    would trade one silent distortion for another.
    """
    limit = _shipped_episode_lane_limit(ActivationConfig())
    bm25_ids, vector_ids = _disjoint_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)
    returned = [eid for eid, _ in fused]

    reserve = fts_lane_reserve(limit, 0.3, 0.7)
    bm25_only = set(bm25_ids) - set(vector_ids)
    placed = bm25_only & set(returned)
    assert len(placed) == reserve == 9
    # the vector lane keeps the rest of the page, in its own order
    kept_vectors = [eid for eid in returned if eid in set(vector_ids)]
    assert kept_vectors == vector_ids[: limit - reserve]


@pytest.mark.asyncio
async def test_the_reserve_seats_the_keyword_lane_in_its_own_rank_order():
    """The seated documents are the BM25 lane's TOP ones, not an arbitrary 9."""
    limit = _shipped_episode_lane_limit(ActivationConfig())
    bm25_ids, vector_ids = _disjoint_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    returned = [eid for eid, _ in await index.search_episodes("q", group_id=GROUP, limit=limit)]

    placed = [eid for eid in returned if eid in set(bm25_ids) - set(vector_ids)]
    assert placed == bm25_ids[: fts_lane_reserve(limit, 0.3, 0.7)]


@pytest.mark.asyncio
async def test_agreeing_lanes_produce_a_byte_identical_page():
    """No gratuitous change: the reserve spends slots only on disagreement.

    A keyword hit the vector lane also found already occupies one of the
    keyword lane's own reserved slots, so the common case is untouched.
    """
    limit = _shipped_episode_lane_limit(ActivationConfig())
    vector_ids = [f"ep_a_{i:04d}" for i in range(limit)]
    # BM25 ranks the same documents in the same order, then trails off
    bm25_ids = [*vector_ids, *[f"ep_tail_{i:04d}" for i in range(limit * _OVERFETCH_FACTOR)]]

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)

    assert [eid for eid, _ in fused] == vector_ids


@pytest.mark.asyncio
async def test_fts_weight_zero_restores_the_pre_fix_page_exactly():
    """The knob is honest at its endpoint, which is what makes it a knob."""
    limit = _shipped_episode_lane_limit(ActivationConfig())
    bm25_ids, vector_ids = _disjoint_lanes(limit)

    index = _index(bm25_ids, vector_ids, EmbeddingConfig(fts_weight=0.0, vec_weight=1.0))
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)

    assert [eid for eid, _ in fused] == vector_ids
    assert TARGET not in [eid for eid, _ in fused]


@pytest.mark.asyncio
async def test_the_reserve_does_not_touch_the_score_scale():
    """Downstream multiplies the fused score into a candidate weight.

    ``pipeline.py`` Step 1.1 does ``weight_semantic * sem_sim * mult``, so a
    reserve that inflated a promoted document's score would be a *second*
    silent distortion. Seated documents keep the score the fusion gave them.
    """
    limit = _shipped_episode_lane_limit(ActivationConfig())
    bm25_ids, vector_ids = _disjoint_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)
    scores = dict(fused)

    assert fused[0][1] == pytest.approx(1.0)
    assert [s for _, s in fused] == sorted((s for _, s in fused), reverse=True)
    # TARGET's raw RRF score is 0.3/61; the top document's is 0.7/61.
    assert scores[TARGET] == pytest.approx((0.3 / 61) / (0.7 / 61))


@pytest.mark.asyncio
async def test_the_reserve_reports_what_it_did():
    """STANDING_GOAL 2.2: "0 promotions" and "not wired up" must not look alike."""
    limit = _shipped_episode_lane_limit(ActivationConfig())

    before = get_rrf_lane_stats()
    agreeing = [f"ep_same_{i:04d}" for i in range(limit)]
    await _index(agreeing, agreeing).search_episodes("q", group_id=GROUP, limit=limit)
    agreed = get_rrf_lane_stats()
    assert agreed["fused_pages"] == before["fused_pages"] + 1
    assert agreed["fts_reserve_promotions"] == before["fts_reserve_promotions"]

    bm25_ids, vector_ids = _disjoint_lanes(limit)
    await _index(bm25_ids, vector_ids).search_episodes("q", group_id=GROUP, limit=limit)
    after = get_rrf_lane_stats()
    assert after["fused_pages"] == agreed["fused_pages"] + 1
    assert after["fts_reserve_pages"] == agreed["fts_reserve_pages"] + 1
    assert after["fts_reserve_promotions"] == agreed["fts_reserve_promotions"] + 9


@pytest.mark.parametrize("episode_retrieval_max", [1, 5, 10, 20])
@pytest.mark.parametrize("passage_first", [True, False])
@pytest.mark.asyncio
async def test_every_reachable_config_can_place_a_bm25_only_document(
    episode_retrieval_max: int, passage_first: bool
):
    """No setting of the shipped knobs re-opens the veto.

    ``episode_retrieval_max`` is bounded 0..20 by its ``Field()``; with or
    without ``passage_first`` the lane limit spans 3..120, i.e. from far inside
    the old veto band to past its crossover.
    """
    limit = episode_retrieval_max * 3 * (2 if passage_first else 1)
    bm25_ids, vector_ids = _disjoint_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)

    assert TARGET in [eid for eid, _ in fused]


@pytest.mark.asyncio
async def test_lane_overlap_no_longer_widens_a_veto_band():
    """Overlap used to make the veto *worse* than the pure crossover.

    A vector-lane document that also appears in the BM25 lane collects both
    contributions, so it outscored the BM25 lane's unique #1 even above the
    crossover. The reserve is rank-based, so overlap no longer buys a veto.
    """
    limit = _veto_crossover() + 10
    bm25_ids, vector_ids = _overlapping_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)

    assert TARGET in [eid for eid, _ in fused]


@pytest.mark.asyncio
async def test_the_same_episode_returns_when_the_ann_lane_also_finds_it():
    """Control matching the ticket's own controls (ep_36ae5019b96a et al.).

    Those episodes returned at rank 1 even pre-fix — because the ANN lane found
    them too, not because their index rows are different. Still true.
    """
    limit = _shipped_episode_lane_limit(ActivationConfig())
    bm25_ids, vector_ids = _overlapping_lanes(limit, target_in_vectors=True)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)

    assert TARGET in [eid for eid, _ in fused]


@pytest.mark.asyncio
async def test_lane_fetch_depths_are_unchanged():
    """The fix is in the fusion, not the fetch. Pinned so a later 'tidy' shows.

    The 3x BM25 overfetch exists for post-hoc group filtering (vectors are
    filtered inside HelixDB when ``group_id`` is set). Pre-fix it also bought
    nothing at all; it now feeds a reserve that reads only the lane's top
    ``round(limit * fts_weight)``, so the depth is still not load-bearing for
    the reserve and remains a group-filter allowance.
    """
    limit = _shipped_episode_lane_limit(ActivationConfig())
    seen: dict[str, int] = {}

    class Provider:
        def dimension(self) -> int:
            return 8

    bm25_ids, vector_ids = _overlapping_lanes(limit)

    class RecordingClient:
        _native_transport = None

        async def query(self, endpoint: str, payload: dict):
            seen[endpoint] = int(payload["k"])
            if endpoint == "search_episodes_bm25":
                return [
                    {"episode_id": e, "group_id": GROUP, "score": 1.0}
                    for e in bm25_ids[: int(payload["k"])]
                ]
            return [
                {"episode_id": e, "group_id": GROUP, "distance": 0.1}
                for e in vector_ids[: int(payload["k"])]
            ]

    index = HelixSearchIndex(
        HelixDBConfig(),
        Provider(),
        EmbeddingConfig(),
        client=RecordingClient(),
        bm25_breaker_enabled=False,
    )

    async def _embed(_t: str) -> list[float]:
        return [0.1] * 8

    index._embed_text = _embed  # type: ignore[method-assign]
    await index.search_episodes("q", group_id=GROUP, limit=limit)

    assert seen["search_episodes_bm25"] == limit * _OVERFETCH_FACTOR
    assert seen["search_episode_vectors_filtered"] == limit


# ---------------------------------------------------------------------------
# Blast radius: the entity and cue lanes had the identical veto
# ---------------------------------------------------------------------------


def _index_for(bm25_endpoint: str, vec_endpoint: str, id_field: str, ranked, vector_ids):
    class Provider:
        def dimension(self) -> int:
            return 8

    class FakeClient:
        _native_transport = None

        async def query(self, endpoint: str, payload: dict):
            k = int(payload["k"])
            if endpoint == bm25_endpoint:
                return [
                    {id_field: i, "group_id": GROUP, "score": float(len(ranked) - n)}
                    for n, i in enumerate(ranked[:k])
                ]
            if endpoint == vec_endpoint:
                return [
                    {id_field: i, "group_id": GROUP, "distance": 0.01 * (n + 1)}
                    for n, i in enumerate(vector_ids[:k])
                ]
            raise AssertionError(f"unexpected endpoint {endpoint}")

    index = HelixSearchIndex(
        HelixDBConfig(),
        Provider(),
        EmbeddingConfig(),
        client=FakeClient(),
        bm25_breaker_enabled=False,
    )

    async def _embed(_t: str) -> list[float]:
        return [0.1] * 8

    index._embed_text = _embed  # type: ignore[method-assign]
    return index


@pytest.mark.asyncio
async def test_entity_lane_is_fixed_too():
    """``HelixSearchIndex.search`` is the PRIMARY recall lane.

    Its blast radius was wider than ticket #21's episodes: entity recall is the
    pool everything else scores against.
    """
    limit = 30
    vector_ids = [f"en_vec_{i:04d}" for i in range(limit)]
    bm25_ids = ["en_target", *[f"en_bm_{i:04d}" for i in range(limit * _OVERFETCH_FACTOR)]]

    index = _index_for(
        "search_entities_bm25", "search_entity_vectors_filtered", "entity_id", bm25_ids, vector_ids
    )
    returned = [eid for eid, _ in await index.search("q", group_id=GROUP, limit=limit)]

    assert "en_target" in returned
    assert returned != vector_ids


@pytest.mark.asyncio
async def test_cue_lane_is_fixed_too():
    limit = 30
    vector_ids = [f"cue_vec_{i:04d}" for i in range(limit)]
    bm25_ids = ["cue_target", *[f"cue_bm_{i:04d}" for i in range(limit * _OVERFETCH_FACTOR)]]

    index = _index_for(
        "search_cues_bm25", "search_cue_vectors_filtered", "episode_id", bm25_ids, vector_ids
    )
    returned = [eid for eid, _ in await index.search_episode_cues("q", group_id=GROUP, limit=limit)]

    assert "cue_target" in returned
    assert returned != vector_ids
