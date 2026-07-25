"""Ticket #21 — why a verbatim-matching episode never reaches the caller.

Measured on the LIVE dogfood brain (read-only LMDB forensics, 2026-07-24), the
three episodes the ticket calls "unretrievable by their own verbatim text" are
NOT missing from any index:

    ep_293a18033a09  Episode node present, group_id="default",
                     BM25 doc_length=317, 223 unique terms,
                     EpisodeVec + CueVec + 2 EpisodeChunk vectors, deleted=0.
                     For its own opening sentence  -> BM25 global rank 2,
                     rank 1 among Episode-label docs.
                     For "what is the flip condition for usage ranking"
                     -> BM25 global rank 4, rank 1 among Episode-label docs.
    ep_103d89337b0c  same shape (doc_length 323)
    ep_bb61718b8e60  same shape (doc_length 262)

So the ticket's stated hypotheses — missing/corrupt embeddings, a BM25 doc-id
collision, a group/scope mismatch — are all falsified. The document is found by
BM25 at the top of the episode lane and still does not come back.

The mechanism is in the FUSION, not the index. ``HelixSearchIndex.search_episodes``
does:

    bm25_fetch_limit = limit * _OVERFETCH_FACTOR      (3x)
    vec_fetch_limit  = limit                          (when group_id is set)
    fused            = _rrf_fusion(fts, vec, 0.3, 0.7)[:limit]

RRF gives rank r (0-based) the weight ``w / (60 + r + 1)``. So:

    worst possible vector-lane contribution = 0.7 / (60 + limit)
    best   possible BM25-only  contribution = 0.3 / 61  = 0.004918

and 0.7 / (60 + limit) > 0.3 / 61 for every ``limit <= 82``. Below that
crossover EVERY document the vector lane returned outranks the BM25 lane's #1
document, and ``[:limit]`` therefore returns exactly the vector lane's page.
**The BM25 lane cannot introduce a single document the ANN lane missed.** It is
fetched 3x deep, it is the slowest native call in the stage (it is why the BM25
circuit breaker exists), and its unique contribution is discarded by
construction — the repo's signature computed-but-silently-inert shape.

Shipped default limit: ``episode_retrieval_max`` (5) * 3 * 2 (retrieval_strategy
defaults to ``passage_first``, which doubles the budget) = 30 — deep inside the
veto band. Even the configured maximum (``episode_retrieval_max`` is capped at
20 -> 20 * 3 = 60 without passage_first) stays inside it. Lane OVERLAP —
the normal case, since the ANN page is usually near the top of the BM25 lane
too — widens the band further, because an overlapping document collects both
contributions.

These tests pin the DEFECT as measured. They are expected to FAIL the day the
fusion is fixed; that is the point — see
``test_above_the_crossover_the_bm25_lane_can_contribute`` for the control that
shows the harness itself is sound.
"""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig, EmbeddingConfig, HelixDBConfig
from engram.storage.helix.search import _OVERFETCH_FACTOR, _RRF_K, HelixSearchIndex

GROUP = "default"
TARGET = "ep_293a18033a09"


def _index(bm25_ids: list[str], vector_ids: list[str]) -> HelixSearchIndex:
    """A search index whose two lanes return exactly the given ranked ids."""

    class Provider:
        def dimension(self) -> int:
            return 8

    class FakeClient:
        _native_transport = None

        async def query(self, endpoint: str, payload: dict):
            if endpoint == "search_episodes_bm25":
                k = int(payload["k"])
                return [
                    {"episode_id": eid, "group_id": GROUP, "score": float(len(bm25_ids) - i)}
                    for i, eid in enumerate(bm25_ids[:k])
                ]
            if endpoint == "search_episode_vectors_filtered":
                k = int(payload["k"])
                return [
                    {"episode_id": eid, "group_id": GROUP, "distance": 0.01 * (i + 1)}
                    for i, eid in enumerate(vector_ids[:k])
                ]
            raise AssertionError(f"unexpected endpoint {endpoint}")

    index = HelixSearchIndex(
        HelixDBConfig(),
        Provider(),
        EmbeddingConfig(),
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
    in the module docstring puts it. Real lanes overlap heavily, which only
    makes the veto worse — see
    ``test_lane_overlap_widens_the_veto_band``.
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
    """Smallest limit at which the BM25 lane's #1 can outrank the vector tail."""
    emb = EmbeddingConfig()
    best_fts_only = emb.fts_weight / (_RRF_K + 1)
    limit = 1
    while emb.vec_weight / (_RRF_K + limit) > best_fts_only:
        limit += 1
    return limit


# ---------------------------------------------------------------------------
# The defect, at the shipped limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bm25_rank_one_episode_is_dropped_at_the_shipped_limit():
    """The measured live case: BM25 rank 1, absent from the ANN page -> gone.

    This is ticket #21's episode. It is rank 1 in the BM25 episode lane on the
    live index and it still cannot reach the caller.
    """
    limit = _shipped_episode_lane_limit(ActivationConfig())
    assert limit == 30, "shipped episode-lane limit moved; re-derive the band"
    bm25_ids, vector_ids = _disjoint_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("verbatim self query", group_id=GROUP, limit=limit)
    returned = [eid for eid, _ in fused]

    assert TARGET not in returned
    assert returned == vector_ids, "the fused page is exactly the vector lane's page"


@pytest.mark.asyncio
async def test_the_bm25_lane_contributes_zero_new_documents_at_the_shipped_limit():
    """Not just the target: NOTHING BM25-only survives the fusion."""
    limit = _shipped_episode_lane_limit(ActivationConfig())
    bm25_ids, vector_ids = _overlapping_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)

    bm25_only = set(bm25_ids) - set(vector_ids)
    assert bm25_only, "fixture must actually have BM25-unique documents"
    assert not (bm25_only & {eid for eid, _ in fused})


@pytest.mark.asyncio
async def test_bm25_lane_overfetches_three_times_what_it_can_ever_use():
    """The wasted work is real: 3x the depth, zero possible contribution."""
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


@pytest.mark.parametrize("episode_retrieval_max", [1, 5, 10, 20])
@pytest.mark.parametrize("passage_first", [True, False])
@pytest.mark.asyncio
async def test_every_reachable_config_stays_inside_the_veto_band(
    episode_retrieval_max: int, passage_first: bool
):
    """No setting of the shipped knobs escapes it.

    ``episode_retrieval_max`` is bounded 0..20 by its Field(); with or without
    ``passage_first`` the lane limit tops out at 120 — but 120 is only reachable
    with passage_first, and every value at or below the crossover vetoes BM25.
    """
    limit = episode_retrieval_max * 3 * (2 if passage_first else 1)
    crossover = _veto_crossover()
    bm25_ids, vector_ids = _disjoint_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)
    returned = [eid for eid, _ in fused]

    if limit < crossover:
        assert TARGET not in returned
        assert returned == vector_ids
    else:
        assert TARGET in returned


# ---------------------------------------------------------------------------
# Controls — the harness can produce the healthy answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_above_the_crossover_the_bm25_lane_can_contribute():
    """Same harness, limit past the crossover: the target comes back.

    Proves the assertions above are about the fusion arithmetic and not about a
    fixture that simply never returns the target.
    """
    crossover = _veto_crossover()
    limit = crossover + 10
    bm25_ids, vector_ids = _disjoint_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)

    assert TARGET in [eid for eid, _ in fused]


@pytest.mark.asyncio
async def test_lane_overlap_widens_the_veto_band():
    """Above the pure crossover the veto STILL holds once the lanes overlap.

    Documents that ``_veto_crossover()`` is a LOWER bound on how deep you would
    have to fetch to un-veto BM25, not the real threshold: a vector-lane
    document that also appears in the BM25 lane collects both contributions.
    """
    limit = _veto_crossover() + 10
    bm25_ids, vector_ids = _overlapping_lanes(limit)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)

    assert TARGET not in [eid for eid, _ in fused]


@pytest.mark.asyncio
async def test_the_same_episode_returns_when_the_ann_lane_also_finds_it():
    """Control matching the ticket's own controls (ep_36ae5019b96a et al.).

    Those episodes return at rank 1 under the identical probe — because the ANN
    lane found them too, not because their index rows are different.
    """
    limit = _shipped_episode_lane_limit(ActivationConfig())
    bm25_ids, vector_ids = _overlapping_lanes(limit, target_in_vectors=True)

    index = _index(bm25_ids, vector_ids)
    fused = await index.search_episodes("q", group_id=GROUP, limit=limit)

    assert TARGET in [eid for eid, _ in fused]


def test_the_crossover_is_where_the_arithmetic_says_it_is():
    """Documents the inequality itself, independently of the search index."""
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
    """Shipped DEFAULTS (not the live process — see STANDING_GOAL 2.11)."""
    cfg = ActivationConfig()
    assert cfg.retrieval_strategy == "passage_first"
    assert cfg.episode_retrieval_max == 5
    assert _shipped_episode_lane_limit(cfg) == 30 < _veto_crossover()


# ---------------------------------------------------------------------------
# Blast radius: the same shape in the entity and cue lanes
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
async def test_entity_lane_has_the_identical_veto():
    """HelixSearchIndex.search (search.py:1600/1676) repeats the shape verbatim.

    Same ``vec_fetch_limit = limit`` / ``bm25_fetch_limit = limit * 3`` /
    ``fused[:limit]``. The entity lane is the PRIMARY recall lane, so the blast
    radius of this defect is wider than ticket #21's episodes.
    """
    limit = 30
    vector_ids = [f"en_vec_{i:04d}" for i in range(limit)]
    bm25_ids = ["en_target", *[f"en_bm_{i:04d}" for i in range(limit * _OVERFETCH_FACTOR)]]

    index = _index_for(
        "search_entities_bm25", "search_entity_vectors_filtered", "entity_id", bm25_ids, vector_ids
    )
    fused = await index.search("q", group_id=GROUP, limit=limit)
    returned = [eid for eid, _ in fused]

    assert "en_target" not in returned
    assert returned == vector_ids


@pytest.mark.asyncio
async def test_cue_lane_has_the_identical_veto():
    """search_episode_cues (search.py:1813/1824) repeats it too."""
    limit = 30
    vector_ids = [f"cue_vec_{i:04d}" for i in range(limit)]
    bm25_ids = ["cue_target", *[f"cue_bm_{i:04d}" for i in range(limit * _OVERFETCH_FACTOR)]]

    index = _index_for(
        "search_cues_bm25", "search_cue_vectors_filtered", "episode_id", bm25_ids, vector_ids
    )
    fused = await index.search_episode_cues("q", group_id=GROUP, limit=limit)
    returned = [eid for eid, _ in fused]

    assert "cue_target" not in returned
    assert returned == vector_ids
