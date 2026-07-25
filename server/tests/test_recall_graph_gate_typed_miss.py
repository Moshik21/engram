"""Ticket 19: a gate refusal must not impersonate a miss.

``GatedGraphStore`` suppresses secondary graph reads once a recall preflight
probe times out. That is the right policy. What was wrong is that it returned
``None``/``[]`` in ~0.11 ms, so the caller was told "no such episode" when the
honest answer was "I stopped trying" — and ``recall_stats_timeout`` /
``graph_expand_timeout`` fire routinely, so this is the COMMON path.

Both tests below drive a REAL ``asyncio.wait_for`` timeout through the real
``retrieve()`` pipeline (a slow ``expand_query_from_graph``); nothing here
mocks the gate's return value, because mocking the symptom would let the bug
survive underneath.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.retrieval.pipeline import retrieve
from engram.retrieval.recall_graph_gate import (
    GATE_REFUSAL_METRIC_KEY,
    GatedGraphStore,
    GraphGateTimeoutError,
)
from engram.retrieval.working_memory import WorkingMemoryBuffer
from engram.utils.dates import utc_now

GROUP = "default"
PRESENT_EPISODE_ID = "ep_present"
ABSENT_EPISODE_ID = "ep_absent"
EPISODE_CONTENT = "The native transport releases the GIL on every query call."


def _episode(episode_id: str) -> Episode:
    return Episode(
        id=episode_id,
        content=EPISODE_CONTENT,
        source="test",
        status=EpisodeStatus.COMPLETED,
        projection_state=EpisodeProjectionState.CUE_ONLY,
        group_id=GROUP,
        created_at=utc_now(),
    )


def _entity(entity_id: str) -> Entity:
    return Entity(
        id=entity_id,
        name=f"Entity {entity_id}",
        entity_type="Thing",
        summary="a summary with real text in it",
        group_id=GROUP,
    )


def _graph_store():
    """Store where one episode EXISTS and one genuinely does not."""

    async def _get_episode(episode_id, _group_id=GROUP, *_a, **_kw):
        return _episode(episode_id) if episode_id == PRESENT_EPISODE_ID else None

    async def _get_entity(entity_id, _group_id=GROUP, *_a, **_kw):
        return _entity(entity_id)

    store = AsyncMock()
    store.get_episode_by_id = AsyncMock(side_effect=_get_episode)
    store.get_entity = AsyncMock(side_effect=_get_entity)
    store.get_active_neighbors_with_weights = AsyncMock(return_value=[])
    store.get_relationships = AsyncMock(return_value=[])
    store.find_entities = AsyncMock(return_value=[])
    store.find_entity_candidates = AsyncMock(return_value=[])
    store.get_identity_core_entities = AsyncMock(return_value=[])
    store.get_stats = AsyncMock(return_value={"entity_count": 100})
    store.get_episode_cue = AsyncMock(return_value=None)
    store.get_episode_entities = AsyncMock(return_value=[])
    store.update_episode = AsyncMock()
    store.update_episode_cue = AsyncMock()
    return store


def _activation_store():
    store = AsyncMock()
    store.batch_get = AsyncMock(return_value={})
    store.get_activation = AsyncMock(return_value=None)
    store.set_activation = AsyncMock()
    store.record_access = AsyncMock()
    store.get_top_activated = AsyncMock(return_value=[])
    return store


class _SearchIndex:
    def __init__(self):
        self.search = AsyncMock(return_value=[("e1", 0.9), ("e2", 0.85)])
        self.search_episodes = AsyncMock(return_value=[(PRESENT_EPISODE_ID, 0.8)])
        self.search_episode_cues = AsyncMock(return_value=[])
        self.search_episodes_fast = AsyncMock(return_value=[(PRESENT_EPISODE_ID, 0.8)])
        self.search_episode_cues_fast = AsyncMock(return_value=[])
        self.compute_similarity = AsyncMock(return_value={})
        self._embeddings_enabled = False


class _RecordingReranker:
    """Records every document batch it is handed. Deliberately NOT NoopReranker."""

    def __init__(self):
        self.calls: list[list[tuple[str, str]]] = []

    async def rerank(self, _query, docs, top_n=None):
        self.calls.append(list(docs))
        return [(doc_id, 1.0 - i * 0.01) for i, (doc_id, _text) in enumerate(docs)][
            : top_n or len(docs)
        ]


def _cfg(**overrides) -> ActivationConfig:
    base = dict(
        consolidation_profile="off",
        recall_profile="off",
        chunk_search_enabled=False,
        cue_recall_enabled=False,
        graph_query_expansion_enabled=True,
        graph_query_expansion_timeout_ms=20,
        reranker_enabled=True,
        reranker_rerank_episodes=True,
        retrieval_reranker_timeout_ms=0,
    )
    base.update(overrides)
    return ActivationConfig(**base)


def _install_slow_expansion(monkeypatch) -> None:
    async def _slow_expand(*_args, stats_out=None, **_kwargs):
        # A read ISSUED against the store and never returning. An expansion
        # that hangs without touching the store is a starved coroutine, not
        # an over-budget graph, and does not arm the gate.
        if stats_out is not None:
            stats_out["attempts"] = 1.0
        await asyncio.sleep(0.2)
        return "expanded query"

    monkeypatch.setattr(
        "engram.retrieval.graph_expansion.expand_query_from_graph",
        _slow_expand,
    )


async def _recall_with_real_probe_timeout(monkeypatch, cfg, reranker=None):
    """Run one real recall whose graph-expansion probe genuinely times out."""
    _install_slow_expansion(monkeypatch)
    graph = _graph_store()
    stage_timings: dict[str, float] = {}
    await retrieve(
        query="Does the native transport release the GIL?",
        group_id=GROUP,
        graph_store=graph,
        activation_store=_activation_store(),
        search_index=_SearchIndex(),
        cfg=cfg,
        reranker=reranker,
        stage_timings_ms=stage_timings,
        record_feedback=False,
    )
    # The timeout must be REAL, not asserted into existence.
    assert "graph_expand_timeout" in stage_timings, stage_timings
    assert stage_timings["graph_expand_timeout"] >= 20
    return graph, stage_timings


@pytest.mark.asyncio
async def test_refusal_is_distinguishable_from_not_found(monkeypatch):
    """TIMEOUT and NOT_FOUND must not both arrive as ``None``.

    Fails before the fix: the gate answered ``None`` for the episode that
    exists, which is byte-identical to the answer for the one that does not.
    """
    cfg = _cfg()
    graph, stage_timings = await _recall_with_real_probe_timeout(monkeypatch, cfg)

    # Same timings dict the pipeline produced, so the gate is armed by a real
    # probe timeout rather than a hand-written marker.
    gate = GatedGraphStore(graph, cfg, stage_timings)

    def _outcome(result=None, refusal=None):
        return ("refused", refusal.probe) if refusal is not None else ("value", result)

    # NOT_FOUND, read directly: the store really has no such episode.
    assert await graph.get_episode_by_id(ABSENT_EPISODE_ID, GROUP) is None

    # TIMEOUT, read through the gate on an episode that DOES exist.
    try:
        gated = _outcome(result=await gate.get_episode_by_id(PRESENT_EPISODE_ID, GROUP))
    except GraphGateTimeoutError as exc:
        gated = _outcome(refusal=exc)

    assert gated[0] == "refused", (
        "gate answered a give-up as a miss: a caller cannot tell "
        f"'no such episode' from 'I stopped trying' (got {gated!r})"
    )
    assert gated[1] == "graph_expand_timeout"

    # And the give-up is countable, not merely fast.
    assert stage_timings.get(GATE_REFUSAL_METRIC_KEY, 0.0) >= 1.0


@pytest.mark.asyncio
async def test_reranker_never_scores_documents_it_could_not_read(monkeypatch):
    """The rerank caller must branch on TIMEOUT, not rerank empty strings.

    Fails before the fix: every ``get_entity``/``get_episode_by_id`` came back
    ``None``, the doc builder turned that into ``(id, "")``, the cross-encoder
    scored empty text, and ``scored.sort()`` reordered real results by it.
    """
    reranker = _RecordingReranker()
    cfg = _cfg()
    graph, stage_timings = await _recall_with_real_probe_timeout(
        monkeypatch,
        cfg,
        reranker=reranker,
    )
    empty_docs = [doc_id for batch in reranker.calls for doc_id, text in batch if not text.strip()]
    assert empty_docs == [], (
        "reranker was handed unreadable documents as empty text "
        f"(the store holds content for them): {empty_docs}"
    )
    assert stage_timings.get(GATE_REFUSAL_METRIC_KEY, 0.0) >= 1.0
    assert "recall_reranker_skipped_probe_timeout" in stage_timings, (
        "rerank was skipped but nothing recorded it — an unmeasured skip is "
        "how this stage read as 'useless' three separate times"
    )


@pytest.mark.asyncio
async def test_no_gated_caller_leaks_the_refusal_out_of_recall(monkeypatch):
    """Every gated stage armed at once: a refusal must never escape ``retrieve``.

    The audit half of ticket 19. Raising instead of returning ``None`` only
    helps if each call site decides what to do; a site that decides nothing
    turns a silent wrong answer into a 500. This arms every stage that reads
    the graph after the gate and asserts recall still returns.
    """
    reranker = _RecordingReranker()
    cfg = _cfg(
        goal_priming_enabled=True,
        cross_domain_penalty_enabled=True,
        inhibitory_spreading_enabled=True,
        emotional_salience_enabled=True,
        state_dependent_retrieval_enabled=True,
        preference_directed_enabled=True,
        gc_mmr_enabled=True,
        mmr_enabled=True,
        temporal_retrieval_enabled=True,
        working_memory_enabled=True,
        entity_query_retrieval_enabled=True,
        episode_graph_signal_enabled=True,
        current_value_entity_boost=1.2,
        passage_first_durable_entity_slots=2,
    )
    _install_slow_expansion(monkeypatch)
    working_memory = WorkingMemoryBuffer()
    wm_now = time.time()
    for entity_id in ("e1", "e2"):
        working_memory.add(entity_id, "entity", 0.9, "prior query", wm_now)
    stage_timings: dict[str, float] = {}

    # "latest"/"now" trips the temporal + current-value branches on purpose.
    await retrieve(
        query="What is the latest state of the native transport now?",
        group_id=GROUP,
        graph_store=_graph_store(),
        activation_store=_activation_store(),
        search_index=_SearchIndex(),
        cfg=cfg,
        reranker=reranker,
        working_memory=working_memory,
        stage_timings_ms=stage_timings,
        record_feedback=False,
    )

    assert "graph_expand_timeout" in stage_timings
    assert stage_timings.get(GATE_REFUSAL_METRIC_KEY, 0.0) >= 1.0
    # Each refusal-aware stage records its own give-up rather than reporting an
    # empty measurement of the corpus.
    assert "recall_reranker_skipped_probe_timeout" in stage_timings
    assert "recall_temporal_graph_reads_skipped_probe_timeout" in stage_timings
    assert "recall_entity_query_skipped_probe_timeout" in stage_timings

    # Step 1.8's entity-first fallback only runs when the entity pool comes back
    # empty, so it needs its own pass — it is the rescue lane for exactly the
    # case where a refusal looks most like "the brain knows nothing".
    empty_index = _SearchIndex()
    empty_index.search = AsyncMock(return_value=[])
    fallback_timings: dict[str, float] = {}
    await retrieve(
        query="What is the latest state of the native transport now?",
        group_id=GROUP,
        graph_store=_graph_store(),
        activation_store=_activation_store(),
        search_index=empty_index,
        cfg=cfg,
        reranker=_RecordingReranker(),
        working_memory=None,
        stage_timings_ms=fallback_timings,
        record_feedback=False,
    )
    assert "graph_expand_timeout" in fallback_timings
    assert "recall_entity_match_skipped_probe_timeout" in fallback_timings


@pytest.mark.asyncio
async def test_working_memory_pool_keeps_its_entities_when_neighbours_are_refused():
    """Regression guard on the audit, not a bug proof.

    ``_working_memory_pool`` wraps its whole body in a blanket ``except`` that
    returns ``[]``. Raising the refusal instead of returning ``[]`` would have
    routed it there and DELETED the working-memory entities — work the old
    falsy sentinel preserved. The handler must degrade to "no neighbour
    expansion", not "no working memory".
    """
    from engram.retrieval.candidate_pool import _working_memory_pool

    cfg = _cfg()
    gate = GatedGraphStore(_graph_store(), cfg, {"recall_stats_timeout": 1500.0})
    working_memory = WorkingMemoryBuffer()
    now = time.time()
    for entity_id in ("e1", "e2"):
        working_memory.add(entity_id, "entity", 0.9, "prior query", now)

    pooled = await _working_memory_pool(working_memory, GROUP, gate, now, 5, 20)
    assert {entity_id for entity_id, _score in pooled} == {"e1", "e2"}


@pytest.mark.asyncio
async def test_durable_feeder_still_collects_type_listings_when_identity_core_is_refused():
    """Regression guard: only the identity_core lane is gated.

    ``find_entities_by_type`` is not a gated method, so a refusal on the
    identity_core listing must not abandon the rest of the feeder.
    """
    from engram.retrieval.candidate_pool import (
        _durable_feeder_ids,
        clear_durable_feeder_cache,
    )

    clear_durable_feeder_cache()
    store = _graph_store()
    store.get_identity_core_entities = AsyncMock(return_value=[_entity("ic1")])
    store.find_entities_by_type = AsyncMock(return_value=[_entity("durable1")])
    cfg = _cfg()
    gate = GatedGraphStore(store, cfg, {"recall_stats_timeout": 1500.0})

    ids = await _durable_feeder_ids(GROUP, gate, time.time())
    assert "durable1" in ids
    assert "ic1" not in ids


@pytest.mark.asyncio
async def test_gate_open_still_reranks_real_documents(monkeypatch):
    """Positive control: with no probe timeout the same setup reranks real text.

    Without this, the test above would also pass on a reranker that never runs.
    """
    reranker = _RecordingReranker()
    cfg = _cfg(graph_query_expansion_enabled=False)
    stage_timings: dict[str, float] = {}
    await retrieve(
        query="Does the native transport release the GIL?",
        group_id=GROUP,
        graph_store=_graph_store(),
        activation_store=_activation_store(),
        search_index=_SearchIndex(),
        cfg=cfg,
        reranker=reranker,
        stage_timings_ms=stage_timings,
        record_feedback=False,
    )

    assert "graph_expand_timeout" not in stage_timings
    assert GATE_REFUSAL_METRIC_KEY not in stage_timings
    assert "recall_reranker_skipped_probe_timeout" not in stage_timings
    assert reranker.calls, "reranker never ran, so the sibling test is vacuous"
    texts = [text for batch in reranker.calls for _doc_id, text in batch]
    assert any(EPISODE_CONTENT in text for text in texts), (
        f"reranker never saw the episode body it was supposed to score: {texts}"
    )
