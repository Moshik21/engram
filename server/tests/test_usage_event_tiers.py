"""RF M1.5/M1.6 — usage-event tier semantics at the writer call sites.

M1.5: surfacing writers (recall materializer P1, get_context delivery P9)
record at the surfaced tier — hygiene weight 1.0 (access_history append),
ranking weight 0 (zero usage_events). Surfacing is ranker output, not an
environmental signal.

M1.6 (F10, w_mentioned=0.1): an organic entity commit in apply_entities is an
environmental mention — exactly one (ts, 0.1) usage event per (entity,
episode). Bootstrap-artifact commits (episode.source == "auto:bootstrap",
the marker stored by ingestion/project_bootstrap.py and the only path by
which bootstrap docs reach apply_entities, via projection of a CUE_ONLY
bootstrap episode) are docs, not user mentions: surfaced tier only.
Re-commits of the same episode (re-projection, replay re-linking) must not
double-fire the mentioned event.

All tests run against the real MemoryActivationStore so "zero usage_events"
is a store-state fact, not a mock artifact. Ranking reads nothing from
usage_events in M1 — these writers are inert by construction (G1/G5).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.extraction.apply import ApplyEngine
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.models import EntityCandidate
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.retrieval.feedback import RecallEntityAccessRecorder
from engram.retrieval.primary_results import RecallPrimaryResultMaterializer
from engram.retrieval.result_builder import RecallResultBuilder
from engram.retrieval.scorer import ScoredResult
from engram.retrieval.working_memory import RecallWorkingMemoryUpdater, WorkingMemoryBuffer
from engram.storage.memory.activation import MemoryActivationStore

GROUP = "usage_tier_group"


def _apply_engine(
    graph: AsyncMock,
    store: MemoryActivationStore,
) -> ApplyEngine:
    return ApplyEngine(
        graph_store=graph,
        activation_store=store,
        cfg=ActivationConfig(),
        canonicalizer=PredicateCanonicalizer(),
    )


def _commit_graph(candidates: list[Entity] | None = None) -> AsyncMock:
    graph = AsyncMock()
    graph.find_entity_candidates = AsyncMock(return_value=candidates or [])
    graph.create_entity = AsyncMock()
    graph.link_episode_entity = AsyncMock()
    graph.update_entity = AsyncMock()
    return graph


def _episode(source: str = "chat") -> Episode:
    return Episode(
        id="ep_rf",
        content="Alex moved to Phoenix.",
        source=source,
        group_id=GROUP,
    )


def _candidate(name: str = "Phoenix") -> EntityCandidate:
    return EntityCandidate(name=name, entity_type="Location", summary="City in Arizona")


async def _single_state(store: MemoryActivationStore, entity_id: str):
    state = await store.get_activation(entity_id)
    assert state is not None
    return state


# ---------------------------------------------------------------------------
# M1.5 — surfacing writers are surfaced-tier (zero usage_events)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_surfacing_appends_zero_usage_events() -> None:
    """P1: the recall materializer's access record is hygiene-only."""
    store = MemoryActivationStore()
    cfg = ActivationConfig()
    entity = Entity(id="ent_react", name="React", entity_type="Technology", group_id=GROUP)
    graph = AsyncMock()
    graph.get_entity = AsyncMock(return_value=entity)
    graph.get_relationships = AsyncMock(return_value=[])
    materializer = RecallPrimaryResultMaterializer(
        graph_store=graph,
        result_builder=RecallResultBuilder(cfg),
        cue_feedback_recorder=AsyncMock(),
        entity_access_recorder=RecallEntityAccessRecorder(
            cfg=cfg,
            activation_store=store,
            event_bus=None,
            labile_tracker=None,
        ),
        interaction_recorder=MagicMock(),
        working_memory_updater=RecallWorkingMemoryUpdater(),
    )

    await materializer.materialize(
        [
            ScoredResult(
                node_id=entity.id,
                score=0.8,
                semantic_similarity=0.8,
                activation=0.1,
                spreading=0.0,
                edge_proximity=0.2,
                result_type="entity",
            )
        ],
        group_id=GROUP,
        query="React",
        record_access=True,
        interaction_type=None,
        interaction_source="recall",
        now=123.0,
        working_memory=WorkingMemoryBuffer(),
    )

    state = await _single_state(store, entity.id)
    assert state.access_history == [123.0]  # hygiene append unchanged
    assert state.usage_events == []  # zero ranking-eligible events


@pytest.mark.asyncio
async def test_get_context_delivery_appends_zero_usage_events() -> None:
    """P9: the get_context delivered-entity loop records surfaced-tier only."""
    from engram.retrieval.context_builder import MemoryContextBuilder

    store = MemoryActivationStore()
    seed_ts = 1_000.0
    await store.record_access("ent_recent", seed_ts, group_id=GROUP)

    graph = MagicMock()
    graph.get_relationships = AsyncMock(return_value=[])
    graph.get_entity = AsyncMock(
        return_value=Entity(
            id="ent_recent",
            name="RecentEnt",
            entity_type="Concept",
            summary="recent",
            group_id=GROUP,
        )
    )
    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=store,
        cfg=ActivationConfig(identity_core_enabled=False, briefing_enabled=False),
        recall=AsyncMock(return_value=[]),
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )

    result = await builder.get_context(group_id=GROUP)

    assert "RecentEnt" in result["context"]  # delivered, so recorded
    state = await _single_state(store, "ent_recent")
    assert len(state.access_history) == 2  # seed + delivery (hygiene unchanged)
    assert state.usage_events == []  # delivery is never a ranking prior


# ---------------------------------------------------------------------------
# M1.6 — entity commits are mentioned-tier (F10: w_mentioned=0.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_organic_entity_commit_appends_exactly_one_mentioned_event() -> None:
    store = MemoryActivationStore()
    graph = _commit_graph()
    engine = _apply_engine(graph, store)

    outcome = await engine.apply_entities([_candidate()], _episode(), GROUP)

    entity_id = outcome.entity_map["Phoenix"]
    state = await _single_state(store, entity_id)
    assert len(state.access_history) == 1  # hygiene append unchanged
    assert len(state.usage_events) == 1
    ts, weight = state.usage_events[0]
    assert ts > 0.0
    assert weight == pytest.approx(0.1)  # F10 pins w_mentioned=0.1


@pytest.mark.asyncio
async def test_duplicate_candidates_fire_one_mentioned_event() -> None:
    """Two same-name candidates in one call: one mentioned event, two hygiene."""
    store = MemoryActivationStore()
    graph = _commit_graph()
    engine = _apply_engine(graph, store)

    outcome = await engine.apply_entities(
        [_candidate(), _candidate()],
        _episode(),
        GROUP,
    )

    entity_id = outcome.entity_map["Phoenix"]
    state = await _single_state(store, entity_id)
    assert len(state.access_history) == 2  # both commits still hygiene-recorded
    assert len(state.usage_events) == 1  # at most one mentioned per (entity, episode)


@pytest.mark.asyncio
async def test_existing_entity_first_mention_in_new_episode_fires_mentioned() -> None:
    """A known entity mentioned by a NEW episode is a fresh environmental mention."""
    store = MemoryActivationStore()
    prior = Entity(
        id="ent_prior",
        name="Phoenix",
        entity_type="Location",
        group_id=GROUP,
        source_episode_ids=["ep_other"],
    )
    graph = _commit_graph([prior])
    engine = _apply_engine(graph, store)

    await engine.apply_entities([_candidate()], _episode(), GROUP)

    state = await _single_state(store, "ent_prior")
    assert len(state.usage_events) == 1
    assert state.usage_events[0][1] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_recommit_of_same_episode_does_not_double_fire() -> None:
    """Replay re-linking / re-projection: episode already in source_episode_ids."""
    store = MemoryActivationStore()
    prior = Entity(
        id="ent_prior",
        name="Phoenix",
        entity_type="Location",
        group_id=GROUP,
        source_episode_ids=["ep_rf"],  # this episode already evidenced
    )
    graph = _commit_graph([prior])
    engine = _apply_engine(graph, store)

    await engine.apply_entities([_candidate()], _episode(), GROUP)

    state = await _single_state(store, "ent_prior")
    assert len(state.access_history) == 1  # hygiene append still happens
    assert state.usage_events == []  # no second mentioned event


@pytest.mark.asyncio
async def test_bootstrap_artifact_commit_appends_no_usage_events() -> None:
    """Bootstrap docs are not user mentions: surfaced tier only."""
    store = MemoryActivationStore()
    graph = _commit_graph()
    engine = _apply_engine(graph, store)

    outcome = await engine.apply_entities(
        [_candidate()],
        _episode(source="auto:bootstrap"),
        GROUP,
    )

    entity_id = outcome.entity_map["Phoenix"]
    state = await _single_state(store, entity_id)
    assert len(state.access_history) == 1  # hygiene append unchanged
    assert state.usage_events == []


def test_bootstrap_source_marker_matches_apply_discriminator() -> None:
    """apply_entities keys the bootstrap exclusion on the exact source marker
    that project_bootstrap stores; if either string drifts, bootstrap docs
    would silently start accruing mentioned-tier ranking events."""
    import inspect

    from engram.extraction import apply as apply_mod
    from engram.ingestion import project_bootstrap

    assert 'source="auto:bootstrap"' in inspect.getsource(project_bootstrap)
    assert 'episode.source == "auto:bootstrap"' in inspect.getsource(apply_mod)
