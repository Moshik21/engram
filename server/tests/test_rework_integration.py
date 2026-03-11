"""End-to-end integration tests for the extraction/recall/consolidation rework."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.replay import EpisodeReplayPhase
from engram.consolidation.phases.semantic_transition import SemanticTransitionPhase
from engram.events.bus import EventBus
from engram.extraction.extractor import ExtractionResult
from engram.graph_manager import GraphManager
from engram.models.consolidation import CycleContext
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.utils.dates import utc_now
from engram.worker import EpisodeWorker
from tests.conftest import MockExtractor


async def _build_lite_manager(
    tmp_path: Path,
    cfg: ActivationConfig,
    extraction_result: ExtractionResult,
    *,
    event_bus: EventBus | None = None,
) -> tuple[GraphManager, SQLiteGraphStore, MemoryActivationStore, FTS5SearchIndex]:
    db_path = tmp_path / "rework_integration.db"
    graph_store = SQLiteGraphStore(str(db_path))
    await graph_store.initialize()
    activation_store = MemoryActivationStore(cfg=cfg)
    search_index = FTS5SearchIndex(str(db_path))
    await search_index.initialize(db=graph_store._db)
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        MockExtractor(extraction_result),
        cfg=cfg,
        event_bus=event_bus,
    )
    return manager, graph_store, activation_store, search_index


async def _drain_events(queue: asyncio.Queue) -> list[dict]:
    events: list[dict] = []
    while not queue.empty():
        events.append(await queue.get())
    return events


def _cue_lookup(result: dict) -> dict:
    cue = result["cue"]
    return {
        "lookup_id": f"cue:{cue['episode_id']}",
        "result_type": "cue_episode",
        "episode_id": cue["episode_id"],
        "cue_text": cue["cue_text"],
        "supporting_spans": cue.get("supporting_spans", []),
        "score": result["score"],
        "count_hit": False,
    }


async def _wait_for(
    predicate,
    *,
    timeout: float = 3.0,
    interval: float = 0.05,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if await predicate():
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for integration state to settle")
        await asyncio.sleep(interval)


@pytest.mark.asyncio
async def test_observe_creates_cue_and_surfaced_recall_stays_latent(tmp_path: Path):
    """Observe creates a cue immediately, and surfaced recall does not escalate it."""
    cfg = ActivationConfig(
        cue_layer_enabled=True,
        cue_recall_enabled=True,
        cue_recall_hit_threshold=5,
        cue_policy_learning_enabled=False,
        episode_retrieval_enabled=False,
        recall_usage_feedback_enabled=True,
        recall_telemetry_enabled=True,
        working_memory_enabled=False,
    )
    bus = EventBus()
    manager, graph_store, _activation_store, _search_index = await _build_lite_manager(
        tmp_path,
        cfg,
        ExtractionResult(entities=[], relationships=[]),
        event_bus=bus,
    )

    try:
        episode_id = await manager.store_episode(
            "React dashboard migration remains in scope for the Phoenix redesign.",
            group_id="default",
            source="auto:prompt",
        )
        episode_before = await graph_store.get_episode_by_id(episode_id, "default")
        cue_before = await graph_store.get_episode_cue(episode_id, "default")
        assert episode_before is not None
        assert cue_before is not None

        queue = bus.subscribe("default")
        await _drain_events(queue)

        results = await manager.recall(
            "React migration Phoenix redesign",
            group_id="default",
            record_access=False,
            interaction_type="surfaced",
            interaction_source="auto_recall",
        )

        assert any(
            result.get("result_type") == "cue_episode" and result["cue"]["episode_id"] == episode_id
            for result in results
        )

        episode_after = await graph_store.get_episode_by_id(episode_id, "default")
        cue_after = await graph_store.get_episode_cue(episode_id, "default")
        assert episode_after is not None
        assert cue_after is not None
        assert episode_after.projection_state == episode_before.projection_state
        assert cue_after.projection_state == cue_before.projection_state
        assert cue_after.used_count == cue_before.used_count

        events = await _drain_events(queue)
        assert not [event for event in events if event["type"] == "episode.projection_scheduled"]
        assert not [event for event in events if event["type"] == "activation.access"]
    finally:
        await graph_store.close()


@pytest.mark.asyncio
async def test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall(
    tmp_path: Path,
):
    """Cue usage promotes an episode, the worker projects it, and entity recall works."""
    cfg = ActivationConfig(
        cue_layer_enabled=True,
        cue_recall_enabled=True,
        cue_policy_learning_enabled=True,
        cue_policy_schedule_threshold=0.6,
        cue_policy_select_weight=0.05,
        cue_policy_use_weight=0.35,
        cue_recall_hit_threshold=20,
        episode_retrieval_enabled=False,
        recall_usage_feedback_enabled=True,
        recall_telemetry_enabled=True,
        triage_enabled=False,
        worker_enabled=True,
        working_memory_enabled=False,
    )
    extraction_result = ExtractionResult(
        entities=[
            {
                "name": "React",
                "entity_type": "Technology",
                "summary": "UI library used for the dashboard migration",
            }
        ],
        relationships=[],
    )
    bus = EventBus()
    manager, graph_store, _activation_store, _search_index = await _build_lite_manager(
        tmp_path,
        cfg,
        extraction_result,
        event_bus=bus,
    )
    worker = EpisodeWorker(manager, cfg)

    try:
        episode_id = await manager.store_episode(
            "React dashboard migration remains in scope for this sprint.",
            group_id="default",
            source="auto:prompt",
        )
        await graph_store.update_episode(
            episode_id,
            {
                "projection_state": EpisodeProjectionState.CUE_ONLY,
                "last_projection_reason": "integration_reset",
            },
            group_id="default",
        )
        await graph_store.update_episode_cue(
            episode_id,
            {
                "projection_state": EpisodeProjectionState.CUE_ONLY,
                "route_reason": "integration_reset",
                "policy_score": 0.45,
                "projection_priority": 0.45,
            },
            group_id="default",
        )

        recall_results = await manager.recall(
            "React dashboard migration",
            group_id="default",
            record_access=False,
            interaction_type="selected",
            interaction_source="chat_tool_select",
        )
        cue_result = next(
            result for result in recall_results if result.get("result_type") == "cue_episode"
        )

        worker.start("default", bus)
        try:
            await manager.apply_memory_interaction(
                [f"cue:{episode_id}"],
                group_id="default",
                interaction_type="used",
                source="chat_response",
                query="What remains in scope?",
                result_lookup={f"cue:{episode_id}": _cue_lookup(cue_result)},
            )
            await _wait_for(
                lambda: _episode_projected(graph_store, episode_id),
            )
        finally:
            await worker.stop()

        stored_episode = await graph_store.get_episode_by_id(episode_id, "default")
        cue_after = await graph_store.get_episode_cue(episode_id, "default")
        assert stored_episode is not None
        assert cue_after is not None
        assert stored_episode.status == EpisodeStatus.COMPLETED
        assert stored_episode.projection_state == EpisodeProjectionState.PROJECTED
        assert cue_after.projection_state == EpisodeProjectionState.PROJECTED
        assert cue_after.used_count == 1

        linked_entity_ids = await graph_store.get_episode_entities(episode_id)
        assert linked_entity_ids

        entity_results = await manager.recall("React", group_id="default", limit=5)
        assert any(
            result.get("result_type") == "entity" and result["entity"]["name"] == "React"
            for result in entity_results
        )
    finally:
        await graph_store.close()


async def _episode_projected(
    graph_store: SQLiteGraphStore,
    episode_id: str,
) -> bool:
    episode = await graph_store.get_episode_by_id(episode_id, "default")
    if episode is None:
        return False
    return (
        episode.status == EpisodeStatus.COMPLETED
        and episode.projection_state == EpisodeProjectionState.PROJECTED
    )


@pytest.mark.asyncio
async def test_semantic_transition_sees_projected_episode_state(tmp_path: Path):
    """A projected episode can be semanticized once its linked entities are mature."""
    cfg = ActivationConfig(
        cue_layer_enabled=True,
        projector_v2_enabled=True,
        projection_planner_enabled=True,
        targeted_projection_enabled=True,
        episode_transition_enabled=True,
        episode_transitional_coverage=0.5,
        episode_transitional_min_cycles=1,
        working_memory_enabled=False,
    )
    extraction_result = ExtractionResult(
        entities=[
            {"name": "Alice", "entity_type": "Person", "summary": "Developer"},
            {"name": "Acme", "entity_type": "Organization", "summary": "Company"},
        ],
        relationships=[
            {"source": "Alice", "target": "Acme", "predicate": "WORKS_AT"},
        ],
    )
    manager, graph_store, activation_store, search_index = await _build_lite_manager(
        tmp_path,
        cfg,
        extraction_result,
    )

    try:
        episode_id = await manager.store_episode(
            "Alice works at Acme on the dashboard migration.",
            group_id="default",
            source="remember",
        )
        await manager.project_episode(episode_id, "default")

        linked_entity_ids = await graph_store.get_episode_entities(episode_id)
        assert linked_entity_ids

        phase = SemanticTransitionPhase()
        context = CycleContext()
        context.matured_entity_ids.update(linked_entity_ids)
        result, records = await phase.execute(
            "default",
            graph_store,
            activation_store,
            search_index,
            cfg,
            "cyc_semanticize",
            dry_run=False,
            context=context,
        )

        updated_episode = await graph_store.get_episode_by_id(episode_id, "default")
        assert updated_episode is not None
        assert result.items_processed >= 1
        assert any(record.episode_id == episode_id for record in records)
        assert updated_episode.memory_tier in {"transitional", "semantic"}
        assert updated_episode.consolidation_cycles >= 1
        assert updated_episode.entity_coverage >= 0.5
    finally:
        await graph_store.close()


@pytest.mark.asyncio
async def test_replay_uses_same_relationship_semantics_as_ingestion(tmp_path: Path):
    """Replay applies relationships through the same canonical ingestion path."""
    cfg = ActivationConfig(
        consolidation_profile="off",
        consolidation_replay_enabled=True,
        consolidation_replay_window_hours=24.0,
        consolidation_replay_min_age_hours=1.0,
        working_memory_enabled=False,
    )
    ingestion_result = ExtractionResult(
        entities=[
            {"name": "Alice", "entity_type": "Person", "summary": "Engineer"},
            {"name": "Acme", "entity_type": "Organization", "summary": "Company"},
        ],
        relationships=[
            {"source": "Alice", "target": "Acme", "predicate": "EMPLOYED_BY"},
        ],
    )
    manager, graph_store, activation_store, search_index = await _build_lite_manager(
        tmp_path,
        cfg,
        ingestion_result,
    )

    try:
        await manager.ingest_episode(
            "Alice is employed by Acme.",
            group_id="default",
            source="remember",
        )
        alice = (await graph_store.find_entities(name="Alice", group_id="default", limit=1))[0]
        acme = (await graph_store.find_entities(name="Acme", group_id="default", limit=1))[0]
        ingest_rel = await graph_store.find_existing_relationship(
            alice.id,
            acme.id,
            "WORKS_AT",
            group_id="default",
        )
        assert ingest_rel is not None

        replay_episode = Episode(
            id="ep_replay_semantics",
            content="Bob is employed at Beta.",
            source="test",
            status=EpisodeStatus.COMPLETED,
            group_id="default",
            created_at=utc_now() - timedelta(hours=3),
        )
        await graph_store.create_episode(replay_episode)

        replay_phase = EpisodeReplayPhase(
            extractor=MockExtractor(
                ExtractionResult(
                    entities=[
                        {"name": "Bob", "entity_type": "Person", "summary": "Engineer"},
                        {"name": "Beta", "entity_type": "Organization", "summary": "Company"},
                    ],
                    relationships=[
                        {"source": "Bob", "target": "Beta", "predicate": "EMPLOYED_AT"},
                    ],
                )
            )
        )
        context = CycleContext()
        context.trigger = "manual"
        result, records = await replay_phase.execute(
            "default",
            graph_store,
            activation_store,
            search_index,
            cfg,
            "cyc_replay_semantics",
            dry_run=False,
            context=context,
        )

        bob = (await graph_store.find_entities(name="Bob", group_id="default", limit=1))[0]
        beta = (await graph_store.find_entities(name="Beta", group_id="default", limit=1))[0]
        replay_rel = await graph_store.find_existing_relationship(
            bob.id,
            beta.id,
            "WORKS_AT",
            group_id="default",
        )

        assert result.items_processed >= 1
        assert any(record.episode_id == replay_episode.id for record in records)
        assert replay_rel is not None
    finally:
        await graph_store.close()
