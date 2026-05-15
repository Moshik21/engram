"""Tests for recall telemetry and interaction semantics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.extraction.extractor import ExtractionResult
from engram.extraction.policy import ProjectionPolicy
from engram.graph_manager import GraphManager
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.retrieval.control import RecallNeedController
from engram.retrieval.feedback import (
    RecallCueFeedbackRecorder,
    RecallEntityAccessRecorder,
    RecallInteractionRecorder,
    RecallMemoryInteractionApplier,
    partition_recall_entities_by_usage,
    partition_recall_targets_by_usage,
)
from engram.retrieval.reconsolidation import LabileWindowTracker
from tests.conftest import MockExtractor


class TestRecallUsageDetection:
    def test_partitions_used_and_dismissed_entities(self):
        recall_results = [
            {
                "entity": {
                    "id": "ent_react",
                    "name": "React",
                    "type": "Technology",
                    "summary": "UI library",
                },
                "score": 0.91,
            },
            {
                "entity": {
                    "id": "ent_redis",
                    "name": "Redis",
                    "type": "Technology",
                    "summary": "Cache",
                },
                "score": 0.73,
            },
        ]

        used, dismissed = partition_recall_entities_by_usage(
            "We should keep React for this dashboard refactor.",
            recall_results,
        )

        assert [entity["entity_id"] for entity in used] == ["ent_react"]
        assert [entity["entity_id"] for entity in dismissed] == ["ent_redis"]

    def test_partitions_used_and_dismissed_cue_targets(self):
        recall_results = [
            {
                "result_type": "cue_episode",
                "cue": {
                    "episode_id": "ep_react",
                    "cue_text": "spans: React dashboard migration remains in scope",
                    "supporting_spans": ["React dashboard migration remains in scope"],
                },
                "episode": {"id": "ep_react"},
                "score": 0.84,
            },
            {
                "result_type": "cue_episode",
                "cue": {
                    "episode_id": "ep_redis",
                    "cue_text": "spans: Redis cache rollout was postponed",
                    "supporting_spans": ["Redis cache rollout was postponed"],
                },
                "episode": {"id": "ep_redis"},
                "score": 0.63,
            },
        ]

        used, dismissed = partition_recall_targets_by_usage(
            "Keep the React dashboard migration in scope for this sprint.",
            recall_results,
        )

        assert [target["episode_id"] for target in used] == ["ep_react"]
        assert [target["episode_id"] for target in dismissed] == ["ep_redis"]


class TestRecallInteractionRecorder:
    def test_entity_interaction_publishes_and_records_need(self):
        bus = EventBus()
        queue = bus.subscribe("default")
        need_controller = MagicMock()
        recorder = RecallInteractionRecorder(
            cfg=ActivationConfig(recall_telemetry_enabled=True),
            event_bus=bus,
            recall_need_controller=need_controller,
        )

        recorder.record_entity_interaction(
            group_id="default",
            entity=Entity(
                id="ent_react",
                name="React",
                entity_type="Technology",
                group_id="default",
            ),
            interaction_type="used",
            source="chat_tool_use",
            query="React",
            score=0.91,
            recorded_access=True,
        )

        event = queue.get_nowait()
        assert event["type"] == "recall.interaction"
        assert event["payload"]["entityId"] == "ent_react"
        assert event["payload"]["interactionType"] == "used"
        assert event["payload"]["recordedAccess"] is True
        need_controller.record_interaction.assert_called_once_with(
            "default",
            "used",
            result_type="entity",
        )

    def test_no_interaction_type_is_noop(self):
        bus = EventBus()
        queue = bus.subscribe("default")
        need_controller = MagicMock()
        recorder = RecallInteractionRecorder(
            cfg=ActivationConfig(recall_telemetry_enabled=True),
            event_bus=bus,
            recall_need_controller=need_controller,
        )

        recorder.record_entity_interaction(
            group_id="default",
            entity=Entity(id="ent_react", name="React", entity_type="Technology"),
            interaction_type=None,
            source="recall",
            query="React",
            score=0.91,
            recorded_access=False,
        )

        assert queue.empty()
        need_controller.record_interaction.assert_not_called()


@pytest.mark.asyncio
class TestRecallCueFeedbackRecorder:
    async def test_promotes_hot_cue_through_shared_projection_state(self, graph_store):
        bus = EventBus()
        queue = bus.subscribe("default")
        need_controller = MagicMock(spec=RecallNeedController)
        cfg = ActivationConfig(
            cue_recall_hit_threshold=2,
            cue_policy_learning_enabled=False,
        )
        recorder = RecallCueFeedbackRecorder(
            cfg=cfg,
            graph_store=graph_store,
            projection_policy=ProjectionPolicy(cfg),
            recall_need_controller=need_controller,
            event_bus=bus,
        )
        episode = Episode(
            id="ep_hot_cue",
            content="React dashboard migration remains in scope.",
            source="test",
            status=EpisodeStatus.COMPLETED,
            projection_state=EpisodeProjectionState.CUE_ONLY,
            group_id="default",
        )
        await graph_store.create_episode(episode)
        await graph_store.upsert_episode_cue(
            EpisodeCue(
                episode_id=episode.id,
                group_id="default",
                projection_state=EpisodeProjectionState.CUE_ONLY,
                route_reason="entity_dense",
                cue_text="spans: React dashboard migration remains in scope",
                first_spans=["React dashboard migration remains in scope."],
                hit_count=1,
                policy_score=0.48,
            ),
        )

        await recorder.record_cue_feedback(
            episode,
            score=0.91,
            query="What remains in scope?",
            interaction_type="surfaced",
        )

        stored_episode = await graph_store.get_episode_by_id(episode.id, "default")
        cue = await graph_store.get_episode_cue(episode.id, "default")
        events = []
        while not queue.empty():
            events.append(await queue.get())

        assert stored_episode is not None
        assert stored_episode.status == EpisodeStatus.QUEUED
        assert stored_episode.projection_state == EpisodeProjectionState.SCHEDULED
        assert cue is not None
        assert cue.hit_count == 2
        assert cue.surfaced_count == 1
        assert cue.projection_state == EpisodeProjectionState.SCHEDULED
        assert cue.route_reason == "cue_recall_hits"
        assert [event for event in events if event["type"] == "cue.hit"]
        assert [event for event in events if event["type"] == "cue.promoted"]
        assert [event for event in events if event["type"] == "episode.projection_scheduled"]
        need_controller.record_interaction.assert_called_once_with(
            "default",
            "surfaced",
            result_type="cue_episode",
        )


@pytest.mark.asyncio
class TestRecallEntityAccessRecorder:
    async def test_records_activation_event_and_labile_window(self, activation_store):
        bus = EventBus()
        queue = bus.subscribe("default")
        labile_tracker = LabileWindowTracker(ttl=300.0)
        recorder = RecallEntityAccessRecorder(
            cfg=ActivationConfig(),
            activation_store=activation_store,
            event_bus=bus,
            labile_tracker=labile_tracker,
        )
        entity = Entity(
            id="ent_react",
            name="React",
            entity_type="Technology",
            summary="UI library",
            group_id="default",
        )

        await recorder.record_entity_access(
            entity,
            group_id="default",
            query="React dashboard",
            source="chat_tool_use",
            timestamp=123.0,
        )

        state = await activation_store.get_activation(entity.id)
        event = await queue.get()
        labile = labile_tracker.get_labile(entity.id)

        assert state is not None
        assert state.access_count == 1
        assert event["type"] == "activation.access"
        assert event["payload"]["entityId"] == "ent_react"
        assert event["payload"]["entityType"] == "Technology"
        assert event["payload"]["accessedVia"] == "chat_tool_use"
        assert labile is not None
        assert labile.query == "React dashboard"


@pytest.mark.asyncio
class TestRecallMemoryInteractionApplier:
    async def test_confirmed_entity_records_access_feedback_event_and_need(
        self,
        graph_store,
        activation_store,
    ):
        bus = EventBus()
        queue = bus.subscribe("default")
        need_controller = MagicMock(spec=RecallNeedController)
        cfg = ActivationConfig(
            recall_telemetry_enabled=True,
            recall_usage_feedback_enabled=True,
            ts_enabled=True,
        )
        interaction_recorder = RecallInteractionRecorder(
            cfg=cfg,
            event_bus=bus,
            recall_need_controller=need_controller,
        )
        cue_feedback_recorder = RecallCueFeedbackRecorder(
            cfg=cfg,
            graph_store=graph_store,
            projection_policy=ProjectionPolicy(cfg),
            recall_need_controller=need_controller,
            event_bus=bus,
        )
        entity_access_recorder = RecallEntityAccessRecorder(
            cfg=cfg,
            activation_store=activation_store,
            event_bus=bus,
            labile_tracker=None,
        )
        applier = RecallMemoryInteractionApplier(
            cfg=cfg,
            graph_store=graph_store,
            activation_store=activation_store,
            cue_feedback_recorder=cue_feedback_recorder,
            entity_access_recorder=entity_access_recorder,
            interaction_recorder=interaction_recorder,
            recall_need_controller=need_controller,
        )
        entity = Entity(
            id="ent_react",
            name="React",
            entity_type="Technology",
            summary="UI library",
            group_id="default",
        )
        await graph_store.create_entity(entity)

        await applier.apply(
            [entity.id],
            group_id="default",
            interaction_type="confirmed",
            source="chat_feedback",
            query="React",
            result_lookup={entity.id: {"score": 0.8}},
        )

        state = await activation_store.get_activation(entity.id)
        events = []
        while not queue.empty():
            events.append(await queue.get())

        assert state is not None
        assert state.access_count == 1
        assert state.ts_alpha > 1.0
        assert [event for event in events if event["type"] == "activation.access"]
        interaction_events = [event for event in events if event["type"] == "recall.interaction"]
        assert len(interaction_events) == 1
        assert interaction_events[0]["payload"]["interactionType"] == "confirmed"
        assert interaction_events[0]["payload"]["recordedAccess"] is True
        need_controller.record_interaction.assert_called_once_with(
            "default",
            "confirmed",
            result_type="entity",
        )


@pytest.mark.asyncio
class TestRecallFeedback:
    async def test_cue_used_interaction_promotes_without_double_counting_hit(
        self,
        graph_store,
        activation_store,
        search_index,
    ):
        cfg = ActivationConfig(
            cue_policy_learning_enabled=True,
            cue_policy_use_weight=0.35,
            cue_policy_schedule_threshold=0.8,
            recall_usage_feedback_enabled=True,
        )
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            MockExtractor(ExtractionResult(entities=[], relationships=[])),
            cfg=cfg,
        )

        episode = Episode(
            id="ep_cue_used",
            content="React dashboard migration remains in scope.",
            source="test",
            status=EpisodeStatus.COMPLETED,
            projection_state=EpisodeProjectionState.CUE_ONLY,
            group_id="default",
        )
        await graph_store.create_episode(episode)
        await graph_store.upsert_episode_cue(
            EpisodeCue(
                episode_id=episode.id,
                group_id="default",
                projection_state=EpisodeProjectionState.CUE_ONLY,
                projection_priority=0.48,
                policy_score=0.48,
                cue_text="spans: React dashboard migration remains in scope",
                first_spans=["React dashboard migration remains in scope."],
                hit_count=1,
                selected_count=1,
                route_reason="entity_dense",
            ),
        )

        await manager.apply_memory_interaction(
            ["cue:ep_cue_used"],
            group_id="default",
            interaction_type="used",
            source="chat_response",
            query="What remains in scope?",
            result_lookup={
                "cue:ep_cue_used": {
                    "lookup_id": "cue:ep_cue_used",
                    "result_type": "cue_episode",
                    "episode_id": "ep_cue_used",
                    "cue_text": "spans: React dashboard migration remains in scope",
                    "supporting_spans": ["React dashboard migration remains in scope."],
                    "score": 0.95,
                    "count_hit": False,
                },
            },
        )

        cue = await graph_store.get_episode_cue("ep_cue_used", "default")
        stored_episode = await graph_store.get_episode_by_id("ep_cue_used", "default")

        assert cue is not None
        assert stored_episode is not None
        assert cue.used_count == 1
        assert cue.hit_count == 1
        assert cue.projection_state == EpisodeProjectionState.SCHEDULED
        assert stored_episode.projection_state == EpisodeProjectionState.SCHEDULED

    async def test_surfaced_recall_skips_access_but_emits_interaction(
        self,
        graph_store,
        activation_store,
        search_index,
    ):
        bus = EventBus()
        extractor = MockExtractor(
            ExtractionResult(
                entities=[{"name": "Python", "entity_type": "Technology", "summary": "Language"}],
                relationships=[],
            )
        )
        cfg = ActivationConfig(
            recall_telemetry_enabled=True,
            recall_usage_feedback_enabled=True,
        )
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            extractor,
            cfg=cfg,
            event_bus=bus,
        )
        await manager.ingest_episode(
            "Using Python in production",
            group_id="default",
            source="test",
        )
        entity = (await graph_store.find_entities(name="Python", group_id="default", limit=1))[0]
        before = await activation_store.get_activation(entity.id)
        before_count = before.access_count if before else 0
        before_alpha = before.ts_alpha if before else 1.0
        before_beta = before.ts_beta if before else 1.0
        queue = bus.subscribe("default")

        await manager.recall(
            query="Python",
            group_id="default",
            record_access=False,
            interaction_type="surfaced",
            interaction_source="auto_recall",
        )

        after = await activation_store.get_activation(entity.id)
        events = []
        while not queue.empty():
            events.append(await queue.get())

        assert before is not None
        assert after is not None
        assert after.access_count == before_count
        assert after.ts_alpha == before_alpha
        assert after.ts_beta == before_beta
        assert not [e for e in events if e["type"] == "activation.access"]

        interaction_events = [e for e in events if e["type"] == "recall.interaction"]
        assert len(interaction_events) == 1
        payload = interaction_events[0]["payload"]
        assert payload["interactionType"] == "surfaced"
        assert payload["source"] == "auto_recall"
        assert payload["recordedAccess"] is False

    async def test_explicit_recall_can_still_record_access_and_emit_interaction(
        self,
        graph_store,
        activation_store,
        search_index,
    ):
        bus = EventBus()
        extractor = MockExtractor(
            ExtractionResult(
                entities=[{"name": "React", "entity_type": "Technology", "summary": "UI library"}],
                relationships=[],
            )
        )
        cfg = ActivationConfig(recall_telemetry_enabled=True)
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            extractor,
            cfg=cfg,
            event_bus=bus,
        )
        await manager.ingest_episode("We use React", group_id="default", source="test")
        entity = (await graph_store.find_entities(name="React", group_id="default", limit=1))[0]
        before = await activation_store.get_activation(entity.id)
        before_count = before.access_count if before else 0
        before_alpha = before.ts_alpha if before else 1.0
        queue = bus.subscribe("default")

        await manager.recall(
            query="React",
            group_id="default",
            interaction_type="used",
            interaction_source="chat_tool_use",
        )

        after = await activation_store.get_activation(entity.id)
        events = []
        while not queue.empty():
            events.append(await queue.get())

        assert before is not None
        assert after is not None
        assert after.access_count > before_count
        assert after.ts_alpha > before_alpha
        assert [e for e in events if e["type"] == "activation.access"]

        interaction_events = [e for e in events if e["type"] == "recall.interaction"]
        assert len(interaction_events) == 1
        payload = interaction_events[0]["payload"]
        assert payload["interactionType"] == "used"
        assert payload["source"] == "chat_tool_use"
        assert payload["recordedAccess"] is True

    async def test_selected_recall_skips_access_and_ts_feedback(
        self,
        graph_store,
        activation_store,
        search_index,
    ):
        bus = EventBus()
        extractor = MockExtractor(
            ExtractionResult(
                entities=[{"name": "Redis", "entity_type": "Technology", "summary": "Cache"}],
                relationships=[],
            )
        )
        cfg = ActivationConfig(
            recall_telemetry_enabled=True,
            recall_usage_feedback_enabled=True,
            ts_enabled=True,
        )
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            extractor,
            cfg=cfg,
            event_bus=bus,
        )
        await manager.ingest_episode("We use Redis", group_id="default", source="test")
        entity = (await graph_store.find_entities(name="Redis", group_id="default", limit=1))[0]
        before = await activation_store.get_activation(entity.id)
        before_count = before.access_count if before else 0
        before_alpha = before.ts_alpha if before else 1.0
        before_beta = before.ts_beta if before else 1.0
        queue = bus.subscribe("default")

        await manager.recall(
            query="Redis",
            group_id="default",
            record_access=False,
            interaction_type="selected",
            interaction_source="chat_tool_select",
        )

        after = await activation_store.get_activation(entity.id)
        events = []
        while not queue.empty():
            events.append(await queue.get())

        assert after is not None
        assert after.access_count == before_count
        assert after.ts_alpha == before_alpha
        assert after.ts_beta == before_beta
        assert not [e for e in events if e["type"] == "activation.access"]

        interaction_events = [e for e in events if e["type"] == "recall.interaction"]
        assert len(interaction_events) == 1
        payload = interaction_events[0]["payload"]
        assert payload["interactionType"] == "selected"
        assert payload["source"] == "chat_tool_select"
        assert payload["recordedAccess"] is False

    async def test_corrected_interaction_applies_negative_feedback_without_access(
        self,
        graph_store,
        activation_store,
        search_index,
    ):
        bus = EventBus()
        extractor = MockExtractor(
            ExtractionResult(
                entities=[{"name": "React", "entity_type": "Technology", "summary": "UI library"}],
                relationships=[],
            )
        )
        cfg = ActivationConfig(
            recall_telemetry_enabled=True,
            recall_usage_feedback_enabled=True,
            ts_enabled=True,
        )
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            extractor,
            cfg=cfg,
            event_bus=bus,
        )
        await manager.ingest_episode("We use React", group_id="default", source="test")
        entity = (await graph_store.find_entities(name="React", group_id="default", limit=1))[0]
        before = await activation_store.get_activation(entity.id)
        before_count = before.access_count if before else 0
        before_beta = before.ts_beta if before else 1.0
        queue = bus.subscribe("default")

        await manager.apply_memory_interaction(
            [entity.id],
            group_id="default",
            interaction_type="corrected",
            source="test_correction",
            query="React",
            result_lookup={
                entity.id: {
                    "entity_name": "React",
                    "entity_type": "Technology",
                    "score": 0.8,
                }
            },
        )

        after = await activation_store.get_activation(entity.id)
        events = []
        while not queue.empty():
            events.append(await queue.get())

        assert after is not None
        assert after.access_count == before_count
        assert after.ts_beta > before_beta
        assert not [e for e in events if e["type"] == "activation.access"]

        interaction_events = [e for e in events if e["type"] == "recall.interaction"]
        assert len(interaction_events) == 1
        payload = interaction_events[0]["payload"]
        assert payload["interactionType"] == "corrected"
        assert payload["source"] == "test_correction"
        assert payload["recordedAccess"] is False

    async def test_dismissed_interaction_is_neutral_without_access(
        self,
        graph_store,
        activation_store,
        search_index,
    ):
        bus = EventBus()
        extractor = MockExtractor(
            ExtractionResult(
                entities=[{"name": "Next.js", "entity_type": "Technology", "summary": "Framework"}],
                relationships=[],
            )
        )
        cfg = ActivationConfig(
            recall_telemetry_enabled=True,
            recall_usage_feedback_enabled=True,
            ts_enabled=True,
        )
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            extractor,
            cfg=cfg,
            event_bus=bus,
        )
        await manager.ingest_episode("We use Next.js", group_id="default", source="test")
        entity = (await graph_store.find_entities(name="Next.js", group_id="default", limit=1))[0]
        before = await activation_store.get_activation(entity.id)
        before_count = before.access_count if before else 0
        before_alpha = before.ts_alpha if before else 1.0
        before_beta = before.ts_beta if before else 1.0
        queue = bus.subscribe("default")

        await manager.apply_memory_interaction(
            [entity.id],
            group_id="default",
            interaction_type="dismissed",
            source="chat_response",
            query="Next.js",
            result_lookup={
                entity.id: {
                    "entity_name": "Next.js",
                    "entity_type": "Technology",
                    "score": 0.7,
                }
            },
        )

        after = await activation_store.get_activation(entity.id)
        events = []
        while not queue.empty():
            events.append(await queue.get())

        assert after is not None
        assert after.access_count == before_count
        assert after.ts_alpha == before_alpha
        assert after.ts_beta == before_beta
        assert not [e for e in events if e["type"] == "activation.access"]

        interaction_events = [e for e in events if e["type"] == "recall.interaction"]
        assert len(interaction_events) == 1
        payload = interaction_events[0]["payload"]
        assert payload["interactionType"] == "dismissed"
        assert payload["source"] == "chat_response"
        assert payload["recordedAccess"] is False
