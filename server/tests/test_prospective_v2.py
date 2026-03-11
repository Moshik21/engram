"""Tests for prospective memory v2: graph-embedded intentions and activation-based triggering."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.prospective import IntentionMeta
from engram.retrieval.prospective import check_intention_activations

# ─── IntentionMeta model tests ──────────────────────────────────────


class TestIntentionMetaModel:
    def test_defaults(self):
        meta = IntentionMeta(
            trigger_text="auth module",
            action_text="Check XSS fix",
        )
        assert meta.trigger_type == "activation"
        assert meta.activation_threshold == 0.5
        assert meta.max_fires == 5
        assert meta.fire_count == 0
        assert meta.enabled is True
        assert meta.cooldown_seconds == 300.0
        assert meta.priority == "normal"
        assert meta.origin == "explicit"
        assert meta.trigger_entity_ids == []
        assert meta.expires_at is None
        assert meta.last_fired is None

    def test_serialization_roundtrip(self):
        meta = IntentionMeta(
            trigger_text="Python upgrades",
            action_text="Migration plan in progress",
            trigger_type="entity_mention",
            activation_threshold=0.7,
            max_fires=3,
            fire_count=1,
            trigger_entity_ids=["ent_abc", "ent_def"],
            cooldown_seconds=600.0,
            priority="high",
            expires_at="2026-12-31T00:00:00",
            last_fired="2026-03-01T12:00:00",
        )
        d = meta.model_dump()
        restored = IntentionMeta(**d)
        assert restored.trigger_text == meta.trigger_text
        assert restored.trigger_entity_ids == ["ent_abc", "ent_def"]
        assert restored.priority == "high"
        assert restored.fire_count == 1
        assert restored.cooldown_seconds == 600.0

    def test_stored_in_entity_attributes(self):
        """IntentionMeta should be storable as Entity.attributes dict."""
        meta = IntentionMeta(
            trigger_text="deploy",
            action_text="Run smoke tests",
        )
        entity = Entity(
            id="int_test123",
            name="deploy",
            entity_type="Intention",
            summary="Run smoke tests",
            group_id="default",
            attributes=meta.model_dump(),
        )
        # Reconstruct from entity
        restored = IntentionMeta(**entity.attributes)
        assert restored.trigger_text == "deploy"
        assert restored.action_text == "Run smoke tests"


# ─── check_intention_activations tests ──────────────────────────────


def _make_intention_entity(
    id: str = "int_test1",
    trigger_text: str = "auth module",
    action_text: str = "Check XSS fix",
    trigger_type: str = "activation",
    threshold: float = 0.5,
    max_fires: int = 5,
    fire_count: int = 0,
    enabled: bool = True,
    cooldown_seconds: float = 300.0,
    last_fired: str | None = None,
    expires_at: str | None = None,
    trigger_entity_ids: list[str] | None = None,
    priority: str = "normal",
) -> Entity:
    meta = IntentionMeta(
        trigger_text=trigger_text,
        action_text=action_text,
        trigger_type=trigger_type,
        activation_threshold=threshold,
        max_fires=max_fires,
        fire_count=fire_count,
        enabled=enabled,
        cooldown_seconds=cooldown_seconds,
        last_fired=last_fired,
        expires_at=expires_at,
        trigger_entity_ids=trigger_entity_ids or [],
        priority=priority,
    )
    return Entity(
        id=id,
        name=trigger_text,
        entity_type="Intention",
        summary=action_text,
        group_id="default",
        attributes=meta.model_dump(),
    )


def _make_activation_state(access_count: int = 5) -> ActivationState:
    now = time.time()
    return ActivationState(
        entity_id="test",
        access_history=[now - i * 3600 for i in range(access_count)],
        access_count=access_count,
        last_accessed=now,
    )


class TestActivationTrigger:
    @pytest.mark.asyncio
    async def test_spreading_pushes_above_threshold(self):
        """Spreading activation reaching an intention should trigger it."""
        entity = _make_intention_entity(threshold=0.3)
        spreading = {entity.id: 0.4}  # above threshold

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 1
        assert matches[0].intention_id == entity.id
        assert matches[0].matched_via == "activation"

    @pytest.mark.asyncio
    async def test_trigger_entity_spreading_fires(self):
        """Spreading through a TRIGGERED_BY entity should fire the intention."""
        trigger_eid = "ent_auth_module"
        entity = _make_intention_entity(
            threshold=0.3,
            trigger_entity_ids=[trigger_eid],
        )
        # Intention itself has no spreading, but trigger entity does
        spreading = {trigger_eid: 0.5}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 1
        assert matches[0].intention_id == entity.id


class TestBelowThreshold:
    @pytest.mark.asyncio
    async def test_insufficient_spreading(self):
        """Spreading below threshold should not fire."""
        entity = _make_intention_entity(threshold=0.8)
        spreading = {entity.id: 0.1}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_no_spreading_no_fire(self):
        """No spreading energy at all should not fire."""
        entity = _make_intention_entity(threshold=0.5)

        matches = await check_intention_activations(
            spreading_results={},
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 0


class TestEntityMentionFallback:
    @pytest.mark.asyncio
    async def test_entity_mention_fires(self):
        """entity_mention trigger fires on entity ID match."""
        trigger_eid = "ent_auth"
        entity = _make_intention_entity(
            trigger_type="entity_mention",
            trigger_entity_ids=[trigger_eid],
        )

        matches = await check_intention_activations(
            spreading_results={},
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids={trigger_eid},
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 1
        assert matches[0].matched_via == "entity_mention"
        assert matches[0].similarity == 1.0

    @pytest.mark.asyncio
    async def test_entity_mention_no_match(self):
        """entity_mention trigger should not fire for unrelated entities."""
        entity = _make_intention_entity(
            trigger_type="entity_mention",
            trigger_entity_ids=["ent_auth"],
        )

        matches = await check_intention_activations(
            spreading_results={},
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids={"ent_unrelated"},
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 0


class TestCooldown:
    @pytest.mark.asyncio
    async def test_recently_fired_skipped(self):
        """Intention that fired recently should be skipped."""
        last_fired = datetime.utcnow().isoformat()
        entity = _make_intention_entity(
            threshold=0.3,
            cooldown_seconds=300.0,
            last_fired=last_fired,
        )
        spreading = {entity.id: 0.5}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_cooldown_expired_fires(self):
        """Intention past cooldown should fire."""
        old_fired = (datetime.utcnow() - timedelta(seconds=600)).isoformat()
        entity = _make_intention_entity(
            threshold=0.3,
            cooldown_seconds=300.0,
            last_fired=old_fired,
        )
        spreading = {entity.id: 0.5}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 1


class TestExhausted:
    @pytest.mark.asyncio
    async def test_fire_count_exceeds_max(self):
        """Exhausted intention should not fire."""
        entity = _make_intention_entity(
            threshold=0.3,
            max_fires=5,
            fire_count=5,
        )
        spreading = {entity.id: 0.5}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 0


class TestDisabled:
    @pytest.mark.asyncio
    async def test_disabled_intention_skipped(self):
        entity = _make_intention_entity(
            threshold=0.3,
            enabled=False,
        )
        spreading = {entity.id: 0.5}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 0


class TestExpired:
    @pytest.mark.asyncio
    async def test_expired_intention_skipped(self):
        expired = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        entity = _make_intention_entity(
            threshold=0.3,
            expires_at=expired,
        )
        spreading = {entity.id: 0.5}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 0


class TestMaxPerEpisode:
    @pytest.mark.asyncio
    async def test_respects_max_per_episode(self):
        """Should return at most max_per_episode matches."""
        entities = [
            _make_intention_entity(
                id=f"int_{i}",
                trigger_text=f"trigger_{i}",
                threshold=0.1,
            )
            for i in range(10)
        ]
        spreading = {e.id: 0.5 for e in entities}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={e.id: None for e in entities},
            intention_entities=entities,
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
            max_per_episode=3,
        )
        assert len(matches) == 3


class TestPrioritySorting:
    @pytest.mark.asyncio
    async def test_critical_before_normal(self):
        """Critical priority intentions should sort before normal."""
        normal = _make_intention_entity(
            id="int_normal",
            trigger_text="normal trigger",
            threshold=0.3,
            priority="normal",
        )
        critical = _make_intention_entity(
            id="int_critical",
            trigger_text="critical trigger",
            threshold=0.3,
            priority="critical",
        )
        spreading = {"int_normal": 0.5, "int_critical": 0.5}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={"int_normal": None, "int_critical": None},
            intention_entities=[normal, critical],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 2
        assert matches[0].intention_id == "int_critical"


# ─── Config tests ──────────────────────────────────────────────────


class TestProspectiveConfig:
    def test_default_values(self):
        cfg = ActivationConfig()
        assert cfg.prospective_activation_threshold == 0.5
        assert cfg.prospective_graph_embedded is True
        assert cfg.prospective_cooldown_seconds == 300.0
        assert cfg.prospective_warmth_levels == [0.3, 0.6, 0.8]

    def test_triggered_by_predicate(self):
        cfg = ActivationConfig()
        assert "TRIGGERED_BY" in cfg.predicate_natural_names
        assert cfg.predicate_natural_names["TRIGGERED_BY"] == "triggered by"
        assert "TRIGGERED_BY" in cfg.predicate_weights
        assert cfg.predicate_weights["TRIGGERED_BY"] == 0.9


# ─── Warmth classification tests ──────────────────────────────────


class TestWarmthGradient:
    def test_warmth_levels(self):
        """Test warmth classification at different activation levels."""
        cfg = ActivationConfig()
        levels = cfg.prospective_warmth_levels  # [0.3, 0.6, 0.8]

        # warmth_ratio = activation / threshold
        # dormant: < 0.3, cool: 0.3-0.6, warming: 0.6-0.8, warm: 0.8-1.0, hot: >= 1.0
        test_cases = [
            (0.1, "dormant"),
            (0.3, "cool"),
            (0.5, "cool"),
            (0.6, "warming"),
            (0.7, "warming"),
            (0.8, "warm"),
            (0.9, "warm"),
            (1.0, "hot"),
            (1.5, "hot"),
        ]

        for ratio, expected_label in test_cases:
            if ratio >= 1.0:
                label = "hot"
            elif ratio >= levels[2]:
                label = "warm"
            elif ratio >= levels[1]:
                label = "warming"
            elif ratio >= levels[0]:
                label = "cool"
            else:
                label = "dormant"
            assert label == expected_label, f"ratio={ratio}, expected={expected_label}, got={label}"


# ─── GraphManager integration tests (mocked) ──────────────────────


class TestGraphManagerCreateIntention:
    @pytest.mark.asyncio
    async def test_create_graph_embedded_intention(self):
        """create_intention should create Entity + TRIGGERED_BY edges."""
        from engram.graph_manager import GraphManager

        # Mock stores
        graph = AsyncMock()
        graph.find_entity_candidates = AsyncMock(
            return_value=[
                Entity(
                    id="ent_auth", name="Auth Module", entity_type="Technology", group_id="default"
                ),
            ]
        )
        graph.create_entity = AsyncMock()
        graph.create_relationship = AsyncMock()

        activation = AsyncMock()
        activation.record_access = AsyncMock()

        search = AsyncMock()
        search.index_entity = AsyncMock()

        extractor = MagicMock()

        cfg = ActivationConfig(
            prospective_graph_embedded=True,
            prospective_memory_enabled=True,
        )

        manager = GraphManager(graph, activation, search, extractor, cfg=cfg)

        intention_id = await manager.create_intention(
            trigger_text="auth module",
            action_text="Check XSS fix",
            trigger_type="activation",
            entity_names=["Auth Module"],
            group_id="default",
        )

        assert intention_id.startswith("int_")
        graph.create_entity.assert_called_once()
        created_entity = graph.create_entity.call_args[0][0]
        assert created_entity.entity_type == "Intention"
        assert created_entity.name == "auth module"

        # Should create TRIGGERED_BY edge
        graph.create_relationship.assert_called_once()
        rel = graph.create_relationship.call_args[0][0]
        assert rel.predicate == "TRIGGERED_BY"
        assert rel.target_id == "ent_auth"

        # Should index for search
        search.index_entity.assert_called_once()

        # Should record access
        activation.record_access.assert_called_once()

    @pytest.mark.asyncio
    async def test_v1_fallback(self):
        """When prospective_graph_embedded=False, should use flat table."""
        from engram.graph_manager import GraphManager

        graph = AsyncMock()
        graph.create_intention = AsyncMock(return_value="int_v1_123")
        activation = AsyncMock()
        search = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(
            prospective_graph_embedded=False,
            prospective_memory_enabled=True,
        )

        manager = GraphManager(graph, activation, search, extractor, cfg=cfg)

        intention_id = await manager.create_intention(
            trigger_text="test",
            action_text="action",
            trigger_type="semantic",
            group_id="default",
        )

        assert intention_id == "int_v1_123"
        graph.create_intention.assert_called_once()


class TestGraphManagerDismissIntention:
    @pytest.mark.asyncio
    async def test_soft_dismiss(self):
        """Soft dismiss should set enabled=False in attributes."""
        from engram.graph_manager import GraphManager

        entity = Entity(
            id="int_test",
            name="test",
            entity_type="Intention",
            group_id="default",
            attributes={"enabled": True, "trigger_text": "test", "action_text": "act"},
        )

        graph = AsyncMock()
        graph.get_entity = AsyncMock(return_value=entity)
        graph.update_entity = AsyncMock()
        activation = AsyncMock()
        search = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(prospective_graph_embedded=True)
        manager = GraphManager(graph, activation, search, extractor, cfg=cfg)

        await manager.dismiss_intention("int_test", "default", hard=False)

        graph.update_entity.assert_called_once()
        call_args = graph.update_entity.call_args
        attrs = call_args[0][1]["attributes"]
        assert attrs["enabled"] is False

    @pytest.mark.asyncio
    async def test_hard_dismiss(self):
        """Hard dismiss should call delete_entity."""
        from engram.graph_manager import GraphManager

        graph = AsyncMock()
        graph.delete_entity = AsyncMock()
        activation = AsyncMock()
        search = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(prospective_graph_embedded=True)
        manager = GraphManager(graph, activation, search, extractor, cfg=cfg)

        await manager.dismiss_intention("int_test", "default", hard=True)

        graph.delete_entity.assert_called_once_with("int_test", "default")


class TestGraphManagerListIntentions:
    @pytest.mark.asyncio
    async def test_list_filters_disabled(self):
        """list_intentions should filter out disabled intentions."""
        from engram.graph_manager import GraphManager

        now = datetime.utcnow()
        enabled_meta = IntentionMeta(
            trigger_text="active",
            action_text="action",
            enabled=True,
            expires_at=(now + timedelta(days=30)).isoformat(),
        )
        disabled_meta = IntentionMeta(
            trigger_text="inactive",
            action_text="action",
            enabled=False,
        )

        entities = [
            Entity(
                id="int_1",
                name="active",
                entity_type="Intention",
                group_id="default",
                attributes=enabled_meta.model_dump(),
            ),
            Entity(
                id="int_2",
                name="inactive",
                entity_type="Intention",
                group_id="default",
                attributes=disabled_meta.model_dump(),
            ),
        ]

        graph = AsyncMock()
        graph.find_entities = AsyncMock(return_value=entities)
        activation = AsyncMock()
        search = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(prospective_graph_embedded=True)
        manager = GraphManager(graph, activation, search, extractor, cfg=cfg)

        result = await manager.list_intentions("default", enabled_only=True)
        assert len(result) == 1
        assert result[0].id == "int_1"

    @pytest.mark.asyncio
    async def test_list_filters_exhausted(self):
        """list_intentions should filter out exhausted intentions."""
        from engram.graph_manager import GraphManager

        exhausted_meta = IntentionMeta(
            trigger_text="done",
            action_text="action",
            max_fires=5,
            fire_count=5,
        )

        entities = [
            Entity(
                id="int_1",
                name="done",
                entity_type="Intention",
                group_id="default",
                attributes=exhausted_meta.model_dump(),
            ),
        ]

        graph = AsyncMock()
        graph.find_entities = AsyncMock(return_value=entities)
        activation = AsyncMock()
        search = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(prospective_graph_embedded=True)
        manager = GraphManager(graph, activation, search, extractor, cfg=cfg)

        result = await manager.list_intentions("default", enabled_only=True)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_list_filters_expired(self):
        """list_intentions should filter out expired intentions."""
        from engram.graph_manager import GraphManager

        expired_meta = IntentionMeta(
            trigger_text="old",
            action_text="action",
            expires_at=(datetime.utcnow() - timedelta(hours=1)).isoformat(),
        )

        entities = [
            Entity(
                id="int_1",
                name="old",
                entity_type="Intention",
                group_id="default",
                attributes=expired_meta.model_dump(),
            ),
        ]

        graph = AsyncMock()
        graph.find_entities = AsyncMock(return_value=entities)
        activation = AsyncMock()
        search = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(prospective_graph_embedded=True)
        manager = GraphManager(graph, activation, search, extractor, cfg=cfg)

        result = await manager.list_intentions("default", enabled_only=True)
        assert len(result) == 0


# ─── EventBus integration tests ──────────────────────────────────


class TestIntentionEvents:
    @pytest.mark.asyncio
    async def test_create_publishes_event(self):
        """create_intention should publish intention.created event."""
        from engram.events.bus import EventBus
        from engram.graph_manager import GraphManager

        bus = EventBus()
        queue = bus.subscribe("default")

        graph = AsyncMock()
        graph.find_entity_candidates = AsyncMock(return_value=[])
        graph.create_entity = AsyncMock()
        activation = AsyncMock()
        activation.record_access = AsyncMock()
        search = AsyncMock()
        search.index_entity = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(
            prospective_graph_embedded=True,
            prospective_memory_enabled=True,
        )
        manager = GraphManager(
            graph,
            activation,
            search,
            extractor,
            cfg=cfg,
            event_bus=bus,
        )

        await manager.create_intention(
            trigger_text="test",
            action_text="action",
            group_id="default",
        )

        # Drain queue to find intention.created event
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        intention_events = [e for e in events if e.get("type") == "intention.created"]
        assert len(intention_events) == 1
        assert intention_events[0]["payload"]["intentionId"].startswith("int_")

    @pytest.mark.asyncio
    async def test_dismiss_publishes_event(self):
        """dismiss_intention should publish intention.dismissed event."""
        from engram.events.bus import EventBus
        from engram.graph_manager import GraphManager

        bus = EventBus()
        queue = bus.subscribe("default")

        entity = Entity(
            id="int_test",
            name="test",
            entity_type="Intention",
            group_id="default",
            attributes={"enabled": True, "trigger_text": "t", "action_text": "a"},
        )

        graph = AsyncMock()
        graph.get_entity = AsyncMock(return_value=entity)
        graph.update_entity = AsyncMock()
        activation = AsyncMock()
        search = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(prospective_graph_embedded=True)
        manager = GraphManager(
            graph,
            activation,
            search,
            extractor,
            cfg=cfg,
            event_bus=bus,
        )

        await manager.dismiss_intention("int_test", "default")

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        dismissed_events = [e for e in events if e.get("type") == "intention.dismissed"]
        assert len(dismissed_events) == 1
        assert dismissed_events[0]["payload"]["intentionId"] == "int_test"
        assert dismissed_events[0]["payload"]["hard"] is False

    @pytest.mark.asyncio
    async def test_fire_publishes_triggered_event(self):
        """_update_intention_fire should publish intention.triggered event."""
        from engram.events.bus import EventBus
        from engram.graph_manager import GraphManager

        bus = EventBus()
        queue = bus.subscribe("default")

        entity = Entity(
            id="int_test",
            name="test",
            entity_type="Intention",
            group_id="default",
            attributes={
                "trigger_text": "auth",
                "action_text": "check xss",
                "fire_count": 0,
            },
        )

        graph = AsyncMock()
        graph.get_entity = AsyncMock(return_value=entity)
        graph.update_entity = AsyncMock()
        activation = AsyncMock()
        search = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(prospective_graph_embedded=True)
        manager = GraphManager(
            graph,
            activation,
            search,
            extractor,
            cfg=cfg,
            event_bus=bus,
        )

        await manager._update_intention_fire("int_test", "default", "ep_123")

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        triggered_events = [e for e in events if e.get("type") == "intention.triggered"]
        assert len(triggered_events) == 1
        assert triggered_events[0]["payload"]["intentionId"] == "int_test"
        assert triggered_events[0]["payload"]["episodeId"] == "ep_123"

        # Check fire_count was incremented
        update_call = graph.update_entity.call_args
        new_attrs = update_call[0][1]["attributes"]
        assert new_attrs["fire_count"] == 1
        assert new_attrs["last_fired"] is not None


# ─── Context and see_also payload tests ──────────────────────────────


class TestIntentionMetaContextFields:
    def test_context_and_see_also_defaults_to_none(self):
        """New fields should default to None for backward compat."""
        meta = IntentionMeta(
            trigger_text="test",
            action_text="action",
        )
        assert meta.context is None
        assert meta.see_also is None

    def test_context_and_see_also_roundtrip(self):
        """context and see_also should survive serialization roundtrip."""
        meta = IntentionMeta(
            trigger_text="voice mode",
            action_text="Remind about Voice Mode progress",
            context="Alex is building a Voice Mode feature for the dashboard.",
            see_also=["WebSocket latency", "audio pipeline"],
        )
        d = meta.model_dump()
        restored = IntentionMeta(**d)
        assert restored.context == "Alex is building a Voice Mode feature for the dashboard."
        assert restored.see_also == ["WebSocket latency", "audio pipeline"]

    def test_backward_compat_old_dict_no_context(self):
        """Old stored dicts without context/see_also should deserialize fine."""
        old_attrs = {
            "trigger_text": "auth",
            "action_text": "check xss",
            "trigger_type": "activation",
            "activation_threshold": 0.5,
            "max_fires": 5,
            "fire_count": 0,
            "enabled": True,
            "trigger_entity_ids": [],
            "cooldown_seconds": 300.0,
            "priority": "normal",
            "origin": "explicit",
        }
        meta = IntentionMeta(**old_attrs)
        assert meta.context is None
        assert meta.see_also is None
        assert meta.trigger_text == "auth"


class TestIntentionMatchContextPropagation:
    @pytest.mark.asyncio
    async def test_activation_path_carries_context(self):
        """IntentionMatch from activation path should carry context/see_also."""
        meta = IntentionMeta(
            trigger_text="voice mode",
            action_text="Remind about progress",
            activation_threshold=0.3,
            context="Voice Mode is 80% done.",
            see_also=["audio pipeline"],
        )
        entity = Entity(
            id="int_voice",
            name="voice mode",
            entity_type="Intention",
            summary="Remind about progress",
            group_id="default",
            attributes=meta.model_dump(),
        )
        spreading = {entity.id: 0.5}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 1
        assert matches[0].context == "Voice Mode is 80% done."
        assert matches[0].see_also == ["audio pipeline"]

    @pytest.mark.asyncio
    async def test_entity_mention_path_carries_context(self):
        """IntentionMatch from entity_mention path should carry context/see_also."""
        trigger_eid = "ent_voice"
        meta = IntentionMeta(
            trigger_text="voice mode",
            action_text="Check progress",
            trigger_type="entity_mention",
            trigger_entity_ids=[trigger_eid],
            context="Voice Mode uses WebRTC.",
            see_also=["latency testing"],
        )
        entity = Entity(
            id="int_voice2",
            name="voice mode",
            entity_type="Intention",
            summary="Check progress",
            group_id="default",
            attributes=meta.model_dump(),
        )

        matches = await check_intention_activations(
            spreading_results={},
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids={trigger_eid},
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 1
        assert matches[0].context == "Voice Mode uses WebRTC."
        assert matches[0].see_also == ["latency testing"]

    @pytest.mark.asyncio
    async def test_none_context_when_not_set(self):
        """IntentionMatch should have None context when not set on meta."""
        entity = _make_intention_entity(threshold=0.3)
        spreading = {entity.id: 0.5}

        matches = await check_intention_activations(
            spreading_results=spreading,
            activation_states={entity.id: None},
            intention_entities=[entity],
            extracted_entity_ids=set(),
            now=time.time(),
            cfg=ActivationConfig(),
        )
        assert len(matches) == 1
        assert matches[0].context is None
        assert matches[0].see_also is None


class TestCreateIntentionStoresContext:
    @pytest.mark.asyncio
    async def test_context_stored_in_entity_attributes(self):
        """create_intention should store context/see_also in entity attributes."""
        from engram.graph_manager import GraphManager

        graph = AsyncMock()
        graph.find_entity_candidates = AsyncMock(return_value=[])
        graph.create_entity = AsyncMock()
        activation = AsyncMock()
        activation.record_access = AsyncMock()
        search = AsyncMock()
        search.index_entity = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(
            prospective_graph_embedded=True,
            prospective_memory_enabled=True,
        )
        manager = GraphManager(graph, activation, search, extractor, cfg=cfg)

        await manager.create_intention(
            trigger_text="voice mode",
            action_text="Remind about progress",
            context="Voice Mode is 80% done, needs audio pipeline.",
            see_also=["WebRTC", "latency"],
            group_id="default",
        )

        graph.create_entity.assert_called_once()
        created_entity = graph.create_entity.call_args[0][0]
        attrs = created_entity.attributes
        assert attrs["context"] == "Voice Mode is 80% done, needs audio pipeline."
        assert attrs["see_also"] == ["WebRTC", "latency"]

    @pytest.mark.asyncio
    async def test_none_context_not_stored_as_key(self):
        """When context/see_also are None, they should still be in attrs (Pydantic default)."""
        from engram.graph_manager import GraphManager

        graph = AsyncMock()
        graph.find_entity_candidates = AsyncMock(return_value=[])
        graph.create_entity = AsyncMock()
        activation = AsyncMock()
        activation.record_access = AsyncMock()
        search = AsyncMock()
        search.index_entity = AsyncMock()
        extractor = MagicMock()

        cfg = ActivationConfig(
            prospective_graph_embedded=True,
            prospective_memory_enabled=True,
        )
        manager = GraphManager(graph, activation, search, extractor, cfg=cfg)

        await manager.create_intention(
            trigger_text="test",
            action_text="action",
            group_id="default",
        )

        created_entity = graph.create_entity.call_args[0][0]
        attrs = created_entity.attributes
        # Pydantic includes None defaults in model_dump()
        assert attrs["context"] is None
        assert attrs["see_also"] is None


class TestResponseBuilderContextFields:
    def test_omits_keys_when_none(self):
        """Response builder dict comprehension should omit context/see_also when None."""
        from engram.models.prospective import IntentionMatch

        m = IntentionMatch(
            intention_id="int_1",
            trigger_text="test",
            action_text="action",
            similarity=0.8,
            matched_via="activation",
        )
        item = {
            "trigger": m.trigger_text,
            "action": m.action_text,
            "similarity": round(m.similarity, 4),
            "matched_via": m.matched_via,
            **({"context": m.context} if m.context else {}),
            **({"see_also": m.see_also} if m.see_also else {}),
        }
        assert "context" not in item
        assert "see_also" not in item

    def test_includes_keys_when_present(self):
        """Response builder should include context/see_also when set."""
        from engram.models.prospective import IntentionMatch

        m = IntentionMatch(
            intention_id="int_1",
            trigger_text="test",
            action_text="action",
            similarity=0.8,
            matched_via="activation",
            context="Rich context here",
            see_also=["topic1", "topic2"],
        )
        item = {
            "trigger": m.trigger_text,
            "action": m.action_text,
            "similarity": round(m.similarity, 4),
            "matched_via": m.matched_via,
            **({"context": m.context} if m.context else {}),
            **({"see_also": m.see_also} if m.see_also else {}),
        }
        assert item["context"] == "Rich context here"
        assert item["see_also"] == ["topic1", "topic2"]
