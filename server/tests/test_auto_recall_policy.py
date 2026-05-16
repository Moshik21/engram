from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from engram.config import ActivationConfig
from engram.retrieval.auto_recall import (
    RecallCooldown,
    apply_mcp_recall_enrichment,
    build_full_auto_recall_surface,
    build_lite_auto_recall_surface,
    build_session_prime_surface,
    compact_auto_recall_surface,
    compact_lite_auto_recall_surface,
    drain_mcp_triggered_intentions,
    extract_recall_query,
    plan_mcp_recall_middleware,
    plan_session_prime,
    should_recall_for_tool,
    store_mcp_auto_observe_turn,
)


def test_extract_recall_query_prefers_proper_nouns() -> None:
    assert (
        extract_recall_query("I'm working with John Smith on the React project")
        == "John Smith React"
    )


def test_extract_recall_query_falls_back_to_first_sentence() -> None:
    assert (
        extract_recall_query("working through native recall today. ignore this sentence")
        == "working through native recall today"
    )


def test_should_recall_for_tool_uses_per_tool_flags() -> None:
    cfg = ActivationConfig(
        auto_recall_on_observe=True,
        auto_recall_on_remember=False,
        auto_recall_on_tool_call=True,
    )

    assert should_recall_for_tool("observe", cfg) is True
    assert should_recall_for_tool("remember", cfg) is False
    assert should_recall_for_tool("recall", cfg) is True
    assert should_recall_for_tool("bootstrap_project", cfg) is False
    assert should_recall_for_tool("recall", None) is False


def test_plan_session_prime_respects_enablement_and_first_call() -> None:
    cfg = ActivationConfig(
        auto_recall_session_prime=True,
        auto_recall_session_prime_max_tokens=321,
    )

    plan = plan_session_prime(
        "Planning the Engram native Helix path",
        cfg,
        already_primed=False,
    )

    assert plan is not None
    assert plan.topic_hint == "Planning Engram Helix"
    assert plan.max_tokens == 321
    assert plan_session_prime("Planning the Engram path", cfg, already_primed=True) is None
    assert (
        plan_session_prime(
            "Planning the Engram path",
            ActivationConfig(auto_recall_session_prime=False),
            already_primed=False,
        )
        is None
    )


def test_plan_session_prime_allows_context_prime_without_topic() -> None:
    cfg = ActivationConfig(auto_recall_session_prime=True)

    plan = plan_session_prime("short text", cfg, already_primed=False)

    assert plan is not None
    assert plan.topic_hint is None


def test_plan_mcp_recall_middleware_plans_read_tool_side_effects() -> None:
    cfg = ActivationConfig(auto_recall_on_tool_call=True)

    plan = plan_mcp_recall_middleware(
        "What should we do about the Engram native Helix route plan?",
        tool_name="route_question",
        cfg=cfg,
        auto_observe=True,
    )

    assert plan.should_recall is True
    assert plan.auto_observe_content is True
    assert plan.ingest_live_turn is True
    assert plan.surface_notifications_when_recall_disabled is False


def test_plan_mcp_recall_middleware_skips_write_tool_ingest() -> None:
    cfg = ActivationConfig(auto_recall_on_observe=True)

    plan = plan_mcp_recall_middleware(
        "Observed content for the Engram native path",
        tool_name="observe",
        cfg=cfg,
        auto_observe=False,
    )

    assert plan.should_recall is True
    assert plan.auto_observe_content is False
    assert plan.ingest_live_turn is False


def test_plan_mcp_recall_middleware_surfaces_get_context_notifications_without_recall() -> None:
    cfg = ActivationConfig(
        auto_recall_on_tool_call=False,
        notification_surfacing_enabled=True,
    )

    plan = plan_mcp_recall_middleware(
        "",
        tool_name="get_context",
        cfg=cfg,
        auto_observe=False,
    )

    assert plan.should_recall is False
    assert plan.surface_notifications_when_recall_disabled is True
    assert plan.auto_observe_content is False
    assert plan.ingest_live_turn is False


def test_recall_cooldown_rate_and_topic_dedup() -> None:
    cooldown = RecallCooldown(max_per_minute=2, cooldown_seconds=60.0)
    cooldown.record("React migration", 0.0)

    assert cooldown.is_throttled("React migration status", 10.0) is True
    assert cooldown.is_throttled("SQLite compaction", 10.0) is False

    cooldown.record("SQLite compaction", 20.0)
    assert cooldown.is_throttled("Helix native", 30.0) is True
    assert cooldown.is_throttled("React migration", 70.0) is False


def test_compact_auto_recall_surface_filters_and_shapes_results() -> None:
    results = [
        {
            "result_type": "entity",
            "entity": {
                "name": "Engram",
                "type": "Project",
                "summary": "A" * 140,
            },
            "relationships": [
                {"predicate": "USES"},
                {"predicate": "PREFERS"},
                {"predicate": "TRACKS"},
                {"predicate": "IGNORED"},
            ],
            "score": 0.91,
        },
        {
            "result_type": "entity",
            "entity": {"name": "Low", "type": "Concept", "summary": "low score"},
            "score": 0.1,
        },
        {
            "result_type": "episode",
            "episode": {"id": "ep_ignored"},
            "score": 0.9,
        },
        {
            "result_type": "cue_episode",
            "cue": {
                "episode_id": "ep_cue",
                "cue_text": "B" * 160,
                "supporting_spans": ["one", "two", "three"],
                "projection_state": "cue_only",
            },
            "score": 0.87654,
        },
    ]

    surface = compact_auto_recall_surface(
        results,
        query="Engram native",
        packets=[{"packetType": "fact_packet"}],
        min_score=0.3,
    )

    assert surface == {
        "source": "auto_recall",
        "query_used": "Engram native",
        "packets": [{"packetType": "fact_packet"}],
        "entities": [
            {
                "name": "Engram",
                "type": "Project",
                "summary": "A" * 100,
                "top_facts": ["USES", "PREFERS", "TRACKS"],
            }
        ],
        "cue_episodes": [
            {
                "episode_id": "ep_cue",
                "cue_text": "B" * 140,
                "supporting_spans": ["one", "two"],
                "projection_state": "cue_only",
                "score": 0.8765,
            }
        ],
    }


def test_compact_auto_recall_surface_returns_none_without_surfaceable_results() -> None:
    assert (
        compact_auto_recall_surface(
            [{"result_type": "episode", "score": 0.9}],
            query="Engram native",
            min_score=0.3,
        )
        is None
    )


def test_compact_lite_auto_recall_surface_shapes_entity_probe_results() -> None:
    assert compact_lite_auto_recall_surface(
        [{"name": "Engram", "type": "Project"}],
        level="medium",
    ) == {
        "source": "recall_medium",
        "entities": [{"name": "Engram", "type": "Project"}],
    }


def test_compact_lite_auto_recall_surface_returns_none_for_empty_results() -> None:
    assert compact_lite_auto_recall_surface([], level="lite") is None


@pytest.mark.asyncio
async def test_build_lite_auto_recall_surface_dispatches_lite() -> None:
    manager = AsyncMock()
    manager.recall_lite.return_value = [{"name": "React"}]
    cache: dict = {}
    cfg = ActivationConfig()

    result = await build_lite_auto_recall_surface(
        manager,
        content="Working on the React migration today",
        group_id="native_brain",
        session_cache=cache,
        cfg=cfg,
    )

    assert result == {"source": "recall_lite", "entities": [{"name": "React"}]}
    manager.recall_lite.assert_awaited_once_with(
        text="Working on the React migration today",
        group_id="native_brain",
        session_cache=cache,
        token_budget=cfg.auto_recall_token_budget,
        cache_ttl=cfg.auto_recall_cache_ttl_seconds,
    )


@pytest.mark.asyncio
async def test_build_lite_auto_recall_surface_dispatches_medium() -> None:
    manager = AsyncMock()
    manager.recall_medium.return_value = [{"name": "React"}]
    cache: dict = {}
    cfg = ActivationConfig(auto_recall_level="medium")

    result = await build_lite_auto_recall_surface(
        manager,
        content="Working on the React migration today",
        group_id="native_brain",
        session_cache=cache,
        cfg=cfg,
    )

    assert result == {"source": "recall_medium", "entities": [{"name": "React"}]}
    manager.recall_medium.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_full_auto_recall_surface_dispatches_recall() -> None:
    manager = AsyncMock()
    manager.recall.return_value = [
        {
            "result_type": "entity",
            "entity": {"name": "Engram", "type": "Project", "summary": "Memory runtime"},
            "relationships": [{"predicate": "USES"}],
            "score": 0.9,
        }
    ]
    cfg = ActivationConfig(auto_recall_enabled=True)

    result = await build_full_auto_recall_surface(
        manager,
        content="Working on Engram native Helix recall surfaces",
        group_id="native_brain",
        cfg=cfg,
        session_last_recall_time=0.0,
        cooldown=None,
        now=100.0,
    )

    assert result == {
        "source": "auto_recall",
        "query_used": "Working Engram Helix",
        "packets": [],
        "entities": [
            {
                "name": "Engram",
                "type": "Project",
                "summary": "Memory runtime",
                "top_facts": ["USES"],
            }
        ],
    }
    manager.recall.assert_awaited_once()
    assert manager.recall.call_args.kwargs["group_id"] == "native_brain"
    assert manager.recall.call_args.kwargs["interaction_source"] == "auto_recall"


@pytest.mark.asyncio
async def test_build_full_auto_recall_surface_skips_recent_explicit_recall() -> None:
    manager = AsyncMock()
    cfg = ActivationConfig(auto_recall_enabled=True)

    result = await build_full_auto_recall_surface(
        manager,
        content="Working on Engram native Helix recall surfaces",
        group_id="native_brain",
        cfg=cfg,
        session_last_recall_time=90.0,
        cooldown=None,
        now=100.0,
    )

    assert result is None
    manager.recall.assert_not_called()


@pytest.mark.asyncio
async def test_build_session_prime_surface_fetches_context_and_marks_primed() -> None:
    manager = AsyncMock()
    manager.get_context.return_value = {"context": "native Helix"}
    cfg = ActivationConfig(
        auto_recall_session_prime=True,
        auto_recall_session_prime_max_tokens=256,
    )

    surface = await build_session_prime_surface(
        manager,
        content="Planning Engram native Helix parity",
        group_id="native_brain",
        cfg=cfg,
        already_primed=False,
    )

    assert surface.context == {"context": "native Helix"}
    assert surface.should_mark_primed is True
    manager.get_context.assert_awaited_once_with(
        group_id="native_brain",
        max_tokens=256,
        topic_hint="Planning Engram Helix",
        format="structured",
    )


@pytest.mark.asyncio
async def test_build_session_prime_surface_skips_when_already_primed() -> None:
    manager = AsyncMock()
    cfg = ActivationConfig(auto_recall_session_prime=True)

    surface = await build_session_prime_surface(
        manager,
        content="Planning Engram native Helix parity",
        group_id="native_brain",
        cfg=cfg,
        already_primed=True,
    )

    assert surface.context is None
    assert surface.should_mark_primed is False
    manager.get_context.assert_not_called()


@pytest.mark.asyncio
async def test_store_mcp_auto_observe_turn_uses_manager_store_episode() -> None:
    manager = AsyncMock()

    await store_mcp_auto_observe_turn(
        manager,
        content="Long route question that should become latent context",
        group_id="native_brain",
    )

    manager.store_episode.assert_awaited_once_with(
        "Long route question that should become latent context",
        "native_brain",
        source="tool_piggyback",
    )


@pytest.mark.asyncio
async def test_store_mcp_auto_observe_turn_swallows_store_errors() -> None:
    manager = AsyncMock()
    manager.store_episode.side_effect = RuntimeError("store failed")

    await store_mcp_auto_observe_turn(
        manager,
        content="Long route question that should become latent context",
        group_id="native_brain",
    )

    manager.store_episode.assert_awaited_once()


@pytest.mark.asyncio
async def test_drain_mcp_triggered_intentions_supports_sync_manager_facade() -> None:
    manager = Mock()
    manager.drain_triggered_intention_views.return_value = [{"trigger": "meeting"}]

    result = await drain_mcp_triggered_intentions(manager)

    assert result == [{"trigger": "meeting"}]
    manager.drain_triggered_intention_views.assert_called_once_with()


@pytest.mark.asyncio
async def test_drain_mcp_triggered_intentions_supports_async_manager_facade() -> None:
    manager = Mock()
    manager.drain_triggered_intention_views = AsyncMock(
        return_value=[{"trigger": "meeting"}]
    )

    result = await drain_mcp_triggered_intentions(manager)

    assert result == [{"trigger": "meeting"}]
    manager.drain_triggered_intention_views.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_drain_mcp_triggered_intentions_returns_none_for_empty_or_missing_facade() -> None:
    assert await drain_mcp_triggered_intentions(object()) is None

    manager = Mock()
    manager.drain_triggered_intention_views.return_value = []
    assert await drain_mcp_triggered_intentions(manager) is None


def test_apply_mcp_recall_enrichment_attaches_only_non_empty_values() -> None:
    response = {"ok": True}

    apply_mcp_recall_enrichment(
        response,
        session_context={},
        recalled_context=None,
        triggered_intentions=[],
        memory_notifications=[{"id": "notif_1", "kind": "surface"}],
    )

    assert response == {
        "ok": True,
        "memory_notifications": [{"id": "notif_1", "kind": "surface"}],
    }


def test_apply_mcp_recall_enrichment_preserves_payload_and_normalizes_sequences() -> None:
    response = {"ok": True, "tool": "recall"}

    apply_mcp_recall_enrichment(
        response,
        session_context={"topic": "native helix"},
        recalled_context={"source": "recall_lite", "entities": []},
        triggered_intentions=({"id": "intent_1", "action": "follow_up"},),
        memory_notifications=({"id": "notif_1", "kind": "surface"},),
    )

    assert response == {
        "ok": True,
        "tool": "recall",
        "session_context": {"topic": "native helix"},
        "recalled_context": {"source": "recall_lite", "entities": []},
        "triggered_intentions": [{"id": "intent_1", "action": "follow_up"}],
        "memory_notifications": [{"id": "notif_1", "kind": "surface"}],
    }
