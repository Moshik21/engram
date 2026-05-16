from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.retrieval.auto_recall import (
    RecallCooldown,
    apply_mcp_recall_enrichment,
    build_lite_auto_recall_surface,
    compact_auto_recall_surface,
    compact_lite_auto_recall_surface,
    extract_recall_query,
    plan_mcp_recall_middleware,
    plan_session_prime,
    should_recall_for_tool,
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
