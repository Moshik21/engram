from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
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
    run_mcp_recall_middleware,
    should_recall_for_tool,
    store_mcp_auto_observe_turn,
)


def _policy_cfg(**kwargs: object) -> ActivationConfig:
    """Isolate policy tests from consolidation/recall profile presets."""
    base = {
        "consolidation_profile": "off",
        "recall_profile": "off",
        "integration_profile": "off",
    }
    base.update(kwargs)
    return ActivationConfig(**base)


def test_mcp_auto_observe_turn_uses_capture_stage_helper() -> None:
    source = inspect.getsource(store_mcp_auto_observe_turn)

    assert "store_observation" in source
    assert "store_episode" not in source


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
    cfg = _policy_cfg(
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
    cfg = _policy_cfg(
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
            _policy_cfg(auto_recall_session_prime=False),
            already_primed=False,
        )
        is None
    )


def test_plan_session_prime_allows_context_prime_without_topic() -> None:
    cfg = _policy_cfg(auto_recall_session_prime=True)

    plan = plan_session_prime("short text", cfg, already_primed=False)

    assert plan is not None
    assert plan.topic_hint is None


def test_plan_mcp_recall_middleware_plans_read_tool_side_effects() -> None:
    cfg = _policy_cfg(auto_recall_on_tool_call=True)

    plan = plan_mcp_recall_middleware(
        "What should we do about the Engram native Helix route plan?",
        tool_name="route_question",
        cfg=cfg,
        auto_observe=True,
    )

    assert plan.should_recall is True
    assert plan.auto_observe_content is True
    assert plan.ingest_live_turn is True
    assert plan.cache_only is False
    assert plan.surface_notifications_when_recall_disabled is False


def test_plan_mcp_recall_middleware_skips_write_tool_ingest() -> None:
    cfg = _policy_cfg(auto_recall_on_observe=True)

    plan = plan_mcp_recall_middleware(
        "Observed content for the Engram native path",
        tool_name="observe",
        cfg=cfg,
        auto_observe=False,
    )

    assert plan.should_recall is True
    assert plan.auto_observe_content is False
    assert plan.ingest_live_turn is False
    assert plan.cache_only is True


def test_plan_mcp_recall_middleware_surfaces_get_context_notifications_without_recall() -> None:
    cfg = _policy_cfg(
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
    cfg = _policy_cfg(auto_recall_level="lite")

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
    cfg = _policy_cfg(auto_recall_level="medium")

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
async def test_build_lite_auto_recall_surface_uses_cached_packets_before_medium() -> None:
    manager = AsyncMock()
    manager.record_memory_operation = Mock()
    manager.get_recent_cached_memory_packets = Mock(
        return_value=[
            {
                "packet_type": "project_home",
                "title": "Project: Engram AXI",
                "summary": "AXI startup and packet cache dogfood evidence",
                "evidence_lines": ["loaded-store recall context performance dogfood"],
            }
        ]
    )
    cfg = _policy_cfg(
        auto_recall_level="medium",
        recall_packets_enabled=True,
        recall_packet_auto_limit=1,
    )

    result = await build_lite_auto_recall_surface(
        manager,
        content="Check Engram AXI loaded-store recall dogfood performance",
        group_id="native_brain",
        session_cache={},
        cfg=cfg,
    )

    assert result is not None
    assert result["source"] == "auto_recall"
    assert result["packets"][0]["title"] == "Project: Engram AXI"
    assert result["gate"]["modeExecuted"] == "cached"
    manager.get_recent_cached_memory_packets.assert_called_once()
    assert manager.get_recent_cached_memory_packets.call_args.kwargs["scopes"] == (
        "session_recent",
        "identity_core",
        "project_home",
        "explicit_recall",
    )
    manager.recall_medium.assert_not_called()
    samples = [call.args[1] for call in manager.record_memory_operation.call_args_list]
    assert any(sample.mode == "auto_recall_packet" and sample.cache_hit for sample in samples)
    assert any(
        sample.mode == "medium"
        and sample.status == "ok"
        and sample.skip_reason == "cache_satisfied"
        for sample in samples
    )


@pytest.mark.asyncio
async def test_build_lite_auto_recall_surface_uses_session_recent_before_medium() -> None:
    manager = AsyncMock()
    manager.record_memory_operation = Mock()
    manager.get_recent_cached_memory_packets = Mock(
        return_value=[
            {
                "_cache_scope": "session_recent",
                "packet_type": "recent_observation",
                "title": "Recent Observation: ep_perf",
                "summary": "Engram PyO3 dogfood medium timeout evidence",
                "evidence_lines": ["AXI value found medium recall_timeout samples"],
            }
        ]
    )
    cfg = _policy_cfg(
        auto_recall_level="medium",
        recall_packets_enabled=True,
        recall_packet_auto_limit=1,
    )

    result = await build_lite_auto_recall_surface(
        manager,
        content="Where do we stand on Engram PyO3 medium timeout evidence?",
        group_id="native_brain",
        session_cache={},
        cfg=cfg,
    )

    assert result is not None
    assert result["source"] == "auto_recall"
    assert result["packets"][0]["packet_type"] == "recent_observation"
    assert result["gate"]["modeExecuted"] == "cached"
    manager.recall_medium.assert_not_called()


@pytest.mark.asyncio
async def test_build_lite_auto_recall_surface_cache_only_skips_probe_on_cache_miss() -> None:
    manager = AsyncMock()
    manager.record_memory_operation = Mock()
    manager.get_recent_cached_memory_packets = Mock(return_value=[])
    cfg = _policy_cfg(
        auto_recall_level="medium",
        recall_packets_enabled=True,
        recall_packet_auto_limit=1,
    )

    result = await build_lite_auto_recall_surface(
        manager,
        content="Check Engram write-side auto recall cache miss latency",
        group_id="native_brain",
        session_cache={},
        cfg=cfg,
        cache_only=True,
    )

    assert result is None
    manager.recall_medium.assert_not_called()
    samples = [call.args[1] for call in manager.record_memory_operation.call_args_list]
    assert any(
        sample.mode == "auto_recall_packet"
        and sample.cache_hit is False
        and sample.packet_count == 0
        for sample in samples
    )
    assert any(
        sample.mode == "medium"
        and sample.status == "skipped"
        and sample.skip_reason == "cache_miss"
        and sample.cache_hit is False
        for sample in samples
    )


@pytest.mark.asyncio
async def test_build_lite_auto_recall_surface_records_short_content_skip() -> None:
    manager = AsyncMock()
    manager.record_memory_operation = Mock()
    cfg = _policy_cfg()

    result = await build_lite_auto_recall_surface(
        manager,
        content="ok",
        group_id="native_brain",
        session_cache={},
        cfg=cfg,
    )

    assert result is None
    manager.recall_lite.assert_not_called()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "auto_recall_gate"
    assert sample.status == "skipped"
    assert sample.skip_reason == "skipped_low_signal"
    assert sample.budget_ms == cfg.recall_budget_auto_lite_ms


@pytest.mark.asyncio
async def test_build_lite_auto_recall_surface_records_empty_result_skip() -> None:
    manager = AsyncMock()
    manager.record_memory_operation = Mock()
    manager.recall_lite.return_value = []
    cfg = _policy_cfg(auto_recall_level="lite")

    result = await build_lite_auto_recall_surface(
        manager,
        content="Working on the React migration today",
        group_id="native_brain",
        session_cache={},
        cfg=cfg,
    )

    assert result is None
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.status == "skipped"
    assert sample.skip_reason == "skipped_no_results"


@pytest.mark.asyncio
async def test_build_lite_auto_recall_surface_degrades_on_probe_timeout() -> None:
    async def slow_recall_lite(**_kwargs):
        await asyncio.sleep(0.05)
        return [{"name": "Engram"}]

    manager = AsyncMock()
    manager.record_memory_operation = Mock()
    manager.recall_lite.side_effect = slow_recall_lite
    cfg = _policy_cfg(auto_recall_level="lite", recall_budget_auto_lite_ms=10)

    result = await build_lite_auto_recall_surface(
        manager,
        content="Working on the Engram native Helix path",
        group_id="native_brain",
        session_cache={},
        cfg=cfg,
    )

    assert result is None
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "auto_recall_gate"
    assert sample.status == "degraded"
    assert sample.skip_reason == "recall_timeout"
    assert sample.timeout is True
    assert sample.degraded is True


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
    cfg = _policy_cfg(
        auto_recall_enabled=True,
        recall_need_analyzer_enabled=True,
    )

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
        "query_used": "Helix",
        "packets": [],
        "gate": {
            "decision": "triggered",
            "needType": "fact_lookup",
            "modeRequested": "deep",
            "modeExecuted": "deep",
            "budgetProfile": "auto_deep",
            "cacheHit": False,
        },
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
async def test_build_full_auto_recall_surface_uses_cached_packets_without_deep_recall() -> None:
    manager = AsyncMock()
    manager.record_memory_operation = Mock()
    manager.record_memory_need_analysis = Mock()
    manager.get_cached_memory_packets = Mock(
        return_value=SimpleNamespace(
            packets=[
                {
                    "packet_type": "project_state",
                    "title": "Project: Engram AXI",
                    "trust": {"source": "cache", "freshness": "fresh"},
                }
            ]
        )
    )
    cfg = _policy_cfg(
        auto_recall_enabled=True,
        recall_need_analyzer_enabled=True,
        recall_packets_enabled=True,
        recall_packet_auto_limit=1,
    )

    result = await build_full_auto_recall_surface(
        manager,
        content="What changed with Engram AXI startup?",
        group_id="native_brain",
        cfg=cfg,
        session_last_recall_time=0.0,
        cooldown=None,
        now=100.0,
    )

    assert result is not None
    assert result["packets"][0]["title"] == "Project: Engram AXI"
    assert result["gate"] == {
        "decision": "skipped_cache_satisfied",
        "needType": "temporal_update",
        "modeRequested": "deep",
        "modeExecuted": "cached",
        "budgetProfile": "auto_deep",
        "skipReason": "skipped_cache_satisfied",
        "cacheHit": True,
        "cacheSatisfied": True,
    }
    manager.recall.assert_not_called()
    group_id, need = manager.record_memory_need_analysis.call_args.args
    assert group_id == "native_brain"
    assert need.cache_satisfied is True
    assert need.mode_executed == "cached"


@pytest.mark.asyncio
async def test_build_full_auto_recall_surface_records_budget_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AsyncMock()
    manager.record_memory_operation = Mock()
    manager.record_memory_need_analysis = Mock()
    cfg = _policy_cfg(
        auto_recall_enabled=True,
        recall_need_analyzer_enabled=True,
    )

    def budget_exhausted(*_args, **_kwargs):
        return SimpleNamespace(
            profile="auto_deep",
            max_results=0,
            max_output_tokens=300,
            budget_ms=750,
            budget_tokens=300,
            timeout_degrades=True,
            exceeded=lambda _duration_ms=None: False,
        )

    monkeypatch.setattr(
        "engram.retrieval.auto_recall.recall_budget_for_profile",
        budget_exhausted,
    )

    result = await build_full_auto_recall_surface(
        manager,
        content="What changed with Engram AXI startup?",
        group_id="native_brain",
        cfg=cfg,
        session_last_recall_time=0.0,
        cooldown=None,
        now=100.0,
    )

    assert result is None
    manager.recall.assert_not_called()
    group_id, need = manager.record_memory_need_analysis.call_args.args
    assert group_id == "native_brain"
    assert need.skip_reason == "skipped_budget"
    assert need.budget_skipped is True
    assert need.mode_requested == "deep"
    assert need.mode_executed == "none"


@pytest.mark.asyncio
async def test_build_full_auto_recall_surface_skips_recent_explicit_recall() -> None:
    manager = AsyncMock()
    manager.record_memory_operation = Mock()
    cfg = _policy_cfg(auto_recall_enabled=True, recall_need_analyzer_enabled=False)

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
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.status == "skipped"
    assert sample.skip_reason == "skipped_recent_explicit"


@pytest.mark.asyncio
async def test_build_full_auto_recall_surface_records_disabled_skip() -> None:
    manager = AsyncMock()
    manager.record_memory_operation = Mock()
    cfg = _policy_cfg(auto_recall_enabled=False)

    result = await build_full_auto_recall_surface(
        manager,
        content="Working on Engram native Helix recall surfaces",
        group_id="native_brain",
        cfg=cfg,
        session_last_recall_time=0.0,
        cooldown=None,
        now=100.0,
    )

    assert result is None
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.status == "skipped"
    assert sample.skip_reason == "skipped_disabled"
    assert sample.budget_ms == cfg.recall_budget_auto_deep_ms


@pytest.mark.asyncio
async def test_build_full_auto_recall_surface_records_memory_need_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AsyncMock()
    manager.record_memory_operation = Mock()

    async def fake_analyze_memory_need(*_args, **_kwargs):
        return SimpleNamespace(
            should_recall=False,
            query_hint="",
            reasons=["acknowledgement"],
        )

    monkeypatch.setattr(
        "engram.retrieval.auto_recall.analyze_memory_need",
        fake_analyze_memory_need,
    )
    cfg = _policy_cfg(
        auto_recall_enabled=True,
        recall_need_analyzer_enabled=True,
    )

    result = await build_full_auto_recall_surface(
        manager,
        content="Thanks for the update",
        group_id="native_brain",
        cfg=cfg,
        session_last_recall_time=0.0,
        cooldown=None,
        now=100.0,
    )

    assert result is None
    manager.recall.assert_not_called()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.status == "skipped"
    assert sample.skip_reason == "skipped_ack"


@pytest.mark.asyncio
async def test_build_session_prime_surface_uses_cached_packets_and_marks_primed() -> None:
    manager = Mock()
    manager.get_cached_memory_packets.return_value = SimpleNamespace(
        packets=[
            {
                "packet_type": "project_home",
                "title": "Project File: docs/install/helix.md",
                "summary": "Native Helix startup path.",
            }
        ]
    )
    manager.get_recent_cached_memory_packets.return_value = []
    manager.record_memory_operation = Mock()
    manager.get_context = AsyncMock()
    cfg = _policy_cfg(
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

    assert surface.context is not None
    assert "Native Helix startup path." in surface.context["context"]
    assert surface.context["packet_cache"]["hit"] is True
    assert surface.should_mark_primed is True
    manager.get_context.assert_not_awaited()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.source == "mcp_session_prime"
    assert sample.status == "ok"
    assert sample.cache_hit is True
    assert sample.budget_miss is False
    assert all(
        call.kwargs.get("sync_persistent") is False
        for call in manager.get_cached_memory_packets.call_args_list
    )


@pytest.mark.asyncio
async def test_build_session_prime_surface_skips_when_cache_misses() -> None:
    manager = Mock()
    manager.get_cached_memory_packets.return_value = None
    manager.get_recent_cached_memory_packets.return_value = []
    manager.record_memory_operation = Mock()
    cfg = _policy_cfg(
        auto_recall_session_prime=True,
        recall_budget_startup_ms=25,
    )

    surface = await build_session_prime_surface(
        manager,
        content="Planning Engram native Helix parity",
        group_id="native_brain",
        cfg=cfg,
        already_primed=False,
    )

    assert surface.context is None
    assert surface.should_mark_primed is True
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "context"
    assert sample.source == "mcp_session_prime"
    assert sample.status == "skipped"
    assert sample.skip_reason == "cache_miss"
    assert sample.timeout is False
    assert sample.degraded is False
    assert sample.budget_miss is False
    manager.get_recent_cached_memory_packets.assert_called_once_with(
        "native_brain",
        scopes=("identity_core", "project_home"),
        limit_packets=cfg.recall_packet_auto_limit,
        sync_persistent=False,
    )


@pytest.mark.asyncio
async def test_build_session_prime_surface_skips_when_already_primed() -> None:
    manager = AsyncMock()
    cfg = _policy_cfg(auto_recall_session_prime=True)

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
async def test_store_mcp_auto_observe_turn_preserves_capture_call_shape() -> None:
    manager = AsyncMock()

    await store_mcp_auto_observe_turn(
        manager,
        content="Long route question that should become latent context",
        group_id="native_brain",
    )

    manager.store_episode.assert_awaited_once_with(
        content="Long route question that should become latent context",
        group_id="native_brain",
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
    manager.drain_triggered_intention_views = AsyncMock(return_value=[{"trigger": "meeting"}])

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


@pytest.mark.asyncio
async def test_run_mcp_recall_middleware_executes_recall_side_effects() -> None:
    cfg = _policy_cfg(
        auto_recall_on_tool_call=True,
        auto_recall_session_prime=True,
        notification_surfacing_enabled=True,
    )
    content = "What is the deployment strategy for the Engram native Helix route plan?"
    manager = Mock()
    manager.store_episode = AsyncMock(return_value="ep_1")
    manager.drain_triggered_intention_views = Mock(
        return_value=[{"trigger": "deployment", "action": "review rollout"}]
    )
    response: dict = {"status": "ok"}
    auto_recall_lite = AsyncMock(return_value={"source": "recall_lite", "entities": []})

    await run_mcp_recall_middleware(
        response,
        content=content,
        tool_name="route_question",
        cfg=cfg,
        group_id="brain",
        get_manager=Mock(return_value=manager),
        load_notifications=Mock(return_value=[{"title": "Found link"}]),
        auto_recall_lite=auto_recall_lite,
        session_prime=AsyncMock(return_value={"context": "briefing"}),
        ingest_live_turn=AsyncMock(),
        auto_observe=True,
    )

    auto_recall_lite.assert_awaited_once_with(content, manager, cfg, cache_only=False)
    manager.store_episode.assert_awaited_once_with(
        content=content,
        group_id="brain",
        source="tool_piggyback",
    )
    assert response == {
        "status": "ok",
        "session_context": {"context": "briefing"},
        "recalled_context": {"source": "recall_lite", "entities": []},
        "triggered_intentions": [{"trigger": "deployment", "action": "review rollout"}],
        "memory_notifications": [{"title": "Found link"}],
    }


@pytest.mark.asyncio
async def test_run_mcp_recall_middleware_uses_cache_only_for_write_tools() -> None:
    cfg = _policy_cfg(auto_recall_on_observe=True)
    content = "Observed Engram write-side auto recall cache miss latency."
    manager = Mock()
    response: dict = {"status": "ok"}
    auto_recall_lite = AsyncMock(return_value=None)

    await run_mcp_recall_middleware(
        response,
        content=content,
        tool_name="observe",
        cfg=cfg,
        group_id="brain",
        get_manager=Mock(return_value=manager),
        load_notifications=Mock(return_value=[]),
        auto_recall_lite=auto_recall_lite,
        session_prime=AsyncMock(return_value=None),
        ingest_live_turn=AsyncMock(),
    )

    auto_recall_lite.assert_awaited_once_with(content, manager, cfg, cache_only=True)


@pytest.mark.asyncio
async def test_run_mcp_recall_middleware_notification_fallback_without_manager() -> None:
    cfg = _policy_cfg(
        auto_recall_on_tool_call=False,
        notification_surfacing_enabled=True,
    )
    response: dict = {}

    await run_mcp_recall_middleware(
        response,
        content="",
        tool_name="get_context",
        cfg=cfg,
        group_id="brain",
        get_manager=Mock(side_effect=AssertionError("manager should not be loaded")),
        load_notifications=Mock(return_value=[{"title": "Found link"}]),
        auto_recall_lite=AsyncMock(),
        session_prime=AsyncMock(),
        ingest_live_turn=AsyncMock(),
    )

    assert response == {"memory_notifications": [{"title": "Found link"}]}
