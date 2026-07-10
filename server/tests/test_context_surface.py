from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.retrieval import context_builder as context_builder_module
from engram.retrieval.context_builder import (
    _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    _PROJECT_FILE_PREFIX_CACHE,
    _PROJECT_FILE_PREFIX_WARMUP_TASKS,
    MemoryContextBuilder,
    _project_file_fallback_packets,
    _read_project_file_prefix,
    build_api_context_surface,
    build_mcp_context_surface,
    build_mcp_context_tool_surface,
    project_file_fallback_packet_payloads,
    schedule_project_file_prefix_warmup,
)
from engram.storage.memory.activation import MemoryActivationStore

CONTEXT_RESULT = {
    "context": "## Active Memory\nAlice works on Engram.",
    "entity_count": 2,
    "fact_count": 3,
    "token_estimate": 42,
    "format": "briefing",
}


@pytest.mark.asyncio
async def test_project_file_prefix_warmup_populates_in_memory_prefix_cache(
    tmp_path,
) -> None:
    project_dir = tmp_path / "Engram"
    docs_dir = project_dir / "docs"
    docs_dir.mkdir(parents=True)
    (project_dir / "README.md").write_text(
        "Engram native PyO3 Helix startup-safe AXI context.\n",
        encoding="utf-8",
    )
    (docs_dir / "CURRENT_HANDOFF.md").write_text(
        "Current handoff covers runtime fast packet warming.\n",
        encoding="utf-8",
    )
    _PROJECT_FILE_PREFIX_CACHE.clear()
    _PROJECT_FILE_PREFIX_WARMUP_TASKS.clear()

    assert schedule_project_file_prefix_warmup(
        str(project_dir),
        topic_hint="runtime fast packet",
    )
    tasks = tuple(_PROJECT_FILE_PREFIX_WARMUP_TASKS.values())
    if tasks:
        await asyncio.gather(*tasks)

    assert _PROJECT_FILE_PREFIX_CACHE
    assert any(cache_key.endswith("README.md") for cache_key in _PROJECT_FILE_PREFIX_CACHE)
    _PROJECT_FILE_PREFIX_WARMUP_TASKS.clear()


@pytest.mark.asyncio
async def test_project_file_prefix_warmup_dedupes_in_flight_task(
    tmp_path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "Engram"
    project_dir.mkdir()
    release = threading.Event()
    calls = 0

    def fake_warm_project_file_prefix_cache(
        _project_path: str,
        *,
        topic_hint: str | None = None,
    ) -> None:
        nonlocal calls
        assert topic_hint == "Engram"
        calls += 1
        release.wait(timeout=1)

    monkeypatch.setattr(
        context_builder_module,
        "_warm_project_file_prefix_cache",
        fake_warm_project_file_prefix_cache,
    )
    _PROJECT_FILE_PREFIX_WARMUP_TASKS.clear()

    assert schedule_project_file_prefix_warmup(str(project_dir))
    assert not schedule_project_file_prefix_warmup(str(project_dir))
    release.set()
    tasks = tuple(_PROJECT_FILE_PREFIX_WARMUP_TASKS.values())
    if tasks:
        await asyncio.gather(*tasks)

    assert calls == 1
    _PROJECT_FILE_PREFIX_WARMUP_TASKS.clear()


@pytest.mark.asyncio
async def test_project_file_context_fallback_uses_dedicated_executor(
    monkeypatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def fake_run_project_file_executor(function, /, *args, **kwargs):
        calls.append((function.__name__, args, kwargs))
        return (
            [
                {
                    "packet_type": "project_home",
                    "title": "Project File: README.md",
                    "summary": "Executor-isolated project fallback.",
                    "trust": {"source": "project_file"},
                }
            ],
            1.5,
        )

    monkeypatch.setattr(
        context_builder_module,
        "_run_project_file_executor",
        fake_run_project_file_executor,
    )

    task = context_builder_module._start_project_file_context_fallback_task(
        topic_hint="executor isolation",
        project_path="/tmp/Engram",
        max_packets=1,
        reason="test",
    )

    assert task is not None
    packets, duration_ms = await task
    assert packets[0]["summary"] == "Executor-isolated project fallback."
    assert duration_ms == 1.5
    assert calls == [
        (
            "_build_project_file_context_fallback_packets",
            (),
            {
                "topic_hint": "executor isolation",
                "project_path": "/tmp/Engram",
                "max_packets": 1,
                "reason": "test",
            },
        )
    ]


@pytest.mark.asyncio
async def test_api_context_surface_maps_rest_keys() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)

    payload = await build_api_context_surface(
        manager,
        group_id="native_brain",
        max_tokens=1200,
        topic_hint="Alice",
        project_path="/tmp/engram",
        format="briefing",
        operation_source="api_context",
    )

    assert payload == {
        "context": "## Active Memory\nAlice works on Engram.",
        "entityCount": 2,
        "factCount": 3,
        "tokenEstimate": 42,
        "format": "briefing",
    }
    manager.get_context.assert_awaited_once_with(
        group_id="native_brain",
        max_tokens=1200,
        topic_hint="Alice",
        project_path="/tmp/engram",
        format="briefing",
        operation_source="api_context",
    )


@pytest.mark.asyncio
async def test_api_context_surface_forwards_cached_packet_metadata() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(
        return_value={
            **CONTEXT_RESULT,
            "cached_packets": [
                {
                    "packet_type": "project_home",
                    "title": "Project Home: Engram",
                    "trust": {"source": "cache", "freshness": "fresh"},
                }
            ],
            "packet_cache": {
                "hit": True,
                "packet_count": 1,
                "scopes": {"project_home": 1},
            },
        }
    )

    payload = await build_api_context_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
    )

    assert payload["cachedPackets"][0]["title"] == "Project Home: Engram"
    assert payload["cachedPackets"][0]["trust"]["source"] == "cache"
    assert payload["packetCache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_home": 1},
    }


@pytest.mark.asyncio
async def test_api_context_surface_maps_degraded_budget_metadata() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(
        return_value={
            **CONTEXT_RESULT,
            "status": "degraded",
            "budget": {
                "profile": "explicit",
                "surface": "rest",
                "mode": "api_context",
                "max_wall_ms": 2000,
                "duration_ms": 2001.0,
                "budget_miss": True,
                "timeout": True,
                "degraded": True,
                "skip_reason": "context_timeout",
            },
            "lifecycle": {
                "stage": "recall",
                "degraded": True,
                "timeout": True,
                "skip_reason": "context_timeout",
            },
        }
    )

    payload = await build_api_context_surface(manager, group_id="native_brain")

    assert payload["status"] == "degraded"
    assert payload["budget"]["maxWallMs"] == 2000
    assert payload["budget"]["skipReason"] == "context_timeout"
    assert payload["lifecycle"]["skipReason"] == "context_timeout"


@pytest.mark.asyncio
async def test_mcp_context_surface_preserves_raw_manager_shape() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="Alice",
    )

    assert payload == CONTEXT_RESULT
    manager.get_context.assert_awaited_once_with(
        group_id="native_brain",
        max_tokens=2000,
        topic_hint="Alice",
        project_path=None,
        format="structured",
        operation_source="mcp_context",
    )


@pytest.mark.asyncio
async def test_mcp_context_surface_returns_cached_packets_without_deep_context() -> None:
    cached_packet = {
        "packet_type": "project_home",
        "title": "Project Home: Engram",
        "summary": "Cached Engram context with ANTHROPIC_API_KEY=sk-ant-secret-value.",
        "trust": {"source": "cache", "freshness": "fresh"},
    }
    manager = MagicMock()

    def cached_packets(_group_id: str, *, scope: str, **_kwargs):
        if scope == "project_home":
            return SimpleNamespace(packets=[cached_packet])
        return None

    manager.get_cached_memory_packets.side_effect = cached_packets
    manager.fast_recall_fallback = None
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        project_path="/Users/konnermoshier/Engram",
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_home": 1},
    }
    assert "## Cached Memory Packets" in payload["context"]
    assert "ANTHROPIC_API_KEY=[redacted]" in payload["context"]
    assert "sk-ant-secret-value" not in payload["context"]
    assert "ANTHROPIC_API_KEY=[redacted]" in payload["cached_packets"][0]["summary"]
    manager.get_context.assert_not_awaited()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "context"
    assert sample.cache_hit is True
    assert sample.packet_count == 1


@pytest.mark.asyncio
async def test_mcp_context_surface_uses_session_recent_cache_before_project_home() -> None:
    cached_packet = {
        "packet_type": "recent_observation",
        "title": "Recent Observation: ep_recent",
        "summary": "Fresh observe latency proof from the current Codex session.",
        "episode_ids": ["ep_recent"],
        "trust": {"source": "mcp_observe", "freshness": "fresh"},
    }
    manager = MagicMock()
    calls: list[tuple[str, str | None, str | None]] = []

    def cached_packets(
        _group_id: str,
        *,
        scope: str,
        topic_hint: str | None = None,
        project_path: str | None = None,
        **_kwargs,
    ):
        calls.append((scope, topic_hint, project_path))
        if scope == "session_recent":
            return SimpleNamespace(packets=[cached_packet])
        return None

    manager.get_cached_memory_packets.side_effect = cached_packets
    manager.fast_recall_fallback = None
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="fresh observe latency proof",
        project_path="/Users/konnermoshier/Engram",
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"session_recent": 1},
    }
    assert "Fresh observe latency proof" in payload["context"]
    assert payload["cached_packets"][0]["episode_ids"] == ["ep_recent"]
    assert calls[0] == ("identity_core", None, None)
    assert calls[1] == ("session_recent", None, None)
    manager.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_context_surface_filters_project_home_when_session_recent_matches() -> None:
    recent_packet = {
        "packet_type": "recent_observation",
        "title": "Recent Observation: ep_recent",
        "summary": "20260527-sunstone hook capture should be immediate context.",
        "episode_ids": ["ep_recent"],
        "trust": {"source": "api_auto_observe", "freshness": "fresh"},
    }
    project_packet = {
        "packet_type": "project_home",
        "title": "Project File: README.md",
        "summary": "Generic project overview with no matching hook token.",
        "trust": {"source": "project_file", "freshness": "local"},
    }
    manager = MagicMock()

    def cached_packets(_group_id: str, *, scope: str, **_kwargs):
        if scope == "session_recent":
            return SimpleNamespace(packets=[recent_packet])
        if scope == "project_home":
            return SimpleNamespace(packets=[project_packet])
        return None

    manager.get_cached_memory_packets.side_effect = cached_packets
    manager.fast_recall_fallback = None
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="20260527-sunstone hook immediate context",
        project_path="/Users/konnermoshier/Engram",
    )

    assert payload["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"session_recent": 1},
    }
    assert payload["cached_packets"][0]["episode_ids"] == ["ep_recent"]
    assert "Generic project overview" not in payload["context"]
    manager.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_context_surface_keeps_project_home_for_broader_recent_topics() -> None:
    recent_packet = {
        "packet_type": "recent_observation",
        "title": "Recent Observation: ep_recent",
        "summary": "Loaded-store recall performance is the current dogfood focus.",
        "episode_ids": ["ep_recent"],
        "provenance": ["episode:ep_recent", "source:mcp"],
        "trust": {"source": "mcp_observe", "freshness": "fresh"},
    }
    project_packet = {
        "packet_type": "cue_packet",
        "title": "Latent Memory: ep_loaded",
        "summary": "Loaded-store recall packet cache evidence from an earlier dogfood run.",
        "episode_ids": ["ep_loaded"],
        "trust": {"source": "cue", "freshness": "unknown"},
    }
    manager = MagicMock()

    def cached_packets(_group_id: str, *, scope: str, **_kwargs):
        if scope == "session_recent":
            return SimpleNamespace(packets=[recent_packet])
        if scope == "project_home":
            return SimpleNamespace(packets=[project_packet])
        return None

    manager.get_cached_memory_packets.side_effect = cached_packets
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="loaded-store recall performance packet cache",
        project_path="/Users/konnermoshier/Engram",
    )

    assert payload["packet_cache"] == {
        "hit": True,
        "packet_count": 2,
        "scopes": {"session_recent": 1, "project_home": 1},
    }
    assert {packet["episode_ids"][0] for packet in payload["cached_packets"]} == {
        "ep_recent",
        "ep_loaded",
    }
    manager.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_context_surface_enriches_session_recent_only_cache() -> None:
    recent_packet = {
        "packet_type": "recent_observation",
        "title": "Recent Observation: ep_recent",
        "summary": "Loaded-store recall performance is the current dogfood focus.",
        "episode_ids": ["ep_recent"],
        "provenance": ["episode:ep_recent", "source:mcp"],
        "trust": {"source": "mcp_observe", "freshness": "fresh"},
    }
    project_file_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/dogfood-startup-validation-goal.md",
        "summary": "Loaded-store recall performance project-file fallback.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:docs/dogfood-startup-validation-goal.md"],
    }
    loaded_store_result = {
        "result_type": "cue_episode",
        "cue": {
            "episode_id": "ep_loaded",
            "cue_text": ("mentions: loaded-store recall performance packet cache older evidence"),
            "supporting_spans": [],
            "projection_state": "projected",
        },
        "episode": {
            "id": "ep_loaded",
            "source": "codex",
            "created_at": "2026-05-27T09:40:42",
        },
        "score": 1.0,
        "score_breakdown": {"semantic": 1.0},
    }
    manager = MagicMock()

    def cached_packets(_group_id: str, *, scope: str, **_kwargs):
        if scope == "session_recent":
            return SimpleNamespace(packets=[recent_packet])
        if scope == "project_home":
            return SimpleNamespace(packets=[project_file_packet])
        return None

    manager.get_activation_config.return_value = ActivationConfig(
        recall_fast_preflight_timeout_ms=200,
        recall_packet_cache_enabled=True,
    )
    manager.get_cached_memory_packets.side_effect = cached_packets
    manager.fast_recall_fallback = AsyncMock(return_value=[loaded_store_result])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="loaded-store recall performance packet cache",
        project_path="/Users/konnermoshier/Engram",
    )

    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 1,
        "scopes": {"loaded_store_context": 1},
    }
    assert payload["cached_packets"][0]["episode_ids"] == ["ep_loaded"]
    manager.fast_recall_fallback.assert_awaited_once()
    manager.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_context_surface_uses_specific_session_recent_without_enrichment() -> None:
    recent_packets = [
        {
            "packet_type": "recent_observation",
            "title": "Recent Observation: ep_aqua",
            "summary": (
                "Dogfood mcp observe latency second sample aquamarinemarker "
                "20260528 write side trace."
            ),
            "episode_ids": ["ep_aqua"],
            "provenance": ["episode:ep_aqua", "source:codex"],
            "trust": {"source": "mcp_observe", "freshness": "fresh"},
        },
        {
            "packet_type": "recent_observation",
            "title": "Recent Observation: ep_citrine",
            "summary": (
                "Dogfood mcp observe latency after session recent in memory "
                "citrinemarker 20260528 write side trace."
            ),
            "episode_ids": ["ep_citrine"],
            "provenance": ["episode:ep_citrine", "source:codex"],
            "trust": {"source": "mcp_observe", "freshness": "fresh"},
        },
    ]
    project_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/CURRENT_HANDOFF.md",
        "summary": (
            "Older dogfood handoff mentions 20260528 matrix evidence, but not "
            "the live marker observations."
        ),
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:docs/CURRENT_HANDOFF.md"],
    }
    loaded_store_result = {
        "result_type": "cue_episode",
        "cue": {
            "episode_id": "ep_loaded",
            "cue_text": "mentions: older loaded-store write-side observe evidence",
            "supporting_spans": [],
            "projection_state": "projected",
        },
        "episode": {
            "id": "ep_loaded",
            "source": "codex",
            "created_at": "2026-05-27T09:40:42",
        },
        "score": 1.0,
        "score_breakdown": {"semantic": 1.0},
    }
    manager = MagicMock()

    def cached_packets(_group_id: str, *, scope: str, **_kwargs):
        if scope == "session_recent":
            return SimpleNamespace(packets=recent_packets)
        if scope == "project_home":
            return SimpleNamespace(packets=[project_packet])
        return None

    manager.get_activation_config.return_value = ActivationConfig(
        recall_fast_preflight_timeout_ms=200,
        recall_packet_cache_enabled=True,
    )
    manager.get_cached_memory_packets.side_effect = cached_packets
    manager.fast_recall_fallback = AsyncMock(return_value=[loaded_store_result])
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint=(
            "write side observe latency citrinemarker aquamarinemarker 20260528 dogfood evidence"
        ),
        project_path="/Users/konnermoshier/Engram",
    )

    assert payload["packet_cache"] == {
        "hit": True,
        "packet_count": 2,
        "scopes": {"session_recent": 2},
    }
    assert "aquamarinemarker" in payload["context"]
    assert "citrinemarker" in payload["context"]
    assert {packet["episode_ids"][0] for packet in payload["cached_packets"]} == {
        "ep_aqua",
        "ep_citrine",
    }
    manager.fast_recall_fallback.assert_not_awaited()
    manager.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_context_surface_prefers_project_loaded_store_preflight(
    tmp_path,
) -> None:
    project_dir = tmp_path / "Engram"
    project_dir.mkdir()
    wrong_result = {
        "result_type": "episode",
        "episode": {
            "id": "ep_wrong",
            "source": "MachineShopScheduler",
            "content": "MachineShopScheduler INSERT OR REPLACE migration note.",
        },
        "score": 1.0,
        "score_breakdown": {"semantic": 1.0},
        "linked_entities": [],
    }
    right_result = {
        "result_type": "episode",
        "episode": {
            "id": "ep_right",
            "source": "project-bootstrap",
            "content": "Engram dogfood finalize INSERT OR REPLACE evidence.",
        },
        "score": 0.9,
        "score_breakdown": {"semantic": 0.9},
        "linked_entities": [],
    }
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_fast_preflight_timeout_ms=200,
        recall_packet_cache_enabled=True,
    )
    manager.get_cached_memory_packets.return_value = None
    manager.fast_recall_fallback = AsyncMock(return_value=[wrong_result, right_result])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="dogfood finalize INSERT OR REPLACE evidence",
        project_path=str(project_dir),
    )

    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 1,
        "scopes": {"loaded_store_context": 1},
    }
    assert payload["cached_packets"][0]["episode_ids"] == ["ep_right"]
    assert "MachineShopScheduler" not in payload["context"]
    manager.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_context_surface_uses_stable_project_cache_for_specific_topic() -> None:
    cached_packet = {
        "packet_type": "project_home",
        "title": "Project Home: Engram",
        "summary": "Stable project cache covers native PyO3 dogfood performance.",
        "trust": {"source": "cache", "freshness": "fresh"},
    }
    manager = MagicMock()
    calls: list[tuple[str, str | None, str | None]] = []

    def cached_packets(
        _group_id: str,
        *,
        scope: str,
        topic_hint: str | None = None,
        project_path: str | None = None,
        **_kwargs,
    ):
        calls.append((scope, topic_hint, project_path))
        if (
            scope == "project_home"
            and topic_hint == "Engram"
            and project_path == "/Users/konnermoshier/Engram"
        ):
            return SimpleNamespace(packets=[cached_packet])
        return None

    manager.get_cached_memory_packets.side_effect = cached_packets
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="Engram native PyO3 dogfood performance goal continuation",
        project_path="/Users/konnermoshier/Engram",
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_home": 1},
    }
    assert "native PyO3 dogfood performance" in payload["context"]
    assert (
        "project_home",
        "Engram",
        "/Users/konnermoshier/Engram",
    ) in calls
    manager.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_context_surface_uses_project_files_when_stable_cache_is_irrelevant(
    tmp_path,
) -> None:
    (tmp_path / "README.md").write_text("# Engram\nGeneral project overview.")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        + ("General filler before the relevant late evidence.\n" * 120)
        + "- Packet cache should keep context useful when recall latency is high.\n"
    )
    cached_packet = {
        "packet_type": "project_home",
        "title": "Project Home: Engram",
        "summary": "Generic overview only.",
        "trust": {"source": "cache", "freshness": "fresh"},
    }

    manager = MagicMock()
    manager.get_cached_memory_packets.return_value = SimpleNamespace(packets=[cached_packet])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="packet cache recall latency",
        project_path=str(tmp_path),
    )

    assert payload["status"] == "ok"
    assert payload["budget"]["skip_reason"] == "cache_relevance_miss"
    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 2,
        "scopes": {"project_file_fallback": 2},
    }
    assert "Project File: docs/memory-value-latency-plan.md" in payload["context"]
    assert "Memory Value and Latency Plan" in payload["context"]
    assert "Packet cache should keep context useful" in payload["context"]
    assert "did not match the specific topic" in payload["context"]
    assert "cache_relevance_miss" in payload["diagnostics"]["stage_timings_ms"]
    assert payload["diagnostics"]["stage_timings_ms"]["project_file_fallback"] >= 0
    assert (
        payload["budget"]["duration_ms"]
        >= payload["diagnostics"]["stage_timings_ms"]["project_file_fallback"]
    )
    manager.get_context.assert_not_awaited()
    cache_keys = {
        (call.kwargs.get("scope"), call.kwargs.get("topic_hint"), call.kwargs.get("project_path"))
        for call in manager.cache_memory_packets.call_args_list
    }
    assert ("project_home", "packet cache recall latency", str(tmp_path)) in cache_keys
    assert ("project_home", tmp_path.name, str(tmp_path)) in cache_keys
    assert all(
        call.kwargs.get("persist") is True for call in manager.cache_memory_packets.call_args_list
    )
    sync_flags = [
        call.kwargs.get("sync_persistent")
        for call in manager.get_cached_memory_packets.call_args_list
    ]
    assert False in sync_flags
    assert True in sync_flags


@pytest.mark.asyncio
async def test_mcp_context_surface_prebuilds_project_files_during_loaded_store_miss(
    monkeypatch,
    tmp_path,
) -> None:
    import engram.retrieval.context_builder as context_builder

    async def empty_preflight(**_kwargs):
        await asyncio.sleep(0.05)
        return []

    packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/memory-value-latency-plan.md",
        "summary": "Prebuilt project fallback packet.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:docs/memory-value-latency-plan.md"],
        "_cache_scope": "project_file_fallback",
    }
    cache_flags: list[bool | None] = []

    def fake_project_file_payloads(*_args, **kwargs):
        cache_flags.append(kwargs.get("cache"))
        time.sleep(0.02)
        return [packet]

    monkeypatch.setattr(
        context_builder,
        "project_file_fallback_packet_payloads",
        fake_project_file_payloads,
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_packet_cache_enabled=True,
        context_fast_preflight_timeout_ms=100,
    )
    manager.get_cached_memory_packets.return_value = None
    manager.fast_recall_fallback = AsyncMock(side_effect=empty_preflight)
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="prebuilt project fallback loaded store miss",
        project_path=str(tmp_path),
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 1,
        "scopes": {"project_file_fallback": 1},
    }
    assert cache_flags == [False]
    stage_timings = payload["diagnostics"]["stage_timings_ms"]
    assert stage_timings["project_file_fallback"] < 50
    assert stage_timings["loaded_store_context_preflight"] >= 0
    assert manager.cache_memory_packets.called


@pytest.mark.asyncio
async def test_mcp_context_surface_soft_waits_loaded_store_before_project_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    import engram.retrieval.context_builder as context_builder

    async def slow_preflight(**_kwargs):
        await asyncio.sleep(0.12)
        return [
            {
                "result_type": "cue_episode",
                "cue": {
                    "episode_id": "ep_late",
                    "cue_text": "late loaded-store context",
                    "supporting_spans": ["late loaded-store context"],
                    "projection_state": "projected",
                },
                "episode": {"id": "ep_late", "source": "codex"},
                "score": 1.0,
            }
        ]

    packet = {
        "packet_type": "project_home",
        "title": "Project File: README.md",
        "summary": "Soft wait project fallback packet.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:README.md"],
        "_cache_scope": "project_file_fallback",
    }

    def fake_project_file_payloads(*_args, **_kwargs):
        return [packet]

    monkeypatch.setattr(
        context_builder,
        "project_file_fallback_packet_payloads",
        fake_project_file_payloads,
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_packet_cache_enabled=True,
        context_fast_preflight_timeout_ms=250,
        context_fast_preflight_soft_wait_ms=25,
    )
    manager.get_cached_memory_packets.return_value = None
    manager.fast_recall_fallback = AsyncMock(side_effect=slow_preflight)
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="soft wait loaded store project fallback",
        project_path=str(tmp_path),
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"]["scopes"] == {"project_file_fallback": 1}
    assert payload["cached_packets"][0]["summary"] == "Soft wait project fallback packet."
    stage_timings = payload["diagnostics"]["stage_timings_ms"]
    assert stage_timings["loaded_store_context_preflight"] < 100
    assert stage_timings["project_file_fallback"] >= 0
    assert manager.fast_recall_fallback.await_count == 1

    await asyncio.sleep(0.13)
    cached_sources = [
        packet_arg["trust"]["source"]
        for call in manager.cache_memory_packets.call_args_list
        for packet_arg in call.kwargs.get("packets", [])
        if isinstance(packet_arg, dict) and isinstance(packet_arg.get("trust"), dict)
    ]
    assert "cue" in cached_sources


@pytest.mark.asyncio
async def test_mcp_context_surface_reuses_recent_relevant_project_file_cache(
    monkeypatch,
    tmp_path,
) -> None:
    project_dir = tmp_path / "Engram"
    project_dir.mkdir()
    topic = "semanticgravity cold exact project file scan latency packet reuse"
    recent_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/CURRENT_HANDOFF.md",
        "summary": "Semanticgravity cold exact project file scan latency packet reuse evidence.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:docs/CURRENT_HANDOFF.md"],
        "_project_file_fallback_topic_hint": "older nearby topic",
        "_project_file_fallback_project_path": str(project_dir),
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }
    unrelated_identity_packet = {
        "packet_type": "identity_core",
        "title": "Identity Core",
        "summary": "Durable unrelated preference.",
        "trust": {"source": "graph", "freshness": "fresh"},
        "provenance": ["entity:user"],
    }

    async def fail_project_file_executor(*_args, **_kwargs):
        raise AssertionError("recent relevant project-file cache should avoid scan")

    def get_cached_packets(
        _group_id,
        *,
        scope,
        topic_hint=None,
        project_path=None,
        **_kwargs,
    ):
        if scope == "identity_core":
            return SimpleNamespace(packets=[unrelated_identity_packet])
        if scope == "project_home" and topic_hint == "Engram" and project_path == str(project_dir):
            return SimpleNamespace(packets=[recent_packet])
        return None

    monkeypatch.setattr(
        context_builder_module,
        "_run_project_file_executor",
        fail_project_file_executor,
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_packet_cache_enabled=True,
        context_fast_preflight_timeout_ms=100,
    )
    manager.get_cached_memory_packets.side_effect = get_cached_packets
    manager.get_recent_cached_memory_packets.return_value = [recent_packet]
    manager.fast_recall_fallback = AsyncMock(return_value=[])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint=topic,
        project_path=str(project_dir),
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_file_recent_reuse": 1},
    }
    assert payload["cached_packets"][0]["summary"] == recent_packet["summary"]
    assert payload["cached_packets"][0]["_project_file_fallback_recent_cache_reuse"] is True
    manager.get_recent_cached_memory_packets.assert_called_once_with(
        "native_brain",
        scopes=("project_home",),
        limit_packets=12,
        sync_persistent=True,
    )
    manager.fast_recall_fallback.assert_not_awaited()
    manager.get_context.assert_not_awaited()
    manager.cache_memory_packets.assert_not_called()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.cache_hit is True


@pytest.mark.asyncio
async def test_mcp_context_surface_ignores_unrelated_recent_project_file_cache(
    monkeypatch,
    tmp_path,
) -> None:
    project_dir = tmp_path / "Engram"
    project_dir.mkdir()
    topic = "semanticgravity cold exact project file scan latency packet reuse"
    unrelated_packet = {
        "packet_type": "project_home",
        "title": "Project File: README.md",
        "summary": "Cached project context and recall notes from 20260528 without the marker.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:README.md"],
        "_project_file_fallback_topic_hint": "older unrelated topic",
        "_project_file_fallback_project_path": str(project_dir),
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }
    fresh_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/CURRENT_HANDOFF.md",
        "summary": "Fresh semanticgravity cold exact project file scan latency evidence.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:docs/CURRENT_HANDOFF.md"],
        "_cache_scope": "project_file_fallback",
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }

    async def fake_run_project_file_executor(function, /, *args, **kwargs):
        assert function.__name__ == "_build_project_file_context_fallback_packets"
        return [fresh_packet], 5.0

    monkeypatch.setattr(
        context_builder_module,
        "_run_project_file_executor",
        fake_run_project_file_executor,
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_packet_cache_enabled=True,
        context_fast_preflight_timeout_ms=100,
        context_fast_preflight_soft_wait_ms=50,
    )
    manager.get_cached_memory_packets.return_value = None
    manager.get_recent_cached_memory_packets.return_value = [unrelated_packet]
    manager.fast_recall_fallback = AsyncMock(return_value=[])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint=topic,
        project_path=str(project_dir),
        operation_source="axi_context",
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 1,
        "scopes": {"project_file_fallback": 1},
    }
    assert payload["cached_packets"][0]["summary"] == fresh_packet["summary"]
    assert "without the marker" not in payload["context"]
    manager.get_recent_cached_memory_packets.assert_called_once_with(
        "native_brain",
        scopes=("project_home",),
        limit_packets=12,
        sync_persistent=True,
    )
    manager.get_context.assert_not_awaited()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.cache_hit is False


@pytest.mark.asyncio
async def test_mcp_context_surface_uses_project_file_cache_rescue_while_scan_runs(
    monkeypatch,
    tmp_path,
) -> None:
    project_dir = tmp_path / "Engram"
    project_dir.mkdir()
    exact_topic = "qxjv norel zaffron plinket 20260527"
    cached_packet = {
        "packet_type": "project_home",
        "title": "Project File: README.md",
        "summary": "Cached stable project packet for qxjv norel zaffron plinket.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:README.md"],
        "_project_file_fallback_topic_hint": "older dogfood topic",
        "_project_file_fallback_project_path": str(project_dir),
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }
    fresh_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/CURRENT_HANDOFF.md",
        "summary": "Fresh project scan packet.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:docs/CURRENT_HANDOFF.md"],
        "_cache_scope": "project_file_fallback",
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }

    async def fake_run_project_file_executor(function, /, *args, **kwargs):
        assert function.__name__ == "_build_project_file_context_fallback_packets"
        await asyncio.sleep(0.06)
        return [fresh_packet], 60.0

    async def slow_preflight(**_kwargs):
        await asyncio.sleep(0.2)
        return []

    stable_lookup_sync_flags: list[bool | None] = []

    def get_cached_packets(
        _group_id,
        *,
        scope,
        topic_hint=None,
        project_path=None,
        sync_persistent=None,
        **_kwargs,
    ):
        if scope == "project_home" and topic_hint == "Engram" and project_path == str(project_dir):
            stable_lookup_sync_flags.append(sync_persistent)
            return SimpleNamespace(packets=[cached_packet])
        return None

    monkeypatch.setattr(
        context_builder_module,
        "_run_project_file_executor",
        fake_run_project_file_executor,
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_packet_cache_enabled=True,
        context_fast_preflight_timeout_ms=100,
        context_fast_preflight_soft_wait_ms=10,
    )
    manager.get_cached_memory_packets.side_effect = get_cached_packets
    manager.fast_recall_fallback = AsyncMock(side_effect=slow_preflight)
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint=exact_topic,
        project_path=str(project_dir),
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_file_cache_rescue": 1},
    }
    assert (
        payload["cached_packets"][0]["summary"]
        == "Cached stable project packet for qxjv norel zaffron plinket."
    )
    assert stable_lookup_sync_flags == [False, True]
    stage_timings = payload["diagnostics"]["stage_timings_ms"]
    assert stage_timings["project_file_fallback"] == 0.0
    assert stage_timings["project_file_fallback_soft_wait"] == 0.0
    assert stage_timings["project_file_fallback_pending"] == 1.0
    manager.fast_recall_fallback.assert_not_awaited()
    manager.get_context.assert_not_awaited()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.cache_hit is True

    await asyncio.sleep(0.08)
    exact_cache_calls = [
        call
        for call in manager.cache_memory_packets.call_args_list
        if call.kwargs.get("scope") == "project_home"
        and call.kwargs.get("topic_hint") == exact_topic
        and call.kwargs.get("project_path") == str(project_dir)
    ]
    assert exact_cache_calls
    assert exact_cache_calls[0].kwargs["packets"][0]["summary"] == "Fresh project scan packet."


@pytest.mark.asyncio
async def test_context_surface_returns_pending_project_packet_when_cold_scan_runs(
    monkeypatch,
    tmp_path,
) -> None:
    project_dir = tmp_path / "Engram"
    project_dir.mkdir()
    (project_dir / "README.md").write_text("# Engram\n", encoding="utf-8")
    exact_topic = "cold start first turn project scan still warming"
    fresh_packet = {
        "packet_type": "project_home",
        "title": "Project File: README.md",
        "summary": "Fresh project scan packet after background completion.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:README.md"],
        "_cache_scope": "project_file_fallback",
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }

    async def fake_run_project_file_executor(function, /, *args, **kwargs):
        assert function.__name__ == "_build_project_file_context_fallback_packets"
        await asyncio.sleep(0.06)
        return [fresh_packet], 60.0

    monkeypatch.setattr(
        context_builder_module,
        "_run_project_file_executor",
        fake_run_project_file_executor,
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_packet_cache_enabled=True,
        context_fast_preflight_timeout_ms=100,
        context_fast_preflight_soft_wait_ms=10,
    )
    manager.get_cached_memory_packets.return_value = None
    manager.fast_recall_fallback = AsyncMock(return_value=[])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint=exact_topic,
        project_path=str(project_dir),
        operation_source="axi_context",
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 1,
        "scopes": {"project_file_pending": 1},
    }
    assert payload["cached_packets"][0]["title"] == "Project Context Warming: Engram"
    assert payload["cached_packets"][0]["trust"]["freshness"] == "pending"
    assert payload["budget"]["budget_miss"] is False
    stage_timings = payload["diagnostics"]["stage_timings_ms"]
    assert stage_timings["project_file_fallback_pending"] == 1.0
    manager.get_context.assert_not_awaited()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.cache_hit is False

    await asyncio.sleep(0.08)
    exact_cache_calls = [
        call
        for call in manager.cache_memory_packets.call_args_list
        if call.kwargs.get("scope") == "project_home"
        and call.kwargs.get("topic_hint") == exact_topic
        and call.kwargs.get("project_path") == str(project_dir)
    ]
    assert exact_cache_calls
    assert (
        exact_cache_calls[0].kwargs["packets"][0]["summary"]
        == "Fresh project scan packet after background completion."
    )


@pytest.mark.asyncio
async def test_axi_context_waits_briefly_for_fresh_project_scan_before_cache_rescue(
    monkeypatch,
    tmp_path,
) -> None:
    project_dir = tmp_path / "Engram"
    project_dir.mkdir()
    exact_topic = "fresh observations starved by project-home cache recency"
    cached_packet = {
        "packet_type": "project_home",
        "title": "Project File: README.md",
        "summary": "Stale stable project packet.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:README.md"],
        "_project_file_fallback_topic_hint": "older dogfood topic",
        "_project_file_fallback_project_path": str(project_dir),
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }
    fresh_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/CURRENT_HANDOFF.md",
        "summary": "Fresh scan found the recall-priority evidence.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:docs/CURRENT_HANDOFF.md"],
        "_cache_scope": "project_file_fallback",
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }

    async def fake_run_project_file_executor(function, /, *args, **kwargs):
        assert function.__name__ == "_build_project_file_context_fallback_packets"
        await asyncio.sleep(0.02)
        return [fresh_packet], 20.0

    def get_cached_packets(
        _group_id,
        *,
        scope,
        topic_hint=None,
        project_path=None,
        **_kwargs,
    ):
        if scope == "project_home" and topic_hint == "Engram" and project_path == str(project_dir):
            return SimpleNamespace(packets=[cached_packet])
        return None

    monkeypatch.setattr(
        context_builder_module,
        "_run_project_file_executor",
        fake_run_project_file_executor,
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_packet_cache_enabled=True,
        context_fast_preflight_timeout_ms=100,
        context_fast_preflight_soft_wait_ms=80,
    )
    manager.get_cached_memory_packets.side_effect = get_cached_packets
    manager.fast_recall_fallback = AsyncMock(return_value=[])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint=exact_topic,
        project_path=str(project_dir),
        operation_source="axi_context",
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 1,
        "scopes": {"project_file_fallback": 1},
    }
    assert (
        payload["cached_packets"][0]["summary"] == "Fresh scan found the recall-priority evidence."
    )
    assert "Stale stable project packet" not in payload["context"]
    assert payload["diagnostics"]["stage_timings_ms"]["project_file_fallback_soft_wait"] >= 0
    manager.get_context.assert_not_awaited()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.cache_hit is False


@pytest.mark.asyncio
async def test_context_cache_rescue_ignores_stable_packets_that_miss_topic(
    monkeypatch,
    tmp_path,
) -> None:
    project_dir = tmp_path / "Engram"
    project_dir.mkdir()
    exact_topic = "soft wait current handoff exact evidence 20260528"
    cached_packet = {
        "packet_type": "project_home",
        "title": "Project File: README.md",
        "summary": "Unrelated stale stable project packet.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:README.md"],
        "_project_file_fallback_topic_hint": "older dogfood topic",
        "_project_file_fallback_project_path": str(project_dir),
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }
    fresh_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/CURRENT_HANDOFF.md",
        "summary": "Fresh exact soft wait current handoff evidence.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:docs/CURRENT_HANDOFF.md"],
        "_cache_scope": "project_file_fallback",
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }

    async def fake_run_project_file_executor(function, /, *args, **kwargs):
        assert function.__name__ == "_build_project_file_context_fallback_packets"
        await asyncio.sleep(0.03)
        return [fresh_packet], 30.0

    def get_cached_packets(
        _group_id,
        *,
        scope,
        topic_hint=None,
        project_path=None,
        **_kwargs,
    ):
        if scope == "project_home" and topic_hint == "Engram" and project_path == str(project_dir):
            return SimpleNamespace(packets=[cached_packet])
        return None

    monkeypatch.setattr(
        context_builder_module,
        "_run_project_file_executor",
        fake_run_project_file_executor,
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_packet_cache_enabled=True,
        context_fast_preflight_timeout_ms=100,
        context_fast_preflight_soft_wait_ms=50,
    )
    manager.get_cached_memory_packets.side_effect = get_cached_packets
    manager.fast_recall_fallback = AsyncMock(return_value=[])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint=exact_topic,
        project_path=str(project_dir),
        operation_source="axi_context",
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 1,
        "scopes": {"project_file_fallback": 1},
    }
    assert payload["cached_packets"][0]["summary"] == fresh_packet["summary"]
    assert "Unrelated stale stable project packet" not in payload["context"]
    manager.get_context.assert_not_awaited()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.cache_hit is False


@pytest.mark.asyncio
async def test_mcp_context_surface_reuses_exact_project_file_fallback_cache(
    tmp_path,
) -> None:
    project_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/memory-value-latency-plan.md",
        "summary": "Generic project-file fallback packet.",
        "trust": {"source": "project_file", "freshness": "local"},
        "_project_file_fallback_topic_hint": "packet cache recall latency",
        "_project_file_fallback_project_path": str(tmp_path),
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }

    def get_cached_packets(
        _group_id,
        *,
        scope,
        topic_hint=None,
        project_path=None,
        **_kwargs,
    ):
        if (
            scope == "project_home"
            and topic_hint == "packet cache recall latency"
            and project_path == str(tmp_path)
        ):
            return SimpleNamespace(packets=[project_packet])
        return None

    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_packet_cache_enabled=True,
    )
    manager.get_cached_memory_packets.side_effect = get_cached_packets
    manager.cache_memory_packets = MagicMock()
    manager.fast_recall_fallback = AsyncMock(return_value=[])
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="packet cache recall latency",
        project_path=str(tmp_path),
    )

    assert payload["status"] == "ok"
    assert payload["budget"]["skip_reason"] is None
    assert payload["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_home": 1},
    }
    assert "Project File: docs/memory-value-latency-plan.md" in payload["context"]
    manager.fast_recall_fallback.assert_not_awaited()
    manager.get_context.assert_not_awaited()
    manager.cache_memory_packets.assert_not_called()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.cache_hit is True
    assert sample.packet_count == 1


@pytest.mark.asyncio
async def test_mcp_context_surface_uses_loaded_store_preflight_before_project_files(
    tmp_path,
) -> None:
    (tmp_path / "README.md").write_text("# Engram\nGeneral project overview.")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        "- Project-file fallback should not beat loaded-store context.\n"
    )
    loaded_store_result = {
        "result_type": "cue_episode",
        "cue": {
            "episode_id": "ep_context",
            "cue_text": (
                "mentions: did you get stuck Engram dogfood performance status broad human update"
            ),
            "supporting_spans": [],
            "projection_state": "projected",
        },
        "episode": {
            "id": "ep_context",
            "source": "codex",
            "created_at": "2026-05-27T09:40:42",
        },
        "score": 1.0,
        "score_breakdown": {"semantic": 1.0},
    }
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_fast_preflight_timeout_ms=200,
        recall_packet_cache_enabled=True,
    )
    manager.get_cached_memory_packets.return_value = None
    manager.fast_recall_fallback = AsyncMock(return_value=[loaded_store_result])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="did you get stuck Engram dogfood performance status broad human update",
        project_path=str(tmp_path),
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 1,
        "scopes": {"loaded_store_context": 1},
    }
    assert payload["cached_packets"][0]["trust"]["source"] == "cue"
    assert payload["cached_packets"][0]["episode_ids"] == ["ep_context"]
    assert "Project File:" not in payload["context"]
    assert "Engram dogfood performance status" in payload["context"]
    assert payload["diagnostics"]["stage_timings_ms"]["loaded_store_context_preflight"] >= 0
    assert payload["diagnostics"]["stage_timings_ms"]["loaded_store_context_search"] >= 0
    assert payload["diagnostics"]["stage_timings_ms"]["loaded_store_context_packet_assembly"] >= 0
    manager.get_context.assert_not_awaited()
    manager.fast_recall_fallback.assert_awaited_once()
    cache_keys = {
        (call.kwargs.get("scope"), call.kwargs.get("topic_hint"), call.kwargs.get("project_path"))
        for call in manager.cache_memory_packets.call_args_list
    }
    assert (
        "project_home",
        "did you get stuck Engram dogfood performance status broad human update",
        str(tmp_path),
    ) in cache_keys
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "context"
    assert sample.cache_hit is False
    assert sample.packet_count == 1


@pytest.mark.asyncio
async def test_axi_project_context_uses_project_file_fallback_without_topic(
    tmp_path,
) -> None:
    (tmp_path / "README.md").write_text(
        "# Engram\nAXI project context should avoid deep project-neighbor expansion.",
        encoding="utf-8",
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_fast_preflight_timeout_ms=200,
        recall_packet_cache_enabled=True,
    )
    manager.get_cached_memory_packets.return_value = None
    manager.fast_recall_fallback = AsyncMock(return_value=[])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint=None,
        project_path=str(tmp_path),
        operation_source="axi_context",
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 1,
        "scopes": {"project_file_fallback": 1},
    }
    assert payload["cached_packets"][0]["title"] == "Project File: README.md"
    assert "Project File: README.md" in payload["context"]
    assert "project_file_fallback" in payload["diagnostics"]["stage_timings_ms"]
    manager.fast_recall_fallback.assert_not_awaited()
    manager.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_axi_project_context_with_topic_skips_loaded_store_preflight(
    tmp_path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (tmp_path / "README.md").write_text(
        "# Engram\nAXI context is the lightweight startup surface.",
        encoding="utf-8",
    )
    (docs_dir / "axi-interface-plan.md").write_text(
        "# AXI Interface Plan\nTopic-specific AXI context should stay startup-safe.",
        encoding="utf-8",
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(
        recall_fast_preflight_timeout_ms=200,
        recall_packet_cache_enabled=True,
    )
    manager.get_cached_memory_packets.return_value = None
    manager.fast_recall_fallback = AsyncMock(return_value=[])
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="topic-specific AXI context startup-safe",
        project_path=str(tmp_path),
        operation_source="axi_context",
    )

    assert payload["status"] == "ok"
    assert payload["packet_cache"]["scopes"] == {"project_file_fallback": 2}
    assert "project_file_fallback" in payload["diagnostics"]["stage_timings_ms"]
    assert "loaded_store_context_preflight" not in payload["diagnostics"]["stage_timings_ms"]
    manager.fast_recall_fallback.assert_not_awaited()
    manager.get_context.assert_not_awaited()


def test_project_file_fallback_bounds_topic_candidate_reads(monkeypatch, tmp_path) -> None:
    (tmp_path / "README.md").write_text("# Engram\nGeneral overview.")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for index in range(45):
        (docs_dir / f"generic-{index:02d}.md").write_text(
            f"# Generic {index}\nGeneral unrelated filler.\n"
        )
    target = docs_dir / "zz-memory-value-latency-plan.md"
    target.write_text(
        "# Memory Value and Latency Plan\n\n"
        "- Packet cache fallback should stay useful under cold context calls.\n"
    )

    import engram.retrieval.context_builder as context_builder

    real_read = context_builder._read_project_file_prefix
    read_paths: list[str] = []

    def tracking_read(path, char_limit):
        read_paths.append(path.relative_to(tmp_path).as_posix())
        return real_read(path, char_limit)

    monkeypatch.setattr(context_builder, "_read_project_file_prefix", tracking_read)

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="memory latency packet cache fallback",
        max_packets=1,
        reason="test",
    )

    assert packets[0]["title"] == "Project File: docs/zz-memory-value-latency-plan.md"
    assert "docs/zz-memory-value-latency-plan.md" in read_paths
    assert len(read_paths) <= context_builder._PROJECT_FILE_TOPIC_CANDIDATE_READ_LIMIT
    assert len(read_paths) < 47


def test_project_file_fallback_reuses_topic_matching_lines(monkeypatch, tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "# Engram\nPacket cache fallback should avoid duplicate content scans.\n",
        encoding="utf-8",
    )

    import engram.retrieval.context_builder as context_builder

    real_matching_lines = context_builder._project_file_matching_lines
    call_count = 0

    def tracking_matching_lines(content, *, topic_hint, limit):
        nonlocal call_count
        call_count += 1
        return real_matching_lines(content, topic_hint=topic_hint, limit=limit)

    monkeypatch.setattr(
        context_builder,
        "_project_file_matching_lines",
        tracking_matching_lines,
    )

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="packet cache fallback duplicate content scans",
        max_packets=1,
        reason="test",
        max_candidates=1,
        candidate_read_limit=1,
    )

    assert packets
    assert call_count == 1


def test_project_file_fallback_prefers_latest_equally_relevant_lines(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "CURRENT_HANDOFF.md").write_text(
        "# Current Handoff\n\n"
        "- startup matrix at `/private/tmp/engram-dogfood-startup-20260527-112804` passed.\n"
        + ("- unrelated historical note.\n" * 20)
        + "- startup matrix at `/private/tmp/engram-dogfood-startup-20260527-140202` passed.\n"
    )

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="startup matrix 20260527",
        max_packets=1,
        reason="test",
    )

    assert packets[0]["title"] == "Project File: docs/CURRENT_HANDOFF.md"
    assert "20260527-140202" in packets[0]["summary"]
    assert packets[0]["evidence_lines"][0].endswith("20260527-140202` passed.")


def test_project_file_fallback_prefers_current_handoff_over_historical_logs(
    tmp_path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "dogfood-startup-validation-goal.md").write_text(
        "# Dogfood Startup Validation Goal\n\n"
        + (
            "- startup matrix at `/private/tmp/engram-dogfood-startup-20260527-112804` passed.\n"
            * 12
        )
    )
    (docs_dir / "CURRENT_HANDOFF.md").write_text(
        "# Current Handoff\n\n"
        + ("- unrelated current-state note.\n" * 40)
        + "- startup matrix at `/private/tmp/engram-dogfood-startup-20260527-140202` passed.\n"
    )

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="startup matrix 20260527",
        max_packets=1,
        reason="test",
    )

    assert packets[0]["title"] == "Project File: docs/CURRENT_HANDOFF.md"
    assert "20260527-140202" in packets[0]["summary"]


def test_project_file_fallback_reads_wrapped_current_evidence(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "CURRENT_HANDOFF.md").write_text(
        "# Current Handoff\n\n"
        "- unrelated previous line.\n"
        "- The refreshed full lifecycle matrix produced\n"
        "`/private/tmp/engram-dogfood-startup-20260527-140202` with "
        "`11 pass, 2 warn, 0 fail, 0 skip`.\n"
    )

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="startup matrix 20260527",
        max_packets=1,
        reason="test",
    )

    assert packets[0]["title"] == "Project File: docs/CURRENT_HANDOFF.md"
    assert "20260527-140202" in packets[0]["summary"]
    assert "unrelated previous line" not in packets[0]["summary"]


def test_project_file_fallback_joins_lowercase_wrapped_evidence(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "CURRENT_HANDOFF.md").write_text(
        "# Current Handoff\n\n"
        "AXI recall for `startup matrix 20260527 tiecheck diamond` rebuilt current Engram\n"
        "evidence in `1249.1747ms` with `project_file_recall_fallback`.\n"
    )

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="startup matrix 20260527 tiecheck diamond project_file_recall_fallback",
        max_packets=1,
        reason="test",
    )

    assert packets[0]["title"] == "Project File: docs/CURRENT_HANDOFF.md"
    assert "startup matrix 20260527 tiecheck diamond" in packets[0]["summary"]
    assert "evidence in `1249.1747ms`" in packets[0]["summary"]


def test_project_file_fallback_uses_wrapped_window_for_evidence_lines(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "CURRENT_HANDOFF.md").write_text(
        "# Current Handoff\n\n"
        "AXI recall for `startup matrix 20260527 tiecheck diamond` rebuilt current Engram\n"
        "evidence in `1249.1747ms` with `project_file_recall_fallback`.\n"
    )

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="evidence project_file_recall_fallback",
        max_packets=1,
        reason="test",
    )

    assert packets[0]["title"] == "Project File: docs/CURRENT_HANDOFF.md"
    assert packets[0]["evidence_lines"][0].startswith("AXI recall for")
    assert "evidence in `1249.1747ms`" in packets[0]["evidence_lines"][0]


def test_project_file_fallback_uses_full_chained_window_for_evidence_lines(
    tmp_path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "CURRENT_HANDOFF.md").write_text(
        "# Current Handoff\n\n"
        "`startup matrix 20260527 tiecheck diamond` rebuilt current Engram project-file\n"
        "evidence in `1249.1747ms` with `project_file_recall_fallback`; AXI context for\n"
        "`native PyO3 dogfood performance continuation cleanline 20260527` rebuilt clean\n"
        "handoff evidence in `931.882ms`.\n"
    )

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="evidence project_file_recall_fallback native handoff 20260527",
        max_packets=1,
        reason="test",
    )

    assert packets[0]["title"] == "Project File: docs/CURRENT_HANDOFF.md"
    assert packets[0]["evidence_lines"][0].startswith("`startup matrix 20260527 tiecheck diamond`")
    assert "project_file_recall_fallback" in packets[0]["evidence_lines"][0]
    assert not packets[0]["evidence_lines"][0].startswith("handoff evidence")


def test_project_file_fallback_trims_prior_sentence_from_wrapped_window(
    tmp_path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "CURRENT_HANDOFF.md").write_text(
        "# Current Handoff\n\n"
        "Unrelated cache sentence should not lead. After reinstall, AXI recall for\n"
        "`startup matrix 20260527 tiecheck diamond` rebuilt current Engram project-file\n"
        "evidence in `1249.1747ms` with `project_file_recall_fallback`.\n"
    )

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="startup matrix 20260527 project_file_recall_fallback",
        max_packets=1,
        reason="test",
    )

    assert packets[0]["title"] == "Project File: docs/CURRENT_HANDOFF.md"
    assert packets[0]["evidence_lines"][0].startswith("After reinstall, AXI recall")
    assert "Unrelated cache sentence" not in packets[0]["evidence_lines"][0]


def test_project_file_fallback_truncates_summary_on_word_boundary(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "CURRENT_HANDOFF.md").write_text(
        "# Current Handoff\n\n"
        "`startup matrix 20260527 tiecheck diamond` rebuilt current Engram "
        "project-file evidence in `1249.1747ms` with "
        "`project_file_recall_fallback`; AXI context for `native PyO3 dogfood "
        "performance continuation cleanline 20260527` rebuilt clean handoff "
        "evidence in `931.882ms` with enough extra text to require truncation.\n"
    )

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="startup matrix 20260527 tiecheck diamond project_file_recall_fallback",
        max_packets=1,
        reason="test",
    )

    assert packets[0]["title"] == "Project File: docs/CURRENT_HANDOFF.md"
    assert packets[0]["summary"].endswith("...")
    assert "dogfood p" not in packets[0]["summary"]


def test_project_file_fallback_does_not_join_unrelated_previous_line(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "CURRENT_HANDOFF.md").write_text(
        "# Current Handoff\n\n"
        "- Helix stats routes for evaluation graph-state refresh.\n"
        "- Latest dogfood performance note: the native PyO3 path now uses "
        "generated bulk routes.\n"
    )

    packets = _project_file_fallback_packets(
        project_path=str(tmp_path),
        topic_hint="native PyO3 dogfood performance",
        max_packets=1,
        reason="test",
    )

    assert packets[0]["title"] == "Project File: docs/CURRENT_HANDOFF.md"
    assert "Latest dogfood performance note" in packets[0]["summary"]
    assert "Helix stats routes" not in packets[0]["summary"]


def test_project_file_fallback_public_wrapper_uses_bounded_topic_scan(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "README.md").write_text("# Engram\nGeneral overview.")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        + ("General filler before the relevant late evidence.\n" * 80)
        + "- Packet cache fallback should keep context calls bounded.\n"
    )

    import engram.retrieval.context_builder as context_builder

    real_read = context_builder._read_project_file_prefix
    read_limits: list[int] = []

    def tracking_read(path, char_limit):
        read_limits.append(char_limit)
        return real_read(path, char_limit)

    monkeypatch.setattr(context_builder, "_read_project_file_prefix", tracking_read)
    manager = SimpleNamespace(
        get_activation_config=lambda: ActivationConfig(recall_packet_cache_enabled=False)
    )

    packets = project_file_fallback_packet_payloads(
        manager,
        group_id="native_brain",
        topic_hint="memory latency packet cache fallback",
        project_path=str(tmp_path),
        max_packets=1,
        reason="test",
        candidate_read_limit=1,
    )

    assert packets[0]["title"] == "Project File: docs/memory-value-latency-plan.md"
    assert read_limits
    assert len(read_limits) == 1
    assert max(read_limits) <= context_builder._PROJECT_FILE_TOPIC_SCAN_CHARS


def test_project_file_prefix_reads_reuse_unchanged_cached_content(
    monkeypatch,
    tmp_path,
) -> None:
    project_file = tmp_path / "README.md"
    project_file.write_text("# Engram\nCached project context fallback.\n")
    _PROJECT_FILE_PREFIX_CACHE.clear()

    assert _read_project_file_prefix(project_file, 64).startswith("# Engram")

    def fail_open(*_args, **_kwargs):
        raise AssertionError("cached prefix should not reopen unchanged file")

    monkeypatch.setattr(type(project_file), "open", fail_open)

    assert _read_project_file_prefix(project_file, 16) == "# Engram\nCached "
    _PROJECT_FILE_PREFIX_CACHE.clear()


def test_context_preflight_timeout_uses_context_specific_budget() -> None:
    import engram.retrieval.context_builder as context_builder

    cfg = ActivationConfig(
        recall_fast_preflight_timeout_ms=250,
        context_fast_preflight_timeout_ms=75,
        context_fast_preflight_soft_wait_ms=50,
    )

    assert context_builder._context_fast_preflight_timeout_seconds(cfg) == 0.075
    assert context_builder._context_fast_preflight_soft_wait_seconds(cfg) == 0.05


def test_context_packet_relevance_ignores_generated_why_now() -> None:
    import engram.retrieval.context_builder as context_builder

    packet = {
        "packet_type": "episode_packet",
        "summary": "Engram dogfood performance continuation.",
        "why_now": ("Relevant to the recall query: xqzvplm brontide nonesuch cymophane vellichor"),
        "evidence_lines": ["Trace fallback latency was fixed."],
    }

    assert not context_builder._context_cached_packets_relevant(
        [packet],
        topic_hint="xqzvplm brontide nonesuch cymophane vellichor",
        project_path="/tmp/Engram",
    )
    assert context_builder._context_cached_packets_relevant(
        [packet],
        topic_hint="trace fallback latency",
        project_path="/tmp/Engram",
    )


def test_context_packet_relevance_rejects_lone_date_match() -> None:
    import engram.retrieval.context_builder as context_builder

    packet = {
        "packet_type": "cue_packet",
        "summary": "Dogfood rolling session proof 20260527 orchid.",
        "evidence_lines": ["Older session packet should remain recallable."],
    }

    assert not context_builder._context_cached_packets_relevant(
        [packet],
        topic_hint="qvanta noexisting loadedstore miss tail 20260527 probeA",
        project_path="/tmp/Engram",
    )
    assert context_builder._context_cached_packets_relevant(
        [packet],
        topic_hint="dogfood rolling 20260527",
        project_path="/tmp/Engram",
    )


@pytest.mark.asyncio
async def test_mcp_context_surface_degrades_on_timeout() -> None:
    async def slow_get_context(**_kwargs):
        await asyncio.sleep(0.15)
        return CONTEXT_RESULT

    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(recall_budget_explicit_ms=100)
    manager.get_context = AsyncMock(side_effect=slow_get_context)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="Engram",
        operation_source="mcp_context",
    )

    assert payload["status"] == "degraded"
    assert payload["entity_count"] == 0
    assert payload["budget"]["skip_reason"] == "context_timeout"
    assert payload["budget"]["timeout"] is True
    assert payload["lifecycle"]["degraded"] is True
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "context"
    assert sample.source == "mcp_context"
    assert sample.timeout is True


@pytest.mark.asyncio
async def test_mcp_context_surface_returns_cached_packets_after_timeout() -> None:
    async def slow_get_context(**_kwargs):
        await asyncio.sleep(0.15)
        return CONTEXT_RESULT

    cached_packet = {
        "packet_type": "project_home",
        "title": "Project Home: Engram",
        "summary": "Cached packet survives context timeout.",
        "trust": {"source": "cache", "freshness": "fresh"},
    }
    lookup_count = 0

    def cached_packets(_group_id: str, *, scope: str, **_kwargs):
        nonlocal lookup_count
        lookup_count += 1
        if lookup_count <= 5:
            return None
        if scope == "project_home":
            return SimpleNamespace(packets=[cached_packet])
        return None

    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(recall_budget_explicit_ms=100)
    manager.get_cached_memory_packets.side_effect = cached_packets
    manager.get_context = AsyncMock(side_effect=slow_get_context)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        project_path="/Users/konnermoshier/Engram",
        operation_source="mcp_context",
    )

    assert payload["status"] == "degraded"
    assert payload["budget"]["skip_reason"] == "context_timeout"
    assert payload["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_home": 1},
    }
    assert "Cached packet survives context timeout" in payload["context"]
    assert payload["cached_packets"][0]["trust"]["source"] == "cache"


@pytest.mark.asyncio
async def test_mcp_context_surface_uses_project_files_after_timeout_when_cache_cold(
    tmp_path,
) -> None:
    async def slow_get_context(**_kwargs):
        await asyncio.sleep(0.15)
        return CONTEXT_RESULT

    (tmp_path / "README.md").write_text("# Engram\nGeneral project overview.")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        "- Packet cache should keep context useful when recall latency is high.\n"
    )
    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(recall_budget_explicit_ms=100)
    manager.get_cached_memory_packets.return_value = None
    manager.cache_memory_packets = MagicMock()
    manager.get_context = AsyncMock(side_effect=slow_get_context)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint=None,
        project_path=str(tmp_path),
        operation_source="mcp_context",
    )

    assert payload["status"] == "degraded"
    assert payload["budget"]["skip_reason"] == "context_timeout"
    assert payload["packet_cache"] == {
        "hit": False,
        "packet_count": 2,
        "scopes": {"project_file_fallback": 2},
    }
    assert "Project File: docs/memory-value-latency-plan.md" in payload["context"]
    assert "Memory Value and Latency Plan" in payload["context"]
    assert "Graph context timed out" in payload["context"]
    assert "context_timeout" in payload["diagnostics"]["stage_timings_ms"]
    assert payload["diagnostics"]["stage_timings_ms"]["project_file_fallback"] >= 0
    assert payload["budget"]["duration_ms"] >= 100
    assert payload["cached_packets"][0]["trust"]["source"] == "project_file"
    assert manager.cache_memory_packets.called
    cache_keys = {
        (call.kwargs.get("scope"), call.kwargs.get("topic_hint"), call.kwargs.get("project_path"))
        for call in manager.cache_memory_packets.call_args_list
    }
    assert ("project_home", tmp_path.name, str(tmp_path)) in cache_keys
    assert all(
        call.kwargs.get("persist") is True for call in manager.cache_memory_packets.call_args_list
    )


@pytest.mark.asyncio
async def test_mcp_context_tool_surface_runs_middleware_with_context_hint() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    recall_middleware = AsyncMock()

    payload = await build_mcp_context_tool_surface(
        manager,
        group_id="native_brain",
        max_tokens=900,
        topic_hint=None,
        project_path="/tmp/engram",
        format="briefing",
        recall_middleware=recall_middleware,
    )

    assert payload == CONTEXT_RESULT
    recall_middleware.assert_awaited_once_with(
        "/tmp/engram",
        payload,
        tool_name="get_context",
    )


@pytest.mark.asyncio
async def test_memory_context_builder_returns_when_topic_recall_times_out() -> None:
    graph = MagicMock()
    graph.find_entities = AsyncMock(return_value=[])
    activation = MemoryActivationStore(cfg=ActivationConfig())

    async def slow_recall(**_kwargs):
        import asyncio

        await asyncio.sleep(1)
        return []

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=ActivationConfig(),
        recall=slow_recall,
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )
    builder._CONTEXT_RECALL_TIMEOUT_SECONDS = 0.01

    result = await builder.get_context(group_id="brain", topic_hint="Engram")

    assert result["format"] == "structured"
    assert result["entity_count"] == 0
    assert "## Recent Activity" in result["context"]


def test_memory_context_builder_redacts_secret_shaped_context_text() -> None:
    rendered = MemoryContextBuilder.render_tier(
        "## Project Context",
        [
            {
                "name": "SQLite",
                "type": "Technology",
                "detail_level": "full",
                "activation": 0.0,
                "summary": "ANTHROPIC_API_KEY=sk-ant-your-key-here",
                "attributes": {"token": "ghp_abc123456789"},
                "facts": ["SQLite USES token=sk-secret-value"],
            }
        ],
        [],
    )

    assert "ANTHROPIC_API_KEY=[redacted]" in rendered
    assert "token: [redacted]" in rendered
    assert "token=[redacted]" in rendered
    assert "sk-ant-your-key-here" not in rendered
    assert "ghp_abc123456789" not in rendered


@pytest.mark.asyncio
async def test_memory_context_builder_uses_budget_to_cap_project_expansion() -> None:
    graph = MagicMock()
    activation = MagicMock()
    cfg = ActivationConfig()
    cfg.identity_core_enabled = False
    cfg.briefing_enabled = False

    project = Entity(id="ent_project", name="Engram", entity_type="Project", group_id="brain")
    neighbors = [
        (
            Entity(
                id=f"ent_neighbor_{index}",
                name=f"Neighbor {index}",
                entity_type="Artifact",
                summary="Project artifact",
                group_id="brain",
            ),
            MagicMock(),
        )
        for index in range(8)
    ]
    graph.find_entities = AsyncMock(return_value=[project])
    graph.get_neighbors = AsyncMock(return_value=neighbors)
    graph.get_relationships = AsyncMock(return_value=[])
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.get_top_activated = AsyncMock(return_value=[])
    recall = AsyncMock(return_value=[])

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=cfg,
        recall=recall,
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )

    result = await builder.get_context(
        group_id="brain",
        project_path="/Users/konnermoshier/Engram",
        max_tokens=200,
    )

    recall.assert_awaited_once_with(query="Engram", group_id="brain", limit=1)
    activation.get_top_activated.assert_awaited_once_with(group_id="brain", limit=2)
    assert "Neighbor 0" in result["context"]
    assert "Neighbor 1" in result["context"]
    assert "Neighbor 2" not in result["context"]


@pytest.mark.asyncio
async def test_memory_context_builder_includes_cached_project_packets() -> None:
    graph = MagicMock()
    graph.find_entities = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.get_entity = AsyncMock(return_value=None)
    graph.get_relationships = AsyncMock(return_value=[])
    activation = MagicMock()
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.get_top_activated = AsyncMock(return_value=[])
    cache_packets = MagicMock()
    recall = AsyncMock(
        return_value=[
            {
                "entity": {
                    "id": "ent_engram",
                    "name": "Engram",
                    "type": "Project",
                    "summary": "Memory runtime",
                },
                "score_breakdown": {},
            }
        ]
    )

    def get_cached_packets(_group_id: str, *, scope: str, **_kwargs):
        if scope == "project_home":
            return SimpleNamespace(
                packets=[
                    {
                        "packet_type": "project_home",
                        "title": "Project Home: Engram",
                        "summary": "Cached Engram project packet.",
                        "why_now": "Project startup context.",
                        "trust": {
                            "source": "cache",
                            "freshness": "recent",
                            "why_now": "Project startup context.",
                        },
                    }
                ]
            )
        return None

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=ActivationConfig(identity_core_enabled=False),
        recall=recall,
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
        get_cached_packets=get_cached_packets,
        cache_packets=cache_packets,
    )

    result = await builder.get_context(
        group_id="brain",
        project_path="/Users/konnermoshier/Engram",
        max_tokens=800,
    )

    assert "## Cached Memory Packets" in result["context"]
    assert "Project Home: Engram" in result["context"]
    assert result["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_home": 1},
    }
    assert result["cached_packets"][0]["trust"]["source"] == "cache"
    assert cache_packets.called


@pytest.mark.asyncio
async def test_memory_context_builder_uses_project_artifacts_when_recall_misses() -> None:
    graph = MagicMock()
    project = Entity(id="ent_project", name="Engram", entity_type="Project", group_id="brain")
    readme = Entity(
        id="art_readme",
        name="README.md",
        entity_type="Artifact",
        summary="Engram artifact README.md",
        attributes={
            "project_path": "/Users/konnermoshier/Engram",
            "rel_path": "README.md",
            "snippet": "General Engram project overview.",
            "claims": [],
        },
        group_id="brain",
    )
    artifact = Entity(
        id="art_latency_plan",
        name="docs/memory-value-latency-plan.md",
        entity_type="Artifact",
        summary="Engram artifact docs/memory-value-latency-plan.md",
        attributes={
            "project_path": "/Users/konnermoshier/Engram",
            "rel_path": "docs/memory-value-latency-plan.md",
            "snippet": "Packet cache and recall latency plan for native PyO3 dogfood.",
            "claims": [
                {
                    "predicate": "heading",
                    "object": "Memory Value and Latency Plan",
                }
            ],
        },
        group_id="brain",
    )

    async def find_entities(**kwargs):
        if kwargs.get("entity_type") == "Project":
            return [project]
        if kwargs.get("entity_type") == "Artifact":
            return [readme, artifact]
        return []

    graph.find_entities = AsyncMock(side_effect=find_entities)
    graph.get_neighbors = AsyncMock(return_value=[])
    graph.get_relationships = AsyncMock(return_value=[])
    activation = MagicMock()
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.get_top_activated = AsyncMock(return_value=[])
    cache_packets = MagicMock()

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=ActivationConfig(identity_core_enabled=False),
        recall=AsyncMock(return_value=[]),
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
        get_cached_packets=MagicMock(return_value=None),
        cache_packets=cache_packets,
    )

    result = await builder.get_context(
        group_id="brain",
        topic_hint="Engram native PyO3 loaded-store recall latency packet cache",
        project_path="/Users/konnermoshier/Engram",
        max_tokens=800,
    )

    assert "docs/memory-value-latency-plan.md" in result["context"]
    assert "Memory Value and Latency Plan" in result["context"]
    assert result["context"].find("docs/memory-value-latency-plan.md") < result["context"].find(
        "README.md"
    )
    assert result["entity_count"] == 2
    assert result["packet_cache"]["hit"] is False
    assert cache_packets.called
    project_cache_writes = [
        call for call in cache_packets.call_args_list if call.kwargs.get("scope") == "project_home"
    ]
    assert project_cache_writes
    assert project_cache_writes[0].kwargs["packets"][0]["entity_ids"] == ["art_latency_plan"]


@pytest.mark.asyncio
async def test_memory_context_builder_falls_back_when_entity_enrichment_is_slow() -> None:
    graph = MagicMock()
    graph.find_entities = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.get_neighbors = AsyncMock(return_value=[])

    async def slow_relationships(*_args, **_kwargs):
        await asyncio.sleep(1)
        return []

    graph.get_relationships = AsyncMock(side_effect=slow_relationships)
    activation = MagicMock()
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.get_top_activated = AsyncMock(return_value=[])
    recall = AsyncMock(
        return_value=[
            {
                "entity": {
                    "id": "ent_runtime",
                    "name": "Engram Runtime",
                    "type": "Project",
                    "summary": "Loaded-store performance work",
                },
                "score_breakdown": {},
            }
        ]
    )

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=ActivationConfig(identity_core_enabled=False),
        recall=recall,
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
        get_cached_packets=MagicMock(return_value=None),
        cache_packets=MagicMock(),
    )

    result = await builder.get_context(
        group_id="brain",
        topic_hint="Engram runtime",
        project_path="/Users/konnermoshier/Engram",
        max_tokens=800,
    )

    assert "Engram Runtime" in result["context"]
    assert "Loaded-store performance work" in result["context"]
    assert result["entity_count"] == 1


@pytest.mark.asyncio
async def test_memory_context_builder_writes_stable_project_cache_for_specific_topic() -> None:
    graph = MagicMock()
    graph.find_entities = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.get_entity = AsyncMock(return_value=None)
    graph.get_relationships = AsyncMock(return_value=[])
    activation = MagicMock()
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.get_top_activated = AsyncMock(return_value=[])
    cache_packets = MagicMock()

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=ActivationConfig(identity_core_enabled=False),
        recall=AsyncMock(
            return_value=[
                {
                    "entity": {
                        "id": "ent_runtime",
                        "name": "Engram Runtime",
                        "type": "Project",
                        "summary": "Dogfood runtime performance work",
                    },
                    "score_breakdown": {},
                }
            ]
        ),
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
        get_cached_packets=MagicMock(return_value=None),
        cache_packets=cache_packets,
    )

    await builder.get_context(
        group_id="brain",
        topic_hint="Engram native PyO3 dogfood performance goal continuation",
        project_path="/Users/konnermoshier/Engram",
        max_tokens=800,
    )

    cache_keys = {
        (call.kwargs.get("scope"), call.kwargs.get("topic_hint"), call.kwargs.get("project_path"))
        for call in cache_packets.call_args_list
    }
    assert (
        "project_home",
        "Engram native PyO3 dogfood performance goal continuation",
        "/Users/konnermoshier/Engram",
    ) in cache_keys
    assert ("project_home", "Engram", "/Users/konnermoshier/Engram") in cache_keys


@pytest.mark.asyncio
async def test_memory_context_builder_skips_project_creation_when_lookup_times_out() -> None:
    graph = MagicMock()
    activation = MagicMock()
    cfg = ActivationConfig()
    cfg.identity_core_enabled = False
    cfg.briefing_enabled = False

    async def slow_find_entities(**_kwargs):
        import asyncio

        await asyncio.sleep(1)
        return []

    graph.find_entities = slow_find_entities
    graph.create_entity = AsyncMock()
    graph.get_neighbors = AsyncMock(return_value=[])
    activation.record_access = AsyncMock()
    activation.get_top_activated = AsyncMock(return_value=[])

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=cfg,
        recall=AsyncMock(return_value=[]),
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )
    builder._CONTEXT_GRAPH_LOOKUP_TIMEOUT_SECONDS = 0.01

    result = await builder.get_context(
        group_id="brain",
        project_path="/Users/konnermoshier/Engram",
        max_tokens=200,
    )

    graph.create_entity.assert_not_awaited()
    assert "## Project Context" not in result["context"]
    assert result["format"] == "structured"
