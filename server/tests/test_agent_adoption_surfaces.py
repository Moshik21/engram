"""Tests for agent adoption surfaces: debt, injection, prompts, budgets, briefing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.axi.surfaces import _ensure_briefing_growth_line, build_home_payload
from engram.config import ActivationConfig
from engram.harness_adoption import priming_instruction_text
from engram.mcp.prompts import ENGRAM_SYSTEM_PROMPT
from engram.retrieval.adoption_debt import (
    adoption_debt_is_actionable,
    build_adoption_debt,
)
from engram.retrieval.auto_recall import (
    apply_mcp_recall_enrichment,
    build_full_auto_recall_surface,
    build_lite_auto_recall_surface,
)
from engram.retrieval.budgets import recall_budget_for_profile
from engram.retrieval.candidate_pool import _graph_pool_timeout_seconds
from engram.retrieval.context_builder import MemoryContextBuilder
from engram.retrieval.runtime_state import (
    _build_agent_adoption_guidance,
    build_fast_runtime_packet,
)


class _InjectionFakeClient:
    server_url = "http://127.0.0.1:8100"
    timeout_seconds = 10.0

    def __init__(self) -> None:
        self.calls: list[str] = []

    def with_timeout(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        return self

    def health(self) -> dict:
        return {"status": "healthy", "mode": "helix"}

    def runtime_fast(self, *, project_path: str | None = None) -> dict:
        return {
            "projectName": "Engram",
            "runtime": {"mode": "helix"},
            "artifactBootstrap": {
                "artifactCount": 3,
                "lastObservedAt": "2026-06-30T00:00:00Z",
            },
            "agentAdoption": {"status": "ready"},
        }

    def storage(self, *, live: bool = False, timeout_seconds: float | None = None) -> dict:
        return {"mode": "helix", "backend": "helix_native", "counts": {"episodes": 2}}

    def context(
        self,
        *,
        max_tokens: int,
        topic_hint: str | None = None,
        project_path: str | None = None,
        format: str = "structured",
    ) -> dict:
        self.calls.append(f"context:{format}")
        return {
            "context": "Memory growth: 4 episodes, 2 cue traces — compounding.",
            "format": format,
        }

    def search_artifacts(
        self,
        query_text: str,
        *,
        project_path: str | None = None,
        limit: int = 5,
    ) -> dict:
        self.calls.append(f"artifacts:{query_text}")
        return {
            "items": [
                {
                    "path": "docs/design/extraction-rework.md",
                    "snippet": "Progressive Projection and Cue-First Memory",
                    "score": 0.9,
                }
            ]
        }


def test_motivation_first_in_system_prompt() -> None:
    motivation_idx = ENGRAM_SYSTEM_PROMPT.index("Why Engram Matters")
    brain_idx = ENGRAM_SYSTEM_PROMPT.index("Brain Loop Contract")
    assert motivation_idx < brain_idx
    assert "You will not know much on day one" in ENGRAM_SYSTEM_PROMPT
    assert "Skipping Engram today steals from future sessions" in ENGRAM_SYSTEM_PROMPT
    assert "I checked project memory" in ENGRAM_SYSTEM_PROMPT


def test_motivation_first_in_priming() -> None:
    text = priming_instruction_text()
    why_idx = text.index("Why this matters")
    before_idx = text.index("Before every substantive answer")
    assert why_idx < before_idx
    assert "savings account" in text
    assert "Liam plays soccer" in text


def test_build_home_payload_injects_briefing_and_artifacts() -> None:
    client = _InjectionFakeClient()
    result = build_home_payload(
        client,
        project_path="/tmp/Engram",
        topic_hint=None,
        budget=800,
    )
    payload = result.payload
    assert payload.get("briefing")
    assert payload.get("artifactHits")
    assert payload["injection"]["status"] == "ok"
    assert "context:briefing" in client.calls
    assert "artifacts:Engram" in client.calls


def test_adoption_debt_from_operation_metrics() -> None:
    debt = build_adoption_debt(
        {
            "operation_counts": {"observe": 47, "context": 1},
            "source_counts": {"api_auto_observe": 47, "mcp_context": 0},
        }
    )
    assert debt["turnsWithoutRecall"] == 47
    assert debt["episodesCaptured"] == 47
    assert debt["agentRecallCount"] == 0
    assert "47 episodes captured, 0 recalled" in debt["consequence"]
    assert adoption_debt_is_actionable(debt)


def test_adoption_debt_cleared_after_context_load() -> None:
    debt = build_adoption_debt(
        {"operation_counts": {"observe": 10}},
        context_loaded_this_session=True,
    )
    assert debt["turnsWithoutRecall"] == 0
    assert not adoption_debt_is_actionable(debt)


def test_agent_adoption_guidance_includes_debt() -> None:
    guidance = _build_agent_adoption_guidance(
        {
            "enabled": True,
            "artifactCount": 3,
            "staleArtifactCount": 0,
            "lastObservedAt": "2026-06-30T00:00:00Z",
        },
        recall_metrics={"total_analyses": 1},
        epistemic_metrics={"route_counts": {"remember": 1}},
        memory_operation_metrics={
            "operation_counts": {"observe": 5},
            "source_counts": {"api_auto_observe": 5},
        },
        project_path="/tmp/Engram",
    )
    assert "adoptionDebt" in guidance
    assert guidance["adoptionDebt"]["episodesCaptured"] == 5


def test_apply_mcp_recall_enrichment_attaches_debt() -> None:
    response: dict = {}
    debt = build_adoption_debt({"operation_counts": {"observe": 3}})
    apply_mcp_recall_enrichment(response, adoption_debt=debt)
    assert response["adoptionDebt"]["episodesCaptured"] == 3


def test_auto_lite_budget_no_longer_uses_aggressive_75ms_defaults() -> None:
    cfg = ActivationConfig()
    budget = recall_budget_for_profile(cfg, "auto_lite", surface="mcp", mode="medium")
    assert budget.max_wall_ms >= 300
    assert budget.max_search_ms >= 150
    assert budget.max_graph_ms >= 100
    timeout = budget.stage_timeout_seconds(budget.max_search_ms)
    assert timeout >= 0.15


def test_template_briefing_includes_growth_stats() -> None:
    builder = MemoryContextBuilder(
        graph_store=MagicMock(),
        activation_store=MagicMock(),
        cfg=ActivationConfig(),
        recall=AsyncMock(return_value=[]),
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )
    briefing = builder.template_briefing(
        "## Cached Memory Packets\n\n- item",
        "default",
        "Engram",
        growth_stats={"episodes": 12, "cues": 8, "promotions": 3},
    )
    assert "Memory growth: 12 episodes, 8 cue traces, 3 promoted to graph" in briefing


def test_graph_pool_timeout_relaxed_for_auto_profiles() -> None:
    cfg = ActivationConfig(
        retrieval_graph_pool_timeout_ms=75,
        retrieval_graph_pool_timeout_auto_ms=250,
    )
    assert _graph_pool_timeout_seconds(cfg) == pytest.approx(0.075)
    assert _graph_pool_timeout_seconds(cfg, budget_profile="auto_lite") == pytest.approx(0.25)
    assert _graph_pool_timeout_seconds(cfg, budget_profile="auto_deep") == pytest.approx(0.25)


def test_fast_runtime_packet_includes_adoption_debt() -> None:
    packet = build_fast_runtime_packet(
        ActivationConfig(),
        runtime_mode="helix",
        project_path="/tmp/Engram",
    )
    assert "adoptionDebt" in packet["agentAdoption"]
    assert "consequence" in packet["agentAdoption"]["adoptionDebt"]


def test_injection_growth_line_from_storage_counts() -> None:
    text = _ensure_briefing_growth_line(
        "## Cached Memory Packets\n\n- item",
        {"episodes": 3, "cues": 2, "entities": 0},
    )
    assert text.startswith("Memory growth: 3 episodes, 2 cue traces")


def _sample_field(sample: object, field: str) -> object | None:
    if isinstance(sample, dict):
        return sample.get(field)
    return getattr(sample, field, None)


@pytest.mark.asyncio
async def test_generate_candidates_auto_profile_graph_stage_evidence() -> None:
    """Real generate_candidates path: auto profile completes graph pool auto timeout would miss."""
    import asyncio
    from unittest.mock import AsyncMock

    from engram.retrieval.candidate_pool import generate_candidates

    graph_delay_seconds = 0.12

    async def slow_neighbors(**_kwargs):
        await asyncio.sleep(graph_delay_seconds)
        return [("n1", 0.8, "RELATED_TO")]

    search_idx = AsyncMock()
    search_idx.search = AsyncMock(return_value=[("e1", 0.9)])
    search_idx.compute_similarity = AsyncMock(return_value={})
    search_idx.search_episodes = AsyncMock(return_value=[])
    search_idx._embeddings_enabled = False

    act_store = AsyncMock()
    act_store.get_top_activated = AsyncMock(return_value=[])
    act_store.batch_get = AsyncMock(return_value={})

    graph = AsyncMock()
    graph.get_active_neighbors_with_weights = AsyncMock(side_effect=slow_neighbors)

    stage_timings: dict[str, float] = {}
    cfg = ActivationConfig(
        multi_pool_enabled=True,
        retrieval_graph_pool_timeout_ms=75,
        retrieval_graph_pool_timeout_auto_ms=250,
    )
    results = await generate_candidates(
        query="Engram harness adoption progressive memory",
        group_id="default",
        search_index=search_idx,
        activation_store=act_store,
        graph_store=graph,
        cfg=cfg,
        stage_timings_ms=stage_timings,
        budget_profile="auto_deep",
    )

    assert any(entity_id == "n1" for entity_id, _score in results)
    assert "recall_graph_pool" in stage_timings
    assert "recall_graph_pool_timeout" not in stage_timings


@pytest.mark.asyncio
async def test_build_lite_auto_recall_surface_executes_without_timeout() -> None:
    gate_samples: list[object] = []

    def record_memory_operation(group_id: str, sample: object) -> None:
        if _sample_field(sample, "operation") == "auto_recall_gate":
            gate_samples.append(sample)

    manager = MagicMock()
    manager.recall_medium = AsyncMock(
        return_value=[
            {
                "entity": {
                    "id": "ent_engram",
                    "name": "Engram",
                    "type": "Project",
                    "summary": "Portable memory",
                },
                "score": 0.9,
                "result_type": "entity",
                "relationships": [],
            }
        ]
    )
    manager.record_memory_operation = record_memory_operation

    cfg = ActivationConfig(
        auto_recall_level="medium",
        recall_packets_enabled=False,
    )
    result = await build_lite_auto_recall_surface(
        manager,
        content="Working on Engram harness adoption progressive memory today",
        group_id="default",
        session_cache={},
        cfg=cfg,
    )
    assert result is not None
    assert result["source"] == "recall_medium"
    manager.recall_medium.assert_awaited_once()
    budget = recall_budget_for_profile(cfg, "auto_lite", surface="mcp", mode="medium")
    assert budget.max_wall_ms >= 300
    assert all(_sample_field(sample, "skip_reason") != "recall_timeout" for sample in gate_samples)
    assert all(not _sample_field(sample, "timeout") for sample in gate_samples)


@pytest.mark.asyncio
async def test_build_full_auto_recall_surface_executes_without_timeout() -> None:
    gate_samples: list[object] = []

    def record_memory_operation(group_id: str, sample: object) -> None:
        if _sample_field(sample, "operation") == "auto_recall_gate":
            gate_samples.append(sample)

    manager = MagicMock()
    manager.recall = AsyncMock(
        return_value=[
            {
                "entity": {
                    "id": "ent_engram",
                    "name": "Engram",
                    "type": "Project",
                    "summary": "Portable memory",
                },
                "score": 0.9,
                "result_type": "entity",
            }
        ]
    )
    manager.record_memory_operation = record_memory_operation
    manager.get_recall_need_graph_probe = MagicMock(return_value=None)

    cfg = ActivationConfig(
        consolidation_profile="off",
        recall_profile="off",
        integration_profile="off",
        auto_recall_enabled=True,
        recall_need_analyzer_enabled=False,
        recall_packets_enabled=False,
        auto_recall_min_score=0.1,
    )
    result = await build_full_auto_recall_surface(
        manager,
        content="Working on Engram harness adoption progressive memory today",
        group_id="default",
        cfg=cfg,
        session_last_recall_time=None,
        cooldown=None,
    )
    assert result is not None
    manager.recall.assert_awaited_once()
    budget = recall_budget_for_profile(cfg, "auto_deep", surface="mcp", mode="auto_recall")
    assert budget.max_wall_ms >= 300
    assert all(_sample_field(sample, "skip_reason") != "recall_timeout" for sample in gate_samples)
    assert all(not _sample_field(sample, "timeout") for sample in gate_samples)


@pytest.mark.asyncio
async def test_collect_growth_stats_from_graph_store() -> None:
    graph = MagicMock()
    graph.get_stats = AsyncMock(
        return_value={
            "episodes": 10,
            "entities": 4,
            "cue_metrics": {"cue_count": 7, "projected_cue_count": 2},
        }
    )
    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=MagicMock(),
        cfg=ActivationConfig(),
        recall=AsyncMock(return_value=[]),
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )
    stats = await builder._collect_growth_stats("default")
    assert stats == {"episodes": 10, "cues": 7, "promotions": 2, "entities": 4}
