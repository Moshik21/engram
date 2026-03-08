"""Tests for planner-driven multi-intent recall."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.models.recall import MemoryNeed, RecallIntent, RecallPlan
from engram.retrieval.context import ConversationContext
from engram.retrieval.pipeline import retrieve
from engram.retrieval.plan import build_recall_plan, execute_recall_plan


class TestBuildRecallPlan:
    def test_builds_direct_topic_and_session_entity_intents(self):
        cfg = ActivationConfig(
            recall_planner_enabled=True,
            conv_multi_query_enabled=True,
            conv_multi_query_turns=2,
            conv_multi_query_top_entities=3,
            conv_context_enabled=True,
        )
        ctx = ConversationContext()
        ctx.add_turn("Working on React migration")
        ctx.add_turn("Also using TypeScript")
        ctx.add_session_entity("e1", "React", "Technology", now=1.0)
        ctx.add_session_entity("e2", "TypeScript", "Technology", now=1.0)

        plan = build_recall_plan("frontend migration", cfg, conv_context=ctx)

        assert [intent.intent_type for intent in plan.intents] == [
            "direct",
            "topic",
            "session_entity",
        ]

    def test_caps_plan_size(self):
        cfg = ActivationConfig(
            recall_planner_enabled=True,
            conv_multi_query_enabled=True,
            recall_planner_max_intents=2,
            conv_context_enabled=True,
        )
        ctx = ConversationContext()
        ctx.add_turn("topic one")
        ctx.add_turn("topic two")
        ctx.add_session_entity("e1", "Redis", "Technology", now=1.0)

        plan = build_recall_plan("cache", cfg, conv_context=ctx)

        assert len(plan.intents) == 2

    def test_carries_seed_entity_ids_from_memory_need(self):
        cfg = ActivationConfig(recall_planner_enabled=True)
        memory_need = MemoryNeed(
            need_type="fact_lookup",
            should_recall=True,
            confidence=0.8,
            detected_entities=["e1", "e2", "e1"],
        )

        plan = build_recall_plan("cache", cfg, memory_need=memory_need)

        assert plan.seed_entity_ids == ["e1", "e2"]


@pytest.mark.asyncio
class TestExecuteRecallPlan:
    async def test_accumulates_support_across_intents(self):
        search_index = AsyncMock()

        async def mock_search(query, group_id, limit):
            if query == "recent work":
                return [("e1", 0.6)]
            if query == "React TypeScript":
                return [("e1", 0.5), ("e2", 0.7)]
            return []

        search_index.search = AsyncMock(side_effect=mock_search)
        plan = RecallPlan(
            query="frontend",
            mode="explicit_recall",
            intents=[
                RecallIntent("direct", "frontend", 1.0, 25),
                RecallIntent("topic", "recent work", 0.35, 25),
                RecallIntent("session_entity", "React TypeScript", 0.30, 25),
            ],
        )

        trace = await execute_recall_plan(
            plan,
            group_id="default",
            search_index=search_index,
            base_candidates=[("e1", 0.4)],
        )

        merged = dict(trace.merged_candidates)
        assert merged["e1"] > 0.4
        assert "e2" in merged
        assert trace.intent_types["e1"] == ["direct", "topic", "session_entity"]

    async def test_injects_seed_entities_into_trace(self):
        search_index = AsyncMock()
        search_index.search = AsyncMock(return_value=[])
        plan = RecallPlan(
            query="frontend",
            mode="explicit_recall",
            intents=[RecallIntent("direct", "frontend", 1.0, 25)],
            seed_entity_ids=["e_seed"],
        )

        trace = await execute_recall_plan(
            plan,
            group_id="default",
            search_index=search_index,
            base_candidates=[],
        )

        assert trace.support_scores["e_seed"] == pytest.approx(0.14)
        assert trace.intent_types["e_seed"] == ["seed_entity"]
        assert trace.support_details["e_seed"][0]["intent_type"] == "seed_entity"


@pytest.mark.asyncio
class TestPlannerPipelineIntegration:
    async def test_retrieve_attaches_planner_trace(self):
        cfg = ActivationConfig(
            recall_planner_enabled=True,
            conv_multi_query_enabled=True,
            conv_context_enabled=True,
            multi_pool_enabled=False,
            mmr_enabled=False,
        )
        ctx = ConversationContext()
        ctx.add_turn("recent work")
        ctx.add_turn("frontend migration")
        ctx.add_session_entity("e1", "React", "Technology", now=1.0)
        ctx.add_session_entity("e2", "TypeScript", "Technology", now=1.0)

        search_index = AsyncMock()

        async def mock_search(query, group_id, limit):
            if query == "frontend":
                return [("e1", 0.4)]
            if query == "recent work frontend migration":
                return [("e1", 0.6)]
            if query == "React TypeScript":
                return [("e1", 0.5), ("e2", 0.7)]
            return []

        search_index.search = AsyncMock(side_effect=mock_search)
        search_index.search_episodes = AsyncMock(return_value=[])
        search_index.compute_similarity = AsyncMock(return_value={})

        activation_store = AsyncMock()
        activation_store.batch_get = AsyncMock(return_value={})

        graph_store = AsyncMock()
        graph_store.get_stats = AsyncMock(return_value={"entity_count": 10})
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        results = await retrieve(
            query="frontend",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
            enable_routing=False,
            conv_context=ctx,
        )

        e1 = next(result for result in results if result.node_id == "e1")
        assert e1.planner_support > 0.4
        assert "topic" in e1.planner_intents
        assert e1.recall_trace
