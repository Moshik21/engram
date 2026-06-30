"""Minimal manager that drives auto-recall through real RecallService + retrieve."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

from engram.config import ActivationConfig
from engram.retrieval.pipeline import retrieve
from engram.retrieval.post_process import RecallPostProcessResult
from engram.retrieval.primary_results import RecallPrimaryMaterialization
from engram.retrieval.scorer import ScoredResult
from engram.retrieval.service import RecallService
from engram.retrieval.working_memory import WorkingMemoryBuffer
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex


@dataclass
class _PassthroughPrimary:
    async def materialize(
        self,
        results: list[ScoredResult],
        **kwargs: Any,
    ) -> RecallPrimaryMaterialization:
        materialized: list[dict[str, Any]] = []
        for scored in results:
            if scored.result_type != "entity":
                continue
            materialized.append(
                {
                    "result_type": "entity",
                    "entity": {
                        "id": scored.node_id,
                        "name": scored.node_id.replace("ent_", "").title(),
                        "type": "Project",
                        "summary": "Portable memory",
                    },
                    "score": scored.score,
                    "relationships": [{"predicate": "USES"}],
                }
            )
        return RecallPrimaryMaterialization(
            results=materialized,
            seen_episode_ids=set(),
        )


@dataclass
class _PassthroughPost:
    async def process(
        self,
        results: list[dict[str, Any]],
        **kwargs: Any,
    ) -> RecallPostProcessResult:
        return RecallPostProcessResult(results=results, near_misses=[])


@dataclass
class MinimalAutoRecallManager:
    """Thin manager facade for builder integration tests."""

    cfg: ActivationConfig
    graph_store: GraphStore
    activation_store: ActivationStore
    search_index: SearchIndex
    recall_calls: list[dict[str, Any]] = field(default_factory=list)
    memory_operations: list[tuple[str, Any]] = field(default_factory=list)
    _neighbor_calls: int = 0
    _graph_delay_seconds: float = 0.12

    def __post_init__(self) -> None:
        self._recall_service = RecallService(
            graph_store=self.graph_store,
            activation_store=self.activation_store,
            search_index=self.search_index,
            cfg=self.cfg,
            primary_materializer=_PassthroughPrimary(),
            post_processor=_PassthroughPost(),
            retrieve_fn=retrieve,
        )
        self._working_memory = WorkingMemoryBuffer()
        self._priming_buffer: dict[str, tuple[float, float]] = {}

    async def _neighbors_with_first_slow(
        self,
        entity_id: str,
        group_id: str | None = None,
        **_kwargs: Any,
    ) -> list[tuple[str, float, str]]:
        self._neighbor_calls += 1
        if self._neighbor_calls == 1:
            await asyncio.sleep(self._graph_delay_seconds)
            return [("n1", 0.8, "RELATED_TO")]
        return []

    def wire_slow_graph_pool(self) -> None:
        """Install a graph neighbor mock that completes only under auto timeout."""
        self._neighbor_calls = 0
        self.graph_store.get_active_neighbors_with_weights = AsyncMock(  # type: ignore[method-assign]
            side_effect=self._neighbors_with_first_slow,
        )
        self.graph_store.get_entity = AsyncMock(return_value=None)  # type: ignore[method-assign]
        self.graph_store.get_relationships = AsyncMock(return_value=[])  # type: ignore[method-assign]

    async def recall(
        self,
        query: str,
        group_id: str = "default",
        limit: int = 10,
        *,
        record_access: bool = True,
        interaction_type: str | None = None,
        interaction_source: str = "recall",
        memory_need: object | None = None,
    ) -> list[dict[str, Any]]:
        self.recall_calls.append(
            {
                "query": query,
                "group_id": group_id,
                "limit": limit,
                "record_access": record_access,
                "interaction_type": interaction_type,
                "interaction_source": interaction_source,
                "memory_need": memory_need,
            }
        )
        recall_result = await self._recall_service.recall(
            query=query,
            group_id=group_id,
            limit=limit,
            record_access=record_access,
            interaction_type=interaction_type,
            interaction_source=interaction_source,
            conv_context=None,
            working_memory=self._working_memory,
            priming_buffer=self._priming_buffer,
            goal_cache=None,
            memory_need=memory_need,
        )
        return recall_result.results

    def get_last_recall_stage_timings(self) -> dict[str, float]:
        return self._recall_service.last_stage_timings()

    def get_recall_need_graph_probe(self) -> None:
        return None

    def record_memory_operation(self, group_id: str, sample: Any) -> None:
        self.memory_operations.append((group_id, sample))

    def gate_samples(self) -> list[Any]:
        return [
            sample
            for _group_id, sample in self.memory_operations
            if getattr(sample, "operation", None) == "auto_recall_gate"
        ]


def build_minimal_auto_recall_manager(
    cfg: ActivationConfig | None = None,
    *,
    graph_delay_seconds: float = 0.12,
) -> MinimalAutoRecallManager:
    """Create a manager with slow-graph mocks wired for auto budget proof."""
    resolved_cfg = cfg or ActivationConfig(
        consolidation_profile="off",
        recall_profile="off",
        integration_profile="off",
        multi_pool_enabled=True,
        episode_retrieval_enabled=False,
        retrieval_graph_pool_timeout_ms=75,
        retrieval_graph_pool_timeout_auto_ms=250,
        reranker_enabled=False,
        mmr_enabled=False,
        graph_query_expansion_enabled=False,
        entity_episode_traversal_enabled=False,
        chunk_search_enabled=False,
        cue_recall_enabled=False,
    )

    search_idx = AsyncMock()
    search_idx.search = AsyncMock(return_value=[("e1", 0.9)])
    search_idx.compute_similarity = AsyncMock(return_value={})
    search_idx.search_episodes = AsyncMock(return_value=[])
    search_idx._embeddings_enabled = False

    act_store = AsyncMock()
    act_store.get_top_activated = AsyncMock(return_value=[])
    act_store.batch_get = AsyncMock(return_value={})

    graph = AsyncMock()
    graph.get_stats = AsyncMock(return_value={"entities": 1, "episodes": 1})

    manager = MinimalAutoRecallManager(
        cfg=resolved_cfg,
        graph_store=graph,
        activation_store=act_store,
        search_index=search_idx,
    )
    manager._graph_delay_seconds = graph_delay_seconds
    manager.wire_slow_graph_pool()
    return manager
