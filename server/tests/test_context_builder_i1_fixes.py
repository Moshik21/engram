"""I1 bug-class fixes for get_context/context_builder.

Covers:
1. forget() invalidates every context cache (durable process cache, manager
   packet cache, briefing cache) — a forgotten fact stops being served within
   the same second.
2. Layer-2 recall entities are recorded once (recall materializer), not twice.
3. Entities cut by the char-budget truncation are not activation-strengthened.
4. No dead manager packet-cache writes under the 'durable_context' scope.
5. get_context's Project auto-create is guarded against concurrent-first-call
   duplicates.
"""

from __future__ import annotations

import asyncio
import inspect as inspect_mod
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.retrieval import context_builder as context_builder_mod
from engram.retrieval.context_builder import (
    MemoryContextBuilder,
    _load_durable_context_process_cache,
    _store_durable_context_process_cache,
    invalidate_durable_context_cache,
)

GROUP = "i1_group"


def _forget_manager() -> GraphManager:
    graph = AsyncMock()
    graph.find_entities = AsyncMock(
        return_value=[Entity(id="ent_old", name="Old Fact", entity_type="Decision", group_id=GROUP)]
    )
    graph.delete_entity = AsyncMock()
    activation = AsyncMock()
    activation.clear_activation = AsyncMock()
    cfg = ActivationConfig(recall_packet_cache_persistence_enabled=False)
    return GraphManager(graph, activation, AsyncMock(), AsyncMock(), cfg=cfg)


def _seed_context_caches(gm: GraphManager) -> None:
    _store_durable_context_process_cache(GROUP, "q", [{"title": "Old Fact"}], {"ent_old"})
    gm.cache_memory_packets(
        GROUP,
        scope="identity_core",
        packets=[{"title": "Old Fact", "entity_ids": ["ent_old"]}],
        persist=False,
    )
    gm._briefing_cache[(GROUP, None, ())] = (time.time(), "stale briefing")  # type: ignore[index]


def _assert_context_caches_cleared(gm: GraphManager) -> None:
    # Durable process cache: next get_context must rebuild, not serve the
    # 45s-TTL pack — the forgotten fact disappears within the same second.
    assert _load_durable_context_process_cache(GROUP, "q") is None
    assert gm.get_cached_memory_packets(GROUP, scope="identity_core", sync_persistent=False) is None
    assert not [key for key in gm._briefing_cache if key[0] == GROUP]


@pytest.mark.asyncio
async def test_forget_entity_invalidates_all_context_caches() -> None:
    gm = _forget_manager()
    _seed_context_caches(gm)
    assert _load_durable_context_process_cache(GROUP, "q") is not None

    result = await gm.forget_entity("Old Fact", GROUP)

    assert result["status"] == "forgotten"
    _assert_context_caches_cleared(gm)


@pytest.mark.asyncio
async def test_forget_fact_invalidates_all_context_caches() -> None:
    gm = _forget_manager()
    rel = MagicMock()
    rel.target_id = "ent_old"
    rel.id = "rel_1"
    gm._graph.get_relationships = AsyncMock(return_value=[rel])
    gm._graph.invalidate_relationship = AsyncMock()
    _seed_context_caches(gm)

    result = await gm.forget_fact("Old Fact", "USES", "Old Fact", GROUP)

    assert result["status"] == "forgotten"
    _assert_context_caches_cleared(gm)


def _builder(
    *,
    graph: MagicMock,
    activation: MagicMock,
    recall: AsyncMock,
    cfg: ActivationConfig | None = None,
) -> MemoryContextBuilder:
    return MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=cfg or ActivationConfig(identity_core_enabled=False, briefing_enabled=False),
        recall=recall,
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_get_context_records_recall_entities_only_once() -> None:
    """Layer-2 recall entities were recorded by the recall materializer already."""
    graph = MagicMock()
    graph.get_relationships = AsyncMock(return_value=[])
    graph.get_entity = AsyncMock(
        return_value=Entity(
            id="ent_recent",
            name="RecentEnt",
            entity_type="Concept",
            summary="recent",
            group_id=GROUP,
        )
    )
    activation = MagicMock()
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.get_top_activated = AsyncMock(
        return_value=[
            ("ent_recent", ActivationState(node_id="ent_recent", access_history=[time.time() - 5]))
        ]
    )
    recall = AsyncMock(
        return_value=[
            {
                "entity": {
                    "id": "ent_recall",
                    "name": "RecallEnt",
                    "type": "Concept",
                    "summary": "from recall",
                },
                "score_breakdown": {},
            }
        ]
    )
    builder = _builder(graph=graph, activation=activation, recall=recall)

    result = await builder.get_context(group_id=GROUP, topic_hint="widgets")

    assert "RecallEnt" in result["context"]
    recorded = [call.args[0] for call in activation.record_access.await_args_list]
    assert recorded == ["ent_recent"]  # recall entity NOT re-recorded by the builder


@pytest.mark.asyncio
async def test_get_context_truncated_entities_not_recorded() -> None:
    """Entities cut by the char budget were never delivered — never strengthened."""
    long_summary = "x" * 600
    entities = {
        "ent_a": Entity(
            id="ent_a", name="Alpha", entity_type="Concept", summary=long_summary, group_id=GROUP
        ),
        "ent_b": Entity(
            id="ent_b", name="Beta", entity_type="Concept", summary=long_summary, group_id=GROUP
        ),
    }
    graph = MagicMock()
    graph.get_relationships = AsyncMock(return_value=[])
    graph.get_entity = AsyncMock(side_effect=lambda eid, _gid: entities.get(eid))
    activation = MagicMock()
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    now = time.time()
    activation.get_top_activated = AsyncMock(
        return_value=[
            ("ent_a", ActivationState(node_id="ent_a", access_history=[now - 10])),
            ("ent_b", ActivationState(node_id="ent_b", access_history=[now - 10])),
        ]
    )
    builder = _builder(graph=graph, activation=activation, recall=AsyncMock(return_value=[]))

    result = await builder.get_context(group_id=GROUP, max_tokens=100)

    assert "Alpha" in result["context"]
    assert "- Beta (Concept" not in result["context"]  # truncated away
    recorded = [call.args[0] for call in activation.record_access.await_args_list]
    assert recorded == ["ent_a"]


def test_no_durable_context_scope_manager_cache_writes() -> None:
    """The 'durable_context' packet-cache mirror had no reader; it must stay deleted."""
    source = inspect_mod.getsource(context_builder_mod)
    assert "_cache_durable_context_packets" not in source
    assert "scope=DURABLE_CONTEXT_PACKET_SCOPE" not in source


def _project_builder(graph: MagicMock, activation: MagicMock) -> MemoryContextBuilder:
    return _builder(graph=graph, activation=activation, recall=AsyncMock(return_value=[]))


@pytest.mark.asyncio
async def test_project_create_recheck_uses_existing_entity() -> None:
    existing = Entity(id="ent_existing", name="Engram", entity_type="Project", group_id=GROUP)
    graph = MagicMock()
    graph.find_entities = AsyncMock(return_value=[existing])
    graph.create_entity = AsyncMock()
    activation = MagicMock()
    activation.record_access = AsyncMock()
    builder = _project_builder(graph, activation)

    from pathlib import Path

    entity_id = await builder._get_or_create_project_entity(
        Path("/tmp/Engram"), "/tmp/Engram", GROUP, time.time(), lookup_timeout=1.0
    )

    assert entity_id == "ent_existing"
    graph.create_entity.assert_not_called()


@pytest.mark.asyncio
async def test_project_create_concurrent_calls_create_once() -> None:
    release = asyncio.Event()

    async def blocked_find(**_kwargs):
        await release.wait()
        return []

    graph = MagicMock()
    graph.find_entities = AsyncMock(side_effect=blocked_find)
    graph.create_entity = AsyncMock()
    activation = MagicMock()
    activation.record_access = AsyncMock()
    builder = _project_builder(graph, activation)

    from pathlib import Path

    first = asyncio.ensure_future(
        builder._get_or_create_project_entity(
            Path("/tmp/Engram"), "/tmp/Engram", GROUP, time.time(), lookup_timeout=5.0
        )
    )
    await asyncio.sleep(0)  # let the first call claim the in-flight key
    second = await builder._get_or_create_project_entity(
        Path("/tmp/Engram"), "/tmp/Engram", GROUP, time.time(), lookup_timeout=5.0
    )
    release.set()
    first_id = await first

    assert second is None  # concurrent call skipped the duplicate create
    assert first_id is not None
    assert graph.create_entity.await_count == 1
    assert not builder._project_creates_in_flight


@pytest.fixture(autouse=True)
def _clean_durable_cache():
    invalidate_durable_context_cache()
    yield
    invalidate_durable_context_cache()
