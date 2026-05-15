from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from engram.config import ActivationConfig
from engram.models.prospective import IntentionMeta
from engram.retrieval.prospective import ProspectiveMemoryService


class FakeGraphStore:
    def __init__(self, *, graph_embedded: bool = True) -> None:
        self.graph_embedded = graph_embedded
        self.meta = IntentionMeta(
            trigger_text="native Helix",
            action_text="Check native path",
            trigger_type="refresh_context",
            activation_threshold=0.5,
            trigger_entity_ids=["ent_native"],
            priority="high",
            refresh_trigger="after_consolidation",
            last_refreshed="2026-05-15T00:00:00+00:00",
            pinned_result="fresh context",
        )
        self.entity = SimpleNamespace(
            id="intent_1",
            attributes=self.meta.model_dump(mode="json"),
        )
        self.flat = SimpleNamespace(
            id="flat_1",
            trigger_text="flat trigger",
            action_text="flat action",
            trigger_type="semantic",
            entity_name="Engram",
            threshold=0.7,
            fire_count=1,
            max_fires=3,
            enabled=True,
            expires_at=datetime(2026, 5, 15, tzinfo=timezone.utc),
        )

    async def find_entities(self, *, entity_type: str, group_id: str, limit: int):
        assert entity_type == "Intention"
        assert group_id == "brain"
        assert limit == 100
        return [self.entity]

    async def list_intentions(self, group_id: str, *, enabled_only: bool):
        assert group_id == "brain"
        assert enabled_only is True
        return [self.flat]


class FakeActivationStore:
    async def get_activation(self, entity_id: str):
        assert entity_id == "intent_1"
        return SimpleNamespace(access_history=[], access_count=0, last_accessed=None)


def _service(cfg: ActivationConfig, graph: FakeGraphStore) -> ProspectiveMemoryService:
    return ProspectiveMemoryService(
        graph_store=graph,
        activation_store=FakeActivationStore(),
        search_index=SimpleNamespace(),
        cfg=cfg,
        publish_event=lambda *_args, **_kwargs: None,
    )


@pytest.mark.asyncio
async def test_intention_views_present_api_graph_embedded_rows() -> None:
    cfg = ActivationConfig(prospective_graph_embedded=True)
    items = await _service(cfg, FakeGraphStore()).list_intention_views(
        group_id="brain",
        enabled_only=True,
        surface="api",
    )

    assert items == [
        {
            "id": "intent_1",
            "triggerText": "native Helix",
            "actionText": "Check native path",
            "triggerType": "refresh_context",
            "threshold": 0.5,
            "fireCount": 0,
            "maxFires": 5,
            "enabled": True,
            "priority": "high",
            "expiresAt": None,
            "warmthRatio": 0.057,
            "linkedEntityIds": ["ent_native"],
            "refreshTrigger": "after_consolidation",
            "lastRefreshed": "2026-05-15T00:00:00+00:00",
            "hasPinnedResult": True,
        }
    ]


@pytest.mark.asyncio
async def test_intention_views_present_mcp_graph_embedded_rows() -> None:
    cfg = ActivationConfig(prospective_graph_embedded=True)
    items = await _service(cfg, FakeGraphStore()).list_intention_views(
        group_id="brain",
        enabled_only=True,
        surface="mcp",
    )

    assert items[0]["trigger_text"] == "native Helix"
    assert items[0]["warmth_ratio"] == 0.057
    assert items[0]["warmth_label"] == "dormant"
    assert items[0]["refresh_trigger"] == "after_consolidation"
    assert items[0]["has_pinned_result"] is True


@pytest.mark.asyncio
async def test_intention_views_present_flat_rows() -> None:
    cfg = ActivationConfig(prospective_graph_embedded=False)
    service = _service(cfg, FakeGraphStore(graph_embedded=False))

    api_items = await service.list_intention_views("brain", True, surface="api")
    mcp_items = await service.list_intention_views("brain", True, surface="mcp")

    assert api_items[0]["triggerText"] == "flat trigger"
    assert mcp_items[0]["trigger_text"] == "flat trigger"
    assert mcp_items[0]["expires_at"] == "2026-05-15T00:00:00+00:00"


def test_effective_intention_threshold_uses_runtime_default() -> None:
    cfg = ActivationConfig(prospective_activation_threshold=0.42)
    service = _service(cfg, FakeGraphStore())

    assert service.effective_activation_threshold(None) == 0.42
    assert service.effective_activation_threshold(0.7) == 0.7
