"""Integration tests: build_full_auto_recall_surface through real recall chain."""

from __future__ import annotations

import json

import pytest

from engram.config import ActivationConfig
from engram.retrieval.auto_recall import build_full_auto_recall_surface
from tests.helpers.minimal_auto_recall_manager import build_minimal_auto_recall_manager


def _auto_recall_cfg(**kwargs: object) -> ActivationConfig:
    base = {
        "consolidation_profile": "off",
        "recall_profile": "off",
        "integration_profile": "off",
        "auto_recall_enabled": True,
        "recall_need_analyzer_enabled": False,
        "recall_packets_enabled": False,
        "auto_recall_min_score": 0.1,
        "multi_pool_enabled": True,
        "episode_retrieval_enabled": False,
        "retrieval_graph_pool_timeout_ms": 75,
        "retrieval_graph_pool_timeout_auto_ms": 250,
        "reranker_enabled": False,
        "mmr_enabled": False,
        "graph_query_expansion_enabled": False,
        "entity_episode_traversal_enabled": False,
        "chunk_search_enabled": False,
        "cue_recall_enabled": False,
    }
    base.update(kwargs)
    return ActivationConfig(**base)


@pytest.mark.asyncio
async def test_build_full_auto_recall_surface_completes_slow_graph_pool() -> None:
    """Shipped builder must reach recall_graph_pool via interaction_source=auto_recall."""
    cfg = _auto_recall_cfg()
    manager = build_minimal_auto_recall_manager(cfg)
    content = "Working on Engram harness adoption progressive memory today"

    result = await build_full_auto_recall_surface(
        manager,
        content=content,
        group_id="default",
        cfg=cfg,
        session_last_recall_time=None,
        cooldown=None,
        now=100.0,
    )

    assert result is not None
    assert manager.recall_calls
    assert manager.recall_calls[0]["interaction_source"] == "auto_recall"

    stage_timings = manager.get_last_recall_stage_timings()
    assert "recall_graph_pool" in stage_timings
    assert "recall_graph_pool_timeout" not in stage_timings
    assert stage_timings["recall_graph_pool"] >= manager._graph_delay_seconds * 1000

    gate_samples = manager.gate_samples()
    assert all(getattr(sample, "skip_reason", None) != "recall_timeout" for sample in gate_samples)
    assert all(not getattr(sample, "timeout", False) for sample in gate_samples)


def format_builder_gate_evidence() -> str:
    """Print gate + timing evidence for verification scripts."""
    import asyncio

    cfg = _auto_recall_cfg()
    manager = build_minimal_auto_recall_manager(cfg)
    content = "Working on Engram harness adoption progressive memory today"
    result = asyncio.run(
        build_full_auto_recall_surface(
            manager,
            content=content,
            group_id="default",
            cfg=cfg,
            session_last_recall_time=None,
            cooldown=None,
            now=100.0,
        )
    )
    payload = {
        "surface_source": result.get("source") if result else None,
        "recall_calls": manager.recall_calls,
        "stage_timings_ms": manager.get_last_recall_stage_timings(),
        "gate_samples": [
            {
                "status": getattr(sample, "status", None),
                "skip_reason": getattr(sample, "skip_reason", None),
                "timeout": getattr(sample, "timeout", None),
            }
            for sample in manager.gate_samples()
        ],
    }
    return json.dumps(payload, indent=2)
