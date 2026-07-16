"""Graph/consol-backed LoopAdjustment store (Helix sidecar contract + lite store)."""

from __future__ import annotations

import pytest

from engram.consolidation.store import SQLiteConsolidationStore
from engram.loop_adjustment import (
    LoopAdjustment,
    clamp_loop_adjustment,
    clear_active_adjustment_async,
    load_active_adjustment_async,
    save_active_adjustment_async,
    stamp_applied,
)


@pytest.mark.asyncio
async def test_lite_consolidation_store_loop_adjustment_roundtrip(tmp_path):
    db_path = tmp_path / "consol.db"
    store = SQLiteConsolidationStore(str(db_path))
    await store.initialize()

    adj = stamp_applied(
        clamp_loop_adjustment(
            LoopAdjustment.from_mapping(
                {
                    "reason": "graph store test",
                    "regime": "debt_heavy",
                    "ttl_hours": 6,
                    "max_risk": "low",
                    "budgets": {"evidence_drain": 1500},
                    "phase_defer": ["dream"],
                }
            )
        ).adjustment
    )

    await store.save_loop_adjustment("default", adj.to_dict())
    loaded = await store.get_loop_adjustment("default")
    assert loaded is not None
    assert loaded["regime"] == "debt_heavy"
    assert loaded["budgets"]["evidence_drain"] == 1500

    # Dual-read API with graph_store
    via = await load_active_adjustment_async(
        "default",
        path=tmp_path / "no-file.json",
        graph_store=store,
    )
    assert via is not None
    assert via.regime == "debt_heavy"

    # Dual-write via save_active_adjustment_async
    adj2 = stamp_applied(
        clamp_loop_adjustment(
            LoopAdjustment.from_mapping(
                {
                    "reason": "second write",
                    "regime": "intake_heavy",
                    "ttl_hours": 3,
                    "max_risk": "low",
                    "budgets": {"cue_hygiene": 900},
                }
            )
        ).adjustment
    )
    await save_active_adjustment_async(
        adj2,
        path=tmp_path / "file-adj.json",
        audit_path=tmp_path / "audit.jsonl",
        graph_store=store,
    )
    # File dual-write
    assert (tmp_path / "file-adj.json").is_file()
    again = await store.get_loop_adjustment("default")
    assert again is not None
    assert again["regime"] == "intake_heavy"

    cleared = await clear_active_adjustment_async(
        "default",
        path=tmp_path / "file-adj.json",
        audit_path=tmp_path / "audit.jsonl",
        graph_store=store,
    )
    assert cleared is True
    assert await store.get_loop_adjustment("default") is None
    await store.close()


@pytest.mark.asyncio
async def test_file_fallback_when_graph_missing(tmp_path, monkeypatch):
    path = tmp_path / "loop-adjustment.json"
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_FILE", str(path))
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE", str(tmp_path / "a.jsonl"))
    adj = stamp_applied(
        clamp_loop_adjustment(
            LoopAdjustment.from_mapping(
                {
                    "reason": "file only",
                    "regime": "latency_degraded",
                    "ttl_hours": 2,
                    "max_risk": "low",
                }
            )
        ).adjustment
    )
    await save_active_adjustment_async(adj, path=path, graph_store=None)
    loaded = await load_active_adjustment_async("default", path=path, graph_store=None)
    assert loaded is not None
    assert loaded.regime == "latency_degraded"
