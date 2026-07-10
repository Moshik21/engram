from __future__ import annotations

import pytest

from engram.config import HelixDBConfig
from engram.models.consolidation import ConsolidationCycle
from engram.storage.helix.consolidation import HelixConsolidationStore


@pytest.mark.asyncio
async def test_helix_consolidation_cycle_cache_is_group_scoped(monkeypatch) -> None:
    store = HelixConsolidationStore(HelixDBConfig())
    store._cache_cycle(101, "cyc_shared", "brain_a")
    store._cache_cycle(202, "cyc_shared", "brain_b")
    updated_payloads: list[dict] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        assert endpoint == "update_consol_cycle"
        updated_payloads.append(payload)
        return []

    monkeypatch.setattr(store, "_query", fake_query)

    cycle = ConsolidationCycle(id="cyc_shared", group_id="brain_b", status="completed")
    await store.update_cycle(cycle)

    assert updated_payloads[0]["id"] == 202
    assert store._cycle_group_id_cache[("brain_a", "cyc_shared")] == 101
    assert store._cycle_group_id_cache[("brain_b", "cyc_shared")] == 202
    assert store._cycle_id_cache["cyc_shared"] is None


@pytest.mark.asyncio
async def test_helix_update_cycle_resolves_from_active_group_only(monkeypatch) -> None:
    store = HelixConsolidationStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "get_consol_cycle_by_cycle_id":
            raise AssertionError("update_cycle should not use unscoped cycle lookup")
        if endpoint == "find_consol_cycles_by_group":
            assert payload == {"gid": "brain_b"}
            return [
                {"id": 101, "cycle_id": "cyc_shared", "group_id": "brain_a"},
                {"id": 202, "cycle_id": "cyc_shared", "group_id": "brain_b"},
            ]
        if endpoint == "update_consol_cycle":
            return []
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    cycle = ConsolidationCycle(id="cyc_shared", group_id="brain_b", status="completed")
    await store.update_cycle(cycle)

    assert (
        "update_consol_cycle",
        {
            "id": 202,
            "status": "completed",
            "phase_results_json": "[]",
            "completed_at": 0.0,
            "total_duration_ms": 0.0,
            "error": "",
        },
    ) in calls
    assert store._cycle_group_id_cache[("brain_b", "cyc_shared")] == 202
