"""RF M4.2: mop-window activation->graph snapshot sync (F3 resolved: WIRE).

snapshot_to_graph is called inside the mop window after the drains, budgeted.
The graph-row access_count/last_accessed columns it writes are APPROXIMATE
(stale up to one mop window); ranking must never read them — the last test
pins that a graph-row access_count can never change scoring.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.hygiene_debt import HygieneDebtSnapshot
from engram.models.consolidation import PhaseResult
from engram.storage.memory.activation import MemoryActivationStore


@pytest.fixture()
def engram_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    return tmp_path


class _RecordingGraphStore:
    def __init__(self) -> None:
        self.updates: list[tuple[str, dict, str]] = []

    async def update_entity(self, entity_id: str, updates: dict, group_id: str) -> None:
        self.updates.append((entity_id, updates, group_id))


def _debt(**overrides) -> HygieneDebtSnapshot:
    import inspect

    fields = dict(
        deferred_evidence=0,
        pending_evidence=0,
        cue_only_episodes=0,
        cue_count=0,
        near_miss_count=0,
        open_adjudication=0,
        orphan_candidates=0,
        low_value_entities=0,
    )
    fields.update(overrides)
    allowed = set(inspect.signature(HygieneDebtSnapshot).parameters)
    return HygieneDebtSnapshot(**{k: v for k, v in fields.items() if k in allowed})


def _mop_patches():
    return (
        patch(
            "engram.consolidation.hygiene_debt.collect_hygiene_debt_from_store",
            new=AsyncMock(return_value=_debt(deferred_evidence=600)),
        ),
        patch(
            "engram.consolidation.evidence_drain.load_deferred_evidence",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "engram.consolidation.evidence_drain.reject_junk_evidence",
            new=AsyncMock(return_value={"rejected": 0}),
        ),
        patch(
            "engram.consolidation.evidence_drain.reject_evidence_rows",
            new=AsyncMock(return_value={"rejected": 0}),
        ),
        patch(
            "engram.consolidation.cue_hygiene.run_cue_hygiene",
            new=AsyncMock(return_value=type("R", (), {"to_dict": lambda self: {}})()),
        ),
        patch("engram.consolidation.phases.prune.PrunePhase"),
    )


async def _run_mop(graph_store, activation_store, *, dry_run: bool = False) -> dict:
    from engram.hygiene_ops import execute_hygiene_mop

    patches = _mop_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as prune_cls:
        prune_cls.return_value.execute = AsyncMock(
            return_value=(PhaseResult(phase="prune", status="completed"), [])
        )
        return await execute_hygiene_mop(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=object(),
            activation_cfg=ActivationConfig(),
            group_id="g",
            budget=100,
            dry_run=dry_run,
        )


class TestMopSnapshotSync:
    @pytest.mark.asyncio
    async def test_mop_syncs_activation_counters_to_graph_rows(self, engram_home: Path):
        activation = MemoryActivationStore()
        now = time.time()
        await activation.record_access("ent-a", now - 10, group_id="brain_a")
        await activation.record_access("ent-a", now - 5, group_id="brain_a")
        await activation.record_access("ent-b", now - 3, group_id="brain_b")
        graph = _RecordingGraphStore()

        report = await _run_mop(graph, activation)

        assert report["mop"]["snapshot_sync"]["entities"] == 2
        assert report["mop"]["snapshot_sync"]["budget"] == 5000
        synced = {eid: (updates, gid) for eid, updates, gid in graph.updates}
        assert synced["ent-a"][0]["access_count"] == 2
        assert synced["ent-a"][1] == "brain_a"
        assert synced["ent-b"][0]["access_count"] == 1
        assert synced["ent-b"][1] == "brain_b"

    @pytest.mark.asyncio
    async def test_dry_run_skips_sync(self, engram_home: Path):
        activation = MemoryActivationStore()
        await activation.record_access("ent-a", time.time(), group_id="g")
        graph = _RecordingGraphStore()

        report = await _run_mop(graph, activation, dry_run=True)

        assert report["mop"]["snapshot_sync"]["skipped"] is True
        assert graph.updates == []

    @pytest.mark.asyncio
    async def test_store_without_snapshot_to_graph_skips(self, engram_home: Path):
        """Redis/compat stores without the API skip gracefully, loudly labeled."""
        report = await _run_mop(_RecordingGraphStore(), object())

        assert report["mop"]["snapshot_sync"]["skipped"] is True
        assert "snapshot_to_graph" in report["mop"]["snapshot_sync"]["reason"]

    @pytest.mark.asyncio
    async def test_budget_caps_and_prefers_most_recent(self):
        """The per-window cap syncs most-recently-accessed entities first."""
        activation = MemoryActivationStore()
        now = time.time()
        await activation.record_access("ent-old", now - 100, group_id="g")
        await activation.record_access("ent-mid", now - 50, group_id="g")
        await activation.record_access("ent-new", now - 1, group_id="g")
        graph = _RecordingGraphStore()

        synced = await activation.snapshot_to_graph(graph, limit=2)

        assert synced == 2
        assert [eid for eid, _u, _g in graph.updates] == ["ent-new", "ent-mid"]


class TestRankingNeverReadsGraphAccessCount:
    """M4.2 DoD pin: the graph-row access_count column (approximate, synced
    only in mop windows) must never influence ranking. Scoring reads counters
    exclusively from ActivationState; an inflated graph-row value passed via
    entity_attributes changes nothing, flag off AND on."""

    @pytest.mark.parametrize("usage_ranking_enabled", [False, True])
    def test_graph_row_access_count_cannot_change_scores(self, usage_ranking_enabled: bool):
        from engram.models.activation import ActivationState
        from engram.retrieval.scorer import score_candidates

        now = time.time()
        cfg = ActivationConfig(usage_ranking_enabled=usage_ranking_enabled)
        candidates = [("ent-a", 0.9), ("ent-b", 0.7)]
        states = {
            "ent-a": ActivationState(node_id="ent-a", access_history=[now - 60.0], access_count=1),
        }

        def _score(entity_attributes):
            return [
                (r.node_id, r.score)
                for r in score_candidates(
                    candidates=list(candidates),
                    spreading_bonuses={},
                    hop_distances={},
                    seed_node_ids=set(),
                    activation_states=states,
                    now=now,
                    cfg=cfg,
                    entity_attributes=entity_attributes,
                )
            ]

        baseline = _score({"ent-a": {}, "ent-b": {}})
        inflated = _score(
            {
                "ent-a": {"access_count": 10**9, "last_accessed": now},
                "ent-b": {"access_count": 10**9, "last_accessed": now},
            }
        )
        assert inflated == baseline
