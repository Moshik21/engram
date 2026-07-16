"""M2 metabolism: mop adjudication/replay passes, no-work skip, cue watermark,
and activation-state persistence across restarts."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.hygiene_debt import HygieneDebtSnapshot
from engram.models.consolidation import PhaseResult


@pytest.fixture()
def engram_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    return tmp_path


def _quiet_debt(**overrides) -> HygieneDebtSnapshot:
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
    try:
        return HygieneDebtSnapshot(**fields)
    except TypeError:
        # Tolerate model drift: construct with only known fields.
        import inspect

        allowed = set(inspect.signature(HygieneDebtSnapshot).parameters)
        return HygieneDebtSnapshot(**{k: v for k, v in fields.items() if k in allowed})


class TestSkipWhenNoWork:
    @pytest.mark.asyncio
    async def test_quiet_debt_skips_all_drains(self, engram_home: Path):
        from engram.hygiene_ops import execute_hygiene_mop

        debt = _quiet_debt()
        with (
            patch(
                "engram.consolidation.hygiene_debt.collect_hygiene_debt_from_store",
                new=AsyncMock(return_value=debt),
            ),
            patch(
                "engram.consolidation.evidence_drain.load_deferred_evidence",
                new=AsyncMock(return_value=[]),
            ) as load_deferred,
        ):
            report = await execute_hygiene_mop(
                graph_store=object(),
                activation_store=object(),
                search_index=object(),
                activation_cfg=ActivationConfig(),
                group_id="g",
                budget=100,
                skip_when_no_work=True,
            )
        assert report["mop"]["skipped"] is True
        load_deferred.assert_not_called()

    @pytest.mark.asyncio
    async def test_actionable_debt_runs_drains(self, engram_home: Path):
        from engram.hygiene_ops import execute_hygiene_mop

        debt = _quiet_debt(deferred_evidence=600)
        with (
            patch(
                "engram.consolidation.hygiene_debt.collect_hygiene_debt_from_store",
                new=AsyncMock(return_value=debt),
            ),
            patch(
                "engram.consolidation.evidence_drain.load_deferred_evidence",
                new=AsyncMock(return_value=[]),
            ) as load_deferred,
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
                new=AsyncMock(side_effect=AssertionError("should not scan")),
            ),
            patch("engram.consolidation.phases.prune.PrunePhase") as prune_cls,
        ):
            prune_cls.return_value.execute = AsyncMock(
                return_value=(PhaseResult(phase="prune", status="completed"), [])
            )
            # Pre-mark cue scan as done so the watermark skips it.
            from engram.hygiene_ops import mark_cue_scan_done

            mark_cue_scan_done()
            report = await execute_hygiene_mop(
                graph_store=object(),
                activation_store=object(),
                search_index=object(),
                activation_cfg=ActivationConfig(),
                group_id="g",
                budget=100,
                skip_when_no_work=True,
            )
        assert "skipped" not in report["mop"]
        assert load_deferred.call_count >= 1
        assert report["mop"]["cue_hygiene"]["skipped"] is True


class TestMetabolizePasses:
    @pytest.mark.asyncio
    async def test_adjudication_and_replay_run_with_manager(self, engram_home: Path):
        from engram.hygiene_ops import execute_hygiene_mop

        debt = _quiet_debt(deferred_evidence=600)
        executed: list[str] = []

        def _phase(name: str):
            class _Fake:
                def __init__(self, *a, **k):
                    pass

                async def execute(self, **kwargs):
                    executed.append(name)
                    return (
                        PhaseResult(
                            phase=name,
                            status="completed",
                            items_processed=3,
                            items_affected=2,
                        ),
                        [],
                    )

            return _Fake

        cfg = ActivationConfig(consolidation_replay_enabled=True)
        with (
            patch(
                "engram.consolidation.hygiene_debt.collect_hygiene_debt_from_store",
                new=AsyncMock(return_value=debt),
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
                new=AsyncMock(side_effect=AssertionError("watermarked off")),
            ),
            patch(
                "engram.consolidation.phases.evidence_adjudication.EvidenceAdjudicationPhase",
                _phase("evidence_adjudication"),
            ),
            patch(
                "engram.consolidation.phases.edge_adjudication.EdgeAdjudicationPhase",
                _phase("edge_adjudication"),
            ),
            patch(
                "engram.consolidation.phases.replay.EpisodeReplayPhase",
                _phase("replay"),
            ),
            patch("engram.consolidation.phases.prune.PrunePhase") as prune_cls,
        ):
            prune_cls.return_value.execute = AsyncMock(
                return_value=(PhaseResult(phase="prune", status="completed"), [])
            )
            from engram.hygiene_ops import mark_cue_scan_done

            mark_cue_scan_done()
            report = await execute_hygiene_mop(
                graph_store=object(),
                activation_store=object(),
                search_index=object(),
                activation_cfg=cfg,
                group_id="g",
                budget=100,
                graph_manager=object(),
                extractor=object(),
            )
        assert executed == ["evidence_adjudication", "edge_adjudication", "replay"]
        assert report["mop"]["evidence_adjudication"]["items_affected"] == 2
        assert report["mop"]["replay"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_no_manager_means_drains_only(self, engram_home: Path):
        from engram.hygiene_ops import execute_hygiene_mop

        debt = _quiet_debt(deferred_evidence=600)
        with (
            patch(
                "engram.consolidation.hygiene_debt.collect_hygiene_debt_from_store",
                new=AsyncMock(return_value=debt),
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
                new=AsyncMock(
                    return_value=type("R", (), {"to_dict": lambda self: {"demoted": 0}})()
                ),
            ),
            patch("engram.consolidation.phases.prune.PrunePhase") as prune_cls,
        ):
            prune_cls.return_value.execute = AsyncMock(
                return_value=(PhaseResult(phase="prune", status="completed"), [])
            )
            report = await execute_hygiene_mop(
                graph_store=object(),
                activation_store=object(),
                search_index=object(),
                activation_cfg=ActivationConfig(),
                group_id="g",
                budget=100,
            )
        assert "evidence_adjudication" not in report["mop"]
        assert "replay" not in report["mop"]


class TestCueScanWatermark:
    def test_due_then_marked_then_not_due(self, engram_home: Path):
        from engram.hygiene_ops import cue_scan_due, mark_cue_scan_done

        assert cue_scan_due() is True
        mark_cue_scan_done()
        assert cue_scan_due() is False
        # A day later it becomes due again.
        assert cue_scan_due(now=time.time() + 25 * 3600) is True

    def test_zero_interval_always_due(self, engram_home: Path):
        from engram.hygiene_ops import cue_scan_due, mark_cue_scan_done

        mark_cue_scan_done()
        assert cue_scan_due(interval_hours=0) is True


class TestActivationPersistence:
    @pytest.mark.asyncio
    async def test_round_trip(self, tmp_path: Path):
        from engram.storage.memory.activation import MemoryActivationStore

        store = MemoryActivationStore()
        now = time.time()
        await store.record_access("e1", now - 100, group_id="g1")
        await store.record_access("e1", now - 10, group_id="g1")
        await store.record_access("e2", now - 5, group_id="g2")
        path = tmp_path / "snap.json"
        assert store.save_to_file(path) == 2

        fresh = MemoryActivationStore()
        assert fresh.load_from_file(path) == 2
        state = await fresh.get_activation("e1")
        assert state is not None
        assert len(state.access_history) == 2
        assert state.access_count == 2
        top = await fresh.get_top_activated(group_id="g2", limit=5)
        assert [eid for eid, _ in top] == ["e2"]

    def test_stale_snapshot_ignored(self, tmp_path: Path):
        import json

        from engram.storage.memory.activation import MemoryActivationStore

        path = tmp_path / "snap.json"
        path.write_text(
            json.dumps(
                {
                    "saved_at": time.time() - 30 * 86400,
                    "states": {"e1": {"node_id": "e1", "access_history": [1.0]}},
                }
            )
        )
        store = MemoryActivationStore()
        assert store.load_from_file(path, max_age_days=14.0) == 0

    @pytest.mark.asyncio
    async def test_live_state_wins_over_snapshot(self, tmp_path: Path):
        from engram.storage.memory.activation import MemoryActivationStore

        store = MemoryActivationStore()
        await store.record_access("e1", 100.0, group_id="g")
        path = tmp_path / "snap.json"
        store.save_to_file(path)

        fresh = MemoryActivationStore()
        await fresh.record_access("e1", 999.0, group_id="g")
        fresh.load_from_file(path)
        state = await fresh.get_activation("e1")
        assert state is not None and state.access_history == [999.0]

    def test_corrupt_file_returns_zero(self, tmp_path: Path):
        from engram.storage.memory.activation import MemoryActivationStore

        path = tmp_path / "snap.json"
        path.write_text("{not json")
        assert MemoryActivationStore().load_from_file(path) == 0
