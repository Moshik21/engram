"""Shell honors LoopAdjustment: phase bias + per-cycle cfg overlay."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.scheduler import ConsolidationScheduler
from engram.loop_adjustment import (
    LoopAdjustment,
    clamp_loop_adjustment,
    effective_activation_config,
    effective_phase_names,
    load_active_adjustment,
    save_active_adjustment,
    stamp_applied,
)
from engram.models.consolidation import PhaseResult


class _RecordingPhase:
    def __init__(self, name: str) -> None:
        self.name = name
        self.seen_cfg: list[ActivationConfig] = []

    def required_graph_store_methods(self, cfg):
        return set()

    async def execute(self, **kwargs):
        self.seen_cfg.append(kwargs["cfg"])
        return PhaseResult(phase=self.name, status="success"), []


@pytest.mark.asyncio
async def test_engine_run_cycle_uses_cfg_override_and_phase_filter(tmp_path: Path):
    base = ActivationConfig()
    adj = stamp_applied(
        clamp_loop_adjustment(
            LoopAdjustment.from_mapping(
                {
                    "reason": "debt recovery night",
                    "regime": "debt_heavy",
                    "ttl_hours": 12,
                    "max_risk": "low",
                    "budgets": {"evidence_drain": 2000, "adjudication_limit": 400},
                    "phase_defer": ["dream", "graph_embed"],
                    "phase_boost": ["evidence_adjudication"],
                }
            )
        ).adjustment
    )
    path = tmp_path / "loop-adjustment.json"
    save_active_adjustment(adj, path=path, audit_path=tmp_path / "a.jsonl")

    loaded = load_active_adjustment("default", path=path)
    assert loaded is not None
    phases = effective_phase_names(
        {"merge", "evidence_adjudication", "dream", "graph_embed"},
        loaded,
    )
    cfg_eff = effective_activation_config(base, loaded)
    assert cfg_eff.consolidation_evidence_drain_max_per_cycle == 2000
    assert "dream" not in (phases or set())
    assert "graph_embed" not in (phases or set())
    assert "merge" in (phases or set())
    assert "evidence_adjudication" in (phases or set())

    # Real engine path with lightweight stub phases (only names we request)
    merge = _RecordingPhase("merge")
    dream = _RecordingPhase("dream")
    embed = _RecordingPhase("graph_embed")
    evidence = _RecordingPhase("evidence_adjudication")

    engine = ConsolidationEngine(
        graph_store=MagicMock(),
        activation_store=MagicMock(),
        search_index=MagicMock(),
        cfg=base,
        consolidation_store=None,
    )
    # Replace phase list with stubs (preserve names engine knows)
    engine._phases = [merge, evidence, dream, embed]  # type: ignore[assignment]
    engine._capabilities.validate = MagicMock()  # type: ignore[method-assign]

    cycle = await engine.run_cycle(
        group_id="default",
        trigger="test",
        phase_names=phases,
        cfg=cfg_eff,
        dry_run=True,
    )
    assert cycle is not None
    ran = {p.name for p in (merge, evidence, dream, embed) if p.seen_cfg}
    assert "dream" not in ran
    assert "graph_embed" not in ran
    assert "merge" in ran
    assert "evidence_adjudication" in ran
    # cfg overlay reached executed phases
    for phase in (merge, evidence):
        assert phase.seen_cfg
        assert phase.seen_cfg[0].consolidation_evidence_drain_max_per_cycle == 2000
        assert phase.seen_cfg[0] is not base


def test_scheduler_overlay_biases_full_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_FILE", str(tmp_path / "loop-adjustment.json"))
    monkeypatch.setenv(
        "ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE", str(tmp_path / "loop-adjustments.jsonl")
    )
    adj = stamp_applied(
        clamp_loop_adjustment(
            LoopAdjustment.from_mapping(
                {
                    "reason": "pressure recovery",
                    "regime": "debt_heavy",
                    "ttl_hours": 6,
                    "max_risk": "low",
                    "budgets": {"evidence_drain": 1500},
                    "phase_defer": ["dream", "graph_embed"],
                }
            )
        ).adjustment
    )
    save_active_adjustment(
        adj,
        path=tmp_path / "loop-adjustment.json",
        audit_path=tmp_path / "loop-adjustments.jsonl",
    )

    engine = MagicMock()
    engine.is_running = False
    engine.run_cycle = AsyncMock()
    cfg = ActivationConfig()
    sched = ConsolidationScheduler(engine=engine, cfg=cfg, default_group_id="default")
    phases, cfg_eff = sched._loop_steward_overlay(None)
    assert phases is not None
    assert "dream" not in phases
    assert "graph_embed" not in phases
    assert cfg_eff.consolidation_evidence_drain_max_per_cycle == 1500
    assert cfg is not cfg_eff


def test_mop_knob_budgets_divergent_per_key():
    from engram.loop_adjustment import mop_knob_budgets

    adj = clamp_loop_adjustment(
        LoopAdjustment.from_mapping(
            {
                "reason": "divergent knobs",
                "ttl_hours": 12,
                "max_risk": "low",
                "budgets": {
                    "evidence_drain": 2000,
                    "already_exists": 777,
                    "stale_reject": 333,
                    "cue_hygiene": 111,
                },
            }
        )
    ).adjustment
    # CLI floor 50; each steward key raises independently
    got = mop_knob_budgets(50, adj)
    assert got == {
        "evidence_drain": 2000,
        "already_exists": 777,
        "stale_reject": 333,
        "cue_hygiene": 111,
    }
    # Without adj, all knobs stay at CLI floor
    assert mop_knob_budgets(50, None) == {
        "evidence_drain": 50,
        "already_exists": 50,
        "stale_reject": 50,
        "cue_hygiene": 50,
    }
    # Partial steward: only set keys raise
    partial = clamp_loop_adjustment(
        LoopAdjustment.from_mapping(
            {
                "reason": "partial",
                "ttl_hours": 12,
                "max_risk": "low",
                "budgets": {"cue_hygiene": 900},
            }
        )
    ).adjustment
    part = mop_knob_budgets(100, partial)
    assert part["cue_hygiene"] == 900
    assert part["evidence_drain"] == 100
    assert part["already_exists"] == 100


@pytest.mark.asyncio
async def test_mop_cli_dry_run_uses_per_knob_steward_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Drive real run_hygiene_command mop path; assert per-knob limits on drains."""
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_FILE", str(tmp_path / "loop-adjustment.json"))
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE", str(tmp_path / "a.jsonl"))
    adj = stamp_applied(
        clamp_loop_adjustment(
            LoopAdjustment.from_mapping(
                {
                    "reason": "mop per-knob",
                    "ttl_hours": 12,
                    "max_risk": "low",
                    "budgets": {
                        "evidence_drain": 2000,
                        "already_exists": 777,
                        "stale_reject": 333,
                        "cue_hygiene": 111,
                    },
                }
            )
        ).adjustment
    )
    save_active_adjustment(
        adj,
        path=tmp_path / "loop-adjustment.json",
        audit_path=tmp_path / "a.jsonl",
    )

    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from engram.consolidation.hygiene_debt import HygieneDebtSnapshot
    from engram.models.consolidation import PhaseResult

    captured: dict = {}

    async def fake_reject_junk(*args, **kwargs):
        captured["junk_max"] = kwargs.get("max_reject")
        return {"rejected": 0, "kept": 0, "errors": 0, "total": 0, "by_reason": {}}

    async def fake_reject_rows(*args, **kwargs):
        prefix = str(kwargs.get("reason_prefix") or "")
        captured[f"limit:{prefix}"] = len(kwargs.get("rows") or [])
        return {"rejected": 0, "errors": 0, "total": 0, "by_reason": {}}

    async def fake_cue(*args, **kwargs):
        captured["cue_max"] = kwargs.get("max_per_cycle")
        return SimpleNamespace(
            demoted=0,
            eligible=0,
            scanned=0,
            to_dict=lambda: {"demoted": 0, "eligible": 0, "scanned": 0},
        )

    async def fake_load_deferred(graph, group_id):
        return []

    async def fake_debt(graph, group_id):
        return HygieneDebtSnapshot()

    def fake_select_redundant(rows, existing, *, limit=None):
        captured["already_limit"] = limit
        return []

    def fake_select_stale(*args, **kwargs):
        captured["stale_limit"] = kwargs.get("limit")
        return []

    class _FakePhase:
        async def execute(self, **kwargs):
            captured["prune_max"] = kwargs["cfg"].consolidation_prune_max_per_cycle
            return PhaseResult(phase="prune", status="success"), []

    graph = MagicMock()
    graph.initialize = AsyncMock()
    args = SimpleNamespace(
        action="mop",
        group_id="default",
        mode="lite",
        helix_data_dir=None,
        dry_run=True,
        budget=50,
        format="json",
    )

    with (
        patch(
            "engram.storage.bootstrap.create_local_runtime_stores",
            return_value=(graph, MagicMock(), MagicMock()),
        ),
        patch(
            "engram.storage.resolver.resolve_mode",
            new_callable=AsyncMock,
            return_value="lite",
        ),
        patch(
            "engram.storage.bootstrap.initialize_search_index_for_graph",
            new_callable=AsyncMock,
        ),
        patch("engram.storage.bootstrap.close_if_supported", new_callable=AsyncMock),
        patch(
            "engram.consolidation.hygiene_debt.collect_hygiene_debt_from_store",
            fake_debt,
        ),
        patch(
            "engram.consolidation.evidence_drain.load_deferred_evidence",
            fake_load_deferred,
        ),
        patch(
            "engram.consolidation.evidence_drain.reject_junk_evidence",
            side_effect=fake_reject_junk,
        ),
        patch(
            "engram.consolidation.evidence_drain.reject_evidence_rows",
            side_effect=fake_reject_rows,
        ),
        patch(
            "engram.consolidation.evidence_drain.select_redundant_entity_evidence",
            fake_select_redundant,
        ),
        patch(
            "engram.consolidation.evidence_drain.select_stale_low_value_evidence",
            fake_select_stale,
        ),
        patch(
            "engram.consolidation.cue_hygiene.run_cue_hygiene",
            side_effect=fake_cue,
        ),
        patch("engram.consolidation.phases.prune.PrunePhase", _FakePhase),
        # Exclusivity guard: pretend no shell is running so the CLI proceeds.
        patch("engram.brain_runtime.shell_is_healthy", return_value=False),
        patch("engram.brain_runtime.serve_process_alive", return_value=False),
    ):
        monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
        from engram.hygiene_cli import run_hygiene_command

        rc = await run_hygiene_command(args)
        assert rc == 0

    assert captured.get("junk_max") == 2000, captured
    assert captured.get("already_limit") == 777, captured
    assert captured.get("stale_limit") == 333, captured
    assert captured.get("cue_max") == 111, captured
    assert captured.get("prune_max") == 2000, captured


@pytest.mark.asyncio
async def test_worker_routing_honors_steward_intake_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """EpisodeWorkerProjectionRouter must use active LoopAdjustment intake floor."""
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_FILE", str(tmp_path / "loop-adjustment.json"))
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE", str(tmp_path / "a.jsonl"))
    adj = stamp_applied(
        clamp_loop_adjustment(
            LoopAdjustment.from_mapping(
                {
                    "reason": "raise auto-capture floor",
                    "ttl_hours": 12,
                    "max_risk": "low",
                    "intake": {"auto_extract_min_score": 0.95},
                }
            )
        ).adjustment
    )
    save_active_adjustment(
        adj,
        path=tmp_path / "loop-adjustment.json",
        audit_path=tmp_path / "a.jsonl",
    )

    from types import SimpleNamespace

    from engram.ingestion.worker_routing import EpisodeWorkerProjectionRouter
    from engram.retrieval.triage_policy import TriageDecision

    episode = SimpleNamespace(source="auto:claude", projection_state="queued")
    graph = SimpleNamespace(
        get_episode_by_id=AsyncMock(return_value=episode),
        update_episode=AsyncMock(),
        update_episode_cue=AsyncMock(),
    )

    router = EpisodeWorkerProjectionRouter(graph, ActivationConfig())
    # Boot default floor is 0.85; steward raises to 0.95
    assert router.auto_capture_extract_score_floor("default") == pytest.approx(0.95)
    assert router.effective_cfg("default").worker_auto_capture_extract_score_floor == pytest.approx(
        0.95
    )

    decision = TriageDecision(
        action="extract",
        score=0.90,
        base_score=0.90,
        threshold_band="test",
        decision_source="test",
    )
    should_project = await router.route_decision("ep1", decision, "default")
    # 0.90 < 0.95 steward floor → cue-only, do not project
    assert should_project is False
    graph.update_episode.assert_awaited()

    # Score above steward floor projects
    decision_hi = TriageDecision(
        action="extract",
        score=0.96,
        base_score=0.96,
        threshold_band="test",
        decision_source="test",
    )
    assert await router.route_decision("ep2", decision_hi, "default") is True
