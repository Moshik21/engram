"""Unit tests for Loop Steward LoopAdjustment control plane."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram.config import ActivationConfig
from engram.loop_adjustment import (
    LoopAdjustment,
    clamp_loop_adjustment,
    clear_active_adjustment,
    effective_activation_config,
    effective_phase_names,
    is_expired,
    load_active_adjustment,
    remaining_ttl_seconds,
    save_active_adjustment,
    stamp_applied,
    status_payload,
)


def _valid(**kwargs) -> LoopAdjustment:
    base = {
        "reason": "deferred high; should_mop",
        "regime": "debt_heavy",
        "ttl_hours": 12,
        "created_by": "harness:test",
        "max_risk": "low",
        "budgets": {"evidence_drain": 2000, "adjudication_limit": 400},
        "phase_boost": ["evidence_adjudication", "prune"],
        "phase_defer": ["dream", "graph_embed"],
        "intake": {"auto_extract_min_score": 0.9, "pattern_junk_reject": True},
    }
    base.update(kwargs)
    return LoopAdjustment.from_mapping(base)


class TestClamp:
    def test_clamp_rejects_empty_reason(self):
        result = clamp_loop_adjustment(_valid(reason=""))
        assert result.rejected
        assert result.reject_reason == "reason_required"

    def test_clamp_rejects_high_risk(self):
        result = clamp_loop_adjustment(_valid(max_risk="medium"))
        assert result.rejected
        assert result.reject_reason == "max_risk_must_be_low"

    def test_clamp_ttl_and_budget_caps(self):
        result = clamp_loop_adjustment(
            _valid(ttl_hours=100, budgets={"evidence_drain": 99999}),
            hard_caps={"evidence_drain": 5000},
        )
        assert not result.rejected
        assert result.adjustment.ttl_hours == 48
        assert result.adjustment.budgets["evidence_drain"] == 5000
        assert any("ttl_clamped_max" in w for w in result.warnings)
        assert any("budget_clamped_max" in w for w in result.warnings)

    def test_clamp_unknown_phases_dropped(self):
        result = clamp_loop_adjustment(
            _valid(phase_boost=["evidence_adjudication", "not_a_phase"], phase_defer=["dream"])
        )
        assert not result.rejected
        assert result.adjustment.phase_boost == ["evidence_adjudication"]
        assert result.adjustment.phase_defer == ["dream"]
        assert any("unknown_phase_boost:not_a_phase" in w for w in result.warnings)

    def test_pattern_junk_reject_cannot_disable(self):
        result = clamp_loop_adjustment(_valid(intake={"pattern_junk_reject": False}))
        assert result.adjustment.intake["pattern_junk_reject"] is True


class TestStoreAndExpiry:
    def test_save_load_replace_clear(self, tmp_path: Path):
        path = tmp_path / "loop-adjustment.json"
        audit = tmp_path / "loop-adjustments.jsonl"
        adj = stamp_applied(clamp_loop_adjustment(_valid()).adjustment)
        save_active_adjustment(adj, path=path, audit_path=audit)
        loaded = load_active_adjustment("default", path=path)
        assert loaded is not None
        assert loaded.regime == "debt_heavy"
        assert loaded.budgets["evidence_drain"] == 2000

        # Replace with single-active second apply
        adj2 = stamp_applied(
            clamp_loop_adjustment(
                _valid(reason="second", budgets={"evidence_drain": 1000})
            ).adjustment
        )
        save_active_adjustment(adj2, path=path, audit_path=audit)
        loaded2 = load_active_adjustment("default", path=path)
        assert loaded2 is not None
        assert loaded2.reason == "second"
        assert loaded2.budgets["evidence_drain"] == 1000

        assert clear_active_adjustment("default", path=path, audit_path=audit) is True
        assert load_active_adjustment("default", path=path) is None
        lines = audit.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line)["event"] for line in lines]
        assert events.count("apply") == 2
        assert "clear" in events

    def test_expiry_clears_on_load(self, tmp_path: Path):
        path = tmp_path / "loop-adjustment.json"
        adj = clamp_loop_adjustment(_valid(ttl_hours=1)).adjustment
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        adj.applied_at = past.isoformat()
        adj.expires_at = (past + timedelta(hours=1)).isoformat()
        path.write_text(json.dumps(adj.to_dict()), encoding="utf-8")
        assert is_expired(adj)
        assert load_active_adjustment("default", path=path) is None
        assert not path.exists()

    def test_status_payload_remaining_ttl(self, tmp_path: Path):
        path = tmp_path / "loop-adjustment.json"
        adj = stamp_applied(clamp_loop_adjustment(_valid(ttl_hours=12)).adjustment)
        save_active_adjustment(adj, path=path, audit_path=tmp_path / "a.jsonl")
        status = status_payload("default", path=path)
        assert status["active"] is True
        assert status["remaining_ttl_seconds"] > 0
        assert remaining_ttl_seconds(adj) > 0


class TestEffectiveOverlay:
    def test_effective_activation_config_maps_budgets_and_intake(self):
        base = ActivationConfig()
        adj = clamp_loop_adjustment(
            _valid(
                budgets={
                    "evidence_drain": 2000,
                    "already_exists": 500,
                    "stale_reject": 500,
                    "cue_hygiene": 300,
                    "adjudication_limit": 400,
                },
                intake={"auto_extract_min_score": 0.92},
            )
        ).adjustment
        eff = effective_activation_config(base, adj)
        assert eff is not base
        assert eff.consolidation_evidence_drain_max_per_cycle == 2000
        assert eff.consolidation_evidence_already_exists_max_per_cycle == 500
        assert eff.consolidation_evidence_stale_max_per_cycle == 500
        assert eff.consolidation_cue_hygiene_max_per_cycle == 300
        assert eff.consolidation_evidence_adjudication_limit == 400
        assert eff.worker_auto_capture_extract_score_floor == pytest.approx(0.92)
        # base unchanged
        assert base.consolidation_evidence_drain_max_per_cycle == 500

    def test_effective_phase_names_defer_and_boost(self):
        adj = clamp_loop_adjustment(
            _valid(
                phase_boost=["evidence_adjudication"],
                phase_defer=["dream", "graph_embed"],
            )
        ).adjustment
        due = {"merge", "evidence_adjudication", "dream", "graph_embed", "prune"}
        out = effective_phase_names(due, adj)
        assert out is not None
        assert "dream" not in out
        assert "graph_embed" not in out
        assert "evidence_adjudication" in out
        assert "prune" in out

    def test_effective_phase_names_full_cycle_defer(self):
        adj = clamp_loop_adjustment(
            _valid(phase_defer=["dream", "graph_embed"], phase_boost=[])
        ).adjustment
        out = effective_phase_names(None, adj)
        assert out is not None
        assert "dream" not in out
        assert "graph_embed" not in out
        assert "merge" in out

    def test_effective_phase_names_never_empty(self):
        adj = clamp_loop_adjustment(
            _valid(phase_defer=["merge", "prune"], phase_boost=[])
        ).adjustment
        due = {"merge", "prune"}
        out = effective_phase_names(due, adj)
        assert out == {"merge", "prune"}

    def test_no_adj_preserves_full_cycle_none(self):
        assert effective_phase_names(None, None) is None
        assert effective_phase_names({"merge"}, None) == {"merge"}
