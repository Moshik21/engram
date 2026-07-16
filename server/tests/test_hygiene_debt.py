"""Hygiene debt scoreboard and pressure alignment."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.consolidation.hygiene_debt import (
    CueHygieneCandidate,
    HygieneDebtSnapshot,
    LowValueEntityCandidate,
    debt_pressure_contribution,
    debt_should_trigger_mop,
    hygiene_debt_from_stats,
    select_cue_hygiene_candidates,
    select_low_value_prune_candidates,
)
from engram.consolidation.pressure import ConsolidationPressure


def test_scoreboard_fields_from_stats() -> None:
    stats = {
        "episodes": 8000,
        "entities": 900,
        "cue_metrics": {
            "cue_count": 8300,
            "cue_near_miss_count": 64,
        },
        "projection_metrics": {
            "state_counts": {"projected": 4000, "cue_only": 3000},
        },
        "adjudication_metrics": {
            "evidence_status_counts": {"deferred": 19000, "pending": 400},
            "open_request_count": 390,
        },
        "orphan_candidates": 12,
    }
    debt = hygiene_debt_from_stats(stats)
    assert debt.deferred_evidence == 19000
    assert debt.pending_evidence == 400
    assert debt.cue_only_episodes == 3000
    assert debt.cue_count == 8300
    assert debt.near_miss_count == 64
    assert debt.open_adjudication == 390
    assert debt.orphan_candidates == 12
    assert debt.open_work == 19000 + 400 + 390
    payload = debt.to_dict()
    for key in (
        "deferred_evidence",
        "cue_only_episodes",
        "near_miss_count",
        "open_adjudication",
        "orphan_candidates",
        "open_work",
        "total_debt_units",
    ):
        assert key in payload


def test_debt_pressure_rises_with_deferred_sludge() -> None:
    clean = HygieneDebtSnapshot()
    dirty = HygieneDebtSnapshot(
        deferred_evidence=10000,
        cue_only_episodes=3000,
        near_miss_count=64,
        open_adjudication=400,
    )
    clean_p = debt_pressure_contribution(clean)
    dirty_p = debt_pressure_contribution(dirty)
    assert clean_p == 0.0
    assert dirty_p > 100.0  # exceeds default threshold
    assert debt_should_trigger_mop(dirty, pressure_threshold=100.0)
    assert not debt_should_trigger_mop(clean, pressure_threshold=100.0)


def test_pressure_compute_includes_hygiene_debt() -> None:
    cfg = ActivationConfig(
        consolidation_pressure_weight_episode=1.0,
        consolidation_pressure_weight_entity=0.5,
        consolidation_pressure_weight_near_miss=2.0,
        consolidation_pressure_time_factor=0.0,
        consolidation_pressure_weight_deferred=0.02,
    )
    p = ConsolidationPressure(episodes_since_last=2, entities_created=2)
    base = p.compute(cfg)
    with_debt = p.compute(
        cfg,
        hygiene_debt=HygieneDebtSnapshot(deferred_evidence=10000),
    )
    assert with_debt > base
    assert with_debt >= 200.0  # 10000 * 0.02


def test_cue_hygiene_selects_never_used_old_cues() -> None:
    cues = [
        CueHygieneCandidate("ep1", hit_count=0, surfaced_count=0, cue_text="old", age_days=20),
        CueHygieneCandidate("ep2", hit_count=3, surfaced_count=1, cue_text="hot", age_days=40),
        CueHygieneCandidate("ep3", hit_count=0, surfaced_count=0, cue_text="young", age_days=2),
        CueHygieneCandidate("ep4", hit_count=0, surfaced_count=0, cue_text="", age_days=30),
    ]
    selected = select_cue_hygiene_candidates(cues, max_age_days=14.0, limit=10)
    assert [c.episode_id for c in selected] == ["ep1"]


def test_low_value_prune_skips_identity_core() -> None:
    entities = [
        LowValueEntityCandidate("c1", "Concept", access_count=0, age_days=40, identity_core=False),
        LowValueEntityCandidate("id1", "Concept", access_count=0, age_days=40, identity_core=True),
        LowValueEntityCandidate("a1", "Artifact", access_count=0, age_days=40, identity_core=False),
        LowValueEntityCandidate("p1", "Person", access_count=0, age_days=40, identity_core=False),
    ]
    selected = select_low_value_prune_candidates(entities, min_age_days=30, limit=10)
    ids = {e.entity_id for e in selected}
    assert ids == {"c1", "a1"}
    assert "id1" not in ids
    assert "p1" not in ids
