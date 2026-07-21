"""Pre-flip closure — the flag-ON ranking view reads ONLY the usage store.

The M2 verifier measured 12/16 realistic tie-scenario inversions of
"used wins ties": flag-ON, the novelty reader (state.access_count) and the
rediscovery reader (state.access_history) still read HYGIENE history, so an
entity with surfaced-only history lost its novelty boost while contributing
nothing to u. This file pins the closure:

- Flag ON: the verifier's 16-scenario grid (hygiene access_count 0/5 on
  each side x history age x usage strength) — used wins ties 16/16.
- Flag ON: surfaced-only hygiene history is byte-invisible to ranking at
  DEFAULT exploration/rediscovery weights (one store, two views).
- Flag OFF: byte-equal to the pre-change scorer via golden hex floats
  captured from the shipped code at HEAD d6cfeb2 BEFORE the edit
  (scratchpad capture_scorer_goldens.py -> scorer_goldens_pre.json).
"""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.scorer import score_candidates

NOW = 1_700_000_000.0
DAY = 86400.0

WEAK = "weak"  # 1 mentioned-tier event (w=0.1), 90d old -> u ~= 0.006
STRONG = "strong"  # 5 used-tier events (w=0.3), ~600s old -> u ~= 0.23


def _score(cfg, candidates, states, **kwargs):
    return score_candidates(
        candidates=candidates,
        spreading_bonuses=kwargs.pop("spreading_bonuses", {}),
        hop_distances=kwargs.pop("hop_distances", {}),
        seed_node_ids=kwargs.pop("seed_node_ids", set()),
        activation_states=states,
        now=NOW,
        cfg=cfg,
        **kwargs,
    )


def _grid_state(
    node_id: str,
    hygiene_count: int,
    hygiene_age_days: float,
    usage: str | None,
) -> ActivationState:
    state = ActivationState(node_id=node_id)
    if hygiene_count:
        base = NOW - hygiene_age_days * DAY
        state.access_history = [base - 60.0 * k for k in range(hygiene_count)]
        state.access_count = hygiene_count
    if usage == WEAK:
        state.record_usage_event(NOW - 90.0 * DAY, 0.1)
    elif usage == STRONG:
        for k in range(5):
            state.record_usage_event(NOW - 600.0 - k, 0.3)
    return state


class TestUsedWinsTiesGrid:
    """The verifier's 16-scenario grid, flag-ON at DEFAULT weights.

    Default cfg deliberately keeps exploration_weight=0.05 and
    rediscovery_weight=0.02 live (the pre-fix inversions came from exactly
    these terms) and the default beta_route=0.10 — the weakest shipped
    usage gain, so a pass here is the conservative bound.
    """

    @pytest.mark.parametrize("usage", [WEAK, STRONG])
    @pytest.mark.parametrize("hygiene_age_days", [0.0, 60.0])
    @pytest.mark.parametrize("fresh_count", [0, 5])
    @pytest.mark.parametrize("used_count", [0, 5])
    def test_used_wins_equal_sem_tie(self, used_count, fresh_count, hygiene_age_days, usage):
        cfg = ActivationConfig(usage_ranking_enabled=True)
        states = {
            "used_ent": _grid_state("used_ent", used_count, hygiene_age_days, usage),
            "fresh_ent": _grid_state("fresh_ent", fresh_count, hygiene_age_days, None),
        }
        scored = _score(cfg, [("fresh_ent", 0.6), ("used_ent", 0.6)], states)
        assert scored[0].node_id == "used_ent"
        assert scored[0].score > scored[1].score


class TestHygieneInvisibleFlagOn:
    """One store, two views: flag-ON ranking never reads hygiene history."""

    def test_surfaced_only_history_scores_byte_equal_to_bare(self):
        cfg = ActivationConfig(usage_ranking_enabled=True)
        bare = _score(cfg, [("e1", 0.6)], {})
        for age_days in (0.0, 60.0):
            got = _score(cfg, [("e1", 0.6)], {"e1": _grid_state("e1", 5, age_days, None)})
            assert got[0].score == bare[0].score
            assert got[0].exploration_bonus == bare[0].exploration_bonus
            assert got[0].activation == 0.0

    def test_hygiene_history_on_top_of_usage_is_byte_invisible(self):
        cfg = ActivationConfig(usage_ranking_enabled=True)
        usage_only = _score(cfg, [("e1", 0.6)], {"e1": _grid_state("e1", 0, 0.0, STRONG)})
        with_hygiene = _score(cfg, [("e1", 0.6)], {"e1": _grid_state("e1", 5, 60.0, STRONG)})
        assert with_hygiene[0].score == usage_only[0].score
        assert with_hygiene[0].exploration_bonus == usage_only[0].exploration_bonus

    def test_never_used_multiplier_is_exactly_one(self):
        """Novel iff usage_weight_sum == 0: u == 0 -> multiplier exactly 1.0."""
        cfg = ActivationConfig(usage_ranking_enabled=True)
        composite = cfg.weight_semantic * 0.6 + cfg.exploration_weight * 0.6
        scored = _score(cfg, [("e1", 0.6)], {"e1": _grid_state("e1", 5, 60.0, None)})
        assert scored[0].score == composite


# --- Flag-OFF golden byte-equality -------------------------------------------
# Captured from the CURRENT code BEFORE this change (HEAD d6cfeb2) with
# capture_scorer_goldens.py; float.hex() round-trips exactly, so `==` below
# is byte equality. Any flag-OFF behavior drift in score_candidates fails here.

_GOLDEN_ORDER = [
    "e_recent5",
    "e_heavy",
    "e_empty",
    "e_dormant5",
    "e_cs",
    "e_mixed",
    "e_usage_only",
    "e_nostate",
    "e_zero_sem",
]

# node_id -> (score, activation, exploration_bonus) as hex floats
_GOLDEN = {
    "default": {
        "e_recent5": ("0x1.49564aeec183bp-1", "0x1.b9c096ac7d9cdp-1", "0x1.9ad0fefd02037p-7"),
        "e_heavy": ("0x1.1d526250c72cfp-1", "0x1.9e1603a0f0eb7p-1", "0x1.4cd71aabdf6dbp-8"),
        "e_empty": ("0x1.dc28f5c28f5c1p-2", "0x0.0p+0", "0x1.1eb851eb851ebp-5"),
        "e_dormant5": ("0x1.95cdf32cfbafbp-2", "0x1.c8c1f71e9a749p-3", "0x1.796fbf73cdc10p-6"),
        "e_cs": ("0x1.8000000000000p-2", "0x0.0p+0", "0x1.999999999999ap-6"),
        "e_mixed": ("0x1.7563cf49471d6p-2", "0x1.0aa809b2a7ff2p-2", "0x1.4017af117fee1p-6"),
        "e_usage_only": ("0x1.55c28f5c28f5bp-2", "0x0.0p+0", "0x1.1eb851eb851ebp-5"),
        "e_nostate": ("0x1.428f5c28f5c28p-2", "0x0.0p+0", "0x1.1eb851eb851ebp-5"),
        "e_zero_sem": ("0x1.9aa0ea1e4538dp-6", "0x1.9aa0ea1e4538dp-4", "0x0.0p+0"),
    },
    "maturation": {
        "e_recent5": ("0x1.49564aeec183bp-1", "0x1.b9c096ac7d9cdp-1", "0x1.9ad0fefd02037p-7"),
        "e_heavy": ("0x1.2dcd2a36c9433p-1", "0x1.e0012338f9448p-1", "0x1.4cd71aabdf6dbp-8"),
        "e_empty": ("0x1.dc28f5c28f5c1p-2", "0x0.0p+0", "0x1.1eb851eb851ebp-5"),
        "e_dormant5": ("0x1.c73f1ef9ce8dcp-2", "0x1.aa25aac298b29p-2", "0x1.796fbf73cdc10p-6"),
        "e_cs": ("0x1.8000000000000p-2", "0x0.0p+0", "0x1.999999999999ap-6"),
        "e_mixed": ("0x1.7563cf49471d6p-2", "0x1.0aa809b2a7ff2p-2", "0x1.4017af117fee1p-6"),
        "e_usage_only": ("0x1.55c28f5c28f5bp-2", "0x0.0p+0", "0x1.1eb851eb851ebp-5"),
        "e_nostate": ("0x1.428f5c28f5c28p-2", "0x0.0p+0", "0x1.1eb851eb851ebp-5"),
        "e_zero_sem": ("0x1.9aa0ea1e4538dp-6", "0x1.9aa0ea1e4538dp-4", "0x0.0p+0"),
    },
}


def _golden_states() -> dict[str, ActivationState]:
    states: dict[str, ActivationState] = {}

    states["e_empty"] = ActivationState(node_id="e_empty")

    s = ActivationState(node_id="e_recent5")
    s.access_history = [NOW - 60.0 * k for k in range(1, 6)]
    s.access_count = 5
    states["e_recent5"] = s

    s = ActivationState(node_id="e_dormant5")
    s.access_history = [NOW - 60.0 * DAY - 3600.0 * k for k in range(5)]
    s.access_count = 5
    states["e_dormant5"] = s

    s = ActivationState(node_id="e_heavy")
    s.access_history = [NOW - 3600.0 * k for k in range(1, 51)]
    s.access_count = 50
    states["e_heavy"] = s

    s = ActivationState(node_id="e_cs")
    s.consolidated_strength = 2.0
    states["e_cs"] = s

    s = ActivationState(node_id="e_usage_only")
    for k in range(3):
        s.record_usage_event(NOW - 600.0 - k, 0.3)
    states["e_usage_only"] = s

    s = ActivationState(node_id="e_mixed")
    s.access_history = [NOW - 30.0 * DAY - 60.0 * k for k in range(5)]
    s.access_count = 5
    for k in range(3):
        s.record_usage_event(NOW - 600.0 - k, 0.3)
    states["e_mixed"] = s

    s = ActivationState(node_id="e_zero_sem")
    s.access_history = [NOW - 60.0 * DAY]
    s.access_count = 1
    states["e_zero_sem"] = s

    return states


def _golden_run(cfg: ActivationConfig):
    return score_candidates(
        candidates=[
            ("e_nostate", 0.7),
            ("e_empty", 0.7),
            ("e_recent5", 0.7),
            ("e_dormant5", 0.7),
            ("e_heavy", 0.5),
            ("e_cs", 0.5),
            ("e_usage_only", 0.7),
            ("e_mixed", 0.7),
            ("e_zero_sem", 0.0),
        ],
        spreading_bonuses={"e_recent5": 0.4, "e_cs": 1.7},
        hop_distances={"e_recent5": 1, "e_dormant5": 2, "e_usage_only": 3},
        seed_node_ids={"e_empty", "e_heavy"},
        activation_states=_golden_states(),
        now=NOW,
        cfg=cfg,
        entity_attributes={
            "e_heavy": {"mat_tier": "semantic"},
            "e_dormant5": {"mat_tier": "transitional"},
        },
    )


class TestFlagOffGoldenByteEquality:
    @pytest.mark.parametrize("label", ["default", "maturation"])
    def test_flag_off_byte_equal_to_pre_change(self, label):
        cfg = (
            ActivationConfig()
            if label == "default"
            else ActivationConfig(memory_maturation_enabled=True)
        )
        assert cfg.usage_ranking_enabled is False  # shipped default
        results = _golden_run(cfg)
        assert [r.node_id for r in results] == _GOLDEN_ORDER
        for r in results:
            score_hex, act_hex, expl_hex = _GOLDEN[label][r.node_id]
            assert r.score == float.fromhex(score_hex), r.node_id
            assert r.activation == float.fromhex(act_hex), r.node_id
            assert r.exploration_bonus == float.fromhex(expl_hex), r.node_id
