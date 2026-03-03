"""Tests for the retrieval scorer."""

from __future__ import annotations

import math
import time

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.scorer import ScoredResult, score_candidates


class TestScoreCandidates:
    def test_no_activation_semantic_only(self):
        """Without activation, score = w_sem * semantic only."""
        cfg = ActivationConfig(
            weight_semantic=0.50,
            weight_activation=0.50,
            weight_spreading=0.0,
            weight_edge_proximity=0.00,
            exploration_weight=0.0,
        )
        now = time.time()
        candidates = [("ent_1", 0.8)]
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
        )
        assert len(scored) == 1
        assert abs(scored[0].score - 0.50 * 0.8) < 0.01
        assert scored[0].activation == 0.0
        assert scored[0].edge_proximity == 0.0

    def test_activation_boosts_rank(self):
        """Recent entity ranks higher than cold entity with equal semantic."""
        cfg = ActivationConfig()
        now = time.time()
        candidates = [("cold", 0.7), ("hot", 0.7)]
        states = {
            "cold": ActivationState(node_id="cold", access_history=[now - 604800]),
            "hot": ActivationState(node_id="hot", access_history=[now - 10]),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        hot_result = next(r for r in scored if r.node_id == "hot")
        cold_result = next(r for r in scored if r.node_id == "cold")
        assert hot_result.score > cold_result.score

    def test_seed_node_edge_proximity(self):
        """Seed node gets edge_proximity = 1.0."""
        cfg = ActivationConfig()
        now = time.time()
        candidates = [("seed_ent", 0.6)]
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids={"seed_ent"},
            activation_states={},
            now=now,
            cfg=cfg,
        )
        assert scored[0].edge_proximity == 1.0

    def test_one_hop_edge_proximity(self):
        """1-hop neighbor gets edge_proximity = 0.5."""
        cfg = ActivationConfig()
        now = time.time()
        candidates = [("neighbor", 0.5)]
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={"neighbor": 1},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
        )
        assert abs(scored[0].edge_proximity - 0.5) < 0.01

    def test_unreachable_edge_proximity(self):
        """Unreachable node gets edge_proximity = 0.0."""
        cfg = ActivationConfig()
        now = time.time()
        candidates = [("isolated", 0.5)]
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
        )
        assert scored[0].edge_proximity == 0.0

    def test_spreading_bonus_clamped(self):
        """Spreading bonus clamped to [0, 1]."""
        cfg = ActivationConfig()
        now = time.time()
        candidates = [("ent_1", 0.5)]
        states = {
            "ent_1": ActivationState(
                node_id="ent_1", access_history=[now - 1]  # very recent = high act
            ),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={"ent_1": 5.0},  # huge bonus
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        assert scored[0].spreading <= 1.0

    def test_composite_weights(self):
        """Composite weights match explicit config (0.50/0.25/0.10/0.15)."""
        cfg = ActivationConfig(
            weight_semantic=0.50,
            weight_activation=0.25,
            weight_spreading=0.10,
            weight_edge_proximity=0.15,
            exploration_weight=0.0,
        )
        now = time.time()
        candidates = [("ent_1", 0.6)]
        states = {
            "ent_1": ActivationState(
                node_id="ent_1", access_history=[now - 10]
            ),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={"ent_1": 0.4},
            hop_distances={},
            seed_node_ids={"ent_1"},
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        r = scored[0]
        expected = (
            0.50 * r.semantic_similarity
            + 0.25 * r.activation
            + 0.10 * r.spreading
            + 0.15 * r.edge_proximity
        )
        assert abs(r.score - expected) < 0.001

    def test_ranking_order(self):
        """high sem + high act > high sem + low act."""
        cfg = ActivationConfig()
        now = time.time()
        candidates = [("low_act", 0.8), ("high_act", 0.8)]
        states = {
            "low_act": ActivationState(node_id="low_act", access_history=[now - 604800]),
            "high_act": ActivationState(node_id="high_act", access_history=[now - 5]),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        assert scored[0].node_id == "high_act"

    def test_top_n_limit(self):
        """Results list can be sliced to top-N."""
        cfg = ActivationConfig()
        now = time.time()
        candidates = [(f"ent_{i}", 0.5 + i * 0.01) for i in range(20)]
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
        )
        top_5 = scored[:5]
        assert len(top_5) == 5
        # Should be in descending order
        for i in range(len(top_5) - 1):
            assert top_5[i].score >= top_5[i + 1].score

    def test_scored_result_has_breakdown(self):
        """ScoredResult contains breakdown for debugging."""
        cfg = ActivationConfig()
        now = time.time()
        candidates = [("ent_1", 0.7)]
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
        )
        r = scored[0]
        assert isinstance(r, ScoredResult)
        assert hasattr(r, "semantic_similarity")
        assert hasattr(r, "activation")
        assert hasattr(r, "edge_proximity")
        assert hasattr(r, "spreading")
        assert hasattr(r, "exploration_bonus")
        assert hasattr(r, "score")

    # ---- Fix 1: Spreading-only entity scoring ----

    def test_spreading_only_entity_can_score(self):
        """Entity with sem_sim=0.0 but activation+spreading+edge_prox can score > 0."""
        cfg = ActivationConfig(
            weight_semantic=0.50,
            weight_activation=0.25,
            weight_spreading=0.15,
            weight_edge_proximity=0.10,
            exploration_weight=0.0,
        )
        now = time.time()
        # Spreading-discovered entity: sem_sim=0.0, but has activation and is 1-hop
        candidates = [("spread_ent", 0.0)]
        states = {
            "spread_ent": ActivationState(
                node_id="spread_ent", access_history=[now - 10],
            ),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={"spread_ent": 0.3},
            hop_distances={"spread_ent": 1},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        assert len(scored) == 1
        r = scored[0]
        assert r.semantic_similarity == 0.0
        assert r.activation > 0.0
        assert r.spreading == 0.3
        assert r.edge_proximity == 0.5
        assert r.score > 0.0

    def test_spreading_only_entity_max_score(self):
        """Max score for spread-only entity = w_act + w_spread + w_edge."""
        cfg = ActivationConfig(
            weight_semantic=0.50,
            weight_activation=0.20,
            weight_spreading=0.15,
            weight_edge_proximity=0.15,
            exploration_weight=0.0,
        )
        now = time.time()
        # Max activation and seed (edge_prox=1.0), huge spreading bonus
        candidates = [("max_spread", 0.0)]
        states = {
            "max_spread": ActivationState(
                node_id="max_spread", access_history=[now - 1],
            ),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={"max_spread": 5.0},
            hop_distances={},
            seed_node_ids={"max_spread"},
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        r = scored[0]
        # sem=0.0, act=high, spread=1.0 (clamped), edge=1.0 (seed)
        max_possible = 0.20 * 1.0 + 0.15 * 1.0 + 0.15 * 1.0
        assert r.score <= max_possible + 0.001

    # ---- Fix 2: Spreading independent of activation ----

    def test_spreading_independent_of_activation(self):
        """Hot entity with spreading gets both signals independently."""
        cfg = ActivationConfig(
            weight_semantic=0.30,
            weight_activation=0.25,
            weight_spreading=0.15,
            weight_edge_proximity=0.10,
            exploration_weight=0.0,
        )
        now = time.time()
        candidates = [("hot_ent", 0.5)]
        states = {
            "hot_ent": ActivationState(
                node_id="hot_ent", access_history=[now - 5],  # very recent
            ),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={"hot_ent": 0.6},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        r = scored[0]
        # Activation is base_act (not base_act + spread)
        assert r.activation < 1.0  # base activation, not clamped sum
        assert r.spreading == 0.6  # independent spreading signal
        # Score includes both signals independently
        expected = (
            0.30 * r.semantic_similarity
            + 0.25 * r.activation
            + 0.15 * r.spreading
            + 0.10 * r.edge_proximity
        )
        assert abs(r.score - expected) < 0.001

    def test_spreading_signal_normalized(self):
        """Spreading bonus > 1.0 is clamped to 1.0."""
        cfg = ActivationConfig(
            weight_spreading=0.15,
            exploration_weight=0.0,
        )
        now = time.time()
        candidates = [("ent_1", 0.5)]
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={"ent_1": 5.0},  # way over 1.0
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
        )
        assert scored[0].spreading == 1.0

    # ---- Fix 5: Exploration bonus ----

    def test_exploration_bonus_low_access(self):
        """Bonus applies when access_count < threshold."""
        cfg = ActivationConfig(
            exploration_threshold=5,
            exploration_weight=0.05,
        )
        now = time.time()
        candidates = [("new_ent", 0.6)]
        states = {
            "new_ent": ActivationState(
                node_id="new_ent", access_history=[now - 100],
                access_count=2,
            ),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        r = scored[0]
        expected_bonus = 0.05 * 0.6 * (1.0 / (1.0 + math.log1p(2)))
        assert abs(r.exploration_bonus - expected_bonus) < 0.001
        assert r.exploration_bonus > 0.0

    def test_exploration_bonus_high_access_diminished(self):
        """Bonus is diminished but nonzero for high access_count (smooth decay)."""
        cfg = ActivationConfig(
            exploration_threshold=5,
            exploration_weight=0.05,
            rediscovery_weight=0.0,  # isolate novelty signal
        )
        now = time.time()
        candidates = [("established", 0.6)]
        states = {
            "established": ActivationState(
                node_id="established", access_history=[now - 100],
                access_count=10,
            ),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        r = scored[0]
        # Smooth formula: novelty = 1/(1+log1p(10)) ≈ 0.29 → bonus ≈ 0.05 * 0.6 * 0.29
        assert r.exploration_bonus > 0.0
        # But significantly less than low-access bonus
        low_access_bonus = 0.05 * 0.6 * (1.0 / (1.0 + math.log1p(2)))
        assert r.exploration_bonus < low_access_bonus

    def test_exploration_bonus_zero_semantic(self):
        """No bonus when sem_sim = 0."""
        cfg = ActivationConfig(
            exploration_threshold=5,
            exploration_weight=0.05,
        )
        now = time.time()
        candidates = [("zero_sem", 0.0)]
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
        )
        assert scored[0].exploration_bonus == 0.0

    def test_exploration_bonus_no_state(self):
        """Bonus applies for brand-new entity (0 accesses, no state)."""
        cfg = ActivationConfig(
            exploration_threshold=5,
            exploration_weight=0.05,
        )
        now = time.time()
        candidates = [("brand_new", 0.8)]
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
        )
        r = scored[0]
        # access_count=0 → bonus = 0.05 * 0.8 * (1 / (1 + log1p(0))) = 0.05 * 0.8 * 1.0
        expected_bonus = 0.05 * 0.8 * 1.0
        assert abs(r.exploration_bonus - expected_bonus) < 0.001
        assert r.exploration_bonus > 0.0

    def test_exploration_bonus_boosts_new_entity(self):
        """Newcomer gets much larger exploration bonus than established entity."""
        cfg = ActivationConfig(
            weight_semantic=0.50,
            weight_activation=0.50,
            weight_edge_proximity=0.00,
            exploration_threshold=5,
            exploration_weight=0.10,
            rediscovery_weight=0.0,  # isolate novelty signal
        )
        now = time.time()
        candidates = [("veteran", 0.5), ("newcomer", 0.5)]
        states = {
            # Veteran: many accesses but old, so low activation
            "veteran": ActivationState(
                node_id="veteran", access_history=[now - 604800],
                access_count=20,
            ),
            # Newcomer: 1 access, very recent
            "newcomer": ActivationState(
                node_id="newcomer", access_history=[now - 100],
                access_count=1,
            ),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        newcomer = next(r for r in scored if r.node_id == "newcomer")
        veteran = next(r for r in scored if r.node_id == "veteran")
        # Both get bonus (smooth decay), but newcomer's is much larger
        assert newcomer.exploration_bonus > 0.0
        assert veteran.exploration_bonus > 0.0
        assert newcomer.exploration_bonus > veteran.exploration_bonus * 2

    def test_exploration_bonus_diminishes(self):
        """Bonus decreases smoothly as access_count increases, even past old threshold."""
        cfg = ActivationConfig(
            exploration_threshold=5,
            exploration_weight=0.05,
            rediscovery_weight=0.0,  # isolate novelty signal
            weight_semantic=0.50,
            weight_activation=0.50,
            weight_edge_proximity=0.00,
        )
        now = time.time()
        bonuses = []
        # Include values past the old threshold (5) to confirm smooth decay
        for ac in [0, 1, 3, 5, 8, 15, 50]:
            candidates = [(f"ent_{ac}", 0.7)]
            states = {
                f"ent_{ac}": ActivationState(
                    node_id=f"ent_{ac}", access_history=[now - 100],
                    access_count=ac,
                ),
            }
            scored = score_candidates(
                candidates=candidates,
                spreading_bonuses={},
                hop_distances={},
                seed_node_ids=set(),
                activation_states=states,
                now=now,
                cfg=cfg,
            )
            bonuses.append(scored[0].exploration_bonus)
        # Bonus should monotonically decrease across the full range
        for i in range(len(bonuses) - 1):
            assert bonuses[i] > bonuses[i + 1]
        # All bonuses should be nonzero (no hard cutoff)
        for b in bonuses:
            assert b > 0.0

    # ---- Smooth exploration + rediscovery tests ----

    def test_smooth_decay_no_cliff(self):
        """Bonus at threshold boundary has no discontinuity (smooth transition)."""
        cfg = ActivationConfig(
            exploration_threshold=5,
            exploration_weight=0.05,
            rediscovery_weight=0.0,
        )
        now = time.time()
        # Access counts just below and at/above old threshold
        bonuses = {}
        for ac in [4, 5, 6]:
            candidates = [(f"ent_{ac}", 0.7)]
            states = {
                f"ent_{ac}": ActivationState(
                    node_id=f"ent_{ac}", access_history=[now - 100],
                    access_count=ac,
                ),
            }
            scored = score_candidates(
                candidates=candidates,
                spreading_bonuses={},
                hop_distances={},
                seed_node_ids=set(),
                activation_states=states,
                now=now,
                cfg=cfg,
            )
            bonuses[ac] = scored[0].exploration_bonus
        # No cliff: difference between consecutive values should be similar
        diff_4_5 = bonuses[4] - bonuses[5]
        diff_5_6 = bonuses[5] - bonuses[6]
        # Both positive (decreasing)
        assert diff_4_5 > 0
        assert diff_5_6 > 0
        # No cliff: ratio should be within 3x (not e.g. 0.03 vs 0.0)
        assert diff_4_5 / diff_5_6 < 3.0

    def test_rediscovery_dormant_entity(self):
        """Dormant entity (last access 60 days ago) gets rediscovery bonus."""
        cfg = ActivationConfig(
            exploration_weight=0.0,  # isolate rediscovery
            rediscovery_weight=0.02,
            rediscovery_halflife_days=30.0,
        )
        now = time.time()
        candidates = [("dormant", 0.7)]
        states = {
            "dormant": ActivationState(
                node_id="dormant",
                access_history=[now - 60 * 86400],  # 60 days ago
                access_count=5,
            ),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        r = scored[0]
        # Rediscovery bonus: 0.02 * 0.7 * (1 - exp(-ln2 * 60/30))
        # = 0.014 * (1 - exp(-1.386)) ≈ 0.014 * 0.75 ≈ 0.0105
        assert r.exploration_bonus > 0.01
        assert r.exploration_bonus < 0.015

    def test_rediscovery_recent_entity(self):
        """Recently accessed entity gets minimal rediscovery bonus."""
        cfg = ActivationConfig(
            exploration_weight=0.0,  # isolate rediscovery
            rediscovery_weight=0.02,
            rediscovery_halflife_days=30.0,
        )
        now = time.time()
        candidates = [("recent", 0.7)]
        states = {
            "recent": ActivationState(
                node_id="recent",
                access_history=[now - 100],  # 100 seconds ago
                access_count=5,
            ),
        }
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
        )
        r = scored[0]
        # 100 seconds ≈ 0.0012 days → bonus ≈ 0.02 * 0.7 * (1 - exp(~0)) ≈ ~0
        assert r.exploration_bonus < 0.001

    def test_no_rediscovery_without_history(self):
        """Entity with no access history gets no rediscovery bonus."""
        cfg = ActivationConfig(
            exploration_weight=0.0,  # isolate rediscovery
            rediscovery_weight=0.02,
            rediscovery_halflife_days=30.0,
        )
        now = time.time()
        candidates = [("no_history", 0.7)]
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
        )
        assert scored[0].exploration_bonus == 0.0
