"""Tests for seed energy formula fix in spreading activation."""

import pytest

from engram.activation.spreading import identify_seeds
from engram.config import ActivationConfig
from engram.models.activation import ActivationState


class TestSeedEnergy:
    def test_low_activation_gets_floor_energy(self):
        """Zero-activation seed should get energy = sem_sim * 0.15."""
        cfg = ActivationConfig(seed_threshold=0.3)
        candidates = [("e1", 0.8)]
        states = {"e1": ActivationState(node_id="e1", access_history=[])}
        seeds = identify_seeds(candidates, states, now=1000.0, cfg=cfg)
        assert len(seeds) == 1
        node_id, energy = seeds[0]
        assert node_id == "e1"
        assert abs(energy - 0.12) < 1e-6

    def test_high_activation_uses_activation(self):
        """High-activation seed should use activation, not the floor."""
        cfg = ActivationConfig(seed_threshold=0.3)
        now = 100.0
        candidates = [("e1", 0.8)]
        states = {
            "e1": ActivationState(
                node_id="e1",
                access_history=[now - 1.0],
                access_count=1,
            ),
        }
        seeds = identify_seeds(candidates, states, now=now, cfg=cfg)
        assert len(seeds) == 1
        _, energy = seeds[0]
        assert energy > 0.12

    def test_zero_semantic_not_a_seed(self):
        """Zero semantic similarity is below threshold, not a seed."""
        cfg = ActivationConfig(seed_threshold=0.3)
        candidates = [("e1", 0.0)]
        seeds = identify_seeds(candidates, {}, now=1000.0, cfg=cfg)
        assert len(seeds) == 0

    def test_below_threshold_excluded(self):
        """Candidates below seed_threshold are not seeds."""
        cfg = ActivationConfig(seed_threshold=0.5)
        candidates = [("e1", 0.4), ("e2", 0.6)]
        seeds = identify_seeds(candidates, {}, now=1000.0, cfg=cfg)
        assert len(seeds) == 1
        assert seeds[0][0] == "e2"

    def test_energy_floor_vs_old_formula(self):
        """New floor (0.15) gives more energy than old formula (sem*0.1)."""
        cfg = ActivationConfig(seed_threshold=0.3)
        candidates = [("e1", 0.5)]
        states = {"e1": ActivationState(node_id="e1", access_history=[])}
        seeds = identify_seeds(candidates, states, now=1000.0, cfg=cfg)
        _, energy = seeds[0]
        assert energy == pytest.approx(0.075)
        old_energy = 0.5 * max(0.0, 0.5 * 0.1)
        assert energy > old_energy

    def test_multiple_seeds(self):
        """Multiple seeds get correct energy values."""
        cfg = ActivationConfig(seed_threshold=0.3)
        candidates = [("e1", 0.8), ("e2", 0.5), ("e3", 0.3)]
        seeds = identify_seeds(candidates, {}, now=1000.0, cfg=cfg)
        assert len(seeds) == 3
        energies = {nid: e for nid, e in seeds}
        assert energies["e1"] == pytest.approx(0.8 * 0.15)
        assert energies["e2"] == pytest.approx(0.5 * 0.15)
        assert energies["e3"] == pytest.approx(0.3 * 0.15)
