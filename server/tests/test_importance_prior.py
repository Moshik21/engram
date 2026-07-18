"""Importance as a prior (M3.3): seeded consolidated_strength at entity commit.

Flag: importance_prior_enabled (default False, EVAL-GATED). Also covers the
flag-independent compact absorbed-mass decay (compact_strength_decay).
"""

from __future__ import annotations

import time

import pytest
import pytest_asyncio

from engram.activation.engine import compute_activation, seed_consolidated_strength
from engram.config import ActivationConfig
from engram.consolidation.phases.compact import (
    AccessHistoryCompactionPhase,
    compute_dropped_strength,
    logarithmic_compact,
)
from engram.extraction.apply import ApplyEngine
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.models import EntityCandidate
from engram.models.activation import ActivationState
from engram.models.episode import Episode
from engram.storage.memory.activation import MemoryActivationStore

DAY = 86400.0


class _FakeGraph:
    """Minimal graph store: remembers created entities for re-resolution."""

    def __init__(self) -> None:
        self.entities: dict[str, object] = {}

    async def find_entity_candidates(self, name: str, group_id: str) -> list:
        return [e for e in self.entities.values() if e.name == name]

    async def create_entity(self, entity) -> None:
        self.entities[entity.id] = entity

    async def get_entity(self, entity_id: str, group_id: str | None = None):
        return self.entities.get(entity_id)

    async def update_entity(self, entity_id: str, updates: dict, group_id: str | None = None):
        return None

    async def link_episode_entity(self, episode_id: str, entity_id: str, group_id=None):
        return None


@pytest_asyncio.fixture
async def activation():
    return MemoryActivationStore(cfg=ActivationConfig())


def _engine(activation, cfg: ActivationConfig) -> ApplyEngine:
    return ApplyEngine(
        graph_store=_FakeGraph(),
        activation_store=activation,
        cfg=cfg,
        canonicalizer=PredicateCanonicalizer(),
    )


def _identity_candidate() -> EntityCandidate:
    return EntityCandidate(
        name="Konner Moshier",
        entity_type="Person",
        raw_payload={"signals": ["client_proposal", "identity_pattern"]},
    )


async def _commit(engine: ApplyEngine, candidate: EntityCandidate) -> str:
    episode = Episode(id="ep_prior", content="My name is Konner.", group_id="default")
    outcome = await engine.apply_entities([candidate], episode, "default")
    return outcome.entity_map[candidate.name]


# ======================================================================
# Arithmetic sanity: one-shot high-value vs mundane at 30 days
# ======================================================================


class TestArithmeticSanity:
    def test_oneshot_identity_within_2x_of_mundane_at_30_days(self):
        """With the prior seeded, a one-shot identity fact at 30 days scores
        within 2x of a 5-access mundane entity at 30 days."""
        cfg = ActivationConfig()
        now = time.time()

        oneshot_history = [now - 30 * DAY]
        mundane_history = [now - d * DAY for d in (2, 9, 16, 23, 30)]

        oneshot_act = compute_activation(oneshot_history, now, cfg, consolidated_strength=0.02)
        mundane_act = compute_activation(mundane_history, now, cfg, consolidated_strength=0.0)

        assert mundane_act / oneshot_act <= 2.0

    def test_without_prior_oneshot_falls_outside_2x(self):
        """Regression contrast: without the seed, the same one-shot fact decays
        past the 2x band — the prior is load-bearing, not decorative."""
        cfg = ActivationConfig()
        now = time.time()

        oneshot_act = compute_activation([now - 30 * DAY], now, cfg, consolidated_strength=0.0)
        mundane_act = compute_activation(
            [now - d * DAY for d in (2, 9, 16, 23, 30)],
            now,
            cfg,
            consolidated_strength=0.0,
        )

        assert mundane_act / oneshot_act > 2.0


# ======================================================================
# Seeding at entity commit (ApplyEngine), flag-gated
# ======================================================================


@pytest.mark.asyncio
class TestSeedAtCommit:
    async def test_flag_off_no_seed(self, activation):
        """Default (flag off): commit behavior identical to today — no
        consolidated_strength, only the record_access timestamp."""
        cfg = ActivationConfig()
        assert cfg.importance_prior_enabled is False
        engine = _engine(activation, cfg)

        entity_id = await _commit(engine, _identity_candidate())

        state = await activation.get_activation(entity_id)
        assert state is not None
        assert state.consolidated_strength == 0.0
        assert state.access_count == 1

    async def test_flag_on_identity_core_seeds(self, activation):
        cfg = ActivationConfig(importance_prior_enabled=True, identity_core_enabled=True)
        engine = _engine(activation, cfg)

        entity_id = await _commit(engine, _identity_candidate())

        state = await activation.get_activation(entity_id)
        assert state.consolidated_strength == pytest.approx(0.02)

    async def test_flag_on_durable_type_seeds(self, activation):
        # identity_core disabled so the Decision exercises the durable branch,
        # not the (higher) identity seed.
        cfg = ActivationConfig(importance_prior_enabled=True, identity_core_enabled=False)
        engine = _engine(activation, cfg)

        entity_id = await _commit(
            engine,
            EntityCandidate(name="Use SQLite for lite mode", entity_type="Decision"),
        )

        state = await activation.get_activation(entity_id)
        assert state.consolidated_strength == pytest.approx(0.01)

    async def test_flag_on_client_proposal_seeds(self, activation):
        # Non-durable, non-identity type with a client_proposal signal.
        cfg = ActivationConfig(importance_prior_enabled=True, identity_core_enabled=False)
        engine = _engine(activation, cfg)

        entity_id = await _commit(
            engine,
            EntityCandidate(
                name="Helix Native",
                entity_type="Technology",
                raw_payload={"signals": ["client_proposal"]},
            ),
        )

        state = await activation.get_activation(entity_id)
        assert state.consolidated_strength == pytest.approx(0.01)

    async def test_flag_on_mundane_entity_not_seeded(self, activation):
        cfg = ActivationConfig(importance_prior_enabled=True)
        engine = _engine(activation, cfg)

        entity_id = await _commit(
            engine,
            EntityCandidate(name="Coffee Shop", entity_type="Concept"),
        )

        state = await activation.get_activation(entity_id)
        assert state.consolidated_strength == 0.0

    async def test_repeated_commits_capped(self, activation):
        """Re-committing the same high-value entity cannot inflate the prior
        past the cap (0.05)."""
        cfg = ActivationConfig(importance_prior_enabled=True, identity_core_enabled=True)
        engine = _engine(activation, cfg)

        first_id = await _commit(engine, _identity_candidate())
        for _ in range(4):
            repeat_id = await _commit(engine, _identity_candidate())
            assert repeat_id == first_id  # resolved, not re-created

        state = await activation.get_activation(first_id)
        assert state.consolidated_strength == pytest.approx(0.05)


def test_seed_helper_respects_existing_strength_above_cap():
    """An entity already above the cap (e.g. via compact absorption) is
    never bumped further."""
    state = ActivationState(node_id="e1", consolidated_strength=0.4)
    assert seed_consolidated_strength(state, 0.02, 0.05) is False
    assert state.consolidated_strength == 0.4


# ======================================================================
# Compact absorbed-mass decay (flag-independent defect fix)
# ======================================================================


@pytest.mark.asyncio
class TestCompactStrengthDecay:
    async def _run_compact(self, activation, cfg: ActivationConfig) -> None:
        phase = AccessHistoryCompactionPhase()
        await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_decay",
            dry_run=False,
        )

    @staticmethod
    def _compactable_state(now: float, prior_strength: float) -> ActivationState:
        # 50 timestamps at 10-12 days old — daily-bucketed, most get dropped.
        history = [now - 10 * DAY - i * 3600 for i in range(50)]
        return ActivationState(
            node_id="ent_decay",
            access_history=history,
            access_count=50,
            consolidated_strength=prior_strength,
        )

    async def test_absorbed_mass_decays_per_pass(self, activation):
        """Previously absorbed strength is multiplied by compact_strength_decay
        (default 0.9) before this pass's dropped mass is absorbed."""
        # Default 1.0 preserves today's scoring; the decay is opt-in until the
        # M4 eval arm decides the production value.
        assert ActivationConfig().compact_strength_decay == pytest.approx(1.0)
        cfg = ActivationConfig(
            consolidation_cue_hygiene_enabled=False,
            compact_strength_decay=0.9,
        )
        now = time.time()
        state = self._compactable_state(now, prior_strength=1.0)
        await activation.set_activation("ent_decay", state)
        activation._group_map["ent_decay"] = "test"

        # Expected: prior * 0.9 + dropped contribution of this pass.
        kept = logarithmic_compact(
            list(state.access_history),
            now,
            cfg.consolidation_compaction_horizon_days * DAY,
            cfg.consolidation_compaction_keep_min,
        )
        dropped = set(state.access_history) - set(kept)
        assert dropped  # sanity: the pass actually compacts
        expected = 1.0 * 0.9 + compute_dropped_strength(
            dropped,
            now,
            cfg.decay_exponent,
            cfg.min_age_seconds,
        )

        await self._run_compact(activation, cfg)

        updated = await activation.get_activation("ent_decay")
        assert updated.consolidated_strength == pytest.approx(expected)
        assert updated.consolidated_strength < 1.0 + compute_dropped_strength(
            dropped, now, cfg.decay_exponent, cfg.min_age_seconds
        )  # no longer a monotone accumulator

    async def test_decay_one_preserves_legacy_accumulation(self, activation):
        """compact_strength_decay=1.0 reproduces the old pure-accumulation math."""
        cfg = ActivationConfig(
            compact_strength_decay=1.0,
            consolidation_cue_hygiene_enabled=False,
        )
        now = time.time()
        state = self._compactable_state(now, prior_strength=1.0)
        await activation.set_activation("ent_decay", state)
        activation._group_map["ent_decay"] = "test"

        kept = logarithmic_compact(
            list(state.access_history),
            now,
            cfg.consolidation_compaction_horizon_days * DAY,
            cfg.consolidation_compaction_keep_min,
        )
        dropped = set(state.access_history) - set(kept)
        expected = 1.0 + compute_dropped_strength(
            dropped,
            now,
            cfg.decay_exponent,
            cfg.min_age_seconds,
        )

        await self._run_compact(activation, cfg)

        updated = await activation.get_activation("ent_decay")
        assert updated.consolidated_strength == pytest.approx(expected)
