"""Tests for the access history compaction phase."""

import time

import pytest
import pytest_asyncio

from engram.activation.engine import compute_activation
from engram.config import ActivationConfig
from engram.consolidation.phases.compact import (
    AccessHistoryCompactionPhase,
    logarithmic_compact,
)
from engram.models.activation import ActivationState
from engram.storage.memory.activation import MemoryActivationStore


@pytest_asyncio.fixture
async def activation():
    return MemoryActivationStore(cfg=ActivationConfig())


class TestLogarithmicCompact:
    """Unit tests for the pure compaction function."""

    def test_keep_all_recent(self):
        now = time.time()
        # 20 timestamps in the last hour
        history = [now - i * 60 for i in range(20)]
        result = logarithmic_compact(history, now, max_age_seconds=90 * 86400, keep_min=5)

        # All within 24h, should keep all
        assert len(result) == 20

    def test_hourly_bucketing(self):
        now = time.time()
        # 48 timestamps, one every 30 minutes, spanning 24 hours (in 1-7d range)
        history = [now - 86400 - i * 1800 for i in range(48)]
        result = logarithmic_compact(history, now, max_age_seconds=90 * 86400, keep_min=5)

        # 48 events across ~24 hours in the 1-7d range → ~24 hourly buckets
        assert len(result) < 48
        assert len(result) >= 5  # At least keep_min

    def test_daily_bucketing(self):
        now = time.time()
        # 100 timestamps, one every 2 hours, starting 10 days ago
        history = [now - 10 * 86400 - i * 7200 for i in range(100)]
        result = logarithmic_compact(history, now, max_age_seconds=90 * 86400, keep_min=5)

        # Most are in 7d+ range → daily buckets, significant reduction
        assert len(result) < 100

    def test_drops_old_timestamps(self):
        now = time.time()
        max_age = 30 * 86400  # 30 days
        # Some within range, some beyond
        history = [
            now - 10 * 86400,  # Within range
            now - 20 * 86400,  # Within range
            now - 40 * 86400,  # Beyond max_age
            now - 50 * 86400,  # Beyond max_age
        ]
        result = logarithmic_compact(history, now, max_age_seconds=max_age, keep_min=1)

        assert len(result) == 2  # Only the 2 within range

    def test_keep_min_enforcement(self):
        now = time.time()
        # All timestamps are very old (beyond max_age)
        history = [now - 200 * 86400 - i * 86400 for i in range(20)]
        result = logarithmic_compact(history, now, max_age_seconds=90 * 86400, keep_min=5)

        # Should keep at least 5 (most recent from original)
        assert len(result) == 5

    def test_empty_history(self):
        result = logarithmic_compact([], time.time(), max_age_seconds=90 * 86400, keep_min=5)
        assert result == []

    def test_result_sorted_descending(self):
        now = time.time()
        history = [now - i * 3600 for i in range(50)]
        result = logarithmic_compact(history, now, max_age_seconds=90 * 86400, keep_min=5)

        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1]


class TestAccessHistoryCompactionPhase:
    """Integration tests for the compaction phase."""

    @pytest.mark.asyncio
    async def test_compacts_old_history(self, activation):
        now = time.time()
        # Create entity with large history spanning 30 days
        history = [now - i * 3600 for i in range(720)]  # One per hour for 30 days
        state = ActivationState(
            node_id="ent_test",
            access_history=history,
            access_count=720,
        )
        await activation.set_activation("ent_test", state)

        # Seed group tracking
        activation._group_map["ent_test"] = "test"

        cfg = ActivationConfig(
            consolidation_compaction_horizon_days=90,
            consolidation_compaction_keep_min=10,
            consolidation_compaction_logarithmic=True,
        )
        phase = AccessHistoryCompactionPhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_affected == 1

        # Verify history was actually compacted
        updated = await activation.get_activation("ent_test")
        assert len(updated.access_history) < 720

    @pytest.mark.asyncio
    async def test_dry_run_no_modification(self, activation):
        now = time.time()
        history = [now - i * 3600 for i in range(200)]
        state = ActivationState(
            node_id="ent_test",
            access_history=history,
            access_count=200,
        )
        await activation.set_activation("ent_test", state)
        activation._group_map["ent_test"] = "test"

        cfg = ActivationConfig(consolidation_compaction_logarithmic=True)
        phase = AccessHistoryCompactionPhase()
        result, _ = await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )

        # History should not be modified
        updated = await activation.get_activation("ent_test")
        assert len(updated.access_history) == 200

    @pytest.mark.asyncio
    async def test_simple_mode(self, activation):
        now = time.time()
        # Mix of recent and old timestamps
        history = (
            [now - i * 3600 for i in range(24)]  # Recent
            + [now - 50 * 86400 - i * 3600 for i in range(100)]  # Old
        )
        state = ActivationState(
            node_id="ent_test",
            access_history=history,
            access_count=124,
        )
        await activation.set_activation("ent_test", state)
        activation._group_map["ent_test"] = "test"

        cfg = ActivationConfig(
            consolidation_compaction_logarithmic=False,
            consolidation_compaction_horizon_days=30,
            consolidation_compaction_keep_min=10,
        )
        phase = AccessHistoryCompactionPhase()
        result, _ = await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_affected == 1
        updated = await activation.get_activation("ent_test")
        assert len(updated.access_history) < 124


class TestConsolidatedStrengthCompaction:
    """Tests for consolidated_strength preservation during compaction."""

    @pytest.mark.asyncio
    async def test_compaction_sets_consolidated_strength(self, activation):
        """After compaction drops timestamps, cs should be > 0."""
        now = time.time()
        # 720 timestamps spanning 30 days — will be compacted
        history = [now - i * 3600 for i in range(720)]
        state = ActivationState(
            node_id="ent_cs",
            access_history=history,
            access_count=720,
        )
        await activation.set_activation("ent_cs", state)
        activation._group_map["ent_cs"] = "test"

        cfg = ActivationConfig(
            consolidation_compaction_horizon_days=90,
            consolidation_compaction_keep_min=10,
            consolidation_compaction_logarithmic=True,
        )
        phase = AccessHistoryCompactionPhase()
        await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_cs",
            dry_run=False,
        )

        updated = await activation.get_activation("ent_cs")
        assert updated.consolidated_strength > 0.0

    @pytest.mark.asyncio
    async def test_consolidated_strength_preserves_activation(self, activation):
        """Activation before compaction ≈ after compaction within 0.1%."""
        now = time.time()
        history = [now - i * 3600 for i in range(720)]
        state = ActivationState(
            node_id="ent_pres",
            access_history=list(history),
            access_count=720,
        )
        await activation.set_activation("ent_pres", state)
        activation._group_map["ent_pres"] = "test"

        cfg = ActivationConfig(
            consolidation_compaction_horizon_days=90,
            consolidation_compaction_keep_min=10,
            consolidation_compaction_logarithmic=True,
        )

        # Compute activation before compaction
        act_before = compute_activation(history, now, cfg)

        phase = AccessHistoryCompactionPhase()
        await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_pres",
            dry_run=False,
        )

        updated = await activation.get_activation("ent_pres")
        act_after = compute_activation(
            updated.access_history,
            now,
            cfg,
            updated.consolidated_strength,
        )

        # Within 0.1% relative error
        assert abs(act_before - act_after) / max(act_before, 1e-10) < 0.001

    @pytest.mark.asyncio
    async def test_cs_accumulates_across_cycles(self, activation):
        """Second compaction adds to existing cs rather than replacing it."""
        now = time.time()
        history = [now - i * 3600 for i in range(720)]
        state = ActivationState(
            node_id="ent_acc",
            access_history=list(history),
            access_count=720,
        )
        await activation.set_activation("ent_acc", state)
        activation._group_map["ent_acc"] = "test"

        cfg = ActivationConfig(
            consolidation_compaction_horizon_days=90,
            consolidation_compaction_keep_min=10,
            consolidation_compaction_logarithmic=True,
        )

        phase = AccessHistoryCompactionPhase()

        # First compaction
        await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_1",
            dry_run=False,
        )
        after_first = await activation.get_activation("ent_acc")
        cs_first = after_first.consolidated_strength

        # Add more old timestamps to force a second compaction
        after_first.access_history.extend([now - 10 * 86400 - i * 3600 for i in range(200)])
        await activation.set_activation("ent_acc", after_first)

        # Second compaction
        await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_2",
            dry_run=False,
        )
        after_second = await activation.get_activation("ent_acc")

        # cs should have increased (accumulated)
        assert after_second.consolidated_strength >= cs_first


class TestDirtyFlagCompaction:
    """Tests for the last_compacted dirty-flag optimization."""

    @pytest.mark.asyncio
    async def test_skips_already_compacted(self, activation):
        """Entity with last_compacted >= last_accessed should be skipped."""
        now = time.time()
        history = [now - i * 3600 for i in range(200)]
        state = ActivationState(
            node_id="ent_skip",
            access_history=history,
            access_count=200,
            last_accessed=now - 100,
            last_compacted=now,  # Compacted more recently than last access
        )
        await activation.set_activation("ent_skip", state)
        activation._group_map["ent_skip"] = "test"

        cfg = ActivationConfig(consolidation_compaction_logarithmic=True)
        phase = AccessHistoryCompactionPhase()
        result, _ = await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_skip",
            dry_run=False,
        )

        assert result.items_processed == 0

    @pytest.mark.asyncio
    async def test_recompacts_after_new_access(self, activation):
        """Entity accessed after last compaction should be re-processed."""
        now = time.time()
        history = [now - i * 3600 for i in range(720)]
        state = ActivationState(
            node_id="ent_recomp",
            access_history=history,
            access_count=720,
            last_accessed=now,  # Accessed just now
            last_compacted=now - 3600,  # Compacted 1 hour ago
        )
        await activation.set_activation("ent_recomp", state)
        activation._group_map["ent_recomp"] = "test"

        cfg = ActivationConfig(
            consolidation_compaction_horizon_days=90,
            consolidation_compaction_keep_min=10,
            consolidation_compaction_logarithmic=True,
        )
        phase = AccessHistoryCompactionPhase()
        result, _ = await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_recomp",
            dry_run=False,
        )

        assert result.items_processed == 1

    @pytest.mark.asyncio
    async def test_compaction_sets_last_compacted(self, activation):
        """After compaction, last_compacted should be set to a recent timestamp."""
        now = time.time()
        history = [now - i * 3600 for i in range(720)]
        state = ActivationState(
            node_id="ent_flag",
            access_history=history,
            access_count=720,
            last_accessed=now,
        )
        await activation.set_activation("ent_flag", state)
        activation._group_map["ent_flag"] = "test"

        cfg = ActivationConfig(
            consolidation_compaction_horizon_days=90,
            consolidation_compaction_keep_min=10,
            consolidation_compaction_logarithmic=True,
        )
        phase = AccessHistoryCompactionPhase()
        await phase.execute(
            group_id="test",
            graph_store=None,
            activation_store=activation,
            search_index=None,
            cfg=cfg,
            cycle_id="cyc_flag",
            dry_run=False,
        )

        updated = await activation.get_activation("ent_flag")
        assert updated.last_compacted > 0
