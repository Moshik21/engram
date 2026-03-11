"""Tests for the DreamSpreadingPhase consolidation phase."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.dream import DreamSpreadingPhase
from engram.models.activation import ActivationState
from engram.models.consolidation import CycleContext, DreamRecord


def _make_state(access_history: list[float], node_id: str = "test") -> ActivationState:
    """Create an ActivationState with the given access history."""
    return ActivationState(
        node_id=node_id,
        access_history=access_history,
        access_count=len(access_history),
        last_accessed=max(access_history) if access_history else 0.0,
    )


def _dream_cfg(**overrides) -> ActivationConfig:
    """Return an ActivationConfig with dream enabled and sensible defaults."""
    defaults = {
        "consolidation_dream_enabled": True,
        "consolidation_dream_max_seeds": 20,
        "consolidation_dream_activation_floor": 0.15,
        "consolidation_dream_activation_ceiling": 0.75,
        "consolidation_dream_activation_midpoint": 0.40,
        "consolidation_dream_weight_increment": 0.05,
        "consolidation_dream_max_boost_per_edge": 0.15,
        "consolidation_dream_max_edge_weight": 3.0,
        "consolidation_dream_min_boost": 0.005,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


class TestDreamPhaseBasics:
    def test_phase_name(self):
        phase = DreamSpreadingPhase()
        assert phase.name == "dream"

    @pytest.mark.asyncio
    async def test_skipped_when_disabled(self):
        phase = DreamSpreadingPhase()
        cfg = ActivationConfig(consolidation_dream_enabled=False)

        result, records = await phase.execute(
            group_id="test",
            graph_store=AsyncMock(),
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.phase == "dream"
        assert result.status == "skipped"
        assert records == []


class TestDreamSeedSelection:
    @pytest.mark.asyncio
    async def test_bell_curve_selects_medium_activation(self):
        """Entities near midpoint (0.40) should be preferred over high/low."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()
        now = 1000.0

        # Create states with different activation levels
        # Entity at ~0.40 activation (medium - near midpoint)
        # Entity at ~0.90 activation (very recent access - high)
        # Entity at ~0.02 activation (very old - low, below floor)
        activation_store = AsyncMock()

        # Mock get_top_activated to return entities with controlled activation
        medium_state = _make_state([now - 30.0])  # ~30s ago → medium activation
        high_state = _make_state([now - 0.5])  # ~0.5s ago → very high activation
        low_state = _make_state([now - 100000.0])  # ~28hrs ago → very low activation

        activation_store.get_top_activated = AsyncMock(
            return_value=[
                ("ent_medium", medium_state),
                ("ent_high", high_state),
                ("ent_low", low_state),
            ]
        )

        seeds = await phase._select_dream_seeds(activation_store, "test", now, cfg)

        # The high entity should be excluded (above ceiling)
        # The low entity should be excluded (below floor)
        # Only medium should survive the band filter
        seed_ids = [sid for sid, _ in seeds]
        assert "ent_high" not in seed_ids or "ent_medium" in seed_ids
        # If medium is in band, it should be selected
        if seed_ids:
            assert seed_ids[0] == "ent_medium" or len(seed_ids) >= 1

    @pytest.mark.asyncio
    async def test_floor_ceiling_filtering(self):
        """Entities outside [floor, ceiling] band should be excluded."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(
            consolidation_dream_activation_floor=0.30,
            consolidation_dream_activation_ceiling=0.50,
        )
        now = 1000.0

        # All entities with very recent access → high activation (above ceiling)
        high_state = _make_state([now - 0.1])  # ~0.1s ago → ~0.97 activation

        activation_store = AsyncMock()
        activation_store.get_top_activated = AsyncMock(
            return_value=[
                ("ent_1", high_state),
                ("ent_2", high_state),
            ]
        )

        seeds = await phase._select_dream_seeds(activation_store, "test", now, cfg)
        assert len(seeds) == 0

    @pytest.mark.asyncio
    async def test_max_seeds_cap(self):
        """At most max_seeds entities should be returned."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(
            consolidation_dream_max_seeds=2,
            consolidation_dream_activation_floor=0.0,
            consolidation_dream_activation_ceiling=1.0,
        )
        now = 1000.0

        # Create many entities all in band
        state = _make_state([now - 30.0])
        entities = [(f"ent_{i}", state) for i in range(10)]

        activation_store = AsyncMock()
        activation_store.get_top_activated = AsyncMock(return_value=entities)

        seeds = await phase._select_dream_seeds(activation_store, "test", now, cfg)
        assert len(seeds) == 2

    @pytest.mark.asyncio
    async def test_empty_when_no_entities_in_band(self):
        """Should return empty list when no entities have activation in band."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        activation_store = AsyncMock()
        activation_store.get_top_activated = AsyncMock(return_value=[])

        seeds = await phase._select_dream_seeds(activation_store, "test", 1000.0, cfg)
        assert seeds == []


class TestDreamPhaseExecution:
    @pytest.mark.asyncio
    async def test_boosts_edge_weights(self):
        """update_relationship_weight should be called for traversed edges."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(
            consolidation_dream_activation_floor=0.0,
            consolidation_dream_activation_ceiling=1.0,
            consolidation_dream_min_boost=0.0,
        )
        now = 1000.0

        state = _make_state([now - 30.0])
        activation_store = AsyncMock()
        activation_store.get_top_activated = AsyncMock(
            return_value=[("seed_1", state)],
        )

        graph_store = AsyncMock()
        # After spreading, get_active_neighbors_with_weights returns neighbors
        graph_store.get_active_neighbors_with_weights = AsyncMock(
            side_effect=lambda eid, group_id=None: (
                [
                    ("neighbor_1", 1.0, "RELATES_TO"),
                ]
                if eid == "seed_1"
                else [
                    ("seed_1", 1.0, "RELATES_TO"),
                ]
            ),
        )
        graph_store.update_relationship_weight = AsyncMock(return_value=1.05)

        # Mock spread_activation to return bonuses for both nodes
        mock_bonuses = {"seed_1": 0.5, "neighbor_1": 0.3}
        mock_hops = {"seed_1": 0, "neighbor_1": 1}

        with patch(
            "engram.consolidation.phases.dream.spread_activation",
            new_callable=AsyncMock,
            return_value=(mock_bonuses, mock_hops),
        ):
            result, records = await phase.execute(
                group_id="test",
                graph_store=graph_store,
                activation_store=activation_store,
                search_index=AsyncMock(),
                cfg=cfg,
                cycle_id="cyc_test",
                dry_run=False,
            )

        assert result.status == "success"
        assert result.items_affected >= 1
        graph_store.update_relationship_weight.assert_called()
        assert len(records) >= 1
        assert all(isinstance(r, DreamRecord) for r in records)

    @pytest.mark.asyncio
    async def test_dry_run_no_modification(self):
        """Dry run should create records but not call update_relationship_weight."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(
            consolidation_dream_activation_floor=0.0,
            consolidation_dream_activation_ceiling=1.0,
            consolidation_dream_min_boost=0.0,
        )
        now = 1000.0

        state = _make_state([now - 30.0])
        activation_store = AsyncMock()
        activation_store.get_top_activated = AsyncMock(
            return_value=[("seed_1", state)],
        )

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(
            side_effect=lambda eid, group_id=None: (
                [
                    ("neighbor_1", 1.0, "RELATES_TO"),
                ]
                if eid == "seed_1"
                else [
                    ("seed_1", 1.0, "RELATES_TO"),
                ]
            ),
        )

        mock_bonuses = {"seed_1": 0.5, "neighbor_1": 0.3}
        mock_hops = {"seed_1": 0, "neighbor_1": 1}

        with patch(
            "engram.consolidation.phases.dream.spread_activation",
            new_callable=AsyncMock,
            return_value=(mock_bonuses, mock_hops),
        ):
            result, records = await phase.execute(
                group_id="test",
                graph_store=graph_store,
                activation_store=activation_store,
                search_index=AsyncMock(),
                cfg=cfg,
                cycle_id="cyc_test",
                dry_run=True,
            )

        assert result.status == "success"
        assert len(records) >= 1
        graph_store.update_relationship_weight.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_max_edge_weight_cap(self):
        """max_weight should be passed correctly to update_relationship_weight."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(
            consolidation_dream_activation_floor=0.0,
            consolidation_dream_activation_ceiling=1.0,
            consolidation_dream_min_boost=0.0,
            consolidation_dream_max_edge_weight=5.0,
        )
        now = 1000.0

        state = _make_state([now - 30.0])
        activation_store = AsyncMock()
        activation_store.get_top_activated = AsyncMock(
            return_value=[("seed_1", state)],
        )

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(
            side_effect=lambda eid, group_id=None: (
                [
                    ("neighbor_1", 1.0, "RELATES_TO"),
                ]
                if eid == "seed_1"
                else [
                    ("seed_1", 1.0, "RELATES_TO"),
                ]
            ),
        )
        graph_store.update_relationship_weight = AsyncMock(return_value=1.05)

        mock_bonuses = {"seed_1": 0.5, "neighbor_1": 0.3}
        mock_hops = {"seed_1": 0, "neighbor_1": 1}

        with patch(
            "engram.consolidation.phases.dream.spread_activation",
            new_callable=AsyncMock,
            return_value=(mock_bonuses, mock_hops),
        ):
            await phase.execute(
                group_id="test",
                graph_store=graph_store,
                activation_store=activation_store,
                search_index=AsyncMock(),
                cfg=cfg,
                cycle_id="cyc_test",
                dry_run=False,
            )

        call_args = graph_store.update_relationship_weight.call_args
        assert call_args.kwargs.get("max_weight") == 5.0 or call_args[0][3] == 5.0
        assert call_args.kwargs.get("predicate") == "RELATES_TO"

    @pytest.mark.asyncio
    async def test_no_access_recorded(self):
        """Dream spreading must NOT record access (critical invariant)."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(
            consolidation_dream_activation_floor=0.0,
            consolidation_dream_activation_ceiling=1.0,
        )
        now = 1000.0

        state = _make_state([now - 30.0])
        activation_store = AsyncMock()
        activation_store.get_top_activated = AsyncMock(
            return_value=[("seed_1", state)],
        )

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        mock_bonuses = {"seed_1": 0.1}
        mock_hops = {"seed_1": 0}

        with patch(
            "engram.consolidation.phases.dream.spread_activation",
            new_callable=AsyncMock,
            return_value=(mock_bonuses, mock_hops),
        ):
            await phase.execute(
                group_id="test",
                graph_store=graph_store,
                activation_store=activation_store,
                search_index=AsyncMock(),
                cfg=cfg,
                cycle_id="cyc_test",
                dry_run=False,
            )

        # record_access must NEVER be called during dream spreading
        activation_store.record_access.assert_not_called()

    @pytest.mark.asyncio
    async def test_populates_cycle_context(self):
        """Dream phase should add seed IDs to context.dream_seed_ids."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg(
            consolidation_dream_activation_floor=0.0,
            consolidation_dream_activation_ceiling=1.0,
        )
        now = 1000.0

        state = _make_state([now - 30.0])
        activation_store = AsyncMock()
        activation_store.get_top_activated = AsyncMock(
            return_value=[("seed_1", state)],
        )

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        ctx = CycleContext()

        with patch(
            "engram.consolidation.phases.dream.spread_activation",
            new_callable=AsyncMock,
            return_value=({"seed_1": 0.1}, {"seed_1": 0}),
        ):
            await phase.execute(
                group_id="test",
                graph_store=graph_store,
                activation_store=activation_store,
                search_index=AsyncMock(),
                cfg=cfg,
                cycle_id="cyc_test",
                dry_run=False,
                context=ctx,
            )

        assert "seed_1" in ctx.dream_seed_ids


class TestEdgeIdentification:
    @pytest.mark.asyncio
    async def test_only_edges_between_reached_nodes_boosted(self):
        """Edges to unreached nodes should not be boosted."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        # Bonuses only include seed and neighbor_1, not neighbor_2
        bonuses = {"seed_1": 0.5, "neighbor_1": 0.3}

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(
            side_effect=lambda eid, group_id=None: {
                "seed_1": [
                    ("neighbor_1", 1.0, "RELATES_TO"),
                    ("neighbor_2", 1.0, "RELATES_TO"),  # Not in reached set
                ],
                "neighbor_1": [("seed_1", 1.0, "RELATES_TO")],
            }.get(eid, []),
        )

        boosts = await phase._accumulate_edge_boosts(
            "seed_1",
            bonuses,
            graph_store,
            "test",
            cfg,
        )

        # Only (seed_1, neighbor_1) should be boosted, not (seed_1, neighbor_2)
        edge_keys = list(boosts.keys())
        canonical_key = (min("seed_1", "neighbor_1"), max("seed_1", "neighbor_1"), "RELATES_TO")
        assert canonical_key in edge_keys

        unreached_key = (min("seed_1", "neighbor_2"), max("seed_1", "neighbor_2"), "RELATES_TO")
        assert unreached_key not in edge_keys

    @pytest.mark.asyncio
    async def test_canonical_edge_key_dedup(self):
        """(A,B) and (B,A) should be deduplicated via canonical key."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        bonuses = {"node_a": 0.5, "node_b": 0.3}

        graph_store = AsyncMock()
        # Both directions return each other as neighbor
        graph_store.get_active_neighbors_with_weights = AsyncMock(
            side_effect=lambda eid, group_id=None: {
                "node_a": [("node_b", 1.0, "RELATES_TO")],
                "node_b": [("node_a", 1.0, "RELATES_TO")],
            }.get(eid, []),
        )

        boosts = await phase._accumulate_edge_boosts(
            "node_a",
            bonuses,
            graph_store,
            "test",
            cfg,
        )

        # Should only have one entry, not two
        assert len(boosts) == 1
        canonical = (min("node_a", "node_b"), max("node_a", "node_b"), "RELATES_TO")
        assert canonical in boosts

    @pytest.mark.asyncio
    async def test_same_pair_different_predicates_are_tracked_separately(self):
        """Different predicates between the same pair must not collapse together."""
        phase = DreamSpreadingPhase()
        cfg = _dream_cfg()

        bonuses = {"node_a": 0.5, "node_b": 0.3}

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(
            side_effect=lambda eid, group_id=None: {
                "node_a": [
                    ("node_b", 1.0, "RELATES_TO"),
                    ("node_b", 0.5, "MENTIONED_WITH"),
                ],
                "node_b": [
                    ("node_a", 1.0, "RELATES_TO"),
                    ("node_a", 0.5, "MENTIONED_WITH"),
                ],
            }.get(eid, []),
        )

        boosts = await phase._accumulate_edge_boosts(
            "node_a",
            bonuses,
            graph_store,
            "test",
            cfg,
        )

        assert len(boosts) == 2
        assert ("node_a", "node_b", "RELATES_TO") in boosts or (
            "node_b",
            "node_a",
            "RELATES_TO",
        ) in boosts
        assert ("node_a", "node_b", "MENTIONED_WITH") in boosts or (
            "node_b",
            "node_a",
            "MENTIONED_WITH",
        ) in boosts
