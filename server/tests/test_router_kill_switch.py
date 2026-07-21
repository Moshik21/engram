"""Router explicit-zero kill-switch tests (M0.5).

A base-config weight of exactly 0.0 means "term disabled" and must survive
routing — apply_route may not resurrect it from _WEIGHT_PROFILES. Profiles
redistribute only among enabled terms. Default configs (no zero weights)
must remain byte-identical to the profile tuples.
"""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig
from engram.retrieval.router import QueryType, apply_route, classify_query

_WEIGHT_FIELDS = (
    "weight_semantic",
    "weight_activation",
    "weight_spreading",
    "weight_edge_proximity",
)

# Pinned literal copies of _WEIGHT_PROFILES — intentionally NOT imported from
# the router, so any profile retune must consciously update this pin.
_PINNED_PROFILES: dict[QueryType, tuple[float, float, float, float]] = {
    QueryType.DIRECT_LOOKUP: (0.75, 0.10, 0.05, 0.10),
    QueryType.TEMPORAL: (0.20, 0.55, 0.15, 0.10),
    QueryType.FREQUENCY: (0.15, 0.60, 0.15, 0.10),
    QueryType.ASSOCIATIVE: (0.55, 0.10, 0.20, 0.15),
    QueryType.CREATION: (0.30, 0.10, 0.25, 0.30),
    QueryType.DEFAULT: (0.40, 0.25, 0.15, 0.15),
}


def _weights(cfg: ActivationConfig) -> tuple[float, float, float, float]:
    return (
        cfg.weight_semantic,
        cfg.weight_activation,
        cfg.weight_spreading,
        cfg.weight_edge_proximity,
    )


class TestExplicitZeroKillSwitch:
    @pytest.mark.asyncio
    async def test_activation_zero_survives_temporal_route(self):
        """cfg.weight_activation=0 + TEMPORAL-routed query => stays 0, rest renormalized."""
        qt = await classify_query("What have I been working on recently?")
        assert qt == QueryType.TEMPORAL

        cfg = ActivationConfig(weight_activation=0.0)
        routed = apply_route(qt, cfg)

        assert routed.weight_activation == 0.0
        # TEMPORAL (0.20, 0.55, 0.15, 0.10): enabled share 0.45 rescaled to 1.0.
        assert routed.weight_semantic == pytest.approx(0.20 / 0.45)
        assert routed.weight_spreading == pytest.approx(0.15 / 0.45)
        assert routed.weight_edge_proximity == pytest.approx(0.10 / 0.45)
        assert sum(_weights(routed)) == pytest.approx(1.0)

    @pytest.mark.parametrize("field", _WEIGHT_FIELDS)
    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_any_zeroed_weight_stays_zero_on_every_route(self, field, query_type):
        """Zeroing any core weight disables that term for every route."""
        cfg = ActivationConfig(**{field: 0.0})
        routed = apply_route(query_type, cfg)

        assert getattr(routed, field) == 0.0
        # Enabled terms absorb the disabled share: total is preserved.
        assert sum(_weights(routed)) == pytest.approx(sum(_PINNED_PROFILES[query_type]))

    def test_all_weights_zero_stays_all_zero(self):
        cfg = ActivationConfig(
            weight_semantic=0.0,
            weight_activation=0.0,
            weight_spreading=0.0,
            weight_edge_proximity=0.0,
        )
        routed = apply_route(QueryType.TEMPORAL, cfg)
        assert _weights(routed) == (0.0, 0.0, 0.0, 0.0)


class TestDefaultConfigByteIdentical:
    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_default_cfg_routes_to_pinned_profile(self, query_type):
        """No base weight is 0 => routed weights are byte-identical to the profile."""
        routed = apply_route(query_type, ActivationConfig())
        assert _weights(routed) == _PINNED_PROFILES[query_type]


class TestSingleTuningPoint:
    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_apply_route_only_changes_the_four_weight_fields(self, query_type):
        """Schema-frozen: _WEIGHT_PROFILES is the single tuning point."""
        cfg = ActivationConfig(
            decay_exponent=0.7,
            retrieval_top_k=100,
            weight_activation=0.0,
        )
        routed = apply_route(query_type, cfg)

        base_dump = cfg.model_dump()
        routed_dump = routed.model_dump()
        for field in _WEIGHT_FIELDS:
            base_dump.pop(field)
            routed_dump.pop(field)
        assert routed_dump == base_dump
