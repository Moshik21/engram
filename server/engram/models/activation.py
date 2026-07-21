"""Activation state model for ACT-R memory dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field

# Tier -> event weight for the ranking-side usage store (RF goal M1.1, F10).
# Single source of truth; config-overridable via
# ActivationConfig.usage_tier_weights. "surfaced" is ranker output and stays
# hygiene-only (w=0); nonzero tiers additionally append to usage_events.
DEFAULT_USAGE_TIER_WEIGHTS: dict[str, float] = {
    "surfaced": 0.0,
    "mentioned": 0.1,
    "used": 0.3,
    "corrected": 0.5,
    "confirmed": 1.0,
}

# Bound on the per-entity usage-event list (recency window; weights stay summed).
USAGE_EVENTS_MAX = 500


@dataclass
class ActivationState:
    """Per-node activation state. Access history drives lazy ACT-R computation."""

    node_id: str
    access_history: list[float] = field(default_factory=list)
    last_accessed: float = 0.0
    access_count: int = 0
    consolidated_strength: float = 0.0  # Absorbed ACT-R contribution from compacted timestamps
    last_compacted: float = 0.0  # Timestamp of last compaction pass
    ts_alpha: float = 1.0  # Beta distribution success parameter
    ts_beta: float = 1.0  # Beta distribution failure parameter
    # Ranking-side usage store (M1.1): (ts, weight) events from nonzero tiers.
    # Inert until M2 — nothing on the ranking path reads usage_events yet.
    usage_events: list[tuple[float, float]] = field(default_factory=list)
    usage_weight_sum: float = 0.0  # cached Σ weight so n_eff is an O(1) read
    usage_last_ts: float = 0.0  # cached max event ts so Δ_last is an O(1) read

    @property
    def n_eff(self) -> float:
        """Tier-weighted effective usage count (O(1))."""
        return self.usage_weight_sum

    def record_usage_event(self, ts: float, weight: float) -> None:
        """Append a ranking-eligible usage event and update the O(1) caches.

        Bounded at USAGE_EVENTS_MAX: the oldest events are trimmed but their
        weight stays in usage_weight_sum (frequency is cumulative; the event
        list exists for recency and journal idempotency, not as the ledger).
        """
        self.usage_events.append((ts, weight))
        self.usage_weight_sum += weight
        if ts > self.usage_last_ts:
            self.usage_last_ts = ts
        if len(self.usage_events) > USAGE_EVENTS_MAX:
            del self.usage_events[: len(self.usage_events) - USAGE_EVENTS_MAX]

    def has_usage_event(self, ts: float, weight: float) -> bool:
        """Exact-membership probe used for idempotent journal replay."""
        return (ts, weight) in set(self.usage_events)
