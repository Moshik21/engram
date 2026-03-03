"""Activation state model for ACT-R memory dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActivationState:
    """Per-node activation state. Access history drives lazy ACT-R computation."""

    node_id: str
    access_history: list[float] = field(default_factory=list)
    spreading_bonus: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    consolidated_strength: float = 0.0  # Absorbed ACT-R contribution from compacted timestamps
    last_compacted: float = 0.0  # Timestamp of last compaction pass
    ts_alpha: float = 1.0  # Beta distribution success parameter
    ts_beta: float = 1.0  # Beta distribution failure parameter
