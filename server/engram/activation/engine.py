"""ACT-R base-level learning equation and activation computation."""

from __future__ import annotations

import math

from engram.config import ActivationConfig
from engram.models.activation import ActivationState


def compute_base_level(
    access_history: list[float],
    now: float,
    cfg: ActivationConfig,
    consolidated_strength: float = 0.0,
) -> float:
    """Compute raw ACT-R base-level activation B_i(t).

    B_i(t) = ln(consolidated_strength + Σ (t - t_j)^(-d))

    Returns -10.0 for empty history with no consolidated strength
    (effectively zero after sigmoid).
    """
    if not access_history and consolidated_strength <= 0.0:
        return -10.0

    total = consolidated_strength
    for t_j in access_history:
        age = now - t_j
        if age < cfg.min_age_seconds:
            age = cfg.min_age_seconds
        total += age ** (-cfg.decay_exponent)

    return math.log(total) if total > 0 else -10.0


def normalize_activation(raw_b: float, cfg: ActivationConfig) -> float:
    """Map raw B_i to [0, 1] via sigmoid.

    activation = 1 / (1 + exp(-(B - B_mid) / B_scale))
    """
    x = (raw_b - cfg.B_mid) / cfg.B_scale
    # Clamp to avoid overflow in exp
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def compute_activation(
    access_history: list[float],
    now: float,
    cfg: ActivationConfig,
    consolidated_strength: float = 0.0,
) -> float:
    """Full pipeline: access_history -> normalized activation in [0, 1]."""
    raw = compute_base_level(access_history, now, cfg, consolidated_strength)
    return normalize_activation(raw, cfg)


def record_access(
    state: ActivationState,
    now: float,
    cfg: ActivationConfig,
) -> None:
    """Record a new access event on the activation state.

    Appends timestamp, increments count, updates last_accessed,
    and trims history to max_history_size.
    """
    state.access_history.append(now)
    state.access_count += 1
    state.last_accessed = now

    if len(state.access_history) > cfg.max_history_size:
        state.access_history = state.access_history[-cfg.max_history_size :]


def batch_compute_activations(
    states: dict[str, ActivationState],
    now: float,
    cfg: ActivationConfig,
) -> dict[str, float]:
    """Compute activation for a batch of nodes."""
    return {
        nid: compute_activation(
            state.access_history,
            now,
            cfg,
            state.consolidated_strength,
        )
        for nid, state in states.items()
    }
