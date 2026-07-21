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
    decay_override: float | None = None,
) -> float:
    """Compute raw ACT-R base-level activation B_i(t).

    B_i(t) = ln(consolidated_strength + Σ (t - t_j)^(-d))

    Returns -10.0 for empty history with no consolidated strength
    (effectively zero after sigmoid).
    """
    if not access_history and consolidated_strength <= 0.0:
        return -10.0

    d = decay_override if decay_override is not None else cfg.decay_exponent
    total = consolidated_strength
    for t_j in access_history:
        age = now - t_j
        if age < cfg.min_age_seconds:
            age = cfg.min_age_seconds
        total += age ** (-d)

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
    decay_override: float | None = None,
) -> float:
    """Full pipeline: access_history -> normalized activation in [0, 1]."""
    raw = compute_base_level(access_history, now, cfg, consolidated_strength, decay_override)
    return normalize_activation(raw, cfg)


def seed_consolidated_strength(
    state: ActivationState,
    amount: float,
    cap: float,
) -> bool:
    """Seed an importance prior into consolidated_strength, bounded by cap.

    The prior sits inside the ACT-R ln-sum (compute_base_level), giving
    one-shot high-value facts a durable activation floor. The cap bounds
    total consolidated_strength so repeated commits cannot inflate it.
    Returns True when the state changed.
    """
    if amount <= 0.0 or state.consolidated_strength >= cap:
        return False
    state.consolidated_strength = min(cap, state.consolidated_strength + amount)
    return True


def compute_u(
    state: ActivationState,
    now: float,
    cfg: ActivationConfig,
) -> float:
    """Ranking-side usage signal u = f * r' in [0, 1] (RF_target_design section 1).

    f      = min(1, ln(1 + n_eff) / ln(1 + N_cap))   frequency, log-compressed
    r      = 2^(-delta_last / h_seconds)             recency, half-life h
    r'     = r_floor + (1 - r_floor) * r             floor keeps old-but-frequent alive
    u      = f * r'                                  0 exactly when n_eff == 0

    Pure and deterministic: reads only the O(1) caches state.n_eff and
    state.usage_last_ts (M1.1). No store access.
    """
    n_eff = state.n_eff
    if n_eff <= 0.0:
        return 0.0

    f = min(1.0, math.log1p(n_eff) / math.log1p(cfg.usage_n_cap))
    h_seconds = cfg.usage_half_life_days * 86400.0
    delta_last = max(0.0, now - state.usage_last_ts)
    r = 2.0 ** (-delta_last / h_seconds)
    r_prime = cfg.usage_r_floor + (1.0 - cfg.usage_r_floor) * r
    return f * r_prime


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
