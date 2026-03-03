"""Thompson Sampling feedback recording for activation states."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.models.activation import ActivationState


async def record_positive_feedback(
    entity_id: str,
    activation_store,
    cfg: ActivationConfig,
) -> None:
    """Record positive feedback: entity was returned in retrieval results."""
    state = await activation_store.get_activation(entity_id)
    if state is None:
        state = ActivationState(node_id=entity_id)
    state.ts_alpha += cfg.ts_positive_increment
    await activation_store.set_activation(entity_id, state)


async def record_negative_feedback(
    entity_id: str,
    activation_store,
    cfg: ActivationConfig,
) -> None:
    """Record negative feedback: entity was a candidate but not returned."""
    state = await activation_store.get_activation(entity_id)
    if state is None:
        state = ActivationState(node_id=entity_id)
    state.ts_beta += cfg.ts_negative_increment
    await activation_store.set_activation(entity_id, state)
