"""Prospective memory trigger matching (Wave 4)."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.prospective import Intention, IntentionMatch, IntentionMeta

logger = logging.getLogger(__name__)

# Priority ordering for tie-breaking
_PRIORITY_ORDER = {"critical": 0, "high": 1, "normal": 2, "low": 3}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _is_active(intention: Intention, now: datetime | None = None) -> bool:
    """Check if an intention is active (enabled, not exhausted, not expired)."""
    if not intention.enabled:
        return False
    if intention.fire_count >= intention.max_fires:
        return False
    if intention.expires_at is not None:
        check_time = now or datetime.utcnow()
        if intention.expires_at <= check_time:
            return False
    return True


async def check_triggers(
    content: str,
    entity_names: list[str],
    intentions: list[Intention],
    embed_fn: Callable[[str], Coroutine[Any, Any, list[float]]] | None = None,
    threshold_override: float | None = None,
) -> list[IntentionMatch]:
    """Check content against active intentions and return matches.

    Args:
        content: The episode content text.
        entity_names: Entity names extracted from the episode.
        intentions: List of intentions to check against.
        embed_fn: Async function to embed text (for semantic triggers).
        threshold_override: Override per-intention thresholds.

    Returns:
        List of IntentionMatch sorted by similarity descending.
    """
    now = datetime.utcnow()
    active = [i for i in intentions if _is_active(i, now)]
    if not active:
        return []

    matches: list[IntentionMatch] = []
    entity_names_lower = {n.lower() for n in entity_names}

    # Separate by trigger type
    entity_triggers = [i for i in active if i.trigger_type == "entity_mention"]
    semantic_triggers = [i for i in active if i.trigger_type == "semantic"]

    # Entity mention matching (instant, no embedding needed)
    for intention in entity_triggers:
        if intention.entity_name and intention.entity_name.lower() in entity_names_lower:
            matches.append(IntentionMatch(
                intention_id=intention.id,
                trigger_text=intention.trigger_text,
                action_text=intention.action_text,
                similarity=1.0,
                matched_via="entity_mention",
            ))

    # Semantic matching (requires embeddings)
    if semantic_triggers and embed_fn is not None and content.strip():
        try:
            content_embedding = await embed_fn(content)
        except Exception:
            logger.warning("Embedding failed for prospective matching", exc_info=True)
            content_embedding = None

        if content_embedding:
            for intention in semantic_triggers:
                try:
                    trigger_embedding = await embed_fn(intention.trigger_text)
                except Exception:
                    continue

                sim = _cosine_similarity(content_embedding, trigger_embedding)
                threshold = threshold_override or intention.threshold
                if sim >= threshold:
                    matches.append(IntentionMatch(
                        intention_id=intention.id,
                        trigger_text=intention.trigger_text,
                        action_text=intention.action_text,
                        similarity=sim,
                        matched_via="semantic",
                    ))

    # Sort by similarity descending
    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches


async def check_intention_activations(
    spreading_results: dict[str, float],
    activation_states: dict[str, ActivationState | None],
    intention_entities: list[Entity],
    extracted_entity_ids: set[str],
    now: float,
    cfg: ActivationConfig,
    max_per_episode: int = 3,
) -> list[IntentionMatch]:
    """Check graph-embedded intentions against spreading activation results.

    This is the v2 activation-based trigger check. For each intention entity:
    1. entity_mention fast path: O(1) set lookup of trigger_entity_ids
    2. activation path: base_activation + spreading_bonus >= threshold

    Args:
        spreading_results: {entity_id: spreading_bonus} from mini spreading pass
        activation_states: {entity_id: ActivationState} for intention entities
        intention_entities: List of Entity objects with type="Intention"
        extracted_entity_ids: Entity IDs extracted from the current episode
        now: Current timestamp (time.time())
        cfg: Activation config
        max_per_episode: Max intentions to fire per episode

    Returns:
        Sorted list of IntentionMatch (by priority then activation).
    """
    from engram.activation.engine import compute_activation

    matches: list[IntentionMatch] = []

    for entity in intention_entities:
        attrs = entity.attributes or {}
        try:
            meta = IntentionMeta(**attrs)
        except Exception:
            continue

        if not meta.enabled:
            continue
        if meta.fire_count >= meta.max_fires:
            continue

        # Check expiry
        if meta.expires_at:
            try:
                exp = datetime.fromisoformat(meta.expires_at)
                if exp <= datetime.utcnow():
                    continue
            except (ValueError, TypeError):
                pass

        # Check cooldown
        if meta.last_fired and meta.cooldown_seconds > 0:
            try:
                last_dt = datetime.fromisoformat(meta.last_fired)
                elapsed = (datetime.utcnow() - last_dt).total_seconds()
                if elapsed < meta.cooldown_seconds:
                    continue
            except (ValueError, TypeError):
                pass

        # Fast path: entity_mention trigger
        if meta.trigger_type == "entity_mention":
            if meta.trigger_entity_ids and extracted_entity_ids & set(meta.trigger_entity_ids):
                matches.append(IntentionMatch(
                    intention_id=entity.id,
                    trigger_text=meta.trigger_text,
                    action_text=meta.action_text,
                    similarity=1.0,
                    matched_via="entity_mention",
                    context=meta.context,
                    see_also=meta.see_also,
                ))
                continue

        # Activation path: base_activation + spreading_bonus
        state = activation_states.get(entity.id)
        base_activation = 0.0
        if state:
            base_activation = compute_activation(state.access_history, now, cfg)

        spreading_bonus = spreading_results.get(entity.id, 0.0)

        # Also check if any TRIGGERED_BY target entities got spreading energy
        trigger_spreading = 0.0
        for tid in meta.trigger_entity_ids:
            trigger_spreading = max(trigger_spreading, spreading_results.get(tid, 0.0))

        # Total activation: own base + own spreading + max trigger spreading
        total_activation = base_activation + spreading_bonus + trigger_spreading

        if total_activation >= meta.activation_threshold:
            matches.append(IntentionMatch(
                intention_id=entity.id,
                trigger_text=meta.trigger_text,
                action_text=meta.action_text,
                similarity=round(total_activation, 4),
                matched_via="activation",
                context=meta.context,
                see_also=meta.see_also,
            ))

    # Sort by priority (lower order first), then by activation (higher first)
    matches.sort(
        key=lambda m: (
            _PRIORITY_ORDER.get(
                next(
                    (
                        IntentionMeta(**(e.attributes or {})).priority
                        for e in intention_entities
                        if e.id == m.intention_id
                    ),
                    "normal",
                ),
                2,
            ),
            -m.similarity,
        ),
    )

    return matches[:max_per_episode]
