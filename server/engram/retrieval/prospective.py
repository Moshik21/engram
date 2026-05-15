"""Prospective memory trigger matching (Wave 4)."""

from __future__ import annotations

import inspect
import logging
import math
import time
import uuid
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta
from typing import Any

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.prospective import Intention, IntentionMatch, IntentionMeta
from engram.models.relationship import Relationship
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex
from engram.utils.dates import utc_now, utc_now_iso

logger = logging.getLogger(__name__)

# Priority ordering for tie-breaking
_PRIORITY_ORDER = {"critical": 0, "high": 1, "normal": 2, "low": 3}


class ProspectiveMemoryService:
    """Create, list, dismiss, and update prospective memory intentions."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        search_index: SearchIndex,
        cfg: ActivationConfig,
        publish_event: Callable[[str, str, dict | None], None],
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index
        self._cfg = cfg
        self._publish = publish_event

    async def create_intention(
        self,
        trigger_text: str,
        action_text: str,
        trigger_type: str = "activation",
        entity_name: str | None = None,
        entity_names: list[str] | None = None,
        threshold: float | None = None,
        priority: str = "normal",
        group_id: str = "default",
        context: str | None = None,
        see_also: list[str] | None = None,
        refresh_trigger: str = "manual",
    ) -> str:
        """Create a prospective memory intention."""
        if not self._cfg.prospective_graph_embedded:
            v1_type = "semantic" if trigger_type == "activation" else trigger_type
            return await self._create_intention_v1(
                trigger_text,
                action_text,
                v1_type,
                entity_name,
                threshold,
                group_id,
            )

        if trigger_type not in ("activation", "entity_mention", "refresh_context"):
            raise ValueError(f"Invalid trigger_type: {trigger_type}")
        if trigger_type == "entity_mention" and not entity_names and not entity_name:
            raise ValueError("entity_names (or entity_name) required for entity_mention trigger")

        now = utc_now()
        intention_id = f"int_{uuid.uuid4().hex[:12]}"

        linked_entity_ids: list[str] = []
        resolved_names = entity_names or ([entity_name] if entity_name else [])
        for name in resolved_names:
            candidates = await self._graph.find_entity_candidates(name, group_id)
            if candidates:
                linked_entity_ids.append(candidates[0].id)

        meta = IntentionMeta(
            trigger_text=trigger_text,
            action_text=action_text,
            trigger_type=trigger_type,
            activation_threshold=threshold or self._cfg.prospective_activation_threshold,
            max_fires=self._cfg.prospective_max_fires,
            fire_count=0,
            enabled=True,
            expires_at=(now + timedelta(days=self._cfg.prospective_ttl_days)).isoformat(),
            trigger_entity_ids=linked_entity_ids,
            cooldown_seconds=self._cfg.prospective_cooldown_seconds,
            priority=priority,
            origin="explicit",
            context=context,
            see_also=see_also,
            refresh_trigger=refresh_trigger,
        )

        entity = Entity(
            id=intention_id,
            name=trigger_text,
            entity_type="Intention",
            summary=action_text,
            group_id=group_id,
            attributes=meta.model_dump(),
            created_at=now,
            updated_at=now,
        )
        await self._graph.create_entity(entity)
        await self._search.index_entity(entity)

        for entity_id in linked_entity_ids:
            relationship = Relationship(
                id=f"rel_{uuid.uuid4().hex[:12]}",
                source_id=intention_id,
                target_id=entity_id,
                predicate="TRIGGERED_BY",
                weight=0.9,
                group_id=group_id,
                source_episode=None,
            )
            await self._graph.create_relationship(relationship)

        await self._activation.record_access(intention_id, time.time(), group_id=group_id)
        self._publish(
            group_id,
            "intention.created",
            {
                "intentionId": intention_id,
                "triggerText": trigger_text,
                "actionText": action_text,
                "linkedEntityIds": linked_entity_ids,
                "threshold": meta.activation_threshold,
            },
        )

        logger.info("Created graph-embedded intention %s: %s", intention_id, trigger_text)
        return intention_id

    async def _create_intention_v1(
        self,
        trigger_text: str,
        action_text: str,
        trigger_type: str = "semantic",
        entity_name: str | None = None,
        threshold: float | None = None,
        group_id: str = "default",
    ) -> str:
        """v1 fallback: create intention in flat table."""
        if trigger_type not in ("semantic", "entity_mention"):
            raise ValueError(f"Invalid trigger_type: {trigger_type}")
        if trigger_type == "entity_mention" and not entity_name:
            raise ValueError("entity_name required for entity_mention trigger")

        now = utc_now()
        intention = Intention(
            id=f"int_{uuid.uuid4().hex[:12]}",
            trigger_text=trigger_text,
            action_text=action_text,
            trigger_type=trigger_type,
            entity_name=entity_name,
            threshold=threshold or self._cfg.prospective_similarity_threshold,
            max_fires=self._cfg.prospective_max_fires,
            fire_count=0,
            enabled=True,
            group_id=group_id,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(days=self._cfg.prospective_ttl_days),
        )
        return await self._graph.create_intention(intention)

    async def list_intentions(
        self,
        group_id: str = "default",
        enabled_only: bool = True,
    ) -> list:
        """List intentions from graph-embedded or flat-table storage."""
        if not self._cfg.prospective_graph_embedded:
            return await self._graph.list_intentions(group_id, enabled_only=enabled_only)

        entities = await self._graph.find_entities(
            entity_type="Intention",
            group_id=group_id,
            limit=100,
        )

        result = []
        now = utc_now()
        for entity in entities:
            attrs = entity.attributes or {}
            try:
                meta = IntentionMeta(**attrs)
            except Exception:
                continue

            if enabled_only:
                if not meta.enabled:
                    continue
                if meta.fire_count >= meta.max_fires:
                    continue
                if meta.expires_at:
                    try:
                        exp = datetime.fromisoformat(meta.expires_at)
                        if exp <= now:
                            continue
                    except (ValueError, TypeError):
                        pass

            result.append(entity)
        return result

    async def list_intention_views(
        self,
        group_id: str = "default",
        enabled_only: bool = True,
        *,
        surface: str,
    ) -> list[dict]:
        """Return API/MCP-ready intention rows with warmth metadata."""
        intentions = await self.list_intentions(group_id=group_id, enabled_only=enabled_only)
        if not self._cfg.prospective_graph_embedded:
            return [self._flat_intention_view(i, surface=surface) for i in intentions]

        from engram.activation.engine import compute_activation

        now = time.time()
        items: list[dict] = []
        for entity in intentions:
            attrs = entity.attributes or {}
            try:
                meta = IntentionMeta(**attrs)
            except Exception:
                continue

            state = await self._activation.get_activation(entity.id)
            activation = 0.0
            if state:
                activation = compute_activation(state.access_history, now, self._cfg)
            warmth_ratio = (
                activation / meta.activation_threshold if meta.activation_threshold > 0 else 0.0
            )
            items.append(
                self._embedded_intention_view(
                    entity_id=entity.id,
                    meta=meta,
                    warmth_ratio=warmth_ratio,
                    surface=surface,
                )
            )
        return items

    def _embedded_intention_view(
        self,
        *,
        entity_id: str,
        meta: IntentionMeta,
        warmth_ratio: float,
        surface: str,
    ) -> dict:
        if surface == "api":
            item = {
                "id": entity_id,
                "triggerText": meta.trigger_text,
                "actionText": meta.action_text,
                "triggerType": meta.trigger_type,
                "threshold": meta.activation_threshold,
                "fireCount": meta.fire_count,
                "maxFires": meta.max_fires,
                "enabled": meta.enabled,
                "priority": meta.priority,
                "expiresAt": meta.expires_at,
                "warmthRatio": round(warmth_ratio, 4),
                "linkedEntityIds": meta.trigger_entity_ids,
            }
            if meta.context is not None:
                item["context"] = meta.context
            if meta.see_also is not None:
                item["seeAlso"] = meta.see_also
            if meta.trigger_type == "refresh_context":
                item["refreshTrigger"] = meta.refresh_trigger
                item["lastRefreshed"] = meta.last_refreshed
                item["hasPinnedResult"] = bool(meta.pinned_result)
            return item

        item = {
            "id": entity_id,
            "trigger_text": meta.trigger_text,
            "action_text": meta.action_text,
            "trigger_type": meta.trigger_type,
            "threshold": meta.activation_threshold,
            "fire_count": meta.fire_count,
            "max_fires": meta.max_fires,
            "enabled": meta.enabled,
            "priority": meta.priority,
            "expires_at": meta.expires_at,
            "warmth_ratio": round(warmth_ratio, 4),
            "warmth_label": self._warmth_label(warmth_ratio),
            "linked_entity_ids": meta.trigger_entity_ids,
        }
        if meta.trigger_type == "refresh_context":
            item["refresh_trigger"] = meta.refresh_trigger
            item["last_refreshed"] = meta.last_refreshed
            if meta.pinned_result:
                item["has_pinned_result"] = True
        return item

    def _flat_intention_view(self, intention, *, surface: str) -> dict:
        if surface == "api":
            return {
                "id": intention.id,
                "triggerText": intention.trigger_text,
                "actionText": intention.action_text,
                "triggerType": intention.trigger_type,
                "threshold": intention.threshold,
                "fireCount": intention.fire_count,
                "maxFires": intention.max_fires,
                "enabled": intention.enabled,
            }
        return {
            "id": intention.id,
            "trigger_text": intention.trigger_text,
            "action_text": intention.action_text,
            "trigger_type": intention.trigger_type,
            "entity_name": intention.entity_name,
            "threshold": intention.threshold,
            "fire_count": intention.fire_count,
            "max_fires": intention.max_fires,
            "enabled": intention.enabled,
            "expires_at": intention.expires_at.isoformat() if intention.expires_at else None,
        }

    def _warmth_label(self, warmth_ratio: float) -> str:
        levels = self._cfg.prospective_warmth_levels
        if warmth_ratio >= 1.0:
            return "hot"
        if warmth_ratio >= levels[2]:
            return "warm"
        if warmth_ratio >= levels[1]:
            return "warming"
        if warmth_ratio >= levels[0]:
            return "cool"
        return "dormant"

    def effective_activation_threshold(self, threshold: float | None = None) -> float:
        """Return the creation threshold after applying runtime defaults."""
        return threshold or self._cfg.prospective_activation_threshold

    async def dismiss_intention(
        self,
        intention_id: str,
        group_id: str = "default",
        hard: bool = False,
    ) -> None:
        """Dismiss an intention. Soft-delete disables it; hard-delete removes it."""
        if not self._cfg.prospective_graph_embedded:
            await self._graph.delete_intention(intention_id, group_id, soft=not hard)
            return

        if hard:
            await self._delete_graph_intention_entity(intention_id, group_id)
        else:
            entity = await self._graph.get_entity(intention_id, group_id)
            if entity:
                attrs = dict(entity.attributes or {})
                attrs["enabled"] = False
                await self._graph.update_entity(
                    intention_id,
                    {"attributes": attrs},
                    group_id=group_id,
                )

        self._publish(
            group_id,
            "intention.dismissed",
            {
                "intentionId": intention_id,
                "hard": hard,
            },
        )

    async def _delete_graph_intention_entity(self, intention_id: str, group_id: str) -> None:
        delete_entity = self._graph.delete_entity
        delete_signature = inspect.signature(delete_entity)
        if "group_id" in delete_signature.parameters:
            group_param = delete_signature.parameters["group_id"]
            if group_param.kind is inspect.Parameter.KEYWORD_ONLY:
                await delete_entity(intention_id, soft=False, group_id=group_id)
            else:
                await delete_entity(intention_id, group_id)
        else:
            await delete_entity(intention_id, group_id)

    async def delete_intention(
        self,
        intention_id: str,
        group_id: str = "default",
    ) -> None:
        """Soft-delete an intention."""
        await self.dismiss_intention(intention_id, group_id, hard=False)

    async def migrate_flat_intentions(self, group_id: str = "default") -> int:
        """Migrate flat-table intentions to graph-embedded Entity nodes."""
        flat_intentions = await self._graph.list_intentions(group_id, enabled_only=False)
        migrated = 0
        for intention in flat_intentions:
            try:
                await self.create_intention(
                    trigger_text=intention.trigger_text,
                    action_text=intention.action_text,
                    trigger_type=(
                        "entity_mention"
                        if intention.trigger_type == "entity_mention"
                        else "activation"
                    ),
                    entity_name=intention.entity_name,
                    group_id=group_id,
                )
                await self._graph.delete_intention(intention.id, group_id, soft=True)
                migrated += 1
            except Exception:
                logger.warning("Failed to migrate intention %s", intention.id, exc_info=True)
        logger.info("Migrated %d flat-table intentions to graph-embedded", migrated)
        return migrated

    async def update_intention_fire(
        self,
        intention_id: str,
        group_id: str,
        episode_id: str | None = None,
    ) -> None:
        """Increment fire count and update last_fired for an intention."""
        entity = await self._graph.get_entity(intention_id, group_id)
        if not entity:
            return
        attrs = dict(entity.attributes or {})
        attrs["fire_count"] = attrs.get("fire_count", 0) + 1
        attrs["last_fired"] = utc_now_iso()
        await self._graph.update_entity(
            intention_id,
            {"attributes": attrs},
            group_id=group_id,
        )
        self._publish(
            group_id,
            "intention.triggered",
            {
                "intentionId": intention_id,
                "triggerText": attrs.get("trigger_text", ""),
                "actionText": attrs.get("action_text", ""),
                "activation": 0.0,
                "episodeId": episode_id,
            },
        )

    async def update_intention_meta(
        self,
        intention_id: str,
        group_id: str,
        updates: dict,
    ) -> None:
        """Update specific fields in an intention's IntentionMeta attributes."""
        entity = await self._graph.get_entity(intention_id, group_id)
        if not entity:
            return
        attrs = dict(entity.attributes or {})
        attrs.update(updates)
        await self._graph.update_entity(
            intention_id,
            {"attributes": attrs},
            group_id=group_id,
        )


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
        check_time = now or utc_now()
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
    now = utc_now()
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
            matches.append(
                IntentionMatch(
                    intention_id=intention.id,
                    trigger_text=intention.trigger_text,
                    action_text=intention.action_text,
                    similarity=1.0,
                    matched_via="entity_mention",
                )
            )

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
                    matches.append(
                        IntentionMatch(
                            intention_id=intention.id,
                            trigger_text=intention.trigger_text,
                            action_text=intention.action_text,
                            similarity=sim,
                            matched_via="semantic",
                        )
                    )

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
                if exp <= utc_now():
                    continue
            except (ValueError, TypeError):
                pass

        # Check cooldown
        if meta.last_fired and meta.cooldown_seconds > 0:
            try:
                last_dt = datetime.fromisoformat(meta.last_fired)
                elapsed = (utc_now() - last_dt).total_seconds()
                if elapsed < meta.cooldown_seconds:
                    continue
            except (ValueError, TypeError):
                pass

        # Fast path: entity_mention trigger
        if meta.trigger_type == "entity_mention":
            if meta.trigger_entity_ids and extracted_entity_ids & set(meta.trigger_entity_ids):
                matches.append(
                    IntentionMatch(
                        intention_id=entity.id,
                        trigger_text=meta.trigger_text,
                        action_text=meta.action_text,
                        similarity=1.0,
                        matched_via="entity_mention",
                        context=meta.context,
                        see_also=meta.see_also,
                    )
                )
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
            matches.append(
                IntentionMatch(
                    intention_id=entity.id,
                    trigger_text=meta.trigger_text,
                    action_text=meta.action_text,
                    similarity=round(total_activation, 4),
                    matched_via="activation",
                    context=meta.context,
                    see_also=meta.see_also,
                )
            )

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
