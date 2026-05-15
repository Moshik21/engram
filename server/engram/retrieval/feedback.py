"""Recall telemetry, interaction, and usage-detection helpers."""

from __future__ import annotations

import re
import time
from typing import Protocol

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.extraction.policy import ProjectionPolicy
from engram.ingestion.projection_state import sync_projection_state
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.recall import MemoryInteractionEvent, MemoryNeed
from engram.retrieval.control import RecallNeedController
from engram.storage.protocols import ActivationStore, GraphStore
from engram.utils.dates import utc_now


class _LabileTracker(Protocol):
    def mark_labile(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        summary: str,
        query: str,
    ) -> None: ...


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _projection_state_value(episode: Episode) -> str | None:
    state = episode.projection_state
    value = getattr(state, "value", None)
    if isinstance(value, str):
        return value
    return state if isinstance(state, str) else None


def publish_memory_need_analysis(
    event_bus: EventBus | None,
    group_id: str,
    need: MemoryNeed,
    *,
    source: str,
    mode: str,
    turn_text: str,
) -> None:
    """Publish a recall.need.analyzed event if an event bus is available."""
    if event_bus is None:
        return
    event_bus.publish(
        group_id,
        "recall.need.analyzed",
        need.to_payload(
            source=source,
            mode=mode,
            turn_preview=turn_text.strip()[:200],
        ),
    )


def publish_memory_interaction(
    event_bus: EventBus | None,
    interaction: MemoryInteractionEvent,
) -> None:
    """Publish a single recall.interaction event."""
    if event_bus is None:
        return
    event_bus.publish(
        interaction.group_id,
        "recall.interaction",
        interaction.to_payload(),
    )


async def publish_activation_access(
    *,
    event_bus: EventBus | None,
    activation_store: ActivationStore,
    cfg: ActivationConfig,
    entity_id: str,
    name: str,
    entity_type: str,
    group_id: str,
    accessed_via: str,
) -> None:
    """Publish an activation.access event with the entity's current activation."""
    if event_bus is None:
        return

    from engram.activation.engine import compute_activation

    now = time.time()
    state = await activation_store.get_activation(entity_id)
    activation = 0.0
    if state:
        activation = compute_activation(state.access_history, now, cfg)
    event_bus.publish(
        group_id,
        "activation.access",
        {
            "entityId": entity_id,
            "name": name,
            "entityType": entity_type,
            "activation": round(activation, 4),
            "accessedVia": accessed_via,
        },
    )


class RecallEntityAccessRecorder:
    """Record true Recall-stage entity access and reconsolidation side effects."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        activation_store: ActivationStore,
        event_bus: EventBus | None,
        labile_tracker: _LabileTracker | None,
    ) -> None:
        self._cfg = cfg
        self._activation = activation_store
        self._event_bus = event_bus
        self._labile_tracker = labile_tracker

    async def publish_access_event(
        self,
        *,
        entity_id: str,
        name: str,
        entity_type: str,
        group_id: str,
        accessed_via: str,
    ) -> None:
        await publish_activation_access(
            event_bus=self._event_bus,
            activation_store=self._activation,
            cfg=self._cfg,
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            group_id=group_id,
            accessed_via=accessed_via,
        )

    async def record_entity_access(
        self,
        entity: Entity,
        *,
        group_id: str,
        query: str,
        source: str,
        timestamp: float | None = None,
    ) -> None:
        now = timestamp if timestamp is not None else time.time()
        await self._activation.record_access(entity.id, now, group_id=group_id)
        await self.publish_access_event(
            entity_id=entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            group_id=group_id,
            accessed_via=source,
        )

        if self._labile_tracker is not None:
            self._labile_tracker.mark_labile(
                entity.id,
                entity.name,
                entity.entity_type,
                entity.summary or "",
                query,
            )


class RecallCueFeedbackRecorder:
    """Record Recall-stage cue feedback and schedule hot cues for projection."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        graph_store: GraphStore,
        projection_policy: ProjectionPolicy,
        recall_need_controller: RecallNeedController,
        event_bus: EventBus | None,
    ) -> None:
        self._cfg = cfg
        self._graph = graph_store
        self._projection_policy = projection_policy
        self._recall_need_controller = recall_need_controller
        self._event_bus = event_bus

    async def record_cue_feedback(
        self,
        episode: Episode,
        score: float,
        query: str,
        *,
        interaction_type: str | None = None,
        near_miss: bool = False,
        count_hit: bool = True,
    ) -> None:
        cue = await self._graph.get_episode_cue(episode.id, episode.group_id)
        if cue is None:
            return

        now_dt = utc_now()
        feedback_type = "near_miss" if near_miss else (interaction_type or "surfaced")
        feedback = self._projection_policy.apply_feedback(
            cue,
            interaction_type=feedback_type,
            score=score,
            count_hit=count_hit,
        )
        cue_updates: dict[str, object] = dict(feedback.updates)
        cue_updates["last_feedback_at"] = now_dt
        if not near_miss and "hit_count" in cue_updates:
            cue_updates["last_hit_at"] = now_dt

        current_projection_state = _projection_state_value(episode)
        event_payload = {
            "episodeId": episode.id,
            "projectionState": current_projection_state,
            "interactionType": feedback_type,
            "score": round(score, 4),
            "query": query[:200],
        }
        if "hit_count" in cue_updates:
            event_payload["hitCount"] = cue_updates["hit_count"]
        if "policy_score" in cue_updates:
            event_payload["policyScore"] = cue_updates["policy_score"]
        self._publish(
            episode.group_id,
            "cue.hit" if not near_miss else "cue.near_miss",
            event_payload,
        )

        if not near_miss and interaction_type in {"surfaced", "selected"}:
            self._recall_need_controller.record_interaction(
                episode.group_id,
                interaction_type,
                result_type="cue_episode",
            )

        promotable_states = {
            EpisodeProjectionState.CUED.value,
            EpisodeProjectionState.CUE_ONLY.value,
            EpisodeProjectionState.QUEUED.value,
            EpisodeProjectionState.FAILED.value,
        }
        hit_count = _coerce_int(
            cue_updates.get("hit_count", cue.hit_count or 0),
            cue.hit_count or 0,
        )
        should_promote = (
            hit_count >= self._cfg.cue_recall_hit_threshold or feedback.should_promote
        ) and current_projection_state in promotable_states

        if should_promote:
            promotion_reason = (
                "cue_recall_hits"
                if hit_count >= self._cfg.cue_recall_hit_threshold
                else (feedback.promotion_reason or "cue_policy")
            )
            await sync_projection_state(
                self._graph,
                episode.id,
                group_id=episode.group_id,
                state=EpisodeProjectionState.SCHEDULED,
                reason=promotion_reason,
                episode_updates={"status": EpisodeStatus.QUEUED.value, "error": None},
                cue_reason=promotion_reason,
                cue_updates=cue_updates,
                cue_layer_enabled=True,
                log_prefix="Recall cue feedback",
            )
            self._publish(
                episode.group_id,
                "cue.promoted",
                {
                    "episodeId": episode.id,
                    "hitCount": hit_count,
                    "reason": promotion_reason,
                    "score": round(score, 4),
                    "policyScore": cue_updates.get("policy_score", cue.policy_score),
                },
            )
            self._publish(
                episode.group_id,
                "episode.projection_scheduled",
                {
                    "episodeId": episode.id,
                    "reason": promotion_reason,
                    "hitCount": hit_count,
                },
            )
            return

        if self._cfg.cue_policy_learning_enabled and "policy_score" in cue_updates:
            self._publish(
                episode.group_id,
                "cue.policy_updated",
                {
                    "episodeId": episode.id,
                    "interactionType": feedback_type,
                    "policyScore": cue_updates["policy_score"],
                    "projectionState": current_projection_state,
                },
            )

        await self._graph.update_episode_cue(episode.id, cue_updates, group_id=episode.group_id)

    def _publish(self, group_id: str, event_type: str, payload: dict | None = None) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(group_id, event_type, payload)


class RecallInteractionRecorder:
    """Publish Recall-stage interaction telemetry and update recall-need learning."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        event_bus: EventBus | None,
        recall_need_controller: RecallNeedController,
    ) -> None:
        self._cfg = cfg
        self._event_bus = event_bus
        self._recall_need_controller = recall_need_controller

    def record_entity_interaction(
        self,
        *,
        group_id: str,
        entity: Entity,
        interaction_type: str | None,
        source: str,
        query: str,
        score: float,
        recorded_access: bool,
    ) -> None:
        self.record_memory_interaction(
            group_id=group_id,
            memory_id=entity.id,
            entity_name=entity.name,
            entity_type=entity.entity_type,
            interaction_type=interaction_type,
            source=source,
            query=query,
            score=score,
            recorded_access=recorded_access,
        )

    def record_memory_interaction(
        self,
        *,
        group_id: str,
        memory_id: str,
        entity_name: str | None,
        entity_type: str | None,
        interaction_type: str | None,
        source: str,
        query: str,
        score: float | None,
        recorded_access: bool,
        result_type: str = "entity",
    ) -> None:
        if not interaction_type:
            return

        if self._cfg.recall_telemetry_enabled or self._cfg.recall_usage_feedback_enabled:
            publish_memory_interaction(
                self._event_bus,
                MemoryInteractionEvent(
                    group_id=group_id,
                    entity_id=memory_id,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    interaction_type=interaction_type,
                    source=source,
                    query=query,
                    score=score,
                    recorded_access=recorded_access,
                ),
            )

        self._recall_need_controller.record_interaction(
            group_id,
            interaction_type,
            result_type=result_type,
        )


class RecallMemoryInteractionApplier:
    """Apply explicit post-response memory feedback to recalled entities and cues."""

    _VALID_TYPES = {
        "surfaced",
        "selected",
        "used",
        "confirmed",
        "dismissed",
        "corrected",
    }

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        cue_feedback_recorder: RecallCueFeedbackRecorder,
        entity_access_recorder: RecallEntityAccessRecorder,
        interaction_recorder: RecallInteractionRecorder,
        recall_need_controller: RecallNeedController,
    ) -> None:
        self._cfg = cfg
        self._graph = graph_store
        self._activation = activation_store
        self._cue_feedback_recorder = cue_feedback_recorder
        self._entity_access_recorder = entity_access_recorder
        self._interaction_recorder = interaction_recorder
        self._recall_need_controller = recall_need_controller

    async def apply(
        self,
        memory_ids: list[str],
        *,
        interaction_type: str,
        group_id: str = "default",
        query: str = "",
        source: str = "recall_feedback",
        result_lookup: dict[str, dict] | None = None,
    ) -> None:
        if interaction_type not in self._VALID_TYPES:
            raise ValueError(f"Unknown interaction_type: {interaction_type}")

        should_record_access = interaction_type in {"used", "confirmed"}
        should_record_positive = interaction_type == "confirmed" and self._cfg.ts_enabled
        should_record_negative = interaction_type == "corrected" and self._cfg.ts_enabled

        seen_ids: set[str] = set()
        now = time.time()
        for memory_id in memory_ids:
            if not memory_id or memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)

            metadata = result_lookup.get(memory_id, {}) if result_lookup else {}
            result_type = metadata.get("result_type")
            if result_type is None and isinstance(memory_id, str) and memory_id.startswith("cue:"):
                result_type = "cue_episode"
            if result_type == "cue_episode":
                await self._apply_cue_interaction(
                    memory_id,
                    metadata,
                    group_id=group_id,
                    query=query,
                    interaction_type=interaction_type,
                )
                continue

            await self._apply_entity_interaction(
                memory_id,
                metadata,
                group_id=group_id,
                query=query,
                source=source,
                interaction_type=interaction_type,
                should_record_access=should_record_access,
                should_record_positive=should_record_positive,
                should_record_negative=should_record_negative,
                timestamp=now,
            )

    async def _apply_cue_interaction(
        self,
        memory_id: str,
        metadata: dict,
        *,
        group_id: str,
        query: str,
        interaction_type: str,
    ) -> None:
        episode_id = metadata.get("episode_id")
        if not episode_id and isinstance(memory_id, str) and memory_id.startswith("cue:"):
            episode_id = memory_id.split(":", 1)[1]
        if not episode_id:
            return
        episode = await self._graph.get_episode_by_id(episode_id, group_id)
        if episode is None:
            return

        cue_score = metadata.get("score")
        await self._cue_feedback_recorder.record_cue_feedback(
            episode,
            float(cue_score) if cue_score is not None else 0.0,
            query,
            interaction_type=interaction_type,
            count_hit=bool(metadata.get("count_hit", False)),
        )
        self._recall_need_controller.record_interaction(
            group_id,
            interaction_type,
            result_type="cue_episode",
        )

    async def _apply_entity_interaction(
        self,
        memory_id: str,
        metadata: dict,
        *,
        group_id: str,
        query: str,
        source: str,
        interaction_type: str,
        should_record_access: bool,
        should_record_positive: bool,
        should_record_negative: bool,
        timestamp: float,
    ) -> None:
        entity_name = metadata.get("entity_name")
        entity_type = metadata.get("entity_type")
        score = metadata.get("score")

        entity = await self._graph.get_entity(memory_id, group_id)
        if entity is not None:
            entity_name = entity.name
            entity_type = entity.entity_type

        recorded_access = False
        if should_record_access and entity is not None:
            await self._entity_access_recorder.record_entity_access(
                entity,
                group_id=group_id,
                query=query,
                source=source,
                timestamp=timestamp,
            )
            recorded_access = True

        if should_record_positive:
            from engram.activation.feedback import record_positive_feedback

            await record_positive_feedback(memory_id, self._activation, self._cfg)

        if should_record_negative:
            from engram.activation.feedback import record_negative_feedback

            await record_negative_feedback(memory_id, self._activation, self._cfg)

        self._interaction_recorder.record_memory_interaction(
            group_id=group_id,
            memory_id=memory_id,
            entity_name=entity_name,
            entity_type=entity_type,
            interaction_type=interaction_type,
            source=source,
            query=query,
            score=score,
            recorded_access=recorded_access,
            result_type="entity",
        )


def extract_recall_targets(recall_results: list[dict]) -> list[dict]:
    """Extract deduplicated feedback targets from raw recall results."""
    targets: list[dict] = []
    seen_ids: set[str] = set()

    for result in recall_results:
        result_type = result.get("result_type")
        cue = result.get("cue")
        if result_type == "cue_episode" or (result_type is None and isinstance(cue, dict)):
            if not isinstance(cue, dict):
                continue
            episode = result.get("episode", {})
            episode_id = cue.get("episode_id") or episode.get("id")
            if not episode_id:
                continue
            lookup_id = f"cue:{episode_id}"
            if lookup_id in seen_ids:
                continue
            seen_ids.add(lookup_id)
            targets.append(
                {
                    "lookup_id": lookup_id,
                    "result_type": "cue_episode",
                    "episode_id": episode_id,
                    "cue_text": cue.get("cue_text"),
                    "supporting_spans": cue.get("supporting_spans", []),
                    "score": result.get("score"),
                    # Post-response upgrades should not double-count the initial cue hit.
                    "count_hit": False,
                }
            )
            continue

        entity = result.get("entity")
        if not isinstance(entity, dict):
            continue
        entity_id = entity.get("id")
        if not entity_id or entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)
        targets.append(
            {
                "lookup_id": entity_id,
                "result_type": "entity",
                "entity_id": entity_id,
                "entity_name": entity.get("name"),
                "entity_type": entity.get("type"),
                "score": result.get("score"),
            }
        )

    return targets


def extract_recall_entities(recall_results: list[dict]) -> list[dict]:
    """Extract deduplicated entity metadata from raw recall results."""
    return [
        target
        for target in extract_recall_targets(recall_results)
        if target.get("result_type") == "entity"
    ]


def partition_recall_targets_by_usage(
    response_text: str,
    recall_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition recalled entities/cues into used vs dismissed via response mention."""
    targets = extract_recall_targets(recall_results)
    if not targets:
        return [], []

    normalized_response = _normalize_text(response_text)
    if not normalized_response:
        return [], targets

    used: list[dict] = []
    dismissed: list[dict] = []
    haystack = f" {normalized_response} "

    for target in targets:
        if target.get("result_type") == "cue_episode":
            if _matches_cue_content(
                haystack,
                target.get("cue_text"),
                target.get("supporting_spans", []),
            ):
                used.append(target)
            else:
                dismissed.append(target)
            continue

        name = target.get("entity_name")
        if _matches_entity_name(haystack, name):
            used.append(target)
        else:
            dismissed.append(target)

    return used, dismissed


def partition_recall_entities_by_usage(
    response_text: str,
    recall_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition recalled entities into used vs dismissed via response mention."""
    used, dismissed = partition_recall_targets_by_usage(response_text, recall_results)
    return (
        [target for target in used if target.get("result_type") == "entity"],
        [target for target in dismissed if target.get("result_type") == "entity"],
    )


def _matches_entity_name(normalized_response: str, entity_name: str | None) -> bool:
    """Heuristic full-name match against normalized response text."""
    normalized_name = _normalize_text(entity_name or "")
    if not normalized_name:
        return False

    parts = normalized_name.split()
    if len(parts) == 1 and len(parts[0]) < 3:
        return False

    pattern = rf"(?<![a-z0-9]){re.escape(normalized_name)}(?![a-z0-9])"
    return re.search(pattern, normalized_response) is not None


def _matches_cue_content(
    normalized_response: str,
    cue_text: str | None,
    supporting_spans: list[str] | None,
) -> bool:
    """Heuristic span match for cue-backed recall results."""
    candidates = list(supporting_spans or [])
    if cue_text:
        candidates.append(cue_text)

    for candidate in candidates:
        if _matches_text_fragment(normalized_response, candidate):
            return True
    return False


def _matches_text_fragment(normalized_response: str, text: str | None) -> bool:
    """Match a meaningful fragment of cue text against the normalized response."""
    normalized_text = _normalize_text(text or "")
    if not normalized_text:
        return False
    if len(normalized_text) >= 12 and _contains_phrase(normalized_response, normalized_text):
        return True

    label_tokens = {"mentions", "spans", "quotes", "time"}
    tokens = [
        token for token in normalized_text.split() if len(token) >= 4 and token not in label_tokens
    ]
    for size in range(min(4, len(tokens)), 1, -1):
        for idx in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[idx : idx + size])
            if len(phrase) < 10:
                continue
            if _contains_phrase(normalized_response, phrase):
                return True
    return False


def _contains_phrase(normalized_response: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return re.search(pattern, normalized_response) is not None


def _normalize_text(text: str) -> str:
    """Normalize text for cheap mention matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
