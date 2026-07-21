"""Recall telemetry, interaction, and usage-detection helpers."""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass
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

logger = logging.getLogger(__name__)


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


# --- M1.4: echo-guarded citation scan (surfaced -> used tier) -----------------
#
# Recall surfacing populates a bounded per-group ring buffer of surfaced
# payloads. The Capture fast path (store_episode via capture_surface) scans the
# next observed turn against it: a surfaced entity mentioned in NOVEL tokens
# (outside any echoed shingle span of the surfaced payload) records a
# `used`-tier access event. Everything is gated on
# `recall_usage_feedback_enabled` (default False) and `w_used` stays 0 in
# ranking until G7 passes — usage_events accumulate inertly in M1.

_USAGE_RING_CAP = 32
_USAGE_DEDUP_WINDOW_SECONDS = 30 * 60.0
_USAGE_SNIPPET_MAX_CHARS = 400
_USAGE_SHINGLE_TOKENS = 5
_USAGE_MIN_SHINGLE_TOKENS = 3
_USAGE_CONTEXT_WINDOW_TOKENS = 12
_USAGE_CONTEXT_MIN_TOKEN_LEN = 4
_USAGE_PROMOTION_ENTITY_CAP = 32


@dataclass(frozen=True)
class SurfacedEntry:
    """One surfaced recall payload remembered for the citation scan."""

    entity_id: str
    name: str
    ts: float
    snippet_tokens: tuple[str, ...]


class SurfacedUsageBuffer:
    """Bounded per-group ring buffer of surfaced payloads + used-event dedup."""

    def __init__(
        self,
        *,
        cap: int = _USAGE_RING_CAP,
        dedup_window_seconds: float = _USAGE_DEDUP_WINDOW_SECONDS,
    ) -> None:
        self._cap = cap
        self._dedup_window = dedup_window_seconds
        self._entries: dict[str, deque[SurfacedEntry]] = {}
        self._last_used_event: dict[tuple[str, str], float] = {}
        # Mask-only ring: shingle sources for ALL surfaced text (episode
        # content, packet summaries) — not entity-bound. Without this, an
        # agent echoing a surfaced EPISODE verbatim reads as "novel" tokens
        # relative to the entity snippets and fires false used events.
        self._text_masks: dict[str, deque[tuple[str, ...]]] = {}

    def note_surfaced(
        self,
        group_id: str,
        *,
        entity_id: str,
        name: str,
        snippet: str,
        ts: float,
    ) -> None:
        if not entity_id or not name:
            return
        tokens = tuple(_normalize_text((snippet or "")[:_USAGE_SNIPPET_MAX_CHARS]).split())
        ring = self._entries.setdefault(group_id, deque(maxlen=self._cap))
        ring.append(SurfacedEntry(entity_id=entity_id, name=name, ts=ts, snippet_tokens=tokens))

    def note_surfaced_text(self, group_id: str, text: str, ts: float) -> None:
        """Register surfaced payload text as an echo-mask source (mask-only).

        No entity binding and no used-event eligibility — these tokens only
        widen the echoed-span mask so parroted result text never counts as
        reliance.
        """
        del ts  # reserved for future age-out; the ring cap bounds growth
        tokens = tuple(_normalize_text((text or "")[: _USAGE_SNIPPET_MAX_CHARS * 4]).split())
        if not tokens:
            return
        ring = self._text_masks.setdefault(group_id, deque(maxlen=self._cap))
        ring.append(tokens)

    def is_empty(self, group_id: str) -> bool:
        return not self._entries.get(group_id)

    def scan_novel_mentions(
        self,
        group_id: str,
        content: str,
        now: float,
    ) -> list[SurfacedEntry]:
        """Return surfaced entries genuinely relied on by ``content``.

        Echo guard: token positions covered by shingles of the buffered
        surfaced payloads count as echoed; a mention wholly inside an echoed
        span never fires. Dedup: at most one used event per (entity, group)
        per rolling window.
        """
        ring = self._entries.get(group_id)
        if not ring:
            return []
        content_tokens = _normalize_text(content).split()
        if not content_tokens:
            return []

        mask_sources = [entry.snippet_tokens for entry in ring]
        mask_sources.extend(self._text_masks.get(group_id) or ())
        echoed = _echoed_token_mask(content_tokens, mask_sources)
        fired: list[SurfacedEntry] = []
        seen: set[str] = set()
        for entry in reversed(ring):
            if entry.entity_id in seen:
                continue
            seen.add(entry.entity_id)
            last = self._last_used_event.get((group_id, entry.entity_id))
            if last is not None and (now - last) < self._dedup_window:
                continue
            if _has_novel_mention(content_tokens, echoed, entry):
                self._last_used_event[(group_id, entry.entity_id)] = now
                fired.append(entry)
        self._prune_dedup(now)
        return fired

    def _prune_dedup(self, now: float) -> None:
        stale = [
            key for key, ts in self._last_used_event.items() if (now - ts) >= self._dedup_window
        ]
        for key in stale:
            del self._last_used_event[key]

    def reset(self) -> None:
        self._entries.clear()
        self._last_used_event.clear()
        self._text_masks.clear()


_USAGE_BUFFER = SurfacedUsageBuffer()


def note_surfaced_texts_from_response(
    group_id: str,
    response: dict,
    cfg,
    *,
    now: float | None = None,
) -> None:
    """Feed every surfaced result/packet text into the echo mask (mask-only).

    Called by the recall surface at response time so the citation scan can
    never mistake a parroted result for reliance. Flag-gated and best-effort:
    masking must never fail a recall.
    """
    if not getattr(cfg, "recall_usage_feedback_enabled", False):
        return
    try:
        ts = now if now is not None else time.time()
        buffer = get_usage_buffer()
        for result in response.get("results") or []:
            text = result.get("text") or result.get("content") or ""
            if text:
                buffer.note_surfaced_text(group_id, str(text), ts)
        for packet in response.get("packets") or []:
            text = packet.get("summary") or packet.get("text") or ""
            title = packet.get("title") or ""
            if text or title:
                buffer.note_surfaced_text(group_id, f"{title} {text}".strip(), ts)
    except Exception:
        # silent-ok: echo masking is protective telemetry; a mask miss only
        # risks an extra used event, never a failed recall.
        logger.debug("Surfaced-text mask feed failed", exc_info=True)


def get_usage_buffer() -> SurfacedUsageBuffer:
    """Process-wide surfaced-usage ring buffer shared by recall + capture."""
    return _USAGE_BUFFER


def _echoed_token_mask(
    content_tokens: list[str],
    snippet_token_lists: list[tuple[str, ...]],
) -> list[bool]:
    """Mark content token positions covered by surfaced-payload shingles."""
    shingles_by_len: dict[int, set[tuple[str, ...]]] = {}
    for tokens in snippet_token_lists:
        if len(tokens) >= _USAGE_SHINGLE_TOKENS:
            bucket = shingles_by_len.setdefault(_USAGE_SHINGLE_TOKENS, set())
            for i in range(len(tokens) - _USAGE_SHINGLE_TOKENS + 1):
                bucket.add(tokens[i : i + _USAGE_SHINGLE_TOKENS])
        elif len(tokens) >= _USAGE_MIN_SHINGLE_TOKENS:
            shingles_by_len.setdefault(len(tokens), set()).add(tokens)

    mask = [False] * len(content_tokens)
    for size, shingles in shingles_by_len.items():
        for i in range(len(content_tokens) - size + 1):
            if tuple(content_tokens[i : i + size]) in shingles:
                for j in range(i, i + size):
                    mask[j] = True
    return mask


def _has_novel_mention(
    content_tokens: list[str],
    echoed: list[bool],
    entry: SurfacedEntry,
) -> bool:
    """True when the entity is mentioned in novel (non-echoed) tokens.

    Single-token names additionally require the novel context around the
    mention to share vocabulary with the surfaced payload — an incidental
    common-word mention with no topical tie to what was surfaced does not
    count as reliance.
    """
    name_tokens = tuple(_normalize_text(entry.name).split())
    if not name_tokens:
        return False
    if len(name_tokens) == 1 and len(name_tokens[0]) < 3:
        return False

    n = len(name_tokens)
    snippet_context = {
        token
        for token in entry.snippet_tokens
        if len(token) >= _USAGE_CONTEXT_MIN_TOKEN_LEN and token not in name_tokens
    }
    for i in range(len(content_tokens) - n + 1):
        if tuple(content_tokens[i : i + n]) != name_tokens:
            continue
        span = range(i, i + n)
        if all(echoed[j] for j in span):
            continue
        if n == 1:
            lo = max(0, i - _USAGE_CONTEXT_WINDOW_TOKENS)
            hi = min(len(content_tokens), i + 1 + _USAGE_CONTEXT_WINDOW_TOKENS)
            context = {content_tokens[j] for j in range(lo, hi) if j != i and not echoed[j]}
            if not (context & snippet_context):
                continue
        return True
    return False


async def record_observed_usage_events(
    *,
    activation_store: ActivationStore,
    cfg: ActivationConfig,
    group_id: str,
    content: str,
    now: float | None = None,
    usage_buffer: SurfacedUsageBuffer | None = None,
) -> list[str]:
    """Record used-tier access events for surfaced entities relied on in content.

    Requires ``recall_usage_feedback_enabled=True``; short-circuits (one dict
    lookup) when no recall has surfaced entities for the group.
    """
    if not cfg.recall_usage_feedback_enabled:
        return []
    buffer = usage_buffer if usage_buffer is not None else _USAGE_BUFFER
    if buffer.is_empty(group_id):
        return []
    ts = now if now is not None else time.time()
    fired = buffer.scan_novel_mentions(group_id, content, ts)
    for entry in fired:
        await activation_store.record_access(
            entry.entity_id,
            ts,
            group_id=group_id,
            tier="used",
        )
    return [entry.entity_id for entry in fired]


class RecallEntityAccessRecorder:
    """Record true Recall-stage entity access and reconsolidation side effects."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        activation_store: ActivationStore,
        event_bus: EventBus | None,
        labile_tracker: _LabileTracker | None,
        usage_buffer: SurfacedUsageBuffer | None = None,
    ) -> None:
        self._cfg = cfg
        self._activation = activation_store
        self._event_bus = event_bus
        self._labile_tracker = labile_tracker
        self._usage_buffer = usage_buffer if usage_buffer is not None else _USAGE_BUFFER

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
        tier: str = "surfaced",
    ) -> None:
        now = timestamp if timestamp is not None else time.time()
        await self._activation.record_access(entity.id, now, group_id=group_id, tier=tier)
        if tier == "surfaced" and self._cfg.recall_usage_feedback_enabled:
            self._usage_buffer.note_surfaced(
                group_id,
                entity_id=entity.id,
                name=entity.name,
                snippet=f"{entity.name} {entity.summary or ''}",
                ts=now,
            )
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
        activation_store: ActivationStore | None = None,
    ) -> None:
        self._cfg = cfg
        self._graph = graph_store
        self._projection_policy = projection_policy
        self._recall_need_controller = recall_need_controller
        self._event_bus = event_bus
        self._activation = activation_store

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
                memory_id=f"cue:{episode.id}",
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
            await self._record_promotion_usage(episode)
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

    async def _record_promotion_usage(self, episode: Episode) -> None:
        """M1.2: cue-hit promotion is a confirmed-tier usage signal.

        Records a confirmed-tier access event for the entities actually linked
        to the promoted episode at promotion time (bounded). Gated on
        `recall_usage_feedback_enabled` so default-config recall output stays
        byte-identical (M1 inertness invariant).
        """
        if self._activation is None or not self._cfg.recall_usage_feedback_enabled:
            return
        try:
            entity_ids = await self._graph.get_episode_entities(
                episode.id,
                group_id=episode.group_id,
            )
        except Exception:
            # silent-ok: usage tagging must never fail or delay cue promotion.
            logger.debug("Cue promotion usage lookup failed", exc_info=True)
            return
        now = time.time()
        for entity_id in list(entity_ids or [])[:_USAGE_PROMOTION_ENTITY_CAP]:
            if not entity_id:
                continue
            try:
                await self._activation.record_access(
                    entity_id,
                    now,
                    group_id=episode.group_id,
                    tier="confirmed",
                )
            except Exception:
                # silent-ok: usage tagging must never fail or delay cue promotion.
                logger.debug("Cue promotion usage record failed", exc_info=True)

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
            memory_id=memory_id,
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

    # M1.2: explicit feedback records tier-tagged access events.
    _ACCESS_TIER_BY_INTERACTION = {
        "used": "used",
        "confirmed": "confirmed",
        "corrected": "corrected",
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

        access_tier = self._ACCESS_TIER_BY_INTERACTION.get(interaction_type)
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
                access_tier=access_tier,
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
            memory_id=f"cue:{episode_id}",
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
        access_tier: str | None,
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
        if access_tier is not None and entity is not None:
            await self._entity_access_recorder.record_entity_access(
                entity,
                group_id=group_id,
                query=query,
                source=source,
                timestamp=timestamp,
                tier=access_tier,
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
