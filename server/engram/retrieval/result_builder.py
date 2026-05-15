"""Raw recall result assembly.

This module builds the internal recall dictionaries emitted by GraphManager.
Surface-specific formatting still belongs in ``retrieval.presenter``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.episode_cue import EpisodeCue
from engram.models.relationship import Relationship
from engram.retrieval.scorer import ScoredResult


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _enum_value(value: object) -> object:
    enum_value = getattr(value, "value", None)
    return enum_value if isinstance(enum_value, str) else value


class RecallResultBuilder:
    """Build GraphManager's raw recall result contract."""

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg

    @staticmethod
    def episode_projection_state_value(episode: object | None) -> str | None:
        """Normalize episode projection state to its string value."""
        if episode is None:
            return None
        state = getattr(episode, "projection_state", None)
        value = getattr(state, "value", None)
        if isinstance(value, str):
            return value
        return state if isinstance(state, str) else None

    @staticmethod
    def cue_result_payload(cue: EpisodeCue, *, hit_increment: int = 0) -> dict[str, object]:
        return {
            "episode_id": cue.episode_id,
            "cue_text": cue.cue_text,
            "supporting_spans": cue.first_spans,
            "projection_state": _enum_value(cue.projection_state),
            "route_reason": cue.route_reason,
            "hit_count": (cue.hit_count or 0) + hit_increment,
            "surfaced_count": cue.surfaced_count,
            "selected_count": cue.selected_count,
            "used_count": cue.used_count,
            "near_miss_count": cue.near_miss_count,
            "policy_score": cue.policy_score,
            "last_feedback_at": _isoformat(cue.last_feedback_at),
            "last_projected_at": _isoformat(cue.last_projected_at),
        }

    def truncate_episode_content(self, episode: Episode, cue: EpisodeCue | None = None) -> str:
        """Truncate episode content based on memory tier and recall config."""
        limit = self._cfg.recall_episode_content_limit

        if self._cfg.recall_tier_aware_truncation_enabled:
            tier = getattr(episode, "memory_tier", "episodic") or "episodic"
            if tier == "transitional":
                limit = self._cfg.recall_transitional_content_limit
                if cue is not None and cue.cue_text:
                    return cue.cue_text[:limit] if limit > 0 else cue.cue_text
            elif tier == "semantic":
                limit = self._cfg.recall_semantic_content_limit
                if cue is not None and cue.cue_text:
                    return cue.cue_text[:limit] if limit > 0 else cue.cue_text

        return episode.content[:limit] if limit > 0 else episode.content

    def episode_result(
        self,
        episode: Episode,
        scored_result: ScoredResult,
        *,
        linked_entities: Sequence[Any],
    ) -> dict[str, Any]:
        result = self.synthetic_episode_result(
            episode,
            score=scored_result.score,
            score_breakdown=self._base_score_breakdown(scored_result),
            linked_entities=linked_entities,
        )
        if scored_result.chunk_context:
            result["chunk_context"] = scored_result.chunk_context
        return result

    def cue_episode_result(
        self,
        episode: Episode,
        cue: EpisodeCue,
        scored_result: ScoredResult,
        *,
        linked_entities: Sequence[Any],
        hit_increment: int = 0,
    ) -> dict[str, Any]:
        return {
            "cue": self.cue_result_payload(cue, hit_increment=hit_increment),
            "episode": self._episode_metadata(episode, include_content=False),
            "score": scored_result.score,
            "score_breakdown": self._base_score_breakdown(scored_result),
            "result_type": "cue_episode",
            "linked_entities": list(linked_entities),
        }

    def synthetic_episode_result(
        self,
        episode: Episode,
        *,
        score: float,
        score_breakdown: dict[str, Any],
        linked_entities: Sequence[Any],
    ) -> dict[str, Any]:
        return {
            "episode": self._episode_metadata(episode, include_content=True),
            "score": score,
            "score_breakdown": score_breakdown,
            "result_type": "episode",
            "linked_entities": list(linked_entities),
        }

    def entity_result(
        self,
        entity: Entity,
        relationships: Sequence[Relationship],
        scored_result: ScoredResult,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "result_type": "entity",
            "entity": {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type,
                "summary": entity.summary,
            },
            "score": scored_result.score,
            "score_breakdown": self._entity_score_breakdown(scored_result),
            "relationships": [
                {
                    "id": rel.id,
                    "predicate": rel.predicate,
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "weight": rel.weight,
                    "polarity": rel.polarity,
                }
                for rel in relationships[:5]
            ],
        }
        if scored_result.planner_intents:
            result["supporting_intents"] = scored_result.planner_intents
        if scored_result.recall_trace:
            result["recall_trace"] = scored_result.recall_trace
        if entity.entity_type == "Intention" and self._cfg.prospective_graph_embedded:
            self._attach_intention_meta(result, entity, scored_result)
        return result

    def _episode_metadata(self, episode: Episode, *, include_content: bool) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "id": episode.id,
            "source": episode.source,
            "created_at": _isoformat(episode.created_at),
            "conversation_date": _isoformat(getattr(episode, "conversation_date", None)),
        }
        if include_content:
            metadata["content"] = self.truncate_episode_content(episode)
        return metadata

    @staticmethod
    def _base_score_breakdown(scored_result: ScoredResult) -> dict[str, float]:
        return {
            "semantic": scored_result.semantic_similarity,
            "activation": scored_result.activation,
            "edge_proximity": scored_result.edge_proximity,
            "exploration_bonus": scored_result.exploration_bonus,
        }

    @staticmethod
    def _entity_score_breakdown(scored_result: ScoredResult) -> dict[str, Any]:
        breakdown: dict[str, Any] = RecallResultBuilder._base_score_breakdown(scored_result)
        breakdown["hop_distance"] = scored_result.hop_distance
        breakdown["planner_support"] = scored_result.planner_support
        return breakdown

    @staticmethod
    def _attach_intention_meta(
        result: dict[str, Any],
        entity: Entity,
        scored_result: ScoredResult,
    ) -> None:
        try:
            from engram.models.prospective import IntentionMeta

            meta = IntentionMeta(**(entity.attributes or {}))
            warmth_ratio = (
                scored_result.activation / meta.activation_threshold
                if meta.activation_threshold > 0
                else 0.0
            )
            result["intention_meta"] = {
                "warmth_ratio": round(warmth_ratio, 4),
                "fire_count": meta.fire_count,
                "max_fires": meta.max_fires,
                "action_text": meta.action_text,
                "priority": meta.priority,
            }
        except Exception:
            pass
