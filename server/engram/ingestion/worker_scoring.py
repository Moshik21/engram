"""Deterministic triage scoring used by the background episode worker."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from engram.config import ActivationConfig
from engram.extraction.discourse import classify_discourse
from engram.retrieval.goals import compute_goal_triage_boost, identify_active_goals
from engram.retrieval.triage_policy import TriageDecision, apply_episode_utility_policy
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex

if TYPE_CHECKING:
    from engram.retrieval.triage_scorer import TriageScorer

logger = logging.getLogger(__name__)


class EpisodeWorkerScoringService:
    """Score queued worker episodes without calling an LLM."""

    def __init__(
        self,
        *,
        graph: GraphStore,
        activation: ActivationStore,
        search: SearchIndex,
        cfg: ActivationConfig,
    ) -> None:
        self._graph = graph
        self._activation = activation
        self._search = search
        self._cfg = cfg
        self._scorer: TriageScorer | None = None

    def _get_scorer(self) -> TriageScorer | None:
        """Lazy-init multi-signal scorer shared with consolidation in-process."""
        if self._scorer is None and self._cfg.triage_multi_signal_enabled:
            from engram.retrieval.triage_scorer import get_shared_triage_scorer

            self._scorer = get_shared_triage_scorer(self._cfg)
        return self._scorer

    async def score(
        self,
        content: str,
        group_id: str = "default",
    ) -> tuple[TriageDecision, Any | None]:
        """Score episode content and return the worker routing decision."""
        if not content:
            return (
                apply_episode_utility_policy(
                    "",
                    self._cfg,
                    0.0,
                    discourse_class="world",
                    mode="worker" if self._cfg.triage_multi_signal_enabled else "phase",
                    score_source="empty",
                ),
                None,
            )

        discourse_class = classify_discourse(content)

        scorer = self._get_scorer()
        if scorer is not None:
            signals = await scorer.score(
                content=content,
                search_index=self._search,
                graph_store=self._graph,
                activation_store=self._activation,
                group_id=group_id,
            )
            return (
                apply_episode_utility_policy(
                    content,
                    self._cfg,
                    signals.composite,
                    discourse_class=discourse_class,
                    mode="worker",
                    score_source="multi_signal",
                ),
                signals,
            )

        length_score = min(len(content) / 500, 1.0) * 0.25
        caps = len(re.findall(r"\b[A-Z][a-z]+\b", content))
        keyword_score = min(caps / 10, 1.0) * 0.20
        novelty_score = 0.15

        emotional_score = 0.0
        if self._cfg.emotional_salience_enabled:
            from engram.extraction.salience import compute_emotional_salience

            salience = compute_emotional_salience(content)
            emotional_score = salience.composite * self._cfg.emotional_triage_weight
            base_score = length_score + keyword_score + novelty_score + emotional_score
            if salience.composite >= self._cfg.triage_personal_floor_threshold:
                return (
                    apply_episode_utility_policy(
                        content,
                        self._cfg,
                        max(base_score, self._cfg.triage_personal_floor),
                        discourse_class=discourse_class,
                        mode="phase",
                        score_source="heuristic",
                    ),
                    None,
                )

        base_score = length_score + keyword_score + novelty_score + emotional_score

        if self._cfg.goal_priming_enabled:
            try:
                goals = await identify_active_goals(
                    self._graph,
                    self._activation,
                    group_id,
                    self._cfg,
                )
                base_score += compute_goal_triage_boost(content, goals, self._cfg)
            except Exception:
                logger.debug("Worker: goal boost failed", exc_info=True)

        return (
            apply_episode_utility_policy(
                content,
                self._cfg,
                base_score,
                discourse_class=discourse_class,
                mode="phase",
                score_source="heuristic",
            ),
            None,
        )

    async def record_projection_outcome(
        self,
        episode_id: str,
        group_id: str,
        signals: Any | None,
    ) -> None:
        """Feed successful worker projections back into the shared scorer."""
        if signals is None:
            return
        get_episode_entities = getattr(self._graph, "get_episode_entities", None)
        if get_episode_entities is None:
            return
        try:
            entity_ids = await get_episode_entities(episode_id, group_id=group_id)
        except Exception:
            return
        if isinstance(entity_ids, list):
            scorer = self._get_scorer()
            if scorer is not None:
                scorer.record_outcome(signals, len(entity_ids))
