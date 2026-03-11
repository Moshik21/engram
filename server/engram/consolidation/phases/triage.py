"""Triage phase: score QUEUED episodes and selectively extract the top fraction.

Supports three scoring modes (checked in priority order):
1. Multi-signal scorer (triage_multi_signal_enabled) — deterministic utility features
2. LLM judge (triage_llm_judge_enabled) — Haiku API call per episode
3. Heuristic fallback — length + keywords + novelty + emotional salience

When multi-signal is enabled, borderline episodes can optionally be escalated
to the LLM for a second opinion. Durable corrections, explicit preferences,
and stable profile facts bypass the ratio gate and are always extracted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, cast

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.extraction.discourse import classify_discourse
from engram.extraction.prompts import TRIAGE_JUDGE_SYSTEM_CACHED
from engram.models.consolidation import (
    CycleContext,
    DecisionOutcomeLabel,
    DecisionTrace,
    PhaseResult,
    TriageRecord,
)
from engram.models.episode import EpisodeProjectionState
from engram.retrieval.goals import compute_goal_triage_boost, identify_active_goals
from engram.retrieval.triage_policy import TriageDecision, apply_episode_utility_policy

logger = logging.getLogger(__name__)


@dataclass
class _ScoredEpisode:
    """Episode plus its utility decision and optional scorer signals."""

    episode: Any
    decision: TriageDecision
    discourse_class: str
    signals: Any | None = None


def _llm_judge_score(content: str, model: str) -> dict:
    """Call LLM to judge episode content. Returns parsed JSON dict."""
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=cast(Any, TRIAGE_JUDGE_SYSTEM_CACHED),
            messages=cast(Any, [{"role": "user", "content": content}]),
        )
        text = _extract_message_text(response.content).strip()
        if text.startswith("```"):
            first_nl = text.index("\n")
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        parsed = json.loads(text)
        return {
            "extract": bool(parsed.get("extract", True)),
            "score": float(parsed.get("score", 0.5)),
            "reason": str(parsed.get("reason", "")),
            "tags": list(parsed.get("tags", [])),
        }
    except Exception as exc:
        logger.warning("LLM triage judge failed: %s", exc)
        return {"extract": True, "score": 0.5, "reason": "llm_error_fallback", "tags": []}


def _extract_message_text(blocks: object) -> str:
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


class TriagePhase(ConsolidationPhase):
    """Phase 0: score QUEUED episodes and promote top-scoring ones for extraction."""

    def __init__(self, graph_manager: Any | None = None) -> None:
        self._manager = graph_manager
        self._scorer: Any | None = None

    def _get_scorer(self, cfg: ActivationConfig):
        """Reuse the shared scorer so calibration state survives across callers."""
        if self._scorer is None:
            from engram.retrieval.triage_scorer import get_shared_triage_scorer

            self._scorer = get_shared_triage_scorer(cfg)
        return self._scorer

    @property
    def name(self) -> str:
        return "triage"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        methods = {"get_episodes_paginated", "update_episode"}
        if cfg.triage_multi_signal_enabled and self._manager is not None:
            methods.add("get_episode_entities")
        return methods

    async def execute(
        self,
        group_id: str,
        graph_store: Any,
        activation_store: Any,
        search_index: Any,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool = False,
        context: CycleContext | None = None,
    ) -> tuple[PhaseResult, list[TriageRecord]]:
        t0 = time.perf_counter()

        if not cfg.triage_enabled:
            return PhaseResult(
                phase=self.name,
                status="skipped",
                duration_ms=_elapsed_ms(t0),
            ), []

        episodes, _cursor = await graph_store.get_episodes_paginated(
            group_id=group_id,
            status="queued",
            limit=100,
        )
        if not episodes:
            return PhaseResult(
                phase=self.name,
                items_processed=0,
                items_affected=0,
                duration_ms=_elapsed_ms(t0),
            ), []

        records: list[TriageRecord] = []
        promoted = 0

        world_episodes = []
        for ep in episodes:
            ep_content = getattr(ep, "content", "") or ""
            if classify_discourse(ep_content) == "system":
                if not dry_run:
                    await graph_store.update_episode(
                        ep.id,
                        {
                            "status": "completed",
                            "skipped_meta": True,
                            "projection_state": EpisodeProjectionState.CUE_ONLY.value,
                            "last_projection_reason": "triage_skip_meta",
                        },
                        group_id=group_id,
                    )
                records.append(
                    TriageRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        episode_id=ep.id,
                        score=0.0,
                        decision="skip_meta",
                        score_breakdown={"discourse_class": "system"},
                    )
                )
            else:
                world_episodes.append(ep)

        goals = await identify_active_goals(
            graph_store,
            activation_store,
            group_id,
            cfg,
        )

        llm_metadata: dict[str, dict] = {}
        scored_episodes: list[_ScoredEpisode] = []

        if cfg.triage_multi_signal_enabled:
            scorer = self._get_scorer(cfg)
            for ep in world_episodes:
                ep_content = getattr(ep, "content", "") or ""
                discourse_class = classify_discourse(ep_content)
                signals = await scorer.score(
                    content=ep_content,
                    search_index=search_index,
                    graph_store=graph_store,
                    activation_store=activation_store,
                    group_id=group_id,
                    goals=goals,
                )
                decision = apply_episode_utility_policy(
                    ep_content,
                    cfg,
                    signals.composite,
                    discourse_class=discourse_class,
                    mode="phase",
                    score_source="multi_signal",
                )
                _merge_breakdown(decision.score_breakdown, signals)
                scored_episodes.append(
                    _ScoredEpisode(
                        episode=ep,
                        decision=decision,
                        discourse_class=discourse_class,
                        signals=signals,
                    )
                )

            if cfg.triage_llm_escalation_enabled and os.environ.get("ANTHROPIC_API_KEY"):
                escalated = 0
                for scored in scored_episodes:
                    if escalated >= cfg.triage_llm_escalation_max_per_cycle:
                        break
                    if scored.decision.guard_reasons:
                        continue
                    if not (
                        cfg.triage_llm_escalation_low
                        <= scored.decision.score
                        <= cfg.triage_llm_escalation_high
                    ):
                        continue
                    ep_content = getattr(scored.episode, "content", "") or ""
                    judge_result = await asyncio.to_thread(
                        _llm_judge_score,
                        ep_content,
                        cfg.triage_llm_judge_model,
                    )
                    llm_metadata[scored.episode.id] = judge_result
                    decision = apply_episode_utility_policy(
                        ep_content,
                        cfg,
                        judge_result["score"],
                        discourse_class=scored.discourse_class,
                        mode="phase",
                        score_source="llm_escalation",
                    )
                    _merge_breakdown(decision.score_breakdown, scored.signals)
                    decision.score_breakdown["llm_escalated"] = 1.0
                    scored.decision = decision
                    escalated += 1

        elif cfg.triage_llm_judge_enabled:
            for ep in world_episodes:
                ep_content = getattr(ep, "content", "") or ""
                discourse_class = classify_discourse(ep_content)
                judge_result = _llm_judge_score(ep_content, cfg.triage_llm_judge_model)
                llm_metadata[ep.id] = judge_result
                score = min(
                    1.0,
                    judge_result["score"] + compute_goal_triage_boost(ep_content, goals, cfg),
                )
                decision = apply_episode_utility_policy(
                    ep_content,
                    cfg,
                    score,
                    discourse_class=discourse_class,
                    mode="phase",
                    score_source="llm",
                )
                scored_episodes.append(
                    _ScoredEpisode(
                        episode=ep,
                        decision=decision,
                        discourse_class=discourse_class,
                    )
                )
        else:
            for ep in world_episodes:
                ep_content = getattr(ep, "content", "") or ""
                discourse_class = classify_discourse(ep_content)
                score = await self._score_episode_async(ep, cfg, search_index, group_id, goals)
                decision = apply_episode_utility_policy(
                    ep_content,
                    cfg,
                    score,
                    discourse_class=discourse_class,
                    mode="phase",
                    score_source="heuristic",
                )
                scored_episodes.append(
                    _ScoredEpisode(
                        episode=ep,
                        decision=decision,
                        discourse_class=discourse_class,
                    )
                )

        scored_episodes.sort(key=lambda item: item.decision.score, reverse=True)
        if not scored_episodes:
            return PhaseResult(
                phase=self.name,
                items_processed=len(episodes),
                items_affected=0,
                duration_ms=_elapsed_ms(t0),
            ), records

        guarded_ids = {item.episode.id for item in scored_episodes if item.decision.guard_reasons}
        eligible = [
            item
            for item in scored_episodes
            if item.decision.action == "extract" and item.episode.id not in guarded_ids
        ]
        extract_budget = 0
        if cfg.triage_extract_ratio > 0 and eligible:
            extract_budget = max(1, int(len(scored_episodes) * cfg.triage_extract_ratio))
            extract_budget = min(extract_budget, len(eligible))

        selected_ids = set(guarded_ids)
        selected_ids.update(item.episode.id for item in eligible[:extract_budget])

        for item in scored_episodes:
            ep = item.episode
            judge_meta = llm_metadata.get(ep.id, {})
            decision_label = "extract" if ep.id in selected_ids else "skip"
            decision_source = item.decision.decision_source
            threshold_band = item.decision.threshold_band
            trace = None
            if decision_label == "skip" and item.decision.action == "extract":
                decision_source = "capacity_policy"
                threshold_band = "capacity_skip"

            score_breakdown = dict(item.decision.score_breakdown)
            if judge_meta:
                score_breakdown["llm_reason"] = judge_meta.get("reason")
                score_breakdown["llm_tags"] = judge_meta.get("tags", [])

            records.append(
                TriageRecord(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    episode_id=ep.id,
                    score=item.decision.score,
                    decision=decision_label,
                    score_breakdown=score_breakdown,
                    llm_reason=judge_meta.get("reason"),
                    llm_tags=judge_meta.get("tags", []),
                )
            )

            if context is not None:
                trace = DecisionTrace(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    phase=self.name,
                    candidate_type="episode",
                    candidate_id=ep.id,
                    decision=decision_label,
                    decision_source=decision_source,
                    confidence=item.decision.score,
                    threshold_band=threshold_band,
                    features=score_breakdown,
                    constraints_hit=list(item.decision.guard_reasons),
                    policy_version="utility_v1",
                )
                context.add_decision_trace(trace)
                context.add_decision_outcome_label(
                    DecisionOutcomeLabel(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        phase=self.name,
                        decision_trace_id=trace.id,
                        outcome_type="routing",
                        label=decision_label,
                        value=item.decision.score,
                        metadata={"threshold_band": threshold_band},
                    )
                )

            if dry_run:
                continue

            if decision_label == "extract" and self._manager:
                try:
                    await self._manager.project_episode(ep.id, group_id)
                    promoted += 1
                    extracted_entities = await self._record_projection_outcome(graph_store, item)
                    if context is not None and trace is not None:
                        context.add_decision_outcome_label(
                            DecisionOutcomeLabel(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                phase=self.name,
                                decision_trace_id=trace.id,
                                outcome_type="projection_yield",
                                label="useful" if extracted_entities > 0 else "empty",
                                value=float(extracted_entities),
                                metadata={"episode_id": ep.id},
                            )
                        )
                    if context is not None:
                        context.triage_promoted_ids.add(ep.id)
                except Exception:
                    logger.warning("Triage: extraction failed for %s", ep.id, exc_info=True)
                    if context is not None and trace is not None:
                        context.add_decision_outcome_label(
                            DecisionOutcomeLabel(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                phase=self.name,
                                decision_trace_id=trace.id,
                                outcome_type="projection_yield",
                                label="failed",
                                value=0.0,
                                metadata={"episode_id": ep.id},
                            )
                        )
            elif decision_label == "skip":
                await graph_store.update_episode(
                    ep.id,
                    {
                        "status": "completed",
                        "skipped_triage": True,
                        "projection_state": EpisodeProjectionState.CUE_ONLY.value,
                        "last_projection_reason": "triage_ratio_skip",
                    },
                    group_id=group_id,
                )

        return PhaseResult(
            phase=self.name,
            items_processed=len(scored_episodes),
            items_affected=promoted,
            duration_ms=_elapsed_ms(t0),
        ), records

    async def _record_projection_outcome(
        self,
        graph_store: Any,
        scored: _ScoredEpisode,
    ) -> int:
        """Feed observed extraction yield back into the shared scorer."""
        if scored.signals is None:
            return 0
        get_episode_entities = getattr(graph_store, "get_episode_entities", None)
        if get_episode_entities is None:
            return 0
        try:
            entity_ids = await get_episode_entities(scored.episode.id)
        except Exception:
            return 0
        count = len(entity_ids) if isinstance(entity_ids, list) else 0
        if isinstance(entity_ids, list):
            scorer_cfg = getattr(self._manager, "_cfg", None) if self._manager else None
            if not isinstance(scorer_cfg, ActivationConfig):
                scorer_cfg = ActivationConfig()
            self._get_scorer(scorer_cfg).record_outcome(
                scored.signals,
                count,
            )
        return count

    @staticmethod
    async def _score_episode_async(
        episode: Any,
        cfg: ActivationConfig,
        search_index: Any = None,
        group_id: str = "default",
        goals: list | None = None,
    ) -> float:
        """Score episode using lightweight heuristic features."""
        content = getattr(episode, "content", "") or ""
        if not content:
            return 0.0

        length_score = min(len(content) / 500, 1.0) * 0.25

        caps = len(re.findall(r"\b[A-Z][a-z]+\b", content))
        numbers = len(re.findall(r"\b\d+\b", content))
        quoted = len(re.findall(r'"[^"]+"|\'[^\']+\'', content))
        keyword_count = caps + numbers + quoted
        keyword_score = min(keyword_count / 10, 1.0) * 0.20

        novelty_score = 0.15
        if search_index and hasattr(search_index, "search_episodes"):
            try:
                query = content[:200].strip()
                if query:
                    matches = await search_index.search_episodes(
                        query,
                        group_id=group_id,
                        limit=3,
                    )
                    if matches:
                        top_score = matches[0][1]
                        similarity = min(top_score / 10.0, 1.0)
                        novelty_score = (1.0 - similarity) * 0.30
                    else:
                        novelty_score = 0.30
            except Exception:
                novelty_score = 0.15

        emotional_score = 0.0
        if cfg.emotional_salience_enabled:
            from engram.extraction.salience import compute_emotional_salience

            salience = compute_emotional_salience(content)
            emotional_score = salience.composite * cfg.emotional_triage_weight
            base_score = length_score + keyword_score + novelty_score + emotional_score
            if salience.composite >= cfg.triage_personal_floor_threshold:
                return max(base_score, cfg.triage_personal_floor)

        base_score = length_score + keyword_score + novelty_score + emotional_score
        if goals:
            base_score += compute_goal_triage_boost(content, goals, cfg)
        return base_score

    @staticmethod
    def _score_episode(episode: Any, cfg: ActivationConfig) -> float:
        """Synchronous scoring fallback (no novelty lookup)."""
        content = getattr(episode, "content", "") or ""
        if not content:
            return 0.0
        length_score = min(len(content) / 500, 1.0) * 0.25
        caps = len(re.findall(r"\b[A-Z][a-z]+\b", content))
        numbers = len(re.findall(r"\b\d+\b", content))
        quoted = len(re.findall(r'"[^"]+"|\'[^\']+\'', content))
        keyword_count = caps + numbers + quoted
        keyword_score = min(keyword_count / 10, 1.0) * 0.20
        novelty_score = 0.15

        emotional_score = 0.0
        if cfg.emotional_salience_enabled:
            from engram.extraction.salience import compute_emotional_salience

            salience = compute_emotional_salience(content)
            emotional_score = salience.composite * cfg.emotional_triage_weight
            base_score = length_score + keyword_score + novelty_score + emotional_score
            if salience.composite >= cfg.triage_personal_floor_threshold:
                return max(base_score, cfg.triage_personal_floor)

        return length_score + keyword_score + novelty_score + emotional_score


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _merge_breakdown(target: dict[str, Any], signals: Any | None) -> None:
    """Flatten signal dataclasses into score-breakdown dictionaries."""
    if signals is None:
        return
    for key, value in vars(signals).items():
        target[key] = value
