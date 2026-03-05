"""Triage phase: score QUEUED episodes and selectively extract the top fraction.

Supports three scoring modes (checked in priority order):
1. Multi-signal scorer (triage_multi_signal_enabled) — 8 deterministic signals, ~2ms/ep
2. LLM judge (triage_llm_judge_enabled) — Haiku API call per episode
3. Heuristic fallback — length + keywords + novelty + emotional salience

When multi-signal is enabled, borderline episodes (score in escalation band)
can optionally be escalated to LLM for a second opinion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.extraction.discourse import classify_discourse
from engram.extraction.prompts import TRIAGE_JUDGE_SYSTEM_CACHED
from engram.models.consolidation import CycleContext, PhaseResult, TriageRecord
from engram.retrieval.goals import compute_goal_triage_boost, identify_active_goals

logger = logging.getLogger(__name__)

_PERSONAL_PATTERNS = re.compile(
    r"\b(?:mom|dad|mother|father|brother|sister|wife|husband|partner|"
    r"family|daughter|son|child|children|friend|"
    r"birthday|wedding|funeral|anniversary|holiday|vacation|"
    r"diagnosed|hospital|surgery|illness|health|cancer|"
    r"love|miss|afraid|excited|proud|grateful|worried|happy|sad|"
    r"home|moved|married|divorced|born|died|retired|graduated)\b",
    re.IGNORECASE,
)


def personal_narrative_boost(content: str, cfg: ActivationConfig) -> float:
    """Return boost if personal narrative keywords found above threshold."""
    if not cfg.triage_personal_boost_enabled:
        return 0.0
    matches = len(_PERSONAL_PATTERNS.findall(content))
    if matches >= cfg.triage_personal_min_matches:
        return cfg.triage_personal_boost
    return 0.0


def _llm_judge_score(content: str, model: str) -> dict:
    """Call LLM to judge episode content. Returns parsed JSON dict.

    Synchronous — intended to be called from async context via asyncio.to_thread
    or directly from consolidation phase.
    """
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=TRIAGE_JUDGE_SYSTEM_CACHED,
            messages=[{"role": "user", "content": content}],
        )
        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            first_nl = text.index("\n")
            text = text[first_nl + 1:]
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


class TriagePhase(ConsolidationPhase):
    """Phase 0: score QUEUED episodes and promote top-scoring ones for extraction."""

    def __init__(self, graph_manager: Any | None = None) -> None:
        self._manager = graph_manager
        self._scorer = None  # Lazy-initialized TriageScorer

    def _get_scorer(self, cfg: ActivationConfig):
        """Lazy-init multi-signal scorer (preserves EMA state across cycles)."""
        if self._scorer is None:
            from engram.retrieval.triage_scorer import TriageScorer
            self._scorer = TriageScorer(cfg)
        return self._scorer

    @property
    def name(self) -> str:
        return "triage"

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

        # 1. Query QUEUED episodes
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
        skipped = 0

        # 2. Filter out system-discourse episodes
        world_episodes = []
        for ep in episodes:
            ep_content = getattr(ep, "content", "") or ""
            if classify_discourse(ep_content) == "system":
                if not dry_run:
                    await graph_store.update_episode(
                        ep.id,
                        {"status": "completed", "skipped_meta": True},
                        group_id=group_id,
                    )
                records.append(
                    TriageRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        episode_id=ep.id,
                        score=0.0,
                        decision="skip_meta",
                    )
                )
                skipped += 1
                logger.debug("Triage: skipped meta-discourse episode %s", ep.id)
            else:
                world_episodes.append(ep)

        # 2b. Identify active goals for triage boost
        goals = await identify_active_goals(
            graph_store, activation_store, group_id, cfg,
        )

        # 3. Score each remaining episode
        llm_metadata: dict[str, dict] = {}  # episode_id → judge result

        if cfg.triage_multi_signal_enabled:
            # --- Multi-signal scoring (primary path) ---
            scorer = self._get_scorer(cfg)
            scored = []
            for ep in world_episodes:
                ep_content = getattr(ep, "content", "") or ""
                signals = await scorer.score(
                    content=ep_content,
                    search_index=search_index,
                    graph_store=graph_store,
                    activation_store=activation_store,
                    group_id=group_id,
                    goals=goals,
                )
                scored.append((ep, signals.composite, signals))

            # LLM escalation for borderline episodes
            if cfg.triage_llm_escalation_enabled and os.environ.get("ANTHROPIC_API_KEY"):
                escalated = 0
                for i, (ep, score, signals) in enumerate(scored):
                    if escalated >= cfg.triage_llm_escalation_max_per_cycle:
                        break
                    if cfg.triage_llm_escalation_low <= score <= cfg.triage_llm_escalation_high:
                        ep_content = getattr(ep, "content", "") or ""
                        judge_result = await asyncio.to_thread(
                            _llm_judge_score, ep_content, cfg.triage_llm_judge_model,
                        )
                        llm_metadata[ep.id] = judge_result
                        # Override composite with LLM score
                        scored[i] = (ep, judge_result["score"], signals)
                        escalated += 1
                        logger.debug(
                            "Triage: LLM escalation for %s (multi-signal=%.3f, llm=%.3f)",
                            ep.id, score, judge_result["score"],
                        )

            # Convert to (ep, score) for sorting
            scored_pairs = [(ep, score) for ep, score, _signals in scored]

        elif cfg.triage_llm_judge_enabled:
            # --- Pure LLM judge ---
            scored_pairs = []
            for ep in world_episodes:
                ep_content = getattr(ep, "content", "") or ""
                judge_result = _llm_judge_score(ep_content, cfg.triage_llm_judge_model)
                llm_metadata[ep.id] = judge_result
                score = judge_result["score"]
                score += compute_goal_triage_boost(ep_content, goals, cfg)
                scored_pairs.append((ep, score))
        else:
            # --- Heuristic fallback ---
            scored_pairs = []
            for ep in world_episodes:
                score = await self._score_episode_async(ep, cfg, search_index, group_id, goals)
                scored_pairs.append((ep, score))

        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        if not scored_pairs:
            return PhaseResult(
                phase=self.name,
                items_processed=len(episodes),
                items_affected=0,
                duration_ms=_elapsed_ms(t0),
            ), records

        # 4. Determine cutoff
        extract_count = max(1, int(len(scored_pairs) * cfg.triage_extract_ratio))

        for i, (ep, score) in enumerate(scored_pairs):
            decision = "extract" if i < extract_count else "skip"
            judge_meta = llm_metadata.get(ep.id, {})
            records.append(
                TriageRecord(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    episode_id=ep.id,
                    score=round(score, 4),
                    decision=decision,
                    llm_reason=judge_meta.get("reason"),
                    llm_tags=judge_meta.get("tags", []),
                )
            )

            if not dry_run:
                if decision == "extract" and self._manager:
                    try:
                        await self._manager.project_episode(ep.id, group_id)
                        promoted += 1
                        if context:
                            context.triage_promoted_ids.add(ep.id)
                    except Exception:
                        logger.warning(
                            "Triage: extraction failed for %s",
                            ep.id,
                            exc_info=True,
                        )
                elif decision == "skip":
                    await graph_store.update_episode(
                        ep.id,
                        {"status": "completed", "skipped_triage": True},
                        group_id=group_id,
                    )
                    skipped += 1

        return PhaseResult(
            phase=self.name,
            items_processed=len(scored_pairs),
            items_affected=promoted,
            duration_ms=_elapsed_ms(t0),
        ), records

    @staticmethod
    async def _score_episode_async(
        episode: Any,
        cfg: ActivationConfig,
        search_index: Any = None,
        group_id: str = "default",
        goals: list | None = None,
    ) -> float:
        """Score episode using length, keyword density, novelty, emotional salience, and goal boost.

        Novelty is computed by searching for similar existing episodes. If the
        top match has a high FTS/vector score, the content is redundant (low
        novelty). If no similar episodes exist, novelty is high.
        """
        content = getattr(episode, "content", "") or ""
        if not content:
            return 0.0

        # Length signal (0-0.25)
        length_score = min(len(content) / 500, 1.0) * 0.25

        # Keyword density (0-0.20)
        caps = len(re.findall(r"\b[A-Z][a-z]+\b", content))
        numbers = len(re.findall(r"\b\d+\b", content))
        quoted = len(re.findall(r'"[^"]+"|\'[^\']+\'', content))
        keyword_count = caps + numbers + quoted
        keyword_score = min(keyword_count / 10, 1.0) * 0.20

        # Novelty signal (0-0.30): search for similar episodes
        novelty_score = 0.15  # Default fallback
        if search_index and hasattr(search_index, "search_episodes"):
            try:
                # Use first 200 chars as query (enough for topic matching)
                query = content[:200].strip()
                if query:
                    matches = await search_index.search_episodes(
                        query, group_id=group_id, limit=3,
                    )
                    if matches:
                        # Top match score indicates redundancy
                        # FTS5 scores vary but higher = more similar
                        top_score = matches[0][1]
                        # Normalize: high FTS score -> low novelty
                        # FTS5 scores are typically 0-20+, cap at 10 for normalization
                        similarity = min(top_score / 10.0, 1.0)
                        novelty_score = (1.0 - similarity) * 0.30
                    else:
                        # No similar episodes -- very novel
                        novelty_score = 0.30
            except Exception:
                novelty_score = 0.15  # Fallback on error

        # Emotional salience signal
        emotional_score = 0.0
        if cfg.emotional_salience_enabled:
            from engram.extraction.salience import compute_emotional_salience

            salience = compute_emotional_salience(content)
            emotional_score = salience.composite * cfg.emotional_triage_weight
            # Personal floor: guarantee extraction for personal content
            base_score = length_score + keyword_score + novelty_score + emotional_score
            if salience.composite >= cfg.triage_personal_floor_threshold:
                return max(base_score, cfg.triage_personal_floor)

        base_score = length_score + keyword_score + novelty_score + emotional_score

        # Goal-relevance boost
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

        # Emotional salience signal
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
