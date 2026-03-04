"""Triage phase: score QUEUED episodes and selectively extract the top fraction."""

from __future__ import annotations

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

        # 3. Score each remaining episode (LLM judge or heuristic)
        llm_metadata: dict[str, dict] = {}  # episode_id → judge result
        if cfg.triage_llm_judge_enabled:
            scored = []
            for ep in world_episodes:
                ep_content = getattr(ep, "content", "") or ""
                judge_result = _llm_judge_score(ep_content, cfg.triage_llm_judge_model)
                llm_metadata[ep.id] = judge_result
                scored.append((ep, judge_result["score"]))
        else:
            scored = [(ep, self._score_episode(ep, cfg)) for ep in world_episodes]
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return PhaseResult(
                phase=self.name,
                items_processed=len(episodes),
                items_affected=0,
                duration_ms=_elapsed_ms(t0),
            ), records

        # 4. Determine cutoff
        extract_count = max(1, int(len(scored) * cfg.triage_extract_ratio))

        for i, (ep, score) in enumerate(scored):
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
            items_processed=len(scored),
            items_affected=promoted,
            duration_ms=_elapsed_ms(t0),
        ), records

    @staticmethod
    def _score_episode(episode: Any, cfg: ActivationConfig) -> float:
        """Lightweight scoring based on length, keyword density, novelty, and personal boost."""
        content = getattr(episode, "content", "") or ""
        if not content:
            return 0.0

        # Length signal (0-0.3): longer content more likely to contain extractable info
        length_score = min(len(content) / 500, 1.0) * 0.3

        # Keyword density (0-0.3): capitalized words, numbers, quoted strings
        caps = len(re.findall(r"\b[A-Z][a-z]+\b", content))
        numbers = len(re.findall(r"\b\d+\b", content))
        quoted = len(re.findall(r'"[^"]+"|\'[^\']+\'', content))
        keyword_count = caps + numbers + quoted
        keyword_score = min(keyword_count / 10, 1.0) * 0.3

        # Base novelty signal (0-0.4): without DB lookup, use moderate default
        # Full novelty would compare against existing episodes via FTS5
        novelty_score = 0.2

        # Personal narrative boost (0-0.15): prevent personal content from being deprioritized
        personal_score = personal_narrative_boost(content, cfg)

        return length_score + keyword_score + novelty_score + personal_score


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
