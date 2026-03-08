"""Shared triage utility policy for consolidation and the episode worker."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from engram.config import ActivationConfig

_PERSONAL_PATTERNS = re.compile(
    r"\b(?:mom|dad|mother|father|brother|sister|wife|husband|partner|"
    r"family|daughter|son|child|children|friend|"
    r"birthday|wedding|funeral|anniversary|holiday|vacation|"
    r"diagnosed|hospital|surgery|illness|health|cancer|"
    r"love|miss|afraid|excited|proud|grateful|worried|happy|sad|"
    r"home|moved|married|divorced|born|died|retired|graduated)\b",
    re.IGNORECASE,
)
_CORRECTION_PATTERNS = re.compile(
    r"\b(?:actually|correction|to clarify|i meant|i mean|not anymore|"
    r"that's wrong|that was wrong|update:?)\b",
    re.IGNORECASE,
)
_PREFERENCE_PATTERNS = re.compile(
    r"\b(?:i (?:like|love|prefer|dislike|hate|avoid)|my favorite|"
    r"please (?:don't|do not)|i'm into|i am into)\b",
    re.IGNORECASE,
)
_PROFILE_PATTERNS = re.compile(
    r"\b(?:my name is|i am|i'm|i live in|i work at|i work for|"
    r"i was born|my birthday is|i'm from|i am from|my role is)\b",
    re.IGNORECASE,
)
_TASK_PATTERNS = re.compile(
    r"\b(?:i need to|i should|i will|i'll|remind me|todo|to do|"
    r"deadline|due on|plan to|planning to)\b",
    re.IGNORECASE,
)
_TEMPORAL_PATTERNS = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}"
    r"(?:,?\s+\d{4})?|yesterday|today|tomorrow|last\s+\w+|next\s+\w+)\b",
    re.IGNORECASE,
)

_CORRECTION_WEIGHT = 0.22
_PREFERENCE_WEIGHT = 0.18
_PROFILE_WEIGHT = 0.18
_TASK_WEIGHT = 0.10
_TEMPORAL_WEIGHT = 0.05
_HYBRID_PENALTY = 0.08


@dataclass
class TriageDecision:
    """Final utility-oriented routing decision for an episode."""

    action: str
    score: float
    base_score: float
    threshold_band: str
    decision_source: str
    guard_reasons: list[str] = field(default_factory=list)
    score_breakdown: dict[str, Any] = field(default_factory=dict)


def personal_narrative_boost(content: str, cfg: ActivationConfig) -> float:
    """Return boost if personal narrative keywords found above threshold."""
    if not cfg.triage_personal_boost_enabled:
        return 0.0
    matches = len(_PERSONAL_PATTERNS.findall(content))
    if matches >= cfg.triage_personal_min_matches:
        return cfg.triage_personal_boost
    return 0.0


def apply_episode_utility_policy(
    content: str,
    cfg: ActivationConfig,
    base_score: float,
    *,
    discourse_class: str = "world",
    mode: str = "phase",
    score_source: str = "rule",
) -> TriageDecision:
    """Convert a raw episode score into a routing decision.

    ``mode="phase"`` yields ``extract`` or ``skip``.
    ``mode="worker"`` yields ``extract``, ``skip``, or ``defer``.
    """
    correction = 1.0 if _CORRECTION_PATTERNS.search(content) else 0.0
    preference = 1.0 if _PREFERENCE_PATTERNS.search(content) else 0.0
    profile = 1.0 if _PROFILE_PATTERNS.search(content) else 0.0
    task = 1.0 if _TASK_PATTERNS.search(content) else 0.0
    temporal = 1.0 if _TEMPORAL_PATTERNS.search(content) else 0.0
    hybrid_penalty = 1.0 if discourse_class == "hybrid" else 0.0
    personal_boost = personal_narrative_boost(content, cfg)

    score = min(
        1.0,
        max(
            0.0,
            base_score
            + personal_boost
            + correction * _CORRECTION_WEIGHT
            + preference * _PREFERENCE_WEIGHT
            + profile * _PROFILE_WEIGHT
            + task * _TASK_WEIGHT
            + temporal * _TEMPORAL_WEIGHT
            - hybrid_penalty * _HYBRID_PENALTY,
        ),
    )

    guard_reasons: list[str] = []
    if correction:
        guard_reasons.append("correction")
    if preference:
        guard_reasons.append("explicit_preference")
    if profile:
        guard_reasons.append("durable_profile")

    breakdown = {
        "base_score": round(base_score, 4),
        "utility_score": round(score, 4),
        "personal_narrative_boost": round(personal_boost, 4),
        "correction_signal": correction,
        "preference_signal": preference,
        "profile_signal": profile,
        "task_signal": task,
        "temporal_specificity": temporal,
        "hybrid_discourse_penalty": hybrid_penalty,
        "discourse_class": discourse_class,
        "guard_reasons": list(guard_reasons),
    }

    if guard_reasons:
        return TriageDecision(
            action="extract",
            score=round(score, 4),
            base_score=round(base_score, 4),
            threshold_band="durable_guard",
            decision_source="durable_guard",
            guard_reasons=guard_reasons,
            score_breakdown=breakdown,
        )

    if mode == "worker":
        if score >= cfg.worker_extract_threshold:
            action = "extract"
            band = "high_confidence"
        elif score < cfg.worker_skip_threshold:
            action = "skip"
            band = "low_confidence"
        else:
            action = "defer"
            band = "uncertain_band"
    else:
        if score >= cfg.triage_min_score:
            action = "extract"
            band = "eligible"
        else:
            action = "skip"
            band = "below_threshold"

    return TriageDecision(
        action=action,
        score=round(score, 4),
        base_score=round(base_score, 4),
        threshold_band=band,
        decision_source=score_source,
        guard_reasons=guard_reasons,
        score_breakdown=breakdown,
    )
