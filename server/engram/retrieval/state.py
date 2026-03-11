"""State-dependent retrieval: cognitive mode + arousal matching.

Biological basis: Encoding specificity (Tulving, 1973) -- retrieval is most
effective when internal state matches state at encoding.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from engram.config import ActivationConfig

_TASK_PATTERNS = re.compile(
    r"\b(?:how do I|how to|fix|debug|implement|build|create|setup|install|"
    r"configure|deploy|write|code|function|error|bug|issue)\b",
    re.IGNORECASE,
)

_EXPLORATORY_PATTERNS = re.compile(
    r"\b(?:what is|who is|tell me about|explain|describe|what are|"
    r"difference between|compare|overview|summary|history of)\b",
    re.IGNORECASE,
)

_REFLECTIVE_PATTERNS = re.compile(
    r"\b(?:I feel|I think|I wonder|I remember|I miss|reminds me|"
    r"looking back|used to|when I was|my experience|personally)\b",
    re.IGNORECASE,
)


@dataclass
class CognitiveState:
    """Current cognitive state inferred from query and session."""

    arousal_level: float = 0.0
    mode: str = "neutral"  # task | exploratory | reflective | neutral
    domain_weights: dict[str, float] = field(default_factory=dict)
    time_bucket: str = "afternoon"


def infer_cognitive_mode(query: str) -> str:
    """Infer cognitive mode from query patterns."""
    if not query:
        return "neutral"
    task_hits = len(_TASK_PATTERNS.findall(query))
    explore_hits = len(_EXPLORATORY_PATTERNS.findall(query))
    reflect_hits = len(_REFLECTIVE_PATTERNS.findall(query))

    scores = {
        "task": task_hits,
        "exploratory": explore_hits,
        "reflective": reflect_hits,
    }
    best = max(scores, key=lambda mode: scores[mode])
    if scores[best] == 0:
        return "neutral"
    return best


def get_time_bucket(hour: int | None = None) -> str:
    """Map current hour to time bucket."""
    if hour is None:
        hour = time.localtime().tm_hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


# Domain to cognitive mode affinity
_MODE_DOMAIN_AFFINITY: dict[str, dict[str, float]] = {
    "task": {"technical": 0.8, "knowledge": 0.5, "personal": 0.2},
    "exploratory": {
        "knowledge": 0.8, "creative": 0.6, "technical": 0.4,
    },
    "reflective": {"personal": 0.9, "health": 0.6, "creative": 0.5},
    "neutral": {},
}


def compute_state_bias(
    state: CognitiveState,
    entity_attrs: dict,
    entity_type: str,
    cfg: ActivationConfig,
    domain_groups: dict[str, list[str]] | None = None,
) -> float:
    """Compute retrieval bias based on cognitive state match.

    Returns a small boost [0, state_domain_weight + state_arousal_match_weight].
    """
    if not cfg.state_dependent_retrieval_enabled:
        return 0.0

    bias = 0.0

    # Domain affinity: does entity's domain match current cognitive mode?
    if cfg.state_domain_weight > 0 and domain_groups:
        entity_domain = None
        for domain, types in domain_groups.items():
            if entity_type in types:
                entity_domain = domain
                break

        if entity_domain:
            mode_affinities = _MODE_DOMAIN_AFFINITY.get(state.mode, {})
            affinity = mode_affinities.get(entity_domain, 0.3)
            bias += cfg.state_domain_weight * affinity

    # Arousal matching: entities encoded at similar arousal get a boost
    if cfg.state_arousal_match_weight > 0:
        emo_composite = entity_attrs.get("emo_composite", 0.0)
        if isinstance(emo_composite, (int, float)) and emo_composite > 0:
            arousal_diff = abs(state.arousal_level - emo_composite)
            match_score = max(0.0, 1.0 - arousal_diff * 2.0)
            bias += cfg.state_arousal_match_weight * match_score

    return bias


# --- Additional helpers used by scorer and tests ---

# Aliases for backward compatibility
infer_mode = infer_cognitive_mode


def infer_time_bucket(now: object = None) -> str:
    """Wrapper: accept datetime or None, delegate to get_time_bucket."""
    if now is not None and hasattr(now, "hour"):
        return get_time_bucket(now.hour)
    return get_time_bucket()


def compute_domain_weights(
    entity_types: list[str],
    domain_groups: dict[str, list[str]],
) -> dict[str, float]:
    """Compute domain weight distribution from recent entity types."""
    if not entity_types:
        return {}
    type_to_domain: dict[str, str] = {}
    for domain, types in domain_groups.items():
        for t in types:
            type_to_domain[t] = domain
    counts: dict[str, int] = {}
    for et in entity_types:
        domain = type_to_domain.get(et, "knowledge")
        counts[domain] = counts.get(domain, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {d: c / total for d, c in counts.items()}


def update_arousal_ema(
    current_arousal: float,
    new_composite: float,
    alpha: float = 0.3,
) -> float:
    """EMA update for arousal level."""
    return alpha * new_composite + (1.0 - alpha) * current_arousal


def entity_type_to_domain(
    entity_type: str,
    domain_groups: dict[str, list[str]],
) -> str:
    """Map entity type to domain name."""
    for domain, types in domain_groups.items():
        if entity_type in types:
            return domain
    return "knowledge"


def compute_state_boost(
    entity_domain: str | None,
    entity_arousal: float,
    state: CognitiveState,
    domain_weight: float,
    arousal_match_weight: float,
) -> float:
    """Compute state-dependent retrieval boost for a single entity."""
    domain_affinity = 0.0
    if entity_domain and state.domain_weights:
        domain_affinity = state.domain_weights.get(entity_domain, 0.0)
    arousal_match = 1.0 - abs(state.arousal_level - entity_arousal)
    return domain_weight * domain_affinity + arousal_match_weight * arousal_match


def infer_cognitive_state(
    query: str,
    recent_entity_types: list[str],
    domain_groups: dict[str, list[str]],
    current_arousal: float = 0.3,
    session_start: float | None = None,
) -> CognitiveState:
    """Build a full CognitiveState from available signals."""
    _session_mins = 0.0
    if session_start is not None:
        _session_mins = (time.time() - session_start) / 60.0
    return CognitiveState(
        arousal_level=current_arousal,
        mode=infer_cognitive_mode(query),
        domain_weights=compute_domain_weights(recent_entity_types, domain_groups),
        time_bucket=get_time_bucket(),
    )
