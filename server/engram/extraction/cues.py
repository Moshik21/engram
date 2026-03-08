"""Deterministic cue extraction for latent episode memory."""

from __future__ import annotations

import re

from engram.config import ActivationConfig
from engram.extraction.discourse import classify_discourse
from engram.extraction.policy import ProjectionPolicy
from engram.models.episode import Episode
from engram.models.episode_cue import EpisodeCue
from engram.utils.dates import utc_now

_PROPER_NAMES = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")
_TECHNICAL_TOKENS = re.compile(
    r"\b(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+|"
    r"[A-Za-z][A-Za-z0-9_-]*\.(?:py|ts|tsx|js|jsx|json|toml|md|yaml|yml)|"
    r"(?:API|SDK|CLI|MCP|LLM|SQL|FTS5|Redis|SQLite|FalkorDB|FastAPI|React|Next\.js|TypeScript))\b"
)
_QUOTED_STRINGS = re.compile(r'"([^"]{3,})"|\'([^\']{3,})\'')
_DATES = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:yesterday|today|tomorrow|last\s+(?:week|month|year)|"
    r"next\s+(?:week|month|year)|since\s+[A-Z][a-z]+))\b",
    re.IGNORECASE,
)
_URLS = re.compile(r"https?://\S+")
_NUMBERS_WITH_CONTEXT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|dollars?|years?|months?|hours?|minutes?|GB|MB|TB|k|K|M)\b"
)
_CONTRADICTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("negation", re.compile(r"\b(?:no longer|doesn't|don't|didn't|not anymore|never)\b", re.I)),
    ("ended", re.compile(r"\b(?:stopped|quit|left|ended|cancelled|canceled)\b", re.I)),
    ("correction", re.compile(r"\b(?:actually|correction|instead|updated|changed)\b", re.I)),
    ("move", re.compile(r"\b(?:moved to|moved from)\b", re.I)),
]
_IDENTITY_PATTERNS = re.compile(
    r"\b(?:my name is|i am|i'm|my wife|my husband|my partner|my mom|my dad|"
    r"i work at|i live in|we live in)\b",
    re.IGNORECASE,
)


def _split_spans(text: str, limit: int = 2) -> list[str]:
    spans = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    return spans[:limit]


def _unique(items: list[str], limit: int = 10) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def _quote_matches(text: str) -> list[str]:
    matches: list[str] = []
    for match in _QUOTED_STRINGS.findall(text):
        for group in match:
            if group:
                matches.append(group.strip())
    return matches


def _entity_mentions(text: str) -> list[dict]:
    mentions: list[dict] = []
    for match in _unique(_PROPER_NAMES.findall(text), limit=8):
        mentions.append({"text": match, "type_hint": "proper_name"})
    for match in _unique(_TECHNICAL_TOKENS.findall(text), limit=8):
        mentions.append({"text": match, "type_hint": "technical"})
    return mentions[:12]


def _cue_score(
    text: str,
    entity_mentions: list[dict],
    temporal_markers: list[str],
    quote_spans: list[str],
    contradiction_keys: list[str],
    salience_score: float,
) -> tuple[float, float, str]:
    struct = min(
        1.0,
        (
            len(entity_mentions) * 0.10
            + len(temporal_markers) * 0.12
            + len(quote_spans) * 0.08
            + len(_URLS.findall(text)) * 0.06
            + len(_NUMBERS_WITH_CONTEXT.findall(text)) * 0.06
        ),
    )
    priority = min(1.0, struct * 0.55 + salience_score * 0.35 + len(contradiction_keys) * 0.10)

    if contradiction_keys:
        return struct, priority, "contradiction_hint"
    if salience_score >= 0.65:
        return struct, priority, "high_salience"
    if _IDENTITY_PATTERNS.search(text):
        return struct, priority, "identity_hint"
    if entity_mentions:
        return struct, priority, "entity_dense"
    return struct, priority, "default"


def build_episode_cue(
    episode: Episode,
    cfg: ActivationConfig,
) -> EpisodeCue | None:
    """Build an `EpisodeCue` for an episode without calling an LLM."""
    content = episode.content or ""
    if not content.strip():
        return None

    discourse_class = classify_discourse(content)
    if discourse_class == "system":
        return None

    salience_score = 0.0
    if cfg.emotional_salience_enabled:
        from engram.extraction.salience import compute_emotional_salience

        salience_score = compute_emotional_salience(content).composite

    mentions = _entity_mentions(content)
    temporal_markers = _unique(_DATES.findall(content), limit=6)
    quote_spans = _unique(_quote_matches(content), limit=4)
    contradiction_keys = [
        name for name, pattern in _CONTRADICTION_PATTERNS if pattern.search(content)
    ]
    first_spans = _split_spans(content, limit=2)

    cue_score, projection_priority, route_reason = _cue_score(
        content,
        mentions,
        temporal_markers,
        quote_spans,
        contradiction_keys,
        salience_score,
    )
    policy = ProjectionPolicy(cfg)
    decision = policy.decide_initial(
        episode=episode,
        discourse_class=discourse_class,
        route_reason=route_reason,
        projection_priority=projection_priority,
    )

    cue_parts = []
    if mentions:
        cue_parts.append("mentions: " + ", ".join(m["text"] for m in mentions[:6]))
    if temporal_markers:
        cue_parts.append("time: " + ", ".join(temporal_markers[:4]))
    if quote_spans:
        cue_parts.append("quotes: " + "; ".join(quote_spans[:2]))
    if first_spans:
        cue_parts.append("spans: " + " | ".join(first_spans))
    cue_text = " || ".join(part[:240] for part in cue_parts if part).strip()
    if not cue_text:
        cue_text = content[:480].strip()

    now = utc_now()
    return EpisodeCue(
        episode_id=episode.id,
        group_id=episode.group_id,
        discourse_class=discourse_class,
        projection_state=decision.projection_state,
        cue_score=round(cue_score, 4),
        salience_score=round(salience_score, 4),
        projection_priority=decision.projection_priority,
        route_reason=decision.route_reason,
        cue_text=cue_text[:1000],
        entity_mentions=mentions,
        temporal_markers=temporal_markers,
        quote_spans=quote_spans,
        contradiction_keys=contradiction_keys,
        first_spans=first_spans,
        policy_score=decision.policy_score,
        created_at=now,
        updated_at=now,
    )
