"""Sparse high-signal agent promotion policy.

Meaning is expensive; raw capture is cheap. This module defines the allowlist,
caps, and filters that keep `remember` focused on durable facts instead of
session recap or bulk ETL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern

# Durable types agents should promote. Everything else is optional noise.
HIGH_SIGNAL_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "Decision",
        "Preference",
        "Person",
        "Correction",
        "Goal",
        "Commitment",
        "Organization",
        "Project",
        "Event",
        "Date",
    }
)

# Client-proposed relationships must use high-signal predicates only.
# Free-form predicates (WORKS_AT, RELATED_TO spam, etc.) still allowed only when
# on the allowlist — everything else is rejected at proposal conversion time.
# The set is closed under PredicateCanonicalizer (canonicalize.py): every
# canonicalization image of an allowed predicate is itself allowed, so the
# gate and the stored edge vocabulary agree (LIVES_IN passes AND LOCATED_IN,
# the form actually written to the graph, passes too).
ALLOWED_CLIENT_PREDICATES: frozenset[str] = frozenset(
    {
        "DECIDED",
        "PREFERS",
        "RELATED_TO",
        "CORRECTS",
        "COMMITS_TO",
        "HAS_GOAL",
        "WORKS_AT",
        "WORKS_ON",
        "PART_OF",
        "OWNS",
        "USES",
        "LIVES_IN",
        "KNOWS",
        "MEMBER_OF",
        "OCCURRED_ON",
        "INSTANCE_OF",
        "DEPENDS_ON",
        "BLOCKED_BY",
        "SUPERSEDES",
        "CONTRADICTS",
        # Canonical targets (closure under canonicalization + named additions).
        "LOCATED_IN",
        "REQUIRES",
        "HAS_ROLE",
        "CREATED",
        "LIKES",
        "DISLIKES",
        "AIMS_FOR",
        "EXPERT_IN",
        "COLLABORATES_WITH",
        "LEADS",
    }
)

# Soft guidance for agents; also used to rank recall.
DURABLE_RECALL_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "Decision",
        "Preference",
        "Person",
        "Correction",
        "Goal",
        "Commitment",
        "Organization",
        "Project",
        "Intention",
    }
)

# Default cap for remember promotions per *promotion window* (not MCP lifetime).
# A window is one agent context era: between harness compactions, or after a long
# idle gap in a multi-day session. 0–5 durable facts per window stays sparse
# without punishing long-running agent sessions.
DEFAULT_SESSION_PROMOTE_CAP = 5  # alias kept for callers/tests
DEFAULT_PROMOTE_CAP_PER_WINDOW = 5
# If the agent is quiet this long, treat the next remember as a new window.
# Multi-day sessions with overnight pauses get a fresh budget without compaction.
DEFAULT_PROMOTION_WINDOW_IDLE_SECONDS = 4 * 60 * 60
# Sources that mean "harness just compacted / restarted context".
COMPACTION_SOURCES: frozenset[str] = frozenset(
    {
        "auto:compaction",
        "auto:compact",
        "harness:compaction",
        "harness:compact",
        "claude:precompact",
        "claude:postcompact",
        "cursor:compaction",
    }
)

# Reject pure session-recap content as a promotion target.
_RECAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhat we (did|worked on|discussed) today\b", re.I),
    re.compile(r"\bsession (summary|recap|notes)\b", re.I),
    re.compile(r"\btoday'?s work\b", re.I),
    re.compile(r"\bin this (session|conversation) we\b", re.I),
)

# Low-value auto-capture sources that should stay cue-only unless explicitly remembered.
AUTO_CAPTURE_SOURCES: frozenset[str] = frozenset(
    {
        "auto:prompt",
        "auto:response",
        "auto:hook",
        "api_auto_observe",
        "auto:warmup",
    }
)

# Bootstrap / Cadence scrap that floods Decision retrieval with false positives.
_DECISION_STATEMENT_NOISE: Pattern[str] = re.compile(
    r"(?i)decision_statement|machineshopscheduler:decision|:decision_statement:"
)


def is_decision_statement_noise(name: str | None) -> bool:
    """True for bootstrap/Cadence decision_statement scrap entity names."""
    return bool(name and _DECISION_STATEMENT_NOISE.search(name))


def should_protect_as_identity_core(entity_type: str | None, *, client_proposal: bool) -> bool:
    """High-signal client-promoted facts should survive prune/merge pressure."""
    return bool(client_proposal and is_high_signal_entity_type(entity_type))


def is_allowed_client_predicate(predicate: str | None) -> bool:
    """True when a client-proposed edge predicate is on the high-signal allowlist."""
    if not predicate:
        return False
    return predicate.strip().upper().replace(" ", "_") in ALLOWED_CLIENT_PREDICATES


def normalize_client_predicate(predicate: str | None) -> str:
    """Normalize predicate to UPPER_SNAKE for allowlist checks."""
    if not predicate:
        return ""
    return predicate.strip().upper().replace(" ", "_").replace("-", "_")


def identity_core_summary_conflict(
    existing_summary: str | None,
    proposed_summary: str | None,
    *,
    entity_type: str | None = None,
) -> bool:
    """True when a client proposal would silently overwrite a protected summary.

    ``proposed_summary`` must be the **raw proposed text**, never the
    ``merge_entity_attributes`` result (``f"{old}; {new}"``). Against the merged
    form, ``old in new`` always holds and this helper would never fire.

    Corrections are allowed to rewrite identity_core text. Empty proposed summary
    is not a conflict (no overwrite). Exact match is not a conflict. Compatible
    expansion (proposed contains existing as a contiguous phrase, or vice versa)
    is allowed only when comparing raw proposed vs existing.
    """
    if not existing_summary or not proposed_summary:
        return False
    if str(entity_type or "").strip().lower() == "correction":
        return False
    old = " ".join(existing_summary.split()).casefold()
    new = " ".join(proposed_summary.split()).casefold()
    if not old or not new:
        return False
    if old == new:
        return False
    # Reject merged-append artifacts if a caller accidentally passes them:
    # "existing; proposed" always contains old as a prefix before "; ".
    if "; " in new and new.startswith(old + "; "):
        # Strip the known-old prefix and re-evaluate proposed tail alone.
        tail = new[len(old) + 2 :].strip()
        if not tail or tail == old:
            return False
        new = tail
        if old == new:
            return False
    # Compatible expansion (raw proposed expands existing, or restates a subset).
    if old in new or new in old:
        return False
    return True


def identity_core_blocks_merge(entity_a: object, entity_b: object) -> bool:
    """Return True when merge would destroy a protected high-signal identity.

    Rules:
    - Unprotected pairs never block here.
    - Never merge identity_core into decision_statement scrap or GOLDEN_ noise.
    - Two identity_core entities only merge when names normalize equal.
    - One identity_core requires near-exact name similarity to absorb the other.
    """
    from engram.extraction.resolver import compute_similarity, normalize_name

    a_core = bool(getattr(entity_a, "identity_core", False))
    b_core = bool(getattr(entity_b, "identity_core", False))
    if not a_core and not b_core:
        return False

    name_a = str(getattr(entity_a, "name", "") or "")
    name_b = str(getattr(entity_b, "name", "") or "")
    if is_decision_statement_noise(name_a) or is_decision_statement_noise(name_b):
        return True

    # Don't let a clean Decision get absorbed into a GOLDEN_* dump name.
    if a_core and not b_core and name_b.startswith("GOLDEN_") and not name_a.startswith("GOLDEN_"):
        return True
    if b_core and not a_core and name_a.startswith("GOLDEN_") and not name_b.startswith("GOLDEN_"):
        return True

    if a_core and b_core:
        return normalize_name(name_a) != normalize_name(name_b)

    # Exactly one identity_core — require high similarity to merge.
    return compute_similarity(name_a, name_b) < 0.92


@dataclass
class PromotionFilterResult:
    """Outcome of filtering agent proposals for sparse promotion."""

    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    is_recap: bool = False
    truncated: bool = False


def is_session_recap(content: str) -> bool:
    """True when content looks like a session recap rather than a durable fact."""
    text = (content or "").strip()
    if not text:
        return False
    if any(pat.search(text) for pat in _RECAP_PATTERNS):
        return True
    # Long multi-bullet dump without a decision/preference cue → recap-ish.
    if len(text) > 1200 and text.count("\n") >= 8:
        lowered = text.lower()
        if not any(
            token in lowered
            for token in (
                "decided",
                "decision",
                "prefer",
                "preference",
                "correction",
                "committed",
                "goal:",
            )
        ):
            return True
    return False


def is_high_signal_entity_type(entity_type: str | None) -> bool:
    if not entity_type:
        return False
    return entity_type.strip() in HIGH_SIGNAL_ENTITY_TYPES


def is_durable_recall_entity_type(entity_type: str | None) -> bool:
    if not entity_type:
        return False
    return entity_type.strip() in DURABLE_RECALL_ENTITY_TYPES


def is_auto_capture_source(source: str | None) -> bool:
    if not source:
        return False
    s = source.strip()
    if s in AUTO_CAPTURE_SOURCES:
        return True
    return s.startswith("auto:")


def filter_promotion_proposals(
    content: str,
    entities: list[dict] | None,
    relationships: list[dict] | None,
    *,
    max_entities: int = 8,
    max_relationships: int = 8,
    prefer_high_signal: bool = True,
) -> PromotionFilterResult:
    """Filter and lightly sanitize agent proposals for remember().

    - Rejects session-recap content entirely (caller should use observe).
    - Prefers high-signal entity types when present.
    - Caps entity/relationship counts so one remember cannot dump a graph dump.
    - Auto-fills source_span from the entity/rel name when missing and present in content.
    """
    result = PromotionFilterResult()
    if is_session_recap(content):
        result.is_recap = True
        result.rejected.append("session_recap")
        return result

    raw_entities = [dict(e) for e in (entities or []) if isinstance(e, dict)]
    raw_rels = [dict(r) for r in (relationships or []) if isinstance(r, dict)]

    # Auto-fill spans from content when the name appears verbatim.
    for ent in raw_entities:
        name = str(ent.get("name") or "").strip()
        if name and not ent.get("source_span"):
            if _span_in_content(name, content):
                ent["source_span"] = name
        # Prefer Decision default only when agent omitted type but name looks like one.
        if not ent.get("entity_type") and name:
            ent["entity_type"] = "Concept"

    for rel in raw_rels:
        if not rel.get("source_span"):
            for key in ("subject", "object", "predicate"):
                piece = str(rel.get(key) or "").strip()
                if piece and _span_in_content(piece, content):
                    rel["source_span"] = piece
                    break
            # Fallback: subject + object phrase if present.
            subj = str(rel.get("subject") or "").strip()
            obj = str(rel.get("object") or "").strip()
            phrase = f"{subj} {obj}".strip()
            if phrase and _span_in_content(phrase, content):
                rel["source_span"] = phrase

    if prefer_high_signal and raw_entities:
        high = [
            e for e in raw_entities if is_high_signal_entity_type(str(e.get("entity_type") or ""))
        ]
        # Keep high-signal first; if any exist, drop pure noise Concept/Artifact bulk
        # beyond the cap by sorting high-signal ahead.
        raw_entities = sorted(
            raw_entities,
            key=lambda e: (
                0 if is_high_signal_entity_type(str(e.get("entity_type") or "")) else 1,
                str(e.get("name") or ""),
            ),
        )
        if high and len(raw_entities) > max_entities:
            # Prefer keeping high-signal when truncating.
            kept_high = high[:max_entities]
            remaining_slots = max_entities - len(kept_high)
            others = [
                e
                for e in raw_entities
                if not is_high_signal_entity_type(str(e.get("entity_type") or ""))
            ][:remaining_slots]
            raw_entities = kept_high + others
            result.truncated = True

    if len(raw_entities) > max_entities:
        raw_entities = raw_entities[:max_entities]
        result.truncated = True
    if len(raw_rels) > max_relationships:
        raw_rels = raw_rels[:max_relationships]
        result.truncated = True

    result.entities = raw_entities
    result.relationships = raw_rels
    return result


def _span_in_content(span: str, content: str) -> bool:
    if not span or not content:
        return False
    needle = " ".join(span.split()).casefold()
    hay = " ".join(content.split()).casefold()
    return bool(needle) and needle in hay


def durable_result_boost(entity_type: str | None) -> float:
    """Score boost for ranking durable entity results above recap packets."""
    if is_durable_recall_entity_type(entity_type):
        if entity_type in {"Decision", "Preference", "Correction", "Person"}:
            return 2.5
        return 1.5
    return 0.0


def is_compaction_source(source: str | None) -> bool:
    """True when capture source indicates harness context compaction."""
    if not source:
        return False
    s = source.strip().lower()
    if s in COMPACTION_SOURCES:
        return True
    return "compact" in s and (s.startswith("auto:") or s.startswith("harness:"))


def default_promotion_window_path() -> str:
    """Path stamped by harness PreCompact hooks (shared across MCP processes)."""
    import os
    from pathlib import Path

    override = os.environ.get("ENGRAM_PROMOTION_WINDOW_FILE")
    if override:
        return override
    return str(Path.home() / ".engram" / "promotion-window.json")


def load_external_compaction_id(
    path: str | None = None,
    *,
    max_age_seconds: float = 24 * 60 * 60,
) -> str | None:
    """Read compaction_id written by PreCompact hook, if fresh enough."""
    import json
    import time
    from pathlib import Path

    window_path = Path(path or default_promotion_window_path())
    if not window_path.is_file():
        return None
    try:
        data = json.loads(window_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    compaction_id = str(data.get("compaction_id") or "").strip()
    if not compaction_id:
        return None
    reset_at = str(data.get("reset_at") or "")
    if reset_at:
        try:
            # Accept trailing Z
            from datetime import datetime

            parsed = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
            age = time.time() - parsed.timestamp()
            if age > max_age_seconds or age < -60:
                return None
        except Exception:
            pass
    return compaction_id


@dataclass
class PromotionWindowState:
    """Sparse-promotion budget for one agent context era."""

    window_id: str
    remember_count: int = 0
    started_at: float = 0.0
    last_remember_at: float | None = None
    reset_reason: str = "session_start"
    cap: int = DEFAULT_PROMOTE_CAP_PER_WINDOW
    idle_seconds: float = DEFAULT_PROMOTION_WINDOW_IDLE_SECONDS

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.remember_count)

    @property
    def at_cap(self) -> bool:
        return self.remember_count >= self.cap

    def to_meta(self) -> dict:
        return {
            "window_id": self.window_id,
            "remember_count": self.remember_count,
            "remaining": self.remaining,
            "cap": self.cap,
            "reset_reason": self.reset_reason,
            "idle_seconds": self.idle_seconds,
            # Back-compat keys used by earlier session-cap responses.
            "session_remember_count": self.remember_count,
            "session_promote_cap": self.cap,
            "promotion_unit": "compaction_window",
        }


def _new_window_id(prefix: str = "win") -> str:
    import time
    import uuid

    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def resolve_promotion_window(
    session: object,
    *,
    compaction_id: str | None = None,
    source: str | None = None,
    now: float | None = None,
    cap: int = DEFAULT_PROMOTE_CAP_PER_WINDOW,
    idle_seconds: float = DEFAULT_PROMOTION_WINDOW_IDLE_SECONDS,
    read_external_window: bool = True,
) -> PromotionWindowState:
    """Return the active promotion window, resetting when the context era changes.

    Reset triggers (in order):
    1. Explicit ``compaction_id`` differs from the session's current window id
    2. Fresh PreCompact file under ``~/.engram/promotion-window.json``
    3. Source looks like a harness compaction event
    4. Idle gap exceeds ``idle_seconds`` (multi-day sessions overnight)
    5. First call on a fresh session (no window yet)

    Engram cannot *see* agent compaction by itself — the harness must pass
    ``compaction_id``, stamp the shared window file, or use a compaction source.
    Idle reset is the fallback when the agent stays connected for days without
    a compact signal.
    """
    import time

    ts = time.time() if now is None else float(now)
    current_id = getattr(session, "promotion_window_id", None)
    count = int(getattr(session, "remember_count", 0) or 0)
    started = float(getattr(session, "promotion_window_started_at", 0.0) or 0.0)
    last = getattr(session, "last_remember_at", None)
    last_f = float(last) if last is not None else None
    reset_reason = str(
        getattr(session, "promotion_window_reset_reason", "session_start") or "session_start"
    )

    should_reset = False
    reason = reset_reason

    effective_compaction_id = (compaction_id or "").strip() or None
    if not effective_compaction_id and read_external_window:
        effective_compaction_id = load_external_compaction_id()

    if effective_compaction_id:
        if current_id != effective_compaction_id:
            should_reset = True
            reason = "compaction_id" if compaction_id else "precompact_file"
            current_id = effective_compaction_id
    elif is_compaction_source(source):
        should_reset = True
        reason = "compaction_source"
        current_id = _new_window_id("compact")
    elif current_id is None:
        should_reset = True
        reason = "session_start"
        current_id = _new_window_id("session")
    elif last_f is not None and (ts - last_f) >= idle_seconds:
        should_reset = True
        reason = "idle_gap"
        current_id = _new_window_id("idle")
    elif started and (ts - started) >= idle_seconds and count == 0 and last_f is None:
        # Window opened ages ago with no remembers — still fine; keep it.
        pass

    if should_reset:
        count = 0
        started = ts
        last_f = None
        # Persist onto session object when it supports attributes.
        try:
            session.promotion_window_id = current_id  # type: ignore[attr-defined]
            session.remember_count = 0  # type: ignore[attr-defined]
            session.promotion_window_started_at = started  # type: ignore[attr-defined]
            session.last_remember_at = None  # type: ignore[attr-defined]
            session.promotion_window_reset_reason = reason  # type: ignore[attr-defined]
        except Exception:
            pass

    return PromotionWindowState(
        window_id=str(current_id or _new_window_id("session")),
        remember_count=count,
        started_at=started or ts,
        last_remember_at=last_f,
        reset_reason=reason,
        cap=cap,
        idle_seconds=idle_seconds,
    )


def record_promotion_in_window(
    session: object, window: PromotionWindowState, *, now: float | None = None
) -> PromotionWindowState:
    """Increment the window's remember count after a successful promotion."""
    import time

    ts = time.time() if now is None else float(now)
    window.remember_count += 1
    window.last_remember_at = ts
    try:
        session.remember_count = window.remember_count  # type: ignore[attr-defined]
        session.last_remember_at = ts  # type: ignore[attr-defined]
        session.promotion_window_id = window.window_id  # type: ignore[attr-defined]
        if not getattr(session, "promotion_window_started_at", None):
            session.promotion_window_started_at = window.started_at  # type: ignore[attr-defined]
    except Exception:
        pass
    return window
