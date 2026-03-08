"""Deterministic memory-need analysis for recall arbitration."""

from __future__ import annotations

import re
import time

from engram.models.recall import MemoryNeed
from engram.retrieval.control import RecallNeedThresholds
from engram.retrieval.graph_probe import ProbeResult
from engram.retrieval.signals import SignalReport, extract_signals

_ACK_PATTERNS = (
    r"^ok(?:ay)?[.!]?$",
    r"^thanks(?: you)?[.!]?$",
    r"^got it[.!]?$",
    r"^sounds good[.!]?$",
    r"^cool[.!]?$",
    r"^nice[.!]?$",
    r"^yep[.!]?$",
    r"^yup[.!]?$",
    r"^sure[.!]?$",
    r"^hi[.!]?$",
    r"^hello[.!]?$",
)
_BROAD_CONTEXT_PATTERNS = (
    r"\bcatch me up\b",
    r"\brecap\b",
    r"\bbrief me\b",
    r"\bwhat do you know\b",
    r"\bgive me (?:the )?context\b",
    r"\bremind me where we are\b",
)
_IDENTITY_PATTERNS = (
    r"\bwho am i\b",
    r"\bwhat(?:'s| is) my name\b",
    r"\babout me\b",
    r"\bwhat do i (?:like|prefer)\b",
    r"\bmy preferences?\b",
)
_OPEN_LOOP_PATTERNS = (
    r"\bdid we decide\b",
    r"\bwhat did we decide\b",
    r"\bwhere did we land\b",
    r"\bnext steps?\b",
    r"\bblocked\b",
    r"\bpending\b",
    r"\bstill waiting\b",
    r"\bopen (?:item|loop|question)\b",
    r"\bfollow(?: |-)?up\b",
)
_TEMPORAL_PATTERNS = (
    r"\bwhat changed\b",
    r"\blatest\b",
    r"\brecent(?:ly)?\b",
    r"\blast time\b",
    r"\bprevious(?:ly)?\b",
    r"\bearlier\b",
    r"\bsince\b",
    r"\btimeline\b",
    r"\bupdate\b",
    r"\bstatus\b",
    r"\bhow'?s .* going\b",
)
_PROSPECTIVE_PATTERNS = (
    r"\bremind me\b",
    r"\bdon't let me forget\b",
    r"\bnext time\b",
    r"\blater\b",
)
_PROJECT_PATTERNS = (
    r"\bproject\b",
    r"\bmigration\b",
    r"\bfeature\b",
    r"\bbug\b",
    r"\btask\b",
    r"\bissue\b",
    r"\broadmap\b",
    r"\brollout\b",
    r"\bauth\b",
    r"\brelease\b",
)
_FACT_QUERY_PREFIX = re.compile(
    r"^(who|what|when|where|which|did|does|do|is|are|was|were|have|has|can)\b",
    re.IGNORECASE,
)
_FOLLOWUP_MARKERS = re.compile(
    r"\b(it|that|this|they|them|those|he|she|we|our|still)\b",
    re.IGNORECASE,
)
_USER_REFERENCE_MARKERS = re.compile(
    r"\b(my|our|we|you said|remember|last time)\b",
    re.IGNORECASE,
)
_DECISION_MARKERS = re.compile(
    r"\b(decid(?:e|ed|ing)|decision|land|settle|choose|go with|option)\b",
    re.IGNORECASE,
)
_NON_ENTITY_TITLE_WORDS = {
    "did",
    "do",
    "does",
    "can",
    "is",
    "are",
    "was",
    "were",
    "have",
    "has",
    "how",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "we",
    "i",
    "you",
    "they",
    "he",
    "she",
    "it",
    "this",
    "that",
}


async def analyze_memory_need(
    current_turn: str,
    *,
    recent_turns: list[str] | None = None,
    session_entity_names: list[str] | None = None,
    mode: str = "auto_recall",
    graph_probe=None,
    group_id: str = "default",
    conv_context=None,
    cfg=None,
    thresholds: RecallNeedThresholds | None = None,
) -> MemoryNeed:
    """Classify whether recall is likely useful for the current turn."""
    started_at = time.perf_counter()
    active_thresholds = thresholds or RecallNeedThresholds()
    text = " ".join(current_turn.strip().split())
    lowered = text.lower()
    recent_turns = [turn.strip() for turn in (recent_turns or []) if turn.strip()]
    session_entity_names = [name for name in (session_entity_names or []) if name]
    has_recent_context = bool(recent_turns)

    if not text:
        return _finalize_need(
            MemoryNeed(
                need_type="none",
                should_recall=False,
                confidence=0.99,
                reasons=["empty_turn"],
            ),
            thresholds=active_thresholds,
            started_at=started_at,
            decision_path="none",
        )

    if _is_acknowledgement(lowered):
        return _finalize_need(
            MemoryNeed(
                need_type="none",
                should_recall=False,
                confidence=0.95,
                reasons=["acknowledgement"],
            ),
            thresholds=active_thresholds,
            started_at=started_at,
            decision_path="none",
        )

    signal_report = extract_signals(
        text,
        lowered,
        recent_turns=recent_turns,
        session_entity_names=session_entity_names,
        conv_context=conv_context,
        cfg=cfg,
    )

    keyword_need = _keyword_match(
        text,
        lowered,
        recent_turns=recent_turns,
        session_entity_names=session_entity_names,
        mode=mode,
        has_recent_context=has_recent_context,
        signal_report=signal_report,
    )
    if keyword_need is not None:
        return _finalize_need(
            keyword_need,
            thresholds=active_thresholds,
            started_at=started_at,
            decision_path="keyword",
        )

    borderline_score = max(
        signal_report.linguistic_score,
        signal_report.pragmatic_score,
        signal_report.structural_score,
    )
    probe_result = ProbeResult()
    probe_latency_ms = 0.0
    probe_triggered = False
    override_probe_candidate = bool(
        cfg is not None
        and getattr(cfg, "recall_need_graph_override_enabled", False)
        and (_has_named_terms(text) or bool(signal_report.all_referents))
    )
    if (
        graph_probe is not None
        and (
            borderline_score >= active_thresholds.borderline_score
            or override_probe_candidate
        )
        and (cfg is None or getattr(cfg, "recall_need_graph_probe_enabled", False))
    ):
        probe_triggered = True
        probe_started_at = time.perf_counter()
        probe_result = await graph_probe.probe(
            text,
            lowered,
            referents=signal_report.all_referents,
            group_id=group_id,
        )
        probe_latency_ms = (time.perf_counter() - probe_started_at) * 1000.0

    graph_lift = (
        signal_report.linguistic_score < active_thresholds.linguistic_score
        and borderline_score >= active_thresholds.borderline_score
        and probe_result.resonance_score >= active_thresholds.resonance_score
    )
    should_recall = (
        signal_report.linguistic_score >= active_thresholds.linguistic_score
        or graph_lift
    )
    if should_recall:
        need_type = _classify_signal_need(signal_report, lowered)
        reasons = [
            f"{signal_report.dominant_family}:{signal_report.dominant_trigger_kind or 'signal'}"
        ]
        if graph_lift:
            reasons.append("graph_resonance_support")
        return _finalize_need(
            _attach_signals(
                MemoryNeed(
                    need_type=need_type,
                    should_recall=True,
                    confidence=_signal_confidence(signal_report, probe_result),
                    reasons=reasons,
                    query_hint=signal_report.best_query_hint
                    or _build_query_hint(text, recent_turns, session_entity_names),
                    urgency=_signal_urgency(need_type, signal_report),
                    packet_budget=1,
                    entity_budget=4 if need_type in {"open_loop", "project_state"} else 3,
                ),
                signal_report,
                probe_result,
            ),
            thresholds=active_thresholds,
            started_at=started_at,
            decision_path="graph_lift" if graph_lift else "linguistic",
            probe_triggered=probe_triggered,
            probe_latency_ms=probe_latency_ms,
        )

    if _should_graph_override(
        signal_report,
        probe_result,
        cfg=cfg,
        thresholds=active_thresholds,
    ):
        return _finalize_need(
            _attach_signals(
                MemoryNeed(
                    need_type="broad_context",
                    should_recall=True,
                    confidence=max(0.72, _signal_confidence(signal_report, probe_result)),
                    reasons=["graph:graph_override", "graph_resonance_override"],
                    query_hint=signal_report.best_query_hint
                    or _build_query_hint(text, recent_turns, session_entity_names),
                    urgency=0.55,
                    packet_budget=1,
                    entity_budget=3,
                ),
                signal_report,
                probe_result,
                trigger_family="graph",
                trigger_kind="graph_override",
            ),
            thresholds=active_thresholds,
            started_at=started_at,
            decision_path="graph_override",
            probe_triggered=probe_triggered,
            probe_latency_ms=probe_latency_ms,
            graph_override_used=True,
        )

    return _finalize_need(
        _attach_signals(
        MemoryNeed(
            need_type="none",
            should_recall=False,
            confidence=0.7 if mode == "chat" else 0.82,
            reasons=["no_strong_memory_signal"],
        ),
        signal_report,
        probe_result,
        ),
        thresholds=active_thresholds,
        started_at=started_at,
        decision_path="none",
        probe_triggered=probe_triggered,
        probe_latency_ms=probe_latency_ms,
    )


def _keyword_match(
    text: str,
    lowered: str,
    *,
    recent_turns: list[str],
    session_entity_names: list[str],
    mode: str,
    has_recent_context: bool,
    signal_report: SignalReport,
) -> MemoryNeed | None:
    if _matches_any(lowered, _BROAD_CONTEXT_PATTERNS):
        return _attach_signals(
            MemoryNeed(
                need_type="broad_context",
                should_recall=True,
                confidence=0.9,
                reasons=["explicit_context_request"],
                query_hint=_build_query_hint(text, recent_turns, session_entity_names),
                urgency=0.5,
                packet_budget=2,
                entity_budget=5,
            ),
            signal_report,
            trigger_family="keyword",
            trigger_kind="broad_context",
        )

    if _matches_any(lowered, _PROSPECTIVE_PATTERNS):
        return _attach_signals(
            MemoryNeed(
                need_type="prospective",
                should_recall=True,
                confidence=0.84,
                reasons=["future_or_trigger_language"],
                query_hint=_build_query_hint(text, recent_turns, session_entity_names),
                urgency=0.65,
                packet_budget=1,
                entity_budget=3,
            ),
            signal_report,
            trigger_family="keyword",
            trigger_kind="prospective",
        )

    if _matches_any(lowered, _IDENTITY_PATTERNS):
        return _attach_signals(
            MemoryNeed(
                need_type="identity",
                should_recall=True,
                confidence=0.9,
                reasons=["identity_reference"],
                query_hint=_build_query_hint(text, recent_turns, session_entity_names),
                urgency=0.6,
                packet_budget=1,
                entity_budget=3,
            ),
            signal_report,
            trigger_family="keyword",
            trigger_kind="identity",
        )

    if _matches_any(lowered, _OPEN_LOOP_PATTERNS):
        return _attach_signals(
            MemoryNeed(
                need_type="open_loop",
                should_recall=True,
                confidence=0.86,
                reasons=["open_loop_language"],
                query_hint=_build_query_hint(text, recent_turns, session_entity_names),
                urgency=0.7,
                packet_budget=1,
                entity_budget=4,
            ),
            signal_report,
            trigger_family="keyword",
            trigger_kind="open_loop",
        )

    if _matches_any(lowered, _TEMPORAL_PATTERNS):
        need_type = (
            "project_state" if _matches_any(lowered, _PROJECT_PATTERNS) else "temporal_update"
        )
        return _attach_signals(
            MemoryNeed(
                need_type=need_type,
                should_recall=True,
                confidence=0.82,
                reasons=["temporal_language"],
                query_hint=_build_query_hint(text, recent_turns, session_entity_names),
                urgency=0.7,
                packet_budget=1,
                entity_budget=4,
            ),
            signal_report,
            trigger_family="keyword",
            trigger_kind="temporal",
        )

    if _matches_any(lowered, _PROJECT_PATTERNS) and (
        has_recent_context or _USER_REFERENCE_MARKERS.search(lowered) or _has_named_terms(text)
    ):
        return _attach_signals(
            MemoryNeed(
                need_type="project_state",
                should_recall=True,
                confidence=0.78,
                reasons=["project_reference"],
                query_hint=_build_query_hint(text, recent_turns, session_entity_names),
                urgency=0.62,
                packet_budget=1,
                entity_budget=4,
            ),
            signal_report,
            trigger_family="keyword",
            trigger_kind="project_state",
        )

    if _FACT_QUERY_PREFIX.search(text) and (
        _USER_REFERENCE_MARKERS.search(lowered)
        or (has_recent_context and _FOLLOWUP_MARKERS.search(lowered))
        or _has_named_terms(text)
        or bool(session_entity_names)
    ):
        return _attach_signals(
            MemoryNeed(
                need_type="fact_lookup",
                should_recall=True,
                confidence=0.8 if mode == "chat" else 0.72,
                reasons=["question_requires_prior_fact"],
                query_hint=_build_query_hint(text, recent_turns, session_entity_names),
                urgency=0.6,
                packet_budget=1,
                entity_budget=4,
            ),
            signal_report,
            trigger_family="keyword",
            trigger_kind="fact_lookup",
        )

    if has_recent_context and _FOLLOWUP_MARKERS.search(lowered):
        return _attach_signals(
            MemoryNeed(
                need_type="open_loop",
                should_recall=True,
                confidence=0.74,
                reasons=["follow_up_reference"],
                query_hint=_build_query_hint(text, recent_turns, session_entity_names),
                urgency=0.55,
                packet_budget=1,
                entity_budget=3,
            ),
            signal_report,
            trigger_family="keyword",
            trigger_kind="follow_up",
        )

    return None


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _is_acknowledgement(text: str) -> bool:
    return any(re.match(pattern, text, re.IGNORECASE) for pattern in _ACK_PATTERNS)


def _has_named_terms(text: str) -> bool:
    return bool(_extract_named_terms(text))


def _extract_named_terms(text: str) -> list[str]:
    matches = re.findall(r"\b[A-Z][A-Za-z0-9.+_-]*(?:\s+[A-Z][A-Za-z0-9.+_-]*)*\b", text)
    deduped: list[str] = []
    seen: set[str] = set()
    for match in matches:
        normalized = match.strip()
        tokens = [token.lower() for token in normalized.split()]
        if all(token in _NON_ENTITY_TITLE_WORDS for token in tokens):
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _attach_signals(
    need: MemoryNeed,
    report: SignalReport,
    probe_result: ProbeResult | None = None,
    *,
    trigger_family: str | None = None,
    trigger_kind: str | None = None,
) -> MemoryNeed:
    """Attach analyzer metadata to a MemoryNeed."""
    scores = report.to_scores_dict()
    if scores:
        need.signal_scores = scores
    if report.all_referents:
        need.detected_referents = report.all_referents
    need.trigger_family = trigger_family or report.dominant_family
    need.trigger_kind = trigger_kind or report.dominant_trigger_kind
    if probe_result is not None:
        if probe_result.detected_entities:
            need.detected_entities = probe_result.detected_entities
        if probe_result.resonance_score > 0:
            need.resonance_score = probe_result.resonance_score
    return need


def _classify_signal_need(report: SignalReport, lowered: str) -> str:
    trigger_kind = report.dominant_trigger_kind or ""
    if trigger_kind == "callback":
        return "open_loop"
    if trigger_kind == "memory_gap":
        return "open_loop" if _DECISION_MARKERS.search(lowered) else "fact_lookup"
    if trigger_kind == "correction":
        return "temporal_update"
    if trigger_kind == "life_update":
        return "project_state" if _matches_any(lowered, _PROJECT_PATTERNS) else "temporal_update"
    if trigger_kind == "identity_claim":
        return "identity"
    if trigger_kind == "status_check":
        return "project_state"
    if trigger_kind in {"continuation", "recurring_problem"}:
        return "open_loop"
    if trigger_kind == "comparison":
        return "broad_context"
    if trigger_kind in {"implicit_preference"}:
        return "identity"
    if trigger_kind == "introduction":
        return "identity" if _is_user_anchored_introduction(lowered, report) else "broad_context"
    if trigger_kind == "planning":
        return "prospective" if _planning_is_first_person(lowered) else "project_state"
    if trigger_kind == "social_graph_update":
        return "temporal_update"
    if trigger_kind == "milestone":
        return "project_state"
    if trigger_kind == "temporal_narrative":
        return "temporal_update"
    if trigger_kind == "delegation":
        return "project_state"
    if trigger_kind in {"causal_context", "emotional_anchor"}:
        return "broad_context"
    return _classify_pragmatic_need(report, lowered)


def _classify_pragmatic_need(report: SignalReport, lowered: str) -> str:
    continuation_kinds = {
        signal.trigger_kind
        for signal in report.signals
        if signal.trigger_kind and signal.trigger_kind.startswith("continuation_")
    }
    if continuation_kinds:
        if any(
            kind in continuation_kinds
            for kind in (
                "continuation_finally",
                "continuation_already",
                "continuation_no_longer",
                "continuation_turns_out",
            )
        ):
            return "temporal_update"
        return "open_loop"
    if _matches_any(lowered, _PROJECT_PATTERNS) and any(
        signal.trigger_kind in {"possessive_relational", "bare_name"} for signal in report.signals
    ):
        return "project_state"
    if report.dominant_family == "impoverishment" and _matches_any(lowered, _PROJECT_PATTERNS):
        return "project_state"
    return "fact_lookup"


def _signal_confidence(report: SignalReport, probe_result: ProbeResult) -> float:
    return min(
        0.9,
        0.48 + (report.linguistic_score * 0.65) + (probe_result.resonance_score * 0.18),
    )


def _signal_urgency(need_type: str, report: SignalReport) -> float:
    base = 0.52
    if need_type in {"open_loop", "project_state"}:
        base = 0.62
    elif need_type == "temporal_update":
        base = 0.58
    elif need_type == "prospective":
        base = 0.64
    return min(0.8, base + (report.impoverishment_score * 0.12))


def _should_graph_override(
    report: SignalReport,
    probe_result: ProbeResult,
    *,
    cfg,
    thresholds: RecallNeedThresholds,
) -> bool:
    if cfg is None or not getattr(cfg, "recall_need_graph_override_enabled", False):
        return False
    if report.linguistic_score >= thresholds.borderline_score:
        return False
    if not probe_result.entity_scores:
        return False
    if probe_result.resonance_score < float(
        getattr(cfg, "recall_need_graph_override_resonance_threshold", 0.72)
    ):
        return False
    top_entity_id, top_entity_score = max(
        probe_result.entity_scores.items(),
        key=lambda item: item[1],
    )
    if top_entity_score < 0.65:
        return False
    return top_entity_id in set(probe_result.anchored_entity_ids)


def _planning_is_first_person(lowered: str) -> bool:
    return bool(
        re.search(
            r"\b(i|we)\b.*\b(thinking about|planning to|want to|going to|considering|"
            r"might|hoping to|aiming to|working toward)\b",
            lowered,
            re.IGNORECASE,
        )
    )


def _is_user_anchored_introduction(report_text: str, report: SignalReport) -> bool:
    referents = {referent.lower() for referent in report.all_referents}
    return (
        any(marker in report_text for marker in ("my ", "our ", "this is my", "we have a new"))
        or bool(referents & {"son", "daughter", "friend", "coworker", "manager", "boss", "kid"})
    )


def _finalize_need(
    need: MemoryNeed,
    *,
    thresholds: RecallNeedThresholds,
    started_at: float,
    decision_path: str,
    probe_triggered: bool = False,
    probe_latency_ms: float = 0.0,
    graph_override_used: bool = False,
) -> MemoryNeed:
    need.decision_path = decision_path
    need.thresholds = thresholds.to_dict()
    need.analyzer_latency_ms = round((time.perf_counter() - started_at) * 1000.0, 4)
    need.probe_triggered = probe_triggered
    need.probe_latency_ms = round(probe_latency_ms, 4)
    need.graph_override_used = graph_override_used
    return need


def _build_query_hint(
    current_turn: str,
    recent_turns: list[str],
    session_entity_names: list[str],
) -> str | None:
    named_terms = _extract_named_terms(current_turn)
    if named_terms:
        return " ".join(named_terms[:3])[:200]

    if session_entity_names:
        return " ".join(session_entity_names[:3])[:200]

    for turn in reversed(recent_turns):
        if len(turn) >= 20:
            return turn[:200]

    if len(current_turn) >= 20:
        first_sentence = current_turn.split(".")[0].strip()
        return first_sentence[:200]

    return None
