"""Signal extraction for recall-need analysis."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

_RELATIONAL_WEIGHTS = {
    "son": 0.40,
    "daughter": 0.40,
    "wife": 0.40,
    "husband": 0.40,
    "partner": 0.40,
    "mom": 0.40,
    "dad": 0.40,
    "mother": 0.40,
    "father": 0.40,
    "brother": 0.40,
    "sister": 0.40,
    "kid": 0.40,
    "kids": 0.40,
    "child": 0.40,
    "children": 0.40,
    "baby": 0.40,
    "friend": 0.30,
    "roommate": 0.30,
    "neighbor": 0.30,
    "boss": 0.35,
    "manager": 0.35,
    "coworker": 0.35,
    "colleague": 0.35,
    "teammate": 0.35,
    "mentor": 0.35,
    "mentee": 0.35,
    "client": 0.35,
    "doctor": 0.35,
    "therapist": 0.35,
    "dentist": 0.35,
    "coach": 0.35,
    "teacher": 0.35,
    "professor": 0.35,
    "tutor": 0.35,
    "vet": 0.35,
}
_RELATIONAL_NOUNS = set(_RELATIONAL_WEIGHTS)
_TECHNICAL_POSSESSIVES = {
    "api",
    "branch",
    "build",
    "cli",
    "cluster",
    "code",
    "codebase",
    "config",
    "container",
    "database",
    "deployment",
    "editor",
    "endpoint",
    "environment",
    "function",
    "instance",
    "migration",
    "package",
    "pipeline",
    "pr",
    "query",
    "repo",
    "schema",
    "server",
    "stack",
    "terminal",
}
_NON_RELATIONAL_POSSESSIVES = {
    "apologies",
    "bad",
    "concern",
    "fault",
    "god",
    "goodness",
    "guess",
    "issue",
    "opinion",
    "pleasure",
    "point",
    "question",
    "take",
    "turn",
    "understanding",
    "way",
}
_POSSESSIVE_PATTERN = re.compile(
    r"(?i:\b(my|our|his|her|their)\s+([A-Za-z][A-Za-z'-]*)\b)(?:\s+([A-Z][a-z]+))?",
)

_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
_COMMUNAL_GROUND = {
    "anthropic",
    "claude",
    "docker",
    "engram",
    "fastapi",
    "falkordb",
    "github",
    "gitlab",
    "go",
    "golang",
    "java",
    "javascript",
    "kubernetes",
    "linux",
    "mysql",
    "next",
    "openai",
    "phoenix",
    "postgres",
    "postgresql",
    "python",
    "react",
    "redis",
    "ruby",
    "rust",
    "sqlite",
    "typescript",
}
_AMBIGUOUS_NAMES = {
    "art",
    "bill",
    "chase",
    "crystal",
    "faith",
    "grace",
    "hope",
    "june",
    "mark",
    "may",
    "pat",
    "rich",
    "ruby",
    "will",
}
_COMMON_CAPITALIZED_WORDS = {
    "actually",
    "also",
    "anyway",
    "broad",
    "btw",
    "can",
    "catch",
    "did",
    "do",
    "does",
    "finally",
    "hello",
    "hey",
    "hi",
    "how",
    "i",
    "if",
    "my",
    "oh",
    "our",
    "please",
    "random",
    "remind",
    "sorry",
    "still",
    "talked",
    "thanks",
    "that",
    "the",
    "this",
    "we",
    "what",
    "when",
    "where",
    "why",
}
_PERSON_FOLLOWING_VERBS = {
    "called",
    "emailed",
    "got",
    "had",
    "is",
    "joined",
    "left",
    "loved",
    "met",
    "moved",
    "said",
    "says",
    "scored",
    "texted",
    "thinks",
    "was",
    "won",
}
_PERSONAL_TERMS = {
    "daughter",
    "family",
    "friend",
    "girlfriend",
    "husband",
    "kid",
    "kids",
    "mom",
    "mother",
    "partner",
    "son",
    "wife",
}
_TECHNICAL_TERMS = {
    "api",
    "branch",
    "build",
    "cache",
    "commit",
    "config",
    "database",
    "deploy",
    "deployment",
    "docker",
    "endpoint",
    "frontend",
    "function",
    "migration",
    "pipeline",
    "pr",
    "query",
    "refactor",
    "release",
    "schema",
    "server",
    "staging",
    "test",
}
_PROJECT_TERMS = {
    "api",
    "auth",
    "bug",
    "feature",
    "issue",
    "migration",
    "project",
    "release",
    "rollout",
    "roadmap",
    "task",
}

_INTRODUCTION_PREFIXES = (
    "my friend",
    "our friend",
    "this guy",
    "this person",
    "a guy named",
    "a person named",
)
_ANAPHORA_PATTERN = re.compile(r"\b(she|he|they|them|it|that|this)\b", re.IGNORECASE)
_EXPLETIVE_IT_PATTERNS = (
    re.compile(r"^it(?:'s|\s+is|\s+was)\s+(?:cold|hot|late|early|fine|okay|ok|raining)\b"),
    re.compile(r"^it\s+(?:seems?|looks?|appears?|turns out)\b"),
    re.compile(r"^it\s+(?:doesn't|does not)\s+matter\b"),
)
_HEDGE_MARKERS = (
    "btw",
    "by the way",
    "anyway",
    "oh and",
    "random but",
    "sorry",
    "side note",
    "speaking of",
    "that reminds me",
    "incidentally",
    "fwiw",
    "fyi",
    "oh right",
)
_CONTINUATION_MARKERS = {
    "still": re.compile(r"\bstill\s+(?:\w+ing\b|\w+ed\b|not\b|the same\b)", re.IGNORECASE),
    "again": re.compile(r"\bagain\b", re.IGNORECASE),
    "finally": re.compile(r"\bfinally\b", re.IGNORECASE),
    "back_to": re.compile(r"\bback to\b", re.IGNORECASE),
    "yet": re.compile(r"\b(?:hasn't|haven't|not)\s+.*\byet\b", re.IGNORECASE),
    "already": re.compile(r"\balready\b", re.IGNORECASE),
    "no_longer": re.compile(r"\bno longer\b", re.IGNORECASE),
    "turns_out": re.compile(r"\bturns out\b", re.IGNORECASE),
}

_CALLBACK_PATTERNS = (
    re.compile(r"\bremember when\b", re.IGNORECASE),
    re.compile(r"\b(back to|going back to|following up on|as we discussed)\b", re.IGNORECASE),
    re.compile(r"\b(what you suggested|that approach|that thing we talked about)\b", re.IGNORECASE),
    re.compile(r"\b(like last time|same as last time|about that)\b", re.IGNORECASE),
)
_MEMORY_GAP_PATTERNS = (
    re.compile(r"\bi (?:can't|cannot|don't) remember\b", re.IGNORECASE),
    re.compile(r"\bi forget\b", re.IGNORECASE),
    re.compile(r"\bnot sure (?:if|what|whether|which|who|where|when)\b", re.IGNORECASE),
    re.compile(r"\bcan't recall\b", re.IGNORECASE),
    re.compile(r"\bthere was (?:a|some) (?:reason|decision|plan|thing)\b", re.IGNORECASE),
)
_CORRECTION_PATTERNS = (
    re.compile(
        r"\b(actually|correction|to clarify|i meant|scratch that|never mind that|"
        r"i was wrong|i misspoke|wait no|on second thought)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(not anymore|no longer|that's changed|that's wrong|that's outdated)\b", re.IGNORECASE
    ),
    re.compile(r"\bnot .* but .*\b", re.IGNORECASE),
    re.compile(r"\bswitched from .* to .*\b", re.IGNORECASE),
)
_IDENTITY_PATTERNS = (
    re.compile(
        r"\bi(?:'m| am) (?:more of|kind of|the type|the kind|really|basically|"
        r"fundamentally|at heart)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bi (?:consider myself|identify as|see myself as)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'ve| have) always been\b", re.IGNORECASE),
)
_STATUS_PATTERNS = (
    re.compile(
        r"\b(how's|what's the status|any (?:progress|update|news)|where are we with)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdid .* (?:work|go through|land|ship|deploy|merge)\b", re.IGNORECASE),
)
_LIFE_UPDATE_PATTERNS = (
    re.compile(
        r"\b(?:i|we|my|our|[A-Z][a-z]+)\b.*\b(moved|switched|joined|left|started|finished|graduated|got|won|scored|shipped|launched|changed|became|promoted)\b"
    ),
    re.compile(r"\b(?:we|i) (?:finally )?(?:shipped|launched|finished|moved)\b", re.IGNORECASE),
)
_CONTINUATION_STRUCTURAL_PATTERNS = (
    re.compile(
        r"\b(?:so i|i) (?:tried|did|went|ended up|decided|finally|actually)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(took your (?:advice|suggestion)|following up|"
        r"as (?:planned|discussed|agreed)|went ahead and)\b",
        re.IGNORECASE,
    ),
)
_COMPARISON_PATTERNS = (
    re.compile(
        r"\b(like (?:the|that|when)|same as|similar to|reminds me of|unlike|"
        r"opposite of|different from|compared to)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(last time|before|previously|the old way)\b", re.IGNORECASE),
)
_PREFERENCE_PATTERNS = (
    re.compile(
        r"\bi (?:always|never|keep|tend to|usually|normally|typically|generally)"
        r"\b.*\b(use|choose|pick|go with|default to|gravitate toward)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bwhenever i .* i (?:use|choose|go with)\b", re.IGNORECASE),
)
_PLANNING_PATTERNS = (
    re.compile(
        r"\b(thinking about|planning to|want to|going to|considering|might|"
        r"hoping to|aiming to|working toward)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bnext (?:sprint|quarter|month|week|year)\b", re.IGNORECASE),
)
_SOCIAL_UPDATE_PATTERNS = (
    re.compile(
        r"\b(?:my|our)\s+(?:manager|boss|lead|partner|coworker)\b.*\b(?:is|was|got)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b[A-Z][a-z]+\b.*\b(?:is now|got moved|got promoted|is leaving|is joining|became)\b"
    ),
)
_CAUSAL_PATTERNS = (
    re.compile(
        r"\b(?:the reason|because|why we chose|why i chose|led to|caused by)\b", re.IGNORECASE
    ),
)
_RECURRING_PROBLEM_PATTERNS = (
    re.compile(
        r"\b(keeps happening|again and again|still broken|same bug|same issue|recurring)\b",
        re.IGNORECASE,
    ),
)
_MILESTONE_PATTERNS = (
    re.compile(
        r"\b(finally|at last)\b.*\b(shipped|launched|finished|done|graduated)\b", re.IGNORECASE
    ),
    re.compile(r"\b(got promoted|got accepted|closed the round|released v\d+)\b", re.IGNORECASE),
)
_INTRODUCTION_PATTERNS = (
    re.compile(r"\bthis is\b.*\b(?:my friend|my coworker|my daughter|my son)\b", re.IGNORECASE),
    re.compile(r"\bwe have a new\b", re.IGNORECASE),
)
_TEMPORAL_NARRATIVE_PATTERNS = (
    re.compile(r"\b(yesterday|today|last night|this morning|earlier)\b", re.IGNORECASE),
    re.compile(r"\bthen\b.*\bafter\b", re.IGNORECASE),
)
_DELEGATION_PATTERNS = (
    re.compile(
        r"\b(?:told|asked|assigned)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+to\s+"
        r"(?:do|handle|fix|review|own|take|ship|follow up)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:sent|gave|shared|handed)\s+me\b",
        re.IGNORECASE,
    ),
)
_EMOTIONAL_ANCHOR_PATTERNS = (
    re.compile(
        r"\b(excited|frustrated|worried|proud|relieved|nervous|stressed|upset)\b"
        r".*\b(about|with|by)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(i'm|i am|we're|we are|i feel|we feel)\b.*\b(excited|frustrated|worried|"
        r"proud|relieved|nervous|stressed|upset)\b",
        re.IGNORECASE,
    ),
)

_MOVE_TYPE_SCORES = {
    "life_update": 0.95,
    "resumption": 0.90,
    "sharing": 0.80,
    "checkin": 0.75,
    "musing": 0.70,
    "opinion": 0.50,
    "asking_personal": 0.45,
    "asking_general": 0.15,
    "commanding": 0.10,
}
_AFFECT_MARKERS = {
    "amazing",
    "awful",
    "excited",
    "frustrated",
    "glad",
    "great",
    "happy",
    "love",
    "nervous",
    "overwhelmed",
    "proud",
    "sad",
    "stressed",
    "thrilled",
    "upset",
    "worried",
}
_INTENSIFIERS = {"really", "so", "super", "very", "extremely", "totally"}
_GREETING_PATTERN = re.compile(r"^(hi|hello|hey)\b", re.IGNORECASE)
_DISCOURSE_SHIFT_MARKERS = {
    "anyway": 0.75,
    "by the way": 0.90,
    "btw": 0.90,
    "oh btw": 0.95,
    "speaking of": 0.80,
    "that reminds me": 0.80,
    "on a different note": 0.95,
    "random but": 0.70,
    "actually": 0.35,
    "also": 0.25,
}
_CONTINUATION_PENALTIES = {"yeah so", "right", "exactly", "because", "and then"}


@dataclass
class SignalResult:
    """Output from a single signal detector."""

    name: str
    score: float
    trigger_kind: str | None = None
    referents: list[str] = field(default_factory=list)
    query_hint: str | None = None
    family: str = "pragmatic"


@dataclass
class SignalReport:
    """Aggregated output from all signal families."""

    pragmatic_score: float = 0.0
    structural_score: float = 0.0
    shift_score: float = 0.0
    impoverishment_score: float = 0.0
    linguistic_score: float = 0.0
    signals: list[SignalResult] = field(default_factory=list)
    dominant_family: str | None = None
    dominant_trigger_kind: str | None = None
    all_referents: list[str] = field(default_factory=list)
    best_query_hint: str | None = None
    dampening_factor: float = 1.0

    def to_scores_dict(self) -> dict[str, float]:
        """Return signal scores for MemoryNeed.signal_scores."""
        scores: dict[str, float] = {}
        if self.pragmatic_score > 0:
            scores["pragmatic"] = self.pragmatic_score
        if self.structural_score > 0:
            scores["structural"] = self.structural_score
        if self.shift_score > 0:
            scores["shift"] = self.shift_score
        if self.impoverishment_score > 0:
            scores["impoverishment"] = self.impoverishment_score
        if self.linguistic_score > 0:
            scores["linguistic"] = self.linguistic_score
        if self.dampening_factor < 1.0:
            scores["dampening_factor"] = self.dampening_factor
        for sig in self.signals:
            if sig.score > 0:
                scores[sig.name] = sig.score
        return scores


def extract_signals(
    text: str,
    lowered: str,
    *,
    recent_turns: list[str] | None = None,
    session_entity_names: list[str] | None = None,
    conv_context=None,
    cfg=None,
) -> SignalReport:
    """Extract pragmatic, structural, shift, and impoverishment signals."""
    recent_turns = recent_turns or []
    session_entity_names = session_entity_names or []
    features = _extract_turn_features(text, lowered)

    pragmatic_results = _extract_pragmatic_signals(
        text,
        lowered,
        recent_turns=recent_turns,
        session_entity_names=session_entity_names,
    )
    structural_enabled = bool(
        cfg is not None and getattr(cfg, "recall_need_structural_enabled", False)
    )
    shift_enabled = bool(cfg is not None and getattr(cfg, "recall_need_shift_enabled", False))
    impoverishment_enabled = bool(
        cfg is not None and getattr(cfg, "recall_need_impoverishment_enabled", False)
    )

    structural_results = (
        _extract_structural_signals(
            text,
            lowered,
            recent_turns=recent_turns,
            session_entity_names=session_entity_names,
            pragmatic_results=pragmatic_results,
        )
        if structural_enabled
        else []
    )
    shift_results = (
        _extract_shift_signals(
            text,
            lowered,
            features,
            conv_context=conv_context,
        )
        if shift_enabled
        else []
    )
    impoverishment_results = (
        _extract_impoverishment_signals(
            text,
            lowered,
            features,
            pragmatic_results=pragmatic_results,
            structural_results=structural_results,
        )
        if impoverishment_enabled
        else []
    )

    pragmatic_score = _noisy_or([result.score for result in pragmatic_results])
    structural_score = _noisy_or([result.score for result in structural_results])
    shift_score = _noisy_or([result.score for result in shift_results])
    impoverishment_score = _noisy_or([result.score for result in impoverishment_results])

    shift_live = bool(conv_context is not None and shift_enabled)
    impoverishment_live = impoverishment_enabled
    structural_live = structural_enabled
    shift_shadow_only = bool(
        cfg is not None and getattr(cfg, "recall_need_shift_shadow_only", True)
    )
    impoverishment_shadow_only = bool(
        cfg is not None and getattr(cfg, "recall_need_impoverishment_shadow_only", True)
    )

    live_scores = [pragmatic_score]
    if structural_live:
        live_scores.append(structural_score)
    if shift_live and not shift_shadow_only:
        live_scores.append(shift_score)
    if impoverishment_live and not impoverishment_shadow_only:
        live_scores.append(impoverishment_score)
    base_linguistic_score = _noisy_or([score for score in live_scores if score > 0])

    referents = _dedupe(
        [
            referent
            for result in pragmatic_results + structural_results
            for referent in result.referents
        ]
    )
    dampening_factor = _compute_dampening(
        lowered,
        features,
        referents=referents,
        pragmatic_results=pragmatic_results,
        structural_results=structural_results,
    )
    linguistic_score = round(base_linguistic_score * dampening_factor, 4)

    all_results = pragmatic_results + structural_results + shift_results + impoverishment_results
    if not all_results:
        return SignalReport(dampening_factor=round(dampening_factor, 4))

    family_scores = {
        "pragmatic": pragmatic_score,
        "structural": structural_score,
        "shift": shift_score,
        "impoverishment": impoverishment_score,
    }
    dominant_family = max(
        (family for family, score in family_scores.items() if score > 0),
        key=lambda family: family_scores[family],
        default=None,
    )
    dominant_result_pool = (
        [result for result in all_results if result.family == dominant_family]
        if dominant_family is not None
        else all_results
    )
    dominant_result = max(dominant_result_pool or all_results, key=lambda result: result.score)
    best_query_hint = next(
        (
            result.query_hint
            for result in sorted(all_results, key=lambda item: item.score, reverse=True)
            if result.query_hint
        ),
        None,
    )
    return SignalReport(
        pragmatic_score=round(pragmatic_score, 4),
        structural_score=round(structural_score, 4),
        shift_score=round(shift_score, 4),
        impoverishment_score=round(impoverishment_score, 4),
        linguistic_score=linguistic_score,
        signals=all_results,
        dominant_family=dominant_family or dominant_result.family,
        dominant_trigger_kind=dominant_result.trigger_kind,
        all_referents=referents,
        best_query_hint=best_query_hint,
        dampening_factor=round(dampening_factor, 4),
    )


def _extract_pragmatic_signals(
    text: str,
    lowered: str,
    *,
    recent_turns: list[str],
    session_entity_names: list[str],
) -> list[SignalResult]:
    results: list[SignalResult] = []
    results.extend(_detect_possessive_relational(text))
    results.extend(_detect_bare_names(text, lowered, session_entity_names))
    results.extend(_detect_cross_session_anaphora(text, lowered, recent_turns))
    results.extend(_detect_continuation_markers(lowered, recent_turns))
    results.extend(_detect_hedged_aside(text, lowered, results))
    return results


def _extract_structural_signals(
    text: str,
    lowered: str,
    *,
    recent_turns: list[str],
    session_entity_names: list[str],
    pragmatic_results: list[SignalResult],
) -> list[SignalResult]:
    results: list[SignalResult] = []
    results.extend(_detect_callback(text, lowered, recent_turns))
    results.extend(_detect_memory_gap(lowered))
    results.extend(_detect_correction(lowered))
    results.extend(_detect_life_update(text, lowered, pragmatic_results, session_entity_names))
    results.extend(_detect_identity_claim(lowered))
    results.extend(_detect_status_check(text, lowered, session_entity_names))
    results.extend(_detect_phase5_structural_patterns(text, lowered, pragmatic_results))
    return results


def _extract_shift_signals(
    text: str,
    lowered: str,
    features: dict[str, object],
    *,
    conv_context,
) -> list[SignalResult]:
    if conv_context is None:
        return []
    entries = conv_context.get_recent_turn_entries(3, live_only=True)
    if not entries:
        return []
    if _GREETING_PATTERN.match(lowered):
        return []
    if _is_generic_command(lowered):
        return []

    now_ts = entries[-1].timestamp
    if now_ts and (_feature_float(features, "timestamp") - now_ts) > 1800.0:
        return []

    recent_features = []
    for entry in entries:
        if entry.analyzer_features is None:
            entry.analyzer_features = _extract_turn_features(entry.text, entry.text.lower())
        recent_features.append(entry.analyzer_features)
    if not recent_features:
        return []

    lexical = _lexical_shift_score(features, recent_features)
    register = _register_shift_score(features, recent_features)
    discourse = _discourse_shift_score(lowered)
    pronoun = _pronoun_shift_score(features, recent_features)
    structural = _structural_shift_score(features, recent_features)
    composite = min(
        1.0,
        (0.30 * lexical)
        + (0.25 * register)
        + (0.20 * discourse)
        + (0.15 * pronoun)
        + (0.10 * structural),
    )
    if _feature_float(features, "personal_ratio") > _window_average(
        recent_features,
        "personal_ratio",
    ):
        composite = min(1.0, composite * 1.2)
    if any(_feature_float(item, "code_ratio") >= 0.30 for item in recent_features[-1:]):
        composite *= 0.6
    if composite < 0.12:
        return []
    return [
        SignalResult("S1_lexical_shift", lexical, "shift_lexical", family="shift"),
        SignalResult("S2_register_shift", register, "shift_register", family="shift"),
        SignalResult("S3_discourse_shift", discourse, "shift_discourse", family="shift"),
        SignalResult("S4_pronoun_shift", pronoun, "shift_pronoun", family="shift"),
        SignalResult("S5_structural_shift", structural, "shift_structural", family="shift"),
        SignalResult(
            "S_shift",
            round(composite, 4),
            "shift_transition",
            referents=_dedupe(_feature_strings(features, "proper_names")),
            query_hint=next(iter(_feature_strings(features, "proper_names")), None),
            family="shift",
        ),
    ]


def _extract_impoverishment_signals(
    text: str,
    lowered: str,
    features: dict[str, object],
    *,
    pragmatic_results: list[SignalResult],
    structural_results: list[SignalResult],
) -> list[SignalResult]:
    if _GREETING_PATTERN.match(lowered) and len(lowered.split()) <= 3:
        return []
    if _is_generic_command(lowered) and not _feature_bool(features, "has_personal_anchor"):
        return []
    move_type, move_score = _detect_move_type(lowered, features, structural_results)
    affect_score = _affect_with_personal_stakes(lowered, features)
    template_risk = _template_response_risk(features, pragmatic_results, structural_results)
    composite = _noisy_or([move_score, affect_score, template_risk])
    if composite < 0.12:
        return []
    return [
        SignalResult("I1_move_type", move_score, f"move_{move_type}", family="impoverishment"),
        SignalResult("I2_affect", affect_score, "affect_personal", family="impoverishment"),
        SignalResult(
            "I3_template_risk",
            template_risk,
            "template_response_risk",
            family="impoverishment",
        ),
        SignalResult(
            "I_impoverishment",
            round(composite, 4),
            "impoverishment_risk",
            referents=_dedupe(_feature_strings(features, "proper_names")),
            query_hint=next(iter(_feature_strings(features, "proper_names")), None),
            family="impoverishment",
        ),
    ]


def _detect_possessive_relational(text: str) -> list[SignalResult]:
    results: list[SignalResult] = []
    for match in _POSSESSIVE_PATTERN.finditer(text):
        noun = match.group(2).lower()
        if noun in _TECHNICAL_POSSESSIVES or noun in _NON_RELATIONAL_POSSESSIVES:
            continue
        if noun not in _RELATIONAL_NOUNS:
            continue
        proper_name = match.group(3)
        score = _RELATIONAL_WEIGHTS[noun]
        query_hint = None
        if proper_name:
            score = max(score, 0.50)
            query_hint = proper_name
        results.append(
            SignalResult(
                name="P1_possessive_relational",
                score=score,
                trigger_kind="possessive_relational",
                referents=_dedupe([noun, proper_name] if proper_name else [noun]),
                query_hint=query_hint,
            )
        )
    return results


def _detect_bare_names(
    text: str,
    lowered: str,
    session_entity_names: list[str],
) -> list[SignalResult]:
    results: list[SignalResult] = []
    session_names = {name.lower() for name in session_entity_names}
    for match in _NAME_PATTERN.finditer(text):
        candidate = match.group(0).strip()
        candidate_lower = candidate.lower()
        if candidate_lower in session_names:
            continue
        if candidate_lower in _COMMUNAL_GROUND or candidate_lower in _COMMON_CAPITALIZED_WORDS:
            continue
        if _is_introduced_name(candidate, lowered):
            continue
        if match.start() == 0 and not _is_plausible_sentence_initial_name(text, match):
            continue
        score = 0.30
        if candidate_lower in _AMBIGUOUS_NAMES:
            if not _has_name_secondary_signal(text, lowered, candidate):
                continue
            score = 0.15
        results.append(
            SignalResult(
                name="P2_bare_name",
                score=score,
                trigger_kind="bare_name",
                referents=[candidate],
                query_hint=candidate,
            )
        )
    return results


def _detect_cross_session_anaphora(
    text: str,
    lowered: str,
    recent_turns: list[str],
) -> list[SignalResult]:
    if not _ANAPHORA_PATTERN.search(lowered):
        return []
    if any(pattern.search(lowered) for pattern in _EXPLETIVE_IT_PATTERNS):
        filtered = [
            pronoun.lower()
            for pronoun in _ANAPHORA_PATTERN.findall(text)
            if pronoun.lower() != "it"
        ]
    else:
        filtered = [pronoun.lower() for pronoun in _ANAPHORA_PATTERN.findall(text)]
    if not filtered:
        return []
    results: list[SignalResult] = []
    for pronoun in _dedupe(filtered):
        score = 0.35
        if recent_turns and _recent_context_has_antecedent(pronoun, recent_turns):
            score = 0.18
        results.append(
            SignalResult(
                name="P3_cross_session_anaphora",
                score=score,
                trigger_kind="cross_session_anaphora",
                referents=[pronoun],
            )
        )
    return results


def _detect_hedged_aside(
    text: str,
    lowered: str,
    prior_results: list[SignalResult],
) -> list[SignalResult]:
    marker = next(
        (candidate for candidate in _HEDGE_MARKERS if lowered.startswith(candidate)), None
    )
    if marker is None:
        return []
    referents = _dedupe([referent for result in prior_results for referent in result.referents])
    if not referents:
        referents = _extract_personal_terms(lowered)
    if not referents:
        return []
    score = 0.22
    tail = lowered[len(marker) :].strip(" ,.!?")
    if tail and len(tail.split()) < 8:
        score = min(0.29, score * 1.3)
    query_hint = next((result.query_hint for result in prior_results if result.query_hint), None)
    if query_hint is None and referents:
        query_hint = " ".join(referents[:2])[:200]
    return [
        SignalResult(
            name="P5_hedged_aside",
            score=score,
            trigger_kind="hedged_aside",
            referents=referents,
            query_hint=query_hint,
        )
    ]


def _detect_continuation_markers(
    lowered: str,
    recent_turns: list[str],
) -> list[SignalResult]:
    results: list[SignalResult] = []
    if lowered.startswith("try again"):
        return results
    for marker, pattern in _CONTINUATION_MARKERS.items():
        if not pattern.search(lowered):
            continue
        score = 0.18 if not recent_turns else 0.09
        results.append(
            SignalResult(
                name="P6_continuation_marker",
                score=score,
                trigger_kind=f"continuation_{marker}",
                referents=[marker.replace("_", " ")],
            )
        )
    return results


def _detect_callback(text: str, lowered: str, recent_turns: list[str]) -> list[SignalResult]:
    for pattern in _CALLBACK_PATTERNS:
        if not pattern.search(text):
            continue
        score = 0.70 if recent_turns else 0.58
        return [
            SignalResult(
                "T1_callback",
                score,
                "callback",
                referents=_extract_named_referents(text),
                query_hint=_first_named_referent(text),
                family="structural",
            )
        ]
    return []


def _detect_memory_gap(lowered: str) -> list[SignalResult]:
    if any(pattern.search(lowered) for pattern in _MEMORY_GAP_PATTERNS):
        return [
            SignalResult(
                "T2_memory_gap",
                0.90,
                "memory_gap",
                family="structural",
            )
        ]
    return []


def _detect_correction(lowered: str) -> list[SignalResult]:
    if any(pattern.search(lowered) for pattern in _CORRECTION_PATTERNS):
        return [
            SignalResult(
                "T3_correction",
                0.84,
                "correction",
                family="structural",
            )
        ]
    return []


def _detect_life_update(
    text: str,
    lowered: str,
    pragmatic_results: list[SignalResult],
    session_entity_names: list[str],
) -> list[SignalResult]:
    if not any(pattern.search(text) for pattern in _LIFE_UPDATE_PATTERNS):
        return []
    referents = _dedupe([referent for result in pragmatic_results for referent in result.referents])
    named = _extract_named_referents(text)
    if (
        not referents
        and not named
        and not session_entity_names
        and not re.search(r"\b(i|we|my|our)\b", lowered)
    ):
        return []
    score = 0.42
    if named or referents:
        score = 0.52
    query_hint = named[0] if named else (session_entity_names[0] if session_entity_names else None)
    return [
        SignalResult(
            "T4_life_update",
            score,
            "life_update",
            referents=_dedupe(referents + named),
            query_hint=query_hint,
            family="structural",
        )
    ]


def _detect_identity_claim(lowered: str) -> list[SignalResult]:
    if any(pattern.search(lowered) for pattern in _IDENTITY_PATTERNS):
        return [SignalResult("T5_identity_claim", 0.82, "identity_claim", family="structural")]
    return []


def _detect_status_check(
    text: str,
    lowered: str,
    session_entity_names: list[str],
) -> list[SignalResult]:
    if not any(pattern.search(lowered) for pattern in _STATUS_PATTERNS):
        return []
    if (
        "how's it going" in lowered
        and not _extract_named_referents(text)
        and not session_entity_names
    ):
        return []
    score = 0.68
    if _extract_named_referents(text) or _has_project_terms(lowered):
        score = 0.78
    return [
        SignalResult(
            "T6_status_check",
            score,
            "status_check",
            referents=_extract_named_referents(text),
            query_hint=_first_named_referent(text)
            or (session_entity_names[0] if session_entity_names else None),
            family="structural",
        )
    ]


def _detect_phase5_structural_patterns(
    text: str,
    lowered: str,
    pragmatic_results: list[SignalResult],
) -> list[SignalResult]:
    results: list[SignalResult] = []
    named = _extract_named_referents(text)
    referents = _dedupe(
        [referent for result in pragmatic_results for referent in result.referents] + named
    )
    query_hint = named[0] if named else None
    if any(pattern.search(text) for pattern in _CONTINUATION_STRUCTURAL_PATTERNS):
        results.append(
            SignalResult(
                "T7_continuation",
                0.52,
                "continuation",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if any(pattern.search(lowered) for pattern in _COMPARISON_PATTERNS):
        results.append(
            SignalResult(
                "T8_comparison",
                0.44,
                "comparison",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if any(pattern.search(lowered) for pattern in _PREFERENCE_PATTERNS):
        results.append(
            SignalResult(
                "T9_implicit_preference",
                0.55,
                "implicit_preference",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if any(pattern.search(lowered) for pattern in _PLANNING_PATTERNS):
        results.append(
            SignalResult(
                "T10_planning",
                0.50,
                "planning",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if any(pattern.search(text) for pattern in _SOCIAL_UPDATE_PATTERNS):
        results.append(
            SignalResult(
                "T11_social_graph_update",
                0.56,
                "social_graph_update",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if any(pattern.search(lowered) for pattern in _CAUSAL_PATTERNS):
        results.append(
            SignalResult(
                "T12_causal_context",
                0.42,
                "causal_context",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if any(pattern.search(lowered) for pattern in _RECURRING_PROBLEM_PATTERNS):
        results.append(
            SignalResult(
                "T13_recurring_problem",
                0.48,
                "recurring_problem",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if any(pattern.search(lowered) for pattern in _MILESTONE_PATTERNS):
        results.append(
            SignalResult(
                "T14_milestone",
                0.58,
                "milestone",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if any(pattern.search(lowered) for pattern in _INTRODUCTION_PATTERNS):
        results.append(
            SignalResult(
                "T15_introduction",
                0.28,
                "introduction",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if sum(1 for pattern in _TEMPORAL_NARRATIVE_PATTERNS if pattern.search(lowered)) >= 2:
        results.append(
            SignalResult(
                "T16_temporal_narrative",
                0.38,
                "temporal_narrative",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if named and any(pattern.search(text) for pattern in _DELEGATION_PATTERNS):
        results.append(
            SignalResult(
                "T17_delegation",
                0.46,
                "delegation",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    if any(pattern.search(lowered) for pattern in _EMOTIONAL_ANCHOR_PATTERNS) and (
        referents or named or re.search(r"\b(i|my|we|our)\b", lowered)
    ):
        score = 0.43
        if any(marker in lowered for marker in ("again", "still", "finally")):
            score = 0.52
        results.append(
            SignalResult(
                "T18_emotional_anchor",
                score,
                "emotional_anchor",
                referents=referents,
                query_hint=query_hint,
                family="structural",
            )
        )
    return results


def _extract_turn_features(text: str, lowered: str) -> dict[str, object]:
    words = re.findall(r"[A-Za-z']+", lowered)
    total_words = max(1, len(words))
    proper_names = set(_extract_named_referents(text))
    personal_count = sum(1 for word in words if word in _PERSONAL_TERMS)
    tech_count = sum(1 for word in words if word in _TECHNICAL_TERMS)
    personal_pronouns = sum(
        1 for word in words if word in {"i", "my", "me", "he", "she", "his", "her"}
    )
    return {
        "timestamp": time.time(),
        "word_count": len(words),
        "proper_names": proper_names,
        "personal_ratio": (personal_count + personal_pronouns) / total_words,
        "technical_ratio": tech_count / total_words,
        "question_ratio": text.count("?") / max(1, len(text)),
        "punctuation_ratio": len(re.findall(r"[?!,;:]", text)) / max(1, len(text)),
        "code_ratio": len(re.findall(r"[{}();`]|=>", text)) / max(1, len(text)),
        "first_person_ratio": personal_pronouns / total_words,
        "is_question": "?" in text
        or bool(re.match(r"^(who|what|when|where|why|how|did|does|do|is|are|can)\b", lowered)),
        "has_project_terms": _has_project_terms(lowered),
        "has_personal_anchor": bool(proper_names or personal_count),
    }


def _lexical_shift_score(
    features: dict[str, object], recent_features: list[dict[str, object]]
) -> float:
    personal_delta = abs(
        _feature_float(features, "personal_ratio")
        - _window_average(recent_features, "personal_ratio")
    )
    technical_delta = abs(
        _feature_float(features, "technical_ratio")
        - _window_average(recent_features, "technical_ratio")
    )
    return min(1.0, (personal_delta + technical_delta) * 1.6)


def _register_shift_score(
    features: dict[str, object], recent_features: list[dict[str, object]]
) -> float:
    current_register = _feature_float(features, "personal_ratio") - _feature_float(
        features,
        "technical_ratio",
    )
    window_register = _window_average(recent_features, "personal_ratio") - _window_average(
        recent_features,
        "technical_ratio",
    )
    return min(1.0, abs(current_register - window_register) * 1.5)


def _discourse_shift_score(lowered: str) -> float:
    score = 0.0
    for marker, value in _DISCOURSE_SHIFT_MARKERS.items():
        if lowered.startswith(marker):
            score = max(score, value * 1.2)
        elif marker in lowered:
            score = max(score, value)
    for marker in _CONTINUATION_PENALTIES:
        if lowered.startswith(marker):
            score = max(0.0, score - 0.2)
    return min(1.0, score)


def _pronoun_shift_score(
    features: dict[str, object], recent_features: list[dict[str, object]]
) -> float:
    delta = abs(
        _feature_float(features, "first_person_ratio")
        - _window_average(recent_features, "first_person_ratio")
    )
    proper_name_bonus = 0.10 if _feature_strings(features, "proper_names") else 0.0
    return min(1.0, (delta * 2.5) + proper_name_bonus)


def _structural_shift_score(
    features: dict[str, object], recent_features: list[dict[str, object]]
) -> float:
    word_delta = (
        abs(_feature_float(features, "word_count") - _window_average(recent_features, "word_count"))
        / 20.0
    )
    question_delta = (
        abs(
            _feature_float(features, "question_ratio")
            - _window_average(recent_features, "question_ratio")
        )
        * 8.0
    )
    punct_delta = (
        abs(
            _feature_float(features, "punctuation_ratio")
            - _window_average(recent_features, "punctuation_ratio")
        )
        * 8.0
    )
    return min(1.0, word_delta + question_delta + punct_delta)


def _detect_move_type(
    lowered: str,
    features: dict[str, object],
    structural_results: list[SignalResult],
) -> tuple[str, float]:
    triggers = {result.trigger_kind for result in structural_results if result.trigger_kind}
    if "life_update" in triggers or "milestone" in triggers:
        return ("life_update", _MOVE_TYPE_SCORES["life_update"])
    if "callback" in triggers or lowered.startswith("so about") or lowered.startswith("back to"):
        return ("resumption", _MOVE_TYPE_SCORES["resumption"])
    if _GREETING_PATTERN.match(lowered):
        return ("checkin", 0.0)
    if bool(features["is_question"]) and bool(features["has_personal_anchor"]):
        return ("asking_personal", _MOVE_TYPE_SCORES["asking_personal"])
    if bool(features["is_question"]):
        return ("asking_general", _MOVE_TYPE_SCORES["asking_general"])
    if re.match(r"^(please|can you|write|show|list|implement|fix)\b", lowered):
        return ("commanding", _MOVE_TYPE_SCORES["commanding"])
    if re.search(r"\b(i think|i feel|i prefer)\b", lowered):
        return ("opinion", _MOVE_TYPE_SCORES["opinion"])
    if re.search(r"\b(been thinking about|wondering if|maybe)\b", lowered):
        return ("musing", _MOVE_TYPE_SCORES["musing"])
    return ("sharing", _MOVE_TYPE_SCORES["sharing"])


def _affect_with_personal_stakes(lowered: str, features: dict[str, object]) -> float:
    affect_hits = sum(1 for marker in _AFFECT_MARKERS if marker in lowered)
    affect_hits += sum(1 for marker in _INTENSIFIERS if f" {marker} " in f" {lowered} ")
    if "!" in lowered:
        affect_hits += 1
    if affect_hits == 0:
        return 0.0
    affect_intensity = min(1.0, affect_hits / 3.0)
    anchor_strength = 1.0 if bool(features["has_personal_anchor"]) else 0.25
    return round(affect_intensity * anchor_strength, 4)


def _template_response_risk(
    features: dict[str, object],
    pragmatic_results: list[SignalResult],
    structural_results: list[SignalResult],
) -> float:
    if structural_results:
        dominant = max(result.score for result in structural_results)
        return min(0.65, 0.25 + (dominant * 0.4))
    if pragmatic_results and bool(features["has_personal_anchor"]):
        return 0.42
    if pragmatic_results:
        return 0.28
    return 0.0


def _compute_dampening(
    lowered: str,
    features: dict[str, object],
    *,
    referents: list[str],
    pragmatic_results: list[SignalResult],
    structural_results: list[SignalResult],
) -> float:
    factor = 1.0
    has_temporal = any(
        marker in lowered
        for marker in ("after", "before", "later", "earlier", "yesterday", "today", "tomorrow")
    )
    has_emotion = any(marker in lowered for marker in _AFFECT_MARKERS)
    imperative = bool(re.match(r"^(please|can you|should|remember to|after .*?,? )", lowered))
    if has_temporal and not referents and not pragmatic_results and not structural_results:
        factor *= 0.6
    if has_emotion and not bool(features["has_personal_anchor"]):
        factor *= 0.7
    if imperative and has_temporal:
        factor *= 0.5
    if lowered.startswith("my "):
        next_word = lowered.split()[1] if len(lowered.split()) > 1 else ""
        if next_word in _TECHNICAL_POSSESSIVES:
            factor *= 0.6
    if len(referents) == 1 and referents[0].lower() in _AMBIGUOUS_NAMES:
        factor *= 0.65
    return max(0.35, min(factor, 1.0))


def _is_introduced_name(candidate: str, lowered: str) -> bool:
    candidate_lower = re.escape(candidate.lower())
    if any(f"{prefix} {candidate.lower()}" in lowered for prefix in _INTRODUCTION_PREFIXES):
        return True
    return any(
        re.search(pattern, lowered) is not None
        for pattern in (
            rf"\bnamed\s+{candidate_lower}\b",
            rf"\bcalled\s+{candidate_lower}\b",
            rf"\b{candidate_lower},?\s+who\s+(?:is|was)\b",
        )
    )


def _is_plausible_sentence_initial_name(text: str, match: re.Match[str]) -> bool:
    candidate_lower = match.group(0).lower()
    if candidate_lower in _COMMON_CAPITALIZED_WORDS:
        return False
    suffix = text[match.end() :].lstrip()
    if suffix.startswith("'s"):
        return True
    next_word_match = re.match(r"([A-Za-z]+)", suffix)
    if next_word_match is None:
        return False
    return next_word_match.group(1).lower() in _PERSON_FOLLOWING_VERBS


def _has_name_secondary_signal(text: str, lowered: str, candidate: str) -> bool:
    candidate_lower = re.escape(candidate.lower())
    return any(
        re.search(pattern, lowered) is not None
        for pattern in (
            rf"\bmy\s+\w+\s+{candidate_lower}\b",
            rf"\b{candidate_lower}\s+(?:said|says|thinks|texted|emailed|scored|won|joined)\b",
        )
    )


def _recent_context_has_antecedent(pronoun: str, recent_turns: list[str]) -> bool:
    recent_text = " ".join(recent_turns)
    if pronoun in {"she", "he", "they", "them"}:
        if _NAME_PATTERN.search(recent_text):
            return True
        return any(
            noun in recent_text.lower() for noun in ("son", "daughter", "friend", "coworker")
        )
    if pronoun in {"it", "that", "this"}:
        lowered_recent = recent_text.lower()
        return any(
            token in lowered_recent for token in ("project", "migration", "bug", "task", "issue")
        )
    return False


def _extract_personal_terms(lowered: str) -> list[str]:
    referents: list[str] = []
    for noun in _RELATIONAL_NOUNS:
        if re.search(rf"\b{re.escape(noun)}\b", lowered):
            referents.append(noun)
    if "kid stuff" in lowered and "kid" not in referents:
        referents.append("kid")
    return _dedupe(referents)


def _extract_named_referents(text: str) -> list[str]:
    values: list[str] = []
    for match in _NAME_PATTERN.finditer(text):
        candidate = match.group(0).strip()
        key = candidate.lower()
        if key in _COMMON_CAPITALIZED_WORDS or key in _COMMUNAL_GROUND:
            continue
        values.append(candidate)
    return _dedupe(values)


def _first_named_referent(text: str) -> str | None:
    named = _extract_named_referents(text)
    return named[0] if named else None


def _has_project_terms(lowered: str) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in _PROJECT_TERMS)


def _window_average(recent_features: list[dict[str, object]], key: str) -> float:
    if not recent_features:
        return 0.0
    values = [_feature_float(item, key) for item in recent_features]
    return sum(values) / len(values)


def _is_generic_command(lowered: str) -> bool:
    return bool(re.match(r"^(can you|could you|please|write|show|list|implement|fix)\b", lowered))


def _noisy_or(scores: list[float]) -> float:
    if not scores:
        return 0.0
    result = 1.0
    for score in scores:
        result *= 1.0 - max(0.0, min(score, 1.0))
    return 1.0 - result


def _feature_float(features: dict[str, object], key: str) -> float:
    value = features.get(key, 0.0)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _feature_bool(features: dict[str, object], key: str) -> bool:
    value = features.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return False


def _feature_strings(features: dict[str, object], key: str) -> list[str]:
    value = features.get(key, [])
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence):
        return []
    return [item for item in value if isinstance(item, str)]


def _dedupe(values: Sequence[str | None]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value:
            continue
        key = value.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped
