"""Identifier-aware lexical analysis for entity deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class NameRegime(str, Enum):
    """Lexical regime for an entity name."""

    NATURAL_LANGUAGE = "natural_language"
    IDENTIFIER = "identifier"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class IdentifierForm:
    """Normalized lexical analysis of an entity name."""

    regime: NameRegime
    normalized: str
    comparison_text: str
    canonical_code: str | None
    alpha_chunks: tuple[str, ...]
    digit_chunks: tuple[str, ...]
    code_chunks: tuple[str, ...]
    has_identifier_label: bool = False


@dataclass(frozen=True)
class DedupPolicyDecision:
    """Shared policy decision for a pair of entity names."""

    allowed: bool
    reason: str
    left: IdentifierForm
    right: IdentifierForm
    exact_identifier_match: bool = False


IDENTIFIER_ENTITY_TYPE = "Identifier"
_COERCIBLE_IDENTIFIER_TYPES = frozenset(
    {"Other", "Thing", "Technology", "Software", "Concept"}
)

_NUMERONYM_RE = re.compile(r"^[a-z]\d+[a-z]$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_ALNUM_RE = re.compile(r"[a-z]+|\d+")
_LEADING_IDENTIFIER_LABEL_RE = re.compile(
    r"^(?:(?:sku|part(?:\s*(?:number|no))?|part\s*#|p\s*/\s*n|serial(?:\s*number)?|s\s*/\s*n)"
    r"\s*[:#-]?\s*)+",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    return _WHITESPACE_RE.sub(" ", lowered)


def _strip_leading_identifier_labels(text: str) -> tuple[str, bool]:
    stripped = _LEADING_IDENTIFIER_LABEL_RE.sub("", text).strip()
    return (stripped or text), stripped != text


def _space_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in text.split():
        clean = re.sub(r"[^a-z0-9]", "", raw_token.lower())
        if clean:
            tokens.append(clean)
    return tokens


def _is_year_like(token: str) -> bool:
    if not token.isdigit() or len(token) != 4:
        return False
    year = int(token)
    return 1900 <= year <= 2100


def _token_kind(token: str) -> str:
    if token.isdigit():
        return "strong_num" if len(token) >= 4 else "number"

    has_alpha = any(ch.isalpha() for ch in token)
    has_digit = any(ch.isdigit() for ch in token)
    if has_alpha and has_digit:
        if _NUMERONYM_RE.fullmatch(token):
            return "numeronym"
        digit_chunks = re.findall(r"\d+", token)
        max_digit_len = max((len(chunk) for chunk in digit_chunks), default=0)
        if max_digit_len >= 3 or len(token) <= 6:
            return "strong_mixed"
        return "mixed"

    if token.isalpha():
        return "short_alpha" if len(token) <= 3 else "long_alpha"

    return "other"


def _marked_code_tokens(tokens: list[str], kinds: list[str]) -> list[bool]:
    marked = [False] * len(tokens)
    strong_indices = [
        idx for idx, kind in enumerate(kinds) if kind in {"strong_num", "strong_mixed"}
    ]
    for idx in strong_indices:
        marked[idx] = True

        left = idx - 1
        while left >= 0 and kinds[left] == "short_alpha":
            marked[left] = True
            left -= 1

        right = idx + 1
        while right < len(tokens) and kinds[right] == "short_alpha" and len(tokens[right]) <= 2:
            marked[right] = True
            right += 1

    return marked


def _is_year_context(tokens: list[str], kinds: list[str], marked: list[bool]) -> bool:
    strong_marked = [
        tokens[idx] for idx, kind in enumerate(kinds) if marked[idx] and kind == "strong_num"
    ]
    if len(strong_marked) != 1 or not _is_year_like(strong_marked[0]):
        return False

    non_code_kinds = [kind for idx, kind in enumerate(kinds) if not marked[idx]]
    if not non_code_kinds:
        return False
    return all(kind in {"long_alpha", "number"} for kind in non_code_kinds)


def analyze_name(name: str) -> IdentifierForm:
    """Analyze a name into a lexical regime and optional canonical code."""
    normalized = _normalize_text(name)
    comparison_text, has_identifier_label = _strip_leading_identifier_labels(normalized)
    tokens = _space_tokens(comparison_text)

    if not tokens:
        return IdentifierForm(
            regime=NameRegime.NATURAL_LANGUAGE,
            normalized=normalized,
            comparison_text=comparison_text,
            canonical_code=None,
            alpha_chunks=(),
            digit_chunks=(),
            code_chunks=(),
            has_identifier_label=has_identifier_label,
        )

    kinds = [_token_kind(token) for token in tokens]
    marked = _marked_code_tokens(tokens, kinds)
    code_tokens = [tokens[idx] for idx, is_marked in enumerate(marked) if is_marked]
    natural_tokens = [tokens[idx] for idx, is_marked in enumerate(marked) if not is_marked]

    if not code_tokens or _is_year_context(tokens, kinds, marked):
        return IdentifierForm(
            regime=NameRegime.NATURAL_LANGUAGE,
            normalized=normalized,
            comparison_text=comparison_text,
            canonical_code=None,
            alpha_chunks=(),
            digit_chunks=(),
            code_chunks=(),
            has_identifier_label=has_identifier_label,
        )

    code_text = "".join(code_tokens)
    code_chunks = tuple(_ALNUM_RE.findall(code_text))
    alpha_chunks = tuple(chunk for chunk in code_chunks if chunk.isalpha())
    digit_chunks = tuple(chunk for chunk in code_chunks if chunk.isdigit())

    regime = NameRegime.HYBRID
    if not natural_tokens:
        if len(code_tokens) == 1:
            regime = NameRegime.IDENTIFIER
        elif len(code_tokens) == 2:
            code_kinds = [kinds[idx] for idx, is_marked in enumerate(marked) if is_marked]
            if not (
                any(kind == "short_alpha" for kind in code_kinds)
                and any(kind in {"strong_num", "strong_mixed"} for kind in code_kinds)
            ):
                regime = NameRegime.IDENTIFIER
        else:
            regime = NameRegime.IDENTIFIER

    return IdentifierForm(
        regime=regime,
        normalized=normalized,
        comparison_text=comparison_text,
        canonical_code="".join(code_chunks) or None,
        alpha_chunks=alpha_chunks,
        digit_chunks=digit_chunks,
        code_chunks=code_chunks,
        has_identifier_label=has_identifier_label,
    )


def dedup_policy(name_a: str, name_b: str) -> DedupPolicyDecision:
    """Determine whether two names are eligible for fuzzy dedup."""
    left = analyze_name(name_a)
    right = analyze_name(name_b)

    if left.canonical_code and right.canonical_code:
        if left.canonical_code == right.canonical_code:
            reason = "identifier_exact_match"
            if NameRegime.HYBRID in {left.regime, right.regime}:
                reason = "hybrid_code_match"
            return DedupPolicyDecision(
                allowed=True,
                reason=reason,
                left=left,
                right=right,
                exact_identifier_match=True,
            )

        reason = "identifier_mismatch"
        if NameRegime.HYBRID in {left.regime, right.regime}:
            reason = "hybrid_code_mismatch"
        return DedupPolicyDecision(
            allowed=False,
            reason=reason,
            left=left,
            right=right,
        )

    if left.canonical_code or right.canonical_code:
        return DedupPolicyDecision(
            allowed=False,
            reason="code_anchor_missing",
            left=left,
            right=right,
        )

    return DedupPolicyDecision(
        allowed=True,
        reason="natural_language_fallback",
        left=left,
        right=right,
    )


def policy_aware_similarity(
    name_a: str,
    name_b: str,
    similarity_fn,
) -> tuple[DedupPolicyDecision, float]:
    """Apply identifier policy before delegating to fuzzy similarity."""
    decision = dedup_policy(name_a, name_b)
    if not decision.allowed:
        return decision, 0.0
    if decision.exact_identifier_match:
        return decision, 1.0
    return decision, similarity_fn(name_a, name_b)


def policy_features(decision: DedupPolicyDecision) -> dict:
    """Summarize a policy decision for traces and audit payloads."""
    return {
        "identifier_policy": decision.reason,
        "regime_a": decision.left.regime.value,
        "regime_b": decision.right.regime.value,
        "canonical_code_a": decision.left.canonical_code,
        "canonical_code_b": decision.right.canonical_code,
        "exact_identifier_match": decision.exact_identifier_match,
    }


def entity_identifier_facets(name: str) -> dict[str, str | bool | None]:
    """Compute stored identifier-related facets for an entity name."""
    form = analyze_name(name)
    return {
        "lexical_regime": form.regime.value,
        "canonical_identifier": form.canonical_code,
        "identifier_label": form.has_identifier_label,
    }


def normalize_extracted_entity_type(
    name: str,
    entity_type: str | None,
) -> tuple[str, IdentifierForm]:
    """Normalize extracted entity types so strict code-shaped names become Identifier."""
    normalized_type = (entity_type or "Other").strip() or "Other"
    form = analyze_name(name)

    if normalized_type == IDENTIFIER_ENTITY_TYPE:
        return normalized_type, form

    if form.regime == NameRegime.IDENTIFIER and normalized_type in _COERCIBLE_IDENTIFIER_TYPES:
        return IDENTIFIER_ENTITY_TYPE, form

    return normalized_type, form


def should_promote_entity_type_to_identifier(entity_type: str | None) -> bool:
    """Return True when an entity type is generic enough to upcast to Identifier."""
    normalized_type = (entity_type or "Other").strip() or "Other"
    return (
        normalized_type == IDENTIFIER_ENTITY_TYPE
        or normalized_type in _COERCIBLE_IDENTIFIER_TYPES
    )


def should_enqueue_identifier_review(
    decision: DedupPolicyDecision,
    raw_similarity: float,
    *,
    min_similarity: float = 0.8,
) -> bool:
    """Return True when a blocked identifier pair is review-worthy."""
    if decision.allowed or raw_similarity < min_similarity:
        return False

    if decision.reason in {"identifier_mismatch", "hybrid_code_mismatch"}:
        return bool(decision.left.canonical_code and decision.right.canonical_code)

    if decision.reason == "code_anchor_missing":
        return bool(decision.left.canonical_code or decision.right.canonical_code)

    return False
