"""Multi-signal deterministic merge scorer for entity deduplication.

Replaces the LLM judge with a weighted ensemble of name analysis,
embedding similarity, neighbor overlap, and summary overlap signals.
"""

from __future__ import annotations

import re

import numpy as np

from engram.entity_dedup_policy import dedup_policy, policy_aware_similarity, policy_features
from engram.extraction.resolver import compute_similarity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOPWORDS = {"the", "a", "an", "of", "for", "and", "in", "on", "to", "is", "at", "by"}

TECH_SUFFIXES = {
    ".js",
    ".py",
    ".rs",
    ".ts",
    " language",
    " framework",
    " library",
    " lang",
    " sdk",
}

# Canonical alias table -- known equivalences that string algorithms can't catch
KNOWN_ALIASES: list[tuple[frozenset[str], float]] = [
    (frozenset({"javascript", "js"}), 0.98),
    (frozenset({"typescript", "ts"}), 0.98),
    (frozenset({"kubernetes", "k8s"}), 0.98),
    (frozenset({"postgresql", "postgres", "psql"}), 0.98),
    (frozenset({"new york city", "nyc", "new york"}), 0.95),
    (frozenset({"artificial intelligence", "ai"}), 0.95),
    (frozenset({"machine learning", "ml"}), 0.95),
    (frozenset({"large language model", "llm"}), 0.95),
    (frozenset({"natural language processing", "nlp"}), 0.95),
    (frozenset({"amazon web services", "aws"}), 0.98),
    (frozenset({"google cloud platform", "gcp"}), 0.98),
    (frozenset({"continuous integration", "ci"}), 0.90),
    (frozenset({"continuous deployment", "cd"}), 0.90),
    (frozenset({"react.js", "reactjs", "react"}), 0.98),
    (frozenset({"node.js", "nodejs", "node"}), 0.95),
    (frozenset({"next.js", "nextjs", "next"}), 0.95),
    (frozenset({"vue.js", "vuejs", "vue"}), 0.95),
    (frozenset({"graphql", "graph ql"}), 0.95),
    (frozenset({"mongodb", "mongo"}), 0.95),
    (frozenset({"docker", "docker engine"}), 0.90),
    (frozenset({"github", "gh"}), 0.90),
    (frozenset({"visual studio code", "vscode", "vs code"}), 0.95),
    (frozenset({"api", "application programming interface"}), 0.90),
    (frozenset({"ui", "user interface"}), 0.90),
    (frozenset({"ux", "user experience"}), 0.90),
    (frozenset({"ci/cd", "ci cd", "cicd"}), 0.95),
    (frozenset({"os", "operating system"}), 0.85),
    (frozenset({"db", "database"}), 0.85),
]

COMPATIBLE_CROSS_TYPES = {
    frozenset({"Technology", "Software"}),
    frozenset({"Technology", "Project"}),
    frozenset({"Software", "Project"}),
    frozenset({"Concept", "Technology"}),
    frozenset({"Article", "CreativeWork"}),
    frozenset({"Goal", "Intention"}),
    frozenset({"Goal", "Preference"}),
    frozenset({"Emotion", "Concept"}),
    frozenset({"Habit", "Preference"}),
    frozenset({"Location", "Organization"}),
    frozenset({"HealthCondition", "Concept"}),
    frozenset({"Identifier", "Technology"}),
    frozenset({"Identifier", "Software"}),
    frozenset({"Identifier", "Concept"}),
}

_SUMMARY_STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "and",
    "but",
    "or",
    "not",
    "that",
    "this",
    "it",
    "its",
}

# ---------------------------------------------------------------------------
# Name analysis helpers
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _normalize(name: str) -> str:
    """Lowercase, replace underscores/hyphens with spaces, strip."""
    return name.lower().replace("_", " ").replace("-", " ").strip()


def _camel_case_split(name: str) -> list[str]:
    """Split CamelCase into words."""
    return _CAMEL_RE.sub(" ", name).split()


def acronym_match(a: str, b: str) -> float:
    """Check if the shorter name is an acronym of the longer name's words.

    Returns 0.95 if match, 0.0 if not.
    """
    na, nb = _normalize(a), _normalize(b)
    if len(na) == len(nb):
        return 0.0

    short, long_ = (na, nb) if len(na) < len(nb) else (nb, na)
    short_upper = short.replace(" ", "").upper()

    # Try splitting by spaces first
    words = long_.split()
    if len(words) >= 2:
        initials = "".join(w[0] for w in words if w).upper()
        if initials == short_upper:
            return 0.95

    # Try camelCase split on original longer name
    orig_long = a if len(na) >= len(nb) else b
    camel_words = _camel_case_split(orig_long)
    if len(camel_words) >= 2:
        initials = "".join(w[0] for w in camel_words if w).upper()
        if initials == short_upper:
            return 0.95

    return 0.0


def numeronym_match(a: str, b: str) -> float:
    """Check K8s/i18n-style numeronym pattern.

    Pattern: first letter + digit count + last letter of the full word.
    Returns 0.95 if match, 0.0 if not.
    """
    na, nb = _normalize(a), _normalize(b)
    if len(na) == len(nb):
        return 0.0

    short, long_ = (na, nb) if len(na) < len(nb) else (nb, na)

    # Numeronym format: letter + digits + letter, e.g. k8s, i18n
    m = re.match(r"^([a-z])(\d+)([a-z])$", short)
    if not m:
        return 0.0

    first, count_str, last = m.group(1), m.group(2), m.group(3)
    count = int(count_str)

    # Check: first letter matches, last letter matches, middle char count matches
    if long_ and long_[0] == first and long_[-1] == last and len(long_) - 2 == count:
        return 0.95

    return 0.0


def containment_match(a: str, b: str) -> float:
    """Check if one name contains the other, or if they match after stripping TECH_SUFFIXES.

    Returns 0.88-0.95 depending on match quality.
    """
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0

    # Strip tech suffixes and compare (checked first — higher confidence)
    def _strip_suffixes(name: str) -> str:
        for suf in TECH_SUFFIXES:
            if name.endswith(suf):
                name = name[: -len(suf)].strip()
        # Also strip leading "the "
        if name.startswith("the "):
            name = name[4:]
        return name.strip()

    sa, sb = _strip_suffixes(na), _strip_suffixes(nb)
    if sa and sb and sa == sb and na != nb:
        return 0.95

    # After stripping, check containment
    if sa and sb and sa != sb and (sa in sb or sb in sa):
        short, long_ = (sa, sb) if len(sa) < len(sb) else (sb, sa)
        ratio = len(short) / len(long_)
        if ratio >= 0.5:
            return 0.90

    # Exact containment on original normalized names
    if na != nb and (na in nb or nb in na):
        short, long_ = (na, nb) if len(na) < len(nb) else (nb, na)
        ratio = len(short) / len(long_)
        if ratio >= 0.5:
            return 0.88

    return 0.0


def canonical_match(a: str, b: str) -> float:
    """Lookup in KNOWN_ALIASES table.

    Returns the alias confidence if both names belong to the same alias set, 0.0 otherwise.
    """
    na, nb = _normalize(a), _normalize(b)
    for alias_set, confidence in KNOWN_ALIASES:
        if na in alias_set and nb in alias_set and na != nb:
            return confidence
    return 0.0


def compute_name_score(a: str, b: str) -> float:
    """Return the max of all name matchers including existing compute_similarity."""
    decision, base_score = policy_aware_similarity(a, b, compute_similarity)
    if not decision.allowed or decision.exact_identifier_match:
        return base_score
    return max(
        base_score,
        acronym_match(a, b),
        numeronym_match(a, b),
        containment_match(a, b),
        canonical_match(a, b),
    )


# ---------------------------------------------------------------------------
# Type compatibility
# ---------------------------------------------------------------------------


def type_compatible(type_a: str, type_b: str) -> bool:
    """Check if two entity types are compatible for merging."""
    if type_a == type_b:
        return True
    if "Other" in (type_a, type_b):
        return True
    return frozenset({type_a, type_b}) in COMPATIBLE_CROSS_TYPES


# ---------------------------------------------------------------------------
# Summary overlap
# ---------------------------------------------------------------------------


def summary_overlap(summary_a: str | None, summary_b: str | None) -> float:
    """Dice coefficient on key terms from summaries."""
    if not summary_a or not summary_b:
        return 0.0
    tokens_a = set(w.lower() for w in re.findall(r"\b\w{3,}\b", summary_a)) - _SUMMARY_STOP
    tokens_b = set(w.lower() for w in re.findall(r"\b\w{3,}\b", summary_b)) - _SUMMARY_STOP
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return 2 * len(intersection) / (len(tokens_a) + len(tokens_b))


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


async def score_merge_pair(
    ea,
    eb,
    search_index,
    graph_store,
    group_id: str,
    cross_encoder_enabled: bool = True,
) -> tuple[str, float, dict]:
    """Score a merge candidate pair using multi-signal ensemble.

    Returns (verdict, confidence, signal_breakdown).
    verdict is "merge", "keep_separate", or "uncertain".
    """
    merge_threshold = 0.82
    reject_threshold = 0.55

    # Signal 2: Type gate
    if not type_compatible(ea.entity_type, eb.entity_type):
        return "keep_separate", 0.0, {"reason": "incompatible_types"}

    policy = dedup_policy(ea.name, eb.name)
    policy_summary = policy_features(policy)
    if not policy.allowed:
        return "keep_separate", 0.0, {"reason": policy.reason, **policy_summary}

    # Signal 1: Name analysis
    name_score = compute_name_score(ea.name, eb.name)
    if ea.entity_type == eb.entity_type and not policy.exact_identifier_match:
        name_score = min(name_score + 0.03, 1.0)

    if policy.exact_identifier_match:
        signals = {
            "name": 1.0,
            "embedding": 0.0,
            "neighbor_overlap": 0.0,
            "summary_overlap": round(
                summary_overlap(getattr(ea, "summary", None), getattr(eb, "summary", None)),
                4,
            ),
            "exclusivity": 0.0,
            "reason": policy.reason,
            **policy_summary,
        }
        return "merge", 0.99, signals

    # Signal 3: Embedding similarity (rescaled)
    emb_score = 0.0
    try:
        embeddings = await search_index.get_entity_embeddings(
            [ea.id, eb.id],
            group_id=group_id,
        )
        if ea.id in embeddings and eb.id in embeddings:
            vec_a = np.array(embeddings[ea.id], dtype=np.float32)
            vec_b = np.array(embeddings[eb.id], dtype=np.float32)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a > 0 and norm_b > 0:
                cos = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                emb_score = max(0.0, (cos - 0.70) / 0.30)
    except Exception:
        pass

    # Signal 4: Neighbor overlap (Jaccard)
    nbr_score = 0.0
    try:
        neighbors_a = await graph_store.get_active_neighbors_with_weights(ea.id, group_id)
        neighbors_b = await graph_store.get_active_neighbors_with_weights(eb.id, group_id)
        nid_a = {nid for nid, _, _, _ in neighbors_a}
        nid_b = {nid for nid, _, _, _ in neighbors_b}
        union = nid_a | nid_b
        if union:
            nbr_score = len(nid_a & nid_b) / len(union)
    except Exception:
        pass

    # Signal 5: Summary overlap (Dice)
    sum_score = summary_overlap(
        getattr(ea, "summary", None),
        getattr(eb, "summary", None),
    )

    # Signal 6: Referential exclusivity (complementary distribution)
    # Entities that never co-occur in episodes but share neighbors = likely same entity
    # Entities that frequently co-occur = provably distinct
    exclusivity_score = 0.0
    try:
        cooccurrence = await graph_store.get_episode_cooccurrence_count(
            ea.id,
            eb.id,
            group_id,
        )
        if cooccurrence == 0:
            # Never co-occur — strong positive signal for merge
            # Scale by neighbor overlap (need shared context to be meaningful)
            if nbr_score > 0:
                exclusivity_score = 0.8 + 0.2 * nbr_score
            else:
                exclusivity_score = 0.3  # No shared neighbors, weak signal
        elif cooccurrence <= 2:
            exclusivity_score = 0.2  # Rare co-occurrence, mild positive
        else:
            # Frequent co-occurrence = anti-merge signal (penalty)
            exclusivity_score = -0.3
    except Exception:
        pass

    # Ensemble
    confidence = (
        0.35 * name_score
        + 0.25 * emb_score
        + 0.15 * nbr_score
        + 0.10 * sum_score
        + 0.15 * max(exclusivity_score, 0.0)  # Only positive contribution in ensemble
    )

    # Anti-merge penalty from co-occurrence (applied separately)
    if exclusivity_score < 0:
        confidence += exclusivity_score * 0.15  # Up to -0.045 penalty

    # Booster rules for high-confidence combinations
    if name_score >= 0.93 and emb_score >= 0.80:
        confidence = max(confidence, 0.95)
    if name_score >= 0.85 and nbr_score >= 0.50:
        confidence = max(confidence, 0.90)
    if emb_score >= 0.95 and sum_score >= 0.60:
        confidence = max(confidence, 0.88)
    # Person first-name/full-name: high fuzzy + high embedding
    if ea.entity_type == eb.entity_type == "Person" and name_score >= 0.70 and emb_score >= 0.70:
        confidence = max(confidence, 0.85)
    # Structural equivalence: high neighbor overlap + never co-occur = same entity
    if nbr_score >= 0.40 and exclusivity_score >= 0.7 and emb_score >= 0.50:
        confidence = max(confidence, 0.88)
    # Strong structural with high embedding even without exclusivity data
    if nbr_score >= 0.60 and emb_score >= 0.70:
        confidence = max(confidence, 0.85)

    signals = {
        "name": round(name_score, 4),
        "embedding": round(emb_score, 4),
        "neighbor_overlap": round(nbr_score, 4),
        "summary_overlap": round(sum_score, 4),
        "exclusivity": round(exclusivity_score, 4),
        **policy_summary,
    }

    if confidence >= merge_threshold:
        return "merge", round(confidence, 4), signals
    elif confidence < reject_threshold:
        return "keep_separate", round(confidence, 4), signals
    else:
        # Uncertain zone — try cross-encoder refinement (Tier 1)
        if cross_encoder_enabled:
            try:
                from engram.consolidation.scorers.cross_encoder import (
                    refine_merge_verdict,
                )

                verdict, refined = await refine_merge_verdict(
                    ea,
                    eb,
                    confidence,
                    merge_threshold,
                )
                signals["cross_encoder_refined"] = True
                return verdict, refined, signals
            except Exception:
                pass
        return "keep_separate", round(confidence, 4), signals
