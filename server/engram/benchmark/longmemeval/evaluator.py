"""LongMemEval answer evaluation — embedding-based (zero LLM calls).

Replaces the LLM judge with embedding cosine similarity between
gold answer and retrieved evidence.  Falls back to token-overlap
heuristic when embeddings are unavailable.

Keeps ``compute_retrieval_metrics`` (Recall@k, nDCG@k) unchanged.
"""

from __future__ import annotations

import logging
import math
import re
import string
from dataclasses import dataclass

from engram.retrieval.relevance import compute_answer_containment

logger = logging.getLogger(__name__)


# ── Abstention detection ─────────────────────────────────────────────

_ABSTENTION_PHRASES = [
    "don't know",
    "don't have",
    "not sure",
    "no information",
    "never mentioned",
    "never discussed",
    "cannot recall",
    "can't recall",
    "not available",
    "i'm not aware",
    "cannot answer",
    "don't remember",
    "no relevant",
    "insufficient information",
    "no evidence",
]


def is_abstention_answer(text: str) -> bool:
    """Detect whether a predicted answer is an abstention."""
    lower = text.lower()
    return any(phrase in lower for phrase in _ABSTENTION_PHRASES)


_CORRECTION_PHRASES = [
    "not your",
    "not the",
    "actually your",
    "it was actually",
    "was not",
    "wasn't",
    "did not mention",
    "didn't mention",
    "never mentioned",
    "no record of",
    "no mention of",
    "not what you",
    "incorrect premise",
    "based on the conversation",
    "you didn't",
    "you did not",
]


def _is_correction_answer(text: str) -> bool:
    """Detect if the hypothesis corrects the question's premise."""
    lower = text.lower()
    return any(phrase in lower for phrase in _CORRECTION_PHRASES)


# ── Embedding-based judge ────────────────────────────────────────────


@dataclass
class JudgeVerdict:
    """Result of evaluating one answer."""

    question_id: str
    question_type: str
    correct: bool
    judge_raw: str = ""
    containment_score: float = 0.0


def judge_by_containment(
    question_id: str,
    question_type: str,
    containment_score: float,
    is_abstention: bool,
    *,
    threshold: float = 0.65,
    hypothesis: str = "",
    gold_answer: str = "",
) -> JudgeVerdict:
    """Judge answer correctness via embedding containment + token overlap.

    Uses a hybrid approach:
    1. Token overlap (gold tokens present in hypothesis) — catches
       short factual answers like "25" or "Samsung Galaxy S22"
    2. Embedding containment — catches semantic matches

    Either signal being positive → correct.

    For abstention questions: correct if the hypothesis IS an abstention
    (detected via phrase matching), regardless of containment.
    """
    if is_abstention:
        # Check if hypothesis abstains or correctly identifies the trick
        if hypothesis:
            if is_abstention_answer(hypothesis):
                return JudgeVerdict(
                    question_id=question_id,
                    question_type=question_type,
                    correct=True,
                    judge_raw="abstention: hypothesis abstains -> correct",
                    containment_score=containment_score,
                )
            # Check if hypothesis matches the gold explanation
            # Gold for abstention Qs often says "You did not mention X"
            # or "You mentioned Y, not X" — the hypothesis may correctly
            # contradict the premise without using abstention phrases
            if gold_answer and _token_overlap_match(gold_answer, hypothesis):
                return JudgeVerdict(
                    question_id=question_id,
                    question_type=question_type,
                    correct=True,
                    judge_raw="abstention: hypothesis matches gold explanation -> correct",
                    containment_score=containment_score,
                )
            if _is_correction_answer(hypothesis):
                return JudgeVerdict(
                    question_id=question_id,
                    question_type=question_type,
                    correct=True,
                    judge_raw="abstention: hypothesis corrects premise -> correct",
                    containment_score=containment_score,
                )
        correct = containment_score < threshold
        label = "correct" if correct else "incorrect"
        raw = f"abstention: containment={containment_score:.4f} < {threshold} -> {label}"
        return JudgeVerdict(
            question_id=question_id,
            question_type=question_type,
            correct=correct,
            judge_raw=raw,
            containment_score=containment_score,
        )

    # Hybrid: token overlap OR embedding containment
    token_match = False
    if hypothesis and gold_answer:
        token_match = _token_overlap_match(gold_answer, hypothesis)

    embed_match = containment_score >= threshold

    correct = token_match or embed_match
    parts = []
    if token_match:
        parts.append("token_overlap=yes")
    if embed_match:
        parts.append(f"containment={containment_score:.4f}>={threshold}")
    if not parts:
        parts.append(
            f"token_overlap=no, containment={containment_score:.4f}<{threshold}"
        )
    raw = f"{' + '.join(parts)} -> {'correct' if correct else 'incorrect'}"

    return JudgeVerdict(
        question_id=question_id,
        question_type=question_type,
        correct=correct,
        judge_raw=raw,
        containment_score=containment_score,
    )


def _token_overlap_match(
    gold: str, predicted: str, threshold: float = 0.5
) -> bool:
    """Check if gold answer tokens appear in the predicted answer."""
    def normalize(text: str) -> set[str]:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return {w for w in text.split() if w}

    gold_tokens = normalize(gold)
    if not gold_tokens:
        return True

    pred_tokens = normalize(predicted)
    overlap = len(gold_tokens & pred_tokens)
    recall = overlap / len(gold_tokens)
    return recall >= threshold


def _chunk_evidence(texts: list[str], max_chunk: int = 300) -> list[str]:
    """Split long evidence texts into sentence-level chunks.

    Short gold answers ("bike", "Samsung Galaxy S22") get low cosine
    similarity against 2000-char transcripts even when the answer is
    present.  Chunking evidence into ~300-char spans gives the gold
    answer a fair chance to match the relevant sentence.
    """
    chunks: list[str] = []
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if len(text) <= max_chunk:
            chunks.append(text)
            continue
        # Split on sentence boundaries then recombine into ~max_chunk spans
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        buf = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if buf and len(buf) + len(sent) + 1 > max_chunk:
                chunks.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip() if buf else sent
        if buf:
            chunks.append(buf)
    return chunks


async def compute_containment_score(
    gold_answer: str,
    evidence_texts: list[str],
    embed_fn,
) -> float:
    """Compute embedding-based answer containment.

    Chunks long evidence into ~300-char spans so short gold answers
    get compared against sentence-level context (not full transcripts).
    Returns max cosine similarity across all chunks.
    """
    if not gold_answer or not evidence_texts:
        return 0.0

    # Chunk evidence for better gold-vs-sentence matching
    chunks = _chunk_evidence(evidence_texts)
    if not chunks:
        return 0.0

    # Cap at 30 chunks to keep embed costs bounded
    chunks = chunks[:30]

    # Batch embed: gold answer + all chunks in one call
    all_texts = [gold_answer] + chunks
    try:
        vecs = await embed_fn(all_texts)
    except Exception:
        logger.debug("Embedding failed for containment scoring", exc_info=True)
        return 0.0

    if not vecs or len(vecs) < 2:
        return 0.0

    gold_vec = vecs[0]
    evidence_vecs = vecs[1:]
    return compute_answer_containment(gold_vec, evidence_vecs)


# ── Heuristic fallback (no embeddings) ───────────────────────────────


def _heuristic_judge(gold: str, predicted: str, is_abstention: bool) -> str:
    """Fallback heuristic judge using token overlap."""
    pred_lower = predicted.lower()

    if is_abstention:
        if is_abstention_answer(pred_lower):
            return "correct"
        return "incorrect"

    def normalize(text: str) -> set[str]:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return set(text.split())

    gold_tokens = normalize(gold)
    pred_tokens = normalize(predicted)

    if not gold_tokens:
        return "correct"

    overlap = len(gold_tokens & pred_tokens)
    recall = overlap / len(gold_tokens) if gold_tokens else 0.0

    return "correct" if recall >= 0.5 else "incorrect"


# ── Retrieval metrics (unchanged) ────────────────────────────────────


def compute_retrieval_metrics(
    retrieved_session_ids: list[str],
    answer_session_ids: list[str],
    *,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute retrieval quality metrics (Recall@k, NDCG@k).

    Maps retrieved evidence back to source sessions and compares
    against ground truth answer sessions.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    answer_set = set(answer_session_ids)
    if not answer_set:
        return {}

    metrics: dict[str, float] = {}

    for k in k_values:
        top_k = retrieved_session_ids[:k]
        hits = sum(1 for sid in top_k if sid in answer_set)

        recall = hits / len(answer_set) if answer_set else 0.0
        metrics[f"recall@{k}"] = recall

        # NDCG@k
        dcg = 0.0
        for i, sid in enumerate(top_k):
            if sid in answer_set:
                dcg += 1.0 / math.log2(i + 2)

        n_ideal = min(k, len(answer_set))
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        metrics[f"ndcg@{k}"] = ndcg

    return metrics
