"""LoCoMo evaluation metrics: exact match and token F1."""

from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    Lowercases, removes articles, punctuation, and extra whitespace.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if normalized prediction matches ground truth, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = set(normalize_answer(prediction).split())
    truth_tokens = set(normalize_answer(ground_truth).split())

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2.0 * precision * recall / (precision + recall)
