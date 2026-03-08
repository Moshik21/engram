"""Deterministic answer-track helpers for the showcase benchmark."""

from __future__ import annotations

from typing import Any

from engram.benchmark.showcase.models import AnswerTask, EvidenceItem


def shared_answer_prompt() -> str:
    """Shared prompt contract for answer-track baselines."""
    return (
        "Answer using only the retrieved evidence. Return compact JSON with the "
        "same field names as the gold answer schema. Use null or empty strings "
        "for unknown fields."
    )


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def synthesize_answer(
    task: AnswerTask,
    evidence: list[EvidenceItem],
) -> Any:
    """Create a deterministic structured answer from surfaced evidence."""
    evidence_text = "\n".join(item.text for item in evidence)
    normalized_evidence = _normalize(evidence_text)

    def _pick(expected: Any) -> Any:
        if isinstance(expected, str):
            return expected if _normalize(expected) in normalized_evidence else ""
        if isinstance(expected, list):
            return [
                item
                for item in expected
                if isinstance(item, str) and _normalize(item) in normalized_evidence
            ]
        if isinstance(expected, dict):
            return {key: _pick(value) for key, value in expected.items()}
        return None

    return _pick(task.gold_answer)


def grade_answer(
    task: AnswerTask,
    answer: Any,
) -> tuple[float, bool, list[str], list[str], list[str], Any]:
    """Grade a deterministic answer against the task's canonical gold."""
    matched_fields: list[str] = []
    missing_fields: list[str] = []
    incorrect_fields: list[str] = []

    def _grade(expected: Any, actual: Any, path: str) -> tuple[int, int]:
        if isinstance(expected, dict):
            matched = 0
            total = 0
            for key, value in expected.items():
                sub_actual = actual.get(key) if isinstance(actual, dict) else None
                sub_path = f"{path}.{key}" if path else key
                sub_matched, sub_total = _grade(value, sub_actual, sub_path)
                matched += sub_matched
                total += sub_total
            return matched, total

        total = 1
        field_name = path or "answer"
        if isinstance(expected, str):
            expected_norm = _normalize(expected)
            actual_norm = _normalize(str(actual or ""))
            if actual_norm == expected_norm and expected_norm:
                matched_fields.append(field_name)
                return 1, total
            if not actual_norm:
                missing_fields.append(field_name)
            else:
                incorrect_fields.append(field_name)
            return 0, total

        if isinstance(expected, list):
            expected_norm = sorted(_normalize(str(item)) for item in expected)
            actual_norm = sorted(_normalize(str(item)) for item in (actual or []))
            if actual_norm == expected_norm and expected_norm:
                matched_fields.append(field_name)
                return 1, total
            if not actual_norm:
                missing_fields.append(field_name)
            else:
                incorrect_fields.append(field_name)
            return 0, total

        if expected == actual:
            matched_fields.append(field_name)
            return 1, total
        if actual in (None, "", []):
            missing_fields.append(field_name)
        else:
            incorrect_fields.append(field_name)
        return 0, total

    matched, total = _grade(task.gold_answer, answer, "")
    score = matched / total if total > 0 else 0.0
    normalized_answer = answer
    return score, score == 1.0, matched_fields, missing_fields, incorrect_fields, normalized_answer
