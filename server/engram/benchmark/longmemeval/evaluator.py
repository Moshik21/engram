"""LongMemEval answer evaluation using LLM judges.

Implements the standard evaluation protocol from the LongMemEval paper:
question-type-specific prompts, binary correct/incorrect scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Question-type-specific judge prompts (aligned with LongMemEval evaluation)
# These are prompt templates sent to the judge LLM — kept readable, not wrapped.

_QUESTION_BLOCK = """
Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted_answer}
"""

_FACTUAL_JUDGE_PROMPT = (
    "You are evaluating a chat assistant's memory ability. "
    "Determine if the predicted answer is correct compared "
    "to the gold answer."
    + _QUESTION_BLOCK
    + """Rules:
- Correct if it contains key information from the gold answer.
- Minor wording differences are acceptable.
- Extra information is fine if the core answer is correct.
- Multi-part gold answers require ALL parts present.

Output ONLY "correct" or "incorrect"."""
)

_PREFERENCE_JUDGE_PROMPT = (
    "You are evaluating a chat assistant's ability to make "
    "personalized recommendations based on stated preferences."
    + _QUESTION_BLOCK
    + """Rules:
- Prediction should align with preferences in the gold answer.
- Exact match not required, but must show preference awareness.
- If gold lists specific preferences, prediction should reflect them.

Output ONLY "correct" or "incorrect"."""
)

_TEMPORAL_JUDGE_PROMPT = (
    "You are evaluating a chat assistant's temporal reasoning "
    "ability about past conversations."
    + _QUESTION_BLOCK
    + """Rules:
- Must correctly handle temporal aspects (dates, sequences).
- Latest information questions must reflect the most recent state.
- Time-related details must be accurate.

Output ONLY "correct" or "incorrect"."""
)

_KNOWLEDGE_UPDATE_JUDGE_PROMPT = (
    "You are evaluating whether a chat assistant correctly tracks "
    "knowledge updates where information changed over time."
    + _QUESTION_BLOCK
    + """Rules:
- Prediction must reflect the MOST RECENT state of information.
- Old/outdated information instead of current = incorrect.
- Must recognize that information can change.

Output ONLY "correct" or "incorrect"."""
)

_ABSTENTION_JUDGE_PROMPT = (
    "You are evaluating whether a chat assistant correctly "
    "abstains when the information was never discussed."
    + _QUESTION_BLOCK
    + """Rules:
- CORRECT if assistant says it doesn't know or lacks information.
- "I don't know", "never mentioned", etc. are correct abstentions.
- INCORRECT if assistant fabricates an answer.

Output ONLY "correct" or "incorrect"."""
)


def _get_judge_prompt(question_type: str, is_abstention: bool) -> str:
    """Select the appropriate judge prompt based on question type."""
    if is_abstention:
        return _ABSTENTION_JUDGE_PROMPT
    if question_type == "single-session-preference":
        return _PREFERENCE_JUDGE_PROMPT
    if question_type == "temporal-reasoning":
        return _TEMPORAL_JUDGE_PROMPT
    if question_type == "knowledge-update":
        return _KNOWLEDGE_UPDATE_JUDGE_PROMPT
    return _FACTUAL_JUDGE_PROMPT


@dataclass
class JudgeVerdict:
    """Result of evaluating one answer."""

    question_id: str
    question_type: str
    correct: bool
    judge_raw: str = ""


async def judge_answer(
    question_id: str,
    question_type: str,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    *,
    is_abstention: bool = False,
    judge_model: str = "claude-haiku-4-5-20251001",
    judge_provider: str = "anthropic",
) -> JudgeVerdict:
    """Evaluate a single answer using an LLM judge.

    Args:
        question_id: The question identifier.
        question_type: One of the LongMemEval question types.
        question: The question text.
        gold_answer: The ground truth answer.
        predicted_answer: The system's predicted answer.
        is_abstention: Whether this is an abstention question.
        judge_model: Model to use for judging.
        judge_provider: Provider ("anthropic" or "openai").
    """
    template = _get_judge_prompt(question_type, is_abstention)
    prompt = template.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
    )

    try:
        if judge_provider == "openai":
            raw = await _call_openai_judge(prompt, judge_model)
        else:
            raw = await _call_anthropic_judge(prompt, judge_model)
    except Exception as exc:
        logger.warning("Judge call failed for %s: %s", question_id, exc)
        # Fallback to token overlap heuristic
        raw = _heuristic_judge(gold_answer, predicted_answer, is_abstention)

    correct = "correct" in raw.lower() and "incorrect" not in raw.lower()

    return JudgeVerdict(
        question_id=question_id,
        question_type=question_type,
        correct=correct,
        judge_raw=raw.strip(),
    )


async def _call_anthropic_judge(prompt: str, model: str) -> str:
    """Call Anthropic API for judging."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def _call_openai_judge(prompt: str, model: str) -> str:
    """Call OpenAI API for judging (standard LongMemEval protocol uses GPT-4o)."""
    import openai

    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _heuristic_judge(gold: str, predicted: str, is_abstention: bool) -> str:
    """Fallback heuristic judge using token overlap."""
    pred_lower = predicted.lower()

    if is_abstention:
        abstention_phrases = [
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
        ]
        if any(phrase in pred_lower for phrase in abstention_phrases):
            return "correct"
        return "incorrect"

    # Token overlap for factual questions
    import re
    import string

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
        import math

        dcg = 0.0
        for i, sid in enumerate(top_k):
            if sid in answer_set:
                dcg += 1.0 / math.log2(i + 2)

        n_ideal = min(k, len(answer_set))
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        metrics[f"ndcg@{k}"] = ndcg

    return metrics
