"""Tests for embedding-based benchmark evaluation (zero LLM)."""

from __future__ import annotations

import pytest

from engram.benchmark.longmemeval.evaluator import (
    JudgeVerdict,
    compute_containment_score,
    compute_retrieval_metrics,
    is_abstention_answer,
    judge_by_containment,
)

# ── is_abstention_answer ─────────────────────────────────────────────

def test_abstention_phrases_detected():
    assert is_abstention_answer("I don't know the answer.")
    assert is_abstention_answer("I cannot recall that information.")
    assert is_abstention_answer("There is no relevant information.")
    assert is_abstention_answer("I'm not aware of that.")


def test_non_abstention():
    assert not is_abstention_answer("The answer is 42.")
    assert not is_abstention_answer("Python is a programming language.")
    assert not is_abstention_answer("He lives in New York.")


# ── judge_by_containment ─────────────────────────────────────────────

def test_factual_correct():
    verdict = judge_by_containment(
        question_id="q1",
        question_type="single-session-user",
        containment_score=0.85,
        is_abstention=False,
        threshold=0.72,
    )
    assert verdict.correct is True
    assert "correct" in verdict.judge_raw


def test_factual_incorrect():
    verdict = judge_by_containment(
        question_id="q2",
        question_type="single-session-user",
        containment_score=0.50,
        is_abstention=False,
        threshold=0.72,
    )
    assert verdict.correct is False
    assert "incorrect" in verdict.judge_raw


def test_abstention_correct():
    """Abstention is correct when evidence does NOT contain the answer."""
    verdict = judge_by_containment(
        question_id="q3_abs",
        question_type="single-session-user",
        containment_score=0.30,  # low containment = evidence doesn't have answer
        is_abstention=True,
        threshold=0.72,
    )
    assert verdict.correct is True


def test_abstention_incorrect():
    """Abstention is incorrect when evidence DOES contain the answer."""
    verdict = judge_by_containment(
        question_id="q4_abs",
        question_type="single-session-user",
        containment_score=0.80,  # high containment = answer IS in evidence
        is_abstention=True,
        threshold=0.72,
    )
    assert verdict.correct is False


def test_custom_threshold():
    verdict = judge_by_containment(
        question_id="q5",
        question_type="temporal-reasoning",
        containment_score=0.60,
        is_abstention=False,
        threshold=0.50,  # lower threshold
    )
    assert verdict.correct is True


def test_verdict_has_containment_score():
    verdict = judge_by_containment(
        question_id="q6",
        question_type="knowledge-update",
        containment_score=0.777,
        is_abstention=False,
    )
    assert verdict.containment_score == pytest.approx(0.777)


def test_exact_threshold():
    """Score exactly at threshold should be correct."""
    verdict = judge_by_containment(
        question_id="q7",
        question_type="single-session-user",
        containment_score=0.72,
        is_abstention=False,
        threshold=0.72,
    )
    assert verdict.correct is True


# ── compute_containment_score ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_containment_score_high():
    """Matching evidence should give high containment."""
    async def embed(texts: list[str]) -> list[list[float]]:
        # Return similar vectors for gold and evidence
        return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

    score = await compute_containment_score(
        gold_answer="Python",
        evidence_texts=["Python is a language"],
        embed_fn=embed,
    )
    assert score == pytest.approx(1.0, abs=1e-4)


@pytest.mark.asyncio
async def test_containment_score_empty_evidence():
    async def embed(texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4] * len(texts)

    score = await compute_containment_score("gold", [], embed)
    assert score == 0.0


@pytest.mark.asyncio
async def test_containment_score_empty_gold():
    async def embed(texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4] * len(texts)

    score = await compute_containment_score("", ["evidence"], embed)
    assert score == 0.0


@pytest.mark.asyncio
async def test_containment_score_embed_failure():
    """Embedding failure should return 0.0, not raise."""
    async def embed(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("API error")

    score = await compute_containment_score("gold", ["evidence"], embed)
    assert score == 0.0


@pytest.mark.asyncio
async def test_containment_picks_best_evidence():
    """Should pick the evidence with highest similarity."""
    async def embed(texts: list[str]) -> list[list[float]]:
        # gold=[1,0,0,0], bad=[0,1,0,0], good=[0.95,0.05,0,0]
        vecs = [
            [1.0, 0.0, 0.0, 0.0],  # gold
            [0.0, 1.0, 0.0, 0.0],  # bad evidence
            [0.95, 0.05, 0.0, 0.0],  # good evidence
        ]
        return vecs[:len(texts)]

    score = await compute_containment_score(
        gold_answer="exact match",
        evidence_texts=["bad", "good"],
        embed_fn=embed,
    )
    # Should be the similarity between [1,0,0,0] and [0.95,0.05,0,0]
    assert score > 0.9


# ── compute_retrieval_metrics (unchanged) ────────────────────────────

def test_retrieval_metrics_perfect():
    metrics = compute_retrieval_metrics(
        retrieved_session_ids=["s1", "s2", "s3"],
        answer_session_ids=["s1", "s2"],
    )
    assert metrics["recall@3"] == 1.0
    assert metrics["ndcg@3"] == 1.0


def test_retrieval_metrics_partial():
    metrics = compute_retrieval_metrics(
        retrieved_session_ids=["s3", "s1"],
        answer_session_ids=["s1", "s2"],
    )
    assert metrics["recall@1"] == 0.0  # s3 not in answer
    assert metrics["recall@3"] == 0.5  # only s1 found


def test_retrieval_metrics_empty():
    metrics = compute_retrieval_metrics([], ["s1"])
    assert metrics["recall@1"] == 0.0


def test_retrieval_metrics_empty_answer():
    metrics = compute_retrieval_metrics(["s1"], [])
    assert metrics == {}


# ── Full flow integration ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_evaluation_flow():
    """Simulate the full benchmark evaluation flow."""
    # 1. Embed gold answer + evidence
    async def embed(texts: list[str]) -> list[list[float]]:
        # Gold answer similar to first evidence, different from second
        vecs = {
            0: [1.0, 0.0, 0.0, 0.0],   # gold
            1: [0.9, 0.1, 0.0, 0.0],    # good evidence
            2: [0.0, 0.0, 1.0, 0.0],    # irrelevant evidence
        }
        return [vecs.get(i, [0.0] * 4) for i in range(len(texts))]

    # 2. Compute containment
    score = await compute_containment_score(
        gold_answer="The answer is in the first evidence",
        evidence_texts=[
            "First evidence with the answer",
            "Completely irrelevant evidence",
        ],
        embed_fn=embed,
    )

    # 3. Judge
    verdict = judge_by_containment(
        question_id="test_q",
        question_type="single-session-user",
        containment_score=score,
        is_abstention=False,
    )

    assert score > 0.8  # good evidence match
    assert verdict.correct is True
    assert isinstance(verdict, JudgeVerdict)
