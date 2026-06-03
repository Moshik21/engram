"""Optional LLM reader + judge for LongMemEval — the apples-to-apples cell.

The default harness is reader-free: correctness = "is the gold answer semantically
contained in the retrieved evidence" (a retrieval metric). Published systems
(Zep/TiMem/Supermemory) instead feed retrieved context to an LLM reader that
*generates* an answer, then grade that answer with an LLM judge. This module adds
that path so Engram's retrieval can be measured on the same footing. Off by
default; enabled via --reader llm / --judge llm.

Uses Claude (not GPT-4o), so numbers are LLM-reader-comparable but not identical to
the published GPT-4o-read figures — report the reader/judge model alongside.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _text(msg) -> str:
    """Concatenate text blocks from an Anthropic message."""
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

_READER_SYSTEM = (
    "You answer a question using ONLY the provided context from a user's memory. "
    "Be concise — give just the answer, no preamble. If the context does not "
    "contain enough information to answer, reply exactly: I don't know."
)

_JUDGE_SYSTEM = (
    "You grade whether a proposed answer is correct against a gold answer for a "
    "long-term-memory benchmark. The proposed answer is correct if it conveys the "
    "gold answer's key fact, even if phrased differently or with extra detail. "
    "For abstention questions (the premise is false / unanswerable), the proposed "
    "answer is correct iff it declines / says it doesn't know / corrects the premise. "
    "Reply with exactly one word: CORRECT or INCORRECT."
)


class LLMReaderJudge:
    """Generates an answer from retrieved evidence and (optionally) grades it."""

    def __init__(self, model: str = "claude-sonnet-4-6", max_evidence: int = 10) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic()
        self._model = model
        self._max_evidence = max_evidence

    async def read(self, question: str, evidence: list[str]) -> str:
        """Generate a concise answer from the retrieved evidence."""
        context = "\n".join(f"- {e}" for e in evidence[: self._max_evidence]) or "(no context)"
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_READER_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                },
            ],
        )
        return _text(msg)

    async def judge(
        self,
        *,
        question: str,
        hypothesis: str,
        gold_answer: str,
        is_abstention: bool,
    ) -> tuple[bool, str]:
        """Grade the generated answer against gold. Returns (correct, raw)."""
        kind = "ABSTENTION question (premise false/unanswerable)" if is_abstention else "answerable"
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=8,
            system=_JUDGE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question type: {kind}\n"
                        f"Question: {question}\n"
                        f"Gold answer: {gold_answer}\n"
                        f"Proposed answer: {hypothesis}\n\n"
                        "Grade (CORRECT or INCORRECT):"
                    ),
                },
            ],
        )
        raw = _text(msg).upper()
        return raw.startswith("CORRECT"), f"llm_judge({self._model}): {raw}"

    async def close(self) -> None:
        try:
            await self._client.close()
        except Exception:
            pass
