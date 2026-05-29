"""Strict, deterministic fact-presence judge for the depth-tier eval.

This is the trustworthy primary judge: it checks whether the SPECIFIC gold fact
(exact entity / value / date) appears as a normalized substring in the retrieved
evidence, NOT topical proximity (the embedding-containment cosine>=0.72 trap that
passes vague-but-topical answers). It is fully deterministic — no LLM, no
embeddings — so the verdict carries no judge nondeterminism or leniency.

Three query classes, scored separately and never blended:

  * multi_hop      — binary: gold present AND no forbidden string present.
  * current_value  — newest-wins: gold present AND no forbidden stale value, AND
                     (when an ordered evidence list is given) the gold must rank
                     ABOVE any forbidden stale value.
  * synthesis      — coverage: fraction of required_facts present; passes iff
                     coverage >= threshold AND no forbidden_fact present.
                     Also reports false_recall (any forbidden_fact hit).

Surface forms: each gold may carry a list of accepted surface forms (e.g.
"Staff Engineer" / "staff-level engineer") to avoid false-negatives when the
session wording differs from the canonical gold string.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from engram.benchmark.showcase.scoring import _normalize


@dataclass
class JudgeVerdict:
    """Result of judging one query's evidence."""

    qid: str
    query_type: str
    passed: bool
    coverage: float = 1.0
    gold_present: bool = False
    forbidden_hit: bool = False
    false_recall: bool = False
    hit_facts: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    notes: str = ""


def _forms(gold: str, accepted: list[str] | None) -> list[str]:
    """Return the normalized surface forms accepted for a gold fact."""
    forms = [gold]
    if accepted:
        forms.extend(accepted)
    return [_normalize(f) for f in forms if f]


def _present(forms: list[str], evidence: list[str]) -> bool:
    """True if any accepted surface form is a substring of any evidence text."""
    norm_ev = [_normalize(e) for e in evidence]
    return any(any(f in ev for ev in norm_ev) for f in forms)


def _first_index(forms: list[str], evidence: list[str]) -> int | None:
    """Index of the first evidence item containing any accepted surface form."""
    for i, ev in enumerate(evidence):
        ne = _normalize(ev)
        if any(f in ne for f in forms):
            return i
    return None


def judge_multi_hop(
    qid: str,
    evidence: list[str],
    *,
    answer: str,
    accepted_forms: list[str] | None = None,
    forbidden: list[str] | None = None,
) -> JudgeVerdict:
    """Binary pass = gold present AND no forbidden string present."""
    gold_forms = _forms(answer, accepted_forms)
    gold_present = _present(gold_forms, evidence)

    forbidden_hit = False
    if forbidden:
        norm_ev = [_normalize(e) for e in evidence]
        forbidden_hit = any(
            any(_normalize(f) in ev for ev in norm_ev) for f in forbidden
        )

    return JudgeVerdict(
        qid=qid,
        query_type="multi_hop",
        passed=gold_present and not forbidden_hit,
        coverage=1.0 if gold_present else 0.0,
        gold_present=gold_present,
        forbidden_hit=forbidden_hit,
        hit_facts=[answer] if gold_present else [],
        missing_facts=[] if gold_present else [answer],
    )


def judge_current_value(
    qid: str,
    evidence: list[str],
    *,
    answer: str,
    forbidden: list[str],
    accepted_forms: list[str] | None = None,
    evidence_is_ordered: bool = True,
) -> JudgeVerdict:
    """Newest-wins judge.

    Passes iff the latest (gold) value is present AND no stale forbidden value
    outranks it. When ``evidence_is_ordered`` is True the evidence list is assumed
    to be newest-first (as produced by the adapter's knowledge-update sort), so
    the gold's first occurrence must precede any forbidden value's first
    occurrence. When unordered, the forbidden value simply must be absent.
    """
    gold_forms = _forms(answer, accepted_forms)
    gold_idx = _first_index(gold_forms, evidence)
    gold_present = gold_idx is not None

    forbidden_idx: int | None = None
    for stale in forbidden:
        idx = _first_index(_forms(stale, None), evidence)
        if idx is not None and (forbidden_idx is None or idx < forbidden_idx):
            forbidden_idx = idx
    forbidden_present = forbidden_idx is not None

    if not gold_present:
        passed = False
        note = "gold value absent"
    elif evidence_is_ordered and forbidden_present:
        # Newest-first ordering: gold must rank above (earlier index) the stale value.
        passed = gold_idx < forbidden_idx
        note = "" if passed else "stale value outranks gold (newest-wins violated)"
    else:
        passed = not forbidden_present
        note = "" if passed else "stale forbidden value present"

    return JudgeVerdict(
        qid=qid,
        query_type="current_value",
        passed=passed,
        coverage=1.0 if gold_present else 0.0,
        gold_present=gold_present,
        forbidden_hit=forbidden_present,
        hit_facts=[answer] if gold_present else [],
        missing_facts=[] if gold_present else [answer],
        notes=note,
    )


def judge_synthesis(
    qid: str,
    evidence: list[str],
    *,
    required_facts: list[str],
    forbidden_facts: list[str] | None = None,
    accepted_forms: dict[str, list[str]] | None = None,
    coverage_threshold: float = 0.66,
) -> JudgeVerdict:
    """Coverage judge over a SET of required facts spread across sessions.

    Pass = coverage >= threshold AND no forbidden_fact present (false-recall guard).
    """
    accepted_forms = accepted_forms or {}
    hit: list[str] = []
    missing: list[str] = []
    for fact in required_facts:
        forms = _forms(fact, accepted_forms.get(fact))
        if _present(forms, evidence):
            hit.append(fact)
        else:
            missing.append(fact)

    coverage = len(hit) / len(required_facts) if required_facts else 1.0

    false_recall = False
    if forbidden_facts:
        norm_ev = [_normalize(e) for e in evidence]
        false_recall = any(
            any(_normalize(f) in ev for ev in norm_ev) for f in forbidden_facts
        )

    passed = coverage >= coverage_threshold and not false_recall
    return JudgeVerdict(
        qid=qid,
        query_type="synthesis",
        passed=passed,
        coverage=coverage,
        gold_present=bool(hit),
        forbidden_hit=false_recall,
        false_recall=false_recall,
        hit_facts=hit,
        missing_facts=missing,
    )


# --------------------------------------------------------------------------- #
# Paired statistics over pooled queries (deterministic; no scipy dependency).  #
# --------------------------------------------------------------------------- #


def mcnemar_p(core_pass: list[bool], depth_pass: list[bool]) -> float:
    """Exact two-sided McNemar test p-value over paired binary outcomes.

    Counts discordant pairs:
      b = core PASS, depth FAIL ;  c = core FAIL, depth PASS.
    Uses the exact binomial test on the discordant pairs (no continuity-corrected
    chi-square, which is unreliable for the small n here).
    """
    if len(core_pass) != len(depth_pass):
        raise ValueError("paired lists must be equal length")
    b = sum(1 for c, d in zip(core_pass, depth_pass) if c and not d)
    c = sum(1 for c, d in zip(core_pass, depth_pass) if not c and d)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Two-sided exact binomial test, p=0.5.
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def pass_rate_bootstrap_ci(
    passes: list[bool],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> tuple[float, float]:
    """Bootstrap CI for a pass-rate. Seeded => deterministic across runs."""
    n = len(passes)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    vals = [1 if p else 0 for p in passes]
    means: list[float] = []
    for _ in range(iterations):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_i = int((1 - confidence) / 2 * iterations)
    hi_i = int((1 + confidence) / 2 * iterations) - 1
    return (means[lo_i], means[hi_i])


def delta_bootstrap_ci(
    core_pass: list[bool],
    depth_pass: list[bool],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> tuple[float, float]:
    """Paired bootstrap CI for (depth_pass_rate - core_pass_rate).

    Resamples query INDICES (paired) so the per-query correlation is preserved.
    A win = CI excludes 0.
    """
    n = len(core_pass)
    if n == 0 or len(depth_pass) != n:
        return (0.0, 0.0)
    rng = random.Random(seed)
    core = [1 if p else 0 for p in core_pass]
    depth = [1 if p else 0 for p in depth_pass]
    deltas: list[float] = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        c = sum(core[i] for i in idx) / n
        d = sum(depth[i] for i in idx) / n
        deltas.append(d - c)
    deltas.sort()
    lo_i = int((1 - confidence) / 2 * iterations)
    hi_i = int((1 + confidence) / 2 * iterations) - 1
    return (deltas[lo_i], deltas[hi_i])
