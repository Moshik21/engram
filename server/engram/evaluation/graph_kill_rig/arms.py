"""The three arms, and the row bookkeeping every pre-flight check reads.

* **A** — today's shipped default. ``entity_episode_traversal_source="results"``
  with ``passage_first_entity_budget=0``, i.e. traversal is enabled and starved.
* **B** — A plus ``entity_episode_traversal_source="candidates"``: additive
  entity->episode surfacing. **Zero relationship triples enter the evidence
  block.** Four independent results agree that triples-in-evidence is
  net-negative (GRAPH_THESIS.md §4, "positions to reject outright"), so B
  spends the graph on *which rows appear*, never on context.
* **C** — the kill arm. Arm A's own results, plus ONE second recall round
  seeded from them: what the harness agent can already do for itself with a
  loop. **No prior Engram A/B has ever run this arm.**

Arm C is the intellectual core of the experiment, so its fairness is the thing
most worth getting right:

* Its first round IS arm A — the same rows, reused, not re-run. There is no
  configuration difference to argue about, and the pairing is exact.
* Its second query is synthesised from the TEXT of round one only. No graph
  read of any kind participates. That is precisely the affordance an agent has:
  it reads what came back and asks again.
* It gets a full second retrieval, not a crippled one. B pays ~3 ms of
  traversal; C pays a whole extra recall, and that is allowed — the question is
  whether the graph buys a capability, not whether it is cheaper.
* Its rows are merged through ``prefer_durable_facts``, the same re-ranking arm
  B's appended rows go through, so neither arm gets a private ordering rule.
* Two merge variants are produced (``C_merged`` scores round-two rows on equal
  footing; ``C_concat`` appends them so they can never displace round one) and
  ``thresholds.select_kill_arm`` takes the STRONGER. Scoring the kill arm at
  its strongest biases the whole experiment against the graph, which is the
  correct direction of error here.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from engram.retrieval.result_selection import prefer_durable_facts

# Config deltas that define each arm. Applied on top of a single base config so
# the arms differ ONLY in these fields.
ARM_A_OVERRIDES: dict[str, Any] = {
    "entity_episode_traversal_source": "results",
    "passage_first_entity_budget": 0,
}
ARM_B_OVERRIDES: dict[str, Any] = {
    "entity_episode_traversal_source": "candidates",
    "passage_first_entity_budget": 0,
}


@dataclass(frozen=True)
class Row:
    """One row as the answerer receives it."""

    result_type: str
    episode_id: str | None
    entity_id: str | None
    score: float
    chars: int
    traversal: bool
    spreading: float
    edge_proximity: float
    activation: float
    relationship_chars: int
    text: str = ""

    @property
    def edge_derived(self) -> bool:
        """True only when an edge had to be WALKED for this row to be here.

        ``edge_proximity > 0`` alone does not qualify: ``derive_episode_signal``
        assigns ``hop_decay ** 1`` to an episode whose entity is a *seed*, i.e.
        something the query already matched, with no traversal involved. Only a
        non-zero spreading bonus (the entity received activation across an edge)
        or literal relationship JSON in the payload proves the graph did work.
        """
        return self.spreading > 0.0 or self.relationship_chars > 0


@dataclass(frozen=True)
class QuestionRun:
    qid: str
    rows: list[Row]
    ms: float
    stage_timings: dict[str, float] = field(default_factory=dict)
    second_query: str | None = None


@dataclass(frozen=True)
class QuestionScore:
    qid: str
    gold_rank: int | None
    link_rank: int | None


class RigScorer(Protocol):
    """Pluggable scorer.

    The rig's default is ``GoldEpisodeReachability``: it resolves a gold episode
    by ID, so it is immune to the containment defect that makes ``engram
    battery`` structurally blind to multi-source answers (GRAPH_THESIS.md M16).

    Lane 1's recall meter (``engram/evaluation/meter.py``, task #18) is adapted
    to this Protocol by :class:`MultiSourceCover` rather than replacing it: an
    id match cannot false-positive, a token cover can, so the two are reported
    side by side and never merged. ``engram battery`` is NOT used at any point —
    its one-row containment rule is blind to the multi-hop answer this whole
    experiment exists to detect.

    Swapping in a different scorer requires no change to the arms, the
    pre-flight, or the thresholds: given a ``BridgeQuestion``-shaped object, the
    ordered rows, and the tag->episode-id map, return a ``QuestionScore``.

    **Stated assumption.** ``MultiSourceCover`` imports ``minimal_cover``
    lazily, so the default id-based path keeps working if lane 1's module moves
    or its signature changes; only the second reading would break, and loudly.
    """

    name: str

    def score(
        self,
        question: Any,
        rows: list[Row],
        tag_to_id: dict[str, str],
    ) -> QuestionScore: ...


class GoldEpisodeReachability:
    """Rank of the gold episode id, and of the LINKING episode id, in the rows."""

    name = "gold_episode_reachability"

    def score(
        self,
        question: Any,
        rows: list[Row],
        tag_to_id: dict[str, str],
    ) -> QuestionScore:
        gold_id = tag_to_id.get(question.gold_tag)
        link_id = tag_to_id.get(question.link_tag)
        gold_rank = link_rank = None
        for index, row in enumerate(rows, start=1):
            if gold_rank is None and gold_id is not None and row.episode_id == gold_id:
                gold_rank = index
            if link_rank is None and link_id is not None and row.episode_id == link_id:
                link_rank = index
        return QuestionScore(qid=question.qid, gold_rank=gold_rank, link_rank=link_rank)


class MultiSourceCover:
    """Lane 1's union rule, applied to the same captured rows.

    Delegates to ``engram.evaluation.meter.minimal_cover`` — the instrument task
    #18 built precisely because ``engram battery`` scores a two-row answer as a
    MISS by construction (GRAPH_THESIS.md M16). Every question in this corpus
    carries a token group that spans the link episode and the gold episode, so a
    HIT here means the answerer could actually ASSEMBLE the answer, which is a
    strictly better proxy for the graph's claim than "the gold id appeared".

    ``max_sources=2`` on purpose: "assembled from two rows" is the multi-hop
    case, and meter.py's own docstring records that a group scattered across
    five rows is coincidence.

    Reported as a SECOND reading, never as a replacement. The default scorer
    stays id-based because an id match cannot false-positive, and the two are
    meant to be read side by side.
    """

    name = "multi_source_cover"

    def __init__(self, max_sources: int = 2) -> None:
        self.max_sources = max_sources

    def score(
        self,
        question: Any,
        rows: list[Row],
        tag_to_id: dict[str, str],
    ) -> QuestionScore:
        from engram.evaluation.meter import minimal_cover

        texts = [row.text for row in rows]
        best: int | None = None
        for group in getattr(question, "expected_tokens", None) or []:
            cover = minimal_cover(group, texts, self.max_sources)
            if cover is None:
                continue
            # The rank at which the answer becomes assemblable: the deepest row
            # the cover needs. reach@5 then means "assemblable from the top 5".
            depth = max(cover) + 1
            best = depth if best is None else min(best, depth)
        link_id = tag_to_id.get(question.link_tag)
        link_rank = next(
            (i for i, row in enumerate(rows, start=1) if row.episode_id == link_id), None
        )
        return QuestionScore(qid=question.qid, gold_rank=best, link_rank=link_rank)


SCORERS: dict[str, Any] = {
    "gold_episode": GoldEpisodeReachability,
    "multi_source_cover": MultiSourceCover,
}


def _flatten(obj: Any, parts: list[str]) -> None:
    if isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _flatten(value, parts)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _flatten(value, parts)


def row_text(raw: dict[str, Any]) -> str:
    """Everything in the row that reaches the answerer, flattened.

    Same flattening shape as ``engram.evaluation.battery.top3_result_texts`` so
    the two instruments agree on what "handed to the answerer" means.
    """
    parts: list[str] = []
    _flatten(raw, parts)
    return "\n".join(parts)


def to_row(raw: dict[str, Any]) -> Row:
    breakdown = raw.get("score_breakdown") or {}
    episode = raw.get("episode") or {}
    entity = raw.get("entity") or {}
    relationships = raw.get("relationships") or []
    rel_parts: list[str] = []
    _flatten(relationships, rel_parts)
    text = row_text(raw)
    return Row(
        text=text,
        result_type=str(raw.get("result_type") or ""),
        episode_id=episode.get("id") if isinstance(episode, dict) else None,
        entity_id=entity.get("id") if isinstance(entity, dict) else None,
        score=float(raw.get("score") or 0.0),
        chars=len(text),
        traversal=bool(breakdown.get("entity_traversal")),
        spreading=float(breakdown.get("spreading") or 0.0),
        edge_proximity=float(breakdown.get("edge_proximity") or 0.0),
        activation=float(breakdown.get("activation") or 0.0),
        relationship_chars=len("\n".join(rel_parts)),
    )


# --- arm C: the agent's own second query ----------------------------------

_TERM = re.compile(r"[A-Za-z][A-Za-z0-9_./-]{3,}")
_STOP = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "they",
        "them",
        "their",
        "were",
        "been",
        "into",
        "than",
        "then",
        "when",
        "what",
        "where",
        "which",
        "about",
        "after",
        "before",
        "there",
        "here",
        "note",
        "only",
        "over",
        "some",
        "more",
        "most",
        "also",
        "just",
        "keep",
        "made",
        "make",
        "does",
        "done",
    }
)


def second_round_query(
    original: str,
    rows: list[dict[str, Any]],
    *,
    top_k: int = 3,
    max_terms: int = 5,
) -> str:
    """Synthesise the follow-up an agent would type after reading round one.

    Text-only by construction: it reads the rows' rendered text and nothing
    else. Identifier-shaped tokens (``recall_graph_gate``, ``pipeline.py``) and
    capitalised tokens are preferred because those are what an agent actually
    latches onto in a code-session transcript. Deterministic: frequency first,
    first-appearance order as the tie-break.
    """
    seen_in_query = {m.group(0).casefold() for m in _TERM.finditer(original)}
    blob = "\n".join(row_text(raw) for raw in rows[:top_k])

    order: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for position, match in enumerate(_TERM.finditer(blob)):
        token = match.group(0)
        key = token.casefold()
        if key in seen_in_query or key in _STOP:
            continue
        interesting = any(ch in token for ch in "_./-") or token[0].isupper()
        if not interesting:
            continue
        counts[token] += 1
        order.setdefault(token, position)

    ranked = sorted(counts, key=lambda t: (-counts[t], order[t]))
    if not ranked:
        return original
    return f"{original} {' '.join(ranked[:max_terms])}"


def merge_variants(
    round_one: list[dict[str, Any]],
    round_two: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Two honest ways an agent could hand two rounds to the answerer.

    ``merged`` re-ranks the union through the same ``prefer_durable_facts`` arm
    B's appended rows go through — round-two rows compete on equal footing.
    ``concat`` keeps round one intact and appends round-two novelties, so a
    second query can never cost the agent a row it already had.
    """
    seen: set[str] = set()

    def _key(raw: dict[str, Any]) -> str:
        episode = raw.get("episode") or {}
        entity = raw.get("entity") or {}
        return str(
            (episode.get("id") if isinstance(episode, dict) else None)
            or (entity.get("id") if isinstance(entity, dict) else None)
            or id(raw)
        )

    union: list[dict[str, Any]] = []
    for raw in [*round_one, *round_two]:
        key = _key(raw)
        if key in seen:
            continue
        seen.add(key)
        union.append(raw)

    novel = union[len(round_one) :]
    return {
        "merged": prefer_durable_facts(list(union)),
        "concat": [*round_one, *novel],
    }


# --- arm runners -----------------------------------------------------------


async def run_arm_c(
    manager: Any,
    questions: list[Any],
    arm_a_raw: dict[str, list[dict[str, Any]]],
    arm_a_ms: dict[str, float],
    *,
    group_id: str,
    limit: int,
) -> dict[str, list[QuestionRun]]:
    """The kill arm: arm A's rows plus one second recall seeded from them.

    ``arm_a_raw`` is arm A's *unmodified* returned payload per question, so
    round one is literally arm A and the pairing is exact.
    """
    merged_runs: list[QuestionRun] = []
    concat_runs: list[QuestionRun] = []
    for question in questions:
        round_one = arm_a_raw.get(question.qid, [])
        follow_up = second_round_query(question.query, round_one)
        started = time.perf_counter()
        round_two = (
            await manager.recall(follow_up, group_id, limit=limit, record_access=False)
            if follow_up != question.query
            else []
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        total_ms = round(arm_a_ms.get(question.qid, 0.0) + elapsed, 4)
        variants = merge_variants(round_one, round_two)
        stage = dict(manager.get_last_recall_stage_timings())
        merged_runs.append(
            QuestionRun(
                qid=question.qid,
                rows=[to_row(raw) for raw in variants["merged"]],
                ms=total_ms,
                stage_timings=stage,
                second_query=follow_up,
            )
        )
        concat_runs.append(
            QuestionRun(
                qid=question.qid,
                rows=[to_row(raw) for raw in variants["concat"]],
                ms=total_ms,
                stage_timings=stage,
                second_query=follow_up,
            )
        )
    return {"merged": merged_runs, "concat": concat_runs}
