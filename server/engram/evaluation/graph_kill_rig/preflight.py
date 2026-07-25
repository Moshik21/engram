"""The VOID pre-flight. A failed check REFUSES the run; it never degrades it.

GRAPH_THESIS.md §5: "the run is VOID if any of these fails". Five prior graph
refutations were each measured with at least one broken half, and §2.2 catalogues
seven inert or fabricating instruments found in a single day. A rig that emits a
number when its preconditions failed would be the eighth. So every check here is
a POSITIVE probe — it must be capable of failing loudly on a dead mechanism, and
``tests/test_graph_kill_rig.py`` neuters each one to prove it does.

The four checks the thesis names, plus two rig guards it does not:

1. **Producer positive probe.** At least one committed *semantic* relationship
   exists in the corpus group, and the per-question bridge structure is verified
   against the real store before anything is scored. "No error in the log" is
   never evidence.
2. **Consumer byte probe** (the census's missing fifth clause). More than zero
   bytes of EDGE-DERIVED content reach the answerer under arm B. The strict
   definition matters: M3.1's incidental finding #1 recorded a run where the
   traversal fired, appended rows, and the brain contained zero relationships —
   a loose byte count would have passed on a corpse.
3. **Gate probe.** Spreading completes on >= 80% of corpus recalls. On the live
   brain today it completes on 0/15.
4. **Residual measurement.** The fraction of questions where arm A fails to rank
   the LINKING episode in its top-10 — the graph's actual addressable market.
   Reported separately and fed to kill criterion K4; it fails the pre-flight only
   when it could not be computed at all.

Two guards this rig adds, each because it was caught by its own absence:

5. **Vector index probe.** Every gold episode must hold a vector. The rig's very
   first run measured a keyword-only system for ten minutes without noticing.
6. **Scored-set floor.** N may not be quietly shrunk below the anchor
   experiment's 36, because the pre-registered deltas are absolute counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engram.evaluation.graph_kill_rig.arms import QuestionRun, QuestionScore

# Predicates written by Engram's non-extraction producers: the repo scanner, the
# decision materializer, consolidation PMI, dream. GRAPH_THESIS.md Lens 3 attributes
# 91.1% of the live graph to these. None of them is evidence that an extractor ran,
# so none of them counts toward the producer probe.
STRUCTURAL_PREDICATES: frozenset[str] = frozenset(
    {
        "PART_OF",
        "MENTIONED_WITH",
        "SUPERSEDED_BY",
        "DOCUMENTED_IN",
        "IMPLEMENTED_BY",
        "DECIDED",
        "DREAM_ASSOCIATED",
    }
)

SPREAD_COMPLETION_MIN = 0.80


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    measured: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "measured": dict(self.measured),
        }


@dataclass(frozen=True)
class BridgeReport:
    """Per-question bridge verification against the real store."""

    present: list[str]
    missing: dict[str, str]
    semantic_relationship_count: int
    structural_relationship_count: int
    predicate_counts: dict[str, int]


@dataclass(frozen=True)
class PreflightReport:
    checks: list[Check]
    residual_rate: float | None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [c.as_dict() for c in self.checks],
            "residual_rate": self.residual_rate,
            "failures": self.failures,
        }


async def verify_bridges(
    graph: Any,
    *,
    group_id: str,
    questions: list[Any],
    tag_to_id: dict[str, str],
) -> BridgeReport:
    """Check every bridge against the store before a single question is scored.

    A bridge exists when, in the real graph: entity A exists, entity B exists,
    an edge joins them, and the gold episode is reachable from B by membership.
    Questions whose bridge did not materialise are DROPPED from the scored set —
    counting them as misses would blame retrieval for an extraction gap, which
    is exactly how R3 (2026-06-04) produced an uninterpretable null.
    """
    present: list[str] = []
    missing: dict[str, str] = {}
    predicate_counts: dict[str, int] = {}
    seen_edge_ids: set[str] = set()

    async def _resolve(name: str) -> Any | None:
        matches = await graph.find_entities(name=name, group_id=group_id, limit=10)
        exact = [e for e in matches if e.name.casefold() == name.casefold()]
        return (exact or matches or [None])[0]

    for question in questions:
        person = await _resolve(question.person)
        topic = await _resolve(question.topic)
        if person is None:
            missing[question.qid] = f"entity A ({question.person!r}) absent"
            continue
        if topic is None:
            missing[question.qid] = f"entity B ({question.topic!r}) absent"
            continue

        rels = await graph.get_relationships(person.id, group_id=group_id)
        for rel in rels:
            if rel.id not in seen_edge_ids:
                seen_edge_ids.add(rel.id)
                predicate_counts[rel.predicate] = predicate_counts.get(rel.predicate, 0) + 1
        linked = any(topic.id in (rel.source_id, rel.target_id) for rel in rels)
        if not linked:
            missing[question.qid] = "no edge joins A and B"
            continue

        gold_id = tag_to_id.get(question.gold_tag)
        episodes = await graph.get_episodes_for_entity(topic.id, group_id=group_id, limit=200)
        if gold_id is None or gold_id not in episodes:
            missing[question.qid] = "gold episode not reachable from B by membership"
            continue
        present.append(question.qid)

    semantic = sum(
        count for pred, count in predicate_counts.items() if pred not in STRUCTURAL_PREDICATES
    )
    structural = sum(
        count for pred, count in predicate_counts.items() if pred in STRUCTURAL_PREDICATES
    )
    return BridgeReport(
        present=present,
        missing=missing,
        semantic_relationship_count=semantic,
        structural_relationship_count=structural,
        predicate_counts=predicate_counts,
    )


def producer_probe(report: BridgeReport, *, questions_requested: int) -> Check:
    """Pre-flight 1 — at least one COMMITTED semantic relationship exists."""
    measured = {
        "semantic_relationship_count": report.semantic_relationship_count,
        "structural_relationship_count": report.structural_relationship_count,
        "bridges_present": len(report.present),
        "bridges_requested": questions_requested,
        "predicate_counts": dict(sorted(report.predicate_counts.items())),
    }
    if report.semantic_relationship_count < 1:
        return Check(
            name="producer_positive_probe",
            passed=False,
            detail=(
                "zero committed semantic relationships in the corpus group — the "
                "producer did not run, or every predicate it proposed was dropped. "
                "Measuring retrieval on this graph would measure the extractor."
            ),
            measured=measured,
        )
    if not report.present:
        return Check(
            name="producer_positive_probe",
            passed=False,
            detail="no bridge verified end to end (A -> edge -> B -> gold episode)",
            measured=measured,
        )
    return Check(
        name="producer_positive_probe",
        passed=True,
        detail=(
            f"{report.semantic_relationship_count} semantic relationships committed; "
            f"{len(report.present)}/{questions_requested} bridges verified in the store"
        ),
        measured=measured,
    )


async def _lite_episode_vector_ids(
    search_index: Any,
    episode_ids: list[str],
    group_id: str,
) -> tuple[set[str] | None, bool]:
    """Exact per-id episode vector presence on the SQLite/lite backend."""
    vectors = getattr(search_index, "_vectors", None)
    db = getattr(vectors, "db", None)
    if db is None:
        return None, False
    placeholders = ",".join("?" for _ in episode_ids)
    cursor = await db.execute(
        f"SELECT id FROM embeddings WHERE content_type = 'episode' "  # noqa: S608
        f"AND group_id = ? AND id IN ({placeholders})",
        (group_id, *episode_ids),
    )
    rows = await cursor.fetchall()
    return {str(row["id"]) for row in rows}, True


async def vector_index_probe(
    search_index: Any,
    *,
    gold_episode_ids: list[str],
    group_id: str,
) -> Check:
    """Rig guard: every gold episode must actually hold a vector.

    Added after this rig's first run measured a keyword-only system without
    noticing. ``FastEmbedProvider.dimension()`` returns 768 even when the ONNX
    model failed to load, so a dimension check passes on a dead embedder; the
    provider then logs one warning and writes zero vectors. An unvectored gold
    episode is unreachable by the vector lane in EVERY arm, which silently
    weakens arm A and flatters the graph — the exact direction of error this
    experiment cannot afford.
    """
    from engram.storage.index_completeness import _ids_with_vectors

    measured: dict[str, Any] = {"gold_episodes": len(gold_episode_ids)}
    if not gold_episode_ids:
        return Check(
            name="vector_index_probe",
            passed=False,
            detail="no gold episodes to probe",
            measured=measured,
        )
    present, exact = await _ids_with_vectors(
        search_index,
        list(gold_episode_ids),
        group_id,
        probe_attr="get_episode_embeddings",
        census_attr="_vector_search_episodes",
    )
    if present is None:
        # The lite backend exposes ``get_entity_embeddings`` but no episode
        # equivalent, and ``compute_similarity`` filters to content_type
        # 'entity'. Rather than assume coverage on the one backend the thesis
        # mandates for this experiment, read the vector table directly. Exact,
        # read-only, and preferred only after the public probe is unavailable.
        present, exact = await _lite_episode_vector_ids(search_index, gold_episode_ids, group_id)
    if present is None:
        return Check(
            name="vector_index_probe",
            passed=False,
            detail=(
                "the search index exposes no by-id vector probe and no ANN census, "
                "so vector coverage cannot be verified — refusing rather than assuming"
            ),
            measured=measured,
        )
    covered = len([eid for eid in gold_episode_ids if eid in present])
    measured.update({"gold_with_vectors": covered, "probe_is_exact": exact})
    if covered < len(gold_episode_ids):
        return Check(
            name="vector_index_probe",
            passed=False,
            detail=(
                f"only {covered}/{len(gold_episode_ids)} gold episodes hold a vector; "
                "the missing ones are unreachable by the vector lane in every arm"
            ),
            measured=measured,
        )
    return Check(
        name="vector_index_probe",
        passed=True,
        detail=f"all {covered} gold episodes hold a vector (exact probe: {exact})",
        measured=measured,
    )


def scored_set_floor_probe(bridges_present: int, *, floor: int) -> Check:
    """Rig guard, not one of the thesis's four: N may not be quietly shrunk.

    The pre-registered deltas are absolute question counts, so they only mean the
    same thing at a comparable N. Dropping unverified bridges is correct; dropping
    so many that the scored set falls under the anchor experiment's N=36 makes the
    thresholds incomparable, and shrinking N is the obvious way to game them.
    """
    passed = bridges_present >= floor
    return Check(
        name="scored_set_floor",
        passed=passed,
        detail=(
            f"{bridges_present} bridges verified, floor is {floor}"
            if passed
            else (
                f"only {bridges_present} bridges verified in the store, below the "
                f"pre-registered floor of {floor}; absolute thresholds are not "
                "comparable at a smaller N"
            )
        ),
        measured={"bridges_present": bridges_present, "floor": floor},
    )


def consumer_byte_probe(arm_b_runs: list[QuestionRun]) -> Check:
    """Pre-flight 2 — > 0 bytes of EDGE-DERIVED content reach the answerer.

    Strict by design. ``Row.edge_derived`` requires a non-zero spreading bonus or
    literal relationship JSON; a traversal row whose parent entity was found
    lexically is counted in ``traversal_chars`` and deliberately excluded from
    the gate, because a graph with zero relationships still produces those.
    """
    edge_chars = sum(row.chars for run in arm_b_runs for row in run.rows if row.edge_derived)
    traversal_chars = sum(row.chars for run in arm_b_runs for row in run.rows if row.traversal)
    relationship_chars = sum(row.relationship_chars for run in arm_b_runs for row in run.rows)
    questions_with_edge_bytes = sum(
        1 for run in arm_b_runs if any(row.edge_derived for row in run.rows)
    )
    measured = {
        "edge_derived_chars": edge_chars,
        "traversal_chars_loose": traversal_chars,
        "relationship_json_chars": relationship_chars,
        "questions_with_edge_derived_bytes": questions_with_edge_bytes,
        "questions": len(arm_b_runs),
    }
    if edge_chars <= 0:
        return Check(
            name="consumer_byte_probe",
            passed=False,
            detail=(
                "0 bytes of edge-derived content reached the answerer under arm B "
                f"(loose traversal bytes: {traversal_chars}). The consumer is the "
                "thing under measurement, not the graph."
            ),
            measured=measured,
        )
    return Check(
        name="consumer_byte_probe",
        passed=True,
        detail=(
            f"{edge_chars} edge-derived chars reached the answerer on "
            f"{questions_with_edge_bytes}/{len(arm_b_runs)} questions"
        ),
        measured=measured,
    )


def _walked_an_edge(run: QuestionRun) -> bool:
    """True when this recall provably traversed at least one relationship.

    The primary evidence is a RESULT-side observable, not a stage counter: a
    non-zero ``spreading`` in a row's score breakdown can only exist if
    ``bonuses[neighbor_id]`` was written, and ``activation/bfs.py`` writes that
    dict for NEIGHBOURS only (``:162``) — seeds are placed in ``hop_distances``
    at hop 0 (``:53-55``) and never in ``bonuses``. So ``spreading > 0`` is
    exactly "an edge was walked", and it is the only such signal that survives
    the pipeline's instrumentation being rewritten underneath this rig, which
    it was: ``recall_spread_reached`` and ``recall_spread_injected`` both
    existed in the working tree while this module was written and neither is
    emitted by the tree it now runs against. Stage counters are read as
    corroboration when present, never as the sole source.
    """
    if any(row.spreading > 0.0 for row in run.rows):
        return True
    timings = run.stage_timings
    return (
        float(timings.get("recall_spread_reached") or 0) >= 1
        or float(timings.get("recall_spread_injected") or 0) > 0
    )


def spread_gate_probe(runs: list[QuestionRun]) -> Check:
    """Pre-flight 3 — spreading completes on >= 80% of corpus recalls.

    Two clauses. The 80% completion bar is the thesis's, verbatim. The extra
    liveness floor (at least one recall provably walked an edge) is this rig's:
    a stage that "completes" having walked zero edges is precisely the
    silent-inert pass that lesson 1 exists to prevent, and ``recall_spread``
    alone cannot distinguish the two.
    """
    total = len(runs)
    completed = sum(1 for run in runs if "recall_spread" in run.stage_timings)
    timed_out = sum(1 for run in runs if "recall_spread_timeout" in run.stage_timings)
    skipped = sum(1 for run in runs if "recall_spread_skipped_probe_timeout" in run.stage_timings)
    reached = sum(1 for run in runs if _walked_an_edge(run))
    rate = completed / total if total else 0.0
    measured = {
        "questions": total,
        "completed": completed,
        "completion_rate": round(rate, 4),
        "timed_out": timed_out,
        "skipped_after_probe_timeout": skipped,
        "recalls_that_walked_an_edge": reached,
    }
    if total == 0:
        return Check(
            name="spread_gate_probe",
            passed=False,
            detail="no recalls to inspect",
            measured=measured,
        )
    if rate < SPREAD_COMPLETION_MIN:
        return Check(
            name="spread_gate_probe",
            passed=False,
            detail=(
                f"spreading completed on {completed}/{total} recalls "
                f"({rate:.0%}), below the {SPREAD_COMPLETION_MIN:.0%} bar "
                f"({timed_out} timed out, {skipped} skipped after a graph-probe timeout)"
            ),
            measured=measured,
        )
    if reached == 0:
        return Check(
            name="spread_gate_probe",
            passed=False,
            detail=(
                f"spreading 'completed' on {completed}/{total} recalls but walked ZERO "
                "edges on every one of them — the stage is inert, not healthy"
            ),
            measured=measured,
        )
    return Check(
        name="spread_gate_probe",
        passed=True,
        detail=(
            f"spreading completed on {completed}/{total} recalls ({rate:.0%}); "
            f"{reached} of them walked at least one edge"
        ),
        measured=measured,
    )


def residual_probe(arm_a_scores: list[QuestionScore], *, k: int = 10) -> tuple[Check, float | None]:
    """Pre-flight 4 — the graph's addressable market.

    Fraction of questions where arm A fails to rank the LINKING episode in its
    top-``k``. When the linking episode ranks, the agent can read "clock skew"
    off it and ask again, which is exactly what arm C does; the graph's unique
    claim lives only in the residual.
    """
    total = len(arm_a_scores)
    if total == 0:
        return (
            Check(
                name="residual_measurement",
                passed=False,
                detail="no arm-A scores to measure the residual from",
                measured={"questions": 0},
            ),
            None,
        )
    unranked = sum(1 for score in arm_a_scores if score.link_rank is None or score.link_rank > k)
    rate = unranked / total
    return (
        Check(
            name="residual_measurement",
            passed=True,
            detail=(
                f"the linking episode failed to rank in arm A's top-{k} on "
                f"{unranked}/{total} questions (residual {rate:.1%})"
            ),
            measured={
                "questions": total,
                "linking_episode_unranked": unranked,
                "residual_rate": round(rate, 4),
                "k": k,
            },
        ),
        rate,
    )
