"""Orchestration for the graph deciding experiment. Refuses more readily than it reports.

Order of operations is load-bearing. Each stage is gated on the previous one, so
a run whose preconditions failed never gets far enough to produce a number that
could be quoted out of context:

1. build/reuse the corpus, open a throwaway lite brain, ingest
2. verify every bridge against the real store -> producer probe, scored-set floor
3. run arm A -> residual measurement
4. run arm B -> consumer byte probe; spread gate over all A+B recalls
5. if ANY check failed: emit ``status="VOID"``, with reachability SUPPRESSED
6. otherwise run arm C and evaluate the pre-registered thresholds

Step 5 is the point of the whole module. The arms that were run are reported
only as diagnostics (row counts, latencies, bridge coverage); every reachability
figure is withheld and the withholding is itself recorded, because a VOID run's
"22/36" would be indistinguishable in a summary from a valid one's.

``--fault`` deliberately breaks one precondition at a time. It exists so the
refusal path can be demonstrated on demand rather than asserted — "prove it can
fail" is the standing requirement for every mechanism in this repo, and it
applies to the instrument at least as much as to the thing measured.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engram.config import ActivationConfig, EngramConfig
from engram.evaluation.graph_kill_rig import preflight
from engram.evaluation.graph_kill_rig.arms import (
    ARM_A_OVERRIDES,
    ARM_B_OVERRIDES,
    GoldEpisodeReachability,
    QuestionRun,
    QuestionScore,
    RigScorer,
    run_arm_c,
    to_row,
)
from engram.evaluation.graph_kill_rig.corpus import Corpus, build_corpus
from engram.evaluation.graph_kill_rig.thresholds import (
    MIN_SCORED_QUESTIONS,
    PRE_REGISTRATION_SOURCE,
    ArmResult,
    evaluate,
    select_kill_arm,
)

FAULTS = ("none", "drop-relationships", "disable-traversal", "starve-spread")

# The knobs worth reading first when comparing two runs by eye. This is a
# READING AID, not the provenance record: the record is the whole activation
# config (see _config_snapshot), because a curated list of "the fields the arms
# depend on" is a frozen answer to a question that changes every time a knob is
# added — and it went stale exactly that way. `retrieval_spread_traversal_budget_ms`
# and `retrieval_spread_max_reads` entered the recall path in 390866b and were
# not added here; `max_reads=0 + budget=0` reproduces pre-fix zero-reach
# behaviour exactly, so a result recorded after the fix was indistinguishable
# from one recorded before it. Both are listed now, but listing them is not
# what fixed it — capturing everything is.
_ARM_CRITICAL_FIELDS = (
    "entity_episode_traversal_enabled",
    "entity_episode_traversal_source",
    "entity_episode_max_entities",
    "entity_episode_max_per_entity",
    "entity_episode_weight",
    "entity_episode_traversal_timeout_ms",
    "passage_first_entity_budget",
    "passage_first_channel_separated",
    "retrieval_spread_timeout_ms",
    "retrieval_spread_traversal_budget_ms",
    "retrieval_spread_max_reads",
    "spread_candidate_injection_max",
    "retrieval_skip_secondary_graph_after_probe_timeout",
    "episode_graph_signal_enabled",
    "weight_graph_structural",
    "recall_profile",
    "integration_profile",
)

# A renamed or deleted knob used to leave a `None` in the provenance block —
# a missing measurement wearing the shape of a real one. The rig refuses to
# import instead: a module that cannot describe its own config has no business
# producing a number.
_UNKNOWN_CRITICAL_FIELDS = tuple(
    name for name in _ARM_CRITICAL_FIELDS if name not in ActivationConfig.model_fields
)
if _UNKNOWN_CRITICAL_FIELDS:  # pragma: no cover - import-time guard
    raise RuntimeError(
        f"graph_kill_rig provenance names knobs ActivationConfig does not have: "
        f"{list(_UNKNOWN_CRITICAL_FIELDS)}"
    )


def arm_overrides(fault: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The config overrides arms A and B run under, for one fault mode.

    Extracted from ``run_rig`` so the set of knobs the rig itself varies is
    DERIVABLE rather than remembered — ``tests/test_graph_kill_rig.py`` walks
    every fault and fails if provenance would omit one. Arm C re-opens the brain
    under arm A's overrides, so it is covered by the A snapshot.
    """
    a = dict(ARM_A_OVERRIDES)
    b = dict(ARM_B_OVERRIDES)
    if fault == "disable-traversal":
        b["entity_episode_traversal_enabled"] = False
    if fault == "starve-spread":
        a["retrieval_spread_timeout_ms"] = 1
        b["retrieval_spread_timeout_ms"] = 1
    return a, b


class _NoopExtractor:
    """Never runs: the proposals path suppresses internal extraction entirely."""

    async def extract(self, *_args: Any, **_kwargs: Any) -> Any:
        from engram.extraction.extractor import ExtractionResult

        return ExtractionResult(entities=[], relationships=[])


@dataclass
class RigOptions:
    scratch_dir: Path
    repo_root: Path
    group_id: str = "graph_kill_rig"
    n: int = 60
    seed: int = 17
    limit: int = 10
    producer: str = "proposals"
    fault: str = "none"
    reuse: bool = False
    distractors_per_person: int = 1
    filler: int = 30


@dataclass
class _ArmRecord:
    runs: list[QuestionRun]
    scores: list[QuestionScore]
    raw: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return round(ordered[index], 4)


def _summarise(arm: str, runs: list[QuestionRun], scores: list[QuestionScore]) -> ArmResult:
    return ArmResult(
        arm=arm,
        n=len(scores),
        reach_at_5=sum(1 for s in scores if s.gold_rank is not None and s.gold_rank <= 5),
        reach_at_10=sum(1 for s in scores if s.gold_rank is not None and s.gold_rank <= 10),
        mean_rows=round(statistics.fmean([len(r.rows) for r in runs]), 4) if runs else 0.0,
        mean_chars=(
            round(statistics.fmean([sum(row.chars for row in r.rows) for r in runs]), 4)
            if runs
            else 0.0
        ),
        p50_ms=_percentile([r.ms for r in runs], 0.5),
        # Absent, not estimated: no tokenizer runs here, and a chars/4 guess is
        # exactly the plausible-but-wrong metric INSTRUMENT_AUDIT.md forbids.
        mean_tokens=None,
    )


def _diagnostics(arm: str, runs: list[QuestionRun]) -> dict[str, Any]:
    """Everything about an arm EXCEPT whether it found the answer."""
    return {
        "arm": arm,
        "questions": len(runs),
        "mean_rows": round(statistics.fmean([len(r.rows) for r in runs]), 4) if runs else 0.0,
        "p50_ms": _percentile([r.ms for r in runs], 0.5),
        "p95_ms": _percentile([r.ms for r in runs], 0.95),
        "reachability": "SUPPRESSED — pre-flight VOID",
    }


async def _open_brain(opts: RigOptions, overrides: dict[str, Any]) -> tuple[Any, Any, Any]:
    """A fresh manager over the scratch lite brain. Never the dogfood store."""
    from dotenv import load_dotenv

    from engram.graph_manager import GraphManager
    from engram.storage.factory import create_stores
    from engram.storage.index_completeness import ensure_embedding_provider_healthy
    from engram.storage.resolver import EngineMode

    # M13, in its nastiest form. ``FASTEMBED_CACHE_PATH`` is a PLAIN env var,
    # not a pydantic setting, so ``EngramConfig()`` reading ~/.engram/.env does
    # not export it. The LaunchAgent does (`set -a; source ...`), a CLI run does
    # not — and the fallback cache holds a different ONNX file than the
    # configured model needs. Measured here: without this line the provider
    # loads, reports dim=768, silently disables embeds, and the whole rig
    # measures a keyword-only system. Load it the way the launcher does.
    load_dotenv(Path.home() / ".engram" / ".env", override=False)

    config = EngramConfig()
    config.mode = "lite"
    config.sqlite.path = str(opts.scratch_dir / "brain.db")
    config.embedding.provider = "local"
    for key, value in overrides.items():
        setattr(config.activation, key, value)

    graph, activation, search = create_stores(EngineMode.LITE, config)
    await graph.initialize()
    await search.initialize()

    # POSITIVE probe, not a dimension check: ``dimension()`` returns 768 on a
    # provider whose model failed to load. Only a real embed call can tell the
    # difference, and telling the difference is the entire point.
    await ensure_embedding_provider_healthy(search)

    if opts.producer == "proposals":
        extractor: Any = _NoopExtractor()
    elif opts.producer == "narrow":
        from engram.extraction.narrow_adapter import NarrowExtractorAdapter

        extractor = NarrowExtractorAdapter(config.activation)
    else:
        from engram.extraction.factory import create_extractor

        config.activation.extraction_provider = opts.producer
        extractor = create_extractor(config)

    manager = GraphManager(
        graph,
        activation,
        search,
        extractor,
        cfg=config.activation,
        runtime_mode="graph_kill_rig",
    )
    return manager, graph, config


async def _drain_capture_indexing(manager: Any) -> None:
    """Wait for the capture service's background episode/cue index tasks."""
    drain = getattr(manager, "drain_cue_indexing", None)
    if not callable(drain):
        service = getattr(manager, "_capture_service", None)
        drain = getattr(service, "drain_cue_indexing", None)
    if callable(drain):
        await drain()


async def _ingest(manager: Any, corpus: Corpus, opts: RigOptions) -> dict[str, str]:
    tag_to_id: dict[str, str] = {}
    use_proposals = opts.producer == "proposals"
    drop_relationships = opts.fault == "drop-relationships"
    for episode in corpus.episodes:
        proposed_entities = episode.proposed_entities if use_proposals else None
        proposed_relationships = (
            episode.proposed_relationships if (use_proposals and not drop_relationships) else None
        )
        episode_id = await manager.ingest_episode(
            episode.content,
            group_id=opts.group_id,
            source=f"rig:{episode.role}",
            proposed_entities=proposed_entities or None,
            proposed_relationships=proposed_relationships or None,
            model_tier="opus",
        )
        tag_to_id[episode.tag] = episode_id
    return tag_to_id


def _config_snapshot(config: EngramConfig) -> dict[str, Any]:
    """Provenance for one arm: the WHOLE activation config, plus a highlight.

    ``all`` is total by construction, which is the only property that cannot go
    stale. The previous snapshot was a curated 15-field tuple and it did go
    stale: ``retrieval_spread_traversal_budget_ms`` and
    ``retrieval_spread_max_reads`` decide whether spreading traverses anything
    at all — ``max_reads=0`` with ``budget=0`` reproduces pre-390866b zero-reach
    behaviour exactly — and neither was captured, so a result recorded after the
    fix was byte-indistinguishable from one recorded before it.

    ``critical`` is a reading aid for a human diffing two envelopes. It is not
    the record, so its going stale is now a legibility bug rather than a
    reproducibility one. It also used to be built with
    ``getattr(..., field, None)``, which turned a renamed knob into a ``None``
    that reads exactly like "this knob was unset" (INSTRUMENT_AUDIT.md pattern
    1); the module-level guard above makes that a refusal instead.
    """
    activation = config.activation.model_dump(mode="json")
    return {
        "critical": {name: activation[name] for name in _ARM_CRITICAL_FIELDS},
        "all": activation,
    }


async def run_rig(opts: RigOptions, scorer: RigScorer | None = None) -> dict[str, Any]:
    """Run the rig end to end. Returns the result envelope; never raises on VOID."""
    if opts.fault not in FAULTS:
        raise ValueError(f"unknown fault {opts.fault!r}; expected one of {FAULTS}")
    scorer = scorer or GoldEpisodeReachability()
    opts.scratch_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = opts.scratch_dir / "corpus.json"
    if opts.reuse and corpus_path.exists():
        corpus = Corpus.from_json(corpus_path.read_text())
    else:
        corpus = build_corpus(
            repo_root=opts.repo_root,
            n=opts.n,
            seed=opts.seed,
            distractors_per_person=opts.distractors_per_person,
            filler=opts.filler,
        )
        corpus_path.write_text(corpus.to_json())

    tag_path = opts.scratch_dir / "tag_to_id.json"
    started = time.perf_counter()

    if not opts.reuse:
        for suffix in ("", "-wal", "-shm"):
            stale = opts.scratch_dir / f"brain.db{suffix}"
            if stale.exists():
                stale.unlink()
        manager, graph, config = await _open_brain(opts, {})
        tag_to_id = await _ingest(manager, corpus, opts)
        # Capture-time vector indexing is handed to a serialized background
        # lane and the capture returns before it runs. Measured 2026-09-04:
        # closing here without draining left 36/60 gold episodes without a
        # vector and five index tasks on a closed SQLiteVectorStore, and the
        # rig VOIDed itself on its own race rather than on the graph.
        await _drain_capture_indexing(manager)
        tag_path.write_text(json.dumps(tag_to_id, indent=1, sort_keys=True))
        await _close(manager, graph)
    else:
        tag_to_id = json.loads(tag_path.read_text())

    ingest_ms = round((time.perf_counter() - started) * 1000, 1)

    # --- pre-flight 1: producer + bridge + vector-index verification --------
    manager, graph, config = await _open_brain(opts, {})
    bridge_report = await preflight.verify_bridges(
        graph,
        group_id=opts.group_id,
        questions=corpus.questions,
        tag_to_id=tag_to_id,
    )
    vector_check = await preflight.vector_index_probe(
        manager._search,
        gold_episode_ids=[
            tag_to_id[q.gold_tag] for q in corpus.questions if q.gold_tag in tag_to_id
        ],
        group_id=opts.group_id,
    )
    embedding_provenance = {
        "model": config.embedding.local_model,
        "cache_path": os.environ.get("FASTEMBED_CACHE_PATH"),
        "provider": type(getattr(manager._search, "_provider", None)).__name__,
    }
    await _close(manager, graph)

    checks = [
        preflight.producer_probe(bridge_report, questions_requested=len(corpus.questions)),
        vector_check,
        preflight.scored_set_floor_probe(len(bridge_report.present), floor=MIN_SCORED_QUESTIONS),
    ]
    scored = [q for q in corpus.questions if q.qid in set(bridge_report.present)]

    envelope: dict[str, Any] = {
        "rig": "graph_kill_rig",
        "pre_registration_source": PRE_REGISTRATION_SOURCE,
        "options": {
            "n_requested": opts.n,
            "seed": opts.seed,
            "limit": opts.limit,
            "producer": opts.producer,
            "fault": opts.fault,
            "group_id": opts.group_id,
        },
        "corpus": corpus.provenance,
        "embedding": embedding_provenance,
        "ingest_ms": ingest_ms,
        "bridges": {
            "present": len(bridge_report.present),
            "missing": bridge_report.missing,
            "predicate_counts": bridge_report.predicate_counts,
        },
        "scorer": scorer.name,
    }

    if not all(check.passed for check in checks):
        return _void(envelope, checks, residual=None, arms_run={})

    # --- arm A -------------------------------------------------------------
    arm_overrides_a, arm_overrides_b = arm_overrides(opts.fault)

    manager, graph, config_a = await _open_brain(opts, arm_overrides_a)
    arm_a = await _run_and_score(manager, scored, tag_to_id, opts, scorer, keep_raw=True)
    await _close(manager, graph)

    residual_check, residual_rate = preflight.residual_probe(arm_a.scores)

    # --- arm B -------------------------------------------------------------
    manager, graph, config_b = await _open_brain(opts, arm_overrides_b)
    arm_b = await _run_and_score(manager, scored, tag_to_id, opts, scorer)
    await _close(manager, graph)

    checks.extend(
        [
            preflight.consumer_byte_probe(arm_b.runs),
            preflight.spread_gate_probe([*arm_a.runs, *arm_b.runs]),
            residual_check,
        ]
    )
    envelope["config"] = {
        "arm_A": _config_snapshot(config_a),
        "arm_B": _config_snapshot(config_b),
    }

    if not all(check.passed for check in checks):
        return _void(
            envelope,
            checks,
            residual=residual_rate,
            arms_run={"A": _diagnostics("A", arm_a.runs), "B": _diagnostics("B", arm_b.runs)},
        )

    # --- arm C: the kill arm ----------------------------------------------
    manager, graph, _ = await _open_brain(opts, arm_overrides_a)
    c_runs = await run_arm_c(
        manager,
        scored,
        arm_a.raw,
        {run.qid: run.ms for run in arm_a.runs},
        group_id=opts.group_id,
        limit=opts.limit,
    )
    await _close(manager, graph)

    a_result = _summarise("A", arm_a.runs, arm_a.scores)
    b_result = _summarise("B", arm_b.runs, arm_b.scores)
    c_variants = []
    for variant, runs in c_runs.items():
        scores = [scorer.score(q, run.rows, tag_to_id) for q, run in zip(scored, runs, strict=True)]
        c_variants.append(_summarise(f"C_{variant}", runs, scores))
    c_result = select_kill_arm(c_variants)

    verdict = evaluate(a_result, b_result, c_result, residual_rate=residual_rate or 0.0)

    envelope.update(
        {
            "status": "RESULT",
            "preflight": preflight.PreflightReport(
                checks=checks, residual_rate=residual_rate
            ).as_dict(),
            "arms": {
                "A": vars(a_result),
                "B": vars(b_result),
                "C_selected": vars(c_result),
                "C_variants": [vars(v) for v in c_variants],
            },
            "verdict": verdict.as_dict(),
            "caveat": (
                "Scores a retrieval list, not an agent's task outcome. GRAPH_THESIS.md §5: "
                "'Do not let it flip a default without an agent-task arm.'"
            ),
        }
    )
    return envelope


def _void(
    envelope: dict[str, Any],
    checks: list[preflight.Check],
    *,
    residual: float | None,
    arms_run: dict[str, Any],
) -> dict[str, Any]:
    report = preflight.PreflightReport(checks=checks, residual_rate=residual)
    envelope.update(
        {
            "status": "VOID",
            "preflight": report.as_dict(),
            "verdict": None,
            "arms": None,
            "arms_diagnostics_only": arms_run,
            "suppressed": (
                "Reachability@5/@10 and the SUCCESS/KILL verdict are withheld. A "
                "pre-flight check failed, so any arm delta measured here is a "
                "measurement of the broken precondition, not of the graph. "
                "GRAPH_THESIS.md §5: 'the run is VOID if any of these fails'."
            ),
            "refusal_reasons": report.failures,
        }
    )
    return envelope


async def _run_and_score(
    manager: Any,
    questions: list[Any],
    tag_to_id: dict[str, str],
    opts: RigOptions,
    scorer: RigScorer,
    *,
    keep_raw: bool = False,
) -> _ArmRecord:
    raw_by_qid: dict[str, list[dict[str, Any]]] = {}
    runs: list[QuestionRun] = []
    for question in questions:
        started = time.perf_counter()
        raw_rows = await manager.recall(
            question.query,
            opts.group_id,
            limit=opts.limit,
            record_access=False,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        if keep_raw:
            raw_by_qid[question.qid] = list(raw_rows)
        runs.append(
            QuestionRun(
                qid=question.qid,
                rows=[to_row(raw) for raw in raw_rows],
                ms=round(elapsed, 4),
                stage_timings=dict(manager.get_last_recall_stage_timings()),
            )
        )
    scores = [scorer.score(q, run.rows, tag_to_id) for q, run in zip(questions, runs, strict=True)]
    return _ArmRecord(runs=runs, scores=scores, raw=raw_by_qid)


async def _close(manager: Any, graph: Any) -> None:
    for closer in (
        getattr(manager, "close_runtime_resources", None),
        getattr(graph, "close", None),
    ):
        if closer is None:
            continue
        try:
            await closer()
        except Exception:  # noqa: BLE001 — teardown of a throwaway scratch brain
            pass


def purge_scratch(scratch_dir: Path) -> None:
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)


__all__ = ["FAULTS", "RigOptions", "purge_scratch", "run_rig"]
