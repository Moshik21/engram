#!/usr/bin/env python3
"""Deterministic depth-tier eval: does the depth (graph) tier win on the three
query classes where it structurally should — unnamed-bridge multi-hop,
current-value (newest-wins), and synthesis (multi-fact coverage)?

Trust-critical design:
  * ANSWER-QUALITY, not session-recall: a strict, deterministic substring judge
    (engram.benchmark.depth.judge) checks the SPECIFIC gold fact is present in the
    retrieved evidence and that forbidden/stale values are absent. No LLM judge,
    no embedding-containment leniency.
  * DETERMINISM by a persistent extraction cache: the corpus is projected ONCE
    under graph-ON with an ExtractionCache warm; the cached (byte-identical)
    verdicts are reused on every subsequent ingest and across the graph-ON/OFF
    arms, so the only stochastic stage (LLM extraction) is frozen. temperature=0
    is set in the extractor for cache-miss reproducibility.
  * HARD-FAIL on parse/api error: a PARSE_ERROR / API_ERROR during projection
    aborts the run (the documented silent-drop noise source) instead of quietly
    losing the answer-bearing episode.
  * PAIRED toggle on ONE frozen store: core-only arm = passage_first_entity_budget=0
    + weight_graph_structural=0.0 (episodes-only top-k); core+depth = graph-ON.
    Only the graph toggle differs, so the per-class delta is attributable to the
    depth tier. Per class we report (core_pass, depth_pass, delta, McNemar p,
    paired bootstrap CI). Precondition-failing (extraction-gap) and
    core-already-passing multi_hop/current_value queries are reported SEPARATELY
    and excluded from the headline delta.

Run (cold, real Haiku extraction — warms the cache):
  export ANTHROPIC_API_KEY=...
  ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native ENGRAM_HELIX__DATA_DIR=/tmp/depth \
    ENGRAM_ACTIVATION__EVIDENCE_EXTRACTION_ENABLED=false \
    uv run python scripts/benchmark_depth_tier.py data/graphthesis/*.json \
      --cache /tmp/depth_extraction_cache.sqlite --top-k 8 --output server/results/depth_tier.json

Re-run (warm cache => byte-identical, zero API): same command. Cache hit-rate=100%.

Determinism floor (no API key): add --extraction narrow (deterministic, zero API).
"""
from __future__ import annotations

# ruff: noqa: E501  (diagnostic script; long report/dict lines are fine)
import argparse
import asyncio
import contextlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engram.benchmark.depth.judge import (
    aggregate_repeated_runs,
    delta_bootstrap_ci,
    judge_current_value,
    judge_multi_hop,
    judge_synthesis,
    mcnemar_p,
    pass_rate_bootstrap_ci,
)
from engram.benchmark.longmemeval.adapter import (
    GROUP_PREFIX,
    EngramLongMemEvalAdapter,
)
from engram.config import EngramConfig
from engram.extraction.extraction_cache import ExtractionCache
from engram.utils import dates as _dates

# A fixed instant for the frozen-clock guard. Any constant works; the only
# requirement is that it is identical across runs. Mid-2026 keeps episode ages
# non-negative relative to the graphthesis conversation dates.
_FROZEN_EPOCH = 1_780_000_000.0  # 2026-05-28T... UTC
_FROZEN_DT = datetime.fromtimestamp(_FROZEN_EPOCH, tz=timezone.utc).replace(tzinfo=None)


@contextlib.contextmanager
def _frozen_clock() -> Iterator[None]:
    """Pin every wall-clock source the recall/ingest path reads to ONE instant.

    Removes wall-clock as a determinism confound: without this, two runs ingest +
    recall at different ``time.time()``, so ACT-R base-level decay (access_history
    age vs ``now``) and the entity ``updated_at`` sort tie-break differ. Freezing
    makes both runs observe identical episode ages and entity timestamps.

    Scope is strictly the eval process. We freeze ONLY ``time.time`` and
    ``engram.utils.dates.utc_now``/``utc_now_iso`` (the float-epoch ACT-R clock
    and the datetime entity-timestamp clock). ``time.perf_counter`` and the
    asyncio event-loop monotonic clock are untouched, so ``perf_counter``-based
    stage timeouts and ``asyncio.wait_for`` budgets keep measuring real elapsed
    time. Production recall behavior is unchanged — this guard exists only here.

    NOTE (measured 2026-05-29): freezing the clock did NOT eliminate the residual
    verdict flips. The dominant residual is native HNSW index-construction
    nondeterminism: recall against a *settled* store is byte-identical across
    processes, but building the same corpus into a fresh store twice yields HNSW
    graphs with slightly different neighbor orderings, perturbing RRF ranks on
    near-tied candidates. The flip SET is non-reproducible across two identical
    ``--repeat 2`` runs (confirming run-to-run build variation, not wall-clock).
    This guard is kept because it is the correct, behavior-preserving way to
    remove the wall-clock variable; closing the residual to 0 requires a
    deterministic (seeded) native HNSW build, tracked separately.
    """
    real_time = time.time
    time.time = lambda: _FROZEN_EPOCH  # type: ignore[assignment]
    _dates.set_now_override(_FROZEN_DT)
    try:
        yield
    finally:
        time.time = real_time  # type: ignore[assignment]
        _dates.set_now_override(None)


class _CachingAdapter(EngramLongMemEvalAdapter):
    """LongMemEval adapter whose extractor is wrapped in a persistent cache.

    The cache is injected at ``_build_extractor`` so the wrapped extractor is the
    one the GraphManager / EpisodeProjector capture at construction — the only
    correct injection point (post-hoc wrapping of ``_manager._extractor`` would
    not reach the projector's captured reference).
    """

    def __init__(self, *args: Any, cache_path: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cache_path = cache_path
        self.extraction_cache: ExtractionCache | None = None

    def _build_extractor(self) -> Any:
        inner = super()._build_extractor()
        self.extraction_cache = ExtractionCache(inner, self._cache_path)
        return self.extraction_cache


# --------------------------------------------------------------------------- #
# Ablation seam (item #6 phase-gating prep).                                    #
#                                                                               #
# A clean, declarative map from a consolidation phase / depth-operator NAME to  #
# the config field(s) that turn it OFF. --ablate <name> sets those fields False #
# on ONE arm's config before the adapter is built, so a later phase-gating loop #
# can attribute a per-phase delta by re-running with each name ablated. The     #
# write-side reflect phase is wired into the eval write path (run_persona): when #
# observer_reflect_enabled is True it runs a manual consolidation cycle after    #
# ingest so the synthesized observations persist as episodes visible to BOTH     #
# arms' episode-vector recall, making --ablate reflect a pure on/off contrast.   #
# Other phases still establish only the seam (names validated) for the loop.     #
# --------------------------------------------------------------------------- #
_ABLATABLE: dict[str, tuple[str, ...]] = {
    # consolidation phases (config _enabled flags)
    "merge": ("consolidation_merge_multi_signal_enabled", "consolidation_merge_llm_enabled"),
    "infer": ("consolidation_infer_pmi_enabled", "consolidation_infer_auto_validation_enabled", "consolidation_infer_llm_enabled"),
    "replay": ("consolidation_replay_enabled",),
    "dream": ("consolidation_dream_enabled", "consolidation_dream_associations_enabled"),
    "triage": ("triage_enabled",),
    "schema": ("schema_formation_enabled",),
    "reflect": ("observer_reflect_enabled",),
    "calibrate": ("consolidation_calibration_enabled",),
    "evidence_adjudication": ("evidence_extraction_enabled",),
    # depth-tier retrieval operators (the graph-side recall path under test)
    "mmr": ("mmr_enabled",),
    "reranker": ("reranker_enabled",),
    "community_spreading": ("community_spreading_enabled",),
    "graph_query_expansion": ("graph_query_expansion_enabled",),
}


def _apply_ablation(cfg: EngramConfig, ablate: list[str]) -> dict[str, list[str]]:
    """Turn the named phases/operators OFF on this config. Returns the applied
    name -> [fields] map (for the report). Unknown names hard-fail (a typo'd
    ablation that silently no-ops would corrupt the attribution)."""
    applied: dict[str, list[str]] = {}
    for name in ablate:
        fields = _ABLATABLE.get(name)
        if fields is None:
            raise SystemExit(
                f"ABORT: unknown --ablate target {name!r}. "
                f"Known: {', '.join(sorted(_ABLATABLE))}."
            )
        for f in fields:
            if not hasattr(cfg.activation, f):
                raise SystemExit(
                    f"ABORT: ablation field {f!r} for {name!r} is not a config "
                    "field (the config drifted from the ablation map)."
                )
            object.__setattr__(cfg.activation, f, False)
        applied[name] = list(fields)
    return applied


def _parse_date(s: str | None):
    from engram.benchmark.longmemeval.adapter import _parse_session_date

    try:
        return _parse_session_date(s) if s else None
    except Exception:
        return None


def _assert_clean_extractor(adapter: _CachingAdapter, allow_narrow: bool) -> str:
    """Assert the active (wrapped) extractor is the expected provider.

    With --extraction auto/anthropic the inner must be the Anthropic
    EntityExtractor (not a silent narrow fallback). With --extraction narrow the
    deterministic narrow adapter is accepted as the determinism floor.
    """
    from engram.extraction.extractor import EntityExtractor
    from engram.extraction.narrow_adapter import NarrowExtractorAdapter

    cache = adapter.extraction_cache
    assert cache is not None
    inner = cache.inner
    kind = type(inner).__name__
    if isinstance(inner, EntityExtractor):
        return kind
    if allow_narrow and isinstance(inner, NarrowExtractorAdapter):
        return kind
    raise SystemExit(
        f"ABORT: active extractor is {kind}, not the Anthropic EntityExtractor. "
        "ANTHROPIC_API_KEY is likely missing/shadowed and extraction silently fell "
        "back to narrow. Export the real key, or pass --extraction narrow to use "
        "the deterministic narrow floor explicitly."
    )


async def _ingest_persona(adapter: _CachingAdapter, group_id: str, persona: dict) -> dict[str, str]:
    """Ingest each session once; HARD-FAIL on extraction parse/api errors.

    A parse/api error surfaces as a ``ProjectionError`` whose message starts with
    ``extractor_parse_error`` / ``extractor_api_error`` / ``extractor_truncated``.
    The legacy graphthesis harness swallowed these (a silent drop of the
    answer-bearing episode — the documented noise source). Here we abort instead.
    """
    from engram.ingestion.projection_execution import ProjectionError

    ep_to_session: dict[str, str] = {}
    for sess in persona["sessions"]:
        sid = sess["session_id"]
        date = sess.get("date")
        turns = sess.get("turns", [])
        body = "\n".join(f"{t.get('role', 'user')}: {t.get('text', '')}" for t in turns)
        content = (f"[Conversation from {date}]\n{body}" if date else body).strip()
        if not content:
            continue
        ep_id = await adapter._manager.store_episode(
            content, group_id=group_id, source=f"gt:{sid}", session_id=sid,
            conversation_date=_parse_date(date),
        )
        ep_to_session[ep_id] = sid
        try:
            await adapter._manager.project_episode(ep_id, group_id=group_id)
        except ProjectionError as e:
            msg = str(e)
            if msg.startswith(("extractor_parse_error", "extractor_api_error", "extractor_truncated")):
                raise SystemExit(
                    f"ABORT: {msg} on session {sid} of {persona.get('persona_id')}. "
                    "Refusing to build a corpus with silently-dropped episodes. "
                    "Re-run (the cache stores only successes, so a transient failure won't recur)."
                ) from e
            raise
    return ep_to_session


async def _build_adapter(
    cfg, *, graph_on: bool, cache_path: str, extraction: str, ablate: list[str] | None = None
) -> tuple[_CachingAdapter, str, dict[str, list[str]]]:
    # Determinism: disable Thompson Sampling exploration for the eval (its RNG is
    # unseeded in production and perturbs near-tied scores every recall). The
    # production default is unchanged; this only pins the measurement arm.
    cfg.activation.ts_enabled = False
    if graph_on:
        # DEPTH arm: mirror the rework profile's depth-tier entity budget so the
        # graph can actually surface entities into the top-k. The eval built a
        # bare EngramConfig() (integration_profile="off"), whose
        # passage_first_entity_budget default is 0 -- so the depth arm was
        # silently EPISODE-ONLY (zero entity slots) and never tested the depth
        # tier at all. Budget 3 = the rework depth tier. The core arm
        # (use_graph=False) keeps budget 0 (set by the adapter).
        cfg.activation.passage_first_entity_budget = 3
    applied_ablation = _apply_ablation(cfg, ablate) if ablate else {}
    adapter = _CachingAdapter(
        cfg=cfg.activation,
        extraction_mode=extraction,
        embedding_provider="local",
        reranker_provider="local",
        use_graph=graph_on,
        shared_group=True,
        cache_path=cache_path,
        top_k=8,
    )
    await adapter._ensure_initialized()
    await adapter._setup_manager("depthtier")
    return adapter, GROUP_PREFIX, applied_ablation


def _evidence_texts(results: list[dict], *, newest_first: bool) -> list[str]:
    """Collect raw evidence texts from recall results.

    For current_value we sort episode evidence newest-first (matching the
    adapter's knowledge-update sort) so the judge can enforce newest-wins ranking.
    """
    entity_ev: list[str] = []
    episode_ev: list[tuple[str, str]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        if "entity" in r:
            ent = r["entity"]
            name = ent.get("name", "")
            summary = ent.get("summary", "")
            if summary:
                entity_ev.append(f"{name}: {summary}")
            for rel in ent.get("relationships", []):
                src = rel.get("source_id") or rel.get("source", "")
                pred = rel.get("predicate", "")
                tgt = rel.get("target_id") or rel.get("target", "")
                if pred:
                    entity_ev.append(f"{src} {pred} {tgt}")
        else:
            ep = r.get("episode") or {}
            content = ep.get("content", "")
            if content:
                date = ep.get("conversation_date") or ep.get("created_at") or ""
                episode_ev.append((str(date), content))
    if newest_first:
        episode_ev.sort(key=lambda x: x[0], reverse=True)
    # Episodes first (raw text carries the literal gold wording), then entities.
    return [t for _, t in episode_ev] + entity_ev


async def _precondition_gate(adapter: _CachingAdapter, group_id: str, persona: dict, session_to_ep: dict[str, list[str]]) -> dict[str, dict]:
    """For multi_hop / current_value: is the bridge entity in the graph AND linked
    (via source-episode provenance) to the answer session? If not, the graph
    cannot connect it — an extraction gap, not a traversal failure."""
    graph = adapter._graph_store
    ents = await graph.find_entities(group_id=group_id, limit=1000)
    by_lower: dict[str, Any] = {}
    for e in ents:
        by_lower.setdefault(e.name.lower(), e)
    gate: dict[str, dict] = {}
    for q in persona["queries"]:
        if q["type"] not in ("multi_hop", "current_value"):
            continue
        bridge = (q.get("bridge_entity") or "").lower().strip()
        bridge_ent = by_lower.get(bridge)
        linked = False
        if bridge_ent is not None:
            srcs = set(bridge_ent.source_episode_ids or [])
            ans_eps = {e for sid in q.get("answer_session_ids", []) for e in session_to_ep.get(sid, [])}
            linked = bool(srcs & ans_eps)
        gate[q["qid"]] = {"bridge_in_graph": bridge_ent is not None, "bridge_linked_to_answer": linked}
    return gate


def _judge(q: dict, evidence: list[str]):
    qtype = q["type"]
    if qtype == "multi_hop":
        return judge_multi_hop(
            q["qid"], evidence, answer=q["answer"],
            accepted_forms=q.get("accepted_forms"), forbidden=q.get("forbidden"),
        )
    if qtype == "current_value":
        return judge_current_value(
            q["qid"], evidence, answer=q["answer"], forbidden=q.get("forbidden", []),
            accepted_forms=q.get("accepted_forms"), evidence_is_ordered=True,
        )
    if qtype == "synthesis":
        return judge_synthesis(
            q["qid"], evidence, required_facts=q["required_facts"],
            forbidden_facts=q.get("forbidden_facts"),
        )
    return None


async def _close(adapter: _CachingAdapter) -> None:
    try:
        if adapter.extraction_cache is not None:
            adapter.extraction_cache.close()
    except Exception:
        pass
    try:
        await adapter.close()
    except Exception:
        pass


def _cfg_for(data_dir: str | None) -> EngramConfig:
    """Build a config, optionally pinning the native store dir for a fresh-store
    repeat. ``--repeat`` gives each run its own dir so the stores are independent
    while the warm ExtractionCache is shared via ``cache_path``."""
    cfg = EngramConfig()
    if data_dir:
        object.__setattr__(cfg.helix, "data_dir", data_dir)
    return cfg


async def run_persona(
    path: Path,
    top_k: int,
    cache_path: str,
    extraction: str,
    allow_narrow: bool,
    *,
    ablate: list[str] | None = None,
    data_dir: str | None = None,
    reflect: bool = False,
) -> dict:
    persona = json.loads(path.read_text())
    pid = persona.get("persona_id", path.stem)

    # ---- build + freeze corpus ONCE under graph-ON (cache warm) ----
    # Ablation toggles depth-side phases/operators OFF on the graph-ON (depth)
    # arm only; the core arm stays the fixed baseline so the delta is attributable.
    cfg = _cfg_for(data_dir)
    # Write-side reflect (#3) ships dark: config post-init force-sets
    # observer_reflect_enabled=False regardless of env, so --reflect must mutate
    # the field AFTER construction (same pattern as passage_first_entity_budget).
    # Observations land on the depth store both arms share, so this is a pure
    # reflect-on/off contrast (and --ablate reflect can still force it back off).
    if reflect:
        object.__setattr__(cfg.activation, "observer_reflect_enabled", True)
    adapter_on, gid, applied_ablation = await _build_adapter(
        cfg, graph_on=True, cache_path=cache_path, extraction=extraction, ablate=ablate
    )
    extractor_kind = _assert_clean_extractor(adapter_on, allow_narrow)
    ep_to_session = await _ingest_persona(adapter_on, gid, persona)
    # Write-side reflect (item #3): when the phase is enabled, run a manual
    # consolidation cycle BEFORE recall so synthesized observation episodes are
    # embedded into the SAME frozen store the core arm reuses. Manual trigger
    # bypasses tiering and runs the cold-tier reflect phase fully. Off by default
    # (and ablatable via --ablate reflect), so the baseline arm is unchanged.
    if cfg.activation.observer_reflect_enabled:
        await adapter_on._manager.trigger_consolidation_cycle(
            group_id=gid, trigger="manual", dry_run=False
        )
    session_to_ep: dict[str, list[str]] = {}
    for ep, sid in ep_to_session.items():
        session_to_ep.setdefault(sid, []).append(ep)
    gate = await _precondition_gate(adapter_on, gid, persona, session_to_ep)
    cache_stats_ingest = adapter_on.extraction_cache.stats() if adapter_on.extraction_cache else {}

    # depth (graph-ON) recall
    depth_rows: dict[str, list[str]] = {}
    for q in persona["queries"]:
        nf = q["type"] == "current_value"
        res = await adapter_on._manager.recall(q["question"], group_id=gid, limit=top_k, record_access=False)
        depth_rows[q["qid"]] = _evidence_texts(res, newest_first=nf)
    await _close(adapter_on)

    # ---- core-only arm: SAME frozen store, no re-ingest, graph toggled OFF ----
    cfg_off = _cfg_for(data_dir)
    adapter_off, gid2, _ = await _build_adapter(cfg_off, graph_on=False, cache_path=cache_path, extraction=extraction)
    core_rows: dict[str, list[str]] = {}
    for q in persona["queries"]:
        nf = q["type"] == "current_value"
        res = await adapter_off._manager.recall(q["question"], group_id=gid2, limit=top_k, record_access=False)
        core_rows[q["qid"]] = _evidence_texts(res, newest_first=nf)
    await _close(adapter_off)

    # ---- judge both arms ----
    out_queries = []
    for q in persona["queries"]:
        if q["type"] == "single_hop_control":
            continue
        depth_v = _judge(q, depth_rows[q["qid"]])
        core_v = _judge(q, core_rows[q["qid"]])
        g = gate.get(q["qid"])
        out_queries.append({
            "qid": q["qid"], "type": q["type"], "bridge": q.get("bridge_entity"),
            "answer": q.get("answer") or q.get("required_facts"),
            "core_pass": core_v.passed, "depth_pass": depth_v.passed,
            "core_coverage": core_v.coverage, "depth_coverage": depth_v.coverage,
            "depth_false_recall": depth_v.false_recall, "core_false_recall": core_v.false_recall,
            "depth_notes": depth_v.notes, "core_notes": core_v.notes,
            "gate": g,
        })
    return {
        "persona_id": pid, "extractor": extractor_kind, "num_sessions": len(persona["sessions"]),
        "top_k": top_k, "cache_stats_ingest": cache_stats_ingest, "queries": out_queries,
        "ablation": applied_ablation,
    }


def _headline_filter(all_personas: list[dict], qtype: str) -> dict:
    """Apply the headline inclusion rules for one query class to a set of personas.

    Precondition (extraction-gap) failures and core-already-passing
    multi_hop/current_value queries are reported separately and excluded from the
    headline delta. Returns the per-query headline outcomes (qid-keyed verdict
    maps so a repeated-run aggregator can pool by qid) plus the exclusion lists.
    """
    core_verdicts: dict[str, bool] = {}
    depth_verdicts: dict[str, bool] = {}
    excluded_precondition: list[str] = []
    excluded_core_pass: list[str] = []
    false_recall = 0
    n_seen = 0
    for per in all_personas:
        # Qualify the verdict key by persona: every persona reuses q1..q6 / q11..
        # q15, so a bare-qid key collapses cross-persona queries (last-writer-wins),
        # silently discarding real per-persona depth passes from the headline pool
        # AND from the repeat aggregator / flip count (which both pool by these
        # keys). Persona-qualifying is the single fix point for all three.
        pid = per.get("persona_id", "?")
        for q in per["queries"]:
            if q["type"] != qtype:
                continue
            n_seen += 1
            qkey = f"{pid}:{q['qid']}"
            if q["depth_false_recall"] or q["core_false_recall"]:
                false_recall += 1
            # For multi_hop/current_value: a precondition (extraction-gap) failure is
            # not a traversal failure -> report separately, exclude from headline.
            if qtype in ("multi_hop", "current_value"):
                g = q.get("gate") or {}
                if not g.get("bridge_linked_to_answer"):
                    excluded_precondition.append(qkey)
                    continue
                # If the core can already answer it, the query does not isolate the
                # depth tier (the answer was term-reachable) -> exclude from headline.
                if q["core_pass"]:
                    excluded_core_pass.append(qkey)
                    continue
            core_verdicts[qkey] = bool(q["core_pass"])
            depth_verdicts[qkey] = bool(q["depth_pass"])
    return {
        "core_verdicts": core_verdicts,
        "depth_verdicts": depth_verdicts,
        "excluded_precondition": excluded_precondition,
        "excluded_core_already_passes": excluded_core_pass,
        "false_recall_count": false_recall,
        "n_seen": n_seen,
    }


def _per_class_report(all_personas: list[dict], qtype: str) -> dict:
    """Headline per-class report with precondition / core-already-pass exclusions."""
    f = _headline_filter(all_personas, qtype)
    included_core = list(f["core_verdicts"].values())
    included_depth = list(f["depth_verdicts"].values())
    excluded_precondition = f["excluded_precondition"]
    excluded_core_pass = f["excluded_core_already_passes"]
    false_recall = f["false_recall_count"]
    n_seen = f["n_seen"]

    n = len(included_core)
    core_rate = sum(included_core) / n if n else 0.0
    depth_rate = sum(included_depth) / n if n else 0.0
    delta_ci = delta_bootstrap_ci(included_core, included_depth)
    win = n > 0 and (delta_ci[0] > 0 or delta_ci[1] < 0)
    return {
        "query_type": qtype,
        "n_seen": n_seen,
        "n_headline": n,
        "core_pass_rate": round(core_rate, 4),
        "depth_pass_rate": round(depth_rate, 4),
        "delta": round(depth_rate - core_rate, 4),
        "mcnemar_p": round(mcnemar_p(included_core, included_depth), 4),
        "delta_ci_95": [round(delta_ci[0], 4), round(delta_ci[1], 4)],
        "core_ci_95": [round(c, 4) for c in pass_rate_bootstrap_ci(included_core)],
        "depth_ci_95": [round(c, 4) for c in pass_rate_bootstrap_ci(included_depth)],
        "ci_excludes_zero": win,
        "false_recall_count": false_recall,
        "excluded_precondition": excluded_precondition,
        "excluded_core_already_passes": excluded_core_pass,
    }


def _repeat_class_report(run_personas: list[list[dict]], qtype: str) -> dict:
    """Aggregate one query class across N repeated runs.

    ``run_personas[i]`` is the persona list for run i. Each run is headline-
    filtered independently, then the per-run pass-rates + per-query verdict maps
    are handed to the deterministic aggregator (mean/std + pooled bootstrap CI +
    McNemar + verdict-flip counts)."""
    per_run: list[dict] = []
    for personas in run_personas:
        f = _headline_filter(personas, qtype)
        cv, dv = f["core_verdicts"], f["depth_verdicts"]
        n = len(cv)
        per_run.append({
            "core_pass_rate": (sum(cv.values()) / n) if n else 0.0,
            "depth_pass_rate": (sum(dv.values()) / n) if n else 0.0,
            "core_verdicts": cv,
            "depth_verdicts": dv,
        })
    agg = aggregate_repeated_runs(per_run)
    agg["query_type"] = qtype
    return agg


def _markdown(report: dict) -> str:
    lines = ["# Depth-Tier Eval Report", ""]
    lines.append(f"- Extractor: {report['extractor_identity']}")
    lines.append(f"- Ingest cache hit-rate: {report['cache_hit_rate']:.2%} (hits={report['cache_hits']}, misses={report['cache_misses']})")
    lines.append("")
    lines.append("Headline = per-class core-only vs core+depth pass-rate on the FROZEN corpus.")
    lines.append("A depth-tier win = delta CI excludes 0 on multi_hop OR current_value.")
    lines.append("")
    lines.append("| class | n (headline) | core | depth | delta | McNemar p | delta CI95 | win | false-recall | excl(precond/core-pass) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in report["per_class"]:
        lines.append(
            f"| {c['query_type']} | {c['n_headline']}/{c['n_seen']} | {c['core_pass_rate']:.2f} | "
            f"{c['depth_pass_rate']:.2f} | {c['delta']:+.2f} | {c['mcnemar_p']:.3f} | "
            f"[{c['delta_ci_95'][0]:+.2f},{c['delta_ci_95'][1]:+.2f}] | {'YES' if c['ci_excludes_zero'] else 'no'} | "
            f"{c['false_recall_count']} | {len(c['excluded_precondition'])}/{len(c['excluded_core_already_passes'])} |"
        )
    return "\n".join(lines) + "\n"


def _repeat_markdown(report: dict) -> str:
    lines = ["# Depth-Tier Eval Report (repeated runs)", ""]
    lines.append(f"- Extractor: {report['extractor_identity']}")
    lines.append(f"- Runs: {report['n_runs']} (fresh stores, shared warm ExtractionCache)")
    if report.get("ablation"):
        lines.append(f"- Ablated (depth arm): {report['ablation']}")
    lines.append("")
    lines.append("Per-class pass-rate mean +/- std over runs; delta = depth - core (paired).")
    lines.append("`flips` = per-query verdict disagreement across runs (0 => deterministic).")
    lines.append("")
    lines.append("| class | core mean+/-std | depth mean+/-std | delta mean+/-std | delta CI95 | McNemar p | win | core flips | depth flips |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in report["per_class"]:
        lines.append(
            f"| {c['query_type']} | {c['core_pass_rate_mean']:.2f}+/-{c['core_pass_rate_std']:.2f} | "
            f"{c['depth_pass_rate_mean']:.2f}+/-{c['depth_pass_rate_std']:.2f} | "
            f"{c['delta_mean']:+.2f}+/-{c['delta_std']:.2f} | "
            f"[{c['delta_ci_95'][0]:+.2f},{c['delta_ci_95'][1]:+.2f}] | {c['mcnemar_p']:.3f} | "
            f"{'YES' if c['ci_excludes_zero'] else 'no'} | {c['core_flips']} | {c['depth_flips']} |"
        )
    return "\n".join(lines) + "\n"


def _single_report(all_personas: list[dict], extractor_identity: str, hits: int, misses: int) -> dict:
    total = hits + misses
    return {
        "extractor_identity": extractor_identity,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate": (hits / total) if total else 0.0,
        "per_class": [
            _per_class_report(all_personas, "multi_hop"),
            _per_class_report(all_personas, "current_value"),
            _per_class_report(all_personas, "synthesis"),
        ],
        "personas": all_personas,
    }


async def _run_once(args, *, data_dir: str | None = None) -> tuple[list[dict], str, int, int]:
    """Run the paired eval over every persona once. Returns (personas, extractor,
    cache_hits, cache_misses)."""
    all_personas: list[dict] = []
    extractor_identity = "unknown"
    hits = misses = 0
    for p in (Path(p) for p in args.files):
        print(f"=== persona {p.name} ===", file=sys.stderr)
        per = await run_persona(
            p, args.top_k, args.cache, args.extraction, args.allow_narrow,
            ablate=args.ablate, data_dir=data_dir, reflect=args.reflect,
        )
        all_personas.append(per)
        extractor_identity = per["extractor"]
        cs = per.get("cache_stats_ingest") or {}
        hits += cs.get("hits", 0)
        misses += cs.get("misses", 0)
    return all_personas, extractor_identity, hits, misses


async def main(args) -> None:
    # Pin a single deterministic wall clock across every ingest + recall in this
    # process (unless explicitly disabled) so episode ages and entity timestamps
    # are identical between runs and ACT-R near-ties cannot flip on wall-clock.
    guard = contextlib.nullcontext() if args.no_frozen_clock else _frozen_clock()
    with guard:
        await _main_inner(args)


async def _main_inner(args) -> None:
    if args.repeat <= 1:
        # ---- default: single run, unchanged behavior ----
        all_personas, extractor_identity, hits, misses = await _run_once(args)
        report = _single_report(all_personas, extractor_identity, hits, misses)
        print(json.dumps(report, indent=2))
        print("\n=== SUMMARY ===", file=sys.stderr)
        for c in report["per_class"]:
            print(
                f"{c['query_type']:>14}: core {c['core_pass_rate']:.2f} -> depth {c['depth_pass_rate']:.2f} "
                f"(delta {c['delta']:+.2f}, CI95 {c['delta_ci_95']}, win={c['ci_excludes_zero']}, "
                f"n_headline={c['n_headline']}/{c['n_seen']})",
                file=sys.stderr,
            )
        print(f"cache hit-rate (ingest): {report['cache_hit_rate']:.2%}", file=sys.stderr)
        if args.output:
            out = Path(args.output)
            out.write_text(json.dumps(report, indent=2))
            Path(str(out).replace(".json", ".md")).write_text(_markdown(report))
        return

    # ---- --repeat N: N fresh stores, shared warm cache, aggregate ----
    base_dir = os.environ.get("ENGRAM_HELIX__DATA_DIR")
    run_personas: list[list[dict]] = []
    extractor_identity = "unknown"
    ablation: dict[str, list[str]] = {}
    for i in range(args.repeat):
        # Each run gets its own fresh store dir so the stores are independent;
        # the warm ExtractionCache (--cache) is shared, so extraction stays frozen
        # and only retrieval/activation nondeterminism can produce a flip.
        if base_dir:
            data_dir = f"{base_dir.rstrip('/')}_run{i}"
        else:
            data_dir = str(Path(tempfile.gettempdir()) / f"depthtier_run{i}")
        print(f"=== repeat {i + 1}/{args.repeat} (store={data_dir}) ===", file=sys.stderr)
        personas, extractor_identity, _, _ = await _run_once(args, data_dir=data_dir)
        run_personas.append(personas)
        if personas and personas[0].get("ablation"):
            ablation = personas[0]["ablation"]

    report = {
        "mode": "repeat",
        "n_runs": args.repeat,
        "extractor_identity": extractor_identity,
        "ablation": ablation,
        "per_class": [
            _repeat_class_report(run_personas, "multi_hop"),
            _repeat_class_report(run_personas, "current_value"),
            _repeat_class_report(run_personas, "synthesis"),
        ],
        "runs": run_personas,
    }
    print(json.dumps(report, indent=2))
    print(f"\n=== REPEAT SUMMARY ({args.repeat} runs) ===", file=sys.stderr)
    for c in report["per_class"]:
        print(
            f"{c['query_type']:>14}: core {c['core_pass_rate_mean']:.2f}+/-{c['core_pass_rate_std']:.2f} "
            f"-> depth {c['depth_pass_rate_mean']:.2f}+/-{c['depth_pass_rate_std']:.2f} "
            f"(delta {c['delta_mean']:+.2f}+/-{c['delta_std']:.2f}, CI95 {c['delta_ci_95']}, "
            f"win={c['ci_excludes_zero']}, flips core={c['core_flips']}/depth={c['depth_flips']})",
            file=sys.stderr,
        )
    if ablation:
        print(f"ablated (depth arm): {ablation}", file=sys.stderr)
    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(report, indent=2))
        Path(str(out).replace(".json", ".md")).write_text(_repeat_markdown(report))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Deterministic depth-tier answer-quality eval.")
    ap.add_argument("files", nargs="+", help="graphthesis persona JSON files")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--cache", required=True, help="path to the persistent extraction-cache SQLite file")
    ap.add_argument("--extraction", default="auto", choices=["auto", "anthropic", "narrow"])
    ap.add_argument("--allow-narrow", action="store_true", help="accept the deterministic narrow extractor (determinism floor)")
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--repeat", type=int, default=1,
        help="run the paired eval N times on FRESH stores (sharing the warm cache) "
             "and aggregate per-class mean/std + paired delta CI + McNemar + verdict-flip counts",
    )
    ap.add_argument(
        "--ablate", action="append", default=None, metavar="PHASE",
        help="turn a consolidation phase / depth operator OFF on the DEPTH arm "
             "(repeatable). Known: " + ", ".join(sorted(_ABLATABLE)),
    )
    ap.add_argument(
        "--reflect", action="store_true",
        help="run the write-side Observer/Reflector phase (#3) before recall so "
             "synthesized observation episodes are present on the shared store "
             "(measures reflect-on vs the default reflect-off baseline)",
    )
    ap.add_argument(
        "--no-frozen-clock", action="store_true",
        help="do NOT pin time.time/utc_now to a fixed instant (default: pinned "
             "for determinism; perf_counter timeouts are unaffected)",
    )
    args = ap.parse_args()
    if args.extraction == "narrow":
        args.allow_narrow = True
    asyncio.run(main(args))
