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
import json
import sys
from pathlib import Path
from typing import Any

from engram.benchmark.depth.judge import (
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


async def _build_adapter(cfg, *, graph_on: bool, cache_path: str, extraction: str) -> tuple[_CachingAdapter, str]:
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
    return adapter, GROUP_PREFIX


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


async def run_persona(path: Path, top_k: int, cache_path: str, extraction: str, allow_narrow: bool) -> dict:
    persona = json.loads(path.read_text())
    pid = persona.get("persona_id", path.stem)

    # ---- build + freeze corpus ONCE under graph-ON (cache warm) ----
    cfg = EngramConfig()
    adapter_on, gid = await _build_adapter(cfg, graph_on=True, cache_path=cache_path, extraction=extraction)
    extractor_kind = _assert_clean_extractor(adapter_on, allow_narrow)
    ep_to_session = await _ingest_persona(adapter_on, gid, persona)
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
    cfg_off = EngramConfig()
    adapter_off, gid2 = await _build_adapter(cfg_off, graph_on=False, cache_path=cache_path, extraction=extraction)
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
    }


def _per_class_report(all_personas: list[dict], qtype: str) -> dict:
    """Headline per-class report with precondition / core-already-pass exclusions."""
    included_core: list[bool] = []
    included_depth: list[bool] = []
    excluded_precondition: list[str] = []
    excluded_core_pass: list[str] = []
    false_recall = 0
    n_seen = 0
    for per in all_personas:
        for q in per["queries"]:
            if q["type"] != qtype:
                continue
            n_seen += 1
            if q["depth_false_recall"] or q["core_false_recall"]:
                false_recall += 1
            # For multi_hop/current_value: a precondition (extraction-gap) failure is
            # not a traversal failure -> report separately, exclude from headline.
            if qtype in ("multi_hop", "current_value"):
                g = q.get("gate") or {}
                if not g.get("bridge_linked_to_answer"):
                    excluded_precondition.append(q["qid"])
                    continue
                # If the core can already answer it, the query does not isolate the
                # depth tier (the answer was term-reachable) -> exclude from headline.
                if q["core_pass"]:
                    excluded_core_pass.append(q["qid"])
                    continue
            included_core.append(q["core_pass"])
            included_depth.append(q["depth_pass"])

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


async def main(args) -> None:
    paths = [Path(p) for p in args.files]
    all_personas = []
    extractor_identity = "unknown"
    hits = misses = 0
    for p in paths:
        print(f"=== persona {p.name} ===", file=sys.stderr)
        per = await run_persona(p, args.top_k, args.cache, args.extraction, args.allow_narrow)
        all_personas.append(per)
        extractor_identity = per["extractor"]
        cs = per.get("cache_stats_ingest") or {}
        hits += cs.get("hits", 0)
        misses += cs.get("misses", 0)

    total = hits + misses
    report = {
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Deterministic depth-tier answer-quality eval.")
    ap.add_argument("files", nargs="+", help="graphthesis persona JSON files")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--cache", required=True, help="path to the persistent extraction-cache SQLite file")
    ap.add_argument("--extraction", default="auto", choices=["auto", "anthropic", "narrow"])
    ap.add_argument("--allow-narrow", action="store_true", help="accept the deterministic narrow extractor (determinism floor)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.extraction == "narrow":
        args.allow_narrow = True
    asyncio.run(main(args))
