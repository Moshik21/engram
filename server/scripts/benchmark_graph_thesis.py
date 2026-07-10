#!/usr/bin/env python3
"""Clean graph-thesis eval: does the knowledge graph help MULTI-HOP recall?

Trust-critical design (lessons from prior confounded runs):
  * ONE shared group_id per persona — ingest the persona's sessions ONCE, then
    query the accumulated graph. No per-question re-ingest, no cross-group dedup.
  * VERIFIED clean extraction — assert the active extractor is the Anthropic
    EntityExtractor (not a silent narrow fallback) before ingesting.
  * PRECONDITION GATE — after ingest, for every multi_hop check the graph actually
    links the named bridge entity to the answer session. If not, the graph CANNOT
    connect the dots regardless of traversal, so we flag it (extraction gap) rather
    than blaming graph retrieval.
  * Metric = answer-session recall@K (did recall surface the answer-bearing
    session's episode?), graph ON vs OFF, same store. Multi-hop is where the graph
    should help; controls are where it must stay neutral.

Reuses the validated EngramLongMemEvalAdapter setup (stores, reranker, fail-fast
vector check) + the real GraphManager recall pipeline. No new retrieval path.

Run (native, LLM extraction):
  export ANTHROPIC_API_KEY=...   # real key, must not be shadowed
  ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native ENGRAM_HELIX__DATA_DIR=/tmp/gt \
    ENGRAM_ACTIVATION__EVIDENCE_EXTRACTION_ENABLED=false \
    uv run python scripts/benchmark_graph_thesis.py data/graphthesis/*.json --top-k 5
"""

from __future__ import annotations

# ruff: noqa: E501  (diagnostic script; long report/dict lines are fine)
import argparse
import asyncio
import json
import sys
from pathlib import Path


def _parse_date(s):
    from engram.benchmark.longmemeval.adapter import _parse_session_date

    try:
        return _parse_session_date(s) if s else None
    except Exception:
        return None


async def _build_manager(activation_cfg, graph_on: bool):
    """Build the validated adapter + manager for one graph mode, sharing the store."""
    from engram.benchmark.longmemeval.adapter import GROUP_PREFIX, EngramLongMemEvalAdapter

    adapter = EngramLongMemEvalAdapter(
        cfg=activation_cfg,
        extraction_mode="auto",  # auto -> Anthropic when key present
        embedding_provider="local",
        reranker_provider="local",
        use_graph=graph_on,
        shared_group=True,
    )
    await adapter._ensure_initialized()
    await adapter._setup_manager("graphthesis")
    return adapter, GROUP_PREFIX


def _assert_llm_extractor(adapter) -> str:
    from engram.extraction.extractor import EntityExtractor

    ex = adapter._manager._extractor
    kind = type(ex).__name__
    if not isinstance(ex, EntityExtractor):
        raise SystemExit(
            f"ABORT: active extractor is {kind}, not the Anthropic EntityExtractor. "
            "The ANTHROPIC_API_KEY is likely missing/shadowed and extraction silently "
            "fell back to narrow. Export the real key before running."
        )
    return kind


async def _ingest_persona(adapter, group_id, persona) -> dict[str, str]:
    """Ingest each session once; return episode_id -> session_id map."""
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
            content,
            group_id=group_id,
            source=f"gt:{sid}",
            session_id=sid,
            conversation_date=_parse_date(date),
        )
        ep_to_session[ep_id] = sid
        try:
            await adapter._manager.project_episode(ep_id, group_id=group_id)
        except Exception as e:
            print(f"  WARN: projection failed for {sid}: {e}", file=sys.stderr)
    return ep_to_session


async def _precondition_gate(adapter, group_id, persona) -> dict[str, bool]:
    """For each multi_hop: is the bridge entity in the graph AND linked (via
    source-episode provenance) to the answer session? If not, the graph cannot
    connect it (extraction gap), so the query is unfair to the traversal."""
    graph = adapter._graph_store
    ents = await graph.find_entities(group_id=group_id, limit=500)
    by_lower = {}
    for e in ents:
        by_lower.setdefault(e.name.lower(), e)
    # map session_id -> episode_id(s) we ingested
    gate = {}
    for q in persona["queries"]:
        if q["type"] != "multi_hop":
            continue
        bridge = (q.get("bridge_entity") or "").lower().strip()
        bridge_ent = by_lower.get(bridge)
        linked = False
        if bridge_ent is not None:
            srcs = set(bridge_ent.source_episode_ids or [])
            # answer sessions -> their episode ids (from our ingest map stored on q)
            ans_eps = set(q.get("_answer_episode_ids", []))
            linked = bool(srcs & ans_eps)
        gate[q["qid"]] = {
            "bridge_in_graph": bridge_ent is not None,
            "bridge_linked_to_answer": linked,
        }
    return gate


def _retrieved_sessions(results, ep_to_session, k):
    out = []
    for r in results[:k]:
        if not isinstance(r, dict):
            continue
        ep = r.get("episode") or {}
        eid = ep.get("id") or r.get("node_id")
        sid = ep.get("session_id") or ep_to_session.get(eid)
        if sid:
            out.append(sid)
    return out


async def run_persona(path: Path, top_k: int) -> dict:
    from engram.config import EngramConfig

    persona = json.loads(path.read_text())
    pid = persona.get("persona_id", path.stem)

    # ---- graph ON: ingest once + recall ----
    cfg = EngramConfig()
    adapter_on, gid = await _build_manager(cfg.activation, graph_on=True)
    extractor_kind = _assert_llm_extractor(adapter_on)
    ep_to_session = await _ingest_persona(adapter_on, gid, persona)
    session_to_ep = {}
    for ep, sid in ep_to_session.items():
        session_to_ep.setdefault(sid, []).append(ep)
    for q in persona["queries"]:
        q["_answer_episode_ids"] = [
            e for sid in q.get("answer_session_ids", []) for e in session_to_ep.get(sid, [])
        ]
    gate = await _precondition_gate(adapter_on, gid, persona)

    rows = []
    for q in persona["queries"]:
        res_on = await adapter_on._manager.recall(
            q["question"], group_id=gid, limit=top_k, record_access=False
        )
        ret_on = _retrieved_sessions(res_on, ep_to_session, top_k)
        rows.append({"q": q, "ret_on": ret_on})
    await _close(adapter_on)

    # ---- graph OFF: same store, no re-ingest ----
    cfg_off = EngramConfig()
    adapter_off, gid2 = await _build_manager(cfg_off.activation, graph_on=False)
    for row in rows:
        q = row["q"]
        res_off = await adapter_off._manager.recall(
            q["question"], group_id=gid2, limit=top_k, record_access=False
        )
        row["ret_off"] = _retrieved_sessions(res_off, ep_to_session, top_k)
    await _close(adapter_off)

    # ---- score ----
    def hit(ret, ans):
        return bool(set(ret) & set(ans))

    out_queries = []
    for row in rows:
        q = row["q"]
        ans = q.get("answer_session_ids", [])
        out_queries.append(
            {
                "qid": q["qid"],
                "type": q["type"],
                "bridge": q.get("bridge_entity"),
                "answer_sessions": ans,
                "on_hit": hit(row["ret_on"], ans),
                "off_hit": hit(row["ret_off"], ans),
                "ret_on": row["ret_on"],
                "ret_off": row["ret_off"],
                "gate": gate.get(q["qid"]),
            }
        )
    return {
        "persona_id": pid,
        "extractor": extractor_kind,
        "num_sessions": len(persona["sessions"]),
        "top_k": top_k,
        "queries": out_queries,
    }


async def _close(adapter):
    try:
        await adapter.close()
    except Exception:
        pass


async def main(args):
    paths = [Path(p) for p in args.files]
    all_personas = []
    for p in paths:
        print(f"=== persona {p.name} ===", file=sys.stderr)
        all_personas.append(await run_persona(p, args.top_k))

    # aggregate
    def agg(qtype):
        on = off = n = 0
        for per in all_personas:
            for q in per["queries"]:
                if q["type"] != qtype:
                    continue
                n += 1
                on += int(q["on_hit"])
                off += int(q["off_hit"])
        return on, off, n

    mh_on, mh_off, mh_n = agg("multi_hop")
    c_on, c_off, c_n = agg("single_hop_control")
    report = {
        "personas": all_personas,
        "summary": {
            "multi_hop": {"graph_on_hits": mh_on, "graph_off_hits": mh_off, "total": mh_n},
            "control": {"graph_on_hits": c_on, "graph_off_hits": c_off, "total": c_n},
        },
    }
    print(json.dumps(report, indent=2))
    print("\n=== SUMMARY ===", file=sys.stderr)
    print(
        f"multi_hop answer-session recall: graph-OFF {mh_off}/{mh_n} -> graph-ON {mh_on}/{mh_n}",
        file=sys.stderr,
    )
    print(
        f"control   answer-session recall: graph-OFF {c_off}/{c_n} -> graph-ON {c_on}/{c_n}",
        file=sys.stderr,
    )
    # precondition transparency
    bad = [
        (per["persona_id"], q["qid"])
        for per in all_personas
        for q in per["queries"]
        if q["type"] == "multi_hop" and q["gate"] and not q["gate"]["bridge_linked_to_answer"]
    ]
    if bad:
        print(
            f"PRECONDITION: {len(bad)} multi_hop queries where the graph never linked bridge->answer "
            f"(extraction gap, not a traversal failure): {bad}",
            file=sys.stderr,
        )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output", default=None)
    asyncio.run(main(ap.parse_args()))
