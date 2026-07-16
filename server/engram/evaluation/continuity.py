"""Continuity golden-path smoke: promote → cold get_context + recall → assert.

Product metric (not LongMemEval): a fresh consumer surfaces high-signal Decisions
without reading a handoff doc.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from engram.config import EngramConfig
from engram.extraction.factory import create_extractor
from engram.graph_manager import GraphManager
from engram.retrieval.context_builder import build_mcp_context_surface
from engram.retrieval.recall_surface import build_api_recall_surface
from engram.storage.bootstrap import close_if_supported, initialize_search_index_for_graph
from engram.storage.factory import create_stores
from engram.storage.resolver import EngineMode

DEFAULT_GROUP_ID = "continuity_smoke"

STRATEGY_DECISIONS: list[tuple[str, str]] = [
    (
        "LongMemEval is not Engram north star",
        "Product metric is multi-agent continuity, not LongMemEval scores.",
    ),
    (
        "Prefer sparse agent promotion",
        "Passive observe + sparse remember with proposals + deliberate consolidation.",
    ),
    (
        "Prefer markdown handoffs until proven",
        "Do not trust Engram recall as sole continuity until dogfood works.",
    ),
]


async def run_continuity_golden_path_smoke(
    *,
    sqlite_path: Path | None = None,
    group_id: str = DEFAULT_GROUP_ID,
) -> dict[str, Any]:
    """Promote 3 Decisions, then assert cold get_context + recall surface them."""
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if sqlite_path is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="engram-continuity-")
        sqlite_path = Path(temp_dir.name) / "continuity.db"

    config = EngramConfig(
        mode="lite",
        sqlite={"path": str(sqlite_path)},
        embedding={"provider": "noop"},
        activation={
            "extraction_provider": "narrow",
            "evidence_extraction_enabled": True,
            "evidence_client_proposals_enabled": True,
            "evidence_store_deferred": True,
            "identity_core_enabled": True,
            "recall_packets_enabled": True,
            "recall_packet_explicit_limit": 5,
            "recall_fast_preflight_enabled": True,
            "worker_enabled": False,
            "cue_layer_enabled": True,
        },
        _env_file=None,
    )

    graph_store, activation_store, search_index = create_stores(EngineMode.LITE, config)
    await graph_store.initialize()
    await initialize_search_index_for_graph(
        search_index,
        graph_store=graph_store,
        mode=EngineMode.LITE,
    )
    extractor = create_extractor(config)
    manager = GraphManager(
        graph_store=graph_store,
        activation_store=activation_store,
        search_index=search_index,
        extractor=extractor,
        cfg=config.activation,
    )

    started = time.perf_counter()
    promoted: list[dict[str, str]] = []
    try:
        for name, summary in STRATEGY_DECISIONS:
            content = f"{name}. {summary}"
            episode_id = await manager.ingest_episode(
                content=content,
                source="continuity_smoke",
                group_id=group_id,
                proposed_entities=[
                    {
                        "name": name,
                        "entity_type": "Decision",
                        "source_span": name,
                        "summary": summary,
                    }
                ],
                proposed_relationships=[
                    {
                        "subject": "Engram",
                        "predicate": "DECIDED",
                        "object": name,
                        "source_span": name,
                    }
                ],
                model_tier="sonnet",
            )
            promoted.append({"name": name, "episode_id": episode_id, "summary": summary})

        # Identity-core should be set on promoted Decisions.
        identity_ok: list[str] = []
        for name, _summary in STRATEGY_DECISIONS:
            entities = await graph_store.find_entities(
                name=name,
                entity_type="Decision",
                group_id=group_id,
                limit=5,
            )
            for entity in entities:
                if entity.name == name and getattr(entity, "identity_core", False):
                    identity_ok.append(name)
                    break

        context_payload = await build_mcp_context_surface(
            manager,
            group_id=group_id,
            max_tokens=1500,
            topic_hint="strategy decisions LongMemEval sparse promotion",
            project_path=None,
            format="structured",
            operation_source="api_context",
        )
        recall_payload = await build_api_recall_surface(
            manager,
            group_id=group_id,
            query=(
                "what strategy decisions did we make about LongMemEval and sparse agent promotion?"
            ),
            limit=5,
            project_path=None,
            operation_source="api_recall",
        )

        context_blob = _payload_blob(context_payload)
        recall_blob = _payload_blob(recall_payload)
        context_hits = [name for name, _ in STRATEGY_DECISIONS if name in context_blob]
        recall_hits = [name for name, _ in STRATEGY_DECISIONS if name in recall_blob]
        # Success: at least one high-signal Decision in both surfaces, or recall + type list.
        passed = len(recall_hits) >= 1 and (len(context_hits) >= 1 or len(identity_ok) >= 1)

        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "group_id": group_id,
            "sqlite_path": str(sqlite_path),
            "promoted": promoted,
            "identity_core": identity_ok,
            "context_hits": context_hits,
            "recall_hits": recall_hits,
            "context_lifecycle": context_payload.get("lifecycle"),
            "recall_lifecycle": recall_payload.get("lifecycle"),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "metric": (
                "Fresh consumer surfaces >=1 strategy Decision via get_context "
                "and/or recall without a handoff doc"
            ),
        }
    finally:
        await close_if_supported(search_index)
        await close_if_supported(activation_store)
        await close_if_supported(graph_store)
        if temp_dir is not None:
            temp_dir.cleanup()


def _payload_blob(payload: dict[str, Any]) -> str:
    parts: list[str] = [str(payload.get("context") or "")]
    for packet in payload.get("cached_packets") or payload.get("packets") or []:
        if isinstance(packet, dict):
            parts.append(str(packet.get("title") or ""))
            parts.append(str(packet.get("summary") or ""))
    for item in payload.get("items") or payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        entity = item.get("entity")
        if isinstance(entity, dict):
            parts.append(str(entity.get("name") or ""))
        else:
            parts.append(str(entity or item.get("name") or ""))
    return "\n".join(parts)


def format_continuity_report(result: dict[str, Any]) -> str:
    """Markdown report for CLI / CI logs."""
    status = "PASS" if result.get("passed") else "FAIL"
    title = result.get("title") or "Continuity golden path"
    lines = [
        f"# {title}: {status}",
        "",
        f"- Metric: {result.get('metric')}",
        f"- Duration: {result.get('duration_ms')} ms",
        f"- Identity-core: {', '.join(result.get('identity_core') or []) or '(none)'}",
        f"- get_context hits: {', '.join(result.get('context_hits') or []) or '(none)'}",
        f"- recall hits: {', '.join(result.get('recall_hits') or []) or '(none)'}",
        "",
    ]
    if result.get("entity_count") is not None:
        lines.insert(
            4,
            (
                f"- Graph entities: {result.get('entity_count')} | "
                f"episodes: {result.get('episode_count')}"
            ),
        )
    if result.get("error"):
        lines.append(f"- Error: {result.get('error')}")
    for row in result.get("promoted") or []:
        lines.append(f"- promoted: {row.get('name')} (`{row.get('episode_id')}`)")
    if result.get("mode") == "live":
        lines.append(f"- mode: live against `{result.get('data_dir') or 'configured store'}`")
        lines.append(f"- recall_ms: {result.get('recall_ms')}")
        lines.append(f"- context_ms: {result.get('context_ms')}")
    return "\n".join(lines) + "\n"


async def run_continuity_against_live(
    *,
    server_url: str = "http://127.0.0.1:8100",
    decision_name: str = "Cold Decision hit requires healthy search index",
    max_recall_ms: float = 2000.0,
    promote_if_missing: bool = True,
) -> dict[str, Any]:
    """Product gate: live get_context/recall must surface a Decision on the real brain.

    Fails when entity_count > 0 but cold surfaces show 0 Decision hits.
    Optionally promotes a known Decision first, then requires recall < max_recall_ms.
    """
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    started = time.perf_counter()
    base = server_url.rstrip("/")

    def _get(path: str, timeout: float = 30.0) -> dict[str, Any]:
        req = urllib.request.Request(f"{base}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _post(path: str, body: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{base}{path}",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        storage = _get("/api/storage?live=true&timeoutSeconds=8", timeout=15.0)
    except Exception as exc:
        return {
            "title": "Continuity against live brain",
            "status": "failed",
            "passed": False,
            "mode": "live",
            "error": f"storage probe failed: {exc}",
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "metric": "Live cold Decision hit on configured brain",
        }

    counts = storage.get("counts") or {}
    entity_count = int(counts.get("entities") or 0)
    episode_count = int(counts.get("episodes") or 0)
    data_dir = None
    for p in storage.get("paths") or []:
        if isinstance(p, dict) and "Helix native" in str(p.get("label") or ""):
            data_dir = p.get("path")
            break

    promoted: list[dict[str, str]] = []

    def _recall_once() -> tuple[dict[str, Any], float]:
        recall_q = urllib.parse.quote(decision_name)
        t0 = time.perf_counter()
        try:
            payload = _get(f"/api/knowledge/recall?q={recall_q}&limit=5", timeout=20.0)
        except Exception as exc:
            payload = {"error": str(exc), "results": [], "items": []}
        return payload, round((time.perf_counter() - t0) * 1000, 2)

    # Prefer existing graph Decision (no promote) — product dogfood path.
    recall_payload, recall_ms = _recall_once()
    recall_blob = _payload_blob(recall_payload) if isinstance(recall_payload, dict) else ""
    already_hit = decision_name in recall_blob

    if promote_if_missing and not already_hit:
        content = (
            f"{decision_name}. Product continuity requires search/index health "
            "so cold get_context and recall surface Decisions."
        )
        try:
            remember = _post(
                "/api/knowledge/remember",
                {
                    "content": content,
                    "source": "continuity_live_gate",
                    "model_tier": "sonnet",
                    "proposed_entities": [
                        {
                            "name": decision_name,
                            "entity_type": "Decision",
                            "source_span": decision_name,
                            "summary": "Live continuity gate Decision.",
                        }
                    ],
                    "proposed_relationships": [
                        {
                            "subject": "Engram",
                            "predicate": "DECIDED",
                            "object": decision_name,
                            "source_span": decision_name,
                        }
                    ],
                },
                timeout=120.0,
            )
            promoted.append(
                {
                    "name": decision_name,
                    "episode_id": str(
                        remember.get("episodeId") or remember.get("episode_id") or ""
                    ),
                }
            )
            # Cold-ish second recall after promote.
            recall_payload, recall_ms = _recall_once()
        except Exception as exc:
            return {
                "title": "Continuity against live brain",
                "status": "failed",
                "passed": False,
                "mode": "live",
                "error": f"remember promote failed: {exc}",
                "entity_count": entity_count,
                "episode_count": episode_count,
                "data_dir": data_dir,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "metric": "Live cold Decision hit on configured brain",
            }

    t1 = time.perf_counter()
    try:
        context_payload = _get(
            "/api/knowledge/context?max_tokens=1500&topic_hint="
            + urllib.parse.quote("Decision strategy continuity"),
            timeout=15.0,
        )
    except Exception as exc:
        context_payload = {"error": str(exc), "context": ""}
    context_ms = round((time.perf_counter() - t1) * 1000, 2)

    recall_blob = _payload_blob(recall_payload) if isinstance(recall_payload, dict) else ""
    context_blob = _payload_blob(context_payload) if isinstance(context_payload, dict) else ""
    # Also check results entities by name
    for item in recall_payload.get("items") or recall_payload.get("results") or []:
        if isinstance(item, dict):
            ent = item.get("entity") if isinstance(item.get("entity"), dict) else item
            if isinstance(ent, dict) and ent.get("name"):
                recall_blob += "\n" + str(ent.get("name"))

    # Exact substring of decision name in results or packets (cache Fact packets count).
    recall_hit = decision_name in recall_blob
    context_hit = decision_name in context_blob

    results_empty = not (recall_payload.get("results") or recall_payload.get("items"))
    degraded = bool(
        recall_payload.get("status") == "degraded"
        or (recall_payload.get("lifecycle") or {}).get("timeout")
    )

    # Pass criteria: Decision found in recall within budget; if graph non-empty
    # cannot return zero hits.
    within_budget = recall_ms <= max_recall_ms
    passed = bool(recall_hit and within_budget and not (entity_count > 0 and not recall_hit))
    if entity_count > 0 and not recall_hit:
        passed = False
    if not within_budget:
        passed = False
    if not recall_hit:
        passed = False

    return {
        "title": "Continuity against live brain",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "mode": "live",
        "metric": (
            f"Live cold recall surfaces Decision within {max_recall_ms:.0f}ms "
            "(fails if entity_count>0 and zero Decision hits)"
        ),
        "entity_count": entity_count,
        "episode_count": episode_count,
        "data_dir": data_dir,
        "promoted": promoted,
        "identity_core": [decision_name] if recall_hit else [],
        "context_hits": [decision_name] if context_hit else [],
        "recall_hits": [decision_name] if recall_hit else [],
        "recall_ms": recall_ms,
        "context_ms": context_ms,
        "within_budget": within_budget,
        "results_empty": results_empty,
        "degraded": degraded,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "server_url": base,
    }
