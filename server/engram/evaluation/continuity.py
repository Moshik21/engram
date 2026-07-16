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


def select_aged_organic_decision(
    entities: list[dict[str, Any]],
    *,
    min_age_days: float = 7.0,
    exclude_names: set[str] | None = None,
    now: Any = None,
) -> dict[str, Any] | None:
    """Pick a real, aged Decision the gate must recall (no self-promotion).

    The v1 gate could write a synthetic Decision and immediately recall it —
    verifying write→read round-trip, not durable continuity. An organic
    target must be ≥ min_age_days old, non-noise, and not gate-created.
    """
    from datetime import datetime, timedelta, timezone

    from engram.extraction.promotion import is_decision_statement_noise

    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=min_age_days)
    excluded = {n.casefold() for n in (exclude_names or set())}
    candidates: list[tuple[Any, dict[str, Any]]] = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        name = str(ent.get("name") or "").strip()
        if not name or name.casefold() in excluded:
            continue
        if str(ent.get("entity_type") or "") != "Decision":
            continue
        if is_decision_statement_noise(name):
            continue
        raw_created = ent.get("created_at") or ent.get("createdAt")
        if not raw_created:
            continue
        try:
            created = datetime.fromisoformat(str(raw_created).replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if created <= cutoff:
            candidates.append((created, ent))
    if not candidates:
        return None
    # Newest qualifying Decision: aged enough to prove durability, recent
    # enough to still matter.
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def count_decision_scrap_top5(recall_payload: dict[str, Any]) -> int:
    """WEEKLY_NORTH_STAR anti-metric: decision_statement scrap in top-5."""
    from engram.extraction.promotion import is_decision_statement_noise

    scrap = 0
    items = recall_payload.get("items") or recall_payload.get("results") or []
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        ent = item.get("entity") if isinstance(item.get("entity"), dict) else item
        if not isinstance(ent, dict):
            continue
        name = str(ent.get("name") or "")
        if name and is_decision_statement_noise(name):
            scrap += 1
    return scrap


async def run_continuity_against_live(
    *,
    server_url: str = "http://127.0.0.1:8100",
    decision_name: str = "Cold Decision hit requires healthy search index",
    max_recall_ms: float = 2000.0,
    promote_if_missing: bool = True,
    require_organic: bool = False,
    min_organic_age_days: float = 7.0,
) -> dict[str, Any]:
    """Product gate: live get_context/recall must surface a Decision on the real brain.

    Fails when entity_count > 0 but cold surfaces show 0 Decision hits.
    Optionally promotes a known Decision first, then requires recall < max_recall_ms.

    require_organic=True is metric v2: the target must be a real Decision at
    least min_organic_age_days old (promote_if_missing is forced off), and
    Decision scrap in the top-5 results fails the gate. A gate that can
    self-satisfy by writing its own Decision measures index round-trip, not
    continuity.
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
    organic_target: dict[str, Any] | None = None

    if require_organic:
        promote_if_missing = False
        synthetic_names = {decision_name}
        try:
            search = _get(
                "/api/entities/search?type=Decision&limit=100",
                timeout=20.0,
            )
            entities = search.get("entities") or search.get("results") or []
        except Exception:
            entities = []
        organic_target = select_aged_organic_decision(
            entities if isinstance(entities, list) else [],
            min_age_days=min_organic_age_days,
            exclude_names=synthetic_names,
        )
        if organic_target is None:
            if entity_count == 0:
                return {
                    "title": "Continuity against live brain (aged organic)",
                    "status": "skipped",
                    "passed": True,
                    "mode": "live",
                    "note": "empty brain: no organic Decisions to age yet",
                    "entity_count": entity_count,
                    "episode_count": episode_count,
                    "data_dir": data_dir,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "metric": (
                        f"Aged (≥{min_organic_age_days:.0f}d) organic Decision "
                        "recallable within budget, 0 scrap in top-5"
                    ),
                }
            return {
                "title": "Continuity against live brain (aged organic)",
                "status": "failed",
                "passed": False,
                "mode": "live",
                "error": (
                    f"no organic Decision ≥{min_organic_age_days:.0f} days old is "
                    "listable — either none survived consolidation or none were "
                    "ever promoted (the v1 synthetic gate masked this)"
                ),
                "entity_count": entity_count,
                "episode_count": episode_count,
                "data_dir": data_dir,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "metric": (
                    f"Aged (≥{min_organic_age_days:.0f}d) organic Decision "
                    "recallable within budget, 0 scrap in top-5"
                ),
            }
        decision_name = str(organic_target.get("name"))

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

    # Precision anti-metric (WEEKLY_NORTH_STAR): decision_statement scrap in
    # the top-5. Reported always; hard-fails the aged-organic gate.
    decision_scrap_top5 = count_decision_scrap_top5(recall_payload)
    if require_organic and decision_scrap_top5 > 0:
        passed = False

    # Availability (blind spot of v1: the gate only ran "when the shell was
    # up", so multi-hour brain-window outages never failed anything).
    availability: dict[str, Any] | None = None
    try:
        from engram.brain_runtime import read_brain_status
        from engram.ops_metrics import brain_status_anomalies, compute_shell_availability

        availability = compute_shell_availability().to_dict()
        availability["brain_anomalies"] = brain_status_anomalies(read_brain_status())
    except Exception:
        availability = None

    title = "Continuity against live brain"
    metric = (
        f"Live cold recall surfaces Decision within {max_recall_ms:.0f}ms "
        "(fails if entity_count>0 and zero Decision hits)"
    )
    if require_organic:
        title = "Continuity against live brain (aged organic)"
        metric = (
            f"Aged (≥{min_organic_age_days:.0f}d) organic Decision recallable "
            f"within {max_recall_ms:.0f}ms, 0 Decision scrap in top-5"
        )

    return {
        "title": title,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "mode": "live",
        "metric": metric,
        "organic_target": (organic_target or {}).get("name") if require_organic else None,
        "decision_scrap_top5": decision_scrap_top5,
        "availability": availability,
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
