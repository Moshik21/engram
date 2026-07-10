"""Continuity golden-path smoke: promote → cold get_context + recall → assert.

Product metric (not LongMemEval): a fresh consumer surfaces high-signal Decisions
without reading a handoff doc.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from engram.config import ActivationConfig, EngramConfig
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
            query="what strategy decisions did we make about LongMemEval and sparse agent promotion?",
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
    lines = [
        f"# Continuity golden path: {status}",
        "",
        f"- Metric: {result.get('metric')}",
        f"- Duration: {result.get('duration_ms')} ms",
        f"- Identity-core: {', '.join(result.get('identity_core') or []) or '(none)'}",
        f"- get_context hits: {', '.join(result.get('context_hits') or []) or '(none)'}",
        f"- recall hits: {', '.join(result.get('recall_hits') or []) or '(none)'}",
        "",
    ]
    for row in result.get("promoted") or []:
        lines.append(f"- promoted: {row.get('name')} (`{row.get('episode_id')}`)")
    return "\n".join(lines) + "\n"
